"""Candidate scorers over one shared rich-state encoding.

Both scorers read the same encoded state, global actor and match context, and
raw semantic action fields. The edit scorer adds the private hand-change
channel: the difference of one shared nonlinear map applied to the before and
after own count/red state of every touched tile, with duplicate face slots
aggregated before pooling.
"""

import torch
from torch import Tensor, nn

from .model_action import ActionView, gather_tiles
from .model_descriptor import DescriptorEncoder
from .model_schema import (
    ACTION_REMOVAL_COUNT,
    MAX_OWN_COUNT,
    TILE_FACES,
)

# Own count and own red for one side of a before/after state pair.
_EDIT_STATE_COUNT = 2


class EditScorer(nn.Module):
    """Fuses the private hand change with the shared public action descriptor."""

    slot_precedes: Tensor

    def __init__(self, width: int) -> None:
        super().__init__()
        self.descriptor = DescriptorEncoder(width)
        self.edit = nn.Sequential(
            nn.Linear(width + _EDIT_STATE_COUNT, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        fused_width = 2 * width
        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_width),
            nn.Linear(fused_width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        slots = ACTION_REMOVAL_COUNT
        self.register_buffer(
            "slot_precedes",
            torch.tril(torch.ones(slots, slots, dtype=torch.bool), diagonal=-1),
            persistent=False,
        )

    def forward(
        self,
        encoded: Tensor,
        candidates: Tensor,
        own_counts: Tensor,
        own_reds: Tensor,
    ) -> Tensor:
        public, view = self.descriptor(encoded, candidates)
        private = self.private_edit(encoded, view, own_counts, own_reds)
        return self.fuse(torch.cat((private, public), dim=-1))

    def private_edit(
        self,
        encoded: Tensor,
        view: ActionView,
        own_counts: Tensor,
        own_reds: Tensor,
    ) -> Tensor:
        """Pool one edit record per affected face, never per duplicate slot.

        The removal block is authoritative: ordinary discards, riichi and
        claims carry their actor-owned removals there. Duplicate face slots are
        aggregated with pairwise slot equality so each affected face yields one
        before/after state pair.
        """
        slot_faces = torch.where(
            view.removal_present, view.removal, torch.full_like(view.removal, -1)
        )
        slot_present = slot_faces >= 0
        safe_faces = slot_faces.clamp(0, TILE_FACES - 1)
        slot_red = torch.where(
            view.removal_present, view.removal_red, torch.zeros_like(view.removal_red)
        )

        count = safe_faces.shape[1]
        same_face = safe_faces.unsqueeze(-1) == safe_faces.unsqueeze(-2)
        same_present = same_face & slot_present.unsqueeze(-2)
        duplicate = (same_present & self.slot_precedes).any(dim=-1)
        unique = slot_present & ~duplicate

        before_count = torch.gather(own_counts.unsqueeze(1).expand(-1, count, -1), 2, safe_faces)
        before_red = torch.gather(own_reds.unsqueeze(1).expand(-1, count, -1), 2, safe_faces)
        removed = same_present.sum(dim=-1).to(before_count.dtype)
        removed_red = (
            (same_present.to(slot_red.dtype) * slot_red.unsqueeze(-2))
            .sum(dim=-1)
            .to(before_red.dtype)
        )
        after_count = (before_count - removed).clamp(0, MAX_OWN_COUNT)
        after_red = (before_red - removed_red).clamp(0, 1)

        tiles = gather_tiles(encoded, safe_faces, slot_present)
        before_state = torch.stack((before_count / MAX_OWN_COUNT, before_red), dim=-1).to(
            tiles.dtype
        )
        after_state = torch.stack((after_count / MAX_OWN_COUNT, after_red), dim=-1).to(tiles.dtype)
        before_input = torch.cat((tiles, before_state), dim=-1)
        after_input = torch.cat((tiles, after_state), dim=-1)
        mapped_before, mapped_after = self.edit(
            torch.stack((before_input, after_input), dim=-2)
        ).unbind(-2)
        difference = mapped_after - mapped_before
        weights = unique.unsqueeze(-1).to(difference.dtype)
        return (difference * weights).sum(dim=-2)


class PlainScorer(nn.Module):
    """Rich-state control without an explicit before/after hand edit."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.descriptor = DescriptorEncoder(width)

    def forward(self, encoded: Tensor, candidates: Tensor) -> Tensor:
        representation, _ = self.descriptor(encoded, candidates)
        return representation
