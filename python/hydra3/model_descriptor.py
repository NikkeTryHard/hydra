"""Raw semantic action descriptor plus global actor and match context."""

import torch
from torch import Tensor, nn

from .model_action import ActionView, action_view, gather_seat, gather_tiles
from .model_schema import (
    ACTION_KIND_COUNT,
    ACTION_REMOVAL_COUNT,
    MATCH_TOKEN,
    SELF_SEAT_TOKEN,
)

# Kind, source seat, primary tile, consumed tiles, match token and actor token.
_VECTOR_COUNT = 6
# Primary red, drawn marker, riichi and claim bits; source and primary presence;
# removal presence and red bits for every removal slot.
_SCALAR_COUNT = 4 + 2 + 2 * ACTION_REMOVAL_COUNT


class DescriptorEncoder(nn.Module):
    """Shared candidate features: kind, touched tiles/seat and public markers."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.kind = nn.Embedding(ACTION_KIND_COUNT, width)
        self.null_seat = nn.Parameter(torch.zeros(width))
        self.null_tile = nn.Parameter(torch.zeros(width))
        self.projection = nn.Sequential(
            nn.Linear(_VECTOR_COUNT * width + _SCALAR_COUNT, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, encoded: Tensor, candidates: Tensor) -> tuple[Tensor, ActionView]:
        view = action_view(candidates)
        dtype = encoded.dtype
        batch, count = candidates.shape[:2]

        primary = gather_tiles(
            encoded,
            view.primary.unsqueeze(-1),
            view.primary_present.unsqueeze(-1),
            self.null_tile,
        ).squeeze(2)
        removal = gather_tiles(encoded, view.removal, view.removal_present)
        removal_total = removal.sum(dim=2)
        source_seat = gather_seat(encoded, view.source, view.source_present, self.null_seat)

        scalars = torch.cat(
            (
                view.primary_red.unsqueeze(-1),
                view.drawn_discard.unsqueeze(-1),
                view.declare_riichi.unsqueeze(-1),
                view.claim_intent.unsqueeze(-1),
                view.source_present.to(torch.float32).unsqueeze(-1),
                view.primary_present.to(torch.float32).unsqueeze(-1),
                view.removal_present.to(torch.float32),
                view.removal_red,
            ),
            dim=-1,
        )

        match = encoded[:, MATCH_TOKEN].unsqueeze(1).expand(batch, count, -1)
        actor = encoded[:, SELF_SEAT_TOKEN].unsqueeze(1).expand(batch, count, -1)
        parts = torch.cat(
            (
                self.kind(view.kind).to(dtype),
                source_seat,
                primary,
                removal_total,
                match,
                actor,
                scalars.to(dtype),
            ),
            dim=-1,
        )
        return self.projection(parts), view
