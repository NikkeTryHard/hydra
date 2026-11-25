"""Sanitization and gathers for legal-candidate action descriptors."""

from typing import NamedTuple

import torch
from torch import Tensor

from .model_schema import (
    ACTION_CLAIM_INTENT,
    ACTION_DECLARE_RIICHI,
    ACTION_DRAWN_DISCARD,
    ACTION_KIND,
    ACTION_KIND_COUNT,
    ACTION_PRIMARY_FACE,
    ACTION_PRIMARY_RED,
    ACTION_REMOVAL_COUNT,
    ACTION_REMOVAL_FIRST,
    ACTION_REMOVAL_RED_FIRST,
    ACTION_SOURCE,
    SEAT_TOKENS,
    SOURCE_ABSENT,
    TILE_FACES,
    TILE_TOKENS,
)


class ActionView(NamedTuple):
    """Sanitized fields of a legal-candidate descriptor block."""

    kind: Tensor
    source: Tensor
    source_present: Tensor
    primary: Tensor
    primary_present: Tensor
    primary_red: Tensor
    drawn_discard: Tensor
    declare_riichi: Tensor
    removal: Tensor
    removal_present: Tensor
    removal_red: Tensor
    claim_intent: Tensor


def _unit_bits(values: Tensor) -> Tensor:
    return values.clamp(0, 1).to(torch.float32)


def action_view(candidates: Tensor) -> ActionView:
    """Sanitize descriptor fields before any learned indexing or gather."""
    candidates = candidates.long()
    kind = candidates[..., ACTION_KIND].clamp(0, ACTION_KIND_COUNT - 1)

    source_raw = candidates[..., ACTION_SOURCE]
    source_present = (source_raw >= 0) & (source_raw < SEAT_TOKENS)
    source = torch.where(source_present, source_raw, torch.full_like(source_raw, SOURCE_ABSENT))

    primary_raw = candidates[..., ACTION_PRIMARY_FACE]
    primary_present = (primary_raw >= 0) & (primary_raw < TILE_FACES)
    primary = torch.where(primary_present, primary_raw, torch.zeros_like(primary_raw))
    primary_red = torch.where(
        primary_present,
        _unit_bits(candidates[..., ACTION_PRIMARY_RED]),
        torch.zeros_like(primary_present, dtype=torch.float32),
    )

    removal_start = ACTION_REMOVAL_FIRST
    removal_raw = candidates[..., removal_start : removal_start + ACTION_REMOVAL_COUNT]
    removal_present = (removal_raw >= 0) & (removal_raw < TILE_FACES)
    removal = torch.where(removal_present, removal_raw, torch.zeros_like(removal_raw))
    red_start = ACTION_REMOVAL_RED_FIRST
    removal_red = torch.where(
        removal_present,
        _unit_bits(candidates[..., red_start : red_start + ACTION_REMOVAL_COUNT]),
        torch.zeros_like(removal_present, dtype=torch.float32),
    )
    return ActionView(
        kind=kind,
        source=source,
        source_present=source_present,
        primary=primary,
        primary_present=primary_present,
        primary_red=primary_red,
        drawn_discard=_unit_bits(candidates[..., ACTION_DRAWN_DISCARD]),
        declare_riichi=_unit_bits(candidates[..., ACTION_DECLARE_RIICHI]),
        removal=removal,
        removal_present=removal_present,
        removal_red=removal_red,
        claim_intent=_unit_bits(candidates[..., ACTION_CLAIM_INTENT]),
    )


def gather_tiles(
    encoded: Tensor, faces: Tensor, present: Tensor, null: Tensor | None = None
) -> Tensor:
    """Gather contextual tile tokens, with optional learned empty-tile fill."""
    # Expand the indices, not the tile states. A candidate-expanded state is a view
    # in forward but creates a large scatter-and-sum buffer in backward. Time full PPO
    # updates; inference-only timing cannot detect this regression.
    indices = faces.clamp(0, TILE_FACES - 1).reshape(encoded.shape[0], -1)
    width = encoded.shape[-1]
    gathered = encoded.gather(1, indices.unsqueeze(-1).expand(-1, -1, width))
    gathered = gathered.reshape(*faces.shape, width)
    mask = present.unsqueeze(-1)
    if null is None:
        return gathered * mask.to(gathered.dtype)
    return torch.where(mask, gathered, null.to(dtype=gathered.dtype, device=gathered.device))


def gather_seat(encoded: Tensor, source: Tensor, present: Tensor, null: Tensor) -> Tensor:
    """Gather one contextual relative-seat token, or a learned empty-seat fill."""
    seat = encoded[:, TILE_TOKENS : TILE_TOKENS + SEAT_TOKENS]
    width = seat.shape[-1]
    safe = source.clamp(0, SEAT_TOKENS - 1)
    index = safe.unsqueeze(-1).expand(-1, -1, width)
    gathered = torch.gather(seat, 1, index)
    return torch.where(
        present.unsqueeze(-1),
        gathered,
        null.to(dtype=gathered.dtype, device=gathered.device),
    )
