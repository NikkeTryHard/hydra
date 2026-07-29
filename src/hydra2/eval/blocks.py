"""SPEC 18.1/18.3 wall-block aggregation and the invalid-block policy.

Games within one wall share the dealt wall deck and every divergent game
remains a member of that one wall block — they are NOT identical
counterfactual paths and NOT independent units (SPEC 18.1). Aggregation
therefore collapses each block to a single contrast value before any
uncertainty method sees it.

Invalid-block policy: a block is EXCLUDED and REPORTED (never silently
imputed or repaired) when telemetry gaps exceed the predeclared tolerance or
a disallowed fallback/timeout/illegal-action flag is set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hydra2.contracts.common import ContractError
from hydra2.eval.telemetry import (
    ResourceTelemetry,
    TelemetryTolerance,
    telemetry_invalid_reason,
)

__all__ = [
    "EXCLUSION_REASONS",
    "BlockAggregateResult",
    "BlockTolerance",
    "ExcludedBlock",
    "WallBlock",
    "aggregate_blocks",
    "aggregate_wall_block",
]


@dataclass(frozen=True, slots=True)
class WallBlock:
    """One complete wall block: atomic unit of confirmation."""

    wall_id: str
    game_ids: tuple[str, ...]
    contrasts: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.wall_id, str) or self.wall_id == "":
            raise ContractError("wall_id must be a nonempty str")
        if len(self.game_ids) != len(self.contrasts):
            raise ContractError("game_ids and contrasts must have equal length")
        if any(game_id == "" for game_id in self.game_ids):
            raise ContractError("game_ids must be nonempty strings")
        for value in self.contrasts:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ContractError(f"contrast must be finite, got {value!r}")


def aggregate_wall_block(block: WallBlock) -> float:
    """Collapse the block to ONE number; games inside are not independent."""
    if len(block.contrasts) == 0:
        raise ContractError(f"wall block {block.wall_id!r} has no games")
    return math.fsum(block.contrasts) / len(block.contrasts)


@dataclass(frozen=True, slots=True)
class BlockTolerance(TelemetryTolerance):
    """Predeclared invalidity tolerances for block admission.

    The boolean flags default to strict (any occurrence invalidates); they
    MUST be frozen before results are seen.
    """

    allow_fallback_used: bool = False
    allow_timeout: bool = False
    allow_illegal_action: bool = False


EXCLUSION_REASONS: tuple[str, ...] = (
    "missing_telemetry",
    "fallback_used",
    "timeout",
    "illegal_action",
    "row_invalid",
    "empty_block",
)


@dataclass(frozen=True, slots=True)
class ExcludedBlock:
    """A reported exclusion: block identity, reason, human-readable detail."""

    wall_id: str
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class BlockAggregateResult:
    """Validated block values plus the full exclusion report."""

    valid: tuple[tuple[str, float], ...]
    excluded: tuple[ExcludedBlock, ...]


def aggregate_blocks(
    blocks: tuple[WallBlock, ...],
    *,
    telemetry_by_game: dict[str, ResourceTelemetry] | None = None,
    tolerance: BlockTolerance | None = None,
) -> BlockAggregateResult:
    """Aggregate whole blocks; exclude-and-report invalid ones."""
    tolerance = tolerance if tolerance is not None else BlockTolerance()
    telemetry_by_game = telemetry_by_game if telemetry_by_game is not None else {}
    valid: list[tuple[str, float]] = []
    excluded: list[ExcludedBlock] = []

    for block in sorted(blocks, key=lambda item: item.wall_id):
        exclusion = _first_disqualification(block, telemetry_by_game, tolerance)
        if exclusion is not None:
            excluded.append(exclusion)
            continue
        valid.append((block.wall_id, aggregate_wall_block(block)))
    return BlockAggregateResult(valid=tuple(valid), excluded=tuple(excluded))


def _first_disqualification(
    block: WallBlock,
    telemetry_by_game: dict[str, ResourceTelemetry],
    tolerance: BlockTolerance,
) -> ExcludedBlock | None:
    if len(block.contrasts) == 0:
        return ExcludedBlock(
            wall_id=block.wall_id, reason="empty_block", detail="block carries no games"
        )
    missing_rows = [game_id for game_id in block.game_ids if game_id not in telemetry_by_game]
    if len(missing_rows) != 0:
        return ExcludedBlock(
            wall_id=block.wall_id,
            reason="missing_telemetry",
            detail=f"no telemetry rows for games {missing_rows}",
        )
    for game_id in block.game_ids:
        row = telemetry_by_game[game_id]
        reason = telemetry_invalid_reason(row, tolerance)
        if reason is None:
            continue
        mapped = "row_invalid" if reason.startswith("row marked") else "missing_telemetry"
        return ExcludedBlock(
            wall_id=block.wall_id,
            reason=mapped,
            detail=f"game {game_id}: {reason}",
        )
    for game_id in block.game_ids:
        row = telemetry_by_game[game_id]
        if row.fallback_used and not tolerance.allow_fallback_used:
            return ExcludedBlock(
                wall_id=block.wall_id,
                reason="fallback_used",
                detail=f"game {game_id} used fallback",
            )
        if row.timeout and not tolerance.allow_timeout:
            return ExcludedBlock(
                wall_id=block.wall_id,
                reason="timeout",
                detail=f"game {game_id} timed out",
            )
        if row.illegal_action and not tolerance.allow_illegal_action:
            return ExcludedBlock(
                wall_id=block.wall_id,
                reason="illegal_action",
                detail=f"game {game_id} produced an illegal action",
            )
    return None
