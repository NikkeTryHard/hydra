"""WP-06 duplicate-block qualification — exact/near duplicates, disjoint walls, block eval.

Implements BUILD WP-06 checklist (M7b):

* exact duplicates: two wall ids sharing the identical wall digest (same bytes)
* near duplicates: two walls sharing the logical fingerprint (same sorted wall
  content — catches walls that differ only by dealing order but contain the
  same tile multiset)
* disjoint walls: no wall id appears in more than one partition/schedule/split
* block splitting: whole-wall blocks as the independent unit; game-level
  observations are never treated as independent
* reports: block manifest, seat-balance audit, telemetry completeness

Fresh-process, seed-protocol commitment, wall-block bootstrap/sign-flip, and
held-out hiding follow SPEC 18.1-18.3; this module is the thin qualification
layer that wires :mod:`hydra2.eval.schedule`, :mod:`hydra2.eval.blocks`,
:mod:`hydra2.eval.telemetry`, and :mod:`hydra2.eval.statistics`.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2._native import eval as _bridge_eval  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import ContractError, DigestText
from hydra2.eval._duplicate_walls import WALL_TILES as WALL_TILES
from hydra2.eval._duplicate_walls import find_exact_duplicates as find_exact_duplicates
from hydra2.eval._duplicate_walls import find_near_duplicates as find_near_duplicates
from hydra2.eval._duplicate_walls import wall_fingerprint as wall_fingerprint
from hydra2.eval._duplicate_walls import wall_hash_from_tiles as wall_hash_from_tiles
from hydra2.eval.blocks import (
    BlockAggregateResult,
    BlockTolerance,
    WallBlock,
    aggregate_blocks,
)
from hydra2.eval.schedule import MatchSchedule, seat_pair_placements_exact
from hydra2.eval.telemetry import (
    ResourceTelemetry,
    TelemetryTolerance,
    telemetry_invalid_reason,
)

__all__ = [
    "BlockManifest",
    "BlockSplit",
    "DuplicateReport",
    "balance_audit",
    "block_manifest_digest",
    "build_wall_blocks",
    "confirmation_sidecar",
    "find_exact_duplicates",
    "find_near_duplicates",
    "make_block_manifest",
    "report_telemetry_completeness",
    "split_blocks_held_out",
    "validate_blocks_disjoint",
    "validate_walls_disjoint",
    "wall_fingerprint",
    "wall_hash_from_tiles",
]


@dataclass(frozen=True, slots=True)
class DuplicateReport:
    """Result of exact/near duplicate checks over a wall ledger."""

    exact_duplicates: tuple[tuple[str, str], ...]
    near_duplicates: tuple[tuple[str, str], ...]


def validate_walls_disjoint(*wall_collections: Iterable[str]) -> None:
    """Raise :class:`ContractError` when a wall id appears in more than one collection.

    Each ``wall_collection`` is an iterable of wall ids. Overlap across any
    two collections is forbidden (e.g., train vs evaluation ledger reuse).

    Thin bridge delegate: ``hydra2._native.contracts.eval_validate_walls_disjoint``
    stages the collections and checks uniqueness detached; this function keeps
    the variadic shape, the ``ContractError`` shaping, and the ``__all__`` name.
    Non-iterables propagate ``TypeError`` (oracle ``for``-loop behaviour).
    """
    try:
        _bridge_contracts.eval_validate_walls_disjoint(list(wall_collections))
    except ValueError as exc:
        raise ContractError(str(exc)) from exc


def validate_blocks_disjoint(blocks: tuple[WallBlock, ...]) -> None:
    """Raise when wall ids or game ids repeat across blocks.

    Whole-wall blocks MUST be disjoint; a game may belong to exactly one
    wall block.

    Thin bridge delegate: ``hydra2._native.eval.record_validate_blocks_disjoint``
    decides detached over translator-staged ``(wall_id, game_ids)`` pairs
    (owner ``hydra_search::eval::partition::validate_blocks_disjoint``);
    this function keeps the live-``WallBlock`` shape, the ``ContractError``
    shaping, and the ``__all__`` name. Extraction ``TypeError`` on exotic
    non-``str`` ids falls back to the oracle body below.
    """
    try:
        leaf = _bridge_eval.record_validate_blocks_disjoint  # type: ignore[attr-defined]  # reason: record_leaves leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        pass
    else:
        try:
            leaf([(block.wall_id, list(block.game_ids)) for block in blocks])
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        except TypeError:
            pass
        else:
            return
    wall_ids: set[str] = set()
    game_ids: set[str] = set()
    for block in blocks:
        if block.wall_id in wall_ids:
            raise ContractError(f"duplicate wall_id {block.wall_id!r} across blocks")
        wall_ids.add(block.wall_id)
        for game_id in block.game_ids:
            if game_id in game_ids:
                raise ContractError(f"game {game_id!r} appears in multiple blocks")
            game_ids.add(game_id)


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------


def build_wall_blocks(
    *,
    schedule: MatchSchedule,
    contrasts_by_game: Mapping[str, float],
    game_ids_by_wall: Mapping[str, Sequence[str]] | None = None,
    telemetry_by_game: Mapping[str, ResourceTelemetry] | None = None,
    tolerance: BlockTolerance | TelemetryTolerance | None = None,
) -> tuple[WallBlock, ...]:
    """Build validated wall blocks from a committed schedule and results.

    ``contrasts_by_game`` maps game id -> expected-final-placement contrast.
    When ``game_ids_by_wall`` is ``None``, deterministic game ids
    ``{wall_id}:g{0..9}`` aligned to schedule order are synthesised (useful
    for synthetic qualification). Each wall MUST contribute ``TOTAL_GAMES_PER_WALL``
    games; missing entries raise :class:`ContractError`.

    ``telemetry_by_game`` and ``tolerance`` are not consumed here but are
    validated for presence when supplied to catch programming errors early.
    """
    from hydra2.eval.schedule import TOTAL_GAMES_PER_WALL

    if not isinstance(schedule, MatchSchedule):
        raise ContractError("schedule must be a MatchSchedule")
    if not isinstance(contrasts_by_game, Mapping):
        raise ContractError("contrasts_by_game must be mapping game_id -> float")
    if telemetry_by_game is not None and not isinstance(telemetry_by_game, Mapping):
        raise ContractError("telemetry_by_game must be mapping or None")
    if tolerance is not None and not isinstance(tolerance, (BlockTolerance, TelemetryTolerance)):
        raise ContractError("tolerance must be BlockTolerance/TelemetryTolerance or None")
    seat_pair_placements_exact(schedule)

    # Validate contrasts are finite.
    for game_id, value in contrasts_by_game.items():
        if not isinstance(game_id, str) or game_id == "":
            raise ContractError(f"game_id must be nonempty str, got {game_id!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"contrast for {game_id!r} must be finite number")
        if not math.isfinite(float(value)):
            raise ContractError(f"contrast for {game_id!r} must be finite, got {value!r}")

    blocks: list[WallBlock] = []
    for _wall_index, wall_id in enumerate(schedule.wall_ids):
        if game_ids_by_wall is not None:
            raw_ids = game_ids_by_wall.get(wall_id)
            if raw_ids is None:
                raise ContractError(f"missing game_ids for wall {wall_id!r}")
            game_ids = tuple(raw_ids)
        else:
            game_ids = tuple(f"{wall_id}:g{slot}" for slot in range(TOTAL_GAMES_PER_WALL))
        if len(game_ids) != TOTAL_GAMES_PER_WALL:
            raise ContractError(
                f"wall {wall_id!r} must carry {TOTAL_GAMES_PER_WALL} games, got {len(game_ids)}"
            )
        try:
            contrasts = tuple(contrasts_by_game[game_id] for game_id in game_ids)
        except KeyError as exc:
            raise ContractError(f"missing contrast for game {exc.args[0]!r}") from exc
        # WallBlock validates finiteness and equal lengths internally.
        blocks.append(WallBlock(wall_id=wall_id, game_ids=game_ids, contrasts=contrasts))

    # Enforce disjointness before returning.
    # Chain: build enforces -> manifest re-checks order/disjointness ->
    # split re-checks partitions (validate_blocks/walls_disjoint).
    validate_blocks_disjoint(tuple(blocks))
    # No silent imputation of telemetry — caller aggregates separately.
    return tuple(blocks)


@dataclass(frozen=True, slots=True)
class BlockManifest:
    """Canonical block manifest — schedule binding + block identity."""

    schedule_hash: DigestText
    wall_ids: tuple[str, ...]
    blocks: tuple[WallBlock, ...]
    digest: DigestText


def make_block_manifest(
    *,
    schedule: MatchSchedule,
    blocks: tuple[WallBlock, ...],
) -> BlockManifest:
    """Assemble a canonical manifest binding a schedule to its wall blocks."""
    if not isinstance(schedule, MatchSchedule):
        raise ContractError("schedule must be a MatchSchedule")
    if not isinstance(blocks, tuple) or len(blocks) == 0:
        raise ContractError("blocks must be nonempty tuple of WallBlock")
    from hydra2.eval.schedule import schedule_commitment_hash

    schedule_hash = schedule_commitment_hash(schedule)
    manifest_wall_ids = tuple(block.wall_id for block in blocks)
    if tuple(schedule.wall_ids) != manifest_wall_ids:
        # Allow subset: blocks may be a held-out slice, but every block wall must
        # belong to the schedule and wall order must respect schedule order.
        schedule_order = {wall_id: index for index, wall_id in enumerate(schedule.wall_ids)}
        for block in blocks:
            if block.wall_id not in schedule_order:
                raise ContractError(f"block wall {block.wall_id!r} not in schedule")

        def _wall_order(b: WallBlock) -> int:
            return schedule_order[b.wall_id]

        ordered = tuple(sorted(blocks, key=_wall_order))
        if ordered != blocks:
            raise ContractError("blocks must be in schedule wall order")
        manifest_wall_ids = tuple(block.wall_id for block in ordered)
        blocks = ordered
    validate_blocks_disjoint(blocks)
    payload = {
        "schedule_hash": schedule_hash,
        "wall_ids": list(manifest_wall_ids),
        "blocks": [
            {
                "wall_id": block.wall_id,
                "game_ids": list(block.game_ids),
                "contrasts": list(block.contrasts),
            }
            for block in blocks
        ],
    }
    digest = of_canonical(payload)
    return BlockManifest(
        schedule_hash=schedule_hash,
        wall_ids=manifest_wall_ids,
        blocks=blocks,
        digest=digest,
    )


def block_manifest_digest(manifest: BlockManifest) -> DigestText:
    _bridge_contracts.make_digest_text(manifest.digest)
    return manifest.digest


# ---------------------------------------------------------------------------
# Held-out splitting (hiding final partition from training/checkpoint selection)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BlockSplit:
    """Deterministic whole-wall train / held-out split — held-out hidden."""

    train_blocks: tuple[WallBlock, ...]
    held_out_blocks: tuple[WallBlock, ...]
    seed: int
    held_out_ratio: float
    digest: DigestText


def _validate_block_split_input(
    blocks: tuple[WallBlock, ...], held_out_ratio: float, seed: int
) -> None:
    if not isinstance(blocks, tuple) or len(blocks) == 0:
        raise ContractError("blocks must be nonempty tuple of WallBlock")
    if any(not isinstance(block, WallBlock) for block in blocks):
        raise ContractError("blocks must contain WallBlock instances")
    if len({block.wall_id for block in blocks}) != len(blocks):
        raise ContractError("blocks must carry unique wall_ids")
    if not 0 < held_out_ratio < 1:
        raise ContractError(f"held_out_ratio must be in (0,1), got {held_out_ratio!r}")
    if not isinstance(seed, int):
        raise ContractError(f"seed must be int, got {seed!r}")


def split_blocks_held_out(
    blocks: tuple[WallBlock, ...],
    *,
    held_out_ratio: float = 0.2,
    seed: int = 0,
) -> BlockSplit:
    """Split wall blocks into train and held-out sets; held-out never revealed to training.

    Deterministic in ``seed`` and ``held_out_ratio``; identical inputs produce
    identical partitions and digest. Whole blocks are the atomic unit — games
    inside a wall are never split across partitions.
    """
    _validate_block_split_input(blocks, held_out_ratio, seed)
    n = len(blocks)
    held_n_oracle = max(1, min(n - 1, round(n * held_out_ratio)))
    try:
        count_leaf = _bridge_eval.eval_held_count  # type: ignore[attr-defined]  # reason: eval_leaves2 leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        held_n = held_n_oracle
    else:
        try:
            held_n: int = count_leaf(n, held_out_ratio)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ContractError(str(exc)) from exc
        if held_n != held_n_oracle:
            raise ContractError(f"held_n: Rust {held_n} != Python oracle {held_n_oracle}")
    generator = torch.Generator().manual_seed(seed)
    # reason: explicit-any on tolist(); randperm(n) yields indices in
    # [0, n) by construction, in-range for blocks_list below.
    perm: list[int] = torch.randperm(n, generator=generator).tolist()  # pyrefly: ignore[explicit-any]
    blocks_list: list[WallBlock] = list(blocks)
    held = tuple(blocks_list[i] for i in perm[:held_n])
    train = tuple(blocks_list[i] for i in perm[held_n:])
    # Sort both sides by wall_id for stable manifest (shuffle determines membership,
    # sorted order makes equality assertion order-insensitive to seed path).
    held_sorted = tuple(sorted(held, key=lambda b: b.wall_id))
    train_sorted = tuple(sorted(train, key=lambda b: b.wall_id))
    all_sorted = sorted(block.wall_id for block in blocks)
    held_ids = sorted(block.wall_id for block in held_sorted)
    train_ids = sorted(block.wall_id for block in train_sorted)
    payload = {
        "all_wall_ids_sorted": all_sorted,
        "held_out_wall_ids": held_ids,
        "train_wall_ids": train_ids,
        "held_out_ratio": held_out_ratio,
        "seed": seed,
    }
    digest = of_canonical(payload)
    # Guarantee disjointness; _validate_block_split_input already ensures input uniqueness
    validate_walls_disjoint(
        tuple(block.wall_id for block in train_sorted),
        tuple(block.wall_id for block in held_sorted),
    )
    return BlockSplit(
        train_blocks=train_sorted,
        held_out_blocks=held_sorted,
        seed=seed,
        held_out_ratio=held_out_ratio,
        digest=digest,
    )


# ---------------------------------------------------------------------------
# Balance audit + telemetry completeness reports
# ---------------------------------------------------------------------------


def balance_audit(schedule: MatchSchedule) -> dict[str, Any]:
    """Audit seat balance and return a canonical report dict.

    Checks 6+4 allocations per wall (via :func:`seat_pair_placements_exact`),
    plus global per-label seat counts. The report is deterministic and digest-friendly.
    """
    seat_pair_placements_exact(schedule)
    # Global per-label counts per seat.
    labels = sorted({label for row in schedule.seat_allocations for label in row})
    seat_count_by_label: dict[str, list[int]] = {label: [0, 0, 0, 0] for label in labels}
    for row in schedule.seat_allocations:
        for seat, label in enumerate(row):
            seat_count_by_label[label][seat] += 1
    total_rows = len(schedule.seat_allocations)
    return {
        "walls": len(schedule.wall_ids),
        "games_per_wall": total_rows // max(1, len(schedule.wall_ids)),
        "seat_count_by_label": {
            label: list(counts) for label, counts in seat_count_by_label.items()
        },
        "schedule_hash": of_canonical(schedule.to_json()),
        "balance_exact": True,
    }


def report_telemetry_completeness(
    blocks: tuple[WallBlock, ...],
    telemetry_by_game: Mapping[str, ResourceTelemetry],
    tolerance: TelemetryTolerance | BlockTolerance | None = None,
) -> dict[str, Any]:
    """Return a telemetry completeness report: valid/excluded counts and details.

    Missing or mode-required telemetry is never imputed; blocks with any
    invalid game become excluded with reason detail. The report mirrors
    :func:`hydra2.eval.blocks.aggregate_blocks` but surfaces per-game reasons
    for block-level exclusion.

    ``tolerance`` defaults to :class:`BlockTolerance`; when a
    :class:`TelemetryTolerance` is supplied it governs per-row missing-field
    tolerance only (fallback/timeout/illegal handling remains strict).
    """
    if not isinstance(blocks, tuple) or any(not isinstance(b, WallBlock) for b in blocks):
        raise ContractError("blocks must be tuple of WallBlock")
    if not isinstance(telemetry_by_game, Mapping):
        raise ContractError("telemetry_by_game must be mapping game_id -> ResourceTelemetry")
    # Normalise tolerance to BlockTolerance for block-level invalidity.
    if tolerance is None:
        block_tolerance = BlockTolerance()
        telem_tolerance = TelemetryTolerance()
    elif isinstance(tolerance, BlockTolerance):
        block_tolerance = tolerance
        telem_tolerance = TelemetryTolerance()
    elif isinstance(tolerance, TelemetryTolerance):
        block_tolerance = BlockTolerance()
        telem_tolerance = tolerance
    else:
        raise ContractError("tolerance must be BlockTolerance/TelemetryTolerance or None")

    result: BlockAggregateResult = aggregate_blocks(
        blocks, telemetry_by_game=dict(telemetry_by_game), tolerance=block_tolerance
    )
    # Per-game invalid reasons (for diagnostics / report completeness).
    invalid_by_game: dict[str, str] = {}
    for block in blocks:
        for game_id in block.game_ids:
            row = telemetry_by_game.get(game_id)
            if row is None:
                invalid_by_game[game_id] = "missing telemetry row"
                continue
            reason = telemetry_invalid_reason(row, telem_tolerance)
            if reason is not None:
                invalid_by_game[game_id] = reason
            elif row.fallback_used and not block_tolerance.allow_fallback_used:
                invalid_by_game[game_id] = "fallback_used"
            elif row.timeout and not block_tolerance.allow_timeout:
                invalid_by_game[game_id] = "timeout"
            elif row.illegal_action and not block_tolerance.allow_illegal_action:
                invalid_by_game[game_id] = "illegal_action"

    return {
        "blocks_total": len(blocks),
        "blocks_valid": len(result.valid),
        "blocks_excluded": len(result.excluded),
        "excluded_detail": [
            {"wall_id": exc.wall_id, "reason": exc.reason, "detail": exc.detail}
            for exc in result.excluded
        ],
        "invalid_games": dict(sorted(invalid_by_game.items())),
        "valid_wall_ids": [wall_id for wall_id, _ in result.valid],
        "digest": of_canonical(
            {
                "blocks_total": len(blocks),
                "valid_wall_ids": sorted(wall_id for wall_id, _ in result.valid),
                "excluded": sorted([[exc.wall_id, exc.reason] for exc in result.excluded]),
            }
        ),
    }


def confirmation_sidecar(
    *,
    schedule: MatchSchedule,
    blocks: BlockAggregateResult,
    telemetry_report: Mapping[str, Any] | None = None,
    admission: str = "full",
) -> dict[str, Any]:
    """Additive confirmation sidecar (SPEC 18.4 PR4).

    Binds schedule commitment + wall-block exclusions + telemetry completeness
    beside (never instead of) a confirmation path's own hashes. Pure function;
    decision outputs are untouched. `admission="not-run"` marks callers that
    hand-roll hashes without admission (e.g. teacher five-arms); their empty
    exclusion list MUST NOT be read as a clean bill. Full admission migration
    is WP-10-owned.
    """
    from hydra2.eval.schedule import schedule_commitment_hash

    if not isinstance(schedule, MatchSchedule):
        raise ContractError("schedule must be MatchSchedule")
    if not isinstance(blocks, BlockAggregateResult):
        raise ContractError("blocks must be BlockAggregateResult")
    if admission not in ("full", "not-run"):
        raise ContractError("admission must be 'full' or 'not-run'")
    telemetry_digest: str | None = None
    if telemetry_report is not None:
        if not isinstance(telemetry_report, Mapping):
            raise ContractError("telemetry_report must be a mapping or None")
        telemetry_digest = str(of_canonical(dict(telemetry_report)))
    return {
        "schedule_commitment_hash": str(schedule_commitment_hash(schedule)),
        "admission": admission,
        "excluded": [
            {"wall_id": exc.wall_id, "reason": exc.reason, "detail": exc.detail}
            for exc in blocks.excluded
        ],
        "valid_wall_ids": sorted(wall_id for wall_id, _ in blocks.valid),
        "telemetry_completeness_digest": telemetry_digest,
    }


# ---------------------------------------------------------------------------
# Legacy aliases for test compatibility
# ---------------------------------------------------------------------------

# Some callers import `detect_exact_duplicates`; provide alias.
detect_exact_duplicates = find_exact_duplicates
detect_near_duplicates = find_near_duplicates
