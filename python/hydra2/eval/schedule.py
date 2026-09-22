"""SPEC 18.1 match schedules — committed before any result exists.

For each wall the schedule fixes, in order:

* six symmetric 2-v-2 allocations — every placement of the A-pair in a
  distinct unordered pair of seats occurs exactly once (C(4,2) = 6);
* four 1-v-3 diagnostic rotations — the focal candidate A rotates through
  all four seats against the three field labels.

The commitment (walls + seat allocations + latency schedule + seed protocol
+ rules identity) is a pure function of the constructor inputs; results are
never an input, so rebuilding the schedule from the same inputs is
byte-identical and the commitment hash can be recorded before play begins.

Latency classes are derived per game from an ``evaluation_schedule``
semantic stream so simulated latency is part of the committed protocol, not
an afterthought.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import of_canonical, validate_digest
from hydra2.contracts.common import ContractError, DigestText
from hydra2.contracts.randomness import RandomStreamKey, make_random_stream_key, semantic_seed

if TYPE_CHECKING:
    from collections.abc import Sequence
__all__ = [
    "LATENCY_CLASSES",
    "SYMMETRIC_ALLOCATIONS_PER_WALL",
    "TOTAL_GAMES_PER_WALL",
    "MatchSchedule",
    "build_match_schedule",
    "schedule_commitment_hash",
    "seat_pair_placements_exact",
]

#: Latency vocabulary (SPEC 18.1, bridge-owned; fallback keeps pre-wire imports green).
LATENCY_CLASSES: tuple[str, str, str] = (
    tuple(_bridge_contracts.LATENCY_CLASSES)  # type: ignore[attr-defined]  # reason: eval_schedule leaf lands with MAIN wiring; hasattr fallback covers stale .so
    if hasattr(_bridge_contracts, "LATENCY_CLASSES")
    else ("low", "moderate", "high")
)
#: Symmetric 2-v-2 rows per wall (bridge-owned).
SYMMETRIC_ALLOCATIONS_PER_WALL: int = (
    int(_bridge_contracts.SYMMETRIC_ALLOCATIONS_PER_WALL)  # type: ignore[attr-defined]  # reason: eval_schedule leaf lands with MAIN wiring; hasattr fallback covers stale .so
    if hasattr(_bridge_contracts, "SYMMETRIC_ALLOCATIONS_PER_WALL")
    else 6
)
#: Focal 1-v-3 rows per wall (bridge-owned; not in __all__ by oracle design).
ROTATION_ALLOCATIONS_PER_WALL: int = (
    int(_bridge_contracts.ROTATION_ALLOCATIONS_PER_WALL)  # type: ignore[attr-defined]  # reason: eval_schedule leaf lands with MAIN wiring; hasattr fallback covers stale .so
    if hasattr(_bridge_contracts, "ROTATION_ALLOCATIONS_PER_WALL")
    else 4
)
#: Total games per wall, derived like the oracle (6 + 4 = 10).
TOTAL_GAMES_PER_WALL: int = (
    int(_bridge_contracts.TOTAL_GAMES_PER_WALL)  # type: ignore[attr-defined]  # reason: eval_schedule leaf lands with MAIN wiring; hasattr fallback covers stale .so
    if hasattr(_bridge_contracts, "TOTAL_GAMES_PER_WALL")
    else SYMMETRIC_ALLOCATIONS_PER_WALL + ROTATION_ALLOCATIONS_PER_WALL
)

_SEED_PROTOCOL_PAYLOAD: dict[str, object] = {
    "protocol": "hydra2_rng_v1",
    "seed_protocol_version": 1,
    "derivation": (
        "sha256(canonical_json({protocol, master_seed.hex, key})) with "
        "purpose-discriminated RandomStreamKey"
    ),
    "schedule_purposes": ["wall", "evaluation_schedule"],
    "commitment_order": ["walls", "seats", "latency", "protocol"],
}


@dataclass(frozen=True, slots=True)
class MatchSchedule:
    """SPEC 18.1 schedule; frozen before results exist."""

    wall_ids: tuple[str, ...]
    walls_hash: DigestText
    seat_allocations: tuple[tuple[str, str, str, str], ...]
    latency_schedule_hash: DigestText
    rules_hash: DigestText
    seed_protocol_hash: DigestText

    def __post_init__(self) -> None:
        if len(self.wall_ids) == 0:
            raise ContractError("schedule needs at least one wall")
        if len(set(self.wall_ids)) != len(self.wall_ids):
            raise ContractError("wall_ids must be unique")
        if len(self.seat_allocations) != len(self.wall_ids) * TOTAL_GAMES_PER_WALL:
            raise ContractError(f"seat_allocations must carry {TOTAL_GAMES_PER_WALL} rows per wall")
        for digest_name in (
            "walls_hash",
            "latency_schedule_hash",
            "rules_hash",
            "seed_protocol_hash",
        ):
            _digest: DigestText = validate_digest(getattr(self, digest_name))
        for row in self.seat_allocations:
            if len(set(row)) != 4:
                raise ContractError(f"allocation row {row} is not a 4-label permutation")

    def to_json(self) -> dict[str, object]:
        return {
            "wall_ids": list(self.wall_ids),
            "walls_hash": self.walls_hash,
            "seat_allocations": [list(row) for row in self.seat_allocations],
            "latency_schedule_hash": self.latency_schedule_hash,
            "rules_hash": self.rules_hash,
            "seed_protocol_hash": self.seed_protocol_hash,
        }


def _validate_labels(labels: Sequence[str]) -> tuple[str, str, str, str]:
    """Validate four distinct nonempty labels (bridge-checked)."""
    try:
        leaf = _bridge_contracts.eval_schedule_check_labels  # type: ignore[attr-defined]  # reason: eval_schedule leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        pass
    else:
        try:
            checked: tuple[str, str, str, str] = leaf(list(labels))
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
        return checked[0], checked[1], checked[2], checked[3]
    if len(labels) != 4 or any(not isinstance(label, str) or label == "" for label in labels):
        raise ContractError("labels must be four nonempty strings")
    if len(set(labels)) != 4:
        raise ContractError("labels must be distinct")
    return labels[0], labels[1], labels[2], labels[3]


def symmetric_pair_allocations(
    pair: tuple[str, str], field: tuple[str, str]
) -> list[tuple[str, str, str, str]]:
    """All six placements of ``pair`` across unordered seat pairs (bridge-checked)."""
    leaf = _bridge_contracts.eval_schedule_symmetric_allocations  # type: ignore[attr-defined]  # reason: eval_schedule leaf on shared contracts submodule; stale .so raises AttributeError
    try:
        return [(row[0], row[1], row[2], row[3]) for row in leaf(list(pair), list(field))]
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def focal_rotation_allocations(
    focal: str, others: tuple[str, str, str]
) -> list[tuple[str, str, str, str]]:
    """Four 1-v-3 diagnostics; focal sits at each seat exactly once (bridge-checked)."""
    leaf = _bridge_contracts.eval_schedule_rotation_allocations  # type: ignore[attr-defined]  # reason: eval_schedule leaf on shared contracts submodule; stale .so raises AttributeError
    try:
        return [(row[0], row[1], row[2], row[3]) for row in leaf(focal, list(others))]
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def build_match_schedule(
    *,
    wall_ids: Sequence[str],
    labels: Sequence[str],
    rules_hash: str,
    master_seed: bytes,
    experiment_id: str,
    split_id: str,
) -> MatchSchedule:
    """Commit walls, seats, latency classes, and seed protocol up front."""
    ids = tuple(wall_ids)
    if len(ids) == 0 or any(not isinstance(wall_id, str) or wall_id == "" for wall_id in ids):
        raise ContractError("wall_ids must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ContractError("wall_ids must be unique")
    first, second, third, fourth = _validate_labels(labels)
    pair = (first, second)
    field = (third, fourth)
    others = (second, third, fourth)

    allocations: list[tuple[str, str, str, str]] = []
    latency_rows: list[list[object]] = []
    for wall_index, wall_id in enumerate(ids):
        allocations.extend(symmetric_pair_allocations(pair, field))
        allocations.extend(focal_rotation_allocations(first, others))
        for slot in range(TOTAL_GAMES_PER_WALL):
            key = make_random_stream_key(
                purpose="evaluation_schedule",
                experiment_id=experiment_id,
                split_id=split_id,
                replicate_id=wall_index * TOTAL_GAMES_PER_WALL + slot,
                attempt_id=0,
            )
            class_index = _latency_draw(master_seed, key)
            latency_rows.append([wall_id, slot, LATENCY_CLASSES[class_index]])

    oracle_proto_hash = of_canonical(_SEED_PROTOCOL_PAYLOAD)
    try:
        proto_leaf = _bridge_contracts.eval_schedule_seed_protocol_hash  # type: ignore[attr-defined]  # reason: eval_schedule leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        seed_protocol_hash = oracle_proto_hash
    else:
        try:
            proto_hex: str = proto_leaf()
            bridged_proto_hash = DigestText(proto_hex)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
        if bridged_proto_hash != oracle_proto_hash:
            raise ContractError(
                "seed_protocol_hash: Rust digest "
                f"{bridged_proto_hash} != Python oracle {oracle_proto_hash}"
            )
        seed_protocol_hash = bridged_proto_hash
    return MatchSchedule(
        wall_ids=ids,
        walls_hash=of_canonical(list(ids)),
        seat_allocations=tuple(allocations),
        latency_schedule_hash=of_canonical(latency_rows),
        rules_hash=validate_digest(rules_hash),
        seed_protocol_hash=seed_protocol_hash,
    )


def _latency_draw(master_seed: bytes, key: RandomStreamKey) -> int:
    """Latency class index via the bridge draw with oracle parity (``%3`` verbatim)."""
    seed = semantic_seed(master_seed, key=key)
    oracle_index = int.from_bytes(seed[:8], "big") % len(LATENCY_CLASSES)
    try:
        leaf = _bridge_contracts.eval_schedule_latency_draw  # type: ignore[attr-defined]  # reason: eval_schedule leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        return oracle_index
    try:
        bridged_index: int = leaf(master_seed, key.experiment_id, key.split_id, key.replicate_id)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ContractError(str(exc)) from exc
    if bridged_index != oracle_index:
        raise ContractError(
            f"_latency_draw: Rust index {bridged_index} != Python oracle {oracle_index}"
        )
    return bridged_index


def _rust_canonical_digest(value: object, oracle: DigestText, *, subject: str) -> DigestText:
    """Rust-first digest via feed::canon+digest (bridge ``canon_rng``).

    Recomputes ``of_canonical(value)`` through the Rust bridge
    (``hydra2._native.canon_rng.of_canonical_json``: feed I-JSON parse +
    JCS canon + sha256, B3 canon-wins single site) and fail-closes on
    mismatch (``ContractError``). Missing bridge raises ``ImportError``
    naming the ``canon_rng`` submodule with a `pixi run build-ext` hint
    (never silently returns the Python ``oracle``). B2/M3 untouched —
    pure digest identity, no draws.
    """
    try:
        import importlib as _importlib

        _ = _importlib.import_module("hydra2._native")
    except ImportError as exc:
        raise ImportError(
            "hydra2._native extension with canon_rng not importable; "
            "build the bridge with `pixi run build-ext` before hashing "
            f"{subject}"
        ) from exc
    try:
        from hydra2 import _rust_bridge as _rust_bridge_mod
        from hydra2.artifacts.canonical import canonical_bytes as _canonical_bytes

        rust_digest = _rust_bridge_mod.of_canonical(_canonical_bytes(value))
    except RuntimeError as exc:
        if "not importable" in str(exc) or "missing" in str(exc):
            raise ImportError(
                "hydra2._native.canon_rng submodule missing (stale .so); "
                "rebuild the bridge with `pixi run build-ext` before hashing "
                f"{subject}"
            ) from exc
        raise
    if rust_digest != oracle:
        raise ContractError(f"{subject}: Rust digest {rust_digest} != Python oracle {oracle}")
    return rust_digest


def schedule_commitment_hash(schedule: MatchSchedule) -> DigestText:
    """Single pre-results commitment binding every schedule facet.

    Rust-first via the dedicated ``eval_schedule_commitment_hash_from_doc``
    leaf (feed::canon+digest, B3 canon-wins; T2 pins the COMMIT golden;
    mismatch raises ContractError, missing leaf falls back to the
    ``canon_rng`` guard). B2/M3 untouched — no split or latency draws here
    (``_latency_draw`` stays ``%3`` verbatim, splits stay torch.randperm).
    """
    payload = schedule.to_json()
    oracle = of_canonical(payload)
    try:
        leaf = _bridge_contracts.eval_schedule_commitment_hash_from_doc  # type: ignore[attr-defined]  # reason: eval_schedule leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        return _rust_canonical_digest(payload, oracle, subject="schedule_commitment_hash")
    try:
        commitment_hex: str = leaf(payload)
        bridged = DigestText(commitment_hex)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    if bridged != oracle:
        raise ContractError(
            f"schedule_commitment_hash: Rust digest {bridged} != Python oracle {oracle}"
        )
    return bridged


def seat_pair_placements_exact(schedule: MatchSchedule) -> None:
    """Exactness gates over the allocation structure; raises when violated.

    Bridge-checked via ``eval_schedule_placements_check`` (wall ids +
    allocation scalars cross; the oracle-exact ``wall '<id>': ...`` texts
    are rendered Rust-side). Per wall, over the six symmetric rows: every
    unordered seat pair hosts the A-pair exactly once and each seat hosts an
    A-pair member exactly three times; over the four rotation rows the focal
    label visits each seat exactly once.
    """
    leaf = _bridge_contracts.eval_schedule_placements_check  # type: ignore[attr-defined]  # reason: eval_schedule leaf on shared contracts submodule; stale .so raises AttributeError
    try:
        leaf(
            list(schedule.wall_ids),
            [list(row) for row in schedule.seat_allocations],
        )
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
