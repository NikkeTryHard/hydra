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

LATENCY_CLASSES: tuple[str, str, str] = ("low", "moderate", "high")
SYMMETRIC_ALLOCATIONS_PER_WALL = 6
ROTATION_ALLOCATIONS_PER_WALL = 4
TOTAL_GAMES_PER_WALL = SYMMETRIC_ALLOCATIONS_PER_WALL + ROTATION_ALLOCATIONS_PER_WALL

_SEAT_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
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
    if len(labels) != 4 or any(not isinstance(label, str) or label == "" for label in labels):
        raise ContractError("labels must be four nonempty strings")
    if len(set(labels)) != 4:
        raise ContractError("labels must be distinct")
    return labels[0], labels[1], labels[2], labels[3]


def symmetric_pair_allocations(
    pair: tuple[str, str], field: tuple[str, str]
) -> list[tuple[str, str, str, str]]:
    """All six placements of ``pair`` across unordered seat pairs."""
    rows = []
    for low_seat, high_seat in _SEAT_PAIRS:
        row: list[str] = ["", "", "", ""]
        row[low_seat], row[high_seat] = pair
        remaining = [seat for seat in range(4) if seat not in (low_seat, high_seat)]
        row[remaining[0]], row[remaining[1]] = field
        rows.append((row[0], row[1], row[2], row[3]))
    return rows


def focal_rotation_allocations(
    focal: str, others: tuple[str, str, str]
) -> list[tuple[str, str, str, str]]:
    """Four 1-v-3 diagnostics; focal sits at each seat exactly once."""
    rows = []
    for seat in range(4):
        row: list[str] = list(others)
        row.insert(seat, focal)
        rows.append((row[0], row[1], row[2], row[3]))
    return rows


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

    return MatchSchedule(
        wall_ids=ids,
        walls_hash=of_canonical(list(ids)),
        seat_allocations=tuple(allocations),
        latency_schedule_hash=of_canonical(latency_rows),
        rules_hash=validate_digest(rules_hash),
        seed_protocol_hash=of_canonical(_SEED_PROTOCOL_PAYLOAD),
    )


def _latency_draw(master_seed: bytes, key: RandomStreamKey) -> int:
    seed = semantic_seed(master_seed, key=key)
    return int.from_bytes(seed[:8], "big") % len(LATENCY_CLASSES)


def schedule_commitment_hash(schedule: MatchSchedule) -> DigestText:
    """Single pre-results commitment binding every schedule facet."""
    return of_canonical(schedule.to_json())


def seat_pair_placements_exact(schedule: MatchSchedule) -> None:
    """Exactness gates over the allocation structure; raises when violated.

    Per wall, over the six symmetric rows: every unordered seat pair hosts
    the A-pair exactly once and each seat hosts an A-pair member exactly
    three times; over the four rotation rows the focal label visits each
    seat exactly once.
    """
    pair_members = frozenset(schedule.seat_allocations[0][:2])
    focal = schedule.seat_allocations[SYMMETRIC_ALLOCATIONS_PER_WALL][0]
    for wall_index in range(len(schedule.wall_ids)):
        base = wall_index * TOTAL_GAMES_PER_WALL
        symmetric = schedule.seat_allocations[base : base + SYMMETRIC_ALLOCATIONS_PER_WALL]
        rotations = schedule.seat_allocations[
            base + SYMMETRIC_ALLOCATIONS_PER_WALL : base + TOTAL_GAMES_PER_WALL
        ]
        placements = sorted(
            tuple(sorted(index for index, label in enumerate(row) if label in pair_members))
            for row in symmetric
        )
        if placements != list(_SEAT_PAIRS):
            raise ContractError(
                f"wall {schedule.wall_ids[wall_index]!r}: A-pair placements not exact"
            )
        seat_hits = [0, 0, 0, 0]
        for row in symmetric:
            for index, label in enumerate(row):
                if label in pair_members:
                    seat_hits[index] += 1
        if seat_hits != [3, 3, 3, 3]:
            raise ContractError(
                f"wall {schedule.wall_ids[wall_index]!r}: seat coverage {seat_hits} != [3,3,3,3]"
            )
        rotation_seats = sorted(
            index for row in rotations for index, label in enumerate(row) if label == focal
        )
        if rotation_seats != [0, 1, 2, 3]:
            raise ContractError(
                f"wall {schedule.wall_ids[wall_index]!r}: focal rotation "
                f"{rotation_seats} != all seats"
            )
