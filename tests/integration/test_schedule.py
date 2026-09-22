"""WP-03B gate: SPEC 18.1 match schedules committed before results.

End-to-end slice across hydra2.eval.schedule, hydra2.contracts.randomness,
hydra2.eval.blocks, and hydra2.eval.statistics:

* replay is byte-identical (canonical bytes and commitment hash);
* six symmetric 2-v-2 placements and four 1-v-3 rotations per wall;
* seat balance is EXACT under the schedule;
* the commitment binds every facet before any result exists;
* schedule-driven synthetic confirmation recovers planted effects using
  whole-wall blocks only.
"""

from __future__ import annotations

from statistics import NormalDist, fmean

import pytest

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed
from hydra2.eval.blocks import BlockTolerance, WallBlock, aggregate_blocks
from hydra2.eval.schedule import (
    SYMMETRIC_ALLOCATIONS_PER_WALL,
    TOTAL_GAMES_PER_WALL,
    build_match_schedule,
    schedule_commitment_hash,
    seat_pair_placements_exact,
)
from hydra2.eval.statistics import bootstrap_blocks, ci_covers

pytestmark = pytest.mark.contract_package("WP-03B")

LABELS = ("cand-a", "partner-a", "base-b", "field-c")
RULES = "sha256:" + "cd" * 32
MASTER = bytes(range(48, 80))
WALL_IDS = tuple(f"w-{index:02d}" for index in range(12))


def _build() -> object:
    return build_match_schedule(
        wall_ids=WALL_IDS,
        labels=LABELS,
        rules_hash=RULES,
        master_seed=MASTER,
        experiment_id="exp-wp03b",
        split_id="confirm",
    )


def test_schedule_replay_is_byte_identical() -> None:
    first = _build()
    second = _build()
    assert first == second
    assert canonical_bytes(first.to_json()) == canonical_bytes(second.to_json())
    assert schedule_commitment_hash(first) == schedule_commitment_hash(second)
    # Any input facet change alters the commitment (mutation sensitivity).
    drifted_latency = build_match_schedule(
        wall_ids=WALL_IDS,
        labels=LABELS,
        rules_hash=RULES,
        master_seed=bytes(range(49, 81)),
        experiment_id="exp-wp03b",
        split_id="confirm",
    )
    assert drifted_latency.seat_allocations == first.seat_allocations
    assert drifted_latency.latency_schedule_hash != first.latency_schedule_hash
    assert schedule_commitment_hash(drifted_latency) != schedule_commitment_hash(first)


def test_six_symmetric_and_four_rotation_allocations_per_wall() -> None:
    schedule = _build()
    assert len(schedule.seat_allocations) == len(WALL_IDS) * TOTAL_GAMES_PER_WALL
    seat_pair_placements_exact(schedule)
    base = 0
    symmetric = schedule.seat_allocations[base : base + SYMMETRIC_ALLOCATIONS_PER_WALL]
    placements = sorted(
        tuple(sorted(i for i, label in enumerate(row) if label in LABELS[:2])) for row in symmetric
    )
    assert placements == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    rotations = schedule.seat_allocations[
        base + SYMMETRIC_ALLOCATIONS_PER_WALL : base + TOTAL_GAMES_PER_WALL
    ]
    assert sorted(row.index(LABELS[0]) for row in rotations) == [0, 1, 2, 3]


def test_seat_balance_exact_under_schedule() -> None:
    """Every row a permutation; pair seats exactly balanced; walls identical."""
    schedule = _build()
    pair_members = frozenset(LABELS[:2])
    for row in schedule.seat_allocations:
        assert sorted(row) == sorted(LABELS)
    per_wall_rows = TOTAL_GAMES_PER_WALL
    for wall_index in range(len(WALL_IDS)):
        start = wall_index * per_wall_rows
        rows = schedule.seat_allocations[start : start + per_wall_rows]
        reference = schedule.seat_allocations[0:per_wall_rows]
        assert rows == reference, "duplicate-wall design repeats identical allocations"
    # Across the whole schedule each label appears equally often overall.
    counts = dict.fromkeys(LABELS, 0)
    for row in schedule.seat_allocations:
        for label in row:
            counts[label] += 1
    assert len(set(counts.values())) == 1
    # Pair-level seat coverage per wall is exactly three hits per seat.
    hits = [0, 0, 0, 0]
    for row in schedule.seat_allocations[:SYMMETRIC_ALLOCATIONS_PER_WALL]:
        for seat, label in enumerate(row):
            if label in pair_members:
                hits[seat] += 1
    assert hits == [3, 3, 3, 3]
    # Latency classes come from the committed protocol, not ad-hoc state.
    assert schedule.seed_protocol_hash == of_canonical(
        {
            "protocol": "hydra2_rng_v1",
            "seed_protocol_version": 1,
            "derivation": (
                "sha256(canonical_json({protocol, master_seed.hex, key})) with "
                "purpose-discriminated RandomStreamKey"
            ),
            "schedule_purposes": ["wall", "evaluation_schedule"],
            "commitment_order": ["walls", "seats", "latency", "protocol"],
        }
    )
    assert schedule.walls_hash == of_canonical(list(WALL_IDS))


def test_committed_before_results_binding() -> None:
    """Commitment is a pure function of pre-results inputs only."""
    schedule = _build()
    commitment = schedule_commitment_hash(schedule)
    rebuilt = build_match_schedule(
        wall_ids=list(WALL_IDS),
        labels=list(LABELS),
        rules_hash=RULES,
        master_seed=MASTER,
        experiment_id="exp-wp03b",
        split_id="confirm",
    )
    assert commitment == schedule_commitment_hash(rebuilt)
    for changed_labels in (
        ("cand-a", "partner-a", "base-b", "field-z"),
        ("partner-a", "cand-a", "base-b", "field-c"),
    ):
        other = build_match_schedule(
            wall_ids=WALL_IDS,
            labels=changed_labels,
            rules_hash=RULES,
            master_seed=MASTER,
            experiment_id="exp-wp03b",
            split_id="confirm",
        )
        assert schedule_commitment_hash(other) != commitment


def test_schedule_driven_synthetic_confirmation_gates() -> None:
    """Planted arm effect recovered using ONLY whole-wall blocks."""
    schedule = _build()
    tolerance = BlockTolerance()

    def contrasts_for(effect: float) -> list[float]:
        blocks = []
        for wall_index, wall_id in enumerate(WALL_IDS):
            game_contrasts = []
            for slot in range(TOTAL_GAMES_PER_WALL):
                row = schedule.seat_allocations[wall_index * TOTAL_GAMES_PER_WALL + slot]
                key = make_random_stream_key(
                    purpose="confirmation",
                    experiment_id="exp-wp03b",
                    split_id="confirm",
                    candidate_id="cand-a",
                    case_id=None,
                    wall_id=wall_id,
                    replicate_id=slot,
                    attempt_id=0,
                )
                stream = RandomStream(semantic_seed(MASTER, key=key))
                gauss = NormalDist().inv_cdf(stream.random_float())
                focal_advantage = effect if row[0] in LABELS[:2] else -effect
                game_contrasts.append(focal_advantage + 0.35 * gauss)
            blocks.append((wall_id, fmean(game_contrasts)))
        return [
            value
            for _, value in aggregate_blocks(
                tuple(
                    WallBlock(wall_id=wall_id, game_ids=(f"{wall_id}-g",), contrasts=(value,))
                    for wall_id, value in blocks
                ),
                telemetry_by_game={f"{wall_id}-g": _telemetry_row(wall_id) for wall_id in WALL_IDS},
                tolerance=tolerance,
            ).valid
        ]

    def _telemetry_row(wall_id: str):
        from hydra2.eval.telemetry import make_resource_telemetry

        return make_resource_telemetry(
            mode="reference_eager_cpu",
            wall_id=wall_id,
            case_id=None,
            candidate_spec_hash=RULES,
            hardware_hash=RULES,
            environment_hash=RULES,
            cold_start=False,
            synchronized_elapsed_ms=1.0,
            model_calls=0,
            exact_transitions=1,
            particles=0,
            fallback_used=False,
            timeout=False,
            illegal_action=False,
            cuda_peak_allocated_bytes=None,
            cuda_peak_reserved_bytes=None,
            host_peak_bytes=None,
            energy_joules=None,
            graph_breaks=None,
            recompiles=None,
            invalid_reason=None,
        )

    zero = contrasts_for(effect=0.0)
    _, low, high = bootstrap_blocks(zero, stream=RandomStream(b"\x11" * 32), resamples=400)
    assert ci_covers((low, high), 0.0), "seat-balanced schedule must show no phantom effect"

    planted = contrasts_for(effect=0.9)
    _, low, high = bootstrap_blocks(planted, stream=RandomStream(b"\x12" * 32), resamples=400)
    assert not ci_covers((low, high), 0.0), "planted effect must separate from 0"
    assert ci_covers((low, high), fmean(planted))
