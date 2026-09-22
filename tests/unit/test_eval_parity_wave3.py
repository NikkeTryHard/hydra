"""Wave 3 eval parity — oracle pins for blocks / telemetry / promotion / schedule.

Deterministic oracle goldens for the evaluation plane: wall-block
aggregation (Neumaier sum-then-divide order pinned, including the
cancellation vector the bridge MUST reproduce), the invalid-block policy,
telemetry validity, promotion digests, the match-schedule commitment, and
duplicate detection. Every seed is rooted in
:mod:`hydra2.contracts.randomness` (semantic counter-based streams);
no wall-clock, no ``random`` module.
"""

from __future__ import annotations

import math

import pytest

from hydra2._native import eval as _bridge_eval
from hydra2.contracts.randomness import (
    RandomStream,
    make_random_stream_key,
    semantic_seed,
)
from hydra2.engines.riichienv.walls import derive_hand_wall
from hydra2.eval.blocks import (
    BlockTolerance,
    WallBlock,
    aggregate_blocks,
)
from hydra2.eval.duplicate import (
    find_exact_duplicates,
    wall_fingerprint,
    wall_hash_from_tiles,
)
from hydra2.eval.promotion import (
    make_promotion_record,
    promotion_digest,
    record_to_json,
)
from hydra2.eval.schedule import (
    TOTAL_GAMES_PER_WALL,
    build_match_schedule,
    schedule_commitment_hash,
)
from hydra2.eval.telemetry import (
    TelemetryTolerance,
    make_resource_telemetry,
    telemetry_invalid_reason,
)

pytestmark = pytest.mark.contract_package("WP-03B")

_MASTER = b"wave3-eval-parity-v1"
_EXPERIMENT = "wave3-eval-parity"
_SPLIT = "oracle"

# Contrasts drawn from the evaluation_schedule stream (replicate_id=0),
# frozen on the first green run: repr() exact.
_CONTRASTS: tuple[float, ...] = (
    1.8775715987007295,
    1.461489090472515,
    -0.7859079284493771,
    1.1538786381138424,
    -0.5755889208294411,
    -0.5749658133287925,
)
_CONTRAST_MEAN = 0.42607944411324605


def _stream(purpose: str, replicate_id: int = 0) -> RandomStream:
    key = make_random_stream_key(
        purpose=purpose,
        experiment_id=_EXPERIMENT,
        split_id=_SPLIT,
        replicate_id=replicate_id,
        attempt_id=0,
    )
    return RandomStream(semantic_seed(_MASTER, key=key))


def _telemetry_row(game_id: str = "g-0", wall_id: str = "w-wave3-001", **overrides: object):
    base: dict[str, object] = {
        "mode": "cuda_eager",
        "wall_id": wall_id,
        "case_id": None,
        "candidate_spec_hash": "sha256:" + "11" * 32,
        "hardware_hash": "sha256:" + "22" * 32,
        "environment_hash": "sha256:" + "33" * 32,
        "cold_start": False,
        "synchronized_elapsed_ms": 12.5,
        "model_calls": 3,
        "exact_transitions": 40,
        "particles": 0,
        "fallback_used": False,
        "timeout": False,
        "illegal_action": False,
        "cuda_peak_allocated_bytes": 1024,
        "cuda_peak_reserved_bytes": 2048,
        "host_peak_bytes": None,
        "energy_joules": None,
        "graph_breaks": None,
        "recompiles": None,
        "invalid_reason": None,
    }
    base.update(overrides)
    return make_resource_telemetry(**base)


def _schedule():
    return build_match_schedule(
        wall_ids=("w-wave3-001", "w-wave3-002"),
        labels=("A", "B", "C", "D"),
        rules_hash="sha256:" + "aa" * 32,
        master_seed=_MASTER,
        experiment_id=_EXPERIMENT,
        split_id=_SPLIT,
    )


# ---------------------------------------------------------------------------
# Block aggregation goldens
# ---------------------------------------------------------------------------


def test_block_contrasts_drawn_from_semantic_stream() -> None:
    rng = _stream("evaluation_schedule")
    assert tuple(rng.random_float() * 4 - 2 for _ in range(6)) == _CONTRASTS


def test_block_aggregation_mean_golden() -> None:
    block = WallBlock(
        wall_id="w-wave3-001",
        game_ids=tuple(f"g-{i}" for i in range(6)),
        contrasts=_CONTRASTS,
    )
    # Neumaier sum-then-divide order pinned (byte-identity trap): the divisor
    # applies AFTER the compensated sum, never folded into the accumulation.
    assert _bridge_eval.aggregate_wall_block(block) == math.fsum(_CONTRASTS) / len(_CONTRASTS)
    assert _bridge_eval.aggregate_wall_block(block) == _CONTRAST_MEAN


def test_block_aggregation_cancellation_vector() -> None:
    """Compensated summation is observable: naive left-fold sums to 0.0."""
    block = WallBlock(
        wall_id="w-cancel",
        game_ids=("g0", "g1", "g2"),
        contrasts=(1e16, 1.0, -1e16),
    )
    assert (1e16 + 1.0 - 1e16) / 3 == 0.0  # what a naive port would compute
    assert _bridge_eval.aggregate_wall_block(block) == 0.3333333333333333


def test_block_aggregation_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty wall block has no games"):
        _bridge_eval.aggregate_wall_block(WallBlock(wall_id="w-empty", game_ids=(), contrasts=()))


def test_blocks_aggregate_sorted_and_valid_golden() -> None:
    blocks = (
        WallBlock(wall_id="w-wave3-002", game_ids=("h-0",), contrasts=(2.0,)),
        WallBlock(
            wall_id="w-wave3-001",
            game_ids=tuple(f"g-{i}" for i in range(6)),
            contrasts=_CONTRASTS,
        ),
    )
    rows = {f"g-{i}": _telemetry_row(f"g-{i}") for i in range(6)}
    rows["h-0"] = _telemetry_row("h-0", wall_id="w-wave3-002")
    result = aggregate_blocks(blocks, telemetry_by_game=rows)
    # Sorted-by-wall_id order pinned (bridge MUST preserve it).
    assert [wall for wall, _ in result.valid] == ["w-wave3-001", "w-wave3-002"]
    assert result.valid[0][1] == _CONTRAST_MEAN
    assert result.valid[1][1] == 2.0
    assert result.excluded == ()


def test_blocks_exclusion_reasons() -> None:
    block = WallBlock(
        wall_id="w-wave3-001",
        game_ids=tuple(f"g-{i}" for i in range(6)),
        contrasts=_CONTRASTS,
    )
    rows = {f"g-{i}": _telemetry_row(f"g-{i}") for i in range(6)}

    fallen = dict(rows)
    fallen["g-1"] = _telemetry_row("g-1", fallback_used=True)
    excluded = aggregate_blocks((block,), telemetry_by_game=fallen).excluded
    assert [(e.wall_id, e.reason) for e in excluded] == [("w-wave3-001", "fallback_used")]

    marked = dict(rows)
    marked["g-2"] = _telemetry_row("g-2", invalid_reason="engine panicked")
    excluded = aggregate_blocks((block,), telemetry_by_game=marked).excluded
    assert [(e.wall_id, e.reason) for e in excluded] == [("w-wave3-001", "row_invalid")]

    excluded = aggregate_blocks((block,), telemetry_by_game={}).excluded
    assert [(e.wall_id, e.reason) for e in excluded] == [("w-wave3-001", "missing_telemetry")]

    strict = aggregate_blocks(
        (block,), telemetry_by_game=fallen, tolerance=BlockTolerance(allow_fallback_used=True)
    )
    assert [wall for wall, _ in strict.valid] == ["w-wave3-001"]


# ---------------------------------------------------------------------------
# Telemetry / promotion digests
# ---------------------------------------------------------------------------


def test_telemetry_clean_row_is_valid() -> None:
    assert telemetry_invalid_reason(_telemetry_row(), TelemetryTolerance()) is None


def test_telemetry_marked_invalid_surfaces_verbatim() -> None:
    row = _telemetry_row(invalid_reason="engine panicked")
    assert (
        telemetry_invalid_reason(row, TelemetryTolerance()) == "row marked invalid: engine panicked"
    )


def test_promotion_digest_golden() -> None:
    schedule = _schedule()
    commitment = schedule_commitment_hash(schedule)
    record = make_promotion_record(
        candidate_spec_hash="sha256:" + "11" * 32,
        utility_manifest_hash="sha256:" + "22" * 32,
        comparator_spec_hashes=("sha256:" + "33" * 32,),
        case_manifest_hash="sha256:" + "44" * 32,
        result_table_hash="sha256:" + "55" * 32,
        resource_view="reference_eager_cpu",
        uncertainty_unit="wall_block",
        pass_inequality="greater",
        observed_estimate=_CONTRAST_MEAN,
        confidence_bounds=(0.1, 0.9),
        gates={"correctness": "passed", "determinism": "passed"},
        disposition="promoted",
        schedule_hash=commitment,
    )
    projected = record_to_json(record)
    assert projected["gates"] == {"correctness": "passed", "determinism": "passed"}
    assert projected["schedule_hash"] == commitment
    assert projected["observed_estimate"] == _CONTRAST_MEAN
    assert (
        promotion_digest(record)
        == "sha256:7e8b0b07bd0161549857320aa7a358d547271934861c0f883c51e2a1c002d9f0"
    )


# ---------------------------------------------------------------------------
# Schedule commitment + duplicates
# ---------------------------------------------------------------------------


def test_schedule_commitment_goldens() -> None:
    schedule = _schedule()
    assert TOTAL_GAMES_PER_WALL == 10
    assert len(schedule.seat_allocations) == 20
    assert schedule.walls_hash == (
        "sha256:5c099f387f11033288a65a84c074e4d066cbaeba6722d6653f98b355281966d8"
    )
    assert schedule.latency_schedule_hash == (
        "sha256:600f8227eded113b2cc266c91da40fefe41c44ea7c618d83701459692914b725"
    )
    assert schedule.seed_protocol_hash == (
        "sha256:30538109867443d98bfc50220c1011b4fee656a702d348f2690632e6f9fa63b4"
    )
    assert schedule_commitment_hash(schedule) == (
        "sha256:f965e53800c02e855fc83671b65bc82b5ddefd51b520e9e6e4706ca4334e60d3"
    )


def test_schedule_determinism_repeat() -> None:
    assert schedule_commitment_hash(_schedule()) == schedule_commitment_hash(_schedule())


def test_duplicate_wall_hash_goldens() -> None:
    wall = derive_hand_wall(
        schedule_digest="sha256:" + "ab" * 32,
        schedule_id="sched-wave3-001",
        hand_index=1,
    )
    tiles = tuple(int(t) for t in wall)
    assert wall_hash_from_tiles(tiles) == (
        "sha256:6f816227ede1bdee61837f64a585cad47c4a5c7db27e4b6605d98ff2ecf17dee"
    )
    assert wall_fingerprint(tiles) == (
        "sha256:87ef3e03a99fdd08632d1e74dda2c6287b549293c6d781c0693789f1095ae25c"
    )
    same = str(wall_hash_from_tiles(tiles))
    assert find_exact_duplicates({"a": same, "b": same}) == [("a", "b")]
