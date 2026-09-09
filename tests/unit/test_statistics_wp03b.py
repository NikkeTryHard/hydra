"""WP-03B gate: SPEC 18.3 uncertainty machinery and synthetic effect gates.

Synthetic gates required by BUILD WP-03B:

* known zero effect recovered (CI covers 0);
* known nonzero effect recovered (CI excludes 0, covers truth);
* NEGATIVE: per-game independent resampling produces wrong coverage on
  clustered synthetic data — the naive method MUST FAIL the gate while the
  whole-block bootstrap passes on identical data.

All randomness flows through semantic streams so every number here is
reproducible.
"""

from __future__ import annotations

import dataclasses
import math
from statistics import NormalDist, fmean

import pytest

from hydra2.contracts.common import ContractError
from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed
from hydra2.eval.blocks import WallBlock
from hydra2.eval.statistics import (
    SelectionConfig,
    bootstrap_blocks,
    ci_covers,
    cluster_bootstrap,
    fixed_n_samples,
    hedged_confidence_sequence,
    hedged_cs_path,
    placement_block_contrast,
    selection_gate_check,
    sequential_design_guard,
    sign_flip_interval,
)

pytestmark = pytest.mark.contract_package("WP-03B")

WALLS = 30
GAMES_PER_WALL = 8
EXPERIMENTS = 120
RESAMPLES = 60
BLOCK_EFFECT_SD = 1.5
GAME_NOISE_SD = 0.35


def _gauss(stream: RandomStream) -> float:
    return NormalDist().inv_cdf(stream.random_float())


def _synthetic_walls(*, experiment_index: int, effect: float) -> list[list[float]]:
    """One synthetic confirmation run: wall-clustered per-game contrasts."""
    key = make_random_stream_key(
        purpose="evaluation_schedule",
        experiment_id="synth-wp03b",
        split_id=f"exp-{experiment_index}",
        replicate_id=experiment_index,
        attempt_id=0,
    )
    stream = RandomStream(semantic_seed(b"wp03b-synth-master", key=key))
    walls: list[list[float]] = []
    for _ in range(WALLS):
        wall_effect = _gauss(stream) * BLOCK_EFFECT_SD
        walls.append(
            [effect + wall_effect + _gauss(stream) * GAME_NOISE_SD for _ in range(GAMES_PER_WALL)]
        )
    return walls


def _block_means(walls: list[list[float]]) -> list[float]:
    """The ONLY legal statistics input: one number per whole wall block."""
    return [fmean(games) for games in walls]


def test_fixed_n_formula_reference_value() -> None:
    """SPEC formula pinned: N = ceil(((z_(1-a) + z_(1-b)) * s / delta)^2)."""
    assert fixed_n_samples(s=1.0, delta=0.5, alpha=0.05, beta=0.20) == 25
    normal = NormalDist()
    reference = ((normal.inv_cdf(0.90) + normal.inv_cdf(0.90)) * 2.3 / 0.7) ** 2
    assert fixed_n_samples(s=2.3, delta=0.7, alpha=0.10, beta=0.10) == math.ceil(reference)
    # Monotonicity: demanding a smaller effect needs more blocks.
    assert fixed_n_samples(s=1.0, delta=0.25, alpha=0.05, beta=0.20) > fixed_n_samples(
        s=1.0, delta=0.50, alpha=0.05, beta=0.20
    )


def test_zero_effect_recovered_by_block_methods() -> None:
    """Gate: known zero effect recovered — coverage across experiments."""
    covered = 0
    runs = 40
    for experiment in range(runs):
        blocks = _block_means(_synthetic_walls(experiment_index=11 + experiment, effect=0.0))
        _, low, high = bootstrap_blocks(
            blocks, stream=RandomStream(bytes([0x20 + experiment % 16]) * 32), resamples=300
        )
        covered += ci_covers((low, high), 0.0)
    assert covered / runs >= 0.90, f"zero-effect coverage {covered / runs:.2f} too low"


def test_nonzero_effect_recovered_and_truth_covered() -> None:
    """Gate: known nonzero effect recovered with the truth inside the CI."""
    excludes_zero = 0
    covers_truth = 0
    runs = 24
    for experiment in range(runs):
        blocks = _block_means(_synthetic_walls(experiment_index=500 + experiment, effect=1.1))
        _, low, high = bootstrap_blocks(
            blocks, stream=RandomStream(bytes([0x40 + experiment % 16]) * 32), resamples=300
        )
        excludes_zero += not ci_covers((low, high), 0.0)
        covers_truth += ci_covers((low, high), 1.1)
    assert excludes_zero / runs >= 0.92, "nonzero effect must separate from 0"
    assert covers_truth / runs >= 0.90, "interval must retain the true effect"


def test_sign_flip_null_is_centered() -> None:
    blocks = _block_means(_synthetic_walls(experiment_index=31, effect=0.0))
    _, low, high = sign_flip_interval(blocks, stream=RandomStream(b"\x05" * 32), resamples=500)
    assert low < 0 < high


def test_negative_naive_per_game_resampling_fails_gate() -> None:
    """NEGATIVE gate: per-game independence produces wrong coverage.

    With wall effects dominating game noise, the effective sample size is the
    number of WALLS; pretending games are independent shrinks intervals far
    below their honest width. The naive method must MISS the true (zero)
    contrast far too often, while the whole-block bootstrap stays nominal on
    exactly the same underlying data.
    """
    naive_misses = 0
    block_misses = 0
    for experiment in range(EXPERIMENTS):
        walls = _synthetic_walls(experiment_index=experiment, effect=0.0)
        games = [value for wall in walls for value in wall]
        count = len(games)
        naive_stream = RandomStream(bytes([0xA0 + experiment % 16]) * 32)
        stats = sorted(
            fmean(games[naive_stream.random_below(count)] for _ in range(count))
            for _ in range(RESAMPLES)
        )
        naive_low = stats[math.floor(0.025 * RESAMPLES)]
        naive_high = stats[math.ceil(0.975 * RESAMPLES) - 1]
        if not ci_covers((naive_low, naive_high), 0.0):
            naive_misses += 1
        _, boot_low, boot_high = bootstrap_blocks(
            _block_means(walls), stream=RandomStream(bytes([0x30 + experiment % 16]) * 32)
        )
        if not ci_covers((boot_low, boot_high), 0.0):
            block_misses += 1
    naive_coverage = 1.0 - naive_misses / EXPERIMENTS
    block_coverage = 1.0 - block_misses / EXPERIMENTS
    # The naive method must demonstrably FAIL the nominal-95% coverage gate...
    assert naive_coverage < 0.80, f"naive per-game coverage {naive_coverage:.3f} unexpectedly good"
    # ...while the mandated whole-block bootstrap stays within tolerance.
    assert block_coverage >= 0.88, f"block bootstrap coverage {block_coverage:.3f}"


def test_hedged_cs_time_uniform_on_zero_effect() -> None:
    """Any-time validity: the sequence covers truth at EVERY declared peek."""
    violations = 0
    final_misses = 0
    peeks = (5, 10, 15, WALLS)
    for experiment in range(100):
        blocks = _block_means(_synthetic_walls(experiment_index=200 + experiment, effect=0.0))
        path = hedged_cs_path(blocks, alpha=0.10, bounds=(-4.0, 4.0), peek_times=peeks)
        if any(not ci_covers(interval, 0.0) for interval in path):
            violations += 1
        if not ci_covers(path[-1], 0.0):
            final_misses += 1
    assert violations <= 18, f"time-uniform violation rate {violations / 100:.2f} too high"
    assert final_misses <= 12, f"final-time miss rate {final_misses / 100:.2f} too high"


def test_hedged_cs_power_against_shift() -> None:
    """With enough shifted blocks the CS must exclude the null contrast."""
    blocks: list[float] = []
    for experiment in range(6):
        blocks.extend(_block_means(_synthetic_walls(experiment_index=77 + experiment, effect=2.2)))
    low, high = hedged_confidence_sequence(blocks, alpha=0.05, bounds=(-4.0, 4.0))
    assert not ci_covers((low, high), 0.0)
    assert ci_covers((low, high), fmean(blocks))


def test_sequential_design_guard() -> None:
    sequential_design_guard(design="fixed_n", declared_peeks=[25])
    with pytest.raises(ContractError, match="forbids intermediate peeks"):
        sequential_design_guard(design="fixed_n", declared_peeks=[10, 25])
    sequential_design_guard(design="time_uniform_cs", declared_peeks=[1, 5, 25])
    with pytest.raises(ContractError, match="strictly increasing"):
        sequential_design_guard(design="time_uniform_cs", declared_peeks=[5, 5])
    with pytest.raises(ContractError, match="unknown design"):
        sequential_design_guard(design="always_peek", declared_peeks=[1])  # type: ignore[arg-type]


def test_cluster_bootstrap_groups_by_player_never_decision() -> None:
    records = [
        {"player_id": "p1", "game_id": "g1", "value": 0.10},
        {"player_id": "p1", "game_id": "g2", "value": 0.30},
        {"player_id": "p2", "game_id": "g3", "value": 0.50},
        {"player_id": "p2", "game_id": "g4", "value": 0.70},
    ]
    estimate, low, high = cluster_bootstrap(
        records,
        grouping="player",
        value_of=lambda record: float(record["value"]),  # type: ignore[arg-type]
        stream=RandomStream(b"\x06" * 32),
        resamples=200,
    )
    assert estimate == pytest.approx(0.40)
    assert low <= high
    missing = [{"game_id": "g1", "value": 1.0}]
    with pytest.raises(ContractError, match="player_id"):
        cluster_bootstrap(
            missing,  # type: ignore[arg-type]
            grouping="player",
            value_of=lambda record: float(record["value"]),  # type: ignore[arg-type]
            stream=RandomStream(b"\x06" * 32),
        )


def test_placement_block_contrast_collapses_sorted() -> None:
    blocks = (
        WallBlock(wall_id="w1", game_ids=("w1-g0", "w1-g1"), contrasts=(0.0, 2.0)),
        WallBlock(wall_id="w0", game_ids=("w0-g0", "w0-g1"), contrasts=(3.0, 1.0)),
    )
    assert placement_block_contrast(blocks) == (2.0, 1.0)
    with pytest.raises(ContractError, match="at least one wall block"):
        placement_block_contrast(())


def test_selection_config_defaults_documented() -> None:
    """Document frozen defaults: fixed_n at N=30, whole-wall-block only, gate only."""
    config = SelectionConfig()
    assert config.N == 30
    assert config.pilot_s == 1.5
    assert config.delta == 0.5
    assert config.alpha == 0.05
    assert config.beta == 0.2
    assert config.design == "fixed_n"
    assert config.declared_peeks == (30,)
    assert config.margin == 0.0
    assert config.uncertainty_unit == "wall_block"
    assert config.resamples == 2000
    assert config.seed == 0
    config.validate()
    selection_gate_check(config, 30)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.N = 31


def test_selection_config_validation() -> None:
    """Frozen-config validation: positive N/s/delta, (0,1) alpha/beta, peek shape."""
    base = SelectionConfig()
    with pytest.raises(ContractError, match="N must be"):
        dataclasses.replace(base, N=0)
    with pytest.raises(ContractError, match="N must be"):
        dataclasses.replace(base, N=-3)
    with pytest.raises(ContractError, match="pilot_s must be"):
        dataclasses.replace(base, pilot_s=0.0)
    with pytest.raises(ContractError, match="delta must be"):
        dataclasses.replace(base, delta=-1.0)
    with pytest.raises(ContractError, match="alpha must lie"):
        dataclasses.replace(base, alpha=0.0)
    with pytest.raises(ContractError, match="alpha must lie"):
        dataclasses.replace(base, alpha=1.0)
    with pytest.raises(ContractError, match="beta must lie"):
        dataclasses.replace(base, beta=1.5)
    with pytest.raises(ContractError, match="forbids intermediate peeks"):
        dataclasses.replace(base, declared_peeks=(10, 30))
    with pytest.raises(ContractError, match="strictly increasing"):
        dataclasses.replace(base, design="time_uniform_cs", declared_peeks=(5, 5, 25))
    with pytest.raises(ContractError, match="strictly increasing"):
        dataclasses.replace(base, design="time_uniform_cs", declared_peeks=())
    with pytest.raises(ContractError, match="positive ints"):
        dataclasses.replace(base, design="time_uniform_cs", declared_peeks=(0, 15))
    with pytest.raises(ContractError, match="unknown design"):
        dataclasses.replace(base, design="always_peek")  # type: ignore[arg-type]


def test_selection_gate_fixed_n_blocks_second_peek() -> None:
    """Gate only: exactly one look at N; early and beyond-N looks raise."""
    config = SelectionConfig(
        N=25,
        pilot_s=1.5,
        delta=0.5,
        alpha=0.05,
        beta=0.2,
        design="fixed_n",
        declared_peeks=(25,),
    )
    selection_gate_check(config, 25)
    with pytest.raises(ContractError, match="extra look"):
        selection_gate_check(config, 10)
    with pytest.raises(ContractError, match="extra look"):
        selection_gate_check(config, 26)
    with pytest.raises(ContractError, match="forbids intermediate peeks"):
        SelectionConfig(
            N=25,
            pilot_s=1.5,
            delta=0.5,
            alpha=0.05,
            beta=0.2,
            design="fixed_n",
            declared_peeks=(10, 25),
        )


def test_selection_gate_cs_schedule_slicing() -> None:
    """Gate only: CS passes at each declared peek, raises between peeks."""
    config = SelectionConfig(
        N=25,
        pilot_s=1.5,
        delta=0.5,
        alpha=0.05,
        beta=0.2,
        design="time_uniform_cs",
        declared_peeks=(5, 15, 25),
    )
    for peek in (5, 15, 25):
        selection_gate_check(config, peek)
    for between in (1, 6, 10, 16, 24):
        with pytest.raises(ContractError, match="extra look"):
            selection_gate_check(config, between)
    with pytest.raises(ContractError, match="strictly increasing"):
        SelectionConfig(
            N=25,
            pilot_s=1.5,
            delta=0.5,
            alpha=0.05,
            beta=0.2,
            design="time_uniform_cs",
            declared_peeks=(5, 5, 25),
        )


def _honest_telemetry_row(**overrides: object):  # type: ignore[no-untyped-def]
    """Valid cuda_eager telemetry row for score_selection tests (append-only helper)."""
    from hydra2.eval.telemetry import make_resource_telemetry

    base: dict[str, object] = {
        "mode": "cuda_eager",
        "wall_id": "w-0",
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


def test_score_selection_excludes_telemetry_invalid_wall() -> None:
    """Telemetry-invalid wall never enters the score: metric covers VALID only."""
    from hydra2.eval.statistics import score_selection

    blocks = (
        WallBlock(wall_id="w-0", game_ids=("g-0",), contrasts=(0.5,)),
        WallBlock(wall_id="w-1", game_ids=("g-1",), contrasts=(1.5,)),
        WallBlock(wall_id="w-2", game_ids=("g-2",), contrasts=(999.0,)),
    )
    telemetry = {
        "g-0": _honest_telemetry_row(wall_id="w-0"),
        "g-1": _honest_telemetry_row(wall_id="w-1"),
    }
    config = SelectionConfig(
        N=2,
        pilot_s=1.5,
        delta=0.5,
        alpha=0.05,
        beta=0.2,
        design="fixed_n",
        declared_peeks=(2,),
        resamples=200,
        seed=7,
    )
    metric, (est, low, high), excluded = score_selection(blocks, telemetry, config, 2)
    assert metric == pytest.approx(1.0)
    assert est == pytest.approx(1.0)
    assert low <= est <= high
    assert len(excluded) == 1
    assert excluded[0].wall_id == "w-2"
    naive = fmean((0.5, 1.5, 999.0))
    assert metric != pytest.approx(naive)


def test_score_selection_guard_blocks_undeclared_peek() -> None:
    """Guard wiring: scoring at an undeclared peek raises even when valid."""
    from hydra2.eval.statistics import score_selection

    blocks = (
        WallBlock(wall_id="w-0", game_ids=("g-0",), contrasts=(0.5,)),
        WallBlock(wall_id="w-1", game_ids=("g-1",), contrasts=(1.5,)),
    )
    telemetry = {
        "g-0": _honest_telemetry_row(wall_id="w-0"),
        "g-1": _honest_telemetry_row(wall_id="w-1"),
    }
    config = SelectionConfig(
        N=3,
        pilot_s=1.5,
        delta=0.5,
        alpha=0.05,
        beta=0.2,
        design="fixed_n",
        declared_peeks=(3,),
        resamples=200,
        seed=11,
    )
    with pytest.raises(ContractError, match="extra look"):
        score_selection(blocks, telemetry, config, 2)


def test_promoted_with_failed_gate_raises() -> None:
    """Promotion binding: failed gates can never promote; rejected stays loose."""
    from hydra2.eval.promotion import make_promotion_record

    digest = "sha256:" + "ab" * 32
    good: dict[str, object] = {
        "candidate_spec_hash": digest,
        "utility_manifest_hash": digest,
        "comparator_spec_hashes": (digest,),
        "case_manifest_hash": digest,
        "result_table_hash": digest,
        "resource_view": "cuda_eager",
        "uncertainty_unit": "wall_block",
        "pass_inequality": "mean_contrast > 0",
        "observed_estimate": 0.42,
        "confidence_bounds": (0.05, 0.79),
        "gates": {"seat_balance": "passed", "coverage": "failed"},
        "disposition": "promoted",
        "schedule_hash": digest,
    }
    with pytest.raises(ContractError, match="all gates passed"):
        make_promotion_record(**good)
    loose = dict(good)
    loose["disposition"] = "rejected"
    assert make_promotion_record(**loose).disposition == "rejected"


def test_promoted_without_schedule_raises() -> None:
    """Promotion binding: promoted needs a schedule; rejected stays loose."""
    from hydra2.eval.promotion import make_promotion_record

    digest = "sha256:" + "ab" * 32
    good: dict[str, object] = {
        "candidate_spec_hash": digest,
        "utility_manifest_hash": digest,
        "comparator_spec_hashes": (digest,),
        "case_manifest_hash": digest,
        "result_table_hash": digest,
        "resource_view": "cuda_eager",
        "uncertainty_unit": "wall_block",
        "pass_inequality": "mean_contrast > 0",
        "observed_estimate": 0.42,
        "confidence_bounds": (0.05, 0.79),
        "gates": {"seat_balance": "passed", "coverage": "passed"},
        "disposition": "promoted",
    }
    with pytest.raises(ContractError, match="schedule_hash"):
        make_promotion_record(**good)
    loose = dict(good)
    loose["disposition"] = "rejected"
    assert make_promotion_record(**loose).disposition == "rejected"


def test_score_selection_exclusions_carried_to_promotion_record() -> None:
    """Exclusions-carried: score return preserves blocks for the record."""
    from hydra2.eval.promotion import make_promotion_record, promotion_digest
    from hydra2.eval.schedule import build_match_schedule, schedule_commitment_hash
    from hydra2.eval.statistics import score_selection

    blocks = (
        WallBlock(wall_id="w-0", game_ids=("g-0",), contrasts=(0.5,)),
        WallBlock(wall_id="w-1", game_ids=("g-1",), contrasts=(1.5,)),
        WallBlock(wall_id="w-2", game_ids=("g-2",), contrasts=(999.0,)),
    )
    telemetry = {
        "g-0": _honest_telemetry_row(wall_id="w-0"),
        "g-1": _honest_telemetry_row(wall_id="w-1"),
    }
    config = SelectionConfig(
        N=2,
        pilot_s=1.5,
        delta=0.5,
        alpha=0.05,
        beta=0.2,
        design="fixed_n",
        declared_peeks=(2,),
        resamples=200,
        seed=7,
    )
    metric, (est, low, high), excluded = score_selection(blocks, telemetry, config, 2)
    assert len(excluded) == 1
    assert excluded[0].wall_id == "w-2"
    schedule = build_match_schedule(
        wall_ids=("w-0", "w-1", "w-2"),
        labels=("candidate-a", "partner-a", "baseline-b", "field-c"),
        rules_hash="sha256:" + "ab" * 32,
        master_seed=bytes(range(48, 80)),
        experiment_id="exp-wp03b",
        split_id="split-wp03b",
    )
    commitment = str(schedule_commitment_hash(schedule))
    digest = "sha256:" + "ab" * 32
    record = make_promotion_record(
        candidate_spec_hash=digest,
        utility_manifest_hash=digest,
        comparator_spec_hashes=(digest,),
        case_manifest_hash=digest,
        result_table_hash=digest,
        resource_view="cuda_eager",
        uncertainty_unit="wall_block",
        pass_inequality="mean_contrast > 0",
        observed_estimate=metric,
        confidence_bounds=(low, high),
        gates={"seat_balance": "passed", "coverage": "passed"},
        disposition="promoted",
        schedule_hash=commitment,
        excluded_blocks=excluded,
    )
    assert record.excluded_blocks == excluded
    assert record.schedule_hash == commitment
    assert est == pytest.approx(metric)
    bare = make_promotion_record(
        candidate_spec_hash=digest,
        utility_manifest_hash=digest,
        comparator_spec_hashes=(digest,),
        case_manifest_hash=digest,
        result_table_hash=digest,
        resource_view="cuda_eager",
        uncertainty_unit="wall_block",
        pass_inequality="mean_contrast > 0",
        observed_estimate=metric,
        confidence_bounds=(low, high),
        gates={"seat_balance": "passed", "coverage": "passed"},
        disposition="promoted",
        schedule_hash=commitment,
    )
    assert promotion_digest(record) != promotion_digest(bare)
