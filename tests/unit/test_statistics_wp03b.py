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

import math
from statistics import NormalDist, fmean

import pytest

from hydra2.contracts.common import ContractError
from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed
from hydra2.eval.statistics import (
    bootstrap_blocks,
    ci_covers,
    cluster_bootstrap,
    fixed_n_samples,
    hedged_confidence_sequence,
    hedged_cs_path,
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
