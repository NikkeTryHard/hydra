"""Wave-3 placement-credit fixtures: orasu flip, telescope, paired wall-block.

Lean-first Wave-1 designs, verbatim. Deterministic integer/Fraction
arithmetic plus whole-block aggregation. No RNG, no engine.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from statistics import fmean, stdev

import pytest

from hydra2.contracts.randomness import RandomStream
from hydra2.eval.blocks import WallBlock, aggregate_blocks
from hydra2.eval.statistics import bootstrap_blocks
from hydra2.eval.telemetry import ResourceTelemetry, make_resource_telemetry


def _telemetry(wall_id: str) -> ResourceTelemetry:
    return make_resource_telemetry(
        mode="reference_eager_cpu",
        wall_id=wall_id,
        case_id="case-pc",
        candidate_spec_hash="sha256:" + "11" * 32,
        hardware_hash="sha256:" + "22" * 32,
        environment_hash="sha256:" + "33" * 32,
        cold_start=False,
        synchronized_elapsed_ms=12.5,
        model_calls=4,
        exact_transitions=4,
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


def test_orasu_flip_score_place_argmax_split() -> None:
    """F1: gamble beats safe on score yet loses on placement — argmax splits."""
    e_score = {"G": Fraction(1, 4) * 38000 + Fraction(3, 4) * 24000, "S": Fraction(27000)}
    assert e_score["G"] == 27500
    assert e_score["G"] > e_score["S"]
    e_place = {"G": Fraction(1, 4) * 3 + Fraction(3, 4) * (-3), "S": Fraction(-1)}
    assert e_place["G"] == Fraction(-3, 2)
    assert e_place["G"] < e_place["S"]
    assert max(e_score, key=lambda arm: e_score[arm]) == "G"
    assert max(e_place, key=lambda arm: e_place[arm]) == "S"


def test_telescope_credits_sum_to_endpoints() -> None:
    """F2: stepwise credits telescope; negatives survive, sum is not the endpoint."""
    phi = [27000, 28500, 26100, 30400]
    round_credits = [after - before for before, after in pairwise(phi)]
    assert round_credits == [1500, -2400, 4300]
    assert sum(round_credits) == phi[-1] - phi[0] == 3400
    assert sum(round_credits) != phi[-1]
    assert round_credits[1] < 0


def test_paired_wall_block_deltas() -> None:
    """F3: same wall, both arms — block deltas with n=2 walls for the CI."""
    blocks = (
        WallBlock(wall_id="w0", game_ids=("w0-g0", "w0-g1"), contrasts=(2.0, 0.0)),
        WallBlock(wall_id="w1", game_ids=("w1-g0", "w1-g1"), contrasts=(-2.0, 0.0)),
    )
    telemetry = {
        game: _telemetry(wall)
        for wall, games in (("w0", ("w0-g0", "w0-g1")), ("w1", ("w1-g0", "w1-g1")))
        for game in games
    }
    result = aggregate_blocks(blocks, telemetry_by_game=telemetry)
    assert len(result.excluded) == 0
    deltas = [value for _, value in result.valid]
    assert deltas == [pytest.approx(1.0), pytest.approx(-1.0)]
    mean_a = fmean([fmean([3, 1]), fmean([0, 2])])
    mean_b = fmean([fmean([1, 1]), fmean([2, 2])])
    assert fmean(deltas) == pytest.approx(mean_a - mean_b)
    assert mean_a - mean_b == 0.0
    # The CI consumes 2 walls, not 4 games.
    assert len(deltas) == 2
    est, low, high = bootstrap_blocks(deltas, stream=RandomStream(b"\x31" * 32))
    assert est == pytest.approx(0.0)
    assert low <= 0.0 <= high
    games = [2.0, 0.0, -2.0, 0.0]
    naive_est, _, _ = bootstrap_blocks(games, stream=RandomStream(b"\x31" * 32))
    assert naive_est == pytest.approx(est)
    # WP03b lesson: pretending the 4 games are independent shrinks the bar
    # (bootstrap quantiles are degenerate at n=2, so the scale comparison
    # carries the lesson: 0.82 < 1.0).
    assert stdev(games) / len(games) ** 0.5 < stdev(deltas) / len(deltas) ** 0.5
