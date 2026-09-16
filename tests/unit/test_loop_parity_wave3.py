"""Wave 3 loop parity — oracle pins for counting / envelopes / firewall / scheduler.

Deterministic oracle goldens for the supervised-loop slivers: accumulation
divisor order (the scale-then-sum byte-identity trap), single-sync window
means (input order preserved), quantile edges and telemetry summaries, the
no-privileged firewall vocabulary (top-level and nested), the warmup/decay
scheduler spans stepped in loop order, and the supervised-loss envelope.
Every stochastic input is drawn from :mod:`hydra2.contracts.randomness`
semantic streams; no wall-clock, no ``random`` module, no
``torch.manual_seed``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from hydra2.contracts.common import ContractError
from hydra2.contracts.randomness import (
    RandomStream,
    make_random_stream_key,
    semantic_seed,
)
from hydra2.training._rc_sections import SchedulerConfig
from hydra2.training.loop_batch import (
    MicrobatchTelemetry,
    _quantile_sorted,
    _validate_batch_no_privileged,
    _window_means,
    summarize_telemetry,
)
from hydra2.training.loop_state import FORBIDDEN_BATCH_KEYS
from hydra2.training.objectives import compute_hot_scalars, masked_cross_entropy
from hydra2.training.run_config import RunConfig
from hydra2.training.stream_build import _build_scheduler

pytestmark = pytest.mark.contract_package("WP-05B")

_MASTER = b"wave3-loop-parity-v1"
_EXPERIMENT = "wave3-loop-parity"
_SPLIT = "oracle"


def _stream(replicate_id: int = 0) -> RandomStream:
    key = make_random_stream_key(
        purpose="training_shuffle",
        experiment_id=_EXPERIMENT,
        split_id=_SPLIT,
        replicate_id=replicate_id,
        attempt_id=0,
    )
    return RandomStream(semantic_seed(_MASTER, key=key))


def _scheduler_trajectory(
    name: str, warmup: int, max_updates: int, final_factor: float
) -> list[float]:
    config = dataclasses.replace(
        RunConfig(),
        scheduler=SchedulerConfig(
            name=name,
            warmup_updates=warmup,
            parameters={},
            final_factor=final_factor,
            warmup_start_factor=0.01,
        ),
        loop=dataclasses.replace(RunConfig().loop, max_updates=max_updates),
    )
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = _build_scheduler(config, optimizer)
    trajectory = [round(float(optimizer.param_groups[0]["lr"]), 10)]
    for _ in range(max_updates):  # loop order: optimizer step, then scheduler
        optimizer.step()
        scheduler.step()
        trajectory.append(round(float(optimizer.param_groups[0]["lr"]), 10))
    return trajectory


# ---------------------------------------------------------------------------
# Accumulation counting (divisor-order trap)
# ---------------------------------------------------------------------------


def test_accumulation_divisor_order_golden() -> None:
    """The loop scales ``total / accumulation_steps``; the divisor MUST NOT be
    folded into the per-microbatch accumulation (that shifts rounding)."""
    total = torch.tensor(6.0)
    assert float(total / 1) == 6.0
    assert float(total / 2) == 3.0
    assert float(total / 4) == 1.5


def test_accumulation_window_sum_equals_minibatch_mean() -> None:
    rng = _stream()
    microbatch = [rng.random_float() * 4.0 - 2.0 for _ in range(4)]
    accumulation_steps = 4
    scaled = [(value / accumulation_steps) for value in microbatch]
    assert math.fsum(scaled) == pytest.approx(math.fsum(microbatch) / accumulation_steps)
    assert math.fsum(scaled) == pytest.approx(
        float(torch.tensor(microbatch).sum() / accumulation_steps)
    )


# ---------------------------------------------------------------------------
# Window means / quantiles / telemetry summaries
# ---------------------------------------------------------------------------


def test_window_means_order_and_empty_golden() -> None:
    assert _window_means(
        [torch.tensor(1.0), torch.tensor(3.0)],
        [],
        [torch.tensor(2.0)],
    ) == [2.0, 0.0, 2.0]  # input order preserved; empty reads 0.0


def test_quantile_edges_and_interpolation_goldens() -> None:
    assert _quantile_sorted([2.0], 0.5) == 2.0
    assert _quantile_sorted([1.0, 2.0, 3.0], 0.0) == 1.0
    assert _quantile_sorted([1.0, 2.0, 3.0], 1.0) == 3.0
    assert _quantile_sorted([1.0, 2.0, 3.0, 4.0], 0.50) == 2.5
    assert _quantile_sorted([1.0, 2.0, 3.0, 4.0], 0.99) == 3.9699999999999998
    with pytest.raises(ContractError, match="at least one value"):
        _quantile_sorted([], 0.5)
    with pytest.raises(ContractError, match="q must be in"):
        _quantile_sorted([1.0], 1.5)


def test_telemetry_summary_goldens() -> None:
    records = [
        MicrobatchTelemetry(
            microstep=i,
            global_update=0,
            queue_wait_ms=1.0 * i,
            fetch_decode_ms=2.0 * i,
            h2d_ms=0.5 * i,
            compute_ms=10.0,
        )
        for i in range(1, 5)
    ]
    assert summarize_telemetry(records) == {
        "queue_wait_ms_p50": 2.5,
        "queue_wait_ms_p99": 3.9699999999999998,
        "fetch_decode_ms_p50": 5.0,
        "fetch_decode_ms_p99": 7.9399999999999995,
        "h2d_ms_p50": 1.25,
        "h2d_ms_p99": 1.9849999999999999,
        "compute_ms_p50": 10.0,
        "compute_ms_p99": 10.0,
        "forward_ms_p50": 0.0,
        "forward_ms_p99": 0.0,
        "loss_ms_p50": 0.0,
        "loss_ms_p99": 0.0,
        "backward_ms_p50": 0.0,
        "backward_ms_p99": 0.0,
    }
    assert summarize_telemetry([]) == {}


# ---------------------------------------------------------------------------
# Firewall vocabulary
# ---------------------------------------------------------------------------


def test_firewall_vocabulary_pinned() -> None:
    assert sorted(FORBIDDEN_BATCH_KEYS) == [
        "dead_wall",
        "full_world",
        "hidden",
        "hidden_tiles",
        "opponent_hand",
        "privileged",
        "privileged_label",
        "wall",
        "wall_remaining",
    ]


def test_firewall_rejects_top_level_and_nested_keys() -> None:
    _validate_batch_no_privileged({"features": 1})  # clean batch passes
    for key in sorted(FORBIDDEN_BATCH_KEYS):
        with pytest.raises(ContractError, match="privileged"):
            _validate_batch_no_privileged({key: 1})
    with pytest.raises(ContractError, match="privileged sub-key 'wall'"):
        _validate_batch_no_privileged({"event_targets": {"wall": 1}})


# ---------------------------------------------------------------------------
# Scheduler spans (stepped in loop order)
# ---------------------------------------------------------------------------


def test_scheduler_cosine_span_golden() -> None:
    assert _scheduler_trajectory("cosine", 2, 6, 0.0) == [
        0.001,
        0.0505,
        0.1,
        0.0853553391,
        0.05,
        0.0146446609,
        0.0,
    ]


def test_scheduler_linear_span_golden() -> None:
    assert _scheduler_trajectory("linear", 2, 6, 0.5) == [
        0.001,
        0.0505,
        0.1,
        0.0875,
        0.075,
        0.0625,
        0.05,
    ]


def test_scheduler_constant_span_golden() -> None:
    assert _scheduler_trajectory("constant", 0, 3, 0.0) == [0.1, 0.1, 0.1, 0.1]


# ---------------------------------------------------------------------------
# Supervised-loss envelope
# ---------------------------------------------------------------------------


def test_masked_ce_envelope_goldens() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0, 0.0]])
    targets = torch.tensor([0])
    legal = torch.tensor([[True, True, False, True]])
    assert float(masked_cross_entropy(logits, targets, legal)) == 0.3063558042049408
    assert (
        float(masked_cross_entropy(logits, targets, legal, label_smoothing=0.1))
        == 0.42302244901657104
    )


def test_hot_scalars_envelope_golden() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    targets = torch.tensor([0, 2])
    legal = torch.tensor([[True, True, False, True], [True, True, True, True]])
    assert compute_hot_scalars(logits, targets, legal) == {
        "masked_nll": 0.8463250994682312,
        "top1": 1.0,
    }
