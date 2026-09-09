"""Attention-kernel dispatch harness — perf arm (a), dispatch-pin, no math change.

Fixed corpus (seed 0) over buckets T=32/64/128/256 (B=8, exact model
H=4/Dh=32, bool [B,1,1,T] mask, eval dropout 0.0): dtype (fp32 eager vs
bf16 autocast) x dispatch (default vs pinned flash->efficient->math vs
math-exclusive). Reports synchronized profiler-disabled timings (cold
first-iteration vs warm mean), peak memory via reset-peak around the
timed section, per-backend health on real-shaped inputs, the
determinism-flag record, and allclose parity of every mode against the
default path. Math-exclusive is measured only to price the fallback
cliff the pin insures against — it is not a candidate arm.

Tolerances follow existing precedent, not new invention: fp32 modes use
the cross-shape agreement rule (atol=1e-6, rtol=1e-5); bf16 modes use
the WP-13 forward-parity convention (atol=rtol=1e-2). No parity-harness
tolerance is touched here.
"""

from __future__ import annotations

import contextlib
import time

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from hydra2.contracts.event import EventEnvelope, EventPayload
from hydra2.contracts.observation import make_actor_observation
from hydra2.models.encoder import ActorTensorBatch, encode_observations
from hydra2.models.model import Hydra2BaselineModel
from hydra2.models.schema import BASELINE_ACTION_COUNT
from hydra2.models.sdpa_dispatch import (
    backend_health,
    describe_sdpa_runtime,
    documented_a100_expectation,
    pinned_sdpa_kernel,
)

pytestmark = pytest.mark.gpu

SEED = 0
BATCH = 8
N_HEADS = 4
HEAD_DIM = 32
WARMUP_ITERS = 5
TIMED_ITERS = 20
FP_ATOL = 1e-6
FP_RTOL = 1e-5
BF16_ATOL = 1e-2
BF16_RTOL = 1e-2


def _make_turn_advance(sequence: int, actor: int = 0) -> EventEnvelope:
    payload = EventPayload(
        kind="turn_advance",
        actor=actor,
        tile=None,
        action_id=None,
        source_seat=None,
        consumed_tiles=(),
        offered_action_ids=(),
        accepted_action_ids=(),
        round_index=None,
        scores=None,
        reason=None,
    )
    return EventEnvelope(
        game_id="g-dispatch",
        sequence=sequence,
        kind="turn_advance",
        actor=actor,
        visibility="public",
        visible_to=(0, 1, 2, 3),
        payload=payload,
        public_delta=(),
        rules_hash="sha256:" + "ab" * 32,
        schema_hash="sha256:" + "ac" * 32,
    )


def _history_of_length(n: int) -> tuple[EventEnvelope, ...]:
    return tuple(_make_turn_advance(i + 1, actor=i % 4) for i in range(n))


def _make_observation(*, actor: int, history: tuple[EventEnvelope, ...]):
    legal = [False] * BASELINE_ACTION_COUNT
    legal[0] = True
    legal[10] = True
    seq = int(history[-1].sequence) if history else 1
    return make_actor_observation(
        game_id="g-dispatch",
        decision_id=f"d-{actor}-{seq}-{len(history)}",
        sequence=seq,
        actor=actor,
        rules_id="tenhou_4p_hanchan_v1",
        rules_hash="sha256:" + "ab" * 32,
        action_table_hash="sha256:" + "ac" * 32,
        event_schema_hash="sha256:" + "ad" * 32,
        observation_schema_hash="sha256:" + "ae" * 32,
        packet_boundary_hash="sha256:" + "af" * 32,
        round_index=0,
        round_wind=27,
        hand_number=0,
        seat_winds=(27, 28, 29, 30),
        honba=0,
        riichi_sticks=0,
        dealer=0,
        scores=(25000, 25000, 25000, 25000),
        turn_actor=0,
        phase="draw_decision",
        live_wall_tiles_remaining=70,
        kan_count=0,
        ippatsu_active=(False, False, False, False),
        actor_furiten="none",
        actor_can_tsumo=True,
        actor_can_riichi=False,
        pending_declaration_discard=None,
        concealed_hand=(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48),
        own_drawn_tile=None,
        visible_discards=((), (), (), ()),
        visible_melds=((), (), (), ()),
        riichi_states=("none", "none", "none", "none"),
        dora_indicators=(-1, -1, -1, -1, -1),
        visible_history=tuple(history),
        legal_mask=tuple(legal),
    )


def _fixed_cpu_batches() -> list[tuple[str, ActorTensorBatch]]:
    """Fixed corpus: full-length rows per bucket + a partial/empty T=32 batch."""
    plans: list[tuple[str, list[int]]] = [
        ("T32-partial-empty", [10, 10, 10, 10, 10, 10, 10, 0]),
        ("T64-full", [64] * BATCH),
        ("T128-full", [128] * BATCH),
        ("T256-full", [256] * BATCH),
    ]
    batches = []
    for label, lengths in plans:
        observations = [
            _make_observation(actor=i % 4, history=_history_of_length(n))
            for i, n in enumerate(lengths)
        ]
        batches.append((label, encode_observations(observations)))
    return batches


def _batch_to_device(batch: ActorTensorBatch, device: torch.device) -> ActorTensorBatch:
    return ActorTensorBatch(
        features={
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in dict(batch.features).items()
        },
        history_mask=batch.history_mask.to(device),
        legal_mask=batch.legal_mask.to(device),
        observation_hashes=batch.observation_hashes,
        actor_seats=batch.actor_seats.to(device),
    )


def _mode_context(mode: str):
    if mode == "pinned":
        return pinned_sdpa_kernel()
    if mode == "math":
        return sdpa_kernel([SDPBackend.MATH])
    return contextlib.nullcontext()


def _timed_forward(
    model: Hydra2BaselineModel, batch: ActorTensorBatch, dtype: torch.dtype, mode: str
) -> tuple[torch.Tensor, float, float, float]:
    """One (logits, cold_ms, warm_mean_ms, peak_MB) measurement, fully synced."""
    torch.cuda.synchronize()
    with _mode_context(mode), torch.no_grad():
        start = time.perf_counter()
        if dtype == torch.bfloat16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cold = model.evaluate(batch).policy_logits.float().cpu()
        else:
            cold = model.evaluate(batch).policy_logits.float().cpu()
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter() - start) * 1e3
        for _ in range(WARMUP_ITERS):
            if dtype == torch.bfloat16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model.evaluate(batch)
            else:
                model.evaluate(batch)
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(TIMED_ITERS):
            start = time.perf_counter()
            if dtype == torch.bfloat16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model.evaluate(batch)
            else:
                model.evaluate(batch)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return cold, cold_ms, sum(samples) / len(samples), peak_mb


@pytest.fixture(scope="module")
def cuda_batches(require_cuda):
    device = require_cuda
    assert device is not None
    return [(label, _batch_to_device(batch, device)) for label, batch in _fixed_cpu_batches()]


@pytest.fixture(scope="module")
def seeded_model(require_cuda):
    torch.manual_seed(SEED)
    model = Hydra2BaselineModel(dropout=0.0)
    model.eval()
    return model.to(require_cuda)


def _health_rows(cuda_batches) -> list[tuple[str, str, dict[str, bool]]]:
    rows = []
    for label, batch in cuda_batches:
        seq_len = batch.history_mask.shape[1]
        mask = torch.ones(BATCH, 1, 1, seq_len, dtype=torch.bool, device="cuda")
        for dtype_name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            query = torch.zeros(BATCH, N_HEADS, seq_len, HEAD_DIM, dtype=dtype, device="cuda")
            report = backend_health(query, query, query, mask, dropout_p=0.0)
            rows.append((label, dtype_name, report))
    return rows


def test_sdpa_backend_health_premise(cuda_batches) -> None:
    """Gate the arm's premise: efficient eligible on every real-shaped case."""
    lines = ["dispatch health (exact QKV shapes, bool mask, dropout 0.0):"]
    for label, dtype_name, report in _health_rows(cuda_batches):
        lines.append(f"  {label:18} {dtype_name:4} {report}")
        assert report["efficient"], (
            f"mem-efficient gated out on {label}/{dtype_name}: arm premise broken"
        )
    # Context mechanics: the pin keeps efficient enabled, math-exclusive drops it.
    batch = cuda_batches[0][1]
    seq_len = batch.history_mask.shape[1]
    mask = torch.ones(BATCH, 1, 1, seq_len, dtype=torch.bool, device="cuda")
    query = torch.zeros(BATCH, N_HEADS, seq_len, HEAD_DIM, device="cuda")
    with pinned_sdpa_kernel():
        assert torch.backends.cuda.mem_efficient_sdp_enabled()
        assert backend_health(query, query, query, mask)["efficient"]
    assert torch.backends.cuda.mem_efficient_sdp_enabled(), "pin must restore flags on exit"
    with sdpa_kernel([SDPBackend.MATH]):
        assert not backend_health(query, query, query, mask)["efficient"]
    print("\n".join(lines))


def test_dispatch_parity_and_numbers(cuda_batches, seeded_model) -> None:
    """Parity of every mode vs default + the measured numbers table."""
    runtime = describe_sdpa_runtime()
    dtypes = (
        ("fp32", torch.float32, FP_ATOL, FP_RTOL),
        ("bf16", torch.bfloat16, BF16_ATOL, BF16_RTOL),
    )
    modes = ("default", "pinned", "math")
    measured: dict[tuple[str, str, str], tuple[torch.Tensor, float, float, float]] = {}
    for label, batch in cuda_batches:
        for dtype_name, dtype, _, _ in dtypes:
            for mode in modes:
                measured[(dtype_name, label, mode)] = _timed_forward(
                    seeded_model, batch, dtype, mode
                )
    lines = [
        f"runtime: {runtime}",
        f"determinism: algorithms={runtime['deterministic_algorithms']} "
        f"cudnn_deterministic={runtime['cudnn_deterministic']} "
        f"cudnn_benchmark={runtime['cudnn_benchmark']} "
        "(conftest forces deterministic on: cuDNN excluded by pin AND by setting)",
        f"a100: {documented_a100_expectation()}",
        "row: dtype mode bucket cold_ms warm_ms peak_MB maxdiff_vs_default",
    ]
    for label, _batch in cuda_batches:
        for dtype_name, _, atol, rtol in dtypes:
            ref = measured[(dtype_name, label, "default")][0]
            assert torch.isfinite(ref).all(), f"non-finite default logits {dtype_name}/{label}"
            for mode in modes:
                logits, cold_ms, warm_ms, peak_mb = measured[(dtype_name, label, mode)]
                assert torch.isfinite(logits).all(), (
                    f"non-finite logits {dtype_name}/{label}/{mode}"
                )
                torch.testing.assert_close(logits, ref, atol=atol, rtol=rtol, equal_nan=False)
                diff = float((logits - ref).abs().max().item())
                lines.append(
                    f"  {dtype_name:4} {mode:7} {label:18} "
                    f"cold={cold_ms:7.3f}ms warm={warm_ms:7.3f}ms "
                    f"peak={peak_mb:6.1f}MB maxdiff={diff:.2e}"
                )
    # All-padding row (last row of the T32 batch) must stay finite, never NaN-pool.
    empty_row = cuda_batches[0][1].history_mask[-1]
    assert not empty_row.any(), "fixed corpus must keep an all-padding row"
    for dtype_name, _, _, _ in dtypes:
        for mode in modes:
            logits = measured[(dtype_name, "T32-partial-empty", mode)][0]
            assert torch.isfinite(logits[-1]).all(), f"empty-history NaN leak {dtype_name}/{mode}"
    print("\n".join(lines))
