#!/usr/bin/env python3
"""Kernel benchmark: profiler-timed replay of the production training step.

Replays the exact probe training step (same builders, same compile, same
loss/optimizer calls as ``run_stream_training``) on schema-driven synthetic
batches at all four history buckets, timed by torch.profiler CUDA-kernel
sums — no fetch, no H2D, no logging syncs inside the timed region.

IMPORTS, NOT COPIES (drift-safety): model/optimizer/scheduler builders,
runtime spec derivation, loss kernel, validators, batch schema, and bucket
lengths are all imported from production modules. Anything this file mirrors
by hand (the ~15-line step body) is called out with its production source
line; the fidelity check is ``nsys`` ground truth (see below), not eyeballing.

FIDELITY: kbench per-bucket kernel_ms must agree with nsys-measured
production per-update kernel sums at the same bucket T within 15%. If it
does not, this instrument is broken — fix it, do not tune against it.
kbench.py itself is frozen after baseline: any edit = new segment.

Fairness: fixed generator seed -> identical batches every run on any box;
kernel time is input-value-independent for these dense ops. Loss values are
NOT compared (synthetic values); finiteness is gated fail-closed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from typing import NoReturn

# MUST precede any CUDA context (mirrors tests/conftest.py + harness env).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from hydra2.models.encoder import ActorTensorBatch
from hydra2.models.model import validate_actor_batch
from hydra2.models.schema import (
    _BASELINE_FIELDS,
    BASELINE_ACTION_COUNT,
    HISTORY_BUCKET_LENGTHS,
)
from hydra2.training.adapters import model_output_to_loss_dict
from hydra2.training.objectives import (
    _check_total_finite,
    global_grad_norm_is_finite,
    supervised_loss_kernel,
    validate_supervised_inputs,
)
from hydra2.training.run_config import load_run_config

_SYNTH_SEED = 20260912


def _fail(msg: str) -> NoReturn:
    print(f"KBENCH FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def _build_synth_batch(
    *, b: int, t: int, a: int, gen: torch.Generator, device: torch.device
) -> tuple[ActorTensorBatch, dict[str, torch.Tensor]]:
    """Schema-driven fixed batch: every field from _BASELINE_FIELDS.

    Ranges come from the spec (valid_min/max, padding_value, mask_field);
    scores/hand counters use realistic magnitudes so the finite-loss gate
    is meaningful. history lengths spread over (T-63, T] like bucketed
    production data; legal sets are random k-subsets with the target drawn
    from the legal set (validator requires target-in-legal).
    """
    import typing

    specs = {f.name: f for f in _BASELINE_FIELDS}
    dtype_map = {
        "bool": torch.bool,
        "int32": torch.int32,
        "int64": torch.int64,
        "float32": torch.float32,
    }

    lengths = torch.randint(max(1, t - 63), t + 1, (b,), generator=gen)
    history_mask = torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)
    legal_k = torch.randint(20, 49, (b,), generator=gen)
    # k-subset per row via argsort ranks (vectorized; guarantees >=1 legal).
    order = torch.argsort(torch.rand((b, a), generator=gen), dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(a).unsqueeze(0).expand(b, -1))
    legal_mask = rank < legal_k.unsqueeze(1)
    chosen = torch.empty((b,), dtype=torch.int64)
    for i in range(b):
        legal_idx = torch.nonzero(legal_mask[i], as_tuple=False).squeeze(1)
        chosen[i] = legal_idx[int(torch.randint(len(legal_idx), (1,), generator=gen).item())]

    feats: dict[str, torch.Tensor] = {}
    for name, spec in specs.items():
        shape = tuple(
            b if d == "B" else t if d == "T" else a if d == "A" else typing.cast("int", d)
            for d in spec.shape
        )
        dt = dtype_map[spec.dtype]
        if name == "history_mask":
            ten = history_mask
        elif name == "legal_mask":
            ten = legal_mask
        elif name == "history_event_kind":
            ten = torch.zeros(shape, dtype=dt)
            ten[history_mask] = torch.randint(
                0, 20, (int(history_mask.sum().item()),), generator=gen
            )
        elif dt is torch.bool:
            ten = torch.rand(shape, generator=gen) < 0.3
        else:
            lo = spec.valid_min if spec.valid_min is not None else 0
            hi = spec.valid_max if spec.valid_max is not None else 8
            if name == "scores":
                lo, hi = -50000, 150000
            elif hi is None or hi > 200:
                hi = 12
            lo_i, hi_i = int(lo), int(hi)
            ten = torch.randint(lo_i, hi_i + 1, shape, generator=gen).to(dt)
        feats[name] = ten

    actor_batch = ActorTensorBatch(
        features=feats,
        history_mask=feats["history_mask"],
        legal_mask=feats["legal_mask"],
        observation_hashes=tuple(["sha256:" + "0" * 64] * b),
        actor_seats=feats["actor_seats"],
    )
    loss_batch = {
        "legal_mask": feats["legal_mask"].to(device),
        "chosen_action_id": chosen.to(device),
    }
    moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in feats.items()}
    actor_batch = ActorTensorBatch(
        features=moved,
        history_mask=moved["history_mask"],
        legal_mask=moved["legal_mask"],
        observation_hashes=actor_batch.observation_hashes,
        actor_seats=moved["actor_seats"],
    )
    return actor_batch, loss_batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="probe run YAML (loads real RunConfig)")
    ap.add_argument("--warmup-per-bucket", type=int, default=2)
    ap.add_argument("--steps-per-bucket", type=int, default=4)
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        _fail("CUDA required (kernel benchmark)")
    device = torch.device("cuda")

    # --- production mirrors (stream_train.run_stream_training order) ---
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    _inductor_ver = "".join(
        c if (c.isalnum() or c in "._-") else "_" for c in str(torch.__version__)
    )
    _inductor_base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join(_inductor_base, "hydra2", f"inductor-torch{_inductor_ver}"),
    )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    from hydra2.runtime.plain import PlainPytorchAdapter
    from hydra2.runtime.protocol import RuntimeSpec, build_runtime
    from hydra2.training import stream_train as st

    if not os.environ.get("HYDRA2_DATA_ROOT"):
        _fail("HYDRA2_DATA_ROOT unset (config interpolation needs it)")
    cfg = load_run_config(args.config)
    b = int(cfg.loop.microbatch_size)
    a = int(cfg.model.action_count)
    if a != BASELINE_ACTION_COUNT:
        _fail(f"action_count {a} != baseline (kbench pins production shapes)")

    model = st._build_model(cfg)
    optimizer = st._build_optimizer(cfg)
    scheduler = st._build_scheduler(cfg, optimizer)
    spec = RuntimeSpec(
        adapter_id=cfg.runtime.adapter_id,  # type: ignore[arg-type]
        device=cfg.runtime.device,
        precision=cfg.loop.precision,  # type: ignore[arg-type]
        compile_mode=cfg.runtime.compile_mode,  # type: ignore[arg-type]
        backward_pass_autocast=st._backward_pass_autocast_for(
            precision=cfg.loop.precision, compile_mode=cfg.runtime.compile_mode
        ),
    )
    handle = build_runtime(
        adapter=PlainPytorchAdapter(), model=model, optimizer=optimizer, spec=spec
    )
    compiled_model = handle.model

    # Loss compile mirror (loop.py ~920-938).
    _torch_compile_loss = torch.compile
    try:
        compiled_loss = _torch_compile_loss(
            supervised_loss_kernel,
            mode="max-autotune-no-cudagraphs",
            dynamic=False,
            fullgraph=False,
            isolate_recompiles=True,
        )
    except TypeError:
        compiled_loss = _torch_compile_loss(
            supervised_loss_kernel,
            mode="max-autotune-no-cudagraphs",
            dynamic=False,
            fullgraph=False,
        )

    weights = {
        "w_policy": cfg.weights.w_policy,
        "w_placement": cfg.weights.w_placement,
        "w_value": cfg.weights.w_value,
        "w_event": dict(cfg.weights.w_event or {}),
        "w_belief": dict(cfg.weights.w_belief or {}),
        "label_smoothing": cfg.weights.label_smoothing,
    }
    use_amp = cfg.loop.precision == "bf16_mixed"
    clip_norm = cfg.loop.gradient_clip_norm

    def autocast():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # --- fixed batches, one per bucket, resident on device ---
    buckets = list(HISTORY_BUCKET_LENGTHS)
    batches: dict[int, tuple[ActorTensorBatch, dict[str, torch.Tensor]]] = {}
    gen = torch.Generator().manual_seed(_SYNTH_SEED)
    for t in buckets:
        batches[t] = _build_synth_batch(b=b, t=t, a=a, gen=gen, device=device)

    def step(t: int) -> float:
        """One production-mirror update (loop.py train-accumulation body)."""
        actor_batch, loss_batch = batches[t]
        validate_actor_batch(actor_batch, a)
        with autocast():
            model_out = model_output_to_loss_dict(compiled_model(actor_batch))
            validate_supervised_inputs(model_out, loss_batch, weights)
            losses = compiled_loss(model_out, loss_batch, weights)
            _check_total_finite(losses["total"])
        total = losses["total"]
        handle.backward(total)
        finite, _norm = global_grad_norm_is_finite(compiled_model)
        if not finite:
            _fail("non-finite grads in kbench step")
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(compiled_model.parameters(), clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return float(total.detach().cpu().item())

    _ = compiled_model.train()
    for t in buckets:
        for _ in range(args.warmup_per_bucket):
            loss_val = step(t)
            if loss_val != loss_val:
                _fail("non-finite warmup loss")

    # --- timed region ---
    # Pass 1 (no profiler): cuda-event wall per step, cycling buckets.
    # Pass 2 (one profiler session per bucket): kernel sums + launch counts.
    # Walls come from pass 1 so Kineto overhead never touches them; kernel
    # sums come from pass 2 where session start/stop syncs are excluded by
    # construction (only kernel durations are summed).
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    steps: list[dict] = []
    kernel_agg: dict[str, float] = {}
    order = [t for _ in range(args.steps_per_bucket) for t in buckets]
    for t in order:
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        loss_val = step(t)
        ev1.record()
        torch.cuda.synchronize()
        wall_us = ev0.elapsed_time(ev1) * 1000.0
        if loss_val != loss_val:
            _fail("non-finite timed loss")
        steps.append({"t": t, "wall_us": wall_us, "loss": loss_val})

    per_bucket_kernel_ms: dict[int, float] = {}
    per_bucket_launches: dict[int, int] = {}
    try:
        for t in buckets:
            with torch.profiler.profile(
                activities=acts, record_shapes=False, with_stack=False
            ) as prof:
                for _ in range(args.steps_per_bucket):
                    step(t)
            ka = prof.key_averages()
            kus, launches = 0.0, 0
            for e in ka:
                try:
                    is_cuda = str(getattr(e, "device_type", "")) == "DeviceType.CUDA"
                except Exception:
                    is_cuda = False
                if not is_cuda:
                    continue
                nm = str(getattr(e, "name", ""))
                if nm.startswith(("Memcpy", "Memset", "Memory")):
                    continue
                kus += float(getattr(e, "self_device_time_total", 0.0))
                launches += int(getattr(e, "count", 0) or 0)
                kernel_agg[nm] = kernel_agg.get(nm, 0.0) + float(
                    getattr(e, "self_device_time_total", 0.0)
                )
            per_bucket_kernel_ms[t] = kus / 1000.0 / args.steps_per_bucket
            per_bucket_launches[t] = launches // args.steps_per_bucket
    except Exception as exc:
        _fail(f"bucket profiler pass failed: {type(exc).__name__}: {exc}")

    kernel_mean = sum(per_bucket_kernel_ms.values()) / len(per_bucket_kernel_ms)
    wall_mean = sum(s["wall_us"] for s in steps) / len(steps) / 1000.0
    launch_mean = sum(per_bucket_launches.values()) // len(per_bucket_launches)
    top = sorted(kernel_agg.items(), key=lambda kv: -kv[1])[:12]

    print(f"KBENCH kernel_ms_per_update={kernel_mean:.3f}", flush=True)
    print(f"KBENCH wall_ms_per_update={wall_mean:.3f}", flush=True)
    print(f"KBENCH launches_per_update={launch_mean}", flush=True)
    print("KBENCH loss_finite=true", flush=True)
    print(
        "KBENCH buckets=" + ",".join(f"{t}:{per_bucket_kernel_ms[t]:.3f}" for t in buckets),
        flush=True,
    )
    report = {
        "config": args.config,
        "microbatch": b,
        "action_count": a,
        "buckets": buckets,
        "kernel_ms_per_update": kernel_mean,
        "wall_ms_per_update": wall_mean,
        "launches_per_update": launch_mean,
        "per_bucket_kernel_ms": {str(t): per_bucket_kernel_ms[t] for t in buckets},
        "per_bucket_launches": {str(t): per_bucket_launches[t] for t in buckets},
        "steps": steps,
        "top_kernels_us": [[n, u] for n, u in top],
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
