"""Supervised-loop batch helpers: forward bridge, device move, telemetry rows.

Owns the per-batch contract shared by the engine and train paths: the
real-model input bridge, the privileged firewall check, the pinned-source
host-to-device move, the wait-telemetry row types and their p50/p99
summaries, and the scorecard-kind labels. Logging-only helpers here never
affect loss, stepping, or checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from hydra2.contracts.common import ContractError
from hydra2.models.encoder import ActorTensorBatch
from hydra2.models.model import validate_actor_batch
from hydra2.training.loop_state import FORBIDDEN_BATCH_KEYS as FORBIDDEN_BATCH_KEYS


def _window_means(*windows: list[torch.Tensor]) -> list[float]:
    """One-sync means over per-head on-device window tensors.

    Each window's mean uses the identical stack-then-mean op the former
    per-head helper used, so values are bitwise identical; the stacked
    means cross the host in a single ``tolist()`` instead of one ``.item()``
    per head.  Empty windows read ``0.0``.  Order of ``windows`` is the
    order of the returned means.
    """
    slots: list[int | None] = []
    means: list[torch.Tensor] = []
    for window in windows:
        if len(window) == 0:
            slots.append(None)
        else:
            stacked = torch.stack([t.detach().float().reshape(()) for t in window])
            means.append(stacked.mean())
            slots.append(len(means) - 1)
    out: list[float] = [0.0] * len(windows)
    if len(means) != 0:
        flat = torch.stack(means).tolist()  # single host sync for all heads
        for pos, ref in enumerate(slots):
            if ref is not None:
                out[pos] = float(flat[ref])
    return out


def _model_forward(model: nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    # Real-model input bridge: real-mode dataset batches carry the encoded
    # ActorTensorBatch under 'actor_batch' (flat features/legal_mask/chosen
    # keys stay byte-identical for compat). When present AND the model
    # exposes evaluate (real Hydra2BaselineModel; stubs use forward(dict)),
    # route model.evaluate(actor_batch) then map ModelOutput onto the
    # supervised-loss dict contract. Else the legacy dict path below runs
    # byte-identical (forward/evaluate(dict) + identical-object dict return).
    # Replay stays stub/dict path by design (replay.py untouched): replay
    # replays synthetic flat batches with stub models, so it never carries
    # actor_batch and never needs this bridge.
    from hydra2.training.adapters import model_output_to_loss_dict

    actor_batch: Any = batch.get("actor_batch")
    if actor_batch is not None and hasattr(model, "evaluate") and callable(model.evaluate):
        # Eager contract gate BEFORE the (possibly compiled) forward: shapes
        # + nonterminal legal mask raise here on the host, so the inductor
        # graph holds zero device→host syncs. Compiled or not, the same
        # ContractError fires for the same bad batch.
        if isinstance(actor_batch, ActorTensorBatch):
            validate_actor_batch(actor_batch, getattr(model, "action_count", None))
        # Route through __call__ (not .evaluate directly): torch.compile wraps forward,
        # and forward() delegates to evaluate() — calling .evaluate bypasses compilation
        # (0 dynamo graphs). Uncompiled models behave identically (forward→evaluate).
        return model_output_to_loss_dict(model(actor_batch))
    # Pre-Wp05A dict path: evaluate(dict) when present else forward(dict).
    if hasattr(model, "evaluate") and callable(model.evaluate):
        out = model.evaluate(batch)
    else:
        out = model(batch)
    if isinstance(out, dict):
        return out
    return model_output_to_loss_dict(out)


def _validate_batch_no_privileged(batch: dict[str, Any]) -> None:
    for key in batch:
        if key in FORBIDDEN_BATCH_KEYS:
            raise ContractError(
                f"batch contains privileged field {key!r} — WP-05B forbids privileged inputs"
            )
        # Nested dicts (event_targets etc.) — also scan string keys
        val: Any = batch[key]
        if isinstance(val, dict):
            for sub_any in val:
                if not isinstance(sub_any, str):
                    continue
                sub: str = sub_any
                if sub in FORBIDDEN_BATCH_KEYS:
                    raise ContractError(f"batch[{key!r}] contains privileged sub-key {sub!r}")


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    # Perf-A §4.4: non_blocking H2D requires pin_memory source;
    # without it flag is no-op and copy serializes.
    # Evidence:
    #  https://docs.pytorch.org/docs/2.13/generated/
    #  torch.Tensor.pin_memory.html
    #  + torch/utils/data/_utils/pin_memory.py background thread;
    #  non_blocking=True overlaps with compute when pinned.
    # Pinning is done in encoder.py / dataset.py when cuda available;
    # here we respect non_blocking regardless (no-op on cpu).
    moved: dict[str, Any] = {}
    for k, v in batch.items():
        if k.startswith("_"):
            moved[k] = v
        elif isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            # event_targets etc may contain tensors
            inner: dict[str, Any] = {}
            v_dict: dict[Any, Any] = v
            for sk_any, sv_any in v_dict.items():
                sk: str = str(sk_any)
                sv: Any = sv_any
                if isinstance(sv, torch.Tensor):
                    inner[sk] = sv.to(device, non_blocking=True)
                else:
                    inner[sk] = sv
            moved[k] = inner
        else:
            # Real-model bridge: 'actor_batch' ActorTensorBatch value. Prefer
            # its .to(device) when present; otherwise rebuild field-wise
            # (frozen dataclass has no .to today) — flat keys never touched.
            # Replay stays stub/dict path (no actor_batch there).
            try:
                from hydra2.models.encoder import ActorTensorBatch
            except Exception:
                moved[k] = v
                continue
            if isinstance(v, ActorTensorBatch):
                _to = getattr(v, "to", None)
                if callable(_to):
                    try:
                        moved[k] = _to(device)
                    except Exception:
                        moved[k] = v
                else:
                    try:
                        _moved_features: dict[str, Any] = {
                            _fk: (
                                _ft.to(device, non_blocking=True)
                                if isinstance(_ft, torch.Tensor)
                                else _ft
                            )
                            for _fk, _ft in dict(v.features).items()
                        }
                        _hm = (
                            v.history_mask.to(device, non_blocking=True)
                            if isinstance(v.history_mask, torch.Tensor)
                            else v.history_mask
                        )
                        _lm = (
                            v.legal_mask.to(device, non_blocking=True)
                            if isinstance(v.legal_mask, torch.Tensor)
                            else v.legal_mask
                        )
                        _seats = (
                            v.actor_seats.to(device, non_blocking=True)
                            if isinstance(v.actor_seats, torch.Tensor)
                            else v.actor_seats
                        )
                        moved[k] = ActorTensorBatch(
                            features=_moved_features,
                            history_mask=_hm,
                            legal_mask=_lm,
                            observation_hashes=v.observation_hashes,
                            actor_seats=_seats,
                        )
                    except Exception:
                        moved[k] = v
            else:
                _v_to = getattr(v, "to", None)
                if callable(_v_to):
                    try:
                        moved[k] = _v_to(device)
                    except Exception:
                        moved[k] = v
                else:
                    moved[k] = v
    return moved


# ------------------------------------------------------------------
# Phase-3 wait telemetry (hooks only; the ring is sibling-owned).
# ------------------------------------------------------------------
#
# Per-microbatch wall-clock split written to JSONL: queue_wait_ms (time
# blocked waiting for a ready slot — 0.0 when no ring feed is attached
# because the synchronous dataset has no queue), fetch_decode_ms
# (dataset.next_batch wall time), h2d_ms (host-to-device copy wall time,
# or the ring's event-timed h2d_ms_last when a feed is attached),
# compute_ms (forward + loss + backward wall time), producer_wait_s
# (best-effort from feed.stats(), 0.0 when unreported).
#
# Scaling rule: scale the side that waits. Sustained queue_wait_ms means
# the consumer starves (add feed parallelism / ring depth); sustained
# fetch_decode_ms with an empty queue means the producer lags (add decode
# workers); h2d_ms above the overlapped budget means the transfer path
# lags (pinning / depth). Wall-clock waits only — utilization ratios are
# never derived here.
# Gated overlap feed: the pinned-ring module (hydra2.training.pinned_ring)
# exposes PinnedRing.open(shapes, ...) -> handle with handle.next(cpu_batch),
# handle.stats() and handle.close(). The loop consumes that contract
# duck-typed (no import; the handle arrives caller-owned via feed=) so the
# sync fallback stays default. No silent default-on: the ring is opt-in
# (caller attaches feed=) and must verify byte-identical to oracle rows
# before any default flip. When attached, train() prefetches next_batch
# (S1-S9: stream pull + expand + encode) on one background thread so
# fetch_decode overlaps compute; _h2d_batch queue_wait_ms covers the slot-
# recycle wait. Without a feed the synchronous move runs and queue_wait_ms
# is 0.0 — no queue exists to wait on. Sync fallback is always kept when
# CUDA/pinned is unavailable.

_TELEMETRY_METRICS: tuple[str, ...] = (
    "queue_wait_ms",
    "fetch_decode_ms",
    "h2d_ms",
    "compute_ms",
    "forward_ms",
    "loss_ms",
    "backward_ms",
)

#: Per-update optimizer/logging stage metrics (summary-only, no extra JSONL
#: rows: update timings ride the file summary so the existing
#: microbatch+summary row contract stays byte-identical for readers).
_UPDATE_TELEMETRY_METRICS: tuple[str, ...] = (
    "optimizer_ms",
    "logging_ms",
)


@dataclass(frozen=True, slots=True)
class MicrobatchTelemetry:
    """One microbatch wait split (all waits wall-clock milliseconds)."""

    microstep: int
    global_update: int
    queue_wait_ms: float
    fetch_decode_ms: float
    h2d_ms: float
    compute_ms: float
    producer_wait_s: float = 0.0
    # Stage split: forward/loss/backward partition of compute_ms
    forward_ms: float = 0.0
    loss_ms: float = 0.0
    backward_ms: float = 0.0
    # GC attribution: cumulative CPython collections per generation at record
    # time (monotone counters; deltas computed offline). Correlates
    # stop-the-world pauses with compute/fetch spikes without changing GC
    # behavior (read-only gc.get_stats, defaults preserve old call sites).
    gc_collections_gen0: int = 0
    gc_collections_gen1: int = 0
    gc_collections_gen2: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "microbatch",
            "microstep": self.microstep,
            "global_update": self.global_update,
            "queue_wait_ms": self.queue_wait_ms,
            "fetch_decode_ms": self.fetch_decode_ms,
            "h2d_ms": self.h2d_ms,
            "compute_ms": self.compute_ms,
            "producer_wait_s": self.producer_wait_s,
            "forward_ms": self.forward_ms,
            "loss_ms": self.loss_ms,
            "backward_ms": self.backward_ms,
            "gc_collections_gen0": self.gc_collections_gen0,
            "gc_collections_gen1": self.gc_collections_gen1,
            "gc_collections_gen2": self.gc_collections_gen2,
        }


@dataclass(frozen=True, slots=True)
class UpdateTelemetry:
    """One optimizer-update stage split (summary-only, no JSONL rows)."""

    global_update: int
    optimizer_ms: float
    logging_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "update",
            "global_update": self.global_update,
            "optimizer_ms": self.optimizer_ms,
            "logging_ms": self.logging_ms,
        }


def _quantile_sorted(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile over an ascending-sorted list."""
    if len(sorted_values) == 0:
        raise ContractError("quantile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ContractError(f"quantile q must be in [0, 1], got {q!r}")
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def summarize_telemetry(records: list[MicrobatchTelemetry]) -> dict[str, float]:
    """p50/p99 per wait metric over microbatch records ({} when empty)."""
    summary: dict[str, float] = {}
    if len(records) == 0:
        return summary
    for metric in _TELEMETRY_METRICS:
        ordered = sorted(float(getattr(record, metric)) for record in records)
        summary[f"{metric}_p50"] = _quantile_sorted(ordered, 0.50)
        summary[f"{metric}_p99"] = _quantile_sorted(ordered, 0.99)
    return summary


def summarize_update_telemetry(records: list[UpdateTelemetry]) -> dict[str, float]:
    """p50/p99 per update-stage metric ({} when empty; summary-only)."""
    summary: dict[str, float] = {}
    if len(records) == 0:
        return summary
    for metric in _UPDATE_TELEMETRY_METRICS:
        ordered = sorted(float(getattr(record, metric)) for record in records)
        summary[f"{metric}_p50"] = _quantile_sorted(ordered, 0.50)
        summary[f"{metric}_p99"] = _quantile_sorted(ordered, 0.99)
    return summary


def _batch_action_kinds(batch: dict[str, Any], targets: torch.Tensor) -> list[str] | None:
    """Per-row action-kind labels for scorecard logging (``None`` when absent).

    Reads the sampler-attached ``"_action_kinds"`` (or ``"action_kinds"``)
    list; returns ``None`` unless it is a length-matching list of non-empty
    strings.  Logging-only: never affects loss, stepping, or checkpoints.
    """
    raw: Any = batch.get("_action_kinds", batch.get("action_kinds"))
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        return None
    raw_list: list[object] = list(raw)
    kinds = [str(k) for k in raw_list]
    if len(kinds) != targets.shape[0]:
        return None
    if any(k == "" for k in kinds):
        return None
    return kinds


__all__ = [
    "_TELEMETRY_METRICS",
    "_UPDATE_TELEMETRY_METRICS",
    "MicrobatchTelemetry",
    "UpdateTelemetry",
    "_batch_action_kinds",
    "_model_forward",
    "_move_batch_to_device",
    "_quantile_sorted",
    "_validate_batch_no_privileged",
    "_window_means",
    "summarize_telemetry",
    "summarize_update_telemetry",
]
