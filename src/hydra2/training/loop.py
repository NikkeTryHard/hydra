"""WP-05B project-owned supervised loop over authoritative data.

Owns optimizer / scheduler / accumulation / checkpoint / RNG / sampler
state.  Plain PyTorch and Lightning-Fabric adapters share *identical* loop
state — only the ``backward`` call is delegated to the ``RuntimeHandle``.
Local artifacts are authoritative; an observer mirror (see tracking/)
never overwrites them.

Checkpoints are published via :mod:`hydra2.runtime.checkpoint` (atomic
``torch.save`` + manifest), so resume restores model, optimizer, scheduler,
step counters, RNG, sampler cursor and manifest identities before any
mutation.  The loop is deterministic under ``torch.use_deterministic_algorithms``.

The loop never imports or reads privileged parquet; any batch containing a
privileged key (``hidden_tiles``, ``wall``, etc.) is rejected before the
forward pass (see :data:`FORBIDDEN_BATCH_KEYS`).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from hydra2.eval.blocks import BlockTolerance, WallBlock
    from hydra2.eval.telemetry import ResourceTelemetry
    from hydra2.tracking.clearml_mirror import ClearmlMirror

import torch
import torch.nn as nn

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.eval.statistics import SelectionConfig, score_selection
from hydra2.models.encoder import ActorTensorBatch
from hydra2.models.model import validate_actor_batch
from hydra2.runtime.checkpoint import (
    build_manifest,
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from hydra2.training.objectives import (
    _check_total_finite,
    compute_hot_scalars,
    compute_metrics,
    compute_per_type_metrics,
    compute_supervised_loss,
    fit_temperature_scaling,
    global_grad_norm_is_finite,
    supervised_loss_kernel,
    validate_supervised_inputs,
)

__all__ = [
    "FORBIDDEN_BATCH_KEYS",
    "MicrobatchTelemetry",
    "SupervisedLoop",
    "TrainingLoopConfig",
    "TrainingState",
    "summarize_telemetry",
]

FORBIDDEN_BATCH_KEYS = frozenset(
    {
        "hidden_tiles",
        "wall",
        "dead_wall",
        "opponent_hand",
        "privileged",
        "full_world",
        "privileged_label",
        "wall_remaining",
        "hidden",
    }
)


_REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "run_spec_hash",
    "model_spec_hash",
    "optimizer_spec_hash",
    "scheduler_spec_hash",
    "environment_hash",
    "rules_hash",
    "utility_manifest_hash",
    "action_schema_hash",
    "observation_schema_hash",
    "dataset_manifest_hash",
)


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ContractError(f"{name} must be sha256:<64 hex>, got {value!r}")
    return value


def _best_ckpt_digest_for(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _atomic_publish_best(source: Path, dest: Path) -> str:
    """Atomically publish ``source`` bytes to ``dest``; return its digest."""
    src = Path(source)
    dst = Path(dest)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        raise ContractError(f"selection ckpt missing: {src}")
    data = src.read_bytes()
    digest = _best_ckpt_digest_for(data)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as fh:
        _ = fh.write(data)  # intentionally discarded: byte count unneeded after fsync
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dst)
    try:
        dir_fd = os.open(dst.parent, os.O_DIRECTORY)
    except Exception:
        return digest
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return digest


def _verify_best_ckpt(checkpoint_dir: Path, *, metric: float | None, digest: str | None) -> None:
    """Fail closed when a promoted best has no matching ``best-ckpt.pt``."""
    if metric is None and digest is None:
        return
    dest = Path(checkpoint_dir) / "best-ckpt.pt"
    if not dest.is_file():
        raise CorruptArtifactError(
            f"best-ckpt.pt missing for best_selection_metric={metric!r} in {dest.parent}"
        )
    if digest is not None:
        actual = _best_ckpt_digest_for(dest.read_bytes())
        if actual != digest:
            raise CorruptArtifactError(
                f"best-ckpt.pt digest mismatch: expected {digest}, got {actual}"
            )


@dataclass(slots=True)
class TrainingState:
    global_update: int = 0
    microstep: int = 0
    epoch: int = 0
    examples_seen: int = 0
    best_selection_metric: float | None = None
    best_ckpt_digest: str | None = None
    sampler_cursor: Any = None  # JSON value — dict with offset/seed/total
    semantic_rng_state: Any = None
    # Precision regime this state was produced under (persisted so resume
    # cannot silently cross fp32<->bf16). Bound into training_state_hash via
    # the checkpoint payload; run_spec_hash binds it at the manifest level.
    precision: str = "fp32"
    # Per-update finite-grad skip counter (non-finite grad norm skipped step).
    skipped_updates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingState:
        return cls(
            global_update=int(raw.get("global_update", 0)),
            microstep=int(raw.get("microstep", 0)),
            epoch=int(raw.get("epoch", 0)),
            examples_seen=int(raw.get("examples_seen", 0)),
            best_selection_metric=raw.get("best_selection_metric"),
            best_ckpt_digest=raw.get("best_ckpt_digest"),
            sampler_cursor=raw.get("sampler_cursor"),
            semantic_rng_state=raw.get("semantic_rng_state"),
            precision=str(raw.get("precision", "fp32")),
            skipped_updates=int(raw.get("skipped_updates", 0)),
        )


@dataclass(frozen=True, slots=True)
class TrainingLoopConfig:
    """Loop-owned hyperparameters and objective weights (explicit, no defaults).

    All weights are model-spec supplied per SPEC 20; zero-weight heads MAY be
    absent.  Accumulation, clipping, checkpoint frequency and scheduler
    identities are fixed here so that resume is byte-identical.
    """

    # Objective weights (explicit)
    w_policy: float = 1.0
    w_placement: float = 0.0
    w_value: float = 0.0
    w_event: dict[str, float] | None = None
    w_belief: dict[str, float] | None = None

    # Optimizer microbatching — project owned
    microbatch_size: int = 4
    accumulation_steps: int = 8

    # Optimization dynamics
    gradient_clip_norm: float | None = 1.0
    max_updates: int = 10
    checkpoint_frequency_updates: int = 5

    # Determinism
    seed: int = 0

    # Mixed precision — loop-owned bf16 autocast around forward+loss only.
    # fp32 default preserves byte-identical behavior; bf16_mixed opts into
    # torch.autocast(cuda, bfloat16) with fp32 master weights, fp32
    # accumulate/optimizer/clip and NO GradScaler. fp16 excluded by design.
    precision: Literal["fp32", "bf16_mixed"] = "fp32"
    # Wave-3B recipe threading (defaults carry the SOTA-quoted constant):
    # legal-only label-smoothing eps in [0, 1) (0.0 = disabled, plain CE;
    # default 0.03 spreads mass over the LEGAL set only, illegal mass stays
    # exactly zero); sampler/scorecard knobs carried for observability
    # (stream single-pass keeps stream order; parquet dataset honors
    # stratification).
    label_smoothing: float = 0.03
    stratified_sampling: bool = False
    sampling_ratios: dict[str, float] | None = None
    log_per_type_metrics: bool = True
    fit_temperature: bool = True
    fetch_prefetch_depth: int = 3

    @property
    def optimizer_minibatch_size(self) -> int:
        return self.microbatch_size * self.accumulation_steps

    def objective_weights(self) -> dict[str, Any]:
        return {
            "w_policy": self.w_policy,
            "w_placement": self.w_placement,
            "w_value": self.w_value,
            "w_event": dict(self.w_event if self.w_event is not None else {}),
            "w_belief": dict(self.w_belief if self.w_belief is not None else {}),
            "label_smoothing": self.label_smoothing,
        }

    def validate(self) -> None:
        if self.microbatch_size <= 0 or self.accumulation_steps <= 0 or self.max_updates <= 0:
            raise ContractError("microbatch/accumulation/max_updates must be positive")
        if self.checkpoint_frequency_updates <= 0:
            raise ContractError("checkpoint_frequency_updates must be positive")
        if self.optimizer_minibatch_size <= 0:
            raise ContractError("optimizer_minibatch_size must be positive")
        if self.w_policy < 0.0 or self.w_placement < 0.0 or self.w_value < 0.0:
            raise ContractError("w_policy/w_placement/w_value must be nonnegative")
        event_vals = (self.w_event if self.w_event is not None else {}).values()
        belief_vals = (self.w_belief if self.w_belief is not None else {}).values()
        for _w in list(event_vals) + list(belief_vals):
            if not isinstance(_w, (int, float)) or _w < 0.0:
                raise ContractError("w_event/w_belief values must be nonnegative")
        if self.precision not in ("fp32", "bf16_mixed"):
            raise ContractError(f"precision must be 'fp32' or 'bf16_mixed', got {self.precision!r}")
        _eps = self.label_smoothing
        if (
            isinstance(_eps, bool)
            or not isinstance(_eps, (int, float))
            or not (0.0 <= float(_eps) < 1.0)
            or float(_eps) != float(_eps)
        ):
            raise ContractError(f"label_smoothing must lie in [0, 1), got {self.label_smoothing!r}")
        if not isinstance(self.stratified_sampling, bool):
            raise ContractError(
                f"stratified_sampling must be a bool, got {self.stratified_sampling!r}"
            )
        if self.sampling_ratios is not None:
            if not isinstance(self.sampling_ratios, dict):
                raise ContractError("sampling_ratios must be a mapping or null")
            for _k, _v in self.sampling_ratios.items():
                if not isinstance(_k, str) or _k == "":
                    raise ContractError("sampling_ratios keys must be non-empty strings")
                if (
                    isinstance(_v, bool)
                    or not isinstance(_v, (int, float))
                    or not (float(_v) == float(_v))
                    or float(_v) in (float("inf"), float("-inf"))
                    or float(_v) <= 0.0
                ):
                    raise ContractError(f"sampling_ratios[{_k!r}] must be positive and finite")
        if not isinstance(self.log_per_type_metrics, bool):
            raise ContractError(
                f"log_per_type_metrics must be a bool, got {self.log_per_type_metrics!r}"
            )
        if not isinstance(self.fit_temperature, bool):
            raise ContractError(f"fit_temperature must be a bool, got {self.fit_temperature!r}")
        if (
            isinstance(self.fetch_prefetch_depth, bool)
            or not isinstance(self.fetch_prefetch_depth, int)
            or not (1 <= self.fetch_prefetch_depth <= 16)
        ):
            raise ContractError(
                f"fetch_prefetch_depth must be an int in [1, 16], got {self.fetch_prefetch_depth!r}"
            )


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
    # Legacy dict path (Wp05A): evaluate(dict) when present else forward(dict).
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
    # Wave-3C stage split: forward/loss/backward partition of compute_ms
    # (sum ≈ compute_ms; defaults preserve old construction call sites).
    forward_ms: float = 0.0
    loss_ms: float = 0.0
    backward_ms: float = 0.0

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


class SupervisedLoop:
    """Project-owned supervised training loop (WP-05B).

    Parameters
    ----------
    model:
        Any ``torch.nn.Module`` that accepts a batch dict
        ``{features: [B,F], legal_mask: [B,A], chosen_action_id: [B]}``
        (plus optional auxiliary targets) and returns a dict containing at
        least ``policy_logits: [B,A]`` and optional auxiliary logits.
    optimizer:
        Project-owned optimizer (e.g. ``torch.optim.AdamW``).  The loop owns
        stepping and zeroing.
    scheduler:
        Optional project-owned LR scheduler.  Stepped once per global update,
        after the optimizer step.
    dataset:
        Authoritative parquet dataset (or any object exposing ``next_batch``,
        ``get_sampler_state``/``set_sampler_state``, ``__len__`` and
        ``cursor``).  The dataset MUST be actor-only; privileged batches are
        rejected before forward.
    handle:
        Optional :class:`hydra2.runtime.protocol.RuntimeHandle`.  When
        supplied, ``handle.backward(loss)`` is used; otherwise
        ``loss.backward()``.  This is how plain vs Fabric adapters share
        identical loop state.
    config:
        Explicit training hyperparameters and objective weights.
    checkpoint_dir:
        Local artifact directory.  This directory is authoritative; an observer
        mirror (see tracking/) may read but MUST NOT overwrite it (no code path does).
    manifest_hashes:
        Real identity digests required by the checkpoint manifest
        (``run_spec_hash``, ``model_spec_hash``, ``optimizer_spec_hash``,
        ``scheduler_spec_hash``, ``environment_hash``, ``rules_hash``,
        ``utility_manifest_hash``, ``action_schema_hash``,
        ``observation_schema_hash``, ``dataset_manifest_hash``).  Every key
        is required and must be ``sha256:<64 hex>``; missing or malformed
        entries raise :class:`ContractError`.  Callers supply the frozen
        spec digests — no defaults or fallbacks are provided.
    device:
        Target device (e.g. ``cuda`` or ``cpu``).  Defaults to inferred
        from ``handle`` or ``cpu``.
    privileged_source:
        Optional privileged target source joined by opaque ``decision_id``
        only (``PrivilegedOracleLoader`` / ``PrivilegedLabelStore`` / label
        ``dict`` mapping ``decision_id`` -> label dict with ``ranks``).
        The join runs after the privileged firewall check and only when
        ``w_placement``/``w_value`` require targets; ``None`` (default)
        preserves the current ``ContractError``-on-missing-target behavior.
    evaluation_wall_ids:
        Set of wall_ids reserved for evaluation that must never enter
        training. Stored as a frozenset ledger (mirroring replay) and
        forwarded into ``join_oracle_targets`` at both join sites; any
        joined row overlapping the set fails closed. ``None`` (default)
        leaves the join unchecked.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: Any,
        config: TrainingLoopConfig,
        checkpoint_dir: Path,
        manifest_hashes: dict[str, str] | None = None,
        scheduler: Any | None = None,
        handle: Any | None = None,
        device: torch.device | str | None = None,
        privileged_source: Any | None = None,
        evaluation_wall_ids: set[str] | frozenset[str] | None = None,
        mirror: ClearmlMirror | None = None,
        mlflow_mirror: Any | None = None,
        runtime_spec: Any | None = None,
        telemetry_path: Path | str | None = None,
        feed: Any | None = None,
    ) -> None:
        config.validate()
        # Precision agreement: RuntimeSpec vs TrainingLoopConfig must match
        # exactly. A fp32 runtime with a bf16 loop (or the reverse) would
        # silently lie about numerics — both directions are unconstructible.
        # Loop-owned autocast stays keyed on config+CUDA (see
        # _forward_autocast); this gate only prevents disagreement.
        if runtime_spec is not None:
            rt_precision = getattr(runtime_spec, "precision", None)
            if rt_precision != config.precision:
                raise ContractError(
                    f"RuntimeSpec precision {rt_precision!r} disagrees with "
                    f"TrainingLoopConfig precision {config.precision!r}; "
                    "runtime and loop precisions must match exactly"
                )
            rt_adapter = getattr(runtime_spec, "adapter_id", None)
            if rt_adapter == "plain_pytorch" and rt_precision not in ("fp32", "bf16_mixed"):
                raise ContractError(
                    f"PlainPytorchAdapter precision {rt_precision!r} not supported "
                    "(want 'fp32' or 'bf16_mixed'; fp16_mixed rejected: no loss scaler)"
                )
        # Handle-carried precision (forward-compat): adapters bind precision
        # into the handle path via runtime_spec; when present it must agree.
        if handle is not None:
            h_precision = getattr(handle, "precision", None)
            if h_precision is not None and h_precision != config.precision:
                raise ContractError(
                    f"RuntimeHandle precision {h_precision!r} disagrees with "
                    f"TrainingLoopConfig precision {config.precision!r}"
                )
        self.model: nn.Module = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.config = config
        self.handle = handle
        self.runtime_spec = runtime_spec
        self.privileged_source: Any | None = privileged_source
        # Evaluation wall ledger — mirrors replay; forwarded at both join sites.
        self.evaluation_wall_ids: frozenset[str] = frozenset(
            evaluation_wall_ids if evaluation_wall_ids is not None else ()
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if device is not None:
            self.device = torch.device(device)
        elif handle is not None and hasattr(handle, "device"):
            _device_attr: object = handle.device
            self.device = torch.device(str(_device_attr))
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Plain bf16 needs CUDA: loop-owned autocast is CUDA-only, so a CPU
        # device here would silently compute fp32 under a bf16 label.
        if (
            getattr(self.runtime_spec, "adapter_id", None) == "plain_pytorch"
            and getattr(self.runtime_spec, "precision", None) == "bf16_mixed"
            and self.device.type != "cuda"
        ):
            raise ContractError(
                "PlainPytorchAdapter bf16_mixed requires a CUDA device "
                "(loop-owned autocast is CUDA-only; never silent CPU fallback)"
            )

        # Determinism: seed all RNGs deterministically on construction.
        _ = torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        # Manifest hashes — every required digest must be supplied; no defaults.
        if manifest_hashes is None:
            raise ContractError("manifest_hashes is required (all 10 digests must be supplied)")
        validated: dict[str, str] = {}
        for key in _REQUIRED_MANIFEST_KEYS:
            value = manifest_hashes.get(key)
            if not isinstance(value, str) or value == "":
                raise ContractError(f"manifest_hashes[{key!r}] is required (missing or empty)")
            validated[key] = _require_sha256(key, value)
        self.manifest_hashes: dict[str, str] = validated

        self.state = TrainingState(
            global_update=0,
            microstep=0,
            epoch=0,
            examples_seen=0,
            best_selection_metric=None,
            sampler_cursor=self._sampler_state_snapshot(),
            semantic_rng_state=None,
            precision=str(config.precision),
            skipped_updates=0,
        )
        # Loss logging: per-global-update history
        self.loss_history: list[dict[str, float]] = []
        self._global_metrics_history: list[dict[str, float]] = []
        # Phase-3 wait telemetry: JSONL sink (None disables file output;
        # in-memory records are always collected, reset per train() run)
        # plus the optional caller-owned ring feed (gated opt-in
        # pinned_ring.py contract, duck-typed: next/stats; the loop never
        # opens or closes the handle — lifecycle stays with the caller;
        # sync fallback stays default, no silent default-on).
        self.telemetry_path: Path | None = (
            Path(telemetry_path) if telemetry_path is not None else None
        )
        self.feed: Any | None = feed
        # Dedicated H2D transfer stream (sync-path overlap where the loop
        # structure allows): pinned-source non_blocking copies issue on this
        # stream, then order the compute stream after the transfer event so
        # copy-engine work overlaps previous compute tails. None on CPU or
        # when CUDA is unavailable (sync fallback, queue_wait stays 0.0).
        self._h2d_stream: Any | None = None
        self._h2d_event: Any | None = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                self._h2d_stream = torch.cuda.Stream(device=self.device)
        self.telemetry_records: list[MicrobatchTelemetry] = []
        # Wave-3C update-stage timings (optimizer/logging per global update;
        # summary-only, reset per train() alongside microbatch records).
        self.update_records: list[UpdateTelemetry] = []

        # Ensure model is on device
        with contextlib.suppress(Exception):
            _ = self.model.to(self.device)
        # Perf-B torch.compile — dynamic shapes, guarded determinism
        # + availability (cite docs).
        # Evidence:
        #  https://docs.pytorch.org/docs/stable/generated/torch.compile.html
        #  + https://pytorch.org/docs/stable/generated/
        #  torch.are_deterministic_algorithms_enabled.html
        #  + https://docs.pytorch.org/docs/2.13/generated/
        #  torch.Tensor.pin_memory.html (pin_memory non_blocking H2D)
        # Fallback on failure preserves correctness
        # (compile_once semantics, torch 2.13 compatible).
        # dynamic=True + fullgraph=False keeps bucket invariance
        # 32/64/128/256 (SDPA bool mask).
        if self.device.type == "cuda":
            try:
                _is_compiling = torch.compiler.is_compiling()
            except Exception:
                _is_compiling = False
            if not _is_compiling and not torch.are_deterministic_algorithms_enabled():
                with contextlib.suppress(Exception):
                    _compiled: Any = torch.compile(
                        self.model,
                        mode="max-autotune-no-cudagraphs",
                        dynamic=True,
                        fullgraph=False,
                    )
                    if isinstance(_compiled, nn.Module):
                        self.model = _compiled
        # Compiled supervised loss (same guards): the smoothing path is
        # memory-bound full-vocab passes; inductor fuses them (measured ~8x
        # on synthetic microbatches; less in-training — see telemetry) with
        # values matching eager to 1ulp (see
        # test_compiled_supervised_loss_matches_eager_bitwise, which pins
        # bitwise equality where exact, inside the repo allclose bar
        # otherwise). Shapes are static per run ([B,A] fixed microbatch,
        # drop_last tails raise), so dynamic=False. Fallback preserves
        # correctness.
        # Compiled entry is the check-free kernel: the loop pre-validates
        # eagerly via _validated_loss (same errors), so the inductor graph
        # holds zero host syncs. Eager fallback stays the validating public
        # fn (replay/tests/direct callers unchanged).
        self._compiled_loss: Any = compute_supervised_loss
        if self.device.type == "cuda":
            try:
                _loss_compiling = torch.compiler.is_compiling()
            except Exception:
                _loss_compiling = False
            if not _loss_compiling and not torch.are_deterministic_algorithms_enabled():
                with contextlib.suppress(Exception):
                    _compiled_loss: Any = torch.compile(
                        supervised_loss_kernel,
                        mode="max-autotune-no-cudagraphs",
                        dynamic=False,
                        fullgraph=False,
                    )
                    if callable(_compiled_loss):
                        self._compiled_loss = _compiled_loss
        # Observer mirror (see hydra2.tracking): disabled by default; when
        # enabled it copies allowlisted scalars + digests, never feeds back.
        if mirror is None:
            from hydra2.tracking.clearml_mirror import make_mirror

            mirror = make_mirror(
                manifest_hashes=dict(validated),
                loop_config={
                    "microbatch_size": config.microbatch_size,
                    "accumulation_steps": config.accumulation_steps,
                    "max_updates": config.max_updates,
                    "checkpoint_frequency_updates": config.checkpoint_frequency_updates,
                    "seed": config.seed,
                    "w_policy": config.w_policy,
                    "w_placement": config.w_placement,
                    "w_value": config.w_value,
                    "precision": config.precision,
                },
            )
            _ = mirror.start_run()
        self._mirror: ClearmlMirror = mirror
        # MLflow quiet mirror (default-on in production, Null under
        # HYDRA2_MLFLOW_DISABLED in tests). Started here only when the loop
        # constructs it; stream_train passes a started instance instead.
        if mlflow_mirror is None:
            from hydra2.tracking.mlflow_mirror import make_mirror as make_mlflow_mirror

            mlflow_mirror = make_mlflow_mirror(
                manifest_hashes=dict(validated),
                loop_config={
                    "microbatch_size": config.microbatch_size,
                    "accumulation_steps": config.accumulation_steps,
                    "max_updates": config.max_updates,
                    "checkpoint_frequency_updates": config.checkpoint_frequency_updates,
                    "seed": config.seed,
                    "w_policy": config.w_policy,
                    "w_placement": config.w_placement,
                    "w_value": config.w_value,
                    "precision": config.precision,
                },
            )
            _ = mlflow_mirror.start_run()
        self._mlflow_mirror: Any = mlflow_mirror

    # ------------------------------------------------------------------
    # Sampler state helpers
    # ------------------------------------------------------------------

    def _sampler_state_snapshot(self) -> dict[str, Any]:
        if hasattr(self.dataset, "get_sampler_state"):
            return self.dataset.get_sampler_state()
        # Fallback: cursor int
        cur = getattr(self.dataset, "cursor", 0)
        tot = len(self.dataset) if hasattr(self.dataset, "__len__") else 0
        return {"offset": int(cur), "seed": self.config.seed, "total": tot, "epoch": 0}

    def _restore_sampler_state(self, state: Any) -> None:
        if hasattr(self.dataset, "set_sampler_state"):
            self.dataset.set_sampler_state(state)
        else:
            # Fallback best-effort: set cursor attribute
            with contextlib.suppress(Exception):
                _raw_offset: object = state.get("offset", 0) if isinstance(state, dict) else 0
                if isinstance(_raw_offset, bool):
                    _offset = int(_raw_offset)
                elif isinstance(_raw_offset, int):
                    _offset = _raw_offset
                else:
                    _offset = int(str(_raw_offset))
                self.dataset.cursor = _offset

    # ------------------------------------------------------------------
    # Phase-3 wait-telemetry hooks
    # ------------------------------------------------------------------

    def _acquire_cpu_batch(self, microbatch_size: int) -> tuple[dict[str, Any], float]:
        """Fetch one CPU batch; return ``(batch, fetch_decode_ms)``."""
        t0 = time.perf_counter()
        raw_any: Any = self.dataset.next_batch(microbatch_size)
        fetch_decode_ms = (time.perf_counter() - t0) * 1000.0
        if raw_any is None:
            raise CorruptArtifactError("authoritative dataset returned None batch")
        return dict(raw_any), fetch_decode_ms

    def _acquire_cpu_batch_with_state(
        self, microbatch_size: int
    ) -> tuple[dict[str, Any], float, dict[str, Any]]:
        """Fetch one CPU batch plus its post-fetch sampler snapshot (prefetch unit).

        Runs on the single background prefetch thread when a feed is
        attached; the snapshot is captured in fetch order immediately after
        ``next_batch`` so the consumer never reads a live cursor that has
        already advanced past unconsumed prefetches (deterministic order,
        byte-identical batches; sampler_cursor tracks consumed, not head).
        """
        batch, fetch_decode_ms = self._acquire_cpu_batch(microbatch_size)
        try:
            snap = self._sampler_state_snapshot()
        except Exception:
            snap = {}
        if not isinstance(snap, dict):
            snap = {"offset": 0, "seed": self.config.seed, "total": 0, "epoch": 0}
        return batch, fetch_decode_ms, snap

    def _h2d_batch(self, batch: dict[str, Any]) -> tuple[dict[str, Any], float, float]:
        """Move one batch host-to-device; return ``(moved, queue_wait_ms, h2d_ms)``.

        With a caller-owned ring feed (gated opt-in pinned_ring.py
        contract): ``h2d_ms`` is the ring's event-timed ``h2d_ms_last``
        when reported, and ``queue_wait_ms`` is the remaining wall time
        (slot-recycle wait). Without a feed the synchronous move runs and
        ``queue_wait_ms`` is 0.0 — no queue exists to wait on. The sync
        path issues pinned-source non_blocking copies on the dedicated
        transfer stream when CUDA is available (real overlap where the
        loop structure allows) and orders the compute stream after the
        transfer event; CPU/pinned-unavailable falls back to a synchronous
        copy.
        """
        feed: Any | None = self.feed
        if feed is not None:
            t0 = time.perf_counter()
            moved_any: Any = feed.next(batch)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            h2d_ms = wall_ms
            stats_any: Any = feed.stats() if hasattr(feed, "stats") else {}
            if isinstance(stats_any, dict):
                last_any: Any = stats_any.get("h2d_ms_last")
                if isinstance(last_any, (int, float)) and not isinstance(last_any, bool):
                    h2d_ms = float(last_any)
            queue_wait_ms = max(0.0, wall_ms - h2d_ms)
            return dict(moved_any), queue_wait_ms, h2d_ms
        stream: Any | None = self._h2d_stream
        if stream is not None and self.device.type == "cuda" and torch.cuda.is_available():
            try:
                t0 = time.perf_counter()
                with torch.cuda.stream(stream):
                    moved = _move_batch_to_device(batch, self.device)
                    evt: Any = torch.cuda.Event()
                    evt.record(stream)
                torch.cuda.current_stream().wait_event(evt)
                self._h2d_event = evt
                h2d_ms = (time.perf_counter() - t0) * 1000.0
                return moved, 0.0, h2d_ms
            except Exception:
                pass
        t0 = time.perf_counter()
        moved = _move_batch_to_device(batch, self.device)
        h2d_ms = (time.perf_counter() - t0) * 1000.0
        return moved, 0.0, h2d_ms

    def _producer_wait_s(self) -> float:
        """Best-effort producer wait from ``feed.stats()`` (0.0 when unreported)."""
        feed: Any | None = self.feed
        if feed is None or not hasattr(feed, "stats"):
            return 0.0
        stats_any: Any = feed.stats()
        if not isinstance(stats_any, dict):
            return 0.0
        wait_any: Any = stats_any.get("producer_wait_s")
        if isinstance(wait_any, (int, float)) and not isinstance(wait_any, bool):
            return float(wait_any)
        return 0.0

    def _record_microbatch_telemetry(
        self,
        *,
        queue_wait_ms: float,
        fetch_decode_ms: float,
        h2d_ms: float,
        compute_ms: float,
        forward_ms: float = 0.0,
        loss_ms: float = 0.0,
        backward_ms: float = 0.0,
    ) -> None:
        """Append one microbatch record (memory always; JSONL when configured)."""
        record = MicrobatchTelemetry(
            microstep=self.state.microstep,
            global_update=self.state.global_update,
            queue_wait_ms=queue_wait_ms,
            fetch_decode_ms=fetch_decode_ms,
            h2d_ms=h2d_ms,
            compute_ms=compute_ms,
            producer_wait_s=self._producer_wait_s(),
            forward_ms=forward_ms,
            loss_ms=loss_ms,
            backward_ms=backward_ms,
        )
        self.telemetry_records.append(record)
        telemetry_path = self.telemetry_path
        if telemetry_path is not None:
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(telemetry_path, "a", encoding="utf-8") as sink:
                payload = json.dumps(record.to_dict(), sort_keys=True) + "\n"
                _ = sink.write(payload)  # intentionally discarded: byte count unneeded

    def _record_update_telemetry(self, *, optimizer_ms: float, logging_ms: float) -> None:
        """Append one update-stage record (memory only; rides file summary)."""
        self.update_records.append(
            UpdateTelemetry(
                global_update=self.state.global_update,
                optimizer_ms=optimizer_ms,
                logging_ms=logging_ms,
            )
        )

    def telemetry_summary(self) -> dict[str, float]:
        """p50/p99 per wait metric over microbatches + update stages."""
        summary = summarize_telemetry(self.telemetry_records)
        summary.update(summarize_update_telemetry(self.update_records))
        return summary

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def _forward_autocast(self) -> Any:
        """Loop-owned bf16 autocast scope for forward+loss only.

        Returns ``torch.autocast(device_type="cuda", dtype=torch.bfloat16)``
        when ``precision == "bf16_mixed"`` on a CUDA device; otherwise a
        no-op ``nullcontext`` (fp32 default is byte-identical, CPU path
        never autocasts). Backward/clip/step always stay outside in fp32
        with fp32 master weights and NO GradScaler.
        """
        if self.config.precision == "bf16_mixed" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _backward(self, loss: torch.Tensor) -> None:
        if self.handle is not None and hasattr(self.handle, "backward"):
            self.handle.backward(loss)
        else:
            _ = loss.backward()

    def _maybe_join_oracle_targets(
        self, batch: dict[str, Any], evaluation_wall_ids: Any | None = None
    ) -> dict[str, Any]:
        # Join privileged placement/value targets by opaque decision_id only.
        # Called AFTER _validate_batch_no_privileged; merged batch is
        # re-validated so the actor batch stays privileged-free
        # (FORBIDDEN_BATCH_KEYS assert post-merge). Defaults unchanged:
        # absent/empty source or zero auxiliary weights is a no-op (the loss
        # then raises ContractError on missing targets, as before). Belief
        # 34-dist distribution support explicitly DEFERRED: belief heads keep
        # the index-CE contract and are never joined here.
        src: Any | None = self.privileged_source
        # Default to the ctor ledger so both join sites fail closed without
        # requiring an explicit arg; explicit arg still wins (tests + replay mirror).
        if evaluation_wall_ids is None:
            evaluation_wall_ids = getattr(self, "evaluation_wall_ids", None)
        if src is None:
            return batch
        if hasattr(src, "__len__"):
            try:
                if len(src) == 0:  # type: ignore[arg-type]
                    return batch
            except TypeError:
                pass
        if self.config.w_placement == 0.0 and self.config.w_value == 0.0:
            return batch
        decision_ids_any: Any = batch.get("_decision_ids", [])
        if not isinstance(decision_ids_any, (list, tuple)) or len(decision_ids_any) == 0:
            return batch
        decision_ids_raw: list[object] = list(decision_ids_any)
        decision_ids: list[str] = [str(x) for x in decision_ids_raw]
        from hydra2.belief.oracle_loader import join_oracle_targets

        joined = join_oracle_targets(decision_ids, src, evaluation_wall_ids=evaluation_wall_ids)
        merged: dict[str, Any] = dict(batch)
        if self.config.w_placement != 0.0 and "placement_target" in joined:
            merged["placement_target"] = joined["placement_target"]
        if self.config.w_value != 0.0 and "value_target" in joined:
            merged["value_target"] = joined["value_target"]
        _validate_batch_no_privileged(merged)
        return merged

    def _validated_loss(self, model_out: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        """Eager pre-validation + compiled kernel + total-finite gate.

        Runs :func:`validate_supervised_inputs` on the host (same errors the
        kernel skips under compile), invokes ``self._compiled_loss`` (kernel
        on CUDA, validating public fn on fallback paths), then gates the
        weighted total with the identical non-finite error.  Both call sites
        (``train_step`` and the accumulation loop) share this so the fail-
        closed contract cannot drift between them.
        """
        weights = self.config.objective_weights()
        validate_supervised_inputs(model_out, batch, weights)
        losses = self._compiled_loss(model_out, batch, weights)
        _check_total_finite(losses["total"])
        return losses

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Single microbatch forward/backward without optimizer stepping.

        Accumulation is managed by :meth:`train`; this is a low-level helper
        exposed for tests that want formula parity.
        """
        _validate_batch_no_privileged(batch)
        batch = self._maybe_join_oracle_targets(batch, evaluation_wall_ids=self.evaluation_wall_ids)
        batch = _move_batch_to_device(batch, self.device)
        # AMP: autocast covers forward+loss only; backward below stays fp32.
        with self._forward_autocast():
            model_out = _model_forward(self.model, batch)
            losses = self._validated_loss(model_out, batch)
        # Caller scales for accumulation; we return the unscaled total for logging
        total_tensor: torch.Tensor = losses["total"]
        total_unscaled: float = float(total_tensor.detach().cpu().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalar; alternative (keep on device) loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        # Backward is caller-owned when accumulating; this helper does backward with scale 1
        # For direct use, backward here; for accumulate loop the caller re-scales.
        # We expose both: this does immediate backward of total
        self._backward(total_tensor)
        aux_scalars: dict[str, float] = {}
        for _aux_key, _aux_value in losses.items():
            if _aux_key in ("_event_per_head", "_belief_per_head", "total"):
                continue
            _aux_tensor: torch.Tensor = _aux_value
            aux_scalars[_aux_key] = float(_aux_tensor.detach().cpu().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for logging scalars; alternative loses logging. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        return {
            "total": total_unscaled,
            **aux_scalars,
        }

    def train(self, *, max_updates: int | None = None) -> list[dict[str, float]]:
        """Run supervised training for ``max_updates`` global updates.

        Each global update consumes ``accumulation_steps`` microbatches of
        ``microbatch_size`` rows (``optimizer_minibatch_size`` rows total).
        Losses are summed exactly over the optimizer minibatch before the step,
        so microbatch size cannot change the objective.

        Returns the per-update loss history (also available as
        ``self.loss_history``).  Checkpoints are written every
        ``checkpoint_frequency_updates`` updates and at the end.
        """
        max_u = max_updates if max_updates is not None else self.config.max_updates
        if max_u <= 0:
            raise ContractError(f"max_updates must be positive, got {max_u}")
        target_global = self.state.global_update + max_u

        _ = self.model.train()
        # Fresh run owns its telemetry: reset records, truncate the JSONL file.
        self.telemetry_records = []
        self.update_records = []
        telemetry_path = self.telemetry_path
        if telemetry_path is not None:
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            # intentionally discarded: char count unneeded for truncation
            _ = telemetry_path.write_text("", encoding="utf-8")

        # Gated fetch prefetch (feed attached only; depth from
        # ``fetch_prefetch_depth``): the whole next_batch chain (stream pull
        # + expand + encode) runs on one background thread while the main
        # thread does H2D + compute, so fetch_decode overlaps compute and
        # _h2d_batch queue_wait covers the slot-recycle wait. Single worker
        # FIFO preserves order exactly (byte-identical batches, deterministic
        # cursor); sampler snapshots travel with the batch (consumed, not
        # live head) so checkpoints resume without skipping
        # prefetched-but-unconsumed rows. Feed None keeps the serial sync
        # fallback (no thread, queue_wait 0.0).
        use_prefetch = self.feed is not None
        total_mb = max_u * self.config.accumulation_steps
        _prefetch_ex: ThreadPoolExecutor | None = None
        _pending: deque[Any] = deque()
        _fetched = 0
        last_snap: dict[str, Any] | None = None
        if use_prefetch:
            _prefetch_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h2d-prefetch")
            _depth = max(1, min(int(self.config.fetch_prefetch_depth), total_mb))
            while _fetched < _depth:
                _pending.append(
                    _prefetch_ex.submit(
                        self._acquire_cpu_batch_with_state, self.config.microbatch_size
                    )
                )
                _fetched += 1
        while self.state.global_update < target_global:
            # Accumulation window — Wave-3C: loss scalars stay on-device as
            # tensors (no per-microbatch .item() syncs); logging aggregates
            # once per update in the logging phase below.
            micro_total_tensors: list[torch.Tensor] = []
            micro_policy_tensors: list[torch.Tensor] = []
            micro_placement_tensors: list[torch.Tensor] = []
            micro_value_tensors: list[torch.Tensor] = []
            micro_event_tensors: list[torch.Tensor] = []
            micro_belief_tensors: list[torch.Tensor] = []
            micro_event_head_tensors: dict[str, list[torch.Tensor]] = {}
            micro_belief_head_tensors: dict[str, list[torch.Tensor]] = {}
            batch: dict[str, Any] = {}
            model_out: dict[str, Any] = {}
            # Zero grad at start of accumulation window
            # (already zeroed after previous update)
            for _acc_step in range(self.config.accumulation_steps):
                if len(_pending) > 0:
                    raw_batch, fetch_decode_ms, _snap = _pending.popleft().result()
                    last_snap = _snap
                    # Lookahead: refill to configured depth while main does H2D+compute.
                    if _fetched < total_mb and _prefetch_ex is not None:
                        _pending.append(
                            _prefetch_ex.submit(
                                self._acquire_cpu_batch_with_state,
                                self.config.microbatch_size,
                            )
                        )
                        _fetched += 1
                else:
                    raw_batch, fetch_decode_ms = self._acquire_cpu_batch(
                        self.config.microbatch_size
                    )
                    last_snap = None
                _validate_batch_no_privileged(raw_batch)
                raw_batch = self._maybe_join_oracle_targets(
                    raw_batch, evaluation_wall_ids=self.evaluation_wall_ids
                )
                batch, queue_wait_ms, h2d_ms = self._h2d_batch(raw_batch)
                # AMP: autocast covers forward+loss only; backward/clip/step below stay fp32.
                # Wave-3C stage split: forward vs loss timed separately inside
                # the shared autocast scope; backward outside; compute_ms keeps
                # the total wall for back-compat.
                compute_t0 = time.perf_counter()
                with self._forward_autocast():
                    _fwd_t0 = time.perf_counter()
                    model_out = _model_forward(self.model, batch)
                    forward_ms = (time.perf_counter() - _fwd_t0) * 1000.0
                    _loss_t0 = time.perf_counter()
                    losses = self._validated_loss(model_out, batch)
                    loss_ms = (time.perf_counter() - _loss_t0) * 1000.0
                # Accumulation: scale loss so that sum over accumulation_steps
                # equals mean over optimizer minibatch (exact numerator/count).
                step_total: torch.Tensor = losses["total"]
                scaled: torch.Tensor = step_total / self.config.accumulation_steps
                _bwd_t0 = time.perf_counter()
                self._backward(scaled)
                backward_ms = (time.perf_counter() - _bwd_t0) * 1000.0
                compute_ms = (time.perf_counter() - compute_t0) * 1000.0
                self._record_microbatch_telemetry(
                    queue_wait_ms=queue_wait_ms,
                    fetch_decode_ms=fetch_decode_ms,
                    h2d_ms=h2d_ms,
                    compute_ms=compute_ms,
                    forward_ms=forward_ms,
                    loss_ms=loss_ms,
                    backward_ms=backward_ms,
                )
                micro_total_tensors.append(step_total.detach())
                placement_loss: torch.Tensor = losses["placement"]
                value_loss: torch.Tensor = losses["value"]
                event_loss: torch.Tensor = losses["event"]
                belief_loss: torch.Tensor = losses["belief"]
                policy_loss: torch.Tensor = losses["policy"]
                micro_placement_tensors.append(placement_loss.detach())
                micro_value_tensors.append(value_loss.detach())
                micro_event_tensors.append(event_loss.detach())
                micro_belief_tensors.append(belief_loss.detach())
                micro_policy_tensors.append(policy_loss.detach())
                _event_heads_any: Any = losses.get("_event_per_head", {})
                if isinstance(_event_heads_any, dict):
                    for _head_id, _head_loss in _event_heads_any.items():
                        if isinstance(_head_loss, torch.Tensor):
                            _event_head_id: object = _head_id
                            _dst = micro_event_head_tensors.setdefault(str(_event_head_id), [])
                            _dst.append(_head_loss.detach())
                _belief_heads_any: Any = losses.get("_belief_per_head", {})
                if isinstance(_belief_heads_any, dict):
                    for _head_id, _head_loss in _belief_heads_any.items():
                        if isinstance(_head_loss, torch.Tensor):
                            _belief_head_id: object = _head_id
                            _dst = micro_belief_head_tensors.setdefault(str(_belief_head_id), [])
                            _dst.append(_head_loss.detach())
                self.state.sampler_cursor = (
                    last_snap
                    if (use_prefetch and last_snap is not None)
                    else self._sampler_state_snapshot()
                )
            # Fail-closed finite-grad skip: per-update global grad-norm finite
            # check BEFORE clip/step. Non-finite grads skip the optimizer (and
            # scheduler) step, zero grads, and count the skip — weights are
            # never poisoned. Mid-window non-finite *loss* still raises
            # ContractError inside compute_supervised_loss (before backward),
            # so there is no double-count: this gate only sees grads.
            # Wave-3C: optimizer phase timed (probe+clip+step+scheduler+zero);
            # logging phase timed separately below (deferred single-sync means).
            _opt_t0 = time.perf_counter()
            _grads_finite, _grad_norm = global_grad_norm_is_finite(self.model)
            if not _grads_finite:
                self.optimizer.zero_grad(set_to_none=True)
                _optimizer_ms = (time.perf_counter() - _opt_t0) * 1000.0
                _log_t0 = time.perf_counter()
                self.state.skipped_updates += 1
                self.state.global_update += 1
                snap_epoch: int = (
                    last_snap.get("epoch", 0)
                    if use_prefetch and last_snap is not None
                    else self._sampler_state_snapshot().get("epoch", 0)
                )
                self.state.epoch = snap_epoch
                self.state.semantic_rng_state = None

                # Deferred single-sync means over the on-device window tensors
                # (one host sync for all heads; see _window_means).
                _skip_means = _window_means(
                    micro_total_tensors,
                    micro_policy_tensors,
                    micro_placement_tensors,
                    micro_value_tensors,
                    micro_event_tensors,
                    micro_belief_tensors,
                )
                _skip_avg = _skip_means[0]
                _skip_entry: dict[str, float] = {
                    "global_update": float(self.state.global_update),
                    "total": _skip_avg,
                    "policy": _skip_means[1],
                    "placement": _skip_means[2],
                    "value": _skip_means[3],
                    "event": _skip_means[4],
                    "belief": _skip_means[5],
                    "masked_nll": _skip_avg,
                    "top1": 0.0,
                    "skipped_updates": float(self.state.skipped_updates),
                    "skipped_this_update": 1.0,
                }
                self.loss_history.append(_skip_entry)
                self._global_metrics_history.append({"masked_nll": _skip_avg})
                _logging_ms = (time.perf_counter() - _log_t0) * 1000.0
                self._record_update_telemetry(optimizer_ms=_optimizer_ms, logging_ms=_logging_ms)
                if (
                    self.state.global_update % self.config.checkpoint_frequency_updates == 0
                    or self.state.global_update == target_global
                ):
                    dest = self.save_checkpoint()
                    self._mirror.log_update(_skip_entry, step=self.state.global_update)
                    self._mirror.log_checkpoint(
                        checkpoint_path=dest,
                        manifest_json={
                            "checkpoint_file": dest.name,
                            "global_update": self.state.global_update,
                            "manifest_hashes": dict(self.manifest_hashes),
                        },
                    )
                    self._mlflow_mirror.log_update(_skip_entry, step=self.state.global_update)
                    self._mlflow_mirror.log_checkpoint(
                        checkpoint_path=dest,
                        manifest_json={
                            "checkpoint_file": dest.name,
                            "global_update": self.state.global_update,
                            "manifest_hashes": dict(self.manifest_hashes),
                        },
                    )
                continue
            if self.config.gradient_clip_norm is not None:
                _ = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            _optimizer_ms = (time.perf_counter() - _opt_t0) * 1000.0

            self.state.global_update += 1
            snap_epoch2: int = (
                last_snap.get("epoch", 0)
                if use_prefetch and last_snap is not None
                else self._sampler_state_snapshot().get("epoch", 0)
            )
            self.state.epoch = snap_epoch2
            self.state.semantic_rng_state = None  # populated at checkpoint via capture_rng_state

            # Logging: mean over accumulation window + per-head metrics on last microbatch
            # Wave-3C: single-sync means over the deferred window tensors (one
            # host sync per head per update, not per microbatch); metrics on
            # the last microbatch stay lawful peeks. Checkpoint save stays
            # outside logging_ms (cadence stall reviewed separately).
            _log_t0 = time.perf_counter()

            # Recompute metrics for reporting (masked NLL, top-k, etc.) on last logits
            # We reuse last batch/model_out already in scope; recompute with stored batch
            # One host sync for every head mean (bitwise-identical values; see
            # _window_means). Fixed-order lists: 6 trunk heads, then per-head
            # event/belief windows in sorted-head order.
            _event_head_keys = sorted(micro_event_head_tensors)
            _belief_head_keys = sorted(micro_belief_head_tensors)
            _all_means = _window_means(
                micro_total_tensors,
                micro_policy_tensors,
                micro_placement_tensors,
                micro_value_tensors,
                micro_event_tensors,
                micro_belief_tensors,
                *[micro_event_head_tensors[_k] for _k in _event_head_keys],
                *[micro_belief_head_tensors[_k] for _k in _belief_head_keys],
            )
            avg_loss = _all_means[0]
            avg_policy = _all_means[1]
            avg_placement = _all_means[2]
            avg_value = _all_means[3]
            avg_event = _all_means[4]
            avg_belief = _all_means[5]
            _extra_means = _all_means[6:]
            # Hot diagnostics: masked NLL + top-1 only (2 host syncs). Richer
            # metrics (top-k/ECE/uniform/support/per-type) ride the eval
            # report; the batch was pre-validated this microbatch, so no
            # re-validation here. We have batch/model_out from last iteration
            # in scope — recompute hot scalars there.
            try:
                train_logits: torch.Tensor = model_out["policy_logits"]
                train_targets: torch.Tensor = batch["chosen_action_id"]
                train_mask: torch.Tensor = batch["legal_mask"]
                metrics = compute_hot_scalars(
                    train_logits.detach(),
                    train_targets.detach(),
                    train_mask.detach(),
                )
            except Exception:
                metrics = {
                    "masked_nll": avg_loss,
                    "top1": 0.0,
                }

            entry: dict[str, float] = {
                "global_update": float(self.state.global_update),
                "total": avg_loss,
                "policy": avg_policy,
                "placement": avg_placement,
                "value": avg_value,
                "event": avg_event,
                "belief": avg_belief,
                "masked_nll": metrics.get("masked_nll", avg_loss),
                "top1": metrics.get("top1", 0.0),
                "skipped_updates": float(self.state.skipped_updates),
                "skipped_this_update": 0.0,
            }
            for _event_pos, _event_head in enumerate(_event_head_keys):
                entry[f"event_{_event_head}"] = _extra_means[_event_pos]
            _belief_base = len(_event_head_keys)
            for _belief_pos, _belief_head in enumerate(_belief_head_keys):
                entry[f"belief_{_belief_head}"] = _extra_means[_belief_base + _belief_pos]
            self.loss_history.append(entry)
            self._global_metrics_history.append(metrics)
            _logging_ms = (time.perf_counter() - _log_t0) * 1000.0
            self._record_update_telemetry(optimizer_ms=_optimizer_ms, logging_ms=_logging_ms)

            # Checkpointing: local authoritative artifact, atomic publish.
            # Mirror hook fires only after a successful save (observer-only).
            if (
                self.state.global_update % self.config.checkpoint_frequency_updates == 0
                or self.state.global_update == target_global
            ):
                dest = self.save_checkpoint()
                self._mirror.log_update(entry, step=self.state.global_update)
                self._mirror.log_checkpoint(
                    checkpoint_path=dest,
                    manifest_json={
                        "checkpoint_file": dest.name,
                        "global_update": self.state.global_update,
                        "manifest_hashes": dict(self.manifest_hashes),
                    },
                )
                self._mlflow_mirror.log_update(entry, step=self.state.global_update)
                self._mlflow_mirror.log_checkpoint(
                    checkpoint_path=dest,
                    manifest_json={
                        "checkpoint_file": dest.name,
                        "global_update": self.state.global_update,
                        "manifest_hashes": dict(self.manifest_hashes),
                    },
                )

        if _prefetch_ex is not None:
            with contextlib.suppress(Exception):
                _prefetch_ex.shutdown(wait=True)
        if self.telemetry_path is not None:
            summary_line: dict[str, Any] = {
                "kind": "summary",
                "microbatches": len(self.telemetry_records),
                **self.telemetry_summary(),
            }
            with open(self.telemetry_path, "a", encoding="utf-8") as sink:
                payload = json.dumps(summary_line, sort_keys=True) + "\n"
                _ = sink.write(payload)  # intentionally discarded: byte count unneeded

        return list(self.loss_history)

    # ------------------------------------------------------------------
    # Checkpointing — local authoritative artifacts
    # ------------------------------------------------------------------

    def save_checkpoint(self, destination: Path | None = None) -> Path:
        """Atomically publish a checkpoint (local authoritative).

        An observer mirror (see tracking/), if configured, may ``shutil.copy``
        from this path but MUST NOT overwrite it — no code path in this module
        writes through a mirror.

        Returns the destination path published.
        """
        if destination is None:
            destination = self.checkpoint_dir / f"checkpoint-{self.state.global_update:06d}.pt"
        else:
            destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Payload sections required by SPEC 10 (6 keys)
        # Scheduler state may be empty dict when no scheduler
        sched_state: Any
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            try:
                sched_state = self.scheduler.state_dict()
            except Exception:
                sched_state = {}
        else:
            sched_state = {}

        # Prefetch-aware sampler state: state.sampler_cursor tracks consumed
        # (not live prefetch head) so resume re-fetches at most the 1-deep
        # lookahead deterministically instead of skipping it. Sync path is
        # identical (consumed == live, no outstanding prefetch).
        _consumed = self.state.sampler_cursor
        sampler_state: Any = (
            _consumed if isinstance(_consumed, dict) else self._sampler_state_snapshot()
        )
        payload: dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": sched_state,
            "training_state": self.state.to_dict(),
            "sampler_state": sampler_state,
            "rng_state": capture_rng_state(),
        }
        manifest = build_manifest(
            run_spec_hash=self.manifest_hashes["run_spec_hash"],
            model_spec_hash=self.manifest_hashes["model_spec_hash"],
            optimizer_spec_hash=self.manifest_hashes["optimizer_spec_hash"],
            scheduler_spec_hash=self.manifest_hashes["scheduler_spec_hash"],
            environment_hash=self.manifest_hashes["environment_hash"],
            rules_hash=self.manifest_hashes["rules_hash"],
            utility_manifest_hash=self.manifest_hashes["utility_manifest_hash"],
            action_schema_hash=self.manifest_hashes["action_schema_hash"],
            observation_schema_hash=self.manifest_hashes["observation_schema_hash"],
            dataset_manifest_hash=self.manifest_hashes["dataset_manifest_hash"],
            rollout_artifact_hash=None,
            payload=payload,
        )
        return save_checkpoint(destination=destination, manifest=manifest, payload=payload)

    def resume_from_checkpoint(self, source: Path) -> None:
        """Verified resume: validates manifest before mutating any runtime object.

        Verifies ``run_spec_hash`` and ``dataset_manifest_hash`` and per-section
        hashes before applying.  Any mismatch raises before mutation.

        On success, restores model, optimizer, scheduler, TrainingState,
        sampler cursor and RNG so that training continues bit-identically.
        """
        source = Path(source)
        manifest, payload = load_checkpoint(
            source=source,
            expected_run_spec_hash=self.manifest_hashes["run_spec_hash"],
            expected_source_hash=self.manifest_hashes["dataset_manifest_hash"],
        )
        # Manifest is already validated inside load_checkpoint; double-check source
        # identities match this loop's expectations
        if manifest.model_spec_hash != self.manifest_hashes["model_spec_hash"]:
            raise CorruptArtifactError(
                f"checkpoint model_spec_hash {manifest.model_spec_hash} "
                f"!= expected {self.manifest_hashes['model_spec_hash']}"
            )
        # Apply after verification (order matters per SPEC 10)
        # Restore model / optimizer / scheduler from payload (these are device-agnostic CPU tensors)
        _ = self.model.load_state_dict(payload["model_state"])
        _ = self.optimizer.load_state_dict(payload["optimizer_state"])
        if (
            self.scheduler is not None
            and "scheduler_state" in payload
            and payload["scheduler_state"]
        ):
            try:
                self.scheduler.load_state_dict(payload["scheduler_state"])
            except Exception as exc:
                raise CorruptArtifactError(f"scheduler state incompatible: {exc}") from exc
        # Restore training state and sampler/RNG
        raw_training = payload.get("training_state")
        if isinstance(raw_training, dict):
            restored = TrainingState.from_dict(raw_training)
            # Precision regime must match: a fp32 checkpoint resumed under a
            # bf16 loop (or reverse) would silently change numerics.
            if restored.precision != self.config.precision:
                raise CorruptArtifactError(
                    f"checkpoint precision {restored.precision!r} != "
                    f"loop precision {self.config.precision!r}; refusing cross-regime resume"
                )
            self.state = restored
        else:
            raise CorruptArtifactError("training_state missing or malformed in checkpoint")
        sampler_state = payload.get("sampler_state")
        if sampler_state is not None:
            self._restore_sampler_state(sampler_state)
            # Keep state.sampler_cursor in sync with dataset snapshot
            self.state.sampler_cursor = self._sampler_state_snapshot()
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            # capture_rng_state / _restore is via runtime.checkpoint apply path; inline here
            from hydra2.runtime.checkpoint import _restore_rng_state

            _restore_rng_state(rng_state)
        _ = self.model.to(self.device)
        _ = self.model.train()
        # Best-ckpt binding: a restored best must still match its published file.
        _verify_best_ckpt(
            self.checkpoint_dir,
            metric=self.state.best_selection_metric,
            digest=self.state.best_ckpt_digest,
        )
        # Note: optimizer state tensors remain on CPU after load; the adapter's
        # handle would have moved them on setup.  For plain loop we keep CPU
        # and let next step handle device transfer via model's device.

    # ------------------------------------------------------------------
    # Reporting — masked NLL, top-k, calibration, support/confusion, strata
    # ------------------------------------------------------------------

    def evaluate_report(
        self,
        eval_batches: list[dict[str, Any]] | Any,
        *,
        weights: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute report over eval batches (no grad) with the frozen model.

        Returns dict with keys: ``masked_nll``, ``top1``/``top3``/``top5``,
        ``calibration_ece``, ``support_min``/``max``,
        ``legal_uniform_nll``/``legal_uniform_gap``,
        ``legal_uniform_comparison`` (checklist alias),
        ``strata`` (kinds present when batches carry ``"_action_kinds"``)
        and ``confusion`` (legacy ``0.0``), plus flattened
        ``per_type/<kind>/{n,nll,top1,top3,ece,recall,low_support}`` when kinds
        are complete (``recall`` equals ``top1`` by construction;
        ``low_support`` flags ``n<30``, report-only),
        and post-hoc ``temperature``/``calibrated_nll``/``calibrated_ece``
        fit on the pooled eval rows (validation-only, never training).
        """
        _ = self.model.eval()
        total_nll = 0.0
        total_top1 = 0.0
        total_top3 = 0.0
        total_top5 = 0.0
        total_ece = 0.0
        total_uniform_nll = 0.0
        total_uniform_gap = 0.0
        n = 0
        # Pooled eval tensors for per-type scorecards + temperature (Wave 3-B
        # logging): overall means below stay mean-of-batch-means for
        # checklist comparability; per-type/temperature pool rows.
        pooled_logits: list[torch.Tensor] = []
        pooled_targets: list[torch.Tensor] = []
        pooled_masks: list[torch.Tensor] = []
        pooled_kinds: list[str] = []
        kinds_complete = True

        # eval_batches may be iterable of batch dicts or a dataset with iter_batches
        if hasattr(eval_batches, "iter_batches"):
            _iter_batches: Callable[..., Iterable[dict[str, Any]]] = eval_batches.iter_batches
            batches = list(_iter_batches(4, max_batches=5))
        elif isinstance(eval_batches, list):
            batches = eval_batches
        else:
            batches = list(eval_batches)

        with torch.no_grad():
            for raw in batches:
                _validate_batch_no_privileged(raw)
                batch = _move_batch_to_device(raw, self.device)
                # AMP: forward runs under bf16 autocast when enabled; metrics stay fp32.
                with self._forward_autocast():
                    out = _model_forward(self.model, batch)
                eval_logits: torch.Tensor = out["policy_logits"]
                eval_targets: torch.Tensor = batch["chosen_action_id"]
                eval_mask: torch.Tensor = batch["legal_mask"]
                metrics = compute_metrics(eval_logits, eval_targets, eval_mask)
                total_nll += metrics["masked_nll"]
                total_top1 += metrics["top1"]
                total_top3 += metrics["top3"]
                total_top5 += metrics["top5"]
                total_ece += metrics["calibration_ece"]
                total_uniform_nll += metrics["legal_uniform_nll"]
                total_uniform_gap += metrics["legal_uniform_gap"]
                n += 1
                pooled_logits.append(eval_logits.detach().to("cpu"))
                pooled_targets.append(eval_targets.detach().to("cpu"))
                pooled_masks.append(eval_mask.detach().to("cpu"))
                batch_kinds = _batch_action_kinds(batch, eval_targets)
                if batch_kinds is None:
                    kinds_complete = False
                else:
                    pooled_kinds.extend(batch_kinds)

        if n == 0:
            raise ContractError("evaluate_report requires at least one eval batch")

        report: dict[str, Any] = {
            "masked_nll": total_nll / n,
            "top1": total_top1 / n,
            "top3": total_top3 / n,
            "top5": total_top5 / n,
            "calibration_ece": total_ece / n,
            "legal_uniform_nll": total_uniform_nll / n,
            "legal_uniform_gap": total_uniform_gap / n,
            "support_min": 0.0,
            "support_max": 0.0,
            "confusion": 0.0,
            "strata": 0.0,
            "legal_uniform_comparison": total_nll / n,  # alias for checklist
            "num_eval_batches": float(n),
        }
        # Pooled per-type scorecards + post-hoc temperature (logging only;
        # failures degrade to legacy keys, never fail the report).
        try:
            flat_logits = torch.cat(pooled_logits, dim=0)
            flat_targets = torch.cat(pooled_targets, dim=0)
            flat_masks = torch.cat(pooled_masks, dim=0)
            flat_masked = flat_logits.to(torch.float32).masked_fill(~flat_masks, float("-inf"))
            flat_probs = torch.softmax(flat_masked, dim=-1)
            _, flat_pred = flat_probs.max(dim=1)
            counts: dict[int, int] = {}
            pred_list: list[int] = flat_pred.tolist()
            for _p in pred_list:
                _pi = _p
                counts[_pi] = counts.get(_pi, 0) + 1
            if len(counts) > 0:
                report["support_min"] = float(min(counts.values()))
                report["support_max"] = float(max(counts.values()))
            _kinds_ok = kinds_complete and len(pooled_kinds) == flat_targets.shape[0]
            if _kinds_ok and self.config.log_per_type_metrics:
                per_type = compute_per_type_metrics(
                    flat_logits, flat_targets, flat_masks, pooled_kinds
                )
                for _kind in sorted(per_type):
                    _km = per_type[_kind]
                    report[f"per_type/{_kind}/n"] = _km["n"]
                    report[f"per_type/{_kind}/nll"] = _km["nll"]
                    report[f"per_type/{_kind}/top1"] = _km["top1"]
                    report[f"per_type/{_kind}/top3"] = _km["top3"]
                    report[f"per_type/{_kind}/ece"] = _km["ece"]
                    report[f"per_type/{_kind}/recall"] = _km["recall"]
                    report[f"per_type/{_kind}/low_support"] = _km["low_support"]
                report["strata"] = float(len(per_type))
            if self.config.fit_temperature:
                temp = fit_temperature_scaling(flat_logits, flat_targets, flat_masks)
                report["temperature"] = temp["temperature"]
                report["calibrated_nll"] = temp["nll_after"]
                report["calibrated_ece"] = temp["ece_after"]
            else:
                report["temperature"] = 1.0
                report["calibrated_nll"] = report["masked_nll"]
                report["calibrated_ece"] = report["calibration_ece"]
        except Exception:
            report.setdefault("temperature", 1.0)
            _ = report.setdefault("calibrated_nll", report["masked_nll"])
            _ = report.setdefault("calibrated_ece", report["calibration_ece"])
        # Also store last metrics for resume comparison
        _ = self.model.train()
        return report

    # ------------------------------------------------------------------
    # Held-out placement selection — best_selection_metric + best-ckpt
    # ------------------------------------------------------------------

    def evaluate_selection(
        self,
        blocks: tuple[WallBlock, ...],
        telemetry_by_game: Mapping[str, ResourceTelemetry],
        config: SelectionConfig,
        peek_index: int,
        tolerance: BlockTolerance | None = None,
    ) -> float:
        """Gated selection score over VALID wall blocks only (lower-better S1).

        Delegates to :func:`score_selection` (telemetry policy + frozen
        peek-discipline guard enforced); returns the metric only. Not called
        by :meth:`train`; the caller scores a wall-disjoint held-out split
        by hand, then promotes by hand via :meth:`maybe_promote_best`.
        """
        metric, _, _ = score_selection(
            blocks, telemetry_by_game, config, peek_index, tolerance=tolerance
        )
        return metric

    def maybe_promote_best(
        self,
        metric: float,
        ckpt: Path,
        *,
        config: SelectionConfig,
        blocks: tuple[WallBlock, ...],
        telemetry_by_game: Mapping[str, ResourceTelemetry],
        peek_index: int,
        tolerance: BlockTolerance | None = None,
    ) -> bool:
        """Atomically publish ``ckpt`` to ``best-ckpt.pt`` iff gated score improves.

        Re-scores ``blocks``/``telemetry_by_game`` through :func:`score_selection`
        (guard + telemetry policy re-enforced) and requires ``metric`` to equal
        the gated score — unguarded promotion is impossible. Best is the
        running minimum. Publish is atomic (temp + ``os.replace`` + fsync)
        and records the published-file digest in
        ``state.best_ckpt_digest``. Never called by :meth:`train`.
        """
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ContractError(f"selection metric must be finite, got {metric!r}")
        if not math.isfinite(float(metric)):
            raise ContractError(f"selection metric must be finite, got {metric!r}")
        if not isinstance(config, SelectionConfig):
            raise ContractError(f"config must be a SelectionConfig, got {type(config).__name__}")
        expected, _, _ = score_selection(
            blocks, telemetry_by_game, config, peek_index, tolerance=tolerance
        )
        if metric != expected:
            raise ContractError(
                f"promotion metric {metric!r} != gated score {expected!r} "
                "for this blocks/telemetry/peek (score first via evaluate_selection)"
            )
        best = self.state.best_selection_metric
        if best is not None and not (metric < best):
            return False
        digest = _atomic_publish_best(Path(ckpt), self.checkpoint_dir / "best-ckpt.pt")
        self.state.best_selection_metric = metric
        self.state.best_ckpt_digest = digest
        return True
