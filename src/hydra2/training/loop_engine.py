"""Supervised-loop engine: construction, fetch, telemetry hooks.

Owns :class:`SupervisedLoop` construction (precision gates, device and
manifest-hash validation, compile guards, observer mirrors) together
with the fetch / host-to-device / telemetry hooks the train path calls
each microbatch. Sampler helpers arrive via the checkpoint mixin base
(shared with the persistence path, mirroring the replay split); the
validated-loss step and the accumulation loop arrive via the loss and
train mixins so the engine file stays inside the review-size ceiling.
"""

from __future__ import annotations

import contextlib
import gc
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.training.loop_batch import (
    MicrobatchTelemetry as MicrobatchTelemetry,
)
from hydra2.training.loop_batch import (
    UpdateTelemetry as UpdateTelemetry,
)
from hydra2.training.loop_batch import (
    _move_batch_to_device as _move_batch_to_device,
)
from hydra2.training.loop_batch import (
    summarize_telemetry as summarize_telemetry,
)
from hydra2.training.loop_batch import (
    summarize_update_telemetry as summarize_update_telemetry,
)
from hydra2.training.loop_checkpoint import (
    SupervisedLoopCheckpointMixin as SupervisedLoopCheckpointMixin,
)
from hydra2.training.loop_state import (
    _REQUIRED_MANIFEST_KEYS as _REQUIRED_MANIFEST_KEYS,
)
from hydra2.training.loop_state import (
    TrainingLoopConfig as TrainingLoopConfig,
)
from hydra2.training.loop_state import (
    TrainingState as TrainingState,
)
from hydra2.training.loop_state import (
    _require_sha256 as _require_sha256,
)
from hydra2.training.objectives import (
    compute_supervised_loss as compute_supervised_loss,
)
from hydra2.training.objectives import supervised_loss_kernel as supervised_loss_kernel

if TYPE_CHECKING:
    from hydra2.tracking.clearml_mirror import ClearmlMirror


class SupervisedLoopEngineMixin(SupervisedLoopCheckpointMixin):
    """Construction plus fetch/telemetry hooks for :class:`SupervisedLoop`.

    Split host for the engine third of :class:`SupervisedLoop`; the
    train subclass adds the validated-loss step and the accumulation
    loop and no overrides. Attribute access is duck-typed through the
    subclass.
    """

    model: Any
    optimizer: Any
    scheduler: Any | None
    dataset: Any
    config: TrainingLoopConfig
    handle: Any | None
    runtime_spec: Any | None
    checkpoint_dir: Path
    device: Any
    manifest_hashes: dict[str, str]
    state: TrainingState
    loss_history: list[dict[str, float]]
    _global_metrics_history: list[dict[str, float]]
    privileged_source: Any | None
    evaluation_wall_ids: frozenset[str]
    telemetry_path: Path | None
    feed: Any | None
    _h2d_stream: Any | None
    _h2d_event: Any | None
    telemetry_records: list[MicrobatchTelemetry]
    update_records: list[UpdateTelemetry]
    _compiled_loss: Any
    _mirror: Any
    _mlflow_mirror: Any
    _sampler_state_snapshot: Any
    _restore_sampler_state: Any

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
        # Update-stage timings (optimizer/logging per global update;
        # summary-only, reset per train() alongside microbatch records).
        self.update_records: list[UpdateTelemetry] = []
        # Item 3: bf16 attention path — opt into bf16 SDPA math under
        # bf16_mixed. Duck-typed (no models import): any submodule carrying
        # an attn_bf16 switch gets enabled. Inert on CPU (forward gates on
        # is_cuda); effective on CUDA under loop-owned autocast.
        if self.config.precision == "bf16_mixed":
            for _m in self.model.modules():
                if hasattr(_m, "attn_bf16"):
                    _m.attn_bf16 = True  # type: ignore[attr-defined]

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
                # isolate_recompiles: per-region recompile budget (a region
                # exhausting its budget falls back to eager for that region
                # only, never fails the step). 2.13 fallback via try/except
                # TypeError (protocol.py precedent).
                with contextlib.suppress(Exception):
                    _torch_compile: Any = torch.compile
                    try:
                        _compiled: Any = _torch_compile(
                            self.model,
                            mode="max-autotune-no-cudagraphs",
                            dynamic=True,
                            fullgraph=False,
                            isolate_recompiles=True,
                        )
                    except TypeError:
                        _compiled = _torch_compile(
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
                # isolate_recompiles: per-region recompile budget (same
                # eager-fallback semantics as the model compile above).
                # 2.13 fallback via try/except TypeError.
                with contextlib.suppress(Exception):
                    _torch_compile_loss: Any = torch.compile
                    try:
                        _compiled_loss: Any = _torch_compile_loss(
                            supervised_loss_kernel,
                            mode="max-autotune-no-cudagraphs",
                            dynamic=False,
                            fullgraph=False,
                            isolate_recompiles=True,
                        )
                    except TypeError:
                        _compiled_loss = _torch_compile_loss(
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
        try:
            gc_stats = gc.get_stats()
            gc_counts = (
                int(gc_stats[0].get("collections", 0)),
                int(gc_stats[1].get("collections", 0)),
                int(gc_stats[2].get("collections", 0)),
            )
        except Exception:
            gc_counts = (0, 0, 0)
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
            gc_collections_gen0=gc_counts[0],
            gc_collections_gen1=gc_counts[1],
            gc_collections_gen2=gc_counts[2],
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


__all__ = [
    "SupervisedLoopEngineMixin",
]
