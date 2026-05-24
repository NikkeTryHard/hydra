from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, cast

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None

from hydra_learner.raw_mjai import RawMjaiBridgeStats, RawMjaiPinnedQueueStats, build_progress_json

if TYPE_CHECKING:
    from hydra_learner.checkpointing import RawMjaiResumeOffsets
    from hydra_learner.metrics import StepStats
    from hydra_learner.raw_mjai import BuildProgress, RawMjaiDirectStream, RawMjaiPinnedStream
    from hydra_learner.shard_contracts import ManifestSummary


_TB_STATIC_SCALAR_SUFFIXES = ("/active", "/status_code")
_TB_STATIC_SCALAR_KEYS = {
    "active",
    "complete",
    "enabled",
    "full_batches",
    "global_step",
    "status_code",
}


def prefixed_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


class ScalarEventWriter:
    def __init__(self, path: Path | None) -> None:
        self._writer: SummaryWriter | None = None
        self._file: IO[str] | None = None
        if path is None:
            return
        path.mkdir(parents=True, exist_ok=True)
        if SummaryWriter is not None:
            self._writer = SummaryWriter(log_dir=str(path))
        else:
            self._file = (path / "scalars.jsonl").open("a", encoding="utf-8")

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._file is not None:
            self._file.close()

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)
            return
        if self._file is None:
            return
        record = {"wall_time": time.time(), "step": step, "tag": tag, "value": value}
        self._file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()
        if self._file is not None:
            self._file.flush()

    @property
    def enabled(self) -> bool:
        return self._writer is not None or self._file is not None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        out = float(value)
        return out if math.isfinite(out) else None
    return None


def _is_tensorboard_status_clutter(key: str) -> bool:
    return key in _TB_STATIC_SCALAR_KEYS or key.endswith(_TB_STATIC_SCALAR_SUFFIXES)


def add_scalars(
    writer: ScalarEventWriter,
    prefix: str,
    metrics: dict[str, object] | dict[str, float],
    step: int,
    *,
    include_status_scalars: bool = False,
) -> None:
    for key, value in metrics.items():
        if not include_status_scalars and _is_tensorboard_status_clutter(key):
            continue
        number = _finite_number(value)
        if number is not None:
            writer.add_scalar(f"{prefix}/{key}", number, step)


def log_step_scalars(
    writer: ScalarEventWriter, stat: StepStats, *, batch: int, samples_seen: int, global_step: int
) -> None:
    metrics: dict[str, object] = {
        "loss": stat.loss,
        "head_losses": stat.head_losses,
        "target_coverage": stat.target_coverage,
        "step_ms": stat.step_ms,
        "fwd_loss_ms": stat.fwd_loss_ms,
        "backward_ms": stat.backward_ms,
        "optimizer_ms": stat.optimizer_ms,
        "train_gpu_ms": stat.train_gpu_ms,
        "fetch_decode_ms": stat.fetch_decode_ms,
        "h2d_wall_ms": stat.h2d_wall_ms,
        "samples_seen": samples_seen,
        "global_step": global_step,
        "lr": stat.lr,
        "grad_norm": stat.grad_norm,
        "lr_progress_games": stat.lr_progress_games,
        "policy_nll": stat.policy_nll,
        "policy_accuracy": stat.policy_accuracy,
        "policy_top3_accuracy": stat.policy_top3_accuracy,
        "policy_top5_accuracy": stat.policy_top5_accuracy,
        "policy_confidence": stat.policy_confidence,
        "policy_entropy": stat.policy_entropy,
        "policy_target_prob": stat.policy_target_prob,
        "policy_margin": stat.policy_margin,
    }
    for head, loss in stat.head_losses.items():
        metrics[f"loss/{head}"] = loss
    for head, coverage in stat.target_coverage.items():
        metrics[f"coverage/{head}/fraction"] = coverage["fraction"]
    if stat.step_ms > 0.0 and math.isfinite(stat.step_ms):
        metrics["samples_per_s"] = batch * 1000.0 / stat.step_ms
    if math.isfinite(stat.fetch_decode_ms) and math.isfinite(stat.h2d_wall_ms):
        input_pipeline_ms = stat.fetch_decode_ms + stat.h2d_wall_ms
        metrics["input_pipeline_wall_ms"] = input_pipeline_ms
        if math.isfinite(stat.train_gpu_ms):
            total_wall_ms = input_pipeline_ms + stat.train_gpu_ms
            metrics["total_wall_ms"] = total_wall_ms
            if total_wall_ms > 0.0:
                metrics["end_to_end_samples_per_s"] = batch * 1000.0 / total_wall_ms
    add_scalars(writer, "train", metrics, global_step)


def log_validation_scalars(
    writer: ScalarEventWriter, metrics: dict[str, object], global_step: int, *, final: bool
) -> None:
    add_scalars(writer, "final_validation" if final else "validation", metrics, global_step)


def _progress_scalars(progress: BuildProgress | None, offsets: RawMjaiResumeOffsets) -> dict[str, object]:
    if progress is None:
        return {}
    return {
        "progress/complete": progress.complete,
        "progress/build_seconds": progress.build_seconds,
        "progress/stream_local_loaded_games": progress.loaded_games,
        "progress/stream_local_skipped_games": progress.skipped_games,
        "progress/stream_local_samples": progress.samples,
        "progress/stream_local_batches": progress.batches,
        "progress/resume_total_loaded_games": progress.loaded_games + offsets.loaded_games,
        "progress/resume_total_skipped_games": progress.skipped_games + offsets.skipped_games,
        "progress/resume_total_samples": progress.samples + offsets.samples,
        "progress/resume_total_batches": progress.batches + offsets.batches,
        "progress/max_games_reached": progress.max_games_reached,
        "progress/max_samples_reached": progress.max_samples_reached,
    }


def _bridge_scalars(stats: RawMjaiBridgeStats | None) -> dict[str, object]:
    if stats is None:
        return {}
    return {
        "bridge/open_count": stats.open_count,
        "bridge/open_scan_plan_ms": stats.open_scan_plan_ms,
        "bridge/last_next_fill_ms": stats.last_next_fill_ms,
        "bridge/last_queue_wait_ms": stats.last_queue_wait_ms,
        "bridge/last_bytes_filled": stats.last_bytes_filled,
        "bridge/last_games_consumed": stats.last_games_consumed,
    }


def _queue_scalars(stats: RawMjaiPinnedQueueStats | None) -> dict[str, object]:
    if stats is None:
        return {}
    return {
        "queue/ready_wait_ms_total": stats.ready_wait_ms_total,
        "queue/ready_wait_count": stats.ready_wait_count,
        "queue/mean_ready_wait_ms": stats.mean_ready_wait_ms,
        "queue/producer_fill_ms_total": stats.producer_fill_ms_total,
        "queue/produced_batches": stats.produced_batches,
        "queue/mean_producer_fill_ms": stats.mean_producer_fill_ms,
        "queue/producer_free_wait_ms_total": stats.producer_free_wait_ms_total,
        "queue/producer_free_wait_count": stats.producer_free_wait_count,
        "queue/mean_producer_free_wait_ms": stats.mean_producer_free_wait_ms,
        "queue/ready_queue_size": stats.ready_queue_size,
        "queue/free_queue_size": stats.free_queue_size,
    }


def raw_mjai_scalar_snapshot(
    raw_stream: RawMjaiDirectStream | None,
    raw_pinned: RawMjaiPinnedStream | None,
    offsets: RawMjaiResumeOffsets,
) -> dict[str, object]:
    if raw_stream is None and raw_pinned is None:
        return {}
    if raw_stream is not None:
        progress = raw_stream.progress()
        bridge_stats = None
        queue_stats = None
    else:
        assert raw_pinned is not None
        progress = raw_pinned.progress()
        bridge_stats = raw_pinned.bridge_stats()
        queue_stats = raw_pinned.queue_stats()
    return _progress_scalars(progress, offsets) | _bridge_scalars(bridge_stats) | _queue_scalars(queue_stats)


def log_final_scalars(writer: ScalarEventWriter, result: dict[str, object], global_step: int) -> None:
    summary = result.get("summary")
    if isinstance(summary, dict):
        add_scalars(writer, "summary", summary, global_step)
    memory = result.get("memory")
    if isinstance(memory, dict):
        add_scalars(writer, "memory", memory, global_step)
    add_scalars(
        writer,
        "run",
        {
            "compile_s": result.get("compile_s"),
            "global_step": result.get("global_step"),
            "samples_seen": result.get("samples_seen"),
            "raw_mjai_training": result.get("raw_mjai_training"),
            "raw_mjai_pinned_pyo3": result.get("raw_mjai_pinned_pyo3"),
            "compile_dry_run": result.get("compile_dry_run"),
            "warmup_steps_counted": result.get("warmup_steps_counted"),
            "samples_consumed_pre_main": result.get("samples_consumed_pre_main"),
            "pre_main_batches_changed_weights": result.get("pre_main_batches_changed_weights"),
        },
        global_step,
    )
    raw_progress = result.get("raw_mjai_progress")
    if isinstance(raw_progress, dict):
        add_scalars(writer, "raw_mjai", {f"progress/stream_local_{k}": v for k, v in raw_progress.items()}, global_step)
    raw_total_progress = result.get("raw_mjai_resume_total_progress")
    if isinstance(raw_total_progress, dict):
        add_scalars(
            writer,
            "raw_mjai",
            {f"progress/resume_total_{k}": v for k, v in raw_total_progress.items()},
            global_step,
        )
    bridge_stats = result.get("raw_mjai_bridge_stats")
    if isinstance(bridge_stats, dict):
        add_scalars(writer, "raw_mjai", {f"bridge/{k}": v for k, v in bridge_stats.items()}, global_step)
    queue_stats = result.get("raw_mjai_queue_stats")
    if isinstance(queue_stats, dict):
        add_scalars(writer, "raw_mjai", {f"queue/{k}": v for k, v in queue_stats.items()}, global_step)


class JsonlLogger:
    def __init__(self, path: Path | None) -> None:
        self._file = None if path is None else path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

    def write(self, event: str, payload: dict[str, object] | None = None) -> None:
        if self._file is None:
            return
        record: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
        }
        if payload is not None:
            record.update(payload)
        self._file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._file.flush()


def torch_env() -> dict[str, object]:
    env: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "optimizer": "AdamW",
    }
    if torch.cuda.is_available():
        env["device_name"] = torch.cuda.get_device_name()
        env["device_capability"] = torch.cuda.get_device_capability()
    return env


def json_manifest_summary(summary: ManifestSummary | None) -> dict[str, object] | None:
    if summary is None:
        return None
    data = asdict(summary)
    data["path"] = str(data["path"])
    return data


def json_raw_mjai_progress(progress: object | None) -> dict[str, object] | None:
    if progress is None:
        return None
    return build_progress_json(cast("BuildProgress", progress))
