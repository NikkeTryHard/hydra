from __future__ import annotations

import importlib.util
import json
import math
import multiprocessing as mp
import sys
import time
from collections.abc import Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from hydra_learner.arena_eval import _write_json, default_arena_pyo3_library_path
from hydra_learner.checkpoint import ModelConfig, load_checkpoint, load_checkpoint_init_only
from hydra_learner.export_inference import ExportConfig, export_loaded_policy
from hydra_learner.hydra_logging import JsonlLogger, ScalarEventWriter, add_scalars
from hydra_learner.losses import LossWeights
from hydra_learner.model import HydraPolicyNet
from hydra_learner.optim import build_optimizer, set_optimizer_lr
from hydra_learner.phase_telemetry import PhaseTelemetry
from hydra_learner.ppo_control_checkpoint import (
    _model,
    _model_config,
    _optimizer_config,
    _runtime_config,
    _save_t1_checkpoint,
    _validate_resume_metadata,
)
from hydra_learner.ppo_control_config import (
    OBJECTIVE,
    PpoControlConfig,
    _compatible_resume_config_digests,
    _config_digest,
    _json_config,
    parse_args,
    validate_args,
)
from hydra_learner.ppo_control_rollout import (
    _batch_from_native_payload_fast,
    _batch_to_device,
    _collect_callback_rollout,
    _collect_native_rollout,
)
from hydra_learner.ppo_rollout import (
    PpoSnapshotMetadata,
    build_ppo_snapshot_metadata,
    load_ppo_policy_snapshot_artifact,
    save_ppo_policy_snapshot_artifact,
)
from hydra_learner.ppo_step import (
    PpoBatch,
    PpoTrainStepConfig,
    PpoTrainStepResult,
    _validate_json_safe_metrics,
    ppo_train_step,
)

_BATCH_TO_DEVICE_FOR_TESTS = _batch_to_device
_MODEL_TO_DEVICE_FOR_TESTS = torch.nn.Module.to
from hydra_learner.rl import EntropyController
from hydra_learner.system_telemetry import resource_delta_metrics, sample_resources, snapshot_metrics


@dataclass(frozen=True)
class _RolloutResult:
    payload: Mapping[str, object]
    batch: PpoBatch
    snapshot: PpoSnapshotMetadata
    rollout_seed: int
    onnx_export_ms: float
    native_rollout_ms: float
    batch_build_ms: float
    rollout_started: float
    future_rollout_ms: float
    snapshot_save_ms: float = 0.0
    snapshot_artifact_bytes: int = 0
    child_start_load_ms: float = 0.0


@dataclass(frozen=True)
class _PipelineFuture:
    future: Future[_RolloutResult]
    snapshot: PpoSnapshotMetadata
    submitted_at: float
    snapshot_save_ms: float
    snapshot_artifact_bytes: int


def _collect_rollout_batch_from_snapshot_artifact(
    *,
    config: PpoControlConfig,
    config_digest: str,
    model_config: object,
    snapshot_path: Path,
    expected_snapshot: PpoSnapshotMetadata,
    export_dir: Path,
) -> _RolloutResult:
    if not isinstance(model_config, type(_model_config(config))):
        raise TypeError("model_config must be a ModelConfig")
    load_started = time.perf_counter()
    artifact = load_ppo_policy_snapshot_artifact(
        snapshot_path, expected_snapshot=expected_snapshot, expected_model_config=model_config
    )
    model = HydraPolicyNet(
        hidden=artifact.model_config.hidden,
        blocks=artifact.model_config.blocks,
        bottleneck=artifact.model_config.bottleneck,
        actions=artifact.model_config.actions,
        residual_profile=artifact.model_config.residual_profile,
        backbone_profile=artifact.model_config.backbone_profile,
        conv_memory_format=artifact.model_config.conv_memory_format,
    )
    model.load_state_dict(artifact.model_state, strict=True)
    rollout_device = torch.device(config.rollout_device or config.device)
    model.to(rollout_device)
    model.eval()
    extension = _load_extension(config.extension_path or default_arena_pyo3_library_path())
    child_start_load_ms = (time.perf_counter() - load_started) * 1000.0
    result = _collect_rollout_batch(
        config=config,
        config_digest=config_digest,
        model_config=model_config,
        extension=extension,
        export_dir=export_dir,
        model=model,
        global_step=expected_snapshot.global_step,
        samples_seen=expected_snapshot.samples_seen,
        completed_games=expected_snapshot.completed_games,
    )
    if result.snapshot != expected_snapshot:
        raise ValueError("child rollout snapshot metadata does not match expected snapshot")
    _validate_batch_snapshot(result.batch, expected_snapshot)
    return _RolloutResult(
        payload=result.payload,
        batch=result.batch,
        snapshot=result.snapshot,
        rollout_seed=result.rollout_seed,
        onnx_export_ms=result.onnx_export_ms,
        native_rollout_ms=result.native_rollout_ms,
        batch_build_ms=result.batch_build_ms,
        rollout_started=result.rollout_started,
        future_rollout_ms=result.future_rollout_ms,
        child_start_load_ms=child_start_load_ms,
    )


def _capture_frozen_model_snapshot(config: PpoControlConfig, model: torch.nn.Module) -> torch.nn.Module:
    snapshot_model = _model(config)
    snapshot_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    snapshot_model.load_state_dict(snapshot_state, strict=True)
    _MODEL_TO_DEVICE_FOR_TESTS(snapshot_model, torch.device(config.rollout_device or config.device))
    snapshot_model.eval()
    return snapshot_model


def _collect_rollout_batch(
    *,
    config: PpoControlConfig,
    config_digest: str,
    model_config: ModelConfig,
    extension: object,
    export_dir: Path,
    model: HydraPolicyNet,
    global_step: int,
    samples_seen: int,
    completed_games: int,
) -> _RolloutResult:
    rollout_seed = config.seed + completed_games
    rollout_started = time.perf_counter()
    policy_dir = export_dir / f"onnx_step_{global_step:08d}"
    snapshot = _snapshot_metadata(config, config_digest, global_step, samples_seen, completed_games, rollout_seed)
    stage_started = time.perf_counter()
    if config.rollout_inference == "rust-ort":
        export_loaded_policy(
            ExportConfig(
                None, "raw", policy_dir, None, 8, max(4096, config.arena_batch_decisions), 18, "ppo_policy_value"
            ),
            model=model,
            model_config=model_config,
            global_step=global_step,
            samples_seen=samples_seen,
            torch_version=torch.__version__,
            source_label=f"ppo_control_snapshot:{config_digest}:{global_step}:{samples_seen}",
        )
        onnx_export_ms = (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        payload = _collect_native_rollout(extension, config, policy_dir, rollout_seed, snapshot)
    else:
        onnx_export_ms = 0.0
        stage_started = time.perf_counter()
        payload = _collect_callback_rollout(extension, config, model, rollout_seed, snapshot)
    native_rollout_ms = (time.perf_counter() - stage_started) * 1000.0
    stage_started = time.perf_counter()
    batch = _batch_from_native_payload_fast(payload, model)
    native_timing = payload.get("timing")
    if isinstance(native_timing, dict):
        native_timing.update(_legal_count_probe_metrics(batch))
    batch_build_ms = (time.perf_counter() - stage_started) * 1000.0
    _validate_batch_snapshot(batch, snapshot)
    return _RolloutResult(
        payload=payload,
        batch=batch,
        snapshot=snapshot,
        rollout_seed=rollout_seed,
        onnx_export_ms=onnx_export_ms,
        native_rollout_ms=native_rollout_ms,
        batch_build_ms=batch_build_ms,
        rollout_started=rollout_started,
        future_rollout_ms=(time.perf_counter() - rollout_started) * 1000.0,
    )


def _validate_batch_snapshot(batch: PpoBatch, snapshot: PpoSnapshotMetadata) -> None:
    metadata = batch.snapshot_metadata
    if metadata is None:
        return
    raw_snapshot_id = metadata.get("snapshot_id")
    if not isinstance(raw_snapshot_id, str):
        raise ValueError("PPO batch snapshot_metadata missing snapshot_id")
    if raw_snapshot_id != snapshot.snapshot_id:
        raise ValueError("PPO batch mixes snapshot metadata from different policies")


def _legal_count_probe_metrics(batch: PpoBatch) -> dict[str, object]:
    counts = batch.legal_count.detach().to(device="cpu", dtype=torch.float32)
    rows = int(counts.numel())
    if rows < 1:
        raise ValueError("PPO legal count probe requires at least one row")
    sorted_counts, _ = torch.sort(counts)

    def percentile(probability: float) -> float:
        index = min(rows - 1, max(0, math.ceil(probability * rows) - 1))
        return float(sorted_counts[index])

    sum_legal = float(counts.sum())
    transport_ratio = (sum_legal + rows) / (rows * 47.0)
    return {
        "legal_count_mean": float(counts.mean()),
        "legal_count_p50": percentile(0.50),
        "legal_count_p90": percentile(0.90),
        "legal_count_p99": percentile(0.99),
        "legal_count_max": float(counts.max()),
        "legal_logits_f32_count": int(sum_legal),
        "legal_only_transport_ratio": float(transport_ratio),
    }


def _native_timing_metrics(payload: Mapping[str, object]) -> dict[str, object]:
    native_timing = payload.get("timing")
    if not isinstance(native_timing, dict):
        return {}
    return {
        f"native_timing/{key}": value
        for key, value in cast("dict[str, object]", native_timing).items()
        if isinstance(value, int | float)
    }


def _train_batch(
    *,
    config: PpoControlConfig,
    model: HydraPolicyNet,
    optimizer: torch.optim.Optimizer,
    entropy: EntropyController,
    batch: PpoBatch,
    global_step: int,
    phase_telemetry: PhaseTelemetry,
) -> tuple[PpoTrainStepResult, float]:
    stage_started = time.perf_counter()
    phase_telemetry.set_phase("train_step", global_step)
    result = ppo_train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        entropy_controller=entropy,
        config=PpoTrainStepConfig(
            bc_kl_reverse_coef=config.bc_kl_reverse_coef,
            grad_clip_norm=config.grad_clip_norm,
            microbatch_size=config.microbatch_size,
            epochs=config.epochs,
            target_kl=config.target_kl,
        ),
    )
    train_step_ms = (time.perf_counter() - stage_started) * 1000.0
    phase_telemetry.set_phase("train_sync", global_step)
    if torch.device(config.device).type == "cuda":
        torch.cuda.synchronize(torch.device(config.device))
        train_step_ms = (time.perf_counter() - stage_started) * 1000.0
    return result, train_step_ms


def _effective_rollout_device(config: PpoControlConfig) -> str:
    return config.rollout_device or config.device


def run_ppo_control(config: PpoControlConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = config.output_dir / "logs"
    checkpoint_dir = config.output_dir / "checkpoints"
    export_dir = config.output_dir / "exports"
    rollout_dir = config.output_dir / "rollouts"
    eval_dir = config.output_dir / "eval"
    tensorboard_dir = config.tensorboard_dir or log_dir / "tensorboard"
    for path in (log_dir, checkpoint_dir, export_dir, rollout_dir, eval_dir):
        path.mkdir(parents=True, exist_ok=True)
    events = JsonlLogger(log_dir / "events.jsonl")
    train_log = JsonlLogger(log_dir / "train_steps.jsonl")
    scalars = ScalarEventWriter(tensorboard_dir)
    phase_telemetry = PhaseTelemetry(log_dir / "phase_telemetry.jsonl")
    phase_telemetry.start()
    model_config = _model_config(config)
    optimizer_config = _optimizer_config(config)
    runtime_config = _runtime_config()
    loss_weights = LossWeights()
    config_digest = _config_digest(config)
    started = time.perf_counter()
    run_start_resources = sample_resources()
    effective_rollout_device = _effective_rollout_device(config)
    events.write(
        "run_start",
        {
            "config": _json_config(config),
            "config_digest_sha256": config_digest,
            "train_device": config.device,
            "rollout_device": effective_rollout_device,
            **snapshot_metrics("resources/start", run_start_resources),
        },
    )
    model = _model(config).to(torch.device(config.device))
    optimizer = build_optimizer(model, optimizer_config)
    entropy = EntropyController(
        alpha=config.entropy_alpha, beta=config.entropy_beta, alpha_max=config.entropy_alpha_max
    )
    global_step = 0
    samples_seen = 0
    completed_games = 0
    if config.resume is not None:
        resume = load_checkpoint(
            config.resume,
            model=model,
            optimizer=optimizer,
            expected_model_config=model_config,
            expected_optimizer_config=optimizer_config,
            expected_runtime_config=runtime_config,
            expected_loss_weights=loss_weights,
            expected_manifest_path=None,
        )
        _validate_resume_metadata(config.resume, config_digest, _compatible_resume_config_digests(config))
        global_step = resume.global_step
        samples_seen = resume.samples_seen
        completed_games = resume.raw_mjai_progress.get("completed_games", 0)
        events.write(
            "resume_complete",
            {"checkpoint_path": str(config.resume), "global_step": global_step, "completed_games": completed_games},
        )
    else:
        init_model = _model(config)
        init = load_checkpoint_init_only(config.init_checkpoint, model=init_model, expected_model_config=model_config)
        model.load_state_dict(init_model.state_dict(), strict=True)
        model.to(torch.device(config.device))
        events.write(
            "init_checkpoint_loaded",
            {
                "checkpoint_path": str(config.init_checkpoint),
                "global_step": init.global_step,
                "samples_seen": init.samples_seen,
            },
        )
    extension = _load_extension(config.extension_path or default_arena_pyo3_library_path())
    first_update_started_at = None
    startup_s = 0.0
    last_metrics: dict[str, object] = {}
    pipeline_executor: ProcessPoolExecutor | None = None
    pipeline_future: _PipelineFuture | None = None
    in_flight_discarded = False
    try:
        if config.ppo_pipeline_depth == 1:
            pipeline_executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        while config.steps is None or global_step < config.steps:
            if first_update_started_at is None:
                first_update_started_at = time.perf_counter()
                startup_s = first_update_started_at - started
            set_optimizer_lr(optimizer, _lr_for_samples(config, samples_seen))
            update_resource_start = sample_resources()
            checkpoint_save_ms = 0.0
            rollout_wait_ms = 0.0
            train_overlap_ms = 0.0
            overlap_efficiency = 0.0
            if pipeline_future is None:
                phase_telemetry.set_phase("rollout_collect", global_step)
                if config.ppo_pipeline_depth == 0 or effective_rollout_device != config.device:
                    rollout_model = _capture_frozen_model_snapshot(config, model)
                else:
                    rollout_model = model
                rollout = _collect_rollout_batch(
                    config=config,
                    config_digest=config_digest,
                    model_config=model_config,
                    extension=extension,
                    export_dir=export_dir,
                    model=cast("HydraPolicyNet", rollout_model),
                    global_step=global_step,
                    samples_seen=samples_seen,
                    completed_games=completed_games,
                )
            else:
                phase_telemetry.set_phase("rollout_wait", global_step)
                wait_started = time.perf_counter()
                rollout = pipeline_future.future.result()
                rollout_wait_ms = (time.perf_counter() - wait_started) * 1000.0
                train_overlap_ms = max(0.0, rollout.future_rollout_ms - rollout_wait_ms)
                overlap_efficiency = (
                    train_overlap_ms / rollout.future_rollout_ms if rollout.future_rollout_ms > 0.0 else 0.0
                )
                rollout = _RolloutResult(
                    payload=rollout.payload,
                    batch=_BATCH_TO_DEVICE_FOR_TESTS(rollout.batch, torch.device(config.device)),
                    snapshot=rollout.snapshot,
                    rollout_seed=rollout.rollout_seed,
                    onnx_export_ms=rollout.onnx_export_ms,
                    native_rollout_ms=rollout.native_rollout_ms,
                    batch_build_ms=rollout.batch_build_ms,
                    rollout_started=rollout.rollout_started,
                    future_rollout_ms=rollout.future_rollout_ms,
                    snapshot_save_ms=pipeline_future.snapshot_save_ms,
                    snapshot_artifact_bytes=pipeline_future.snapshot_artifact_bytes,
                    child_start_load_ms=rollout.child_start_load_ms,
                )
                pipeline_future = None
            phase_telemetry.set_phase("batch_validate", global_step)
            _validate_batch_snapshot(rollout.batch, rollout.snapshot)
            rollout.batch.validate()
            result, train_step_ms = _train_batch(
                config=config,
                model=model,
                optimizer=optimizer,
                entropy=entropy,
                batch=rollout.batch,
                global_step=global_step,
                phase_telemetry=phase_telemetry,
            )
            entropy = result.entropy_controller
            global_step += 1
            rows = rollout.batch.obs.shape[0]
            samples_seen += rows
            completed_games += config.games_per_update
            elapsed = time.perf_counter() - rollout.rollout_started
            update_resource_end = sample_resources()
            payload_snapshot_raw = rollout.payload.get("snapshot_metadata")
            payload_snapshot = (
                payload_snapshot_raw if isinstance(payload_snapshot_raw, dict) else rollout.snapshot.to_payload()
            )
            if config.ppo_pipeline_depth == 1 and (config.steps is None or global_step < config.steps):
                if pipeline_executor is None:
                    raise RuntimeError("PPO process pipeline executor is not initialized")
                next_step = global_step
                next_samples_seen = samples_seen
                next_completed_games = completed_games
                next_seed = config.seed + next_completed_games
                next_snapshot = _snapshot_metadata(
                    config, config_digest, next_step, next_samples_seen, next_completed_games, next_seed
                )
                snapshot_started = time.perf_counter()
                snapshot_path, snapshot_bytes = save_ppo_policy_snapshot_artifact(
                    config.output_dir, model=model, model_config=model_config, snapshot=next_snapshot
                )
                snapshot_save_ms = (time.perf_counter() - snapshot_started) * 1000.0
                future = pipeline_executor.submit(
                    _collect_rollout_batch_from_snapshot_artifact,
                    config=config,
                    config_digest=config_digest,
                    model_config=model_config,
                    snapshot_path=snapshot_path,
                    expected_snapshot=next_snapshot,
                    export_dir=export_dir,
                )
                pipeline_future = _PipelineFuture(
                    future=future,
                    snapshot=next_snapshot,
                    submitted_at=time.perf_counter(),
                    snapshot_save_ms=snapshot_save_ms,
                    snapshot_artifact_bytes=snapshot_bytes,
                )
            metrics: dict[str, object] = {
                **result.metrics,
                "global_step": global_step,
                "samples_seen": samples_seen,
                "completed_games": completed_games,
                "snapshot_id": rollout.snapshot.snapshot_id,
                "snapshot_global_step": rollout.snapshot.global_step,
                "snapshot_samples_seen": rollout.snapshot.samples_seen,
                "snapshot_completed_games": rollout.snapshot.completed_games,
                "snapshot_metadata": payload_snapshot,
                "train_device": config.device,
                "rollout_device": effective_rollout_device,
                "rollout_seed": rollout.rollout_seed,
                "rollout_rows": rows,
                "rollout_games": config.games_per_update,
                "rollout_update_ms": elapsed * 1000.0,
                "samples_per_s": rows / elapsed if elapsed > 0.0 else 0.0,
                "lr": optimizer.param_groups[0]["lr"],
                "checkpoint_save_ms": checkpoint_save_ms,
                "onnx_export_ms": rollout.onnx_export_ms,
                **_native_timing_metrics(rollout.payload),
                "lr_schedule_progress_samples": samples_seen,
                "native_rollout_ms": rollout.native_rollout_ms,
                "batch_build_ms": rollout.batch_build_ms,
                "h2d_ms": 0.0,
                "train_step_ms": train_step_ms,
                "startup_s": startup_s,
                "pipeline_depth": config.ppo_pipeline_depth,
                "pipeline_enabled": config.ppo_pipeline_depth == 1,
                "pipeline_mode": "process_short_lived" if config.ppo_pipeline_depth == 1 else "serial",
                "rollout_wait_ms": rollout_wait_ms,
                "train_overlap_ms": train_overlap_ms,
                "overlap_efficiency": overlap_efficiency,
                "future_rollout_ms": rollout.future_rollout_ms,
                "snapshot_save_ms": rollout.snapshot_save_ms,
                "snapshot_artifact_bytes": rollout.snapshot_artifact_bytes,
                "child_start_load_ms": rollout.child_start_load_ms,
                "pipeline_in_flight_discarded": False,
                "in_flight_discarded": False,
                **resource_delta_metrics("resources/update", update_resource_start, update_resource_end),
            }
            _validate_json_safe_metrics(metrics)
            if global_step % config.log_every_steps == 0:
                train_log.write("train_step", metrics)
                add_scalars(scalars, "t1_ppo", metrics, global_step)
                scalars.flush()
            if global_step % config.checkpoint_every_steps == 0 or global_step == config.steps:
                _save_t1_checkpoint(
                    checkpoint_dir / "latest.pt",
                    config,
                    model,
                    optimizer,
                    model_config,
                    optimizer_config,
                    runtime_config,
                    loss_weights,
                    global_step,
                    samples_seen,
                    completed_games,
                    config_digest,
                    rollout.snapshot,
                )
                if config.keep_step_checkpoints:
                    _save_t1_checkpoint(
                        checkpoint_dir / f"step_{global_step}.pt",
                        config,
                        model,
                        optimizer,
                        model_config,
                        optimizer_config,
                        runtime_config,
                        loss_weights,
                        global_step,
                        samples_seen,
                        completed_games,
                        config_digest,
                        rollout.snapshot,
                    )
            last_metrics = metrics
    except BaseException:
        if pipeline_future is not None:
            in_flight_discarded = True
            pipeline_future.future.cancel()
            events.write(
                "pipeline_in_flight_discarded",
                {"snapshot_id": pipeline_future.snapshot.snapshot_id, "global_step": global_step},
            )
        events.close()
        phase_telemetry.set_phase("shutdown", global_step)
        phase_telemetry.close()
        train_log.close()
        scalars.close()
        raise
    finally:
        if pipeline_executor is not None:
            pipeline_executor.shutdown(wait=False, cancel_futures=True)
        if in_flight_discarded and last_metrics:
            last_metrics["pipeline_in_flight_discarded"] = True
            last_metrics["in_flight_discarded"] = True
    total_s = time.perf_counter() - started
    summary: dict[str, object] = {
        "objective": OBJECTIVE,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "completed_games": completed_games,
        "checkpoint_path": str(checkpoint_dir / "latest.pt"),
        "summary": {
            "samples_per_s": samples_seen / total_s if total_s > 0.0 else 0.0,
            "last_train_metrics": last_metrics,
        },
        "config_digest_sha256": config_digest,
        "paths": {
            "run_dir": str(config.output_dir),
            "logs": str(log_dir),
            "checkpoints": str(checkpoint_dir),
            "exports": str(export_dir),
            "rollouts": str(rollout_dir),
            "eval": str(eval_dir),
            "tensorboard": str(tensorboard_dir),
        },
    }
    _write_json(config.output_dir / "summary.json", summary)
    _write_json(config.output_dir / "ppo_control_result.json", summary)
    _write_json(config.output_dir / "launch_metadata.json", {"config": _json_config(config), **summary})
    events.write("run_complete", summary)
    events.close()
    phase_telemetry.set_phase("shutdown", global_step)
    phase_telemetry.close()
    train_log.close()
    scalars.close()
    if not config.quiet:
        print(json.dumps(summary, sort_keys=True))
    return summary


def _snapshot_metadata(
    config: PpoControlConfig,
    config_digest: str,
    global_step: int,
    samples_seen: int,
    completed_games: int,
    rollout_seed: int,
) -> PpoSnapshotMetadata:
    return build_ppo_snapshot_metadata(
        config_digest_sha256=config_digest,
        global_step=global_step,
        samples_seen=samples_seen,
        completed_games=completed_games,
        rollout_seed=rollout_seed,
        temperature=config.temperature,
        inference_backend=config.rollout_inference,
        device=_effective_rollout_device(config),
        hidden=config.hidden,
        blocks=config.blocks,
        bottleneck=config.bottleneck,
        residual_profile=config.residual_profile,
        backbone_profile=config.backbone_profile,
        conv_memory_format=config.conv_memory_format,
        encoder_shape=(192, 34),
        action_space=46,
    )


def _lr_for_samples(config: PpoControlConfig, samples_seen: int) -> float:
    if config.lr_warmup_samples > 0 and samples_seen < config.lr_warmup_samples:
        warmup_fraction = samples_seen / config.lr_warmup_samples
        return max(config.min_lr, config.lr * warmup_fraction)
    if config.lr_decay_samples is None:
        return config.lr
    decay_samples = max(1, config.lr_decay_samples - config.lr_warmup_samples)
    decay_index = min(max(0, samples_seen - config.lr_warmup_samples), decay_samples)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_index / decay_samples))
    return config.min_lr + (config.lr - config.min_lr) * cosine


def _load_extension(path: Path) -> Any:
    if not path.exists():
        raise ImportError(f"PyO3 arena extension not found at {path}")
    name = path.stem.removeprefix("lib")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    config = validate_args(parse_args(argv))
    result = run_ppo_control(config)
    if config.quiet:
        print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0
