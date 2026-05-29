from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

from hydra_learner.arena_eval import _write_json, default_arena_pyo3_library_path
from hydra_learner.checkpoint import load_checkpoint, load_checkpoint_init_only
from hydra_learner.export_inference import ExportConfig, export_loaded_policy
from hydra_learner.hydra_logging import JsonlLogger, ScalarEventWriter, add_scalars
from hydra_learner.losses import LossWeights
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
from hydra_learner.ppo_step import PpoTrainStepConfig, _validate_json_safe_metrics, ppo_train_step

_BATCH_TO_DEVICE_FOR_TESTS = _batch_to_device
from hydra_learner.rl import EntropyController
from hydra_learner.system_telemetry import resource_delta_metrics, sample_resources, snapshot_metrics


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
    events.write(
        "run_start",
        {
            "config": _json_config(config),
            "config_digest_sha256": config_digest,
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
        _validate_resume_metadata(config.resume, config_digest)
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
    while config.steps is None or global_step < config.steps:
        if first_update_started_at is None:
            first_update_started_at = time.perf_counter()
            startup_s = first_update_started_at - started
        set_optimizer_lr(optimizer, _lr_for_samples(config, samples_seen))
        rollout_seed = config.seed + completed_games
        update_resource_start = sample_resources()
        rollout_started = time.perf_counter()
        checkpoint_save_ms = 0.0
        policy_dir = export_dir / f"onnx_step_{global_step:08d}"
        stage_started = time.perf_counter()
        phase_telemetry.set_phase("rollout_export", global_step)
        if config.rollout_inference == "rust-ort":
            export_loaded_policy(
                ExportConfig(None, "raw", policy_dir, None, 8, max(4096, config.arena_batch_decisions), 18),
                model=model,
                model_config=model_config,
                global_step=global_step,
                samples_seen=samples_seen,
                torch_version=torch.__version__,
                source_label=f"ppo_control_live:{config_digest}:{global_step}:{samples_seen}",
            )
            onnx_export_ms = (time.perf_counter() - stage_started) * 1000.0
            stage_started = time.perf_counter()
            phase_telemetry.set_phase("rollout_collect", global_step)
            payload = _collect_native_rollout(extension, config, policy_dir, rollout_seed)
        else:
            onnx_export_ms = 0.0
            stage_started = time.perf_counter()
            phase_telemetry.set_phase("rollout_collect", global_step)
            payload = _collect_callback_rollout(extension, config, model, rollout_seed)
        native_rollout_ms = (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        phase_telemetry.set_phase("batch_build", global_step)
        batch = _batch_from_native_payload_fast(payload, model)
        batch_build_ms = (time.perf_counter() - stage_started) * 1000.0
        h2d_ms = 0.0
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
            ),
        )
        train_step_ms = (time.perf_counter() - stage_started) * 1000.0
        phase_telemetry.set_phase("train_sync", global_step)
        if torch.device(config.device).type == "cuda":
            torch.cuda.synchronize(torch.device(config.device))
            train_step_ms = (time.perf_counter() - stage_started) * 1000.0
        entropy = result.entropy_controller
        native_timing = payload.get("timing")
        native_timing_metrics = (
            {
                f"native_timing/{key}": value
                for key, value in cast("dict[str, object]", native_timing).items()
                if isinstance(value, int | float)
            }
            if isinstance(native_timing, dict)
            else {}
        )
        global_step += 1
        rows = batch.obs.shape[0]
        samples_seen += rows
        completed_games += config.games_per_update
        elapsed = time.perf_counter() - rollout_started
        update_resource_end = sample_resources()
        metrics: dict[str, object] = {
            **result.metrics,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "completed_games": completed_games,
            "rollout_seed": rollout_seed,
            "rollout_rows": rows,
            "rollout_games": config.games_per_update,
            "rollout_update_ms": elapsed * 1000.0,
            "samples_per_s": rows / elapsed if elapsed > 0.0 else 0.0,
            "lr": optimizer.param_groups[0]["lr"],
            "checkpoint_save_ms": checkpoint_save_ms,
            "onnx_export_ms": onnx_export_ms,
            **native_timing_metrics,
            "lr_schedule_progress_samples": samples_seen,
            "native_rollout_ms": native_rollout_ms,
            "batch_build_ms": batch_build_ms,
            "h2d_ms": h2d_ms,
            "train_step_ms": train_step_ms,
            "startup_s": startup_s,
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
                )
        last_metrics = metrics
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
