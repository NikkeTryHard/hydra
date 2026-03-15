from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np

from .augment import apply_train_augmentation
from .checkpoints import (
    ResumeState,
    RuntimeResumeContract,
    read_resume_state,
    restore_checkpoint,
    save_checkpoint,
    write_resume_state,
)
from .config import ExperimentConfig, RunConfig, load_config
from .dataset import (
    ExportDataset,
    ShardRecord,
    iter_train_epoch_batches,
    load_shard_arrays,
)
from .eval_step import eval_step
from .export_weights import load_exported_weights
from .logging import append_jsonl, ensure_logging_dirs
from .model import HydraModel
from .train_state import create_train_state, restore_train_state
from .train_step import (
    train_logical_batch,
    train_logical_batch_data_parallel,
    train_step,
)


def _as_batch(arrays: Mapping[str, object]) -> dict[str, jnp.ndarray]:
    return {
        "obs": jnp.asarray(arrays["obs"]),
        "policy_target": jax.nn.one_hot(jnp.asarray(arrays["action"]), 46),
        "legal_mask": jnp.asarray(arrays["legal_mask"]),
        "value_target": jnp.asarray(arrays["value_target"]),
        "grp_target": jnp.asarray(arrays["grp_target"]),
        "tenpai_target": jnp.asarray(arrays["tenpai_target"]),
        "danger_target": jnp.asarray(arrays["danger_target"]),
        "danger_mask": jnp.asarray(arrays["danger_mask"]),
        "opp_next_target": jnp.asarray(arrays["opp_next_target"]),
        "score_pdf_target": jnp.asarray(arrays["score_pdf_target"]),
        "score_cdf_target": jnp.asarray(arrays["score_cdf_target"]),
        "safety_residual_target": jnp.asarray(arrays["safety_residual_target"]),
        "safety_residual_mask": jnp.asarray(arrays["safety_residual_mask"]),
        "safety_residual_present": jnp.asarray(arrays["safety_residual_present"]),
    }


def _agreement(
    policy_target: jnp.ndarray, legal_mask: jnp.ndarray, logits: jnp.ndarray
) -> float:
    masked_logits = logits + (1.0 - legal_mask) * (-1e9)
    predicted = jnp.argmax(masked_logits, axis=-1)
    target = jnp.argmax(policy_target, axis=-1)
    return float(jnp.mean((predicted == target).astype(jnp.float32)))


def _is_better_validation(
    current: dict[str, float], best: dict[str, float] | None
) -> bool:
    if best is None:
        return True
    return current["policy_loss"] < best["policy_loss"] or (
        abs(current["policy_loss"] - best["policy_loss"]) <= 1e-12
        and current["agreement"] > best["agreement"]
    )


def _iter_microbatches(arrays: Mapping[str, np.ndarray], batch_size: int):
    total = int(arrays["obs"].shape[0])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield {key: value[start:end] for key, value in arrays.items()}


def _effective_run_config(config: ExperimentConfig, cli_smoke_test: bool) -> RunConfig:
    run = config.run.model_copy(deep=True)
    if cli_smoke_test:
        run.smoke_test = True
    if run.smoke_test:
        run.num_epochs = min(run.num_epochs, 1)
        run.max_validation_samples = min(int(run.max_validation_samples or 256), 256)
        run.validate_every_n_steps = 1
        run.checkpoint_every_n_steps = 1
        run.validation_every_n_epochs = 1
    if run.validation_microbatch_size is None:
        run.validation_microbatch_size = int(run.microbatch_size or run.batch_size)
    return run


def _iter_logical_batches(arrays: Mapping[str, np.ndarray], logical_batch_size: int):
    total = int(arrays["obs"].shape[0])
    for start in range(0, total, logical_batch_size):
        end = min(start + logical_batch_size, total)
        yield {key: value[start:end] for key, value in arrays.items()}


def _validation_microbatch_size(config: ExperimentConfig) -> int:
    size = config.run.validation_microbatch_size
    if size is None:
        raise ValueError("validation_microbatch_size must be resolved before training")
    return int(size)


def _can_shard_batch(batch: Mapping[str, jnp.ndarray], device_count: int) -> bool:
    return device_count > 1 and int(batch["obs"].shape[0]) % device_count == 0


def _shard_batch(
    batch: Mapping[str, jnp.ndarray], device_count: int
) -> dict[str, jnp.ndarray]:
    if not _can_shard_batch(batch, device_count):
        raise ValueError(
            f"batch size {int(batch['obs'].shape[0])} is not divisible by local device count {device_count}"
        )
    per_device = int(batch["obs"].shape[0]) // device_count
    return {
        key: value.reshape((device_count, per_device) + tuple(value.shape[1:]))
        for key, value in batch.items()
    }


def _build_model(config: ExperimentConfig) -> HydraModel:
    dtype_map = {
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
    }
    return HydraModel(
        num_blocks=config.model.num_blocks,
        input_channels=config.model.input_channels,
        hidden_channels=config.model.hidden_channels,
        num_groups=config.model.num_groups,
        se_bottleneck=config.model.se_bottleneck,
        action_space=config.model.action_space,
        score_bins=config.model.score_bins,
        num_opponents=config.model.num_opponents,
        grp_classes=config.model.grp_classes,
        num_belief_components=config.model.num_belief_components,
        opponent_hand_type_classes=config.model.opponent_hand_type_classes,
        dtype=dtype_map[config.model.compute_dtype],
    )


def _initial_state(config: ExperimentConfig, model: HydraModel, dataset: ExportDataset):
    train_shards = dataset.shards("train")
    if not train_shards:
        raise ValueError("no train shards found in export root")
    first_arrays = load_shard_arrays(train_shards[0])
    first_microbatch_size = int(config.run.microbatch_size or config.run.batch_size)
    if int(first_arrays["obs"].shape[0]) == 0:
        raise ValueError("first train shard contains zero samples")
    first_batch = _as_batch(
        next(_iter_microbatches(first_arrays, first_microbatch_size))
    )
    if config.run.init_weights_from is not None:
        prefix = config.run.init_weights_from
        metadata_path = prefix.with_suffix(".json")
        archive_path = prefix.with_suffix(".npz")
        params = load_exported_weights(metadata_path, archive_path)
    else:
        key = jax.random.PRNGKey(config.run.seed)
        params = model.init(key, first_batch["obs"])["params"]
    return create_train_state(
        model.apply,
        params,
        config.optimizer.learning_rate,
        config.optimizer.min_learning_rate,
        config.optimizer.weight_decay,
        config.optimizer.grad_clip_norm,
        config.optimizer.warmup_steps,
    )


def _config_fingerprint(config_section) -> str:
    return json.dumps(config_section.model_dump(), sort_keys=True)


def _load_best_validation(checkpoint_dir: Path) -> dict[str, float] | None:
    if checkpoint_dir.parent.name == "best":
        best_state_path = checkpoint_dir.parent / "latest_state.json"
    else:
        best_state_path = checkpoint_dir.parent.parent / "best" / "latest_state.json"
    if not best_state_path.exists():
        return None
    raw = json.loads(best_state_path.read_text())
    if "validation_policy_loss" not in raw or "validation_agreement" not in raw:
        return None
    return {
        "policy_loss": float(raw["validation_policy_loss"]),
        "agreement": float(raw["validation_agreement"]),
    }


def _validate(
    state,
    model: HydraModel,
    shards: list[ShardRecord],
    validation_microbatch_size: int,
    max_samples: int | None,
) -> dict[str, float]:
    total = 0.0
    policy = 0.0
    agreement = 0.0
    samples = 0
    for shard in shards:
        arrays = load_shard_arrays(shard)
        for batch_np in _iter_microbatches(arrays, validation_microbatch_size):
            if max_samples is not None:
                remaining = max_samples - samples
                if remaining <= 0:
                    break
                if int(batch_np["obs"].shape[0]) > remaining:
                    batch_np = {
                        key: value[:remaining] for key, value in batch_np.items()
                    }
            batch = _as_batch(batch_np)
            breakdown = eval_step(state.params, batch, model)
            output = model.apply({"params": state.params}, batch["obs"])
            batch_size = int(batch["obs"].shape[0])
            total += float(breakdown.total) * batch_size
            policy += float(breakdown.policy) * batch_size
            agreement += (
                _agreement(
                    batch["policy_target"], batch["legal_mask"], output.policy_logits
                )
                * batch_size
            )
            samples += batch_size
            if max_samples is not None and samples >= max_samples:
                break
        if max_samples is not None and samples >= max_samples:
            break
    if samples == 0:
        raise ValueError(
            "validation produced zero samples; validation split is missing or empty"
        )
    return {
        "total_loss": total / samples,
        "policy_loss": policy / samples,
        "agreement": agreement / samples,
        "samples": float(samples),
    }


def _save_latest(
    output_dir: Path,
    state,
    summary: dict[str, float | int | str],
    resume_state: ResumeState,
) -> None:
    save_checkpoint(
        output_dir / "latest_orbax",
        {"params": state.params, "opt_state": state.opt_state},
        summary,
    )
    write_resume_state(output_dir / "latest_orbax", resume_state)
    (output_dir / "latest_state.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )


def _runtime_contract(
    config: ExperimentConfig, dataset: ExportDataset
) -> RuntimeResumeContract:
    return RuntimeResumeContract(
        manifest_fingerprint=dataset.manifest_fingerprint,
        batch_size=int(config.run.batch_size),
        train_microbatch_size=int(config.run.microbatch_size or config.run.batch_size),
        validation_microbatch_size=int(
            config.run.validation_microbatch_size
            or config.run.microbatch_size
            or config.run.batch_size
        ),
        model_fingerprint=_config_fingerprint(config.model),
        optimizer_fingerprint=_config_fingerprint(config.optimizer),
    )


def _runtime_info(config: ExperimentConfig) -> dict[str, str]:
    devices = jax.devices()
    local_devices = jax.local_devices()
    process_count = jax.process_count()
    backend = devices[0].platform if devices else jax.default_backend()
    if config.run.required_backend != "any" and backend != config.run.required_backend:
        raise ValueError(
            f"required backend {config.run.required_backend} not available; got {backend}"
        )
    if process_count != 1:
        raise ValueError(
            f"only single-host execution is supported right now; got process_count={process_count}"
        )
    device_kind = devices[0].device_kind if devices else "unknown"
    return {
        "backend": backend,
        "device_kind": device_kind,
        "device_count": str(len(devices)),
        "local_device_count": str(len(local_devices)),
        "process_count": str(process_count),
        "execution_mode": "data_parallel"
        if len(local_devices) > 1
        else "single_device",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run = _effective_run_config(config, args.smoke_test)
    config = config.model_copy(update={"run": run})

    if run.xla_cache_dir is not None:
        run.xla_cache_dir.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(run.xla_cache_dir))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    output_dir = run.output_dir / "bc_tpu_phase0"
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_logging_dirs(output_dir)

    dataset = ExportDataset(config.data.export_root)
    train_shards = dataset.shards("train")
    validation_shards = dataset.shards("validation")
    model = _build_model(config)
    runtime = _runtime_contract(config, dataset)
    runtime_info = _runtime_info(config)

    global_step = 0
    start_epoch = 0
    start_batch_index = 0
    best_validation: dict[str, float] | None = None
    if config.run.resume_from is not None:
        template_state = _initial_state(config, model, dataset)
        restored = restore_checkpoint(
            config.run.resume_from,
            item={
                "params": template_state.params,
                "opt_state": template_state.opt_state,
            },
        )
        resume_state = read_resume_state(config.run.resume_from)
        if resume_state.runtime != runtime:
            raise ValueError(
                f"resume runtime mismatch: checkpoint={resume_state.runtime} current={runtime}"
            )
        state = restore_train_state(
            model.apply,
            restored["params"],
            restored["opt_state"],
            resume_state.global_step,
            config.optimizer.learning_rate,
            config.optimizer.min_learning_rate,
            config.optimizer.weight_decay,
            config.optimizer.grad_clip_norm,
            config.optimizer.warmup_steps,
        )
        global_step = int(resume_state.global_step)
        start_epoch = resume_state.epoch
        start_batch_index = int(resume_state.next_batch_index)
        best_validation = _load_best_validation(config.run.resume_from)
    else:
        state = _initial_state(config, model, dataset)

    last_train_loss = math.nan
    training_log_path = output_dir / "training_log.jsonl"
    step_log_path = output_dir / "step_log.jsonl"
    local_device_count = int(runtime_info["local_device_count"])
    data_parallel = local_device_count > 1
    if data_parallel:
        state = jax_utils.replicate(state)

    for epoch in range(start_epoch, config.run.num_epochs):
        microbatch_size = int(config.run.microbatch_size or config.run.batch_size)
        train_batches = iter_train_epoch_batches(
            dataset,
            seed=config.run.seed,
            epoch=epoch,
            buffer_games=config.data.buffer_games,
            buffer_samples=config.data.buffer_samples,
            logical_batch_size=int(config.run.batch_size),
        )
        for batch_index, batch_np in enumerate(train_batches):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
            batch_np = apply_train_augmentation(batch_np)
            batch = _as_batch(batch_np)
            if data_parallel and _can_shard_batch(batch, local_device_count):
                if microbatch_size % local_device_count != 0:
                    raise ValueError(
                        f"microbatch_size {microbatch_size} must be divisible by local device count {local_device_count}"
                    )
                sharded_batch = _shard_batch(batch, local_device_count)
                state, loss, breakdown = train_logical_batch_data_parallel(
                    state,
                    sharded_batch,
                    model,
                    microbatch_size // local_device_count,
                )
                breakdown = jax_utils.unreplicate(breakdown)
            else:
                host_state = jax_utils.unreplicate(state) if data_parallel else state
                host_state, loss, breakdown = train_logical_batch(
                    host_state, batch, model, microbatch_size
                )
                state = jax_utils.replicate(host_state) if data_parallel else host_state
            global_step += 1
            last_train_loss = float(loss)
            if not math.isfinite(last_train_loss):
                raise ValueError(
                    f"non-finite train loss at step {global_step}: {last_train_loss}"
                )

            step_summary = {
                "manifest_fingerprint": dataset.manifest_fingerprint,
                "train_shards": len(train_shards),
                "validation_shards": len(validation_shards),
                "epoch": epoch,
                "global_step": global_step,
                "backend": runtime_info["backend"],
                "device_kind": runtime_info["device_kind"],
                "execution_mode": runtime_info["execution_mode"],
                "train_loss": last_train_loss,
                "policy_loss": float(breakdown.policy),
            }
            if global_step % config.run.log_every_n_steps == 0:
                append_jsonl(step_log_path, step_summary)

            if global_step % config.run.validate_every_n_steps == 0:
                host_state = jax_utils.unreplicate(state) if data_parallel else state
                validation = _validate(
                    host_state,
                    model,
                    validation_shards,
                    _validation_microbatch_size(config),
                    config.run.max_validation_samples,
                )
                if not math.isfinite(validation["total_loss"]):
                    raise ValueError(
                        f"non-finite validation loss at step {global_step}: {validation}"
                    )
                summary = {
                    "manifest_fingerprint": dataset.manifest_fingerprint,
                    "train_shards": len(train_shards),
                    "validation_shards": len(validation_shards),
                    "epoch": epoch,
                    "global_step": global_step,
                    "backend": runtime_info["backend"],
                    "device_kind": runtime_info["device_kind"],
                    "execution_mode": runtime_info["execution_mode"],
                    "train_loss": last_train_loss,
                    "policy_loss": float(breakdown.policy),
                    "validation_total_loss": validation["total_loss"],
                    "validation_policy_loss": validation["policy_loss"],
                    "validation_agreement": validation["agreement"],
                    "validation_samples": validation["samples"],
                }
                _save_latest(
                    output_dir,
                    jax_utils.unreplicate(state) if data_parallel else state,
                    summary,
                    ResumeState(
                        epoch=epoch,
                        global_step=global_step,
                        next_batch_index=batch_index + 1,
                        runtime=runtime,
                    ),
                )
                append_jsonl(training_log_path, summary)
                if _is_better_validation(validation, best_validation):
                    best_validation = validation
                    _save_latest(
                        output_dir / "best",
                        jax_utils.unreplicate(state) if data_parallel else state,
                        summary,
                        ResumeState(
                            epoch=epoch,
                            global_step=global_step,
                            next_batch_index=batch_index + 1,
                            runtime=runtime,
                        ),
                    )

            if global_step % config.run.checkpoint_every_n_steps == 0:
                summary = {
                    "manifest_fingerprint": dataset.manifest_fingerprint,
                    "train_shards": len(train_shards),
                    "validation_shards": len(validation_shards),
                    "epoch": epoch,
                    "global_step": global_step,
                    "backend": runtime_info["backend"],
                    "device_kind": runtime_info["device_kind"],
                    "execution_mode": runtime_info["execution_mode"],
                    "train_loss": last_train_loss,
                    "policy_loss": float(breakdown.policy),
                }
                _save_latest(
                    output_dir,
                    jax_utils.unreplicate(state) if data_parallel else state,
                    summary,
                    ResumeState(
                        epoch=epoch,
                        global_step=global_step,
                        next_batch_index=batch_index + 1,
                        runtime=runtime,
                    ),
                )

        if (epoch + 1) % config.run.validation_every_n_epochs == 0:
            host_state = jax_utils.unreplicate(state) if data_parallel else state
            validation = _validate(
                host_state,
                model,
                validation_shards,
                _validation_microbatch_size(config),
                config.run.max_validation_samples,
            )
            if not math.isfinite(validation["total_loss"]):
                raise ValueError(
                    f"non-finite validation loss at epoch {epoch}: {validation}"
                )
            summary = {
                "manifest_fingerprint": dataset.manifest_fingerprint,
                "train_shards": len(train_shards),
                "validation_shards": len(validation_shards),
                "epoch": epoch,
                "global_step": global_step,
                "backend": runtime_info["backend"],
                "device_kind": runtime_info["device_kind"],
                "execution_mode": runtime_info["execution_mode"],
                "train_loss": last_train_loss,
                "validation_total_loss": validation["total_loss"],
                "validation_policy_loss": validation["policy_loss"],
                "validation_agreement": validation["agreement"],
                "validation_samples": validation["samples"],
            }
            append_jsonl(training_log_path, summary)
            if _is_better_validation(validation, best_validation):
                best_validation = validation
                _save_latest(
                    output_dir / "best",
                    host_state,
                    summary,
                    ResumeState(
                        epoch=epoch + 1,
                        global_step=global_step,
                        next_batch_index=0,
                        runtime=runtime,
                    ),
                )
            _save_latest(
                output_dir,
                host_state,
                summary,
                ResumeState(
                    epoch=epoch + 1,
                    global_step=global_step,
                    next_batch_index=0,
                    runtime=runtime,
                ),
            )

    host_state = jax_utils.unreplicate(state) if data_parallel else state
    final_validation = _validate(
        host_state,
        model,
        validation_shards,
        _validation_microbatch_size(config),
        config.run.max_validation_samples,
    )
    summary = {
        "manifest_fingerprint": dataset.manifest_fingerprint,
        "train_shards": len(train_shards),
        "validation_shards": len(validation_shards),
        "backend": runtime_info["backend"],
        "device_kind": runtime_info["device_kind"],
        "execution_mode": runtime_info["execution_mode"],
        "global_step": global_step,
        "train_loss": last_train_loss,
        "validation_total_loss": final_validation["total_loss"],
        "validation_policy_loss": final_validation["policy_loss"],
        "validation_agreement": final_validation["agreement"],
        "validation_samples": final_validation["samples"],
    }

    if not math.isfinite(summary["train_loss"]) or not math.isfinite(
        summary["validation_total_loss"]
    ):
        raise ValueError(f"non-finite final summary: {summary}")

    print(summary)
    append_jsonl(training_log_path, summary)
    _save_latest(
        output_dir,
        host_state,
        summary,
        ResumeState(
            epoch=int(config.run.num_epochs),
            global_step=global_step,
            next_batch_index=0,
            runtime=runtime,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
