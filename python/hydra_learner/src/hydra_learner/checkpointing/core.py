"""Data-only checkpoint contract for the experimental PyTorch learner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn as nn

from hydra_learner.checkpointing.ema import _cpu_ema_state, _load_ema_resume_state, _model_state_with_ema_weights
from hydra_learner.checkpointing.rng import capture_rng_state, restore_rng_state
from hydra_learner.checkpointing.schema import (
    CHECKPOINT_SCHEMA_VERSION,
    ENCODER_SHAPE,
    HEAD_MODE,
    _checkpoint_optimizer_config_for_resume,
    _expect_equal,
    _normalize_expected_loss_weights,
    _normalize_loss_weights,
    _normalize_target_contract_for_weights,
    _validate_checkpoint_root,
    _validate_json_payload,
    manifest_digest,
    target_contract_from_manifest,
)
from hydra_learner.model import (
    ACTION_SPACE,
    BACKBONE_PROFILE_DEFAULT,
    CONV_MEMORY_FORMAT_DEFAULT,
    RESIDUAL_PROFILE_DEFAULT,
    HydraPolicyNet,
)

if TYPE_CHECKING:
    from hydra_learner.model.losses import LossWeights
    from hydra_learner.typing_boundaries import JsonObject, TorchCheckpointPayload

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "EmaConfig",
    "EmaResumeState",
    "InitOnlyState",
    "ModelConfig",
    "OptimizerConfig",
    "ResumeState",
    "RuntimeConfig",
    "capture_rng_state",
    "load_checkpoint",
    "load_checkpoint_init_only",
    "load_checkpoint_metadata",
    "manifest_digest",
    "restore_rng_state",
    "save_checkpoint",
    "target_contract_from_manifest",
]


@dataclass(frozen=True)
class ModelConfig:
    hidden: int
    blocks: int
    bottleneck: int
    actions: int = ACTION_SPACE
    residual_profile: str = RESIDUAL_PROFILE_DEFAULT
    backbone_profile: str = BACKBONE_PROFILE_DEFAULT
    conv_memory_format: str = CONV_MEMORY_FORMAT_DEFAULT
    head_mode: str = HEAD_MODE
    encoder_shape: tuple[int, int] = ENCODER_SHAPE


@dataclass(frozen=True)
class OptimizerConfig:
    name: Literal["AdamW"]
    lr: float
    min_lr: float
    lr_schedule: Literal["constant", "cosine"] = "cosine"
    lr_warmup_steps: int = 0
    schedule_total_steps: int | None = None
    target_games: int | None = None
    grad_clip_norm: float | None = None
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    foreach: bool | None = None
    fused: bool | None = None


EmaDevice = Literal["auto", "cuda", "cpu"]


@dataclass(frozen=True)
class EmaConfig:
    enabled: bool = False
    decay: float = 0.999
    start_step: int = 0
    update_every_steps: int = 1
    device: EmaDevice = "auto"


@dataclass(frozen=True)
class RuntimeConfig:
    variant: str
    loss_mode: str
    precision_mode: str
    compile_fullgraph_check: bool
    compile_dry_run_mode: str = "snapshot_restore_first_batch"
    warmup_mode: str = "non_mutating_replay_first_batch"


@dataclass(frozen=True)
class EmaResumeState:
    state_dict: dict[str, torch.Tensor]
    update_count: int
    last_update_step: int = 0


@dataclass(frozen=True)
class ResumeState:
    global_step: int
    samples_seen: int
    raw_mjai_progress: dict[str, int]
    ema: EmaResumeState | None = None


def _raw_progress_counters(raw_progress: object) -> dict[str, int]:
    if not isinstance(raw_progress, dict):
        raise ValueError("checkpoint raw_mjai_progress must be a dict")
    counters: dict[str, int] = {}
    for key, value in raw_progress.items():
        if isinstance(value, bool):
            raise ValueError(f"checkpoint raw_mjai_progress counter {key!r} must be int-like")
        if isinstance(value, int):
            counters[str(key)] = value
        elif isinstance(value, dict):
            continue
        else:
            raise ValueError(f"checkpoint raw_mjai_progress counter {key!r} must be int-like")
    return counters


def _checkpoint_int(checkpoint: TorchCheckpointPayload, key: str) -> int:
    value = checkpoint[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"checkpoint {key} must be an int")
    return value


def _checkpoint_mapping(checkpoint: TorchCheckpointPayload, key: str) -> Mapping[str, object]:
    value = checkpoint[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {key} must be a dict")
    return value


@dataclass(frozen=True)
class InitOnlyState:
    model_config: JsonObject
    weight_source: Literal["raw", "ema"]
    global_step: int
    samples_seen: int
    manifest: JsonObject


def save_checkpoint(
    path: Path,
    *,
    model: HydraPolicyNet,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    runtime_config: RuntimeConfig,
    loss_weights: LossWeights,
    manifest_path: Path | None,
    global_step: int,
    samples_seen: int,
    raw_mjai_progress: Mapping[str, object] | None = None,
    ema_config: EmaConfig | None = None,
    ema_state: dict[str, torch.Tensor] | None = None,
    ema_update_count: int = 0,
    ema_last_update_step: int = 0,
    weight_source: Literal["raw", "ema"] = "raw",
    training_objective: Mapping[str, object] | None = None,
    target_contract: Mapping[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "torch_version": f"{torch.__version__}",
        "cuda_version": torch.version.cuda,
        "device": _device_metadata(),
        "model_config": asdict(model_config),
        "loss_weights": asdict(loss_weights),
        "optimizer_config": asdict(optimizer_config),
        "optimizer_state": optimizer.state_dict(),
        "model_state": model.state_dict(),
        "rng_state": capture_rng_state(),
        "manifest": {
            "path": None if manifest_path is None else str(manifest_path),
            "digest_sha256": manifest_digest(manifest_path),
        },
        "global_step": global_step,
        "samples_seen": samples_seen,
        "raw_mjai_progress": dict[str, int]() if raw_mjai_progress is None else dict(raw_mjai_progress),
        "compile": asdict(runtime_config),
    }
    normalized_target_contract = _normalize_target_contract_for_weights(target_contract, loss_weights)
    if normalized_target_contract is not None:
        checkpoint["target_contract"] = normalized_target_contract
    if training_objective is not None:
        _validate_json_payload(training_objective, "training_objective")
        checkpoint["training_objective"] = dict(training_objective)
    if ema_config is not None:
        checkpoint["ema_config"] = asdict(ema_config)
        checkpoint["ema_state"] = {} if ema_state is None else _cpu_ema_state(ema_state)
        checkpoint["ema_update_count"] = ema_update_count
        checkpoint["ema_last_update_step"] = ema_last_update_step
    checkpoint["weight_source"] = weight_source
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    *,
    model: HydraPolicyNet,
    optimizer: torch.optim.Optimizer,
    expected_model_config: ModelConfig,
    expected_optimizer_config: OptimizerConfig,
    expected_runtime_config: RuntimeConfig,
    expected_loss_weights: LossWeights,
    expected_manifest_path: Path | None,
    expected_ema_config: EmaConfig | None = None,
    expected_target_contract: Mapping[str, object] | None = None,
) -> ResumeState:
    checkpoint = _torch_load(path)
    _validate_checkpoint_root(checkpoint)
    _expect_equal(checkpoint["model_config"], asdict(expected_model_config), "model_config")
    if expected_loss_weights.deltaq > 0.0:
        raise ValueError("delta_q_output_contract_missing")
    _expect_equal(
        _normalize_loss_weights(checkpoint["loss_weights"]),
        _normalize_expected_loss_weights(expected_loss_weights),
        "loss_weights",
    )
    expected_target_contract_normalized = _normalize_target_contract_for_weights(
        expected_target_contract, expected_loss_weights
    )
    actual_target_contract = checkpoint.get("target_contract")
    if expected_target_contract_normalized is not None:
        _expect_equal(actual_target_contract, expected_target_contract_normalized, "target_contract")
    elif actual_target_contract is not None:
        _validate_json_payload(actual_target_contract, "target_contract")
    _expect_equal(
        _checkpoint_optimizer_config_for_resume(checkpoint["optimizer_config"], expected_optimizer_config),
        asdict(expected_optimizer_config),
        "optimizer_config",
    )
    _expect_equal(checkpoint["compile"], asdict(expected_runtime_config), "compile")
    _expect_equal(
        checkpoint["manifest"],
        {
            "path": None if expected_manifest_path is None else str(expected_manifest_path),
            "digest_sha256": manifest_digest(expected_manifest_path),
        },
        "manifest",
    )
    weight_source = checkpoint.get("weight_source", "raw")
    if weight_source != "raw":
        raise ValueError(f"checkpoint weight_source must be 'raw' for resume, got {weight_source!r}")
    mutable_checkpoint = cast("dict[str, Any]", checkpoint)
    ema_resume = _load_ema_resume_state(mutable_checkpoint, expected_ema_config, model)
    _load_model_state_strict(model, checkpoint["model_state"])
    optimizer.load_state_dict(mutable_checkpoint["optimizer_state"])
    restore_rng_state(mutable_checkpoint["rng_state"])
    raw_progress = checkpoint.get("raw_mjai_progress", {})
    return ResumeState(
        global_step=_checkpoint_int(checkpoint, "global_step"),
        samples_seen=_checkpoint_int(checkpoint, "samples_seen"),
        raw_mjai_progress=_raw_progress_counters(raw_progress),
        ema=ema_resume,
    )


def load_checkpoint_metadata(path: Path) -> ResumeState:
    checkpoint = _torch_load(path)
    _validate_checkpoint_root(checkpoint)
    raw_progress = checkpoint.get("raw_mjai_progress", {})
    return ResumeState(
        global_step=_checkpoint_int(checkpoint, "global_step"),
        samples_seen=_checkpoint_int(checkpoint, "samples_seen"),
        raw_mjai_progress=_raw_progress_counters(raw_progress),
        ema=None,
    )


def load_checkpoint_init_only(
    path: Path,
    *,
    model: HydraPolicyNet,
    expected_model_config: ModelConfig,
    weight_source: Literal["raw", "ema"] = "raw",
) -> InitOnlyState:
    checkpoint = _torch_load(path)
    _validate_checkpoint_root(checkpoint)
    _expect_equal(checkpoint["model_config"], asdict(expected_model_config), "model_config")
    checkpoint_weight_source = checkpoint.get("weight_source", "raw")
    if checkpoint_weight_source not in {"raw", "ema"}:
        raise ValueError(f"checkpoint weight_source must be 'raw' or 'ema', got {checkpoint_weight_source!r}")
    if weight_source == "raw":
        state = checkpoint["model_state"]
    else:
        if "ema_state" not in checkpoint:
            raise ValueError("checkpoint ema_state missing for init-only EMA load")
        state = _model_state_with_ema_weights(checkpoint["model_state"], checkpoint["ema_state"], model)
    _load_model_state_strict(model, state)
    manifest = checkpoint["manifest"]
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be a dict")
    return InitOnlyState(
        model_config=dict(_checkpoint_mapping(checkpoint, "model_config")),
        weight_source=weight_source,
        global_step=_checkpoint_int(checkpoint, "global_step"),
        samples_seen=_checkpoint_int(checkpoint, "samples_seen"),
        manifest=dict(manifest),
    )


def _torch_load(path: Path) -> TorchCheckpointPayload:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"failed to load checkpoint {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("checkpoint root must be a dict")
    return cast("TorchCheckpointPayload", obj)


def _load_model_state_strict(model: nn.Module, state: object) -> None:
    if not isinstance(state, dict):
        raise ValueError("checkpoint model_state must be a dict")
    current = model.state_dict()
    state_dict = cast("dict[str, torch.Tensor]", state)
    if set(state_dict) != set(current):
        missing = sorted(set(current).difference(state_dict))
        extra = sorted(set(state_dict).difference(current))
        raise ValueError(f"model_state keys mismatch: missing={missing} extra={extra}")
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"model_state[{key}] is not a tensor")
        expected = current[key]
        if tensor.shape != expected.shape:
            raise ValueError(
                f"model_state[{key}] shape mismatch: got {tuple(tensor.shape)} expected {tuple(expected.shape)}"
            )
        if tensor.dtype != expected.dtype:
            raise ValueError(f"model_state[{key}] dtype mismatch: got {tensor.dtype} expected {expected.dtype}")
    model.load_state_dict(state_dict, strict=True)


def _device_metadata() -> JsonObject:
    if not torch.cuda.is_available():
        return {"type": "cpu", "name": None, "capability": None}
    return {
        "type": "cuda",
        "name": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
    }
