"""Data-only checkpoint contract for the experimental PyTorch learner."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn

from hydra_learner.model import (
    ACTION_SPACE,
    BACKBONE_PROFILE_DEFAULT,
    CONV_MEMORY_FORMAT_DEFAULT,
    OBS_CHANNELS,
    RESIDUAL_PROFILE_DEFAULT,
    TILE_WIDTH,
    HydraPolicyNet,
)

if TYPE_CHECKING:
    from hydra_learner.losses import LossWeights


TARGET_CONTRACT_SEMANTICS = {
    "exit": "exit_root_child_visits_v1",
    "delta_q": "delta_q_child_minus_root_v1",
}
TARGET_CONTRACT_PROVENANCE = "search-derived"

CHECKPOINT_SCHEMA_VERSION = 1
ENCODER_SHAPE = (OBS_CHANNELS, TILE_WIDTH)
HEAD_MODE = "base_plus_optional_oracle_safety"


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


@dataclass(frozen=True)
class InitOnlyState:
    model_config: dict[str, Any]
    weight_source: Literal["raw", "ema"]
    global_step: int
    samples_seen: int
    manifest: dict[str, Any]


def manifest_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def target_contract_from_manifest(manifest_path: Path | None, weights: LossWeights) -> dict[str, object] | None:
    lanes: list[tuple[str, str]] = []
    if weights.exit > 0.0:
        lanes.append(("exit", "exit_sidecar"))
    if weights.deltaq > 0.0:
        raise ValueError("delta_q_output_contract_missing")
    if not lanes:
        return None
    if manifest_path is None:
        raise ValueError("target_contract metadata requires compact shard manifest")
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, Mapping):
        raise ValueError("target_contract metadata requires manifest object")
    digest = manifest_digest(manifest_path)
    contract: dict[str, object] = {}
    for lane, manifest_key in lanes:
        sidecar = manifest.get(manifest_key)
        if not isinstance(sidecar, Mapping):
            raise ValueError(f"target_contract.{lane} requires {manifest_key} manifest metadata")
        path = sidecar.get("path")
        source_net_hash = sidecar.get("source_net_hash")
        source_version = sidecar.get("source_version")
        if not isinstance(path, str) or path == "":
            raise ValueError(f"target_contract.{lane}.sidecar_path is required")
        if not isinstance(source_net_hash, int):
            raise ValueError(f"target_contract.{lane}.source_net_hash is required")
        if not isinstance(source_version, int):
            raise ValueError(f"target_contract.{lane}.source_version is required")
        lane_contract: dict[str, object] = {
            "lane": lane,
            "sidecar_path": path,
            "source_net_hash": source_net_hash,
            "source_version": source_version,
            "semantics": TARGET_CONTRACT_SEMANTICS[lane],
            "provenance": TARGET_CONTRACT_PROVENANCE,
            "manifest_path": str(manifest_path),
            "manifest_digest_sha256": digest,
            "coverage_fraction": 1.0,
        }
        _validate_target_lane_contract(lane, lane_contract)
        contract[lane] = lane_contract
    return contract


def _numpy_rng_state() -> tuple[Any, ...]:
    return cast("tuple[Any, ...]", np.random.get_state())


def capture_rng_state() -> dict[str, Any]:
    np_name, np_keys, np_pos, np_has_gauss, np_cached_gaussian = _numpy_rng_state()
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": {
            "name": np_name,
            "keys": torch.from_numpy(np_keys.astype(np.uint32, copy=False).copy()),
            "pos": np_pos,
            "has_gauss": np_has_gauss,
            "cached_gaussian": np_cached_gaussian,
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_random"])
    numpy_state = state["numpy_random"]
    np.random.set_state(
        (
            numpy_state["name"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = state.get("torch_cuda", [])
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)


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
    raw_mjai_progress: dict[str, int] | None = None,
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
    ema_resume = _load_ema_resume_state(checkpoint, expected_ema_config, model)
    _load_model_state_strict(model, checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    restore_rng_state(checkpoint["rng_state"])
    raw_progress = checkpoint.get("raw_mjai_progress", {})
    if not isinstance(raw_progress, dict):
        raise ValueError("checkpoint raw_mjai_progress must be a dict")
    return ResumeState(
        global_step=int(checkpoint["global_step"]),
        samples_seen=int(checkpoint["samples_seen"]),
        raw_mjai_progress={str(key): int(value) for key, value in raw_progress.items()},
        ema=ema_resume,
    )


def load_checkpoint_metadata(path: Path) -> ResumeState:
    checkpoint = _torch_load(path)
    _validate_checkpoint_root(checkpoint)
    raw_progress = checkpoint.get("raw_mjai_progress", {})
    if not isinstance(raw_progress, dict):
        raise ValueError("checkpoint raw_mjai_progress must be a dict")
    return ResumeState(
        global_step=int(checkpoint["global_step"]),
        samples_seen=int(checkpoint["samples_seen"]),
        raw_mjai_progress={str(key): int(value) for key, value in raw_progress.items()},
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
        model_config=dict[str, Any](checkpoint["model_config"]),
        weight_source=weight_source,
        global_step=int(checkpoint["global_step"]),
        samples_seen=int(checkpoint["samples_seen"]),
        manifest=dict[str, Any](manifest),
    )


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"failed to load checkpoint {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("checkpoint root must be a dict")
    return cast("dict[str, Any]", obj)


def _validate_checkpoint_root(checkpoint: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "torch_version",
        "cuda_version",
        "device",
        "model_config",
        "loss_weights",
        "optimizer_config",
        "optimizer_state",
        "model_state",
        "rng_state",
        "manifest",
        "global_step",
        "samples_seen",
        "compile",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint missing keys: {sorted(missing)}")
    if checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema_version mismatch: {checkpoint['schema_version']!r}")
    _normalize_loss_weights(checkpoint["loss_weights"])


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


def _cpu_ema_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().to(device="cpu", dtype=torch.float32) for key, tensor in state.items()}


def _load_ema_resume_state(
    checkpoint: dict[str, Any], expected: EmaConfig | None, model: nn.Module
) -> EmaResumeState | None:
    has_ema = "ema_config" in checkpoint or "ema_state" in checkpoint
    if expected is None:
        if has_ema:
            raise ValueError("checkpoint ema_config mismatch: EMA state present but EMA disabled")
        return None
    if not has_ema:
        raise ValueError("checkpoint ema_config mismatch: EMA state missing")
    _expect_equal(checkpoint.get("ema_config"), asdict(expected), "ema_config")
    state = checkpoint.get("ema_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint ema_state must be a dict")
    current = model.state_dict()
    ema_state = cast("dict[str, torch.Tensor]", state)
    param_keys = {key for key, tensor in current.items() if tensor.is_floating_point()}
    if set(ema_state) != param_keys:
        missing = sorted(param_keys.difference(ema_state))
        extra = sorted(set(ema_state).difference(param_keys))
        raise ValueError(f"ema_state keys mismatch: missing={missing} extra={extra}")
    for key, tensor in ema_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"ema_state[{key}] is not a tensor")
        expected_tensor = current[key]
        if tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"ema_state[{key}] shape mismatch: got {tuple(tensor.shape)} expected {tuple(expected_tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(f"ema_state[{key}] dtype mismatch: got {tensor.dtype} expected torch.float32")
    return EmaResumeState(
        state_dict={key: tensor.detach().clone() for key, tensor in ema_state.items()},
        update_count=int(checkpoint.get("ema_update_count", 0)),
        last_update_step=int(checkpoint.get("ema_last_update_step", 0)),
    )


def _model_state_with_ema_weights(model_state: object, ema_state: object, model: nn.Module) -> dict[str, torch.Tensor]:
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint model_state must be a dict")
    if not isinstance(ema_state, dict):
        raise ValueError("checkpoint ema_state must be a dict")
    current = model.state_dict()
    result: dict[str, torch.Tensor] = {}
    typed_model_state = cast("dict[str, torch.Tensor]", model_state)
    typed_ema_state = cast("dict[str, torch.Tensor]", ema_state)
    for key, current_tensor in current.items():
        source = typed_ema_state.get(key) if current_tensor.is_floating_point() else typed_model_state.get(key)
        if source is None:
            raise ValueError(f"model_state keys mismatch: missing=['{key}'] extra=[]")
        result[key] = source.to(dtype=current_tensor.dtype)
    return result


def _checkpoint_optimizer_config_for_resume(actual: object, expected: OptimizerConfig) -> object:
    if not isinstance(actual, dict):
        return actual
    normalized = dict(actual)
    expected_dict = asdict(expected)
    normalized.setdefault("target_games", None)
    for key in ("foreach", "fused"):
        if normalized.get(key) is None and expected_dict.get(key) is not None:
            normalized[key] = expected_dict[key]
    for key in ("lr", "min_lr", "grad_clip_norm", "weight_decay", "beta1", "beta2", "eps"):
        expected_value = expected_dict.get(key)
        if (
            key in normalized
            and isinstance(normalized[key], float)
            and isinstance(expected_value, float)
            and abs(normalized[key] - expected_value) <= max(1.0e-12, abs(expected_value) * 1.0e-6)
        ):
            normalized[key] = expected_value
    return normalized


def _normalize_loss_weights(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint loss_weights must be an object")
    normalized = dict(value)
    normalized.setdefault("exit", 0.0)
    normalized.setdefault("deltaq", 0.0)
    return normalized


def _normalize_expected_loss_weights(weights: LossWeights) -> dict[str, object]:
    expected = asdict(weights)
    expected.setdefault("exit", 0.0)
    expected.setdefault("deltaq", 0.0)
    return expected


def _normalize_target_contract_for_weights(
    value: Mapping[str, object] | None, weights: LossWeights
) -> dict[str, object] | None:
    lanes: list[str] = []
    if weights.exit > 0.0:
        lanes.append("exit")
    if weights.deltaq > 0.0:
        raise ValueError("delta_q_output_contract_missing")
    if not lanes:
        if value is not None:
            _validate_json_payload(value, "target_contract")
        return None if value is None else dict(value)
    if value is None:
        raise ValueError("target_contract metadata is required when target loss weights are positive")
    _validate_json_payload(value, "target_contract")
    contract = dict(value)
    for lane in lanes:
        lane_value = contract.get(lane)
        if not isinstance(lane_value, Mapping):
            raise ValueError(f"target_contract.{lane} metadata is required")
        _validate_target_lane_contract(lane, lane_value)
    return contract


def _validate_target_lane_contract(lane: str, value: Mapping[str, object]) -> None:
    required = {
        "lane",
        "sidecar_path",
        "source_net_hash",
        "source_version",
        "semantics",
        "provenance",
        "manifest_path",
        "manifest_digest_sha256",
        "coverage_fraction",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(f"target_contract.{lane} missing keys: {sorted(missing)}")
    if value["lane"] != lane:
        raise ValueError(f"target_contract.{lane}.lane mismatch")
    if value["semantics"] != TARGET_CONTRACT_SEMANTICS[lane]:
        raise ValueError(f"target_contract.{lane}.semantics mismatch")
    if value["provenance"] != TARGET_CONTRACT_PROVENANCE:
        raise ValueError(f"target_contract.{lane}.provenance mismatch")
    coverage = value["coverage_fraction"]
    if (
        not isinstance(coverage, int | float)
        or not math.isfinite(float(coverage))
        or not (0.0 < float(coverage) <= 1.0)
    ):
        raise ValueError(f"target_contract.{lane}.coverage_fraction must be in (0, 1]")
    _validate_sidecar_tuple(value, lane)


def _validate_sidecar_tuple(value: Mapping[str, object], lane: str) -> None:
    tuple_keys = ("sidecar_path", "source_net_hash", "source_version")
    present = [value.get(key) is not None for key in tuple_keys]
    if any(present) and not all(present):
        raise ValueError(f"target_contract.{lane} sidecar tuple must have path/hash/version all present or all absent")


def _validate_json_payload(value: object, path: str) -> None:
    if isinstance(value, bool | str) or value is None:
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            _validate_json_payload(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _validate_json_payload(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


def _expect_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise ValueError(f"checkpoint {name} mismatch: got {actual!r} expected {expected!r}")


def _device_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"type": "cpu", "name": None, "capability": None}
    return {
        "type": "cuda",
        "name": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
    }
