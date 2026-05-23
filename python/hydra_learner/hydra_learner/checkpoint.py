"""Data-only checkpoint contract for the experimental PyTorch learner."""

from __future__ import annotations

import hashlib
import random
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
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    foreach: bool | None = None
    fused: bool | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    variant: str
    loss_mode: str
    precision_mode: str
    compile_fullgraph_check: bool


@dataclass(frozen=True)
class ResumeState:
    global_step: int
    samples_seen: int
    raw_mjai_progress: dict[str, int]


def manifest_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
) -> ResumeState:
    checkpoint = _torch_load(path)
    _validate_checkpoint_root(checkpoint)
    _expect_equal(checkpoint["model_config"], asdict(expected_model_config), "model_config")
    _expect_equal(checkpoint["loss_weights"], asdict(expected_loss_weights), "loss_weights")
    _expect_equal(checkpoint["optimizer_config"], asdict(expected_optimizer_config), "optimizer_config")
    _expect_equal(checkpoint["compile"], asdict(expected_runtime_config), "compile")
    _expect_equal(
        checkpoint["manifest"],
        {
            "path": None if expected_manifest_path is None else str(expected_manifest_path),
            "digest_sha256": manifest_digest(expected_manifest_path),
        },
        "manifest",
    )
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
