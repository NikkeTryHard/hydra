"""EMA checkpoint serialization and resume helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from hydra_learner.checkpoint import EmaConfig, EmaResumeState


class _EmaResumeState(NamedTuple):
    state_dict: dict[str, torch.Tensor]
    update_count: int
    last_update_step: int = 0


from hydra_learner.checkpoint_schema import _expect_equal


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
    return cast(
        "EmaResumeState",
        _EmaResumeState(
            state_dict={key: tensor.detach().clone() for key, tensor in ema_state.items()},
            update_count=int(checkpoint.get("ema_update_count", 0)),
            last_update_step=int(checkpoint.get("ema_last_update_step", 0)),
        ),
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
