"""Minimal real-model PPO train step for Phase 1A."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from hydra_learner.model import ACTION_SPACE, OBS_CHANNELS, TILE_WIDTH
from hydra_learner.rl import (
    EntropyController,
    PpoLossConfig,
    default_entropy_target_fraction,
    legal_count_bucket_means,
    masked_entropy,
    masked_log_prob,
    ppo_loss,
)


@dataclass(frozen=True)
class PpoBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    legal_mask: torch.Tensor
    old_logprob: torch.Tensor
    value_old: torch.Tensor
    raw_advantages: torch.Tensor
    returns: torch.Tensor
    bc_logits: torch.Tensor
    legal_count: torch.Tensor
    player_id: torch.Tensor | None = None
    seat_id: torch.Tensor | None = None
    game_id: torch.Tensor | None = None
    turn: torch.Tensor | None = None
    rank_utility_used: str | None = None

    def validate(self) -> None:
        batch = _require_obs(self.obs)
        _require_vector(self.actions, batch, torch.int64, "actions")
        _require_vector(self.old_logprob, batch, torch.float32, "old_logprob", finite=True)
        _require_vector(self.value_old, batch, torch.float32, "value_old", finite=True)
        _require_vector(self.raw_advantages, batch, torch.float32, "raw_advantages", finite=True)
        _require_vector(self.returns, batch, torch.float32, "returns", finite=True)
        _require_vector(self.legal_count, batch, torch.int64, "legal_count")
        if self.legal_mask.shape != (batch, ACTION_SPACE):
            raise ValueError(f"legal_mask must have shape [B,{ACTION_SPACE}]")
        if self.legal_mask.dtype is not torch.bool:
            raise TypeError("legal_mask must be bool")
        if self.bc_logits.shape != (batch, ACTION_SPACE):
            raise ValueError(f"bc_logits must have shape [B,{ACTION_SPACE}]")
        if self.bc_logits.dtype is not torch.float32:
            raise TypeError("bc_logits must be float32")
        _require_finite(self.bc_logits, "bc_logits")
        if not bool(self.legal_mask.any(dim=1).all()):
            raise ValueError("legal_mask has an all-illegal row")
        expected_legal_count = self.legal_mask.sum(dim=1).to(dtype=torch.int64)
        if not torch.equal(self.legal_count, expected_legal_count):
            raise ValueError("legal_count must equal legal_mask.sum(dim=1)")
        if not bool(((self.actions >= 0) & (self.actions < ACTION_SPACE)).all()):
            raise ValueError("actions must be in action range")
        selected_legal = self.legal_mask.gather(1, self.actions.unsqueeze(1)).squeeze(1)
        if not bool(selected_legal.all()):
            raise ValueError("actions must be legal")
        _require_optional_vector(self.player_id, batch, torch.int64, "player_id", minimum=0, maximum=3)
        _require_optional_vector(self.seat_id, batch, torch.int64, "seat_id", minimum=0, maximum=3)
        _require_optional_vector(self.game_id, batch, torch.int64, "game_id")
        _require_optional_vector(self.turn, batch, torch.int64, "turn", minimum=0)


@dataclass(frozen=True)
class PpoTrainStepConfig:
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    bc_kl_reverse_coef: float = 0.0
    grad_clip_norm: float | None = 0.5
    advantage_epsilon: float = 1.0e-8


@dataclass(frozen=True)
class PpoTrainStepResult:
    metrics: dict[str, Any]
    entropy_controller: EntropyController


def ppo_train_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: PpoBatch,
    entropy_controller: EntropyController,
    config: PpoTrainStepConfig,
) -> PpoTrainStepResult:
    batch.validate()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch.obs)
    values = outputs.value.squeeze(-1)
    loss_config = PpoLossConfig(
        clip_epsilon=config.clip_epsilon,
        value_coef=config.value_coef,
        entropy_alpha=entropy_controller.alpha,
        bc_kl_reverse_coef=config.bc_kl_reverse_coef,
        advantage_epsilon=config.advantage_epsilon,
    )
    loss_out = ppo_loss(
        outputs.policy_logits,
        values,
        batch.actions,
        batch.legal_mask,
        batch.old_logprob,
        batch.raw_advantages,
        batch.returns,
        bc_logits=batch.bc_logits,
        config=loss_config,
    )
    loss_value = float(loss_out.total.detach())
    if not math.isfinite(loss_value):
        raise RuntimeError(f"non-finite PPO loss: {loss_value}")
    loss_out.total.backward()
    grad_norm = _total_grad_norm(model.parameters())
    if config.grad_clip_norm is not None and config.grad_clip_norm > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        grad_norm = float(grad_norm_tensor.detach())
    if not math.isfinite(grad_norm):
        raise RuntimeError(f"non-finite PPO grad norm: {grad_norm}")
    optimizer.step()

    with torch.no_grad():
        entropy_per_row = masked_entropy(outputs.policy_logits.detach(), batch.legal_mask)
        ratio = (
            masked_log_prob(outputs.policy_logits.detach(), batch.legal_mask, batch.actions) - batch.old_logprob
        ).exp()
        clip_mask = ((ratio < 1.0 - config.clip_epsilon) | (ratio > 1.0 + config.clip_epsilon)).to(dtype=torch.float32)
    next_controller = entropy_controller.update_default(entropy_per_row, batch.legal_count)
    metrics: dict[str, Any] = {
        "loss_total": loss_value,
        "loss_policy": float(loss_out.metrics.policy_loss),
        "loss_value": float(loss_out.metrics.value_loss),
        "entropy": float(loss_out.metrics.entropy),
        "entropy_alpha_before": entropy_controller.alpha,
        "entropy_alpha_after": next_controller.alpha,
        "bc_kl_reverse": float(loss_out.metrics.bc_kl_reverse),
        "approx_kl_old": float(loss_out.metrics.approx_kl_old),
        "clip_fraction": float(loss_out.metrics.clip_fraction),
        "ratio_mean": float(loss_out.metrics.ratio_mean),
        "explained_variance": float(loss_out.metrics.explained_variance),
        "advantage_raw_mean": float(loss_out.metrics.advantage_raw_mean),
        "advantage_raw_std": float(loss_out.metrics.advantage_raw_std),
        "advantage_normalized_mean": float(loss_out.metrics.advantage_normalized_mean),
        "advantage_normalized_std": float(loss_out.metrics.advantage_normalized_std),
        "grad_norm": grad_norm,
        "illegal_action_count": 0,
        "legal_count_bucket_entropy": legal_count_bucket_means(entropy_per_row, batch.legal_mask),
        "legal_count_bucket_clip_fraction": legal_count_bucket_means(clip_mask, batch.legal_mask),
        "entropy_target_fraction_mean": float(default_entropy_target_fraction(batch.legal_count).mean()),
    }
    _validate_json_safe_metrics(metrics)
    return PpoTrainStepResult(metrics=metrics, entropy_controller=next_controller)


def _require_obs(obs: torch.Tensor) -> int:
    if obs.ndim != 3 or obs.shape[1:] != (OBS_CHANNELS, TILE_WIDTH):
        raise ValueError(f"obs must have shape [B,{OBS_CHANNELS},{TILE_WIDTH}]")
    if obs.dtype is not torch.float32:
        raise TypeError("obs must be float32")
    _require_finite(obs, "obs")
    return obs.shape[0]


def _require_vector(tensor: torch.Tensor, batch: int, dtype: torch.dtype, name: str, *, finite: bool = False) -> None:
    if tensor.shape != (batch,):
        raise ValueError(f"{name} must have shape [B]")
    if tensor.dtype is not dtype:
        raise TypeError(f"{name} must be {dtype}")
    if finite:
        _require_finite(tensor, name)


def _require_optional_vector(
    tensor: torch.Tensor | None,
    batch: int,
    dtype: torch.dtype,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if tensor is None:
        return
    _require_vector(tensor, batch, dtype, name)
    if minimum is not None and not bool((tensor >= minimum).all()):
        raise ValueError(f"{name} below minimum")
    if maximum is not None and not bool((tensor <= maximum).all()):
        raise ValueError(f"{name} above maximum")


def _total_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for parameter in parameters:
        if not isinstance(parameter, torch.nn.Parameter) or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        param_norm = float(torch.linalg.vector_norm(grad))
        total_sq += param_norm * param_norm
    return math.sqrt(total_sq)


def _validate_json_safe_metrics(value: object, path: str = "metrics") -> None:
    if isinstance(value, bool | str) or value is None:
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str | int):
                raise TypeError(f"{path} key must be str or int")
            _validate_json_safe_metrics(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_safe_metrics(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains non-JSON metric type {type(value).__name__}")


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
