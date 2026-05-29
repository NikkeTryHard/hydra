"""Minimal real-model PPO train step."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from hydra_learner.model import ACTION_SPACE, OBS_CHANNELS, TILE_WIDTH
from hydra_learner.rl import (
    EntropyController,
    PpoLossConfig,
    default_entropy_target_fraction,
    legal_count_bucket_means,
    masked_log_softmax,
    normalize_advantages,
    value_mse,
)


class PolicyValueModel(Protocol):
    def train(self, mode: bool = True) -> object: ...
    def parameters(self) -> Iterator[torch.nn.Parameter]: ...
    def policy_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


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
    microbatch_size: int | None = None


@dataclass(frozen=True)
class PpoTrainStepResult:
    metrics: dict[str, Any]
    entropy_controller: EntropyController


def ppo_train_step(
    *,
    model: PolicyValueModel,
    optimizer: torch.optim.Optimizer,
    batch: PpoBatch,
    entropy_controller: EntropyController,
    config: PpoTrainStepConfig,
) -> PpoTrainStepResult:
    if batch.obs.is_cuda:
        batch.validate()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_config = PpoLossConfig(
        clip_epsilon=config.clip_epsilon,
        value_coef=config.value_coef,
        entropy_alpha=entropy_controller.alpha,
        bc_kl_reverse_coef=config.bc_kl_reverse_coef,
        advantage_epsilon=config.advantage_epsilon,
    )
    batch_rows = batch.obs.shape[0]
    microbatch_size = config.microbatch_size or batch_rows
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be >= 1")
    advantages = (
        normalize_advantages(batch.raw_advantages, config.advantage_epsilon)
        if batch.raw_advantages.numel() > 2
        else batch.raw_advantages
    )
    model_device = next(model.parameters()).device
    batch_on_model_device = batch.obs.device == model_device
    loss_weighted: dict[str, torch.Tensor] = {}
    entropy_sum = torch.zeros((), dtype=torch.float32, device=model_device)
    clip_sum = torch.zeros((), dtype=torch.float32, device=model_device)
    value_sum_t = torch.zeros((), dtype=torch.float32, device=model_device)
    value_sq_sum_t = torch.zeros((), dtype=torch.float32, device=model_device)
    return_sum_t = torch.zeros((), dtype=torch.float32, device=model_device)
    return_sq_sum_t = torch.zeros((), dtype=torch.float32, device=model_device)
    value_return_sum_t = torch.zeros((), dtype=torch.float32, device=model_device)
    forward_backward_ms = 0.0
    h2d_ms = 0.0
    for start in range(0, batch_rows, microbatch_size):
        end = min(start + microbatch_size, batch_rows)
        row_slice = slice(start, end)
        row_count = end - start
        chunk_weight = row_count / batch_rows
        transfer_started = time.perf_counter()
        if batch_on_model_device:
            obs = batch.obs[row_slice]
            actions = batch.actions[row_slice]
            legal_mask = batch.legal_mask[row_slice]
            old_logprob = batch.old_logprob[row_slice]
            advantages_chunk = advantages[row_slice]
            returns = batch.returns[row_slice]
            bc_logits = batch.bc_logits[row_slice]
        else:
            obs = batch.obs[row_slice].to(model_device, non_blocking=True)
            actions = batch.actions[row_slice].to(model_device, non_blocking=True)
            legal_mask = batch.legal_mask[row_slice].to(model_device, non_blocking=True)
            old_logprob = batch.old_logprob[row_slice].to(model_device, non_blocking=True)
            advantages_chunk = advantages[row_slice].to(model_device, non_blocking=True)
            returns = batch.returns[row_slice].to(model_device, non_blocking=True)
            bc_logits = batch.bc_logits[row_slice].to(model_device, non_blocking=True)
        h2d_ms += (time.perf_counter() - transfer_started) * 1000.0
        chunk_started = time.perf_counter()
        policy_logits, value = model.policy_value(obs)
        values = value.squeeze(-1)
        loss_out = _ppo_loss_for_advantages(
            policy_logits,
            values,
            actions,
            legal_mask,
            old_logprob,
            advantages_chunk,
            returns,
            bc_logits=bc_logits,
            config=loss_config,
        )
        if not bool(torch.isfinite(loss_out["total"].detach()).item()):
            raise RuntimeError("non-finite PPO loss")
        (loss_out["total"] * chunk_weight).backward()
        forward_backward_ms += (time.perf_counter() - chunk_started) * 1000.0
        for key, value in loss_out.items():
            weighted = value.detach() * row_count
            loss_weighted[key] = weighted if key not in loss_weighted else loss_weighted[key] + weighted
        with torch.no_grad():
            entropy_sum = entropy_sum + loss_out["entropy"].detach() * row_count
            clip_sum = clip_sum + loss_out["clip_fraction"].detach() * row_count
            value_f = values.detach().to(dtype=torch.float32)
            returns_f = returns.detach().to(dtype=torch.float32)
            value_sum_t = value_sum_t + value_f.sum()
            value_sq_sum_t = value_sq_sum_t + (value_f * value_f).sum()
            return_sum_t = return_sum_t + returns_f.sum()
            return_sq_sum_t = return_sq_sum_t + (returns_f * returns_f).sum()
            value_return_sum_t = value_return_sum_t + (value_f * returns_f).sum()
    grad_started = time.perf_counter()
    if config.grad_clip_norm is not None and config.grad_clip_norm > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        grad_norm = float(grad_norm_tensor.detach())
    else:
        grad_norm = _total_grad_norm(model.parameters())
    if not math.isfinite(grad_norm):
        raise RuntimeError(f"non-finite PPO grad norm: {grad_norm}")
    optimizer_started = time.perf_counter()
    grad_clip_ms = (optimizer_started - grad_started) * 1000.0
    optimizer.step()
    optimizer_ms = (time.perf_counter() - optimizer_started) * 1000.0

    metrics_started = time.perf_counter()

    entropy_mean = float((entropy_sum / batch_rows).detach())
    clip_fraction = float((clip_sum / batch_rows).detach())
    values_mean = float((value_sum_t / batch_rows).detach())
    returns_mean = float((return_sum_t / batch_rows).detach())
    value_sq_mean = float((value_sq_sum_t / batch_rows).detach())
    return_sq_mean = float((return_sq_sum_t / batch_rows).detach())
    value_return_mean = float((value_return_sum_t / batch_rows).detach())
    values_var = max(0.0, value_sq_mean - values_mean * values_mean)
    returns_var = max(0.0, return_sq_mean - returns_mean * returns_mean)
    covariance = value_return_mean - values_mean * returns_mean
    explained_var = 0.0
    if returns_var > 0.0:
        explained_var = 1.0 - max(0.0, returns_var - 2.0 * covariance + values_var) / returns_var
    next_controller = entropy_controller.update_default(
        torch.full((batch_rows,), entropy_mean, dtype=torch.float32, device=batch.legal_count.device),
        batch.legal_count,
    )
    metric_means = {key: float((value / batch_rows).detach()) for key, value in loss_weighted.items()}
    raw_mean = batch.raw_advantages.mean()
    raw_std = torch.sqrt(((batch.raw_advantages - raw_mean) ** 2).mean())
    norm_mean = advantages.mean()
    norm_std = torch.sqrt(((advantages - norm_mean) ** 2).mean())
    metrics: dict[str, Any] = {
        "loss_total": metric_means["total"],
        "loss_policy": metric_means["policy_loss"],
        "loss_value": metric_means["value_loss"],
        "entropy": metric_means["entropy"],
        "entropy_alpha_before": entropy_controller.alpha,
        "entropy_alpha_after": next_controller.alpha,
        "bc_kl_reverse": metric_means["bc_kl_reverse"],
        "approx_kl_old": metric_means["approx_kl_old"],
        "clip_fraction": metric_means["clip_fraction"],
        "ratio_mean": metric_means["ratio_mean"],
        "explained_variance": explained_var,
        "legal_count_bucket_entropy": legal_count_bucket_means(
            torch.full((batch_rows,), entropy_mean, dtype=torch.float32, device=batch.legal_mask.device),
            batch.legal_mask,
        ),
        "legal_count_bucket_clip_fraction": legal_count_bucket_means(
            torch.full((batch_rows,), clip_fraction, dtype=torch.float32, device=batch.legal_mask.device),
            batch.legal_mask,
        ),
        "advantage_raw_mean": float(raw_mean.detach()),
        "advantage_raw_std": float(raw_std.detach()),
        "advantage_normalized_mean": float(norm_mean.detach()),
        "advantage_normalized_std": float(norm_std.detach()),
        "grad_norm": grad_norm,
        "illegal_action_count": 0,
        "entropy_target_fraction_mean": float(default_entropy_target_fraction(batch.legal_count).mean()),
        "microbatch_size": microbatch_size,
        "forward_backward_ms": forward_backward_ms,
        "h2d_ms": h2d_ms,
        "grad_clip_ms": grad_clip_ms,
        "optimizer_ms": optimizer_ms,
        "microbatch_count": math.ceil(batch_rows / microbatch_size),
        "post_metrics_ms": (time.perf_counter() - metrics_started) * 1000.0,
    }
    _validate_json_safe_metrics(metrics)
    return PpoTrainStepResult(metrics=metrics, entropy_controller=next_controller)


class _PpoLossMetricSums:
    def __init__(self) -> None:
        self._weighted: dict[str, float] = {}

    def add(self, loss_out: dict[str, torch.Tensor], row_count: int) -> None:
        for key, value in loss_out.items():
            self._weighted[key] = self._weighted.get(key, 0.0) + float(value.detach()) * row_count

    def means(self, total_rows: int) -> dict[str, float]:
        return {key: value / total_rows for key, value in self._weighted.items()}


def _ppo_loss_for_advantages(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    bc_logits: torch.Tensor,
    config: PpoLossConfig,
) -> dict[str, torch.Tensor]:
    if config.clip_epsilon <= 0.0:
        raise ValueError("clip_epsilon must be > 0")
    _require_finite(policy_logits, "policy_logits")
    _require_finite(values, "values")
    _require_finite(old_logprob, "old_logprob")
    _require_finite(advantages, "advantages")
    _require_finite(returns, "returns")
    _require_finite(bc_logits, "bc_logits")
    current_log_probs = masked_log_softmax(policy_logits, legal_mask)
    action_ids = actions.unsqueeze(1)
    new_logprob = current_log_probs.gather(1, action_ids).squeeze(1)
    flat_values = values.reshape(-1)
    for name, tensor in (
        ("old_logprob", old_logprob),
        ("advantages", advantages),
        ("returns", returns),
        ("values", flat_values),
    ):
        if tensor.shape != new_logprob.shape:
            raise ValueError(f"{name} must have shape [batch]")
    ratio = (new_logprob - old_logprob).exp()
    _require_finite(ratio, "ratio")
    clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = value_mse(flat_values, returns).mean()
    current_probs = current_log_probs.exp().masked_fill(~legal_mask, 0.0)
    safe_current_log = current_log_probs.masked_fill(~legal_mask, 0.0)
    entropy = -(current_probs * safe_current_log).sum(dim=1).mean()
    reference_log_probs = masked_log_softmax(bc_logits, legal_mask)
    bc_kl_reverse = (
        (current_probs * (safe_current_log - reference_log_probs.masked_fill(~legal_mask, 0.0))).sum(dim=1).mean()
    )
    total = (
        policy_loss
        + config.value_coef * value_loss
        + config.bc_kl_reverse_coef * bc_kl_reverse
        - config.entropy_alpha * entropy
    )
    clip_fraction = (
        ((ratio < 1.0 - config.clip_epsilon) | (ratio > 1.0 + config.clip_epsilon)).to(dtype=policy_logits.dtype).mean()
    )
    return {
        "total": total,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "ratio_mean": ratio.mean().detach(),
        "clip_fraction": clip_fraction.detach(),
        "approx_kl_old": (old_logprob - new_logprob).mean().detach(),
        "bc_kl_reverse": bc_kl_reverse.detach(),
    }


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
