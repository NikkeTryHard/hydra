"""Direct sampled Actor-Critic Hedge train step."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from hydra_learner.ppo.rl import (
    AchLossConfig,
    EntropyController,
    ach_loss,
    default_entropy_target_fraction,
    legal_count_bucket_means,
)
from hydra_learner.ppo.step import PpoBatch, _validate_json_safe_metrics


@dataclass(frozen=True)
class AchTrainStepConfig:
    eta: float = 1.0
    eps: float = 0.5
    l_th: float = 8.0
    pi_old_min: float = 1.0e-8
    advantage_epsilon: float = 1.0e-8
    value_coef: float = 0.5
    bc_kl_reverse_coef: float = 0.0
    grad_clip_norm: float | None = 0.5


@dataclass(frozen=True)
class AchTrainStepResult:
    metrics: dict[str, Any]
    entropy_controller: EntropyController


def ach_train_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: PpoBatch,
    entropy_controller: EntropyController,
    config: AchTrainStepConfig,
) -> AchTrainStepResult:
    batch.validate()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch.obs)
    values = outputs.value.squeeze(-1)
    loss_config = AchLossConfig(
        eta=config.eta,
        eps=config.eps,
        l_th=config.l_th,
        pi_old_min=config.pi_old_min,
        advantage_epsilon=config.advantage_epsilon,
        value_coef=config.value_coef,
        entropy_alpha=entropy_controller.alpha,
        bc_kl_reverse_coef=config.bc_kl_reverse_coef,
    )
    loss_out = ach_loss(
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
        raise RuntimeError(f"non-finite ACH loss: {loss_value}")
    loss_out.total.backward()
    grad_norm = _total_grad_norm(model.parameters())
    if config.grad_clip_norm is not None and config.grad_clip_norm > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        grad_norm = float(grad_norm_tensor.detach())
    if not math.isfinite(grad_norm):
        raise RuntimeError(f"non-finite ACH grad norm: {grad_norm}")
    optimizer.step()

    metrics: dict[str, Any] = loss_out.metric_dict()
    next_controller = entropy_controller.update_default(loss_out.entropy_per_row, batch.legal_count)
    metrics.update(
        {
            "loss_total": loss_value,
            "entropy_alpha_before": entropy_controller.alpha,
            "entropy_alpha_after": next_controller.alpha,
            "grad_norm": grad_norm,
            "illegal_action_count": 0,
            "legal_count_bucket_entropy": legal_count_bucket_means(loss_out.entropy_per_row, batch.legal_mask),
            "legal_count_bucket_gate_fraction": _bucket_optional_means(
                loss_out.gate_per_row, batch.legal_mask, torch.ones_like(loss_out.gate_per_row, dtype=torch.bool)
            ),
            "legal_count_bucket_pos_gate_fraction": _bucket_optional_means(
                loss_out.gate_per_row, batch.legal_mask, batch.raw_advantages >= 0.0
            ),
            "legal_count_bucket_neg_gate_fraction": _bucket_optional_means(
                loss_out.gate_per_row, batch.legal_mask, batch.raw_advantages < 0.0
            ),
            "legal_count_bucket_ratio_clipped_fraction": _bucket_optional_means(
                loss_out.ratio_clipped_per_row,
                batch.legal_mask,
                torch.ones_like(loss_out.ratio_clipped_per_row, dtype=torch.bool),
            ),
            "legal_count_bucket_bc_kl": legal_count_bucket_means(loss_out.bc_kl_per_row, batch.legal_mask),
            "entropy_fraction_mean": float(loss_out.metrics.entropy_fraction_mean),
            "entropy_target_fraction_mean": float(default_entropy_target_fraction(batch.legal_count).mean()),
        }
    )
    _validate_json_safe_metrics(metrics)
    return AchTrainStepResult(metrics=metrics, entropy_controller=next_controller)


def _total_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for parameter in parameters:
        if not isinstance(parameter, torch.nn.Parameter) or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        param_norm = float(torch.linalg.vector_norm(grad))
        total_sq += param_norm * param_norm
    return math.sqrt(total_sq)


def _bucket_optional_means(
    metric: torch.Tensor, legal_mask: torch.Tensor, selector: torch.Tensor
) -> dict[int, float | None]:
    if metric.ndim != 1 or metric.shape[0] != legal_mask.shape[0] or selector.shape != metric.shape:
        raise ValueError("bucket metric and selector must have shape [batch]")
    counts = legal_mask.to(dtype=torch.bool).sum(dim=1)
    result: dict[int, float | None] = {}
    for count in torch.unique(counts.detach().cpu(), sorted=True).tolist():
        selected = (counts == int(count)) & selector
        result[int(count)] = float(metric[selected].mean().detach().cpu()) if bool(selected.any()) else None
    return result
