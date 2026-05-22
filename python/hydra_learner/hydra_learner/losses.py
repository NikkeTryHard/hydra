"""Rust-equivalent base BC losses for the experimental PyTorch learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from hydra_learner.model import HydraBaseOutput

MASKED_LOGIT_SENTINEL = -1.0e9


@dataclass(frozen=True)
class LossWeights:
    policy: float = 1.0
    value: float = 0.5
    grp: float = 0.2
    tenpai: float = 0.1
    danger: float = 0.1
    opp_next: float = 0.1
    score: float = 0.025
    oracle_critic: float = 0.0
    safety_residual: float = 0.0


DEFAULT_LOSS_WEIGHTS = LossWeights()


@dataclass(frozen=True)
class BaseTargets:
    policy_target: torch.Tensor
    legal_mask: torch.Tensor
    value_target: torch.Tensor
    grp_target: torch.Tensor
    tenpai_target: torch.Tensor
    danger_target: torch.Tensor
    danger_mask: torch.Tensor
    opp_next_target: torch.Tensor
    score_pdf_target: torch.Tensor
    score_cdf_target: torch.Tensor
    oracle_target: torch.Tensor | None = None
    oracle_target_mask: torch.Tensor | None = None
    safety_target: torch.Tensor | None = None
    safety_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    grp: torch.Tensor
    tenpai: torch.Tensor
    danger: torch.Tensor
    opp_next: torch.Tensor
    score_pdf: torch.Tensor
    score_cdf: torch.Tensor
    oracle_critic: torch.Tensor
    safety_residual: torch.Tensor


def masked_policy_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits + (torch.ones_like(mask, dtype=logits.dtype) - mask.to(dtype=logits.dtype)) * MASKED_LOGIT_SENTINEL
    log_probs = F.log_softmax(masked, dim=1)
    return -(target.to(dtype=logits.dtype) * log_probs).sum(dim=1)


def value_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred.squeeze(1) - target
    return diff * diff * 0.5


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target.to(dtype=logits.dtype) * F.log_softmax(logits, dim=1)).sum(dim=1)


def bce_logits_mean(logits: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target.to(dtype=logits.dtype), reduction="none").mean(dim=dim)


def danger_focal_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = target.to(dtype=logits.dtype)
    mask = mask.to(dtype=logits.dtype)
    alpha = 0.25
    gamma = 2.0
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = target * p + (torch.ones_like(target) - target) * (torch.ones_like(p) - p)
    focal = ((torch.ones_like(p_t) - p_t) ** gamma) * alpha * bce * mask
    return focal.sum(dim=(1, 2))


def opp_next_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    batch, opponents, tiles = logits.shape
    logits_flat = logits.reshape(batch * opponents, tiles)
    target_flat = target.reshape(batch * opponents, tiles).to(dtype=logits.dtype)
    per_opp = -(target_flat * F.log_softmax(logits_flat, dim=1)).sum(dim=1)
    return per_opp.reshape(batch, opponents).mean(dim=1)


def oracle_critic_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    centered = pred - pred.mean(dim=1, keepdim=True)
    mse = ((centered - target.to(dtype=pred.dtype)) ** 2).mean(dim=1) * 0.5
    zero_sum_penalty = (pred.sum(dim=1) ** 2) * 10.0
    per_sample = mse + zero_sum_penalty
    mask = mask.to(dtype=pred.dtype)
    return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)


def safety_residual_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=pred.dtype)
    sq = ((pred - target.to(dtype=pred.dtype)) ** 2) * 0.5
    return (sq * mask).sum() / mask.sum().clamp(min=1.0)


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _require_tensor(tensor: torch.Tensor | None, name: str) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"{name} is required when its loss weight is positive")
    return tensor


def base_loss(outputs: HydraBaseOutput, targets: BaseTargets, weights: LossWeights | None = None) -> LossBreakdown:
    if weights is None:
        weights = DEFAULT_LOSS_WEIGHTS
    l_policy = masked_policy_ce(outputs.policy_logits, targets.policy_target, targets.legal_mask).mean()
    l_value = value_mse(outputs.value, targets.value_target).mean()
    l_grp = soft_ce(outputs.grp, targets.grp_target).mean()
    l_tenpai = bce_logits_mean(outputs.opp_tenpai, targets.tenpai_target, dim=1).mean()
    l_danger = danger_focal_bce(outputs.danger, targets.danger_target, targets.danger_mask).mean()
    l_opp = opp_next_ce(outputs.opp_next_discard, targets.opp_next_target).mean()
    l_pdf = soft_ce(outputs.score_pdf, targets.score_pdf_target).mean()
    l_cdf = bce_logits_mean(outputs.score_cdf, targets.score_cdf_target, dim=1).mean()
    l_oracle = _zero_like_loss(l_policy)
    if weights.oracle_critic > 0.0:
        l_oracle = oracle_critic_loss(
            outputs.oracle_critic,
            _require_tensor(targets.oracle_target, "oracle_target"),
            _require_tensor(targets.oracle_target_mask, "oracle_target_mask"),
        )
    l_safety = _zero_like_loss(l_policy)
    if weights.safety_residual > 0.0:
        l_safety = safety_residual_loss(
            outputs.safety_residual,
            _require_tensor(targets.safety_target, "safety_target"),
            _require_tensor(targets.safety_mask, "safety_mask"),
        )
    total = (
        l_policy * weights.policy
        + l_value * weights.value
        + l_grp * weights.grp
        + l_tenpai * weights.tenpai
        + l_danger * weights.danger
        + l_opp * weights.opp_next
        + l_pdf * weights.score
        + l_cdf * weights.score
        + l_oracle * weights.oracle_critic
        + l_safety * weights.safety_residual
    )
    return LossBreakdown(
        total=total,
        policy=l_policy,
        value=l_value,
        grp=l_grp,
        tenpai=l_tenpai,
        danger=l_danger,
        opp_next=l_opp,
        score_pdf=l_pdf,
        score_cdf=l_cdf,
        oracle_critic=l_oracle,
        safety_residual=l_safety,
    )
