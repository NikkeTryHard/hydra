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
LOSS_HEADS = (
    "policy",
    "value",
    "score_pdf",
    "score_cdf",
    "tenpai",
    "grp",
    "oracle_critic",
    "safety_residual",
    "opp_next",
    "danger",
)


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


def masked_policy_ce_indices(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask.to(dtype=torch.bool), MASKED_LOGIT_SENTINEL)
    return F.cross_entropy(masked, target.to(dtype=torch.int64), reduction="none")


def policy_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if target.ndim == 1:
        return masked_policy_ce_indices(logits, target, mask)
    return masked_policy_ce(logits, target, mask)


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


def active_loss_heads(weights: LossWeights | None = None, loss_mode: str = "full_base") -> tuple[str, ...]:
    if weights is None:
        weights = DEFAULT_LOSS_WEIGHTS
    if loss_mode == "policy_only":
        return ("policy",)
    heads = ["policy"]
    if weights.value > 0.0:
        heads.append("value")
    if weights.score > 0.0:
        heads.extend(("score_pdf", "score_cdf"))
    if weights.tenpai > 0.0:
        heads.append("tenpai")
    if weights.grp > 0.0:
        heads.append("grp")
    if weights.oracle_critic > 0.0:
        heads.append("oracle_critic")
    if weights.safety_residual > 0.0:
        heads.append("safety_residual")
    if weights.opp_next > 0.0:
        heads.append("opp_next")
    if weights.danger > 0.0:
        heads.append("danger")
    return tuple(heads)


def loss_breakdown_dict(
    breakdown: LossBreakdown, weights: LossWeights | None = None, loss_mode: str = "full_base"
) -> dict[str, float]:
    return {head: float(getattr(breakdown, head).detach()) for head in active_loss_heads(weights, loss_mode)}


def _required_coverage(tensor: torch.Tensor, elements_per_sample: int) -> tuple[str, float]:
    present = tensor.detach().to(dtype=torch.float32)
    if elements_per_sample == 1:
        positive = present.reshape(present.shape[0], -1).any(dim=1)
    else:
        positive = present.reshape(present.shape[0], elements_per_sample).sum(dim=1) > 0.0
    return "present_positive" if bool(positive.any()) else "present_zero", float(
        positive.to(dtype=torch.float32).mean()
    )


def _optional_mask_coverage(mask: torch.Tensor | None, elements_per_sample: int) -> tuple[str, float]:
    if mask is None:
        return "absent", 0.0
    return _required_coverage(mask, elements_per_sample)


def target_coverage_dict(
    targets: BaseTargets, weights: LossWeights | None = None, loss_mode: str = "full_base"
) -> dict[str, dict[str, float | str]]:
    active = set(active_loss_heads(weights, loss_mode))
    coverage: dict[str, dict[str, float | str]] = {}

    def add(head: str, status: str, fraction: float) -> None:
        coverage[head] = {"active": head in active, "status": status, "fraction": fraction}

    add("policy", "present_positive", 1.0)
    add("value", "present_positive", 1.0)
    add("score_pdf", *_required_coverage(targets.score_pdf_target, targets.score_pdf_target.shape[1]))
    add("score_cdf", *_required_coverage(targets.score_cdf_target, targets.score_cdf_target.shape[1]))
    add("tenpai", "present_positive", 1.0)
    add("grp", *_required_coverage(targets.grp_target, targets.grp_target.shape[1]))
    add("oracle_critic", *_optional_mask_coverage(targets.oracle_target_mask, 1))
    safety_width = 1 if targets.safety_mask is None else targets.safety_mask.shape[1]
    add("safety_residual", *_optional_mask_coverage(targets.safety_mask, safety_width))
    add(
        "opp_next",
        *_required_coverage(
            targets.opp_next_target, targets.opp_next_target.shape[1] * targets.opp_next_target.shape[2]
        ),
    )
    add("danger", *_required_coverage(targets.danger_mask, targets.danger_mask.shape[1] * targets.danger_mask.shape[2]))
    return coverage


def base_loss(outputs: HydraBaseOutput, targets: BaseTargets, weights: LossWeights | None = None) -> LossBreakdown:
    if weights is None:
        weights = DEFAULT_LOSS_WEIGHTS
    l_policy = policy_ce(outputs.policy_logits, targets.policy_target, targets.legal_mask).mean()
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
