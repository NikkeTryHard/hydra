"""Torch-only masked PPO math; native code owns row identity, order, and storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from hydra2.contracts.common import ContractError


@dataclass(frozen=True, slots=True)
class PpoParameters:
    clip_eps: float
    eps_std: float
    w_value: float
    w_bc: float
    alpha: float


class PpoTerms(NamedTuple):
    loss: torch.Tensor
    ratio_sum: torch.Tensor
    clip_count: torch.Tensor
    entropy_sum: torch.Tensor
    value_squared_sum: torch.Tensor
    bc_kl_sum: torch.Tensor
    approximate_kl_sum: torch.Tensor
    residual_sum_by_seat: torch.Tensor
    residual_squared_sum_by_seat: torch.Tensor
    return_sum_by_seat: torch.Tensor
    return_squared_sum_by_seat: torch.Tensor


def _assert_tensor(condition: torch.Tensor, message: str) -> None:
    # Device assertions poison invalid work without a per-microstep host sync;
    # CPU assertions become typed errors, while CUDA failures surface at fences.
    try:
        torch._assert_async(condition, message)
    except RuntimeError as exc:
        raise ContractError(message) from exc


def prepare_advantages(raw_advantage: torch.Tensor, *, eps_std: float) -> torch.Tensor:
    """Freeze complete-minibatch FP32 population normalization before slicing."""
    if raw_advantage.ndim != 1 or raw_advantage.numel() == 0:
        raise ContractError("PPO requires a nonempty complete advantage vector")
    if not 0.0 < eps_std < float("inf"):
        raise ContractError("PPO eps_std must be positive and finite")
    raw = raw_advantage.detach().to(torch.float32)
    _assert_tensor(torch.isfinite(raw).all(), "PPO advantages must be finite")
    centered = raw - raw.mean()
    return centered / torch.sqrt(centered.square().mean() + eps_std)


def ppo_micro_terms(
    policy_logits: torch.Tensor,
    value_vector: torch.Tensor,
    legal_mask: torch.Tensor,
    selected_actions: torch.Tensor,
    old_selected_logp: torch.Tensor,
    normalized_advantage: torch.Tensor,
    return_vector: torch.Tensor,
    *,
    minibatch_count: int,
    parameters: PpoParameters,
    bc_logits: torch.Tensor | None = None,
) -> PpoTerms:
    """Sum exact numerators against the true optimizer count, including tails.

    ``normalized_advantage`` must come from the complete optimizer minibatch;
    normalizing separately here would change both the objective and gradients.
    """
    if policy_logits.ndim != 2:
        raise ContractError("PPO policy logits must have shape [B,A]")
    batch, actions = policy_logits.shape
    if batch == 0 or actions == 0 or isinstance(minibatch_count, bool):
        raise ContractError("PPO requires positive row and action counts")
    if minibatch_count < batch:
        raise ContractError("PPO microbatch exceeds the true optimizer count")
    if legal_mask.shape != policy_logits.shape or legal_mask.dtype != torch.bool:
        raise ContractError("PPO legal mask must be bool [B,A]")
    if value_vector.shape != (batch, 4) or return_vector.shape != (batch, 4):
        raise ContractError("PPO values and returns must retain all four seats")
    if selected_actions.shape != (batch,) or selected_actions.dtype != torch.int64:
        raise ContractError("PPO selected actions must be int64 [B]")
    if any(tensor.shape != (batch,) for tensor in (old_selected_logp, normalized_advantage)):
        raise ContractError("PPO selected log probabilities and advantages must be [B]")
    if not 0.0 < parameters.clip_eps < 1.0:
        raise ContractError("PPO clip_eps must be strictly between zero and one")
    if not all(
        0.0 <= weight < float("inf")
        for weight in (parameters.w_value, parameters.w_bc, parameters.alpha)
    ):
        raise ContractError("PPO objective weights must be finite and nonnegative")
    if (parameters.w_bc > 0.0) != (bc_logits is not None):
        raise ContractError("PPO BC logits are required exactly when w_bc is positive")
    if bc_logits is not None and bc_logits.shape != policy_logits.shape:
        raise ContractError("PPO BC logits must share the policy action geometry")

    _assert_tensor(legal_mask.any(dim=1).all(), "PPO rejects all-illegal rows")
    _assert_tensor(
        ((selected_actions >= 0) & (selected_actions < actions)).all(),
        "PPO selected action is outside the action vocabulary",
    )
    chosen = selected_actions.unsqueeze(1)
    _assert_tensor(legal_mask.gather(1, chosen).all(), "PPO selected action must be legal")
    logits = policy_logits.to(torch.float32)
    values = value_vector.to(torch.float32)
    targets = return_vector.detach().to(torch.float32)
    old_logp = old_selected_logp.detach().to(torch.float32)
    advantage = normalized_advantage.detach().to(torch.float32)
    old_probability = old_logp.exp()
    _assert_tensor(
        torch.isfinite(logits).all()
        & torch.isfinite(values).all()
        & torch.isfinite(targets).all()
        & torch.isfinite(old_logp).all()
        & torch.isfinite(advantage).all()
        & (old_probability > 0.0).all()
        & (old_probability <= 1.0).all(),
        "PPO requires finite facts and representable old probabilities in (0,1]",
    )
    logp = logits.masked_fill(~legal_mask, -torch.inf).log_softmax(dim=1)
    selected_logp = logp.gather(1, chosen).squeeze(1)
    log_ratio = selected_logp - old_logp
    ratio = log_ratio.exp()
    _assert_tensor(torch.isfinite(ratio).all(), "PPO importance ratios must be finite")
    clipped_ratio = ratio.clamp(1.0 - parameters.clip_eps, 1.0 + parameters.clip_eps)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    probability = logp.exp()
    safe_logp = logp.masked_fill(~legal_mask, 0.0)
    entropy_sum = -(probability * safe_logp).sum()
    residual = values - targets
    value_squared_sum = residual.square().sum()
    if bc_logits is None:
        bc_kl_sum = logits.new_zeros(())
    else:
        reference = bc_logits.detach().to(torch.float32)
        _assert_tensor(torch.isfinite(reference).all(), "PPO BC logits must be finite")
        reference_logp = reference.masked_fill(~legal_mask, -torch.inf).log_softmax(dim=1)
        bc_kl_sum = (probability * (safe_logp - reference_logp.masked_fill(~legal_mask, 0.0))).sum()
    count = float(minibatch_count)
    loss = (
        -surrogate.sum() / count
        + parameters.w_value * value_squared_sum / (4.0 * count)
        + parameters.w_bc * bc_kl_sum / count
        - parameters.alpha * entropy_sum / count
    )
    return PpoTerms(
        loss=loss,
        ratio_sum=ratio.sum(),
        clip_count=(
            (ratio < 1.0 - parameters.clip_eps) | (ratio > 1.0 + parameters.clip_eps)
        ).sum(),
        entropy_sum=entropy_sum,
        value_squared_sum=value_squared_sum,
        bc_kl_sum=bc_kl_sum,
        approximate_kl_sum=(ratio - 1.0 - log_ratio).sum(),
        residual_sum_by_seat=residual.detach().sum(dim=0),
        residual_squared_sum_by_seat=residual.detach().square().sum(dim=0),
        return_sum_by_seat=targets.sum(dim=0),
        return_squared_sum_by_seat=targets.square().sum(dim=0),
    )
