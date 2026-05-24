"""Phase-0 masked PPO/GAE tensor utilities.

These helpers are deliberately pure tensor/list code. They do not own rollout I/O,
checkpoint resume, or self-play orchestration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from hydra_learner.model import ACTION_SPACE

PLACEMENT_UTILITY_DEFAULT: tuple[float, float, float, float] = (1.0, 0.3, -0.3, -1.0)
PLACEMENT_UTILITY_TENHOU_RANK_1_06_02_NEG1: tuple[float, float, float, float] = (1.0, 0.6, 0.2, -1.0)
DEFAULT_GAE_GAMMA = 0.995
DEFAULT_GAE_LAMBDA = 0.95
_MASKED_LOGIT_SENTINEL = -1.0e9


@dataclass(frozen=True)
class PlayerDecisionStep:
    player_id: int
    value_old: float
    truncation_bootstrap_value: float | None = None


@dataclass(frozen=True)
class PlayerLocalGae:
    rewards: torch.Tensor
    raw_advantages: torch.Tensor
    returns: torch.Tensor
    terminal_player_stream: torch.Tensor
    truncation: torch.Tensor


@dataclass(frozen=True)
class EntropyController:
    alpha: float
    beta: float
    alpha_max: float

    def update(
        self, observed_entropy: torch.Tensor, legal_count: torch.Tensor, target_fraction: float | torch.Tensor
    ) -> EntropyController:
        target = entropy_target(legal_count, target_fraction).to(
            dtype=observed_entropy.dtype, device=observed_entropy.device
        )
        delta = float((target - observed_entropy).mean().detach())
        alpha = min(max(self.alpha + self.beta * delta, 0.0), self.alpha_max)
        return EntropyController(alpha=alpha, beta=self.beta, alpha_max=self.alpha_max)

    def update_default(self, observed_entropy: torch.Tensor, legal_count: torch.Tensor) -> EntropyController:
        return self.update(observed_entropy, legal_count, default_entropy_target_fraction(legal_count))


@dataclass(frozen=True)
class PpoLossConfig:
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_alpha: float = 0.0
    bc_kl_reverse_coef: float = 0.0
    advantage_epsilon: float = 1.0e-8


@dataclass(frozen=True)
class PpoLossMetrics:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    ratio_mean: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl_old: torch.Tensor
    bc_kl_reverse: torch.Tensor
    explained_variance: torch.Tensor
    advantage_raw_mean: torch.Tensor
    advantage_raw_std: torch.Tensor
    advantage_normalized_mean: torch.Tensor
    advantage_normalized_std: torch.Tensor


@dataclass(frozen=True)
class PpoLossOutput:
    total: torch.Tensor
    metrics: PpoLossMetrics


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")


def _require_action_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, actions]")
    if logits.shape[1] != ACTION_SPACE:
        raise ValueError(f"logits action dimension must be {ACTION_SPACE}")
    if legal_mask.shape != logits.shape:
        raise ValueError("legal_mask shape must match logits")
    if legal_mask.dtype is not torch.bool:
        raise TypeError("legal_mask must be bool")
    _require_finite(logits, "logits")
    if not bool(legal_mask.any(dim=1).all()):
        raise ValueError("legal_mask has an all-illegal row")
    return legal_mask


def _require_actions(actions: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if actions.ndim != 1 or actions.shape[0] != legal_mask.shape[0]:
        raise ValueError("actions must have shape [batch]")
    action_ids = actions.to(dtype=torch.int64)
    if not bool(((action_ids >= 0) & (action_ids < legal_mask.shape[1])).all()):
        raise ValueError("actions must be in action range")
    selected_legal = legal_mask.gather(1, action_ids.unsqueeze(1)).squeeze(1)
    if not bool(selected_legal.all()):
        raise ValueError("selected action is illegal")
    return action_ids


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    mask = _require_action_logits(logits, legal_mask)
    masked = logits.masked_fill(~mask, _MASKED_LOGIT_SENTINEL)
    return F.log_softmax(masked, dim=1).masked_fill(~mask, _MASKED_LOGIT_SENTINEL)


def masked_probs(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    mask = _require_action_logits(logits, legal_mask)
    probs = masked_log_softmax(logits, mask).exp()
    return probs.masked_fill(~mask, 0.0)


def masked_log_prob(logits: torch.Tensor, legal_mask: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    mask = _require_action_logits(logits, legal_mask)
    action_ids = _require_actions(actions, mask)
    return masked_log_softmax(logits, mask).gather(1, action_ids.unsqueeze(1)).squeeze(1)


def masked_entropy(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    mask = _require_action_logits(logits, legal_mask)
    log_probs = masked_log_softmax(logits, mask)
    probs = log_probs.exp().masked_fill(~mask, 0.0)
    return -(probs * log_probs.masked_fill(~mask, 0.0)).sum(dim=1)


def masked_kl(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    direction: Literal["current_to_reference", "reference_to_current"] = "current_to_reference",
) -> torch.Tensor:
    mask = _require_action_logits(current_logits, legal_mask)
    if reference_logits.shape != current_logits.shape:
        raise ValueError("reference_logits shape must match current_logits")
    current_log = masked_log_softmax(current_logits, mask)
    reference_log = masked_log_softmax(reference_logits, mask)
    if direction == "current_to_reference":
        base_log, other_log = current_log, reference_log
    elif direction == "reference_to_current":
        base_log, other_log = reference_log, current_log
    else:
        raise ValueError(f"unsupported KL direction {direction!r}")
    base_prob = base_log.exp().masked_fill(~mask, 0.0)
    return (base_prob * (base_log - other_log).masked_fill(~mask, 0.0)).sum(dim=1)


def entropy_fraction(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    mask = _require_action_logits(logits, legal_mask)
    counts = mask.sum(dim=1)
    entropy = masked_entropy(logits, mask)
    denom = counts.to(dtype=entropy.dtype).log()
    return torch.where(counts > 1, entropy / denom, torch.zeros_like(entropy))


def entropy_target(legal_count: torch.Tensor, target_fraction: float | torch.Tensor) -> torch.Tensor:
    counts = legal_count.to(dtype=torch.float32)
    if not bool((counts >= 1).all()):
        raise ValueError("legal_count must be >= 1")
    if isinstance(target_fraction, torch.Tensor):
        if target_fraction.shape != legal_count.shape:
            raise ValueError("target_fraction tensor shape must match legal_count")
        fraction = target_fraction.to(dtype=torch.float32, device=counts.device)
    else:
        if target_fraction < 0.0:
            raise ValueError("target_fraction must be >= 0")
        fraction = torch.full_like(counts, target_fraction)
    if not bool((fraction >= 0.0).all()):
        raise ValueError("target_fraction must be >= 0")
    return fraction * counts.log()


def default_entropy_target_fraction(legal_count: torch.Tensor) -> torch.Tensor:
    counts = legal_count.to(dtype=torch.int64)
    if not bool((counts >= 1).all()):
        raise ValueError("legal_count must be >= 1")
    return torch.where(
        counts <= 4,
        torch.full_like(counts, 0.40, dtype=torch.float32),
        torch.full_like(counts, 0.70, dtype=torch.float32),
    )


def normalize_advantages(advantages: torch.Tensor, epsilon: float = 1.0e-8) -> torch.Tensor:
    mean = advantages.mean()
    variance = ((advantages - mean) ** 2).mean()
    return (advantages - mean) / torch.sqrt(variance + epsilon)


def compute_gae_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    has_next: torch.Tensor,
    *,
    gamma: float = DEFAULT_GAE_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim != 1 or values.shape != rewards.shape or next_values.shape != rewards.shape:
        raise ValueError("rewards, values, and next_values must have shape [steps]")
    if has_next.shape != rewards.shape:
        raise ValueError("has_next must have shape [steps]")
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1]")
    if not (0.0 < gae_lambda <= 1.0):
        raise ValueError("gae_lambda must be in (0, 1]")
    mask = has_next.to(dtype=values.dtype)
    deltas = rewards + gamma * mask * next_values - values
    advantages = torch.empty_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(rewards.shape[0] - 1, -1, -1):
        running = deltas[index] + gamma * gae_lambda * mask[index] * running
        advantages[index] = running
    return advantages, advantages + values


def compute_player_local_gae(
    steps: Sequence[PlayerDecisionStep],
    *,
    final_placements: Sequence[int] | None,
    placement_utility: Sequence[float] = PLACEMENT_UTILITY_DEFAULT,
    gamma: float = DEFAULT_GAE_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
) -> PlayerLocalGae:
    if any(step.player_id < 0 or step.player_id >= 4 for step in steps):
        raise ValueError("player_id must be in 0..3")
    if final_placements is not None:
        if len(final_placements) != 4:
            raise ValueError("final_placements must contain four placements")
        if any(placement < 0 or placement >= 4 for placement in final_placements):
            raise ValueError("final placement must be in 0..3")

    if len(placement_utility) != 4:
        raise ValueError("placement_utility must contain four values")
    rewards = torch.zeros(len(steps), dtype=torch.float32)
    values = torch.tensor([step.value_old for step in steps], dtype=torch.float32)
    next_values = torch.zeros(len(steps), dtype=torch.float32)
    has_next = torch.zeros(len(steps), dtype=torch.bool)
    terminal_player_stream = torch.zeros(len(steps), dtype=torch.bool)
    truncation = torch.zeros(len(steps), dtype=torch.bool)

    for player in range(4):
        indices = [index for index, step in enumerate(steps) if step.player_id == player]
        for local, index in enumerate(indices):
            if local + 1 < len(indices):
                has_next[index] = True
                next_values[index] = values[indices[local + 1]]
                continue
            step = steps[index]
            if step.truncation_bootstrap_value is not None:
                has_next[index] = True
                truncation[index] = True
                next_values[index] = step.truncation_bootstrap_value
                continue
            terminal_player_stream[index] = True
            if final_placements is None:
                raise ValueError("final_placements required for terminal player streams")
            placement = final_placements[player]
            if placement < 0 or placement >= 4:
                raise ValueError("final placement must be in 0..3")
            rewards[index] = placement_utility[placement]

    raw_advantages = torch.empty_like(rewards)
    returns = torch.empty_like(rewards)
    for player in range(4):
        indices = [index for index, step in enumerate(steps) if step.player_id == player]
        if not indices:
            continue
        player_advantages, player_returns = compute_gae_returns(
            rewards[indices],
            values[indices],
            next_values[indices],
            has_next[indices],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        raw_advantages[indices] = player_advantages
        returns[indices] = player_returns
    return PlayerLocalGae(
        rewards=rewards,
        raw_advantages=raw_advantages,
        returns=returns,
        terminal_player_stream=terminal_player_stream,
        truncation=truncation,
    )


def explained_variance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_variance = ((target - target.mean()) ** 2).mean()
    if float(target_variance.detach()) == 0.0:
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return 1.0 - (((target - prediction) ** 2).mean() / target_variance)


def ppo_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
    old_logprob: torch.Tensor,
    raw_advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    bc_logits: torch.Tensor | None = None,
    config: PpoLossConfig | None = None,
) -> PpoLossOutput:
    if config is None:
        config = PpoLossConfig()
    if config.clip_epsilon <= 0.0:
        raise ValueError("clip_epsilon must be > 0")
    _require_finite(policy_logits, "policy_logits")
    _require_finite(values, "values")
    _require_finite(old_logprob, "old_logprob")
    _require_finite(raw_advantages, "raw_advantages")
    _require_finite(returns, "returns")
    if bc_logits is not None:
        _require_finite(bc_logits, "bc_logits")
    new_logprob = masked_log_prob(policy_logits, legal_mask, actions)
    flat_values = values.squeeze(-1)
    for name, tensor in (
        ("old_logprob", old_logprob),
        ("raw_advantages", raw_advantages),
        ("returns", returns),
        ("values", flat_values),
    ):
        if tensor.shape != new_logprob.shape:
            raise ValueError(f"{name} must have shape [batch]")
    advantages = (
        normalize_advantages(raw_advantages, config.advantage_epsilon) if raw_advantages.numel() > 2 else raw_advantages
    )
    ratio = (new_logprob - old_logprob).exp()
    _require_finite(ratio, "ratio")
    clipped_ratio = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
    policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = value_mse(flat_values, returns).mean()
    entropy = masked_entropy(policy_logits, legal_mask).mean()
    if bc_logits is None:
        bc_kl_reverse = torch.zeros((), dtype=policy_logits.dtype, device=policy_logits.device)
    else:
        bc_kl_reverse = masked_kl(policy_logits, bc_logits, legal_mask, direction="current_to_reference").mean()
    total = (
        policy_loss
        + config.value_coef * value_loss
        + config.bc_kl_reverse_coef * bc_kl_reverse
        - config.entropy_alpha * entropy
    )
    clip_fraction = (
        ((ratio < 1.0 - config.clip_epsilon) | (ratio > 1.0 + config.clip_epsilon)).to(dtype=policy_logits.dtype).mean()
    )
    raw_mean = raw_advantages.mean()
    raw_std = torch.sqrt(((raw_advantages - raw_mean) ** 2).mean())
    norm_mean = advantages.mean()
    norm_std = torch.sqrt(((advantages - norm_mean) ** 2).mean())
    return PpoLossOutput(
        total=total,
        metrics=PpoLossMetrics(
            policy_loss=policy_loss.detach(),
            value_loss=value_loss.detach(),
            entropy=entropy.detach(),
            ratio_mean=ratio.mean().detach(),
            clip_fraction=clip_fraction.detach(),
            approx_kl_old=(old_logprob - new_logprob).mean().detach(),
            bc_kl_reverse=bc_kl_reverse.detach(),
            explained_variance=explained_variance(flat_values.detach(), returns.detach()).detach(),
            advantage_raw_mean=raw_mean.detach(),
            advantage_raw_std=raw_std.detach(),
            advantage_normalized_mean=norm_mean.detach(),
            advantage_normalized_std=norm_std.detach(),
        ),
    )


def value_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    return diff * diff * 0.5


def legal_count_bucket_means(metric: torch.Tensor, legal_mask: torch.Tensor) -> dict[int, float]:
    if metric.ndim != 1 or metric.shape[0] != legal_mask.shape[0]:
        raise ValueError("metric must have shape [batch]")
    counts = legal_mask.to(dtype=torch.bool).sum(dim=1)
    result: dict[int, float] = {}
    for count in torch.unique(counts.detach().cpu(), sorted=True).tolist():
        selected = counts == int(count)
        result[int(count)] = float(metric[selected].mean().detach().cpu())
    return result
