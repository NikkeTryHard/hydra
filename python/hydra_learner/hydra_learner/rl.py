"""Masked PPO/GAE tensor utilities.

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
from hydra_learner.reward_shaping import RewardShapingConfig, apply_pbrs_reward

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
    phi_t: float | None = None
    phi_next: float | None = None
    terminal_next_phi: bool = False


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


@dataclass(frozen=True)
class AchLossConfig:
    eta: float = 1.0
    eps: float = 0.5
    l_th: float = 8.0
    pi_old_min: float = 1.0e-8
    advantage_epsilon: float = 1.0e-8
    value_coef: float = 0.5
    entropy_alpha: float = 0.0
    bc_kl_reverse_coef: float = 0.0


@dataclass(frozen=True)
class AchLossMetrics:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_min: torch.Tensor
    ratio_max: torch.Tensor
    ratio_clipped_fraction: torch.Tensor
    gate_fraction: torch.Tensor
    pos_gate_fraction: torch.Tensor | None
    neg_gate_fraction: torch.Tensor | None
    pi_old_clamp_fraction: torch.Tensor
    pi_old_min: torch.Tensor
    pi_old_raw_min: torch.Tensor
    approx_kl_old: torch.Tensor
    bc_kl_reverse: torch.Tensor
    entropy_fraction_mean: torch.Tensor
    advantage_raw_mean: torch.Tensor
    advantage_raw_std: torch.Tensor
    advantage_raw_rms: torch.Tensor
    advantage_scaled_mean: torch.Tensor
    advantage_scaled_std: torch.Tensor
    advantage_scaled_rms: torch.Tensor
    advantage_positive_count: torch.Tensor
    advantage_negative_count: torch.Tensor
    advantage_zero_count: torch.Tensor
    illegal_probability_max: torch.Tensor


@dataclass(frozen=True)
class AchLossOutput:
    total: torch.Tensor
    metrics: AchLossMetrics
    entropy_per_row: torch.Tensor
    gate_per_row: torch.Tensor
    ratio_clipped_per_row: torch.Tensor
    bc_kl_per_row: torch.Tensor

    def metric_dict(self) -> dict[str, float | int | None]:
        return {
            "loss_policy": float(self.metrics.policy_loss),
            "loss_value": float(self.metrics.value_loss),
            "entropy": float(self.metrics.entropy),
            "ratio_mean": float(self.metrics.ratio_mean),
            "ratio_min": float(self.metrics.ratio_min),
            "ratio_max": float(self.metrics.ratio_max),
            "ratio_clipped_fraction": float(self.metrics.ratio_clipped_fraction),
            "ach_gate_fraction": float(self.metrics.gate_fraction),
            "ach_pos_gate_fraction": _optional_float(self.metrics.pos_gate_fraction),
            "ach_neg_gate_fraction": _optional_float(self.metrics.neg_gate_fraction),
            "pi_old_clamp_fraction": float(self.metrics.pi_old_clamp_fraction),
            "pi_old_min": float(self.metrics.pi_old_min),
            "pi_old_raw_min": float(self.metrics.pi_old_raw_min),
            "approx_kl_old": float(self.metrics.approx_kl_old),
            "bc_kl_reverse": float(self.metrics.bc_kl_reverse),
            "entropy_fraction_mean": float(self.metrics.entropy_fraction_mean),
            "advantage_raw_mean": float(self.metrics.advantage_raw_mean),
            "advantage_raw_std": float(self.metrics.advantage_raw_std),
            "advantage_raw_rms": float(self.metrics.advantage_raw_rms),
            "advantage_scaled_mean": float(self.metrics.advantage_scaled_mean),
            "advantage_scaled_std": float(self.metrics.advantage_scaled_std),
            "advantage_scaled_rms": float(self.metrics.advantage_scaled_rms),
            "advantage_positive_count": int(self.metrics.advantage_positive_count),
            "advantage_negative_count": int(self.metrics.advantage_negative_count),
            "advantage_zero_count": int(self.metrics.advantage_zero_count),
            "illegal_probability_max": float(self.metrics.illegal_probability_max),
        }


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
    reward_shaping: RewardShapingConfig | None = None,
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
    shaping_config = reward_shaping
    shaping_enabled = shaping_config is not None and shaping_config.enabled and shaping_config.pbrs_beta > 0.0
    rewards = torch.zeros(len(steps), dtype=torch.float32)
    values = torch.tensor([step.value_old for step in steps], dtype=torch.float32)
    next_values = torch.zeros(len(steps), dtype=torch.float32)
    has_next = torch.zeros(len(steps), dtype=torch.bool)
    terminal_player_stream = torch.zeros(len(steps), dtype=torch.bool)
    truncation = torch.zeros(len(steps), dtype=torch.bool)
    phi_t = torch.empty(0, dtype=torch.float32)
    phi_next = torch.empty(0, dtype=torch.float32)
    terminal_next_phi = torch.empty(0, dtype=torch.bool)
    if shaping_enabled:
        phi_t = torch.zeros(len(steps), dtype=torch.float32)
        phi_next = torch.zeros(len(steps), dtype=torch.float32)
        terminal_next_phi = torch.zeros(len(steps), dtype=torch.bool)

    for player in range(4):
        indices = [index for index, step in enumerate(steps) if step.player_id == player]
        for local, index in enumerate(indices):
            if local + 1 < len(indices):
                has_next[index] = True
                next_index = indices[local + 1]
                next_values[index] = values[next_index]
                if shaping_enabled:
                    current_phi = steps[index].phi_t
                    if current_phi is not None:
                        phi_t[index] = current_phi
                    current_phi_next = steps[index].phi_next
                    next_phi_t = steps[next_index].phi_t
                    if current_phi_next is not None:
                        phi_next[index] = current_phi_next
                    elif next_phi_t is not None:
                        phi_next[index] = next_phi_t
                    terminal_next_phi[index] = steps[index].terminal_next_phi
                continue
            step = steps[index]
            if shaping_enabled:
                if step.phi_t is not None:
                    phi_t[index] = step.phi_t
                if step.phi_next is not None:
                    phi_next[index] = step.phi_next
                terminal_next_phi[index] = step.terminal_next_phi
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
            if shaping_enabled:
                terminal_next_phi[index] = True

    if shaping_enabled:
        assert shaping_config is not None
        shaping_config.validate(
            gamma=gamma, gae_lambda=gae_lambda, rank_utility_id="U_A", rank_utility=placement_utility
        )
        rewards = apply_pbrs_reward(
            rewards,
            phi_t,
            phi_next,
            pbrs_beta=shaping_config.pbrs_beta,
            gamma=gamma,
            terminal_next=terminal_next_phi,
            truncation=truncation,
        )
    elif shaping_config is not None:
        shaping_config.validate(
            gamma=gamma, gae_lambda=gae_lambda, rank_utility_id="U_A", rank_utility=placement_utility
        )
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


def ach_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
    old_logprob: torch.Tensor,
    raw_advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    bc_logits: torch.Tensor | None = None,
    config: AchLossConfig | None = None,
) -> AchLossOutput:
    if config is None:
        config = AchLossConfig()
    if config.eta < 0.0 or not torch.isfinite(torch.tensor(config.eta)):
        raise ValueError("eta must be finite and >= 0")
    if config.eps <= 0.0 or not torch.isfinite(torch.tensor(config.eps)):
        raise ValueError("eps must be finite and > 0")
    if config.l_th <= 0.0 or not torch.isfinite(torch.tensor(config.l_th)):
        raise ValueError("l_th must be finite and > 0")
    if config.pi_old_min <= 0.0 or not torch.isfinite(torch.tensor(config.pi_old_min)):
        raise ValueError("pi_old_min must be finite and > 0")
    _require_finite(policy_logits, "policy_logits")
    _require_finite(values, "values")
    _require_finite(old_logprob, "old_logprob")
    _require_finite(raw_advantages, "raw_advantages")
    _require_finite(returns, "returns")
    if bc_logits is not None:
        _require_finite(bc_logits, "bc_logits")
    mask = _require_action_logits(policy_logits, legal_mask)
    action_ids = _require_actions(actions, mask)
    flat_values = values.reshape(-1)
    for name, tensor in (
        ("old_logprob", old_logprob),
        ("raw_advantages", raw_advantages),
        ("returns", returns),
        ("values", flat_values),
    ):
        if tensor.shape != action_ids.shape:
            raise ValueError(f"{name} must have shape [batch]")

    legal = mask.to(dtype=policy_logits.dtype)
    legal_count = legal.sum(dim=1)
    legal_mean = (policy_logits * legal).sum(dim=1, keepdim=True) / legal_count.unsqueeze(1)
    centered = policy_logits - legal_mean
    clamped = centered.clamp(-config.l_th, config.l_th)
    log_probs = masked_log_softmax(clamped, mask)
    probs = log_probs.exp().masked_fill(~mask, 0.0)
    selected = action_ids.unsqueeze(1)
    selected_logprob = log_probs.gather(1, selected).squeeze(1)
    y_a = clamped.gather(1, selected).squeeze(1)
    pi_a = probs.gather(1, selected).squeeze(1)
    old_prob_raw = old_logprob.exp()
    pi_old = old_prob_raw.clamp_min(config.pi_old_min)
    ratio = pi_a / pi_old
    _require_finite(ratio, "ratio")

    advantage_rms = torch.sqrt((raw_advantages * raw_advantages).mean() + config.advantage_epsilon)
    scaled_advantages = raw_advantages / advantage_rms
    _require_finite(scaled_advantages, "scaled_advantages")
    pos = scaled_advantages >= 0.0
    neg = scaled_advantages < 0.0
    gate_pos = pos & (ratio < 1.0 + config.eps) & (y_a < config.l_th)
    gate_neg = neg & (ratio > 1.0 - config.eps) & (y_a > -config.l_th)
    gate_bool = gate_pos | gate_neg
    gate = gate_bool.to(dtype=policy_logits.dtype)
    policy_loss = -(gate * config.eta * y_a * scaled_advantages.detach() / pi_old).mean()
    value_loss = value_mse(flat_values, returns).mean()
    entropy_per_row = -(probs * log_probs.masked_fill(~mask, 0.0)).sum(dim=1)
    entropy = entropy_per_row.mean()
    if bc_logits is None:
        bc_kl_per_row = torch.zeros_like(entropy_per_row)
    else:
        bc_kl_per_row = masked_kl(clamped, bc_logits, mask, direction="current_to_reference")
    bc_kl_reverse = bc_kl_per_row.mean()
    total = (
        policy_loss
        + config.value_coef * value_loss
        + config.bc_kl_reverse_coef * bc_kl_reverse
        - config.entropy_alpha * entropy
    )
    illegal_probs = probs.masked_select(~mask)
    illegal_probability_max = (
        illegal_probs.max()
        if illegal_probs.numel() > 0
        else torch.zeros((), dtype=policy_logits.dtype, device=policy_logits.device)
    )
    raw_mean = raw_advantages.mean()
    scaled_mean = scaled_advantages.mean()
    ratio_clipped = (ratio <= 1.0 - config.eps) | (ratio >= 1.0 + config.eps)
    pos_count = pos.sum()
    neg_count = neg.sum()
    zero_count = (scaled_advantages == 0.0).sum()
    return AchLossOutput(
        total=total,
        metrics=AchLossMetrics(
            policy_loss=policy_loss.detach(),
            value_loss=value_loss.detach(),
            entropy=entropy.detach(),
            ratio_mean=ratio.mean().detach(),
            ratio_min=ratio.min().detach(),
            ratio_max=ratio.max().detach(),
            ratio_clipped_fraction=ratio_clipped.to(dtype=policy_logits.dtype).mean().detach(),
            gate_fraction=gate.mean().detach(),
            pos_gate_fraction=_safe_selected_mean(gate, pos),
            neg_gate_fraction=_safe_selected_mean(gate, neg),
            pi_old_clamp_fraction=(old_prob_raw < config.pi_old_min).to(dtype=policy_logits.dtype).mean().detach(),
            pi_old_min=torch.tensor(config.pi_old_min, dtype=policy_logits.dtype, device=policy_logits.device),
            pi_old_raw_min=old_prob_raw.min().detach(),
            approx_kl_old=(old_logprob - selected_logprob).mean().detach(),
            bc_kl_reverse=bc_kl_reverse.detach(),
            entropy_fraction_mean=entropy_fraction(clamped, mask).mean().detach(),
            advantage_raw_mean=raw_mean.detach(),
            advantage_raw_std=torch.sqrt(((raw_advantages - raw_mean) ** 2).mean()).detach(),
            advantage_raw_rms=torch.sqrt((raw_advantages * raw_advantages).mean()).detach(),
            advantage_scaled_mean=scaled_mean.detach(),
            advantage_scaled_std=torch.sqrt(((scaled_advantages - scaled_mean) ** 2).mean()).detach(),
            advantage_scaled_rms=torch.sqrt((scaled_advantages * scaled_advantages).mean()).detach(),
            advantage_positive_count=pos_count.detach(),
            advantage_negative_count=neg_count.detach(),
            advantage_zero_count=zero_count.detach(),
            illegal_probability_max=illegal_probability_max.detach(),
        ),
        entropy_per_row=entropy_per_row.detach(),
        gate_per_row=gate.detach(),
        ratio_clipped_per_row=ratio_clipped.to(dtype=policy_logits.dtype).detach(),
        bc_kl_per_row=bc_kl_per_row.detach(),
    )


def _safe_selected_mean(metric: torch.Tensor, selector: torch.Tensor) -> torch.Tensor | None:
    if bool(selector.any()):
        return metric[selector].mean().detach()
    return None


def _optional_float(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value)


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
