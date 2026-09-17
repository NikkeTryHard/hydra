"""WP-07B oracle models — privileged teacher and actor-visible student.

Owns the deterministic teacher/student networks and the KL/MSE distillation
loss against teacher soft targets. The student forward rejects privileged
batches through the guard firewall; the teacher stays frozen during updates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from hydra2.belief.oracle_guard import (
    validate_actor_batch_no_privileged as validate_actor_batch_no_privileged,
)
from hydra2.contracts.common import ContractError


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    """Frozen distillation hyperparameters.

    All fields are required and validated; no implicit defaults in the
    training loop beyond this config.
    """

    seed: int
    feature_dim: int
    privileged_dim: int
    num_actions: int
    hidden_dim: int
    temperature: float
    w_belief: float
    w_value: float
    w_policy: float
    learning_rate: float
    weight_decay: float
    max_updates: int
    minibatch_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or not (0 <= self.seed < 2**31):
            raise ContractError(f"seed must be int in [0, 2^31), got {self.seed!r}")
        if (
            self.feature_dim <= 0
            or self.privileged_dim < 0
            or self.num_actions <= 0
            or self.hidden_dim <= 0
        ):
            raise ContractError("feature/privileged/num_actions/hidden must be positive")
        if not (0.1 <= self.temperature <= 10.0):
            raise ContractError(f"temperature must be in [0.1,10], got {self.temperature!r}")
        for name in ("w_belief", "w_value", "w_policy"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0:
                raise ContractError(f"{name} must be finite >=0, got {v!r}")
        if self.w_belief + self.w_value + self.w_policy == 0:
            raise ContractError("at least one of w_belief/w_value/w_policy must be >0")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ContractError(f"learning_rate must be finite >0, got {self.learning_rate!r}")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ContractError(f"weight_decay must be finite >=0, got {self.weight_decay!r}")
        if self.max_updates <= 0 or self.minibatch_size <= 0:
            raise ContractError("max_updates and minibatch_size must be positive")


# ---------------------------------------------------------------------------
# Tiny teacher / student models (deterministic)
# ---------------------------------------------------------------------------


class OracleTeacher(nn.Module):
    """Privileged teacher — sees hidden tiles (teacher soft targets).

    In production the teacher would be a larger model trained on privileged
    reconstructions; here it is a deterministic MLP over concatenated
    [actor_features, privileged_features] that produces belief/value/event
    logits. The teacher is frozen during distillation (no grad update of its
    params).
    """

    def __init__(
        self,
        *,
        feature_dim: int = 16,
        privileged_dim: int = 8,
        hidden_dim: int = 32,
        num_actions: int = 16,
    ) -> None:
        super().__init__()
        in_dim = feature_dim + privileged_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.belief_head = nn.Linear(hidden_dim, 34)
        self.value_head = nn.Linear(hidden_dim, 4)
        self.event_head = nn.Linear(hidden_dim, 20)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        # Deterministic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _ = nn.init.xavier_uniform_(m.weight)
                _ = nn.init.zeros_(m.bias)

    def forward(
        self,
        actor_features: torch.Tensor,
        privileged_features: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if actor_features.dim() != 2 or privileged_features.dim() != 2:
            raise ContractError("actor_features and privileged_features must be [B, D]")
        if actor_features.shape[0] != privileged_features.shape[0]:
            raise ContractError("batch size mismatch between actor and privileged")
        x: torch.Tensor = torch.cat([actor_features, privileged_features], dim=-1)
        h: torch.Tensor = self.net(x)
        out: dict[str, torch.Tensor] = {
            "belief_logits": self.belief_head(h),
            "value_logits": self.value_head(h),
            "event_logits": self.event_head(h),
            "policy_logits": self.policy_head(h),
        }
        if legal_mask is not None:
            _legal_shape: torch.Size = legal_mask.shape  # type: ignore[assignment]
            _policy_shape: torch.Size = out["policy_logits"].shape  # type: ignore[assignment]
            if _legal_shape != _policy_shape:
                raise ContractError(
                    f"legal_mask shape {tuple(_legal_shape)} != policy_logits {tuple(_policy_shape)}"  # noqa: E501
                )
            # Validate mask
            if legal_mask.dtype != torch.bool:
                raise ContractError(f"legal_mask must be bool, got {legal_mask.dtype}")
            if not bool(torch.all(legal_mask.any(dim=1)).item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # explicit bool, sync intentional
                raise ContractError("nonterminal all-false legal row is hard error")
            # Mask illegal to -inf for inference correctness
            _policy_logits: torch.Tensor = out["policy_logits"]
            out["policy_logits"] = torch.where(
                legal_mask,
                _policy_logits,
                torch.tensor(float("-inf"), device=_policy_logits.device),  # pyrefly: ignore[unknown-argument-type] # Tensor device known
            )
        return out


class StudentBeliefModel(nn.Module):
    """Actor-visible student — never sees privileged features.

    Encodes only actor-visible observation tensors. The forward validates that
    no privileged keys are present in the supplied batch dict via
    :func:`validate_actor_batch_no_privileged` when a dict is passed.
    For tensor-level tests, it simply consumes features.
    """

    def __init__(
        self, *, feature_dim: int = 16, hidden_dim: int = 32, num_actions: int = 16
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.belief_head = nn.Linear(hidden_dim, 34)
        self.value_head = nn.Linear(hidden_dim, 4)
        self.event_head = nn.Linear(hidden_dim, 20)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _ = nn.init.xavier_uniform_(m.weight)
                _ = nn.init.zeros_(m.bias)

    def forward(
        self,
        actor_features: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
        batch_dict: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        if batch_dict is not None:
            validate_actor_batch_no_privileged(batch_dict)
        if actor_features.dim() != 2:
            raise ContractError(f"actor_features must be [B, D], got {tuple(actor_features.shape)}")
        if actor_features.shape[1] != self.feature_dim:
            raise ContractError(
                f"actor_features dim {actor_features.shape[1]} != configured {self.feature_dim}"
            )
        h: torch.Tensor = self.net(actor_features)
        logits: torch.Tensor = self.policy_head(h)
        out: dict[str, torch.Tensor] = {
            "belief_logits": self.belief_head(h),
            "value_logits": self.value_head(h),
            "event_logits": self.event_head(h),
            "policy_logits": logits,
        }
        if legal_mask is not None:
            _legal_shape2: torch.Size = legal_mask.shape  # type: ignore[assignment]
            _logits_shape: torch.Size = logits.shape  # type: ignore[assignment]
            if _legal_shape2 != _logits_shape:
                raise ContractError(
                    f"legal_mask shape {tuple(_legal_shape2)} != logits {tuple(_logits_shape)}"
                )
            if legal_mask.dtype != torch.bool:
                raise ContractError(f"legal_mask must be bool, got {legal_mask.dtype}")
            if not bool(torch.all(legal_mask.any(dim=1)).item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # explicit bool, sync intentional
                raise ContractError("nonterminal all-false legal row is hard error")
            # Mask illegal logits to -inf for correct softmax
            masked: torch.Tensor = torch.where(
                legal_mask,
                logits,
                torch.tensor(float("-inf"), device=logits.device),  # pyrefly: ignore[unknown-argument-type] # device known
            )
            out["policy_logits"] = masked
        return out


# ---------------------------------------------------------------------------
# Distillation loss (deterministic)
# ---------------------------------------------------------------------------


def distillation_loss(
    student_out: dict[str, torch.Tensor],
    teacher_probs: dict[str, torch.Tensor],
    legal_mask: torch.Tensor | None = None,
    *,
    config: DistillationConfig,
) -> dict[str, torch.Tensor]:
    """KL + MSE distillation against teacher soft targets.

    Teacher probs are expected to be softmax-normalized. Student logits are
    compared via KL divergence (belief/policy/event) and MSE (value if needed).
    Illegal policy logits are masked to -inf before softmax inside.

    Returns dict with ``total`` plus per-head losses (all finite).
    """
    losses: dict[str, torch.Tensor] = {}
    total = torch.tensor(0.0, dtype=torch.float32, device=next(iter(student_out.values())).device)
    temp = config.temperature
    # Belief KL: teacher belief_probs (34) vs student belief_logits
    if config.w_belief > 0:
        s_logits = student_out["belief_logits"] / temp
        t_probs = teacher_probs["belief_probs"]
        if s_logits.shape != t_probs.shape:
            raise ContractError(
                f"belief shape mismatch {tuple(s_logits.shape)} vs {tuple(t_probs.shape)}"
            )
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        # KL(t || s) = sum t * (log t - log s)
        # Use t * log t - t * log s ; t log t is constant w.r.t student but we include for correctness  # noqa: E501
        t_log = torch.log(torch.clamp(t_probs, min=1e-8))
        kl = torch.sum(t_probs * (t_log - s_log_probs), dim=-1).mean()
        if not bool(torch.isfinite(kl).item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # explicit bool, sync intentional
            raise ContractError(f"belief KL non-finite: {kl!r}")
        losses["belief"] = kl
        total = total + config.w_belief * kl
    # Value: MSE between softmax probs (or direct)
    if config.w_value > 0:
        s_logits = student_out["value_logits"] / temp
        t_probs = teacher_probs["value_probs"]
        if s_logits.shape != t_probs.shape:
            raise ContractError(
                f"value shape mismatch {tuple(s_logits.shape)} vs {tuple(t_probs.shape)}"
            )
        s_probs = F.softmax(s_logits, dim=-1)
        mse = F.mse_loss(s_probs, t_probs, reduction="mean")
        if not bool(torch.isfinite(mse).item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # explicit bool, sync intentional
            raise ContractError(f"value MSE non-finite: {mse!r}")
        losses["value"] = mse
        total = total + config.w_value * mse
    # Policy KL with legal masking
    if config.w_policy > 0:
        s_logits = student_out["policy_logits"] / temp
        t_probs = teacher_probs["policy_probs"]
        if s_logits.shape != t_probs.shape:
            raise ContractError(
                f"policy shape mismatch {tuple(s_logits.shape)} vs {tuple(t_probs.shape)}"
            )
        if legal_mask is not None:
            if legal_mask.shape != s_logits.shape:
                raise ContractError("legal_mask shape mismatch in distillation policy loss")
            # Teacher illegal probs should already be zero; enforce
            # Student illegal logits already -inf; log_softmax will handle
            # Zero out illegal teacher mass and renormalize for safety
            t_probs = torch.where(legal_mask, t_probs, torch.zeros_like(t_probs))
            t_sum = t_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            t_probs = t_probs / t_sum
        s_log_probs = F.log_softmax(s_logits.float(), dim=-1)
        # For illegal positions, s_log_probs is -inf; but t_prob is 0 there, so product is 0
        # Replace -inf with 0 for those positions via masking
        if legal_mask is not None:
            s_log_probs = torch.where(legal_mask, s_log_probs, torch.zeros_like(s_log_probs))
        t_log = torch.log(torch.clamp(t_probs, min=1e-8))
        kl = torch.sum(t_probs * (t_log - s_log_probs), dim=-1).mean()
        if not bool(torch.isfinite(kl).item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # explicit bool, sync intentional
            raise ContractError(f"policy KL non-finite: {kl!r}")
        losses["policy"] = kl
        total = total + config.w_policy * kl
    losses["total"] = total
    for k, v in losses.items():
        if not bool(torch.isfinite(v).item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # explicit bool, sync intentional
            raise ContractError(f"loss {k} non-finite: {v!r}")
    return losses


__all__ = [
    "DistillationConfig",
    "OracleTeacher",
    "StudentBeliefModel",
    "distillation_loss",
]
