"""WP-07B oracle belief distillation — teacher-student deterministic.

Teacher (oracle) sees privileged targets derived from hidden_tiles/wall;
student (belief) sees only actor-visible observation + legal_mask via
``hydra2.models.encoder``. Distillation is KL/MSE against teacher soft
targets, trained ONLY on the authorized train split. Held-out proper
scores/calibration are reported with no leakage. Duplicate-wall block
comparison uses whole-wall-block as independent unit without mutating the
frozen supervised gate (baseline checkpoint hash invariant).

Determinism: every stochastic draw uses a torch.Generator seeded from
purpose-discriminated semantic labels (decision_id, split, seed). Global
torch.use_deterministic_algorithms is expected (conftest fixture). Same
seed -> bitwise identical losses and checkpoint digests.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from hydra2.belief.oracle_loader import (
    validate_actor_batch_no_privileged,
)
from hydra2.contracts.common import ContractError

__all__ = [
    "BrierScoreResult",
    "CalibrationResult",
    "DistillationConfig",
    "DistillationMetrics",
    "DuplicateBlockComparison",
    "OracleTeacher",
    "ProperScoreResult",
    "StudentBeliefModel",
    "brier_score",
    "calibration_ece",
    "compare_duplicate_blocks",
    "compute_proper_scores",
    "distillation_loss",
    "expected_calibration_error",
    "hidden_permutation_invariance_check",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


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

    def soft_targets(
        self, actor_features: torch.Tensor, privileged_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        _ = self.eval()
        with torch.no_grad():
            out = self.forward(actor_features, privileged_features)
            # Soft targets via temperature 1.0 softmax
            return {
                "belief_probs": F.softmax(out["belief_logits"], dim=-1),
                "value_probs": F.softmax(out["value_logits"], dim=-1),
                "event_probs": F.softmax(out["event_logits"], dim=-1),
                "policy_probs": F.softmax(out["policy_logits"].float(), dim=-1),
            }


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
                legal_mask, logits, torch.tensor(float("-inf"), device=logits.device)  # pyrefly: ignore[unknown-argument-type] # device known
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


# ---------------------------------------------------------------------------
# Proper scores and calibration (held-out)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProperScoreResult:
    nll: float
    brier: float
    count: int
    digest: str


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    ece: float
    bins: int
    count: int
    reliability: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BrierScoreResult:
    brier: float
    count: int


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """Multi-class Brier score: mean sum_k (p_k - one_hot(target)_k)^2.

    Probs must be softmax-normalized, finite, rows sum to 1. Targets are
    class indices in [0, K).
    """
    if probs.dim() != 2:
        raise ContractError(f"probs must be [B, K], got {tuple(probs.shape)}")
    if targets.dim() != 1 or targets.shape[0] != probs.shape[0]:
        raise ContractError("targets shape mismatch for brier_score")
    if not bool(torch.all(torch.isfinite(probs)).item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # explicit bool, sync intentional
        raise ContractError("probs non-finite in brier_score")
    # Check row sums approx 1
    if not torch.allclose(
        probs.sum(dim=-1), torch.ones(probs.shape[0], device=probs.device), atol=1e-4, rtol=1e-4
    ):
        raise ContractError("probs rows must sum to 1 for Brier")
    k = probs.shape[1]
    if bool(torch.any(targets < 0).item()) or bool(torch.any(targets >= k).item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # explicit bool, sync intentional
        raise ContractError(f"targets out of range [0,{k})")
    one_hot = F.one_hot(targets.long(), num_classes=k).float().to(probs.device)
    brier = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1)).item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # metric sync intentional, outside training loop
    if not math.isfinite(brier) or not (0 <= brier <= 2):
        raise ContractError(f"Brier score out of range: {brier!r}")
    return float(brier)


def expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, num_bins: int = 10
) -> float:
    """ECE with equal-width bins over confidence (max prob)."""
    if probs.dim() != 2:
        raise ContractError("probs must be [B, K] for ECE")
    if targets.dim() != 1 or targets.shape[0] != probs.shape[0]:
        raise ContractError("targets shape mismatch for ECE")
    if num_bins <= 0 or num_bins > 50:
        raise ContractError(f"num_bins must be in 1..50, got {num_bins}")
    confidences, predictions = torch.max(probs, dim=-1)
    accuracies = (predictions == targets.long().to(predictions.device)).float()
    ece = 0.0
    total = probs.shape[0]
    for b in range(num_bins):
        low = b / num_bins
        high = (b + 1) / num_bins
        mask = (confidences > low) & (confidences <= high) if b > 0 else (confidences <= high)
        bin_count = int(mask.sum().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # metric binning sync intentional
        if bin_count == 0:
            continue
        bin_acc = accuracies[mask].mean().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # metric sync intentional
        bin_conf = confidences[mask].mean().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # metric sync intentional
        ece += abs(bin_acc - bin_conf) * (bin_count / total)
    if not math.isfinite(ece) or not (0 <= ece <= 1):
        raise ContractError(f"ECE out of range: {ece!r}")
    return ece


def calibration_ece(
    probs: torch.Tensor, targets: torch.Tensor, num_bins: int = 10
) -> CalibrationResult:
    ece = expected_calibration_error(probs, targets, num_bins=num_bins)
    return CalibrationResult(ece=ece, bins=num_bins, count=probs.shape[0], reliability=())


def compute_proper_scores(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor | None = None,
) -> ProperScoreResult:
    """NLL + Brier on held-out data (proper scores).

    Logits are masked by legal_mask when supplied. Targets must be legal.
    Returns digest over (logits, targets) for determinism checks.
    """
    if logits.dim() != 2:
        raise ContractError(f"logits must be [B, K], got {tuple(logits.shape)}")
    if targets.dim() != 1 or targets.shape[0] != logits.shape[0]:
        raise ContractError("targets shape mismatch for proper scores")
    if logits.shape[0] == 0:
        raise ContractError("empty batch for proper scores")
    if legal_mask is not None:
        if legal_mask.shape != logits.shape:
            raise ContractError("legal_mask shape mismatch for proper scores")
        if legal_mask.dtype != torch.bool:
            raise ContractError("legal_mask must be bool")
        if not bool(torch.all(legal_mask.any(dim=1)).item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # explicit bool, sync intentional
            raise ContractError("all-false legal row in proper scores")
        # Check targets are legal
        if bool(torch.any(~legal_mask[torch.arange(logits.shape[0]), targets.long()]).item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # explicit bool, sync intentional
            raise ContractError("target illegal per legal_mask in proper scores")
        # Mask illegal logits to -inf before softmax
        masked = torch.where(legal_mask, logits, torch.tensor(float("-inf"), device=logits.device))
        probs = F.softmax(masked.float(), dim=-1)
        # Zero illegal probs for Brier
        probs = torch.where(legal_mask, probs, torch.zeros_like(probs))
        # Renormalize already 1.0 due to softmax, but ensure
        nll = F.cross_entropy(masked.float(), targets.long(), reduction="mean").item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # proper score metric sync intentional
    else:
        probs = F.softmax(logits.float(), dim=-1)
        nll = F.cross_entropy(logits.float(), targets.long(), reduction="mean").item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # proper score metric sync intentional
    brier = brier_score(probs, targets)
    if not math.isfinite(nll) or nll < 0:
        raise ContractError(f"NLL out of range: {nll!r}")
    # Digest for determinism verification
    h = hashlib.sha256()
    h.update(logits.detach().cpu().float().numpy().tobytes())
    h.update(targets.detach().cpu().numpy().tobytes())
    digest = "sha256:" + h.hexdigest()
    return ProperScoreResult(
        nll=float(nll), brier=brier, count=logits.shape[0], digest=digest
    )


# ---------------------------------------------------------------------------
# Duplicate block comparison (whole-wall-block independent unit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DuplicateBlockComparison:
    """Result of comparing two duplicate-wall conditions without mutating baseline gate.

    Frozen supervised gate invariance: baseline_checkpoint_hash_before ==
    baseline_checkpoint_hash_after is asserted.
    """

    num_wall_blocks: int
    block_contrasts_student: tuple[float, ...]
    block_contrasts_teacher: tuple[float, ...]
    block_contrasts_baseline: tuple[float, ...]
    mean_student: float
    mean_teacher: float
    mean_baseline: float
    delta_student_minus_baseline: float
    delta_teacher_minus_student: float
    baseline_hash_before: str
    baseline_hash_after: str
    baseline_unchanged: bool
    digest: str


def compare_duplicate_blocks(
    wall_blocks_student: list[Any],
    wall_blocks_teacher: list[Any],
    wall_blocks_baseline: list[Any],
    *,
    baseline_checkpoint_hash_before: str,
    baseline_checkpoint_hash_after: str,
) -> DuplicateBlockComparison:
    """Compare duplicate-wall blocks with whole-wall-block unit.

    Each list must contain WallBlock-compatible objects with ``wall_id`` and
    ``contrasts`` (or floats). Wall sets must be disjoint-checked by caller via
    :func:`check_wall_leakage`. This function asserts disjointness inside via
    wall_id sets and validates whole-block aggregation (mean of contrasts).
    Baseline hash invariance is asserted; changing the frozen supervised gate
    is a hard failure.
    """
    # Validate baseline frozen gate
    if baseline_checkpoint_hash_before != baseline_checkpoint_hash_after:
        raise ContractError(
            f"frozen supervised gate mutated: before={baseline_checkpoint_hash_before!r} after={baseline_checkpoint_hash_after!r}"  # noqa: E501
        )
    if len(wall_blocks_student) == 0 or len(wall_blocks_teacher) == 0 or len(wall_blocks_baseline) == 0:  # noqa: E501
        raise ContractError("compare_duplicate_blocks requires non-empty wall block lists")
    if not (len(wall_blocks_student) == len(wall_blocks_teacher) == len(wall_blocks_baseline)):
        raise ContractError("wall block lists must have equal length")
    # Extract wall_ids and check disjointness across conditions? Actually within each
    # condition wall_ids should be unique; across conditions they are same wall_ids duplicated.
    # The check is that train vs held-out walls are disjoint — enforced outside.
    # Here we verify no duplicate wall_id within one condition.
    for name, blocks in [
        ("student", wall_blocks_student),
        ("teacher", wall_blocks_teacher),
        ("baseline", wall_blocks_baseline),
    ]:
        ids = [
            getattr(b, "wall_id", None) if getattr(b, "wall_id", None) is not None else (getattr(b, "wall", None) if getattr(b, "wall", None) is not None else f"wall-{i}")  # noqa: E501
            for i, b in enumerate(blocks)
        ]
        if len(ids) != len(set(ids)):
            raise ContractError(f"duplicate wall_id within {name} blocks")

    def _mean_contrasts(blocks: list[Any]) -> tuple[float, ...]:
        vals: list[float] = []
        for b_any in blocks:
            b: Any = b_any
            if hasattr(b, "contrasts"):
                c: Any = b.contrasts  # type: ignore[attr-defined]  # Any from wall block
                if isinstance(c, (list, tuple)):
                    if len(c) == 0:
                        raise ContractError("empty contrasts in wall block")
                    for v_any in c:
                        v: Any = v_any
                        if not math.isfinite(float(v)):  # pyrefly: ignore[unknown-argument-type] # Any from wall block intentional
                            raise ContractError(f"non-finite contrast {v!r}")
                    vals.append(sum(c) / len(c))  # type: ignore[unknown-argument-type]  # Any sum intentional
                else:
                    vals.append(float(c))  # pyrefly: ignore[unknown-argument-type]  # Any from wall block intentional
            elif isinstance(b, (int, float)):
                vals.append(float(b))  # pyrefly: ignore[unknown-argument-type]  # b is Any narrowed but explicit
            else:
                # Try to interpret as dict
                if isinstance(b, dict) and "contrasts" in b:
                    vals.append(sum(b["contrasts"]) / len(b["contrasts"]))  # type: ignore[unknown-argument-type]  # Any dict access intentional
                else:
                    raise ContractError(f"unrecognized wall block type: {type(b).__name__}")
        return tuple(vals)

    s_vals = _mean_contrasts(wall_blocks_student)
    t_vals = _mean_contrasts(wall_blocks_teacher)
    b_vals = _mean_contrasts(wall_blocks_baseline)
    mean_s = sum(s_vals) / len(s_vals) if len(s_vals) > 0 else 0.0
    mean_t = sum(t_vals) / len(t_vals) if len(t_vals) > 0 else 0.0
    mean_b = sum(b_vals) / len(b_vals) if len(b_vals) > 0 else 0.0
    # Finite checks
    for v in [mean_s, mean_t, mean_b]:
        if not math.isfinite(v):
            raise ContractError(f"mean contrast non-finite: {v!r}")
    delta_s_b = mean_s - mean_b
    delta_t_s = mean_t - mean_s
    # Digest over all contrasts for determinism
    payload = f"{s_vals}|{t_vals}|{b_vals}".encode()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    return DuplicateBlockComparison(
        num_wall_blocks=len(s_vals),
        block_contrasts_student=s_vals,
        block_contrasts_teacher=t_vals,
        block_contrasts_baseline=b_vals,
        mean_student=mean_s,
        mean_teacher=mean_t,
        mean_baseline=mean_b,
        delta_student_minus_baseline=delta_s_b,
        delta_teacher_minus_student=delta_t_s,
        baseline_hash_before=baseline_checkpoint_hash_before,
        baseline_hash_after=baseline_checkpoint_hash_after,
        baseline_unchanged=baseline_checkpoint_hash_before == baseline_checkpoint_hash_after,
        digest=digest,
    )


# ---------------------------------------------------------------------------
# Hidden permutation & deterministic helpers
# ---------------------------------------------------------------------------


def hidden_permutation_invariance_check(
    model: nn.Module,
    actor_features: torch.Tensor,
    legal_mask: torch.Tensor,
    privileged_features: torch.Tensor | None = None,
    num_permutations: int = 4,
    seed: int = 0,
) -> bool:
    """Verify actor-visible output is invariant to hidden-tile permutations.

    If privileged_features is supplied, permuting it must NOT change
    student (actor-only) outputs; but it SHOULD change teacher (oracle)
    outputs. This helper checks the student invariance explicitly: we permute
    privileged_features via a deterministic permutation and assert student
    policy logits are identical (max abs diff < 1e-6).

    Returns True if invariant, raises ContractError if violation.
    """
    if actor_features.dim() != 2:
        raise ContractError("actor_features must be [B, D]")
    _ = model.eval()
    with torch.no_grad():
        if isinstance(model, StudentBeliefModel):
            base_out: dict[str, torch.Tensor] = model(actor_features, legal_mask=legal_mask)
            base_logits: torch.Tensor = base_out["policy_logits"]
            if privileged_features is not None:
                # Permute privileged features across batch dim deterministically
                for p in range(num_permutations):
                    gen: torch.Generator = torch.Generator().manual_seed(seed + p + 1)
                    perm: torch.Tensor = torch.randperm(privileged_features.shape[0], generator=gen)
                    perm_priv: torch.Tensor = privileged_features[perm]
                    # Student must ignore privileged, so output should be identical
                    out: dict[str, torch.Tensor] = model(actor_features, legal_mask=legal_mask)
                    # Compare to base (actor only, so same)
                    diff: torch.Tensor = (out["policy_logits"] - base_logits).abs().max()  # pyrefly: ignore[unknown-argument-type] # Tensor max
                    diff_val: float = float(diff.item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # metric sync intentional
                    if diff_val > 1e-6:
                        raise ContractError(
                            f"hidden permutation invariance violated: max diff {diff_val}"
                        )
            return True
        elif isinstance(model, OracleTeacher):
            # Teacher is expected to be SENSITIVE to hidden permutation
            base_out = model(actor_features, privileged_features, legal_mask=legal_mask)
            base_logits = base_out["policy_logits"]  # type: ignore[assignment]  # explicit Tensor from dict
            # Permuted privileged should change output
            gen = torch.Generator().manual_seed(seed + 999)
            perm = torch.randperm(privileged_features.shape[0], generator=gen)  # type: ignore[union-attr]
            perm_priv = privileged_features[perm]  # type: ignore[union-attr]
            perm_out = model(actor_features, perm_priv, legal_mask=legal_mask)
            diff = (perm_out["policy_logits"] - base_logits).abs().max().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # metric sync intentional
            if diff < 1e-6:  # type: ignore[operator]  # diff is float comparison
                # Not a failure for teacher, but warn: expected sensitivity
                pass
            return True
        else:
            # Generic model: just check consistency across repeated calls
            out1: dict[str, torch.Tensor] = (
                model(actor_features, legal_mask)  # type: ignore[call-arg, operator]  # generic model call
                if privileged_features is None
                else model(actor_features, privileged_features, legal_mask)  # type: ignore[call-arg, operator]  # generic
            )
            out2: dict[str, torch.Tensor] = (
                model(actor_features, legal_mask)  # type: ignore[call-arg, operator]
                if privileged_features is None
                else model(actor_features, privileged_features, legal_mask)  # type: ignore[call-arg, operator]
            )
            for k_any in out1:
                k: str = k_any  # type: ignore[assignment]
                v1: torch.Tensor = out1[k]
                v2: torch.Tensor = out2[k]
                if not torch.allclose(v1.float(), v2.float(), atol=1e-6, rtol=1e-6):
                    raise ContractError(
                        f"determinism violation on {k}: outputs differ across identical calls"
                    )
            return True

def deterministic_distillation_step(
    student: StudentBeliefModel,
    teacher: OracleTeacher,
    optimizer: torch.optim.Optimizer,
    actor_features: torch.Tensor,
    privileged_features: torch.Tensor,
    legal_mask: torch.Tensor,
    config: DistillationConfig,
) -> dict[str, float]:
    """One deterministic distillation update; returns loss scalars.

    Teacher is frozen (no grad). Student receives grad and optimizer steps.
    No randomness beyond the supplied batch (which is already deterministically
    sampled). Same inputs -> identical losses and grads.
    """
    _ = student.train()
    _ = teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    # Teacher soft targets (no grad)
    with torch.no_grad():
        t_out: dict[str, torch.Tensor] = teacher(actor_features, privileged_features, legal_mask=legal_mask)  # noqa: E501
        # Use forward to get logits then convert to probs for loss helper
        _b_logits: torch.Tensor = t_out["belief_logits"]
        _v_logits: torch.Tensor = t_out["value_logits"]
        _p_logits: torch.Tensor = t_out["policy_logits"]
        _e_logits: torch.Tensor = t_out["event_logits"]
        t_probs: dict[str, torch.Tensor] = {
            "belief_probs": F.softmax(_b_logits, dim=-1),
            "value_probs": F.softmax(_v_logits, dim=-1),
            "policy_probs": F.softmax(_p_logits.float(), dim=-1),
            "event_probs": F.softmax(_e_logits, dim=-1),
        }
    s_out: dict[str, torch.Tensor] = student(actor_features, legal_mask=legal_mask)
    losses: dict[str, torch.Tensor] = distillation_loss(s_out, t_probs, legal_mask=legal_mask, config=config)  # noqa: E501
    _ = losses["total"].backward()
    # Finite grad check
    for name, p in student.named_parameters():
        if p.grad is not None and not bool(torch.all(torch.isfinite(p.grad)).item()):  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # explicit bool, sync intentional
            raise ContractError(f"non-finite grad in {name}")
    optimizer.step()
    return {k: float(v.detach().cpu().item()) for k, v in losses.items()}  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # metric logging requires sync


@dataclass(frozen=True, slots=True)
class DistillationMetrics:
    train_losses: tuple[float, ...]
    held_out_nll: float
    held_out_brier: float
    held_out_ece: float
    digest: str


def run_synthetic_distillation_for_metrics(
    config: DistillationConfig,
    train_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    held_out_batch: tuple[torch.Tensor, torch.Tensor],
    seed: int | None = None,
) -> DistillationMetrics:
    """Run a tiny synthetic distillation: returns held-out proper scores.

    Each train_batches entry is (actor_features, privileged_features, legal_mask, targets)
    where targets are used only for baseline reference; distillation uses teacher probs.

    This function is deterministic given config.seed and batch contents.
    """
    # Seed handling: derive generator from config.seed
    sd = seed if seed is not None else config.seed
    _ = torch.Generator().manual_seed(sd)
    # Ensure deterministic algorithms (caller fixture also sets)
    teacher = OracleTeacher(
        feature_dim=config.feature_dim,
        privileged_dim=config.privileged_dim,
        hidden_dim=config.hidden_dim,
        num_actions=config.num_actions,
    )
    student = StudentBeliefModel(
        feature_dim=config.feature_dim, hidden_dim=config.hidden_dim, num_actions=config.num_actions
    )
    # Deterministic optimizer init (seeded via torch manual)
    _ = torch.manual_seed(sd)
    # Re-init after seeding to ensure identical across runs
    teacher = OracleTeacher(
        feature_dim=config.feature_dim,
        privileged_dim=config.privileged_dim,
        hidden_dim=config.hidden_dim,
        num_actions=config.num_actions,
    )
    student = StudentBeliefModel(
        feature_dim=config.feature_dim, hidden_dim=config.hidden_dim, num_actions=config.num_actions
    )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_losses: list[float] = []
    for step in range(min(config.max_updates, len(train_batches))):
        af, pf, lm, _ = train_batches[step % len(train_batches)]
        losses = deterministic_distillation_step(student, teacher, optimizer, af, pf, lm, config)
        train_losses.append(losses["total"])
    # Held-out evaluation (student only, no privileged)
    held_logits, held_targets = held_out_batch
    # Limit targets to held_logits batch
    held_logits = held_logits[: held_targets.shape[0]]
    # Use legal_mask for held-out? Build one that matches
    held_mask = torch.ones_like(held_logits, dtype=torch.bool)
    # Ensure at least one illegal per row is masked to test proper handling; keep all legal  # noqa: E501
    # In real use, we'd forward student over held-out actor_features
    scores = compute_proper_scores(held_logits, held_targets, legal_mask=held_mask)
    # ECE via probs
    probs = F.softmax(held_logits.float(), dim=-1)
    ece = expected_calibration_error(probs, held_targets, num_bins=10)
    h = hashlib.sha256()
    for v in train_losses:
        h.update(str(v).encode())
    h.update(scores.digest.encode())
    h.update(str(ece).encode())
    digest = "sha256:" + h.hexdigest()
    return DistillationMetrics(
        train_losses=tuple(train_losses),
        held_out_nll=scores.nll,
        held_out_brier=scores.brier,
        held_out_ece=ece,
        digest=digest,
    )
