"""Fail-closed GRP potential and PBRS helpers for Phase 4C."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from collections.abc import Sequence as AbcSequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F

from hydra_learner.model import ACTION_SPACE, GRP_CLASSES, OBS_CHANNELS, TILE_WIDTH

PLACEMENT_UTILITY_DEFAULT: tuple[float, float, float, float] = (1.0, 0.3, -0.3, -1.0)
DEFAULT_GAE_GAMMA = 0.995
DEFAULT_GAE_LAMBDA = 0.95
PHI_DEFINITION_GRP_EXPECTED_U_A_V1 = "grp_expected_u_a_v1"
STATE_BOUNDARY_PLAYER_LOCAL_DECISION_STREAM_V1 = "player_local_decision_stream_v1"
RANK_UTILITY_U_A = "U_A"
REWARD_BASE_TERMINAL_U_A = "terminal_U_A"
REWARD_SHAPING_METADATA_KIND_NONE = "none"
REWARD_SHAPING_METADATA_KIND_PBRS = "pbrs"
VALIDATION_REPORT_SCHEMA_VERSION = 1
VALIDATION_REPORT_CONTRACT_VERSION = "phase4c_reward_shaping_validation_v1"
VALIDATION_THRESHOLDS_ABSENT_REASON = "phase4c_thresholds_absent"
THRESHOLD_CONTRACT_VERSION_PHASE4C_V1 = "phase4c_threshold_contract_v1"
REQUIRED_GATE_RESULT_CATEGORIES: tuple[str, ...] = (
    "calibration_logloss_brier_ece",
    "bucket_sample_bias",
    "phi_error_bias",
    "pbrs_magnitude_telescoping",
    "downstream_paired_arena_no_regression",
)

GRP_PERM_TABLE: tuple[tuple[int, int, int, int], ...] = tuple(
    (a, b, c, 6 - a - b - c) for a in range(4) for b in range(4) if b != a for c in range(4) if c != a and c != b
)

DEFAULT_REWARD_SHAPING_METADATA: dict[str, object] = {
    "enabled": False,
    "kind": REWARD_SHAPING_METADATA_KIND_NONE,
    "pbrs_beta": 0.0,
    "phi_definition": PHI_DEFINITION_GRP_EXPECTED_U_A_V1,
    "state_boundary": STATE_BOUNDARY_PLAYER_LOCAL_DECISION_STREAM_V1,
    "base_reward": REWARD_BASE_TERMINAL_U_A,
    "validated": False,
    "validation_artifact_id": None,
    "validation_artifact_path": None,
    "validation_artifact_hash": None,
}


@dataclass(frozen=True)
class MetricSummary:
    sample_count: int = 0
    coverage: float = 0.0
    nll: float | None = None
    brier: float | None = None
    ece: float | None = None
    top1_accuracy: float | None = None
    mse_to_u_a: float | None = None
    rmse_to_u_a: float | None = None
    mae_to_u_a: float | None = None
    signed_bias: float | None = None


@dataclass(frozen=True)
class BucketMetricSummary:
    bucket: str
    sample_count: int
    metrics: MetricSummary


@dataclass(frozen=True)
class BaselineMetricSummary:
    score_only: MetricSummary | None = None
    rank_only: MetricSummary | None = None
    round_only: MetricSummary | None = None
    combined_score_rank_round: MetricSummary | None = None
    missing: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardShapingValidationReport:
    schema_version: int
    contract_version: str
    predictor_checkpoint_hash: str | None
    predictor_checkpoint_id: str | None
    predictor_checkpoint_path: str | None
    model_config: dict[str, object]
    source_identity: dict[str, object]
    encoder_shape: tuple[int, int]
    action_space: int
    grp_classes: int
    rank_utility_id: str
    rank_utility: tuple[float, float, float, float]
    phi_definition: str
    state_boundary: str
    gamma: float
    gae_lambda: float
    public_only: bool
    validated: bool
    beta_activation_allowed: bool
    metrics: dict[str, object]
    bucket_metrics: dict[str, object]
    baselines: dict[str, object]
    threshold_contract_version: str | None = None
    threshold_contract_id: str | None = None
    threshold_contract_hash: str | None = None
    gate_results: dict[str, object] = field(default_factory=dict)
    reason: str | None = None

    @classmethod
    def thresholds_absent(
        cls,
        *,
        predictor_checkpoint_hash: str | None = None,
        predictor_checkpoint_id: str | None = None,
        predictor_checkpoint_path: str | None = None,
        model_config: Mapping[str, object] | None = None,
        source_identity: Mapping[str, object] | None = None,
        gamma: float = DEFAULT_GAE_GAMMA,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
    ) -> RewardShapingValidationReport:
        return cls(
            schema_version=VALIDATION_REPORT_SCHEMA_VERSION,
            contract_version=VALIDATION_REPORT_CONTRACT_VERSION,
            predictor_checkpoint_hash=predictor_checkpoint_hash,
            predictor_checkpoint_id=predictor_checkpoint_id,
            predictor_checkpoint_path=predictor_checkpoint_path,
            model_config=dict(model_config or {}),
            source_identity=dict(source_identity or {}),
            encoder_shape=(OBS_CHANNELS, TILE_WIDTH),
            action_space=ACTION_SPACE,
            grp_classes=GRP_CLASSES,
            rank_utility_id=RANK_UTILITY_U_A,
            rank_utility=PLACEMENT_UTILITY_DEFAULT,
            phi_definition=PHI_DEFINITION_GRP_EXPECTED_U_A_V1,
            state_boundary=STATE_BOUNDARY_PLAYER_LOCAL_DECISION_STREAM_V1,
            gamma=gamma,
            gae_lambda=gae_lambda,
            public_only=True,
            validated=False,
            beta_activation_allowed=False,
            metrics={},
            bucket_metrics={},
            baselines=_missing_baselines(),
            threshold_contract_version=None,
            threshold_contract_id=None,
            threshold_contract_hash=None,
            gate_results={},
            reason=VALIDATION_THRESHOLDS_ABSENT_REASON,
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["encoder_shape"] = list(self.encoder_shape)
        data["rank_utility"] = list(self.rank_utility)
        _validate_validation_report_dict(data)
        return cast("dict[str, object]", data)

    def validate(self) -> None:
        _validate_validation_report_dict(self.to_dict())

    def write(self, path: Path) -> None:
        payload = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> RewardShapingValidationReport:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid reward shaping validation JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("validation report root must be a dict")
        _validate_validation_report_dict(cast("dict[str, object]", payload))
        return _report_from_dict(cast("dict[str, object]", payload))


@dataclass(frozen=True)
class RewardShapingConfig:
    enabled: bool = False
    pbrs_beta: float = 0.0
    validation_artifact_path: Path | None = None
    phi_definition: str = PHI_DEFINITION_GRP_EXPECTED_U_A_V1
    state_boundary: str = STATE_BOUNDARY_PLAYER_LOCAL_DECISION_STREAM_V1

    def validate(
        self,
        *,
        gamma: float = DEFAULT_GAE_GAMMA,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
        rank_utility_id: str = RANK_UTILITY_U_A,
        rank_utility: Sequence[float] = PLACEMENT_UTILITY_DEFAULT,
        source_identity: Mapping[str, object] | None = None,
    ) -> RewardShapingValidationReport | None:
        if not math.isfinite(self.pbrs_beta) or self.pbrs_beta < 0.0:
            raise ValueError("pbrs_beta must be finite and >= 0")
        if self.phi_definition != PHI_DEFINITION_GRP_EXPECTED_U_A_V1:
            raise ValueError("unsupported phi_definition")
        if self.state_boundary != STATE_BOUNDARY_PLAYER_LOCAL_DECISION_STREAM_V1:
            raise ValueError("unsupported state_boundary")
        if not self.enabled and self.pbrs_beta > 0.0:
            raise ValueError("enabled=false requires pbrs_beta=0")
        if self.pbrs_beta == 0.0:
            return None
        if self.validation_artifact_path is None:
            raise ValueError("pbrs_beta > 0 requires validation_artifact_path")
        report = RewardShapingValidationReport.load(self.validation_artifact_path)
        validate_activation_report(
            report,
            gamma=gamma,
            gae_lambda=gae_lambda,
            rank_utility_id=rank_utility_id,
            rank_utility=rank_utility,
            phi_definition=self.phi_definition,
            state_boundary=self.state_boundary,
            source_identity=source_identity,
        )
        return report

    def metadata(
        self,
        *,
        gamma: float = DEFAULT_GAE_GAMMA,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
        rank_utility_id: str = RANK_UTILITY_U_A,
        rank_utility: Sequence[float] = PLACEMENT_UTILITY_DEFAULT,
        source_identity: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        report = self.validate(
            gamma=gamma,
            gae_lambda=gae_lambda,
            rank_utility_id=rank_utility_id,
            rank_utility=rank_utility,
            source_identity=source_identity,
        )
        if self.pbrs_beta == 0.0:
            return default_reward_shaping_metadata(gamma=gamma, gae_lambda=gae_lambda)
        assert report is not None
        return reward_shaping_metadata_from_report(self, report, gamma=gamma, gae_lambda=gae_lambda)


def default_reward_shaping_metadata(
    *, gamma: float = DEFAULT_GAE_GAMMA, gae_lambda: float = DEFAULT_GAE_LAMBDA
) -> dict[str, object]:
    metadata = dict(DEFAULT_REWARD_SHAPING_METADATA)
    metadata["gae_gamma"] = gamma
    metadata["gae_lambda"] = gae_lambda
    return metadata


def normalize_reward_shaping_metadata(value: Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return default_reward_shaping_metadata()
    metadata = dict(value)
    _validate_json_primitive(metadata, "reward_shaping")
    if "enabled" not in metadata or not isinstance(metadata["enabled"], bool):
        raise ValueError("reward_shaping.enabled must be bool")
    if metadata["enabled"]:
        required = {
            "kind",
            "pbrs_beta",
            "phi_definition",
            "state_boundary",
            "base_reward",
            "rank_utility_id",
            "rank_utility",
            "grp_classes",
            "gae_gamma",
            "gae_lambda",
            "validated",
            "validation_artifact_id",
            "validation_artifact_hash",
            "validation_artifact_path",
            "threshold_contract_version",
            "threshold_contract_id",
            "threshold_contract_hash",
        }
        missing = required.difference(metadata)
        if missing:
            raise ValueError(f"reward_shaping missing keys: {sorted(missing)}")
        if metadata["kind"] != REWARD_SHAPING_METADATA_KIND_PBRS:
            raise ValueError("enabled reward_shaping.kind must be pbrs")
        if metadata["validated"] is not True:
            raise ValueError("enabled reward_shaping requires validated=true")
        beta = metadata["pbrs_beta"]
        if not isinstance(beta, int | float) or not math.isfinite(float(beta)) or float(beta) <= 0.0:
            raise ValueError("enabled reward_shaping.pbrs_beta must be finite and > 0")
        for name in ("validation_artifact_id", "validation_artifact_hash", "validation_artifact_path"):
            field_value = metadata[name]
            if not isinstance(field_value, str) or not field_value:
                raise ValueError(f"enabled reward_shaping.{name} must be a non-empty string")
        if metadata["threshold_contract_version"] != THRESHOLD_CONTRACT_VERSION_PHASE4C_V1:
            raise ValueError("enabled reward_shaping.threshold_contract_version mismatch")
        for name in ("threshold_contract_id", "threshold_contract_hash"):
            field_value = metadata[name]
            if not isinstance(field_value, str) or not field_value:
                raise ValueError(f"enabled reward_shaping.{name} must be a non-empty string")
    else:
        merged = default_reward_shaping_metadata()
        merged.update(metadata)
        merged["enabled"] = False
        if "pbrs_beta" in merged:
            beta = merged["pbrs_beta"]
            if not isinstance(beta, int | float) or not math.isfinite(float(beta)) or float(beta) != 0.0:
                raise ValueError("disabled reward_shaping requires pbrs_beta=0")
        metadata = merged
    return metadata


def reward_shaping_metadata_from_report(
    config: RewardShapingConfig,
    report: RewardShapingValidationReport,
    *,
    gamma: float,
    gae_lambda: float,
) -> dict[str, object]:
    return normalize_reward_shaping_metadata(
        {
            "enabled": True,
            "kind": REWARD_SHAPING_METADATA_KIND_PBRS,
            "pbrs_beta": config.pbrs_beta,
            "phi_definition": config.phi_definition,
            "state_boundary": config.state_boundary,
            "base_reward": REWARD_BASE_TERMINAL_U_A,
            "rank_utility_id": report.rank_utility_id,
            "rank_utility": list(report.rank_utility),
            "grp_classes": report.grp_classes,
            "gae_gamma": gamma,
            "gae_lambda": gae_lambda,
            "validated": report.validated,
            "validation_artifact_id": report.predictor_checkpoint_id,
            "validation_artifact_hash": report.predictor_checkpoint_hash,
            "validation_artifact_path": str(config.validation_artifact_path)
            if config.validation_artifact_path is not None
            else None,
            "threshold_contract_version": report.threshold_contract_version,
            "threshold_contract_id": report.threshold_contract_id,
            "threshold_contract_hash": report.threshold_contract_hash,
        }
    )


def grp_expected_phi(
    *,
    grp_logits: torch.Tensor | None = None,
    grp_probs: torch.Tensor | None = None,
    actor: torch.Tensor,
    terminal: torch.Tensor | None = None,
    rank_utility: Sequence[float] = PLACEMENT_UTILITY_DEFAULT,
) -> torch.Tensor:
    if (grp_logits is None) == (grp_probs is None):
        raise ValueError("exactly one of grp_logits or grp_probs is required")
    source = grp_logits if grp_logits is not None else grp_probs
    assert source is not None
    if source.ndim != 2 or source.shape[1] != GRP_CLASSES:
        raise ValueError(f"GRP tensor must have shape [B,{GRP_CLASSES}]")
    if actor.ndim != 1 or actor.shape[0] != source.shape[0]:
        raise ValueError("actor must have shape [B]")
    if actor.dtype not in (torch.int32, torch.int64):
        raise TypeError("actor must be integer tensor")
    if len(rank_utility) != 4:
        raise ValueError("rank_utility must contain four values")
    if not bool(torch.isfinite(source).all()):
        raise ValueError("GRP tensor must be finite")
    actor_i64 = actor.to(dtype=torch.int64)
    if not bool(((actor_i64 >= 0) & (actor_i64 < 4)).all()):
        raise ValueError("actor ids must be in 0..3")
    if grp_logits is not None:
        probs = F.softmax(grp_logits, dim=1)
    else:
        probs = grp_probs
        assert probs is not None
        if not bool((probs >= 0.0).all()):
            raise ValueError("grp_probs must be non-negative")
        torch.testing.assert_close(
            probs.sum(dim=1),
            torch.ones(source.shape[0], dtype=probs.dtype, device=probs.device),
            rtol=1.0e-5,
            atol=1.0e-6,
            msg="grp_probs rows must sum to one",
        )
    utility = torch.tensor(tuple(rank_utility), dtype=probs.dtype, device=probs.device)
    actor_rank_utility = torch.empty((4, GRP_CLASSES), dtype=probs.dtype, device=probs.device)
    for class_index, permutation in enumerate(GRP_PERM_TABLE):
        for rank, player in enumerate(permutation):
            actor_rank_utility[player, class_index] = utility[rank]
    phi = (probs * actor_rank_utility.index_select(0, actor_i64)).sum(dim=1)
    if terminal is not None:
        if terminal.ndim != 1 or terminal.shape[0] != source.shape[0]:
            raise ValueError("terminal must have shape [B]")
        if terminal.dtype != torch.bool:
            raise TypeError("terminal must be bool")
        phi = torch.where(terminal.to(device=phi.device), torch.zeros_like(phi), phi)
    return phi


def apply_pbrs_reward(
    base_terminal_u_a: torch.Tensor,
    phi_t: torch.Tensor,
    phi_next: torch.Tensor,
    *,
    pbrs_beta: float,
    gamma: float,
    terminal_next: torch.Tensor | None = None,
    truncation: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        base_terminal_u_a.ndim != 1
        or phi_t.shape != base_terminal_u_a.shape
        or phi_next.shape != base_terminal_u_a.shape
    ):
        raise ValueError("base_terminal_u_a, phi_t, and phi_next must have shape [B]")
    if not math.isfinite(pbrs_beta) or pbrs_beta < 0.0:
        raise ValueError("pbrs_beta must be finite and >= 0")
    if not (0.0 < gamma <= 1.0) or not math.isfinite(gamma):
        raise ValueError("gamma must be finite and in (0, 1]")
    for tensor, name in ((base_terminal_u_a, "base_terminal_u_a"), (phi_t, "phi_t"), (phi_next, "phi_next")):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
    if truncation is not None and (truncation.shape != base_terminal_u_a.shape or truncation.dtype != torch.bool):
        raise ValueError("truncation must be bool with shape [B]")
    next_phi = phi_next
    if terminal_next is not None and (
        terminal_next.shape != base_terminal_u_a.shape or terminal_next.dtype != torch.bool
    ):
        raise ValueError("terminal_next must be bool with shape [B]")
    if terminal_next is not None:
        next_phi = torch.where(terminal_next.to(device=phi_next.device), torch.zeros_like(phi_next), phi_next)
    if pbrs_beta == 0.0:
        return base_terminal_u_a
    return base_terminal_u_a + pbrs_beta * (gamma * next_phi - phi_t)


def telescoping_terminal_residual(phi: torch.Tensor, *, gamma: float, tolerance: float = 1.0e-5) -> float:
    if phi.ndim != 1 or phi.shape[0] < 2:
        raise ValueError("phi must have shape [steps_plus_terminal]")
    if not bool(torch.isfinite(phi).all()):
        raise ValueError("phi must be finite")
    if abs(float(phi[-1])) > tolerance:
        raise ValueError("terminal phi must be zero")
    increments = gamma * phi[1:] - phi[:-1]
    powers = torch.pow(
        torch.tensor(gamma, dtype=phi.dtype, device=phi.device),
        torch.arange(increments.shape[0], dtype=phi.dtype, device=phi.device),
    )
    residual = (powers * increments).sum() + phi[0]
    return float(residual.detach().cpu())


def truncation_shaping_residual(phi: torch.Tensor, *, gamma: float, bootstrap_phi: float) -> float:
    if phi.ndim != 1 or phi.shape[0] < 1:
        raise ValueError("phi must have shape [steps]")
    tail = torch.cat([phi, torch.tensor([bootstrap_phi], dtype=phi.dtype, device=phi.device)])
    increments = gamma * tail[1:] - tail[:-1]
    powers = torch.pow(
        torch.tensor(gamma, dtype=phi.dtype, device=phi.device),
        torch.arange(increments.shape[0], dtype=phi.dtype, device=phi.device),
    )
    expected = -phi[0] + (gamma ** increments.shape[0]) * bootstrap_phi
    return float(((powers * increments).sum() - expected).detach().cpu())


def compute_grp_metrics(
    grp_probs: torch.Tensor,
    target_class: torch.Tensor,
    target_u_a: torch.Tensor,
    phi: torch.Tensor,
    *,
    ece_bins: int = 10,
) -> MetricSummary:
    if grp_probs.ndim != 2 or grp_probs.shape[1] != GRP_CLASSES:
        raise ValueError(f"grp_probs must have shape [B,{GRP_CLASSES}]")
    if (
        target_class.shape != (grp_probs.shape[0],)
        or target_u_a.shape != target_class.shape
        or phi.shape != target_class.shape
    ):
        raise ValueError("targets and phi must have shape [B]")
    if target_class.dtype not in (torch.int32, torch.int64):
        raise TypeError("target_class must be integer")
    if (
        not bool(torch.isfinite(grp_probs).all())
        or not bool(torch.isfinite(target_u_a).all())
        or not bool(torch.isfinite(phi).all())
    ):
        raise ValueError("metrics inputs must be finite")
    if grp_probs.shape[0] == 0:
        return MetricSummary()
    target = target_class.to(dtype=torch.int64)
    if not bool(((target >= 0) & (target < GRP_CLASSES)).all()):
        raise ValueError("target_class must be in 0..23")
    chosen = grp_probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1.0e-12)
    nll = -chosen.log().mean()
    one_hot = F.one_hot(target, GRP_CLASSES).to(dtype=grp_probs.dtype)
    brier = ((grp_probs - one_hot) ** 2).sum(dim=1).mean()
    confidence, pred = grp_probs.max(dim=1)
    top1 = (pred == target).to(dtype=torch.float32).mean()
    ece = _ece(confidence, pred == target, ece_bins)
    diff = phi - target_u_a.to(dtype=phi.dtype)
    mse = (diff * diff).mean()
    mae = diff.abs().mean()
    return MetricSummary(
        sample_count=grp_probs.shape[0],
        coverage=1.0,
        nll=float(nll),
        brier=float(brier),
        ece=float(ece),
        top1_accuracy=float(top1),
        mse_to_u_a=float(mse),
        rmse_to_u_a=math.sqrt(float(mse)),
        mae_to_u_a=float(mae),
        signed_bias=float(diff.mean()),
    )


def validate_activation_report(
    report: RewardShapingValidationReport,
    *,
    gamma: float,
    gae_lambda: float,
    rank_utility_id: str,
    rank_utility: Sequence[float],
    phi_definition: str,
    state_boundary: str,
    source_identity: Mapping[str, object] | None,
) -> None:
    report.validate()
    if report.validated is not True or report.beta_activation_allowed is not True:
        raise ValueError("validation artifact does not authorize PBRS activation")
    _validate_activation_authority(report)
    if report.schema_version != VALIDATION_REPORT_SCHEMA_VERSION:
        raise ValueError("validation report schema_version mismatch")
    if report.contract_version != VALIDATION_REPORT_CONTRACT_VERSION:
        raise ValueError("validation report contract_version mismatch")
    if report.encoder_shape != (OBS_CHANNELS, TILE_WIDTH):
        raise ValueError("validation report encoder_shape mismatch")
    if report.action_space != ACTION_SPACE:
        raise ValueError("validation report action_space mismatch")
    if report.grp_classes != GRP_CLASSES:
        raise ValueError("validation report grp_classes mismatch")
    if report.rank_utility_id != rank_utility_id:
        raise ValueError("validation report rank_utility_id mismatch")
    if tuple(report.rank_utility) != tuple(rank_utility):
        raise ValueError("validation report rank_utility mismatch")
    if report.phi_definition != phi_definition:
        raise ValueError("validation report phi_definition mismatch")
    if report.state_boundary != state_boundary:
        raise ValueError("validation report state_boundary mismatch")
    if not math.isclose(report.gamma, gamma, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("validation report gamma mismatch")
    if not math.isclose(report.gae_lambda, gae_lambda, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("validation report gae_lambda mismatch")
    if report.public_only is not True:
        raise ValueError("validation report must be public_only")
    if source_identity is not None and report.source_identity != dict(source_identity):
        raise ValueError("validation report source_identity mismatch")


def _ece(confidence: torch.Tensor, correct: torch.Tensor, bins: int) -> torch.Tensor:
    if bins < 1:
        raise ValueError("ece_bins must be >= 1")
    total = confidence.numel()
    out = torch.zeros((), dtype=torch.float32, device=confidence.device)
    correct_f = correct.to(dtype=torch.float32)
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        if index + 1 == bins:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        if bool(mask.any()):
            out = out + (mask.to(dtype=torch.float32).mean() * (confidence[mask].mean() - correct_f[mask].mean()).abs())
    return out if total > 0 else torch.zeros((), dtype=torch.float32, device=confidence.device)


def _missing_baselines() -> dict[str, object]:
    reason = "required_fields_absent"
    return {
        "score_only": {"missing": True, "reason": reason},
        "rank_only": {"missing": True, "reason": reason},
        "round_only": {"missing": True, "reason": reason},
        "combined_score_rank_round": {"missing": True, "reason": reason},
    }


def _report_from_dict(payload: dict[str, object]) -> RewardShapingValidationReport:
    return RewardShapingValidationReport(
        schema_version=cast("int", payload["schema_version"]),
        contract_version=cast("str", payload["contract_version"]),
        predictor_checkpoint_hash=cast("str | None", payload["predictor_checkpoint_hash"]),
        predictor_checkpoint_id=cast("str | None", payload["predictor_checkpoint_id"]),
        predictor_checkpoint_path=cast("str | None", payload["predictor_checkpoint_path"]),
        model_config=cast("dict[str, object]", payload["model_config"]),
        source_identity=cast("dict[str, object]", payload["source_identity"]),
        encoder_shape=cast("tuple[int, int]", tuple(cast("list[int]", payload["encoder_shape"]))),
        action_space=cast("int", payload["action_space"]),
        grp_classes=cast("int", payload["grp_classes"]),
        rank_utility_id=cast("str", payload["rank_utility_id"]),
        rank_utility=cast("tuple[float, float, float, float]", tuple(cast("list[float]", payload["rank_utility"]))),
        phi_definition=cast("str", payload["phi_definition"]),
        state_boundary=cast("str", payload["state_boundary"]),
        gamma=float(cast("int | float", payload["gamma"])),
        gae_lambda=float(cast("int | float", payload["gae_lambda"])),
        public_only=cast("bool", payload["public_only"]),
        validated=cast("bool", payload["validated"]),
        beta_activation_allowed=cast("bool", payload["beta_activation_allowed"]),
        metrics=cast("dict[str, object]", payload["metrics"]),
        bucket_metrics=cast("dict[str, object]", payload["bucket_metrics"]),
        baselines=cast("dict[str, object]", payload["baselines"]),
        threshold_contract_version=cast("str | None", payload.get("threshold_contract_version")),
        threshold_contract_id=cast("str | None", payload.get("threshold_contract_id")),
        threshold_contract_hash=cast("str | None", payload.get("threshold_contract_hash")),
        gate_results=cast("dict[str, object]", payload.get("gate_results", {})),
        reason=cast("str | None", payload.get("reason")),
    )


def _validate_validation_report_dict(payload: dict[str, object]) -> None:
    required = {
        "schema_version",
        "contract_version",
        "predictor_checkpoint_hash",
        "predictor_checkpoint_id",
        "predictor_checkpoint_path",
        "model_config",
        "source_identity",
        "encoder_shape",
        "action_space",
        "grp_classes",
        "rank_utility_id",
        "rank_utility",
        "phi_definition",
        "state_boundary",
        "gamma",
        "gae_lambda",
        "public_only",
        "validated",
        "beta_activation_allowed",
        "metrics",
        "bucket_metrics",
        "baselines",
        "threshold_contract_version",
        "threshold_contract_id",
        "threshold_contract_hash",
        "gate_results",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"validation report missing keys: {sorted(missing)}")
    _validate_json_primitive(payload, "validation_report")
    if payload["schema_version"] != VALIDATION_REPORT_SCHEMA_VERSION:
        raise ValueError("validation report schema_version mismatch")
    if payload["contract_version"] != VALIDATION_REPORT_CONTRACT_VERSION:
        raise ValueError("validation report contract_version mismatch")
    if payload["encoder_shape"] != [OBS_CHANNELS, TILE_WIDTH]:
        raise ValueError("validation report encoder_shape mismatch")
    if payload["action_space"] != ACTION_SPACE:
        raise ValueError("validation report action_space mismatch")
    if payload["grp_classes"] != GRP_CLASSES:
        raise ValueError("validation report grp_classes mismatch")
    if payload["rank_utility_id"] != RANK_UTILITY_U_A:
        raise ValueError("validation report rank_utility_id mismatch")
    if payload["rank_utility"] != list(PLACEMENT_UTILITY_DEFAULT):
        raise ValueError("validation report rank_utility mismatch")
    if payload["phi_definition"] != PHI_DEFINITION_GRP_EXPECTED_U_A_V1:
        raise ValueError("validation report phi_definition mismatch")
    if payload["state_boundary"] != STATE_BOUNDARY_PLAYER_LOCAL_DECISION_STREAM_V1:
        raise ValueError("validation report state_boundary mismatch")
    for name in ("gamma", "gae_lambda"):
        value = payload[name]
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"validation report {name} must be finite")
    for name in ("public_only", "validated", "beta_activation_allowed"):
        if not isinstance(payload[name], bool):
            raise ValueError(f"validation report {name} must be bool")
    if payload["public_only"] is not True:
        raise ValueError("validation report must be public_only")
    if payload["beta_activation_allowed"] is True and payload["validated"] is not True:
        raise ValueError("beta_activation_allowed requires validated=true")
    if payload.get("uses_hidden_private_oracle") is True:
        raise ValueError("validation report must not use hidden/private/oracle inputs")


def _validate_activation_authority(report: RewardShapingValidationReport) -> None:
    if report.threshold_contract_version != THRESHOLD_CONTRACT_VERSION_PHASE4C_V1:
        raise ValueError("validation report threshold_contract_version mismatch")
    if not report.threshold_contract_id:
        raise ValueError("validation report threshold_contract_id required")
    if not report.threshold_contract_hash:
        raise ValueError("validation report threshold_contract_hash required")
    for category in REQUIRED_GATE_RESULT_CATEGORIES:
        raw = report.gate_results.get(category)
        if not isinstance(raw, dict):
            raise ValueError(f"validation report gate_results missing {category}")
        if raw.get("passed") is not True:
            raise ValueError(f"validation report gate_results {category} did not pass")


def _validate_json_primitive(value: object, path: str) -> None:
    if isinstance(value, bool | str) or value is None:
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite float")
        return
    if isinstance(value, AbcSequence) and not isinstance(value, str):
        for index, item in enumerate(value):
            _validate_json_primitive(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains non-string key")
            if key in {
                "raw_score_phi",
                "rank_change_phi",
                "final_score_phi",
                "grp_argmax_phi",
                "oracle_phi",
                "hidden_private_phi",
            }:
                raise ValueError(f"{path} contains forbidden Phi shortcut {key}")
            _validate_json_primitive(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")
