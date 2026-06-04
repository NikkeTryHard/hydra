"""Default-off DRDA residual adapter ACH path."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, override

import torch
from torch import nn

if TYPE_CHECKING:
    from hydra_learner.model.losses import LossWeights
from hydra_learner.checkpointing.core import ModelConfig, OptimizerConfig, RuntimeConfig, _torch_load, save_checkpoint
from hydra_learner.model import ACTION_SPACE, OBS_CHANNELS, TILE_WIDTH, HydraBaseOutput, HydraPolicyNet
from hydra_learner.ppo.rl import (
    AchLossConfig,
    EntropyController,
    ach_loss,
    default_entropy_target_fraction,
    legal_count_bucket_means,
    masked_kl,
)
from hydra_learner.ppo.step import PpoBatch, _validate_json_safe_metrics
from hydra_learner.rl_experiments.ach_step import (
    AchTrainStepConfig,
    AchTrainStepResult,
    _bucket_optional_means,
    _total_grad_norm,
)

MIN_TAU_DRDA = 2.0
DEFAULT_TAU_DRDA = 4.0
DRDA_RESIDUAL_OBJECTIVE = "drda_residual_adapter_ach"
DRDA_RESIDUAL_MODE = "residual_adapter_no_rebase"
DRDA_OPTIMIZER_SCOPE = "residual_only"
DRDA_REBASE_CAPABILITY = "unsupported"
DRDA_POLICY_PRESERVATION = "not_claimed"
DRDA_RESIDUAL_REPRESENTATION = "obs_mlp_policy_logits"
DrdaWeightSource = Literal["raw", "ema"]


@dataclass(frozen=True)
class DrdaResidualConfig:
    tau_drda: float = DEFAULT_TAU_DRDA
    rebase_enabled: bool = False
    residual_init_scale: float = 0.0

    def __post_init__(self) -> None:
        validate_tau_drda(self.tau_drda)
        if self.rebase_enabled:
            raise ValueError("DRDA residual_adapter_no_rebase requires rebase_enabled=False")
        if not math.isfinite(self.residual_init_scale):
            raise ValueError("residual_init_scale must be finite")
        if self.residual_init_scale < 0.0:
            raise ValueError("residual_init_scale must be nonnegative")


def validate_tau_drda(tau_drda: float) -> float:
    if not math.isfinite(tau_drda):
        raise ValueError("tau_drda must be finite")
    if tau_drda < MIN_TAU_DRDA:
        raise ValueError(f"tau_drda must be >= {MIN_TAU_DRDA}")
    return tau_drda


def combined_logits(base_logits: torch.Tensor, residual_logits: torch.Tensor, tau_drda: float) -> torch.Tensor:
    validate_tau_drda(tau_drda)
    _validate_logit_pair(base_logits, residual_logits)
    return base_logits + residual_logits / tau_drda


def drda_rebase(*_: object, **__: object) -> None:
    raise RuntimeError("DRDA residual_adapter_no_rebase does not support neural rebase or optimizer reset")


class ResidualPolicyAdapter(nn.Module):
    """Small public-observation residual policy head for no-rebase DRDA."""

    def __init__(self, hidden: int = 64, residual_init_scale: float = 0.0) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError("residual adapter hidden must be positive")
        if not math.isfinite(residual_init_scale) or residual_init_scale < 0.0:
            raise ValueError("residual_init_scale must be finite and nonnegative")
        self.hidden = hidden
        self.residual_init_scale = residual_init_scale
        self.fc1 = nn.Linear(OBS_CHANNELS * TILE_WIDTH, hidden)
        self.fc2 = nn.Linear(hidden, ACTION_SPACE)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.fc1.bias)
        if self.residual_init_scale == 0.0:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            nn.init.normal_(self.fc2.weight, mean=0.0, std=self.residual_init_scale)
            nn.init.zeros_(self.fc2.bias)

    @override
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3 or obs.shape[1:] != (OBS_CHANNELS, TILE_WIDTH):
            raise ValueError(f"obs must have shape [B,{OBS_CHANNELS},{TILE_WIDTH}]")
        if obs.dtype != torch.float32:
            raise ValueError("obs must be float32")
        if not bool(torch.isfinite(obs).all()):
            raise ValueError("obs must be finite")
        flat = obs.flatten(1)
        return self.fc2(torch.relu(self.fc1(flat)))


class DrdaResidualPolicyNet(nn.Module):
    """Frozen HydraPolicyNet plus trainable policy-logit residual adapter."""

    def __init__(
        self,
        base: HydraPolicyNet,
        config: DrdaResidualConfig | None = None,
        residual: ResidualPolicyAdapter | None = None,
    ) -> None:
        super().__init__()
        self.config = DrdaResidualConfig() if config is None else config
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.residual = (
            ResidualPolicyAdapter(residual_init_scale=self.config.residual_init_scale) if residual is None else residual
        )
        for parameter in self.residual.parameters():
            parameter.requires_grad_(True)

    @override
    def forward(self, obs: torch.Tensor) -> HydraBaseOutput:
        with torch.no_grad():
            base_out = self.base(obs)
        residual_logits = self.residual(obs)
        policy_logits = combined_logits(base_out.policy_logits, residual_logits, self.config.tau_drda)
        return HydraBaseOutput(
            policy_logits=policy_logits,
            value=base_out.value,
            score_pdf=base_out.score_pdf,
            score_cdf=base_out.score_cdf,
            opp_tenpai=base_out.opp_tenpai,
            grp=base_out.grp,
            oracle_critic=base_out.oracle_critic,
            safety_residual=base_out.safety_residual,
            opp_next_discard=base_out.opp_next_discard,
            danger=base_out.danger,
        )

    def residual_parameters(self) -> list[nn.Parameter]:
        return list(self.residual.parameters())

    def rebase(self) -> None:
        drda_rebase()


def residual_optimizer_parameters(model: DrdaResidualPolicyNet) -> list[nn.Parameter]:
    return model.residual_parameters()


def drda_ach_train_step(
    *,
    model: DrdaResidualPolicyNet,
    optimizer: torch.optim.Optimizer,
    batch: PpoBatch,
    entropy_controller: EntropyController,
    config: AchTrainStepConfig,
) -> AchTrainStepResult:
    batch.validate()
    _validate_optimizer_scope(optimizer, model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        base_outputs = model.base(batch.obs)
    residual_logits = model.residual(batch.obs)
    policy_logits = combined_logits(base_outputs.policy_logits, residual_logits, model.config.tau_drda)
    values = base_outputs.value.squeeze(-1)
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
        policy_logits,
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
        raise RuntimeError(f"non-finite DRDA ACH loss: {loss_value}")
    loss_out.total.backward()
    grad_norm = _total_grad_norm(model.residual_parameters())
    if config.grad_clip_norm is not None and config.grad_clip_norm > 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.residual_parameters(), config.grad_clip_norm)
        grad_norm = float(grad_norm_tensor.detach())
    if not math.isfinite(grad_norm):
        raise RuntimeError(f"non-finite DRDA ACH grad norm: {grad_norm}")
    optimizer.step()

    with torch.no_grad():
        kl_to_base = masked_kl(base_outputs.policy_logits, policy_logits.detach(), batch.legal_mask)
    next_controller = entropy_controller.update_default(loss_out.entropy_per_row, batch.legal_count)
    metrics: dict[str, Any] = loss_out.metric_dict()
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
            "objective": DRDA_RESIDUAL_OBJECTIVE,
            "tau_drda": model.config.tau_drda,
            "rebase_enabled": False,
            "total_rebases": 0,
            "base_frozen": True,
            "optimizer_scope": DRDA_OPTIMIZER_SCOPE,
            "kl_to_base_mean": float(kl_to_base.mean()),
        }
    )
    _validate_json_safe_metrics(metrics)
    return AchTrainStepResult(metrics=metrics, entropy_controller=next_controller)


def drda_training_objective_metadata(
    *,
    config: DrdaResidualConfig,
    base_checkpoint_path: Path,
    base_model_config: ModelConfig | Mapping[str, object],
    base_weight_source: DrdaWeightSource = "raw",
    base_checkpoint_sha256: str | None = None,
) -> dict[str, object]:
    if config.rebase_enabled:
        raise ValueError("DRDA checkpoint metadata requires rebase_enabled=False")
    model_config_payload: Mapping[str, object]
    if isinstance(base_model_config, ModelConfig):
        model_config_payload = asdict(base_model_config)
    else:
        model_config_payload = dict(base_model_config)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "mode": DRDA_RESIDUAL_MODE,
        "objective": DRDA_RESIDUAL_OBJECTIVE,
        "tau_drda": validate_tau_drda(config.tau_drda),
        "min_tau_drda": MIN_TAU_DRDA,
        "base_checkpoint_path": str(base_checkpoint_path),
        "base_checkpoint_sha256": base_checkpoint_sha256 or sha256_file(base_checkpoint_path),
        "base_model_config": dict(model_config_payload),
        "base_weight_source": base_weight_source,
        "encoder_shape": [OBS_CHANNELS, TILE_WIDTH],
        "action_space": ACTION_SPACE,
        "residual_representation": DRDA_RESIDUAL_REPRESENTATION,
        "rebase_enabled": False,
        "rebase_capability": DRDA_REBASE_CAPABILITY,
        "total_rebases": 0,
        "optimizer_scope": DRDA_OPTIMIZER_SCOPE,
        "policy_preservation": DRDA_POLICY_PRESERVATION,
        "export_supported": False,
    }
    return metadata


def save_drda_checkpoint(
    path: Path,
    *,
    model: DrdaResidualPolicyNet,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    runtime_config: RuntimeConfig,
    loss_weights: LossWeights,
    manifest_path: Path | None,
    global_step: int,
    samples_seen: int,
    base_checkpoint_path: Path,
    base_weight_source: DrdaWeightSource = "raw",
) -> None:
    _validate_optimizer_scope(optimizer, model)
    save_checkpoint(
        path,
        model=cast(HydraPolicyNet, model),
        optimizer=optimizer,
        model_config=model_config,
        optimizer_config=optimizer_config,
        runtime_config=runtime_config,
        loss_weights=loss_weights,
        manifest_path=manifest_path,
        global_step=global_step,
        samples_seen=samples_seen,
        training_objective=drda_training_objective_metadata(
            config=model.config,
            base_checkpoint_path=base_checkpoint_path,
            base_model_config=model_config,
            base_weight_source=base_weight_source,
        ),
    )


def validate_drda_checkpoint_metadata(
    metadata: Mapping[str, object],
    *,
    expected_config: DrdaResidualConfig,
    expected_base_checkpoint_path: Path,
    expected_base_model_config: ModelConfig | Mapping[str, object],
    expected_base_weight_source: DrdaWeightSource = "raw",
) -> None:
    expected = drda_training_objective_metadata(
        config=expected_config,
        base_checkpoint_path=expected_base_checkpoint_path,
        base_model_config=expected_base_model_config,
        base_weight_source=expected_base_weight_source,
    )
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise ValueError(f"DRDA checkpoint metadata {key} mismatch: got {actual!r} expected {expected_value!r}")


def load_drda_checkpoint(
    path: Path,
    *,
    model: DrdaResidualPolicyNet,
    optimizer: torch.optim.Optimizer,
    expected_model_config: ModelConfig,
    expected_base_checkpoint_path: Path,
    expected_base_weight_source: DrdaWeightSource = "raw",
) -> dict[str, object]:
    checkpoint = _torch_load(path)
    raw_metadata = checkpoint.get("training_objective")
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("DRDA checkpoint missing training_objective metadata")
    validate_drda_checkpoint_metadata(
        raw_metadata,
        expected_config=model.config,
        expected_base_checkpoint_path=expected_base_checkpoint_path,
        expected_base_model_config=expected_model_config,
        expected_base_weight_source=expected_base_weight_source,
    )
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("DRDA checkpoint model_state must be a dict")
    model.load_state_dict(cast(dict[str, torch.Tensor], state), strict=True)
    optimizer_state = checkpoint.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        raise ValueError("DRDA checkpoint optimizer_state must be a dict")
    optimizer.load_state_dict(optimizer_state)
    return dict(raw_metadata)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_logit_pair(base_logits: torch.Tensor, residual_logits: torch.Tensor) -> None:
    if base_logits.shape != residual_logits.shape:
        raise ValueError("base_logits and residual_logits must have the same shape")
    if base_logits.ndim != 2 or base_logits.shape[1] != ACTION_SPACE:
        raise ValueError(f"DRDA logits must have shape [B,{ACTION_SPACE}]")
    if base_logits.device != residual_logits.device:
        raise ValueError("base_logits and residual_logits must be on the same device")
    if base_logits.dtype != residual_logits.dtype:
        raise ValueError("base_logits and residual_logits must have the same dtype")
    if not bool(torch.isfinite(base_logits).all()):
        raise ValueError("base_logits must be finite")
    if not bool(torch.isfinite(residual_logits).all()):
        raise ValueError("residual_logits must be finite")


def _validate_optimizer_scope(optimizer: torch.optim.Optimizer, model: DrdaResidualPolicyNet) -> None:
    residual_ids = {id(parameter) for parameter in model.residual_parameters()}
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id in seen:
                raise ValueError("DRDA optimizer must not contain duplicate residual parameters")
            seen.add(parameter_id)
            if parameter_id not in residual_ids:
                raise ValueError("DRDA optimizer scope must be residual_only")
    if seen != residual_ids:
        raise ValueError("DRDA optimizer must contain every residual parameter exactly once")
