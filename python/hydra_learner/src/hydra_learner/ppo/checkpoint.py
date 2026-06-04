from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from hydra_learner.checkpointing.core import ModelConfig, OptimizerConfig, RuntimeConfig, save_checkpoint
from hydra_learner.model import HydraPolicyNet
from hydra_learner.model.optim import adamw_flag_value
from hydra_learner.ppo.config import OBJECTIVE, RANK_UTILITY, PpoControlConfig
from hydra_learner.ppo.rl import DEFAULT_GAE_GAMMA, DEFAULT_GAE_LAMBDA
from hydra_learner.rl_experiments.reward_shaping import default_reward_shaping_metadata

if TYPE_CHECKING:
    from hydra_learner.model.losses import LossWeights
    from hydra_learner.ppo.rollout import PpoSnapshotMetadata


def _save_t1_checkpoint(
    path: Path,
    config: PpoControlConfig,
    model: HydraPolicyNet,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    runtime_config: RuntimeConfig,
    loss_weights: LossWeights,
    global_step: int,
    samples_seen: int,
    completed_games: int,
    config_digest: str,
    snapshot: PpoSnapshotMetadata | None = None,
) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        optimizer_config=optimizer_config,
        runtime_config=runtime_config,
        loss_weights=loss_weights,
        manifest_path=None,
        global_step=global_step,
        samples_seen=samples_seen,
        raw_mjai_progress={
            "completed_games": completed_games,
            "rollout_seed_cursor": config.seed + completed_games,
            "latest_snapshot": None if snapshot is None else snapshot.to_payload(),
        },
        training_objective={
            "schema_version": 1,
            "objective": OBJECTIVE,
            "mode": OBJECTIVE,
            "rank_utility_used": RANK_UTILITY,
            "gae_gamma": DEFAULT_GAE_GAMMA,
            "gae_lambda": DEFAULT_GAE_LAMBDA,
            "reward_shaping": default_reward_shaping_metadata(gamma=DEFAULT_GAE_GAMMA, gae_lambda=DEFAULT_GAE_LAMBDA),
            "config_digest_sha256": config_digest,
            "latest_snapshot": None if snapshot is None else snapshot.to_payload(),
            "disabled_capabilities": {
                "ach": True,
                "drda": True,
                "pbrs": True,
                "exit_deltaq_positive_weights": True,
                "privileged_critic": True,
            },
        },
    )


def _validate_resume_metadata(
    path: Path, config_digest: str, compatible_config_digests: set[str] | None = None
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    objective = payload.get("training_objective") if isinstance(payload, dict) else None
    if not isinstance(objective, Mapping):
        raise ValueError("T1 PPO resume checkpoint missing training_objective")
    if objective.get("objective") != OBJECTIVE or objective.get("mode") != OBJECTIVE:
        raise ValueError("T1 PPO resume checkpoint objective mismatch")
    allowed_digests = {config_digest}
    if compatible_config_digests is not None:
        allowed_digests.update(compatible_config_digests)
    if objective.get("config_digest_sha256") not in allowed_digests:
        raise ValueError("T1 PPO resume config digest mismatch")
    if objective.get("rank_utility_used") != RANK_UTILITY:
        raise ValueError("T1 PPO resume rank utility mismatch")


def _model_config(config: PpoControlConfig) -> ModelConfig:
    return ModelConfig(
        hidden=config.hidden,
        blocks=config.blocks,
        bottleneck=config.bottleneck,
        residual_profile=config.residual_profile,
        backbone_profile=config.backbone_profile,
        conv_memory_format=config.conv_memory_format,
    )


def _model(config: PpoControlConfig) -> HydraPolicyNet:
    return HydraPolicyNet(
        hidden=config.hidden,
        blocks=config.blocks,
        bottleneck=config.bottleneck,
        residual_profile=config.residual_profile,
        backbone_profile=config.backbone_profile,
        conv_memory_format=config.conv_memory_format,
    )


def _optimizer_config(config: PpoControlConfig) -> OptimizerConfig:
    return OptimizerConfig(
        name="AdamW",
        lr=config.lr,
        min_lr=config.min_lr,
        lr_schedule="constant" if config.lr_decay_samples is None else "cosine",
        lr_warmup_steps=0,
        schedule_total_steps=None,
        target_games=config.lr_decay_samples,
        grad_clip_norm=config.grad_clip_norm,
        weight_decay=config.weight_decay,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        eps=config.adam_eps,
        foreach=adamw_flag_value(config.adamw_foreach),
        fused=adamw_flag_value(config.adamw_fused),
    )


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        variant=OBJECTIVE, loss_mode="masked_ppo_gae", precision_mode="fp32", compile_fullgraph_check=False
    )
