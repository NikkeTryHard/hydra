"""Data-only PPO rollout artifact contract.

Artifacts contain tensors/primitives only and are loaded with ``weights_only=True``.
They carry PPO-ready terminal-rank-utility returns/advantages; raw score fields are
only diagnostic and not part of this training batch contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from hydra_learner.ach_step import AchTrainStepConfig, ach_train_step
from hydra_learner.checkpoint import ModelConfig, OptimizerConfig, RuntimeConfig, save_checkpoint
from hydra_learner.drda import DrdaResidualPolicyNet, drda_ach_train_step
from hydra_learner.ppo_step import PpoBatch, PpoTrainStepConfig, _validate_json_safe_metrics, ppo_train_step
from hydra_learner.reward_shaping import normalize_reward_shaping_metadata
from hydra_learner.rl import DEFAULT_GAE_GAMMA, DEFAULT_GAE_LAMBDA, EntropyController

if TYPE_CHECKING:
    from hydra_learner.losses import LossWeights
    from hydra_learner.model import HydraPolicyNet

PPO_ROLLOUT_SCHEMA_VERSION = 1
PPO_ROLLOUT_CONTRACT_VERSION = "ppo_rollout_v1"


@dataclass(frozen=True)
class PpoRolloutMetadata:
    rank_utility_used: str | None = None
    gae_gamma: float = DEFAULT_GAE_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    reward_shaping: Mapping[str, object] | None = None


@dataclass(frozen=True)
class PpoRolloutArtifact:
    schema_version: int
    contract_version: str
    obs: torch.Tensor
    actions: torch.Tensor
    legal_mask: torch.Tensor
    old_logprob: torch.Tensor
    value_old: torch.Tensor
    raw_advantages: torch.Tensor
    returns: torch.Tensor
    bc_logits: torch.Tensor
    legal_count: torch.Tensor
    metadata: PpoRolloutMetadata
    player_id: torch.Tensor | None = None
    seat_id: torch.Tensor | None = None
    game_id: torch.Tensor | None = None
    turn: torch.Tensor | None = None

    def validate(self) -> None:
        if self.schema_version != PPO_ROLLOUT_SCHEMA_VERSION:
            raise ValueError(f"unsupported PPO rollout schema_version {self.schema_version!r}")
        if self.contract_version != PPO_ROLLOUT_CONTRACT_VERSION:
            raise ValueError(f"unsupported PPO rollout contract_version {self.contract_version!r}")
        artifact_to_ppo_batch(self).validate()
        if not (0.0 < self.metadata.gae_gamma <= 1.0):
            raise ValueError("gae_gamma must be in (0, 1]")
        if not (0.0 < self.metadata.gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in (0, 1]")
        normalize_reward_shaping_metadata(self.metadata.reward_shaping)


@dataclass(frozen=True)
class PpoArtifactTrainStepResult:
    metrics: dict[str, object]
    entropy_controller: EntropyController
    artifact_metadata: dict[str, object]


def save_ppo_rollout_artifact(path: Path, batch: PpoBatch, metadata: PpoRolloutMetadata | None = None) -> None:
    batch.validate()
    if metadata is None:
        metadata = PpoRolloutMetadata(rank_utility_used=batch.rank_utility_used)
    payload: dict[str, object] = {
        "schema_version": PPO_ROLLOUT_SCHEMA_VERSION,
        "contract_version": PPO_ROLLOUT_CONTRACT_VERSION,
        "obs": batch.obs.detach().cpu(),
        "actions": batch.actions.detach().cpu(),
        "legal_mask": batch.legal_mask.detach().cpu(),
        "old_logprob": batch.old_logprob.detach().cpu(),
        "value_old": batch.value_old.detach().cpu(),
        "raw_advantages": batch.raw_advantages.detach().cpu(),
        "returns": batch.returns.detach().cpu(),
        "bc_logits": batch.bc_logits.detach().cpu(),
        "legal_count": batch.legal_count.detach().cpu(),
        "metadata": {
            "rank_utility_used": metadata.rank_utility_used,
            "gae_gamma": metadata.gae_gamma,
            "gae_lambda": metadata.gae_lambda,
            "reward_shaping": _rollout_reward_shaping_metadata(metadata),
        },
    }
    _put_optional_tensor(payload, "player_id", batch.player_id)
    _put_optional_tensor(payload, "seat_id", batch.seat_id)
    _put_optional_tensor(payload, "game_id", batch.game_id)
    _put_optional_tensor(payload, "turn", batch.turn)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_ppo_rollout_artifact(path: Path) -> PpoRolloutArtifact:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"failed to load PPO rollout artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("PPO rollout artifact root must be a dict")
    artifact = _artifact_from_payload(cast("dict[str, object]", payload))
    artifact.validate()
    return artifact


def artifact_to_ppo_batch(artifact: PpoRolloutArtifact) -> PpoBatch:
    return PpoBatch(
        obs=artifact.obs,
        actions=artifact.actions,
        legal_mask=artifact.legal_mask,
        old_logprob=artifact.old_logprob,
        value_old=artifact.value_old,
        raw_advantages=artifact.raw_advantages,
        returns=artifact.returns,
        bc_logits=artifact.bc_logits,
        legal_count=artifact.legal_count,
        player_id=artifact.player_id,
        seat_id=artifact.seat_id,
        game_id=artifact.game_id,
        turn=artifact.turn,
        rank_utility_used=artifact.metadata.rank_utility_used,
    )


def train_step_from_rollout_artifact(
    *,
    artifact_path: Path,
    model: HydraPolicyNet,
    optimizer: torch.optim.Optimizer,
    entropy_controller: EntropyController,
    config: PpoTrainStepConfig,
) -> PpoArtifactTrainStepResult:
    artifact = load_ppo_rollout_artifact(artifact_path)
    result = ppo_train_step(
        model=model,
        optimizer=optimizer,
        batch=artifact_to_ppo_batch(artifact),
        entropy_controller=entropy_controller,
        config=config,
    )
    metadata = _artifact_metadata_dict(artifact)
    metrics: dict[str, object] = dict(result.metrics)
    metrics["rollout_schema_version"] = artifact.schema_version
    metrics["rollout_contract_version"] = artifact.contract_version
    _validate_json_safe_metrics(metrics)
    return PpoArtifactTrainStepResult(
        metrics=metrics,
        entropy_controller=result.entropy_controller,
        artifact_metadata=metadata,
    )


def train_ach_step_from_rollout_artifact(
    *,
    artifact_path: Path,
    model: HydraPolicyNet,
    optimizer: torch.optim.Optimizer,
    entropy_controller: EntropyController,
    config: AchTrainStepConfig,
) -> PpoArtifactTrainStepResult:
    artifact = load_ppo_rollout_artifact(artifact_path)
    result = ach_train_step(
        model=model,
        optimizer=optimizer,
        batch=artifact_to_ppo_batch(artifact),
        entropy_controller=entropy_controller,
        config=config,
    )
    metadata = _artifact_metadata_dict(artifact)
    metrics: dict[str, object] = dict(result.metrics)
    metrics["rollout_schema_version"] = artifact.schema_version
    metrics["rollout_contract_version"] = artifact.contract_version
    _validate_json_safe_metrics(metrics)
    return PpoArtifactTrainStepResult(
        metrics=metrics,
        entropy_controller=result.entropy_controller,
        artifact_metadata=metadata,
    )


def train_drda_ach_step_from_rollout_artifact(
    *,
    artifact_path: Path,
    model: DrdaResidualPolicyNet,
    optimizer: torch.optim.Optimizer,
    entropy_controller: EntropyController,
    config: AchTrainStepConfig,
) -> PpoArtifactTrainStepResult:
    artifact = load_ppo_rollout_artifact(artifact_path)
    result = drda_ach_train_step(
        model=model,
        optimizer=optimizer,
        batch=artifact_to_ppo_batch(artifact),
        entropy_controller=entropy_controller,
        config=config,
    )
    metadata = _artifact_metadata_dict(artifact)
    metrics: dict[str, object] = dict(result.metrics)
    metrics["rollout_schema_version"] = artifact.schema_version
    metrics["rollout_contract_version"] = artifact.contract_version
    _validate_json_safe_metrics(metrics)
    return PpoArtifactTrainStepResult(
        metrics=metrics,
        entropy_controller=result.entropy_controller,
        artifact_metadata=metadata,
    )


def append_ppo_metrics_jsonl(path: Path, metrics: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(metrics, allow_nan=False, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.write("\n")


def save_ppo_training_checkpoint(
    path: Path,
    *,
    model: HydraPolicyNet,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    runtime_config: RuntimeConfig,
    loss_weights: LossWeights,
    manifest_path: Path | None,
    global_step: int,
    samples_seen: int,
) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        optimizer_config=optimizer_config,
        runtime_config=runtime_config,
        loss_weights=loss_weights,
        manifest_path=manifest_path,
        global_step=global_step,
        samples_seen=samples_seen,
    )


def _put_optional_tensor(payload: dict[str, object], key: str, tensor: torch.Tensor | None) -> None:
    if tensor is not None:
        payload[key] = tensor.detach().cpu()


def _artifact_from_payload(payload: dict[str, object]) -> PpoRolloutArtifact:
    required = {
        "schema_version",
        "contract_version",
        "obs",
        "actions",
        "legal_mask",
        "old_logprob",
        "value_old",
        "raw_advantages",
        "returns",
        "bc_logits",
        "legal_count",
        "metadata",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"PPO rollout artifact missing keys: {sorted(missing)}")
    metadata_raw = payload["metadata"]
    if not isinstance(metadata_raw, dict):
        raise ValueError("PPO rollout metadata must be a dict")
    metadata = _metadata_from_payload(cast("dict[str, object]", metadata_raw))
    return PpoRolloutArtifact(
        schema_version=_require_int(payload["schema_version"], "schema_version"),
        contract_version=_require_str(payload["contract_version"], "contract_version"),
        obs=_require_tensor(payload["obs"], "obs"),
        actions=_require_tensor(payload["actions"], "actions"),
        legal_mask=_require_tensor(payload["legal_mask"], "legal_mask"),
        old_logprob=_require_tensor(payload["old_logprob"], "old_logprob"),
        value_old=_require_tensor(payload["value_old"], "value_old"),
        raw_advantages=_require_tensor(payload["raw_advantages"], "raw_advantages"),
        returns=_require_tensor(payload["returns"], "returns"),
        bc_logits=_require_tensor(payload["bc_logits"], "bc_logits"),
        legal_count=_require_tensor(payload["legal_count"], "legal_count"),
        metadata=metadata,
        player_id=_optional_tensor(payload, "player_id"),
        seat_id=_optional_tensor(payload, "seat_id"),
        game_id=_optional_tensor(payload, "game_id"),
        turn=_optional_tensor(payload, "turn"),
    )


def _metadata_from_payload(payload: dict[str, object]) -> PpoRolloutMetadata:
    rank = payload.get("rank_utility_used")
    if rank is not None and not isinstance(rank, str):
        raise ValueError("rank_utility_used must be a string or None")
    gamma = payload.get("gae_gamma", DEFAULT_GAE_GAMMA)
    lam = payload.get("gae_lambda", DEFAULT_GAE_LAMBDA)
    if not isinstance(gamma, int | float):
        raise ValueError("gae_gamma must be numeric")
    if not isinstance(lam, int | float):
        raise ValueError("gae_lambda must be numeric")
    reward_shaping_raw = payload.get("reward_shaping")
    if reward_shaping_raw is not None and not isinstance(reward_shaping_raw, dict):
        raise ValueError("reward_shaping must be a dict")
    reward_shaping = _normalize_payload_reward_shaping(
        cast("Mapping[str, object] | None", reward_shaping_raw), gamma=float(gamma), gae_lambda=float(lam)
    )
    return PpoRolloutMetadata(
        rank_utility_used=rank, gae_gamma=float(gamma), gae_lambda=float(lam), reward_shaping=reward_shaping
    )


def _artifact_metadata_dict(artifact: PpoRolloutArtifact) -> dict[str, object]:
    return {
        "schema_version": artifact.schema_version,
        "contract_version": artifact.contract_version,
        "rank_utility_used": artifact.metadata.rank_utility_used,
        "gae_gamma": artifact.metadata.gae_gamma,
        "gae_lambda": artifact.metadata.gae_lambda,
        "reward_shaping": _rollout_reward_shaping_metadata(artifact.metadata),
        "batch_rows": artifact.obs.shape[0],
    }


def _rollout_reward_shaping_metadata(metadata: PpoRolloutMetadata) -> dict[str, object]:
    return _normalize_payload_reward_shaping(
        metadata.reward_shaping, gamma=metadata.gae_gamma, gae_lambda=metadata.gae_lambda
    )


def _normalize_payload_reward_shaping(
    value: Mapping[str, object] | None, *, gamma: float, gae_lambda: float
) -> dict[str, object]:
    if value is None:
        return normalize_reward_shaping_metadata({"enabled": False, "gae_gamma": gamma, "gae_lambda": gae_lambda})
    return normalize_reward_shaping_metadata(value)


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    return value


def _optional_tensor(payload: dict[str, object], name: str) -> torch.Tensor | None:
    value = payload.get(name)
    if value is None:
        return None
    return _require_tensor(value, name)


def _require_int(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value
