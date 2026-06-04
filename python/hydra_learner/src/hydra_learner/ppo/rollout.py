"""Data-only PPO rollout artifact contract.

Artifacts contain tensors/primitives only and are loaded with ``weights_only=True``.
They carry PPO-ready terminal-rank-utility returns/advantages; raw score fields are
only diagnostic and not part of this training batch contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from hydra_learner.checkpointing.core import ModelConfig, OptimizerConfig, RuntimeConfig, save_checkpoint
from hydra_learner.ppo.rl import DEFAULT_GAE_GAMMA, DEFAULT_GAE_LAMBDA, EntropyController
from hydra_learner.ppo.step import PpoBatch, PpoTrainStepConfig, _validate_json_safe_metrics, ppo_train_step
from hydra_learner.rl_experiments.ach_step import AchTrainStepConfig, ach_train_step
from hydra_learner.rl_experiments.drda import DrdaResidualPolicyNet, drda_ach_train_step
from hydra_learner.rl_experiments.reward_shaping import normalize_reward_shaping_metadata

if TYPE_CHECKING:
    from hydra_learner.model import HydraPolicyNet
    from hydra_learner.model.losses import LossWeights

PPO_ROLLOUT_SCHEMA_VERSION = 1
PPO_ROLLOUT_CONTRACT_VERSION = "ppo_rollout_v1"
PPO_SNAPSHOT_CONTRACT_VERSION = "ppo_snapshot_v1"
PPO_POLICY_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PpoPolicySnapshotArtifact:
    schema_version: int
    contract_version: str
    snapshot_metadata: PpoSnapshotMetadata
    model_config: ModelConfig
    model_state: dict[str, torch.Tensor]
    torch_version: str


def save_ppo_policy_snapshot_artifact(
    run_dir: Path, *, model: HydraPolicyNet, model_config: ModelConfig, snapshot: PpoSnapshotMetadata
) -> tuple[Path, int]:
    payload: dict[str, object] = {
        "schema_version": PPO_POLICY_SNAPSHOT_SCHEMA_VERSION,
        "contract_version": PPO_SNAPSHOT_CONTRACT_VERSION,
        "snapshot_metadata": snapshot.to_payload(),
        "model_config": asdict(model_config),
        "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "torch_version": str(torch.__version__),
    }
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    final_path = snapshot_dir / f"snapshot_{snapshot.global_step:08d}_{snapshot.snapshot_id}.pt"
    tmp_path = final_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(final_path)
    return final_path, final_path.stat().st_size


def load_ppo_policy_snapshot_artifact(
    path: Path, *, expected_snapshot: PpoSnapshotMetadata, expected_model_config: ModelConfig
) -> PpoPolicySnapshotArtifact:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"failed to load PPO policy snapshot artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("PPO policy snapshot artifact root must be a dict")
    schema_version = _require_int(payload.get("schema_version"), "schema_version")
    if schema_version != PPO_POLICY_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"unsupported PPO policy snapshot schema_version {schema_version!r}")
    contract_version = _require_str(payload.get("contract_version"), "contract_version")
    if contract_version != PPO_SNAPSHOT_CONTRACT_VERSION:
        raise ValueError(f"unsupported PPO policy snapshot contract_version {contract_version!r}")
    metadata_raw = payload.get("snapshot_metadata")
    if not isinstance(metadata_raw, Mapping):
        raise ValueError("PPO policy snapshot metadata must be a mapping")
    metadata = snapshot_metadata_from_payload(metadata_raw)
    if metadata != expected_snapshot:
        raise ValueError("PPO policy snapshot metadata does not match expected snapshot")
    model_config_raw = payload.get("model_config")
    if not isinstance(model_config_raw, Mapping):
        raise ValueError("PPO policy snapshot model_config must be a mapping")
    model_config = _model_config_from_payload(model_config_raw)
    if model_config != expected_model_config:
        raise ValueError("PPO policy snapshot model_config does not match expected config")
    model_state_raw = payload.get("model_state")
    if not isinstance(model_state_raw, dict):
        raise ValueError("PPO policy snapshot model_state must be a dict")
    model_state: dict[str, torch.Tensor] = {}
    for name, tensor in model_state_raw.items():
        if not isinstance(name, str):
            raise ValueError("PPO policy snapshot model_state keys must be strings")
        model_state[name] = _require_tensor(tensor, f"model_state[{name}]")
        if model_state[name].device.type != "cpu":
            raise ValueError("PPO policy snapshot model_state tensors must be CPU tensors")
    return PpoPolicySnapshotArtifact(
        schema_version=schema_version,
        contract_version=contract_version,
        snapshot_metadata=metadata,
        model_config=model_config,
        model_state=model_state,
        torch_version=_require_str(payload.get("torch_version"), "torch_version"),
    )


@dataclass(frozen=True)
class PpoSnapshotMetadata:
    snapshot_contract_version: str
    snapshot_id: str
    config_digest_sha256: str
    global_step: int
    samples_seen: int
    completed_games: int
    rollout_seed: int
    temperature: float
    inference_backend: str
    hidden: int
    blocks: int
    bottleneck: int
    residual_profile: str
    backbone_profile: str
    conv_memory_format: str
    encoder_shape: tuple[int, int]
    action_space: int
    device: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "snapshot_contract_version": self.snapshot_contract_version,
            "snapshot_id": self.snapshot_id,
            "config_digest_sha256": self.config_digest_sha256,
            "global_step": self.global_step,
            "samples_seen": self.samples_seen,
            "completed_games": self.completed_games,
            "rollout_seed": self.rollout_seed,
            "temperature": self.temperature,
            "inference_backend": self.inference_backend,
            "device": self.device,
            "hidden": self.hidden,
            "blocks": self.blocks,
            "bottleneck": self.bottleneck,
            "residual_profile": self.residual_profile,
            "backbone_profile": self.backbone_profile,
            "conv_memory_format": self.conv_memory_format,
            "encoder_shape": list(self.encoder_shape),
            "action_space": self.action_space,
        }


@dataclass(frozen=True)
class PpoRolloutMetadata:
    rank_utility_used: str | None = None
    gae_gamma: float = DEFAULT_GAE_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    reward_shaping: Mapping[str, object] | None = None
    snapshot: PpoSnapshotMetadata | None = None


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
            "snapshot": None if metadata.snapshot is None else metadata.snapshot.to_payload(),
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
        snapshot_metadata=None if artifact.metadata.snapshot is None else artifact.metadata.snapshot.to_payload(),
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
    _put_snapshot_metrics(metrics, artifact.metadata.snapshot)
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
    _put_snapshot_metrics(metrics, artifact.metadata.snapshot)
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
    _put_snapshot_metrics(metrics, artifact.metadata.snapshot)
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
    snapshot_raw = payload.get("snapshot")
    if snapshot_raw is not None and not isinstance(snapshot_raw, dict):
        raise ValueError("snapshot metadata must be a dict")
    snapshot = None if snapshot_raw is None else snapshot_metadata_from_payload(cast("dict[str, object]", snapshot_raw))
    if payload.get("snapshot_required") is True and snapshot_raw is None:
        raise ValueError("snapshot metadata is required")
    return PpoRolloutMetadata(
        rank_utility_used=rank,
        gae_gamma=float(gamma),
        gae_lambda=float(lam),
        reward_shaping=reward_shaping,
        snapshot=snapshot,
    )


def build_ppo_snapshot_metadata(
    *,
    config_digest_sha256: str,
    global_step: int,
    samples_seen: int,
    completed_games: int,
    rollout_seed: int,
    temperature: float,
    inference_backend: str,
    hidden: int,
    blocks: int,
    bottleneck: int,
    residual_profile: str,
    backbone_profile: str,
    conv_memory_format: str,
    encoder_shape: tuple[int, int],
    action_space: int,
    device: str | None = None,
) -> PpoSnapshotMetadata:
    fields: dict[str, object] = {
        "snapshot_contract_version": PPO_SNAPSHOT_CONTRACT_VERSION,
        "config_digest_sha256": config_digest_sha256,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "completed_games": completed_games,
        "rollout_seed": rollout_seed,
        "temperature": temperature,
        "inference_backend": inference_backend,
        "device": device,
        "hidden": hidden,
        "blocks": blocks,
        "bottleneck": bottleneck,
        "residual_profile": residual_profile,
        "backbone_profile": backbone_profile,
        "conv_memory_format": conv_memory_format,
        "encoder_shape": list(encoder_shape),
        "action_space": action_space,
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    snapshot_id = hashlib.sha256(encoded).hexdigest()
    return PpoSnapshotMetadata(
        snapshot_contract_version=cast("str", fields["snapshot_contract_version"]),
        snapshot_id=snapshot_id,
        config_digest_sha256=config_digest_sha256,
        global_step=global_step,
        samples_seen=samples_seen,
        completed_games=completed_games,
        rollout_seed=rollout_seed,
        temperature=temperature,
        inference_backend=inference_backend,
        device=device,
        hidden=hidden,
        blocks=blocks,
        bottleneck=bottleneck,
        residual_profile=residual_profile,
        backbone_profile=backbone_profile,
        conv_memory_format=conv_memory_format,
        encoder_shape=encoder_shape,
        action_space=action_space,
    )


def snapshot_metadata_from_payload(payload: Mapping[str, object]) -> PpoSnapshotMetadata:
    device_raw = payload.get("device")
    if device_raw is not None and not isinstance(device_raw, str):
        raise ValueError("device must be a string")
    snapshot = PpoSnapshotMetadata(
        snapshot_contract_version=_require_str(payload.get("snapshot_contract_version"), "snapshot_contract_version"),
        snapshot_id=_require_str(payload.get("snapshot_id"), "snapshot_id"),
        config_digest_sha256=_require_str(payload.get("config_digest_sha256"), "config_digest_sha256"),
        global_step=_require_int(payload.get("global_step"), "global_step"),
        samples_seen=_require_int(payload.get("samples_seen"), "samples_seen"),
        completed_games=_require_int(payload.get("completed_games"), "completed_games"),
        rollout_seed=_require_int(payload.get("rollout_seed"), "rollout_seed"),
        temperature=_require_float(payload.get("temperature"), "temperature"),
        inference_backend=_require_str(payload.get("inference_backend"), "inference_backend"),
        device=device_raw,
        hidden=_require_int(payload.get("hidden"), "hidden"),
        blocks=_require_int(payload.get("blocks"), "blocks"),
        bottleneck=_require_int(payload.get("bottleneck"), "bottleneck"),
        residual_profile=_require_str(payload.get("residual_profile"), "residual_profile"),
        backbone_profile=_require_str(payload.get("backbone_profile"), "backbone_profile"),
        conv_memory_format=_require_str(payload.get("conv_memory_format"), "conv_memory_format"),
        encoder_shape=_require_int_pair(payload.get("encoder_shape"), "encoder_shape"),
        action_space=_require_int(payload.get("action_space"), "action_space"),
    )
    if snapshot.snapshot_contract_version != PPO_SNAPSHOT_CONTRACT_VERSION:
        raise ValueError(f"unsupported PPO snapshot_contract_version {snapshot.snapshot_contract_version!r}")
    expected = build_ppo_snapshot_metadata(
        config_digest_sha256=snapshot.config_digest_sha256,
        global_step=snapshot.global_step,
        samples_seen=snapshot.samples_seen,
        completed_games=snapshot.completed_games,
        rollout_seed=snapshot.rollout_seed,
        temperature=snapshot.temperature,
        inference_backend=snapshot.inference_backend,
        device=snapshot.device,
        hidden=snapshot.hidden,
        blocks=snapshot.blocks,
        bottleneck=snapshot.bottleneck,
        residual_profile=snapshot.residual_profile,
        backbone_profile=snapshot.backbone_profile,
        conv_memory_format=snapshot.conv_memory_format,
        encoder_shape=snapshot.encoder_shape,
        action_space=snapshot.action_space,
    )
    if snapshot.snapshot_id != expected.snapshot_id:
        raise ValueError("snapshot_id does not match snapshot metadata")
    return snapshot


def _model_config_from_payload(payload: Mapping[str, object]) -> ModelConfig:
    return ModelConfig(
        hidden=_require_int(payload.get("hidden"), "hidden"),
        blocks=_require_int(payload.get("blocks"), "blocks"),
        bottleneck=_require_int(payload.get("bottleneck"), "bottleneck"),
        actions=_require_int(payload.get("actions"), "actions"),
        residual_profile=_require_str(payload.get("residual_profile"), "residual_profile"),
        backbone_profile=_require_str(payload.get("backbone_profile"), "backbone_profile"),
        conv_memory_format=_require_str(payload.get("conv_memory_format"), "conv_memory_format"),
        head_mode=_require_str(payload.get("head_mode"), "head_mode"),
        encoder_shape=_require_int_pair(payload.get("encoder_shape"), "encoder_shape"),
    )


def _put_snapshot_metrics(metrics: dict[str, object], snapshot: PpoSnapshotMetadata | None) -> None:
    if snapshot is None:
        return
    metrics["snapshot_id"] = snapshot.snapshot_id
    metrics["snapshot_global_step"] = snapshot.global_step
    metrics["snapshot_samples_seen"] = snapshot.samples_seen
    metrics["snapshot_completed_games"] = snapshot.completed_games


def _artifact_metadata_dict(artifact: PpoRolloutArtifact) -> dict[str, object]:
    return {
        "schema_version": artifact.schema_version,
        "contract_version": artifact.contract_version,
        "rank_utility_used": artifact.metadata.rank_utility_used,
        "gae_gamma": artifact.metadata.gae_gamma,
        "gae_lambda": artifact.metadata.gae_lambda,
        "reward_shaping": _rollout_reward_shaping_metadata(artifact.metadata),
        "batch_rows": artifact.obs.shape[0],
        "snapshot": None if artifact.metadata.snapshot is None else artifact.metadata.snapshot.to_payload(),
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


def _require_float(value: object, name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    out = float(value)
    if not torch.isfinite(torch.tensor(out)):
        raise ValueError(f"{name} must be finite")
    return out


def _require_int_pair(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, list) and not isinstance(value, tuple):
        raise ValueError(f"{name} must contain two ints")
    values = cast("tuple[object, ...] | list[object]", value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain two ints")
    first, second = values
    if not isinstance(first, int) or not isinstance(second, int):
        raise ValueError(f"{name} must contain two ints")
    return (first, second)


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value
