"""Small deterministic PPO-vs-direct-sampled-ACH control-run harness."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import torch

from hydra_learner import arena_eval
from hydra_learner.ach_step import AchTrainStepConfig, ach_train_step
from hydra_learner.checkpoint import ModelConfig, OptimizerConfig, RuntimeConfig, save_checkpoint
from hydra_learner.checkpoint_eval import (
    PairedCheckpointEvalThresholds,
    build_paired_checkpoint_eval_summary,
    paired_checkpoint_eval_summary_to_dict,
)
from hydra_learner.ppo_rollout import PpoRolloutArtifact, artifact_to_ppo_batch, load_ppo_rollout_artifact
from hydra_learner.ppo_step import PpoTrainStepConfig, _validate_json_safe_metrics, ppo_train_step
from hydra_learner.reward_shaping import normalize_reward_shaping_metadata

if TYPE_CHECKING:
    from hydra_learner.losses import LossWeights
    from hydra_learner.model import HydraPolicyNet
    from hydra_learner.rl import EntropyController

ObjectiveName = Literal["ppo", "direct_sampled_ach"]
EvalPairCallable = Callable[[Path, Path, int], Mapping[str, object]]


@dataclass(frozen=True)
class RlObjectiveConfig:
    ppo: PpoTrainStepConfig
    ach: AchTrainStepConfig


@dataclass(frozen=True)
class RlCheckpointConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    runtime: RuntimeConfig
    loss_weights: LossWeights
    manifest_path: Path | None = None


@dataclass(frozen=True)
class RlEvalConfig:
    baseline_objective: ObjectiveName
    candidate_objective: ObjectiveName
    seed: int
    thresholds: PairedCheckpointEvalThresholds | None = None


@dataclass(frozen=True)
class RlControlRunConfig:
    run_id: str
    artifact_paths: tuple[Path, ...]
    update_steps: int
    source_init_id: str
    objectives: RlObjectiveConfig
    checkpoint: RlCheckpointConfig
    output_dir: Path
    eval: RlEvalConfig | None = None


@dataclass(frozen=True)
class RlControlRunResult:
    summary: dict[str, object]
    ppo_checkpoint_path: Path
    ach_checkpoint_path: Path


def make_native_arena_eval_pair(
    *,
    games: int,
    output_path: Path,
    temperature: float = 1.0,
    per_game_path: Path | None = None,
    tensorboard_dir: Path | None = None,
    weight_source: arena_eval.WeightSource = "raw",
    device: str = "cuda:0",
    extension: str | None = None,
    extension_path: Path | None = None,
    arena_batch_decisions: int = 1024,
    rust_native: bool = True,
    arena_threads: int = 0,
    hidden: int = arena_eval.DEFAULT_HIDDEN,
    blocks: int = arena_eval.DEFAULT_BLOCKS,
    bottleneck: int = arena_eval.DEFAULT_SE_BOTTLENECK,
    residual_profile: str = arena_eval.RESIDUAL_PROFILE_DEFAULT,
    backbone_profile: str = arena_eval.BACKBONE_PROFILE_DEFAULT,
    conv_memory_format: str = arena_eval.CONV_MEMORY_FORMAT_DEFAULT,
) -> EvalPairCallable:
    """Build a RL control-run eval_pair callable backed by arena_eval."""

    def eval_pair(baseline: Path, candidate: Path, seed: int) -> Mapping[str, object]:
        return run_rl_native_eval_pair(
            baseline=baseline,
            candidate=candidate,
            seed=seed,
            games=games,
            output_path=output_path,
            temperature=temperature,
            per_game_path=per_game_path,
            tensorboard_dir=tensorboard_dir,
            weight_source=weight_source,
            device=device,
            extension=extension,
            extension_path=extension_path,
            arena_batch_decisions=arena_batch_decisions,
            rust_native=rust_native,
            arena_threads=arena_threads,
            hidden=hidden,
            blocks=blocks,
            bottleneck=bottleneck,
            residual_profile=residual_profile,
            backbone_profile=backbone_profile,
            conv_memory_format=conv_memory_format,
        )

    return eval_pair


def run_rl_native_eval_pair(
    *,
    baseline: Path,
    candidate: Path,
    seed: int,
    games: int,
    output_path: Path,
    temperature: float = 1.0,
    per_game_path: Path | None = None,
    tensorboard_dir: Path | None = None,
    weight_source: arena_eval.WeightSource = "raw",
    device: str = "cuda:0",
    extension: str | None = None,
    extension_path: Path | None = None,
    arena_batch_decisions: int = 1024,
    rust_native: bool = True,
    arena_threads: int = 0,
    hidden: int = arena_eval.DEFAULT_HIDDEN,
    blocks: int = arena_eval.DEFAULT_BLOCKS,
    bottleneck: int = arena_eval.DEFAULT_SE_BOTTLENECK,
    residual_profile: str = arena_eval.RESIDUAL_PROFILE_DEFAULT,
    backbone_profile: str = arena_eval.BACKBONE_PROFILE_DEFAULT,
    conv_memory_format: str = arena_eval.CONV_MEMORY_FORMAT_DEFAULT,
) -> Mapping[str, object]:
    """Run one baseline-vs-candidate arena eval and return normalized arena metrics input."""
    summary = arena_eval.run_arena_eval(
        arena_eval.ArenaEvalConfig(
            baseline=baseline,
            candidates=(candidate,),
            games=games,
            seed=seed,
            temperature=temperature,
            output_path=output_path,
            per_game_path=per_game_path,
            tensorboard_dir=tensorboard_dir,
            weight_source=weight_source,
            device=device,
            extension=extension,
            extension_path=extension_path,
            arena_batch_decisions=arena_batch_decisions,
            rust_native=rust_native,
            arena_threads=arena_threads,
            hidden=hidden,
            blocks=blocks,
            bottleneck=bottleneck,
            residual_profile=residual_profile,
            backbone_profile=backbone_profile,
            conv_memory_format=conv_memory_format,
        )
    )
    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("arena_eval summary must contain exactly one candidate")
    candidate_result = candidates[0]
    if not isinstance(candidate_result, Mapping):
        raise TypeError("arena_eval candidate summary must be a mapping")
    metrics = candidate_result.get("result")
    if not isinstance(metrics, Mapping):
        raise TypeError("arena_eval candidate result must be a mapping")
    return _json_round_trip(metrics)


def run_rl_control_run(
    *,
    config: RlControlRunConfig,
    model_factory: Callable[[], HydraPolicyNet],
    initial_state_dict: Mapping[str, torch.Tensor],
    optimizer_factory: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
    entropy_controller: EntropyController,
    eval_pair: EvalPairCallable | None = None,
) -> RlControlRunResult:
    if config.update_steps < 1:
        raise ValueError("update_steps must be >= 1")
    if not config.artifact_paths:
        raise ValueError("artifact_paths must not be empty")
    if config.eval is not None and eval_pair is None:
        raise ValueError("eval_pair is required when eval config is provided")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ppo_model = model_factory()
    ach_model = model_factory()
    _load_initial_state(ppo_model, initial_state_dict)
    _load_initial_state(ach_model, initial_state_dict)
    _assert_same_model_state(ppo_model, ach_model)

    ppo_optimizer = optimizer_factory(ppo_model.parameters())
    ach_optimizer = optimizer_factory(ach_model.parameters())
    ppo_controller = copy.deepcopy(entropy_controller)
    ach_controller = copy.deepcopy(entropy_controller)

    artifacts = _load_artifact_sequence(config.artifact_paths)
    artifact_metadata = [_artifact_metadata(path, artifact) for path, artifact in artifacts]
    _validate_artifact_reward_contract(artifact_metadata)
    samples_seen = 0
    ppo_metrics: dict[str, object] = {}
    ach_metrics: dict[str, object] = {}

    for step_index in range(config.update_steps):
        _, artifact = artifacts[step_index % len(artifacts)]
        batch = artifact_to_ppo_batch(artifact)
        batch.validate()
        samples_seen += batch.obs.shape[0]

        ppo_result = ppo_train_step(
            model=ppo_model,
            optimizer=ppo_optimizer,
            batch=batch,
            entropy_controller=ppo_controller,
            config=config.objectives.ppo,
        )
        ach_result = ach_train_step(
            model=ach_model,
            optimizer=ach_optimizer,
            batch=batch,
            entropy_controller=ach_controller,
            config=config.objectives.ach,
        )
        ppo_controller = ppo_result.entropy_controller
        ach_controller = ach_result.entropy_controller
        ppo_metrics = dict(ppo_result.metrics)
        ach_metrics = dict(ach_result.metrics)
        _validate_json_safe_metrics(ppo_metrics, "ppo_metrics")
        _validate_json_safe_metrics(ach_metrics, "ach_metrics")

    ppo_checkpoint_path = output_dir / "ppo.pt"
    ach_checkpoint_path = output_dir / "direct_sampled_ach.pt"
    common_training_metadata = {
        "schema_version": 1,
        "source_init_id": config.source_init_id,
        "artifacts": artifact_metadata,
        "artifact_sequence_digest_sha256": _artifact_sequence_digest(artifact_metadata),
        "update_step_count": config.update_steps,
        "reward_contract": _reward_contract(artifact_metadata),
    }
    _save_objective_checkpoint(
        ppo_checkpoint_path,
        model=ppo_model,
        optimizer=ppo_optimizer,
        checkpoint_config=config.checkpoint,
        global_step=config.update_steps,
        samples_seen=samples_seen,
        training_objective={**common_training_metadata, "objective": "ppo"},
    )
    _save_objective_checkpoint(
        ach_checkpoint_path,
        model=ach_model,
        optimizer=ach_optimizer,
        checkpoint_config=config.checkpoint,
        global_step=config.update_steps,
        samples_seen=samples_seen,
        training_objective={**common_training_metadata, "objective": "direct_sampled_ach"},
    )

    checkpoints = {"ppo": str(ppo_checkpoint_path), "direct_sampled_ach": str(ach_checkpoint_path)}
    eval_summary: dict[str, object] | None = None
    if config.eval is not None:
        assert eval_pair is not None
        baseline_path = _objective_checkpoint_path(
            config.eval.baseline_objective, ppo_checkpoint_path, ach_checkpoint_path
        )
        candidate_path = _objective_checkpoint_path(
            config.eval.candidate_objective, ppo_checkpoint_path, ach_checkpoint_path
        )
        arena_metrics = eval_pair(baseline_path, candidate_path, config.eval.seed)
        paired = build_paired_checkpoint_eval_summary(
            baseline=str(baseline_path),
            candidate=str(candidate_path),
            arena_metrics=arena_metrics,
            thresholds=config.eval.thresholds,
            seed=config.eval.seed,
        )
        eval_summary = paired_checkpoint_eval_summary_to_dict(paired)

    summary: dict[str, object] = {
        "run_id": config.run_id,
        "objective_configs": {
            "ppo": _ppo_config_dict(config.objectives.ppo),
            "direct_sampled_ach": asdict(config.objectives.ach),
        },
        "artifact_metadata": artifact_metadata,
        "artifact_sequence_digest_sha256": common_training_metadata["artifact_sequence_digest_sha256"],
        "final_train_metrics": {"ppo": ppo_metrics, "direct_sampled_ach": ach_metrics},
        "checkpoint_paths": checkpoints,
        "source_init_id": config.source_init_id,
        "update_step_count": config.update_steps,
        "reward_contract": common_training_metadata["reward_contract"],
        "paired_eval": eval_summary,
    }
    _validate_json_safe_metrics(summary, "summary")
    summary = _json_round_trip(summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return RlControlRunResult(
        summary=summary,
        ppo_checkpoint_path=ppo_checkpoint_path,
        ach_checkpoint_path=ach_checkpoint_path,
    )


def _ppo_config_dict(config: PpoTrainStepConfig) -> dict[str, object]:
    payload = asdict(config)
    if payload.get("microbatch_size") is None:
        del payload["microbatch_size"]
    return payload


def _load_artifact_sequence(paths: tuple[Path, ...]) -> list[tuple[Path, PpoRolloutArtifact]]:
    return [(path, load_ppo_rollout_artifact(path)) for path in paths]


def _load_initial_state(model: torch.nn.Module, initial_state_dict: Mapping[str, torch.Tensor]) -> None:
    state = {key: tensor.detach().clone() for key, tensor in initial_state_dict.items()}
    model.load_state_dict(state, strict=True)


def _assert_same_model_state(left: torch.nn.Module, right: torch.nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        raise ValueError("model_factory produced incompatible state keys")
    for key, left_tensor in left_state.items():
        if not torch.equal(left_tensor.detach().cpu(), right_state[key].detach().cpu()):
            raise ValueError(f"initial model states differ at {key}")


def _artifact_metadata(path: Path, artifact: PpoRolloutArtifact) -> dict[str, object]:
    return {
        "path": str(path),
        "digest_sha256": _file_sha256(path),
        "kind": "ppo_rollout",
        "schema_version": artifact.schema_version,
        "contract_version": artifact.contract_version,
        "batch_rows": artifact.obs.shape[0],
        "rank_utility_used": artifact.metadata.rank_utility_used,
        "gae_gamma": artifact.metadata.gae_gamma,
        "gae_lambda": artifact.metadata.gae_lambda,
        "reward_shaping": normalize_reward_shaping_metadata(artifact.metadata.reward_shaping),
    }


def _artifact_sequence_digest(metadata: list[dict[str, object]]) -> str:
    payload = json.dumps(metadata, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reward_contract(metadata: list[dict[str, object]]) -> dict[str, object]:
    _validate_artifact_reward_contract(metadata)
    first = metadata[0]
    reward_shaping = normalize_reward_shaping_metadata(cast("Mapping[str, object]", first["reward_shaping"]))
    return {
        "name": "U_A",
        "base_reward": "terminal_U_A",
        "rank_utility_used": first["rank_utility_used"],
        "gae_gamma": first["gae_gamma"],
        "gae_lambda": first["gae_lambda"],
        "state_boundary": reward_shaping["state_boundary"],
        "reward_shaping": reward_shaping,
    }


def _validate_artifact_reward_contract(metadata: list[dict[str, object]]) -> None:
    if not metadata:
        raise ValueError("artifact metadata must not be empty")
    keys = ("rank_utility_used", "gae_gamma", "gae_lambda", "reward_shaping")
    first = {key: metadata[0][key] for key in keys}
    for entry in metadata[1:]:
        for key, expected in first.items():
            if entry[key] != expected:
                raise ValueError(f"mixed rollout reward metadata: {key}")


def _json_round_trip(payload: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")))


def _save_objective_checkpoint(
    path: Path,
    *,
    model: HydraPolicyNet,
    optimizer: torch.optim.Optimizer,
    checkpoint_config: RlCheckpointConfig,
    global_step: int,
    samples_seen: int,
    training_objective: Mapping[str, object],
) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        model_config=checkpoint_config.model,
        optimizer_config=checkpoint_config.optimizer,
        runtime_config=checkpoint_config.runtime,
        loss_weights=checkpoint_config.loss_weights,
        manifest_path=checkpoint_config.manifest_path,
        global_step=global_step,
        samples_seen=samples_seen,
        training_objective=training_objective,
    )


def _objective_checkpoint_path(objective: ObjectiveName, ppo_path: Path, ach_path: Path) -> Path:
    if objective == "ppo":
        return ppo_path
    return ach_path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
