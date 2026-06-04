"""Equal-substrate PPO vs direct sampled ACH comparison harness."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from hydra_learner.ppo.rollout import artifact_to_ppo_batch, load_ppo_rollout_artifact
from hydra_learner.ppo.step import PpoTrainStepConfig, _validate_json_safe_metrics, ppo_train_step
from hydra_learner.rl_experiments.ach_step import AchTrainStepConfig, ach_train_step

if TYPE_CHECKING:
    from hydra_learner.model import HydraPolicyNet
    from hydra_learner.ppo.rl import EntropyController

ObjectiveName = Literal["ppo", "direct_sampled_ach"]


@dataclass(frozen=True)
class ObjectiveComparisonConfig:
    ppo: PpoTrainStepConfig
    ach: AchTrainStepConfig


@dataclass(frozen=True)
class ObjectiveComparisonResult:
    metrics: dict[str, object]
    ppo_metrics: dict[str, object]
    ach_metrics: dict[str, object]


def compare_ppo_and_ach_on_rollout_artifact(
    *,
    artifact_path: Path,
    model_factory: Callable[[], HydraPolicyNet],
    initial_state_dict: Mapping[str, torch.Tensor],
    optimizer_factory: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
    entropy_controller: EntropyController,
    config: ObjectiveComparisonConfig,
) -> ObjectiveComparisonResult:
    artifact = load_ppo_rollout_artifact(artifact_path)
    batch = artifact_to_ppo_batch(artifact)
    batch.validate()

    ppo_model = model_factory()
    ach_model = model_factory()
    _load_initial_state(ppo_model, initial_state_dict)
    _load_initial_state(ach_model, initial_state_dict)
    _assert_same_model_state(ppo_model, ach_model)

    ppo_optimizer = optimizer_factory(ppo_model.parameters())
    ach_optimizer = optimizer_factory(ach_model.parameters())
    ppo_entropy_controller = copy.deepcopy(entropy_controller)
    ach_entropy_controller = copy.deepcopy(entropy_controller)

    ppo_result = ppo_train_step(
        model=ppo_model,
        optimizer=ppo_optimizer,
        batch=batch,
        entropy_controller=ppo_entropy_controller,
        config=config.ppo,
    )
    ach_result = ach_train_step(
        model=ach_model,
        optimizer=ach_optimizer,
        batch=batch,
        entropy_controller=ach_entropy_controller,
        config=config.ach,
    )

    ppo_metrics: dict[str, object] = dict(ppo_result.metrics)
    ach_metrics: dict[str, object] = dict(ach_result.metrics)
    _validate_json_safe_metrics(ppo_metrics, "ppo_metrics")
    _validate_json_safe_metrics(ach_metrics, "ach_metrics")

    metrics = _comparison_metrics(
        ppo_metrics=ppo_metrics,
        ach_metrics=ach_metrics,
        batch_rows=batch.obs.shape[0],
        rank_utility_used=artifact.metadata.rank_utility_used,
        gae_gamma=artifact.metadata.gae_gamma,
        gae_lambda=artifact.metadata.gae_lambda,
        rollout_schema_version=artifact.schema_version,
        rollout_contract_version=artifact.contract_version,
    )
    _validate_json_safe_metrics(metrics)
    return ObjectiveComparisonResult(metrics=metrics, ppo_metrics=ppo_metrics, ach_metrics=ach_metrics)


def append_objective_comparison_metrics_jsonl(path: Path, metrics: Mapping[str, object]) -> None:
    _validate_json_safe_metrics(dict(metrics))
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(metrics, allow_nan=False, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.write("\n")


def _load_initial_state(model: torch.nn.Module, initial_state_dict: Mapping[str, torch.Tensor]) -> None:
    state = {key: tensor.detach().clone() for key, tensor in initial_state_dict.items()}
    model.load_state_dict(state, strict=True)


def _assert_same_model_state(left: torch.nn.Module, right: torch.nn.Module) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        raise ValueError("model_factory produced incompatible state keys")
    for key, left_tensor in left_state.items():
        right_tensor = right_state[key]
        if not torch.equal(left_tensor.detach().cpu(), right_tensor.detach().cpu()):
            raise ValueError(f"initial model states differ at {key}")


def _comparison_metrics(
    *,
    ppo_metrics: Mapping[str, object],
    ach_metrics: Mapping[str, object],
    batch_rows: int,
    rank_utility_used: str | None,
    gae_gamma: float,
    gae_lambda: float,
    rollout_schema_version: int,
    rollout_contract_version: str,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for key, value in ppo_metrics.items():
        metrics[f"ppo.{key}"] = value
    for key, value in ach_metrics.items():
        metrics[f"ach.{key}"] = value
    metrics.update(
        {
            "comparison.objectives": ["ppo", "direct_sampled_ach"],
            "comparison.rollout_schema_version": rollout_schema_version,
            "comparison.rollout_contract_version": rollout_contract_version,
            "comparison.same_artifact": True,
            "comparison.ppo_loss_total": _required_number(ppo_metrics, "loss_total", "ppo_metrics"),
            "comparison.ach_loss_total": _required_number(ach_metrics, "loss_total", "ach_metrics"),
            "comparison.ppo_entropy": _required_number(ppo_metrics, "entropy", "ppo_metrics"),
            "comparison.ach_entropy": _required_number(ach_metrics, "entropy", "ach_metrics"),
            "comparison.ppo_bc_kl_reverse": _required_number(ppo_metrics, "bc_kl_reverse", "ppo_metrics"),
            "comparison.ach_bc_kl_reverse": _required_number(ach_metrics, "bc_kl_reverse", "ach_metrics"),
            "comparison.ppo_grad_norm": _required_number(ppo_metrics, "grad_norm", "ppo_metrics"),
            "comparison.ach_grad_norm": _required_number(ach_metrics, "grad_norm", "ach_metrics"),
            "artifact.batch_rows": batch_rows,
            "artifact.rank_utility_used": rank_utility_used,
            "artifact.gae_gamma": gae_gamma,
            "artifact.gae_lambda": gae_lambda,
        }
    )
    return metrics


def _required_number(metrics: Mapping[str, object], key: str, owner: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{owner}.{key} must be numeric")
    return float(value)
