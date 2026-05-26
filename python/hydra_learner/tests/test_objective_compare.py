from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import torch

from hydra_learner.ach_step import AchTrainStepConfig
from hydra_learner.model import ACTION_SPACE, HydraPolicyNet
from hydra_learner.objective_compare import (
    ObjectiveComparisonConfig,
    append_objective_comparison_metrics_jsonl,
    compare_ppo_and_ach_on_rollout_artifact,
)
from hydra_learner.ppo_rollout import PpoRolloutMetadata, load_ppo_rollout_artifact, save_ppo_rollout_artifact
from hydra_learner.ppo_step import PpoBatch, PpoTrainStepConfig
from hydra_learner.rl import EntropyController, masked_log_prob


def test_compare_runs_ppo_and_ach_same_artifact_initial_weights_json_safe(tmp_path: Path) -> None:
    torch.manual_seed(201)
    initial_model = _model_factory()
    initial_state = _clone_state(initial_model)
    artifact_path = tmp_path / "rollout.pt"
    save_ppo_rollout_artifact(
        artifact_path,
        _batch_from_model(initial_model),
        PpoRolloutMetadata(rank_utility_used="U_A", gae_gamma=0.995, gae_lambda=0.95),
    )

    result = compare_ppo_and_ach_on_rollout_artifact(
        artifact_path=artifact_path,
        model_factory=_model_factory,
        initial_state_dict=initial_state,
        optimizer_factory=_optimizer_factory,
        entropy_controller=_entropy_controller(),
        config=_comparison_config(),
    )

    json.dumps(result.metrics, allow_nan=False)
    assert result.metrics["comparison.objectives"] == ["ppo", "direct_sampled_ach"]
    assert result.metrics["comparison.same_artifact"] is True
    assert result.metrics["comparison.rollout_schema_version"] == 1
    assert result.metrics["comparison.rollout_contract_version"] == "ppo_rollout_v1"
    assert result.metrics["artifact.batch_rows"] == 2
    assert result.metrics["artifact.rank_utility_used"] == "U_A"
    assert result.metrics["artifact.gae_gamma"] == pytest.approx(0.995)
    assert result.metrics["artifact.gae_lambda"] == pytest.approx(0.95)
    for key in (
        "ppo.loss_total",
        "ach.loss_total",
        "comparison.ppo_loss_total",
        "comparison.ach_loss_total",
        "comparison.ppo_entropy",
        "comparison.ach_entropy",
        "comparison.ppo_bc_kl_reverse",
        "comparison.ach_bc_kl_reverse",
        "comparison.ppo_grad_norm",
        "comparison.ach_grad_norm",
    ):
        assert key in result.metrics
    assert result.metrics["comparison.ppo_loss_total"] == result.ppo_metrics["loss_total"]
    assert result.metrics["comparison.ach_loss_total"] == result.ach_metrics["loss_total"]
    approved_prefixes = ("ppo.", "ach.", "comparison.", "artifact.")
    assert all(key.startswith(approved_prefixes) for key in result.metrics)
    assert not {"batch_rows", "rank_utility_used", "gae_gamma", "gae_lambda"} & result.metrics.keys()


def test_compare_isolates_model_and_optimizer_state(tmp_path: Path) -> None:
    torch.manual_seed(203)
    initial_model = _model_factory()
    initial_state = _clone_state(initial_model)
    captured_initial_loads: list[dict[str, torch.Tensor]] = []
    constructed: list[HydraPolicyNet] = []

    def factory() -> HydraPolicyNet:
        model = _model_factory()
        original_load = model.load_state_dict

        def load_and_capture(
            state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
        ) -> torch.nn.modules.module._IncompatibleKeys:
            out = original_load(state_dict, strict=strict, assign=assign)
            captured_initial_loads.append(_clone_state(model))
            return out

        model.load_state_dict = load_and_capture  # type: ignore[method-assign] # Test-local hook captures load state.
        constructed.append(model)
        return model

    artifact_path = tmp_path / "rollout.pt"
    save_ppo_rollout_artifact(
        artifact_path, _batch_from_model(initial_model), PpoRolloutMetadata(rank_utility_used="U_A")
    )

    result = compare_ppo_and_ach_on_rollout_artifact(
        artifact_path=artifact_path,
        model_factory=factory,
        initial_state_dict=initial_state,
        optimizer_factory=_optimizer_factory,
        entropy_controller=_entropy_controller(),
        config=_comparison_config(),
    )

    assert len(captured_initial_loads) == 2
    _assert_states_equal(captured_initial_loads[0], initial_state)
    _assert_states_equal(captured_initial_loads[1], initial_state)
    _assert_states_equal(captured_initial_loads[0], captured_initial_loads[1])
    assert len(constructed) == 2
    assert constructed[0] is not constructed[1]
    assert _any_parameter_changed(constructed[0].state_dict(), initial_state)
    assert _any_parameter_changed(constructed[1].state_dict(), initial_state)
    assert result.metrics["comparison.same_artifact"] is True


def test_compare_keeps_ppo_rollout_artifact_schema_unchanged(tmp_path: Path) -> None:
    torch.manual_seed(205)
    model = _model_factory()
    path = tmp_path / "rollout.pt"
    save_ppo_rollout_artifact(path, _batch_from_model(model), PpoRolloutMetadata(rank_utility_used="U_A"))

    artifact = load_ppo_rollout_artifact(path)
    artifact.validate()
    assert artifact.schema_version == 1
    assert artifact.contract_version == "ppo_rollout_v1"

    result = compare_ppo_and_ach_on_rollout_artifact(
        artifact_path=path,
        model_factory=_model_factory,
        initial_state_dict=_clone_state(model),
        optimizer_factory=_optimizer_factory,
        entropy_controller=_entropy_controller(),
        config=_comparison_config(),
    )
    assert result.metrics["comparison.rollout_contract_version"] == "ppo_rollout_v1"


def test_objective_compare_scope_has_no_neurd_residual_tau_or_rebase_config() -> None:
    field_names = {field.name for field in fields(ObjectiveComparisonConfig)}
    nested_names = {field.name for field in fields(PpoTrainStepConfig)} | {
        field.name for field in fields(AchTrainStepConfig)
    }
    forbidden = {"neurd", "residual", "tau", "rebase"}
    assert not (field_names | nested_names) & forbidden


def test_compare_deterministic_for_same_cpu_fixture(tmp_path: Path) -> None:
    torch.manual_seed(207)
    initial_model = _model_factory()
    initial_state = _clone_state(initial_model)
    artifact_path = tmp_path / "rollout.pt"
    save_ppo_rollout_artifact(
        artifact_path, _batch_from_model(initial_model), PpoRolloutMetadata(rank_utility_used="U_A")
    )

    first = compare_ppo_and_ach_on_rollout_artifact(
        artifact_path=artifact_path,
        model_factory=_model_factory,
        initial_state_dict=initial_state,
        optimizer_factory=_optimizer_factory,
        entropy_controller=_entropy_controller(),
        config=_comparison_config(),
    )
    second = compare_ppo_and_ach_on_rollout_artifact(
        artifact_path=artifact_path,
        model_factory=_model_factory,
        initial_state_dict=initial_state,
        optimizer_factory=_optimizer_factory,
        entropy_controller=_entropy_controller(),
        config=_comparison_config(),
    )

    assert first.metrics == second.metrics
    assert first.ppo_metrics == second.ppo_metrics
    assert first.ach_metrics == second.ach_metrics


def test_compare_failure_propagates_invalid_artifact_and_batch(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="failed to load PPO rollout artifact"):
        compare_ppo_and_ach_on_rollout_artifact(
            artifact_path=missing,
            model_factory=_model_factory,
            initial_state_dict=_clone_state(_model_factory()),
            optimizer_factory=_optimizer_factory,
            entropy_controller=_entropy_controller(),
            config=_comparison_config(),
        )

    model = _model_factory()
    bad_path = tmp_path / "bad.pt"
    bad_batch = _batch_from_model(model, bad_legal_count=True)
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract_version": "ppo_rollout_v1",
        "obs": bad_batch.obs,
        "actions": bad_batch.actions,
        "legal_mask": bad_batch.legal_mask,
        "old_logprob": bad_batch.old_logprob,
        "value_old": bad_batch.value_old,
        "raw_advantages": bad_batch.raw_advantages,
        "returns": bad_batch.returns,
        "bc_logits": bad_batch.bc_logits,
        "legal_count": bad_batch.legal_count,
        "metadata": {"rank_utility_used": "U_A", "gae_gamma": 0.995, "gae_lambda": 0.95},
    }
    torch.save(payload, bad_path)

    with pytest.raises(ValueError, match=r"legal_count must equal legal_mask\.sum"):
        compare_ppo_and_ach_on_rollout_artifact(
            artifact_path=bad_path,
            model_factory=_model_factory,
            initial_state_dict=_clone_state(model),
            optimizer_factory=_optimizer_factory,
            entropy_controller=_entropy_controller(),
            config=_comparison_config(),
        )


def test_append_objective_comparison_metrics_jsonl_strict(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "objective_compare.jsonl"
    append_objective_comparison_metrics_jsonl(path, {"comparison.same_artifact": True, "ppo.loss_total": 1.0})
    assert path.read_text(encoding="utf-8") == '{"comparison.same_artifact":true,"ppo.loss_total":1.0}\n'
    with pytest.raises(ValueError, match="must be finite"):
        append_objective_comparison_metrics_jsonl(path, {"bad": float("nan")})


def _model_factory() -> HydraPolicyNet:
    return HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)


def _optimizer_factory(parameters: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=1.0e-3)


def _entropy_controller() -> EntropyController:
    return EntropyController(alpha=1.0e-3, beta=1.0e-2, alpha_max=0.05)


def _comparison_config() -> ObjectiveComparisonConfig:
    return ObjectiveComparisonConfig(
        ppo=PpoTrainStepConfig(bc_kl_reverse_coef=0.01, grad_clip_norm=None),
        ach=AchTrainStepConfig(bc_kl_reverse_coef=0.01, grad_clip_norm=None),
    )


def _batch_from_model(model: HydraPolicyNet, *, bad_legal_count: bool = False) -> PpoBatch:
    obs = torch.randn(2, 192, 34, dtype=torch.float32)
    legal_mask = torch.zeros(2, ACTION_SPACE, dtype=torch.bool)
    legal_mask[0, :2] = True
    legal_mask[1, :5] = True
    actions = torch.tensor([0, 1], dtype=torch.int64)
    with torch.inference_mode():
        out = model(obs)
        old_logprob = masked_log_prob(out.policy_logits, legal_mask, actions).to(dtype=torch.float32)
        value_old = out.value.squeeze(1).to(dtype=torch.float32)
        bc_logits = out.policy_logits.detach().clone().to(dtype=torch.float32)
    legal_count = legal_mask.sum(dim=1).to(dtype=torch.int64)
    if bad_legal_count:
        legal_count = legal_count + torch.tensor([1, 0], dtype=torch.int64)
    return PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=old_logprob,
        value_old=value_old,
        raw_advantages=torch.tensor([1.0, -1.0], dtype=torch.float32),
        returns=value_old + torch.tensor([0.25, -0.25], dtype=torch.float32),
        bc_logits=bc_logits,
        legal_count=legal_count,
        player_id=torch.tensor([0, 1], dtype=torch.int64),
        seat_id=torch.tensor([0, 1], dtype=torch.int64),
        game_id=torch.tensor([1, 1], dtype=torch.int64),
        turn=torch.tensor([0, 1], dtype=torch.int64),
        rank_utility_used="U_A",
    )


def _clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _assert_states_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for key, left_tensor in left.items():
        torch.testing.assert_close(left_tensor, right[key], rtol=0.0, atol=0.0)


def _any_parameter_changed(after: Mapping[str, torch.Tensor], before: Mapping[str, torch.Tensor]) -> bool:
    return any(tensor.is_floating_point() and not torch.equal(tensor, before[name]) for name, tensor in after.items())
