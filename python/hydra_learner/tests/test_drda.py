from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path
from typing import cast

import pytest
import torch

from hydra_learner.checkpointing.core import ModelConfig, OptimizerConfig, RuntimeConfig
from hydra_learner.model import ACTION_SPACE, OBS_CHANNELS, TILE_WIDTH, HydraPolicyNet
from hydra_learner.model.losses import LossWeights
from hydra_learner.ppo.rl import AchLossConfig, EntropyController, ach_loss, masked_kl, masked_probs
from hydra_learner.ppo.rollout import (
    PpoRolloutMetadata,
    save_ppo_rollout_artifact,
    train_drda_ach_step_from_rollout_artifact,
)
from hydra_learner.ppo.step import PpoBatch
from hydra_learner.rl_experiments.ach_step import AchTrainStepConfig
from hydra_learner.rl_experiments.drda import (
    DEFAULT_TAU_DRDA,
    DRDA_OPTIMIZER_SCOPE,
    DRDA_POLICY_PRESERVATION,
    DRDA_REBASE_CAPABILITY,
    DRDA_RESIDUAL_MODE,
    DRDA_RESIDUAL_OBJECTIVE,
    MIN_TAU_DRDA,
    DrdaResidualConfig,
    DrdaResidualPolicyNet,
    combined_logits,
    drda_ach_train_step,
    drda_rebase,
    drda_training_objective_metadata,
    load_drda_checkpoint,
    residual_optimizer_parameters,
    save_drda_checkpoint,
    validate_drda_checkpoint_metadata,
    validate_tau_drda,
)


def test_drda_combined_logits_formula() -> None:
    base = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    residual = torch.tensor([[4.0, 8.0, 12.0]], dtype=torch.float32)

    actual = combined_logits(_pad_logits(base), _pad_logits(residual), 4.0)

    torch.testing.assert_close(actual[:, :3], torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float32))


@pytest.mark.parametrize("tau", [math.nan, math.inf, -math.inf, 1.999])
def test_drda_tau_validation_rejects_invalid(tau: float) -> None:
    with pytest.raises(ValueError, match="tau_drda"):
        validate_tau_drda(tau)


@pytest.mark.parametrize("tau", [2.0, DEFAULT_TAU_DRDA, 8.0])
def test_drda_tau_validation_accepts_supported_grid(tau: float) -> None:
    assert validate_tau_drda(tau) == tau


def test_drda_rejects_rebase_enabled_config() -> None:
    with pytest.raises(ValueError, match="rebase_enabled=False"):
        DrdaResidualConfig(rebase_enabled=True)


def test_drda_combined_logits_validates_contract() -> None:
    base = torch.zeros(1, ACTION_SPACE)
    residual = torch.zeros(1, ACTION_SPACE)
    with pytest.raises(ValueError, match="shape"):
        combined_logits(torch.zeros(1, ACTION_SPACE + 1), torch.zeros(1, ACTION_SPACE + 1), 4.0)
    with pytest.raises(ValueError, match="same shape"):
        combined_logits(base, torch.zeros(2, ACTION_SPACE), 4.0)
    with pytest.raises(ValueError, match="same dtype"):
        combined_logits(base, residual.to(dtype=torch.float64), 4.0)
    bad = residual.clone()
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="residual_logits"):
        combined_logits(base, bad, 4.0)


def test_drda_zero_residual_matches_base_logits_and_masked_distribution() -> None:
    torch.manual_seed(10)
    base = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    model = DrdaResidualPolicyNet(base, DrdaResidualConfig(tau_drda=4.0, residual_init_scale=0.0))
    obs = torch.randn(2, OBS_CHANNELS, TILE_WIDTH)
    legal_mask = torch.zeros(2, ACTION_SPACE, dtype=torch.bool)
    legal_mask[0, :2] = True
    legal_mask[1, :5] = True

    with torch.inference_mode():
        base_out = base(obs)
        drda_out = model(obs)

    torch.testing.assert_close(drda_out.policy_logits, base_out.policy_logits)
    torch.testing.assert_close(
        masked_probs(drda_out.policy_logits, legal_mask),
        masked_probs(base_out.policy_logits, legal_mask),
    )
    torch.testing.assert_close(drda_out.value, base_out.value)
    torch.testing.assert_close(drda_out.safety_residual, base_out.safety_residual)


def test_drda_tau_controls_kl_to_base() -> None:
    legal_mask = torch.ones(1, ACTION_SPACE, dtype=torch.bool)
    base = torch.zeros(1, ACTION_SPACE)
    residual = torch.zeros(1, ACTION_SPACE)
    residual[0, 0] = 4.0
    residual[0, 1] = -2.0

    kl_tau_2 = masked_kl(base, combined_logits(base, residual, 2.0), legal_mask)
    kl_tau_8 = masked_kl(base, combined_logits(base, residual, 8.0), legal_mask)

    assert float(kl_tau_2) > float(kl_tau_8)


def test_drda_base_frozen_residual_trainable_and_optimizer_scope() -> None:
    model = DrdaResidualPolicyNet(HydraPolicyNet(hidden=8, blocks=1, bottleneck=4))
    params = residual_optimizer_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=1.0e-3)

    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert all(parameter.requires_grad for parameter in model.residual.parameters())
    assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == {
        id(parameter) for parameter in model.residual.parameters()
    }
    with pytest.raises(ValueError, match="residual_only"):
        drda_ach_train_step(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1.0e-3),
            batch=_batch_from_model(model.base),
            entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
            config=AchTrainStepConfig(grad_clip_norm=None),
        )
    duplicate_optimizer = torch.optim.AdamW(params, lr=1.0e-3)
    duplicate_optimizer.param_groups[0]["params"].append(params[0])
    with pytest.raises(ValueError, match="duplicate residual parameters"):
        drda_ach_train_step(
            model=model,
            optimizer=duplicate_optimizer,
            batch=_batch_from_model(model.base),
            entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
            config=AchTrainStepConfig(grad_clip_norm=None),
        )


def test_drda_ach_step_updates_residual_only_and_records_metrics() -> None:
    torch.manual_seed(12)
    model = DrdaResidualPolicyNet(HydraPolicyNet(hidden=8, blocks=1, bottleneck=4))
    optimizer = torch.optim.AdamW(residual_optimizer_parameters(model), lr=1.0e-2)
    batch = _batch_from_model(model.base)
    base_before = _clone_state(model.base)
    residual_before = _clone_state(model.residual)

    result = drda_ach_train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=AchTrainStepConfig(value_coef=0.0, grad_clip_norm=None),
    )

    json.dumps(result.metrics, allow_nan=False)
    assert result.metrics["objective"] == DRDA_RESIDUAL_OBJECTIVE
    assert result.metrics["tau_drda"] == pytest.approx(DEFAULT_TAU_DRDA)
    assert result.metrics["rebase_enabled"] is False
    assert result.metrics["total_rebases"] == 0
    assert result.metrics["base_frozen"] is True
    assert result.metrics["optimizer_scope"] == DRDA_OPTIMIZER_SCOPE
    assert "kl_to_base_mean" in result.metrics
    _assert_states_equal(model.base.state_dict(), base_before)
    assert _any_parameter_changed(model.residual.state_dict(), residual_before)


def test_drda_ach_uses_combined_logits_without_changing_direct_ach() -> None:
    base_logits = torch.zeros(1, ACTION_SPACE)
    residual_logits = torch.zeros(1, ACTION_SPACE)
    residual_logits[0, 0] = 8.0
    logits = combined_logits(base_logits, residual_logits, 4.0)
    mask = torch.zeros(1, ACTION_SPACE, dtype=torch.bool)
    mask[0, :2] = True
    cfg = AchLossConfig(value_coef=0.0, entropy_alpha=0.0)

    drda_out = ach_loss(
        logits,
        torch.zeros(1),
        torch.tensor([0]),
        mask,
        torch.log(torch.tensor([0.5])),
        torch.tensor([1.0]),
        torch.zeros(1),
        config=cfg,
    )
    direct_out = ach_loss(
        base_logits,
        torch.zeros(1),
        torch.tensor([0]),
        mask,
        torch.log(torch.tensor([0.5])),
        torch.tensor([1.0]),
        torch.zeros(1),
        config=cfg,
    )

    assert drda_out.metrics.entropy != pytest.approx(float(direct_out.metrics.entropy))
    assert drda_out.metrics.gate_fraction != pytest.approx(float(direct_out.metrics.gate_fraction))
    direct_fields = {field.name for field in fields(AchTrainStepConfig)} | {
        field.name for field in fields(AchLossConfig)
    }
    assert not {"tau", "tau_drda", "rebase", "residual"} & direct_fields


def test_drda_rollout_artifact_consumes_same_ppo_batch_without_schema_change(tmp_path: Path) -> None:
    model = DrdaResidualPolicyNet(HydraPolicyNet(hidden=8, blocks=1, bottleneck=4))
    optimizer = torch.optim.AdamW(residual_optimizer_parameters(model), lr=1.0e-2)
    path = tmp_path / "rollout.pt"
    save_ppo_rollout_artifact(path, _batch_from_model(model.base), PpoRolloutMetadata(rank_utility_used="U_A"))

    result = train_drda_ach_step_from_rollout_artifact(
        artifact_path=path,
        model=model,
        optimizer=optimizer,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=AchTrainStepConfig(value_coef=0.0, grad_clip_norm=None),
    )

    assert result.metrics["objective"] == DRDA_RESIDUAL_OBJECTIVE
    assert result.metrics["rollout_schema_version"] == 1
    assert result.metrics["rollout_contract_version"] == "ppo_rollout_v1"
    assert result.artifact_metadata["contract_version"] == "ppo_rollout_v1"


def test_drda_checkpoint_metadata_roundtrip_and_rejects_mismatches(tmp_path: Path) -> None:
    model = DrdaResidualPolicyNet(HydraPolicyNet(hidden=8, blocks=1, bottleneck=4), DrdaResidualConfig(tau_drda=4.0))
    optimizer = torch.optim.AdamW(residual_optimizer_parameters(model), lr=1.0e-3)
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base checkpoint")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint_path = tmp_path / "drda.pt"
    model_config = ModelConfig(hidden=8, blocks=1, bottleneck=4)

    save_drda_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        optimizer_config=_optimizer_config(),
        runtime_config=_runtime_config(),
        loss_weights=LossWeights(),
        manifest_path=manifest,
        global_step=5,
        samples_seen=10,
        base_checkpoint_path=base_checkpoint,
    )
    loaded_model = DrdaResidualPolicyNet(
        HydraPolicyNet(hidden=8, blocks=1, bottleneck=4), DrdaResidualConfig(tau_drda=4.0)
    )
    loaded_optimizer = torch.optim.AdamW(residual_optimizer_parameters(loaded_model), lr=1.0e-3)
    metadata = load_drda_checkpoint(
        checkpoint_path,
        model=loaded_model,
        optimizer=loaded_optimizer,
        expected_model_config=model_config,
        expected_base_checkpoint_path=base_checkpoint,
    )

    assert metadata["schema_version"] == 1
    assert metadata["mode"] == DRDA_RESIDUAL_MODE
    assert metadata["objective"] == DRDA_RESIDUAL_OBJECTIVE
    assert metadata["tau_drda"] == pytest.approx(4.0)
    assert metadata["min_tau_drda"] == pytest.approx(MIN_TAU_DRDA)
    assert metadata["base_checkpoint_path"] == str(base_checkpoint)
    assert metadata["base_model_config"] == cast(object, model_config.__dict__)
    assert metadata["base_weight_source"] == "raw"
    assert metadata["encoder_shape"] == [OBS_CHANNELS, TILE_WIDTH]
    assert metadata["action_space"] == ACTION_SPACE
    assert metadata["rebase_enabled"] is False
    assert metadata["rebase_capability"] == DRDA_REBASE_CAPABILITY
    assert metadata["total_rebases"] == 0
    assert metadata["optimizer_scope"] == DRDA_OPTIMIZER_SCOPE
    assert metadata["policy_preservation"] == DRDA_POLICY_PRESERVATION
    assert metadata["export_supported"] is False
    _assert_states_equal(loaded_model.state_dict(), model.state_dict())

    _assert_metadata_rejects(tmp_path, metadata, "base_checkpoint_sha256", "bad", "base_checkpoint_sha256")
    _assert_metadata_rejects(tmp_path, metadata, "base_model_config", {"hidden": 9}, "base_model_config")
    _assert_metadata_rejects(tmp_path, metadata, "action_space", 45, "action_space")
    _assert_metadata_rejects(tmp_path, metadata, "encoder_shape", [85, 34], "encoder_shape")
    _assert_metadata_rejects(tmp_path, metadata, "tau_drda", 8.0, "tau_drda")
    bad = dict(metadata)
    bad.pop("base_checkpoint_sha256")
    _assert_metadata_rejects(tmp_path, bad, None, None, "base_checkpoint_sha256")


def test_drda_rebase_fail_closed_and_metadata_makes_no_preservation_claim(tmp_path: Path) -> None:
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base")
    config = DrdaResidualConfig()
    metadata = drda_training_objective_metadata(
        config=config,
        base_checkpoint_path=base_checkpoint,
        base_model_config=ModelConfig(hidden=8, blocks=1, bottleneck=4),
    )
    model = DrdaResidualPolicyNet(HydraPolicyNet(hidden=8, blocks=1, bottleneck=4), config)

    with pytest.raises(RuntimeError, match="does not support"):
        drda_rebase()
    with pytest.raises(RuntimeError, match="does not support"):
        model.rebase()
    assert metadata["rebase_enabled"] is False
    assert metadata["rebase_capability"] == DRDA_REBASE_CAPABILITY
    assert metadata["policy_preservation"] == DRDA_POLICY_PRESERVATION


def test_drda_path_has_no_scope_creep_terms(tmp_path: Path) -> None:
    base_checkpoint = tmp_path / "base.pt"
    base_checkpoint.write_bytes(b"base")
    metadata = drda_training_objective_metadata(
        config=DrdaResidualConfig(),
        base_checkpoint_path=base_checkpoint,
        base_model_config=ModelConfig(hidden=8, blocks=1, bottleneck=4),
    )
    payload = json.dumps(metadata, sort_keys=True)

    forbidden = ("pbrs", "grp", "privileged", "search", "exit", "delta_q", "population")
    assert not any(term in payload.lower() for term in forbidden)
    assert DRDA_RESIDUAL_OBJECTIVE != "ppo"
    assert DRDA_RESIDUAL_OBJECTIVE != "direct_sampled_ach"


def _assert_metadata_rejects(
    tmp_path: Path,
    metadata: dict[str, object],
    key: str | None,
    value: object,
    match: str,
) -> None:
    base_checkpoint = tmp_path / "base.pt"
    changed = dict(metadata)
    if key is not None:
        changed[key] = value
    with pytest.raises(ValueError, match=match):
        validate_drda_checkpoint_metadata(
            changed,
            expected_config=DrdaResidualConfig(),
            expected_base_checkpoint_path=base_checkpoint,
            expected_base_model_config=ModelConfig(hidden=8, blocks=1, bottleneck=4),
        )


def _pad_logits(prefix: torch.Tensor) -> torch.Tensor:
    logits = torch.full((prefix.shape[0], ACTION_SPACE), -17.0, dtype=prefix.dtype, device=prefix.device)
    logits[:, : prefix.shape[1]] = prefix
    return logits


def _batch_from_model(model: HydraPolicyNet) -> PpoBatch:
    obs = torch.randn(2, OBS_CHANNELS, TILE_WIDTH, dtype=torch.float32)
    legal_mask = torch.zeros(2, ACTION_SPACE, dtype=torch.bool)
    legal_mask[0, :2] = True
    legal_mask[1, :5] = True
    actions = torch.tensor([0, 1], dtype=torch.int64)
    with torch.inference_mode():
        outputs = model(obs)
        current_logprob = torch.log_softmax(outputs.policy_logits.masked_fill(~legal_mask, -1.0e9), dim=1)
        old_logprob = current_logprob.gather(1, actions.unsqueeze(1)).squeeze(1).to(dtype=torch.float32)
        value_old = outputs.value.squeeze(1).to(dtype=torch.float32)
        bc_logits = outputs.policy_logits.detach().clone().to(dtype=torch.float32)
    return PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=old_logprob,
        value_old=value_old,
        raw_advantages=torch.tensor([1.0, -1.0], dtype=torch.float32),
        returns=value_old + torch.tensor([0.25, -0.25]),
        bc_logits=bc_logits,
        legal_count=legal_mask.sum(dim=1).to(dtype=torch.int64),
        rank_utility_used="U_A",
    )


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def _assert_states_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert set(left) == set(right)
    for key, value in left.items():
        torch.testing.assert_close(value, right[key], rtol=0.0, atol=0.0)


def _any_parameter_changed(after: dict[str, torch.Tensor], before: dict[str, torch.Tensor]) -> bool:
    return any(tensor.is_floating_point() and not torch.equal(tensor, before[name]) for name, tensor in after.items())


def _optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(name="AdamW", lr=1.0e-3, min_lr=1.0e-6)


def _runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        variant="eager_bf16", loss_mode="full_base", precision_mode="bf16_autocast", compile_fullgraph_check=False
    )
