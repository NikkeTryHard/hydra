from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path
from typing import cast

import pytest
import torch

import hydra_learner.ach_step as ach_step_module
import hydra_learner.rl as rl_module
from hydra_learner.ach_step import AchTrainStepConfig, ach_train_step
from hydra_learner.model import ACTION_SPACE, HydraPolicyNet
from hydra_learner.ppo_rollout import (
    PpoRolloutMetadata,
    save_ppo_rollout_artifact,
    train_ach_step_from_rollout_artifact,
)
from hydra_learner.ppo_step import PpoBatch
from hydra_learner.rl import AchLossConfig, EntropyController, ach_loss, normalize_advantages


def test_ach_formula_matches_rust_reference_fixture_centered_clamped() -> None:
    logits = torch.tensor([[3.0, 1.0, -2.0, 99.0], [-20.0, 4.0, 2.0, -99.0]], dtype=torch.float32)
    logits = _pad_logits(logits)
    mask = torch.zeros(2, ACTION_SPACE, dtype=torch.bool)
    mask[0, :3] = True
    mask[1, :3] = True
    actions = torch.tensor([0, 1], dtype=torch.int64)
    raw_adv = torch.tensor([2.0, -1.0], dtype=torch.float32)
    cfg = AchLossConfig(eta=1.25, eps=0.5, l_th=3.0, value_coef=0.0, entropy_alpha=0.0)
    centered = logits[:, :3] - logits[:, :3].mean(dim=1, keepdim=True)
    clamped = centered.clamp(-cfg.l_th, cfg.l_th)
    expected_probs = torch.softmax(clamped, dim=1)
    old_logprob = torch.log(expected_probs[torch.arange(2), actions])

    out = ach_loss(logits, torch.zeros(2), actions, mask, old_logprob, raw_adv, torch.zeros(2), config=cfg)

    rms_scaled = raw_adv / torch.sqrt((raw_adv * raw_adv).mean() + cfg.advantage_epsilon)
    y_a = clamped[torch.arange(2), actions]
    expected_policy = -(cfg.eta * y_a * rms_scaled / expected_probs[torch.arange(2), actions]).mean()
    torch.testing.assert_close(out.metrics.policy_loss, expected_policy)
    assert out.metrics.ratio_mean == pytest.approx(1.0)
    assert out.metrics.illegal_probability_max == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("advantage", "old_prob", "selected_logit", "expected_sign"),
    [
        (1.0, 0.4, 0.25, -1.0),
        (-1.0, 0.4, 0.25, 1.0),
    ],
)
def test_ach_ungated_selected_preference_gradient_direction(
    advantage: float, old_prob: float, selected_logit: float, expected_sign: float
) -> None:
    logits = torch.zeros(1, ACTION_SPACE, requires_grad=True)
    with torch.no_grad():
        logits[0, 0] = selected_logit
    out = ach_loss(
        logits,
        torch.zeros(1),
        torch.tensor([0]),
        _mask_with_count(2),
        torch.log(torch.tensor([old_prob])),
        torch.tensor([advantage]),
        torch.zeros(1),
        config=AchLossConfig(value_coef=0.0),
    )
    out.total.backward()

    assert logits.grad is not None
    assert math.copysign(1.0, float(logits.grad[0, 0])) == expected_sign
    assert out.metrics.gate_fraction == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("advantage", "old_prob", "selected_logit", "match"),
    [
        (1.0, 0.01, 0.0, "ratio"),
        (-1.0, 1.1, 0.0, "ratio"),
        (1.0, 0.5, 99.0, "positive clamp"),
        (-1.0, 0.5, -99.0, "negative clamp"),
    ],
)
def test_ach_gate_blocks_policy_gradient(advantage: float, old_prob: float, selected_logit: float, match: str) -> None:
    logits = torch.zeros(1, ACTION_SPACE, requires_grad=True)
    with torch.no_grad():
        logits[0, 0] = selected_logit
    out = ach_loss(
        logits,
        torch.zeros(1),
        torch.tensor([0]),
        _mask_with_count(2),
        torch.log(torch.tensor([old_prob])),
        torch.tensor([advantage]),
        torch.zeros(1),
        config=AchLossConfig(value_coef=0.0, entropy_alpha=0.0),
    )
    out.total.backward()

    assert match
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad[0], torch.zeros(ACTION_SPACE), atol=1.0e-7, rtol=0.0)
    assert out.metrics.gate_fraction == pytest.approx(0.0)


def test_ach_mask_safety_illegal_logits_probability_and_gradient() -> None:
    logits = torch.zeros(1, ACTION_SPACE, requires_grad=True)
    with torch.no_grad():
        logits[0, 2:] = 1000.0
    out = ach_loss(
        logits,
        torch.zeros(1),
        torch.tensor([0]),
        _mask_with_count(2),
        torch.log(torch.tensor([0.5])),
        torch.tensor([1.0]),
        torch.zeros(1),
        config=AchLossConfig(value_coef=0.0),
    )
    out.total.backward()

    assert out.metrics.illegal_probability_max == pytest.approx(0.0)
    assert logits.grad is not None
    assert bool((logits.grad[0, 2:] == 0.0).all())
    with pytest.raises(ValueError, match="all-illegal"):
        ach_loss(
            logits.detach(),
            torch.zeros(1),
            torch.tensor([0]),
            torch.zeros(1, ACTION_SPACE, dtype=torch.bool),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )
    with pytest.raises(ValueError, match="selected action is illegal"):
        ach_loss(
            logits.detach(),
            torch.zeros(1),
            torch.tensor([3]),
            _mask_with_count(2),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )


def test_ach_advantage_rms_scaling_preserves_signs_and_zero_finite() -> None:
    advantages = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    out = ach_loss(
        torch.zeros(3, ACTION_SPACE),
        torch.zeros(3),
        torch.tensor([0, 0, 0]),
        torch.ones(3, ACTION_SPACE, dtype=torch.bool),
        torch.full((3,), -math.log(float(ACTION_SPACE))),
        advantages,
        torch.zeros(3),
        config=AchLossConfig(value_coef=0.0),
    )
    assert out.metrics.advantage_positive_count == 3
    assert float(out.metrics.advantage_scaled_mean) > 0.0
    assert bool((normalize_advantages(advantages) < 0.0).any())

    zero = ach_loss(
        torch.zeros(2, ACTION_SPACE),
        torch.zeros(2),
        torch.tensor([0, 1]),
        torch.ones(2, ACTION_SPACE, dtype=torch.bool),
        torch.full((2,), -math.log(float(ACTION_SPACE))),
        torch.zeros(2),
        torch.zeros(2),
        config=AchLossConfig(value_coef=0.0),
    )
    assert bool(torch.isfinite(zero.total))
    assert zero.metrics.advantage_zero_count == 2
    assert zero.metrics.advantage_positive_count == 2
    assert zero.metrics.advantage_zero_count == 2
    assert zero.metric_dict()["ach_neg_gate_fraction"] is None


# ACH contract: zero advantage is in the nonnegative/positive group for gate metrics.
def test_ach_empty_sign_groups_emit_none_and_zero_is_positive() -> None:
    positive = ach_loss(
        torch.zeros(3, ACTION_SPACE),
        torch.zeros(3),
        torch.tensor([0, 1, 2]),
        torch.ones(3, ACTION_SPACE, dtype=torch.bool),
        torch.full((3,), -math.log(float(ACTION_SPACE))),
        torch.tensor([1.0, 0.0, 2.0]),
        torch.zeros(3),
        config=AchLossConfig(value_coef=0.0),
    )
    positive_metrics = positive.metric_dict()
    assert positive_metrics["ach_neg_gate_fraction"] is None
    assert positive_metrics["ach_pos_gate_fraction"] == pytest.approx(1.0)
    assert positive_metrics["advantage_positive_count"] == 3
    assert positive_metrics["advantage_zero_count"] == 1
    json.dumps(positive_metrics, allow_nan=False)

    negative = ach_loss(
        torch.zeros(2, ACTION_SPACE),
        torch.zeros(2),
        torch.tensor([0, 1]),
        torch.ones(2, ACTION_SPACE, dtype=torch.bool),
        torch.full((2,), -math.log(float(ACTION_SPACE))),
        torch.tensor([-1.0, -2.0]),
        torch.zeros(2),
        config=AchLossConfig(value_coef=0.0),
    )
    negative_metrics = negative.metric_dict()
    assert negative_metrics["ach_pos_gate_fraction"] is None
    assert negative_metrics["ach_neg_gate_fraction"] == pytest.approx(1.0)
    assert negative_metrics["advantage_negative_count"] == 2
    json.dumps(negative_metrics, allow_nan=False)


def test_ach_bc_kl_uses_centered_clamped_current_policy() -> None:
    logits = torch.zeros(1, ACTION_SPACE)
    logits[0, 0] = 100.0
    mask = _mask_with_count(3)
    bc_logits = torch.zeros_like(logits)
    bc_logits[0, 1] = 3.0
    cfg = AchLossConfig(l_th=2.0, value_coef=0.0, bc_kl_reverse_coef=1.0)

    out = ach_loss(
        logits,
        torch.zeros(1),
        torch.tensor([0]),
        mask,
        torch.log(torch.tensor([0.5])),
        torch.tensor([1.0]),
        torch.zeros(1),
        bc_logits=bc_logits,
        config=cfg,
    )

    centered = logits - (logits * mask.to(dtype=logits.dtype)).sum(dim=1, keepdim=True) / mask.sum(dim=1).unsqueeze(1)
    clamped = centered.clamp(-cfg.l_th, cfg.l_th)
    current_log = torch.log_softmax(clamped.masked_fill(~mask, -1.0e9), dim=1)
    reference_log = torch.log_softmax(bc_logits.masked_fill(~mask, -1.0e9), dim=1)
    expected = (current_log.exp().masked_fill(~mask, 0.0) * (current_log - reference_log).masked_fill(~mask, 0.0)).sum()
    raw_current_log = torch.log_softmax(logits.masked_fill(~mask, -1.0e9), dim=1)
    raw_kl = (
        raw_current_log.exp().masked_fill(~mask, 0.0) * (raw_current_log - reference_log).masked_fill(~mask, 0.0)
    ).sum()

    torch.testing.assert_close(out.metrics.bc_kl_reverse, expected)
    assert not torch.isclose(out.metrics.bc_kl_reverse, raw_kl)


def test_ach_old_logprob_clamp_and_json_safety() -> None:
    out = ach_loss(
        torch.zeros(2, ACTION_SPACE),
        torch.zeros(2),
        torch.tensor([0, 1]),
        torch.ones(2, ACTION_SPACE, dtype=torch.bool),
        torch.tensor([-100.0, -math.log(float(ACTION_SPACE))]),
        torch.tensor([1.0, -1.0]),
        torch.zeros(2),
        config=AchLossConfig(value_coef=0.0),
    )
    metrics = out.metric_dict()
    assert metrics["pi_old_clamp_fraction"] == pytest.approx(0.5)
    assert metrics["pi_old_raw_min"] == pytest.approx(0.0)
    json.dumps(metrics, allow_nan=False)
    with pytest.raises(ValueError, match="old_logprob"):
        ach_loss(
            torch.zeros(1, ACTION_SPACE),
            torch.zeros(1),
            torch.tensor([0]),
            _mask_with_count(ACTION_SPACE),
            torch.tensor([torch.nan]),
            torch.zeros(1),
            torch.zeros(1),
        )


def test_ach_train_step_real_model_metrics_and_no_residual_config() -> None:
    torch.manual_seed(101)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    batch = _batch_from_model(model)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    result = ach_train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        entropy_controller=EntropyController(alpha=1.0e-3, beta=1.0e-2, alpha_max=0.05),
        config=AchTrainStepConfig(bc_kl_reverse_coef=0.01),
    )

    json.dumps(result.metrics, allow_nan=False)
    for key in (
        "loss_total",
        "loss_policy",
        "loss_value",
        "entropy",
        "entropy_fraction_mean",
        "ratio_min",
        "ratio_max",
        "ach_gate_fraction",
        "pi_old_clamp_fraction",
        "pi_old_raw_min",
        "legal_count_bucket_entropy",
        "legal_count_bucket_gate_fraction",
        "legal_count_bucket_pos_gate_fraction",
        "legal_count_bucket_neg_gate_fraction",
        "legal_count_bucket_ratio_clipped_fraction",
        "legal_count_bucket_bc_kl",
    ):
        assert key in result.metrics
    assert _any_parameter_changed(model.state_dict(), before)
    field_names = {field.name for field in fields(AchTrainStepConfig)} | {field.name for field in fields(AchLossConfig)}
    assert not {"tau", "rebase", "residual"} & field_names
    assert not hasattr(model, "policy_logits_residual")


def test_ach_step_sign_bucket_none_and_entropy_fraction_use_clamped_policy() -> None:
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    batch = _manual_batch(raw_advantages=torch.tensor([1.0, 0.0], dtype=torch.float32), legal_counts=(2, 2))
    with torch.no_grad():
        model.base_heads.weight.zero_()
        model.base_heads.bias.zero_()
        model.base_heads.bias[0] = 50.0

    result = ach_train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=AchTrainStepConfig(l_th=1.0, value_coef=0.0, bc_kl_reverse_coef=0.0, grad_clip_norm=None),
    )

    assert result.metrics["ach_neg_gate_fraction"] is None
    assert result.metrics["legal_count_bucket_neg_gate_fraction"][2] is None
    assert result.metrics["legal_count_bucket_pos_gate_fraction"][2] == pytest.approx(
        result.metrics["ach_pos_gate_fraction"]
    )
    raw_entropy_fraction = 0.0
    expected_entropy_fraction = float(
        -(torch.softmax(torch.tensor([1.0, -1.0]), dim=0) * torch.log_softmax(torch.tensor([1.0, -1.0]), dim=0)).sum()
        / math.log(2.0)
    )
    assert result.metrics["entropy_fraction_mean"] != pytest.approx(raw_entropy_fraction)
    assert result.metrics["entropy_fraction_mean"] == pytest.approx(expected_entropy_fraction)

    negative_batch = _manual_batch(raw_advantages=torch.tensor([-1.0, -2.0], dtype=torch.float32), legal_counts=(2, 2))
    negative_result = ach_train_step(
        model=model,
        optimizer=optimizer,
        batch=negative_batch,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=AchTrainStepConfig(l_th=1.0, value_coef=0.0, bc_kl_reverse_coef=0.0, grad_clip_norm=None),
    )
    assert negative_result.metrics["ach_pos_gate_fraction"] is None
    assert negative_result.metrics["legal_count_bucket_pos_gate_fraction"][2] is None


def test_ach_rollout_artifact_trains_without_schema_change(tmp_path: Path) -> None:
    torch.manual_seed(103)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    path = tmp_path / "rollout.pt"
    save_ppo_rollout_artifact(path, _batch_from_model(model), PpoRolloutMetadata(rank_utility_used="U_A"))

    result = train_ach_step_from_rollout_artifact(
        artifact_path=path,
        model=model,
        optimizer=optimizer,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=AchTrainStepConfig(grad_clip_norm=None),
    )

    json.dumps(result.metrics, allow_nan=False)
    assert result.metrics["rollout_schema_version"] == 1
    assert result.metrics["rollout_contract_version"] == "ppo_rollout_v1"
    assert result.artifact_metadata["contract_version"] == "ppo_rollout_v1"
    assert cast("dict[str, object]", result.artifact_metadata["reward_shaping"])["enabled"] is False


def test_ach_default_artifact_metadata_does_not_change_current_direct_sampled_contract(tmp_path: Path) -> None:
    torch.manual_seed(107)
    base_model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    artifact_model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    direct_model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    artifact_model.load_state_dict(base_model.state_dict())
    direct_model.load_state_dict(base_model.state_dict())
    artifact_optimizer = torch.optim.AdamW(artifact_model.parameters(), lr=1.0e-3)
    direct_optimizer = torch.optim.AdamW(direct_model.parameters(), lr=1.0e-3)
    batch = _batch_from_model(base_model)
    path = tmp_path / "rollout.pt"
    save_ppo_rollout_artifact(path, batch, PpoRolloutMetadata(rank_utility_used="U_A"))

    artifact_result = train_ach_step_from_rollout_artifact(
        artifact_path=path,
        model=artifact_model,
        optimizer=artifact_optimizer,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=AchTrainStepConfig(grad_clip_norm=None),
    )
    direct_result = ach_train_step(
        model=direct_model,
        optimizer=direct_optimizer,
        batch=batch,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=AchTrainStepConfig(grad_clip_norm=None),
    )

    comparable = ("loss_total", "loss_policy", "loss_value", "entropy", "bc_kl_reverse", "ach_gate_fraction")
    for key in comparable:
        assert artifact_result.metrics[key] == pytest.approx(direct_result.metrics[key])


def test_ach_has_no_neurd_selector_or_objective() -> None:
    names = set(dir(ach_step_module)) | set(dir(rl_module))
    assert not any("neurd" in name.lower() for name in names)


def _mask_with_count(count: int) -> torch.Tensor:
    mask = torch.zeros(1, ACTION_SPACE, dtype=torch.bool)
    mask[0, :count] = True
    return mask


def _pad_logits(prefix: torch.Tensor) -> torch.Tensor:
    logits = torch.full((prefix.shape[0], ACTION_SPACE), -17.0, dtype=prefix.dtype)
    logits[:, : prefix.shape[1]] = prefix
    return logits


def _batch_from_model(model: HydraPolicyNet) -> PpoBatch:
    obs = torch.randn(2, 192, 34, dtype=torch.float32)
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


def _manual_batch(*, raw_advantages: torch.Tensor, legal_counts: tuple[int, ...]) -> PpoBatch:
    batch = raw_advantages.shape[0]
    obs = torch.zeros(batch, 192, 34, dtype=torch.float32)
    legal_mask = torch.zeros(batch, ACTION_SPACE, dtype=torch.bool)
    actions = torch.zeros(batch, dtype=torch.int64)
    for row, count in enumerate(legal_counts):
        legal_mask[row, :count] = True
        actions[row] = min(row, count - 1)
    return PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=torch.full((batch,), -math.log(2.0), dtype=torch.float32),
        value_old=torch.zeros(batch, dtype=torch.float32),
        raw_advantages=raw_advantages,
        returns=torch.zeros(batch, dtype=torch.float32),
        bc_logits=torch.zeros(batch, ACTION_SPACE, dtype=torch.float32),
        legal_count=legal_mask.sum(dim=1).to(dtype=torch.int64),
        rank_utility_used="U_A",
    )


def _any_parameter_changed(after: dict[str, torch.Tensor], before: dict[str, torch.Tensor]) -> bool:
    return any(tensor.is_floating_point() and not torch.equal(tensor, before[name]) for name, tensor in after.items())
