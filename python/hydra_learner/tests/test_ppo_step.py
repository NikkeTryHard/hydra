from __future__ import annotations

import json
import math
from typing import cast

import pytest
import torch

from hydra_learner.model import ACTION_SPACE, HydraPolicyNet
from hydra_learner.ppo_step import PpoBatch, PpoTrainStepConfig, ppo_train_step
from hydra_learner.rl import EntropyController, masked_log_prob


def _replace_batch(batch: PpoBatch, **updates: torch.Tensor | str | None) -> PpoBatch:
    values: dict[str, torch.Tensor | str | None] = {
        "obs": batch.obs,
        "actions": batch.actions,
        "legal_mask": batch.legal_mask,
        "old_logprob": batch.old_logprob,
        "value_old": batch.value_old,
        "raw_advantages": batch.raw_advantages,
        "returns": batch.returns,
        "bc_logits": batch.bc_logits,
        "legal_count": batch.legal_count,
        "player_id": batch.player_id,
        "seat_id": batch.seat_id,
        "game_id": batch.game_id,
        "turn": batch.turn,
        "rank_utility_used": batch.rank_utility_used,
    }
    values.update(updates)
    return PpoBatch(
        obs=cast("torch.Tensor", values["obs"]),
        actions=cast("torch.Tensor", values["actions"]),
        legal_mask=cast("torch.Tensor", values["legal_mask"]),
        old_logprob=cast("torch.Tensor", values["old_logprob"]),
        value_old=cast("torch.Tensor", values["value_old"]),
        raw_advantages=cast("torch.Tensor", values["raw_advantages"]),
        returns=cast("torch.Tensor", values["returns"]),
        bc_logits=cast("torch.Tensor", values["bc_logits"]),
        legal_count=cast("torch.Tensor", values["legal_count"]),
        player_id=cast("torch.Tensor | None", values["player_id"]),
        seat_id=cast("torch.Tensor | None", values["seat_id"]),
        game_id=cast("torch.Tensor | None", values["game_id"]),
        turn=cast("torch.Tensor | None", values["turn"]),
        rank_utility_used=cast("str | None", values["rank_utility_used"]),
    )


def test_ppo_batch_accepts_valid_contract() -> None:
    _valid_batch().validate()


def test_ppo_batch_validation_failures() -> None:
    batch = _valid_batch()
    bad_obs = _replace_batch(batch, obs=torch.zeros(2, 191, 34, dtype=torch.float32))
    with pytest.raises(ValueError, match="obs"):
        bad_obs.validate()
    bad_mask = _replace_batch(batch, legal_mask=batch.legal_mask.to(dtype=torch.float32))
    with pytest.raises(TypeError, match="legal_mask"):
        bad_mask.validate()
    bad_count = _replace_batch(batch, legal_count=torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="legal_count"):
        bad_count.validate()
    bad_action = _replace_batch(batch, actions=torch.tensor([45, 1], dtype=torch.int64))
    with pytest.raises(ValueError, match="legal"):
        bad_action.validate()
    bad_logits = batch.bc_logits.clone()
    bad_logits[0, 0] = torch.nan
    with pytest.raises(ValueError, match="bc_logits"):
        _replace_batch(batch, bc_logits=bad_logits).validate()


def test_ppo_train_step_real_model_finite_update_and_metrics() -> None:
    torch.manual_seed(7)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    batch = _batch_from_model(model, raw_advantages=torch.tensor([1.0, -1.0], dtype=torch.float32))
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    controller = EntropyController(alpha=1.0e-3, beta=1.0e-2, alpha_max=0.05)

    result = ppo_train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        entropy_controller=controller,
        config=PpoTrainStepConfig(clip_epsilon=0.2, value_coef=0.5, bc_kl_reverse_coef=0.01, grad_clip_norm=0.5),
    )

    assert _any_parameter_changed(model.state_dict(), before)
    for key in (
        "loss_total",
        "loss_policy",
        "loss_value",
        "entropy",
        "entropy_alpha_before",
        "entropy_alpha_after",
        "bc_kl_reverse",
        "approx_kl_old",
        "clip_fraction",
        "ratio_mean",
        "explained_variance",
        "advantage_raw_mean",
        "advantage_raw_std",
        "advantage_normalized_mean",
        "advantage_normalized_std",
        "grad_norm",
        "forward_backward_ms",
        "grad_clip_ms",
        "optimizer_ms",
    ):
        assert key in result.metrics
        assert isinstance(result.metrics[key], float)
    assert result.metrics["illegal_action_count"] == 0
    assert set(result.metrics["legal_count_bucket_entropy"]) == {2, 5}
    assert set(result.metrics["legal_count_bucket_clip_fraction"]) == {2, 5}
    assert 0.0 <= result.entropy_controller.alpha <= controller.alpha_max
    json.dumps(result.metrics, allow_nan=False)


def test_ppo_train_step_same_current_bc_logits_near_zero_kl_noop_policy_loss() -> None:
    torch.manual_seed(11)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=0.0)
    batch = _batch_from_model(model, raw_advantages=torch.zeros(2, dtype=torch.float32), returns_from_model=True)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    result = ppo_train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=PpoTrainStepConfig(value_coef=0.0, bc_kl_reverse_coef=0.0, grad_clip_norm=None),
    )

    assert result.metrics["bc_kl_reverse"] == pytest.approx(0.0, abs=1.0e-7)
    assert result.metrics["loss_policy"] == pytest.approx(0.0, abs=1.0e-7)
    assert not _any_parameter_changed(model.state_dict(), before)
    assert result.entropy_controller.alpha == 0.0
    json.dumps(result.metrics, allow_nan=False)


def test_ppo_train_step_without_grad_clip_has_json_safe_grad_norm() -> None:
    torch.manual_seed(13)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    batch = _batch_from_model(model, raw_advantages=torch.tensor([1.0, -1.0], dtype=torch.float32))

    result = ppo_train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
        config=PpoTrainStepConfig(value_coef=0.5, bc_kl_reverse_coef=0.0, grad_clip_norm=None),
    )

    json.dumps(result.metrics, allow_nan=False)
    assert math.isfinite(result.metrics["grad_norm"])


def test_ppo_train_step_microbatch_matches_full_batch_update() -> None:
    torch.manual_seed(19)
    base_model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    batch = _batch_from_model_rows(base_model, rows=6)
    full_model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    micro_model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    full_model.load_state_dict(base_model.state_dict(), strict=True)
    micro_model.load_state_dict(base_model.state_dict(), strict=True)
    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=1.0e-4)
    micro_optimizer = torch.optim.SGD(micro_model.parameters(), lr=1.0e-4)
    controller = EntropyController(alpha=1.0e-3, beta=1.0e-2, alpha_max=0.05)
    step_config = PpoTrainStepConfig(
        clip_epsilon=0.2,
        value_coef=0.5,
        bc_kl_reverse_coef=0.01,
        grad_clip_norm=None,
    )

    full = ppo_train_step(
        model=full_model,
        optimizer=full_optimizer,
        batch=batch,
        entropy_controller=controller,
        config=step_config,
    )
    micro = ppo_train_step(
        model=micro_model,
        optimizer=micro_optimizer,
        batch=batch,
        entropy_controller=controller,
        config=PpoTrainStepConfig(
            clip_epsilon=step_config.clip_epsilon,
            value_coef=step_config.value_coef,
            bc_kl_reverse_coef=step_config.bc_kl_reverse_coef,
            grad_clip_norm=step_config.grad_clip_norm,
            microbatch_size=2,
        ),
    )

    for key in (
        "loss_total",
        "loss_policy",
        "loss_value",
        "entropy",
        "bc_kl_reverse",
        "approx_kl_old",
        "clip_fraction",
        "ratio_mean",
        "explained_variance",
        "advantage_raw_mean",
        "advantage_raw_std",
        "advantage_normalized_mean",
        "advantage_normalized_std",
        "grad_norm",
        "entropy_alpha_after",
    ):
        assert micro.metrics[key] == pytest.approx(full.metrics[key], abs=1.0e-6)
    assert micro.metrics["microbatch_count"] == 3
    assert micro.entropy_controller == full.entropy_controller
    for name, tensor in full_model.state_dict().items():
        if tensor.is_floating_point():
            assert torch.allclose(micro_model.state_dict()[name], tensor, atol=1.0e-7, rtol=1.0e-6), name
        else:
            assert torch.equal(micro_model.state_dict()[name], tensor), name


def test_ppo_train_step_extreme_old_logprob_hard_errors_before_metrics() -> None:
    torch.manual_seed(17)
    model = HydraPolicyNet(hidden=8, blocks=1, bottleneck=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    batch = _batch_from_model(model, raw_advantages=torch.tensor([1.0, -1.0], dtype=torch.float32))
    batch = _replace_batch(batch, old_logprob=torch.full((2,), -1.0e38, dtype=torch.float32))

    with pytest.raises(ValueError, match="ratio"):
        ppo_train_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            entropy_controller=EntropyController(alpha=0.0, beta=0.0, alpha_max=0.0),
            config=PpoTrainStepConfig(value_coef=0.5, bc_kl_reverse_coef=0.0, grad_clip_norm=None),
        )


def _valid_batch() -> PpoBatch:
    obs = torch.zeros(2, 192, 34, dtype=torch.float32)
    legal_mask = torch.zeros(2, ACTION_SPACE, dtype=torch.bool)
    legal_mask[0, :2] = True
    legal_mask[1, :5] = True
    actions = torch.tensor([1, 3], dtype=torch.int64)
    return PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=torch.zeros(2, dtype=torch.float32),
        value_old=torch.zeros(2, dtype=torch.float32),
        raw_advantages=torch.tensor([1.0, -1.0], dtype=torch.float32),
        returns=torch.zeros(2, dtype=torch.float32),
        bc_logits=torch.zeros(2, ACTION_SPACE, dtype=torch.float32),
        legal_count=legal_mask.sum(dim=1).to(dtype=torch.int64),
        player_id=torch.tensor([0, 1], dtype=torch.int64),
        seat_id=torch.tensor([0, 1], dtype=torch.int64),
        game_id=torch.tensor([123, 123], dtype=torch.int64),
        turn=torch.tensor([0, 1], dtype=torch.int64),
        rank_utility_used="U_A",
    )


def _batch_from_model(
    model: HydraPolicyNet, *, raw_advantages: torch.Tensor, returns_from_model: bool = False
) -> PpoBatch:
    obs = torch.randn(2, 192, 34, dtype=torch.float32)
    legal_mask = torch.zeros(2, ACTION_SPACE, dtype=torch.bool)
    legal_mask[0, :2] = True
    legal_mask[1, :5] = True
    actions = torch.tensor([0, 1], dtype=torch.int64)
    with torch.inference_mode():
        out = model(obs)
        old_logprob = masked_log_prob(out.policy_logits, legal_mask, actions).to(dtype=torch.float32)
        value_old = out.value.squeeze(1).to(dtype=torch.float32)
        returns = value_old.clone() if returns_from_model else value_old + torch.tensor([0.25, -0.25])
        bc_logits = out.policy_logits.detach().clone().to(dtype=torch.float32)
    return PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=old_logprob,
        value_old=value_old,
        raw_advantages=raw_advantages,
        returns=returns,
        bc_logits=bc_logits,
        legal_count=legal_mask.sum(dim=1).to(dtype=torch.int64),
        player_id=torch.tensor([0, 1], dtype=torch.int64),
        seat_id=torch.tensor([0, 1], dtype=torch.int64),
        game_id=torch.tensor([1, 1], dtype=torch.int64),
        turn=torch.tensor([0, 1], dtype=torch.int64),
        rank_utility_used="U_A",
    )


def _batch_from_model_rows(model: HydraPolicyNet, *, rows: int) -> PpoBatch:
    obs = torch.randn(rows, 192, 34, dtype=torch.float32)
    legal_mask = torch.zeros(rows, ACTION_SPACE, dtype=torch.bool)
    actions = torch.empty(rows, dtype=torch.int64)
    for row in range(rows):
        count = 2 + row % 4
        legal_mask[row, :count] = True
        actions[row] = row % count
    with torch.inference_mode():
        out = model(obs)
        old_logprob = masked_log_prob(out.policy_logits, legal_mask, actions).to(dtype=torch.float32)
        value_old = out.value.squeeze(1).to(dtype=torch.float32)
        returns = value_old + torch.linspace(-0.3, 0.3, rows, dtype=torch.float32)
        bc_logits = out.policy_logits.detach().clone().to(dtype=torch.float32)
    return PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=old_logprob,
        value_old=value_old,
        raw_advantages=torch.linspace(-1.0, 1.0, rows, dtype=torch.float32),
        returns=returns,
        bc_logits=bc_logits,
        legal_count=legal_mask.sum(dim=1).to(dtype=torch.int64),
        player_id=torch.arange(rows, dtype=torch.int64) % 4,
        seat_id=torch.arange(rows, dtype=torch.int64) % 4,
        game_id=torch.zeros(rows, dtype=torch.int64),
        turn=torch.arange(rows, dtype=torch.int64),
        rank_utility_used="U_A",
    )


def _any_parameter_changed(after: dict[str, torch.Tensor], before: dict[str, torch.Tensor]) -> bool:
    return any(tensor.is_floating_point() and not torch.equal(tensor, before[name]) for name, tensor in after.items())
