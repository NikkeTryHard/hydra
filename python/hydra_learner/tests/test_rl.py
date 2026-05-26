from __future__ import annotations

import math
from typing import cast

import pytest
import torch

from hydra_learner.model import ACTION_SPACE
from hydra_learner.reward_shaping import RewardShapingConfig
from hydra_learner.rl import (
    EntropyController,
    PlayerDecisionStep,
    PpoLossConfig,
    compute_gae_returns,
    compute_player_local_gae,
    default_entropy_target_fraction,
    entropy_fraction,
    entropy_target,
    legal_count_bucket_means,
    masked_entropy,
    masked_kl,
    masked_log_prob,
    masked_probs,
    normalize_advantages,
    ppo_loss,
)


def _mask_with_count(count: int) -> torch.Tensor:
    mask = torch.zeros(1, ACTION_SPACE, dtype=torch.bool)
    mask[0, :count] = True
    return mask


@pytest.mark.parametrize("count", [1, 2, 5, 20, ACTION_SPACE])
def test_masked_probs_sum_to_one_and_zero_illegal(count: int) -> None:
    logits = torch.linspace(-1.0, 1.0, ACTION_SPACE).reshape(1, ACTION_SPACE)
    logits[0, count:] = 1000.0
    mask = _mask_with_count(count)

    probs = masked_probs(logits, mask)

    torch.testing.assert_close(probs[mask].sum(), torch.tensor(1.0))
    assert bool((probs[~mask] == 0.0).all())
    assert bool(torch.isfinite(probs).all())


def test_masked_gradient_is_zero_for_illegal_large_logits() -> None:
    logits = torch.zeros(1, ACTION_SPACE, requires_grad=True)
    with torch.no_grad():
        logits[0, 5:] = 1000.0
    mask = _mask_with_count(5)

    loss = -masked_log_prob(logits, mask, torch.tensor([2])).mean()
    loss.backward()

    assert logits.grad is not None
    assert bool((logits.grad[~mask] == 0.0).all())
    assert bool(logits.grad[mask].abs().sum() > 0.0)


def test_masked_entropy_excludes_illegal_actions() -> None:
    logits = torch.zeros(1, ACTION_SPACE)
    logits[0, 2:] = 1000.0
    mask = _mask_with_count(2)

    torch.testing.assert_close(masked_entropy(logits, mask), torch.tensor([math.log(2.0)]))
    torch.testing.assert_close(entropy_fraction(logits, mask), torch.tensor([1.0]))


def test_masked_kl_excludes_illegal_high_logit() -> None:
    current = torch.zeros(1, ACTION_SPACE)
    reference = torch.zeros(1, ACTION_SPACE)
    current[0, 2:] = 500.0
    reference[0, 2:] = -500.0
    mask = _mask_with_count(2)

    kl = masked_kl(current, reference, mask)

    torch.testing.assert_close(kl, torch.zeros(1))


def test_masked_kl_identical_near_zero_and_drop_mode_rises() -> None:
    reference = torch.zeros(1, ACTION_SPACE)
    reference[0, 0] = 4.0
    current = reference.clone()
    mask = _mask_with_count(5)

    torch.testing.assert_close(masked_kl(current, reference, mask), torch.zeros(1), atol=1.0e-7, rtol=0.0)
    current[0, 0] = -4.0
    assert float(masked_kl(current, reference, mask)) > 1.0


def test_all_illegal_and_selected_illegal_hard_error() -> None:
    logits = torch.zeros(1, ACTION_SPACE)
    with pytest.raises(ValueError, match="all-illegal"):
        masked_probs(logits, torch.zeros(1, ACTION_SPACE, dtype=torch.bool))
    with pytest.raises(ValueError, match="selected action is illegal"):
        masked_log_prob(logits, _mask_with_count(2), torch.tensor([3]))


def test_non_bool_legal_mask_hard_errors() -> None:
    logits = torch.zeros(1, ACTION_SPACE)
    with pytest.raises(TypeError, match="legal_mask must be bool"):
        masked_probs(logits, torch.ones(1, ACTION_SPACE, dtype=torch.float32))
    with pytest.raises(TypeError, match="legal_mask must be bool"):
        masked_probs(logits, torch.ones(1, ACTION_SPACE, dtype=torch.int64))


def test_gae_single_step_terminal_value_target_is_return() -> None:
    advantages, returns = compute_gae_returns(
        torch.tensor([1.0]), torch.tensor([0.25]), torch.tensor([0.0]), torch.tensor([False])
    )

    torch.testing.assert_close(advantages, torch.tensor([0.75]))
    torch.testing.assert_close(returns, torch.tensor([1.0]))


def test_gae_multi_step_and_truncation_bootstrap() -> None:
    rewards = torch.tensor([0.0, 0.0])
    values = torch.tensor([0.5, 0.25])
    next_values = torch.tensor([0.25, 0.75])
    has_next = torch.tensor([True, True])

    advantages, returns = compute_gae_returns(rewards, values, next_values, has_next, gamma=1.0, gae_lambda=0.5)

    deltas = torch.tensor([-0.25, 0.5])
    expected_advantages = torch.tensor([deltas[0] + 0.5 * deltas[1], deltas[1]])
    torch.testing.assert_close(advantages, expected_advantages)
    torch.testing.assert_close(returns, expected_advantages + values)


def test_player_local_stream_ignores_opponent_steps_and_assigns_terminal_reward_once() -> None:
    steps = [
        PlayerDecisionStep(player_id=0, value_old=0.0),
        PlayerDecisionStep(player_id=1, value_old=10.0),
        PlayerDecisionStep(player_id=0, value_old=0.0),
    ]

    out = compute_player_local_gae(steps, final_placements=[0, 3, 1, 2], gamma=1.0, gae_lambda=1.0)

    torch.testing.assert_close(out.rewards, torch.tensor([0.0, -1.0, 1.0]))
    torch.testing.assert_close(out.raw_advantages[[0, 2]], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(out.returns[[0, 2]], torch.tensor([1.0, 1.0]))
    assert out.terminal_player_stream.tolist() == [False, True, True]


def test_player_local_truncation_uses_bootstrap_not_terminal_reward() -> None:
    out = compute_player_local_gae(
        [PlayerDecisionStep(player_id=0, value_old=0.25, truncation_bootstrap_value=0.75)],
        final_placements=None,
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(out.rewards, torch.tensor([0.0]))
    torch.testing.assert_close(out.returns, torch.tensor([0.75]))
    assert out.truncation.tolist() == [True]


def test_player_local_gae_default_equivalence_and_explicit_beta_zero_shaping() -> None:
    steps = [
        PlayerDecisionStep(
            player_id=0, value_old=0.1, phi_t=cast("float", _boom("phi_t")), phi_next=cast("float", _boom("phi_next"))
        ),
        PlayerDecisionStep(
            player_id=0, value_old=0.2, phi_t=cast("float", _boom("phi_t")), phi_next=cast("float", _boom("phi_next"))
        ),
    ]

    base = compute_player_local_gae(steps, final_placements=[0, 1, 2, 3], gamma=0.9, gae_lambda=0.8)
    shaped_zero = compute_player_local_gae(
        steps,
        final_placements=[0, 1, 2, 3],
        gamma=0.9,
        gae_lambda=0.8,
        reward_shaping=RewardShapingConfig(enabled=False, pbrs_beta=0.0),
    )

    torch.testing.assert_close(shaped_zero.rewards, base.rewards)
    torch.testing.assert_close(shaped_zero.raw_advantages, base.raw_advantages)
    torch.testing.assert_close(shaped_zero.returns, base.returns)
    shaped_zero_enabled = compute_player_local_gae(
        steps,
        final_placements=[0, 1, 2, 3],
        gamma=0.9,
        gae_lambda=0.8,
        reward_shaping=RewardShapingConfig(enabled=True, pbrs_beta=0.0),
    )
    torch.testing.assert_close(shaped_zero_enabled.rewards, base.rewards)
    torch.testing.assert_close(shaped_zero_enabled.raw_advantages, base.raw_advantages)
    torch.testing.assert_close(shaped_zero_enabled.returns, base.returns)


def test_player_local_gae_nonzero_beta_requires_validation_artifact() -> None:
    steps = [PlayerDecisionStep(player_id=0, value_old=0.0, phi_t=0.5, phi_next=0.0, terminal_next_phi=True)]
    with pytest.raises(ValueError, match="validation_artifact_path"):
        compute_player_local_gae(
            steps,
            final_placements=[0, 1, 2, 3],
            reward_shaping=RewardShapingConfig(enabled=True, pbrs_beta=0.1),
        )


class _ExplodingPhi:
    def __float__(self) -> float:
        raise AssertionError("phi should not be converted on default/no-shaping path")


def _boom(_name: str) -> object:
    return _ExplodingPhi()


def test_player_local_gae_rejects_invalid_player_and_placements() -> None:
    with pytest.raises(ValueError, match="player_id"):
        compute_player_local_gae([PlayerDecisionStep(player_id=4, value_old=0.0)], final_placements=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="four placements"):
        compute_player_local_gae([PlayerDecisionStep(player_id=0, value_old=0.0)], final_placements=[0, 1, 2])
    with pytest.raises(ValueError, match="final placement"):
        compute_player_local_gae([PlayerDecisionStep(player_id=0, value_old=0.0)], final_placements=[0, 1, 2, 4])


def test_normalize_advantages_mean_std_for_ppo() -> None:
    normalized = normalize_advantages(torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(normalized.mean(), torch.tensor(0.0), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(torch.sqrt(((normalized - normalized.mean()) ** 2).mean()), torch.tensor(1.0))


def test_ppo_positive_advantage_increases_selected_legal_logit() -> None:
    logits = torch.zeros(2, ACTION_SPACE, requires_grad=True)
    mask = torch.ones(2, ACTION_SPACE, dtype=torch.bool)
    actions = torch.tensor([0, 1])
    old_logprob = torch.full((2,), -math.log(float(ACTION_SPACE)))
    returns = torch.zeros(2)

    loss = ppo_loss(
        logits,
        torch.zeros(2),
        actions,
        mask,
        old_logprob,
        torch.tensor([1.0, 1.0]),
        returns,
        config=PpoLossConfig(value_coef=0.0),
    ).total
    loss.backward()

    assert logits.grad is not None
    assert float(logits.grad[0, 0]) < 0.0
    assert float(logits.grad[1, 1]) < 0.0


def test_ppo_negative_advantage_decreases_selected_legal_logit() -> None:
    logits = torch.zeros(2, ACTION_SPACE, requires_grad=True)
    mask = torch.ones(2, ACTION_SPACE, dtype=torch.bool)
    actions = torch.tensor([0, 1])
    old_logprob = torch.full((2,), -math.log(float(ACTION_SPACE)))

    loss = ppo_loss(
        logits,
        torch.zeros(2),
        actions,
        mask,
        old_logprob,
        torch.tensor([-1.0, -1.0]),
        torch.zeros(2),
        config=PpoLossConfig(value_coef=0.0),
    ).total
    loss.backward()

    assert logits.grad is not None
    assert float(logits.grad[0, 0]) > 0.0
    assert float(logits.grad[1, 1]) > 0.0


def test_ppo_clipped_ratio_has_zero_selected_gradient_when_clipped_positive_advantage() -> None:
    logits = torch.zeros(2, ACTION_SPACE, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = 10.0
    mask = torch.ones(2, ACTION_SPACE, dtype=torch.bool)

    out = ppo_loss(
        logits,
        torch.zeros(2),
        torch.tensor([0, 0]),
        mask,
        torch.full((2,), -math.log(float(ACTION_SPACE))),
        torch.tensor([1.0, 1.0]),
        torch.zeros(2),
        config=PpoLossConfig(clip_epsilon=0.1, value_coef=0.0),
    )
    out.total.backward()

    assert logits.grad is not None
    torch.testing.assert_close(logits.grad[:, 0], torch.zeros(2), atol=1.0e-7, rtol=0.0)
    assert float(out.metrics.clip_fraction) == 1.0


def test_ppo_uses_stored_old_logprob_not_current_policy_recompute() -> None:
    logits = torch.zeros(2, ACTION_SPACE)
    mask = torch.ones(2, ACTION_SPACE, dtype=torch.bool)
    actions = torch.tensor([0, 1])
    out_a = ppo_loss(
        logits,
        torch.zeros(2),
        actions,
        mask,
        torch.full((2,), -math.log(float(ACTION_SPACE))),
        torch.tensor([1.0, -1.0]),
        torch.zeros(2),
        config=PpoLossConfig(value_coef=0.0),
    )
    out_b = ppo_loss(
        logits,
        torch.zeros(2),
        actions,
        mask,
        torch.full((2,), -20.0),
        torch.tensor([1.0, -1.0]),
        torch.zeros(2),
        config=PpoLossConfig(value_coef=0.0),
    )

    assert not torch.equal(out_a.total, out_b.total)


def test_ppo_bc_kl_entropy_value_metrics_and_buckets_are_finite() -> None:
    logits = torch.zeros(2, ACTION_SPACE)
    mask = torch.stack([_mask_with_count(2).squeeze(0), _mask_with_count(5).squeeze(0)])
    out = ppo_loss(
        logits,
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([0, 1]),
        mask,
        torch.tensor([-math.log(2.0), -math.log(5.0)]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([0.5, -0.5]),
        bc_logits=logits.clone(),
        config=PpoLossConfig(value_coef=0.5, entropy_alpha=0.01, bc_kl_reverse_coef=0.1),
    )

    assert bool(torch.isfinite(out.total))
    torch.testing.assert_close(out.metrics.bc_kl_reverse, torch.tensor(0.0))
    bucket_means = legal_count_bucket_means(masked_entropy(logits, mask), mask)
    assert set(bucket_means) == {2, 5}
    assert bucket_means[2] == pytest.approx(math.log(2.0))
    assert bucket_means[5] == pytest.approx(math.log(5.0))


def test_ppo_loss_rejects_nonfinite_boundary_tensors() -> None:
    logits = torch.zeros(1, ACTION_SPACE)
    mask = torch.ones(1, ACTION_SPACE, dtype=torch.bool)
    actions = torch.tensor([0])
    old_logprob = torch.tensor([0.0])
    raw_advantages = torch.tensor([1.0])
    returns = torch.tensor([0.0])
    values = torch.tensor([0.0])

    bad_logits = logits.clone()
    bad_logits[0, 0] = torch.nan
    with pytest.raises(ValueError, match="policy_logits"):
        ppo_loss(bad_logits, values, actions, mask, old_logprob, raw_advantages, returns)
    with pytest.raises(ValueError, match="raw_advantages"):
        ppo_loss(logits, values, actions, mask, old_logprob, torch.tensor([torch.inf]), returns)
    with pytest.raises(ValueError, match="old_logprob"):
        ppo_loss(logits, values, actions, mask, torch.tensor([torch.nan]), raw_advantages, returns)
    bad_bc = logits.clone()
    bad_bc[0, 1] = torch.nan
    with pytest.raises(ValueError, match="bc_logits"):
        ppo_loss(logits, values, actions, mask, old_logprob, raw_advantages, returns, bc_logits=bad_bc)


def test_entropy_controller_direction_clamps_and_target_log_legal_count() -> None:
    legal_count = torch.tensor([2, 5])
    target = entropy_target(legal_count, 0.5)
    torch.testing.assert_close(target, torch.tensor([0.5 * math.log(2.0), 0.5 * math.log(5.0)]))

    controller = EntropyController(alpha=0.1, beta=0.5, alpha_max=0.2)
    increased = controller.update(torch.zeros(2), legal_count, 0.5)
    assert increased.alpha > controller.alpha
    decreased = controller.update(torch.full((2,), 10.0), legal_count, 0.5)
    assert decreased.alpha == 0.0
    clamped = controller.update(torch.zeros(2), torch.tensor([ACTION_SPACE, ACTION_SPACE]), 1.0)
    assert clamped.alpha == 0.2


def test_default_entropy_target_fraction_is_bucket_aware() -> None:
    legal_count = torch.tensor([1, 4, 5, ACTION_SPACE])
    torch.testing.assert_close(default_entropy_target_fraction(legal_count), torch.tensor([0.40, 0.40, 0.70, 0.70]))


def test_entropy_controller_default_uses_bucket_targets() -> None:
    legal_count = torch.tensor([2, 20])
    observed = torch.zeros(2)
    controller = EntropyController(alpha=0.0, beta=0.1, alpha_max=1.0)

    updated = controller.update_default(observed, legal_count)
    expected_target = entropy_target(legal_count, default_entropy_target_fraction(legal_count))

    assert updated.alpha == pytest.approx(0.1 * float(expected_target.mean()))
