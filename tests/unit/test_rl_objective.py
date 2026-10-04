"""Exact masked PPO, complete-minibatch statistics, and real short tails."""

from __future__ import annotations

import pytest
import torch

from hydra2.contracts.common import ContractError
from hydra2.training.rl_objective import PpoParameters, ppo_micro_terms, prepare_advantages

pytestmark = pytest.mark.contract_package("WP-11")

PARAMETERS = PpoParameters(clip_eps=0.2, eps_std=1e-8, w_value=0.005, w_bc=0.0, alpha=0.01)


def test_advantage_population_variance_and_epsilon_placement() -> None:
    actual = prepare_advantages(torch.tensor([1.0, 3.0]), eps_std=3.0)
    assert torch.equal(actual, torch.tensor([-0.5, 0.5]))
    assert torch.equal(prepare_advantages(torch.tensor([7.0]), eps_std=1e-8), torch.zeros(1))
    assert torch.equal(prepare_advantages(torch.full((5,), 7.0), eps_std=1e-8), torch.zeros(5))


@pytest.mark.parametrize("count", [1, 127, 128, 129, 904, 4096])
def test_uneven_microsteps_match_full_optimizer_loss_and_gradients(count: int) -> None:
    row = torch.arange(count, dtype=torch.float32)
    base_logits = torch.stack((row.remainder(3) * 0.1, row.remainder(5) * 0.2, row * 0.0), dim=1)
    base_value = row.remainder(7).unsqueeze(1).expand(-1, 4).clone()
    targets = torch.stack((row * 0.01, row * -0.02, row * 0.03, row * -0.04), dim=1)
    mask = torch.tensor([True, True, False]).expand(count, -1)
    selected = torch.arange(count, dtype=torch.int64).remainder(2)
    old_logp = (base_logits + torch.tensor([0.2, -0.1, 0.0])).masked_fill(~mask, -torch.inf)
    old_logp = old_logp.log_softmax(dim=1).gather(1, selected.unsqueeze(1)).squeeze(1)
    advantage = prepare_advantages(row.remainder(11) - 5.0, eps_std=PARAMETERS.eps_std)
    full_logits = base_logits.clone().requires_grad_()
    full_value = base_value.clone().requires_grad_()
    full = ppo_micro_terms(
        full_logits,
        full_value,
        mask,
        selected,
        old_logp,
        advantage,
        targets,
        minibatch_count=count,
        parameters=PARAMETERS,
    )
    full.loss.backward()
    micro_logits = base_logits.clone().requires_grad_()
    micro_value = base_value.clone().requires_grad_()
    accumulated = torch.zeros(())
    for start in range(0, count, 128):
        stop = min(count, start + 128)
        terms = ppo_micro_terms(
            micro_logits[start:stop],
            micro_value[start:stop],
            mask[start:stop],
            selected[start:stop],
            old_logp[start:stop],
            advantage[start:stop],
            targets[start:stop],
            minibatch_count=count,
            parameters=PARAMETERS,
        )
        accumulated += terms.loss.detach()
        terms.loss.backward()
    assert torch.allclose(accumulated, full.loss.detach(), atol=1e-6, rtol=1e-5)
    assert torch.allclose(micro_logits.grad, full_logits.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(micro_value.grad, full_value.grad, atol=1e-6, rtol=1e-5)
    assert torch.equal(full_logits.grad[:, 2], torch.zeros(count))


def test_four_seat_value_loss_and_legal_entropy_formula() -> None:
    logits = torch.zeros((1, 3), requires_grad=True)
    values = torch.zeros((1, 4), requires_grad=True)
    parameters = PpoParameters(0.2, 1e-8, 1.0, 0.0, 0.5)
    terms = ppo_micro_terms(
        logits,
        values,
        torch.tensor([[True, True, False]]),
        torch.tensor([0]),
        torch.tensor([-torch.log(torch.tensor(2.0))]),
        torch.zeros(1),
        torch.tensor([[2.0, 3.0, 5.0, 7.0]]),
        minibatch_count=1,
        parameters=parameters,
    )
    expected = torch.tensor((4.0 + 9.0 + 25.0 + 49.0) / 4.0) - 0.5 * torch.log(torch.tensor(2.0))
    assert torch.allclose(terms.loss, expected, atol=1e-6, rtol=1e-5)
    terms.loss.backward()
    assert torch.equal(values.grad, torch.tensor([[-1.0, -1.5, -2.5, -3.5]]))
    assert logits.grad[0, 2] == 0.0


@pytest.mark.parametrize(
    "kind", ["all_illegal", "selected_illegal", "outside", "nonfinite", "old_underflow"]
)
def test_invalid_policy_facts_fail_closed(kind: str) -> None:
    logits = torch.zeros((1, 3))
    mask = torch.tensor([[True, True, False]])
    selected = torch.tensor([0])
    old = torch.tensor([-0.7])
    if kind == "all_illegal":
        mask = torch.zeros_like(mask)
    elif kind == "selected_illegal":
        selected = torch.tensor([2])
    elif kind == "outside":
        selected = torch.tensor([3])
    elif kind == "nonfinite":
        logits[0, 2] = torch.nan
    else:
        old = torch.tensor([-1000.0])
    with pytest.raises(ContractError):
        ppo_micro_terms(
            logits,
            torch.zeros((1, 4)),
            mask,
            selected,
            old,
            torch.ones(1),
            torch.zeros((1, 4)),
            minibatch_count=1,
            parameters=PARAMETERS,
        )


def test_bc_pairing_and_exact_legal_kl() -> None:
    mask = torch.tensor([[True, True, False]])
    logits = torch.tensor([[0.0, 0.0, 99.0]], requires_grad=True)
    parameters = PpoParameters(0.2, 1e-8, 0.0, 1.0, 0.0)
    args = (
        logits,
        torch.zeros((1, 4)),
        mask,
        torch.tensor([0]),
        torch.tensor([-0.7]),
        torch.zeros(1),
        torch.zeros((1, 4)),
    )
    with pytest.raises(ContractError, match="BC logits"):
        ppo_micro_terms(*args, minibatch_count=1, parameters=parameters)
    terms = ppo_micro_terms(
        *args,
        minibatch_count=1,
        parameters=parameters,
        bc_logits=torch.tensor([[0.0, torch.log(torch.tensor(3.0)), -999.0]]),
    )
    expected = 0.5 * torch.log(torch.tensor(2.0)) + 0.5 * torch.log(torch.tensor(2.0 / 3.0))
    assert torch.allclose(terms.loss, expected, atol=1e-6, rtol=1e-5)
    terms.loss.backward()
    assert logits.grad[0, 2] == 0.0
