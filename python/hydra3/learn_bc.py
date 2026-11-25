"""Masked behaviour cloning with terminal-placement value warm start."""

from collections.abc import Iterable

import torch

from .learn_batch import (
    MetricValue,
    assert_finite_parameters,
    finite_metrics,
    flatten_games,
    gather_action_log_probs,
    masked_log_probs,
    minibatch_order,
    pad_candidates,
    score_policy,
    step_optimizer,
)
from .learn_schema import GameRecord
from .runtime import derive_seed

BC_VALUE_COEF = 0.5
MINIBATCH_STREAM = "bc/minibatch"


def update_bc(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    games: Iterable[GameRecord],
    *,
    device: torch.device,
    epochs: int,
    minibatch: int,
    seed: int,
) -> dict[str, MetricValue]:
    """Imitate legal labelled actions; forced no-choice rows only train value."""
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if type(minibatch) is not int or minibatch < 1:
        raise ValueError("minibatch must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")

    ordered = tuple(games)
    decisions = flatten_games(ordered)
    count = decisions.tokens.shape[0]
    actor_mask = ~decisions.forced
    trainable = int(actor_mask.sum())
    if trainable == 0:
        raise ValueError("behaviour cloning requires at least one non-forced decision")

    generator = torch.Generator().manual_seed(derive_seed(seed, MINIBATCH_STREAM, 0))
    policy_sum = torch.zeros((), device=device)
    value_sum = torch.zeros((), device=device)
    correct_hits = torch.zeros((), device=device, dtype=torch.int64)
    policy_rows = 0
    value_rows = 0
    updates = 0

    for _ in range(epochs):
        for index in minibatch_order(count, minibatch, generator):
            actor_cpu = actor_mask.index_select(0, index)
            rows = int(actor_cpu.sum())
            tokens = decisions.tokens.index_select(0, index).to(device)
            actions = pad_candidates(decisions, index).to(device)
            chosen = decisions.chosen_index.index_select(0, index).to(device)
            target = decisions.target.index_select(0, index).to(device)
            actor = actor_cpu.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, value, _ = score_policy(model, tokens, actions)
            log_probs = masked_log_probs(logits)
            cross_entropy = -gather_action_log_probs(log_probs, chosen)
            if rows:
                policy_loss = (cross_entropy * actor).sum() / rows
            else:
                policy_loss = logits.new_zeros(())
            value_loss = 0.5 * ((value - target) ** 2).mean()
            loss = policy_loss + BC_VALUE_COEF * value_loss
            loss.backward()
            step_optimizer(optimizer, model)

            if rows:
                policy_sum += (cross_entropy * actor).sum().detach()
                correct = (logits.argmax(dim=1) == chosen) & actor
                correct_hits += correct.sum()
                policy_rows += rows
            value_sum += (0.5 * ((value - target) ** 2)).sum().detach()
            value_rows += index.numel()
            updates += 1

    totals = torch.stack((policy_sum, value_sum)).cpu()
    correct_count = int(correct_hits.cpu())
    assert_finite_parameters(model)
    policy_total, value_total = totals.tolist()
    metrics: dict[str, MetricValue] = {
        "games": len(ordered),
        "decisions": count,
        "trainable_decisions": trainable,
        "updates": updates,
        "policy_loss": policy_total / policy_rows if policy_rows else 0.0,
        "value_loss": value_total / value_rows,
        "accuracy": correct_count / policy_rows if policy_rows else 0.0,
    }
    return finite_metrics(metrics)
