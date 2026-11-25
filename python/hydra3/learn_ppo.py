"""Clipped PPO over one generation of frozen-policy decisions."""

from functools import cache
from time import perf_counter_ns

import torch

from .learn_batch import (
    MetricValue,
    assert_finite_parameters,
    finite_metrics,
    gather_action_log_probs,
    masked_entropy,
    masked_log_probs,
    minibatch_order,
    pad_candidates,
    score_policy,
    step_optimizer,
)
from .learn_collect import Rollout
from .runtime import derive_seed

PPO_CLIP = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_LOG_RATIO = 20.0
MINIBATCH_STREAM = "train/minibatch"


def _objective(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    actions: torch.Tensor,
    chosen: torch.Tensor,
    target: torch.Tensor,
    old_lp: torch.Tensor,
    adv: torch.Tensor,
    actor: torch.Tensor,
    rows: torch.Tensor | int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, value, mask = score_policy(model, tokens, actions)
    logp = gather_action_log_probs(masked_log_probs(logits), chosen)
    log_ratio = logp - old_lp
    ratio = torch.exp(torch.clamp(log_ratio, max=MAX_LOG_RATIO))
    clipped = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP)
    per_policy = -torch.minimum(ratio * adv, clipped * adv)
    per_entropy = masked_entropy(logits, mask)
    policy_sum = (per_policy * actor).sum()
    entropy_sum = (per_entropy * actor).sum()
    per_value = 0.5 * ((value - target) ** 2)
    loss = VALUE_COEF * per_value.mean()
    if rows is not None:
        loss = policy_sum / rows + loss - ENTROPY_COEF * (entropy_sum / rows)
    detached_ratio = ratio.detach() - 1.0
    return (
        loss,
        policy_sum.detach(),
        entropy_sum.detach(),
        per_value.detach().sum(),
        ((detached_ratio - log_ratio.detach()) * actor).sum(),
        ((detached_ratio.abs() > PPO_CLIP) & actor).sum(),
    )


def update_ppo(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    *,
    device: torch.device,
    epochs: int,
    minibatch: int,
    seed: int,
    telemetry: dict[str, int] | None = None,
    compile_learner: bool = False,
) -> dict[str, MetricValue]:
    """Update against stored behaviour probabilities and per-seat terminal credit."""
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if type(minibatch) is not int or minibatch < 1:
        raise ValueError("minibatch must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if compile_learner and device.type != "cuda":
        raise ValueError("compiled PPO requires CUDA")

    start = perf_counter_ns() if telemetry is not None else 0
    decisions = rollout.decisions
    count = decisions.tokens.shape[0]
    archive_device = decisions.tokens.device
    old_logp = rollout.old_logp.to(archive_device)
    old_value = rollout.old_value.to(archive_device)
    if tuple(old_logp.shape) != (count,) or tuple(old_value.shape) != (count,):
        raise ValueError("stored behaviour tensors do not match the collected decisions")
    if not bool(torch.isfinite(old_logp).all()) or not bool(torch.isfinite(old_value).all()):
        raise FloatingPointError("stored behaviour log probabilities or values are non-finite")

    advantages = decisions.target.to(archive_device) - old_value
    actor_mask = ~decisions.forced
    generator = torch.Generator().manual_seed(derive_seed(seed, MINIBATCH_STREAM, 0))

    policy_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    value_sum = torch.zeros((), device=device)
    kl_sum = torch.zeros((), device=device)
    clip_hits = torch.zeros((), device=device, dtype=torch.int64)
    policy_rows = 0
    value_rows = 0
    updates = 0
    if telemetry is not None:
        telemetry["learner_setup_ns"] = perf_counter_ns() - start
    input_ns = compute_ns = update_ns = 0

    for _ in range(epochs):
        for index in minibatch_order(count, minibatch, generator):
            start = perf_counter_ns() if telemetry is not None else 0
            rows_index = index.to(archive_device)
            actor_cpu = actor_mask.index_select(0, rows_index)
            rows = int(actor_cpu.sum())
            tokens = decisions.tokens.index_select(0, rows_index).to(device)
            compiled = compile_learner and index.numel() == minibatch and rows > 0
            actions = pad_candidates(decisions, rows_index, bucket=compiled).to(device)
            chosen = decisions.chosen_index.index_select(0, rows_index).to(device)
            target = decisions.target.index_select(0, rows_index).to(device)
            old_lp = old_logp.index_select(0, rows_index).to(device)
            adv = advantages.index_select(0, rows_index).to(device)
            actor = actor_cpu.to(device)
            if telemetry is not None:
                input_ns += perf_counter_ns() - start
                start = perf_counter_ns()

            optimizer.zero_grad(set_to_none=True)
            objective = _objective
            if compiled:
                objective = _compiled_objective()
            denominator = actor.sum() if compiled else (rows or None)
            loss, policy, entropy, value, kl, hits = objective(
                model, tokens, actions, chosen, target, old_lp, adv, actor, denominator
            )
            loss.backward()
            if telemetry is not None:
                compute_ns += perf_counter_ns() - start
                start = perf_counter_ns()
            step_optimizer(optimizer, model)
            if telemetry is not None:
                update_ns += perf_counter_ns() - start

            if rows:
                policy_sum += policy
                entropy_sum += entropy
                kl_sum += kl
                clip_hits += hits
                policy_rows += rows
            value_sum += value
            value_rows += index.numel()
            updates += 1

    if telemetry is not None:
        telemetry.update(
            learner_input_ns=input_ns,
            learner_forward_backward_submit_ns=compute_ns,
            learner_clip_step_ns=update_ns,
        )

    totals = torch.stack((policy_sum, entropy_sum, value_sum, kl_sum)).cpu()
    clip_count = int(clip_hits.cpu())
    assert_finite_parameters(model)
    policy_total, entropy_total, value_total, kl_total = totals.tolist()
    metrics: dict[str, MetricValue] = {
        "games": len(rollout.seeds),
        "decisions": count,
        "trainable_decisions": int(actor_mask.sum()),
        "updates": updates,
        "policy_loss": policy_total / policy_rows if policy_rows else 0.0,
        "value_loss": value_total / value_rows,
        "entropy": entropy_total / policy_rows if policy_rows else 0.0,
        "approx_kl": kl_total / policy_rows if policy_rows else 0.0,
        "clip_fraction": clip_count / policy_rows if policy_rows else 0.0,
    }
    return finite_metrics(metrics)


@cache
def _compiled_objective():
    # Lazy Triton module destruction invalidates multi-shape CUDA capture on
    # the pinned runtime. Keep fusion, but leave learner launches uncaptured.
    return torch.compile(_objective, fullgraph=True, dynamic=False)
