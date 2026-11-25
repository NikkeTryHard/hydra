"""Training tensor layout, candidate masking and optimizer details."""

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

from .learn_schema import GameRecord
from .model_schema import ACTION_WIDTH, TOKEN_COUNT, TOKEN_WIDTH

GRAD_CLIP_NORM = 1.0

MetricValue = int | float | str


@dataclass(frozen=True, slots=True)
class DecisionTensors:
    """Flattened decisions with legal descriptors kept dense and unpadded."""

    tokens: torch.Tensor
    actions: torch.Tensor
    offsets: torch.Tensor
    legal: torch.Tensor
    chosen_index: torch.Tensor
    target: torch.Tensor
    forced: torch.Tensor
    seats: torch.Tensor


def flatten_games(games: Iterable[GameRecord]) -> DecisionTensors:
    """Bind each decision to its actor's terminal 2.5-rank utility."""
    ordered = tuple(games)
    if not ordered:
        raise ValueError("at least one game is required")
    count = sum(game.tokens.shape[0] for game in ordered)
    tokens = torch.cat([game.tokens for game in ordered], dim=0)
    actions = torch.cat([game.actions for game in ordered], dim=0)
    legal = torch.cat([game.legal for game in ordered], dim=0)
    chosen = torch.cat([game.chosen for game in ordered], dim=0)
    seats = torch.cat([game.seats for game in ordered], dim=0)
    target = torch.cat([2.5 - game.ranks[game.seats].float() for game in ordered], dim=0)

    offsets = torch.cumsum(legal, dim=0) - legal
    decision_of_row = torch.repeat_interleave(torch.arange(count), legal)
    matches = actions[:, 0].eq(chosen.repeat_interleave(legal))
    match_counts = torch.zeros(count, dtype=torch.int64)
    match_counts.index_add_(0, decision_of_row, matches.to(torch.int64))
    if not bool((match_counts == 1).all()):
        raise ValueError("chosen canonical id does not name one legal descriptor")
    match_positions = matches.nonzero(as_tuple=False).squeeze(1)
    decisions_hit = decision_of_row.index_select(0, match_positions)
    if not torch.equal(decisions_hit, torch.arange(count)):
        raise ValueError("chosen canonical matches are not in decision order")
    chosen_index = match_positions - offsets.index_select(0, decisions_hit)

    return DecisionTensors(
        tokens=tokens,
        actions=actions,
        offsets=offsets,
        legal=legal,
        chosen_index=chosen_index,
        target=target,
        forced=legal == 1,
        seats=seats,
    )


def pad_candidates(
    decisions: DecisionTensors, indices: torch.Tensor, *, bucket: bool = False
) -> torch.Tensor:
    """Gather selected legal rows and pad only to their largest rectangle."""
    selected = indices.to(device=decisions.actions.device, dtype=torch.int64)
    if selected.ndim != 1 or selected.numel() == 0:
        raise ValueError("candidate padding requires a nonempty decision index")
    counts = decisions.legal.index_select(0, selected)
    starts = decisions.offsets.index_select(0, selected)
    widest = int(counts.max())
    if bucket:
        # A few widths prevent recompilation for every observed candidate count.
        widest = max(16, 1 << (widest - 1).bit_length())
    columns = torch.arange(widest, dtype=torch.int64, device=selected.device)
    positions = starts.unsqueeze(1) + columns.unsqueeze(0)
    valid = columns.unsqueeze(0) < counts.unsqueeze(1)
    safe = positions.clamp_(0, decisions.actions.shape[0] - 1)
    padded = decisions.actions.index_select(0, safe.reshape(-1)).view(
        selected.numel(), widest, ACTION_WIDTH
    )
    return padded.masked_fill(~valid.unsqueeze(-1), -1)


def minibatch_order(count: int, size: int, generator: torch.Generator) -> Iterator[torch.Tensor]:
    """Yield deterministic shuffled index chunks from a named generator."""
    if count <= 0 or size <= 0:
        raise ValueError("minibatching requires a positive count and size")
    order = torch.randperm(count, generator=generator)
    for start in range(0, count, size):
        yield order[start : start + size]


def score_policy(
    model: torch.nn.Module, tokens: torch.Tensor, actions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the shared actor forward and mask candidate padding."""
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (TOKEN_COUNT, TOKEN_WIDTH):
        raise ValueError(f"tokens must be [batch, {TOKEN_COUNT}, {TOKEN_WIDTH}]")
    if actions.ndim != 3 or actions.shape[2] != ACTION_WIDTH:
        raise ValueError(f"actions must be [batch, candidates, {ACTION_WIDTH}]")
    if actions.shape[0] != tokens.shape[0] or actions.device != tokens.device:
        raise ValueError("tokens and actions must share a batch and device")
    if actions.dtype not in (torch.int32, torch.int64):
        raise TypeError("candidate descriptors must be integer tensors")

    if tokens.device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(tokens, actions)
    else:
        output = model(tokens, actions)
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("policy must return (logits, value)")
    logits, value = output
    if not isinstance(logits, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("policy must return two tensors")
    if tuple(logits.shape) != tuple(actions.shape[:2]):
        raise ValueError("policy logits must be [batch, candidates]")
    if tuple(value.shape) != (tokens.shape[0],):
        raise ValueError("policy value must be actor-only [batch]")

    logits = logits.float()
    value = value.float()
    mask = actions[:, :, 0] >= 0
    masked = logits.masked_fill(~mask, float("-inf"))
    return masked, value, mask


def masked_log_probs(masked_logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(masked_logits, dim=1)


def gather_action_log_probs(log_probs: torch.Tensor, chosen_index: torch.Tensor) -> torch.Tensor:
    return log_probs.gather(1, chosen_index.unsqueeze(1)).squeeze(1)


def masked_entropy(masked_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Entropy over legal actions with a finite zero log-probability on padding."""
    probabilities = torch.softmax(masked_logits, dim=1)
    log_probs = torch.log_softmax(masked_logits, dim=1).masked_fill(~mask, 0.0)
    return -(probabilities * log_probs).sum(dim=1)


def step_optimizer(optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> None:
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    if not bool(torch.isfinite(norm).all()):
        raise FloatingPointError("non-finite gradient norm before optimizer step")
    optimizer.step()


def assert_finite_parameters(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(f"non-finite parameter after update: {name}")


def finite_metrics(metrics: dict[str, MetricValue]) -> dict[str, MetricValue]:
    for name, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"non-finite training metric: {name}")
    return metrics
