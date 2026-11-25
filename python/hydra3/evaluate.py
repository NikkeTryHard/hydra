"""Evaluate one challenger against three frozen opponents on rotated duplicate walls."""

from statistics import fmean

import numpy as np
import torch

from .mahjax_runtime import MahJaxArena
from .model import Policy
from .runtime import ArenaBatch, derive_seed


def _select(
    batch: ArenaBatch, focal: torch.Tensor, candidate: Policy, reference: Policy
) -> torch.Tensor:
    ids = torch.empty(len(batch.seat), dtype=torch.int32, device=batch.actions.device)
    owned = batch.seat == focal
    for active, policy in ((owned, candidate), (~owned, reference)):
        rows = active.nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        idx = rows.to(batch.actions.device)
        actions = batch.actions.index_select(0, idx)
        logits, _ = policy(batch.tokens.index_select(0, idx), actions)
        choice = logits.argmax(dim=1, keepdim=True)
        ids.index_copy_(0, idx, actions[:, :, 0].gather(1, choice).squeeze(1))
    return ids


def _interval(contrasts: list[float], seed: int) -> list[float] | None:
    if len(contrasts) < 2:
        return None
    # At most 13 distinct four-seat rank sums occur. Resampling their counts has the same
    # law as resampling whole wall blocks, without building a draws-by-blocks array.
    values, counts = np.unique(contrasts, return_counts=True)
    n = len(contrasts)
    rng = np.random.default_rng(derive_seed(seed, "eval/block-bootstrap", 0))
    draws = rng.multinomial(n, counts / n, size=2000) @ values / n
    return np.quantile(draws, [0.025, 0.975], method="inverted_cdf").tolist()


@torch.inference_mode()
def evaluate(
    candidate: Policy,
    reference: Policy,
    *,
    blocks: int,
    seed: int,
    device: torch.device,
    cohort: int = 32,
) -> dict:
    if blocks <= 0 or cohort <= 0:
        raise ValueError("blocks and cohort must be positive")
    candidate.eval()
    reference.eval()
    contrasts: list[float] = []
    seat_totals = [0] * 4
    placements = [0] * 4
    for first in range(0, blocks, cohort):
        walls = [
            derive_seed(seed, "eval/wall", i) for i in range(first, min(first + cohort, blocks))
        ]
        # One lane per wall and focal seat. Lanes share only their wall seed, so this
        # plays the same four rotated games as four single-lane arenas while paying a
        # quarter of the device round trips per turn.
        lanes = [wall for wall in walls for _ in range(4)]
        block_ranks: list[list[int]] = [[] for _ in walls]
        arena = MahJaxArena(lanes, device)
        while (batch := arena.batch()) is not None:
            arena.step(
                batch.slot, _select(batch, torch.remainder(batch.slot, 4), candidate, reference)
            )
        _, ranks = arena.outcomes()
        for lane, rank_row in enumerate(ranks.tolist()):
            focal = lane % 4
            rank = rank_row[focal]
            block_ranks[lane // 4].append(rank)
            placements[rank - 1] += 1
            seat_totals[focal] += rank
        # The four rotations share walls. Their mean is one observation, not four independent games.
        contrasts.extend((4 * fmean(ranks) - 10) / 3 for ranks in block_ranks)
    return {
        "wall_blocks": blocks,
        "completed_hanchan": 4 * blocks,
        "candidate_mean_rank": sum(seat_totals) / (4 * blocks),
        "rank_by_focal_seat": [total / blocks for total in seat_totals],
        "placements_first_to_fourth": placements,
        "rank_delta_vs_mean_opponent_lower_is_better": fmean(contrasts),
        "wall_block_bootstrap_95pct_descriptive": _interval(contrasts, seed),
    }
