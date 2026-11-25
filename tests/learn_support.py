"""Deterministic policy and arena doubles used only by learner tests."""

import math
from collections.abc import Mapping, Sequence

import hydra3.learn_collect as collect_module
import pytest
import torch
from hydra3.learn_schema import GameRecord
from hydra3.model_schema import ACTION_WIDTH, TOKEN_COUNT, TOKEN_WIDTH
from hydra3.runtime import ArenaBatch


class TinyPolicy(torch.nn.Module):
    """A parameterized policy whose preferred action is easy to predict."""

    def __init__(self, *, prefer_first: bool = True, value_bias: float = 0.0) -> None:
        super().__init__()
        self.prefer_first = prefer_first
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.value_bias = torch.nn.Parameter(torch.tensor(value_bias))

    def forward(
        self, tokens: torch.Tensor, candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidates_count, _ = candidates.shape
        if self.prefer_first:
            preferred = torch.zeros(
                (batch, candidates_count), device=tokens.device, dtype=tokens.dtype
            )
            preferred[:, 0] = 10000.0
            logits = preferred + self.bias
        else:
            logits = self.bias + candidates[:, :, 0].to(tokens.dtype) * 0.01
        value = self.value_bias + tokens.mean(dim=(1, 2))
        return logits, value


def assert_finite_metrics(metrics: Mapping[str, object]) -> None:
    for value in metrics.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(float(value))


def float_metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics[name]
    assert isinstance(value, (int, float))
    return float(value)


def descriptor_rows(ids: Sequence[int]) -> torch.Tensor:
    rows = torch.full((len(ids), ACTION_WIDTH), -1, dtype=torch.int32)
    rows[:, 0] = torch.tensor(list(ids), dtype=torch.int32)
    rows[:, 1] = 0
    rows[:, 2] = 4
    rows[:, 3] = -1
    rows[:, 4] = 0
    rows[:, 5] = 0
    rows[:, 6] = 0
    rows[:, 11:15] = 0
    rows[:, 15] = 0
    return rows


def make_batch(
    rows: Sequence[tuple[int, int, Sequence[int]]], device: torch.device | None = None
) -> ArenaBatch:
    device = torch.device("cpu") if device is None else device
    slots = torch.tensor([row[0] for row in rows], dtype=torch.int32)
    seats = torch.tensor([row[1] for row in rows], dtype=torch.uint8)
    tokens = torch.zeros((len(rows), TOKEN_COUNT, TOKEN_WIDTH), dtype=torch.float32)
    widest = max(len(row[2]) for row in rows)
    actions = torch.full((len(rows), widest, ACTION_WIDTH), -1, dtype=torch.int32)
    for row, (_, _, ids) in enumerate(rows):
        actions[row, : len(ids)] = descriptor_rows(ids)
    return ArenaBatch(slots, seats, tokens.to(device), actions.to(device))


def make_game(
    rows: Sequence[tuple[int, Sequence[int]]],
    *,
    ranks: Sequence[int] = (1, 2, 3, 4),
    chosen: Sequence[int] | None = None,
    seed: int = 1,
    scores: Sequence[int] = (25000, 25000, 25000, 25000),
) -> GameRecord:
    tokens = torch.zeros((len(rows), TOKEN_COUNT, TOKEN_WIDTH), dtype=torch.float32)
    actions = torch.cat([descriptor_rows(ids) for _, ids in rows], dim=0)
    legal = torch.tensor([len(ids) for _, ids in rows], dtype=torch.int64)
    chosen_ids = [ids[0] for _, ids in rows] if chosen is None else list(chosen)
    seats = torch.tensor([seat for seat, _ in rows], dtype=torch.int64)
    return GameRecord(
        seed=seed,
        tokens=tokens,
        actions=actions,
        legal=legal,
        chosen=torch.tensor(chosen_ids, dtype=torch.int64),
        seats=seats,
        forced=legal == 1,
        scores=torch.tensor(scores, dtype=torch.int32),
        ranks=torch.tensor(list(ranks), dtype=torch.int32),
    )


def empty_game() -> GameRecord:
    return GameRecord(
        seed=1,
        tokens=torch.zeros((0, TOKEN_COUNT, TOKEN_WIDTH), dtype=torch.float32),
        actions=torch.empty((0, ACTION_WIDTH), dtype=torch.int32),
        legal=torch.empty(0, dtype=torch.int64),
        chosen=torch.empty(0, dtype=torch.int64),
        seats=torch.empty(0, dtype=torch.int64),
        forced=torch.empty(0, dtype=torch.bool),
        scores=torch.zeros(4, dtype=torch.int32),
        ranks=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
    )


class ScriptedArena:
    """Minimal sequence of batches followed by final outcomes."""

    def __init__(self, seeds: list[int], device: torch.device) -> None:
        self.seeds = list(seeds)
        self.device = device
        self.scripts: list[ArenaBatch] = []
        self.outcome_payload: list[tuple[int, list[int], list[int]]] = []
        self.steps: list[list[int]] = []
        self.batch_calls = 0

    def step(self, slots: torch.Tensor, ids: torch.Tensor) -> None:
        self.steps.append(ids.tolist())

    def outcomes(self) -> tuple[torch.Tensor, torch.Tensor]:
        ordered = sorted(self.outcome_payload)
        return (
            torch.tensor([scores for _, scores, _ in ordered], dtype=torch.int32),
            torch.tensor([ranks for _, _, ranks in ordered], dtype=torch.int32),
        )

    def batch(self) -> ArenaBatch | None:
        if self.batch_calls >= len(self.scripts):
            return None
        batch = self.scripts[self.batch_calls]
        self.batch_calls += 1
        return ArenaBatch(
            batch.slot,
            batch.seat,
            batch.tokens.to(self.device),
            batch.actions.to(self.device),
        )


def install_scripted_arena(
    monkeypatch: pytest.MonkeyPatch,
    scripts: Sequence[ArenaBatch],
    outcomes: Sequence[tuple[int, Sequence[int], Sequence[int]]],
) -> list[ScriptedArena]:
    created: list[ScriptedArena] = []
    scripted = list(scripts)
    terminal = [(slot, list(scores), list(ranks)) for slot, scores, ranks in outcomes]

    class FakeArena(ScriptedArena):
        def __init__(self, seeds: list[int], device: torch.device) -> None:
            super().__init__(seeds, device)
            self.scripts = scripted
            self.outcome_payload = terminal
            created.append(self)

    monkeypatch.setattr(collect_module, "MahJaxArena", FakeArena)
    return created
