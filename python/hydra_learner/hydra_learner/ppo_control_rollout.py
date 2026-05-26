from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from hydra_learner.ppo_control_config import RANK_UTILITY, PpoControlConfig
from hydra_learner.ppo_smoke import RustDecisionRow, RustGameRollout, build_ppo_batch_from_rust_rollout
from hydra_learner.ppo_step import PpoBatch
from hydra_learner.rl import DEFAULT_GAE_GAMMA, DEFAULT_GAE_LAMBDA

if TYPE_CHECKING:
    from hydra_learner.model import HydraPolicyNet


def _collect_native_rollout(
    extension: Any, config: PpoControlConfig, policy_dir: Path, seed: int
) -> Mapping[str, object]:
    collect = getattr(extension, "collect_ppo_rollouts_rust_native", None)
    if not callable(collect):
        raise ValueError("arena extension missing collect_ppo_rollouts_rust_native")
    payload = collect(
        config.games_per_update,
        seed,
        str(policy_dir),
        config.arena_batch_decisions,
        config.device,
        config.arena_threads,
        config.temperature,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("native PPO rollout collector must return a mapping")
    return payload


def _batch_from_native_payload(payload: Mapping[str, object], model: HydraPolicyNet) -> PpoBatch:
    rows = payload.get("rows")
    terminals = payload.get("games")
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise ValueError("native rollout rows must be a sequence")
    if not isinstance(terminals, Sequence) or isinstance(terminals, str | bytes):
        raise ValueError("native rollout games must be a sequence")
    rows_by_game: dict[int, list[Mapping[str, object]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("native rollout row must be an object")
        game_id = int(row["game_id"])
        rows_by_game.setdefault(game_id, []).append(row)
    batches: list[PpoBatch] = []
    for terminal in terminals:
        if not isinstance(terminal, Mapping):
            raise ValueError("native rollout game terminal must be an object")
        game_id = int(terminal["game_id"])
        game_rows = rows_by_game.get(game_id)
        if not game_rows:
            raise ValueError(f"native rollout game {game_id} has no rows")
        rollout = RustGameRollout(
            rows=tuple(_decision_row(row) for row in game_rows),
            final_scores=_tuple4(terminal["final_scores"], "final_scores"),
            placements=_tuple4(terminal["placements"], "placements"),
            seed=int(terminal["seed"]),
        )
        batches.append(
            build_ppo_batch_from_rust_rollout(
                rollout,
                model=model,
                torch_seed=rollout.seed,
                gae_gamma=DEFAULT_GAE_GAMMA,
                gae_lambda=DEFAULT_GAE_LAMBDA,
            )
        )
    return _concat_batches(batches)


def _decision_row(row: Mapping[str, object]) -> RustDecisionRow:
    obs = torch.tensor(cast(Sequence[float], row["obs"]), dtype=torch.float32).reshape(192, 34)
    legal_mask = torch.tensor(cast(Sequence[bool], row["legal_mask"]), dtype=torch.bool)
    return RustDecisionRow(
        obs=obs,
        legal_mask=legal_mask,
        player_id=_int_field(row, "player_id"),
        seat_id=_int_field(row, "seat_id"),
        game_id=_int_field(row, "game_id"),
        turn=_int_field(row, "turn"),
        action=_int_field(row, "action"),
        legal_count=_int_field(row, "legal_count"),
    )


def _int_field(row: Mapping[str, object], key: str) -> int:
    value = row[key]
    if not isinstance(value, int):
        raise TypeError(f"native rollout row {key} must be int")
    return value


def _tuple4(value: object, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or len(value) != 4:
        raise ValueError(f"{name} must contain four ints")
    return (int(value[0]), int(value[1]), int(value[2]), int(value[3]))


def _concat_batches(batches: list[PpoBatch]) -> PpoBatch:
    if not batches:
        raise ValueError("rollout produced no PPO batches")
    return PpoBatch(
        obs=torch.cat([b.obs for b in batches], dim=0),
        actions=torch.cat([b.actions for b in batches], dim=0),
        legal_mask=torch.cat([b.legal_mask for b in batches], dim=0),
        old_logprob=torch.cat([b.old_logprob for b in batches], dim=0),
        value_old=torch.cat([b.value_old for b in batches], dim=0),
        raw_advantages=torch.cat([b.raw_advantages for b in batches], dim=0),
        returns=torch.cat([b.returns for b in batches], dim=0),
        bc_logits=torch.cat([b.bc_logits for b in batches], dim=0),
        legal_count=torch.cat([b.legal_count for b in batches], dim=0),
        player_id=torch.cat([b.player_id for b in batches if b.player_id is not None], dim=0),
        seat_id=torch.cat([b.seat_id for b in batches if b.seat_id is not None], dim=0),
        game_id=torch.cat([b.game_id for b in batches if b.game_id is not None], dim=0),
        turn=torch.cat([b.turn for b in batches if b.turn is not None], dim=0),
        rank_utility_used=RANK_UTILITY,
    )


def _batch_to_device(batch: PpoBatch, device: torch.device) -> PpoBatch:
    return PpoBatch(
        obs=batch.obs.to(device),
        actions=batch.actions.to(device),
        legal_mask=batch.legal_mask.to(device),
        old_logprob=batch.old_logprob.to(device),
        value_old=batch.value_old.to(device),
        raw_advantages=batch.raw_advantages.to(device),
        returns=batch.returns.to(device),
        bc_logits=batch.bc_logits.to(device),
        legal_count=batch.legal_count.to(device),
        player_id=None if batch.player_id is None else batch.player_id.to(device),
        seat_id=None if batch.seat_id is None else batch.seat_id.to(device),
        game_id=None if batch.game_id is None else batch.game_id.to(device),
        turn=None if batch.turn is None else batch.turn.to(device),
        rank_utility_used=batch.rank_utility_used,
    )
