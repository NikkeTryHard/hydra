"""Deterministic PPO smoke artifact builder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from hydra_learner.model import ACTION_SPACE, OBS_CHANNELS, TILE_WIDTH, HydraPolicyNet
from hydra_learner.ppo.rl import (
    DEFAULT_GAE_GAMMA,
    DEFAULT_GAE_LAMBDA,
    PlayerDecisionStep,
    compute_player_local_gae,
    masked_log_prob,
)
from hydra_learner.ppo.rollout import PpoRolloutMetadata, save_ppo_rollout_artifact
from hydra_learner.ppo.step import PpoBatch
from hydra_learner.rl_experiments.reward_shaping import default_reward_shaping_metadata

RANK_UTILITY_U_A = "U_A"


@dataclass(frozen=True)
class RustDecisionRow:
    obs: torch.Tensor
    legal_mask: torch.Tensor
    player_id: int
    seat_id: int
    game_id: int
    turn: int
    action: int | None = None
    legal_count: int | None = None


@dataclass(frozen=True)
class RustGameRollout:
    rows: tuple[RustDecisionRow, ...]
    final_scores: tuple[int, int, int, int]
    placements: tuple[int, int, int, int]
    seed: int


@dataclass(frozen=True)
class PpoSmokeArtifactResult:
    artifact_path: Path
    batch: PpoBatch
    metrics: dict[str, object]


def build_ppo_batch_from_rust_rollout(
    rollout: RustGameRollout,
    *,
    model: HydraPolicyNet,
    torch_seed: int,
    rank_utility_used: str = RANK_UTILITY_U_A,
    gae_gamma: float = DEFAULT_GAE_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
    output_device: torch.device | None = None,
) -> PpoBatch:
    _validate_rollout_metadata(rollout)
    if rank_utility_used != RANK_UTILITY_U_A:
        raise ValueError("rank_utility_used must be U_A")
    obs = torch.stack([_require_obs(row.obs) for row in rollout.rows]).to(dtype=torch.float32)
    legal_mask = torch.stack([_require_legal_mask(row.legal_mask) for row in rollout.rows]).to(dtype=torch.bool)
    legal_count = torch.tensor([_require_legal_count(row) for row in rollout.rows], dtype=torch.int64)
    player_id = torch.tensor(
        [_require_range(row.player_id, "player_id", 0, 3) for row in rollout.rows], dtype=torch.int64
    )
    seat_id = torch.tensor([_require_range(row.seat_id, "seat_id", 0, 3) for row in rollout.rows], dtype=torch.int64)
    game_id = torch.tensor([row.game_id for row in rollout.rows], dtype=torch.int64)
    turn = torch.tensor([_require_minimum(row.turn, "turn", 0) for row in rollout.rows], dtype=torch.int64)
    if not bool((legal_count > 0).all()):
        raise ValueError("legal_mask has an all-illegal row")

    model_device = next(model.parameters()).device
    batch_device = torch.device("cpu") if output_device is None else output_device
    obs_for_model = obs.to(model_device)
    legal_mask_for_model = legal_mask.to(model_device)
    with torch.inference_mode():
        model.eval()
        outputs = model(obs_for_model)
        logits_model = outputs.policy_logits.detach().to(dtype=torch.float32)
        value_old_model = outputs.value.squeeze(1).detach().to(dtype=torch.float32)
        actions = _actions_from_rows_or_sample(rollout.rows, logits_model, legal_mask_for_model, torch_seed)
        old_logprob_model = (
            masked_log_prob(logits_model, legal_mask_for_model, actions).detach().to(dtype=torch.float32)
        )

    logits = logits_model.to(batch_device)
    value_old = value_old_model.to(batch_device)
    old_logprob = old_logprob_model.to(batch_device)
    actions = actions.to(batch_device)
    obs = obs.to(batch_device)
    legal_mask = legal_mask.to(batch_device)
    legal_count = legal_count.to(batch_device)
    player_id = player_id.to(batch_device)
    seat_id = seat_id.to(batch_device)
    game_id = game_id.to(batch_device)
    turn = turn.to(batch_device)
    gae = compute_player_local_gae(
        [
            PlayerDecisionStep(player_id=int(pid), value_old=float(value))
            for pid, value in zip(player_id.cpu().tolist(), value_old.cpu().tolist(), strict=True)
        ],
        final_placements=rollout.placements,
        gamma=gae_gamma,
        gae_lambda=gae_lambda,
    )
    raw_advantages = gae.raw_advantages.to(batch_device)
    returns = gae.returns.to(batch_device)
    batch = PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=old_logprob,
        value_old=value_old,
        raw_advantages=raw_advantages,
        returns=returns,
        bc_logits=logits.clone(),
        legal_count=legal_count,
        player_id=player_id,
        seat_id=seat_id,
        game_id=game_id,
        turn=turn,
        rank_utility_used=rank_utility_used,
    )
    batch.validate()
    _validate_terminal_reward_once(gae.terminal_player_stream.to(player_id.device), player_id)
    return batch


def write_ppo_smoke_rollout_artifact(
    path: Path,
    rollout: RustGameRollout,
    *,
    model: HydraPolicyNet,
    torch_seed: int,
) -> PpoSmokeArtifactResult:
    batch = build_ppo_batch_from_rust_rollout(rollout, model=model, torch_seed=torch_seed)
    save_ppo_rollout_artifact(
        path,
        batch,
        PpoRolloutMetadata(
            rank_utility_used=RANK_UTILITY_U_A,
            gae_gamma=DEFAULT_GAE_GAMMA,
            gae_lambda=DEFAULT_GAE_LAMBDA,
            reward_shaping=default_reward_shaping_metadata(gamma=DEFAULT_GAE_GAMMA, gae_lambda=DEFAULT_GAE_LAMBDA),
        ),
    )
    metrics = smoke_artifact_boundary_metrics(path, rollout, batch)
    return PpoSmokeArtifactResult(artifact_path=path, batch=batch, metrics=metrics)


def load_rust_game_rollout_json(path: Path) -> RustGameRollout:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rollout root must be a JSON object")
    rows_raw = payload.get("rows")
    if not isinstance(rows_raw, list):
        raise ValueError("rows must be a list")
    return RustGameRollout(
        rows=tuple(_row_from_json(row) for row in rows_raw),
        final_scores=_int_tuple4(payload.get("final_scores"), "final_scores"),
        placements=_int_tuple4(payload.get("placements"), "placements"),
        seed=_require_json_int(payload.get("seed"), "seed"),
    )


def _row_from_json(value: object) -> RustDecisionRow:
    if not isinstance(value, dict):
        raise ValueError("row must be a JSON object")
    obs_raw = value.get("obs")
    legal_raw = value.get("legal_mask")
    if not isinstance(obs_raw, list) or len(obs_raw) != OBS_CHANNELS * TILE_WIDTH:
        raise ValueError("row obs must be flat [192*34]")
    if not isinstance(legal_raw, list) or len(legal_raw) != ACTION_SPACE:
        raise ValueError("row legal_mask must be flat [46]")
    obs = torch.tensor([float(item) for item in obs_raw], dtype=torch.float32).reshape(OBS_CHANNELS, TILE_WIDTH)
    legal_mask = torch.tensor([_require_json_bool(item, "legal_mask item") for item in legal_raw], dtype=torch.bool)
    return RustDecisionRow(
        obs=obs,
        legal_mask=legal_mask,
        player_id=_require_json_int(value.get("player_id"), "player_id"),
        seat_id=_require_json_int(value.get("seat_id"), "seat_id"),
        game_id=_require_json_int(value.get("game_id"), "game_id"),
        turn=_require_json_int(value.get("turn"), "turn"),
        action=_require_json_int(value.get("action"), "action"),
        legal_count=_require_json_int(value.get("legal_count"), "legal_count"),
    )


def _int_tuple4(value: object, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must contain four integers")
    return cast("tuple[int, int, int, int]", tuple(_require_json_int(item, name) for item in value))


def _require_json_int(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value


def _require_json_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def smoke_artifact_boundary_metrics(path: Path, rollout: RustGameRollout, batch: PpoBatch) -> dict[str, object]:
    batch.validate()
    illegal = ~batch.legal_mask.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
    histogram = [0, 0, 0, 0]
    for placement in rollout.placements:
        histogram[placement] += 1
    return {
        "artifact_rows": batch.obs.shape[0],
        "num_games": 1,
        "num_decisions": batch.obs.shape[0],
        "illegal_action_count": int(illegal.sum().item()),
        "all_illegal_count": int((~batch.legal_mask.any(dim=1)).sum().item()),
        "mean_U_A": float(batch.raw_advantages.mean().item()),
        "placement_histogram": histogram,
        "checkpoint_path": None,
        "artifact_path": str(path),
        "seed": rollout.seed,
    }


def _actions_from_rows_or_sample(
    rows: tuple[RustDecisionRow, ...], logits: torch.Tensor, legal_mask: torch.Tensor, torch_seed: int
) -> torch.Tensor:
    explicit = [row.action for row in rows]
    if all(action is not None for action in explicit):
        action_ids = [action for action in explicit if action is not None]
        actions = torch.tensor(action_ids, dtype=torch.int64, device=logits.device)
        if not bool(legal_mask.gather(1, actions.unsqueeze(1)).squeeze(1).all()):
            raise ValueError("row action must be legal")
        return actions
    if any(action is not None for action in explicit):
        raise ValueError("actions must be all present or all absent")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(torch_seed)
    logits_cpu = logits.cpu()
    legal_mask_cpu = legal_mask.cpu()
    masked_logits = logits_cpu.masked_fill(~legal_mask_cpu, -1.0e9)
    probs = torch.softmax(masked_logits, dim=1).masked_fill(~legal_mask_cpu, 0.0)
    return (
        torch.multinomial(probs, num_samples=1, replacement=True, generator=generator)
        .squeeze(1)
        .to(device=logits.device, dtype=torch.int64)
    )


def _validate_rollout_metadata(rollout: RustGameRollout) -> None:
    if not rollout.rows:
        raise ValueError("rollout rows must not be empty")
    if len(rollout.final_scores) != 4:
        raise ValueError("final_scores must contain four scores")
    if len(rollout.placements) != 4 or sorted(rollout.placements) != [0, 1, 2, 3]:
        raise ValueError("placements must be a permutation of 0..3")


def _require_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.shape != (OBS_CHANNELS, TILE_WIDTH):
        raise ValueError(f"obs must have shape [{OBS_CHANNELS},{TILE_WIDTH}]")
    if obs.dtype is not torch.float32:
        raise TypeError("obs must be float32")
    if not bool(torch.isfinite(obs).all()):
        raise ValueError("obs must be finite")
    return obs


def _require_legal_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != (ACTION_SPACE,):
        raise ValueError(f"legal_mask must have shape [{ACTION_SPACE}]")
    if mask.dtype is not torch.bool:
        raise TypeError("legal_mask must be bool")
    return mask


def _require_legal_count(row: RustDecisionRow) -> int:
    expected = int(row.legal_mask.sum().item())
    if row.legal_count is None:
        return expected
    if row.legal_count != expected:
        raise ValueError("legal_count must equal legal_mask.sum()")
    return row.legal_count


def _require_range(value: int, name: str, minimum: int, maximum: int) -> int:
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return value


def _require_minimum(value: int, name: str, minimum: int) -> int:
    if value < minimum:
        raise ValueError(f"{name} below minimum")
    return value


def _validate_terminal_reward_once(terminal_player_stream: torch.Tensor, player_id: torch.Tensor) -> None:
    for player in range(4):
        count = int(((player_id == player) & terminal_player_stream).sum().item())
        if count != 1:
            raise ValueError("terminal reward must be assigned once per player stream")
