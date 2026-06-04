from __future__ import annotations

import importlib
import os
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.utils.dlpack as torch_dlpack

from hydra_learner.ppo.rl import DEFAULT_GAE_GAMMA, DEFAULT_GAE_LAMBDA, PLACEMENT_UTILITY_DEFAULT
from hydra_learner.ppo.step import PpoBatch

if TYPE_CHECKING:
    from hydra_learner.model import HydraPolicyNet
    from hydra_learner.ppo.config import PpoControlConfig
    from hydra_learner.ppo.rollout import PpoSnapshotMetadata


@dataclass(frozen=True)
class MahjaxPpoRolloutResult:
    batch: PpoBatch
    timing: dict[str, float]
    row_count: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class _MahjaxKernels:
    jax: Any
    jnp: Any
    observe_project_run: Any
    step_update_run: Any
    init_run: Any
    empty_safety_bank_batch_jax: Any


_KERNEL_CACHE: dict[tuple[int, int, str, bool], _MahjaxKernels] = {}
_DEFAULT_JAX_COMPILATION_CACHE_DIR = "local/jax_cache/mahjax_ppo"
_JAX_COMPILATION_CACHE_CONFIGURED_DIR: list[str] = []
_DEFAULT_MAHJAX_COMPLETION_SYNC_INTERVAL = 32
# Completion checks synchronize the device stream. Default 32 is evidence-backed;
# tune with HYDRA_MAHJAX_COMPLETION_SYNC_INTERVAL only for focused benchmarks.


def _jax_compilation_cache_dir() -> str:
    return os.environ.get("HYDRA_MAHJAX_JAX_CACHE_DIR", _DEFAULT_JAX_COMPILATION_CACHE_DIR)


def _use_jax_aot() -> bool:
    raw = os.environ.get("HYDRA_MAHJAX_AOT", "1").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError("HYDRA_MAHJAX_AOT must be one of 1/0, true/false, yes/no, on/off")


def _completion_sync_interval() -> int:
    raw = os.environ.get("HYDRA_MAHJAX_COMPLETION_SYNC_INTERVAL")
    if raw is None:
        return _DEFAULT_MAHJAX_COMPLETION_SYNC_INTERVAL
    interval = int(raw)
    if interval < 1:
        raise ValueError("HYDRA_MAHJAX_COMPLETION_SYNC_INTERVAL must be positive")
    return interval


def _sync_timing_enabled() -> bool:
    raw = os.environ.get("HYDRA_MAHJAX_SYNC_TIMING", "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError("HYDRA_MAHJAX_SYNC_TIMING must be one of 1/0, true/false, yes/no, on/off")


def _configure_jax_compilation_cache(jax: Any) -> None:
    cache_dir = _jax_compilation_cache_dir()
    if _JAX_COMPILATION_CACHE_CONFIGURED_DIR:
        if cache_dir != _JAX_COMPILATION_CACHE_CONFIGURED_DIR[0]:
            raise ValueError("HYDRA_MAHJAX_JAX_CACHE_DIR cannot change after MahJAX JAX cache configuration")
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    with suppress(ValueError):
        jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    _JAX_COMPILATION_CACHE_CONFIGURED_DIR.append(cache_dir)


def _get_mahjax_kernels(env_count: int, device_index: int) -> _MahjaxKernels:
    use_aot = _use_jax_aot()
    key = (env_count, device_index, _jax_compilation_cache_dir(), use_aot)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    jax = importlib.import_module("jax")
    _configure_jax_compilation_cache(jax)

    jnp = importlib.import_module("jax.numpy")
    mahjax = importlib.import_module("mahjax")
    contract = importlib.import_module("hydra_learner.mahjax.contract")
    obs_adapter = importlib.import_module("hydra_learner.mahjax.observation")
    safety_adapter = importlib.import_module("hydra_learner.mahjax.safety")

    rollout_device = jax.devices("gpu")[device_index]
    env = mahjax.make("red_mahjong", round_mode="single", observe_type="dict", next_round_style="auto")
    step_batch = jax.vmap(lambda state, action: env.step(state, action))
    init_batch_jit = jax.jit(jax.vmap(env.init), device=rollout_device)
    observe_batch = jax.vmap(env.observe)
    select_safety = jax.vmap(safety_adapter.select_observer_safety_jax)
    project_mask = jax.vmap(contract.project_mask_jax)
    map_action = jax.vmap(contract.map_hydra_action_jax)

    def observe_project(batch_state: Any, batch_safety: Any, alive: Any) -> tuple[Any, Any, Any]:
        obs = observe_batch(batch_state)
        observer_safety = select_safety(batch_safety, batch_state.current_player)
        hydra_obs = obs_adapter.mahjax_observation_to_hydra_batch_jax(obs, batch_state, observer_safety).obs
        hydra_mask = project_mask(batch_state.legal_action_mask, batch_state.round_state.last_draw)
        active = jnp.asarray(alive, dtype=jnp.bool_) & jnp.any(hydra_mask, axis=1)
        return hydra_obs, hydra_mask, active

    def step_update(
        batch_state: Any,
        batch_safety: Any,
        final_scores: Any,
        alive: Any,
        hydra_action: Any,
        active_mask: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        mapped_action = map_action(hydra_action, batch_state.legal_action_mask, batch_state.round_state.last_draw)
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
        mahjax_action = jnp.where(active, mapped_action, contract.inactive_action_jax(jnp))
        last_draw = batch_state.round_state.last_draw
        stepped = step_batch(batch_state, mahjax_action)
        done = stepped.terminated | stepped.truncated
        next_alive = jnp.asarray(alive, dtype=jnp.bool_) & jnp.logical_not(done)
        terminal_scores = stepped.round_state.score
        next_final_scores = jnp.where((done & active).reshape((env_count, 1)), terminal_scores, final_scores)
        updated_safety = safety_adapter.update_safety_bank_batch_for_action_jax(
            batch_safety, batch_state.current_player, mahjax_action, last_draw
        )
        next_safety = jax.tree_util.tree_map(
            lambda old, new: jnp.where(active.reshape((env_count,) + (1,) * (new.ndim - 1)), new, old),
            batch_safety,
            updated_safety,
        )
        return stepped, next_safety, next_final_scores, next_alive, done

    observe_project_jit = jax.jit(observe_project, device=rollout_device)
    step_update_jit = jax.jit(step_update, device=rollout_device)
    if use_aot:
        key_shape = jax.ShapeDtypeStruct((env_count, 2), jnp.uint32)
        state_shape = jax.eval_shape(init_batch_jit, key_shape)
        safety_shape = jax.eval_shape(lambda: safety_adapter.empty_safety_bank_batch_jax(env_count))
        scores_shape = jax.ShapeDtypeStruct((env_count, 4), jnp.int32)
        action_shape = jax.ShapeDtypeStruct((env_count,), jnp.int32)
        alive_shape = jax.ShapeDtypeStruct((env_count,), jnp.bool_)
        active_shape = jax.ShapeDtypeStruct((env_count,), jnp.bool_)
        init_run = init_batch_jit.lower(key_shape).compile()
        observe_project_run = observe_project_jit.lower(state_shape, safety_shape, alive_shape).compile()
        step_update_run = step_update_jit.lower(
            state_shape, safety_shape, scores_shape, alive_shape, action_shape, active_shape
        ).compile()
    else:
        init_run = init_batch_jit
        observe_project_run = observe_project_jit
        step_update_run = step_update_jit
    kernels = _MahjaxKernels(
        jax=jax,
        jnp=jnp,
        observe_project_run=observe_project_run,
        step_update_run=step_update_run,
        init_run=init_run,
        empty_safety_bank_batch_jax=safety_adapter.empty_safety_bank_batch_jax,
    )
    _KERNEL_CACHE[key] = kernels
    return kernels


def collect_mahjax_ppo_rollout(
    *,
    config: PpoControlConfig,
    model: HydraPolicyNet,
    seed: int,
    snapshot_metadata: PpoSnapshotMetadata,
) -> MahjaxPpoRolloutResult:
    if config.rollout_device is not None and config.rollout_device != config.device:
        raise ValueError("mahjax-gpu rollout uses the training CUDA device; separate rollout_device is unsupported")
    train_device = torch.device(config.device)
    if train_device.type != "cuda":
        raise ValueError("mahjax-gpu rollout requires CUDA training device")
    if config.games_per_update < 1:
        raise ValueError("games_per_update must be positive")

    device_index = train_device.index or 0
    env_count = config.games_per_update
    kernels = _get_mahjax_kernels(env_count, device_index)
    jax = kernels.jax
    jnp = kernels.jnp
    init_run = kernels.init_run
    observe_project_run = kernels.observe_project_run
    step_update_run = kernels.step_update_run
    empty_safety_bank_batch_jax = kernels.empty_safety_bank_batch_jax
    sync_interval = _completion_sync_interval()

    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    state = init_run(jax.random.split(init_key, env_count))
    sync_timing = _sync_timing_enabled()
    safety_bank = empty_safety_bank_batch_jax(env_count)
    final_scores_jax = jnp.zeros((env_count, 4), dtype=jnp.int32)
    alive_jax = jnp.ones((env_count,), dtype=jnp.bool_)
    full_action_torch = torch.empty((env_count,), dtype=torch.int32, device=train_device)
    completed_count: int = 0
    decisions_since_sync: int = 0
    row_parts: list[_RowPart] = []

    timing = {
        "mahjax_compile_warmup_ms": 0.0,
        "mahjax_observe_project_ms": 0.0,
        "mahjax_policy_ms": 0.0,
        "mahjax_action_map_step_ms": 0.0,
        "mahjax_finalize_ms": 0.0,
        "mahjax_aot_enabled": 1.0 if _use_jax_aot() else 0.0,
        "mahjax_completion_sync_interval": float(sync_interval),
    }
    # Per-phase rollout timers measure host enqueue time by default; setting
    # HYDRA_MAHJAX_SYNC_TIMING=1 makes them synchronize for bottleneck probes.
    compile_started = time.perf_counter()
    _warm_obs, warm_mask, warm_active = observe_project_run(state, safety_bank, alive_jax)
    warm_action = jnp.argmax(warm_mask, axis=1).astype(jnp.int32)
    _warm_next, _warm_safety, _warm_final_scores, _warm_alive, warm_done = step_update_run(
        state, safety_bank, final_scores_jax, alive_jax, warm_action, warm_active
    )
    warm_done.block_until_ready()
    timing["mahjax_compile_warmup_ms"] = (time.perf_counter() - compile_started) * 1000.0
    generator = torch.Generator(device=train_device)
    generator.manual_seed(seed)

    was_training = model.training
    model.eval()
    started = time.perf_counter()
    try:
        while completed_count < config.games_per_update:
            iter_started = time.perf_counter()
            hydra_obs_jax, hydra_mask_jax, active_jax = observe_project_run(state, safety_bank, alive_jax)
            if sync_timing:
                hydra_obs_jax.block_until_ready()
            timing["mahjax_observe_project_ms"] += (time.perf_counter() - iter_started) * 1000.0

            policy_started = time.perf_counter()
            obs_torch = torch_dlpack.from_dlpack(hydra_obs_jax).to(device=train_device, dtype=torch.float32)
            legal_mask_torch = torch_dlpack.from_dlpack(hydra_mask_jax).to(device=train_device, dtype=torch.bool)
            active_torch = torch_dlpack.from_dlpack(active_jax).to(device=train_device, dtype=torch.bool)
            active_slots = active_torch.nonzero(as_tuple=False).flatten()
            active_count = active_slots.shape[0]
            if active_count > 0:
                active_obs = obs_torch.index_select(0, active_slots)
                active_legal_mask = legal_mask_torch.index_select(0, active_slots)
                with torch.inference_mode():
                    logits, values = model.policy_value(active_obs)
                    masked_logits = logits.masked_fill(~active_legal_mask, -1.0e9)
                    if config.temperature != 1.0:
                        masked_logits = masked_logits / config.temperature
                    probs = torch.softmax(masked_logits, dim=1).masked_fill(~active_legal_mask, 0.0)
                    action_torch = (
                        torch.multinomial(probs, num_samples=1, replacement=True, generator=generator)
                        .squeeze(1)
                        .to(torch.int64)
                    )
                    old_logprob = torch.log(probs.gather(1, action_torch.unsqueeze(1)).squeeze(1).clamp_min(1.0e-12))
                    value_old = values.squeeze(1).detach().to(dtype=torch.float32)
                    logits = logits.detach().to(dtype=torch.float32)
                del masked_logits, probs, values
                full_action_torch.index_copy_(0, active_slots, action_torch.to(dtype=torch.int32))
                current_player_torch = torch_dlpack.from_dlpack(state.current_player).to(
                    device=train_device, dtype=torch.int64
                )
                step_count_torch = torch_dlpack.from_dlpack(state.step_count).to(device=train_device, dtype=torch.int64)
                active_current_player = current_player_torch.index_select(0, active_slots).detach()
                row_parts.append(
                    _RowPart(
                        obs=active_obs.detach(),
                        legal_mask=active_legal_mask.detach(),
                        action=action_torch.detach(),
                        old_logprob=old_logprob,
                        value_old=value_old,
                        logits=logits,
                        player_id=active_current_player,
                        game_id=active_slots.detach(),
                        turn=step_count_torch.index_select(0, active_slots).detach(),
                    )
                )
                del active_obs, active_legal_mask, action_torch, old_logprob, value_old, logits, active_current_player
                del obs_torch, legal_mask_torch, current_player_torch, step_count_torch
                del hydra_obs_jax, hydra_mask_jax
            else:
                del obs_torch, legal_mask_torch, hydra_obs_jax, hydra_mask_jax
            timing["mahjax_policy_ms"] += (time.perf_counter() - policy_started) * 1000.0

            step_started = time.perf_counter()
            action_jax = jax.dlpack.from_dlpack(full_action_torch)
            next_state, safety_bank, final_scores_jax, alive_jax, done_jax = step_update_run(
                state, safety_bank, final_scores_jax, alive_jax, action_jax, active_jax
            )
            if sync_timing:
                done_jax.block_until_ready()
            decisions_since_sync += 1
            if decisions_since_sync >= sync_interval:
                completed_count = env_count - int(jax.device_get(jnp.sum(alive_jax)))
                decisions_since_sync = 0
            state = next_state
            del action_jax, done_jax, next_state
            del active_jax, active_torch, active_slots
            timing["mahjax_action_map_step_ms"] += (time.perf_counter() - step_started) * 1000.0
    finally:
        model.train(was_training)

    finalize_started = time.perf_counter()
    final_scores_torch = torch_dlpack.from_dlpack(final_scores_jax).to(device=train_device, dtype=torch.int32)
    outcome_metrics = _final_score_metrics(final_scores_torch)
    del state, safety_bank, alive_jax, final_scores_jax, full_action_torch
    batch = _parts_to_batch(
        row_parts, final_scores=final_scores_torch, model=model, snapshot_metadata=snapshot_metadata
    )
    timing["mahjax_finalize_ms"] = (time.perf_counter() - finalize_started) * 1000.0
    timing["mahjax_total_ms"] = (time.perf_counter() - started) * 1000.0
    timing["mahjax_completed_games"] = float(config.games_per_update)
    timing["mahjax_rows_per_s"] = batch.obs.shape[0] / (timing["mahjax_total_ms"] / 1000.0)
    return MahjaxPpoRolloutResult(batch=batch, timing=timing, row_count=batch.obs.shape[0], metrics=outcome_metrics)


@dataclass(frozen=True)
class _RowPart:
    obs: torch.Tensor
    legal_mask: torch.Tensor
    action: torch.Tensor
    old_logprob: torch.Tensor
    value_old: torch.Tensor
    logits: torch.Tensor
    player_id: torch.Tensor
    game_id: torch.Tensor
    turn: torch.Tensor


def _parts_to_batch(
    parts: list[_RowPart],
    *,
    final_scores: torch.Tensor,
    model: HydraPolicyNet,
    snapshot_metadata: PpoSnapshotMetadata,
) -> PpoBatch:
    if not parts:
        raise ValueError("mahjax-gpu produced no PPO rows")
    device = next(model.parameters()).device
    obs_parts = [part.obs for part in parts]
    legal_mask_parts = [part.legal_mask for part in parts]
    action_parts = [part.action for part in parts]
    old_logprob_parts = [part.old_logprob for part in parts]
    value_old_parts = [part.value_old for part in parts]
    logits_parts = [part.logits for part in parts]
    player_id_parts = [part.player_id for part in parts]
    game_id_parts = [part.game_id for part in parts]
    turn_parts = [part.turn for part in parts]
    parts.clear()

    obs = torch.cat(obs_parts, dim=0).contiguous()
    del obs_parts
    legal_mask = torch.cat(legal_mask_parts, dim=0).contiguous()
    del legal_mask_parts
    actions = torch.cat(action_parts, dim=0).contiguous()
    del action_parts
    old_logprob = torch.cat(old_logprob_parts, dim=0).contiguous()
    del old_logprob_parts
    value_old = torch.cat(value_old_parts, dim=0).contiguous()
    del value_old_parts
    logits = torch.cat(logits_parts, dim=0).contiguous()
    del logits_parts
    player_id = torch.cat(player_id_parts, dim=0).contiguous()
    del player_id_parts
    game_id = torch.cat(game_id_parts, dim=0).contiguous()
    del game_id_parts
    turn = torch.cat(turn_parts, dim=0).contiguous()
    del turn_parts

    raw_advantages, returns = _gae_for_slots(
        player_id=player_id, value_old=value_old, game_id=game_id, final_scores=final_scores, device=device
    )
    legal_count = legal_mask.sum(dim=1).to(dtype=torch.int64)
    batch = PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=old_logprob,
        value_old=value_old,
        raw_advantages=raw_advantages,
        returns=returns,
        bc_logits=logits,
        legal_count=legal_count,
        player_id=player_id,
        seat_id=player_id,
        game_id=game_id,
        turn=turn,
        rank_utility_used="U_A",
        snapshot_metadata=snapshot_metadata.to_payload(),
    )
    batch.validate()
    return batch


def _final_score_metrics(final_scores: torch.Tensor) -> dict[str, float]:
    scores = final_scores.to(device="cpu", dtype=torch.float32)
    if scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError("mahjax final scores must have shape [games, 4]")
    games = scores.shape[0]
    if games < 1:
        raise ValueError("mahjax final score metrics require at least one game")
    ordered_players = torch.argsort(-scores, dim=1, stable=True)
    placements = torch.empty_like(ordered_players)
    rank_ids = torch.arange(4, dtype=ordered_players.dtype).expand_as(ordered_players)
    placements.scatter_(1, ordered_players, rank_ids)
    utility = torch.tensor(PLACEMENT_UTILITY_DEFAULT, dtype=torch.float32)
    rewards = utility[placements]
    score_mean = scores.mean(dim=0)
    reward_mean = rewards.mean(dim=0)
    placement_counts = torch.stack([(placements == rank).sum(dim=0) for rank in range(4)]).to(dtype=torch.float32)
    inv_games = 1.0 / float(games)
    metrics: dict[str, float] = {
        "episode_score_mean": float(scores.mean().item()),
        "episode_score_std": float(scores.std(unbiased=False).item()),
        "episode_reward_mean": float(rewards.mean().item()),
        "episode_reward_std": float(rewards.std(unbiased=False).item()),
    }
    for seat in range(4):
        metrics[f"seat{seat}_score_mean"] = float(score_mean[seat].item())
        metrics[f"seat{seat}_reward_mean"] = float(reward_mean[seat].item())
        metrics[f"seat{seat}_first_rate"] = float((placement_counts[0, seat] * inv_games).item())
        metrics[f"seat{seat}_last_rate"] = float((placement_counts[3, seat] * inv_games).item())
    return metrics


def _gae_for_slots(
    *,
    player_id: torch.Tensor,
    value_old: torch.Tensor,
    game_id: torch.Tensor,
    final_scores: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    del device
    target_device = value_old.device
    game_id = game_id.to(device=target_device, dtype=torch.int64)
    player_id = player_id.to(device=target_device, dtype=torch.int64)
    final_scores = final_scores.to(device=target_device, dtype=torch.int32)

    group_key = game_id * 4 + player_id
    order = torch.argsort(group_key, stable=True)
    sorted_key = group_key.index_select(0, order)
    sorted_value = value_old.index_select(0, order)
    unique_key, counts = torch.unique_consecutive(sorted_key, return_counts=True)
    group_count = unique_key.shape[0]
    max_count = int(counts.max().item())

    group_index = torch.repeat_interleave(torch.arange(group_count, device=target_device), counts)
    offsets = torch.cumsum(counts, dim=0) - counts
    position = torch.arange(sorted_key.shape[0], device=target_device) - torch.repeat_interleave(offsets, counts)

    values_dense = torch.zeros((group_count, max_count), dtype=value_old.dtype, device=target_device)
    values_dense[group_index, position] = sorted_value

    ordered_players = torch.argsort(-final_scores, dim=1, stable=True)
    placements = torch.empty_like(ordered_players)
    rank_ids = torch.arange(4, device=target_device, dtype=ordered_players.dtype).expand_as(ordered_players)
    placements.scatter_(1, ordered_players, rank_ids)

    utility = torch.tensor(PLACEMENT_UTILITY_DEFAULT, dtype=value_old.dtype, device=target_device)
    group_game = unique_key // 4
    group_player = unique_key % 4
    reward = utility[placements[group_game, group_player]]

    discount = DEFAULT_GAE_GAMMA * DEFAULT_GAE_LAMBDA
    running = torch.zeros(group_count, dtype=value_old.dtype, device=target_device)
    next_value = torch.zeros_like(running)
    has_next = torch.zeros(group_count, dtype=torch.bool, device=target_device)
    adv_dense = torch.zeros_like(values_dense)
    for pos in range(max_count - 1, -1, -1):
        valid = pos < counts
        value = values_dense[:, pos]
        delta = torch.where(
            has_next,
            (DEFAULT_GAE_GAMMA * next_value) - value,
            reward - value,
        )
        running = torch.where(has_next, delta + (discount * running), delta)
        running = torch.where(valid, running, torch.zeros_like(running))
        adv_dense[:, pos] = torch.where(valid, running, torch.zeros_like(running))
        next_value = torch.where(valid, value, next_value)
        has_next = has_next | valid

    sorted_advantages = adv_dense[group_index, position]
    sorted_returns = sorted_advantages + sorted_value
    raw_advantages = torch.empty_like(value_old)
    returns = torch.empty_like(value_old)
    raw_advantages.index_copy_(0, order, sorted_advantages)
    returns.index_copy_(0, order, sorted_returns)
    return raw_advantages, returns
