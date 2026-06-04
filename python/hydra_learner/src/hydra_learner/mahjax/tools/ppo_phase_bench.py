from __future__ import annotations

# Diagnostic-only JAX-to-Torch phase benchmark; uses synthetic PPO targets.
import argparse
import importlib
import json
import time
from typing import Any

import numpy as np
import torch

from hydra_learner.model import HydraPolicyNet
from hydra_learner.ppo.rl import EntropyController, masked_log_prob
from hydra_learner.ppo.step import PpoBatch, PpoTrainStepConfig, ppo_train_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MahJAX GPU rollout plus Torch PPO train phase gap.")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=96)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--rollout-chunk-steps", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--bottleneck", type=int, default=96)
    parser.add_argument("--residual-profile", default="mish_se")
    parser.add_argument("--backbone-profile", default="conv2d_local3")
    parser.add_argument("--conv-memory-format", default="contiguous")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch-size", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _tree_block_until_ready(jax: Any, tree: Any) -> None:
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def main() -> None:
    args = parse_args()
    if args.num_envs < 1 or args.num_steps < 1:
        raise ValueError("num-envs and num-steps must be positive")

    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    mahjax = importlib.import_module("mahjax")
    auto_reset = importlib.import_module("mahjax.wrappers.auto_reset_wrapper").auto_reset
    jax_compat = importlib.import_module("hydra_learner.mahjax.jax_compat")
    obs_adapter = importlib.import_module("hydra_learner.mahjax.observation")
    safety_adapter = importlib.import_module("hydra_learner.mahjax.safety")

    env = mahjax.make("red_mahjong", round_mode="single", observe_type="dict", next_round_style="auto")
    step_env = auto_reset(env.step, env.init)
    init_batch = jax.jit(jax.vmap(env.init))
    observe_batch = jax.vmap(env.observe)
    step_batch = jax.vmap(step_env)

    def rollout(state: Any, safety_bank: Any, keys: Any) -> tuple[tuple[Any, Any], dict[str, Any]]:
        def body(carry: tuple[Any, Any], key: Any) -> tuple[tuple[Any, Any], dict[str, Any]]:
            carry_state, carry_safety = carry
            obs = observe_batch(carry_state)
            observer_safety = jax.vmap(safety_adapter.select_observer_safety_jax)(
                carry_safety, carry_state.current_player
            )
            hydra_obs = obs_adapter.mahjax_observation_to_hydra_batch_jax(obs, carry_state, observer_safety).obs
            hydra_mask = jax.vmap(jax_compat.mahjax_mask_to_hydra_jax)(
                carry_state.legal_action_mask, carry_state.round_state.last_draw
            )
            hydra_action = jnp.argmax(hydra_mask, axis=-1).astype(jnp.int32)
            action = jax.vmap(jax_compat.hydra_action_to_mahjax_jax)(
                hydra_action, carry_state.legal_action_mask, carry_state.round_state.last_draw
            )
            last_draw = carry_state.round_state.last_draw
            next_state = step_batch(carry_state, action, jax.random.split(key, args.num_envs))
            next_safety = safety_adapter.update_safety_bank_batch_for_action_jax(
                carry_safety, carry_state.current_player, action, last_draw
            )
            metrics = {
                "obs": hydra_obs,
                "legal_mask": hydra_mask,
                "actions": hydra_action,
                "player_id": carry_state.current_player,
                "turn": carry_state.step_count,
            }
            return (next_state, next_safety), metrics

        return jax.lax.scan(body, (state, safety_bank), keys, length=keys.shape[0])

    jit_rollout = jax.jit(rollout)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    state = init_batch(jax.random.split(init_key, args.num_envs))
    safety_bank = safety_adapter.empty_safety_bank_batch_jax(args.num_envs)
    rollout_chunk_steps = args.rollout_chunk_steps or args.num_steps
    if rollout_chunk_steps < 1:
        raise ValueError("rollout-chunk-steps must be positive")
    _tree_block_until_ready(jax, (state, safety_bank))
    for _ in range(args.warmup_runs):
        rng, key = jax.random.split(rng)
        (state, safety_bank), warm_metrics = jit_rollout(state, safety_bank, jax.random.split(key, rollout_chunk_steps))
        _tree_block_until_ready(jax, warm_metrics)

    obs_chunks = []
    legal_chunks = []
    actions_chunks = []
    player_chunks = []
    turn_chunks = []
    rollout_ms = 0.0
    export_ms = 0.0
    remaining_steps = args.num_steps
    while remaining_steps > 0:
        chunk_steps = min(rollout_chunk_steps, remaining_steps)
        rng, key = jax.random.split(rng)
        chunk_started = time.perf_counter()
        (state, safety_bank), metrics = jit_rollout(state, safety_bank, jax.random.split(key, chunk_steps))
        _tree_block_until_ready(jax, metrics)
        rollout_ms += (time.perf_counter() - chunk_started) * 1000.0

        transfer_started = time.perf_counter()
        chunk_rows = args.num_envs * chunk_steps
        obs_chunks.append(jax.device_get(metrics["obs"]).reshape(chunk_rows, 192, 34).copy())
        legal_chunks.append(jax.device_get(metrics["legal_mask"]).reshape(chunk_rows, 46).copy())
        actions_chunks.append(jax.device_get(metrics["actions"]).reshape(chunk_rows).copy())
        player_chunks.append(jax.device_get(metrics["player_id"]).reshape(chunk_rows).copy())
        turn_chunks.append(jax.device_get(metrics["turn"]).reshape(chunk_rows).copy())
        export_ms += (time.perf_counter() - transfer_started) * 1000.0
        remaining_steps -= chunk_steps

    obs_np = np.concatenate(obs_chunks, axis=0)
    legal_np = np.concatenate(legal_chunks, axis=0)
    actions_np = np.concatenate(actions_chunks, axis=0)
    player_np = np.concatenate(player_chunks, axis=0)
    turn_np = np.concatenate(turn_chunks, axis=0)

    device = torch.device(args.device)
    model = HydraPolicyNet(
        hidden=args.hidden,
        blocks=args.blocks,
        bottleneck=args.bottleneck,
        residual_profile=args.residual_profile,
        backbone_profile=args.backbone_profile,
        conv_memory_format=args.conv_memory_format,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-5, fused=True)

    batch_started = time.perf_counter()
    obs = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32)
    legal_mask = torch.from_numpy(legal_np).to(device=device, dtype=torch.bool)
    actions = torch.from_numpy(actions_np).to(device=device, dtype=torch.int64)
    legal_count = legal_mask.sum(dim=1).to(dtype=torch.int64)
    old_logits_chunks = []
    value_chunks = []
    old_logprob_chunks = []
    with torch.inference_mode():
        model.eval()
        for start in range(0, obs.shape[0], args.microbatch_size):
            end = min(start + args.microbatch_size, obs.shape[0])
            logits_chunk, values_chunk = model.policy_value(obs[start:end])
            value_chunk = values_chunk.squeeze(1).detach().to(dtype=torch.float32)
            old_logits_chunks.append(logits_chunk.detach().to(dtype=torch.float32))
            value_chunks.append(value_chunk)
            old_logprob_chunks.append(
                masked_log_prob(logits_chunk, legal_mask[start:end], actions[start:end])
                .detach()
                .to(dtype=torch.float32)
            )
    logits = torch.cat(old_logits_chunks, dim=0)
    old_logprob = torch.cat(old_logprob_chunks, dim=0)
    value_old = torch.cat(value_chunks, dim=0)
    del old_logits_chunks, old_logprob_chunks, value_chunks
    torch.cuda.empty_cache()
    raw_advantages = torch.zeros_like(value_old)
    returns = value_old.detach().clone()
    batch = PpoBatch(
        obs=obs,
        actions=actions,
        legal_mask=legal_mask,
        old_logprob=old_logprob,
        value_old=value_old,
        raw_advantages=raw_advantages,
        returns=returns,
        bc_logits=logits.detach().to(dtype=torch.float32),
        legal_count=legal_count,
        player_id=torch.from_numpy(player_np).to(device=device, dtype=torch.int64),
        seat_id=torch.from_numpy(player_np).to(device=device, dtype=torch.int64),
        game_id=torch.arange(args.num_envs, dtype=torch.int64).repeat(args.num_steps).to(device),
        turn=torch.from_numpy(turn_np).to(device=device, dtype=torch.int64),
        rank_utility_used="U_A",
    )
    batch.validate()
    torch.cuda.synchronize(device)
    batch_build_ms = (time.perf_counter() - batch_started) * 1000.0

    train_started = time.perf_counter()
    result = ppo_train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        entropy_controller=EntropyController(alpha=1.0e-3, beta=1.0e-2, alpha_max=0.05),
        config=PpoTrainStepConfig(
            grad_clip_norm=args.grad_clip_norm,
            microbatch_size=args.microbatch_size,
            epochs=args.epochs,
            target_kl=None,
            bc_kl_reverse_coef=0.0,
        ),
    )
    torch.cuda.synchronize(device)
    train_ms = (time.perf_counter() - train_started) * 1000.0
    rows = args.num_envs * args.num_steps
    print(
        json.dumps(
            {
                "backend": "mahjax_gpu_probe",
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(device_) for device_ in jax.devices()],
                "rows": rows,
                "num_envs": args.num_envs,
                "num_steps": args.num_steps,
                "rollout_ms": rollout_ms,
                "jax_to_torch_batch_build_ms": batch_build_ms,
                "jax_device_get_ms": export_ms,
                "rollout_chunk_steps": rollout_chunk_steps,
                "rollout_plus_train_ms": rollout_ms + batch_build_ms + train_ms,
                "rollout_export_batch_ms": rollout_ms + export_ms + batch_build_ms,
                "rollout_rows_per_s": rows / (rollout_ms / 1000.0),
                "train_step_ms": train_ms,
                "train_rows_per_s": rows / (train_ms / 1000.0),
                "supported_channels": int(
                    jnp.sum(obs_adapter.supported_channel_mask_jax(include_state=True, include_safety=True)).tolist()
                ),
                "ppo_forward_backward_ms": result.metrics["forward_backward_ms"],
                "ppo_optimizer_ms": result.metrics["optimizer_ms"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
