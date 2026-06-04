from __future__ import annotations

# Diagnostic-only MahJAX throughput probe; not a production rollout backend.
import argparse
import importlib
import json
import time
from typing import Any


def _json_default(value: object) -> object:
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return str(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe MahJAX vectorized simulator throughput.")
    parser.add_argument("--env", choices=("red_mahjong", "no_red_mahjong"), default="red_mahjong")
    parser.add_argument("--round-mode", choices=("single", "east", "half"), default="single")
    parser.add_argument("--next-round-style", choices=("auto", "dummy_share"), default="auto")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--project-hydra", action="store_true")
    parser.add_argument("--adapt-hydra-obs", action="store_true")
    parser.add_argument("--adapt-hydra-safety", action="store_true")
    parser.add_argument("--dummy-policy", action="store_true")
    parser.add_argument("--dummy-policy-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    return parser.parse_args(argv)


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    _validate_positive("--num-envs", args.num_envs)
    _validate_positive("--num-steps", args.num_steps)
    _validate_positive("--warmup-runs", args.warmup_runs)

    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    mahjax = importlib.import_module("mahjax")
    auto_reset = importlib.import_module("mahjax.wrappers.auto_reset_wrapper").auto_reset
    jax_compat = importlib.import_module("hydra_learner.mahjax.jax_compat")
    obs_adapter = importlib.import_module("hydra_learner.mahjax.observation")
    safety_adapter = importlib.import_module("hydra_learner.mahjax.safety")

    env = mahjax.make(
        args.env,
        round_mode=args.round_mode,
        observe_type="dict",
        next_round_style=args.next_round_style,
    )
    step_env = auto_reset(env.step, env.init)
    init_batch = jax.jit(jax.vmap(env.init))
    step_batch = jax.vmap(step_env)
    observe_batch = jax.vmap(env.observe)
    policy_dtype = jnp.bfloat16 if args.dummy_policy_dtype == "bfloat16" else jnp.float32

    def choose_projected_action(state: Any) -> Any:
        return jax.vmap(jax_compat.choose_lowest_projected_mahjax_action_jax)(
            state.legal_action_mask, state.round_state.last_draw
        )

    def choose_lowest_action(state: Any) -> Any:
        return jnp.argmax(state.legal_action_mask, axis=-1).astype(jnp.int32)

    def masked_argmax_hydra_action(hydra_mask: Any, logits: Any) -> Any:
        masked = jnp.where(hydra_mask, logits, jnp.asarray(-1.0e30, dtype=logits.dtype))
        return jnp.argmax(masked, axis=-1).astype(jnp.int32)

    def rollout(
        state: Any,
        safety_bank: Any,
        keys: Any,
        policy_weights: Any,
        observe: bool,
        project_hydra: bool,
        adapt_hydra_obs: bool,
        adapt_hydra_safety: bool,
        dummy_policy: bool,
    ) -> tuple[tuple[Any, Any], dict[str, Any]]:
        def body(carry: tuple[Any, Any], key: Any) -> tuple[tuple[Any, Any], dict[str, Any]]:
            carry_state, carry_safety = carry
            needs_obs = observe or adapt_hydra_obs or dummy_policy
            if needs_obs:
                obs = observe_batch(carry_state)
                obs_checksum = sum(
                    jnp.sum(jnp.asarray(leaf, dtype=jnp.float32)) for leaf in jax.tree_util.tree_leaves(obs)
                )
            else:
                obs = None
                obs_checksum = jnp.asarray(0.0, dtype=jnp.float32)

            if adapt_hydra_obs or dummy_policy:
                observer_safety = jax.vmap(safety_adapter.select_observer_safety_jax)(
                    carry_safety, carry_state.current_player
                )
                hydra_obs = obs_adapter.mahjax_observation_to_hydra_batch_jax(
                    obs,
                    carry_state,
                    observer_safety if adapt_hydra_safety else None,
                ).obs
                adapter_obs_checksum = jnp.sum(hydra_obs, dtype=jnp.float32)
            else:
                hydra_obs = None
                adapter_obs_checksum = jnp.asarray(0.0, dtype=jnp.float32)

            if dummy_policy:
                assert hydra_obs is not None
                hydra_mask = jax.vmap(jax_compat.mahjax_mask_to_hydra_jax)(
                    carry_state.legal_action_mask, carry_state.round_state.last_draw
                )
                flat_obs = hydra_obs.reshape((hydra_obs.shape[0], -1)).astype(policy_dtype)
                logits = (flat_obs @ policy_weights).astype(jnp.float32)
                hydra_action = masked_argmax_hydra_action(hydra_mask, logits)
                action = jax.vmap(jax_compat.hydra_action_to_mahjax_jax)(
                    hydra_action, carry_state.legal_action_mask, carry_state.round_state.last_draw
                )
            elif project_hydra:
                action = choose_projected_action(carry_state)
            else:
                action = choose_lowest_action(carry_state)

            last_draw = carry_state.round_state.last_draw
            next_state = step_batch(carry_state, action, jax.random.split(key, args.num_envs))
            next_safety = jax.lax.cond(
                adapt_hydra_safety,
                lambda: safety_adapter.update_safety_bank_batch_for_action_jax(
                    carry_safety, carry_state.current_player, action, last_draw
                ),
                lambda: carry_safety,
            )
            done = next_state.terminated | next_state.truncated
            metrics = {
                "action_sum": jnp.sum(action, dtype=jnp.int32),
                "adapter_obs_checksum": adapter_obs_checksum,
                "done_count": jnp.sum(done, dtype=jnp.int32),
                "obs_checksum": obs_checksum,
                "reward_sum": jnp.sum(next_state.rewards, dtype=jnp.float32),
            }
            return (next_state, next_safety), metrics

        return jax.lax.scan(body, (state, safety_bank), keys, length=keys.shape[0])

    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    state = init_batch(jax.random.split(init_key, args.num_envs))
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, state)
    safety_bank = safety_adapter.empty_safety_bank_batch_jax(args.num_envs)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, safety_bank)

    rng, policy_key = jax.random.split(rng)
    policy_weights = jax.random.normal(
        policy_key,
        (obs_adapter.HYDRA_OBS_CHANNELS * obs_adapter.HYDRA_TILE_WIDTH, jax_compat.HYDRA_ACTION_SPACE),
        dtype=policy_dtype,
    )
    policy_weights = policy_weights * jnp.asarray(0.01, dtype=policy_dtype)
    policy_weights.block_until_ready()

    jit_rollout = jax.jit(
        rollout,
        static_argnames=("observe", "project_hydra", "adapt_hydra_obs", "adapt_hydra_safety", "dummy_policy"),
    )
    for _ in range(args.warmup_runs):
        rng, warm_key = jax.random.split(rng)
        warm_keys = jax.random.split(warm_key, args.num_steps)
        (state, safety_bank), warm_metrics = jit_rollout(
            state,
            safety_bank,
            warm_keys,
            policy_weights,
            args.observe,
            args.project_hydra,
            args.adapt_hydra_obs,
            args.adapt_hydra_safety,
            args.dummy_policy,
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, warm_metrics)

    rng, bench_key = jax.random.split(rng)
    bench_keys = jax.random.split(bench_key, args.num_steps)
    started = time.perf_counter()
    (state, safety_bank), metrics = jit_rollout(
        state,
        safety_bank,
        bench_keys,
        policy_weights,
        args.observe,
        args.project_hydra,
        args.adapt_hydra_obs,
        args.adapt_hydra_safety,
        args.dummy_policy,
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, metrics)
    elapsed = time.perf_counter() - started

    total_steps = args.num_envs * args.num_steps
    supported = obs_adapter.supported_channel_mask_jax(
        include_state=args.adapt_hydra_obs or args.dummy_policy,
        include_safety=args.adapt_hydra_safety,
    )
    return {
        "action_sum": int(jnp.sum(metrics["action_sum"]).tolist()),
        "adapt_hydra_obs": args.adapt_hydra_obs,
        "adapt_hydra_safety": args.adapt_hydra_safety,
        "adapter_obs_checksum": float(jnp.sum(metrics["adapter_obs_checksum"]).tolist()),
        "adapter_supported_channels": int(jnp.sum(supported).tolist()),
        "done_count": int(jnp.sum(metrics["done_count"]).tolist()),
        "dummy_policy": args.dummy_policy,
        "dummy_policy_dtype": args.dummy_policy_dtype,
        "elapsed_s": elapsed,
        "env": args.env,
        "env_steps_per_s": total_steps / elapsed if elapsed > 0.0 else 0.0,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "next_round_style": args.next_round_style,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "obs_checksum": float(jnp.sum(metrics["obs_checksum"]).tolist()),
        "observe": args.observe,
        "project_hydra": args.project_hydra,
        "reward_sum": float(jnp.sum(metrics["reward_sum"]).tolist()),
        "round_mode": args.round_mode,
        "total_env_steps": total_steps,
        "warmup_runs": args.warmup_runs,
    }


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run_probe(parse_args(argv)), sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
