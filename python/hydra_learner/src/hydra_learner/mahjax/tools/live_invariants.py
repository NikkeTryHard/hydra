from __future__ import annotations

import argparse
import importlib
import json
from typing import Any

from hydra_learner.mahjax.compat import (
    HYDRA_ACTION_SPACE,
    MAHJAX_DUMMY,
    MAHJAX_RED_ACTION_SPACE,
    hydra_action_to_mahjax,
    mahjax_mask_to_hydra,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MahJAX live states against Hydra action adapter invariants.")
    parser.add_argument("--env", choices=("red_mahjong",), default="red_mahjong")
    parser.add_argument("--round-mode", choices=("single", "east", "half"), default="single")
    parser.add_argument("--next-round-style", choices=("auto",), default="auto")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _check_state_row(row_mask: list[object], last_draw: int) -> int:
    last_draw_arg = last_draw if 0 <= last_draw <= 36 else None
    hydra_mask = mahjax_mask_to_hydra(row_mask, last_draw=last_draw_arg)
    if len(hydra_mask) != HYDRA_ACTION_SPACE:
        raise AssertionError("Hydra mask projection changed width")
    if not any(hydra_mask):
        raise AssertionError("Nonterminal MahJAX state projected to empty Hydra legal mask")
    if bool(row_mask[MAHJAX_DUMMY]):
        projected_without_dummy = row_mask.copy()
        projected_without_dummy[MAHJAX_DUMMY] = False
        if mahjax_mask_to_hydra(projected_without_dummy, last_draw=last_draw_arg) != hydra_mask:
            raise AssertionError("MahJAX DUMMY affected Hydra policy mask projection")

    checked_hydra_actions = 0
    for hydra_action in range(HYDRA_ACTION_SPACE):  # pyrefly: ignore[non-convergent-recursion] - fixed ABI loop.
        if not hydra_mask[hydra_action]:
            continue
        mahjax_action = hydra_action_to_mahjax(hydra_action, legal_mask=row_mask, last_draw=last_draw_arg)
        if not bool(row_mask[mahjax_action]):
            raise AssertionError(f"Hydra action {hydra_action} reversed to illegal MahJAX action {mahjax_action}")
        checked_hydra_actions += 1
    return checked_hydra_actions


def _check_state_batch(*, masks: Any, last_draws: Any, terminals: Any) -> tuple[int, int]:
    np = importlib.import_module("numpy")
    masks_np: Any = np.asarray(masks, dtype=bool)
    last_draws_np: Any = np.asarray(last_draws, dtype=int)
    terminals_np: Any = np.asarray(terminals, dtype=bool)
    if masks_np.ndim != 2 or masks_np.shape[1] != MAHJAX_RED_ACTION_SPACE:
        raise ValueError(f"MahJAX mask batch must have shape [N,{MAHJAX_RED_ACTION_SPACE}], got {masks_np.shape}")
    if last_draws_np.shape[0] != masks_np.shape[0] or terminals_np.shape[0] != masks_np.shape[0]:
        raise ValueError("MahJAX state batch fields disagree on batch width")

    checked_states = 0
    checked_hydra_actions = 0
    for idx in range(int(masks_np.shape[0])):  # pyrefly: ignore[non-convergent-recursion] - numpy row loop.
        terminal = bool(terminals_np[idx])
        if terminal:
            continue
        row_mask = list(masks_np[idx].tolist())
        checked_hydra_actions += _check_state_row(row_mask, int(last_draws_np[idx]))
        checked_states += 1
    return checked_states, checked_hydra_actions


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    _validate_positive("--num-envs", args.num_envs)
    _validate_positive("--num-steps", args.num_steps)

    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    mahjax = importlib.import_module("mahjax")
    auto_reset = importlib.import_module("mahjax.wrappers.auto_reset_wrapper").auto_reset

    env = mahjax.make(
        args.env,
        round_mode=args.round_mode,
        observe_type="dict",
        next_round_style=args.next_round_style,
    )
    step_env = auto_reset(env.step, env.init)
    init_batch = jax.jit(jax.vmap(env.init))
    step_batch = jax.jit(jax.vmap(step_env))

    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    state = init_batch(jax.random.split(init_key, args.num_envs))
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, state)

    checked_states = 0
    checked_hydra_actions = 0
    for _ in range(args.num_steps):
        terminal = state.terminated | state.truncated
        state_count, action_count = _check_state_batch(
            masks=state.legal_action_mask,
            last_draws=state.round_state.last_draw,
            terminals=terminal,
        )
        checked_states += state_count
        checked_hydra_actions += action_count
        actions = jnp.argmax(state.legal_action_mask, axis=-1).astype(jnp.int32)
        rng, step_key = jax.random.split(rng)
        state = step_batch(state, actions, jax.random.split(step_key, args.num_envs))
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, state)

    return {
        "env": args.env,
        "round_mode": args.round_mode,
        "next_round_style": args.next_round_style,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "checked_states": checked_states,
        "checked_hydra_actions": checked_hydra_actions,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
    }


def main(argv: list[str] | None = None) -> None:
    result = run_check(parse_args(argv))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
