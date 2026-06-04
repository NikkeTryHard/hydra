from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

UInt8Array = NDArray[np.uint8]

from hydra_learner.mahjax.compat import HYDRA_PASS, MAHJAX_DUMMY, MAHJAX_PASS, mahjax_action_to_hydra
from hydra_learner.mahjax.constructor import mahjax_action_from_mjai_event
from hydra_learner.mahjax.jax_compat import hydra_action_to_mahjax_jax, mahjax_mask_to_hydra_jax

if TYPE_CHECKING:
    from hydra_learner.typing_boundaries import JaxArray, JaxModule, MahjaxState, MahjaxStepFn


def response_phase(state: MahjaxState) -> bool:
    return int(state.round_state.target) >= 0


def current_actor(state: MahjaxState) -> int:
    return int(state.current_player)


def inactive_action() -> int:
    return MAHJAX_DUMMY


def inactive_action_jax(jnp: JaxModule) -> JaxArray:
    return jnp.asarray(MAHJAX_DUMMY, dtype=jnp.int32)


def project_mask_jax(mask87: JaxArray, last_draw_value: JaxArray) -> JaxArray:
    return mahjax_mask_to_hydra_jax(mask87, last_draw_value)


def map_hydra_action_jax(action46: JaxArray, mask87: JaxArray, last_draw_value: JaxArray) -> JaxArray:
    return hydra_action_to_mahjax_jax(action46, mask87, last_draw_value)


def last_draw(state: MahjaxState) -> int:
    return int(state.round_state.last_draw)


def projected_hydra_mask(state: MahjaxState) -> UInt8Array:
    return np.asarray(mahjax_mask_to_hydra_jax(state.legal_action_mask, state.round_state.last_draw), dtype=np.uint8)


def projected_response_hydra_mask(state: MahjaxState, actor: int) -> UInt8Array:
    mask = np.asarray(
        mahjax_mask_to_hydra_jax(state.players.legal_action_mask[actor], state.round_state.last_draw), dtype=np.uint8
    )
    if mask.any():
        mask[HYDRA_PASS] = 1
    return mask


def hydra_mask_for_actor(state: MahjaxState, actor: int) -> UInt8Array:
    if response_phase(state):
        return projected_response_hydra_mask(state, actor)
    if current_actor(state) != actor:
        raise ValueError(f"actor {actor} does not match MahJAX current_player {current_actor(state)}")
    return projected_hydra_mask(state)


def apply_mahjax_action(step_fn: MahjaxStepFn, state: MahjaxState, action: int) -> MahjaxState:
    jnp = importlib.import_module("jax.numpy")
    return step_fn(state, jnp.asarray(action, dtype=jnp.int32))


def advance_response_to_actor(step_fn: MahjaxStepFn, state: MahjaxState, actor: int) -> MahjaxState:
    while response_phase(state) and current_actor(state) != actor:
        if not bool(state.legal_action_mask[MAHJAX_PASS]):
            raise ValueError(f"actor {actor} is not reachable from current_player {current_actor(state)}")
        state = apply_mahjax_action(step_fn, state, MAHJAX_PASS)
    return state


def apply_all_response_passes(step_fn: MahjaxStepFn, state: MahjaxState) -> MahjaxState:
    while bool(state.legal_action_mask[MAHJAX_PASS]):
        state = apply_mahjax_action(step_fn, state, MAHJAX_PASS)
    return state


def apply_mjai_event_action(step_fn: MahjaxStepFn, state: MahjaxState, event: Mapping[str, object]) -> MahjaxState:
    action = mahjax_action_from_mjai_event(state, event)
    if action is None:
        return state
    return apply_mahjax_action(step_fn, state, action)


def projected_hydra_action_for_mjai_event(state: MahjaxState, event: Mapping[str, object]) -> int | None:
    action = mahjax_action_from_mjai_event(state, event)
    if action is None:
        return None
    return mahjax_action_to_hydra(action, last_draw=last_draw(state))
