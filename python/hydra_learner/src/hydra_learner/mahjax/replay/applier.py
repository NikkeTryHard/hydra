from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from hydra_learner.mahjax.constructor import mjai_tile_to_mahjax_id

MAHJAX_TSUMOGIRI = 71
MAHJAX_RIICHI = 72
MAHJAX_TSUMO = 73
MAHJAX_RON = 74
MAHJAX_PASS = 84


def _jnp() -> Any:
    return importlib.import_module("jax.numpy")


def _jax() -> Any:
    return importlib.import_module("jax")


def _env() -> Any:
    return importlib.import_module("mahjax").make("red_mahjong", observe_type="dict")


def _base_tile(tile_id: int) -> int:
    if tile_id == 34:
        return 4
    if tile_id == 35:
        return 13
    if tile_id == 36:
        return 22
    return tile_id


def mjai_dahai_to_mahjax_action(state: Any, event: Mapping[str, Any]) -> int:
    actor = int(event["actor"])
    if actor != int(state.current_player):
        raise ValueError(f"dahai actor {actor} does not match current_player {int(state.current_player)}")
    tile_id = mjai_tile_to_mahjax_id(str(event["pai"]))
    if bool(event.get("tsumogiri", False)):
        if _base_tile(tile_id) != _base_tile(int(state.round_state.last_draw)):
            raise ValueError("tsumogiri tile does not match MahJAX last_draw")
        return MAHJAX_TSUMOGIRI
    return tile_id


def mjai_reach_to_mahjax_action(state: Any, event: Mapping[str, Any]) -> int:
    actor = int(event["actor"])
    if actor != int(state.current_player):
        raise ValueError(f"reach actor {actor} does not match current_player {int(state.current_player)}")
    return MAHJAX_RIICHI


def mjai_hora_to_mahjax_action(state: Any, event: Mapping[str, Any]) -> int:
    actor = int(event["actor"])
    if actor != int(state.current_player):
        raise ValueError(f"hora actor {actor} does not match current_player {int(state.current_player)}")
    if event.get("target") is None:
        return MAHJAX_TSUMO
    return MAHJAX_RON


def mjai_none_to_mahjax_action(state: Any, event: Mapping[str, Any]) -> int:
    actor = int(event.get("actor", state.current_player))
    if actor != int(state.current_player):
        raise ValueError(f"none actor {actor} does not match current_player {int(state.current_player)}")
    return MAHJAX_PASS


def apply_mjai_event_slice(state: Any, event: Mapping[str, Any], *, seed: int = 0) -> Any:
    event_type = str(event["type"])
    if event_type == "dahai":
        action = mjai_dahai_to_mahjax_action(state, event)
    elif event_type == "reach":
        action = mjai_reach_to_mahjax_action(state, event)
    elif event_type == "hora":
        action = mjai_hora_to_mahjax_action(state, event)
    elif event_type in {"none", "skip"}:
        action = mjai_none_to_mahjax_action(state, event)
    else:
        raise NotImplementedError(f"MahJAX replay applier slice does not support event type {event_type!r}")
    action_arr = _jnp().asarray(action, dtype=_jnp().int32)
    return _env().step(state, action_arr, _jax().random.PRNGKey(seed))


def apply_mjai_events_slice(state: Any, events: list[Mapping[str, Any]], *, seed: int = 0) -> Any:
    out = state
    for offset, event in enumerate(events):
        out = apply_mjai_event_slice(out, event, seed=seed + offset)
    return out
