from __future__ import annotations

import pytest

from hydra_learner.mahjax.constructor import mahjax_state_from_start_kyoku
from hydra_learner.mahjax.replay.applier import (
    apply_mjai_event_slice,
    mjai_dahai_to_mahjax_action,
    mjai_hora_to_mahjax_action,
    mjai_none_to_mahjax_action,
)

pytest.importorskip("jax")

START_TEHAIS = [
    ["1m", "2m", "3m", "4m", "5m", "5mr", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
    ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N"],
    ["P", "F", "C", "1m", "1m", "2m", "2m", "3m", "3m", "4m", "4m", "5m", "5m"],
    ["6p", "6p", "7p", "7p", "8p", "8p", "9p", "9p", "1s", "1s", "2s", "2s", "3s"],
]


def _state() -> object:
    return mahjax_state_from_start_kyoku(
        tehais=START_TEHAIS,
        scores=[25000, 26000, 24000, 20000],
        dora_marker="1m",
        oya=0,
        kyoku=1,
        honba=0,
        kyotaku=0,
        first_draw="5sr",
    )


def test_dahai_event_maps_red_tedashi_to_matching_action() -> None:
    state = _state()

    assert mjai_dahai_to_mahjax_action(state, {"type": "dahai", "actor": 0, "pai": "5sr", "tsumogiri": False}) == 36


def test_dahai_event_maps_tsumogiri_to_tsumogiri_action() -> None:
    state = _state()

    assert mjai_dahai_to_mahjax_action(state, {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True}) == 71


def test_dahai_replay_applier_advances_current_player() -> None:
    state = _state()
    next_state = apply_mjai_event_slice(state, {"type": "dahai", "actor": 0, "pai": "5sr", "tsumogiri": False})

    assert int(next_state.current_player) != 0
    assert int(next_state.round_state.last_player) == 0
    assert bool(next_state.legal_action_mask.any())


def test_dahai_actor_mismatch_fails_closed() -> None:
    state = _state()

    with pytest.raises(ValueError, match="does not match current_player"):
        mjai_dahai_to_mahjax_action(state, {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False})


def test_hora_action_maps_tsumo_and_ron() -> None:
    state = _state()

    assert mjai_hora_to_mahjax_action(state, {"type": "hora", "actor": 0, "target": None}) == 73
    assert mjai_hora_to_mahjax_action(state, {"type": "hora", "actor": 0, "target": 1}) == 74


def test_none_action_maps_pass_and_checks_actor() -> None:
    state = _state()

    assert mjai_none_to_mahjax_action(state, {"type": "none", "actor": 0}) == 84
    with pytest.raises(ValueError, match="does not match current_player"):
        mjai_none_to_mahjax_action(state, {"type": "none", "actor": 1})
