from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from hydra_learner.mahjax import safety

pytest.importorskip("jax")


def _jnp() -> Any:
    return importlib.import_module("jax.numpy")


def test_empty_safety_channels_shape_and_zero() -> None:
    encoded = np.asarray(safety.encode_safety_channels_jax(safety.empty_safety_state_jax()))

    assert encoded.shape == (23, 34)
    assert encoded.dtype == np.float32
    assert encoded.sum() == 0.0
    assert tuple(range(62, 85)) == safety.MAHJAX_EXACT_SAFETY_CHANNELS


def test_tenpai_hints_match_riichi_or_cached_prediction_threshold() -> None:
    state = safety.empty_safety_state_jax()
    state = safety.on_riichi_jax(state, 0)
    state = safety.set_tenpai_prediction_jax(state, 1, 0.5)
    state = safety.set_tenpai_prediction_jax(state, 2, 0.6)
    encoded = np.asarray(safety.encode_safety_channels_jax(state))

    assert encoded.shape == (23, 34)
    assert tuple(range(62, 85)) == safety.MAHJAX_EXACT_SAFETY_CHANNELS
    assert encoded[20].sum() == 34.0
    assert encoded[21].sum() == 0.0
    assert encoded[22].sum() == 34.0


def test_discard_sets_genbutsu_tedashi_suji_matagi_and_visible_counts() -> None:
    state = safety.empty_safety_state_jax()
    state = safety.on_discard_jax(state, 3, 0, True)  # 4m tedashi from kamicha.
    encoded = np.asarray(safety.encode_safety_channels_jax(state))

    assert encoded[0, 3] == 1.0  # genbutsu_all opp0
    assert encoded[3, 3] == 1.0  # genbutsu_tedashi opp0
    assert encoded[9, 0] == 1.0  # suji 1m from 4m
    assert encoded[9, 6] == 1.0  # suji 7m from 4m
    assert encoded[15, 2] == 1.0  # matagi 3m
    assert encoded[15, 4] == 1.0  # matagi 5m

    for _ in range(3):
        state = safety.on_discard_jax(state, 8, 1, False)
    encoded = np.asarray(safety.encode_safety_channels_jax(state))
    assert encoded[19, 8] == 1.0
    assert encoded[18, 8] == 0.0
    state = safety.on_discard_jax(state, 8, 1, False)
    encoded = np.asarray(safety.encode_safety_channels_jax(state))
    assert encoded[18, 8] == 1.0
    assert encoded[19, 8] == 0.0


def test_riichi_era_genbutsu_and_half_suji() -> None:
    state = safety.empty_safety_state_jax()
    state = safety.on_discard_jax(state, 0, 0, False)
    state = safety.on_riichi_jax(state, 0)
    state = safety.on_discard_jax(state, 6, 0, False)
    encoded = np.asarray(safety.encode_safety_channels_jax(state))

    assert encoded[6, 6] == 1.0  # genbutsu_riichi_era opp0 for post-riichi 7m
    assert encoded[9, 3] == 1.0  # center 4m both 1m/7m safe -> full suji
    assert encoded[12, 3] == 0.0

    state2 = safety.empty_safety_state_jax()
    state2 = safety.on_discard_jax(state2, 0, 0, False)
    encoded2 = np.asarray(safety.encode_safety_channels_jax(state2))
    assert encoded2[9, 3] == 0.5
    assert encoded2[12, 3] == 1.0


def test_call_updates_visible_counts_only() -> None:
    state = safety.empty_safety_state_jax()
    state = safety.on_call_jax(state, _jnp().asarray([13, 13, 13], dtype=_jnp().int32))
    encoded = np.asarray(safety.encode_safety_channels_jax(state))

    assert encoded[19, 13] == 1.0
    assert encoded[:18, 13].sum() == 0.0


def test_safety_bank_updates_all_observers_except_actor() -> None:
    bank = safety.empty_safety_bank_jax()
    bank = safety.update_safety_bank_for_action_jax(bank, 0, 3, -1)

    observer0 = safety.encode_safety_channels_jax(safety.select_observer_safety_jax(bank, 0))
    observer1 = safety.encode_safety_channels_jax(safety.select_observer_safety_jax(bank, 1))
    observer2 = safety.encode_safety_channels_jax(safety.select_observer_safety_jax(bank, 2))
    observer3 = safety.encode_safety_channels_jax(safety.select_observer_safety_jax(bank, 3))

    assert np.asarray(observer0).sum() == 0.0
    assert np.asarray(observer1)[2, 3] == 1.0
    assert np.asarray(observer2)[1, 3] == 1.0
    assert np.asarray(observer3)[0, 3] == 1.0


def test_safety_bank_uses_tsumogiri_last_draw_base_tile() -> None:
    bank = safety.empty_safety_bank_jax()
    bank = safety.update_safety_bank_for_action_jax(bank, 2, safety.MAHJAX_TSUMOGIRI, 34)
    observer0 = np.asarray(safety.encode_safety_channels_jax(safety.select_observer_safety_jax(bank, 0)))

    assert observer0[1, 4] == 1.0
    assert observer0[4, 4] == 0.0
