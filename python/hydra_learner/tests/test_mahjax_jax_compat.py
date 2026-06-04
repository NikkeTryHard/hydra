from __future__ import annotations

import importlib
from typing import Any

import pytest

from hydra_learner.mahjax import compat as cpu_compat
from hydra_learner.mahjax import jax_compat

pytest.importorskip("jax")


def _jnp() -> Any:
    return importlib.import_module("jax.numpy")


def test_jax_mask_projection_matches_cpu_adapter_for_collapsed_actions() -> None:
    jnp = _jnp()
    mask = [False] * cpu_compat.MAHJAX_RED_ACTION_SPACE
    for action in [
        0,
        cpu_compat.MAHJAX_TSUMOGIRI,
        cpu_compat.MAHJAX_TSUMO,
        cpu_compat.MAHJAX_RON,
        cpu_compat.MAHJAX_PON_RED,
        cpu_compat.MAHJAX_CHI_RIGHT_RED,
        cpu_compat.MAHJAX_OPEN_KAN,
        cpu_compat.MAHJAX_DUMMY,
    ]:
        mask[action] = True

    expected = cpu_compat.mahjax_mask_to_hydra(mask, last_draw=cpu_compat.HYDRA_AKA_5P)
    actual = jax_compat.mahjax_mask_to_hydra_jax(jnp.asarray(mask), cpu_compat.HYDRA_AKA_5P).tolist()

    assert actual == expected


def test_jax_batch_projection_matches_single_projection() -> None:
    jnp = _jnp()
    masks = [[False] * cpu_compat.MAHJAX_RED_ACTION_SPACE for _ in range(2)]
    masks[0][1] = True
    masks[0][cpu_compat.MAHJAX_TSUMOGIRI] = True
    masks[1][cpu_compat.MAHJAX_SELF_KAN_START + 4] = True
    masks[1][cpu_compat.MAHJAX_PASS] = True

    actual = jax_compat.mahjax_mask_to_hydra_batch_jax(jnp.asarray(masks), jnp.asarray([34, -1])).tolist()

    assert actual[0] == cpu_compat.mahjax_mask_to_hydra(masks[0], last_draw=34)
    assert actual[1] == cpu_compat.mahjax_mask_to_hydra(masks[1], last_draw=None)


def test_jax_lowest_projected_action_round_trips_to_legal_mahjax_action() -> None:
    jnp = _jnp()
    mask = [False] * cpu_compat.MAHJAX_RED_ACTION_SPACE
    mask[cpu_compat.MAHJAX_TSUMOGIRI] = True
    mask[cpu_compat.HYDRA_AKA_5S] = True
    mask[cpu_compat.MAHJAX_RIICHI] = True

    hydra_mask = jax_compat.mahjax_mask_to_hydra_jax(jnp.asarray(mask), cpu_compat.HYDRA_AKA_5S)
    hydra_action = int(jax_compat.choose_lowest_legal_hydra_action_jax(hydra_mask).tolist())
    mahjax_action = int(
        jax_compat.hydra_action_to_mahjax_jax(hydra_action, jnp.asarray(mask), cpu_compat.HYDRA_AKA_5S).tolist()
    )

    assert hydra_action == cpu_compat.HYDRA_AKA_5S
    assert mahjax_action == cpu_compat.MAHJAX_TSUMOGIRI
    assert mask[mahjax_action]


def test_jax_projected_choice_prefers_open_kan_and_tsumo() -> None:
    jnp = _jnp()
    mask = [False] * cpu_compat.MAHJAX_RED_ACTION_SPACE
    mask[cpu_compat.MAHJAX_SELF_KAN_START + 2] = True
    mask[cpu_compat.MAHJAX_OPEN_KAN] = True
    mask[cpu_compat.MAHJAX_TSUMO] = True
    mask[cpu_compat.MAHJAX_RON] = True

    # KAN is lower than AGARI in Hydra's compact action order, then open-kan is preferred for collapsed KAN.
    chosen = int(jax_compat.choose_lowest_projected_mahjax_action_jax(jnp.asarray(mask), -1).tolist())

    assert chosen == cpu_compat.MAHJAX_OPEN_KAN
