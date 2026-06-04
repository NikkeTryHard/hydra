from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from hydra_learner.mahjax.compat import (
    HYDRA_ACTION_SPACE,
    HYDRA_AGARI,
    HYDRA_CHI_LEFT,
    HYDRA_CHI_MID,
    HYDRA_CHI_RIGHT,
    HYDRA_KAN,
    HYDRA_PASS,
    HYDRA_PON,
    HYDRA_RIICHI,
    HYDRA_RYUUKYOKU,
    MAHJAX_CHI_LEFT,
    MAHJAX_CHI_LEFT_RED,
    MAHJAX_CHI_MID,
    MAHJAX_CHI_MID_RED,
    MAHJAX_CHI_RIGHT,
    MAHJAX_CHI_RIGHT_RED,
    MAHJAX_DUMMY,
    MAHJAX_KYUUSHU,
    MAHJAX_OPEN_KAN,
    MAHJAX_PASS,
    MAHJAX_PON,
    MAHJAX_PON_RED,
    MAHJAX_RIICHI,
    MAHJAX_RON,
    MAHJAX_SELF_KAN_END,
    MAHJAX_SELF_KAN_START,
    MAHJAX_TSUMO,
    MAHJAX_TSUMOGIRI,
)

if TYPE_CHECKING:
    from hydra_learner.typing_boundaries import JaxArray, JaxModule


def _jnp() -> JaxModule:
    return importlib.import_module("jax.numpy")


def mahjax_mask_to_hydra_jax(mask87: JaxArray, last_draw: JaxArray) -> JaxArray:
    """Project one MahJAX red legal mask to Hydra 46 entirely in JAX."""
    jnp = _jnp()
    mask = jnp.asarray(mask87, dtype=jnp.bool_)
    last_draw_i = jnp.asarray(last_draw, dtype=jnp.int32)
    tsumogiri_legal = mask[MAHJAX_TSUMOGIRI] & (last_draw_i >= 0) & (last_draw_i <= 36)
    tsumogiri_one_hot = jnp.arange(HYDRA_ACTION_SPACE, dtype=jnp.int32) == last_draw_i

    hydra = jnp.zeros((HYDRA_ACTION_SPACE,), dtype=jnp.bool_)
    hydra = hydra.at[:37].set(mask[:37])
    hydra = hydra | (tsumogiri_legal & tsumogiri_one_hot)
    hydra = hydra.at[HYDRA_RIICHI].set(mask[MAHJAX_RIICHI])
    hydra = hydra.at[HYDRA_CHI_LEFT].set(mask[MAHJAX_CHI_LEFT] | mask[MAHJAX_CHI_LEFT_RED])
    hydra = hydra.at[HYDRA_CHI_MID].set(mask[MAHJAX_CHI_MID] | mask[MAHJAX_CHI_MID_RED])
    hydra = hydra.at[HYDRA_CHI_RIGHT].set(mask[MAHJAX_CHI_RIGHT] | mask[MAHJAX_CHI_RIGHT_RED])
    hydra = hydra.at[HYDRA_PON].set(mask[MAHJAX_PON] | mask[MAHJAX_PON_RED])
    hydra = hydra.at[HYDRA_KAN].set(
        jnp.any(mask[MAHJAX_SELF_KAN_START : MAHJAX_SELF_KAN_END + 1]) | mask[MAHJAX_OPEN_KAN]
    )
    hydra = hydra.at[HYDRA_AGARI].set(mask[MAHJAX_TSUMO] | mask[MAHJAX_RON])
    hydra = hydra.at[HYDRA_RYUUKYOKU].set(mask[MAHJAX_KYUUSHU])
    return hydra.at[HYDRA_PASS].set(mask[MAHJAX_PASS])


def choose_lowest_legal_hydra_action_jax(hydra_mask46: JaxArray) -> JaxArray:
    """Return lowest legal Hydra action id from a 46-wide JAX bool mask."""
    jnp = _jnp()
    action_ids = jnp.arange(HYDRA_ACTION_SPACE, dtype=jnp.int32)
    scores = jnp.where(jnp.asarray(hydra_mask46, dtype=jnp.bool_), -action_ids, -HYDRA_ACTION_SPACE)
    return jnp.argmax(scores).astype(jnp.int32)


def hydra_action_to_mahjax_jax(action46: JaxArray, mask87: JaxArray, last_draw: JaxArray) -> JaxArray:
    """Choose a legal MahJAX action for a Hydra action inside JAX.

    This deterministic perf helper mirrors the common rollout path: lowest-id legal
    Hydra action, non-red call variants preferred, open kan before self-kan, tsumo
    before ron. Context-rich red-call and explicit kan-tile selection remain in the
    Python adapter until policy rollout needs them.
    """
    jnp = _jnp()
    mask = jnp.asarray(mask87, dtype=jnp.bool_)
    action = jnp.asarray(action46, dtype=jnp.int32)
    last_draw_i = jnp.asarray(last_draw, dtype=jnp.int32)

    self_kan_mask = mask[MAHJAX_SELF_KAN_START : MAHJAX_SELF_KAN_END + 1]
    first_self_kan = MAHJAX_SELF_KAN_START + jnp.argmax(self_kan_mask.astype(jnp.int32))
    kan_choice = jnp.where(mask[MAHJAX_OPEN_KAN], MAHJAX_OPEN_KAN, first_self_kan)
    discard_choice = jnp.where(
        (last_draw_i == action) & mask[MAHJAX_TSUMOGIRI],
        MAHJAX_TSUMOGIRI,
        action,
    )

    return jnp.select(
        [
            action <= 36,
            action == HYDRA_RIICHI,
            action == HYDRA_CHI_LEFT,
            action == HYDRA_CHI_MID,
            action == HYDRA_CHI_RIGHT,
            action == HYDRA_PON,
            action == HYDRA_KAN,
            action == HYDRA_AGARI,
            action == HYDRA_RYUUKYOKU,
            action == HYDRA_PASS,
        ],
        [
            discard_choice,
            MAHJAX_RIICHI,
            jnp.where(mask[MAHJAX_CHI_LEFT], MAHJAX_CHI_LEFT, MAHJAX_CHI_LEFT_RED),
            jnp.where(mask[MAHJAX_CHI_MID], MAHJAX_CHI_MID, MAHJAX_CHI_MID_RED),
            jnp.where(mask[MAHJAX_CHI_RIGHT], MAHJAX_CHI_RIGHT, MAHJAX_CHI_RIGHT_RED),
            jnp.where(mask[MAHJAX_PON], MAHJAX_PON, MAHJAX_PON_RED),
            kan_choice,
            jnp.where(mask[MAHJAX_TSUMO], MAHJAX_TSUMO, MAHJAX_RON),
            MAHJAX_KYUUSHU,
            MAHJAX_PASS,
        ],
        default=MAHJAX_DUMMY,
    ).astype(jnp.int32)


def choose_lowest_projected_mahjax_action_jax(mask87: JaxArray, last_draw: JaxArray) -> JaxArray:
    hydra_mask = mahjax_mask_to_hydra_jax(mask87, last_draw)
    hydra_action = choose_lowest_legal_hydra_action_jax(hydra_mask)
    return hydra_action_to_mahjax_jax(hydra_action, mask87, last_draw)


def mahjax_mask_to_hydra_batch_jax(mask87: JaxArray, last_draw: JaxArray) -> JaxArray:
    jax = importlib.import_module("jax")
    return jax.vmap(mahjax_mask_to_hydra_jax)(mask87, last_draw)


def choose_lowest_projected_mahjax_action_batch_jax(mask87: JaxArray, last_draw: JaxArray) -> JaxArray:
    jax = importlib.import_module("jax")
    return jax.vmap(choose_lowest_projected_mahjax_action_jax)(mask87, last_draw)
