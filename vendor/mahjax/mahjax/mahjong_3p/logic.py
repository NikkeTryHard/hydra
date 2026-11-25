# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The 3-player game's own action, kita (北抜き), on top of the shared rules in ``mahjax.common.logic``."""

from typing import Literal, Optional

import jax
import jax.numpy as jnp

from mahjax._src.types import Array, PRNGKey
from mahjax.common.config import GameConfig
from mahjax.common.constants import FALSE, TRUE
from mahjax.common.hand import Hand
from mahjax.common.logic import (
    CURRENT_TILE,
    State,
    _advance_to_next_decision,
    _append_action_history,
    _apply_action,
    _draw_replacement,
    _is_furiten,
    _next_ron_player,
    _replace_state,
    _replacements_drawn,
    _resolve_game_config,
    _reveal_pending_kan_dora,
    _rinshan_ix,
    _set_player_hand,
    _yaku_lookahead,
)

from .action import Action
from .constants import NORTH


def _step(
    state: State,
    action: Array,
    key: PRNGKey,
    game_config: Optional[GameConfig] = None,
    round_end: Literal["auto", "confirm"] = "auto",
) -> State:
    """``mahjax.common.logic._step`` with KITA: the shared actions' state and the kita's
    are both computed, then ``action``'s is kept."""
    state = _append_action_history(state, action)
    is_kita = action == Action.KITA
    shared_state = _apply_action(state, jnp.where(is_kita, 0, action), key, game_config, round_end)
    kita_state = _kita(state, game_config)
    state = jax.lax.cond(is_kita, lambda: kita_state, lambda: shared_state)
    state = jax.lax.cond(
        state.round_state.kita_declared
        & ~state.round_state.is_abortive_draw_normal
        & ~state.players.legal_action_mask[:, Action.RON].any(),
        lambda: _draw_kita_replacement(state, game_config),
        lambda: state,
    )
    return _advance_to_next_decision(state, game_config)


def _kita(state: State, game_config: Optional[GameConfig] = None) -> State:
    """
    Set a north aside (北抜き)
    - The north counts as a dora for the player, who then draws a replacement tile
    - First the others may ron the north as if it were discarded, without chankan
    - Like a call it interrupts the first go-around and, once nobody rons the north, ends every ippatsu
    """
    config = _resolve_game_config(game_config)
    c_p = state.current_player
    state = _set_player_hand(state, c_p, Hand.sub(state.players.hand_with_red[c_p], NORTH))
    state = _replace_state(state, kita=state.players.kita.at[c_p].add(1))
    # The replacement for this kita, which is already counted.
    replacement_tile = state.round_state.deck[_rinshan_ix(state, _replacements_drawn(state) - 1)]
    # As for a kan: a ron on the north is scored with an earlier kan's indicator still
    # face down, and the replacement with it turned over (``_draw_kita_replacement``).
    face_down = state.round_state
    revealed = _reveal_pending_kan_dora(state, game_config)
    has_yaku, fan, fu = _yaku_lookahead(
        revealed,
        NORTH,
        replacement_tile,
        ron_dora_indicators=face_down.dora_indicators,
        ron_ura_dora_indicators=face_down.ura_dora_indicators,
        use_red_fives=config.use_red_fives,
    )
    state = _replace_state(state, has_yaku=has_yaku, fan=fan, fu=fu)
    others = jnp.arange(state.constants.NUM_PLAYERS) != c_p
    completes = state.players.can_win[:, NORTH] & others
    can_ron = completes & state.players.has_yaku[:, CURRENT_TILE] & ~_is_furiten(state)
    # A win the north completes without a yaku is not offered, but letting it go is furiten.
    state = _replace_state(state, furiten_by_pass=state.players.furiten_by_pass | (completes & ~can_ron))
    ron_mask = state.constants.ZERO_MASK_2D.at[:, Action.RON].set(can_ron)
    next_ron_player, can_any_ron = _next_ron_player(ron_mask, c_p)
    return jax.lax.cond(
        can_any_ron,
        lambda: _replace_state(
            state,
            target=jnp.int8(NORTH),
            last_player=c_p,
            current_player=jnp.int8(next_ron_player),
            legal_action_mask=ron_mask.at[next_ron_player, Action.PASS].set(TRUE),
            kita_declared=TRUE,
            draw_next=FALSE,
        ),
        lambda: _replace_state(
            state,
            target=jnp.int8(-1),
            last_player=c_p,
            legal_action_mask=state.constants.ZERO_MASK_2D,
            kita_declared=TRUE,
            draw_next=FALSE,
        ),
    )


def _draw_kita_replacement(state: State, game_config: Optional[GameConfig] = None) -> State:
    """Nobody ronned the north: the kita's player draws its replacement, which moves the
    haitei line like a kan's and can win by rinshan kaihou."""
    # An added or open kan's indicator still face down turns over first, as before a kan's rinshan draw.
    state = _reveal_pending_kan_dora(state, game_config)
    replacement_tile = state.round_state.deck[_rinshan_ix(state, _replacements_drawn(state) - 1)]
    state = _replace_state(
        state,
        ippatsu=jnp.zeros_like(state.players.ippatsu),  # like a call, once the ron window has closed
        can_after_kan=TRUE,
        kita_declared=FALSE,
        last_deck_ix=state.round_state.last_deck_ix + 1,
    )
    return _draw_replacement(state, replacement_tile, game_config)
