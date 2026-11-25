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

"""Players for the 3-player table.

The rule-based player plays the 4-player one's heuristics (``mahjong_4p/players.py``)
less chi, which this table does not have, and sets every north aside it can.
"""

import jax
import jax.numpy as jnp

from mahjax._src.types import Array, PRNGKey
from mahjax.common.hand import Hand
from mahjax.common.meld import Meld
from mahjax.common.shanten import Shanten
from mahjax.common.tile import River, Tile
from mahjax.common.yaku import Yaku

from .action import Action
from .state import State

# Discard first the tiles with the highest number: honours, then terminals.
# 2m-8m are not in the wall, so their entries never count.
PRIORITY_MASK = jnp.array(
    [5, 4, 3, 2, 1, 2, 3, 4, 5]
    + [5, 4, 3, 2, 1, 2, 3, 4, 5]
    + [5, 4, 3, 2, 1, 2, 3, 4, 5]
    + [6, 7, 7, 7, 6, 6, 6]
)

OUTSIDE_MASK = jnp.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 1]
    + [1, 0, 0, 0, 0, 0, 0, 0, 1]
    + [1, 0, 0, 0, 0, 0, 0, 0, 1]
    + [1, 1, 1, 1, 1, 1, 1]
)

# East and the three dragons.
YAKU_TILES = jnp.array([27, 31, 32, 33])

# Hyperparameters for Pon
BASIC_PON_PROB = 0.04
YAKU_PON_PROB = 0.7
YAKU_MELD_PON_PROB = 0.6
HAS_PUNG_PON_PROB = 0.05
# Hyperparameters for Open Kan
OPEN_KAN_PROB = 0.05
# Hyperparameters for Riichi
RIICHI_PROB = 0.9


def has_river_tile(hand: Array, tile: Array) -> Array:
    tile_type = Tile.to_tile_type(tile)
    return jnp.where(
        tile == -1,
        jnp.zeros(34, dtype=jnp.int32),
        jnp.zeros(34, dtype=jnp.int32).at[tile_type].set(hand[tile_type]),
    )


def _discard_action_from_tile_type(unflatten_hand: Array, tile_type: Array) -> Array:
    tile_type = jnp.int32(tile_type)
    has_black = unflatten_hand[tile_type] > 0
    has_red = Tile.is_tile_type_five(tile_type) & (unflatten_hand[Tile.to_red(tile_type)] > 0)
    return jnp.where(has_black, tile_type, jnp.where(has_red, Tile.to_red(tile_type), tile_type))


def _discard_logic(state: State, unflatten_hand: Array, flatten_hand: Array) -> Array:
    """
    - Discard the tile that minimizes the shanten number.
    - Prioritize the outside tiles when discarding.
    - When listening, discard the tile that makes the wait wider.
    - If another player is in riichi and the hand is two or more from ready, discard a tile they have discarded.
    """
    c_p = state.current_player
    n_meld = state.players.meld_counts[c_p]
    # Shanten
    detailed_shantens = Shanten.detailed_discard(flatten_hand)
    best_shanten_normal = jnp.min(detailed_shantens[:, 0])
    best_shanten_7_pairs = jnp.min(detailed_shantens[:, 1])
    best_shanten_13_orphan = jnp.min(detailed_shantens[:, 2])

    use_normal = jnp.logical_or(best_shanten_normal < best_shanten_7_pairs, best_shanten_7_pairs <= 3)
    best_shanten = jnp.where(use_normal, best_shanten_normal, best_shanten_7_pairs)
    shantens = jnp.where(use_normal, detailed_shantens[:, 0], detailed_shantens[:, 1])
    use_orphans = best_shanten >= best_shanten_13_orphan + 2
    best_shanten = jnp.where(use_orphans, best_shanten_13_orphan, best_shanten)
    shantens = jnp.where(use_orphans, detailed_shantens[:, 2], shantens)
    best_shanten = jnp.where(n_meld > 0, best_shanten_normal, best_shanten)
    shantens = jnp.where(n_meld > 0, detailed_shantens[:, 0], shantens)
    # Score only the tiles in hand, by their 34-type counts.
    priority_mask = (shantens == best_shanten) * PRIORITY_MASK * (flatten_hand > 0)
    best_shanten_action = jnp.argmax(priority_mask)
    # Waiting
    is_tempai = best_shanten == 0

    def f(i):
        i = jnp.int32(i)
        return jax.lax.cond(
            jnp.logical_or(flatten_hand[i] == 0, shantens[i] != 0),
            lambda: jnp.int32(-1),
            lambda: jax.vmap(Hand.can_ron, in_axes=(None, 0))(
                flatten_hand.at[i].set(flatten_hand[i] - 1),
                jnp.arange(Tile.NUM_TILE_TYPE),
            )
            .astype(jnp.int32)
            .sum(),
        )

    can_rons = jax.vmap(f)(jnp.arange(Tile.NUM_TILE_TYPE))  # (34,)
    action = jnp.where(is_tempai, jnp.argmax(can_rons), best_shanten_action)
    # Defense, against one riichi player
    other_riichi = jnp.asarray(state.players.riichi).at[c_p].set(False)
    riichi_player_river = River.decode_tile(state.players.river[jnp.argmax(other_riichi)])
    hand_within_river = jax.vmap(has_river_tile, in_axes=(None, 0))(flatten_hand, riichi_player_river).sum(
        axis=-1
    )  # (34,)
    # With no tile they have discarded, keep advancing the hand.
    defense_action = jnp.where(hand_within_river.sum() == 0, action, jnp.argmax(hand_within_river))
    action = jnp.where(other_riichi.any() & (best_shanten >= 2), defense_action, action)
    return _discard_action_from_tile_type(unflatten_hand, action)


def _is_equal(a: Array, arr: Array) -> Array:
    return (a == arr).any()


def _pon_logic(state: State, unflatten_hand: Array, flatten_hand: Array, rng: PRNGKey) -> Array:
    """
    - Pon the yaku tile with 70% probability.
    - Other tiles,
        - If the yaku tile is melded, pon with 60% probability
    - If the tile is a pung, pon with 5% probability
    """
    target_tile = Tile.to_tile_type(state.round_state.target)
    seat_wind = 27 + state.round_state.seat_wind[state.current_player]
    is_yaku = (target_tile == YAKU_TILES).any() | (target_tile == seat_wind)

    meld_tile = Meld.target(state.players.melds[state.current_player])
    is_yaku_meld = (jax.vmap(_is_equal, in_axes=(0, None))(meld_tile, YAKU_TILES)).any()
    is_yaku_meld = is_yaku_meld | (meld_tile == seat_wind).any()

    has_pung = flatten_hand[target_tile] >= 3

    basic_prob = (flatten_hand * ~OUTSIDE_MASK).sum() * BASIC_PON_PROB
    prob = jnp.where(is_yaku, YAKU_PON_PROB, basic_prob)
    prob = jnp.where(is_yaku_meld, YAKU_MELD_PON_PROB, prob)
    prob = jnp.where(has_pung, HAS_PUNG_PON_PROB, prob)
    do_pon = jax.random.bernoulli(rng, prob)
    # Only PON_RED is legal when the pon has to take the red five.
    pon_action = jnp.where(state.legal_action_mask[Action.PON], Action.PON, Action.PON_RED)
    return jnp.where(do_pon, pon_action, Action.PASS)


def _open_kan_logic(state: State, unflatten_hand: Array, flatten_hand: Array, rng: PRNGKey) -> Array:
    """
    - Open kan with 5% probability
    """
    do_open_kan = jax.random.bernoulli(rng, OPEN_KAN_PROB)
    return jnp.where(do_open_kan, Action.OPEN_KAN, Action.PASS)


def _riichi_logic(state: State, current_action: Array, rng: PRNGKey) -> Array:
    do_riichi = jax.random.bernoulli(rng, RIICHI_PROB)
    return jnp.where(do_riichi, Action.RIICHI, current_action)


def rule_based_player(state: State, rng: PRNGKey) -> Array:
    unflatten_hand = state.players.hand_with_red[state.current_player]
    melds = state.players.melds[state.current_player]
    n_meld = state.players.meld_counts[state.current_player]
    legal_action_mask = state.legal_action_mask
    flatten_hand = Yaku.flatten(unflatten_hand, melds, n_meld)
    # discard logic
    discard_action = _discard_logic(state, unflatten_hand, flatten_hand)
    discard_action = jnp.where(discard_action == state.round_state.last_draw, Action.TSUMOGIRI, discard_action)

    is_legal = legal_action_mask[discard_action]
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    random_action = jax.random.categorical(rng, logits=logits)
    action = jnp.where(is_legal, discard_action, random_action)

    # Melded tiles
    is_pon = legal_action_mask[Action.PON] | legal_action_mask[Action.PON_RED]
    is_open_kan = legal_action_mask[Action.OPEN_KAN]
    rng, pon_rng, open_kan_rng = jax.random.split(rng, 3)
    action = jnp.where(is_pon, _pon_logic(state, unflatten_hand, flatten_hand, pon_rng), action)
    action = jnp.where(is_open_kan, _open_kan_logic(state, unflatten_hand, flatten_hand, open_kan_rng), action)

    # Riichi
    is_riichi = legal_action_mask[Action.RIICHI]
    rng, riichi_rng = jax.random.split(rng)
    action = jnp.where(is_riichi, _riichi_logic(state, action, riichi_rng), action)

    # If the player can riichi, riichi
    action = jnp.where(legal_action_mask[Action.RIICHI], Action.RIICHI, action)
    # If the player can set a north aside, do: it is a dora and a free draw, and
    # riichi can still follow it.
    action = jnp.where(legal_action_mask[Action.KITA], Action.KITA, action)
    # If the player can tsumo, tsumo
    action = jnp.where(legal_action_mask[Action.TSUMO], Action.TSUMO, action)
    # If the player can ron, ron
    action = jnp.where(legal_action_mask[Action.RON], Action.RON, action)
    return action


def random_player(state: State, rng: PRNGKey) -> Array:
    legal_action_mask = state.legal_action_mask
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits=logits)
