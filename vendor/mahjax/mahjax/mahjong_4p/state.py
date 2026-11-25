from __future__ import annotations

from types import ModuleType
from typing import ClassVar

import jax.numpy as jnp

from mahjax._src.struct import dataclass
from mahjax.common.constants import (
    MAX_HAND_TILES,
    MAX_MELDS_PER_PLAYER,
    NUM_TILE_TYPES,
    NUM_TILE_TYPES_WITH_RED,
    SENTINEL_TILE_ID,
)
from mahjax.common.meld import EMPTY_MELD
from mahjax.common.tile import EMPTY_RIVER

from . import constants as _constants
from .constants import (
    DEAD_WALL_TILES,
    FIRST_DRAW_IDX,
    LEGAL_ACTION_SIZE,
    MAX_DISCARDS_PER_PLAYER,
    MAX_DORA_INDICATORS,
    NUM_PHYSICAL_TILES,
    NUM_PLAYERS,
)

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class PlayerStateArrays:
    hand: jnp.ndarray = jnp.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=jnp.int8)
    hand_with_red: jnp.ndarray = jnp.zeros((NUM_PLAYERS, NUM_TILE_TYPES_WITH_RED), dtype=jnp.int8)
    # Reserved for the hand's tile ids (0-135). Nothing fills it yet, so every
    # slot is SENTINEL_TILE_ID; read the hand from ``hand_with_red``.
    hand_ids: jnp.ndarray = jnp.full((NUM_PLAYERS, MAX_HAND_TILES), SENTINEL_TILE_ID, dtype=jnp.int16)
    legal_action_mask: jnp.ndarray = jnp.zeros((NUM_PLAYERS, LEGAL_ACTION_SIZE), dtype=jnp.bool_)
    can_win: jnp.ndarray = jnp.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=jnp.bool_)
    has_yaku: jnp.ndarray = jnp.zeros((NUM_PLAYERS, 2), dtype=jnp.bool_)
    fan: jnp.ndarray = jnp.zeros((NUM_PLAYERS, 2), dtype=jnp.int32)
    fu: jnp.ndarray = jnp.zeros((NUM_PLAYERS, 2), dtype=jnp.int32)
    melds: jnp.ndarray = jnp.full((NUM_PLAYERS, MAX_MELDS_PER_PLAYER), EMPTY_MELD, dtype=jnp.uint16)
    meld_counts: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8)
    river: jnp.ndarray = jnp.full((NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER), EMPTY_RIVER, dtype=jnp.uint16)
    discard_counts: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8)
    riichi: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    riichi_declared: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    double_riichi: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    ippatsu: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    furiten_by_discard: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    furiten_by_pass: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    is_hand_concealed: jnp.ndarray = jnp.ones((NUM_PLAYERS,), dtype=jnp.bool_)
    pon: jnp.ndarray = jnp.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=jnp.int32)
    has_won: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    n_kan: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8)
    has_nagashi_mangan: jnp.ndarray = jnp.ones((NUM_PLAYERS,), dtype=jnp.bool_)


@dataclass
class RoundState:
    action_history: jnp.ndarray = jnp.full((3, 200), -1, dtype=jnp.int8)
    # Write cursor into ``action_history``. It lives on RoundState so that it is
    # reset to 0 together with the buffer at every round boundary. Do NOT index
    # ``action_history`` with ``EnvState.step_count``: that counter is
    # hanchan-global (``_start_next_round_or_end_game`` carries it across rounds),
    # so it walks past the end of a per-round buffer and JAX silently drops the
    # out-of-bounds scatter, leaving the history all -1.
    round_step: jnp.ndarray = jnp.int32(0)
    # Set when a round produces more actions than ``action_history`` can hold,
    # so that truncation is observable instead of silent.
    history_overflow: jnp.ndarray = FALSE
    round: jnp.ndarray = jnp.int8(0)
    round_limit: jnp.ndarray = jnp.int8(7)
    terminated_round: jnp.ndarray = FALSE
    # int32: with no renchan limit, int8 would wrap from 127 honba.
    honba: jnp.ndarray = jnp.int32(0)
    kyotaku: jnp.ndarray = jnp.int8(0)
    init_wind: jnp.ndarray = jnp.array([0, 1, 2, 3], dtype=jnp.int8)
    seat_wind: jnp.ndarray = jnp.array([0, 1, 2, 3], dtype=jnp.int8)
    dealer: jnp.ndarray = jnp.int8(0)
    order_points: jnp.ndarray = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
    score: jnp.ndarray = jnp.full((NUM_PLAYERS,), 250, dtype=jnp.int32)
    deck: jnp.ndarray = jnp.zeros((NUM_PHYSICAL_TILES,), dtype=jnp.int8)
    next_deck_ix: jnp.ndarray = jnp.int32(FIRST_DRAW_IDX)
    last_deck_ix: jnp.ndarray = jnp.int8(DEAD_WALL_TILES)
    draw_next: jnp.ndarray = FALSE
    last_draw: jnp.ndarray = jnp.int8(-1)
    last_player: jnp.ndarray = jnp.int8(-1)
    dora_indicators: jnp.ndarray = jnp.full((MAX_DORA_INDICATORS,), -1, dtype=jnp.int8)
    ura_dora_indicators: jnp.ndarray = jnp.full((MAX_DORA_INDICATORS,), -1, dtype=jnp.int8)
    is_abortive_draw_normal: jnp.ndarray = FALSE
    dummy_count: jnp.ndarray = jnp.int8(0)
    is_haitei: jnp.ndarray = FALSE
    target: jnp.ndarray = jnp.int8(-1)
    n_kan_doras: jnp.ndarray = jnp.int8(0)
    kan_declared: jnp.ndarray = FALSE
    can_after_kan: jnp.ndarray = FALSE
    can_robbing_kan: jnp.ndarray = FALSE


@dataclass
class EnvState:
    # The table this state is played at (players, wall, payments, ...). The rules read
    # it from here, so the same rules play any table; it is not part of the pytree.
    constants: ClassVar[ModuleType] = _constants

    current_player: jnp.ndarray = jnp.int8(0)
    legal_action_mask: jnp.ndarray = jnp.zeros((LEGAL_ACTION_SIZE,), dtype=jnp.bool_)
    players: PlayerStateArrays = PlayerStateArrays()
    round_state: RoundState = RoundState()
    step_count: jnp.ndarray = jnp.int32(0)
    rewards: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32)
    # The rons declared on the current discard and what they pay, held back until
    # everyone who can ron it has answered; see ``_ron``. Not public until then.
    pending_rewards: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32)
    pending_winners: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE

    @property
    def env_id(self) -> str:
        # The state does not record its preset; this only picks the renderer.
        return "red_4p"


State = EnvState


def default_state() -> EnvState:
    return EnvState()
