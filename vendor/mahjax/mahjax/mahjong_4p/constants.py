from __future__ import annotations

from typing import Final

import jax.numpy as jnp

from mahjax.common.action import Action
from mahjax.common.constants import FALSE, INITIAL_HAND_TILES, TRUE  # noqa: F401  (re-exported)

NUM_PLAYERS: Final[int] = 4
NUM_PHYSICAL_TILES: Final[int] = 136
# The tile ids the wall is shuffled from: all 136.
TILE_IDS = jnp.arange(NUM_PHYSICAL_TILES, dtype=jnp.int32)

HAS_CHI: Final[bool] = True
HAS_KITA: Final[bool] = False
# Special abortive draws that need four players.
HAS_FOUR_WIND_DRAW: Final[bool] = True
HAS_FOUR_RIICHI_DRAW: Final[bool] = True

MAX_DISCARDS_PER_PLAYER: Final[int] = 24
MAX_DORA_INDICATORS: Final[int] = 5
DEAD_WALL_TILES: Final[int] = 14

STARTING_POINTS: Final[int] = 25_000
TARGET_POINTS: Final[int] = 30_000
HONBA_BONUS: Final[int] = 300
RIICHI_BET: Final[int] = 1_000

NO_WINNER_NONE: Final[int] = 0
NO_WINNER_NORMAL: Final[int] = 1

# ``score`` is in units of 100 points, ``TARGET_POINTS`` in points.
TARGET_SCORE: Final[int] = TARGET_POINTS // 100
RIICHI_BET_SCORE: Final[int] = RIICHI_BET // 100
HONBA_SCORE: Final[int] = HONBA_BONUS // 100
# A ron's discarder pays all of a honba; a tsumo's payers share it.
HONBA_PER_PAYER: Final[int] = HONBA_SCORE // (NUM_PLAYERS - 1)
# The noten players of an exhaustive draw pay 3000 points between them, in score units.
NOTEN_PENALTY_SCORE: Final[int] = 30
# Riichi needs at least this many live-wall draws left after the declarer's.
MIN_LIVE_DRAWS_FOR_RIICHI: Final[int] = 4
# The extra rounds (西入) are one more wind: a kyoku per player.
SUDDEN_DEATH_ROUNDS: Final[int] = NUM_PLAYERS

LEGAL_ACTION_SIZE: Final[int] = Action.NUM_ACTION

ZERO_MASK_1D = jnp.zeros(Action.NUM_ACTION, dtype=jnp.bool_)
ZERO_MASK_2D = jnp.zeros((NUM_PLAYERS, Action.NUM_ACTION), dtype=jnp.bool_)

# The wall is ``deck``. The hands are dealt from its end and the draws go down from
# ``FIRST_DRAW_IDX``. ``deck[:DEAD_WALL_TILES]`` is the dead wall: the n-th kan's
# replacement tile is ``deck[RINSHAN_IX + n]``, the k-th dora indicator is
# ``deck[FIRST_DORA_INDICATOR_IX - 2 * k]`` and its ura is the tile below it.
FIRST_DRAW_IDX = NUM_PHYSICAL_TILES - 1 - INITIAL_HAND_TILES * NUM_PLAYERS
RINSHAN_IX: Final[int] = 10
FIRST_DORA_INDICATOR_IX: Final[int] = 9

# The dora each indicator points at, by tile type.
DORA_ARRAY = jnp.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        0,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        9,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        18,
        28,
        29,
        30,
        27,
        32,
        33,
        31,
    ],
    dtype=jnp.int8,
)
