"""The 3-player table (三人麻雀), as Tenhou plays it: no 2m-8m, no chi, and norths set aside."""

from __future__ import annotations

from typing import Final

import jax.numpy as jnp

from mahjax.common.constants import FALSE, INITIAL_HAND_TILES, TRUE  # noqa: F401  (re-exported)

from .action import Action

NUM_PLAYERS: Final[int] = 3
# The tile ids the wall is shuffled from: every id but 2m-8m (types 1-7), 108 tiles.
TILE_IDS = jnp.array([tile_id for tile_id in range(136) if not 1 <= tile_id // 4 <= 7], dtype=jnp.int32)
NUM_PHYSICAL_TILES: Final[int] = 108

HAS_CHI: Final[bool] = False
HAS_KITA: Final[bool] = True
KITA_ACTION: Final[int] = Action.KITA
NORTH: Final[int] = 30
# Special abortive draws that need four players.
HAS_FOUR_WIND_DRAW: Final[bool] = False
HAS_FOUR_RIICHI_DRAW: Final[bool] = False

MAX_DISCARDS_PER_PLAYER: Final[int] = 32
MAX_DORA_INDICATORS: Final[int] = 5
DEAD_WALL_TILES: Final[int] = 14

STARTING_POINTS: Final[int] = 35_000
TARGET_POINTS: Final[int] = 40_000
HONBA_BONUS: Final[int] = 200
RIICHI_BET: Final[int] = 1_000

# ``score`` is in units of 100 points, ``TARGET_POINTS`` in points.
TARGET_SCORE: Final[int] = TARGET_POINTS // 100
RIICHI_BET_SCORE: Final[int] = RIICHI_BET // 100
HONBA_SCORE: Final[int] = HONBA_BONUS // 100
# A ron's discarder pays all of a honba; a tsumo's payers share it.
HONBA_PER_PAYER: Final[int] = HONBA_SCORE // (NUM_PLAYERS - 1)
# The noten players of an exhaustive draw pay 2000 points between them, in score units.
NOTEN_PENALTY_SCORE: Final[int] = 20
# Riichi needs at least this many live-wall draws left after the declarer's.
MIN_LIVE_DRAWS_FOR_RIICHI: Final[int] = 3
# The extra rounds (西入) are one more wind: a kyoku per player.
SUDDEN_DEATH_ROUNDS: Final[int] = NUM_PLAYERS

LEGAL_ACTION_SIZE: Final[int] = Action.NUM_ACTION

ZERO_MASK_1D = jnp.zeros(Action.NUM_ACTION, dtype=jnp.bool_)
ZERO_MASK_2D = jnp.zeros((NUM_PLAYERS, Action.NUM_ACTION), dtype=jnp.bool_)

# The wall is ``deck``. The hands are dealt from its end and the draws go down from
# ``FIRST_DRAW_IDX``. ``deck[:DEAD_WALL_TILES]`` is the dead wall: the n-th replacement
# tile, for a kan or a kita, is ``deck[RINSHAN_IX + n]``, the k-th dora indicator is
# ``deck[FIRST_DORA_INDICATOR_IX - 2 * k]`` and its ura is the tile below it. Every
# replacement moves the haitei line up, so tiles past the dead wall keep it at 14.
FIRST_DRAW_IDX = NUM_PHYSICAL_TILES - 1 - INITIAL_HAND_TILES * NUM_PLAYERS
RINSHAN_IX: Final[int] = 10
FIRST_DORA_INDICATOR_IX: Final[int] = 9

# The dora each indicator points at, by tile type. With 2m-8m out, 1m and 9m point at
# each other; types 1-7 never appear.
DORA_ARRAY = jnp.array(
    [8, 2, 3, 4, 5, 6, 7, 8, 0]
    + [10, 11, 12, 13, 14, 15, 16, 17, 9]
    + [19, 20, 21, 22, 23, 24, 25, 26, 18]
    + [28, 29, 30, 27]
    + [32, 33, 31],
    dtype=jnp.int8,
)
