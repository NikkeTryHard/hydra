from __future__ import annotations

from typing import Final

import jax.numpy as jnp

NUM_TILE_TYPES: Final[int] = 34
NUM_TILE_TYPES_WITH_RED: Final[int] = 37
COPIES_PER_TILE: Final[int] = 4

INITIAL_HAND_TILES: Final[int] = 13
MAX_HAND_TILES: Final[int] = 14
MAX_MELDS_PER_PLAYER: Final[int] = 4
MAX_KANS_PER_ROUND: Final[int] = 4

RED_FIVE_TILE_TYPES: Final[tuple[int, int, int]] = (4, 13, 22)
RED_FIVE_TILE_IDS: Final[tuple[int, ...]] = tuple(tile_type * 4 for tile_type in RED_FIVE_TILE_TYPES)

SENTINEL_TILE_ID: Final[int] = -1

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

TILE_RANGE = jnp.arange(NUM_TILE_TYPES, dtype=jnp.int32)
