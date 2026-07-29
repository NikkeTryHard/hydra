"""Physical tile id <-> RiichiEnv mjai-string conversion.

Verified against HandEvaluator.to_text and the engine's own mjai_to_tid
(WP-03A probe): suits are sequential four-copy blocks with the red five
FIRST ({16, 52, 88}), honors follow E/S/W/N/haku/hatsu/chun from 108.
"""

from __future__ import annotations

from hydra2.contracts.common import InvalidTileError, TileId

__all__ = ["mjai_string_of", "physical_of"]

_SUIT_BASES = {"m": 0, "p": 36, "s": 72}
_HONOR_BASES = {"E": 108, "S": 112, "W": 116, "N": 120, "P": 124, "F": 128, "C": 132}
_RED_ALIASES = {"0m": 16, "0p": 52, "0s": 88}


def physical_of(mjai_tile: str) -> TileId:
    """Parse one mjai tile string into its exact physical id.

    Suits are sequential four-copy blocks with the red five FIRST
    ({16, 52, 88}); the unsuffixed "5x" string therefore resolves to the
    SECOND copy (17/53/89) and the red five is addressed as "5xr"
    (engine ``mjai_to_tid``: "5sr" -> 88, "5s" -> 89). Honors follow
    E/S/W/N/haku/hatsu/chun from 108.
    """
    text = mjai_tile
    alias = _RED_ALIASES.get(text)
    if alias is not None:
        return TileId(alias)
    if len(text) == 3 and text[2] == "r" and text[1] in _SUIT_BASES:
        if text[0] != "5":
            raise InvalidTileError(f"red suffix on non-five tile {text!r}")
        return TileId(_SUIT_BASES[text[1]] + 16)
    if len(text) == 2 and text[1] in _SUIT_BASES:
        try:
            number = int(text[0])
        except ValueError as exc:
            raise InvalidTileError(f"invalid mjai tile {text!r}") from exc
        if not 1 <= number <= 9:
            raise InvalidTileError(f"invalid mjai tile {text!r}")
        if number == 5:
            # The red five is the FIRST copy of the block; an unsuffixed
            # "5x" therefore resolves to the second copy. Mirrors the
            # engine's own mjai_to_tid ("5s" -> 89).
            return TileId(_SUIT_BASES[text[1]] + 17)
        return TileId(_SUIT_BASES[text[1]] + 4 * (number - 1))
    if text in _HONOR_BASES:
        return TileId(_HONOR_BASES[text])
    raise InvalidTileError(f"invalid mjai tile {text!r}")


def mjai_string_of(tile: int) -> str:
    """Render one physical id as its canonical mjai string (red fives marked)."""
    value = tile
    if not 0 <= value <= 135:
        raise InvalidTileError(f"physical tile out of range: {value}")
    for suffix, base in (("m", 0), ("p", 36), ("s", 72)):
        if base <= value < base + 36:
            number = (value - base) // 4 + 1
            if number == 5 and value % 4 == 0:
                return f"5{suffix}r"
            return f"{number}{suffix}"
    for letter, base in _HONOR_BASES.items():
        if base <= value < base + 4:
            return letter
    raise InvalidTileError(f"no mjai rendering for tile {value}")  # pragma: no cover
