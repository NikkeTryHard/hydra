"""SPEC 8 actor observation vocabulary — phases, melds, and field validators.

Owns the closed inventories the actor modules build on: the fixed dora
indicator shape with its sentinel, the phase and meld-kind unions, the
tile/scalar field validators shared by the observation dataclass and the
builder, and the exposed-meld record with its canonical reference rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from hydra2.contracts.common import (
    ContractError,
    Seat,
    TileId,
    TileType,
    make_seat,
    make_tile_id,
)

__all__ = [
    "DORA_SENTINEL",
    "DORA_SHAPE",
    "MELD_KINDS",
    "PHASES",
    "MeldKind",
    "Phase",
    "VisibleMeld",
    "visible_meld_id",
]

# ---------------------------------------------------------------------------
# Fixed dora indicator shape (BUILD checklist: `(5,)`, declared sentinel).
# ---------------------------------------------------------------------------
# Dora = bonus-indicator tiles; unrevealed slots carry -1, never a tile.

#: Sentinel for an unrevealed dora indicator slot.
DORA_SENTINEL = -1
#: Exact indicator shape; rejected otherwise, NEVER padded implicitly.
DORA_SHAPE = (5,)

# ---------------------------------------------------------------------------
# SPEC 8 phase union - canonical definition (action.py re-imports).
# ---------------------------------------------------------------------------

Phase = Literal[
    "round_start",
    "draw_decision",
    "discard_response",
    "kan_response",
    "round_end",
    "game_end",
]

#: Every phase literal accepted by :class:`ActionContext` and observations.
PHASES: tuple[Phase, ...] = (
    "round_start",
    "draw_decision",
    "discard_response",
    "kan_response",
    "round_end",
    "game_end",
)

# ---------------------------------------------------------------------------
# Visible melds - canonical home (SPEC 8).
# ---------------------------------------------------------------------------

#: Meld kinds carried by :class:`VisibleMeld`.
MeldKind = Literal["chi", "pon", "daiminkan", "ankan", "kakan"]
MELD_KINDS: tuple[MeldKind, ...] = ("chi", "pon", "daiminkan", "ankan", "kakan")

_FURIETEN_STATES = ("none", "temporary", "riichi", "discard")
_RIICHI_STATES = ("none", "declared", "accepted")

_WIND_TILE_TYPES = (27, 28, 29, 30)  # East, South, West, North logical types


def _tile_type_of(tile: int) -> int:
    """Logical tile type 0..33 of a validated physical id."""
    return tile // 4


def _require_plain_int(value: object, *, name: str, minimum: int, maximum: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ContractError(f"{name}={value} outside [{minimum}, {maximum}]")
    return value


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_str(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a str, got {type(value).__name__}")
    return value


def _require_enum(value: object, *, name: str, allowed: tuple[str, ...]) -> str:
    text = _require_str(value, name=name)
    if text not in allowed:
        raise ContractError(f"{name}={text!r} must be one of {list(allowed)}")
    return text


def _tile_tuple(values: Sequence[int], *, name: str) -> tuple[TileId, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractError(f"{name} must be a sequence of tile ids")
    return tuple(make_tile_id(v) for v in values)


def _quad(values: object, *, name: str, validator) -> tuple:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 4:
        raise ContractError(f"{name} must be exactly four entries")
    return tuple(validator(v, name=f"{name}[{i}]") for i, v in enumerate(values))


def _validate_wind_type(v: object, name: str) -> TileType:
    return TileType(
        _require_plain_int(v, name=name, minimum=_WIND_TILE_TYPES[0], maximum=_WIND_TILE_TYPES[-1])
    )


def _validate_score(v: object, name: str) -> int:
    return _require_plain_int(v, name=name, minimum=-(10**9), maximum=10**9)


@dataclass(frozen=True, slots=True)
class VisibleMeld:
    """An exposed meld visible to every actor (SPEC 8 exact field order).

    ``tiles`` are strictly ascending physical ids composing the meld. Chi/pon/
    daiminkan carry the offering ``source_seat`` and claimed ``called_tile``;
    ankan/kakan are self-contained. ``meld_id`` may be ``None``, in which case
    the canonical :func:`visible_meld_id` reference is derived.
    """

    meld_id: str | None
    kind: MeldKind
    owner: Seat
    source_seat: Seat | None = None
    called_tile: TileId | None = None
    tiles: tuple[TileId, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in MELD_KINDS:
            raise ContractError(f"meld kind must be one of {MELD_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "owner", make_seat(self.owner))
        tiles = tuple(make_tile_id(t) for t in self.tiles)
        if len(tiles) == 0 or list(tiles) != sorted(set(tiles)):
            raise ContractError(
                f"{self.kind} meld tiles must be non-empty, unique, ascending: {tiles!r}"
            )
        object.__setattr__(self, "tiles", tiles)
        expected_len = {"chi": 3, "pon": 3, "daiminkan": 4, "ankan": 4, "kakan": 4}[self.kind]
        if len(tiles) != expected_len:
            raise ContractError(
                f"{self.kind} meld must hold {expected_len} tiles, got {len(tiles)}"
            )
        types = [_tile_type_of(t) for t in tiles]
        if self.kind == "chi":
            if (
                any(t >= 27 for t in types)
                or len({t // 9 for t in types}) != 1
                or max(types) - min(types) != 2
                or len(set(types)) != 3
            ):
                raise ContractError(f"chi meld is not a same-suit run: {tiles!r}")
        elif len(set(types)) != 1:
            raise ContractError(f"{self.kind} meld tiles must share one logical type: {tiles!r}")
        if self.kind in ("ankan", "kakan"):
            if self.called_tile is not None or self.source_seat is not None:
                raise ContractError(f"{self.kind} meld has no called tile or source seat")
            if self.kind == "ankan":
                base = 4 * types[0]
                if tiles != tuple(range(base, base + 4)):
                    raise ContractError(
                        f"ankan meld tiles {tiles!r} must be consecutive 4 of type {types[0]}"
                    )
        else:
            if self.called_tile is None or self.source_seat is None:
                raise ContractError(f"{self.kind} meld requires called_tile and source_seat")
            if isinstance(self.called_tile, bool) or not isinstance(self.called_tile, int):
                raise ContractError(
                    "called_tile must be a tile id int 0..135, got "
                    f"{type(self.called_tile).__name__}"
                )
            if isinstance(self.source_seat, bool) or not isinstance(self.source_seat, int):
                raise ContractError(
                    f"source_seat must be a seat int 0..3, got {type(self.source_seat).__name__}"
                )
            called = make_tile_id(int(self.called_tile))
            source = make_seat(int(self.source_seat))
            if source == self.owner:
                raise ContractError(f"{self.kind} meld source seat equals owner")
            if called not in tiles:
                raise ContractError(f"{self.kind} meld called tile {called} not among tiles")
        resolved = self.meld_id if self.meld_id is not None else visible_meld_id(self)
        if not isinstance(resolved, str) or resolved == "":
            raise ContractError("meld_id must resolve to a non-empty string")
        object.__setattr__(self, "meld_id", resolved)

    def to_json(self) -> dict[str, object]:
        return {
            "meld_id": self.meld_id,
            "kind": self.kind,
            "owner": int(self.owner),
            "source_seat": None if self.source_seat is None else int(self.source_seat),
            "called_tile": None if self.called_tile is None else int(self.called_tile),
            "tiles": [int(t) for t in self.tiles],
        }


def visible_meld_id(meld: VisibleMeld) -> str:
    """Canonical prior-meld reference used by kakan metadata (SPEC 6.2)."""
    return f"{meld.kind}:{'.'.join(str(int(t)) for t in meld.tiles)}"
