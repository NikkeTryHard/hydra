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

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.contracts.common import (
    ContractError,
    Seat,
    TileId,
    TileType,
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

#: Sentinel for an unrevealed dora indicator slot (re-exported from the bridge).
DORA_SENTINEL: int = _bridge_contracts.DORA_SENTINEL
#: Exact indicator shape; rejected otherwise, NEVER padded implicitly (re-exported).
DORA_SHAPE: tuple[int, ...] = _bridge_contracts.DORA_SHAPE

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

#: Every phase literal accepted by :class:`ActionContext` and observations (re-exported).
PHASES: tuple[str, ...] = _bridge_contracts.PHASES

# ---------------------------------------------------------------------------
# Visible melds - canonical home (SPEC 8).
# ---------------------------------------------------------------------------

#: Meld kinds carried by :class:`VisibleMeld`.
MeldKind = Literal["chi", "pon", "daiminkan", "ankan", "kakan"]
MELD_KINDS: tuple[str, ...] = _bridge_contracts.MELD_KINDS

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
    try:
        tiles: list[TileId] = []
        for v in values:
            tile: TileId = _bridge_contracts.make_tile_id(v)
            tiles.append(tile)
        return tuple(tiles)
    except (ValueError, TypeError) as exc:
        raise ContractError(f"{name} rejected: {exc}") from exc


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
        """Thin bridge translator: validation lives in ``hydra2._native.contracts.VisibleMeld``.

        Scalars only cross the boundary (kind/owner/tiles/meld_id/source/called);
        the bridge owns the gate, Python keeps the typed ContractError surface.
        """
        try:
            record = _bridge_contracts.VisibleMeld(  # type: ignore[attr-defined]
                kind=self.kind,
                owner=self.owner,
                tiles=list(self.tiles),
                meld_id=self.meld_id,
                source_seat=self.source_seat,
                called_tile=self.called_tile,
            )
        except (ValueError, TypeError) as exc:
            raise ContractError(f"VisibleMeld rejected: {exc}") from exc
        kind: str = record.kind
        object.__setattr__(self, "kind", kind)
        owner: Seat = record.owner
        object.__setattr__(self, "owner", owner)
        tile_ids: list[TileId] = []
        for t in record.tiles:
            tile: TileId = t
            tile_ids.append(tile)
        object.__setattr__(self, "tiles", tuple(tile_ids))
        if record.source_seat is None:
            object.__setattr__(self, "source_seat", None)
        else:
            source: Seat = record.source_seat
            object.__setattr__(self, "source_seat", source)
        if record.called_tile is None:
            object.__setattr__(self, "called_tile", None)
        else:
            called: TileId = record.called_tile
            object.__setattr__(self, "called_tile", called)
        meld_id: str = record.meld_id
        object.__setattr__(self, "meld_id", meld_id)

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
    """Canonical prior-meld reference used by kakan metadata (SPEC 6.2).

    Thin bridge translator: scalars (kind + tiles) cross the boundary, never
    the live meld object; bridge rejections surface as ContractError.
    """
    try:
        return _bridge_contracts.visible_meld_id(meld.kind, list(meld.tiles))  # type: ignore[attr-defined]
    except (ValueError, TypeError) as exc:
        raise ContractError(f"visible_meld_id rejected: {exc}") from exc
