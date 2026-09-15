"""Shared wall-less oracle primitives."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING as TYPE_CHECKING

from hydra2.contracts.common import ContractError as ContractError

if TYPE_CHECKING:
    from typing import Any as Any

__all__ = [
    "_BAKAZE_TO_WIND",
    "_LIVE_WALL_BASE",
    "_TRANSPARENT_KINDS",
    "_WINDOW_HEAD_TYPES",
    "_adapter_hash",
    "_legal_mjai_type",
]


#: Live-wall countdown base: 136 tiles minus 4x13 dealt minus the 14-tile
#: dead wall (mirrors the tehais-install countdown semantics).
_LIVE_WALL_BASE = 136 - 52 - 14


#: Bakaze letter -> round-wind TileType (same scale as seat winds, 27..30).
_BAKAZE_TO_WIND = {"E": 27, "S": 28, "W": 29, "N": 30}


_TRANSPARENT_KINDS = frozenset({"dora", "reach_accepted"})


_WINDOW_HEAD_TYPES = frozenset({"none", "chi", "pon", "daiminkan", "hora"})


def _legal_mjai_type(raw: Any) -> str:
    """MJAI type string of a yielded engine action (public mapping only)."""
    try:
        mjai: Any = raw.to_mjai()
    except Exception as exc:
        raise ContractError(f"engine action without an MJAI mapping: {exc}") from exc
    if isinstance(mjai, str):
        try:
            mjai = json.loads(mjai)
        except ValueError as exc:
            raise ContractError(f"engine action MJAI unparseable: {exc}") from exc
    if not isinstance(mjai, dict) or not isinstance(mjai.get("type"), str):
        raise ContractError(f"engine action without a type string: {mjai!r}")
    return mjai["type"]


def _adapter_hash() -> str:
    from hydra2.data import replay_expand as _re

    return _re._adapter_hash()
