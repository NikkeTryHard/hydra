"""SPEC 6.1 action kinds — frozen vocabulary and context-free validators.

Owns the context-free kind layer: the frozen ``ActionKind`` ordinals 0..12,
the canonical JSON metadata domain with per-kind declarations, the
phase-gating table, and the tile validators shared by the action record and
template constructors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from hydra2.contracts.common import (
    ContractError,
    TileId,
)
from hydra2.contracts.observation_types import (
    MELD_KINDS as MELD_KINDS,
)
from hydra2.contracts.observation_types import (
    PHASES as PHASES,
)
from hydra2.contracts.observation_types import (
    Phase as Phase,
)
from hydra2.contracts.observation_types import (
    VisibleMeld as VisibleMeld,
)
from hydra2.contracts.observation_types import (
    visible_meld_id as visible_meld_id,
)

__all__ = [
    "ACTION_KINDS",
    "ACTION_KIND_ORDINALS",
    "ACTION_PHASES",
    "CLAIM_KINDS",
    "KAKAN_METADATA_KEYS",
    "MELD_KINDS",
    "METADATA_KEYS_BY_KIND",
    "PHASES",
    "JsonValue",
    "Phase",
    "VisibleMeld",
]

# ---------------------------------------------------------------------------
# SPEC 6.1 - Action kinds and stable IDs.
# ---------------------------------------------------------------------------


ActionKind = Literal[
    "pass",
    "discard",
    "tsumogiri",
    "riichi_discard",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "ron",
    "tsumo",
    "abort_nine_terminals",
    "accept_abortive_draw",
]

#: Frozen kind -> ordinal mapping (SPEC 6.1). NEVER reordered or extended in place.
ACTION_KIND_ORDINALS: dict[str, int] = {
    "pass": 0,
    "discard": 1,
    "tsumogiri": 2,
    "riichi_discard": 3,
    "chi": 4,
    "pon": 5,
    "daiminkan": 6,
    "ankan": 7,
    "kakan": 8,
    "ron": 9,
    "tsumo": 10,
    "abort_nine_terminals": 11,
    "accept_abortive_draw": 12,
}


def _kind_ordinal(kind: str) -> int:
    return ACTION_KIND_ORDINALS[kind]


#: Kinds in frozen ordinal order.
ACTION_KINDS: tuple[str, ...] = tuple(sorted(ACTION_KIND_ORDINALS, key=_kind_ordinal))

JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | tuple["JsonValue", ...]
    | list["JsonValue"]
    | Mapping[str, "JsonValue"]
)

#: Owner-decision D-WP02C-1: coarse engine-phase gating per action kind. The
#: contract rejects structurally valid actions in phases where no engine can
#: ever offer them; finer rule predicates (yaku/riichi shape, abort offers)
#: stay with the rules contract. Ron accepts a robbed kakan at kan_response;
#: accept_abortive_draw covers discard- and kan-offered aborts (four riichi /
#: four winds / four kans).
ACTION_PHASES: dict[str, frozenset[str]] = {
    "pass": frozenset({"discard_response", "kan_response"}),
    "discard": frozenset({"draw_decision"}),
    "tsumogiri": frozenset({"draw_decision"}),
    "riichi_discard": frozenset({"draw_decision"}),
    "chi": frozenset({"discard_response"}),
    "pon": frozenset({"discard_response"}),
    "daiminkan": frozenset({"discard_response"}),
    "ankan": frozenset({"draw_decision"}),
    "kakan": frozenset({"draw_decision"}),
    "ron": frozenset({"discard_response", "kan_response"}),
    "tsumo": frozenset({"draw_decision"}),
    "abort_nine_terminals": frozenset({"draw_decision"}),
    "accept_abortive_draw": frozenset({"discard_response", "kan_response"}),
}

#: Kinds that claim an offered tile from another seat.
CLAIM_KINDS: frozenset[str] = frozenset({"chi", "pon", "daiminkan"})

#: Schema-declared metadata keys per kind (SPEC 6.2: arbitrary extension
#: metadata is rejected). Only kakan carries metadata in v1.
METADATA_KEYS_BY_KIND: dict[str, tuple[str, ...]] = {
    kind: ("prior_pon_meld_id",) if kind == "kakan" else () for kind in ACTION_KINDS
}
KAKAN_METADATA_KEYS = METADATA_KEYS_BY_KIND["kakan"]

_SOURCE_OFFSETS_CLAIM: tuple[int, ...] = (-1, 1, 2)


def _tile_type(tile: int) -> int:
    """Logical tile type 0..33 of a validated physical id (SPEC 4.1)."""
    return tile // 4


def _is_honor(tile_type: int) -> bool:
    return tile_type >= 27


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_json_value(value: object) -> None:
    """Restrict metadata values to the canonical JSON domain (SPEC 2.2)."""
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ContractError(f"metadata int out of canonical range: {value!r}")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("metadata floats must be finite")
        return
    if isinstance(value, str):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)  # pyrefly: ignore[unknown-argument-type]  # reason: isinstance branches narrow only the container; item stays object
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_json_value(item)  # pyrefly: ignore[unknown-argument-type]  # reason: isinstance branches narrow only the container; item stays object
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"metadata object keys must be str: {key!r}")
            _validate_json_value(item)  # pyrefly: ignore[unknown-argument-type]  # reason: isinstance branches narrow only the container; item stays object
        return


def _validated_metadata(
    kind: str, metadata: Sequence[tuple[str, JsonValue]]
) -> tuple[tuple[str, JsonValue], ...]:
    allowed = METADATA_KEYS_BY_KIND[kind]
    seen: list[str] = []
    normalized: list[tuple[str, JsonValue]] = []
    for pair in metadata:
        if not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[0], str):
            raise ContractError(f"{kind} metadata entries must be (str, JsonValue) pairs")
        key, value = pair
        _validate_json_value(value)
        if key not in allowed:
            raise ContractError(
                f"{kind} does not declare metadata key {key!r}; declared keys are {list(allowed)}"
            )
        seen.append(key)
        normalized.append((key, value))
    if seen != sorted(seen):
        raise ContractError(f"{kind} metadata keys must be sorted ascending: {seen!r}")
    if len(set(seen)) != len(seen):
        raise ContractError(f"{kind} metadata keys must be unique: {seen!r}")
    if kind == "kakan" and tuple(seen) != ("prior_pon_meld_id",):
        raise ContractError(
            f"kakan requires exactly the prior-pon meld reference {list(KAKAN_METADATA_KEYS)}"
        )
    if kind != "kakan" and len(seen) > 0:
        raise ContractError(f"{kind} carries no metadata in schema v1")
    return tuple(normalized)


def _consumed_pair_forms_run(called_tile: TileId, consumed: Sequence[TileId]) -> bool:
    """True iff called + two consumed form one same-suit consecutive run."""
    types = [_tile_type(called_tile), *(_tile_type(t) for t in consumed)]
    suits = {t // 9 for t in types}
    if len(suits) != 1 or any(_is_honor(t) for t in types):
        return False
    return max(types) - min(types) == 2 and len(set(types)) == 3


def _all_same_type(tiles: Sequence[TileId]) -> bool:
    first = _tile_type(tiles[0])
    return all(_tile_type(t) == first for t in tiles[1:])
