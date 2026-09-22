"""SPEC 6.1 action kinds — frozen vocabulary (values) and context-free validators.

The frozen kind tables live in the ``hydra2._native.contracts`` Rust surface
(single source); this module keeps thin re-exports plus its ``Literal`` types,
validators, metadata logic, and the ``observation_types`` re-exports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from hydra2._native import contracts as _kinds_bridge  # pyrefly: ignore[missing-import]
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

#: Frozen kind -> ordinal mapping (SPEC 6.1, re-exported from the bridge).
_ORDINALS_RAW: dict[str, int] = _kinds_bridge.ACTION_KIND_ORDINALS
ACTION_KIND_ORDINALS: dict[str, int] = dict(_ORDINALS_RAW)
#: Kinds in frozen ordinal order (re-exported from the bridge).
_KINDS_RAW: tuple[str, ...] = _kinds_bridge.ACTION_KINDS
ACTION_KINDS: tuple[str, ...] = tuple(_KINDS_RAW)

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
_PHASES_RAW: dict[str, frozenset[str]] = _kinds_bridge.ACTION_PHASES
ACTION_PHASES: dict[str, frozenset[str]] = {
    kind: frozenset(members) for kind, members in _PHASES_RAW.items()
}
#: Kinds that claim an offered tile from another seat (re-exported).
_CLAIM_RAW: frozenset[str] = _kinds_bridge.CLAIM_KINDS
CLAIM_KINDS: frozenset[str] = frozenset(_CLAIM_RAW)

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
