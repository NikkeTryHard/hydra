"""SPEC 6 canonical action vocabulary: kinds, invariants, table, codec.

Contract layer (SPEC 1: imports only the Python standard library and sibling
contract modules). This module owns:

- SPEC 6.1 frozen ``ActionKind`` ordinals 0..12 and ``CanonicalAction``.
- SPEC 6.2 structural invariants, enforced in constructors (context-free) and
  again against an :class:`ActionContext` by the codec (phase, offered
  tile/source, owned physical tiles, required prior meld).
- SPEC 6.3 ``CanonicalActionTemplate`` / ``ActionContext`` / ``ActionTable`` /
  ``ActionCodec`` with lexicographic None-first generation order over
  ``(kind_ordinal, tile, called_tile, consumed_tiles, source_offset,
  declares_riichi, meld_ref_required)``.
- The versioned action-table artifact document for
  ``configs/contracts/action_table_v1.json`` (SPEC 2.2 envelope; RFC 8785
  canonical bytes; sha256 digest freezing template indices).

Boundary notes:
- ``Phase``/``PHASES``, ``MeldKind``/``MELD_KINDS``, ``VisibleMeld``, and
  ``visible_meld_id`` are canonically defined in
  :mod:`hydra2.contracts.observation` (SPEC section 8 owns them; WP-02D clean
  cutover) and re-exported here for the codec's callers.
- ``canonical_json_bytes`` lives in the dependency-free leaf
  :mod:`hydra2.contracts.canonical` (moved verbatim, WP-02D cutover) and is
  re-exported; it stays byte-equal with the WP-02A authority
  ``hydra2.artifacts.canonical`` (pinned by tests).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from hydra2.contracts.canonical import canonical_json_bytes
from hydra2.contracts.common import (
    ActionId,
    ContractError,
    DigestMismatchError,
    DigestText,
    IncompatibleSchemaError,
    InvalidActionError,
    InvalidTileError,
    SchemaVersion,
    Seat,
    TileId,
    make_action_id,
    make_digest_text,
    make_schema_version,
    make_seat,
    make_tile_id,
)
from hydra2.contracts.observation import (
    MELD_KINDS,
    PHASES,
    Phase,
    VisibleMeld,
    visible_meld_id,
)

__all__ = [
    "ACTION_KINDS",
    "ACTION_KIND_ORDINALS",
    "ACTION_PHASES",
    "ACTION_TABLE_ARTIFACT_TYPE",
    "ACTION_TABLE_RELPATH",
    "ACTION_TABLE_SCHEMA_VERSION",
    "CLAIM_KINDS",
    "KAKAN_METADATA_KEYS",
    "MELD_KINDS",
    "METADATA_KEYS_BY_KIND",
    "PHASES",
    "ActionContext",
    "ActionTable",
    "CanonicalAction",
    "CanonicalActionCodec",
    "CanonicalActionTemplate",
    "JsonValue",
    "Phase",
    "VisibleMeld",
    "action_table_envelope",
    "build_action_table",
    "canonical_action_codec",
    "generate_action_templates",
    "load_action_table",
    "template_sort_key",
    "visible_meld_id",
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

#: Kinds in frozen ordinal order.
ACTION_KINDS: tuple[str, ...] = tuple(
    sorted(ACTION_KIND_ORDINALS, key=lambda k: ACTION_KIND_ORDINALS[k])
)

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
            _validate_json_value(item)  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_json_value(item)  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"metadata object keys must be str: {key!r}")
            _validate_json_value(item)  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
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


# ---------------------------------------------------------------------------
# SPEC 6.1 - CanonicalAction with SPEC 6.2 invariants.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanonicalAction:
    """A fully parameterized canonical action (SPEC 6.1)."""

    kind: ActionKind
    actor: Seat
    tile: TileId | None
    called_tile: TileId | None
    consumed_tiles: tuple[TileId, ...]
    source_seat: Seat | None
    declares_riichi: bool
    metadata: tuple[tuple[str, JsonValue], ...]

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KIND_ORDINALS:
            raise ContractError(f"unknown action kind {self.kind!r}")
        object.__setattr__(self, "actor", make_seat(self.actor))
        tile = None if self.tile is None else make_tile_id(self.tile)
        called = None if self.called_tile is None else make_tile_id(self.called_tile)
        source = None if self.source_seat is None else make_seat(self.source_seat)
        consumed = tuple(make_tile_id(t) for t in self.consumed_tiles)
        if list(consumed) != sorted(set(consumed)):
            raise ContractError(
                f"{self.kind}: consumed_tiles must be unique and ascending: {consumed!r}"
            )
        declares = _require_bool(self.declares_riichi, name="declares_riichi")
        if declares != (self.kind == "riichi_discard"):
            raise ContractError(
                f"{self.kind}: declares_riichi must be {self.kind == 'riichi_discard'}"
            )
        object.__setattr__(self, "tile", tile)
        object.__setattr__(self, "called_tile", called)
        object.__setattr__(self, "source_seat", source)
        object.__setattr__(self, "consumed_tiles", consumed)
        object.__setattr__(self, "metadata", _validated_metadata(self.kind, self.metadata))

        if self.kind in ("pass", "ron"):
            if len(consumed) > 0 or called is not None:
                raise InvalidActionError(f"{self.kind}: no consumed/called tiles allowed")
            if self.kind == "pass" and tile is not None:
                raise InvalidActionError("pass: no tile allowed")
            if self.kind == "ron" and tile is None:
                raise InvalidActionError("ron: winning offered tile required")
            if source is not None and source == self.actor:
                raise InvalidActionError(f"{self.kind}: source seat equals actor")
        elif self.kind in ("discard", "tsumogiri", "riichi_discard"):
            if tile is None or called is not None or len(consumed) > 0 or source is not None:
                raise InvalidActionError(f"{self.kind}: exactly tile, no source/called/consumed")
        elif self.kind == "chi":
            if tile is not None or called is None or len(consumed) != 2:
                raise InvalidActionError("chi: one called tile plus two consumed tiles")
            if source is None:
                raise InvalidActionError("chi: source seat required")
            if source != (self.actor + 3) % 4:
                raise InvalidActionError(
                    f"chi source must be previous seat, got {source} (actor {self.actor})"
                )
            if not _consumed_pair_forms_run(called, consumed):
                raise InvalidActionError(
                    f"chi tiles are not three consecutive same-suit values "
                    f"(honors forbidden): {[called, *consumed]!r}"
                )
        elif self.kind in ("pon", "daiminkan"):
            needed = 2 if self.kind == "pon" else 3
            if tile is not None or called is None or len(consumed) != needed:
                raise InvalidActionError(
                    f"{self.kind}: one called tile plus exactly {needed} consumed"
                )
            if source is None or source == self.actor:
                raise InvalidActionError(f"{self.kind}: source must differ from actor")
            group = [called, *consumed]
            if len({int(g) for g in group}) != len(group):
                raise InvalidActionError(f"{self.kind}: physical tiles must be distinct")
            if not _all_same_type(group):
                raise InvalidActionError(
                    f"{self.kind}: consumed tiles must share the called logical type: {group!r}"
                )
        elif self.kind == "ankan":
            if tile is not None or called is not None or source is not None:
                raise InvalidActionError("ankan: no source/called/tile fields")
            if len(consumed) != 4 or not _all_same_type(consumed):
                raise InvalidActionError(
                    f"ankan: exactly four same-logical-type tiles required: {consumed!r}"
                )
            base = 4 * _tile_type(consumed[0])
            if consumed != tuple(range(base, base + 4)):
                raise InvalidActionError(
                    f"ankan must consume all four physical copies: {consumed!r}"
                )
        elif self.kind == "kakan":
            if tile is None or called is not None or len(consumed) > 0 or source is not None:
                raise InvalidActionError(
                    "kakan: added physical tile only; source/called/consumed empty"
                )
        elif self.kind == "tsumo":
            if tile is None or called is not None or len(consumed) > 0 or source is not None:
                raise InvalidActionError("tsumo: winning drawn tile only")
        elif self.kind in ("abort_nine_terminals", "accept_abortive_draw"):
            if (
                tile is not None
                or called is not None
                or len(consumed) > 0
                or source is not None
                or len(self.metadata) > 0
            ):
                raise InvalidActionError(f"{self.kind}: parameterless abort action")


# ---------------------------------------------------------------------------
# SPEC 6.3 - Templates, context, table, codec.
# ---------------------------------------------------------------------------

SourceOffset = Literal[-1, 0, 1, 2]


@dataclass(frozen=True, slots=True)
class CanonicalActionTemplate:
    """Actor-independent identity of one canonical action slot (SPEC 6.3)."""

    kind: ActionKind
    tile: TileId | None
    called_tile: TileId | None
    consumed_tiles: tuple[TileId, ...]
    source_offset: SourceOffset | None  # relative modulo four; 0 only when kind permits self
    declares_riichi: bool
    meld_ref_required: bool

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KIND_ORDINALS:
            raise ContractError(f"unknown action kind {self.kind!r}")
        tile = None if self.tile is None else make_tile_id(self.tile)
        called = None if self.called_tile is None else make_tile_id(self.called_tile)
        consumed = tuple(make_tile_id(t) for t in self.consumed_tiles)
        if list(consumed) != sorted(set(consumed)):
            raise ContractError(f"template consumed_tiles must be unique ascending: {consumed!r}")
        offset = self.source_offset
        if offset is not None and offset not in (-1, 0, 1, 2):
            raise ContractError(f"source_offset must be one of (None, -1, 0, 1, 2): {offset!r}")
        declares = _require_bool(self.declares_riichi, name="declares_riichi")
        meld_ref = _require_bool(self.meld_ref_required, name="meld_ref_required")
        if declares != (self.kind == "riichi_discard"):
            raise ContractError(
                f"{self.kind} template declares_riichi must be {self.kind == 'riichi_discard'}"
            )
        if meld_ref != (self.kind == "kakan"):
            raise ContractError(
                f"{self.kind} template meld_ref_required must be {self.kind == 'kakan'}"
            )

        empty: tuple[TileId, ...] = ()
        if self.kind in ("pass", "discard", "tsumogiri", "riichi_discard", "tsumo"):
            if called is not None or consumed != empty:
                raise ContractError(f"{self.kind} template carries no called/consumed tiles")
            needs_tile = self.kind != "pass"
            if (tile is not None) != needs_tile:
                raise ContractError(
                    f"{self.kind} template {'requires' if needs_tile else 'forbids'} tile"
                )
            allowed = (None,) if self.kind != "pass" else ((None, *_SOURCE_OFFSETS_CLAIM))
            if offset not in allowed:
                raise InvalidActionError(f"{self.kind} template source_offset must be in {allowed}")
        elif self.kind == "chi":
            if tile is not None or called is None or len(consumed) != 2:
                raise ContractError("chi template: one called tile plus two consumed tiles")
            if offset != -1:
                raise ContractError("chi template source_offset must be -1 (previous seat)")
            if not _consumed_pair_forms_run(called, consumed):
                raise ContractError(
                    f"chi template tiles must form a same-suit consecutive run: "
                    f"{[called, *consumed]!r}"
                )
        elif self.kind in ("pon", "daiminkan"):
            needed = 2 if self.kind == "pon" else 3
            if tile is not None or called is None or len(consumed) != needed:
                raise ContractError(
                    f"{self.kind} template: one called tile plus exactly {needed} consumed"
                )
            if offset not in _SOURCE_OFFSETS_CLAIM:
                raise ContractError(
                    f"{self.kind} template source_offset must be one of {_SOURCE_OFFSETS_CLAIM}"
                )
            group = [called, *consumed]
            if len({int(g) for g in group}) != len(group) or not _all_same_type(group):
                raise ContractError(
                    f"{self.kind} template tiles must be distinct same-type physical ids: {group!r}"
                )
        elif self.kind == "ankan":
            if tile is not None or called is not None or offset is not None:
                raise ContractError("ankan template: no tile/called/source_offset")
            base_types = {_tile_type(t) for t in consumed}
            if len(consumed) != 4 or len(base_types) != 1:
                raise ContractError(f"ankan template needs four same-type tiles: {consumed!r}")
            base = 4 * _tile_type(consumed[0])
            if consumed != tuple(range(base, base + 4)):
                raise ContractError(f"ankan template must span all four copies: {consumed!r}")
        elif self.kind == "kakan":
            if tile is None or called is not None or consumed != empty or offset is not None:
                raise ContractError(
                    "kakan template: added tile only, no called/consumed/source_offset"
                )
        elif self.kind == "ron":
            if tile is None or called is not None or consumed != empty:
                raise ContractError("ron template: offered winning tile only")
            if offset not in _SOURCE_OFFSETS_CLAIM:
                raise ContractError(
                    f"ron template source_offset must be one of {_SOURCE_OFFSETS_CLAIM}"
                )
        elif self.kind in ("abort_nine_terminals", "accept_abortive_draw"):
            if tile is not None or called is not None or consumed != empty or offset is not None:
                raise ContractError(f"{self.kind} template is parameterless")


def template_sort_key(template: CanonicalActionTemplate) -> tuple:
    """SPEC 6.3 generation order: lexicographic, ``None`` before integers."""
    ordinal = ACTION_KIND_ORDINALS[template.kind]

    def none_first_int(value: int | None) -> tuple[int, int]:
        return (0, 0) if value is None else (1, value)

    return (
        ordinal,
        none_first_int(template.tile),
        none_first_int(template.called_tile),
        template.consumed_tiles,
        none_first_int(template.source_offset),
        template.declares_riichi,
        template.meld_ref_required,
    )


def generate_action_templates() -> tuple[CanonicalActionTemplate, ...]:
    """Enumerate every and only structurally valid template once (SPEC 6.3).

    Census (analytic): pass 4, discard/tsumogiri/riichi_discard 136 each,
    chi 4032, pon 1224, daiminkan 408, ankan 34, kakan 136, ron 1224,
    tsumo 136, both abort kinds 1 each => 7608 templates.
    """
    templates: list[CanonicalActionTemplate] = []

    def add(kind: str, **kwargs: object) -> None:
        templates.append(CanonicalActionTemplate(kind=kind, **kwargs))  # type: ignore[arg-type]  # reason: kwargs statically object; per-kind shapes validated in CanonicalActionTemplate.__post_init__

    for offset in (None, -1, 1, 2):
        add(
            "pass",
            tile=None,
            called_tile=None,
            consumed_tiles=(),
            source_offset=offset,
            declares_riichi=False,
            meld_ref_required=False,
        )
    for kind in ("discard", "tsumogiri", "riichi_discard"):
        for tile in range(136):
            add(
                kind,
                tile=tile,
                called_tile=None,
                consumed_tiles=(),
                source_offset=None,
                declares_riichi=(kind == "riichi_discard"),
                meld_ref_required=False,
            )
    for suit_base in (0, 9, 18):
        for low in range(7):
            run_types = (suit_base + low, suit_base + low + 1, suit_base + low + 2)
            for position, called_type in enumerate(run_types):
                others = run_types[:position] + run_types[position + 1 :]
                for called_copy in range(4):
                    called = 4 * called_type + called_copy
                    for copy_a in range(4):
                        for copy_b in range(4):
                            pair = sorted((4 * others[0] + copy_a, 4 * others[1] + copy_b))
                            add(
                                "chi",
                                tile=None,
                                called_tile=called,
                                consumed_tiles=tuple(pair),
                                source_offset=-1,
                                declares_riichi=False,
                                meld_ref_required=False,
                            )
    for called in range(136):
        ctype = _tile_type(called)
        others = [4 * ctype + c for c in range(4) if 4 * ctype + c != called]
        pon_pairs = [(others[i], others[j]) for i in range(3) for j in range(i + 1, 3)]
        for offset in _SOURCE_OFFSETS_CLAIM:
            for pair in pon_pairs:
                add(
                    "pon",
                    tile=None,
                    called_tile=called,
                    consumed_tiles=tuple(sorted(pair)),
                    source_offset=offset,
                    declares_riichi=False,
                    meld_ref_required=False,
                )
            add(
                "daiminkan",
                tile=None,
                called_tile=called,
                consumed_tiles=tuple(sorted(others)),
                source_offset=offset,
                declares_riichi=False,
                meld_ref_required=False,
            )
    for tile_type in range(34):
        add(
            "ankan",
            tile=None,
            called_tile=None,
            consumed_tiles=tuple(range(4 * tile_type, 4 * tile_type + 4)),
            source_offset=None,
            declares_riichi=False,
            meld_ref_required=False,
        )
    for tile in range(136):
        add(
            "kakan",
            tile=tile,
            called_tile=None,
            consumed_tiles=(),
            source_offset=None,
            declares_riichi=False,
            meld_ref_required=True,
        )
    for tile in range(136):
        for offset in _SOURCE_OFFSETS_CLAIM:
            add(
                "ron",
                tile=tile,
                called_tile=None,
                consumed_tiles=(),
                source_offset=offset,
                declares_riichi=False,
                meld_ref_required=False,
            )
    for tile in range(136):
        add(
            "tsumo",
            tile=tile,
            called_tile=None,
            consumed_tiles=(),
            source_offset=None,
            declares_riichi=False,
            meld_ref_required=False,
        )
    for kind in ("abort_nine_terminals", "accept_abortive_draw"):
        add(
            kind,
            tile=None,
            called_tile=None,
            consumed_tiles=(),
            source_offset=None,
            declares_riichi=False,
            meld_ref_required=False,
        )

    return tuple(sorted(templates, key=template_sort_key))


@dataclass(frozen=True, slots=True)
class ActionTable:
    """Versioned canonical action vocabulary with frozen integer IDs."""

    schema_version: SchemaVersion
    actions: tuple[CanonicalActionTemplate, ...]
    digest: DigestText
    _keys: tuple[tuple, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", make_schema_version(self.schema_version))
        actions = self.actions
        if len(actions) == 0 or not all(isinstance(a, CanonicalActionTemplate) for a in actions):
            raise ContractError("actions must be a non-empty CanonicalActionTemplate sequence")
        keys = tuple(template_sort_key(a) for a in actions)
        if list(keys) != sorted(keys):
            raise ContractError("action templates must be stored in generation order")
        if len(set(keys)) != len(keys):
            raise ContractError("duplicate action templates are rejected")
        object.__setattr__(self, "digest", make_digest_text(self.digest))
        object.__setattr__(self, "_keys", keys)

    def index_of(self, template: CanonicalActionTemplate) -> int | None:
        """Generation-order index of ``template``, or ``None`` when absent."""
        if not isinstance(template, CanonicalActionTemplate):
            raise ContractError("index_of expects a CanonicalActionTemplate")
        position = bisect_left(self._keys, template_sort_key(template))
        if position < len(self._keys) and self._keys[position] == template_sort_key(template):
            return position
        return None


@dataclass(frozen=True, slots=True)
class ActionContext:
    """Everything the codec may consult beyond the action itself (SPEC 6.3)."""

    actor: Seat
    action_table_hash: DigestText
    phase: Phase
    offered_tile: TileId | None
    offered_by: Seat | None
    own_concealed_tiles: tuple[TileId, ...]
    visible_melds: tuple[VisibleMeld, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor", make_seat(self.actor))
        object.__setattr__(self, "action_table_hash", make_digest_text(self.action_table_hash))
        if self.phase not in PHASES:
            raise ContractError(f"phase must be one of {PHASES}, got {self.phase!r}")
        offered_tile = None if self.offered_tile is None else make_tile_id(self.offered_tile)
        offered_by = None if self.offered_by is None else make_seat(self.offered_by)
        if (offered_tile is None) != (offered_by is None):
            raise ContractError("offered_tile and offered_by must both be set or both be None")
        if offered_by is not None and offered_by == self.actor:
            raise ContractError("offered_by must differ from actor; nobody offers to self")
        object.__setattr__(self, "offered_tile", offered_tile)
        object.__setattr__(self, "offered_by", offered_by)
        concealed = tuple(make_tile_id(t) for t in self.own_concealed_tiles)
        if list(concealed) != sorted(set(concealed)):
            raise ContractError(f"own_concealed_tiles must be unique and ascending: {concealed!r}")
        object.__setattr__(self, "own_concealed_tiles", concealed)
        melds = tuple(self.visible_melds)
        if not all(isinstance(m, VisibleMeld) for m in melds):
            raise ContractError("visible_melds entries must be VisibleMeld instances")
        object.__setattr__(self, "visible_melds", melds)


class ActionCodec:
    """SPEC 6.3 codec protocol; use :data:`canonical_action_codec`."""

    __slots__ = ()

    def encode(
        self,
        action: CanonicalAction,
        *,
        table: ActionTable,
        context: ActionContext,
    ) -> ActionId:
        raise NotImplementedError

    def decode(
        self,
        action_id: ActionId,
        *,
        table: ActionTable,
        context: ActionContext,
    ) -> CanonicalAction:
        raise NotImplementedError


_OFFSET_DELTA: dict[int, int] = {-1: 3, 0: 0, 1: 1, 2: 2}


def _offset_from_source(source: Seat | None, actor: Seat) -> SourceOffset | None:
    if source is None:
        return None
    delta = (int(source) - int(actor)) % 4
    if delta == 0:  # unreachable for validated actions; defensive
        raise InvalidActionError("source seat equals actor")
    assert delta in (1, 2, 3)
    if delta == 3:
        return cast("SourceOffset", -1)
    assert delta in (0, 1, 2)
    return cast("SourceOffset", delta)


def _resolve_source(offset: SourceOffset | None, actor: Seat) -> Seat | None:
    if offset is None:
        return None
    assert offset in (-1, 0, 1, 2)
    delta = _OFFSET_DELTA[int(offset)]
    assert delta in (0, 1, 2, 3)
    return make_seat((int(actor) + delta) % 4)


def _find_kakan_base(context: ActionContext, added_tile: TileId) -> VisibleMeld:
    added_type = _tile_type(added_tile)
    candidates = [
        meld
        for meld in context.visible_melds
        if meld.kind == "pon"
        and meld.owner == context.actor
        and _tile_type(meld.tiles[0]) == added_type
    ]
    if len(candidates) == 0:
        raise InvalidActionError(
            f"kakan of tile {added_tile}: no prior pon of type {added_type} owned by actor"
        )
    if len(candidates) > 1:  # impossible with 4 copies; defensive
        raise InvalidActionError(f"kakan of tile {added_tile}: ambiguous prior pons")
    meld = candidates[0]
    missing = set(range(4 * added_type, 4 * added_type + 4)) - {int(t) for t in meld.tiles}
    if missing != {int(added_tile)}:
        raise InvalidActionError(
            f"kakan of tile {added_tile}: prior pon {meld.tiles!r} leaves {sorted(missing)} free"
        )
    return meld


class _ContextValidator:
    """Shared contextual gate for encode/decode (SPEC 6.3 decode duties)."""

    __slots__ = ("context",)

    def __init__(self, context: ActionContext) -> None:
        self.context = context

    def validate(
        self,
        *,
        kind: str,
        tile: TileId | None,
        called_tile: TileId | None,
        consumed: tuple[TileId, ...],
        source_seat: Seat | None,
        metadata: tuple[tuple[str, JsonValue], ...],
    ) -> None:
        context = self.context
        if context.phase not in ACTION_PHASES[kind]:
            raise InvalidActionError(
                f"{kind} is illegal in phase {context.phase!r}; legal phases: "
                f"{sorted(ACTION_PHASES[kind])}"
            )
        if kind in CLAIM_KINDS:
            if context.offered_tile != called_tile or context.offered_by != source_seat:
                raise InvalidActionError(
                    f"{kind}: offered tile/source do not match called tile {called_tile} "
                    f"and source {source_seat}"
                )
        elif kind == "ron":
            if (
                source_seat is None
                or context.offered_tile != tile
                or context.offered_by != source_seat
            ):
                raise InvalidActionError(
                    f"ron: offered tile/source must match winning tile {tile} and its seat"
                )
        elif (
            kind == "pass"
            and source_seat is not None
            and (context.offered_by != source_seat or context.offered_tile is None)
        ):
            raise InvalidActionError(
                f"pass: recorded source {source_seat} does not match the open offer"
            )
        hand = context.own_concealed_tiles

        def owns(item: TileId) -> bool:
            return item in hand

        if kind in ("discard", "tsumogiri", "riichi_discard", "tsumo", "kakan"):
            assert tile is not None
            if not owns(tile):
                raise InvalidActionError(
                    f"{kind}: tile {tile} is not among actor's concealed tiles"
                )
        elif kind == "chi":
            missing = [t for t in consumed if not owns(t)]
            if len(missing) > 0:
                raise InvalidActionError(f"chi: consumed tiles {missing} not owned by actor")
        elif kind in ("pon", "daiminkan", "ankan"):
            missing = [t for t in consumed if not owns(t)]
            if len(missing) > 0:
                raise InvalidActionError(f"{kind}: consumed tiles {missing} not owned by actor")
        if kind == "kakan":
            assert tile is not None
            base = _find_kakan_base(context, tile)
            declared = dict(metadata).get("prior_pon_meld_id")
            if declared != visible_meld_id(base):
                raise InvalidActionError(
                    f"kakan metadata prior_pon_meld_id {declared!r} does not reference "
                    f"prior pon {base.tiles!r} ({visible_meld_id(base)!r})"
                )


class CanonicalActionCodec(ActionCodec):
    """Bidirectional codec; invalid table/context raises, never partial output."""

    __slots__ = ()

    def encode(
        self,
        action: CanonicalAction,
        *,
        table: ActionTable,
        context: ActionContext,
    ) -> ActionId:
        if not isinstance(action, CanonicalAction):
            raise InvalidActionError("encode expects a CanonicalAction")
        _require_matching_table(table, context)
        if action.actor != context.actor:
            raise InvalidActionError(
                f"action actor {action.actor} differs from context actor {context.actor}"
            )
        offset = _offset_from_source(action.source_seat, action.actor)
        try:
            template = CanonicalActionTemplate(
                kind=action.kind,
                tile=action.tile,
                called_tile=action.called_tile,
                consumed_tiles=action.consumed_tiles,
                source_offset=offset,
                declares_riichi=action.declares_riichi,
                meld_ref_required=(action.kind == "kakan"),
            )
        except (ContractError, InvalidTileError) as exc:
            raise InvalidActionError(f"action not representable in vocabulary: {exc}") from exc
        index = table.index_of(template)
        if index is None:
            raise InvalidActionError(
                f"{action.kind} combination is not part of action table {table.digest}"
            )
        _ContextValidator(context).validate(
            kind=action.kind,
            tile=action.tile,
            called_tile=action.called_tile,
            consumed=action.consumed_tiles,
            source_seat=action.source_seat,
            metadata=action.metadata,
        )
        return make_action_id(index)

    def decode(
        self,
        action_id: ActionId,
        *,
        table: ActionTable,
        context: ActionContext,
    ) -> CanonicalAction:
        _require_matching_table(table, context)
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise ContractError(
                f"action_id={action_id!r} must be an int, got {type(action_id).__name__}"
            )
        index = action_id
        if index < 0 or index >= len(table.actions):
            raise InvalidActionError(
                f"action_id {index} outside vocabulary size {len(table.actions)}"
            )
        template = table.actions[index]
        source_seat = _resolve_source(template.source_offset, context.actor)
        metadata: tuple[tuple[str, JsonValue], ...] = ()
        if template.meld_ref_required:
            assert template.tile is not None
            metadata = (
                ("prior_pon_meld_id", visible_meld_id(_find_kakan_base(context, template.tile))),
            )
        action = CanonicalAction(
            kind=template.kind,
            actor=context.actor,
            tile=template.tile,
            called_tile=template.called_tile,
            consumed_tiles=template.consumed_tiles,
            source_seat=source_seat,
            declares_riichi=template.declares_riichi,
            metadata=metadata,
        )
        _ContextValidator(context).validate(
            kind=action.kind,
            tile=action.tile,
            called_tile=action.called_tile,
            consumed=action.consumed_tiles,
            source_seat=action.source_seat,
            metadata=action.metadata,
        )
        return action


canonical_action_codec = CanonicalActionCodec()


def _require_matching_table(table: ActionTable, context: ActionContext) -> None:
    if not isinstance(table, ActionTable):
        raise InvalidActionError("codec requires an ActionTable")
    if not isinstance(context, ActionContext):
        raise InvalidActionError("codec requires an ActionContext")
    if context.action_table_hash != table.digest:
        raise InvalidActionError(
            f"context action_table_hash {context.action_table_hash} does not match "
            f"table digest {table.digest}"
        )


# ---------------------------------------------------------------------------
# Versioned artifact document (configs/contracts/action_table_v1.json).
# ---------------------------------------------------------------------------

ACTION_TABLE_ARTIFACT_TYPE = "hydra2.action_table"
ACTION_TABLE_SCHEMA_VERSION = SchemaVersion("1.0.0")
ACTION_TABLE_RELPATH = Path("configs") / "contracts" / "action_table_v1.json"

_TEMPLATE_JSON_FIELDS = (
    "called_tile",
    "consumed_tiles",
    "declares_riichi",
    "kind",
    "meld_ref_required",
    "source_offset",
    "tile",
)


def _reject_constant(token: str) -> object:
    raise ContractError(f"{token} is outside the canonical JSON domain")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _template_to_json(template: CanonicalActionTemplate) -> dict[str, object]:
    return {
        "called_tile": template.called_tile,
        "consumed_tiles": list(template.consumed_tiles),
        "declares_riichi": template.declares_riichi,
        "kind": template.kind,
        "meld_ref_required": template.meld_ref_required,
        "source_offset": template.source_offset,
        "tile": template.tile,
    }


def _identity_document(schema_version: str, actions: tuple[CanonicalActionTemplate, ...]) -> dict:
    return {
        "schema_version": schema_version,
        "actions": [_template_to_json(t) for t in actions],
    }


def compute_table_digest(
    actions: tuple[CanonicalActionTemplate, ...],
    schema_version: SchemaVersion = ACTION_TABLE_SCHEMA_VERSION,
) -> DigestText:
    """sha256 over the RFC 8785 canonical bytes of the digest-free payload."""
    identity = canonical_json_bytes(_identity_document(schema_version, actions))
    return DigestText("sha256:" + hashlib.sha256(identity).hexdigest())


def build_action_table(
    actions: tuple[CanonicalActionTemplate, ...] | None = None,
    *,
    schema_version: SchemaVersion = ACTION_TABLE_SCHEMA_VERSION,
) -> ActionTable:
    """Deterministically build the versioned table (digest included)."""
    ordered = tuple(
        sorted(
            actions if actions is not None else generate_action_templates(), key=template_sort_key
        )
    )
    return ActionTable(
        schema_version=schema_version,
        actions=ordered,
        digest=compute_table_digest(ordered, schema_version),
    )


def action_table_envelope(
    table: ActionTable, *, compatibility: Literal["exact", "backward_read"] = "exact"
) -> dict:
    """SPEC 2.2 envelope whose canonical bytes are the published artifact."""
    if compatibility != "exact":
        raise ContractError("action_table_v1 publishes exact compatibility only")
    payload = _identity_document(table.schema_version, table.actions)
    payload["digest"] = table.digest
    return {
        "artifact_type": ACTION_TABLE_ARTIFACT_TYPE,
        "schema_version": table.schema_version,
        "compatibility": compatibility,
        "payload": payload,
    }


def load_action_table(path: Path) -> ActionTable:
    """Parse and fully verify an action-table artifact; returns the table.

    Recomputes the digest before decoding the semantic payload (SPEC 2.2),
    validates the envelope, template ordering, and cross-checks the declared
    digest against a fresh :func:`build_action_table`.
    """
    raw_bytes = Path(path).read_bytes()
    try:
        document = json.loads(
            raw_bytes,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ContractError(f"{path}: not valid JSON: {exc}") from exc
    except ContractError:
        raise
    return _table_from_document(document, origin=str(path))


def parse_action_table(raw_bytes: bytes) -> ActionTable:
    """Verify action-table artifact bytes (same checks as :func:`load_action_table`)."""
    try:
        document = json.loads(
            raw_bytes,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ContractError(f"not valid JSON: {exc}") from exc
    except ContractError:
        raise
    return _table_from_document(document, origin="<bytes>")


def _table_from_document(document: object, *, origin: str) -> ActionTable:
    if not isinstance(document, Mapping):
        raise ContractError(f"{origin}: artifact must be a JSON object")
    if document.get("artifact_type") != ACTION_TABLE_ARTIFACT_TYPE:
        raise ContractError(
            f"{origin}: artifact_type must be {ACTION_TABLE_ARTIFACT_TYPE!r}, "
            f"got {document.get('artifact_type')!r}"
        )
    compatibility = document.get("compatibility")
    if compatibility != "exact":
        raise ContractError(f"{origin}: unsupported compatibility {compatibility!r}")
    envelope_version = document.get("schema_version")
    if not isinstance(envelope_version, str):
        raise ContractError(f"{origin}: schema_version must be a string")
    if envelope_version.split(".")[0] != ACTION_TABLE_SCHEMA_VERSION.split(".")[0]:
        raise IncompatibleSchemaError(
            f"{origin}: unknown major schema version {envelope_version!r}"
        )
    if envelope_version != ACTION_TABLE_SCHEMA_VERSION:
        raise IncompatibleSchemaError(
            f"{origin}: schema_version {envelope_version!r} newer than supported "
            f"{ACTION_TABLE_SCHEMA_VERSION!r}"
        )
    payload = document.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "actions", "digest"}:
        raise ContractError(f"{origin}: payload must hold exactly schema_version/actions/digest")
    if payload["schema_version"] != envelope_version:
        raise ContractError(
            f"{origin}: payload schema_version {payload['schema_version']!r} != envelope "
            f"{envelope_version!r}"
        )
    declared_digest: object = payload["digest"]
    if not isinstance(declared_digest, str):
        raise ContractError(f"{origin}: digest must be a string")

    raw_actions: object = payload["actions"]
    if not isinstance(raw_actions, list) or len(raw_actions) == 0:
        raise ContractError(f"{origin}: actions must be a non-empty array")
    templates = tuple(_template_from_json(entry, origin=origin)  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
 for entry in raw_actions)  # type: ignore[attr-defined]  # reason: raw_actions isinstance-checked as list above; checker cannot narrow object
    recomputed = compute_table_digest(templates, make_schema_version(envelope_version))
    if not hmac.compare_digest(str(recomputed), str(declared_digest)):
        raise DigestMismatchError(
            f"{origin}: declared digest {declared_digest!r} != recomputed {recomputed!r}"
        )
    table = build_action_table(templates, schema_version=make_schema_version(envelope_version))
    if not hmac.compare_digest(str(table.digest), str(declared_digest)):
        raise DigestMismatchError(
            f"{origin}: rebuilt table digest {table.digest!r} != declared {declared_digest!r}"
        )
    return table

def _template_from_json(entry: object, *, origin: str) -> CanonicalActionTemplate:
    if not isinstance(entry, Mapping) or set(entry) != set(_TEMPLATE_JSON_FIELDS):
        raise ContractError(
            f"{origin}: template entries must hold exactly {list(_TEMPLATE_JSON_FIELDS)}"
        )
    consumed_raw: object = entry["consumed_tiles"]  # type: ignore[index]  # reason: entry Mapping-checked above; checker cannot narrow object index
    if not isinstance(consumed_raw, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in consumed_raw
    ):
        raise ContractError(f"{origin}: consumed_tiles must be an array of ints")
    for name in ("tile", "called_tile"):
        value: object = entry[name]  # type: ignore[index]  # reason: entry Mapping-checked above; checker cannot narrow object index
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ContractError(f"{origin}: {name} must be null or int")
    offset: object = entry["source_offset"]  # type: ignore[index]  # reason: entry Mapping-checked above; checker cannot narrow object index
    if offset is not None and (isinstance(offset, bool) or offset not in (-1, 0, 1, 2)):  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        raise ContractError(f"{origin}: source_offset invalid: {offset!r}")
    kind: object = entry["kind"]  # type: ignore[index]  # reason: entry Mapping-checked above; checker cannot narrow object index
    if not isinstance(kind, str) or kind not in ACTION_KIND_ORDINALS:
        raise ContractError(f"{origin}: unknown template kind {kind!r}")
    for name in ("declares_riichi", "meld_ref_required"):
        if not isinstance(entry[name], bool):
            raise ContractError(f"{origin}: {name} must be a bool")
    return CanonicalActionTemplate(
        kind=kind,  # type: ignore[arg-type]  # reason: kind isinstance-checked as str against ACTION_KIND_ORDINALS above
        tile=entry["tile"],  # type: ignore[arg-type]  # reason: null-or-int checked above; ctor re-validates
        called_tile=entry["called_tile"],  # type: ignore[arg-type]  # reason: null-or-int checked above; ctor re-validates
        consumed_tiles=tuple(consumed_raw),  # type: ignore[arg-type]  # reason: list-of-ints checked above
        source_offset=offset,  # type: ignore[arg-type]  # reason: offset range-checked above
        declares_riichi=entry["declares_riichi"],  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        meld_ref_required=entry["meld_ref_required"],  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
    )
