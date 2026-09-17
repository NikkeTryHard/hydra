"""SPEC 6 action records — validated action, template, and census.

Owns the validated record layer: the context-free ``CanonicalAction`` with
its SPEC 6.2 structural invariants, the actor-independent
``CanonicalActionTemplate`` with its lexicographic order, and the census
enumerating every structurally valid template once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.contracts.action_kinds import (
    _SOURCE_OFFSETS_CLAIM,
    ACTION_KIND_ORDINALS,
    ActionKind,
    JsonValue,
    _all_same_type,
    _consumed_pair_forms_run,
    _require_bool,
    _tile_type,
    _validated_metadata,
)
from hydra2.contracts.common import (
    ContractError,
    InvalidActionError,
    Seat,
    TileId,
)

__all__ = [
    "CanonicalAction",
    "CanonicalActionTemplate",
    "generate_action_templates",
    "template_sort_key",
]

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
        object.__setattr__(self, "actor", _bridge_contracts.make_seat(self.actor))
        tile = None if self.tile is None else _bridge_contracts.make_tile_id(self.tile)
        called = (
            None if self.called_tile is None else _bridge_contracts.make_tile_id(self.called_tile)
        )
        source = None if self.source_seat is None else _bridge_contracts.make_seat(self.source_seat)
        consumed = tuple(_bridge_contracts.make_tile_id(t) for t in self.consumed_tiles)
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
        tile = None if self.tile is None else _bridge_contracts.make_tile_id(self.tile)
        called = (
            None if self.called_tile is None else _bridge_contracts.make_tile_id(self.called_tile)
        )
        consumed = tuple(_bridge_contracts.make_tile_id(t) for t in self.consumed_tiles)
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


def _require_census_bridge() -> object:
    """Resolve the census bridge, fail closed when not built (no oracle fallback)."""
    try:
        from hydra2_replay_rs import contracts as bridge  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2 census authority requires the hydra2_replay_rs bridge; "
            "run `pixi run build-ext` to build the extension before use"
        ) from exc
    if not hasattr(bridge, "action_census"):
        raise ImportError(
            "hydra2_replay_rs.contracts submodule missing (stale .so); "
            "rebuild the bridge (`pixi run build-ext`)"
        )
    return bridge


def generate_action_templates() -> tuple[CanonicalActionTemplate, ...]:
    """Enumerate every and only structurally valid template once (SPEC 6.3).

    Census (analytic): pass 4, discard/tsumogiri/riichi_discard 136 each,
    chi 4032, pon 1224, daiminkan 408, ankan 34, kakan 136, ron 408,
    tsumo 136, both abort kinds 1 each => 6792 templates.

    Cutover (shrink end-state): the enumeration loops live in the
    ``hydra-feed`` census authority; this is a thin hard-Rust delegate that
    converts the bridge ``action_census`` records to
    :class:`CanonicalActionTemplate` (re-validated on construction).
    ``ImportError`` raises with a ``build-ext`` hint — NO oracle fallback.
    Generation order is the bridge order (bit-identical to the retired
    ``template_sort_key`` sort; verified by the WP-02C goldens).
    """
    bridge = _require_census_bridge()
    records = bridge.action_census()  # type: ignore[attr-defined]
    return tuple(
        CanonicalActionTemplate(
            kind=record.kind,
            tile=record.tile,
            called_tile=record.called_tile,
            consumed_tiles=tuple(TileId(t) for t in record.consumed_tiles),
            source_offset=record.source_offset,
            declares_riichi=bool(record.declares_riichi),
            meld_ref_required=bool(record.meld_ref_required),
        )
        for record in records
    )
