"""SPEC 6.3 action table and bidirectional codec.

Owns the indexed vocabulary and its gates: the generation-ordered table,
the context carrying phase/offer/ownership state, the shared contextual
validator, and the bidirectional codec bound to a pinned table digest.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import cast

from hydra2.contracts.action_kinds import (
    ACTION_PHASES,
    CLAIM_KINDS,
    JsonValue,
    _tile_type,
)
from hydra2.contracts.action_model import (
    CanonicalAction,
    CanonicalActionTemplate,
    SourceOffset,
    template_sort_key,
)
from hydra2.contracts.common import (
    ActionId,
    ContractError,
    DigestText,
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

try:
    from hydra2_replay_rs import contracts as bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    bridge = None  # type: ignore[assignment]

__all__ = [
    "ActionContext",
    "ActionTable",
    "CanonicalActionCodec",
    "canonical_action_codec",
]


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
        if bridge is None:
            raise ImportError(
                "hydra2 census authority requires the hydra2_replay_rs bridge; "
                "run `pixi run build-ext` to build the extension before use"
            )
        try:
            result = bridge.census_index_of(  # type: ignore[attr-defined]
                template.kind,
                list(template.consumed_tiles),
                tile=template.tile,
                called_tile=template.called_tile,
                source_offset=template.source_offset,
            )
        except ValueError as exc:
            raise ContractError(f"index_of probe rejected: {exc}") from exc
        if result is None:
            return None
        index = int(result)
        # Canonical table IS the census: bridge and table indices coincide.
        # Non-canonical subsets (no current caller; ``build_action_table`` with
        # explicit actions only) keep the retired bisect so a present template
        # still resolves to its table position instead of misreporting None.
        if 0 <= index < len(self.actions) and self.actions[index] == template:
            return index
        if len(self.actions) != 6792:
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


def _offset_from_source(source: Seat | None, actor: Seat) -> SourceOffset | None:
    """Relative source offset seen from actor (thin bridge translator).

    The gate lives in ``hydra2_replay_rs.contracts.offset_from_source``;
    bridge rejections (ValueError/TypeError) surface as InvalidActionError
    so the codec error contract is unchanged. None in, None out.
    """
    if source is None:
        return None
    if bridge is None:
        raise ImportError(
            "hydra2 codec authority requires the hydra2_replay_rs bridge; "
            "run `pixi run build-ext` to build the extension before use"
        )
    try:
        result = bridge.offset_from_source(source, actor)  # type: ignore[attr-defined]
    except (ValueError, TypeError) as exc:
        raise InvalidActionError(f"source offset rejected: {exc}") from exc
    if result is None:
        return None
    return cast("SourceOffset", int(result))


def _resolve_source(offset: SourceOffset | None, actor: Seat) -> Seat | None:
    """Absolute source seat seen from actor (thin bridge translator).

    The gate lives in ``hydra2_replay_rs.contracts.resolve_source``;
    bridge rejections surface as InvalidActionError so the codec error
    contract is unchanged. None in, None out.
    """
    if offset is None:
        return None
    if bridge is None:
        raise ImportError(
            "hydra2 codec authority requires the hydra2_replay_rs bridge; "
            "run `pixi run build-ext` to build the extension before use"
        )
    try:
        result = bridge.resolve_source(offset, actor)  # type: ignore[attr-defined]
    except (ValueError, TypeError) as exc:
        raise InvalidActionError(f"source resolve rejected: {exc}") from exc
    if result is None:
        return None
    return make_seat(int(result))


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
