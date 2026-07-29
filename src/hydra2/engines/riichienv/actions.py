"""Canonical <-> RiichiEnv action mapping and legal-mask construction.

Engine facts used here (verified against 0.4.8 runtime probes):
* ``Observation.legal_actions()`` returns :class:`riichienv.Action` objects;
  DISCARD carries the physical tile, CHI/PON/DAIMINKAN carry called tile +
  consumed hand tiles, ANKAN carries all four physical tiles, KAKAN carries
  the added tile + prior pon triple, RIICHI/KYUSHU_KYUHAI/PASS are
  parameterless, RON/TSUMO carry the winning tile.
* ``Observation.find_action`` is avoided on purpose (unreleased aka tie-break
  fix b3efe3f750); matching is done by exact field comparison instead.
* Canonical-only twins are expanded at the adapter boundary: ``tsumogiri``
  alongside ``discard`` of the drawn tile, and one ``riichi_discard`` per
  engine-valid declaration candidate when RIICHI is offered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import riichienv

from hydra2.contracts.action import ACTION_PHASES, CanonicalAction, canonical_action_codec
from hydra2.contracts.common import (
    ContractError,
    InvalidActionError,
    make_seat,
    make_tile_id,
)
from hydra2.contracts.observation import VisibleMeld, visible_meld_id

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hydra2.contracts.action import ActionContext, ActionKind, ActionTable, JsonValue, Phase
__all__ = ["engine_matches_canonical", "expand_engine_legals", "legal_view"]

_AT = riichienv.ActionType


def _kind_of(engine_type: Any) -> str | None:
    return {
        int(_AT.DISCARD): "discard",
        int(_AT.CHI): "chi",
        int(_AT.PON): "pon",
        int(_AT.DAIMINKAN): "daiminkan",
        int(_AT.RON): "ron",
        int(_AT.RIICHI): "riichi",
        int(_AT.TSUMO): "tsumo",
        int(_AT.PASS): "pass",
        int(_AT.ANKAN): "ankan",
        int(_AT.KAKAN): "kakan",
        int(_AT.KYUSHU_KYUHAI): "abort_nine_terminals",
    }.get(int(cast("Any", engine_type)))


def _action(
    kind: str,
    actor: int,
    *,
    tile: int | None,
    called_tile: int | None,
    consumed: tuple[int, ...],
    source: int | None,
    metadata: tuple[tuple[str, object], ...] = (),
) -> CanonicalAction:
    return CanonicalAction(
        kind=cast("ActionKind", kind),
        actor=make_seat(actor),
        tile=None if tile is None else make_tile_id(tile),
        called_tile=None if called_tile is None else make_tile_id(called_tile),
        consumed_tiles=tuple(make_tile_id(t) for t in consumed),
        source_seat=None if source is None else make_seat(source),
        declares_riichi=(kind == "riichi_discard"),
        metadata=cast("tuple[tuple[str, JsonValue], ...]", metadata),
    )


def canonical_from_engine(
    action: Any,
    *,
    actor: int,
    offered_by: int | None = None,
) -> CanonicalAction:
    """One-to-one translation of an engine legal action slot."""
    action_type: Any = action.action_type
    kind = _kind_of(action_type)
    if kind is None:
        raise InvalidActionError(f"unsupported engine action type {int(cast('Any', action_type))}")
    tile_val: Any = action.tile
    tile: int | None = None if tile_val is None else int(cast("Any", tile_val))
    consume_tiles_val: Any = action.consume_tiles
    consumed: tuple[int, ...] = tuple(
        sorted(int(cast("Any", t)) for t in cast("Any", consume_tiles_val))
    )
    if kind == "discard":
        return _action("discard", actor, tile=tile, called_tile=None, consumed=(), source=None)
    if kind == "chi":
        return _action(
            "chi", actor, tile=None, called_tile=tile, consumed=consumed, source=offered_by
        )
    if kind == "pon":
        return _action(
            "pon", actor, tile=None, called_tile=tile, consumed=consumed, source=offered_by
        )
    if kind == "daiminkan":
        return _action(
            "daiminkan", actor, tile=None, called_tile=tile, consumed=consumed, source=offered_by
        )
    if kind == "ankan":
        return _action("ankan", actor, tile=None, called_tile=None, consumed=consumed, source=None)
    if kind == "kakan":
        return _action("kakan", actor, tile=tile, called_tile=None, consumed=(), source=None)
    if kind == "ron":
        return _action("ron", actor, tile=tile, called_tile=None, consumed=(), source=offered_by)
    if kind == "tsumo":
        return _action("tsumo", actor, tile=tile, called_tile=None, consumed=(), source=None)
    if kind == "pass":
        return _action("pass", actor, tile=None, called_tile=None, consumed=(), source=None)
    if kind == "abort_nine_terminals":
        return _action(
            "abort_nine_terminals", actor, tile=None, called_tile=None, consumed=(), source=None
        )
    raise InvalidActionError(f"engine kind {kind} is expanded by the caller (riichi)")


def expand_engine_legals(
    engine_actions: Any,
    *,
    actor: int,
    phase: Phase,
    drawn_tile: int | None,
    own_hand: Sequence[int],
    offered_by: int | None,
    actor_melds: Sequence[VisibleMeld] = (),
) -> tuple[CanonicalAction, ...]:
    """Full canonical expansion of one seat's engine legal set.

    Adds the canonical-only twins the engine folds together: ``tsumogiri``
    next to ``discard`` of the drawn tile and one ``riichi_discard`` per
    valid declaration candidate when the engine offers RIICHI. Kakan slots
    carry the schema-required prior-pon reference directly (the metadata is
    mandatory at construction). Results are gated through the contract's
    coarse phase table.
    """
    out: list[CanonicalAction] = []
    has_riichi = False
    for engine_action_any in cast("Any", engine_actions):
        engine_action: Any = engine_action_any
        kind = _kind_of(cast("Any", engine_action.action_type))
        if kind == "kakan":
            tile_val: Any = engine_action.tile
            tile: int | None = None if tile_val is None else int(cast("Any", tile_val))
            if tile is None:
                raise InvalidActionError("engine offered a kakan without an added tile")
            added_type: int = tile // 4
            prior = next(
                (
                    meld
                    for meld in actor_melds
                    if meld.kind == "pon" and int(meld.tiles[0]) // 4 == added_type
                ),
                None,
            )
            if prior is None:
                raise ContractError(
                    f"kakan of type {added_type}: no visible prior pon for seat {actor}"
                )
            out.append(
                _action(
                    "kakan",
                    actor,
                    tile=tile,
                    called_tile=None,
                    consumed=(),
                    source=None,
                    metadata=(("prior_pon_meld_id", visible_meld_id(prior)),),
                )
            )
        elif kind == "discard":
            out.append(canonical_from_engine(engine_action, actor=actor))
            tile_raw: Any = engine_action.tile
            if drawn_tile is not None and int(cast("Any", tile_raw)) == drawn_tile:
                out.append(
                    _action(
                        "tsumogiri",
                        actor,
                        tile=drawn_tile,
                        called_tile=None,
                        consumed=(),
                        source=None,
                    )
                )
        elif kind == "riichi":
            has_riichi = True
        else:
            out.append(canonical_from_engine(engine_action, actor=actor, offered_by=offered_by))
    if has_riichi:
        hand_set = set(own_hand)
        candidates_any: Any = riichienv.check_riichi_candidates(list(own_hand))
        for candidate_any in sorted(set(cast("Any", candidates_any))):
            candidate: int = int(cast("Any", candidate_any))
            if candidate in hand_set:
                out.append(
                    _action(
                        "riichi_discard",
                        actor,
                        tile=candidate,
                        called_tile=None,
                        consumed=(),
                        source=None,
                    )
                )
    return tuple(a for a in out if phase in ACTION_PHASES[a.kind])


def engine_matches_canonical(
    engine_action: Any, wanted: CanonicalAction, *, riichi_candidate_tiles: frozenset[int]
) -> bool:
    """Exact-match test between a requested canonical action and an engine slot."""
    kind = _kind_of(cast("Any", engine_action.action_type))
    consume_tiles_val: Any = engine_action.consume_tiles
    consumed: tuple[int, ...] = tuple(
        sorted(int(cast("Any", t)) for t in cast("Any", consume_tiles_val))
    )
    tile_val: Any = engine_action.tile
    tile: int | None = None if tile_val is None else int(cast("Any", tile_val))
    wanted_tile: int | None = None if wanted.tile is None else int(wanted.tile)
    wanted_called: int | None = None if wanted.called_tile is None else int(wanted.called_tile)
    if kind == "discard":
        return wanted.kind in ("discard", "tsumogiri") and wanted_tile == tile
    if kind == "riichi":
        return wanted.kind == "riichi_discard" and wanted_tile in riichi_candidate_tiles
    if wanted.kind == "pass":
        return kind == "pass"
    if wanted.kind == "abort_nine_terminals":
        return kind == "abort_nine_terminals"
    if kind != wanted.kind:
        return False
    if wanted.kind == "ankan":
        return consumed == tuple(wanted.consumed_tiles)
    if wanted.kind == "kakan":
        return tile == wanted_tile
    if wanted.kind in ("chi", "pon", "daiminkan"):
        return tile == wanted_called and consumed == tuple(wanted.consumed_tiles)
    if wanted.kind in ("ron", "tsumo"):
        return tile == wanted_tile
    return False


def legal_view(
    *,
    table: ActionTable,
    context: ActionContext,
    engine_actions: Any,
    drawn_tile: int | None,
    own_hand: Sequence[int],
    melds_of_actor: Sequence[VisibleMeld],
    offered_by: int | None,
) -> tuple[tuple[CanonicalAction, ...], tuple[bool, ...]]:
    """Sorted canonical actions plus their mask over the published table."""
    expanded = expand_engine_legals(
        cast("Any", engine_actions),
        actor=int(context.actor),
        phase=context.phase,
        drawn_tile=drawn_tile,
        own_hand=own_hand,
        offered_by=offered_by,
        actor_melds=melds_of_actor,
    )
    encoded: dict[int, CanonicalAction] = {}
    for action in expanded:
        action_id = canonical_action_codec.encode(action, table=table, context=context)
        encoded[int(action_id)] = action
    mask = [False] * len(table.actions)
    ordered: list[CanonicalAction] = []
    for action_id in sorted(encoded):
        mask[action_id] = True
        ordered.append(encoded[action_id])
    return tuple(ordered), tuple(mask)
