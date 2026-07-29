"""Canonical event-envelope construction helpers for the RiichiEnv adapter.

The adapter orchestrates translation (it owns cursor/permutation state); this
module supplies validated constructors, the ryukyoku-reason mapping, and
public-state delta builders so every emitted envelope passes the WP-02D
schema grammar by construction.

Owner decision D-WP03A-5: RiichiEnv resolves claims from ONE simultaneous
step over all responders, so the adapter buffers responder decisions and
emits ``call_window`` when a window opens and the single ``call_resolved``
plus outcome envelopes when the buffered set completes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from hydra2.contracts.common import (
    Seat,
    UnsupportedRuleError,
    make_action_id,
    make_digest_text,
    make_seat,
    make_sequence_no,
    make_tile_id,
)
from hydra2.contracts.event import (
    EVENT_KINDS,
    EventEnvelope,
    EventPayload,
    PublicStateDelta,
    _require_enum,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hydra2.contracts.event import DeltaOperation, EventKind, Visibility

__all__ = [
    "ABORTIVE_REASONS",
    "DRAW_END_REASONS",
    "make_delta",
    "make_envelope",
    "meld_delta_value",
    "reason_kind",
]

#: ryukyoku reasons that end the hand with tenpai payments (exhaustive draw).
DRAW_END_REASONS: frozenset[str] = frozenset({"exhaustive_draw", "nagashi_mangan"})

#: Abortive-draw reasons mapped onto the rules manifest literals.
ABORTIVE_REASONS: dict[str, str] = {
    "kyushu_kyuhai": "kyuushu_kyuuhai",
    "kyuushu_kyuuhai": "kyuushu_kyuuhai",
    "suucha_riichi": "suucha_riichi",
    "sanchaho": "sanchahou",
    "sanchahou": "sanchahou",
    "suukaikan": "suukaikan",
    "suukansansen": "suukaikan",
    "suufon_renda": "suufon_renda",
    "sufuurenta": "suufon_renda",
}

_MELD_KIND_BY_ENGINE_EVENT = {"chi": "chi", "pon": "pon", "daiminkan": "daiminkan"}


def reason_kind(reason: str) -> str:
    """Classify an engine ryukyoku reason; unknown values are hard failures."""
    if reason in DRAW_END_REASONS:
        return "draw_end"
    if reason in ABORTIVE_REASONS:
        return "abortive_draw"
    raise UnsupportedRuleError(
        f"unmapped ryukyoku reason {reason!r}; persisting counterexample "
        f"(D-WP03A-7): engine reports a hand termination the v1 event grammar "
        f"cannot classify"
    )


def make_delta(path: tuple[str | int, ...], operation: str, value: object) -> PublicStateDelta:
    return PublicStateDelta(path=path, operation=cast("DeltaOperation", operation), value=value)


def meld_delta_value(
    *,
    kind: str,
    owner: int,
    source_seat: int | None,
    called_tile: int | None,
    tiles: Sequence[int],
) -> dict[str, object]:
    """Meld document matching the schema's declared meld object keys."""
    ordered = sorted(t for t in tiles)
    meld_id = f"{kind}:" + ".".join(str(t) for t in ordered)
    return {
        "meld_id": meld_id,
        "kind": _MELD_KIND_BY_ENGINE_EVENT.get(kind, kind),
        "owner": owner,
        "source_seat": None if source_seat is None else source_seat,
        "called_tile": None if called_tile is None else called_tile,
        "tiles": ordered,
    }


def make_envelope(
    *,
    game_id: str,
    sequence: int,
    kind: str,
    visibility: Visibility,
    rules_hash: str,
    schema_hash: str,
    actor: int | None = None,
    tile: int | None = None,
    action_id: int | None = None,
    source_seat: int | None = None,
    consumed_tiles: Sequence[int] = (),
    offered_action_ids: Sequence[int] = (),
    accepted_action_ids: Sequence[int] = (),
    round_index: int | None = None,
    scores: Sequence[int] | None = None,
    reason: str | None = None,
    public_delta: Sequence[PublicStateDelta] = (),
) -> EventEnvelope:
    """Single validated envelope constructor used by the whole adapter.

    NewType boundaries are normalized here so every call site passes plain
    ints/strs; the contracts' own validation remains authoritative.
    """
    if visibility == "public":
        visible_to: tuple[Seat, ...] = (Seat(0), Seat(1), Seat(2), Seat(3))
    elif visibility == "actor_private" and actor is not None:
        visible_to = (make_seat(actor),)
    else:
        visible_to = ()
    checked_kind = cast("EventKind", _require_enum(kind, name="kind", allowed=EVENT_KINDS))
    payload = EventPayload(
        kind=checked_kind,
        actor=None if actor is None else make_seat(actor),
        tile=None if tile is None else make_tile_id(tile),
        action_id=None if action_id is None else make_action_id(action_id),
        source_seat=None if source_seat is None else make_seat(source_seat),
        consumed_tiles=tuple(make_tile_id(t) for t in consumed_tiles),
        offered_action_ids=tuple(make_action_id(a) for a in offered_action_ids),
        accepted_action_ids=tuple(make_action_id(a) for a in accepted_action_ids),
        round_index=round_index,
        scores=None
        if scores is None
        else cast("tuple[int, int, int, int]", tuple(s for s in scores)),
        reason=reason,
    )
    return EventEnvelope(
        game_id=game_id,
        sequence=make_sequence_no(sequence),
        kind=checked_kind,
        actor=None if actor is None else make_seat(actor),
        visibility=visibility,
        visible_to=tuple(make_seat(int(s)) for s in visible_to),
        payload=payload,
        public_delta=tuple(public_delta),
        rules_hash=make_digest_text(rules_hash),
        schema_hash=make_digest_text(schema_hash),
    )
