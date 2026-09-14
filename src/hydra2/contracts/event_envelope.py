"""SPEC 7.1 envelopes — payload, delta, visibility, and stream grammar.

Owns the routed event dataclasses with every validation bullet: the
visibility<->visible_to matrix, strictly increasing stream sequences, the
per-kind payload composition matrix, stream grammar transitions, and the
actor projection helpers with envelope identity digests. Unknown payload
data and undeclared delta paths are rejected here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from hydra2.contracts.canonical import canonical_json_bytes
from hydra2.contracts.common import (
    ActionId,
    ContractError,
    DigestText,
    Seat,
    SequenceNo,
    TileId,
    VisibilityViolationError,
    make_digest_text,
    make_seat,
)
from hydra2.contracts.event_vocab import (
    _PUBLIC_VISIBLE_TO,
    DELTA_OPERATIONS,
    EVENT_KINDS,
    PAYLOAD_SCALAR_FIELDS,
    PAYLOAD_TUPLE_FIELDS,
    VISIBILITIES,
    DeltaOperation,
    EventKind,
    Visibility,
    _action_tuple,
    _optional_action,
    _optional_seat,
    _optional_tile,
    _require_enum,
    _require_plain_int,
    _require_str,
    _score_quad,
    _tile_tuple,
    _validate_delta_path,
    _validate_delta_value,
)

__all__ = [
    "EventEnvelope",
    "EventPayload",
    "PublicStateDelta",
    "envelope_digest",
    "envelope_identity_document",
    "filter_events_for_actor",
    "validate_event_stream",
    "visible_to_actor",
]


@dataclass(frozen=True, slots=True)
class PublicStateDelta:
    """One public-state mutation carried by an :class:`EventEnvelope`."""

    path: tuple[str | int, ...]
    operation: DeltaOperation
    value: object  # JsonValue

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "path",
            _validate_delta_path(self.path),
        )
        object.__setattr__(
            self,
            "operation",
            _require_enum(self.operation, name="operation", allowed=DELTA_OPERATIONS),
        )
        _validate_delta_value(self.path, self.operation, self.value)

    def to_json(self) -> dict[str, object]:
        return {"path": list(self.path), "operation": self.operation, "value": self.value}


@dataclass(frozen=True, slots=True)
class EventPayload:
    """Closed payload of one event; unknown fields are unrepresentable (slots)."""

    kind: EventKind
    actor: Seat | None
    tile: TileId | None
    action_id: ActionId | None
    source_seat: Seat | None
    consumed_tiles: tuple[TileId, ...]
    offered_action_ids: tuple[ActionId, ...]
    accepted_action_ids: tuple[ActionId, ...]
    round_index: int | None
    scores: tuple[int, int, int, int] | None
    reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _require_enum(self.kind, name="kind", allowed=EVENT_KINDS))
        object.__setattr__(self, "actor", _optional_seat(self.actor, name="actor"))
        object.__setattr__(self, "tile", _optional_tile(self.tile, name="tile"))
        object.__setattr__(self, "action_id", _optional_action(self.action_id, name="action_id"))
        object.__setattr__(
            self, "source_seat", _optional_seat(self.source_seat, name="source_seat")
        )
        object.__setattr__(
            self,
            "consumed_tiles",
            _tile_tuple(
                self.consumed_tiles if self.consumed_tiles is not None else (),
                name="consumed_tiles",
            ),
        )
        object.__setattr__(
            self,
            "offered_action_ids",
            _action_tuple(
                self.offered_action_ids if self.offered_action_ids is not None else (),
                name="offered_action_ids",
            ),
        )
        object.__setattr__(
            self,
            "accepted_action_ids",
            _action_tuple(
                self.accepted_action_ids if self.accepted_action_ids is not None else (),
                name="accepted_action_ids",
            ),
        )
        if self.round_index is not None:
            object.__setattr__(
                self,
                "round_index",
                _require_plain_int(self.round_index, name="round_index", minimum=0, maximum=None),
            )
        if self.scores is not None:
            object.__setattr__(self, "scores", _score_quad(self.scores, name="scores"))
        if self.reason is not None:
            object.__setattr__(self, "reason", _require_str(self.reason, name="reason"))

    def field(self, name: str) -> object:
        """Payload field accessor used by the schema matrix validator."""
        if name == "actor":
            return self.actor
        if name in PAYLOAD_SCALAR_FIELDS:
            return getattr(self, name)
        if name in PAYLOAD_TUPLE_FIELDS:
            return getattr(self, name)
        raise ContractError(f"unknown payload field {name!r}")

    def to_json(self) -> dict[str, object]:
        document: dict[str, object] = {"kind": self.kind, "actor": _maybe_int(self.actor)}
        document["tile"] = _maybe_int(self.tile)
        document["action_id"] = _maybe_int(self.action_id)
        document["source_seat"] = _maybe_int(self.source_seat)
        document["consumed_tiles"] = [_maybe_int(t) for t in self.consumed_tiles]
        document["offered_action_ids"] = [_maybe_int(a) for a in self.offered_action_ids]
        document["accepted_action_ids"] = [_maybe_int(a) for a in self.accepted_action_ids]
        document["round_index"] = self.round_index
        document["scores"] = None if self.scores is None else [_maybe_int(s) for s in self.scores]
        document["reason"] = self.reason
        return document


def _maybe_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[arg-type]  # reason: None filtered by ternary; int() validates


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """Routed, sequenced event with visibility and public-state effects."""

    game_id: str
    sequence: SequenceNo
    kind: EventKind
    actor: Seat | None
    visibility: Visibility
    visible_to: tuple[Seat, ...]
    payload: EventPayload
    public_delta: tuple[PublicStateDelta, ...]
    rules_hash: DigestText
    schema_hash: DigestText

    def __post_init__(self) -> None:
        object.__setattr__(self, "game_id", _require_str(self.game_id, name="game_id"))
        if self.game_id == "":
            raise ContractError("game_id must be non-empty")
        object.__setattr__(self, "actor", _optional_seat(self.actor, name="actor"))
        object.__setattr__(
            self,
            "visibility",
            _require_enum(self.visibility, name="visibility", allowed=VISIBILITIES),
        )
        if isinstance(self.visible_to, (str, bytes)) or not isinstance(self.visible_to, Sequence):
            raise ContractError("visible_to must be a sequence of seats")
        seats = tuple(make_seat(int(s)) for s in self.visible_to)
        if seats != tuple(sorted(set(seats))):
            raise ContractError("visible_to must be strictly ascending with unique seats")
        object.__setattr__(self, "visible_to", seats)
        if not isinstance(self.payload, EventPayload):
            raise ContractError("payload must be an EventPayload")
        if self.payload.kind != self.kind:
            raise ContractError(
                f"envelope kind {self.kind!r} disagrees with payload kind {self.payload.kind!r}"
            )
        if self.payload.actor != self.actor:
            raise ContractError("envelope actor disagrees with payload actor")
        if isinstance(self.public_delta, (str, bytes)) or not isinstance(
            self.public_delta, Sequence
        ):
            raise ContractError("public_delta must be a sequence of PublicStateDelta")
        deltas = []
        for delta in self.public_delta:
            if not isinstance(delta, PublicStateDelta):
                raise ContractError("public_delta entries must be PublicStateDelta")
            deltas.append(delta)
        object.__setattr__(self, "public_delta", tuple(deltas))
        object.__setattr__(self, "rules_hash", make_digest_text(self.rules_hash))
        object.__setattr__(self, "schema_hash", make_digest_text(self.schema_hash))
        _validate_visibility_matrix(self)
        _validate_kind_shape(self)

    def to_json(self) -> dict[str, object]:
        return {
            "game_id": self.game_id,
            "sequence": int(self.sequence),
            "kind": self.kind,
            "actor": _maybe_int(self.actor),
            "visibility": self.visibility,
            "visible_to": [int(s) for s in self.visible_to],
            "payload": self.payload.to_json(),
            "public_delta": [delta.to_json() for delta in self.public_delta],
            "rules_hash": self.rules_hash,
            "schema_hash": self.schema_hash,
        }

    def __repr__(self) -> str:  # pragma: no cover - exercised via leak tests
        return (
            f"EventEnvelope(game_id={self.game_id!r}, sequence={int(self.sequence)}, "
            f"kind={self.kind!r}, actor={_maybe_int(self.actor)}, "
            f"visibility={self.visibility!r}, visible_to={[int(s) for s in self.visible_to]}, "
            f"payload_sha256={envelope_digest(self)})"
        )


def _validate_visibility_matrix(envelope: EventEnvelope) -> None:
    """SPEC 7.1 bullet 1-3: visibility <-> visible_to matrix."""
    visibility = envelope.visibility
    seen = envelope.visible_to
    if visibility == "public":
        if tuple(int(s) for s in seen) != _PUBLIC_VISIBLE_TO:
            raise ContractError("public events are visible to exactly (0, 1, 2, 3)")
    elif visibility == "actor_private":
        if len(seen) != 1:
            raise ContractError("actor_private events must name exactly one seat")
        if envelope.kind == "draw_tile" and int(seen[0]) != int(envelope.payload.actor):  # type: ignore[index]  # reason: visible_to statically object; length checked above, int() validates
            raise VisibilityViolationError("draw_tile is actor-private to the drawing seat itself")
    else:  # server_private
        if len(seen) > 0:
            raise VisibilityViolationError("server_private events must have empty visible_to")


_CALL_KINDS = ("chi", "pon", "daiminkan", "ankan", "kakan")


def _validate_kind_shape(envelope: EventEnvelope) -> None:
    """SPEC 7.1 bullets 4-8: per-kind payload composition (schema v1 core)."""
    kind = envelope.kind
    payload = envelope.payload
    visibility = envelope.visibility

    def _forbid(*names: str) -> None:
        for name in names:
            value = payload.field(name)
            if isinstance(value, tuple):
                if len(value) > 0:
                    raise ContractError(f"{kind}: {name} must be empty")
            elif value is not None:
                raise ContractError(f"{kind}: {name} must be null")

    def _require(*names: str) -> None:
        for name in names:
            if name == "actor":
                if payload.actor is None:
                    raise ContractError(f"{kind}: actor is required")
            elif payload.field(name) is None:
                raise ContractError(f"{kind}: {name} is required")

    if kind == "turn_advance":
        if visibility != "public":
            raise ContractError("turn_advance is public")
        _require("actor")
        _forbid(*PAYLOAD_SCALAR_FIELDS, *PAYLOAD_TUPLE_FIELDS)
    elif kind == "draw_tile":
        if visibility != "actor_private" or tuple(int(s) for s in envelope.visible_to) != (
            int(payload.actor),  # type: ignore[arg-type]  # reason: payload.actor statically object; int() validates
        ):
            raise VisibilityViolationError(
                "draw_tile must be actor_private addressed to the drawing actor only"
            )
        _require("actor", "tile")
        _forbid(
            "action_id", "source_seat", "round_index", "scores", "reason", *PAYLOAD_TUPLE_FIELDS
        )
    elif kind == "discard":
        if visibility != "public":
            raise ContractError("discard is public")
        _require("actor", "tile", "action_id")
        _forbid("source_seat", "round_index", "scores", "reason", *PAYLOAD_TUPLE_FIELDS)
    elif kind in ("riichi_declared",):
        if visibility != "public":
            raise ContractError("riichi_declared is public")
        _require("actor", "tile", "action_id")
        _forbid("source_seat", "round_index", "scores", "reason", *PAYLOAD_TUPLE_FIELDS)
    elif kind == "riichi_accepted":
        if visibility != "public":
            raise ContractError("riichi_accepted is public")
        _require("actor")
        _forbid(
            "tile",
            "action_id",
            "source_seat",
            "round_index",
            "scores",
            "reason",
            *PAYLOAD_TUPLE_FIELDS,
        )
    elif kind == "call_window":
        if visibility != "public":
            raise ContractError("call_window is public")
        _forbid("actor", *PAYLOAD_SCALAR_FIELDS, *PAYLOAD_TUPLE_FIELDS)
    elif kind == "call_resolved":
        if visibility != "server_private" or len(envelope.visible_to) > 0:
            raise VisibilityViolationError(
                "call_resolved with full offer sets is server_private to no seat"
            )
        offered = payload.offered_action_ids
        accepted = payload.accepted_action_ids
        if len(offered) > 0:
            if len(set(offered)) != len(offered):
                raise ContractError("call_resolved offered_action_ids must be distinct")
            if len(accepted) != 1 or accepted[0] not in offered:
                raise ContractError(
                    "call_resolved accepts exactly one offered action id (D-WP02D-2)"
                )
        elif len(accepted) > 0:
            raise ContractError("call_resolved pass resolution accepts no actions")
        _forbid(
            "tile", "action_id", "source_seat", "round_index", "scores", "reason", "consumed_tiles"
        )
        _forbid("actor")
    elif kind in ("chi", "pon"):
        if visibility != "public":
            raise ContractError(f"{kind} is public")
        _require("actor", "tile", "action_id", "source_seat")
        if len(payload.consumed_tiles) != 2:
            raise ContractError(f"{kind} consumes exactly two hand tiles")
        _forbid("round_index", "scores", "reason", "offered_action_ids", "accepted_action_ids")
    elif kind == "daiminkan":
        if visibility != "public":
            raise ContractError("daiminkan is public")
        _require("actor", "tile", "action_id", "source_seat")
        if len(payload.consumed_tiles) != 3:
            raise ContractError("daiminkan consumes exactly three hand tiles")
        _forbid("round_index", "scores", "reason", "offered_action_ids", "accepted_action_ids")
    elif kind == "ankan":
        if visibility != "public":
            raise ContractError("ankan is public")
        _require("actor", "action_id")
        if len(payload.consumed_tiles) != 4:
            raise ContractError("ankan consumes exactly four concealed tiles")
        _forbid(
            "tile",
            "source_seat",
            "round_index",
            "scores",
            "reason",
            "offered_action_ids",
            "accepted_action_ids",
        )
    elif kind == "kakan":
        if visibility != "public":
            raise ContractError("kakan is public")
        _require("actor", "tile", "action_id")
        _forbid(
            "source_seat",
            "consumed_tiles",
            "round_index",
            "scores",
            "reason",
            "offered_action_ids",
            "accepted_action_ids",
        )
    elif kind == "dora_revealed":
        if visibility != "public":
            raise ContractError("dora_revealed is public")
        _require("tile")
        _forbid(
            "actor",
            "action_id",
            "source_seat",
            "round_index",
            "scores",
            "reason",
            *PAYLOAD_TUPLE_FIELDS,
        )
    elif kind == "ron":
        if visibility != "public":
            raise ContractError("ron is public")
        _require("actor", "tile", "action_id", "source_seat")
        _forbid(
            "consumed_tiles",
            "offered_action_ids",
            "accepted_action_ids",
            "round_index",
            "scores",
            "reason",
        )
    elif kind == "tsumo":
        if visibility != "public":
            raise ContractError("tsumo is public")
        _require("actor", "tile", "action_id")
        _forbid(
            "source_seat",
            "consumed_tiles",
            "offered_action_ids",
            "accepted_action_ids",
            "round_index",
            "scores",
            "reason",
        )
    elif kind == "game_start":
        if visibility != "public":
            raise ContractError("game_start is public")
        _require("round_index", "scores")
        _forbid("actor", "tile", "action_id", "source_seat", "reason", *PAYLOAD_TUPLE_FIELDS)
    elif kind == "round_start":
        if visibility != "public":
            raise ContractError("round_start is public")
        _require("actor", "round_index", "scores")
        _forbid("tile", "action_id", "source_seat", "reason", *PAYLOAD_TUPLE_FIELDS)
    elif kind == "draw_end":
        if visibility != "public":
            raise ContractError("draw_end is public")
        _require("scores", "reason")
        _forbid("actor", "tile", "action_id", "source_seat", "round_index", *PAYLOAD_TUPLE_FIELDS)
    elif kind == "abortive_draw":
        if visibility != "public":
            raise ContractError("abortive_draw is public")
        _require("round_index", "scores", "reason")
        _forbid("actor", "tile", "action_id", "source_seat", *PAYLOAD_TUPLE_FIELDS)
    elif kind == "round_end":
        if visibility != "public":
            raise ContractError("round_end is public")
        _require("round_index", "scores")
        _forbid("actor", "tile", "action_id", "source_seat", "reason", *PAYLOAD_TUPLE_FIELDS)
    elif kind == "game_end":
        if visibility != "public":
            raise ContractError("game_end is public")
        _require("round_index", "scores", "reason")
        _forbid("actor", "tile", "action_id", "source_seat", *PAYLOAD_TUPLE_FIELDS)
    else:  # pragma: no cover - EVENT_KINDS is exhaustively handled above
        raise ContractError(f"unhandled event kind {kind!r}")

    from hydra2.contracts.event_schema import EVENT_SCHEMA_ROWS

    for delta in envelope.public_delta:
        allowed_paths = EVENT_SCHEMA_ROWS[kind].allowed_delta_paths
        if delta.path not in allowed_paths:
            raise ContractError(f"{kind}: undeclared public-state delta path {list(delta.path)}")


# ---------------------------------------------------------------------------
# Stream validation: monotonic sequences + closed predecessor/successor grammar.
# ---------------------------------------------------------------------------

FIRST_EVENT_KIND: EventKind = "game_start"


def validate_event_stream(events: Sequence[EventEnvelope]) -> None:
    """Strictly increasing sequences, grammar transitions, single game."""
    from hydra2.contracts.event_schema import EVENT_SCHEMA_ROWS

    if len(events) == 0:
        return
    previous: EventEnvelope | None = None
    for event in events:
        if not isinstance(event, EventEnvelope):
            raise ContractError("stream entries must be EventEnvelope instances")
        if previous is not None:
            if event.sequence <= previous.sequence:
                raise ContractError(
                    f"sequence {int(event.sequence)} does not strictly increase past "
                    f"{int(previous.sequence)}"
                )
            if event.game_id != previous.game_id:
                raise ContractError("stream mixes games")
            row = EVENT_SCHEMA_ROWS[event.kind]
            if previous.kind not in row.predecessors:
                raise ContractError(f"{event.kind} cannot follow {previous.kind}")
        elif event.kind != FIRST_EVENT_KIND:
            raise ContractError(f"streams begin with {FIRST_EVENT_KIND!r}")
        previous = event


def visible_to_actor(event: EventEnvelope, actor: Seat) -> bool:
    """Whether ``event`` may enter ``actor``'s history (server_private: never)."""
    actor_seat = make_seat(int(actor))
    if event.visibility == "public":
        return True
    if event.visibility == "actor_private":
        return len(event.visible_to) == 1 and int(event.visible_to[0]) == int(actor_seat)
    return False


def filter_events_for_actor(
    events: Sequence[EventEnvelope], actor: Seat
) -> tuple[EventEnvelope, ...]:
    """Projection of a stream onto what ``actor`` may legitimately hold.

    Server-private events are dropped here; they can never be serialized into
    any actor history (SPEC 7.1 bullet 3).
    """
    actor_seat = make_seat(int(actor))
    return tuple(event for event in events if visible_to_actor(event, actor_seat))


def envelope_identity_document(envelope: EventEnvelope) -> dict[str, object]:
    """Canonical identity document of an envelope (all fields)."""
    return envelope.to_json()


def envelope_digest(envelope: EventEnvelope) -> DigestText:
    """sha256 over RFC 8785 canonical bytes of the whole envelope."""
    identity = canonical_json_bytes(envelope_identity_document(envelope))
    return DigestText("sha256:" + hashlib.sha256(identity).hexdigest())
