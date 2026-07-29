"""SPEC 7 events, packets, and visibility — WP-02D contract module.

This module owns:

- SPEC 7.1 ``Visibility``/``EventKind`` unions, ``EventPayload``,
  ``PublicStateDelta``, and ``EventEnvelope`` with every validation bullet:
  visibility<->visible_to matrix (public == (0,1,2,3); actor_private == exactly
  one seat, for ``draw_tile`` the actor itself; server_private == empty
  ``visible_to`` and never serializable into any actor history), strictly
  increasing sequences across a stream, ``turn_advance`` public and tile-free,
  ``draw_tile`` actor-private carrying the exact physical tile, public
  discard/call events carrying only revealed tiles, and call-resolution
  completeness for exactly one unambiguous successor.
- The versioned closed ``EventSchema`` artifact document for
  ``configs/contracts/event_schema_v1.json`` (per-kind required-null/non-null
  payload fields, visibility, actor constraint, allowed ``PublicStateDelta``
  paths/operations/value types, ordering predecessor/successor sets). Unknown
  payload data and undeclared delta paths are rejected.
- SPEC 7.2 ``ActorVisiblePacket`` with
  ``packet_id = sha256(canonical bytes excluding packet_id)`` and the
  ``PacketBoundarySpec`` published at ``configs/contracts/packet_boundary_v1.json``.

Boundary notes:
- Contracts import stdlib and sibling contract modules only (SPEC 1); RFC 8785
  identity bytes come from :func:`hydra2.contracts.canonical.canonical_json_bytes`
  (moved verbatim out of action.py in the WP-02D cutover), byte-equal to the
  WP-02A authority (pinned by tests).
- Owner decision D-WP02D-1: ``call_window`` is a public, tile-free marker; the
  offered responses reach seats exclusively through their legal masks. The
  complete offered/priority outcome travels in the server_private
  ``call_resolved`` envelope; actor streams receive only the legally visible
  consequences (the public chi/pon/daiminkan/ron events, or the next public
  ``turn_advance`` when every seat passed).
- Owner decision D-WP02D-2: a nonempty ``call_resolved`` resolution accepts
  exactly one action id (one unambiguous successor). Multi-winner settlements
  decompose into consecutive resolutions with strictly increasing sequences.
- Debug representations never embed payload contents: ``EventEnvelope.__repr__``
  exposes only routing facts plus a payload digest, so server_private data
  cannot leak into logs (BUILD adversarial checklist).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from hydra2.contracts.canonical import canonical_json_bytes
from hydra2.contracts.common import (
    ActionId,
    ContractError,
    DigestText,
    PacketId,
    Seat,
    SequenceNo,
    TileId,
    VisibilityViolationError,
    make_action_id,
    make_digest_text,
    make_packet_id,
    make_seat,
    make_sequence_no,
    make_tile_id,
)

__all__ = [
    "ACCEPTED_CONSTRAINT",
    "DEFAULT_PACKET_BOUNDARY_SPEC",
    "DELTA_OPERATIONS",
    "DELTA_PATH_VOCABULARY",
    "EVENT_KINDS",
    "EVENT_SCHEMA_ARTIFACT_TYPE",
    "EVENT_SCHEMA_RELPATH",
    "EVENT_SCHEMA_ROWS",
    "EVENT_SCHEMA_SCHEMA_VERSION",
    "PACKET_BOUNDARY_ARTIFACT_TYPE",
    "PACKET_BOUNDARY_RELPATH",
    "PACKET_BOUNDARY_SCHEMA_VERSION",
    "VISIBILITIES",
    "ActorVisiblePacket",
    "EventEnvelope",
    "EventPayload",
    "EventSchemaRow",
    "PacketBoundarySpec",
    "PublicStateDelta",
    "Visibility",
    "build_event_schema_envelope",
    "build_event_schema_payload",
    "build_packet_boundary_envelope",
    "build_packet_boundary_payload",
    "compute_event_schema_digest",
    "compute_packet_id",
    "envelope_digest",
    "envelope_identity_document",
    "filter_events_for_actor",
    "load_event_schema",
    "load_packet_boundary_spec",
    "make_actor_visible_packet",
    "parse_event_schema",
    "partition_actor_packets",
    "public_state_chain_hash",
    "validate_event_stream",
    "validate_packet_partition",
    "visible_to_actor",
]

# ---------------------------------------------------------------------------
# SPEC 7.1 - Unions.
# ---------------------------------------------------------------------------

Visibility = Literal["public", "actor_private", "server_private"]
VISIBILITIES: tuple[Visibility, ...] = ("public", "actor_private", "server_private")

#: Every EventKind literal in SPEC declaration order; frozen vocabulary v1.
EventKind = Literal[
    "game_start",
    "round_start",
    "turn_advance",
    "draw_tile",
    "discard",
    "riichi_declared",
    "riichi_accepted",
    "call_window",
    "call_resolved",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "dora_revealed",
    "ron",
    "tsumo",
    "draw_end",
    "abortive_draw",
    "round_end",
    "game_end",
]
EVENT_KINDS: tuple[EventKind, ...] = (
    "game_start",
    "round_start",
    "turn_advance",
    "draw_tile",
    "discard",
    "riichi_declared",
    "riichi_accepted",
    "call_window",
    "call_resolved",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "dora_revealed",
    "ron",
    "tsumo",
    "draw_end",
    "abortive_draw",
    "round_end",
    "game_end",
)

DeltaOperation = Literal["set", "append", "increment"]
DELTA_OPERATIONS: tuple[DeltaOperation, ...] = ("set", "append", "increment")

_PUBLIC_VISIBLE_TO = (0, 1, 2, 3)

#: Scalar payload fields besides ``kind``/``actor`` (closed inventory).
PAYLOAD_SCALAR_FIELDS: tuple[str, ...] = (
    "tile",
    "action_id",
    "source_seat",
    "round_index",
    "scores",
    "reason",
)
#: Tuple payload fields (closed inventory).
PAYLOAD_TUPLE_FIELDS: tuple[str, ...] = (
    "consumed_tiles",
    "offered_action_ids",
    "accepted_action_ids",
)


def _require_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _require_str(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a str, got {type(value).__name__}")
    return value


def _require_enum(value: object, *, name: str, allowed: tuple[str, ...] | frozenset[str]) -> str:
    text = _require_str(value, name=name)
    if text not in allowed:
        raise ContractError(f"{name}={text!r} must be one of {sorted(allowed)}")
    return text


def _require_plain_int(value: object, *, name: str, minimum: int, maximum: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ContractError(f"{name}={value} outside [{minimum}, {maximum}]")
    return value


def _optional_seat(value: object, *, name: str) -> Seat | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be a seat int 0..3, got {type(value).__name__}")
    return make_seat(int(value))


def _optional_tile(value: object, *, name: str) -> TileId | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be a tile id int 0..135, got {type(value).__name__}")
    return make_tile_id(int(value))


def _optional_action(value: object, *, name: str) -> ActionId | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(
            f"{name} must be an action id nonnegative int, got {type(value).__name__}"
        )
    return make_action_id(int(value))


def _tile_tuple(values: Sequence[int], *, name: str) -> tuple[TileId, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractError(f"{name} must be a sequence of tile ids")
    return tuple(make_tile_id(v) for v in values)


def _action_tuple(values: Sequence[int], *, name: str) -> tuple[ActionId, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractError(f"{name} must be a sequence of action ids")
    return tuple(make_action_id(v) for v in values)


def _score_quad(values: object, *, name: str) -> tuple[int, int, int, int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 4:
        raise ContractError(f"{name} must be exactly four score ints")
    scores = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"{name}[{index}] must be an int")
        scores.append(value)
    first, second, third, fourth = scores
    return (first, second, third, fourth)


def _validate_json_value(value: object) -> None:
    """Restrict delta values to the canonical JSON domain (SPEC 2.2)."""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        if abs(value) >= 2**53:
            raise ContractError("delta int value exceeds the canonical safe range")
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ContractError("delta float value must be finite")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item)  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ContractError("delta object keys must be strings")
        for item in value.values():
            _validate_json_value(item)  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        return
    raise ContractError(f"delta value outside JSON domain: {type(value).__name__}")


# ---------------------------------------------------------------------------
# PublicStateDelta paths: closed vocabulary with typed operations/values.
# ---------------------------------------------------------------------------

_DELTA_ROOTS: frozenset[str] = frozenset(
    {
        "scores",
        "honba",
        "riichi_sticks",
        "round_index",
        "dora_indicators",
        "melds",
        "riichi_states",
        "ippatsu",
        "kan_count",
        "live_wall_tiles_remaining",
    }
)
_PATH_ACTOR_PLACEHOLDER = "actor"
_RIICHI_STATE_VALUES = ("none", "declared", "accepted")

_MELD_OBJECT_KEYS = ("meld_id", "kind", "owner", "source_seat", "called_tile", "tiles")


def _validate_delta_path_element(element: object, *, depth: int) -> str | int:
    if isinstance(element, bool):
        raise ContractError("delta path elements must be str or int, not bool")
    if isinstance(element, str):
        if element == "":
            raise ContractError("delta path string elements must be non-empty")
        if depth > 0 and element != _PATH_ACTOR_PLACEHOLDER:
            raise ContractError(f"delta path element {element!r} invalid at position {depth}")
        return element
    if isinstance(element, int):
        return _require_plain_int(element, name="delta path seat", minimum=0, maximum=3)
    raise ContractError(f"delta path element must be str or int: {type(element).__name__}")


def _validate_delta_path(path: Sequence[str | int]) -> tuple[str | int, ...]:
    if isinstance(path, str) or not isinstance(path, Sequence) or len(path) == 0:
        raise ContractError("delta path must be a non-empty sequence of str|int")
    if len(path) > 2:
        raise ContractError("delta paths are at most two elements deep in schema v1")
    root = path[0]
    if not isinstance(root, str) or root not in _DELTA_ROOTS:
        raise ContractError(f"delta path root {root!r} outside the closed vocabulary")
    validated: list[str | int] = [root]
    for depth, element in enumerate(path[1:], start=1):
        validated.append(_validate_delta_path_element(element, depth=depth))
    second = validated[1] if len(validated) > 1 else None
    if root in ("melds", "riichi_states", "ippatsu"):
        if second is None or not isinstance(second, (str, int)):
            raise ContractError(f"delta path {root!r} requires a seat element")
    elif root == "scores":
        # Both ("scores",) quad-set and ("scores", seat) int deltas are
        # published in DELTA_PATH_VOCABULARY; any second element must be a
        # validated seat.
        if second is None:
            pass
        elif not isinstance(second, int):
            raise ContractError("delta path scores second element must be a seat 0..3")
    elif second is not None:
        raise ContractError(f"delta path root {root!r} takes no second element")
    return tuple(validated)


def _validate_delta_value(path: tuple[str | int, ...], operation: str, value: object) -> None:
    """Closed (path, operation) -> value-type rules; undeclared pairs rejected."""
    _validate_json_value(value)
    root = path[0]
    seated_second = path[1] if len(path) > 1 else None

    def _plain_int(candidate: object, name: str) -> None:
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise ContractError(f"delta {name} requires an int value")

    def _tile_value(candidate: object) -> None:
        _plain_int(candidate, "dora_indicators.append")
        make_tile_id(candidate)  # type: ignore[arg-type]  # reason: _plain_int bool-checks candidate above; range validated inside make_tile_id

    if root == "scores":
        if seated_second is None:
            if operation != "set":
                raise ContractError("delta scores supports operation 'set' only")
            _ = _score_quad(value, name="scores.set")
            return
        _plain_int(value, "scores[..]")
        return
    if root in ("honba", "riichi_sticks", "kan_count", "live_wall_tiles_remaining"):
        _plain_int(value, str(root))
        return
    if root == "round_index":
        if operation != "set":
            raise ContractError("delta round_index supports operation 'set' only")
        _plain_int(value, "round_index.set")
        return
    if root == "dora_indicators":
        if operation != "append":
            raise ContractError("delta dora_indicators supports operation 'append' only")
        _tile_value(value)
        return
    if root == "melds":
        if operation != "append":
            raise ContractError("delta melds supports operation 'append' only")
        if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(
            sorted(_MELD_OBJECT_KEYS)
        ):
            raise ContractError("melds.append requires the canonical meld object shape")
        _ = _require_str(value["meld_id"], name="meld.meld_id")  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        _ = _require_enum(  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
            value["kind"], name="meld.kind", allowed=("chi", "pon", "daiminkan", "ankan", "kakan")  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        )
        make_seat(value["owner"])  # type: ignore[arg-type]  # reason: meld field statically object; validated inside make_seat
        if value["source_seat"] is not None:
            make_seat(value["source_seat"])  # type: ignore[arg-type]  # reason: meld field statically object; validated inside make_seat
        if value["called_tile"] is not None:
            make_tile_id(value["called_tile"])  # type: ignore[arg-type]  # reason: meld field statically object; validated inside make_tile_id
        _ = _tile_tuple(value["tiles"], name="meld.tiles")  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
        return
    if root == "riichi_states":
        if operation != "set":
            raise ContractError("delta riichi_states supports operation 'set' only")
        _ = _require_enum(value, name="riichi_states.set", allowed=_RIICHI_STATE_VALUES)
        del seated_second
        return
    if root == "ippatsu":
        if operation != "set":
            raise ContractError("delta ippatsu supports operation 'set' only")
        _ = _require_bool(value, name="ippatsu.set")
        return
    raise ContractError(f"delta path root {root!r} has no value rules")


#: Serialized vocabulary published inside event_schema_v1.json: every allowed
#: (path, operation, value_type) triple. ``path`` elements: str root, then a
#: seat int 0..3 or the "actor" placeholder where applicable.
DELTA_PATH_VOCABULARY: tuple[dict[str, object], ...] = (  # type: ignore[bad-assignment]  # reason: literal infers narrower value type; dict[str, object] invariance
    {"path": ("scores",), "operations": ("set",), "value_type": "int4"},
    *[
        {"path": ("scores", seat), "operations": ("set", "increment"), "value_type": "int"}
        for seat in range(4)
    ],
    {"path": ("honba",), "operations": ("set", "increment"), "value_type": "int"},
    {"path": ("riichi_sticks",), "operations": ("set", "increment"), "value_type": "int"},
    {"path": ("round_index",), "operations": ("set",), "value_type": "int"},
    {"path": ("dora_indicators",), "operations": ("append",), "value_type": "tile_id"},
    {"path": ("melds", "actor"), "operations": ("append",), "value_type": "meld"},
    *[
        {"path": ("riichi_states", seat), "operations": ("set",), "value_type": "riichi_state"}
        for seat in range(4)
    ],
    {"path": ("riichi_states", "actor"), "operations": ("set",), "value_type": "riichi_state"},
    *[
        {"path": ("ippatsu", seat), "operations": ("set",), "value_type": "bool"}
        for seat in range(4)
    ],
    {"path": ("ippatsu", "actor"), "operations": ("set",), "value_type": "bool"},
    {"path": ("kan_count",), "operations": ("set", "increment"), "value_type": "int"},
    {
        "path": ("live_wall_tiles_remaining",),
        "operations": ("set", "increment"),
        "value_type": "int",
    },
)


# ---------------------------------------------------------------------------
# SPEC 7.1 - Dataclasses.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# EventSchema: closed per-kind matrix + versioned artifact.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EventSchemaRow:
    """Authoritative per-kind schema row (SPEC 7.1 EventSchema contents)."""

    kind: str
    visibility: str
    actor_required: bool
    scalar_fields: tuple[tuple[str, bool], ...]  # (field, required_non_null)
    tuple_fields: tuple[tuple[str, str], ...]  # (field, "empty" | "any")
    constraints: tuple[str, ...]
    allowed_delta_paths: tuple[tuple[str | int, ...], ...]
    predecessors: tuple[str, ...]
    successors: tuple[str, ...]


def _row(
    kind: str,
    *,
    visibility: str,
    actor: bool,
    required: Sequence[str] = (),
    tuples: Sequence[tuple[str, str]] = (),
    constraints: Sequence[str] = (),
    delta_paths: Sequence[Sequence[str | int]] = (),
    predecessors: Sequence[str] = (),
    successors: Sequence[str] = (),
) -> EventSchemaRow:
    scalars: list[tuple[str, bool]] = []
    for name in PAYLOAD_SCALAR_FIELDS:
        scalars.append((name, name in required))
    return EventSchemaRow(
        kind=kind,
        visibility=visibility,
        actor_required=actor,
        scalar_fields=tuple(scalars),
        tuple_fields=(
            tuple(tuples)
            if len(tuples) > 0
            else tuple((name, "empty") for name in PAYLOAD_TUPLE_FIELDS)
        ),
        constraints=tuple(constraints),
        allowed_delta_paths=_expand_actor_paths(delta_paths),
        predecessors=tuple(predecessors),
        successors=tuple(successors),
    )


def _expand_actor_paths(
    delta_paths: Sequence[Sequence[str | int]],
) -> tuple[tuple[str | int, ...], ...]:
    """Expand the "actor" placeholder to every concrete seat as well.

    Published events carry concrete seats; observation-side documents may use
    the actor-relative form. Both spellings validate against one row.
    """
    expanded: list[tuple[str | int, ...]] = []
    for path in delta_paths:
        if len(path) == 2 and path[1] == _PATH_ACTOR_PLACEHOLDER:
            expanded.append((path[0], _PATH_ACTOR_PLACEHOLDER))
            expanded.extend((path[0], seat) for seat in range(4))
        else:
            expanded.append(tuple(path))
    return tuple(expanded)


_ALL_SEAT_RESET_PATHS: tuple[tuple[str | int, ...], ...] = (
    ("riichi_states", 0),
    ("riichi_states", 1),
    ("riichi_states", 2),
    ("riichi_states", 3),
    ("ippatsu", 0),
    ("ippatsu", 1),
    ("ippatsu", 2),
    ("ippatsu", 3),
)
_SETTLEMENT_PATHS: tuple[tuple[str | int, ...], ...] = (
    ("scores",),
    ("honba",),
    ("riichi_sticks",),
)

#: Compiled schema matrix; the published artifact MUST stay byte-consistent.
EVENT_SCHEMA_ROWS: dict[str, EventSchemaRow] = {
    row.kind: row
    for row in (
        _row(
            "game_start",
            visibility="public",
            actor=False,
            required=("round_index", "scores"),
            delta_paths=(("scores",),),
            predecessors=(),
            successors=("round_start",),
        ),
        _row(
            "round_start",
            visibility="public",
            actor=True,
            required=("round_index", "scores"),
            delta_paths=(
                ("round_index",),
                ("honba",),
                ("riichi_sticks",),
                ("scores",),
            ),
            predecessors=("game_start", "round_end", "draw_end", "abortive_draw"),
            successors=("turn_advance",),
        ),
        _row(
            "turn_advance",
            visibility="public",
            actor=True,
            predecessors=(
                "round_start",
                "discard",
                "riichi_accepted",
                "call_resolved",
                "dora_revealed",
                "ankan",
                "kakan",
            ),
            successors=("draw_tile",),
        ),
        _row(
            "draw_tile",
            visibility="actor_private",
            actor=True,
            required=("tile",),
            predecessors=("turn_advance", "ankan", "kakan", "daiminkan"),
            successors=("discard", "riichi_declared", "ankan", "kakan", "tsumo"),
        ),
        _row(
            "discard",
            visibility="public",
            actor=True,
            required=("tile", "action_id"),
            predecessors=("draw_tile", "riichi_declared", "chi", "pon", "dora_revealed"),
            successors=("call_window", "turn_advance", "abortive_draw", "draw_end"),
        ),
        _row(
            "riichi_declared",
            visibility="public",
            actor=True,
            required=("tile", "action_id"),
            delta_paths=(("riichi_states", "actor"),),
            predecessors=("draw_tile",),
            successors=("discard",),
        ),
        _row(
            "riichi_accepted",
            visibility="public",
            actor=True,
            delta_paths=(("riichi_states", "actor"), ("riichi_sticks",), ("ippatsu", "actor")),
            predecessors=("discard", "call_resolved"),
            successors=("turn_advance",),
        ),
        _row(
            "call_window",
            visibility="public",
            actor=False,
            predecessors=("discard",),
            successors=("call_resolved",),
        ),
        _row(
            "call_resolved",
            visibility="server_private",
            actor=False,
            tuples=(
                ("consumed_tiles", "empty"),
                ("offered_action_ids", "any"),
                ("accepted_action_ids", "any"),
            ),
            constraints=("call_resolved_resolution_shape",),
            predecessors=("call_window",),
            successors=("chi", "pon", "daiminkan", "ron", "turn_advance", "riichi_accepted"),
        ),
        _row(
            "chi",
            visibility="public",
            actor=True,
            required=("tile", "action_id", "source_seat"),
            tuples=(
                ("consumed_tiles", "any"),
                ("offered_action_ids", "empty"),
                ("accepted_action_ids", "empty"),
            ),
            delta_paths=(("melds", "actor"),),
            predecessors=("call_resolved",),
            successors=("discard",),
        ),
        _row(
            "pon",
            visibility="public",
            actor=True,
            required=("tile", "action_id", "source_seat"),
            tuples=(
                ("consumed_tiles", "any"),
                ("offered_action_ids", "empty"),
                ("accepted_action_ids", "empty"),
            ),
            delta_paths=(("melds", "actor"),),
            predecessors=("call_resolved",),
            successors=("discard",),
        ),
        _row(
            "daiminkan",
            visibility="public",
            actor=True,
            required=("tile", "action_id", "source_seat"),
            tuples=(
                ("consumed_tiles", "any"),
                ("offered_action_ids", "empty"),
                ("accepted_action_ids", "empty"),
            ),
            delta_paths=(("melds", "actor"), ("kan_count",)),
            predecessors=("call_resolved",),
            successors=("dora_revealed", "draw_tile"),
        ),
        _row(
            "ankan",
            visibility="public",
            actor=True,
            required=("action_id",),
            tuples=(
                ("consumed_tiles", "any"),
                ("offered_action_ids", "empty"),
                ("accepted_action_ids", "empty"),
            ),
            delta_paths=(("melds", "actor"), ("kan_count",)),
            predecessors=("draw_tile",),
            successors=("dora_revealed",),
        ),
        _row(
            "kakan",
            visibility="public",
            actor=True,
            required=("tile", "action_id"),
            delta_paths=(("melds", "actor"), ("kan_count",)),
            predecessors=("draw_tile",),
            successors=("dora_revealed", "ron", "draw_tile"),
        ),
        _row(
            "dora_revealed",
            visibility="public",
            actor=False,
            required=("tile",),
            delta_paths=(("dora_indicators",),),
            predecessors=("ankan", "kakan", "daiminkan", "discard", "draw_tile"),
            successors=(
                "draw_tile",
                "turn_advance",
                "discard",
                "ron",
                "tsumo",
                "round_end",
                "abortive_draw",
                "draw_end",
            ),
        ),
        _row(
            "ron",
            visibility="public",
            actor=True,
            required=("tile", "action_id", "source_seat"),
            delta_paths=_SETTLEMENT_PATHS,
            predecessors=("call_resolved", "kakan"),
            successors=("round_end",),
        ),
        _row(
            "tsumo",
            visibility="public",
            actor=True,
            required=("tile", "action_id"),
            delta_paths=_SETTLEMENT_PATHS,
            predecessors=("draw_tile",),
            successors=("round_end",),
        ),
        _row(
            "draw_end",
            visibility="public",
            actor=False,
            required=("scores", "reason"),
            delta_paths=_SETTLEMENT_PATHS,
            predecessors=("discard",),
            successors=("round_end", "game_end"),
        ),
        _row(
            "abortive_draw",
            visibility="public",
            actor=False,
            required=("round_index", "scores", "reason"),
            delta_paths=(("scores",), ("riichi_sticks",)),
            predecessors=("discard", "draw_tile", "ankan"),
            successors=("round_start", "game_end"),
        ),
        _row(
            "round_end",
            visibility="public",
            actor=False,
            required=("round_index", "scores"),
            delta_paths=(*_SETTLEMENT_PATHS, ("round_index",), *_ALL_SEAT_RESET_PATHS),
            predecessors=("ron", "tsumo", "draw_end"),
            successors=("round_start", "game_end"),
        ),
        _row(
            "game_end",
            visibility="public",
            actor=False,
            required=("round_index", "scores", "reason"),
            delta_paths=(("scores",),),
            predecessors=("round_end", "draw_end", "abortive_draw"),
            successors=(),
        ),
    )
}

#: Named structural constraint checked in EventEnvelope.__post_init__.
ACCEPTED_CONSTRAINT = "call_resolved_resolution_shape"

EVENT_SCHEMA_ARTIFACT_TYPE = "hydra2.event_schema"
EVENT_SCHEMA_SCHEMA_VERSION = "1.0.0"
EVENT_SCHEMA_RELPATH = Path("configs") / "contracts" / "event_schema_v1.json"

_ENVELOPE_JSON_FIELDS = (
    "artifact_type",
    "schema_version",
    "compatibility",
    "payload",
)


def _row_to_json(row: EventSchemaRow) -> dict[str, object]:
    return {
        "visibility": row.visibility,
        "actor": "required" if row.actor_required else "forbidden",
        "scalar_fields": {
            name: ("required" if needed else "null") for name, needed in row.scalar_fields
        },
        "tuple_fields": dict(row.tuple_fields),
        "constraints": list(row.constraints),
        "allowed_delta_paths": [list(path) for path in row.allowed_delta_paths],
        "predecessors": list(row.predecessors),
        "successors": list(row.successors),
    }


def build_event_schema_payload() -> dict[str, object]:
    """Deterministic schema payload WITHOUT the digest field."""
    return {
        "schema_version": EVENT_SCHEMA_SCHEMA_VERSION,
        "visibilities": list(VISIBILITIES),
        "kinds": list(EVENT_KINDS),
        "field_spec_values": ["required", "null"],
        "tuple_spec_values": ["empty", "any"],
        "operations": list(DELTA_OPERATIONS),
        "delta_path_vocabulary": [
            {
                "path": list(entry["path"]),  # type: ignore[bad-argument-type]  # reason: vocab entry value statically object; vocabulary frozen by artifact digest
                "operations": list(entry["operations"]),  # type: ignore[attr-defined]  # reason: vocab entry statically object; vocabulary frozen by artifact digest
                "value_type": entry["value_type"],
            }
            for entry in DELTA_PATH_VOCABULARY
        ],
        "constraint_ids": [ACCEPTED_CONSTRAINT],
        "kinds_rows": {kind: _row_to_json(EVENT_SCHEMA_ROWS[kind]) for kind in EVENT_KINDS},
    }


def compute_event_schema_digest(payload_without_digest: Mapping[str, object]) -> DigestText:
    identity = canonical_json_bytes(dict(payload_without_digest))
    return DigestText("sha256:" + hashlib.sha256(identity).hexdigest())


def build_event_schema_envelope() -> dict[str, object]:
    """SPEC 2.2 envelope whose canonical bytes are the published artifact."""
    payload = build_event_schema_payload()
    payload["digest"] = compute_event_schema_digest(payload)
    return {
        "artifact_type": EVENT_SCHEMA_ARTIFACT_TYPE,
        "schema_version": EVENT_SCHEMA_SCHEMA_VERSION,
        "compatibility": "exact",
        "payload": payload,
    }


def _reject_constant(token: str) -> object:
    raise ContractError(f"{token} is outside the canonical JSON domain")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_event_schema(raw_bytes: bytes) -> dict[str, object]:
    """Verify event-schema artifact bytes; returns the envelope document."""
    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContractError(f"event_schema artifact is not valid JSON: {exc}") from exc
    if not isinstance(document, Mapping) or tuple(sorted(document)) != tuple(
        sorted(_ENVELOPE_JSON_FIELDS)
    ):
        raise ContractError("event_schema artifact must be a SPEC 2.2 envelope")
    if document["artifact_type"] != EVENT_SCHEMA_ARTIFACT_TYPE:
        raise ContractError(f"artifact_type must be {EVENT_SCHEMA_ARTIFACT_TYPE!r}")
    if document["compatibility"] != "exact":
        raise ContractError("event_schema compatibility must be exact")
    payload: object = document["payload"]  # type: ignore[index]  # reason: document Mapping-checked above; checker cannot narrow object index
    if not isinstance(payload, Mapping) or "digest" not in payload:  # type: ignore[attr-defined, operator]  # reason: isinstance-narrowed Mapping; checker flags 'in' on bare Mapping
        raise ContractError("event_schema payload missing digest")
    expected = compute_event_schema_digest({k: v for k, v in payload.items() if k != "digest"})  # type: ignore[attr-defined]  # reason: payload Mapping-narrowed above; checker flags .items on bare Mapping
    recorded = make_digest_text(str(payload["digest"]))  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict  # type: ignore[index]  # reason: payload Mapping-narrowed above; index on bare Mapping
    if not hmac.compare_digest(str(recorded), str(expected)):
        from hydra2.contracts.common import DigestMismatchError

        raise DigestMismatchError(
            f"event_schema digest mismatch: recorded {recorded} != recomputed {expected}"
        )
    compiled = build_event_schema_payload()
    if {k: v for k, v in payload.items() if k != "digest"} != compiled:
        raise ContractError("event_schema artifact diverges from the compiled schema matrix")
    return dict(document)


def load_event_schema(path: Path) -> dict[str, object]:
    return parse_event_schema(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# SPEC 7.2 - PacketBoundarySpec and ActorVisiblePacket.
# ---------------------------------------------------------------------------

PACKET_BOUNDARY_ARTIFACT_TYPE = "hydra2.packet_boundary"
PACKET_BOUNDARY_SCHEMA_VERSION = "1.0.0"
PACKET_BOUNDARY_RELPATH = Path("configs") / "contracts" / "packet_boundary_v1.json"

_UPDATE_BOUNDARY_KINDS: tuple[str, ...] = (
    "discard",
    "riichi_declared",
    "riichi_accepted",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "dora_revealed",
    "ron",
    "tsumo",
)
_TERMINAL_BOUNDARY_KINDS: tuple[str, ...] = (
    "round_end",
    "abortive_draw",
    "draw_end",
    "game_end",
)


@dataclass(frozen=True, slots=True)
class PacketBoundarySpec:
    """Published packet partition authority (SPEC 7.2); search cannot redefine it."""

    root_actor: Seat
    start_boundary_kind: str
    decision_boundary_kind: str
    update_boundary_kinds: tuple[str, ...]
    call_group_kinds: tuple[str, ...]
    claim_priority_order: tuple[str, ...]
    terminal_boundary_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_actor", make_seat(self.root_actor))
        for name, value in (
            ("start_boundary_kind", self.start_boundary_kind),
            ("decision_boundary_kind", self.decision_boundary_kind),
        ):
            _ = _require_enum(value, name=name, allowed=EVENT_KINDS)
        for group in ("update_boundary_kinds", "call_group_kinds", "terminal_boundary_kinds"):
            kinds = getattr(self, group)
            if (
                isinstance(kinds, (str, bytes))
                or not isinstance(kinds, Sequence)
                or len(kinds) == 0
            ):
                raise ContractError(f"{group} must be a non-empty sequence")
            for kind in kinds:
                _ = _require_enum(cast("Any", kind), name=f"{group} entry", allowed=EVENT_KINDS)  # pyrefly: ignore[explicit-any]  # reason: deliberate Any for dynamic kind sequence; _require_enum validates
            object.__setattr__(self, group, tuple(kind for kind in kinds))
        order = self.claim_priority_order
        if tuple(sorted(order)) != ("chi", "daiminkan", "pon", "ron"):
            raise ContractError("claim_priority_order must permute ron>daiminkan>pon>chi")

    def to_json(self) -> dict[str, object]:
        return {
            "root_actor": int(self.root_actor),
            "start_boundary": {"event_kind": self.start_boundary_kind},
            "decision_boundary": {
                "event_kind": self.decision_boundary_kind,
                "actor_scope": "root_actor_only",
            },
            "update_boundary": {"event_kinds": list(self.update_boundary_kinds)},
            "call_pass_grouping": {
                "group_kinds": list(self.call_group_kinds),
                "claim_priority_order": list(self.claim_priority_order),
                "successor_packets_per_group": 1,
                "pass_permitted": True,
            },
            "terminal_boundary": {"event_kinds": list(self.terminal_boundary_kinds)},
            "packet_identity": {
                "algorithm": "sha256",
                "canonical_form": "rfc8785",
                "excluded_fields": ["packet_id"],
            },
            "partition_rules": {
                "mutually_exclusive": True,
                "exhaustive": True,
                "nonempty": True,
            },
        }


#: Owner decision D-WP02D-3: the boundary spec fixed for schema v1.
DEFAULT_PACKET_BOUNDARY_SPEC = PacketBoundarySpec(
    root_actor=Seat(0),
    start_boundary_kind="round_start",
    decision_boundary_kind="draw_tile",
    update_boundary_kinds=_UPDATE_BOUNDARY_KINDS,
    call_group_kinds=("discard", "call_window", "call_resolved"),
    claim_priority_order=("ron", "daiminkan", "pon", "chi"),
    terminal_boundary_kinds=_TERMINAL_BOUNDARY_KINDS,
)


def build_packet_boundary_payload() -> dict[str, object]:
    payload = DEFAULT_PACKET_BOUNDARY_SPEC.to_json()
    payload["schema_version"] = PACKET_BOUNDARY_SCHEMA_VERSION
    payload["digest"] = compute_event_schema_digest(
        {k: v for k, v in payload.items() if k != "digest"}
    )
    return payload


def build_packet_boundary_envelope() -> dict[str, object]:
    return {
        "artifact_type": PACKET_BOUNDARY_ARTIFACT_TYPE,
        "schema_version": PACKET_BOUNDARY_SCHEMA_VERSION,
        "compatibility": "exact",
        "payload": build_packet_boundary_payload(),
    }


def parse_packet_boundary_spec(raw_bytes: bytes) -> PacketBoundarySpec:
    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContractError(f"packet_boundary artifact is not valid JSON: {exc}") from exc
    if not isinstance(document, Mapping) or tuple(sorted(document)) != tuple(
        sorted(_ENVELOPE_JSON_FIELDS)
    ):
        raise ContractError("packet_boundary artifact must be a SPEC 2.2 envelope")
    if document["artifact_type"] != PACKET_BOUNDARY_ARTIFACT_TYPE:
        raise ContractError(f"artifact_type must be {PACKET_BOUNDARY_ARTIFACT_TYPE!r}")
    payload: Any = document["payload"]  # pyrefly: ignore[explicit-any]  # reason: Any by design; Mapping-narrowed immediately below
    if not isinstance(payload, Mapping):
        raise ContractError("packet_boundary payload must be an object")
    expected = compute_event_schema_digest({k: v for k, v in payload.items() if k != "digest"})
    recorded = make_digest_text(str(payload.get("digest")))
    if recorded != expected:
        from hydra2.contracts.common import DigestMismatchError

        raise DigestMismatchError(
            f"packet_boundary digest mismatch: recorded {recorded} != recomputed {expected}"
        )
    compiled = build_packet_boundary_payload()
    stripped_compiled = {k: v for k, v in compiled.items() if k != "digest"}
    if {k: v for k, v in payload.items() if k != "digest"} != stripped_compiled:
        raise ContractError("packet_boundary artifact diverges from the compiled spec")
    return _spec_from_payload(payload)


def _spec_from_payload(payload: Mapping[str, object]) -> PacketBoundarySpec:
    grouping = payload.get("call_pass_grouping")
    if not isinstance(grouping, Mapping):
        raise ContractError("call_pass_grouping missing")
    return PacketBoundarySpec(
        root_actor=payload.get("root_actor"),  # type: ignore[arg-type]  # reason: Mapping.get yields object; PacketBoundarySpec validates on construction
        start_boundary_kind=str(payload.get("start_boundary", {}).get("event_kind")),  # type: ignore[union-attr]  # reason: chained .get on object-typed value; str() coerces at runtime
        decision_boundary_kind=str(payload.get("decision_boundary", {}).get("event_kind")),  # type: ignore[union-attr]  # reason: chained .get on object-typed value; str() coerces at runtime
        update_boundary_kinds=tuple(payload.get("update_boundary", {}).get("event_kinds", ())),  # type: ignore[union-attr]  # reason: chained .get on object-typed value; tuple() coerces at runtime
        call_group_kinds=tuple(cast("Any", grouping.get("group_kinds", ()))),  # pyrefly: ignore[explicit-any]  # reason: deliberate Any for dynamic grouping; tuple() coerces at runtime
        claim_priority_order=tuple(cast("Any", grouping.get("claim_priority_order", ()))),  # pyrefly: ignore[explicit-any]  # reason: deliberate Any for dynamic grouping; tuple() coerces at runtime
        terminal_boundary_kinds=tuple(payload.get("terminal_boundary", {}).get("event_kinds", ())),  # type: ignore[union-attr]  # reason: chained .get on object-typed value; tuple() coerces at runtime
    )


def load_packet_boundary_spec(path: Path) -> PacketBoundarySpec:
    return parse_packet_boundary_spec(Path(path).read_bytes())


@dataclass(frozen=True, slots=True)
class ActorVisiblePacket:
    """SPEC 7.2 packet: mutually exclusive/exhaustive/nonempty partition unit."""

    packet_id: PacketId | None
    actor_view: Seat
    source_sequence_start: SequenceNo
    source_sequence_end: SequenceNo
    events: tuple[EventEnvelope, ...]
    public_state_hash_before: DigestText
    public_state_hash_after: DigestText
    observation_hash_after: DigestText

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_view", make_seat(self.actor_view))
        object.__setattr__(
            self, "source_sequence_start", make_sequence_no(self.source_sequence_start)
        )
        object.__setattr__(self, "source_sequence_end", make_sequence_no(self.source_sequence_end))
        if len(self.events) == 0:
            raise ContractError("packets are nonempty (SPEC 7.2)")
        if isinstance(self.events, (str, bytes)) or not isinstance(self.events, Sequence):
            raise ContractError("events must be a sequence of EventEnvelope")
        sequences = [int(event.sequence) for event in self.events]
        if sequences != sorted(set(sequences)):
            raise ContractError("packet events must be strictly sequence-ordered")
        if (
            int(self.source_sequence_start) != sequences[0]
            or int(self.source_sequence_end) != sequences[-1]
        ):
            raise ContractError("packet boundaries must match contained event sequences")
        object.__setattr__(
            self, "public_state_hash_before", make_digest_text(self.public_state_hash_before)
        )
        object.__setattr__(
            self, "public_state_hash_after", make_digest_text(self.public_state_hash_after)
        )
        object.__setattr__(
            self, "observation_hash_after", make_digest_text(self.observation_hash_after)
        )
        if self.packet_id is not None:
            object.__setattr__(self, "packet_id", make_packet_id(self.packet_id))
            expected = compute_packet_id(self)
            if self.packet_id != expected:
                from hydra2.contracts.common import DigestMismatchError

                raise DigestMismatchError(
                    f"packet_id mismatch: recorded {self.packet_id} != recomputed {expected}"
                )

    def to_json(self) -> dict[str, object]:
        return {
            "packet_id": self.packet_id,
            "actor_view": int(self.actor_view),
            "source_sequence_start": int(self.source_sequence_start),
            "source_sequence_end": int(self.source_sequence_end),
            "events": [event.to_json() for event in self.events],
            "public_state_hash_before": self.public_state_hash_before,
            "public_state_hash_after": self.public_state_hash_after,
            "observation_hash_after": self.observation_hash_after,
        }


def packet_identity_document(packet: ActorVisiblePacket) -> dict[str, object]:
    """Canonical bytes input: the packet WITHOUT its packet_id field."""
    document = packet.to_json()
    _ = document.pop("packet_id", None)
    return document


def compute_packet_id(packet: ActorVisiblePacket) -> PacketId:
    """sha256 over canonical bytes excluding packet_id (SPEC 7.2)."""
    identity = canonical_json_bytes(packet_identity_document(packet))
    return PacketId(hashlib.sha256(identity).hexdigest())


def make_actor_visible_packet(
    *,
    actor_view: Seat,
    events: Sequence[EventEnvelope],
    public_state_hash_before: DigestText,
    public_state_hash_after: DigestText,
    observation_hash_after: DigestText,
) -> ActorVisiblePacket:
    """Construct a packet with its packet_id bound to the identity bytes."""
    staged = ActorVisiblePacket(
        packet_id=None,
        actor_view=actor_view,
        source_sequence_start=make_sequence_no(int(events[0].sequence)),
        source_sequence_end=make_sequence_no(int(events[-1].sequence)),
        events=tuple(events),
        public_state_hash_before=public_state_hash_before,
        public_state_hash_after=public_state_hash_after,
        observation_hash_after=observation_hash_after,
    )
    packet_id = compute_packet_id(staged)
    return ActorVisiblePacket(
        packet_id=packet_id,
        actor_view=staged.actor_view,
        source_sequence_start=staged.source_sequence_start,
        source_sequence_end=staged.source_sequence_end,
        events=staged.events,
        public_state_hash_before=staged.public_state_hash_before,
        public_state_hash_after=staged.public_state_hash_after,
        observation_hash_after=staged.observation_hash_after,
    )


def _fold_public_hash(prefix: DigestText, event: EventEnvelope) -> DigestText:
    identity = canonical_json_bytes({"prefix": prefix, "event": envelope_identity_document(event)})
    return DigestText("sha256:" + hashlib.sha256(identity).hexdigest())


_EMPTY_CHAIN_DIGEST = DigestText("sha256:" + hashlib.sha256(b"").hexdigest())


def public_state_chain_hash(events: Sequence[EventEnvelope]) -> DigestText:
    """Fold public event identities into a chained state hash (deterministic)."""
    digest = _EMPTY_CHAIN_DIGEST
    for event in events:
        if event.visibility == "public":
            digest = _fold_public_hash(digest, event)
    return digest


def validate_packet_partition(packets: Sequence[ActorVisiblePacket]) -> None:
    """Check what packets alone can prove: nonempty, ordered, mutually exclusive.

    Actor-visible streams legitimately skip sequence numbers (server-private
    and other seats' private events), so numeric range adjacency is NOT
    required between consecutive packets. Exhaustiveness against a concrete
    stream is enforced by :func:`partition_actor_packets`.
    """
    if len(packets) == 0:
        return
    views = {int(packet.actor_view) for packet in packets}
    for view in views:
        view_packets = [p for p in packets if int(p.actor_view) == view]
        view_packets.sort(key=lambda p: int(cast("Any", p.source_sequence_start)))  # pyrefly: ignore[explicit-any]  # reason: deliberate Any for packet sequence field; int() coerces at runtime
        previous_end: int | None = None
        seen_sequences: set[int] = set()
        for packet in view_packets:
            for event in packet.events:
                sequence = int(event.sequence)
                if sequence in seen_sequences:
                    raise ContractError(
                        f"sequence {sequence} appears in two packets (mutual exclusivity violated)"
                    )
                seen_sequences.add(sequence)
            if previous_end is not None and int(packet.source_sequence_start) <= previous_end:
                raise ContractError("packets overlap (mutual exclusivity violated)")
            previous_end = int(packet.source_sequence_end)


def partition_actor_packets(
    events: Sequence[EventEnvelope],
    spec: PacketBoundarySpec,
    *,
    actor_view: Seat,
    observation_hash_of,
) -> tuple[ActorVisiblePacket, ...]:
    """Partition the actor-visible stream per the published boundary spec.

    ``observation_hash_of(view, end_sequence)`` supplies the genuine post-
    packet observation hash; packets remain mutually exclusive, exhaustive,
    and nonempty over the visible stream.
    """
    view = make_seat(int(actor_view))
    visible = filter_events_for_actor(events, view)
    if len(visible) == 0:
        return ()

    segments: list[list[EventEnvelope]] = []
    current: list[EventEnvelope] = []

    def _close() -> None:
        if len(current) != 0:
            segments.append(list(current))
            current.clear()

    pending_call_group = False
    for event in visible:
        kind = event.kind
        if pending_call_group:
            current.append(event)
            if kind not in spec.call_group_kinds:
                pending_call_group = False
                _close()
            continue
        if kind == "discard":
            # Owner decision D-WP02D-5: [discard .. call_resolved] (or the
            # pass outcome) always forms ONE packet, regardless of what the
            # discard follows.
            _close()
            current.append(event)
            pending_call_group = True
            continue
        boundary = (
            kind in spec.update_boundary_kinds
            or kind in spec.terminal_boundary_kinds
            or kind == spec.start_boundary_kind
            or (
                kind == spec.decision_boundary_kind and int(event.actor) == int(spec.root_actor)  # type: ignore[arg-type]  # reason: actor fields statically object; int() validates seat equality
            )
        )
        if len(current) == 0:
            current.append(event)
            if boundary:
                _close()
            continue
        current.append(event)
        if boundary:
            _close()
    _close()

    packets: list[ActorVisiblePacket] = []
    for segment in segments:
        prefix_sequences = [e for e in visible if int(e.sequence) < int(segment[0].sequence)]
        before = public_state_chain_hash(prefix_sequences)
        after = public_state_chain_hash(prefix_sequences + list(segment))
        packets.append(
            make_actor_visible_packet(
                actor_view=view,
                events=tuple(segment),
                public_state_hash_before=before,
                public_state_hash_after=after,
                observation_hash_after=cast(  # pyrefly: ignore[explicit-any]  # reason: deliberate Any passthrough; digest produced by observation_hash_of
                    "Any",
                    observation_hash_of(view, int(segment[-1].sequence)),  # pyrefly: ignore[unknown-argument-type] # Any intentional for raw dict
                ),
            )
        )
    covered = {int(e.sequence) for p in packets for e in p.events}
    expected = {int(e.sequence) for e in visible}
    if covered != expected:
        from hydra2.contracts.common import PacketPartitionError

        raise PacketPartitionError(
            "packet partition is not exhaustive over the actor-visible stream: "
            f"missing {sorted(expected - covered)}, extra {sorted(covered - expected)}"
        )
    validate_packet_partition(packets)
    return tuple(packets)
