"""SPEC 7.1 event schema — closed per-kind matrix and versioned artifact.

Owns the authoritative per-kind schema rows (required/null payload fields,
visibility, actor constraint, allowed delta paths, predecessor/successor
grammar), the compiled payload with its content digest, and the artifact
parse/load entry points. The published document stays byte-consistent with
this matrix; envelopes validate their delta paths against it.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.contracts.canonical import canonical_json_bytes
from hydra2.contracts.common import ContractError, DigestText
from hydra2.contracts.event_vocab import (
    _ENVELOPE_JSON_FIELDS,
    _PATH_ACTOR_PLACEHOLDER,
    DELTA_OPERATIONS,
    DELTA_PATH_VOCABULARY,
    EVENT_KINDS,
    PAYLOAD_SCALAR_FIELDS,
    PAYLOAD_TUPLE_FIELDS,
    VISIBILITIES,
    _reject_constant,
    _reject_duplicate_keys,
)

__all__ = [
    "ACCEPTED_CONSTRAINT",
    "EVENT_SCHEMA_ARTIFACT_TYPE",
    "EVENT_SCHEMA_RELPATH",
    "EVENT_SCHEMA_ROWS",
    "EVENT_SCHEMA_SCHEMA_VERSION",
    "EventSchemaRow",
    "build_event_schema_envelope",
    "build_event_schema_payload",
    "compute_event_schema_digest",
    "event_schema_digest",
    "event_schema_payload",
    "load_event_schema",
    "parse_event_schema",
]


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


#: Compiled event-schema payload, process-constant (built from module
#: constants). Module-owned: compare against it, never mutate it. Same
#: pattern as ``_RULES_CACHE`` (replay_expand) and
#: ``_OBSERVATION_SCHEMA_DIGEST_CACHE`` (observation).
_EVENT_SCHEMA_PAYLOAD_CACHE: dict[str, object] | None = None

#: Digest of :func:`event_schema_payload`; stamped onto envelopes at capture.
_EVENT_SCHEMA_DIGEST_CACHE: DigestText | None = None


def event_schema_payload() -> dict[str, object]:
    """Compiled event-schema payload WITHOUT the digest field.

    Built once per process from module constants; the returned mapping is
    module-owned (compare, never mutate). Builders needing an owned copy
    must copy before stamping ``digest``.
    """
    global _EVENT_SCHEMA_PAYLOAD_CACHE
    cached = _EVENT_SCHEMA_PAYLOAD_CACHE
    if cached is None:
        cached = build_event_schema_payload()
        _EVENT_SCHEMA_PAYLOAD_CACHE = cached
    return cached


def event_schema_digest() -> DigestText:
    """Digest of the compiled event-schema payload; computed once per process."""
    global _EVENT_SCHEMA_DIGEST_CACHE
    cached = _EVENT_SCHEMA_DIGEST_CACHE
    if cached is None:
        cached = compute_event_schema_digest(event_schema_payload())
        _EVENT_SCHEMA_DIGEST_CACHE = cached
    return cached


def build_event_schema_envelope() -> dict[str, object]:
    """SPEC 2.2 envelope whose canonical bytes are the published artifact."""
    payload = copy.deepcopy(event_schema_payload())
    payload["digest"] = event_schema_digest()
    return {
        "artifact_type": EVENT_SCHEMA_ARTIFACT_TYPE,
        "schema_version": EVENT_SCHEMA_SCHEMA_VERSION,
        "compatibility": "exact",
        "payload": payload,
    }


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
    recorded = _bridge_contracts.make_digest_text(str(payload["digest"]))  # pyrefly: ignore[unknown-argument-type]  # reason: payload Mapping-narrowed above; str() coerces; Any intentional for raw dict  # type: ignore[index]  # reason: payload Mapping-narrowed above; index on bare Mapping
    if not hmac.compare_digest(str(recorded), str(expected)):
        from hydra2.contracts.common import DigestMismatchError

        raise DigestMismatchError(
            f"event_schema digest mismatch: recorded {recorded} != recomputed {expected}"
        )
    compiled = event_schema_payload()
    if {k: v for k, v in payload.items() if k != "digest"} != compiled:
        raise ContractError("event_schema artifact diverges from the compiled schema matrix")
    return dict(document)


def load_event_schema(path: Path) -> dict[str, object]:
    return parse_event_schema(Path(path).read_bytes())
