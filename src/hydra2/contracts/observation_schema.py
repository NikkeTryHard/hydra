"""SPEC 8 observation schema — versioned artifact derived from the dataclass.

Owns the closed versioned artifact: the schema tables compiled from the
live dataclass, the payload builder with its digest and cache, and the
envelope with its strict parser. The per-seat builder lives in the sibling
assembly module and stamps this digest onto every built observation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

from hydra2.contracts.canonical import canonical_json_bytes
from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    DigestText,
    make_digest_text,
)
from hydra2.contracts.event import EVENT_SCHEMA_SCHEMA_VERSION
from hydra2.contracts.observation_actor import ActorObservation
from hydra2.contracts.observation_types import (
    _FURIETEN_STATES,
    _RIICHI_STATES,
    DORA_SENTINEL,
    MELD_KINDS,
    PHASES,
)

__all__ = [
    "OBSERVATION_SCHEMA_ARTIFACT_TYPE",
    "OBSERVATION_SCHEMA_RELPATH",
    "OBSERVATION_SCHEMA_SCHEMA_VERSION",
    "build_observation_schema_envelope",
    "build_observation_schema_payload",
    "compute_observation_schema_digest",
    "load_observation_schema",
    "observation_schema_digest",
    "parse_observation_schema",
]

# ---------------------------------------------------------------------------
# ObservationSchema - closed versioned artifact derived from ActorObservation
# (SPEC 8 tail: fixes field order, enum/range constraints, serialization;
# SPEC 21 config category). Contracts never import hydra2.artifacts, so the
# digest is computed through the byte-identical leaf canonicalizer.
# ---------------------------------------------------------------------------

OBSERVATION_SCHEMA_ARTIFACT_TYPE = "hydra2.observation_schema"
OBSERVATION_SCHEMA_SCHEMA_VERSION = "1.0.0"
OBSERVATION_SCHEMA_RELPATH = Path("configs") / "contracts" / "observation_schema_v1.json"

_OBSERVATION_SCHEMA_ENVELOPE_FIELDS = (
    "artifact_type",
    "schema_version",
    "compatibility",
    "payload",
)

_SEAT_SPEC: dict[str, object] = {"dtype": "seat", "minimum": 0, "maximum": 3}
_TILE_ID_SPEC: dict[str, object] = {"dtype": "tile_id", "minimum": 0, "maximum": 135}
_DIGEST_SPEC: dict[str, object] = {"dtype": "digest_text", "pattern": "sha256:[0-9a-f]{64}"}
_NONEMPTY_STRING_SPEC: dict[str, object] = {"dtype": "string", "min_length": 1}

#: Serialization row for one exposed meld (SPEC 8 ``VisibleMeld`` field order).
_VISIBLE_MELD_ROW: dict[str, object] = {
    "field_order": ["meld_id", "kind", "owner", "source_seat", "called_tile", "tiles"],
    "fields": {
        "meld_id": {"dtype": "string", "nullable": True, "derived_when_null": "visible_meld_id"},
        "kind": {"dtype": "enum", "values": list(MELD_KINDS)},
        "owner": dict(_SEAT_SPEC),
        "source_seat": {**_SEAT_SPEC, "nullable": True},
        "called_tile": {**_TILE_ID_SPEC, "nullable": True},
        "tiles": {
            "dtype": "tile_id_array",
            "item_minimum": 0,
            "item_maximum": 135,
            "constraints": ["nonempty_unique_ascending"],
        },
    },
    "kind_tile_counts": {"chi": 3, "pon": 3, "daiminkan": 4, "ankan": 4, "kakan": 4},
    "kind_shapes": {
        "chi": "three consecutive logical types inside one suit",
        "pon_daiminkan_ankan_kakan": "all tiles share one logical type",
    },
    "ankan_requires_all_four_copies": True,
    "call_kinds_with_source_and_called_tile": ["chi", "pon", "daiminkan"],
    "self_kinds_without_source_or_called_tile": ["ankan", "kakan"],
    "source_seat_must_differ_from_owner": True,
}

#: One closed constraint row per ``ActorObservation`` field. The table is
#: checked against the live dataclass when the payload is built: adding,
#: removing, renaming, or reordering a field without extending the schema
#: raises instead of silently drifting from the published artifact.
_FIELD_CONSTRAINTS: dict[str, dict[str, object]] = {
    "game_id": dict(_NONEMPTY_STRING_SPEC),
    "decision_id": dict(_NONEMPTY_STRING_SPEC),
    "sequence": {"dtype": "sequence_no", "minimum": 0},
    "actor": dict(_SEAT_SPEC),
    "rules_id": dict(_NONEMPTY_STRING_SPEC),
    "rules_hash": dict(_DIGEST_SPEC),
    "action_table_hash": dict(_DIGEST_SPEC),
    "event_schema_hash": dict(_DIGEST_SPEC),
    "observation_schema_hash": dict(_DIGEST_SPEC),
    "packet_boundary_hash": dict(_DIGEST_SPEC),
    "round_index": {"dtype": "integer", "minimum": 0},
    "round_wind": {"dtype": "tile_type", "minimum": 0, "maximum": 33},
    "hand_number": {"dtype": "integer", "minimum": 0},
    "seat_winds": {
        "dtype": "tile_type",
        "length": 4,
        "item_minimum": 27,
        "item_maximum": 30,
        "constraints": ["permutes_east_south_west_north_aligned_by_seat"],
    },
    "honba": {"dtype": "integer", "minimum": 0},
    "riichi_sticks": {"dtype": "integer", "minimum": 0},
    "dealer": dict(_SEAT_SPEC),
    "scores": {
        "dtype": "integer",
        "length": 4,
        "item_minimum": -(10**9),
        "item_maximum": 10**9,
        "int32_safe": True,
    },
    "turn_actor": dict(_SEAT_SPEC),
    "phase": {"dtype": "enum", "values": list(PHASES)},
    "live_wall_tiles_remaining": {"dtype": "integer", "minimum": 0},
    "kan_count": {"dtype": "integer", "minimum": 0, "maximum": 4},
    "ippatsu_active": {"dtype": "boolean", "length": 4},
    "actor_furiten": {"dtype": "enum", "values": list(_FURIETEN_STATES)},
    "actor_can_tsumo": {"dtype": "boolean"},
    "actor_can_riichi": {"dtype": "boolean"},
    "pending_declaration_discard": {**_TILE_ID_SPEC, "nullable": True},
    "concealed_hand": {
        "dtype": "tile_id_array",
        "item_minimum": 0,
        "item_maximum": 135,
        "constraints": ["serialized_ascending_duplicates_allowed"],
    },
    "own_drawn_tile": {**_TILE_ID_SPEC, "nullable": True},
    "visible_discards": {
        "dtype": "array",
        "length": 4,
        "items": {"dtype": "tile_id_array", "item_minimum": 0, "item_maximum": 135},
    },
    "visible_melds": {
        "dtype": "array",
        "length": 4,
        "items": {"dtype": "visible_meld_array"},
        "element_row_ref": "visible_meld_row",
    },
    "riichi_states": {"dtype": "enum", "length": 4, "values": list(_RIICHI_STATES)},
    "dora_indicators": {
        "dtype": "dora_slot_array",
        "length": 5,
        "sentinel": DORA_SENTINEL,
        "revealed_minimum": 0,
        "revealed_maximum": 135,
        "constraints": [
            "shape_exactly_five_never_padded",
            "revealed_contiguous_from_index_zero",
            "sentinel_tail_only",
        ],
    },
    "visible_history": {
        "dtype": "event_envelope_array",
        "element_artifact": {
            "artifact_type": "hydra2.event_schema",
            "schema_version": EVENT_SCHEMA_SCHEMA_VERSION,
            "relpath": "configs/contracts/event_schema_v1.json",
        },
        "constraints": ["every_entry_visible_to_the_observed_actor"],
    },
    "legal_mask": {
        "dtype": "boolean_array",
        "min_length": 1,
        "constraints": [
            "at_least_one_true_at_every_decision",
            "length_equals_published_action_table_actions",
        ],
    },
    "observation_hash": dict(_DIGEST_SPEC),
}


def build_observation_schema_payload() -> dict[str, object]:
    """Deterministic ObservationSchema payload WITHOUT the digest field."""
    # The drift-guard test patches the constraint table on the stable
    # ``hydra2.contracts.observation`` path, so resolve it through the shim
    # instead of this module's globals.
    import hydra2.contracts.observation as _shim

    names = tuple(field.name for field in fields(ActorObservation))
    declared = set(_shim._FIELD_CONSTRAINTS)
    if set(names) != declared:
        raise ContractError(
            "observation schema must stay closed over ActorObservation: "
            f"fields without constraint rows={sorted(set(names) - declared)} "
            f"constraint rows without fields={sorted(declared - set(names))}"
        )
    return {
        "schema_version": OBSERVATION_SCHEMA_SCHEMA_VERSION,
        "field_order": list(names),
        "fields": {name: _shim._FIELD_CONSTRAINTS[name] for name in names},
        "visible_meld_row": _VISIBLE_MELD_ROW,
        "enums": {
            "phase": list(PHASES),
            "actor_furiten": list(_FURIETEN_STATES),
            "riichi_state": list(_RIICHI_STATES),
            "meld_kind": list(MELD_KINDS),
        },
        "identity": {
            "hash_field": "observation_hash",
            "excluded_fields": ["observation_hash"],
            "hash_rule": (
                "sha256 over RFC 8785 canonical bytes of the serialized field "
                "document WITHOUT observation_hash"
            ),
        },
        "serialization": {
            "encoding": "rfc8785_canonical_json_utf8_no_bom",
            "document_order": "ActorObservation declaration order",
            "concealed_hand": "ascending physical TileId; drawn tile stays separate",
            "seats_tiles_sequences": "serialized as JSON integers",
            "enums": "serialized as literal strings",
            "null_semantics": "explicit null; absent keys prohibited",
        },
        "visibility_boundary": {
            "forbidden_content": [
                "wall",
                "dead_wall",
                "opponent_concealed_tiles",
                "unrevealed_dora_indicators",
                "ura_dora",
                "engine_rng_state",
                "future_events",
                "server_private_events",
                "opponent_legal_masks",
                "privileged_labels",
            ],
        },
    }


def compute_observation_schema_digest(payload_without_digest: Mapping[str, object]) -> DigestText:
    """sha256 over canonical bytes of the digest-stripped payload."""
    identity = canonical_json_bytes(dict(payload_without_digest))
    return DigestText("sha256:" + hashlib.sha256(identity).hexdigest())


_OBSERVATION_SCHEMA_DIGEST_CACHE: DigestText | None = None


def observation_schema_digest() -> DigestText:
    """Digest of the compiled schema; stamped onto every built observation."""
    global _OBSERVATION_SCHEMA_DIGEST_CACHE
    cached = _OBSERVATION_SCHEMA_DIGEST_CACHE
    if cached is None:
        cached = compute_observation_schema_digest(build_observation_schema_payload())
        _OBSERVATION_SCHEMA_DIGEST_CACHE = cached
    return cached


def build_observation_schema_envelope() -> dict[str, object]:
    """SPEC 2.2 envelope whose canonical bytes are the published artifact."""
    payload = build_observation_schema_payload()
    payload["digest"] = compute_observation_schema_digest(payload)
    return {
        "artifact_type": OBSERVATION_SCHEMA_ARTIFACT_TYPE,
        "schema_version": OBSERVATION_SCHEMA_SCHEMA_VERSION,
        "compatibility": "exact",
        "payload": payload,
    }


def _reject_json_constant(token: str) -> object:
    raise ContractError(f"{token} is outside the canonical JSON domain")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_observation_schema(raw_bytes: bytes) -> dict[str, object]:
    """Verify observation-schema artifact bytes; returns the envelope document."""
    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ContractError(f"observation_schema artifact is not valid JSON: {exc}") from exc
    if not isinstance(document, Mapping) or tuple(sorted(document)) != tuple(
        sorted(_OBSERVATION_SCHEMA_ENVELOPE_FIELDS)
    ):
        raise ContractError("observation_schema artifact must be a SPEC 2.2 envelope")
    if document["artifact_type"] != OBSERVATION_SCHEMA_ARTIFACT_TYPE:
        raise ContractError("artifact_type must be 'hydra2.observation_schema'")
    if document["compatibility"] != "exact":
        raise ContractError("observation_schema compatibility must be exact")
    payload: object = document["payload"]  # type: ignore[index]  # reason: document Mapping-checked above; checker cannot narrow object index
    if not isinstance(payload, Mapping) or "digest" not in payload:  # type: ignore[attr-defined]  # reason: isinstance-narrowed Mapping; checker flags 'in' on bare Mapping
        raise ContractError("observation_schema payload missing digest")
    expected = compute_observation_schema_digest(
        {k: v for k, v in payload.items() if k != "digest"}  # type: ignore[attr-defined]  # reason: payload Mapping-narrowed above; checker flags .items on bare Mapping
    )
    recorded = make_digest_text(str(payload["digest"]))  # pyrefly: ignore[unknown-argument-type]  # reason: payload Mapping-narrowed above; str() coerces; Any intentional for raw dict  # type: ignore[index]  # reason: payload Mapping-narrowed above; index on bare Mapping
    if not hmac.compare_digest(str(recorded), str(expected)):
        raise DigestMismatchError(
            f"observation_schema digest mismatch: recorded {recorded} != recomputed {expected}"
        )
    compiled = build_observation_schema_payload()
    if {k: v for k, v in payload.items() if k != "digest"} != compiled:
        raise ContractError("observation_schema artifact diverges from the compiled schema")
    return dict(document)


def load_observation_schema(path: Path) -> dict[str, object]:
    """Read and verify the published artifact at ``path``."""
    return parse_observation_schema(Path(path).read_bytes())
