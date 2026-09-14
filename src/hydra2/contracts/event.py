"""SPEC 7 events, packets, and visibility — WP-02D contract module.

Re-export facade over the split modules: :mod:`hydra2.contracts.event_vocab`
(frozen literal sets and delta-value rules),
:mod:`hydra2.contracts.event_envelope` (payload, delta, visibility, stream
grammar), :mod:`hydra2.contracts.event_schema` (closed per-kind matrix and
versioned artifact), and :mod:`hydra2.contracts.event_packet` (boundary
spec, identity, actor partition). Import from this path; it preserves every
public name and ``__all__``.
"""

from __future__ import annotations

import json as json  # re-exported: contract tests read published artifacts via ``ev.json``

from hydra2.contracts.event_envelope import (
    EventEnvelope as EventEnvelope,
)
from hydra2.contracts.event_envelope import (
    EventPayload as EventPayload,
)
from hydra2.contracts.event_envelope import (
    PublicStateDelta as PublicStateDelta,
)
from hydra2.contracts.event_envelope import (
    envelope_digest as envelope_digest,
)
from hydra2.contracts.event_envelope import (
    envelope_identity_document as envelope_identity_document,
)
from hydra2.contracts.event_envelope import (
    filter_events_for_actor as filter_events_for_actor,
)
from hydra2.contracts.event_envelope import (
    validate_event_stream as validate_event_stream,
)
from hydra2.contracts.event_envelope import (
    visible_to_actor as visible_to_actor,
)
from hydra2.contracts.event_packet import (
    DEFAULT_PACKET_BOUNDARY_SPEC as DEFAULT_PACKET_BOUNDARY_SPEC,
)
from hydra2.contracts.event_packet import (
    PACKET_BOUNDARY_ARTIFACT_TYPE as PACKET_BOUNDARY_ARTIFACT_TYPE,
)
from hydra2.contracts.event_packet import (
    PACKET_BOUNDARY_RELPATH as PACKET_BOUNDARY_RELPATH,
)
from hydra2.contracts.event_packet import (
    PACKET_BOUNDARY_SCHEMA_VERSION as PACKET_BOUNDARY_SCHEMA_VERSION,
)
from hydra2.contracts.event_packet import (
    ActorVisiblePacket as ActorVisiblePacket,
)
from hydra2.contracts.event_packet import (
    PacketBoundarySpec as PacketBoundarySpec,
)
from hydra2.contracts.event_packet import (
    build_packet_boundary_envelope as build_packet_boundary_envelope,
)
from hydra2.contracts.event_packet import (
    build_packet_boundary_payload as build_packet_boundary_payload,
)
from hydra2.contracts.event_packet import (
    compute_packet_id as compute_packet_id,
)
from hydra2.contracts.event_packet import (
    load_packet_boundary_spec as load_packet_boundary_spec,
)
from hydra2.contracts.event_packet import (
    make_actor_visible_packet as make_actor_visible_packet,
)
from hydra2.contracts.event_packet import (
    packet_identity_document as packet_identity_document,
)
from hydra2.contracts.event_packet import (
    parse_packet_boundary_spec as parse_packet_boundary_spec,
)
from hydra2.contracts.event_packet import (
    partition_actor_packets as partition_actor_packets,
)
from hydra2.contracts.event_packet import (
    public_state_chain_hash as public_state_chain_hash,
)
from hydra2.contracts.event_packet import (
    validate_packet_partition as validate_packet_partition,
)
from hydra2.contracts.event_schema import (
    ACCEPTED_CONSTRAINT as ACCEPTED_CONSTRAINT,
)
from hydra2.contracts.event_schema import (
    EVENT_SCHEMA_ARTIFACT_TYPE as EVENT_SCHEMA_ARTIFACT_TYPE,
)
from hydra2.contracts.event_schema import (
    EVENT_SCHEMA_RELPATH as EVENT_SCHEMA_RELPATH,
)
from hydra2.contracts.event_schema import (
    EVENT_SCHEMA_ROWS as EVENT_SCHEMA_ROWS,
)
from hydra2.contracts.event_schema import (
    EVENT_SCHEMA_SCHEMA_VERSION as EVENT_SCHEMA_SCHEMA_VERSION,
)
from hydra2.contracts.event_schema import (
    EventSchemaRow as EventSchemaRow,
)
from hydra2.contracts.event_schema import (
    build_event_schema_envelope as build_event_schema_envelope,
)
from hydra2.contracts.event_schema import (
    build_event_schema_payload as build_event_schema_payload,
)
from hydra2.contracts.event_schema import (
    compute_event_schema_digest as compute_event_schema_digest,
)
from hydra2.contracts.event_schema import (
    event_schema_digest as event_schema_digest,
)
from hydra2.contracts.event_schema import (
    event_schema_payload as event_schema_payload,
)
from hydra2.contracts.event_schema import (
    load_event_schema as load_event_schema,
)
from hydra2.contracts.event_schema import (
    parse_event_schema as parse_event_schema,
)
from hydra2.contracts.event_vocab import (
    DELTA_OPERATIONS as DELTA_OPERATIONS,
)
from hydra2.contracts.event_vocab import (
    DELTA_PATH_VOCABULARY as DELTA_PATH_VOCABULARY,
)
from hydra2.contracts.event_vocab import (
    EVENT_KINDS as EVENT_KINDS,
)
from hydra2.contracts.event_vocab import (
    VISIBILITIES as VISIBILITIES,
)
from hydra2.contracts.event_vocab import (
    Visibility as Visibility,
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
    "event_schema_digest",
    "event_schema_payload",
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

# Names importable from this path before the split that live in the
# submodules now (kept so engine helpers and type-checking imports resolve).
from hydra2.contracts.event_envelope import _CALL_KINDS as _CALL_KINDS
from hydra2.contracts.event_envelope import FIRST_EVENT_KIND as FIRST_EVENT_KIND
from hydra2.contracts.event_vocab import PAYLOAD_SCALAR_FIELDS as PAYLOAD_SCALAR_FIELDS
from hydra2.contracts.event_vocab import PAYLOAD_TUPLE_FIELDS as PAYLOAD_TUPLE_FIELDS
from hydra2.contracts.event_vocab import DeltaOperation as DeltaOperation
from hydra2.contracts.event_vocab import EventKind as EventKind
from hydra2.contracts.event_vocab import _require_enum as _require_enum
