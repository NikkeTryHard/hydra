"""SPEC 7.2 packets — boundary spec, identity, and actor partition.

Owns the published packet partition authority with its content digest,
the actor-visible packet with its canonical identity binding, the public
state hash chain, and the partition entry points. Search cannot redefine
the boundary; packets stay mutually exclusive, exhaustive, and nonempty
over the visible stream.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from hydra2.artifacts.canonical import canonical_bytes as canonical_json_bytes
from hydra2.artifacts.canonical import canonical_bytes_batch
from hydra2.contracts.common import (
    ContractError,
    DigestText,
    PacketId,
    Seat,
    SequenceNo,
    make_packet_id,
    make_sequence_no,
)
from hydra2.contracts.event_envelope import (
    EventEnvelope,
    envelope_identity_document,
    filter_events_for_actor,
)
from hydra2.contracts.event_schema import compute_event_schema_digest
from hydra2.contracts.event_vocab import (
    EVENT_KINDS,
    _reject_constant,
    _reject_duplicate_keys,
    _require_enum,
)

try:
    from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _bridge_contracts = None  # type: ignore[assignment]


def _require_packet_bridge() -> Any:
    """Resolve the bridge, fail closed when the extension is not built."""
    if _bridge_contracts is None:
        raise ImportError(
            "hydra2 packet authority requires the hydra2._native bridge; "
            "run `pixi run build-ext` to build the extension before use"
        )
    return _bridge_contracts


__all__ = [
    "DEFAULT_PACKET_BOUNDARY_SPEC",
    "PACKET_BOUNDARY_ARTIFACT_TYPE",
    "PACKET_BOUNDARY_RELPATH",
    "PACKET_BOUNDARY_SCHEMA_VERSION",
    "ActorVisiblePacket",
    "PacketBoundarySpec",
    "build_packet_boundary_envelope",
    "build_packet_boundary_payload",
    "compute_packet_id",
    "compute_packet_ids",
    "load_packet_boundary_spec",
    "make_actor_visible_packet",
    "make_actor_visible_packets",
    "packet_identity_document",
    "parse_packet_boundary_spec",
    "partition_actor_packets",
    "public_state_chain_hash",
    "public_state_chain_hash_prefixes",
    "validate_packet_partition",
]


_packet_bridge = _require_packet_bridge()

#: Published packet partition authority identity (re-exported from the bridge).
PACKET_BOUNDARY_ARTIFACT_TYPE: str = _packet_bridge.PACKET_BOUNDARY_ARTIFACT_TYPE
#: Published packet boundary schema version (re-exported from the bridge).
PACKET_BOUNDARY_SCHEMA_VERSION: str = _packet_bridge.PACKET_BOUNDARY_SCHEMA_VERSION
#: Packet boundary artifact relpath (re-exported; wrapped to keep the Path type).
_PACKET_RELPATH_TEXT: str = _packet_bridge.PACKET_BOUNDARY_RELPATH
PACKET_BOUNDARY_RELPATH = Path(_PACKET_RELPATH_TEXT)

_UPDATE_RAW: tuple[str, ...] = _packet_bridge.PACKET_UPDATE_BOUNDARY_KINDS
_UPDATE_BOUNDARY_KINDS: tuple[str, ...] = tuple(_UPDATE_RAW)
_TERMINAL_RAW: tuple[str, ...] = _packet_bridge.PACKET_TERMINAL_BOUNDARY_KINDS
_TERMINAL_BOUNDARY_KINDS: tuple[str, ...] = tuple(_TERMINAL_RAW)


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
        if _bridge_contracts is None:
            raise ImportError(
                "hydra2 packet authority requires the hydra2._native bridge; "
                "run `pixi run build-ext` to build the extension before use"
            )
        try:
            root_actor: Seat = _bridge_contracts.make_seat(self.root_actor)
        except (ValueError, TypeError) as exc:
            raise ContractError(f"root_actor rejected: {exc}") from exc
        object.__setattr__(self, "root_actor", root_actor)
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
    if not isinstance(document, dict) or set(document) != {
        "artifact_type",
        "schema_version",
        "compatibility",
        "payload",
    }:
        raise ContractError("packet_boundary artifact must be a SPEC 2.2 envelope")
    if document["artifact_type"] != PACKET_BOUNDARY_ARTIFACT_TYPE:
        raise ContractError(f"artifact_type must be {PACKET_BOUNDARY_ARTIFACT_TYPE!r}")
    payload: Any = document["payload"]  # pyrefly: ignore[explicit-any]  # reason: Any by design; Mapping-narrowed immediately below
    if not isinstance(payload, Mapping):
        raise ContractError("packet_boundary payload must be an object")
    expected = compute_event_schema_digest({k: v for k, v in payload.items() if k != "digest"})
    try:
        recorded: DigestText = _require_packet_bridge().make_digest_text(str(payload.get("digest")))
    except (ValueError, TypeError) as exc:
        raise ContractError(f"packet digest rejected: {exc}") from exc
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
        try:
            bridge = _require_packet_bridge()
            actor_view: Seat = bridge.make_seat(self.actor_view)
            hash_before: DigestText = bridge.make_digest_text(self.public_state_hash_before)
            hash_after: DigestText = bridge.make_digest_text(self.public_state_hash_after)
            obs_after: DigestText = bridge.make_digest_text(self.observation_hash_after)
        except (ValueError, TypeError) as exc:
            raise ContractError(f"ActorVisiblePacket rejected: {exc}") from exc
        object.__setattr__(self, "actor_view", actor_view)
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
        object.__setattr__(self, "public_state_hash_before", hash_before)
        object.__setattr__(self, "public_state_hash_after", hash_after)
        object.__setattr__(self, "observation_hash_after", obs_after)
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
    doc_bytes = canonical_json_bytes(packet_identity_document(packet))
    return PacketId(hashlib.sha256(doc_bytes).hexdigest())


def compute_packet_ids(packets: Sequence[ActorVisiblePacket]) -> list[PacketId]:
    """sha256 over canonical bytes for each packet via ONE bridge FFI (SPEC 7.2).

    Byte-identical to ``[compute_packet_id(p) for p in packets]``: the packet
    identity docs serialize in one :func:`canonical_bytes_batch` call, then
    hash with ``hashlib`` exactly like the single path. Empty input returns
    ``[]`` without touching the bridge. Bridge rejects raise, never silent.
    """
    items = list(packets)
    if len(items) == 0:
        return []
    blobs = canonical_bytes_batch([packet_identity_document(p) for p in items])
    return [PacketId(hashlib.sha256(blob).hexdigest()) for blob in blobs]


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


def make_actor_visible_packets(
    staged: Sequence[ActorVisiblePacket],
) -> tuple[ActorVisiblePacket, ...]:
    """Bind packet_ids for pre-staged (``packet_id=None``) packets via ONE batch FFI.

    Byte-identical to ``tuple(make_actor_visible_packet(...) for ...)`` built
    from the same staged packets: ids come from :func:`compute_packet_ids`,
    and each final packet still re-verifies its id through the
    :class:`ActorVisiblePacket` constructor (fail closed, same as the single
    path). Non-staged input (``packet_id`` already set) is rejected.
    """
    items = list(staged)
    for item in items:
        if item.packet_id is not None:
            raise ContractError("make_actor_visible_packets requires packet_id=None staged packets")
    packet_ids = compute_packet_ids(items)
    return tuple(
        ActorVisiblePacket(
            packet_id=packet_id,
            actor_view=item.actor_view,
            source_sequence_start=item.source_sequence_start,
            source_sequence_end=item.source_sequence_end,
            events=item.events,
            public_state_hash_before=item.public_state_hash_before,
            public_state_hash_after=item.public_state_hash_after,
            observation_hash_after=item.observation_hash_after,
        )
        for item, packet_id in zip(items, packet_ids, strict=True)
    )


_EMPTY_CHAIN_DIGEST = DigestText("sha256:" + hashlib.sha256(b"").hexdigest())


def _fold_chain_digest(prefix: DigestText, event: EventEnvelope) -> DigestText:
    """One public fold step (shared by the single and prefix-batch chain paths)."""
    fold_doc = {"prefix": str(prefix), "event": envelope_identity_document(event)}
    fold_bytes = canonical_json_bytes(fold_doc)
    return DigestText("sha256:" + hashlib.sha256(fold_bytes).hexdigest())


def public_state_chain_hash(events: Sequence[EventEnvelope]) -> DigestText:
    """Fold public event identities into a chained state hash (deterministic)."""
    digest = _EMPTY_CHAIN_DIGEST
    for event in events:
        if event.visibility == "public":
            digest = _fold_chain_digest(digest, event)
    return digest


def public_state_chain_hash_prefixes(
    events: Sequence[EventEnvelope],
) -> tuple[DigestText, ...]:
    """Chain digest after each prefix: ``out[k] == public_state_chain_hash(events[:k])``.

    Incremental single pass over the stream (same fold bytes as the single
    path, hashed with ``hashlib``): replaces the per-segment full re-walk in
    :func:`partition_actor_packets` (O(segments*events) serializations) with
    O(events). Non-public events carry the running digest forward unchanged.
    """
    out: list[DigestText] = [_EMPTY_CHAIN_DIGEST]
    digest = _EMPTY_CHAIN_DIGEST
    for event in events:
        if event.visibility == "public":
            digest = _fold_chain_digest(digest, event)
        out.append(digest)
    return tuple(out)


def validate_packet_partition(packets: Sequence[ActorVisiblePacket]) -> None:
    """Check what packets alone can prove: nonempty, ordered, mutually exclusive.

    Actor-visible streams legitimately skip sequence numbers (server-private
    and other seats' private events), so numeric range adjacency is NOT
    required between consecutive packets. Exhaustiveness against a concrete
    stream is enforced by :func:`partition_actor_packets`.

    Thin bridge translator: event-level exclusivity (sequence sets across
    packets) stays Python as it needs envelope objects; the span half
    (actor_view/start/end ordered, mutually exclusive) delegates via scalar
    triples, never live objects. Bridge rejections surface as ContractError.
    """
    if len(packets) == 0:
        return
    views = {int(packet.actor_view) for packet in packets}
    for view in views:
        view_packets = [p for p in packets if int(p.actor_view) == view]
        view_packets.sort(key=lambda p: int(cast("Any", p.source_sequence_start)))  # pyrefly: ignore[explicit-any]  # reason: deliberate Any for packet sequence field; int() coerces at runtime
        seen_sequences: set[int] = set()
        for packet in view_packets:
            for event in packet.events:
                sequence = int(event.sequence)
                if sequence in seen_sequences:
                    raise ContractError(
                        f"sequence {sequence} appears in two packets (mutual exclusivity violated)"
                    )
                seen_sequences.add(sequence)
    if _bridge_contracts is None:
        raise ImportError(
            "hydra2 packet authority requires the hydra2._native bridge; "
            "run `pixi run build-ext` to build the extension before use"
        )
    spans = [
        (int(p.actor_view), int(p.source_sequence_start), int(p.source_sequence_end))
        for p in packets
    ]
    try:
        _bridge_contracts.validate_packet_spans(spans)  # type: ignore[attr-defined]
    except (ValueError, TypeError) as exc:
        raise ContractError(f"packet partition rejected: {exc}") from exc


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
    view: Seat = _require_packet_bridge().make_seat(int(actor_view))
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

    # Batch form: segments partition `visible` contiguously in order (single
    # split pass above appends each visible event exactly once), so segment
    # boundaries are visible indices — verified by identity below, fail closed.
    # One incremental chain pass replaces the per-segment full re-walk
    # (O(segments*events) serializations -> O(events)); one batch FFI binds
    # every packet id. Bytes/hashes identical to the retired per-segment loop.
    ranges: list[tuple[int, int]] = []
    pos = 0
    for segment in segments:
        for event in segment:
            if event is not visible[pos]:
                raise ContractError("packet segmentation is not contiguous over the visible stream")
            pos += 1
        ranges.append((pos - len(segment), pos))
    if pos != len(visible):
        raise ContractError("packet segmentation is not exhaustive over the visible stream")
    prefixes = public_state_chain_hash_prefixes(visible)
    staged: list[ActorVisiblePacket] = []
    for (start, end), segment in zip(ranges, segments, strict=True):
        staged.append(
            ActorVisiblePacket(
                packet_id=None,
                actor_view=view,
                source_sequence_start=make_sequence_no(int(segment[0].sequence)),
                source_sequence_end=make_sequence_no(int(segment[-1].sequence)),
                events=tuple(segment),
                public_state_hash_before=prefixes[start],
                public_state_hash_after=prefixes[end],
                observation_hash_after=cast(  # pyrefly: ignore[explicit-any]  # reason: deliberate Any passthrough; digest produced by observation_hash_of
                    "Any",
                    observation_hash_of(view, int(segment[-1].sequence)),  # pyrefly: ignore[unknown-argument-type]  # reason: segments hold validated envelopes; int() coerces the sequence
                ),
            )
        )
    packets = list(make_actor_visible_packets(staged))
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
