"""WP-02D packet grouping and partition (call-window successors, exclusivity).

Covers owner decisions D-WP02D-2/-5 plus packet partition exclusivity,
exhaustiveness, nonemptiness, and identity binding on the scripted round.
Helpers and fixtures arrive from `test_events_obs_wp02d`; this file owns
only the packet test classes. Failure mode is fail-closed `ContractError`
or `DigestMismatchError` on bad grouping or identity, never silent.
"""

from __future__ import annotations

import hashlib

import pytest
from tests.contracts.test_events_obs_wp02d import _PACKET_BOUNDARY_HASH as _PACKET_BOUNDARY_HASH
from tests.contracts.test_events_obs_wp02d import Round as Round

from hydra2.artifacts.canonical import canonical_bytes as canonical_json_bytes
from hydra2.contracts.common import DigestMismatchError, Seat
from hydra2.contracts.event_envelope import EventEnvelope, filter_events_for_actor
from hydra2.contracts.event_packet import (
    DEFAULT_PACKET_BOUNDARY_SPEC,
    ActorVisiblePacket,
    partition_actor_packets,
    validate_packet_partition,
)

pytestmark = pytest.mark.contract_package("WP-02D")


@pytest.fixture(scope="module")
def stream() -> list[EventEnvelope]:
    return Round().build_stream()


# ---------------------------------------------------------------------------
# Owner decisions D-WP02D-2/-5: one successor per call window; grouping.
# ---------------------------------------------------------------------------


class TestCallWindowSingleSuccessorGrouping:
    def test_call_group_discard_through_call_resolved_forms_one_packet(self, stream):
        spec = DEFAULT_PACKET_BOUNDARY_SPEC
        assert spec.call_group_kinds == ("discard", "call_window", "call_resolved")
        assert spec.claim_priority_order == ("ron", "daiminkan", "pon", "chi")
        for seat in range(4):
            packets = partition_actor_packets(
                filter_events_for_actor(stream, Seat(seat)),
                spec,
                actor_view=Seat(seat),
                observation_hash_of=lambda view, end: _PACKET_BOUNDARY_HASH,
            )
            claim_packets = [p for p in packets if int(p.source_sequence_start) == 5]
            assert len(claim_packets) == 1  # exactly one successor packet
            assert claim_packets[0].events[-1].kind == "chi"

    def test_pass_path_groups_discard_with_turn_advance(self):
        round_ = Round()
        scores = (25000,) * 4
        s0, s1 = Seat(0), Seat(1)
        round_.add("game_start", ridx=0, scores=scores)
        round_.add("round_start", actor=s0, ridx=0, scores=scores)
        round_.add("turn_advance", actor=s0)
        round_.add("draw_tile", actor=s0, tile=99, visibility="actor_private", visible_to=(s0,))
        discard = round_.add("discard", actor=s0, tile=99, action=7)
        round_.add("call_window")
        resolved = round_.add(
            "call_resolved", visibility="server_private", visible_to=(), offered=(), accepted=()
        )
        round_.add("turn_advance", actor=s1)
        events = round_.events
        packets = partition_actor_packets(
            filter_events_for_actor(events, Seat(2)),
            DEFAULT_PACKET_BOUNDARY_SPEC,
            actor_view=Seat(2),
            observation_hash_of=lambda view, end: _PACKET_BOUNDARY_HASH,
        )
        grouped = [
            p for p in packets if any(int(e.sequence) == int(discard.sequence) for e in p.events)
        ]
        assert len(grouped) == 1
        kinds = [e.kind for e in grouped[0].events]
        assert kinds == ["discard", "call_window", "turn_advance"]  # pass allowed
        assert resolved.visibility == "server_private"
        assert resolved.payload.accepted_action_ids == ()


# ---------------------------------------------------------------------------
# Packet partition: exclusivity / exhaustiveness / nonemptiness on the round.
# ---------------------------------------------------------------------------


class TestPacketPartitionScriptedRound:
    def test_partition_is_exclusive_exhaustive_nonempty_for_every_seat(self, stream):
        spec = DEFAULT_PACKET_BOUNDARY_SPEC
        for seat in range(4):
            visible = filter_events_for_actor(stream, Seat(seat))
            packets = partition_actor_packets(
                visible,
                spec,
                actor_view=Seat(seat),
                observation_hash_of=lambda view, end: _PACKET_BOUNDARY_HASH,
            )
            assert packets, seat
            validate_packet_partition(packets)
            covered: set[int] = set()
            ordered = sorted(packets, key=lambda p: int(p.source_sequence_start))
            previous_end = None
            for packet in ordered:
                assert packet.events  # nonempty
                sequences = [int(e.sequence) for e in packet.events]
                assert len(set(sequences)) == len(sequences)
                covered.update(sequences)
                if previous_end is not None:
                    assert int(packet.source_sequence_start) > previous_end
            assert covered == {int(e.sequence) for e in visible}
            assert all(e.visibility != "server_private" for p in packets for e in p.events)

    def test_packet_identity_binds_canonical_bytes_minus_packet_id(self, stream):
        packets = partition_actor_packets(
            filter_events_for_actor(stream, Seat(0)),
            DEFAULT_PACKET_BOUNDARY_SPEC,
            actor_view=Seat(0),
            observation_hash_of=lambda view, end: _PACKET_BOUNDARY_HASH,
        )
        sample = packets[0]
        identity = sample.to_json()
        recorded = identity.pop("packet_id")
        recomputed = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        assert recorded == recomputed
        with pytest.raises(DigestMismatchError):
            ActorVisiblePacket(
                packet_id="0" * 64,
                actor_view=sample.actor_view,
                source_sequence_start=sample.source_sequence_start,
                source_sequence_end=sample.source_sequence_end,
                events=sample.events,
                public_state_hash_before=sample.public_state_hash_before,
                public_state_hash_after=sample.public_state_hash_after,
                observation_hash_after=sample.observation_hash_after,
            )
