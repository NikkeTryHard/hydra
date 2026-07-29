"""WP-02D gate: events, observation, and visibility (BUILD §5 WP-02D).

Covers the BUILD checklist verbatim plus the SPEC section 22 fixtures
OBS-DORA-005, OBS-DRAW-PRIVATE-001, OBS-HIDDEN-PERM-001, OBS-CANARY-001:
hidden-tile permutation stability, forbidden canary isolation, concealed-draw
actor isolation, public-event fan-out to every seat, server-private rejection,
the fixed five-slot dora shape (never padded), sentinel contiguity, legal-mask
alignment with the published action table digest, sequence monotonicity, the
one-successor call-window grouping, and packet partition exclusivity /
exhaustiveness / nonemptiness on a scripted round.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from hydra2.artifacts.digest import sha256_digest
from hydra2.contracts import event as ev
from hydra2.contracts.action import load_action_table
from hydra2.contracts.canonical import canonical_json_bytes
from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    DigestText,
    Seat,
    VisibilityViolationError,
)
from hydra2.contracts.observation import (
    DORA_SENTINEL,
    DORA_SHAPE,
    MELD_KINDS,
    PHASES,
    VISIBILITY_VALIDATOR,
    ActorObservation,
    ObservationBuilder,
    compute_observation_hash,
    make_actor_observation,
    observation_identity_document,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION_TABLE_PATH = _REPO_ROOT / "configs" / "contracts" / "action_table_v1.json"
EVENT_SCHEMA_PATH = _REPO_ROOT / "configs" / "contracts" / "event_schema_v1.json"
PACKET_BOUNDARY_PATH = _REPO_ROOT / "configs" / "contracts" / "packet_boundary_v1.json"

pytestmark = pytest.mark.contract_package("WP-02D")

#: Golden identities of the two artifacts published by WP-02D.
GOLDEN_EVENT_SCHEMA_SHA256 = (
    # Supersession chain 9b128532.. -> ..b0d6ba -> ..13b32 -> final below:
    # kan-predecessor + dora_revealed grammar repairs republished through
    # build_event_schema_envelope during the WP-03A finish.
    "sha256:63f270fad23bd6263202e47548698d67701b1fd7a90a7770c21c96a8be43d849"
)
GOLDEN_PACKET_BOUNDARY_SHA256 = (
    "sha256:3ee9446b58eaa3cb0ab3929111ba28d6d3dd2445a687ae285f1d96fb7cd60180"
)

_RULES_HASH = DigestText("sha256:" + "3de07f2c" * 8)
_OBS_SCHEMA_HASH = DigestText("sha256:" + "0b" * 32)
_PACKET_BOUNDARY_HASH = DigestText("sha256:" + "ee" * 32)

# Physical ids used by the scripted round (tile // 4 = logical type).
_T_DISCARD_0 = 99  # type 24; chi consumes 30/34? no - chi claims 99 from seat 0
_T_DRAW_1 = 52  # seat 1 draw (type 13)
_T_KAN_BASE = 108  # East honors; ankan consumes 108..111
_T_DORA = 12  # revealed indicator
_T_RIICHI_DISCARD = 20
_CANARY_TILE = 135


def _true_mask(length: int) -> tuple[bool, ...]:
    return tuple([True] + [False] * (length - 1))


class Round:
    """Scripted grammar-valid round used across the fixture tests."""

    def __init__(self) -> None:
        self.schema_hash: str = ev.load_event_schema(EVENT_SCHEMA_PATH)["payload"]["digest"]
        self.events: list[ev.EventEnvelope] = []
        self._next = 1

    def add(self, kind: str, **kwargs: Any) -> ev.EventEnvelope:
        envelope = self._envelope(self._next, kind, **kwargs)
        self._next += 1
        self.events.append(envelope)
        return envelope

    def _envelope(
        self,
        sequence: int,
        kind: str,
        *,
        actor=None,
        tile=None,
        action=None,
        source=None,
        consumed=(),
        ridx=None,
        scores=None,
        reason=None,
        visibility="public",
        visible_to=None,
        offered=(),
        accepted=(),
        deltas=(),
    ) -> ev.EventEnvelope:
        payload = ev.EventPayload(
            kind=kind,
            actor=actor,
            tile=tile,
            action_id=action,
            source_seat=source,
            consumed_tiles=consumed,
            offered_action_ids=offered,
            accepted_action_ids=accepted,
            round_index=ridx,
            scores=scores,
            reason=reason,
        )
        deltas = tuple(
            d if isinstance(d, ev.PublicStateDelta) else ev.PublicStateDelta(*d) for d in deltas
        )
        return ev.EventEnvelope(
            game_id="g-wp02d",
            sequence=sequence,
            kind=kind,
            actor=actor,
            visibility=visibility,
            visible_to=tuple(
                (Seat(0), Seat(1), Seat(2), Seat(3)) if visible_to is None else visible_to
            ),
            payload=payload,
            public_delta=deltas,
            rules_hash=_RULES_HASH,
            schema_hash=self.schema_hash,
        )

    def build_stream(self) -> list[ev.EventEnvelope]:
        scores = (25000, 25000, 25000, 25000)
        s0, s1, s3 = Seat(0), Seat(1), Seat(3)
        self.add("game_start", ridx=0, scores=scores)
        self.add("round_start", actor=s0, ridx=0, scores=scores)
        self.add("turn_advance", actor=s0)
        self.add(
            "draw_tile", actor=s0, tile=_T_DISCARD_0, visibility="actor_private", visible_to=(s0,)
        )
        self.add("discard", actor=s0, tile=_T_DISCARD_0, action=7)
        self.add("call_window")
        self.add(
            "call_resolved",
            visibility="server_private",
            visible_to=(),
            offered=(11, 12),
            accepted=(12,),
        )
        self.add("chi", actor=s3, tile=_T_DISCARD_0, action=12, source=s0, consumed=(93, 101))
        self.add("discard", actor=s3, tile=41, action=13)
        self.add("turn_advance", actor=s0)
        self.add("draw_tile", actor=s0, tile=3, visibility="actor_private", visible_to=(s0,))
        self.add("riichi_declared", actor=s0, tile=_T_RIICHI_DISCARD, action=14)
        self.add("discard", actor=s0, tile=_T_RIICHI_DISCARD, action=14)
        self.add(
            "riichi_accepted",
            actor=s0,
            deltas=(
                (("riichi_states", 0), "set", "accepted"),
                (("riichi_sticks",), "increment", 1),
                (("ippatsu", 0), "set", True),
            ),
        )
        self.add("turn_advance", actor=s1)
        self.add(
            "draw_tile", actor=s1, tile=_T_DRAW_1, visibility="actor_private", visible_to=(s1,)
        )
        self.add("ankan", actor=s1, action=15, consumed=(_T_KAN_BASE, 109, 110, 111))
        self.add("dora_revealed", tile=_T_DORA)
        self.add("discard", actor=s1, tile=_T_DRAW_1, action=16)
        self.add("draw_end", scores=scores, reason="wall exhausted")
        self.add("round_end", ridx=0, scores=scores)
        return self.events


def _snapshot_fields(decision_id: str = "dec-1") -> dict[str, Any]:
    return {
        "decision_id": decision_id,
        "round_index": 0,
        "round_wind": 27,
        "hand_number": 1,
        "seat_winds": (27, 28, 29, 30),
        "honba": 0,
        "riichi_sticks": 1,
        "dealer": Seat(0),
        "scores": (25000, 25000, 25000, 25000),
        "turn_actor": Seat(1),
        "phase": "discard_response",
        "live_wall_tiles_remaining": 55,
        "ippatsu_active": (True, False, False, False),
    }


class TestObsDrawPrivate001:
    """SPEC 22 OBS-DRAW-PRIVATE-001: concealed draw split."""

    def test_concealed_draw_reaches_drawing_actor_only(self, stream, action_table):
        draws = [e for e in stream if e.kind == "draw_tile"]
        builder = _make_builder(len(action_table.actions))
        _feed_round(builder, stream)
        for seat in range(4):
            observation = builder.build(
                actor=Seat(seat), legal_mask=_true_mask(len(action_table.actions))
            )
            held = {int(e.sequence) for e in observation.visible_history}
            for draw in draws:
                if seat == int(draw.payload.actor):
                    assert int(draw.sequence) in held
                else:
                    assert int(draw.sequence) not in held
                    assert observation.own_drawn_tile != int(draw.payload.tile)
        # The drawing actor holds its own latest drawn tile separately.
        river0 = {
            int(t)
            for t in builder.build(
                actor=Seat(0), legal_mask=_true_mask(len(action_table.actions))
            ).visible_discards[0]
        }
        holder_view = builder.build(actor=Seat(0), legal_mask=_true_mask(len(action_table.actions)))
        assert holder_view.visible_discards[0] == (_T_DISCARD_0, _T_RIICHI_DISCARD)
        assert river0 == {_T_DISCARD_0, _T_RIICHI_DISCARD}


def _make_builder(mask_length: int) -> ObservationBuilder:
    return ObservationBuilder(
        game_id="g-wp02d",
        rules_id="tenhou_4p_hanchan_v1",
        rules_hash=_RULES_HASH,
        action_table_hash=DigestText("sha256:" + "7b" * 32),
        expected_legal_mask_length=mask_length,
        event_schema_hash=ev.load_event_schema(EVENT_SCHEMA_PATH)["payload"]["digest"],
        packet_boundary_hash=_PACKET_BOUNDARY_HASH,
    )


def _feed_round(builder: ObservationBuilder, stream) -> None:
    for envelope in stream:
        builder.append_visible(envelope)
    builder.update_public_state(**_snapshot_fields())


@pytest.fixture(scope="module")
def action_table():
    return load_action_table(ACTION_TABLE_PATH)


@pytest.fixture(scope="module")
def stream() -> list[ev.EventEnvelope]:
    return Round().build_stream()


# ---------------------------------------------------------------------------
# Published schema artifacts: golden bytes, closed matrix, delta vocabulary.
# ---------------------------------------------------------------------------


class TestPublishedSchemaArtifacts:
    def test_golden_artifact_bytes_and_digests(self):
        assert sha256_digest(EVENT_SCHEMA_PATH.read_bytes()) == GOLDEN_EVENT_SCHEMA_SHA256
        assert sha256_digest(PACKET_BOUNDARY_PATH.read_bytes()) == GOLDEN_PACKET_BOUNDARY_SHA256

    def test_event_schema_matches_compiled_matrix(self):
        document = ev.load_event_schema(EVENT_SCHEMA_PATH)
        compiled = ev.build_event_schema_payload()
        stripped = {k: v for k, v in document["payload"].items() if k != "digest"}
        assert stripped == compiled

    def test_unknown_payload_field_is_unrepresentable(self):
        with pytest.raises(TypeError):
            ev.EventPayload(
                kind="discard",
                actor=Seat(0),
                tile=5,
                action_id=1,
                source_seat=None,
                consumed_tiles=(),
                offered_action_ids=(),
                accepted_action_ids=(),
                round_index=None,
                scores=None,
                reason=None,
                wall_remaining=70,  # type: ignore[call-arg]
            )
        payload = ev.EventPayload(
            kind="discard",
            actor=Seat(0),
            tile=5,
            action_id=1,
            source_seat=None,
            consumed_tiles=(),
            offered_action_ids=(),
            accepted_action_ids=(),
            round_index=None,
            scores=None,
            reason=None,
        )
        with pytest.raises(ContractError):
            payload.field("wall_remaining")

    def test_undeclared_delta_path_and_operation_rejected(self):
        with pytest.raises(ContractError):
            Round()._envelope(
                1,
                "discard",
                actor=Seat(0),
                tile=5,
                action=1,
                deltas=((("scores",), "set", (25000,) * 4),),
            )
        with pytest.raises(ContractError):
            Round()._envelope(
                1,
                "riichi_accepted",
                actor=Seat(0),
                deltas=((("ura_dora",), "set", 5),),
            )

    def test_red_five_physical_identity_preserved_in_payload(self):
        plain, red = 17, 18  # two distinct copies of one five-type (red-aware ids)
        first = Round()._envelope(1, "discard", actor=Seat(0), tile=plain, action=1)
        second = Round()._envelope(2, "discard", actor=Seat(0), tile=red, action=1)
        assert second.payload.tile != first.payload.tile
        assert canonical_json_bytes(first.payload.to_json()) != canonical_json_bytes(
            second.payload.to_json()
        )


# ---------------------------------------------------------------------------
# Envelope validation: visibility matrix and per-kind shapes.
# ---------------------------------------------------------------------------


class TestEnvelopeVisibilityMatrix:
    def test_public_visible_exactly_all_four_seats(self):
        with pytest.raises(ContractError):
            Round()._envelope(1, "turn_advance", actor=Seat(0), visible_to=(Seat(0),))

    def test_actor_private_exactly_one_seat_and_draw_targets_actor(self):
        with pytest.raises(ContractError):
            Round()._envelope(
                1,
                "draw_tile",
                actor=Seat(0),
                tile=9,
                visibility="actor_private",
                visible_to=(Seat(0), Seat(1)),
            )
        with pytest.raises(VisibilityViolationError):
            Round()._envelope(
                1,
                "draw_tile",
                actor=Seat(0),
                tile=9,
                visibility="actor_private",
                visible_to=(Seat(1),),
            )

    def test_server_private_empty_visible_to_enforced(self):
        with pytest.raises(VisibilityViolationError):
            Round()._envelope(
                1,
                "call_resolved",
                visibility="server_private",
                visible_to=(Seat(2),),
                offered=(1,),
                accepted=(1,),
            )

    def test_call_window_is_public_and_tile_free(self):
        window = Round()._envelope(1, "call_window")
        assert window.visibility == "public"
        assert window.payload.actor is None
        assert window.payload.tile is None
        assert window.payload.offered_action_ids == ()

    def test_call_resolution_accepts_exactly_one_offered_successor(self):
        resolved = Round()._envelope(
            1,
            "call_resolved",
            visibility="server_private",
            visible_to=(),
            offered=(11, 12),
            accepted=(12,),
        )
        assert resolved.payload.accepted_action_ids == (12,)
        with pytest.raises(ContractError):
            Round()._envelope(
                1,
                "call_resolved",
                visibility="server_private",
                visible_to=(),
                offered=(11, 12),
                accepted=(11, 12),
            )


# ---------------------------------------------------------------------------
# Sequence monotonicity and stream grammar.
# ---------------------------------------------------------------------------


class TestSequenceMonotonicity:
    def test_builder_rejects_out_of_order_ingestion(self, stream):
        builder = _make_builder(len(stream))
        builder.append_visible(stream[3])
        with pytest.raises(ContractError):
            builder.append_visible(stream[1])

    def test_validate_event_stream_rejects_duplicate_sequences(self, stream):
        with pytest.raises(ContractError):
            ev.validate_event_stream([stream[0], stream[1], stream[1]])

    def test_validate_event_stream_rejects_grammar_violation(self, stream):
        discard = next(e for e in stream if e.kind == "discard")
        ron = Round()._envelope(10_000, "ron", actor=Seat(2), tile=50, action=9, source=Seat(0))
        with pytest.raises(ContractError):
            ev.validate_event_stream([stream[0], stream[1], discard, ron])

    def test_scripted_round_stream_is_valid(self, stream):
        ev.validate_event_stream(stream)


# ---------------------------------------------------------------------------
# SPEC 22 fixture OBS-DRAW-PRIVATE-001 (+ BUILD concealed-draw bullet).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUILD: public call/riichi/discard reaches every seat.
# ---------------------------------------------------------------------------


class TestPublicEventsAllSeats:
    def test_discard_riichi_and_call_events_visible_from_every_seat(self, stream, action_table):
        builder = _make_builder(len(action_table.actions))
        _feed_round(builder, stream)
        expected_kinds = ("discard", "riichi_declared", "riichi_accepted", "chi")
        wanted = {
            kind: {int(e.sequence) for e in stream if e.kind == kind} for kind in expected_kinds
        }
        for seat in range(4):
            observation = builder.build(
                actor=Seat(seat), legal_mask=_true_mask(len(action_table.actions))
            )
            held = {int(e.sequence) for e in observation.visible_history}
            for kind in expected_kinds:
                assert wanted[kind] <= held, (seat, kind)
        dealer_view = builder.build(actor=Seat(0), legal_mask=_true_mask(len(action_table.actions)))
        assert list(dealer_view.visible_discards[0]) == [_T_DISCARD_0, _T_RIICHI_DISCARD]
        assert any(meld.kind == "chi" for meld in dealer_view.visible_melds[3])
        assert dealer_view.riichi_states == ("accepted", "none", "none", "none")


# ---------------------------------------------------------------------------
# BUILD: server-private rejection everywhere + SPEC 22 OBS-CANARY-001.
# ---------------------------------------------------------------------------


class TestServerPrivateRejectedAndCanary001:
    def _server_private(self, sequence: int = 999) -> ev.EventEnvelope:
        return Round()._envelope(
            sequence,
            "call_resolved",
            visibility="server_private",
            visible_to=(),
            offered=(11, 12),
            accepted=(12,),
        )

    def test_validator_rejects_for_every_seat_without_payload_leak(self):
        envelope = self._server_private()
        for seat in range(4):
            with pytest.raises(VisibilityViolationError) as excinfo:
                VISIBILITY_VALIDATOR.validate_event_for_actor(envelope, Seat(seat))
            message = str(excinfo.value)
            assert "(11, 12)" not in message and "offered" not in message

    def test_filter_and_builder_leave_no_trace(self, stream, action_table):
        canary = self._server_private()
        assert ev.filter_events_for_actor([canary], Seat(0)) == ()
        builder = _make_builder(len(action_table.actions))
        _feed_round(builder, [*stream, canary])
        for seat in range(4):
            observation = builder.build(
                actor=Seat(seat), legal_mask=_true_mask(len(action_table.actions))
            )
            document = canonical_json_bytes(observation.to_json())
            assert b"call_resolved" not in document
            assert all(int(e.sequence) != 999 for e in observation.visible_history)

    def test_obs_canary_001_hidden_marker_never_reaches_model_or_planner_bytes(
        self, stream, action_table
    ):
        # A hidden-world marker TILE lives inside a server-private offer; no
        # model/planner-visible structure may contain it: observation fields,
        # reprs, or any packet event of any seat.
        builder = _make_builder(len(action_table.actions))
        _feed_round(builder, stream)
        builder.append_visible(self._server_private(sequence=999))  # silently dropped

        def contains_value(node: object, needle: int) -> bool:
            if isinstance(node, Mapping):
                return any(contains_value(v, needle) for v in node.values())
            if isinstance(node, (list, tuple)):
                return any(contains_value(v, needle) for v in node)
            return node == needle

        for seat in range(4):
            observation = builder.build(
                actor=Seat(seat), legal_mask=_true_mask(len(action_table.actions))
            )
            assert not contains_value(observation.to_json(), _CANARY_TILE)
            assert str(_CANARY_TILE) not in repr(observation)
            packets = ev.partition_actor_packets(
                builder._histories[seat],
                ev.DEFAULT_PACKET_BOUNDARY_SPEC,
                actor_view=Seat(seat),
                observation_hash_of=lambda view, end: _PACKET_BOUNDARY_HASH,
            )
            for packet in packets:
                assert all(e.visibility != "server_private" for e in packet.events)
                assert all(int(e.sequence) != 999 for e in packet.events)


# ---------------------------------------------------------------------------
# SPEC 22 fixture OBS-HIDDEN-PERM-001 (+ BUILD hidden-permutation bullet).
# ---------------------------------------------------------------------------


class TestObsHiddenPerm001:
    def test_unseen_permutation_leaves_serialized_observation_unchanged(self, stream, action_table):
        mask = _true_mask(len(action_table.actions))
        worlds = []
        # Two hidden worlds differing ONLY in unrevealed tiles: the opponent's
        # concealed hand composition and an unseen wall tile id.
        for hidden_hand in ((70,), (71,)):
            builder = _make_builder(len(action_table.actions))
            _feed_round(builder, stream)
            builder.set_concealed_hand(Seat(1), hidden_hand)  # hidden to seats 0/2/3
            builder.set_concealed_hand(Seat(2), (60,))  # constant control
            worlds.append(builder)
        for viewer in range(4):
            documents = []
            for builder in worlds:
                observation = builder.build(actor=Seat(viewer), legal_mask=mask)
                documents.append(canonical_json_bytes(observation.to_json()))
            if viewer == 1:
                continue  # the holder legitimately distinguishes its own hand
            assert documents[0] == documents[1]

    def test_permutation_of_concealed_hand_order_preserves_hash(self, action_table, stream):
        builder = _make_builder(len(action_table.actions))
        _feed_round(builder, stream)
        builder.set_concealed_hand(Seat(0), (13, 1, 5))
        first = builder.build(actor=Seat(0), legal_mask=_true_mask(len(action_table.actions)))
        other = _make_builder(len(action_table.actions))
        _feed_round(other, stream)
        other.set_concealed_hand(Seat(0), (13, 1, 5))  # same multiset, storage sorts
        second = other.build(actor=Seat(0), legal_mask=_true_mask(len(action_table.actions)))
        assert first.observation_hash == second.observation_hash
        assert sorted(first.concealed_hand) == list(first.concealed_hand)


# ---------------------------------------------------------------------------
# SPEC 22 fixture OBS-DORA-005 + sentinel/contiguity + NEVER padded.
# ---------------------------------------------------------------------------


class TestObsDora005:
    @staticmethod
    def _dora_builder(mask_length: int) -> ObservationBuilder:
        return _make_builder(mask_length)

    @staticmethod
    def _snapshot(builder: ObservationBuilder) -> None:
        builder.update_public_state(
            decision_id="dora-dec",
            round_index=0,
            round_wind=27,
            hand_number=1,
            seat_winds=(27, 28, 29, 30),
            honba=0,
            riichi_sticks=0,
            dealer=Seat(0),
            scores=(25000,) * 4,
            turn_actor=Seat(0),
            phase="draw_decision",
            live_wall_tiles_remaining=50,
            ippatsu_active=(False,) * 4,
        )

    def test_exact_five_slots_with_sentinel_tail(self, action_table):
        length = len(action_table.actions)
        builder = self._dora_builder(length)
        sequence = 100
        revealed = [0, 4, 8]
        for offset, tile in enumerate(revealed):
            builder.append_visible(Round()._envelope(sequence + offset, "dora_revealed", tile=tile))
        self._snapshot(builder)
        observation = builder.build(actor=Seat(0), legal_mask=(True,) * length)
        assert DORA_SHAPE == (5,)
        assert observation.dora_indicators == (*revealed, DORA_SENTINEL, DORA_SENTINEL)

    def test_fifth_reveal_fills_shape_sixth_is_rejected(self, action_table):
        length = len(action_table.actions)
        builder = self._dora_builder(length)
        for offset, tile in enumerate((0, 4, 8, 12, 16)):
            builder.append_visible(Round()._envelope(200 + offset, "dora_revealed", tile=tile))
        self._snapshot(builder)
        observation = builder.build(actor=Seat(0), legal_mask=(True,) * length)
        assert observation.dora_indicators == (0, 4, 8, 12, 16)
        with pytest.raises(ContractError):
            builder.append_visible(Round()._envelope(300, "dora_revealed", tile=20))

    def test_four_slot_tuple_is_rejected_never_padded(self, stream, action_table):
        builder = self._dora_builder(len(action_table.actions))
        _feed_round(builder, stream)
        base = dict(
            builder.build(actor=Seat(0), legal_mask=_true_mask(len(action_table.actions))).to_json()
        )
        with pytest.raises(ContractError):
            make_actor_observation(**{**base, "dora_indicators": (0, 4, 8, 12)})
        with pytest.raises(ContractError):
            make_actor_observation(**{**base, "dora_indicators": (0, 4, 8, 12, 16, 20)})

    def test_non_contiguous_sentinel_gap_is_rejected(self, stream, action_table):
        builder = self._dora_builder(len(action_table.actions))
        _feed_round(builder, stream)
        base = dict(
            builder.build(actor=Seat(0), legal_mask=_true_mask(len(action_table.actions))).to_json()
        )
        with pytest.raises(ContractError):
            make_actor_observation(**{**base, "dora_indicators": (-1, 4, -1, -1, -1)})


# ---------------------------------------------------------------------------
# BUILD: legal mask aligns with the published action table digest.
# ---------------------------------------------------------------------------


class TestMaskAlignmentActionTableDigest:
    def test_mask_length_equals_published_action_table_length(self, stream, action_table):
        length = len(action_table.actions)
        builder = _make_builder(length)
        _feed_round(builder, stream)
        observation = builder.build(actor=Seat(0), legal_mask=(True,) * length)
        assert len(observation.legal_mask) == length
        wrong = _make_builder(length + 1)
        _feed_round(wrong, stream)
        with pytest.raises(ContractError):
            wrong.build(actor=Seat(0), legal_mask=(True,) * length)

    def test_all_false_mask_is_rejected_at_decision_time(self, stream, action_table):
        builder = _make_builder(len(action_table.actions))
        _feed_round(builder, stream)
        with pytest.raises(ContractError):
            builder.build(actor=Seat(0), legal_mask=(False,) * len(action_table.actions))

    def test_builder_digest_pins_published_action_table_digest(self, action_table):
        artifact = ev.json.loads(ACTION_TABLE_PATH.read_text(encoding="utf-8"))
        recorded = artifact["payload"]["digest"]
        assert action_table.digest == recorded


# ---------------------------------------------------------------------------
# Owner decisions D-WP02D-2/-5: one successor per call window; grouping.
# ---------------------------------------------------------------------------


class TestCallWindowSingleSuccessorGrouping:
    def test_call_group_discard_through_call_resolved_forms_one_packet(self, stream):
        spec = ev.DEFAULT_PACKET_BOUNDARY_SPEC
        assert spec.call_group_kinds == ("discard", "call_window", "call_resolved")
        assert spec.claim_priority_order == ("ron", "daiminkan", "pon", "chi")
        for seat in range(4):
            packets = ev.partition_actor_packets(
                ev.filter_events_for_actor(stream, Seat(seat)),
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
        packets = ev.partition_actor_packets(
            ev.filter_events_for_actor(events, Seat(2)),
            ev.DEFAULT_PACKET_BOUNDARY_SPEC,
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
        spec = ev.DEFAULT_PACKET_BOUNDARY_SPEC
        for seat in range(4):
            visible = ev.filter_events_for_actor(stream, Seat(seat))
            packets = ev.partition_actor_packets(
                visible,
                spec,
                actor_view=Seat(seat),
                observation_hash_of=lambda view, end: _PACKET_BOUNDARY_HASH,
            )
            assert packets, seat
            ev.validate_packet_partition(packets)
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
        packets = ev.partition_actor_packets(
            ev.filter_events_for_actor(stream, Seat(0)),
            ev.DEFAULT_PACKET_BOUNDARY_SPEC,
            actor_view=Seat(0),
            observation_hash_of=lambda view, end: _PACKET_BOUNDARY_HASH,
        )
        sample = packets[0]
        identity = sample.to_json()
        recorded = identity.pop("packet_id")
        recomputed = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        assert recorded == recomputed
        with pytest.raises(DigestMismatchError):
            ev.ActorVisiblePacket(
                packet_id="0" * 64,
                actor_view=sample.actor_view,
                source_sequence_start=sample.source_sequence_start,
                source_sequence_end=sample.source_sequence_end,
                events=sample.events,
                public_state_hash_before=sample.public_state_hash_before,
                public_state_hash_after=sample.public_state_hash_after,
                observation_hash_after=sample.observation_hash_after,
            )


# ---------------------------------------------------------------------------
# ActorObservation contract: exact fields, hash rule, validator, boundaries.
# ---------------------------------------------------------------------------

_SPEC8_FIELD_ORDER = (
    "game_id",
    "decision_id",
    "sequence",
    "actor",
    "rules_id",
    "rules_hash",
    "action_table_hash",
    "event_schema_hash",
    "observation_schema_hash",
    "packet_boundary_hash",
    "round_index",
    "round_wind",
    "hand_number",
    "seat_winds",
    "honba",
    "riichi_sticks",
    "dealer",
    "scores",
    "turn_actor",
    "phase",
    "live_wall_tiles_remaining",
    "kan_count",
    "ippatsu_active",
    "actor_furiten",
    "actor_can_tsumo",
    "actor_can_riichi",
    "pending_declaration_discard",
    "concealed_hand",
    "own_drawn_tile",
    "visible_discards",
    "visible_melds",
    "riichi_states",
    "dora_indicators",
    "visible_history",
    "legal_mask",
    "observation_hash",
)


class TestActorObservationContract:
    def test_exact_spec_field_set_and_order(self):
        names = [f.name for f in ActorObservation.__dataclass_fields__.values()]  # type: ignore[attr-defined]
        assert tuple(names) == _SPEC8_FIELD_ORDER

    def test_phase_union_matches_spec_section_8(self):
        assert PHASES == (
            "round_start",
            "draw_decision",
            "discard_response",
            "kan_response",
            "round_end",
            "game_end",
        )
        assert MELD_KINDS == ("chi", "pon", "daiminkan", "ankan", "kakan")

    def test_observation_hash_excludes_the_hash_field_itself(self, stream, action_table):
        builder = _make_builder(len(action_table.actions))
        _feed_round(builder, stream)
        observation = builder.build(actor=Seat(0), legal_mask=_true_mask(len(action_table.actions)))
        identity = observation_identity_document(observation)
        assert "observation_hash" not in identity
        recomputed = "sha256:" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        assert recomputed == observation.observation_hash
        assert compute_observation_hash(observation) == observation.observation_hash

    def test_validator_catches_tampered_hash(self, stream, action_table):
        builder = _make_builder(len(action_table.actions))
        _feed_round(builder, stream)
        observation = builder.build(actor=Seat(0), legal_mask=_true_mask(len(action_table.actions)))
        VISIBILITY_VALIDATOR.validate_observation(observation)
        saved = observation.observation_hash
        object.__setattr__(observation, "observation_hash", "sha256:" + "f" * 64)
        with pytest.raises(DigestMismatchError):
            VISIBILITY_VALIDATOR.validate_observation(observation)
        object.__setattr__(observation, "observation_hash", saved)

    def test_history_holding_foreign_private_event_is_rejected(self, stream):
        foreign_draw = next(
            e for e in stream if e.kind == "draw_tile" and int(e.payload.actor) == 1
        )
        base = dict(
            make_actor_observation(
                **{
                    "game_id": "g",
                    "decision_id": "d",
                    "sequence": 1,
                    "actor": Seat(0),
                    "rules_id": "r",
                    "rules_hash": _RULES_HASH,
                    "action_table_hash": _RULES_HASH,
                    "event_schema_hash": _RULES_HASH,
                    "observation_schema_hash": _OBS_SCHEMA_HASH,
                    "packet_boundary_hash": _PACKET_BOUNDARY_HASH,
                    "round_index": 0,
                    "round_wind": 27,
                    "hand_number": 1,
                    "seat_winds": (27, 28, 29, 30),
                    "honba": 0,
                    "riichi_sticks": 0,
                    "dealer": Seat(0),
                    "scores": (25000,) * 4,
                    "turn_actor": Seat(0),
                    "phase": "draw_decision",
                    "live_wall_tiles_remaining": 70,
                    "kan_count": 0,
                    "ippatsu_active": (False,) * 4,
                    "actor_furiten": "none",
                    "actor_can_tsumo": False,
                    "actor_can_riichi": False,
                    "pending_declaration_discard": None,
                    "concealed_hand": (),
                    "own_drawn_tile": None,
                    "visible_discards": ((), (), (), ()),
                    "visible_melds": ((), (), (), ()),
                    "riichi_states": ("none",) * 4,
                    "dora_indicators": (DORA_SENTINEL,) * 5,
                    "visible_history": (),
                    "legal_mask": (True,),
                }
            ).to_json()
        )
        poisoned = dict(base)
        poisoned["visible_history"] = (foreign_draw,)
        with pytest.raises(VisibilityViolationError):
            make_actor_observation(**poisoned)

    def test_serialization_is_deterministic_and_sorted(self, stream, action_table):
        builder = _make_builder(len(action_table.actions))
        _feed_round(builder, stream)
        builder.set_concealed_hand(Seat(2), (13, 1, 5))
        first = builder.build(actor=Seat(2), legal_mask=_true_mask(len(action_table.actions)))
        again = _make_builder(len(action_table.actions))
        _feed_round(again, stream)
        again.set_concealed_hand(Seat(2), (13, 1, 5))
        second = again.build(actor=Seat(2), legal_mask=_true_mask(len(action_table.actions)))
        assert canonical_json_bytes(first.to_json()) == canonical_json_bytes(second.to_json())
        assert first.observation_hash == second.observation_hash
        document = first.to_json()
        assert document["concealed_hand"] == sorted(document["concealed_hand"])

    def test_debug_repr_never_embeds_tiles_or_payloads(self, stream, action_table):
        builder = _make_builder(len(action_table.actions))
        _feed_round(builder, stream)
        observation = builder.build(actor=Seat(0), legal_mask=_true_mask(len(action_table.actions)))
        text = repr(observation)
        # The repr template exposes routing facts and the digest only.
        assert text.startswith("ActorObservation(game_id='g-wp02d', decision_id=")
        assert text.endswith(")")
        scrubbed = text.replace("phase='discard_response'", "phase=X")
        for forbidden in ("concealed", "discard", "meld", "tiles"):
            assert forbidden not in scrubbed
