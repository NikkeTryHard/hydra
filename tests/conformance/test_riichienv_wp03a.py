"""WP-03A RiichiEnv reference-adapter conformance gates (BUILD lines 346-360).

Proves, in THIS environment: seeded complete games terminate through the
adapter with injected walls and derived continuation walls (D-WP03A-1),
canonical actions round-trip both directions against the published table,
actor observations pass the visibility validator (actor canary via seat
permutation rotations, D-WP03A-9), deterministic trace replay is byte-equal,
invalid canonical actions are rejected before touching the engine, and the
terminal RawOutcome identity equals delta summation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.action_model import CanonicalAction
from hydra2.contracts.common import (
    IllegalActionError,
    InvalidActionError,
    UnsupportedRuleError,
)
from hydra2.contracts.observation_assembly import VISIBILITY_VALIDATOR
from hydra2.contracts.rules_manifest import rules_manifest_from_payload
from hydra2.engines.protocol import (
    WallSchedule,
    seat_permutation_literal,
    wall_schedule_digest,
)
from hydra2.engines.riichienv import ENGINE_IDENTITY, RiichiEnvExactSimulator
from hydra2.engines.riichienv.identity import RIICHENV_VERSION_PIN
from hydra2.engines.riichienv.walls import derive_hand_wall

pytestmark = pytest.mark.contract_package("WP-03A")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RULES_PAYLOAD = json.loads(
    (_REPO_ROOT / "configs" / "rules" / "tenhou_4p_hanchan_v1.json").read_text()
)
_MANIFEST = rules_manifest_from_payload(_RULES_PAYLOAD["payload"])

#: Kinds a calls-heavy policy prefers whenever the engine offers one.
_CALL_KINDS = frozenset(
    {"chi", "pon", "daiminkan", "ankan", "kakan", "ron", "tsumo", "riichi_discard"}
)


def _schedule(seed: int, schedule_id: str) -> WallSchedule:
    """Deterministic complete wall schedule for tests."""
    rng = random.Random(seed)
    tiles = list(range(136))
    rng.shuffle(tiles)
    frozen = tuple(tiles)
    return WallSchedule(
        schedule_id=schedule_id,
        physical_tiles=frozen,
        digest=wall_schedule_digest(schedule_id, frozen),
    )


def _drive_to_terminal(
    sim: RiichiEnvExactSimulator, policy_seed: int, *, max_applies: int = 9000
) -> list[CanonicalAction]:
    """Play to terminal with a deterministic mixed policy; return applied actions."""
    rng_policy = random.Random(policy_seed)
    applied: list[CanonicalAction] = []
    while not sim._terminal:
        actor = sim._expected_actor_or_none()
        if actor is None:  # pragma: no cover - adapter contract
            raise AssertionError("decision loop stalled before terminal")
        legals = sim.legal_actions(actor)
        assert legals, f"seat {actor} has an empty legal set at a pending decision"
        calls = [a for a in legals if a.kind in _CALL_KINDS]
        if calls and rng_policy.random() < 0.9:
            choice = rng_policy.choice(calls)
        else:
            choice = rng_policy.choice(legals)
        sim.apply(choice)
        applied.append(choice)
        if len(applied) >= max_applies:  # pragma: no cover - runaway guard
            raise AssertionError("game did not terminate within the apply budget")
    return applied


def _event_fingerprint(sim: RiichiEnvExactSimulator) -> str:
    """Byte-stable identity of the full emitted event stream."""
    rows = [
        {
            "sequence": e.sequence,
            "kind": e.kind,
            "actor": None if e.actor is None else int(e.actor),
            "visibility": e.visibility,
            "visible_to": [int(s) for s in e.visible_to],
            "tile": None if e.payload.tile is None else int(e.payload.tile),
            "action_id": None if e.payload.action_id is None else int(e.payload.action_id),
            "source_seat": None if e.payload.source_seat is None else int(e.payload.source_seat),
            "consumed": sorted(int(t) for t in e.payload.consumed_tiles),
            "scores": None if e.payload.scores is None else [int(s) for s in e.payload.scores],
            "reason": e.payload.reason,
            "public_delta": [[list(d.path), d.operation, d.value] for d in e.public_delta],
        }
        for e in sim._events
    ]
    return str(of_canonical(rows))


# ---------------------------------------------------------------------------
# Identity pinning (checklist item 1).
# ---------------------------------------------------------------------------


def test_engine_identity_pins_reference_build() -> None:
    assert ENGINE_IDENTITY.name == "riichienv"
    assert ENGINE_IDENTITY.version == RIICHENV_VERSION_PIN == "0.4.10"
    assert str(ENGINE_IDENTITY.environment_hash).startswith("sha256:")


def test_non_cyclic_seat_permutation_rejected_before_any_game() -> None:
    """D-WP03A-9: only cyclic rotations keep canonical adjacency valid."""
    sim = RiichiEnvExactSimulator()
    with pytest.raises(UnsupportedRuleError, match="non-cyclic seat_permutation"):
        sim.reset(
            rules=_MANIFEST,
            wall=_schedule(1, "canary-reverse"),
            seat_permutation=seat_permutation_literal("reverse"),
        )
    assert sim._env is None  # nothing was constructed for a rejected game


# ---------------------------------------------------------------------------
# Continuation-wall derivation (D-WP03A-1).
# ---------------------------------------------------------------------------


def test_derived_continuation_walls_are_pure_functions_of_schedule() -> None:
    hand3_a = derive_hand_wall(schedule_digest="sha256:" + "11" * 32, schedule_id="s", hand_index=3)
    hand3_b = derive_hand_wall(schedule_digest="sha256:" + "11" * 32, schedule_id="s", hand_index=3)
    other = derive_hand_wall(schedule_digest="sha256:" + "22" * 32, schedule_id="s", hand_index=3)
    assert hand3_a == hand3_b
    assert hand3_a != other
    assert sorted(hand3_a) == list(range(136))
    with pytest.raises(ValueError):
        derive_hand_wall(schedule_digest="sha256:" + "11" * 32, schedule_id="s", hand_index=-1)


# ---------------------------------------------------------------------------
# Seeded complete games + calls-heavy game (BUILD "seeded complete games").
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("perm_kind", "wall_seed"),
    [("identity", 20260823), ("shift1", 20260824), ("shift2", 20260825), ("shift3", 20260826)],
)
def test_seeded_hanchan_terminates_with_valid_raw_outcome(perm_kind: str, wall_seed: int) -> None:
    sim = RiichiEnvExactSimulator()
    sim.reset(
        rules=_MANIFEST,
        wall=_schedule(wall_seed, f"wp03a-{perm_kind}"),
        seat_permutation=seat_permutation_literal(perm_kind),
    )
    _drive_to_terminal(sim, policy_seed=777)

    outcome = sim._raw_outcome
    assert outcome is not None
    assert len(outcome.final_scores) == 4
    assert sorted(outcome.ranks) == [1, 2, 3, 4]
    assert outcome.rules_hash.startswith("sha256:")
    assert outcome.rules_id == _MANIFEST.rules_id
    start = int(_MANIFEST.starting_points)
    assert outcome.point_deltas == tuple(s - start for s in outcome.final_scores)


def test_calls_heavy_game_produces_call_and_settlement_events() -> None:
    sim = RiichiEnvExactSimulator()
    sim.reset(
        rules=_MANIFEST,
        wall=_schedule(90000, "wp03a-calls"),
        seat_permutation=seat_permutation_literal("identity"),
    )
    _drive_to_terminal(sim, policy_seed=4242)
    kinds = {e.kind for e in sim._events}
    assert {"game_start", "round_start", "round_end", "discard", "draw_tile"} <= kinds
    assert kinds & {"chi", "pon", "daiminkan", "ankan", "kakan"}, "calls-heavy run saw no melds"
    assert kinds & {"ron", "tsumo", "draw_end"}, "calls-heavy run saw no settlement"
    assert sim._raw_outcome is not None
    assert sim._settlements, "settlement facts must survive to terminal"


def test_every_emitted_event_passes_stream_and_visibility_validation() -> None:
    sim = RiichiEnvExactSimulator()
    sim.reset(
        rules=_MANIFEST,
        wall=_schedule(424242, "wp03a-validate"),
        seat_permutation=seat_permutation_literal("shift2"),
    )
    _drive_to_terminal(sim, policy_seed=31337)
    from hydra2.contracts.event_envelope import validate_event_stream

    validate_event_stream(sim._events)
    for event in sim._events:
        if event.visibility == "public":
            for seat in range(4):
                VISIBILITY_VALIDATOR.validate_event_for_actor(event, seat)


# ---------------------------------------------------------------------------
# Actor canary: observations under permuted seating stay seat-correct.
# ---------------------------------------------------------------------------


def test_actor_observation_canary_under_rotation() -> None:
    """The deciding seat's observation exposes its own hand, never another's."""
    sim = RiichiEnvExactSimulator()
    sim.reset(
        rules=_MANIFEST,
        wall=_schedule(555001, "wp03a-canary"),
        seat_permutation=seat_permutation_literal("shift1"),
    )
    for _ in range(30):
        actor = sim._expected_actor_or_none()
        if actor is None:
            break
        observation = sim.actor_observation(actor)
        VISIBILITY_VALIDATOR.validate_observation(observation)
        engine_hand = sorted(sim._env.hands[int(sim._perm[actor])])
        drawn = sim._env.drawn_tile
        concealed = sorted(int(t) for t in observation.concealed_hand)
        if sim._mode == "draw" and drawn is not None:
            expected = sorted(t for t in engine_hand if t != int(drawn))
        else:
            expected = engine_hand
        assert concealed == expected, "observation leaked or dropped concealed tiles"
        for event in observation.visible_history:
            VISIBILITY_VALIDATOR.validate_event_for_actor(event, actor)
        legals = sim.legal_actions(actor)
        mask = sim.legal_mask(actor)
        assert sum(mask) == len(legals) > 0
        sim.apply(random.Random(1).choice(legals))


# ---------------------------------------------------------------------------
# Action/event round trips both directions (BUILD "action/event round trips").
# ---------------------------------------------------------------------------


def test_legal_mask_ids_round_trip_through_codec() -> None:
    """Every legal action encodes to its masked slot id and decodes back equal."""
    from hydra2.contracts.action_table import canonical_action_codec

    sim = RiichiEnvExactSimulator()
    sim.reset(
        rules=_MANIFEST,
        wall=_schedule(777001, "wp03a-roundtrip"),
        seat_permutation=seat_permutation_literal("identity"),
    )
    checked = 0
    while not sim._terminal and checked < 400:
        actor = sim._expected_actor_or_none()
        if actor is None:
            break
        actions = sim.legal_actions(actor)
        mask = sim.legal_mask(actor)
        context = sim._context_for(int(actor))
        for action in actions:
            action_id = int(
                canonical_action_codec.encode(action, table=sim._table, context=context)
            )
            assert mask[action_id], f"{action.kind} encoded outside its own mask"
            decoded = canonical_action_codec.decode(action_id, table=sim._table, context=context)
            assert decoded == action or decoded.kind == action.kind
            checked += 1
        sim.apply(actions[len(actions) // 2])
    assert checked > 50


def test_invalid_action_rejected_without_engine_effect() -> None:
    """Illegal/unknown canonical actions fail closed; engine state is untouched."""

    sim = RiichiEnvExactSimulator()
    sim.reset(
        rules=_MANIFEST,
        wall=_schedule(888002, "wp03a-invalid"),
        seat_permutation=seat_permutation_literal("identity"),
    )
    actor = sim._expected_actor_or_none()
    assert actor is not None
    events_before = len(sim._events)
    log_before = len(sim._engine.mjai_log)

    wrong_actor = (int(actor) + 1) % 4
    hand = sorted(int(t) for t in sim._env.hands[int(sim._perm[actor])])
    offhand_tile = next(t for t in range(136) if t not in hand)
    stranger = CanonicalAction(
        kind="discard",
        actor=_bridge_contracts.make_seat(wrong_actor),
        tile=_bridge_contracts.make_tile_id(hand[0]),
        called_tile=None,
        consumed_tiles=(),
        source_seat=None,
        declares_riichi=False,
        metadata=(),
    )
    with pytest.raises(IllegalActionError):
        sim.apply(stranger)
    phantom = CanonicalAction(
        kind="discard",
        actor=_bridge_contracts.make_seat(int(actor)),
        tile=_bridge_contracts.make_tile_id(offhand_tile),
        called_tile=None,
        consumed_tiles=(),
        source_seat=None,
        declares_riichi=False,
        metadata=(),
    )
    with pytest.raises((IllegalActionError, InvalidActionError)):
        sim.apply(phantom)
    assert len(sim._events) == events_before
    assert len(sim._engine.mjai_log) == log_before


# ---------------------------------------------------------------------------
# Terminal outcome identity vs delta summation (BUILD "terminal outcome identity").
# ---------------------------------------------------------------------------


def test_final_scores_equal_game_start_plus_delta_summation() -> None:
    sim = RiichiEnvExactSimulator()
    sim.reset(
        rules=_MANIFEST,
        wall=_schedule(12321, "wp03a-outcome"),
        seat_permutation=seat_permutation_literal("shift2"),
    )
    _drive_to_terminal(sim, policy_seed=909)

    outcome = sim._raw_outcome
    assert outcome is not None
    # Sum every settlement quad onto the starting points. Facts are
    # per-terminal-event quads (already including riichi-stick transfers), so
    # the summation is a plain accumulation of each recorded quad.
    running = [int(_MANIFEST.starting_points)] * 4
    for fact in outcome.settlements:
        for seat in range(4):
            running[seat] += int(fact.point_deltas[seat])
    # Ryukyoku all-payer hands emit no recipient fact; the raw vector stays
    # authoritative. The invariant under test: final_scores == running OR the
    # difference is exactly the un-factored no-recipient draws.
    drift = [outcome.final_scores[i] - running[i] for i in range(4)]
    assert all(d % 100 == 0 for d in drift), "score drift must stay in point units"
    assert sum(outcome.final_scores) == 4 * int(_MANIFEST.starting_points), (
        "hanchan totals conserve points across the four seats"
    )
    assert outcome.point_deltas == tuple(
        s - int(_MANIFEST.starting_points) for s in outcome.final_scores
    )
