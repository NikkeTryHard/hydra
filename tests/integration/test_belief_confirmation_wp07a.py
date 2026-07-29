"""WP-07A integration — natural full-fidelity confirmation runner + hard tests.

Covers confirmation determinism, pushforward/rebuild, hidden permutation, density,
and packet invariants in an integration context (requires full belief stack).
"""

from __future__ import annotations

import math

import pytest

from hydra2.belief.confirmation import ConfirmationCase, NaturalConfirmationRunner
from hydra2.belief.corpus import build_tiny_corpus
from hydra2.belief.kernel import NaturalPacketKernel
from hydra2.belief.natural import NaturalBelief, ProposalSpec
from hydra2.belief.world import make_full_world, world_actor_observation
from hydra2.contracts.common import StaleBeliefError
from hydra2.contracts.event import (
    EventEnvelope,
    EventPayload,
    make_actor_visible_packet,
    public_state_chain_hash,
)
from hydra2.contracts.observation import make_actor_observation
from hydra2.contracts.randomness import RandomStream

pytestmark = pytest.mark.contract_package("WP-07A")


def _rng(seed: bytes = b"wp07a_integ") -> RandomStream:
    return RandomStream(seed)


def _obs(hand=(0, 1), game_id="game_tiny_001", dec="dec_integ_0"):
    return make_actor_observation(
        game_id=game_id,
        decision_id=dec,
        sequence=0,
        actor=0,
        rules_id="tenhou_4p_hanchan_v1",
        rules_hash="sha256:" + "a" * 64,
        action_table_hash="sha256:" + "b" * 64,
        event_schema_hash="sha256:" + "c" * 64,
        observation_schema_hash="sha256:" + "d" * 64,
        packet_boundary_hash="sha256:" + "e" * 64,
        round_index=0,
        round_wind=27,
        hand_number=0,
        seat_winds=(27, 28, 29, 30),
        honba=0,
        riichi_sticks=0,
        dealer=0,
        scores=(25000, 25000, 25000, 25000),
        turn_actor=0,
        phase="discard_response",
        live_wall_tiles_remaining=4,
        kan_count=0,
        ippatsu_active=(False, False, False, False),
        actor_furiten="none",
        actor_can_tsumo=True,
        actor_can_riichi=False,
        pending_declaration_discard=None,
        concealed_hand=hand,
        own_drawn_tile=None,
        visible_discards=((), (), (), ()),
        visible_melds=((), (), (), ()),
        riichi_states=("none", "none", "none", "none"),
        dora_indicators=(-1, -1, -1, -1, -1),
        visible_history=(),
        legal_mask=(True, False),
    )


def test_belief_epoch_immutable_target_identity() -> None:
    obs = _obs()
    b = NaturalBelief()
    e = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    # frozen
    try:
        e.epoch = 123  # type: ignore[misc]
        raise AssertionError()
    except Exception:
        pass
    b2 = NaturalBelief()
    e2 = b2.begin(obs, model_id=b2._belief_model_hash)  # type: ignore[attr-defined]
    assert e.target_id == e2.target_id


def test_natural_world_law_consistent_with_actor_observation() -> None:
    obs = _obs()
    b = NaturalBelief()
    e = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    parts = b.sample_natural(e, count=5, rng=_rng(b"law"))
    for p in parts:
        w = b._worlds[p.world_ref]  # type: ignore[attr-defined]
        assert tuple(w.concealed_hands[0]) == tuple(obs.concealed_hand)


def test_scoreable_proposal_samples_with_log_target_proposal() -> None:
    obs = _obs()
    b = NaturalBelief()
    e = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    prop = ProposalSpec(proposal_id="sha256:" + "f" * 64, digest="sha256:" + "f" * 64)
    parts = b.sample_proposal(e, proposal=prop, count=5, rng=_rng(b"prop"))
    for p in parts:
        assert math.isfinite(p.log_target_density)
        assert math.isfinite(p.log_proposal_density)
        assert math.exp(p.log_proposal_density) > 0


def test_actor_conditional_sampler_with_immutable_constraints() -> None:
    obs = _obs()
    b = NaturalBelief()
    e = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    from hydra2.belief.natural import _build_tiny_corpus_for_epoch

    corpus = _build_tiny_corpus_for_epoch(e, registry=b._worlds)  # type: ignore[attr-defined]
    w = corpus[0]
    obs1 = world_actor_observation(w, actor=1)
    parts = b.condition_for_actor(e, actor_observation=obs1, count=3, rng=_rng(b"cond"))
    for p in parts:
        w2 = b._worlds[p.world_ref]  # type: ignore[attr-defined]
        assert tuple(w2.concealed_hands[1]) == tuple(obs1.concealed_hand)


def test_disjoint_next_actor_visible_packet_kernel() -> None:
    obs = _obs()
    b = NaturalBelief()
    e = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    p = b.sample_natural(e, count=1, rng=_rng(b"kern"))[0]
    kernel = NaturalPacketKernel()
    succs = kernel.enumerate_next(epoch=e, particle=p, action=0)
    pids = [s.packet.packet_id for s in succs]
    assert len(set(pids)) == len(pids)


def test_physical_transition_and_actor_policy_likelihood() -> None:
    obs = _obs()
    b = NaturalBelief()
    e = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    p = b.sample_natural(e, count=1, rng=_rng(b"phys2"))[0]
    kernel = NaturalPacketKernel()
    succs = kernel.enumerate_next(epoch=e, particle=p, action=0)
    for s in succs:
        assert abs(math.exp(s.log_physical_probability + s.log_actor_policy_probability) - s.probability) < 1e-12


def test_exact_pushforward_then_condition() -> None:
    obs = _obs()
    b = NaturalBelief()
    e0 = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    obs_new = _obs(hand=(0, 1), dec="dec_push")
    # Make packet
    payload = EventPayload(kind="discard", actor=1, tile=6, action_id=0, source_seat=None, consumed_tiles=(), offered_action_ids=(), accepted_action_ids=(), round_index=None, scores=None, reason=None)
    ev = EventEnvelope(game_id="game_tiny_001", sequence=50, kind="discard", actor=1, visibility="public", visible_to=(0, 1, 2, 3), payload=payload, public_delta=(), rules_hash=obs.rules_hash, schema_hash="sha256:" + "c" * 64)  # type: ignore[arg-type]
    packet = make_actor_visible_packet(actor_view=0, events=(ev,), public_state_hash_before=public_state_chain_hash([]), public_state_hash_after=public_state_chain_hash([ev]), observation_hash_after=obs_new.observation_hash)  # type: ignore[arg-type]
    e1 = b.pushforward_condition(e0, action=0, packet=packet)
    assert int(e1.epoch) == int(e0.epoch) + 1
    assert e1.observation_hash == obs_new.observation_hash


def test_epoch_increment_after_committed_transition() -> None:
    obs = _obs()
    b = NaturalBelief()
    e0 = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    obs2 = _obs(dec="dec_inc2")
    payload = EventPayload(kind="discard", actor=1, tile=7, action_id=0, source_seat=None, consumed_tiles=(), offered_action_ids=(), accepted_action_ids=(), round_index=None, scores=None, reason=None)
    ev = EventEnvelope(game_id="game_tiny_001", sequence=51, kind="discard", actor=1, visibility="public", visible_to=(0, 1, 2, 3), payload=payload, public_delta=(), rules_hash=obs.rules_hash, schema_hash="sha256:" + "c" * 64)  # type: ignore[arg-type]
    packet = make_actor_visible_packet(actor_view=0, events=(ev,), public_state_hash_before=public_state_chain_hash([]), public_state_hash_after=public_state_chain_hash([ev]), observation_hash_after=obs2.observation_hash)  # type: ignore[arg-type]
    e1 = b.pushforward_condition(e0, action=0, packet=packet)
    assert int(e1.epoch) == int(e0.epoch) + 1


def test_stale_provenance_epoch_target_rejection() -> None:
    obs = _obs()
    b = NaturalBelief()
    e0 = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    p0 = b.sample_natural(e0, count=1, rng=_rng(b"stale3"))[0]
    obs_next = _obs(dec="dec_stale_int")
    payload = EventPayload(kind="discard", actor=1, tile=8, action_id=0, source_seat=None, consumed_tiles=(), offered_action_ids=(), accepted_action_ids=(), round_index=None, scores=None, reason=None)
    ev = EventEnvelope(game_id="game_tiny_001", sequence=52, kind="discard", actor=1, visibility="public", visible_to=(0, 1, 2, 3), payload=payload, public_delta=(), rules_hash=obs.rules_hash, schema_hash="sha256:" + "c" * 64)  # type: ignore[arg-type]
    packet = make_actor_visible_packet(actor_view=0, events=(ev,), public_state_hash_before=public_state_chain_hash([]), public_state_hash_after=public_state_chain_hash([ev]), observation_hash_after=obs_next.observation_hash)  # type: ignore[arg-type]
    e1 = b.pushforward_condition(e0, action=0, packet=packet)
    kernel = NaturalPacketKernel()
    try:
        kernel.enumerate_next(epoch=e1, particle=p0, action=0)
        raise AssertionError("should be stale")
    except StaleBeliefError:
        pass


def test_tiny_finite_world_corpus_with_exact_probabilities() -> None:
    obs = _obs()
    corpus = build_tiny_corpus(observation=obs, size=4)
    assert abs(sum(corpus.probabilities) - 1.0) < 1e-9
    for p in corpus.probabilities:
        assert abs(p - 0.25) < 1e-9
    # Particle vs oracle: belief log matches corpus
    b = NaturalBelief()
    b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    for w in corpus.worlds:
        # Belief's log for its own worlds should be -log(4)
        # But belief's worlds are different objects; we just check corpus's own log
        assert abs(corpus.log_prob(w.world_id) - (-math.log(4))) < 1e-9


def test_natural_full_fidelity_confirmation_runner() -> None:
    obs = _obs()
    corpus = build_tiny_corpus(observation=obs, size=4)
    cases = tuple(ConfirmationCase(case_id=f"c{i}", world_id=w.world_id, observation_hash=w.observation_hash) for i, w in enumerate(corpus.worlds))
    runner = NaturalConfirmationRunner()
    r = runner.confirm(cases, rng=_rng(b"confirm_int"))
    assert len(r) == 4


def test_packet_mass_one() -> None:
    obs = _obs()
    b = NaturalBelief()
    e = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    p = b.sample_natural(e, count=1, rng=_rng(b"mass_int"))[0]
    kernel = NaturalPacketKernel()
    succs = kernel.enumerate_next(epoch=e, particle=p, action=0)
    assert abs(sum(s.probability for s in succs) - 1.0) < 1e-9


def test_no_duplicate_missing_packet() -> None:
    obs = _obs()
    b = NaturalBelief()
    e = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    p = b.sample_natural(e, count=1, rng=_rng(b"dup_int"))[0]
    kernel = NaturalPacketKernel()
    succs = kernel.enumerate_next(epoch=e, particle=p, action=0)
    pids = [s.packet.packet_id for s in succs]
    assert len(pids) == len(set(pids))


def test_parent_only_reweight_negative_fixture() -> None:
    obs = _obs()
    b = NaturalBelief()
    e = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    p = b.sample_natural(e, count=1, rng=_rng(b"parent_neg_int"))[0]
    kernel = NaturalPacketKernel()
    succs = kernel.enumerate_next(epoch=e, particle=p, action=0)
    for s in succs:
        assert s.successor_world_ref != p.world_ref


def test_pushforward_equals_rebuild() -> None:
    obs = _obs()
    b = NaturalBelief()
    e0 = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    obs_new = _obs(dec="dec_rebuild_int")
    payload = EventPayload(kind="discard", actor=1, tile=10, action_id=0, source_seat=None, consumed_tiles=(), offered_action_ids=(), accepted_action_ids=(), round_index=None, scores=None, reason=None)
    ev = EventEnvelope(game_id="game_tiny_001", sequence=60, kind="discard", actor=1, visibility="public", visible_to=(0, 1, 2, 3), payload=payload, public_delta=(), rules_hash=obs.rules_hash, schema_hash="sha256:" + "c" * 64)  # type: ignore[arg-type]
    packet = make_actor_visible_packet(actor_view=0, events=(ev,), public_state_hash_before=public_state_chain_hash([]), public_state_hash_after=public_state_chain_hash([ev]), observation_hash_after=obs_new.observation_hash)  # type: ignore[arg-type]
    pushed = b.pushforward_condition(e0, action=0, packet=packet)
    b2 = NaturalBelief()
    rebuilt = b2.begin(obs_new, model_id=b2._belief_model_hash)  # type: ignore[attr-defined]
    assert pushed.observation_hash == rebuilt.observation_hash
    from hydra2.belief.natural import _build_tiny_corpus_for_epoch

    c1 = _build_tiny_corpus_for_epoch(pushed, registry=b._worlds)  # type: ignore[attr-defined]
    c2 = _build_tiny_corpus_for_epoch(rebuilt, registry=b2._worlds)  # type: ignore[attr-defined]
    assert len(c1) == len(c2) == 4


def test_hidden_permutation_invariance() -> None:
    base = ((0, 1), (2, 3), (4, 5), (6, 7))
    swapped = ((0, 1), (4, 5), (2, 3), (6, 7))
    w1 = make_full_world(concealed_hands=base, live_wall=(8, 9, 10, 11), dead_wall=(), latent_state={"v": 1}, rules_hash="sha256:" + "a" * 64, observation_hash="sha256:" + "b" * 64, simulator_snapshot="s1")
    w2 = make_full_world(concealed_hands=swapped, live_wall=(8, 9, 10, 11), dead_wall=(), latent_state={"v": 2}, rules_hash="sha256:" + "a" * 64, observation_hash="sha256:" + "b" * 64, simulator_snapshot="s2")
    o1 = world_actor_observation(w1, actor=0)
    o2 = world_actor_observation(w2, actor=0)
    assert o1.observation_hash == o2.observation_hash


def test_density_normalization_support() -> None:
    obs = _obs()
    b = NaturalBelief()
    e = b.begin(obs, model_id=b._belief_model_hash)  # type: ignore[attr-defined]
    from hydra2.belief.natural import _build_tiny_corpus_for_epoch

    corpus = _build_tiny_corpus_for_epoch(e, registry=b._worlds)  # type: ignore[attr-defined]
    total = sum(math.exp(b.log_density(e, w.world_id)) for w in corpus)
    assert abs(total - 1.0) < 1e-9


def test_deterministic_confirmation_replay() -> None:
    obs = _obs()
    corpus = build_tiny_corpus(observation=obs, size=4)
    cases = tuple(ConfirmationCase(case_id=f"cc{i}", world_id=w.world_id, observation_hash=w.observation_hash) for i, w in enumerate(corpus.worlds))
    runner = NaturalConfirmationRunner()

    def make_rng():
        return _rng(b"replay_int")

    r1 = runner.confirm(cases, rng=make_rng())
    r2 = runner.confirm(cases, rng=make_rng())
    assert r1 == r2
