"""WP-08C Candidate 2 Natural DESPOT — checklist coverage.

Implements BUILD Wave 8 WP-08C checklist:

- Natural scenarios (world, semantic randomness) only.
- No arbitrary proposal weights.
- Blueprint value is feasible lower estimate, not optimality certificate.
- Never label priority proxy upper bound without proof.
- Packet partition and proposal-reversal fixtures.
- Determinism under semantic seeds.
- Budget (calls/transitions/joules) enforcement and resource views.
- Compare policy/ISMCTS/DESPOT under declared views (smoke).
"""

from __future__ import annotations

import math
import time

import pytest

from hydra2.belief.natural import NaturalBelief
from hydra2.belief.world import make_full_world, world_actor_observation
from hydra2.contracts.action import CanonicalAction
from hydra2.contracts.common import PacketPartitionError
from hydra2.contracts.randomness import RandomStream
from hydra2.search.common import SearchRequest, candidate_spec_hash
from hydra2.search.despot_natural import (
    NaturalDespotPlanner,
    NaturalScenario,
    make_despot_candidate_spec,
    packet_aliasing_rejected,
    proposal_reversal_fixture,
    validate_packet_partition,
)

pytestmark = pytest.mark.contract_package("WP-08C")

_MASTER_RULES = "sha256:" + "a" * 64


def _world_and_obs():
    world = make_full_world(
        concealed_hands=((0, 1), (2, 3), (4, 5), (6, 7)),
        live_wall=tuple(range(8, 40)),
        dead_wall=(),
        rules_hash=_MASTER_RULES,
        observation_hash="sha256:" + "0" * 64,
    )
    obs = world_actor_observation(world, actor=0)
    return world, obs


def _legal_pair():
    a0 = CanonicalAction(
        kind="pass",
        actor=0,
        tile=None,
        called_tile=None,
        consumed_tiles=(),
        source_seat=None,
        declares_riichi=False,
        metadata=(),
    )
    a1 = CanonicalAction(
        kind="discard",
        actor=0,
        tile=0,
        called_tile=None,
        consumed_tiles=(),
        source_seat=None,
        declares_riichi=False,
        metadata=(),
    )
    return (a0, a1)


def _belief_and_epoch():
    _, obs = _world_and_obs()
    belief = NaturalBelief()
    epoch = belief.begin(obs)
    return belief, epoch, obs


# ---------------------------------------------------------------------------
# 1 Natural scenarios only
# ---------------------------------------------------------------------------


def test_despot_natural_scenarios_only() -> None:
    belief, epoch, _obs = _belief_and_epoch()
    cand = make_despot_candidate_spec(num_scenarios=4, rules_hash=_MASTER_RULES)
    planner = NaturalDespotPlanner(candidate_spec=cand, belief=belief)
    scenarios = planner._sample_natural_scenarios(
        belief_epoch=epoch, candidate_id=cand.candidate_id, case_id="case_nat", k=4
    )
    assert len(scenarios) == 4
    for sc in scenarios:
        assert isinstance(sc, NaturalScenario)
        assert sc.weight == pytest.approx(0.25)
        assert sc.log_target_density == sc.log_proposal_density
        assert math.isfinite(sc.weight) and sc.weight > 0
    # All weights sum to 1
    assert sum(sc.weight for sc in scenarios) == pytest.approx(1.0)
    # Natural sampling never consults proposal distribution; proposal_spec_hash is None for natural
    assert cand.proposal_spec_hash is None


def test_despot_no_proposal_weights() -> None:
    belief, epoch, _ = _belief_and_epoch()
    cand = make_despot_candidate_spec(num_scenarios=8, rules_hash=_MASTER_RULES)
    planner = NaturalDespotPlanner(candidate_spec=cand, belief=belief)
    # Sample twice with same case_id -> same world_refs? Deterministic same distribution
    sc1 = planner._sample_natural_scenarios(
        belief_epoch=epoch, candidate_id=cand.candidate_id, case_id="case_same", k=8
    )
    sc2 = planner._sample_natural_scenarios(
        belief_epoch=epoch, candidate_id=cand.candidate_id, case_id="case_same", k=8
    )
    for a, b in zip(sc1, sc2, strict=False):
        assert a.world_ref == b.world_ref
        assert a.semantic_seed_bytes == b.semantic_seed_bytes
        # No importance ratio; proposal equals target
        assert a.log_target_density == a.log_proposal_density


# ---------------------------------------------------------------------------
# 2 Feasible lower estimate, not bound
# ---------------------------------------------------------------------------


def test_despot_feasible_lower_not_bound() -> None:
    belief, epoch, _obs = _belief_and_epoch()
    legal = _legal_pair()
    cand = make_despot_candidate_spec(num_scenarios=4, rules_hash=_MASTER_RULES)
    planner = NaturalDespotPlanner(candidate_spec=cand, belief=belief)
    scenarios = planner._sample_natural_scenarios(
        belief_epoch=epoch, candidate_id=cand.candidate_id, case_id="case_feas", k=4
    )
    for act in legal:
        v = planner._lower_value_for_action(
            action=act, scenarios=scenarios, legal_actions=legal, candidate_id=cand.candidate_id
        )
        assert math.isfinite(v) and 0.0 <= v <= 1.0
        # Lower value is feasible; we explicitly do NOT claim it is an upper bound
        # The planner's priority proxy is labeled proxy, not bound
        proxy = planner._priority_proxy_for(act, v, visits=0)
        assert proxy >= v  # proxy is heuristic above lower, but not certified bound
        # Ensure no attribute named upper_bound exists
        assert not hasattr(proxy, "upper_bound")
        doc = (type(planner)._priority_proxy_for.__doc__ or "").lower()
        assert "not an upper bound" in doc


def test_despot_priority_proxy_not_upper_bound() -> None:
    # Verify documentation and field naming never claims bound
    assert (
        "upper bound" not in NaturalDespotPlanner._priority_proxy_for.__doc__.lower()
        or "not" in NaturalDespotPlanner._priority_proxy_for.__doc__.lower()
    )
    # The method name is proxy, not bound
    assert NaturalDespotPlanner._priority_proxy_for.__name__ == "_priority_proxy_for"
    # Source inspection: never label as bound
    src = NaturalDespotPlanner._priority_proxy_for.__doc__ or ""
    assert "NOT an upper bound" in src or "not an upper bound" in src.lower()


# ---------------------------------------------------------------------------
# 3 Packet partition and proposal reversal
# ---------------------------------------------------------------------------


def test_despot_packet_partition() -> None:
    belief, epoch, _ = _belief_and_epoch()
    kernel = planner_kernel(belief)
    rs = RandomStream(b"pkt_test")
    particles = belief.sample_natural(epoch, count=1, rng=rs)
    legal = _legal_pair()
    for act in legal:
        succ = kernel.enumerate_next(epoch=epoch, particle=particles[0], action=act)
        # Validate partition invariants
        validate_packet_partition(succ)
        assert len(succ) == 2
        total = sum(s.probability for s in succ)
        assert total == pytest.approx(1.0, abs=1e-9)
        pids = [s.packet.packet_id for s in succ]
        assert len(pids) == len(set(pids))


def planner_kernel(belief):
    from hydra2.belief.kernel import NaturalPacketKernel

    return NaturalPacketKernel()


def test_despot_proposal_reversal() -> None:
    fix = proposal_reversal_fixture()
    # Fixture proves unweighted non-natural reverses decision
    assert fix["reversal"] is True
    assert fix["correction_restores"] is True
    assert fix["natural_choice"] != fix["proposal_unweighted_choice"]
    assert fix["natural_choice"] == fix["proposal_weighted_choice"]
    # Also ensure natural DESPOT would not use proposal weights: planner never exposes weighted API
    assert not hasattr(NaturalDespotPlanner, "sample_proposal_scenarios")


def test_despot_packet_aliasing_rejected() -> None:
    # Duplicate packet_id must be rejected
    from dataclasses import dataclass

    @dataclass
    class FakeSucc:
        packet: object
        probability: float

    dup = [
        FakeSucc(packet=type("P", (), {"packet_id": "dup"})(), probability=0.5),
        FakeSucc(packet=type("P", (), {"packet_id": "dup"})(), probability=0.5),
    ]
    assert packet_aliasing_rejected(dup) is True
    ok = [
        FakeSucc(packet=type("P", (), {"packet_id": "a"})(), probability=0.5),
        FakeSucc(packet=type("P", (), {"packet_id": "b"})(), probability=0.5),
    ]
    assert packet_aliasing_rejected(ok) is False
    with pytest.raises(PacketPartitionError):
        validate_packet_partition(dup)


# ---------------------------------------------------------------------------
# 4 Determinism
# ---------------------------------------------------------------------------


def test_despot_determinism() -> None:
    belief, epoch, obs = _belief_and_epoch()
    legal = _legal_pair()
    cand = make_despot_candidate_spec(num_scenarios=8, rules_hash=_MASTER_RULES)
    planner = NaturalDespotPlanner(candidate_spec=cand, belief=belief)
    deadline = time.monotonic_ns() + 5_000_000_000
    req = SearchRequest(
        observation=obs,
        legal_actions=legal,
        candidate_spec=cand,
        deadline_monotonic_ns=deadline,
        belief_epoch=epoch,
    )
    r1 = planner.act(req)
    r2 = planner.act(req)
    assert r1.selected_action == r2.selected_action
    assert r1.value_vectors == r2.value_vectors
    assert r1.candidate_spec_hash == r2.candidate_spec_hash
    # Different case_id must be reproducible but distinct
    cand2 = make_despot_candidate_spec(num_scenarios=8, rules_hash=_MASTER_RULES)
    # Use same planner but different case via observation decision_id change
    world2 = make_full_world(
        concealed_hands=((0, 2), (1, 3), (4, 5), (6, 7)),
        live_wall=tuple(range(8, 40)),
        dead_wall=(),
        rules_hash=_MASTER_RULES,
        observation_hash="sha256:" + "1" * 64,
    )
    obs2 = world_actor_observation(world2, actor=0, game_id="game_tiny_002")
    epoch2 = belief.begin(obs2)
    req2 = SearchRequest(
        observation=obs2,
        legal_actions=legal,
        candidate_spec=cand2,
        deadline_monotonic_ns=deadline,
        belief_epoch=epoch2,
    )
    r3 = planner.act(req2)
    assert r3.selected_action in legal
    # Ensure determinism across re-instantiated planner with same seeds
    planner2 = NaturalDespotPlanner(candidate_spec=cand, belief=belief)
    r1b = planner2.act(req)
    assert r1.selected_action == r1b.selected_action


def test_despot_determinism_across_replay_with_kernel() -> None:
    # Ensure kernel enumeration is also deterministic for same seed
    belief, epoch, _ = _belief_and_epoch()
    kernel = planner_kernel(belief)
    rs1 = RandomStream(b"replay_seed")
    rs2 = RandomStream(b"replay_seed")
    p1 = belief.sample_natural(epoch, count=1, rng=rs1)[0]
    p2 = belief.sample_natural(epoch, count=1, rng=rs2)[0]
    assert p1.world_ref == p2.world_ref
    legal = _legal_pair()
    s1 = kernel.enumerate_next(epoch=epoch, particle=p1, action=legal[0])
    s2 = kernel.enumerate_next(epoch=epoch, particle=p2, action=legal[0])
    assert [s.packet.packet_id for s in s1] == [s.packet.packet_id for s in s2]


# ---------------------------------------------------------------------------
# 5 Budget enforcement and resource views
# ---------------------------------------------------------------------------


def test_despot_budget_enforcement() -> None:
    belief, epoch, obs = _belief_and_epoch()
    legal = _legal_pair()
    # Tiny budget: only 1 model call allowed, but we need 2 for lower values -> incomplete
    from hydra2.search.common import ResourceBudget

    tight = ResourceBudget(
        mode="gameplay_5s",
        deadline_ms=5000,
        fallback_margin_ms=200,
        max_model_calls=1,
        max_transitions=64,
        max_particles=4,
        max_memory_bytes=None,
    )
    cand = make_despot_candidate_spec(
        num_scenarios=4, rules_hash=_MASTER_RULES, resource_budget=tight
    )
    planner = NaturalDespotPlanner(candidate_spec=cand, belief=belief)
    deadline = time.monotonic_ns() + 5_000_000_000
    req = SearchRequest(
        observation=obs,
        legal_actions=legal,
        candidate_spec=cand,
        deadline_monotonic_ns=deadline,
        belief_epoch=epoch,
    )
    res = planner.act(req)
    # Budget exhausted -> completed False, fallback used
    assert res.completed is False
    assert res.telemetry.fallback_used is True or res.telemetry.timeout is True
    assert res.telemetry.model_calls <= 1
    # With generous budget, should complete
    generous = ResourceBudget(
        mode="gameplay_5s",
        deadline_ms=5000,
        fallback_margin_ms=200,
        max_model_calls=64,
        max_transitions=256,
        max_particles=16,
        max_memory_bytes=None,
    )
    cand2 = make_despot_candidate_spec(
        num_scenarios=4, rules_hash=_MASTER_RULES, resource_budget=generous
    )
    planner2 = NaturalDespotPlanner(candidate_spec=cand2, belief=belief)
    req2 = SearchRequest(
        observation=obs,
        legal_actions=legal,
        candidate_spec=cand2,
        deadline_monotonic_ns=deadline,
        belief_epoch=epoch,
    )
    res2 = planner2.act(req2)
    assert res2.completed is True
    assert res2.selected_action in legal


def test_despot_resource_views_calls_transitions_joules() -> None:
    belief, epoch, obs = _belief_and_epoch()
    legal = _legal_pair()
    cand = make_despot_candidate_spec(num_scenarios=8, rules_hash=_MASTER_RULES)
    planner = NaturalDespotPlanner(candidate_spec=cand, belief=belief)
    deadline = time.monotonic_ns() + 5_000_000_000
    req = SearchRequest(
        observation=obs,
        legal_actions=legal,
        candidate_spec=cand,
        deadline_monotonic_ns=deadline,
        belief_epoch=epoch,
    )
    res = planner.act(req)
    tel = res.telemetry
    # All resource dimensions are recorded
    assert isinstance(tel.model_calls, int) and tel.model_calls > 0
    assert isinstance(tel.exact_transitions, int) and tel.exact_transitions >= 0
    assert (
        isinstance(tel.energy_joules, float)
        and math.isfinite(tel.energy_joules)
        and tel.energy_joules >= 0
    )
    assert isinstance(tel.synchronized_elapsed_ms, float) and tel.synchronized_elapsed_ms >= 0
    # Joules view is deterministic from calls/transitions
    expected_joules = tel.model_calls * 0.5 + tel.exact_transitions * 0.2
    assert tel.energy_joules == pytest.approx(expected_joules)
    # Resource view comparison: different budgets should give comparable telemetry
    assert tel.particles == cand.parameters["num_scenarios"]


def test_despot_resource_views_equal_budget_comparison() -> None:
    # Simulate comparing policy / ISMCTS / DESPOT under equal calls view
    # For DESPOT, model_calls should be at least num_scenarios dependent
    belief, epoch, obs = _belief_and_epoch()
    legal = _legal_pair()
    for _view in ("calls", "transitions", "joules"):
        from hydra2.search.common import ResourceBudget

        budget = ResourceBudget(
            mode="gameplay_5s",
            deadline_ms=5000,
            fallback_margin_ms=200,
            max_model_calls=32,
            max_transitions=64,
            max_particles=6,
            max_memory_bytes=None,
        )
        cand_view = make_despot_candidate_spec(
            num_scenarios=6, rules_hash=_MASTER_RULES, resource_budget=budget
        )
        # The planner's config resource_view should match
        planner = NaturalDespotPlanner(candidate_spec=cand_view, belief=belief)
        assert planner._config.resource_view in ("calls", "transitions", "joules")
        deadline = time.monotonic_ns() + 5_000_000_000
        req = SearchRequest(
            observation=obs,
            legal_actions=legal,
            candidate_spec=cand_view,
            deadline_monotonic_ns=deadline,
            belief_epoch=epoch,
        )
        res = planner.act(req)
        assert res.telemetry.model_calls > 0


# ---------------------------------------------------------------------------
# 6 Report and candidate spec hash
# ---------------------------------------------------------------------------


def test_despot_candidate_spec_hash_stable() -> None:
    cand = make_despot_candidate_spec(num_scenarios=8, rules_hash=_MASTER_RULES)
    h1 = candidate_spec_hash(cand)
    h2 = candidate_spec_hash(cand)
    assert h1 == h2
    assert h1.startswith("sha256:")
    # Different param should change hash
    cand2 = make_despot_candidate_spec(num_scenarios=9, rules_hash=_MASTER_RULES)
    h3 = candidate_spec_hash(cand2)
    assert h3 != h1


def test_despot_report_telemetry_and_evidence() -> None:
    belief, epoch, obs = _belief_and_epoch()
    legal = _legal_pair()
    cand = make_despot_candidate_spec(num_scenarios=4, rules_hash=_MASTER_RULES)
    planner = NaturalDespotPlanner(candidate_spec=cand, belief=belief)
    deadline = time.monotonic_ns() + 5_000_000_000
    req = SearchRequest(
        observation=obs,
        legal_actions=legal,
        candidate_spec=cand,
        deadline_monotonic_ns=deadline,
        belief_epoch=epoch,
    )
    res = planner.act(req)
    assert res.candidate_spec_hash.startswith("sha256:")
    assert isinstance(res.evidence_refs, tuple) and len(res.evidence_refs) == 1
    assert res.evidence_refs[0].startswith("sha256:")
    # Telemetry must be valid ResourceTelemetry
    assert res.telemetry.candidate_spec_hash == res.candidate_spec_hash
    assert res.telemetry.mode == "gameplay_5s"
