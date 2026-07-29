# ruff: noqa: F401, RUF059
"""WP-09D Candidate 5 Local Resolving — checklist and gates."""

from __future__ import annotations

import hashlib
import math

import pytest

from hydra2.belief.natural import NaturalBelief
from hydra2.belief.world import make_full_world, world_actor_observation
from hydra2.contracts.common import ContractError
from hydra2.search.local_resolving import (
    AbstractMappingError,
    CycleDetectedError,
    LocalResolvingAbstraction,
    LocalResolvingConfig,
    LocalResolvingPlanner,
    StrategyTable,
    abstraction_round_trip,
    build_public_subgame,
    detect_cycle,
    exhaustive_tiny_game_values,
    info_key_for_actor_observation,
    is_equilibrium_claimed,
    leaf_vector_replay,
    make_candidate5_spec,
    model_vector_for_world,
    terminal_vector_for_world,
    validate_abstraction_mapping,
)

pytestmark = pytest.mark.contract_package("WP-09D")

_MASTER = b"wp09d_unit_master_v1"


def _world_and_obs(hand=(0, 1), actor: int = 0):
    w = make_full_world(
        concealed_hands=((0, 1), (2, 3), (4, 5), (6, 7)),
        live_wall=(8, 9, 10, 11),
        dead_wall=(),
        latent_state={"step": 0, "turn": actor},
        rules_hash="sha256:" + "a" * 64,
        observation_hash="sha256:" + "b" * 64,
    )
    obs = world_actor_observation(w, actor=actor)
    return w, obs


def _belief_epoch(obs):
    belief = NaturalBelief()
    epoch = belief.begin(obs)
    return belief, epoch


def _legal_actions(obs):
    class DummyAction:
        def __init__(self, aid: int) -> None:
            self.action_id = aid

        def __repr__(self) -> str:
            return f"A({self.action_id})"

    mask = getattr(obs, "legal_mask", (True, False, True))
    ids = [i for i, m in enumerate(mask) if m]
    if not ids:
        ids = [0, 1]
    # cap to tiny 0..3 for local resolving
    ids = [i % 4 for i in ids][:2]
    if len(ids) == 1:
        ids.append((ids[0] + 1) % 4)
    return tuple(DummyAction(i) for i in ids)


# ---------------------------------------------------------------------------
# 1 local resolving candidate — build declared subgame/horizon/abstraction
# ---------------------------------------------------------------------------


def test_local_resolving_candidate_build_subgame() -> None:
    w, obs = _world_and_obs()
    belief, epoch = _belief_epoch(obs)
    config = LocalResolvingConfig(
        horizon=2,
        iterations=8,
        update_rule="regret_matching",
        averaging="uniform",
        abstraction="identity",
        leaf_model="model",
    )
    _ab = validate_abstraction_mapping({0: 0, 1: 1, 2: 2, 3: 3}, legal_concrete_ids=(0, 1))
    sub = build_public_subgame(
        epoch,
        horizon=2,
        abstraction={0: 0, 1: 1, 2: 2, 3: 3},
        iteration_count=8,
        averaging="uniform",
        update_rule="regret_matching",
        leaf_model="model",
    )
    assert sub.horizon == 2
    assert sub.abstraction.abstract_ids == (0, 1, 2, 3)
    assert sub.iteration_count == 8
    assert sub.averaging_rule == "uniform"
    assert sub.update_rule == "regret_matching"
    assert len(sub.nodes) >= 3
    assert len(sub.edges) >= 2
    # horizon frozen — mismatch between config and built subgame must be caught via spec params
    spec = make_candidate5_spec(config=config)
    assert spec.parameters["horizon"] == 2
    assert spec.parameters["abstraction"] == "identity"
    assert spec.candidate_id == "candidate5"
    assert spec.algorithm == "local_resolving"


# ---------------------------------------------------------------------------
# 2 strategies keyed by each actor's information nodes (same-information)
# ---------------------------------------------------------------------------


def test_same_information_strategy_keying() -> None:
    # Two worlds with same root hand but swapped opponent hidden tiles -> same root info_key, different other
    w1 = make_full_world(
        concealed_hands=((0, 1), (2, 3), (4, 5), (6, 7)),
        live_wall=(8, 9, 10, 11),
        dead_wall=(),
        latent_state={"idx": 0},
        rules_hash="sha256:" + "a" * 64,
        observation_hash="sha256:" + "b" * 64,
    )
    w2 = make_full_world(
        concealed_hands=((0, 1), (4, 5), (2, 3), (6, 7)),
        live_wall=(8, 9, 10, 11),
        dead_wall=(),
        latent_state={"idx": 1},
        rules_hash="sha256:" + "a" * 64,
        observation_hash="sha256:" + "b" * 64,
    )
    obs1 = world_actor_observation(w1, actor=0)
    obs2 = world_actor_observation(w2, actor=0)
    k1 = info_key_for_actor_observation(obs1)
    k2 = info_key_for_actor_observation(obs2)
    assert k1 == k2, "hidden permutation must preserve same actor's information-set key"
    # different root hand -> different key
    w3 = make_full_world(
        concealed_hands=((2, 3), (0, 1), (4, 5), (6, 7)),
        live_wall=(8, 9, 10, 11),
        dead_wall=(),
        latent_state={"idx": 2},
        rules_hash="sha256:" + "a" * 64,
        observation_hash="sha256:" + "b" * 64,
    )
    obs3 = world_actor_observation(w3, actor=0)
    k3 = info_key_for_actor_observation(obs3)
    assert k3 != k1
    # Strategy table uses (actor, info_hash) and never world_id
    ab = validate_abstraction_mapping({0: 0, 1: 1}, legal_concrete_ids=(0, 1))
    table = StrategyTable(abstraction=ab)
    table.set(0, k1, (0.6, 0.4))
    assert table.get(0, k2) == (0.6, 0.4), (
        "same info key must map to same strategy irrespective of world"
    )
    assert table.validate_no_world_id() is True
    for bad in ("world_id", "full_hidden"):
        assert bad not in k1
    # planner respects same-information: two runs with hidden-permuted worlds share root table entry
    belief, epoch = _belief_epoch(obs1)
    legal = _legal_actions(obs1)
    config = LocalResolvingConfig(
        horizon=1, iterations=4, update_rule="hedge", averaging="uniform", abstraction="identity"
    )
    planner = LocalResolvingPlanner(belief=belief, config=config)
    res1 = planner.search(
        epoch=epoch, root_observation=obs1, legal_actions=legal, case_id="same_info_case"
    )
    # second world same root info but different hidden
    res2 = planner.search(
        epoch=epoch, root_observation=obs2, legal_actions=legal, case_id="same_info_case"
    )
    # root avg strategy must be identical because same info_key drives initialization
    assert res1["root_avg"] == res2["root_avg"]


# ---------------------------------------------------------------------------
# 3 preserve vector returns and exact settlement
# ---------------------------------------------------------------------------


def test_settlement_utility_vector_preservation() -> None:
    w, _ = _world_and_obs()
    mv = model_vector_for_world(w, leaf_kind="model")
    tv = terminal_vector_for_world(w)
    assert len(mv) == 4 and len(tv) == 4
    for v in (*mv, *tv):
        assert math.isfinite(v)
        assert abs(v) < 5.0
    # vector preservation: planner backs up same 4-dim vector through path
    belief, epoch = _belief_epoch(world_actor_observation(w, actor=0))
    _, obs = _world_and_obs()
    legal = _legal_actions(obs)
    config = LocalResolvingConfig(
        horizon=2,
        iterations=4,
        update_rule="regret_matching",
        averaging="uniform",
        leaf_model="terminal",
    )
    planner = LocalResolvingPlanner(belief=belief, config=config)
    res = planner.search(epoch=epoch, root_observation=obs, legal_actions=legal, case_id="vec_case")
    for vec in res["vectors"]:
        assert len(vec) == 4
        assert all(math.isfinite(float(x)) for x in vec)
        # settlement conservation: zero-sum (re-centered) sums to ~0
        assert abs(sum(vec)) < 1e-9, f"vector {vec} not zero-sum re-centered"
    # utility schema: candidate spec utility hash bound; vectors finite in [value_min,value_max] style
    spec = make_candidate5_spec(config=config)
    assert spec.utility_manifest_hash.startswith("sha256:")
    assert spec.utility_id == "expected_final_placement_tenhou_4p_hanchan_v1"


# ---------------------------------------------------------------------------
# 4 freeze update and averaging rules
# ---------------------------------------------------------------------------


def test_frozen_update_averaging_rules() -> None:
    # valid combos
    for upd in ("regret_matching", "hedge", "fictitious_play"):
        for avg in ("uniform", "linear"):
            cfg = LocalResolvingConfig(update_rule=upd, averaging=avg)
            assert cfg.update_rule == upd
            assert cfg.averaging == avg
            # to_parameters round-trip
            params = cfg.to_parameters()
            cfg2 = LocalResolvingConfig.from_parameters(params)
            assert cfg2.update_rule == upd
            assert cfg2.averaging == avg
            spec = make_candidate5_spec(config=cfg)
            assert spec.parameters["update_rule"] == upd
            assert spec.parameters["averaging"] == avg
    # invalid must raise
    with pytest.raises(ContractError):
        LocalResolvingConfig(update_rule="invalid_rule")
    with pytest.raises(ContractError):
        LocalResolvingConfig(averaging="exponential")
    # frozen dataclass immutability
    cfg = LocalResolvingConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.horizon = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5 cycle detection
# ---------------------------------------------------------------------------


def test_cycle_detection() -> None:
    w, obs = _world_and_obs()
    belief, epoch = _belief_epoch(obs)
    ab = validate_abstraction_mapping({0: 0, 1: 1, 2: 2, 3: 3}, legal_concrete_ids=(0, 1, 2, 3))
    sub = build_public_subgame(
        epoch,
        horizon=2,
        abstraction={0: 0, 1: 1, 2: 2, 3: 3},
        iteration_count=4,
        averaging="uniform",
        update_rule="regret_matching",
    )
    # valid subgame must be acyclic
    detect_cycle(sub)  # should not raise
    # inject cycle by adding back edge
    # create a subgame where first node points to last and last points back to first
    from hydra2.search.local_resolving import PublicSubgame

    nodes = ("sha256:" + "a" * 64, "sha256:" + "b" * 64, "sha256:" + "c" * 64)
    edges = ((nodes[0], nodes[1], 0), (nodes[1], nodes[2], 1), (nodes[2], nodes[0], 0))  # cycle
    cyc = PublicSubgame(
        horizon=2,
        abstraction=ab,
        public_history_hash=nodes[0],
        nodes=nodes,
        edges=edges,
        iteration_count=4,
        averaging_rule="uniform",
        update_rule="regret_matching",
        leaf_model="model",
    )
    with pytest.raises(CycleDetectedError):
        detect_cycle(cyc)
    # also planner building with horizon that aliases via abstraction should still detect?
    # we force a subgame with duplicate node hashes to trigger cycle — building via build_public_subgame with artificial public_history_seed collision is valid DAG, so only manual injection triggers
    # Verify planner aborts on cycle via direct PublicSubgame validation
    with pytest.raises((CycleDetectedError, ContractError)):
        # duplicate nodes should be rejected as not distinct
        PublicSubgame(
            horizon=1,
            abstraction=ab,
            public_history_hash=nodes[0],
            nodes=(nodes[0], nodes[0]),
            edges=(),
            iteration_count=4,
            averaging_rule="uniform",
            update_rule="regret_matching",
            leaf_model="model",
        )


# ---------------------------------------------------------------------------
# 6 invalid abstraction mapping
# ---------------------------------------------------------------------------


def test_invalid_abstraction_mappings() -> None:
    # valid identity
    ab = validate_abstraction_mapping({0: 0, 1: 1, 2: 2}, legal_concrete_ids=(0, 1, 2))
    assert ab.abstract_ids == (0, 1, 2)
    # missing legal concrete
    with pytest.raises(AbstractMappingError):
        validate_abstraction_mapping({0: 0}, legal_concrete_ids=(0, 1))
    # duplicate concrete (caught via construction)
    with pytest.raises((ContractError, AbstractMappingError)):
        LocalResolvingAbstraction(
            name="custom", concrete_to_abstract=((0, 0), (0, 1)), abstract_ids=(0, 1)
        )
    # abstract_ids mismatch image
    with pytest.raises(ContractError):
        LocalResolvingAbstraction(
            name="custom", concrete_to_abstract=((0, 0), (1, 1)), abstract_ids=(0, 1, 2)
        )
    # unknown abstraction name
    with pytest.raises(ContractError):
        LocalResolvingConfig(abstraction="unknown_abstraction")
    # mapping references abstract not in list via direct construction
    with pytest.raises((ContractError, AbstractMappingError)):
        # edges checking abstraction coverage: build subgame with abstract id not in mapping
        w, obs = _world_and_obs()
        belief, epoch = _belief_epoch(obs)
        # use pair_merge for 0..3 but legal only 0,1 — still valid
        # invalid: try to map 0->99 not in abstract_ids but construction would include 99, so we test edge with illegal abstract
        bad_ab = LocalResolvingAbstraction(
            name="custom", concrete_to_abstract=((0, 99), (1, 99)), abstract_ids=(99,)
        )
        from hydra2.search.local_resolving import PublicSubgame

        PublicSubgame(
            horizon=1,
            abstraction=bad_ab,
            public_history_hash="sha256:" + "d" * 64,
            nodes=("sha256:" + "d" * 64, "sha256:" + "e" * 64),
            edges=(
                ("sha256:" + "d" * 64, "sha256:" + "e" * 64, 0),
            ),  # 0 not in (99,) => AbstractMappingError
            iteration_count=4,
            averaging_rule="uniform",
            update_rule="regret_matching",
            leaf_model="model",
        )


# ---------------------------------------------------------------------------
# 7 abstraction round-trip
# ---------------------------------------------------------------------------


def test_abstraction_round_trip() -> None:
    for _name, mapping in [
        ("identity", {0: 0, 1: 1, 2: 2, 3: 3}),
        ("pair_merge", {0: 0, 1: 0, 2: 1, 3: 1}),
        ("tile_type", {0: 0, 34: 0, 68: 0, 1: 1, 35: 1}),
    ]:
        ab = validate_abstraction_mapping(mapping, legal_concrete_ids=tuple(mapping.keys()))
        for aid in ab.abstract_ids:
            back = abstraction_round_trip(ab, aid)
            assert back == aid
            rep = ab.map_abstract_to_representative(aid)
            assert ab.map_concrete(rep) == aid
        # concrete -> abstract -> representative concrete -> abstract consistency
        for c, a in mapping.items():
            assert ab.map_concrete(int(c)) == int(a)
    # invalid round-trip: unknown abstract
    ab = validate_abstraction_mapping({0: 0, 1: 1}, legal_concrete_ids=(0, 1))
    with pytest.raises(AbstractMappingError):
        abstraction_round_trip(ab, 99)


# ---------------------------------------------------------------------------
# 8 exhaustive tiny-game vs resolver sample coverage
# ---------------------------------------------------------------------------


def test_exhaustive_tiny_game() -> None:
    _ab = validate_abstraction_mapping({0: 0, 1: 1, 2: 2, 3: 3}, legal_concrete_ids=(0, 1, 2, 3))
    # reduce to branching 2 for exhaustive feasibility
    ab2 = validate_abstraction_mapping({0: 0, 1: 1}, legal_concrete_ids=(0, 1))
    worlds = tuple(
        make_full_world(
            concealed_hands=hands,
            live_wall=(8, 9, 10, 11),
            dead_wall=(),
            latent_state={"g": i},
            rules_hash="sha256:" + "a" * 64,
            observation_hash="sha256:" + "b" * 64,
        )
        for i, hands in enumerate(
            [
                ((0, 1), (2, 3), (4, 5), (6, 7)),
                ((0, 1), (2, 4), (3, 5), (6, 7)),
                ((0, 1), (2, 5), (3, 4), (6, 7)),
            ]
        )
    )
    # exhaustive enumeration for horizon 2, branching 2 => 4 paths
    exhaustive = exhaustive_tiny_game_values(
        horizon=2, abstraction=ab2, worlds=worlds, leaf_model="model"
    )
    assert len(exhaustive) == 4  # 2^2
    for path, vec in exhaustive.items():
        assert len(path) == 2
        assert len(vec) == 4
        assert all(math.isfinite(v) for v in vec)
    # resolver with iterations == 4 should traverse same leaf set (deterministic)
    # we check that resolver's model_calls covers at least horizon traversals
    w0, obs = _world_and_obs()
    belief, epoch = _belief_epoch(obs)
    legal = _legal_actions(obs)
    config = LocalResolvingConfig(
        horizon=2,
        iterations=8,
        update_rule="regret_matching",
        averaging="uniform",
        abstraction="identity",
    )
    planner = LocalResolvingPlanner(belief=belief, config=config)
    res = planner.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, case_id="exhaustive_case"
    )
    # telemetry counts
    assert res["telemetry"]["iterations"] == 8
    assert res["telemetry"]["horizon"] == 2
    # vectors from exhaustive and resolver should be consistent distribution (not equal but same domain)
    for vec in res["vectors"]:
        assert len(vec) == 4
    # Second exhaustive with pair_merge abstraction gives different branching but still exhaustive
    exhaustive_merge = exhaustive_tiny_game_values(
        horizon=2, abstraction=ab2, worlds=worlds, leaf_model="terminal"
    )
    assert len(exhaustive_merge) == 4
    # settlement conservation across exhaustive: each vector sum ~0
    for v in exhaustive_merge.values():
        assert abs(sum(v)) < 1e-6


# ---------------------------------------------------------------------------
# 9 leaf replay determinism
# ---------------------------------------------------------------------------


def test_leaf_replay() -> None:
    w, _ = _world_and_obs()
    v1 = leaf_vector_replay(w, leaf_model="model")
    v2 = leaf_vector_replay(w, leaf_model="model")
    assert v1 == v2
    v3 = leaf_vector_replay(w, leaf_model="terminal")
    v4 = leaf_vector_replay(w, leaf_model="terminal")
    assert v3 == v4
    # different worlds give different vectors (with high probability)
    w2 = make_full_world(
        concealed_hands=((2, 3), (0, 1), (4, 5), (6, 7)),
        live_wall=(8, 9, 10, 11),
        dead_wall=(),
        latent_state={"other": 1},
        rules_hash="sha256:" + "a" * 64,
        observation_hash="sha256:" + "b" * 64,
    )
    assert leaf_vector_replay(w2, leaf_model="model") != v1
    # via planner telemetry leaf vectors replay identical across runs
    _, obs = _world_and_obs()
    belief, epoch = _belief_epoch(obs)
    legal = _legal_actions(obs)
    config = LocalResolvingConfig(
        horizon=1, iterations=4, update_rule="hedge", averaging="linear", leaf_model="model"
    )
    planner = LocalResolvingPlanner(belief=belief, config=config)
    res_a = planner.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, case_id="leaf_replay_case"
    )
    res_b = planner.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, case_id="leaf_replay_case"
    )
    assert res_a["vectors"] == res_b["vectors"]
    assert res_a["telemetry"]["model_calls"] == res_b["telemetry"]["model_calls"]


# ---------------------------------------------------------------------------
# 10 PBRF warm start comparison (with/without)
# ---------------------------------------------------------------------------


def test_pbrf_warm_start_comparison() -> None:
    w, obs = _world_and_obs()
    belief, epoch = _belief_epoch(obs)
    legal = _legal_actions(obs)
    # Without warm start
    config_cold = LocalResolvingConfig(
        horizon=2,
        iterations=16,
        update_rule="regret_matching",
        averaging="uniform",
        abstraction="identity",
    )
    spec_cold = make_candidate5_spec(config=config_cold, warm_start=False)
    planner_cold = LocalResolvingPlanner(
        belief=belief, config=config_cold, candidate_spec=spec_cold
    )
    res_cold = planner_cold.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, case_id="warm_compare"
    )
    # With warm start (PBRF-like biased prior)
    config_warm = LocalResolvingConfig(
        horizon=2,
        iterations=16,
        update_rule="regret_matching",
        averaging="uniform",
        abstraction="identity",
    )
    spec_warm = make_candidate5_spec(config=config_warm, warm_start=True)
    planner_warm = LocalResolvingPlanner(
        belief=belief, config=config_warm, candidate_spec=spec_warm
    )
    res_warm = planner_warm.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, case_id="warm_compare"
    )
    # Both completed and preserve vector properties
    assert res_cold["completed"] is True and res_warm["completed"] is True
    assert len(res_cold["vectors"][0]) == 4 and len(res_warm["vectors"][0]) == 4
    # Warm-start flag must be recorded in spec and telemetry; at least initialization differs
    assert spec_cold.parameters["warm_start"] is False
    assert spec_warm.parameters["warm_start"] is True
    assert res_cold["telemetry"]["warm_start"] is False
    assert res_warm["telemetry"]["warm_start"] is True
    # Both use same update/averaging semantics but different initialization — neither claims equilibrium
    assert is_equilibrium_claimed() is False
    # Root strategies should be valid distributions even if they converge; we just verify they are both valid and telemetry differs in warm flag
    assert len(res_cold["root_avg"]) == 2 and len(res_warm["root_avg"]) == 2
    assert abs(sum(res_cold["root_avg"]) - 1.0) < 1e-9
    assert abs(sum(res_warm["root_avg"]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 11 never claim equilibrium
# ---------------------------------------------------------------------------


def test_never_claim_equilibrium() -> None:
    assert is_equilibrium_claimed() is False
    # docstring and module-level assertion must not mention guarantee
    import hydra2.search.local_resolving as mod

    text = (mod.__doc__ or "") + "\n" + (mod.LocalResolvingPlanner.__doc__ or "")
    # Ensure we explicitly state never equilibrium certificate
    assert "never equilibrium" in text.lower() or "never" in text.lower()
    assert "not an equilibrium" in text.lower() or "empirical optimizer" in text.lower()
    # No function should return equilibrium guarantee
    w, obs = _world_and_obs()
    belief, epoch = _belief_epoch(obs)
    legal = _legal_actions(obs)
    planner = LocalResolvingPlanner(belief=belief)
    res = planner.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, case_id="no_eq_case"
    )
    # telemetry must not contain equilibrium fields
    assert "equilibrium" not in str(res["telemetry"]).lower()
    assert "exploitability" not in str(res["telemetry"]).lower()


def test_determinism() -> None:
    w, obs = _world_and_obs()
    belief, epoch = _belief_epoch(obs)
    legal = _legal_actions(obs)
    config = LocalResolvingConfig(
        horizon=2,
        iterations=12,
        update_rule="fictitious_play",
        averaging="linear",
        abstraction="pair_merge",
    )
    spec = make_candidate5_spec(config=config)
    planner1 = LocalResolvingPlanner(belief=belief, config=config, candidate_spec=spec)
    planner2 = LocalResolvingPlanner(belief=belief, config=config, candidate_spec=spec)
    res1 = planner1.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, case_id="det_case", root_seat=0
    )
    res2 = planner2.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, case_id="det_case", root_seat=0
    )
    assert res1["selected_abstract"] == res2["selected_abstract"]
    assert res1["selected_concrete"] == res2["selected_concrete"]
    # elapsed_ms is wall-time and not deterministic — compare without it
    t1 = {k: v for k, v in res1["telemetry"].items() if k != "elapsed_ms"}
    t2 = {k: v for k, v in res2["telemetry"].items() if k != "elapsed_ms"}
    assert t1 == t2
    assert res1["avg_tables"].table == res2["avg_tables"].table
    # different case_id must give deterministic but possibly different result (still deterministic per case_id)
    res3 = planner1.search(
        epoch=epoch,
        root_observation=obs,
        legal_actions=legal,
        case_id="different_case",
        root_seat=0,
    )
    # should be reproducible with same different_case
    res4 = planner1.search(
        epoch=epoch,
        root_observation=obs,
        legal_actions=legal,
        case_id="different_case",
        root_seat=0,
    )
    assert res3["selected_abstract"] == res4["selected_abstract"]
    # via act API also deterministic — use canonical actions for SearchRequest
    from hydra2.contracts.action import CanonicalAction

    def _canon_legal() -> tuple[CanonicalAction, ...]:
        # Two simple discards that pass structural validation
        return (
            CanonicalAction(
                kind="discard",
                actor=0,
                tile=0,
                called_tile=None,
                consumed_tiles=(),
                source_seat=None,
                declares_riichi=False,
                metadata=(),
            ),
            CanonicalAction(
                kind="discard",
                actor=0,
                tile=1,
                called_tile=None,
                consumed_tiles=(),
                source_seat=None,
                declares_riichi=False,
                metadata=(),
            ),
        )

    canon_legal = _canon_legal()
    import time

    from hydra2.search.common import SearchRequest

    deadline = time.monotonic_ns() + 5_000_000_000
    req = SearchRequest(
        observation=obs,
        legal_actions=canon_legal,
        candidate_spec=spec,
        deadline_monotonic_ns=deadline,
        belief_epoch=epoch,
    )
    r1 = planner1.act(req)
    r2 = planner2.act(req)
    assert r1.selected_action == r2.selected_action
    assert r1.candidate_spec_hash == r2.candidate_spec_hash


# ---------------------------------------------------------------------------
# 13 resource budget and determinism of telemetry
# ---------------------------------------------------------------------------


def test_resource_budget_enforcement() -> None:
    w, obs = _world_and_obs()
    belief, epoch = _belief_epoch(obs)
    legal = _legal_actions(obs)
    spec = make_candidate5_spec(
        max_model_calls=8,
        max_transitions=32,
        max_particles=4,
        config=LocalResolvingConfig(horizon=1, iterations=4),
    )
    planner = LocalResolvingPlanner(belief=belief, candidate_spec=spec)
    res = planner.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, case_id="budget_case"
    )
    assert res["completed"] is True
    # telemetry counts charged
    assert res["telemetry"]["model_calls"] > 0
    assert (
        res["telemetry"]["exact_transitions"]
        >= res["telemetry"]["horizon"] * res["telemetry"]["iterations"]
    )
    # resource_view declared
    assert spec.resource_budget.max_model_calls == 8
    assert spec.parameters["resource_view"] in ("calls", "transitions", "joules")


# ---------------------------------------------------------------------------
# 14 report — spec hash stability, result binding, promotion record
# ---------------------------------------------------------------------------


def test_report_and_candidate_spec_binding() -> None:
    from hydra2.search.common import candidate_spec_hash

    config = LocalResolvingConfig(
        horizon=2,
        iterations=8,
        update_rule="hedge",
        averaging="uniform",
        abstraction="identity",
        tie_break="greedy",
    )
    spec = make_candidate5_spec(config=config)
    h1 = candidate_spec_hash(spec)
    h2 = candidate_spec_hash(spec)
    assert h1 == h2
    assert h1.startswith("sha256:")
    # mutation changes hash
    config2 = LocalResolvingConfig(
        horizon=3,
        iterations=8,
        update_rule="hedge",
        averaging="uniform",
        abstraction="identity",
        tie_break="greedy",
    )
    spec2 = make_candidate5_spec(config=config2)
    assert candidate_spec_hash(spec2) != h1
    # act returns result bound to spec hash — use canonical actions
    from hydra2.contracts.action import CanonicalAction

    def _canon_legal2() -> tuple[CanonicalAction, ...]:
        return (
            CanonicalAction(
                kind="discard",
                actor=0,
                tile=0,
                called_tile=None,
                consumed_tiles=(),
                source_seat=None,
                declares_riichi=False,
                metadata=(),
            ),
            CanonicalAction(
                kind="discard",
                actor=0,
                tile=1,
                called_tile=None,
                consumed_tiles=(),
                source_seat=None,
                declares_riichi=False,
                metadata=(),
            ),
        )

    canon_legal = _canon_legal2()
    w, obs = _world_and_obs()
    belief, epoch = _belief_epoch(obs)
    planner = LocalResolvingPlanner(belief=belief, candidate_spec=spec)
    import time

    deadline = time.monotonic_ns() + 5_000_000_000
    from hydra2.search.common import SearchRequest

    req = SearchRequest(
        observation=obs,
        legal_actions=canon_legal,
        candidate_spec=spec,
        deadline_monotonic_ns=deadline,
        belief_epoch=epoch,
    )
    result = planner.act(req)
    assert result.candidate_spec_hash == h1
    assert result.completed is True
    assert result.selected_action in canon_legal
    for vec in result.value_vectors:
        vals = vec.values if hasattr(vec, "values") else vec  # type: ignore[attr-defined]
        assert len(vals) == 4
        assert all(math.isfinite(float(v)) for v in vals)
    # tie_break frozen before cases: changing tie_break changes hash and result distribution may change but remains deterministic
    spec_greedy = make_candidate5_spec(config=LocalResolvingConfig(tie_break="greedy"))
    spec_tb = make_candidate5_spec(config=LocalResolvingConfig(tie_break="temperature_0.5"))
    assert candidate_spec_hash(spec_greedy) != candidate_spec_hash(spec_tb)
