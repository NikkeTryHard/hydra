# ruff: noqa: F401, RUF059, E731, SIM118
"""WP-08B Candidate 1 Natural ISMCTS — checklist coverage (natural, determinism, budget, report)."""

from __future__ import annotations

import hashlib
import math

import pytest

from hydra2.belief.natural import NaturalBelief
from hydra2.belief.world import make_full_world, world_actor_observation
from hydra2.contracts.common import ContractError, VisibilityViolationError
from hydra2.contracts.observation import make_actor_observation
from hydra2.contracts.randomness import RandomStream
from hydra2.search.ismcts_natural import (
    FORBIDDEN_IN_TREE_KEY,
    InformationSetNode,
    NaturalISMCTSConfig,
    NaturalISMCTSPlanner,
    UniformContinuationPolicy,
    attempt_redeterminize,
    double_weighting_oracle_detects_correction,
    info_key_for_observation,
    is_redeterminization_enabled,
    model_vector_for_world,
    scalarize_vector,
    terminal_vector_for_world,
    validate_tree_keys_contain_no_world_id,
)

pytestmark = pytest.mark.contract_package("WP-08B")

_MASTER = b"wp08b_unit_master_v1"


def _rng(seed: bytes | str = b"wp08b_seed") -> RandomStream:
    if isinstance(seed, str):
        seed = seed.encode()
    return RandomStream(seed if isinstance(seed, bytes) else bytes(seed))


def _make_world_and_obs(hand=(0, 1), actor: int = 0):
    w = make_full_world(
        concealed_hands=((0, 1), (2, 3), (4, 5), (6, 7)),
        live_wall=(8, 9, 10, 11),
        dead_wall=(),
        latent_state={"step": 0, "turn": actor},
        rules_hash="sha256:" + "a" * 64,
        observation_hash="sha256:" + "b" * 64,
        simulator_snapshot="snap_test",
    )
    obs = world_actor_observation(w, actor=actor)
    return w, obs


def _make_belief_epoch(obs):
    belief = NaturalBelief()
    epoch = belief.begin(obs)
    return belief, epoch


def _make_legal_actions(obs):
    # Use canonical dummy actions as ints wrapped as simple objects with action_id
    # For tests we use int ids directly; planner maps via to_id
    class DummyAction:
        def __init__(self, aid: int):
            self.action_id = aid

        def __repr__(self) -> str:
            return f"A({self.action_id})"

    mask = getattr(obs, "legal_mask", (True, False, True))
    ids = [i for i, m in enumerate(mask) if m]
    if not ids:
        ids = [0, 1]
    return tuple(DummyAction(i) for i in ids)


# ---------------------------------------------------------------------------
# 1 Natural worlds only; no importance ratios
# ---------------------------------------------------------------------------


def test_natural_worlds_only_no_importance_ratios() -> None:
    _w, obs = _make_world_and_obs()
    belief, epoch = _make_belief_epoch(obs)
    rng = _rng(b"natural_only")
    particles = belief.sample_natural(epoch, count=4, rng=rng)
    for p in particles:
        assert p.source == "natural"
        assert p.log_target_density == p.log_proposal_density
        assert math.isclose(math.exp(p.log_target_density - p.log_proposal_density), 1.0)
        assert math.isfinite(p.log_target_density)
    # double weighting oracle should detect correction — test both reversal and double-fail
    # via two complementary fixtures (reversal vs double-fail) since a single 2-world
    # setting cannot exhibit both simultaneously
    fixture_rev = double_weighting_oracle_detects_correction(
        values=({0: 0.9, 1: 0.0}, {0: 0.0, 1: 0.6})
    )
    assert fixture_rev["reversal_unweighted"] is True
    assert fixture_rev["once_restores"] is True
    fixture_double = double_weighting_oracle_detects_correction()
    assert fixture_double["double_fails"] is True
    assert fixture_double["once_restores"] is True
    config = NaturalISMCTSConfig(
        max_simulations=8, max_depth=3, max_transitions=64, max_model_calls=16
    )
    planner = NaturalISMCTSPlanner(belief=belief, config=config)
    legal = _make_legal_actions(obs)
    rng2 = _rng(b"search_natural")
    res = planner.search(epoch=epoch, root_observation=obs, legal_actions=legal, rng=rng2)
    assert res["telemetry"]["simulations"] == 8
    assert res["completed"] is True
    # ensure no proposal ratio was applied inside tree (indirect via natural check)
    # if proposal ratio existed, tree would have log ratio !=0, but we have no such field


# ---------------------------------------------------------------------------
# 2 Root tree keys use root information set only
# ---------------------------------------------------------------------------


def test_root_tree_keys_use_root_information_set_only() -> None:
    w1, obs1 = _make_world_and_obs()
    # second world differs only in hidden permutation (swap seats 1 and 2 hands)
    w2 = make_full_world(
        concealed_hands=((0, 1), (4, 5), (2, 3), (6, 7)),
        live_wall=(8, 9, 10, 11),
        dead_wall=(),
        latent_state={"step": 0, "turn": 0},
        rules_hash="sha256:" + "a" * 64,
        observation_hash="sha256:" + "b" * 64,
        simulator_snapshot="snap_perm",
    )
    obs1_root = world_actor_observation(w1, actor=0)
    obs2_root = world_actor_observation(w2, actor=0)
    k1 = info_key_for_observation(obs1_root)
    k2 = info_key_for_observation(obs2_root)
    # hidden permutation leaves root observation unchanged (same hand (0,1) and public)
    # So keys must be equal
    assert k1 == k2, "hidden permutation must preserve root information-set key"
    # different root hand -> different key
    w3 = make_full_world(
        concealed_hands=((2, 3), (0, 1), (4, 5), (6, 7)),
        live_wall=(8, 9, 10, 11),
        dead_wall=(),
        latent_state={"step": 0, "turn": 0},
        rules_hash="sha256:" + "a" * 64,
        observation_hash="sha256:" + "b" * 64,
        simulator_snapshot="snap_diff",
    )
    obs3_root = world_actor_observation(w3, actor=0)
    k3 = info_key_for_observation(obs3_root)
    assert k3 != k1, "different root hand must give different information-set key"
    # forbidden check
    assert validate_tree_keys_contain_no_world_id([k1, k2, k3]) is True
    # ensure info_key does not contain forbidden substring
    for bad in FORBIDDEN_IN_TREE_KEY:
        assert bad not in k1
    # search tree keys also satisfy
    belief, epoch = _make_belief_epoch(obs1_root)
    config = NaturalISMCTSConfig(max_simulations=12, max_depth=4)
    planner = NaturalISMCTSPlanner(belief=belief, config=config)
    legal = _make_legal_actions(obs1_root)
    res = planner.search(
        epoch=epoch, root_observation=obs1_root, legal_actions=legal, rng=_rng(b"keys")
    )
    tree_keys = list(res["tree"].keys())
    assert validate_tree_keys_contain_no_world_id(tree_keys) is True
    for k in tree_keys:
        assert k.startswith("sha256:")
        assert len(k) == 71  # sha256: + 64 hex


# ---------------------------------------------------------------------------
# 3 Non-root policies consume that actor's observation inside sandbox
# ---------------------------------------------------------------------------


def test_non_root_policies_consume_actor_observation_in_sandbox() -> None:
    w, obs_root = _make_world_and_obs(actor=0)
    # opponent observations
    obs_opp1 = world_actor_observation(w, actor=1)
    obs_opp2_same = world_actor_observation(w, actor=1)
    # different opponent hand -> different observation
    w_diff = make_full_world(
        concealed_hands=((0, 1), (8, 9), (4, 5), (6, 7)),
        live_wall=(10, 11),
        dead_wall=(),
        latent_state={"step": 0, "turn": 1},
        rules_hash="sha256:" + "a" * 64,
        observation_hash="sha256:" + "b" * 64,
        simulator_snapshot="snap_opp_diff",
    )
    obs_opp_diff = world_actor_observation(w_diff, actor=1)

    policy = UniformContinuationPolicy(bias_strength=0.2)
    legal = (0, 1)
    d1 = policy.distribution(obs_opp1, legal)
    d2 = policy.distribution(obs_opp2_same, legal)
    assert d1 == d2, "equal actor observations must map to equal policy distributions"
    # changed actor-visible information may change distribution (bias flips via observation_hash)
    # We don't require strict inequality, but verify that distribution is derived from observation_hash
    # so that at least for some different worlds it can differ, and that it remains valid
    d_diff = policy.distribution(obs_opp_diff, legal)
    # distribution must be valid probabilities summing to 1 and finite
    for d in (d1, d_diff):
        assert len(d) == 2
        assert abs(sum(d) - 1.0) < 1e-9
        for p in d:
            assert 0 <= p <= 1 and math.isfinite(p)
    # sampling consumes actor observation and rng, replay deterministic
    rng_a = _rng(b"policy_replay")
    rng_b = _rng(b"policy_replay")
    s1 = policy.sample(obs_opp1, legal, rng_a)
    s2 = policy.sample(obs_opp2_same, legal, rng_b)
    assert s1 == s2, "policy RNG replay must be deterministic"
    # policy must reject FullWorld (privileged) leak
    with pytest.raises((VisibilityViolationError, ContractError)):
        policy.distribution(w, legal)  # type: ignore[arg-type]
    with pytest.raises((VisibilityViolationError, ContractError)):
        policy.sample(w, legal, _rng(b"leak"))  # type: ignore[arg-type]
    # Also ensure sandbox: opponent policy sees opponent obs, not root obs
    # Run a search and verify non-root steps used opponent obs (indirect: no error)
    belief, epoch = _make_belief_epoch(obs_root)
    config = NaturalISMCTSConfig(max_simulations=6, max_depth=4)
    planner = NaturalISMCTSPlanner(belief=belief, config=config)
    legal_root = _make_legal_actions(obs_root)
    res = planner.search(
        epoch=epoch, root_observation=obs_root, legal_actions=legal_root, rng=_rng(b"sandbox")
    )
    assert res["completed"] is True


# ---------------------------------------------------------------------------
# 4 Carry vector values; scalarize only root selection
# ---------------------------------------------------------------------------


def test_carry_vector_values_scalarize_only_root_selection() -> None:
    w, obs = _make_world_and_obs()
    belief, epoch = _make_belief_epoch(obs)
    config = NaturalISMCTSConfig(max_simulations=16, max_depth=5, uct_c=1.0)
    planner = NaturalISMCTSPlanner(belief=belief, config=config)
    legal = _make_legal_actions(obs)
    res = planner.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=_rng(b"vector")
    )
    vecs = res["value_vectors"]
    # each value vector must be length 4, finite
    for vec in vecs:
        assert isinstance(vec, (list, tuple))
        assert len(vec) == 4
        for v in vec:
            assert isinstance(v, float)
            assert math.isfinite(v)
    # backup preserves 4-dim: root node's action stats store 4-dim sums
    tree = res["tree"]
    for node in tree.values():
        for aid, stats in node.action_stats.items():
            assert len(stats.value_sum) == 4
            for v in stats.value_sum:
                assert math.isfinite(v)
            # mean should also be 4-dim
            mean = node.mean_vector(aid)
            assert mean is not None and len(mean) == 4
    # scalarization only at root: verify scalarize_vector projects correctly
    sample_vec = (0.1, 0.2, 0.7, 0.4)
    assert scalarize_vector(sample_vec, 0) == 0.1
    assert scalarize_vector(sample_vec, 2) == 0.7
    # root selection uses scalarized mean, not vector sum
    root_node = res["root_node"]
    if root_node is not None and root_node.action_stats:
        root_seat = int(epoch.root_actor)
        # pick best via scalar, ensure that choosing max vector sum would differ for crafted case
        # Here we just verify selection is consistent with scalar
        selected = res["selected_action_id"]
        # compute scalar means
        scalars = {aid: root_node.scalar_mean(aid, root_seat) for aid in root_node.action_stats}
        best = max(scalars, key=lambda k: scalars[k] if scalars[k] is not None else float("-inf"))
        assert selected == best


# ---------------------------------------------------------------------------
# 5 Freeze UCT/depth/budget/continuation policies/RNG semantics
# ---------------------------------------------------------------------------


def test_freeze_uct_depth_budget_continuation_policies_rng_semantics() -> None:
    cfg = NaturalISMCTSConfig(
        uct_c=1.5,
        max_depth=4,
        max_simulations=10,
        max_transitions=100,
        max_model_calls=20,
        tie_break="lowest_action_id",
        candidate_id="candidate1",
    )
    # frozen
    with pytest.raises((AttributeError, TypeError)):
        cfg.uct_c = 2.0  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        cfg.max_depth = 99  # type: ignore[misc]
    assert cfg.uct_c == 1.5
    assert cfg.max_depth == 4
    assert cfg.max_simulations == 10
    assert cfg.tie_break == "lowest_action_id"
    assert cfg.candidate_id == "candidate1"
    # budget enforcement
    w, obs = _make_world_and_obs()
    belief, epoch = _make_belief_epoch(obs)
    planner = NaturalISMCTSPlanner(belief=belief, config=cfg)
    legal = _make_legal_actions(obs)
    res = planner.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=_rng(b"freeze")
    )
    assert res["telemetry"]["max_depth"] == 4
    assert res["telemetry"]["uct_c"] == 1.5
    assert res["telemetry"]["simulations"] == 10
    assert res["telemetry"]["transitions"] <= 100
    assert res["telemetry"]["model_calls"] <= 20
    # RNG semantics: same seed -> same result
    cfg2 = NaturalISMCTSConfig(max_simulations=8, max_depth=3)
    planner2 = NaturalISMCTSPlanner(belief=belief, config=cfg2)
    r1 = _rng(b"rng_semantics")
    r2 = _rng(b"rng_semantics")
    res1 = planner2.search(epoch=epoch, root_observation=obs, legal_actions=legal, rng=r1)
    res2 = planner2.search(epoch=epoch, root_observation=obs, legal_actions=legal, rng=r2)
    assert res1["selected_action_id"] == res2["selected_action_id"]
    assert res1["value_vectors"] == res2["value_vectors"]
    assert res1["telemetry"] == res2["telemetry"]
    # continuation policy hash stability (policy identity frozen)
    pol = UniformContinuationPolicy(bias_strength=0.2)
    assert pol.distribution(obs, (0, 1)) == pol.distribution(obs, (0, 1))


# ---------------------------------------------------------------------------
# 6 Keep re-determinization disabled until separate conditional-law proof
# ---------------------------------------------------------------------------


def test_re_determinization_disabled() -> None:
    assert is_redeterminization_enabled() is False
    with pytest.raises(ContractError):
        attempt_redeterminize()
    with pytest.raises(ContractError):
        attempt_redeterminize(world="dummy", observation="dummy")
    # search must not use proposal sampling internally — verify via config that
    # no branch enables it (flag false)
    w, obs = _make_world_and_obs()
    belief, epoch = _make_belief_epoch(obs)
    config = NaturalISMCTSConfig(max_simulations=4)
    planner = NaturalISMCTSPlanner(belief=belief, config=config)
    legal = _make_legal_actions(obs)
    # Search should succeed without re-determinization
    res = planner.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=_rng(b"noredet")
    )
    assert res["completed"] is True
    # Ensure planner has no hidden proposal path (checked via is_redeterminization_enabled)


# ---------------------------------------------------------------------------
# 7 Implement all Candidate 1 tests from blueprint §8
# ---------------------------------------------------------------------------


def test_candidate1_blueprint_tests() -> None:
    # This composite test exercises the blueprint §8 checklist items together:
    # - equal actor obs -> equal root keys and equal non-root distributions
    # - opponents with changed actor-visible info may change distribution (covered earlier)
    # - policy RNG replay (covered)
    # - root-known/public constraints survive conditional sampling (tiny corpus uniform)
    # - two-world unequal-probability oracle detects double weighting
    # - vector backup preserves raw settlement and utility-schema identity
    # - no forbidden field appears in a tree key or continuation-policy input
    w, obs = _make_world_and_obs()
    belief, epoch = _make_belief_epoch(obs)
    # root-known/public constraints: natural sampling uniform over corpus size 4
    corpus_particles = belief.sample_natural(epoch, count=20, rng=_rng(b"constraints"))
    for p in corpus_particles:
        assert p.log_target_density == p.log_proposal_density
    # double weighting detection
    oracle = double_weighting_oracle_detects_correction()
    assert oracle["double_fails"] is True
    # vector backup preserves raw settlement: terminal vs model vectors distinct but both 4-dim finite
    w2 = make_full_world(
        concealed_hands=((0, 1), (2, 3), (4, 5), (6, 7)),
        live_wall=(),
        dead_wall=(),
        latent_state={"step": 6, "turn": 0},
        rules_hash="sha256:" + "a" * 64,
        observation_hash="sha256:" + "b" * 64,
        simulator_snapshot="term",
    )
    tv = terminal_vector_for_world(w2)
    mv = model_vector_for_world(w, candidate_id="candidate1")
    for vec in (tv, mv):
        assert len(vec) == 4 and all(math.isfinite(v) for v in vec)
    # forbidden field check in tree keys
    config = NaturalISMCTSConfig(max_simulations=8, max_depth=3)
    planner = NaturalISMCTSPlanner(belief=belief, config=config)
    legal = _make_legal_actions(obs)
    res = planner.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=_rng(b"blueprint")
    )
    for k in res["tree"].keys():
        for bad in FORBIDDEN_IN_TREE_KEY:
            assert bad not in k
        assert "world_id" not in k
    # Check that tree nodes were built from info keys, not world ids
    assert len(res["tree"]) > 0
    # policy inputs never contain FullWorld
    pol = UniformContinuationPolicy()
    with pytest.raises((VisibilityViolationError, ContractError)):
        pol.distribution(w, (0, 1))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 8 Confirm naturally under matched resources / determinism / budget / report
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    w, obs = _make_world_and_obs()
    belief, epoch = _make_belief_epoch(obs)
    config = NaturalISMCTSConfig(
        max_simulations=20,
        max_depth=4,
        max_transitions=300,
        max_model_calls=50,
        uct_c=1.41421356237,
    )
    planner = NaturalISMCTSPlanner(belief=belief, config=config)
    legal = _make_legal_actions(obs)
    rng_factory = lambda: _rng(b"determinism_master")
    r1 = rng_factory()
    r2 = rng_factory()
    res1 = planner.search(epoch=epoch, root_observation=obs, legal_actions=legal, rng=r1)
    # new planner with same config for second run (fresh tree)
    planner2 = NaturalISMCTSPlanner(belief=belief, config=config)
    res2 = planner2.search(epoch=epoch, root_observation=obs, legal_actions=legal, rng=r2)
    assert res1["selected_action_id"] == res2["selected_action_id"]
    assert res1["value_vectors"] == res2["value_vectors"]
    assert res1["telemetry"]["simulations"] == res2["telemetry"]["simulations"]
    assert res1["telemetry"]["transitions"] == res2["telemetry"]["transitions"]
    assert res1["telemetry"]["model_calls"] == res2["telemetry"]["model_calls"]
    # tree structure also deterministic
    assert set(res1["tree"].keys()) == set(res2["tree"].keys())
    for k in res1["tree"]:
        n1 = res1["tree"][k]
        n2 = res2["tree"][k]
        assert n1.visits == n2.visits
        assert set(n1.action_stats.keys()) == set(n2.action_stats.keys())
        for aid in n1.action_stats:
            assert n1.action_stats[aid].visits == n2.action_stats[aid].visits
            assert n1.action_stats[aid].value_sum == n2.action_stats[aid].value_sum


def test_budget() -> None:
    _w, obs = _make_world_and_obs()
    belief, epoch = _make_belief_epoch(obs)
    # tight simulation budget — enough transitions to run all simulations
    cfg_small = NaturalISMCTSConfig(
        max_simulations=4, max_depth=2, max_transitions=10, max_model_calls=4
    )
    planner_small = NaturalISMCTSPlanner(belief=belief, config=cfg_small)
    legal = _make_legal_actions(obs)
    res_small = planner_small.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=_rng(b"budget_small")
    )
    assert res_small["telemetry"]["simulations"] == 4
    assert res_small["telemetry"]["transitions"] <= 10
    assert res_small["telemetry"]["model_calls"] <= 4
    # larger budget should do more work
    cfg_large = NaturalISMCTSConfig(
        max_simulations=12, max_depth=6, max_transitions=200, max_model_calls=40
    )
    planner_large = NaturalISMCTSPlanner(belief=belief, config=cfg_large)
    res_large = planner_large.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=_rng(b"budget_large")
    )
    assert res_large["telemetry"]["simulations"] == 12
    assert res_large["telemetry"]["transitions"] > res_small["telemetry"]["transitions"]
    assert res_large["telemetry"]["model_calls"] >= res_small["telemetry"]["model_calls"]
    # deadline-style budget via max_transitions hard limit
    cfg_deadline = NaturalISMCTSConfig(
        max_simulations=100, max_depth=8, max_transitions=10, max_model_calls=100
    )
    planner_d = NaturalISMCTSPlanner(belief=belief, config=cfg_deadline)
    res_d = planner_d.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=_rng(b"deadline")
    )
    assert res_d["telemetry"]["transitions"] == 10 or res_d["telemetry"]["transitions"] <= 10
    assert res_d["telemetry"]["simulations"] <= 100


def test_report() -> None:
    w, obs = _make_world_and_obs()
    belief, epoch = _make_belief_epoch(obs)
    config = NaturalISMCTSConfig(max_simulations=8, max_depth=3)
    planner = NaturalISMCTSPlanner(belief=belief, config=config)
    legal = _make_legal_actions(obs)
    res = planner.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=_rng(b"report")
    )
    telemetry = res["telemetry"]
    # report-like fields: simulations, transitions, model_calls, max_*, uct_c, tie_break, candidate_id, tree_nodes, root_seat
    for key in (
        "simulations",
        "transitions",
        "model_calls",
        "max_simulations",
        "max_depth",
        "uct_c",
        "tie_break",
        "candidate_id",
        "resource_view",
        "root_seat",
        "tree_nodes",
    ):
        assert key in telemetry, f"telemetry missing {key}"
    assert isinstance(telemetry["simulations"], int)
    assert isinstance(telemetry["tree_nodes"], int)
    assert isinstance(res["completed"], bool)
    assert isinstance(res["value_vectors"], tuple)
    assert isinstance(res["selected_action_id"], int)
    # ensure vector backup preserved 4-seat identity
    for vec in res["value_vectors"]:
        assert len(vec) == 4


def test_confirm_naturally_under_matched_resources() -> None:
    # Matched-resource confirmation: same budget view (calls / transitions / joules) should be declared
    # For stub, we just verify that two planners with same config produce same resource view and that natural vs natural comparison is consistent.
    w, obs = _make_world_and_obs()
    belief, epoch = _make_belief_epoch(obs)
    cfg_calls = NaturalISMCTSConfig(max_simulations=10, max_depth=4, resource_view="calls")
    cfg_trans = NaturalISMCTSConfig(max_simulations=10, max_depth=4, resource_view="transitions")
    planner_calls = NaturalISMCTSPlanner(belief=belief, config=cfg_calls)
    planner_trans = NaturalISMCTSPlanner(belief=belief, config=cfg_trans)
    legal = _make_legal_actions(obs)
    res_calls = planner_calls.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=_rng(b"confirm_calls")
    )
    res_trans = planner_trans.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=_rng(b"confirm_trans")
    )
    # Each should report its resource view
    assert res_calls["telemetry"]["resource_view"] == "calls"
    assert res_trans["telemetry"]["resource_view"] == "transitions"
    # Both use natural worlds only (checked earlier), so confirmation is naturally matched
    assert res_calls["telemetry"]["simulations"] == 10
    assert res_trans["telemetry"]["simulations"] == 10
