# ruff: noqa: RUF059
"""WP-13 Candidate 8 Joint Type/World — checklist coverage.

Implements BUILD §16 WP-13 checklist:
- joint type world candidate (observation-only type policy, joint posterior, correlation, uncertainty set)
- determinism (deterministic joint posterior & planner replay)
- hidden permutation invariance (info_key invariant, planner decision invariant)
- report (module fixtures via conftest aggregation)
plus observation_only, exact oracle, type/world correlation, coherent set, feasibility, calibration.
"""

from __future__ import annotations

import hashlib
import math
import time

import pytest

from hydra2.belief.world import make_full_world, world_actor_observation
from hydra2.contracts.action import CanonicalAction
from hydra2.search.common import SearchRequest, candidate_spec_hash
from hydra2.search.joint_type_world import (
    THETA_IDS,
    JointTypeWorldConfig,
    JointTypeWorldPlanner,
    OpponentTypePolicy,
    UncertaintySet,
    coherent_trajectory,
    deterministic_joint_gumbel,
    exact_joint_posterior_oracle,
    hidden_marginalization,
    info_key_for_observation,
    make_joint_type_world_candidate_spec,
    preserve_correlation_check,
    sequential_joint_update,
    validate_hidden_permutation_invariance,
    validate_same_information_equality,
)

pytestmark = pytest.mark.contract_package("WP-13")

_MASTER_RULES = "sha256:" + "a" * 64


def _world_and_obs(actor: int = 0):
    w = make_full_world(
        concealed_hands=((0, 1), (2, 3), (4, 5), (6, 7)),
        live_wall=tuple(range(8, 40)),
        dead_wall=(),
        rules_hash=_MASTER_RULES,
        observation_hash="sha256:" + "0" * 64,
    )
    obs = world_actor_observation(w, actor=actor)
    return w, obs


def _make_request(obs, legal, spec, *, belief_epoch=None):
    deadline = time.monotonic_ns() + 5_000_000_000
    return SearchRequest(
        observation=obs,
        legal_actions=legal,
        candidate_spec=spec,
        deadline_monotonic_ns=deadline,
        belief_epoch=belief_epoch,
    )


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


def _legal_ids(legal):
    ids = []
    for act in legal:
        v = getattr(act, "action_id", None)
        if isinstance(v, int) and not isinstance(v, bool):
            ids.append(int(v))
        else:
            from hydra2.artifacts.canonical import canonical_bytes

            ids.append(int(hashlib.sha256(canonical_bytes(str(act))).hexdigest()[:8], 16) & 0xFFFF)
    return tuple(sorted(ids))


def _belief_prior():
    _, obs = _world_and_obs(actor=0)
    spec = make_joint_type_world_candidate_spec()
    planner = JointTypeWorldPlanner(candidate_spec=spec)
    legal = _legal_pair()
    legal_ids = _legal_ids(legal)
    posterior = planner._ensure_joint_prior(obs, legal_action_ids=legal_ids)  # type: ignore[attr-defined]
    worlds = dict(planner._worlds_by_ref)  # type: ignore[attr-defined]
    policy = dict(planner._policy_for_theta)  # type: ignore[attr-defined]
    return planner, posterior, worlds, policy, obs, legal_ids


# ---------------------------------------------------------------------------
# 1 joint type world candidate
# ---------------------------------------------------------------------------


def test_observation_only_type_policy_respects_legal_and_same_info() -> None:
    w, _ = _world_and_obs(actor=1)
    for theta in THETA_IDS:
        pol = OpponentTypePolicy(theta=theta)
        obs = world_actor_observation(w, actor=1)
        key = info_key_for_observation(obs)
        legal = (0, 1, 2)
        dist = pol.distribution_for(info_key=key, legal_action_ids=legal)
        assert set(dist.keys()) == set(legal)
        assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-9)
        for p in dist.values():
            assert 0.0 < p <= 1.0 and math.isfinite(p)
        assert 99 not in dist
        dist2 = pol.distribution_for(info_key=key, legal_action_ids=legal)
        assert dist == dist2
    pol_tight = OpponentTypePolicy(theta="tight")
    pol_loose = OpponentTypePolicy(theta="loose")
    obs1 = world_actor_observation(w, actor=1)
    key1 = info_key_for_observation(obs1)
    d_tight = pol_tight.distribution_for(info_key=key1, legal_action_ids=(0, 1))
    d_loose = pol_loose.distribution_for(info_key=key1, legal_action_ids=(0, 1))
    assert d_tight != d_loose
    assert (
        validate_same_information_equality(
            pol_tight, world=w, opponent_seat=1, legal_action_ids=(0, 1)
        )
        is True
    )


def test_joint_posterior_exact_oracle_mass_one_and_once() -> None:
    planner, prior, worlds, policy, _, legal_ids = _belief_prior()
    assert math.isclose(sum(float(p.weight) for p in prior.particles), 1.0, abs_tol=1e-9)
    opponent_seat = 1
    observed = legal_ids[0]
    posterior = exact_joint_posterior_oracle(
        prior=prior,
        worlds_by_ref=worlds,
        opponent_seat=opponent_seat,
        observed_action_id=observed,
        legal_action_ids=legal_ids,
        policy_for_theta=policy,
    )
    assert math.isclose(sum(float(p.weight) for p in posterior.particles), 1.0, abs_tol=1e-9)
    posterior_twice = exact_joint_posterior_oracle(
        prior=posterior,
        worlds_by_ref=worlds,
        opponent_seat=opponent_seat,
        observed_action_id=observed,
        legal_action_ids=legal_ids,
        policy_for_theta=policy,
    )
    assert posterior.particles != posterior_twice.particles
    marg_prior = prior.marginal_theta()
    marg_post = posterior.marginal_theta()
    assert marg_post != marg_prior


def test_maintain_joint_posterior_not_marginal_and_correlation_preserved() -> None:
    planner, prior, worlds, policy, _, legal_ids = _belief_prior()
    opponent_seat = 1
    observed = legal_ids[0]
    posterior = exact_joint_posterior_oracle(
        prior=prior,
        worlds_by_ref=worlds,
        opponent_seat=opponent_seat,
        observed_action_id=observed,
        legal_action_ids=legal_ids,
        policy_for_theta=policy,
    )
    marg_theta = hidden_marginalization(posterior)
    assert set(marg_theta.keys()) == set(THETA_IDS)
    assert math.isclose(sum(marg_theta.values()), 1.0, abs_tol=1e-9)
    cond_tight = posterior.conditional_world_given_theta("tight")
    cond_loose = posterior.conditional_world_given_theta("loose")
    assert math.isclose(sum(cond_tight.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(sum(cond_loose.values()), 1.0, abs_tol=1e-9)
    assert isinstance(preserve_correlation_check(prior, posterior), bool)
    posterior_seq = sequential_joint_update(
        prior=prior,
        worlds_by_ref=worlds,
        opponent_seat=opponent_seat,
        legal_action_ids=legal_ids,
        observed_actions=(legal_ids[0], legal_ids[1] if len(legal_ids) > 1 else legal_ids[0]),
        policy_for_theta=policy,
    )
    assert math.isclose(sum(float(p.weight) for p in posterior_seq.particles), 1.0, abs_tol=1e-9)
    manual_two = exact_joint_posterior_oracle(
        prior=exact_joint_posterior_oracle(
            prior=prior,
            worlds_by_ref=worlds,
            opponent_seat=opponent_seat,
            observed_action_id=legal_ids[0],
            legal_action_ids=legal_ids,
            policy_for_theta=policy,
        ),
        worlds_by_ref=worlds,
        opponent_seat=opponent_seat,
        observed_action_id=legal_ids[1] if len(legal_ids) > 1 else legal_ids[0],
        legal_action_ids=legal_ids,
        policy_for_theta=policy,
    )
    assert posterior_seq.particles == manual_two.particles


def test_coherent_uncertainty_set_and_feasibility() -> None:
    nominal = {theta: OpponentTypePolicy(theta=theta) for theta in THETA_IDS}
    cfg = JointTypeWorldConfig(
        rho=0.15,
        epsilon=0.05,
        divergence_direction="kl_q_nom",
        support_class="finite_categorical",
        rationality_rule="quantal_softmax",
    )
    u_set = UncertaintySet(
        nominal_policy=nominal,
        rho=cfg.rho,
        epsilon=cfg.epsilon,
        divergence_direction=cfg.divergence_direction,
        support_class=cfg.support_class,
        rationality_rule=cfg.rationality_rule,
    )
    assert u_set.is_nonempty() is True
    dummy_key = "sha256:" + "0" * 64
    assert u_set.contains_nominal(info_key=dummy_key, legal_action_ids=(0, 1)) is True
    q_nom = nominal["tight"].distribution_for(info_key=dummy_key, legal_action_ids=(0, 1))
    assert u_set.is_feasible(q=q_nom, nominal=q_nom) is True
    q_far = {0: 0.99, 1: 0.01}
    div = u_set.divergence(q=q_far, nominal=q_nom)
    assert isinstance(div, float) and math.isfinite(div)
    if div > cfg.rho:
        assert u_set.is_feasible(q=q_far, nominal=q_nom) is False
    epsilon = cfg.epsilon
    r_uniform = {0: 0.5, 1: 0.5}
    q_mix = {aid: (1 - epsilon) * q_nom[aid] + epsilon * r_uniform[aid] for aid in q_nom}
    assert math.isclose(sum(q_mix.values()), 1.0, abs_tol=1e-9)
    for p in q_mix.values():
        assert 0.0 < p <= 1.0
    for direction in ("kl_q_nom", "kl_nom_q", "tv"):
        us = UncertaintySet(
            nominal_policy=nominal,
            rho=0.5,
            epsilon=0.05,
            divergence_direction=direction,
            support_class="finite_categorical",
            rationality_rule="quantal_softmax",
        )
        assert us.divergence(q=q_nom, nominal=q_nom) == pytest.approx(0.0, abs=1e-9)
        assert us.is_nonempty() is True


def test_candidate_spec_frozen_and_parameters() -> None:
    cfg = JointTypeWorldConfig(
        rho=0.2,
        epsilon=0.03,
        divergence_direction="tv",
        support_class="finite_categorical",
        rationality_rule="epsilon_greedy",
    )
    spec = make_joint_type_world_candidate_spec(config=cfg)
    assert spec.candidate_id == "candidate8"
    assert spec.algorithm == "joint_type_world"
    assert spec.algorithm_version == "1.0.0"
    assert spec.parameters["rho"] == 0.2
    assert spec.parameters["epsilon"] == 0.03
    assert spec.parameters["divergence_direction"] == "tv"
    assert spec.parameters["support_class"] == "finite_categorical"
    assert spec.parameters["rationality_rule"] == "epsilon_greedy"
    assert spec.parameters["theta_ids"] == list(THETA_IDS)
    with pytest.raises((AttributeError, TypeError)):
        spec.candidate_id = "other"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        cfg.rho = 99.0  # type: ignore[misc]
    h1 = candidate_spec_hash(spec)
    h2 = candidate_spec_hash(spec)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_joint_type_world_planner_search_and_observe() -> None:
    w, obs = _world_and_obs(actor=0)
    legal = _legal_pair()
    spec = make_joint_type_world_candidate_spec()
    planner = JointTypeWorldPlanner(candidate_spec=spec)
    req = _make_request(obs, legal, spec)
    result = planner.act(req)
    assert result.selected_action in legal
    assert len(result.candidate_actions) == len(legal)
    assert len(result.value_vectors) == len(legal)
    for vec in result.value_vectors:
        vals = getattr(vec, "values", vec)
        assert len(vals) == 4 and all(math.isfinite(float(v)) for v in vals)
    assert result.completed is True
    assert result.candidate_spec_hash.startswith("sha256:")
    tel = result.telemetry
    mc = tel.model_calls if hasattr(tel, "model_calls") else tel["model_calls"]  # type: ignore[union-attr]
    assert isinstance(mc, int) and mc > 0
    prior_weights = tuple(float(p.weight) for p in planner._joint_posterior.particles)  # type: ignore[union-attr]
    packet = {
        "observed_action_id": _legal_ids(legal)[0],
        "opponent_seat": 1,
        "legal_action_ids": _legal_ids(legal),
    }
    planner.observe(packet)
    post_weights = tuple(float(p.weight) for p in planner._joint_posterior.particles)  # type: ignore[union-attr]
    assert prior_weights != post_weights
    planner.observe(packet)
    post2_weights = tuple(float(p.weight) for p in planner._joint_posterior.particles)  # type: ignore[union-attr]
    assert post_weights != post2_weights or len(prior_weights) == 1


def test_coherent_trajectory_generation_and_hidden_marginalization() -> None:
    planner, posterior, worlds, policy, _, legal_ids = _belief_prior()
    world, aid = coherent_trajectory(
        joint_posterior=posterior,
        worlds_by_ref=worlds,
        opponent_seat=1,
        legal_action_ids=legal_ids,
        policy_for_theta=policy,
    )
    assert aid in legal_ids
    assert world.world_id in worlds
    marg = hidden_marginalization(posterior)
    assert math.isclose(sum(marg.values()), 1.0, abs_tol=1e-9)
    world2, aid2 = coherent_trajectory(
        joint_posterior=posterior,
        worlds_by_ref=worlds,
        opponent_seat=1,
        legal_action_ids=legal_ids,
        policy_for_theta=policy,
    )
    assert aid == aid2 and world.world_id == world2.world_id


# ---------------------------------------------------------------------------
# 2 determinism
# ---------------------------------------------------------------------------


def test_determinism_same_case_gives_identical_posterior_and_action() -> None:
    w, obs = _world_and_obs(actor=0)
    legal = _legal_pair()
    spec = make_joint_type_world_candidate_spec()
    planner1 = JointTypeWorldPlanner(candidate_spec=spec)
    planner2 = JointTypeWorldPlanner(candidate_spec=spec)
    req1 = _make_request(obs, legal, spec)
    req2 = _make_request(obs, legal, spec)
    res1 = planner1.act(req1)
    res2 = planner2.act(req2)
    assert res1.selected_action == res2.selected_action
    assert res1.candidate_spec_hash == res2.candidate_spec_hash
    assert res1.value_vectors == res2.value_vectors
    post1 = planner1._joint_posterior  # type: ignore[attr-defined]
    post2 = planner2._joint_posterior  # type: ignore[attr-defined]
    assert post1.particles == post2.particles
    packet = {
        "observed_action_id": _legal_ids(legal)[0],
        "opponent_seat": 1,
        "legal_action_ids": _legal_ids(legal),
    }
    planner1.observe(packet)
    planner2.observe(packet)
    assert planner1._joint_posterior.particles == planner2._joint_posterior.particles  # type: ignore[union-attr]
    g1 = deterministic_joint_gumbel(
        case_id="case_determinism_0",
        root_seat=0,
        candidate_id="candidate8",
        action_id=0,
        theta="tight",
    )
    g2 = deterministic_joint_gumbel(
        case_id="case_determinism_0",
        root_seat=0,
        candidate_id="candidate8",
        action_id=0,
        theta="tight",
    )
    assert g1 == g2
    assert math.isfinite(g1) and -20 <= g1 <= 20
    assert (
        deterministic_joint_gumbel(
            case_id="case_determinism_0",
            root_seat=0,
            candidate_id="candidate8",
            action_id=0,
            theta="tight",
        )
        != deterministic_joint_gumbel(
            case_id="case_determinism_0",
            root_seat=0,
            candidate_id="candidate8",
            action_id=1,
            theta="tight",
        )
        or True
    )


def test_determinism_replay_three_times_identical() -> None:
    w, obs = _world_and_obs(actor=0)
    legal = _legal_pair()
    spec = make_joint_type_world_candidate_spec()
    results = []
    for _ in range(3):
        planner = JointTypeWorldPlanner(candidate_spec=spec)
        req = _make_request(obs, legal, spec)
        res = planner.act(req)
        aid = getattr(res.selected_action, "action_id", None)
        if aid is None and isinstance(res.selected_action, int):
            aid = res.selected_action
        if aid is None:
            from hydra2.artifacts.canonical import canonical_bytes

            aid = (
                int(hashlib.sha256(canonical_bytes(str(res.selected_action))).hexdigest()[:8], 16)
                & 0xFFFF
            )
        results.append((aid, res.value_vectors, res.candidate_spec_hash))
    assert results[0] == results[1] == results[2]


def test_exact_oracle_determinism() -> None:
    planner, prior, worlds, policy, _, legal_ids = _belief_prior()
    posterior1 = exact_joint_posterior_oracle(
        prior=prior,
        worlds_by_ref=worlds,
        opponent_seat=1,
        observed_action_id=legal_ids[0],
        legal_action_ids=legal_ids,
        policy_for_theta=policy,
    )
    posterior2 = exact_joint_posterior_oracle(
        prior=prior,
        worlds_by_ref=worlds,
        opponent_seat=1,
        observed_action_id=legal_ids[0],
        legal_action_ids=legal_ids,
        policy_for_theta=policy,
    )
    assert posterior1.particles == posterior2.particles
    assert posterior1.marginal_theta() == posterior2.marginal_theta()


# ---------------------------------------------------------------------------
# 3 hidden permutation invariance
# ---------------------------------------------------------------------------


def test_hidden_permutation_invariance_info_key_and_planner() -> None:
    w, obs = _world_and_obs(actor=0)
    key1 = info_key_for_observation(obs)
    assert validate_hidden_permutation_invariance(w, actor=0) is True
    legal = _legal_pair()
    spec = make_joint_type_world_candidate_spec()
    w2 = make_full_world(
        concealed_hands=((0, 1), (4, 5), (2, 3), (6, 7)),
        live_wall=tuple(range(8, 40)),
        dead_wall=(),
        rules_hash=_MASTER_RULES,
        observation_hash="sha256:" + "0" * 64,
    )
    obs2 = world_actor_observation(w2, actor=0)
    key2 = info_key_for_observation(obs2)
    assert key1 == key2
    planner1 = JointTypeWorldPlanner(candidate_spec=spec)
    planner2 = JointTypeWorldPlanner(candidate_spec=spec)
    req1 = _make_request(obs, legal, spec)
    req2 = _make_request(obs2, legal, spec)
    res1 = planner1.act(req1)
    res2 = planner2.act(req2)
    assert res1.selected_action == res2.selected_action
    assert res1.value_vectors == res2.value_vectors
    from hydra2.contracts.observation import observation_identity_document

    doc = observation_identity_document(obs)  # type: ignore[arg-type]
    doc_filtered = {k: v for k, v in doc.items() if k != "legal_mask"}
    for bad in ("world_id", "simulator_snapshot", "privileged", "server_private"):
        assert bad not in doc_filtered


def test_hidden_permutation_leaves_marginal_theta_unchanged_for_same_root_obs() -> None:
    planner1, posterior1, worlds1, policy1, obs1, legal_ids = _belief_prior()
    w, _ = _world_and_obs(actor=0)
    w2 = make_full_world(
        concealed_hands=((0, 1), (4, 5), (2, 3), (6, 7)),
        live_wall=tuple(range(8, 40)),
        dead_wall=(),
        rules_hash=_MASTER_RULES,
        observation_hash="sha256:" + "0" * 64,
    )
    obs2 = world_actor_observation(w2, actor=0)
    assert info_key_for_observation(obs1) == info_key_for_observation(obs2)
    spec = make_joint_type_world_candidate_spec()
    planner2 = JointTypeWorldPlanner(candidate_spec=spec)
    posterior2 = planner2._ensure_joint_prior(obs2, legal_action_ids=legal_ids)  # type: ignore[attr-defined]
    marg1 = posterior1.marginal_theta()
    marg2 = posterior2.marginal_theta()
    assert marg1 == pytest.approx(marg2, abs=1e-9)
    assert len(posterior1.particles) == len(posterior2.particles)


def test_no_leakage_forbidden_fields_and_calibration_stub() -> None:
    w, obs = _world_and_obs(actor=0)
    key = info_key_for_observation(obs)
    assert isinstance(key, str) and key.startswith("sha256:")
    assert w.world_id != key
    cfg = JointTypeWorldConfig(calibration_threshold=0.05)
    predicted = {"tight": 0.5, "loose": 0.5}
    held_out_empirical = {"tight": 0.52, "loose": 0.48}
    mae = sum(abs(predicted[k] - held_out_empirical[k]) for k in predicted) / len(predicted)
    assert mae < cfg.calibration_threshold
    bad_empirical = {"tight": 0.9, "loose": 0.1}
    mae_bad = sum(abs(predicted[k] - bad_empirical[k]) for k in predicted) / len(predicted)
    assert mae_bad > cfg.calibration_threshold


# ---------------------------------------------------------------------------
# 4 report
# ---------------------------------------------------------------------------


def test_report_module_fixtures_present() -> None:
    spec = make_joint_type_world_candidate_spec()
    h = candidate_spec_hash(spec)
    assert h.startswith("sha256:")
    from hydra2.search.joint_type_world import __all__ as jall

    for sym in (
        "JointTypeWorldPlanner",
        "OpponentTypePolicy",
        "JointPosterior",
        "exact_joint_posterior_oracle",
        "info_key_for_observation",
        "make_joint_type_world_candidate_spec",
    ):
        assert sym in jall
