"""Wave 2 search parity — oracle pins for Gumbel halving + ISMCTS descent.

Oracle-only (no bridge imports). All RNG via contracts.randomness semantic
streams (counter-based, deterministic). Goldens frozen from the Python oracle;
verticals later prove bridge == oracle through these same tests.
"""

from __future__ import annotations

import math

from hydra2.belief.natural import NaturalBelief
from hydra2.belief.world import make_full_world, world_actor_observation
from hydra2.contracts.action import CanonicalAction
from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed
from hydra2.search.gumbel import (
    GumbelSearchPlanner,
    deterministic_gumbel,
    deterministic_root_gumbels,
    exact_transition,
    make_gumbel_candidate_spec,
)
from hydra2.search.gumbel import (
    model_vector_for_world as gumbel_model_vector,
)
from hydra2.search.gumbel import (
    scalarize_vector as gumbel_scalarize,
)
from hydra2.search.gumbel import (
    terminal_vector_for_world as gumbel_terminal_vector,
)
from hydra2.search.ismcts_core import (
    InformationSetNode,
    NaturalISMCTSConfig,
    _ActionStats,
    _uct_select,
    validate_tree_keys_contain_no_world_id,
)
from hydra2.search.ismcts_core import (
    info_key_for_observation as ismcts_info_key,
)
from hydra2.search.ismcts_core import (
    model_vector_for_world as ismcts_model_vector,
)
from hydra2.search.ismcts_core import (
    scalarize_vector as ismcts_scalarize,
)
from hydra2.search.ismcts_core import (
    terminal_vector_for_world as ismcts_terminal_vector,
)
from hydra2.search.ismcts_natural import NaturalISMCTSPlanner

_MASTER = b"wave2_search_parity_v1"
_EXPERIMENT = "wave2-search-parity"
_SPLIT = "oracle"

_GUMB_CASE = "wave2-gumbel-halving"
_ISMCTS_CASE = "wave2-ismcts-descent"


def _stream(*, purpose: str, candidate_id: str, case_id: str, **extra: object) -> RandomStream:
    base: dict[str, object] = {
        "purpose": purpose,
        "experiment_id": _EXPERIMENT,
        "split_id": _SPLIT,
        "candidate_id": candidate_id,
        "case_id": case_id,
        "replicate_id": 0,
        "attempt_id": 0,
    }
    base.update(extra)
    key = make_random_stream_key(**base)
    return RandomStream(semantic_seed(_MASTER, key=key))


def _world_and_obs():
    world = make_full_world(
        concealed_hands=((0, 1), (2, 3), (4, 5), (6, 7)),
        live_wall=(8, 9, 10, 11),
        dead_wall=(),
        latent_state={"step": 0, "turn": 0},
        rules_hash="sha256:" + "a" * 64,
        observation_hash="sha256:" + "b" * 64,
        simulator_snapshot="snap_test",
    )
    return world, world_actor_observation(world, actor=0)


def _belief_and_epoch(obs):
    belief = NaturalBelief()
    return belief, belief.begin(obs)


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


class _DummyAction:
    def __init__(self, aid: int) -> None:
        self.action_id = aid

    def __repr__(self) -> str:  # pragma: no cover
        return f"A({self.action_id})"


def _dummy_legal(obs):
    mask = getattr(obs, "legal_mask", (True, False, True))
    ids = [i for i, m in enumerate(mask) if m] or [0, 1]
    return tuple(_DummyAction(i) for i in ids)


# ---------------------------------------------------------------------------
# Gumbel sequential halving
# ---------------------------------------------------------------------------


def test_gumbel_halving_final_decision_and_visits() -> None:
    _, obs = _world_and_obs()
    belief, epoch = _belief_and_epoch(obs)
    spec = make_gumbel_candidate_spec(halving_rounds=2, visits_per_round=(4, 2), max_depth=4)
    planner = GumbelSearchPlanner(candidate_spec=spec, belief=belief)
    rng = _stream(purpose="actor_policy_sample", candidate_id="candidate6", case_id=_GUMB_CASE)
    res = planner.search(
        epoch=epoch,
        root_observation=obs,
        legal_actions=_legal_pair(),
        rng=rng,
        case_id=_GUMB_CASE,
    )
    # Action-id mapping: pass -> 346545019, discard(tile 0) -> 169577016.
    assert res["selected_action_id"] == 346545019
    assert res["selected_action"].kind == "pass"
    assert res["survivors"] == (346545019,)
    assert res["gumbels"] == {169577016: 0.5656721255310457, 346545019: 0.7022640836861735}
    visits = {aid: st.visits for aid, st in res["stats"].items()}
    assert visits == {169577016: 4, 346545019: 4}
    means = {aid: st.mean_vector() for aid, st in res["stats"].items()}
    assert means[169577016] == (0.425, 0.42, 0.42999999999999994, 0.46)
    assert means[346545019] == (0.35, 0.72, 0.48500000000000004, 0.6150000000000001)
    assert res["value_vectors"] == (
        (0.425, 0.42, 0.42999999999999994, 0.46),
        (0.35, 0.72, 0.48500000000000004, 0.6150000000000001),
    )
    tel = res["telemetry"]
    assert tel["simulations"] == 8
    assert tel["transitions"] == 32
    assert tel["model_calls"] == 0
    assert tel["halving_rounds"] == 2
    assert tel["visits_per_round"] == (4, 2)
    assert tel["root_seat"] == 0
    assert res["completed"] is True


def test_gumbel_deterministic_replay() -> None:
    _, obs = _world_and_obs()
    belief, epoch = _belief_and_epoch(obs)
    spec = make_gumbel_candidate_spec(halving_rounds=2, visits_per_round=(4, 2), max_depth=4)
    legal = _legal_pair()
    planner1 = GumbelSearchPlanner(candidate_spec=spec, belief=belief)
    rng1 = _stream(purpose="actor_policy_sample", candidate_id="candidate6", case_id=_GUMB_CASE)
    r1 = planner1.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=rng1, case_id=_GUMB_CASE
    )
    planner2 = GumbelSearchPlanner(candidate_spec=spec, belief=belief)
    rng2 = _stream(purpose="actor_policy_sample", candidate_id="candidate6", case_id=_GUMB_CASE)
    r2 = planner2.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=rng2, case_id=_GUMB_CASE
    )
    assert r1["selected_action_id"] == r2["selected_action_id"] == 346545019
    assert r1["gumbels"] == r2["gumbels"]
    assert r1["value_vectors"] == r2["value_vectors"]
    assert {a: s.visits for a, s in r1["stats"].items()} == {
        a: s.visits for a, s in r2["stats"].items()
    }
    assert r1["survivors"] == r2["survivors"] == (346545019,)


def test_gumbel_core_deterministic_roots_and_transition() -> None:
    world, _ = _world_and_obs()
    g0 = deterministic_gumbel(
        case_id=_GUMB_CASE, root_seat=0, candidate_id="candidate6", action_id=169577016
    )
    g1 = deterministic_gumbel(
        case_id=_GUMB_CASE, root_seat=0, candidate_id="candidate6", action_id=346545019
    )
    assert g0 == 0.5656721255310457
    assert g1 == 0.7022640836861735
    batch = deterministic_root_gumbels(
        case_id=_GUMB_CASE,
        root_seat=0,
        candidate_id="candidate6",
        legal_action_ids=(169577016, 346545019),
    )
    assert batch == {169577016: g0, 346545019: g1}
    # Halving cut shape: g + q decides the survivor (pass wins).
    assert (g1 + 0.35) > (g0 + 0.425)
    assert gumbel_scalarize((0.35, 0.72, 0.485, 0.615), 0) == 0.35
    assert gumbel_model_vector(world) == (0.02, 0.06, 0.61, 0.23)
    assert gumbel_terminal_vector(world) == (0.62, 0.21999999999999997, 0.74, 0.72)
    t1 = exact_transition(world, 0, 0)
    t2 = exact_transition(world, 0, 0)
    assert (
        t1.world_id
        == t2.world_id
        == "sha256:ee137ee54d3463555387294a8ef26a9f72cb139287fa4b6450daa2090bc0c142"
    )


# ---------------------------------------------------------------------------
# ISMCTS descent
# ---------------------------------------------------------------------------


def test_ismcts_descent_within_budget() -> None:
    _, obs = _world_and_obs()
    belief, epoch = _belief_and_epoch(obs)
    cfg = NaturalISMCTSConfig(
        max_simulations=12,
        max_depth=4,
        max_transitions=64,
        max_model_calls=24,
        uct_c=1.41421356237,
        tie_break="lowest_action_id",
        candidate_id="candidate1",
    )
    planner = NaturalISMCTSPlanner(belief=belief, config=cfg)
    legal = _dummy_legal(obs)
    assert [a.action_id for a in legal] == [0, 2]
    rng = _stream(purpose="actor_policy_sample", candidate_id="candidate1", case_id=_ISMCTS_CASE)
    res = planner.search(epoch=epoch, root_observation=obs, legal_actions=legal, rng=rng)
    tel = res["telemetry"]
    assert tel["simulations"] == 12
    assert tel["transitions"] == 48
    assert tel["model_calls"] == 0
    assert tel["transitions"] <= 64
    assert tel["model_calls"] <= 24
    assert tel["tree_nodes"] == 1
    assert res["selected_action_id"] == 0
    assert (
        res["root_key"] == "sha256:0962179d2a50471795ff4cff24fbd6566c67f94bf12270c0d66684c860c472d1"
    )
    node = res["root_node"]
    assert node is not None
    assert node.legal_actions == (0, 2)
    assert node.visits == 12
    assert node.action_stats[0].visits == 7
    assert node.action_stats[2].visits == 5
    assert node.mean_vector(0) == (
        0.5542857142857143,
        0.4942857142857143,
        0.36285714285714293,
        0.5428571428571429,
    )
    assert node.mean_vector(2) == (0.43200000000000005, 0.4640000000000001, 0.188, 0.576)
    assert res["value_vectors"] == (
        (0.5542857142857143, 0.4942857142857143, 0.36285714285714293, 0.5428571428571429),
        (0.43200000000000005, 0.4640000000000001, 0.188, 0.576),
    )
    assert res["completed"] is True
    assert res["budget_exhausted"] is False


def test_ismcts_deterministic_replay() -> None:
    _, obs = _world_and_obs()
    belief, epoch = _belief_and_epoch(obs)
    cfg = NaturalISMCTSConfig(
        max_simulations=12,
        max_depth=4,
        max_transitions=64,
        max_model_calls=24,
        uct_c=1.41421356237,
        tie_break="lowest_action_id",
        candidate_id="candidate1",
    )
    legal = _dummy_legal(obs)
    planner1 = NaturalISMCTSPlanner(belief=belief, config=cfg)
    rng1 = _stream(purpose="actor_policy_sample", candidate_id="candidate1", case_id=_ISMCTS_CASE)
    r1 = planner1.search(epoch=epoch, root_observation=obs, legal_actions=legal, rng=rng1)
    planner2 = NaturalISMCTSPlanner(belief=belief, config=cfg)
    rng2 = _stream(purpose="actor_policy_sample", candidate_id="candidate1", case_id=_ISMCTS_CASE)
    r2 = planner2.search(epoch=epoch, root_observation=obs, legal_actions=legal, rng=rng2)
    assert r1["selected_action_id"] == r2["selected_action_id"] == 0
    assert r1["value_vectors"] == r2["value_vectors"]
    assert r1["root_key"] == r2["root_key"]
    assert {a: s.visits for a, s in r1["root_node"].action_stats.items()} == {
        a: s.visits for a, s in r2["root_node"].action_stats.items()
    }


def test_ismcts_core_info_key_firewall_and_vectors() -> None:
    _, obs = _world_and_obs()
    key = ismcts_info_key(obs)
    assert key == "sha256:0962179d2a50471795ff4cff24fbd6566c67f94bf12270c0d66684c860c472d1"
    assert validate_tree_keys_contain_no_world_id([key]) is True
    assert ismcts_scalarize((0.5, 0.1, 0.2, 0.3), 0) == 0.5
    world, _ = _world_and_obs()
    assert ismcts_model_vector(world) == (0.2, 0.51, 0.06, 0.22)
    assert ismcts_terminal_vector(world) == (0.62, 0.21999999999999997, 0.74, 0.72)
    # UCT step shape: empty node takes lowest unvisited; visited node exploits.
    node = InformationSetNode(key="k", legal_actions=(0, 2))
    assert _uct_select(node, (0, 2), 0, 1.41421356237, "lowest_action_id") == 0
    node.action_stats[0] = _ActionStats(visits=2, value_sum=(1.0, 0.0, 0.0, 0.0))
    node.action_stats[2] = _ActionStats(visits=1, value_sum=(0.1, 0.0, 0.0, 0.0))
    node.visits = 3
    assert _uct_select(node, (0, 2), 0, 1.41421356237, "lowest_action_id") == 2
    assert math.isfinite(node.mean_vector(0)[0])
