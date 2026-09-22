"""ISMCTS descent-driver parity — frozen oracle pins vs the Rust driver.

The Rust-batch search driver must reproduce the retired Python oracle's
outputs bit-identically (parity FIRST per inversion backlog); throughput
SECOND. Goldens below were frozen from the Python oracle on its last green
run: telemetry, selection, arm visits/sums/means/scalars, mean vectors,
tree digest, and the info-key / leaf-vector spot checks. Per-sim selection
sequences and leaf traces from the retired ``_search_once`` replay are not
pinned here (the Rust batch owns descent and exposes aggregates only);
backup correctness is pinned through the arm sums/means and the tree
digest, which ARE the backed values.

Shapes:
- ``shape-a``: 12 simulations x depth 4 (hottest small shape).
- ``shape-b``: 48 simulations x depth 6 (default-config shape).
"""

from __future__ import annotations

import struct
from hashlib import sha256

import pytest

from hydra2.belief.natural import NaturalBelief
from hydra2.belief.world import make_full_world, world_actor_observation
from hydra2.contracts.common import ContractError
from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed
from hydra2.search.ismcts_act import NaturalISMCTSPlanner
from hydra2.search.ismcts_core import (
    NaturalISMCTSConfig,
    attempt_redeterminize,
    is_redeterminization_enabled,
    scalarize_vector,
)
from hydra2.search.ismcts_core import info_key_for_observation as info_key
from hydra2.search.ismcts_core import model_vector_for_world as model_vector
from hydra2.search.ismcts_core import terminal_vector_for_world as terminal_vector

_MASTER = b"ismcts_driver_parity_v1"
_EXPERIMENT = "ismcts-driver-parity"
_SPLIT = "oracle"

_INFO_KEY = "sha256:0962179d2a50471795ff4cff24fbd6566c67f94bf12270c0d66684c860c472d1"
_MODEL_VEC = (0.2, 0.51, 0.06, 0.22)
_TERMINAL_VEC = (0.62, 0.21999999999999997, 0.74, 0.72)
_REDET_MESSAGE = "re-determinization disabled: requires named conditional law q_j(x | I_j, immutable_constraints) with exact b/q proof"

_SHAPE_A_CFG = {"max_simulations": 12, "max_depth": 4, "max_transitions": 64, "max_model_calls": 24}
_SHAPE_B_CFG = {
    "max_simulations": 48,
    "max_depth": 6,
    "max_transitions": 256,
    "max_model_calls": 48,
}


_SHAPE_A_12X4_TELEMETRY = {
    "simulations": 12,
    "transitions": 48,
    "model_calls": 0,
    "max_simulations": 12,
    "max_transitions": 64,
    "max_model_calls": 24,
    "max_depth": 4,
    "uct_c": 1.41421356237,
    "tie_break": "lowest_action_id",
    "candidate_id": "candidate1",
    "resource_view": "calls",
    "root_seat": 0,
    "tree_nodes": 1,
}
_SHAPE_A_12X4_SELECTED = 0
_SHAPE_A_12X4_COMPLETED = True
_SHAPE_A_12X4_BUDGET_EXHAUSTED = False
_SHAPE_A_12X4_ROOT_KEY = "sha256:0962179d2a50471795ff4cff24fbd6566c67f94bf12270c0d66684c860c472d1"
_SHAPE_A_12X4_TREE_KEYS = [
    "sha256:0962179d2a50471795ff4cff24fbd6566c67f94bf12270c0d66684c860c472d1"
]
_SHAPE_A_12X4_NODE_VISITS = 12
_SHAPE_A_12X4_LEGAL = (0, 2)
_SHAPE_A_12X4_ARM0_VISITS = 7
_SHAPE_A_12X4_ARM0_SUM = (5.04, 2.98, 3.42, 3.7800000000000002)
_SHAPE_A_12X4_ARM0_MEAN = (0.72, 0.4257142857142857, 0.48857142857142855, 0.54)
_SHAPE_A_12X4_ARM0_SCALAR = 0.72
_SHAPE_A_12X4_ARM2_VISITS = 5
_SHAPE_A_12X4_ARM2_SUM = (2.9, 1.5999999999999999, 2.66, 3.26)
_SHAPE_A_12X4_ARM2_MEAN = (0.58, 0.31999999999999995, 0.532, 0.6519999999999999)
_SHAPE_A_12X4_ARM2_SCALAR = 0.58
_SHAPE_A_12X4_VALUE_VECTORS = (
    (0.72, 0.4257142857142857, 0.48857142857142855, 0.54),
    (0.58, 0.31999999999999995, 0.532, 0.6519999999999999),
)
_SHAPE_A_12X4_TREE_DIGEST = (
    "sha256:eb88bb0175a25c081dd75140bb1c79b07725562d56a4e810de7e9af5a9ae4b01"
)


_SHAPE_B_48X6_TELEMETRY = {
    "simulations": 48,
    "transitions": 192,
    "model_calls": 0,
    "max_simulations": 48,
    "max_transitions": 256,
    "max_model_calls": 48,
    "max_depth": 6,
    "uct_c": 1.41421356237,
    "tie_break": "lowest_action_id",
    "candidate_id": "candidate1",
    "resource_view": "calls",
    "root_seat": 0,
    "tree_nodes": 1,
}
_SHAPE_B_48X6_SELECTED = 0
_SHAPE_B_48X6_COMPLETED = True
_SHAPE_B_48X6_BUDGET_EXHAUSTED = False
_SHAPE_B_48X6_ROOT_KEY = "sha256:0962179d2a50471795ff4cff24fbd6566c67f94bf12270c0d66684c860c472d1"
_SHAPE_B_48X6_TREE_KEYS = [
    "sha256:0962179d2a50471795ff4cff24fbd6566c67f94bf12270c0d66684c860c472d1"
]
_SHAPE_B_48X6_NODE_VISITS = 48
_SHAPE_B_48X6_LEGAL = (0, 2)
_SHAPE_B_48X6_ARM0_VISITS = 28
_SHAPE_B_48X6_ARM0_SUM = (15.76, 10.920000000000002, 13.320000000000002, 17.26)
_SHAPE_B_48X6_ARM0_MEAN = (
    0.5628571428571428,
    0.39000000000000007,
    0.4757142857142858,
    0.6164285714285714,
)
_SHAPE_B_48X6_ARM0_SCALAR = 0.5628571428571428
_SHAPE_B_48X6_ARM2_VISITS = 20
_SHAPE_B_48X6_ARM2_SUM = (9.42, 7.940000000000001, 7.919999999999999, 11.559999999999999)
_SHAPE_B_48X6_ARM2_MEAN = (0.471, 0.3970000000000001, 0.39599999999999996, 0.578)
_SHAPE_B_48X6_ARM2_SCALAR = 0.471
_SHAPE_B_48X6_VALUE_VECTORS = (
    (0.5628571428571428, 0.39000000000000007, 0.4757142857142858, 0.6164285714285714),
    (0.471, 0.3970000000000001, 0.39599999999999996, 0.578),
)
_SHAPE_B_48X6_TREE_DIGEST = (
    "sha256:5de2daac73d82410dda5226d068e4ff0f753bf71c35d7ea1b577f0d9d6e16c5f"
)


def _stream(*, case_id: str) -> RandomStream:
    key = make_random_stream_key(
        purpose="actor_policy_sample",
        experiment_id=_EXPERIMENT,
        split_id=_SPLIT,
        candidate_id="candidate1",
        case_id=case_id,
        replicate_id=0,
        attempt_id=0,
    )
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


class _DummyAction:
    def __init__(self, aid: int) -> None:
        self.action_id = aid


def _dummy_legal(obs):
    mask = getattr(obs, "legal_mask", (True, False, True))
    ids = [i for i, m in enumerate(mask) if m] or [0, 1]
    return tuple(_DummyAction(i) for i in ids)


def _tree_digest(tree) -> str:
    """Canonical digest over visits + little-endian double bytes of value sums."""
    parts = []
    for key in sorted(tree.keys()):
        node = tree[key]
        acts = ",".join(
            f"{aid}:v={node.action_stats[aid].visits}"
            f":s={''.join(struct.pack('<d', float(x)).hex() for x in node.action_stats[aid].value_sum)}"
            for aid in sorted(node.action_stats.keys())
        )
        legal_s = ",".join(map(str, node.legal_actions))
        parts.append(f"{key}#n={node.visits}#legal={legal_s}#{acts}")
    return "sha256:" + sha256(";".join(parts).encode()).hexdigest()


def _run(tag: str, cfg_kwargs: dict):
    """Run the Rust-batch driver once; return the structured result dict."""
    _, obs = _world_and_obs()
    belief = NaturalBelief()
    epoch = belief.begin(obs)
    legal = _dummy_legal(obs)
    cfg = NaturalISMCTSConfig(
        uct_c=1.41421356237,
        tie_break="lowest_action_id",
        candidate_id="candidate1",
        **cfg_kwargs,
    )
    planner = NaturalISMCTSPlanner(belief=belief, config=cfg)
    return planner.search(
        epoch=epoch, root_observation=obs, legal_actions=legal, rng=_stream(case_id=tag)
    )


def _check_full_tree(
    res,
    *,
    telemetry,
    selected,
    completed,
    budget_exhausted,
    root_key,
    tree_keys,
    node_visits,
    legal,
    arms,
    value_vectors,
    tree_digest,
):
    assert res["telemetry"] == telemetry
    assert res["selected_action_id"] == selected
    assert res["completed"] is completed
    assert res["budget_exhausted"] is budget_exhausted
    assert res["root_key"] == root_key
    assert sorted(res["tree"].keys()) == tree_keys
    assert res["telemetry"]["tree_nodes"] == len(tree_keys)
    node = res["root_node"]
    assert node is not None
    assert node.visits == node_visits
    assert node.legal_actions == legal
    assert sorted(node.action_stats.keys()) == sorted(arms.keys())
    for aid, want in arms.items():
        st = node.action_stats[aid]
        assert st.visits == want[0]
        assert tuple(st.value_sum) == want[1]
        assert node.mean_vector(aid) == want[2]
        assert node.scalar_mean(aid, res["telemetry"]["root_seat"]) == want[3]
    assert res["value_vectors"] == value_vectors
    assert _tree_digest(res["tree"]) == tree_digest


def _check_backup_consistency(res) -> None:
    """Arm sums are internally consistent backed values (means match)."""
    node = res["root_node"]
    total_visits = sum(st.visits for st in node.action_stats.values())
    assert node.visits == total_visits == res["telemetry"]["simulations"]
    for aid, st in node.action_stats.items():
        assert node.mean_vector(aid) is not None
        assert st.visits > 0


def test_info_key_and_leaf_vectors_pinned() -> None:
    world, obs = _world_and_obs()
    assert info_key(obs) == _INFO_KEY
    assert model_vector(world) == _MODEL_VEC
    assert terminal_vector(world) == _TERMINAL_VEC
    assert scalarize_vector((0.5, 0.1, 0.2, 0.3), 0) == 0.5


def test_redeterminize_negative_control() -> None:
    assert is_redeterminization_enabled() is False
    with pytest.raises(ContractError, match="re-determinization disabled"):
        attempt_redeterminize()
    with pytest.raises(ContractError, match="re-determinization disabled"):
        attempt_redeterminize("world", key="info")
    try:
        attempt_redeterminize()
        raise AssertionError("attempt_redeterminize must raise")
    except ContractError as exc:
        assert str(exc) == _REDET_MESSAGE


def test_shape_a_full_tree_pinned() -> None:
    res = _run("shape-a-12x4", _SHAPE_A_CFG)
    _check_full_tree(
        res,
        telemetry=_SHAPE_A_12X4_TELEMETRY,
        selected=_SHAPE_A_12X4_SELECTED,
        completed=_SHAPE_A_12X4_COMPLETED,
        budget_exhausted=_SHAPE_A_12X4_BUDGET_EXHAUSTED,
        root_key=_SHAPE_A_12X4_ROOT_KEY,
        tree_keys=_SHAPE_A_12X4_TREE_KEYS,
        node_visits=_SHAPE_A_12X4_NODE_VISITS,
        legal=_SHAPE_A_12X4_LEGAL,
        arms={
            0: (
                _SHAPE_A_12X4_ARM0_VISITS,
                _SHAPE_A_12X4_ARM0_SUM,
                _SHAPE_A_12X4_ARM0_MEAN,
                _SHAPE_A_12X4_ARM0_SCALAR,
            ),
            2: (
                _SHAPE_A_12X4_ARM2_VISITS,
                _SHAPE_A_12X4_ARM2_SUM,
                _SHAPE_A_12X4_ARM2_MEAN,
                _SHAPE_A_12X4_ARM2_SCALAR,
            ),
        },
        value_vectors=_SHAPE_A_12X4_VALUE_VECTORS,
        tree_digest=_SHAPE_A_12X4_TREE_DIGEST,
    )
    _check_backup_consistency(res)


def test_shape_b_full_tree_pinned() -> None:
    res = _run("shape-b-48x6", _SHAPE_B_CFG)
    _check_full_tree(
        res,
        telemetry=_SHAPE_B_48X6_TELEMETRY,
        selected=_SHAPE_B_48X6_SELECTED,
        completed=_SHAPE_B_48X6_COMPLETED,
        budget_exhausted=_SHAPE_B_48X6_BUDGET_EXHAUSTED,
        root_key=_SHAPE_B_48X6_ROOT_KEY,
        tree_keys=_SHAPE_B_48X6_TREE_KEYS,
        node_visits=_SHAPE_B_48X6_NODE_VISITS,
        legal=_SHAPE_B_48X6_LEGAL,
        arms={
            0: (
                _SHAPE_B_48X6_ARM0_VISITS,
                _SHAPE_B_48X6_ARM0_SUM,
                _SHAPE_B_48X6_ARM0_MEAN,
                _SHAPE_B_48X6_ARM0_SCALAR,
            ),
            2: (
                _SHAPE_B_48X6_ARM2_VISITS,
                _SHAPE_B_48X6_ARM2_SUM,
                _SHAPE_B_48X6_ARM2_MEAN,
                _SHAPE_B_48X6_ARM2_SCALAR,
            ),
        },
        value_vectors=_SHAPE_B_48X6_VALUE_VECTORS,
        tree_digest=_SHAPE_B_48X6_TREE_DIGEST,
    )
    _check_backup_consistency(res)


def test_deterministic_replay_identical() -> None:
    for tag, cfg in (("shape-a-12x4", _SHAPE_A_CFG), ("shape-b-48x6", _SHAPE_B_CFG)):
        r1 = _run(tag, cfg)
        r2 = _run(tag, cfg)
        assert r1["selected_action_id"] == r2["selected_action_id"]
        assert r1["value_vectors"] == r2["value_vectors"]
        assert r1["root_key"] == r2["root_key"]
        assert _tree_digest(r1["tree"]) == _tree_digest(r2["tree"])
        assert {a: s.visits for a, s in r1["root_node"].action_stats.items()} == {
            a: s.visits for a, s in r2["root_node"].action_stats.items()
        }
        assert {a: tuple(s.value_sum) for a, s in r1["root_node"].action_stats.items()} == {
            a: tuple(s.value_sum) for a, s in r2["root_node"].action_stats.items()
        }
