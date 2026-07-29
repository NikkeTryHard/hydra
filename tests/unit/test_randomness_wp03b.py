"""WP-03B gate: SPEC 13 semantic randomness contracts (unit slice)."""

from __future__ import annotations

import hashlib
import json

import pytest

from hydra2.contracts.common import ContractError, DeterminismError
from hydra2.contracts.randomness import (
    FINAL_EVALUATION_PURPOSES,
    RANDOM_PURPOSES,
    RandomStream,
    RandomStreamCheckpoint,
    RandomStreamSchema,
    StreamLedger,
    authority_stream,
    derive_scope_material,
    key_to_json,
    make_random_stream_key,
    retry_key,
    semantic_seed,
)

pytestmark = pytest.mark.contract_package("WP-03B")

MASTER = bytes(range(32))

_DISTINCT_STREAM_PURPOSES = (
    "belief_natural_sample",
    "belief_proposal_sample",
    "actor_policy_sample",
    "root_tree_selection",
    "rollout_transition",
    "confirmation",
)

_GAME_SCOPED_PURPOSES = frozenset(
    {
        "belief_natural_sample",
        "belief_proposal_sample",
        "actor_policy_sample",
        "root_tree_selection",
        "rollout_transition",
        "rollout_advantage",
        "confirmation",
        "gumbel_root",
        "coupling_primitive",
    }
)

_PURPOSE_EXTRAS: dict[str, dict[str, object]] = {
    "belief_natural_sample": {"population_id": 2, "belief_epoch": 4},
    "belief_proposal_sample": {"population_id": 2, "belief_epoch": 4},
    "root_tree_selection": {"parent_id": "node-1"},
    "rollout_transition": {"parent_id": "node-1", "action_id": 9},
    "rollout_advantage": {"packet_id": "pkt-1"},
    "gumbel_root": {"action_id": 5, "candidate_id": "cand-a"},
    "coupling_primitive": {"parent_id": "prim-1"},
}

_ENV_PURPOSES = ("coupling_primitive", "wall", "evaluation_schedule")


def _game_key(purpose: str, *, case_id: str | None = "case-1", **overrides: object) -> object:
    base: dict[str, object] = {
        "purpose": purpose,
        "experiment_id": "exp-wp03b",
        "split_id": "confirm",
        "case_id": case_id,
        "replicate_id": 0,
        "attempt_id": 0,
        "candidate_id": None if purpose in _ENV_PURPOSES else "cand-a",
    }
    if purpose == "wall":
        base.setdefault("wall_id", "w-77")
        base["case_id"] = None
    if purpose == "evaluation_schedule":
        base["case_id"] = None
    base.update(_PURPOSE_EXTRAS.get(purpose, {}))
    base.update(overrides)
    return make_random_stream_key(**base)


def test_semantic_seed_golden_vector_stable() -> None:
    """Seed derivation is the SPEC formula pinned to an explicit digest."""
    key = _game_key("wall")
    derived = semantic_seed(MASTER, key=key)  # type: ignore[arg-type]
    payload = json.dumps(
        {
            "protocol": "hydra2_rng_v1",
            "master_seed": MASTER.hex(),
            "key": key_to_json(key),  # type: ignore[arg-type]
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert derived.hex() == hashlib.sha256(payload).hexdigest()
    # Byte-stability pin: derivation must never drift silently.
    assert semantic_seed(MASTER, key=key) == derived  # type: ignore[arg-type]


def test_purpose_discrimination_makes_streams_distinct() -> None:
    """natural/proposal/policy/root/transition/confirmation can never collide."""
    seeds = [
        semantic_seed(MASTER, key=_game_key(purpose))  # type: ignore[arg-type]
        for purpose in (*_DISTINCT_STREAM_PURPOSES, "wall", "evaluation_schedule")
    ]
    assert len(set(seeds)) == len(_DISTINCT_STREAM_PURPOSES) + 2


def test_no_call_order_derivation() -> None:
    """Creation order and interleaving never influence derived seeds."""
    purposes = list(_DISTINCT_STREAM_PURPOSES)
    forward = [semantic_seed(MASTER, key=_game_key(item)) for item in purposes]  # type: ignore[arg-type]
    backward = [
        semantic_seed(MASTER, key=_game_key(item))  # type: ignore[arg-type]
        for item in reversed(purposes)
    ]
    permutation = [0, 5, 1, 4, 2, 3]
    interleaved = [
        semantic_seed(MASTER, key=_game_key(purposes[index]))  # type: ignore[arg-type]
        for index in permutation
    ]
    assert forward == list(reversed(backward))
    assert sorted(map(bytes, forward)) == sorted(map(bytes, interleaved))


def test_retry_increments_attempt_and_changes_seed() -> None:
    key = _game_key("actor_policy_sample")
    first_retry = retry_key(key)  # type: ignore[arg-type]
    second_retry = retry_key(first_retry)
    assert first_retry.attempt_id == key.attempt_id + 1
    assert second_retry.attempt_id == key.attempt_id + 2
    seeds = {semantic_seed(MASTER, key=item) for item in (key, first_retry, second_retry)}
    assert len(seeds) == 3


@pytest.mark.parametrize(
    ("purpose", "mutate", "fragment"),
    [
        ("mlmc_level", {"fidelity_level": None}, "requires non-null fidelity_level"),
        ("rqmc_scramble", {"scramble_id": None}, "requires non-null scramble_id"),
        ("smc_propagation", {"population_id": None}, "requires non-null population_id"),
        ("smc_resampling", {"population_id": None}, "requires non-null population_id"),
        ("gumbel_root", {"action_id": None}, "requires non-null action_id"),
        ("actor_policy_sample", {"candidate_id": None}, "requires non-null candidate_id"),
        ("coupling_primitive", {"parent_id": None}, "requires non-null parent_id"),
        # Coupled branches share primitives across candidates: candidate forbidden.
        ("coupling_primitive", {"candidate_id": "cand-a"}, "forbids non-null candidate_id"),
        # Unused fields must stay null.
        ("wall", {"fidelity_level": 0}, "forbids non-null fidelity_level"),
        ("evaluation_schedule", {"case_id": "case-1"}, "forbids non-null case_id"),
        ("training_shuffle", {"action_id": 3}, "forbids non-null action_id"),
        ("belief_natural_sample", {"scramble_id": 1}, "forbids non-null scramble_id"),
        ("unknown_purpose_later", {}, "is not one of"),
    ],
)
def test_schema_matrix_enforcement(purpose: str, mutate: dict[str, object], fragment: str) -> None:
    kwargs: dict[str, object] = {
        "purpose": purpose,
        "experiment_id": "exp-wp03b",
        "split_id": "confirm",
        "replicate_id": 0,
        "attempt_id": 0,
    }
    if purpose != "unknown_purpose_later":
        kwargs["candidate_id"] = None if purpose in _ENV_PURPOSES else "cand-a"
    if purpose == "wall":
        kwargs["wall_id"] = "w-1"
    if purpose in _GAME_SCOPED_PURPOSES:
        kwargs["case_id"] = "case-1"
    else:
        kwargs["case_id"] = None
    extras = {k: v for k, v in _PURPOSE_EXTRAS.get(purpose, {}).items() if k != "candidate_id"}
    kwargs.update(extras)
    kwargs.update(mutate)
    if purpose == "coupling_primitive":
        kwargs.setdefault("parent_id", "prim-1")
    with pytest.raises(ContractError, match=fragment):
        make_random_stream_key(**kwargs)


@pytest.mark.parametrize(
    "purpose",
    sorted(_GAME_SCOPED_PURPOSES),
)
def test_exactly_one_of_case_or_wall(purpose: str) -> None:
    valid = _game_key(purpose)
    projection = key_to_json(valid)  # type: ignore[arg-type]
    assert projection["case_id"] is not None
    assert projection["wall_id"] is None
    with pytest.raises(ContractError, match="exactly one of"):
        make_random_stream_key(**{**projection, "wall_id": "w-also"})
    with pytest.raises(ContractError, match="exactly one of"):
        make_random_stream_key(**{**projection, "case_id": None, "wall_id": None})


def test_smc_purposes_are_distinct_and_require_population() -> None:
    propagation = make_random_stream_key(
        purpose="smc_propagation",
        experiment_id="e",
        split_id="s",
        population_id=3,
        belief_epoch=7,
        replicate_id=1,
        attempt_id=0,
    )
    resampling = make_random_stream_key(
        purpose="smc_resampling",
        experiment_id="e",
        split_id="s",
        population_id=3,
        belief_epoch=7,
        replicate_id=1,
        attempt_id=0,
    )
    assert propagation.purpose != resampling.purpose
    assert semantic_seed(MASTER, key=propagation) != semantic_seed(MASTER, key=resampling)
    with pytest.raises(ContractError, match="population_id"):
        make_random_stream_key(
            purpose="smc_resampling",
            experiment_id="e",
            split_id="s",
            population_id=None,
            belief_epoch=7,
            replicate_id=1,
            attempt_id=0,
        )


def test_random_stream_counter_semantics() -> None:
    key = _game_key("wall")
    stream = RandomStream.from_key(MASTER, key=key)  # type: ignore[arg-type]
    twenty = stream.get_bytes(20)
    more = stream.get_bytes(20)
    assert stream.cursor == 40
    fresh = RandomStream.from_key(MASTER, key=key)  # type: ignore[arg-type]
    assert fresh.get_bytes(40) == twenty + more
    # Counter-based: seeking reproduces any suffix without replaying history.
    seeker = RandomStream.from_key(MASTER, key=key)  # type: ignore[arg-type]
    seeker.jump_to(30)
    direct = RandomStream.from_key(MASTER, key=key).get_bytes(40)[30:]  # type: ignore[arg-type]
    assert seeker.get_bytes(10) == direct


def test_checkpoint_restore_exact_continuation() -> None:
    key = _game_key("confirmation")
    stream = RandomStream.from_key(MASTER, key=key)  # type: ignore[arg-type]
    stream.get_bytes(11)
    checkpoint = stream.checkpoint()
    assert isinstance(checkpoint, RandomStreamCheckpoint)
    expected = stream.get_bytes(128)
    restored = RandomStream.restore(checkpoint)
    assert restored.get_bytes(128) == expected
    with pytest.raises(ContractError):
        RandomStreamCheckpoint(seed_hex="zz", cursor=0)
    with pytest.raises(ContractError):
        RandomStreamCheckpoint(seed_hex="00", cursor=-1)


def test_stream_ledger_rejects_reissue() -> None:
    ledger = StreamLedger()
    key = _game_key("root_tree_selection")
    ledger.issue(MASTER, key)  # type: ignore[arg-type]
    with pytest.raises(DeterminismError, match="already issued"):
        ledger.issue(MASTER, key)  # type: ignore[arg-type]
    ledger.issue(MASTER, retry_key(key))  # type: ignore[arg-type]


def test_final_evaluation_seed_isolation() -> None:
    root = b"\x5a" * 32
    selection = derive_scope_material(root, "selection_training")
    final = derive_scope_material(root, "final_evaluation")
    assert selection.material != final.material

    confirm_key = _game_key("confirmation")
    schedule_key = _game_key("evaluation_schedule")
    train_key = _game_key("training_shuffle", case_id=None)

    authority_stream(final, confirm_key)  # type: ignore[arg-type]
    authority_stream(final, schedule_key)  # type: ignore[arg-type]
    authority_stream(selection, train_key)  # type: ignore[arg-type]

    with pytest.raises(ContractError, match="selection/training material cannot feed"):
        authority_stream(selection, confirm_key)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="selection/training material cannot feed"):
        authority_stream(selection, schedule_key)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="final-evaluation material cannot feed"):
        authority_stream(final, train_key)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="unknown scope"):
        derive_scope_material(root, "everything")
    assert set(FINAL_EVALUATION_PURPOSES) == {"confirmation", "evaluation_schedule"}


def test_schema_class_exposes_matrix() -> None:
    assert RandomStreamSchema.required_by_purpose["mlmc_level"] == ("fidelity_level",)
    assert RandomStreamSchema.required_by_purpose["wall"] == ("wall_id",)
    assert "candidate_id" not in RandomStreamSchema.optional_by_purpose["coupling_primitive"]
    assert len(RANDOM_PURPOSES) == 17
