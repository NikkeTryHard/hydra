"""Wave 2 belief parity — oracle pins for kernel / natural / sampled.

Oracle-only (no bridge imports). All RNG via contracts.randomness semantic
streams (counter-based, deterministic). Goldens frozen from the Python oracle;
verticals later prove bridge == oracle through these same tests.
"""

from __future__ import annotations

import math

from hydra2.belief.kernel import NaturalPacketKernel
from hydra2.belief.natural import NaturalBelief, _build_tiny_corpus_for_epoch
from hydra2.belief.sampled_kernel import (
    SAMPLED_KERNEL_MODE,
    SampledKernelConfig,
    enumerate_sampled,
)
from hydra2.belief.world import make_full_world, world_actor_observation
from hydra2.contracts.randomness import RandomStream, make_random_stream_key, semantic_seed

_MASTER = b"wave2_belief_parity_v1"
_EXPERIMENT = "wave2-belief-parity"
_SPLIT = "oracle"

_NATURAL_CASE = "wave2-natural-001"
_SAMPLED_CASE = "wave2-sampled-001"

_CORPUS_ORDER = (
    "sha256:2647ffe382a0c8f56fc80aa005efadbc6f76943eecb1c17ccfc58ad4de3f5512",
    "sha256:525b57d4a59ff9a1a9f626924c90f2c2e44f5fa3e6494a5889fa04fe39d1a07e",
    "sha256:b73b8d7cb99d387467cdbc6c5183d29642dca9ead168a19818370e5fbf68eda7",
    "sha256:e2069aae69ed05888ec60722b619b9d202d13457d9076f56846e78bf2dfcc091",
)

_NATURAL_REFS = (
    "sha256:525b57d4a59ff9a1a9f626924c90f2c2e44f5fa3e6494a5889fa04fe39d1a07e",
    "sha256:525b57d4a59ff9a1a9f626924c90f2c2e44f5fa3e6494a5889fa04fe39d1a07e",
    "sha256:2647ffe382a0c8f56fc80aa005efadbc6f76943eecb1c17ccfc58ad4de3f5512",
    "sha256:525b57d4a59ff9a1a9f626924c90f2c2e44f5fa3e6494a5889fa04fe39d1a07e",
    "sha256:2647ffe382a0c8f56fc80aa005efadbc6f76943eecb1c17ccfc58ad4de3f5512",
    "sha256:525b57d4a59ff9a1a9f626924c90f2c2e44f5fa3e6494a5889fa04fe39d1a07e",
    "sha256:525b57d4a59ff9a1a9f626924c90f2c2e44f5fa3e6494a5889fa04fe39d1a07e",
    "sha256:525b57d4a59ff9a1a9f626924c90f2c2e44f5fa3e6494a5889fa04fe39d1a07e",
)

# Corpus indices (into _CORPUS_ORDER) for _NATURAL_REFS — bridge compares this
# integer vector for the CTR rejection-sampling replication.
_NATURAL_INDICES = (1, 1, 0, 1, 0, 1, 1, 1)


def _stream(
    *, purpose: str, case_id: str, candidate_id: str = "candidate1", **extra: object
) -> RandomStream:
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


def _natural_particles(belief, epoch):
    rng = _stream(
        purpose="belief_natural_sample",
        case_id=_NATURAL_CASE,
        candidate_id="candidate1",
        belief_epoch=0,
        population_id=1,
    )
    return belief.sample_natural(epoch, count=8, rng=rng)


def test_corpus_order_pinned() -> None:
    _, obs = _world_and_obs()
    belief, epoch = _belief_and_epoch(obs)
    corpus = _build_tiny_corpus_for_epoch(epoch, registry=belief._worlds)
    assert tuple(w.world_id for w in corpus) == _CORPUS_ORDER
    # Registry order is world_id-sorted and stable across fresh beliefs.
    _, obs2 = _world_and_obs()
    belief2, epoch2 = _belief_and_epoch(obs2)
    corpus2 = _build_tiny_corpus_for_epoch(epoch2, registry=belief2._worlds)
    assert tuple(w.world_id for w in corpus2) == _CORPUS_ORDER


def test_natural_sample_reproducibility_and_corpus_order() -> None:
    _, obs = _world_and_obs()
    belief, epoch = _belief_and_epoch(obs)
    parts = _natural_particles(belief, epoch)
    assert tuple(p.world_ref for p in parts) == _NATURAL_REFS
    order = {wid: idx for idx, wid in enumerate(_CORPUS_ORDER)}
    assert tuple(order[p.world_ref] for p in parts) == _NATURAL_INDICES
    for p in parts:
        assert p.source == "natural"
        assert p.log_target_density == p.log_proposal_density == -math.log(4)
        assert math.isfinite(p.log_target_density)
    assert [p.parent_id for p in parts] == [
        "525b57d4a59ff9a1",
        "525b57d4a59ff9a1",
        "2647ffe382a0c8f5",
        "525b57d4a59ff9a1",
        "2647ffe382a0c8f5",
        "525b57d4a59ff9a1",
        "525b57d4a59ff9a1",
        "525b57d4a59ff9a1",
    ]
    # Deterministic replay with a fresh semantic stream over the same key.
    replay = _natural_particles(belief, epoch)
    assert tuple(p.world_ref for p in replay) == _NATURAL_REFS
    assert tuple(p.parent_id for p in replay) == tuple(p.parent_id for p in parts)


def test_kernel_enumerate_next_packets_successors_chain() -> None:
    _, obs = _world_and_obs()
    belief, epoch = _belief_and_epoch(obs)
    parts = _natural_particles(belief, epoch)
    p0 = parts[0]
    kernel = NaturalPacketKernel()
    succs = kernel.enumerate_next(epoch=epoch, particle=p0, action=0)
    assert len(succs) == 2
    assert tuple(s.probability for s in succs) == (0.5, 0.5)
    assert abs(sum(s.probability for s in succs) - 1.0) < 1e-9
    assert tuple(s.packet.packet_id for s in succs) == (
        "3e2ad9fd05784143e0dd4318a4c1ff6d24643f5f0104b55132500358c1e19f97",
        "53d2f4c0a08f5f247f0782870f20a6983cb1267a1d18bcd39010b849c5ba0ff3",
    )
    assert tuple(s.successor_world_ref for s in succs) == (
        "world_succ:9309e218cb5043a4",
        "world_succ:c7e5dc552d7fb4bd",
    )
    assert tuple(s.delta_ref for s in succs) == (
        "delta:28653bc8a988f452",
        "delta:acffffb0cc1f741e",
    )
    for s in succs:
        assert s.log_physical_probability == math.log(0.5)
        assert s.log_actor_policy_probability == 0.0
    # Packet chain: empty-before hash, per-successor after/observation hashes.
    assert succs[0].packet.public_state_hash_before == (
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert succs[1].packet.public_state_hash_before == (
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert succs[0].packet.public_state_hash_after == (
        "sha256:fc3741d8b43dbb723c5dac8ad5359e2488f11fa437b8001c2a8280428a0fe6c8"
    )
    assert succs[1].packet.public_state_hash_after == (
        "sha256:65cb7e7da0cc1108b2a9867aaba87e7ef5f1b5b9ca494c8e368fb042fdd903ab"
    )
    assert succs[0].packet.observation_hash_after == (
        "sha256:4081b2393da3cd1f40bc73b67c6daed8b8cfd2330adaa11adb6c4e1f919e8c17"
    )
    assert succs[1].packet.observation_hash_after == (
        "sha256:3fd72b9aa9db1f398f0ee2e6b2b9cedc132afe7d54a16e4333279ba4796a13cf"
    )
    # Packets are pairwise disjoint and replay-identical.
    assert succs[0].packet.packet_id != succs[1].packet.packet_id
    again = kernel.enumerate_next(epoch=epoch, particle=p0, action=0)
    assert tuple(s.packet.packet_id for s in again) == tuple(s.packet.packet_id for s in succs)
    assert tuple(s.successor_world_ref for s in again) == tuple(
        s.successor_world_ref for s in succs
    )


def test_sampled_provenance_mode_and_draws() -> None:
    assert SAMPLED_KERNEL_MODE == "natural_trace_sample_v1"
    _, obs = _world_and_obs()
    belief, epoch = _belief_and_epoch(obs)
    parts = _natural_particles(belief, epoch)
    p0 = parts[0]
    cfg = SampledKernelConfig(samples_per_parent_action=4)
    rng = _stream(
        purpose="kernel_sample",
        case_id=_SAMPLED_CASE,
        candidate_id="candidate1",
        parent_id=p0.parent_id,
        action_id=0,
        belief_epoch=0,
    )
    batch = enumerate_sampled(epoch=epoch, particle=p0, action=0, config=cfg, rng=rng)
    assert len(batch) == 4
    # Frame order is [3e2ad9fd(tile 8), 53d2f4c0(tile 9)]; chosen frame indices.
    assert tuple(s.packet.packet_id for s in batch) == (
        "53d2f4c0a08f5f247f0782870f20a6983cb1267a1d18bcd39010b849c5ba0ff3",
        "53d2f4c0a08f5f247f0782870f20a6983cb1267a1d18bcd39010b849c5ba0ff3",
        "3e2ad9fd05784143e0dd4318a4c1ff6d24643f5f0104b55132500358c1e19f97",
        "3e2ad9fd05784143e0dd4318a4c1ff6d24643f5f0104b55132500358c1e19f97",
    )
    for s in batch:
        assert s.raw_weight == 0.125
        assert s.provenance["mode"] == SAMPLED_KERNEL_MODE
        assert s.provenance["samples_per_parent_action"] == 4
        assert s.provenance["frame_mass"] == 1.0
    assert sum(s.raw_weight for s in batch) == 0.5
    # Deterministic replay with a fresh semantic stream over the same key.
    rng2 = _stream(
        purpose="kernel_sample",
        case_id=_SAMPLED_CASE,
        candidate_id="candidate1",
        parent_id=p0.parent_id,
        action_id=0,
        belief_epoch=0,
    )
    batch2 = enumerate_sampled(epoch=epoch, particle=p0, action=0, config=cfg, rng=rng2)
    assert tuple(s.packet.packet_id for s in batch2) == tuple(s.packet.packet_id for s in batch)
    assert tuple(s.raw_weight for s in batch2) == tuple(s.raw_weight for s in batch)
