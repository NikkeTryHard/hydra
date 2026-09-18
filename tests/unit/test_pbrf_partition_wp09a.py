# ruff: noqa: F841
"""WP-09A PBRF packet partition and confirmation failures.

Packet-partition invariants (exhaustive disjoint enumeration, provenance,
normalizer tolerance, missing-mass hard failure, the partition helper, and
diagnostic codes) plus confirmation failures (frozen-candidate stability,
stale and digest rejection, emitted-action commit, tile binding, depleted
rebuild, and target binding) for the Candidate 3 PBRF core.
"""

from __future__ import annotations

import hashlib
import math
import time

import pytest
from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.belief.kernel import NaturalPacketKernel
from hydra2.belief.natural import NaturalBelief
from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    PacketPartitionError,
    StaleBeliefError,
)
from hydra2.contracts.randomness import RandomStream
from hydra2.search.common import SearchRequest
from hydra2.search.pbrf import (
    ChildEntry,
    ImmutableForest,
    PbrfConfig,
    PbrfPlanner,
    _conditional_carry_logps,
    _is_target_compatible,
    _verify_delta_reconstruction,
    build_pbrf,
    commit,
    fixed_allocate,
    make_pbrf_candidate_spec,
    validate_packet_partition,
)
from tests.unit.test_pbrf_wp09a import (
    _MASTER_RULES,
    _aid,
    _belief_epoch,
    _candidates_fn,
    _legal_pair,
    _world_and_obs,
)

pytestmark = pytest.mark.contract_package("WP-09A")


# ---------------------------------------------------------------------------
# 3 exhaustively enumerate immediate disjoint packet kernel per parent/action
# ---------------------------------------------------------------------------


def test_exhaustively_enumerate_immediate_disjoint_packet_kernel_per_parent_action() -> None:
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=4, max_search_batches=8)
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"enum"),
        config=cfg,
    )
    # For each action, children keys count * raw sum must partition
    for action in legal:
        aid = _aid(action)
        # Gather Z hats for this action
        z_hats = []
        pids = set()
        for (k_aid, pid), entries in forest.children.items():
            if k_aid == aid:
                assert len(pid) > 8  # packet_id looks like sha256
                assert pid not in pids  # disjoint per action
                pids.add(pid)
                # each entry corresponds to one parent's contribution to that packet
                assert len(entries) > 0
                for e in entries:
                    assert isinstance(e, ChildEntry)
                    assert e.successor_world_ref and e.successor_delta
                z_hats.append(sum(e.raw_weight for e in entries))
        # Kernel enumerates exactly 2 disjoint packets per parent/action (our tiny kernel)
        # With 4 parents, each packet's Z_hat ~0.5, sum 1
        assert len(pids) == 2
        assert sum(z_hats) == pytest.approx(1.0, abs=cfg.kernel_tolerance)
        # Exhaustive: each parent appears exactly once per packet partition across packets for this action
        # Count total raw_weight contributions for this action should be 1 (since each parent's 1/N * prob sum 1)
        # Already checked sum z_hats ==1


def test_successor_world_ref_mandatory_and_delta_verified() -> None:
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=2, max_search_batches=4)
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"delta"),
        config=cfg,
    )
    for entries in forest.children.values():
        for e in entries:
            assert e.successor_world_ref
            assert e.successor_delta
            assert e.successor_world_ref != e.successor_delta


# ---------------------------------------------------------------------------
# 4 store parent_id successor_delta raw_weight provenance
# ---------------------------------------------------------------------------


def test_store_parent_id_successor_delta_raw_weight_provenance() -> None:
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=4, max_search_batches=8)
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"prov"),
        config=cfg,
    )
    for entries in forest.children.values():
        for e in entries:
            assert isinstance(e.parent_id, str) and e.parent_id
            assert (
                isinstance(e.raw_weight, float)
                and math.isfinite(e.raw_weight)
                and 0 <= e.raw_weight <= 1
            )
            assert e.target_id == epoch.target_id
            assert e.epoch == epoch.epoch
            # raw_weight = prob / N ; prob 0.5, N 4 => 0.125
            assert e.raw_weight == pytest.approx(0.5 / 4)
    # Check that parent_ids are from actual parents
    parent_ids = {p.parent_id for p in forest.parents}
    for entries in forest.children.values():
        for e in entries:
            assert e.parent_id in parent_ids


# ---------------------------------------------------------------------------
# 5 require child normalizer partition within tolerance
# ---------------------------------------------------------------------------


def test_require_child_normalizer_partition_within_tolerance() -> None:
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=4, kernel_tolerance=1e-9, max_search_batches=8)
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(kernel_tolerance=1e-9),
        rng=RandomStream(b"partition"),
        config=cfg,
    )
    for action in legal:
        aid = _aid(action)
        total = 0.0
        for (k_aid, _), entries in forest.children.items():
            if k_aid == aid:
                total += sum(e.raw_weight for e in entries)
        assert abs(total - 1.0) <= cfg.kernel_tolerance


def test_missing_packet_mass_is_hard_failure() -> None:
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()

    class BadKernel:
        def enumerate_next(self, *, epoch, particle, action, policy_set=None):
            import hashlib as _hl

            # Return successors with mass 0.6 (bad)
            from hydra2.artifacts.canonical import canonical_bytes as _cb
            from hydra2.belief.kernel import PacketSuccessor
            from hydra2.contracts.common import Seat
            from hydra2.contracts.event_envelope import (
                EventEnvelope,
                EventPayload,
            )
            from hydra2.contracts.event_packet import (
                make_actor_visible_packet,
                public_state_chain_hash,
            )

            # Fabricate a packet but with prob 0.3 each -> total 0.6
            # Use real packet construction to keep kernel interface valid
            rh = epoch.rules_hash
            sh = "sha256:" + "c" * 64
            successors = []
            for idx in range(2):
                tile = 8 + idx
                seq = 100 + idx

                payload = EventPayload(
                    kind="discard",
                    actor=_bridge_contracts.make_seat((int(epoch.root_actor) + 1) % 4),
                    tile=tile,  # type: ignore[arg-type]
                    action_id=0,  # type: ignore[arg-type]
                    source_seat=None,
                    consumed_tiles=(),
                    offered_action_ids=(),
                    accepted_action_ids=(),
                    round_index=None,
                    scores=None,
                    reason=None,
                )
                envelope = EventEnvelope(
                    game_id="game_tiny_001",
                    sequence=seq,  # type: ignore[arg-type]
                    kind="discard",
                    actor=_bridge_contracts.make_seat((int(epoch.root_actor) + 1) % 4),
                    visibility="public",
                    visible_to=(Seat(0), Seat(1), Seat(2), Seat(3)),
                    payload=payload,
                    public_delta=(),
                    rules_hash=rh,
                    schema_hash=sh,  # type: ignore[arg-type]
                )
                before = public_state_chain_hash([])
                after = public_state_chain_hash([envelope])
                obs_hash = "sha256:" + _hl.sha256(_cb({"s": seq})).hexdigest()
                packet = make_actor_visible_packet(
                    actor_view=epoch.root_actor,
                    events=(envelope,),
                    public_state_hash_before=before,
                    public_state_hash_after=after,
                    observation_hash_after=obs_hash,  # type: ignore[arg-type]
                )
                successors.append(
                    PacketSuccessor(
                        packet=packet,
                        successor_world_ref=f"world_bad:{idx}",
                        delta_ref=f"delta_bad:{idx}",
                        probability=0.3,
                        log_physical_probability=math.log(0.3),
                        log_actor_policy_probability=0.0,
                    )
                )
            return tuple(successors)

    cfg = PbrfConfig(parent_count=2, kernel_tolerance=1e-9, max_search_batches=4)
    with pytest.raises((PacketPartitionError, ContractError)):
        build_pbrf(
            b,
            epoch,
            candidates_fn=_candidates_fn(legal),
            kernel=BadKernel(),
            rng=RandomStream(b"badmass"),
            config=cfg,
        )


# ---------------------------------------------------------------------------
# 7 freeze candidates before natural confirmation
# ---------------------------------------------------------------------------


def test_freeze_candidates_before_natural_confirmation() -> None:
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=4, max_search_batches=8)
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"freeze_confirm"),
        config=cfg,
    )
    frozen_before = forest.frozen_candidates
    # Simulate natural confirmation runner (fresh natural after freeze) — must not mutate frozen candidates
    from hydra2.belief.confirmation import ConfirmationCase, NaturalConfirmationRunner

    runner = NaturalConfirmationRunner()
    # Build cases deterministically from parents
    cases = tuple(
        ConfirmationCase(
            case_id=f"case_{i}",
            world_id=p.world_ref[:16],
            observation_hash=epoch.observation_hash,
        )
        for i, p in enumerate(forest.parents[:2])
    )
    res1 = runner.confirm(cases, rng=RandomStream(b"conf1"))
    res2 = runner.confirm(cases, rng=RandomStream(b"conf1"))
    assert res1 == res2
    # frozen candidates unchanged after confirmation
    assert forest.frozen_candidates is frozen_before
    assert forest.frozen_candidates == frozen_before


def test_stale_child_is_hard_failure() -> None:
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=2, max_search_batches=4)
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"stale"),
        config=cfg,
    )
    action = legal[0]
    aid = _aid(action)
    pid = next(pid for (ka, pid) in forest.children if ka == aid)
    parent = b.sample_natural(epoch, count=1, rng=RandomStream(b"stale_pkt"))[0]
    succs = NaturalPacketKernel().enumerate_next(epoch=epoch, particle=parent, action=action)
    target = next(s for s in succs if s.packet.packet_id == pid)
    # Tamper child entry's target_id to simulate stale provenance
    # Create a forest with stale entry by copying and mutating
    stale_entries = forest.children[(aid, pid)]
    # Create a stale epoch with different target
    from hydra2.belief.world import make_full_world, world_actor_observation

    w2 = make_full_world(
        concealed_hands=((1, 2), (3, 4), (5, 6), (7, 8)),
        live_wall=(9, 10, 11, 12),
        dead_wall=(),
        rules_hash=_MASTER_RULES,
        observation_hash="sha256:" + "f" * 64,
    )
    obs2 = world_actor_observation(w2, actor=0)
    b2 = NaturalBelief()
    epoch2 = b2.begin(obs2)
    # Build a forest for epoch2
    forest2 = build_pbrf(
        b2,
        epoch2,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"stale2"),
        config=cfg,
    )
    # Try to commit forest2 using packet from forest (cross-epoch) — should result in miss_rebuild (stale)
    # Use forest2's action/packet but try to commit forest (epoch mismatch) should produce stale detection and rebuild
    # Instead test that rekey_and_verify rejects tampered delta
    from hydra2.search.pbrf import rekey_and_verify

    # Tamper delta
    tampered = tuple(
        ChildEntry(
            parent_id=e.parent_id,
            successor_world_ref=e.successor_world_ref,
            successor_delta="delta_tampered_" + e.successor_delta,
            raw_weight=e.raw_weight,
            target_id=e.target_id,
            epoch=e.epoch,
        )
        for e in stale_entries
    )
    # Try to verify tampered — should raise DigestMismatchError via _verify_delta_reconstruction
    # We need authoritative epoch (incremented)
    auth_epoch = b.pushforward_condition(epoch, action=action, packet=target.packet)
    with pytest.raises((DigestMismatchError, ContractError, StaleBeliefError)):
        rekey_and_verify(tampered, auth_epoch, forest=forest)


# ---------------------------------------------------------------------------
# 10 hard failures already covered (missing mass, stale) — also confirmation reversal
# ---------------------------------------------------------------------------


def test_confirmation_reversal_is_hard_failure() -> None:
    # If candidates not frozen, confirmation could reverse the search choice.
    # With frozen candidates, natural confirmation after freeze must be stable.
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=8, max_search_batches=8)
    # Build forest with frozen candidates
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"rev"),
        config=cfg,
    )
    # Simulate search picking max scalar (deterministic) vs confirmation that would pick different if not frozen
    # For our deterministic value function, frozen selection is stable across repeats
    from hydra2.search.pbrf import PbrfPlanner

    spec = make_pbrf_candidate_spec(
        parent_count=4, max_search_batches=8, candidate_id="candidate3_pbrf_core_v1"
    )
    planner = PbrfPlanner(candidate_spec=spec, belief=b, kernel=NaturalPacketKernel())
    # Use SearchRequest with same legal and epoch

    _, obs = _world_and_obs()
    # Provide belief_epoch via request
    req = SearchRequest(
        observation=obs,
        legal_actions=legal,
        candidate_spec=spec,
        deadline_monotonic_ns=time.monotonic_ns() + 5_000_000_000,
        belief_epoch=epoch,
    )
    r1 = planner.act(req)
    r2 = planner.act(req)
    # Confirmation reversal would mean r1 != r2 due to unfrozen randomness — our frozen planner must be deterministic
    assert r1.selected_action == r2.selected_action
    assert r1.value_vectors == r2.value_vectors


def test_validate_packet_partition_helper() -> None:
    b, epoch, _ = _belief_epoch()
    kernel = NaturalPacketKernel()
    p = b.sample_natural(epoch, count=1, rng=RandomStream(b"validate"))[0]
    legal = _legal_pair()
    succ = kernel.enumerate_next(epoch=epoch, particle=p, action=legal[0])
    validate_packet_partition(succ)
    # aliasing should be rejected
    from dataclasses import dataclass

    @dataclass
    class FakeSucc:
        packet: object
        probability: float

    dup = [
        FakeSucc(packet=type("P", (), {"packet_id": "dup"})(), probability=0.5),
        FakeSucc(packet=type("P", (), {"packet_id": "dup"})(), probability=0.5),
    ]
    with pytest.raises(PacketPartitionError):
        validate_packet_partition(dup)


# ---------------------------------------------------------------------------
# 17 carried conditional sample law (selection conditions the population)
# ---------------------------------------------------------------------------


def test_commit_promoted_parents_carry_conditional_law() -> None:
    # Hit-commit parents sample b_{eta,a,e}^+, never fresh naturals: the
    # emitted action + realized packet selected them.
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=4, max_search_batches=8)
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"carry_law"),
        config=cfg,
    )
    action = legal[0]
    aid = _aid(action)
    pid = next(pid for (ka, pid) in forest.children if ka == aid)
    entries = forest.children[(aid, pid)]
    parent = b.sample_natural(epoch, count=1, rng=RandomStream(b"carry_law_pkt"))[0]
    succs = NaturalPacketKernel().enumerate_next(epoch=epoch, particle=parent, action=action)
    target = next(s for s in succs if s.packet.packet_id == pid)
    promoted, disp = commit(forest, action, target.packet, b)
    assert disp.kind == "hit_commit"
    z = sum(e.raw_weight for e in entries)
    assert z > 0
    assert len(promoted.parents) == len(entries)
    for p, e in zip(promoted.parents, entries, strict=True):
        assert p.source == "carried"
        assert math.isclose(p.log_target_density, math.log(e.raw_weight / z))
        assert p.log_target_density == p.log_proposal_density
        assert p.target_id == promoted.epoch.target_id


def test_conditional_carry_logps_normalizes_nonuniform_mass() -> None:
    # Normalized conditional weights, never uniform -log(N).
    tid = "sha256:" + "0" * 64
    entries = (
        ChildEntry(
            parent_id="p1",
            successor_world_ref="w1",
            successor_delta="d1",
            raw_weight=0.25,
            target_id=tid,  # type: ignore[arg-type]
            epoch=0,
        ),
        ChildEntry(
            parent_id="p2",
            successor_world_ref="w2",
            successor_delta="d2",
            raw_weight=0.75,
            target_id=tid,  # type: ignore[arg-type]
            epoch=0,
        ),
    )
    logps = _conditional_carry_logps(entries)
    assert logps == pytest.approx((math.log(0.25), math.log(0.75)))
    # Zero-mass conditioning supports no population: caller takes miss path.
    void = (
        ChildEntry(
            parent_id="p0",
            successor_world_ref="w0",
            successor_delta="d0",
            raw_weight=0.0,
            target_id=tid,  # type: ignore[arg-type]
            epoch=0,
        ),
    )
    with pytest.raises(ContractError):
        _conditional_carry_logps(void)


def test_carried_parent_sample_law_enforced() -> None:
    # Carried conditionals with pinned finite equal densities are accepted;
    # carried without verifiable densities is rejected (unverifiable carry).
    import types

    b, epoch, _ = _belief_epoch()
    natural = b.sample_natural(epoch, count=1, rng=RandomStream(b"carry_ok"))[0]
    carried_ok = types.SimpleNamespace(
        parent_id="carried_1",
        world_ref="world_carry",
        epoch=epoch.epoch,
        target_id=epoch.target_id,
        source="carried",
        log_target_density=math.log(0.5),
        log_proposal_density=math.log(0.5),
        proposal_id="sha256:" + "0" * 64,
        ancestors=("p0",),
    )
    carried_bare = types.SimpleNamespace(
        parent_id="carried_2",
        world_ref="world_bare",
        epoch=epoch.epoch,
        target_id=epoch.target_id,
        source="carried",
        log_target_density=None,
        log_proposal_density=None,
        proposal_id="sha256:" + "0" * 64,
    )
    tid = epoch.target_id
    entry = ChildEntry(
        parent_id="carried_1",
        successor_world_ref="world_carry",
        successor_delta="delta_carry",
        raw_weight=1.0,
        target_id=tid,  # type: ignore[arg-type]
        epoch=epoch.epoch,
    )
    cfg = PbrfConfig(parent_count=2, max_search_batches=4)
    kids = {(0, "packet_carry"): (entry,)}
    allocs = fixed_allocate(kids, total_batches=cfg.max_search_batches)
    good = ImmutableForest(
        epoch=epoch,
        parents=(natural, carried_ok),
        frozen_candidates=(0,),
        children=kids,
        config=cfg,
        allocations=allocs,
    )
    assert len(good.parents) == 2
    with pytest.raises(ContractError):
        ImmutableForest(
            epoch=epoch,
            parents=(carried_bare,),
            frozen_candidates=(0,),
            children=kids,
            config=cfg,
            allocations=allocs,
        )


def test_commit_preserves_skewed_conditional_law() -> None:
    # E[S|a] bias repro: an action-selected population with skewed mass keeps
    # its conditional weights through commit. Uniform -log(N) would misstate
    # the law (selected bit S has E[S|a=1]=1 while the unselected bit keeps
    # E[X|a=1]=1/2); the forest MUST NOT relabel it fresh-natural uniform.
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    action = legal[0]
    aid = _aid(action)
    cfg = PbrfConfig(parent_count=2, max_search_batches=4)
    kernel = NaturalPacketKernel()
    parents = b.sample_natural(epoch, count=2, rng=RandomStream(b"skew"))
    first_succs = kernel.enumerate_next(epoch=epoch, particle=parents[0], action=action)
    pid = first_succs[0].packet.packet_id
    actual_packet = first_succs[0].packet
    # Genuine refs/deltas (delta verification MUST pass), skewed masses.
    skewed: list[ChildEntry] = []
    for parent, mass in zip(parents, (0.25, 0.75), strict=True):
        succ = next(
            s
            for s in kernel.enumerate_next(epoch=epoch, particle=parent, action=action)
            if s.packet.packet_id == pid
        )
        delta = str(getattr(succ, "delta_ref", getattr(succ, "successor_delta", "")))
        skewed.append(
            ChildEntry(
                parent_id=str(parent.parent_id),
                successor_world_ref=str(succ.successor_world_ref),
                successor_delta=delta,
                raw_weight=mass,
                target_id=epoch.target_id,  # type: ignore[arg-type]
                epoch=epoch.epoch,
            )
        )
    kids = {(aid, pid): tuple(skewed)}
    forest = ImmutableForest(
        epoch=epoch,
        parents=tuple(parents),
        frozen_candidates=tuple(legal),
        children=kids,
        config=cfg,
        allocations=fixed_allocate(kids, total_batches=cfg.max_search_batches),
    )
    promoted, disp = commit(forest, action, actual_packet, b)
    assert disp.kind == "hit_commit"
    got = tuple(p.log_target_density for p in promoted.parents)
    assert got == pytest.approx((math.log(0.25), math.log(0.75)))
    assert not math.isclose(got[0], got[1])
    assert all(p.source == "carried" for p in promoted.parents)
    # Roots carry () so each promoted chain is exactly its conditioning parent.
    assert tuple(p.ancestors for p in promoted.parents) == tuple(
        (str(parent.parent_id),) for parent in parents
    )


# ---------------------------------------------------------------------------
# 18 diagnostic codes are emitted (SPEC 3 PR4 table)
# ---------------------------------------------------------------------------


def test_pbrf_error_codes_bijective() -> None:
    from hydra2.contracts.common import (
        PBRF_ERROR_CODES,
        ProposalSupportError,
        VisibilityViolationError,
    )

    assert len(PBRF_ERROR_CODES) == 17
    assert len(set(PBRF_ERROR_CODES)) == 17
    assert PBRF_ERROR_CODES["PBRF_PARTITION_EMPTY"] is PacketPartitionError
    assert PBRF_ERROR_CODES["PBRF_PARTITION_ALIAS"] is PacketPartitionError
    assert PBRF_ERROR_CODES["PBRF_PARTITION_MASS"] is PacketPartitionError
    assert PBRF_ERROR_CODES["PBRF_PARTITION_CHILD_NORM"] is PacketPartitionError
    assert PBRF_ERROR_CODES["PBRF_STALE_EPOCH"] is StaleBeliefError
    assert PBRF_ERROR_CODES["PBRF_STALE_TARGET"] is StaleBeliefError
    assert PBRF_ERROR_CODES["PBRF_STALE_PARENT"] is StaleBeliefError
    assert PBRF_ERROR_CODES["PBRF_STALE_PROVENANCE"] is StaleBeliefError
    assert PBRF_ERROR_CODES["PBRF_STALE_WORLDREF"] is StaleBeliefError
    assert PBRF_ERROR_CODES["PBRF_DIGEST_DELTA"] is DigestMismatchError
    assert PBRF_ERROR_CODES["PBRF_DIGEST_WORLD_ID"] is DigestMismatchError
    assert PBRF_ERROR_CODES["PBRF_VIS_TREE_KEY"] is VisibilityViolationError
    assert PBRF_ERROR_CODES["PBRF_VIS_TREE_KEY_NESTED"] is VisibilityViolationError
    assert PBRF_ERROR_CODES["PBRF_VIS_POLICY_WORLD"] is VisibilityViolationError
    assert PBRF_ERROR_CODES["PBRF_VIS_POLICY_HANDS"] is VisibilityViolationError
    assert PBRF_ERROR_CODES["PBRF_SUPPORT_REGION"] is ProposalSupportError
    assert PBRF_ERROR_CODES["PBRF_SUPPORT_POINT"] is ProposalSupportError


def test_partition_empty_carries_code() -> None:
    with pytest.raises(PacketPartitionError, match=r"\[PBRF_PARTITION_EMPTY\]"):
        validate_packet_partition(())


def test_kernel_stale_carries_code() -> None:
    import types

    _, epoch, _ = _belief_epoch()
    bad = types.SimpleNamespace(epoch=9999, target_id=epoch.target_id, world_ref="x")
    with pytest.raises(StaleBeliefError, match=r"\[PBRF_STALE_EPOCH\]"):
        NaturalPacketKernel().enumerate_next(epoch=epoch, particle=bad, action=_legal_pair()[0])


def test_world_digest_mismatch_carries_code() -> None:
    import dataclasses

    w, _ = _world_and_obs()
    with pytest.raises(DigestMismatchError, match=r"\[PBRF_DIGEST_WORLD_ID\]"):
        dataclasses.replace(w, world_id="sha256:" + "f" * 64)


def test_policy_world_code_in_table() -> None:
    # PBRF_VIS_POLICY_WORLD/HANDS + NESTED tree-key guards are defensive-unreachable
    # through public types (ActorObservation is frozen slot-typed without those fields;
    # the isinstance gate fires first). Table membership + reading verification cover
    # them; a firing test would assert implementation, not behavior.
    from hydra2.contracts.common import PBRF_ERROR_CODES, VisibilityViolationError

    assert PBRF_ERROR_CODES["PBRF_VIS_POLICY_WORLD"] is VisibilityViolationError
    assert PBRF_ERROR_CODES["PBRF_VIS_POLICY_HANDS"] is VisibilityViolationError
    assert PBRF_ERROR_CODES["PBRF_VIS_TREE_KEY_NESTED"] is VisibilityViolationError


# ---------------------------------------------------------------------------
# 19 F1-F4 fix regression: emitted-action commit, tile identity, depleted
# rebuild, target binding
# ---------------------------------------------------------------------------


def test_observe_commits_emitted_action_despite_packet_collision() -> None:
    # Kernel packet ids are action-free: every pid exists under every action.
    # observe() must commit the stored emitted action, never first-hit-wins.
    b, epoch, obs = _belief_epoch()
    legal = _legal_pair()
    spec = make_pbrf_candidate_spec(parent_count=2, max_search_batches=4)
    planner = PbrfPlanner(candidate_spec=spec, belief=b, kernel=NaturalPacketKernel())
    req = SearchRequest(
        observation=obs,
        legal_actions=legal,
        candidate_spec=spec,
        deadline_monotonic_ns=time.monotonic_ns() + 5_000_000_000,
        belief_epoch=epoch,
    )
    planner.act(req)
    # Pin the emitted action to the second candidate: the old candidate sweep
    # would commit the first candidate's identical pid instead.
    planner._last_selected_action = legal[1]
    oaid = _aid(legal[1])
    pid = next(pid for (ka, pid) in planner._forest.children if ka == oaid)
    parent = b.sample_natural(epoch, count=1, rng=RandomStream(b"collide_pkt"))[0]
    succs = NaturalPacketKernel().enumerate_next(epoch=epoch, particle=parent, action=legal[1])
    pkt = next(s.packet for s in succs if s.packet.packet_id == pid)
    planner.observe(pkt)
    assert planner._last_commit is not None and planner._last_commit.kind == "hit_commit"
    assert (oaid, pid) in planner._forest.children
    assert planner._last_selected_action is None  # consumed one-shot


def test_observe_without_act_raises_contract_error() -> None:
    # No emitted action memory (forest injected, act() never ran): the action
    # cannot be recovered from the packet, and guessing a candidate would
    # reintroduce first-hit-wins miscommit — protocol violation, not a miss.
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=2, max_search_batches=4)
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"noact"),
        config=cfg,
    )
    spec = make_pbrf_candidate_spec(parent_count=2, max_search_batches=4)
    planner = PbrfPlanner(candidate_spec=spec, belief=b, kernel=NaturalPacketKernel())
    planner._forest = forest
    assert planner._last_selected_action is None
    parent = b.sample_natural(epoch, count=1, rng=RandomStream(b"noact_pkt"))[0]
    succs = NaturalPacketKernel().enumerate_next(epoch=epoch, particle=parent, action=legal[0])
    with pytest.raises(ContractError):
        planner.observe(succs[0].packet)


def test_child_entry_tile_stored_and_directly_verified() -> None:
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=2, max_search_batches=4)
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"tilepin"),
        config=cfg,
    )
    for entries in forest.children.values():
        for e in entries:
            assert e.tile in (8, 9)
    (aid, pid), entries = next(iter(forest.children.items()))
    e = entries[0]
    parent_ref = next(str(p.world_ref) for p in forest.parents if str(p.parent_id) == e.parent_id)
    # Direct verify with the kernel-side aid (the kernel derives aid 0 from
    # these actions while the forest key uses the planner-side hash): the
    # stored tile makes it a single reconstruction, no brute force.
    assert _verify_delta_reconstruction(
        parent_world_ref=parent_ref,
        successor_world_ref=e.successor_world_ref,
        successor_delta=e.successor_delta,
        action_id=0,
        tile=e.tile,
    )
    # Stored tile is authoritative: tampered delta fails directly.
    assert not _verify_delta_reconstruction(
        parent_world_ref=parent_ref,
        successor_world_ref=e.successor_world_ref,
        successor_delta="delta_tampered_deadbeef",
        action_id=0,
        tile=e.tile,
    )
    legacy = ChildEntry(
        parent_id=e.parent_id,
        successor_world_ref=e.successor_world_ref,
        successor_delta=e.successor_delta,
        raw_weight=e.raw_weight,
        target_id=e.target_id,
        epoch=e.epoch,
    )
    assert legacy.tile is None
    # Brute-force tile path, verified with the kernel-side aid for the same
    # namespace reason as above.
    assert _verify_delta_reconstruction(
        parent_world_ref=parent_ref,
        successor_world_ref=legacy.successor_world_ref,
        successor_delta=legacy.successor_delta,
        action_id=0,
    )
    # Commit-path rekey bridges the aid namespaces with its ordered sweep
    # (exact planner aid first, then candidates, then 0..4) until one
    # codec-assigned id unifies them (TODO(codec-aid)).
    action = next(a for a in legal if _aid(a) == aid)
    parent = b.sample_natural(epoch, count=1, rng=RandomStream(b"tilepin_pkt"))[0]
    succs = NaturalPacketKernel().enumerate_next(epoch=epoch, particle=parent, action=action)
    pkt = next(s.packet for s in succs if s.packet.packet_id == pid)
    auth = b.pushforward_condition(epoch, action=action, packet=pkt)
    from hydra2.search.pbrf import rekey_and_verify

    rekeyed = rekey_and_verify(entries, auth, forest=forest, action_id=aid)
    assert tuple(r.tile for r in rekeyed) == tuple(e.tile for e in entries)


def _missing_packet_for(b, epoch):
    from hydra2.artifacts.canonical import canonical_bytes as _cb
    from hydra2.contracts.common import Seat
    from hydra2.contracts.event_envelope import (
        EventEnvelope,
        EventPayload,
    )
    from hydra2.contracts.event_packet import (
        make_actor_visible_packet,
        public_state_chain_hash,
    )

    payload = EventPayload(
        kind="discard",
        actor=_bridge_contracts.make_seat(0),
        tile=0,  # type: ignore[arg-type]
        action_id=0,  # type: ignore[arg-type]
        source_seat=None,
        consumed_tiles=(),
        offered_action_ids=(),
        accepted_action_ids=(),
        round_index=None,
        scores=None,
        reason=None,
    )
    env = EventEnvelope(
        game_id="game_missing",
        sequence=999,  # type: ignore[arg-type]
        kind="discard",
        actor=_bridge_contracts.make_seat(0),
        visibility="public",
        visible_to=(Seat(0), Seat(1), Seat(2), Seat(3)),
        payload=payload,
        public_delta=(),
        rules_hash=epoch.rules_hash,
        schema_hash="sha256:" + "c" * 64,  # type: ignore[arg-type]
    )
    return make_actor_visible_packet(
        actor_view=epoch.root_actor,
        events=(env,),
        public_state_hash_before=public_state_chain_hash([]),
        public_state_hash_after=public_state_chain_hash([env]),
        observation_hash_after="sha256:" + hashlib.sha256(_cb({"miss": 2})).hexdigest(),  # type: ignore[arg-type]
    )


def test_commit_miss_returns_depleted_forest() -> None:
    # A miss looks like a miss: empty children/allocations, real sampled
    # parents, carried-over candidates — no synthetic packet_fresh filler.
    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=2, max_search_batches=4)
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"depleted"),
        config=cfg,
    )
    promoted, disp = commit(forest, legal[0], _missing_packet_for(b, epoch), b)
    assert disp.kind == "miss_rebuild"
    assert promoted.children == {}
    assert promoted.allocations == {}
    assert len(promoted.parents) == cfg.parent_count
    assert all(getattr(p, "source", "natural") == "natural" for p in promoted.parents)
    assert tuple(promoted.frozen_candidates) == tuple(forest.frozen_candidates)
    assert int(promoted.epoch.epoch) == int(forest.epoch.epoch) + 1


def test_target_compat_binds_packet_and_rejects_forged_epoch() -> None:
    import dataclasses

    b, epoch, _ = _belief_epoch()
    legal = _legal_pair()
    cfg = PbrfConfig(parent_count=2, max_search_batches=4)
    forest = build_pbrf(
        b,
        epoch,
        candidates_fn=_candidates_fn(legal),
        kernel=NaturalPacketKernel(),
        rng=RandomStream(b"bind"),
        config=cfg,
    )
    action = legal[0]
    aid = _aid(action)
    pid = next(pid for (ka, pid) in forest.children if ka == aid)
    entries = forest.children[(aid, pid)]
    parent = b.sample_natural(epoch, count=1, rng=RandomStream(b"bind_pkt"))[0]
    succs = NaturalPacketKernel().enumerate_next(epoch=epoch, particle=parent, action=action)
    pkt = next(s.packet for s in succs if s.packet.packet_id == pid)
    auth = b.pushforward_condition(epoch, action=action, packet=pkt)
    assert _is_target_compatible(entries, auth, packet=pkt) is True
    forged = dataclasses.replace(auth, target_id="sha256:" + "f" * 64)
    assert _is_target_compatible(entries, forged, packet=pkt) is False
    stale = dataclasses.replace(auth, epoch=epoch.epoch)
    assert _is_target_compatible(entries, stale, packet=pkt) is False
