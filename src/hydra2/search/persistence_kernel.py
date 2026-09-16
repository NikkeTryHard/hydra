"""Persistence factorial kernel — B/F/R/P/C arms, packets, forest state.

Owns the SPEC 17 arm vocabulary shared by the planner and the report:
the frozen :class:`PersistenceArm` record with its exact per-arm
invariants, the :data:`ARM_DEFS` table, the :func:`make_persistence_arm`
factory, the finite packet kernel (:class:`FinitePacket`,
:func:`compute_packet_id`, :class:`BeliefEpochLite`,
:func:`enumerate_packets_for`, :func:`fresh_rebuild_epoch`,
:func:`commit_equals_rebuild`), the deterministic helpers
(:func:`_obs_hash_from_epoch`, :func:`_action_key`,
:func:`_distribute_quota`), and the speculative :class:`ForestState`
retained by the R/P arms. The per-arm state machine lives in
:mod:`hydra2.search.persistence_planner` and the frozen whole-block
report in :mod:`hydra2.search.persistence_report` so each file stays
inside the review-size ceiling.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError, DigestText, make_digest_text

try:
    from hydra2.search.common import (
        DEPLOYABLE_DEADLINE_MS,
        CandidateSpec,
    )

    _COMMON_AVAILABLE = True
except ImportError as exc:
    raise ImportError(
        "hydra2.search.common is required for persistence_kernel; "
        "the minimal-contract fallback was removed (single authority is search.common)"
    ) from exc


__all__ = [
    "ARM_DEFS",
    "BeliefEpochLite",
    "CandidateSpec",
    "FinitePacket",
    "ForestState",
    "PersistenceArm",
    "_action_key",
    "_distribute_quota",
    "_obs_hash_from_epoch",
    "commit_equals_rebuild",
    "compute_packet_id",
    "enumerate_packets_for",
    "fresh_rebuild_epoch",
    "make_persistence_arm",
]


# ---------------------------------------------------------------------------
# PersistenceArm — SPEC 17 exact
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PersistenceArm:
    """SPEC 17 PersistenceArm — frozen, validated.

    Field order matches specification exactly.
    """

    id: Literal["B", "F", "R", "P", "C"]
    retain_state: bool
    opponent_time_compute: bool
    own_deadline_ms: int
    extra_wait_allowance_ms: int
    deployable: bool

    def __post_init__(self) -> None:
        if self.id not in ("B", "F", "R", "P", "C"):
            raise ContractError(f"PersistenceArm id must be B/F/R/P/C, got {self.id!r}")
        if not isinstance(self.retain_state, bool):
            raise ContractError("retain_state must be bool")
        if not isinstance(self.opponent_time_compute, bool):
            raise ContractError("opponent_time_compute must be bool")
        for name in ("own_deadline_ms", "extra_wait_allowance_ms"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ContractError(f"{name} must be nonneg int, got {v!r}")
            if name == "own_deadline_ms" and v <= 0:
                raise ContractError("own_deadline_ms must be positive")
        if not isinstance(self.deployable, bool):
            raise ContractError("deployable must be bool")
        # SPEC invariants per arm
        expected = ARM_DEFS[self.id]
        for k in ("retain_state", "opponent_time_compute", "deployable"):
            if getattr(self, k) != expected[k]:
                raise ContractError(
                    f"Arm {self.id} invariant: {k} must be {expected[k]}, got {getattr(self, k)!r}"
                )
        # B/F/R/P must share deployable deadline <=5000
        if self.id in ("B", "F", "R", "P"):
            if self.own_deadline_ms > DEPLOYABLE_DEADLINE_MS:
                raise ContractError(
                    f"deployable arm {self.id} deadline must be <=5000, got {self.own_deadline_ms}"
                )
            if self.extra_wait_allowance_ms != 0:
                raise ContractError(f"deployable arm {self.id} extra_wait_allowance must be 0")
        else:  # C laboratory
            if self.extra_wait_allowance_ms <= 0:
                raise ContractError("C must have positive extra_wait_allowance_ms")
            if self.deployable:
                raise ContractError("C must not be deployable")


ARM_DEFS: dict[str, dict[str, Any]] = {
    "B": {
        "retain_state": False,
        "opponent_time_compute": False,
        "own_deadline_ms": DEPLOYABLE_DEADLINE_MS,
        "extra_wait_allowance_ms": 0,
        "deployable": True,
        "description": "Frozen policy, no search.",
    },
    "F": {
        "retain_state": False,
        "opponent_time_compute": False,
        "own_deadline_ms": DEPLOYABLE_DEADLINE_MS,
        "extra_wait_allowance_ms": 0,
        "deployable": True,
        "description": "Fresh search at each own decision; discard state; no opponent-time compute.",
    },
    "R": {
        "retain_state": True,
        "opponent_time_compute": False,
        "own_deadline_ms": DEPLOYABLE_DEADLINE_MS,
        "extra_wait_allowance_ms": 0,
        "deployable": True,
        "description": "Retain compatible state but pause all search during opponent turns.",
    },
    "P": {
        "retain_state": True,
        "opponent_time_compute": True,
        "own_deadline_ms": DEPLOYABLE_DEADLINE_MS,
        "extra_wait_allowance_ms": 0,
        "deployable": True,
        "description": "Retain state and ponder only after emitted action until its next actor-visible packet.",
    },
    "C": {
        "retain_state": False,
        "opponent_time_compute": False,
        "own_deadline_ms": DEPLOYABLE_DEADLINE_MS,
        "extra_wait_allowance_ms": 2000,
        "deployable": False,
        "description": "Laboratory-only fresh-search control with extended allowance; never deployable.",
    },
}


def make_persistence_arm(arm_id: Literal["B", "F", "R", "P", "C"]) -> PersistenceArm:
    """Construct validated PersistenceArm for the named arm id."""
    if arm_id not in ARM_DEFS:
        raise ContractError(f"unknown arm {arm_id!r}")
    d = ARM_DEFS[arm_id]
    return PersistenceArm(
        id=arm_id,
        retain_state=d["retain_state"],
        opponent_time_compute=d["opponent_time_compute"],
        own_deadline_ms=d["own_deadline_ms"],
        extra_wait_allowance_ms=d["extra_wait_allowance_ms"],
        deployable=d["deployable"],
    )


# ---------------------------------------------------------------------------
# Packet and forest state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FinitePacket:
    """One actor-visible packet in the finite kernel.

    Partition invariant: per (parent, action) the set of packets is pairwise
    disjoint by packet_id and sums probability to one.
    """

    packet_id: DigestText
    action_id: int
    epoch_before: str
    epoch_after: str
    probability: float
    delta: tuple[int, ...]  # successor delta placeholder (opaque but deterministic)

    def __post_init__(self) -> None:
        _ = make_digest_text(self.packet_id)
        if not 0.0 < self.probability <= 1.0:
            raise ContractError(f"packet probability must be in (0,1], got {self.probability!r}")
        if not math.isfinite(self.probability):
            raise ContractError("packet probability must be finite")


def compute_packet_id(*, epoch_before: str, action_id: int, branch: int) -> DigestText:
    raw = canonical_bytes({"epoch_before": epoch_before, "action_id": action_id, "branch": branch})
    return DigestText("sha256:" + hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True, slots=True)
class BeliefEpochLite:
    """Minimal belief epoch for persistence tests (mirrors natural harness identity)."""

    epoch: str  # BeliefEpochId string
    observation_hash: DigestText
    target_id: DigestText
    root_actor: int

    def __post_init__(self) -> None:
        _ = make_digest_text(self.observation_hash)
        _ = make_digest_text(self.target_id)


def _obs_hash_from_epoch(epoch: BeliefEpochLite | str) -> str:
    if isinstance(epoch, str):
        return epoch
    return epoch.observation_hash


def _action_key(a: Any) -> int:
    """Deterministic integer key for a CanonicalAction without mutating it."""
    try:
        from hydra2.contracts.action import ACTION_KIND_ORDINALS

        _ord = ACTION_KIND_ORDINALS
    except Exception:
        _ord = {"pass": 0, "discard": 1, "tsumogiri": 2}
    kind = getattr(a, "kind", None)
    tile = getattr(a, "tile", None)
    if tile is not None:
        try:
            return int(tile)
        except Exception:
            pass
    if kind in _ord:
        return int(_ord[kind])  # type: ignore[index]
    return int(hashlib.sha256(repr(a).encode()).hexdigest()[:8], 16)


def enumerate_packets_for(
    *,
    epoch: BeliefEpochLite | str,
    action_id: int,
    num_branches: int = 2,
) -> tuple[FinitePacket, ...]:
    """Exhaustive disjoint packet kernel per (epoch, action) — mass one.

    Deterministic via semantic seeds: (epoch, action_id). Probabilities are
    fixed by branch index to keep fixtures reproducible; they sum to one and
    are pairwise disjoint by packet_id.
    """
    if num_branches <= 0:
        raise ContractError("num_branches must be positive")
    epoch_id = epoch.epoch if isinstance(epoch, BeliefEpochLite) else epoch
    packets: list[FinitePacket] = []
    # Use simple Dirichlet-like split: uniform for tests unless branch 0 is dominant
    # For determinism, branch 0 gets 0.7, remaining share 0.3
    if num_branches == 1:
        probs = [1.0]
    elif num_branches == 2:
        probs = [0.7, 0.3]
    else:
        rem = 1.0 / num_branches
        probs = [rem] * num_branches
    for b in range(num_branches):
        pid = compute_packet_id(epoch_before=epoch_id, action_id=action_id, branch=b)
        # epoch_after is hash of predecessor plus packet
        after_raw = canonical_bytes({"epoch_before": epoch_id, "packet_id": pid})
        epoch_after = "epoch:" + hashlib.sha256(after_raw).hexdigest()[:16]
        pkt = FinitePacket(
            packet_id=pid,
            action_id=action_id,
            epoch_before=epoch_id,
            epoch_after=epoch_after,
            probability=probs[b],
            delta=(action_id, b),
        )
        packets.append(pkt)
    # Validate partition
    total = sum(p.probability for p in packets)
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ContractError(f"packet mass {total} != 1")
    pids = [p.packet_id for p in packets]
    if len(pids) != len(set(pids)):
        raise ContractError("packet ids must be disjoint")
    return tuple(packets)


def fresh_rebuild_epoch(
    *,
    epoch_before: BeliefEpochLite | str,
    packet: FinitePacket,
) -> str:
    """Authoritative fresh posterior epoch_after (rebuild).

    Must digest-equal the successor stored in packet.epoch_after when the packet
    is the REALIZED one (mass-one partition guarantee).
    """
    epoch_id = epoch_before.epoch if isinstance(epoch_before, BeliefEpochLite) else epoch_before
    if packet.epoch_before != epoch_id:
        raise ContractError(f"packet epoch_before {packet.epoch_before!r} != epoch {epoch_id!r}")
    raw = canonical_bytes({"epoch_before": epoch_id, "packet_id": packet.packet_id})
    rebuilt = "epoch:" + hashlib.sha256(raw).hexdigest()[:16]
    return rebuilt


def commit_equals_rebuild(
    *,
    epoch_before: BeliefEpochLite | str,
    packet: FinitePacket,
) -> bool:
    """Check commit/rebuild equality fixture."""
    rebuilt = fresh_rebuild_epoch(epoch_before=epoch_before, packet=packet)
    return rebuilt == packet.epoch_after


def _distribute_quota(sorted_pids: list[str], quota: int) -> dict[str, int]:
    """Deterministically spread quota units across sorted child ids.

    Round-robin one unit per pid until quota exhausts. Pure function of its
    inputs; the returned per-pid units sum to min(quota, distributed). Callers
    charge every counter from the returned mapping so stats stay coherent.
    """
    dist: dict[str, int] = dict.fromkeys(sorted_pids, 0)
    remaining = quota
    while remaining > 0 and len(dist) > 0:
        for pid in sorted_pids:
            if remaining <= 0:
                break
            dist[pid] += 1
            remaining -= 1
    return dist


@dataclass(slots=True)
class ForestState:
    """Speculative forest retained by R/P arms.

    - parent_epoch: epoch before action
    - action_id: emitted action
    - children: packet-conditioned child views (speculative)
    - child_stats: per-child visit counters for ponder work
    - provenance_epoch: binds forest to belief target; stale if mismatch
    """

    parent_epoch: str
    action_id: int
    children: dict[str, FinitePacket] = field(default_factory=dict)
    child_stats: dict[str, int] = field(default_factory=dict)
    provenance_target: str | None = None
    ponder_calls: int = 0
    created_at_ns: int = 0

    def is_empty(self) -> bool:
        return len(self.children) == 0

    def clear(self) -> None:
        self.children.clear()
        self.child_stats.clear()
        self.ponder_calls = 0
