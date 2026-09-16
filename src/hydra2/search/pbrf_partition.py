# ruff: noqa: B904  # reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (SIM105 fallback-chain try/except-pass idiom; B007/F841 intentional scratch loop locals; B904 ContractError preconditions; N814 upstream casing; F401 guard shims homed here for the split modules). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 3 PBRF partition — frozen config, child records, allocation guards.

Owns the frozen :class:`PbrfConfig` hyper-parameters, the
:class:`ChildEntry` particle record, the :class:`CommitDisposition`
result tag, and the packet-partition helpers beside their only callers:
the action-id derivation, the exhaustive-disjoint partition guards, the
root-candidate freezer, the deterministic batch allocator, and the
per-key normalizer diagnostics (Z-hat, normalized weights, ESS). It is
also the single home for the guarded dependency blocks with their
fail-closed ``_REQUIRE_*``/``_IMPORT_ERROR`` guards, re-exported by the other ``pbrf_*`` modules.
The forest and core builder live in :mod:`hydra2.search.pbrf_forest`,
the commit path in :mod:`hydra2.search.pbrf_commit`, the CandidateSpec
factory in :mod:`hydra2.search.pbrf_spec`, and the Planner runner in
:mod:`hydra2.search.pbrf_search` plus :mod:`hydra2.search.pbrf_act` so
each file stays inside the review-size ceiling.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import (
    ContractError,
    DigestText,
    PacketPartitionError,
    make_digest_text,
    make_parent_id,
    make_tile_id,
)

try:
    from hydra2.contracts.randomness import RandomStream

    _RANDOM_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    RandomStream = Any  # placeholder; _require_random_stream() raises on use
    _RANDOM_IMPORT_ERROR = exc


def _require_random_stream() -> Any:
    """Fail-closed RNG access (lazy ImportError with build-ext hint)."""
    if _RANDOM_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.contracts.randomness not importable "
            f"({_RANDOM_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before PBRF search"
        ) from _RANDOM_IMPORT_ERROR
    return RandomStream


try:
    from hydra2.belief.kernel import NaturalPacketKernel, PacketSuccessor

    _KERNEL_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    NaturalPacketKernel = Any  # placeholder; _require_kernel() raises on use
    PacketSuccessor = Any
    _KERNEL_IMPORT_ERROR = exc


def _require_kernel() -> Any:
    """Fail-closed kernel access (lazy ImportError with build-ext hint)."""
    if _KERNEL_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.belief.kernel not importable "
            f"({_KERNEL_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before PBRF search"
        ) from _KERNEL_IMPORT_ERROR
    return NaturalPacketKernel


try:
    from hydra2.belief.natural import BeliefEpoch, NaturalBelief, Particle, PolicySet

    _BELIEF_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    BeliefEpoch = Any  # placeholder; _require_belief() raises on use
    NaturalBelief = Any
    Particle = Any
    PolicySet = Any
    _BELIEF_IMPORT_ERROR = exc


def _require_belief() -> None:
    """Fail-closed belief access (lazy ImportError with build-ext hint)."""
    if _BELIEF_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.belief.natural not importable "
            f"({_BELIEF_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before PBRF search"
        ) from _BELIEF_IMPORT_ERROR


try:
    from hydra2.contracts.event_packet import ActorVisiblePacket

    _PACKET_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    ActorVisiblePacket = Any  # placeholder; _require_packet() raises on use
    _PACKET_IMPORT_ERROR = exc


def _require_packet() -> Any:
    """Fail-closed packet access (lazy ImportError with build-ext hint)."""
    if _PACKET_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.contracts.event_packet not importable "
            f"({_PACKET_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before PBRF search"
        ) from _PACKET_IMPORT_ERROR
    return ActorVisiblePacket


try:
    from hydra2.eval.telemetry import ResourceTelemetry, make_resource_telemetry

    _TELEMETRY_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover
    ResourceTelemetry = Any  # placeholder; _require_telemetry() raises on use
    make_resource_telemetry = Any
    _TELEMETRY_IMPORT_ERROR = exc


def _require_telemetry() -> Any:
    """Fail-closed telemetry access (lazy ImportError with build-ext hint)."""
    if _TELEMETRY_IMPORT_ERROR is not None:
        raise ImportError(
            "hydra2.eval.telemetry not importable "
            f"({_TELEMETRY_IMPORT_ERROR}); build the bridge with `pixi run build-ext` "
            "before PBRF search"
        ) from _TELEMETRY_IMPORT_ERROR
    return make_resource_telemetry


logger = logging.getLogger(__name__)

__all__ = [
    "ActorVisiblePacket",
    "BeliefEpoch",
    "ChildEntry",
    "CommitDisposition",
    "NaturalBelief",
    "NaturalPacketKernel",
    "PacketSuccessor",
    "Particle",
    "PbrfConfig",
    "PolicySet",
    "RandomStream",
    "ResourceTelemetry",
    "_action_id",
    "_ess_for_key",
    "_freeze_candidates",
    "_normalized_weights",
    "_require_partition",
    "_z_hat_for_key",
    "fixed_allocate",
    "logger",
    "make_resource_telemetry",
    "validate_packet_partition",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PbrfConfig:
    """Frozen PBRF hyper-parameters (part of CandidateSpec.parameters)."""

    parent_count: int = 16
    kernel_tolerance: float = 1e-9
    max_search_batches: int = 64
    resource_view: Literal["calls", "transitions", "joules"] = "calls"
    tie_break: str = "lexicographic"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parent_count, int)
            or isinstance(self.parent_count, bool)
            or self.parent_count <= 0
        ):
            raise ContractError("parent_count must be positive int")
        if (
            not isinstance(self.kernel_tolerance, float)
            or not math.isfinite(self.kernel_tolerance)
            or not (0 < self.kernel_tolerance < 0.01)
        ):
            raise ContractError("kernel_tolerance must be small positive float in (0,0.01)")
        if (
            not isinstance(self.max_search_batches, int)
            or isinstance(self.max_search_batches, bool)
            or self.max_search_batches <= 0
        ):
            raise ContractError("max_search_batches must be positive int")
        if self.resource_view not in ("calls", "transitions", "joules"):
            raise ContractError("resource_view must be calls, transitions, or joules")
        if self.tie_break not in ("lexicographic", "stable_hash"):
            raise ContractError("tie_break must be lexicographic or stable_hash")


@dataclass(frozen=True, slots=True)
class ChildEntry:
    """One particle's contribution to a specific (action, packet) child.

    Mirrors SPEC 16.4 pseudocode: parent_id, successor_world_ref, successor_delta,
    raw_weight (= probability/len(parents)), target_id, epoch.

    ``tile`` is the originating transition tile (TileId 0..135) captured at
    enumeration time from the successor packet events. It lets delta
    verification reconstruct directly instead of brute-forcing the hash
    preimage. ``None`` marks legacy/unknown provenance and keeps the
    brute-force fallback available; new entries from ``build_pbrf`` always
    carry it.
    """

    parent_id: str
    successor_world_ref: str
    successor_delta: str
    raw_weight: float
    target_id: DigestText
    epoch: Any
    ancestors: tuple[str, ...] = ()
    tile: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parent_id, str) or self.parent_id == "":
            raise ContractError("parent_id must be non-empty str")
        _: Any = make_parent_id(self.parent_id)
        if not isinstance(self.successor_world_ref, str) or self.successor_world_ref == "":
            raise ContractError("successor_world_ref must be non-empty str")
        if not isinstance(self.successor_delta, str) or self.successor_delta == "":
            raise ContractError("successor_delta must be non-empty str")
        if (
            not isinstance(self.raw_weight, float)
            or not math.isfinite(self.raw_weight)
            or self.raw_weight < 0
        ):
            raise ContractError("raw_weight must be finite nonnegative float")
        object.__setattr__(self, "target_id", make_digest_text(self.target_id))
        if not isinstance(self.ancestors, tuple) or any(
            not isinstance(a, str) or a == "" for a in self.ancestors
        ):
            raise ContractError("ancestors must be a tuple of non-empty parent-id strings")
        for a in self.ancestors:
            _: Any = make_parent_id(a)
        if self.tile is not None:
            if isinstance(self.tile, bool) or not isinstance(self.tile, int):
                raise ContractError("tile must be a TileId int or None")
            try:
                object.__setattr__(self, "tile", int(make_tile_id(self.tile)))
            except ContractError:
                raise ContractError("tile must be a TileId in 0..135 or None")


@dataclass(frozen=True, slots=True)
class CommitDisposition:
    kind: Literal["hit_commit", "miss_rebuild"]

    def __post_init__(self) -> None:
        if self.kind not in ("hit_commit", "miss_rebuild"):
            raise ContractError("CommitDisposition kind must be hit_commit or miss_rebuild")


# ---------------------------------------------------------------------------
# Helpers — packet partition, action ids, freezing, allocation
# ---------------------------------------------------------------------------


def _action_id(action: Any) -> int:
    # Wave 2 bridge audit: kept Python — action-id normalization (incl. raw-int
    # passthrough and hash fallback) has no pyfn cover; bridge takes aid explicitly.
    v: Any = getattr(action, "action_id", None)
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(action, int) and not isinstance(action, bool):
        return action
    # fallback deterministic hash
    return int(hashlib.sha256(str(action).encode()).hexdigest()[:8], 16) & 0xFFFF


def validate_packet_partition(successors: Any, *, tolerance: float = 1e-9) -> None:
    """Validate exhaustive disjoint packet partition (SPEC 14.3).

    Checks finite nonnegative probabilities sum to 1 within tolerance and
    pairwise distinct packet_id. Mirrors DESPOT helper for cross-test consistency.
    """
    if not bool(successors):
        raise PacketPartitionError("successors must be non-empty [PBRF_PARTITION_EMPTY]")
    pids: list[str] = []
    total = 0.0
    for s_any in successors:
        s: Any = s_any
        pid: Any = getattr(getattr(s, "packet", None), "packet_id", None)
        if pid is None:
            _pid_tmp: Any = getattr(s, "packet_id", None)
            pid = _pid_tmp if _pid_tmp is not None and bool(_pid_tmp) else str(s)
        if not isinstance(pid, str) or pid == "":
            raise ContractError("successor packet_id must be non-empty str")
        pids.append(pid)
        prob = getattr(s, "probability", None)
        if prob is not None:
            if (
                not isinstance(prob, (int, float))
                or not math.isfinite(float(prob))
                or float(prob) < 0
            ):
                raise ContractError("probability must be finite nonnegative")
            total += float(prob)
    if len(pids) != len(set(pids)):
        raise PacketPartitionError(
            f"packet aliasing: duplicate packet_id in {[p[:12] for p in pids]} [PBRF_PARTITION_ALIAS]"
        )
    if any(hasattr(s, "probability") for s in successors) and abs(total - 1.0) > tolerance:
        raise PacketPartitionError(
            f"packet mass {total} != 1 within {tolerance} [PBRF_PARTITION_MASS]"
        )


def _require_partition(successors: Any, tolerance: float) -> None:
    validate_packet_partition(successors, tolerance=tolerance)


def _freeze_candidates(candidates: Any) -> tuple[Any, ...]:
    """Freeze root candidate generator before search evidence (immutable)."""
    if not isinstance(candidates, (list, tuple)):
        raise ContractError("candidates must be a sequence")
    if len(candidates) == 0:
        raise ContractError("candidates must be non-empty")
    # Deduplicate by action_id deterministically, preserve lexicographic order
    seen: dict[int, Any] = {}
    for c_any in candidates:
        c: Any = c_any
        aid: int = _action_id(c)
        if aid not in seen:
            seen[aid] = c
    # sort by action_id for determinism (lexicographic)
    ordered = tuple(seen[k] for k in sorted(seen))
    # verify immutability: callers must not mutate returned tuple (frozen)
    return ordered


def fixed_allocate(
    children: dict[tuple[int, str], tuple[ChildEntry, ...] | list[ChildEntry]],
    *,
    total_batches: int,
) -> dict[tuple[int, str], int]:
    """Allocate fixed search batches deterministically across children.

    Wave 2 bridge audit: kept Python — frozen schedule allocation is not a
    sequential-halving cut (no survivors/means/gumbels); no pyfn covers it.
    Deterministic: sorted keys get base batches, remainder distributed by hash order.
    The schedule is frozen before search; outcome-derived reallocation is prohibited.
    """
    if not isinstance(total_batches, int) or isinstance(total_batches, bool) or total_batches <= 0:
        raise ContractError("total_batches must be positive int")
    if len(children) == 0:
        return {}
    keys = sorted(children.keys(), key=lambda k: (k[0], k[1]))
    n = len(keys)
    base = total_batches // n
    rem = total_batches % n
    # hash order for remainder (deterministic via canonical bytes)
    hash_order = sorted(
        keys,
        key=lambda k: hashlib.sha256(canonical_bytes({"aid": k[0], "pid": k[1]})).hexdigest(),
    )
    rem_set = set(hash_order[:rem])
    allocations: dict[tuple[int, str], int] = {}
    for k in keys:
        allocations[k] = base + (1 if k in rem_set else 0)
    # verify sum equals total
    assert sum(allocations.values()) == total_batches
    return allocations


def _z_hat_for_key(
    entries: tuple[ChildEntry, ...],
) -> float:
    return sum(e.raw_weight for e in entries)


def _normalized_weights(entries: tuple[ChildEntry, ...]) -> tuple[float, ...] | None:
    z = _z_hat_for_key(entries)
    if z <= 0 or not math.isfinite(z):
        return None  # ESS diagnostic only; normalized only when normalizer > 0 per spec
    return tuple(e.raw_weight / z for e in entries)


def _ess_for_key(entries: tuple[ChildEntry, ...]) -> float | None:
    norm = _normalized_weights(entries)
    if norm is None:
        return None
    s = sum(w * w for w in norm)
    if s <= 0:
        return None
    return 1.0 / s
