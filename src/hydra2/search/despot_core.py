"""Candidate 2 natural DESPOT — scenarios, packet guards, seeding, spec factory.

Owns the natural-scenario vocabulary (uniform ``1/K`` weight, equal
target/proposal densities), the frozen DESPOT hyper-parameters, the packet
partition guards with the proposal-reversal negative control, the
semantic-seed derivation over ``(candidate_id, case_id, scenario_idx,
attempt_id)``, and the candidate-spec factory with its default budget.
The search policy lives in :mod:`hydra2.search.despot_search`, the
expansion loop in :mod:`hydra2.search.despot_act`, and result assembly in
:mod:`hydra2.search.despot_result`.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import (
    ContractError,
    PacketPartitionError,
)

logger = logging.getLogger(__name__)

try:
    from hydra2.contracts.randomness import RandomStream

    _HAS_RANDOM = True
except ImportError:  # pragma: no cover
    _HAS_RANDOM = False
    RandomStream = Any

try:
    from hydra2.belief.kernel import NaturalPacketKernel
    from hydra2.belief.natural import BeliefEpoch, NaturalBelief

    _HAS_BELIEF = True
except ImportError:  # pragma: no cover
    _HAS_BELIEF = False
    NaturalBelief = Any
    BeliefEpoch = Any
    NaturalPacketKernel = Any

try:
    from hydra2.eval.telemetry import ResourceTelemetry, make_resource_telemetry

    _HAS_TELEMETRY = True
except ImportError:  # pragma: no cover
    _HAS_TELEMETRY = False
    ResourceTelemetry = Any
    make_resource_telemetry = Any

try:
    from hydra2.contracts.utility import UtilityVector

    _HAS_UTILITY = True
except ImportError:  # pragma: no cover
    _HAS_UTILITY = False
    UtilityVector = Any

__all__ = [
    "_MASTER_SEED",
    "DespotConfig",
    "NaturalScenario",
    "_DespotNode",
    "_default_budget",
    "_hash_tie_break",
    "_scenario_seed_bytes",
    "make_despot_candidate_spec",
    "packet_aliasing_rejected",
    "proposal_reversal_fixture",
    "validate_packet_partition",
]


# ---------------------------------------------------------------------------
# Shared search contract lives in common.py, which is authoritative.
# The fallback below stays for offline/unit testability without the full
# contract stack.
# ---------------------------------------------------------------------------

try:  # shared contracts live in common.py
    from hydra2.search.common import (
        CandidateSpec,
        Planner,
        ResourceBudget,
        SearchRequest,
        SearchResult,
    )

    _COMMON_AVAILABLE = True
except ImportError:  # fallback minimal contracts compatible with SPEC 15
    _COMMON_AVAILABLE = False

    @dataclass(frozen=True, slots=True)
    class ResourceBudget:
        mode: Literal["gameplay_5s", "ponder", "analysis"] = "gameplay_5s"
        deadline_ms: int = 5000
        fallback_margin_ms: int = 200
        max_model_calls: int | None = 32
        max_transitions: int | None = 128
        max_particles: int | None = 64
        max_memory_bytes: int | None = None

        def __post_init__(self) -> None:
            if self.deadline_ms <= 0:
                raise ValueError("deadline_ms must be positive")
            if self.fallback_margin_ms < 0 or self.fallback_margin_ms >= self.deadline_ms:
                raise ValueError("fallback_margin_ms must be in [0, deadline_ms)")
            for name in ("max_model_calls", "max_transitions", "max_particles"):
                v = getattr(self, name)
                if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                    raise ValueError(f"{name} must be positive int or None")

    @dataclass(frozen=True, slots=True)
    class CandidateSpec:
        candidate_id: str
        algorithm: str = "despot_natural"
        algorithm_version: str = "1.0.0"
        rules_hash: str = "sha256:" + "a" * 64
        # dummy-until-real: pilot default, replaced by _canonical_hashes/caller before commit.
        utility_id: str = "expected_final_placement"
        utility_manifest_hash: str = "sha256:" + "b" * 64
        action_table_hash: str = "sha256:" + "c" * 64
        observation_schema_hash: str = "sha256:" + "d" * 64
        packet_boundary_hash: str = "sha256:" + "e" * 64
        model_hash: str = "sha256:" + "f" * 64
        belief_model_hash: str | None = None
        event_model_hash: str | None = None
        continuation_policy_hashes: tuple[str, ...] = ()
        proposal_spec_hash: str | None = None
        case_manifest_hash: str = "sha256:" + "0" * 64
        resource_budget: ResourceBudget = field(default_factory=ResourceBudget)
        fallback_candidate_id: Literal["candidate0"] = "candidate0"
        tie_break: str = "lexicographic"
        rng_protocol_hash: str = "sha256:" + "1" * 64
        random_stream_schema_hash: str = "sha256:" + "2" * 64
        parameters: dict[str, Any] = field(default_factory=dict)

    @dataclass(frozen=True, slots=True)
    class SearchRequest:
        observation: Any
        legal_actions: tuple[Any, ...]
        candidate_spec: CandidateSpec
        deadline_monotonic_ns: int | None = None
        belief_epoch: Any | None = None
        case_id: str | None = None
        root_seat: int | None = None

    @dataclass(frozen=True, slots=True)
    class SearchResult:
        selected_action: Any
        candidate_actions: tuple[Any, ...]
        value_vectors: tuple[Any, ...]
        candidate_spec_hash: str
        telemetry: Any
        evidence_refs: tuple[str, ...]
        completed: bool

    class Planner:
        def act(self, request: SearchRequest) -> SearchResult:  # pragma: no cover
            raise NotImplementedError

        def observe(self, packet: Any) -> None:  # pragma: no cover
            pass

        def ponder(self, *, deadline_monotonic_ns: int) -> None:  # pragma: no cover
            pass

# ---------------------------------------------------------------------------
# Scenario — natural (world, semantic randomness)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NaturalScenario:
    """One natural scenario: ``(world_ref, semantic_rng)``.

    ``weight`` is uniform ``1/K`` (natural), never proposal-weighted.
    ``log_target_density`` and ``log_proposal_density`` are kept equal for
    natural sampling (ratio one); proposal variants MUST NOT reuse this type.
    """

    scenario_id: int
    world_ref: str
    # semantic seed bytes derived deterministically from (case_id, candidate_id, scenario_id)
    semantic_seed_bytes: bytes
    log_target_density: float
    log_proposal_density: float
    weight: float  # 1/K

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scenario_id, int)
            or isinstance(self.scenario_id, bool)
            or self.scenario_id < 0
        ):
            raise ContractError("scenario_id must be nonnegative int")
        if not isinstance(self.world_ref, str) or self.world_ref == "":
            raise ContractError("world_ref must be non-empty str")
        if not isinstance(self.semantic_seed_bytes, bytes) or len(self.semantic_seed_bytes) != 32:
            raise ContractError("semantic_seed_bytes must be 32 bytes")
        if not math.isfinite(self.log_target_density) or not math.isfinite(
            self.log_proposal_density
        ):
            raise ContractError("log densities must be finite")
        if self.log_target_density != self.log_proposal_density:
            raise ContractError("natural scenario requires log_target == log_proposal (ratio one)")
        if not math.isfinite(self.weight) or self.weight <= 0 or self.weight > 1:
            raise ContractError("weight must be finite in (0,1]")


@dataclass(frozen=True, slots=True)
class DespotConfig:
    """Frozen DESPOT hyper-parameters (part of CandidateSpec.parameters)."""

    num_scenarios: int = 16
    regularization: float | None = None  # None = no regularization; otherwise heuristic
    max_depth: int = 4
    tie_break: str = "lexicographic"
    # budget view: which dimension is the declared comparison view
    resource_view: Literal["calls", "transitions", "joules"] = "calls"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.num_scenarios, int)
            or isinstance(self.num_scenarios, bool)
            or self.num_scenarios <= 0
        ):
            raise ContractError("num_scenarios must be positive int")
        if self.max_depth <= 0:
            raise ContractError("max_depth must be positive")
        if self.tie_break not in ("lexicographic", "stable_hash"):
            raise ContractError("tie_break must be lexicographic or stable_hash")
        if self.regularization is not None and (
            not isinstance(self.regularization, float) or not math.isfinite(self.regularization)
        ):
            raise ContractError("regularization must be finite float or None")


# ---------------------------------------------------------------------------
# Helpers — packet partition, proposal-reversal, aliasing
# ---------------------------------------------------------------------------


def validate_packet_partition(successors: Any, *, tolerance: float = 1e-9) -> None:
    """Validate that packet successors form a disjoint exhaustive partition.

    Checks:
    - pairwise packet_id distinct (aliasing rejected),
    - probabilities finite, nonnegative, sum to 1 within tolerance,
    - each successor carries distinct packet identity (no aliasing).
    """
    if successors is None:
        raise PacketPartitionError("successors must be non-empty [PBRF_PARTITION_EMPTY]")
    try:
        if len(successors) == 0:  # type: ignore[arg-type]
            raise PacketPartitionError("successors must be non-empty [PBRF_PARTITION_EMPTY]")
    except TypeError:
        pass
    pids: list[str] = []
    total = 0.0
    for s in successors:
        pid = getattr(getattr(s, "packet", None), "packet_id", None)
        if pid is None:
            # fallback: s itself may be packet_id string
            pid_raw: Any = getattr(s, "packet_id", None)
            pid = pid_raw if isinstance(pid_raw, str) and pid_raw != "" else str(s)
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


def packet_aliasing_rejected(successors: Any) -> bool:
    """Return True iff successors would be rejected for aliasing/mass error."""
    try:
        validate_packet_partition(successors)
    except (PacketPartitionError, ContractError):
        return True
    return False


def proposal_reversal_fixture() -> dict[str, Any]:
    """Tiny fixture proving unweighted non-natural scenarios can flip the action.

    Constructs a 2-world, 2-action case where:
    - natural law is uniform (0.5, 0.5),
    - true value favors action 0,
    - proposal law is biased 0.9/0.1 toward world 1 (which favors action 1),
    - unweighted proposal mean chooses the *wrong* action, while correctly
      weighted (b/q) or natural mean chooses correctly.

    The fixture is used as a negative control: calling an arbitrary weighted
    number an upper bound is prohibited; this demonstrates that proposal bias
    without correction reverses decisions.
    """
    # world 0: values {a0: 0.8, a1: 0.2}, world 1: {a0: 0.1, a1: 0.9}
    # natural expected: a0=0.45, a1=0.55? Actually to make a0 correct, swap:
    # Let's set world0 a0=1.0 a1=0.0, world1 a0=0.0 a1=1.0 but with proposal bias toward world1,
    # natural uniform gives tie 0.5 vs 0.5. Need bias to flip.
    # Instead make values: world0 a0=0.9 a1=0.0, world1 a0=0.0 a1=0.6
    # natural: a0=0.45, a1=0.30 -> a0 wins.
    # proposal biased 0.9 to world1: unweighted a0≈0.09 a1≈0.54 -> a1 wins (reversal)
    # weighted correction restores natural.

    natural_probs = (0.5, 0.5)
    proposal_probs = (0.1, 0.9)  # heavily favors world 1
    values = ({0: 0.9, 1: 0.0}, {0: 0.0, 1: 0.6})

    def expected(probs: tuple[float, ...], vals: tuple[dict[int, float], ...]) -> dict[int, float]:
        acc: dict[int, float] = {0: 0.0, 1: 0.0}
        for p, v in zip(probs, vals, strict=False):
            for a in acc:
                acc[a] = acc[a] + p * v[a]
        return acc

    natural_mean: dict[int, float] = expected(natural_probs, values)
    proposal_unweighted: dict[int, float] = expected(proposal_probs, values)
    # weighted correction: each proposal sample weighted by b/q
    weighted: dict[int, float] = {0: 0.0, 1: 0.0}
    for i, (pb, qb) in enumerate(zip(natural_probs, proposal_probs, strict=False)):
        w: float = pb / qb if qb > 0 else 0.0
        # expected contribution of proposal-weighted estimator = sum_q [w * v * q] = sum_b v
        # but for fixture illustration we compute reweighted mean
        for a in weighted:
            weighted[a] = weighted[a] + proposal_probs[i] * w * values[i][a]

    # natural chooses a0 (0.45 > 0.30)
    def _max_key_mean(d: dict[int, float], k: int) -> float:
        return d[k]

    natural_choice: int = max(natural_mean, key=lambda k: natural_mean[cast("int", k)])
    proposal_choice: int = max(
        proposal_unweighted, key=lambda k: proposal_unweighted[cast("int", k)]
    )
    weighted_choice: int = max(weighted, key=lambda k: weighted[cast("int", k)])
    return {
        "natural_mean": natural_mean,
        "proposal_unweighted_mean": proposal_unweighted,
        "proposal_weighted_mean": weighted,
        "natural_choice": natural_choice,
        "proposal_unweighted_choice": proposal_choice,
        "proposal_weighted_choice": weighted_choice,
        "reversal": natural_choice != proposal_choice,
        "correction_restores": natural_choice == weighted_choice,
        "note": "unweighted non-natural reverses; weighted restores — proves proposal bias without correction is unsafe",
    }


# ---------------------------------------------------------------------------
# Deterministic scenario seeding
# ---------------------------------------------------------------------------

_MASTER_SEED = b"wp08c_despot_natural_v1"


def _scenario_seed_bytes(
    *, candidate_id: str, case_id: str, scenario_idx: int, attempt_id: int = 0
) -> bytes:
    """Deterministic 32-byte seed for one scenario.

    Uses semantic-seed derivation when available; otherwise falls back to
    SHA-256 over a canonical payload. Determinism is over (candidate_id,
    case_id, scenario_idx, attempt_id) — never call order.
    """
    payload = canonical_bytes(
        {
            "candidate_id": candidate_id,
            "case_id": case_id,
            "scenario_idx": scenario_idx,
            "attempt_id": attempt_id,
            "master": _MASTER_SEED.hex(),
        }
    )
    return hashlib.sha256(payload).digest()


def _hash_tie_break(actions: tuple[Any, ...], candidate_id: str) -> Any:
    """Stable lexicographic tie break via hash of candidate_id + action id."""

    def aid(a: Any) -> int:
        v = getattr(a, "action_id", None)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        if isinstance(a, int) and not isinstance(a, bool):
            return a
        return hash(str(a)) & 0xFFFFFFFF

    best = None
    best_key = None
    for a in actions:
        # use hash of (candidate_id, aid) to be deterministic but not call-order dependent
        h = hashlib.sha256(f"{candidate_id}:{aid(a)}".encode()).hexdigest()
        if best_key is None or h < best_key:
            best_key = h
            best = a
    return best


# ---------------------------------------------------------------------------
# Planner node
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _DespotNode:
    """One DESPOT belief-action node (actor-visible)."""

    node_id: str
    depth: int
    lower_value: float  # feasible policy estimate at this node (not a bound)
    priority_proxy: float  # heuristic search priority; explicitly NOT an upper bound
    visits: int = 0
    children: dict[Any, _DespotNode] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Factory helper for CandidateSpec binding
# ---------------------------------------------------------------------------


def _default_budget() -> Any:
    """Default gameplay_5s budget for tests (uses common contract when available)."""
    if _COMMON_AVAILABLE:
        return ResourceBudget(
            mode="gameplay_5s",
            deadline_ms=5000,
            fallback_margin_ms=200,
            max_model_calls=64,
            max_transitions=256,
            max_particles=16,
            max_memory_bytes=None,
        )
    # fallback local
    return ResourceBudget(
        mode="gameplay_5s",
        deadline_ms=5000,
        fallback_margin_ms=200,
        max_model_calls=64,
        max_transitions=256,
        max_particles=16,
        max_memory_bytes=None,
    )


def make_despot_candidate_spec(
    *,
    candidate_id: str = "candidate2_despot_natural",
    num_scenarios: int = 16,
    regularization: float | None = None,
    max_depth: int = 4,
    resource_budget: Any | None = None,
    # dummy-until-real: pilot default, replaced by _canonical_hashes/caller before commit.
    rules_hash: str = "sha256:" + "a" * 64,
) -> Any:
    """Build a frozen CandidateSpec for natural DESPOT (test helper)."""
    if resource_budget is None:
        resource_budget = _default_budget()
    params = {
        "num_scenarios": num_scenarios,
        "max_depth": max_depth,
        "regularization": regularization,
        "tie_break": "lexicographic",
        "resource_view": "calls",
    }
    try:
        return CandidateSpec(
            candidate_id=candidate_id,
            algorithm="despot_natural",
            algorithm_version="1.0.0",
            rules_hash=rules_hash,
            utility_id="expected_final_placement",
            utility_manifest_hash="sha256:" + "b" * 64,
            action_table_hash="sha256:" + "c" * 64,
            observation_schema_hash="sha256:" + "d" * 64,
            packet_boundary_hash="sha256:" + "e" * 64,
            model_hash="sha256:" + "f" * 64,
            belief_model_hash=None,
            event_model_hash=None,
            continuation_policy_hashes=(),
            proposal_spec_hash=None,
            case_manifest_hash="sha256:" + "0" * 64,
            resource_budget=resource_budget,
            fallback_candidate_id="candidate0",
            tie_break="lexicographic",
            rng_protocol_hash="sha256:" + "1" * 64,
            random_stream_schema_hash="sha256:" + "2" * 64,
            parameters=params,
        )
    except (AttributeError, ValueError, TypeError) as exc:
        logger.debug("despot: CandidateSpec fallback minimal", exc_info=exc)
        # WHY retained: offline/unit tests can build a minimal spec when the
        # shared contract stack is unavailable; common.py stays authoritative.
        # Fallback for when common is unavailable (uses local minimal CandidateSpec)
        return CandidateSpec(  # type: ignore[missing-argument]  # pyrefly: ignore[missing-argument]
            candidate_id=candidate_id, parameters=params, resource_budget=resource_budget
        )  # type: ignore[call-arg]  # pyrefly: ignore[missing-argument]
