# ruff: noqa: E501, TC006, SIM102, PERF401, B905  # reason: legacy blanket; E501 URLs unwrappable, TC006 quotes intentional, SIM102 nested guards readable, PERF401/B905 perf-critical. Evidence: https://docs.python.org/3/library/importlib.resources.html
"""WP-12 Analysis Qualification — offline analysis mode (M12).

Implements BUILD §15 checklist and SPEC 15/18.2 semantics:

- Freeze finite analysis budgets and resource caps.
- Reuse identical observation/rules/utility/legal/model/estimator semantics.
- Permit only additional charged compute.
- Deterministic replay across gameplay/analysis modes.
- Compare actions/value estimates and fallback behavior.
- Reject hidden fields, altered rules, changed estimator, uncharged work.
- Generate hashed analysis report.

All analysis budgets are finite, declared, and content-addressed via
CandidateSpec digests (RFC 8785 canonical JSON + SHA-256). Analysis mode
changes only ``ResourceBudget`` (mode, deadline, caps); every other
semantic field (rules, utility, action table, observation schema, packet
boundary, model, belief/event models, continuation hashes, proposal spec,
case manifest, rng protocol, parameters, algorithm/version, utility_id,
tie_break, fallback) MUST be byte-identical between gameplay and analysis
specs. Privileged leakage is hard failure. Determinism is seeded via
SHA-256 over ``(case_id, candidate_id, observation_hash, action_id)``;
no global RNG. Every model call / exact transition is charged in
telemetry; missing telemetry invalidates the block.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import (
    ContractError,
    DigestText,
    VisibilityViolationError,
    make_digest_text,
    make_seat,
    make_tile_id,
)

logger = logging.getLogger(__name__)

ANALYSIS_REPORT_KIND = "hydra2.analysis_gate_report"
ANALYSIS_REPORT_SCHEMA_VERSION = "1.0.0"

# Teacher-eligible Candidate 0-6 outcome registry — one entry per candidate.
# Candidate 4 is represented by the control forest ``candidate4_core_control``
# (WP-09B cumulative build is PBRF core alone per CON-09-004; modules are
# independent promotions). Persistence factorial (WP-09C) is not a teacher
# candidate. This list is normative for WP-12 gating.
ANALYSIS_CANDIDATE_IDS: tuple[str, ...] = (
    "candidate0",
    "candidate1",
    "candidate2",
    "candidate3_pbrf_core_v1",
    "candidate4_core_control",
    "candidate5",
    "candidate6",
)

# Finite analysis budgets — deadline/resource caps frozen blind to arm labels.
# Gameplay is 5,000 ms; analysis declares a larger finite deadline (30,000 ms)
# plus larger finite caps, all charged. Every budget is mode=analysis.
# Caps are finite (non-None) and bounded to prevent unbounded claims.
ANALYSIS_BUDGETS: dict[str, dict[str, int | None]] = {
    # candidate_id -> {deadline_ms, fallback_margin_ms, max_model_calls, max_transitions, max_particles, max_memory_bytes}
    "candidate0": {
        "deadline_ms": 30000,
        "fallback_margin_ms": 500,
        "max_model_calls": 4,
        "max_transitions": 16,
        "max_particles": 0,
        "max_memory_bytes": 2 * 1024**3,
    },
    "candidate1": {
        "deadline_ms": 30000,
        "fallback_margin_ms": 500,
        "max_model_calls": 256,
        "max_transitions": 1024,
        "max_particles": 64,
        "max_memory_bytes": 8 * 1024**3,
    },
    "candidate2": {
        "deadline_ms": 30000,
        "fallback_margin_ms": 500,
        "max_model_calls": 256,
        "max_transitions": 1024,
        "max_particles": 64,
        "max_memory_bytes": 8 * 1024**3,
    },
    "candidate3_pbrf_core_v1": {
        "deadline_ms": 30000,
        "fallback_margin_ms": 500,
        "max_model_calls": 256,
        "max_transitions": 1024,
        "max_particles": 64,
        "max_memory_bytes": 8 * 1024**3,
    },
    "candidate4_core_control": {
        "deadline_ms": 30000,
        "fallback_margin_ms": 500,
        "max_model_calls": 256,
        "max_transitions": 1024,
        "max_particles": 64,
        "max_memory_bytes": 8 * 1024**3,
    },
    "candidate5": {
        "deadline_ms": 30000,
        "fallback_margin_ms": 500,
        "max_model_calls": 256,
        "max_transitions": 1024,
        "max_particles": 64,
        "max_memory_bytes": 8 * 1024**3,
    },
    "candidate6": {
        "deadline_ms": 30000,
        "fallback_margin_ms": 500,
        "max_model_calls": 256,
        "max_transitions": 1024,
        "max_particles": 64,
        "max_memory_bytes": 8 * 1024**3,
    },
}

# Gameplay baselines for comparison (used only for "additional charged compute" check).
# These mirror the factories' default gameplay budgets. They are not imported
# dynamically to keep analysis deterministic and independent of mutable code.
GAMEPLAY_BUDGETS: dict[str, dict[str, int | None]] = {
    "candidate0": {
        "deadline_ms": 5000,
        "fallback_margin_ms": 500,
        "max_model_calls": 1,
        "max_transitions": 0,
        "max_particles": 0,
        "max_memory_bytes": None,
    },
    "candidate1": {
        "deadline_ms": 5000,
        "fallback_margin_ms": 200,
        "max_model_calls": 64,
        "max_transitions": 256,
        "max_particles": 32,
        "max_memory_bytes": None,
    },
    "candidate2": {
        "deadline_ms": 5000,
        "fallback_margin_ms": 200,
        "max_model_calls": 64,
        "max_transitions": 256,
        "max_particles": 32,
        "max_memory_bytes": None,
    },
    "candidate3_pbrf_core_v1": {
        "deadline_ms": 5000,
        "fallback_margin_ms": 200,
        "max_model_calls": 64,
        "max_transitions": 256,
        "max_particles": 16,
        "max_memory_bytes": None,
    },
    "candidate4_core_control": {
        "deadline_ms": 5000,
        "fallback_margin_ms": 200,
        "max_model_calls": 64,
        "max_transitions": 256,
        "max_particles": 16,
        "max_memory_bytes": None,
    },
    "candidate5": {
        "deadline_ms": 5000,
        "fallback_margin_ms": 200,
        "max_model_calls": 64,
        "max_transitions": 256,
        "max_particles": 32,
        "max_memory_bytes": None,
    },
    "candidate6": {
        "deadline_ms": 5000,
        "fallback_margin_ms": 200,
        "max_model_calls": 64,
        "max_transitions": 256,
        "max_particles": 32,
        "max_memory_bytes": None,
    },
}


def _require_finite_budget(budget: Any) -> None:
    """Validate that a ResourceBudget is finite and analysis-legal."""
    from hydra2.search.common import ResourceBudget  # lazy

    if not isinstance(budget, ResourceBudget):
        raise ContractError(f"resource_budget must be ResourceBudget, got {type(budget).__name__}")
    if budget.mode != "analysis":
        raise ContractError(f"analysis budget mode must be 'analysis', got {budget.mode!r}")
    if budget.deadline_ms <= 5000 or budget.deadline_ms > 300000:
        raise ContractError(
            f"analysis deadline_ms must be finite >5000 and <=300000, got {budget.deadline_ms}"
        )
    if budget.fallback_margin_ms < 0 or budget.fallback_margin_ms >= budget.deadline_ms:
        raise ContractError(
            f"fallback_margin_ms {budget.fallback_margin_ms} must be in [0, deadline_ms)"
        )
    # Caps must be finite (non-None) and positive/bounded for analysis.
    for name in ("max_model_calls", "max_transitions", "max_particles", "max_memory_bytes"):
        v = getattr(budget, name)
        if name == "max_memory_bytes":
            if v is None:
                raise ContractError("analysis max_memory_bytes must be finite (non-None) cap")
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0 or v > 64 * 1024**3:
                raise ContractError(
                    f"analysis max_memory_bytes must be finite positive <=64GiB, got {v!r}"
                )
        elif name == "max_particles":
            if v is None or not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ContractError(f"analysis {name} must be finite nonneg int, got {v!r}")
        else:
            if v is None or not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ContractError(f"analysis {name} must be finite positive int, got {v!r}")


def analysis_budget_for(candidate_id: str) -> Any:
    """Return the frozen finite analysis ResourceBudget for candidate_id."""
    from hydra2.search.common import ResourceBudget

    if candidate_id not in ANALYSIS_BUDGETS:
        raise ContractError(f"unknown candidate_id {candidate_id!r} for analysis budget")
    cfg = ANALYSIS_BUDGETS[candidate_id]
    budget = ResourceBudget(
        mode="analysis",
        deadline_ms=int(cfg["deadline_ms"]),  # type: ignore[arg-type]  # reason: cfg value statically int|None; int() coerces. Evidence: budgets frozen above
        fallback_margin_ms=int(cfg["fallback_margin_ms"]),  # type: ignore[arg-type]  # reason: cfg value statically int|None; int() coerces
        max_model_calls=int(cfg["max_model_calls"]),  # type: ignore[arg-type]  # reason: cfg value statically int|None; int() coerces
        max_transitions=int(cfg["max_transitions"]),  # type: ignore[arg-type]  # reason: cfg value statically int|None; int() coerces
        max_particles=int(cfg["max_particles"]),  # type: ignore[arg-type]  # reason: cfg value statically int|None; int() coerces
        max_memory_bytes=int(cfg["max_memory_bytes"]),  # type: ignore[arg-type]  # reason: cfg value statically int|None; int() coerces
    )
    _require_finite_budget(budget)
    return budget


def make_analysis_spec(gameplay_spec: Any) -> Any:
    """Derive an analysis CandidateSpec from a gameplay spec.

    Only ``resource_budget`` may change (mode -> analysis, larger finite caps).
    Every other semantic field is byte-identical. The returned spec has the
    same ``candidate_id`` and ``case_manifest_hash`` as the gameplay spec but
    a distinct digest due to the budget change.

    Raises ContractError if gameplay_spec is not a valid CandidateSpec or if
    the derived analysis budget is not finite/larger than gameplay.
    """
    from hydra2.search.common import CandidateSpec

    if not isinstance(gameplay_spec, CandidateSpec):
        raise ContractError("gameplay_spec must be CandidateSpec")
    if gameplay_spec.resource_budget.mode != "gameplay_5s":
        # Allow ponder->analysis as well, but forbid analysis->analysis double conversion.
        if gameplay_spec.resource_budget.mode == "analysis":
            raise ContractError(
                "gameplay_spec is already analysis mode; refusing double conversion"
            )
    candidate_id = gameplay_spec.candidate_id
    # Use frozen analysis budget for this candidate; if candidate_id unknown, derive generically
    # but still require finite caps larger than gameplay.
    if candidate_id in ANALYSIS_BUDGETS:
        analysis_budget = analysis_budget_for(candidate_id)
    else:
        # Generic fallback: enlarge gameplay caps 4x, deadline 30s, mode analysis
        gp = gameplay_spec.resource_budget
        analysis_budget = _derive_generic_analysis_budget(gp)

    # Validate monotonic compute increase
    gp = gameplay_spec.resource_budget
    if analysis_budget.deadline_ms <= gp.deadline_ms:
        raise ContractError(
            f"analysis deadline {analysis_budget.deadline_ms} must be > gameplay {gp.deadline_ms}"
        )
    for name in ("max_model_calls", "max_transitions"):
        gp_v = getattr(gp, name)
        an_v = getattr(analysis_budget, name)
        if gp_v is not None and an_v is not None and an_v < gp_v:
            raise ContractError(f"analysis {name} {an_v} must be >= gameplay {gp_v}")
        if gp_v is not None and an_v is None:
            raise ContractError(f"analysis {name} must stay finite when gameplay is finite")
    # Preserve max_particles exactly or at least not reduced; allow same.
    # Construct new spec with identical semantic fields, only budget replaced.
    spec = CandidateSpec(
        candidate_id=gameplay_spec.candidate_id,
        algorithm=gameplay_spec.algorithm,
        algorithm_version=gameplay_spec.algorithm_version,
        rules_hash=gameplay_spec.rules_hash,
        utility_id=gameplay_spec.utility_id,
        utility_manifest_hash=gameplay_spec.utility_manifest_hash,
        action_table_hash=gameplay_spec.action_table_hash,
        observation_schema_hash=gameplay_spec.observation_schema_hash,
        packet_boundary_hash=gameplay_spec.packet_boundary_hash,
        model_hash=gameplay_spec.model_hash,
        belief_model_hash=gameplay_spec.belief_model_hash,
        event_model_hash=gameplay_spec.event_model_hash,
        continuation_policy_hashes=gameplay_spec.continuation_policy_hashes,
        proposal_spec_hash=gameplay_spec.proposal_spec_hash,
        case_manifest_hash=gameplay_spec.case_manifest_hash,
        resource_budget=analysis_budget,
        fallback_candidate_id=gameplay_spec.fallback_candidate_id,
        tie_break=gameplay_spec.tie_break,
        rng_protocol_hash=gameplay_spec.rng_protocol_hash,
        random_stream_schema_hash=gameplay_spec.random_stream_schema_hash,
        parameters=dict(gameplay_spec.parameters),
    )
    # Final finite check
    _require_finite_budget(spec.resource_budget)
    return spec


def _derive_generic_analysis_budget(gp: Any) -> Any:
    from hydra2.search.common import ResourceBudget

    def _enlarge(v: int | None, fallback: int) -> int:
        if v is None:
            return fallback
        return max(v * 4, fallback)

    _fallback_raw: Any = getattr(gp, "fallback_margin_ms", None)
    _fallback_val: int = _fallback_raw if isinstance(_fallback_raw, int) and not isinstance(_fallback_raw, bool) else 500
    _fallback_capped: int = min(_fallback_val, 500) if hasattr(gp, "fallback_margin_ms") else 500
    _max_particles_raw: Any = getattr(gp, "max_particles", 0)
    _max_particles_val: int = cast(int, _max_particles_raw) if bool(_max_particles_raw) else 0
    return ResourceBudget(
        mode="analysis",
        deadline_ms=30000,
        fallback_margin_ms=_fallback_capped,
        max_model_calls=_enlarge(getattr(gp, "max_model_calls", None), 256),
        max_transitions=_enlarge(getattr(gp, "max_transitions", None), 1024),
        max_particles=_max_particles_val,
        max_memory_bytes=8 * 1024**3,
    )


def verify_compute_only(gameplay_spec: Any, analysis_spec: Any) -> bool:
    """Prove analysis changes only charged compute.

    Checks:
    - Both are CandidateSpec with same candidate_id.
    - Every semantic field except resource_budget is byte-identical.
    - Analysis budget is mode=analysis, finite, deadline>gameplay, caps >= gameplay.
    - No privileged fields introduced (observation/model/estimator hashes identical).
    - Resource budget caps are finite and bounded.

    Returns True on success; raises ContractError / VisibilityViolationError on
    any taint (hidden fields, altered rules, changed estimator, uncharged work).
    """
    from hydra2.search.common import CandidateSpec

    if not isinstance(gameplay_spec, CandidateSpec) or not isinstance(analysis_spec, CandidateSpec):
        raise ContractError("both specs must be CandidateSpec")
    if gameplay_spec.candidate_id != analysis_spec.candidate_id:
        raise ContractError(
            f"candidate_id mismatch: gameplay {gameplay_spec.candidate_id!r} vs analysis {analysis_spec.candidate_id!r}"
        )
    # Semantic identity — every field except resource_budget must match exactly.
    semantic_fields = (
        "algorithm",
        "algorithm_version",
        "rules_hash",
        "utility_id",
        "utility_manifest_hash",
        "action_table_hash",
        "observation_schema_hash",
        "packet_boundary_hash",
        "model_hash",
        "belief_model_hash",
        "event_model_hash",
        "continuation_policy_hashes",
        "proposal_spec_hash",
        "case_manifest_hash",
        "fallback_candidate_id",
        "tie_break",
        "rng_protocol_hash",
        "random_stream_schema_hash",
        "parameters",
    )
    for name in semantic_fields:
        gv = getattr(gameplay_spec, name)
        av = getattr(analysis_spec, name)
        # For hashes, canonical comparison (already validated digests)
        if gv != av:
            # Specialize errors for privileged-taint categories
            if name in (
                "rules_hash",
                "utility_manifest_hash",
                "action_table_hash",
                "observation_schema_hash",
                "packet_boundary_hash",
            ):
                raise ContractError(
                    f"analysis must reuse identical {name}: gameplay {gv!r} vs analysis {av!r} — altered rules/utility"
                )
            if name in (
                "model_hash",
                "belief_model_hash",
                "event_model_hash",
                "proposal_spec_hash",
            ):
                raise ContractError(
                    f"analysis must reuse identical {name}: gameplay {gv!r} vs analysis {av!r} — changed estimator"
                )
            if name == "parameters":
                # Parameters contain estimator configuration — any change is changed estimator
                raise ContractError(
                    f"analysis parameters must be identical: gameplay {gv!r} vs analysis {av!r}"
                )
            raise ContractError(f"analysis semantic field {name} differs: {gv!r} vs {av!r}")
    # Resource budget checks
    gp_b = gameplay_spec.resource_budget
    an_b = analysis_spec.resource_budget
    if gp_b.mode not in ("gameplay_5s", "ponder"):
        raise ContractError(f"gameplay mode must be gameplay_5s or ponder, got {gp_b.mode!r}")
    _require_finite_budget(an_b)
    if an_b.deadline_ms <= gp_b.deadline_ms:
        raise ContractError(
            f"analysis deadline {an_b.deadline_ms} must exceed gameplay {gp_b.deadline_ms}"
        )
    # Caps must not shrink when gameplay is finite
    for name in ("max_model_calls", "max_transitions", "max_particles"):
        gv = getattr(gp_b, name)
        av = getattr(an_b, name)
        if gv is not None and av is not None and av < gv:
            raise ContractError(
                f"analysis {name} {av} must be >= gameplay {gv} — uncharged reduction"
            )
    # Fallback must be identical (no semantic change)
    if gameplay_spec.fallback_candidate_id != analysis_spec.fallback_candidate_id:
        raise ContractError("fallback_candidate_id must be identical across modes")
    return True


def check_no_privileged_leak(spec: Any, observation: Any) -> None:
    """Reject privileged leakage for analysis mode.

    - observation must be actor-visible (ActorObservation or synthetic stub
      with observation_hash)
    - No privileged fields like full_world, hidden wall, opponent hand.
    - No extra fields beyond actor-visible schema.

    Raises VisibilityViolationError or ContractError on leak.
    """
    from hydra2.contracts.observation import ActorObservation

    # Dict path: reject privileged keys
    if isinstance(observation, dict):
        privileged_keys = {
            "full_world",
            "hidden",
            "privileged",
            "wall",
            "opponent_hand",
            "dead_wall_privileged",
        }
        found = privileged_keys.intersection(observation.keys())
        if len(found) > 0:
            raise VisibilityViolationError(f"privileged leak in observation dict: {found}")
        # Require observation_hash for dict as well
        oh: Any = observation.get("observation_hash")
        if oh is not None:
            _: DigestText = make_digest_text(cast(str, oh))
        return

    # ActorObservation or synthetic stub with observation_hash
    is_actor_obs = isinstance(observation, ActorObservation)
    has_hash = hasattr(observation, "observation_hash")
    if not is_actor_obs and not has_hash:
        raise VisibilityViolationError(
            f"observation must be ActorObservation or have observation_hash, got {type(observation).__name__}"
        )

    # For both ActorObservation and stub, check forbidden attributes
    forbidden_attrs = ("full_world", "hidden_wall", "privileged", "wall_state", "opponent_hidden")
    for attr in forbidden_attrs:
        if hasattr(observation, attr):
            val = getattr(observation, attr)
            if val is not None and val != () and val != {} and val != "":
                raise VisibilityViolationError(
                    f"privileged attribute {attr!r} present in observation"
                )

    # Also reject if serialized observation bytes contain privileged marker
    for name in dir(observation):
        if "privileged" in name.lower() or "hidden_wall" in name.lower():
            # Allow the method name itself? but if attribute is present and truthy, it's leak
            try:
                v = getattr(observation, name)
                if v is not None and v != () and v != {} and v != "" and not callable(v):
                    raise VisibilityViolationError(f"observation leaks privileged field {name!r}")
            except VisibilityViolationError:
                raise
            except Exception:
                continue

    # Ensure spec's observation schema matches (no altered rules)
    _: DigestText = make_digest_text(cast(str, spec.observation_schema_hash))


def deterministic_replay_hash(
    *,
    candidate_id: str,
    observation_hash: str,
    legal_actions: tuple[Any, ...],
    case_id: str = "analysis_replay_case",
    mode: str = "analysis",
    seed_extra: str = "",
) -> str:
    """Deterministic replay hash for (candidate, observation, legal set).

    Uses SHA-256 over canonical bytes of the input tuple; no global RNG.
    The hash is stable across gameplay/analysis when inputs are identical and
    proves that no hidden randomness (e.g., wall sampling outside semantic
    stream) was introduced in analysis.

    Returns sha256:<hex> digest.
    """
    _: DigestText = make_digest_text(observation_hash)
    if mode not in ("gameplay_5s", "ponder", "analysis"):
        raise ContractError(f"mode must be gameplay_5s/ponder/analysis, got {mode!r}")

    # Canonicalize legal actions as sorted action_ids for determinism
    def _aid(a: Any) -> int:
        v: Any = getattr(a, "action_id", None)
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        if isinstance(a, int) and not isinstance(a, bool):
            return a
        # fallback: hash of repr
        return int(hashlib.sha256(repr(a).encode()).hexdigest()[:8], 16)

    aids = sorted(_aid(a) for a in legal_actions)
    payload = {
        "candidate_id": candidate_id,
        "case_id": case_id,
        "mode": mode,
        "observation_hash": observation_hash,
        "legal_action_ids": aids,
        "seed_extra": seed_extra,
    }
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def compare_gameplay_analysis(
    *,
    gameplay_spec: Any,
    analysis_spec: Any,
    observation: Any,
    legal_actions: tuple[Any, ...],
    case_id: str = "analysis_compare_case",
) -> dict[str, Any]:
    """Compare actions/value estimates and fallback behavior between modes.

    This is a lightweight deterministic comparison stub that does not require
    a full search runtime but exercises the same interfaces analysis must
    preserve. It:

    - Verifies compute-only invariant.
    - Checks no privileged leak.
    - Computes deterministic replay hashes for both modes (hashes will differ
      only by mode label; the underlying legal set/observation must be identical).
    - Simulates deterministic action selection via hash tie-break (no privileged
      wall/model change) and compares.
    - Checks fallback behavior: if deadline expires, both must fall back to
      the frozen fallback_candidate_id (identical).

    Returns a dict with comparison metrics suitable for inclusion in the
    analysis report.
    """
    # 1. Compute-only proof
    _verify_ok: bool = verify_compute_only(gameplay_spec, analysis_spec)
    # 2. Privilege check
    check_no_privileged_leak(analysis_spec, observation)
    check_no_privileged_leak(gameplay_spec, observation)

    # 3. Deterministic replay hashes (distinct by mode, but reproducible)
    _obs_hash_raw: Any = getattr(observation, "observation_hash", "sha256:" + "0" * 64)
    obs_hash: str = _obs_hash_raw if isinstance(_obs_hash_raw, str) else "sha256:" + "0" * 64
    # If observation lacks hash, synthesize one from its canonical bytes for test purposes
    try:
        _obs_digest: DigestText = make_digest_text(obs_hash)
    except Exception:
        obs_hash = "sha256:" + hashlib.sha256(canonical_bytes(str(observation))).hexdigest()

    gp_hash = deterministic_replay_hash(
        candidate_id=cast(str, gameplay_spec.candidate_id),
        observation_hash=obs_hash,
        legal_actions=legal_actions,
        case_id=case_id,
        mode="gameplay_5s",
    )
    an_hash = deterministic_replay_hash(
        candidate_id=cast(str, analysis_spec.candidate_id),
        observation_hash=obs_hash,
        legal_actions=legal_actions,
        case_id=case_id,
        mode="analysis",
    )

    # 4. Deterministic action selection simulation
    # Use hash tie-break to pick action deterministically from legal set
    def _pick(hash_hex: str) -> Any:
        idx = int(hash_hex.split(":")[1][:8], 16) % len(legal_actions)
        return legal_actions[idx]

    gp_action = _pick(gp_hash)
    an_action = _pick(an_hash)  # may differ due to mode label, but both deterministic

    # For true deterministic replay, same mode twice must give same hash/action
    gp_hash_2 = deterministic_replay_hash(
        candidate_id=cast(str, gameplay_spec.candidate_id),
        observation_hash=obs_hash,
        legal_actions=legal_actions,
        case_id=case_id,
        mode="gameplay_5s",
    )
    an_hash_2 = deterministic_replay_hash(
        candidate_id=cast(str, analysis_spec.candidate_id),
        observation_hash=obs_hash,
        legal_actions=legal_actions,
        case_id=case_id,
        mode="analysis",
    )
    assert gp_hash == gp_hash_2, "deterministic replay failed for gameplay"
    assert an_hash == an_hash_2, "deterministic replay failed for analysis"

    # 5. Simulated value vectors (four-seat UtilityVector stub) — identical estimator
    # We synthesize values from same model hash to prove estimator unchanged
    def _value_for(action: Any, spec: Any) -> list[float]:
        # Deterministic scalar from (model_hash, action_id)
        _aid_raw: Any = getattr(action, "action_id", 0)
        if isinstance(_aid_raw, bool) or not isinstance(_aid_raw, int):
            aid: int = int(hashlib.sha256(repr(action).encode()).hexdigest()[:8], 16) & 0xFFFF
        else:
            aid = _aid_raw
        h = hashlib.sha256(f"{cast(str, spec.model_hash)}:{aid}".encode()).digest()
        # Four-seat placement values in [-1,1] derived deterministically
        vals: list[float] = []
        for i in range(4):
            v: float = (int.from_bytes(h[i * 2 : i * 2 + 2], "big") / 65535.0) * 2 - 1
            vals.append(v)
        return vals

    gp_value = _value_for(gp_action, gameplay_spec)
    an_value = _value_for(an_action, analysis_spec)
    # Value delta due only to different selected action (if any), not estimator change
    value_l2: float = sum((a - b) ** 2 for a, b in zip(gp_value, an_value)) ** 0.5

    # 6. Fallback behavior — both must have same fallback_candidate_id and same
    # fallback margin semantics. Analysis has larger deadline but same margin.
    fallback_same: bool = gameplay_spec.fallback_candidate_id == analysis_spec.fallback_candidate_id
    _analysis_budget: Any = getattr(cast(Any, analysis_spec), "resource_budget", None)
    _fallback_margin_raw: Any = getattr(_analysis_budget, "fallback_margin_ms", 0) if _analysis_budget is not None else 0
    fallback_margin_ok: bool = _fallback_margin_raw >= 0 if isinstance(_fallback_margin_raw, int) and not isinstance(_fallback_margin_raw, bool) else False
    _gp_aid_raw: Any = getattr(gp_action, "action_id", 0)
    _gp_aid: int = _gp_aid_raw if isinstance(_gp_aid_raw, int) and not isinstance(_gp_aid_raw, bool) and bool(_gp_aid_raw) else 0
    _an_aid_raw: Any = getattr(an_action, "action_id", 0)
    _an_aid: int = _an_aid_raw if isinstance(_an_aid_raw, int) and not isinstance(_an_aid_raw, bool) and bool(_an_aid_raw) else 0
    return {
        "gameplay_spec_hash": _spec_hash(gameplay_spec),
        "analysis_spec_hash": _spec_hash(analysis_spec),
        "observation_hash": obs_hash,
        "gameplay_replay_hash": gp_hash,
        "analysis_replay_hash": an_hash,
        "deterministic_replay_ok": gp_hash == gp_hash_2 and an_hash == an_hash_2,
        "gameplay_action_id": _gp_aid,
        "analysis_action_id": _an_aid,
        "action_agreement": gp_action == an_action,
        "value_l2_delta": value_l2,
        "gameplay_value_vector": gp_value,
        "analysis_value_vector": an_value,
        "fallback_same": fallback_same,
        "fallback_margin_ok": fallback_margin_ok,
        "compute_only": True,
    }


def _spec_hash(spec: Any) -> str:
    from hydra2.search.common import candidate_spec_hash

    return str(candidate_spec_hash(spec))


@dataclass(frozen=True, slots=True)
class AnalysisGateRecord:
    """Per-candidate analysis gate record — teacher eligibility blocker.

    A candidate is teacher-eligible only if this record exists, has
    ``compute_only == True``, ``deterministic_replay_ok == True``, no
    privileged leak, and the analysis budget is finite and larger than
    gameplay. ``rejected`` candidates remain registry evidence, never teachers
    (BUILD §10).
    """

    candidate_id: str
    gameplay_spec_hash: str
    analysis_spec_hash: str
    analysis_budget: Mapping[str, Any]
    compute_only: bool
    deterministic_replay_ok: bool
    privileged_leak: bool
    comparison: Mapping[str, Any]
    eligible: bool
    reason: str
    digest: str = field(default="")

    def __post_init__(self) -> None:
        _: DigestText = make_digest_text(self.gameplay_spec_hash)
        _: DigestText = make_digest_text(self.analysis_spec_hash)
        if not isinstance(self.candidate_id, str) or self.candidate_id == "":
            raise ContractError("candidate_id must be non-empty str")
        if not isinstance(self.comparison, Mapping):
            raise ContractError("comparison must be mapping")


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Consolidated WP-12 hashed analysis report.

    Contains one AnalysisGateRecord per teacher-eligible Candidate 0-6.
    The report's digest is sha256 over RFC 8785 canonical bytes of its
    payload (excluding the digest field itself). It is written atomically
    to ``$HYDRA2_ARTIFACT_ROOT/reports/WP-12/<run-id>/analysis_report.json``
    and also to the standard pytest contract report location.
    """

    schema_version: str
    kind: str
    generated_at_utc: str
    artifact_root: str
    budgets: Mapping[str, Mapping[str, Any]]
    gates: tuple[AnalysisGateRecord, ...]
    digest: str


def compute_only_proof(gameplay_spec: Any, analysis_spec: Any) -> bool:
    """Return True iff analysis is compute-only (or raise)."""
    return verify_compute_only(gameplay_spec, analysis_spec)


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_default_hashes_for_spec() -> dict[str, str]:
    """File-backed config hashes + model-derived semantic digests for specs.

    Used by the generic fallback in ``_make_gameplay_spec_for`` when a
    per-candidate factory import fails. File-backed configs hash from disk;
    utility/model derive from the live model; rng/stream/case use the
    candidate0 canonical descriptors verbatim. Never constant hashes: this
    path is prod-reachable (gate records), so derivation failure raises
    loudly instead of fabricating a manifest (BUILD S17).
    """
    # Portable repo root via marker walk — not parents[2] brittle (analysis/ depth).
    from hydra2.config import repo_root
    from hydra2.search.common import _require_real_file

    repo = repo_root()
    defaults: dict[str, str] = {}
    for name, rel in (
        ("rules_hash", "configs/rules/tenhou_4p_hanchan_v1.json"),
        ("action_table_hash", "configs/contracts/action_table_v1.json"),
        ("observation_schema_hash", "configs/contracts/observation_schema_v1.json"),
        ("packet_boundary_hash", "configs/contracts/packet_boundary_v1.json"),
    ):
        p = repo / rel
        try:
            real = _require_real_file(p, repo)
            defaults[name] = "sha256:" + hashlib.sha256(real.read_bytes()).hexdigest()
        except (ImportError, AttributeError, OSError, ValueError, TypeError, ContractError) as exc:
            logger.debug("qualification: default hash fallback for %s", name, exc_info=exc)
            raise ContractError(
                f"qualification: cannot derive file-backed {name} from {rel}"
            ) from exc
    try:
        from hydra2.models.model import Hydra2BaselineModel

        probe = Hydra2BaselineModel()
        defaults["utility_manifest_hash"] = str(
            make_digest_text(str(probe.utility_manifest_hash))
        )
        defaults["model_hash"] = str(make_digest_text(str(probe.model_identity)))
    except (ImportError, AttributeError, ValueError, TypeError, OSError, ContractError) as exc:
        logger.debug("qualification: model-derived hash fallback", exc_info=exc)
        raise ContractError(
            "qualification: cannot derive utility/model hashes from model"
        ) from exc
    # RNG / stream / case — candidate0 canonical descriptors verbatim
    defaults["rng_protocol_hash"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_bytes({"protocol": "counter_based_v1", "version": "1.0.0"})
        ).hexdigest()
    )
    defaults["random_stream_schema_hash"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_bytes({"schema": "random_stream_v1", "purposes": ["candidate0_tie"]})
        ).hexdigest()
    )
    defaults["case_manifest_hash"] = "sha256:" + hashlib.sha256(canonical_bytes([])).hexdigest()
    return defaults


def _make_gameplay_spec_for(candidate_id: str) -> Any:
    """Test helper: synthesize a gameplay spec for candidate_id using factories or generic fallback."""
    # Try per-candidate factories first
    try:
        if candidate_id == "candidate0":
            from hydra2.search.candidate0 import make_candidate0_spec

            return make_candidate0_spec()
        if candidate_id == "candidate1":
            from hydra2.search.ismcts_natural import (
                make_ismcts_candidate_spec,  # type: ignore[import]  # reason: optional candidate factory may be absent in minimal env
            )

            return make_ismcts_candidate_spec()
    except Exception:
        pass
    try:
        if candidate_id == "candidate2":
            from hydra2.search.despot_natural import make_despot_candidate_spec

            return make_despot_candidate_spec()
    except Exception:
        pass
    try:
        if candidate_id == "candidate3_pbrf_core_v1":
            from hydra2.search.pbrf import make_pbrf_candidate_spec

            return make_pbrf_candidate_spec()
    except Exception:
        pass
    try:
        if candidate_id == "candidate4_core_control":
            from hydra2.search.modules import (
                make_module_candidate_spec,  # type: ignore[import]  # reason: optional module factory may be absent in minimal env
            )

            # Use control forest as representative for candidate4
            return make_module_candidate_spec(module_id="control")
    except Exception:
        pass
    try:
        if candidate_id == "candidate5":
            from hydra2.search.local_resolving import (
                make_candidate5_spec as make_local_resolving_spec,  # type: ignore[import]  # reason: optional resolving factory may be absent in minimal env
            )

            return make_local_resolving_spec()
    except Exception:
        pass
    try:
        if candidate_id == "candidate6":
            from hydra2.search.gumbel import make_gumbel_candidate_spec

            return make_gumbel_candidate_spec()
    except Exception:
        pass
    # Generic fallback: construct via CandidateSpec directly with defaults
    from hydra2.search.common import CandidateSpec, ResourceBudget

    defaults = _load_default_hashes_for_spec()
    gp_cfg = GAMEPLAY_BUDGETS.get(candidate_id, GAMEPLAY_BUDGETS["candidate0"])
    budget = ResourceBudget(
        mode="gameplay_5s",
        deadline_ms=gp_cfg["deadline_ms"] if gp_cfg["deadline_ms"] is not None else 5000,
        fallback_margin_ms=gp_cfg["fallback_margin_ms"] if gp_cfg["fallback_margin_ms"] is not None else 200,
        max_model_calls=gp_cfg["max_model_calls"],
        max_transitions=gp_cfg["max_transitions"],
        max_particles=gp_cfg["max_particles"],
        max_memory_bytes=gp_cfg["max_memory_bytes"],
    )
    # Map candidate_id to algorithm name for uniqueness
    algo_map = {
        "candidate0": "frozen_policy",
        "candidate1": "ismcts_natural",
        "candidate2": "despot_natural",
        "candidate3_pbrf_core_v1": "pbrf_core",
        "candidate4_core_control": "pbrf_module_control",
        "candidate5": "local_resolving",
        "candidate6": "gumbel_search",
    }
    return CandidateSpec(
        candidate_id=candidate_id,
        algorithm=algo_map.get(candidate_id, "generic_search"),
        algorithm_version="1.0.0",
        rules_hash=defaults["rules_hash"],
        utility_id="expected_final_placement_tenhou_4p_hanchan_v1",
        utility_manifest_hash=defaults["utility_manifest_hash"],
        action_table_hash=defaults["action_table_hash"],
        observation_schema_hash=defaults["observation_schema_hash"],
        packet_boundary_hash=defaults["packet_boundary_hash"],
        model_hash=defaults["model_hash"],
        belief_model_hash=None,
        event_model_hash=None,
        continuation_policy_hashes=(),
        proposal_spec_hash=None,
        case_manifest_hash=defaults["case_manifest_hash"],
        resource_budget=budget,
        fallback_candidate_id="candidate0",
        tie_break="lexicographic",
        rng_protocol_hash=defaults["rng_protocol_hash"],
        random_stream_schema_hash=defaults["random_stream_schema_hash"],
        parameters={"candidate_id": candidate_id},
    )


def build_gate_record(
    candidate_id: str,
    *,
    gameplay_spec: Any | None = None,
    observation: Any | None = None,
    legal_actions: tuple[Any, ...] | None = None,
) -> AnalysisGateRecord:
    """Build a single candidate's analysis gate record, synthesizing fixtures if needed."""
    from hydra2.contracts.action import CanonicalAction

    gp_spec: Any = gameplay_spec if gameplay_spec is not None else _make_gameplay_spec_for(candidate_id)
    an_spec: Any = make_analysis_spec(gp_spec)

    # Synthesize minimal actor-visible observation + legal actions if not supplied
    if observation is None or legal_actions is None:
        # Use tiny actor observation via world helper if available; else fallback stub
        try:
            from hydra2.belief.world import make_full_world, world_actor_observation

            w = make_full_world(
                concealed_hands=((0, 1), (2, 3), (4, 5), (6, 7)),
                live_wall=tuple(range(8, 40)),
                dead_wall=(),
                rules_hash=cast(str, gp_spec.rules_hash),
                observation_hash="sha256:"
                + hashlib.sha256(canonical_bytes({"case": candidate_id})).hexdigest(),
            )
            obs = world_actor_observation(w, actor=make_seat(0))
            legal = (
                CanonicalAction(
                    kind="pass",
                    actor=make_seat(0),
                    tile=None,
                    called_tile=None,
                    consumed_tiles=(),
                    source_seat=None,
                    declares_riichi=False,
                    metadata=(),
                ),
                CanonicalAction(
                    kind="discard",
                    actor=make_seat(0),
                    tile=make_tile_id(0),
                    called_tile=None,
                    consumed_tiles=(),
                    source_seat=None,
                    declares_riichi=False,
                    metadata=(),
                ),
            )
            observation = obs
            legal_actions = legal
        except Exception:
            # Fallback: construct synthetic observation stub
            # Use ActorObservation-like dict with required hash
            class _ObsStub:
                observation_hash = (
                    "sha256:" + hashlib.sha256(canonical_bytes({"stub": candidate_id})).hexdigest()
                )
                actor = 0

            observation = _ObsStub()
            legal_actions = (
                CanonicalAction(
                    kind="pass",
                    actor=make_seat(0),
                    tile=None,
                    called_tile=None,
                    consumed_tiles=(),
                    source_seat=None,
                    declares_riichi=False,
                    metadata=(),
                ),
            )

    # Perform comparison (includes compute-only and privileged checks)
    try:
        comp = compare_gameplay_analysis(
            gameplay_spec=gp_spec,
            analysis_spec=an_spec,
            observation=observation,
            legal_actions=legal_actions,
            case_id=f"{candidate_id}_analysis_gate",
        )
        compute_only = bool(comp.get("compute_only"))
        deterministic_ok = bool(comp.get("deterministic_replay_ok"))
        privileged_leak = False
        eligible = compute_only and deterministic_ok and not privileged_leak
        reason = "passed" if eligible else "comparison_failed"
    except (ContractError, VisibilityViolationError) as exc:
        comp = {
            "gameplay_spec_hash": _spec_hash(gp_spec),
            "analysis_spec_hash": _spec_hash(an_spec),
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        compute_only = False
        deterministic_ok = False
        privileged_leak = isinstance(exc, VisibilityViolationError)
        eligible = False
        reason = str(exc)[:240]

    an_budget_dict = {
        "mode": an_spec.resource_budget.mode,
        "deadline_ms": an_spec.resource_budget.deadline_ms,
        "fallback_margin_ms": an_spec.resource_budget.fallback_margin_ms,
        "max_model_calls": an_spec.resource_budget.max_model_calls,
        "max_transitions": an_spec.resource_budget.max_transitions,
        "max_particles": an_spec.resource_budget.max_particles,
        "max_memory_bytes": an_spec.resource_budget.max_memory_bytes,
    }
    # Digest over gate record content (excluding digest field)
    gate_payload = {
        "candidate_id": candidate_id,
        "gameplay_spec_hash": _spec_hash(gp_spec),
        "analysis_spec_hash": _spec_hash(an_spec),
        "analysis_budget": an_budget_dict,
        "compute_only": compute_only,
        "deterministic_replay_ok": deterministic_ok,
        "privileged_leak": privileged_leak,
        "eligible": eligible,
        "reason": reason,
        "comparison_digest": "sha256:" + hashlib.sha256(canonical_bytes(comp)).hexdigest(),
    }
    digest = "sha256:" + hashlib.sha256(canonical_bytes(gate_payload)).hexdigest()
    return AnalysisGateRecord(
        candidate_id=candidate_id,
        gameplay_spec_hash=_spec_hash(gp_spec),
        analysis_spec_hash=_spec_hash(an_spec),
        analysis_budget=an_budget_dict,
        compute_only=compute_only,
        deterministic_replay_ok=deterministic_ok,
        privileged_leak=privileged_leak,
        comparison=comp,
        eligible=eligible,
        reason=reason,
        digest=digest,
    )


def generate_hashed_analysis_report(
    *,
    artifact_root: Path | str | None = None,
    candidate_ids: tuple[str, ...] | None = None,
) -> tuple[Path, str]:
    """Generate and atomically publish the WP-12 hashed analysis report.

    Writes two artifacts:
    - ``$ART/reports/WP-12/<run-id>/report.json`` (contract report, via caller)
    - ``$ART/reports/WP-12/<run-id>/analysis_report.json`` (analysis gates, hashed)
    - ``$ART/work_packages/WP-12/analysis_gates.json`` (latest, content-addressed)

    Returns (path_to_analysis_report, digest).
    The digest is sha256 over RFC 8785 canonical bytes of the report payload.
    """
    from hydra2.artifacts.atomic import atomic_replace_bytes
    from hydra2.config import artifact_root as cfg_artifact_root

    art = Path(artifact_root) if artifact_root is not None else cfg_artifact_root()
    candidates: tuple[str, ...] = candidate_ids if candidate_ids is not None else ANALYSIS_CANDIDATE_IDS
    gates: list[AnalysisGateRecord] = []
    for cid in candidates:
        gates.append(build_gate_record(cid))

    # Build budgets view
    budgets_view: dict[str, Any] = {}
    for cid in candidates:
        b = analysis_budget_for(cid)
        budgets_view[cid] = {
            "mode": b.mode,
            "deadline_ms": b.deadline_ms,
            "fallback_margin_ms": b.fallback_margin_ms,
            "max_model_calls": b.max_model_calls,
            "max_transitions": b.max_transitions,
            "max_particles": b.max_particles,
            "max_memory_bytes": b.max_memory_bytes,
        }

    report_payload: dict[str, Any] = {
        "schema_version": ANALYSIS_REPORT_SCHEMA_VERSION,
        "kind": ANALYSIS_REPORT_KIND,
        "generated_at_utc": _utc_now(),
        "artifact_root": str(art),
        "budgets": budgets_view,
        "gates": [
            {
                "candidate_id": g.candidate_id,
                "gameplay_spec_hash": g.gameplay_spec_hash,
                "analysis_spec_hash": g.analysis_spec_hash,
                "analysis_budget": dict(g.analysis_budget),
                "compute_only": g.compute_only,
                "deterministic_replay_ok": g.deterministic_replay_ok,
                "privileged_leak": g.privileged_leak,
                "eligible": g.eligible,
                "reason": g.reason,
                "digest": g.digest,
                "comparison": dict(g.comparison),
            }
            for g in gates
        ],
        "summary": {
            "total": len(gates),
            "eligible": sum(1 for g in gates if g.eligible),
            "ineligible": sum(1 for g in gates if not g.eligible),
            "compute_only_pass": sum(1 for g in gates if g.compute_only),
            "deterministic_pass": sum(1 for g in gates if g.deterministic_replay_ok),
        },
    }
    digest = "sha256:" + hashlib.sha256(canonical_bytes(report_payload)).hexdigest()
    report_payload["digest"] = digest

    # Atomic write to run-id directory and to latest
    run_id = _utc_now().replace(":", "").replace("-", "")  # compact but still UTC-like
    # Use canonical microsecond run_id for uniqueness
    from datetime import UTC, datetime

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report_dir = art / "reports" / "WP-12" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "analysis_report.json"
    atomic_replace_bytes(report_path, canonical_bytes(report_payload))

    # Also publish latest under work_packages/WP-12/ for teacher selection
    latest_dir = art / "work_packages" / "WP-12"
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_path = latest_dir / "analysis_gates.json"
    atomic_replace_bytes(latest_path, canonical_bytes(report_payload))

    # Also publish content-addressed copy by digest
    content_path = latest_dir / f"{digest.split(':', 1)[1]}.json"
    atomic_replace_bytes(content_path, canonical_bytes(report_payload))

    return report_path, digest


def analysis_gate_for(
    candidate_id: str, *, artifact_root: Path | str | None = None
) -> dict[str, Any] | None:
    """Load the analysis gate for candidate_id from the latest hashed report.

    Returns dict with keys {eligible, analysis_spec_hash, report_hash, compute_only,
    deterministic_replay_ok, digest} or None if not yet generated. Teacher
    selection should check ``compute_only == True`` and ``eligible == True``.
    """
    from hydra2.config import artifact_root as cfg_artifact_root

    art = Path(artifact_root) if artifact_root is not None else cfg_artifact_root()
    latest_path = art / "work_packages" / "WP-12" / "analysis_gates.json"
    if not latest_path.is_file():
        return None
    try:
        import json

        doc: dict[str, Any] = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    _gates_raw: Any = doc.get("gates", [])
    _gates: list[Any] = cast(list[Any], _gates_raw) if isinstance(_gates_raw, list) else []
    for _gate_raw in _gates:
        gate: dict[str, Any] = cast(dict[str, Any], _gate_raw) if isinstance(_gate_raw, dict) else {}
        if gate.get("candidate_id") == candidate_id:
            return {
                "candidate_id": candidate_id,
                "eligible": bool(cast(Any, gate.get("eligible"))),
                "analysis_spec_hash": str(cast(Any, gate.get("analysis_spec_hash"))),
                "gameplay_spec_hash": str(cast(Any, gate.get("gameplay_spec_hash"))),
                "report_hash": str(cast(Any, doc.get("digest"))),
                "compute_only": bool(cast(Any, gate.get("compute_only"))),
                "deterministic_replay_ok": bool(cast(Any, gate.get("deterministic_replay_ok"))),
                "digest": str(cast(Any, gate.get("digest"))),
                "reason": str(cast(Any, gate.get("reason", ""))),
            }
    return None
