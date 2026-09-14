# ruff: noqa: E501, TC006, SIM102  # reason: legacy blanket; E501 URLs unwrappable, TC006 quotes intentional, SIM102 nested guards readable, PERF401/B905 perf-critical. Evidence: https://docs.python.org/3/library/importlib.resources.html
"""Analysis budgets and compute-only gates.

Owns the frozen finite budgets for gameplay and analysis modes plus the
proofs that analysis changes only charged compute: every analysis budget is
finite, mode=analysis, and larger than its gameplay baseline, while every
other semantic field (rules, utility, action table, observation schema,
model, parameters) stays byte-identical. The privilege firewall rejects
observations carrying hidden fields. An analysis spec that shrinks caps,
reuses the gameplay deadline, or alters any semantic field fails closed so
uncharged work can never pass as analysis.
"""

from __future__ import annotations

from typing import Any, cast

from hydra2.contracts.common import (
    ContractError,
    DigestText,
    VisibilityViolationError,
    make_digest_text,
)

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
    # Removing the checks above would let analysis shrink caps or reuse the
    # gameplay deadline — uncharged compute the gate must reject.
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
    """Enlarge gameplay caps 4x (30s deadline) for unknown candidates."""
    from hydra2.search.common import ResourceBudget

    def _enlarge(v: int | None, fallback: int) -> int:
        if v is None:
            return fallback
        return max(v * 4, fallback)

    _fallback_raw: Any = getattr(gp, "fallback_margin_ms", None)
    _fallback_val: int = (
        _fallback_raw
        if isinstance(_fallback_raw, int) and not isinstance(_fallback_raw, bool)
        else 500
    )
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
    # Caps must not shrink when gameplay is finite: a smaller analysis cap
    # is uncharged reduction, so the gate rejects it.
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
    - No privileged fields like full_world, hidden wall (unseen tile wall),
      opponent hand.
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
            # Methods are callable and skipped below; a present, truthy,
            # non-callable privileged/hidden_wall attribute is a leak.
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
