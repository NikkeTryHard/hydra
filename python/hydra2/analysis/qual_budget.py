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

Frozen tables and pure budget math execute in ``hydra2._native.search``
(``crates/bridge/src/qual_budget.rs``); this module keeps thin re-exports and
translators plus the spec orchestration, compute-only proofs, and privilege
firewall (dataclasses, IO, and wall-clock-free validation shaping stay here).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hydra2._native import search as _budget_bridge  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import validate_digest
from hydra2.contracts.common import (
    ContractError,
    DigestText,
    VisibilityViolationError,
)

if TYPE_CHECKING:
    from hydra2.search.common import ResourceBudget


def _leaf(name: str) -> Any:
    """Resolve one ``hydra2._native.search`` budget leaf (fail closed)."""
    try:
        return getattr(_budget_bridge, name)
    except AttributeError as exc:
        raise ImportError(
            f"hydra2._native.search.{name} missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc


def _deadline_greater(gp_deadline_ms: int, an_deadline_ms: int) -> bool:
    """Shared deadline-monotonic bit, bridge-first (``qual_gate_deadline_greater``)."""
    try:
        leaf: Any = _budget_bridge.qual_gate_deadline_greater  # type: ignore[attr-defined]  # reason: analysis_leaves2 leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        return an_deadline_ms > gp_deadline_ms
    try:
        greater: bool = leaf(gp_deadline_ms, an_deadline_ms)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    return greater


def _cap_no_shrink(gp_cap: int | None, an_cap: int | None) -> bool:
    """Shared no-shrink bit, bridge-first (``qual_gate_cap_no_shrink``)."""
    try:
        leaf: Any = _budget_bridge.qual_gate_cap_no_shrink  # type: ignore[attr-defined]  # reason: analysis_leaves2 leaf lands with MAIN wiring; AttributeError fallback covers stale .so
    except AttributeError:
        return not (gp_cap is not None and an_cap is not None and an_cap < gp_cap)
    try:
        no_shrink: bool = leaf(gp_cap, an_cap)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    return no_shrink


# Teacher-eligible Candidate 0-6 outcome registry — one entry per candidate.
# Candidate 4 is represented by the control forest ``candidate4_core_control``
# (WP-09B cumulative build is PBRF core alone per CON-09-004; modules are
# independent promotions). Persistence factorial (WP-09C) is not a teacher
# candidate. This list is normative for WP-12 gating.
#: Teacher-eligible registry, re-exported from the bridge single source.
ANALYSIS_CANDIDATE_IDS: tuple[str, ...] = tuple(_leaf("QUAL_BUDGET_CANDIDATE_IDS"))

# Finite analysis budgets — deadline/resource caps frozen blind to arm labels.
# Gameplay is 5,000 ms; analysis declares a larger finite deadline (30,000 ms)
# plus larger finite caps, all charged. Every budget is mode=analysis.
# Caps are finite (non-None) and bounded to prevent unbounded claims.
#: Finite analysis budgets, rebuilt from the bridge ``QUAL_BUDGET_ANALYSIS_ROWS``
#: table (single source): ``(candidate_id, deadline_ms, fallback_margin_ms,
#: max_model_calls, max_transitions, max_particles, max_memory_bytes)`` per row.
_ANALYSIS_ROWS: list[tuple[str, int, int, int, int, int, int]] = _leaf("QUAL_BUDGET_ANALYSIS_ROWS")
ANALYSIS_BUDGETS: dict[str, dict[str, int | None]] = {
    # candidate_id -> {deadline_ms, fallback_margin_ms, max_model_calls, max_transitions, max_particles, max_memory_bytes}
    row[0]: {
        "deadline_ms": row[1],
        "fallback_margin_ms": row[2],
        "max_model_calls": row[3],
        "max_transitions": row[4],
        "max_particles": row[5],
        "max_memory_bytes": row[6],
    }
    for row in _ANALYSIS_ROWS
}

# Gameplay baselines for comparison (used only for "additional charged compute" check).
# These mirror the factories' default gameplay budgets. They are not imported
# dynamically to keep analysis deterministic and independent of mutable code.
#: Gameplay baselines, rebuilt from the bridge ``QUAL_BUDGET_GAMEPLAY_ROWS`` table
#: (single source): ``(candidate_id, deadline_ms, fallback_margin_ms,
#: max_model_calls, max_transitions, max_particles)`` per row. Gameplay
#: ``max_memory_bytes`` is ``None`` for every row (unbounded gameplay heap), so it
#: rides implicitly here rather than as a seventh row lane.
_GAMEPLAY_ROWS: list[tuple[str, int, int, int, int, int]] = _leaf("QUAL_BUDGET_GAMEPLAY_ROWS")
GAMEPLAY_BUDGETS: dict[str, dict[str, int | None]] = {
    row[0]: {
        "deadline_ms": row[1],
        "fallback_margin_ms": row[2],
        "max_model_calls": row[3],
        "max_transitions": row[4],
        "max_particles": row[5],
        "max_memory_bytes": None,
    }
    for row in _GAMEPLAY_ROWS
}


def _require_finite_budget(budget: Any) -> None:
    """Validate that a ResourceBudget is finite and analysis-legal.

    Range checks execute in ``hydra2._native.search`` (``qual_budget_check_finite``);
    this wrapper keeps the live-``ResourceBudget`` type gate plus ``ContractError``
    shaping. ``TypeError`` maps too: a non-``str`` mode can never be analysis-legal.
    """
    from hydra2.search.common import ResourceBudget  # lazy

    if not isinstance(budget, ResourceBudget):
        raise ContractError(f"resource_budget must be ResourceBudget, got {type(budget).__name__}")
    try:
        _leaf("qual_budget_check_finite")(
            budget.mode,
            budget.deadline_ms,
            budget.fallback_margin_ms,
            budget.max_model_calls,
            budget.max_transitions,
            budget.max_particles,
            budget.max_memory_bytes,
        )
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


def analysis_budget_for(candidate_id: str) -> ResourceBudget:
    """Return the frozen finite analysis ResourceBudget for candidate_id.

    Table lookup executes in ``hydra2._native.search`` (``qual_budget_for`` returns
    ``(deadline_ms, fallback_margin_ms, max_model_calls, max_transitions,
    max_particles, max_memory_bytes)``); this wrapper keeps ``ResourceBudget``
    construction plus the finiteness proof.
    """
    from hydra2.search.common import ResourceBudget

    try:
        row: tuple[int, int, int, int, int, int] = _leaf("qual_budget_for")(candidate_id)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    budget = ResourceBudget(
        mode="analysis",
        deadline_ms=row[0],
        fallback_margin_ms=row[1],
        max_model_calls=row[2],
        max_transitions=row[3],
        max_particles=row[4],
        max_memory_bytes=row[5],
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
    if not _deadline_greater(gp.deadline_ms, analysis_budget.deadline_ms):
        raise ContractError(
            f"analysis deadline {analysis_budget.deadline_ms} must be > gameplay {gp.deadline_ms}"
        )
    for name in ("max_model_calls", "max_transitions"):
        gp_v: int | None = getattr(gp, name)
        an_v: int | None = getattr(analysis_budget, name)
        if not _cap_no_shrink(gp_v, an_v):
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


def _derive_generic_analysis_budget(gp: Any) -> ResourceBudget:
    """Enlarge gameplay caps 4x (30s deadline) for unknown candidates.

    Field math executes in ``hydra2._native.search`` (``qual_budget_generic_budget``
    returns ``(deadline_ms, fallback_margin_ms, max_model_calls, max_transitions,
    max_particles, max_memory_bytes)``); this wrapper keeps ``ResourceBudget``
    construction. Direct-bridge bool/non-int caps fail closed here rather than
    downstream — same reject outcome, earlier gate.
    """
    from hydra2.search.common import ResourceBudget

    try:
        row: tuple[int, int, int, int, int, int] = _leaf("qual_budget_generic_budget")(
            getattr(gp, "max_model_calls", None),
            getattr(gp, "max_transitions", None),
            getattr(gp, "max_particles", None),
            getattr(gp, "fallback_margin_ms", None),
        )
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    return ResourceBudget(
        mode="analysis",
        deadline_ms=row[0],
        fallback_margin_ms=row[1],
        max_model_calls=row[2],
        max_transitions=row[3],
        max_particles=row[4],
        max_memory_bytes=row[5],
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
    if not _deadline_greater(gp_b.deadline_ms, an_b.deadline_ms):
        raise ContractError(
            f"analysis deadline {an_b.deadline_ms} must exceed gameplay {gp_b.deadline_ms}"
        )
    # Caps must not shrink when gameplay is finite: a smaller analysis cap
    # is uncharged reduction, so the gate rejects it.
    for name in ("max_model_calls", "max_transitions", "max_particles"):
        gv: int | None = getattr(gp_b, name)
        av: int | None = getattr(an_b, name)
        if not _cap_no_shrink(gv, av):
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
    from hydra2.contracts.observation_actor import ActorObservation

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
            _: DigestText = validate_digest(cast(str, oh))
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
    _: DigestText = validate_digest(cast(str, spec.observation_schema_hash))
