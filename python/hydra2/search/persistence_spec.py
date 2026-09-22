"""Persistence factorial spec factory — per-arm CandidateSpec and choice helper.

Owns the deployable-budget half of the persistence factorial: the file-backed
:func:`_default_hashes` digests, the :func:`make_persistence_candidate_spec`
builder binding each B/F/R/P/C arm to its frozen manifest hashes and
resource budget, the :func:`validate_deadline_and_fallback` deadline guard,
and the deterministic :func:`deterministic_gumbel_for_arm` choice scalar.
The arm/packet vocabulary lives in
:mod:`hydra2.search.persistence_kernel` and the per-arm state machine in
:mod:`hydra2.search.persistence_planner` so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Literal

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError
from hydra2.search.persistence_kernel import DEPLOYABLE_DEADLINE_MS as DEPLOYABLE_DEADLINE_MS
from hydra2.search.persistence_kernel import PersistenceArm, make_persistence_arm

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "_default_hashes",
    "deterministic_gumbel_for_arm",
    "make_persistence_candidate_spec",
    "validate_deadline_and_fallback",
]


# ---------------------------------------------------------------------------
# CandidateSpec factory per arm
# ---------------------------------------------------------------------------


def _default_hashes() -> dict[str, str]:
    from hydra2.search.common import PLACEHOLDER_A, PLACEHOLDER_B, REPO_ROOT

    repo = REPO_ROOT

    # Provide deterministic fallback hashes without requiring files
    def _sha(p: Path) -> str:
        try:
            return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:
            return "sha256:" + PLACEHOLDER_A

    out: dict[str, str] = {}
    for key, rel in (
        ("rules_hash", "configs/rules/tenhou_4p_hanchan_v1.json"),
        ("action_table_hash", "configs/contracts/action_table_v1.json"),
        ("observation_schema_hash", "configs/contracts/observation_schema_v1.json"),
        ("packet_boundary_hash", "configs/contracts/packet_boundary_v1.json"),
    ):
        out[key] = _sha(repo / rel)
    # model/utility placeholders
    out["model_hash"] = "sha256:" + hashlib.sha256(b"hydra2-baseline-model-v1").hexdigest()
    out["utility_manifest_hash"] = "sha256:" + PLACEHOLDER_B

    out["rng_protocol_hash"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_bytes({"protocol": "counter_based_v1", "version": "1.0.0"})
        ).hexdigest()
    )
    out["random_stream_schema_hash"] = (
        "sha256:"
        + hashlib.sha256(
            canonical_bytes({"schema": "random_stream_v1", "purposes": ["persistence_factorial"]})
        ).hexdigest()
    )
    out["case_manifest_hash"] = "sha256:" + hashlib.sha256(canonical_bytes([])).hexdigest()
    return out


def make_persistence_candidate_spec(
    *,
    arm_id: Literal["B", "F", "R", "P", "C"],
    deadline_ms: int | None = None,
    fallback_margin_ms: int | None = None,
    max_model_calls: int | None = None,
    max_transitions: int | None = None,
    parameters: dict[str, Any] | None = None,
) -> Any:
    """Build CandidateSpec for a persistence arm (SPEC 15, deployable check)."""
    from hydra2.search.common import CandidateSpec, ResourceBudget

    arm = make_persistence_arm(arm_id)
    defaults = _default_hashes()
    if deadline_ms is None:
        deadline_ms = arm.own_deadline_ms
    if fallback_margin_ms is None:
        fallback_margin_ms = 500
    # Validate deadline / deployable invariants via PersistenceArm
    if arm_id in ("B", "F", "R", "P") and deadline_ms > DEPLOYABLE_DEADLINE_MS:
        raise ContractError(
            f"deployable arm {arm_id} deadline {deadline_ms} >{DEPLOYABLE_DEADLINE_MS}"
        )
    if arm_id == "C":
        # C gets extended budget: deadline + extra_wait_allowance is the *scheduled max*
        # The ResourceBudget deadline reflects the extended allowance (lab control).
        if max_model_calls is None:
            max_model_calls = 64
        if max_transitions is None:
            max_transitions = 256
    else:
        if max_model_calls is None:
            max_model_calls = 1 if arm_id == "B" else 32
        if max_transitions is None:
            max_transitions = 0 if arm_id == "B" else 128
    budget = ResourceBudget(
        mode="gameplay_5s",
        deadline_ms=deadline_ms + (arm.extra_wait_allowance_ms if arm_id == "C" else 0),
        fallback_margin_ms=fallback_margin_ms,
        max_model_calls=max_model_calls,
        max_transitions=max_transitions,
        max_particles=32,
        max_memory_bytes=None,
    )
    cand_id = f"persistence-{arm_id}"
    return CandidateSpec(
        candidate_id=cand_id,
        algorithm="persistence_factorial",
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
        tie_break="greedy",
        rng_protocol_hash=defaults["rng_protocol_hash"],
        random_stream_schema_hash=defaults["random_stream_schema_hash"],
        parameters=dict(
            {
                "persistence_arm": arm_id,
                "retain_state": arm.retain_state,
                "opponent_time_compute": arm.opponent_time_compute,
                "deployable": arm.deployable,
                "own_deadline_ms": arm.own_deadline_ms,
                "extra_wait_allowance_ms": arm.extra_wait_allowance_ms,
            }
            | (parameters if parameters is not None else {})
        ),
    )


def validate_deadline_and_fallback(
    *,
    arm: PersistenceArm,
    deadline_ms: int,
    fallback_margin_ms: int,
) -> None:
    """Deadline guard — bridge-owned (``search.persistence_spec_validate_deadline_and_fallback``)."""
    try:
        from hydra2._native import search as _bridge_spec  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native.search missing; rebuild the bridge with `pixi run build-ext`"
        ) from exc
    try:
        _bridge_spec.persistence_spec_validate_deadline_and_fallback(  # type: ignore[attr-defined]
            arm.id, deadline_ms, fallback_margin_ms
        )
    except (AttributeError, ImportError) as exc:
        raise ImportError(
            "hydra2._native.search.persistence_spec_validate_deadline_and_fallback missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc
    except (ValueError, TypeError) as exc:
        raise ContractError(f"persistence spec validate rejected: {exc}") from exc


# ---------------------------------------------------------------------------
# Deterministic choice and vector helpers
# ---------------------------------------------------------------------------


def deterministic_gumbel_for_arm(
    *, arm_id: str, case_id: str, action_id: int, seed_bytes: bytes = b"hydra2-persistence-v1"
) -> float:
    """Deterministic scalar in [0,1) — bridge-owned (``search.persistence_spec_gumbel_for_arm``)."""
    try:
        from hydra2._native import search as _bridge_spec  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2._native.search missing; rebuild the bridge with `pixi run build-ext`"
        ) from exc
    try:
        out: float = _bridge_spec.persistence_spec_gumbel_for_arm(  # type: ignore[attr-defined]
            arm_id, case_id, action_id, seed_bytes
        )
        return out
    except (AttributeError, ImportError) as exc:
        raise ImportError(
            "hydra2._native.search.persistence_spec_gumbel_for_arm missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc
    except (ValueError, TypeError) as exc:
        raise ContractError(f"persistence spec gumbel rejected: {exc}") from exc
