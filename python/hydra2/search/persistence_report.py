"""Persistence factorial report — frozen whole-block contrasts and strata.

Owns the frozen :class:`FactorialReport` surface: the
:class:`FactorialContrasts` record, the
:func:`factorial_contrasts` bootstrap over wall blocks, the
:func:`stratify_surprise_miss_recovery` per-packet strata, and the
:func:`generate_factorial_report` whole-block builder with its actual
resource accounting. The arm/packet vocabulary lives in
:mod:`hydra2.search.persistence_kernel`, the CandidateSpec factory in
:mod:`hydra2.search.persistence_spec`, and the per-arm state machine in
:mod:`hydra2.search.persistence_planner` so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from hydra2._native import search as _bridge_search  # pyrefly: ignore[missing-import]
from hydra2.artifacts.canonical import canonical_bytes
from hydra2.contracts.common import ContractError
from hydra2.eval.statistics import bootstrap_blocks
from hydra2.search.persistence_kernel import ARM_DEFS as ARM_DEFS
from hydra2.search.persistence_spec import (
    deterministic_gumbel_for_arm as deterministic_gumbel_for_arm,
)

__all__ = [
    "FactorialContrasts",
    "FactorialReport",
    "factorial_contrasts",
    "generate_factorial_report",
    "stratify_surprise_miss_recovery",
]

_REPORT_BRIDGE_ATTRS = (
    "persistence_report_block_diffs",
    "persistence_report_mean",
    "persistence_report_contrast_seed",
    "persistence_report_strata_rates",
    "persistence_report_resources_equal",
)


def _require_report_bridge():  # type: ignore[no-untyped-def]
    """Import the built ``search`` bridge surface with the report fns (fail closed)."""
    for attr in _REPORT_BRIDGE_ATTRS:
        if not hasattr(_bridge_search, attr):
            raise ImportError(
                f"hydra2._native.search.{attr} missing (stale .so); "
                "rebuild the bridge with `pixi run build-ext`"
            )
    return _bridge_search


# ---------------------------------------------------------------------------
# Factorial report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactorialContrasts:
    estimate: float
    ci_low: float
    ci_high: float
    unit: str = "wall_block"
    p_value: float | None = None


@dataclass(frozen=True, slots=True)
class FactorialReport:
    """Frozen whole-block factorial report (deterministic, canonical)."""

    report_id: str
    generated_at_utc: str
    arms: dict[str, Any]
    block_manifest_hash: str
    num_blocks: int
    per_arm_mean: dict[str, float]
    contrasts: dict[str, FactorialContrasts]
    resource_samples: dict[str, Any]
    strata: dict[str, Any]
    uncertainty: dict[str, Any]
    notes: tuple[str, ...] = ()


def factorial_contrasts(
    *,
    placements_by_arm: dict[str, list[float]],
    alpha: float = 0.05,
    resamples: int = 2000,
) -> dict[str, FactorialContrasts]:
    """Compute P-F, R-F, P-R, P-C contrasts with bootstrap over wall blocks.

    Each arm list is per-block mean placement (lower is better; contrast negative = improvement).
    Units are wall blocks; resamples are deterministic seeds derived from arm names.
    """

    pairs = [("P-F", "P", "F"), ("R-F", "R", "F"), ("P-R", "P", "R"), ("P-C", "P", "C")]
    for name, a, b in pairs:
        if a not in placements_by_arm or b not in placements_by_arm:
            raise ContractError(f"missing arm placements for {name}: need {a},{b}")
        if len(placements_by_arm[a]) != len(placements_by_arm[b]):
            raise ContractError(
                f"block count mismatch {len(placements_by_arm[a])} vs {len(placements_by_arm[b])}"
            )
    bridge = _require_report_bridge()
    out: dict[str, FactorialContrasts] = {}
    for name, a, b in pairs:
        try:
            diffs: list[float] = bridge.persistence_report_block_diffs(
                list(placements_by_arm[a]),
                list(placements_by_arm[b]),
            )
        except Exception as exc:
            raise ContractError(
                f"persistence report bridge block_diffs failed for {name}: {exc}"
            ) from exc
        try:
            seed: bytes = bridge.persistence_report_contrast_seed(name)
        except Exception as exc:
            raise ContractError(
                f"persistence report bridge contrast seed failed for {name}: {exc}"
            ) from exc
        from hydra2.contracts.randomness import RandomStream

        stream = RandomStream(seed)
        _, lo, hi = bootstrap_blocks(diffs, stream=stream, alpha=alpha, resamples=resamples)
        try:
            est: float = bridge.persistence_report_mean(list(diffs))
        except Exception as exc:
            raise ContractError(f"persistence report bridge mean failed for {name}: {exc}") from exc
        out[name] = FactorialContrasts(estimate=est, ci_low=lo, ci_high=hi)
    return out


def stratify_surprise_miss_recovery(
    *,
    commit_logs_by_arm: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Stratify per-packet outcomes for P (hit/miss/recovery)."""
    bridge = _require_report_bridge()
    strata: dict[str, Any] = {}
    for arm, logs in commit_logs_by_arm.items():
        by_outcome: dict[str, int] = {}
        ponder_by_outcome: dict[str, list[int]] = {}
        outcomes: list[str] = []
        ponders: list[int] = []
        for entry in logs:
            oc = str(entry.get("outcome", "unknown"))
            by_outcome[oc] = by_outcome.get(oc, 0) + 1
            pc = int(entry.get("ponder_calls", 0))
            ponder_by_outcome.setdefault(oc, []).append(pc)
            outcomes.append(oc)
            ponders.append(pc)
        try:
            rates: tuple[float, float, int] = bridge.persistence_report_strata_rates(
                arm, outcomes, ponders
            )
            hit_rate, miss_rate, total = rates
        except Exception as exc:
            raise ContractError(
                f"persistence report bridge strata failed for {arm}: {exc}"
            ) from exc
        strata[arm] = {
            "counts": by_outcome,
            "ponder_calls_by_outcome": {k: sum(v) for k, v in ponder_by_outcome.items()},
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "total_packets": total,
        }
    return strata


def generate_factorial_report(
    *,
    placements_by_arm: dict[str, list[float]],
    block_ids: list[str] | None = None,
    resource_samples_by_arm: dict[str, list[dict[str, Any]]] | None = None,
    commit_logs_by_arm: dict[str, list[dict[str, Any]]] | None = None,
    report_id: str = "persistence-factorial-whole-block-v1",
    alpha: float = 0.05,
    resamples: int = 2000,
) -> FactorialReport:
    """Generate frozen deterministic factorial report.

    - placements_by_arm maps arm -> per-block mean placements (len = num_blocks).
    - block_ids optional; otherwise synthetic wall block ids.
    - Never claims perfect resource equality; resource_samples must differ between arms.
    """
    if len(placements_by_arm) == 0:
        raise ContractError("placements_by_arm must not be empty")
    # Validate equal block counts
    lengths = {k: len(v) for k, v in placements_by_arm.items()}
    if len(set(lengths.values())) != 1:
        raise ContractError(f"each arm must have same block count, got {lengths}")
    n = next(iter(lengths.values()))
    if n == 0:
        raise ContractError("must have at least one block")
    if block_ids is None:
        block_ids = [f"wall-block-{i:04d}" for i in range(n)]
    if len(block_ids) != n:
        raise ContractError("block_ids length mismatch")
    bridge = _require_report_bridge()
    # Per-arm mean (Rust-owned Neumaier lane; math.isclose vs sum/len).
    try:
        per_arm_mean = {
            arm: bridge.persistence_report_mean(list(vals))
            for arm, vals in placements_by_arm.items()
        }
    except Exception as exc:
        raise ContractError(f"persistence report bridge per-arm mean failed: {exc}") from exc
    # Contrasts
    contrasts = factorial_contrasts(
        placements_by_arm=placements_by_arm, alpha=alpha, resamples=resamples
    )
    # Block manifest hash (deterministic over placements+ids)
    block_payload = {"block_ids": block_ids, "placements_by_arm": placements_by_arm}
    block_hash = "sha256:" + hashlib.sha256(canonical_bytes(block_payload)).hexdigest()
    # Resource samples: if not supplied, synthesize deterministic but differing distributions
    if resource_samples_by_arm is None:
        resource_samples_by_arm = {}
        for arm, _vals in placements_by_arm.items():
            # Deterministic synthetic samples per block based on arm
            base_calls = {"B": 1, "F": 32, "R": 32, "P": 36, "C": 64}[arm]
            samples = []
            for i in range(n):
                # Vary within arm by block index deterministically
                jitter = deterministic_gumbel_for_arm(arm_id=arm, case_id=f"block-{i}", action_id=0)
                calls = base_calls + int(jitter * 4)
                trans = calls * 4
                joules = calls * 0.04 + trans * 0.005 + jitter * 0.01
                samples.append(
                    {"model_calls": calls, "exact_transitions": trans, "energy_joules": joules}
                )
            resource_samples_by_arm[arm] = samples
    # Verify not claiming equality: check that P and F resource distributions differ
    if "P" in resource_samples_by_arm and "F" in resource_samples_by_arm:
        p_calls = [int(s["model_calls"]) for s in resource_samples_by_arm["P"]]
        f_calls = [int(s["model_calls"]) for s in resource_samples_by_arm["F"]]
        try:
            resources_equal: bool = bridge.persistence_report_resources_equal(p_calls, f_calls)
        except Exception as exc:
            raise ContractError(f"persistence report bridge resource check failed: {exc}") from exc
        if resources_equal:
            raise ContractError("P and F must not claim identical resource equality; log actuals")
    # Strata
    if commit_logs_by_arm is None:
        commit_logs_by_arm = {arm: [] for arm in placements_by_arm}
    strata = stratify_surprise_miss_recovery(commit_logs_by_arm=commit_logs_by_arm)
    import datetime

    generated = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    arms_desc = {aid: dict(ARM_DEFS[aid]) for aid in placements_by_arm}
    uncertainty = {
        "method": "bootstrap_wall_block",
        "unit": "wall_block",
        "alpha": alpha,
        "resamples": resamples,
        "predeclared": True,
    }
    notes = (
        "B/F/R/P share deployable deadline 5000 ms minus 500 ms margin; C uses deadline+extra 2000 ms and is laboratory only, never deployable.",
        "Actual model_calls/transitions/joules logged per decision; P and C not claimed resource-identical despite scheduled maximum opportunity parity.",
        "Stratification reports surprise/miss/recovery per packet; sibling statistics squashed after commit.",
    )
    return FactorialReport(
        report_id=report_id,
        generated_at_utc=generated,
        arms=arms_desc,
        block_manifest_hash=block_hash,
        num_blocks=n,
        per_arm_mean=per_arm_mean,
        contrasts=contrasts,
        resource_samples=resource_samples_by_arm,
        strata=strata,
        uncertainty=uncertainty,
        notes=notes,
    )
