"""Promotion record — retained for every candidate outcome.

All Candidates 0-6 retain records including failure/rejection; candidate
selection and natural confirmation use disjoint semantic streams; a promotion
that cannot name its schedule, machine, and excluded walls is not
reproducible and MUST NOT issue disposition promoted.

The record is an immutable value object over validated fields,
digest-identified via :func:`promotion_digest` so registries can pin outcomes
by content.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Literal

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import validate_digest
from hydra2.contracts.common import ContractError, DigestText
from hydra2.eval.blocks import ExcludedBlock


def _eval_fn(name: str) -> Any | None:
    """Bridge leaf on ``hydra2._native.eval`` or ``None`` when stale."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError:
        return None
    sub = getattr(_ext, "eval", None)
    return getattr(sub, name, None) if sub is not None else None


__all__ = [
    "UNCERTAINTY_UNITS",
    "PromotionRecord",
    "make_promotion_record",
    "promotion_digest",
]

UncertaintyUnit = Literal[
    "case",
    "iid_pair",
    "wall_block",
    "smc_population",
    "rqmc_scramble",
    "game_cluster",
]

#: Uncertainty-unit vocabulary (case, iid_pair, wall_block, smc_population,
#: rqmc_scramble, game_cluster — game_cluster only for held-out
#: model/calibration; bridge-owned single source).
UNCERTAINTY_UNITS: tuple[str, ...] = _bridge_contracts.PROMOTION_UNCERTAINTY_UNITS
_GATE_VALUES: tuple[str, ...] = _bridge_contracts.PROMOTION_GATE_VALUES
_DISPOSITIONS: tuple[str, ...] = _bridge_contracts.PROMOTION_DISPOSITIONS


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    """Promotion record; frozen field order (candidate_spec_hash,
    utility_manifest_hash, comparator_spec_hashes, case_manifest_hash,
    result_table_hash, resource_view, uncertainty_unit, pass_inequality,
    observed_estimate, confidence_bounds, gates, disposition, plus optional
    schedule_hash, environment_hash, excluded_blocks covered by the digest).

    ``schedule_hash`` is the schedule commitment hash
    (:func:`hydra2.eval.schedule.schedule_commitment_hash`, binding every
    schedule facet — walls, seats, latency, rules, seed protocol), never a
    bare ``walls_hash``. ``make_block_manifest`` schedule hashes are
    consumed as-is for wall provenance; promotion binding is the commitment.
    """

    candidate_spec_hash: str
    utility_manifest_hash: str
    comparator_spec_hashes: tuple[str, ...]
    case_manifest_hash: str
    result_table_hash: str
    resource_view: str
    uncertainty_unit: UncertaintyUnit
    pass_inequality: str
    observed_estimate: float
    confidence_bounds: tuple[float, float]
    gates: Mapping[str, Literal["passed", "failed", "not_applicable"]]
    disposition: Literal["promoted", "rejected", "blocked"]
    schedule_hash: str | None = None
    environment_hash: str | None = None
    excluded_blocks: tuple[ExcludedBlock, ...] = ()


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a sha256 digest string")
    return str(validate_digest(value))


def make_promotion_record(**kwargs: object) -> PromotionRecord:
    """Validate and construct a :class:`PromotionRecord`.

    ``promoted`` is fail-closed: every gate must be ``"passed"`` (nonempty
    gate set) and ``schedule_hash`` must bind the schedule commitment
    (:func:`hydra2.eval.schedule.schedule_commitment_hash`, never a bare
    ``walls_hash``). ``rejected`` and ``blocked`` stay loose so failures
    remain recordable without a binding.
    """
    names = [item.name for item in fields(PromotionRecord)]
    unknown = set(kwargs) - set(names)
    if len(unknown) != 0:
        raise ContractError(f"unknown PromotionRecord fields: {sorted(unknown)}")
    missing = [name for name in names if name not in kwargs]
    optional = {"schedule_hash", "environment_hash", "excluded_blocks"}
    missing = [name for name in missing if name not in optional]
    if len(missing) != 0:
        raise ContractError(f"missing PromotionRecord fields: {missing}")

    comparators = kwargs["comparator_spec_hashes"]
    if not isinstance(comparators, tuple) or not all(isinstance(item, str) for item in comparators):
        raise ContractError("comparator_spec_hashes must be a tuple of digest strings")
    # reason: type arg-type on kwargs object narrowed by the tuple-of-str
    # guard above; _require_digest runtime-validates each digest.
    comparators_t = tuple(_require_digest("comparator entry", item) for item in comparators)  # type: ignore[arg-type]

    resource_view = kwargs["resource_view"]
    if not isinstance(resource_view, str) or resource_view == "":
        raise ContractError("resource_view must be a nonempty str")
    unit = kwargs["uncertainty_unit"]
    if unit not in UNCERTAINTY_UNITS:
        raise ContractError(f"uncertainty_unit {unit!r} not in {UNCERTAINTY_UNITS}")
    inequality = kwargs["pass_inequality"]
    if not isinstance(inequality, str) or inequality == "":
        raise ContractError("pass_inequality must be a nonempty str")

    estimate = kwargs["observed_estimate"]
    if (
        isinstance(estimate, bool)
        or not isinstance(estimate, (int, float))
        or not math.isfinite(float(estimate))
    ):
        raise ContractError("observed_estimate must be a finite number")
    bounds = kwargs["confidence_bounds"]
    if (
        not isinstance(bounds, tuple)
        or len(bounds) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in bounds
        )
    ):
        raise ContractError("confidence_bounds must be two finite floats")
    # reason: type arg-type on object-typed bounds elements; tuple of two
    # finite numbers validated just above, float() coerces.
    low, high = float(bounds[0]), float(bounds[1])  # type: ignore[arg-type]
    if low > high:
        raise ContractError("confidence_bounds must be ordered (low <= high)")

    gates = kwargs["gates"]
    if not isinstance(gates, Mapping):
        raise ContractError("gates must be a mapping")
    try:
        _bridge_contracts.promotion_check_gates(dict(gates))
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    disposition = kwargs["disposition"]
    try:
        checked_disposition: Literal["promoted", "rejected", "blocked"] = (
            _bridge_contracts.promotion_check_disposition(disposition)
        )
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    disposition = checked_disposition
    schedule_hash = kwargs.get("schedule_hash")
    if schedule_hash is not None:
        schedule_hash = _require_digest("schedule_hash", schedule_hash)
    try:
        _bridge_contracts.promotion_check_promoted(
            dict(gates), schedule_hash is not None, disposition == "promoted"
        )
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    environment_hash = kwargs.get("environment_hash")
    if environment_hash is not None:
        environment_hash = _require_digest("environment_hash", environment_hash)
    excluded_blocks = kwargs.get("excluded_blocks", ())
    if not isinstance(excluded_blocks, tuple) or not all(
        isinstance(item, ExcludedBlock) for item in excluded_blocks
    ):
        raise ContractError("excluded_blocks must be a tuple of ExcludedBlock")

    return PromotionRecord(
        candidate_spec_hash=_require_digest("candidate_spec_hash", kwargs["candidate_spec_hash"]),
        utility_manifest_hash=_require_digest(
            "utility_manifest_hash", kwargs["utility_manifest_hash"]
        ),
        comparator_spec_hashes=comparators_t,
        case_manifest_hash=_require_digest("case_manifest_hash", kwargs["case_manifest_hash"]),
        result_table_hash=_require_digest("result_table_hash", kwargs["result_table_hash"]),
        resource_view=resource_view,
        # reason: type arg-type on object-typed unit; membership in
        # UNCERTAINTY_UNITS validated above, Literal narrowing is runtime.
        uncertainty_unit=unit,  # type: ignore[arg-type]
        pass_inequality=inequality,
        observed_estimate=float(estimate),
        confidence_bounds=(low, high),
        gates=dict(gates),
        disposition=disposition,
        schedule_hash=schedule_hash,
        environment_hash=environment_hash,
        excluded_blocks=excluded_blocks,
    )


def record_to_json(record: PromotionRecord) -> dict[str, object]:
    """Canonical JSON projection (gates as a plain sorted mapping)."""
    bridge = _eval_fn("record_to_json")
    if bridge is not None:
        try:
            raw_projection: dict[str, object] = bridge(
                record.candidate_spec_hash,
                record.utility_manifest_hash,
                list(record.comparator_spec_hashes),
                record.case_manifest_hash,
                record.result_table_hash,
                record.resource_view,
                record.uncertainty_unit,
                record.pass_inequality,
                record.observed_estimate,
                tuple(record.confidence_bounds),
                dict(record.gates),
                record.disposition,
                record.schedule_hash,
                record.environment_hash,
                [(b.wall_id, b.reason, b.detail) for b in record.excluded_blocks],
            )
            return dict(raw_projection)
        except (ImportError, AttributeError):
            pass
    return {
        "candidate_spec_hash": record.candidate_spec_hash,
        "utility_manifest_hash": record.utility_manifest_hash,
        "comparator_spec_hashes": list(record.comparator_spec_hashes),
        "case_manifest_hash": record.case_manifest_hash,
        "result_table_hash": record.result_table_hash,
        "resource_view": record.resource_view,
        "uncertainty_unit": record.uncertainty_unit,
        "pass_inequality": record.pass_inequality,
        "observed_estimate": record.observed_estimate,
        "confidence_bounds": list(record.confidence_bounds),
        "gates": dict(sorted(record.gates.items())),
        "disposition": record.disposition,
        "schedule_hash": record.schedule_hash,
        "environment_hash": record.environment_hash,
        "excluded_blocks": [
            {"wall_id": item.wall_id, "reason": item.reason, "detail": item.detail}
            for item in record.excluded_blocks
        ],
    }


def promotion_digest(record: PromotionRecord) -> DigestText:
    """Content identity of a promotion decision."""
    hex_text: str = _bridge_contracts.promotion_digest_from_doc(record_to_json(record))
    return DigestText(hex_text)
