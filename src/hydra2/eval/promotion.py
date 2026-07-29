"""SPEC 18.4 promotion record — retained for every candidate outcome.

All Candidates 0-6 retain records including failure/rejection; the record is
an immutable value object over validated fields, digest-identified via
:func:`promotion_digest` so registries can pin outcomes by content.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Literal

from hydra2.artifacts.digest import of_canonical, validate_digest
from hydra2.contracts.common import ContractError, DigestText

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

UNCERTAINTY_UNITS: tuple[str, ...] = (
    "case",
    "iid_pair",
    "wall_block",
    "smc_population",
    "rqmc_scramble",
    "game_cluster",
)

_GATE_VALUES = ("passed", "failed", "not_applicable")
_DISPOSITIONS = ("promoted", "rejected", "blocked")


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    """SPEC 18.4 promotion record; field order matches the specification."""

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


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a sha256 digest string")
    return str(validate_digest(value))


def make_promotion_record(**kwargs: object) -> PromotionRecord:
    """Validate and construct a :class:`PromotionRecord`."""
    names = [item.name for item in fields(PromotionRecord)]
    unknown = set(kwargs) - set(names)
    if len(unknown) != 0:
        raise ContractError(f"unknown PromotionRecord fields: {sorted(unknown)}")
    missing = [name for name in names if name not in kwargs]
    if len(missing) != 0:
        raise ContractError(f"missing PromotionRecord fields: {missing}")

    comparators = kwargs["comparator_spec_hashes"]
    if not isinstance(comparators, tuple) or not all(
        isinstance(item, str)
        for item in comparators
    ):
        raise ContractError("comparator_spec_hashes must be a tuple of digest strings")
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
        or not math.isfinite(
            float(estimate)
        )
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
    low, high = float(bounds[0]), float(bounds[1])  # type: ignore[arg-type]
    if low > high:
        raise ContractError("confidence_bounds must be ordered (low <= high)")

    gates = kwargs["gates"]
    if not isinstance(gates, Mapping):
        raise ContractError("gates must be a mapping")
    for gate_name, gate_value in gates.items():
        if not isinstance(gate_name, str) or gate_name == "":
            raise ContractError("gate names must be nonempty strings")
        if not isinstance(gate_value, str) or gate_value not in _GATE_VALUES:
            raise ContractError(f"gate {gate_name!r} value {gate_value!r} not in {_GATE_VALUES}")
    disposition = kwargs["disposition"]
    if disposition not in _DISPOSITIONS:
        raise ContractError(f"disposition {disposition!r} not in {_DISPOSITIONS}")

    return PromotionRecord(
        candidate_spec_hash=_require_digest("candidate_spec_hash", kwargs["candidate_spec_hash"]),
        utility_manifest_hash=_require_digest(
            "utility_manifest_hash", kwargs["utility_manifest_hash"]
        ),
        comparator_spec_hashes=comparators_t,
        case_manifest_hash=_require_digest("case_manifest_hash", kwargs["case_manifest_hash"]),
        result_table_hash=_require_digest("result_table_hash", kwargs["result_table_hash"]),
        resource_view=resource_view,
        uncertainty_unit=unit,  # type: ignore[arg-type]
        pass_inequality=inequality,
        observed_estimate=float(estimate),
        confidence_bounds=(low, high),
        gates=dict(gates),
        disposition=disposition,
    )


def record_to_json(record: PromotionRecord) -> dict[str, object]:
    """Canonical JSON projection (gates as a plain sorted mapping)."""
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
    }


def promotion_digest(record: PromotionRecord) -> DigestText:
    """Content identity of a promotion decision."""
    return of_canonical(record_to_json(record))
