"""Evaluation case declarations (SPEC 18.3/18.4 unit binding).

A case fixes the primary contrast, the two opaque arm labels, and the
uncertainty unit BEFORE results exist. The unit vocabulary is the SPEC 18.4
literal list; ``game_cluster`` is legal only for held-out model/calibration
diagnostics (SPEC 18.4: "``game_cluster`` only for held-out
model/calibration metrics") and is rejected for confirmation cases.
"""

from __future__ import annotations

from dataclasses import dataclass

from hydra2.artifacts.digest import of_canonical, validate_digest
from hydra2.contracts.common import ContractError, DigestText
from hydra2.eval.promotion import UNCERTAINTY_UNITS

__all__ = [
    "PRIMARY_METRIC",
    "EvalCase",
    "case_manifest_hash",
    "make_eval_case",
]

#: Declared primary block outcome contrast (SPEC 18.3).
PRIMARY_METRIC = "expected_final_placement_contrast"


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One declared evaluation contrast with its uncertainty unit."""

    case_id: str
    arms: tuple[str, str]
    primary_metric: str
    uncertainty_unit: str
    rules_hash: str
    diagnostic_only: bool


def make_eval_case(
    *,
    case_id: str,
    arms: tuple[str, str],
    rules_hash: str,
    uncertainty_unit: str,
    diagnostic_only: bool = False,
) -> EvalCase:
    """Validate and construct an :class:`EvalCase`."""
    if not isinstance(case_id, str) or case_id == "":
        raise ContractError("case_id must be a nonempty str")
    if (
        not isinstance(arms, tuple)
        or len(arms) != 2
        or any(not isinstance(arm, str) or arm == "" for arm in arms)
    ):
        raise ContractError("arms must be two nonempty opaque labels")
    if arms[0] == arms[1]:
        raise ContractError("arms must be distinct")
    if uncertainty_unit not in UNCERTAINTY_UNITS:
        raise ContractError(f"uncertainty_unit {uncertainty_unit!r} not in {UNCERTAINTY_UNITS}")
    if uncertainty_unit == "game_cluster" and not diagnostic_only:
        raise ContractError(
            "uncertainty_unit 'game_cluster' is reserved for held-out "
            "model/calibration diagnostics; pass diagnostic_only=True"
        )
    return EvalCase(
        case_id=case_id,
        arms=(arms[0], arms[1]),
        primary_metric=PRIMARY_METRIC,
        uncertainty_unit=uncertainty_unit,
        rules_hash=str(validate_digest(rules_hash)),
        diagnostic_only=diagnostic_only,
    )


def eval_case_to_json(case: EvalCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "arms": list(case.arms),
        "primary_metric": case.primary_metric,
        "uncertainty_unit": case.uncertainty_unit,
        "rules_hash": case.rules_hash,
        "diagnostic_only": case.diagnostic_only,
    }


def case_manifest_hash(cases: tuple[EvalCase, ...]) -> DigestText:
    """Digest binding the committed case set (order-sensitive, pre-results)."""
    return of_canonical([eval_case_to_json(case) for case in cases])
