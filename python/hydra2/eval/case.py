"""Evaluation case declarations.

Uncertainty-unit binding: case for independent decision cases, iid_pair for
paired natural confirmations, wall_block for duplicate matches,
smc_population for independent controlled-SMC populations, rqmc_scramble for
independent scrambles, game_cluster only for held-out model/calibration
metrics.

A case fixes the primary contrast, the two opaque arm labels, and the
uncertainty unit BEFORE results exist. The unit vocabulary is the frozen
literal list (case, iid_pair, wall_block, smc_population, rqmc_scramble,
game_cluster); ``game_cluster`` is legal only for held-out model/calibration
diagnostics ("``game_cluster`` only for held-out model/calibration metrics")
and is rejected for confirmation cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import validate_digest
from hydra2.contracts.common import ContractError, DigestText
from hydra2.eval.promotion import UNCERTAINTY_UNITS


def _eval_fn(name: str) -> Any | None:
    """Bridge leaf on ``hydra2._native.eval`` or ``None`` when stale."""
    try:
        from hydra2 import _native as _ext  # pyrefly: ignore[missing-import]
    except ImportError:
        return None
    sub = getattr(_ext, "eval", None)
    return getattr(sub, name, None) if sub is not None else None


__all__ = [
    "PRIMARY_METRIC",
    "EvalCase",
    "case_manifest_hash",
    "make_eval_case",
]

#: Declared primary block outcome contrast (declared expected-final-placement
#: contrast; bridge-owned single source).
PRIMARY_METRIC: str = _bridge_contracts.EVAL_PRIMARY_METRIC


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
    """Validate and construct an :class:`EvalCase`.

    Field gates delegate to ``hydra2._native.contracts`` (``eval_check_*``);
    this function keeps the :class:`ContractError` shaping, the
    :class:`EvalCase` construction, and the ``__all__`` names.
    """
    try:
        case_id: str = _bridge_contracts.eval_check_case_id(case_id)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    try:
        arms: tuple[str, str] = _bridge_contracts.eval_check_arms(arms)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    try:
        uncertainty_unit: str = _bridge_contracts.eval_check_uncertainty_unit(
            uncertainty_unit, diagnostic_only, list(UNCERTAINTY_UNITS)
        )
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    return EvalCase(
        case_id=case_id,
        arms=(arms[0], arms[1]),
        primary_metric=PRIMARY_METRIC,
        uncertainty_unit=uncertainty_unit,
        rules_hash=str(validate_digest(rules_hash)),
        diagnostic_only=diagnostic_only,
    )


def eval_case_to_json(case: EvalCase) -> dict[str, object]:
    """Canonical JSON projection (bridge-first, byte-identical fallback)."""
    bridge = _eval_fn("eval_case_to_json")
    if bridge is not None:
        try:
            raw_projection: dict[str, object] = bridge(
                case.case_id,
                list(case.arms),
                case.primary_metric,
                case.uncertainty_unit,
                case.rules_hash,
                case.diagnostic_only,
            )
            return dict(raw_projection)
        except (ImportError, AttributeError):
            pass
    return {
        "case_id": case.case_id,
        "arms": list(case.arms),
        "primary_metric": case.primary_metric,
        "uncertainty_unit": case.uncertainty_unit,
        "rules_hash": case.rules_hash,
        "diagnostic_only": case.diagnostic_only,
    }


def case_manifest_hash(cases: tuple[EvalCase, ...]) -> DigestText:
    """Digest binding the committed case set (order-sensitive, pre-results).

    Thin bridge delegate: ``hydra2._native.contracts.eval_case_manifest_hash_from_docs``
    seals the ``eval_case_to_json`` projection through the feed canon/digest owners.
    """
    docs = [eval_case_to_json(case) for case in cases]
    hex_text: str = _bridge_contracts.eval_case_manifest_hash_from_docs(docs)
    return DigestText(hex_text)
