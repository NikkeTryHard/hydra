"""WP-03B gate: SPEC 18.4 promotion records and evaluation case declarations."""

from __future__ import annotations

import pytest

from hydra2.contracts.common import ContractError
from hydra2.eval.case import PRIMARY_METRIC, case_manifest_hash, make_eval_case
from hydra2.eval.promotion import (
    UNCERTAINTY_UNITS,
    make_promotion_record,
    promotion_digest,
)

pytestmark = pytest.mark.contract_package("WP-03B")

_H = "sha256:" + "ab" * 32


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "candidate_spec_hash": _H,
        "utility_manifest_hash": _H,
        "comparator_spec_hashes": (_H,),
        "case_manifest_hash": _H,
        "result_table_hash": _H,
        "resource_view": "cuda_eager",
        "uncertainty_unit": "wall_block",
        "pass_inequality": "mean_contrast > 0",
        "observed_estimate": 0.42,
        "confidence_bounds": (0.05, 0.79),
        "gates": {"seat_balance": "passed", "coverage": "passed"},
        "disposition": "promoted",
    }
    base.update(overrides)
    return base


def test_promotion_record_round_trip_and_digest_stability() -> None:
    record = make_promotion_record(**_record())
    assert record.disposition == "promoted"
    assert promotion_digest(record) == promotion_digest(make_promotion_record(**_record()))
    changed = make_promotion_record(**_record(disposition="rejected"))
    assert promotion_digest(record) != promotion_digest(changed)


def test_uncertainty_units_are_the_spec_literal_list() -> None:
    assert UNCERTAINTY_UNITS == (
        "case",
        "iid_pair",
        "wall_block",
        "smc_population",
        "rqmc_scramble",
        "game_cluster",
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_spec_hash": "sha256:XYZ"},
        {"uncertainty_unit": "per_decision"},
        {"disposition": "maybe"},
        {"gates": {"g": "excellent"}},
        {"confidence_bounds": (1.0, -1.0)},
        {"confidence_bounds": (0.0, float("nan"))},
        {"observed_estimate": float("inf")},
        {"resource_view": ""},
        {"missing_field": True},
    ],
)
def test_promotion_record_rejects_invalid_payloads(overrides: dict[str, object]) -> None:
    with pytest.raises(ContractError):
        make_promotion_record(**_record(**overrides))


def test_case_declaration_binds_primary_metric_and_unit() -> None:
    case = make_eval_case(
        case_id="confirm-a-vs-b",
        arms=("cand-a", "base-b"),
        rules_hash=_H,
        uncertainty_unit="wall_block",
    )
    assert case.primary_metric == PRIMARY_METRIC == "expected_final_placement_contrast"
    manifest = case_manifest_hash((case,))
    assert manifest == case_manifest_hash(
        (
            make_eval_case(
                case_id="confirm-a-vs-b",
                arms=("cand-a", "base-b"),
                rules_hash=_H,
                uncertainty_unit="wall_block",
            ),
        )
    )
    other = make_eval_case(
        case_id="other",
        arms=("cand-a", "base-b"),
        rules_hash=_H,
        uncertainty_unit="wall_block",
    )
    assert manifest != case_manifest_hash((other,))


def test_game_cluster_reserved_for_diagnostics() -> None:
    with pytest.raises(ContractError, match="game_cluster"):
        make_eval_case(
            case_id="confirm-diag",
            arms=("a", "b"),
            rules_hash=_H,
            uncertainty_unit="game_cluster",
        )
    diagnostic = make_eval_case(
        case_id="heldout-calibration",
        arms=("model", "calib"),
        rules_hash=_H,
        uncertainty_unit="game_cluster",
        diagnostic_only=True,
    )
    assert diagnostic.diagnostic_only is True
