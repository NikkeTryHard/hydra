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
        "schedule_hash": _H,
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


def test_promotion_record_rejects_malformed_digest_binding() -> None:
    with pytest.raises(ValueError):
        make_promotion_record(**_record(candidate_spec_hash="sha256:XYZ"))


def test_promotion_record_missing_required_field_raises() -> None:
    record = _record()
    del record["disposition"]
    with pytest.raises(ContractError, match="missing"):
        make_promotion_record(**record)


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


def test_promotion_record_additive_bindings_default() -> None:
    # PR4: promoted requires schedule binding; rejected stays loose.
    from hydra2.eval.promotion import record_to_json

    record = make_promotion_record(**_record())
    assert record.schedule_hash == _H
    assert record.environment_hash is None
    assert record.excluded_blocks == ()
    projected = record_to_json(record)
    assert projected["schedule_hash"] == _H
    assert projected["environment_hash"] is None
    assert projected["excluded_blocks"] == []
    assert promotion_digest(record) == promotion_digest(make_promotion_record(**_record()))
    loose = make_promotion_record(**_record(disposition="rejected", schedule_hash=None))
    assert loose.schedule_hash is None
    with pytest.raises(ContractError, match="schedule_hash"):
        make_promotion_record(**_record(schedule_hash=None))


def test_promotion_record_additive_bindings_bound() -> None:
    # PR4: bound schedule/env hashes + exclusions verify and digest-bind.
    from hydra2.eval.blocks import ExcludedBlock
    from hydra2.eval.promotion import record_to_json

    excluded = (ExcludedBlock(wall_id="w0001", reason="timeout", detail="t>budget"),)
    record = make_promotion_record(
        **_record(
            schedule_hash=_H,
            environment_hash=_H,
            excluded_blocks=excluded,
        )
    )
    assert record.schedule_hash == _H
    assert record.environment_hash == _H
    assert record.excluded_blocks == excluded
    projected = record_to_json(record)
    assert projected["excluded_blocks"] == [
        {"wall_id": "w0001", "reason": "timeout", "detail": "t>budget"}
    ]
    assert promotion_digest(record) != promotion_digest(make_promotion_record(**_record()))


def test_promotion_record_rejects_bad_bindings() -> None:
    with pytest.raises(ValueError):
        make_promotion_record(**_record(schedule_hash="sha256:XYZ"))
    with pytest.raises(ContractError):
        make_promotion_record(**_record(excluded_blocks=("not-a-block",)))


def test_promoted_without_schedule_commitment_raises() -> None:
    """Promotion binding: promoted needs a schedule commitment; rejected stays loose."""
    with pytest.raises(ContractError, match="schedule_hash"):
        make_promotion_record(**_record(schedule_hash=None))
    missing = _record()
    del missing["schedule_hash"]
    with pytest.raises(ContractError, match="schedule_hash"):
        make_promotion_record(**missing)
    loose = make_promotion_record(**_record(disposition="rejected", schedule_hash=None))
    assert loose.schedule_hash is None
    missing_loose = _record(disposition="rejected")
    del missing_loose["schedule_hash"]
    assert make_promotion_record(**missing_loose).schedule_hash is None


def test_promotion_schedule_hash_binds_commitment_not_walls_hash() -> None:
    """Binding: commitment covers every schedule facet; walls_hash does not."""
    from hydra2.eval.schedule import build_match_schedule, schedule_commitment_hash

    wall_ids = ("w-01", "w-02", "w-03")
    schedule = build_match_schedule(
        wall_ids=wall_ids,
        labels=("candidate-a", "partner-a", "baseline-b", "field-c"),
        rules_hash="sha256:" + "ab" * 32,
        master_seed=bytes(range(48, 80)),
        experiment_id="exp-wp03b",
        split_id="split-wp03b",
    )
    commitment = str(schedule_commitment_hash(schedule))
    walls_hash = str(schedule.walls_hash)
    assert commitment != walls_hash
    # Same walls, different rules: walls_hash is blind, commitment binds.
    drifted = build_match_schedule(
        wall_ids=wall_ids,
        labels=("candidate-a", "partner-a", "baseline-b", "field-c"),
        rules_hash="sha256:" + "cd" * 32,
        master_seed=bytes(range(48, 80)),
        experiment_id="exp-wp03b",
        split_id="split-wp03b",
    )
    assert str(drifted.walls_hash) == walls_hash
    drifted_commitment = str(schedule_commitment_hash(drifted))
    assert drifted_commitment != commitment
    record = make_promotion_record(**_record(schedule_hash=commitment))
    assert record.schedule_hash == commitment
    drifted_record = make_promotion_record(**_record(schedule_hash=drifted_commitment))
    assert promotion_digest(record) != promotion_digest(drifted_record)
