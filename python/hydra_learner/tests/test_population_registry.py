from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Literal, cast

import pytest

from hydra_learner.checkpoint_eval import PairedCheckpointEvalThresholds, build_paired_checkpoint_eval_summary
from hydra_learner.population_registry import (
    CheckpointEntry,
    EvalSchedule,
    OpponentPool,
    PopulationRegistry,
    PromotionDecisionRecord,
    PromotionRecord,
    SeedBank,
    build_promotion_record,
    build_seed_bank,
    checkpoint_entry_to_dict,
    pin_mutable_checkpoint_for_registry,
    register_immutable_checkpoint,
    registry_to_dict,
    write_json_file,
    write_promotion_artifact,
)


def test_seed_bank_serializes_exact_seeds_and_eval_config(tmp_path: Path) -> None:
    seed_bank = build_seed_bank(
        seed_set_id="promotion_v1",
        seeds=(101, 202, 303),
        games_per_seed=4,
        temperature=0.0,
        arena_options={"device": "cuda:0", "native": True, "arena_batch_decisions": 1024},
    )
    registry = _registry(tmp_path, seed_bank=seed_bank)

    payload = registry_to_dict(registry)

    assert payload["seed_banks"] == {
        "promotion_v1": {
            "arena_options": {"arena_batch_decisions": 1024, "device": "cuda:0", "native": True},
            "games_per_seed": 4,
            "seed_set_id": "promotion_v1",
            "seeds": [101, 202, 303],
            "temperature": 0.0,
        }
    }
    assert payload["eval_schedule"] == {
        "enabled": True,
        "min_games": 12,
        "opponent_pool_id": "active_baseline",
        "seed_set_id": "promotion_v1",
        "thresholds": {"max_fourth_rate_delta": 0.0, "min_games": 12, "min_mean_u_a_delta": 0.0},
    }


def test_paired_seed_provenance_is_persisted_and_reconstructable(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    summary = _checkpoint_eval_summary("baseline", "candidate", games=16, seed=11)
    arena_path = _arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate)

    record = build_promotion_record(
        registry=registry,
        promotion_id="promotion-1",
        candidate_checkpoint_id="candidate",
        baseline_checkpoint_id="baseline",
        opponent_pool_id="active_baseline",
        seed_set_id="promotion_v1",
        arena_summary_path=arena_path,
        checkpoint_eval_summary=summary,
        paired_eval_summary_path=arena_path,
    )
    path = write_promotion_artifact(tmp_path, record)

    payload = json.loads(path.read_text(encoding="utf-8"))
    registry_payload = registry_to_dict(registry)
    seed_banks = cast("dict[str, dict[str, object]]", registry_payload["seed_banks"])
    seed_bank = seed_banks[cast("str", payload["seed_set_id"])]
    assert seed_bank["seeds"] == [11, 22, 33]
    assert payload["normalized_metrics"]["games"] == 16
    assert payload["paired_eval_summary_path"] == str(arena_path.resolve(strict=True))
    assert payload["evidence_seed"] == 11
    assert payload["seat_coverage_verified"] is False
    assert payload["seat_coverage"] == {}


def test_seed_mismatch_rejects_and_seat_coverage_persists(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    arena_path = _arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate, seed=99)

    with pytest.raises(ValueError, match="seed"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-seed-mismatch",
            candidate_checkpoint_id="candidate",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=_checkpoint_eval_summary("baseline", "candidate", seed=11),
        )

    arena_path = _arena_summary_path(
        tmp_path,
        registry.checkpoints["baseline"],
        candidate,
        result={**_arena_metrics(), "seat_rotations": 4, "seat_coverage": [0, 1, 2, 3]},
    )
    record = build_promotion_record(
        registry=registry,
        promotion_id="promotion-seat-coverage",
        candidate_checkpoint_id="candidate",
        baseline_checkpoint_id="baseline",
        opponent_pool_id="active_baseline",
        seed_set_id="promotion_v1",
        arena_summary_path=arena_path,
        checkpoint_eval_summary=_checkpoint_eval_summary("baseline", "candidate"),
    )
    assert record.seat_coverage_verified
    assert record.seat_coverage == {"seat_coverage": [0, 1, 2, 3], "seat_rotations": 4}


def test_register_immutable_checkpoint_with_hash(tmp_path: Path) -> None:
    entry = _checkpoint(tmp_path, "candidate", b"weights", role="candidate")

    payload = checkpoint_entry_to_dict(entry)

    assert entry.path_sha256 == "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"
    assert entry.policy_json_sha256 is None
    assert payload["weight_source"] == "raw"


def test_mutable_latest_without_pin_rejects(tmp_path: Path) -> None:
    latest = tmp_path / "checkpoints" / "latest.pt"
    latest.parent.mkdir()
    latest.write_bytes(b"mutable")

    with pytest.raises(ValueError, match=r"mutable latest\.pt"):
        register_immutable_checkpoint(checkpoint_id="candidate", role="candidate", path=latest, weight_source="raw")

    pinned = pin_mutable_checkpoint_for_registry(checkpoint_id="candidate", source_path=latest, output_dir=tmp_path)
    entry = register_immutable_checkpoint(checkpoint_id="candidate", role="candidate", path=pinned, weight_source="raw")
    assert Path(entry.path).name == "candidate.pt"
    assert entry.path_sha256 is not None


def test_registered_artifact_missing_or_mutated_rejects(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    arena_path = _arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate)
    summary = _checkpoint_eval_summary("baseline", "candidate")

    Path(candidate.path).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-mutated",
            candidate_checkpoint_id="candidate",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=summary,
        )

    candidate = _checkpoint(tmp_path, "candidate-missing", b"candidate", role="candidate")
    registry = _registry(tmp_path).with_checkpoint(candidate)
    arena_path = _arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate)
    Path(candidate.path).unlink()
    with pytest.raises(ValueError, match="artifact missing"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-missing",
            candidate_checkpoint_id="candidate-missing",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=_checkpoint_eval_summary("baseline", "candidate-missing"),
        )


def test_missing_baseline_or_candidate_rejects(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    arena_path = _arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate)
    summary = _checkpoint_eval_summary("baseline", "candidate")

    with pytest.raises(ValueError, match="candidate checkpoint is not registered"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-1",
            candidate_checkpoint_id="candidate",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=summary,
        )

    registry = registry.with_checkpoint(candidate)
    with pytest.raises(ValueError, match="baseline checkpoint does not match"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-1",
            candidate_checkpoint_id="candidate",
            baseline_checkpoint_id="other-baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=summary,
        )


def test_promote_updates_active_champion_only_on_promote(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    record = build_promotion_record(
        registry=registry,
        promotion_id="promotion-1",
        candidate_checkpoint_id="candidate",
        baseline_checkpoint_id="baseline",
        opponent_pool_id="active_baseline",
        seed_set_id="promotion_v1",
        arena_summary_path=_arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate),
        checkpoint_eval_summary=_checkpoint_eval_summary("baseline", "candidate"),
    )

    updated = registry.with_promotion(record)

    assert updated.active_baseline_id == "candidate"
    assert updated.checkpoints["candidate"].role == "champion"
    assert updated.checkpoints["candidate"].status == "promoted"
    pool = updated.opponent_pools["active_baseline"]
    assert pool.baseline_checkpoint_id == "candidate"
    assert pool.opponent_checkpoint_ids == ("candidate",)
    assert pool.checkpoint_ids == ("candidate",)


def test_active_baseline_pool_mismatch_or_unregistered_rejects(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    mismatched_pool = OpponentPool(
        pool_id="active_baseline",
        strategy="active_baseline_only",
        baseline_checkpoint_id="other",
        opponent_checkpoint_ids=("other",),
        checkpoint_ids=("other",),
    )
    mismatched = PopulationRegistry(
        schema_version=registry.schema_version,
        registry_id=registry.registry_id,
        run_id=registry.run_id,
        active_baseline_id=registry.active_baseline_id,
        latest_candidate_id=registry.latest_candidate_id,
        checkpoints=registry.checkpoints,
        seed_banks=registry.seed_banks,
        opponent_pools={"active_baseline": mismatched_pool},
        eval_schedule=registry.eval_schedule,
        promotions=registry.promotions,
    )
    with pytest.raises(ValueError, match="active baseline"):
        mismatched.validate()

    unregistered_pool = OpponentPool(
        pool_id="active_baseline",
        strategy="active_baseline_only",
        baseline_checkpoint_id="ghost",
        opponent_checkpoint_ids=("ghost",),
        checkpoint_ids=("ghost",),
    )
    unregistered = PopulationRegistry(
        schema_version=registry.schema_version,
        registry_id=registry.registry_id,
        run_id=registry.run_id,
        active_baseline_id="ghost",
        latest_candidate_id=registry.latest_candidate_id,
        checkpoints=registry.checkpoints,
        seed_banks=registry.seed_banks,
        opponent_pools={"active_baseline": unregistered_pool},
        eval_schedule=registry.eval_schedule,
        promotions=registry.promotions,
    )
    with pytest.raises(ValueError, match="active baseline checkpoint"):
        unregistered.validate()


def test_same_filename_wrong_path_rejects(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left_path = left_dir / "candidate.pt"
    right_path = right_dir / "candidate.pt"
    left_path.write_bytes(b"left")
    right_path.write_bytes(b"right")
    candidate = register_immutable_checkpoint(
        checkpoint_id="candidate-left",
        role="candidate",
        path=left_path,
        weight_source="raw",
        registered_at_unix_ms=1234,
    )
    registry = registry.with_checkpoint(candidate)
    wrong_entry = register_immutable_checkpoint(
        checkpoint_id="candidate-right",
        role="candidate",
        path=right_path,
        weight_source="raw",
        registered_at_unix_ms=1234,
    )
    arena_path = _arena_summary_path(tmp_path, registry.checkpoints["baseline"], wrong_entry)

    with pytest.raises(ValueError, match="arena candidate identity mismatch"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-wrong-path",
            candidate_checkpoint_id="candidate-left",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=_checkpoint_eval_summary("baseline", "candidate-left"),
        )


def test_checkpoint_eval_must_match_arena_metrics(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    arena_path = _arena_summary_path(
        tmp_path,
        registry.checkpoints["baseline"],
        candidate,
        result={**_arena_metrics(), "candidate_fourth_rate": 0.3, "baseline_fourth_rate": 0.2},
    )

    with pytest.raises(ValueError, match="metric mismatch"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-mismatch",
            candidate_checkpoint_id="candidate",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=_checkpoint_eval_summary("baseline", "candidate"),
        )


def test_forged_promote_fourth_rate_threshold_violation_rejects(tmp_path: Path) -> None:
    _assert_forged_promote_rejects(
        tmp_path,
        {**_arena_metrics(), "candidate_fourth_rate": 0.4, "baseline_fourth_rate": 0.2},
        match="configured registry thresholds",
    )


def test_forged_promote_mean_u_a_threshold_violation_rejects(tmp_path: Path) -> None:
    _assert_forged_promote_rejects(
        tmp_path,
        {**_arena_metrics(), "candidate_mean_placement": 2.6, "baseline_mean_placement": 2.4},
        match="configured registry thresholds",
    )


def test_forged_promote_top2_threshold_violation_rejects_when_configured(tmp_path: Path) -> None:
    _assert_forged_promote_rejects(
        tmp_path,
        {
            **_arena_metrics(),
            "candidate_top2_rate": 0.51,
            "baseline_top2_rate": 0.50,
        },
        thresholds={"min_games": 8, "max_fourth_rate_delta": 0.0, "min_mean_u_a_delta": 0.0, "min_top2_delta": 0.05},
        match="configured registry thresholds",
    )


def test_forged_promote_min_games_violation_rejects(tmp_path: Path) -> None:
    _assert_forged_promote_rejects(
        tmp_path,
        {**_arena_metrics(), "games": 4},
        match="configured registry thresholds",
    )


def test_forged_promote_missing_configured_metric_rejects(tmp_path: Path) -> None:
    _assert_forged_promote_rejects(
        tmp_path,
        {"games": 16},
        match="metric mismatch|missing configured metric gate",
    )


def test_with_promotion_rejects_materialized_forged_threshold_violations(tmp_path: Path) -> None:
    cases: tuple[tuple[dict[str, object], dict[str, float | int | None] | None], ...] = (
        ({**_arena_metrics(), "candidate_fourth_rate": 0.4, "baseline_fourth_rate": 0.2}, None),
        ({**_arena_metrics(), "candidate_mean_placement": 2.6, "baseline_mean_placement": 2.4}, None),
        (
            {**_arena_metrics(), "candidate_top2_rate": 0.51, "baseline_top2_rate": 0.50},
            {"min_games": 8, "max_fourth_rate_delta": 0.0, "min_mean_u_a_delta": 0.0, "min_top2_delta": 0.05},
        ),
        ({**_arena_metrics(), "games": 4}, None),
        ({**_arena_metrics(), "candidate_mean_placement": None}, None),
    )
    for index, (metrics, thresholds) in enumerate(cases):
        registry = _registry(tmp_path)
        candidate = _checkpoint(tmp_path, f"candidate-forged-{index}", b"candidate", role="candidate")
        registry = registry.with_checkpoint(candidate)
        record = _materialized_promotion_record(
            candidate.checkpoint_id,
            _forged_promote_summary("baseline", candidate.checkpoint_id, dict(metrics)),
            thresholds=thresholds,
        )
        with pytest.raises(ValueError, match=r"promotion decision|missing configured metric gate"):
            registry.with_promotion(record)
        assert registry.active_baseline_id == "baseline"


def test_with_promotion_accepts_valid_materialized_promote_and_reject_terminal_records(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    promoted = registry.with_promotion(
        _materialized_promotion_record("candidate", _checkpoint_eval_summary("baseline", "candidate"))
    )
    assert promoted.active_baseline_id == "candidate"

    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate-reject", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    reject_summary = _checkpoint_eval_summary(
        "baseline",
        "candidate-reject",
        arena_metrics={**_arena_metrics(), "candidate_fourth_rate": 0.3, "baseline_fourth_rate": 0.2},
    )
    rejected = registry.with_promotion(_materialized_promotion_record("candidate-reject", reject_summary))
    assert rejected.active_baseline_id == "baseline"
    assert rejected.checkpoints["candidate-reject"].status == "rejected"

    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate-insufficient", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    insufficient_summary = _checkpoint_eval_summary(
        "baseline", "candidate-insufficient", arena_metrics={**_arena_metrics(), "games": 4}
    )
    insufficient = registry.with_promotion(
        _materialized_promotion_record("candidate-insufficient", insufficient_summary)
    )
    assert insufficient.active_baseline_id == "baseline"
    assert insufficient.checkpoints["candidate-insufficient"].status == "rejected"


def test_reject_and_insufficient_games_leave_champion_unchanged(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    arena_path = _arena_summary_path(
        tmp_path,
        registry.checkpoints["baseline"],
        candidate,
        result={**_arena_metrics(), "candidate_fourth_rate": 0.3, "baseline_fourth_rate": 0.2},
    )

    reject_record = build_promotion_record(
        registry=registry,
        promotion_id="promotion-reject",
        candidate_checkpoint_id="candidate",
        baseline_checkpoint_id="baseline",
        opponent_pool_id="active_baseline",
        seed_set_id="promotion_v1",
        arena_summary_path=arena_path,
        checkpoint_eval_summary=_checkpoint_eval_summary(
            "baseline",
            "candidate",
            arena_metrics={**_arena_metrics(), "candidate_fourth_rate": 0.3, "baseline_fourth_rate": 0.2},
        ),
    )
    after_reject = registry.with_promotion(reject_record)
    assert after_reject.active_baseline_id == "baseline"

    insufficient_arena_path = _arena_summary_path(
        tmp_path, registry.checkpoints["baseline"], candidate, result={**_arena_metrics(), "games": 4}
    )
    insufficient_record = build_promotion_record(
        registry=registry,
        promotion_id="promotion-insufficient",
        candidate_checkpoint_id="candidate",
        baseline_checkpoint_id="baseline",
        opponent_pool_id="active_baseline",
        seed_set_id="promotion_v1",
        arena_summary_path=insufficient_arena_path,
        checkpoint_eval_summary=_checkpoint_eval_summary(
            "baseline", "candidate", arena_metrics={**_arena_metrics(), "games": 4}
        ),
    )
    after_insufficient = registry.with_promotion(insufficient_record)
    assert after_insufficient.active_baseline_id == "baseline"
    assert after_insufficient.checkpoints["candidate"].status == "rejected"


def test_fourth_rate_regression_rejects_even_with_positive_score_delta(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    record = build_promotion_record(
        registry=registry,
        promotion_id="promotion-1",
        candidate_checkpoint_id="candidate",
        baseline_checkpoint_id="baseline",
        opponent_pool_id="active_baseline",
        seed_set_id="promotion_v1",
        arena_summary_path=_arena_summary_path(
            tmp_path,
            registry.checkpoints["baseline"],
            candidate,
            result={
                **_arena_metrics(),
                "candidate_fourth_rate": 0.3,
                "baseline_fourth_rate": 0.2,
                "score_delta": 1000.0,
            },
        ),
        checkpoint_eval_summary=_checkpoint_eval_summary(
            "baseline",
            "candidate",
            arena_metrics={
                **_arena_metrics(),
                "candidate_fourth_rate": 0.3,
                "baseline_fourth_rate": 0.2,
                "score_delta": 1000.0,
            },
        ),
    )

    assert record.decision.decision == "reject"
    assert record.decision.reasons == ("fourth_rate_delta",)


def test_illegal_action_or_non_finite_metric_blocks_promotion(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    arena_path = _arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate)
    summary = _checkpoint_eval_summary("baseline", "candidate")
    metrics = cast("dict[str, object]", summary["metrics"])
    metrics["illegal_action_count"] = 1

    with pytest.raises(ValueError, match="illegal_action_count"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-illegal",
            candidate_checkpoint_id="candidate",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=summary,
            thresholds={"min_games": 8, "max_illegal_action_count": 0.0},
        )

    summary = _checkpoint_eval_summary("baseline", "candidate")
    metrics = cast("dict[str, object]", summary["metrics"])
    metrics["score_delta"] = math.inf
    with pytest.raises(ValueError, match="finite"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-nonfinite",
            candidate_checkpoint_id="candidate",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=summary,
        )


def test_json_strictness_rejects_nan_before_write(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        write_json_file(tmp_path / "bad.json", {"metric": math.nan})
    assert not (tmp_path / "bad.json").exists()


def test_weight_source_mismatch_rejects(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    arena_path = _arena_summary_path(
        tmp_path, registry.checkpoints["baseline"], candidate, baseline_weight_source="ema"
    )
    with pytest.raises(ValueError, match="baseline weight_source mismatch"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-baseline-weight-mismatch",
            candidate_checkpoint_id="candidate",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=_checkpoint_eval_summary("baseline", "candidate"),
        )

    arena_path = _arena_summary_path(
        tmp_path, registry.checkpoints["baseline"], candidate, candidate_weight_source="ema"
    )
    with pytest.raises(ValueError, match="candidate weight_source mismatch"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-candidate-weight-mismatch",
            candidate_checkpoint_id="candidate",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=_checkpoint_eval_summary("baseline", "candidate"),
        )


def test_offline_loss_or_training_metrics_do_not_auto_promote(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    arena_path = _arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate)
    summary = _checkpoint_eval_summary("baseline", "candidate")
    summary["metrics"] = {"games": 16, "offline_loss_delta": -0.2}

    with pytest.raises(ValueError, match="missing configured metric gate"):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-1",
            candidate_checkpoint_id="candidate",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=summary,
        )


def test_deltaq_requires_arena_confirmation_does_not_count_as_acceptance(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    record = build_promotion_record(
        registry=registry,
        promotion_id="promotion-1",
        candidate_checkpoint_id="candidate",
        baseline_checkpoint_id="baseline",
        opponent_pool_id="active_baseline",
        seed_set_id="promotion_v1",
        arena_summary_path=_arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate),
        checkpoint_eval_summary=_checkpoint_eval_summary("baseline", "candidate"),
        delta_q_summary={"recommendation": "requires_arena_confirmation"},
    )

    assert record.decision.decision == "blocked"
    assert "delta_q_requires_arena_confirmation_is_not_acceptance" in record.decision.reasons
    assert registry.with_promotion(record).active_baseline_id == "baseline"


def test_deltaq_arena_decision_reject_blocks_promotion(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    record = build_promotion_record(
        registry=registry,
        promotion_id="promotion-1",
        candidate_checkpoint_id="candidate",
        baseline_checkpoint_id="baseline",
        opponent_pool_id="active_baseline",
        seed_set_id="promotion_v1",
        arena_summary_path=_arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate),
        checkpoint_eval_summary=_checkpoint_eval_summary("baseline", "candidate"),
        delta_q_summary={"recommendation": "requires_arena_confirmation", "arena_decision": "Reject"},
    )

    assert record.decision.decision == "blocked"
    assert "delta_q_arena_reject" in record.decision.reasons
    assert registry.with_promotion(record).active_baseline_id == "baseline"


def test_deltaq_arena_reject_blocks_without_recommendation(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    record = build_promotion_record(
        registry=registry,
        promotion_id="promotion-deltaq-reject-only",
        candidate_checkpoint_id="candidate",
        baseline_checkpoint_id="baseline",
        opponent_pool_id="active_baseline",
        seed_set_id="promotion_v1",
        arena_summary_path=_arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate),
        checkpoint_eval_summary=_checkpoint_eval_summary("baseline", "candidate"),
        delta_q_summary={"arena_decision": "Reject"},
    )

    assert record.decision.decision == "blocked"
    assert "delta_q_arena_reject" in record.decision.reasons
    assert registry.with_promotion(record).active_baseline_id == "baseline"


def test_registry_roundtrip_stable_and_compact(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    path = registry.save(tmp_path)
    loaded = PopulationRegistry.load(path)

    first = path.read_text(encoding="utf-8")
    loaded.save(tmp_path)
    second = path.read_text(encoding="utf-8")

    assert first == second
    assert "\n" not in first[:-1]
    assert registry_to_dict(loaded) == registry_to_dict(registry)


def test_no_psro_pfsp_exploiter_search_or_objective_fields_accepted(tmp_path: Path) -> None:
    field_names = {field.name for field in fields(CheckpointEntry)}
    forbidden = {"psro", "pfsp", "exploiter", "search_teacher", "search_objective", "payoff_matrix"}
    assert not field_names & forbidden

    with pytest.raises(ValueError, match="out-of-scope"):
        write_json_file(tmp_path / "bad.json", {"pfsp_weights": []})

    payload = registry_to_dict(_registry(tmp_path))
    payload["payoff_matrix"] = cast("object", [])
    with pytest.raises(ValueError, match=r"unsupported fields|out-of-scope"):
        PopulationRegistry.load(_write_raw_json(tmp_path / "bad_registry.json", payload))


def _registry(tmp_path: Path, *, seed_bank: SeedBank | None = None) -> PopulationRegistry:
    baseline = _checkpoint(tmp_path, "baseline", b"baseline", role="champion")
    active_seed_bank = seed_bank or build_seed_bank(
        seed_set_id="promotion_v1",
        seeds=(11, 22, 33),
        games_per_seed=4,
        temperature=0.0,
        arena_options={"device": "cuda:0", "native": True},
    )
    schedule = EvalSchedule(
        enabled=True,
        seed_set_id="promotion_v1",
        opponent_pool_id="active_baseline",
        min_games=12,
        thresholds={"min_games": 12, "max_fourth_rate_delta": 0.0, "min_mean_u_a_delta": 0.0},
    )
    return PopulationRegistry.create(
        registry_id="registry-1",
        run_id="run-1",
        active_baseline_id="baseline",
        baseline=baseline,
        seed_bank=active_seed_bank,
        eval_schedule=schedule,
    )


def _checkpoint(
    tmp_path: Path, checkpoint_id: str, content: bytes, *, role: Literal["candidate", "champion", "rejected", "seed"]
) -> CheckpointEntry:
    path = tmp_path / f"{checkpoint_id}.pt"
    path.write_bytes(content)
    return register_immutable_checkpoint(
        checkpoint_id=checkpoint_id,
        role=role,
        path=path,
        weight_source="raw",
        global_step=1,
        samples_seen=128,
        model_config={"hidden": 8, "blocks": 1},
        source={"manifest_path": "manifest.json"},
        registered_at_unix_ms=1234,
    )


def _checkpoint_eval_summary(
    baseline: str,
    candidate: str,
    *,
    games: int = 16,
    seed: int = 11,
    arena_metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    metrics = arena_metrics or {
        "games": games,
        "candidate_mean_placement": 2.3,
        "baseline_mean_placement": 2.4,
        "candidate_fourth_rate": 0.1,
        "baseline_fourth_rate": 0.2,
    }
    metrics.setdefault("candidate_mean_placement", 2.3)
    metrics.setdefault("baseline_mean_placement", 2.4)
    metrics.setdefault("candidate_fourth_rate", 0.1)
    metrics.setdefault("baseline_fourth_rate", 0.2)
    metrics.setdefault("games", games)
    summary = build_paired_checkpoint_eval_summary(
        baseline=baseline,
        candidate=candidate,
        seed=seed,
        thresholds=PairedCheckpointEvalThresholds(max_fourth_rate_delta=0.0, min_mean_u_a_delta=0.0, min_games=8),
        arena_metrics=metrics,
    )
    return {
        "baseline": summary.baseline,
        "candidate": summary.candidate,
        "games": summary.games,
        "seed": summary.seed,
        "metrics": dict(summary.metrics),
        "decision": {"decision": summary.decision.decision, "reasons": list(summary.decision.reasons)},
    }


def _forged_promote_summary(baseline: str, candidate: str, metrics: dict[str, object]) -> dict[str, object]:
    summary = _checkpoint_eval_summary(baseline, candidate, arena_metrics=metrics)
    summary["decision"] = {"decision": "promote", "reasons": ["all_configured_gates_passed"]}
    return summary


def _materialized_promotion_record(
    candidate_id: str,
    summary: dict[str, object],
    *,
    thresholds: dict[str, float | int | None] | None = None,
) -> PromotionRecord:
    active_thresholds = thresholds or {"min_games": 8, "max_fourth_rate_delta": 0.0, "min_mean_u_a_delta": 0.0}
    metrics = cast("dict[str, float | int | str | None]", summary["metrics"])
    decision = cast("dict[str, object]", summary["decision"])
    return PromotionRecord(
        schema_version=1,
        promotion_id=f"promotion-{candidate_id}",
        candidate_checkpoint_id=candidate_id,
        baseline_checkpoint_id="baseline",
        opponent_pool_id="active_baseline",
        seed_set_id="promotion_v1",
        arena_summary_path="/evidence/arena.json",
        paired_eval_summary_path="/evidence/paired.json",
        normalized_metrics=metrics,
        thresholds=active_thresholds,
        decision=PromotionDecisionRecord(
            decision=cast("Literal['blocked', 'insufficient_games', 'promote', 'reject']", decision["decision"]),
            reasons=tuple(cast("list[str]", decision["reasons"])),
            metrics=metrics,
            thresholds=active_thresholds,
        ),
        registry_update={"active_baseline_id_before": "baseline", "active_baseline_id_after": candidate_id},
        created_at_unix_ms=1234,
    )


def _assert_forged_promote_rejects(
    tmp_path: Path,
    arena_metrics: dict[str, object],
    *,
    match: str,
    thresholds: dict[str, float | int | None] | None = None,
) -> None:
    registry = _registry(tmp_path)
    candidate = _checkpoint(tmp_path, "candidate", b"candidate", role="candidate")
    registry = registry.with_checkpoint(candidate)
    arena_path = _arena_summary_path(tmp_path, registry.checkpoints["baseline"], candidate, result=arena_metrics)
    with pytest.raises(ValueError, match=match):
        build_promotion_record(
            registry=registry,
            promotion_id="promotion-forged",
            candidate_checkpoint_id="candidate",
            baseline_checkpoint_id="baseline",
            opponent_pool_id="active_baseline",
            seed_set_id="promotion_v1",
            arena_summary_path=arena_path,
            checkpoint_eval_summary=_forged_promote_summary("baseline", "candidate", dict(arena_metrics)),
            thresholds=thresholds,
        )


def _arena_summary_path(
    tmp_path: Path,
    baseline: CheckpointEntry,
    candidate: CheckpointEntry,
    *,
    result: dict[str, object] | None = None,
    seed: int = 11,
    baseline_weight_source: str = "raw",
    candidate_weight_source: str = "raw",
) -> Path:
    path = tmp_path / "arena_summary.json"
    payload = {
        "baseline": {
            "path": baseline.path,
            "global_step": baseline.global_step,
            "samples_seen": baseline.samples_seen,
            "weight_source": baseline_weight_source,
        },
        "candidates": [
            {
                "candidate": candidate.checkpoint_id,
                "candidate_path": candidate.path,
                "global_step": candidate.global_step,
                "samples_seen": candidate.samples_seen,
                "weight_source": candidate_weight_source,
                "result": result or _arena_metrics(),
            }
        ],
        "config": {"games": (result or _arena_metrics())["games"], "seed": seed},
    }
    path.write_text(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _arena_metrics() -> dict[str, object]:
    return {
        "games": 16,
        "candidate_mean_placement": 2.3,
        "baseline_mean_placement": 2.4,
        "candidate_fourth_rate": 0.1,
        "baseline_fourth_rate": 0.2,
    }


def _write_raw_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
    return path
