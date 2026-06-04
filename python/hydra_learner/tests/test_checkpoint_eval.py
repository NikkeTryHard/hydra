from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hydra_learner.checkpointing.eval import (
    PairedCheckpointEvalThresholds,
    append_paired_checkpoint_eval_jsonl,
    build_paired_checkpoint_eval_summary,
    normalize_paired_arena_metrics,
    paired_checkpoint_eval_summary_to_dict,
    write_paired_checkpoint_eval_summary,
)


def test_checkpoint_eval_normalizes_native_arena_metrics_json_safe() -> None:
    summary = build_paired_checkpoint_eval_summary(
        baseline="baseline-export",
        candidate="candidate-export",
        seed=123,
        thresholds=PairedCheckpointEvalThresholds(max_fourth_rate_delta=0.0, min_mean_u_a_delta=0.0),
        arena_metrics={
            "games": 8,
            "candidate_winrate": 0.375,
            "candidate_avg_rank": 2.25,
            "baseline_avg_rank": 2.5,
            "candidate_top2": 0.625,
            "baseline_top2": 0.5,
            "candidate_fourth": 0.125,
            "baseline_fourth": 0.25,
            "candidate_avg_score": 27100.0,
            "score_delta": 1500.0,
            "pt_delta": 1.5,
        },
    )

    assert summary.games == 8
    assert summary.seed == 123
    assert summary.metrics == {
        "games": 8,
        "candidate_winrate": 0.375,
        "candidate_mean_placement": 2.25,
        "candidate_avg_rank": 2.25,
        "candidate_top2_rate": 0.625,
        "candidate_fourth_rate": 0.125,
        "candidate_avg_score": 27100.0,
        "score_delta": 1500.0,
        "pt_delta": 1.5,
        "top2_delta": 0.125,
        "fourth_rate_delta": -0.125,
        "mean_u_a_delta": 0.25,
    }
    assert summary.decision.decision == "promote"
    payload = paired_checkpoint_eval_summary_to_dict(summary)
    payload_text = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    metric_payload = payload["metrics"]
    assert isinstance(metric_payload, dict)
    payload_keys = set(payload) | set(metric_payload)
    assert "confidence" not in payload_text
    assert not any("confidence" in key or key.endswith("_ci") or "confidence_bound" in key for key in payload_keys)
    assert "delta_q" not in payload_text.lower()


def test_checkpoint_eval_min_games_gate_insufficient() -> None:
    summary = build_paired_checkpoint_eval_summary(
        baseline="b",
        candidate="c",
        thresholds=PairedCheckpointEvalThresholds(min_games=16),
        arena_metrics={"games": 8},
    )

    assert summary.decision.decision == "insufficient_games"
    assert summary.decision.reasons == ("insufficient_games",)


def test_checkpoint_eval_fourth_rate_regression_rejects_positive_score_delta() -> None:
    summary = build_paired_checkpoint_eval_summary(
        baseline="b",
        candidate="c",
        thresholds=PairedCheckpointEvalThresholds(max_fourth_rate_delta=0.0, min_mean_u_a_delta=None),
        arena_metrics={
            "games": 32,
            "candidate_fourth_rate": 0.3,
            "baseline_fourth_rate": 0.2,
            "score_delta": 2000.0,
        },
    )

    assert summary.metrics["fourth_rate_delta"] == pytest.approx(0.1)
    assert summary.metrics["score_delta"] == 2000.0
    assert summary.decision.decision == "reject"
    assert summary.decision.reasons == ("fourth_rate_delta",)


def test_checkpoint_eval_top2_gate_rejects_when_configured() -> None:
    summary = build_paired_checkpoint_eval_summary(
        baseline="b",
        candidate="c",
        thresholds=PairedCheckpointEvalThresholds(
            max_fourth_rate_delta=None,
            min_mean_u_a_delta=None,
            min_top2_delta=0.05,
        ),
        arena_metrics={
            "games": 32,
            "candidate_top2_rate": 0.51,
            "baseline_top2_rate": 0.50,
        },
    )

    assert summary.metrics["top2_delta"] == pytest.approx(0.01)
    assert summary.decision.decision == "reject"
    assert summary.decision.reasons == ("top2_delta",)


def test_checkpoint_eval_missing_mean_u_a_delta_rejects_when_configured() -> None:
    summary = build_paired_checkpoint_eval_summary(
        baseline="b",
        candidate="c",
        thresholds=PairedCheckpointEvalThresholds(max_fourth_rate_delta=None, min_mean_u_a_delta=0.0),
        arena_metrics={"games": 32, "candidate_top2_rate": 0.6, "baseline_top2_rate": 0.5},
    )

    assert summary.metrics["mean_u_a_delta"] is None
    assert summary.decision.decision == "reject"
    assert summary.decision.reasons == ("missing_mean_u_a_delta",)


def test_checkpoint_eval_promotes_when_configured_available_gates_pass() -> None:
    summary = build_paired_checkpoint_eval_summary(
        baseline="b",
        candidate="c",
        thresholds=PairedCheckpointEvalThresholds(
            max_fourth_rate_delta=0.01,
            min_mean_u_a_delta=0.0,
            min_top2_delta=0.05,
        ),
        arena_metrics={
            "games": 64,
            "candidate_mean_placement": 2.35,
            "baseline_mean_placement": 2.45,
            "candidate_top2_rate": 0.58,
            "baseline_top2_rate": 0.50,
            "candidate_fourth_rate": 0.18,
            "baseline_fourth_rate": 0.20,
        },
    )

    assert summary.decision.decision == "promote"
    assert summary.decision.reasons == ("all_configured_gates_passed",)


def test_checkpoint_eval_json_writers_reject_nan_without_emitting(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    summary = build_paired_checkpoint_eval_summary(
        baseline="b",
        candidate="c",
        thresholds=PairedCheckpointEvalThresholds(max_fourth_rate_delta=None, min_mean_u_a_delta=None),
        arena_metrics={"games": 1},
    )
    summary.metrics["score_delta"] = math.nan

    with pytest.raises(ValueError, match=r"non-finite|finite"):
        write_paired_checkpoint_eval_summary(path, summary)
    assert not path.exists()

    jsonl_path = tmp_path / "summary.jsonl"
    with pytest.raises(ValueError, match=r"non-finite|finite"):
        append_paired_checkpoint_eval_jsonl(jsonl_path, {"games": 1, "score_delta": math.nan})
    assert not jsonl_path.exists()


def test_checkpoint_eval_json_helpers_write_strict_sorted_payloads(tmp_path: Path) -> None:
    summary = build_paired_checkpoint_eval_summary(
        baseline="b",
        candidate="c",
        thresholds=PairedCheckpointEvalThresholds(max_fourth_rate_delta=None, min_mean_u_a_delta=None),
        arena_metrics={"games": 1, "candidate_winrate": 1.0},
    )
    summary_path = tmp_path / "summary.json"
    jsonl_path = tmp_path / "summary.jsonl"

    write_paired_checkpoint_eval_summary(summary_path, summary)
    append_paired_checkpoint_eval_jsonl(jsonl_path, summary)
    append_paired_checkpoint_eval_jsonl(jsonl_path, {"z": 1, "a": 2})

    assert json.loads(summary_path.read_text(encoding="utf-8"))["decision"]["decision"] == "promote"
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[1] == '{"a":2,"z":1}'


def test_checkpoint_eval_normalization_rejects_nonfinite_native_metric() -> None:
    with pytest.raises(ValueError, match="candidate_winrate must be finite"):
        normalize_paired_arena_metrics({"games": 1, "candidate_winrate": math.inf})
