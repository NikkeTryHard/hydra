"""Normalize paired arena output into Python checkpoint promotion decisions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

MetricValue = float | int | str | None
DecisionName = Literal["promote", "reject", "insufficient_games"]


@dataclass(frozen=True)
class PairedCheckpointEvalThresholds:
    max_fourth_rate_delta: float | None = 0.0
    min_mean_u_a_delta: float | None = 0.0
    min_top2_delta: float | None = None
    min_games: int = 1


@dataclass(frozen=True)
class PairedCheckpointEvalDecision:
    decision: DecisionName
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PairedCheckpointEvalSummary:
    baseline: str
    candidate: str
    games: int
    seed: int | None
    metrics: dict[str, MetricValue]
    decision: PairedCheckpointEvalDecision


def build_paired_checkpoint_eval_summary(
    *,
    baseline: str,
    candidate: str,
    arena_metrics: Mapping[str, object],
    thresholds: PairedCheckpointEvalThresholds | None = None,
    seed: int | None = None,
) -> PairedCheckpointEvalSummary:
    active_thresholds = thresholds or PairedCheckpointEvalThresholds()
    metrics = normalize_paired_arena_metrics(arena_metrics)
    games_value = metrics["games"]
    if not isinstance(games_value, int):
        raise TypeError("normalized games must be int")
    decision = decide_paired_checkpoint_eval(metrics, active_thresholds)
    return PairedCheckpointEvalSummary(
        baseline=baseline,
        candidate=candidate,
        games=games_value,
        seed=seed,
        metrics=metrics,
        decision=decision,
    )


def normalize_paired_arena_metrics(arena_metrics: Mapping[str, object]) -> dict[str, MetricValue]:
    games = _required_int(arena_metrics, "games")
    metrics: dict[str, MetricValue] = {"games": games}

    candidate_top2 = _optional_float(arena_metrics, "candidate_top2_rate", "candidate_top2")
    baseline_top2 = _optional_float(arena_metrics, "baseline_top2_rate", "baseline_top2")
    candidate_fourth = _optional_float(arena_metrics, "candidate_fourth_rate", "candidate_fourth")
    baseline_fourth = _optional_float(arena_metrics, "baseline_fourth_rate", "baseline_fourth")
    candidate_mean_placement = _optional_float(
        arena_metrics,
        "candidate_mean_placement",
        "candidate_avg_rank",
        "mean_placement",
    )
    baseline_mean_placement = _optional_float(arena_metrics, "baseline_mean_placement", "baseline_avg_rank")

    metrics["candidate_winrate"] = _optional_float(arena_metrics, "candidate_winrate")
    metrics["candidate_mean_placement"] = candidate_mean_placement
    metrics["candidate_avg_rank"] = candidate_mean_placement
    metrics["candidate_top2_rate"] = candidate_top2
    metrics["candidate_fourth_rate"] = candidate_fourth
    metrics["candidate_avg_score"] = _optional_float(arena_metrics, "candidate_avg_score")
    metrics["score_delta"] = _optional_float(arena_metrics, "score_delta")
    metrics["pt_delta"] = _optional_float(arena_metrics, "pt_delta")
    metrics["top2_delta"] = _delta(candidate_top2, baseline_top2)
    metrics["fourth_rate_delta"] = _delta(candidate_fourth, baseline_fourth)
    metrics["mean_u_a_delta"] = _placement_utility_delta(candidate_mean_placement, baseline_mean_placement)
    _validate_json_safe_metrics(metrics)
    return metrics


def decide_paired_checkpoint_eval(
    metrics: Mapping[str, MetricValue], thresholds: PairedCheckpointEvalThresholds
) -> PairedCheckpointEvalDecision:
    if thresholds.min_games < 1:
        raise ValueError("min_games must be >= 1")
    games = _metric_int(metrics, "games")
    if games < thresholds.min_games:
        return PairedCheckpointEvalDecision("insufficient_games", ("insufficient_games",))

    reasons: list[str] = []
    if thresholds.max_fourth_rate_delta is not None:
        fourth_rate_delta = _metric_float_or_none(metrics, "fourth_rate_delta")
        if fourth_rate_delta is None:
            reasons.append("missing_fourth_rate_delta")
        elif fourth_rate_delta > thresholds.max_fourth_rate_delta:
            reasons.append("fourth_rate_delta")

    if thresholds.min_mean_u_a_delta is not None:
        mean_u_a_delta = _metric_float_or_none(metrics, "mean_u_a_delta")
        if mean_u_a_delta is None:
            reasons.append("missing_mean_u_a_delta")
        elif mean_u_a_delta < thresholds.min_mean_u_a_delta:
            reasons.append("mean_u_a_delta")

    if thresholds.min_top2_delta is not None:
        top2_delta = _metric_float_or_none(metrics, "top2_delta")
        if top2_delta is None:
            reasons.append("missing_top2_delta")
        elif top2_delta < thresholds.min_top2_delta:
            reasons.append("top2_delta")

    if reasons:
        return PairedCheckpointEvalDecision("reject", tuple(reasons))
    return PairedCheckpointEvalDecision("promote", ("all_configured_gates_passed",))


def paired_checkpoint_eval_summary_to_dict(summary: PairedCheckpointEvalSummary) -> dict[str, object]:
    payload: dict[str, object] = {
        "baseline": summary.baseline,
        "candidate": summary.candidate,
        "games": summary.games,
        "seed": summary.seed,
        "metrics": dict(summary.metrics),
        "decision": asdict(summary.decision),
    }
    _validate_json_payload(payload)
    return payload


def write_paired_checkpoint_eval_summary(path: Path, summary: PairedCheckpointEvalSummary) -> None:
    payload = paired_checkpoint_eval_summary_to_dict(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_line(payload) + "\n", encoding="utf-8")


def append_paired_checkpoint_eval_jsonl(
    path: Path, summary: PairedCheckpointEvalSummary | Mapping[str, object]
) -> None:
    payload: Mapping[str, object]
    if isinstance(summary, PairedCheckpointEvalSummary):
        payload = paired_checkpoint_eval_summary_to_dict(summary)
    else:
        payload = summary
        _validate_json_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_json_line(payload))
        fh.write("\n")


def _json_line(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _required_int(metrics: Mapping[str, object], key: str) -> int:
    value = metrics.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_float(metrics: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{key} must be numeric when present")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{key} must be finite")
        return result
    return None


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _placement_utility_delta(candidate_mean: float | None, baseline_mean: float | None) -> float | None:
    if candidate_mean is None or baseline_mean is None:
        return None
    return baseline_mean - candidate_mean


def _metric_int(metrics: Mapping[str, MetricValue], key: str) -> int:
    value = metrics.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"metrics.{key} must be int")
    return value


def _metric_float_or_none(metrics: Mapping[str, MetricValue], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"metrics.{key} must be numeric or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"metrics.{key} must be finite")
    return result


def _validate_json_safe_metrics(metrics: Mapping[str, MetricValue]) -> None:
    for key, value in metrics.items():
        if isinstance(value, bool):
            raise TypeError(f"metrics.{key} must not be bool")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"metrics.{key} must be finite")


def _validate_json_payload(value: object) -> None:
    if isinstance(value, bool | str) or value is None:
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON payload contains non-finite float")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_json_payload(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, str):
        for item in value:
            _validate_json_payload(item)
        return
    raise TypeError(f"JSON payload contains unsupported {type(value).__name__}")
