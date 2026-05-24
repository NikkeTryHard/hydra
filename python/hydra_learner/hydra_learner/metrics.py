"""Metrics helpers for Hydra experimental PyTorch learner."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class StepStats:
    step_ms: float
    fwd_loss_ms: float
    backward_ms: float
    optimizer_ms: float
    loss: float
    head_losses: dict[str, float]
    target_coverage: dict[str, dict[str, float | str]]
    policy_nll: float = math.nan
    policy_accuracy: float = math.nan
    policy_top3_accuracy: float = math.nan
    policy_top5_accuracy: float = math.nan
    policy_confidence: float = math.nan
    policy_entropy: float = math.nan
    policy_target_prob: float = math.nan
    policy_margin: float = math.nan
    fetch_decode_ms: float = math.nan
    h2d_wall_ms: float = math.nan
    train_gpu_ms: float = math.nan
    lr: float = math.nan
    grad_norm: float = math.nan
    lr_progress_games: float = math.nan


@dataclass
class EvalStats:
    loss: float
    policy: float
    value: float
    grp: float
    tenpai: float
    danger: float
    opp_next: float
    score_pdf: float
    score_cdf: float
    oracle_critic: float
    safety_residual: float
    target_coverage: dict[str, dict[str, float | str]]
    policy_accuracy: float
    policy_top3_accuracy: float
    policy_top5_accuracy: float
    policy_nll: float
    policy_confidence: float
    policy_ece: float
    samples: int


def summarize_eval(stats: list[EvalStats]) -> dict[str, object]:
    total_samples = sum(s.samples for s in stats)

    def weighted(values: list[tuple[float, int]]) -> float:
        if total_samples == 0:
            return math.nan
        return sum(value * samples for value, samples in values) / total_samples

    def weighted_coverage(head: str) -> float:
        values: list[tuple[float, int]] = []
        for stat in stats:
            coverage = stat.target_coverage.get(head)
            if coverage is None:
                continue
            fraction = coverage.get("fraction")
            if isinstance(fraction, float):
                values.append((fraction, stat.samples))
        return weighted(values)

    def coverage_status(head: str) -> str:
        statuses = {str(stat.target_coverage[head]["status"]) for stat in stats if head in stat.target_coverage}
        if "present_positive" in statuses:
            return "present_positive"
        if "present_zero" in statuses:
            return "present_zero"
        return "absent"

    metrics: dict[str, object]
    metrics = {
        "batches": float(len(stats)),
        "samples": float(total_samples),
        "loss": weighted([(s.loss, s.samples) for s in stats]),
        "policy": weighted([(s.policy, s.samples) for s in stats]),
        "value": weighted([(s.value, s.samples) for s in stats]),
        "grp": weighted([(s.grp, s.samples) for s in stats]),
        "tenpai": weighted([(s.tenpai, s.samples) for s in stats]),
        "danger": weighted([(s.danger, s.samples) for s in stats]),
        "opp_next": weighted([(s.opp_next, s.samples) for s in stats]),
        "score_pdf": weighted([(s.score_pdf, s.samples) for s in stats]),
        "score_cdf": weighted([(s.score_cdf, s.samples) for s in stats]),
        "oracle_critic": weighted([(s.oracle_critic, s.samples) for s in stats]),
        "safety_residual": weighted([(s.safety_residual, s.samples) for s in stats]),
        "policy_accuracy": weighted([(s.policy_accuracy, s.samples) for s in stats]),
        "policy_top3_accuracy": weighted([(s.policy_top3_accuracy, s.samples) for s in stats]),
        "policy_top5_accuracy": weighted([(s.policy_top5_accuracy, s.samples) for s in stats]),
        "policy_nll": weighted([(s.policy_nll, s.samples) for s in stats]),
        "policy_confidence": weighted([(s.policy_confidence, s.samples) for s in stats]),
        "policy_ece": weighted([(s.policy_ece, s.samples) for s in stats]),
    }
    if stats:
        metrics |= {f"coverage/{head}/fraction": weighted_coverage(head) for head in stats[0].target_coverage}
        metrics |= {
            f"coverage/{head}/status_code": float(
                ("absent", "present_zero", "present_positive").index(coverage_status(head))
            )
            for head in stats[0].target_coverage
        }
        for head in stats[0].target_coverage:
            metrics[f"coverage/{head}/status"] = coverage_status(head)
    return metrics


def summarize_steps(stats: list[StepStats], batch: int) -> dict[str, float]:
    step = [s.step_ms for s in stats]
    fwd = [s.fwd_loss_ms for s in stats if not math.isnan(s.fwd_loss_ms)]
    bwd = [s.backward_ms for s in stats if not math.isnan(s.backward_ms)]
    opt = [s.optimizer_ms for s in stats if not math.isnan(s.optimizer_ms)]
    loss = [s.loss for s in stats]
    fetch = [s.fetch_decode_ms for s in stats if not math.isnan(s.fetch_decode_ms)]
    h2d_wall = [s.h2d_wall_ms for s in stats if not math.isnan(s.h2d_wall_ms)]
    train_gpu = [s.train_gpu_ms for s in stats if not math.isnan(s.train_gpu_ms)]
    policy_nll = [s.policy_nll for s in stats if not math.isnan(s.policy_nll)]
    policy_accuracy = [s.policy_accuracy for s in stats if not math.isnan(s.policy_accuracy)]
    policy_top3_accuracy = [s.policy_top3_accuracy for s in stats if not math.isnan(s.policy_top3_accuracy)]
    policy_top5_accuracy = [s.policy_top5_accuracy for s in stats if not math.isnan(s.policy_top5_accuracy)]
    policy_confidence = [s.policy_confidence for s in stats if not math.isnan(s.policy_confidence)]
    policy_entropy = [s.policy_entropy for s in stats if not math.isnan(s.policy_entropy)]
    policy_target_prob = [s.policy_target_prob for s in stats if not math.isnan(s.policy_target_prob)]
    policy_margin = [s.policy_margin for s in stats if not math.isnan(s.policy_margin)]

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else math.nan

    mean_step = avg(step)
    mean_fetch = avg(fetch)
    mean_h2d_wall = avg(h2d_wall)
    mean_train_gpu = avg(train_gpu)
    mean_total_wall = mean_fetch + mean_h2d_wall + mean_train_gpu
    return {
        "steps": float(len(stats)),
        "mean_step_ms": mean_step,
        "samples_per_s": batch * 1000.0 / mean_step,
        "mean_fwd_loss_ms": avg(fwd),
        "mean_loss": avg(loss),
        "mean_backward_ms": avg(bwd),
        "mean_optimizer_ms": avg(opt),
        "mean_fetch_decode_ms": mean_fetch,
        "mean_h2d_wall_ms": mean_h2d_wall,
        "mean_train_gpu_ms": mean_train_gpu,
        "mean_input_pipeline_wall_ms": mean_fetch + mean_h2d_wall,
        "mean_total_wall_ms": mean_total_wall,
        "end_to_end_samples_per_s": batch * 1000.0 / mean_total_wall,
        "mean_policy_nll": avg(policy_nll),
        "mean_policy_accuracy": avg(policy_accuracy),
        "mean_policy_top3_accuracy": avg(policy_top3_accuracy),
        "mean_policy_top5_accuracy": avg(policy_top5_accuracy),
        "mean_policy_confidence": avg(policy_confidence),
        "mean_policy_entropy": avg(policy_entropy),
        "mean_policy_target_prob": avg(policy_target_prob),
        "mean_policy_margin": avg(policy_margin),
    }
