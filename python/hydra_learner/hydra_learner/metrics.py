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
    fetch_decode_ms: float = math.nan
    h2d_wall_ms: float = math.nan
    h2d_gpu_ms: float = math.nan
    train_gpu_ms: float = math.nan


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
    policy_accuracy: float


def summarize_eval(stats: list[EvalStats]) -> dict[str, float]:
    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else math.nan

    return {
        "batches": float(len(stats)),
        "loss": avg([s.loss for s in stats]),
        "policy": avg([s.policy for s in stats]),
        "value": avg([s.value for s in stats]),
        "grp": avg([s.grp for s in stats]),
        "tenpai": avg([s.tenpai for s in stats]),
        "danger": avg([s.danger for s in stats]),
        "opp_next": avg([s.opp_next for s in stats]),
        "score_pdf": avg([s.score_pdf for s in stats]),
        "score_cdf": avg([s.score_cdf for s in stats]),
        "policy_accuracy": avg([s.policy_accuracy for s in stats]),
    }


def summarize_steps(stats: list[StepStats], batch: int) -> dict[str, float]:
    step = [s.step_ms for s in stats]
    fwd = [s.fwd_loss_ms for s in stats if not math.isnan(s.fwd_loss_ms)]
    bwd = [s.backward_ms for s in stats if not math.isnan(s.backward_ms)]
    opt = [s.optimizer_ms for s in stats if not math.isnan(s.optimizer_ms)]
    loss = [s.loss for s in stats]
    fetch = [s.fetch_decode_ms for s in stats if not math.isnan(s.fetch_decode_ms)]
    h2d_wall = [s.h2d_wall_ms for s in stats if not math.isnan(s.h2d_wall_ms)]
    h2d_gpu = [s.h2d_gpu_ms for s in stats if not math.isnan(s.h2d_gpu_ms)]
    train_gpu = [s.train_gpu_ms for s in stats if not math.isnan(s.train_gpu_ms)]

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else math.nan

    mean_step = avg(step)
    mean_fetch = avg(fetch)
    mean_h2d_wall = avg(h2d_wall)
    mean_h2d_gpu = avg(h2d_gpu)
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
        "mean_h2d_gpu_ms": mean_h2d_gpu,
        "mean_train_gpu_ms": mean_train_gpu,
        "mean_input_pipeline_wall_ms": mean_fetch + mean_h2d_wall,
        "mean_total_wall_ms": mean_total_wall,
        "end_to_end_samples_per_s": batch * 1000.0 / mean_total_wall,
        "h2d_train_ratio": mean_h2d_gpu / mean_train_gpu,
    }
