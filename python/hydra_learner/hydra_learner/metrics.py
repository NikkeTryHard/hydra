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


def summarize_steps(stats: list[StepStats], batch: int) -> dict[str, float]:
    step = [s.step_ms for s in stats]
    fwd = [s.fwd_loss_ms for s in stats if not math.isnan(s.fwd_loss_ms)]
    bwd = [s.backward_ms for s in stats if not math.isnan(s.backward_ms)]
    opt = [s.optimizer_ms for s in stats if not math.isnan(s.optimizer_ms)]
    loss = [s.loss for s in stats]

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else math.nan

    mean_step = avg(step)
    return {
        "steps": float(len(stats)),
        "mean_step_ms": mean_step,
        "samples_per_s": batch * 1000.0 / mean_step,
        "mean_fwd_loss_ms": avg(fwd),
        "mean_loss": avg(loss),
        "mean_backward_ms": avg(bwd),
        "mean_optimizer_ms": avg(opt),
    }
