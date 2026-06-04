"""Validation convergence and scalar helpers for BC training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import TYPE_CHECKING, cast

from hydra_learner.telemetry.logging import prefixed_metrics

if TYPE_CHECKING:
    from hydra_learner.training.validation import ValidationSourceInfo


@dataclass
class ValidationConvergenceState:
    best_policy_nll: float = math.inf
    best_step: int = 0
    validation_count: int = 0
    best_validation_count: int = 0
    last_policy_nll: float | None = None
    recent_policy_nll: list[float] | None = None

    def update(self, policy_nll: float, global_step: int) -> dict[str, float]:
        if self.recent_policy_nll is None:
            self.recent_policy_nll = []
        self.validation_count += 1
        delta_since_last = math.nan if self.last_policy_nll is None else policy_nll - self.last_policy_nll
        self.last_policy_nll = policy_nll
        if policy_nll < self.best_policy_nll:
            self.best_policy_nll = policy_nll
            self.best_step = global_step
            self.best_validation_count = self.validation_count
        self.recent_policy_nll.append(policy_nll)
        if len(self.recent_policy_nll) > 8:
            del self.recent_policy_nll[0]
        slope = _simple_slope(self.recent_policy_nll)
        return {
            "policy_nll_best": self.best_policy_nll,
            "policy_nll_delta_from_best": policy_nll - self.best_policy_nll,
            "validations_since_best": float(self.validation_count - self.best_validation_count),
            "steps_since_best": float(global_step - self.best_step),
            "policy_nll_delta_since_last": delta_since_last,
            "policy_nll_recent_slope": slope,
        }


def _simple_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return math.nan
    x_mean = (n - 1) * 0.5
    y_mean = sum(values) / n
    denom = sum((i - x_mean) * (i - x_mean) for i in range(n))
    return sum((i - x_mean) * (value - y_mean) for i, value in enumerate(values)) / denom


def _numeric_validation_source_scalars(source: ValidationSourceInfo) -> dict[str, object]:
    return {
        key: value
        for key, value in asdict(source).items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _raw_ema_delta_scalars(raw: dict[str, object], ema: dict[str, object] | None) -> dict[str, float]:
    if ema is None:
        return {}
    out: dict[str, float] = {}
    for key, raw_value in raw.items():
        ema_value = ema.get(key)
        if (
            isinstance(raw_value, Real)
            and not isinstance(raw_value, bool)
            and isinstance(ema_value, Real)
            and not isinstance(ema_value, bool)
        ):
            out[key] = float(ema_value) - float(raw_value)
    return out


def _validation_scalar_metrics(
    *,
    raw_metrics: dict[str, object],
    ema_metrics: dict[str, object] | None,
    source_info: ValidationSourceInfo,
    convergence_metrics: dict[str, float],
) -> dict[str, object]:
    scalar_metrics = prefixed_metrics("raw", raw_metrics)
    if ema_metrics is not None:
        scalar_metrics |= prefixed_metrics("ema", ema_metrics)
        scalar_metrics |= prefixed_metrics(
            "ema_delta", cast(dict[str, object], _raw_ema_delta_scalars(raw_metrics, ema_metrics))
        )
    scalar_metrics |= prefixed_metrics("convergence", cast(dict[str, object], convergence_metrics))
    scalar_metrics |= prefixed_metrics("source", _numeric_validation_source_scalars(source_info))
    return scalar_metrics
