"""WP-05B supervised objectives: masked behavior cloning + auxiliary heads.

Re-export facade over the split modules: :mod:`hydra2.training.objectives_loss`
(masked cross-entropy, supervised loss kernel, contract checks) and
:mod:`hydra2.training.objectives_metrics` (report NLL/top-k, hot scalars,
per-type scorecards, temperature scaling). Import from this path; it preserves
every public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.training.objectives_loss import (
    compute_supervised_loss as compute_supervised_loss,
)
from hydra2.training.objectives_loss import (
    global_grad_norm_is_finite as global_grad_norm_is_finite,
)
from hydra2.training.objectives_loss import (
    masked_cross_entropy as masked_cross_entropy,
)
from hydra2.training.objectives_loss import (
    supervised_loss_kernel as supervised_loss_kernel,
)
from hydra2.training.objectives_loss import (
    validate_supervised_inputs as validate_supervised_inputs,
)
from hydra2.training.objectives_metrics import (
    compute_hot_scalars as compute_hot_scalars,
)
from hydra2.training.objectives_metrics import (
    compute_metrics as compute_metrics,
)
from hydra2.training.objectives_metrics import (
    compute_per_type_metrics as compute_per_type_metrics,
)
from hydra2.training.objectives_metrics import (
    fit_temperature_scaling as fit_temperature_scaling,
)
from hydra2.training.objectives_metrics import (
    masked_topk_accuracy as masked_topk_accuracy,
)

__all__ = [
    "compute_hot_scalars",
    "compute_metrics",
    "compute_per_type_metrics",
    "compute_supervised_loss",
    "fit_temperature_scaling",
    "global_grad_norm_is_finite",
    "masked_cross_entropy",
    "masked_topk_accuracy",
    "supervised_loss_kernel",
    "validate_supervised_inputs",
]

# Names importable from this path before the split that live in the
# submodules now (kept so engine helpers and type-checking imports resolve).
from hydra2.training.objectives_loss import _MASKED_LOGIT_NEG as _MASKED_LOGIT_NEG
from hydra2.training.objectives_loss import (
    LABEL_SMOOTHING_SOTA_DEFAULT as LABEL_SMOOTHING_SOTA_DEFAULT,
)
from hydra2.training.objectives_loss import _check_legal_rows as _check_legal_rows
from hydra2.training.objectives_loss import _check_placement_range as _check_placement_range
from hydra2.training.objectives_loss import _check_targets_in_range as _check_targets_in_range
from hydra2.training.objectives_loss import _check_targets_legal as _check_targets_legal
from hydra2.training.objectives_loss import _check_total_finite as _check_total_finite
from hydra2.training.objectives_loss import _fail_closed_gate as _fail_closed_gate
from hydra2.training.objectives_loss import _generic_ce_loss as _generic_ce_loss
from hydra2.training.objectives_loss import _generic_mse_loss as _generic_mse_loss
from hydra2.training.objectives_metrics import _ECE_NUM_BINS as _ECE_NUM_BINS
from hydra2.training.objectives_metrics import _PER_TYPE_MIN_N as _PER_TYPE_MIN_N
from hydra2.training.objectives_metrics import _TEMPERATURE_MAX as _TEMPERATURE_MAX
from hydra2.training.objectives_metrics import _TEMPERATURE_MIN as _TEMPERATURE_MIN
from hydra2.training.objectives_metrics import _ece_from_confidence as _ece_from_confidence
from hydra2.training.objectives_metrics import _per_type_breakdown as _per_type_breakdown
from hydra2.training.objectives_metrics import _support_min_max as _support_min_max
from hydra2.training.objectives_metrics import _validate_metric_inputs as _validate_metric_inputs
