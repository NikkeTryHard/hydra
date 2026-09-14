"""WP-05C Baseline Qualification — metrics, held-out, deterministic, report.

Re-export facade over the split modules: :mod:`hydra2.eval.baseline_metrics`
(masked NLL, top-k, calibration, legal-uniform comparison, metric bundle,
held-out partition) and :mod:`hydra2.eval.baseline_eval` (fresh-process
repeat, tiny-shard overfit, reference games, permutation invariance,
canonical report). Import from this path; it preserves every public name
and ``__all__``.
"""

from __future__ import annotations

from hydra2.eval.baseline_eval import (
    BaselineReport as BaselineReport,
)
from hydra2.eval.baseline_eval import (
    check_hidden_permutation_invariance as check_hidden_permutation_invariance,
)
from hydra2.eval.baseline_eval import (
    evaluate_reference_games as evaluate_reference_games,
)
from hydra2.eval.baseline_eval import (
    fresh_process_metrics as fresh_process_metrics,
)
from hydra2.eval.baseline_eval import (
    make_baseline_report as make_baseline_report,
)
from hydra2.eval.baseline_eval import (
    tiny_shard_overfit as tiny_shard_overfit,
)
from hydra2.eval.baseline_metrics import (
    BASELINE_METRICS_VERSION as BASELINE_METRICS_VERSION,
)
from hydra2.eval.baseline_metrics import (
    OVERFIT_NLL_THRESHOLD as OVERFIT_NLL_THRESHOLD,
)
from hydra2.eval.baseline_metrics import (
    OVERFIT_TOP1_THRESHOLD as OVERFIT_TOP1_THRESHOLD,
)
from hydra2.eval.baseline_metrics import (
    BaselineMetrics as BaselineMetrics,
)
from hydra2.eval.baseline_metrics import (
    HeldOutSplit as HeldOutSplit,
)
from hydra2.eval.baseline_metrics import (
    compute_baseline_metrics as compute_baseline_metrics,
)
from hydra2.eval.baseline_metrics import (
    expected_calibration_error as expected_calibration_error,
)
from hydra2.eval.baseline_metrics import (
    legal_uniform_nll as legal_uniform_nll,
)
from hydra2.eval.baseline_metrics import (
    masked_cross_entropy as masked_cross_entropy,
)
from hydra2.eval.baseline_metrics import (
    split_held_out as split_held_out,
)
from hydra2.eval.baseline_metrics import (
    top_k_accuracy as top_k_accuracy,
)
from hydra2.eval.baseline_metrics import (
    verify_held_out_disjoint as verify_held_out_disjoint,
)

__all__ = [
    "BASELINE_METRICS_VERSION",
    "OVERFIT_NLL_THRESHOLD",
    "OVERFIT_TOP1_THRESHOLD",
    "BaselineMetrics",
    "BaselineReport",
    "HeldOutSplit",
    "check_hidden_permutation_invariance",
    "compute_baseline_metrics",
    "evaluate_reference_games",
    "expected_calibration_error",
    "fresh_process_metrics",
    "legal_uniform_nll",
    "make_baseline_report",
    "masked_cross_entropy",
    "split_held_out",
    "tiny_shard_overfit",
    "top_k_accuracy",
    "verify_held_out_disjoint",
]

# Names importable from this path before the split that live in the
# submodules now (kept so the fresh-process subprocess and sibling
# modules resolve without touching the new paths).
from hydra2.eval.baseline_eval import _to_jsonable as _to_jsonable
from hydra2.eval.baseline_metrics import COMPILE_ORDER as COMPILE_ORDER
from hydra2.eval.baseline_metrics import EAGER_ORACLE_ID as EAGER_ORACLE_ID
from hydra2.eval.baseline_metrics import _require_finite as _require_finite
from hydra2.eval.baseline_metrics import _require_legal_mask as _require_legal_mask
from hydra2.eval.baseline_metrics import _require_logits as _require_logits
from hydra2.eval.baseline_metrics import _require_targets as _require_targets
from hydra2.eval.baseline_metrics import _seed_everything as _seed_everything
