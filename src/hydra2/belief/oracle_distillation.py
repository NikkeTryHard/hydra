"""WP-07B oracle belief distillation — teacher-student deterministic.

Re-export facade over the split modules: :mod:`hydra2.belief.oracle_models`
(privileged teacher, actor-visible student, distillation loss) and
:mod:`hydra2.belief.oracle_scores` (held-out proper scores, duplicate-wall
comparison, hidden-permutation checks, synthetic distillation runner).
Import from this path; it preserves every public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.belief.oracle_models import (
    DistillationConfig as DistillationConfig,
)
from hydra2.belief.oracle_models import (
    OracleTeacher as OracleTeacher,
)
from hydra2.belief.oracle_models import (
    StudentBeliefModel as StudentBeliefModel,
)
from hydra2.belief.oracle_models import (
    distillation_loss as distillation_loss,
)
from hydra2.belief.oracle_scores import (
    BrierScoreResult as BrierScoreResult,
)
from hydra2.belief.oracle_scores import (
    CalibrationResult as CalibrationResult,
)
from hydra2.belief.oracle_scores import (
    DistillationMetrics as DistillationMetrics,
)
from hydra2.belief.oracle_scores import (
    DuplicateBlockComparison as DuplicateBlockComparison,
)
from hydra2.belief.oracle_scores import (
    ProperScoreResult as ProperScoreResult,
)
from hydra2.belief.oracle_scores import (
    brier_score as brier_score,
)
from hydra2.belief.oracle_scores import (
    calibration_ece as calibration_ece,
)
from hydra2.belief.oracle_scores import (
    compare_duplicate_blocks as compare_duplicate_blocks,
)
from hydra2.belief.oracle_scores import (
    compute_proper_scores as compute_proper_scores,
)
from hydra2.belief.oracle_scores import (
    deterministic_distillation_step as deterministic_distillation_step,
)
from hydra2.belief.oracle_scores import (
    expected_calibration_error as expected_calibration_error,
)
from hydra2.belief.oracle_scores import (
    hidden_permutation_invariance_check as hidden_permutation_invariance_check,
)
from hydra2.belief.oracle_scores import (
    run_synthetic_distillation_for_metrics as run_synthetic_distillation_for_metrics,
)

__all__ = [
    "BrierScoreResult",
    "CalibrationResult",
    "DistillationConfig",
    "DistillationMetrics",
    "DuplicateBlockComparison",
    "OracleTeacher",
    "ProperScoreResult",
    "StudentBeliefModel",
    "brier_score",
    "calibration_ece",
    "compare_duplicate_blocks",
    "compute_proper_scores",
    "distillation_loss",
    "expected_calibration_error",
    "hidden_permutation_invariance_check",
]
