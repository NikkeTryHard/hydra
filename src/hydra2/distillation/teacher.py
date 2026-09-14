"""WP-10 Candidate 7 Teacher Distillation — deterministic selection, trajectories.

Re-export facade over the split modules: :mod:`hydra2.distillation._teacher_gate`
(registry, analysis-gate loader, selection justification),
:mod:`hydra2.distillation._teacher_cases` (case observations, teacher priors),
:mod:`hydra2.distillation._teacher_records` (trajectory records, generation),
:mod:`hydra2.distillation._teacher_student` (student model, loss, training),
and :mod:`hydra2.distillation._teacher_eval` (five-arm comparison, leakage
audits, frozen helpers). Import from this path; it preserves every public
name and ``__all__``.
"""

from __future__ import annotations

from hydra2.distillation._teacher_cases import _case_observation as _case_observation
from hydra2.distillation._teacher_cases import (
    _teacher_policy_and_value as _teacher_policy_and_value,
)
from hydra2.distillation._teacher_eval import audit_leakage as audit_leakage
from hydra2.distillation._teacher_eval import calibration_report as calibration_report
from hydra2.distillation._teacher_eval import (
    check_teacher_replacement_invalidates as check_teacher_replacement_invalidates,
)
from hydra2.distillation._teacher_eval import evaluate_five_arms as evaluate_five_arms
from hydra2.distillation._teacher_eval import (
    frozen_checkpoint_identity as frozen_checkpoint_identity,
)
from hydra2.distillation._teacher_eval import frozen_split_manifest as frozen_split_manifest
from hydra2.distillation._teacher_eval import (
    student_plus_search_policy as student_plus_search_policy,
)
from hydra2.distillation._teacher_eval import (
    teacher_plus_search_policy as teacher_plus_search_policy,
)
from hydra2.distillation._teacher_gate import REJECTED_CANDIDATES as REJECTED_CANDIDATES
from hydra2.distillation._teacher_gate import TEACHER_CANDIDATES as TEACHER_CANDIDATES
from hydra2.distillation._teacher_gate import TeacherJustification as TeacherJustification
from hydra2.distillation._teacher_gate import _real_candidate_spec as _real_candidate_spec
from hydra2.distillation._teacher_gate import load_analysis_gate as load_analysis_gate
from hydra2.distillation._teacher_gate import select_teacher as select_teacher
from hydra2.distillation._teacher_records import TrajectoryRecord as TrajectoryRecord
from hydra2.distillation._teacher_records import (
    generate_privileged_labels as generate_privileged_labels,
)
from hydra2.distillation._teacher_records import generate_trajectories as generate_trajectories
from hydra2.distillation._teacher_records import make_trajectory_record as make_trajectory_record
from hydra2.distillation._teacher_records import (
    validate_trajectory_record as validate_trajectory_record,
)
from hydra2.distillation._teacher_student import DistillationConfig as DistillationConfig
from hydra2.distillation._teacher_student import StudentModel as StudentModel
from hydra2.distillation._teacher_student import build_student_model as build_student_model
from hydra2.distillation._teacher_student import (
    compute_distillation_loss as compute_distillation_loss,
)
from hydra2.distillation._teacher_student import features_for_record as features_for_record
from hydra2.distillation._teacher_student import (
    train_student_distillation as train_student_distillation,
)

__all__ = [
    "REJECTED_CANDIDATES",
    "TEACHER_CANDIDATES",
    "DistillationConfig",
    "StudentModel",
    "TeacherJustification",
    "TrajectoryRecord",
    "audit_leakage",
    "build_student_model",
    "calibration_report",
    "check_teacher_replacement_invalidates",
    "compute_distillation_loss",
    "evaluate_five_arms",
    "features_for_record",
    "frozen_checkpoint_identity",
    "frozen_split_manifest",
    "generate_privileged_labels",
    "generate_trajectories",
    "load_analysis_gate",
    "make_trajectory_record",
    "select_teacher",
    "student_plus_search_policy",
    "teacher_plus_search_policy",
    "train_student_distillation",
    "validate_trajectory_record",
]
