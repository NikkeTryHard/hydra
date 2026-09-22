"""Hydra2 distillation package — WP-10 Candidate 7 Teacher Distillation.

Owned: distillation/teacher — teacher-eligibility gate, trajectory records,
five-arm evaluation, and student training.
"""

from __future__ import annotations

from hydra2.distillation._teacher_eval import (
    audit_leakage,
    check_teacher_replacement_invalidates,
    evaluate_five_arms,
)
from hydra2.distillation._teacher_gate import (
    TeacherJustification,
    load_analysis_gate,
    select_teacher,
)
from hydra2.distillation._teacher_records import (
    TrajectoryRecord,
    generate_trajectories,
    validate_trajectory_record,
)
from hydra2.distillation._teacher_student import (
    DistillationConfig,
    build_student_model,
    compute_distillation_loss,
)

__all__ = [
    "DistillationConfig",
    "TeacherJustification",
    "TrajectoryRecord",
    "audit_leakage",
    "build_student_model",
    "check_teacher_replacement_invalidates",
    "compute_distillation_loss",
    "evaluate_five_arms",
    "generate_trajectories",
    "load_analysis_gate",
    "select_teacher",
    "validate_trajectory_record",
]
