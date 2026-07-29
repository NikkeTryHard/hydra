"""Hydra2 distillation package — WP-10 Candidate 7 Teacher Distillation.

Owned: distillation/teacher (BUILD §13 / SPEC 16.8).
"""

from __future__ import annotations

from hydra2.distillation.teacher import (
    DistillationConfig,
    TeacherJustification,
    TrajectoryRecord,
    audit_leakage,
    build_student_model,
    check_teacher_replacement_invalidates,
    compute_distillation_loss,
    evaluate_five_arms,
    generate_trajectories,
    load_analysis_gate,
    select_teacher,
    validate_trajectory_record,
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
