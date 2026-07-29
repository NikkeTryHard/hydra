"""Hydra2 analysis qualification — offline analysis mode (M12/WP-12)."""

from hydra2.analysis.qualification import (
    ANALYSIS_CANDIDATE_IDS,
    ANALYSIS_REPORT_KIND,
    AnalysisGateRecord,
    AnalysisReport,
    analysis_budget_for,
    analysis_gate_for,
    compute_only_proof,
    deterministic_replay_hash,
    generate_hashed_analysis_report,
    make_analysis_spec,
    verify_compute_only,
)

__all__ = [
    "ANALYSIS_CANDIDATE_IDS",
    "ANALYSIS_REPORT_KIND",
    "AnalysisGateRecord",
    "AnalysisReport",
    "analysis_budget_for",
    "analysis_gate_for",
    "compute_only_proof",
    "deterministic_replay_hash",
    "generate_hashed_analysis_report",
    "make_analysis_spec",
    "verify_compute_only",
]
