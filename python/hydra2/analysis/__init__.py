"""Hydra2 analysis qualification — offline analysis mode (M12/WP-12)."""

from hydra2.analysis.qual_budget import (
    ANALYSIS_CANDIDATE_IDS,
    analysis_budget_for,
    make_analysis_spec,
    verify_compute_only,
)
from hydra2.analysis.qual_gates import (
    ANALYSIS_REPORT_KIND,
    AnalysisGateRecord,
    AnalysisReport,
    analysis_gate_for,
    compute_only_proof,
    generate_hashed_analysis_report,
)
from hydra2.analysis.qual_replay import (
    deterministic_replay_hash,
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
