"""WP-12 Analysis Qualification — offline analysis mode (M12).

Re-export facade over the split modules: :mod:`hydra2.analysis.qual_budget`
(frozen budgets, compute-only proof, privilege firewall),
:mod:`hydra2.analysis.qual_replay` (deterministic replay, mode comparison),
and :mod:`hydra2.analysis.qual_gates` (gate records, candidate factories,
hashed report). Import from this path; it preserves every public name.
"""

from __future__ import annotations

from hydra2.analysis.qual_budget import ANALYSIS_BUDGETS as ANALYSIS_BUDGETS
from hydra2.analysis.qual_budget import ANALYSIS_CANDIDATE_IDS as ANALYSIS_CANDIDATE_IDS
from hydra2.analysis.qual_budget import GAMEPLAY_BUDGETS as GAMEPLAY_BUDGETS
from hydra2.analysis.qual_budget import (
    _derive_generic_analysis_budget as _derive_generic_analysis_budget,
)
from hydra2.analysis.qual_budget import _require_finite_budget as _require_finite_budget
from hydra2.analysis.qual_budget import analysis_budget_for as analysis_budget_for
from hydra2.analysis.qual_budget import check_no_privileged_leak as check_no_privileged_leak
from hydra2.analysis.qual_budget import make_analysis_spec as make_analysis_spec
from hydra2.analysis.qual_budget import verify_compute_only as verify_compute_only
from hydra2.analysis.qual_gates import ANALYSIS_REPORT_KIND as ANALYSIS_REPORT_KIND
from hydra2.analysis.qual_gates import (
    ANALYSIS_REPORT_SCHEMA_VERSION as ANALYSIS_REPORT_SCHEMA_VERSION,
)
from hydra2.analysis.qual_gates import AnalysisGateRecord as AnalysisGateRecord
from hydra2.analysis.qual_gates import AnalysisReport as AnalysisReport
from hydra2.analysis.qual_gates import (
    _load_default_hashes_for_spec as _load_default_hashes_for_spec,
)
from hydra2.analysis.qual_gates import _make_gameplay_spec_for as _make_gameplay_spec_for
from hydra2.analysis.qual_gates import _utc_now as _utc_now
from hydra2.analysis.qual_gates import analysis_gate_for as analysis_gate_for
from hydra2.analysis.qual_gates import build_gate_record as build_gate_record
from hydra2.analysis.qual_gates import compute_only_proof as compute_only_proof
from hydra2.analysis.qual_gates import (
    generate_hashed_analysis_report as generate_hashed_analysis_report,
)
from hydra2.analysis.qual_replay import _spec_hash as _spec_hash
from hydra2.analysis.qual_replay import compare_gameplay_analysis as compare_gameplay_analysis
from hydra2.analysis.qual_replay import deterministic_replay_hash as deterministic_replay_hash
