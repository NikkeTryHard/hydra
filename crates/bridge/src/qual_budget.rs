//! qual_budget: frozen WP-12 analysis/gameplay budget tables + pure validators.
//!
//! DAG: pyo3 ONLY (frozen budget literals + pure int arithmetic; no feed/search-crate
//! edge — the `ResourceBudget`/`CandidateSpec` dataclasses stay Python in
//! `search/common.py`, the privilege firewall stays Python in `qual_budget.py` over
//! `contracts::observation_actor`, and digest validation stays canon-owned).
//! FORBIDS: dataclass construction, digest/hash math, file/JSON IO, wall-clock,
//! torch/CUDA, and builder/orchestration logic (spec derivation, compute-only proofs,
//! leak checks — callers pass validated scalar fields across; the dataclasses stay
//! Python in `analysis/qual_budget.py`).
//!
//! PyO3 precedent cites (one per API, all read this wave): `pub fn register(sub)` +
//! `let py = sub.py()` mirror `action_artifact::register`
//! (`crates/bridge/src/action_artifact.rs:42-43`); `sub.add` for consts mirrors
//! `action_artifact::register` (`crates/bridge/src/action_artifact.rs:44`);
//! `PyTuple::new` for the frozen rows mirrors the template-fields const
//! (`crates/bridge/src/action_artifact.rs:47`); `#[pyfunction]` mirrors
//! `contracts::is_seat` (`crates/bridge/src/contracts.rs:87-88`);
//! `wrap_pyfunction!(f, sub)` with a borrowed `sub` mirrors
//! `search_profiles::register` (`crates/bridge/src/search_profiles.rs:371-375`);
//! attached-extract + one `py.detach` with zero Python API inside mirrors
//! `search_profiles::profiles_jobs_for`
//! (`crates/bridge/src/search_profiles.rs:189-201`); `ValueError` mapping mirrors
//! `search::persistence_pick_batch`
//! (`crates/bridge/src/search.rs:1615-1628,1669`); `is_instance_of::<PyBool>`
//! bool-rejection mirrors `contracts::plain_int_value`
//! (`crates/bridge/src/contracts.rs:69-74`); `Option<Bound<'_, PyAny>>` for nullable
//! caps mirrors `search_profiles::profiles_admit`
//! (`crates/bridge/src/search_profiles.rs:241-249`);
//! `#[allow(clippy::too_many_arguments)]` mirrors `search_profiles::profiles_admit`
//! (`crates/bridge/src/search_profiles.rs:240`).
//!
//! Single-cdylib tree: registers its consts + pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `search_profiles::register` on the shared submodule); no new entry point.
//! Wiring is MAIN-ONLY (`search::register` calls this `register` with its `sub`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyTuple};

/// Frozen gameplay deadline (`qual_budget.py:109-163`): every gameplay row declares 5,000 ms.
pub const QUAL_BUDGET_GAMEPLAY_DEADLINE_MS: u64 = 5_000;
/// Frozen analysis deadline (`qual_budget.py:47-101`): every analysis row declares 30,000 ms.
pub const QUAL_BUDGET_ANALYSIS_DEADLINE_MS: u64 = 30_000;
/// Analysis deadline ceiling (`_require_finite_budget`, `qual_budget.py:175-178`):
/// analysis deadlines live in `(5000, 300000]`.
pub const QUAL_BUDGET_MAX_DEADLINE_MS: u64 = 300_000;
/// Analysis memory ceiling (`qual_budget.py:189-192`): `<= 64 GiB`, always finite.
pub const QUAL_BUDGET_MAX_MEMORY_BYTES: u64 = 64u64 * 1024 * 1024 * 1024;
/// Generic-fallback memory (`_derive_generic_analysis_budget`, `qual_budget.py:322`): 8 GiB.
pub const QUAL_BUDGET_GENERIC_MEMORY_BYTES: u64 = 8u64 * 1024 * 1024 * 1024;
/// Generic-fallback margin (`qual_budget.py:307-312`): 500 ms, capped via `min(val, 500)`.
pub const QUAL_BUDGET_GENERIC_FALLBACK_MS: u64 = 500;
/// Generic-fallback model-call floor (`qual_budget.py:319`): `max(v * 4, 256)`.
pub const QUAL_BUDGET_GENERIC_CALLS_FLOOR: u64 = 256;
/// Generic-fallback transition floor (`qual_budget.py:320`): `max(v * 4, 1024)`.
pub const QUAL_BUDGET_GENERIC_TRANSITIONS_FLOOR: u64 = 1024;

/// Teacher-eligible Candidate 0-6 registry (`ANALYSIS_CANDIDATE_IDS`,
/// `qual_budget.py:30-38`): WP-12 normative order. Candidate 4 is the control forest
/// `candidate4_core_control`; the persistence factorial is not a teacher candidate.
const CANDIDATE_IDS: [&str; 7] = [
    "candidate0",
    "candidate1",
    "candidate2",
    "candidate3_pbrf_core_v1",
    "candidate4_core_control",
    "candidate5",
    "candidate6",
];

/// One frozen analysis row (`ANALYSIS_BUDGETS`, `qual_budget.py:44-102`):
/// `(candidate_id, deadline_ms, fallback_margin_ms, max_model_calls, max_transitions,
/// max_particles, max_memory_bytes)`. Every cap is finite; candidate0 runs the frozen
/// policy (4 calls / 16 transitions / 2 GiB), the rest run full search
/// (256 calls / 1024 transitions / 64 particles / 8 GiB).
struct AnalysisRow {
    id: &'static str,
    deadline_ms: u64,
    fallback_margin_ms: u64,
    max_model_calls: u64,
    max_transitions: u64,
    max_particles: u64,
    max_memory_bytes: u64,
}

const ANALYSIS_ROWS: [AnalysisRow; 7] = [
    AnalysisRow {
        id: "candidate0",
        deadline_ms: 30_000,
        fallback_margin_ms: 500,
        max_model_calls: 4,
        max_transitions: 16,
        max_particles: 0,
        max_memory_bytes: 2u64 * 1024 * 1024 * 1024,
    },
    AnalysisRow {
        id: "candidate1",
        deadline_ms: 30_000,
        fallback_margin_ms: 500,
        max_model_calls: 256,
        max_transitions: 1024,
        max_particles: 64,
        max_memory_bytes: 8u64 * 1024 * 1024 * 1024,
    },
    AnalysisRow {
        id: "candidate2",
        deadline_ms: 30_000,
        fallback_margin_ms: 500,
        max_model_calls: 256,
        max_transitions: 1024,
        max_particles: 64,
        max_memory_bytes: 8u64 * 1024 * 1024 * 1024,
    },
    AnalysisRow {
        id: "candidate3_pbrf_core_v1",
        deadline_ms: 30_000,
        fallback_margin_ms: 500,
        max_model_calls: 256,
        max_transitions: 1024,
        max_particles: 64,
        max_memory_bytes: 8u64 * 1024 * 1024 * 1024,
    },
    AnalysisRow {
        id: "candidate4_core_control",
        deadline_ms: 30_000,
        fallback_margin_ms: 500,
        max_model_calls: 256,
        max_transitions: 1024,
        max_particles: 64,
        max_memory_bytes: 8u64 * 1024 * 1024 * 1024,
    },
    AnalysisRow {
        id: "candidate5",
        deadline_ms: 30_000,
        fallback_margin_ms: 500,
        max_model_calls: 256,
        max_transitions: 1024,
        max_particles: 64,
        max_memory_bytes: 8u64 * 1024 * 1024 * 1024,
    },
    AnalysisRow {
        id: "candidate6",
        deadline_ms: 30_000,
        fallback_margin_ms: 500,
        max_model_calls: 256,
        max_transitions: 1024,
        max_particles: 64,
        max_memory_bytes: 8u64 * 1024 * 1024 * 1024,
    },
];

/// One frozen gameplay row (`GAMEPLAY_BUDGETS`, `qual_budget.py:107-164`):
/// gameplay `max_memory_bytes` is `None` for every row, so it rides implicitly
/// (Python rebuilds the dicts with `None`); the other five caps are finite ints.
/// Particle counts are 16 for the PBRF pair (candidate3/4), 32 elsewhere, 0 for
/// the frozen policy (candidate0).
struct GameplayRow {
    id: &'static str,
    deadline_ms: u64,
    fallback_margin_ms: u64,
    max_model_calls: u64,
    max_transitions: u64,
    max_particles: u64,
}

const GAMEPLAY_ROWS: [GameplayRow; 7] = [
    GameplayRow {
        id: "candidate0",
        deadline_ms: 5_000,
        fallback_margin_ms: 500,
        max_model_calls: 1,
        max_transitions: 0,
        max_particles: 0,
    },
    GameplayRow {
        id: "candidate1",
        deadline_ms: 5_000,
        fallback_margin_ms: 200,
        max_model_calls: 64,
        max_transitions: 256,
        max_particles: 32,
    },
    GameplayRow {
        id: "candidate2",
        deadline_ms: 5_000,
        fallback_margin_ms: 200,
        max_model_calls: 64,
        max_transitions: 256,
        max_particles: 32,
    },
    GameplayRow {
        id: "candidate3_pbrf_core_v1",
        deadline_ms: 5_000,
        fallback_margin_ms: 200,
        max_model_calls: 64,
        max_transitions: 256,
        max_particles: 16,
    },
    GameplayRow {
        id: "candidate4_core_control",
        deadline_ms: 5_000,
        fallback_margin_ms: 200,
        max_model_calls: 64,
        max_transitions: 256,
        max_particles: 16,
    },
    GameplayRow {
        id: "candidate5",
        deadline_ms: 5_000,
        fallback_margin_ms: 200,
        max_model_calls: 64,
        max_transitions: 256,
        max_particles: 32,
    },
    GameplayRow {
        id: "candidate6",
        deadline_ms: 5_000,
        fallback_margin_ms: 200,
        max_model_calls: 64,
        max_transitions: 256,
        max_particles: 32,
    },
];

/// Read a Python int that must not be a bool (`contracts.rs:69-74` `plain_int_value`
/// shape): bools, `None`, non-ints, negatives-when-unsigned, and out-of-`u64` values
/// all fail closed with the caller's message.
fn as_u64(obj: &Bound<'_, PyAny>, err: &str) -> Result<u64, String> {
    if obj.is_instance_of::<PyBool>() {
        return Err(err.to_string());
    }
    obj.extract::<u64>().map_err(|_| err.to_string())
}

/// Nullable variant for gameplay caps that may be `None` (`search_profiles.rs:249`
/// `Option<Bound<'_, PyAny>>` shape): `None` stays `None`; bools/non-ints fail closed.
fn as_opt_u64(arg: &Option<Bound<'_, PyAny>>, err: &str) -> Result<Option<u64>, String> {
    match arg {
        None => Ok(None),
        Some(obj) => as_u64(obj, err).map(Some),
    }
}

/// Table-lookup half of `analysis_budget_for` (`qual_budget.py:205-207`). Pure and
/// detach-safe; unknown ids raise, never a default.
fn lookup_analysis(candidate_id: &str) -> Result<&'static AnalysisRow, String> {
    ANALYSIS_ROWS
        .iter()
        .find(|row| row.id == candidate_id)
        .ok_or_else(|| format!("unknown candidate_id {candidate_id:?} for analysis budget"))
}

/// Range half of `_require_finite_budget` (`qual_budget.py:173-198`): mode, deadline,
/// fallback, and caps. Pure and detach-safe. `None`/bool/non-int rejection happens
/// attached (the extractors above), so this sees only finite `u64`s: the `max_particles`
/// nonneg arm is structural (`u64` cannot go negative) and the positive arms gate on
/// zero exactly like the oracle's `v <= 0` rejects.
fn check_finite_fields(
    mode: &str,
    deadline_ms: u64,
    fallback_margin_ms: u64,
    max_model_calls: u64,
    max_transitions: u64,
    max_particles: u64,
    max_memory_bytes: u64,
) -> Result<(), String> {
    if mode != "analysis" {
        return Err(format!(
            "analysis budget mode must be 'analysis', got {mode:?}"
        ));
    }
    if deadline_ms <= QUAL_BUDGET_GAMEPLAY_DEADLINE_MS || deadline_ms > QUAL_BUDGET_MAX_DEADLINE_MS
    {
        return Err(format!(
            "analysis deadline_ms must be finite >5000 and <=300000, got {deadline_ms}"
        ));
    }
    if fallback_margin_ms >= deadline_ms {
        return Err(format!(
            "fallback_margin_ms {fallback_margin_ms} must be in [0, deadline_ms)"
        ));
    }
    if max_model_calls == 0 {
        return Err("analysis max_model_calls must be finite positive int, got 0".to_string());
    }
    if max_transitions == 0 {
        return Err("analysis max_transitions must be finite positive int, got 0".to_string());
    }
    let _ = max_particles;
    if max_memory_bytes == 0 || max_memory_bytes > QUAL_BUDGET_MAX_MEMORY_BYTES {
        return Err(format!(
            "analysis max_memory_bytes must be finite positive <=64GiB, got {max_memory_bytes}"
        ));
    }
    Ok(())
}

/// `_enlarge` (`qual_budget.py:301-304`): `None` reads as the floor, otherwise
/// `max(v * 4, floor)` saturating (Python ints are unbounded; saturation only fires on
/// adversarial direct-bridge calls since no real budget exceeds `u64::MAX / 4`).
/// Pure and detach-safe.
fn enlarge(value: Option<u64>, floor: u64) -> u64 {
    match value {
        None => floor,
        Some(v) => v.saturating_mul(4).max(floor),
    }
}

/// Generic-budget field kernel (`_derive_generic_analysis_budget`,
/// `qual_budget.py:297-323`): deadline 30 s, fallback `min(val, 500)` defaulting to 500
/// when the gameplay budget carries none, calls/transitions enlarged 4x onto the
/// 256/1024 floors, particles preserved-or-zero, memory 8 GiB. Pure and detach-safe.
/// (Direct-bridge bool/non-int gameplay fields fail closed attached; the oracle's
/// lenient non-int-fallback-reads-as-500 arm only triggers on synthetic stubs that
/// can never be real `ResourceBudget`s.)
fn generic_fields(
    gp_calls: Option<u64>,
    gp_transitions: Option<u64>,
    gp_particles: Option<u64>,
    gp_fallback: Option<u64>,
) -> (u64, u64, u64, u64, u64, u64) {
    (
        QUAL_BUDGET_ANALYSIS_DEADLINE_MS,
        gp_fallback.map_or(QUAL_BUDGET_GENERIC_FALLBACK_MS, |v| {
            v.min(QUAL_BUDGET_GENERIC_FALLBACK_MS)
        }),
        enlarge(gp_calls, QUAL_BUDGET_GENERIC_CALLS_FLOOR),
        enlarge(gp_transitions, QUAL_BUDGET_GENERIC_TRANSITIONS_FLOOR),
        gp_particles.unwrap_or(0),
        QUAL_BUDGET_GENERIC_MEMORY_BYTES,
    )
}

/// Frozen-budget lookup (`analysis_budget_for` table half, `qual_budget.py:205-207`):
/// attached arg move, detached scan, unknown ids raise, never a default.
#[pyfunction]
fn qual_budget_for(
    py: Python<'_>,
    candidate_id: String,
) -> PyResult<(u64, u64, u64, u64, u64, u64)> {
    py.detach(|| lookup_analysis(&candidate_id))
        .map(|row| {
            (
                row.deadline_ms,
                row.fallback_margin_ms,
                row.max_model_calls,
                row.max_transitions,
                row.max_particles,
                row.max_memory_bytes,
            )
        })
        .map_err(PyValueError::new_err)
}

/// Finite-budget validator over validated scalar fields (`_require_finite_budget`,
/// `qual_budget.py:167-198`): attached bool/`None`-rejecting extraction (like
/// `profiles_check_profile`, `search_profiles.rs:166-183`), detached range checks;
/// `ValueError` on any reject, never a default. The live-`ResourceBudget` type gate
/// stays Python in the caller.
#[pyfunction]
#[pyo3(signature = (
    mode, deadline_ms, fallback_margin_ms, max_model_calls, max_transitions,
    max_particles, max_memory_bytes
))]
#[allow(clippy::too_many_arguments)]
fn qual_budget_check_finite(
    py: Python<'_>,
    mode: String,
    deadline_ms: Bound<'_, PyAny>,
    fallback_margin_ms: Bound<'_, PyAny>,
    max_model_calls: Bound<'_, PyAny>,
    max_transitions: Bound<'_, PyAny>,
    max_particles: Bound<'_, PyAny>,
    max_memory_bytes: Bound<'_, PyAny>,
) -> PyResult<()> {
    let deadline = as_u64(
        &deadline_ms,
        "analysis deadline_ms must be finite >5000 and <=300000",
    )
    .map_err(PyValueError::new_err)?;
    let fallback = as_u64(
        &fallback_margin_ms,
        "analysis fallback_margin_ms must satisfy 0 <= margin < deadline",
    )
    .map_err(PyValueError::new_err)?;
    let calls = as_u64(
        &max_model_calls,
        "analysis max_model_calls must be finite positive int",
    )
    .map_err(PyValueError::new_err)?;
    let transitions = as_u64(
        &max_transitions,
        "analysis max_transitions must be finite positive int",
    )
    .map_err(PyValueError::new_err)?;
    let particles = as_u64(
        &max_particles,
        "analysis max_particles must be finite nonneg int",
    )
    .map_err(PyValueError::new_err)?;
    let memory = as_u64(
        &max_memory_bytes,
        "analysis max_memory_bytes must be finite (non-None) cap",
    )
    .map_err(PyValueError::new_err)?;
    py.detach(|| {
        check_finite_fields(
            &mode,
            deadline,
            fallback,
            calls,
            transitions,
            particles,
            memory,
        )
    })
    .map_err(PyValueError::new_err)
}

/// Cap-enlargement kernel (`_enlarge`, `qual_budget.py:301-304`): attached extraction,
/// detached multiply-max; `ValueError` on bool/non-int, never a default.
#[pyfunction]
#[pyo3(signature = (value, floor))]
fn qual_budget_enlarge(
    py: Python<'_>,
    value: Option<Bound<'_, PyAny>>,
    floor: Bound<'_, PyAny>,
) -> PyResult<u64> {
    let floor = as_u64(&floor, "qual_budget_enlarge floor must be a plain int")
        .map_err(PyValueError::new_err)?;
    let v = as_opt_u64(
        &value,
        "qual_budget_enlarge value must be a plain int or None",
    )
    .map_err(PyValueError::new_err)?;
    Ok(py.detach(|| enlarge(v, floor)))
}

/// Generic-budget derivation over validated scalar fields
/// (`_derive_generic_analysis_budget`, `qual_budget.py:297-323`): attached
/// bool/`None`-tolerant extraction, detached field kernel; `ValueError` on
/// bool/non-int caps, never a default. `ResourceBudget` construction stays Python.
#[pyfunction]
#[pyo3(signature = (
    gp_max_model_calls, gp_max_transitions, gp_max_particles, gp_fallback_margin_ms
))]
#[allow(clippy::too_many_arguments)]
fn qual_budget_generic_budget(
    py: Python<'_>,
    gp_max_model_calls: Option<Bound<'_, PyAny>>,
    gp_max_transitions: Option<Bound<'_, PyAny>>,
    gp_max_particles: Option<Bound<'_, PyAny>>,
    gp_fallback_margin_ms: Option<Bound<'_, PyAny>>,
) -> PyResult<(u64, u64, u64, u64, u64, u64)> {
    let calls = as_opt_u64(
        &gp_max_model_calls,
        "qual_budget_generic_budget gp_max_model_calls must be a plain int or None",
    )
    .map_err(PyValueError::new_err)?;
    let transitions = as_opt_u64(
        &gp_max_transitions,
        "qual_budget_generic_budget gp_max_transitions must be a plain int or None",
    )
    .map_err(PyValueError::new_err)?;
    let particles = as_opt_u64(
        &gp_max_particles,
        "qual_budget_generic_budget gp_max_particles must be a plain int or None",
    )
    .map_err(PyValueError::new_err)?;
    let fallback = as_opt_u64(
        &gp_fallback_margin_ms,
        "qual_budget_generic_budget gp_fallback_margin_ms must be a plain int or None",
    )
    .map_err(PyValueError::new_err)?;
    Ok(py.detach(|| generic_fields(calls, transitions, particles, fallback)))
}

/// Register the budget consts + kernels on the EXISTING `search` submodule
/// (mirrors `search_profiles::register`, `search_profiles.rs:364-377`, on the
/// shared submodule; MAIN calls this from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add(
        "QUAL_BUDGET_GAMEPLAY_DEADLINE_MS",
        QUAL_BUDGET_GAMEPLAY_DEADLINE_MS,
    )?;
    sub.add(
        "QUAL_BUDGET_ANALYSIS_DEADLINE_MS",
        QUAL_BUDGET_ANALYSIS_DEADLINE_MS,
    )?;
    sub.add("QUAL_BUDGET_MAX_DEADLINE_MS", QUAL_BUDGET_MAX_DEADLINE_MS)?;
    sub.add("QUAL_BUDGET_MAX_MEMORY_BYTES", QUAL_BUDGET_MAX_MEMORY_BYTES)?;
    sub.add(
        "QUAL_BUDGET_GENERIC_MEMORY_BYTES",
        QUAL_BUDGET_GENERIC_MEMORY_BYTES,
    )?;
    sub.add(
        "QUAL_BUDGET_GENERIC_FALLBACK_MS",
        QUAL_BUDGET_GENERIC_FALLBACK_MS,
    )?;
    sub.add(
        "QUAL_BUDGET_GENERIC_CALLS_FLOOR",
        QUAL_BUDGET_GENERIC_CALLS_FLOOR,
    )?;
    sub.add(
        "QUAL_BUDGET_GENERIC_TRANSITIONS_FLOOR",
        QUAL_BUDGET_GENERIC_TRANSITIONS_FLOOR,
    )?;
    sub.add(
        "QUAL_BUDGET_CANDIDATE_IDS",
        PyTuple::new(py, CANDIDATE_IDS)?,
    )?;
    let analysis_rows: Vec<(&str, u64, u64, u64, u64, u64, u64)> = ANALYSIS_ROWS
        .iter()
        .map(|row| {
            (
                row.id,
                row.deadline_ms,
                row.fallback_margin_ms,
                row.max_model_calls,
                row.max_transitions,
                row.max_particles,
                row.max_memory_bytes,
            )
        })
        .collect();
    sub.add(
        "QUAL_BUDGET_ANALYSIS_ROWS",
        PyTuple::new(py, analysis_rows)?,
    )?;
    let gameplay_rows: Vec<(&str, u64, u64, u64, u64, u64)> = GAMEPLAY_ROWS
        .iter()
        .map(|row| {
            (
                row.id,
                row.deadline_ms,
                row.fallback_margin_ms,
                row.max_model_calls,
                row.max_transitions,
                row.max_particles,
            )
        })
        .collect();
    sub.add(
        "QUAL_BUDGET_GAMEPLAY_ROWS",
        PyTuple::new(py, gameplay_rows)?,
    )?;
    sub.add_function(wrap_pyfunction!(qual_budget_for, sub)?)?;
    sub.add_function(wrap_pyfunction!(qual_budget_check_finite, sub)?)?;
    sub.add_function(wrap_pyfunction!(qual_budget_enlarge, sub)?)?;
    sub.add_function(wrap_pyfunction!(qual_budget_generic_budget, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod qual_budget_tests {
    use super::*;

    fn analysis_tuple(row: &AnalysisRow) -> (&'static str, u64, u64, u64, u64, u64, u64) {
        (
            row.id,
            row.deadline_ms,
            row.fallback_margin_ms,
            row.max_model_calls,
            row.max_transitions,
            row.max_particles,
            row.max_memory_bytes,
        )
    }

    #[test]
    fn frozen_ids_match_oracle_registry() {
        assert_eq!(
            CANDIDATE_IDS,
            [
                "candidate0",
                "candidate1",
                "candidate2",
                "candidate3_pbrf_core_v1",
                "candidate4_core_control",
                "candidate5",
                "candidate6",
            ]
        );
    }

    #[test]
    fn frozen_analysis_rows_match_oracle() {
        let got: Vec<(&'static str, u64, u64, u64, u64, u64, u64)> =
            ANALYSIS_ROWS.iter().map(analysis_tuple).collect();
        let gib = 1024 * 1024 * 1024;
        assert_eq!(
            got,
            vec![
                ("candidate0", 30_000, 500, 4, 16, 0, 2 * gib),
                ("candidate1", 30_000, 500, 256, 1024, 64, 8 * gib),
                ("candidate2", 30_000, 500, 256, 1024, 64, 8 * gib),
                (
                    "candidate3_pbrf_core_v1",
                    30_000,
                    500,
                    256,
                    1024,
                    64,
                    8 * gib
                ),
                (
                    "candidate4_core_control",
                    30_000,
                    500,
                    256,
                    1024,
                    64,
                    8 * gib
                ),
                ("candidate5", 30_000, 500, 256, 1024, 64, 8 * gib),
                ("candidate6", 30_000, 500, 256, 1024, 64, 8 * gib),
            ]
        );
    }

    #[test]
    fn frozen_gameplay_rows_match_oracle() {
        let got: Vec<(&'static str, u64, u64, u64, u64, u64)> = GAMEPLAY_ROWS
            .iter()
            .map(|row| {
                (
                    row.id,
                    row.deadline_ms,
                    row.fallback_margin_ms,
                    row.max_model_calls,
                    row.max_transitions,
                    row.max_particles,
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                ("candidate0", 5_000, 500, 1, 0, 0),
                ("candidate1", 5_000, 200, 64, 256, 32),
                ("candidate2", 5_000, 200, 64, 256, 32),
                ("candidate3_pbrf_core_v1", 5_000, 200, 64, 256, 16),
                ("candidate4_core_control", 5_000, 200, 64, 256, 16),
                ("candidate5", 5_000, 200, 64, 256, 32),
                ("candidate6", 5_000, 200, 64, 256, 32),
            ]
        );
    }

    #[test]
    fn bound_consts_match_oracle() {
        assert_eq!(
            (
                QUAL_BUDGET_GAMEPLAY_DEADLINE_MS,
                QUAL_BUDGET_ANALYSIS_DEADLINE_MS,
                QUAL_BUDGET_MAX_DEADLINE_MS,
            ),
            (5_000, 30_000, 300_000)
        );
        assert_eq!(QUAL_BUDGET_MAX_MEMORY_BYTES, 64u64 * 1024 * 1024 * 1024);
        assert_eq!(
            (
                QUAL_BUDGET_GENERIC_MEMORY_BYTES,
                QUAL_BUDGET_GENERIC_FALLBACK_MS,
                QUAL_BUDGET_GENERIC_CALLS_FLOOR,
                QUAL_BUDGET_GENERIC_TRANSITIONS_FLOOR,
            ),
            (8u64 * 1024 * 1024 * 1024, 500, 256, 1024)
        );
    }

    #[test]
    fn finite_validator_matches_oracle_boundaries() {
        let gib = 1024 * 1024 * 1024;
        assert!(check_finite_fields("analysis", 30_000, 500, 4, 16, 0, 2 * gib).is_ok());
        assert!(check_finite_fields("analysis", 30_000, 500, 256, 1024, 64, 8 * gib).is_ok());
        assert!(check_finite_fields("gameplay_5s", 30_000, 500, 4, 16, 0, 2 * gib).is_err());
        assert!(check_finite_fields("", 30_000, 500, 4, 16, 0, 2 * gib).is_err());
        assert!(check_finite_fields("analysis", 5_000, 500, 4, 16, 0, 2 * gib).is_err());
        assert!(check_finite_fields("analysis", 5_001, 500, 4, 16, 0, 2 * gib).is_ok());
        assert!(check_finite_fields("analysis", 300_000, 500, 4, 16, 0, 2 * gib).is_ok());
        assert!(check_finite_fields("analysis", 300_001, 500, 4, 16, 0, 2 * gib).is_err());
        assert!(check_finite_fields("analysis", 30_000, 30_000, 4, 16, 0, 2 * gib).is_err());
        assert!(check_finite_fields("analysis", 30_000, 29_999, 4, 16, 0, 2 * gib).is_ok());
        assert!(check_finite_fields("analysis", 30_000, 0, 4, 16, 0, 2 * gib).is_ok());
        assert!(check_finite_fields("analysis", 30_000, 500, 0, 16, 0, 2 * gib).is_err());
        assert!(check_finite_fields("analysis", 30_000, 500, 4, 0, 0, 2 * gib).is_err());
        assert!(check_finite_fields("analysis", 30_000, 500, 4, 16, 0, 0).is_err());
        assert!(check_finite_fields("analysis", 30_000, 500, 4, 16, 0, 64 * gib).is_ok());
        assert!(check_finite_fields("analysis", 30_000, 500, 4, 16, 0, 64 * gib + 1).is_err());
    }

    #[test]
    fn frozen_analysis_rows_are_finite() {
        for row in ANALYSIS_ROWS.iter() {
            assert!(
                check_finite_fields(
                    "analysis",
                    row.deadline_ms,
                    row.fallback_margin_ms,
                    row.max_model_calls,
                    row.max_transitions,
                    row.max_particles,
                    row.max_memory_bytes,
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn enlarge_matches_oracle() {
        assert_eq!(enlarge(None, 256), 256);
        assert_eq!(enlarge(None, 1024), 1024);
        assert_eq!(enlarge(Some(64), 256), 256);
        assert_eq!(enlarge(Some(100), 256), 400);
        assert_eq!(enlarge(Some(256), 1024), 1024);
        assert_eq!(enlarge(Some(300), 1024), 1200);
        assert_eq!(enlarge(Some(1), 256), 256);
        assert_eq!(enlarge(Some(u64::MAX), 256), u64::MAX);
    }

    #[test]
    fn generic_fields_match_oracle() {
        let gib = 1024 * 1024 * 1024;
        assert_eq!(
            generic_fields(None, None, None, None),
            (30_000, 500, 256, 1024, 0, 8 * gib)
        );
        assert_eq!(
            generic_fields(Some(64), Some(256), Some(32), Some(200)),
            (30_000, 200, 256, 1024, 32, 8 * gib)
        );
        assert_eq!(
            generic_fields(Some(1), Some(1), Some(0), Some(900)),
            (30_000, 500, 256, 1024, 0, 8 * gib)
        );
        let fields = generic_fields(Some(64), Some(256), Some(16), Some(200));
        assert!(fields.0 > QUAL_BUDGET_GAMEPLAY_DEADLINE_MS);
        assert!(fields.2 >= 64 && fields.3 >= 256 && fields.4 >= 16);
        assert!(
            check_finite_fields(
                "analysis", fields.0, fields.1, fields.2, fields.3, fields.4, fields.5
            )
            .is_ok()
        );
    }

    #[test]
    fn generic_from_gameplay_rows_is_finite() {
        for row in GAMEPLAY_ROWS.iter() {
            let (deadline, fallback, calls, transitions, particles, memory) = generic_fields(
                Some(row.max_model_calls),
                Some(row.max_transitions),
                Some(row.max_particles),
                Some(row.fallback_margin_ms),
            );
            assert!(deadline > row.deadline_ms);
            assert!(
                check_finite_fields(
                    "analysis",
                    deadline,
                    fallback,
                    calls,
                    transitions,
                    particles,
                    memory,
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn lookup_matches_oracle() {
        assert_eq!(
            lookup_analysis("candidate0").map(|row| row.max_model_calls),
            Ok(4)
        );
        assert_eq!(
            lookup_analysis("candidate6").map(|row| row.max_particles),
            Ok(64)
        );
        assert!(lookup_analysis("candidate7").is_err());
        assert!(lookup_analysis("").is_err());
        assert!(lookup_analysis("persistence-factorial").is_err());
    }
}
