//! persistence_report: frozen whole-block contrast/strata math over the `hydra-search` owner.
//!
//! DAG: this module depends on the search crate (`persistence_report`,
//! `persistence_kernel`, `persistence_planner`, `eval`) + pyo3 ONLY —
//! arg-check + detach + scalar/vector return. The contrast/strata math lives
//! search-side in `hydra_search::persistence_report::{block_diffs,
//! persistence_contrast_seed, stratify_surprise_miss_recovery}` and the mean
//! in `hydra_search::eval::block_mean` (Neumaier, matching `math.fsum` on the
//! recorded adversarial set); this file only validates arguments attached,
//! runs the pure compute detached, and maps [`SearchError`] onto `ValueError`
//! attached. FORBIDS: bootstrap draws (numpy PCG64 stays the draw source in
//! `eval::statistics::bootstrap_blocks` — intervals are caller-supplied here),
//! canon bytes/hash math (feed-owned; the block-manifest hash stays Python),
//! wall-clock (report timestamps stay Python), resource synthesis
//! (deterministic-gumbel rows stay Python), and per-planner state (the
//! `PersistencePlanner` arena design does not exist yet).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + `let py =
//! sub.py()` mirror `action_artifact::register`
//! (`crates/bridge/src/action_artifact.rs:42-43`); `#[pyfunction]` mirrors
//! `eval::aggregate_wall_block` (`crates/bridge/src/eval.rs:91-98`);
//! `wrap_pyfunction!(f, sub)` with a borrowed `sub` mirrors
//! `rows_seal::register` (`crates/bridge/src/rows_seal.rs:485`);
//! `py.detach` + `search_err` mirror `search::gumbel_for_action`
//! (`crates/bridge/src/search.rs:180-182,272`); `PyTuple::new` for the frozen
//! pairs mirrors the template-fields const
//! (`crates/bridge/src/action_artifact.rs:47`); `Python::initialize()` in tests
//! mirrors `search_tests::single_detach_discipline`
//! (`crates/bridge/src/search.rs:1813-1814`).
//!
//! Single-cdylib tree: registers its fns/consts on the shared
//! `hydra2._native.search` submodule via `register` (mirrors
//! `action_artifact::register` on `contracts`); no new entry point. Wiring is
//! MAIN-ONLY (`search::register` calls this `register` with its `sub`).
//!
//! Tolerances (float-bit parity NOT claimed — stats floats, `math.isclose` lane):
//! - `persistence_report_block_diffs`: bit-exact (IEEE754 `x - y`, same order both sides).
//! - `persistence_report_mean`: `math.isclose(..., rel_tol=1e-12, abs_tol=1e-12)` vs
//!   `sum(vals)/len` (Neumaier vs CPython builtin-sum compensation; identical on
//!   benign vectors, at most 1ulp on adversarial cancellation).
//! - `persistence_report_strata_rates`: bit-exact (`hit/total` is a
//!   correctly-rounded division of small exact ints; empty logs give
//!   `(0.0, 0.0, 1)`, never NaN).
//! - `persistence_report_contrast_seed`: byte-exact (`sha256("persistence-{name}-v1")`).
//! - `persistence_report_resources_equal`: exact (`u64` memcmp).

use std::collections::HashMap;

use hydra_search::SearchError;
use hydra_search::eval::block_mean as block_mean_owner;
use hydra_search::persistence_kernel::ArmId;
use hydra_search::persistence_planner::CommitEntry;
use hydra_search::persistence_report as report_owner;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

/// Frozen contrast pairs (`persistence_report.py:86`): P-F, R-F, P-R, P-C.
pub const PERSISTENCE_REPORT_CONTRAST_PAIRS: [(&str, &str, &str); 4] = [
    ("P-F", "P", "F"),
    ("R-F", "R", "F"),
    ("P-R", "P", "R"),
    ("P-C", "P", "C"),
];

/// Frozen contrast unit (`FactorialContrasts.unit`, `persistence_report.py:47`).
pub const PERSISTENCE_REPORT_UNIT: &str = "wall_block";

/// Map an owner error onto the bridge failure (`ValueError`, never default).
fn search_err(err: SearchError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Per-block differences for one contrast pair (`_block_diffs`, `:81-84` via
/// `block_diffs`): `a[i] - b[i]`, lengths must match. Bit-exact.
#[pyfunction]
fn persistence_report_block_diffs(py: Python<'_>, a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    let out = py.detach(|| report_owner::block_diffs(&a, &b));
    out.map_err(search_err)
}

/// Mean of one float lane (`sum(vals)/len` via `eval::block_mean`): per-arm
/// means (`:160`) and contrast estimates share this lane. `math.isclose` vs
/// the oracle (`rel_tol=1e-12, abs_tol=1e-12`); empty/non-finite lanes raise.
#[pyfunction]
fn persistence_report_mean(py: Python<'_>, values: Vec<f64>) -> PyResult<f64> {
    let out = py.detach(|| block_mean_owner(&values));
    out.map_err(search_err)
}

/// Persistence stream seed (`:94` via `persistence_contrast_seed`):
/// `sha256("persistence-{name}-v1")` bytes feeding the numpy-lane
/// `RandomStream`. Byte-exact; empty names raise.
#[pyfunction]
fn persistence_report_contrast_seed(py: Python<'_>, name: String) -> PyResult<Vec<u8>> {
    if name.is_empty() {
        return Err(PyValueError::new_err(
            "search persistence_report_contrast_seed: name must be non-empty",
        ));
    }
    let out = py.detach(|| report_owner::persistence_contrast_seed(&name).to_vec());
    Ok(out)
}

/// Per-arm hit/miss rates over commit outcomes (`:100-127` via the
/// `stratify_surprise_miss_recovery` owner): `arm` is an opaque log key —
/// the oracle accepts any dict key, so there is no `ArmId` gate here (the
/// owner only reads `outcome`/`ponder_calls` per entry —
/// `hydra-search persistence_report.rs:128-131`); `outcomes`/`ponder_calls`
/// ride aligned 1:1 (the facade keeps counts/ponder sums). Returns
/// `(hit_rate, miss_rate, total_packets)` with `total = raw or 1` (never NaN).
/// Bit-exact.
#[pyfunction]
fn persistence_report_strata_rates(
    py: Python<'_>,
    arm: String,
    outcomes: Vec<String>,
    ponder_calls: Vec<usize>,
) -> PyResult<(f64, f64, usize)> {
    // No arm allowlist: the oracle keys strata by whatever key the caller logged.
    if outcomes.len() != ponder_calls.len() {
        return Err(PyValueError::new_err(
            "search persistence_report_strata_rates: outcomes/ponder_calls length mismatch",
        ));
    }
    let out = py.detach(|| -> Result<(f64, f64, usize), SearchError> {
        let mut entries: Vec<CommitEntry> = Vec::with_capacity(outcomes.len());
        for (idx, (outcome, ponder)) in outcomes.iter().zip(ponder_calls.iter()).enumerate() {
            entries.push(CommitEntry {
                // Placeholder: `stratify_...` never reads `.arm` (outcome/ponder only).
                arm: ArmId::B,
                packet_id: format!("{arm}:{idx}"),
                outcome: outcome.clone(),
                ponder_calls: *ponder,
            });
        }
        let mut logs: HashMap<String, Vec<CommitEntry>> = HashMap::new();
        logs.insert(arm.clone(), entries);
        let strata = report_owner::stratify_surprise_miss_recovery(&logs);
        let lane = strata.get(&arm).ok_or(SearchError::InvalidArg {
            detail: "persistence_report strata missing arm",
        })?;
        Ok((lane.hit_rate, lane.miss_rate, lane.total_packets))
    });
    out.map_err(search_err)
}

/// Resource-equality refusal (`:186-190`): true when the P and F `model_calls`
/// lanes are identical (the facade raises `ContractError`, never claiming
/// resource equality). Exact `u64` compare.
#[pyfunction]
fn persistence_report_resources_equal(
    py: Python<'_>,
    p_calls: Vec<u64>,
    f_calls: Vec<u64>,
) -> PyResult<bool> {
    Ok(py.detach(|| p_calls == f_calls))
}

/// Register the report fns/consts on the shared `search` submodule (mirrors
/// `action_artifact::register` on `contracts`): compute detached, wrap
/// attached; single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add_function(wrap_pyfunction!(persistence_report_block_diffs, sub)?)?;
    sub.add_function(wrap_pyfunction!(persistence_report_mean, sub)?)?;
    sub.add_function(wrap_pyfunction!(persistence_report_contrast_seed, sub)?)?;
    sub.add_function(wrap_pyfunction!(persistence_report_strata_rates, sub)?)?;
    sub.add_function(wrap_pyfunction!(persistence_report_resources_equal, sub)?)?;
    sub.add(
        "PERSISTENCE_REPORT_CONTRAST_PAIRS",
        PyTuple::new(py, PERSISTENCE_REPORT_CONTRAST_PAIRS)?,
    )?;
    sub.add("PERSISTENCE_REPORT_UNIT", PERSISTENCE_REPORT_UNIT)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_diffs_exact() {
        Python::initialize();
        Python::attach(|py| {
            let out = persistence_report_block_diffs(
                py,
                vec![2.0, 2.0, 2.0, 2.0],
                vec![2.5, 2.5, 2.5, 2.5],
            )
            .unwrap();
            assert_eq!(out, vec![-0.5, -0.5, -0.5, -0.5]);
            assert!(persistence_report_block_diffs(py, vec![1.0, 2.0], vec![1.0]).is_err());
        });
    }

    #[test]
    fn mean_close_to_oracle() {
        Python::initialize();
        Python::attach(|py| {
            let got = persistence_report_mean(py, vec![-0.2, -0.1]).unwrap();
            assert!((got - (-0.15000000000000002)).abs() < 1e-15);
            assert!(persistence_report_mean(py, Vec::new()).is_err());
        });
    }

    #[test]
    fn contrast_seed_golden() {
        Python::initialize();
        Python::attach(|py| {
            let seed = persistence_report_contrast_seed(py, "P-F".to_string()).unwrap();
            let hex: String = seed.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(
                hex,
                "84ffb831308eb027f903420744139de37bd8a2a5e88e41135b4bc5d7275b44f8"
            );
            assert!(persistence_report_contrast_seed(py, String::new()).is_err());
        });
    }

    #[test]
    fn strata_rates_owner_shaped() {
        Python::initialize();
        Python::attach(|py| {
            let (hit, miss, total) = persistence_report_strata_rates(
                py,
                "P".to_string(),
                vec![
                    "hit".to_string(),
                    "miss_recovery".to_string(),
                    "rebuild_no_forest".to_string(),
                ],
                vec![4, 2, 0],
            )
            .unwrap();
            assert!((hit - 1.0 / 3.0).abs() < 1e-15);
            assert!((miss - 2.0 / 3.0).abs() < 1e-15);
            assert_eq!(total, 3);
            let (hit0, miss0, total0) =
                persistence_report_strata_rates(py, "P".to_string(), Vec::new(), Vec::new())
                    .unwrap();
            assert_eq!((hit0, miss0, total0), (0.0, 0.0, 1));
            assert!(
                persistence_report_strata_rates(
                    py,
                    "P".to_string(),
                    vec!["hit".to_string()],
                    Vec::new()
                )
                .is_err()
            );
            // Opaque keys pass through like the oracle (any dict key accepted).
            let (hit_x, miss_x, total_x) = persistence_report_strata_rates(
                py,
                "X".to_string(),
                vec!["hit".to_string(), "miss_recovery".to_string()],
                vec![1, 1],
            )
            .unwrap();
            assert!((hit_x - 0.5).abs() < 1e-15);
            assert!((miss_x - 0.5).abs() < 1e-15);
            assert_eq!(total_x, 2);
        });
    }

    #[test]
    fn resources_equal_exact() {
        Python::initialize();
        Python::attach(|py| {
            assert!(!persistence_report_resources_equal(py, vec![36, 37], vec![32, 33]).unwrap());
            assert!(persistence_report_resources_equal(py, vec![32, 33], vec![32, 33]).unwrap());
        });
    }
}
