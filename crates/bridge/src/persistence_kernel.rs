//! persistence_kernel: packet-id + epoch observers over the `hydra-search` owner.
//!
//! DAG: this module depends on the search crate (`persistence_kernel`,
//! `persistence_planner`) + pyo3 ONLY — arg-check + detach + scalar return.
//! The packet math lives search-side in
//! `hydra_search::persistence_kernel::compute_packet_id` (canon-wins B3 —
//! this file owns no hash bytes) and the epoch hash in
//! `hydra_search::persistence_planner::PersistencePlanner::epoch_for_obs`
//! (the observation-unpacking + canonical fallback stays Python in
//! `persistence_planner._new_epoch_for_obs`; only the final
//! `sha256("{observation_hash}:{arm}")` crosses); this file only validates
//! arguments attached, runs the pure compute detached, and maps
//! [`SearchError`] onto `ValueError` attached. FORBIDS: packet enumeration
//! (the `search::persistence_packets_for` batch lane stays the enumerate
//! path), forest/commit orchestration (the `PersistencePlanner` arena
//! design does not cross), file/JSON IO, wall-clock, torch/CUDA, and second
//! sources for the hashes (the canon bytes + sha live owner-side, never
//! retyped here).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + `let py =
//! sub.py()` mirror `action_artifact::register`
//! (`crates/bridge/src/action_artifact.rs:42-43`); `#[pyfunction]` mirrors
//! `eval::aggregate_wall_block` (`crates/bridge/src/eval.rs:91-98`);
//! `wrap_pyfunction!(f, sub)` with a borrowed `sub` mirrors
//! `rows_seal::register` (`crates/bridge/src/rows_seal.rs:485`);
//! `py.detach` + `search_err` mirror `search::gumbel_for_action`
//! (`crates/bridge/src/search.rs:180-182,272`); bool-rejecting int reads
//! mirror `as_u64` (`crates/bridge/src/search_profiles.rs:42-47`);
//! `Python::initialize()` in tests mirrors
//! `search_tests::single_detach_discipline`
//! (`crates/bridge/src/search.rs:1813-1814`).
//!
//! Single-cdylib tree: registers its fns on the shared
//! `hydra2._native.search` submodule via `register` (mirrors
//! `action_artifact::register` on `contracts`); no new entry point. Wiring is
//! MAIN-ONLY (`search::register` calls this `register` with its `sub`).
//!
//! Tolerances (float-free lane — byte parity CLAIMED):
//! - `persistence_compute_packet_id`: byte-exact (`sha256:` over canon
//!   `{epoch_before, action_id, branch}`; the owner golden below pins it).
//! - `persistence_epoch_for_obs`: byte-exact (`epoch:` + first 16 hex of
//!   `sha256("{observation_hash}:{arm}")`; arm-discriminated, deterministic).

use hydra_search::SearchError;
use hydra_search::persistence_kernel as kernel_owner;
use hydra_search::persistence_kernel::ArmId;
use hydra_search::persistence_planner::PersistencePlanner;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyModule};

/// Map an owner error onto the bridge failure (`ValueError`, never default).
fn search_err(err: SearchError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Read a Python int that must fit `u64` and must not be a bool
/// (`search_profiles::as_u64` shape, `search_profiles.rs:42-47`).
fn as_u64(obj: &Bound<'_, PyAny>, err: &'static str) -> Result<u64, PyErr> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(err));
    }
    obj.extract::<u64>().map_err(|_| PyTypeError::new_err(err))
}

/// Packet id (`persistence_kernel.py:233-235` via `compute_packet_id`):
/// `sha256:` over canon `{epoch_before, action_id, branch}`. Byte-exact;
/// bool/non-int action ids fail closed here (the facade guards them to the
/// oracle, so `True == 1` never crosses).
#[pyfunction]
#[pyo3(signature = (epoch_before, action_id, branch))]
fn persistence_compute_packet_id(
    py: Python<'_>,
    epoch_before: String,
    action_id: Bound<'_, PyAny>,
    branch: Bound<'_, PyAny>,
) -> PyResult<String> {
    let aid = as_u64(
        &action_id,
        "search persistence_compute_packet_id: action_id must be a nonneg int",
    )?;
    let br = as_u64(
        &branch,
        "search persistence_compute_packet_id: branch must be a nonneg int",
    )?;
    py.detach(|| kernel_owner::compute_packet_id(&epoch_before, aid, br))
        .map_err(search_err)
}

/// Deterministic epoch id from an observation digest + arm
/// (`persistence_planner.py:118-132` via `epoch_for_obs`): `epoch:` +
/// first 16 hex of `sha256("{observation_hash}:{arm}")`. The caller derives
/// `observation_hash` (attr/dict/canonical fallback stays Python); only the
/// final hash crosses, through the owner planner (default budget/seed — the
/// epoch lane reads `arm.id` only, so planner seeds MUST NOT unify here).
/// Byte-exact; unknown arms fail closed.
#[pyfunction]
#[pyo3(signature = (arm_id, observation_hash))]
fn persistence_epoch_for_obs(
    py: Python<'_>,
    arm_id: String,
    observation_hash: String,
) -> PyResult<String> {
    let arm = ArmId::parse(&arm_id).map_err(search_err)?;
    py.detach(|| {
        let planner = PersistencePlanner::new(arm, None, None, None, None, None)?;
        Ok::<String, SearchError>(planner.epoch_for_obs(&observation_hash))
    })
    .map_err(search_err)
}

/// Register the kernel fns on the shared `search` submodule (mirrors
/// `action_artifact::register` on `contracts`): compute detached, wrap
/// attached; single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(persistence_compute_packet_id, sub)?)?;
    sub.add_function(wrap_pyfunction!(persistence_epoch_for_obs, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_id_golden_matches_owner() {
        Python::initialize();
        Python::attach(|py| {
            // Owner golden (`search persistence_kernel.rs:504-507`, itself
            // derived from the `persistence_kernel.py` oracle).
            let three = py.eval(c"3", None, None).unwrap();
            let zero = py.eval(c"0", None, None).unwrap();
            let got =
                persistence_compute_packet_id(py, "epoch:abc123".to_string(), three, zero).unwrap();
            assert_eq!(
                got,
                "sha256:d453adbd4607030ee80d3edd1f73a2b2306791efac74832a19fdda8a2cd9c0e8"
            );
            // Bool ids fail closed (never `True == 1`).
            let truthy = py.eval(c"True", None, None).unwrap();
            let zero_again = py.eval(c"0", None, None).unwrap();
            assert!(
                persistence_compute_packet_id(py, "e".to_string(), truthy, zero_again).is_err()
            );
            // Negative ids fail closed (u64 cannot carry them).
            let neg = py.eval(c"-1", None, None).unwrap();
            let zero_third = py.eval(c"0", None, None).unwrap();
            assert!(persistence_compute_packet_id(py, "e".to_string(), neg, zero_third).is_err());
        });
    }

    #[test]
    fn epoch_determinism_and_arm_split() {
        Python::initialize();
        Python::attach(|py| {
            let a =
                persistence_epoch_for_obs(py, "P".to_string(), "sha256:abc".to_string()).unwrap();
            let b =
                persistence_epoch_for_obs(py, "P".to_string(), "sha256:abc".to_string()).unwrap();
            assert_eq!(a, b);
            assert!(a.starts_with("epoch:"));
            assert_eq!(a.len(), 6 + 16);
            let f =
                persistence_epoch_for_obs(py, "F".to_string(), "sha256:abc".to_string()).unwrap();
            assert_ne!(a, f);
            // Unknown arms fail closed.
            assert!(
                persistence_epoch_for_obs(py, "Z".to_string(), "sha256:abc".to_string()).is_err()
            );
        });
    }
}
