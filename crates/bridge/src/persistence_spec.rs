//! persistence_spec: frozen per-arm choice scalar + deadline/fallback guard over the `hydra-search` owner.
//!
//! DAG: this module depends on the search crate (`persistence_spec`,
//! `persistence_kernel`) + pyo3 ONLY — arg-check + detach + scalar return.
//! The choice math lives search-side in
//! `hydra_search::persistence_spec::{deterministic_gumbel_for_arm,
//! validate_deadline_and_fallback}` and the arm table in
//! `hydra_search::persistence_kernel::{ArmId, make_persistence_arm}` (canon-wins
//! B3 — this file owns no hash bytes and no table rows); this file only validates
//! arguments attached, runs the pure compute detached, and maps [`SearchError`]
//! onto `ValueError` attached. FORBIDS: file/JSON IO (`_default_hashes` stays
//! Python), factory orchestration (`make_persistence_candidate_spec` stays Python —
//! live `CandidateSpec`/`ResourceBudget` dataclasses never cross), wall-clock,
//! torch/CUDA, and second sources for the seeds/margin (the seed bytes below
//! reference the owners, never retype them).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + `let py =
//! sub.py()` mirror `action_artifact::register`
//! (`crates/bridge/src/action_artifact.rs:42-43`); `#[pyfunction]` mirrors
//! `eval::aggregate_wall_block` (`crates/bridge/src/eval.rs:91-98`);
//! `wrap_pyfunction!(f, sub)` with a borrowed `sub` mirrors
//! `rows_seal::register` (`crates/bridge/src/rows_seal.rs:485`);
//! `py.detach` + `search_err` mirror `search::gumbel_for_action`
//! (`crates/bridge/src/search.rs:180-182,272`); `PyBytes::new` for the frozen
//! seeds mirrors the master-seed const
//! (`crates/bridge/src/search_shared.rs:60-64`); `Python::initialize()` in tests
//! mirrors `search_tests::single_detach_discipline`
//! (`crates/bridge/src/search.rs:1813-1814`); bool-rejecting int reads mirror
//! `as_u64` (`crates/bridge/src/search_profiles.rs:42-47`).
//!
//! Single-cdylib tree: registers its fns/consts on the shared
//! `hydra2._native.search` submodule via `register` (mirrors
//! `action_artifact::register` on `contracts`); no new entry point. Wiring is
//! MAIN-ONLY (`search::register` calls this `register` with its `sub`).
//!
//! Tolerances (float-bit parity CLAIMED — integer-sha lane):
//! - `persistence_spec_gumbel_for_arm`: bit-exact (`u64BE(sha256(seed +
//!   ":arm:case:action")[:8]) as f64 / 2^64`; `u64 -> f64` rounds half-even and
//!   the division is correctly rounded on both sides; the owner goldens below
//!   pin the exact bits).

use hydra_search::SearchError;
use hydra_search::persistence_kernel::{ArmId, make_persistence_arm};
use hydra_search::persistence_spec as spec_owner;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyModule};

/// Default fallback margin ms (`persistence_spec.py:100`, `persistence_planner.py:154`).
pub const PERSISTENCE_SPEC_DEFAULT_FALLBACK_MARGIN_MS: u64 = 500;

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

/// Read a Python int that must fit `i64` and must not be a bool (same shape as
/// [`as_u64`): negativity is reported by the caller as `ValueError` with the
/// owner detail, never silently wrapped.
fn as_i64(obj: &Bound<'_, PyAny>, err: &'static str) -> Result<i64, PyErr> {
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(err));
    }
    obj.extract::<i64>().map_err(|_| PyTypeError::new_err(err))
}

/// Deterministic scalar in `[0,1)` from `(arm, case, action)`
/// (`persistence_spec.py:185-193` via `deterministic_gumbel_for_arm`):
/// `u64BE(sha256(seed + ":arm:case:action")[:8]) / 2^64` — no RNG call-order
/// dependence. `seed` defaults to the frozen
/// `GUMBEL_SEED_DEFAULT` (`b"hydra2-persistence-v1"`); pass the planner seed
/// explicitly for the planner lane (the two MUST NOT unify). Bit-exact.
#[pyfunction]
#[pyo3(signature = (arm_id, case_id, action_id, seed=None))]
fn persistence_spec_gumbel_for_arm(
    py: Python<'_>,
    arm_id: String,
    case_id: String,
    action_id: Bound<'_, PyAny>,
    seed: Option<Vec<u8>>,
) -> PyResult<f64> {
    let aid = as_u64(
        &action_id,
        "search persistence_spec_gumbel_for_arm: action_id must be a nonneg int",
    )?;
    let arm = ArmId::parse(&arm_id).map_err(search_err)?;
    let seed_owned: Vec<u8> = seed.unwrap_or_else(|| spec_owner::GUMBEL_SEED_DEFAULT.to_vec());
    Ok(py.detach(|| spec_owner::deterministic_gumbel_for_arm(arm, &case_id, aid, &seed_owned)))
}

/// Validate deadline + fallback margin (`persistence_spec.py:164-177` via
/// `validate_deadline_and_fallback`): margin in `[0, deadline)`, deployable
/// arms (`B`/`F`/`R`/`P`) pin `deadline <= 5000`, `C` keeps its positive extra
/// allowance. The arm row comes from the frozen table ([`make_persistence_arm`],
/// single source — validated `PersistenceArm` objects pin `deployable`/allowance
/// to the same table, so no live dataclass crosses). Negativity fails here as
/// `ValueError` with the owner detail (`u64` cannot carry it to the owner).
#[pyfunction]
#[pyo3(signature = (arm_id, deadline_ms, fallback_margin_ms))]
fn persistence_spec_validate_deadline_and_fallback(
    py: Python<'_>,
    arm_id: String,
    deadline_ms: Bound<'_, PyAny>,
    fallback_margin_ms: Bound<'_, PyAny>,
) -> PyResult<()> {
    let arm = ArmId::parse(&arm_id).map_err(search_err)?;
    let deadline = as_i64(
        &deadline_ms,
        "search persistence_spec_validate_deadline_and_fallback: deadline_ms must be an int",
    )?;
    let margin = as_i64(
        &fallback_margin_ms,
        "search persistence_spec_validate_deadline_and_fallback: fallback_margin_ms must be an int",
    )?;
    if deadline < 0 || margin < 0 {
        return Err(PyValueError::new_err(
            "invalid search argument: fallback_margin must be in [0, deadline)",
        ));
    }
    // proof: both >= 0 (checked above); `u64::try_from` fails closed on huge.
    let deadline_u: u64 = u64::try_from(deadline)
        .map_err(|_| PyValueError::new_err("invalid search argument: deadline must fit u64"))?;
    let margin_u: u64 = u64::try_from(margin)
        .map_err(|_| PyValueError::new_err("invalid search argument: margin must fit u64"))?;
    py.detach(move || {
        let arm_state = make_persistence_arm(arm);
        spec_owner::validate_deadline_and_fallback(&arm_state, deadline_u, margin_u)
    })
    .map_err(search_err)?;
    Ok(())
}

/// Register the spec fns/consts on the shared `search` submodule (mirrors
/// `action_artifact::register` on `contracts`): compute detached, wrap
/// attached; single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add_function(wrap_pyfunction!(persistence_spec_gumbel_for_arm, sub)?)?;
    sub.add_function(wrap_pyfunction!(
        persistence_spec_validate_deadline_and_fallback,
        sub
    )?)?;
    sub.add(
        "PERSISTENCE_SPEC_GUMBEL_SEED_DEFAULT",
        PyBytes::new(py, spec_owner::GUMBEL_SEED_DEFAULT),
    )?;
    sub.add(
        "PERSISTENCE_SPEC_GUMBEL_SEED_PLANNER",
        PyBytes::new(py, spec_owner::GUMBEL_SEED_PLANNER),
    )?;
    sub.add(
        "PERSISTENCE_SPEC_DEFAULT_FALLBACK_MARGIN_MS",
        PERSISTENCE_SPEC_DEFAULT_FALLBACK_MARGIN_MS,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_consts_match_oracle() {
        assert_eq!(spec_owner::GUMBEL_SEED_DEFAULT, b"hydra2-persistence-v1");
        assert_eq!(spec_owner::GUMBEL_SEED_PLANNER, b"persistence-factorial-v1");
        assert_eq!(PERSISTENCE_SPEC_DEFAULT_FALLBACK_MARGIN_MS, 500);
    }

    #[test]
    fn gumbel_goldens_match_owner() {
        Python::initialize();
        Python::attach(|py| {
            // Hand-derived from the owner goldens (`search persistence_spec.rs:189-204`,
            // themselves derived from the `persistence_spec.py:185-193` oracle).
            let three = py.eval(c"3", None, None).unwrap();
            let got = persistence_spec_gumbel_for_arm(
                py,
                "F".to_string(),
                "case-001".to_string(),
                three,
                None,
            )
            .unwrap();
            assert_eq!(got, 0.6055335298833656);
            // Explicit default seed repeats the default lane exactly.
            let three_again = py.eval(c"3", None, None).unwrap();
            let got_again = persistence_spec_gumbel_for_arm(
                py,
                "F".to_string(),
                "case-001".to_string(),
                three_again,
                Some(spec_owner::GUMBEL_SEED_DEFAULT.to_vec()),
            )
            .unwrap();
            assert_eq!(got_again, got);
            // Planner seed MUST NOT unify with the default lane.
            let three_planner = py.eval(c"3", None, None).unwrap();
            let got_planner = persistence_spec_gumbel_for_arm(
                py,
                "F".to_string(),
                "case-001".to_string(),
                three_planner,
                Some(spec_owner::GUMBEL_SEED_PLANNER.to_vec()),
            )
            .unwrap();
            assert_ne!(got_planner, got);
            // Bool action ids fail closed (never `True == 1`).
            let truthy = py.eval(c"True", None, None).unwrap();
            assert!(
                persistence_spec_gumbel_for_arm(py, "F".to_string(), "c".to_string(), truthy, None)
                    .is_err()
            );
            // Unknown arms fail closed.
            let zero = py.eval(c"0", None, None).unwrap();
            assert!(
                persistence_spec_gumbel_for_arm(py, "Z".to_string(), "c".to_string(), zero, None)
                    .is_err()
            );
        });
    }

    #[test]
    fn validate_mirrors_oracle_guards() {
        Python::initialize();
        Python::attach(|py| {
            // Hand-derived from `persistence_spec.py:164-177` via the WP-09C
            // expectations (`test_persistence_factorial_wp09c.py:405-424`).
            let deadline = py.eval(c"5000", None, None).unwrap();
            let margin = py.eval(c"500", None, None).unwrap();
            assert!(
                persistence_spec_validate_deadline_and_fallback(
                    py,
                    "B".to_string(),
                    deadline,
                    margin
                )
                .is_ok()
            );
            // Margin at the deadline rejects.
            let deadline_full = py.eval(c"5000", None, None).unwrap();
            let margin_full = py.eval(c"5000", None, None).unwrap();
            assert!(
                persistence_spec_validate_deadline_and_fallback(
                    py,
                    "B".to_string(),
                    deadline_full,
                    margin_full
                )
                .is_err()
            );
            // Deployable arm over 5000 rejects.
            let deadline_over = py.eval(c"6000", None, None).unwrap();
            let margin_ok = py.eval(c"500", None, None).unwrap();
            assert!(
                persistence_spec_validate_deadline_and_fallback(
                    py,
                    "B".to_string(),
                    deadline_over,
                    margin_ok
                )
                .is_err()
            );
            // C laboratory passes the deployable-shaped budget.
            let deadline_c = py.eval(c"5000", None, None).unwrap();
            let margin_c = py.eval(c"500", None, None).unwrap();
            assert!(
                persistence_spec_validate_deadline_and_fallback(
                    py,
                    "C".to_string(),
                    deadline_c,
                    margin_c
                )
                .is_ok()
            );
            // Negative margins fail closed as ValueError, never wrap.
            let deadline_neg = py.eval(c"5000", None, None).unwrap();
            let margin_neg = py.eval(c"-1", None, None).unwrap();
            assert!(
                persistence_spec_validate_deadline_and_fallback(
                    py,
                    "C".to_string(),
                    deadline_neg,
                    margin_neg
                )
                .is_err()
            );
            // Bool margins fail closed (never `True == 1`).
            let deadline_bool = py.eval(c"5000", None, None).unwrap();
            let margin_bool = py.eval(c"True", None, None).unwrap();
            assert!(
                persistence_spec_validate_deadline_and_fallback(
                    py,
                    "B".to_string(),
                    deadline_bool,
                    margin_bool
                )
                .is_err()
            );
        });
    }
}
