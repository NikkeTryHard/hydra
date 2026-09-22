//! search_profiles: frozen candidate-profile rows + cost-gated admission math.
//!
//! DAG: pyo3 ONLY (frozen SPEC 16.7 PR3 priors + pure int/float accounting;
//! no feed/search-crate edge — profile rows are labeled priors, not digests
//! or arena state; the RTX pilot fixture promotes them, never this file).
//! FORBIDS: planner/adapter state (`act()` owners stay `search.rs`),
//! wall-clock budgets (monotonic deadlines cannot cross `detach` without a
//! clock policy), torch/CUDA, IO, digest/hash math (canon owns bytes), and
//! live `CandidateProfile` objects (callers pass validated scalar fields
//! across; the dataclass stays Python in `search/profiles.py`).
//!
//! Single-cdylib tree: registers its const + pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `search::register` in `search.rs:1690-1731`); no new entry point.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyFloat, PyTuple};

/// One frozen profile row (`profiles.py:89-98`): `(name, candidate_cap,
/// horizon, carry_quota, halving_rounds)` in dataclass field order
/// (`profiles.py:44-48`).
struct FrozenRow {
    name: &'static str,
    cap: u64,
    horizon: u64,
    carry: u64,
    rounds: u64,
}

/// Frozen SPEC 16.7 PR3 priors (`profiles.py:89-98`): labeled, pilot-promoted
/// only — never read as measured capacities. Single source for
/// `PROFILES_ROWS`; Python rebuilds its `CandidateProfile` tuple from it.
const FROZEN_ROWS: [FrozenRow; 3] = [
    FrozenRow {
        name: "small",
        cap: 16,
        horizon: 2,
        carry: 128,
        rounds: 4,
    },
    FrozenRow {
        name: "medium",
        cap: 32,
        horizon: 4,
        carry: 256,
        rounds: 5,
    },
    FrozenRow {
        name: "large",
        cap: 64,
        horizon: 4,
        carry: 512,
        rounds: 6,
    },
];

/// Read a Python int that must not be a bool (`contracts.rs:69-74`
/// `plain_int_value` shape; `canon_rng.rs:279` `is_instance_of::<PyBool>`).
fn as_u64(obj: &Bound<'_, PyAny>, err: &str) -> Result<u64, String> {
    if obj.is_instance_of::<PyBool>() {
        return Err(err.to_string());
    }
    obj.extract::<u64>().map_err(|_| err.to_string())
}

/// Rounds lane for the equality arm: non-int shapes collapse to a sentinel
/// no real schedule can equal (`ilog2` peaks at 63), so the
/// `halving_rounds != log2(M)` reject fires with the exact
/// `profiles.py:58-60` text instead of a novel message.
fn as_rounds_or_sentinel(obj: &Bound<'_, PyAny>) -> u64 {
    if obj.is_instance_of::<PyBool>() {
        return u64::MAX;
    }
    obj.extract::<u64>().unwrap_or(u64::MAX)
}

/// Range half of `CandidateProfile.__post_init__` (`profiles.py:54-60`):
/// `M >= 2`, power-of-two halving schedule, `rounds == log2(M)`. Pure and
/// detach-safe.
fn check_cap_rounds(cap: u64, rounds: u64) -> Result<(), String> {
    if cap < 2 {
        return Err("candidate_cap must be an int >= 2".to_string());
    }
    if !cap.is_power_of_two() {
        return Err("candidate_cap must be a power of two (halving schedule)".to_string());
    }
    let expect = u64::from(cap.ilog2());
    if rounds != expect {
        return Err(format!(
            "halving_rounds must equal log2(M) = {expect} for M = {cap}"
        ));
    }
    Ok(())
}

/// Full field validator mirroring `CandidateProfile.__post_init__`
/// (`profiles.py:50-72`), check order verbatim. Pure and detach-safe.
fn check_fields(name: &str, cap: u64, horizon: u64, carry: u64, rounds: u64) -> Result<(), String> {
    if name.is_empty() {
        return Err("profile name must be a non-empty str".to_string());
    }
    check_cap_rounds(cap, rounds)?;
    if !(1..=16).contains(&horizon) {
        return Err("horizon must be an int in 1..16".to_string());
    }
    if carry == 0 {
        return Err("carry_quota must be a positive int".to_string());
    }
    Ok(())
}

/// Exact added rollout jobs: `4M.log2(M)` (`profiles.py:101-108`). Pure and
/// detach-safe.
fn jobs_of(cap: u64, rounds: u64) -> u64 {
    4 * cap * rounds
}

/// Exact transitions bound: `jobs x H` (`profiles.py:111-120`). Pure and
/// detach-safe.
fn bound_of(cap: u64, rounds: u64, horizon: u64) -> u64 {
    jobs_of(cap, rounds) * horizon
}

/// Cost-gated admission scan (`profiles.py:163-177`): largest-M first
/// (stable order, matching `sorted(..., reverse=True)`), first row whose
/// exact bound fits the deadline margin (and the optional transition cap)
/// wins; the returned index is into the caller's order, `None` reads as
/// Candidate 0. Rows arrive validated (`CandidateProfile` construction); the
/// pyfn re-checks ranges so direct bridge calls fail closed. Pure and detach-safe.
fn admit_pick(
    caps: &[u64],
    horizons: &[u64],
    rounds: &[u64],
    deadline_ms: u64,
    margin_ms: u64,
    seconds_per_transition: f64,
    max_transitions: Option<u64>,
) -> Option<usize> {
    // proof: ms budgets are u64 (< 2^53), exact; budget seconds tolerate 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let budget_s: f64 = (deadline_ms - margin_ms) as f64 / 1000.0;
    let mut order: Vec<usize> = (0..caps.len()).collect();
    order.sort_by(|&a, &b| caps[b].cmp(&caps[a]));
    for idx in order {
        let bound = bound_of(caps[idx], rounds[idx], horizons[idx]);
        // proof: transition bound is u64 (< 2^53), exact; budget compare tolerates 1ulp.
        #[allow(clippy::cast_precision_loss)]
        let bound_f: f64 = bound as f64;
        if bound_f * seconds_per_transition > budget_s {
            continue;
        }
        if max_transitions.is_some_and(|cap| bound > cap) {
            continue;
        }
        return Some(idx);
    }
    None
}

/// Accounting loop (`profiles.py:206-211`): `jobs += survivors * visits`
/// with ceil halving; cheaper arm by strict comparison, else tie. Pure and
/// detach-safe.
fn compare_counts(n_actions: u64, visits: &[u64], sims: u64) -> (u64, u64, &'static str) {
    let mut survivors = n_actions;
    let mut gumbel_jobs = 0u64;
    for &v in visits {
        gumbel_jobs += survivors * v;
        survivors = survivors.div_ceil(2);
    }
    let cheaper = if gumbel_jobs < sims {
        "gumbel"
    } else if sims < gumbel_jobs {
        "puct"
    } else {
        "tie"
    };
    (gumbel_jobs, sims, cheaper)
}

/// Field validator mirroring `CandidateProfile.__post_init__`
/// (`profiles.py:50-72`): attached checks only (like
/// `is_validation_error_class`, `validate.rs:99-102`); `ValueError` on any
/// reject, never a default.
#[pyfunction]
#[pyo3(signature = (name, candidate_cap, horizon, carry_quota, halving_rounds))]
fn profiles_check_profile(
    name: String,
    candidate_cap: Bound<'_, PyAny>,
    horizon: Bound<'_, PyAny>,
    carry_quota: Bound<'_, PyAny>,
    halving_rounds: Bound<'_, PyAny>,
) -> PyResult<()> {
    let cap = as_u64(&candidate_cap, "candidate_cap must be an int >= 2")
        .map_err(PyValueError::new_err)?;
    let horizon =
        as_u64(&horizon, "horizon must be an int in 1..16").map_err(PyValueError::new_err)?;
    let carry = as_u64(&carry_quota, "carry_quota must be a positive int")
        .map_err(PyValueError::new_err)?;
    let rounds = as_rounds_or_sentinel(&halving_rounds);
    check_fields(&name, cap, horizon, carry, rounds).map_err(PyValueError::new_err)
}

/// Exact added rollout jobs over validated scalar fields
/// (`profiles.py:101-108`): attached validation, detached compute (one
/// `py.detach`, zero Python API inside — `validate.rs:83` shape);
/// `ValueError` on any reject, never a default.
#[pyfunction]
#[pyo3(signature = (candidate_cap, halving_rounds))]
fn profiles_jobs_for(
    py: Python<'_>,
    candidate_cap: Bound<'_, PyAny>,
    halving_rounds: Bound<'_, PyAny>,
) -> PyResult<u64> {
    let cap = as_u64(&candidate_cap, "candidate_cap must be an int >= 2")
        .map_err(PyValueError::new_err)?;
    let rounds = as_rounds_or_sentinel(&halving_rounds);
    check_cap_rounds(cap, rounds).map_err(PyValueError::new_err)?;
    Ok(py.detach(|| jobs_of(cap, rounds)))
}

/// Exact transitions bound over validated scalar fields
/// (`profiles.py:111-120`): attached validation, detached compute;
/// `ValueError` on any reject, never a default.
#[pyfunction]
#[pyo3(signature = (candidate_cap, halving_rounds, horizon))]
fn profiles_transitions_bound(
    py: Python<'_>,
    candidate_cap: Bound<'_, PyAny>,
    halving_rounds: Bound<'_, PyAny>,
    horizon: Bound<'_, PyAny>,
) -> PyResult<u64> {
    let cap = as_u64(&candidate_cap, "candidate_cap must be an int >= 2")
        .map_err(PyValueError::new_err)?;
    let rounds = as_rounds_or_sentinel(&halving_rounds);
    let horizon =
        as_u64(&horizon, "horizon must be an int in 1..16").map_err(PyValueError::new_err)?;
    check_cap_rounds(cap, rounds).map_err(PyValueError::new_err)?;
    if !(1..=16).contains(&horizon) {
        return Err(PyValueError::new_err("horizon must be an int in 1..16"));
    }
    Ok(py.detach(|| bound_of(cap, rounds, horizon)))
}

/// Cost-gated admission over parallel validated rows
/// (`profiles.py:123-177`): row lanes ride aligned 1:1 in caller order as
/// `Vec<Bound<PyAny>>` (`canon_rng.rs:333` shape) so every element is
/// bool-rejected and range-checked here — direct bridge calls fail closed
/// exactly like constructed `CandidateProfile` fields (container shape is
/// facade-checked; the bridge never sees live `CandidateProfile` objects).
/// Returns the winner's index into the caller's order, `None` for
/// Candidate 0. Attached validation, detached scan; `ValueError` on any
/// reject, never a default.
#[pyfunction]
#[pyo3(signature = (
    caps, horizons, rounds, deadline_ms, fallback_margin_ms,
    seconds_per_transition, max_transitions
))]
#[allow(clippy::too_many_arguments)]
fn profiles_admit(
    py: Python<'_>,
    caps: Vec<Bound<'_, PyAny>>,
    horizons: Vec<Bound<'_, PyAny>>,
    rounds: Vec<Bound<'_, PyAny>>,
    deadline_ms: Bound<'_, PyAny>,
    fallback_margin_ms: Bound<'_, PyAny>,
    seconds_per_transition: Bound<'_, PyAny>,
    max_transitions: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<usize>> {
    if caps.is_empty() || caps.len() != horizons.len() || caps.len() != rounds.len() {
        return Err(PyValueError::new_err(
            "search profiles rows must be a non-empty aligned 1:1 set",
        ));
    }
    let caps_v: Vec<u64> = caps
        .iter()
        .map(|obj| as_u64(obj, "candidate_cap must be an int >= 2"))
        .collect::<Result<Vec<u64>, String>>()
        .map_err(PyValueError::new_err)?;
    let horizons_v: Vec<u64> = horizons
        .iter()
        .map(|obj| as_u64(obj, "horizon must be an int in 1..16"))
        .collect::<Result<Vec<u64>, String>>()
        .map_err(PyValueError::new_err)?;
    let rounds_v: Vec<u64> = rounds.iter().map(as_rounds_or_sentinel).collect();
    for ((&cap, &horizon), &round) in caps_v.iter().zip(&horizons_v).zip(&rounds_v) {
        check_cap_rounds(cap, round).map_err(PyValueError::new_err)?;
        if !(1..=16).contains(&horizon) {
            return Err(PyValueError::new_err("horizon must be an int in 1..16"));
        }
    }
    let deadline = as_u64(&deadline_ms, "deadline_ms must be a positive int")
        .map_err(PyValueError::new_err)?;
    if deadline == 0 {
        return Err(PyValueError::new_err("deadline_ms must be a positive int"));
    }
    let margin = as_u64(
        &fallback_margin_ms,
        "fallback_margin_ms must satisfy 0 <= margin < deadline",
    )
    .map_err(PyValueError::new_err)?;
    if margin >= deadline {
        return Err(PyValueError::new_err(
            "fallback_margin_ms must satisfy 0 <= margin < deadline",
        ));
    }
    if !seconds_per_transition.is_instance_of::<PyFloat>() {
        return Err(PyValueError::new_err(
            "seconds_per_transition must be a positive finite float",
        ));
    }
    let spt: f64 = seconds_per_transition.extract().map_err(|_| {
        PyValueError::new_err("seconds_per_transition must be a positive finite float")
    })?;
    if !spt.is_finite() || spt <= 0.0 {
        return Err(PyValueError::new_err(
            "seconds_per_transition must be a positive finite float",
        ));
    }
    let max_t: Option<u64> = match max_transitions {
        None => None,
        Some(obj) => {
            let value = as_u64(&obj, "max_transitions must be a positive int or None")
                .map_err(PyValueError::new_err)?;
            if value == 0 {
                return Err(PyValueError::new_err(
                    "max_transitions must be a positive int or None",
                ));
            }
            Some(value)
        }
    };
    Ok(py.detach(|| {
        admit_pick(
            &caps_v,
            &horizons_v,
            &rounds_v,
            deadline,
            margin,
            spt,
            max_t,
        )
    }))
}

/// Accounting-level Gumbel-vs-PUCT comparison (`profiles.py:180-217`):
/// `(gumbel_jobs, puct_jobs, cheaper)`; the facade echoes `n_actions` and
/// rebuilds the dict in oracle key order. Attached validation, detached
/// loop; `ValueError` on any reject, never a default.
#[pyfunction]
#[pyo3(signature = (n_actions, gumbel_visits, puct_simulations))]
fn profiles_compare_gumbel_puct(
    py: Python<'_>,
    n_actions: Bound<'_, PyAny>,
    gumbel_visits: Vec<Bound<'_, PyAny>>,
    puct_simulations: Bound<'_, PyAny>,
) -> PyResult<(u64, u64, String)> {
    let n =
        as_u64(&n_actions, "n_actions must be a positive int").map_err(PyValueError::new_err)?;
    if n == 0 {
        return Err(PyValueError::new_err("n_actions must be a positive int"));
    }
    if gumbel_visits.is_empty() {
        return Err(PyValueError::new_err(
            "gumbel_visits must be a non-empty tuple of positive ints",
        ));
    }
    let mut visits: Vec<u64> = Vec::with_capacity(gumbel_visits.len());
    for item in &gumbel_visits {
        let value = as_u64(
            item,
            "gumbel_visits must be a non-empty tuple of positive ints",
        )
        .map_err(PyValueError::new_err)?;
        if value == 0 {
            return Err(PyValueError::new_err(
                "gumbel_visits must be a non-empty tuple of positive ints",
            ));
        }
        visits.push(value);
    }
    let sims = as_u64(&puct_simulations, "puct_simulations must be a positive int")
        .map_err(PyValueError::new_err)?;
    if sims == 0 {
        return Err(PyValueError::new_err(
            "puct_simulations must be a positive int",
        ));
    }
    Ok(py.detach(|| {
        let (gumbel_jobs, puct_jobs, cheaper) = compare_counts(n, &visits, sims);
        (gumbel_jobs, puct_jobs, cheaper.to_string())
    }))
}

/// Register the profile const + kernels on the EXISTING `search` submodule
/// (mirrors `action_artifact::register`, `action_artifact.rs:42-48`, on the
/// shared submodule; MAIN calls this from `search::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    let rows: Vec<(&str, u64, u64, u64, u64)> = FROZEN_ROWS
        .iter()
        .map(|row| (row.name, row.cap, row.horizon, row.carry, row.rounds))
        .collect();
    sub.add("PROFILES_ROWS", PyTuple::new(py, rows)?)?;
    sub.add_function(wrap_pyfunction!(profiles_check_profile, sub)?)?;
    sub.add_function(wrap_pyfunction!(profiles_jobs_for, sub)?)?;
    sub.add_function(wrap_pyfunction!(profiles_transitions_bound, sub)?)?;
    sub.add_function(wrap_pyfunction!(profiles_admit, sub)?)?;
    sub.add_function(wrap_pyfunction!(profiles_compare_gumbel_puct, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod profiles_tests {
    use super::*;

    #[test]
    fn frozen_rows_match_spec_table() {
        let names: Vec<&str> = FROZEN_ROWS.iter().map(|row| row.name).collect();
        assert_eq!(names, vec!["small", "medium", "large"]);
        let quads: Vec<(u64, u64, u64, u64)> = FROZEN_ROWS
            .iter()
            .map(|row| (row.cap, row.horizon, row.carry, row.rounds))
            .collect();
        assert_eq!(
            quads,
            vec![(16, 2, 128, 4), (32, 4, 256, 5), (64, 4, 512, 6)]
        );
    }

    #[test]
    fn jobs_formula_matches_oracle() {
        let got: Vec<(&str, u64, u64)> = FROZEN_ROWS
            .iter()
            .map(|row| {
                let jobs = jobs_of(row.cap, row.rounds);
                (row.name, jobs, bound_of(row.cap, row.rounds, row.horizon))
            })
            .collect();
        assert_eq!(
            got,
            vec![
                ("small", 256, 512),
                ("medium", 640, 2560),
                ("large", 1536, 6144)
            ]
        );
    }

    #[test]
    fn field_guards_match_post_init() {
        assert!(check_fields("small", 16, 2, 128, 4).is_ok());
        assert!(check_fields("", 16, 2, 128, 4).is_err());
        assert!(check_fields("bad", 1, 2, 1, 0).is_err());
        assert!(check_fields("bad", 20, 2, 1, 4).is_err());
        assert!(check_fields("bad", 16, 2, 1, 3).is_err());
        assert!(check_fields("bad", 16, 0, 1, 4).is_err());
        assert!(check_fields("bad", 16, 17, 1, 4).is_err());
        assert!(check_fields("bad", 16, 2, 0, 4).is_err());
        assert!(check_fields("bad", 16, 2, 1, u64::MAX).is_err());
    }

    #[test]
    fn admit_scan_matches_oracle() {
        let caps: Vec<u64> = FROZEN_ROWS.iter().map(|row| row.cap).collect();
        let horizons: Vec<u64> = FROZEN_ROWS.iter().map(|row| row.horizon).collect();
        let rounds: Vec<u64> = FROZEN_ROWS.iter().map(|row| row.rounds).collect();
        assert_eq!(
            admit_pick(&caps, &horizons, &rounds, 5000, 200, 0.0001, None),
            Some(2)
        );
        assert_eq!(
            admit_pick(&caps, &horizons, &rounds, 5000, 200, 100.0, None),
            None
        );
        assert_eq!(
            admit_pick(&caps, &horizons, &rounds, 5000, 200, 0.0001, Some(600)),
            Some(0)
        );
    }

    #[test]
    fn admit_scan_prefers_largest_m_on_unsorted_input() {
        let caps = vec![64u64, 16, 32];
        let horizons = vec![4u64, 2, 4];
        let rounds = vec![6u64, 4, 5];
        assert_eq!(
            admit_pick(&caps, &horizons, &rounds, 5000, 200, 0.0001, None),
            Some(0)
        );
    }

    #[test]
    fn compare_loop_matches_oracle() {
        assert_eq!(compare_counts(16, &[8, 8], 64), (192, 64, "puct"));
        assert_eq!(compare_counts(4, &[2], 8), (8, 8, "tie"));
        assert_eq!(compare_counts(4, &[2], 4), (8, 4, "puct"));
    }
}
