//! eval_stats: frozen SPEC 18.3 pure stat leaves over `hydra-search::eval::statistics`.
//!
//! DAG: pyo3 + `hydra-search` (`eval::statistics`) ONLY (both already bridge
//! dependencies, see `crates/bridge/Cargo.toml:19-25`; no new dependency).
//! FORBIDS: numpy/PCG64 draws (the whole-block bootstrap, sign-flip, and
//! cluster resample lanes stay Python in `python/hydra2/eval/statistics.py`
//! — M2 measure-first, draws never cross); WallBlock/telemetry dataclass
//! orchestration (`placement_block_contrast`, `score_selection`,
//! `SelectionConfig` stay Python — attached-construction over live objects);
//! hedged-CS capital math (already bridged via `eval::hedged_cs_path`,
//! `crates/bridge/src/eval.rs:107-120` — never a second owner); hash/digest
//! math (feed-owned); wall-clock; IO.
//!
//! TABLE ported here (frozen values + pure compute, oracle
//! `python/hydra2/eval/statistics.py` via `hydra_search::eval::statistics`,
//! `crates/search/src/eval/statistics.rs`):
//! - `eval_stats_fixed_n_samples` <- `fixed_n_samples` (`:115-124`) via the
//!   owner: `N = ceil(((z_(1-a) + z_(1-b)) * s / d)^2)`.
//! - `eval_stats_percentile_indexes` <- `_bootstrap_percentile_indexes`
//!   (`:127-138`) via `percentile_indexes`: `low = floor(a/2*R)`,
//!   `high = ceil((1-a/2)*R)-1` (T3 pins `(50, 1949)` at `R=2000, a=0.05`).
//! - `eval_stats_ci_covers` <- `ci_covers` (`:223-225`) via the owner.
//! - `eval_stats_sequential_design_guard` <- `sequential_design_guard`
//!   (`:304-323`) via the owner.
//!
//! LOGIC staying Python (`eval/statistics.py`): the numpy PCG64 draw lanes
//! (`bootstrap_blocks`, `sign_flip_interval`, `cluster_bootstrap` — draws +
//! assembly stay together until a bit-parity KAT lands a draw-free assembly
//! leaf); `placement_block_contrast` (live `WallBlock` objects);
//! `hedged_cs_path` / `hedged_confidence_sequence` (already on
//! `hydra2._native.eval`); `SelectionConfig` + `selection_gate_check` +
//! `score_selection` (frozen dataclass + telemetry orchestration); every
//! `ContractError` translator (the bridge raises `ValueError` with
//! byte-identical text; `statistics.py` maps to `ContractError`).
//!
//! Tolerances (float-bit parity NOT claimed beyond the noted arms — stats
//! floats, `math.isclose` lane per the wave-3 `persistence_report` precedent):
//! - `eval_stats_fixed_n_samples`: int-exact on the recorded goldens
//!   (`s=1.0/delta=0.5/a=0.05/b=0.20` -> `25`; `s=2.3/delta=0.7/a=b=0.10`
//!   -> `71`, the oracle `math.ceil` of `70.9237...`). The owner integrates
//!   Acklam's `inv_cdf` (~1e-9 class) where the oracle uses
//!   `NormalDist.inv_cdf`; the `ceil` margin (`0.64` on the canonical
//!   `SelectionConfig`, owner-documented) dwarfs the approximation error, so
//!   no realistic input flips the `ceil`.
//! - `eval_stats_percentile_indexes`: bit-exact (`floor`/`ceil` of the
//!   identically-associated `f64` products `(a/2)*R`, `(1-a/2)*R`).
//! - `eval_stats_ci_covers`: bit-exact (IEEE754 `<=`, same order both sides).
//! - `eval_stats_sequential_design_guard`: exact (`str`/`int` compares; the
//!   unknown-design arm renders Python-repr single-quoted, exact for the
//!   closed-domain ids — same note as `eval_schedule::wall_repr`).
//!
//! Oracle-divergence notes (deliberate, garbage-input only):
//! - Non-`f64` scalars (`str` s/delta/alpha/beta, `str` bounds/truth) fail
//!   typed extraction (`TypeError`, mapped to `ContractError` by the facade)
//!   where the oracle raises `TypeError` (`math.isfinite`) or compares
//!   through — both fail closed; valid `f64` callers (`int` and `bool`-as-1.0
//!   included, matching the oracle's `isfinite` acceptance) are unaffected.
//! - `str`/`float` resamples render the oracle sentence here (`Bound`
//!   staging) where the oracle raises the same `ContractError` — identical.
//!   Absurd-huge ints (`> i64::MAX`) fail closed here where the oracle would
//!   accept-then-OOM downstream in numpy; realistic callers (`100..1e6`) are
//!   unaffected.
//! - Non-`str` designs fail typed extraction (`TypeError`, mapped by the
//!   facade) where the oracle raises `ContractError(unknown design ...)` —
//!   both fail closed.
//! - `float` peek elements fail typed extraction here where the oracle's
//!   `sorted(set(...))` comparison would accept integral floats; `str` peeks
//!   fail here where the oracle `fixed_n` arm (len-only) would accept.
//!   Declared peeks are closed-domain positive ints; well-formed callers are
//!   unaffected.
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` +
//! `wrap_pyfunction!(f, sub)` with a borrowed `sub` mirror
//! `persistence_report::register`
//! (`crates/bridge/src/persistence_report.rs:163-176`); `#[pyfunction]`
//! mirrors `eval::aggregate_wall_block`
//! (`crates/bridge/src/eval.rs:91-98`); attached-extract + one
//! `py.detach(|| ...)` with zero Python API inside mirrors the
//! `eval_schedule` leaves (`crates/bridge/src/eval_schedule.rs:520-538`);
//! `SearchError` mapped onto `ValueError` mirrors `eval::search_err`
//! (`crates/bridge/src/eval.rs:60-63`), except `InvalidArg` details render
//! bare (no `invalid search argument: ` prefix) plus the oracle-sentence map
//! below so validator texts stay byte-identical to the oracle, mirroring
//! `eval_schedule::search_detail`
//! (`crates/bridge/src/eval_schedule.rs:222-227`);
//! `is_instance_of::<PyBool>` bool-rejection mirrors
//! `contracts::plain_int_value` (`crates/bridge/src/contracts.rs:69-74`);
//! `Bound<'_, PyAny>` int staging mirrors `qual_budget::as_u64`
//! (`crates/bridge/src/qual_budget.rs:230-235`); `Python::initialize()` +
//! `Python::attach` in tests mirror `persistence_report::tests`
//! (`crates/bridge/src/persistence_report.rs:179-185`).
//!
//! Feed owners (never reimplemented here): fixed-N math, percentile
//! assembly, cover predicate, and peek-discipline branching via
//! `hydra_search::eval::statistics::{fixed_n_samples, percentile_indexes,
//! ci_covers, sequential_design_guard}`
//! (`crates/search/src/eval/statistics.rs:167-190,257-260,388-423`); the
//! Acklam `inv_normal_cdf` stays behind the owner's `fixed_n_samples` (never
//! recopied). The `i64`-domain peek adapter below exists only because the
//! `usize` boundary cannot carry the negative ints the oracle sentences
//! describe; it renders oracle-exact text, then delegates to the owner.
//!
//! Single-cdylib tree: registers its pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `persistence_report::register` on the shared submodule); no new entry
//! point. Wiring is MAIN-ONLY (`search::register` calls this `register`
//! with its `sub`).
//!
//! Frozen consts: NONE — `statistics.py` carries no frozen value tables
//! (only the private `_STD = NormalDist()` helper, the `ClusterGrouping`
//! two-literal kept Python-side by construction, and the `SelectionConfig`
//! frozen dataclass that stays the authoritative Python home for
//! N/s/delta/alpha/beta). Verified: no `^[A-Z_]+ =` binding in
//! `python/hydra2/eval/statistics.py` outside those three.

use hydra_search::SearchError;
use hydra_search::eval::statistics as stats_owner;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyModule};

/// Strip the `invalid search argument: ` prefix so validator texts stay
/// byte-identical to the oracle (`InvalidArg` details are already the oracle
/// sentences or the short keys mapped below; every other variant is
/// unreachable for these closed inputs and keeps its full rendering).
fn invalid_detail(err: &SearchError) -> String {
    match err {
        SearchError::InvalidArg { detail } => (*detail).to_string(),
        other => other.to_string(),
    }
}

/// Map the owner's short fixed-N keys onto the oracle sentences
/// (`fixed_n_samples`, `statistics.py:117-121`): `s`/`delta` arrive bare
/// from the owner; the alpha/beta sentence is already oracle-exact.
fn fixed_n_oracle_text(detail: &str) -> String {
    match detail {
        "s" => "s must be positive and finite".to_string(),
        "delta" => "delta must be positive and finite".to_string(),
        _ => detail.to_string(),
    }
}

/// Map the owner's short peek-guard keys onto the oracle sentences
/// (`sequential_design_guard`, `statistics.py:312-323`). The CS-schedule
/// sentence is already oracle-exact; the fixed-N arm gains its declared
/// design suffix; unknown designs render Python-repr single-quoted (exact
/// for the closed-domain ids — same note as `eval_schedule::wall_repr`).
fn sequential_oracle_text(design: &str, detail: &str) -> String {
    match detail {
        "fixed-N forbids intermediate peeks" => {
            "fixed-N design forbids intermediate peeks; declare 'time_uniform_cs' instead"
                .to_string()
        }
        "unknown design" => format!("unknown design '{design}'"),
        _ => detail.to_string(),
    }
}

/// Stage a Python resample count that must be an `int >= 100`
/// (`_validate_alpha_resamples`, `statistics.py:88-89`): bools, `None`,
/// non-ints, small, and out-of-`i64` values all fail closed with the oracle
/// sentence (absurd-huge ints fail here where the oracle would accept-then-
/// OOM downstream in numpy — garbage-only arm, documented above).
fn stage_resamples(raw: &Bound<'_, PyAny>) -> Result<usize, String> {
    const ERR: &str = "resamples must be an int >= 100";
    if raw.is_instance_of::<PyBool>() {
        return Err(ERR.to_string());
    }
    let value: i64 = raw.extract().map_err(|_| ERR.to_string())?;
    if value < 100 {
        return Err(ERR.to_string());
    }
    usize::try_from(value).map_err(|_| ERR.to_string())
}

/// `i64`-domain peek adapter (`sequential_design_guard`,
/// `statistics.py:312-323`): the `usize` boundary cannot carry the negative
/// ints the oracle sentences describe, so this renders oracle-exact text
/// over `i64` first, then the caller delegates to the owner (whose
/// re-validation is then unreachable). `fixed_n` stays len-only verbatim
/// (values unread, negatives included — the `as` wrap there is harmless);
/// `time_uniform_cs` replicates `peeks != sorted(set(peeks)) or peeks[0] < 1`
/// as nonempty + strictly-increasing + first `>= 1` over `i64`.
fn adapt_peeks(design: &str, peeks: &[i64]) -> Result<Vec<usize>, String> {
    if design == "fixed_n" {
        if peeks.len() != 1 {
            return Err(
                "fixed-N design forbids intermediate peeks; declare 'time_uniform_cs' instead"
                    .to_string(),
            );
        }
        return peeks
            .iter()
            .map(|p| usize::try_from(*p).map_err(|_| "peek must be a non-negative int".to_string()))
            .collect();
    }
    if design == "time_uniform_cs" {
        const ERR: &str = "declared CS peek schedule must be nonempty, strictly increasing";
        if peeks.is_empty() {
            return Err(ERR.to_string());
        }
        let mut i = 0;
        while i < peeks.len() {
            if peeks[i] < 1 {
                return Err(ERR.to_string());
            }
            if i > 0 && peeks[i - 1] >= peeks[i] {
                return Err(ERR.to_string());
            }
            i += 1;
        }
        // proof: each `*p` >= 1 (checked in the loop above), fits `usize`.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let out: Vec<usize> = peeks.iter().map(|p| *p as usize).collect();
        return Ok(out);
    }
    Err(format!("unknown design '{design}'"))
}

/// SPEC 18.3 fixed-N (`fixed_n_samples`, `statistics.py:115-124` via
/// `fixed_n_samples`): `N = ceil(((z_(1-a) + z_(1-b)) * s / d)^2)`.
/// Int-exact on the recorded goldens (Acklam `~1e-9` `inv_cdf` behind the
/// owner's `ceil`, margin dwarfs error — see module docs). Compute runs
/// detached; only owned scalars cross.
#[pyfunction]
fn eval_stats_fixed_n_samples(
    py: Python<'_>,
    s: f64,
    delta: f64,
    alpha: f64,
    beta: f64,
) -> PyResult<u64> {
    let out = py.detach(|| stats_owner::fixed_n_samples(s, delta, alpha, beta));
    out.map_err(|err| PyValueError::new_err(fixed_n_oracle_text(&invalid_detail(&err))))
}

/// Percentile indexes (`_bootstrap_percentile_indexes`,
/// `statistics.py:127-138` via `percentile_indexes`):
/// `low = floor(a/2*R)`, `high = ceil((1-a/2)*R)-1`. Bit-exact.
/// `resamples` stages attached (bool/non-int/small fail with the oracle
/// sentence); the index math runs detached.
#[pyfunction]
fn eval_stats_percentile_indexes(
    py: Python<'_>,
    alpha: f64,
    resamples: Bound<'_, PyAny>,
) -> PyResult<(usize, usize)> {
    let count = stage_resamples(&resamples).map_err(PyValueError::new_err)?;
    let out = py.detach(|| stats_owner::percentile_indexes(alpha, count));
    out.map_err(|err| PyValueError::new_err(invalid_detail(&err)))
}

/// Gate helper (`ci_covers`, `statistics.py:223-225` via `ci_covers`):
/// `low <= truth <= high`. Bit-exact. No validation on either side — pure
/// `<=` comparison, detached.
#[pyfunction]
fn eval_stats_ci_covers(py: Python<'_>, low: f64, high: f64, truth: f64) -> PyResult<bool> {
    Ok(py.detach(|| stats_owner::ci_covers((low, high), truth)))
}

/// Peek-discipline guard (`sequential_design_guard`, `statistics.py:304-323`
/// via `sequential_design_guard`): `fixed_n` allows exactly one look,
/// `time_uniform_cs` takes a nonempty strictly-increasing 1-based schedule,
/// anything else raises with the oracle sentence. The `i64` adapter renders
/// the unrepresentable (negative) arm attached; the owner decision runs
/// detached.
#[pyfunction]
fn eval_stats_sequential_design_guard(
    py: Python<'_>,
    design: String,
    declared_peeks: Vec<i64>,
) -> PyResult<()> {
    let owned = adapt_peeks(&design, &declared_peeks).map_err(PyValueError::new_err)?;
    let out = py.detach(|| stats_owner::sequential_design_guard(&design, &owned));
    out.map_err(|err| PyValueError::new_err(sequential_oracle_text(&design, &invalid_detail(&err))))
}

/// Register the stat leaves on the shared `search` submodule (mirrors
/// `persistence_report::register` on the shared submodule): compute
/// detached, wrap attached; single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(eval_stats_fixed_n_samples, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_stats_percentile_indexes, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_stats_ci_covers, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_stats_sequential_design_guard, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_n_goldens_match_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // HEAD oracle pins
            // (test_statistics_wp03b::test_fixed_n_formula_reference_value):
            // s=1.0/delta=0.5/a=0.05/b=0.20 -> 25;
            // s=2.3/delta=0.7/a=b=0.10 -> ceil(70.9237...) = 71.
            assert_eq!(
                eval_stats_fixed_n_samples(py, 1.0, 0.5, 0.05, 0.20).unwrap(),
                25
            );
            assert_eq!(
                eval_stats_fixed_n_samples(py, 2.3, 0.7, 0.10, 0.10).unwrap(),
                71
            );
            let err = eval_stats_fixed_n_samples(py, 0.0, 0.5, 0.05, 0.20).unwrap_err();
            assert!(err.to_string().contains("s must be positive and finite"));
            let err = eval_stats_fixed_n_samples(py, 1.0, -1.0, 0.05, 0.20).unwrap_err();
            assert!(
                err.to_string()
                    .contains("delta must be positive and finite")
            );
            let err = eval_stats_fixed_n_samples(py, 1.0, 0.5, 0.0, 0.20).unwrap_err();
            assert!(
                err.to_string()
                    .contains("alpha and beta must lie in (0, 1)")
            );
        });
    }

    #[test]
    fn percentile_indexes_match_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // T3 pins R=2000, alpha=0.05 -> (50, 1949) (statistics.py:130-133).
            let full = 2000i64.into_pyobject(py).unwrap().into_any();
            assert_eq!(
                eval_stats_percentile_indexes(py, 0.05, full).unwrap(),
                (50, 1949)
            );
            let again = 2000i64.into_pyobject(py).unwrap().into_any();
            let err = eval_stats_percentile_indexes(py, 0.5, again).unwrap_err();
            assert!(err.to_string().contains("alpha must lie in (0, 0.5)"));
            let small = 99i64.into_pyobject(py).unwrap().into_any();
            let err = eval_stats_percentile_indexes(py, 0.05, small).unwrap_err();
            assert!(err.to_string().contains("resamples must be an int >= 100"));
            let flag = true.into_pyobject(py).unwrap().to_owned().into_any();
            let err = eval_stats_percentile_indexes(py, 0.05, flag).unwrap_err();
            assert!(err.to_string().contains("resamples must be an int >= 100"));
            let flot = 2000.0f64.into_pyobject(py).unwrap().into_any();
            let err = eval_stats_percentile_indexes(py, 0.05, flot).unwrap_err();
            assert!(err.to_string().contains("resamples must be an int >= 100"));
        });
    }

    #[test]
    fn ci_covers_exact() {
        Python::initialize();
        Python::attach(|py| {
            assert!(eval_stats_ci_covers(py, 0.1, 0.9, 0.5).unwrap());
            assert!(!eval_stats_ci_covers(py, 0.1, 0.9, 0.95).unwrap());
            // Boundary inclusion matches the oracle `<=` chain.
            assert!(eval_stats_ci_covers(py, 0.1, 0.9, 0.9).unwrap());
            assert!(eval_stats_ci_covers(py, -4.0, 4.0, 0.0).unwrap());
        });
    }

    #[test]
    fn sequential_guard_matches_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // Valid arms pass (statistics.py:312-322).
            eval_stats_sequential_design_guard(py, "fixed_n".to_string(), vec![25]).unwrap();
            eval_stats_sequential_design_guard(py, "time_uniform_cs".to_string(), vec![1, 5, 25])
                .unwrap();
            // Second fixed-N look raises with the declared-design suffix.
            let err = eval_stats_sequential_design_guard(py, "fixed_n".to_string(), vec![10, 25])
                .unwrap_err();
            assert!(err.to_string().contains(
                "fixed-N design forbids intermediate peeks; declare 'time_uniform_cs' instead"
            ));
            // Duplicate peeks break the strictly-increasing arm.
            let err =
                eval_stats_sequential_design_guard(py, "time_uniform_cs".to_string(), vec![5, 5])
                    .unwrap_err();
            assert!(
                err.to_string()
                    .contains("declared CS peek schedule must be nonempty, strictly increasing")
            );
            // Zero peeks fail the 1-based arm with the same sentence.
            let err =
                eval_stats_sequential_design_guard(py, "time_uniform_cs".to_string(), vec![0, 15])
                    .unwrap_err();
            assert!(
                err.to_string()
                    .contains("declared CS peek schedule must be nonempty, strictly increasing")
            );
            // Unknown designs render Python-repr single-quoted.
            let err = eval_stats_sequential_design_guard(py, "always_peek".to_string(), vec![1])
                .unwrap_err();
            assert!(err.to_string().contains("unknown design 'always_peek'"));
        });
    }
}
