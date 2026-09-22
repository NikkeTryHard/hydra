//! despot_result: frozen DESPOT policy + result-math leaves.
//!
//! DAG: pyo3 ONLY (no new dependency — pure IEEE-double closed forms over
//! owned scalars; see `crates/bridge/Cargo.toml:18-25`). FORBIDS: scenario
//! sampling (belief sampler, stays Python in `despot_result.py`), telemetry
//! construction (live `ResourceTelemetry` + wall-clock, stays Python),
//! result assembly (live `UtilityVector`/`SearchResult`, stays Python),
//! budget enforcement (monotonic clock, caller-checked per standing policy),
//! ponder/observe mutation (stays Python), evidence digest (feed `canon`
//! owns bytes — never a second printer/hasher here).
//!
//! TABLE ported here (frozen values + pure compute, oracle
//! `python/hydra2/search/despot_result.py`):
//! - `despot_priority_proxy` <- `_priority_proxy_for` (`:200-209`):
//!   `0.05 / sqrt(visits)` visited bonus, `0.1` unvisited bonus (negative
//!   visits read as `0.0`, matching the oracle's fall-through), then
//!   `bonus *= 1.0 + regularization` when not `None`, returning
//!   `lower_value + bonus` in the oracle's exact op order.
//! - `despot_joules` <- `_make_telemetry` (`:261`):
//!   `model_calls * 0.5 + transitions * 0.2` left-assoc (single addition,
//!   never a compensated sum — same IEEE doubles in the same order).
//! - `despot_clamp_lower` <- `_make_result` (`:348`):
//!   `max(-5.0, min(5.0, v))` (inner `min`, then outer `max`, matching the
//!   oracle's nesting; NaN/inf behavior matches: NaN reads as `5.0` on both
//!   sides since neither side propagates it).
//!
//! TABLED (stays Python in `despot_result.py`): `_sample_natural_scenarios`
//! (live belief sampler — samplers OUT per standing policy);
//! `_lower_value_for_action` (already bridged as
//! `search.despot_lower_values`); `_budget_exhausted` /
//! `budget_exhausted_for_test` (monotonic clock budgets stay Python);
//! `_make_telemetry` minus the joules line (duration wall-clock + live
//! telemetry object); `_make_result` minus the clamp line (live utility
//! vectors + `SearchResult`); `observe`/`ponder` (planner-owned mutation);
//! the evidence digest (one `canonical_bytes_batch` call — feed canon owns
//! the bytes, hashlib renders; no second hasher here).
//!
//! NONE-duplication note: `rg` for
//! `despot_priority_proxy|despot_joules|despot_clamp_lower` over
//! `crates/bridge/src` + `python/hydra2` hits only this file (plus the
//! `despot_result.py` translators after this wave); `rg` for
//! `priority_proxy` over `crates/bridge/src` + `crates/search/src` +
//! `crates/feed/src` hits nothing (no owner exists — frozen here with
//! oracle line cites). The `0.04/0.005` joules coefficients in
//! `crates/search/src/persistence_planner.rs:262-263,307` are a different
//! planner's frozen law, never this module's `0.5/0.2` (`DESPOT_` prefix
//! keeps the two apart).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `despot_seeds::register`
//! (`crates/bridge/src/despot_seeds.rs:281-286`); `#[pyfunction]` +
//! `#[pyo3(signature = ...)]` with a `None` default mirrors
//! `persistence_packets_for` (`crates/bridge/src/search.rs:1324-1329`);
//! the stage-attached / `py.detach` / wrap-attached shape mirrors
//! `despot_lower_values` (`crates/bridge/src/search.rs:1024-1041`);
//! scalar `sub.add("CONST", CONST)` mirrors `belief_kernel::register`
//! (`crates/bridge/src/belief_kernel.rs:254-263`); pure-core +
//! `#[cfg(test)] mod tests` with hand-derived expectations mirrors
//! `despot_seeds` tests (`crates/bridge/src/despot_seeds.rs:288-360`).
//!
//! Single-cdylib tree: registers its pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `despot_seeds::register` on the shared submodule); no new entry point.
//! MAIN wiring: `pub mod despot_result;` in `lib.rs` (alphabetical, between
//! `despot_seeds` and `distill_leaves`) plus
//! `crate::despot_result::register(&sub)?;` in `search::register` next to
//! the `despot_seeds` line (`crates/bridge/src/search.rs:1735`).

use pyo3::prelude::*;

/// Unvisited-arm priority bonus (`despot_result.py:206`).
pub const DESPOT_PRIORITY_UNVISITED_BONUS: f64 = 0.1;
/// Visited-arm priority numerator (`despot_result.py:204`: `0.05 / sqrt(visits)`).
pub const DESPOT_PRIORITY_VISITED_NUMER: f64 = 0.05;
/// Joules per model call (`despot_result.py:261`).
pub const DESPOT_JOULES_PER_CALL: f64 = 0.5;
/// Joules per transition (`despot_result.py:261`).
pub const DESPOT_JOULES_PER_TRANSITION: f64 = 0.2;
/// Lower-value clamp floor (`despot_result.py:348`).
pub const DESPOT_LOWER_CLAMP_MIN: f64 = -5.0;
/// Lower-value clamp ceiling (`despot_result.py:348`).
pub const DESPOT_LOWER_CLAMP_MAX: f64 = 5.0;

/// Pure priority core (`despot_result.py:200-209`): oracle op order verbatim
/// (visited `0.05/sqrt`, unvisited `0.1`, negative reads as `0.0`, optional
/// `1.0 + regularization` scale, then `lower_value + bonus`). Owned plain
/// data only, zero Python API.
fn priority_core(lower_value: f64, visits: i64, regularization: Option<f64>) -> f64 {
    let mut bonus = 0.0;
    if visits > 0 {
        // proof: visit counts are small (< 2^53), exact; priority bonus tolerates 1ulp.
        #[allow(clippy::cast_precision_loss)]
        let visits_f: f64 = visits as f64;
        bonus = DESPOT_PRIORITY_VISITED_NUMER / visits_f.sqrt();
    } else if visits == 0 {
        bonus = DESPOT_PRIORITY_UNVISITED_BONUS;
    }
    if let Some(reg) = regularization {
        bonus *= 1.0 + reg;
    }
    lower_value + bonus
}

/// Pure joules core (`despot_result.py:261`): `(calls * 0.5) + (trans * 0.2)`
/// left-assoc, same IEEE doubles in the same order as the oracle. Owned
/// plain data only, zero Python API.
fn joules_core(model_calls: u64, transitions: u64) -> f64 {
    // proof: call counters are u64 (< 2^53), exact; joules accounting tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let calls_f: f64 = model_calls as f64;
    #[allow(clippy::cast_precision_loss)]
    let trans_f: f64 = transitions as f64;
    calls_f * DESPOT_JOULES_PER_CALL + trans_f * DESPOT_JOULES_PER_TRANSITION
}

/// Pure clamp core (`despot_result.py:348`): `clamp` + explicit NaN→`5.0`
/// (matches the oracle's inner-`min`/outer-`max` nesting where NaN reads as
/// `5.0`; `clamp` alone would return NaN — never reversed). Owned plain
/// data only, zero Python API.
fn clamp_core(value: f64) -> f64 {
    if value.is_nan() {
        DESPOT_LOWER_CLAMP_MAX
    } else {
        value.clamp(DESPOT_LOWER_CLAMP_MIN, DESPOT_LOWER_CLAMP_MAX)
    }
}

/// Heuristic search priority (`despot_result.py:200-209` — explicitly NOT an
/// upper bound). Compute runs detached; only owned scalars cross.
#[pyfunction]
#[pyo3(signature = (lower_value, visits, regularization=None))]
fn despot_priority_proxy(
    py: Python<'_>,
    lower_value: f64,
    visits: i64,
    regularization: Option<f64>,
) -> PyResult<f64> {
    Ok(py.detach(|| priority_core(lower_value, visits, regularization)))
}

/// Joules resource view (`despot_result.py:261`). Compute runs detached;
/// only owned scalars cross.
#[pyfunction]
#[pyo3(signature = (model_calls, transitions))]
fn despot_joules(py: Python<'_>, model_calls: u64, transitions: u64) -> PyResult<f64> {
    Ok(py.detach(|| joules_core(model_calls, transitions)))
}

/// Lower-value manifest clamp (`despot_result.py:348`). Compute runs
/// detached; only owned scalars cross.
#[pyfunction]
#[pyo3(signature = (value,))]
fn despot_clamp_lower(py: Python<'_>, value: f64) -> PyResult<f64> {
    Ok(py.detach(|| clamp_core(value)))
}

/// Register the despot-result leaves on the EXISTING `search` submodule
/// (mirrors `despot_seeds::register` on the shared submodule): compute
/// detached, wrap attached; single cdylib, no new entry point. MAIN calls
/// this from `search::register` — no new submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add(
        "DESPOT_PRIORITY_UNVISITED_BONUS",
        DESPOT_PRIORITY_UNVISITED_BONUS,
    )?;
    sub.add(
        "DESPOT_PRIORITY_VISITED_NUMER",
        DESPOT_PRIORITY_VISITED_NUMER,
    )?;
    sub.add("DESPOT_JOULES_PER_CALL", DESPOT_JOULES_PER_CALL)?;
    sub.add("DESPOT_JOULES_PER_TRANSITION", DESPOT_JOULES_PER_TRANSITION)?;
    sub.add("DESPOT_LOWER_CLAMP_MIN", DESPOT_LOWER_CLAMP_MIN)?;
    sub.add("DESPOT_LOWER_CLAMP_MAX", DESPOT_LOWER_CLAMP_MAX)?;
    sub.add_function(wrap_pyfunction!(despot_priority_proxy, sub)?)?;
    sub.add_function(wrap_pyfunction!(despot_joules, sub)?)?;
    sub.add_function(wrap_pyfunction!(despot_clamp_lower, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_core_matches_head_oracle() {
        // Hand-derived from HEAD (`despot_result.py:200-209`):
        // visits=0 -> 0.1 bonus; visits=4 -> 0.05/2.0 = 0.025;
        // regularization scales the bonus; negative visits read as 0.0.
        assert_eq!(priority_core(1.0, 0, None), 1.1);
        assert_eq!(priority_core(1.0, 4, None), 1.025);
        assert_eq!(priority_core(0.0, 1, Some(1.0)), 0.1);
        assert_eq!(priority_core(2.5, -3, None), 2.5);
        assert_eq!(priority_core(2.5, -3, Some(0.5)), 2.5);
    }

    #[test]
    fn joules_core_matches_head_oracle() {
        // Hand-derived from HEAD (`despot_result.py:261`):
        // 10 calls * 0.5 + 5 transitions * 0.2 = 6.0.
        assert_eq!(joules_core(10, 5), 6.0);
        assert_eq!(joules_core(0, 0), 0.0);
        assert_eq!(joules_core(1, 1), 0.7);
    }

    #[test]
    fn clamp_core_matches_head_oracle() {
        // Hand-derived from HEAD (`despot_result.py:348`): clamp to [-5, 5].
        assert_eq!(clamp_core(-7.25), -5.0);
        assert_eq!(clamp_core(3.25), 3.25);
        assert_eq!(clamp_core(9.5), 5.0);
        assert_eq!(clamp_core(-5.0), -5.0);
        assert_eq!(clamp_core(5.0), 5.0);
    }
}
