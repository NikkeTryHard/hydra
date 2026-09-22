//! sampled_kernel: SPEC 14.3.1 sampled-mode frozen consts + pure sampling-math
//! leaves on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (pure `std` f64/int math plus `str` equality; the categorical
//! draws already route through the search owner `sampled_draws`
//! (`search.rs:449` over `hydra_search::belief::sampled_draws`) — same edge as
//! `search.rs`, no new dependency). FORBIDS: JCS/sha reimplementation (hashing
//! stays with the feed `canon`/`digest` owners), wall-clock/RNG (every
//! `RandomStream` draw stays caller-side; `sampled_expected_weight` takes the
//! validated `(prob, draws)` scalars and `sampled_weights_match` takes the two
//! floats), live-object orchestration (`SampledKernelConfig`/`SampledSuccessor`
//! construction, frame enumeration via `NaturalPacketKernel`, CTR cursor
//! replay, and the provenance mapping stay Python), and `ContractError`
//! shaping (the bridge raises `ValueError`; thin Python translators map to
//! `ContractError` with byte-identical oracle text).
//!
//! TABLE ported here — oracle `python/hydra2/belief/sampled_kernel.py`
//! (worktree tag at port time `sampled_kernel.py#305B`; HEAD
//! `src/hydra2/belief/sampled_kernel.py` verified identical this session via
//! `git show HEAD:src/hydra2/belief/sampled_kernel.py`):
//! - const `SAMPLED_KERNEL_MODE` <- mode identity (`sampled_kernel.py:43`):
//!   `"natural_trace_sample_v1"`, fresh literal (no feed/search owner).
//! - const `SAMPLED_KERNEL_TOL_DEFAULT` <- config default
//!   (`sampled_kernel.py:51`): `1e-9`, fresh literal.
//! - const `SAMPLED_KERNEL_TOL_UPPER` <- tolerance gate
//!   (`sampled_kernel.py:63`): `0.01` exclusive upper, fresh literal.
//! - const `SAMPLED_WEIGHT_TOL` <- bridge==oracle noise gate
//!   (`sampled_kernel.py:178`): `1e-15`, fresh literal.
//! - `sampled_check_samples` <- `SampledKernelConfig` draw-count gate
//!   (`sampled_kernel.py:54-59`): positive `int`, `bool` excluded.
//! - `sampled_check_tolerance` <- `SampledKernelConfig` tolerance gate
//!   (`sampled_kernel.py:60-65`): strict `float` in `(0, 0.01)`.
//! - `sampled_check_raw_weight` <- `SampledSuccessor` weight gate
//!   (`sampled_kernel.py:83-88`): strict `float`, finite, `>= 0`.
//! - `sampled_check_provenance_mode` <- `SampledSuccessor` mode gate
//!   (`sampled_kernel.py:89-91`): `mode == SAMPLED_KERNEL_MODE`.
//! - `sampled_frame_probability` <- frame-law gate
//!   (`sampled_kernel.py:94-98`): finite nonnegative (the caller-side
//!   `float(getattr(...))` conversion stays Python, preserving the oracle
//!   exception types for garbage).
//! - `sampled_expected_weight` <- raw-weight fold (`sampled_kernel.py:177`):
//!   `P(chosen) / draws` over validated scalars.
//! - `sampled_weights_match` <- byte-identity predicate
//!   (`sampled_kernel.py:178`): exact equality or `|w - e| <= 1e-15`.
//!
//! LOGIC staying Python: the two dataclasses (validating roots per the
//! `contracts.rs:21-26` rank-3 note) and every `__all__`, the successor
//! world/delta non-empty-`str` gates (generic one-line hygiene, cheaper than
//! an FFI round-trip), the stale/particle/rng gates, the frame-mass `fsum`
//! fold and its positivity gate, the pick range gate (Python-`repr` text is
//! formatted caller-side), the `sampled_draws` fan-out plus `jump_to` replay,
//! the packet/refs/provenance assembly, and the `_ctr_seed_cursor` /
//! `_require_search_bridge` patch-points imported from `natural.py`.
//!
//! NONE-owner note: `rg` for `SAMPLED_|sampled_|natural_trace_sample` over
//! `crates/bridge/src` is empty before this file, and `rg` for
//! `SAMPLED|natural_trace_sample|samples_per_parent` over
//! `crates/{search,feed,shard}/src` is empty — the mode string and numeric
//! gates are fresh literals with no feed/search owner to reference (checked
//! this session).
//!
//! Oracle-divergence notes (deliberate, garbage-input only):
//! - Draw counts beyond `i64::MAX` reject at the config gate with the
//!   config-identical message instead of travelling to the later
//!   `sampled_draws` rejection; well-typed callers (small `L`) are unaffected.
//! - A numeric-string frame probability (e.g. `"0.5"`, which the oracle's
//!   `float(...)` accepts) rejects with the frame-identical message; frame
//!   successors always carry real floats.
//! - `weights_match(inf, inf)` reports a match (`inf == inf`); bridge weights
//!   are finite by construction, so the predicate never sees infinities via
//!   the oracle path.
//!
//! Shape per fn: attached staging (`bool`-rejection plus strict-`float`/`str`
//! gates touch Python memory, never detached — the `kernel_check_tolerance`
//! precedent at `belief_kernel.rs:203-216`) -> ONE `py.detach(|| ...)` over
//! owned plain data with zero Python API inside for the two arithmetic leaves
//! (the `world_id_from_doc` precedent at `belief_leaves.rs:155-160`) ->
//! attached wrap as `PyValueError` (per `contracts.rs:117-123`). Consts via
//! `sub.add` (per `contracts.rs:1153-1156`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `belief_leaves.rs:192-229`); no new entry
//! point. MAIN wiring: `pub mod sampled_kernel;` in `lib.rs` plus
//! `crate::sampled_kernel::register(&sub)?;` in `contracts.rs` next to
//! `crate::belief_natural::register(&sub)?;` (`contracts.rs:1307`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyFloat, PyModule, PyString};

/// Frozen mode identity (`sampled_kernel.py:43`): every sampled batch and
/// downstream key binds this string. Fresh literal — no feed/search owner
/// (see the NONE-owner note above).
const SAMPLED_KERNEL_MODE: &str = "natural_trace_sample_v1";

/// Frozen default tolerance (`sampled_kernel.py:51`): `SampledKernelConfig`
/// default `kernel_tolerance`. Fresh literal.
const SAMPLED_KERNEL_TOL_DEFAULT: f64 = 1e-9;

/// Exclusive tolerance upper bound (`sampled_kernel.py:63`): `0.01` rejects.
/// Fresh literal.
const SAMPLED_KERNEL_TOL_UPPER: f64 = 0.01;

/// Bridge==oracle noise gate (`sampled_kernel.py:178`): `1e-15` admits float
/// noise only, never drift. Fresh literal.
const SAMPLED_WEIGHT_TOL: f64 = 1e-15;

/// Pure draw-count core: `value > 0` (mirrors `sampled_kernel.py:54-59` past
/// the caller-side `bool`-excluded `int` staging). No Python API inside
/// (detach-safe); failures are marker `&str`s the pyfn maps to `ValueError`.
fn samples_core(value: i64) -> Result<u64, &'static str> {
    if value <= 0 {
        return Err("samples");
    }
    // proof: `value > 0` (checked above); `try_from` fails closed on huge.
    u64::try_from(value).map_err(|_| "samples")
}

/// Pure tolerance core: strict `(0, 0.01)` gate (mirrors
/// `sampled_kernel.py:60-65`). Non-finite values reject here (TOTAL: NaN
/// slips the oracle's `<` comparisons, so the bridge closes the gap with the
/// same oracle-identical message the pyfn raises).
fn tolerance_core(value: f64) -> Result<f64, &'static str> {
    if !value.is_finite() || value <= 0.0 || value >= SAMPLED_KERNEL_TOL_UPPER {
        return Err("tolerance");
    }
    Ok(value)
}

/// Pure raw-weight core: finite `>= 0` gate (mirrors
/// `sampled_kernel.py:83-88`; `-0.0` passes since the oracle's `< 0` is false
/// for negative zero). No Python API inside (detach-safe).
fn raw_weight_core(value: f64) -> Result<f64, &'static str> {
    if !value.is_finite() || value < 0.0 {
        return Err("raw_weight");
    }
    Ok(value)
}

/// Pure provenance-mode core: `mode == SAMPLED_KERNEL_MODE` (mirrors
/// `sampled_kernel.py:89-91`). No Python API inside (detach-safe).
fn provenance_mode_core(mode: &str) -> Result<(), &'static str> {
    if mode == SAMPLED_KERNEL_MODE {
        Ok(())
    } else {
        Err("mode")
    }
}

/// Pure frame-probability core: finite nonnegative gate (mirrors
/// `sampled_kernel.py:94-98` past the caller-side `float(...)` conversion).
/// No Python API inside (detach-safe).
fn frame_prob_core(value: f64) -> Result<f64, &'static str> {
    if !value.is_finite() || value < 0.0 {
        return Err("frame_prob");
    }
    Ok(value)
}

/// Pure raw-weight fold: `P(chosen) / draws` (mirrors
/// `sampled_kernel.py:177`). Callers pass validated inputs (`prob` finite
/// nonnegative, `draws >= 1`), so the division is infallible: the divisor is
/// nonzero and the quotient is bounded by the finite `prob`. No Python API
/// inside (detach-safe).
fn expected_weight_core(prob: f64, draws: u64) -> f64 {
    // proof: draw counts are small (< 2^53), exact; weight tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let draws_f: f64 = draws as f64;
    prob / draws_f
}

/// Pure byte-identity predicate (mirrors `sampled_kernel.py:178`): exact
/// equality wins, otherwise `|w - e| <= SAMPLED_WEIGHT_TOL` admits float
/// noise only. TOTAL over all `f64` pairs — never rejects (`NaN` simply
/// mismatches). No Python API inside (detach-safe).
fn weights_match_core(bridge_w: f64, expected: f64) -> bool {
    bridge_w == expected || (bridge_w - expected).abs() <= SAMPLED_WEIGHT_TOL
}

/// Validated samples-per-parent-action (mirrors the `SampledKernelConfig`
/// gate bit-for-bit: only a non-`bool` `int` above zero passes — the
/// `checked_u32` precedent at `belief_kernel.rs:107-113` for the
/// `bool`-first exclusion). The message is the oracle text
/// (`sampled_kernel.py:59`), so the Python translator re-raises it
/// byte-identical.
#[pyfunction]
#[pyo3(signature = (value,))]
fn sampled_check_samples(value: Bound<'_, PyAny>) -> PyResult<u64> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(
            "samples_per_parent_action must be a positive int",
        ));
    }
    let staged: i64 = value
        .extract()
        .map_err(|_| PyValueError::new_err("samples_per_parent_action must be a positive int"))?;
    samples_core(staged)
        .map_err(|_| PyValueError::new_err("samples_per_parent_action must be a positive int"))
}

/// Validated sampled-mode tolerance (mirrors the `SampledKernelConfig` gate
/// bit-for-bit: only a `float` in `(0, 0.01)` passes — `int`/`bool` reject
/// via the strict-`float` gate, the `is_instance_of::<PyFloat>` precedent at
/// `search_profiles.rs:288-299` via `belief_kernel.rs:203-216`). The message
/// is the oracle text (`sampled_kernel.py:65`), so the Python translator
/// re-raises it byte-identical.
#[pyfunction]
#[pyo3(signature = (value,))]
fn sampled_check_tolerance(value: Bound<'_, PyAny>) -> PyResult<f64> {
    if !value.is_instance_of::<PyFloat>() {
        return Err(PyValueError::new_err(
            "kernel_tolerance must be a float in (0, 0.01)",
        ));
    }
    let staged: f64 = value
        .extract()
        .map_err(|_| PyValueError::new_err("kernel_tolerance must be a float in (0, 0.01)"))?;
    tolerance_core(staged)
        .map_err(|_| PyValueError::new_err("kernel_tolerance must be a float in (0, 0.01)"))
}

/// Validated successor raw weight (mirrors the `SampledSuccessor` gate
/// bit-for-bit: only a `float` that is finite and `>= 0` passes — same
/// strict-`float` staging as `sampled_check_tolerance`). The message is the
/// oracle text (`sampled_kernel.py:88`), so the Python translator re-raises
/// it byte-identical.
#[pyfunction]
#[pyo3(signature = (value,))]
fn sampled_check_raw_weight(value: Bound<'_, PyAny>) -> PyResult<f64> {
    if !value.is_instance_of::<PyFloat>() {
        return Err(PyValueError::new_err(
            "raw_weight must be a finite nonnegative float",
        ));
    }
    let staged: f64 = value
        .extract()
        .map_err(|_| PyValueError::new_err("raw_weight must be a finite nonnegative float"))?;
    raw_weight_core(staged)
        .map_err(|_| PyValueError::new_err("raw_weight must be a finite nonnegative float"))
}

/// Validated provenance mode (mirrors the `SampledSuccessor` gate: only the
/// frozen `SAMPLED_KERNEL_MODE` string passes — the `is_instance_of::<PyString>`
/// staging precedent at `rc_require.rs:485-511`). The message renders the
/// Python-repr single-quoted mode (the oracle
/// `f"provenance mode must be {SAMPLED_KERNEL_MODE!r}"` text, built from the
/// const so drift fails the `provenance_message` test first); the Python
/// translator re-raises it byte-identical.
#[pyfunction]
#[pyo3(signature = (value,))]
fn sampled_check_provenance_mode(value: Bound<'_, PyAny>) -> PyResult<String> {
    let err = || PyValueError::new_err(format!("provenance mode must be '{SAMPLED_KERNEL_MODE}'"));
    if !value.is_instance_of::<PyString>() {
        return Err(err());
    }
    let text: String = value.extract().map_err(|_| err())?;
    provenance_mode_core(&text)
        .map(|()| text)
        .map_err(|_| err())
}

/// Validated frame-law probability (mirrors `_frame_probability` past the
/// caller-side `float(getattr(...))` conversion: any Python number crossing
/// as `f64` is gated finite-nonnegative). The message is the oracle text
/// (`sampled_kernel.py:97`), so the Python translator re-raises it
/// byte-identical.
#[pyfunction]
#[pyo3(signature = (value,))]
fn sampled_frame_probability(value: Bound<'_, PyAny>) -> PyResult<f64> {
    let staged: f64 = value.extract().map_err(|_| {
        PyValueError::new_err("frame successor probability must be finite nonnegative")
    })?;
    frame_prob_core(staged).map_err(|_| {
        PyValueError::new_err("frame successor probability must be finite nonnegative")
    })
}

/// Expected raw weight `P(chosen) / draws` over validated scalars (mirrors
/// `sampled_kernel.py:177` without live successors — callers pass the
/// frame-validated probability and the config-validated draw count, the
/// `PacketSuccessor` precedent at `contracts.rs:259-261`). Staging runs
/// attached with the draw-count and frame-probability oracle texts; the
/// division runs detached with zero Python API inside (the
/// `world_id_from_doc` precedent at `belief_leaves.rs:155-160`).
#[pyfunction]
#[pyo3(signature = (prob, draws))]
fn sampled_expected_weight(
    py: Python<'_>,
    prob: Bound<'_, PyAny>,
    draws: Bound<'_, PyAny>,
) -> PyResult<f64> {
    let staged_prob: f64 = prob.extract().map_err(|_| {
        PyValueError::new_err("frame successor probability must be finite nonnegative")
    })?;
    frame_prob_core(staged_prob).map_err(|_| {
        PyValueError::new_err("frame successor probability must be finite nonnegative")
    })?;
    if draws.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(
            "samples_per_parent_action must be a positive int",
        ));
    }
    let staged_draws: i64 = draws
        .extract()
        .map_err(|_| PyValueError::new_err("samples_per_parent_action must be a positive int"))?;
    let count = samples_core(staged_draws)
        .map_err(|_| PyValueError::new_err("samples_per_parent_action must be a positive int"))?;
    Ok(py.detach(|| expected_weight_core(staged_prob, count)))
}

/// Byte-identity predicate for the bridge weight check (mirrors
/// `sampled_kernel.py:178` without formatting — Rust never renders Python
/// reprs, so the translator formats the oracle-identical
/// `bridge raw_weight ... (bridge!=oracle)` text). Compute runs detached
/// with zero Python API inside; `NaN` pairs simply mismatch, never reject.
#[pyfunction]
#[pyo3(signature = (bridge_w, expected))]
fn sampled_weights_match(py: Python<'_>, bridge_w: f64, expected: f64) -> PyResult<bool> {
    Ok(py.detach(|| weights_match_core(bridge_w, expected)))
}

/// Register the sampled-mode consts + leaves on the shared `contracts`
/// submodule (mirrors `belief_leaves.rs:192-229`): `let py = sub.py();`
/// (per `contracts.rs:1115`), fns via `wrap_pyfunction!(f, sub)` (per
/// `belief_leaves.rs:194`), consts via `sub.add` (per
/// `contracts.rs:1153-1156`); single cdylib, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add_function(wrap_pyfunction!(sampled_check_samples, sub)?)?;
    sub.add_function(wrap_pyfunction!(sampled_check_tolerance, sub)?)?;
    sub.add_function(wrap_pyfunction!(sampled_check_raw_weight, sub)?)?;
    sub.add_function(wrap_pyfunction!(sampled_check_provenance_mode, sub)?)?;
    sub.add_function(wrap_pyfunction!(sampled_frame_probability, sub)?)?;
    sub.add_function(wrap_pyfunction!(sampled_expected_weight, sub)?)?;
    sub.add_function(wrap_pyfunction!(sampled_weights_match, sub)?)?;
    sub.add("SAMPLED_KERNEL_MODE", SAMPLED_KERNEL_MODE)?;
    sub.add("SAMPLED_KERNEL_TOL_DEFAULT", SAMPLED_KERNEL_TOL_DEFAULT)?;
    sub.add("SAMPLED_KERNEL_TOL_UPPER", SAMPLED_KERNEL_TOL_UPPER)?;
    sub.add("SAMPLED_WEIGHT_TOL", SAMPLED_WEIGHT_TOL)?;
    // Touch `py` the way const-bearing leaves do (per `contracts.rs:1156`):
    // keeps the handle live for the scalar adds above.
    let _ = py;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_consts_match_oracle_literals() {
        // Hand-derived from the oracle literals (`sampled_kernel.py:43,51,63,
        // 178`); any drift fails here first.
        assert_eq!(SAMPLED_KERNEL_MODE, "natural_trace_sample_v1");
        assert_eq!(SAMPLED_KERNEL_TOL_DEFAULT, 1e-9);
        assert_eq!(SAMPLED_KERNEL_TOL_UPPER, 0.01);
        assert_eq!(SAMPLED_WEIGHT_TOL, 1e-15);
    }

    #[test]
    fn provenance_message_renders_python_repr() {
        // The bridge formats the oracle `f"... {MODE!r}"` text with explicit
        // single quotes; compared programmatically, never by eye.
        assert_eq!(
            format!("provenance mode must be '{SAMPLED_KERNEL_MODE}'"),
            "provenance mode must be 'natural_trace_sample_v1'"
        );
    }

    #[test]
    fn samples_core_matches_config_gate() {
        // Hand-derived from `sampled_kernel.py:54-59` (positive `int`;
        // `bool` exclusion and huge-positive staging live in the pyfn with
        // the same message — see the module divergence note).
        assert_eq!(samples_core(1), Ok(1));
        assert_eq!(samples_core(4), Ok(4));
        assert!(samples_core(0).is_err());
        assert!(samples_core(-1).is_err());
    }

    #[test]
    fn tolerance_core_matches_config_gate() {
        // Hand-derived from `sampled_kernel.py:60-65` (strict `(0, 0.01)`
        // plus the TOTAL gap closure: NaN slips the oracle's `<` chain, so
        // the bridge rejects it with the same oracle-identical message).
        assert_eq!(tolerance_core(1e-9), Ok(1e-9));
        assert_eq!(tolerance_core(0.009), Ok(0.009));
        assert!(tolerance_core(0.0).is_err());
        assert!(tolerance_core(-1e-9).is_err());
        assert!(tolerance_core(0.01).is_err());
        assert!(tolerance_core(1.0).is_err());
        assert!(tolerance_core(f64::NAN).is_err());
        assert!(tolerance_core(f64::INFINITY).is_err());
    }

    #[test]
    fn raw_weight_core_matches_successor_gate() {
        // Hand-derived from `sampled_kernel.py:83-88` (`-0.0` passes: the
        // oracle's `< 0` is false for negative zero).
        assert_eq!(raw_weight_core(0.0), Ok(0.0));
        assert_eq!(raw_weight_core(0.125), Ok(0.125));
        assert_eq!(raw_weight_core(-0.0), Ok(-0.0));
        assert!(raw_weight_core(-1e-12).is_err());
        assert!(raw_weight_core(f64::NAN).is_err());
        assert!(raw_weight_core(f64::INFINITY).is_err());
    }

    #[test]
    fn provenance_mode_core_binds_frozen_mode() {
        // Hand-derived from `sampled_kernel.py:89-91` (exact match only;
        // case variants and empty strings reject).
        assert_eq!(provenance_mode_core("natural_trace_sample_v1"), Ok(()));
        assert!(provenance_mode_core("").is_err());
        assert!(provenance_mode_core("natural_trace_sample_v2").is_err());
        assert!(provenance_mode_core("NATURAL_TRACE_SAMPLE_V1").is_err());
    }

    #[test]
    fn frame_prob_core_matches_frame_gate() {
        // Hand-derived from `sampled_kernel.py:94-98`.
        assert_eq!(frame_prob_core(0.0), Ok(0.0));
        assert_eq!(frame_prob_core(0.5), Ok(0.5));
        assert!(frame_prob_core(-1e-9).is_err());
        assert!(frame_prob_core(f64::NAN).is_err());
        assert!(frame_prob_core(f64::INFINITY).is_err());
    }

    #[test]
    fn expected_weight_core_matches_raw_fold() {
        // Hand-derived from `sampled_kernel.py:177` (`P/draws`): `0.5 / 4`
        // divides a power of two, so `0.125` is exact in binary — the same
        // value the parity test pins per draw.
        assert_eq!(expected_weight_core(0.5, 4), 0.125);
        assert_eq!(expected_weight_core(0.5, 1), 0.5);
        assert_eq!(expected_weight_core(0.0, 4), 0.0);
        assert_eq!(expected_weight_core(1.0, 4), 0.25);
    }

    #[test]
    fn weights_match_core_matches_identity_gate() {
        // Hand-derived from `sampled_kernel.py:178`: exact equality wins,
        // else `|w - e| <= 1e-15`; `NaN` never matches (both slots).
        assert!(weights_match_core(0.125, 0.125));
        assert!(weights_match_core(0.125 + 1e-16, 0.125));
        assert!(!weights_match_core(0.126, 0.125));
        assert!(!weights_match_core(0.5, 0.125));
        assert!(!weights_match_core(f64::NAN, 0.125));
        assert!(!weights_match_core(0.125, f64::NAN));
    }
}
