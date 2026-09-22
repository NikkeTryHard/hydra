//! analysis_leaves2: WP-12 qual-gate closed forms on the shared `search` submodule.
//!
//! DAG: pyo3 ONLY (frozen report-kind literals + pure bool/int/str kernels; no
//! feed/search-crate edge — every canon+hash identity already routes through the
//! feed `canon`/`digest` owners via `canon_rng::{sha256_hex, of_canonical_json}`
//! and the Python `of_canonical` call sites, and every budget table already
//! lives in `qual_budget.rs`). FORBIDS: dataclass construction (every
//! `AnalysisGateRecord` / `AnalysisReport` / `CandidateSpec` / `ResourceBudget`
//! stays Python in `analysis/qual_gates.py` / `search/common.py`), JCS/sha
//! reimplementation (no second printer, no second hasher), file/JSON IO,
//! wall-clock (`_utc_now` stays Python), RNG, torch/CUDA, float sums
//! (`value_l2` stays Python per bridge contract), and live-object orchestration
//! (spec factories, world helpers, `CanonicalAction` fixtures, comparison
//! assembly — callers pass validated scalars across).
//!
//! TABLE ported here (pure leaves, oracles in `python/hydra2/analysis/`):
//! - `QUAL_GATE_REPORT_KIND` <- `qual_gates.py:48` (`ANALYSIS_REPORT_KIND`).
//! - `QUAL_GATE_REPORT_SCHEMA_VERSION` <- `qual_gates.py:49`
//!   (`ANALYSIS_REPORT_SCHEMA_VERSION`).
//! - `QUAL_GATE_REASON_MAX_CHARS` <- `qual_gates.py:373` (`str(exc)[:240]`).
//! - `qual_gate_eligible` <- `build_gate_record` eligibility closed form
//!   (`qual_gates.py:360`: `compute_only and deterministic_ok and not
//!   privileged_leak`).
//! - `qual_gate_reason_truncate` <- error-path reason truncation
//!   (`qual_gates.py:373`).
//! - `qual_gate_summary` <- report summary counts (`qual_gates.py:475-481`:
//!   `total` / `eligible` / `ineligible` / `compute_only_pass` /
//!   `deterministic_pass`).
//! - `qual_gate_fallback_margin_ok` <- fallback-margin gate
//!   (`qual_replay.py:214-218`: `isinstance int, not bool, and >= 0`).
//! - `qual_gate_aid_or_zero` <- action-id truthiness projection
//!   (`qual_replay.py:219-230`: `int, not bool, and truthy, else 0`).
//! - `qual_gate_deadline_greater` <- shared deadline-monotonic core of
//!   `make_analysis_spec` (`qual_budget.py:177-180`) and `verify_compute_only`
//!   (`qual_budget.py:331-334`): returns the `an > gp` bit; the two
//!   site-specific messages (`must be >` vs `must exceed`) stay Python so both
//!   translators stay byte-identical.
//! - `qual_gate_cap_no_shrink` <- shared no-shrink core of `make_analysis_spec`
//!   (`qual_budget.py:181-187`) and `verify_compute_only`
//!   (`qual_budget.py:337-343`): returns false iff both caps are finite and
//!   `an < gp`; the `None`-when-finite arm and every site-specific message
//!   stay Python.
//!
//! LOGIC staying Python (named reasons):
//! - `AnalysisGateRecord` / `AnalysisReport` dataclasses + `compute_only_proof`
//!   delegate (live types; the bridge takes scalars, never dataclasses —
//!   `qual_budget.rs:3-10` construction barrier).
//! - `_utc_now` (wall-clock — bridge FORBIDDEN), `_load_default_hashes_for_spec`
//!   (file IO + torch model — bridge FORBIDDEN), `_make_gameplay_spec_for`
//!   (candidate factories + file-backed fallback orchestration).
//! - `build_gate_record` body (world helper, `CanonicalAction` fixtures,
//!   `make_seat`/`make_tile_id` bridge calls, `of_canonical` digests via the
//!   canon owner, comparison assembly) except the two kernels above.
//! - `generate_hashed_analysis_report` body (clocks, `atomic_replace_bytes`
//!   publishes, budgets view over live `ResourceBudget`s) except the summary
//!   kernel; `analysis_gate_for` (file IO + JSON — bridge FORBIDDEN).
//! - `verify_compute_only` semantic-field loop (19 live-`CandidateSpec` string
//!   fields + type gate — dataclass orchestration) except the monotonic bits
//!   above; `check_no_privileged_leak` (`ActorObservation` type gate +
//!   `dir()` scan + `VisibilityViolationError` — object introspection).
//! - `make_analysis_spec` construction + generic-fallback dispatch (live
//!   `CandidateSpec`/`ResourceBudget` assembly) except the monotonic bits.
//! - Replay `_aid` repr-hash canonicalization (Python `repr` has no Rust
//!   equivalent — `qual_replay.rs:13-18` TABLE, not revisited), `_pick` index
//!   modulo (two-line modulo, bridge round-trip exceeds its cost —
//!   `qual_replay.rs:19-21` TABLE), `value_l2` (`** 0.5`, no bit-exact Rust
//!   counterpart — `qual_replay.rs:19-23` TABLE; float sums stay Python per
//!   contract), `_spec_hash` (delegates to `search.common.candidate_spec_hash`,
//!   a different owner), observation-hash synthesis (`of_canonical(str(obs))`
//!   via the canon owner).
//!
//! Crate-owner search (this wave, before restating any const):
//! - No `crates/analysis-core` member exists (workspace members are feed,
//!   shard, search, bridge + standalone packager); the report-kind literals
//!   live only in `qual_gates.py:48-49`, so they are single-sourced here like
//!   `CANDIDATE_IDS` in `qual_budget.rs:60-71`.
//! - `hydra-shard` `assert_no_privileged_leak` (`crates/shard/src/writer.rs:396`)
//!   gates columnar `RecordBatch` schemas (parquet domain) — not the analysis
//!   observation firewall; not reused.
//! - Digest shape `feed::partition::is_digest_text` (`crates/feed/src/partition.rs:583`)
//!   and `sha256_hex` (`crates/feed/src/digest.rs:31`) already ride the
//!   `canon_rng` / `contracts` bridge; gate/comparison/report digests stay on
//!   those call sites — this file hashes nothing.
//!
//! PyO3 precedent cites (one per API, all read this wave): `pub fn register(sub)` +
//! `wrap_pyfunction!(f, sub)` with a borrowed `sub` mirror
//! `qual_replay::register` (`crates/bridge/src/qual_replay.rs:174-181`);
//! `sub.add` for consts mirrors `qual_budget::register`
//! (`crates/bridge/src/qual_budget.rs:482-516`); `#[pyfunction]` mirrors
//! `qual_replay::qual_replay_hash` (`crates/bridge/src/qual_replay.rs:133-134`);
//! attached-extract + one `py.detach` with zero Python API inside mirrors
//! `qual_budget::qual_budget_for` (`crates/bridge/src/qual_budget.rs:339-356`);
//! `is_instance_of::<PyBool>` bool-rejection mirrors `search_profiles::as_u64`
//! (`crates/bridge/src/search_profiles.rs:42-47`); `Option<Bound<'_, PyAny>>`
//! for nullable caps mirrors `qual_budget::as_opt_u64`
//! (`crates/bridge/src/qual_budget.rs:239-244`); `ValueError` mapping mirrors
//! `qual_budget` kernels (`crates/bridge/src/qual_budget.rs:369-421`);
//! `Python::attach` in tests mirrors `canon_rng.rs:688-689` (via
//! `eval_leaves.rs:55-56` cite); try-leaf/`except AttributeError` translators
//! mirror `blocks.py:55-63,93-104`.
//!
//! Single-cdylib tree: registers its consts + pyfns on the EXISTING
//! `hydra2._native.search` submodule via `register` (mirrors
//! `qual_replay::register` on the shared submodule, alongside
//! `qual_budget`/`qual_replay` in `search.rs:1732-1733`); no new entry point.
//! Wiring is MAIN-ONLY (`search::register` calls this `register` with its `sub`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyModule};

/// Report kind literal (`ANALYSIS_REPORT_KIND`, `qual_gates.py:48`).
pub const QUAL_GATE_REPORT_KIND: &str = "hydra2.analysis_gate_report";
/// Report schema version (`ANALYSIS_REPORT_SCHEMA_VERSION`, `qual_gates.py:49`).
pub const QUAL_GATE_REPORT_SCHEMA_VERSION: &str = "1.0.0";
/// Error-reason truncation width (`build_gate_record`, `qual_gates.py:373`).
pub const QUAL_GATE_REASON_MAX_CHARS: usize = 240;

/// Eligibility closed form (`build_gate_record`, `qual_gates.py:357-361`):
/// `compute_only and deterministic_ok and not privileged_leak`. Pure and
/// detach-safe; the `bool(...)` coercions happen in the translator, so this
/// sees only real bools.
fn gate_eligible_kernel(compute_only: bool, deterministic_ok: bool, privileged_leak: bool) -> bool {
    compute_only && deterministic_ok && !privileged_leak
}

/// Reason truncation (`build_gate_record` except path, `qual_gates.py:373`):
/// `str(exc)[:240]` — Python slices by Unicode scalar values, so this takes
/// `chars()`, never bytes (byte-slicing would split multi-byte sequences the
/// oracle keeps whole). Pure and detach-safe.
fn truncate_reason_kernel(reason: &str) -> String {
    if reason.chars().count() <= QUAL_GATE_REASON_MAX_CHARS {
        return reason.to_string();
    }
    reason.chars().take(QUAL_GATE_REASON_MAX_CHARS).collect()
}

/// Report summary counts (`generate_hashed_analysis_report`, `qual_gates.py:475-481`):
/// `(total, eligible, ineligible, compute_only_pass, deterministic_pass)` over
/// three parallel flag vectors from the same gate list. Pure and detach-safe.
fn summary_kernel(
    compute_only: &[bool],
    deterministic_ok: &[bool],
    eligible: &[bool],
) -> Result<(u64, u64, u64, u64, u64), String> {
    if compute_only.len() != deterministic_ok.len() || compute_only.len() != eligible.len() {
        return Err(format!(
            "qual gate summary flag vectors must share one length, got {}/{}/{}",
            compute_only.len(),
            deterministic_ok.len(),
            eligible.len()
        ));
    }
    let total = compute_only.len() as u64;
    let eligible_n = eligible.iter().filter(|&&v| v).count() as u64;
    let compute_n = compute_only.iter().filter(|&&v| v).count() as u64;
    let deterministic_n = deterministic_ok.iter().filter(|&&v| v).count() as u64;
    Ok((
        total,
        eligible_n,
        total - eligible_n,
        compute_n,
        deterministic_n,
    ))
}

/// Deadline-monotonic bit shared by `make_analysis_spec` (`qual_budget.py:177-180`)
/// and `verify_compute_only` (`qual_budget.py:331-334`): `an > gp`. Pure and
/// detach-safe.
fn deadline_greater_kernel(gp_deadline_ms: u64, an_deadline_ms: u64) -> bool {
    an_deadline_ms > gp_deadline_ms
}

/// No-shrink bit shared by `make_analysis_spec` (`qual_budget.py:181-187`) and
/// `verify_compute_only` (`qual_budget.py:337-343`): false iff both caps are
/// finite and `an < gp`. `None` gameplay means no constraint (true); the
/// `None`-when-finite reject arm stays Python with its site-specific message.
/// Pure and detach-safe.
fn cap_no_shrink_kernel(gp: Option<u64>, an: Option<u64>) -> bool {
    match (gp, an) {
        (Some(g), Some(a)) => a >= g,
        _ => true,
    }
}

/// Read a Python int that must not be a bool (`search_profiles.rs:42-47`
/// `as_u64` shape): bools, `None`, non-ints, and out-of-`u64` values all fail
/// closed with the caller's message.
fn as_u64(obj: &Bound<'_, PyAny>, err: &str) -> Result<u64, String> {
    if obj.is_instance_of::<PyBool>() {
        return Err(err.to_string());
    }
    obj.extract::<u64>().map_err(|_| err.to_string())
}

/// Nullable variant for caps that may be `None` (`qual_budget.rs:239-244`
/// `as_opt_u64` shape): `None` stays `None`; bools/non-ints fail closed.
fn as_opt_u64(arg: &Option<Bound<'_, PyAny>>, err: &str) -> Result<Option<u64>, String> {
    match arg {
        None => Ok(None),
        Some(obj) => {
            if obj.is_none() {
                return Ok(None);
            }
            as_u64(obj, err).map(Some)
        }
    }
}

/// Gate eligibility (`build_gate_record`, `qual_gates.py:360`): attached strict
/// `bool` extraction (translators pass `bool(...)` results, so well-typed
/// callers are exact; truthy non-`bool`s fail closed here where the oracle
/// would coerce — garbage-input only, same reject outcome), detached bit.
#[pyfunction]
#[pyo3(signature = (compute_only, deterministic_ok, privileged_leak))]
fn qual_gate_eligible(
    py: Python<'_>,
    compute_only: Bound<'_, PyAny>,
    deterministic_ok: Bound<'_, PyAny>,
    privileged_leak: Bound<'_, PyAny>,
) -> PyResult<bool> {
    let co: bool = compute_only
        .extract()
        .map_err(|_| PyValueError::new_err("qual_gate_eligible: compute_only must be a bool"))?;
    let de: bool = deterministic_ok.extract().map_err(|_| {
        PyValueError::new_err("qual_gate_eligible: deterministic_ok must be a bool")
    })?;
    let pl: bool = privileged_leak
        .extract()
        .map_err(|_| PyValueError::new_err("qual_gate_eligible: privileged_leak must be a bool"))?;
    Ok(py.detach(|| gate_eligible_kernel(co, de, pl)))
}

/// Reason truncation (`build_gate_record` except path, `qual_gates.py:373`):
/// attached `str` move, detached char-prefix. `TypeError` on non-`str`
/// (translators pass `str(exc)`, so well-typed callers are exact).
#[pyfunction]
#[pyo3(signature = (reason,))]
fn qual_gate_reason_truncate(py: Python<'_>, reason: String) -> PyResult<String> {
    Ok(py.detach(|| truncate_reason_kernel(&reason)))
}

/// Report summary (`generate_hashed_analysis_report`, `qual_gates.py:475-481`):
/// attached `Vec<bool>` extraction (`TypeError` on non-`bool` elements —
/// translators pass `bool(...)` projections), detached counts; ragged vectors
/// fail closed (`ValueError`), never a partial tuple.
#[pyfunction]
#[pyo3(signature = (compute_only, deterministic_ok, eligible))]
fn qual_gate_summary(
    py: Python<'_>,
    compute_only: Vec<bool>,
    deterministic_ok: Vec<bool>,
    eligible: Vec<bool>,
) -> PyResult<(u64, u64, u64, u64, u64)> {
    py.detach(|| summary_kernel(&compute_only, &deterministic_ok, &eligible))
        .map_err(PyValueError::new_err)
}

/// Fallback-margin gate (`compare_gameplay_analysis`, `qual_replay.py:214-218`):
/// `isinstance int, not bool, and >= 0` — returns the bit, never raises on
/// type mismatch (the oracle returns `False` there too). `i128` covers every
/// realistic margin; astronomically-windowed ints read `False` (garbage-input
/// only — real budgets are `u64`-windowed small ints).
#[pyfunction]
#[pyo3(signature = (fallback_margin_ms,))]
fn qual_gate_fallback_margin_ok(fallback_margin_ms: Bound<'_, PyAny>) -> PyResult<bool> {
    if fallback_margin_ms.is_instance_of::<PyBool>() {
        return Ok(false);
    }
    match fallback_margin_ms.extract::<i128>() {
        Ok(v) => Ok(v >= 0),
        Err(_) => Ok(false),
    }
}

/// Action-id truthiness projection (`compare_gameplay_analysis`,
/// `qual_replay.py:219-230`): `int, not bool` passes through (zero maps to
/// zero either way, so the oracle's extra `bool(v)` arm is structural);
/// bools and non-ints fail closed (`ValueError` — translators only pass the
/// `getattr(action, "action_id", 0)` projection, so well-typed callers are
/// exact). Beyond-`i64` aids fail closed (garbage-input only — real aids are
/// small table ids).
#[pyfunction]
#[pyo3(signature = (value,))]
fn qual_gate_aid_or_zero(value: Bound<'_, PyAny>) -> PyResult<i64> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(
            "qual_gate_aid_or_zero: action_id must be an int, got bool",
        ));
    }
    value.extract::<i64>().map_err(|_| {
        PyValueError::new_err("qual_gate_aid_or_zero: action_id must be an int".to_string())
    })
}

/// Deadline-monotonic bit (`qual_budget.py:177-180` + `:331-334`): attached
/// bool-rejecting `u64` extraction (`ValueError` on bool/`None`/non-int —
/// translators catch `ValueError`/`TypeError` into `ContractError` exactly
/// like the `qual_budget_check_finite` path), detached `an > gp`. Translators
/// raise the site-specific text, so both messages stay byte-identical.
#[pyfunction]
#[pyo3(signature = (gp_deadline_ms, an_deadline_ms))]
fn qual_gate_deadline_greater(
    py: Python<'_>,
    gp_deadline_ms: Bound<'_, PyAny>,
    an_deadline_ms: Bound<'_, PyAny>,
) -> PyResult<bool> {
    let gp = as_u64(
        &gp_deadline_ms,
        "qual_gate_deadline_greater: gp_deadline_ms must be an int",
    )
    .map_err(PyValueError::new_err)?;
    let an = as_u64(
        &an_deadline_ms,
        "qual_gate_deadline_greater: an_deadline_ms must be an int",
    )
    .map_err(PyValueError::new_err)?;
    Ok(py.detach(|| deadline_greater_kernel(gp, an)))
}

/// No-shrink bit (`qual_budget.py:181-187` + `:337-343`): attached
/// `None`-tolerant bool-rejecting extraction, detached kernel. Translators
/// raise the site-specific shrink text, keeping both call sites byte-identical.
#[pyfunction]
#[pyo3(signature = (gp_cap, an_cap))]
fn qual_gate_cap_no_shrink(
    py: Python<'_>,
    gp_cap: Option<Bound<'_, PyAny>>,
    an_cap: Option<Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let gp = as_opt_u64(
        &gp_cap,
        "qual_gate_cap_no_shrink: gp_cap must be an int or None",
    )
    .map_err(PyValueError::new_err)?;
    let an = as_opt_u64(
        &an_cap,
        "qual_gate_cap_no_shrink: an_cap must be an int or None",
    )
    .map_err(PyValueError::new_err)?;
    Ok(py.detach(|| cap_no_shrink_kernel(gp, an)))
}

/// Register the gate consts + kernels on the EXISTING `search` submodule
/// (mirrors `qual_replay::register`, `qual_replay.rs:174-181`, on the shared
/// submodule; MAIN calls this from `search::register` next to
/// `crate::qual_replay::register(&sub)?;`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add("QUAL_GATE_REPORT_KIND", QUAL_GATE_REPORT_KIND)?;
    sub.add(
        "QUAL_GATE_REPORT_SCHEMA_VERSION",
        QUAL_GATE_REPORT_SCHEMA_VERSION,
    )?;
    sub.add(
        "QUAL_GATE_REASON_MAX_CHARS",
        QUAL_GATE_REASON_MAX_CHARS as u64,
    )?;
    sub.add_function(wrap_pyfunction!(qual_gate_eligible, sub)?)?;
    sub.add_function(wrap_pyfunction!(qual_gate_reason_truncate, sub)?)?;
    sub.add_function(wrap_pyfunction!(qual_gate_summary, sub)?)?;
    sub.add_function(wrap_pyfunction!(qual_gate_fallback_margin_ok, sub)?)?;
    sub.add_function(wrap_pyfunction!(qual_gate_aid_or_zero, sub)?)?;
    sub.add_function(wrap_pyfunction!(qual_gate_deadline_greater, sub)?)?;
    sub.add_function(wrap_pyfunction!(qual_gate_cap_no_shrink, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod analysis_leaves2_tests {
    use super::*;

    #[test]
    fn report_consts_match_oracle_literals() {
        assert_eq!(QUAL_GATE_REPORT_KIND, "hydra2.analysis_gate_report");
        assert_eq!(QUAL_GATE_REPORT_SCHEMA_VERSION, "1.0.0");
        assert_eq!(QUAL_GATE_REASON_MAX_CHARS, 240);
    }

    #[test]
    fn eligible_truth_table() {
        assert!(gate_eligible_kernel(true, true, false));
        assert!(!gate_eligible_kernel(false, true, false));
        assert!(!gate_eligible_kernel(true, false, false));
        assert!(!gate_eligible_kernel(true, true, true));
        assert!(!gate_eligible_kernel(false, false, true));
    }

    #[test]
    fn truncate_short_passthrough_and_boundary() {
        assert_eq!(truncate_reason_kernel(""), "");
        assert_eq!(truncate_reason_kernel("passed"), "passed");
        let exactly: String = "x".repeat(240);
        assert_eq!(truncate_reason_kernel(&exactly), exactly);
        let over: String = "y".repeat(241);
        assert_eq!(truncate_reason_kernel(&over), "y".repeat(240));
    }

    #[test]
    fn truncate_counts_chars_not_bytes() {
        // "é" is 2 bytes / 1 char: 239 ascii + "é" is exactly 240 chars.
        let s = format!("{}é", "a".repeat(239));
        assert_eq!(s.chars().count(), 240);
        assert_eq!(truncate_reason_kernel(&s), s);
        // 240 ascii + "é" is 241 chars; [:240] keeps the 240 ascii, drops é.
        let over = format!("{}é", "a".repeat(240));
        assert_eq!(truncate_reason_kernel(&over), "a".repeat(240));
        // Multi-byte chars are never split: 239 ascii + 2 "é" (241 chars)
        // keeps 239 ascii + the first "é" whole.
        let over_mb = format!("{}éé", "a".repeat(239));
        assert_eq!(
            truncate_reason_kernel(&over_mb),
            format!("{}é", "a".repeat(239))
        );
        assert!(truncate_reason_kernel(&over).is_char_boundary(0));
    }

    #[test]
    fn summary_counts_match_report_oracle() {
        // Mirrors qual_gates.py:475-481 over three gates.
        let (total, eligible_n, ineligible, compute_n, deterministic_n) = summary_kernel(
            &[true, true, false],
            &[true, false, false],
            &[true, false, false],
        )
        .unwrap();
        assert_eq!(
            (total, eligible_n, ineligible, compute_n, deterministic_n),
            (3, 1, 2, 2, 1)
        );
    }

    #[test]
    fn summary_empty_and_all_eligible() {
        let all = summary_kernel(&[], &[], &[]).unwrap();
        assert_eq!(all, (0, 0, 0, 0, 0));
        let full = summary_kernel(&[true, true], &[true, true], &[true, true]).unwrap();
        assert_eq!(full, (2, 2, 0, 2, 2));
    }

    #[test]
    fn summary_ragged_fails_closed() {
        assert!(summary_kernel(&[true], &[], &[]).is_err());
        assert!(summary_kernel(&[true], &[true], &[true, true]).is_err());
    }

    #[test]
    fn deadline_is_strictly_greater() {
        assert!(deadline_greater_kernel(5_000, 30_000));
        assert!(!deadline_greater_kernel(5_000, 5_000));
        assert!(!deadline_greater_kernel(30_000, 5_000));
    }

    #[test]
    fn cap_no_shrink_matches_both_call_sites() {
        assert!(cap_no_shrink_kernel(Some(64), Some(256)));
        assert!(cap_no_shrink_kernel(Some(64), Some(64)));
        assert!(!cap_no_shrink_kernel(Some(256), Some(64)));
        assert!(cap_no_shrink_kernel(None, Some(1)));
        assert!(cap_no_shrink_kernel(None, None));
        // None-when-finite is NOT a shrink here (translator's arm owns it).
        assert!(cap_no_shrink_kernel(Some(64), None));
    }
}
