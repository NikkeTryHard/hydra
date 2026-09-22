//! rc_cross_validate: run-config cross-section coherence on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (no feed/shard/search edge; pure argument validation, so no
//! crate owner exists — same NONE position as `rc_require.rs:32-38`; `rg` for
//! `loop.precision|autocast scope|positive-requires-binding|effective run id`
//! over `crates/feed/src` + `crates/search/src` hits no coherence authority).
//! FORBIDS: section parsing, YAML/file IO, `${VAR}` interpolation,
//! base+override composition (all stay Python in `_rc_root.py` /
//! `_rc_require.py`), dataclass state (all section records stay Python in
//! `_rc_sections.py`), run-id pattern ownership (the shape test restates the
//! frozen pattern owned by `rc_sections.rs:79` `RUN_ID_PATTERN`, never a new
//! printer — same restatement precedent as `rc_parse_aux.rs:5-6`), and
//! `ContractError` shaping (the bridge raises `ValueError`/`TypeError`; the
//! thin Python translator maps to `ContractError` with byte-identical
//! messages).
//!
//! TABLE ported here — the single coherence gate, oracle
//! `python/hydra2/training/_rc_root.py:105-141` (worktree at port time equals
//! `HEAD:src/hydra2/training/_rc_root.py`; the four checks, in order):
//! - `rc_cross_validate` ← `_cross_validate` (:105-141; loop/runtime
//!   precision coherence, effective-run-id shape, bf16-requires-CUDA,
//!   positive-requires-binding)
//!
//! LOGIC staying Python: `_SECTION_PARSERS`, `_parse_root` (section dispatch
//! plus dataclass assembly), `_read_yaml_mapping`/`load_run_config` (YAML
//! framing, deep-merge, interpolation), `__all__`, and the `ContractError`
//! translator (identical messages, `__all__` unchanged).
//!
//! Shape per fn: attached staging (owned `String` extraction plus live
//! `repr()` for every `{value!r}` slot, so quoting/escapes render byte-exact)
//! → ONE `py.detach(|| …)` over owned plain data with zero Python API inside
//! (per `contracts.rs:434-437`, `validate.rs:83-95`) → attached wrap as
//! `PyValueError` (per `contracts.rs:117-123`). Inputs arrive post-validation
//! (dataclass fields projected caller-side to plain scalars, never raw
//! mappings — the `rc_resume.rs:31-38` projection precedent), so
//! bool-exclusion staging is unnecessary. Non-`str` scalars in the string
//! slots fail attached with a free slot-typed text (oracle-unreachable: the
//! translator only forwards typed dataclass fields); non-`str` numerics and
//! `None`-shape violations surface as the extractor's `TypeError`, which the
//! translator already maps. Check order inside the detach mirrors the oracle
//! statement order, so the FIRST error on multi-violation inputs is identical.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `rc_resume.rs:170-176`); no new entry
//! point. MAIN wiring: `pub mod rc_cross_validate;` in `lib.rs` plus
//! `crate::rc_cross_validate::register(&sub)?;` in `contracts.rs` next to
//! `crate::rc_sections::register(&sub)?;` (`contracts.rs:1300`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `rc_resume::register`
//! (`crates/bridge/src/rc_resume.rs:173-175`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `rc_resume::rc_check_compat` (`crates/bridge/src/rc_resume.rs:128-168`);
//! `#[allow(clippy::too_many_arguments)]` on the 11-arg pyfn mirrors
//! (`crates/bridge/src/rc_resume.rs:129`); live-`repr()` staging mirrors
//! `rc_require::py_repr` (`crates/bridge/src/rc_require.rs:63-65`); the
//! `is_instance_of::<PyString>` + `extract::<String>()` text arm mirrors
//! `rc_parse::stage_opt_str` (`crates/bridge/src/rc_parse.rs:510-517`);
//! nullable `Option<Bound<'_, PyAny>>` pyfn args mirror `census_index_of`
//! (`crates/bridge/src/contracts.rs:368-370`); the `unwrap_or_else` repr
//! fallback mirrors (`crates/bridge/src/rc_parse.rs:486`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyString};

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Stage one projected `str` slot to `(text, repr)`. Non-`str` input fails
/// attached with a free slot-typed text (oracle-unreachable: the translator
/// forwards typed dataclass fields only).
fn stage_str(obj: Bound<'_, PyAny>, slot: &str) -> PyResult<(String, String)> {
    if obj.is_instance_of::<PyString>() {
        let text: String = obj.extract().map_err(|e| {
            PyValueError::new_err(format!(
                "contracts rc_cross_validate {slot} unreadable: {e}"
            ))
        })?;
        Ok((text, py_repr(&obj)?))
    } else {
        let repr = py_repr(&obj).unwrap_or_else(|_| "?".to_owned());
        Err(PyValueError::new_err(format!(
            "contracts rc_cross_validate {slot} must be str, got {repr}"
        )))
    }
}

/// Run-id shape test restating the frozen pattern owned by `rc_sections.rs:79`
/// (`RUN_ID_PATTERN` `[A-Za-z0-9][A-Za-z0-9_-]*` via `fullmatch`): ASCII-only,
/// first char alphanumeric, rest alphanumeric/`_`/`-`. Empty fails. Mirrors
/// `rc_parse.rs:185-192` exactly.
fn is_run_id_shape(text: &str) -> bool {
    let mut chars = text.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphanumeric() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Owned plain-data projection of a `RunConfig` for the coherence gate. The
/// translator fills every field positionally from live dataclass attributes;
/// `*_repr` carries the attached live-`repr()` of the same value for the
/// oracle `{value!r}` slots.
struct CrossValidateView {
    runtime_precision: String,
    runtime_precision_repr: String,
    loop_precision: String,
    loop_precision_repr: String,
    runtime_device: String,
    output_run_id: Option<String>,
    output_run_id_repr: Option<String>,
    run_run_id: String,
    run_run_id_repr: String,
    w_placement: f64,
    w_value: f64,
    w_event: Vec<f64>,
    w_belief: Vec<f64>,
    has_privileged_source_hash: bool,
}

/// Effective run id plus its staged repr (oracle `_rc_root.py:115-118`:
/// `output.run_id` wins when non-`None` and non-empty, else `run.run_id`).
/// Byte-emptiness coincides with the oracle `len() > 0` test.
fn effective_id(view: &CrossValidateView) -> (&str, &str) {
    match (&view.output_run_id, &view.output_run_id_repr) {
        (Some(text), Some(repr)) if !text.is_empty() => (text.as_str(), repr.as_str()),
        _ => (view.run_run_id.as_str(), view.run_run_id_repr.as_str()),
    }
}

/// The four coherence checks in oracle statement order (never reordered:
/// multi-violation inputs surface the earliest gate). Owned plain data only,
/// zero Python API.
fn check_cross_validate(view: &CrossValidateView) -> Result<(), String> {
    let expected_loop = if view.runtime_precision == "fp32" {
        "fp32"
    } else {
        "bf16_mixed"
    };
    if view.loop_precision != expected_loop {
        return Err(format!(
            "loop.precision {} disagrees with runtime.precision {} (expected '{expected_loop}'); autocast scope must match",
            view.loop_precision_repr, view.runtime_precision_repr,
        ));
    }
    let (effective_text, effective_repr) = effective_id(view);
    if !is_run_id_shape(effective_text) {
        return Err(format!("effective run id invalid: {effective_repr}"));
    }
    if view.runtime_precision == "bf16_mixed" && view.runtime_device == "cpu" {
        return Err(
            "runtime.precision 'bf16_mixed' requires a CUDA device (never silent CPU fallback; device availability itself is checked at bind time via require_device_available)"
                .to_owned(),
        );
    }
    let auxiliary_positive = view.w_placement > 0.0
        || view.w_value > 0.0
        || view.w_event.iter().any(|w: &f64| *w > 0.0)
        || view.w_belief.iter().any(|w: &f64| *w > 0.0);
    if auxiliary_positive && !view.has_privileged_source_hash {
        return Err(
            "positive placement/value/event/belief weight REQUIRES weights.privileged_source_hash (positive-requires-binding): privileged labels must come from the pinned manifest"
                .to_owned(),
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pyfn (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// `_cross_validate` gate (oracle `_rc_root.py:105-141`): the translator
/// passes projected dataclass fields positionally in this exact order —
/// precisions (`runtime_precision`, `loop_precision`), device
/// (`runtime_device`), run ids (`output_run_id` nullable, `run_run_id`),
/// weights (`w_placement`, `w_value`, `w_event`/`w_belief` value lists),
/// `has_privileged_source_hash` last; the accept/reject decision runs
/// detached.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rc_cross_validate(
    py: Python<'_>,
    runtime_precision: Bound<'_, PyAny>,
    loop_precision: Bound<'_, PyAny>,
    runtime_device: String,
    output_run_id: Option<Bound<'_, PyAny>>,
    run_run_id: Bound<'_, PyAny>,
    w_placement: f64,
    w_value: f64,
    w_event: Vec<f64>,
    w_belief: Vec<f64>,
    has_privileged_source_hash: bool,
) -> PyResult<()> {
    let (runtime_precision_text, runtime_precision_repr) =
        stage_str(runtime_precision, "runtime.precision")?;
    let (loop_precision_text, loop_precision_repr) = stage_str(loop_precision, "loop.precision")?;
    let (output_text, output_repr) = match output_run_id {
        None => (None, None),
        Some(obj) => {
            let (text, repr) = stage_str(obj, "output.run_id")?;
            (Some(text), Some(repr))
        }
    };
    let (run_text, run_repr) = stage_str(run_run_id, "run.run_id")?;
    let view = CrossValidateView {
        runtime_precision: runtime_precision_text,
        runtime_precision_repr,
        loop_precision: loop_precision_text,
        loop_precision_repr,
        runtime_device,
        output_run_id: output_text,
        output_run_id_repr: output_repr,
        run_run_id: run_text,
        run_run_id_repr: run_repr,
        w_placement,
        w_value,
        w_event,
        w_belief,
        has_privileged_source_hash,
    };
    py.detach(|| check_cross_validate(&view))
        .map_err(PyValueError::new_err)
}

/// Register the cross-section coherence gate on the shared `contracts`
/// submodule (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?`
/// shape at `rc_resume.rs:174`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(rc_cross_validate, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Base inputs: every gate passes (fp32/fp32 on CUDA, valid run id, all
    /// weights zero so no binding is required).
    fn base() -> CrossValidateView {
        CrossValidateView {
            runtime_precision: "fp32".to_owned(),
            runtime_precision_repr: "'fp32'".to_owned(),
            loop_precision: "fp32".to_owned(),
            loop_precision_repr: "'fp32'".to_owned(),
            runtime_device: "cuda".to_owned(),
            output_run_id: None,
            output_run_id_repr: None,
            run_run_id: "wp14-run-001".to_owned(),
            run_run_id_repr: "'wp14-run-001'".to_owned(),
            w_placement: 0.0,
            w_value: 0.0,
            w_event: Vec::new(),
            w_belief: Vec::new(),
            has_privileged_source_hash: false,
        }
    }

    #[test]
    fn accepts_coherent_config() {
        assert!(check_cross_validate(&base()).is_ok());
    }

    /// Hand-derived oracle bytes (`_rc_root.py:105-141` at port time, equals
    /// HEAD): each single-fault input surfaces its own gate text, and the
    /// first gate wins on multi-violation inputs.
    #[test]
    fn gate_text_matches_oracle() {
        let mut faulty = base();
        faulty.loop_precision = "bf16_mixed".to_owned();
        faulty.loop_precision_repr = "'bf16_mixed'".to_owned();
        assert_eq!(
            check_cross_validate(&faulty).unwrap_err(),
            "loop.precision 'bf16_mixed' disagrees with runtime.precision 'fp32' (expected 'fp32'); autocast scope must match",
        );
        let mut faulty = base();
        faulty.runtime_precision = "bf16_mixed".to_owned();
        faulty.runtime_precision_repr = "'bf16_mixed'".to_owned();
        assert_eq!(
            check_cross_validate(&faulty).unwrap_err(),
            "loop.precision 'fp32' disagrees with runtime.precision 'bf16_mixed' (expected 'bf16_mixed'); autocast scope must match",
        );
        let mut faulty = base();
        faulty.run_run_id = "has space".to_owned();
        faulty.run_run_id_repr = "'has space'".to_owned();
        assert_eq!(
            check_cross_validate(&faulty).unwrap_err(),
            "effective run id invalid: 'has space'",
        );
        let mut faulty = base();
        faulty.runtime_precision = "bf16_mixed".to_owned();
        faulty.runtime_precision_repr = "'bf16_mixed'".to_owned();
        faulty.loop_precision = "bf16_mixed".to_owned();
        faulty.loop_precision_repr = "'bf16_mixed'".to_owned();
        faulty.runtime_device = "cpu".to_owned();
        assert_eq!(
            check_cross_validate(&faulty).unwrap_err(),
            "runtime.precision 'bf16_mixed' requires a CUDA device (never silent CPU fallback; device availability itself is checked at bind time via require_device_available)",
        );
        let mut faulty = base();
        faulty.w_placement = 0.5;
        assert_eq!(
            check_cross_validate(&faulty).unwrap_err(),
            "positive placement/value/event/belief weight REQUIRES weights.privileged_source_hash (positive-requires-binding): privileged labels must come from the pinned manifest",
        );
        // First gate wins when several fire at once.
        let mut faulty = base();
        faulty.loop_precision = "bf16_mixed".to_owned();
        faulty.loop_precision_repr = "'bf16_mixed'".to_owned();
        faulty.run_run_id = "has space".to_owned();
        faulty.run_run_id_repr = "'has space'".to_owned();
        assert_eq!(
            check_cross_validate(&faulty).unwrap_err(),
            "loop.precision 'bf16_mixed' disagrees with runtime.precision 'fp32' (expected 'fp32'); autocast scope must match",
        );
    }

    #[test]
    fn output_override_selects_effective_id() {
        // A valid override wins over an invalid `run.run_id`.
        let mut override_wins = base();
        override_wins.run_run_id = "bad id".to_owned();
        override_wins.run_run_id_repr = "'bad id'".to_owned();
        override_wins.output_run_id = Some("out-1".to_owned());
        override_wins.output_run_id_repr = Some("'out-1'".to_owned());
        assert!(check_cross_validate(&override_wins).is_ok());
        // An empty override falls back to `run.run_id` (oracle `len() > 0`).
        let mut empty_falls_back = base();
        empty_falls_back.run_run_id = "bad id".to_owned();
        empty_falls_back.run_run_id_repr = "'bad id'".to_owned();
        empty_falls_back.output_run_id = Some(String::new());
        empty_falls_back.output_run_id_repr = Some("''".to_owned());
        assert_eq!(
            check_cross_validate(&empty_falls_back).unwrap_err(),
            "effective run id invalid: 'bad id'",
        );
        // An invalid override fails even when `run.run_id` is valid.
        let mut bad_override = base();
        bad_override.output_run_id = Some("bad id".to_owned());
        bad_override.output_run_id_repr = Some("'bad id'".to_owned());
        assert_eq!(
            check_cross_validate(&bad_override).unwrap_err(),
            "effective run id invalid: 'bad id'",
        );
    }

    #[test]
    fn weight_binding_covers_event_belief_and_hash() {
        // Positive event/belief entries trigger the same binding gate.
        let mut faulty = base();
        faulty.w_event = vec![0.0, 0.25];
        assert!(
            check_cross_validate(&faulty)
                .unwrap_err()
                .contains("positive-requires-binding")
        );
        let mut faulty = base();
        faulty.w_belief = vec![1.5];
        assert!(
            check_cross_validate(&faulty)
                .unwrap_err()
                .contains("positive-requires-binding")
        );
        // A present pin satisfies the gate for every positive shape.
        let mut pinned = base();
        pinned.w_placement = 0.5;
        pinned.w_value = 1.0;
        pinned.w_event = vec![0.25];
        pinned.w_belief = vec![1.5];
        pinned.has_privileged_source_hash = true;
        assert!(check_cross_validate(&pinned).is_ok());
    }

    #[test]
    fn run_id_shape_restates_frozen_pattern() {
        assert!(is_run_id_shape("wp14-run-001"));
        assert!(is_run_id_shape("a"));
        assert!(is_run_id_shape("A_0-9"));
        assert!(!is_run_id_shape(""));
        assert!(!is_run_id_shape("-lead"));
        assert!(!is_run_id_shape("has space"));
    }
}
