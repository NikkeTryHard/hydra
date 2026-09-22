//! eval_leaves: evaluation-case + resource-telemetry pure leaves on the shared `contracts` submodule.
//!
//! DAG: pyo3 + `hydra-feed` (`canon`, `digest`) ONLY (both already bridge
//! dependencies, see `crates/bridge/Cargo.toml:19,23`; no new dependency).
//! FORBIDS: dataclass construction (every `EvalCase` / `ResourceTelemetry` /
//! `TelemetryTolerance` record stays Python in `python/hydra2/eval/case.py` /
//! `telemetry.py`), live-object orchestration (block aggregation, schedule
//! wiring, and `block_missing_telemetry_report` stay Python), and any new JCS
//! printer — all bytes/hash math lives in `hydra_feed::{canon, digest}`;
//! this file only stages closed-domain values attached, runs the checks and
//! the feed pure functions detached, and returns scalars/text.
//!
//! TABLE ported here (frozen values + pure checks, oracle `python/hydra2/eval/`):
//! - `EVAL_PRIMARY_METRIC` ← `case.py:26` (`PRIMARY_METRIC`).
//! - `eval_check_case_id` ← `make_eval_case` case-id gate (`case.py:50-51`).
//! - `eval_check_arms` ← `make_eval_case` arms gates (`case.py:52-59`).
//! - `eval_check_uncertainty_unit` ← `make_eval_case` unit gates
//!   (`case.py:60-66`; the allowed vocabulary stays owned by
//!   `promotion.UNCERTAINTY_UNITS` and arrives as an argument, never a copy).
//! - `eval_case_manifest_hash_from_docs` ← `case_manifest_hash`
//!   (`case.py:88-90`): the caller projects
//!   `[eval_case_to_json(case) for case in cases]` Python-side (live objects
//!   never cross); this pyfn stages the doc list and seals it through the
//!   feed canon/digest owners.
//! - `TELEMETRY_REQUIRED_CORE_FIELDS` ← `telemetry.py:61-74`.
//! - `TELEMETRY_MODE_REQUIRED_EXTRAS` ← `telemetry.py:77-86`.
//! - `telemetry_tolerance_check` ← `TelemetryTolerance.__post_init__`
//!   (`telemetry.py:187-190`).
//! - `telemetry_required_for` ← `TelemetryTolerance.required_for`
//!   (`telemetry.py:192-197`).
//! - `telemetry_invalid_reason_for` ← `telemetry_invalid_reason`
//!   (`telemetry.py:200-207`): the caller passes the row's scalars (`mode`,
//!   the `None`-valued field names, `invalid_reason`) plus the tolerance's
//!   `allow_missing`, so no live dataclass crosses.
//!
//! LOGIC staying Python: the three dataclasses, `eval_case_to_json`,
//! `make_eval_case` / `make_resource_telemetry` construction (bridge-checked
//! scalars assembled Python-side), the `_require_*` scalar helpers and
//! `TelemetryTolerance`/`ResourceTelemetry` types, `TELEMETRY_FIELDS`
//! (derived from the dataclass via `fields()`, single source by design),
//! `block_missing_telemetry_report` (mapping orchestration over the leaf),
//! and every `ContractError` translator (the bridge raises `ValueError` with
//! byte-identical text; `case.py` maps to `ContractError`. Telemetry raises
//! `TypeError`/`ValueError` natively, so its translators let them propagate).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `walls::register`
//! (`crates/bridge/src/walls.rs:346-364`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `canon_rng::batch_canonical_bytes`
//! (`crates/bridge/src/canon_rng.rs:376-394`); the staging arms mirror
//! `canon_rng::py_to_value` (`crates/bridge/src/canon_rng.rs:275-363`);
//! `PyDict::new` + `set_item` mirror (`crates/bridge/src/contracts.rs:1227-1231`);
//! `PyTuple::new` for frozen tuples mirrors (`crates/bridge/src/contracts.rs:1156`);
//! `Python::initialize()` + `Python::attach` in tests mirror
//! (`crates/bridge/src/canon_rng.rs:688-689`).
//!
//! Feed owners (never reimplemented here): staged-`Value` JCS entry
//! `hydra_feed::canon::canonical_bytes_value` (`crates/feed/src/canon.rs:133`)
//! + the double-safe window `hydra_feed::canon::MAX_SAFE_INTEGER`
//!   (`crates/feed/src/canon.rs:146`) + `hydra_feed::digest::sha256_hex`
//!   (`crates/feed/src/digest.rs:31`, `sha256:`-prefixed lowercase hex,
//!   infallible — called bare below).
//!
//! Oracle-divergence notes (deliberate, garbage-input only):
//! - Lone-surrogate `str` labels fail `String` extraction and are rejected by
//!   the case-id/arms gates, where the oracle would accept the label (it only
//!   checks `isinstance` + emptiness) and fail later at the canon seal. Both
//!   sides fail closed; well-formed UTF-8 callers are unaffected.
//! - Non-`str` `allow_missing` elements fail `Vec<String>` extraction
//!   (`TypeError`), where the oracle's set-intersection silently ignores
//!   them. The declared field type is `frozenset[str]`; well-typed callers
//!   are unaffected.
//! - Unhashable `mode` values (e.g. a `list`) read as "no extras" here, where
//!   the oracle's `MODE_REQUIRED_EXTRAS.get(mode, ())` raises `TypeError`
//!   (unhashable). Every hashable input — all realistic modes — matches.
//! - `diagnostic_only` extracts as a strict `bool`: truthy non-`bool`s (e.g.
//!   `1`) fail extraction (`TypeError`) where the oracle applies truthiness.
//!   The declared type is `bool`; well-typed callers are unaffected.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `validate.rs:111-127`); no new entry
//! point. MAIN wiring: `pub mod eval_leaves;` in `lib.rs` plus
//! `crate::eval_leaves::register(&sub)?;` in `contracts::register` next to
//! `crate::rc_sections::register(&sub)?;` (`contracts.rs:1295`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PyString, PyTuple};

/// Declared primary block outcome contrast (`case.py:26`, SPEC 18.3).
const EVAL_PRIMARY_METRIC: &str = "expected_final_placement_contrast";

/// Fields every telemetry row MUST carry regardless of resource view
/// (`telemetry.py:61-74`).
const TELEMETRY_REQUIRED_CORE_FIELDS: [&str; 12] = [
    "mode",
    "candidate_spec_hash",
    "hardware_hash",
    "environment_hash",
    "cold_start",
    "synchronized_elapsed_ms",
    "model_calls",
    "exact_transitions",
    "particles",
    "fallback_used",
    "timeout",
    "illegal_action",
];

/// Extra fields the `cuda_eager` resource view turns from optional into
/// required (`telemetry.py:78`).
const MODE_EXTRAS_CUDA_EAGER: [&str; 2] = ["cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"];
/// Extra fields the `torch_compile` resource view turns from optional into
/// required (`telemetry.py:79-84`).
const MODE_EXTRAS_TORCH_COMPILE: [&str; 4] = [
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "graph_breaks",
    "recompiles",
];
/// Extra fields the `energy_metered` resource view turns from optional into
/// required (`telemetry.py:85`).
const MODE_EXTRAS_ENERGY_METERED: [&str; 1] = ["energy_joules"];

/// Mode extras lookup (mirrors `MODE_REQUIRED_EXTRAS.get(mode, ())` for
/// hashable modes; unhashable inputs are staged as "no extras", see header).
fn mode_extras(mode: &str) -> &'static [&'static str] {
    if mode == "cuda_eager" {
        &MODE_EXTRAS_CUDA_EAGER
    } else if mode == "torch_compile" {
        &MODE_EXTRAS_TORCH_COMPILE
    } else if mode == "energy_metered" {
        &MODE_EXTRAS_ENERGY_METERED
    } else {
        &[]
    }
}

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Python-`list` rendering of plain field names (`['a', 'b']`, `[]` when
/// empty; mirrors `rc_sections.rs:105-117`). Domain: SPEC field identifiers
/// (no quotes/backslashes), so single-quote joining is exact there.
fn py_str_list(items: &[String]) -> String {
    let mut out = String::from("[");
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(item);
        out.push('\'');
    }
    out.push(']');
    out
}

/// Python-`tuple` rendering (`()`, `('a',)`, `('a', 'b')`). Same closed
/// identifier domain as [`py_str_list`], so the trailing-comma single form
/// is exact there.
fn py_str_tuple(items: &[String]) -> String {
    if items.is_empty() {
        return String::from("()");
    }
    if items.len() == 1 {
        return format!("('{}',)", items[0]);
    }
    let mut out = String::from("(");
    for (index, item) in items.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(item);
        out.push('\'');
    }
    out.push(')');
    out
}

// ---------------------------------------------------------------------------
// Detached checks (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// Case-id gate (`case.py:50-51`): `None` stages every non-`str` input;
/// empty text rejects here too so the helper is total on its own.
fn check_case_id(staged: &Option<String>) -> Result<String, String> {
    match staged {
        Some(text) if !text.is_empty() => Ok(text.clone()),
        _ => Err(String::from("case_id must be a nonempty str")),
    }
}

/// Arms gates (`case.py:52-59`): `None` stages every non-tuple, wrong-length,
/// or empty/non-`str` element shape; equal labels fail distinctly.
fn check_arms(staged: &Option<(String, String)>) -> Result<(String, String), String> {
    match staged {
        None => Err(String::from("arms must be two nonempty opaque labels")),
        Some((first, second)) if first == second => Err(String::from("arms must be distinct")),
        Some((first, second)) => Ok((first.clone(), second.clone())),
    }
}

/// Uncertainty-unit gates (`case.py:60-66`): unknown units render with the
/// staged live `repr`s (byte-exact for every input); `game_cluster` needs
/// the diagnostic flag. The allowed vocabulary arrives from
/// `promotion.UNCERTAINTY_UNITS` (never copied here).
fn check_uncertainty_unit(
    staged: &Option<String>,
    unit_repr: &str,
    diagnostic_only: bool,
    allowed: &[String],
) -> Result<String, String> {
    let name = match staged {
        Some(text) if allowed.iter().any(|entry| entry == text) => text,
        _ => {
            return Err(format!(
                "uncertainty_unit {unit_repr} not in {}",
                py_str_tuple(allowed)
            ));
        }
    };
    if name == "game_cluster" && !diagnostic_only {
        return Err(String::from(
            "uncertainty_unit 'game_cluster' is reserved for held-out \
             model/calibration diagnostics; pass diagnostic_only=True",
        ));
    }
    Ok(name.clone())
}

/// Tolerance-construction gate (`telemetry.py:187-190`): `allow_missing` may
/// excuse only genuinely optional fields. Sorted + deduped to match the
/// oracle's `sorted(set & ...)` rendering.
fn check_tolerance(allow_missing: &[String]) -> Result<(), String> {
    let mut forbidden: Vec<String> = allow_missing
        .iter()
        .filter(|name| {
            TELEMETRY_REQUIRED_CORE_FIELDS.contains(&name.as_str())
                || name.as_str() == "invalid_reason"
        })
        .cloned()
        .collect();
    forbidden.sort();
    forbidden.dedup();
    if forbidden.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "tolerance cannot excuse required fields: {}",
            py_str_list(&forbidden)
        ))
    }
}

/// Mode-required closure (`telemetry.py:192-197`): core plus the mode extras;
/// excusing a mode-required field fails (tuple rendering matches the
/// oracle's `{tolerated_extras}` interpolation, extras order preserved).
fn required_for(mode: Option<&str>, allow_missing: &[String]) -> Result<Vec<String>, String> {
    let extras: &[&str] = mode.map_or(&[], mode_extras);
    let tolerated: Vec<String> = extras
        .iter()
        .filter(|extra| allow_missing.iter().any(|name| name == **extra))
        .map(|extra| (*extra).to_string())
        .collect();
    if !tolerated.is_empty() {
        return Err(format!(
            "tolerance cannot excuse mode-required fields: {}",
            py_str_tuple(&tolerated)
        ));
    }
    let mut required: Vec<String> = TELEMETRY_REQUIRED_CORE_FIELDS
        .iter()
        .map(|field| (*field).to_string())
        .collect();
    required.extend(extras.iter().map(|field| (*field).to_string()));
    Ok(required)
}

/// Row-invalidity policy (`telemetry.py:200-207`): caller-marked rows name
/// their reason; otherwise every `None`-valued required field is named in
/// required order (never imputed). `null_fields` carries the row's
/// `None`-valued names; non-required `None`s are ignored here.
fn invalid_reason_for(
    mode: Option<&str>,
    allow_missing: &[String],
    null_fields: &[String],
    invalid_reason: &Option<String>,
) -> Result<Option<String>, String> {
    if let Some(reason) = invalid_reason {
        return Ok(Some(format!("row marked invalid: {reason}")));
    }
    let required = required_for(mode, allow_missing)?;
    let missing: Vec<String> = required
        .into_iter()
        .filter(|name| null_fields.iter().any(|null| null == name))
        .collect();
    if missing.is_empty() {
        Ok(None)
    } else {
        Ok(Some(format!(
            "missing required telemetry (never imputed): {}",
            py_str_list(&missing)
        )))
    }
}

/// Stage one closed-domain Python object to a serde `Value` (attached only:
/// touches Python memory, never detached). Mirrors
/// `rc_digest::py_to_value` arm-for-arm (`None` -> null; `bool` BEFORE `int`
/// since Python `bool` subclasses `int`; `int` outside the double-safe
/// window rejected; finite `float` only; `str`; `list` element-wise, where
/// `tuple` is NOT a list and is rejected; `dict` with non-`str` keys
/// rejected; anything else rejected). `record` names the digest call.
fn py_to_value(obj: &Bound<'_, PyAny>, record: &str) -> Result<serde_json::Value, String> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        let flag = obj
            .extract::<bool>()
            .map_err(|e| format!("{record}: bool unreadable: {e}"))?;
        return Ok(serde_json::Value::Bool(flag));
    }
    if obj.is_instance_of::<PyInt>() {
        match obj.extract::<i64>() {
            Ok(n) => {
                if !(-hydra_feed::canon::MAX_SAFE_INTEGER..=hydra_feed::canon::MAX_SAFE_INTEGER)
                    .contains(&n)
                {
                    return Err(format!(
                        "{record}: integer {n} exceeds the IEEE 754 double-safe range; serialize it as a float or string"
                    ));
                }
                return Ok(serde_json::Value::Number(n.into()));
            }
            Err(_) => {
                return Err(format!(
                    "{record}: integer exceeds the IEEE 754 double-safe range; serialize it as a float or string"
                ));
            }
        }
    }
    if obj.is_instance_of::<PyFloat>() {
        let value = obj
            .extract::<f64>()
            .map_err(|e| format!("{record}: float unreadable: {e}"))?;
        if !value.is_finite() {
            return Err(format!(
                "{record}: non-finite number {value:?} has no canonical serialization"
            ));
        }
        match serde_json::Number::from_f64(value) {
            Some(number) => return Ok(serde_json::Value::Number(number)),
            None => {
                return Err(format!(
                    "{record}: non-finite number has no canonical serialization"
                ));
            }
        }
    }
    if obj.is_instance_of::<PyString>() {
        match obj.extract::<String>() {
            Ok(text) => return Ok(serde_json::Value::String(text)),
            Err(e) => {
                return Err(format!(
                    "{record}: string contains an unpaired surrogate (invalid Unicode): {e}"
                ));
            }
        }
    }
    if obj.is_instance_of::<PyList>() {
        let elements: Vec<Bound<'_, PyAny>> = obj
            .extract()
            .map_err(|e| format!("{record}: list unreadable: {e}"))?;
        let mut array = Vec::with_capacity(elements.len());
        for item in &elements {
            array.push(py_to_value(item, record)?);
        }
        return Ok(serde_json::Value::Array(array));
    }
    if obj.is_instance_of::<PyDict>() {
        let dict: Bound<'_, PyDict> = obj
            .extract()
            .map_err(|e| format!("{record}: dict unreadable: {e}"))?;
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            if !key.is_instance_of::<PyString>() {
                return Err(format!(
                    "{record}: object key is not a string; JSON objects are string-keyed only"
                ));
            }
            let name: String = key.extract().map_err(|e| {
                format!(
                    "{record}: object key contains an unpaired surrogate (invalid Unicode): {e}"
                )
            })?;
            map.insert(name, py_to_value(&value, record)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    Err(format!(
        "{record}: value is outside the canonical JSON domain (null, bool, string, finite number, array, string-keyed object only)"
    ))
}

/// Pure digest over a staged value: JCS-emit through the feed canon owner,
/// then `sha256:`-hash through the feed digest owner. Byte-identical to
/// `artifacts/digest.py::of_canonical` of the same value by construction
/// (same `canonical_bytes_value` entry `canon_rng::batch_canonical_bytes`
/// uses, see `canon_rng.rs:388`).
fn case_manifest_digest_value(value: &serde_json::Value) -> Result<String, String> {
    let record = "bridge:eval_case_manifest_hash_from_docs";
    let bytes = hydra_feed::canon::canonical_bytes_value(value, record)
        .map_err(|e| format!("canon JCS emit rejected: {e:?}"))?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

// ---------------------------------------------------------------------------
// Pyfns (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// Case-id gate (`case.py:50-51`): nonempty `str`, staged attached (a failed
/// `String` extraction stages `None`, covering non-`str`, empty, and
/// lone-surrogate inputs); the accept/reject decision runs detached.
#[pyfunction]
fn eval_check_case_id(py: Python<'_>, case_id: Bound<'_, PyAny>) -> PyResult<String> {
    let staged: Option<String> = case_id
        .extract()
        .ok()
        .filter(|text: &String| !text.is_empty());
    py.detach(|| check_case_id(&staged))
        .map_err(PyValueError::new_err)
}

/// Arms gates (`case.py:52-59`): a `tuple` of two nonempty opaque labels,
/// distinct. Non-tuples (including 2-lists), wrong lengths, and
/// empty/non-`str` elements stage `None` and share the shape message;
/// equality fails distinctly. Decision runs detached.
#[pyfunction]
fn eval_check_arms(py: Python<'_>, arms: Bound<'_, PyAny>) -> PyResult<(String, String)> {
    let staged: Option<(String, String)> = if arms.is_instance_of::<PyTuple>() {
        match arms.extract::<Bound<'_, PyTuple>>() {
            Ok(tuple) if tuple.len() == 2 => match (tuple.get_item(0), tuple.get_item(1)) {
                (Ok(first), Ok(second)) => {
                    match (
                        first
                            .extract::<String>()
                            .ok()
                            .filter(|s: &String| !s.is_empty()),
                        second
                            .extract::<String>()
                            .ok()
                            .filter(|s: &String| !s.is_empty()),
                    ) {
                        (Some(a), Some(b)) => Some((a, b)),
                        _ => None,
                    }
                }
                _ => None,
            },
            _ => None,
        }
    } else {
        None
    };
    py.detach(|| check_arms(&staged))
        .map_err(PyValueError::new_err)
}

/// Uncertainty-unit gates (`case.py:60-66`): membership in the caller-owned
/// `allowed` vocabulary (from `promotion.UNCERTAINTY_UNITS`), then the
/// `game_cluster`-needs-diagnostic gate. The live `repr(unit)` is staged
/// attached so `{unit!r}` renders byte-exact for every input; the decision
/// runs detached.
#[pyfunction]
fn eval_check_uncertainty_unit(
    py: Python<'_>,
    unit: Bound<'_, PyAny>,
    diagnostic_only: bool,
    allowed: Vec<String>,
) -> PyResult<String> {
    let unit_repr = py_repr(&unit)?;
    let staged: Option<String> = unit.extract().ok();
    py.detach(|| check_uncertainty_unit(&staged, &unit_repr, diagnostic_only, &allowed))
        .map_err(PyValueError::new_err)
}

/// Case-manifest seal (`case.py:88-90`) over caller-projected JSON docs: the
/// caller maps `[eval_case_to_json(case) for case in cases]` (live `EvalCase`
/// objects never cross); this pyfn stages the doc list attached, then runs
/// ONE detach around JCS-emit + hash through the feed owners and returns
/// the `sha256:<hex>` text.
#[pyfunction]
fn eval_case_manifest_hash_from_docs(py: Python<'_>, docs: Bound<'_, PyAny>) -> PyResult<String> {
    let staged = py_to_value(&docs, "bridge:eval_case_manifest_hash_from_docs")
        .map_err(PyValueError::new_err)?;
    py.detach(|| case_manifest_digest_value(&staged))
        .map_err(PyValueError::new_err)
}

/// Tolerance-construction gate (`telemetry.py:187-190`): fails when
/// `allow_missing` excuses a required-core field or `invalid_reason`.
/// The oracle raises `ValueError`; the bridge raises `ValueError` with the
/// identical text (telemetry translators let it propagate, never remap to
/// `ContractError`).
#[pyfunction]
fn telemetry_tolerance_check(py: Python<'_>, allow_missing: Vec<String>) -> PyResult<()> {
    py.detach(|| check_tolerance(&allow_missing))
        .map_err(PyValueError::new_err)
}

/// Mode-required closure (`telemetry.py:192-197`): core plus the mode extras
/// (`unknown modes carry no extras`, mirroring `.get(mode, ())` for every
/// hashable input). Non-`str` modes stage `None` (same "no extras" outcome);
/// excusing a mode-required field fails with the oracle text.
#[pyfunction]
fn telemetry_required_for(
    py: Python<'_>,
    mode: Bound<'_, PyAny>,
    allow_missing: Vec<String>,
) -> PyResult<Vec<String>> {
    let staged: Option<String> = mode.extract().ok();
    py.detach(|| required_for(staged.as_deref(), &allow_missing))
        .map_err(PyValueError::new_err)
}

/// Row-invalidity policy (`telemetry.py:200-207`) over caller-passed
/// scalars: `mode`, the tolerance's `allow_missing`, the row's `None`-valued
/// field names, and the row's `invalid_reason`. Returns the exclusion reason
/// or `None` when the row is usable. Compute runs detached.
#[pyfunction]
#[pyo3(signature = (mode, allow_missing, null_fields, invalid_reason=None))]
fn telemetry_invalid_reason_for(
    py: Python<'_>,
    mode: Bound<'_, PyAny>,
    allow_missing: Vec<String>,
    null_fields: Vec<String>,
    invalid_reason: Option<String>,
) -> PyResult<Option<String>> {
    let staged_mode: Option<String> = mode.extract().ok();
    py.detach(|| {
        invalid_reason_for(
            staged_mode.as_deref(),
            &allow_missing,
            &null_fields,
            &invalid_reason,
        )
    })
    .map_err(PyValueError::new_err)
}

/// Register the eval-case + telemetry leaves on the shared `contracts`
/// submodule (mirrors `validate.rs:111-127`); MAIN calls this from
/// `contracts::register` — no new submodule, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("EVAL_PRIMARY_METRIC", EVAL_PRIMARY_METRIC)?;
    sub.add(
        "TELEMETRY_REQUIRED_CORE_FIELDS",
        PyTuple::new(py, TELEMETRY_REQUIRED_CORE_FIELDS)?,
    )?;
    let extras = PyDict::new(py);
    extras.set_item("cuda_eager", PyTuple::new(py, MODE_EXTRAS_CUDA_EAGER)?)?;
    extras.set_item(
        "torch_compile",
        PyTuple::new(py, MODE_EXTRAS_TORCH_COMPILE)?,
    )?;
    extras.set_item(
        "energy_metered",
        PyTuple::new(py, MODE_EXTRAS_ENERGY_METERED)?,
    )?;
    sub.add("TELEMETRY_MODE_REQUIRED_EXTRAS", extras)?;
    sub.add_function(wrap_pyfunction!(eval_check_case_id, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_check_arms, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_check_uncertainty_unit, sub)?)?;
    sub.add_function(wrap_pyfunction!(eval_case_manifest_hash_from_docs, sub)?)?;
    sub.add_function(wrap_pyfunction!(telemetry_tolerance_check, sub)?)?;
    sub.add_function(wrap_pyfunction!(telemetry_required_for, sub)?)?;
    sub.add_function(wrap_pyfunction!(telemetry_invalid_reason_for, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_consts_match_oracle() {
        Python::initialize();
        // Oracle values hand-traced from `case.py:26` + `telemetry.py:61-86`.
        assert_eq!(EVAL_PRIMARY_METRIC, "expected_final_placement_contrast");
        assert_eq!(
            TELEMETRY_REQUIRED_CORE_FIELDS,
            [
                "mode",
                "candidate_spec_hash",
                "hardware_hash",
                "environment_hash",
                "cold_start",
                "synchronized_elapsed_ms",
                "model_calls",
                "exact_transitions",
                "particles",
                "fallback_used",
                "timeout",
                "illegal_action",
            ]
        );
        assert_eq!(
            MODE_EXTRAS_CUDA_EAGER,
            ["cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"]
        );
        assert_eq!(
            MODE_EXTRAS_TORCH_COMPILE,
            [
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
                "graph_breaks",
                "recompiles",
            ]
        );
        assert_eq!(MODE_EXTRAS_ENERGY_METERED, ["energy_joules"]);
        assert!(!mode_extras("cuda_eager").contains(&"graph_breaks"));
        assert_eq!(mode_extras("unknown_mode").len(), 0);
    }

    #[test]
    fn case_gates_match_oracle() {
        Python::initialize();
        // `case.py:50-51`: nonempty str only.
        assert_eq!(
            check_case_id(&Some(String::from("wp03b"))),
            Ok(String::from("wp03b"))
        );
        assert_eq!(
            check_case_id(&None),
            Err(String::from("case_id must be a nonempty str"))
        );
        assert_eq!(
            check_case_id(&Some(String::new())),
            Err(String::from("case_id must be a nonempty str"))
        );
        // `case.py:52-59`: two nonempty labels, distinct.
        assert_eq!(
            check_arms(&Some((String::from("a"), String::from("b")))),
            Ok((String::from("a"), String::from("b")))
        );
        assert_eq!(
            check_arms(&None),
            Err(String::from("arms must be two nonempty opaque labels"))
        );
        assert_eq!(
            check_arms(&Some((String::from("a"), String::from("a")))),
            Err(String::from("arms must be distinct"))
        );
        // `case.py:60-66` with the promotion vocabulary (`promotion.py:35-42`).
        let allowed: Vec<String> = [
            "case",
            "iid_pair",
            "wall_block",
            "smc_population",
            "rqmc_scramble",
            "game_cluster",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        assert_eq!(
            check_uncertainty_unit(&Some(String::from("case")), "'case'", false, &allowed),
            Ok(String::from("case"))
        );
        assert_eq!(
            check_uncertainty_unit(
                &Some(String::from("game_cluster")),
                "'game_cluster'",
                true,
                &allowed
            ),
            Ok(String::from("game_cluster"))
        );
        assert_eq!(
            check_uncertainty_unit(
                &Some(String::from("game_cluster")),
                "'game_cluster'",
                false,
                &allowed
            ),
            Err(String::from(
                "uncertainty_unit 'game_cluster' is reserved for held-out \
                 model/calibration diagnostics; pass diagnostic_only=True"
            ))
        );
        assert_eq!(
            check_uncertainty_unit(&Some(String::from("nope")), "'nope'", false, &allowed),
            Err(format!(
                "uncertainty_unit 'nope' not in {}",
                py_str_tuple(&allowed)
            ))
        );
        assert_eq!(
            check_uncertainty_unit(&None, "None", false, &allowed),
            Err(format!(
                "uncertainty_unit None not in {}",
                py_str_tuple(&allowed)
            ))
        );
    }

    #[test]
    fn telemetry_policy_matches_oracle() {
        Python::initialize();
        // `telemetry.py:187-190`: only optional fields are excusable.
        assert_eq!(check_tolerance(&[]), Ok(()));
        assert_eq!(check_tolerance(&[String::from("energy_joules")]), Ok(()));
        assert_eq!(
            check_tolerance(&[String::from("mode")]),
            Err(String::from(
                "tolerance cannot excuse required fields: ['mode']"
            ))
        );
        assert_eq!(
            check_tolerance(&[String::from("invalid_reason")]),
            Err(String::from(
                "tolerance cannot excuse required fields: ['invalid_reason']"
            ))
        );
        // `telemetry.py:192-197`: mode extras join the required closure.
        let required = required_for(Some("cuda_eager"), &[]).expect("valid tolerance");
        assert_eq!(required.len(), 14);
        assert_eq!(
            required[12..],
            [
                String::from("cuda_peak_allocated_bytes"),
                String::from("cuda_peak_reserved_bytes")
            ]
        );
        let plain = required_for(Some("plain"), &[]).expect("unknown mode carries no extras");
        assert_eq!(plain.len(), 12);
        assert_eq!(
            required_for(
                Some("cuda_eager"),
                &[String::from("cuda_peak_allocated_bytes")]
            ),
            Err(String::from(
                "tolerance cannot excuse mode-required fields: ('cuda_peak_allocated_bytes',)"
            ))
        );
        // `telemetry.py:200-207`: marked rows name their reason; gaps list
        // missing required fields in required order; usable rows yield None.
        assert_eq!(
            invalid_reason_for(Some("plain"), &[], &[], &Some(String::from("bad row"))),
            Ok(Some(String::from("row marked invalid: bad row")))
        );
        assert_eq!(invalid_reason_for(Some("plain"), &[], &[], &None), Ok(None));
        assert_eq!(
            invalid_reason_for(
                Some("energy_metered"),
                &[],
                &[String::from("energy_joules")],
                &None
            ),
            Ok(Some(String::from(
                "missing required telemetry (never imputed): ['energy_joules']"
            )))
        );
    }

    #[test]
    fn pyfn_wrappers_agree_with_detached_checks() {
        Python::initialize();
        Python::attach(|py| {
            let ok = eval_check_case_id(py, PyString::new(py, "wp03b").into_any())
                .expect("valid case id");
            assert_eq!(ok, "wp03b");
            assert!(eval_check_case_id(py, PyString::new(py, "").into_any()).is_err());
            let pair = eval_check_arms(
                py,
                PyTuple::new(py, [PyString::new(py, "a"), PyString::new(py, "b")])
                    .expect("arms tuple")
                    .into_any(),
            )
            .expect("valid arms");
            assert_eq!(pair, (String::from("a"), String::from("b")));
            assert!(telemetry_tolerance_check(py, vec![String::from("energy_joules")]).is_ok());
            assert!(telemetry_tolerance_check(py, vec![String::from("mode")]).is_err());
            let required =
                telemetry_required_for(py, PyString::new(py, "cuda_eager").into_any(), vec![])
                    .expect("valid mode");
            assert_eq!(required.len(), 14);
            let reason = telemetry_invalid_reason_for(
                py,
                PyString::new(py, "plain").into_any(),
                vec![],
                vec![],
                None,
            )
            .expect("usable row");
            assert_eq!(reason, None);
        });
    }
}
