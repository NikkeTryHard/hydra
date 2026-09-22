//! rc_require: run-config require-gate validators on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (no feed/shard/search edge; pure argument validation, so no
//! crate owner exists — see the NONE note below; `crates/bridge/Cargo.toml:25`
//! carries the only dependency, no new dependency). FORBIDS: YAML/file IO,
//! `${VAR}` interpolation, base+override deep-merge composition (all stay
//! Python in `python/hydra2/training/_rc_require.py`), wall-clock/RNG, and
//! `ContractError` shaping (the bridge raises `ValueError`/`TypeError`; thin
//! Python translators map to `ContractError` with byte-identical messages).
//!
//! TABLE ported here — the 11 pure gates, oracle
//! `python/hydra2/training/_rc_require.py:90-228` (HEAD md5 `c6d2f233…`, equals
//! worktree at port time):
//! - `rc_reject_unknown` ← `_reject_unknown` (:90-96)
//! - `rc_require_nonempty_str` ← `_require_nonempty_str` (:99-103)
//! - `rc_require_positive_int` ← `_require_positive_int` (:106-110)
//! - `rc_require_bounded_int` ← `_require_bounded_int` (:113-118)
//! - `rc_require_nonnegative_int` ← `_require_nonnegative_int` (:121-125)
//! - `rc_require_nonnegative_float` ← `_require_nonnegative_float` (:128-135)
//! - `rc_weight_map` ← `_weight_map` (:138-157)
//! - `rc_require_bounded_float` ← `_require_bounded_float` (:160-182)
//! - `rc_positive_float_map` ← `_positive_float_map` (:185-205)
//! - `rc_require_loop_bool` ← `_require_loop_bool` (:208-215; literal `loop.`
//!   prefix kept, there is no `where` param)
//! - `rc_digest_pin_or_none` ← `_digest_pin_or_none` (:221-228)
//!
//! LOGIC staying Python: `_interpolate_string`/`_interpolate_tree` (regex plus
//! environ mapping), `deep_merge` (recursive `Any` composition), `_DIGEST_RE`
//! (compiled-pattern literal), the `INTERPOLATION_ALLOWLIST` re-export, and
//! all 11 `ContractError` translators (identical messages, `__all__` unchanged).
//!
//! NONE-owner note (wave-3 lesson 2): `rg` for
//! `reject_unknown|positive_int|bounded_float|weight_map|loop_bool|digest_pin|
//! INTERPOLAT|ALLOWLIST` over `crates/feed/src` + `crates/search/src` hits only
//! `sha256:`-shape comments and the unrelated arena budget (`search.rs:97`
//! `MAX_DEADLINE_MS`); the lowercase digest-shape test restates
//! `contracts.rs:99-107` `is_digest_shape` (private there) with the oracle
//! line cited per use.
//!
//! Shape per fn: attached staging (repr/type-name/strip via the live Python
//! API, so `{value!r}` and `type(x).__name__` render byte-exact) → ONE
//! `py.detach(|| …)` over owned plain data with zero Python API inside (per
//! `contracts.rs:434-437`, `validate.rs:83-95`) → attached wrap as
//! `PyValueError` (per `contracts.rs:117-123`). Bool is excluded before every
//! int check (per `contracts.rs:69-74`, `canon_rng.rs:279-286`); a missing
//! `get_item` slot reads as `None` exactly like the oracle's `raw.get(key)`
//! (per `packet_decode.rs:375-378`); result dicts are built with `PyDict::new`
//! plus `set_item` (per `contracts.rs:1227-1231`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `validate.rs:111-127`); no new entry
//! point. MAIN wiring: `pub mod rc_require;` in `lib.rs` plus
//! `crate::rc_require::register(&sub)?;` in `contracts.rs` next to
//! `crate::validate::register(&sub)?;` (`contracts.rs:1291`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyModule, PyString};

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot. GIL-held by construction; `extract::<String>()` is the
/// `canon_rng.rs:322-323` half of the same round-trip.
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Attached staging helper: exact `type(value).__name__` text for the two
/// `must be a mapping` slots (`_reject_unknown`, `_weight_map`,
/// `_positive_float_map`).
fn py_type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.get_type().name()?.to_str()?.to_owned())
}

/// Owned plain-data view of an int-or-other Python value. Bool is excluded
/// first (it subclasses `int`), exactly like `contracts.rs:69-74`; big ints
/// past `i64` keep `is_int` true with `as_i64` unset, and their sign comes
/// from the staged repr (`-`-leading iff negative — no other int repr leads
/// with `-`), so no rich comparison crosses into the detach.
struct IntView {
    repr: String,
    is_int: bool,
    as_i64: Option<i64>,
    negative: bool,
}

fn stage_int(obj: &Bound<'_, PyAny>) -> PyResult<IntView> {
    let repr = py_repr(obj)?;
    if obj.is_instance_of::<PyBool>() || !obj.is_instance_of::<PyInt>() {
        return Ok(IntView {
            repr,
            is_int: false,
            as_i64: None,
            negative: false,
        });
    }
    let as_i64 = obj.extract::<i64>().ok();
    let negative = as_i64.map_or_else(|| repr.starts_with('-'), |v| v < 0);
    Ok(IntView {
        repr,
        is_int: true,
        as_i64,
        negative,
    })
}

/// Missing slot under the oracle's `raw.get(key)`: always rejects, rendering
/// as `None` (`{None!r}`).
fn missing_int() -> IntView {
    IntView {
        repr: "None".to_owned(),
        is_int: false,
        as_i64: None,
        negative: false,
    }
}

/// Owned plain-data view of a number-or-other value. `number` is `Some` iff
/// the value is a non-bool int/float. A huge int that overflows `f64` keeps
/// the original (Overflow) error at stage time — exactly where the oracle's
/// `float(value)` raises, never a `ContractError` shape.
struct NumView {
    repr: String,
    number: Option<f64>,
}

fn stage_num(obj: &Bound<'_, PyAny>) -> PyResult<NumView> {
    let repr = py_repr(obj)?;
    if obj.is_instance_of::<PyBool>() {
        return Ok(NumView { repr, number: None });
    }
    if !(obj.is_instance_of::<PyInt>() || obj.is_instance_of::<PyFloat>()) {
        return Ok(NumView { repr, number: None });
    }
    Ok(NumView {
        repr,
        number: Some(obj.extract::<f64>()?),
    })
}

fn missing_num() -> NumView {
    NumView {
        repr: "None".to_owned(),
        number: None,
    }
}

/// One staged bound of `_require_bounded_float`: value plus the exact float
/// repr (Python `str`/`repr` of a float coincide, so `{lo}`/`{hi}` render
/// byte-exact from these).
struct FloatBound {
    value: f64,
    repr: String,
    open: bool,
}

fn stage_bound(obj: &Bound<'_, PyAny>) -> PyResult<(f64, String)> {
    let repr = py_repr(obj)?;
    let value = obj.extract::<f64>().map_err(|_| {
        PyValueError::new_err(format!(
            "contracts rc_require_bounded_float bound {repr} must be a number"
        ))
    })?;
    Ok((value, repr))
}

/// One staged entry of the `{name: mult}` maps. `number` is `Some` iff the
/// weight is a non-bool int/float (overflow propagates at stage time, same
/// order the oracle evaluates `float(weight)` in).
struct MapEntry {
    key: String,
    key_repr: String,
    key_ok: bool,
    number: Option<f64>,
}

fn stage_map_number(obj: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(None);
    }
    if !(obj.is_instance_of::<PyInt>() || obj.is_instance_of::<PyFloat>()) {
        return Ok(None);
    }
    obj.extract::<f64>().map(Some)
}

/// Digest-shape test for `_digest_pin_or_none` (oracle `_rc_require.py:218`
/// `_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")` via `fullmatch`):
/// lowercase-only, restating `contracts.rs:99-107` (private there).
fn is_digest_shape(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 7 + 64 || &bytes[..7] != b"sha256:" {
        return false;
    }
    bytes[7..]
        .iter()
        .all(|c| c.is_ascii_digit() || matches!(c, b'a'..=b'f'))
}

// ---------------------------------------------------------------------------
// Detached checks (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

fn check_positive_int(where_: &str, key: &str, view: &IntView) -> Result<(), String> {
    let ok = view.is_int
        && match view.as_i64 {
            Some(v) => v > 0,
            None => !view.negative,
        };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "{where_}.{key} must be a positive int, got {}",
            view.repr
        ))
    }
}

fn check_bounded_int(
    where_: &str,
    key: &str,
    view: &IntView,
    lo: i64,
    hi: i64,
) -> Result<(), String> {
    let ok = view.is_int
        && match view.as_i64 {
            Some(v) => lo <= v && v <= hi,
            None => false,
        };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "{where_}.{key} must be an int in [{lo}, {hi}], got {}",
            view.repr
        ))
    }
}

fn check_nonnegative_int(where_: &str, key: &str, view: &IntView) -> Result<(), String> {
    let ok = view.is_int
        && match view.as_i64 {
            Some(v) => v >= 0,
            None => !view.negative,
        };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "{where_}.{key} must be a non-negative int, got {}",
            view.repr
        ))
    }
}

fn check_nonnegative_float(where_: &str, key: &str, view: &NumView) -> Result<f64, String> {
    let Some(number) = view.number else {
        return Err(format!(
            "{where_}.{key} must be a non-negative number, got {}",
            view.repr
        ));
    };
    if !number.is_finite() || number < 0.0 {
        return Err(format!(
            "{where_}.{key} must be finite and non-negative, got {}",
            view.repr
        ));
    }
    Ok(number)
}

fn check_bounded_float(
    where_: &str,
    key: &str,
    view: &NumView,
    lo: &FloatBound,
    hi: &FloatBound,
) -> Result<f64, String> {
    let Some(number) = view.number else {
        return Err(format!(
            "{where_}.{key} must be a number, got {}",
            view.repr
        ));
    };
    if !number.is_finite() {
        return Err(format!("{where_}.{key} must be finite, got {}", view.repr));
    }
    let lo_ok = if lo.open {
        lo.value < number
    } else {
        lo.value <= number
    };
    let hi_ok = if hi.open {
        number < hi.value
    } else {
        number <= hi.value
    };
    if !(lo_ok && hi_ok) {
        let bound = format!(
            "{}{}, {}{}",
            if lo.open { '(' } else { '[' },
            lo.repr,
            hi.repr,
            if hi.open { ')' } else { ']' }
        );
        return Err(format!(
            "{where_}.{key} must lie in {bound}, got {}",
            view.repr
        ));
    }
    Ok(number)
}

fn check_weight_map(
    where_: &str,
    key: &str,
    entries: &[MapEntry],
) -> Result<Vec<(String, f64)>, String> {
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        if !entry.key_ok || entry.key.is_empty() {
            return Err(format!("{where_}.{key} keys must be non-empty strings"));
        }
        let Some(number) = entry.number else {
            return Err(format!(
                "{where_}.{key}[{}] must be finite non-negative",
                entry.key_repr
            ));
        };
        if !number.is_finite() || number < 0.0 {
            return Err(format!(
                "{where_}.{key}[{}] must be finite non-negative",
                entry.key_repr
            ));
        }
        out.push((entry.key.clone(), number));
    }
    Ok(out)
}

fn check_positive_float_map(
    where_: &str,
    key: &str,
    entries: &[MapEntry],
) -> Result<Vec<(String, f64)>, String> {
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        if !entry.key_ok || entry.key.is_empty() {
            return Err(format!("{where_}.{key} keys must be non-empty strings"));
        }
        let Some(number) = entry.number else {
            return Err(format!(
                "{where_}.{key}[{}] must be positive and finite",
                entry.key_repr
            ));
        };
        if !number.is_finite() || number <= 0.0 {
            return Err(format!(
                "{where_}.{key}[{}] must be positive and finite",
                entry.key_repr
            ));
        }
        out.push((entry.key.clone(), number));
    }
    Ok(out)
}

/// Staged keys plus allowlist for `_reject_unknown`. `allowed` arrives as
/// plain strings (section/field identifiers — single-quote rendering below is
/// exact on that domain); unknown keys keep their staged Python reprs, so the
/// `[...]` list renders byte-exact.
struct UnknownView {
    keys: Vec<(String, Option<String>)>,
    allowed: Vec<String>,
    where_: String,
}

fn check_unknown(view: &UnknownView) -> Result<(), String> {
    let mut allowed_sorted = view.allowed.clone();
    allowed_sorted.sort();
    let allowed_list = format!(
        "[{}]",
        allowed_sorted
            .iter()
            .map(|s| format!("'{s}'"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut unknown: Vec<&(String, Option<String>)> = view
        .keys
        .iter()
        .filter(|(_, as_str)| as_str.as_ref().is_none_or(|s| !view.allowed.contains(s)))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    if unknown.len() > 1 && unknown.iter().any(|(_, as_str)| as_str.is_none()) {
        // The oracle's `sorted()` raises TypeError over mixed incomparable
        // keys; fail closed on the same shape (translators map ValueError and
        // TypeError alike to ContractError).
        return Err(format!(
            "{} has unorderable unknown keys; allowed={allowed_list}",
            view.where_
        ));
    }
    unknown.sort_by(|a, b| a.1.cmp(&b.1));
    let unknown_list = format!(
        "[{}]",
        unknown
            .iter()
            .map(|(repr, _)| repr.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Err(format!(
        "{} has unknown keys {unknown_list}; allowed={allowed_list}",
        view.where_
    ))
}

// ---------------------------------------------------------------------------
// Pyfns (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// `_reject_unknown` gate (oracle `_rc_require.py:90-96`): the mapping copy
/// is returned on success; non-mappings and unknown keys fail closed.
#[pyfunction]
fn rc_reject_unknown(
    py: Python<'_>,
    raw: Bound<'_, PyAny>,
    allowed: Vec<String>,
    where_: String,
) -> PyResult<Py<PyDict>> {
    let dict: Bound<'_, PyDict> = match raw.extract() {
        Ok(dict) => dict,
        Err(_) => {
            let type_name = py_type_name(&raw).unwrap_or_else(|_| "?".to_owned());
            return Err(PyValueError::new_err(format!(
                "{where_} must be a mapping, got {type_name}"
            )));
        }
    };
    let mut keys = Vec::with_capacity(dict.len());
    for (head, _value) in dict.iter() {
        let head_repr = py_repr(&head)?;
        keys.push((head_repr, head.extract::<String>().ok()));
    }
    let view = UnknownView {
        keys,
        allowed,
        where_,
    };
    py.detach(|| check_unknown(&view))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    for (head, value) in dict.iter() {
        out.set_item(head, value)?;
    }
    Ok(out.unbind())
}

/// `_require_nonempty_str` gate (oracle `_rc_require.py:99-103`): blank is
/// decided by live `strip()` at stage time (Unicode-exact, no Rust-`trim`
/// surrogate), the accept/reject decision runs detached.
#[pyfunction]
fn rc_require_nonempty_str(
    py: Python<'_>,
    raw: Bound<'_, PyDict>,
    key: String,
    where_: String,
) -> PyResult<String> {
    let slot = raw.get_item(key.as_str()).map_err(|e| {
        PyValueError::new_err(format!(
            "contracts rc_require_nonempty_str field {key:?} unreadable: {e}"
        ))
    })?;
    let checked: Option<String> = match slot {
        Some(obj) if obj.is_instance_of::<PyString>() => {
            let text: String = obj.extract().map_err(|e| {
                PyValueError::new_err(format!(
                    "contracts rc_require_nonempty_str field {key:?} unreadable: {e}"
                ))
            })?;
            let stripped: String = obj
                .call_method0("strip")
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "contracts rc_require_nonempty_str field {key:?} unreadable: {e}"
                    ))
                })?
                .extract()
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "contracts rc_require_nonempty_str field {key:?} unreadable: {e}"
                    ))
                })?;
            if stripped.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        _ => None,
    };
    py.detach(|| match checked {
        Some(text) => Ok(text),
        None => Err(format!("{where_}.{key} must be a non-empty string")),
    })
    .map_err(PyValueError::new_err)
}

/// `_require_positive_int` gate (oracle `_rc_require.py:106-110`): the
/// validated original is returned unchanged, so big ints pass through exactly
/// like the oracle's `return value`.
#[pyfunction]
fn rc_require_positive_int(
    py: Python<'_>,
    raw: Bound<'_, PyDict>,
    key: String,
    where_: String,
) -> PyResult<Py<PyAny>> {
    let slot = raw.get_item(key.as_str()).map_err(|e| {
        PyValueError::new_err(format!(
            "contracts rc_require_positive_int field {key:?} unreadable: {e}"
        ))
    })?;
    let (view, handle): (IntView, Py<PyAny>) = match slot {
        None => (missing_int(), py.None()),
        Some(obj) => (stage_int(&obj)?, obj.unbind()),
    };
    py.detach(|| check_positive_int(&where_, &key, &view))
        .map_err(PyValueError::new_err)?;
    Ok(handle)
}

/// `_require_bounded_int` gate (oracle `_rc_require.py:113-118`).
#[pyfunction]
fn rc_require_bounded_int(
    py: Python<'_>,
    raw: Bound<'_, PyDict>,
    key: String,
    where_: String,
    lo: i64,
    hi: i64,
) -> PyResult<Py<PyAny>> {
    let slot = raw.get_item(key.as_str()).map_err(|e| {
        PyValueError::new_err(format!(
            "contracts rc_require_bounded_int field {key:?} unreadable: {e}"
        ))
    })?;
    let (view, handle): (IntView, Py<PyAny>) = match slot {
        None => (missing_int(), py.None()),
        Some(obj) => (stage_int(&obj)?, obj.unbind()),
    };
    py.detach(|| check_bounded_int(&where_, &key, &view, lo, hi))
        .map_err(PyValueError::new_err)?;
    Ok(handle)
}

/// `_require_nonnegative_int` gate (oracle `_rc_require.py:121-125`).
#[pyfunction]
fn rc_require_nonnegative_int(
    py: Python<'_>,
    raw: Bound<'_, PyDict>,
    key: String,
    where_: String,
) -> PyResult<Py<PyAny>> {
    let slot = raw.get_item(key.as_str()).map_err(|e| {
        PyValueError::new_err(format!(
            "contracts rc_require_nonnegative_int field {key:?} unreadable: {e}"
        ))
    })?;
    let (view, handle): (IntView, Py<PyAny>) = match slot {
        None => (missing_int(), py.None()),
        Some(obj) => (stage_int(&obj)?, obj.unbind()),
    };
    py.detach(|| check_nonnegative_int(&where_, &key, &view))
        .map_err(PyValueError::new_err)?;
    Ok(handle)
}

/// `_require_nonnegative_float` gate (oracle `_rc_require.py:128-135`):
/// ints widen to `f64` exactly like the oracle's `float(value)`.
#[pyfunction]
fn rc_require_nonnegative_float(
    py: Python<'_>,
    raw: Bound<'_, PyDict>,
    key: String,
    where_: String,
) -> PyResult<f64> {
    let slot = raw.get_item(key.as_str()).map_err(|e| {
        PyValueError::new_err(format!(
            "contracts rc_require_nonnegative_float field {key:?} unreadable: {e}"
        ))
    })?;
    let view = match slot {
        None => missing_num(),
        Some(obj) => stage_num(&obj)?,
    };
    py.detach(|| check_nonnegative_float(&where_, &key, &view))
        .map_err(PyValueError::new_err)
}

/// `_weight_map` gate (oracle `_rc_require.py:138-157`): absent/null reads as
/// `None`; entries keep mapping order.
#[pyfunction]
fn rc_weight_map(
    py: Python<'_>,
    raw: Bound<'_, PyDict>,
    key: String,
    where_: String,
) -> PyResult<Option<Py<PyDict>>> {
    let slot = raw.get_item(key.as_str()).map_err(|e| {
        PyValueError::new_err(format!(
            "contracts rc_weight_map field {key:?} unreadable: {e}"
        ))
    })?;
    let Some(value) = slot else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let dict: Bound<'_, PyDict> = value.extract().map_err(|_| {
        let type_name = py_type_name(&value).unwrap_or_else(|_| "?".to_owned());
        PyValueError::new_err(format!(
            "{where_}.{key} must be a mapping or null, got {type_name}"
        ))
    })?;
    let mut entries = Vec::with_capacity(dict.len());
    for (head, weight) in dict.iter() {
        let head_repr = py_repr(&head)?;
        let Ok(head_str) = head.extract::<String>() else {
            entries.push(MapEntry {
                key: String::new(),
                key_repr: head_repr,
                key_ok: false,
                number: None,
            });
            continue;
        };
        entries.push(MapEntry {
            key: head_str,
            key_repr: head_repr,
            key_ok: true,
            number: stage_map_number(&weight)?,
        });
    }
    let pairs = py
        .detach(|| check_weight_map(&where_, &key, &entries))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    for (head, number) in pairs {
        out.set_item(head, number)?;
    }
    Ok(Some(out.unbind()))
}

/// `_require_bounded_float` gate (oracle `_rc_require.py:160-182`), strict
/// `[lo, hi]` with `lo_open`/`hi_open` selecting open ends (defaults mirror
/// the oracle signature; precedent `eval.rs:108`).
#[pyfunction]
#[pyo3(signature = (raw, key, where_, lo, hi, lo_open = false, hi_open = false))]
#[allow(clippy::too_many_arguments)]
fn rc_require_bounded_float(
    py: Python<'_>,
    raw: Bound<'_, PyDict>,
    key: String,
    where_: String,
    lo: Bound<'_, PyAny>,
    hi: Bound<'_, PyAny>,
    lo_open: bool,
    hi_open: bool,
) -> PyResult<f64> {
    let slot = raw.get_item(key.as_str()).map_err(|e| {
        PyValueError::new_err(format!(
            "contracts rc_require_bounded_float field {key:?} unreadable: {e}"
        ))
    })?;
    let view = match slot {
        None => missing_num(),
        Some(obj) => stage_num(&obj)?,
    };
    let (lo_value, lo_repr) = stage_bound(&lo)?;
    let (hi_value, hi_repr) = stage_bound(&hi)?;
    let lo_bound = FloatBound {
        value: lo_value,
        repr: lo_repr,
        open: lo_open,
    };
    let hi_bound = FloatBound {
        value: hi_value,
        repr: hi_repr,
        open: hi_open,
    };
    py.detach(|| check_bounded_float(&where_, &key, &view, &lo_bound, &hi_bound))
        .map_err(PyValueError::new_err)
}

/// `_positive_float_map` gate (oracle `_rc_require.py:185-205`).
#[pyfunction]
fn rc_positive_float_map(
    py: Python<'_>,
    raw: Bound<'_, PyDict>,
    key: String,
    where_: String,
) -> PyResult<Option<Py<PyDict>>> {
    let slot = raw.get_item(key.as_str()).map_err(|e| {
        PyValueError::new_err(format!(
            "contracts rc_positive_float_map field {key:?} unreadable: {e}"
        ))
    })?;
    let Some(value) = slot else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    let dict: Bound<'_, PyDict> = value.extract().map_err(|_| {
        let type_name = py_type_name(&value).unwrap_or_else(|_| "?".to_owned());
        PyValueError::new_err(format!(
            "{where_}.{key} must be a mapping or null, got {type_name}"
        ))
    })?;
    let mut entries = Vec::with_capacity(dict.len());
    for (head, mult) in dict.iter() {
        let head_repr = py_repr(&head)?;
        let Ok(head_str) = head.extract::<String>() else {
            entries.push(MapEntry {
                key: String::new(),
                key_repr: head_repr,
                key_ok: false,
                number: None,
            });
            continue;
        };
        entries.push(MapEntry {
            key: head_str,
            key_repr: head_repr,
            key_ok: true,
            number: stage_map_number(&mult)?,
        });
    }
    let pairs = py
        .detach(|| check_positive_float_map(&where_, &key, &entries))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    for (head, number) in pairs {
        out.set_item(head, number)?;
    }
    Ok(Some(out.unbind()))
}

/// `_require_loop_bool` gate (oracle `_rc_require.py:208-215`): absent/null
/// reads as `default`; the literal `loop.` prefix is kept verbatim.
#[pyfunction]
fn rc_require_loop_bool(
    py: Python<'_>,
    raw: Bound<'_, PyDict>,
    key: String,
    default: bool,
) -> PyResult<bool> {
    let slot = raw.get_item(key.as_str()).map_err(|e| {
        PyValueError::new_err(format!(
            "contracts rc_require_loop_bool field {key:?} unreadable: {e}"
        ))
    })?;
    let staged: (Option<bool>, String) = match slot {
        None => (Some(default), String::new()),
        Some(obj) if obj.is_none() => (Some(default), String::new()),
        Some(obj) if obj.is_instance_of::<PyBool>() => {
            let value: bool = obj.extract().map_err(|e| {
                PyValueError::new_err(format!(
                    "contracts rc_require_loop_bool field {key:?} unreadable: {e}"
                ))
            })?;
            (Some(value), String::new())
        }
        Some(obj) => (None, py_repr(&obj)?),
    };
    py.detach(|| match staged.0 {
        Some(value) => Ok(value),
        None => Err(format!("loop.{key} must be a bool, got {}", staged.1)),
    })
    .map_err(PyValueError::new_err)
}

/// Digest-shape test shared by the pin staging below.
#[derive(Debug)]
enum PinKind {
    Absent,
    Text(String),
    Other,
}

/// `_digest_pin_or_none` gate (oracle `_rc_require.py:221-228`): absent/null
/// reads as `None`.
#[pyfunction]
fn rc_digest_pin_or_none(
    py: Python<'_>,
    raw: Bound<'_, PyDict>,
    key: String,
    where_: String,
) -> PyResult<Option<String>> {
    let slot = raw.get_item(key.as_str()).map_err(|e| {
        PyValueError::new_err(format!(
            "contracts rc_digest_pin_or_none field {key:?} unreadable: {e}"
        ))
    })?;
    let staged: (PinKind, String) = match slot {
        None => (PinKind::Absent, String::new()),
        Some(obj) if obj.is_none() => (PinKind::Absent, String::new()),
        Some(obj) if obj.is_instance_of::<PyString>() => {
            let text: String = obj.extract().map_err(|e| {
                PyValueError::new_err(format!(
                    "contracts rc_digest_pin_or_none field {key:?} unreadable: {e}"
                ))
            })?;
            (PinKind::Text(text), String::new())
        }
        Some(obj) => (PinKind::Other, py_repr(&obj)?),
    };
    py.detach(|| match &staged.0 {
        PinKind::Absent => Ok(None),
        PinKind::Text(text) => {
            if is_digest_shape(text) {
                Ok(Some(text.clone()))
            } else {
                Err(format!(
                    "{where_}.{key} must be null or sha256:<64 hex>, got '{text}'",
                ))
            }
        }
        PinKind::Other => Err(format!(
            "{where_}.{key} must be null or sha256:<64 hex>, got {}",
            staged.1
        )),
    })
    .map_err(PyValueError::new_err)
}

/// Register the run-config require gates on the shared `contracts` submodule
/// (mirrors `validate.rs:111-127`); MAIN calls this from
/// `contracts::register` — no new submodule, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(rc_reject_unknown, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_require_nonempty_str, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_require_positive_int, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_require_bounded_int, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_require_nonnegative_int, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_require_nonnegative_float, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_weight_map, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_require_bounded_float, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_positive_float_map, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_require_loop_bool, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_digest_pin_or_none, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyFloat, PyList};

    fn dict_with(py: Python<'_>) -> Bound<'_, PyDict> {
        PyDict::new(py)
    }

    #[test]
    fn reject_unknown_accepts_known_and_copies() {
        Python::initialize();
        Python::attach(|py| {
            let raw = dict_with(py);
            raw.set_item("id", "r1").unwrap();
            let out = rc_reject_unknown(
                py,
                raw.into_any(),
                vec!["id".to_owned(), "kind".to_owned()],
                "run".to_owned(),
            )
            .unwrap();
            assert_eq!(out.bind(py).len(), 1);
            assert_eq!(
                out.bind(py)
                    .get_item("id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "r1"
            );
            // Unknown keys render exactly like the oracle
            // (`_rc_require.py:95`): sorted unknowns, sorted allowlist.
            let bad = dict_with(py);
            bad.set_item("zzz", 1).unwrap();
            bad.set_item("id", "r1").unwrap();
            let err = rc_reject_unknown(
                py,
                bad.into_any(),
                vec!["id".to_owned(), "data".to_owned()],
                "run-config".to_owned(),
            )
            .unwrap_err();
            assert!(
                err.to_string()
                    .contains("run-config has unknown keys ['zzz']; allowed=['data', 'id']"),
                "unexpected error text: {err}"
            );
            // Non-mapping input names its type (`_rc_require.py:92`).
            let list = PyList::new(py, [1]).unwrap().into_any();
            let err =
                rc_reject_unknown(py, list, vec!["id".to_owned()], "run".to_owned()).unwrap_err();
            assert!(
                err.to_string().contains("run must be a mapping, got list"),
                "unexpected error text: {err}"
            );
        });
    }

    #[test]
    fn nonempty_str_matches_oracle() {
        Python::initialize();
        Python::attach(|py| {
            let raw = dict_with(py);
            raw.set_item("root", "/data").unwrap();
            assert_eq!(
                rc_require_nonempty_str(py, raw.clone(), "root".to_owned(), "data".to_owned())
                    .unwrap(),
                "/data"
            );
            for (slot, label) in [("   ", "blank"), ("", "empty")] {
                let d = dict_with(py);
                d.set_item("root", slot).unwrap();
                let err = rc_require_nonempty_str(py, d, "root".to_owned(), "data".to_owned())
                    .unwrap_err();
                assert!(
                    err.to_string()
                        .contains("data.root must be a non-empty string"),
                    "unexpected {label} error text: {err}"
                );
            }
            let missing = dict_with(py);
            let err = rc_require_nonempty_str(py, missing, "root".to_owned(), "data".to_owned())
                .unwrap_err();
            assert!(
                err.to_string()
                    .contains("data.root must be a non-empty string"),
                "unexpected missing error text: {err}"
            );
        });
    }

    #[test]
    fn int_gates_match_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // Positive: valid passes through, zero/negatives/bool/str/missing fail.
            let raw = dict_with(py);
            raw.set_item("world_size", 4).unwrap();
            let back = rc_require_positive_int(py, raw, "world_size".to_owned(), "data".to_owned())
                .unwrap();
            assert_eq!(back.bind(py).extract::<i64>().unwrap(), 4);
            for (value, want) in [
                (0i64, "data.world_size must be a positive int, got 0"),
                (-2i64, "data.world_size must be a positive int, got -2"),
            ] {
                let d = dict_with(py);
                d.set_item("world_size", value).unwrap();
                let err =
                    rc_require_positive_int(py, d, "world_size".to_owned(), "data".to_owned())
                        .unwrap_err();
                assert!(
                    err.to_string().contains(want),
                    "unexpected error text: {err}"
                );
            }
            let d = dict_with(py);
            d.set_item("world_size", true).unwrap();
            let err = rc_require_positive_int(py, d, "world_size".to_owned(), "data".to_owned())
                .unwrap_err();
            assert!(
                err.to_string()
                    .contains("data.world_size must be a positive int, got True"),
                "unexpected bool error text: {err}"
            );
            let d = dict_with(py);
            d.set_item("world_size", "x").unwrap();
            let err = rc_require_positive_int(py, d, "world_size".to_owned(), "data".to_owned())
                .unwrap_err();
            assert!(
                err.to_string()
                    .contains("data.world_size must be a positive int, got 'x'"),
                "unexpected str error text: {err}"
            );
            // Big ints pass through exactly (oracle `return value`).
            let d = dict_with(py);
            d.set_item("world_size", 10u128.pow(30)).unwrap();
            let back =
                rc_require_positive_int(py, d, "world_size".to_owned(), "data".to_owned()).unwrap();
            let text: String = back.bind(py).str().unwrap().extract().unwrap();
            assert_eq!(text, format!("1{}", "0".repeat(30)));
            let d = dict_with(py);
            d.set_item("world_size", -(10i128.pow(30))).unwrap();
            let err = rc_require_positive_int(py, d, "world_size".to_owned(), "data".to_owned())
                .unwrap_err();
            assert!(
                err.to_string().contains("must be a positive int, got -1"),
                "unexpected big-negative error text: {err}"
            );
            // Bounded / non-negative shapes.
            let d = dict_with(py);
            d.set_item("decode_prefetch", 64).unwrap();
            let back = rc_require_bounded_int(
                py,
                d,
                "decode_prefetch".to_owned(),
                "data".to_owned(),
                1,
                1024,
            )
            .unwrap();
            assert_eq!(back.bind(py).extract::<i64>().unwrap(), 64);
            let d = dict_with(py);
            d.set_item("decode_prefetch", 0).unwrap();
            let err = rc_require_bounded_int(
                py,
                d,
                "decode_prefetch".to_owned(),
                "data".to_owned(),
                1,
                1024,
            )
            .unwrap_err();
            assert!(
                err.to_string()
                    .contains("data.decode_prefetch must be an int in [1, 1024], got 0"),
                "unexpected bounded error text: {err}"
            );
            let d = dict_with(py);
            d.set_item("num_workers", 0).unwrap();
            let back =
                rc_require_nonnegative_int(py, d, "num_workers".to_owned(), "data".to_owned())
                    .unwrap();
            assert_eq!(back.bind(py).extract::<i64>().unwrap(), 0);
            let d = dict_with(py);
            d.set_item("num_workers", -1).unwrap();
            let err =
                rc_require_nonnegative_int(py, d, "num_workers".to_owned(), "data".to_owned())
                    .unwrap_err();
            assert!(
                err.to_string()
                    .contains("data.num_workers must be a non-negative int, got -1"),
                "unexpected non-negative error text: {err}"
            );
        });
    }

    #[test]
    fn float_gates_match_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // Ints widen to f64; bools / strings / NaN / inf reject.
            let d = dict_with(py);
            d.set_item("w_policy", 2).unwrap();
            assert_eq!(
                rc_require_nonnegative_float(py, d, "w_policy".to_owned(), "weights".to_owned())
                    .unwrap(),
                2.0
            );
            let d = dict_with(py);
            d.set_item("w_policy", 0.5).unwrap();
            assert_eq!(
                rc_require_nonnegative_float(py, d, "w_policy".to_owned(), "weights".to_owned())
                    .unwrap(),
                0.5
            );
            let d = dict_with(py);
            d.set_item("w_policy", true).unwrap();
            let err =
                rc_require_nonnegative_float(py, d, "w_policy".to_owned(), "weights".to_owned())
                    .unwrap_err();
            assert!(
                err.to_string()
                    .contains("weights.w_policy must be a non-negative number, got True"),
                "unexpected bool error text: {err}"
            );
            let d = dict_with(py);
            d.set_item("w_policy", f64::NAN).unwrap();
            let err =
                rc_require_nonnegative_float(py, d, "w_policy".to_owned(), "weights".to_owned())
                    .unwrap_err();
            assert!(
                err.to_string()
                    .contains("weights.w_policy must be finite and non-negative, got nan"),
                "unexpected nan error text: {err}"
            );
            let d = dict_with(py);
            d.set_item("w_policy", f64::NEG_INFINITY).unwrap();
            let err =
                rc_require_nonnegative_float(py, d, "w_policy".to_owned(), "weights".to_owned())
                    .unwrap_err();
            assert!(
                err.to_string()
                    .contains("weights.w_policy must be finite and non-negative, got -inf"),
                "unexpected -inf error text: {err}"
            );
            // Bounded float: closed and open ends render Python-style.
            let lo = PyFloat::new(py, 0.0).into_any();
            let hi = PyFloat::new(py, 1.0).into_any();
            let d = dict_with(py);
            d.set_item("label_smoothing", 0.03).unwrap();
            assert_eq!(
                rc_require_bounded_float(
                    py,
                    d,
                    "label_smoothing".to_owned(),
                    "weights".to_owned(),
                    lo.clone(),
                    hi.clone(),
                    false,
                    true,
                )
                .unwrap(),
                0.03
            );
            let d = dict_with(py);
            d.set_item("label_smoothing", 1.0).unwrap();
            let err = rc_require_bounded_float(
                py,
                d,
                "label_smoothing".to_owned(),
                "weights".to_owned(),
                lo.clone(),
                hi.clone(),
                false,
                true,
            )
            .unwrap_err();
            assert!(
                err.to_string()
                    .contains("weights.label_smoothing must lie in [0.0, 1.0), got 1.0"),
                "unexpected open-hi error text: {err}"
            );
            let d = dict_with(py);
            d.set_item("label_smoothing", f64::NAN).unwrap();
            let err = rc_require_bounded_float(
                py,
                d,
                "label_smoothing".to_owned(),
                "weights".to_owned(),
                lo,
                hi,
                false,
                true,
            )
            .unwrap_err();
            assert!(
                err.to_string()
                    .contains("weights.label_smoothing must be finite, got nan"),
                "unexpected nan error text: {err}"
            );
        });
    }

    #[test]
    fn map_gates_match_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // Weight map: order kept, absent → None, bad entries fail closed.
            let d = dict_with(py);
            let inner = PyDict::new(py);
            inner.set_item("a", 1.0).unwrap();
            inner.set_item("b", 2).unwrap();
            d.set_item("w_event", inner).unwrap();
            let out = rc_weight_map(py, d, "w_event".to_owned(), "weights".to_owned())
                .unwrap()
                .unwrap();
            assert_eq!(out.bind(py).len(), 2);
            assert_eq!(
                out.bind(py)
                    .get_item("b")
                    .unwrap()
                    .unwrap()
                    .extract::<f64>()
                    .unwrap(),
                2.0
            );
            let d = dict_with(py);
            assert!(
                rc_weight_map(py, d, "w_event".to_owned(), "weights".to_owned())
                    .unwrap()
                    .is_none()
            );
            let d = dict_with(py);
            d.set_item("w_event", 5).unwrap();
            let err = rc_weight_map(py, d, "w_event".to_owned(), "weights".to_owned()).unwrap_err();
            assert!(
                err.to_string()
                    .contains("weights.w_event must be a mapping or null, got int"),
                "unexpected non-map error text: {err}"
            );
            let d = dict_with(py);
            let inner = PyDict::new(py);
            inner.set_item("a", f64::INFINITY).unwrap();
            d.set_item("w_event", inner).unwrap();
            let err = rc_weight_map(py, d, "w_event".to_owned(), "weights".to_owned()).unwrap_err();
            assert!(
                err.to_string()
                    .contains("weights.w_event['a'] must be finite non-negative"),
                "unexpected inf-entry error text: {err}"
            );
            // Positive map: zero rejects, positives pass.
            let d = dict_with(py);
            let inner = PyDict::new(py);
            inner.set_item("m", 3.0).unwrap();
            d.set_item("head_lr_mult", inner).unwrap();
            let out = rc_positive_float_map(py, d, "head_lr_mult".to_owned(), "opt".to_owned())
                .unwrap()
                .unwrap();
            assert_eq!(out.bind(py).len(), 1);
            let d = dict_with(py);
            let inner = PyDict::new(py);
            inner.set_item("m", 0.0).unwrap();
            d.set_item("head_lr_mult", inner).unwrap();
            let err = rc_positive_float_map(py, d, "head_lr_mult".to_owned(), "opt".to_owned())
                .unwrap_err();
            assert!(
                err.to_string()
                    .contains("opt.head_lr_mult['m'] must be positive and finite"),
                "unexpected zero-entry error text: {err}"
            );
        });
    }

    #[test]
    fn loop_bool_and_digest_pin_match_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // Absent/null read the default; non-bools name the loop prefix.
            let d = dict_with(py);
            assert!(rc_require_loop_bool(py, d.clone(), "drop_last".to_owned(), true).unwrap());
            d.set_item("drop_last", py.None()).unwrap();
            assert!(!rc_require_loop_bool(py, d.clone(), "drop_last".to_owned(), false).unwrap());
            d.set_item("drop_last", true).unwrap();
            assert!(rc_require_loop_bool(py, d.clone(), "drop_last".to_owned(), false).unwrap());
            d.set_item("drop_last", 1).unwrap();
            let err = rc_require_loop_bool(py, d, "drop_last".to_owned(), false).unwrap_err();
            assert!(
                err.to_string()
                    .contains("loop.drop_last must be a bool, got 1"),
                "unexpected bool error text: {err}"
            );
            // Digest pin: exact lowercase shape, null when absent.
            let good = format!("sha256:{}", "ab".repeat(32));
            let d = dict_with(py);
            d.set_item("h", good.clone()).unwrap();
            assert_eq!(
                rc_digest_pin_or_none(py, d, "h".to_owned(), "data".to_owned()).unwrap(),
                Some(good.clone())
            );
            let d = dict_with(py);
            assert!(
                rc_digest_pin_or_none(py, d, "h".to_owned(), "data".to_owned())
                    .unwrap()
                    .is_none()
            );
            let d = dict_with(py);
            d.set_item("h", good.to_uppercase()).unwrap();
            let err = rc_digest_pin_or_none(py, d, "h".to_owned(), "data".to_owned()).unwrap_err();
            assert!(
                err.to_string()
                    .contains("data.h must be null or sha256:<64 hex>, got 'SHA256:"),
                "unexpected uppercase error text: {err}"
            );
        });
    }
}
