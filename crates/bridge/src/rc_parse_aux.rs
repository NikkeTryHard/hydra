//! rc_parse_aux: run-config auxiliary section parsers on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (no feed/shard/search edge; pure argument validation, so no
//! crate owner exists — same NONE position as `rc_require.rs:32-38`; the
//! run-id shape test restates the frozen pattern owned by `rc_sections.rs:79`
//! (`RUN_ID_PATTERN`), never a new printer). FORBIDS: YAML/file IO, `${VAR}`
//! interpolation, base+override composition (all stay Python in
//! `_rc_require.py` / `_rc_root.py`), dataclass state (all six section records
//! stay Python in `_rc_sections.py`), the cross-module
//! `eval.statistics.SelectionConfig` authoritative construction (stays Python
//! in `_rc_parse_aux.py`), wall-clock/RNG, and `ContractError` shaping (the
//! bridge raises `ValueError`; thin Python translators map to `ContractError`
//! with byte-identical messages).
//!
//! TABLE ported here — the six observer-side section parsers, oracle
//! `python/hydra2/training/_rc_parse_aux.py` (worktree md5 `88818379…`,
//! equals `HEAD:src/hydra2/training/_rc_parse_aux.py` at port time):
//! - `rc_parse_seeds` ← `_parse_seeds` (:36-48)
//! - `rc_parse_selection` ← `_parse_selection` (:51-139)
//! - `rc_parse_telemetry` ← `_parse_telemetry` (:142-181)
//! - `rc_parse_mirror` ← `_parse_mirror` (:184-202)
//! - `rc_parse_eval` ← `_parse_eval` (:205-223)
//! - `rc_parse_output` ← `_parse_output` (:226-236)
//!
//! LOGIC staying Python: the six frozen dataclass constructions
//! (`SeedsConfig`/`SelectionConfig`/`TelemetryConfig`/`MirrorConfig`/
//! `EvalConfig`/`OutputConfig`), `__all__`, the bridge-missing helper, all six
//! `ContractError` translators, and the `eval.statistics` authoritative
//! selection check (oracle `:122-138`, `ImportError`-tolerant).
//!
//! Shape per fn: attached staging (reprs via live `repr()`, `len()` for
//! emptiness so lone-surrogate strings never decode, live `strip()` for
//! `artifact_root` per `rc_require.rs:492-504`, `float()` widening via
//! `extract::<f64>()` so huge-int `OverflowError` surfaces at stage time
//! exactly where the oracle's `float(value)` raises per
//! `rc_require.rs:126-138`) → ONE `py.detach(|| …)` over owned plain data
//! with zero Python API inside (per `contracts.rs:434-437`) → attached wrap
//! as `PyValueError` (per `contracts.rs:117-123`) into a resolved `PyDict`
//! the translator splats into the dataclass. Bool is excluded before every
//! int/float check (per `contracts.rs:69-74`); a missing `get_item` slot reads
//! as the oracle default exactly like the oracle's `fields.get(key, default)`
//! (per `packet_decode.rs:375-378`); result dicts are built with
//! `PyDict::new` plus `set_item` (per `contracts.rs:1227-1231`). Check order
//! inside each detach mirrors the oracle statement order, so the FIRST error
//! on multi-violation inputs is identical.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `rc_require.rs:849-862`); no new entry
//! point. MAIN wiring: `pub mod rc_parse_aux;` in `lib.rs` plus
//! `crate::rc_parse_aux::register(&sub)?;` in `contracts.rs` next to
//! `crate::rc_sections::register(&sub)?;` (`contracts.rs:1295`).
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `rc_require::register`
//! (`crates/bridge/src/rc_require.rs:849-850`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `rc_require::rc_require_positive_int`
//! (`crates/bridge/src/rc_require.rs:519-538`); per-slot `get_item` reads
//! mirror (`crates/bridge/src/rc_require.rs:526-530`); `PyTuple::new` mirrors
//! (`crates/bridge/src/contracts.rs:1156`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PyString, PyTuple};

/// Optional bool slot staging: `((is_bool, repr), handle)`.
type OptBoolStaged = (Option<(bool, String)>, Option<Py<PyAny>>);

/// Allowed keys per section (oracle `_rc_parse_aux.py` `_reject_unknown`
/// call sites; the unknown-keys message sorts them, so order here is the
/// oracle call-site order, not the message order).
const SEEDS_ALLOWED: [&str; 3] = ["data_seed", "train_seed", "selection_seed"];
const SELECTION_ALLOWED: [&str; 10] = [
    "N",
    "pilot_s",
    "delta",
    "alpha",
    "beta",
    "design",
    "declared_peeks",
    "margin",
    "resamples",
    "seed",
];
const TELEMETRY_ALLOWED: [&str; 4] = [
    "mlflow_enabled",
    "verbose_enabled",
    "verbose_interval_ms",
    "profiler_captures",
];
const MIRROR_ALLOWED: [&str; 4] = ["enabled", "project", "task_name", "offline_dir"];
const EVAL_ALLOWED: [&str; 4] = [
    "frequency_updates",
    "num_batches",
    "walls_dir",
    "microbatch_size",
];
const OUTPUT_ALLOWED: [&str; 2] = ["artifact_root", "run_id"];

/// Frozen selection designs (`_rc_sections.py:108`, single-sourced as
/// `RC_SELECTION_DESIGNS` in `rc_sections.rs:74`); the design error text
/// renders this list Python-`repr`-style, frozen here.
const SELECTION_DESIGNS: [&str; 2] = ["fixed_n", "time_uniform_cs"];

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Attached staging helper: exact `type(value).__name__` text for the
/// `must be a mapping` slots (mirrors `rc_require.rs:70-72`).
fn py_type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.get_type().name()?.to_str()?.to_owned())
}

/// Owned plain-data view of an int-or-other Python value (mirrors
/// `rc_require.rs:79-84`: bool excluded first, big ints past `i64` keep
/// `is_int` with `as_i64` unset and sign from the staged repr).
struct IntView {
    repr: String,
    is_int: bool,
    as_i64: Option<i64>,
    negative: bool,
}

/// Stage one int-or-other slot (mirrors `rc_require.rs:86-104`).
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

/// Owned plain-data view of a number-or-other value (mirrors
/// `rc_require.rs:121-124`): `number` is `Some` iff the value is a non-bool
/// int/float. A huge int that overflows `f64` keeps the original (Overflow)
/// error at stage time — exactly where the oracle's `float(value)` raises,
/// never a `ContractError` shape.
struct NumView {
    repr: String,
    number: Option<f64>,
}

/// Stage one number-or-other slot (mirrors `rc_require.rs:126-138`).
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

/// Staged keys plus allowlist for the unknown-key envelope gate (mirrors
/// `rc_require.rs:376-380`).
struct UnknownView {
    keys: Vec<(String, Option<String>)>,
    allowed: Vec<String>,
    where_: String,
}

/// Envelope gate over owned data only (mirrors `rc_require.rs:382-427`
/// exactly: allowed sorted and single-quote rendered, unknown sorted,
/// mixed non-str unknowns fail as unorderable like the oracle's `sorted()`).
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

/// Detached positive-int gate (mirrors `rc_require.rs:203-217`).
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

/// Detached non-negative-int gate (mirrors `rc_require.rs:241-255`).
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

/// Run-id shape test for `_parse_output` (oracle `_rc_parse_aux.py:234`
/// `_RUN_ID_RE.fullmatch`, pattern text `rc_sections.rs:79`
/// `"[A-Za-z0-9][A-Za-z0-9_-]*\\Z"`): ASCII-only, first char alphanumeric,
/// rest alphanumeric/`_`/`-`. Empty fails.
fn is_run_id_shape(text: &str) -> bool {
    let mut chars = text.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphanumeric() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

// ---------------------------------------------------------------------------
// Attached staging helpers (one per slot shape; all GIL-held).
// ---------------------------------------------------------------------------

/// Extract the section mapping or fail with the envelope's `must be a
/// mapping` text (mirrors `rc_require.rs:442-450`).
fn as_section_mapping<'py>(raw: &Bound<'py, PyAny>, where_: &str) -> PyResult<Bound<'py, PyDict>> {
    match raw.extract() {
        Ok(dict) => Ok(dict),
        Err(_) => {
            let type_name = py_type_name(raw).unwrap_or_else(|_| "?".to_owned());
            Err(PyValueError::new_err(format!(
                "{where_} must be a mapping, got {type_name}"
            )))
        }
    }
}

/// Stage the unknown-key envelope view (mirrors `rc_require.rs:451-460`).
fn stage_unknown(
    dict: &Bound<'_, PyDict>,
    allowed: &[&str],
    where_: &str,
) -> PyResult<UnknownView> {
    let mut keys = Vec::with_capacity(dict.len());
    for (head, _value) in dict.iter() {
        let head_repr = py_repr(&head)?;
        keys.push((head_repr, head.extract::<String>().ok()));
    }
    Ok(UnknownView {
        keys,
        allowed: allowed.iter().map(|s| s.to_string()).collect(),
        where_: where_.to_owned(),
    })
}

/// Read one optional slot: missing (`get_item` → `None`, mirroring the
/// oracle's `fields.get`) vs present (mirrors `rc_require.rs:526-530`).
fn slot_of<'a>(
    dict: &Bound<'a, PyDict>,
    key: &str,
    owner: &str,
) -> PyResult<Option<Bound<'a, PyAny>>> {
    dict.get_item(key).map_err(|e| {
        PyValueError::new_err(format!("contracts {owner} field {key:?} unreadable: {e}"))
    })
}

/// Stage one optional int slot: `None` when the oracle default applies,
/// otherwise the int view plus the original handle (big ints pass through
/// exactly like the oracle's `return value`).
fn stage_opt_int(
    dict: &Bound<'_, PyDict>,
    key: &str,
    owner: &str,
) -> PyResult<(Option<IntView>, Option<Py<PyAny>>)> {
    match slot_of(dict, key, owner)? {
        None => Ok((None, None)),
        Some(obj) => {
            let view = stage_int(&obj)?;
            Ok((Some(view), Some(obj.unbind())))
        }
    }
}

/// Stage one optional number slot (`None` when the oracle default applies).
/// Huge-int `OverflowError` propagates here, exactly where the oracle's
/// `float(value)` raises.
fn stage_opt_num(dict: &Bound<'_, PyDict>, key: &str, owner: &str) -> PyResult<Option<NumView>> {
    match slot_of(dict, key, owner)? {
        None => Ok(None),
        Some(obj) => Ok(Some(stage_num(&obj)?)),
    }
}

/// Stage one optional bool slot: `None` when the oracle default applies,
/// otherwise `(is_bool, repr)` plus the original handle.
fn stage_opt_bool(dict: &Bound<'_, PyDict>, key: &str, owner: &str) -> PyResult<OptBoolStaged> {
    match slot_of(dict, key, owner)? {
        None => Ok((None, None)),
        Some(obj) => {
            let is_bool = obj.is_instance_of::<PyBool>();
            let repr = py_repr(&obj)?;
            Ok((Some((is_bool, repr)), Some(obj.unbind())))
        }
    }
}

/// Plain-data state of an optional string slot: `Missing` (oracle default),
/// `Null` (explicit null → `None`), or `Value` (present non-null; `ok` is
/// the oracle accept bit).
enum OptStrState {
    Missing,
    Null,
    Value { ok: bool },
}

/// Stage one optional string slot. `stripped` selects the oracle emptiness
/// test: plain `== ""` (mirror/eval `project`/`task_name`/`walls_dir`) vs
/// live-`strip() == ""` (`output.artifact_root`). Emptiness uses `len()`, so
/// lone-surrogate strings never decode. Non-strings stage `ok: false` and
/// fail detached with the oracle static text.
fn stage_opt_str(
    dict: &Bound<'_, PyDict>,
    key: &str,
    owner: &str,
    stripped: bool,
) -> PyResult<(OptStrState, Option<Py<PyAny>>)> {
    let unreadable = |e: pyo3::PyErr| {
        PyValueError::new_err(format!("contracts {owner} field {key:?} unreadable: {e}"))
    };
    match slot_of(dict, key, owner)? {
        None => Ok((OptStrState::Missing, None)),
        Some(obj) => {
            if obj.is_none() {
                return Ok((OptStrState::Null, None));
            }
            let handle = obj.unbind();
            let bound = handle.bind(dict.py());
            if !bound.is_instance_of::<PyString>() {
                return Ok((OptStrState::Value { ok: false }, Some(handle)));
            }
            let ok = if stripped {
                let surface = bound.call_method0("strip").map_err(unreadable)?;
                surface.len().map_err(unreadable)? != 0
            } else {
                bound.len().map_err(unreadable)? != 0
            };
            Ok((OptStrState::Value { ok }, Some(handle)))
        }
    }
}

/// Plain-data state of the `output.run_id` slot.
enum RunIdState {
    Missing,
    Null,
    Value { repr: String, ok: bool },
}

/// Stage the `output.run_id` slot: null when absent-or-null, otherwise the
/// repr plus the ASCII shape bit. `extract::<String>()` failing (non-str, or
/// undecodable) stages `ok: false`, matching the oracle's
/// `not isinstance(str) or fullmatch is None` disjunction exactly.
fn stage_run_id(
    dict: &Bound<'_, PyDict>,
    owner: &str,
) -> PyResult<(RunIdState, Option<Py<PyAny>>)> {
    match slot_of(dict, "run_id", owner)? {
        None => Ok((RunIdState::Missing, None)),
        Some(obj) => {
            if obj.is_none() {
                return Ok((RunIdState::Null, None));
            }
            let repr = py_repr(&obj)?;
            let ok = obj
                .extract::<String>()
                .ok()
                .is_some_and(|t| is_run_id_shape(&t));
            Ok((RunIdState::Value { repr, ok }, Some(obj.unbind())))
        }
    }
}

/// Plain-data state of the `selection.design` slot.
enum DesignState {
    Missing,
    Value { repr: String, ok: bool },
}

/// Stage `selection.design`: `extract::<String>()` failing stages `ok:
/// false`, matching the oracle's `design not in _SELECTION_DESIGNS` exactly
/// (any non-member, of any type, fails with the same `got {design!r}` text).
fn stage_design(
    dict: &Bound<'_, PyDict>,
    owner: &str,
) -> PyResult<(DesignState, Option<Py<PyAny>>)> {
    match slot_of(dict, "design", owner)? {
        None => Ok((DesignState::Missing, None)),
        Some(obj) => {
            let repr = py_repr(&obj)?;
            let ok = obj
                .extract::<String>()
                .ok()
                .is_some_and(|t| SELECTION_DESIGNS.contains(&t.as_str()));
            Ok((DesignState::Value { repr, ok }, Some(obj.unbind())))
        }
    }
}

/// Plain-data view of `selection.declared_peeks`: the shape bit (non-empty
/// list/tuple), the oracle `tuple(peeks_raw)` repr rebuilt exactly from the
/// staged element reprs (`(x,)` for one element, `(x, y)` otherwise — CPython
/// tuple-`repr` joining, no Python API needed), and the per-element int
/// views. `None` (missing) takes the oracle `(30,)` default.
struct PeeksCheck {
    shape_ok: bool,
    tuple_repr: String,
    elems: Vec<IntView>,
}

/// Stage `selection.declared_peeks`, keeping the original element handles so
/// the wrap rebuilds the oracle `tuple(peeks_raw)` (big ints pass through).
fn stage_peeks(
    dict: &Bound<'_, PyDict>,
    owner: &str,
) -> PyResult<(Option<PeeksCheck>, Vec<Py<PyAny>>)> {
    let slot = slot_of(dict, "declared_peeks", owner)?;
    let Some(obj) = slot else {
        return Ok((None, Vec::new()));
    };
    let items: Option<Vec<Bound<'_, PyAny>>> = if let Ok(list) = obj.cast::<PyList>() {
        Some(list.iter().collect())
    } else if let Ok(tup) = obj.cast::<PyTuple>() {
        Some(tup.iter().collect())
    } else {
        None
    };
    let Some(items) = items else {
        return Ok((
            Some(PeeksCheck {
                shape_ok: false,
                tuple_repr: String::new(),
                elems: Vec::new(),
            }),
            Vec::new(),
        ));
    };
    if items.is_empty() {
        return Ok((
            Some(PeeksCheck {
                shape_ok: false,
                tuple_repr: String::new(),
                elems: Vec::new(),
            }),
            Vec::new(),
        ));
    }
    let mut elems = Vec::with_capacity(items.len());
    let mut handles = Vec::with_capacity(items.len());
    for item in &items {
        elems.push(stage_int(item)?);
    }
    for item in items {
        handles.push(item.unbind());
    }
    let parts: Vec<&str> = elems.iter().map(|e| e.repr.as_str()).collect();
    let tuple_repr = if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    };
    Ok((
        Some(PeeksCheck {
            shape_ok: true,
            tuple_repr,
            elems,
        }),
        handles,
    ))
}

// ---------------------------------------------------------------------------
// Detached checks (owned plain data only, zero Python API). Check order
// mirrors the oracle statement order in each parser.
// ---------------------------------------------------------------------------

/// `_parse_seeds` decisions (oracle `_rc_parse_aux.py:36-48`).
fn validate_seeds(
    unknown: &UnknownView,
    data_seed: &Option<IntView>,
    train_seed: &Option<IntView>,
    selection_seed: &Option<IntView>,
) -> Result<(), String> {
    check_unknown(unknown)?;
    if let Some(view) = data_seed {
        check_nonnegative_int("seeds", "data_seed", view)?;
    }
    if let Some(view) = train_seed {
        check_nonnegative_int("seeds", "train_seed", view)?;
    }
    if let Some(view) = selection_seed {
        check_nonnegative_int("seeds", "selection_seed", view)?;
    }
    Ok(())
}

/// `_parse_selection` decisions (oracle `_rc_parse_aux.py:51-121`): envelope,
/// design vocab, peeks shape then entries, `pilot_s`/`delta` positive-finite,
/// `alpha`/`beta` in `(0, 1)`, finite `margin`, then the `N`/`resamples`/
/// `seed` int gates in construction order. Returns the five widened floats
/// for the wrap (the oracle's `float(...)` calls, infallible post-check).
#[allow(clippy::too_many_arguments)]
fn validate_selection(
    unknown: &UnknownView,
    design: &DesignState,
    peeks: &Option<PeeksCheck>,
    pilot_s: &Option<NumView>,
    delta: &Option<NumView>,
    alpha: &Option<NumView>,
    beta: &Option<NumView>,
    margin: &Option<NumView>,
    n: &Option<IntView>,
    resamples: &Option<IntView>,
    seed: &Option<IntView>,
) -> Result<[f64; 5], String> {
    check_unknown(unknown)?;
    if let DesignState::Value { repr, ok: false } = design {
        return Err(format!(
            "selection.design must be one of ['fixed_n', 'time_uniform_cs'], got {repr}"
        ));
    }
    if let Some(check) = peeks {
        if !check.shape_ok {
            return Err(
                "selection.declared_peeks must be a non-empty list of positive ints".to_owned(),
            );
        }
        let entries_ok = check.elems.iter().all(|v| {
            v.is_int
                && match v.as_i64 {
                    Some(x) => x >= 1,
                    None => !v.negative,
                }
        });
        if !entries_ok {
            return Err(format!(
                "selection.declared_peeks entries must be positive ints, got {}",
                check.tuple_repr
            ));
        }
    }
    let widen = |key: &str, view: &Option<NumView>, default: f64| -> Result<f64, String> {
        match view {
            None => Ok(default),
            Some(v) => match v.number {
                Some(n) if n.is_finite() && n > 0.0 => Ok(n),
                _ => Err(format!(
                    "selection.{key} must be positive and finite, got {}",
                    v.repr
                )),
            },
        }
    };
    let pilot_s = widen("pilot_s", pilot_s, 1.5)?;
    let delta = widen("delta", delta, 0.5)?;
    let ranged = |key: &str, view: &Option<NumView>, default: f64| -> Result<f64, String> {
        match view {
            None => Ok(default),
            Some(v) => match v.number {
                Some(n) if 0.0 < n && n < 1.0 => Ok(n),
                _ => Err(format!(
                    "selection.{key} must lie in (0, 1), got {}",
                    v.repr
                )),
            },
        }
    };
    let alpha = ranged("alpha", alpha, 0.05)?;
    let beta = ranged("beta", beta, 0.2)?;
    let margin = match margin {
        None => 0.0,
        Some(v) => match v.number {
            Some(n) if n.is_finite() => n,
            _ => {
                return Err(format!("selection.margin must be finite, got {}", v.repr));
            }
        },
    };
    if let Some(view) = n {
        check_positive_int("selection", "N", view)?;
    }
    if let Some(view) = resamples {
        check_positive_int("selection", "resamples", view)?;
    }
    if let Some(view) = seed {
        check_nonnegative_int("selection", "seed", view)?;
    }
    Ok([pilot_s, delta, alpha, beta, margin])
}

/// `_parse_telemetry` decisions (oracle `_rc_parse_aux.py:142-181`).
fn validate_telemetry(
    unknown: &UnknownView,
    mlflow_enabled: &Option<(bool, String)>,
    verbose_enabled: &Option<(bool, String)>,
    verbose_interval_ms: &Option<IntView>,
    profiler_captures: &Option<IntView>,
) -> Result<(), String> {
    check_unknown(unknown)?;
    if let Some((is_bool, repr)) = mlflow_enabled
        && !is_bool
    {
        return Err(format!(
            "telemetry.mlflow_enabled must be a bool, got {repr}"
        ));
    }
    if let Some((is_bool, repr)) = verbose_enabled
        && !is_bool
    {
        return Err(format!(
            "telemetry.verbose_enabled must be a bool, got {repr}"
        ));
    }
    if let Some(view) = verbose_interval_ms {
        let ok = view.is_int && matches!(view.as_i64, Some(20) | Some(50));
        if !ok {
            return Err(format!(
                "telemetry.verbose_interval_ms must be 20 or 50, got {}",
                view.repr
            ));
        }
    }
    if let Some(view) = profiler_captures {
        let ok = view.is_int && matches!(view.as_i64, Some(0..=16));
        if !ok {
            return Err(format!(
                "telemetry.profiler_captures must be an int in [0, 16], got {}",
                view.repr
            ));
        }
    }
    Ok(())
}

/// `_parse_mirror` decisions (oracle `_rc_parse_aux.py:184-202`).
fn validate_mirror(
    unknown: &UnknownView,
    enabled: &Option<(bool, String)>,
    project: &OptStrState,
    task_name: &OptStrState,
    offline_dir: &OptStrState,
) -> Result<(), String> {
    check_unknown(unknown)?;
    if let Some((is_bool, repr)) = enabled
        && !is_bool
    {
        return Err(format!("mirror.enabled must be a bool, got {repr}"));
    }
    if matches!(project, OptStrState::Value { ok: false }) {
        return Err("mirror.project must be a non-empty string".to_owned());
    }
    if matches!(task_name, OptStrState::Value { ok: false }) {
        return Err("mirror.task_name must be null or a non-empty string".to_owned());
    }
    if matches!(offline_dir, OptStrState::Value { ok: false }) {
        return Err("mirror.offline_dir must be null or a non-empty string".to_owned());
    }
    Ok(())
}

/// `_parse_eval` decisions (oracle `_rc_parse_aux.py:205-223`): note the
/// oracle gates `walls_dir` BEFORE the int slots.
fn validate_eval(
    unknown: &UnknownView,
    walls_dir: &OptStrState,
    frequency_updates: &Option<IntView>,
    num_batches: &Option<IntView>,
    microbatch_size: &Option<IntView>,
) -> Result<(), String> {
    check_unknown(unknown)?;
    if matches!(walls_dir, OptStrState::Value { ok: false }) {
        return Err("eval.walls_dir must be null or a non-empty string".to_owned());
    }
    if let Some(view) = frequency_updates {
        check_positive_int("eval", "frequency_updates", view)?;
    }
    if let Some(view) = num_batches {
        check_positive_int("eval", "num_batches", view)?;
    }
    if let Some(view) = microbatch_size {
        check_positive_int("eval", "microbatch_size", view)?;
    }
    Ok(())
}

/// `_parse_output` decisions (oracle `_rc_parse_aux.py:226-235`).
fn validate_output(
    unknown: &UnknownView,
    artifact_root: &OptStrState,
    run_id: &RunIdState,
) -> Result<(), String> {
    check_unknown(unknown)?;
    if matches!(artifact_root, OptStrState::Value { ok: false }) {
        return Err("output.artifact_root must be null or a non-empty string".to_owned());
    }
    if let RunIdState::Value { repr, ok: false } = run_id {
        return Err(format!(
            "output.run_id must match [A-Za-z0-9][A-Za-z0-9_-]*, got {repr}"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Attached wrap helpers (dict assembly only; run after the detach succeeds).
// ---------------------------------------------------------------------------

/// Emit one optional int slot: the validated original, or the oracle default.
fn emit_opt_int(
    out: &Bound<'_, PyDict>,
    key: &str,
    handle: Option<Py<PyAny>>,
    default: i64,
) -> PyResult<()> {
    match handle {
        Some(h) => out.set_item(key, h)?,
        None => out.set_item(key, default)?,
    }
    Ok(())
}

/// Emit one optional bool slot: the validated original, or the default.
fn emit_opt_bool(
    out: &Bound<'_, PyDict>,
    key: &str,
    handle: Option<Py<PyAny>>,
    default: bool,
) -> PyResult<()> {
    match handle {
        Some(h) => out.set_item(key, h)?,
        None => out.set_item(key, default)?,
    }
    Ok(())
}

/// Emit one optional string slot: the validated original, the oracle literal
/// default `&str`, or null.
fn emit_opt_str(
    py: Python<'_>,
    out: &Bound<'_, PyDict>,
    key: &str,
    state: &OptStrState,
    handle: Option<Py<PyAny>>,
    default_str: Option<&str>,
) -> PyResult<()> {
    match (state, handle) {
        (OptStrState::Value { .. }, Some(h)) => out.set_item(key, h)?,
        (_, _) => match default_str {
            Some(text) => out.set_item(key, text)?,
            None => out.set_item(key, py.None())?,
        },
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Pyfns (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// `_parse_seeds` gate (oracle `_rc_parse_aux.py:36-48`).
#[pyfunction]
fn rc_parse_seeds(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    const OWNER: &str = "rc_parse_seeds";
    let dict = as_section_mapping(&raw, "seeds")?;
    let unknown = stage_unknown(&dict, &SEEDS_ALLOWED, "seeds")?;
    let (data_view, data_handle) = stage_opt_int(&dict, "data_seed", OWNER)?;
    let (train_view, train_handle) = stage_opt_int(&dict, "train_seed", OWNER)?;
    let (selection_view, selection_handle) = stage_opt_int(&dict, "selection_seed", OWNER)?;
    py.detach(|| validate_seeds(&unknown, &data_view, &train_view, &selection_view))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    emit_opt_int(&out, "data_seed", data_handle, 0)?;
    emit_opt_int(&out, "train_seed", train_handle, 0)?;
    emit_opt_int(&out, "selection_seed", selection_handle, 0)?;
    Ok(out.unbind())
}

/// `_parse_selection` gate (oracle `_rc_parse_aux.py:51-121`; the
/// `eval.statistics` authoritative construction at `:122-138` stays Python).
#[pyfunction]
fn rc_parse_selection(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    const OWNER: &str = "rc_parse_selection";
    let dict = as_section_mapping(&raw, "selection")?;
    let unknown = stage_unknown(&dict, &SELECTION_ALLOWED, "selection")?;
    let (design_state, design_handle) = stage_design(&dict, OWNER)?;
    let (peeks_check, peeks_handles) = stage_peeks(&dict, OWNER)?;
    let pilot_s_view = stage_opt_num(&dict, "pilot_s", OWNER)?;
    let delta_view = stage_opt_num(&dict, "delta", OWNER)?;
    let alpha_view = stage_opt_num(&dict, "alpha", OWNER)?;
    let beta_view = stage_opt_num(&dict, "beta", OWNER)?;
    let margin_view = stage_opt_num(&dict, "margin", OWNER)?;
    let (n_view, n_handle) = stage_opt_int(&dict, "N", OWNER)?;
    let (resamples_view, resamples_handle) = stage_opt_int(&dict, "resamples", OWNER)?;
    let (seed_view, seed_handle) = stage_opt_int(&dict, "seed", OWNER)?;
    let floats = py
        .detach(|| {
            validate_selection(
                &unknown,
                &design_state,
                &peeks_check,
                &pilot_s_view,
                &delta_view,
                &alpha_view,
                &beta_view,
                &margin_view,
                &n_view,
                &resamples_view,
                &seed_view,
            )
        })
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    emit_opt_int(&out, "N", n_handle, 30)?;
    out.set_item("pilot_s", floats[0])?;
    out.set_item("delta", floats[1])?;
    out.set_item("alpha", floats[2])?;
    out.set_item("beta", floats[3])?;
    match design_handle {
        Some(h) => out.set_item("design", h)?,
        None => out.set_item("design", "fixed_n")?,
    }
    match peeks_check {
        Some(_) => out.set_item("declared_peeks", PyTuple::new(py, peeks_handles)?)?,
        None => out.set_item("declared_peeks", PyTuple::new(py, [30])?)?,
    }
    out.set_item("margin", floats[4])?;
    emit_opt_int(&out, "resamples", resamples_handle, 2000)?;
    emit_opt_int(&out, "seed", seed_handle, 0)?;
    Ok(out.unbind())
}

/// `_parse_telemetry` gate (oracle `_rc_parse_aux.py:142-181`).
#[pyfunction]
fn rc_parse_telemetry(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    const OWNER: &str = "rc_parse_telemetry";
    let dict = as_section_mapping(&raw, "telemetry")?;
    let unknown = stage_unknown(&dict, &TELEMETRY_ALLOWED, "telemetry")?;
    let (mlflow_view, mlflow_handle) = stage_opt_bool(&dict, "mlflow_enabled", OWNER)?;
    let (verbose_view, verbose_handle) = stage_opt_bool(&dict, "verbose_enabled", OWNER)?;
    let (interval_view, interval_handle) = stage_opt_int(&dict, "verbose_interval_ms", OWNER)?;
    let (captures_view, captures_handle) = stage_opt_int(&dict, "profiler_captures", OWNER)?;
    py.detach(|| {
        validate_telemetry(
            &unknown,
            &mlflow_view,
            &verbose_view,
            &interval_view,
            &captures_view,
        )
    })
    .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    emit_opt_bool(&out, "mlflow_enabled", mlflow_handle, true)?;
    emit_opt_bool(&out, "verbose_enabled", verbose_handle, false)?;
    emit_opt_int(&out, "verbose_interval_ms", interval_handle, 50)?;
    emit_opt_int(&out, "profiler_captures", captures_handle, 0)?;
    Ok(out.unbind())
}

/// `_parse_mirror` gate (oracle `_rc_parse_aux.py:184-202`).
#[pyfunction]
fn rc_parse_mirror(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    const OWNER: &str = "rc_parse_mirror";
    let dict = as_section_mapping(&raw, "mirror")?;
    let unknown = stage_unknown(&dict, &MIRROR_ALLOWED, "mirror")?;
    let (enabled_view, enabled_handle) = stage_opt_bool(&dict, "enabled", OWNER)?;
    let (project_state, project_handle) = stage_opt_str(&dict, "project", OWNER, false)?;
    let (task_state, task_handle) = stage_opt_str(&dict, "task_name", OWNER, false)?;
    let (offline_state, offline_handle) = stage_opt_str(&dict, "offline_dir", OWNER, false)?;
    py.detach(|| {
        validate_mirror(
            &unknown,
            &enabled_view,
            &project_state,
            &task_state,
            &offline_state,
        )
    })
    .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    emit_opt_bool(&out, "enabled", enabled_handle, false)?;
    emit_opt_str(
        py,
        &out,
        "project",
        &project_state,
        project_handle,
        Some("hydra2-tenhou-4p"),
    )?;
    emit_opt_str(py, &out, "task_name", &task_state, task_handle, None)?;
    emit_opt_str(
        py,
        &out,
        "offline_dir",
        &offline_state,
        offline_handle,
        None,
    )?;
    Ok(out.unbind())
}

/// `_parse_eval` gate (oracle `_rc_parse_aux.py:205-223`).
#[pyfunction]
fn rc_parse_eval(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    const OWNER: &str = "rc_parse_eval";
    let dict = as_section_mapping(&raw, "eval")?;
    let unknown = stage_unknown(&dict, &EVAL_ALLOWED, "eval")?;
    let (walls_state, walls_handle) = stage_opt_str(&dict, "walls_dir", OWNER, false)?;
    let (freq_view, freq_handle) = stage_opt_int(&dict, "frequency_updates", OWNER)?;
    let (batches_view, batches_handle) = stage_opt_int(&dict, "num_batches", OWNER)?;
    let micro_slot = slot_of(&dict, "microbatch_size", OWNER)?;
    let (micro_view, micro_handle): (Option<IntView>, Option<Py<PyAny>>) = match micro_slot {
        None => (None, None),
        Some(obj) => {
            if obj.is_none() {
                (None, None)
            } else {
                let view = stage_int(&obj)?;
                (Some(view), Some(obj.unbind()))
            }
        }
    };
    py.detach(|| {
        validate_eval(
            &unknown,
            &walls_state,
            &freq_view,
            &batches_view,
            &micro_view,
        )
    })
    .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    emit_opt_int(&out, "frequency_updates", freq_handle, 500)?;
    emit_opt_int(&out, "num_batches", batches_handle, 10)?;
    emit_opt_str(py, &out, "walls_dir", &walls_state, walls_handle, None)?;
    match micro_handle {
        Some(h) => out.set_item("microbatch_size", h)?,
        None => out.set_item("microbatch_size", py.None())?,
    }
    Ok(out.unbind())
}

/// `_parse_output` gate (oracle `_rc_parse_aux.py:226-236`).
#[pyfunction]
fn rc_parse_output(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    const OWNER: &str = "rc_parse_output";
    let dict = as_section_mapping(&raw, "output")?;
    let unknown = stage_unknown(&dict, &OUTPUT_ALLOWED, "output")?;
    let (artifact_state, artifact_handle) = stage_opt_str(&dict, "artifact_root", OWNER, true)?;
    let (run_id_state, run_id_handle) = stage_run_id(&dict, OWNER)?;
    py.detach(|| validate_output(&unknown, &artifact_state, &run_id_state))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    emit_opt_str(
        py,
        &out,
        "artifact_root",
        &artifact_state,
        artifact_handle,
        None,
    )?;
    match (&run_id_state, run_id_handle) {
        (RunIdState::Value { .. }, Some(h)) => out.set_item("run_id", h)?,
        (_, _) => out.set_item("run_id", py.None())?,
    }
    Ok(out.unbind())
}

/// Register the auxiliary section parsers on the shared `contracts`
/// submodule (mirrors `rc_require.rs:849-862`); MAIN calls this from
/// `contracts::register` — no new submodule, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(rc_parse_seeds, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_parse_selection, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_parse_telemetry, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_parse_mirror, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_parse_eval, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_parse_output, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty selection staging: `(state, peeks, 5×num, 3×int)`.
    type EmptySelection = (
        DesignState,
        Option<PeeksCheck>,
        Option<NumView>,
        Option<NumView>,
        Option<NumView>,
        Option<NumView>,
        Option<NumView>,
        Option<IntView>,
        Option<IntView>,
        Option<IntView>,
    );

    fn unknown_view(
        where_: &str,
        allowed: &[&str],
        keys: Vec<(String, Option<String>)>,
    ) -> UnknownView {
        UnknownView {
            keys,
            allowed: allowed.iter().map(|s| s.to_string()).collect(),
            where_: where_.to_owned(),
        }
    }

    fn seeds_unknown() -> UnknownView {
        unknown_view(
            "seeds",
            &SEEDS_ALLOWED,
            vec![("'zzz'".to_owned(), Some("zzz".to_owned()))],
        )
    }

    fn int_view(repr: &str, is_int: bool, as_i64: Option<i64>, negative: bool) -> IntView {
        IntView {
            repr: repr.to_owned(),
            is_int,
            as_i64,
            negative,
        }
    }

    fn num_view(repr: &str, number: Option<f64>) -> NumView {
        NumView {
            repr: repr.to_owned(),
            number,
        }
    }

    fn all_none_selection() -> EmptySelection {
        (
            DesignState::Missing,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    #[test]
    fn seeds_unknown_keys_exact() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:37` via
        // `rc_require` `_reject_unknown` (allowed sorted, unknown sorted).
        let err = validate_seeds(&seeds_unknown(), &None, &None, &None).expect_err("must reject");
        assert_eq!(
            err,
            "seeds has unknown keys ['zzz']; allowed=['data_seed', 'selection_seed', 'train_seed']"
        );
    }

    #[test]
    fn seeds_negative_rejected() {
        let bad = Some(int_view("-1", true, Some(-1), true));
        let err = validate_seeds(
            &unknown_view("seeds", &SEEDS_ALLOWED, vec![]),
            &bad,
            &None,
            &None,
        )
        .expect_err("must reject");
        assert_eq!(err, "seeds.data_seed must be a non-negative int, got -1");
    }

    #[test]
    fn seeds_bool_rejected() {
        // Oracle excludes bool before the int check, same gate message.
        let bad = Some(int_view("True", false, None, false));
        let err = validate_seeds(
            &unknown_view("seeds", &SEEDS_ALLOWED, vec![]),
            &None,
            &bad,
            &None,
        )
        .expect_err("must reject");
        assert_eq!(err, "seeds.train_seed must be a non-negative int, got True");
    }

    #[test]
    fn seeds_defaults_accept() {
        assert_eq!(
            validate_seeds(
                &unknown_view("seeds", &SEEDS_ALLOWED, vec![]),
                &None,
                &None,
                &None
            ),
            Ok(())
        );
    }

    #[test]
    fn selection_design_exact() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:69-72`.
        let (d, p, ps, dl, a, b, m, n, r, s) = all_none_selection();
        let design = DesignState::Value {
            repr: "'quux'".to_owned(),
            ok: false,
        };
        let err = validate_selection(
            &unknown_view("selection", &SELECTION_ALLOWED, vec![]),
            &design,
            &p,
            &ps,
            &dl,
            &a,
            &b,
            &m,
            &n,
            &r,
            &s,
        )
        .expect_err("must reject");
        assert_eq!(
            err,
            "selection.design must be one of ['fixed_n', 'time_uniform_cs'], got 'quux'"
        );
        let _ = d;
    }

    #[test]
    fn selection_design_nonstring_repr() {
        let (_, p, ps, dl, a, b, m, n, r, s) = all_none_selection();
        let design = DesignState::Value {
            repr: "7".to_owned(),
            ok: false,
        };
        let err = validate_selection(
            &unknown_view("selection", &SELECTION_ALLOWED, vec![]),
            &design,
            &p,
            &ps,
            &dl,
            &a,
            &b,
            &m,
            &n,
            &r,
            &s,
        )
        .expect_err("must reject");
        assert_eq!(
            err,
            "selection.design must be one of ['fixed_n', 'time_uniform_cs'], got 7"
        );
    }

    #[test]
    fn selection_peeks_empty_static() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:74-75`.
        let (_, _, ps, dl, a, b, m, n, r, s) = all_none_selection();
        let peeks = Some(PeeksCheck {
            shape_ok: false,
            tuple_repr: String::new(),
            elems: Vec::new(),
        });
        let err = validate_selection(
            &unknown_view("selection", &SELECTION_ALLOWED, vec![]),
            &DesignState::Missing,
            &peeks,
            &ps,
            &dl,
            &a,
            &b,
            &m,
            &n,
            &r,
            &s,
        )
        .expect_err("must reject");
        assert_eq!(
            err,
            "selection.declared_peeks must be a non-empty list of positive ints"
        );
    }

    #[test]
    fn selection_peeks_entries_tuple_repr() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:77-81`: the error
        // renders the TUPLE repr, not the raw list repr.
        let (_, _, ps, dl, a, b, m, n, r, s) = all_none_selection();
        let peeks = Some(PeeksCheck {
            shape_ok: true,
            tuple_repr: "(30, 0)".to_owned(),
            elems: vec![
                int_view("30", true, Some(30), false),
                int_view("0", true, Some(0), false),
            ],
        });
        let err = validate_selection(
            &unknown_view("selection", &SELECTION_ALLOWED, vec![]),
            &DesignState::Missing,
            &peeks,
            &ps,
            &dl,
            &a,
            &b,
            &m,
            &n,
            &r,
            &s,
        )
        .expect_err("must reject");
        assert_eq!(
            err,
            "selection.declared_peeks entries must be positive ints, got (30, 0)"
        );
    }

    #[test]
    fn selection_peeks_single_tuple_repr() {
        let (_, _, ps, dl, a, b, m, n, r, s) = all_none_selection();
        let peeks = Some(PeeksCheck {
            shape_ok: true,
            tuple_repr: "(0,)".to_owned(),
            elems: vec![int_view("0", true, Some(0), false)],
        });
        let err = validate_selection(
            &unknown_view("selection", &SELECTION_ALLOWED, vec![]),
            &DesignState::Missing,
            &peeks,
            &ps,
            &dl,
            &a,
            &b,
            &m,
            &n,
            &r,
            &s,
        )
        .expect_err("must reject");
        assert_eq!(
            err,
            "selection.declared_peeks entries must be positive ints, got (0,)"
        );
    }

    #[test]
    fn selection_pilot_zero_rejected() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:82-91`.
        let (_, p, _, dl, a, b, m, n, r, s) = all_none_selection();
        let pilot = Some(num_view("0", Some(0.0)));
        let err = validate_selection(
            &unknown_view("selection", &SELECTION_ALLOWED, vec![]),
            &DesignState::Missing,
            &p,
            &pilot,
            &dl,
            &a,
            &b,
            &m,
            &n,
            &r,
            &s,
        )
        .expect_err("must reject");
        assert_eq!(err, "selection.pilot_s must be positive and finite, got 0");
    }

    #[test]
    fn selection_alpha_boundary_rejected() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:92-99`
        // (`0.0 < value < 1.0`, so exactly 1.0 fails).
        let (_, p, ps, dl, _, b, m, n, r, s) = all_none_selection();
        let alpha = Some(num_view("1.0", Some(1.0)));
        let err = validate_selection(
            &unknown_view("selection", &SELECTION_ALLOWED, vec![]),
            &DesignState::Missing,
            &p,
            &ps,
            &dl,
            &alpha,
            &b,
            &m,
            &n,
            &r,
            &s,
        )
        .expect_err("must reject");
        assert_eq!(err, "selection.alpha must lie in (0, 1), got 1.0");
    }

    #[test]
    fn selection_margin_nan_rejected() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:100-107`.
        let (_, p, ps, dl, a, b, _, n, r, s) = all_none_selection();
        let margin = Some(num_view("nan", Some(f64::NAN)));
        let err = validate_selection(
            &unknown_view("selection", &SELECTION_ALLOWED, vec![]),
            &DesignState::Missing,
            &p,
            &ps,
            &dl,
            &a,
            &b,
            &margin,
            &n,
            &r,
            &s,
        )
        .expect_err("must reject");
        assert_eq!(err, "selection.margin must be finite, got nan");
    }

    #[test]
    fn selection_defaults_widen() {
        // All-absent selection widens to the oracle defaults
        // (`:108-121`: 1.5/0.5/0.05/0.2/0.0).
        let (d, p, ps, dl, a, b, m, n, r, s) = all_none_selection();
        assert_eq!(
            validate_selection(
                &unknown_view("selection", &SELECTION_ALLOWED, vec![]),
                &d,
                &p,
                &ps,
                &dl,
                &a,
                &b,
                &m,
                &n,
                &r,
                &s,
            ),
            Ok([1.5, 0.5, 0.05, 0.2, 0.0])
        );
    }

    #[test]
    fn selection_order_design_before_n() {
        // Oracle checks design (`:68-72`) before the `N` int gate (`:109`):
        // the FIRST error on a doubly-bad input is the design one.
        let (_, p, ps, dl, a, b, m, _, r, s) = all_none_selection();
        let design = DesignState::Value {
            repr: "'quux'".to_owned(),
            ok: false,
        };
        let n = Some(int_view("-4", true, Some(-4), true));
        let err = validate_selection(
            &unknown_view("selection", &SELECTION_ALLOWED, vec![]),
            &design,
            &p,
            &ps,
            &dl,
            &a,
            &b,
            &m,
            &n,
            &r,
            &s,
        )
        .expect_err("must reject");
        assert_eq!(
            err,
            "selection.design must be one of ['fixed_n', 'time_uniform_cs'], got 'quux'"
        );
    }

    #[test]
    fn telemetry_interval_float_rejected() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:154-164`: bool
        // excluded, `int` only — `50.0` fails even though it equals 50.
        let interval = Some(int_view("50.0", false, None, false));
        let err = validate_telemetry(
            &unknown_view("telemetry", &TELEMETRY_ALLOWED, vec![]),
            &None,
            &None,
            &interval,
            &None,
        )
        .expect_err("must reject");
        assert_eq!(
            err,
            "telemetry.verbose_interval_ms must be 20 or 50, got 50.0"
        );
    }

    #[test]
    fn telemetry_captures_range_rejected() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:165-175`.
        let captures = Some(int_view("17", true, Some(17), false));
        let err = validate_telemetry(
            &unknown_view("telemetry", &TELEMETRY_ALLOWED, vec![]),
            &None,
            &None,
            &None,
            &captures,
        )
        .expect_err("must reject");
        assert_eq!(
            err,
            "telemetry.profiler_captures must be an int in [0, 16], got 17"
        );
    }

    #[test]
    fn telemetry_bool_gate() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:148-150`.
        let mlflow: Option<(bool, String)> = Some((false, "1".to_owned()));
        let err = validate_telemetry(
            &unknown_view("telemetry", &TELEMETRY_ALLOWED, vec![]),
            &mlflow,
            &None,
            &None,
            &None,
        )
        .expect_err("must reject");
        assert_eq!(err, "telemetry.mlflow_enabled must be a bool, got 1");
    }

    #[test]
    fn mirror_statics() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:191-199`
        // (static texts carry no value).
        let enabled: Option<(bool, String)> = Some((true, "True".to_owned()));
        let err = validate_mirror(
            &unknown_view("mirror", &MIRROR_ALLOWED, vec![]),
            &enabled,
            &OptStrState::Value { ok: false },
            &OptStrState::Null,
            &OptStrState::Missing,
        )
        .expect_err("must reject");
        assert_eq!(err, "mirror.project must be a non-empty string");
        let err = validate_mirror(
            &unknown_view("mirror", &MIRROR_ALLOWED, vec![]),
            &enabled,
            &OptStrState::Value { ok: true },
            &OptStrState::Value { ok: false },
            &OptStrState::Missing,
        )
        .expect_err("must reject");
        assert_eq!(err, "mirror.task_name must be null or a non-empty string");
    }

    #[test]
    fn eval_walls_before_ints() {
        // Oracle gates `walls_dir` (`:209-211`) BEFORE `frequency_updates`
        // (`:213-215`): the FIRST error on a doubly-bad input is walls.
        let freq = Some(int_view("-2", true, Some(-2), true));
        let err = validate_eval(
            &unknown_view("eval", &EVAL_ALLOWED, vec![]),
            &OptStrState::Value { ok: false },
            &freq,
            &None,
            &None,
        )
        .expect_err("must reject");
        assert_eq!(err, "eval.walls_dir must be null or a non-empty string");
    }

    #[test]
    fn run_id_shape_exact() {
        // Hand-derived from the frozen pattern `rc_sections.rs:79`
        // (`[A-Za-z0-9][A-Za-z0-9_-]*` fullmatch).
        assert!(is_run_id_shape("wp14-run-001"));
        assert!(is_run_id_shape("0"));
        assert!(is_run_id_shape("A1_-"));
        assert!(!is_run_id_shape(""));
        assert!(!is_run_id_shape("-x"));
        assert!(!is_run_id_shape("a b"));
        assert!(!is_run_id_shape("a!"));
    }

    #[test]
    fn output_run_id_text() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:233-235`.
        let run_id = RunIdState::Value {
            repr: "'bad id!'".to_owned(),
            ok: false,
        };
        let err = validate_output(
            &unknown_view("output", &OUTPUT_ALLOWED, vec![]),
            &OptStrState::Null,
            &run_id,
        )
        .expect_err("must reject");
        assert_eq!(
            err,
            "output.run_id must match [A-Za-z0-9][A-Za-z0-9_-]*, got 'bad id!'"
        );
    }

    #[test]
    fn output_artifact_static() {
        // Hand-derived from HEAD oracle `_rc_parse_aux.py:228-232`.
        let err = validate_output(
            &unknown_view("output", &OUTPUT_ALLOWED, vec![]),
            &OptStrState::Value { ok: false },
            &RunIdState::Missing,
        )
        .expect_err("must reject");
        assert_eq!(
            err,
            "output.artifact_root must be null or a non-empty string"
        );
    }
}
