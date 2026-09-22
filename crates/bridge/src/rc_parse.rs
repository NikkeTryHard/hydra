//! rc_parse: run-config core section parsers on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (no feed/shard/search edge; pure argument validation, so no
//! crate owner exists — same position as `rc_require.rs:1-9`; `crates/bridge/Cargo.toml:25`
//! carries the only dependency, no new dependency). FORBIDS: YAML/file IO,
//! `${VAR}` interpolation, base+override composition, root assembly and
//! cross-checks (all stay Python in `python/hydra2/training/_rc_root.py`),
//! wall-clock/RNG, and `ContractError` shaping (the bridge raises
//! `ValueError`/`TypeError`; thin Python translators in
//! `python/hydra2/training/_rc_parse.py` map to `ContractError` with
//! byte-identical messages).
//!
//! TABLE ported here — the 8 core section parsers, oracle
//! `src/hydra2/training/_rc_parse.py:60-425` (HEAD md5 `23f1974e…`, equals
//! worktree `python/hydra2/training/_rc_parse.py` at port time; pure rename):
//! - `rc_parse_run` ← `_parse_run` (:60-71)
//! - `rc_parse_data` ← `_parse_data` (:74-162)
//! - `rc_parse_model` ← `_parse_model` (:165-193; registry + baseline stay Python)
//! - `rc_parse_weights` ← `_parse_weights` (:196-230)
//! - `rc_parse_optimizer` ← `_parse_optimizer` (:233-282)
//! - `rc_parse_scheduler` ← `_parse_scheduler` (:285-314)
//! - `rc_parse_runtime` ← `_parse_runtime` (:317-360; protocol check stays Python)
//! - `rc_parse_loop` ← `_parse_loop` (:363-425)
//!
//! Each pyfn takes the raw section mapping positionally and returns a plain
//! validated dict the Python translator builds a dataclass from; unknown keys
//! fail closed first (same gate as `rc_require::rc_reject_unknown`, oracle
//! `_rc_require.py:90-96`). Gate-equivalent checks restate the wave-4
//! `rc_require.rs` check strings (proven byte-identical there against the same
//! HEAD oracle), cited per use.
//!
//! LOGIC staying Python: file IO/YAML/entry orchestration (`_rc_root.py`),
//! the `KNOWN_ARCHITECTURES` registry + `BASELINE_ACTION_COUNT` equality
//! (torch-owned `models/schema.py` patch-point), the `validate_runtime_spec`
//! protocol check (import-guarded `runtime/protocol.py` patch-point), all 8
//! `ContractError` translators, `__all__`, and every dataclass.
//!
//! Shape per fn: attached staging (repr/type-name/strip/`float()` via the live
//! Python API, so `{value!r}`, `type(x).__name__`, Unicode `strip`, and
//! `float(x)` errors render byte-exact) → ONE `py.detach(|| …)` over owned
//! plain data with zero Python API inside (per `contracts.rs:434-437`,
//! `validate.rs:83-95`) → attached dict assembly (`PyDict::new` plus
//! `set_item`, per `contracts.rs:1227-1231`) with `PyValueError` wrap (per
//! `contracts.rs:117-123`). Bool is excluded before every int check (per
//! `contracts.rs:69-74`, `canon_rng.rs:279-286`); big ints past `i64` keep
//! their original handle on pass-through gates (per `rc_require.rs:516-538`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `validate.rs:111-127`); no new entry
//! point. MAIN wiring: `pub mod rc_parse;` in `lib.rs` plus
//! `crate::rc_parse::register(&sub)?;` in `contracts.rs` next to
//! `crate::rc_sections::register(&sub)?;` (`contracts.rs:1295`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyModule, PyString};

/// Frozen section vocabulary, restating `rc_sections.rs:45-72` (single-sourced
/// there; repeated here so each section renders its own `one of [...]` list
/// without a cross-module edge — shared modules are MAIN-owned).
pub(crate) const RUN_KINDS: [&str; 1] = ["supervised"];
pub(crate) const TENHOU_SCOPE: &str = "tenhou-4p-hanchan";
pub(crate) const OPTIMIZER_IDS: [&str; 3] = ["adamw", "adam", "sgd"];
pub(crate) const SCHEDULER_IDS: [&str; 3] = ["cosine", "constant", "linear"];
pub(crate) const ADAPTER_IDS: [&str; 1] = ["plain_pytorch"];
pub(crate) const RUNTIME_PRECISIONS: [&str; 2] = ["fp32", "bf16_mixed"];
pub(crate) const COMPILE_MODES: [&str; 4] = [
    "eager",
    "default",
    "max-autotune-no-cudagraphs",
    "max-autotune",
];
const REPLAY_BACKENDS: [&str; 1] = ["rust"];

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (restates `rc_require.rs:63-65`, same `canon_rng.rs:322-323` round-trip).
pub(crate) fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Attached staging helper: exact `type(value).__name__` text for the
/// `must be a mapping` slots (restates `rc_require.rs:70-72`).
fn py_type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.get_type().name()?.to_str()?.to_owned())
}

/// Render `['a', 'b']` exactly like Python `list(tuple_of_ids)`
/// (single-quote joins; restates the `py_str_list` idea in
/// `rc_sections.rs:133-137`).
pub(crate) fn one_of_list(ids: &[&str]) -> String {
    format!(
        "[{}]",
        ids.iter()
            .map(|s| format!("'{s}'"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Owned plain-data view of an int-or-other Python value (restates
/// `rc_require.rs:79-104`: bool excluded first, big ints keep sign via repr).
pub(crate) struct IntView {
    pub(crate) repr: String,
    pub(crate) is_int: bool,
    pub(crate) as_i64: Option<i64>,
    pub(crate) negative: bool,
}

pub(crate) fn stage_int(obj: &Bound<'_, PyAny>) -> PyResult<IntView> {
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

/// Owned plain-data view of a number-or-other value (restates
/// `rc_require.rs:121-138`; overflow propagates at stage time, same order the
/// oracle evaluates `float(value)` in).
pub(crate) struct NumView {
    pub(crate) repr: String,
    pub(crate) number: Option<f64>,
}

pub(crate) fn stage_num(obj: &Bound<'_, PyAny>) -> PyResult<NumView> {
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

/// One staged entry of the `{name: mult}` maps (restates
/// `rc_require.rs:168-184`).
pub(crate) struct MapEntry {
    pub(crate) key: String,
    pub(crate) key_repr: String,
    pub(crate) key_ok: bool,
    pub(crate) number: Option<f64>,
}

pub(crate) fn stage_map_number(obj: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(None);
    }
    if !(obj.is_instance_of::<PyInt>() || obj.is_instance_of::<PyFloat>()) {
        return Ok(None);
    }
    obj.extract::<f64>().map(Some)
}

/// Digest-shape test (restates `rc_require.rs:189-197`, oracle
/// `_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")` via `fullmatch`).
fn is_digest_shape(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 7 + 64 || &bytes[..7] != b"sha256:" {
        return false;
    }
    bytes[7..]
        .iter()
        .all(|c| c.is_ascii_digit() || matches!(c, b'a'..=b'f'))
}

/// Run-id shape test (oracle `_RUN_ID_RE` pattern text
/// `rc_sections.rs:79` `[A-Za-z0-9][A-Za-z0-9_-]*` via `fullmatch`).
pub(crate) fn run_id_ok(text: &str) -> bool {
    let mut chars = text.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphanumeric() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// CUDA-device shape test (oracle `_CUDA_DEVICE_RE` pattern text
/// `rc_sections.rs:81` `^cuda(:[0-9]+)?$` via `fullmatch`).
pub(crate) fn cuda_device_ok(text: &str) -> bool {
    if text == "cuda" {
        return true;
    }
    if let Some(suffix) = text.strip_prefix("cuda:") {
        return !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit());
    }
    false
}

// ---------------------------------------------------------------------------
// Detached checks (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// Staged keys plus allowlist (restates `rc_require.rs:376-380`).
pub(crate) struct UnknownView {
    pub(crate) keys: Vec<(String, Option<String>)>,
    pub(crate) allowed: Vec<String>,
    pub(crate) where_: String,
}

pub(crate) fn check_unknown(view: &UnknownView) -> Result<(), String> {
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
        // Same guard as `rc_require.rs:405-412`: the oracle's `sorted()`
        // raises TypeError over mixed incomparable keys.
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

pub(crate) fn check_positive_int(where_: &str, key: &str, view: &IntView) -> Result<(), String> {
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

pub(crate) fn check_bounded_int(
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

pub(crate) fn check_nonnegative_int(where_: &str, key: &str, view: &IntView) -> Result<(), String> {
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

pub(crate) fn check_nonnegative_float(
    where_: &str,
    key: &str,
    view: &NumView,
) -> Result<f64, String> {
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

/// Strict float ends; `lo_repr`/`hi_repr` are the Python `str()` texts of the
/// frozen bounds (oracle `f"{'(' if lo_open else '['}{lo}, {hi}{')' if hi_open
/// else ']'}"`), so `[0.0, 1.0)`-style bounds render byte-exact.
pub(crate) struct FloatEnds {
    pub(crate) lo: f64,
    pub(crate) lo_repr: &'static str,
    pub(crate) lo_open: bool,
    pub(crate) hi: f64,
    pub(crate) hi_repr: &'static str,
    pub(crate) hi_open: bool,
}

pub(crate) fn check_bounded_float(
    where_: &str,
    key: &str,
    view: &NumView,
    ends: &FloatEnds,
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
    let lo_ok = if ends.lo_open {
        ends.lo < number
    } else {
        ends.lo <= number
    };
    let hi_ok = if ends.hi_open {
        number < ends.hi
    } else {
        number <= ends.hi
    };
    if !(lo_ok && hi_ok) {
        let bound = format!(
            "{}{}, {}{}",
            if ends.lo_open { '(' } else { '[' },
            ends.lo_repr,
            ends.hi_repr,
            if ends.hi_open { ')' } else { ']' }
        );
        return Err(format!(
            "{where_}.{key} must lie in {bound}, got {}",
            view.repr
        ));
    }
    Ok(number)
}

pub(crate) fn check_weight_map(
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

pub(crate) fn check_positive_float_map(
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

// ---------------------------------------------------------------------------
// Attached staging helpers shared by the eight pyfns.
// ---------------------------------------------------------------------------

/// Read one field slot: missing vs present-None vs present-value.
pub(crate) fn field_slot<'a>(
    dict: &Bound<'a, PyDict>,
    key: &str,
    fn_tag: &str,
) -> PyResult<Option<Bound<'a, PyAny>>> {
    dict.get_item(key).map_err(|e| {
        PyValueError::new_err(format!("contracts {fn_tag} field {key:?} unreadable: {e}"))
    })
}

/// Stage the unknown-key view for one section (attached key walk, detached
/// decision — same split as `rc_require::rc_reject_unknown`).
pub(crate) fn stage_unknown(
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
        allowed: allowed.iter().map(ToString::to_string).collect(),
        where_: where_.to_owned(),
    })
}

/// Extract the section mapping or fail with the `{where} must be a mapping`
/// text (attached, per the `rc_reject_unknown:442-449` precedent).
pub(crate) fn section_dict<'py>(
    raw: &Bound<'py, PyAny>,
    where_: &str,
) -> PyResult<Bound<'py, PyDict>> {
    match raw.extract::<Bound<'py, PyDict>>() {
        Ok(dict) => Ok(dict),
        Err(_) => {
            let type_name = py_type_name(raw).unwrap_or_else(|_| "?".to_owned());
            Err(PyValueError::new_err(format!(
                "{where_} must be a mapping, got {type_name}"
            )))
        }
    }
}

/// Staged optional string slot: repr plus text when the slot holds a `str`.
/// `missing_text` feeds the oracle's `fields.get(key, default)` default.
pub(crate) struct OptStr {
    pub(crate) repr: String,
    pub(crate) text: Option<String>,
}

pub(crate) fn stage_opt_str(
    slot: Option<Bound<'_, PyAny>>,
    missing_text: &str,
) -> PyResult<OptStr> {
    match slot {
        None => Ok(OptStr {
            // Python-repr-style single quotes (`got ''` for a missing id);
            // missing slots otherwise read valid defaults, so this only
            // renders on the missing-`run.id` path (oracle `{run_id!r}`).
            repr: format!("'{missing_text}'"),
            text: Some(missing_text.to_owned()),
        }),
        Some(obj) if obj.is_instance_of::<PyString>() => {
            let text: String = obj.extract().map_err(|e| {
                PyValueError::new_err(format!("contracts rc_parse field unreadable: {e}"))
            })?;
            Ok(OptStr {
                repr: py_repr(&obj)?,
                text: Some(text),
            })
        }
        Some(obj) => Ok(OptStr {
            repr: py_repr(&obj)?,
            text: None,
        }),
    }
}

/// Staged digest pin: absent/null reads as `Absent` (restates
/// `rc_require.rs:800-844`).
pub(crate) enum PinKind {
    Absent,
    Text(String),
    Other,
}

pub(crate) struct PinStage {
    pub(crate) kind: PinKind,
    pub(crate) repr: String,
}

pub(crate) fn stage_pin(slot: Option<Bound<'_, PyAny>>) -> PyResult<PinStage> {
    match slot {
        None => Ok(PinStage {
            kind: PinKind::Absent,
            repr: String::new(),
        }),
        Some(obj) if obj.is_none() => Ok(PinStage {
            kind: PinKind::Absent,
            repr: String::new(),
        }),
        Some(obj) if obj.is_instance_of::<PyString>() => {
            let text: String = obj.extract().map_err(|e| {
                PyValueError::new_err(format!("contracts rc_parse field unreadable: {e}"))
            })?;
            Ok(PinStage {
                kind: PinKind::Text(text),
                repr: String::new(),
            })
        }
        Some(obj) => Ok(PinStage {
            kind: PinKind::Other,
            repr: py_repr(&obj)?,
        }),
    }
}

pub(crate) fn check_pin(
    where_: &str,
    key: &str,
    staged: &PinStage,
) -> Result<Option<String>, String> {
    match &staged.kind {
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
            staged.repr
        )),
    }
}

/// Staged strict-bool slot with a default for absent/null (restates the
/// `rc_require_loop_bool:758-790` decision; the literal `loop.` prefix is kept
/// by the caller).
pub(crate) struct BoolStage {
    pub(crate) value: Option<bool>,
    pub(crate) repr: String,
}

/// Staged `{name: mult}` map slot: absent/null reads as `None` (restates
/// `rc_require.rs:610-661` / `:704-756`).
pub(crate) struct MapStage {
    pub(crate) entries: Option<Vec<MapEntry>>,
    pub(crate) type_name: Option<String>,
}

pub(crate) fn stage_map(slot: Option<Bound<'_, PyAny>>) -> PyResult<MapStage> {
    let Some(value) = slot else {
        return Ok(MapStage {
            entries: None,
            type_name: None,
        });
    };
    if value.is_none() {
        return Ok(MapStage {
            entries: None,
            type_name: None,
        });
    }
    let Ok(dict) = value.extract::<Bound<'_, PyDict>>() else {
        let type_name = py_type_name(&value).unwrap_or_else(|_| "?".to_owned());
        return Ok(MapStage {
            entries: None,
            type_name: Some(type_name),
        });
    };
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
    Ok(MapStage {
        entries: Some(entries),
        type_name: None,
    })
}

/// Staged gated-int slot: present flag plus view and pass-through handle
/// (big ints that pass keep the original object, per `rc_require.rs:516-538`).
pub(crate) struct IntSlot {
    pub(crate) present: bool,
    pub(crate) view: IntView,
    pub(crate) handle: Option<Py<PyAny>>,
}

pub(crate) fn stage_int_slot(_py: Python<'_>, slot: Option<Bound<'_, PyAny>>) -> PyResult<IntSlot> {
    match slot {
        None => Ok(IntSlot {
            present: false,
            view: IntView {
                repr: "None".to_owned(),
                is_int: false,
                as_i64: None,
                negative: false,
            },
            handle: None,
        }),
        Some(obj) => {
            let view = stage_int(&obj)?;
            Ok(IntSlot {
                present: true,
                view,
                handle: Some(obj.unbind()),
            })
        }
    }
}

/// Detached int result: `Some(v)` fits `i64`, `None` is a huge int that
/// passed (the attached side reuses the staged handle).
pub(crate) type HugeInt = Option<i64>;

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

const RUN_ALLOWED: [&str; 3] = ["id", "kind", "description"];

struct RunStaged {
    unknown: UnknownView,
    run_id: OptStr,
    kind: OptStr,
    description: Option<String>,
}

/// `_parse_run` (oracle `_rc_parse.py:60-71`): identity plus kind gate.
#[pyfunction]
fn rc_parse_run(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let dict = section_dict(&raw, "run")?;
    let unknown = stage_unknown(&dict, &RUN_ALLOWED, "run")?;
    let run_id = stage_opt_str(field_slot(&dict, "id", "rc_parse_run")?, "")?;
    let kind = stage_opt_str(field_slot(&dict, "kind", "rc_parse_run")?, "supervised")?;
    let description_slot = field_slot(&dict, "description", "rc_parse_run")?;
    let description: Option<String> = match description_slot {
        None => Some(String::new()),
        Some(obj) if obj.is_instance_of::<PyString>() => Some(obj.extract().map_err(|e| {
            PyValueError::new_err(format!("contracts rc_parse_run field unreadable: {e}"))
        })?),
        Some(_) => None,
    };
    let staged = RunStaged {
        unknown,
        run_id,
        kind,
        description,
    };
    let valid = py
        .detach(|| validate_run(&staged))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("run_id", valid.0)?;
    out.set_item("kind", valid.1)?;
    out.set_item("description", valid.2)?;
    Ok(out.unbind())
}

fn validate_run(staged: &RunStaged) -> Result<(String, String, String), String> {
    check_unknown(&staged.unknown)?;
    let run_id = match &staged.run_id.text {
        Some(text) if run_id_ok(text) => text.clone(),
        _ => {
            return Err(format!(
                "run.id must match [A-Za-z0-9][A-Za-z0-9_-]*, got {}",
                staged.run_id.repr
            ));
        }
    };
    let kind = match &staged.kind.text {
        Some(text) if RUN_KINDS.contains(&text.as_str()) => text.clone(),
        _ => {
            return Err(format!(
                "run.kind must be one of {}, got {}",
                one_of_list(&RUN_KINDS),
                staged.kind.repr
            ));
        }
    };
    let Some(description) = &staged.description else {
        return Err("run.description must be a string".to_owned());
    };
    Ok((run_id, kind, description.clone()))
}

// ---------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------

const DATA_ALLOWED: [&str; 17] = [
    "root",
    "scope",
    "train_split",
    "val_split",
    "wall_disjoint",
    "leakage_check",
    "dora_width",
    "actor_privileged_split",
    "dataset_manifest_hash",
    "shuffle_buffer_size",
    "drop_last",
    "num_workers",
    "world_size",
    "replay_backend",
    "decode_prefetch",
    "expand_batch_games",
    "homogeneous_buckets",
];

pub(crate) struct FlagStage {
    is_true: bool,
}

struct DataStaged {
    unknown: UnknownView,
    root: Option<String>,
    scope: OptStr,
    train_split: OptStr,
    val_split: OptStr,
    wall_disjoint: FlagStage,
    leakage_check: FlagStage,
    actor_privileged_split: FlagStage,
    dora_repr: String,
    dora_is_five: bool,
    dora_present: bool,
    drop_last: BoolStage,
    homogeneous_buckets: BoolStage,
    replay_backend: OptStr,
    dataset_manifest_hash: PinStage,
    shuffle_buffer_size: IntSlot,
    num_workers: IntSlot,
    world_size: IntSlot,
    decode_prefetch: IntSlot,
    expand_batch_games: IntSlot,
}

struct DataValid {
    root: String,
    scope: String,
    train_split: String,
    val_split: String,
    dora: i64,
    drop_last: bool,
    homogeneous_buckets: bool,
    replay_backend: String,
    dataset_manifest_hash: Option<String>,
    shuffle_buffer_size: Option<HugeInt>,
    num_workers: Option<HugeInt>,
    world_size: Option<HugeInt>,
    decode_prefetch: Option<HugeInt>,
    expand_batch_games: Option<HugeInt>,
}

/// `_parse_data` (oracle `_rc_parse.py:74-162`): streaming source root plus
/// contract flags; gate-evaluated args run in constructor order.
#[pyfunction]
fn rc_parse_data(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let dict = section_dict(&raw, "data")?;
    let unknown = stage_unknown(&dict, &DATA_ALLOWED, "data")?;
    // `root`: nonempty gate (Unicode `strip`, per `rc_require.rs:484-505`) then
    // the absolute-path check on the staged text.
    let root_slot = field_slot(&dict, "root", "rc_parse_data")?;
    let root: Option<String> = match root_slot {
        None => None,
        Some(obj) if obj.is_instance_of::<PyString>() => {
            let text: String = obj.extract().map_err(|e| {
                PyValueError::new_err(format!("contracts rc_parse_data field unreadable: {e}"))
            })?;
            let stripped: String = obj
                .call_method0("strip")
                .map_err(|e| {
                    PyValueError::new_err(format!("contracts rc_parse_data field unreadable: {e}"))
                })?
                .extract()
                .map_err(|e| {
                    PyValueError::new_err(format!("contracts rc_parse_data field unreadable: {e}"))
                })?;
            if stripped.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        Some(_) => None,
    };
    let scope = stage_opt_str(field_slot(&dict, "scope", "rc_parse_data")?, TENHOU_SCOPE)?;
    let train_split = stage_opt_str(field_slot(&dict, "train_split", "rc_parse_data")?, "train")?;
    let val_split = stage_opt_str(
        field_slot(&dict, "val_split", "rc_parse_data")?,
        "validation",
    )?;
    let wall_disjoint = stage_flag(field_slot(&dict, "wall_disjoint", "rc_parse_data")?)?;
    let leakage_check = stage_flag(field_slot(&dict, "leakage_check", "rc_parse_data")?)?;
    let actor_privileged_split = stage_flag(field_slot(
        &dict,
        "actor_privileged_split",
        "rc_parse_data",
    )?)?;
    // `dora_width`: oracle `value != 5` (Python equality: `5.0` passes, bools
    // and every other shape reject).
    let dora_slot = field_slot(&dict, "dora_width", "rc_parse_data")?;
    let (dora_repr, dora_is_five, dora_present): (String, bool, bool) = match dora_slot {
        None => (String::new(), true, false),
        Some(obj) => {
            let repr = py_repr(&obj)?;
            let is_five = if obj.is_instance_of::<PyBool>() {
                false
            } else if let Ok(v) = obj.extract::<i64>() {
                v == 5
            } else if let Ok(v) = obj.extract::<f64>() {
                v == 5.0
            } else {
                false
            };
            (repr, is_five, true)
        }
    };
    let drop_last = stage_data_bool(field_slot(&dict, "drop_last", "rc_parse_data")?, true)?;
    let homogeneous_buckets = stage_data_bool(
        field_slot(&dict, "homogeneous_buckets", "rc_parse_data")?,
        false,
    )?;
    let replay_backend = stage_opt_str(
        field_slot(&dict, "replay_backend", "rc_parse_data")?,
        "rust",
    )?;
    let dataset_manifest_hash =
        stage_pin(field_slot(&dict, "dataset_manifest_hash", "rc_parse_data")?)?;
    let shuffle_buffer_size = stage_int_slot(
        py,
        field_slot(&dict, "shuffle_buffer_size", "rc_parse_data")?,
    )?;
    let num_workers = stage_int_slot(py, field_slot(&dict, "num_workers", "rc_parse_data")?)?;
    let world_size = stage_int_slot(py, field_slot(&dict, "world_size", "rc_parse_data")?)?;
    let decode_prefetch =
        stage_int_slot(py, field_slot(&dict, "decode_prefetch", "rc_parse_data")?)?;
    let expand_batch_games = stage_int_slot(
        py,
        field_slot(&dict, "expand_batch_games", "rc_parse_data")?,
    )?;
    let staged = DataStaged {
        unknown,
        root,
        scope,
        train_split,
        val_split,
        wall_disjoint,
        leakage_check,
        actor_privileged_split,
        dora_repr,
        dora_is_five,
        dora_present,
        drop_last,
        homogeneous_buckets,
        replay_backend,
        dataset_manifest_hash,
        shuffle_buffer_size,
        num_workers,
        world_size,
        decode_prefetch,
        expand_batch_games,
    };
    let valid = py
        .detach(|| validate_data(&staged))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("root", valid.root)?;
    out.set_item("scope", valid.scope)?;
    out.set_item("train_split", valid.train_split)?;
    out.set_item("val_split", valid.val_split)?;
    out.set_item("wall_disjoint", true)?;
    out.set_item("leakage_check", true)?;
    out.set_item("dora_width", valid.dora)?;
    out.set_item("actor_privileged_split", true)?;
    out.set_item("dataset_manifest_hash", valid.dataset_manifest_hash)?;
    set_int_field(
        &out,
        "shuffle_buffer_size",
        &valid.shuffle_buffer_size,
        10000,
        staged.shuffle_buffer_size.handle.as_ref(),
        py,
    )?;
    out.set_item("drop_last", valid.drop_last)?;
    set_int_field(
        &out,
        "num_workers",
        &valid.num_workers,
        0,
        staged.num_workers.handle.as_ref(),
        py,
    )?;
    set_int_field(
        &out,
        "world_size",
        &valid.world_size,
        1,
        staged.world_size.handle.as_ref(),
        py,
    )?;
    out.set_item("replay_backend", valid.replay_backend)?;
    set_int_field(
        &out,
        "decode_prefetch",
        &valid.decode_prefetch,
        64,
        staged.decode_prefetch.handle.as_ref(),
        py,
    )?;
    set_int_field(
        &out,
        "expand_batch_games",
        &valid.expand_batch_games,
        64,
        staged.expand_batch_games.handle.as_ref(),
        py,
    )?;
    out.set_item("homogeneous_buckets", valid.homogeneous_buckets)?;
    Ok(out.unbind())
}

/// `fields.get(flag, True) is not True` rejects everything but the `True`
/// singleton (including `1`).
pub(crate) fn stage_flag(slot: Option<Bound<'_, PyAny>>) -> PyResult<FlagStage> {
    match slot {
        None => Ok(FlagStage { is_true: true }),
        Some(obj) => {
            let is_true = obj.is_instance_of::<PyBool>() && obj.extract::<bool>().unwrap_or(false);
            Ok(FlagStage { is_true })
        }
    }
}

/// Strict data bool (absent → default; non-bool, including `None`, rejects at
/// validate time with its staged repr).
pub(crate) fn stage_data_bool(
    slot: Option<Bound<'_, PyAny>>,
    default: bool,
) -> PyResult<BoolStage> {
    match slot {
        None => Ok(BoolStage {
            value: Some(default),
            repr: String::new(),
        }),
        Some(obj) if obj.is_instance_of::<PyBool>() => {
            let value: bool = obj.extract().map_err(|e| {
                PyValueError::new_err(format!("contracts rc_parse field unreadable: {e}"))
            })?;
            Ok(BoolStage {
                value: Some(value),
                repr: String::new(),
            })
        }
        Some(obj) => Ok(BoolStage {
            value: None,
            repr: py_repr(&obj)?,
        }),
    }
}

/// Attach one gated int into the output dict: absent → default, huge → the
/// staged original handle, else the validated `i64`.
pub(crate) fn set_int_field(
    out: &Bound<'_, PyDict>,
    key: &str,
    valid: &Option<HugeInt>,
    default: i64,
    handle: Option<&Py<PyAny>>,
    py: Python<'_>,
) -> PyResult<()> {
    match valid {
        None => {
            out.set_item(key, default)?;
        }
        Some(Some(value)) => {
            out.set_item(key, *value)?;
        }
        Some(None) => {
            if let Some(handle) = handle {
                out.set_item(key, handle.bind(py))?;
            } else {
                out.set_item(key, default)?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn validate_data(staged: &DataStaged) -> Result<DataValid, String> {
    check_unknown(&staged.unknown)?;
    let Some(root) = &staged.root else {
        return Err("data.root must be a non-empty string".to_owned());
    };
    if !root.starts_with('/') {
        return Err(format!(
            "data.root must be an absolute path after interpolation, got {root:?}"
        ));
    }
    match &staged.scope.text {
        Some(text) if text == TENHOU_SCOPE => {}
        _ => {
            return Err(format!(
                "data.scope is Tenhou-only in v1 (must be {TENHOU_SCOPE:?}), got {}",
                staged.scope.repr
            ));
        }
    }
    // Oracle checks `== ""` only (no strip): whitespace-only splits pass.
    match &staged.train_split.text {
        Some(text) if !text.is_empty() => {}
        _ => return Err("data.train_split must be a non-empty string".to_owned()),
    }
    match &staged.val_split.text {
        Some(text) if !text.is_empty() => {}
        _ => return Err("data.val_split must be a non-empty string".to_owned()),
    }
    for (flag, stage) in [
        ("wall_disjoint", &staged.wall_disjoint),
        ("leakage_check", &staged.leakage_check),
        ("actor_privileged_split", &staged.actor_privileged_split),
    ] {
        if !stage.is_true {
            return Err(format!(
                "data.{flag} must stay true (ephemeral rows carry the full validate/quarantine/split/leakage contract); refusing to weaken it"
            ));
        }
    }
    if staged.dora_present && !staged.dora_is_five {
        return Err(format!(
            "data.dora_width must be 5 (frozen (5,) dora), got {}",
            staged.dora_repr
        ));
    }
    let Some(drop_last) = staged.drop_last.value else {
        return Err(format!(
            "data.drop_last must be a bool, got {}",
            staged.drop_last.repr
        ));
    };
    let Some(homogeneous_buckets) = staged.homogeneous_buckets.value else {
        return Err(format!(
            "data.homogeneous_buckets must be a bool, got {}",
            staged.homogeneous_buckets.repr
        ));
    };
    match &staged.replay_backend.text {
        Some(text) if REPLAY_BACKENDS.contains(&text.as_str()) => {}
        _ => {
            return Err(format!(
                "data.replay_backend must be 'rust', got {}",
                staged.replay_backend.repr
            ));
        }
    }
    // Constructor order from here (oracle `:141-161`).
    let dataset_manifest_hash = check_pin(
        "data",
        "dataset_manifest_hash",
        &staged.dataset_manifest_hash,
    )?;
    let shuffle_buffer_size = check_present_int(
        "data",
        "shuffle_buffer_size",
        &staged.shuffle_buffer_size,
        check_nonnegative_int,
    )?;
    let num_workers = check_present_int(
        "data",
        "num_workers",
        &staged.num_workers,
        check_nonnegative_int,
    )?;
    let world_size =
        check_present_int("data", "world_size", &staged.world_size, check_positive_int)?;
    let decode_prefetch =
        check_present_bounded("data", "decode_prefetch", &staged.decode_prefetch, 1, 1024)?;
    let expand_batch_games = check_present_bounded(
        "data",
        "expand_batch_games",
        &staged.expand_batch_games,
        1,
        1024,
    )?;
    Ok(DataValid {
        root: root.clone(),
        scope: staged
            .scope
            .text
            .clone()
            .unwrap_or_else(|| TENHOU_SCOPE.to_owned()),
        train_split: staged.train_split.text.clone().unwrap_or_default(),
        val_split: staged.val_split.text.clone().unwrap_or_default(),
        dora: 5,
        drop_last,
        homogeneous_buckets,
        replay_backend: staged
            .replay_backend
            .text
            .clone()
            .unwrap_or_else(|| "rust".to_owned()),
        dataset_manifest_hash,
        shuffle_buffer_size,
        num_workers,
        world_size,
        decode_prefetch,
        expand_batch_games,
    })
}

/// Gate a present-or-absent int slot: absent → `None` (caller substitutes the
/// default); present → validated `i64` or huge marker.
fn check_present_int(
    where_: &str,
    key: &str,
    slot: &IntSlot,
    check: fn(&str, &str, &IntView) -> Result<(), String>,
) -> Result<Option<HugeInt>, String> {
    if !slot.present {
        return Ok(None);
    }
    check(where_, key, &slot.view)?;
    Ok(Some(slot.view.as_i64))
}

fn check_present_bounded(
    where_: &str,
    key: &str,
    slot: &IntSlot,
    lo: i64,
    hi: i64,
) -> Result<Option<HugeInt>, String> {
    if !slot.present {
        return Ok(None);
    }
    check_bounded_int(where_, key, &slot.view, lo, hi)?;
    Ok(Some(slot.view.as_i64))
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

const MODEL_ALLOWED: [&str; 3] = ["architecture_id", "action_count", "parameters"];

struct ModelStaged {
    unknown: UnknownView,
    architecture_id: Option<String>,
    action_count: IntSlot,
    parameters_present: bool,
    parameters_is_dict: bool,
    parameters: Option<Py<PyDict>>,
}

/// `_parse_model` (oracle `_rc_parse.py:165-193`): architecture identity plus
/// frozen action width. The `KNOWN_ARCHITECTURES` registry and the
/// `BASELINE_ACTION_COUNT` equality stay Python (torch-owned patch-points).
#[pyfunction]
fn rc_parse_model(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let dict = section_dict(&raw, "model")?;
    let unknown = stage_unknown(&dict, &MODEL_ALLOWED, "model")?;
    // Oracle strips for the emptiness test only; the stored id is unstripped.
    let arch_slot = field_slot(&dict, "architecture_id", "rc_parse_model")?;
    let architecture_id: Option<String> = match arch_slot {
        None => Some("hydra2_baseline_transformer_v1".to_owned()),
        Some(obj) if obj.is_instance_of::<PyString>() => {
            let text: String = obj.extract().map_err(|e| {
                PyValueError::new_err(format!("contracts rc_parse_model field unreadable: {e}"))
            })?;
            let stripped: String = obj
                .call_method0("strip")
                .map_err(|e| {
                    PyValueError::new_err(format!("contracts rc_parse_model field unreadable: {e}"))
                })?
                .extract()
                .map_err(|e| {
                    PyValueError::new_err(format!("contracts rc_parse_model field unreadable: {e}"))
                })?;
            if stripped.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        Some(_) => None,
    };
    let action_count = stage_int_slot(py, field_slot(&dict, "action_count", "rc_parse_model")?)?;
    let parameters_slot = field_slot(&dict, "parameters", "rc_parse_model")?;
    let (parameters_present, parameters_is_dict, parameters): (bool, bool, Option<Py<PyDict>>) =
        match parameters_slot {
            None => (false, true, None),
            Some(obj) => match obj.extract::<Bound<'_, PyDict>>() {
                Ok(params) => (true, true, Some(params.unbind())),
                Err(_) => (true, false, None),
            },
        };
    let staged = ModelStaged {
        unknown,
        architecture_id,
        action_count,
        parameters_present,
        parameters_is_dict,
        parameters,
    };
    let valid = py
        .detach(|| validate_model(&staged))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("architecture_id", valid.0)?;
    match valid.1 {
        Some(value) => {
            out.set_item("action_count", value)?;
        }
        None => {
            if let Some(handle) = staged.action_count.handle.as_ref() {
                out.set_item("action_count", handle.bind(py))?;
            } else {
                out.set_item("action_count", 6792)?;
            }
        }
    }
    let params = PyDict::new(py);
    if let Some(stored) = staged.parameters.as_ref() {
        for (head, value) in stored.bind(py).iter() {
            params.set_item(head, value)?;
        }
    }
    out.set_item("parameters", params)?;
    Ok(out.unbind())
}

fn validate_model(staged: &ModelStaged) -> Result<(String, HugeInt), String> {
    check_unknown(&staged.unknown)?;
    let Some(architecture_id) = &staged.architecture_id else {
        return Err("model.architecture_id must be a non-empty string".to_owned());
    };
    let action: HugeInt = if !staged.action_count.present {
        Some(6792)
    } else {
        check_positive_int("model", "action_count", &staged.action_count.view)?;
        staged.action_count.view.as_i64
    };
    // Positive huge ints pass (oracle `value <= 0` is false for them) and keep
    // the original handle; the Python side still applies the baseline gate.
    if staged.parameters_present && !staged.parameters_is_dict {
        return Err("model.parameters must be a mapping".to_owned());
    }
    Ok((architecture_id.clone(), action))
}

// ---------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------

pub(crate) const WEIGHTS_ALLOWED: [&str; 7] = [
    "w_policy",
    "w_placement",
    "w_value",
    "w_event",
    "w_belief",
    "privileged_source_hash",
    "label_smoothing",
];

pub(crate) const LABEL_SMOOTHING_ENDS: FloatEnds = FloatEnds {
    lo: 0.0,
    lo_repr: "0.0",
    lo_open: false,
    hi: 1.0,
    hi_repr: "1.0",
    hi_open: true,
};

// ---------------------------------------------------------------------------
// loop
// ---------------------------------------------------------------------------

const LOOP_ALLOWED: [&str; 12] = [
    "microbatch_size",
    "accumulation_steps",
    "gradient_clip_norm",
    "max_updates",
    "checkpoint_frequency_updates",
    "precision",
    "keep_last_checkpoints",
    "stratified_sampling",
    "sampling_ratios",
    "log_per_type_metrics",
    "fit_temperature",
    "fetch_prefetch_depth",
];

struct LoopStaged {
    unknown: UnknownView,
    clip_present: bool,
    clip_is_none: bool,
    clip: NumView,
    precision: OptStr,
    microbatch_size: IntSlot,
    accumulation_steps: IntSlot,
    max_updates: IntSlot,
    checkpoint_frequency_updates: IntSlot,
    keep_last_present: bool,
    keep_last_is_none: bool,
    keep_last: IntSlot,
    stratified_sampling: BoolStage,
    sampling_ratios: MapStage,
    log_per_type_metrics: BoolStage,
    fit_temperature: BoolStage,
    fetch_prefetch_depth: IntSlot,
}

struct LoopValid {
    microbatch_size: HugeInt,
    accumulation_steps: HugeInt,
    gradient_clip_norm: Option<f64>,
    max_updates: HugeInt,
    checkpoint_frequency_updates: HugeInt,
    precision: String,
    keep_last_checkpoints: Option<HugeInt>,
    stratified_sampling: bool,
    sampling_ratios: Option<Vec<(String, f64)>>,
    log_per_type_metrics: bool,
    fit_temperature: bool,
    fetch_prefetch_depth: HugeInt,
}

/// `_parse_loop` (oracle `_rc_parse.py:363-425`): microbatching, clipping,
/// horizon, and checkpoint cadence.
#[pyfunction]
#[allow(clippy::too_many_lines)]
fn rc_parse_loop(py: Python<'_>, raw: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let dict = section_dict(&raw, "loop")?;
    let unknown = stage_unknown(&dict, &LOOP_ALLOWED, "loop")?;
    // Oracle: `fields.get("gradient_clip_norm", 1.0)` then `None` passes
    // through, everything else gates positive-finite.
    let clip_slot = field_slot(&dict, "gradient_clip_norm", "rc_parse_loop")?;
    let (clip_present, clip_is_none, clip): (bool, bool, NumView) = match clip_slot {
        None => (
            false,
            false,
            NumView {
                repr: "1.0".to_owned(),
                number: Some(1.0),
            },
        ),
        Some(obj) if obj.is_none() => (
            true,
            true,
            NumView {
                repr: "None".to_owned(),
                number: None,
            },
        ),
        Some(obj) => (true, false, stage_num(&obj)?),
    };
    let precision = stage_opt_str(field_slot(&dict, "precision", "rc_parse_loop")?, "fp32")?;
    let microbatch_size =
        stage_int_slot(py, field_slot(&dict, "microbatch_size", "rc_parse_loop")?)?;
    let accumulation_steps = stage_int_slot(
        py,
        field_slot(&dict, "accumulation_steps", "rc_parse_loop")?,
    )?;
    let max_updates = stage_int_slot(py, field_slot(&dict, "max_updates", "rc_parse_loop")?)?;
    let checkpoint_frequency_updates = stage_int_slot(
        py,
        field_slot(&dict, "checkpoint_frequency_updates", "rc_parse_loop")?,
    )?;
    // Oracle: `fields.get("keep_last_checkpoints") is None` (absent AND
    // explicit null) reads as `None`, else the positive gate.
    let keep_slot = field_slot(&dict, "keep_last_checkpoints", "rc_parse_loop")?;
    let (keep_last_present, keep_last_is_none): (bool, bool) = match &keep_slot {
        None => (false, true),
        Some(obj) if obj.is_none() => (true, true),
        Some(_) => (true, false),
    };
    let keep_last = stage_int_slot(py, keep_slot)?;
    let stratified_sampling = stage_loop_bool(
        field_slot(&dict, "stratified_sampling", "rc_parse_loop")?,
        false,
    )?;
    let sampling_ratios = stage_map(field_slot(&dict, "sampling_ratios", "rc_parse_loop")?)?;
    let log_per_type_metrics = stage_loop_bool(
        field_slot(&dict, "log_per_type_metrics", "rc_parse_loop")?,
        true,
    )?;
    let fit_temperature =
        stage_loop_bool(field_slot(&dict, "fit_temperature", "rc_parse_loop")?, true)?;
    let fetch_prefetch_depth = stage_int_slot(
        py,
        field_slot(&dict, "fetch_prefetch_depth", "rc_parse_loop")?,
    )?;
    let staged = LoopStaged {
        unknown,
        clip_present,
        clip_is_none,
        clip,
        precision,
        microbatch_size,
        accumulation_steps,
        max_updates,
        checkpoint_frequency_updates,
        keep_last_present,
        keep_last_is_none,
        keep_last,
        stratified_sampling,
        sampling_ratios,
        log_per_type_metrics,
        fit_temperature,
        fetch_prefetch_depth,
    };
    let valid = py
        .detach(|| validate_loop(&staged))
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    set_loop_int(
        &out,
        py,
        "microbatch_size",
        &staged.microbatch_size.handle,
        &valid.microbatch_size,
        4,
    )?;
    set_loop_int(
        &out,
        py,
        "accumulation_steps",
        &staged.accumulation_steps.handle,
        &valid.accumulation_steps,
        8,
    )?;
    out.set_item("gradient_clip_norm", valid.gradient_clip_norm)?;
    set_loop_int(
        &out,
        py,
        "max_updates",
        &staged.max_updates.handle,
        &valid.max_updates,
        10000,
    )?;
    set_loop_int(
        &out,
        py,
        "checkpoint_frequency_updates",
        &staged.checkpoint_frequency_updates.handle,
        &valid.checkpoint_frequency_updates,
        500,
    )?;
    out.set_item("precision", valid.precision)?;
    match &valid.keep_last_checkpoints {
        None => {
            let none = py.None();
            out.set_item("keep_last_checkpoints", &none)?;
        }
        Some(Some(value)) => {
            out.set_item("keep_last_checkpoints", *value)?;
        }
        Some(None) => {
            if let Some(handle) = staged.keep_last.handle.as_ref() {
                out.set_item("keep_last_checkpoints", handle.bind(py))?;
            } else {
                let none = py.None();
                out.set_item("keep_last_checkpoints", &none)?;
            }
        }
    }
    out.set_item("stratified_sampling", valid.stratified_sampling)?;
    match valid.sampling_ratios.as_ref() {
        None => {
            let none = py.None();
            out.set_item("sampling_ratios", &none)?;
        }
        Some(entries) => {
            let ratios = PyDict::new(py);
            for (head, number) in entries {
                ratios.set_item(head, *number)?;
            }
            out.set_item("sampling_ratios", ratios)?;
        }
    }
    out.set_item("log_per_type_metrics", valid.log_per_type_metrics)?;
    out.set_item("fit_temperature", valid.fit_temperature)?;
    set_loop_int(
        &out,
        py,
        "fetch_prefetch_depth",
        &staged.fetch_prefetch_depth.handle,
        &valid.fetch_prefetch_depth,
        3,
    )?;
    Ok(out.unbind())
}

/// Strict loop bool staging: absent/null read as `default`, present bools pass
/// through, present non-bools stage their repr for the detached error
/// (restates `rc_require.rs:761-790` without the `where` param; the literal
/// `loop.` prefix is kept by the caller).
fn stage_loop_bool(slot: Option<Bound<'_, PyAny>>, default: bool) -> PyResult<BoolStage> {
    match slot {
        None => Ok(BoolStage {
            value: Some(default),
            repr: String::new(),
        }),
        Some(obj) if obj.is_none() => Ok(BoolStage {
            value: Some(default),
            repr: String::new(),
        }),
        Some(obj) if obj.is_instance_of::<PyBool>() => {
            let value: bool = obj.extract().map_err(|e| {
                PyValueError::new_err(format!("contracts rc_parse field unreadable: {e}"))
            })?;
            Ok(BoolStage {
                value: Some(value),
                repr: String::new(),
            })
        }
        Some(obj) => Ok(BoolStage {
            value: None,
            repr: py_repr(&obj)?,
        }),
    }
}

fn set_loop_int(
    out: &Bound<'_, PyDict>,
    py: Python<'_>,
    key: &str,
    handle: &Option<Py<PyAny>>,
    valid: &HugeInt,
    default: i64,
) -> PyResult<()> {
    match valid {
        Some(value) => {
            out.set_item(key, *value)?;
        }
        None => {
            if let Some(handle) = handle {
                out.set_item(key, handle.bind(py))?;
            } else {
                out.set_item(key, default)?;
            }
        }
    }
    Ok(())
}

fn check_loop_bool(key: &str, staged: &BoolStage) -> Result<bool, String> {
    match staged.value {
        Some(value) => Ok(value),
        None => Err(format!("loop.{key} must be a bool, got {}", staged.repr)),
    }
}

#[allow(clippy::too_many_lines)]
fn validate_loop(staged: &LoopStaged) -> Result<LoopValid, String> {
    check_unknown(&staged.unknown)?;
    // Oracle `:382-392`: absent reads `1.0`, explicit null reads `None`,
    // everything else gates positive-finite.
    let gradient_clip_norm = if !staged.clip_present {
        Some(1.0)
    } else if staged.clip_is_none {
        None
    } else {
        let Some(number) = staged.clip.number else {
            return Err(format!(
                "loop.gradient_clip_norm must be null or positive finite, got {}",
                staged.clip.repr
            ));
        };
        if !number.is_finite() || number <= 0.0 {
            return Err(format!(
                "loop.gradient_clip_norm must be null or positive finite, got {}",
                staged.clip.repr
            ));
        }
        Some(number)
    };
    let keep_last_checkpoints = if !staged.keep_last_present || staged.keep_last_is_none {
        None
    } else {
        check_positive_int("loop", "keep_last_checkpoints", &staged.keep_last.view)?;
        Some(staged.keep_last.view.as_i64)
    };
    let stratified_sampling = check_loop_bool("stratified_sampling", &staged.stratified_sampling)?;
    let sampling_ratios = match &staged.sampling_ratios.entries {
        None => {
            if staged.sampling_ratios.type_name.is_some() {
                return Err(format!(
                    "loop.sampling_ratios must be a mapping or null, got {}",
                    staged.sampling_ratios.type_name.clone().unwrap_or_default()
                ));
            }
            None
        }
        Some(entries) => Some(check_positive_float_map(
            "loop",
            "sampling_ratios",
            entries,
        )?),
    };
    let log_per_type_metrics =
        check_loop_bool("log_per_type_metrics", &staged.log_per_type_metrics)?;
    let fit_temperature = check_loop_bool("fit_temperature", &staged.fit_temperature)?;
    if staged.fetch_prefetch_depth.present {
        check_bounded_int(
            "loop",
            "fetch_prefetch_depth",
            &staged.fetch_prefetch_depth.view,
            1,
            16,
        )?;
    }
    // Absent ints read their oracle defaults; present ones validated above
    // (`Some(None)` is a huge int keeping its staged handle).
    let or_default = |slot: &IntSlot, valid: Option<HugeInt>, default: i64| -> HugeInt {
        if slot.present {
            valid.unwrap_or(None)
        } else {
            Some(default)
        }
    };
    Ok(LoopValid {
        microbatch_size: or_default(
            &staged.microbatch_size,
            check_loop_present("microbatch_size", &staged.microbatch_size)?,
            4,
        ),
        accumulation_steps: or_default(
            &staged.accumulation_steps,
            check_loop_present("accumulation_steps", &staged.accumulation_steps)?,
            8,
        ),
        gradient_clip_norm,
        max_updates: or_default(
            &staged.max_updates,
            check_loop_present("max_updates", &staged.max_updates)?,
            10000,
        ),
        checkpoint_frequency_updates: or_default(
            &staged.checkpoint_frequency_updates,
            check_loop_present(
                "checkpoint_frequency_updates",
                &staged.checkpoint_frequency_updates,
            )?,
            500,
        ),
        precision: staged
            .precision
            .text
            .clone()
            .unwrap_or_else(|| "fp32".to_owned()),
        keep_last_checkpoints,
        stratified_sampling,
        sampling_ratios,
        log_per_type_metrics,
        fit_temperature,
        fetch_prefetch_depth: or_default(
            &staged.fetch_prefetch_depth,
            check_loop_present("fetch_prefetch_depth", &staged.fetch_prefetch_depth)?,
            3,
        ),
    })
}

/// Present-or-absent positive int for the loop section: absent → `None`
/// (caller substitutes the oracle default).
fn check_loop_present(key: &str, slot: &IntSlot) -> Result<Option<HugeInt>, String> {
    if !slot.present {
        return Ok(None);
    }
    check_positive_int("loop", key, &slot.view)?;
    Ok(Some(slot.view.as_i64))
}

/// Register the eight run-config core section parsers on the shared
/// `contracts` submodule (mirrors `validate.rs:111-127`); MAIN calls this
/// from `contracts::register` — no new submodule, no new entry point.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(rc_parse_run, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_parse_data, sub)?)?;
    sub.add_function(wrap_pyfunction!(rc_parse_model, sub)?)?;
    crate::rc_parse_train::register_train(sub)?;
    sub.add_function(wrap_pyfunction!(rc_parse_loop, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rc_parse_train::{
        OPTIMIZER_ALLOWED, RUNTIME_ALLOWED, SCHEDULER_ALLOWED, rc_parse_optimizer,
    };

    #[test]
    fn section_shapes_match_oracle() {
        // Hand-verified against HEAD `src/hydra2/training/_rc_parse.py`
        // allowlists (:61, :75-95, :166, :197-209, :234-236, :286-290,
        // :318-320, :364-380) and `rc_sections.rs:45-72` vocab.
        assert_eq!(RUN_ALLOWED, ["id", "kind", "description"]);
        assert_eq!(DATA_ALLOWED.len(), 17);
        assert_eq!(MODEL_ALLOWED.len(), 3);
        assert_eq!(WEIGHTS_ALLOWED.len(), 7);
        assert_eq!(OPTIMIZER_ALLOWED.len(), 5);
        assert_eq!(SCHEDULER_ALLOWED.len(), 5);
        assert_eq!(RUNTIME_ALLOWED.len(), 4);
        assert_eq!(LOOP_ALLOWED.len(), 12);
        assert_eq!(RUN_KINDS, ["supervised"]);
        assert_eq!(TENHOU_SCOPE, "tenhou-4p-hanchan");
        assert_eq!(OPTIMIZER_IDS, ["adamw", "adam", "sgd"]);
        assert_eq!(SCHEDULER_IDS, ["cosine", "constant", "linear"]);
        assert_eq!(ADAPTER_IDS, ["plain_pytorch"]);
        assert_eq!(RUNTIME_PRECISIONS, ["fp32", "bf16_mixed"]);
        assert_eq!(
            COMPILE_MODES,
            [
                "eager",
                "default",
                "max-autotune-no-cudagraphs",
                "max-autotune"
            ]
        );
    }

    #[test]
    fn scalar_shapes_match_oracle() {
        // Hand-derived from the oracle bodies: run-id charset, CUDA device
        // grammar, digest shape, and the one-of list rendering.
        assert!(run_id_ok("wp14-run-001"));
        assert!(run_id_ok("a"));
        assert!(!run_id_ok(""));
        assert!(!run_id_ok("-lead"));
        assert!(!run_id_ok("has space"));
        assert!(!run_id_ok("uniçode"));
        assert!(cuda_device_ok("cuda"));
        assert!(cuda_device_ok("cuda:0"));
        assert!(cuda_device_ok("cuda:12"));
        assert!(!cuda_device_ok("cpu"));
        assert!(!cuda_device_ok("cuda:"));
        assert!(!cuda_device_ok("cuda:x"));
        assert!(!cuda_device_ok("CUDA"));
        assert!(is_digest_shape(&format!("sha256:{}", "ab".repeat(32))));
        assert!(!is_digest_shape(&format!("SHA256:{}", "AB".repeat(32))));
        assert_eq!(one_of_list(&RUN_KINDS), "['supervised']");
        assert_eq!(one_of_list(&OPTIMIZER_IDS), "['adamw', 'adam', 'sgd']");
        assert_eq!(
            one_of_list(&COMPILE_MODES),
            "['eager', 'default', 'max-autotune-no-cudagraphs', 'max-autotune']"
        );
    }

    #[test]
    fn unknown_and_bound_texts_match_oracle() {
        // Hand-derived from HEAD `_rc_require.py:90-96` (unknown gate) and
        // `_rc_parse.py:225-227` (`[0.0, 1.0)` smoothing bound).
        let view = UnknownView {
            keys: vec![
                ("'zzz'".to_owned(), Some("zzz".to_owned())),
                ("'mmm'".to_owned(), Some("mmm".to_owned())),
            ],
            allowed: vec!["id".to_owned()],
            where_: "run".to_owned(),
        };
        assert_eq!(
            check_unknown(&view).unwrap_err(),
            "run has unknown keys ['mmm', 'zzz']; allowed=['id']"
        );
        let ends = LABEL_SMOOTHING_ENDS;
        let view = NumView {
            repr: "1.0".to_owned(),
            number: Some(1.0),
        };
        assert_eq!(
            check_bounded_float("weights", "label_smoothing", &view, &ends).unwrap_err(),
            "weights.label_smoothing must lie in [0.0, 1.0), got 1.0"
        );
        let view = NumView {
            repr: "True".to_owned(),
            number: None,
        };
        assert_eq!(
            check_nonnegative_float("weights", "w_policy", &view).unwrap_err(),
            "weights.w_policy must be a non-negative number, got True"
        );
        let view = IntView {
            repr: "0".to_owned(),
            is_int: true,
            as_i64: Some(0),
            negative: false,
        };
        assert_eq!(
            check_positive_int("loop", "max_updates", &view).unwrap_err(),
            "loop.max_updates must be a positive int, got 0"
        );
    }

    #[test]
    fn attached_parsers_match_oracle() {
        Python::initialize();
        Python::attach(|py| {
            // Valid run section (oracle `:60-71` defaults).
            let raw = PyDict::new(py);
            raw.set_item("id", "wp14-run-001").unwrap();
            let out = rc_parse_run(py, raw.into_any()).unwrap();
            let bound = out.bind(py);
            assert_eq!(
                bound
                    .get_item("run_id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "wp14-run-001"
            );
            assert_eq!(
                bound
                    .get_item("kind")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "supervised"
            );
            // Bad run id echoes Python-repr style with single quotes.
            let raw = PyDict::new(py);
            raw.set_item("id", "-bad").unwrap();
            let err = rc_parse_run(py, raw.into_any()).unwrap_err();
            assert!(
                err.to_string()
                    .contains("run.id must match [A-Za-z0-9][A-Za-z0-9_-]*, got '-bad'"),
                "unexpected run-id error text: {err}"
            );
            // Unknown keys fail closed with the sorted allowlist.
            let raw = PyDict::new(py);
            raw.set_item("id", "ok-id").unwrap();
            raw.set_item("zzz", 1).unwrap();
            let err = rc_parse_run(py, raw.into_any()).unwrap_err();
            assert!(
                err.to_string().contains(
                    "run has unknown keys ['zzz']; allowed=['description', 'id', 'kind']"
                ),
                "unexpected unknown-keys error text: {err}"
            );
            // Non-mapping sections name the type, unquoted.
            let raw = PyString::new(py, "nope");
            let err = rc_parse_data(py, raw.into_any()).unwrap_err();
            assert!(
                err.to_string().contains("data must be a mapping, got str"),
                "unexpected mapping error text: {err}"
            );
            // Optimizer betas render the converted tuple on range failure.
            let raw = PyDict::new(py);
            raw.set_item("betas", vec![0.9, 1.5]).unwrap();
            let err = rc_parse_optimizer(py, raw.into_any()).unwrap_err();
            assert!(
                err.to_string()
                    .contains("optimizer.betas entries must lie in [0, 1), got (0.9, 1.5)"),
                "unexpected betas error text: {err}"
            );
            // Loop clip accepts null, rejects non-positive.
            let raw = PyDict::new(py);
            raw.set_item("gradient_clip_norm", py.None()).unwrap();
            let out = rc_parse_loop(py, raw.into_any()).unwrap();
            assert!(
                out.bind(py)
                    .get_item("gradient_clip_norm")
                    .unwrap()
                    .unwrap()
                    .is_none()
            );
            let raw = PyDict::new(py);
            raw.set_item("gradient_clip_norm", 0.0).unwrap();
            let err = rc_parse_loop(py, raw.into_any()).unwrap_err();
            assert!(
                err.to_string()
                    .contains("loop.gradient_clip_norm must be null or positive finite, got 0.0"),
                "unexpected clip error text: {err}"
            );
        });
    }
}
