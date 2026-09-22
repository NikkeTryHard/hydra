//! rc_sections: frozen run-config section vocabulary + training cursor envelope.
//!
//! DAG: this module depends on pyo3 ONLY (no feed/shard/search edge; every
//! value below is a frozen literal owned by
//! `python/hydra2/training/_rc_sections.py`, single-sourced here after the
//! wave-4 port). FORBIDS: interpolation logic and file IO (both stay Python
//! in `_rc_require.py` / `_rc_root.py`), regex compilation (Python keeps the
//! compiled `_INTERPOLATION_RE` / `_RUN_ID_RE` / `_CUDA_DEVICE_RE`; only the
//! pattern strings are consts here), and dataclass state (all section
//! records stay Python; only the cursor-envelope gates move).
//! Arg-check runs attached, the envelope + int gates run under ONE
//! `py.detach(|| ...)` over owned data with zero Python API inside, and
//! results are wrapped attached. Fail-closed: every shape violation is
//! `PyValueError` (Python re-raises `ContractError` with the identical
//! text), never a default or a fallback.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule (wired by MAIN in `contracts.rs`), mirroring
//! `event_schema::register` / `validate::register`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFrozenSet, PyModule, PyTuple};

/// The fourteen RunSpec YAML sections, in canonical order
/// (`_rc_sections.py:59-74`; unknown top-level keys are rejected).
const CONFIG_SECTIONS: [&str; 14] = [
    "run",
    "data",
    "model",
    "weights",
    "optimizer",
    "scheduler",
    "runtime",
    "loop",
    "seeds",
    "selection",
    "mirror",
    "telemetry",
    "eval",
    "output",
];

/// Run kinds accepted by the streaming-first v1 surface (`_rc_sections.py:79`).
const RUN_KINDS: [&str; 1] = ["supervised"];

/// Env vars allowed inside `${VAR}` interpolation (`_rc_sections.py:84-91`).
const INTERPOLATION_ALLOWLIST: [&str; 4] = [
    "HYDRA2_ARTIFACT_ROOT",
    "HYDRA2_DATA_ROOT",
    "HOME",
    "XDG_CACHE_HOME",
];

/// Tenhou-only v1 scope (`_rc_sections.py:96`).
const TENHOU_SCOPE: &str = "tenhou-4p-hanchan";

/// Registered optimizer ids (`_rc_sections.py:98`).
const OPTIMIZER_IDS: [&str; 3] = ["adamw", "adam", "sgd"];
/// Registered scheduler ids (`_rc_sections.py:99`).
const SCHEDULER_IDS: [&str; 3] = ["cosine", "constant", "linear"];
/// Registered adapter ids (`_rc_sections.py:100`).
const ADAPTER_IDS: [&str; 1] = ["plain_pytorch"];
/// Runtime precisions (`_rc_sections.py:101`).
const RUNTIME_PRECISIONS: [&str; 2] = ["fp32", "bf16_mixed"];
/// Compile modes (`_rc_sections.py:102-107`).
const COMPILE_MODES: [&str; 4] = [
    "eager",
    "default",
    "max-autotune-no-cudagraphs",
    "max-autotune",
];
/// Selection designs (`_rc_sections.py:108`).
const SELECTION_DESIGNS: [&str; 2] = ["fixed_n", "time_uniform_cs"];

/// Pattern text of `_INTERPOLATION_RE` (`_rc_sections.py:93`; compiled in Python).
const INTERPOLATION_PATTERN: &str = "\\$\\{([A-Za-z_][A-Za-z0-9_]*)\\}";
/// Pattern text of `_RUN_ID_RE` (`_rc_sections.py:94`; compiled in Python).
const RUN_ID_PATTERN: &str = "[A-Za-z0-9][A-Za-z0-9_-]*\\Z";
/// Pattern text of `_CUDA_DEVICE_RE` (`_rc_sections.py:95`; compiled in Python).
const CUDA_DEVICE_PATTERN: &str = "^cuda(:[0-9]+)?$";

/// Training cursor envelope fields, in canonical order
/// (`StreamCursor.from_dict`, `_rc_sections.py:468`).
const CURSOR_FIELDS: [&str; 5] = ["file_index", "byte_offset", "games_seen", "seed", "epoch"];

/// One extracted cursor slot: plain ints carry the value; `Missing` feeds the
/// envelope gate while `Rejected` (bool, non-int) feeds the value gate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CursorSlot {
    Missing,
    Rejected,
    Int(i64),
}

/// Owned snapshot of a raw cursor mapping: one `(field, slot)` per
/// [`CURSOR_FIELDS`] entry in field order, plus the off-envelope keys.
struct CursorSnapshot {
    fields: Vec<(String, CursorSlot)>,
    unknown: Vec<String>,
}

/// Python-`repr` of a string list (`['a', 'b']`, `[]` when empty), matching
/// the `from_dict` oracle's f-string interpolation of `missing`/`unknown`.
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

/// Envelope + value gates over owned data only (zero Python API): mirrors
/// `StreamCursor.from_dict` in `python/hydra2/training/_rc_sections.py`
/// exactly — missing in envelope order, unknown sorted, first bad value in
/// envelope order, `{key!r}` single-quoted.
fn validate_training_cursor(snapshot: CursorSnapshot) -> Result<[u64; 5], String> {
    let mut unknown = snapshot.unknown;
    unknown.sort();
    let missing: Vec<String> = snapshot
        .fields
        .iter()
        .filter(|(_, slot)| *slot == CursorSlot::Missing)
        .map(|(key, _)| key.clone())
        .collect();
    if !missing.is_empty() || !unknown.is_empty() {
        return Err(format!(
            "stream cursor envelope mismatch; missing={} unknown={}",
            py_str_list(&missing),
            py_str_list(&unknown)
        ));
    }
    let mut out = [0_u64; CURSOR_FIELDS.len()];
    for (index, (key, slot)) in snapshot.fields.iter().enumerate() {
        match slot {
            CursorSlot::Int(value) if *value >= 0 => {
                out[index] = u64::try_from(*value)
                    .map_err(|_| format!("stream cursor '{key}' must be a non-negative int"))?;
            }
            _ => {
                return Err(format!("stream cursor '{key}' must be a non-negative int"));
            }
        }
    }
    Ok(out)
}

/// Training cursor unpack: five-field YAML-resume mapping in, five frontier
/// fields out (attached arg-check — `get_item` per envelope key per
/// `resume.rs:597`, bool excluded first per `resume.rs:605` — then ONE
/// `detach` over the owned snapshot per `contracts.rs:435`).
#[pyfunction]
fn rc_unpack_cursor(py: Python<'_>, raw: Bound<'_, PyDict>) -> PyResult<(u64, u64, u64, u64, u64)> {
    let mut fields = Vec::with_capacity(CURSOR_FIELDS.len());
    for key in CURSOR_FIELDS {
        let value = raw
            .get_item(key)
            .map_err(|e| PyValueError::new_err(format!("cursor field {key:?} unreadable: {e}")))?;
        let slot = match value {
            None => CursorSlot::Missing,
            Some(item) if item.is_instance_of::<PyBool>() => CursorSlot::Rejected,
            Some(item) => match item.extract::<i64>() {
                Ok(number) => CursorSlot::Int(number),
                Err(_) => CursorSlot::Rejected,
            },
        };
        fields.push((key.to_string(), slot));
    }
    let mut unknown = Vec::new();
    for (key, _) in raw.iter() {
        let name: String = key
            .extract()
            .map_err(|e| PyValueError::new_err(format!("stream cursor key unreadable: {e}")))?;
        if !CURSOR_FIELDS.contains(&name.as_str()) {
            unknown.push(name);
        }
    }
    let out = py
        .detach(|| validate_training_cursor(CursorSnapshot { fields, unknown }))
        .map_err(PyValueError::new_err)?;
    let [file_index, byte_offset, games_seen, seed, epoch] = out;
    Ok((file_index, byte_offset, games_seen, seed, epoch))
}

/// Register the run-config section vocabulary on the shared `contracts`
/// submodule: fourteen `RC_*` consts plus `rc_unpack_cursor` (`sub.add` per
/// `contracts.rs:1156`, `PyFrozenSet` per `contracts.rs:1186`,
/// `wrap_pyfunction!` per `contracts.rs:1119`, `let py = sub.py()` per
/// `contracts.rs:1115`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("RC_CONFIG_SECTIONS", PyTuple::new(py, CONFIG_SECTIONS)?)?;
    sub.add("RC_RUN_KINDS", PyTuple::new(py, RUN_KINDS)?)?;
    sub.add(
        "RC_INTERPOLATION_ALLOWLIST",
        PyFrozenSet::new(py, INTERPOLATION_ALLOWLIST)?,
    )?;
    sub.add("RC_TENHOU_SCOPE", TENHOU_SCOPE)?;
    sub.add("RC_OPTIMIZER_IDS", PyTuple::new(py, OPTIMIZER_IDS)?)?;
    sub.add("RC_SCHEDULER_IDS", PyTuple::new(py, SCHEDULER_IDS)?)?;
    sub.add("RC_ADAPTER_IDS", PyTuple::new(py, ADAPTER_IDS)?)?;
    sub.add(
        "RC_RUNTIME_PRECISIONS",
        PyTuple::new(py, RUNTIME_PRECISIONS)?,
    )?;
    sub.add("RC_COMPILE_MODES", PyTuple::new(py, COMPILE_MODES)?)?;
    sub.add("RC_SELECTION_DESIGNS", PyTuple::new(py, SELECTION_DESIGNS)?)?;
    sub.add("RC_INTERPOLATION_PATTERN", INTERPOLATION_PATTERN)?;
    sub.add("RC_RUN_ID_PATTERN", RUN_ID_PATTERN)?;
    sub.add("RC_CUDA_DEVICE_PATTERN", CUDA_DEVICE_PATTERN)?;
    sub.add("RC_CURSOR_FIELDS", PyTuple::new(py, CURSOR_FIELDS)?)?;
    sub.add_function(wrap_pyfunction!(rc_unpack_cursor, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_snapshot() -> CursorSnapshot {
        CursorSnapshot {
            fields: vec![
                ("file_index".to_string(), CursorSlot::Int(3)),
                ("byte_offset".to_string(), CursorSlot::Int(128)),
                ("games_seen".to_string(), CursorSlot::Int(10)),
                ("seed".to_string(), CursorSlot::Int(7)),
                ("epoch".to_string(), CursorSlot::Int(2)),
            ],
            unknown: vec![],
        }
    }

    fn missing_snapshot() -> CursorSnapshot {
        CursorSnapshot {
            fields: vec![
                ("file_index".to_string(), CursorSlot::Int(0)),
                ("byte_offset".to_string(), CursorSlot::Int(0)),
                ("games_seen".to_string(), CursorSlot::Int(0)),
                ("seed".to_string(), CursorSlot::Missing),
                ("epoch".to_string(), CursorSlot::Missing),
            ],
            unknown: vec!["zzz".to_string(), "mmm".to_string()],
        }
    }

    #[test]
    fn section_vocab_exact() {
        // Hand-verified against HEAD `python/hydra2/training/_rc_sections.py:59-108`.
        assert_eq!(
            CONFIG_SECTIONS,
            [
                "run",
                "data",
                "model",
                "weights",
                "optimizer",
                "scheduler",
                "runtime",
                "loop",
                "seeds",
                "selection",
                "mirror",
                "telemetry",
                "eval",
                "output",
            ]
        );
        assert_eq!(CONFIG_SECTIONS.len(), 14);
        assert_eq!(RUN_KINDS, ["supervised"]);
        assert_eq!(
            INTERPOLATION_ALLOWLIST,
            [
                "HYDRA2_ARTIFACT_ROOT",
                "HYDRA2_DATA_ROOT",
                "HOME",
                "XDG_CACHE_HOME",
            ]
        );
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
                "max-autotune",
            ]
        );
        assert_eq!(SELECTION_DESIGNS, ["fixed_n", "time_uniform_cs"]);
        assert_eq!(INTERPOLATION_PATTERN, "\\$\\{([A-Za-z_][A-Za-z0-9_]*)\\}");
        assert_eq!(RUN_ID_PATTERN, "[A-Za-z0-9][A-Za-z0-9_-]*\\Z");
        assert_eq!(CUDA_DEVICE_PATTERN, "^cuda(:[0-9]+)?$");
        assert_eq!(
            CURSOR_FIELDS,
            ["file_index", "byte_offset", "games_seen", "seed", "epoch"]
        );
    }

    #[test]
    fn cursor_round_trip_values() {
        assert_eq!(
            validate_training_cursor(ok_snapshot()),
            Ok([3, 128, 10, 7, 2])
        );
    }

    #[test]
    fn cursor_envelope_mismatch_exact() {
        // Oracle: missing in envelope order, unknown sorted; f-string of lists.
        let err = validate_training_cursor(missing_snapshot()).expect_err("must reject");
        assert_eq!(
            err,
            "stream cursor envelope mismatch; missing=['seed', 'epoch'] unknown=['mmm', 'zzz']"
        );
    }

    #[test]
    fn cursor_negative_rejected() {
        // Oracle iterates envelope order: file_index ok, byte_offset fails first.
        let mut snap = ok_snapshot();
        snap.fields[1].1 = CursorSlot::Int(-1);
        let err = validate_training_cursor(snap).expect_err("must reject");
        assert_eq!(
            err,
            "stream cursor 'byte_offset' must be a non-negative int"
        );
    }

    #[test]
    fn cursor_bool_rejected() {
        // Oracle excludes bool before the int check, same gate message.
        let mut snap = ok_snapshot();
        snap.fields[3].1 = CursorSlot::Rejected;
        let err = validate_training_cursor(snap).expect_err("must reject");
        assert_eq!(err, "stream cursor 'seed' must be a non-negative int");
    }
}
