//! common_roots: SPEC 2.1 remaining maker roots on the shared `contracts` submodule.
//!
//! DAG: pyo3 ONLY (pure argument validation, so no feed/shard/search owner
//! exists — same edge as `rc_require`, see `crates/bridge/Cargo.toml:25`;
//! no new dependency). FORBIDS: message texts diverging from the oracle
//! (the bridge raises `ValueError` with byte-identical oracle sentences;
//! thin Python translators in `python/hydra2/contracts/common.py` map to
//! `ContractError`), wall-clock/RNG, and live-object orchestration.
//!
//! TABLE ported here — the 8 pure makers, oracle
//! `src/hydra2/contracts/common.py:173-219` (HEAD; worktree
//! `python/hydra2/contracts/common.py:188-234`, makers region
//! byte-identical at port time):
//! - `make_sequence_no` <- `make_sequence_no` (:173-174)
//! - `make_tile_type` <- `make_tile_type` (:177-178; distinct from
//!   `tile_type_of`, `contracts.rs:163-172`, which derives `tile // 4` from
//!   a validated physical id)
//! - `make_belief_epoch_id` <- `make_belief_epoch_id` (:181-182)
//! - `make_parent_id` <- `make_parent_id` (:185-189)
//! - `make_packet_id` <- `make_packet_id` (:192-196)
//! - `make_run_id` <- `make_run_id` (:199-203)
//! - `make_utc_timestamp` <- `make_utc_timestamp` (:206-212)
//! - `make_schema_version` <- `make_schema_version` (:215-219)
//!
//! ALREADY-BRIDGED (reused, never re-ported): `is_seat` / `is_tile_id` /
//! `is_digest_text` / `make_seat` / `make_tile_id` / `make_action_id` /
//! `make_digest_text` (`contracts.rs:87-160`); those names are absent from
//! `common.py` (deletion wave, see its module docstring).
//!
//! LOGIC staying Python: `_require_int` / `_require_str` (private generic
//! helpers — the oracle fallback behind each translator while the extension
//! is stale; the bridge stages the same checks inline), the `Seat` /
//! `SequenceNo` / ... `NewType` aliases, `_UTC_TS_RE` / `_SCHEMA_VERSION_RE`
//! (compiled-pattern literals), the SPEC 3 hierarchy (`common_errors.rs`-
//! owned), `PBRF_ERROR_CODES`, and all 8 `ContractError` translators
//! (identical messages, no `__all__` change — `common.py` carries none).
//!
//! NONE-owner note: `rg` for `UTC_TS_RE|utc_timestamp.*RFC 3339|
//! MAJOR\.MINOR\.PATCH|SCHEMA_VERSION_RE` over `crates/feed/src` +
//! `crates/search/src` hits only the packager `utc_timestamp` *generator*
//! (`packager/src/integrity.rs:644`, wall-clock render — the opposite
//! direction, never a validator) and shard `schema_version` payload-field
//! plumbing (never a `MAJOR.MINOR.PATCH` shape test); no feed/search shape
//! owner exists, so both shape tests are restated here with the oracle line
//! cited per use (same restatement precedent as `rc_require.rs:186-197`).
//!
//! Shape per fn: attached staging (repr/type-name via the live Python API,
//! so `{value}` and `{text!r}` and `type(x).__name__` render byte-exact —
//! per `rc_require.rs:60-72`) → ONE `py.detach(|| …)` over owned plain data
//! with zero Python API inside (per `validate.rs:83-95`) → attached wrap as
//! `PyValueError` (per `contracts.rs:117-123`). Bool is excluded before
//! every int check (per `contracts.rs:69-74`); validated int originals are
//! returned unchanged so big ints pass through exactly like the oracle's
//! `return value` (per `rc_require.rs:516-518`).
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `validate.rs:111-127`); no new entry
//! point. MAIN wiring: `pub mod common_roots;` in `lib.rs` (after
//! `common_errors`) plus `crate::common_roots::register(&sub)?;` in
//! `contracts.rs` next to `crate::common_errors::register(&sub)?;`
//! (`contracts.rs:1316`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyInt, PyModule, PyString};

/// Attached staging helper: exact `repr(value)` text for every `{value}` /
/// `{text!r}` slot (per `rc_require.rs:60-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Attached staging helper: exact `type(value).__name__` text (per
/// `rc_require.rs:67-72`).
fn py_type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.get_type().name()?.to_str()?.to_owned())
}

/// Owned plain-data view of an int-or-other Python value (per
/// `rc_require.rs:79-84`), plus the staged type name: this oracle renders
/// type failures with the type name (`{name} must be an int, got bool`) and
/// range failures with the value (`{name}={value} out of range (...)`).
struct IntView {
    repr: String,
    type_name: String,
    is_int: bool,
    as_i64: Option<i64>,
    negative: bool,
}

fn stage_int(obj: &Bound<'_, PyAny>) -> PyResult<IntView> {
    let repr = py_repr(obj)?;
    let type_name = py_type_name(obj)?;
    if obj.is_instance_of::<PyBool>() || !obj.is_instance_of::<PyInt>() {
        return Ok(IntView {
            repr,
            type_name,
            is_int: false,
            as_i64: None,
            negative: false,
        });
    }
    let as_i64 = obj.extract::<i64>().ok();
    let negative = as_i64.map_or_else(|| repr.starts_with('-'), |v| v < 0);
    Ok(IntView {
        repr,
        type_name,
        is_int: true,
        as_i64,
        negative,
    })
}

/// Owned plain-data view of a str-or-other Python value.
struct StrView {
    text: Option<String>,
    repr: String,
    type_name: String,
}

fn stage_str(obj: &Bound<'_, PyAny>) -> PyResult<StrView> {
    let repr = py_repr(obj)?;
    let type_name = py_type_name(obj)?;
    if obj.is_instance_of::<PyString>() {
        let text: String = obj.extract()?;
        return Ok(StrView {
            text: Some(text),
            repr,
            type_name,
        });
    }
    Ok(StrView {
        text: None,
        repr,
        type_name,
    })
}

// ---------------------------------------------------------------------------
// Detached checks (owned plain data only, zero Python API).
// ---------------------------------------------------------------------------

/// `_require_int` core (oracle `common.py:157-164`): bool-excluded ints
/// only, then the `[minimum, maximum]` gate (`maximum: None` = unbounded
/// above, oracle bound `>={minimum}`; else `in [{minimum}, {hi}]`).
fn check_int(name: &str, view: &IntView, minimum: i64, maximum: Option<i64>) -> Result<(), String> {
    if !view.is_int {
        return Err(format!("{name} must be an int, got {}", view.type_name));
    }
    let in_range = match view.as_i64 {
        Some(value) => value >= minimum && maximum.is_none_or(|hi| value <= hi),
        // Huge int past `i64`: sign comes from the staged repr (per
        // `rc_require.rs:97`); unbounded-above passes iff nonnegative, any
        // upper bound rejects what cannot fit.
        None => maximum.is_none() && !view.negative,
    };
    if in_range {
        Ok(())
    } else {
        let bound = match maximum {
            None => format!(">={minimum}"),
            Some(hi) => format!("in [{minimum}, {hi}]"),
        };
        Err(format!("{name}={} out of range ({bound})", view.repr))
    }
}

/// Non-empty-str core for `make_parent_id` / `make_packet_id` /
/// `make_run_id` (oracle `common.py:185-203`): the empty reject is the
/// literal `{name} must be non-empty`.
fn check_nonempty_str(name: &str, view: &StrView) -> Result<String, String> {
    match &view.text {
        Some(text) if !text.is_empty() => Ok(text.clone()),
        Some(_) => Err(format!("{name} must be non-empty")),
        None => Err(format!("{name} must be a str, got {}", view.type_name)),
    }
}

/// UTC-shape test (oracle `common.py:153` `_UTC_TS_RE` via `fullmatch`,
/// `common.py:206-212`). ASCII-digit only: Python `\d` also matches Unicode
/// decimal digits, but bridge slots carry ASCII timestamps; realistic
/// callers are unaffected.
fn is_utc_timestamp_shape(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() < 20 || bytes[bytes.len() - 1] != b'Z' {
        return false;
    }
    let core = &bytes[..bytes.len() - 1];
    let head = &core[..19];
    for (index, byte) in head.iter().enumerate() {
        let separator = match index {
            4 | 7 => Some(b'-'),
            10 => Some(b'T'),
            13 | 16 => Some(b':'),
            _ => None,
        };
        match separator {
            Some(mark) => {
                if *byte != mark {
                    return false;
                }
            }
            None => {
                if !byte.is_ascii_digit() {
                    return false;
                }
            }
        }
    }
    if core.len() == 19 {
        return true;
    }
    let frac = &core[19..];
    if frac.len() < 2 || frac[0] != b'.' {
        return false;
    }
    frac[1..].iter().all(|byte| byte.is_ascii_digit())
}

/// Schema-version shape test (oracle `common.py:154` `_SCHEMA_VERSION_RE`
/// via `fullmatch`, `common.py:215-219`). ASCII-digit only, same note as
/// [`is_utc_timestamp_shape`].
fn is_schema_version_shape(text: &str) -> bool {
    let mut parts = 0;
    for part in text.split('.') {
        if part.is_empty() || !part.bytes().all(|byte| byte.is_ascii_digit()) {
            return false;
        }
        parts += 1;
    }
    parts == 3
}

// ---------------------------------------------------------------------------
// Pyfns (attached staging → ONE detach → attached wrap).
// ---------------------------------------------------------------------------

/// `make_sequence_no` root (oracle `common.py:173-174`).
#[pyfunction]
fn make_sequence_no(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let view = stage_int(&value)?;
    let handle = value.unbind();
    py.detach(|| check_int("sequence_no", &view, 0, None))
        .map_err(PyValueError::new_err)?;
    Ok(handle)
}

/// `make_tile_type` root (oracle `common.py:177-178`).
#[pyfunction]
fn make_tile_type(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let view = stage_int(&value)?;
    let handle = value.unbind();
    py.detach(|| check_int("tile_type", &view, 0, Some(33)))
        .map_err(PyValueError::new_err)?;
    Ok(handle)
}

/// `make_belief_epoch_id` root (oracle `common.py:181-182`).
#[pyfunction]
fn make_belief_epoch_id(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let view = stage_int(&value)?;
    let handle = value.unbind();
    py.detach(|| check_int("belief_epoch_id", &view, 0, None))
        .map_err(PyValueError::new_err)?;
    Ok(handle)
}

/// `make_parent_id` root (oracle `common.py:185-189`).
#[pyfunction]
fn make_parent_id(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<String> {
    let view = stage_str(&value)?;
    py.detach(|| check_nonempty_str("parent_id", &view))
        .map_err(PyValueError::new_err)
}

/// `make_packet_id` root (oracle `common.py:192-196`).
#[pyfunction]
fn make_packet_id(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<String> {
    let view = stage_str(&value)?;
    py.detach(|| check_nonempty_str("packet_id", &view))
        .map_err(PyValueError::new_err)
}

/// `make_run_id` root (oracle `common.py:199-203`).
#[pyfunction]
fn make_run_id(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<String> {
    let view = stage_str(&value)?;
    py.detach(|| check_nonempty_str("run_id", &view))
        .map_err(PyValueError::new_err)
}

/// `make_utc_timestamp` root (oracle `common.py:206-212`).
#[pyfunction]
fn make_utc_timestamp(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<String> {
    let view = stage_str(&value)?;
    py.detach(|| match &view.text {
        Some(text) if is_utc_timestamp_shape(text) => Ok(text.clone()),
        Some(_) => Err(format!(
            "utc_timestamp {} must be RFC 3339 UTC (YYYY-MM-DDTHH:MM:SS[.ffffff]Z)",
            view.repr
        )),
        None => Err(format!(
            "utc_timestamp must be a str, got {}",
            view.type_name
        )),
    })
    .map_err(PyValueError::new_err)
}

/// `make_schema_version` root (oracle `common.py:215-219`).
#[pyfunction]
fn make_schema_version(py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<String> {
    let view = stage_str(&value)?;
    py.detach(|| match &view.text {
        Some(text) if is_schema_version_shape(text) => Ok(text.clone()),
        Some(_) => Err(format!(
            "schema_version {} must be MAJOR.MINOR.PATCH",
            view.repr
        )),
        None => Err(format!(
            "schema_version must be a str, got {}",
            view.type_name
        )),
    })
    .map_err(PyValueError::new_err)
}

/// Register the eight SPEC 2.1 maker roots on the shared `contracts` submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(make_sequence_no, sub)?)?;
    sub.add_function(wrap_pyfunction!(make_tile_type, sub)?)?;
    sub.add_function(wrap_pyfunction!(make_belief_epoch_id, sub)?)?;
    sub.add_function(wrap_pyfunction!(make_parent_id, sub)?)?;
    sub.add_function(wrap_pyfunction!(make_packet_id, sub)?)?;
    sub.add_function(wrap_pyfunction!(make_run_id, sub)?)?;
    sub.add_function(wrap_pyfunction!(make_utc_timestamp, sub)?)?;
    sub.add_function(wrap_pyfunction!(make_schema_version, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_view(
        repr: &str,
        type_name: &str,
        is_int: bool,
        as_i64: Option<i64>,
        negative: bool,
    ) -> IntView {
        IntView {
            repr: repr.to_owned(),
            type_name: type_name.to_owned(),
            is_int,
            as_i64,
            negative,
        }
    }

    fn str_view(text: Option<&str>, repr: &str, type_name: &str) -> StrView {
        StrView {
            text: text.map(str::to_owned),
            repr: repr.to_owned(),
            type_name: type_name.to_owned(),
        }
    }

    #[test]
    fn sequence_no_accepts_zero() {
        let view = int_view("0", "int", true, Some(0), false);
        assert!(check_int("sequence_no", &view, 0, None).is_ok());
    }

    #[test]
    fn sequence_no_rejects_negative_with_oracle_text() {
        let view = int_view("-1", "int", true, Some(-1), true);
        assert_eq!(
            check_int("sequence_no", &view, 0, None),
            Err("sequence_no=-1 out of range (>=0)".to_owned())
        );
    }

    #[test]
    fn sequence_no_rejects_bool_with_oracle_text() {
        let view = int_view("True", "bool", false, None, false);
        assert_eq!(
            check_int("sequence_no", &view, 0, None),
            Err("sequence_no must be an int, got bool".to_owned())
        );
    }

    #[test]
    fn sequence_no_rejects_str_with_oracle_text() {
        let view = int_view("'x'", "str", false, None, false);
        assert_eq!(
            check_int("sequence_no", &view, 0, None),
            Err("sequence_no must be an int, got str".to_owned())
        );
    }

    #[test]
    fn sequence_no_passes_huge_nonnegative_like_oracle() {
        let big = "123456789012345678901234567890";
        let view = int_view(big, "int", true, None, false);
        assert!(check_int("sequence_no", &view, 0, None).is_ok());
    }

    #[test]
    fn tile_type_bounds_match_oracle() {
        let ok = int_view("33", "int", true, Some(33), false);
        assert!(check_int("tile_type", &ok, 0, Some(33)).is_ok());
        let bad = int_view("34", "int", true, Some(34), false);
        assert_eq!(
            check_int("tile_type", &bad, 0, Some(33)),
            Err("tile_type=34 out of range (in [0, 33])".to_owned())
        );
    }

    #[test]
    fn belief_epoch_matches_oracle() {
        let ok = int_view("7", "int", true, Some(7), false);
        assert!(check_int("belief_epoch_id", &ok, 0, None).is_ok());
        let bad = int_view("-1", "int", true, Some(-1), true);
        assert_eq!(
            check_int("belief_epoch_id", &bad, 0, None),
            Err("belief_epoch_id=-1 out of range (>=0)".to_owned())
        );
    }

    #[test]
    fn nonempty_ids_match_oracle() {
        assert_eq!(
            check_nonempty_str("parent_id", &str_view(Some("abc"), "'abc'", "str")),
            Ok("abc".to_owned())
        );
        assert_eq!(
            check_nonempty_str("parent_id", &str_view(Some(""), "''", "str")),
            Err("parent_id must be non-empty".to_owned())
        );
        assert_eq!(
            check_nonempty_str("packet_id", &str_view(Some(""), "''", "str")),
            Err("packet_id must be non-empty".to_owned())
        );
        assert_eq!(
            check_nonempty_str("run_id", &str_view(Some(""), "''", "str")),
            Err("run_id must be non-empty".to_owned())
        );
        assert_eq!(
            check_nonempty_str("run_id", &str_view(None, "123", "int")),
            Err("run_id must be a str, got int".to_owned())
        );
    }

    #[test]
    fn utc_shape_matches_oracle_regex() {
        assert!(is_utc_timestamp_shape("2026-08-22T12:00:00Z"));
        assert!(is_utc_timestamp_shape("2026-08-22T12:00:00.123456Z"));
        assert!(!is_utc_timestamp_shape("2026-08-22 12:00:00Z"));
        assert!(!is_utc_timestamp_shape("2026-08-22T12:00:00"));
        assert!(!is_utc_timestamp_shape("not-a-time"));
        assert!(!is_utc_timestamp_shape("2026-13-01T00:00:00Zx"));
        // Shape-only like the oracle regex: month 13 passes the shape gate.
        assert!(is_utc_timestamp_shape("2026-13-01T00:00:00Z"));
        assert!(!is_utc_timestamp_shape("2026-08-22T12:00:00.Z"));
    }

    #[test]
    fn schema_shape_matches_oracle_regex() {
        assert!(is_schema_version_shape("1.0.0"));
        for bad in ["1", "1.0", "v1.0.0", "1.0.0-dev", ""] {
            assert!(!is_schema_version_shape(bad), "bad={bad}");
        }
    }
}
