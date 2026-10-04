//? rl/config_ffi.rs: batch PyO3 surface over `rl::config`.
//!
//! DAG: pyo3 + `super::config` + serde_json ONLY (no feed/shard/search edge;
//! pure registry validation, so no crate owner exists — same NONE position
//! as `rc_require.rs:32-38`). FORBIDS: YAML/file IO, `${VAR}`
//! interpolation, run-config section parsing (all stay in `rc_parse.rs` /
//! Python `_rc_*.py`), wall-clock/RNG, `ContractError` shaping (this layer
//! raises `ValueError`/`TypeError`; thin Python translators map to
//! `ContractError` with byte-identical messages), and any new JCS printer
//! (all bytes/hash math lives in `hydra_feed::{canon, digest}` via
//! `super::config`).
//!
//! Shape per fn: attached staging (owned `String`/`Vec` extraction, live
//! `repr()` for every `{value!r}` slot so quoting renders byte-exact) → ONE
//! `py.detach(|| …)` over owned plain data with zero Python API inside (per
//! `contracts.rs:434-437`, `validate.rs:83-95`) → attached wrap as
//! `PyValueError` (per `contracts.rs:117-123`). Results are plain validated
//! values the Python translator forwards; bool is excluded before every int
//! check (per `contracts.rs:69-74`).
//!
//! Single-cdylib tree: registers onto the already-created `hydra2._native.rl`
//! submodule via `register` (mirrors `canon_rng::register` /
//! `walls::register` taking the submodule). NativeWall owns `rl/mod.rs`,
//! `lib.rs`, and the `rl::register` registrar; this file is never wired by
//! ConfigContracts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyModule};

use super::config;

/// Attached staging helper: exact `repr(value)` text for every `{value!r}`
/// slot (mirrors `rc_require.rs:63-65`).
fn py_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.repr()?.to_str()?.to_owned())
}

/// Resolve an artifact-type string to its registered record (attached: the
/// error renders the live repr, like `rc_parse::stage_opt_str`).
fn stage_artifact(obj: &Bound<'_, PyAny>) -> PyResult<(&'static config::RlSchemaRecord, String)> {
    let repr = py_repr(obj)?;
    let text: Option<String> = obj.extract().ok();
    match text {
        Some(text) => match config::RL_SCHEMA_RECORDS
            .iter()
            .find(|record| record.artifact_type == text)
        {
            Some(record) => Ok((record, repr)),
            None => Err(PyValueError::new_err(format!(
                "rl schema artifact must be one of ['hydra2.rl_control_schema', 'hydra2.rl_native_snapshot_schema', 'hydra2.rl_pipeline_schema', 'hydra2.rl_replay_index_schema'], got {repr}"
            ))),
        },
        None => Err(PyValueError::new_err(format!(
            "rl schema artifact must be one of ['hydra2.rl_control_schema', 'hydra2.rl_native_snapshot_schema', 'hydra2.rl_pipeline_schema', 'hydra2.rl_replay_index_schema'], got {repr}"
        ))),
    }
}

/// Stage one envelope payload level: top-level keys plus one
/// `(parent, children)` entry per nested mapping value (arrays of objects
/// union their item keys under the array parent, e.g. `tickets`).
/// Non-`str` keys reject exactly like the oracle `sorted()` path: the staged
/// repr carries the failure (mirrors `rc_parse::stage_unknown`).
struct EnvelopeStage {
    keys: Vec<String>,
    nested: Vec<(String, Vec<String>)>,
}

fn stage_envelope(payload: &Bound<'_, PyDict>) -> PyResult<EnvelopeStage> {
    let mut keys = Vec::with_capacity(payload.len());
    let mut nested = Vec::new();
    for (head, value) in payload.iter() {
        let head_text: String = head.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "rl envelope keys must be strings, got {}",
                py_repr(&head).unwrap_or_else(|_| "?".to_owned()),
            ))
        })?;
        keys.push(head_text.clone());
        if let Ok(sub) = value.extract::<Bound<'_, PyDict>>() {
            let mut children = Vec::with_capacity(sub.len());
            for (sub_head, _) in sub.iter() {
                let sub_text: String = sub_head.extract().map_err(|_| {
                    PyValueError::new_err(format!(
                        "rl envelope nested keys must be strings, got {}",
                        py_repr(&sub_head).unwrap_or_else(|_| "?".to_owned()),
                    ))
                })?;
                children.push(sub_text);
            }
            nested.push((head_text, children));
        } else if let Ok(items) = value.extract::<Bound<'_, PyList>>() {
            let mut children = Vec::new();
            for item in items.iter() {
                if let Ok(sub) = item.extract::<Bound<'_, PyDict>>() {
                    for (sub_head, _) in sub.iter() {
                        let sub_text: String = sub_head.extract().map_err(|_| {
                            PyValueError::new_err(format!(
                                "rl envelope nested keys must be strings, got {}",
                                py_repr(&sub_head).unwrap_or_else(|_| "?".to_owned()),
                            ))
                        })?;
                        if !children.contains(&sub_text) {
                            children.push(sub_text);
                        }
                    }
                }
            }
            if !children.is_empty() {
                nested.push((head_text, children));
            }
        }
    }
    Ok(EnvelopeStage { keys, nested })
}

/// Stage a payload mapping to a `serde_json::Value` for the identity check
/// (attached only: touches Python memory, never detached). Mirrors the
/// closed-domain arms of `rc_digest::py_to_value` (`None` → null; `bool`
/// BEFORE `int`; finite `float` only; `str`; `list` element-wise where
/// `tuple` is NOT a list and rejects; `dict` with `str` keys only).
fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        let flag: bool = obj.extract().map_err(|e| {
            PyValueError::new_err(format!("rl envelope value unreadable: {e}"))
        })?;
        return Ok(serde_json::Value::Bool(flag));
    }
    if let Ok(dict) = obj.extract::<Bound<'_, PyDict>>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (head, value) in dict.iter() {
            let head_text: String = head.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "rl envelope keys must be strings, got {}",
                    py_repr(&head).unwrap_or_else(|_| "?".to_owned()),
                ))
            })?;
            map.insert(head_text, py_to_value(&value)?);
        }
        return Ok(serde_json::Value::Object(map));
    }
    if let Ok(items) = obj.extract::<Bound<'_, PyList>>() {
        let mut out = Vec::with_capacity(items.len());
        for item in items.iter() {
            out.push(py_to_value(&item)?);
        }
        return Ok(serde_json::Value::Array(out));
    }
    if let Ok(text) = obj.extract::<String>() {
        return Ok(serde_json::Value::String(text));
    }
    if let Ok(number) = obj.extract::<i64>() {
        return Ok(serde_json::Value::from(number));
    }
    if let Ok(number) = obj.extract::<f64>() {
        if !number.is_finite() {
            return Err(PyValueError::new_err(format!(
                "rl envelope floats must be finite, got {}",
                py_repr(obj).unwrap_or_else(|_| "?".to_owned()),
            )));
        }
        return Ok(serde_json::Value::from(number));
    }
    Err(PyValueError::new_err(format!(
        "rl envelope values must be null/bool/int/float/str/list/dict, got {}",
        py_repr(obj).unwrap_or_else(|_| "?".to_owned()),
    )))
}

/// `rl_schema_digest(artifact) -> str`: the registered payload digest for one
/// envelope (`sha256:<hex>`, JCS payload without digest). Cold identity only.
#[pyfunction]
fn rl_schema_digest(py: Python<'_>, artifact: Bound<'_, PyAny>) -> PyResult<String> {
    let (record, _) = stage_artifact(&artifact)?;
    let digest = record.digest.to_owned();
    Ok(py.detach(|| digest))
}

/// `rl_schema_record(artifact) -> dict`: `{artifact_type, schema_version,
/// path, digest}` for one envelope. Cold identity only.
#[pyfunction]
fn rl_schema_record(py: Python<'_>, artifact: Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let (record, _) = stage_artifact(&artifact)?;
    let (artifact_type, schema_version, path, digest) = (
        record.artifact_type.to_owned(),
        record.schema_version.to_owned(),
        record.path.to_owned(),
        record.digest.to_owned(),
    );
    let resolved = py.detach(|| (artifact_type, schema_version, path, digest));
    let out = PyDict::new(py);
    out.set_item("artifact_type", resolved.0)?;
    out.set_item("schema_version", resolved.1)?;
    out.set_item("path", resolved.2)?;
    out.set_item("digest", resolved.3)?;
    Ok(out.unbind())
}

/// `rl_validate_envelope(artifact, payload) -> str`: closed-field gate plus
/// own-digest-exclusion identity over one envelope payload mapping. Returns
/// the verified digest. Every row validated before admission — no spot
/// checks; `hash(prev || row)` chaining is rejected by construction (the
/// identity always covers the full canonical payload bytes).
#[pyfunction]
fn rl_validate_envelope(
    py: Python<'_>,
    artifact: Bound<'_, PyAny>,
    payload: Bound<'_, PyDict>,
) -> PyResult<String> {
    let (record, _) = stage_artifact(&artifact)?;
    let staged = stage_envelope(&payload)?;
    let value = py_to_value(&payload.into_any())?;
    let artifact_type = record.artifact_type.to_owned();
    py.detach(|| -> Result<String, String> {
        config::check_envelope_keys(&artifact_type, &staged.keys, &staged.nested)?;
        if artifact_type == config::RL_PIPELINE_ARTIFACT {
            config::check_pipeline_envelope(&value)?;
        }
        let (digest, _) = config::identity_of_payload_without_digest(&value)?;
        Ok(digest)
    })
    .map_err(PyValueError::new_err)
}

/// `rl_validate_envelope_batch(artifact, payloads) -> list[str]`: the batch
/// form of [`rl_validate_envelope`] — one FFI crossing per batch, verified
/// digests in input order. Native owns the loop; Python never revalidates
/// row-by-row.
#[pyfunction]
fn rl_validate_envelope_batch(
    py: Python<'_>,
    artifact: Bound<'_, PyAny>,
    payloads: Bound<'_, PyList>,
) -> PyResult<Vec<String>> {
    let (record, _) = stage_artifact(&artifact)?;
    let mut staged_all = Vec::with_capacity(payloads.len());
    let mut values = Vec::with_capacity(payloads.len());
    for item in payloads.iter() {
        let payload = item.extract::<Bound<'_, PyDict>>().map_err(|_| {
            PyValueError::new_err(format!(
                "rl envelope batch items must be mappings, got {}",
                py_repr(&item).unwrap_or_else(|_| "?".to_owned()),
            ))
        })?;
        staged_all.push(stage_envelope(&payload)?);
        values.push(py_to_value(&payload.into_any())?);
    }
    let artifact_type = record.artifact_type.to_owned();
    py.detach(|| -> Result<Vec<String>, String> {
        let mut digests = Vec::with_capacity(values.len());
        for (staged, value) in staged_all.iter().zip(values.iter()) {
            config::check_envelope_keys(&artifact_type, &staged.keys, &staged.nested)?;
            if artifact_type == config::RL_PIPELINE_ARTIFACT {
                config::check_pipeline_envelope(value)?;
            }
            let (digest, _) = config::identity_of_payload_without_digest(value)?;
            digests.push(digest);
        }
        Ok(digests)
    })
    .map_err(PyValueError::new_err)
}

/// Register the RL config-registry surface onto the already-created
/// `hydra2._native.rl` submodule (mirrors the
/// `sub.add_function(wrap_pyfunction!(..., sub)?)?` shape at
/// `rows_seal.rs:485`; no new submodule, no new entry point). NativeWall
/// calls this from the `rl::register` registrar alongside `ffi::register`.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(rl_schema_digest, sub)?)?;
    sub.add_function(wrap_pyfunction!(rl_schema_record, sub)?)?;
    sub.add_function(wrap_pyfunction!(rl_validate_envelope, sub)?)?;
    sub.add_function(wrap_pyfunction!(rl_validate_envelope_batch, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lists_four_exact_envelopes() {
        assert_eq!(config::RL_SCHEMA_RECORDS.len(), 4);
        let versions: Vec<&str> =
            config::RL_SCHEMA_RECORDS.iter().map(|r| r.schema_version).collect();
        assert_eq!(
            versions,
            [
                config::RL_PIPELINE_SCHEMA_VERSION,
                config::RL_SCHEMA_VERSION,
                config::RL_SCHEMA_VERSION,
                config::RL_SCHEMA_VERSION,
            ]
        );
    }
}
