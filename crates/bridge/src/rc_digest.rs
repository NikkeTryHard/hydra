//! rc_digest: run-config digest over the `hydra-feed` canon/digest owner.
//!
//! DAG: this module depends on the feed crate (`canon`, `digest`) + pyo3 ONLY
//! (same edge as the rest of the bridge; no new dependency). FORBIDS: layout
//! mkdir/marker side effects (OS authority stays Python in
//! `training/_rc_digest.py`), YAML framing (the resolved-config mapping is
//! built Python-side via `run_config_to_dict`), wall-clock/RNG, and any new
//! JCS printer — all bytes/hash math lives in `hydra_feed::{canon, digest}`;
//! this file only stages the closed-domain dict attached, runs the feed pure
//! functions detached, and returns the `sha256:<hex>` text.
//!
//! PyO3 precedent cites (one per API): `pub fn register(sub)` + borrowed
//! `sub` in `wrap_pyfunction!(f, sub)` mirror `rows_seal::register`
//! (`crates/bridge/src/rows_seal.rs:484-485`); `#[pyfunction]` + the
//! stage-attached / `py.detach` / wrap-attached shape mirror
//! `canon_rng::batch_canonical_bytes`
//! (`crates/bridge/src/canon_rng.rs:376-394`); the staging arms mirror
//! `canon_rng::py_to_value` (`crates/bridge/src/canon_rng.rs:275-363`);
//! `Python::initialize()` + `Python::attach` in tests mirror
//! (`crates/bridge/src/canon_rng.rs:688-689`); `PyDict::new` + `set_item` in
//! tests mirror (`crates/bridge/src/validate.rs:181-183`).
//!
//! Feed owners (never reimplemented here): staged-`Value` JCS entry
//! `hydra_feed::canon::canonical_bytes_value` (`crates/feed/src/canon.rs:133`)
//! + the double-safe window `hydra_feed::canon::MAX_SAFE_INTEGER`
//!   (`crates/feed/src/canon.rs:146`) + `hydra_feed::digest::sha256_hex`
//!   (`crates/feed/src/digest.rs:31`, `sha256:`-prefixed lowercase hex,
//!   infallible — called bare below).
//!
//! Single-cdylib tree: registers its pyfn on the EXISTING
//! `hydra2._native.contracts` submodule via `register` (mirrors
//! `action_artifact`/`event_schema`/`validate` on `contracts`); no new
//! submodule, no new entry point. Wiring is MAIN-ONLY (`contracts::register`
//! calls this `register` with its `sub`, alongside
//! `crates/bridge/src/contracts.rs:1288-1291`).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PyString};

/// Stage one closed-domain Python object to a serde `Value` (attached only:
/// touches Python memory, never detached).
///
/// Mirrors `canon_rng::py_to_value` arm-for-arm (`None` -> null; `bool`
/// BEFORE `int` since Python `bool` subclasses `int`; `int` with
/// `|n| > MAX_SAFE_INTEGER` rejected, ints beyond `i64` over-window by
/// construction; finite `float` only; `str`, where lone surrogates surface as
/// the UTF-8 extraction failure; `list` element-wise, where `tuple` is NOT a
/// list and is rejected exactly like the Python authority; `dict` with
/// non-`str` keys rejected; anything else rejected). `record` names the
/// digest call for errors.
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
fn run_config_digest_value(value: &serde_json::Value) -> Result<String, String> {
    let record = "bridge:run_config_digest_from_dict";
    let bytes = hydra_feed::canon::canonical_bytes_value(value, record)
        .map_err(|e| format!("canon JCS emit rejected: {e:?}"))?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

/// Run-config digest from a plain mapping: one detach around JCS-emit + hash.
///
/// The caller (`training/_rc_digest.py::run_config_digest`) frames the
/// resolved config via `run_config_to_dict` (YAML-shaped plain dict, always
/// in-domain); this pyfn stages it (attached), digests it (detached), and
/// returns the `sha256:<hex>` text. Out-of-domain values fail the whole call
/// as `ValueError` (row-level record ids belong to the packet slice, not this
/// thin boundary). Counter-free deterministic, never wall-clock.
#[pyfunction]
fn run_config_digest_from_dict(py: Python<'_>, doc: Bound<'_, PyAny>) -> PyResult<String> {
    let staged =
        py_to_value(&doc, "bridge:run_config_digest_from_dict").map_err(PyValueError::new_err)?;
    py.detach(|| run_config_digest_value(&staged))
        .map_err(PyValueError::new_err)
}

/// Attach the digest pyfn to the caller-provided `contracts` submodule
/// (mirrors the `sub.add_function(wrap_pyfunction!(..., sub)?)?` shape at
/// `rows_seal.rs:485`; no new submodule, no new entry point).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(run_config_digest_from_dict, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod rc_digest_tests {
    use super::*;

    /// V1 oracle (`of_canonical({"b": 1, "a": 2})`, canon `{"a":2,"b":1}`):
    /// key order on the wire is canonical regardless of dict order.
    #[test]
    fn v1_key_order_irrelevant_golden() {
        Python::initialize();
        Python::attach(|py| {
            let doc = PyDict::new(py);
            doc.set_item("b", 1i64).unwrap();
            doc.set_item("a", 2i64).unwrap();
            assert_eq!(
                run_config_digest_from_dict(py, doc.into_any()).unwrap(),
                "sha256:d3626ac30a87e6f7a6428233b3c68299976865fa5508e4267c5415c76af7a772"
            );
        });
    }

    /// V2 oracle (nested dict + float + bool + null + list) over the pure
    /// helper: canon `{"a":{"m":-12.25,"n":0},"z":[3.5,true,null,"x"]}`.
    #[test]
    fn v2_nested_golden() {
        let value: serde_json::Value =
            serde_json::from_str(r#"{"z": [3.5, true, null, "x"], "a": {"m": -12.25, "n": 0}}"#)
                .unwrap();
        assert_eq!(
            run_config_digest_value(&value).unwrap(),
            "sha256:82835bf7c4e29658edc8161bd285aa9fd7e277e1636ef6d6deb41af057b74ca1"
        );
    }

    /// V3 oracle (unicode + `MAX_SAFE_INTEGER` edge + empty containers) over
    /// the pure helper: the name rides as raw UTF-8 in the canon bytes and
    /// `big` sits exactly on the double-safe window edge.
    #[test]
    fn v3_unicode_and_int_window_edge_golden() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{"name": "h\u00e9llo-\u4e16\u754c", "big": 9007199254740991, "empty": {}, "list": []}"#,
        )
        .unwrap();
        assert_eq!(
            run_config_digest_value(&value).unwrap(),
            "sha256:7b2a9ff5264c19722fd00ef11e69208c8a2b14e46b652b8de416b4bea7969f47"
        );
    }

    /// Boundary: integers outside the double-safe window fail closed, never
    /// silently rounded (Python ints beyond `i64` miss the `extract` and land
    /// on the same gate by construction).
    #[test]
    fn int_window_reject() {
        Python::initialize();
        Python::attach(|py| {
            let doc = PyDict::new(py);
            doc.set_item("n", 9_007_199_254_740_992i64).unwrap();
            assert!(run_config_digest_from_dict(py, doc.into_any()).is_err());
        });
    }
}
