//! validate: frozen validation taxonomy + seal leaf on the shared `contracts` submodule.
//!
//! DAG: `hydra-feed` (`validate` + `digest` owners) + pyo3 + `serde_json`
//! ONLY (both already bridge dependencies, see `crates/bridge/Cargo.toml:19,23`;
//! no new dependency). FORBIDS: record validation over live objects (the
//! `validate_batch` / `validation_hash_of` pyfns stay `packet_decode`-owned —
//! they take `PyGameRecord` handles, never plain values), engine/sim probing
//! (trap-8: `RiichiEnvExactSimulator.reset` stays Python, Rust takes the
//! `adapter_ok` flag), and file/JSON IO (rules-doc loading stays Python).
//!
//! TABLE ported here (frozen values + pure compute, single feed-owned source):
//! - `VALIDATION_CHECK_ORDER`: re-export of `feed::validate::CHECK_ORDER`
//!   (oracle insertion order, `validate.rs:96-107`).
//! - `VALIDATION_ERROR_CLASSES`: rendered from `feed::validate::ErrorClass::as_str`
//!   (`validate.rs:128-137`) — derived from the enum, never a second copy.
//! - `VALIDATION_SKIPPED_ADAPTER_PREFIX` / `VALIDATION_ADAPTER_NOT_WIRED`:
//!   the oracle-exact skipped label family (`validate.rs:447-450`, bridge pin
//!   `packet_decode.rs:108`).
//! - `validation_hash(game_id, checks)`: the `compute_validation_hash`
//!   (`data/validate.py:74-76`) seal leaf — `sha256(canonical({game_id,
//!   checks}))` through the feed digest owner. NOTE: `packet_decode` already
//!   exposes `validation_hash_of(record)` (seal of a live record); this fn is
//!   the value-level seal over an explicit checks map, a different shape.
//! - `is_validation_error_class` / `is_validation_check`: narrowing
//!   predicates over the two tables above.
//!
//! LOGIC staying Python (`data/validate.py`): `ValidationError` /
//! `ValidationOutcome` dataclasses, `validate_game` orchestration (opaque
//! decode handle + adapter probe + outcome shaping), `_adapter_ok` (engine
//! sim + rules-file IO), `_require_packet_decode` / `_require_contracts`
//! fail-closed import guards.
//!
//! Single-cdylib tree: registers on the shared `hydra2._native.contracts`
//! submodule via `register` (mirrors `contracts::register`); no new entry point.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};

use hydra_feed::validate::{CHECK_ORDER, ErrorClass};

/// Skipped-adapter check-value prefix (`validate.rs:447-450`): the legality
/// value is `skipped_adapter_error:<err_name>` when the trap-8 probe fails.
const SKIPPED_ADAPTER_PREFIX: &str = "skipped_adapter_error:";
/// Bridge-pinned probe label: no Python exception exists on the Rust path,
/// so the bridge pins one name (mirrors `packet_decode.rs:108`).
const ADAPTER_NOT_WIRED: &str = "NotWired";

/// Closed validation error taxonomy, rendered from the feed owner
/// (`ErrorClass::as_str`, `validate.rs:128-137`): enum order, byte-exact.
fn error_class_table() -> [&'static str; 6] {
    [
        ErrorClass::Structure.as_str(),
        ErrorClass::EventOrder.as_str(),
        ErrorClass::TileConservation.as_str(),
        ErrorClass::RedIdentity.as_str(),
        ErrorClass::Legality.as_str(),
        ErrorClass::DoraShape.as_str(),
    ]
}

/// Seal leaf: `sha256(canonical({game_id, checks}))` (mirrors
/// `compute_validation_hash`, `data/validate.py:74-76`, and the feed
/// `seal_hash`, `validate.rs:288-297`). Dict staged attached (str-only —
/// non-str keys/values fail closed); canon + digest run under ONE detach
/// with zero Python API inside; `ValueError` on any reject, never a default.
#[pyfunction]
fn validation_hash(py: Python<'_>, game_id: String, checks: Bound<'_, PyDict>) -> PyResult<String> {
    let mut pairs: Vec<(String, String)> = Vec::with_capacity(checks.len());
    for (key, value) in checks.iter() {
        let key: String = key.extract().map_err(|_| {
            PyValueError::new_err("contracts validation_hash checks keys must be str")
        })?;
        let value: String = value.extract().map_err(|_| {
            PyValueError::new_err("contracts validation_hash checks values must be str")
        })?;
        pairs.push((key, value));
    }
    py.detach(|| -> Result<String, String> {
        let mut map = serde_json::Map::with_capacity(pairs.len());
        for (key, value) in &pairs {
            map.insert(key.clone(), serde_json::Value::String(value.clone()));
        }
        let payload = serde_json::json!({
            "game_id": game_id,
            "checks": serde_json::Value::Object(map),
        });
        hydra_feed::digest::of_canonical(&payload)
            .map_err(|err| format!("contracts validation_hash rejected: {err:?}"))
    })
    .map_err(PyValueError::new_err)
}

/// Narrowing predicate: true iff the value names a closed validation error class.
#[pyfunction]
fn is_validation_error_class(value: &str) -> bool {
    error_class_table().contains(&value)
}

/// Narrowing predicate: true iff the value names a validation check in oracle order.
#[pyfunction]
fn is_validation_check(value: &str) -> bool {
    CHECK_ORDER.contains(&value)
}

/// Register the validation taxonomy + seal leaf on the shared `contracts` submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("VALIDATION_CHECK_ORDER", PyTuple::new(py, CHECK_ORDER)?)?;
    sub.add(
        "VALIDATION_ERROR_CLASSES",
        PyTuple::new(py, error_class_table())?,
    )?;
    sub.add("VALIDATION_SKIPPED_ADAPTER_PREFIX", SKIPPED_ADAPTER_PREFIX)?;
    sub.add("VALIDATION_ADAPTER_NOT_WIRED", ADAPTER_NOT_WIRED)?;
    sub.add_function(wrap_pyfunction!(validation_hash, sub)?)?;
    sub.add_function(wrap_pyfunction!(is_validation_error_class, sub)?)?;
    sub.add_function(wrap_pyfunction!(is_validation_check, sub)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;

    #[test]
    fn taxonomy_tables_match_feed_owner() {
        assert_eq!(
            CHECK_ORDER,
            [
                "structure",
                "event_order",
                "tile_conservation",
                "red_identity",
                "legality",
                "calls",
                "scores",
                "termination",
                "dora_shape",
                "trailing_data",
            ]
        );
        assert_eq!(
            error_class_table(),
            [
                "structure",
                "event_order",
                "tile_conservation",
                "red_identity",
                "legality",
                "dora_shape",
            ]
        );
        assert!(is_validation_check("structure"));
        assert!(is_validation_check("trailing_data"));
        assert!(!is_validation_check("bogus"));
        assert!(!is_validation_check(""));
        for class in error_class_table() {
            assert!(is_validation_error_class(class));
        }
        assert!(!is_validation_error_class("trailing_data"));
        assert!(!is_validation_error_class("other"));
        assert_eq!(
            format!("{SKIPPED_ADAPTER_PREFIX}{ADAPTER_NOT_WIRED}"),
            "skipped_adapter_error:NotWired"
        );
    }

    #[test]
    fn seal_leaf_matches_feed_digest_owner() {
        Python::initialize();
        Python::attach(|py| {
            let checks = PyDict::new(py);
            checks.set_item("structure", "ok").unwrap();
            checks
                .set_item("legality", "skipped_adapter_error:NotWired")
                .unwrap();
            let got = validation_hash(py, "g1".to_owned(), checks).unwrap();
            let mut map = serde_json::Map::new();
            map.insert(
                "structure".to_owned(),
                serde_json::Value::String("ok".to_owned()),
            );
            map.insert(
                "legality".to_owned(),
                serde_json::Value::String("skipped_adapter_error:NotWired".to_owned()),
            );
            let payload = serde_json::json!({
                "game_id": "g1",
                "checks": serde_json::Value::Object(map),
            });
            let expect = hydra_feed::digest::of_canonical(&payload).unwrap();
            assert_eq!(got, expect);
            assert!(got.starts_with("sha256:"));
            assert_eq!(got.len(), "sha256:".len() + 64);
            // Key order on the wire is canonical: reversed insertion still seals equal.
            let rev = PyDict::new(py);
            rev.set_item("legality", "skipped_adapter_error:NotWired")
                .unwrap();
            rev.set_item("structure", "ok").unwrap();
            assert_eq!(validation_hash(py, "g1".to_owned(), rev).unwrap(), expect);
            // Non-str values fail closed.
            let bad = PyDict::new(py);
            bad.set_item("structure", 7).unwrap();
            assert!(validation_hash(py, "g1".to_owned(), bad).is_err());
        });
    }
}
