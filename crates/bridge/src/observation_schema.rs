//! observation_schema: frozen observation-schema artifact identities.
//!
//! TABLE owner for `python/hydra2/contracts/observation_schema.py` consts:
//! the artifact-type/version strings plus the SPEC 2.2 envelope field tuple.
//! Builders, digest fns + cache, parsers, and every constraint table
//! (`_FIELD_CONSTRAINTS`, `_VISIBLE_MELD_ROW`, `_SEAT_SPEC`, ...) stay
//! Python-side (see remainder in the port report). MAIN wires this onto the
//! existing `hydra2._native.contracts` submodule (no new submodule entry).

// Precedent `use pyo3::prelude::*;` -> contracts.rs:52.
use pyo3::prelude::*;
// Precedent `PyModule` + `PyTuple` imports -> contracts.rs:53.
use pyo3::types::{PyModule, PyTuple};

/// Register the frozen observation-schema identities onto the `contracts`
/// submodule (mirrors `contracts::register`; single cdylib, no new entry).
///
/// Precedents: `register` signature shape -> contracts.rs:1114;
/// `let py = sub.py()` -> contracts.rs:1115;
/// `sub.add(<str>, <&str>)` -> contracts.rs:1153;
/// `PyTuple::new(py, [...])` for frozen tuples -> contracts.rs:1156.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    // `observation_schema.py:54`.
    sub.add(
        "OBSERVATION_SCHEMA_ARTIFACT_TYPE",
        "hydra2.observation_schema",
    )?;
    // `observation_schema.py:55`.
    sub.add("OBSERVATION_SCHEMA_SCHEMA_VERSION", "1.0.0")?;
    // `observation_schema.py:58-63`.
    sub.add(
        "_OBSERVATION_SCHEMA_ENVELOPE_FIELDS",
        PyTuple::new(
            py,
            [
                "artifact_type",
                "schema_version",
                "compatibility",
                "payload",
            ],
        )?,
    )?;
    Ok(())
}
