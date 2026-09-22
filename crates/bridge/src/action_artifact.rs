//! action_artifact: frozen SPEC 6 action-table artifact values.
//!
//! DAG: pyo3 ONLY (frozen literals; no feed dependency — the template census,
//! canon bytes, and digest owners stay Python-side, see `contracts` rank 6-7
//! docs). FORBIDS: digest computation (needs the Python canon authority plus
//! live `CanonicalActionTemplate` objects), file/JSON IO (OS authority stays
//! Python), and builder/loader logic (validating roots stay Python).
//!
//! Single-cdylib tree: registers its consts on the shared
//! `hydra2._native.contracts` submodule via `register` (mirrors
//! `contracts::register`); no new entry point.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Published artifact type string (`action_artifact.py:49`).
const ACTION_TABLE_ARTIFACT_TYPE: &str = "hydra2.action_table";

/// Frozen schema version text (`action_artifact.py:50`): `SchemaVersion` is a
/// `NewType` over `str` (`common.py:150`), so the runtime value is this text.
const ACTION_TABLE_SCHEMA_VERSION: &str = "1.0.0";

/// Artifact relpath text (`action_artifact.py:51`): `Path("configs") /
/// "contracts" / "action_table_v1.json"` joined (POSIX); Python re-wraps with
/// `Path(...)` to keep the `Path` type.
const ACTION_TABLE_RELPATH: &str = "configs/contracts/action_table_v1.json";

/// Exact template JSON field inventory (`action_artifact.py:53-61`), order
/// byte-exact (alphabetical); the loader requires exactly this set
/// (`action_artifact.py:216-218`).
const TEMPLATE_JSON_FIELDS: [&str; 7] = [
    "called_tile",
    "consumed_tiles",
    "declares_riichi",
    "kind",
    "meld_ref_required",
    "source_offset",
    "tile",
];

/// Register the action-artifact consts on the shared `contracts` submodule.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = sub.py();
    sub.add("ACTION_TABLE_ARTIFACT_TYPE", ACTION_TABLE_ARTIFACT_TYPE)?;
    sub.add("ACTION_TABLE_SCHEMA_VERSION", ACTION_TABLE_SCHEMA_VERSION)?;
    sub.add("ACTION_TABLE_RELPATH", ACTION_TABLE_RELPATH)?;
    sub.add(
        "TEMPLATE_JSON_FIELDS",
        PyTuple::new(py, TEMPLATE_JSON_FIELDS)?,
    )?;
    Ok(())
}
