//! hydra-bridge: thin S5 PyO3 boundary (Mutex-Option + detach + stats).
//!
//! DAG: this crate depends on the feed crate + pyo3 ONLY (no cold-crate edge;
//! CI triple-grep enforces). FORBIDS: parse/encode/hash/pool logic —
//! arg-check + detach + struct return only.
//!
//! P3-B: owns the full `py_stream` handoff (the root facade is deleted).
//! The extension module keeps its legacy name (`import hydra2_replay_rs`)
//! with the plane-filling `PyHydraStream` surface (plan §6.5).

pub mod replay;
pub mod sink;
pub mod stream;
use pyo3::prelude::*;

/// Extension module: `import hydra2_replay_rs`.
#[pymodule]
pub fn hydra2_replay_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<stream::PyHydraStream>()?;
    m.add_class::<stream::PyFill>()?;
    m.add_class::<stream::PyStats>()?;
    m.add_class::<stream::PyQuar>()?;
    replay::register(m)?;
    Ok(())
}
