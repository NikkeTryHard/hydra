//! hydra-bridge: thin S5 PyO3 boundary (Mutex-Option + detach + stats).
//!
//! DAG: this crate depends on feed + shard + pyo3 (columnar capsule surface
//! needs the shard expand/schema edge; Wave5 B6 records the change explicitly;
//! the no-cold-crate rule it replaces is superseded). FORBIDS: parse/encode/hash/pool logic —
//! arg-check + detach + struct return only.
//!
//! Owns the full `py_stream` handoff (plane-filling stream surface;
//! stats and quarantines stay readable post-close).
//! The extension module keeps its legacy name (`import hydra2_replay_rs`)
//! with the plane-filling `PyHydraStream` surface plus a `canon_rng`
//! submodule (single cdylib, one `#[pymodule]`; Phase-6 maturin
//! `module-name` cutover re-exposes this same tree as
//! `hydra_bridge._native` without renaming this crate).

pub mod canon_rng;
pub mod columnar;
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
    canon_rng::register(m)?;
    columnar::register(m)?;
    Ok(())
}
