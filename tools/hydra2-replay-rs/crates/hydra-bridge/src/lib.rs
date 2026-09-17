//! hydra-bridge: thin S5 PyO3 boundary (Mutex-Option + detach + stats).
//!
//! DAG: this crate depends on feed + shard + search + pyo3 (columnar capsule surface
//! needs the shard expand/schema edge, the `search` act surface needs the arena edge;
//! Wave5 B6 records the shard change explicitly; the hydra-search member + path edge
//! are Arena-owned integration, see `search.rs` ASSUMPTION-MANIFEST). FORBIDS: parse/
//! encode/hash/pool/selection logic — arg-check + detach + struct return only.
//!
//! Owns the full `py_stream` handoff (plane-filling stream surface;
//! stats and quarantines stay readable post-close).
//! The extension module keeps its legacy name (`import hydra2_replay_rs`)
//! with the plane-filling `PyHydraStream` surface plus `canon_rng`/`columnar`/
//! `search` submodules (single cdylib, one `#[pymodule]`; Phase-6 maturin
//! `module-name` cutover re-exposes this same tree as
//! `hydra_bridge._native` without renaming this crate).

pub mod canon_rng;
pub mod columnar;
pub mod contracts;
pub mod eval;
pub mod mirror;
pub mod packet;
pub mod packet_decode;
pub mod replay;
pub mod resume;
pub mod ring;
pub mod search;
pub mod sink;
pub mod stream;
pub mod stream_driver;
pub mod tiles;
use pyo3::prelude::*;

/// Extension module: `import hydra2_replay_rs`.
#[pymodule]
pub fn hydra2_replay_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<stream::PyHydraStream>()?;
    m.add_class::<stream::PyFill>()?;
    m.add_class::<stream::PyStats>()?;
    m.add_class::<stream::PyQuar>()?;
    stream_driver::register(m)?;
    replay::register(m)?;
    canon_rng::register(m)?;
    resume::register(m)?;
    ring::register(m)?;
    columnar::register(m)?;
    contracts::register(m)?;
    search::register(m)?;
    packet::register(m)?;
    mirror::register(m)?;
    packet_decode::register(m)?;
    tiles::register(m)?;
    eval::register(m)?;
    Ok(())
}
