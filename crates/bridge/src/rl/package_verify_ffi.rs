//! rl/package_verify_ffi: batch PyO3 surface over `rl::package_verify`.
//!
//! DAG: pyo3 + `super::package_verify` + serde-free opaque bytes ONLY (no
//! feed/shard/search edge; the verifier owns its file IO detached, so no
//! crate owner exists — same NONE position as `rc_require.rs:32-38`).
//! FORBIDS: per-row Python callbacks, wall-clock/RNG, `ContractError`
//! shaping (this layer raises `ValueError`; the thin Python translator maps
//! to `ContractError`), any new JCS printer (all bytes/hash math lives in
//! `hydra_feed::{canon}` via `super::package_verify`), and verification
//! bypasses (every argument crosses into the verifier; success is a verdict
//! receipt, never an exception-free pass).
//!
//! Shape per fn: attached staging (owned `String` extraction) → ONE
//! `py.detach(|| …)` over owned plain data with zero Python API inside (per
//! `contracts.rs:434-437`, `validate.rs:83-95`) → attached wrap as
//! `PyValueError` (per `contracts.rs:117-123`). Results are opaque canonical
//! receipt bytes the Python translator decodes; no row loops, no hash logic
//! in Python.
//!
//! Single-cdylib tree: registers onto the already-created `hydra2._native.rl`
//! submodule via `register` (mirrors `canon_rng::register` /
//! `walls::register` taking the submodule). NativeIntegrate owns `rl/mod.rs`
//! and calls this from the `rl::register` registrar alongside
//! `ffi::register`; this file is never wired anywhere else.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `verify_work_package(*, work_package, artifact_root) -> bytes`: run the
/// pure-Rust completion verifier and return the canonical receipt bytes.
/// `Ok` receipts carry `verdict` passed|failed|blocked; fail-closed argument
/// or publication trouble raises `ValueError` (never a false pass).
#[pyfunction]
#[pyo3(signature = (*, work_package, artifact_root))]
fn verify_work_package(py: Python<'_>, work_package: String, artifact_root: String) -> PyResult<Vec<u8>> {
    py.detach(|| {
        super::package_verify::verify_work_package(&work_package, &artifact_root)
            .map_err(|e| PyValueError::new_err(format!("package_verify verify rejected: {}", e.0)))
    })
}

/// Register the package-verify surface onto the already-created
/// `hydra2._native.rl` submodule (mirrors the
/// `sub.add_function(wrap_pyfunction!(..., sub)?)?` shape at
/// `rows_seal.rs:485`; no new submodule, no new entry point).
/// NativeIntegrate calls this from the `rl::register` registrar.
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(verify_work_package, sub)?)?;
    Ok(())
}
