//! rl/runtime_snapshot_ffi: batch PyO3 surface for runtime persistence.
//!
//! DAG: pyo3 + `super::runtime_snapshot{,_control,_pool,_tape}` (pure
//! native; never the lifecycle/pool/artifact leaf modules directly) +
//! `std::path`. Shape per fn: attached staging (owned extraction) -> ONE
//! `py.detach(|| ...)` with zero Python API inside -> attached wrap as
//! `PyValueError`. Results are opaque canonical bytes (or a tape handle)
//! the Main loop forwards; rows, arenas, and pool internals never cross.
//! FORBIDS: YAML/file IO in wrappers (paths only), wall-clock, RNG
//! seeding, Torch/CUDA access, per-row callbacks.
//!
//! Single-cdylib tree: registers onto the already-created `hydra2._native.rl`
//! submodule via `register` (registrar handoff by message, never a
//! concurrent mod edit). No new top-level submodule, no new entry point.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use super::runtime_snapshot::{
    checkpoint_sections as sections_native, write_runtime_blob as blob_native,
};
use super::runtime_snapshot_control::{restore_runtime as restore_native, runtime_control as control_native};
use super::runtime_snapshot_pool::{restore_pool_doc as pool_doc_native, save_pool as pool_native};
use super::runtime_snapshot_tape::{restore_tape as tape_restore_native, save_tape as tape_native};
use crate::rl::trajectory::TapeError;

fn runtime_error(context: &str, detail: String) -> PyErr {
    PyValueError::new_err(format!("rl_runtime {context} rejected: {detail}"))
}

fn tape_err(context: &'static str) -> impl Fn(TapeError) -> PyErr {
    move |e: TapeError| runtime_error(context, e.0)
}

/// Content-addressed publish of one canonical JSON document.
/// Returns canonical receipt bytes (`digest`, `locator`).
#[pyfunction]
#[pyo3(signature = (*, root_dir, document))]
fn write_runtime_blob(
    py: Python<'_>,
    root_dir: String,
    document: Vec<u8>,
) -> PyResult<Py<PyBytes>> {
    let out = py.detach(|| {
        blob_native(&PathBuf::from(root_dir), &document).map_err(tape_err("write_runtime_blob"))
    })?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Persist the FULL owned tape behind `tape_handle` (post-freeze only) as
/// event + tape CAS blobs. Returns tape ref bytes
/// (`digest`, `locator`, `row_count`, `game_count`).
#[pyfunction]
#[pyo3(signature = (*, root_dir, tape_handle))]
fn save_tape(py: Python<'_>, root_dir: String, tape_handle: u64) -> PyResult<Py<PyBytes>> {
    let out = py.detach(|| {
        tape_native(&PathBuf::from(root_dir), tape_handle).map_err(tape_err("save_tape"))
    })?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Restore a verified tape blob into a fresh registry handle (unstamped;
/// `restore_runtime` re-applies the tip-committed stamp). Returns the handle.
#[pyfunction]
#[pyo3(signature = (*, root_dir, tape_ref))]
fn restore_tape(py: Python<'_>, root_dir: String, tape_ref: Vec<u8>) -> PyResult<u64> {
    py.detach(|| {
        tape_restore_native(&PathBuf::from(root_dir), &tape_ref)
            .map_err(tape_err("restore_tape"))
    })
}

/// Persist an actual `PyNativePool.snapshot` payload as a CAS pool blob
/// (the CoreSnapshotFullBlob domain). Returns canonical bytes (`digest`,
/// `locator`, `frontier`, `terminal_set`, `tickets`, `captured_rows`).
#[pyfunction]
#[pyo3(signature = (*, root_dir, pool_snapshot_json))]
fn save_pool(
    py: Python<'_>,
    root_dir: String,
    pool_snapshot_json: String,
) -> PyResult<Py<PyBytes>> {
    let out = py.detach(|| {
        pool_native(&PathBuf::from(root_dir), &pool_snapshot_json).map_err(tape_err("save_pool"))
    })?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Return the verified pool snapshot bytes for `PyNativePool.restore`.
#[pyfunction]
#[pyo3(signature = (*, root_dir, pool_ref))]
fn restore_pool_doc(
    py: Python<'_>,
    root_dir: String,
    pool_ref: Vec<u8>,
) -> PyResult<Py<PyBytes>> {
    let out = py.detach(|| {
        pool_doc_native(&PathBuf::from(root_dir), &pool_ref)
            .map_err(tape_err("restore_pool_doc"))
    })?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Project the bindings' deterministic skeletons onto the true completed
/// cursors (`batch_cursor` local within `max_updates`, `completed_updates`
/// global cumulative, `micro_offset` always 0). Returns canonical bytes
/// (`sampler_state`, `training_state`).
#[pyfunction]
#[pyo3(signature = (*, generation_bindings_json, batch_cursor, completed_updates))]
fn checkpoint_sections(
    py: Python<'_>,
    generation_bindings_json: Vec<u8>,
    batch_cursor: u64,
    completed_updates: u64,
) -> PyResult<Py<PyBytes>> {
    let out = py.detach(|| {
        sections_native(&generation_bindings_json, batch_cursor, completed_updates)
            .map_err(tape_err("checkpoint_sections"))
    })?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Commit one runtime stop: validate the opaque contract state, verify
/// every referenced blob digest, assemble the exact registered control
/// via the lifecycle publisher, publish, advance the runtime pointer.
/// Returns the committed control tip bytes.
#[pyfunction]
#[pyo3(signature = (*, root_dir, descriptor_json, generation_bindings_json, learner_receipt_json, actor_receipt_json, runtime_state_json, expected_parent))]
#[allow(clippy::too_many_arguments)]
fn runtime_control(
    py: Python<'_>,
    root_dir: String,
    descriptor_json: Vec<u8>,
    generation_bindings_json: Option<Vec<u8>>,
    learner_receipt_json: Vec<u8>,
    actor_receipt_json: Vec<u8>,
    runtime_state_json: Vec<u8>,
    expected_parent: Option<String>,
) -> PyResult<Py<PyBytes>> {
    let out = py.detach(|| {
        control_native(
            &PathBuf::from(root_dir),
            &descriptor_json,
            generation_bindings_json.as_deref(),
            &learner_receipt_json,
            &actor_receipt_json,
            &runtime_state_json,
            expected_parent.as_deref(),
        )
        .map_err(tape_err("runtime_control"))
    })?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Verified resume from the last-committed runtime pointer (cross-checked
/// against the control tip). Returns canonical bytes (state, control,
/// learner/actor locators + receipts, current/future pool/tape/index
/// locators, live tape handles). No inference, resampling, or new RNG.
#[pyfunction]
#[pyo3(signature = (*, root_dir, descriptor_json))]
fn restore_runtime(
    py: Python<'_>,
    root_dir: String,
    descriptor_json: Vec<u8>,
) -> PyResult<Py<PyBytes>> {
    let out = py.detach(|| {
        restore_native(&PathBuf::from(root_dir), &descriptor_json)
            .map_err(tape_err("restore_runtime"))
    })?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Register the runtime persistence surface onto the already-created `rl`
/// submodule (registrar-owned `rl/mod.rs` calls this after
/// `lifecycle_ffi::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(write_runtime_blob, sub)?)?;
    sub.add_function(wrap_pyfunction!(save_tape, sub)?)?;
    sub.add_function(wrap_pyfunction!(restore_tape, sub)?)?;
    sub.add_function(wrap_pyfunction!(save_pool, sub)?)?;
    sub.add_function(wrap_pyfunction!(restore_pool_doc, sub)?)?;
    sub.add_function(wrap_pyfunction!(checkpoint_sections, sub)?)?;
    sub.add_function(wrap_pyfunction!(runtime_control, sub)?)?;
    sub.add_function(wrap_pyfunction!(restore_runtime, sub)?)?;
    Ok(())
}
