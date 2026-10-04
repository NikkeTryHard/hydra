//! rl/live_ffi: batch PyO3 surface for the native-live subtree.
//!
//! DAG: pyo3 + `super::live` (pure logic only) + serde_json bytes. Shape per
//! fn: attached staging (owned extraction) -> ONE `py.detach(|| ...)` with
//! zero Python API inside -> attached wrap as `PyValueError`. Results are
//! opaque canonical JSON bytes or digest strings the thin Python wrapper
//! forwards; row/policy internals never cross per-row. FORBIDS: YAML/file
//! IO, wall-clock, RNG seeding, Torch/CUDA access.
//!
//! Single-cdylib tree: registers onto the already-created `hydra2._native.rl`
//! submodule via `register`. No new top-level submodule, no new entry point.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use super::live;

fn live_error(detail: impl Into<String>) -> PyErr {
    PyValueError::new_err(format!("rl_live rejected: {}", detail.into()))
}

/// Deterministic snapshot-lineage identifier for the frozen behavior actor
/// collecting `generation` (g0 is C0; g1..7 hash the lineage object).
/// Returns the digest string; never a weight hash.
#[pyfunction]
#[pyo3(signature = (*, run_descriptor_json, generation, parent_policy_id))]
fn policy_identity(
    py: Python<'_>,
    run_descriptor_json: Vec<u8>,
    generation: u64,
    parent_policy_id: String,
) -> PyResult<String> {
    py.detach(|| live::policy_identity(&run_descriptor_json, generation, &parent_policy_id))
        .map_err(|e| live_error(e.0))
}

/// Archive-free collector binding for one generation (profile, context, and
/// frozen shuffle key only). Returns canonical JSON bytes.
#[pyfunction]
#[pyo3(signature = (*, run_descriptor_json, generation, policy_id))]
fn live_collector_spec(
    py: Python<'_>,
    run_descriptor_json: Vec<u8>,
    generation: u64,
    policy_id: String,
) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| live::live_collector_spec(&run_descriptor_json, generation, &policy_id))
        .map_err(|e| live_error(e.0))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Register typed replay directly from a drained tape (no seal stamps or
/// file round trips). Returns `(replay_handle, live_descriptor_bytes)`.
#[pyfunction]
#[pyo3(signature = (*, tape_handle, collector_spec_json, run_descriptor_json))]
fn register_live_replay(
    py: Python<'_>,
    tape_handle: u64,
    collector_spec_json: Vec<u8>,
    run_descriptor_json: Vec<u8>,
) -> PyResult<(u64, Py<PyBytes>)> {
    let (handle, descriptor) = py
        .detach(|| {
            crate::rl::replay_index::register_live_replay(
                tape_handle,
                &collector_spec_json,
                &run_descriptor_json,
            )
        })
        .map_err(|e| live_error(e.0))?;
    Ok((handle, PyBytes::new(py, &descriptor).unbind()))
}

/// Live production fill into caller-owned pinned buffers (Torch-owned
/// contiguous `uint8` spans): features plane-major plus interleaved 48B
/// facts. The replay must carry this exact tape handle in its live source;
/// no index bytes and no metas path cross on this hot path. Returns the
/// live slab descriptor bytes (no checksums or envelopes).
/// Caller contract: both spans stay alive, unresized, and non-overlapping
/// for the call (the learner's pinned slots satisfy this by construction,
/// matching `readinto` semantics).
#[pyfunction]
#[pyo3(signature = (*, tape_handle, replay_handle, batch_idx, features_buffer, facts_buffer))]
fn fill_live_slab(
    py: Python<'_>,
    tape_handle: u64,
    replay_handle: u64,
    batch_idx: usize,
    features_buffer: Bound<'_, PyAny>,
    facts_buffer: Bound<'_, PyAny>,
) -> PyResult<Py<PyBytes>> {
    use pyo3::buffer::PyBuffer;
    // Acquire and validate both exports attached; the guards stay alive
    // across detach below, so the buffer protocol locks the spans against
    // resize or destroy while the fill runs (a resizer raises instead of
    // reallocating under the live slice).
    let features_guard = PyBuffer::<u8>::get(&features_buffer)
        .map_err(|e| live_error(format!("features protocol rejected: {e}")))?;
    let facts_guard = PyBuffer::<u8>::get(&facts_buffer)
        .map_err(|e| live_error(format!("facts protocol rejected: {e}")))?;
    if !features_guard.is_c_contiguous() {
        return Err(live_error("features must be C-contiguous".to_string()));
    }
    if !facts_guard.is_c_contiguous() {
        return Err(live_error("facts must be C-contiguous".to_string()));
    }
    if features_guard.readonly() {
        return Err(live_error("features must be writable".to_string()));
    }
    if facts_guard.readonly() {
        return Err(live_error("facts must be writable".to_string()));
    }
    let span_of = |guard: &PyBuffer<u8>| -> (usize, usize) {
        let len = guard.len_bytes();
        if len == 0 {
            // Aligned dangling base: from_raw_parts_mut(null, 0) is UB even
            // unindexed, so empty spans never materialize a null pointer.
            (std::ptr::NonNull::<u8>::dangling().as_ptr() as usize, 0)
        } else {
            (guard.get_ptr(&[0]) as usize, len)
        }
    };
    let (feature_ptr, feature_len) = span_of(&features_guard);
    let (facts_ptr, facts_len) = span_of(&facts_guard);
    let overlaps = feature_ptr < facts_ptr.saturating_add(facts_len)
        && facts_ptr < feature_ptr.saturating_add(feature_len);
    if overlaps {
        return Err(live_error("spans must not overlap".to_string()));
    }
    let descriptor = py.detach(|| {
        // SAFETY: bases and lengths come from buffer exports held live
        // across this call (resize/destroy locked out on the exporter
        // side); the core gates its own prefix before touching any byte.
        let features =
            unsafe { std::slice::from_raw_parts_mut(feature_ptr as *mut u8, feature_len) };
        let facts = unsafe { std::slice::from_raw_parts_mut(facts_ptr as *mut u8, facts_len) };
        crate::rl::replay_index::fill_live_slab_buffers(
            tape_handle,
            replay_handle,
            batch_idx,
            features,
            facts,
        )
        .map_err(|e| live_error(e.0))
    })?;
    // Post-fill overlap recheck attached (defense-in-depth: the exports
    // were held across the fill, so the spans could not move or resize
    // underneath it; this only guards against aliasing the caller
    // introduced between the two checks).
    let feature_view = PyBuffer::<u8>::get(&features_buffer)
        .map_err(|e| live_error(format!("features lost: {e}")))?;
    let facts_view = PyBuffer::<u8>::get(&facts_buffer)
        .map_err(|e| live_error(format!("facts lost: {e}")))?;
    let post_span_of = |view: &PyBuffer<u8>| -> (usize, usize) {
        let len = view.len_bytes();
        if len == 0 {
            (std::ptr::NonNull::<u8>::dangling().as_ptr() as usize, 0)
        } else {
            (view.get_ptr(&[0]) as usize, len)
        }
    };
    let (post_feature_ptr, _) = post_span_of(&feature_view);
    let (post_facts_ptr, _) = post_span_of(&facts_view);
    let overlaps = post_feature_ptr < post_facts_ptr.saturating_add(facts_view.len_bytes())
        && post_facts_ptr < post_feature_ptr.saturating_add(feature_view.len_bytes());
    if overlaps {
        return Err(live_error("spans overlap after fill".to_string()));
    }
    Ok(PyBytes::new(py, &descriptor).unbind())
}

/// Shared slot geometry for one `(T, capacity)` learner lane (wraps the
/// single `replay_index::slab_geometry` home, never a second calculation).
#[pyfunction]
fn slab_geometry(py: Python<'_>, bucket_t: u32, capacity: usize) -> PyResult<(usize, usize)> {
    py.detach(|| crate::rl::replay_index::slab_geometry(bucket_t, capacity))
        .map_err(|e| live_error(e.0))
}

/// Validate a generation restart descriptor against the run descriptor
/// (closed shape, pipeline/C0/lineage gates). Returns canonical bytes.
#[pyfunction]
#[pyo3(signature = (*, run_descriptor_json, descriptor_json))]
fn validate_restart_descriptor(
    py: Python<'_>,
    run_descriptor_json: Vec<u8>,
    descriptor_json: Vec<u8>,
) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| live::validate_restart_descriptor(&run_descriptor_json, &descriptor_json))
        .map_err(|e| live_error(e.0))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Validate a restart pointer (closed shape, fixed locator law).
#[pyfunction]
#[pyo3(signature = (*, run_descriptor_json, pointer_json))]
fn validate_restart_pointer(
    py: Python<'_>,
    run_descriptor_json: Vec<u8>,
    pointer_json: Vec<u8>,
) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| live::validate_restart_pointer(&run_descriptor_json, &pointer_json))
        .map_err(|e| live_error(e.0))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Publish the restart pointer after its container landed (derives the
/// fixed locator; atomic temp + fsync + rename + dir fsync).
#[pyfunction]
#[pyo3(signature = (*, root_dir, run_descriptor_json, descriptor_json))]
fn publish_restart_pointer(
    py: Python<'_>,
    root_dir: String,
    run_descriptor_json: Vec<u8>,
    descriptor_json: Vec<u8>,
) -> PyResult<()> {
    py.detach(|| {
        live::publish_restart_pointer(&root_dir, &run_descriptor_json, &descriptor_json)
    })
    .map_err(|e| live_error(e.0))
}

/// Begin one live replay preparation (at most one outstanding): claims the
/// tape and spawns the registration worker. Returns the job id.
#[pyfunction]
#[pyo3(signature = (*, tape_handle, collector_spec_json, run_descriptor_json))]
fn begin_live_replay(
    py: Python<'_>,
    tape_handle: u64,
    collector_spec_json: Vec<u8>,
    run_descriptor_json: Vec<u8>,
) -> PyResult<u64> {
    py.detach(|| {
        live::begin_live_replay(tape_handle, collector_spec_json, run_descriptor_json)
    })
    .map_err(|e| live_error(e.0))
}

/// Poll one preparation job: `running`, or `done` with replay/descriptor.
/// A done poll consumes the job once (matching wait).
#[pyfunction]
#[pyo3(signature = (*, job_id))]
fn poll_live_replay(py: Python<'_>, job_id: u64) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| live::poll_live_replay(job_id))
        .map_err(|e| live_error(e.0))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Block for one preparation job and consume it (matching poll-done).
/// Returns `(replay_handle, descriptor_bytes)`.
#[pyfunction]
#[pyo3(signature = (*, job_id))]
fn wait_live_replay(py: Python<'_>, job_id: u64) -> PyResult<(u64, Py<PyBytes>)> {
    let (handle, descriptor) = py
        .detach(|| live::wait_live_replay(job_id))
        .map_err(|e| live_error(e.0))?;
    Ok((handle, PyBytes::new(py, &descriptor).unbind()))
}

/// Register the live fns on the already-created `hydra2._native.rl`
/// submodule (Main calls this from `rl/mod.rs` beside `lifecycle_ffi`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(policy_identity, sub)?)?;
    sub.add_function(wrap_pyfunction!(live_collector_spec, sub)?)?;
    sub.add_function(wrap_pyfunction!(register_live_replay, sub)?)?;
    sub.add_function(wrap_pyfunction!(fill_live_slab, sub)?)?;
    sub.add_function(wrap_pyfunction!(slab_geometry, sub)?)?;
    sub.add_function(wrap_pyfunction!(validate_restart_descriptor, sub)?)?;
    sub.add_function(wrap_pyfunction!(validate_restart_pointer, sub)?)?;
    sub.add_function(wrap_pyfunction!(publish_restart_pointer, sub)?)?;
    sub.add_function(wrap_pyfunction!(begin_live_replay, sub)?)?;
    sub.add_function(wrap_pyfunction!(poll_live_replay, sub)?)?;
    sub.add_function(wrap_pyfunction!(wait_live_replay, sub)?)?;
    Ok(())
}
