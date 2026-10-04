//! Frozen-replay PyO3 surface: order building, cold slab oracle, slot
//! geometry, one-shot replay registration, teardown, and the production
//! direct fill. Called by the parent `artifact_ffi` registrar onto the same
//! `rl_artifact` submodule, so stable Python names never move.
//!
//! Lifecycle (one registration per generation/resume): `build_order` still
//! yields the order descriptor plus opaque index bytes (persisted once);
//! `register_replay` freezes them natively into a replay handle;
//! `fill_slab` serves each optimizer batch from `(tape, replay)` handles
//! with no index bytes and no metas path on the hot path; `drop_replay`
//! releases the handle. `export_slab` stays the cold tooling oracle on the
//! identical encoder/value law.

use pyo3::prelude::*;
use serde_json::Value;

use crate::rl::replay_index::{ShuffleKey, export_learner_slab};
use crate::rl::trajectory::TapeError;

use super::{artifact_err, tape_err};

fn shuffle_key_from_json(doc: &[u8]) -> Result<ShuffleKey, String> {
    let value: Value =
        serde_json::from_slice(doc).map_err(|e| format!("shuffle key is not JSON: {e}"))?;
    let map = value
        .as_object()
        .ok_or_else(|| "shuffle key must be an object".to_string())?;
    let text = |name: &str| {
        map.get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("shuffle key lacks {name}"))
            .map(str::to_string)
    };
    let num = |name: &str| {
        map.get(name)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("shuffle key lacks {name}"))
    };
    Ok(ShuffleKey {
        master_seed_hex: text("master_seed_hex")?,
        pipeline_spec_id: text("pipeline_spec_id")?,
        experiment_id: text("experiment_id")?,
        generation: num("generation")?,
        epoch: num("epoch")?,
        replicate_id: num("replicate_id")?,
        attempt_id: num("attempt_id")?,
    })
}

#[pyfunction]
fn build_order(
    py: Python<'_>,
    metas_path: String,
    shuffle_key_doc: Vec<u8>,
) -> PyResult<(Vec<u8>, Vec<u8>)> {
    let key =
        shuffle_key_from_json(&shuffle_key_doc).map_err(|detail| artifact_err("build_order", detail))?;
    py.detach(|| {
        let metas_bytes = std::fs::read(&metas_path)
            .map_err(|e| tape_err("build_order")(TapeError(format!("metas unreadable: {e}"))))?;
        let (plan, index_bytes) =
            crate::rl::replay_index::build_order(&metas_bytes, &key).map_err(tape_err("build_order"))?;
        let descriptor = serde_json::json!({
            "artifact_digest": plan.artifact_digest,
            "order_digest": plan.order_digest,
            "row_count": plan.row_count,
            "batches": plan.batches.iter().map(|b| serde_json::json!({
                "t": b.t,
                "count": b.positions.len(),
            })).collect::<Vec<_>>(),
        });
        let descriptor_bytes =
            hydra_feed::canon::canonical_bytes_value(&descriptor, "rl_artifact:order")
                .map_err(|e| {
                    tape_err("build_order")(TapeError(format!("order JCS failed: {e:?}")))
                })?;
        Ok((descriptor_bytes, index_bytes))
    })
}

/// Cold export (gate fixtures/tooling): one owned tuple per optimizer
/// batch. Production uses `fill_slab` (caller-owned pinned buffers).
#[pyfunction]
fn export_slab(
    py: Python<'_>,
    tape_handle: u64,
    index_bytes: Vec<u8>,
    metas_path: String,
    batch_idx: usize,
) -> PyResult<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)> {
    py.detach(|| {
        let metas_bytes = std::fs::read(&metas_path)
            .map_err(|e| tape_err("export_slab")(TapeError(format!("metas unreadable: {e}"))))?;
        let slab =
            export_learner_slab(tape_handle, &index_bytes, &metas_bytes, batch_idx)
                .map_err(tape_err("export_slab"))?;
        let spec_bytes = crate::rl::replay_index::slab_spec_bytes(&slab)
            .map_err(tape_err("export_slab"))?;
        Ok((
            spec_bytes,
            slab.features,
            slab.old_lp_le,
            slab.raw_a_le,
            slab.g4_f64_le,
            slab.selected_i64_le,
            slab.legal_bytes,
            slab.facts48,
            slab.batch_envelope,
        ))
    })
}

/// Cold slot geometry for one `(T, capacity)` learner lane: exact
/// `(feature_bytes, fact_bytes)` the fill spans validate against.
#[pyfunction]
fn slab_geometry(py: Python<'_>, bucket_t: u32, capacity: usize) -> PyResult<(usize, usize)> {
    py.detach(|| {
        crate::rl::replay_index::slab_geometry(bucket_t, capacity)
            .map_err(tape_err("slab_geometry"))
    })
}

/// Freeze one generation's order index plus its metas companion natively
/// (cold, exactly once per generation/resume): parses and validates both
/// documents once, freezes typed meta rows, and precomputes per-batch
/// layout and offsets. Returns the process replay handle the production
/// `fill_slab` serves. The tape binding is checked per fill against the
/// tape's seal stamp, so a stale or wrong-source fill rejects.
#[pyfunction]
fn register_replay(py: Python<'_>, index_bytes: Vec<u8>, metas_path: String) -> PyResult<u64> {
    py.detach(|| {
        let metas_bytes = std::fs::read(&metas_path)
            .map_err(|e| tape_err("register_replay")(TapeError(format!("metas unreadable: {e}"))))?;
        crate::rl::replay_index::register_replay_from_bytes(&index_bytes, &metas_bytes)
            .map_err(tape_err("register_replay"))
    })
}

/// Release one frozen replay handle (generation teardown). Unknown handles
/// fail closed, so a double-drop or use-after-drop is loud.
#[pyfunction]
fn drop_replay(py: Python<'_>, replay_handle: u64) -> PyResult<()> {
    py.detach(|| {
        crate::rl::replay_index::drop_replay(replay_handle).map_err(tape_err("drop_replay"))
    })
}

/// Finished generations release their native rows and event arenas only after
/// durable publication and replay teardown; stale handles must fail loudly.
#[pyfunction]
fn drop_tape(py: Python<'_>, tape_handle: u64) -> PyResult<()> {
    py.detach(|| {
        if crate::rl::trajectory::prep_owns_tape(tape_handle) {
            return Err(artifact_err(
                "drop_tape",
                "live replay preparation owns tape".to_string(),
            ));
        }
        if crate::rl::trajectory::remove_tape(tape_handle) {
            Ok(())
        } else {
            Err(artifact_err("drop_tape", "unknown tape handle".to_string()))
        }
    })
}

/// Production fill into caller-owned pinned buffers (Torch-owned contiguous
/// `uint8` spans): features plane-major plus interleaved 48B facts
/// (selected i64 @0, old-LP f32 @8, raw-A f32 @12, G4 f64x4 @16). The order
/// index and metas companion arrive frozen via the replay handle (one
/// `register_replay` per generation); this hot path takes no index bytes
/// and no metas path, re-parses nothing, and stages no slab-sized
/// intermediate. Span-validated; tails fill the prefix. Returns the
/// canonical spec bytes.
/// Staging-adjacent work runs detached over raw span pointers; the two spans
/// are re-validated on re-acquired views attached (a concurrent resize fails
/// closed instead of truncating). Caller contract: both spans stay alive,
/// unresized, and non-overlapping for the call (the learner's pinned slots
/// satisfy this by construction, matching `readinto` semantics).
#[pyfunction]
fn fill_slab(
    py: Python<'_>,
    tape_handle: u64,
    replay_handle: u64,
    batch_idx: usize,
    features_buffer: Bound<'_, PyAny>,
    facts_buffer: Bound<'_, PyAny>,
) -> PyResult<Vec<u8>> {
    use pyo3::buffer::PyBuffer;
    // Acquire and validate both exports attached; the guards stay alive
    // across detach below, so the buffer protocol locks the spans against
    // resize or destroy while the fill runs (a resizer raises instead of
    // reallocating under the live slice).
    let features_guard = PyBuffer::<u8>::get(&features_buffer)
        .map_err(|e| artifact_err("fill_slab", format!("features protocol rejected: {e}")))?;
    let facts_guard = PyBuffer::<u8>::get(&facts_buffer)
        .map_err(|e| artifact_err("fill_slab", format!("facts protocol rejected: {e}")))?;
    if !features_guard.is_c_contiguous() {
        return Err(artifact_err("fill_slab", "features must be C-contiguous".to_string()));
    }
    if !facts_guard.is_c_contiguous() {
        return Err(artifact_err("fill_slab", "facts must be C-contiguous".to_string()));
    }
    if features_guard.readonly() {
        return Err(artifact_err("fill_slab", "features must be writable".to_string()));
    }
    if facts_guard.readonly() {
        return Err(artifact_err("fill_slab", "facts must be writable".to_string()));
    }
    let span_of = |guard: &PyBuffer<u8>| -> (usize, usize) {
        let len = guard.len_bytes();
        if len == 0 {
            // Aligned dangling base: from_raw_parts_mut(null, 0) is UB even
            // unindexed, so empty spans never materialize a null pointer.
            // `[0]` for the 1-D base (see `pool_ffi::buffer_span`): `&[]`
            // reads out of bounds in `PyBuffer_GetPointer`.
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
        return Err(artifact_err("fill_slab", "spans must not overlap".to_string()));
    }
    // Fill detached, directly into the caller spans (no staged copies).
    // The exports above stay alive across this call, locking the spans;
    // the core span-gates before touching any byte.
    let spec_bytes = py.detach(|| {
        // SAFETY: bases and lengths come from buffer exports held live
        // across this call (resize/destroy locked out on the exporter
        // side); the core gates its own prefix before touching any byte.
        let features =
            unsafe { std::slice::from_raw_parts_mut(feature_ptr as *mut u8, feature_len) };
        let facts = unsafe { std::slice::from_raw_parts_mut(facts_ptr as *mut u8, facts_len) };
        let spec = crate::rl::replay_index::fill_slab_buffers(
            tape_handle,
            replay_handle,
            batch_idx,
            features,
            facts,
        )
        .map_err(tape_err("fill_slab"))?;
        crate::rl::replay_index::slab_spec_bytes_of(&spec).map_err(tape_err("fill_slab"))
    })?;
    // Post-fill overlap recheck attached (defense-in-depth: the exports
    // were held across the fill, so the spans could not move or resize
    // underneath it; this only guards against aliasing the caller
    // introduced between the two checks).
    let feature_view = PyBuffer::<u8>::get(&features_buffer)
        .map_err(|e| artifact_err("fill_slab", format!("features lost: {e}")))?;
    let facts_view = PyBuffer::<u8>::get(&facts_buffer)
        .map_err(|e| artifact_err("fill_slab", format!("facts lost: {e}")))?;
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
        return Err(artifact_err("fill_slab", "spans overlap after fill".to_string()));
    }
    Ok(spec_bytes)
}

/// Register the frozen-replay FFI on the shared `rl_artifact` submodule
/// (stable Python names; called by the parent registrar).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(build_order, sub)?)?;
    sub.add_function(wrap_pyfunction!(export_slab, sub)?)?;
    sub.add_function(wrap_pyfunction!(slab_geometry, sub)?)?;
    sub.add_function(wrap_pyfunction!(register_replay, sub)?)?;
    sub.add_function(wrap_pyfunction!(drop_replay, sub)?)?;
    sub.add_function(wrap_pyfunction!(drop_tape, sub)?)?;
    sub.add_function(wrap_pyfunction!(fill_slab, sub)?)?;
    Ok(())
}
