//! rl/lifecycle_ffi: batch PyO3 surface for the cold lifecycle subtree.
//!
//! DAG: pyo3 + `super::lifecycle` (facade only — never the scoped leaf
//! modules directly) + serde_json + `hydra_feed::canon` (report/digest
//! bytes). Shape per fn: attached staging (owned extraction) -> ONE
//! `py.detach(|| ...)` with zero Python API inside -> attached wrap as
//! `PyValueError`. Results are opaque canonical JSON bytes the thin Python
//! wrapper forwards; row/policy internals never cross per-row. FORBIDS:
//! YAML/file IO, wall-clock, RNG seeding, Torch/CUDA access.
//!
//! Single-cdylib tree: registers onto the already-created `hydra2._native.rl`
//! submodule via `register`. Main owns `rl/mod.rs` and calls it after
//! `pool_ffi::register` (registrar handoff by message, never a concurrent
//! mod edit). No new top-level submodule, no new entry point.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use super::lifecycle;

fn lifecycle_error(detail: impl Into<String>) -> PyErr {
    PyValueError::new_err(format!("rl_lifecycle rejected: {}", detail.into()))
}

/// Cold run descriptor: validate config + manifest against real repo
/// assets, then form every structural identity natively. Returns canonical
/// descriptor bytes.
#[pyfunction]
#[pyo3(signature = (*, run_config_json, initial_manifest_json, root_dir, repo_dir))]
fn prepare_run(
    py: Python<'_>,
    run_config_json: String,
    initial_manifest_json: String,
    root_dir: String,
    repo_dir: String,
) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| {
            lifecycle::prepare_run(&run_config_json, &initial_manifest_json, &root_dir, &repo_dir)
        })
        .map_err(|e| lifecycle_error(e.0))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Runtime collector arguments bind a real behavior policy before sealing;
/// no placeholder policy may enter the profile, ledgers, or artifact header.
#[pyfunction]
#[pyo3(signature = (*, run_descriptor_json, generation, policy_checkpoint_id))]
fn collector_spec(
    py: Python<'_>,
    run_descriptor_json: Vec<u8>,
    generation: u64,
    policy_checkpoint_id: String,
) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| lifecycle::collector_spec(&run_descriptor_json, generation, &policy_checkpoint_id))
        .map_err(|e| lifecycle_error(e.0))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Generation boundary bindings: canonical RunSpec + header + skeletons
/// after the validated seal + order. The actor id stays the source
/// behavior C_g (never the first-update C_{g+1}).
#[pyfunction]
#[pyo3(signature = (*, run_descriptor_json, generation, actor_manifest_id, learner_initial_manifest_id, artifact_receipt_json, order_descriptor_json))]
#[allow(clippy::too_many_arguments)]
fn prepare_generation(
    py: Python<'_>,
    run_descriptor_json: Vec<u8>,
    generation: u64,
    actor_manifest_id: String,
    learner_initial_manifest_id: String,
    artifact_receipt_json: Vec<u8>,
    order_descriptor_json: Vec<u8>,
) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| {
            lifecycle::prepare_generation(
                &run_descriptor_json,
                generation,
                &actor_manifest_id,
                &learner_initial_manifest_id,
                &artifact_receipt_json,
                &order_descriptor_json,
            )
        })
        .map_err(|e| lifecycle_error(e.0))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Assemble the control document from landed receipts and runtime facts
/// (no digest here; `publish_control` digests and commits).
#[pyfunction]
#[pyo3(signature = (*, generation_bindings_json, learner_receipt_json, actor_receipt_json, pool_snapshot_digest, tape_blob_digest, tape_blob_locator, event_blob_digest, event_blob_locator, frontier_current_json, frontier_next_json, terminal_set, tickets_json, committed_row_offset, committed_update_offset, publication_phase, generation_statuses_json, source_refs_json, ledger_refs_json))]
#[allow(clippy::too_many_arguments)]
fn build_control(
    py: Python<'_>,
    generation_bindings_json: Vec<u8>,
    learner_receipt_json: Vec<u8>,
    actor_receipt_json: Vec<u8>,
    pool_snapshot_digest: String,
    tape_blob_digest: String,
    tape_blob_locator: String,
    event_blob_digest: String,
    event_blob_locator: String,
    frontier_current_json: String,
    frontier_next_json: String,
    terminal_set: Vec<String>,
    tickets_json: Vec<u8>,
    committed_row_offset: u64,
    committed_update_offset: u64,
    publication_phase: String,
    generation_statuses_json: Vec<u8>,
    source_refs_json: Vec<u8>,
    ledger_refs_json: Vec<u8>,
) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| {
            lifecycle::build_control(
                &generation_bindings_json,
                &learner_receipt_json,
                &actor_receipt_json,
                &pool_snapshot_digest,
                &tape_blob_digest,
                &tape_blob_locator,
                &event_blob_digest,
                &event_blob_locator,
                &frontier_current_json,
                &frontier_next_json,
                &terminal_set,
                &tickets_json,
                committed_row_offset,
                committed_update_offset,
                &publication_phase,
                &generation_statuses_json,
                &source_refs_json,
                &ledger_refs_json,
            )
        })
        .map_err(|e| lifecycle_error(e.0))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Verify a Python-built ordinary manifest against the generation bindings;
/// returns the FULL-JCS binding receipt (`manifest_id` + bound identities).
#[pyfunction]
#[pyo3(signature = (*, generation_bindings_json, manifest_json))]
fn ordinary_bindings(
    py: Python<'_>,
    generation_bindings_json: Vec<u8>,
    manifest_json: Vec<u8>,
) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| lifecycle::ordinary_bindings(&generation_bindings_json, &manifest_json))
        .map_err(|e| lifecycle_error(e.0))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Cold-init Torch seed bytes (32): deterministic construction RNG derived
/// BEFORE any shell exists. Torch applies the first 8 bytes BE as its u64.
#[pyfunction]
#[pyo3(signature = (*, master_seed_hex, experiment_id, replicate_id, attempt_id, initial_checkpoint_id))]
fn cold_init_seed(
    py: Python<'_>,
    master_seed_hex: String,
    experiment_id: String,
    replicate_id: u64,
    attempt_id: u64,
    initial_checkpoint_id: String,
) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| {
            lifecycle::cold_init_seed(
                &master_seed_hex,
                &experiment_id,
                replicate_id,
                attempt_id,
                &initial_checkpoint_id,
            )
        })
        .map_err(|e| lifecycle_error(e.0))?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Pure scheduling choice: serve the preferred class when eligible, fall
/// back immediately, toggle only on served work. Returns
/// `(kind, next_preferred_inference)`.
#[pyfunction]
#[pyo3(signature = (*, preferred_inference, inference_eligible, learner_eligible))]
fn choose_work(
    py: Python<'_>,
    preferred_inference: bool,
    inference_eligible: bool,
    learner_eligible: bool,
) -> PyResult<(String, bool)> {
    let (kind, next) =
        py.detach(|| lifecycle::choose_work(preferred_inference, inference_eligible, learner_eligible));
    Ok((kind.as_str().to_string(), next))
}

/// Reduce one accumulated 23-f32 PPO buffer at the true count into finite
/// report means + per-seat explained variance (canonical report bytes).
#[pyfunction]
#[pyo3(signature = (*, count, stats))]
fn ppo_report(py: Python<'_>, count: u64, stats: Vec<f32>) -> PyResult<Py<PyBytes>> {
    if stats.len() != lifecycle::PPO_REPORT_LEN {
        return Err(lifecycle_error(format!(
            "rl_lifecycle ppo_report needs {} scalars, got {}",
            lifecycle::PPO_REPORT_LEN,
            stats.len()
        )));
    }
    let out = py
        .detach(|| {
            let report = lifecycle::ppo_report(count, &stats).map_err(|e| e.0)?;
            hydra_feed::canon::canonical_bytes_value(&report, "rl:ppo-report")
                .map_err(|e| format!("rl lifecycle ppo report not canonical-safe: {e}"))
        })
        .map_err(lifecycle_error)?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Frozen recipe composition digests with no inputs (pinning tool for the
/// RL config; `prepare_run` enforces the pinned values exactly).
#[pyfunction]
fn recipe_digests(py: Python<'_>) -> PyResult<Py<PyBytes>> {
    let out = py
        .detach(|| {
            let digests = lifecycle::recipe_digests().map_err(|e| e.0)?;
            hydra_feed::canon::canonical_bytes_value(&digests, "rl:recipe-digests")
                .map_err(|e| format!("rl lifecycle recipe digests not canonical-safe: {e}"))
        })
        .map_err(lifecycle_error)?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// Register the cold lifecycle surface onto the already-created `rl`
/// submodule (Main calls this from `rl/mod.rs` after `pool_ffi::register`).
pub fn register(sub: &Bound<'_, PyModule>) -> PyResult<()> {
    sub.add_function(wrap_pyfunction!(prepare_run, sub)?)?;
    sub.add_function(wrap_pyfunction!(collector_spec, sub)?)?;
    sub.add_function(wrap_pyfunction!(prepare_generation, sub)?)?;
    sub.add_function(wrap_pyfunction!(build_control, sub)?)?;
    sub.add_function(wrap_pyfunction!(ordinary_bindings, sub)?)?;
    sub.add_function(wrap_pyfunction!(cold_init_seed, sub)?)?;
    sub.add_function(wrap_pyfunction!(choose_work, sub)?)?;
    sub.add_function(wrap_pyfunction!(ppo_report, sub)?)?;
    sub.add_function(wrap_pyfunction!(recipe_digests, sub)?)?;
    Ok(())
}
