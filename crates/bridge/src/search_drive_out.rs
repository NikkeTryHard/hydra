//! Drive-batch outcome converters (split from `search` to hold the gate).
//!
//! Pure `DriveOut` to `Py*Out` translators for DESPOT, PBRF, joint, and
//! persistence; no interpreter interaction, no I/O. Failure mode is none
//! (infallible field copies; validation lives in the drivers).
use pyo3::prelude::*;
/// One drive-batch outcome crossing the boundary (selection + means +
/// digest + counters + end cursor + opaque blob; world blobs never cross back).
#[pyclass(name = "DespotOut", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyDespotOut {
    #[pyo3(get)]
    pub selected_id: u32,
    #[pyo3(get)]
    pub candidate_ids: Vec<u32>,
    #[pyo3(get)]
    pub value_vectors: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub visits: Vec<u64>,
    #[pyo3(get)]
    pub decision_digest: String,
    #[pyo3(get)]
    pub sims_run: u64,
    #[pyo3(get)]
    pub transitions: u64,
    #[pyo3(get)]
    pub model_calls: u64,
    #[pyo3(get)]
    pub completed: bool,
    #[pyo3(get)]
    pub end_cursor: u64,
    #[pyo3(get)]
    pub next_state_blob: Vec<u8>,
}

#[pyclass(name = "PbrfOut", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyPbrfOut {
    #[pyo3(get)]
    pub selected_id: u32,
    #[pyo3(get)]
    pub candidate_ids: Vec<u32>,
    #[pyo3(get)]
    pub value_vectors: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub visits: Vec<u64>,
    #[pyo3(get)]
    pub decision_digest: String,
    #[pyo3(get)]
    pub sims_run: u64,
    #[pyo3(get)]
    pub transitions: u64,
    #[pyo3(get)]
    pub model_calls: u64,
    #[pyo3(get)]
    pub completed: bool,
    #[pyo3(get)]
    pub end_cursor: u64,
    #[pyo3(get)]
    pub next_state_blob: Vec<u8>,
}

#[pyclass(name = "JointOut", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyJointOut {
    #[pyo3(get)]
    pub selected_id: u32,
    #[pyo3(get)]
    pub candidate_ids: Vec<u32>,
    #[pyo3(get)]
    pub value_vectors: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub visits: Vec<u64>,
    #[pyo3(get)]
    pub decision_digest: String,
    #[pyo3(get)]
    pub sims_run: u64,
    #[pyo3(get)]
    pub transitions: u64,
    #[pyo3(get)]
    pub model_calls: u64,
    #[pyo3(get)]
    pub completed: bool,
    #[pyo3(get)]
    pub end_cursor: u64,
    #[pyo3(get)]
    pub next_state_blob: Vec<u8>,
}

#[pyclass(name = "PersistenceOut", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyPersistenceOut {
    #[pyo3(get)]
    pub selected_id: u32,
    #[pyo3(get)]
    pub candidate_ids: Vec<u32>,
    #[pyo3(get)]
    pub value_vectors: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub visits: Vec<u64>,
    #[pyo3(get)]
    pub decision_digest: String,
    #[pyo3(get)]
    pub sims_run: u64,
    #[pyo3(get)]
    pub transitions: u64,
    #[pyo3(get)]
    pub model_calls: u64,
    #[pyo3(get)]
    pub completed: bool,
    #[pyo3(get)]
    pub end_cursor: u64,
    #[pyo3(get)]
    pub next_state_blob: Vec<u8>,
}

pub(crate) fn drive_out_to_despot(out: hydra_search::drive_batch::DriveOut) -> PyDespotOut {
    let mut value_vectors: Vec<Vec<f64>> = Vec::with_capacity(out.value_vectors.len());
    let mut i = 0;
    while i < out.value_vectors.len() {
        let vector = out.value_vectors[i];
        value_vectors.push(vec![vector[0], vector[1], vector[2], vector[3]]);
        i += 1;
    }
    PyDespotOut {
        selected_id: out.selected_id,
        candidate_ids: out.candidate_ids,
        value_vectors,
        visits: out.visits,
        decision_digest: out.decision_digest,
        sims_run: out.sims_run,
        transitions: out.transitions,
        model_calls: out.model_calls,
        completed: out.completed,
        end_cursor: out.end_cursor,
        next_state_blob: out.next_state_blob,
    }
}

pub(crate) fn drive_out_to_pbrf(out: hydra_search::drive_batch::DriveOut) -> PyPbrfOut {
    let mut value_vectors: Vec<Vec<f64>> = Vec::with_capacity(out.value_vectors.len());
    let mut i = 0;
    while i < out.value_vectors.len() {
        let vector = out.value_vectors[i];
        value_vectors.push(vec![vector[0], vector[1], vector[2], vector[3]]);
        i += 1;
    }
    PyPbrfOut {
        selected_id: out.selected_id,
        candidate_ids: out.candidate_ids,
        value_vectors,
        visits: out.visits,
        decision_digest: out.decision_digest,
        sims_run: out.sims_run,
        transitions: out.transitions,
        model_calls: out.model_calls,
        completed: out.completed,
        end_cursor: out.end_cursor,
        next_state_blob: out.next_state_blob,
    }
}

pub(crate) fn drive_out_to_joint(out: hydra_search::drive_batch::DriveOut) -> PyJointOut {
    let mut value_vectors: Vec<Vec<f64>> = Vec::with_capacity(out.value_vectors.len());
    let mut i = 0;
    while i < out.value_vectors.len() {
        let vector = out.value_vectors[i];
        value_vectors.push(vec![vector[0], vector[1], vector[2], vector[3]]);
        i += 1;
    }
    PyJointOut {
        selected_id: out.selected_id,
        candidate_ids: out.candidate_ids,
        value_vectors,
        visits: out.visits,
        decision_digest: out.decision_digest,
        sims_run: out.sims_run,
        transitions: out.transitions,
        model_calls: out.model_calls,
        completed: out.completed,
        end_cursor: out.end_cursor,
        next_state_blob: out.next_state_blob,
    }
}

pub(crate) fn drive_out_to_persistence(
    out: hydra_search::drive_batch::DriveOut,
) -> PyPersistenceOut {
    let mut value_vectors: Vec<Vec<f64>> = Vec::with_capacity(out.value_vectors.len());
    let mut i = 0;
    while i < out.value_vectors.len() {
        let vector = out.value_vectors[i];
        value_vectors.push(vec![vector[0], vector[1], vector[2], vector[3]]);
        i += 1;
    }
    PyPersistenceOut {
        selected_id: out.selected_id,
        candidate_ids: out.candidate_ids,
        value_vectors,
        visits: out.visits,
        decision_digest: out.decision_digest,
        sims_run: out.sims_run,
        transitions: out.transitions,
        model_calls: out.model_calls,
        completed: out.completed,
        end_cursor: out.end_cursor,
        next_state_blob: out.next_state_blob,
    }
}
