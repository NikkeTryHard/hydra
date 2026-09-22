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
//! The extension module installs as `hydra2._native` (maturin `module-name`;
//! package `hydra-bridge`, `[lib] name = "_native"`) with the plane-filling
//! `PyHydraStream` surface plus `canon_rng`/`columnar`/`search` submodules
//! (single cdylib, one `#[pymodule]`).

pub mod action_artifact;
pub mod action_model;
pub mod analysis_leaves2;
pub mod belief_kernel;
pub mod belief_leaves;
pub mod belief_natural;
pub mod candidate0_act;
pub mod canon_rng;
pub mod columnar;
pub mod common_errors;
pub mod common_roots;
pub mod contracts;
pub mod dataset_parse;
pub mod despot_result;
pub mod despot_seeds;
pub mod distill_leaves;
pub mod encoder;
pub mod eval;
pub mod eval_blocks;
pub mod eval_duplicate;
pub mod eval_leaves;
pub mod eval_leaves2;
pub mod eval_schedule;
pub mod eval_stats;
pub mod event_packet;
pub mod event_schema;
pub mod gumbel_core;
pub mod gumbel_spec;
pub mod ingest_leaves;
pub mod ismcts_core;
pub mod joint_types;
pub mod joint_uncertainty;
pub mod joint_world;
pub mod local_graph;
pub mod local_spec;
pub mod loop_state;
pub mod mirror;
pub mod obs_actor;
pub mod obs_assembly;
pub mod observation_schema;
pub mod oracle_guard;
pub mod oracle_join;
pub mod oracle_targets;
pub mod packet;
pub mod packet_decode;
pub mod partition_dups;
pub mod pbrf_forest;
pub mod pbrf_spec;
pub mod persistence_kernel;
pub mod persistence_report;
pub mod persistence_spec;
pub mod promotion;
pub mod qual_budget;
pub mod qual_replay;
pub mod quarantine_inbox;
pub mod randomness;
pub mod rc_cross_validate;
pub mod rc_digest;
pub mod rc_parse;
pub mod rc_parse_aux;
pub mod rc_parse_train;
pub mod rc_require;
pub mod rc_resume;
pub mod rc_resume_gates;
pub mod rc_sections;
pub mod record_leaves;
pub mod replay;
pub mod resume;
pub mod ring;
pub mod rows_seal;
pub mod sampled_kernel;
pub mod script_leaves;
pub mod search;
pub mod search_drive_out;
pub mod search_profiles;
pub mod search_shared;
pub mod sink;
pub mod stream;
pub mod stream_decode;
pub mod stream_driver;
pub mod stream_manifest;
pub mod stream_report;
pub mod stream_scan;
pub mod tiles;
pub mod tracking_leaves;
pub mod training_leaves;
pub mod utility;
pub mod validate;
pub mod walls;
use pyo3::prelude::*;

/// Extension module: `import hydra2._native`.
#[pymodule]
pub fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<stream::PyHydraStream>()?;
    m.add_class::<stream::PyFill>()?;
    m.add_class::<stream::PyStats>()?;
    m.add_class::<stream::PyQuar>()?;
    stream_driver::register(m)?;
    replay::register(m)?;
    canon_rng::register(m)?;
    resume::register(m)?;
    ring::register(m)?;
    encoder::register(m)?;
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
