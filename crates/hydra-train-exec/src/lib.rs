//! Training execution support modules migrated from the train binary.

#![deny(missing_docs)]

/// Runtime advisory formatting and selection helpers.
pub mod advisory;
#[allow(
    missing_docs,
    reason = "migrated fixed-shape BC executor API preserves runtime compatibility"
)]
/// Fixed-shape behavioral-cloning train/probe execution helpers.
pub mod bc_fixed_shape;
/// Training bootstrap initialization and prepared runtime state.
pub mod bootstrap;
/// CUDA graph and pinned-memory FFI wrappers for exec-owned GPU adapters.
#[cfg(feature = "cuda-graph")]
pub mod cuda_graph;
/// Exec-owned streaming MJAI data pipeline for preflight and epoch runners.
pub mod data_pipeline;
/// Delta-Q promotion mode execution and paired arena helpers.
pub mod delta_q_promotion;
/// Epoch-runner execution helpers shared by train execution.
pub mod epoch_runner;
/// CUDA graph probe parent/child execution.
pub mod graph_probe;
/// Train binary mode dispatch facade.
pub mod modes;
/// CUDA pinned host staging and reusable device materialization for BC shards.
#[cfg(feature = "cuda-graph")]
pub mod pinned_transfer;
#[allow(
    missing_docs,
    reason = "migrated train preflight API is still a compatibility seam"
)]
/// Heavy preflight and probe execution runner.
pub mod preflight_runtime;
/// Pure presentation formatting helpers shared by train execution seams.
pub mod presentation;
/// Probe candidate ladder helpers shared by preflight search.
pub mod probe_ladder;
/// Probe child process transport helpers.
pub mod probe_process;
/// Probe search orchestration helpers.
pub mod probe_search;
/// Probe result summary helpers shared by execution support modules.
pub mod probe_summary;
/// Probe artifact transport helpers.
pub mod probe_transport;
/// Progress DTOs and scalar accumulation helpers shared by train execution.
pub mod progress;
/// Resume state contracts and helpers.
pub mod resume;
/// RL training-loop execution helpers.
pub mod rl_runner;
/// RL train-step wrapper below the train facade.
pub mod rl_step;
#[allow(
    missing_docs,
    reason = "migrated runtime autotune API is still a compatibility seam"
)]
/// Loader runtime autotune support used by migrated preflight execution.
pub mod runtime_autotune_shim;
/// Heavy validation execution runner.
pub mod validation_runner;

/// Artifact path and log-only helpers shared across training execution seams.
pub mod artifacts;
/// BC shard manifest adapters shared by train bootstrap.
pub mod bc_shard_adapter;
#[cfg(test)]
mod test_loose_replay_fixtures;
#[cfg(test)]
mod test_support;
/// Validation snapshot and gate DTOs shared across training execution seams.
pub mod validation;

/// Marker for the train execution boundary.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct TrainExec;

impl TrainExec {
    /// Creates a train execution boundary marker.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}
