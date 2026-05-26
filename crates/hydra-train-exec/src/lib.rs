//! Training execution support modules migrated from the train binary.

#![deny(missing_docs)]

/// Runtime advisory formatting and selection helpers.
pub mod advisory;
#[allow(
    missing_docs,
    reason = "migrated fixed-shape BC executor API preserves runtime compatibility"
)]
/// Fixed-shape behavioral-cloning train/probe execution helpers.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod bc_fixed_shape;
/// Behavioral-cloning metric readback helpers.
pub mod bc_metrics;
/// Behavioral-cloning execution adapters.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod bc_runtime;
/// BC shard build execution helpers.
pub mod bc_shard_builder;
#[cfg(all(feature = "burn-libtorch-obsolete", feature = "bf16-autocast-proof"))]
mod bf16_autocast_proof;
/// Training bootstrap initialization and prepared runtime state.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod bootstrap;
/// Shared campaign/run artifact layout helpers.
mod campaign_layout;
/// LibTorch/Rayon runtime helpers for execution-owned config materialization.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod config_runtime;
/// CUDA graph and pinned-memory FFI wrappers for exec-owned GPU adapters.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod cuda_graph;
#[allow(
    missing_docs,
    reason = "migrated MJAI data adapter API preserves train facade compatibility"
)]
/// Burn-facing MJAI sample collation and validation data adapters.
pub mod data;
/// Exec-owned streaming MJAI data pipeline for preflight and epoch runners.
pub mod data_pipeline;
/// Delta-Q promotion mode execution and paired arena helpers.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod delta_q_promotion;
/// Epoch-runner execution helpers shared by train execution.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod epoch_runner;
#[cfg(test)]
#[cfg(feature = "burn-libtorch-obsolete")]
mod epoch_runner_tests;
/// Global Python-launcher GPU performance environment flags.
#[cfg(feature = "training")]
pub mod gpu_config;
/// CUDA graph probe parent/child execution.
pub mod graph_probe;
/// Training execution loss adapter.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod losses;
/// Training execution model adapters.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod model;
/// Train binary mode dispatch facade.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod modes;
/// NVTX profiling scope adapter.
pub mod nvtx;
/// Supervised/RL phase orchestration helpers below train compatibility facades.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod orchestrator;
/// CUDA pinned host staging and reusable device materialization for BC shards.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod pinned_transfer;
#[allow(
    missing_docs,
    reason = "migrated train preflight API is still a compatibility seam"
)]
/// Heavy preflight and probe execution runner.
#[cfg(feature = "burn-libtorch-obsolete")]
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
/// Experimental Python learner process launcher.
pub mod python_learner;
/// Direct raw-MJAI host-batch stream for the Python learner.
pub mod raw_mjai_stream;
/// Resume state contracts and helpers.
pub mod resume;
/// RL training-loop execution helpers.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod rl_runner;
/// RL train-step wrapper below the train facade.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod rl_step;
#[allow(
    missing_docs,
    reason = "migrated runtime autotune API is still a compatibility seam"
)]
/// Loader runtime autotune support used by migrated preflight execution.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod runtime_autotune_shim;
/// Lightweight system metrics telemetry helpers.
pub mod system_metrics;
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod torch_profiler;
/// Heavy validation execution runner.
#[cfg(feature = "burn-libtorch-obsolete")]
pub mod validation_runner;

/// Artifact path and log-only helpers shared across training execution seams.
pub mod artifacts;
/// BC shard manifest adapters shared by train bootstrap.
pub mod bc_shard_adapter;
#[cfg(test)]
#[cfg(feature = "burn-libtorch-obsolete")]
mod test_loose_replay_fixtures;
#[cfg(test)]
#[cfg(feature = "burn-libtorch-obsolete")]
mod test_support;
#[cfg(test)]
#[cfg(feature = "burn-libtorch-obsolete")]
mod tests;
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
