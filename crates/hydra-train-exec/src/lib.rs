//! Training execution support modules migrated from the train binary.

#![deny(missing_docs)]

/// Runtime advisory formatting and selection helpers.
pub mod advisory;
/// Pure presentation formatting helpers shared by train execution seams.
pub mod presentation;
/// Probe result summary helpers shared by execution support modules.
pub mod probe_summary;
/// Progress DTOs and scalar accumulation helpers shared by train execution.
pub mod progress;
/// Resume state contracts and helpers.
pub mod resume;

/// Artifact path and log-only helpers shared across training execution seams.
pub mod artifacts;
/// BC shard manifest adapters shared by train bootstrap.
pub mod bc_shard_adapter;
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
