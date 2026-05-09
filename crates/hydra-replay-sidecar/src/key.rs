//! Replay decision identity shared by sidecar contracts.

use serde::{Deserialize, Serialize};

/// Replay-indexed state identity for generated sidecar labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayDecisionKey {
    /// Stable hash of replay/source identity.
    pub source_hash: u64,
    /// Zero-based MJAI event index where the decision was observed.
    pub event_index: u32,
    /// Acting player at the decision.
    pub actor: u8,
    /// Observation hash for collision-resistant stale-sidecar rejection.
    pub obs_hash: u64,
}
