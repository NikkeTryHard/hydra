use serde::{Deserialize, Serialize};

use crate::key::ReplayDecisionKey;

/// Lookup key for replay delta-q labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayDeltaQLookupKey {
    /// Replay decision identity.
    pub replay: ReplayDecisionKey,
    /// Chosen action id at the replay decision.
    pub action: u8,
}

/// Version-1 replay delta-q JSONL sidecar record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayDeltaQRecordV1 {
    /// Schema version. Must be 1.
    pub version: u32,
    /// Semantics tag. Must match [`crate::provenance::REPLAY_DELTA_Q_SEMANTICS_V1`].
    pub semantics: String,
    /// Provenance tag. Must match [`crate::provenance::REPLAY_DELTA_Q_PROVENANCE`].
    pub provenance: String,
    /// Replay decision identity.
    pub key: ReplayDecisionKey,
    /// Chosen action id at the replay decision.
    pub action: u8,
    /// Digest of the legal action mask used to generate this label.
    pub legal_mask_digest: u64,
    /// Hash of the network/checkpoint identity used to generate this label.
    pub source_net_hash: u64,
    /// Version of the network/checkpoint identity contract.
    pub source_version: u32,
    /// Action-space delta-q targets.
    pub target: Vec<f32>,
    /// Action-space support mask.
    pub mask: Vec<f32>,
}
