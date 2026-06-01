use serde::{Deserialize, Serialize};

use crate::key::ReplayDecisionKey;

/// Lookup key for replay ExIt labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayExitLookupKey {
    /// Replay decision identity.
    pub replay: ReplayDecisionKey,
    /// Chosen action id at the replay decision.
    pub action: u8,
}

/// Version-1 replay ExIt JSONL sidecar record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayExitRecordV1 {
    /// Schema version. Must be 1.
    pub version: u32,
    /// Semantics tag. Must match [`crate::provenance::REPLAY_EXIT_SEMANTICS_V1`].
    pub semantics: String,
    /// Provenance tag. Must match [`crate::provenance::REPLAY_EXIT_PROVENANCE`].
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
    /// Root search visit budget used for the record.
    pub root_visit_count: u32,
    /// Number of legal discard actions at the replay decision.
    pub legal_discard_count: u8,
    /// Number of target-supported actions in the record.
    pub supported_actions: u8,
    /// Supported-action coverage over legal discards.
    pub coverage: f32,
    /// KL divergence from the base policy.
    pub kl_to_base: f32,
    /// Action-space target probabilities.
    pub target: Vec<f32>,
    /// Action-space support mask.
    pub mask: Vec<f32>,
}
