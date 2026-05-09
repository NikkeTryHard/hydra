//! Pure replay sidecar JSONL contracts.
//!
//! This crate owns replay-indexed, model-independent sidecar keys,
//! records, indexes, hashes, digests, and JSONL readers.

pub mod delta_q;
pub mod exit;
pub mod jsonl;
pub mod key;
pub mod label;
pub mod provenance;

pub use delta_q::{
    DeltaQSidecarIndex, ReplayDeltaQLookupKey, ReplayDeltaQRecordV1, validate_delta_q_contract,
};
pub use exit::{ExitSidecarIndex, ReplayExitLookupKey, ReplayExitRecordV1};
pub use jsonl::read_jsonl_records;
pub use key::ReplayDecisionKey;
pub use label::{copy_label_arrays, legal_mask_digest_from_bool, legal_mask_digest_from_f32};
pub use provenance::{
    REPLAY_DELTA_Q_PROVENANCE, REPLAY_DELTA_Q_SEMANTICS_V1, REPLAY_EXIT_PROVENANCE,
    REPLAY_EXIT_SEMANTICS_V1, source_hash_from_identity, source_net_hash_from_checkpoint_identity,
};
