//! Pure replay sidecar JSONL contracts.
//!
//! This crate owns replay-indexed, model-independent sidecar keys,
//! records, indexes, hashes, digests, and JSONL readers.

pub mod delta_q;
pub mod error;
pub mod exit;
pub mod jsonl;
pub mod key;
pub mod label;
pub mod provenance;

/// Fixed 46-action target and support-mask label pair.
pub type ActionLabelPair = (
    [f32; hydra_core::action::HYDRA_ACTION_SPACE],
    [f32; hydra_core::action::HYDRA_ACTION_SPACE],
);

pub use delta_q::{
    DeltaQSidecarIndex, ReplayDeltaQLookupKey, ReplayDeltaQRecordV1, validate_delta_q_contract,
};
pub use error::{SidecarContractError, SidecarKind};
pub use exit::{ExitSidecarIndex, ReplayExitLookupKey, ReplayExitRecordV1};
pub use jsonl::read_jsonl_records;
pub use key::ReplayDecisionKey;
pub use label::{copy_label_arrays, legal_mask_digest_from_bool, legal_mask_digest_from_f32};
pub use provenance::{
    REPLAY_DELTA_Q_PROVENANCE, REPLAY_DELTA_Q_SEMANTICS_V1, REPLAY_EXIT_PROVENANCE,
    REPLAY_EXIT_SEMANTICS_V1, source_hash_from_identity, source_net_hash_from_checkpoint_identity,
};
