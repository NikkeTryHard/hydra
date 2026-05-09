//! Backward-compatible replay delta-q exports.
//!
//! The canonical implementation lives in `hydra_search_labels::replay_delta_q`.

pub use hydra_replay_sidecar::{
    ReplayDecisionKey, copy_label_arrays, legal_mask_digest_from_bool, legal_mask_digest_from_f32,
    read_jsonl_records, source_hash_from_identity, source_net_hash_from_checkpoint_identity,
};
pub use hydra_search_labels::replay_delta_q::*;
