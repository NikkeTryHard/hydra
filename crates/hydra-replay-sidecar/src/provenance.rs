//! Sidecar provenance, semantics tags, and stable source hashes.

/// Semantics tag for replay ExIt sidecar records.
pub const REPLAY_EXIT_SEMANTICS_V1: &str = "exit_root_child_visits_v1";

/// Provenance tag for replay ExIt sidecar records.
pub const REPLAY_EXIT_PROVENANCE: &str = "search-derived";

/// Semantics tag for replay delta-q sidecar records.
pub const REPLAY_DELTA_Q_SEMANTICS_V1: &str = "delta_q_child_minus_root_v1";

/// Provenance tag for replay delta-q sidecar records.
pub const REPLAY_DELTA_Q_PROVENANCE: &str = "search-derived";

/// Returns a stable FNV-1a hash for replay source identity strings.
pub fn source_hash_from_identity(identity: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in identity.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Returns the stable source-network hash for checkpoint identity strings.
pub fn source_net_hash_from_checkpoint_identity(identity: &str) -> u64 {
    source_hash_from_identity(identity)
}
