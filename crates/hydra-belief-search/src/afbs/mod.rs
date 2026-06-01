//! Anytime Factored-Belief Search (AFBS) with PUCT selection.
//!
//! Includes provenance-aware caching: every [`PonderResult`] carries
//! `source_net_hash`, `source_version`, [`TrustLevel`], and
//! [`CacheNamespace`] so consumers can decide whether a cached result
//! is safe to reuse at runtime vs. learner-only.

mod batch;
mod cache;
mod ponder;
mod tree;

pub use batch::{LeafBatch, MIN_BATCH};
pub use cache::{CacheNamespace, PonderCache, PonderResult, TrustLevel};
pub use ponder::{GameStateSnapshot, PonderManager, PonderTask, compute_ponder_priority};
pub use tree::{
    AfbsNode, AfbsTree, C_PUCT, NodeIdx, TOP_K, has_any_legal_action, legal_action_count,
    predicted_child_hash,
};

#[cfg(test)]
mod tests;
