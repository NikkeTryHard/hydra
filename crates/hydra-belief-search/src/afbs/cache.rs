use dashmap::DashMap;
use hydra_runtime_types::action::HYDRA_ACTION_SPACE;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use super::tree::{AfbsTree, NodeIdx, predicted_child_hash};

/// Trust level assigned to a cached ponder result.
///
/// The archive (answer_20, answer_16-1) prescribes a strict trust hierarchy:
/// all current ponder outputs default to `LearnerOnly` until provenance and
/// admission gates are satisfied.  Runtime action selection should only use
/// results with `Authoritative` trust.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrustLevel {
    /// Result may be consumed by the learner/training pipeline only.
    /// It must NOT influence live action selection.
    LearnerOnly,
    /// Result may be shown to a human or logged, but must NOT influence
    /// action selection or be treated as ground truth.
    Advisory,
    /// Result may warm-start a new search tree (same-episode, observed-root
    /// only) but must NOT be used as a final action authority.
    WarmStart,
    /// Result has passed all admission gates and may be used for live
    /// action selection.  Nothing currently qualifies.
    Authoritative,
}

impl TrustLevel {
    /// Returns `true` if `self` is at least as trusted as `min`.
    pub fn meets(&self, min: TrustLevel) -> bool {
        (*self as u8) >= (min as u8)
    }
}

/// Namespace partitioning for cache entries.
///
/// Keeps observed roots, speculative child hints, and learner-only targets
/// in logically separate buckets even when stored in the same physical map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheNamespace {
    /// Entry was produced from a real observed game state.
    ObservedRoot,
    /// Entry was produced speculatively via `predicted_child_hash`.
    SpeculativeChildHint,
    /// Entry exists only for learner/training label production.
    LearnerTarget,
}

#[derive(Debug, Clone, Copy)]
pub struct PonderResult {
    pub exit_policy: [f32; HYDRA_ACTION_SPACE],
    pub value: f32,
    pub search_depth: u8,
    pub visit_count: u32,
    pub timestamp: Instant,
    pub source_net_hash: u64,
    pub source_version: u32,
    pub trust_level: TrustLevel,
    pub cache_namespace: CacheNamespace,
    pub generation: u64,
}

impl PonderResult {
    /// Builds a result from a completed AFBS search tree.
    ///
    /// The caller must supply provenance (`source_net_hash`, `source_version`)
    /// identifying which model produced the search.  Trust level defaults to
    /// `LearnerOnly` and namespace to `ObservedRoot`; callers may override
    /// after construction.  Generation is set to 0 and stamped by the cache
    /// on insertion.
    pub fn from_tree(
        tree: &AfbsTree,
        root_idx: NodeIdx,
        value: f32,
        tau: f32,
        source_net_hash: u64,
        source_version: u32,
    ) -> Self {
        Self {
            exit_policy: tree.root_exit_policy(root_idx, tau),
            value,
            search_depth: tree.max_depth(root_idx),
            visit_count: tree.root_visit_count(root_idx),
            timestamp: Instant::now(),
            source_net_hash,
            source_version,
            trust_level: TrustLevel::LearnerOnly,
            cache_namespace: CacheNamespace::ObservedRoot,
            generation: 0,
        }
    }

    /// Creates a learner-only result with zero provenance.
    ///
    /// Use for test fixtures or when the producing net is not yet tracked.
    pub fn learner_only_stub(
        exit_policy: [f32; HYDRA_ACTION_SPACE],
        value: f32,
        search_depth: u8,
        visit_count: u32,
    ) -> Self {
        Self {
            exit_policy,
            value,
            search_depth,
            visit_count,
            timestamp: Instant::now(),
            source_net_hash: 0,
            source_version: 0,
            trust_level: TrustLevel::LearnerOnly,
            cache_namespace: CacheNamespace::LearnerTarget,
            generation: 0,
        }
    }
}

/// Generation-aware ponder cache with trust-level gating.
///
/// Each entry carries a generation stamp set on insertion.  When the cache
/// generation is bumped (e.g. on checkpoint change), older entries are
/// rejected on lookup.  Runtime consumers can further filter by
/// [`TrustLevel`].
pub struct PonderCache {
    entries: DashMap<u64, PonderResult>,
    generation: AtomicU64,
}

impl PonderCache {
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
            generation: AtomicU64::new(1),
        }
    }

    pub fn current_generation(&self) -> u64 {
        self.generation.load(AtomicOrdering::Relaxed)
    }

    /// Inserts an entry, stamping the current cache generation.
    pub fn insert(&self, hash: u64, mut result: PonderResult) {
        result.generation = self.current_generation();
        self.entries.insert(hash, result);
    }

    /// Looks up an entry, rejecting stale generations.
    pub fn get(&self, hash: u64) -> Option<PonderResult> {
        let current_gen = self.current_generation();
        self.entries
            .get(&hash)
            .map(|entry| *entry.value())
            .filter(|r| r.generation >= current_gen)
    }

    /// Looks up an entry, rejecting stale generations and entries below `min_trust`.
    pub fn get_trusted(&self, hash: u64, min_trust: TrustLevel) -> Option<PonderResult> {
        self.get(hash).filter(|r| r.trust_level.meets(min_trust))
    }

    pub fn predicted_child_key(parent_hash: u64, action: u8) -> u64 {
        predicted_child_hash(parent_hash, action)
    }

    pub fn get_predicted_child(&self, parent_hash: u64, action: u8) -> Option<PonderResult> {
        self.get(Self::predicted_child_key(parent_hash, action))
    }

    pub fn insert_predicted_child(&self, parent_hash: u64, action: u8, mut result: PonderResult) {
        result.cache_namespace = CacheNamespace::SpeculativeChildHint;
        self.insert(Self::predicted_child_key(parent_hash, action), result);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn remove(&self, hash: u64) -> Option<PonderResult> {
        self.entries.remove(&hash).map(|(_, value)| value)
    }

    pub fn summary(&self) -> String {
        format!(
            "cache(entries={}, gen={})",
            self.entries.len(),
            self.current_generation()
        )
    }

    pub fn contains(&self, hash: u64) -> bool {
        self.entries.contains_key(&hash)
    }

    pub fn clear(&self) {
        self.entries.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bumps the generation counter, logically invalidating all existing entries.
    ///
    /// Entries remain in physical storage but will be rejected by `get()`
    /// and `get_trusted()` until re-inserted at the new generation.
    pub fn invalidate(&self) -> u64 {
        self.generation.fetch_add(1, AtomicOrdering::Relaxed) + 1
    }

    /// Removes all entries and bumps the generation.
    pub fn flush(&self) {
        self.invalidate();
        self.entries.clear();
    }
}

impl Default for PonderCache {
    fn default() -> Self {
        Self::new()
    }
}
