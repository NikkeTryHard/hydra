//! Anytime Factored-Belief Search (AFBS) with PUCT selection.
//!
//! Includes provenance-aware caching: every [`PonderResult`] carries
//! `source_net_hash`, `source_version`, [`TrustLevel`], and
//! [`CacheNamespace`] so consumers can decide whether a cached result
//! is safe to reuse at runtime vs. learner-only.

use dashmap::DashMap;
use hydra_runtime_types::action::HYDRA_ACTION_SPACE;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

pub const C_PUCT: f32 = 2.5;
pub const TOP_K: usize = 5;
const OBS_SIZE: usize = 192 * 34;

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

pub fn has_any_legal_action(mask: &[bool; HYDRA_ACTION_SPACE]) -> bool {
    mask.iter().any(|&m| m)
}

pub fn legal_action_count(mask: &[bool; HYDRA_ACTION_SPACE]) -> usize {
    mask.iter().filter(|&&m| m).count()
}

fn masked_action_priors(
    policy_logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> Vec<(u8, f32)> {
    let legal_actions: Vec<(u8, f32)> = (0..HYDRA_ACTION_SPACE as u8)
        .filter(|&a| legal_mask[a as usize])
        .map(|a| (a, policy_logits[a as usize]))
        .collect();
    if legal_actions.is_empty() {
        return Vec::new();
    }

    let max_logit = legal_actions
        .iter()
        .map(|(_, logit)| *logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = legal_actions
        .iter()
        .map(|(_, logit)| (*logit - max_logit).exp())
        .sum();
    if exp_sum <= 0.0 || !exp_sum.is_finite() {
        let uniform = 1.0 / legal_actions.len() as f32;
        return legal_actions
            .into_iter()
            .map(|(action, _)| (action, uniform))
            .collect();
    }

    legal_actions
        .into_iter()
        .map(|(action, logit)| (action, (logit - max_logit).exp() / exp_sum))
        .collect()
}

pub type NodeIdx = u32;
type ChildList = SmallVec<[(u8, NodeIdx); TOP_K]>;

pub struct AfbsNode {
    pub info_state_hash: u64,
    pub visit_count: u32,
    pub total_value: f64,
    pub prior: f32,
    pub children: ChildList,
    pub is_opponent: bool,
    pub particle_handle: Option<u32>,
}

impl AfbsNode {
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn is_expanded(&self) -> bool {
        !self.children.is_empty()
    }

    pub fn ucb_score(&self, parent_visits: u32, c_puct: f32) -> f32 {
        let q = self.q_value();
        let u =
            c_puct * self.prior * (parent_visits as f32).sqrt() / (1.0 + self.visit_count as f32);
        q + u
    }

    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            return 0.0;
        }
        (self.total_value / self.visit_count as f64) as f32
    }
}

pub struct AfbsTree {
    pub nodes: Vec<AfbsNode>,
}

pub fn predicted_child_hash(parent_hash: u64, action: u8) -> u64 {
    parent_hash ^ (action as u64).wrapping_mul(0x9e3779b97f4a7c15)
}

impl AfbsTree {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, hash: u64, prior: f32, is_opponent: bool) -> NodeIdx {
        let idx = self.nodes.len() as NodeIdx;
        self.nodes.push(AfbsNode {
            info_state_hash: hash,
            visit_count: 0,
            total_value: 0.0,
            prior,
            children: SmallVec::new(),
            is_opponent,
            particle_handle: None,
        });
        idx
    }

    pub fn puct_select(&self, parent_idx: NodeIdx) -> Option<(u8, NodeIdx)> {
        let parent = &self.nodes[parent_idx as usize];
        if parent.children.is_empty() {
            return None;
        }
        let sqrt_n = (parent.visit_count as f32).sqrt();
        let mut best_ucb = f32::NEG_INFINITY;
        let mut best = None;
        for &(action, child_idx) in &parent.children {
            let child = &self.nodes[child_idx as usize];
            let q = child.q_value();
            let u = C_PUCT * child.prior * sqrt_n / (1.0 + child.visit_count as f32);
            let ucb = q + u;
            if ucb > best_ucb {
                best_ucb = ucb;
                best = Some((action, child_idx));
            }
        }
        best
    }

    pub fn expand_node(
        &mut self,
        parent_idx: NodeIdx,
        policy_logits: &[f32; HYDRA_ACTION_SPACE],
        legal_mask: &[bool; HYDRA_ACTION_SPACE],
        is_opponent: bool,
    ) {
        let Some(parent) = self.nodes.get(parent_idx as usize) else {
            return;
        };
        if parent.is_expanded() || !has_any_legal_action(legal_mask) {
            return;
        }

        let parent_hash = parent.info_state_hash;
        let mut priors = masked_action_priors(policy_logits, legal_mask);
        priors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        priors.truncate(TOP_K);

        let selected_mass: f32 = priors.iter().map(|(_, prior)| *prior).sum();
        let norm = if selected_mass > 0.0 {
            selected_mass
        } else {
            1.0
        };
        let mut children = ChildList::new();
        for (action, prior) in priors {
            let child_hash = predicted_child_hash(parent_hash, action);
            let child_idx = self.add_node(child_hash, prior / norm, is_opponent);
            children.push((action, child_idx));
        }
        self.nodes[parent_idx as usize].children = children;
    }

    fn selection_path(&self, root_idx: NodeIdx) -> Vec<NodeIdx> {
        let mut path = Vec::new();
        let mut current = root_idx;
        while let Some(node) = self.nodes.get(current as usize) {
            path.push(current);
            if node.children.is_empty() || path.len() > self.nodes.len() {
                break;
            }
            let Some((_, child_idx)) = self.puct_select(current) else {
                break;
            };
            current = child_idx;
        }
        path
    }

    pub fn backpropagate(&mut self, path: &[NodeIdx], value: f32) {
        for &idx in path {
            let node = &mut self.nodes[idx as usize];
            node.visit_count += 1;
            node.total_value += value as f64;
        }
    }

    pub fn run_search_iterations(
        &mut self,
        root_idx: NodeIdx,
        num_iters: u32,
        eval_fn: &dyn Fn(NodeIdx) -> f32,
    ) {
        if self.nodes.get(root_idx as usize).is_none() {
            return;
        }
        for _ in 0..num_iters {
            let path = self.selection_path(root_idx);
            let Some(&leaf_idx) = path.last() else {
                continue;
            };
            let value = eval_fn(leaf_idx);
            self.backpropagate(&path, value);
        }
    }

    pub fn root_exit_policy(&self, root_idx: NodeIdx, tau: f32) -> [f32; HYDRA_ACTION_SPACE] {
        let mut policy = [0.0f32; HYDRA_ACTION_SPACE];
        let Some(root) = self.nodes.get(root_idx as usize) else {
            return policy;
        };
        if root.children.is_empty() {
            return policy;
        }

        if !tau.is_finite() || tau <= 0.0 {
            if let Some((action, _)) = root.children.iter().max_by(|(_, lhs), (_, rhs)| {
                self.nodes[*lhs as usize]
                    .q_value()
                    .partial_cmp(&self.nodes[*rhs as usize].q_value())
                    .unwrap_or(Ordering::Equal)
            }) {
                policy[*action as usize] = 1.0;
            }
            return policy;
        }

        let mut max_q = f32::NEG_INFINITY;
        for &(_, child_idx) in &root.children {
            let q = self.nodes[child_idx as usize].q_value();
            if q > max_q {
                max_q = q;
            }
        }
        let mut total = 0.0f32;
        for &(action, child_idx) in &root.children {
            let q = self.nodes[child_idx as usize].q_value();
            let exp_q = ((q - max_q) / tau).exp();
            policy[action as usize] = exp_q;
            total += exp_q;
        }
        if total > 0.0 {
            for p in &mut policy {
                *p /= total;
            }
        }
        policy
    }

    pub fn best_action(&self, root_idx: NodeIdx) -> Option<u8> {
        let root = &self.nodes[root_idx as usize];
        root.children
            .iter()
            .max_by_key(|(_, idx)| self.nodes[*idx as usize].visit_count)
            .map(|(action, _)| *action)
    }

    pub fn find_child_by_action(&self, parent_idx: NodeIdx, action: u8) -> Option<NodeIdx> {
        self.nodes
            .get(parent_idx as usize)
            .and_then(|node| node.children.iter().find(|(a, _)| *a == action))
            .map(|(_, idx)| *idx)
    }

    pub fn shift_root_to_child(&self, root_idx: NodeIdx, observed_action: u8) -> Option<NodeIdx> {
        self.find_child_by_action(root_idx, observed_action)
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    pub fn expanded_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_expanded()).count()
    }
    pub fn total_visits(&self) -> u64 {
        self.nodes.iter().map(|n| n.visit_count as u64).sum()
    }
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    pub fn child_actions(&self, node: NodeIdx) -> Vec<u8> {
        self.nodes
            .get(node as usize)
            .map(|n| n.children.iter().map(|(a, _)| *a).collect())
            .unwrap_or_default()
    }

    pub fn node_q_value(&self, node: NodeIdx) -> f32 {
        self.nodes.get(node as usize).map_or(0.0, |n| n.q_value())
    }

    pub fn num_children(&self, node: NodeIdx) -> usize {
        self.nodes
            .get(node as usize)
            .map_or(0, |n| n.children.len())
    }

    pub fn summary(&self, root: NodeIdx) -> String {
        format!(
            "afbs(nodes={}, visits={}, depth={})",
            self.tree_size(),
            self.root_visit_count(root),
            self.max_depth(root)
        )
    }

    pub fn root_visit_count(&self, root: NodeIdx) -> u32 {
        self.nodes.get(root as usize).map_or(0, |n| n.visit_count)
    }

    pub fn tree_size(&self) -> usize {
        self.nodes.len()
    }

    pub fn max_depth(&self, root: NodeIdx) -> u8 {
        let node = &self.nodes[root as usize];
        if node.children.is_empty() {
            return 0;
        }
        let mut max_d = 0u8;
        for &(_, child) in &node.children {
            let d = self.max_depth(child);
            if d > max_d {
                max_d = d;
            }
        }
        max_d + 1
    }
}

impl Default for AfbsTree {
    fn default() -> Self {
        Self::new()
    }
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

pub struct PonderTask {
    pub info_state_hash: u64,
    pub priority_score: f32,
    pub game_state_snapshot: GameStateSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GameStateSnapshot {
    pub info_state_hash: u64,
    pub top2_policy_gap: f32,
    pub risk_score: f32,
    pub particle_ess: f32,
}

impl Eq for PonderTask {}

impl PartialEq for PonderTask {
    fn eq(&self, other: &Self) -> bool {
        self.info_state_hash == other.info_state_hash
            && self.priority_score.to_bits() == other.priority_score.to_bits()
    }
}

impl Ord for PonderTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_score
            .partial_cmp(&other.priority_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.info_state_hash.cmp(&other.info_state_hash))
    }
}

impl PartialOrd for PonderTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn compute_ponder_priority(top2_gap: f32, risk_score: f32, particle_ess: f32) -> f32 {
    let gap_term = (0.1 - top2_gap).max(0.0) * 10.0;
    let risk_term = risk_score.max(0.0);
    let ess_term = (1.0 - particle_ess).max(0.0);
    gap_term + risk_term + ess_term
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

pub struct PonderManager {
    pub cache: PonderCache,
    pub priority_queue: std::collections::BinaryHeap<PonderTask>,
    pub worker_handle: Option<std::thread::JoinHandle<()>>,
}

impl PonderManager {
    pub fn new() -> Self {
        Self {
            cache: PonderCache::new(),
            priority_queue: std::collections::BinaryHeap::new(),
            worker_handle: None,
        }
    }

    pub fn enqueue(&mut self, task: PonderTask) {
        self.priority_queue.push(task);
    }

    pub fn enqueue_snapshot(&mut self, snapshot: GameStateSnapshot) {
        let priority_score = compute_ponder_priority(
            snapshot.top2_policy_gap,
            snapshot.risk_score,
            snapshot.particle_ess,
        );
        self.enqueue(PonderTask {
            info_state_hash: snapshot.info_state_hash,
            priority_score,
            game_state_snapshot: snapshot,
        });
    }

    pub fn pop_task(&mut self) -> Option<PonderTask> {
        self.priority_queue.pop()
    }

    pub fn cache_result(&self, hash: u64, result: PonderResult) {
        self.cache.insert(hash, result);
    }

    pub fn lookup(&self, hash: u64) -> Option<PonderResult> {
        self.cache.get(hash)
    }

    pub fn lookup_trusted(&self, hash: u64, min_trust: TrustLevel) -> Option<PonderResult> {
        self.cache.get_trusted(hash, min_trust)
    }

    /// Invalidates all cached entries (e.g. on checkpoint change).
    pub fn invalidate_cache(&self) -> u64 {
        self.cache.invalidate()
    }

    pub fn queue_len(&self) -> usize {
        self.priority_queue.len()
    }

    pub fn has_worker(&self) -> bool {
        self.worker_handle.is_some()
    }
}

impl Default for PonderManager {
    fn default() -> Self {
        Self::new()
    }
}

pub const MIN_BATCH: usize = 32;

pub struct LeafBatch {
    pub obs_buffer: Vec<f32>,
    pub node_indices: Vec<NodeIdx>,
    pub batch_size: usize,
}

impl LeafBatch {
    pub fn new() -> Self {
        Self::with_capacity(MIN_BATCH)
    }

    pub fn with_capacity(batch_capacity: usize) -> Self {
        Self {
            obs_buffer: Vec::with_capacity(batch_capacity * OBS_SIZE),
            node_indices: Vec::with_capacity(batch_capacity),
            batch_size: 0,
        }
    }

    pub fn clear(&mut self) {
        self.obs_buffer.clear();
        self.node_indices.clear();
        self.batch_size = 0;
    }

    pub fn add(&mut self, obs: &[f32], node_idx: NodeIdx) {
        assert_eq!(
            obs.len(),
            OBS_SIZE,
            "leaf observation must have OBS_SIZE elements"
        );
        self.obs_buffer.extend_from_slice(obs);
        self.node_indices.push(node_idx);
        self.batch_size += 1;
    }

    pub fn is_ready(&self) -> bool {
        self.batch_size >= MIN_BATCH
    }

    pub fn len(&self) -> usize {
        self.batch_size
    }

    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    pub fn capacity(&self) -> usize {
        self.node_indices.capacity()
    }
}

impl Default for LeafBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
