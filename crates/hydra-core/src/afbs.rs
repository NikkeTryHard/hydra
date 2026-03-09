//! Anytime Factored-Belief Search (AFBS) with PUCT selection.
//!
//! Includes provenance-aware caching: every [`PonderResult`] carries
//! `source_net_hash`, `source_version`, [`TrustLevel`], and
//! [`CacheNamespace`] so consumers can decide whether a cached result
//! is safe to reuse at runtime vs. learner-only.

use crate::action::HYDRA_ACTION_SPACE;
use dashmap::DashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

pub const C_PUCT: f32 = 2.5;
pub const TOP_K: usize = 5;

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
            obs_buffer: Vec::with_capacity(batch_capacity * crate::encoder::OBS_SIZE),
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
            crate::encoder::OBS_SIZE,
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
mod tests {
    use super::*;

    #[test]
    fn puct_selects_high_prior_unvisited() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(0, 1.0, false);
        let c1 = tree.add_node(1, 0.8, false);
        let c2 = tree.add_node(2, 0.2, false);
        tree.nodes[root as usize].children = smallvec::smallvec![(0, c1), (1, c2)];
        tree.nodes[root as usize].visit_count = 1;
        let (action, _) = tree.puct_select(root).expect("should select");
        assert_eq!(action, 0, "should select high-prior child");
    }

    #[test]
    fn expand_creates_top_k_children() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(0, 1.0, false);
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits[0] = 5.0;
        logits[1] = 4.0;
        logits[2] = 3.0;
        logits[3] = 2.0;
        logits[4] = 1.0;
        logits[5] = 0.5;
        let mut mask = [false; HYDRA_ACTION_SPACE];
        for val in mask.iter_mut().take(10) {
            *val = true;
        }
        tree.expand_node(root, &logits, &mask, false);
        assert_eq!(tree.nodes[root as usize].children.len(), TOP_K);
    }

    #[test]
    fn expand_masks_illegal_actions_even_with_high_logits() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(0, 1.0, false);
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits[0] = 100.0;
        logits[1] = 10.0;
        logits[2] = 9.0;
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[1] = true;
        mask[2] = true;

        tree.expand_node(root, &logits, &mask, false);
        let actions = tree.child_actions(root);
        assert_eq!(actions, vec![1, 2]);
    }

    #[test]
    fn expand_is_idempotent_once_children_exist() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(0, 1.0, false);
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits[0] = 5.0;
        logits[1] = 4.0;
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[1] = true;

        tree.expand_node(root, &logits, &mask, false);
        let first_children = tree.child_actions(root);
        tree.expand_node(root, &logits, &mask, false);
        assert_eq!(tree.child_actions(root), first_children);
        assert_eq!(tree.num_children(root), 2);
    }

    #[test]
    fn backprop_updates_visits() {
        let mut tree = AfbsTree::new();
        let n0 = tree.add_node(0, 1.0, false);
        let n1 = tree.add_node(1, 0.5, false);
        tree.backpropagate(&[n0, n1], 1.0);
        assert_eq!(tree.nodes[n0 as usize].visit_count, 1);
        assert_eq!(tree.nodes[n1 as usize].visit_count, 1);
        assert!((tree.nodes[n1 as usize].q_value() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn search_iterations_descend_to_deepest_leaf() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(0, 1.0, false);
        let child = tree.add_node(1, 0.9, false);
        let leaf = tree.add_node(2, 0.8, false);
        tree.nodes[root as usize].children = smallvec::smallvec![(0, child)];
        tree.nodes[child as usize].children = smallvec::smallvec![(1, leaf)];
        tree.nodes[root as usize].visit_count = 1;
        tree.nodes[child as usize].visit_count = 1;

        tree.run_search_iterations(root, 3, &|idx| if idx == leaf { 0.75 } else { -1.0 });

        assert_eq!(tree.nodes[root as usize].visit_count, 4);
        assert_eq!(tree.nodes[child as usize].visit_count, 4);
        assert_eq!(tree.nodes[leaf as usize].visit_count, 3);
        assert!((tree.nodes[leaf as usize].q_value() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn search_iterations_evaluate_unexpanded_root() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(5, 1.0, false);
        tree.run_search_iterations(root, 2, &|idx| if idx == root { 0.5 } else { 0.0 });
        assert_eq!(tree.nodes[root as usize].visit_count, 2);
        assert!((tree.nodes[root as usize].q_value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn batched_eval_correct_size() {
        let mut batch = LeafBatch::new();
        let obs = [0.0f32; crate::encoder::OBS_SIZE];
        for i in 0..32 {
            batch.add(&obs, i);
        }
        assert_eq!(batch.batch_size, 32);
        assert!(batch.is_ready());
        assert_eq!(batch.node_indices.len(), 32);
        assert_eq!(batch.obs_buffer.len(), 32 * crate::encoder::OBS_SIZE);
    }

    #[test]
    fn leaf_batch_preallocates_min_batch_capacity() {
        let batch = LeafBatch::new();
        assert!(batch.capacity() >= MIN_BATCH);
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    #[should_panic(expected = "leaf observation must have OBS_SIZE elements")]
    fn leaf_batch_rejects_wrong_observation_width() {
        let mut batch = LeafBatch::new();
        let bad = [0.0f32; 4];
        batch.add(&bad, 0);
    }

    #[test]
    fn ponder_cache_hit_reuses_search() {
        let cache = PonderCache::new();
        let result = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.5, 4, 100);
        cache.insert(42, result);
        assert_eq!(cache.len(), 1);
        let hit = cache.get(42).expect("should find cached result");
        assert_eq!(hit.visit_count, 100);
        assert!((hit.value - 0.5).abs() < 1e-5);
        assert!(cache.get(99).is_none(), "miss should return None");
    }

    #[test]
    fn predictive_child_key_matches_tree_expansion_hash() {
        let parent_hash = 12345;
        let action = 7;
        assert_eq!(
            PonderCache::predicted_child_key(parent_hash, action),
            predicted_child_hash(parent_hash, action)
        );
    }

    #[test]
    fn shift_root_reuses_matching_child() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(100, 1.0, false);
        let child = tree.add_node(predicted_child_hash(100, 3), 0.7, false);
        tree.nodes[root as usize].children.push((3, child));
        let shifted = tree.shift_root_to_child(root, 3).expect("matching child");
        assert_eq!(shifted, child);
        assert!(tree.shift_root_to_child(root, 4).is_none());
    }

    #[test]
    fn ponder_manager_prioritizes_higher_score() {
        let mut manager = PonderManager::new();
        manager.enqueue_snapshot(GameStateSnapshot {
            info_state_hash: 1,
            top2_policy_gap: 0.2,
            risk_score: 0.1,
            particle_ess: 0.9,
        });
        manager.enqueue_snapshot(GameStateSnapshot {
            info_state_hash: 2,
            top2_policy_gap: 0.01,
            risk_score: 0.9,
            particle_ess: 0.2,
        });
        let next = manager.pop_task().expect("queued task");
        assert_eq!(
            next.info_state_hash, 2,
            "highest priority task should pop first"
        );
    }

    #[test]
    fn predictive_ponder_cache_roundtrip() {
        let cache = PonderCache::new();
        let parent_hash = 777;
        let action = 11;
        let result = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.25, 6, 48);
        cache.insert_predicted_child(parent_hash, action, result);
        let hit = cache
            .get_predicted_child(parent_hash, action)
            .expect("predicted child cache hit");
        assert_eq!(hit.visit_count, 48);
        assert!((hit.value - 0.25).abs() < 1e-6);
        assert_eq!(
            hit.cache_namespace,
            CacheNamespace::SpeculativeChildHint,
            "insert_predicted_child should set namespace"
        );
    }

    #[test]
    fn ponder_result_from_tree_reflects_root_stats() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(10, 1.0, false);
        let c0 = tree.add_node(11, 0.7, false);
        let c1 = tree.add_node(12, 0.3, false);
        tree.nodes[root as usize].children = smallvec::smallvec![(2, c0), (4, c1)];
        tree.nodes[root as usize].visit_count = 9;
        tree.nodes[c0 as usize].visit_count = 6;
        tree.nodes[c0 as usize].total_value = 3.0;
        tree.nodes[c1 as usize].visit_count = 3;
        tree.nodes[c1 as usize].total_value = 2.4;

        let result = PonderResult::from_tree(&tree, root, 0.42, 1.0, 0xDEAD, 1);
        assert_eq!(result.visit_count, 9);
        assert_eq!(result.search_depth, 1);
        assert!((result.value - 0.42).abs() < 1e-6);
        assert_eq!(result.source_net_hash, 0xDEAD);
        assert_eq!(result.source_version, 1);
        assert_eq!(result.trust_level, TrustLevel::LearnerOnly);
        let sum: f32 = result.exit_policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "ponder exit policy should be normalized"
        );
    }

    #[test]
    fn puct_balances_exploration_exploitation() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(0, 1.0, false);
        let c0 = tree.add_node(1, 0.5, false);
        let c1 = tree.add_node(2, 0.5, false);
        tree.nodes[root as usize].children = smallvec::smallvec![(0, c0), (1, c1)];
        tree.nodes[root as usize].visit_count = 10;
        tree.nodes[c0 as usize].visit_count = 8;
        tree.nodes[c0 as usize].total_value = 4.0;
        tree.nodes[c1 as usize].visit_count = 2;
        tree.nodes[c1 as usize].total_value = 1.5;
        let (action, _) = tree.puct_select(root).expect("select");
        assert_eq!(action, 1, "should explore less-visited child");
    }

    #[test]
    fn exit_policy_sums_to_one() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(0, 1.0, false);
        let c0 = tree.add_node(1, 0.5, false);
        let c1 = tree.add_node(2, 0.3, false);
        let c2 = tree.add_node(3, 0.2, false);
        tree.nodes[root as usize].children = smallvec::smallvec![(0, c0), (5, c1), (10, c2)];
        tree.nodes[c0 as usize].visit_count = 10;
        tree.nodes[c0 as usize].total_value = 5.0;
        tree.nodes[c1 as usize].visit_count = 5;
        tree.nodes[c1 as usize].total_value = 3.0;
        tree.nodes[c2 as usize].visit_count = 3;
        tree.nodes[c2 as usize].total_value = 0.9;
        let policy = tree.root_exit_policy(root, 1.0);
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "exit policy sum: {sum}");
        assert!(policy[0] > 0.0);
        assert!(policy[5] > 0.0);
        assert!(policy[10] > 0.0);
    }

    #[test]
    fn exit_policy_with_zero_tau_becomes_argmax() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(0, 1.0, false);
        let c0 = tree.add_node(1, 0.5, false);
        let c1 = tree.add_node(2, 0.5, false);
        tree.nodes[root as usize].children = smallvec::smallvec![(3, c0), (7, c1)];
        tree.nodes[c0 as usize].visit_count = 2;
        tree.nodes[c0 as usize].total_value = 1.0;
        tree.nodes[c1 as usize].visit_count = 2;
        tree.nodes[c1 as usize].total_value = 3.0;

        let policy = tree.root_exit_policy(root, 0.0);
        assert_eq!(policy[7], 1.0);
        assert_eq!(policy.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn has_any_legal_action_checks() {
        let empty = [false; HYDRA_ACTION_SPACE];
        assert!(!has_any_legal_action(&empty));
        let mut one = [false; HYDRA_ACTION_SPACE];
        one[45] = true;
        assert!(has_any_legal_action(&one));
    }

    #[test]
    fn trust_level_ordering() {
        assert!(TrustLevel::Authoritative.meets(TrustLevel::LearnerOnly));
        assert!(TrustLevel::Authoritative.meets(TrustLevel::Authoritative));
        assert!(TrustLevel::WarmStart.meets(TrustLevel::Advisory));
        assert!(!TrustLevel::LearnerOnly.meets(TrustLevel::Advisory));
        assert!(!TrustLevel::Advisory.meets(TrustLevel::WarmStart));
        assert!(!TrustLevel::WarmStart.meets(TrustLevel::Authoritative));
    }

    #[test]
    fn cache_generation_invalidation_rejects_stale() {
        let cache = PonderCache::new();
        let result = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.5, 4, 100);
        cache.insert(42, result);
        assert!(cache.get(42).is_some());

        cache.invalidate();
        assert!(
            cache.get(42).is_none(),
            "stale entry should be rejected after invalidation"
        );
        assert_eq!(cache.len(), 1, "physical entry still present");

        let fresh = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.7, 3, 50);
        cache.insert(42, fresh);
        let hit = cache.get(42).expect("fresh entry should be found");
        assert!((hit.value - 0.7).abs() < 1e-6);
    }

    #[test]
    fn cache_flush_clears_and_bumps_generation() {
        let cache = PonderCache::new();
        let gen_before = cache.current_generation();
        let result = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.5, 4, 100);
        cache.insert(42, result);
        cache.flush();
        assert!(cache.is_empty());
        assert!(cache.current_generation() > gen_before);
    }

    #[test]
    fn cache_get_trusted_filters_by_trust_level() {
        let cache = PonderCache::new();
        let result = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.5, 4, 100);
        cache.insert(42, result);

        assert!(
            cache.get_trusted(42, TrustLevel::LearnerOnly).is_some(),
            "LearnerOnly should match LearnerOnly"
        );
        assert!(
            cache.get_trusted(42, TrustLevel::Advisory).is_none(),
            "LearnerOnly should not meet Advisory"
        );
        assert!(
            cache.get_trusted(42, TrustLevel::Authoritative).is_none(),
            "LearnerOnly should not meet Authoritative"
        );

        let mut auth_result =
            PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.9, 8, 500);
        auth_result.trust_level = TrustLevel::Authoritative;
        cache.insert(99, auth_result);
        assert!(cache.get_trusted(99, TrustLevel::Authoritative).is_some());
        assert!(cache.get_trusted(99, TrustLevel::LearnerOnly).is_some());
    }

    #[test]
    fn insert_stamps_current_generation() {
        let cache = PonderCache::new();
        let gen1 = cache.current_generation();
        let result = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.5, 4, 100);
        cache.insert(1, result);
        let hit = cache.get(1).unwrap();
        assert_eq!(hit.generation, gen1);

        cache.invalidate();
        let gen2 = cache.current_generation();
        assert!(gen2 > gen1);
        let result2 = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.6, 3, 50);
        cache.insert(2, result2);
        let hit2 = cache.get(2).unwrap();
        assert_eq!(hit2.generation, gen2);
    }

    #[test]
    fn insert_predicted_child_sets_speculative_namespace() {
        let cache = PonderCache::new();
        let result = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.5, 4, 100);
        assert_eq!(result.cache_namespace, CacheNamespace::LearnerTarget);
        cache.insert_predicted_child(100, 5, result);
        let hit = cache.get_predicted_child(100, 5).unwrap();
        assert_eq!(hit.cache_namespace, CacheNamespace::SpeculativeChildHint);
    }

    #[test]
    fn ponder_manager_uses_provenance_cache() {
        let manager = PonderManager::new();
        let result = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.5, 4, 100);
        manager.cache_result(42, result);
        assert!(manager.lookup(42).is_some());

        manager.invalidate_cache();
        assert!(
            manager.lookup(42).is_none(),
            "invalidated entries should be rejected"
        );
    }

    #[test]
    fn ponder_manager_lookup_trusted() {
        let manager = PonderManager::new();
        let result = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.5, 4, 100);
        manager.cache_result(42, result);
        assert!(
            manager
                .lookup_trusted(42, TrustLevel::LearnerOnly)
                .is_some()
        );
        assert!(
            manager
                .lookup_trusted(42, TrustLevel::Authoritative)
                .is_none()
        );
    }

    #[test]
    fn from_tree_provenance_fields_are_set() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(10, 1.0, false);
        let c0 = tree.add_node(11, 0.7, false);
        tree.nodes[root as usize].children = smallvec::smallvec![(2, c0)];
        tree.nodes[root as usize].visit_count = 5;
        tree.nodes[c0 as usize].visit_count = 5;

        let result = PonderResult::from_tree(&tree, root, 0.5, 1.0, 0xBEEF, 42);
        assert_eq!(result.source_net_hash, 0xBEEF);
        assert_eq!(result.source_version, 42);
        assert_eq!(result.trust_level, TrustLevel::LearnerOnly);
        assert_eq!(result.cache_namespace, CacheNamespace::ObservedRoot);
        assert_eq!(
            result.generation, 0,
            "from_tree sets generation=0; cache stamps on insert"
        );
    }
}
