<combined_run_record run_id="answer_19" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 19 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_19_IMPLEMENT_TINY_PONDER_PRIORITY_SCORER.md">
  <![CDATA[# Prompt 19 — tiny ponder-priority scorer blueprint

<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Do not tell us to write a separate spec.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest blueprint for a tiny learned scorer that improves ponder priority or hard-state precompute ordering.

Keep the problem narrow.
Do not turn this into a broad runtime router.
Do not broaden into full metareasoning theory.

We want an answer that tells us what the narrow decision is, how to supervise it, what cheap signals to use, how to benchmark it, how to roll it out safely, and when to kill it.

Use the artifacts below to derive your conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- include reasoning
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples with numbers
- include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)
- distinguish direct artifact support from your own inference
- use search/browse to find the original paper, then inspect the full PDF with skill; use abstracts or summaries only for discovery, not as the final evidence base
- use the bash tool to run Python for calculations, math checks, and validation when rigor matters
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>

Artifact A — AFBS/search module full excerpt:

```rust
//! Anytime Factored-Belief Search (AFBS) with PUCT selection.

use crate::action::HYDRA_ACTION_SPACE;
use dashmap::DashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::time::Instant;

pub const C_PUCT: f32 = 2.5;
pub const TOP_K: usize = 5;

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
}

impl PonderResult {
    pub fn from_tree(tree: &AfbsTree, root_idx: NodeIdx, value: f32, tau: f32) -> Self {
        Self {
            exit_policy: tree.root_exit_policy(root_idx, tau),
            value,
            search_depth: tree.max_depth(root_idx),
            visit_count: tree.root_visit_count(root_idx),
            timestamp: Instant::now(),
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

pub struct PonderCache {
    entries: DashMap<u64, PonderResult>,
}

impl PonderCache {
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
        }
    }

    pub fn get(&self, hash: u64) -> Option<PonderResult> {
        self.entries.get(&hash).map(|entry| *entry.value())
    }

    pub fn insert(&self, hash: u64, result: PonderResult) {
        self.entries.insert(hash, result);
    }

    pub fn predicted_child_key(parent_hash: u64, action: u8) -> u64 {
        predicted_child_hash(parent_hash, action)
    }

    pub fn get_predicted_child(&self, parent_hash: u64, action: u8) -> Option<PonderResult> {
        self.get(Self::predicted_child_key(parent_hash, action))
    }

    pub fn insert_predicted_child(&self, parent_hash: u64, action: u8, result: PonderResult) {
        self.insert(Self::predicted_child_key(parent_hash, action), result);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn remove(&self, hash: u64) -> Option<PonderResult> {
        self.entries.remove(&hash).map(|(_, value)| value)
    }

    pub fn summary(&self) -> String {
        format!("cache(entries={})", self.entries.len())
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
}

impl Default for PonderCache {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PonderManager {
    pub cache: DashMap<u64, PonderResult>,
    pub priority_queue: std::collections::BinaryHeap<PonderTask>,
    pub worker_handle: Option<std::thread::JoinHandle<()>>,
}

impl PonderManager {
    pub fn new() -> Self {
        Self {
            cache: DashMap::new(),
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
        self.cache.get(&hash).map(|entry| *entry.value())
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
        let result = PonderResult {
            exit_policy: [0.0; HYDRA_ACTION_SPACE],
            value: 0.5,
            search_depth: 4,
            visit_count: 100,
            timestamp: Instant::now(),
        };
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
        let result = PonderResult {
            exit_policy: [0.0; HYDRA_ACTION_SPACE],
            value: 0.25,
            search_depth: 6,
            visit_count: 48,
            timestamp: Instant::now(),
        };
        cache.insert_predicted_child(parent_hash, action, result);
        let hit = cache
            .get_predicted_child(parent_hash, action)
            .expect("predicted child cache hit");
        assert_eq!(hit.visit_count, 48);
        assert!((hit.value - 0.25).abs() < 1e-6);
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

        let result = PonderResult::from_tree(&tree, root, 0.42, 1.0);
        assert_eq!(result.visit_count, 9);
        assert_eq!(result.search_depth, 1);
        assert!((result.value - 0.42).abs() < 1e-6);
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
}
```

Artifact B — inference/runtime seam full excerpt:

```rust
//! Inference server: fast path (network + SaF) and slow path (pondered AFBS).

use burn::prelude::*;
use burn::tensor::activation;
use dashmap::DashMap;
use hydra_core::action::{AGARI, HYDRA_ACTION_SPACE};
use hydra_core::afbs::PonderResult;
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES};
use std::sync::Arc;

use crate::model::ActorNet;
use crate::saf::{SafConfig, SafMlp, apply_saf_logit, saf_tensor_from_observation};

pub const OBS_FLAT_SIZE: usize = NUM_CHANNELS * NUM_TILES;

pub struct InferenceConfig {
    pub on_turn_budget_ms: u64,
    pub call_reaction_budget_ms: u64,
    pub agari_guard: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            on_turn_budget_ms: 150,
            call_reaction_budget_ms: 50,
            agari_guard: true,
        }
    }
}

impl InferenceConfig {
    pub fn summary(&self) -> String {
        format!(
            "infer(turn={}ms, call={}ms, guard={})",
            self.on_turn_budget_ms, self.call_reaction_budget_ms, self.agari_guard
        )
    }
}

pub struct InferenceServer<B: Backend> {
    pub actor: ActorNet<B>,
    pub ponder_cache: Arc<DashMap<u64, PonderResult>>,
    pub saf_mlp: SafMlp<B>,
    pub config: InferenceConfig,
    saf_alpha: f32,
    device: B::Device,
}

impl<B: Backend> InferenceServer<B> {
    pub fn new(
        actor: ActorNet<B>,
        ponder_cache: Arc<DashMap<u64, PonderResult>>,
        saf_mlp: SafMlp<B>,
        saf_alpha: f32,
        config: InferenceConfig,
        device: B::Device,
    ) -> Self {
        Self {
            actor,
            ponder_cache,
            saf_mlp,
            config,
            saf_alpha,
            device,
        }
    }

    pub fn from_configs(
        actor: ActorNet<B>,
        saf_config: &SafConfig,
        config: InferenceConfig,
        device: B::Device,
    ) -> Self {
        let saf_alpha = saf_config.alpha;
        let saf_mlp = saf_config.init(&device);
        Self::new(
            actor,
            Arc::new(DashMap::new()),
            saf_mlp,
            saf_alpha,
            config,
            device,
        )
    }

    pub fn info_state_hash(obs: &[f32; OBS_FLAT_SIZE]) -> u64 {
        obs.iter().fold(0xcbf29ce484222325, |hash, value| {
            hash.wrapping_mul(0x100000001b3) ^ value.to_bits() as u64
        })
    }

    pub fn cache_ponder_result(&self, info_state_hash: u64, result: PonderResult) {
        self.ponder_cache.insert(info_state_hash, result);
    }

    pub fn lookup_ponder(&self, info_state_hash: u64) -> Option<PonderResult> {
        self.ponder_cache
            .get(&info_state_hash)
            .map(|entry| *entry.value())
    }

    pub fn infer(
        &self,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
    ) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
        let (action, policy, _) = self.infer_timed(obs, legal);
        (action, policy)
    }

    pub fn infer_call_reaction(
        &self,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
    ) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
        let (action, policy, _) = self.infer_call_reaction_timed(obs, legal);
        (action, policy)
    }

    pub fn infer_timed(
        &self,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
    ) -> (u8, [f32; HYDRA_ACTION_SPACE], bool) {
        self.infer_with_budget(obs, legal, self.config.on_turn_budget_ms)
    }

    pub fn infer_call_reaction_timed(
        &self,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
    ) -> (u8, [f32; HYDRA_ACTION_SPACE], bool) {
        self.infer_with_budget(obs, legal, self.config.call_reaction_budget_ms)
    }

    fn infer_with_budget(
        &self,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
        budget_ms: u64,
    ) -> (u8, [f32; HYDRA_ACTION_SPACE], bool) {
        let start = std::time::Instant::now();
        let info_state_hash = Self::info_state_hash(obs);

        if let Some(pondered) = self.lookup_ponder(info_state_hash) {
            let policy = mask_policy_cpu(&pondered.exit_policy, legal);
            let action = self.guard_action(argmax_legal(&policy, legal), legal);
            let within = start.elapsed().as_millis() as u64 <= budget_ms;
            return (action, policy, within);
        }

        let input = Tensor::<B, 1>::from_floats(obs.as_slice(), &self.device).reshape([
            1,
            NUM_CHANNELS,
            NUM_TILES,
        ]);
        let base_logits = self.actor.policy_logits_for(input);
        let logits = self.apply_saf_fast_path(base_logits, obs, legal);
        let (action, policy, within) = infer_action_timed(logits, legal, budget_ms);
        (self.guard_action(action, legal), policy, within)
    }

    fn apply_saf_fast_path(
        &self,
        base_logits: Tensor<B, 2>,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
    ) -> Tensor<B, 2> {
        let saf_features = saf_tensor_from_observation::<B>(obs.as_slice(), &self.device);
        let saf_delta = self
            .saf_mlp
            .forward(saf_features)
            .reshape([1, HYDRA_ACTION_SPACE]);
        let mask_tensor = legal_mask_to_tensor(legal, &self.device);
        apply_saf_logit(base_logits, saf_delta, mask_tensor, self.saf_alpha)
    }

    fn guard_action(&self, action: u8, legal: &[bool; HYDRA_ACTION_SPACE]) -> u8 {
        if self.config.agari_guard && action == AGARI && !legal[action as usize] {
            return argmax_legal(&mask_policy_cpu(&[0.0; HYDRA_ACTION_SPACE], legal), legal);
        }
        if legal[action as usize] {
            action
        } else {
            argmax_legal(&mask_policy_cpu(&[0.0; HYDRA_ACTION_SPACE], legal), legal)
        }
    }
}

pub fn legal_mask_to_tensor<B: Backend>(
    mask: &[bool; HYDRA_ACTION_SPACE],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut f32_mask = [0.0f32; HYDRA_ACTION_SPACE];
    for (i, &m) in mask.iter().enumerate() {
        f32_mask[i] = if m { 1.0 } else { 0.0 };
    }
    Tensor::<B, 1>::from_floats(&f32_mask[..], device).unsqueeze_dim::<2>(0)
}

pub fn normalize_policy_cpu(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut adjusted = [f32::NEG_INFINITY; HYDRA_ACTION_SPACE];
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            adjusted[i] = logits[i];
            if logits[i] > max_val {
                max_val = logits[i];
            }
        }
    }
    let mut probs = [0.0f32; HYDRA_ACTION_SPACE];
    let mut total = 0.0f32;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            probs[i] = (adjusted[i] - max_val).exp();
            total += probs[i];
        }
    }
    if total > 0.0 {
        for p in &mut probs {
            *p /= total;
        }
    }
    probs
}

pub fn mask_policy_cpu(
    policy: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut masked = [0.0f32; HYDRA_ACTION_SPACE];
    let mut total = 0.0f32;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            masked[i] = policy[i].max(0.0);
            total += masked[i];
        }
    }
    if total > 0.0 {
        for value in &mut masked {
            *value /= total;
        }
        return masked;
    }

    let legal_count = legal_mask.iter().filter(|&&m| m).count();
    if legal_count > 0 {
        let uniform = 1.0 / legal_count as f32;
        for (i, value) in masked.iter_mut().enumerate() {
            if legal_mask[i] {
                *value = uniform;
            }
        }
    }
    masked
}

pub fn validate_legal_mask(mask: &[bool; HYDRA_ACTION_SPACE]) -> bool {
    mask.iter().any(|&m| m)
}

pub fn policy_entropy(probs: &[f32; HYDRA_ACTION_SPACE]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 1e-8 {
            h -= p * p.ln();
        }
    }
    h
}

pub fn action_rank(probs: &[f32; HYDRA_ACTION_SPACE], action: u8) -> usize {
    let p = probs[action as usize];
    probs.iter().filter(|&&q| q > p).count()
}

pub fn needs_search(probs: &[f32; HYDRA_ACTION_SPACE], gap_threshold: f32) -> bool {
    policy_top2_gap(probs) < gap_threshold
}

pub fn is_confident(probs: &[f32; HYDRA_ACTION_SPACE], threshold: f32) -> bool {
    policy_top1_confidence(probs) >= threshold
}

pub fn sample_from_policy(probs: &[f32; HYDRA_ACTION_SPACE], rng_val: f32) -> u8 {
    let mut cumsum = 0.0f32;
    let mut last_positive = 0u8;
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            last_positive = i as u8;
        }
        cumsum += p;
        if rng_val <= cumsum {
            return i as u8;
        }
    }
    last_positive
}

pub fn num_legal_actions(mask: &[bool; HYDRA_ACTION_SPACE]) -> usize {
    mask.iter().filter(|&&m| m).count()
}

pub fn argmax_legal(probs: &[f32; HYDRA_ACTION_SPACE], mask: &[bool; HYDRA_ACTION_SPACE]) -> u8 {
    let mut best = 0u8;
    let mut best_p = f32::NEG_INFINITY;
    for (i, (&p, &m)) in probs.iter().zip(mask.iter()).enumerate() {
        if m && p > best_p {
            best_p = p;
            best = i as u8;
        }
    }
    best
}

pub fn compute_entropy_from_logits(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> f32 {
    let probs = normalize_policy_cpu(logits, legal_mask);
    policy_entropy(&probs)
}

pub fn policy_top2_gap(probs: &[f32; HYDRA_ACTION_SPACE]) -> f32 {
    let mut first = 0.0f32;
    let mut second = 0.0f32;
    for &p in probs {
        if p > first {
            second = first;
            first = p;
        } else if p > second {
            second = p;
        }
    }
    first - second
}

pub fn policy_top1_confidence(probs: &[f32; HYDRA_ACTION_SPACE]) -> f32 {
    probs.iter().cloned().fold(0.0f32, f32::max)
}

pub fn batch_legal_masks_to_tensor<B: Backend>(
    masks: &[[bool; HYDRA_ACTION_SPACE]],
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch = masks.len();
    let mut flat = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    for (i, mask) in masks.iter().enumerate() {
        for (j, &m) in mask.iter().enumerate() {
            if m {
                flat[i * HYDRA_ACTION_SPACE + j] = 1.0;
            }
        }
    }
    Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([batch, HYDRA_ACTION_SPACE])
}

pub fn infer_action_timed<B: Backend>(
    policy_logits: Tensor<B, 2>,
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    budget_ms: u64,
) -> (u8, [f32; HYDRA_ACTION_SPACE], bool) {
    let start = std::time::Instant::now();
    let (action, policy) = infer_action(policy_logits, legal_mask);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let within_budget = elapsed_ms <= budget_ms;
    (action, policy, within_budget)
}

pub fn illegal_action_rate<B: Backend>(logits: Tensor<B, 2>, legal_mask: Tensor<B, 2>) -> f32 {
    let neg_inf = (legal_mask.clone().ones_like() - legal_mask.clone()) * (-1e9f32);
    let raw_predicted = logits.clone().argmax(1);
    let masked = logits + neg_inf;
    let predicted = masked.argmax(1);
    let same = predicted.equal(raw_predicted).int().sum();
    let batch = legal_mask.dims()[0] as f32;
    1.0 - same.into_scalar().elem::<f32>() / batch
}

pub fn infer_action<B: Backend>(
    policy_logits: Tensor<B, 2>,
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
    let device = policy_logits.device();
    let mask_tensor = legal_mask_to_tensor(legal_mask, &device);
    let neg_inf = (mask_tensor.ones_like() - mask_tensor) * (-1e9f32);
    let masked = policy_logits + neg_inf;
    let probs = activation::softmax(masked, 1);
    let probs_data = probs.to_data();
    let mut policy = [0.0f32; HYDRA_ACTION_SPACE];
    if let Ok(probs_slice) = probs_data.as_slice::<f32>() {
        policy.copy_from_slice(&probs_slice[..HYDRA_ACTION_SPACE]);
    }

    let mut best_action = 0u8;
    let mut best_prob = f32::NEG_INFINITY;
    for (i, &p) in policy.iter().enumerate() {
        if legal_mask[i] && p > best_prob {
            best_prob = p;
            best_action = i as u8;
        }
    }
    (best_action, policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn make_server(device: &<B as Backend>::Device) -> InferenceServer<B> {
        let actor = crate::model::HydraModelConfig::actor().init::<B>(device);
        InferenceServer::from_configs(
            actor,
            &SafConfig::new(),
            InferenceConfig::default(),
            *device,
        )
    }

    #[test]
    fn inference_picks_legal_action() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats(
            [[
                10.0, -10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]],
            &device,
        );
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[1] = true;
        mask[2] = true;
        let (action, policy) = infer_action(logits, &mask);
        assert!(mask[action as usize], "picked illegal action {action}");
        let sum: f32 = policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "policy should sum to 1, got {sum}"
        );
    }

    #[test]
    fn agari_guard_prevents_illegal() {
        let device = Default::default();
        let mut logits_data = [0.0f32; HYDRA_ACTION_SPACE];
        logits_data[43] = 100.0;
        let logits = Tensor::<B, 1>::from_floats(&logits_data[..], &device).unsqueeze_dim::<2>(0);
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[45] = true;
        let (action, _) = infer_action(logits, &mask);
        assert_ne!(action, 43, "agari (43) is illegal but has highest logit");
        assert!(mask[action as usize], "must pick legal: got {action}");
    }

    #[test]
    fn inference_config_defaults() {
        let cfg = InferenceConfig::default();
        assert_eq!(cfg.on_turn_budget_ms, 150);
        assert_eq!(cfg.call_reaction_budget_ms, 50);
        assert!(cfg.agari_guard);
    }

    #[test]
    fn illegal_actions_get_zero_probability() {
        let device = Default::default();
        let mut logits_data = [0.0f32; HYDRA_ACTION_SPACE];
        logits_data[0] = 5.0;
        logits_data[1] = 3.0;
        logits_data[2] = 1.0;
        let logits = Tensor::<B, 1>::from_floats(&logits_data[..], &device).unsqueeze_dim::<2>(0);
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[2] = true;
        let (_, policy) = infer_action(logits, &mask);
        assert!(
            policy[1] < 1e-6,
            "illegal action 1 should have ~0 prob: {}",
            policy[1]
        );
        assert!(policy[0] > 0.1, "legal action 0 should have significant prob");
        assert!(policy[2] > 0.01, "legal action 2 should have some prob");
    }

    #[test]
    fn normalize_policy_cpu_sums_to_one() {
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits[0] = 5.0;
        logits[5] = 3.0;
        logits[10] = 1.0;
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[5] = true;
        mask[10] = true;
        let probs = normalize_policy_cpu(&logits, &mask);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum: {sum}");
        assert!(probs[0] > probs[5]);
        assert!(probs[5] > probs[10]);
    }

    #[test]
    fn mask_policy_cpu_renormalizes_legal_mass() {
        let mut policy = [0.0f32; HYDRA_ACTION_SPACE];
        policy[1] = 0.8;
        policy[2] = 0.2;
        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[2] = true;
        legal[3] = true;
        let masked = mask_policy_cpu(&policy, &legal);
        assert_eq!(masked[1], 0.0);
        assert!((masked[2] - 1.0).abs() < 1e-6);
        assert_eq!(masked[3], 0.0);
    }

    #[test]
    fn sample_from_policy_respects_distribution() {
        let mut probs = [0.0f32; HYDRA_ACTION_SPACE];
        probs[0] = 0.7;
        probs[1] = 0.3;
        let a0 = sample_from_policy(&probs, 0.0);
        assert_eq!(a0, 0);
        let a1 = sample_from_policy(&probs, 0.8);
        assert_eq!(a1, 1);
    }

    #[test]
    fn inference_respects_time_budget() {
        let device = Default::default();
        let model = crate::model::HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([1, NUM_CHANNELS, 34], &device);
        let out = model.forward(x);
        let mut mask = [true; HYDRA_ACTION_SPACE];
        mask[45] = false;
        let (action, policy, within) = infer_action_timed(out.policy_logits, &mask, 5000);
        assert!(mask[action as usize], "must pick legal action");
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "policy sum: {sum}");
        assert!(within, "5s budget should be plenty for CPU inference");
    }

    #[test]
    fn inference_server_respects_time_budget() {
        let device = Default::default();
        let mut server = make_server(&device);
        server.config.on_turn_budget_ms = 5_000;
        let obs = [0.0f32; OBS_FLAT_SIZE];
        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[0] = true;
        legal[1] = true;
        let (action, policy, within) = server.infer_timed(&obs, &legal);
        assert!(legal[action as usize]);
        assert!(
            within,
            "5s budget should be sufficient for fast-path inference"
        );
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn inference_server_reuses_cached_ponder_policy() {
        let device = Default::default();
        let server = make_server(&device);
        let obs = [0.0f32; OBS_FLAT_SIZE];
        let hash = InferenceServer::<B>::info_state_hash(&obs);
        let mut exit_policy = [0.0f32; HYDRA_ACTION_SPACE];
        exit_policy[5] = 0.9;
        exit_policy[6] = 0.1;
        server.cache_ponder_result(
            hash,
            PonderResult {
                exit_policy,
                value: 0.3,
                search_depth: 5,
                visit_count: 64,
                timestamp: std::time::Instant::now(),
            },
        );
        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[5] = true;
        legal[6] = true;
        let (action, policy) = server.infer(&obs, &legal);
        assert_eq!(action, 5);
        assert!(policy[5] > policy[6]);
    }

    #[test]
    fn inference_server_uses_call_reaction_budget() {
        let device = Default::default();
        let mut server = make_server(&device);
        server.config.call_reaction_budget_ms = 5_000;
        let obs = [0.0f32; OBS_FLAT_SIZE];
        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[3] = true;
        legal[4] = true;
        let (action, policy, within) = server.infer_call_reaction_timed(&obs, &legal);
        assert!(legal[action as usize]);
        assert!(
            within,
            "call reaction inference should honor the call budget"
        );
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_needs_search_close_gap() {
        let mut probs = [0.0f32; HYDRA_ACTION_SPACE];
        probs[0] = 0.35;
        probs[1] = 0.34;
        probs[2] = 0.31;
        assert!(
            needs_search(&probs, 0.05),
            "top-2 gap of 0.01 < threshold 0.05 should trigger search"
        );
    }
}
```

Artifact C — benchmark and doctrine excerpts:

```text
Pondering = label amplification.
75% idle time used for: deepening current root search + precomputing searches for predicted near-future states. Every completed search yields additional labeled training examples.
```

```text
Playout cap randomization: More compute when top-2 policy gap is small, in high-risk defense contexts, or when particle ESS is low.
```

```rust
pub struct BenchmarkGates {
    pub afbs_on_turn_ms: f32,
    pub ct_smc_dp_ms: f32,
    pub endgame_ms: f32,
    pub self_play_games_per_sec: f32,
    pub distill_kl_drift: f32,
}

impl BenchmarkGates {
    pub fn passes(&self) -> bool {
        self.afbs_on_turn_ms < 150.0
            && self.ct_smc_dp_ms < 1.0
            && self.endgame_ms < 100.0
            && self.self_play_games_per_sec > 20.0
            && self.distill_kl_drift < 0.1
    }
}
```

```text
Evaluation harness summary fields: mean placement, stable dan, win rate, deal-in rate, tsumo rate.
```

</artifacts>
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_19.md">
  <![CDATA[## Blueprint: tiny learned scorer for the next ponder quantum

### 1. The decision to learn

Learn exactly one thing:

> **Given the current ponder queue and one fixed idle-search quantum `Q`, which queued task should get that next quantum?**

Do **not** learn:

* action selection,
* whether to ponder at all,
* total idle budget,
* a full runtime router,
* ACT/PonderNet-style variable halting.

Keep any existing top-level budget split unchanged. The scorer only orders tasks **inside** the ponder queue.

This is the right seam because your artifacts already expose a scalar priority slot:

* `PonderManager` uses `BinaryHeap<PonderTask>`.
* `PonderTask` already carries a `priority_score`.
* the current heuristic is narrow and hand-coded:

[
h_{\text{hand}}(s)=10\max(0,0.1-\text{gap})+\max(0,\text{risk})+\max(0,1-\text{ESS})
]

That is already a “tiny scorer”; the learned replacement should stay that small.

### 2. What the artifacts directly support, and what that implies

**Direct artifact support**

* Queue ordering is already scalar and local: `priority_score` on `PonderTask`.
* The current hand score uses exactly three cheap uncertainty signals: `top2_policy_gap`, `risk_score`, `particle_ess`.
* Doctrine says “pondering = label amplification” and that extra compute is used on small top-2 gaps, high-risk defense, and low ESS states.
* `InferenceServer` only consumes cached ponder results by exact `info_state_hash(obs)`.
* `PonderResult` already stores `visit_count`, `search_depth`, and `timestamp`.
* `expand_node` truncates to `TOP_K`, then renormalizes the kept priors.

**Inference / recommendation**

* The model should predict **marginal value of one fixed search increment**, not total state value.
* Runtime ordering should multiply that intrinsic marginal value by **reach probability** for speculative future states.
* Label-harvest ordering should use the intrinsic marginal value **without** reach probability.

That shape is much narrower than ACT/PonderNet-style learned halting, which are about end-to-end adaptive computation. ACT explicitly adds a hand-chosen time penalty and is sensitive to that coefficient; PonderNet improves adaptive compute but still learns a full halting policy. Good background, wrong serving seam here. ([arXiv][1])

### 3. Blockers to fix before training anything

These are not optional.

#### 3.1 Resolve the speculative-cache key seam

In Artifact A, speculative child cache keys are built with:

```rust
predicted_child_hash(parent_hash, action)
```

which is an XOR-derived synthetic key.

In Artifact B, serving looks up ponder results with:

```rust
InferenceServer::info_state_hash(obs)
```

which is an FNV hash over the actual observation floats.

Unless some unshown reconciliation layer exists, those are different key spaces. If that mismatch is real, predicted-child precompute cannot hit the fast path, so improving speculative ordering will not improve runtime.

**Required fix**

* either enqueue speculative children with the **actual child observation hash**,
* or keep both:

  * `spec_key` for background bookkeeping,
  * `actual_info_state_hash` once realized,
  * and reconcile them on state realization.

If you do not fix this, kill the project before model work.

#### 3.2 Do not use renormalized top-k priors as reach probability

`expand_node` does:

1. softmax over legal actions,
2. truncate to `TOP_K`,
3. renormalize selected mass to 1.

So `child.prior` after expansion is **not** the true probability of reaching that child under the parent action distribution.

**Required fix**

* store `reach_prob_raw = parent_policy[action]` **before** `TOP_K` truncation/renorm.
* use that raw probability for runtime priority.

#### 3.3 Add staleness and dedup

The excerpt shows:

* `timestamp` exists on `PonderResult`, but no freshness gate in serving.
* no obvious queue dedup / generation check.
* multiple cache abstractions (`PonderCache`, `PonderManager.cache`, `InferenceServer.ponder_cache`) that need auditing to ensure they are the same live cache path.

**Required fix**

* add queue generation counters or stale-bit discard,
* add `max_cache_age_ms`,
* ensure the inference server and ponder worker share the same cache instance end to end.

### 4. The fixed compute quantum

Use a constant quantum:

[
Q = 32 \text{ leaf evals / AFBS iterations}
]

Reason:

* `MIN_BATCH = 32` already exists.
* one batch is a natural indivisible unit.
* keeping `Q` fixed makes labels comparable.

Use a fixed deeper reference budget:

[
R = 16Q = 512
]

Do not make `Q` input-dependent. The scorer is supposed to rank tasks for the **next** quantum, not learn halting.

### 5. The target to supervise

Let a queued task be in its **current** search state at enqueue time, not necessarily from zero visits.

Define:

* `π0, v0`: current shallow output before the extra quantum,
* `πQ, vQ`: output after adding exactly `Q` more iterations,
* `πR, vR`: output after continuing to `R` total extra iterations.

Use the same legal masking and normalization used in serving.

#### 5.1 Intrinsic marginal gain target

Policy component:

[
u_\pi = \max\left(0,; D_{\mathrm{KL}}(\pi_R ,|, \pi_0) - D_{\mathrm{KL}}(\pi_R ,|, \pi_Q)\right)
]

Value component:

[
u_v = \max\left(0,; |v_R-v_0| - |v_R-v_Q|\right)
]

Action-correction bonus:

[
u_a =
\begin{cases}
1 & \text{if } \arg\max \pi_0 \neq \arg\max \pi_Q \text{ and } \arg\max \pi_Q = \arg\max \pi_R \
0 & \text{otherwise}
\end{cases}
]

Combined target:

[
y_{\text{hard}} = u_\pi + \lambda_v u_v + \lambda_a u_a
]

Start with:

* `λ_v = 2.0`
* `λ_a = 0.25`

Then retune on held-out replay only if needed.

This target says: “How much closer did one more quantum move me toward a much stronger search result?” That is the right supervision for a tiny scorer that schedules compute. It mirrors the learned-branching pattern: imitate the marginal benefit of a stronger but expensive expert rather than optimize long-horizon outcome directly. Learned branching papers make this exact tradeoff argument: a cheaper predictor can improve total solve speed even if it is weaker than the expensive expert on a per-decision basis. ([arXiv][2])

#### 5.2 Runtime priority vs label priority

Train the model on `y_hard` only.

Then use two serving formulas:

Runtime ponder priority:
[
p_{\text{runtime}} = p_{\text{reach}} \cdot \hat y_{\text{hard}}
]

Hard-state / label-harvest priority:
[
p_{\text{label}} = \hat y_{\text{hard}}
]

This is the key narrow trick:

* runtime cares whether the state is likely to be reached,
* label amplification cares whether the state is intrinsically informative.

That matches your doctrine that pondering is both runtime help and label generation, and it is very close in spirit to KataGo’s compute allocation: vary search effort, but use stronger-search outputs as the valuable training target. KataGo’s playout-cap randomization explicitly mixed fast and full search, recorded only full-search turns for training, and improved compute efficiency over fixed playout counts. ([ar5iv][3])

### 6. Cheap signals to use

Keep the model tiny. Start with signals that are already in the artifacts or are derivable by existing helpers.

#### 6.1 Mandatory v1 features

Use these five first:

[
x_1 = \mathrm{clip}\left(\frac{0.10 - \text{gap}}{0.10}, 0, 1\right)
]

[
x_2 = \mathrm{robust_scale}(\text{risk_score})
]

[
x_3 = \mathrm{clip}(1-\text{particle_ess}, 0, 1)
]

[
x_4 = \mathbf{1}{\text{root_task}}
]

[
x_5 = \mathbf{1}{\text{call_reaction_mode}}
]

`reach_prob` is **not** an input feature. It is an external multiplier at serving time.

Why:

* `gap`, `risk`, `ESS` are direct artifact signals.
* `root_task` matters because root deepening is guaranteed reachable.
* `call_reaction_mode` matters because the runtime regime differs (50 ms vs 150 ms budgets in the seam).

#### 6.2 Optional zero-cost extras

Only add these if they are already available at enqueue time without extra model calls:

[
x_6 = \frac{H(\pi)}{\log(\max(2, |\mathcal{A}_{legal}|))}
]

[
x_7 = \max_a \pi(a)
]

[
x_8 = \frac{|\mathcal{A}_{legal}|}{\text{HYDRA_ACTION_SPACE}}
]

[
x_9 = \frac{\log(1+\text{visit_count})}{\log(1+V_{\max})}
]

[
x_{10} = \frac{\text{search_depth}}{D_{\max}}
]

Use `policy_entropy`, `policy_top1_confidence`, and `num_legal_actions` from the inference seam, and `visit_count` / `search_depth` from `PonderResult`.

Interpretation:

* entropy / many legal actions often mean more room for search to matter,
* high current visits / depth mean diminishing marginal benefit from one more quantum.

Do **not** add raw observation embeddings, full hidden-state features, or broad runtime counters in v1. That turns this into a router.

### 7. The model

Use a sign-constrained linear model.

[
\hat y_{\text{hard}} = \max(0,; b + w^\top x)
]

Sign constraints:

* `w_gap_need >= 0`
* `w_risk >= 0`
* `w_lowess >= 0`
* `w_entropy >= 0` if used
* `w_legal_frac >= 0` if used
* `w_log_visits <= 0` if used
* `w_depth <= 0` if used

This preserves the monotonic shape already encoded by the hand heuristic while letting the weights be learned.

Do **not** start with a tree model or MLP. A tiny local linear scorer is more than enough here. SmartChoices is a strong deployment pattern for this: local CPU inference, fallback to a default policy, validation gates relative to baseline, canary and rollback. HALP is the same lesson in a different domain: augment a heuristic with ML rather than replace it end to end. ([arXiv][4])

#### 7.1 Training objective

Fit on:

[
t = \log(1+y_{\text{hard}})
]

with Huber loss and `L2` regularization, under the sign constraints.

That gives stable regression while preserving ranking.

Only move to pairwise ranking loss if offline replay shows regression loss saturates.

### 8. Data you need to log

For each queued task, log one row:

```text
game_id
turn_idx
task_kind              # root vs predicted child
info_state_hash
spec_key               # if speculative
parent_hash
parent_action
reach_prob_raw         # pre-TOP_K probability
top2_gap
risk_score
particle_ess
policy_entropy         # optional
top1_confidence        # optional
legal_action_count     # optional
visit_count            # optional
search_depth           # optional
call_reaction_mode
enqueue_ts
pi0, v0
piQ, vQ
piR, vR
was_realized           # did this predicted state actually occur?
cache_hit_on_realize   # yes/no
```

Two rules matter:

1. `reach_prob_raw` must be from the pre-truncation parent policy.
2. `pi0 -> piQ -> piR` must continue from the task’s **actual current search state**, not restart from zero.

### 9. Rust serving shape

```rust
pub const PONDER_Q_ITERS: u32 = 32;
pub const PONDER_R_ITERS: u32 = 512;
pub const PONDER_CACHE_TTL_MS: u64 = 5_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PonderTaskKind {
    RootDeepen,
    PredictedChild,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GameStateSnapshot {
    pub info_state_hash: u64,
    pub top2_policy_gap: f32,
    pub risk_score: f32,
    pub particle_ess: f32,

    pub reach_prob_raw: f32,   // new; 1.0 for root tasks
    pub task_kind: PonderTaskKind,
    pub call_reaction_mode: bool,

    pub policy_entropy: f32,   // optional
    pub top1_confidence: f32,  // optional
    pub legal_action_count: u8, // optional
    pub visit_count: u32,      // optional
    pub search_depth: u8,      // optional
}
```

```rust
pub struct TinyPonderScorer {
    pub w: [f32; 10],
    pub b: f32,
    pub lo: [f32; 10],
    pub hi: [f32; 10],
}

#[inline]
fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

#[inline]
fn robust_scale(x: f32, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return 0.0;
    }
    clamp01((x - lo) / (hi - lo))
}

impl TinyPonderScorer {
    pub fn hardness_gain(&self, s: &GameStateSnapshot) -> f32 {
        let mut x = [0.0f32; 10];

        x[0] = clamp01((0.10 - s.top2_policy_gap) / 0.10);
        x[1] = robust_scale(s.risk_score, self.lo[1], self.hi[1]);
        x[2] = clamp01(1.0 - s.particle_ess);
        x[3] = if s.task_kind == PonderTaskKind::RootDeepen { 1.0 } else { 0.0 };
        x[4] = if s.call_reaction_mode { 1.0 } else { 0.0 };

        x[5] = robust_scale(s.policy_entropy, self.lo[5], self.hi[5]);
        x[6] = robust_scale(s.top1_confidence, self.lo[6], self.hi[6]);
        x[7] = robust_scale(s.legal_action_count as f32, self.lo[7], self.hi[7]);
        x[8] = robust_scale((1.0 + s.visit_count as f32).ln(), self.lo[8], self.hi[8]);
        x[9] = robust_scale(s.search_depth as f32, self.lo[9], self.hi[9]);

        let mut z = self.b;
        for i in 0..10 {
            z += self.w[i] * x[i];
        }
        z.max(0.0)
    }

    pub fn runtime_priority(&self, s: &GameStateSnapshot) -> f32 {
        self.hardness_gain(s) * clamp01(s.reach_prob_raw)
    }

    pub fn label_priority(&self, s: &GameStateSnapshot) -> f32 {
        self.hardness_gain(s)
    }
}
```

### 10. Python label builder

```python
import numpy as np

EPS = 1e-6
LAMBDA_V = 2.0
LAMBDA_A = 0.25

def legal_renorm(pi, legal_mask):
    pi = np.asarray(pi, dtype=np.float32).copy()
    pi[~legal_mask] = 0.0
    s = pi.sum()
    if s <= 0:
        pi[legal_mask] = 1.0 / legal_mask.sum()
    else:
        pi /= s
    return pi

def kl_teacher(ref_pi, student_pi):
    p = np.clip(ref_pi, EPS, 1.0)
    q = np.clip(student_pi, EPS, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def marginal_hard_gain(pi0, v0, pi_q, v_q, pi_r, v_r):
    u_pi = max(0.0, kl_teacher(pi_r, pi0) - kl_teacher(pi_r, pi_q))
    u_v  = max(0.0, abs(v_r - v0) - abs(v_r - v_q))

    a0 = int(np.argmax(pi0))
    aq = int(np.argmax(pi_q))
    ar = int(np.argmax(pi_r))
    u_a = 1.0 if (a0 != aq and aq == ar) else 0.0

    return u_pi + LAMBDA_V * u_v + LAMBDA_A * u_a
```

### 11. Benchmarking

Use three levels.

#### 11.1 Offline queue-replay metric

For each idle slot `t`, let `Q_t` be the queued tasks and `B_t` the number of quanta available.

For a scheduler `S`, define realized replay utility:

[
U(S)=\sum_t\sum_{i\in S(Q_t,B_t)} \mathbf{1}{\text{was_realized}*i}; y*{\text{hard},i}
]

Then compare:

* current heuristic,
* heuristic + `reach_prob_raw`,
* learned v1 mandatory features,
* learned v1 + optional extras,
* oracle ranking by realized `was_realized * y_hard`.

Primary metric:

[
\text{Capture}(S)=\frac{U(S)}{U(\text{oracle})}
]

This is the metric that matters most.

#### 11.2 Label-amplification metric

Ignore realization and measure:

[
U_{\text{label}}(S)=\sum_t\sum_{i\in S(Q_t,B_t)} y_{\text{hard},i}
]

This tells you whether the scorer is surfacing intrinsically informative hard states.

#### 11.3 End-to-end gates

Keep your existing artifact gates unchanged:

* `afbs_on_turn_ms < 150`
* `ct_smc_dp_ms < 1`
* `endgame_ms < 100`
* `self_play_games_per_sec > 20`
* `distill_kl_drift < 0.1`

Also keep the evaluation harness fields unchanged:

* mean placement,
* stable dan,
* win rate,
* deal-in rate,
* tsumo rate.

Do **not** train on those. Use them only as outer acceptance metrics.

### 12. Acceptance criteria and rollout

#### 12.1 Ship criteria

Recommended minimums:

1. `Capture(learned) >= 1.05 * Capture(current_heuristic)`
2. `Capture(learned) >= 1.02 * Capture(heuristic_plus_reach)`
3. no benchmark gate regressions
4. no worse mean placement / deal-in on held-out self-play
5. scorer inference cost < 1 µs per task on target CPU

#### 12.2 Rollout stages

Stage 0: fix key semantics, freshness, dedup.

Stage 1: shadow mode.

* compute learned scores,
* keep scheduling with heuristic,
* log both rankings and realized outcomes.

Stage 2: tie-break mode.

* use learned score only among tasks whose hand score falls in the same bucket.

Stage 3: predicted-child-only mode.

* learned ordering only for speculative precompute,
* root deepening stays heuristic.

Stage 4: full ponder-queue mode.

* learned priority everywhere,
* fallback to heuristic on any parse/NaN/missing-model error.

This is the same deployment style that SmartChoices argues for: local inference, validation requirements, fallback default behavior, canary, rollback. HALP is another good example of shipping a heuristic+ML hybrid instead of replacing the heuristic cold turkey. ([arXiv][4])

### 13. Kill criteria

Kill it immediately if any of these are true:

1. **Speculative keys do not reconcile to serving keys.**
   Then predicted-child precompute cannot hit, so the scorer cannot help runtime.

2. **Heuristic + reach probability captures almost all the gain.**
   If `Capture(heuristic_plus_reach)` is within 1 percentage point of the learned scorer, ship the simpler rule and stop. SmartChoices explicitly treats “a simpler model is not better” as a required check. ([arXiv][4])

3. **No replay win.**
   After three held-out replay shards, learned capture is <2% better than heuristic+reach.

4. **System regression.**
   Any failure of the artifact gates, especially:

   * `self_play_games_per_sec <= 20`,
   * `afbs_on_turn_ms >= 150`,
   * `distill_kl_drift >= 0.1`.

5. **Bad serving hygiene.**

   * stale cache reuse rises,
   * queue duplicates dominate pops,
   * `cache_hit_on_realize` for speculative tasks does not improve.

6. **Degenerate scorer.**

   * almost all predicted priorities collapse into one bucket,
   * or the learned weights simply rediscover the old heuristic with no measurable gain.

### 14. Worked example

Three queued tasks:

* `A_root`: current root, `gap=0.04`, `risk=0.4`, `ESS=0.9`, `reach=1.0`
* `B_lowreach_hard`: speculative child, `gap=0.01`, `risk=0.8`, `ESS=0.2`, `reach=0.05`
* `C_midreach_easy`: speculative child, `gap=0.50`, `risk=0.1`, `ESS=0.95`, `reach=0.35`

Current hand heuristic:

[
h_{\text{hand}} = 10\max(0,0.1-gap)+risk+(1-ESS)
]

gives:

* `A_root = 1.10`
* `B_lowreach_hard = 2.50`
* `C_midreach_easy = 0.15`

So the current queue pops `B`.

Now define marginal hard-gain labels from a deeper reference:

* `A_root`: `y_hard ≈ 0.214991`
* `B_lowreach_hard`: `y_hard ≈ 0.216032`
* `C_midreach_easy`: `y_hard ≈ 0.020980`

For **label-harvest** ordering:

* `B ≈ A >> C`

For **runtime** ordering:
[
p_{\text{runtime}} = p_{\text{reach}} \cdot y_{\text{hard}}
]

so:

* `A_root = 1.0 * 0.214991 = 0.214991`
* `B_lowreach_hard = 0.05 * 0.216032 = 0.010802`
* `C_midreach_easy = 0.35 * 0.020980 = 0.007343`

So runtime should pop `A`, while label-harvest could still prefer `B`.

That is exactly the behavior you want:

* runtime compute goes to likely-to-matter states,
* label amplification can still mine intrinsically hard but low-reach states.

### 15. Why this exact blueprint, not something broader

ACT and PonderNet justify the general idea that harder inputs deserve more compute, but they solve a broader halting problem and bring tuning complexity you do not need at this seam. ([arXiv][1])

KataGo is the closer pattern for supervision: vary search effort, use deeper search as the target, and keep the training target tied to the stronger search rather than the raw exploratory search dynamics. ([ar5iv][3])

Learned branching is the closest analogy for the tiny scorer itself: imitate the marginal value of an expensive decision rule with a cheap model, and judge success by **overall throughput-adjusted utility**, not by matching the expensive expert perfectly on every state. ([arXiv][2])

SmartChoices and HALP are the closest deployment analogies: local tiny inference, strict fallback, metric constraints versus baseline, canary/rollback, and a bias toward augmenting heuristics instead of replacing them. ([arXiv][4])

---

That is the scorer to build: **a sign-constrained linear estimate of the intrinsic value of one fixed ponder quantum, multiplied by raw reach probability at runtime, with separate label-harvest use, hard fallback to the current heuristic, and explicit kill criteria if the simpler heuristic+reach baseline captures the same gain.**

[1]: https://arxiv.org/pdf/2107.05407.pdf "https://arxiv.org/pdf/2107.05407.pdf"
[2]: https://arxiv.org/pdf/2307.01434 "https://arxiv.org/pdf/2307.01434"
[3]: https://ar5iv.org/pdf/1902.10565 "https://ar5iv.org/pdf/1902.10565"
[4]: https://arxiv.org/pdf/2304.13033 "https://arxiv.org/pdf/2304.13033"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
