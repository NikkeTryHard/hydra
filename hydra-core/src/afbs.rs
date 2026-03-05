//! Anytime Factored-Belief Search (AFBS) with PUCT selection.

use crate::action::HYDRA_ACTION_SPACE;

pub const C_PUCT: f32 = 2.5;
pub const TOP_K: usize = 5;

pub fn has_any_legal_action(mask: &[bool; HYDRA_ACTION_SPACE]) -> bool {
    mask.iter().any(|&m| m)
}

pub fn legal_action_count(mask: &[bool; HYDRA_ACTION_SPACE]) -> usize {
    mask.iter().filter(|&&m| m).count()
}

pub type NodeIdx = u32;

pub struct AfbsNode {
    pub info_state_hash: u64,
    pub visit_count: u32,
    pub total_value: f64,
    pub prior: f32,
    pub children: Vec<(u8, NodeIdx)>,
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
            children: Vec::new(),
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
        let mut scored: Vec<(u8, f32)> = (0..HYDRA_ACTION_SPACE as u8)
            .filter(|&a| legal_mask[a as usize])
            .map(|a| (a, policy_logits[a as usize]))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(TOP_K);

        let max_logit = scored.first().map(|s| s.1).unwrap_or(0.0);
        let exp_sum: f32 = scored.iter().map(|s| (s.1 - max_logit).exp()).sum();

        let parent_hash = self.nodes[parent_idx as usize].info_state_hash;
        for (action, logit) in &scored {
            let prior = (logit - max_logit).exp() / exp_sum;
            let child_hash = parent_hash ^ (*action as u64).wrapping_mul(0x9e3779b97f4a7c15);
            let child_idx = self.add_node(child_hash, prior, is_opponent);
            self.nodes[parent_idx as usize]
                .children
                .push((*action, child_idx));
        }
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
        for _ in 0..num_iters {
            if let Some((_, child_idx)) = self.puct_select(root_idx) {
                let value = eval_fn(child_idx);
                self.backpropagate(&[root_idx, child_idx], value);
            }
        }
    }

    pub fn root_exit_policy(&self, root_idx: NodeIdx, tau: f32) -> [f32; HYDRA_ACTION_SPACE] {
        let root = &self.nodes[root_idx as usize];
        let mut policy = [0.0f32; HYDRA_ACTION_SPACE];
        if root.children.is_empty() {
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

    pub fn clear(&mut self) {
        self.nodes.clear();
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

pub struct PonderResult {
    pub exit_policy: [f32; HYDRA_ACTION_SPACE],
    pub value: f32,
    pub search_depth: u8,
    pub visit_count: u32,
    pub timestamp_ns: u64,
}

pub struct PonderTask {
    pub info_state_hash: u64,
    pub priority_score: f32,
}

pub struct PonderCache {
    entries: std::collections::HashMap<u64, PonderResult>,
}

impl PonderCache {
    pub fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }

    pub fn get(&self, hash: u64) -> Option<&PonderResult> {
        self.entries.get(&hash)
    }

    pub fn insert(&mut self, hash: u64, result: PonderResult) {
        self.entries.insert(hash, result);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn remove(&mut self, hash: u64) -> Option<PonderResult> {
        self.entries.remove(&hash)
    }

    pub fn contains(&self, hash: u64) -> bool {
        self.entries.contains_key(&hash)
    }

    pub fn clear(&mut self) {
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

pub const MIN_BATCH: usize = 32;

pub struct LeafBatch {
    pub obs_buffer: Vec<f32>,
    pub node_indices: Vec<NodeIdx>,
    pub batch_size: usize,
}

impl LeafBatch {
    pub fn new() -> Self {
        Self {
            obs_buffer: Vec::new(),
            node_indices: Vec::new(),
            batch_size: 0,
        }
    }

    pub fn clear(&mut self) {
        self.obs_buffer.clear();
        self.node_indices.clear();
        self.batch_size = 0;
    }

    pub fn add(&mut self, obs: &[f32], node_idx: NodeIdx) {
        self.obs_buffer.extend_from_slice(obs);
        self.node_indices.push(node_idx);
        self.batch_size += 1;
    }

    pub fn is_ready(&self) -> bool {
        self.batch_size >= MIN_BATCH
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
        tree.nodes[root as usize].children = vec![(0, c1), (1, c2)];
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
        for i in 0..10 {
            mask[i] = true;
        }
        tree.expand_node(root, &logits, &mask, false);
        assert_eq!(tree.nodes[root as usize].children.len(), TOP_K);
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
    fn batched_eval_correct_size() {
        let mut batch = LeafBatch::new();
        let obs = [0.0f32; 85 * 34];
        for i in 0..32 {
            batch.add(&obs, i);
        }
        assert_eq!(batch.batch_size, 32);
        assert!(batch.is_ready());
        assert_eq!(batch.node_indices.len(), 32);
        assert_eq!(batch.obs_buffer.len(), 32 * 85 * 34);
    }

    #[test]
    fn ponder_cache_hit_reuses_search() {
        let mut cache = PonderCache::new();
        let result = PonderResult {
            exit_policy: [0.0; HYDRA_ACTION_SPACE],
            value: 0.5,
            search_depth: 4,
            visit_count: 100,
            timestamp_ns: 12345,
        };
        cache.insert(42, result);
        assert_eq!(cache.len(), 1);
        let hit = cache.get(42).expect("should find cached result");
        assert_eq!(hit.visit_count, 100);
        assert!((hit.value - 0.5).abs() < 1e-5);
        assert!(cache.get(99).is_none(), "miss should return None");
    }

    #[test]
    fn puct_balances_exploration_exploitation() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(0, 1.0, false);
        let c0 = tree.add_node(1, 0.5, false);
        let c1 = tree.add_node(2, 0.5, false);
        tree.nodes[root as usize].children = vec![(0, c0), (1, c1)];
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
        tree.nodes[root as usize].children = vec![(0, c0), (5, c1), (10, c2)];
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
    fn has_any_legal_action_checks() {
        let empty = [false; HYDRA_ACTION_SPACE];
        assert!(!has_any_legal_action(&empty));
        let mut one = [false; HYDRA_ACTION_SPACE];
        one[45] = true;
        assert!(has_any_legal_action(&one));
    }
}
