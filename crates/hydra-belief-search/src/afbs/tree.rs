use hydra_runtime_types::action::HYDRA_ACTION_SPACE;
use smallvec::SmallVec;
use std::cmp::Ordering;

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

    /// Returns a node by index without panicking on stale or foreign indices.
    pub fn node(&self, idx: NodeIdx) -> Option<&AfbsNode> {
        self.nodes.get(idx as usize)
    }

    /// Returns a mutable node by index without panicking on stale or foreign indices.
    pub fn node_mut(&mut self, idx: NodeIdx) -> Option<&mut AfbsNode> {
        self.nodes.get_mut(idx as usize)
    }

    /// Adds a child edge if both parent and child indices are valid.
    pub fn add_child(&mut self, parent_idx: NodeIdx, action: u8, child_idx: NodeIdx) -> bool {
        if action as usize >= HYDRA_ACTION_SPACE {
            return false;
        }
        if self.node(child_idx).is_none() {
            return false;
        }
        let Some(parent) = self.node_mut(parent_idx) else {
            return false;
        };
        parent.children.push((action, child_idx));
        true
    }

    pub fn puct_select(&self, parent_idx: NodeIdx) -> Option<(u8, NodeIdx)> {
        let parent = self.node(parent_idx)?;
        if parent.children.is_empty() {
            return None;
        }
        let sqrt_n = (parent.visit_count as f32).sqrt();
        let mut best_ucb = f32::NEG_INFINITY;
        let mut best = None;
        for &(action, child_idx) in &parent.children {
            let Some(child) = self.node(child_idx) else {
                continue;
            };
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
            let Some(node) = self.node_mut(idx) else {
                continue;
            };
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
            if let Some((action, _)) = root
                .children
                .iter()
                .filter(|(_, idx)| self.node(*idx).is_some())
                .max_by(|(_, lhs), (_, rhs)| {
                    self.node(*lhs)
                        .map_or(0.0, AfbsNode::q_value)
                        .partial_cmp(&self.node(*rhs).map_or(0.0, AfbsNode::q_value))
                        .unwrap_or(Ordering::Equal)
                })
            {
                policy[*action as usize] = 1.0;
            }
            return policy;
        }

        let mut max_q = f32::NEG_INFINITY;
        for &(_, child_idx) in &root.children {
            let Some(child) = self.node(child_idx) else {
                continue;
            };
            let q = child.q_value();
            if q > max_q {
                max_q = q;
            }
        }
        if !max_q.is_finite() {
            return policy;
        }
        let mut total = 0.0f32;
        for &(action, child_idx) in &root.children {
            let Some(child) = self.node(child_idx) else {
                continue;
            };
            let q = child.q_value();
            let exp_q = ((q - max_q) / tau).exp();
            if exp_q.is_finite() && exp_q > 0.0 {
                policy[action as usize] = exp_q;
                total += exp_q;
            }
        }
        if total > 0.0 && total.is_finite() {
            for p in &mut policy {
                *p /= total;
            }
        }
        policy
    }

    pub fn best_action(&self, root_idx: NodeIdx) -> Option<u8> {
        let root = self.node(root_idx)?;
        root.children
            .iter()
            .filter_map(|(action, idx)| self.node(*idx).map(|node| (*action, node.visit_count)))
            .max_by_key(|(_, visits)| *visits)
            .map(|(action, _)| action)
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
        self.try_max_depth(root).unwrap_or(0)
    }

    pub fn try_max_depth(&self, root: NodeIdx) -> Option<u8> {
        let mut visiting = vec![false; self.nodes.len()];
        self.try_max_depth_inner(root, &mut visiting)
    }

    fn try_max_depth_inner(&self, root: NodeIdx, visiting: &mut [bool]) -> Option<u8> {
        let root_idx = root as usize;
        let node = self.nodes.get(root_idx)?;
        if visiting[root_idx] {
            return Some(0);
        }
        if node.children.is_empty() {
            return Some(0);
        }
        visiting[root_idx] = true;
        let mut max_d = 0u8;
        for &(_, child) in &node.children {
            let Some(d) = self.try_max_depth_inner(child, visiting) else {
                continue;
            };
            if d > max_d {
                max_d = d;
            }
        }
        visiting[root_idx] = false;
        Some(max_d.saturating_add(1))
    }
}

impl Default for AfbsTree {
    fn default() -> Self {
        Self::new()
    }
}
