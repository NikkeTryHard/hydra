//! ExIt pipeline: search target generation and safety valve.

use burn::prelude::*;
use burn::tensor::activation;
use hydra_core::action::{AKA_5M, AKA_5P, AKA_5S, DISCARD_END, HYDRA_ACTION_SPACE};
use hydra_core::afbs::{AfbsTree, NodeIdx};

pub const MIN_EXIT_CHILD_VISITS: u32 = 2;
pub const MIN_EXIT_AVG_ROOT_VISITS_PER_LEGAL_DISCARD: f32 = 8.0;
pub const MIN_EXIT_COVERAGE: f32 = 0.60;

#[derive(Config, Debug)]
pub struct ExitConfig {
    #[config(default = "1.0")]
    pub tau_exit: f32,
    #[config(default = "0.5")]
    pub exit_weight: f32,
    #[config(default = "64")]
    pub min_visits: u32,
    #[config(default = "0.1")]
    pub hard_state_threshold: f32,
    #[config(default = "2.0")]
    pub safety_valve_max_kl: f32,
}

impl ExitConfig {
    pub fn summary(&self) -> String {
        format!(
            "exit(tau={:.1}, w={:.1}, visits>={}, kl<{:.1})",
            self.tau_exit, self.exit_weight, self.min_visits, self.safety_valve_max_kl
        )
    }

    pub fn default_phase3() -> Self {
        Self::new()
    }
    pub fn min_visits_reached(&self, visit_count: u32) -> bool {
        visit_count >= self.min_visits
    }

    pub fn effective_weight(&self, phase: u8, progress: f32) -> f32 {
        anneal_exit_weight(self.exit_weight, phase, progress)
    }

    pub fn should_apply_exit(&self, visit_count: u32) -> bool {
        visit_count >= self.min_visits
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.tau_exit <= 0.0 {
            return Err("tau_exit must be positive");
        }
        if self.exit_weight < 0.0 {
            return Err("exit_weight must be non-negative");
        }
        if self.safety_valve_max_kl <= 0.0 {
            return Err("max_kl must be positive");
        }
        Ok(())
    }
}

pub fn anneal_exit_weight(base_weight: f32, phase: u8, progress: f32) -> f32 {
    match phase {
        0 | 1 => 0.0,
        2 => {
            let progress = progress.clamp(0.0, 1.0);
            if progress <= 0.5 {
                0.0
            } else {
                base_weight * ((progress - 0.5) / 0.5)
            }
        }
        _ => base_weight,
    }
}

pub fn is_hard_state(policy: &[f32], threshold: f32) -> bool {
    if policy.len() < 2 {
        return false;
    }
    let mut sorted: Vec<f32> = policy.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted[0] - sorted[1] < threshold
}

pub fn exit_policy_from_q(q_values: &[f32], tau: f32, legal_mask: Option<&[bool]>) -> Vec<f32> {
    let n = q_values.len();
    let max_q = q_values
        .iter()
        .enumerate()
        .filter(|(i, _)| legal_mask.is_none_or(|m| *i < m.len() && m[*i]))
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; n];
    let mut total = 0.0f32;
    for i in 0..n {
        let is_legal = legal_mask.is_none_or(|m| i < m.len() && m[i]);
        if is_legal {
            probs[i] = ((q_values[i] - max_q) / tau).exp();
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

pub fn exit_loss<B: Backend>(
    model_logits: Tensor<B, 2>,
    exit_target: Tensor<B, 2>,
    exit_mask: Tensor<B, 2>,
    weight: f32,
) -> Tensor<B, 1> {
    let neg_inf = (exit_mask.ones_like() - exit_mask) * (-1e9f32);
    let log_pi = activation::log_softmax(model_logits + neg_inf, 1);
    let ce = (exit_target * log_pi).sum_dim(1).neg().mean();
    ce * weight
}

pub fn make_exit_target(
    q_values: &[f32],
    base_pi: &[f32],
    tau_exit: f32,
    min_visits: u32,
    visit_count: u32,
    max_kl: f32,
    legal_mask: Option<&[bool]>,
) -> Option<Vec<f32>> {
    let exit_pi = exit_policy_from_q(q_values, tau_exit, legal_mask);
    if safety_valve_check(base_pi, &exit_pi, max_kl, min_visits, visit_count) {
        Some(exit_pi)
    } else {
        None
    }
}

pub fn compatible_discard_state(legal_mask: &[f32]) -> bool {
    if legal_mask.len() != HYDRA_ACTION_SPACE {
        return false;
    }
    let non_discard_legal = legal_mask[(DISCARD_END as usize + 1)..]
        .iter()
        .any(|&x| x > 0.0);
    if non_discard_legal {
        return false;
    }
    let aka_legal = legal_mask[AKA_5M as usize] > 0.0
        || legal_mask[AKA_5P as usize] > 0.0
        || legal_mask[AKA_5S as usize] > 0.0;
    !aka_legal
}

pub fn make_exit_target_from_child_visits(
    base_pi: &[f32],
    legal_mask: &[f32],
    child_visits: &[(u8, u32)],
    min_visits: u32,
    root_visit_count: u32,
    max_kl: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if base_pi.len() != HYDRA_ACTION_SPACE || legal_mask.len() != HYDRA_ACTION_SPACE {
        return None;
    }
    if !compatible_discard_state(legal_mask) || root_visit_count < min_visits {
        return None;
    }

    let legal_discard_actions: Vec<usize> = legal_mask[..=DISCARD_END as usize]
        .iter()
        .enumerate()
        .filter_map(|(idx, &is_legal)| (is_legal > 0.0).then_some(idx))
        .collect();
    if legal_discard_actions.len() < 2 {
        return None;
    }

    let avg_root_visits_per_legal_discard =
        root_visit_count as f32 / legal_discard_actions.len() as f32;
    if avg_root_visits_per_legal_discard < MIN_EXIT_AVG_ROOT_VISITS_PER_LEGAL_DISCARD {
        return None;
    }

    let mut target = vec![0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = vec![0.0f32; HYDRA_ACTION_SPACE];
    let mut covered_actions = 0usize;
    let mut covered_visit_total = 0u32;

    for &(action, visits) in child_visits {
        let idx = action as usize;
        if idx > DISCARD_END as usize || idx >= HYDRA_ACTION_SPACE {
            continue;
        }
        if legal_mask[idx] <= 0.0 || visits < MIN_EXIT_CHILD_VISITS {
            continue;
        }
        target[idx] = visits as f32;
        mask[idx] = 1.0;
        covered_actions += 1;
        covered_visit_total += visits;
    }

    if covered_visit_total == 0 {
        return None;
    }

    let exit_coverage = covered_actions as f32 / legal_discard_actions.len() as f32;
    if exit_coverage < MIN_EXIT_COVERAGE {
        return None;
    }

    let covered_visit_total = covered_visit_total as f32;
    for idx in 0..HYDRA_ACTION_SPACE {
        if mask[idx] > 0.0 {
            target[idx] /= covered_visit_total;
        }
    }

    if safety_valve_check(base_pi, &target, max_kl, min_visits, root_visit_count) {
        Some((target, mask))
    } else {
        None
    }
}

/// Bridges AfbsTree search results to exit target production.
///
/// Extracts child visit counts from the given tree root, then delegates
/// to [`make_exit_target_from_child_visits`] with all gating checks.
/// Returns `None` if the tree root is missing, if the state is not a
/// compatible discard state, or if any gating threshold is not met.
///
/// This is the canonical producer-side entry point for wiring AFBS search
/// output into RL training batches.
pub fn build_exit_from_afbs_tree(
    tree: &AfbsTree,
    root_idx: NodeIdx,
    base_pi: &[f32],
    legal_mask: &[f32],
    min_visits: u32,
    max_kl: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let root = tree.nodes.get(root_idx as usize)?;
    let root_visit_count = root.visit_count;
    let child_visits: Vec<(u8, u32)> = root
        .children
        .iter()
        .map(|&(action, child_idx)| (action, tree.nodes[child_idx as usize].visit_count))
        .collect();
    make_exit_target_from_child_visits(
        base_pi,
        legal_mask,
        &child_visits,
        min_visits,
        root_visit_count,
        max_kl,
    )
}

pub fn build_delta_q_from_afbs_tree(
    tree: &AfbsTree,
    root_idx: NodeIdx,
    legal_mask: &[f32],
) -> Option<(Vec<f32>, Vec<f32>)> {
    if legal_mask.len() != HYDRA_ACTION_SPACE || !compatible_discard_state(legal_mask) {
        return None;
    }
    let root = tree.nodes.get(root_idx as usize)?;
    if root.visit_count == 0 {
        return None;
    }
    let root_q = tree.node_q_value(root_idx);
    if !root_q.is_finite() {
        return None;
    }

    let mut target = vec![0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = vec![0.0f32; HYDRA_ACTION_SPACE];
    let mut any_supported = false;

    for &(action, child_idx) in &root.children {
        let idx = action as usize;
        if idx > DISCARD_END as usize || idx >= HYDRA_ACTION_SPACE || legal_mask[idx] <= 0.0 {
            continue;
        }
        let child = tree.nodes.get(child_idx as usize)?;
        if child.visit_count == 0 {
            continue;
        }
        let child_q = tree.node_q_value(child_idx);
        if !child_q.is_finite() {
            return None;
        }
        target[idx] = child_q - root_q;
        mask[idx] = 1.0;
        any_supported = true;
    }

    any_supported.then_some((target, mask))
}

/// Collates per-sample exit targets into batch tensors for `RlBatch`.
///
/// Takes a slice of per-sample results from [`build_exit_from_afbs_tree`]
/// or [`make_exit_target_from_child_visits`]. If all samples are `None`,
/// returns `(None, None)`. Otherwise, samples without targets get zero
/// target and zero mask rows, ensuring per-sample masking in the loss.
pub fn collate_exit_targets<B: Backend>(
    samples: &[Option<(Vec<f32>, Vec<f32>)>],
    device: &B::Device,
) -> (Option<Tensor<B, 2>>, Option<Tensor<B, 2>>) {
    if samples.is_empty() || samples.iter().all(|s| s.is_none()) {
        return (None, None);
    }
    let batch = samples.len();
    let mut target_data = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    let mut mask_data = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    for (i, sample) in samples.iter().enumerate() {
        if let Some((target, mask)) = sample {
            let offset = i * HYDRA_ACTION_SPACE;
            target_data[offset..offset + HYDRA_ACTION_SPACE].copy_from_slice(target);
            mask_data[offset..offset + HYDRA_ACTION_SPACE].copy_from_slice(mask);
        }
    }
    let target_tensor = Tensor::<B, 1>::from_floats(target_data.as_slice(), device)
        .reshape([batch, HYDRA_ACTION_SPACE]);
    let mask_tensor = Tensor::<B, 1>::from_floats(mask_data.as_slice(), device)
        .reshape([batch, HYDRA_ACTION_SPACE]);
    (Some(target_tensor), Some(mask_tensor))
}

pub fn collate_delta_q_targets<B: Backend>(
    samples: &[Option<(Vec<f32>, Vec<f32>)>],
    device: &B::Device,
) -> (Option<Tensor<B, 2>>, Option<Tensor<B, 2>>) {
    if samples.is_empty() || samples.iter().all(|s| s.is_none()) {
        return (None, None);
    }
    let batch = samples.len();
    let mut target_data = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    let mut mask_data = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    for (i, sample) in samples.iter().enumerate() {
        if let Some((target, mask)) = sample {
            let offset = i * HYDRA_ACTION_SPACE;
            target_data[offset..offset + HYDRA_ACTION_SPACE].copy_from_slice(target);
            mask_data[offset..offset + HYDRA_ACTION_SPACE].copy_from_slice(mask);
        }
    }
    let target_tensor = Tensor::<B, 1>::from_floats(target_data.as_slice(), device)
        .reshape([batch, HYDRA_ACTION_SPACE]);
    let mask_tensor = Tensor::<B, 1>::from_floats(mask_data.as_slice(), device)
        .reshape([batch, HYDRA_ACTION_SPACE]);
    (Some(target_tensor), Some(mask_tensor))
}

pub fn safety_valve_check(
    base_pi: &[f32],
    exit_pi: &[f32],
    max_kl: f32,
    min_visits: u32,
    visit_count: u32,
) -> bool {
    if visit_count < min_visits {
        return false;
    }
    let mut kl = 0.0f32;
    for i in 0..base_pi.len() {
        if exit_pi[i] > 1e-10 && base_pi[i] > 1e-10 {
            kl += exit_pi[i] * (exit_pi[i] / base_pi[i]).ln();
        }
    }
    kl <= max_kl
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use hydra_core::afbs::predicted_child_hash;

    type B = NdArray<f32>;

    #[test]
    fn exit_safety_valve_skips_low_visits() {
        let base = vec![0.5, 0.3, 0.2];
        let exit = vec![0.4, 0.4, 0.2];
        assert!(!safety_valve_check(&base, &exit, 2.0, 64, 10));
    }

    #[test]
    fn exit_safety_valve_passes_good_target() {
        let p = vec![0.5, 0.3, 0.2];
        assert!(safety_valve_check(&p, &p, 2.0, 64, 100));
    }

    #[test]
    fn exit_policy_sums_to_one() {
        let q = vec![1.0, 2.0, 0.5, 3.0];
        let pi = exit_policy_from_q(&q, 1.0, None);
        let sum: f32 = pi.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "should sum to 1, got {sum}");
    }

    #[test]
    fn exit_policy_from_q_with_mask() {
        let q = vec![1.0, 2.0, 0.5, 3.0];
        let mask = vec![true, false, true, false];
        let pi = exit_policy_from_q(&q, 1.0, Some(&mask));
        assert!(
            pi[1].abs() < 1e-10,
            "illegal action 1 should get 0.0 prob, got {}",
            pi[1]
        );
        assert!(
            pi[3].abs() < 1e-10,
            "illegal action 3 should get 0.0 prob, got {}",
            pi[3]
        );
        let legal_sum: f32 = pi[0] + pi[2];
        assert!(
            (legal_sum - 1.0).abs() < 1e-5,
            "legal actions should sum to 1, got {legal_sum}"
        );
    }

    #[test]
    fn exit_safety_valve_rejects_high_kl() {
        let base = vec![0.9, 0.05, 0.05];
        let exit = vec![0.05, 0.05, 0.9];
        assert!(!safety_valve_check(&base, &exit, 0.5, 64, 100));
    }

    #[test]
    fn compatible_discard_state_rejects_non_discard_or_aka_actions() {
        let mut legal = vec![0.0f32; HYDRA_ACTION_SPACE];
        legal[1] = 1.0;
        legal[2] = 1.0;
        assert!(compatible_discard_state(&legal));

        legal[AKA_5M as usize] = 1.0;
        assert!(!compatible_discard_state(&legal));

        legal[AKA_5M as usize] = 0.0;
        legal[(DISCARD_END as usize) + 1] = 1.0;
        assert!(!compatible_discard_state(&legal));
    }

    #[test]
    fn child_visit_exit_target_accepts_masked_distribution() {
        let mut base = vec![1e-6f32; HYDRA_ACTION_SPACE];
        base[1] = 0.4;
        base[2] = 0.35;
        base[5] = 0.25;
        let mut legal = vec![0.0f32; HYDRA_ACTION_SPACE];
        legal[1] = 1.0;
        legal[2] = 1.0;
        legal[5] = 1.0;
        let child_visits = vec![(1, 8), (2, 5), (5, 3)];

        let (target, mask) =
            make_exit_target_from_child_visits(&base, &legal, &child_visits, 8, 24, 2.0)
                .expect("accepted target");
        let sum: f32 = target.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((target[1] - 0.5).abs() < 1e-6);
        assert!((target[2] - 0.3125).abs() < 1e-6);
        assert!((target[5] - 0.1875).abs() < 1e-6);
        assert_eq!(mask[1], 1.0);
        assert_eq!(mask[2], 1.0);
        assert_eq!(mask[5], 1.0);
    }

    #[test]
    fn child_visit_exit_target_rejects_low_visits_or_coverage() {
        let mut base = vec![1e-6f32; HYDRA_ACTION_SPACE];
        base[1] = 0.4;
        base[2] = 0.35;
        base[5] = 0.25;
        let mut legal = vec![0.0f32; HYDRA_ACTION_SPACE];
        legal[1] = 1.0;
        legal[2] = 1.0;
        legal[5] = 1.0;

        assert!(
            make_exit_target_from_child_visits(&base, &legal, &[(1, 8), (2, 5)], 8, 12, 2.0)
                .is_none()
        );
        assert!(make_exit_target_from_child_visits(&base, &legal, &[(1, 8)], 8, 24, 2.0).is_none());
    }

    #[test]
    fn exit_policy_concentrates_on_best() {
        let q = vec![10.0, 0.0, 0.0, 0.0];
        let pi = exit_policy_from_q(&q, 0.1, None);
        assert!(
            pi[0] > 0.99,
            "low tau should concentrate on best action: {}",
            pi[0]
        );
    }

    #[test]
    fn anneal_exit_weight_phases() {
        assert!((anneal_exit_weight(0.5, 0, 0.5) - 0.0).abs() < 1e-6);
        assert!((anneal_exit_weight(0.5, 1, 0.5) - 0.0).abs() < 1e-6);
        assert!((anneal_exit_weight(0.5, 2, 0.5) - 0.0).abs() < 1e-6);
        assert!((anneal_exit_weight(0.5, 2, 0.75) - 0.25).abs() < 1e-6);
        assert!((anneal_exit_weight(0.5, 2, 1.0) - 0.5).abs() < 1e-6);
        assert!((anneal_exit_weight(0.5, 3, 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn default_phase3_matches_roadmap_defaults() {
        let cfg = ExitConfig::default_phase3();
        assert!((cfg.exit_weight - 0.5).abs() < 1e-6);
        assert_eq!(cfg.min_visits, 64);
    }

    #[test]
    fn is_hard_state_close_gap() {
        assert!(is_hard_state(&[0.45, 0.44, 0.11], 0.1));
        assert!(!is_hard_state(&[0.8, 0.1, 0.1], 0.1));
        assert!(!is_hard_state(&[1.0], 0.1));
    }

    fn make_test_tree() -> (AfbsTree, NodeIdx) {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(7, 1.0, false);
        let mut legal_mask = [false; HYDRA_ACTION_SPACE];
        legal_mask[1] = true;
        legal_mask[2] = true;
        legal_mask[5] = true;
        let mut policy_logits = [0.0f32; HYDRA_ACTION_SPACE];
        policy_logits[1] = 3.0;
        policy_logits[2] = 2.0;
        policy_logits[5] = 1.0;
        tree.expand_node(root, &policy_logits, &legal_mask, false);
        let children = tree.nodes[root as usize].children.clone();
        for &(action, child) in &children {
            let node = &mut tree.nodes[child as usize];
            match action {
                1 => {
                    node.visit_count = 10;
                    node.total_value = 9.0;
                }
                2 => {
                    node.visit_count = 8;
                    node.total_value = 4.0;
                }
                5 => {
                    node.visit_count = 6;
                    node.total_value = 0.6;
                }
                _ => unreachable!(),
            }
        }
        tree.nodes[root as usize].visit_count = 24;
        (tree, root)
    }

    #[test]
    fn build_exit_from_tree_accepts_good_search() {
        let (tree, root) = make_test_tree();
        let mut base_pi = vec![1e-6f32; HYDRA_ACTION_SPACE];
        base_pi[1] = 0.45;
        base_pi[2] = 0.35;
        base_pi[5] = 0.20;
        let mut legal = vec![0.0f32; HYDRA_ACTION_SPACE];
        legal[1] = 1.0;
        legal[2] = 1.0;
        legal[5] = 1.0;

        let (target, mask) = build_exit_from_afbs_tree(&tree, root, &base_pi, &legal, 8, 5.0)
            .expect("should accept well-visited tree");
        let sum: f32 = target.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "target should sum to 1, got {sum}"
        );
        assert_eq!(mask[1], 1.0);
        assert_eq!(mask[2], 1.0);
        assert_eq!(mask[5], 1.0);
    }

    #[test]
    fn build_exit_from_tree_rejects_insufficient_visits() {
        let (tree, root) = make_test_tree();
        let base_pi = vec![1e-6f32; HYDRA_ACTION_SPACE];
        let mut legal = vec![0.0f32; HYDRA_ACTION_SPACE];
        legal[1] = 1.0;
        legal[2] = 1.0;
        legal[5] = 1.0;

        assert!(
            build_exit_from_afbs_tree(&tree, root, &base_pi, &legal, 100, 5.0).is_none(),
            "should reject when min_visits > root visit count"
        );
    }

    #[test]
    fn build_exit_from_tree_rejects_invalid_root() {
        let tree = AfbsTree::new();
        let base_pi = vec![1e-6f32; HYDRA_ACTION_SPACE];
        let legal = vec![0.0f32; HYDRA_ACTION_SPACE];
        assert!(build_exit_from_afbs_tree(&tree, 999, &base_pi, &legal, 8, 5.0).is_none());
    }

    #[test]
    fn build_delta_q_from_tree_uses_root_child_q_delta() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(7, 1.0, false);
        let c1 = tree.add_node(predicted_child_hash(7, 1), 0.45, false);
        let c2 = tree.add_node(predicted_child_hash(7, 2), 0.35, false);
        tree.nodes[root as usize].children.push((1, c1));
        tree.nodes[root as usize].children.push((2, c2));
        tree.nodes[root as usize].visit_count = 10;
        tree.nodes[root as usize].total_value = 4.0;
        tree.nodes[c1 as usize].visit_count = 4;
        tree.nodes[c1 as usize].total_value = 3.2;
        tree.nodes[c2 as usize].visit_count = 4;
        tree.nodes[c2 as usize].total_value = 0.4;

        let mut legal = [0.0f32; HYDRA_ACTION_SPACE];
        legal[1] = 1.0;
        legal[2] = 1.0;
        let (target, mask) =
            build_delta_q_from_afbs_tree(&tree, root, &legal).expect("delta_q target");
        assert!((target[1] - 0.4).abs() < 1e-6);
        assert!((target[2] + 0.3).abs() < 1e-6);
        assert_eq!(mask[1], 1.0);
        assert_eq!(mask[2], 1.0);
        assert_eq!(mask[3], 0.0);
    }

    #[test]
    fn build_delta_q_from_tree_rejects_empty_support() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(7, 1.0, false);
        let child = tree.add_node(predicted_child_hash(7, 1), 1.0, false);
        tree.nodes[root as usize].children.push((1, child));
        tree.nodes[root as usize].visit_count = 10;
        tree.nodes[root as usize].total_value = 4.0;
        tree.nodes[child as usize].visit_count = 0;

        let mut legal = [0.0f32; HYDRA_ACTION_SPACE];
        legal[1] = 1.0;
        assert!(build_delta_q_from_afbs_tree(&tree, root, &legal).is_none());
    }

    #[test]
    fn collate_delta_q_targets_mixed_batch() {
        let device = Default::default();
        let mut target = vec![0.0f32; HYDRA_ACTION_SPACE];
        let mut mask = vec![0.0f32; HYDRA_ACTION_SPACE];
        target[1] = 0.4;
        target[2] = -0.3;
        mask[1] = 1.0;
        mask[2] = 1.0;
        let samples = vec![Some((target, mask)), None];
        let (target, mask) = collate_delta_q_targets::<B>(&samples, &device);
        let target = target.expect("target");
        let mask = mask.expect("mask");
        assert_eq!(target.dims(), [2, HYDRA_ACTION_SPACE]);
        assert_eq!(mask.dims(), [2, HYDRA_ACTION_SPACE]);
        let target_data = target.to_data().as_slice::<f32>().expect("f32").to_vec();
        let mask_data = mask.to_data().as_slice::<f32>().expect("f32").to_vec();
        assert!((target_data[1] - 0.4).abs() < 1e-6);
        assert!((target_data[2] + 0.3).abs() < 1e-6);
        assert_eq!(mask_data[1], 1.0);
        assert_eq!(mask_data[2], 1.0);
        assert_eq!(mask_data[HYDRA_ACTION_SPACE + 1], 0.0);
    }

    #[test]
    fn collate_exit_targets_all_none_returns_none() {
        use burn::backend::NdArray;
        type B = NdArray<f32>;
        let device = Default::default();
        let samples: Vec<Option<(Vec<f32>, Vec<f32>)>> = vec![None, None, None];
        let (target, mask) = collate_exit_targets::<B>(&samples, &device);
        assert!(target.is_none());
        assert!(mask.is_none());
    }

    #[test]
    fn collate_exit_targets_empty_returns_none() {
        use burn::backend::NdArray;
        type B = NdArray<f32>;
        let device = Default::default();
        let samples: Vec<Option<(Vec<f32>, Vec<f32>)>> = vec![];
        let (target, mask) = collate_exit_targets::<B>(&samples, &device);
        assert!(target.is_none());
        assert!(mask.is_none());
    }

    #[test]
    fn collate_exit_targets_mixed_batch() {
        use burn::backend::NdArray;
        type B = NdArray<f32>;
        let device = Default::default();

        let mut t1 = vec![0.0f32; HYDRA_ACTION_SPACE];
        t1[1] = 0.6;
        t1[2] = 0.4;
        let mut m1 = vec![0.0f32; HYDRA_ACTION_SPACE];
        m1[1] = 1.0;
        m1[2] = 1.0;

        let samples = vec![Some((t1.clone(), m1.clone())), None, Some((t1, m1))];
        let (target, mask) = collate_exit_targets::<B>(&samples, &device);
        let target = target.expect("should be Some when any sample has exit target");
        let mask = mask.expect("should be Some when any sample has exit mask");

        assert_eq!(target.dims(), [3, HYDRA_ACTION_SPACE]);
        assert_eq!(mask.dims(), [3, HYDRA_ACTION_SPACE]);

        let target_data = target.to_data();
        let target_slice = target_data.as_slice::<f32>().unwrap();
        assert!((target_slice[1] - 0.6).abs() < 1e-6, "sample 0 action 1");
        assert!((target_slice[2] - 0.4).abs() < 1e-6, "sample 0 action 2");

        let row2_offset = HYDRA_ACTION_SPACE;
        assert!(
            target_slice[row2_offset..row2_offset + HYDRA_ACTION_SPACE]
                .iter()
                .all(|&v| v == 0.0),
            "sample 1 (None) should be all zeros"
        );

        let mask_data = mask.to_data();
        let mask_slice = mask_data.as_slice::<f32>().unwrap();
        let mask_row2 = &mask_slice[row2_offset..row2_offset + HYDRA_ACTION_SPACE];
        assert!(
            mask_row2.iter().all(|&v| v == 0.0),
            "sample 1 mask should be all zeros"
        );
    }
}
