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
        make_exit_target_from_child_visits(&base, &legal, &[(1, 8), (2, 5)], 8, 12, 2.0).is_none()
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
fn default_live_exit_matches_roadmap_defaults() {
    let cfg = ExitConfig::default_live_exit();
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
    let (target, mask) = build_delta_q_from_afbs_tree(&tree, root, &legal).expect("delta_q target");
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
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
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
    let samples: Vec<Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])>> =
        vec![None, None, None];
    let (target, mask) = collate_exit_targets::<B>(&samples, &device);
    assert!(target.is_none());
    assert!(mask.is_none());
}

#[test]
fn collate_exit_targets_empty_returns_none() {
    use burn::backend::NdArray;
    type B = NdArray<f32>;
    let device = Default::default();
    let samples: Vec<Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])>> = vec![];
    let (target, mask) = collate_exit_targets::<B>(&samples, &device);
    assert!(target.is_none());
    assert!(mask.is_none());
}

#[test]
fn collate_exit_targets_mixed_batch() {
    use burn::backend::NdArray;
    type B = NdArray<f32>;
    let device = Default::default();

    let mut t1 = [0.0f32; HYDRA_ACTION_SPACE];
    t1[1] = 0.6;
    t1[2] = 0.4;
    let mut m1 = [0.0f32; HYDRA_ACTION_SPACE];
    m1[1] = 1.0;
    m1[2] = 1.0;

    let samples = vec![Some((t1, m1)), None, Some((t1, m1))];
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
