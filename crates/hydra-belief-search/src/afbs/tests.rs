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
fn public_node_accessors_are_checked() {
    let mut tree = AfbsTree::new();
    let root = tree.add_node(0, 1.0, false);
    assert!(tree.node(root).is_some());
    assert!(tree.node(NodeIdx::MAX).is_none());
    assert!(tree.node_mut(root).is_some());
    assert!(tree.node_mut(NodeIdx::MAX).is_none());
    assert_eq!(tree.puct_select(NodeIdx::MAX), None);
    assert_eq!(tree.best_action(NodeIdx::MAX), None);
    assert_eq!(tree.try_max_depth(NodeIdx::MAX), None);
}

#[test]
fn dangling_child_edges_do_not_panic_public_queries() {
    let mut tree = AfbsTree::new();
    let root = tree.add_node(0, 1.0, false);
    tree.nodes[root as usize].children = smallvec::smallvec![(3, NodeIdx::MAX)];
    assert_eq!(tree.puct_select(root), None);
    assert_eq!(tree.best_action(root), None);
    assert_eq!(tree.root_exit_policy(root, 1.0).iter().sum::<f32>(), 0.0);
    assert_eq!(tree.root_exit_policy(root, 0.0).iter().sum::<f32>(), 0.0);
    assert_eq!(tree.max_depth(root), 1);
}

#[test]
fn add_child_rejects_invalid_indices_and_actions() {
    let mut tree = AfbsTree::new();
    let root = tree.add_node(0, 1.0, false);
    let child = tree.add_node(1, 0.5, false);
    assert!(tree.add_child(root, 2, child));
    assert!(!tree.add_child(root, HYDRA_ACTION_SPACE as u8, child));
    assert!(!tree.add_child(root, 3, NodeIdx::MAX));
    assert!(!tree.add_child(NodeIdx::MAX, 3, child));
    assert_eq!(tree.child_actions(root), vec![2]);
}

#[test]
fn cyclic_child_edges_do_not_recurse_forever() {
    let mut tree = AfbsTree::new();
    let root = tree.add_node(0, 1.0, false);
    let child = tree.add_node(1, 0.5, false);
    tree.nodes[root as usize].children = smallvec::smallvec![(0, child)];
    tree.nodes[child as usize].children = smallvec::smallvec![(1, root)];

    assert_eq!(tree.try_max_depth(root), Some(2));
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
    let obs = [0.0f32; OBS_SIZE];
    for i in 0..32 {
        batch.add(&obs, i);
    }
    assert_eq!(batch.batch_size, 32);
    assert!(batch.is_ready());
    assert_eq!(batch.node_indices.len(), 32);
    assert_eq!(batch.obs_buffer.len(), 32 * OBS_SIZE);
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

    let mut auth_result = PonderResult::learner_only_stub([0.0; HYDRA_ACTION_SPACE], 0.9, 8, 500);
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
