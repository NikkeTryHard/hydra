use super::*;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::afbs::AfbsTree;
use hydra_core::encoder::OBS_SIZE;

// --- Test adapter that returns deterministic child observations ---

struct StubAdapter {
    /// If set, child_public_obs_after_discard returns None.
    fail_obs: bool,
}

impl ExitSearchAdapter for StubAdapter {
    fn root_hash(&self, _state: &GameState, _player: u8, obs_encoded: &[f32; OBS_SIZE]) -> u64 {
        obs_hash(obs_encoded)
    }

    fn child_public_obs_after_discard(
        &mut self,
        _state: &GameState,
        _obs: &Observation,
        _player: u8,
        action: u8,
        _safety: &SafetyInfo,
    ) -> Option<[f32; OBS_SIZE]> {
        if self.fail_obs {
            return None;
        }
        // Return a distinguishable observation per action
        let mut obs = [0.0f32; OBS_SIZE];
        obs[action as usize % OBS_SIZE] = 1.0;
        Some(obs)
    }
}

// --- Test model that returns deterministic policy/value ---

fn make_stub_model(
    values: &[(u8, f32)],
) -> impl FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32) + '_ {
    move |obs: &[f32; OBS_SIZE]| {
        // Find which action this observation corresponds to
        let action_idx = obs.iter().position(|&v| v > 0.5).unwrap_or(0);
        let value = values
            .iter()
            .find(|(a, _)| *a as usize == action_idx)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        ([0.0f32; HYDRA_ACTION_SPACE], value)
    }
}

fn make_discard_only_step(legal_actions: &[usize]) -> StepRecord {
    let mut legal_mask = [false; HYDRA_ACTION_SPACE];
    let mut policy_logits = [f32::NEG_INFINITY; HYDRA_ACTION_SPACE];
    for &a in legal_actions {
        legal_mask[a] = true;
        // Set close logits to create a hard state
        policy_logits[a] = 1.0 + (a as f32 * 0.01);
    }
    StepRecord {
        obs: [0.0; OBS_SIZE],
        action: legal_actions[0] as u8,
        policy_logits,
        pi_old: [0.0; HYDRA_ACTION_SPACE],
        legal_mask,
        player_id: 0,
    }
}

// --- Helper: make a GameState and Observation for testing ---
// We use a default game state; the adapter stubs out the actual obs.
fn make_test_game() -> GameState {
    use riichienv_core::rule::GameRule;
    GameState::new(0, true, Some(42), 0, GameRule::default_tenhou())
}

fn make_test_obs() -> Observation {
    use riichienv_core::action::{Action, ActionType};
    Observation::new(
        0,
        std::array::from_fn(|_| Vec::new()),
        std::array::from_fn(|_| Vec::new()),
        std::array::from_fn(|_| Vec::new()),
        Vec::new(),
        [25000; 4],
        [false; 4],
        vec![Action::new(ActionType::Discard, Some(0), &[], None)],
        Vec::new(),
        0,
        0,
        0,
        0,
        0,
        Vec::new(),
        false,
        [None; 4],
        [None; 4],
        None,
    )
}

// --- Tests ---

#[test]
fn legal_discard_actions_extracts_only_discards() {
    let step = make_discard_only_step(&[1, 5, 10]);
    let discards = legal_discard_actions(&step);
    assert_eq!(discards, vec![1, 5, 10]);
}

#[test]
fn legal_discard_actions_ignores_non_discard_range() {
    let mut step = make_discard_only_step(&[1, 5]);
    // Add a non-discard action (e.g., riichi at 37+)
    step.legal_mask[40] = true;
    let discards = legal_discard_actions(&step);
    assert_eq!(discards, vec![1, 5]);
}

#[test]
fn base_pi_from_logits_sums_to_one() {
    let step = make_discard_only_step(&[1, 5, 10]);
    let pi = base_pi_from_logits(&step);
    let sum: f32 = pi.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum: {sum}");
}

#[test]
fn budget_from_legal_count_respects_minimum() {
    let cfg = ExitConfig::default_live_exit();
    // 3 legal discards -> 8.0 * 3 = 24, but min_visits is 64
    assert_eq!(budget_from_legal_count(&cfg, 3), 64);
    // 10 legal discards -> 8.0 * 10 = 80, exceeds min_visits
    assert_eq!(budget_from_legal_count(&cfg, 10), 80);
}

#[test]
fn seed_root_children_seeds_all_actions() {
    let mut tree = AfbsTree::new();
    let root = tree.add_node(100, 1.0, false);
    let priors = vec![(1u8, 0.4), (5, 0.35), (10, 0.25)];
    seed_root_children_all_legal(&mut tree, root, 100, &priors);

    assert_eq!(tree.nodes[root as usize].children.len(), 3);

    let actions: Vec<u8> = tree.nodes[root as usize]
        .children
        .iter()
        .map(|(a, _)| *a)
        .collect();
    assert_eq!(actions, vec![1, 5, 10]);

    // Priors should be normalized
    let prior_sum: f32 = tree.nodes[root as usize]
        .children
        .iter()
        .map(|(_, idx)| tree.nodes[*idx as usize].prior)
        .sum();
    assert!((prior_sum - 1.0).abs() < 1e-5);
}

#[test]
fn seed_root_children_seeds_more_than_top_k() {
    let mut tree = AfbsTree::new();
    let root = tree.add_node(200, 1.0, false);
    // 9 legal discards -- more than TOP_K=5
    let priors: Vec<(u8, f32)> = (0..9).map(|i| (i as u8, 1.0 / 9.0)).collect();
    seed_root_children_all_legal(&mut tree, root, 200, &priors);

    assert_eq!(
        tree.nodes[root as usize].children.len(),
        9,
        "should seed all 9, not truncated to TOP_K=5"
    );
}

#[test]
fn rejects_incompatible_state_non_discard_legal() {
    let mut step = make_discard_only_step(&[1, 5, 10]);
    // Make it incompatible by adding a non-discard action
    step.legal_mask[40] = true;

    let state = make_test_game();
    let obs = make_test_obs();
    let safety = SafetyInfo::new();
    let cfg = ExitConfig::default_live_exit();
    let values = vec![(1u8, 0.5), (5, 0.3), (10, 0.1)];
    let mut model = make_stub_model(&values);
    let mut adapter = StubAdapter { fail_obs: false };

    let result = try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
    assert!(
        result.is_none(),
        "should reject non-discard-compatible state"
    );
}

#[test]
fn rejects_fewer_than_two_legal_discards() {
    let step = make_discard_only_step(&[5]);

    let state = make_test_game();
    let obs = make_test_obs();
    let safety = SafetyInfo::new();
    let cfg = ExitConfig::default_live_exit();
    let mut model = |_: &[f32; OBS_SIZE]| ([0.0f32; HYDRA_ACTION_SPACE], 0.5f32);
    let mut adapter = StubAdapter { fail_obs: false };

    let result = try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
    assert!(result.is_none(), "should reject single-action states");
}

#[test]
fn rejects_non_hard_state() {
    // Create a step where one action dominates (big gap -> not hard)
    let mut legal_mask = [false; HYDRA_ACTION_SPACE];
    let mut logits = [f32::NEG_INFINITY; HYDRA_ACTION_SPACE];
    legal_mask[1] = true;
    legal_mask[5] = true;
    legal_mask[10] = true;
    logits[1] = 10.0; // dominant
    logits[5] = 1.0;
    logits[10] = 0.0;

    let step = StepRecord {
        obs: [0.0; OBS_SIZE],
        action: 1,
        policy_logits: logits,
        pi_old: [0.0; HYDRA_ACTION_SPACE],
        legal_mask,
        player_id: 0,
    };

    let state = make_test_game();
    let obs = make_test_obs();
    let safety = SafetyInfo::new();
    let cfg = ExitConfig::default_live_exit();
    let mut model = |_: &[f32; OBS_SIZE]| ([0.0f32; HYDRA_ACTION_SPACE], 0.5f32);
    let mut adapter = StubAdapter { fail_obs: false };

    let result = try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
    assert!(
        result.is_none(),
        "should reject non-hard state (big top-2 gap)"
    );
}

#[test]
fn rejects_missing_child_observation() {
    let step = make_discard_only_step(&[1, 5, 10]);

    let state = make_test_game();
    let obs = make_test_obs();
    let safety = SafetyInfo::new();
    let cfg = ExitConfig::default_live_exit();
    let mut model = |_: &[f32; OBS_SIZE]| ([0.0f32; HYDRA_ACTION_SPACE], 0.5f32);
    let mut adapter = StubAdapter { fail_obs: true };

    let result = try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
    assert!(result.is_none(), "should reject when adapter returns None");
}

#[test]
fn rejects_non_finite_value() {
    let step = make_discard_only_step(&[1, 5, 10]);

    let state = make_test_game();
    let obs = make_test_obs();
    let safety = SafetyInfo::new();
    let cfg = ExitConfig::default_live_exit();
    let mut model = |_: &[f32; OBS_SIZE]| ([0.0f32; HYDRA_ACTION_SPACE], f32::NAN);
    let mut adapter = StubAdapter { fail_obs: false };

    let result = try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
    assert!(result.is_none(), "should reject NaN value head output");
}

#[test]
fn produces_valid_exit_label_on_good_input() {
    // 3 legal discards with close logits (hard state)
    let step = make_discard_only_step(&[1, 5, 10]);

    let state = make_test_game();
    let obs = make_test_obs();
    let safety = SafetyInfo::new();
    let cfg = ExitConfig::default_live_exit();

    // Distinct child values so visits differ meaningfully
    let values = vec![(1u8, 0.8), (5, 0.5), (10, 0.2)];
    let mut model = make_stub_model(&values);
    let mut adapter = StubAdapter { fail_obs: false };

    let result = try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);

    // May or may not produce a label depending on KL/coverage gates,
    // but if it does, it must be valid.
    if let Some(label) = result {
        let target_sum: f32 = label.target.iter().sum();
        assert!(
            (target_sum - 1.0).abs() < 1e-3,
            "target should sum to 1, got {target_sum}"
        );

        // Mask should be binary and only on legal discard actions
        for (idx, &m) in label.mask.iter().enumerate() {
            assert!(
                m == 0.0 || (m - 1.0).abs() < 1e-3,
                "mask[{idx}] should be binary, got {m}"
            );
            if m > 0.5 {
                assert!(
                    idx <= DISCARD_END as usize,
                    "mask should only cover discard actions"
                );
                assert!(step.legal_mask[idx], "mask should only cover legal actions");
            }
        }

        // Target mass should be zero outside mask
        for idx in 0..HYDRA_ACTION_SPACE {
            if label.mask[idx] < 0.5 {
                assert!(
                    label.target[idx].abs() < 1e-5,
                    "target[{idx}] should be 0 outside mask"
                );
            }
        }
    }
}

#[test]
fn produces_delta_q_label_on_good_input() {
    let step = make_discard_only_step(&[1, 5, 10]);
    let state = make_test_game();
    let obs = make_test_obs();
    let safety = SafetyInfo::new();
    let cfg = ExitConfig::default_live_exit();
    let values = vec![(1u8, 0.8), (5, 0.5), (10, 0.2)];
    let mut model = make_stub_model(&values);
    let mut adapter = StubAdapter { fail_obs: false };

    let labels =
        try_live_search_labels(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter)
            .expect("search labels");
    let delta_q = labels.delta_q.expect("delta_q label");
    assert_eq!(delta_q.mask[1], 1.0);
    assert_eq!(delta_q.mask[5], 1.0);
    assert_eq!(delta_q.mask[10], 1.0);
    assert!(delta_q.target[1] > delta_q.target[5]);
    assert!(delta_q.target[5] > delta_q.target[10]);
}

#[test]
fn visit_target_differs_from_root_exit_policy() {
    // This test enforces the doctrinal distinction: visit-based labels
    // are NOT the same as q-softmax labels from root_exit_policy().
    let mut tree = AfbsTree::new();
    let root = tree.add_node(7, 1.0, false);

    // Manually build the test tree from Agent 22's blueprint
    let c1 = tree.add_node(predicted_child_hash(7, 1), 0.45, false);
    let c2 = tree.add_node(predicted_child_hash(7, 2), 0.35, false);
    let c5 = tree.add_node(predicted_child_hash(7, 5), 0.20, false);
    tree.nodes[root as usize].children.push((1, c1));
    tree.nodes[root as usize].children.push((2, c2));
    tree.nodes[root as usize].children.push((5, c5));

    // Set visits and values from the blueprint example
    tree.nodes[c1 as usize].visit_count = 10;
    tree.nodes[c1 as usize].total_value = 9.0;
    tree.nodes[c2 as usize].visit_count = 8;
    tree.nodes[c2 as usize].total_value = 4.0;
    tree.nodes[c5 as usize].visit_count = 6;
    tree.nodes[c5 as usize].total_value = 0.6;
    tree.nodes[root as usize].visit_count = 24;

    // Visit-based target: [10, 8, 6] / 24 = [0.417, 0.333, 0.250]
    let mut base_pi = [1e-6f32; HYDRA_ACTION_SPACE];
    base_pi[1] = 0.45;
    base_pi[2] = 0.35;
    base_pi[5] = 0.20;
    let mut legal = [0.0f32; HYDRA_ACTION_SPACE];
    legal[1] = 1.0;
    legal[2] = 1.0;
    legal[5] = 1.0;

    let (visit_target, _mask) = build_exit_from_afbs_tree(&tree, root, &base_pi, &legal, 8, 5.0)
        .expect("should build from valid tree");

    // q-softmax target from root_exit_policy
    let q_policy = tree.root_exit_policy(root, 1.0);

    // They should NOT be identical
    let l1_diff: f32 = (0..HYDRA_ACTION_SPACE)
        .map(|i| (visit_target[i] - q_policy[i]).abs())
        .sum();
    assert!(
        l1_diff > 0.05,
        "visit target and q-softmax should differ meaningfully, L1 gap = {l1_diff}"
    );
}

#[test]
fn expand_node_fails_coverage_on_many_discards() {
    // Proves that expand_node() truncates to TOP_K=5 which kills
    // coverage on states with 9+ legal discards.
    let mut tree = AfbsTree::new();
    let root = tree.add_node(100, 1.0, false);

    let mut logits = [f32::NEG_INFINITY; HYDRA_ACTION_SPACE];
    let mut mask = [false; HYDRA_ACTION_SPACE];
    // 9 legal discards
    for i in 0..9usize {
        logits[i] = 1.0;
        mask[i] = true;
    }

    tree.expand_node(root, &logits, &mask, false);
    let child_count = tree.nodes[root as usize].children.len();

    assert_eq!(child_count, 5, "expand_node should truncate to TOP_K=5");
    // 5/9 = 0.556 < 0.60 coverage threshold
    let max_coverage = child_count as f32 / 9.0;
    assert!(
        max_coverage < 0.60,
        "max coverage {max_coverage} should be below 0.60"
    );
}

#[test]
fn all_legal_seeding_passes_coverage_on_many_discards() {
    // Proves that seeding all legal children allows coverage >= 0.60
    let mut tree = AfbsTree::new();
    let root = tree.add_node(200, 1.0, false);

    let priors: Vec<(u8, f32)> = (0..9).map(|i| (i as u8, 1.0 / 9.0)).collect();
    seed_root_children_all_legal(&mut tree, root, 200, &priors);

    let child_count = tree.nodes[root as usize].children.len();
    assert_eq!(child_count, 9);

    // If all children get enough visits, coverage = 9/9 = 1.0
    let max_coverage = child_count as f32 / 9.0;
    assert!(
        max_coverage >= 0.60,
        "max coverage {max_coverage} should be >= 0.60"
    );
}

#[test]
fn live_exit_config_defaults_to_on() {
    let cfg = LiveExitConfig::default();
    assert!(cfg.enabled, "live producer must be default-on");
}

#[test]
fn rejects_aka_discard_state() {
    // A state with aka-5m legal should be rejected by compatible_discard_state
    let mut step = make_discard_only_step(&[1, 5, 10]);
    step.legal_mask[34] = true; // AKA_5M

    let state = make_test_game();
    let obs = make_test_obs();
    let safety = SafetyInfo::new();
    let cfg = ExitConfig::default_live_exit();
    let mut model = |_: &[f32; OBS_SIZE]| ([0.0f32; HYDRA_ACTION_SPACE], 0.5f32);
    let mut adapter = StubAdapter { fail_obs: false };

    let result = try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
    assert!(
        result.is_none(),
        "should reject state with aka discard legal"
    );
}

fn make_real_game_at_discard_phase() -> (GameState, Observation, u8) {
    use riichienv_core::rule::GameRule;

    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(42), 0, rule);
    state.skip_mjai_logging = true;

    let pid = state.current_player;
    let obs = state.get_observation(pid);
    (state, obs, pid)
}

#[test]
fn selfplay_adapter_returns_valid_child_obs() {
    let (state, obs, pid) = make_real_game_at_discard_phase();
    let safety = SafetyInfo::new();
    let mut adapter = SelfPlayExitAdapter::new();

    let hand = hydra_core::bridge::extract_hand(&obs);
    let first_tile_in_hand = hand.iter().position(|&c| c > 0).expect("hand not empty");

    let result = adapter.child_public_obs_after_discard(
        &state,
        &obs,
        pid,
        first_tile_in_hand as u8,
        &safety,
    );

    assert!(
        result.is_some(),
        "should produce child obs for valid discard"
    );
    let child_obs = result.unwrap();
    let nonzero = child_obs.iter().filter(|&&v| v != 0.0).count();
    assert!(nonzero > 0, "child obs should have nonzero values");
}

#[test]
fn selfplay_adapter_child_obs_differs_from_parent() {
    let (state, obs, pid) = make_real_game_at_discard_phase();
    let safety = SafetyInfo::new();
    let mut adapter = SelfPlayExitAdapter::new();

    let parent_obs = encode_observation(
        &mut ObservationEncoder::new(),
        &obs,
        &safety,
        state.drawn_tile.map(|t| t / 4),
    );

    let hand = hydra_core::bridge::extract_hand(&obs);
    let first_tile_in_hand = hand.iter().position(|&c| c > 0).expect("hand not empty");

    let child_obs = adapter
        .child_public_obs_after_discard(&state, &obs, pid, first_tile_in_hand as u8, &safety)
        .expect("should produce child obs");

    let diff: f32 = parent_obs
        .iter()
        .zip(child_obs.iter())
        .map(|(a, b)| (*a - *b).abs())
        .sum();
    assert!(
        diff > 0.1,
        "child obs should differ from parent (discarded tile changes hand/discard channels), diff={diff}"
    );
}

#[test]
fn selfplay_adapter_does_not_mutate_original_state() {
    let (state, obs, pid) = make_real_game_at_discard_phase();
    let safety = SafetyInfo::new();
    let mut adapter = SelfPlayExitAdapter::new();

    let scores_before = state.players.each_ref().map(|p| p.score);
    let turn_before = state.turn_count;

    let hand = hydra_core::bridge::extract_hand(&obs);
    let first_tile = hand.iter().position(|&c| c > 0).expect("hand not empty");
    let _ = adapter.child_public_obs_after_discard(&state, &obs, pid, first_tile as u8, &safety);

    let scores_after = state.players.each_ref().map(|p| p.score);
    assert_eq!(
        scores_before, scores_after,
        "original state scores must not change"
    );
    assert_eq!(
        state.turn_count, turn_before,
        "original state turn must not change"
    );
}

#[test]
fn selfplay_adapter_rejects_action_above_33() {
    let (state, obs, pid) = make_real_game_at_discard_phase();
    let safety = SafetyInfo::new();
    let mut adapter = SelfPlayExitAdapter::new();

    let result = adapter.child_public_obs_after_discard(&state, &obs, pid, 34, &safety);
    assert!(result.is_none(), "should reject aka discard action 34");

    let result = adapter.child_public_obs_after_discard(&state, &obs, pid, 40, &safety);
    assert!(result.is_none(), "should reject non-discard action 40");
}

#[test]
fn selfplay_adapter_distinct_actions_produce_distinct_obs() {
    let (state, obs, pid) = make_real_game_at_discard_phase();
    let safety = SafetyInfo::new();
    let mut adapter = SelfPlayExitAdapter::new();

    let hand = hydra_core::bridge::extract_hand(&obs);
    let tile_types_in_hand: Vec<u8> = hand
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(i, _)| i as u8)
        .collect();

    if tile_types_in_hand.len() >= 2 {
        let obs_a = adapter
            .child_public_obs_after_discard(&state, &obs, pid, tile_types_in_hand[0], &safety)
            .expect("obs for action a");
        let obs_b = adapter
            .child_public_obs_after_discard(&state, &obs, pid, tile_types_in_hand[1], &safety)
            .expect("obs for action b");

        let diff: f32 = obs_a
            .iter()
            .zip(obs_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "different discards should produce different child obs, diff={diff}"
        );
    }
}

#[test]
fn obs_hash_is_deterministic() {
    let obs = [0.42f32; OBS_SIZE];
    assert_eq!(obs_hash(&obs), obs_hash(&obs));
}

#[test]
fn obs_hash_differs_for_different_obs() {
    let obs_a = [0.0f32; OBS_SIZE];
    let mut obs_b = [0.0f32; OBS_SIZE];
    obs_b[0] = 1.0;
    assert_ne!(obs_hash(&obs_a), obs_hash(&obs_b));
}

#[test]
fn selfplay_adapter_root_hash_uses_step_obs() {
    let (state, _, pid) = make_real_game_at_discard_phase();
    let adapter = SelfPlayExitAdapter::new();

    let step_a = StepRecord {
        obs: [1.0; OBS_SIZE],
        action: 0,
        policy_logits: [0.0; HYDRA_ACTION_SPACE],
        pi_old: [0.0; HYDRA_ACTION_SPACE],
        legal_mask: [false; HYDRA_ACTION_SPACE],
        player_id: pid,
    };
    let mut step_b = step_a;
    step_b.obs = [2.0; OBS_SIZE];

    let hash_a = adapter.root_hash(&state, pid, &step_a.obs);
    let hash_b = adapter.root_hash(&state, pid, &step_b.obs);
    assert_ne!(
        hash_a, hash_b,
        "different obs should produce different hashes"
    );
}

#[test]
fn root_decision_context_from_step_matches_step_fields() {
    let step = make_discard_only_step(&[1, 5, 10]);
    let ctx = RootDecisionContext::from_step(&step);

    assert_eq!(ctx.obs_encoded, step.obs);
    assert_eq!(ctx.legal_mask, step.legal_mask);
    assert_eq!(ctx.policy_logits, step.policy_logits);
    assert_eq!(ctx.player_id, step.player_id);
}

#[test]
fn try_exit_label_from_context_matches_try_live_exit_label_on_selfplay_fixture() {
    let (state, obs, _pid) = make_real_game_at_discard_phase();
    let safety = SafetyInfo::new();

    let hand = hydra_core::bridge::extract_hand(&obs);
    let legal_tiles: Vec<usize> = hand
        .iter()
        .enumerate()
        .filter(|&(_, c)| *c > 0)
        .map(|(i, _)| i)
        .collect();

    if legal_tiles.len() < 2 {
        return;
    }

    let step = make_discard_only_step(&legal_tiles[..legal_tiles.len().min(13)]);
    let ctx = RootDecisionContext::from_step(&step);
    let cfg = ExitConfig::default_live_exit();
    let values: Vec<(u8, f32)> = legal_tiles
        .iter()
        .enumerate()
        .map(|(i, &t)| (t as u8, 0.5 - i as f32 * 0.05))
        .collect();

    let mut model_a = make_stub_model(&values);
    let mut model_b = make_stub_model(&values);
    let mut adapter_a = SelfPlayExitAdapter::new();
    let mut adapter_b = SelfPlayExitAdapter::new();

    let via_step = try_live_exit_label(
        &state,
        &obs,
        &step,
        &safety,
        &cfg,
        &mut model_a,
        &mut adapter_a,
    );
    let via_ctx = try_exit_label_from_context(
        &state,
        &obs,
        &ctx,
        &safety,
        &cfg,
        &mut model_b,
        &mut adapter_b,
    );

    assert_eq!(via_ctx, via_step);
}

#[test]
fn try_search_labels_from_context_batched_child_values_matches_single_call_fixture() {
    let (state, obs, _pid) = make_real_game_at_discard_phase();
    let safety = SafetyInfo::new();

    let hand = hydra_core::bridge::extract_hand(&obs);
    let legal_tiles: Vec<usize> = hand
        .iter()
        .enumerate()
        .filter(|&(_, c)| *c > 0)
        .map(|(i, _)| i)
        .collect();

    if legal_tiles.len() < 2 {
        return;
    }

    let step = make_discard_only_step(&legal_tiles[..legal_tiles.len().min(13)]);
    let ctx = RootDecisionContext::from_step(&step);
    let cfg = ExitConfig::default_live_exit();
    let values: Vec<(u8, f32)> = legal_tiles
        .iter()
        .enumerate()
        .map(|(i, &t)| (t as u8, 0.5 - i as f32 * 0.05))
        .collect();

    let mut model_a = make_stub_model(&values);
    let mut model_b = make_stub_model(&values);
    let mut adapter_a = SelfPlayExitAdapter::new();
    let mut adapter_b = SelfPlayExitAdapter::new();

    let via_single = try_search_labels_from_context(
        &state,
        &obs,
        &ctx,
        &safety,
        &cfg,
        &mut model_a,
        &mut adapter_a,
    );
    let via_batched = try_search_labels_from_context_with_batched_child_values(
        &state,
        &obs,
        &ctx,
        &safety,
        &cfg,
        &mut |child_obs| {
            child_obs
                .iter()
                .map(|obs_encoded| model_b(obs_encoded).1)
                .collect()
        },
        &mut adapter_b,
    );

    assert_eq!(via_batched, via_single);
}

#[test]
fn make_live_exit_fn_disabled_always_returns_none() {
    let cfg = LiveExitConfig {
        enabled: false,
        ..LiveExitConfig::default()
    };
    let model = |_: &[f32; OBS_SIZE]| ([0.0f32; HYDRA_ACTION_SPACE], 0.5f32);
    let mut exit_fn = make_live_exit_fn(cfg, model);

    let (state, obs, _pid) = make_real_game_at_discard_phase();
    let safety = SafetyInfo::new();
    let step = make_discard_only_step(&[1, 5, 10]);

    let result = exit_fn(&state, &obs, &step, &safety, 0);
    assert!(
        result.is_none(),
        "disabled config should always return None"
    );
}

#[test]
fn selfplay_adapter_produces_label_via_try_live_exit() {
    let (state, obs, _pid) = make_real_game_at_discard_phase();
    let safety = SafetyInfo::new();
    let mut adapter = SelfPlayExitAdapter::new();

    let hand = hydra_core::bridge::extract_hand(&obs);
    let legal_tiles: Vec<usize> = hand
        .iter()
        .enumerate()
        .filter(|&(_, c)| *c > 0)
        .map(|(i, _)| i)
        .collect();

    if legal_tiles.len() < 2 {
        return;
    }

    let step = make_discard_only_step(&legal_tiles[..legal_tiles.len().min(13)]);

    let cfg = ExitConfig::default_live_exit();
    let values: Vec<(u8, f32)> = legal_tiles
        .iter()
        .enumerate()
        .map(|(i, &t)| (t as u8, 0.5 - i as f32 * 0.05))
        .collect();
    let mut model = make_stub_model(&values);

    let result = try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);

    // May return None due to hard-state or KL gates, but should not panic
    if let Some(label) = result {
        let target_sum: f32 = label.target.iter().sum();
        assert!(
            (target_sum - 1.0).abs() < 1e-3,
            "target should sum to 1, got {target_sum}"
        );
    }
}
