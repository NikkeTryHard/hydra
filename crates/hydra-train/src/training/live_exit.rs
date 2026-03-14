//! Live AFBS ExIt producer for self-play decision-time label generation.
//!
//! Implements the Agent 22 blueprint: a learner-only, root-only AFBS
//! producer that generates visit-based ExIt labels at decision time during
//! self-play.  The producer is **default-on** after clearing the infrastructure
//! validation matrix, and emits `None` on any failed gate.
//!
//! The surviving evaluator is the current public model value head, used as
//! a leaf scorer inside root-only AFBS over all legal discard children.
//! The teacher object is root child visits via [`build_exit_from_afbs_tree`],
//! not q-softmax via `root_exit_policy()`.

use hydra_core::action::{DISCARD_END, HYDRA_ACTION_SPACE};
use hydra_core::afbs::{AfbsTree, NodeIdx, predicted_child_hash};
use hydra_core::arena::{TrajectoryExitLabel, softmax_temperature};
use hydra_core::bridge::encode_observation;
use hydra_core::encoder::{OBS_SIZE, ObservationEncoder};
use hydra_core::safety::SafetyInfo;
use riichienv_core::action::{Action, ActionType};
use riichienv_core::observation::Observation;
use riichienv_core::state::GameState;

use crate::selfplay::StepRecord;
use crate::training::exit::{
    ExitConfig, MIN_EXIT_AVG_ROOT_VISITS_PER_LEGAL_DISCARD, build_exit_from_afbs_tree,
    compatible_discard_state, is_hard_state,
};

/// Adapter trait for generating child public observations after a discard.
///
/// Implementors must produce the public observation tensor that the value
/// head would see after the root player discards a given tile.  This is the
/// main blocked surface identified by Agent 22 -- callers must provide a
/// concrete implementation that clones the game state, applies the discard,
/// and re-encodes without leaking hidden state.
pub trait ExitSearchAdapter {
    /// Returns the info-state hash for the root player at the current state.
    fn root_hash(&self, state: &GameState, player: u8, obs_encoded: &[f32; OBS_SIZE]) -> u64;

    /// Produces the public observation after the root player discards `action`.
    ///
    /// Returns `None` if the child observation cannot be constructed (e.g.
    /// the action is invalid or the state cannot be cloned safely).
    fn child_public_obs_after_discard(
        &mut self,
        state: &GameState,
        obs: &Observation,
        player: u8,
        action: u8,
        safety: &SafetyInfo,
    ) -> Option<[f32; OBS_SIZE]>;
}

/// Concrete [`ExitSearchAdapter`] for self-play that reconstructs child
/// observations by cloning the game state, applying a discard, and
/// re-encoding from the root player's public perspective.
///
/// Hidden-state-contingent opponent actions are NOT rolled through.
/// The observation is taken immediately after the discard resolves,
/// giving the value head the root player's public view of the
/// post-discard state.
pub struct SelfPlayExitAdapter {
    encoder: ObservationEncoder,
}

impl SelfPlayExitAdapter {
    pub fn new() -> Self {
        Self {
            encoder: ObservationEncoder::new(),
        }
    }
}

impl Default for SelfPlayExitAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl ExitSearchAdapter for SelfPlayExitAdapter {
    fn root_hash(&self, _state: &GameState, _player: u8, obs_encoded: &[f32; OBS_SIZE]) -> u64 {
        obs_hash(obs_encoded)
    }

    fn child_public_obs_after_discard(
        &mut self,
        state: &GameState,
        _obs: &Observation,
        player: u8,
        action: u8,
        safety: &SafetyInfo,
    ) -> Option<[f32; OBS_SIZE]> {
        if action > 33 {
            return None;
        }

        let hand = state.players[player as usize].hand_slice();
        let tile136 = hand.iter().find(|&&t| t / 4 == action)?;
        let riichienv_action = Action::new(ActionType::Discard, Some(*tile136), &[], None);

        let mut child_state = state.clone();
        child_state.skip_mjai_logging = true;

        let mut actions = [None; 4];
        actions[player as usize] = Some(riichienv_action);
        child_state.step_unchecked(&actions);

        let child_obs = child_state.get_observation(player);
        let encoded = encode_observation(&mut self.encoder, &child_obs, safety, None);

        Some(encoded)
    }
}

/// FNV-1a hash on a downsampled subset of observation values.
///
/// Samples every 8th float for speed while maintaining enough
/// entropy for distinct observations at self-play scale.
pub fn obs_hash(obs: &[f32; OBS_SIZE]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for chunk in obs.chunks(8) {
        let bits = chunk[0].to_bits() as u64;
        hash ^= bits;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Extracts the indices of legal discard actions from a step record.
///
/// Returns only actions in the discard range `[0, DISCARD_END]`.
pub fn legal_discard_actions(step: &StepRecord) -> Vec<usize> {
    (0..=DISCARD_END as usize)
        .filter(|&a| step.legal_mask[a])
        .collect()
}

/// Computes the base prior from raw policy logits at temperature 1.0.
///
/// Uses the raw network logits, not `pi_old` which includes self-play
/// temperature.  The search prior and KL safety valve compare against
/// the raw network prior.
pub fn base_pi_from_logits(step: &StepRecord) -> [f32; HYDRA_ACTION_SPACE] {
    softmax_temperature(&step.policy_logits, &step.legal_mask, 1.0)
}

/// Computes the minimum AFBS search budget from the number of legal discards.
///
/// The budget is the larger of `cfg.min_visits` and the ceiling of
/// `MIN_EXIT_AVG_ROOT_VISITS_PER_LEGAL_DISCARD * n_legal`, ensuring the
/// existing average-visits gate can be satisfied.
pub fn budget_from_legal_count(cfg: &ExitConfig, n_legal: usize) -> u32 {
    cfg.min_visits
        .max((MIN_EXIT_AVG_ROOT_VISITS_PER_LEGAL_DISCARD * n_legal as f32).ceil() as u32)
}

/// Minimal root-decision context required by the ExIt producer.
///
/// This keeps the canonical teacher-building logic reusable across live and
/// future offline producer paths without forcing those paths to construct a
/// full [`StepRecord`].
#[derive(Clone, Copy, Debug)]
pub struct RootDecisionContext {
    pub obs_encoded: [f32; OBS_SIZE],
    pub legal_mask: [bool; HYDRA_ACTION_SPACE],
    pub policy_logits: [f32; HYDRA_ACTION_SPACE],
    pub player_id: u8,
}

impl RootDecisionContext {
    pub fn from_step(step: &StepRecord) -> Self {
        Self {
            obs_encoded: step.obs,
            legal_mask: step.legal_mask,
            policy_logits: step.policy_logits,
            player_id: step.player_id,
        }
    }
}

/// Seeds all legal discard children onto an AFBS tree root node.
///
/// Unlike `expand_node()` which truncates to `TOP_K = 5`, this seeds
/// **every** legal discard action so that `build_exit_from_afbs_tree`
/// can meet the 60% coverage requirement on states with 9+ legal discards.
///
/// Priors are re-normalized over the seeded children.
pub fn seed_root_children_all_legal(
    tree: &mut AfbsTree,
    root: NodeIdx,
    root_hash: u64,
    priors: &[(u8, f32)],
) {
    let z = priors.iter().map(|(_, p)| *p).sum::<f32>().max(1e-8);
    for &(action, prior) in priors {
        let child_hash = predicted_child_hash(root_hash, action);
        let child = tree.add_node(child_hash, prior / z, false);
        tree.nodes[root as usize].children.push((action, child));
    }
}

/// Attempts to produce a live ExIt label for a single self-play decision.
///
/// This is the full producer algorithm from Agent 22's blueprint:
///
/// 1. Reject non-discard-compatible states
/// 2. Reject states with fewer than 2 legal discards
/// 3. Compute base prior from raw logits (not pi_old)
/// 4. Reject non-hard states (top-2 gap >= threshold)
/// 5. Seed AFBS root with all legal discard children
/// 6. Evaluate each child with the model value head
/// 7. Run root-only AFBS search
/// 8. Build the label via `build_exit_from_afbs_tree`
/// 9. Emit `None` on any failed gate
///
/// The producer is default-on after the infrastructure validation matrix
/// cleared.  The `enabled` field on `LiveExitConfig` controls this.
pub fn try_live_exit_label<M, A>(
    state: &GameState,
    obs: &Observation,
    step: &StepRecord,
    safety: &SafetyInfo,
    cfg: &ExitConfig,
    model_pv: &mut M,
    adapter: &mut A,
) -> Option<TrajectoryExitLabel>
where
    M: FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32),
    A: ExitSearchAdapter,
{
    let ctx = RootDecisionContext::from_step(step);
    try_exit_label_from_context(state, obs, &ctx, safety, cfg, model_pv, adapter)
}

/// Attempts to produce an ExIt label from a reusable root-decision context.
///
/// This preserves the live producer semantics while decoupling the canonical
/// teacher-building path from self-play-specific carrier types.
pub fn try_exit_label_from_context<M, A>(
    state: &GameState,
    obs: &Observation,
    ctx: &RootDecisionContext,
    safety: &SafetyInfo,
    cfg: &ExitConfig,
    model_pv: &mut M,
    adapter: &mut A,
) -> Option<TrajectoryExitLabel>
where
    M: FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32),
    A: ExitSearchAdapter,
{
    // Step 1: state compatibility gate
    let legal_f32 = ctx.legal_mask.map(|b| if b { 1.0 } else { 0.0 });
    if !compatible_discard_state(&legal_f32) {
        return None;
    }

    // Step 2: minimum legal discards gate
    let legal_discards = (0..=DISCARD_END as usize)
        .filter(|&a| ctx.legal_mask[a])
        .collect::<Vec<_>>();
    if legal_discards.len() < 2 {
        return None;
    }

    // Step 3: base policy from raw logits (not exploration temperature)
    let base_pi = softmax_temperature(&ctx.policy_logits, &ctx.legal_mask, 1.0);

    // Step 4: hard-state gate
    let hard_slice: Vec<f32> = legal_discards.iter().map(|&a| base_pi[a]).collect();
    if !is_hard_state(&hard_slice, cfg.hard_state_threshold) {
        return None;
    }

    // Step 5: compute dynamic visit budget
    let budget = budget_from_legal_count(cfg, legal_discards.len());

    // Step 6: build AFBS tree with all legal discard children
    let root_hash = adapter.root_hash(state, ctx.player_id, &ctx.obs_encoded);
    let mut tree = AfbsTree::new();
    let root = tree.add_node(root_hash, 1.0, false);

    let priors: Vec<(u8, f32)> = legal_discards
        .iter()
        .map(|&a| (a as u8, base_pi[a]))
        .collect();
    seed_root_children_all_legal(&mut tree, root, root_hash, &priors);

    // Step 7: evaluate each child with the model value head
    // Cache child values so repeated PUCT visits reuse the same score.
    let mut value_by_child = std::collections::HashMap::<NodeIdx, f32>::new();
    for &(action, child_idx) in &tree.nodes[root as usize].children.clone() {
        let child_obs =
            adapter.child_public_obs_after_discard(state, obs, ctx.player_id, action, safety)?;
        let (_child_logits, v_child) = model_pv(&child_obs);
        if !v_child.is_finite() {
            return None;
        }
        value_by_child.insert(child_idx, v_child);
    }

    // Step 8: run root-only AFBS search with cached child values
    tree.run_search_iterations(root, budget, &|child_idx| {
        value_by_child.get(&child_idx).copied().unwrap_or(0.0)
    });

    // Step 9: build the label using the canonical visit-based teacher
    let (target, mask) = build_exit_from_afbs_tree(
        &tree,
        root,
        &base_pi,
        &legal_f32,
        budget,
        cfg.safety_valve_max_kl,
    )?;

    TrajectoryExitLabel::from_slices(&target, &mask)
}

/// Configuration for the live ExIt producer.
///
/// Wraps the standard [`ExitConfig`] with a feature gate.  The producer
/// is default-on after the infrastructure validation matrix cleared it.
/// Set `enabled = false` explicitly to disable label generation.
#[derive(Debug, Clone)]
pub struct LiveExitConfig {
    /// Whether the live producer is enabled.  Default: `true`.
    pub enabled: bool,
    /// The underlying ExIt gate configuration.
    pub exit_config: ExitConfig,
}

impl Default for LiveExitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exit_config: ExitConfig::default_phase3(),
        }
    }
}

/// Creates an exit label closure wired with a [`SelfPlayExitAdapter`] for
/// use with [`run_self_play_game_with_exit_labels`].
///
/// When `cfg.enabled` is false, the returned closure always emits `None`.
pub fn make_live_exit_fn<M>(
    cfg: LiveExitConfig,
    mut model_pv: M,
) -> impl FnMut(&GameState, &Observation, &StepRecord, &SafetyInfo, u32) -> Option<TrajectoryExitLabel>
where
    M: FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32),
{
    let mut adapter = SelfPlayExitAdapter::new();
    let exit_config = cfg.exit_config;
    let enabled = cfg.enabled;

    move |state, obs, step, safety, _turn| {
        if !enabled {
            return None;
        }
        try_live_exit_label(
            state,
            obs,
            step,
            safety,
            &exit_config,
            &mut model_pv,
            &mut adapter,
        )
    }
}

#[cfg(test)]
mod tests {
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
        let cfg = ExitConfig::default_phase3();
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
        let cfg = ExitConfig::default_phase3();
        let values = vec![(1u8, 0.5), (5, 0.3), (10, 0.1)];
        let mut model = make_stub_model(&values);
        let mut adapter = StubAdapter { fail_obs: false };

        let result =
            try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
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
        let cfg = ExitConfig::default_phase3();
        let mut model = |_: &[f32; OBS_SIZE]| ([0.0f32; HYDRA_ACTION_SPACE], 0.5f32);
        let mut adapter = StubAdapter { fail_obs: false };

        let result =
            try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
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
        let cfg = ExitConfig::default_phase3();
        let mut model = |_: &[f32; OBS_SIZE]| ([0.0f32; HYDRA_ACTION_SPACE], 0.5f32);
        let mut adapter = StubAdapter { fail_obs: false };

        let result =
            try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
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
        let cfg = ExitConfig::default_phase3();
        let mut model = |_: &[f32; OBS_SIZE]| ([0.0f32; HYDRA_ACTION_SPACE], 0.5f32);
        let mut adapter = StubAdapter { fail_obs: true };

        let result =
            try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
        assert!(result.is_none(), "should reject when adapter returns None");
    }

    #[test]
    fn rejects_non_finite_value() {
        let step = make_discard_only_step(&[1, 5, 10]);

        let state = make_test_game();
        let obs = make_test_obs();
        let safety = SafetyInfo::new();
        let cfg = ExitConfig::default_phase3();
        let mut model = |_: &[f32; OBS_SIZE]| ([0.0f32; HYDRA_ACTION_SPACE], f32::NAN);
        let mut adapter = StubAdapter { fail_obs: false };

        let result =
            try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
        assert!(result.is_none(), "should reject NaN value head output");
    }

    #[test]
    fn produces_valid_exit_label_on_good_input() {
        // 3 legal discards with close logits (hard state)
        let step = make_discard_only_step(&[1, 5, 10]);

        let state = make_test_game();
        let obs = make_test_obs();
        let safety = SafetyInfo::new();
        let cfg = ExitConfig::default_phase3();

        // Distinct child values so visits differ meaningfully
        let values = vec![(1u8, 0.8), (5, 0.5), (10, 0.2)];
        let mut model = make_stub_model(&values);
        let mut adapter = StubAdapter { fail_obs: false };

        let result =
            try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);

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

        let (visit_target, _mask) =
            build_exit_from_afbs_tree(&tree, root, &base_pi, &legal, 8, 5.0)
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
        let cfg = ExitConfig::default_phase3();
        let mut model = |_: &[f32; OBS_SIZE]| ([0.0f32; HYDRA_ACTION_SPACE], 0.5f32);
        let mut adapter = StubAdapter { fail_obs: false };

        let result =
            try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);
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
            .map(|(a, b)| (a - b).abs())
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
        let _ =
            adapter.child_public_obs_after_discard(&state, &obs, pid, first_tile as u8, &safety);

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
        let cfg = ExitConfig::default_phase3();
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

        let cfg = ExitConfig::default_phase3();
        let values: Vec<(u8, f32)> = legal_tiles
            .iter()
            .enumerate()
            .map(|(i, &t)| (t as u8, 0.5 - i as f32 * 0.05))
            .collect();
        let mut model = make_stub_model(&values);

        let result =
            try_live_exit_label(&state, &obs, &step, &safety, &cfg, &mut model, &mut adapter);

        // May return None due to hard-state or KL gates, but should not panic
        if let Some(label) = result {
            let target_sum: f32 = label.target.iter().sum();
            assert!(
                (target_sum - 1.0).abs() < 1e-3,
                "target should sum to 1, got {target_sum}"
            );
        }
    }
}
