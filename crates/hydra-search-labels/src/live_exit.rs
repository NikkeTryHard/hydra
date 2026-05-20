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
use hydra_core::arena::{TrajectoryDeltaQLabel, TrajectoryExitLabel, softmax_temperature};
#[cfg(test)]
use hydra_core::bridge::encode_observation;
use hydra_core::bridge::encode_observation_ref;
use hydra_core::encoder::{OBS_SIZE, ObservationEncoder};
use hydra_core::safety::SafetyInfo;
use riichienv_core::action::{Action, ActionType};
use riichienv_core::observation::Observation;
use riichienv_core::observation_ref::ObservationRef;
use riichienv_core::state::GameState;

use crate::exit::{
    ExitConfig, MIN_EXIT_AVG_ROOT_VISITS_PER_LEGAL_DISCARD, build_delta_q_from_afbs_tree,
    build_exit_from_afbs_tree, compatible_discard_state, is_hard_state,
};

pub use hydra_train_types::selfplay::{RootDecisionContext, StepRecord};
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrajectorySearchLabels {
    pub exit: Option<TrajectoryExitLabel>,
    pub delta_q: Option<TrajectoryDeltaQLabel>,
}

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
    scratch_state: Option<GameState>,
}

impl SelfPlayExitAdapter {
    pub fn new() -> Self {
        Self {
            encoder: ObservationEncoder::new(),
            scratch_state: None,
        }
    }

    pub fn reset(&mut self) {
        self.scratch_state = None;
    }

    pub fn child_public_obs_after_discard_ref(
        &mut self,
        state: &GameState,
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

        let child_state = self.scratch_state.get_or_insert_with(|| state.clone());
        child_state.clone_from(state);
        child_state.skip_mjai_logging = true;

        let mut actions = [None; 4];
        actions[player as usize] = Some(riichienv_action);
        child_state.step_unchecked(&actions);

        let child_obs = child_state.observe(player);
        let encoded = encode_observation_ref(&mut self.encoder, &child_obs, safety);

        Some(encoded)
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
        self.child_public_obs_after_discard_ref(state, player, action, safety)
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

fn legal_discard_actions_from_mask(legal_mask: &[bool; HYDRA_ACTION_SPACE]) -> Vec<usize> {
    (0..=DISCARD_END as usize)
        .filter(|&action| legal_mask[action])
        .collect()
}

struct LiveSearchRoot {
    legal_f32: [f32; HYDRA_ACTION_SPACE],
    legal_discards: Vec<usize>,
    base_pi: [f32; HYDRA_ACTION_SPACE],
    budget: u32,
    tree: AfbsTree,
    root: NodeIdx,
}

fn prepare_live_search_root(
    state: &GameState,
    ctx: &RootDecisionContext,
    cfg: &ExitConfig,
    adapter: &impl ExitSearchAdapter,
) -> Option<LiveSearchRoot> {
    let legal_f32 = ctx.legal_mask.map(|b| if b { 1.0 } else { 0.0 });
    if !compatible_discard_state(&legal_f32) {
        return None;
    }

    let legal_discards = legal_discard_actions_from_mask(&ctx.legal_mask);
    if legal_discards.len() < 2 {
        return None;
    }

    let base_pi = softmax_temperature(&ctx.policy_logits, &ctx.legal_mask, 1.0);
    let mut hard_slice = [0.0f32; 34];
    for (idx, &action) in legal_discards.iter().enumerate() {
        hard_slice[idx] = base_pi[action];
    }
    if !is_hard_state(
        &hard_slice[..legal_discards.len()],
        cfg.hard_state_threshold,
    ) {
        return None;
    }

    let budget = budget_from_legal_count(cfg, legal_discards.len());
    let root_hash = adapter.root_hash(state, ctx.player_id, &ctx.obs_encoded);
    let mut tree = AfbsTree::new();
    let root = tree.add_node(root_hash, 1.0, false);
    let priors: Vec<(u8, f32)> = legal_discards
        .iter()
        .map(|&action| (action as u8, base_pi[action]))
        .collect();
    seed_root_children_all_legal(&mut tree, root, root_hash, &priors);

    Some(LiveSearchRoot {
        legal_f32,
        legal_discards,
        base_pi,
        budget,
        tree,
        root,
    })
}

fn run_prepared_root_with_values(search: &mut LiveSearchRoot, values: &[f32]) {
    let first_child_idx = search.tree.nodes[search.root as usize]
        .children
        .first()
        .map(|&(_, idx)| idx);
    search
        .tree
        .run_search_iterations(search.root, search.budget, &|child_idx| {
            first_child_idx
                .and_then(|first| child_idx.checked_sub(first))
                .and_then(|offset| values.get(offset as usize).copied())
                .unwrap_or(0.0)
        });
}

fn labels_from_prepared_root(
    search: &LiveSearchRoot,
    cfg: &ExitConfig,
) -> Option<TrajectorySearchLabels> {
    let exit = build_exit_from_afbs_tree(
        &search.tree,
        search.root,
        &search.base_pi,
        &search.legal_f32,
        search.budget,
        cfg.safety_valve_max_kl,
    )
    .and_then(|(target, mask)| TrajectoryExitLabel::from_slices(&target, &mask));
    let delta_q = build_delta_q_from_afbs_tree(&search.tree, search.root, &search.legal_f32)
        .and_then(|(target, mask)| TrajectoryDeltaQLabel::from_slices(&target, &mask));

    if exit.is_none() && delta_q.is_none() {
        None
    } else {
        Some(TrajectorySearchLabels { exit, delta_q })
    }
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
    try_live_search_labels(state, obs, step, safety, cfg, model_pv, adapter).and_then(|l| l.exit)
}

pub fn try_live_search_labels<M, A>(
    state: &GameState,
    obs: &Observation,
    step: &StepRecord,
    safety: &SafetyInfo,
    cfg: &ExitConfig,
    model_pv: &mut M,
    adapter: &mut A,
) -> Option<TrajectorySearchLabels>
where
    M: FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32),
    A: ExitSearchAdapter,
{
    let ctx = RootDecisionContext::from_step(step);
    try_search_labels_from_context(state, obs, &ctx, safety, cfg, model_pv, adapter)
}

pub fn try_live_search_labels_selfplay<M>(
    state: &GameState,
    _obs: &ObservationRef<'_>,
    step: &StepRecord,
    safety: &SafetyInfo,
    cfg: &ExitConfig,
    model_pv: &mut M,
    adapter: &mut SelfPlayExitAdapter,
) -> Option<TrajectorySearchLabels>
where
    M: FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32),
{
    let ctx = RootDecisionContext::from_step(step);
    let mut search = prepare_live_search_root(state, &ctx, cfg, adapter)?;

    let mut values = Vec::with_capacity(search.legal_discards.len());
    for &action in &search.legal_discards {
        let child_obs = adapter.child_public_obs_after_discard_ref(
            state,
            ctx.player_id,
            action as u8,
            safety,
        )?;
        let (_child_logits, v_child) = model_pv(&child_obs);
        if !v_child.is_finite() {
            return None;
        }
        values.push(v_child);
    }
    run_prepared_root_with_values(&mut search, &values);
    labels_from_prepared_root(&search, cfg)
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
    try_search_labels_from_context(state, obs, ctx, safety, cfg, model_pv, adapter)
        .and_then(|l| l.exit)
}

/// Attempts to produce an ExIt label using batched child value evaluation.
///
/// This is the replay/offline variant of [`try_exit_label_from_context`]: it
/// builds all child observations first, asks the caller to score them in one
/// batch, then applies the same AFBS visit-count teacher and safety gates.
pub fn try_exit_label_from_context_with_batched_child_values<M, A>(
    state: &GameState,
    obs: &Observation,
    ctx: &RootDecisionContext,
    safety: &SafetyInfo,
    cfg: &ExitConfig,
    child_values: &mut M,
    adapter: &mut A,
) -> Option<TrajectoryExitLabel>
where
    M: FnMut(&[[f32; OBS_SIZE]]) -> Vec<f32>,
    A: ExitSearchAdapter,
{
    try_search_labels_from_context_with_batched_child_values(
        state,
        obs,
        ctx,
        safety,
        cfg,
        child_values,
        adapter,
    )
    .and_then(|labels| labels.exit)
}

pub fn try_search_labels_from_context<M, A>(
    state: &GameState,
    obs: &Observation,
    ctx: &RootDecisionContext,
    safety: &SafetyInfo,
    cfg: &ExitConfig,
    model_pv: &mut M,
    adapter: &mut A,
) -> Option<TrajectorySearchLabels>
where
    M: FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32),
    A: ExitSearchAdapter,
{
    let mut search = prepare_live_search_root(state, ctx, cfg, adapter)?;

    let mut values = Vec::with_capacity(search.legal_discards.len());
    for &action in &search.legal_discards {
        let child_obs = adapter.child_public_obs_after_discard(
            state,
            obs,
            ctx.player_id,
            action as u8,
            safety,
        )?;
        let (_child_logits, v_child) = model_pv(&child_obs);
        if !v_child.is_finite() {
            return None;
        }
        values.push(v_child);
    }
    run_prepared_root_with_values(&mut search, &values);
    labels_from_prepared_root(&search, cfg)
}

/// Attempts to produce ExIt and delta-q labels using batched child values.
///
/// The caller supplies a batch value callback for all legal discard children.
/// Returns `None` when compatibility, hard-state, child-observation, value, or
/// target-construction gates reject the decision.
pub fn try_search_labels_from_context_with_batched_child_values<M, A>(
    state: &GameState,
    obs: &Observation,
    ctx: &RootDecisionContext,
    safety: &SafetyInfo,
    cfg: &ExitConfig,
    child_values: &mut M,
    adapter: &mut A,
) -> Option<TrajectorySearchLabels>
where
    M: FnMut(&[[f32; OBS_SIZE]]) -> Vec<f32>,
    A: ExitSearchAdapter,
{
    let mut search = prepare_live_search_root(state, ctx, cfg, adapter)?;

    let mut child_observations = Vec::with_capacity(search.legal_discards.len());
    for &action in &search.legal_discards {
        let child_obs = adapter.child_public_obs_after_discard(
            state,
            obs,
            ctx.player_id,
            action as u8,
            safety,
        )?;
        child_observations.push(child_obs);
    }

    let values = child_values(&child_observations);
    if values.len() != search.legal_discards.len() || values.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    run_prepared_root_with_values(&mut search, &values);
    labels_from_prepared_root(&search, cfg)
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
) -> impl FnMut(&GameState, &Observation, &StepRecord, &SafetyInfo, u32) -> Option<TrajectorySearchLabels>
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
        try_live_search_labels(
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
mod tests;
