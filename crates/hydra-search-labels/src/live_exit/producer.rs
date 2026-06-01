use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::arena::TrajectoryExitLabel;
use hydra_core::encoder::OBS_SIZE;
use hydra_core::safety::SafetyInfo;
use riichienv_core::observation::Observation;
use riichienv_core::observation_ref::ObservationRef;
use riichienv_core::state::GameState;

use crate::exit::ExitConfig;

use super::adapter::{ExitSearchAdapter, SelfPlayExitAdapter};
use super::root::{
    labels_from_prepared_root, prepare_live_search_root, run_prepared_root_with_values,
};
use super::{RootDecisionContext, StepRecord, TrajectorySearchLabels};

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
