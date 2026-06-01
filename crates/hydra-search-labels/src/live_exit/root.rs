use hydra_core::action::{DISCARD_END, HYDRA_ACTION_SPACE};
use hydra_core::afbs::{AfbsTree, NodeIdx, predicted_child_hash};
use hydra_core::arena::{TrajectoryDeltaQLabel, TrajectoryExitLabel, softmax_temperature};
use riichienv_core::state::GameState;

use crate::exit::{
    ExitConfig, MIN_EXIT_AVG_ROOT_VISITS_PER_LEGAL_DISCARD, build_delta_q_from_afbs_tree,
    build_exit_from_afbs_tree, compatible_discard_state, is_hard_state,
};

use super::adapter::ExitSearchAdapter;
use super::{RootDecisionContext, StepRecord, TrajectorySearchLabels};

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

pub(super) struct LiveSearchRoot {
    pub(super) legal_f32: [f32; HYDRA_ACTION_SPACE],
    pub(super) legal_discards: Vec<usize>,
    pub(super) base_pi: [f32; HYDRA_ACTION_SPACE],
    pub(super) budget: u32,
    pub(super) tree: AfbsTree,
    pub(super) root: NodeIdx,
}

pub(super) fn prepare_live_search_root(
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

pub(super) fn run_prepared_root_with_values(search: &mut LiveSearchRoot, values: &[f32]) {
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

pub(super) fn labels_from_prepared_root(
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
