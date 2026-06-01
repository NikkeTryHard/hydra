use std::time::Duration;

use crate::action::{HYDRA_ACTION_SPACE, riichienv_to_hydra};
use crate::encoder::OBS_SIZE;
use riichienv_core::action::Action;

#[derive(Debug, Clone, Copy)]
pub struct CachedLegalActions {
    pub(super) turn: u32,
    pub(super) player_id: u8,
    pub(super) by_hydra_id: [Option<Action>; HYDRA_ACTION_SPACE],
}

pub(super) fn legal_action_map(legal_actions: &[Action]) -> [Option<Action>; HYDRA_ACTION_SPACE] {
    let mut by_hydra_id = [None; HYDRA_ACTION_SPACE];
    for &action in legal_actions {
        if let Ok(hydra) = riichienv_to_hydra(&action) {
            let idx = usize::from(hydra.id());
            if idx < HYDRA_ACTION_SPACE && by_hydra_id[idx].is_none() {
                by_hydra_id[idx] = Some(action);
            }
        }
    }
    by_hydra_id
}

#[derive(Debug, Clone)]
pub struct PendingDecision {
    pub obs: [f32; OBS_SIZE],
    pub legal_mask: [bool; HYDRA_ACTION_SPACE],
    pub legal_count: u8,
    pub player_id: u8,
    pub seat_id: u8,
    pub turn: u32,
    pub legal_actions: CachedLegalActions,
}

#[derive(Default)]
pub struct PendingDecisionTiming {
    pub calls: u64,
    pub decisions: u64,
    pub advanced: u64,
    pub complete: u64,
    pub wait_act: u64,
    pub wait_response: u64,
    pub legal_actions: Duration,
    pub observe: Duration,
    pub encode: Duration,
    pub legal_pack: Duration,
}
