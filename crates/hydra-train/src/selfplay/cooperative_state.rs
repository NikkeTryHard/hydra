use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::afbs::{AfbsTree, NodeIdx};
use hydra_core::arena::TrajectoryStep;
use hydra_core::encoder::OBS_SIZE;
use riichienv_core::action::Action;
use riichienv_core::observation::Observation;

use super::StepRecord;

#[derive(Clone)]
pub(super) struct PendingPolicyRequest {
    pub(super) pid: u8,
    pub(super) obs: Observation,
    pub(super) obs_encoded: [f32; OBS_SIZE],
    pub(super) drawn_tile_before_action: Option<u8>,
    pub(super) turn: u32,
}

pub(super) struct ExitChildRequest {
    pub(super) child_idx: NodeIdx,
    pub(super) obs: [f32; OBS_SIZE],
}

pub(super) struct PendingExitStep {
    pub(super) step_record: StepRecord,
    pub(super) turn: u32,
    pub(super) tree: AfbsTree,
    pub(super) root: NodeIdx,
    pub(super) base_pi: [f32; HYDRA_ACTION_SPACE],
    pub(super) legal_f32: [f32; HYDRA_ACTION_SPACE],
    pub(super) budget: u32,
    pub(super) child_offset: usize,
    pub(super) child_count: usize,
    pub(super) output_index: usize,
}

pub(super) struct ExitSearchState {
    pub(super) steps: Vec<PendingExitStep>,
    pub(super) child_requests: Vec<ExitChildRequest>,
}

impl ExitSearchState {
    pub(super) fn new() -> Self {
        Self {
            steps: Vec::new(),
            child_requests: Vec::new(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

pub(super) struct PreparedExitSearch {
    pub(super) step: PendingExitStep,
    pub(super) child_requests: Vec<(NodeIdx, [f32; OBS_SIZE])>,
}

pub(super) struct PendingTurnState {
    pub(super) chosen_actions: [Option<Action>; 4],
    pub(super) players: Vec<u8>,
    pub(super) next_index: usize,
    pub(super) turn: u32,
    pub(super) pending_steps: Vec<Option<TrajectoryStep>>,
    pub(super) pending_values: Vec<f32>,
}

impl PendingTurnState {
    pub(super) fn new(players: Vec<u8>, turn: u32) -> Self {
        let pending_steps = Vec::with_capacity(players.len());
        Self {
            chosen_actions: [None; 4],
            players,
            next_index: 0,
            turn,
            pending_steps,
            pending_values: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct GameAdvance {
    pub(super) needs_policy: bool,
}
