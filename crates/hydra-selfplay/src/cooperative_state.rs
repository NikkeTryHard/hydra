//! Cooperative self-play runner state records.
//!
//! These types carry pending policy/search work between cooperative runner
//! ticks. They are backend-independent and intentionally avoid `hydra-train`
//! dependencies so search-label and training adapters can share the seam.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::afbs::{AfbsTree, NodeIdx};
use hydra_core::arena::TrajectoryStep;
use hydra_core::encoder::OBS_SIZE;
use hydra_train_types::selfplay::StepRecord;
use riichienv_core::action::Action;

/// Pending neural-policy request for one player decision.
#[derive(Clone)]
pub struct PendingPolicyRequest {
    /// Acting player id, `0..4`.
    pub pid: u8,
    /// Encoded observation tensor flattened to the canonical observation size.
    pub obs_encoded: [f32; OBS_SIZE],
    /// Drawn tile before applying the selected action, if any.
    pub drawn_tile_before_action: Option<u8>,
    /// Self-play turn index for this request.
    pub turn: u32,
}

/// Pending child value request for live ExIt search.
pub struct ExitChildRequest {
    /// Child node index in the AFBS tree.
    pub child_idx: NodeIdx,
    /// Encoded child observation to evaluate.
    pub obs: [f32; OBS_SIZE],
}

/// Pending ExIt step waiting for batched child values.
pub struct PendingExitStep {
    /// Original step record for the root decision.
    pub step_record: StepRecord,
    /// Self-play turn index for this step.
    pub turn: u32,
    /// AFBS tree containing root and child nodes.
    pub tree: AfbsTree,
    /// Root node index for the decision.
    pub root: NodeIdx,
    /// Base policy over Hydra actions.
    pub base_pi: [f32; HYDRA_ACTION_SPACE],
    /// Legal action mask as `0.0/1.0` floats.
    pub legal_f32: [f32; HYDRA_ACTION_SPACE],
    /// Search visit budget.
    pub budget: u32,
    /// Offset into the shared child request/value buffer.
    pub child_offset: usize,
    /// Number of child requests for this root.
    pub child_count: usize,
    /// Output trajectory step index to fill once labels are ready.
    pub output_index: usize,
}

/// Accumulated pending ExIt search work for a runner.
pub struct ExitSearchState {
    /// Root steps awaiting child values.
    pub steps: Vec<PendingExitStep>,
    /// Child observations to evaluate in batch.
    pub child_requests: Vec<ExitChildRequest>,
}

impl ExitSearchState {
    /// Creates an empty search-state accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            child_requests: Vec::new(),
        }
    }

    /// Returns true when no root steps are pending.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl Default for ExitSearchState {
    fn default() -> Self {
        Self::new()
    }
}

/// Prepared ExIt root plus extracted child requests.
pub struct PreparedExitSearch {
    /// Pending root step metadata.
    pub step: PendingExitStep,
    /// Child node ids and observations requiring value evaluation.
    pub child_requests: Vec<(NodeIdx, [f32; OBS_SIZE])>,
}

/// Per-turn action staging for cooperative multi-response handling.
pub struct PendingTurnState {
    /// Chosen riichi-env actions by player id.
    pub chosen_actions: [Option<Action>; 4],
    /// Players that must act/respond this turn.
    pub players: Vec<u8>,
    /// Next player index inside `players`.
    pub next_index: usize,
    /// Self-play turn index.
    pub turn: u32,
    /// Pending trajectory steps staged before turn flush.
    pub pending_steps: Vec<Option<TrajectoryStep>>,
    /// Pending value estimates aligned with staged steps.
    pub pending_values: Vec<f32>,
}

impl PendingTurnState {
    /// Creates pending turn state for the supplied acting players.
    #[must_use]
    pub fn new(players: Vec<u8>, turn: u32) -> Self {
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

/// Result of advancing a cooperative game runner until it blocks.
#[derive(Clone, Copy, Debug, Default)]
pub struct GameAdvance {
    /// True when the runner needs a policy inference result.
    pub needs_policy: bool,
}
