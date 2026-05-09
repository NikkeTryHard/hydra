//! Shared self-play decision DTOs used across training producers.
//!
//! These types stay below `hydra-train` so live and replay label producers can
//! exchange root decision data without depending on each other's modules.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;

/// Network decision record captured at a self-play action.
#[derive(Debug, Clone, Copy)]
pub struct StepRecord {
    pub obs: [f32; OBS_SIZE],
    pub action: u8,
    pub policy_logits: [f32; HYDRA_ACTION_SPACE],
    pub pi_old: [f32; HYDRA_ACTION_SPACE],
    pub legal_mask: [bool; HYDRA_ACTION_SPACE],
    pub player_id: u8,
}

/// Minimal root-decision context required by ExIt-style producers.
///
/// This keeps the canonical teacher-building logic reusable across live and
/// replay producer paths without forcing those paths to construct a full
/// [`StepRecord`].
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
