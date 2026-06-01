//! Legacy/reference Rust/Burn inference server: network + SaF plus pondered AFBS.

mod policy;
mod server;
mod tensor;

pub use policy::{
    action_rank, argmax_legal, compute_entropy_from_logits, is_confident, mask_policy_cpu,
    needs_search, normalize_policy_cpu, num_legal_actions, policy_entropy, policy_top1_confidence,
    policy_top2_gap, sample_from_policy, validate_legal_mask,
};
pub use server::InferenceServer;
pub use tensor::{
    batch_legal_masks_to_tensor, illegal_action_rate, infer_action, infer_action_timed,
    legal_mask_to_tensor,
};

#[cfg(test)]
use crate::saf::SafConfig;
#[cfg(test)]
use burn::prelude::*;
#[cfg(test)]
use hydra_core::action::HYDRA_ACTION_SPACE;
#[cfg(test)]
use hydra_core::afbs::{PonderResult, TrustLevel};

use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES};

pub const OBS_FLAT_SIZE: usize = NUM_CHANNELS * NUM_TILES;

pub struct InferenceConfig {
    pub on_turn_budget_ms: u64,
    pub call_reaction_budget_ms: u64,
    pub agari_guard: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            on_turn_budget_ms: 150,
            call_reaction_budget_ms: 50,
            agari_guard: true,
        }
    }
}

impl InferenceConfig {
    pub fn summary(&self) -> String {
        format!(
            "infer(turn={}ms, call={}ms, guard={})",
            self.on_turn_budget_ms, self.call_reaction_budget_ms, self.agari_guard
        )
    }
}

#[cfg(test)]
mod tests;
