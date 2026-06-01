//! ExIt pipeline: search target generation and safety valve.

use burn::prelude::*;

pub mod gates;
pub mod loss;
pub mod policy;
pub mod targets;

pub use gates::{compatible_discard_state, safety_valve_check};
pub use loss::exit_loss;
pub use policy::{anneal_exit_weight, exit_policy_from_q, is_hard_state};
pub use targets::{
    build_delta_q_from_afbs_tree, build_exit_from_afbs_tree, collate_delta_q_targets,
    collate_exit_targets, make_exit_target, make_exit_target_from_child_visits,
};

pub const MIN_EXIT_CHILD_VISITS: u32 = 2;
pub const MIN_EXIT_AVG_ROOT_VISITS_PER_LEGAL_DISCARD: f32 = 8.0;
pub const MIN_EXIT_COVERAGE: f32 = 0.60;

#[derive(Config, Debug)]
pub struct ExitConfig {
    #[config(default = "1.0")]
    pub tau_exit: f32,
    #[config(default = "0.5")]
    pub exit_weight: f32,
    #[config(default = "64")]
    pub min_visits: u32,
    #[config(default = "0.1")]
    pub hard_state_threshold: f32,
    #[config(default = "2.0")]
    pub safety_valve_max_kl: f32,
}

impl ExitConfig {
    pub fn summary(&self) -> String {
        format!(
            "exit(tau={:.1}, w={:.1}, visits>={}, kl<{:.1})",
            self.tau_exit, self.exit_weight, self.min_visits, self.safety_valve_max_kl
        )
    }

    pub fn default_live_exit() -> Self {
        Self::new()
    }
    pub fn min_visits_reached(&self, visit_count: u32) -> bool {
        visit_count >= self.min_visits
    }

    pub fn effective_weight(&self, phase: u8, progress: f32) -> f32 {
        anneal_exit_weight(self.exit_weight, phase, progress)
    }

    pub fn should_apply_exit(&self, visit_count: u32) -> bool {
        visit_count >= self.min_visits
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.tau_exit <= 0.0 {
            return Err("tau_exit must be positive");
        }
        if self.exit_weight < 0.0 {
            return Err("exit_weight must be non-negative");
        }
        if self.safety_valve_max_kl <= 0.0 {
            return Err("max_kl must be positive");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
