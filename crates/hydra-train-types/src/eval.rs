//! Backend-independent arena evaluation DTOs shared by training crates.

use burn::config::Config;

use crate::delta_q_promotion::ArenaPromotionDecision;

#[derive(Config, Debug)]
pub struct PairedArenaEvalConfig {
    #[config(default = "10000")]
    pub min_games: usize,
    #[config(default = "42")]
    pub seed: u64,
    #[config(default = "0.025")]
    pub max_mean_placement_regression: f32,
    #[config(default = "0.0")]
    pub strong_promotion_mean_placement_target: f32,
    #[config(default = "true")]
    pub same_seeds: bool,
    #[config(default = "true")]
    pub same_seat_rotation_schedule: bool,
    #[config(default = "true")]
    pub same_search_budget: bool,
    #[config(default = "true")]
    pub same_temperature: bool,
    #[config(default = "true")]
    pub same_frozen_opponent_pool: bool,
}

impl PairedArenaEvalConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.min_games == 0 {
            return Err("min_games must be > 0");
        }
        if self.max_mean_placement_regression < 0.0 {
            return Err("max_mean_placement_regression must be >= 0");
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!(
            "paired_arena(min_games={}, seed={}, max_reg={:.3}, strong_target={:.3}, same_seeds={}, same_rotation={}, same_budget={}, same_temp={}, frozen_pool={})",
            self.min_games,
            self.seed,
            self.max_mean_placement_regression,
            self.strong_promotion_mean_placement_target,
            self.same_seeds,
            self.same_seat_rotation_schedule,
            self.same_search_budget,
            self.same_temperature,
            self.same_frozen_opponent_pool,
        )
    }
}

#[derive(Debug, Clone)]
pub struct PairedArenaEvalResult {
    pub candidate_mean_placement: f32,
    pub baseline_mean_placement: f32,
    pub delta_mean_placement: f32,
    pub candidate_stable_dan: f32,
    pub baseline_stable_dan: f32,
    pub delta_stable_dan: f32,
    pub upper_confidence_bound_mean_placement: f32,
    pub compared_games: usize,
}

impl PairedArenaEvalResult {
    pub fn passes_non_regression(&self, config: &PairedArenaEvalConfig) -> bool {
        self.upper_confidence_bound_mean_placement <= config.max_mean_placement_regression
    }

    pub fn passes_strong_promotion(&self, config: &PairedArenaEvalConfig) -> bool {
        self.upper_confidence_bound_mean_placement <= config.strong_promotion_mean_placement_target
    }

    pub fn recommendation(&self, config: &PairedArenaEvalConfig) -> ArenaPromotionDecision {
        if self.passes_strong_promotion(config) {
            ArenaPromotionDecision::StrongPromotion
        } else if self.passes_non_regression(config) {
            ArenaPromotionDecision::NonRegressionOnly
        } else {
            ArenaPromotionDecision::Reject
        }
    }

    pub fn summary(&self, config: &PairedArenaEvalConfig) -> String {
        format!(
            "paired_arena(candidate_mp={:.3}, baseline_mp={:.3}, delta_mp={:+.3}, candidate_dan={:.3}, baseline_dan={:.3}, delta_dan={:+.3}, upper_ci={:.3}, games={}, decision={})",
            self.candidate_mean_placement,
            self.baseline_mean_placement,
            self.delta_mean_placement,
            self.candidate_stable_dan,
            self.baseline_stable_dan,
            self.delta_stable_dan,
            self.upper_confidence_bound_mean_placement,
            self.compared_games,
            self.recommendation(config).summary(),
        )
    }
}
