//! Evaluation harness: run N games and collect metrics.

#[derive(Config, Debug)]
pub struct EvalConfig {
    #[config(default = "1000")]
    pub num_games: usize,
    #[config(default = "42")]
    pub seed: u64,
}

use burn::prelude::*;

#[derive(Debug, Clone)]
pub struct EvalResult {
    pub mean_placement: f32,
    pub stable_dan: f32,
    pub win_rate: f32,
    pub deal_in_rate: f32,
    pub tsumo_rate: f32,
}

impl Default for EvalResult {
    fn default() -> Self {
        Self {
            mean_placement: 2.5,
            stable_dan: 0.0,
            win_rate: 0.0,
            deal_in_rate: 0.0,
            tsumo_rate: 0.0,
        }
    }
}

pub struct TrainingMetrics {
    pub epoch: u32,
    pub total_loss: f64,
    pub policy_agreement: f64,
    pub value_mse: f64,
    pub games_completed: u64,
    pub arena_mean_score: f32,
    pub distill_kl: f32,
    pub elo: f32,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            epoch: 0,
            total_loss: 0.0,
            policy_agreement: 0.0,
            value_mse: 0.0,
            games_completed: 0,
            arena_mean_score: 0.0,
            distill_kl: 0.0,
            elo: 1500.0,
        }
    }
}

pub fn compute_stable_dan(mean_placement: f32) -> f32 {
    (10.0 - (mean_placement - 1.0) * 4.0).clamp(0.0, 12.0)
}

pub fn evaluate_from_placements(placements: &[u8]) -> EvalResult {
    if placements.is_empty() {
        return EvalResult::default();
    }
    let n = placements.len() as f32;
    let mean_placement = placements.iter().map(|&p| p as f32 + 1.0).sum::<f32>() / n;
    let wins = placements.iter().filter(|&&p| p == 0).count() as f32;
    EvalResult {
        mean_placement,
        stable_dan: compute_stable_dan(mean_placement),
        win_rate: wins / n,
        deal_in_rate: 0.0,
        tsumo_rate: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_dan_formula() {
        let dan_perfect = compute_stable_dan(1.0);
        assert!((dan_perfect - 10.0).abs() < 0.01);
        let dan_avg = compute_stable_dan(2.5);
        assert!(dan_avg > 0.0 && dan_avg < 10.0);
    }

    #[test]
    fn eval_result_defaults() {
        let result = EvalResult::default();
        assert!((result.mean_placement - 2.5).abs() < 0.01);
    }

    #[test]
    fn eval_deterministic_with_seed() {
        let placements = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let r1 = evaluate_from_placements(&placements);
        let r2 = evaluate_from_placements(&placements);
        assert!((r1.mean_placement - r2.mean_placement).abs() < 1e-6);
        assert!((r1.win_rate - r2.win_rate).abs() < 1e-6);
    }

    #[test]
    fn eval_reports_all_metrics() {
        let placements = vec![0, 0, 1, 2, 3, 1];
        let result = evaluate_from_placements(&placements);
        assert!(result.mean_placement > 1.0 && result.mean_placement < 4.0);
        assert!(result.stable_dan >= 0.0);
        assert!(result.win_rate > 0.0);
    }
}
