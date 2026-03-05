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

pub fn compute_stable_dan(mean_placement: f32) -> f32 {
    (10.0 - (mean_placement - 1.0) * 4.0).clamp(0.0, 12.0)
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
}
