//! Evaluation harness: run N games and collect metrics.

use burn::prelude::*;
use hydra_core::arena::compute_placements;

use crate::model::HydraModel;
use crate::selfplay::run_mixed_policy_game_scores;

#[derive(Config, Debug)]
pub struct EvalConfig {
    #[config(default = "1000")]
    pub num_games: usize,
    #[config(default = "42")]
    pub seed: u64,
}

impl EvalConfig {
    pub fn with_games(self, n: usize) -> Self {
        Self::new().with_num_games(n).with_seed(self.seed)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_games == 0 {
            return Err("num_games must be > 0");
        }
        Ok(())
    }

    pub fn summary(&self) -> String {
        format!("eval(games={}, seed={})", self.num_games, self.seed)
    }
}

#[derive(Debug, Clone)]
pub struct EvalResult {
    pub mean_placement: f32,
    pub stable_dan: f32,
    pub win_rate: f32,
    pub deal_in_rate: f32,
    pub tsumo_rate: f32,
}

impl EvalResult {
    pub fn meets_target(&self, target_dan: f32) -> bool {
        self.stable_dan >= target_dan
    }

    pub fn is_mortal_level(&self) -> bool {
        self.stable_dan >= 8.0
    }

    pub fn is_tendan_plus(&self) -> bool {
        self.stable_dan >= 10.0
    }

    pub fn summary(&self) -> String {
        format!(
            "placement={:.2} dan={:.1} win={:.1}% deal_in={:.1}%",
            self.mean_placement,
            self.stable_dan,
            self.win_rate * 100.0,
            self.deal_in_rate * 100.0
        )
    }
}

impl EvalResult {
    pub fn from_mean_placement(mean_placement: f32) -> Self {
        Self {
            mean_placement,
            stable_dan: compute_stable_dan(mean_placement),
            ..Default::default()
        }
    }
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

pub struct BenchmarkGates {
    pub afbs_on_turn_ms: f32,
    pub ct_smc_dp_ms: f32,
    pub endgame_ms: f32,
    pub self_play_games_per_sec: f32,
    pub distill_kl_drift: f32,
}

impl BenchmarkGates {
    pub fn summary(&self) -> String {
        format!(
            "afbs={:.0}ms smc={:.2}ms endgame={:.0}ms play={:.0}g/s kl={:.3}",
            self.afbs_on_turn_ms,
            self.ct_smc_dp_ms,
            self.endgame_ms,
            self.self_play_games_per_sec,
            self.distill_kl_drift
        )
    }
    pub fn passes(&self) -> bool {
        self.afbs_on_turn_ms < 150.0
            && self.ct_smc_dp_ms < 1.0
            && self.endgame_ms < 100.0
            && self.self_play_games_per_sec > 20.0
            && self.distill_kl_drift < 0.1
    }
}

impl TrainingMetrics {
    pub fn is_improving(&self, prev_loss: f64) -> bool {
        self.total_loss < prev_loss
    }

    pub fn summary(&self) -> String {
        format!(
            "epoch={} loss={:.4} agree={:.2}% games={} elo={:.0}",
            self.epoch,
            self.total_loss,
            self.policy_agreement * 100.0,
            self.games_completed,
            self.elo
        )
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ArenaPromotionDecision {
    Reject,
    NonRegressionOnly,
    StrongPromotion,
}

impl ArenaPromotionDecision {
    pub fn summary(self) -> &'static str {
        match self {
            Self::Reject => "reject",
            Self::NonRegressionOnly => "non_regression_only",
            Self::StrongPromotion => "strong_promotion",
        }
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

#[derive(Debug, Clone)]
pub struct DeltaQArenaEvalOutcome {
    pub paired_result: PairedArenaEvalResult,
    pub lower_confidence_bound_mean_placement: f32,
}

pub fn run_paired_delta_q_arena_confirmation<B: Backend>(
    candidate_model: &HydraModel<B>,
    baseline_model: &HydraModel<B>,
    device: &B::Device,
    config: &PairedArenaEvalConfig,
    temperature: f32,
) -> DeltaQArenaEvalOutcome {
    let mut candidate_placements = Vec::with_capacity(config.min_games);
    let mut baseline_placements = Vec::with_capacity(config.min_games);

    for game_idx in 0..config.min_games {
        let challenger_seat = if config.same_seat_rotation_schedule {
            (game_idx % 4) as u8
        } else {
            0
        };
        let game_seed = if config.same_seeds {
            config.seed.wrapping_add(game_idx as u64)
        } else {
            config
                .seed
                .wrapping_add(game_idx as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        };
        let rng_seed = game_seed ^ 0xA5A5_A5A5_5A5A_5A5A;

        let baseline_seats = [
            baseline_model,
            baseline_model,
            baseline_model,
            baseline_model,
        ];
        let mut candidate_seats = baseline_seats;
        candidate_seats[challenger_seat as usize] = candidate_model;

        let candidate_scores =
            run_mixed_policy_game_scores(game_seed, temperature, rng_seed, candidate_seats, device);
        let baseline_scores = if std::ptr::eq(candidate_model, baseline_model) {
            candidate_scores
        } else {
            run_mixed_policy_game_scores(game_seed, temperature, rng_seed, baseline_seats, device)
        };

        candidate_placements.push(compute_placements(candidate_scores)[challenger_seat as usize]);
        baseline_placements.push(compute_placements(baseline_scores)[challenger_seat as usize]);
    }

    let (lower_ci, upper_ci) = paired_bootstrap_mean_placement_ci(
        &candidate_placements,
        &baseline_placements,
        config.seed,
        1024,
    );

    DeltaQArenaEvalOutcome {
        paired_result: paired_arena_result_from_placements(
            &candidate_placements,
            &baseline_placements,
            upper_ci,
        ),
        lower_confidence_bound_mean_placement: lower_ci,
    }
}

fn paired_bootstrap_mean_placement_ci(
    candidate_placements: &[u8],
    baseline_placements: &[u8],
    seed: u64,
    resamples: usize,
) -> (f32, f32) {
    let count = candidate_placements.len().min(baseline_placements.len());
    if count == 0 {
        return (0.0, 0.0);
    }

    let deltas: Vec<f32> = candidate_placements
        .iter()
        .zip(baseline_placements.iter())
        .take(count)
        .map(|(&candidate, &baseline)| candidate as f32 - baseline as f32)
        .collect();
    if deltas.len() == 1 {
        return (deltas[0], deltas[0]);
    }

    let mut rng = seed.max(1);
    let mut means = Vec::with_capacity(resamples.max(1));
    for _ in 0..resamples.max(1) {
        let mut sample_sum = 0.0f32;
        for _ in 0..deltas.len() {
            let idx = next_bootstrap_index(&mut rng, deltas.len());
            sample_sum += deltas[idx];
        }
        means.push(sample_sum / deltas.len() as f32);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lower_idx = ((means.len() as f32 - 1.0) * 0.025).floor() as usize;
    let upper_idx = ((means.len() as f32 - 1.0) * 0.975).ceil() as usize;
    let upper_idx = upper_idx.min(means.len() - 1);
    (means[lower_idx], means[upper_idx])
}

fn next_bootstrap_index(state: &mut u64, len: usize) -> usize {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as usize) % len.max(1)
}

pub fn paired_arena_result_from_placements(
    candidate_placements: &[u8],
    baseline_placements: &[u8],
    upper_confidence_bound_mean_placement: f32,
) -> PairedArenaEvalResult {
    let candidate_mean_placement = compute_mean_placement(candidate_placements);
    let baseline_mean_placement = compute_mean_placement(baseline_placements);
    let candidate_stable_dan = compute_stable_dan(candidate_mean_placement);
    let baseline_stable_dan = compute_stable_dan(baseline_mean_placement);
    PairedArenaEvalResult {
        candidate_mean_placement,
        baseline_mean_placement,
        delta_mean_placement: candidate_mean_placement - baseline_mean_placement,
        candidate_stable_dan,
        baseline_stable_dan,
        delta_stable_dan: candidate_stable_dan - baseline_stable_dan,
        upper_confidence_bound_mean_placement,
        compared_games: candidate_placements.len().min(baseline_placements.len()),
    }
}

pub fn compute_stable_dan(mean_placement: f32) -> f32 {
    (10.0 - (mean_placement - 1.0) * 4.0).clamp(0.0, 12.0)
}

pub fn avg_stable_dan(placements: &[u8]) -> f32 {
    compute_stable_dan(compute_mean_placement(placements))
}

pub fn placement_histogram(placements: &[u8]) -> [f32; 4] {
    let n = placements.len().max(1) as f32;
    let mut hist = [0.0f32; 4];
    for &p in placements {
        if (p as usize) < 4 {
            hist[p as usize] += 1.0;
        }
    }
    for h in &mut hist {
        *h /= n;
    }
    hist
}

pub fn compute_top2_rate(placements: &[u8]) -> f32 {
    if placements.is_empty() {
        return 0.0;
    }
    placements.iter().filter(|&&p| p <= 1).count() as f32 / placements.len() as f32
}

pub fn compute_4th_rate(placements: &[u8]) -> f32 {
    if placements.is_empty() {
        return 0.0;
    }
    placements.iter().filter(|&&p| p == 3).count() as f32 / placements.len() as f32
}

pub fn compute_mean_placement(placements: &[u8]) -> f32 {
    if placements.is_empty() {
        return 2.5;
    }
    placements.iter().map(|&p| p as f32 + 1.0).sum::<f32>() / placements.len() as f32
}

pub fn compute_win_rate(placements: &[u8]) -> f32 {
    if placements.is_empty() {
        return 0.0;
    }
    placements.iter().filter(|&&p| p == 0).count() as f32 / placements.len() as f32
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
    use crate::model::HydraModelConfig;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

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

    #[test]
    fn placement_histogram_uniform() {
        let placements = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let hist = placement_histogram(&placements);
        for &h in &hist {
            assert!((h - 0.25).abs() < 0.01);
        }
    }

    #[test]
    fn compute_mean_placement_correct() {
        let p = vec![0, 0, 0, 0];
        assert!((compute_mean_placement(&p) - 1.0).abs() < 0.01);
    }

    #[test]
    fn paired_arena_eval_config_summary_is_stable() {
        let cfg = PairedArenaEvalConfig::new();
        assert!(cfg.summary().contains("paired_arena("));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn paired_arena_result_recommends_non_regression_then_strong_promotion() {
        let cfg = PairedArenaEvalConfig::new()
            .with_max_mean_placement_regression(0.025)
            .with_strong_promotion_mean_placement_target(0.0);
        let candidate = vec![0, 1, 1, 2, 2, 2, 0, 1];
        let baseline = vec![1, 2, 2, 3, 2, 3, 1, 2];

        let result = paired_arena_result_from_placements(&candidate, &baseline, 0.02);
        assert_eq!(
            result.recommendation(&cfg),
            ArenaPromotionDecision::NonRegressionOnly
        );

        let strong = paired_arena_result_from_placements(&candidate, &baseline, -0.01);
        assert_eq!(
            strong.recommendation(&cfg),
            ArenaPromotionDecision::StrongPromotion
        );
    }

    #[test]
    fn paired_arena_result_rejects_regression() {
        let cfg = PairedArenaEvalConfig::new().with_max_mean_placement_regression(0.025);
        let candidate = vec![2, 2, 3, 3, 2, 3, 3, 2];
        let baseline = vec![0, 1, 1, 2, 1, 2, 1, 2];
        let result = paired_arena_result_from_placements(&candidate, &baseline, 0.05);
        assert_eq!(result.recommendation(&cfg), ArenaPromotionDecision::Reject);
    }

    #[test]
    fn paired_delta_q_arena_confirmation_is_zero_delta_for_identical_models() {
        let device = Default::default();
        let model = HydraModelConfig::new(1)
            .with_hidden_channels(8)
            .with_se_bottleneck(2)
            .with_num_groups(2)
            .init::<B>(&device);
        let cfg = PairedArenaEvalConfig::new()
            .with_min_games(2)
            .with_seed(123);

        let outcome = run_paired_delta_q_arena_confirmation(&model, &model, &device, &cfg, 1.0);

        assert_eq!(outcome.paired_result.compared_games, 2);
        assert!(outcome.paired_result.delta_mean_placement.abs() < 1e-6);
        assert!(
            outcome
                .paired_result
                .upper_confidence_bound_mean_placement
                .abs()
                < 1e-6
        );
        assert!(outcome.lower_confidence_bound_mean_placement.abs() < 1e-6);
    }

    #[test]
    fn paired_bootstrap_ci_tracks_candidate_regression_direction() {
        let candidate = [2, 2, 3, 3, 2, 3, 3, 2];
        let baseline = [0, 1, 1, 2, 1, 2, 1, 2];

        let (lower, upper) = paired_bootstrap_mean_placement_ci(&candidate, &baseline, 99, 128);

        assert!(lower > 0.0);
        assert!(upper > 0.0);
        let result = paired_arena_result_from_placements(&candidate, &baseline, upper);
        assert!(result.delta_mean_placement > 0.0);
    }

    #[test]
    fn eval_and_benchmark_helpers_cover_defaults_and_thresholds() {
        let cfg = EvalConfig::new();
        assert!(cfg.summary().contains("eval(games=1000, seed=42)"));
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.with_games(12).num_games, 12);

        let bad = EvalConfig::new().with_num_games(0);
        assert_eq!(bad.validate(), Err("num_games must be > 0"));

        let gates = BenchmarkGates {
            afbs_on_turn_ms: 149.0,
            ct_smc_dp_ms: 0.9,
            endgame_ms: 99.0,
            self_play_games_per_sec: 21.0,
            distill_kl_drift: 0.09,
        };
        assert!(gates.passes());
        assert!(gates.summary().contains("afbs=149ms"));

        let failing = BenchmarkGates {
            afbs_on_turn_ms: 151.0,
            ..gates
        };
        assert!(!failing.passes());
    }

    #[test]
    fn eval_result_helper_methods_cover_target_checks_and_summary() {
        let result = EvalResult {
            mean_placement: 1.75,
            stable_dan: 8.5,
            win_rate: 0.25,
            deal_in_rate: 0.10,
            tsumo_rate: 0.15,
        };

        assert!(result.meets_target(8.0));
        assert!(result.is_mortal_level());
        assert!(!result.is_tendan_plus());
        assert!(result.summary().contains("placement=1.75"));

        let from = EvalResult::from_mean_placement(2.0);
        assert_eq!(from.mean_placement, 2.0);
        assert_eq!(from.win_rate, 0.0);
    }

    #[test]
    fn training_metrics_default_summary_and_improvement_behave() {
        let default_metrics = TrainingMetrics::default();
        assert_eq!(default_metrics.elo, 1500.0);
        assert!(default_metrics.summary().contains("epoch=0 loss=0.0000"));
        assert!(!default_metrics.is_improving(-1.0));

        let improving = TrainingMetrics {
            epoch: 3,
            total_loss: 0.5,
            policy_agreement: 0.7,
            value_mse: 0.2,
            games_completed: 99,
            arena_mean_score: 10.0,
            distill_kl: 0.05,
            elo: 1600.0,
        };
        assert!(improving.is_improving(0.6));
        assert!(improving.summary().contains("agree=70.00%"));
    }

    #[test]
    fn paired_result_and_bootstrap_helpers_cover_empty_and_singleton_edges() {
        let cfg = PairedArenaEvalConfig::new();
        let single = paired_arena_result_from_placements(&[0], &[1], 0.1);
        assert_eq!(single.compared_games, 1);
        assert!(single.summary(&cfg).contains("decision="));

        let (lower, upper) = paired_bootstrap_mean_placement_ci(&[], &[], 7, 0);
        assert_eq!((lower, upper), (0.0, 0.0));

        let (lower, upper) = paired_bootstrap_mean_placement_ci(&[0], &[1], 7, 0);
        assert_eq!((lower, upper), (-1.0, -1.0));
    }

    #[test]
    fn placement_histogram_and_rates_cover_empty_and_non_uniform_inputs() {
        let empty_hist = placement_histogram(&[]);
        assert_eq!(empty_hist, [0.0; 4]);
        assert_eq!(compute_top2_rate(&[]), 0.0);
        assert_eq!(compute_4th_rate(&[]), 0.0);
        assert_eq!(compute_win_rate(&[]), 0.0);

        let placements = [0, 1, 1, 3];
        let hist = placement_histogram(&placements);
        assert!((hist[0] - 0.25).abs() < 1e-6);
        assert!((hist[1] - 0.5).abs() < 1e-6);
        assert!((hist[3] - 0.25).abs() < 1e-6);
        assert!((compute_top2_rate(&placements) - 0.75).abs() < 1e-6);
        assert!((compute_4th_rate(&placements) - 0.25).abs() < 1e-6);
        assert!((compute_win_rate(&placements) - 0.25).abs() < 1e-6);
    }
}
