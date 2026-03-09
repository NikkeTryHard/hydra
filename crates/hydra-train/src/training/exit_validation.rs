//! ExIt validation harness for measuring producer label quality.
//!
//! Implements the "validation matrix" from Agent 22 doctrine: the live
//! ExIt producer must prove it generates useful, sane labels before
//! enablement. This harness runs the producer in shadow mode through the
//! existing self-play pipeline and collects metrics for pass/fail evaluation.

use std::fmt;

use burn::prelude::Backend;
use hydra_core::action::{DISCARD_END, HYDRA_ACTION_SPACE};
use hydra_core::arena::{softmax_temperature, TrajectoryStep};

use crate::model::HydraModel;
use crate::selfplay::generate_self_play_batch_source;
use crate::training::exit::{compatible_discard_state, is_hard_state, ExitConfig};
use crate::training::live_exit::{budget_from_legal_count, LiveExitConfig};

/// Aggregated metrics from a shadow ExIt validation run.
///
/// Each field corresponds to one criterion from the Agent 22/9/16
/// blueprint. The harness collects these by running the live producer
/// on self-play states without using the labels for training.
#[derive(Debug, Clone)]
pub struct ExitValidationReport {
    /// Total decision states examined.
    pub total_states: u64,
    /// States that passed the compatible-discard-only gate.
    pub compatible_discard_states: u64,
    /// States that passed the hard-state gate (top2_policy_gap < 0.10).
    pub hard_states: u64,
    /// States where the producer emitted a real label (not None).
    pub labels_emitted: u64,
    /// States where the producer returned None (any gate failed).
    pub labels_rejected: u64,
    /// Rejected because state was not a compatible discard state.
    pub rejected_incompatible_state: u64,
    /// Rejected because fewer than 2 legal discards.
    pub rejected_too_few_discards: u64,
    /// Rejected because state was not hard (top-2 gap >= threshold).
    pub rejected_not_hard_state: u64,
    /// Rejected by child observation failure.
    pub rejected_child_obs_failure: u64,
    /// Rejected by coverage gate (< 0.60).
    pub rejected_low_coverage: u64,
    /// Rejected by KL safety valve.
    pub rejected_kl_safety: u64,
    /// Rejected by other or currently un-attributed gates.
    pub rejected_other: u64,
    /// Sum of coverage values across emitted labels.
    pub coverage_sum: f64,
    /// Sum of supported action counts across emitted labels.
    pub supported_actions_sum: u64,
    /// Sum of root visit counts across emitted labels.
    pub root_visits_sum: u64,
    /// Count of emitted labels where top-1 action matches base policy top-1.
    pub top1_agreement_count: u64,
    /// Sum of KL(base || exit) across emitted labels.
    pub kl_sum: f64,
}

impl ExitValidationReport {
    /// Creates an empty report with all counters at zero.
    pub fn new() -> Self {
        Self {
            total_states: 0,
            compatible_discard_states: 0,
            hard_states: 0,
            labels_emitted: 0,
            labels_rejected: 0,
            rejected_incompatible_state: 0,
            rejected_too_few_discards: 0,
            rejected_not_hard_state: 0,
            rejected_child_obs_failure: 0,
            rejected_low_coverage: 0,
            rejected_kl_safety: 0,
            rejected_other: 0,
            coverage_sum: 0.0,
            supported_actions_sum: 0,
            root_visits_sum: 0,
            top1_agreement_count: 0,
            kl_sum: 0.0,
        }
    }

    /// Merges another report into this one.
    pub fn merge(&mut self, other: &ExitValidationReport) {
        self.total_states += other.total_states;
        self.compatible_discard_states += other.compatible_discard_states;
        self.hard_states += other.hard_states;
        self.labels_emitted += other.labels_emitted;
        self.labels_rejected += other.labels_rejected;
        self.rejected_incompatible_state += other.rejected_incompatible_state;
        self.rejected_too_few_discards += other.rejected_too_few_discards;
        self.rejected_not_hard_state += other.rejected_not_hard_state;
        self.rejected_child_obs_failure += other.rejected_child_obs_failure;
        self.rejected_low_coverage += other.rejected_low_coverage;
        self.rejected_kl_safety += other.rejected_kl_safety;
        self.rejected_other += other.rejected_other;
        self.coverage_sum += other.coverage_sum;
        self.supported_actions_sum += other.supported_actions_sum;
        self.root_visits_sum += other.root_visits_sum;
        self.top1_agreement_count += other.top1_agreement_count;
        self.kl_sum += other.kl_sum;
    }

    /// Returns the label emission rate.
    pub fn emission_rate(&self) -> f64 {
        ratio_u64(self.labels_emitted, self.total_states)
    }

    /// Returns the hard-state rate.
    pub fn hard_state_rate(&self) -> f64 {
        ratio_u64(self.hard_states, self.total_states)
    }

    /// Returns the mean coverage across emitted labels.
    pub fn mean_coverage(&self) -> f64 {
        ratio_f64(self.coverage_sum, self.labels_emitted)
    }

    /// Returns the mean supported actions across emitted labels.
    pub fn mean_supported_actions(&self) -> f64 {
        ratio_u64(self.supported_actions_sum, self.labels_emitted)
    }

    /// Returns the mean root visits across emitted labels.
    pub fn mean_root_visits(&self) -> f64 {
        ratio_u64(self.root_visits_sum, self.labels_emitted)
    }

    /// Returns the top-1 action agreement rate.
    pub fn top1_agreement_rate(&self) -> f64 {
        ratio_u64(self.top1_agreement_count, self.labels_emitted)
    }

    /// Returns the mean KL divergence between base policy and ExIt labels.
    pub fn mean_kl(&self) -> f64 {
        ratio_f64(self.kl_sum, self.labels_emitted)
    }
}

impl Default for ExitValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ExitValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== ExIt Validation Report ===")?;
        writeln!(f, "States examined:       {}", self.total_states)?;
        writeln!(
            f,
            "Compatible discard:    {} ({:.1}%)",
            self.compatible_discard_states,
            ratio_u64(self.compatible_discard_states, self.total_states) * 100.0
        )?;
        writeln!(
            f,
            "Hard states:           {} ({:.1}%)",
            self.hard_states,
            self.hard_state_rate() * 100.0
        )?;
        writeln!(
            f,
            "Labels emitted:        {} ({:.2}%)",
            self.labels_emitted,
            self.emission_rate() * 100.0
        )?;
        writeln!(f, "Labels rejected:       {}", self.labels_rejected)?;
        writeln!(f, "--- Rejection breakdown ---")?;
        writeln!(
            f,
            "  Incompatible state:  {}",
            self.rejected_incompatible_state
        )?;
        writeln!(
            f,
            "  Too few discards:    {}",
            self.rejected_too_few_discards
        )?;
        writeln!(f, "  Not hard state:      {}", self.rejected_not_hard_state)?;
        writeln!(
            f,
            "  Child obs failure:   {}",
            self.rejected_child_obs_failure
        )?;
        writeln!(f, "  Low coverage:        {}", self.rejected_low_coverage)?;
        writeln!(f, "  KL safety valve:     {}", self.rejected_kl_safety)?;
        writeln!(f, "  Other:               {}", self.rejected_other)?;
        writeln!(f, "--- Label quality ---")?;
        writeln!(f, "  Mean coverage:       {:.3}", self.mean_coverage())?;
        writeln!(
            f,
            "  Mean supported acts: {:.1}",
            self.mean_supported_actions()
        )?;
        writeln!(f, "  Mean root visits:    {:.0}", self.mean_root_visits())?;
        writeln!(
            f,
            "  Top-1 agreement:     {:.1}%",
            self.top1_agreement_rate() * 100.0
        )?;
        writeln!(f, "  Mean KL:             {:.4}", self.mean_kl())?;
        Ok(())
    }
}

/// Thresholds for the ExIt validation pass/fail decision.
///
/// Derived from Agent 22 + Agent 9 + Agent 16 blueprints. These are the
/// minimum requirements before the producer can be enabled.
#[derive(Debug, Clone)]
pub struct ExitValidationThresholds {
    /// Minimum fraction of total states that must emit labels.
    pub min_emission_rate: f64,
    /// Minimum mean coverage across emitted labels.
    pub min_mean_coverage: f64,
    /// Minimum mean supported actions per emitted label.
    pub min_mean_supported_actions: f64,
    /// Maximum mean KL divergence between base policy and ExIt labels.
    pub max_mean_kl: f64,
    /// Minimum top-1 agreement rate.
    pub min_top1_agreement: f64,
    /// Minimum total states examined for the report to be meaningful.
    pub min_sample_size: u64,
}

impl Default for ExitValidationThresholds {
    fn default() -> Self {
        Self {
            min_emission_rate: 0.01,
            min_mean_coverage: 0.70,
            min_mean_supported_actions: 3.0,
            max_mean_kl: 0.05,
            min_top1_agreement: 0.95,
            min_sample_size: 1000,
        }
    }
}

/// Result of evaluating an [`ExitValidationReport`] against thresholds.
#[derive(Debug, Clone)]
pub struct ExitValidationResult {
    /// Whether all criteria passed.
    pub passed: bool,
    /// Per-criterion pass or fail details.
    pub criteria: Vec<ExitCriterionResult>,
}

/// A single pass/fail criterion with its name, value, and threshold.
#[derive(Debug, Clone)]
pub struct ExitCriterionResult {
    /// Human-readable criterion name.
    pub name: String,
    /// Measured value from the report.
    pub measured: f64,
    /// Required threshold.
    pub threshold: f64,
    /// Whether this criterion passed.
    pub passed: bool,
    /// Direction of the threshold comparison.
    pub direction: ThresholdDirection,
}

/// Direction for threshold comparisons.
#[derive(Debug, Clone, Copy)]
pub enum ThresholdDirection {
    /// Measured must be greater than or equal to the threshold.
    Min,
    /// Measured must be less than or equal to the threshold.
    Max,
}

impl fmt::Display for ExitValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== ExIt Validation Result: {} ===",
            if self.passed { "PASS" } else { "FAIL" }
        )?;
        for criterion in &self.criteria {
            let status = if criterion.passed { "PASS" } else { "FAIL" };
            let direction = match criterion.direction {
                ThresholdDirection::Min => ">=",
                ThresholdDirection::Max => "<=",
            };
            writeln!(
                f,
                "  [{}] {}: {:.4} ({} {:.4})",
                status, criterion.name, criterion.measured, direction, criterion.threshold
            )?;
        }
        Ok(())
    }
}

/// Evaluates an [`ExitValidationReport`] against the given thresholds.
///
/// Returns a detailed [`ExitValidationResult`] with per-criterion pass/fail.
/// The overall result passes only if all criteria pass.
pub fn evaluate_report(
    report: &ExitValidationReport,
    thresholds: &ExitValidationThresholds,
) -> ExitValidationResult {
    let mut criteria = Vec::with_capacity(6);

    push_min_criterion(
        &mut criteria,
        "sample_size",
        report.total_states as f64,
        thresholds.min_sample_size as f64,
    );
    push_min_criterion(
        &mut criteria,
        "emission_rate",
        report.emission_rate(),
        thresholds.min_emission_rate,
    );
    push_min_criterion(
        &mut criteria,
        "mean_coverage",
        report.mean_coverage(),
        thresholds.min_mean_coverage,
    );
    push_min_criterion(
        &mut criteria,
        "mean_supported_actions",
        report.mean_supported_actions(),
        thresholds.min_mean_supported_actions,
    );
    push_max_criterion(
        &mut criteria,
        "mean_kl",
        report.mean_kl(),
        thresholds.max_mean_kl,
    );
    push_min_criterion(
        &mut criteria,
        "top1_agreement",
        report.top1_agreement_rate(),
        thresholds.min_top1_agreement,
    );

    if report.labels_emitted == 0 {
        for criterion in &mut criteria {
            if matches!(
                criterion.name.as_str(),
                "mean_coverage" | "mean_supported_actions" | "mean_kl" | "top1_agreement"
            ) {
                criterion.passed = false;
            }
        }
    }

    let passed = criteria.iter().all(|criterion| criterion.passed);
    ExitValidationResult { passed, criteria }
}

/// Collects validation metrics for a single stored self-play step.
///
/// This is the step-level shadow collector for the v1 harness. It uses the
/// already-recorded trajectory step plus fresh model inference on `step.obs`
/// to reconstruct the producer gates and measure label quality. When the
/// stored `exit_label` is `None` after the reconstructable gates pass, the
/// rejection is attributed to `rejected_other` because v1 does not instrument
/// the deeper producer gates separately.
pub fn collect_validation_metrics_for_step<B: Backend>(
    step: &TrajectoryStep,
    model: &HydraModel<B>,
    device: &B::Device,
    cfg: &ExitConfig,
    report: &mut ExitValidationReport,
) {
    report.total_states += 1;

    let legal_f32 = step
        .legal_mask
        .map(|is_legal| if is_legal { 1.0 } else { 0.0 });
    if !compatible_discard_state(&legal_f32) {
        report.labels_rejected += 1;
        report.rejected_incompatible_state += 1;
        return;
    }
    report.compatible_discard_states += 1;

    let legal_discards = legal_discard_actions(step);
    if legal_discards.len() < 2 {
        report.labels_rejected += 1;
        report.rejected_too_few_discards += 1;
        return;
    }

    let (policy_logits, _) = model.policy_value_cpu(&step.obs, device);
    let base_pi = softmax_temperature(&policy_logits, &step.legal_mask, 1.0);
    let hard_slice: Vec<f32> = legal_discards
        .iter()
        .map(|&action| base_pi[action])
        .collect();
    if !is_hard_state(&hard_slice, cfg.hard_state_threshold) {
        report.labels_rejected += 1;
        report.rejected_not_hard_state += 1;
        return;
    }
    report.hard_states += 1;

    let Some(label) = step.exit_label else {
        report.labels_rejected += 1;
        report.rejected_other += 1;
        return;
    };

    report.labels_emitted += 1;

    let supported = label.mask[..=DISCARD_END as usize]
        .iter()
        .filter(|&&mask_value| mask_value > 0.0)
        .count();
    report.supported_actions_sum += supported as u64;
    report.coverage_sum += supported as f64 / legal_discards.len() as f64;
    report.root_visits_sum += budget_from_legal_count(cfg, legal_discards.len()) as u64;

    let base_top1 = top1_index(&base_pi, &legal_discards);
    let exit_top1 = top1_index(&label.target, &legal_discards);
    if base_top1 == exit_top1 {
        report.top1_agreement_count += 1;
    }

    report.kl_sum += kl_divergence(&base_pi, &label.target, &label.mask);
}

/// Runs shadow ExIt validation over self-play games.
///
/// The live producer is force-enabled for data collection, but the produced
/// labels are only inspected on the returned trajectories and are not used for
/// training. This keeps the harness fully observational while reusing the
/// existing self-play infrastructure.
pub fn run_exit_validation<B: Backend>(
    game_seeds: &[u64],
    temperature: f32,
    rng_seed: u64,
    model: &HydraModel<B>,
    device: &B::Device,
    exit_config: ExitConfig,
) -> ExitValidationReport {
    let source = generate_self_play_batch_source(
        game_seeds,
        temperature,
        rng_seed,
        model,
        device,
        LiveExitConfig {
            enabled: true,
            exit_config: exit_config.clone(),
        },
    );

    let mut report = ExitValidationReport::new();
    for trajectory in &source.trajectories {
        let mut trajectory_report = ExitValidationReport::new();
        for step in &trajectory.steps {
            collect_validation_metrics_for_step(
                step,
                model,
                device,
                &exit_config,
                &mut trajectory_report,
            );
        }
        report.merge(&trajectory_report);
    }

    report
}

fn push_min_criterion(
    criteria: &mut Vec<ExitCriterionResult>,
    name: &str,
    measured: f64,
    threshold: f64,
) {
    criteria.push(ExitCriterionResult {
        name: name.to_string(),
        measured,
        threshold,
        passed: measured >= threshold,
        direction: ThresholdDirection::Min,
    });
}

fn push_max_criterion(
    criteria: &mut Vec<ExitCriterionResult>,
    name: &str,
    measured: f64,
    threshold: f64,
) {
    criteria.push(ExitCriterionResult {
        name: name.to_string(),
        measured,
        threshold,
        passed: measured <= threshold,
        direction: ThresholdDirection::Max,
    });
}

fn ratio_u64(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn ratio_f64(numerator: f64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator / denominator as f64
    }
}

fn legal_discard_actions(step: &TrajectoryStep) -> Vec<usize> {
    (0..=DISCARD_END as usize)
        .filter(|&action| step.legal_mask[action])
        .collect()
}

fn top1_index(values: &[f32; HYDRA_ACTION_SPACE], actions: &[usize]) -> usize {
    let mut best_action = 0usize;
    let mut best_value = f32::NEG_INFINITY;

    for &action in actions {
        let value = values[action];
        if value > best_value {
            best_value = value;
            best_action = action;
        }
    }

    best_action
}

fn kl_divergence(
    base_pi: &[f32; HYDRA_ACTION_SPACE],
    exit_target: &[f32; HYDRA_ACTION_SPACE],
    exit_mask: &[f32; HYDRA_ACTION_SPACE],
) -> f64 {
    let mut kl = 0.0f64;
    for action in 0..HYDRA_ACTION_SPACE {
        let p = base_pi[action] as f64;
        let q = exit_target[action] as f64;
        if exit_mask[action] > 0.0 && p > 1e-8 && q > 1e-8 {
            kl += p * (p.ln() - q.ln());
        }
    }
    kl.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn passing_report() -> ExitValidationReport {
        ExitValidationReport {
            total_states: 2_000,
            compatible_discard_states: 1_500,
            hard_states: 300,
            labels_emitted: 100,
            labels_rejected: 1_900,
            rejected_incompatible_state: 500,
            rejected_too_few_discards: 200,
            rejected_not_hard_state: 900,
            rejected_child_obs_failure: 0,
            rejected_low_coverage: 0,
            rejected_kl_safety: 0,
            rejected_other: 300,
            coverage_sum: 75.0,
            supported_actions_sum: 350,
            root_visits_sum: 6_400,
            top1_agreement_count: 97,
            kl_sum: 3.0,
        }
    }

    fn criterion<'a>(result: &'a ExitValidationResult, name: &str) -> &'a ExitCriterionResult {
        result
            .criteria
            .iter()
            .find(|criterion| criterion.name == name)
            .unwrap_or_else(|| panic!("missing criterion: {name}"))
    }

    #[test]
    fn test_empty_report_defaults() {
        let report = ExitValidationReport::new();

        assert_eq!(report.total_states, 0);
        assert_eq!(report.labels_emitted, 0);
        assert_eq!(report.labels_rejected, 0);
        assert_eq!(report.emission_rate(), 0.0);
        assert_eq!(report.hard_state_rate(), 0.0);
        assert_eq!(report.mean_coverage(), 0.0);
        assert_eq!(report.mean_supported_actions(), 0.0);
        assert_eq!(report.mean_root_visits(), 0.0);
        assert_eq!(report.top1_agreement_rate(), 0.0);
        assert_eq!(report.mean_kl(), 0.0);
    }

    #[test]
    fn test_report_merge() {
        let mut lhs = ExitValidationReport {
            total_states: 10,
            compatible_discard_states: 7,
            hard_states: 4,
            labels_emitted: 2,
            labels_rejected: 8,
            rejected_incompatible_state: 1,
            rejected_too_few_discards: 2,
            rejected_not_hard_state: 3,
            rejected_child_obs_failure: 4,
            rejected_low_coverage: 5,
            rejected_kl_safety: 6,
            rejected_other: 7,
            coverage_sum: 1.2,
            supported_actions_sum: 8,
            root_visits_sum: 9,
            top1_agreement_count: 1,
            kl_sum: 0.3,
        };
        let rhs = ExitValidationReport {
            total_states: 5,
            compatible_discard_states: 3,
            hard_states: 2,
            labels_emitted: 1,
            labels_rejected: 4,
            rejected_incompatible_state: 2,
            rejected_too_few_discards: 3,
            rejected_not_hard_state: 4,
            rejected_child_obs_failure: 5,
            rejected_low_coverage: 6,
            rejected_kl_safety: 7,
            rejected_other: 8,
            coverage_sum: 0.8,
            supported_actions_sum: 4,
            root_visits_sum: 11,
            top1_agreement_count: 1,
            kl_sum: 0.2,
        };

        lhs.merge(&rhs);

        assert_eq!(lhs.total_states, 15);
        assert_eq!(lhs.compatible_discard_states, 10);
        assert_eq!(lhs.hard_states, 6);
        assert_eq!(lhs.labels_emitted, 3);
        assert_eq!(lhs.labels_rejected, 12);
        assert_eq!(lhs.rejected_incompatible_state, 3);
        assert_eq!(lhs.rejected_too_few_discards, 5);
        assert_eq!(lhs.rejected_not_hard_state, 7);
        assert_eq!(lhs.rejected_child_obs_failure, 9);
        assert_eq!(lhs.rejected_low_coverage, 11);
        assert_eq!(lhs.rejected_kl_safety, 13);
        assert_eq!(lhs.rejected_other, 15);
        assert!((lhs.coverage_sum - 2.0).abs() < 1e-9);
        assert_eq!(lhs.supported_actions_sum, 12);
        assert_eq!(lhs.root_visits_sum, 20);
        assert_eq!(lhs.top1_agreement_count, 2);
        assert!((lhs.kl_sum - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_emission_rate_calculation() {
        let report = ExitValidationReport {
            total_states: 80,
            labels_emitted: 20,
            ..ExitValidationReport::default()
        };

        assert!((report.emission_rate() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_mean_coverage_calculation() {
        let report = ExitValidationReport {
            labels_emitted: 4,
            coverage_sum: 3.0,
            ..ExitValidationReport::default()
        };

        assert!((report.mean_coverage() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_evaluate_report_all_pass() {
        let report = passing_report();
        let result = evaluate_report(&report, &ExitValidationThresholds::default());

        assert!(result.passed);
        assert!(result.criteria.iter().all(|criterion| criterion.passed));
    }

    #[test]
    fn test_evaluate_report_low_emission_fails() {
        let mut report = passing_report();
        report.labels_emitted = 10;
        report.coverage_sum = 8.0;
        report.supported_actions_sum = 40;
        report.root_visits_sum = 640;
        report.top1_agreement_count = 10;
        report.kl_sum = 0.1;

        let result = evaluate_report(&report, &ExitValidationThresholds::default());

        assert!(!result.passed);
        assert!(!criterion(&result, "emission_rate").passed);
    }

    #[test]
    fn test_evaluate_report_high_kl_fails() {
        let mut report = passing_report();
        report.kl_sum = 6.0;

        let result = evaluate_report(&report, &ExitValidationThresholds::default());

        assert!(!result.passed);
        assert!(!criterion(&result, "mean_kl").passed);
    }

    #[test]
    fn test_evaluate_report_low_agreement_fails() {
        let mut report = passing_report();
        report.top1_agreement_count = 80;

        let result = evaluate_report(&report, &ExitValidationThresholds::default());

        assert!(!result.passed);
        assert!(!criterion(&result, "top1_agreement").passed);
    }

    #[test]
    fn test_evaluate_report_insufficient_samples_fails() {
        let mut report = passing_report();
        report.total_states = 999;

        let result = evaluate_report(&report, &ExitValidationThresholds::default());

        assert!(!result.passed);
        assert!(!criterion(&result, "sample_size").passed);
    }

    #[test]
    fn test_evaluate_report_no_labels_fails() {
        let report = ExitValidationReport {
            total_states: 2_000,
            compatible_discard_states: 1_000,
            hard_states: 100,
            labels_emitted: 0,
            labels_rejected: 2_000,
            ..ExitValidationReport::default()
        };

        let result = evaluate_report(&report, &ExitValidationThresholds::default());

        assert!(!result.passed);
        assert!(!criterion(&result, "emission_rate").passed);
        assert!(!criterion(&result, "mean_coverage").passed);
        assert!(!criterion(&result, "mean_supported_actions").passed);
        assert!(!criterion(&result, "mean_kl").passed);
        assert!(!criterion(&result, "top1_agreement").passed);
    }

    #[test]
    fn test_display_formatting() {
        let report = passing_report();
        let report_text = format!("{report}");
        let result_text = format!(
            "{}",
            evaluate_report(&report, &ExitValidationThresholds::default())
        );

        assert!(report_text.contains("ExIt Validation Report"));
        assert!(report_text.contains("Mean KL"));
        assert!(result_text.contains("ExIt Validation Result: PASS"));
        assert!(result_text.contains("sample_size"));
    }

    #[test]
    #[ignore]
    fn run_shadow_exit_validation_harness() {
        use crate::model::HydraModelConfig;
        use burn::backend::NdArray;

        type B = NdArray<f32>;
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<B>(&device);

        let seeds: Vec<u64> = (0..20).collect();
        let exit_config = crate::training::exit::ExitConfig::default_phase3();

        let report = run_exit_validation(&seeds, 1.0, 42, &model, &device, exit_config);
        let thresholds = ExitValidationThresholds::default();
        let result = evaluate_report(&report, &thresholds);

        eprintln!("{report}");
        eprintln!("{result}");
    }
}
