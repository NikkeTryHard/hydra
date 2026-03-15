//! RL-only delta-q validation harness for the live self-play lane.

use std::fmt;

use burn::prelude::Backend;
use hydra_core::action::DISCARD_END;
use hydra_core::arena::{softmax_temperature, TrajectoryStep};

use crate::model::HydraModel;
use crate::selfplay::generate_self_play_batch_source;
use crate::training::exit::{compatible_discard_state, is_hard_state, ExitConfig};
use crate::training::live_exit::{budget_from_legal_count, LiveExitConfig};

/// Aggregated metrics from an observational delta-q validation run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaQValidationReport {
    pub total_states: u64,
    pub compatible_discard_states: u64,
    pub hard_states: u64,
    pub labels_emitted: u64,
    pub labels_rejected: u64,
    pub rejected_incompatible_state: u64,
    pub rejected_too_few_discards: u64,
    pub rejected_not_hard_state: u64,
    pub rejected_other: u64,
    pub coverage_sum: f64,
    pub supported_actions_sum: u64,
    pub root_visits_sum: u64,
    pub masked_abs_sum: f64,
    pub masked_entry_count: u64,
    pub masked_zero_count: u64,
    pub masked_positive_count: u64,
    pub masked_negative_count: u64,
}

impl DeltaQValidationReport {
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
            rejected_other: 0,
            coverage_sum: 0.0,
            supported_actions_sum: 0,
            root_visits_sum: 0,
            masked_abs_sum: 0.0,
            masked_entry_count: 0,
            masked_zero_count: 0,
            masked_positive_count: 0,
            masked_negative_count: 0,
        }
    }

    pub fn merge(&mut self, other: &DeltaQValidationReport) {
        self.total_states += other.total_states;
        self.compatible_discard_states += other.compatible_discard_states;
        self.hard_states += other.hard_states;
        self.labels_emitted += other.labels_emitted;
        self.labels_rejected += other.labels_rejected;
        self.rejected_incompatible_state += other.rejected_incompatible_state;
        self.rejected_too_few_discards += other.rejected_too_few_discards;
        self.rejected_not_hard_state += other.rejected_not_hard_state;
        self.rejected_other += other.rejected_other;
        self.coverage_sum += other.coverage_sum;
        self.supported_actions_sum += other.supported_actions_sum;
        self.root_visits_sum += other.root_visits_sum;
        self.masked_abs_sum += other.masked_abs_sum;
        self.masked_entry_count += other.masked_entry_count;
        self.masked_zero_count += other.masked_zero_count;
        self.masked_positive_count += other.masked_positive_count;
        self.masked_negative_count += other.masked_negative_count;
    }

    pub fn emission_rate(&self) -> f64 {
        ratio_u64(self.labels_emitted, self.total_states)
    }

    pub fn hard_state_rate(&self) -> f64 {
        ratio_u64(self.hard_states, self.total_states)
    }

    pub fn mean_coverage(&self) -> f64 {
        ratio_f64(self.coverage_sum, self.labels_emitted)
    }

    pub fn mean_supported_actions(&self) -> f64 {
        ratio_u64(self.supported_actions_sum, self.labels_emitted)
    }

    pub fn mean_root_visits(&self) -> f64 {
        ratio_u64(self.root_visits_sum, self.labels_emitted)
    }

    pub fn mean_abs(&self) -> f64 {
        ratio_f64(self.masked_abs_sum, self.masked_entry_count)
    }

    pub fn positive_fraction(&self) -> f64 {
        ratio_u64(self.masked_positive_count, self.masked_entry_count)
    }

    pub fn negative_fraction(&self) -> f64 {
        ratio_u64(self.masked_negative_count, self.masked_entry_count)
    }

    pub fn zero_fraction(&self) -> f64 {
        ratio_u64(self.masked_zero_count, self.masked_entry_count)
    }
}

impl Default for DeltaQValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DeltaQValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== DeltaQ Validation Report ===")?;
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
        writeln!(f, "  Other:               {}", self.rejected_other)?;
        writeln!(f, "--- Label structure ---")?;
        writeln!(f, "  Mean coverage:       {:.3}", self.mean_coverage())?;
        writeln!(
            f,
            "  Mean supported acts: {:.1}",
            self.mean_supported_actions()
        )?;
        writeln!(f, "  Mean root visits:    {:.0}", self.mean_root_visits())?;
        writeln!(f, "  Mean |delta_q|:      {:.4}", self.mean_abs())?;
        writeln!(f, "  Positive frac:       {:.3}", self.positive_fraction())?;
        writeln!(f, "  Negative frac:       {:.3}", self.negative_fraction())?;
        writeln!(f, "  Zero frac:           {:.3}", self.zero_fraction())?;
        Ok(())
    }
}

/// Thresholds for the structural delta-q validation decision.
#[derive(Debug, Clone)]
pub struct DeltaQValidationThresholds {
    pub min_emission_rate: f64,
    pub min_mean_coverage: f64,
    pub min_mean_supported_actions: f64,
    pub min_sample_size: u64,
}

impl Default for DeltaQValidationThresholds {
    fn default() -> Self {
        Self {
            min_emission_rate: 0.01,
            min_mean_coverage: 0.70,
            min_mean_supported_actions: 3.0,
            min_sample_size: 1000,
        }
    }
}

/// Result of evaluating a delta-q validation report against thresholds.
#[derive(Debug, Clone)]
pub struct DeltaQValidationResult {
    pub passed: bool,
    pub criteria: Vec<DeltaQCriterionResult>,
}

/// One pass/fail criterion from the delta-q validation result.
#[derive(Debug, Clone)]
pub struct DeltaQCriterionResult {
    pub name: String,
    pub measured: f64,
    pub threshold: f64,
    pub passed: bool,
    pub direction: ThresholdDirection,
}

/// Direction for threshold comparisons.
#[derive(Debug, Clone, Copy)]
pub enum ThresholdDirection {
    Min,
    Max,
}

impl fmt::Display for DeltaQValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== DeltaQ Validation Result: {} ===",
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

/// Evaluates a delta-q report against structural thresholds.
pub fn evaluate_report(
    report: &DeltaQValidationReport,
    thresholds: &DeltaQValidationThresholds,
) -> DeltaQValidationResult {
    let mut criteria = Vec::with_capacity(4);

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

    if report.labels_emitted == 0 {
        for criterion in &mut criteria {
            if matches!(
                criterion.name.as_str(),
                "mean_coverage" | "mean_supported_actions"
            ) {
                criterion.passed = false;
            }
        }
    }

    DeltaQValidationResult {
        passed: criteria.iter().all(|criterion| criterion.passed),
        criteria,
    }
}

/// Collects observational delta-q metrics for one stored self-play step.
pub fn collect_validation_metrics_for_step<B: Backend>(
    step: &TrajectoryStep,
    model: &HydraModel<B>,
    device: &B::Device,
    cfg: &ExitConfig,
    report: &mut DeltaQValidationReport,
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

    let Some(label) = step.delta_q_label else {
        report.labels_rejected += 1;
        report.rejected_other += 1;
        return;
    };

    report.labels_emitted += 1;
    let mut supported = 0usize;
    for action in 0..=DISCARD_END as usize {
        if label.mask[action] <= 0.0 {
            continue;
        }
        supported += 1;
        let value = label.target[action] as f64;
        report.masked_abs_sum += value.abs();
        report.masked_entry_count += 1;
        if value > 0.0 {
            report.masked_positive_count += 1;
        } else if value < 0.0 {
            report.masked_negative_count += 1;
        } else {
            report.masked_zero_count += 1;
        }
    }
    report.supported_actions_sum += supported as u64;
    report.coverage_sum += supported as f64 / legal_discards.len() as f64;
    report.root_visits_sum += budget_from_legal_count(cfg, legal_discards.len()) as u64;
}

/// Runs an observational delta-q validation pass over self-play trajectories.
pub fn run_delta_q_validation<B: Backend>(
    game_seeds: &[u64],
    temperature: f32,
    rng_seed: u64,
    model: &HydraModel<B>,
    device: &B::Device,
    exit_config: ExitConfig,
) -> DeltaQValidationReport {
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

    let mut report = DeltaQValidationReport::new();
    for trajectory in &source.trajectories {
        let mut trajectory_report = DeltaQValidationReport::new();
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
    criteria: &mut Vec<DeltaQCriterionResult>,
    name: &str,
    measured: f64,
    threshold: f64,
) {
    criteria.push(DeltaQCriterionResult {
        name: name.to_string(),
        measured,
        threshold,
        passed: measured >= threshold,
        direction: ThresholdDirection::Min,
    });
}

fn legal_discard_actions(step: &TrajectoryStep) -> Vec<usize> {
    (0..=DISCARD_END as usize)
        .filter(|&action| step.legal_mask[action])
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn passing_report() -> DeltaQValidationReport {
        DeltaQValidationReport {
            total_states: 2_000,
            compatible_discard_states: 1_500,
            hard_states: 300,
            labels_emitted: 100,
            labels_rejected: 1_900,
            rejected_incompatible_state: 500,
            rejected_too_few_discards: 200,
            rejected_not_hard_state: 900,
            rejected_other: 300,
            coverage_sum: 75.0,
            supported_actions_sum: 350,
            root_visits_sum: 6_400,
            masked_abs_sum: 42.0,
            masked_entry_count: 350,
            masked_zero_count: 20,
            masked_positive_count: 170,
            masked_negative_count: 160,
        }
    }

    fn criterion<'a>(result: &'a DeltaQValidationResult, name: &str) -> &'a DeltaQCriterionResult {
        result
            .criteria
            .iter()
            .find(|criterion| criterion.name == name)
            .unwrap_or_else(|| panic!("missing criterion: {name}"))
    }

    #[test]
    fn test_empty_report_defaults() {
        let report = DeltaQValidationReport::new();
        assert_eq!(report.total_states, 0);
        assert_eq!(report.labels_emitted, 0);
        assert_eq!(report.emission_rate(), 0.0);
        assert_eq!(report.mean_coverage(), 0.0);
        assert_eq!(report.mean_supported_actions(), 0.0);
        assert_eq!(report.mean_abs(), 0.0);
        assert_eq!(report.positive_fraction(), 0.0);
        assert_eq!(report.negative_fraction(), 0.0);
        assert_eq!(report.zero_fraction(), 0.0);
    }

    #[test]
    fn test_report_merge() {
        let mut lhs = DeltaQValidationReport {
            total_states: 10,
            compatible_discard_states: 7,
            hard_states: 4,
            labels_emitted: 2,
            labels_rejected: 8,
            rejected_incompatible_state: 1,
            rejected_too_few_discards: 2,
            rejected_not_hard_state: 3,
            rejected_other: 4,
            coverage_sum: 1.2,
            supported_actions_sum: 8,
            root_visits_sum: 9,
            masked_abs_sum: 1.5,
            masked_entry_count: 8,
            masked_zero_count: 1,
            masked_positive_count: 4,
            masked_negative_count: 3,
        };
        let rhs = DeltaQValidationReport {
            total_states: 5,
            compatible_discard_states: 3,
            hard_states: 2,
            labels_emitted: 1,
            labels_rejected: 4,
            rejected_incompatible_state: 2,
            rejected_too_few_discards: 3,
            rejected_not_hard_state: 4,
            rejected_other: 5,
            coverage_sum: 0.8,
            supported_actions_sum: 4,
            root_visits_sum: 11,
            masked_abs_sum: 0.7,
            masked_entry_count: 4,
            masked_zero_count: 0,
            masked_positive_count: 2,
            masked_negative_count: 2,
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
        assert_eq!(lhs.rejected_other, 9);
        assert!((lhs.coverage_sum - 2.0).abs() < 1e-9);
        assert_eq!(lhs.supported_actions_sum, 12);
        assert_eq!(lhs.root_visits_sum, 20);
        assert!((lhs.masked_abs_sum - 2.2).abs() < 1e-9);
        assert_eq!(lhs.masked_entry_count, 12);
        assert_eq!(lhs.masked_zero_count, 1);
        assert_eq!(lhs.masked_positive_count, 6);
        assert_eq!(lhs.masked_negative_count, 5);
    }

    #[test]
    fn test_evaluate_report_all_pass() {
        let report = passing_report();
        let result = evaluate_report(&report, &DeltaQValidationThresholds::default());
        assert!(result.passed);
        assert!(result.criteria.iter().all(|criterion| criterion.passed));
    }

    #[test]
    fn test_evaluate_report_low_emission_fails() {
        let mut report = passing_report();
        report.labels_emitted = 10;
        report.coverage_sum = 8.0;
        report.supported_actions_sum = 40;

        let result = evaluate_report(&report, &DeltaQValidationThresholds::default());
        assert!(!result.passed);
        assert!(!criterion(&result, "emission_rate").passed);
    }

    #[test]
    fn test_evaluate_report_insufficient_samples_fails() {
        let mut report = passing_report();
        report.total_states = 999;

        let result = evaluate_report(&report, &DeltaQValidationThresholds::default());
        assert!(!result.passed);
        assert!(!criterion(&result, "sample_size").passed);
    }

    #[test]
    fn test_evaluate_report_no_labels_fails() {
        let report = DeltaQValidationReport {
            total_states: 2_000,
            compatible_discard_states: 1_000,
            hard_states: 100,
            labels_emitted: 0,
            labels_rejected: 2_000,
            ..DeltaQValidationReport::default()
        };

        let result = evaluate_report(&report, &DeltaQValidationThresholds::default());
        assert!(!result.passed);
        assert!(!criterion(&result, "emission_rate").passed);
        assert!(!criterion(&result, "mean_coverage").passed);
        assert!(!criterion(&result, "mean_supported_actions").passed);
    }

    #[test]
    fn test_display_formatting() {
        let report = passing_report();
        let report_text = format!("{report}");
        let result_text = format!(
            "{}",
            evaluate_report(&report, &DeltaQValidationThresholds::default())
        );

        assert!(report_text.contains("DeltaQ Validation Report"));
        assert!(report_text.contains("Mean |delta_q|"));
        assert!(result_text.contains("DeltaQ Validation Result: PASS"));
        assert!(result_text.contains("sample_size"));
    }

    #[test]
    #[ignore]
    fn run_shadow_delta_q_validation_harness() {
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

        let report = run_delta_q_validation(&seeds, 1.0, 42, &model, &device, exit_config);
        let thresholds = DeltaQValidationThresholds::default();
        let result = evaluate_report(&report, &thresholds);

        eprintln!("{report}");
        eprintln!("{result}");
    }
}
