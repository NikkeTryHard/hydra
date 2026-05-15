//! RL-only delta-q validation harness for the live self-play lane.

use std::fmt;

use burn::prelude::Backend;
use hydra_core::action::DISCARD_END;
use hydra_core::arena::TrajectoryStep;

use hydra_model::model::HydraModel;

use crate::exit::ExitConfig;
use crate::live_exit::budget_from_legal_count;
use crate::validation_common::{
    CommonGateOutcome, evaluate_common_validation_gate, ratio_f64, ratio_u64,
};

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
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
    cfg: &ExitConfig,
    report: &mut DeltaQValidationReport,
) {
    report.total_states += 1;

    let gate = evaluate_common_validation_gate(step, model, device, cfg.hard_state_threshold);
    let legal_discard_count = match gate {
        CommonGateOutcome::IncompatibleState => {
            report.labels_rejected += 1;
            report.rejected_incompatible_state += 1;
            return;
        }
        CommonGateOutcome::TooFewDiscards => {
            report.compatible_discard_states += 1;
            report.labels_rejected += 1;
            report.rejected_too_few_discards += 1;
            return;
        }
        CommonGateOutcome::NotHardState => {
            report.compatible_discard_states += 1;
            report.labels_rejected += 1;
            report.rejected_not_hard_state += 1;
            return;
        }
        CommonGateOutcome::Pass(pass) => {
            let pass = *pass;
            report.compatible_discard_states += 1;
            report.hard_states += 1;
            pass.legal_discard_count
        }
    };

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
    report.coverage_sum += supported as f64 / legal_discard_count as f64;
    report.root_visits_sum += budget_from_legal_count(cfg, legal_discard_count) as u64;
}

/// Runs observational delta-q validation over self-play trajectories supplied by the caller.
///
/// `trajectories` must come from a producer run with live search labels enabled.
/// This crate intentionally does not depend on the training self-play runtime.
pub fn run_delta_q_validation_over_trajectories<B: Backend>(
    trajectories: &[hydra_core::arena::Trajectory],
    model: &HydraModel<B>,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
    exit_config: &ExitConfig,
) -> DeltaQValidationReport {
    let mut report = DeltaQValidationReport::new();
    for trajectory in trajectories {
        let mut trajectory_report = DeltaQValidationReport::new();
        for step in &trajectory.steps {
            collect_validation_metrics_for_step(
                step,
                model,
                device,
                exit_config,
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

#[cfg(test)]
mod tests;
