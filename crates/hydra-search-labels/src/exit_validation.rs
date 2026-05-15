//! ExIt validation harness for measuring producer label quality.
//!
//! Implements the "validation matrix" from Agent 22 doctrine: the live
//! ExIt producer must prove it generates useful, sane labels before
//! enablement. This harness runs the producer in shadow mode through the
//! existing self-play pipeline and collects metrics for pass/fail evaluation.

use std::fmt;

use burn::prelude::Backend;
use hydra_core::action::{DISCARD_END, HYDRA_ACTION_SPACE};
use hydra_core::arena::TrajectoryStep;

use hydra_model::model::HydraModel;

use crate::exit::ExitConfig;
use crate::live_exit::budget_from_legal_count;
use crate::validation_common::{
    CommonGateOutcome, evaluate_common_validation_gate, ratio_f64, ratio_u64,
};

/// Aggregated metrics from a shadow ExIt validation run.
///
/// Each field corresponds to one criterion from the Agent 22/9/16
/// blueprint. The harness collects these by running the live producer
/// on self-play states without using the labels for training.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExitValidationReport {
    /// Total decision states examined.
    pub total_states: u64,
    pub compatible_discard_states: u64,
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
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
    cfg: &ExitConfig,
    report: &mut ExitValidationReport,
) {
    report.total_states += 1;

    let gate = evaluate_common_validation_gate(step, model, device, cfg.hard_state_threshold);
    let (base_pi, legal_discards, legal_discard_count) = match gate {
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
            (pass.base_pi, pass.legal_discards, pass.legal_discard_count)
        }
    };

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
    report.coverage_sum += supported as f64 / legal_discard_count as f64;
    report.root_visits_sum += budget_from_legal_count(cfg, legal_discard_count) as u64;

    let base_top1 = top1_index(&base_pi, &legal_discards);
    let exit_top1 = top1_index(&label.target, &legal_discards);
    if base_top1 == exit_top1 {
        report.top1_agreement_count += 1;
    }

    report.kl_sum += kl_divergence(&base_pi, &label.target, &label.mask);
}

/// Runs shadow ExIt validation over self-play games supplied by the caller.
///
/// `trajectories` must come from a producer run with live ExIt enabled. This
/// crate intentionally does not depend on the training self-play runtime.
pub fn run_exit_validation_over_trajectories<B: Backend>(
    trajectories: &[hydra_core::arena::Trajectory],
    model: &HydraModel<B>,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
    exit_config: &ExitConfig,
) -> ExitValidationReport {
    let mut report = ExitValidationReport::new();
    for trajectory in trajectories {
        let mut trajectory_report = ExitValidationReport::new();
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
mod tests;
