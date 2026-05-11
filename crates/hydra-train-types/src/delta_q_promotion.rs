//! Backend-independent DeltaQ promotion gate scalar types and evaluators.
//!
//! This module owns the serializable reports, thresholds, recommendations, and
//! slice-based metric collectors used by `hydra-train` tensor wrappers. It must
//! stay free of `hydra-train` dependencies so the training crate can re-export
//! these public APIs without creating a dependency cycle.

use std::fmt;

use crate::eval::PairedArenaEvalResult;

/// Per-state comparison between teacher-backed target action, baseline action,
/// and candidate action for DeltaQ promotion metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaQDecisionComparison {
    pub teacher_best_action: usize,
    pub baseline_action: usize,
    pub candidate_action: usize,
    pub baseline_regret: f64,
    pub candidate_regret: f64,
    pub decision_lift: f64,
    pub top_gap: f64,
    pub supported_actions: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct DeltaQComparisonInputs<'a> {
    teacher_target: &'a [f32],
    teacher_mask: &'a [f32],
    legal_mask: &'a [f32],
    baseline_scores: &'a [f32],
    candidate_scores: &'a [f32],
}

/// Flat slice inputs for DeltaQ promotion metric collection.
#[derive(Debug, Clone, Copy)]
pub struct DeltaQPromotionSliceInputs<'a> {
    pub policy_logits: &'a [f32],
    pub policy_rows: usize,
    pub policy_width: usize,
    pub candidate_delta_q: &'a [f32],
    pub candidate_delta_q_rows: usize,
    pub candidate_delta_q_width: usize,
    pub teacher_target: &'a [f32],
    pub teacher_target_rows: usize,
    pub teacher_target_width: usize,
    pub teacher_mask: &'a [f32],
    pub teacher_mask_rows: usize,
    pub teacher_mask_width: usize,
    pub legal_mask: &'a [f32],
    pub legal_mask_rows: usize,
    pub legal_mask_width: usize,
}

/// Flat slice inputs for DeltaQ policy-transfer metric collection.
#[derive(Debug, Clone, Copy)]
pub struct DeltaQPolicyTransferSliceInputs<'a> {
    pub candidate_policy_logits: &'a [f32],
    pub candidate_policy_rows: usize,
    pub candidate_policy_width: usize,
    pub baseline_policy_logits: &'a [f32],
    pub baseline_policy_rows: usize,
    pub baseline_policy_width: usize,
    pub teacher_target: &'a [f32],
    pub teacher_target_rows: usize,
    pub teacher_target_width: usize,
    pub teacher_mask: &'a [f32],
    pub teacher_mask_rows: usize,
    pub teacher_mask_width: usize,
    pub legal_mask: &'a [f32],
    pub legal_mask_rows: usize,
    pub legal_mask_width: usize,
}

/// Arena confirmation report persisted with DeltaQ promotion artifacts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaQArenaReport {
    pub compared_games: usize,
    pub baseline_mean_placement: f64,
    pub candidate_mean_placement: f64,
    pub delta_mean_placement: f64,
    pub baseline_stable_dan: f64,
    pub candidate_stable_dan: f64,
    pub delta_stable_dan: f64,
    pub lower_confidence_bound_mean_placement: f64,
    pub upper_confidence_bound_mean_placement: f64,
}

impl DeltaQArenaReport {
    #[allow(
        clippy::too_many_arguments,
        reason = "metric DTO constructor mirrors persisted arena report fields"
    )]
    pub fn from_arena_metrics(
        compared_games: usize,
        baseline_mean_placement: f64,
        candidate_mean_placement: f64,
        delta_mean_placement: f64,
        baseline_stable_dan: f64,
        candidate_stable_dan: f64,
        delta_stable_dan: f64,
        lower_confidence_bound_mean_placement: f64,
        upper_confidence_bound_mean_placement: f64,
    ) -> Self {
        Self {
            compared_games,
            baseline_mean_placement,
            candidate_mean_placement,
            delta_mean_placement,
            baseline_stable_dan,
            candidate_stable_dan,
            delta_stable_dan,
            lower_confidence_bound_mean_placement,
            upper_confidence_bound_mean_placement,
        }
    }

    pub fn from_paired_eval(
        result: &PairedArenaEvalResult,
        lower_confidence_bound_mean_placement: f32,
    ) -> Self {
        Self::from_arena_metrics(
            result.compared_games,
            result.baseline_mean_placement as f64,
            result.candidate_mean_placement as f64,
            result.delta_mean_placement as f64,
            result.baseline_stable_dan as f64,
            result.candidate_stable_dan as f64,
            result.delta_stable_dan as f64,
            lower_confidence_bound_mean_placement as f64,
            result.upper_confidence_bound_mean_placement as f64,
        )
    }
}

/// Offline DeltaQ promotion metrics accumulated over teacher-supported states.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaQPromotionReport {
    pub eligible_states: u64,
    pub compared_states: u64,
    pub masked_entries: u64,
    pub supported_actions_sum: u64,
    pub candidate_top1_agreement_count: u64,
    pub baseline_top1_agreement_count: u64,
    pub candidate_high_gap_top1_count: u64,
    pub baseline_high_gap_top1_count: u64,
    pub high_gap_states: u64,
    pub candidate_regret_sum: f64,
    pub baseline_regret_sum: f64,
    pub decision_lift_sum: f64,
    pub negative_lift_count: u64,
    pub candidate_regret_beats_baseline_count: u64,
    pub candidate_top1_beats_baseline_count: u64,
}

/// DeltaQ policy-transfer metrics accumulated over teacher-supported states.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaQPolicyTransferReport {
    pub compared_states: u64,
    pub candidate_policy_top1_to_teacher_count: u64,
    pub baseline_policy_top1_to_teacher_count: u64,
    pub candidate_policy_regret_sum: f64,
    pub baseline_policy_regret_sum: f64,
    pub candidate_beats_baseline_count: u64,
    pub negative_transfer_count: u64,
}

/// Thresholds for the DeltaQ policy-transfer offline gate.
#[derive(Debug, Clone)]
pub struct DeltaQPolicyTransferThresholds {
    pub min_compared_states: u64,
    pub max_candidate_policy_mean_teacher_regret_ratio: f64,
    pub max_negative_transfer_fraction: f64,
    pub min_candidate_beats_baseline_rate: f64,
}

impl Default for DeltaQPolicyTransferThresholds {
    fn default() -> Self {
        Self {
            min_compared_states: 1_000,
            max_candidate_policy_mean_teacher_regret_ratio: 0.95,
            max_negative_transfer_fraction: 0.45,
            min_candidate_beats_baseline_rate: 0.55,
        }
    }
}
/// Arena promotion decision produced by paired arena confirmation.
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

/// Result of evaluating DeltaQ policy-transfer metrics against thresholds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaQPolicyTransferResult {
    pub passed: bool,
    pub criteria: Vec<DeltaQPromotionCriterionResult>,
}

impl DeltaQPolicyTransferReport {
    pub fn new() -> Self {
        Self {
            compared_states: 0,
            candidate_policy_top1_to_teacher_count: 0,
            baseline_policy_top1_to_teacher_count: 0,
            candidate_policy_regret_sum: 0.0,
            baseline_policy_regret_sum: 0.0,
            candidate_beats_baseline_count: 0,
            negative_transfer_count: 0,
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.compared_states += other.compared_states;
        self.candidate_policy_top1_to_teacher_count += other.candidate_policy_top1_to_teacher_count;
        self.baseline_policy_top1_to_teacher_count += other.baseline_policy_top1_to_teacher_count;
        self.candidate_policy_regret_sum += other.candidate_policy_regret_sum;
        self.baseline_policy_regret_sum += other.baseline_policy_regret_sum;
        self.candidate_beats_baseline_count += other.candidate_beats_baseline_count;
        self.negative_transfer_count += other.negative_transfer_count;
    }

    pub fn candidate_policy_top1_to_teacher(&self) -> f64 {
        ratio_u64(
            self.candidate_policy_top1_to_teacher_count,
            self.compared_states,
        )
    }

    pub fn baseline_policy_top1_to_teacher(&self) -> f64 {
        ratio_u64(
            self.baseline_policy_top1_to_teacher_count,
            self.compared_states,
        )
    }

    pub fn candidate_policy_mean_teacher_regret(&self) -> f64 {
        ratio_f64(self.candidate_policy_regret_sum, self.compared_states)
    }

    pub fn baseline_policy_mean_teacher_regret(&self) -> f64 {
        ratio_f64(self.baseline_policy_regret_sum, self.compared_states)
    }

    pub fn mean_regret_improvement(&self) -> f64 {
        self.baseline_policy_mean_teacher_regret() - self.candidate_policy_mean_teacher_regret()
    }

    pub fn candidate_beats_baseline_rate(&self) -> f64 {
        ratio_u64(self.candidate_beats_baseline_count, self.compared_states)
    }

    pub fn negative_transfer_fraction(&self) -> f64 {
        ratio_u64(self.negative_transfer_count, self.compared_states)
    }
}

impl Default for DeltaQPolicyTransferReport {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaQPolicyTransferResult {
    pub fn recommendation(&self) -> DeltaQPromotionRecommendation {
        if self.passed {
            DeltaQPromotionRecommendation::RequiresArenaConfirmation
        } else {
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        }
    }
}

impl DeltaQPromotionReport {
    pub fn new() -> Self {
        Self {
            eligible_states: 0,
            compared_states: 0,
            masked_entries: 0,
            supported_actions_sum: 0,
            candidate_top1_agreement_count: 0,
            baseline_top1_agreement_count: 0,
            candidate_high_gap_top1_count: 0,
            baseline_high_gap_top1_count: 0,
            high_gap_states: 0,
            candidate_regret_sum: 0.0,
            baseline_regret_sum: 0.0,
            decision_lift_sum: 0.0,
            negative_lift_count: 0,
            candidate_regret_beats_baseline_count: 0,
            candidate_top1_beats_baseline_count: 0,
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.eligible_states += other.eligible_states;
        self.compared_states += other.compared_states;
        self.masked_entries += other.masked_entries;
        self.supported_actions_sum += other.supported_actions_sum;
        self.candidate_top1_agreement_count += other.candidate_top1_agreement_count;
        self.baseline_top1_agreement_count += other.baseline_top1_agreement_count;
        self.candidate_high_gap_top1_count += other.candidate_high_gap_top1_count;
        self.baseline_high_gap_top1_count += other.baseline_high_gap_top1_count;
        self.high_gap_states += other.high_gap_states;
        self.candidate_regret_sum += other.candidate_regret_sum;
        self.baseline_regret_sum += other.baseline_regret_sum;
        self.decision_lift_sum += other.decision_lift_sum;
        self.negative_lift_count += other.negative_lift_count;
        self.candidate_regret_beats_baseline_count += other.candidate_regret_beats_baseline_count;
        self.candidate_top1_beats_baseline_count += other.candidate_top1_beats_baseline_count;
    }

    pub fn mean_supported_actions(&self) -> f64 {
        ratio_u64(self.supported_actions_sum, self.compared_states)
    }

    pub fn candidate_top1_agreement(&self) -> f64 {
        ratio_u64(self.candidate_top1_agreement_count, self.compared_states)
    }

    pub fn baseline_top1_agreement(&self) -> f64 {
        ratio_u64(self.baseline_top1_agreement_count, self.compared_states)
    }

    pub fn candidate_mean_regret(&self) -> f64 {
        ratio_f64(self.candidate_regret_sum, self.compared_states)
    }

    pub fn baseline_mean_regret(&self) -> f64 {
        ratio_f64(self.baseline_regret_sum, self.compared_states)
    }

    pub fn mean_decision_lift(&self) -> f64 {
        ratio_f64(self.decision_lift_sum, self.compared_states)
    }

    pub fn negative_lift_fraction(&self) -> f64 {
        ratio_u64(self.negative_lift_count, self.compared_states)
    }

    pub fn candidate_regret_beats_baseline_rate(&self) -> f64 {
        ratio_u64(
            self.candidate_regret_beats_baseline_count,
            self.compared_states,
        )
    }

    pub fn candidate_top1_beats_baseline_rate(&self) -> f64 {
        ratio_u64(
            self.candidate_top1_beats_baseline_count,
            self.compared_states,
        )
    }

    pub fn candidate_high_gap_top1(&self) -> f64 {
        ratio_u64(self.candidate_high_gap_top1_count, self.high_gap_states)
    }

    pub fn baseline_high_gap_top1(&self) -> f64 {
        ratio_u64(self.baseline_high_gap_top1_count, self.high_gap_states)
    }
}

impl Default for DeltaQPromotionReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DeltaQPromotionReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== DeltaQ Promotion Report ===")?;
        writeln!(f, "Eligible states:         {}", self.eligible_states)?;
        writeln!(f, "Compared states:         {}", self.compared_states)?;
        writeln!(f, "Masked entries:          {}", self.masked_entries)?;
        writeln!(
            f,
            "Mean supported acts:     {:.2}",
            self.mean_supported_actions()
        )?;
        writeln!(
            f,
            "Candidate top1 agree:    {:.2}%",
            self.candidate_top1_agreement() * 100.0
        )?;
        writeln!(
            f,
            "Baseline top1 agree:     {:.2}%",
            self.baseline_top1_agreement() * 100.0
        )?;
        writeln!(
            f,
            "Candidate mean regret:   {:.6}",
            self.candidate_mean_regret()
        )?;
        writeln!(
            f,
            "Baseline mean regret:    {:.6}",
            self.baseline_mean_regret()
        )?;
        writeln!(
            f,
            "Mean decision lift:      {:.6}",
            self.mean_decision_lift()
        )?;
        writeln!(
            f,
            "Negative lift frac:      {:.3}",
            self.negative_lift_fraction()
        )?;
        writeln!(
            f,
            "Regret beats baseline:   {:.2}%",
            self.candidate_regret_beats_baseline_rate() * 100.0
        )?;
        writeln!(
            f,
            "Top1 beats baseline:     {:.2}%",
            self.candidate_top1_beats_baseline_rate() * 100.0
        )?;
        writeln!(f, "High-gap states:         {}", self.high_gap_states)?;
        writeln!(
            f,
            "Candidate high-gap top1: {:.2}%",
            self.candidate_high_gap_top1() * 100.0
        )?;
        writeln!(
            f,
            "Baseline high-gap top1:  {:.2}%",
            self.baseline_high_gap_top1() * 100.0
        )?;
        Ok(())
    }
}

/// Thresholds for the DeltaQ promotion offline gate.
#[derive(Debug, Clone)]
pub struct DeltaQPromotionThresholds {
    pub min_compared_states: u64,
    pub min_mean_supported_actions: f64,
    pub max_candidate_mean_regret: f64,
    pub max_negative_lift_fraction: f64,
    pub min_candidate_top1_agreement: f64,
    pub min_regret_beats_baseline_rate: f64,
    pub min_top1_beats_baseline_rate: f64,
}

impl Default for DeltaQPromotionThresholds {
    fn default() -> Self {
        Self {
            min_compared_states: 1_000,
            min_mean_supported_actions: 3.0,
            max_candidate_mean_regret: 0.10,
            max_negative_lift_fraction: 0.40,
            min_candidate_top1_agreement: 0.60,
            min_regret_beats_baseline_rate: 0.55,
            min_top1_beats_baseline_rate: 0.40,
        }
    }
}

/// Result of evaluating DeltaQ promotion metrics against thresholds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaQPromotionResult {
    pub passed: bool,
    pub criteria: Vec<DeltaQPromotionCriterionResult>,
}

impl DeltaQPromotionResult {
    pub fn recommendation(&self) -> DeltaQPromotionRecommendation {
        if self.passed {
            DeltaQPromotionRecommendation::RequiresPolicyTransferGate
        } else {
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        }
    }
}

/// Single threshold criterion result recorded by DeltaQ gates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaQPromotionCriterionResult {
    pub name: String,
    pub measured: f64,
    pub threshold: f64,
    pub passed: bool,
    pub direction: PromotionThresholdDirection,
}

/// Direction used to evaluate a DeltaQ threshold criterion.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum PromotionThresholdDirection {
    Min,
    Max,
}

/// Next-step recommendation after a DeltaQ gate result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DeltaQPromotionRecommendation {
    RejectAtOfflineGate,
    RequiresPolicyTransferGate,
    RequiresArenaConfirmation,
}

/// Required invariants for an arena confirmation after DeltaQ offline gates pass.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaQArenaConfirmationRequest {
    pub same_seeds: bool,
    pub same_seat_rotation_schedule: bool,
    pub same_search_budget: bool,
    pub same_temperature: bool,
    pub same_frozen_opponent_pool: bool,
    pub min_games: u64,
}

impl Default for DeltaQArenaConfirmationRequest {
    fn default() -> Self {
        Self {
            same_seeds: true,
            same_seat_rotation_schedule: true,
            same_search_budget: true,
            same_temperature: true,
            same_frozen_opponent_pool: true,
            min_games: 10_000,
        }
    }
}

impl fmt::Display for DeltaQPromotionRecommendation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::RejectAtOfflineGate => "reject_at_offline_gate",
            Self::RequiresPolicyTransferGate => "requires_policy_transfer_gate",
            Self::RequiresArenaConfirmation => "requires_arena_confirmation",
        };
        f.write_str(label)
    }
}

impl DeltaQArenaConfirmationRequest {
    pub fn summary(&self) -> String {
        format!(
            "same_seeds={} same_rotation={} same_budget={} same_temp={} frozen_pool={} min_games={}",
            self.same_seeds,
            self.same_seat_rotation_schedule,
            self.same_search_budget,
            self.same_temperature,
            self.same_frozen_opponent_pool,
            self.min_games,
        )
    }
}

impl fmt::Display for DeltaQPromotionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== DeltaQ Promotion Result: {} ===",
            if self.passed { "PASS" } else { "FAIL" }
        )?;
        for criterion in &self.criteria {
            let status = if criterion.passed { "PASS" } else { "FAIL" };
            let direction = match criterion.direction {
                PromotionThresholdDirection::Min => ">=",
                PromotionThresholdDirection::Max => "<=",
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

impl fmt::Display for DeltaQPolicyTransferResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== DeltaQ Policy Transfer Result: {} ===",
            if self.passed { "PASS" } else { "FAIL" }
        )?;
        for criterion in &self.criteria {
            let status = if criterion.passed { "PASS" } else { "FAIL" };
            let direction = match criterion.direction {
                PromotionThresholdDirection::Min => ">=",
                PromotionThresholdDirection::Max => "<=",
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

/// Evaluate a DeltaQ promotion report against offline promotion thresholds.
pub fn evaluate_promotion_report(
    report: &DeltaQPromotionReport,
    thresholds: &DeltaQPromotionThresholds,
) -> DeltaQPromotionResult {
    let mut criteria = Vec::with_capacity(7);
    push_min_criterion(
        &mut criteria,
        "compared_states",
        report.compared_states as f64,
        thresholds.min_compared_states as f64,
    );
    push_min_criterion(
        &mut criteria,
        "mean_supported_actions",
        report.mean_supported_actions(),
        thresholds.min_mean_supported_actions,
    );
    push_max_criterion(
        &mut criteria,
        "candidate_mean_regret",
        report.candidate_mean_regret(),
        thresholds.max_candidate_mean_regret,
    );
    push_max_criterion(
        &mut criteria,
        "negative_lift_fraction",
        report.negative_lift_fraction(),
        thresholds.max_negative_lift_fraction,
    );
    push_min_criterion(
        &mut criteria,
        "candidate_top1_agreement",
        report.candidate_top1_agreement(),
        thresholds.min_candidate_top1_agreement,
    );
    push_min_criterion(
        &mut criteria,
        "regret_beats_baseline_rate",
        report.candidate_regret_beats_baseline_rate(),
        thresholds.min_regret_beats_baseline_rate,
    );
    push_min_criterion(
        &mut criteria,
        "top1_beats_baseline_rate",
        report.candidate_top1_beats_baseline_rate(),
        thresholds.min_top1_beats_baseline_rate,
    );
    DeltaQPromotionResult {
        passed: criteria.iter().all(|criterion| criterion.passed),
        criteria,
    }
}

/// Evaluate a DeltaQ policy-transfer report against offline transfer thresholds.
pub fn evaluate_policy_transfer_report(
    report: &DeltaQPolicyTransferReport,
    thresholds: &DeltaQPolicyTransferThresholds,
) -> DeltaQPolicyTransferResult {
    let mut criteria = Vec::with_capacity(4);
    push_min_criterion(
        &mut criteria,
        "compared_states",
        report.compared_states as f64,
        thresholds.min_compared_states as f64,
    );
    let baseline_mean_regret = report.baseline_policy_mean_teacher_regret();
    let regret_ratio = if baseline_mean_regret <= f64::EPSILON {
        if report.candidate_policy_mean_teacher_regret() <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        report.candidate_policy_mean_teacher_regret() / baseline_mean_regret
    };
    push_max_criterion(
        &mut criteria,
        "candidate_policy_mean_teacher_regret_ratio",
        regret_ratio,
        thresholds.max_candidate_policy_mean_teacher_regret_ratio,
    );
    push_max_criterion(
        &mut criteria,
        "negative_transfer_fraction",
        report.negative_transfer_fraction(),
        thresholds.max_negative_transfer_fraction,
    );
    push_min_criterion(
        &mut criteria,
        "candidate_beats_baseline_rate",
        report.candidate_beats_baseline_rate(),
        thresholds.min_candidate_beats_baseline_rate,
    );
    DeltaQPolicyTransferResult {
        passed: criteria.iter().all(|criterion| criterion.passed),
        criteria,
    }
}

/// Collect DeltaQ promotion metrics from flat row-major slices.
pub fn collect_promotion_metrics_from_slices(
    inputs: DeltaQPromotionSliceInputs<'_>,
    high_gap_quantile: f64,
) -> DeltaQPromotionReport {
    let batch = inputs
        .policy_rows
        .min(inputs.candidate_delta_q_rows)
        .min(inputs.teacher_target_rows)
        .min(inputs.teacher_mask_rows)
        .min(inputs.legal_mask_rows);
    let mut comparisons = Vec::with_capacity(batch);
    let mut report = DeltaQPromotionReport::new();
    for row in 0..batch {
        let policy_row = row_slice(inputs.policy_logits, row, inputs.policy_width);
        let delta_q_row = row_slice(
            inputs.candidate_delta_q,
            row,
            inputs.candidate_delta_q_width,
        );
        let target_row = row_slice(inputs.teacher_target, row, inputs.teacher_target_width);
        let mask_row = row_slice(inputs.teacher_mask, row, inputs.teacher_mask_width);
        let legal_row = row_slice(inputs.legal_mask, row, inputs.legal_mask_width);
        let (
            Some(policy_row),
            Some(delta_q_row),
            Some(target_row),
            Some(mask_row),
            Some(legal_row),
        ) = (policy_row, delta_q_row, target_row, mask_row, legal_row)
        else {
            continue;
        };
        report.eligible_states += 1;
        if let Some(comparison) =
            compare_delta_q_state(policy_row, delta_q_row, target_row, mask_row, legal_row)
        {
            report.compared_states += 1;
            report.masked_entries += comparison.supported_actions as u64;
            report.supported_actions_sum += comparison.supported_actions as u64;
            report.candidate_regret_sum += comparison.candidate_regret;
            report.baseline_regret_sum += comparison.baseline_regret;
            report.decision_lift_sum += comparison.decision_lift;
            if comparison.candidate_action == comparison.teacher_best_action {
                report.candidate_top1_agreement_count += 1;
            }
            if comparison.baseline_action == comparison.teacher_best_action {
                report.baseline_top1_agreement_count += 1;
            }
            if comparison.candidate_regret <= comparison.baseline_regret {
                report.candidate_regret_beats_baseline_count += 1;
            }
            if (comparison.candidate_action == comparison.teacher_best_action)
                && (comparison.baseline_action != comparison.teacher_best_action)
            {
                report.candidate_top1_beats_baseline_count += 1;
            }
            if comparison.decision_lift < 0.0 {
                report.negative_lift_count += 1;
            }
            comparisons.push(comparison);
        }
    }
    if comparisons.is_empty() {
        return report;
    }
    let mut top_gaps: Vec<f64> = comparisons
        .iter()
        .map(|comparison| comparison.top_gap)
        .collect();
    top_gaps.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    let quantile = high_gap_quantile.clamp(0.0, 1.0);
    let start = ((top_gaps.len() as f64) * quantile).floor() as usize;
    let cutoff_index = start.min(top_gaps.len().saturating_sub(1));
    let cutoff = top_gaps[cutoff_index];
    for comparison in comparisons {
        if comparison.top_gap >= cutoff {
            report.high_gap_states += 1;
            if comparison.candidate_action == comparison.teacher_best_action {
                report.candidate_high_gap_top1_count += 1;
            }
            if comparison.baseline_action == comparison.teacher_best_action {
                report.baseline_high_gap_top1_count += 1;
            }
        }
    }
    report
}

/// Collect DeltaQ policy-transfer metrics from flat row-major slices.
pub fn collect_policy_transfer_metrics_from_slices(
    inputs: DeltaQPolicyTransferSliceInputs<'_>,
) -> DeltaQPolicyTransferReport {
    let batch = inputs
        .candidate_policy_rows
        .min(inputs.baseline_policy_rows)
        .min(inputs.teacher_target_rows)
        .min(inputs.teacher_mask_rows)
        .min(inputs.legal_mask_rows);
    let mut report = DeltaQPolicyTransferReport::new();
    for row in 0..batch {
        let candidate_row = row_slice(
            inputs.candidate_policy_logits,
            row,
            inputs.candidate_policy_width,
        );
        let baseline_row = row_slice(
            inputs.baseline_policy_logits,
            row,
            inputs.baseline_policy_width,
        );
        let target_row = row_slice(inputs.teacher_target, row, inputs.teacher_target_width);
        let mask_row = row_slice(inputs.teacher_mask, row, inputs.teacher_mask_width);
        let legal_row = row_slice(inputs.legal_mask, row, inputs.legal_mask_width);
        let (
            Some(candidate_row),
            Some(baseline_row),
            Some(target_row),
            Some(mask_row),
            Some(legal_row),
        ) = (candidate_row, baseline_row, target_row, mask_row, legal_row)
        else {
            continue;
        };
        if let Some(comparison) = compare_policy_transfer_state(
            candidate_row,
            baseline_row,
            target_row,
            mask_row,
            legal_row,
        ) {
            report.compared_states += 1;
            report.candidate_policy_regret_sum += comparison.candidate_regret;
            report.baseline_policy_regret_sum += comparison.baseline_regret;
            if comparison.candidate_action == comparison.teacher_best_action {
                report.candidate_policy_top1_to_teacher_count += 1;
            }
            if comparison.baseline_action == comparison.teacher_best_action {
                report.baseline_policy_top1_to_teacher_count += 1;
            }
            if comparison.candidate_regret <= comparison.baseline_regret {
                report.candidate_beats_baseline_count += 1;
            }
            if comparison.candidate_regret > comparison.baseline_regret {
                report.negative_transfer_count += 1;
            }
        }
    }
    report
}

fn row_slice(values: &[f32], row: usize, width: usize) -> Option<&[f32]> {
    let start = row.checked_mul(width)?;
    let end = start.checked_add(width)?;
    values.get(start..end)
}

fn compare_delta_q_state(
    policy_logits: &[f32],
    candidate_delta_q: &[f32],
    teacher_target: &[f32],
    teacher_mask: &[f32],
    legal_mask: &[f32],
) -> Option<DeltaQDecisionComparison> {
    compare_with_teacher(DeltaQComparisonInputs {
        teacher_target,
        teacher_mask,
        legal_mask,
        baseline_scores: policy_logits,
        candidate_scores: candidate_delta_q,
    })
}

fn compare_policy_transfer_state(
    candidate_policy_logits: &[f32],
    baseline_policy_logits: &[f32],
    teacher_target: &[f32],
    teacher_mask: &[f32],
    legal_mask: &[f32],
) -> Option<DeltaQDecisionComparison> {
    compare_with_teacher(DeltaQComparisonInputs {
        teacher_target,
        teacher_mask,
        legal_mask,
        baseline_scores: baseline_policy_logits,
        candidate_scores: candidate_policy_logits,
    })
}

fn compare_with_teacher(inputs: DeltaQComparisonInputs<'_>) -> Option<DeltaQDecisionComparison> {
    let supported_actions: Vec<usize> = inputs
        .teacher_mask
        .iter()
        .enumerate()
        .filter_map(|(action, &mask)| {
            if mask > 0.0 && inputs.legal_mask.get(action).copied().unwrap_or(0.0) > 0.0 {
                Some(action)
            } else {
                None
            }
        })
        .collect();
    if supported_actions.len() < 2 {
        return None;
    }
    let mut sorted_targets: Vec<f64> = supported_actions
        .iter()
        .map(|&action| inputs.teacher_target[action] as f64)
        .collect();
    sorted_targets.sort_by(|lhs, rhs| rhs.partial_cmp(lhs).unwrap_or(std::cmp::Ordering::Equal));
    let top_gap = sorted_targets[0] - sorted_targets[1];
    let teacher_best_action = argmax_over_actions(inputs.teacher_target, &supported_actions)?;
    let baseline_action = argmax_over_actions(inputs.baseline_scores, &supported_actions)?;
    let candidate_action = argmax_over_actions(inputs.candidate_scores, &supported_actions)?;
    let teacher_best_value = inputs.teacher_target[teacher_best_action] as f64;
    let baseline_regret = teacher_best_value - inputs.teacher_target[baseline_action] as f64;
    let candidate_regret = teacher_best_value - inputs.teacher_target[candidate_action] as f64;
    Some(DeltaQDecisionComparison {
        teacher_best_action,
        baseline_action,
        candidate_action,
        baseline_regret,
        candidate_regret,
        decision_lift: baseline_regret - candidate_regret,
        top_gap,
        supported_actions: supported_actions.len(),
    })
}

fn argmax_over_actions(values: &[f32], actions: &[usize]) -> Option<usize> {
    let mut best_action = None;
    let mut best_value = f32::NEG_INFINITY;
    for &action in actions {
        let value = *values.get(action)?;
        if value > best_value {
            best_value = value;
            best_action = Some(action);
        }
    }
    best_action
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

fn push_min_criterion(
    criteria: &mut Vec<DeltaQPromotionCriterionResult>,
    name: &str,
    measured: f64,
    threshold: f64,
) {
    criteria.push(DeltaQPromotionCriterionResult {
        name: name.to_string(),
        measured,
        threshold,
        passed: measured >= threshold,
        direction: PromotionThresholdDirection::Min,
    });
}

fn push_max_criterion(
    criteria: &mut Vec<DeltaQPromotionCriterionResult>,
    name: &str,
    measured: f64,
    threshold: f64,
) {
    criteria.push(DeltaQPromotionCriterionResult {
        name: name.to_string(),
        measured,
        threshold,
        passed: measured <= threshold,
        direction: PromotionThresholdDirection::Max,
    });
}

#[cfg(test)]
mod tests;
