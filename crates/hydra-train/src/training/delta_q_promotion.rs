use std::fmt;

use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::model::HydraOutput;
use crate::training::losses::HydraTargets;

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
    pub fn from_paired_eval(
        result: &crate::eval::PairedArenaEvalResult,
        lower_confidence_bound_mean_placement: f32,
    ) -> Self {
        Self {
            compared_games: result.compared_games,
            baseline_mean_placement: result.baseline_mean_placement as f64,
            candidate_mean_placement: result.candidate_mean_placement as f64,
            delta_mean_placement: result.delta_mean_placement as f64,
            baseline_stable_dan: result.baseline_stable_dan as f64,
            candidate_stable_dan: result.candidate_stable_dan as f64,
            delta_stable_dan: result.delta_stable_dan as f64,
            lower_confidence_bound_mean_placement: lower_confidence_bound_mean_placement as f64,
            upper_confidence_bound_mean_placement: result.upper_confidence_bound_mean_placement
                as f64,
        }
    }
}

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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaQPromotionCriterionResult {
    pub name: String,
    pub measured: f64,
    pub threshold: f64,
    pub passed: bool,
    pub direction: PromotionThresholdDirection,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum PromotionThresholdDirection {
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DeltaQPromotionRecommendation {
    RejectAtOfflineGate,
    RequiresPolicyTransferGate,
    RequiresArenaConfirmation,
}

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

pub fn collect_promotion_metrics_from_outputs<B: Backend>(
    output: &HydraOutput<B>,
    targets: &HydraTargets<B>,
    high_gap_quantile: f64,
) -> DeltaQPromotionReport {
    let Some(delta_q_target) = &targets.delta_q_target else {
        return DeltaQPromotionReport::new();
    };
    let Some(delta_q_mask) = &targets.delta_q_mask else {
        return DeltaQPromotionReport::new();
    };

    let policy = tensor_to_rows_f32(output.policy_logits.clone());
    let delta_q = tensor_to_rows_f32(output.delta_q.clone());
    let target = tensor_to_rows_f32(delta_q_target.clone());
    let mask = tensor_to_rows_f32(delta_q_mask.clone());
    let legal = tensor_to_rows_f32(targets.legal_mask.clone());

    let batch = policy
        .len()
        .min(delta_q.len())
        .min(target.len())
        .min(mask.len())
        .min(legal.len());

    let mut comparisons = Vec::with_capacity(batch);
    let mut report = DeltaQPromotionReport::new();

    for row in 0..batch {
        report.eligible_states += 1;
        if let Some(comparison) = compare_delta_q_state(
            &policy[row],
            &delta_q[row],
            &target[row],
            &mask[row],
            &legal[row],
        ) {
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

pub fn collect_policy_transfer_metrics_from_policy_outputs<B: Backend>(
    candidate_policy_logits: Tensor<B, 2>,
    baseline_policy_logits: Tensor<B, 2>,
    targets: &HydraTargets<B>,
) -> DeltaQPolicyTransferReport {
    let Some(delta_q_target) = &targets.delta_q_target else {
        return DeltaQPolicyTransferReport::new();
    };
    let Some(delta_q_mask) = &targets.delta_q_mask else {
        return DeltaQPolicyTransferReport::new();
    };

    let candidate = tensor_to_rows_f32(candidate_policy_logits);
    let baseline = tensor_to_rows_f32(baseline_policy_logits);
    let target = tensor_to_rows_f32(delta_q_target.clone());
    let mask = tensor_to_rows_f32(delta_q_mask.clone());
    let legal = tensor_to_rows_f32(targets.legal_mask.clone());

    let batch = candidate
        .len()
        .min(baseline.len())
        .min(target.len())
        .min(mask.len())
        .min(legal.len());

    let mut report = DeltaQPolicyTransferReport::new();
    for row in 0..batch {
        if let Some(comparison) = compare_policy_transfer_state(
            &candidate[row],
            &baseline[row],
            &target[row],
            &mask[row],
            &legal[row],
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

fn tensor_to_rows_f32<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<Vec<f32>> {
    let data = tensor.to_data();
    let values = data
        .as_slice::<f32>()
        .expect("promotion metrics require f32 tensor data")
        .to_vec();
    let dims = data.shape;
    let rows = dims.first().copied().unwrap_or(0);
    let row_width = dims.iter().skip(1).product::<usize>();
    if row_width == 0 {
        return Vec::new();
    }
    values
        .chunks(row_width)
        .take(rows)
        .map(|chunk| chunk.to_vec())
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
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    fn tensor2(values: Vec<f32>, rows: usize, cols: usize) -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(values, [rows, cols]), &Default::default())
    }

    fn dummy_targets() -> HydraTargets<B> {
        let device = Default::default();
        HydraTargets {
            policy_target: Tensor::zeros([2, 46], &device),
            legal_mask: Tensor::ones([2, 46], &device),
            value_target: Tensor::zeros([2], &device),
            grp_target: Tensor::zeros([2, 24], &device),
            tenpai_target: Tensor::zeros([2, 3], &device),
            danger_target: Tensor::zeros([2, 3, 34], &device),
            danger_mask: Tensor::zeros([2, 3, 34], &device),
            opp_next_target: Tensor::zeros([2, 3, 34], &device),
            score_pdf_target: Tensor::zeros([2, 64], &device),
            score_cdf_target: Tensor::zeros([2, 64], &device),
            oracle_target: None,
            belief_fields_target: None,
            belief_fields_mask: None,
            mixture_weight_target: None,
            mixture_weight_mask: None,
            opponent_hand_type_target: None,
            delta_q_target: None,
            delta_q_mask: None,
            safety_residual_target: None,
            safety_residual_mask: None,
            oracle_guidance_mask: None,
        }
    }

    fn dummy_output(policy_logits: Tensor<B, 2>, delta_q: Tensor<B, 2>) -> HydraOutput<B> {
        let device = Default::default();
        HydraOutput {
            policy_logits,
            value: Tensor::zeros([2, 1], &device),
            score_pdf: Tensor::zeros([2, 64], &device),
            score_cdf: Tensor::zeros([2, 64], &device),
            opp_tenpai: Tensor::zeros([2, 3], &device),
            grp: Tensor::zeros([2, 24], &device),
            opp_next_discard: Tensor::zeros([2, 3, 34], &device),
            danger: Tensor::zeros([2, 3, 34], &device),
            oracle_critic: Tensor::zeros([2, 4], &device),
            belief_fields: Tensor::zeros([2, 16, 34], &device),
            mixture_weight_logits: Tensor::zeros([2, 4], &device),
            opponent_hand_type: Tensor::zeros([2, 24], &device),
            delta_q,
            safety_residual: Tensor::zeros([2, 46], &device),
        }
    }

    #[test]
    fn compare_delta_q_state_requires_two_supported_actions() {
        let policy = [0.0f32; 46];
        let candidate = [0.0f32; 46];
        let mut target = [0.0f32; 46];
        target[0] = 1.0;
        let mut mask = [0.0f32; 46];
        mask[0] = 1.0;
        let legal = [1.0f32; 46];
        assert!(compare_delta_q_state(&policy, &candidate, &target, &mask, &legal).is_none());
    }

    #[test]
    fn compare_delta_q_state_computes_regret_and_lift() {
        let mut policy = [0.0f32; 46];
        let mut candidate = [0.0f32; 46];
        let mut target = [0.0f32; 46];
        let mut mask = [0.0f32; 46];
        let legal = [1.0f32; 46];
        for entry in mask.iter_mut().take(3) {
            *entry = 1.0;
        }
        target[0] = 0.40;
        target[1] = 0.10;
        target[2] = -0.30;
        policy[1] = 3.0;
        candidate[0] = 4.0;

        let comparison =
            compare_delta_q_state(&policy, &candidate, &target, &mask, &legal).expect("state");

        assert_eq!(comparison.teacher_best_action, 0);
        assert_eq!(comparison.baseline_action, 1);
        assert_eq!(comparison.candidate_action, 0);
        assert!((comparison.baseline_regret - 0.30).abs() < 1e-6);
        assert!(comparison.candidate_regret.abs() < 1e-6);
        assert!((comparison.decision_lift - 0.30).abs() < 1e-6);
        assert!((comparison.top_gap - 0.30).abs() < 1e-6);
    }

    #[test]
    fn collect_promotion_metrics_reports_candidate_advantage() {
        let mut targets = dummy_targets();
        let mut delta_q_target = vec![0.0f32; 2 * 46];
        let mut delta_q_mask = vec![0.0f32; 2 * 46];
        let mut policy_logits = vec![0.0f32; 2 * 46];
        let mut candidate_delta_q = vec![0.0f32; 2 * 46];

        delta_q_target[0] = 0.5;
        delta_q_target[1] = 0.2;
        delta_q_target[2] = -0.1;
        delta_q_mask[0] = 1.0;
        delta_q_mask[1] = 1.0;
        delta_q_mask[2] = 1.0;
        policy_logits[1] = 5.0;
        candidate_delta_q[0] = 2.0;

        let row = 46;
        delta_q_target[row] = 0.10;
        delta_q_target[row + 1] = 0.40;
        delta_q_target[row + 2] = 0.35;
        delta_q_mask[row] = 1.0;
        delta_q_mask[row + 1] = 1.0;
        delta_q_mask[row + 2] = 1.0;
        policy_logits[row] = 3.0;
        candidate_delta_q[row + 1] = 4.0;

        targets.delta_q_target = Some(tensor2(delta_q_target, 2, 46));
        targets.delta_q_mask = Some(tensor2(delta_q_mask, 2, 46));

        let output = dummy_output(
            tensor2(policy_logits, 2, 46),
            tensor2(candidate_delta_q, 2, 46),
        );
        let report = collect_promotion_metrics_from_outputs(&output, &targets, 0.5);

        assert_eq!(report.eligible_states, 2);
        assert_eq!(report.compared_states, 2);
        assert_eq!(report.masked_entries, 6);
        assert_eq!(report.candidate_top1_agreement_count, 2);
        assert_eq!(report.baseline_top1_agreement_count, 0);
        assert_eq!(report.candidate_regret_beats_baseline_count, 2);
        assert_eq!(report.candidate_top1_beats_baseline_count, 2);
        assert!(report.mean_decision_lift() > 0.0);
        assert_eq!(report.negative_lift_count, 0);
    }

    #[test]
    fn collect_policy_transfer_metrics_reports_policy_advantage() {
        let mut targets = dummy_targets();
        let mut delta_q_target = vec![0.0f32; 2 * 46];
        let mut delta_q_mask = vec![0.0f32; 2 * 46];
        let mut candidate_policy = vec![0.0f32; 2 * 46];
        let mut baseline_policy = vec![0.0f32; 2 * 46];

        delta_q_target[0] = 0.5;
        delta_q_target[1] = 0.2;
        delta_q_target[2] = -0.1;
        delta_q_mask[0] = 1.0;
        delta_q_mask[1] = 1.0;
        delta_q_mask[2] = 1.0;
        candidate_policy[0] = 4.0;
        baseline_policy[1] = 3.0;

        let row = 46;
        delta_q_target[row] = 0.10;
        delta_q_target[row + 1] = 0.40;
        delta_q_target[row + 2] = 0.35;
        delta_q_mask[row] = 1.0;
        delta_q_mask[row + 1] = 1.0;
        delta_q_mask[row + 2] = 1.0;
        candidate_policy[row + 1] = 2.5;
        baseline_policy[row] = 3.2;

        targets.delta_q_target = Some(tensor2(delta_q_target, 2, 46));
        targets.delta_q_mask = Some(tensor2(delta_q_mask, 2, 46));

        let report = collect_policy_transfer_metrics_from_policy_outputs(
            tensor2(candidate_policy, 2, 46),
            tensor2(baseline_policy, 2, 46),
            &targets,
        );

        assert_eq!(report.compared_states, 2);
        assert_eq!(report.candidate_policy_top1_to_teacher_count, 2);
        assert_eq!(report.baseline_policy_top1_to_teacher_count, 0);
        assert_eq!(report.candidate_beats_baseline_count, 2);
        assert_eq!(report.negative_transfer_count, 0);
        assert!(report.mean_regret_improvement() > 0.0);
    }

    #[test]
    fn evaluate_policy_transfer_report_requires_candidate_advantage() {
        let mut report = DeltaQPolicyTransferReport::new();
        report.compared_states = 1_500;
        report.candidate_policy_top1_to_teacher_count = 900;
        report.baseline_policy_top1_to_teacher_count = 700;
        report.candidate_policy_regret_sum = 90.0;
        report.baseline_policy_regret_sum = 150.0;
        report.candidate_beats_baseline_count = 950;
        report.negative_transfer_count = 250;

        let result = evaluate_policy_transfer_report(&report, &Default::default());
        assert!(result.passed);
        assert_eq!(
            result.recommendation(),
            DeltaQPromotionRecommendation::RequiresArenaConfirmation
        );

        report.candidate_policy_regret_sum = 170.0;
        let fail = evaluate_policy_transfer_report(&report, &Default::default());
        assert!(!fail.passed);
        assert_eq!(
            fail.recommendation(),
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        );
    }

    #[test]
    fn evaluate_promotion_report_fails_when_candidate_regresses() {
        let report = DeltaQPromotionReport {
            eligible_states: 1_200,
            compared_states: 1_100,
            masked_entries: 4_400,
            supported_actions_sum: 4_400,
            candidate_top1_agreement_count: 500,
            baseline_top1_agreement_count: 700,
            candidate_high_gap_top1_count: 100,
            baseline_high_gap_top1_count: 150,
            high_gap_states: 200,
            candidate_regret_sum: 220.0,
            baseline_regret_sum: 110.0,
            decision_lift_sum: -110.0,
            negative_lift_count: 700,
            candidate_regret_beats_baseline_count: 100,
            candidate_top1_beats_baseline_count: 50,
        };
        let result = evaluate_promotion_report(&report, &DeltaQPromotionThresholds::default());
        assert!(!result.passed);
        assert!(result.criteria.iter().any(|criterion| !criterion.passed));
        assert!(result
            .criteria
            .iter()
            .any(|criterion| criterion.name == "regret_beats_baseline_rate" && !criterion.passed));
        assert_eq!(
            result.recommendation(),
            DeltaQPromotionRecommendation::RejectAtOfflineGate
        );
    }

    #[test]
    fn promotion_result_recommends_arena_after_gate_pass() {
        let mut report = DeltaQPromotionReport::new();
        report.compared_states = 1_500;
        report.supported_actions_sum = 6_000;
        report.candidate_regret_sum = 45.0;
        report.baseline_regret_sum = 120.0;
        report.candidate_top1_agreement_count = 1_020;
        report.negative_lift_count = 150;
        report.candidate_regret_beats_baseline_count = 1_050;
        report.candidate_top1_beats_baseline_count = 700;
        let result = evaluate_promotion_report(&report, &DeltaQPromotionThresholds::default());
        assert!(result.passed);
        assert_eq!(
            result.recommendation(),
            DeltaQPromotionRecommendation::RequiresPolicyTransferGate
        );
    }

    #[test]
    fn arena_confirmation_request_default_summary_is_stable() {
        let request = DeltaQArenaConfirmationRequest::default();
        let summary = request.summary();
        assert!(summary.contains("same_seeds=true"));
        assert!(summary.contains("same_rotation=true"));
        assert!(summary.contains("same_budget=true"));
        assert!(summary.contains("same_temp=true"));
        assert!(summary.contains("frozen_pool=true"));
        assert!(summary.contains("min_games=10000"));
    }

    #[test]
    fn delta_q_arena_report_maps_from_paired_eval() {
        let paired =
            crate::eval::paired_arena_result_from_placements(&[0, 1, 1, 2], &[1, 2, 2, 3], 0.02);
        let report = DeltaQArenaReport::from_paired_eval(&paired, -0.01);
        assert_eq!(report.compared_games, 4);
        assert!((report.lower_confidence_bound_mean_placement + 0.01).abs() < 1e-9);
        assert!((report.upper_confidence_bound_mean_placement - 0.02).abs() < 1e-9);
    }
}
