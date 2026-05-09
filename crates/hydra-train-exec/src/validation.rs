//! Validation snapshot and gate DTOs shared across training execution seams.

/// Serializable DeltaQ promotion metrics captured with validation results.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct DeltaQPromotionSnapshot {
    /// Number of compared states contributing to the snapshot.
    pub compared_states: u64,
    /// Candidate top-1 agreement rate.
    pub candidate_top1_agreement: f64,
    /// Candidate mean regret.
    pub candidate_mean_regret: f64,
    /// Baseline mean regret.
    pub baseline_mean_regret: f64,
    /// Mean decision lift versus baseline.
    pub mean_decision_lift: f64,
    /// Fraction of decisions with negative lift.
    pub negative_lift_fraction: f64,
    /// Rate where candidate regret beats baseline regret.
    pub regret_beats_baseline_rate: f64,
    /// Rate where candidate top-1 beats baseline top-1.
    pub top1_beats_baseline_rate: f64,
    /// Whether the promotion thresholds passed.
    pub passed: bool,
}

/// Serializable DeltaQ policy-transfer metrics captured with validation results.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct DeltaQPolicyTransferSnapshot {
    /// Number of compared states contributing to the snapshot.
    pub compared_states: u64,
    /// Candidate policy top-1 agreement with teacher.
    pub candidate_policy_top1_to_teacher: f64,
    /// Baseline policy top-1 agreement with teacher.
    pub baseline_policy_top1_to_teacher: f64,
    /// Candidate policy mean teacher regret.
    pub candidate_policy_mean_teacher_regret: f64,
    /// Baseline policy mean teacher regret.
    pub baseline_policy_mean_teacher_regret: f64,
    /// Rate where candidate beats baseline.
    pub candidate_beats_baseline_rate: f64,
    /// Fraction of transfers with negative effect.
    pub negative_transfer_fraction: f64,
}

/// One validation gate criterion and its observed threshold comparison.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ValidationGateCriterion {
    /// Stable criterion name serialized into gate artifacts.
    pub name: String,
    /// Whether this criterion passed.
    pub passed: bool,
    /// Observed value, when numeric.
    pub observed: Option<f64>,
    /// Threshold value, when numeric.
    pub threshold: Option<f64>,
    /// Human-readable decision message.
    pub message: String,
}

/// Validation gate decision serialized into promotion artifacts.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ValidationGateDecision {
    /// Whether validation gates were enabled for this decision.
    pub enabled: bool,
    /// Overall pass/fail result.
    pub passed: bool,
    /// Criteria evaluated for this decision.
    pub criteria: Vec<ValidationGateCriterion>,
}

impl ValidationGateDecision {
    /// Decision returned when validation gates are disabled.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            passed: true,
            criteria: Vec::new(),
        }
    }

    /// Names of failed validation criteria.
    #[must_use]
    pub fn failed_names(&self) -> Vec<String> {
        self.criteria
            .iter()
            .filter(|criterion| !criterion.passed)
            .map(|criterion| criterion.name.clone())
            .collect()
    }
}
