//! Validation snapshot and gate DTOs shared across training execution seams.

use hydra_train_runtime::config::{AdvancedLossConfig, ValidationGateConfig};

use crate::resume::BestValidation;

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
/// Scalar validation summary fields shared across training execution seams.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ValidationScalarSummary {
    /// Total validation loss.
    pub total_loss: f64,
    /// Validation policy loss used for best-checkpoint ordering.
    pub policy_loss: f64,
    /// Validation policy agreement used as best-checkpoint tiebreaker.
    pub agreement: f64,
    /// Number of validation samples consumed.
    pub samples: usize,
    /// Whether ExIt sidecar targets were present in validation.
    pub saw_exit_targets: bool,
    /// Whether DeltaQ sidecar targets were present in validation.
    pub saw_delta_q_targets: bool,
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

/// Returns true when `summary` should replace the current best validation scalar.
#[must_use]
pub fn is_better_validation(
    summary: &ValidationScalarSummary,
    best: Option<BestValidation>,
) -> bool {
    match best {
        None => true,
        Some(best) => {
            summary.policy_loss < best.policy_loss
                || ((summary.policy_loss - best.policy_loss).abs() <= f64::EPSILON
                    && summary.agreement > best.agreement)
        }
    }
}

/// Evaluates scalar validation gates without depending on heavyweight validation runtime types.
#[must_use]
pub fn evaluate_validation_gates(
    gates: &ValidationGateConfig,
    advanced_loss: Option<&AdvancedLossConfig>,
    summary: &ValidationScalarSummary,
    best: Option<BestValidation>,
) -> ValidationGateDecision {
    if !gates.enabled {
        return ValidationGateDecision::disabled();
    }
    let mut criteria = Vec::new();
    if let Some(min_samples) = gates.min_validation_samples {
        let observed = summary.samples as f64;
        criteria.push(ValidationGateCriterion {
            name: "min_validation_samples".to_string(),
            passed: summary.samples >= min_samples,
            observed: Some(observed),
            threshold: Some(min_samples as f64),
            message: format!("validation samples {observed} >= {min_samples}"),
        });
    }
    if let (Some(best), Some(max_regression)) = (best, gates.max_policy_loss_regression) {
        let threshold = best.policy_loss + max_regression;
        criteria.push(ValidationGateCriterion {
            name: "max_policy_loss_regression".to_string(),
            passed: summary.policy_loss <= threshold,
            observed: Some(summary.policy_loss),
            threshold: Some(threshold),
            message: format!("policy loss {:.6} <= {:.6}", summary.policy_loss, threshold),
        });
    }
    if let (Some(best), Some(min_delta)) = (best, gates.min_policy_agreement_delta) {
        let threshold = best.agreement + min_delta;
        criteria.push(ValidationGateCriterion {
            name: "min_policy_agreement_delta".to_string(),
            passed: summary.agreement >= threshold,
            observed: Some(summary.agreement),
            threshold: Some(threshold),
            message: format!(
                "policy agreement {:.6} >= {:.6}",
                summary.agreement, threshold
            ),
        });
    }
    if gates.require_sidecar_coverage_when_weighted {
        if advanced_loss
            .and_then(|loss| loss.exit)
            .is_some_and(|weight| weight > 0.0)
        {
            criteria.push(ValidationGateCriterion {
                name: "exit_sidecar_coverage".to_string(),
                passed: summary.saw_exit_targets,
                observed: Some(if summary.saw_exit_targets { 1.0 } else { 0.0 }),
                threshold: Some(1.0),
                message: "ExIt sidecar targets present in validation".to_string(),
            });
        }
        if advanced_loss
            .and_then(|loss| loss.delta_q)
            .is_some_and(|weight| weight > 0.0)
        {
            criteria.push(ValidationGateCriterion {
                name: "delta_q_sidecar_coverage".to_string(),
                passed: summary.saw_delta_q_targets,
                observed: Some(if summary.saw_delta_q_targets {
                    1.0
                } else {
                    0.0
                }),
                threshold: Some(1.0),
                message: "DeltaQ sidecar targets present in validation".to_string(),
            });
        }
    }
    let passed = criteria.iter().all(|criterion| criterion.passed);
    ValidationGateDecision {
        enabled: true,
        passed,
        criteria,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn better_validation_prefers_lower_loss_then_higher_agreement() {
        let summary = ValidationScalarSummary {
            policy_loss: 1.0,
            agreement: 0.35,
            ..ValidationScalarSummary::default()
        };
        assert!(is_better_validation(&summary, None));
        assert!(is_better_validation(
            &summary,
            Some(BestValidation {
                policy_loss: 1.1,
                agreement: 0.60,
            })
        ));
        assert!(is_better_validation(
            &ValidationScalarSummary {
                policy_loss: 1.0,
                agreement: 0.40,
                ..ValidationScalarSummary::default()
            },
            Some(BestValidation {
                policy_loss: 1.0,
                agreement: 0.39,
            })
        ));
        assert!(!is_better_validation(
            &ValidationScalarSummary {
                policy_loss: 1.0,
                agreement: 0.40,
                ..ValidationScalarSummary::default()
            },
            Some(BestValidation {
                policy_loss: 1.0,
                agreement: 0.41,
            })
        ));
    }

    #[test]
    fn validation_gates_preserve_scalar_criteria() {
        let gates = ValidationGateConfig {
            enabled: true,
            min_validation_samples: Some(128),
            max_policy_loss_regression: Some(0.05),
            min_policy_agreement_delta: Some(0.01),
            require_sidecar_coverage_when_weighted: true,
            ..ValidationGateConfig::default()
        };
        let advanced_loss = AdvancedLossConfig {
            exit: Some(1.0),
            delta_q: Some(1.0),
            ..AdvancedLossConfig::default()
        };

        let decision = evaluate_validation_gates(
            &gates,
            Some(&advanced_loss),
            &ValidationScalarSummary {
                policy_loss: 1.04,
                agreement: 0.52,
                samples: 128,
                saw_exit_targets: true,
                saw_delta_q_targets: false,
                ..ValidationScalarSummary::default()
            },
            Some(BestValidation {
                policy_loss: 1.0,
                agreement: 0.50,
            }),
        );

        assert!(decision.enabled);
        assert!(!decision.passed);
        assert_eq!(
            decision.failed_names(),
            vec!["delta_q_sidecar_coverage".to_string()]
        );
    }
}
