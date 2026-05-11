use hydra_train_runtime::config::{AdvancedLossConfig, ValidationGateConfig};

use crate::resume::BestValidation;
use crate::validation::{ValidationScalarSummary, evaluate_validation_gates, is_better_validation};

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
fn better_validation_rejects_higher_loss_and_lower_agreement_ties() {
    let best = BestValidation {
        policy_loss: 1.0,
        agreement: 0.4,
    };

    assert!(!is_better_validation(
        &ValidationScalarSummary {
            policy_loss: 1.1,
            agreement: 0.9,
            ..ValidationScalarSummary::default()
        },
        Some(best),
    ));

    assert!(!is_better_validation(
        &ValidationScalarSummary {
            policy_loss: best.policy_loss,
            agreement: best.agreement,
            ..ValidationScalarSummary::default()
        },
        Some(best),
    ));

    assert!(is_better_validation(
        &ValidationScalarSummary {
            policy_loss: best.policy_loss + f64::EPSILON / 2.0,
            agreement: best.agreement + 0.05,
            ..ValidationScalarSummary::default()
        },
        Some(best),
    ));

    assert!(!is_better_validation(
        &ValidationScalarSummary {
            policy_loss: best.policy_loss + f64::EPSILON / 2.0,
            agreement: best.agreement - 0.05,
            ..ValidationScalarSummary::default()
        },
        Some(best),
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
