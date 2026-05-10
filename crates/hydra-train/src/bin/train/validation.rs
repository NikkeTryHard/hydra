pub(super) use hydra_train_exec::data_pipeline::TrainValidationLoader;
#[cfg(test)]
pub(super) use hydra_train_exec::validation::DeltaQPolicyTransferSnapshot;
pub(super) use hydra_train_exec::validation::{
    DeltaQPromotionSnapshot, ValidationGateDecision, ValidationSummary,
};
pub(super) use hydra_train_exec::validation_runner::{
    ValidationContext, ValidationRuntime, materialize_validation_samples, run_validation,
};
use hydra_train_runtime::validation::ValidationRunConfig;

use hydra_train_exec::resume::BestValidation;

pub(super) fn validation_loader(
    config: &hydra_train_exec::data_pipeline::StreamingLoaderConfig,
) -> TrainValidationLoader<'_> {
    TrainValidationLoader { config }
}

pub(super) fn evaluate_validation_gates(
    config: &ValidationRunConfig,
    summary: &ValidationSummary,
    best: Option<BestValidation>,
) -> ValidationGateDecision {
    hydra_train_exec::validation::evaluate_validation_gates(
        &config.gates,
        config.advanced_loss.as_ref(),
        &summary.scalar_summary(),
        best,
    )
}

pub(super) fn is_better_validation(
    summary: &ValidationSummary,
    best: Option<BestValidation>,
) -> bool {
    hydra_train_exec::validation::is_better_validation(&summary.scalar_summary(), best)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_summary(policy_loss: f64, agreement: f64) -> ValidationSummary {
        ValidationSummary {
            total_loss: policy_loss + 1.0,
            policy_loss,
            agreement,
            samples: 64,
            rare_actions: hydra_train_runtime::progress::RareActionMetrics::default(),
            saw_exit_targets: false,
            saw_delta_q_targets: false,
            profiling: None,
            delta_q_promotion: None,
            delta_q_promotion_result: None,
            delta_q_promotion_snapshot: None,
            delta_q_policy_transfer: None,
            delta_q_policy_transfer_result: None,
            delta_q_policy_transfer_snapshot: None,
        }
    }

    #[test]
    fn better_validation_rejects_higher_loss_and_lower_agreement_ties() {
        let summary = empty_summary(1.0, 0.4);

        assert!(!is_better_validation(
            &empty_summary(1.1, 0.9),
            Some(BestValidation {
                policy_loss: summary.policy_loss,
                agreement: summary.agreement,
            }),
        ));

        assert!(!is_better_validation(
            &empty_summary(summary.policy_loss, summary.agreement),
            Some(BestValidation {
                policy_loss: summary.policy_loss,
                agreement: summary.agreement,
            }),
        ));

        assert!(is_better_validation(
            &empty_summary(
                summary.policy_loss + f64::EPSILON / 2.0,
                summary.agreement + 0.05,
            ),
            Some(BestValidation {
                policy_loss: summary.policy_loss,
                agreement: summary.agreement,
            }),
        ));

        assert!(!is_better_validation(
            &empty_summary(
                summary.policy_loss + f64::EPSILON / 2.0,
                summary.agreement - 0.05,
            ),
            Some(BestValidation {
                policy_loss: summary.policy_loss,
                agreement: summary.agreement,
            }),
        ));
    }

    #[test]
    fn better_validation_accepts_first_result_without_prior_best() {
        assert!(is_better_validation(&empty_summary(1.2, 0.3), None));
    }
}
