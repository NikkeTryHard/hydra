use burn::module::AutodiffModule;
use burn::prelude::*;
use indicatif::ProgressBar;

use hydra_train::data::pipeline::{DataManifest, StreamingLoaderConfig, stream_val_pass};
use hydra_train::data::sample::collate_batch_samples;
use hydra_train::model::{HydraModel, HydraOutput};
use hydra_train::training::bc::{
    BcExitConfig, bc_total_with_exit, gated_bc_context, policy_agreement,
    target_actions_from_policy_target,
};
use hydra_train::training::delta_q_promotion::{
    DeltaQPolicyTransferReport, DeltaQPolicyTransferResult, DeltaQPolicyTransferThresholds,
    DeltaQPromotionReport, DeltaQPromotionResult, DeltaQPromotionThresholds,
    collect_policy_transfer_metrics_from_policy_outputs, collect_promotion_metrics_from_outputs,
    evaluate_policy_transfer_report, evaluate_promotion_report,
};
use hydra_train::training::head_gates::HeadActivationController;
use hydra_train::training::losses::{HydraLoss, HydraTargets};

use super::config::{TrainConfig, validation_microbatch_size, validation_sample_limit};
use super::progress::{BatchStats, ScalarAverages, batch_stats_from_breakdown};
use super::resume::BestValidation;
use super::{TrainBackend, ValidBackend};

pub(super) struct ValidationContext<'a> {
    pub(super) config: &'a TrainConfig,
    pub(super) loader_config: &'a StreamingLoaderConfig,
    pub(super) manifest: &'a DataManifest,
    pub(super) device: &'a <ValidBackend as Backend>::Device,
    pub(super) loss_fn: &'a HydraLoss<ValidBackend>,
    pub(super) exit_cfg: &'a BcExitConfig,
}

pub(super) struct ValidationRuntime<'a> {
    pub(super) head_controller: Option<&'a mut HeadActivationController>,
    pub(super) progress: Option<&'a ProgressBar>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(super) struct DeltaQPromotionSnapshot {
    pub(super) compared_states: u64,
    pub(super) candidate_top1_agreement: f64,
    pub(super) candidate_mean_regret: f64,
    pub(super) baseline_mean_regret: f64,
    pub(super) mean_decision_lift: f64,
    pub(super) negative_lift_fraction: f64,
    pub(super) regret_beats_baseline_rate: f64,
    pub(super) top1_beats_baseline_rate: f64,
    pub(super) passed: bool,
}

impl DeltaQPromotionSnapshot {
    fn from_report(report: &DeltaQPromotionReport, result: &DeltaQPromotionResult) -> Self {
        Self {
            compared_states: report.compared_states,
            candidate_top1_agreement: report.candidate_top1_agreement(),
            candidate_mean_regret: report.candidate_mean_regret(),
            baseline_mean_regret: report.baseline_mean_regret(),
            mean_decision_lift: report.mean_decision_lift(),
            negative_lift_fraction: report.negative_lift_fraction(),
            regret_beats_baseline_rate: report.candidate_regret_beats_baseline_rate(),
            top1_beats_baseline_rate: report.candidate_top1_beats_baseline_rate(),
            passed: result.passed,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(super) struct DeltaQPolicyTransferSnapshot {
    pub(super) compared_states: u64,
    pub(super) candidate_policy_top1_to_teacher: f64,
    pub(super) baseline_policy_top1_to_teacher: f64,
    pub(super) candidate_policy_mean_teacher_regret: f64,
    pub(super) baseline_policy_mean_teacher_regret: f64,
    pub(super) candidate_beats_baseline_rate: f64,
    pub(super) negative_transfer_fraction: f64,
}

impl DeltaQPolicyTransferSnapshot {
    fn from_report(report: &DeltaQPolicyTransferReport) -> Self {
        Self {
            compared_states: report.compared_states,
            candidate_policy_top1_to_teacher: report.candidate_policy_top1_to_teacher(),
            baseline_policy_top1_to_teacher: report.baseline_policy_top1_to_teacher(),
            candidate_policy_mean_teacher_regret: report.candidate_policy_mean_teacher_regret(),
            baseline_policy_mean_teacher_regret: report.baseline_policy_mean_teacher_regret(),
            candidate_beats_baseline_rate: report.candidate_beats_baseline_rate(),
            negative_transfer_fraction: report.negative_transfer_fraction(),
        }
    }
}

#[derive(Clone)]
pub(super) struct ValidationSummary {
    pub(super) total_loss: f64,
    pub(super) policy_loss: f64,
    pub(super) agreement: f64,
    pub(super) samples: usize,
    pub(super) delta_q_promotion: Option<DeltaQPromotionReport>,
    pub(super) delta_q_promotion_result: Option<DeltaQPromotionResult>,
    pub(super) delta_q_promotion_snapshot: Option<DeltaQPromotionSnapshot>,
    pub(super) delta_q_policy_transfer: Option<DeltaQPolicyTransferReport>,
    pub(super) delta_q_policy_transfer_result: Option<DeltaQPolicyTransferResult>,
    pub(super) delta_q_policy_transfer_snapshot: Option<DeltaQPolicyTransferSnapshot>,
}

pub(super) fn validation_batch_stats<B: Backend>(
    sample_count: usize,
    output: &HydraOutput<B>,
    batch: &hydra_train::data::sample::MjaiBatch<B>,
    targets: &HydraTargets<B>,
    loss_fn: &HydraLoss<B>,
    exit_cfg: &BcExitConfig,
) -> BatchStats {
    let target_actions = target_actions_from_policy_target(targets.policy_target.clone());
    let agreement = policy_agreement(
        output.policy_logits.clone(),
        targets.legal_mask.clone(),
        target_actions,
    );
    let breakdown = loss_fn.total_loss(output, targets);
    let mut stats = batch_stats_from_breakdown(sample_count, agreement, &breakdown);
    stats.total_loss = bc_total_with_exit(output, batch, targets, loss_fn, exit_cfg)
        .into_scalar()
        .elem::<f64>();
    stats
}

pub(super) fn is_better_validation(
    summary: &ValidationSummary,
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

pub(super) fn run_validation(
    model: &HydraModel<TrainBackend>,
    context: ValidationContext<'_>,
    runtime: ValidationRuntime<'_>,
) -> Result<ValidationSummary, String> {
    run_validation_with_policy_baseline(model, model, context, runtime)
}

pub(super) fn run_validation_with_policy_baseline(
    model: &HydraModel<TrainBackend>,
    baseline_model: &HydraModel<TrainBackend>,
    context: ValidationContext<'_>,
    runtime: ValidationRuntime<'_>,
) -> Result<ValidationSummary, String> {
    let ValidationContext {
        config,
        loader_config,
        manifest,
        device,
        loss_fn,
        exit_cfg,
    } = context;
    let ValidationRuntime {
        head_controller,
        progress,
    } = runtime;
    let model_valid = model.valid();
    let baseline_valid = baseline_model.valid();
    let validation_batch_size = validation_microbatch_size(config);
    let validation_sample_limit = validation_sample_limit(config);
    let mut stats = ScalarAverages::default();
    let mut total_samples = 0usize;
    let mut head_controller = head_controller;
    let mut delta_q_promotion = DeltaQPromotionReport::new();
    let mut delta_q_policy_transfer = DeltaQPolicyTransferReport::new();
    let mut saw_delta_q_targets = false;

    for buffer_result in stream_val_pass(manifest, loader_config, progress) {
        let buffer = buffer_result.map_err(|err| format!("validation stream failed: {err}"))?;
        for chunk in buffer.chunks(validation_batch_size) {
            if let Some(limit) = validation_sample_limit
                && total_samples >= limit
            {
                break;
            }
            let capped_chunk = if let Some(limit) = validation_sample_limit {
                let remaining = limit.saturating_sub(total_samples);
                &chunk[..chunk.len().min(remaining)]
            } else {
                chunk
            };
            if capped_chunk.is_empty() {
                break;
            }
            let Some((obs, batch)) =
                collate_batch_samples::<ValidBackend>(capped_chunk, false, device)
                    .map_err(|err| format!("validation collation failed: {err}"))?
            else {
                continue;
            };
            let targets = batch.to_hydra_targets();
            let (active_loss_fn, warmup_heads) =
                gated_bc_context(head_controller.as_deref_mut(), loss_fn, &targets);
            let output =
                model_valid.forward_with_warmup(obs.clone(), &active_loss_fn.config, &warmup_heads);
            let baseline_output = baseline_valid.forward(obs);
            let batch_stats = validation_batch_stats(
                capped_chunk.len(),
                &output,
                &batch,
                &targets,
                &active_loss_fn,
                exit_cfg,
            );
            if targets.delta_q_target.is_some() && targets.delta_q_mask.is_some() {
                delta_q_promotion.merge(&collect_promotion_metrics_from_outputs(
                    &output, &targets, 0.75,
                ));
                delta_q_policy_transfer.merge(
                    &collect_policy_transfer_metrics_from_policy_outputs(
                        output.policy_logits.clone(),
                        baseline_output.policy_logits.clone(),
                        &targets,
                    ),
                );
                saw_delta_q_targets = true;
            }
            stats.record_batch(batch_stats);
            total_samples += capped_chunk.len();
        }
        if let Some(limit) = validation_sample_limit
            && total_samples >= limit
        {
            break;
        }
    }

    if total_samples == 0 {
        Ok(ValidationSummary {
            total_loss: 0.0,
            policy_loss: 0.0,
            agreement: 0.0,
            samples: 0,
            delta_q_promotion: None,
            delta_q_promotion_result: None,
            delta_q_promotion_snapshot: None,
            delta_q_policy_transfer: None,
            delta_q_policy_transfer_result: None,
            delta_q_policy_transfer_snapshot: None,
        })
    } else {
        let stats = stats.finalize();
        let (
            delta_q_promotion,
            delta_q_promotion_result,
            delta_q_promotion_snapshot,
            delta_q_policy_transfer,
            delta_q_policy_transfer_result,
            delta_q_policy_transfer_snapshot,
        ) = if saw_delta_q_targets {
            let result = evaluate_promotion_report(
                &delta_q_promotion,
                &DeltaQPromotionThresholds::default(),
            );
            let policy_transfer_result = evaluate_policy_transfer_report(
                &delta_q_policy_transfer,
                &DeltaQPolicyTransferThresholds::default(),
            );
            let snapshot = DeltaQPromotionSnapshot::from_report(&delta_q_promotion, &result);
            let policy_transfer_snapshot =
                DeltaQPolicyTransferSnapshot::from_report(&delta_q_policy_transfer);
            (
                Some(delta_q_promotion),
                Some(result),
                Some(snapshot),
                Some(delta_q_policy_transfer),
                Some(policy_transfer_result),
                Some(policy_transfer_snapshot),
            )
        } else {
            (None, None, None, None, None, None)
        };
        Ok(ValidationSummary {
            total_loss: stats.total_loss,
            policy_loss: stats.loss_policy,
            agreement: stats.policy_agreement,
            samples: total_samples,
            delta_q_promotion,
            delta_q_promotion_result,
            delta_q_promotion_snapshot,
            delta_q_policy_transfer,
            delta_q_policy_transfer_result,
            delta_q_policy_transfer_snapshot,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    use burn::backend::libtorch::LibTorchDevice;
    use burn::tensor::Tensor;
    use hydra_train::data::sample::MjaiBatch;
    use hydra_train::model::HydraModelConfig;
    use hydra_train::preflight::PreflightConfig;
    use hydra_train::training::bc::BcExitConfig;
    use hydra_train::training::losses::HydraLossConfig;

    use crate::config::{BcHyperparamConfig, TrainConfig};
    use crate::resume::BestValidation;

    fn dummy_config() -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/data"),
            output_dir: PathBuf::from("/output"),
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(32),
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            rl: None,
            bc: BcHyperparamConfig::default(),
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: Some(2),
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 10,
            validate_every_n_steps: 10,
            checkpoint_every_n_steps: 10,
            max_train_steps: None,
            max_validation_batches: None,
            max_validation_samples: None,
            preflight: PreflightConfig::default(),
        }
    }

    fn empty_manifest() -> DataManifest {
        DataManifest {
            sources: Vec::new(),
            total_games: 0,
            train_count: 0,
            val_count: 0,
            counts_exact: true,
        }
    }

    fn empty_summary(policy_loss: f64, agreement: f64) -> ValidationSummary {
        ValidationSummary {
            total_loss: policy_loss + 1.0,
            policy_loss,
            agreement,
            samples: 64,
            delta_q_promotion: None,
            delta_q_promotion_result: None,
            delta_q_promotion_snapshot: None,
            delta_q_policy_transfer: None,
            delta_q_policy_transfer_result: None,
            delta_q_policy_transfer_snapshot: None,
        }
    }

    fn empty_batch(device: &LibTorchDevice, batch: usize) -> MjaiBatch<ValidBackend> {
        MjaiBatch {
            obs: Tensor::zeros([batch, hydra_train::config::INPUT_CHANNELS, 34], device),
            actions: Tensor::zeros([batch], device),
            legal_mask: Tensor::ones([batch, 46], device),
            value_target: Tensor::zeros([batch], device),
            grp_target: Tensor::zeros([batch, 24], device),
            oracle_target: None,
            oracle_target_mask: Tensor::zeros([batch], device),
            tenpai_target: Tensor::zeros([batch, 3], device),
            danger_target: Tensor::zeros([batch, 3, 34], device),
            danger_mask: Tensor::zeros([batch, 3, 34], device),
            safety_residual_target: None,
            safety_residual_mask: None,
            exit_target: None,
            exit_mask: None,
            delta_q_target: None,
            delta_q_mask: None,
            belief_fields_target: None,
            mixture_weight_target: None,
            belief_fields_mask: None,
            mixture_weight_mask: None,
            opp_next_target: Tensor::zeros([batch, 3, 34], device),
            score_pdf_target: Tensor::zeros([batch, 64], device),
            score_cdf_target: Tensor::zeros([batch, 64], device),
        }
    }

    #[test]
    fn delta_q_promotion_snapshot_reflects_report_metrics_and_result() {
        let report = DeltaQPromotionReport {
            eligible_states: 16,
            compared_states: 8,
            masked_entries: 2,
            supported_actions_sum: 24,
            candidate_top1_agreement_count: 6,
            baseline_top1_agreement_count: 4,
            candidate_high_gap_top1_count: 3,
            baseline_high_gap_top1_count: 2,
            high_gap_states: 5,
            candidate_regret_sum: 2.0,
            baseline_regret_sum: 4.0,
            decision_lift_sum: 1.5,
            negative_lift_count: 1,
            candidate_regret_beats_baseline_count: 7,
            candidate_top1_beats_baseline_count: 5,
        };
        let result = DeltaQPromotionResult {
            passed: true,
            criteria: Vec::new(),
        };

        let snapshot = DeltaQPromotionSnapshot::from_report(&report, &result);

        assert_eq!(snapshot.compared_states, 8);
        assert!((snapshot.candidate_top1_agreement - 0.75).abs() < 1e-12);
        assert!((snapshot.candidate_mean_regret - 0.25).abs() < 1e-12);
        assert!((snapshot.baseline_mean_regret - 0.5).abs() < 1e-12);
        assert!((snapshot.mean_decision_lift - 0.1875).abs() < 1e-12);
        assert!((snapshot.negative_lift_fraction - 0.125).abs() < 1e-12);
        assert!((snapshot.regret_beats_baseline_rate - 0.875).abs() < 1e-12);
        assert!((snapshot.top1_beats_baseline_rate - 0.625).abs() < 1e-12);
        assert!(snapshot.passed);
    }

    #[test]
    fn delta_q_policy_transfer_snapshot_reflects_report_metrics() {
        let report = DeltaQPolicyTransferReport {
            compared_states: 8,
            candidate_policy_top1_to_teacher_count: 5,
            baseline_policy_top1_to_teacher_count: 3,
            candidate_policy_regret_sum: 1.6,
            baseline_policy_regret_sum: 2.4,
            candidate_beats_baseline_count: 6,
            negative_transfer_count: 1,
        };

        let snapshot = DeltaQPolicyTransferSnapshot::from_report(&report);

        assert_eq!(snapshot.compared_states, 8);
        assert!((snapshot.candidate_policy_top1_to_teacher - 0.625).abs() < 1e-12);
        assert!((snapshot.baseline_policy_top1_to_teacher - 0.375).abs() < 1e-12);
        assert!((snapshot.candidate_policy_mean_teacher_regret - 0.2).abs() < 1e-12);
        assert!((snapshot.baseline_policy_mean_teacher_regret - 0.3).abs() < 1e-12);
        assert!((snapshot.candidate_beats_baseline_rate - 0.75).abs() < 1e-12);
        assert!((snapshot.negative_transfer_fraction - 0.125).abs() < 1e-12);
    }

    #[test]
    fn delta_q_snapshots_handle_zero_compared_states() {
        let promotion_snapshot = DeltaQPromotionSnapshot::from_report(
            &DeltaQPromotionReport::new(),
            &DeltaQPromotionResult {
                passed: false,
                criteria: Vec::new(),
            },
        );
        assert_eq!(promotion_snapshot.compared_states, 0);
        assert_eq!(promotion_snapshot.candidate_top1_agreement, 0.0);
        assert_eq!(promotion_snapshot.candidate_mean_regret, 0.0);
        assert_eq!(promotion_snapshot.baseline_mean_regret, 0.0);
        assert_eq!(promotion_snapshot.mean_decision_lift, 0.0);
        assert_eq!(promotion_snapshot.negative_lift_fraction, 0.0);
        assert_eq!(promotion_snapshot.regret_beats_baseline_rate, 0.0);
        assert_eq!(promotion_snapshot.top1_beats_baseline_rate, 0.0);
        assert!(!promotion_snapshot.passed);

        let transfer_snapshot =
            DeltaQPolicyTransferSnapshot::from_report(&DeltaQPolicyTransferReport::new());
        assert_eq!(transfer_snapshot.compared_states, 0);
        assert_eq!(transfer_snapshot.candidate_policy_top1_to_teacher, 0.0);
        assert_eq!(transfer_snapshot.baseline_policy_top1_to_teacher, 0.0);
        assert_eq!(transfer_snapshot.candidate_policy_mean_teacher_regret, 0.0);
        assert_eq!(transfer_snapshot.baseline_policy_mean_teacher_regret, 0.0);
        assert_eq!(transfer_snapshot.candidate_beats_baseline_rate, 0.0);
        assert_eq!(transfer_snapshot.negative_transfer_fraction, 0.0);
    }

    #[test]
    fn empty_summary_preserves_optional_delta_q_fields_as_none() {
        let summary = empty_summary(0.8, 0.6);

        assert_eq!(summary.total_loss, 1.8);
        assert_eq!(summary.policy_loss, 0.8);
        assert_eq!(summary.agreement, 0.6);
        assert_eq!(summary.samples, 64);
        assert!(summary.delta_q_promotion.is_none());
        assert!(summary.delta_q_promotion_result.is_none());
        assert!(summary.delta_q_promotion_snapshot.is_none());
        assert!(summary.delta_q_policy_transfer.is_none());
        assert!(summary.delta_q_policy_transfer_result.is_none());
        assert!(summary.delta_q_policy_transfer_snapshot.is_none());
    }

    #[test]
    fn empty_batch_initializes_optional_targets_and_shapes_consistently() {
        let device = LibTorchDevice::Cpu;
        let batch = empty_batch(&device, 3);

        assert_eq!(
            batch.obs.dims(),
            [3, hydra_train::config::INPUT_CHANNELS, 34]
        );
        assert_eq!(batch.actions.dims(), [3]);
        assert_eq!(batch.legal_mask.dims(), [3, 46]);
        assert_eq!(batch.grp_target.dims(), [3, 24]);
        assert_eq!(batch.tenpai_target.dims(), [3, 3]);
        assert_eq!(batch.danger_target.dims(), [3, 3, 34]);
        assert_eq!(batch.opp_next_target.dims(), [3, 3, 34]);
        assert_eq!(batch.score_pdf_target.dims(), [3, 64]);
        assert_eq!(batch.score_cdf_target.dims(), [3, 64]);
        assert!(batch.oracle_target.is_none());
        assert!(batch.exit_target.is_none());
        assert!(batch.delta_q_target.is_none());
        assert!(batch.belief_fields_target.is_none());
        assert!(batch.mixture_weight_target.is_none());
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

    #[test]
    fn validation_batch_stats_projects_breakdown_and_exit_adjusted_total() {
        let device = LibTorchDevice::Cpu;
        let model = HydraModelConfig::actor().init::<ValidBackend>(&device);
        let batch = empty_batch(&device, 2);
        let targets = batch.to_hydra_targets();
        let output = model.forward(batch.obs.clone());
        let loss_fn = HydraLoss::<ValidBackend>::new(HydraLossConfig::new());
        let exit_cfg = BcExitConfig::default();

        let stats = validation_batch_stats(2, &output, &batch, &targets, &loss_fn, &exit_cfg);
        let expected_total = bc_total_with_exit(&output, &batch, &targets, &loss_fn, &exit_cfg)
            .into_scalar()
            .elem::<f64>();

        assert_eq!(stats.sample_count, 2);
        assert!(stats.policy_agreement.is_finite());
        assert!(stats.loss_policy.is_finite());
        assert!(stats.loss_value.is_finite());
        assert!(stats.loss_grp.is_finite());
        assert!(stats.loss_tenpai.is_finite());
        assert!(stats.loss_danger.is_finite());
        assert!(stats.loss_opp_next.is_finite());
        assert!(stats.loss_score_pdf.is_finite());
        assert!(stats.loss_score_cdf.is_finite());
        assert!((stats.total_loss - expected_total).abs() < 1e-12);
    }

    #[test]
    fn run_validation_returns_zero_summary_for_empty_manifest() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = empty_manifest();
        let device = LibTorchDevice::Cpu;
        let model = HydraModelConfig::actor().init::<TrainBackend>(&device);
        let loss_fn = HydraLoss::<ValidBackend>::new(HydraLossConfig::new());

        let summary = run_validation_with_policy_baseline(
            &model,
            &model,
            ValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                device: &device,
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
            },
            ValidationRuntime {
                head_controller: None,
                progress: None,
            },
        )
        .expect("empty manifest validation should succeed");

        assert_eq!(summary.total_loss, 0.0);
        assert_eq!(summary.policy_loss, 0.0);
        assert_eq!(summary.agreement, 0.0);
        assert_eq!(summary.samples, 0);
        assert!(summary.delta_q_promotion.is_none());
        assert!(summary.delta_q_promotion_result.is_none());
        assert!(summary.delta_q_promotion_snapshot.is_none());
        assert!(summary.delta_q_policy_transfer.is_none());
        assert!(summary.delta_q_policy_transfer_result.is_none());
        assert!(summary.delta_q_policy_transfer_snapshot.is_none());
    }

    #[test]
    fn run_validation_wrapper_matches_policy_baseline_variant_on_empty_manifest() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = empty_manifest();
        let device = LibTorchDevice::Cpu;
        let model = HydraModelConfig::actor().init::<TrainBackend>(&device);
        let loss_fn = HydraLoss::<ValidBackend>::new(HydraLossConfig::new());

        let wrapped = run_validation(
            &model,
            ValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                device: &device,
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
            },
            ValidationRuntime {
                head_controller: None,
                progress: None,
            },
        )
        .expect("wrapper validation should succeed");

        let direct = run_validation_with_policy_baseline(
            &model,
            &model,
            ValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                device: &device,
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
            },
            ValidationRuntime {
                head_controller: None,
                progress: None,
            },
        )
        .expect("direct validation should succeed");

        assert_eq!(wrapped.total_loss, direct.total_loss);
        assert_eq!(wrapped.policy_loss, direct.policy_loss);
        assert_eq!(wrapped.agreement, direct.agreement);
        assert_eq!(wrapped.samples, direct.samples);
        assert!(wrapped.delta_q_promotion.is_none());
        assert!(wrapped.delta_q_policy_transfer.is_none());
    }
}
