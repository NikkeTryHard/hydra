use super::*;

use crate::bc_runtime::bc_total_with_exit_from_breakdown;
use crate::data::sample::MjaiBatch;
use crate::model::{HydraModelConfig, HydraModelInit};
use burn::backend::libtorch::LibTorchDevice;
use burn::tensor::Tensor;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use hydra_train_types::losses::HydraLossConfig;

type TrainBackend = burn::backend::Autodiff<burn::backend::LibTorch>;
type TestValidBackend = ValidBackendOf<TrainBackend>;

struct EmptyLoader;

impl ValidationDataLoader for EmptyLoader {
    fn stream_val_microbatches(
        &self,
        _manifest: &DataManifest,
        _microbatch_size: usize,
        _progress: Option<&ProgressBar>,
    ) -> Box<dyn Iterator<Item = io::Result<Vec<MjaiSample>>> + '_> {
        Box::new(std::iter::empty())
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

fn dummy_config() -> hydra_train_runtime::config::TrainConfig {
    hydra_train_runtime::config::TrainConfig {
        data_dir: std::path::PathBuf::new(),
        output_dir: std::path::PathBuf::new(),
        num_epochs: 1,
        batch_size: 1,
        microbatch_size: None,
        validation_microbatch_size: Some(1),
        exit_sidecar_path: None,
        delta_q_sidecar_path: None,
        bc_shards_manifest_path: None,
        shard_prefetch_depth: None,
        train_fraction: 0.9,
        source_filters: hydra_data_core::SourceFilterConfig::default(),
        augment: false,
        resume_checkpoint: None,
        seed: 0,
        advanced_loss: None,
        validation_gates: hydra_train_runtime::config::ValidationGateConfig::default(),
        rl: None,
        bc: hydra_train_runtime::config::BcHyperparamConfig::default(),
        nsight_trace: None,
        device: "cpu".to_string(),
        precision_mode: hydra_train_runtime::config::PrecisionMode::Fp32,
        buffer_games: 1,
        buffer_samples: 1,
        num_threads: Some(1),
        tensorboard: false,
        archive_queue_bound: 1,
        validation_every_n_epochs: 1,
        max_skip_logs_per_source: 1,
        log_every_n_steps: 1,
        validate_every_n_steps: 0,
        checkpoint_every_n_steps: 0,
        max_train_steps: None,
        max_validation_batches: None,
        max_validation_samples: Some(1),
        preflight: hydra_train_runtime::preflight::PreflightConfig::default(),
    }
}

fn tiny_validation_model_config() -> HydraModelConfig {
    HydraModelConfig::new(1)
        .with_input_channels(NUM_CHANNELS)
        .with_hidden_channels(4)
        .with_num_groups(4)
        .with_se_bottleneck(1)
}

fn empty_batch(device: &LibTorchDevice, batch: usize) -> MjaiBatch<TestValidBackend> {
    MjaiBatch {
        obs: Tensor::zeros([batch, NUM_CHANNELS, 34], device),
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
        target_presence: None,
    }
}

fn delta_q_sample() -> MjaiSample {
    let mut sample = MjaiSample {
        obs: [0.0; OBS_SIZE],
        action: 0,
        legal_mask: [1.0; HYDRA_ACTION_SPACE],
        placement: 0,
        score_delta: 0,
        grp_label: 0,
        oracle_target: None,
        tenpai: [0.0; 3],
        opp_next: [255; 3],
        danger: [0.0; 102],
        danger_mask: [0.0; 102],
        safety_residual: None,
        safety_residual_mask: None,
        exit_target: None,
        exit_mask: None,
        delta_q_target: None,
        delta_q_mask: None,
        belief_fields: None,
        mixture_weights: None,
        belief_fields_present: false,
        mixture_weights_present: false,
    };
    sample.delta_q_target = Some([0.0; HYDRA_ACTION_SPACE]);
    sample.delta_q_mask = Some([1.0; HYDRA_ACTION_SPACE]);
    sample
}

fn tensor_rows_f32<const D: usize>(tensor: Tensor<TestValidBackend, D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("tensor data should be readable as f32")
        .to_vec()
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

    let snapshot = delta_q_promotion_snapshot_from_report(&report, &result);

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

    let snapshot = delta_q_policy_transfer_snapshot_from_report(&report);

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
    let promotion_snapshot = delta_q_promotion_snapshot_from_report(
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
        delta_q_policy_transfer_snapshot_from_report(&DeltaQPolicyTransferReport::new());
    assert_eq!(transfer_snapshot.compared_states, 0);
    assert_eq!(transfer_snapshot.candidate_policy_top1_to_teacher, 0.0);
    assert_eq!(transfer_snapshot.baseline_policy_top1_to_teacher, 0.0);
    assert_eq!(transfer_snapshot.candidate_policy_mean_teacher_regret, 0.0);
    assert_eq!(transfer_snapshot.baseline_policy_mean_teacher_regret, 0.0);
    assert_eq!(transfer_snapshot.candidate_beats_baseline_rate, 0.0);
    assert_eq!(transfer_snapshot.negative_transfer_fraction, 0.0);
}

#[test]
fn validation_batch_stats_projects_breakdown_and_exit_adjusted_total() {
    let device = LibTorchDevice::Cpu;
    let model = tiny_validation_model_config().init::<TestValidBackend>(&device);
    let batch = empty_batch(&device, 2);
    let targets = batch.to_hydra_targets();
    let output = model.forward(batch.obs.clone());
    let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());
    let exit_cfg = BcExitConfig::default();

    let breakdown = loss_fn.total_loss(&output, &targets);
    let total = bc_total_with_exit_from_breakdown(&output, &batch, &breakdown, &exit_cfg);
    let stats = crate::bc_metrics::batch_stats_from_outputs(
        2,
        output.policy_logits.clone(),
        targets.legal_mask.clone(),
        batch.actions.clone(),
        total.clone(),
        &breakdown,
    );
    let expected_total: f64 = total.clone().into_scalar().elem();

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
    let loader = EmptyLoader;
    let manifest = empty_manifest();
    let device = LibTorchDevice::Cpu;
    let model = tiny_validation_model_config().init::<TrainBackend>(&device);
    let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());

    let summary = run_validation_with_policy_baseline(
        &model,
        &model,
        ValidationContext {
            config: &config,
            loader: &loader,
            manifest: &manifest,
            cached_samples: None,
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
    assert_eq!(
        summary.profiling.as_ref().map(|p| p.stage.as_str()),
        Some("validation")
    );
    assert!(summary.delta_q_promotion.is_none());
    assert!(summary.delta_q_promotion_result.is_none());
    assert!(summary.delta_q_promotion_snapshot.is_none());
    assert!(summary.delta_q_policy_transfer.is_none());
    assert!(summary.delta_q_policy_transfer_result.is_none());
    assert!(summary.delta_q_policy_transfer_snapshot.is_none());
}

#[test]
fn run_validation_same_model_short_circuits_baseline_policy_forward() {
    let device = LibTorchDevice::Cpu;
    let model = tiny_validation_model_config().init::<TrainBackend>(&device);
    let valid = model.valid();
    let config = dummy_config();
    let loader = EmptyLoader;
    let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());
    let cached_samples = vec![vec![delta_q_sample()].into_boxed_slice()].into_boxed_slice();

    let summary = run_validation_with_policy_baseline(
        &model,
        &model,
        ValidationContext {
            config: &config,
            loader: &loader,
            manifest: &empty_manifest(),
            cached_samples: Some(&cached_samples),
            device: &device,
            loss_fn: &loss_fn,
            exit_cfg: &BcExitConfig::default(),
        },
        ValidationRuntime {
            head_controller: None,
            progress: None,
        },
    )
    .expect("same-model validation should succeed");

    let profiling = summary.profiling.expect("profiling should exist");
    let baseline_stage = profiling
        .children
        .iter()
        .find(|child| child.stage == PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD)
        .expect("baseline child stage should exist");
    assert_eq!(baseline_stage.elapsed_seconds, 0.0);

    let obs = Tensor::zeros([1, NUM_CHANNELS, 34], &device);
    let (policy_only_logits, _) = valid.forward_policy_value(obs.clone());
    let full_logits = valid.forward(obs).policy_logits;
    let policy_only_rows = tensor_rows_f32(policy_only_logits);
    let full_rows = tensor_rows_f32(full_logits);
    assert_eq!(policy_only_rows, full_rows);
}

#[test]
fn run_validation_distinct_baseline_uses_policy_only_forward_without_drift() {
    let device = LibTorchDevice::Cpu;
    let model = tiny_validation_model_config().init::<TrainBackend>(&device);
    let baseline = tiny_validation_model_config().init::<TrainBackend>(&device);
    let baseline_valid = baseline.valid();
    let config = dummy_config();
    let loader = EmptyLoader;
    let loss_fn = HydraLoss::<TestValidBackend>::new(HydraLossConfig::new());
    let cached_samples = vec![vec![delta_q_sample()].into_boxed_slice()].into_boxed_slice();

    let summary = run_validation_with_policy_baseline(
        &model,
        &baseline,
        ValidationContext {
            config: &config,
            loader: &loader,
            manifest: &empty_manifest(),
            cached_samples: Some(&cached_samples),
            device: &device,
            loss_fn: &loss_fn,
            exit_cfg: &BcExitConfig::default(),
        },
        ValidationRuntime {
            head_controller: None,
            progress: None,
        },
    )
    .expect("distinct-baseline validation should succeed");

    assert!(summary.delta_q_promotion.is_some());
    assert!(summary.delta_q_policy_transfer.is_some());

    let obs = Tensor::zeros([1, NUM_CHANNELS, 34], &device);
    let (policy_only_logits, _) = baseline_valid.forward_policy_value(obs.clone());
    let full_logits = baseline_valid.forward(obs).policy_logits;
    let policy_only_rows = tensor_rows_f32(policy_only_logits);
    let full_rows = tensor_rows_f32(full_logits);
    assert_eq!(policy_only_rows, full_rows);
}
