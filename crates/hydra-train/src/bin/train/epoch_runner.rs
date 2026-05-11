#[cfg(test)]
pub(super) use hydra_train_exec::artifacts::{
    BcArtifactPaths, LatestCheckpointState, save_latest_checkpoint_and_state,
};
#[cfg(test)]
pub(super) use hydra_train_exec::epoch_runner::{
    EpochEndValidationContext, EpochFinalizeContext, EpochRunnerContext, EpochRuntimeMut,
    IntervalStepSummaryContext, PeriodicCheckpointContext, PeriodicCheckpointState,
    TrainLogicalBatchConfig, ValidationExecutor, ValidationStepContext, bc_epoch_profiling,
    bc_interval_profiling, build_epoch_continuation, child_elapsed_seconds,
    emit_interval_step_summary, finalize_epoch_outputs,
    interval_timing_input_for_config as interval_timing_input, maybe_run_interval_validation,
    maybe_run_interval_validation_with_executor, maybe_save_periodic_checkpoint,
    record_drained_batch_stats, run_epoch, run_epoch_end_validation,
    run_epoch_end_validation_with_executor, should_run_epoch_end_validation, train_logical_batch,
};
#[cfg(test)]
pub(super) use hydra_train_exec::progress::TrainSubStageTiming;

#[cfg(test)]
pub(super) use hydra_train_algo::bc::BcExitConfig;
#[cfg(test)]
pub(super) use hydra_train_exec::data::sample::MjaiSample;
#[cfg(test)]
pub(super) use hydra_train_exec::data_pipeline::{DataManifest, StreamingLoaderConfig};
#[cfg(test)]
pub(super) use hydra_train_exec::losses::HydraLoss;
#[cfg(test)]
pub(super) use hydra_train_exec::model::HydraModel;
#[cfg(test)]
pub(super) use hydra_train_exec::resume::{
    BestValidation, EpochContinuation, RuntimeResumeContract,
};
#[cfg(test)]
pub(super) use hydra_train_exec::validation::ValidationSummary;
#[cfg(test)]
pub(super) use hydra_train_runtime::preflight::{
    PROFILING_STAGE_CHECKPOINT, PROFILING_STAGE_H2D_TRANSFER, PROFILING_STAGE_LOGGING,
    PROFILING_STAGE_PRODUCER_WAIT,
};
#[cfg(test)]
pub(super) use hydra_train_runtime::progress::{BatchStats, ScalarAverages};

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_train_exec::epoch_runner as exec_epoch;
    use hydra_train_types::config::BCTrainerConfig;

    use std::fs;
    use std::path::{Path, PathBuf};
    use std::thread;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    use burn::backend::libtorch::LibTorchDevice;
    use burn::optim::AdamConfig;
    use burn::tensor::backend::AutodiffBackend;
    use hydra_train_exec::model::{HydraModelConfig, HydraModelInit};
    use hydra_train_exec::validation_runner::{ValidationContext, ValidationRuntime};
    use hydra_train_runtime::head_gates::{HeadActivationConfig, HeadActivationController};
    use hydra_train_runtime::preflight::PreflightConfig;
    use hydra_train_types::losses::HydraLossConfig;
    use indicatif::MultiProgress;
    use tboard::EventWriter;

    use crate::TrainBackend;

    type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;
    type TestValidBackend = ValidBackendOf<TrainBackend>;

    use crate::config::{BcHyperparamConfig, TrainConfig};
    use crate::resume::read_resume_state;

    fn batch_stats(sample_count: usize, total_loss: f64, policy_agreement: f64) -> BatchStats {
        BatchStats {
            sample_count,
            batch_count: 1,
            total_loss,
            policy_agreement,
            loss_policy: total_loss + 0.1,
            loss_value: total_loss + 0.2,
            loss_grp: total_loss + 0.3,
            loss_tenpai: total_loss + 0.4,
            loss_danger: total_loss + 0.5,
            loss_opp_next: total_loss + 0.6,
            loss_score_pdf: total_loss + 0.7,
            loss_score_cdf: total_loss + 0.8,
            rare_actions: crate::progress::RareActionMetrics::default(),
        }
    }

    fn dummy_config() -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/data"),
            output_dir: PathBuf::from("/output"),
            num_epochs: 5,
            batch_size: 16,
            microbatch_size: Some(4),
            validation_microbatch_size: Some(4),
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            bc_shards_manifest_path: None,
            shard_prefetch_depth: None,
            train_fraction: 0.9,
            source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            validation_gates: crate::config::ValidationGateConfig::default(),
            rl: None,
            bc: BcHyperparamConfig::default(),
            nsight_trace: None,
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: Some(2),
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 2,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 5,
            validate_every_n_steps: 4,
            checkpoint_every_n_steps: 5,
            max_train_steps: Some(20),
            max_validation_batches: None,
            max_validation_samples: None,
            preflight: PreflightConfig::default(),
            precision_mode: crate::config::PrecisionMode::Fp32,
        }
    }

    fn dummy_manifest(counts_exact: bool) -> DataManifest {
        DataManifest {
            sources: Vec::new(),
            total_games: 24,
            train_count: 18,
            val_count: 6,
            counts_exact,
        }
    }

    fn dummy_validation_summary(policy_loss: f64, agreement: f64) -> ValidationSummary {
        ValidationSummary {
            total_loss: policy_loss + 0.5,
            policy_loss,
            agreement,
            samples: 64,
            rare_actions: crate::progress::RareActionMetrics::default(),
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

    struct FakeValidationExecutor {
        calls: usize,
        summary: ValidationSummary,
    }

    impl ValidationExecutor<TrainBackend> for FakeValidationExecutor {
        fn run_validation(
            &mut self,
            _model: &HydraModel<TrainBackend>,
            _context: ValidationContext<
                '_,
                TrainConfig,
                hydra_train_exec::data_pipeline::TrainValidationLoader<'_>,
                ValidBackendOf<TrainBackend>,
            >,
            _runtime: ValidationRuntime<'_>,
        ) -> Result<ValidationSummary, String> {
            self.calls += 1;
            Ok(self.summary.clone())
        }
    }

    fn dummy_runtime_resume_contract() -> RuntimeResumeContract {
        RuntimeResumeContract {
            batch_size: 16,
            train_microbatch_size: 4,
            validation_microbatch_size: 4,
            accum_steps: 4,
            precision_mode: crate::config::PrecisionMode::Fp32,
        }
    }

    fn temp_dir_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("hydra_epoch_runner_{label}_{unique}"))
    }

    fn test_artifacts(label: &str) -> BcArtifactPaths {
        let output_dir = temp_dir_path(label);
        fs::create_dir_all(&output_dir).expect("create test output dir");
        let artifacts = BcArtifactPaths::new(&output_dir, 0);
        artifacts.create_root_dir().expect("create artifacts root");
        artifacts
    }

    fn tiny_dummy_model(device: &LibTorchDevice) -> HydraModel<TrainBackend> {
        HydraModelConfig::new(1)
            .with_input_channels(hydra_core::encoder::NUM_CHANNELS)
            .with_hidden_channels(4)
            .with_num_groups(4)
            .with_se_bottleneck(1)
            .init::<TrainBackend>(device)
    }

    fn dummy_train_sample(action: u8) -> MjaiSample {
        let mut legal_mask = [0.0f32; hydra_core::action::HYDRA_ACTION_SPACE];
        legal_mask[action as usize] = 1.0;
        legal_mask[45] = 1.0;
        MjaiSample {
            obs: [0.1f32; hydra_core::encoder::OBS_SIZE],
            action,
            legal_mask,
            placement: 0,
            score_delta: 0,
            grp_label: 0,
            oracle_target: None,
            tenpai: [0.0; 3],
            opp_next: [0, 1, 255],
            danger: [0.0; 102],
            danger_mask: [1.0; 102],
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
        }
    }

    fn dummy_valid_loss() -> HydraLoss<TestValidBackend> {
        HydraLoss::<TestValidBackend>::new(HydraLossConfig::new())
    }

    fn dummy_train_loss() -> HydraLoss<TrainBackend> {
        HydraLoss::<TrainBackend>::new(HydraLossConfig::new())
    }

    fn read_jsonl_entry(path: &Path) -> serde_json::Value {
        let raw = fs::read_to_string(path).expect("read jsonl file");
        let line = raw.lines().next().expect("jsonl entry line");
        serde_json::from_str(line).expect("parse jsonl entry")
    }

    fn modified_time(path: &Path) -> SystemTime {
        fs::metadata(path)
            .expect("read file metadata")
            .modified()
            .expect("read file modified time")
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn train_logical_batch_empty_keeps_model_slot_populated() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch: Vec<MjaiSample> = Vec::new();

        let (drained, _sub_timing) = train_logical_batch(
            &logical_batch,
            TrainLogicalBatchConfig {
                microbatch_size: 1,
                use_amp: false,
                augment: false,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                lr: 1.0e-4,
            },
            &mut head_controller,
            &mut model_slot,
            &mut optimizer,
        )
        .expect("empty logical batch should succeed");

        assert!(drained.is_empty());
        assert!(model_slot.is_some());
    }

    #[test]
    fn train_logical_batch_reports_clear_error_when_model_slot_is_empty() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = None;
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![dummy_train_sample(0)];

        let result = train_logical_batch(
            &logical_batch,
            TrainLogicalBatchConfig {
                microbatch_size: 1,
                use_amp: false,
                augment: false,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                lr: 1.0e-4,
            },
            &mut head_controller,
            &mut model_slot,
            &mut optimizer,
        );

        let err = match result {
            Ok(_) => panic!("empty model slot should return a clear error"),
            Err(err) => err,
        };

        assert!(err.contains("epoch runner model slot should stay populated"));
        assert!(model_slot.is_none());
    }

    #[test]
    fn train_logical_batch_non_empty_keeps_model_slot_populated_and_returns_stats() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];

        let (drained, _sub_timing) = train_logical_batch(
            &logical_batch,
            TrainLogicalBatchConfig {
                microbatch_size: 1,
                use_amp: false,
                augment: false,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                lr: 1.0e-4,
            },
            &mut head_controller,
            &mut model_slot,
            &mut optimizer,
        )
        .expect("train logical batch with samples");

        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].sample_count, 2);
        assert_eq!(drained[0].batch_count, 2);
        assert!(drained.iter().all(|stats| stats.total_loss.is_finite()));
        assert!(
            drained
                .iter()
                .all(|stats| stats.policy_agreement.is_finite())
        );
        assert!(model_slot.is_some());
    }

    #[test]
    fn train_logical_batch_full_microbatch_keeps_model_slot_populated() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];

        let (drained, _sub_timing) = train_logical_batch(
            &logical_batch,
            TrainLogicalBatchConfig {
                microbatch_size: logical_batch.len(),
                use_amp: false,
                augment: false,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                lr: 1.0e-4,
            },
            &mut head_controller,
            &mut model_slot,
            &mut optimizer,
        )
        .expect("full microbatch path should succeed");

        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].sample_count, logical_batch.len());
        assert_eq!(drained[0].batch_count, 1);
        assert!(drained.iter().all(|stats| stats.total_loss.is_finite()));
        assert!(
            drained
                .iter()
                .all(|stats| stats.policy_agreement.is_finite())
        );
        assert!(model_slot.is_some());
    }

    #[test]
    fn train_logical_batch_records_microbatch_sub_stage_scope_order() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];

        let (_, events) = crate::nvtx::with_test_recorder(|| {
            train_logical_batch(
                &logical_batch,
                TrainLogicalBatchConfig {
                    microbatch_size: 1,
                    use_amp: false,
                    augment: false,
                    train_device: &device,
                    loss_fn: &train_loss_fn,
                    bc_exit_cfg: &BcExitConfig::default(),
                    lr: 1.0e-4,
                },
                &mut head_controller,
                &mut model_slot,
                &mut optimizer,
            )
            .expect("train logical batch with NVTX recording");
        });

        assert!(
            events.contains(&"push:collation".to_string()),
            "should record collation sub-stage"
        );
        assert!(
            events.contains(&"push:forward".to_string()),
            "should record forward sub-stage"
        );
        assert!(
            events.contains(&"push:loss".to_string()),
            "should record loss sub-stage"
        );
        assert!(
            events.contains(&"push:backward".to_string()),
            "should record backward sub-stage"
        );
        assert!(
            events.contains(&"push:optimizer_step".to_string()),
            "should record optimizer_step sub-stage"
        );

        for push_event in events.iter().filter(|e| e.starts_with("push:")) {
            let stage = push_event.strip_prefix("push:").unwrap();
            let pop = format!("pop:{stage}");
            assert!(
                events.contains(&pop),
                "every push should have a matching pop: {push_event}"
            );
            let push_idx = events.iter().position(|e| e == push_event).unwrap();
            let pop_idx = events.iter().position(|e| e == &pop).unwrap();
            assert!(pop_idx > push_idx, "pop should come after push for {stage}");
        }
    }

    #[test]
    fn train_logical_batch_sub_stage_timing_has_nonzero_values() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];

        let (_drained, sub_timing) = train_logical_batch(
            &logical_batch,
            TrainLogicalBatchConfig {
                microbatch_size: 1,
                use_amp: false,
                augment: false,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                lr: 1.0e-4,
            },
            &mut head_controller,
            &mut model_slot,
            &mut optimizer,
        )
        .expect("train logical batch for sub-timing check");

        assert!(
            sub_timing.collation_seconds > 0.0,
            "collation should have measurable time"
        );
        assert!(
            sub_timing.forward_seconds > 0.0,
            "forward should have measurable time"
        );
        assert!(
            sub_timing.loss_seconds > 0.0,
            "loss should have measurable time"
        );
        assert!(
            sub_timing.backward_seconds > 0.0,
            "backward should have measurable time"
        );
        assert!(
            sub_timing.optimizer_step_seconds > 0.0,
            "optimizer_step should have measurable time"
        );
        assert!(
            sub_timing.metric_readback_seconds > 0.0,
            "metric_readback should have measurable time"
        );

        let children = sub_timing.to_profiling_children();
        assert_eq!(children.len(), 8);
        assert!(
            children.iter().all(|c| {
                c.stage == PROFILING_STAGE_PRODUCER_WAIT
                    || c.stage == PROFILING_STAGE_H2D_TRANSFER
                    || c.elapsed_seconds > 0.0
            }),
            "active profiling children should have positive elapsed_seconds"
        );
    }

    #[test]
    fn emit_interval_step_summary_records_logging_scope_order() {
        let artifacts = test_artifacts("nvtx_interval_logging_scope");
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let multi = MultiProgress::new();
        let config = dummy_config();
        let manifest = dummy_manifest(true);
        let validation_summary = dummy_validation_summary(0.4, 0.7);

        let (_, events) = crate::nvtx::with_test_recorder(|| {
            emit_interval_step_summary(
                &multi,
                &mut tb,
                &mut step_log,
                IntervalStepSummaryContext {
                    manifest: &manifest,
                    config: &config,
                    session_start_global_step: 0,
                    global_step: 5,
                    epoch: 1,
                    lr: 1.0e-4,
                    best_validation: Some(BestValidation {
                        policy_loss: 0.5,
                        agreement: 0.6,
                    }),
                    val_summary: Some(validation_summary),
                    seen_samples: 16,
                    assumed_games_seen: 4,
                    epoch_optimizer_steps: 5,
                    window_stats: ScalarAverages::default().finalize(),
                    step_rate: 12.0,
                    profiling: None,
                    advisories: Vec::new(),
                },
            )
            .expect("emit interval step summary should succeed");
        });

        assert_eq!(events, vec!["push:logging", "pop:logging"]);
    }

    #[test]
    fn epoch_end_validation_runs_on_interval_or_final_epoch() {
        assert!(should_run_epoch_end_validation(0, 3, 1));
        assert!(!should_run_epoch_end_validation(0, 3, 2));
        assert!(should_run_epoch_end_validation(1, 3, 2));
        assert!(should_run_epoch_end_validation(2, 3, 5));
    }

    #[test]
    fn epoch_end_validation_skips_non_boundary_epochs() {
        assert!(!should_run_epoch_end_validation(0, 5, 3));
        assert!(!should_run_epoch_end_validation(1, 5, 3));
        assert!(should_run_epoch_end_validation(2, 5, 3));
        assert!(!should_run_epoch_end_validation(3, 5, 3));
    }

    #[test]
    fn epoch_end_validation_uses_injected_executor_on_boundary() {
        let mut config = dummy_config();
        config.num_epochs = 5;
        config.validation_every_n_epochs = 2;
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("epoch_validation_executor_seam");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = None;
        let mut executor = FakeValidationExecutor {
            calls: 0,
            summary: dummy_validation_summary(0.33, 0.77),
        };

        let summary = run_epoch_end_validation_with_executor(
            1,
            &model,
            EpochEndValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                cached_validation_samples: None,
            },
            None,
            &mut best_validation,
            2.5,
            &mut executor,
        )
        .expect("epoch-end validation through fake executor")
        .expect("epoch boundary returns validation summary");

        assert_eq!(executor.calls, 1);
        assert_eq!(summary.policy_loss, 0.33);
        assert_eq!(best_validation.map(|best| best.policy_loss), Some(0.33));
    }

    #[test]
    fn epoch_end_validation_skip_does_not_call_injected_executor() {
        let mut config = dummy_config();
        config.num_epochs = 5;
        config.validation_every_n_epochs = 3;
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("epoch_validation_executor_skip");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = None;
        let mut executor = FakeValidationExecutor {
            calls: 0,
            summary: dummy_validation_summary(0.33, 0.77),
        };

        let summary = run_epoch_end_validation_with_executor(
            0,
            &model,
            EpochEndValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                cached_validation_samples: None,
            },
            None,
            &mut best_validation,
            2.5,
            &mut executor,
        )
        .expect("skip epoch-end validation through fake executor");

        assert!(summary.is_none());
        assert_eq!(executor.calls, 0);
        assert_eq!(best_validation, None);
    }

    #[test]
    fn build_epoch_continuation_matches_completion_state() {
        let completed = build_epoch_continuation(2, true, 99);
        assert_eq!(completed.next_epoch, 3);
        assert_eq!(completed.skip_optimizer_steps_in_epoch, 0);
        assert!(completed.epoch_completed);

        let partial = build_epoch_continuation(2, false, 99);
        assert_eq!(partial.next_epoch, 2);
        assert_eq!(partial.skip_optimizer_steps_in_epoch, 99);
        assert!(!partial.epoch_completed);
    }

    #[test]
    fn build_epoch_continuation_resets_skip_count_after_empty_completed_epoch() {
        let continuation = build_epoch_continuation(7, true, 0);

        assert_eq!(continuation.next_epoch, 8);
        assert_eq!(continuation.skip_optimizer_steps_in_epoch, 0);
        assert!(continuation.epoch_completed);
    }

    #[test]
    fn epoch_end_validation_runs_on_final_epoch_even_when_interval_is_larger() {
        assert!(should_run_epoch_end_validation(4, 5, 10));
        assert!(!should_run_epoch_end_validation(3, 5, 10));
    }

    #[test]
    fn record_drained_batch_stats_updates_both_accumulators_with_weighted_values() {
        let mut stats = ScalarAverages::default();
        let mut window = ScalarAverages::default();

        record_drained_batch_stats(
            vec![batch_stats(2, 1.5, 0.25), batch_stats(3, 4.0, 0.75)],
            &mut stats,
            &mut window,
        );

        let stats = stats.finalize();
        let window = window.finalize();

        for aggregate in [stats, window] {
            assert_eq!(aggregate.num_batches, 2);
            assert_eq!(aggregate.num_samples, 5);
            assert!((aggregate.total_loss - 3.0).abs() < 1e-12);
            assert!((aggregate.policy_agreement - 0.55).abs() < 1e-12);
            assert!((aggregate.loss_policy - 3.1).abs() < 1e-12);
            assert!((aggregate.loss_value - 3.2).abs() < 1e-12);
            assert!((aggregate.loss_grp - 3.3).abs() < 1e-12);
            assert!((aggregate.loss_tenpai - 3.4).abs() < 1e-12);
            assert!((aggregate.loss_danger - 3.5).abs() < 1e-12);
            assert!((aggregate.loss_opp_next - 3.6).abs() < 1e-12);
            assert!((aggregate.loss_score_pdf - 3.7).abs() < 1e-12);
            assert!((aggregate.loss_score_cdf - 3.8).abs() < 1e-12);
        }
    }

    #[test]
    fn record_drained_batch_stats_ignores_empty_drains() {
        let mut stats = ScalarAverages::default();
        let mut window = ScalarAverages::default();

        record_drained_batch_stats(Vec::new(), &mut stats, &mut window);

        let stats = stats.finalize();
        let window = window.finalize();

        assert_eq!(stats.num_batches, 0);
        assert_eq!(stats.num_samples, 0);
        assert_eq!(stats.total_loss, 0.0);
        assert_eq!(window.num_batches, 0);
        assert_eq!(window.num_samples, 0);
        assert_eq!(window.total_loss, 0.0);
    }

    #[test]
    fn record_drained_batch_stats_preserves_zero_weight_guard_for_both_accumulators() {
        let mut stats = ScalarAverages::default();
        let mut window = ScalarAverages::default();

        record_drained_batch_stats(
            vec![batch_stats(0, 99.0, 0.99), batch_stats(4, 2.5, 0.4)],
            &mut stats,
            &mut window,
        );

        let stats = stats.finalize();
        let window = window.finalize();

        for aggregate in [stats, window] {
            assert_eq!(aggregate.num_batches, 1);
            assert_eq!(aggregate.num_samples, 4);
            assert!((aggregate.total_loss - 2.5).abs() < 1e-12);
            assert!((aggregate.policy_agreement - 0.4).abs() < 1e-12);
            assert!((aggregate.loss_policy - 2.6).abs() < 1e-12);
        }
    }

    #[test]
    fn interval_validation_uses_injected_executor_on_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_executor_seam");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.8,
            agreement: 0.6,
        });
        let mut executor = FakeValidationExecutor {
            calls: 0,
            summary: dummy_validation_summary(0.25, 0.75),
        };
        let multi = MultiProgress::new();

        let summary = maybe_run_interval_validation_with_executor(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            14,
            1.5,
            &mut executor,
        )
        .expect("interval validation through fake executor")
        .expect("boundary returns validation summary");

        assert_eq!(executor.calls, 1);
        assert_eq!(summary.samples, 64);
        assert_eq!(summary.policy_loss, 0.25);
        assert_eq!(best_validation.map(|best| best.policy_loss), Some(0.25));
    }

    #[test]
    fn interval_validation_skip_does_not_call_injected_executor() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_executor_skip");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = None;
        let mut executor = FakeValidationExecutor {
            calls: 0,
            summary: dummy_validation_summary(0.25, 0.75),
        };
        let multi = MultiProgress::new();

        let summary = maybe_run_interval_validation_with_executor(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            13,
            1.5,
            &mut executor,
        )
        .expect("skip interval validation through fake executor");

        assert!(summary.is_none());
        assert_eq!(executor.calls, 0);
        assert_eq!(best_validation, None);
    }
    #[test]
    fn maybe_run_interval_validation_skips_until_step_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_skip");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.8,
            agreement: 0.6,
        });
        let multi = MultiProgress::new();

        let zero_step = maybe_run_interval_validation(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            10,
            1.5,
        )
        .expect("skip zero session step validation");

        let off_interval = maybe_run_interval_validation(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            13,
            1.5,
        )
        .expect("skip non-boundary validation");

        assert!(zero_step.is_none());
        assert!(off_interval.is_none());
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.8,
                agreement: 0.6,
            })
        );
        assert!(!artifacts.best_model_base.with_extension("mpk").exists());
    }

    #[test]
    fn progress_message_refreshes_only_on_display_boundaries() {
        let mut config = dummy_config();
        config.log_every_n_steps = 10;
        config.validate_every_n_steps = 4;
        config.checkpoint_every_n_steps = 6;
        config.max_train_steps = Some(11);

        assert!(!exec_epoch::should_refresh_train_progress_message(
            &exec_epoch::EpochCadenceInput::from(&config),
            100,
            100
        ));
        assert!(exec_epoch::should_refresh_train_progress_message(
            &exec_epoch::EpochCadenceInput::from(&config),
            101,
            100
        ));
        assert!(!exec_epoch::should_refresh_train_progress_message(
            &exec_epoch::EpochCadenceInput::from(&config),
            103,
            100
        ));
        assert!(exec_epoch::should_refresh_train_progress_message(
            &exec_epoch::EpochCadenceInput::from(&config),
            104,
            100
        ));
        assert!(exec_epoch::should_refresh_train_progress_message(
            &exec_epoch::EpochCadenceInput::from(&config),
            106,
            100
        ));
        assert!(exec_epoch::should_refresh_train_progress_message(
            &exec_epoch::EpochCadenceInput::from(&config),
            110,
            100
        ));
        assert!(exec_epoch::should_refresh_train_progress_message(
            &exec_epoch::EpochCadenceInput::from(&config),
            111,
            100
        ));
    }

    #[test]
    fn checkpoint_boundary_helper_matches_session_relative_cadence() {
        let mut config = dummy_config();
        config.checkpoint_every_n_steps = 5;

        assert!(!exec_epoch::should_save_periodic_checkpoint(
            &exec_epoch::EpochCadenceInput::from(&config),
            100,
            100
        ));
        assert!(!exec_epoch::should_save_periodic_checkpoint(
            &exec_epoch::EpochCadenceInput::from(&config),
            104,
            100
        ));
        assert!(exec_epoch::should_save_periodic_checkpoint(
            &exec_epoch::EpochCadenceInput::from(&config),
            105,
            100
        ));
        assert!(!exec_epoch::should_save_periodic_checkpoint(
            &exec_epoch::EpochCadenceInput::from(&config),
            109,
            100
        ));
        assert!(exec_epoch::should_save_periodic_checkpoint(
            &exec_epoch::EpochCadenceInput::from(&config),
            110,
            100
        ));
    }

    #[test]
    fn maybe_save_periodic_checkpoint_skips_when_session_step_is_not_on_boundary() {
        let config = dummy_config();
        let artifacts = test_artifacts("periodic_checkpoint_skip");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let optimizer = AdamConfig::new().init();

        maybe_save_periodic_checkpoint(
            &model,
            &optimizer,
            PeriodicCheckpointContext {
                config: &config,
                artifacts: &artifacts,
                epoch: 3,
                session_start_global_step: 10,
                current_runtime: dummy_runtime_resume_contract(),
            },
            PeriodicCheckpointState {
                global_step: 10,
                epoch_optimizer_steps: 4,
                total_loss: 1.25,
                best_validation: None,
            },
        )
        .expect("skip checkpoint at session start");

        maybe_save_periodic_checkpoint(
            &model,
            &optimizer,
            PeriodicCheckpointContext {
                config: &config,
                artifacts: &artifacts,
                epoch: 3,
                session_start_global_step: 10,
                current_runtime: dummy_runtime_resume_contract(),
            },
            PeriodicCheckpointState {
                global_step: 14,
                epoch_optimizer_steps: 4,
                total_loss: 1.25,
                best_validation: None,
            },
        )
        .expect("skip checkpoint off interval");

        assert!(!artifacts.latest_state_path.exists());
        assert!(!artifacts.latest_model_base.with_extension("mpk").exists());
        assert!(
            !artifacts
                .latest_optimizer_base
                .with_extension("bin")
                .exists()
        );
    }

    #[test]
    fn maybe_save_periodic_checkpoint_persists_resume_state_on_boundary() {
        let config = dummy_config();
        let artifacts = test_artifacts("periodic_checkpoint_save");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let optimizer = AdamConfig::new().init();
        let expected_best = Some(BestValidation {
            policy_loss: 0.7,
            agreement: 0.8,
        });

        maybe_save_periodic_checkpoint(
            &model,
            &optimizer,
            PeriodicCheckpointContext {
                config: &config,
                artifacts: &artifacts,
                epoch: 3,
                session_start_global_step: 10,
                current_runtime: dummy_runtime_resume_contract(),
            },
            PeriodicCheckpointState {
                global_step: 15,
                epoch_optimizer_steps: 7,
                total_loss: 1.25,
                best_validation: expected_best,
            },
        )
        .expect("save checkpoint on interval");

        let state = read_resume_state(&artifacts.latest_state_path).expect("read resume state");
        assert_eq!(state.next_epoch, 3);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 7);
        assert_eq!(state.global_step, 15);
        assert_eq!(state.best_validation, expected_best);
        assert_eq!(state.runtime, dummy_runtime_resume_contract());
        assert!(artifacts.latest_state_path.exists());
        assert!(artifacts.latest_model_base.with_extension("mpk").exists());
        assert!(
            artifacts
                .latest_model_base
                .with_extension("meta.json")
                .exists()
        );
        assert!(
            artifacts
                .latest_optimizer_base
                .with_extension("bin")
                .exists()
        );
    }

    #[test]
    fn maybe_save_periodic_checkpoint_preserves_absent_best_validation_on_boundary() {
        let config = dummy_config();
        let artifacts = test_artifacts("periodic_checkpoint_save_without_best");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let optimizer = AdamConfig::new().init();

        maybe_save_periodic_checkpoint(
            &model,
            &optimizer,
            PeriodicCheckpointContext {
                config: &config,
                artifacts: &artifacts,
                epoch: 1,
                session_start_global_step: 0,
                current_runtime: dummy_runtime_resume_contract(),
            },
            PeriodicCheckpointState {
                global_step: 5,
                epoch_optimizer_steps: 2,
                total_loss: 0.5,
                best_validation: None,
            },
        )
        .expect("save checkpoint without best validation");

        let state = read_resume_state(&artifacts.latest_state_path).expect("read resume state");
        assert_eq!(state.next_epoch, 1);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 2);
        assert_eq!(state.global_step, 5);
        assert_eq!(state.best_validation, None);
    }

    #[test]
    fn maybe_run_interval_validation_updates_best_and_saves_checkpoint_on_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_save");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.8,
            agreement: 0.6,
        });
        let multi = MultiProgress::new();

        let summary = maybe_run_interval_validation(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            14,
            1.5,
        )
        .expect("run interval validation on boundary");

        let summary = summary.expect("validation summary on boundary");
        assert_eq!(summary.samples, 0);
        assert_eq!(summary.total_loss, 0.0);
        assert_eq!(summary.policy_loss, 0.0);
        assert_eq!(summary.agreement, 0.0);
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.0,
                agreement: 0.0,
            })
        );
        assert!(artifacts.best_model_base.with_extension("mpk").exists());
        assert!(
            artifacts
                .best_model_base
                .with_extension("meta.json")
                .exists()
        );
    }

    #[test]
    fn maybe_run_interval_validation_keeps_existing_best_when_summary_is_not_better() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_keep_best");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: -0.1,
            agreement: 0.9,
        });
        let multi = MultiProgress::new();

        let summary = maybe_run_interval_validation(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            14,
            1.5,
        )
        .expect("run interval validation without best update");

        let summary = summary.expect("validation summary on boundary");
        assert_eq!(summary.samples, 0);
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: -0.1,
                agreement: 0.9,
            })
        );
        assert!(!artifacts.best_model_base.with_extension("mpk").exists());
    }

    #[test]
    fn emit_interval_step_summary_writes_skipped_validation_step_log() {
        let config = dummy_config();
        let artifacts = test_artifacts("step_summary_skipped_validation");
        let manifest = dummy_manifest(false);
        let multi = MultiProgress::new();
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let window_stats = ScalarAverages::default();

        emit_interval_step_summary(
            &multi,
            &mut tb,
            &mut step_log,
            IntervalStepSummaryContext {
                manifest: &manifest,
                config: &config,
                session_start_global_step: 0,
                global_step: 9,
                epoch: 1,
                lr: 1.0e-4,
                best_validation: None,
                val_summary: None,
                seen_samples: 32,
                assumed_games_seen: 0,
                epoch_optimizer_steps: 2,
                window_stats,
                step_rate: 12.5,
                profiling: None,
                advisories: Vec::new(),
            },
        )
        .expect("emit skipped validation interval summary");

        let entry = read_jsonl_entry(&artifacts.step_log_path);
        assert_eq!(entry["global_step"].as_u64(), Some(9));
        assert_eq!(entry["epoch"].as_u64(), Some(2));
        assert_close(entry["lr"].as_f64().expect("step log lr"), 1.0e-4);
        assert_eq!(entry["profiling"]["stage"].as_str(), Some("bc_interval"));
        assert!(entry["profiling"]["elapsed_seconds"].as_f64().is_some());
        assert!(entry["profiling"]["children"].is_array());
        assert_eq!(entry["val_total_loss"], serde_json::Value::Null);
        assert_eq!(entry["val_policy_loss"], serde_json::Value::Null);
        assert_eq!(entry["best_val_policy_loss"], serde_json::Value::Null);
    }

    #[test]
    fn emit_interval_step_summary_writes_validation_and_best_metrics() {
        let config = dummy_config();
        let artifacts = test_artifacts("step_summary_validation_metrics");
        let manifest = dummy_manifest(true);
        let multi = MultiProgress::new();
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut window_stats = ScalarAverages::default();
        window_stats.record_batch(batch_stats(4, 2.5, 0.4));
        let window_stats = window_stats.finalize();
        let val_summary = dummy_validation_summary(0.9, 0.65);

        emit_interval_step_summary(
            &multi,
            &mut tb,
            &mut step_log,
            IntervalStepSummaryContext {
                manifest: &manifest,
                config: &config,
                session_start_global_step: 5,
                global_step: 11,
                epoch: 2,
                lr: 2.5e-4,
                best_validation: Some(BestValidation {
                    policy_loss: 0.8,
                    agreement: 0.7,
                }),
                val_summary: Some(val_summary.clone()),
                seen_samples: 48,
                assumed_games_seen: 6,
                epoch_optimizer_steps: 3,
                window_stats,
                step_rate: 3.0,
                profiling: None,
                advisories: Vec::new(),
            },
        )
        .expect("emit interval summary with validation");

        let entry = read_jsonl_entry(&artifacts.step_log_path);
        assert_eq!(entry["global_step"].as_u64(), Some(11));
        assert_eq!(entry["epoch"].as_u64(), Some(3));
        assert_close(
            entry["train_total_loss"]
                .as_f64()
                .expect("train total loss"),
            2.5,
        );
        assert_close(
            entry["train_policy_agreement"]
                .as_f64()
                .expect("train policy agreement"),
            0.4,
        );
        assert_close(
            entry["val_total_loss"].as_f64().expect("val total loss"),
            val_summary.total_loss,
        );
        assert_close(
            entry["val_policy_loss"].as_f64().expect("val policy loss"),
            val_summary.policy_loss,
        );
        assert_close(
            entry["val_policy_agreement"]
                .as_f64()
                .expect("val policy agreement"),
            val_summary.agreement,
        );
        assert_close(
            entry["best_val_policy_loss"]
                .as_f64()
                .expect("best val policy loss"),
            0.8,
        );
        assert_close(
            entry["best_val_agreement"]
                .as_f64()
                .expect("best val agreement"),
            0.7,
        );
        assert_eq!(entry["profiling"]["stage"].as_str(), Some("bc_interval"));
        assert!(entry["profiling"]["elapsed_seconds"].as_f64().is_some());
        assert!(entry["profiling"]["children"].is_array());
    }

    #[test]
    fn run_epoch_end_validation_returns_none_when_epoch_is_not_a_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("epoch_end_validation_skip");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.6,
            agreement: 0.75,
        });

        let summary = run_epoch_end_validation(
            0,
            &model,
            EpochEndValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                cached_validation_samples: None,
            },
            None,
            &mut best_validation,
            1.2,
        )
        .expect("skip epoch-end validation");

        assert!(summary.is_none());
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.6,
                agreement: 0.75,
            })
        );
        assert!(!artifacts.best_model_base.with_extension("mpk").exists());
    }

    #[test]
    fn run_epoch_end_validation_updates_best_and_saves_checkpoint_on_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("epoch_end_validation_save");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.7,
            agreement: 0.8,
        });

        let summary = run_epoch_end_validation(
            1,
            &model,
            EpochEndValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                cached_validation_samples: None,
            },
            None,
            &mut best_validation,
            1.2,
        )
        .expect("run epoch-end validation on boundary");

        let summary = summary.expect("epoch-end validation summary");
        assert_eq!(summary.samples, 0);
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.0,
                agreement: 0.0,
            })
        );
        assert!(artifacts.best_model_base.with_extension("mpk").exists());
        assert!(
            artifacts
                .best_model_base
                .with_extension("meta.json")
                .exists()
        );
    }

    #[test]
    fn finalize_epoch_outputs_writes_training_log_with_validation_metrics() {
        let config = dummy_config();
        let artifacts = test_artifacts("finalize_epoch_outputs");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut train_stats = ScalarAverages::default();
        train_stats.record_batch(batch_stats(4, 3.5, 0.55));
        let train_stats = train_stats.finalize();
        let val_summary = dummy_validation_summary(0.95, 0.68);

        finalize_epoch_outputs::<Vec<u8>, _, _, hydra_train_exec::advisory::RuntimeAdvisory>(
            &mut tb,
            &mut training_log,
            EpochFinalizeContext::new(
                &config,
                &train_cfg,
                2,
                17,
                train_stats,
                Some(val_summary.clone()),
                Some(BestValidation {
                    policy_loss: 0.9,
                    agreement: 0.7,
                }),
                2.0e-4,
                None,
            ),
        )
        .expect("finalize epoch outputs");

        let entry = read_jsonl_entry(&artifacts.training_log_path);
        assert_eq!(entry["epoch"].as_u64(), Some(3));
        assert_eq!(entry["global_step"].as_u64(), Some(17));
        assert_eq!(entry["num_batches"].as_u64(), Some(1));
        assert_close(
            entry["train_total_loss"]
                .as_f64()
                .expect("train total loss"),
            3.5,
        );
        assert_close(
            entry["val_total_loss"].as_f64().expect("val total loss"),
            val_summary.total_loss,
        );
        assert_close(
            entry["val_policy_loss"].as_f64().expect("val policy loss"),
            val_summary.policy_loss,
        );
        assert_close(
            entry["best_val_policy_loss"]
                .as_f64()
                .expect("best val policy loss"),
            0.9,
        );
        assert_close(
            entry["best_val_agreement"]
                .as_f64()
                .expect("best val agreement"),
            0.7,
        );
        assert_eq!(entry["profiling"]["stage"].as_str(), Some("bc_epoch"));
        assert!(entry["profiling"]["elapsed_seconds"].as_f64().is_some());
        assert!(entry["profiling"]["children"].is_array());
    }

    #[test]
    fn finalize_epoch_outputs_preserves_train_sub_stage_children_in_json() {
        let config = dummy_config();
        let artifacts = test_artifacts("finalize_epoch_outputs_sub_stages");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut train_stats = ScalarAverages::default();
        train_stats.record_batch(batch_stats(4, 3.5, 0.55));
        let train_stats = train_stats.finalize();

        let sub_timing = TrainSubStageTiming {
            producer_wait_seconds: 0.04,
            collation_seconds: 0.01,
            h2d_transfer_seconds: 0.06,
            h2d_pageable_to_pinned_seconds: 0.01,
            h2d_tensor_materialize_seconds: 0.04,
            h2d_stream_sync_seconds: 0.01,
            forward_seconds: 0.5,
            loss_seconds: 0.02,
            backward_seconds: 0.3,
            metric_readback_seconds: 0.02,
            optimizer_step_seconds: 0.05,
        };
        let profiling = bc_epoch_profiling(0.88, &sub_timing, None, 0.1, 0.0);

        finalize_epoch_outputs::<Vec<u8>, _, _, hydra_train_exec::advisory::RuntimeAdvisory>(
            &mut tb,
            &mut training_log,
            EpochFinalizeContext::new(
                &config,
                &train_cfg,
                0,
                5,
                train_stats,
                None,
                None,
                1.0e-4,
                Some(profiling),
            ),
        )
        .expect("finalize epoch outputs with sub-stage profiling");

        let entry = read_jsonl_entry(&artifacts.training_log_path);
        let profiling = &entry["profiling"];
        assert_eq!(profiling["stage"].as_str(), Some("bc_epoch"));

        let children = profiling["children"]
            .as_array()
            .expect("profiling should have children array");
        let train_child = children
            .iter()
            .find(|c| c["stage"].as_str() == Some("train"))
            .expect("should have a 'train' child");
        let train_sub_children = train_child["children"]
            .as_array()
            .expect("train child should have sub-stage children");

        let expected_sub_stages = ["collation", "forward", "loss", "backward", "optimizer_step"];
        for stage_name in &expected_sub_stages {
            let found = train_sub_children
                .iter()
                .find(|c| c["stage"].as_str() == Some(stage_name));
            assert!(
                found.is_some(),
                "train sub-stage '{}' should be present in JSON",
                stage_name
            );
            let elapsed = found.unwrap()["elapsed_seconds"].as_f64();
            assert!(
                elapsed.is_some() && elapsed.unwrap() > 0.0,
                "train sub-stage '{}' should have positive elapsed_seconds",
                stage_name
            );
        }

        let h2d_child = train_sub_children
            .iter()
            .find(|c| c["stage"].as_str() == Some("h2d_transfer"))
            .expect("h2d transfer stage should be present in JSON");
        let h2d_sub_children = h2d_child["children"]
            .as_array()
            .expect("h2d child should have materialization sub-stage children");
        for stage_name in &[
            "h2d_pageable_to_pinned",
            "h2d_tensor_materialize",
            "h2d_stream_sync",
        ] {
            let elapsed = h2d_sub_children
                .iter()
                .find(|c| c["stage"].as_str() == Some(stage_name))
                .and_then(|c| c["elapsed_seconds"].as_f64())
                .expect("h2d sub-stage should carry elapsed seconds");
            assert!(
                elapsed > 0.0,
                "h2d sub-stage '{stage_name}' should be positive"
            );
        }
    }

    #[test]
    fn bc_interval_profiling_records_checkpoint_separately_from_logging() {
        let sub_timing = TrainSubStageTiming::default();

        let profiling = bc_interval_profiling(1.0, &sub_timing, None, 0.25);

        assert_eq!(
            child_elapsed_seconds(&profiling, PROFILING_STAGE_CHECKPOINT),
            0.25
        );
        assert_eq!(
            exec_epoch::child_elapsed_seconds(&profiling, PROFILING_STAGE_LOGGING),
            0.0
        );
        let input = interval_timing_input(&dummy_config(), &profiling, 4);
        assert_eq!(input.checkpoint_seconds, 0.25);
        assert_eq!(input.logging_seconds, 0.0);
    }

    #[test]
    fn finalize_epoch_outputs_writes_skipped_validation_epoch_log() {
        let config = dummy_config();
        let artifacts = test_artifacts("finalize_epoch_outputs_skipped_validation");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");

        finalize_epoch_outputs::<Vec<u8>, _, _, hydra_train_exec::advisory::RuntimeAdvisory>(
            &mut tb,
            &mut training_log,
            EpochFinalizeContext::new(
                &config,
                &train_cfg,
                0,
                3,
                ScalarAverages::default(),
                None,
                None,
                5.0e-5,
                None,
            ),
        )
        .expect("finalize epoch outputs without validation");

        let entry = read_jsonl_entry(&artifacts.training_log_path);
        assert_eq!(entry["epoch"].as_u64(), Some(1));
        assert_eq!(entry["global_step"].as_u64(), Some(3));
        assert_eq!(entry["num_batches"].as_u64(), Some(0));
        assert_close(entry["lr"].as_f64().expect("epoch log lr"), 5.0e-5);
        assert_eq!(entry["val_total_loss"], serde_json::Value::Null);
        assert_eq!(entry["val_policy_loss"], serde_json::Value::Null);
        assert_eq!(entry["best_val_policy_loss"], serde_json::Value::Null);
        assert_eq!(entry["best_val_agreement"], serde_json::Value::Null);
        assert_eq!(entry["profiling"]["stage"].as_str(), Some("bc_epoch"));
        assert!(entry["profiling"]["elapsed_seconds"].as_f64().is_some());
        assert!(entry["profiling"]["children"].is_array());
    }

    #[test]
    fn run_epoch_empty_manifest_finalizes_and_advances_epoch() {
        let mut config = dummy_config();
        config.num_epochs = 3;
        config.validation_every_n_epochs = 2;
        config.max_train_steps = Some(20);
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("run_epoch_empty_manifest_complete");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let train_loss_fn = dummy_train_loss();
        let valid_loss_fn = dummy_valid_loss();
        let device = LibTorchDevice::Cpu;
        let mut model = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut global_step = 7usize;
        let mut best_validation = None;
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut last_log_step = 0usize;
        let mut last_log_time = Instant::now();
        let run_start = Instant::now();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));

        let outcome = run_epoch(
            EpochRunnerContext {
                epoch: 1,
                config: &config,
                manifest: &manifest,
                loader_config: &loader_config,
                artifacts: &artifacts,
                train_cfg: &train_cfg,
                loss_fn: &train_loss_fn,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                train_device: &device,
                session_start_global_step: 0,
                steps_to_skip: 3,
                microbatch_size: 4,
                use_amp: false,
                total_steps: 100,
                current_runtime: dummy_runtime_resume_contract(),
                run_start: &run_start,
                head_controller: &mut head_controller,
                cached_validation_samples: None,
            },
            EpochRuntimeMut {
                model: &mut model,
                optimizer: &mut optimizer,
                global_step: &mut global_step,
                best_validation: &mut best_validation,
                tb: &mut tb,
                training_log: &mut training_log,
                step_log: &mut step_log,
                last_log_step: &mut last_log_step,
                last_log_time: &mut last_log_time,
            },
        )
        .expect("run epoch with empty manifest");

        assert!(!outcome.stop_after_epoch);
        assert_eq!(global_step, 7);
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.0,
                agreement: 0.0,
            })
        );

        let state = read_resume_state(&artifacts.latest_state_path).expect("read latest state");
        assert_eq!(state.next_epoch, 2);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 0);
        assert_eq!(state.global_step, 7);
        assert_eq!(state.best_validation, None);
        assert_eq!(state.runtime, dummy_runtime_resume_contract());
        assert!(artifacts.best_model_base.with_extension("mpk").exists());
        assert!(artifacts.training_log_path.exists());

        let entry = read_jsonl_entry(&artifacts.training_log_path);
        assert_eq!(entry["epoch"].as_u64(), Some(2));
        assert_eq!(entry["global_step"].as_u64(), Some(7));
        assert_close(
            entry["val_total_loss"]
                .as_f64()
                .expect("epoch val total loss"),
            0.0,
        );
        assert_close(
            entry["best_val_policy_loss"]
                .as_f64()
                .expect("epoch best validation loss"),
            0.0,
        );
    }

    #[test]
    fn run_epoch_empty_manifest_stops_when_session_budget_is_already_exhausted() {
        let mut config = dummy_config();
        config.num_epochs = 4;
        config.validation_every_n_epochs = 3;
        config.max_train_steps = Some(0);
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(false);
        let artifacts = test_artifacts("run_epoch_empty_manifest_budget_stop");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let train_loss_fn = dummy_train_loss();
        let valid_loss_fn = dummy_valid_loss();
        let device = LibTorchDevice::Cpu;
        let mut model = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut global_step = 12usize;
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.4,
            agreement: 0.5,
        });
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut last_log_step = 11usize;
        let mut last_log_time = Instant::now();
        let run_start = Instant::now();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));

        let outcome = run_epoch(
            EpochRunnerContext {
                epoch: 0,
                config: &config,
                manifest: &manifest,
                loader_config: &loader_config,
                artifacts: &artifacts,
                train_cfg: &train_cfg,
                loss_fn: &train_loss_fn,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                train_device: &device,
                session_start_global_step: 12,
                steps_to_skip: 0,
                microbatch_size: 4,
                use_amp: false,
                total_steps: 100,
                current_runtime: dummy_runtime_resume_contract(),
                run_start: &run_start,
                head_controller: &mut head_controller,
                cached_validation_samples: None,
            },
            EpochRuntimeMut {
                model: &mut model,
                optimizer: &mut optimizer,
                global_step: &mut global_step,
                best_validation: &mut best_validation,
                tb: &mut tb,
                training_log: &mut training_log,
                step_log: &mut step_log,
                last_log_step: &mut last_log_step,
                last_log_time: &mut last_log_time,
            },
        )
        .expect("run epoch with exhausted session budget");

        assert!(outcome.stop_after_epoch);
        assert_eq!(global_step, 12);
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.4,
                agreement: 0.5,
            })
        );

        let state = read_resume_state(&artifacts.latest_state_path).expect("read latest state");
        assert_eq!(state.next_epoch, 1);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 0);
        assert_eq!(state.global_step, 12);
        assert_eq!(state.best_validation, best_validation);

        let entry = read_jsonl_entry(&artifacts.training_log_path);
        assert_eq!(entry["epoch"].as_u64(), Some(1));
        assert_eq!(entry["global_step"].as_u64(), Some(12));
        assert_eq!(entry["val_total_loss"], serde_json::Value::Null);
        assert_eq!(entry["best_val_policy_loss"].as_f64(), Some(0.4));
        assert!(!artifacts.best_model_base.with_extension("mpk").exists());
    }

    #[test]
    fn run_epoch_empty_manifest_completes_with_latest_state_and_best_checkpoint() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(false);
        let artifacts = test_artifacts("run_epoch_empty_manifest_checkpoint_contract");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let train_loss_fn = dummy_train_loss();
        let valid_loss_fn = dummy_valid_loss();
        let device = LibTorchDevice::Cpu;
        let mut model = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut global_step = 7usize;
        let mut best_validation = None;
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut last_log_step = 0usize;
        let mut last_log_time = Instant::now();
        let run_start = Instant::now();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));

        run_epoch(
            EpochRunnerContext {
                epoch: 1,
                config: &config,
                manifest: &manifest,
                loader_config: &loader_config,
                artifacts: &artifacts,
                train_cfg: &train_cfg,
                loss_fn: &train_loss_fn,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                train_device: &device,
                session_start_global_step: 0,
                steps_to_skip: 3,
                microbatch_size: 4,
                use_amp: false,
                total_steps: 100,
                current_runtime: dummy_runtime_resume_contract(),
                run_start: &run_start,
                head_controller: &mut head_controller,
                cached_validation_samples: None,
            },
            EpochRuntimeMut {
                model: &mut model,
                optimizer: &mut optimizer,
                global_step: &mut global_step,
                best_validation: &mut best_validation,
                tb: &mut tb,
                training_log: &mut training_log,
                step_log: &mut step_log,
                last_log_step: &mut last_log_step,
                last_log_time: &mut last_log_time,
            },
        )
        .expect("run epoch with empty manifest");

        let state = read_resume_state(&artifacts.latest_state_path).expect("read latest state");
        assert_eq!(state.next_epoch, 2);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 0);
        assert_eq!(state.global_step, 7);
        assert_eq!(state.best_validation, None);
        assert!(artifacts.latest_model_base.with_extension("mpk").exists());
        assert!(
            artifacts
                .latest_model_base
                .with_extension("meta.json")
                .exists()
        );
        assert!(
            artifacts
                .latest_optimizer_base
                .with_extension("bin")
                .exists()
        );
        assert!(artifacts.best_model_base.with_extension("mpk").exists());
        assert!(
            artifacts
                .best_model_base
                .with_extension("meta.json")
                .exists()
        );
    }

    #[test]
    fn latest_checkpoint_can_refresh_state_without_rewriting_payload_files() {
        let artifacts = test_artifacts("latest_checkpoint_refreshes_state_only");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let optimizer = AdamConfig::new().init();

        save_latest_checkpoint_and_state(
            &artifacts,
            &model,
            &optimizer,
            LatestCheckpointState {
                global_step: 15,
                train_loss: 1.25,
                best_validation: Some(BestValidation {
                    policy_loss: 0.7,
                    agreement: 0.8,
                }),
                continuation: &EpochContinuation {
                    next_epoch: 3,
                    skip_optimizer_steps_in_epoch: 7,
                    epoch_completed: false,
                },
                runtime: dummy_runtime_resume_contract(),
            },
        )
        .expect("initial latest checkpoint save");

        let latest_model_path = artifacts.latest_model_base.with_extension("mpk");
        let latest_meta_path = artifacts.latest_model_base.with_extension("meta.json");
        let latest_optimizer_path = artifacts.latest_optimizer_base.with_extension("bin");
        let model_before = modified_time(&latest_model_path);
        let meta_before = modified_time(&latest_meta_path);
        let optimizer_before = modified_time(&latest_optimizer_path);
        let state_before = modified_time(&artifacts.latest_state_path);
        thread::sleep(std::time::Duration::from_millis(1100));

        save_latest_checkpoint_and_state(
            &artifacts,
            &model,
            &optimizer,
            LatestCheckpointState {
                global_step: 15,
                train_loss: 1.25,
                best_validation: Some(BestValidation {
                    policy_loss: 0.7,
                    agreement: 0.8,
                }),
                continuation: &EpochContinuation {
                    next_epoch: 4,
                    skip_optimizer_steps_in_epoch: 0,
                    epoch_completed: true,
                },
                runtime: dummy_runtime_resume_contract(),
            },
        )
        .expect("refresh latest checkpoint state without rewriting payloads");

        let state = read_resume_state(&artifacts.latest_state_path).expect("read latest state");
        assert_eq!(state.next_epoch, 4);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 0);
        assert_eq!(state.global_step, 15);
        assert_eq!(modified_time(&latest_model_path), model_before);
        assert_eq!(modified_time(&latest_meta_path), meta_before);
        assert_eq!(modified_time(&latest_optimizer_path), optimizer_before);
        assert!(modified_time(&artifacts.latest_state_path) > state_before);
    }

    #[test]
    fn run_epoch_empty_manifest_records_bc_epoch_scope_order() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(false);
        let artifacts = test_artifacts("nvtx_bc_epoch_scope");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let train_loss_fn = dummy_train_loss();
        let valid_loss_fn = dummy_valid_loss();
        let device = LibTorchDevice::Cpu;
        let mut model = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut global_step = 4usize;
        let mut best_validation = None;
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut last_log_step = 0usize;
        let mut last_log_time = Instant::now();
        let run_start = Instant::now();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));

        let (_, events) = crate::nvtx::with_test_recorder(|| {
            run_epoch(
                EpochRunnerContext {
                    epoch: 1,
                    config: &config,
                    manifest: &manifest,
                    loader_config: &loader_config,
                    artifacts: &artifacts,
                    train_cfg: &train_cfg,
                    loss_fn: &train_loss_fn,
                    valid_loss_fn: &valid_loss_fn,
                    bc_exit_cfg: &BcExitConfig::default(),
                    train_device: &device,
                    session_start_global_step: 0,
                    steps_to_skip: 0,
                    microbatch_size: 4,
                    use_amp: false,
                    total_steps: 100,
                    current_runtime: dummy_runtime_resume_contract(),
                    run_start: &run_start,
                    head_controller: &mut head_controller,
                    cached_validation_samples: None,
                },
                EpochRuntimeMut {
                    model: &mut model,
                    optimizer: &mut optimizer,
                    global_step: &mut global_step,
                    best_validation: &mut best_validation,
                    tb: &mut tb,
                    training_log: &mut training_log,
                    step_log: &mut step_log,
                    last_log_step: &mut last_log_step,
                    last_log_time: &mut last_log_time,
                },
            )
            .expect("run epoch with empty manifest should succeed");
        });

        assert_eq!(
            events,
            vec![
                "push:bc_epoch",
                "push:checkpoint",
                "pop:checkpoint",
                "push:validation",
                "pop:validation",
                "push:checkpoint",
                "pop:checkpoint",
                "push:logging",
                "pop:logging",
                "pop:bc_epoch",
            ]
        );
    }
}
