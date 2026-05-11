#[cfg(test)]
use std::path::Path;

#[cfg(test)]
use hydra_train_exec::artifacts::append_step_log_to_writer;
#[cfg(test)]
pub(crate) use hydra_train_exec::artifacts::log_tensorboard;
pub(crate) use hydra_train_exec::artifacts::{BcArtifactPaths, append_advisory_event_to_writer};
#[cfg(test)]
pub(crate) use hydra_train_exec::artifacts::{
    PersistedDeltaQPromotionArtifact, write_delta_q_promotion_artifact,
};
#[cfg(test)]
pub(crate) use hydra_train_exec::artifacts::{
    PreflightBenchmarkPaths, PreflightBenchmarkReport, PreflightPaths, read_preflight_cache,
    write_preflight_benchmark_report, write_preflight_cache,
};
#[cfg(test)]
pub(crate) use hydra_train_exec::artifacts::{
    RlArtifactPaths, RlPreflightPaths, append_rl_step_log_to_writer, open_rl_step_log_appender,
    open_training_log_appender,
};
#[cfg(test)]
pub(crate) use hydra_train_exec::artifacts::{
    append_training_log_to_writer, manifest_cache_matches, open_step_log_appender,
    read_manifest_cache, scan_and_write_manifest_cache, write_manifest_cache,
};

#[cfg(test)]
pub(crate) fn append_step_log(
    path: &Path,
    entry: &super::progress::StepLogEntry,
) -> Result<(), String> {
    let mut file = open_step_log_appender(path)?;
    append_step_log_to_writer(&mut file, entry)
}

#[cfg(test)]
pub(crate) fn append_training_log(
    path: &Path,
    entry: &super::progress::EpochLogEntry,
) -> Result<(), String> {
    let mut file = open_training_log_appender(path)?;
    append_training_log_to_writer(&mut file, entry)
}

#[cfg(test)]
pub(crate) fn append_rl_step_log(
    path: &Path,
    entry: &super::progress::RlStepLogEntry,
) -> Result<(), String> {
    let mut file = open_rl_step_log_appender(path)?;
    append_rl_step_log_to_writer(&mut file, entry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RlPhaseConfig;
    use crate::resume::{RlResumeSemantics, RlResumeState, RlRuntimeResumeContract};
    use crate::validation::DeltaQPromotionSnapshot;
    use hydra_train::config::{PipelineState, TrainingPhase};
    use hydra_train::data::pipeline::{DataManifest, DataSource};
    use hydra_train::eval::ArenaPromotionDecision;
    use hydra_train::preflight::{
        BenchmarkMetadata, BenchmarkMode, BenchmarkResult, BenchmarkRuntimeConfig, BenchmarkScore,
        EffectiveRuntimeConfig, HardwareFingerprint, LoaderRuntimeConfig, ManifestCacheEntry,
        PreflightCacheEntry, PreflightCacheKey, ProfilingEnvelope, SelectedRuntimeConfig,
        WorkloadFingerprint, default_cache_name, default_manifest_cache_name,
    };
    use hydra_train::training::delta_q_promotion::{
        DeltaQArenaConfirmationRequest, DeltaQPolicyTransferReport, DeltaQPolicyTransferResult,
        DeltaQPromotionRecommendation, DeltaQPromotionReport, DeltaQPromotionResult,
        delta_q_arena_report_from_paired_eval,
    };

    use crate::progress::{EpochLogEntry, RlStepLogEntry, ScalarAverages, StepLogEntry};
    use crate::resume::{BestValidation, write_rl_resume_state};
    use crate::validation::ValidationSummary;
    use hydra_train_exec::advisory::{AdvisoryEvent, RuntimeAdvisory};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tboard::{EventWriter, SummaryReader};

    fn temp_dir_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("hydra_{label}_{unique}"))
    }

    fn sample_preflight_cache_entry() -> PreflightCacheEntry {
        PreflightCacheEntry {
            cache_key: PreflightCacheKey {
                hardware: HardwareFingerprint {
                    device_label: "test-gpu".to_string(),
                    backend: "wgpu".to_string(),
                    cpu_logical_cores: 16,
                    total_memory_bytes: Some(64 * 1024),
                },
                workload: WorkloadFingerprint {
                    batch_size: 128,
                    augment: true,
                    precision_mode: "fp32".to_string(),
                    train_fraction_bits: 1234,
                    max_skip_logs_per_source: 5,
                    max_validation_batches: Some(7),
                    max_validation_samples: Some(256),
                    model_signature: "model-sig".to_string(),
                    code_signature: "code-sig".to_string(),
                    advanced_loss_signature: "loss-sig".to_string(),
                    preflight_config_signature: "preflight-sig".to_string(),
                    explicit_train_microbatch: Some(32),
                    explicit_validation_microbatch: Some(64),
                },
            },
            runtime: EffectiveRuntimeConfig {
                selected: SelectedRuntimeConfig {
                    train_microbatch_size: 32,
                    validation_microbatch_size: 64,
                    accum_steps: 4,
                },
                loader: LoaderRuntimeConfig {
                    num_threads: Some(8),
                    buffer_games: 512,
                    buffer_samples: 2048,
                    archive_queue_bound: 32,
                },
            },
            benchmark: None,
        }
    }

    fn sample_preflight_cache_entry_with_benchmark() -> PreflightCacheEntry {
        let mut entry = sample_preflight_cache_entry();
        entry.benchmark = Some(BenchmarkResult {
            runtime: BenchmarkRuntimeConfig {
                train_microbatch_size: 32,
                validation_microbatch_size: 64,
                accum_steps: 4,
                loader: LoaderRuntimeConfig {
                    num_threads: Some(8),
                    buffer_games: 512,
                    buffer_samples: 2048,
                    archive_queue_bound: 32,
                },
            },
            score: BenchmarkScore {
                wall_clock_samples_per_second: 111.0,
                train_only_samples_per_second: 140.0,
                train_seconds: 10.0,
                validation_seconds: 2.0,
                checkpoint_seconds: 0.5,
                logging_seconds: 0.25,
                total_elapsed_seconds: 12.75,
                train_steps: 64,
                validation_samples: 1024,
            },
            metadata: BenchmarkMetadata {
                mode: BenchmarkMode::CadenceAwareProjection,
                selection_metric: "wall_clock_effective_throughput".to_string(),
                train_probe_candidates_considered: 4,
                validation_probe_candidates_considered: 4,
                loader_candidates_considered: 3,
                finalists_benchmarked: 2,
                warmup_steps: 8,
                measured_train_steps: 64,
                projected_validation_events: 6.0,
                projected_checkpoint_events: 6.0,
                projected_logging_events: 6.0,
            },
            profiling: Some(ProfilingEnvelope::nested(
                "stage_2_benchmark",
                12.75,
                vec![
                    ProfilingEnvelope::leaf("train", 10.0),
                    ProfilingEnvelope::nested(
                        "validation",
                        2.0,
                        vec![
                            ProfilingEnvelope::leaf("candidate_forward_and_loss", 1.5),
                            ProfilingEnvelope::leaf("delta_q_baseline_forward", 0.5),
                        ],
                    ),
                    ProfilingEnvelope::leaf("checkpoint", 0.5),
                    ProfilingEnvelope::leaf("logging", 0.25),
                ],
            )),
        });
        entry
    }

    fn sample_manifest_cache_entry() -> ManifestCacheEntry {
        ManifestCacheEntry {
            data_dir: PathBuf::from("/data"),
            train_fraction_bits: 0.9f32.to_bits(),
            include_source_patterns: Vec::new(),
            exclude_source_patterns: Vec::new(),
            manifest: DataManifest {
                sources: vec![
                    DataSource::LooseFile(PathBuf::from("/data/a.mjai.json")),
                    DataSource::Archive(PathBuf::from("/data/b.tar.zst")),
                ],
                total_games: 1,
                train_count: 1,
                val_count: 0,
                counts_exact: false,
            },
        }
    }

    use hydra_train_runtime::config::SourceFilterConfig;
    #[test]
    fn manifest_cache_matches_checks_data_dir_fraction_and_filters() {
        let entry = sample_manifest_cache_entry();
        let mut filters = SourceFilterConfig::default();

        assert!(manifest_cache_matches(
            &entry,
            Path::new("/data"),
            0.9,
            &filters,
        ));

        assert!(!manifest_cache_matches(
            &entry,
            Path::new("/other"),
            0.9,
            &filters,
        ));
        assert!(!manifest_cache_matches(
            &entry,
            Path::new("/data"),
            0.8,
            &filters,
        ));

        filters.include_source_patterns.push("jade".to_string());
        assert!(!manifest_cache_matches(
            &entry,
            Path::new("/data"),
            0.9,
            &filters,
        ));

        let mut exclude_filters = SourceFilterConfig::default();
        exclude_filters
            .exclude_source_patterns
            .push("tenhou".to_string());
        assert!(!manifest_cache_matches(
            &entry,
            Path::new("/data"),
            0.9,
            &exclude_filters,
        ));
    }

    #[test]
    fn scan_and_write_manifest_cache_persists_scanned_manifest() {
        let output_dir = temp_dir_path("scan_and_write_manifest_cache");
        let data_dir = output_dir.join("data");
        fs::create_dir_all(&data_dir).expect("create temp data dir");
        let replay_path = data_dir.join("game.mjai.json");
        fs::write(
            &replay_path,
            "{\"type\":\"start_game\",\"seed\":0}\n{\"type\":\"end_game\"}\n",
        )
        .expect("write replay fixture");
        let cache_path = output_dir.join("preflight_manifest.json");
        let filters = SourceFilterConfig::default();

        let manifest =
            scan_and_write_manifest_cache(&cache_path, &data_dir, 1.0, &filters, None, "test data")
                .expect("scan and write manifest cache");

        assert_eq!(manifest.total_games, 1);
        assert_eq!(manifest.train_count, 1);
        assert_eq!(manifest.val_count, 0);

        let restored = read_manifest_cache(&cache_path)
            .expect("read written manifest cache")
            .expect("manifest cache entry present");
        assert_eq!(restored.data_dir, data_dir);
        assert_eq!(restored.train_fraction_bits, 1.0f32.to_bits());
        assert_eq!(restored.manifest, manifest);

        cleanup_dir(&output_dir);
    }

    fn sample_epoch_log_entry() -> EpochLogEntry {
        EpochLogEntry {
            epoch: 3,
            global_step: 17,
            lr: 0.01,
            train_total_loss: 1.5,
            train_policy_agreement: 0.25,
            train_loss_policy: 0.5,
            train_loss_value: 0.1,
            train_loss_grp: 0.2,
            train_loss_tenpai: 0.3,
            train_loss_danger: 0.4,
            train_loss_opp_next: 0.6,
            train_loss_score_pdf: 0.7,
            train_loss_score_cdf: 0.8,
            train_rare_actions: crate::progress::RareActionMetrics::default(),
            val_total_loss: Some(1.2),
            val_policy_loss: Some(0.9),
            val_policy_agreement: Some(0.75),
            val_delta_q_promotion: None,
            val_rare_actions: None,
            profiling: Some(ProfilingEnvelope::leaf("bc_epoch", 1.2)),
            advisories: Vec::new(),
            best_val_policy_loss: Some(0.8),
            best_val_agreement: Some(0.77),
            num_batches: 4,
        }
    }

    fn sample_step_log_entry() -> StepLogEntry {
        StepLogEntry {
            global_step: 17,
            epoch: 3,
            lr: 0.01,
            train_total_loss: 1.5,
            train_policy_agreement: 0.25,
            train_loss_policy: 0.5,
            train_loss_value: 0.1,
            train_loss_grp: 0.2,
            train_loss_tenpai: 0.3,
            train_loss_danger: 0.4,
            train_loss_opp_next: 0.6,
            train_loss_score_pdf: 0.7,
            train_loss_score_cdf: 0.8,
            train_rare_actions: crate::progress::RareActionMetrics::default(),
            val_total_loss: Some(1.2),
            val_policy_loss: Some(0.9),
            val_policy_agreement: Some(0.75),
            val_delta_q_promotion: None,
            val_rare_actions: None,
            profiling: Some(ProfilingEnvelope::leaf("bc_interval", 0.6)),
            advisories: Vec::new(),
            best_val_policy_loss: Some(0.8),
            best_val_agreement: Some(0.77),
        }
    }

    fn sample_rl_step_log_entry() -> RlStepLogEntry {
        RlStepLogEntry {
            global_step: 12,
            phase: "exit_pondering".to_string(),
            loss: 0.55,
            effective_lr: 0.005,
            exit_weight: 0.25,
            games_per_batch: 8,
            samples_in_batch: 64,
            total_games: 1024,
            total_samples: 8192,
            delta_q_state: "Active".to_string(),
            profiling: Some(ProfilingEnvelope::leaf("rl_step", 0.4)),
            advisories: Vec::new(),
        }
    }

    fn sample_validation_summary() -> ValidationSummary {
        let promotion_report = DeltaQPromotionReport::new();
        let promotion_result = DeltaQPromotionResult {
            passed: true,
            criteria: Vec::new(),
        };
        ValidationSummary {
            total_loss: 1.2,
            policy_loss: 0.8,
            agreement: 0.7,
            samples: 64,
            rare_actions: crate::progress::RareActionMetrics::default(),
            saw_exit_targets: false,
            saw_delta_q_targets: true,
            profiling: Some(ProfilingEnvelope::leaf("validation", 0.9)),
            delta_q_promotion: Some(promotion_report.clone()),
            delta_q_promotion_result: Some(promotion_result.clone()),
            delta_q_promotion_snapshot: Some(DeltaQPromotionSnapshot {
                compared_states: promotion_report.compared_states,
                candidate_top1_agreement: promotion_report.candidate_top1_agreement(),
                candidate_mean_regret: promotion_report.candidate_mean_regret(),
                baseline_mean_regret: promotion_report.baseline_mean_regret(),
                mean_decision_lift: promotion_report.mean_decision_lift(),
                negative_lift_fraction: promotion_report.negative_lift_fraction(),
                regret_beats_baseline_rate: promotion_report.candidate_regret_beats_baseline_rate(),
                top1_beats_baseline_rate: promotion_report.candidate_top1_beats_baseline_rate(),
                passed: promotion_result.passed,
            }),
            delta_q_policy_transfer: Some(DeltaQPolicyTransferReport::new()),
            delta_q_policy_transfer_result: Some(DeltaQPolicyTransferResult {
                passed: true,
                criteria: Vec::new(),
            }),
            delta_q_policy_transfer_snapshot: None,
        }
    }

    fn cleanup_dir(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }

    fn tensorboard_tags_from_dir(path: &Path) -> Vec<String> {
        let event_path = fs::read_dir(path)
            .expect("read tensorboard dir")
            .map(|entry| entry.expect("tensorboard dir entry").path())
            .find(|entry| entry.is_file())
            .expect("tensorboard event file");
        let file = fs::File::open(event_path).expect("open tensorboard event file");
        let mut tags = Vec::new();
        for event in SummaryReader::new(file).skip(1) {
            let event = event.expect("decode tensorboard event");
            let summary = match event.what.expect("event payload") {
                tboard::tensorboard::event::What::Summary(summary) => summary,
                other => panic!("expected summary event, got {other:?}"),
            };
            for value in summary.value {
                tags.push(value.tag);
            }
        }
        tags
    }

    #[test]
    fn bc_artifact_paths_build_expected_names() {
        let output_dir = temp_dir_path("bc_artifact_paths");
        let artifacts = BcArtifactPaths::new(&output_dir, 42);

        assert_eq!(artifacts.root, output_dir.join("bc"));
        assert_eq!(artifacts.tb_root, artifacts.root.join("tb"));
        assert_eq!(
            artifacts.latest_model_base,
            artifacts.root.join("latest_model")
        );
        assert_eq!(
            artifacts.latest_optimizer_base,
            artifacts.root.join("latest_optimizer")
        );
        assert_eq!(artifacts.best_model_base, artifacts.root.join("best_model"));
        assert_eq!(
            artifacts.latest_state_path,
            artifacts.root.join("latest_state.yaml")
        );
        assert_eq!(
            artifacts.training_log_path,
            artifacts.root.join("training_log.jsonl")
        );
        assert_eq!(
            artifacts.step_log_path,
            artifacts.root.join("step_log.jsonl")
        );
        assert_eq!(
            artifacts.delta_q_promotion_path,
            artifacts.root.join("delta_q_promotion.json")
        );
        let tb_session = artifacts
            .tb_session_dir
            .file_name()
            .expect("tensorboard session dir name")
            .to_string_lossy();
        assert!(tb_session.starts_with("run_g00000042_"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn rl_artifact_paths_build_expected_names() {
        let output_dir = temp_dir_path("rl_artifact_paths");
        let artifacts = RlArtifactPaths::new(&output_dir, 7);

        assert_eq!(artifacts.root, output_dir.join("rl"));
        assert_eq!(artifacts.tb_root, artifacts.root.join("tb"));
        assert_eq!(
            artifacts.latest_model_base,
            artifacts.root.join("latest_model")
        );
        assert_eq!(
            artifacts.latest_optimizer_base,
            artifacts.root.join("latest_optimizer")
        );
        assert_eq!(
            artifacts.latest_state_path,
            artifacts.root.join("latest_state.yaml")
        );
        assert_eq!(
            artifacts.step_log_path,
            artifacts.root.join("step_log.jsonl")
        );
        let tb_session = artifacts
            .tb_session_dir
            .file_name()
            .expect("tensorboard session dir name")
            .to_string_lossy();
        assert!(tb_session.starts_with("run_g00000007_"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn preflight_paths_use_default_cache_name_under_artifact_roots() {
        let output_dir = temp_dir_path("preflight_paths");
        let bc_artifacts = BcArtifactPaths::new(&output_dir, 0);
        let rl_artifacts = RlArtifactPaths::new(&output_dir, 0);

        let bc_preflight = PreflightPaths::new(&bc_artifacts);
        let rl_preflight = RlPreflightPaths::new(&rl_artifacts);

        assert_eq!(
            bc_preflight.cache_path,
            bc_artifacts.root.join(default_cache_name())
        );
        assert_eq!(
            bc_preflight.manifest_cache_path,
            bc_artifacts.root.join(default_manifest_cache_name())
        );
        assert_eq!(
            rl_preflight.cache_path,
            rl_artifacts.root.join(default_cache_name())
        );

        cleanup_dir(&output_dir);
    }

    #[test]
    fn artifact_directory_creators_make_expected_directories() {
        let output_dir = temp_dir_path("artifact_dir_create");
        let bc_artifacts = BcArtifactPaths::new(&output_dir, 3);
        let rl_artifacts = RlArtifactPaths::new(&output_dir, 9);

        bc_artifacts.create_root_dir().expect("create bc root");
        bc_artifacts
            .create_tensorboard_dirs()
            .expect("create bc tensorboard dirs");
        rl_artifacts.create_root_dir().expect("create rl root");
        rl_artifacts
            .create_tensorboard_dirs()
            .expect("create rl tensorboard dirs");

        assert!(bc_artifacts.root.is_dir());
        assert!(bc_artifacts.tb_root.is_dir());
        assert!(bc_artifacts.tb_session_dir.is_dir());
        assert!(rl_artifacts.root.is_dir());
        assert!(rl_artifacts.tb_root.is_dir());
        assert!(rl_artifacts.tb_session_dir.is_dir());

        cleanup_dir(&output_dir);
    }

    #[test]
    fn read_preflight_cache_returns_none_for_missing_file() {
        let output_dir = temp_dir_path("missing_preflight_cache");
        let path = output_dir.join("missing.json");

        let entry = read_preflight_cache(&path).expect("read missing cache path");
        assert_eq!(entry, None);
    }

    #[test]
    fn preflight_cache_roundtrips_through_json() {
        let output_dir = temp_dir_path("preflight_cache_roundtrip");
        fs::create_dir_all(&output_dir).expect("create temp dir");
        let path = output_dir.join("preflight_cache.json");
        let entry = sample_preflight_cache_entry();

        write_preflight_cache(&path, &entry).expect("write preflight cache");
        let restored = read_preflight_cache(&path)
            .expect("read preflight cache")
            .expect("cache entry present");

        assert_eq!(restored, entry);

        cleanup_dir(&output_dir);
    }

    #[test]
    fn preflight_cache_roundtrips_nested_benchmark_profiling() {
        let output_dir = temp_dir_path("preflight_cache_benchmark_roundtrip");
        fs::create_dir_all(&output_dir).expect("create temp dir");
        let path = output_dir.join("preflight_cache.json");
        let entry = sample_preflight_cache_entry_with_benchmark();

        write_preflight_cache(&path, &entry).expect("write preflight cache with benchmark");
        let restored = read_preflight_cache(&path)
            .expect("read preflight cache")
            .expect("cache entry present");

        assert_eq!(restored, entry);

        cleanup_dir(&output_dir);
    }

    #[test]
    fn manifest_cache_roundtrips_through_json() {
        let output_dir = temp_dir_path("manifest_cache_roundtrip");
        fs::create_dir_all(&output_dir).expect("create temp dir");
        let path = output_dir.join("preflight_manifest.json");
        let entry = sample_manifest_cache_entry();

        write_manifest_cache(&path, &entry).expect("write manifest cache");
        let restored = read_manifest_cache(&path)
            .expect("read manifest cache")
            .expect("manifest cache entry present");

        assert_eq!(restored, entry);

        cleanup_dir(&output_dir);
    }

    #[test]
    fn preflight_benchmark_paths_report_path_is_stable() {
        let artifacts = BcArtifactPaths::new(Path::new("/tmp/out"), 0);
        let paths = PreflightBenchmarkPaths::new(&artifacts);
        assert_eq!(
            paths.report_path(),
            Path::new("/tmp/out").join("bc/preflight_benchmark/report.json")
        );
    }

    #[test]
    fn preflight_benchmark_report_roundtrips_through_json() {
        let output_dir = temp_dir_path("preflight_benchmark_report_roundtrip");
        fs::create_dir_all(&output_dir).expect("create temp dir");
        let path = output_dir.join("preflight_benchmark/report.json");
        let benchmark = hydra_train::preflight::BenchmarkResult {
            runtime: hydra_train::preflight::BenchmarkRuntimeConfig {
                train_microbatch_size: 8,
                validation_microbatch_size: 4,
                accum_steps: 2,
                loader: hydra_train::preflight::LoaderRuntimeConfig {
                    num_threads: Some(2),
                    buffer_games: 32,
                    buffer_samples: 128,
                    archive_queue_bound: 4,
                },
            },
            score: hydra_train::preflight::BenchmarkScore {
                wall_clock_samples_per_second: 123.456,
                train_only_samples_per_second: 200.0,
                train_seconds: 1.0,
                validation_seconds: 0.5,
                checkpoint_seconds: 0.1,
                logging_seconds: 0.05,
                total_elapsed_seconds: 1.65,
                train_steps: 10,
                validation_samples: 50,
            },
            metadata: hydra_train::preflight::BenchmarkMetadata {
                mode: hydra_train::preflight::BenchmarkMode::CadenceAwareProjection,
                ..Default::default()
            },
            profiling: Some(hydra_train::preflight::ProfilingEnvelope::leaf(
                "stage_2_benchmark",
                1.5,
            )),
        };
        let report = PreflightBenchmarkReport {
            cache_key: sample_preflight_cache_entry().cache_key,
            runtime: sample_preflight_cache_entry().runtime,
            benchmark,
        };

        write_preflight_benchmark_report(&path, &report).expect("write preflight benchmark report");

        let raw = fs::read_to_string(&path).expect("read preflight benchmark report");
        let restored: PreflightBenchmarkReport =
            serde_json::from_str(&raw).expect("parse preflight benchmark report");

        assert_eq!(restored.cache_key, report.cache_key);
        assert_eq!(restored.runtime, report.runtime);
        assert_eq!(restored.benchmark, report.benchmark);

        cleanup_dir(&output_dir);
    }

    #[test]
    fn append_training_log_appends_jsonl_lines() {
        let output_dir = temp_dir_path("append_training_log");
        fs::create_dir_all(&output_dir).expect("create temp dir");
        let path = output_dir.join("training_log.jsonl");
        let first = sample_epoch_log_entry();
        let second = EpochLogEntry {
            epoch: first.epoch + 1,
            global_step: first.global_step + 10,
            ..sample_epoch_log_entry()
        };

        append_training_log(&path, &first).expect("append first training log line");
        append_training_log(&path, &second).expect("append second training log line");

        let raw = fs::read_to_string(&path).expect("read training log");
        let lines: Vec<_> = raw.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"epoch\":3"));
        assert!(lines[1].contains("\"epoch\":4"));
        assert!(lines[0].contains("\"profiling\":{"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn append_step_logs_append_jsonl_lines() {
        let output_dir = temp_dir_path("append_step_logs");
        fs::create_dir_all(&output_dir).expect("create temp dir");
        let step_path = output_dir.join("step_log.jsonl");
        let rl_path = output_dir.join("rl_step_log.jsonl");

        append_step_log(&step_path, &sample_step_log_entry()).expect("append step log");
        append_rl_step_log(&rl_path, &sample_rl_step_log_entry()).expect("append rl step log");

        let step_raw = fs::read_to_string(&step_path).expect("read step log");
        let rl_raw = fs::read_to_string(&rl_path).expect("read rl step log");
        assert!(step_raw.contains("\"global_step\":17"));
        assert!(step_raw.contains("\"best_val_policy_loss\":0.8"));
        assert!(step_raw.contains("\"profiling\":{"));
        assert!(rl_raw.contains("\"phase\":\"exit_pondering\""));
        assert!(rl_raw.contains("\"delta_q_state\":\"Active\""));
        assert!(rl_raw.contains("\"profiling\":{"));

        let advisories = vec![RuntimeAdvisory::warning(
            "cuda_shards_without_pinned_async_h2d",
            "CUDA shard run is under-optimized",
        )];
        let mut advisory_raw = Vec::new();
        append_advisory_event_to_writer(&mut advisory_raw, &AdvisoryEvent::startup(&advisories))
            .expect("append advisory event");
        let advisory_line = String::from_utf8(advisory_raw).expect("advisory json utf8");
        assert!(advisory_line.contains("\"event\":\"runtime_advisories\""));
        assert!(advisory_line.contains("\"scope\":\"startup\""));
        assert!(advisory_line.contains("cuda_shards_without_pinned_async_h2d"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn write_rl_resume_state_serializes_expected_yaml_fields() {
        let output_dir = temp_dir_path("write_rl_resume_state");
        fs::create_dir_all(&output_dir).expect("create temp dir");
        let path = output_dir.join("latest_state.yaml");
        let state = RlResumeState {
            schema_version: 1,
            resume_semantics: RlResumeSemantics::RestoreOptimizerFreshSelfPlay,
            global_step: 19,
            pipeline_state: PipelineState {
                phase: TrainingPhase::DrdaAchSelfPlay,
                gpu_hours_used: 12.5,
                total_games: 500,
                total_samples: 4000,
                learner_version: 3,
                actor_version: 4,
            },
            runtime: RlRuntimeResumeContract {
                games_per_batch: 16,
                microbatch_size: 32,
                phase: RlPhaseConfig::ExitPondering,
                precision_mode: crate::config::PrecisionMode::Fp32,
            },
            saved_at_unix_s: 123,
        };

        write_rl_resume_state(&path, &state).expect("write rl resume state");

        let raw = fs::read_to_string(&path).expect("read rl resume state");
        assert!(raw.contains("schema_version: 1"));
        assert!(raw.contains("global_step: 19"));
        assert!(raw.contains("phase: DrdaAchSelfPlay"));
        assert!(raw.contains("games_per_batch: 16"));
        assert!(raw.contains("microbatch_size: 32"));
        assert!(raw.contains("phase: exit_pondering"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn log_tensorboard_writes_core_and_validation_scalars() {
        let output_dir = temp_dir_path("tensorboard_full");
        fs::create_dir_all(&output_dir).expect("create tensorboard dir");
        let train = ScalarAverages {
            total_loss: 2.0,
            policy_agreement: 0.25,
            ..Default::default()
        };
        let val_summary = sample_validation_summary();
        let best_validation = BestValidation {
            policy_loss: 0.5,
            agreement: 0.9,
        };
        let mut tb = EventWriter::create(&output_dir).expect("create tb writer");

        log_tensorboard(
            &mut tb,
            11,
            &train,
            Some(&val_summary),
            0.001,
            Some(best_validation),
        )
        .expect("write tensorboard scalars");

        drop(tb);
        let tags = tensorboard_tags_from_dir(&output_dir);

        assert!(tags.iter().any(|tag| tag == "train/total_loss"));
        assert!(tags.iter().any(|tag| tag == "train/policy_agreement"));
        assert!(tags.iter().any(|tag| tag == "val/policy_agreement"));
        assert!(tags.iter().any(|tag| tag == "val/policy_loss"));
        assert!(tags.iter().any(|tag| tag == "val/total_loss"));
        assert!(
            tags.iter()
                .any(|tag| tag == "val/delta_q_candidate_top1_agreement")
        );
        assert!(
            tags.iter()
                .any(|tag| tag == "val/delta_q_offline_gate_passed")
        );
        assert!(tags.iter().any(|tag| tag == "lr"));
        assert!(tags.iter().any(|tag| tag == "val/best_policy_loss"));
        assert!(tags.iter().any(|tag| tag == "val/best_policy_agreement"));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn log_tensorboard_skips_optional_validation_scalars_when_absent() {
        let output_dir = temp_dir_path("tensorboard_train_only");
        fs::create_dir_all(&output_dir).expect("create tensorboard dir");
        let train = ScalarAverages {
            total_loss: 3.0,
            policy_agreement: 0.4,
            ..Default::default()
        };
        let mut tb = EventWriter::create(&output_dir).expect("create tb writer");

        log_tensorboard(&mut tb, 5, &train, None, 0.05, None).expect("write tensorboard scalars");

        drop(tb);
        let tags = tensorboard_tags_from_dir(&output_dir);

        assert!(tags.iter().any(|tag| tag == "train/total_loss"));
        assert!(tags.iter().any(|tag| tag == "train/policy_agreement"));
        assert!(tags.iter().any(|tag| tag == "lr"));
        assert!(!tags.iter().any(|tag| tag.starts_with("val/")));

        cleanup_dir(&output_dir);
    }

    #[test]
    fn delta_q_promotion_artifact_serializes_arena_fields() {
        let dir = temp_dir_path("delta_q_promotion_artifact");
        fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("delta_q_promotion.json");

        let report = DeltaQPromotionReport::new();
        let result = DeltaQPromotionResult {
            passed: true,
            criteria: Vec::new(),
        };
        let arena_request = DeltaQArenaConfirmationRequest::default();
        let paired = hydra_train::eval::paired_arena_result_from_placements(
            &[0, 1, 1, 2],
            &[1, 2, 2, 3],
            0.02,
        );
        let arena_report = delta_q_arena_report_from_paired_eval(&paired, -0.01);

        write_delta_q_promotion_artifact(
            &path,
            &PersistedDeltaQPromotionArtifact {
                scope: "promotion_mode",
                step_or_epoch: 0,
                recommendation: DeltaQPromotionRecommendation::RequiresArenaConfirmation,
                stage: "offline_transfer_and_arena_gate",
                arena_confirmation: Some(arena_request),
                arena_decision: Some(ArenaPromotionDecision::NonRegressionOnly),
                arena_report: Some(&arena_report),
                report: &report,
                result: &result,
                policy_transfer: None,
                policy_transfer_result: None,
            },
        )
        .expect("write artifact");

        let raw = fs::read_to_string(&path).expect("read artifact");
        assert!(raw.contains("\"arena_confirmation\""));
        assert!(raw.contains("\"arena_decision\""));
        assert!(raw.contains("\"arena_report\""));
        assert!(raw.contains("\"lower_confidence_bound_mean_placement\""));
        assert!(raw.contains("\"upper_confidence_bound_mean_placement\""));

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(&dir);
    }
}
