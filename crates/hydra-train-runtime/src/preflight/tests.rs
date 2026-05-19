use super::*;
use crate::config::{
    BcHyperparamConfig, EffectivePrecision, PrecisionMode, SourceFilterConfig, TrainConfig,
    ValidationGateConfig,
};

#[test]
fn candidate_ladder_is_sorted_unique_and_bounded() {
    let config = PreflightConfig {
        candidate_microbatches: vec![64, 512, 64, 256, 8, 1024],
        min_microbatch_size: 8,
        ..Default::default()
    };
    assert_eq!(candidate_ladder(&config, 256), vec![256, 64, 8]);
}

#[test]
fn resolve_runtime_config_preserves_batch_semantics() {
    let runtime = resolve_runtime_config(
        256,
        ExplicitSettings {
            train_microbatch_explicit: false,
            validation_microbatch_explicit: false,
        },
        64,
        128,
    );
    assert_eq!(runtime.train_microbatch_size, 64);
    assert_eq!(runtime.validation_microbatch_size, 128);
    assert_eq!(runtime.accum_steps, 4);
    assert_eq!(runtime.unsafe_selected_batch_size, None);
}

#[test]
fn preflight_defaults_include_search_policy_controls() {
    let config = PreflightConfig::default();
    assert_eq!(config.tuning_mode, PreflightTuningMode::Safe);
    assert_eq!(config.validation_growth_patience, 2);
    assert_eq!(config.validation_growth_max_steps, 6);
    assert!((config.measure_noise_tolerance_ratio - 0.02).abs() < f64::EPSILON);
    assert_eq!(config.loader_runtime_rounds, 2);
    assert!(config.local_refinement_enabled);
    assert_eq!(config.local_refinement_max_candidates, 3);
    assert_eq!(config.local_refinement_min_gap, 8);
    assert_eq!(config.local_refinement_extra_measure_steps, 2);
    assert_eq!(config.search_coordinate_rounds, 2);
    assert_eq!(config.search_top_k, 3);
    assert_eq!(config.rl_probe_min_free_memory_bytes, 0);
    assert!((config.rl_probe_memory_headroom_ratio - 0.0).abs() < f64::EPSILON);
    assert!((config.rl_probe_growth_safety_factor - 1.35).abs() < f64::EPSILON);
    assert!(config.unsafe_candidate_batch_sizes.is_empty());
}

#[test]
fn preflight_tuning_mode_round_trips_as_snake_case() {
    let config = PreflightConfig {
        tuning_mode: PreflightTuningMode::Unsafe,
        ..Default::default()
    };
    let serialized = serde_json::to_string(&config).expect("preflight config should serialize");
    assert!(serialized.contains("\"tuning_mode\":\"unsafe\""));

    let decoded: PreflightConfig = serde_json::from_str("{\"tuning_mode\":\"safe\"}")
        .expect("partial preflight config should deserialize with defaults");
    assert_eq!(decoded.tuning_mode, PreflightTuningMode::Safe);
}

#[test]
fn unsafe_candidate_batch_sizes_default_empty_and_round_trip() {
    let config = PreflightConfig {
        tuning_mode: PreflightTuningMode::Unsafe,
        unsafe_candidate_batch_sizes: vec![512, 1024],
        unsafe_candidate_lr_scales: vec![0.5, 2.0],
        unsafe_candidate_warmup_steps: vec![500, 1000],
        ..Default::default()
    };

    let serialized = serde_json::to_string(&config).expect("preflight config should serialize");
    assert!(serialized.contains("\"unsafe_candidate_batch_sizes\":[512,1024]"));
    assert!(serialized.contains("\"unsafe_candidate_lr_scales\":[0.5,2.0]"));
    assert!(serialized.contains("\"unsafe_candidate_warmup_steps\":[500,1000]"));

    let decoded: PreflightConfig = serde_json::from_str(
        "{\"tuning_mode\":\"unsafe\",\"unsafe_candidate_batch_sizes\":[128],\"unsafe_candidate_lr_scales\":[1.5],\"unsafe_candidate_warmup_steps\":[250]}",
    )
    .expect("unsafe candidate batch sizes should deserialize");
    assert_eq!(decoded.tuning_mode, PreflightTuningMode::Unsafe);
    assert_eq!(decoded.unsafe_candidate_batch_sizes, vec![128]);
    assert_eq!(decoded.unsafe_candidate_lr_scales, vec![1.5]);
    assert_eq!(decoded.unsafe_candidate_warmup_steps, vec![250]);
}

#[test]
fn candidate_ladder_falls_back_to_max_of_batch_and_minimum() {
    let config = PreflightConfig {
        candidate_microbatches: vec![8, 12],
        min_microbatch_size: 16,
        ..Default::default()
    };

    assert_eq!(candidate_ladder(&config, 10), vec![16]);
    assert_eq!(candidate_ladder(&config, 64), vec![64]);
}

#[test]
fn resolve_runtime_config_clamps_zero_microbatches() {
    let runtime = resolve_runtime_config(
        32,
        ExplicitSettings {
            train_microbatch_explicit: true,
            validation_microbatch_explicit: true,
        },
        0,
        0,
    );

    assert_eq!(runtime.train_microbatch_size, 1);
    assert_eq!(runtime.validation_microbatch_size, 1);
    assert_eq!(runtime.accum_steps, 32);
    assert_eq!(runtime.unsafe_selected_batch_size, None);
}

#[test]
fn resolve_runtime_config_caps_train_microbatch_without_overcounting_accumulation() {
    let runtime = resolve_runtime_config(
        32,
        ExplicitSettings {
            train_microbatch_explicit: false,
            validation_microbatch_explicit: false,
        },
        128,
        4,
    );

    assert_eq!(runtime.train_microbatch_size, 32);
    assert_eq!(runtime.validation_microbatch_size, 4);
    assert_eq!(runtime.accum_steps, 1);
}

#[test]
fn effective_runtime_config_reports_requested_and_effective_precision() {
    let selected = SelectedRuntimeConfig {
        train_microbatch_size: 64,
        validation_microbatch_size: 32,
        accum_steps: 4,
        unsafe_selected_batch_size: None,
        unsafe_selected_learning_rate: None,
        unsafe_selected_min_learning_rate: None,
        unsafe_selected_warmup_steps: None,
    };
    let loader = LoaderRuntimeConfig {
        num_threads: Some(2),
        buffer_games: 16,
        buffer_samples: 128,
        archive_queue_bound: 8,
    };

    let fp32_config = dummy_config();
    let fp32_runtime = EffectiveRuntimeConfig::from_config(selected, loader, &fp32_config);
    assert_eq!(fp32_runtime.requested_precision, PrecisionMode::Fp32);
    assert_eq!(fp32_runtime.effective_precision, EffectivePrecision::Fp32);

    let mut bf16_config = dummy_config();
    bf16_config.precision_mode = PrecisionMode::Bf16Autocast;
    bf16_config.device = "cuda:0".to_string();
    let bf16_runtime = EffectiveRuntimeConfig::from_config(selected, loader, &bf16_config);
    assert_eq!(
        bf16_runtime.requested_precision,
        PrecisionMode::Bf16Autocast
    );
    assert_eq!(
        bf16_runtime.effective_precision,
        EffectivePrecision::Bf16Amp
    );
}

#[test]
fn effective_precision_fp32_noop_serde_matches_display_spelling() {
    let precision = EffectivePrecision::Fp32NoopForBf16Request;
    let serialized =
        serde_json::to_string(&precision).expect("effective precision should serialize");
    assert_eq!(serialized, format!("\"{}\"", precision));

    let decoded: EffectivePrecision =
        serde_json::from_str(&serialized).expect("display spelling should deserialize");
    assert_eq!(decoded, precision);

    let decoded_from_canonical: EffectivePrecision =
        serde_json::from_str("\"fp32_noop\"").expect("canonical fp32_noop should deserialize");
    assert_eq!(decoded_from_canonical, precision);
}

#[test]
fn benchmark_metadata_defaults_are_backward_safe() {
    let metadata = BenchmarkMetadata::default();

    assert_eq!(metadata.mode, BenchmarkMode::NotRecorded);
    assert!(metadata.selection_metric.is_empty());
    assert_eq!(metadata.finalists_benchmarked, 0);
    assert_eq!(metadata.projected_validation_events, 0.0);
    assert_eq!(metadata.projected_checkpoint_events, 0.0);
    assert_eq!(metadata.projected_logging_events, 0.0);
}

#[test]
fn profiling_envelope_merges_nested_children_by_stage() {
    let mut envelope = ProfilingEnvelope::nested(
        PROFILING_STAGE_VALIDATION,
        1.0,
        vec![ProfilingEnvelope::leaf(
            PROFILING_STAGE_CANDIDATE_FORWARD_AND_LOSS,
            0.75,
        )],
    );
    envelope.merge_assign(&ProfilingEnvelope::nested(
        PROFILING_STAGE_VALIDATION,
        2.0,
        vec![
            ProfilingEnvelope::leaf(PROFILING_STAGE_CANDIDATE_FORWARD_AND_LOSS, 1.25),
            ProfilingEnvelope::leaf(PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD, 0.5),
        ],
    ));

    assert_eq!(envelope.stage, PROFILING_STAGE_VALIDATION);
    assert!((envelope.elapsed_seconds - 3.0).abs() < 1e-12);
    assert_eq!(envelope.children.len(), 2);
    assert_eq!(
        envelope.children[0].stage,
        PROFILING_STAGE_CANDIDATE_FORWARD_AND_LOSS
    );
    assert!((envelope.children[0].elapsed_seconds - 2.0).abs() < 1e-12);
    assert_eq!(
        envelope.children[1].stage,
        PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD
    );
    assert!((envelope.children[1].elapsed_seconds - 0.5).abs() < 1e-12);
}

#[test]
fn benchmark_result_defaults_optional_profiling_when_missing() {
    let benchmark: BenchmarkResult = serde_json::from_str(
        r#"{
            "runtime": {
                "train_microbatch_size": 64,
                "validation_microbatch_size": 128,
                "accum_steps": 4,
                "loader": {
                    "num_threads": 8,
                    "buffer_games": 512,
                    "buffer_samples": 2048,
                    "archive_queue_bound": 32
                }
            },
            "score": {
                "wall_clock_samples_per_second": 123.0,
                "train_only_samples_per_second": 150.0,
                "train_seconds": 10.0,
                "validation_seconds": 2.0,
                "checkpoint_seconds": 0.5,
                "logging_seconds": 0.25,
                "total_elapsed_seconds": 12.75,
                "train_steps": 64,
                "validation_samples": 4096
            },
            "metadata": {
                "mode": "cadence_aware_projection",
                "selection_metric": "wall_clock_effective_throughput",
                "train_probe_candidates_considered": 4,
                "validation_probe_candidates_considered": 4,
                "loader_candidates_considered": 3,
                "finalists_benchmarked": 2,
                "warmup_steps": 8,
                "measured_train_steps": 64,
                "projected_validation_events": 6.0,
                "projected_checkpoint_events": 6.0,
                "projected_logging_events": 6.0
            }
        }"#,
    )
    .expect("benchmark without profiling should deserialize");

    assert!(benchmark.profiling.is_none());
    assert_eq!(
        benchmark.metadata.mode,
        BenchmarkMode::CadenceAwareProjection
    );
}

#[test]
fn system_metrics_event_serializes_as_lightweight_sparse_schema() {
    let event = SystemMetricsEvent {
        kind: SystemMetricEventKind::ProbeHostMemory,
        probe_kind: Some(ProbeKind::Train),
        candidate_microbatch: Some(64),
        mem_available_bytes: Some(1024),
        mem_total_bytes: Some(2048),
        ..SystemMetricsEvent::default()
    };

    let encoded = serde_json::to_string(&event).expect("system metric event should serialize");

    assert!(encoded.contains("\"kind\":\"probe_host_memory\""));
    assert!(encoded.contains("\"probe_kind\":\"train\""));
    assert!(encoded.contains("\"candidate_microbatch\":64"));
    assert!(encoded.contains("\"mem_available_bytes\":1024"));
    assert!(!encoded.contains("model_init_ms"));
}

#[test]
fn progress_metric_event_serializes_sparse_rates_and_planned_counts() {
    let event = SystemMetricsEvent {
        kind: SystemMetricEventKind::Progress,
        phase: Some("manifest_scan".to_string()),
        completed: Some(10),
        planned: Some(20),
        elapsed_seconds: Some(2.0),
        files_per_second: Some(5.0),
        ..SystemMetricsEvent::default()
    };

    let encoded = serde_json::to_string(&event).expect("progress metric should serialize");

    assert!(encoded.contains("\"kind\":\"progress\""));
    assert!(encoded.contains("\"phase\":\"manifest_scan\""));
    assert!(encoded.contains("\"completed\":10"));
    assert!(encoded.contains("\"planned\":20"));
    assert!(encoded.contains("\"files_per_second\":5.0"));
    assert!(!encoded.contains("gpu_util_percent"));
}

#[test]
fn from_children_sums_child_elapsed_seconds() {
    let envelope = ProfilingEnvelope::from_children(
        "parent",
        vec![
            ProfilingEnvelope::leaf("train", 1.5),
            ProfilingEnvelope::leaf("validation", 0.5),
            ProfilingEnvelope::leaf("checkpoint", 0.25),
        ],
    );

    assert_eq!(envelope.stage, "parent");
    assert!((envelope.elapsed_seconds - 2.25).abs() < 1e-10);
    assert_eq!(envelope.children.len(), 3);
}

#[test]
fn merge_assign_non_matching_stages_pushes_as_new_child() {
    let mut base =
        ProfilingEnvelope::from_children("epoch", vec![ProfilingEnvelope::leaf("train", 1.0)]);
    let other =
        ProfilingEnvelope::from_children("step", vec![ProfilingEnvelope::leaf("validation", 0.5)]);

    base.merge_assign(&other);

    assert_eq!(base.children.len(), 2);
    assert_eq!(base.children[0].stage, "train");
    assert_eq!(base.children[1].stage, "step");
    assert!((base.elapsed_seconds - 1.5).abs() < 1e-10);
}

#[test]
fn merge_assign_matching_stages_aggregates_children() {
    let mut base = ProfilingEnvelope::from_children(
        "epoch",
        vec![
            ProfilingEnvelope::leaf("train", 1.0),
            ProfilingEnvelope::leaf("validation", 0.5),
        ],
    );
    let other = ProfilingEnvelope::from_children(
        "epoch",
        vec![
            ProfilingEnvelope::leaf("train", 2.0),
            ProfilingEnvelope::leaf("checkpoint", 0.3),
        ],
    );

    base.merge_assign(&other);

    assert_eq!(base.children.len(), 3);
    let train = base.children.iter().find(|c| c.stage == "train").unwrap();
    assert!((train.elapsed_seconds - 3.0).abs() < 1e-10);
    let checkpoint = base
        .children
        .iter()
        .find(|c| c.stage == "checkpoint")
        .unwrap();
    assert!((checkpoint.elapsed_seconds - 0.3).abs() < 1e-10);
    assert!((base.elapsed_seconds - 3.8).abs() < 1e-10);
}

#[test]
fn profiling_stage_constants_are_all_distinct() {
    let all = [
        PROFILING_STAGE_STAGE_2_BENCHMARK,
        PROFILING_STAGE_BC_INTERVAL,
        PROFILING_STAGE_BC_EPOCH,
        PROFILING_STAGE_RL_STEP,
        PROFILING_STAGE_TRAIN,
        PROFILING_STAGE_VALIDATION,
        PROFILING_STAGE_CHECKPOINT,
        PROFILING_STAGE_LOGGING,
        PROFILING_STAGE_SELF_PLAY,
        PROFILING_STAGE_CANDIDATE_FORWARD_AND_LOSS,
        PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD,
        PROFILING_STAGE_COLLATION,
        PROFILING_STAGE_FORWARD,
        PROFILING_STAGE_LOSS,
        PROFILING_STAGE_LOSS_POLICY_CE,
        PROFILING_STAGE_LOSS_VALUE_MSE,
        PROFILING_STAGE_LOSS_BASE_HEADS,
        PROFILING_STAGE_LOSS_ADVANCED_HEADS,
        PROFILING_STAGE_LOSS_TOTAL_COMBINE,
        PROFILING_STAGE_LOSS_EXIT,
        PROFILING_STAGE_BACKWARD,
        PROFILING_STAGE_OPTIMIZER_STEP,
        PROFILING_STAGE_PRODUCER_WAIT,
        PROFILING_STAGE_H2D_TRANSFER,
        PROFILING_STAGE_H2D_PAGEABLE_TO_PINNED,
        PROFILING_STAGE_H2D_TENSOR_MATERIALIZE,
        PROFILING_STAGE_H2D_STREAM_SYNC,
        PROFILING_STAGE_METRIC_READBACK,
        PROFILING_STAGE_DATA_LOAD,
        PROFILING_STAGE_PREFLIGHT_MODEL_INIT,
        PROFILING_STAGE_PREFLIGHT_OPTIMIZER_INIT,
        PROFILING_STAGE_PREFLIGHT_LOSS_INIT,
        PROFILING_STAGE_PREFLIGHT_DATA_STREAM_INIT,
        PROFILING_STAGE_PREFLIGHT_SHARD_LOAD,
        PROFILING_STAGE_PREFLIGHT_CUDA_STAGING,
    ];
    let set: std::collections::HashSet<&str> = all.iter().copied().collect();
    assert_eq!(
        set.len(),
        all.len(),
        "profiling stage constants must be unique"
    );
}

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
        bc_shards_manifest_path: None,
        shard_prefetch_depth: None,
        train_fraction: 0.875,
        source_filters: SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        seed: 7,
        advanced_loss: None,
        validation_gates: ValidationGateConfig::default(),
        rl: None,
        bc: BcHyperparamConfig::default(),
        nsight_trace: None,
        device: "cpu".to_string(),
        precision_mode: PrecisionMode::Fp32,
        buffer_games: 16,
        buffer_samples: 128,
        num_threads: Some(2),
        tensorboard: false,
        archive_queue_bound: 8,
        validation_every_n_epochs: 1,
        max_skip_logs_per_source: 4,
        log_every_n_steps: 5,
        validate_every_n_steps: 6,
        checkpoint_every_n_steps: 7,
        max_train_steps: Some(9),
        max_validation_batches: Some(3),
        max_validation_samples: Some(99),
    }
}

#[test]
fn preflight_bench_report_has_no_cache_or_runtime_fields() {
    let report = PreflightBenchReport {
        schema_version: 1,
        rows: vec![PreflightBenchRow {
            index: 0,
            status: PreflightBenchStatus::Pass,
            device: "cpu".to_string(),
            mode: PreflightBenchMode::LoaderOnly,
            batch_size: 1024,
            ring_batches: 2,
            loader_threads: 1,
            prefetch_batches: 1,
            shuffle: PreflightShuffleMode::None,
            codec: PreflightCodec::None,
            samples_per_second: Some(1.0),
            mib_per_second: None,
            p50_batch_ms: None,
            p95_batch_ms: None,
            producer_wait_ratio: None,
            consumer_wait_ratio: None,
            disk_wait_ratio: None,
            gpu_input_wait_ratio: None,
            cpu_user_seconds: None,
            cpu_system_seconds: None,
            error: None,
        }],
        total_elapsed_seconds: 0.0,
    };
    let json = serde_json::to_string(&report).expect("report serializes");
    for forbidden in [
        "selected",
        "cache_hit",
        "runtime",
        "cache_key",
        "saved",
        "best",
        "recommended",
    ] {
        assert!(
            !json.contains(forbidden),
            "forbidden word in report: {forbidden}"
        );
    }
    assert!(json.contains("disk_wait_ratio"));
    assert!(json.contains("gpu_input_wait_ratio"));
}
