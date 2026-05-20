use super::*;

use hydra_train_runtime::preflight::ProbeStatus;
use std::path::PathBuf;

fn dummy_train_config() -> TrainConfig {
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
        train_fraction: 0.9,
        source_filters: Default::default(),
        augment: true,
        resume_checkpoint: None,
        seed: 0,
        advanced_loss: None,
        validation_gates: Default::default(),
        rl: None,
        bc: Default::default(),
        nsight_trace: None,
        device: "cpu".to_string(),
        precision_mode: Default::default(),
        buffer_games: 1,
        buffer_samples: 1,
        num_threads: Some(1),
        tensorboard: false,
        archive_queue_bound: 1,
        validation_every_n_epochs: 1,
        max_skip_logs_per_source: 0,
        log_every_n_steps: 50,
        validate_every_n_steps: 200,
        checkpoint_every_n_steps: 200,
        max_train_steps: None,
        max_validation_batches: None,
        max_validation_samples: Some(1),
    }
}

fn config() -> TrainConfig {
    let mut config = dummy_train_config();
    config.data_dir = PathBuf::from("/data");
    config.output_dir = PathBuf::from("/output");
    config
}

fn keys(advisories: &[RuntimeAdvisory]) -> Vec<&'static str> {
    advisories.iter().map(|advisory| advisory.key).collect()
}

fn advisory<'a>(advisories: &'a [RuntimeAdvisory], key: &str) -> &'a RuntimeAdvisory {
    advisories
        .iter()
        .find(|advisory| advisory.key == key)
        .expect("advisory key present")
}

fn probe_result(kind: ProbeKind, candidate: usize, samples_per_second: Option<f64>) -> ProbeResult {
    ProbeResult {
        kind,
        candidate_microbatch: candidate,
        status: if samples_per_second.is_some() {
            ProbeStatus::Success
        } else {
            ProbeStatus::Oom
        },
        measured_samples_per_second: samples_per_second,
        elapsed_seconds: samples_per_second.map(|_| 1.0),
        detail: String::new(),
    }
}

#[test]
fn startup_advisories_report_cpu_training() {
    let mut config = config();
    config.device = "cpu".to_string();

    let advisories =
        startup_runtime_advisories(&config, MicrobatchExplicitness::from_config(&config));

    assert_eq!(
        advisory(&advisories, "cpu_device_for_training").severity,
        AdvisorySeverity::Info
    );
}

#[test]
fn startup_advisories_report_cpu_with_cuda_available_when_detected() {
    let mut config = config();
    config.device = "cpu".to_string();

    let advisories =
        startup_runtime_advisories(&config, MicrobatchExplicitness::from_config(&config));
    let keys = keys(&advisories);

    if tch::Cuda::device_count() > 0 {
        let advisory = advisory(&advisories, "cpu_device_with_cuda_available");
        assert_eq!(advisory.severity, AdvisorySeverity::Warning);
        assert!(advisory.message.contains("CPU training is super slow"));
        assert!(advisory.message.contains("model forward/backward/optimizer/H2D"));
        assert!(advisory.message.contains("materialization still runs on CPU"));
    } else {
        assert!(keys.contains(&"cpu_device_for_training"));
    }
}

#[test]
fn startup_advisories_report_cuda_loose_replay() {
    let mut config = config();
    config.device = "cuda:0".to_string();
    config.bc_shards_manifest_path = None;

    let advisories =
        startup_runtime_advisories(&config, MicrobatchExplicitness::from_config(&config));

    assert_eq!(
        advisory(&advisories, "steady_state_cuda_bc_uses_loose_replay").severity,
        AdvisorySeverity::Info
    );
    assert_eq!(
        advisory(&advisories, "optimized_path_raw_replay").severity,
        AdvisorySeverity::Info
    );
    assert!(
        advisory(&advisories, "optimized_path_raw_replay")
            .message
            .contains("input=raw_replay pinned_h2d=off")
    );
}

#[cfg(feature = "cuda-graph")]
#[test]
fn startup_advisories_report_cuda_raw_pinned_h2d_staging_when_feature_active() {
    let mut config = config();
    config.device = "cuda:0".to_string();
    config.bc_shards_manifest_path = None;

    let advisories =
        startup_runtime_advisories(&config, MicrobatchExplicitness::from_config(&config));

    assert!(
        advisory(&advisories, "optimized_path_raw_replay")
            .message
            .contains("input=raw_replay pinned_h2d=on prealloc_gpu_tensors=on")
    );
}

#[cfg(not(feature = "cuda-graph"))]
#[test]
fn startup_advisories_report_cuda_shards_without_pinned_async_h2d() {
    let mut config = config();
    config.device = "cuda:0".to_string();
    config.bc_shards_manifest_path = Some(PathBuf::from("/shards/manifest.json"));

    let advisories =
        startup_runtime_advisories(&config, MicrobatchExplicitness::from_config(&config));

    assert_eq!(
        advisory(&advisories, "cuda_shards_without_pinned_async_h2d").severity,
        AdvisorySeverity::Info
    );
    assert_eq!(
        advisory(&advisories, "steady_state_cuda_bc_shards_enabled").severity,
        AdvisorySeverity::Info
    );
    assert_eq!(
        advisory(&advisories, "optimized_path_bc_shards").severity,
        AdvisorySeverity::Info
    );
    assert!(
        advisory(&advisories, "optimized_path_bc_shards")
            .message
            .contains("input=bc_shards pinned_h2d=off")
    );
}

#[cfg(feature = "cuda-graph")]
#[test]
fn startup_advisories_report_pinned_h2d_staging_when_feature_active() {
    let mut config = config();
    config.device = "cuda:0".to_string();
    config.bc_shards_manifest_path = Some(PathBuf::from("/shards/manifest.json"));

    let advisories =
        startup_runtime_advisories(&config, MicrobatchExplicitness::from_config(&config));
    let keys = keys(&advisories);

    assert!(!keys.contains(&"cuda_shards_without_pinned_async_h2d"));
    assert_eq!(
        advisory(&advisories, "steady_state_cuda_bc_shards_enabled").severity,
        AdvisorySeverity::Info
    );
    assert_eq!(
        advisory(&advisories, "cuda_shards_pinned_h2d_staging_enabled").severity,
        AdvisorySeverity::Info
    );
    assert!(
        advisory(&advisories, "optimized_path_bc_shards")
            .message
            .contains("input=bc_shards pinned_h2d=on prealloc_gpu_tensors=on")
    );
}

#[test]
fn startup_advisories_explain_threads_do_not_parallelize_shards() {
    let mut config = config();
    config.device = "cuda:0".to_string();
    config.bc_shards_manifest_path = Some(PathBuf::from("/shards/manifest.json"));
    config.num_threads = Some(8);

    let advisories =
        startup_runtime_advisories(&config, MicrobatchExplicitness::from_config(&config));

    assert!(keys(&advisories).contains(&"num_threads_not_shard_parallelism"));
}

#[test]
fn startup_advisories_report_explicit_microbatch_and_cadence() {
    let mut config = config();
    config.device = "cuda:0".to_string();
    config.batch_size = 128;
    config.microbatch_size = Some(32);
    config.validation_microbatch_size = Some(64);
    config.log_every_n_steps = 1;
    config.validate_every_n_steps = 1;

    let advisories =
        startup_runtime_advisories(&config, MicrobatchExplicitness::from_config(&config));

    assert_eq!(
        advisory(&advisories, "small_microbatch_high_accumulation_overhead").severity,
        AdvisorySeverity::Info
    );
    assert_eq!(
        advisory(
            &advisories,
            "explicit_microbatch_blocks_faster_candidate_search"
        )
        .severity,
        AdvisorySeverity::Info
    );
    assert_eq!(
        advisory(&advisories, "logging_or_metric_sync_overhead").severity,
        AdvisorySeverity::Info
    );
    assert_eq!(
        advisory(&advisories, "validation_or_checkpoint_cadence_overhead").severity,
        AdvisorySeverity::Info
    );
}

#[test]
fn startup_advisories_do_not_treat_cached_runtime_as_user_explicit() {
    let mut config = config();
    config.device = "cuda:0".to_string();
    config.microbatch_size = None;
    config.validation_microbatch_size = None;
    let original_explicitness = MicrobatchExplicitness::from_config(&config);
    config.microbatch_size = Some(128);
    config.validation_microbatch_size = Some(128);

    let advisories = startup_runtime_advisories(&config, original_explicitness);

    assert!(!keys(&advisories).contains(&"explicit_microbatch_blocks_faster_candidate_search"));
}

#[test]
fn startup_advisories_report_enabled_full_microbatch_as_info() {
    let mut config = config();
    config.device = "cuda:0".to_string();
    config.batch_size = 128;
    config.microbatch_size = Some(128);
    config.validation_microbatch_size = Some(128);

    let advisories = startup_runtime_advisories(&config, MicrobatchExplicitness::default());

    assert_eq!(
        advisory(&advisories, "full_train_microbatch_enabled").severity,
        AdvisorySeverity::Info
    );
    assert!(!keys(&advisories).contains(&"small_microbatch_high_accumulation_overhead"));
}

#[test]
fn advisory_deduper_keeps_first_key_once() {
    let mut deduper = AdvisoryDeduper::new();
    let first = RuntimeAdvisory::info("same", "first");
    let second = RuntimeAdvisory::warning("same", "second");

    let kept = deduper.retain_new(vec![first.clone(), second]);
    let kept_again = deduper.retain_new(vec![first]);

    assert_eq!(kept.len(), 1);
    assert!(kept_again.is_empty());
}

#[test]
fn selected_runtime_probe_advisory_reports_materially_slower_train_candidate() {
    let advisories = selected_runtime_probe_advisories(
        ProbeKind::Train,
        32,
        &[
            probe_result(ProbeKind::Train, 32, Some(790.0)),
            probe_result(ProbeKind::Train, 64, Some(1000.0)),
        ],
    );

    assert_eq!(advisories.len(), 1);
    assert_eq!(
        advisories[0].key,
        "selected_train_runtime_slower_than_best_probe_candidate"
    );
    assert_eq!(advisories[0].severity, AdvisorySeverity::Warning);
    assert!(
        advisories[0]
            .message
            .contains("selected train microbatch=32")
    );
    assert!(
        advisories[0]
            .message
            .contains("best stable measured train microbatch=64")
    );
}

#[test]
fn selected_runtime_probe_advisory_ignores_non_material_probe_gap() {
    let advisories = selected_runtime_probe_advisories(
        ProbeKind::Train,
        32,
        &[
            probe_result(ProbeKind::Train, 32, Some(810.0)),
            probe_result(ProbeKind::Train, 64, Some(1000.0)),
        ],
    );

    assert!(advisories.is_empty());
}

#[test]
fn selected_runtime_probe_advisory_ignores_selected_best_or_unmeasured() {
    assert!(
        selected_runtime_probe_advisories(
            ProbeKind::Train,
            64,
            &[
                probe_result(ProbeKind::Train, 32, Some(790.0)),
                probe_result(ProbeKind::Train, 64, Some(1000.0)),
            ],
        )
        .is_empty()
    );
    assert!(
        selected_runtime_probe_advisories(
            ProbeKind::Train,
            32,
            &[
                probe_result(ProbeKind::Train, 32, None),
                probe_result(ProbeKind::Train, 64, Some(1000.0)),
            ],
        )
        .is_empty()
    );
}

#[test]
fn selected_runtime_probe_advisory_uses_distinct_train_validation_keys() {
    let train = selected_runtime_probe_advisories(
        ProbeKind::Train,
        32,
        &[
            probe_result(ProbeKind::Train, 32, Some(790.0)),
            probe_result(ProbeKind::Train, 64, Some(1000.0)),
        ],
    );
    let validation = selected_runtime_probe_advisories(
        ProbeKind::Validation,
        32,
        &[
            probe_result(ProbeKind::Validation, 32, Some(790.0)),
            probe_result(ProbeKind::Validation, 64, Some(1000.0)),
        ],
    );

    let mut deduper = AdvisoryDeduper::new();
    let kept = deduper.retain_new([train, validation].concat());

    assert_eq!(kept.len(), 2);
    let keys = keys(&kept);
    assert!(keys.contains(&"selected_train_runtime_slower_than_best_probe_candidate"));
    assert!(keys.contains(&"selected_validation_runtime_slower_than_best_probe_candidate"));
}

fn interval_input() -> IntervalTimingInput {
    IntervalTimingInput {
        producer_wait_seconds: 0.0,
        collation_seconds: 0.0,
        h2d_transfer_seconds: 0.0,
        h2d_pageable_to_pinned_seconds: 0.0,
        h2d_tensor_materialize_seconds: 0.0,
        h2d_stream_sync_seconds: 0.0,
        metric_readback_seconds: 0.0,
        forward_seconds: 0.0,
        backward_seconds: 0.0,
        optimizer_step_seconds: 0.0,
        validation_seconds: 0.0,
        checkpoint_seconds: 0.0,
        logging_seconds: 0.0,
        kernel_launch_count: None,
        tiny_kernel_fraction: None,
        cuda_runtime_launch_seconds: None,
        total_seconds: 10.0,
        steps: 4,
        is_cuda: true,
    }
}

#[test]
fn interval_advisories_report_data_and_metric_overheads() {
    let mut input = interval_input();
    input.producer_wait_seconds = 1.5;
    input.collation_seconds = 1.0;
    input.h2d_transfer_seconds = 1.0;
    input.metric_readback_seconds = 0.6;
    input.logging_seconds = 0.5;

    let advisories = interval_runtime_advisories(input);
    let keys = keys(&advisories);

    assert!(keys.contains(&"cpu_producer_lag"));
    assert!(keys.contains(&"logging_or_metric_sync_overhead"));
}

#[test]
fn interval_advisories_report_h2d_framework_materialization_overhead() {
    let mut input = interval_input();
    input.h2d_transfer_seconds = 2.0;
    input.h2d_pageable_to_pinned_seconds = 0.2;
    input.h2d_tensor_materialize_seconds = 1.4;
    input.h2d_stream_sync_seconds = 0.2;

    let advisories = interval_runtime_advisories(input);
    let advisory = advisory(&advisories, "h2d_framework_materialization_overhead");

    assert_eq!(advisory.severity, AdvisorySeverity::Warning);
    assert!(advisory.message.contains("tensor_materialize=1.400s"));
    assert!(
        advisory
            .message
            .contains("raw PCIe copy may not be the bottleneck")
    );
}

#[test]
fn interval_advisories_report_cuda_launch_fragmentation_from_trace_metrics() {
    let mut input = interval_input();
    input.kernel_launch_count = Some(419_721);
    input.tiny_kernel_fraction = Some(0.82);
    input.cuda_runtime_launch_seconds = Some(1.012);

    let advisories = interval_runtime_advisories(input);
    let advisory = advisory(&advisories, "cuda_launch_fragmentation_overhead");

    assert_eq!(advisory.severity, AdvisorySeverity::Warning);
    assert!(advisory.message.contains("419721 kernel launches"));
    assert!(advisory.message.contains("CUDA graph replay"));
}

#[test]
fn interval_advisories_report_optimizer_and_model_dominance() {
    let mut input = interval_input();
    input.optimizer_step_seconds = 3.0;
    input.forward_seconds = 2.0;
    input.backward_seconds = 2.6;

    let advisories = interval_runtime_advisories(input);
    let keys = keys(&advisories);

    assert!(keys.contains(&"optimizer_step_dominates_cuda_interval"));
    assert!(keys.contains(&"model_compute_dominates_cuda_interval"));
    let optimizer = advisory(&advisories, "optimizer_step_dominates_cuda_interval");
    assert_eq!(optimizer.severity, AdvisorySeverity::Warning);
    assert!(optimizer.message.contains("Burn Adam is unfused"));
    let model = advisory(&advisories, "model_compute_dominates_cuda_interval");
    assert_eq!(model.severity, AdvisorySeverity::Info);
    assert!(model.message.contains("static-input parity proof"));
}

#[test]
fn interval_advisories_suppress_tiny_windows_and_cpu_data_claims() {
    let mut input = interval_input();
    input.steps = 1;
    input.producer_wait_seconds = 9.0;
    assert!(interval_runtime_advisories(input).is_empty());

    input.steps = 4;
    input.is_cuda = false;
    let advisories = interval_runtime_advisories(input);
    assert!(!keys(&advisories).contains(&"cpu_producer_lag"));
}

#[test]
fn interval_event_serializes_interval_scope() {
    let advisories = vec![RuntimeAdvisory::info(
        "validation_or_checkpoint_cadence_overhead",
        "cadence",
    )];
    let json = serde_json::to_value(AdvisoryEvent::interval(&advisories)).expect("serialize event");

    assert_eq!(json["event"], "runtime_advisories");
    assert_eq!(json["scope"], "interval");
    assert_eq!(
        json["advisories"][0]["key"],
        "validation_or_checkpoint_cadence_overhead"
    );
}
