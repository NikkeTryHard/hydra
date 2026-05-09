use std::collections::BTreeSet;

use hydra_train::preflight::{ProbeKind, ProbeResult};
use serde::Serialize;

use super::config::TrainConfig;
use super::config_runtime::{
    default_num_threads_for_system, train_microbatch_size, validation_microbatch_size,
};
use super::probe_summary::{best_probe_summary, candidate_average, probe_kind_name};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum AdvisorySeverity {
    Info,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct RuntimeAdvisory {
    pub(super) key: &'static str,
    pub(super) severity: AdvisorySeverity,
    pub(super) message: String,
}

impl RuntimeAdvisory {
    pub(super) fn info(key: &'static str, message: impl Into<String>) -> Self {
        Self {
            key,
            severity: AdvisorySeverity::Info,
            message: message.into(),
        }
    }

    pub(super) fn warning(key: &'static str, message: impl Into<String>) -> Self {
        Self {
            key,
            severity: AdvisorySeverity::Warning,
            message: message.into(),
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct AdvisoryDeduper {
    seen: BTreeSet<&'static str>,
}

impl AdvisoryDeduper {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn retain_new(&mut self, advisories: Vec<RuntimeAdvisory>) -> Vec<RuntimeAdvisory> {
        advisories
            .into_iter()
            .filter(|advisory| self.seen.insert(advisory.key))
            .collect()
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct MicrobatchExplicitness {
    pub(super) train: bool,
    pub(super) validation: bool,
}

impl MicrobatchExplicitness {
    pub(super) fn from_config(config: &TrainConfig) -> Self {
        Self {
            train: config.microbatch_size.is_some(),
            validation: config.validation_microbatch_size.is_some(),
        }
    }

    fn any(self) -> bool {
        self.train || self.validation
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum AdvisoryEventScope {
    Startup,
    Interval,
}

#[derive(Debug, Serialize)]
pub(super) struct AdvisoryEvent<'a> {
    pub(super) event: &'static str,
    pub(super) scope: AdvisoryEventScope,
    pub(super) advisories: &'a [RuntimeAdvisory],
}

impl<'a> AdvisoryEvent<'a> {
    pub(super) fn startup(advisories: &'a [RuntimeAdvisory]) -> Self {
        Self {
            event: "runtime_advisories",
            scope: AdvisoryEventScope::Startup,
            advisories,
        }
    }

    pub(super) fn interval(advisories: &'a [RuntimeAdvisory]) -> Self {
        Self {
            event: "runtime_advisories",
            scope: AdvisoryEventScope::Interval,
            advisories,
        }
    }
}

pub(super) fn startup_runtime_advisories(
    config: &TrainConfig,
    explicitness: MicrobatchExplicitness,
) -> Vec<RuntimeAdvisory> {
    let device = config.device.trim().to_ascii_lowercase();
    let is_cuda = device == "cuda" || device.starts_with("cuda:");
    let train_microbatch = train_microbatch_size(config);
    let validation_microbatch = validation_microbatch_size(config);
    let mut advisories = Vec::new();

    if !is_cuda {
        advisories.push(RuntimeAdvisory::warning(
            "cpu_device_for_training",
            "device is CPU; CUDA feeding optimizations, pinned H2D staging, and CUDA graphs are off",
        ));
    }

    if is_cuda && config.bc_shards_manifest_path.is_none() {
        advisories.push(RuntimeAdvisory::warning(
            "steady_state_cuda_bc_uses_loose_replay",
            "CUDA BC run uses loose/archive replay input; for steady-state throughput build BC shards and set bc_shards_manifest_path",
        ));
        advisories.push(RuntimeAdvisory::info(
            "optimized_path_raw_replay",
            "runtime path: input=raw_replay pinned_h2d=off prealloc_gpu_tensors=off cuda_graph_replay=production_off_probe_only copy_compute_overlap=off",
        ));
    }

    if is_cuda && config.bc_shards_manifest_path.is_some() {
        advisories.push(RuntimeAdvisory::info(
            "steady_state_cuda_bc_shards_enabled",
            "CUDA BC shard input is on; replay parsing/encoding is out of the steady-state training hot path",
        ));

        if cuda_graph_feature_active() {
            advisories.push(RuntimeAdvisory::info(
                "cuda_shards_pinned_h2d_staging_enabled",
                "CUDA shard run uses cuda-graph feature path: reusable pinned H2D staging and preallocated device tensors are on; current path is single-buffered and waits before compute, so Nsight is still required to prove copy/compute overlap",
            ));
        } else {
            advisories.push(RuntimeAdvisory::warning(
                "cuda_shards_without_pinned_async_h2d",
                "CUDA shard run is semantically valid but built without cuda-graph; reusable pinned H2D staging and preallocated device tensors are off, so pageable materialization may limit throughput",
            ));
        }
        advisories.push(RuntimeAdvisory::info(
            "optimized_path_bc_shards",
            format!(
                "runtime path: input=bc_shards pinned_h2d={} prealloc_gpu_tensors={} cuda_graph_replay=production_off_probe_only copy_compute_overlap={}",
                if cuda_graph_feature_active() { "on" } else { "off" },
                if cuda_graph_feature_active() { "on" } else { "off" },
                if cuda_graph_feature_active() {
                    "unproven-single-buffer"
                } else {
                    "off"
                },
            ),
        ));
    }

    if is_cuda && config.bc_shards_manifest_path.is_some() {
        let configured_threads = config
            .num_threads
            .unwrap_or_else(default_num_threads_for_system);
        if configured_threads > 1 {
            advisories.push(RuntimeAdvisory::info(
                "num_threads_not_shard_parallelism",
                format!(
                    "num_threads={} tunes loose/archive replay loading; BC shard training uses one mmap prefetch producer, so raise throughput with larger stable microbatch/pinned transfer unless producer_wait dominates",
                    configured_threads
                ),
            ));
        }
    }
    if train_microbatch < config.batch_size {
        advisories.push(RuntimeAdvisory::warning(
            "small_microbatch_high_accumulation_overhead",
            format!(
                "microbatch_size={} with batch_size={} requires {} accumulation steps; larger stable microbatches may reduce CPU/framework overhead",
                train_microbatch,
                config.batch_size,
                config.batch_size.div_ceil(train_microbatch)
            ),
        ));
    } else {
        advisories.push(RuntimeAdvisory::info(
            "full_train_microbatch_enabled",
            format!(
                "microbatch_size={} matches batch_size={}; gradient accumulation overhead is off",
                train_microbatch, config.batch_size
            ),
        ));
    }

    if explicitness.any() {
        advisories.push(RuntimeAdvisory::warning(
            "explicit_microbatch_blocks_faster_candidate_search",
            format!(
                "explicit microbatch settings are active (train_explicit={} effective_train={} validation_explicit={} effective_validation={}); automatic faster-candidate override is off unless allow_override_explicit_microbatch=true",
                explicitness.train, train_microbatch, explicitness.validation, validation_microbatch
            ),
        ));
    }

    if config.log_every_n_steps == 1 {
        advisories.push(RuntimeAdvisory::warning(
            "logging_or_metric_sync_overhead",
            "log_every_n_steps=1 records every optimizer step; metric readback/log formatting can reduce CUDA throughput",
        ));
    }

    if config.validate_every_n_steps == 1 || config.checkpoint_every_n_steps == 1 {
        advisories.push(RuntimeAdvisory::warning(
            "validation_or_checkpoint_cadence_overhead",
            format!(
                "validate_every_n_steps={} checkpoint_every_n_steps={} may spend most wall time on safety cadence instead of training throughput",
                config.validate_every_n_steps, config.checkpoint_every_n_steps
            ),
        ));
    }

    advisories
}
const SELECTED_RUNTIME_SLOWER_RATIO: f64 = 0.80;

pub(super) fn selected_runtime_probe_advisories(
    kind: ProbeKind,
    selected_candidate: usize,
    results: &[ProbeResult],
) -> Vec<RuntimeAdvisory> {
    let Some(best) = best_probe_summary(results) else {
        return Vec::new();
    };
    if best.candidate_microbatch == selected_candidate {
        return Vec::new();
    }
    let Some(best_average) = best.average_samples_per_second else {
        return Vec::new();
    };
    let Some(selected_average) = candidate_average(results, selected_candidate) else {
        return Vec::new();
    };
    if best_average <= f64::EPSILON || selected_average <= f64::EPSILON {
        return Vec::new();
    }
    if selected_average >= best_average * SELECTED_RUNTIME_SLOWER_RATIO {
        return Vec::new();
    }

    let kind_name = probe_kind_name(kind);
    let key = match kind {
        ProbeKind::Train => "selected_train_runtime_slower_than_best_probe_candidate",
        ProbeKind::Validation => "selected_validation_runtime_slower_than_best_probe_candidate",
        ProbeKind::RlGames => "selected_rl_games_runtime_slower_than_best_probe_candidate",
        ProbeKind::RlMicrobatch => {
            "selected_rl_microbatch_runtime_slower_than_best_probe_candidate"
        }
    };
    let slower_percent = ((best_average - selected_average) / best_average) * 100.0;
    vec![RuntimeAdvisory::warning(
        key,
        format!(
            "selected {kind_name} microbatch={} averaged {:.2} samples/s, below best stable measured {kind_name} microbatch={} at {:.2} samples/s ({:.1}% slower); explicit settings or conservative stability selection may be limiting throughput",
            selected_candidate,
            selected_average,
            best.candidate_microbatch,
            best_average,
            slower_percent
        ),
    )]
}

#[derive(Debug, Clone, Copy)]
pub(super) struct IntervalTimingInput {
    pub(super) producer_wait_seconds: f64,
    pub(super) collation_seconds: f64,
    pub(super) h2d_transfer_seconds: f64,
    pub(super) h2d_pageable_to_pinned_seconds: f64,
    pub(super) h2d_tensor_materialize_seconds: f64,
    pub(super) h2d_stream_sync_seconds: f64,
    pub(super) metric_readback_seconds: f64,
    pub(super) forward_seconds: f64,
    pub(super) backward_seconds: f64,
    pub(super) optimizer_step_seconds: f64,
    pub(super) validation_seconds: f64,
    pub(super) checkpoint_seconds: f64,
    pub(super) logging_seconds: f64,
    pub(super) kernel_launch_count: Option<u64>,
    pub(super) tiny_kernel_fraction: Option<f64>,
    pub(super) cuda_runtime_launch_seconds: Option<f64>,
    pub(super) total_seconds: f64,
    pub(super) steps: usize,
    pub(super) is_cuda: bool,
}

pub(super) fn interval_runtime_advisories(input: IntervalTimingInput) -> Vec<RuntimeAdvisory> {
    if input.steps < 2 || input.total_seconds <= f64::EPSILON {
        return Vec::new();
    }

    let pct = |seconds: f64| seconds.max(0.0) / input.total_seconds;
    let mut advisories = Vec::new();

    let cpu_input_seconds = input.producer_wait_seconds + input.collation_seconds;
    if input.is_cuda && pct(cpu_input_seconds) >= 0.25 {
        advisories.push(RuntimeAdvisory::warning(
            "cpu_producer_lag",
            format!(
                "CPU input used {:.1}% of interval wall time (producer_wait={:.3}s collation={:.3}s); GPU may be waiting on producer/collation",
                pct(cpu_input_seconds) * 100.0,
                input.producer_wait_seconds,
                input.collation_seconds,
            ),
        ));
    }

    let h2d_substage_seconds = input.h2d_pageable_to_pinned_seconds
        + input.h2d_tensor_materialize_seconds
        + input.h2d_stream_sync_seconds;
    let h2d_framework_seconds =
        input.h2d_tensor_materialize_seconds + input.h2d_stream_sync_seconds;
    if input.is_cuda
        && pct(input.h2d_transfer_seconds) >= 0.10
        && h2d_substage_seconds > f64::EPSILON
        && h2d_framework_seconds / input.h2d_transfer_seconds.max(f64::EPSILON) >= 0.50
    {
        advisories.push(RuntimeAdvisory::warning(
            "h2d_framework_materialization_overhead",
            format!(
                "H2D stage used {:.1}% of interval wall time and is mostly tensor materialization/sync (h2d={:.3}s pageable_to_pinned={:.3}s tensor_materialize={:.3}s stream_sync={:.3}s); raw PCIe copy may not be the bottleneck",
                pct(input.h2d_transfer_seconds) * 100.0,
                input.h2d_transfer_seconds,
                input.h2d_pageable_to_pinned_seconds,
                input.h2d_tensor_materialize_seconds,
                input.h2d_stream_sync_seconds,
            ),
        ));
    }

    if input.is_cuda && pct(input.metric_readback_seconds + input.logging_seconds) >= 0.10 {
        advisories.push(RuntimeAdvisory::warning(
            "logging_or_metric_sync_overhead",
            format!(
                "metric readback/logging used {:.1}% of interval wall time (metric_readback={:.3}s logging={:.3}s); consider less frequent logging for throughput runs",
                pct(input.metric_readback_seconds + input.logging_seconds) * 100.0,
                input.metric_readback_seconds,
                input.logging_seconds
            ),
        ));
    }

    if pct(input.validation_seconds + input.checkpoint_seconds) >= 0.20 {
        advisories.push(RuntimeAdvisory::warning(
            "validation_or_checkpoint_cadence_overhead",
            format!(
                "validation/checkpoint used {:.1}% of interval wall time (validation={:.3}s checkpoint={:.3}s); safety cadence may dominate short runs",
                pct(input.validation_seconds + input.checkpoint_seconds) * 100.0,
                input.validation_seconds,
                input.checkpoint_seconds
            ),
        ));
    }

    if input.is_cuda && pct(input.optimizer_step_seconds) >= 0.25 {
        advisories.push(RuntimeAdvisory::warning(
            "optimizer_step_dominates_cuda_interval",
            format!(
                "optimizer_step used {:.1}% of interval wall time ({:.3}s over {} steps); Burn Adam is unfused/per-parameter on tch, so fused optimizer or graph-safe optimizer capture is the likely next optimization lane",
                pct(input.optimizer_step_seconds) * 100.0,
                input.optimizer_step_seconds,
                input.steps,
            ),
        ));
    }

    if input.is_cuda && pct(input.forward_seconds + input.backward_seconds) >= 0.45 {
        advisories.push(RuntimeAdvisory::info(
            "model_compute_dominates_cuda_interval",
            format!(
                "forward+backward used {:.1}% of interval wall time (forward={:.3}s backward={:.3}s); optimize model kernels or graph-capture compute only after static-input parity proof",
                pct(input.forward_seconds + input.backward_seconds) * 100.0,
                input.forward_seconds,
                input.backward_seconds,
            ),
        ));
    }

    if let Some(launch_count) = input.kernel_launch_count {
        let tiny_fraction = input.tiny_kernel_fraction.unwrap_or(0.0);
        if input.is_cuda && launch_count >= 100_000 && tiny_fraction >= 0.50 {
            let launch_seconds = input.cuda_runtime_launch_seconds.unwrap_or(0.0);
            advisories.push(RuntimeAdvisory::warning(
                "cuda_launch_fragmentation_overhead",
                format!(
                    "CUDA trace shows {} kernel launches with {:.1}% tiny kernels and {:.3}s launch API time; backend op fusion or CUDA graph capture may improve throughput without changing BC targets/losses",
                    launch_count,
                    tiny_fraction * 100.0,
                    launch_seconds,
                ),
            ));
        }
    }

    advisories
}

#[cfg(feature = "cuda-graph")]
fn cuda_graph_feature_active() -> bool {
    true
}

#[cfg(not(feature = "cuda-graph"))]
fn cuda_graph_feature_active() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::dummy_train_config;
    use hydra_train::preflight::ProbeStatus;
    use std::path::PathBuf;

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

    fn probe_result(
        kind: ProbeKind,
        candidate: usize,
        samples_per_second: Option<f64>,
    ) -> ProbeResult {
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
            AdvisorySeverity::Warning
        );
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
            AdvisorySeverity::Warning
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
            AdvisorySeverity::Warning
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
            AdvisorySeverity::Warning
        );
        assert_eq!(
            advisory(
                &advisories,
                "explicit_microbatch_blocks_faster_candidate_search"
            )
            .severity,
            AdvisorySeverity::Warning
        );
        assert_eq!(
            advisory(&advisories, "logging_or_metric_sync_overhead").severity,
            AdvisorySeverity::Warning
        );
        assert_eq!(
            advisory(&advisories, "validation_or_checkpoint_cadence_overhead").severity,
            AdvisorySeverity::Warning
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
        assert!(advisory.message.contains("CUDA graph capture"));
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
        let json =
            serde_json::to_value(AdvisoryEvent::interval(&advisories)).expect("serialize event");

        assert_eq!(json["event"], "runtime_advisories");
        assert_eq!(json["scope"], "interval");
        assert_eq!(
            json["advisories"][0]["key"],
            "validation_or_checkpoint_cadence_overhead"
        );
    }
}
