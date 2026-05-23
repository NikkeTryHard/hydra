#![allow(
    missing_docs,
    reason = "moved train execution support preserves existing public surface"
)]

use std::collections::BTreeSet;

use hydra_train_runtime::preflight::{ProbeKind, ProbeResult};
use serde::Serialize;

use super::probe_summary::{best_probe_summary, candidate_average, probe_kind_name};
use hydra_train_runtime::config::TrainConfig;
use hydra_train_runtime::config::{
    default_num_threads_for_system, train_microbatch_size, validation_microbatch_size,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdvisorySeverity {
    Info,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeAdvisory {
    pub key: &'static str,
    pub severity: AdvisorySeverity,
    pub message: String,
}

impl RuntimeAdvisory {
    pub fn info(key: &'static str, message: impl Into<String>) -> Self {
        Self {
            key,
            severity: AdvisorySeverity::Info,
            message: message.into(),
        }
    }

    pub fn warning(key: &'static str, message: impl Into<String>) -> Self {
        Self {
            key,
            severity: AdvisorySeverity::Warning,
            message: message.into(),
        }
    }
}

#[derive(Debug, Default)]
pub struct AdvisoryDeduper {
    seen: BTreeSet<&'static str>,
}

impl AdvisoryDeduper {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn retain_new(&mut self, advisories: Vec<RuntimeAdvisory>) -> Vec<RuntimeAdvisory> {
        advisories
            .into_iter()
            .filter(|advisory| self.seen.insert(advisory.key))
            .collect()
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MicrobatchExplicitness {
    pub train: bool,
    pub validation: bool,
}

impl MicrobatchExplicitness {
    pub fn from_config(config: &TrainConfig) -> Self {
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
pub enum AdvisoryEventScope {
    Startup,
    Interval,
}

#[derive(Debug, Serialize)]
pub struct AdvisoryEvent<'a> {
    pub event: &'static str,
    pub scope: AdvisoryEventScope,
    pub advisories: &'a [RuntimeAdvisory],
}

impl<'a> AdvisoryEvent<'a> {
    pub fn startup(advisories: &'a [RuntimeAdvisory]) -> Self {
        Self {
            event: "runtime_advisories",
            scope: AdvisoryEventScope::Startup,
            advisories,
        }
    }

    pub fn interval(advisories: &'a [RuntimeAdvisory]) -> Self {
        Self {
            event: "runtime_advisories",
            scope: AdvisoryEventScope::Interval,
            advisories,
        }
    }
}

fn cuda_device_count() -> i64 {
    0
}

pub fn startup_runtime_advisories(
    config: &TrainConfig,
    explicitness: MicrobatchExplicitness,
) -> Vec<RuntimeAdvisory> {
    let device = config.device.trim().to_ascii_lowercase();
    let is_cuda = device == "cuda" || device.starts_with("cuda:");
    let train_microbatch = train_microbatch_size(config);
    let validation_microbatch = validation_microbatch_size(config);
    let mut advisories = Vec::new();

    if !is_cuda {
        let cuda_device_count = cuda_device_count();
        if cuda_device_count > 0 {
            advisories.push(RuntimeAdvisory::warning(
                "cpu_device_with_cuda_available",
                format!(
                    "device is CPU but {cuda_device_count} CUDA device(s) were detected; CPU training is super slow and should be used only when intentionally debugging CPU behavior. GPU mode accelerates model forward/backward/optimizer/H2D only; raw replay/shard materialization still runs on CPU workers."
                ),
            ));
        } else {
            advisories.push(RuntimeAdvisory::info(
                "cpu_device_for_training",
                "device is CPU; CUDA feeding optimizations and pinned H2D staging are off",
            ));
        }
    }

    if is_cuda && config.bc_shards_manifest_path.is_none() {
        advisories.push(RuntimeAdvisory::info(
            "steady_state_cuda_bc_uses_loose_replay",
            "CUDA BC run uses loose/archive replay input; for steady-state throughput build BC shards and set bc_shards_manifest_path",
        ));
        advisories.push(RuntimeAdvisory::info(
            "optimized_path_raw_replay",
            format!(
                "runtime path: input=raw_replay pinned_h2d={} prealloc_gpu_tensors={} cuda_graph_replay=experimental_probe_only copy_compute_overlap={}",
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
        advisories.push(RuntimeAdvisory::info(
            "steady_state_cuda_bc_shards_enabled",
            "CUDA BC shard input is on; replay parsing/encoding is out of the steady-state training hot path",
        ));

        if cuda_graph_feature_active() {
            advisories.push(RuntimeAdvisory::info(
                "cuda_shards_pinned_h2d_staging_enabled",
                "CUDA shard run uses reusable pinned H2D staging and preallocated device tensors; current path is single-buffered and waits before compute, so Nsight is still required to prove copy/compute overlap",
            ));
        } else {
            advisories.push(RuntimeAdvisory::info(
                "cuda_shards_without_pinned_async_h2d",
                "CUDA shard run is semantically valid but built without cuda-graph; reusable pinned H2D staging and preallocated device tensors are off, so pageable materialization may limit throughput",
            ));
        }
        advisories.push(RuntimeAdvisory::info(
            "optimized_path_bc_shards",
            format!(
                "runtime path: input=bc_shards pinned_h2d={} prealloc_gpu_tensors={} cuda_graph_replay=experimental_probe_only copy_compute_overlap={}",
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
        advisories.push(RuntimeAdvisory::info(
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
        advisories.push(RuntimeAdvisory::info(
            "explicit_microbatch_blocks_faster_candidate_search",
            format!(
                "explicit microbatch settings are active (train_explicit={} effective_train={} validation_explicit={} effective_validation={}); automatic faster-candidate override is off unless allow_override_explicit_microbatch=true",
                explicitness.train, train_microbatch, explicitness.validation, validation_microbatch
            ),
        ));
    }

    if config.log_every_n_steps == 1 {
        advisories.push(RuntimeAdvisory::info(
            "logging_or_metric_sync_overhead",
            "log_every_n_steps=1 records every optimizer step; metric readback/log formatting can reduce CUDA throughput",
        ));
    }

    if config.validate_every_n_steps == 1 || config.checkpoint_every_n_steps == 1 {
        advisories.push(RuntimeAdvisory::info(
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

pub fn selected_runtime_probe_advisories(
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
pub struct IntervalTimingInput {
    pub producer_wait_seconds: f64,
    pub collation_seconds: f64,
    pub h2d_transfer_seconds: f64,
    pub h2d_pageable_to_pinned_seconds: f64,
    pub h2d_tensor_materialize_seconds: f64,
    pub h2d_stream_sync_seconds: f64,
    pub metric_readback_seconds: f64,
    pub forward_seconds: f64,
    pub backward_seconds: f64,
    pub optimizer_step_seconds: f64,
    pub validation_seconds: f64,
    pub checkpoint_seconds: f64,
    pub logging_seconds: f64,
    pub kernel_launch_count: Option<u64>,
    pub tiny_kernel_fraction: Option<f64>,
    pub cuda_runtime_launch_seconds: Option<f64>,
    pub total_seconds: f64,
    pub steps: usize,
    pub is_cuda: bool,
}

pub fn interval_runtime_advisories(input: IntervalTimingInput) -> Vec<RuntimeAdvisory> {
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
                "optimizer_step used {:.1}% of interval wall time ({:.3}s over {} steps); the optimizer dominates this interval, so fused optimizer or graph-safe optimizer capture is the likely next optimization lane",
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
                "forward+backward used {:.1}% of interval wall time (forward={:.3}s backward={:.3}s); optimize model kernels or use the CUDA graph probe only after static-input parity proof",
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
                    "CUDA trace shows {} kernel launches with {:.1}% tiny kernels and {:.3}s launch API time; backend op fusion may improve throughput without changing BC targets/losses; CUDA graph replay remains experimental/probe-only",
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
mod tests;
