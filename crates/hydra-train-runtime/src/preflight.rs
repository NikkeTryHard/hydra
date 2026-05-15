use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PreflightTuningMode {
    #[default]
    Safe,
    Unsafe,
}

pub fn default_preflight_tuning_mode() -> PreflightTuningMode {
    PreflightTuningMode::Safe
}

pub fn default_allow_override_explicit_microbatch() -> bool {
    false
}

pub fn default_fast_repeated_run_profile() -> bool {
    false
}

pub fn default_warmup_steps() -> usize {
    2
}

pub fn default_measure_steps() -> usize {
    2
}

pub fn default_required_successes() -> usize {
    2
}

pub fn default_candidate_microbatches() -> Vec<usize> {
    vec![
        512, 384, 320, 288, 256, 224, 192, 160, 144, 128, 112, 104, 96, 80, 72, 64, 48, 32, 24, 16,
    ]
}

pub fn default_unsafe_candidate_batch_sizes() -> Vec<usize> {
    Vec::new()
}

pub fn default_unsafe_candidate_lr_scales() -> Vec<f64> {
    Vec::new()
}

pub fn default_unsafe_candidate_warmup_steps() -> Vec<usize> {
    Vec::new()
}

pub fn default_fast_repeated_run_candidate_window() -> usize {
    1
}

pub fn default_min_microbatch_size() -> usize {
    16
}

pub fn default_validation_growth_patience() -> usize {
    2
}

pub fn default_validation_growth_max_steps() -> usize {
    6
}

pub fn default_measure_noise_tolerance_ratio() -> f64 {
    0.02
}

pub fn default_loader_runtime_rounds() -> usize {
    2
}

pub fn default_real_benchmark_enabled() -> bool {
    true
}

pub fn default_real_benchmark_train_candidates() -> usize {
    4
}

pub fn default_real_benchmark_validation_candidates() -> usize {
    4
}

pub fn default_real_benchmark_loader_candidates() -> usize {
    3
}

pub fn default_real_benchmark_max_finalists() -> usize {
    8
}

pub fn default_real_benchmark_warmup_steps() -> usize {
    8
}

pub fn default_real_benchmark_train_steps() -> usize {
    64
}

pub fn default_real_benchmark_tie_margin_ratio() -> f64 {
    0.02
}

pub fn default_real_benchmark_extra_finalists() -> usize {
    2
}

pub fn default_finalist_margin_ratio() -> f64 {
    0.05
}

pub fn default_finalist_max_candidates() -> usize {
    2
}

pub fn default_finalist_extra_measure_steps() -> usize {
    3
}

pub fn default_finalist_extra_successes() -> usize {
    1
}

pub fn default_loader_tuple_margin_ratio() -> f64 {
    0.01
}

pub fn default_loader_tuple_extra_samples() -> usize {
    2
}

pub fn default_target_warmup_seconds() -> f64 {
    6.0
}

pub fn default_target_measure_seconds() -> f64 {
    12.0
}

pub fn default_max_adaptive_warmup_steps() -> usize {
    6
}

pub fn default_max_adaptive_measure_steps() -> usize {
    8
}

pub fn default_local_refinement_enabled() -> bool {
    true
}

pub fn default_local_refinement_max_candidates() -> usize {
    3
}

pub fn default_local_refinement_min_gap() -> usize {
    8
}

pub fn default_local_refinement_extra_measure_steps() -> usize {
    2
}

pub fn default_search_coordinate_rounds() -> usize {
    2
}

pub fn default_search_top_k() -> usize {
    3
}

pub fn default_rl_probe_min_free_memory_bytes() -> u64 {
    0
}

pub fn default_rl_probe_memory_headroom_ratio() -> f64 {
    0.0
}

pub fn default_rl_probe_growth_safety_factor() -> f64 {
    1.35
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PreflightConfig {
    #[serde(default = "default_preflight_tuning_mode")]
    pub tuning_mode: PreflightTuningMode,
    #[serde(default = "default_fast_repeated_run_profile")]
    pub fast_repeated_run_profile: bool,
    #[serde(default = "default_allow_override_explicit_microbatch")]
    pub allow_override_explicit_microbatch: bool,
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: usize,
    #[serde(default = "default_measure_steps")]
    pub measure_steps: usize,
    #[serde(default = "default_required_successes")]
    pub required_successes: usize,
    #[serde(default = "default_min_microbatch_size")]
    pub min_microbatch_size: usize,
    #[serde(default = "default_candidate_microbatches")]
    pub candidate_microbatches: Vec<usize>,
    #[serde(default = "default_unsafe_candidate_batch_sizes")]
    pub unsafe_candidate_batch_sizes: Vec<usize>,
    #[serde(default = "default_unsafe_candidate_lr_scales")]
    pub unsafe_candidate_lr_scales: Vec<f64>,
    #[serde(default = "default_unsafe_candidate_warmup_steps")]
    pub unsafe_candidate_warmup_steps: Vec<usize>,
    #[serde(default = "default_fast_repeated_run_candidate_window")]
    pub fast_repeated_run_candidate_window: usize,
    #[serde(default = "default_validation_growth_patience")]
    pub validation_growth_patience: usize,
    #[serde(default = "default_validation_growth_max_steps")]
    pub validation_growth_max_steps: usize,
    #[serde(default = "default_measure_noise_tolerance_ratio")]
    pub measure_noise_tolerance_ratio: f64,
    #[serde(default = "default_loader_runtime_rounds")]
    pub loader_runtime_rounds: usize,
    #[serde(default = "default_real_benchmark_enabled")]
    pub real_benchmark_enabled: bool,
    #[serde(default = "default_real_benchmark_train_candidates")]
    pub real_benchmark_train_candidates: usize,
    #[serde(default = "default_real_benchmark_validation_candidates")]
    pub real_benchmark_validation_candidates: usize,
    #[serde(default = "default_real_benchmark_loader_candidates")]
    pub real_benchmark_loader_candidates: usize,
    #[serde(default = "default_real_benchmark_max_finalists")]
    pub real_benchmark_max_finalists: usize,
    #[serde(default = "default_real_benchmark_warmup_steps")]
    pub real_benchmark_warmup_steps: usize,
    #[serde(default = "default_real_benchmark_train_steps")]
    pub real_benchmark_train_steps: usize,
    #[serde(default = "default_real_benchmark_tie_margin_ratio")]
    pub real_benchmark_tie_margin_ratio: f64,
    #[serde(default = "default_real_benchmark_extra_finalists")]
    pub real_benchmark_extra_finalists: usize,
    #[serde(default = "default_finalist_margin_ratio")]
    pub finalist_margin_ratio: f64,
    #[serde(default = "default_finalist_max_candidates")]
    pub finalist_max_candidates: usize,
    #[serde(default = "default_finalist_extra_measure_steps")]
    pub finalist_extra_measure_steps: usize,
    #[serde(default = "default_finalist_extra_successes")]
    pub finalist_extra_successes: usize,
    #[serde(default = "default_loader_tuple_margin_ratio")]
    pub loader_tuple_margin_ratio: f64,
    #[serde(default = "default_loader_tuple_extra_samples")]
    pub loader_tuple_extra_samples: usize,
    #[serde(default = "default_target_warmup_seconds")]
    pub target_warmup_seconds: f64,
    #[serde(default = "default_target_measure_seconds")]
    pub target_measure_seconds: f64,
    #[serde(default = "default_max_adaptive_warmup_steps")]
    pub max_adaptive_warmup_steps: usize,
    #[serde(default = "default_max_adaptive_measure_steps")]
    pub max_adaptive_measure_steps: usize,
    #[serde(default = "default_local_refinement_enabled")]
    pub local_refinement_enabled: bool,
    #[serde(default = "default_local_refinement_max_candidates")]
    pub local_refinement_max_candidates: usize,
    #[serde(default = "default_local_refinement_min_gap")]
    pub local_refinement_min_gap: usize,
    #[serde(default = "default_local_refinement_extra_measure_steps")]
    pub local_refinement_extra_measure_steps: usize,
    #[serde(default = "default_search_coordinate_rounds")]
    pub search_coordinate_rounds: usize,
    #[serde(default = "default_search_top_k")]
    pub search_top_k: usize,
    #[serde(default = "default_rl_probe_min_free_memory_bytes")]
    pub rl_probe_min_free_memory_bytes: u64,
    #[serde(default = "default_rl_probe_memory_headroom_ratio")]
    pub rl_probe_memory_headroom_ratio: f64,
    #[serde(default = "default_rl_probe_growth_safety_factor")]
    pub rl_probe_growth_safety_factor: f64,
}

impl Default for PreflightConfig {
    fn default() -> Self {
        Self {
            tuning_mode: default_preflight_tuning_mode(),
            allow_override_explicit_microbatch: default_allow_override_explicit_microbatch(),
            fast_repeated_run_profile: default_fast_repeated_run_profile(),
            warmup_steps: default_warmup_steps(),
            measure_steps: default_measure_steps(),
            required_successes: default_required_successes(),
            min_microbatch_size: default_min_microbatch_size(),
            candidate_microbatches: default_candidate_microbatches(),
            unsafe_candidate_batch_sizes: default_unsafe_candidate_batch_sizes(),
            unsafe_candidate_lr_scales: default_unsafe_candidate_lr_scales(),
            unsafe_candidate_warmup_steps: default_unsafe_candidate_warmup_steps(),
            fast_repeated_run_candidate_window: default_fast_repeated_run_candidate_window(),
            validation_growth_patience: default_validation_growth_patience(),
            validation_growth_max_steps: default_validation_growth_max_steps(),
            measure_noise_tolerance_ratio: default_measure_noise_tolerance_ratio(),
            loader_runtime_rounds: default_loader_runtime_rounds(),
            real_benchmark_enabled: default_real_benchmark_enabled(),
            real_benchmark_train_candidates: default_real_benchmark_train_candidates(),
            real_benchmark_validation_candidates: default_real_benchmark_validation_candidates(),
            real_benchmark_loader_candidates: default_real_benchmark_loader_candidates(),
            real_benchmark_max_finalists: default_real_benchmark_max_finalists(),
            real_benchmark_warmup_steps: default_real_benchmark_warmup_steps(),
            real_benchmark_train_steps: default_real_benchmark_train_steps(),
            real_benchmark_tie_margin_ratio: default_real_benchmark_tie_margin_ratio(),
            real_benchmark_extra_finalists: default_real_benchmark_extra_finalists(),
            finalist_margin_ratio: default_finalist_margin_ratio(),
            finalist_max_candidates: default_finalist_max_candidates(),
            finalist_extra_measure_steps: default_finalist_extra_measure_steps(),
            finalist_extra_successes: default_finalist_extra_successes(),
            loader_tuple_margin_ratio: default_loader_tuple_margin_ratio(),
            loader_tuple_extra_samples: default_loader_tuple_extra_samples(),
            target_warmup_seconds: default_target_warmup_seconds(),
            target_measure_seconds: default_target_measure_seconds(),
            max_adaptive_warmup_steps: default_max_adaptive_warmup_steps(),
            max_adaptive_measure_steps: default_max_adaptive_measure_steps(),
            local_refinement_enabled: default_local_refinement_enabled(),
            local_refinement_max_candidates: default_local_refinement_max_candidates(),
            local_refinement_min_gap: default_local_refinement_min_gap(),
            local_refinement_extra_measure_steps: default_local_refinement_extra_measure_steps(),
            search_coordinate_rounds: default_search_coordinate_rounds(),
            search_top_k: default_search_top_k(),
            rl_probe_min_free_memory_bytes: default_rl_probe_min_free_memory_bytes(),
            rl_probe_memory_headroom_ratio: default_rl_probe_memory_headroom_ratio(),
            rl_probe_growth_safety_factor: default_rl_probe_growth_safety_factor(),
        }
    }
}

fn total_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kb = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kb.saturating_mul(1024))
}

/// Returns the stable precision-mode fragment used in preflight cache keys.
pub fn precision_mode_signature(mode: crate::config::PrecisionMode) -> String {
    match mode {
        crate::config::PrecisionMode::Fp32 => "fp32".to_string(),
        crate::config::PrecisionMode::Bf16Autocast => "bf16_autocast".to_string(),
    }
}

/// Returns the advanced-loss fragment used in preflight cache keys.
pub fn advanced_loss_signature(config: Option<&crate::config::AdvancedLossConfig>) -> String {
    match config {
        Some(config) => serde_json::to_string(config)
            .unwrap_or_else(|_| "advanced_loss:unserializable".to_string()),
        None => "advanced_loss:none".to_string(),
    }
}

/// Returns the preflight-config fragment used in preflight cache keys.
pub fn preflight_config_signature(config: &crate::config::TrainConfig) -> String {
    serde_json::to_string(&config.preflight)
        .unwrap_or_else(|_| "preflight_config:unserializable".to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFingerprintInput {
    pub num_blocks: usize,
    pub input_channels: usize,
    pub hidden_channels: usize,
    pub num_groups: usize,
    pub action_space: usize,
    pub score_bins: usize,
}

impl ModelFingerprintInput {
    pub fn signature(&self) -> String {
        format!(
            "blocks:{} input:{} hidden:{} groups:{} action:{} score_bins:{}",
            self.num_blocks,
            self.input_channels,
            self.hidden_channels,
            self.num_groups,
            self.action_space,
            self.score_bins,
        )
    }
}

/// Builds the workload fingerprint portion of the preflight cache key.
pub fn workload_fingerprint(
    config: &crate::config::TrainConfig,
    model: &ModelFingerprintInput,
) -> WorkloadFingerprint {
    WorkloadFingerprint {
        batch_size: config.batch_size,
        augment: config.augment,
        precision_mode: precision_mode_signature(config.precision_mode),
        train_fraction_bits: config.train_fraction.to_bits(),
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        max_validation_batches: config.max_validation_batches,
        max_validation_samples: config.max_validation_samples,
        model_signature: model.signature(),
        code_signature: format!(
            "hydra-train:{}:{}:preflight-v4",
            env!("CARGO_PKG_VERSION"),
            env!("CARGO_PKG_NAME")
        ),
        advanced_loss_signature: advanced_loss_signature(config.advanced_loss.as_ref()),
        preflight_config_signature: preflight_config_signature(config),
        explicit_train_microbatch: config.microbatch_size,
        explicit_validation_microbatch: config.validation_microbatch_size,
    }
}

/// Builds the hardware fingerprint portion of the preflight cache key.
pub fn hardware_fingerprint(device_label: &str, cpu_logical_cores: usize) -> HardwareFingerprint {
    HardwareFingerprint {
        device_label: device_label.to_string(),
        backend: "burn-libtorch".to_string(),
        cpu_logical_cores,
        total_memory_bytes: total_memory_bytes(),
    }
}

/// Builds the complete preflight cache key.
pub fn preflight_cache_key(
    config: &crate::config::TrainConfig,
    model: &ModelFingerprintInput,
    device_label: &str,
    cpu_logical_cores: usize,
) -> PreflightCacheKey {
    PreflightCacheKey {
        hardware: hardware_fingerprint(device_label, cpu_logical_cores),
        workload: workload_fingerprint(config, model),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareFingerprint {
    pub device_label: String,
    pub backend: String,
    pub cpu_logical_cores: usize,
    pub total_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkloadFingerprint {
    pub batch_size: usize,
    pub augment: bool,
    pub precision_mode: String,
    pub train_fraction_bits: u32,
    pub max_skip_logs_per_source: usize,
    pub max_validation_batches: Option<usize>,
    pub max_validation_samples: Option<usize>,
    pub model_signature: String,
    pub code_signature: String,
    pub advanced_loss_signature: String,
    /// Serialized PreflightConfig capturing all probe/search knobs that affect
    /// which runtime gets selected. Any knob change invalidates the cache.
    #[serde(default)]
    pub preflight_config_signature: String,
    /// Explicit train microbatch override from TrainConfig, if set. Constrains
    /// the probe search space so a change should invalidate the cache.
    #[serde(default)]
    pub explicit_train_microbatch: Option<usize>,
    /// Explicit validation microbatch override from TrainConfig, if set.
    #[serde(default)]
    pub explicit_validation_microbatch: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreflightCacheKey {
    pub hardware: HardwareFingerprint,
    pub workload: WorkloadFingerprint,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProbeStatus {
    Success,
    Oom,
    BackendError,
    DataError,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProbeKind {
    Train,
    Validation,
    RlGames,
    RlMicrobatch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeResult {
    pub kind: ProbeKind,
    pub candidate_microbatch: usize,
    pub status: ProbeStatus,
    pub measured_samples_per_second: Option<f64>,
    pub elapsed_seconds: Option<f64>,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SelectedRuntimeConfig {
    pub train_microbatch_size: usize,
    pub validation_microbatch_size: usize,
    pub accum_steps: usize,
    #[serde(default)]
    pub unsafe_selected_batch_size: Option<usize>,
    #[serde(default)]
    pub unsafe_selected_learning_rate: Option<f64>,
    #[serde(default)]
    pub unsafe_selected_min_learning_rate: Option<f64>,
    #[serde(default)]
    pub unsafe_selected_warmup_steps: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoaderRuntimeConfig {
    pub num_threads: Option<usize>,
    pub buffer_games: usize,
    pub buffer_samples: usize,
    pub archive_queue_bound: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct EffectiveRuntimeConfig {
    pub selected: SelectedRuntimeConfig,
    pub loader: LoaderRuntimeConfig,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkRuntimeConfig {
    pub train_microbatch_size: usize,
    pub validation_microbatch_size: usize,
    pub accum_steps: usize,
    pub loader: LoaderRuntimeConfig,
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub min_learning_rate: Option<f64>,
    #[serde(default)]
    pub warmup_steps: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SystemMetricEventKind {
    #[default]
    ProbeHostMemory,
    ProbeChildInit,
    ResourceSnapshot,
    Progress,
    PipelineStage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SystemMetricsEvent {
    pub kind: SystemMetricEventKind,
    pub probe_kind: Option<ProbeKind>,
    pub candidate_microbatch: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planned: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_seconds: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process_rss_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_read_mb_per_sec: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_write_mb_per_sec: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_util_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_mem_used_mb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_mem_free_mb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_oom_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples_per_second: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_available_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_total_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_init_ms: Option<u128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_init_ms: Option<u128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_init_ms: Option<u128>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMode {
    #[default]
    NotRecorded,
    CadenceAwareProjection,
}

pub const PROFILING_STAGE_STAGE_2_BENCHMARK: &str = "stage_2_benchmark";
pub const PROFILING_STAGE_BC_INTERVAL: &str = "bc_interval";
pub const PROFILING_STAGE_BC_EPOCH: &str = "bc_epoch";
pub const PROFILING_STAGE_RL_STEP: &str = "rl_step";
pub const PROFILING_STAGE_TRAIN: &str = "train";
pub const PROFILING_STAGE_VALIDATION: &str = "validation";
pub const PROFILING_STAGE_CHECKPOINT: &str = "checkpoint";
pub const PROFILING_STAGE_LOGGING: &str = "logging";
pub const PROFILING_STAGE_SELF_PLAY: &str = "self_play";
pub const PROFILING_STAGE_CANDIDATE_FORWARD_AND_LOSS: &str = "candidate_forward_and_loss";
pub const PROFILING_STAGE_DELTA_Q_BASELINE_FORWARD: &str = "delta_q_baseline_forward";
pub const PROFILING_STAGE_COLLATION: &str = "collation";
pub const PROFILING_STAGE_FORWARD: &str = "forward";
pub const PROFILING_STAGE_LOSS: &str = "loss";
pub const PROFILING_STAGE_BACKWARD: &str = "backward";
pub const PROFILING_STAGE_OPTIMIZER_STEP: &str = "optimizer_step";
pub const PROFILING_STAGE_PRODUCER_WAIT: &str = "producer_wait";
pub const PROFILING_STAGE_H2D_TRANSFER: &str = "h2d_transfer";
pub const PROFILING_STAGE_H2D_PAGEABLE_TO_PINNED: &str = "h2d_pageable_to_pinned";
pub const PROFILING_STAGE_H2D_TENSOR_MATERIALIZE: &str = "h2d_tensor_materialize";
pub const PROFILING_STAGE_H2D_STREAM_SYNC: &str = "h2d_stream_sync";
pub const PROFILING_STAGE_METRIC_READBACK: &str = "metric_readback";
pub const PROFILING_STAGE_DATA_LOAD: &str = "data_load";
pub const PROFILING_STAGE_PREFLIGHT_MODEL_INIT: &str = "preflight_model_init";
pub const PROFILING_STAGE_PREFLIGHT_OPTIMIZER_INIT: &str = "preflight_optimizer_init";
pub const PROFILING_STAGE_PREFLIGHT_LOSS_INIT: &str = "preflight_loss_init";
pub const PROFILING_STAGE_PREFLIGHT_DATA_STREAM_INIT: &str = "preflight_data_stream_init";
pub const PROFILING_STAGE_PREFLIGHT_SHARD_LOAD: &str = "preflight_shard_load";
pub const PROFILING_STAGE_PREFLIGHT_CUDA_STAGING: &str = "preflight_cuda_staging";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ProfilingEnvelope {
    pub stage: String,
    pub elapsed_seconds: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<ProfilingEnvelope>,
}

impl ProfilingEnvelope {
    pub fn leaf(stage: impl Into<String>, elapsed_seconds: f64) -> Self {
        Self {
            stage: stage.into(),
            elapsed_seconds,
            children: Vec::new(),
        }
    }

    pub fn nested(
        stage: impl Into<String>,
        elapsed_seconds: f64,
        children: Vec<ProfilingEnvelope>,
    ) -> Self {
        Self {
            stage: stage.into(),
            elapsed_seconds,
            children,
        }
    }

    pub fn from_children(stage: impl Into<String>, children: Vec<ProfilingEnvelope>) -> Self {
        let elapsed_seconds = children.iter().map(|child| child.elapsed_seconds).sum();
        Self::nested(stage, elapsed_seconds, children)
    }

    pub fn merge_assign(&mut self, other: &ProfilingEnvelope) {
        if self.stage != other.stage {
            self.children.push(other.clone());
            self.elapsed_seconds += other.elapsed_seconds;
            return;
        }

        self.elapsed_seconds += other.elapsed_seconds;
        for child in &other.children {
            if let Some(existing) = self
                .children
                .iter_mut()
                .find(|existing| existing.stage == child.stage)
            {
                existing.merge_assign(child);
            } else {
                self.children.push(child.clone());
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct BenchmarkMetadata {
    #[serde(default)]
    pub mode: BenchmarkMode,
    #[serde(default)]
    pub selection_metric: String,
    #[serde(default)]
    pub train_probe_candidates_considered: usize,
    #[serde(default)]
    pub validation_probe_candidates_considered: usize,
    #[serde(default)]
    pub loader_candidates_considered: usize,
    #[serde(default)]
    pub finalists_benchmarked: usize,
    #[serde(default)]
    pub warmup_steps: usize,
    #[serde(default)]
    pub measured_train_steps: usize,
    #[serde(default)]
    pub projected_validation_events: f64,
    #[serde(default)]
    pub projected_checkpoint_events: f64,
    #[serde(default)]
    pub projected_logging_events: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkScore {
    pub wall_clock_samples_per_second: f64,
    pub train_only_samples_per_second: f64,
    pub train_seconds: f64,
    pub validation_seconds: f64,
    pub checkpoint_seconds: f64,
    pub logging_seconds: f64,
    pub total_elapsed_seconds: f64,
    pub train_steps: usize,
    pub validation_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkResult {
    pub runtime: BenchmarkRuntimeConfig,
    pub score: BenchmarkScore,
    #[serde(default)]
    pub metadata: BenchmarkMetadata,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profiling: Option<ProfilingEnvelope>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExplicitSettings {
    pub train_microbatch_explicit: bool,
    pub validation_microbatch_explicit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightCacheEntry {
    pub cache_key: PreflightCacheKey,
    pub runtime: EffectiveRuntimeConfig,
    #[serde(default)]
    pub benchmark: Option<BenchmarkResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreflightArtifactEventKind {
    Started,
    Completed,
    Skipped,
    Interrupted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightArtifactEvent {
    pub phase: String,
    pub kind: PreflightArtifactEventKind,
    pub elapsed_seconds: Option<f64>,
    pub detail: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub planned: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightCandidateRecord {
    pub phase: String,
    pub result: ProbeResult,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightCompletedPhase {
    pub phase: String,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightState {
    pub cache_key: PreflightCacheKey,
    pub completed_phases: Vec<PreflightCompletedPhase>,
    pub selected_runtime: Option<EffectiveRuntimeConfig>,
    pub cache_written: bool,
    #[serde(default)]
    pub candidate_records: Vec<PreflightCandidateRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightReport {
    pub cache_key: PreflightCacheKey,
    pub runtime: EffectiveRuntimeConfig,
    pub cache_hit: bool,
    pub train_probe_results: usize,
    pub validation_probe_results: usize,
    pub benchmark: Option<BenchmarkResult>,
    pub total_elapsed_seconds: f64,
}

pub fn candidate_ladder(config: &PreflightConfig, batch_size: usize) -> Vec<usize> {
    let mut candidates: Vec<usize> = config
        .candidate_microbatches
        .iter()
        .copied()
        .filter(|value| *value >= config.min_microbatch_size && *value <= batch_size)
        .collect();
    candidates.sort_unstable_by(|a, b| b.cmp(a));
    candidates.dedup();
    if candidates.is_empty() {
        candidates.push(batch_size.max(config.min_microbatch_size));
    }
    candidates
}

pub fn resolve_runtime_config(
    batch_size: usize,
    explicit: ExplicitSettings,
    train_microbatch: usize,
    validation_microbatch: usize,
) -> SelectedRuntimeConfig {
    let _ = explicit;
    let train_microbatch_size = train_microbatch.min(batch_size).max(1);
    SelectedRuntimeConfig {
        train_microbatch_size,
        validation_microbatch_size: validation_microbatch.max(1),
        accum_steps: batch_size.div_ceil(train_microbatch.max(1)).max(1),
        unsafe_selected_batch_size: None,
        unsafe_selected_learning_rate: None,
        unsafe_selected_min_learning_rate: None,
        unsafe_selected_warmup_steps: None,
    }
}

pub fn measure_samples_per_second(samples: usize, elapsed: std::time::Duration) -> f64 {
    if samples == 0 {
        return 0.0;
    }
    let seconds = elapsed.as_secs_f64();
    if seconds <= f64::EPSILON {
        0.0
    } else {
        samples as f64 / seconds
    }
}

pub fn classify_probe_detail(detail: &str) -> ProbeStatus {
    let lowered = detail.to_ascii_lowercase();
    if lowered.contains("out of memory") || lowered.contains("oom") {
        ProbeStatus::Oom
    } else if lowered.contains("cuda") || lowered.contains("cudnn") || lowered.contains("libtorch")
    {
        ProbeStatus::BackendError
    } else if lowered.contains("data") || lowered.contains("collate") || lowered.contains("replay")
    {
        ProbeStatus::DataError
    } else {
        ProbeStatus::BackendError
    }
}

pub fn format_probe_attempt_message(
    kind: ProbeKind,
    candidate: usize,
    attempt: usize,
    total_attempts: usize,
) -> String {
    format!(
        "[preflight:{}] candidate_mb={} attempt {}/{}",
        probe_kind_name(kind),
        candidate,
        attempt,
        total_attempts.max(1)
    )
}

pub fn probe_kind_name(kind: ProbeKind) -> &'static str {
    match kind {
        ProbeKind::Train => "train",
        ProbeKind::Validation => "validation",
        ProbeKind::RlGames => "rl_games",
        ProbeKind::RlMicrobatch => "rl_microbatch",
    }
}

pub fn format_probe_result_summary(result: &ProbeResult) -> String {
    let status = match result.status {
        ProbeStatus::Success => "success",
        ProbeStatus::Oom => "oom",
        ProbeStatus::BackendError => "backend_error",
        ProbeStatus::DataError => "data_error",
    };
    let throughput = result
        .measured_samples_per_second
        .map(|value| format!(" samples_per_second={value:.2}"))
        .unwrap_or_default();
    let elapsed = result
        .elapsed_seconds
        .map(|value| format!(" elapsed={value:.2}s"))
        .unwrap_or_default();
    format!(
        "[preflight:{}] candidate_mb={} status={}{}{} detail={}",
        probe_kind_name(result.kind),
        result.candidate_microbatch,
        status,
        throughput,
        elapsed,
        result.detail
    )
}
pub fn default_cache_name() -> PathBuf {
    PathBuf::from("preflight_cache.json")
}

pub fn default_manifest_cache_name() -> PathBuf {
    PathBuf::from("preflight_manifest.json")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestCacheEntry {
    pub data_dir: PathBuf,
    pub train_fraction_bits: u32,
    #[serde(default)]
    pub include_source_patterns: Vec<String>,
    #[serde(default)]
    pub exclude_source_patterns: Vec<String>,
    pub manifest: hydra_data_core::DataManifest,
}

#[cfg(test)]
mod tests;
