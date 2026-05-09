use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
            allow_override_explicit_microbatch: default_allow_override_explicit_microbatch(),
            fast_repeated_run_profile: default_fast_repeated_run_profile(),
            warmup_steps: default_warmup_steps(),
            measure_steps: default_measure_steps(),
            required_successes: default_required_successes(),
            min_microbatch_size: default_min_microbatch_size(),
            candidate_microbatches: default_candidate_microbatches(),
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelectedRuntimeConfig {
    pub train_microbatch_size: usize,
    pub validation_microbatch_size: usize,
    pub accum_steps: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoaderRuntimeConfig {
    pub num_threads: Option<usize>,
    pub buffer_games: usize,
    pub buffer_samples: usize,
    pub archive_queue_bound: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
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
    SelectedRuntimeConfig {
        train_microbatch_size: train_microbatch.min(batch_size).max(1),
        validation_microbatch_size: validation_microbatch.max(1),
        accum_steps: batch_size.div_ceil(train_microbatch.max(1)).max(1),
    }
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
    pub manifest: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn preflight_defaults_include_search_policy_controls() {
        let config = PreflightConfig::default();
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
    fn default_cache_name_is_stable() {
        assert_eq!(default_cache_name(), PathBuf::from("preflight_cache.json"));
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
        let other = ProfilingEnvelope::from_children(
            "step",
            vec![ProfilingEnvelope::leaf("validation", 0.5)],
        );

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
}
