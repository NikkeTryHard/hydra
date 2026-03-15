use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub fn default_allow_override_explicit_microbatch() -> bool {
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

pub fn default_finalist_margin_ratio() -> f64 {
    0.015
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
    #[serde(default = "default_validation_growth_patience")]
    pub validation_growth_patience: usize,
    #[serde(default = "default_validation_growth_max_steps")]
    pub validation_growth_max_steps: usize,
    #[serde(default = "default_measure_noise_tolerance_ratio")]
    pub measure_noise_tolerance_ratio: f64,
    #[serde(default = "default_loader_runtime_rounds")]
    pub loader_runtime_rounds: usize,
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
            warmup_steps: default_warmup_steps(),
            measure_steps: default_measure_steps(),
            required_successes: default_required_successes(),
            min_microbatch_size: default_min_microbatch_size(),
            candidate_microbatches: default_candidate_microbatches(),
            validation_growth_patience: default_validation_growth_patience(),
            validation_growth_max_steps: default_validation_growth_max_steps(),
            measure_noise_tolerance_ratio: default_measure_noise_tolerance_ratio(),
            loader_runtime_rounds: default_loader_runtime_rounds(),
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
    pub train_fraction_bits: u32,
    pub max_skip_logs_per_source: usize,
    pub max_validation_batches: Option<usize>,
    pub max_validation_samples: Option<usize>,
    pub model_signature: String,
    pub code_signature: String,
    pub advanced_loss_signature: String,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExplicitSettings {
    pub train_microbatch_explicit: bool,
    pub validation_microbatch_explicit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightCacheEntry {
    pub cache_key: PreflightCacheKey,
    pub runtime: EffectiveRuntimeConfig,
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
}
