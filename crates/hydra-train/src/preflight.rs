use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub fn default_enabled() -> bool {
    true
}

pub fn default_advisory_only() -> bool {
    false
}

pub fn default_reuse_cache() -> bool {
    true
}

pub fn default_allow_override_explicit_microbatch() -> bool {
    false
}

pub fn default_warmup_steps() -> usize {
    2
}

pub fn default_measure_steps() -> usize {
    2
}

pub fn default_safety_backoff_rungs() -> usize {
    0
}

pub fn default_required_successes() -> usize {
    2
}

pub fn default_candidate_microbatches() -> Vec<usize> {
    vec![
        512, 384, 320, 288, 256, 224, 192, 160, 144, 128, 112, 104, 96, 80, 72, 64, 48, 32, 24, 16,
        12, 8, 4, 2, 1,
    ]
}

pub fn default_min_microbatch_size() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProbeOnlyConfig {
    pub kind: ProbeKind,
    pub candidate_microbatch: usize,
    pub warmup_steps: Option<usize>,
    pub measure_steps: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PreflightConfig {
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default = "default_advisory_only")]
    pub advisory_only: bool,
    #[serde(default = "default_reuse_cache")]
    pub reuse_cache: bool,
    #[serde(default = "default_allow_override_explicit_microbatch")]
    pub allow_override_explicit_microbatch: bool,
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: usize,
    #[serde(default = "default_measure_steps")]
    pub measure_steps: usize,
    #[serde(default = "default_safety_backoff_rungs")]
    pub safety_backoff_rungs: usize,
    #[serde(default = "default_required_successes")]
    pub required_successes: usize,
    #[serde(default = "default_min_microbatch_size")]
    pub min_microbatch_size: usize,
    #[serde(default = "default_candidate_microbatches")]
    pub candidate_microbatches: Vec<usize>,
    #[serde(default)]
    pub probe_only: Option<ProbeOnlyConfig>,
}

impl Default for PreflightConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            advisory_only: default_advisory_only(),
            reuse_cache: default_reuse_cache(),
            allow_override_explicit_microbatch: default_allow_override_explicit_microbatch(),
            warmup_steps: default_warmup_steps(),
            measure_steps: default_measure_steps(),
            safety_backoff_rungs: default_safety_backoff_rungs(),
            required_successes: default_required_successes(),
            min_microbatch_size: default_min_microbatch_size(),
            candidate_microbatches: default_candidate_microbatches(),
            probe_only: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwareFingerprint {
    pub device_label: String,
    pub backend: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkloadFingerprint {
    pub batch_size: usize,
    pub augment: bool,
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
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbeResult {
    pub kind: ProbeKind,
    pub candidate_microbatch: usize,
    pub status: ProbeStatus,
    pub measured_samples_per_second: Option<f64>,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SelectedRuntimeConfig {
    pub train_microbatch_size: usize,
    pub validation_microbatch_size: usize,
    pub accum_steps: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExplicitSettings {
    pub train_microbatch_explicit: bool,
    pub validation_microbatch_explicit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightReport {
    pub schema_version: u32,
    pub cache_key: PreflightCacheKey,
    pub selected: SelectedRuntimeConfig,
    pub explicit: ExplicitSettings,
    pub advisory_only: bool,
    pub cache_hit: bool,
    pub train_probe_results: Vec<ProbeResult>,
    pub validation_probe_results: Vec<ProbeResult>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreflightCacheEntry {
    pub cache_key: PreflightCacheKey,
    pub selected: SelectedRuntimeConfig,
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

pub fn default_notes() -> Vec<String> {
    vec![
        "preflight tunes runtime microbatch sizing for measured throughput while keeping batch_size semantics fixed"
            .to_string(),
        "resolved settings should remain fixed across resume for comparable BC runs".to_string(),
        "throughput and rough ETA are advisory, not guaranteed".to_string(),
    ]
}

pub fn default_report_name() -> PathBuf {
    PathBuf::from("preflight_report.json")
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
    fn default_notes_describe_throughput_target() {
        let config = PreflightConfig {
            safety_backoff_rungs: 0,
            ..Default::default()
        };
        assert_eq!(config.safety_backoff_rungs, 0);
        assert!(default_notes()[0].contains("measured throughput"));
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
}
