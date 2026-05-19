use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use crate::preflight::{PreflightBenchTuple, PreflightConfig, ProbeKind};
pub use hydra_data_core::SourceFilterConfig;
use hydra_train_types::phase::TrainingPhase as PipelineTrainingPhase;

pub use super::config_runtime::{
    default_num_threads_for_system, display_num_threads, loader_runtime_config,
    shard_prefetch_depth, train_microbatch_size, validate_config, validate_preflight_config,
    validation_microbatch_size, validation_sample_limit,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    pub data_dir: PathBuf,
    pub output_dir: PathBuf,
    pub num_epochs: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default)]
    pub microbatch_size: Option<usize>,
    #[serde(default)]
    pub validation_microbatch_size: Option<usize>,
    #[serde(default)]
    pub exit_sidecar_path: Option<PathBuf>,
    #[serde(default)]
    pub delta_q_sidecar_path: Option<PathBuf>,
    #[serde(default)]
    pub bc_shards_manifest_path: Option<PathBuf>,
    #[serde(default)]
    pub shard_prefetch_depth: Option<usize>,
    #[serde(default = "default_train_fraction")]
    pub train_fraction: f32,
    #[serde(default)]
    pub source_filters: SourceFilterConfig,
    #[serde(default = "default_augment")]
    pub augment: bool,
    pub resume_checkpoint: Option<PathBuf>,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub advanced_loss: Option<AdvancedLossConfig>,
    #[serde(default)]
    pub validation_gates: ValidationGateConfig,
    pub rl: Option<RlTrainConfig>,
    #[serde(default)]
    pub bc: BcHyperparamConfig,
    #[serde(default)]
    pub nsight_trace: Option<NsightTraceConfig>,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default)]
    pub precision_mode: PrecisionMode,
    #[serde(default = "default_buffer_games")]
    pub buffer_games: usize,
    #[serde(default = "default_buffer_samples")]
    pub buffer_samples: usize,
    #[serde(default)]
    pub num_threads: Option<usize>,
    #[serde(default = "default_tensorboard")]
    pub tensorboard: bool,
    #[serde(default = "default_archive_queue_bound")]
    pub archive_queue_bound: usize,
    #[serde(default = "default_validation_every_n_epochs")]
    pub validation_every_n_epochs: usize,
    #[serde(default = "default_max_skip_logs_per_source")]
    pub max_skip_logs_per_source: usize,
    #[serde(default = "default_log_every_n_steps")]
    pub log_every_n_steps: usize,
    #[serde(default = "default_validate_every_n_steps")]
    pub validate_every_n_steps: usize,
    #[serde(default = "default_checkpoint_every_n_steps")]
    pub checkpoint_every_n_steps: usize,
    #[serde(default)]
    pub max_train_steps: Option<usize>,
    #[serde(default)]
    pub max_validation_batches: Option<usize>,
    #[serde(default = "default_max_validation_samples")]
    pub max_validation_samples: Option<usize>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PrecisionMode {
    #[default]
    Fp32,
    Bf16Autocast,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EffectivePrecision {
    #[default]
    #[serde(rename = "fp32")]
    Fp32,
    #[serde(rename = "fp32_noop")]
    Fp32NoopForBf16Request,
    #[serde(rename = "bf16_amp")]
    Bf16Amp,
}

impl EffectivePrecision {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp32NoopForBf16Request => "fp32_noop",
            Self::Bf16Amp => "bf16_amp",
        }
    }
}

impl std::fmt::Display for EffectivePrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl TrainConfig {
    pub fn use_amp(&self) -> bool {
        matches!(self.precision_mode, PrecisionMode::Bf16Autocast)
    }

    pub fn effective_precision(&self) -> EffectivePrecision {
        match self.precision_mode {
            PrecisionMode::Fp32 => EffectivePrecision::Fp32,
            PrecisionMode::Bf16Autocast => EffectivePrecision::Bf16Amp,
        }
    }

    pub fn default_preflight_bench() -> Self {
        Self {
            data_dir: PathBuf::new(),
            output_dir: PathBuf::from("preflight_bench"),
            num_epochs: 0,
            batch_size: default_batch_size(),
            microbatch_size: None,
            validation_microbatch_size: None,
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            bc_shards_manifest_path: None,
            shard_prefetch_depth: Some(default_shard_prefetch_depth()),
            train_fraction: default_train_fraction(),
            source_filters: SourceFilterConfig::default(),
            augment: default_augment(),
            resume_checkpoint: None,
            seed: default_seed(),
            advanced_loss: None,
            validation_gates: ValidationGateConfig::default(),
            rl: None,
            bc: BcHyperparamConfig::default(),
            nsight_trace: None,
            device: default_device(),
            precision_mode: PrecisionMode::default(),
            buffer_games: default_buffer_games(),
            buffer_samples: default_buffer_samples(),
            num_threads: None,
            tensorboard: default_tensorboard(),
            archive_queue_bound: default_archive_queue_bound(),
            validation_every_n_epochs: default_validation_every_n_epochs(),
            max_skip_logs_per_source: default_max_skip_logs_per_source(),
            log_every_n_steps: default_log_every_n_steps(),
            validate_every_n_steps: default_validate_every_n_steps(),
            checkpoint_every_n_steps: default_checkpoint_every_n_steps(),
            max_train_steps: None,
            max_validation_batches: None,
            max_validation_samples: default_max_validation_samples(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RlPhaseConfig {
    DrdaAchSelfPlay,
    ExitPondering,
}

impl RlPhaseConfig {
    pub fn to_training_phase(self) -> PipelineTrainingPhase {
        match self {
            Self::DrdaAchSelfPlay => PipelineTrainingPhase::DrdaAchSelfPlay,
            Self::ExitPondering => PipelineTrainingPhase::ExitPondering,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct RlTrainConfig {
    #[serde(default = "default_rl_games_per_batch")]
    pub games_per_batch: usize,
    #[serde(default = "default_rl_temperature")]
    pub temperature: f32,
    #[serde(default = "default_rl_phase")]
    pub phase: RlPhaseConfig,
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub exit_weight: Option<f32>,
    #[serde(default)]
    pub aux_weight: Option<f32>,
    #[serde(default)]
    pub microbatch_size: Option<usize>,
}

impl Default for RlTrainConfig {
    fn default() -> Self {
        Self {
            games_per_batch: default_rl_games_per_batch(),
            temperature: default_rl_temperature(),
            phase: default_rl_phase(),
            learning_rate: None,
            exit_weight: None,
            aux_weight: None,
            microbatch_size: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeCliRequest {
    pub kind: ProbeKind,
    pub candidate_microbatch: usize,
    pub warmup_steps: Option<usize>,
    pub measure_steps: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeSingleChildRequest {
    pub request: ProbeCliRequest,
    pub result_path: PathBuf,
    pub manifest_cache_path: Option<PathBuf>,
    pub discovery_summary_path: Option<PathBuf>,
    pub discovery_index_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeBatchChildRequest {
    pub request: ProbeCliRequest,
    pub attempts: usize,
    pub results_path: PathBuf,
    pub manifest_cache_path: Option<PathBuf>,
    pub discovery_summary_path: Option<PathBuf>,
    pub discovery_index_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeChildRequest {
    Single(ProbeSingleChildRequest),
    Batch(ProbeBatchChildRequest),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrainCli {
    pub config_path: Option<PathBuf>,
    pub list_devices: bool,
    pub preflight: Option<PreflightCliOptions>,
    pub delta_q_promotion: bool,
    pub delta_q_baseline_checkpoint: Option<PathBuf>,
    pub probe_only: Option<ProbeCliRequest>,
    pub probe_child: Option<ProbeChildRequest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreflightProfile {
    Default,
    FastRepeatedRun,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreflightCliOptions {
    pub preflight_config: PreflightConfig,
    pub profile: PreflightProfile,
    pub output_dir: PathBuf,
    pub device: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct BcHyperparamConfig {
    #[serde(default = "default_bc_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_bc_min_learning_rate")]
    pub min_learning_rate: f64,
    #[serde(default = "default_bc_weight_decay")]
    pub weight_decay: f32,
    #[serde(default = "default_bc_grad_clip_norm")]
    pub grad_clip_norm: f32,
    #[serde(default = "default_bc_warmup_steps")]
    pub warmup_steps: usize,
}

impl Default for BcHyperparamConfig {
    fn default() -> Self {
        Self {
            learning_rate: default_bc_learning_rate(),
            min_learning_rate: default_bc_min_learning_rate(),
            weight_decay: default_bc_weight_decay(),
            grad_clip_norm: default_bc_grad_clip_norm(),
            warmup_steps: default_bc_warmup_steps(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct NsightTraceConfig {
    pub kernel_launch_count: Option<u64>,
    pub tiny_kernel_fraction: Option<f64>,
    pub cuda_runtime_launch_seconds: Option<f64>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct AdvancedLossConfig {
    pub exit: Option<f32>,
    pub safety_residual: Option<f32>,
    pub belief_fields: Option<f32>,
    pub mixture_weight: Option<f32>,
    pub opponent_hand_type: Option<f32>,
    pub delta_q: Option<f32>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct ValidationGateConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_validation_gate_min_samples")]
    pub min_validation_samples: Option<usize>,
    #[serde(default = "default_validation_gate_max_policy_loss_regression")]
    pub max_policy_loss_regression: Option<f64>,
    #[serde(default)]
    pub min_policy_agreement_delta: Option<f64>,
    #[serde(default)]
    pub fail_training_on_gate_failure: bool,
    #[serde(default = "default_true")]
    pub require_sidecar_coverage_when_weighted: bool,
}

impl Default for ValidationGateConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_validation_samples: default_validation_gate_min_samples(),
            max_policy_loss_regression: default_validation_gate_max_policy_loss_regression(),
            min_policy_agreement_delta: None,
            fail_training_on_gate_failure: false,
            require_sidecar_coverage_when_weighted: true,
        }
    }
}

fn default_validation_gate_min_samples() -> Option<usize> {
    Some(1024)
}

fn default_validation_gate_max_policy_loss_regression() -> Option<f64> {
    Some(0.0)
}

fn default_true() -> bool {
    true
}
pub fn default_batch_size() -> usize {
    2048
}

pub fn default_rl_games_per_batch() -> usize {
    4
}

pub fn default_rl_temperature() -> f32 {
    1.0
}

pub fn default_rl_phase() -> RlPhaseConfig {
    RlPhaseConfig::DrdaAchSelfPlay
}

pub fn default_bc_learning_rate() -> f64 {
    2.5e-4
}

pub fn default_bc_min_learning_rate() -> f64 {
    1e-6
}

pub fn default_bc_weight_decay() -> f32 {
    1e-5
}

pub fn default_bc_grad_clip_norm() -> f32 {
    1.0
}

pub fn default_bc_warmup_steps() -> usize {
    1000
}

pub fn default_train_fraction() -> f32 {
    0.9
}

pub fn default_augment() -> bool {
    true
}

pub fn default_seed() -> u64 {
    0
}

pub fn default_device() -> String {
    "cpu".to_string()
}

pub fn default_buffer_games() -> usize {
    50_000
}

pub fn default_buffer_samples() -> usize {
    32_768
}

pub fn default_tensorboard() -> bool {
    true
}

pub fn default_archive_queue_bound() -> usize {
    128
}

pub fn default_shard_prefetch_depth() -> usize {
    2
}

pub fn default_validation_every_n_epochs() -> usize {
    1
}

pub fn default_max_skip_logs_per_source() -> usize {
    32
}

pub fn default_log_every_n_steps() -> usize {
    50
}

pub fn default_validate_every_n_steps() -> usize {
    200
}

pub fn default_checkpoint_every_n_steps() -> usize {
    200
}

pub fn default_max_validation_samples() -> Option<usize> {
    Some(8_192)
}

pub fn default_preflight_config_for_profile(profile: PreflightProfile) -> PreflightConfig {
    let mut config = PreflightConfig::default();
    match profile {
        PreflightProfile::Default => {}
        PreflightProfile::FastRepeatedRun => {
            config.fast_repeated_run_profile = true;
            config.fast_repeated_run_candidate_window = 1;
            config.required_successes = 1;
            config.warmup_steps = 1;
            config.measure_steps = 1;
            config.loader_runtime_rounds = 0;
            config.loader_tuple_extra_samples = 0;
            config.real_benchmark_enabled = false;
        }
    }
    config
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreflightModeArg {
    Safe,
    Unsafe,
}

fn normalize_long_flag(arg: &str) -> String {
    arg.replace('_', "-")
}

fn parse_preflight_mode(value: &str) -> Result<PreflightModeArg, String> {
    match value {
        "safe" => Ok(PreflightModeArg::Safe),
        "unsafe" => Ok(PreflightModeArg::Unsafe),
        _ => Err(format!(
            "unsupported --preflight-mode value '{value}'; expected safe or unsafe"
        )),
    }
}

fn parse_preflight_profile(value: &str) -> Result<PreflightProfile, String> {
    match value {
        "default" => Ok(PreflightProfile::Default),
        "fast-repeated-run" => Ok(PreflightProfile::FastRepeatedRun),
        _ => Err(format!(
            "unsupported --pf-profile value '{value}'; expected default or fast-repeated-run"
        )),
    }
}

fn parse_positive_usize_text(flag: &str, raw: &str) -> Result<usize, String> {
    if raw.is_empty() || raw.starts_with('+') || raw.starts_with('-') {
        return Err(format!(
            "invalid {flag} value '{raw}': expected positive integer"
        ));
    }
    let value = raw
        .parse::<usize>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if value == 0 {
        return Err(format!("{flag} must be greater than 0"));
    }
    Ok(value)
}

fn parse_usize_flag_allowing_zero(
    flag: &str,
    value: Option<String>,
    allow_zero: bool,
) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    if raw != raw.trim()
        || raw.contains(char::is_whitespace)
        || raw.starts_with('+')
        || raw.starts_with('-')
    {
        return Err(format!("invalid {flag} value '{raw}': expected integer"));
    }
    let parsed = raw
        .parse::<usize>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if !allow_zero && parsed == 0 {
        return Err(format!("{flag} must be greater than 0"));
    }
    Ok(parsed)
}

fn parse_u64_flag_allowing_zero(
    flag: &str,
    value: Option<String>,
    allow_zero: bool,
) -> Result<u64, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    if raw != raw.trim()
        || raw.contains(char::is_whitespace)
        || raw.starts_with('+')
        || raw.starts_with('-')
    {
        return Err(format!("invalid {flag} value '{raw}': expected integer"));
    }
    let parsed = raw
        .parse::<u64>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if !allow_zero && parsed == 0 {
        return Err(format!("{flag} must be greater than 0"));
    }
    Ok(parsed)
}

fn parse_f64_flag(flag: &str, value: Option<String>) -> Result<f64, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.contains(char::is_whitespace) {
        return Err(format!(
            "invalid {flag} value '{raw}': expected finite number"
        ));
    }
    let parsed = trimmed
        .parse::<f64>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))?;
    if !parsed.is_finite() {
        return Err(format!("{flag} must be finite"));
    }
    Ok(parsed)
}

fn parse_bool_flag(flag: &str, value: Option<String>) -> Result<bool, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    match raw.as_str() {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("invalid {flag} value '{raw}'; expected 0 or 1")),
    }
}

fn parse_usize_range_list(flag: &str, raw: &str) -> Result<Vec<usize>, String> {
    let mut values = Vec::new();
    for segment in raw.trim().split(',') {
        let atom = segment.trim();
        if atom.is_empty() {
            return Err(format!("invalid {flag} value '{raw}': empty range segment"));
        }
        if atom.contains(char::is_whitespace) {
            return Err(format!(
                "invalid {flag} value '{raw}': whitespace inside range atom"
            ));
        }
        parse_usize_range_atom(flag, atom, &mut values)?;
    }
    Ok(values)
}

fn parse_usize_range_atom(flag: &str, atom: &str, out: &mut Vec<usize>) -> Result<(), String> {
    if let Some((start, rest)) = atom.split_once('-') {
        let (end, step, multiply) = if let Some((end, step)) = rest.split_once('+') {
            (end, parse_positive_usize_text(flag, step)?, false)
        } else if let Some((end, factor)) = rest.split_once('*') {
            (end, parse_positive_usize_text(flag, factor)?, true)
        } else {
            (rest, 1, false)
        };
        let start = parse_positive_usize_text(flag, start)?;
        let end = parse_positive_usize_text(flag, end)?;
        if start > end {
            return Err(format!("invalid {flag} range '{atom}': start exceeds end"));
        }
        if multiply && step == 1 {
            return Err(format!(
                "invalid {flag} range '{atom}': multiplicative step must be greater than 1"
            ));
        }
        let mut current = start;
        while current <= end {
            out.push(current);
            current = if multiply {
                current.checked_mul(step)
            } else {
                current.checked_add(step)
            }
            .ok_or_else(|| format!("invalid {flag} range '{atom}': overflow"))?;
        }
    } else {
        out.push(parse_positive_usize_text(flag, atom)?);
    }
    Ok(())
}

fn parse_f64_list(flag: &str, raw: &str) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    for segment in raw.trim().split(',') {
        let atom = segment.trim();
        if atom.is_empty() {
            return Err(format!("invalid {flag} value '{raw}': empty float segment"));
        }
        if atom.contains(char::is_whitespace) {
            return Err(format!(
                "invalid {flag} value '{raw}': whitespace inside float atom"
            ));
        }
        let value = atom
            .parse::<f64>()
            .map_err(|err| format!("invalid {flag} value '{atom}': {err}"))?;
        if !value.is_finite() {
            return Err(format!("{flag} entries must be finite"));
        }
        if value <= 0.0 {
            return Err(format!("{flag} entries must be greater than 0"));
        }
        values.push(value);
    }
    Ok(values)
}

fn parse_preflight_bench_candidate_tuples(raw: &str) -> Result<Vec<PreflightBenchTuple>, String> {
    let mut out = Vec::new();
    for atom in raw.split(',') {
        let atom = atom.trim();
        if atom.is_empty() {
            return Err("--pf-candidate-tuples contains an empty tuple".to_string());
        }
        let mut fields = atom.split(':');
        let batch_size = parse_positive_usize_text(
            "--pf-candidate-tuples batch",
            fields.next().unwrap_or_default(),
        )?;
        let ring_batches = parse_positive_usize_text(
            "--pf-candidate-tuples ring",
            fields.next().unwrap_or_default(),
        )?;
        let loader_threads = parse_positive_usize_text(
            "--pf-candidate-tuples threads",
            fields.next().unwrap_or_default(),
        )?;
        let prefetch_batches = parse_positive_usize_text(
            "--pf-candidate-tuples prefetch",
            fields.next().unwrap_or_default(),
        )?;
        if fields.next().is_some() {
            return Err(format!(
                "invalid --pf-candidate-tuples tuple {atom}: expected batch:ring:threads:prefetch"
            ));
        }
        out.push(PreflightBenchTuple {
            batch_size,
            ring_batches,
            loader_threads,
            prefetch_batches,
        });
    }
    if out.is_empty() {
        return Err("--pf-candidate-tuples must contain at least one tuple".to_string());
    }
    Ok(out)
}

pub fn usage(program: &str) -> String {
    format!(
        "Usage:\n  {program} <config.yaml>\n  {program} --preflight [--device <cpu|cuda[:N]>] [--output-dir <dir>] [--pf-candidate-tuples <batch:ring:threads:prefetch,...>] [--pf-warmup-steps <N>] [--pf-measure-steps <N>] [--pf-repetitions <N>] [--pf-output md]\n  {program} --list-devices\n  {program} <config.yaml> --delta-q-promotion [--delta-q-baseline-checkpoint <path>]\n  {program} <config.yaml> --probe-kind <train|validation> --probe-candidate-microbatch <N> [--probe-warmup-steps <N>] [--probe-measure-steps <N>]\n"
    )
}

pub fn version(program: &str) -> String {
    format!("{program} {}", env!("CARGO_PKG_VERSION"))
}

fn parse_probe_kind(value: &str) -> Result<ProbeKind, String> {
    match value {
        "train" => Ok(ProbeKind::Train),
        "validation" => Ok(ProbeKind::Validation),
        "rl_games" => Ok(ProbeKind::RlGames),
        "rl_microbatch" => Ok(ProbeKind::RlMicrobatch),
        _ => Err(format!(
            "unsupported --probe-kind value '{value}'; expected train, validation, rl_games, or rl_microbatch"
        )),
    }
}

fn parse_usize_flag(flag: &str, value: Option<String>) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("missing value for {flag}"))?;
    raw.parse::<usize>()
        .map_err(|err| format!("invalid {flag} value '{raw}': {err}"))
}

pub fn parse_args<I>(args: I) -> Result<TrainCli, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter();
    let program = args.next().unwrap_or_else(|| "train".to_string());
    let first = args.next().ok_or_else(|| usage(&program))?;
    if first == "--" {
        return Err(usage(&program));
    }
    if first == "--list-devices" {
        if args.next().is_some() {
            return Err(
                "--list-devices cannot be combined with config path or train mode flags"
                    .to_string(),
            );
        }
        return Ok(TrainCli {
            config_path: None,
            list_devices: true,
            preflight: None,
            delta_q_promotion: false,
            delta_q_baseline_checkpoint: None,
            probe_only: None,
            probe_child: None,
        });
    }
    let mut config_path = None;
    let mut pending_arg = Some(first);
    let mut probe_kind = None;
    let mut candidate_microbatch = None;
    let mut warmup_steps = None;
    let mut measure_steps = None;
    let mut probe_attempts = None;
    let mut probe_result_path = None;
    let mut probe_results_path = None;
    let mut probe_manifest_cache_path = None;
    let mut probe_discovery_summary_path = None;
    let mut probe_discovery_index_path = None;
    let mut preflight_enabled = false;
    let mut preflight_mode = None;
    let mut preflight_profile = PreflightProfile::Default;
    let mut preflight_config = PreflightConfig::default();
    let mut unsafe_batch_seen = false;
    let mut unsafe_lr_seen = false;
    let mut unsafe_warmup_seen = false;
    let mut preflight_flag_seen = false;
    let mut unsafe_flag_seen = false;
    let mut delta_q_promotion = false;
    let mut delta_q_baseline_checkpoint = None;
    let mut preflight_output_dir = PathBuf::from("preflight_bench");
    let mut preflight_device = default_device();
    while let Some(arg) = pending_arg.take().or_else(|| args.next()) {
        let normalized = normalize_long_flag(&arg);
        if !arg.starts_with('-') {
            if config_path.is_some() {
                return Err(usage(&program));
            }
            config_path = Some(PathBuf::from(arg));
            continue;
        }
        match normalized.as_str() {
            "--help" | "-h" => return Err(usage(&program)),
            "--version" | "-V" => return Err(version(&program)),
            "--list-devices" => {
                return Err(
                    "--list-devices cannot be combined with config path or train mode flags"
                        .to_string(),
                );
            }
            "--preflight" => preflight_enabled = true,
            "--preflight-mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --preflight-mode".to_string())?;
                preflight_mode = Some(parse_preflight_mode(&value)?);
            }
            "--pf-profile" => {
                preflight_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-profile".to_string())?;
                preflight_profile = parse_preflight_profile(&value)?;
                preflight_config = default_preflight_config_for_profile(preflight_profile);
                match preflight_mode {
                    Some(PreflightModeArg::Safe) => {
                        preflight_config.tuning_mode = crate::preflight::PreflightTuningMode::Safe
                    }
                    Some(PreflightModeArg::Unsafe) => {
                        preflight_config.tuning_mode = crate::preflight::PreflightTuningMode::Unsafe
                    }
                    None => {}
                }
            }
            "--pf-candidate-microbatch" => {
                return Err("--pf-candidate-microbatch is deprecated for benchmark preflight; use --pf-candidate-tuples <batch:ring:threads:prefetch,...>".to_string());
            }
            "--pf-candidate-tuples" => {
                preflight_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-candidate-tuples".to_string())?;
                preflight_config.bench_candidate_tuples =
                    parse_preflight_bench_candidate_tuples(&value)?;
            }
            "--pf-output" => {
                preflight_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-output".to_string())?;
                if value != "md" {
                    return Err("--pf-output only supports md in benchmark preflight".to_string());
                }
                preflight_config.bench_output = value;
            }
            "--output-dir" => {
                preflight_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --output-dir".to_string())?;
                preflight_output_dir = PathBuf::from(value);
            }
            "--device" => {
                preflight_flag_seen = true;
                preflight_device = args
                    .next()
                    .ok_or_else(|| "missing value for --device".to_string())?;
            }
            "--pf-min-microbatch" => {
                preflight_flag_seen = true;
                preflight_config.min_microbatch_size =
                    parse_usize_flag_allowing_zero("--pf-min-microbatch", args.next(), false)?;
            }
            "--pf-allow-explicit-microbatch-override" => {
                preflight_flag_seen = true;
                preflight_config.allow_override_explicit_microbatch =
                    parse_bool_flag("--pf-allow-explicit-microbatch-override", args.next())?;
            }
            "--pf-warmup-steps" => {
                preflight_flag_seen = true;
                preflight_config.warmup_steps =
                    parse_usize_flag_allowing_zero("--pf-warmup-steps", args.next(), false)?;
            }
            "--pf-measure-steps" => {
                preflight_flag_seen = true;
                preflight_config.measure_steps =
                    parse_usize_flag_allowing_zero("--pf-measure-steps", args.next(), false)?;
            }
            "--pf-required-successes" => {
                preflight_flag_seen = true;
                preflight_config.required_successes =
                    parse_usize_flag_allowing_zero("--pf-required-successes", args.next(), false)?;
            }
            "--pf-repetitions" => {
                preflight_flag_seen = true;
                preflight_config.required_successes =
                    parse_usize_flag_allowing_zero("--pf-repetitions", args.next(), false)?;
            }
            "--pf-noise-tolerance" => {
                preflight_flag_seen = true;
                preflight_config.measure_noise_tolerance_ratio =
                    parse_f64_flag("--pf-noise-tolerance", args.next())?;
            }
            "--pf-loader-rounds" => {
                preflight_flag_seen = true;
                preflight_config.loader_runtime_rounds =
                    parse_usize_flag_allowing_zero("--pf-loader-rounds", args.next(), true)?;
            }
            "--pf-loader-tuple-margin" => {
                preflight_flag_seen = true;
                preflight_config.loader_tuple_margin_ratio =
                    parse_f64_flag("--pf-loader-tuple-margin", args.next())?;
            }
            "--pf-loader-extra-samples" => {
                preflight_flag_seen = true;
                preflight_config.loader_tuple_extra_samples =
                    parse_usize_flag_allowing_zero("--pf-loader-extra-samples", args.next(), true)?;
            }
            "--pf-real-benchmark" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_enabled =
                    parse_bool_flag("--pf-real-benchmark", args.next())?;
            }
            "--pf-real-benchmark-train-candidates" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_train_candidates = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-train-candidates",
                    args.next(),
                    false,
                )?;
            }
            "--pf-real-benchmark-validation-candidates" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_validation_candidates =
                    parse_usize_flag_allowing_zero(
                        "--pf-real-benchmark-validation-candidates",
                        args.next(),
                        false,
                    )?;
            }
            "--pf-real-benchmark-loader-candidates" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_loader_candidates = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-loader-candidates",
                    args.next(),
                    false,
                )?;
            }
            "--pf-real-benchmark-max-finalists" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_max_finalists = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-max-finalists",
                    args.next(),
                    false,
                )?;
            }
            "--pf-real-benchmark-warmup-steps" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_warmup_steps = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-warmup-steps",
                    args.next(),
                    false,
                )?;
            }
            "--pf-real-benchmark-train-steps" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_train_steps = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-train-steps",
                    args.next(),
                    false,
                )?;
            }
            "--pf-real-benchmark-tie-margin" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_tie_margin_ratio =
                    parse_f64_flag("--pf-real-benchmark-tie-margin", args.next())?;
            }
            "--pf-real-benchmark-extra-finalists" => {
                preflight_flag_seen = true;
                preflight_config.real_benchmark_extra_finalists = parse_usize_flag_allowing_zero(
                    "--pf-real-benchmark-extra-finalists",
                    args.next(),
                    false,
                )?;
            }
            "--pf-finalist-margin" => {
                preflight_flag_seen = true;
                preflight_config.finalist_margin_ratio =
                    parse_f64_flag("--pf-finalist-margin", args.next())?;
            }
            "--pf-finalist-max-candidates" => {
                preflight_flag_seen = true;
                preflight_config.finalist_max_candidates = parse_usize_flag_allowing_zero(
                    "--pf-finalist-max-candidates",
                    args.next(),
                    false,
                )?;
            }
            "--pf-finalist-extra-measure-steps" => {
                preflight_flag_seen = true;
                preflight_config.finalist_extra_measure_steps = parse_usize_flag_allowing_zero(
                    "--pf-finalist-extra-measure-steps",
                    args.next(),
                    false,
                )?;
            }
            "--pf-finalist-extra-successes" => {
                preflight_flag_seen = true;
                preflight_config.finalist_extra_successes = parse_usize_flag_allowing_zero(
                    "--pf-finalist-extra-successes",
                    args.next(),
                    false,
                )?;
            }
            "--pf-target-warmup-seconds" => {
                preflight_flag_seen = true;
                preflight_config.target_warmup_seconds =
                    parse_f64_flag("--pf-target-warmup-seconds", args.next())?;
            }
            "--pf-target-measure-seconds" => {
                preflight_flag_seen = true;
                preflight_config.target_measure_seconds =
                    parse_f64_flag("--pf-target-measure-seconds", args.next())?;
            }
            "--pf-max-adaptive-warmup-steps" => {
                preflight_flag_seen = true;
                preflight_config.max_adaptive_warmup_steps = parse_usize_flag_allowing_zero(
                    "--pf-max-adaptive-warmup-steps",
                    args.next(),
                    false,
                )?;
            }
            "--pf-max-adaptive-measure-steps" => {
                preflight_flag_seen = true;
                preflight_config.max_adaptive_measure_steps = parse_usize_flag_allowing_zero(
                    "--pf-max-adaptive-measure-steps",
                    args.next(),
                    false,
                )?;
            }
            "--pf-local-refinement" => {
                preflight_flag_seen = true;
                preflight_config.local_refinement_enabled =
                    parse_bool_flag("--pf-local-refinement", args.next())?;
            }
            "--pf-local-refinement-max-candidates" => {
                preflight_flag_seen = true;
                preflight_config.local_refinement_max_candidates = parse_usize_flag_allowing_zero(
                    "--pf-local-refinement-max-candidates",
                    args.next(),
                    false,
                )?;
            }
            "--pf-local-refinement-min-gap" => {
                preflight_flag_seen = true;
                preflight_config.local_refinement_min_gap = parse_usize_flag_allowing_zero(
                    "--pf-local-refinement-min-gap",
                    args.next(),
                    false,
                )?;
            }
            "--pf-local-refinement-extra-measure-steps" => {
                preflight_flag_seen = true;
                preflight_config.local_refinement_extra_measure_steps =
                    parse_usize_flag_allowing_zero(
                        "--pf-local-refinement-extra-measure-steps",
                        args.next(),
                        false,
                    )?;
            }
            "--pf-search-coordinate-rounds" => {
                preflight_flag_seen = true;
                preflight_config.search_coordinate_rounds = parse_usize_flag_allowing_zero(
                    "--pf-search-coordinate-rounds",
                    args.next(),
                    false,
                )?;
            }
            "--pf-search-top-k" => {
                preflight_flag_seen = true;
                preflight_config.search_top_k =
                    parse_usize_flag_allowing_zero("--pf-search-top-k", args.next(), false)?;
            }
            "--pf-fast-repeated-run-window" => {
                preflight_flag_seen = true;
                preflight_config.fast_repeated_run_candidate_window =
                    parse_usize_flag_allowing_zero(
                        "--pf-fast-repeated-run-window",
                        args.next(),
                        false,
                    )?;
            }
            "--pf-unsafe-batch-size" => {
                preflight_flag_seen = true;
                unsafe_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-unsafe-batch-size".to_string())?;
                if !unsafe_batch_seen {
                    preflight_config.unsafe_candidate_batch_sizes.clear();
                    unsafe_batch_seen = true;
                }
                preflight_config
                    .unsafe_candidate_batch_sizes
                    .extend(parse_usize_range_list("--pf-unsafe-batch-size", &value)?);
            }
            "--pf-unsafe-lr-scale" => {
                preflight_flag_seen = true;
                unsafe_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-unsafe-lr-scale".to_string())?;
                if !unsafe_lr_seen {
                    preflight_config.unsafe_candidate_lr_scales.clear();
                    unsafe_lr_seen = true;
                }
                preflight_config
                    .unsafe_candidate_lr_scales
                    .extend(parse_f64_list("--pf-unsafe-lr-scale", &value)?);
            }
            "--pf-unsafe-warmup-steps" => {
                preflight_flag_seen = true;
                unsafe_flag_seen = true;
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --pf-unsafe-warmup-steps".to_string())?;
                if !unsafe_warmup_seen {
                    preflight_config.unsafe_candidate_warmup_steps.clear();
                    unsafe_warmup_seen = true;
                }
                preflight_config
                    .unsafe_candidate_warmup_steps
                    .extend(parse_usize_range_list("--pf-unsafe-warmup-steps", &value)?);
            }
            "--pf-rl-min-free-memory-bytes" => {
                preflight_flag_seen = true;
                preflight_config.rl_probe_min_free_memory_bytes = parse_u64_flag_allowing_zero(
                    "--pf-rl-min-free-memory-bytes",
                    args.next(),
                    true,
                )?;
            }
            "--pf-rl-memory-headroom-ratio" => {
                preflight_flag_seen = true;
                preflight_config.rl_probe_memory_headroom_ratio =
                    parse_f64_flag("--pf-rl-memory-headroom-ratio", args.next())?;
            }
            "--pf-rl-growth-safety-factor" => {
                preflight_flag_seen = true;
                preflight_config.rl_probe_growth_safety_factor =
                    parse_f64_flag("--pf-rl-growth-safety-factor", args.next())?;
            }
            "--delta-q-promotion" => delta_q_promotion = true,
            "--delta-q-baseline-checkpoint" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --delta-q-baseline-checkpoint".to_string())?;
                delta_q_baseline_checkpoint = Some(PathBuf::from(value));
            }
            "--probe-kind" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-kind".to_string())?;
                probe_kind = Some(parse_probe_kind(&value)?);
            }
            "--probe-candidate-microbatch" => {
                candidate_microbatch = Some(parse_usize_flag(
                    "--probe-candidate-microbatch",
                    args.next(),
                )?);
            }
            "--probe-warmup-steps" => {
                warmup_steps = Some(parse_usize_flag("--probe-warmup-steps", args.next())?);
            }
            "--probe-measure-steps" => {
                measure_steps = Some(parse_usize_flag("--probe-measure-steps", args.next())?);
            }
            "--probe-attempts" => {
                probe_attempts = Some(parse_usize_flag("--probe-attempts", args.next())?);
            }
            "--probe-result-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-result-path".to_string())?;
                probe_result_path = Some(PathBuf::from(value));
            }
            "--probe-results-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-results-path".to_string())?;
                probe_results_path = Some(PathBuf::from(value));
            }
            "--probe-manifest-cache-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-manifest-cache-path".to_string())?;
                probe_manifest_cache_path = Some(PathBuf::from(value));
            }
            "--probe-discovery-summary-path" => {
                let value = args.next().ok_or_else(|| {
                    "missing value for --probe-discovery-summary-path".to_string()
                })?;
                probe_discovery_summary_path = Some(PathBuf::from(value));
            }
            "--probe-discovery-index-path" => {
                let value = args
                    .next()
                    .ok_or_else(|| "missing value for --probe-discovery-index-path".to_string())?;
                probe_discovery_index_path = Some(PathBuf::from(value));
            }
            _ => return Err(usage(&program)),
        }
    }

    if preflight_flag_seen && !preflight_enabled {
        return Err("--pf-* flags require --preflight".to_string());
    }
    let preflight = if preflight_enabled {
        if config_path.is_some() {
            return Err(
                "--preflight does not accept a config path; pass benchmark flags explicitly"
                    .to_string(),
            );
        }
        let mode = preflight_mode.unwrap_or(PreflightModeArg::Safe);
        match mode {
            PreflightModeArg::Safe => {
                if unsafe_flag_seen {
                    return Err("unsafe --pf-* flags require --preflight-mode unsafe".to_string());
                }
                preflight_config.tuning_mode = crate::preflight::PreflightTuningMode::Safe;
            }
            PreflightModeArg::Unsafe => {
                preflight_config.tuning_mode = crate::preflight::PreflightTuningMode::Unsafe;
            }
        }
        validate_preflight_config(&preflight_config)?;
        Some(PreflightCliOptions {
            preflight_config,
            profile: preflight_profile,
            output_dir: preflight_output_dir.clone(),
            device: preflight_device.clone(),
        })
    } else {
        None
    };

    if preflight.is_some()
        && (probe_kind.is_some()
            || probe_result_path.is_some()
            || probe_results_path.is_some()
            || probe_attempts.is_some()
            || delta_q_promotion
            || delta_q_baseline_checkpoint.is_some())
    {
        return Err(format!(
            "{}\n--preflight cannot be combined with probe-only flags",
            usage(&program)
        ));
    }
    if delta_q_promotion
        && (probe_kind.is_some()
            || probe_result_path.is_some()
            || probe_results_path.is_some()
            || probe_attempts.is_some())
    {
        return Err(format!(
            "{}\n--delta-q-promotion cannot be combined with probe-only flags",
            usage(&program)
        ));
    }
    if delta_q_baseline_checkpoint.is_some() && !delta_q_promotion {
        return Err(format!(
            "{}\n--delta-q-baseline-checkpoint requires --delta-q-promotion",
            usage(&program)
        ));
    }
    if probe_result_path.is_some() && (probe_results_path.is_some() || probe_attempts.is_some()) {
        return Err(format!(
            "{}\ninternal probe child mode cannot combine --probe-result-path with --probe-attempts/--probe-results-path",
            usage(&program)
        ));
    }
    if probe_results_path.is_some() ^ probe_attempts.is_some() {
        return Err(format!(
            "{}\ninternal probe batch child mode requires both --probe-attempts and --probe-results-path",
            usage(&program)
        ));
    }
    if probe_discovery_summary_path.is_some() ^ probe_discovery_index_path.is_some() {
        return Err(format!(
            "{}\ninternal probe child mode requires both --probe-discovery-summary-path and --probe-discovery-index-path",
            usage(&program)
        ));
    }
    match (
        probe_kind,
        candidate_microbatch,
        probe_result_path,
        probe_results_path,
        probe_attempts,
    ) {
        (None, None, None, None, None) => Ok(TrainCli {
            config_path,
            list_devices: false,
            preflight,
            delta_q_promotion,
            delta_q_baseline_checkpoint,
            probe_only: None,
            probe_child: None,
        }),
        (Some(kind), Some(candidate_microbatch), None, None, None) => Ok(TrainCli {
            config_path: Some(config_path.ok_or_else(|| usage(&program))?),
            list_devices: false,
            preflight: None,
            delta_q_promotion: false,
            delta_q_baseline_checkpoint: None,
            probe_only: Some(ProbeCliRequest {
                kind,
                candidate_microbatch,
                warmup_steps,
                measure_steps,
            }),
            probe_child: None,
        }),
        (Some(kind), Some(candidate_microbatch), Some(result_path), None, None) => Ok(TrainCli {
            config_path: Some(config_path.ok_or_else(|| usage(&program))?),
            list_devices: false,
            preflight: None,
            delta_q_promotion: false,
            delta_q_baseline_checkpoint: None,
            probe_only: None,
            probe_child: Some(ProbeChildRequest::Single(ProbeSingleChildRequest {
                request: ProbeCliRequest {
                    kind,
                    candidate_microbatch,
                    warmup_steps,
                    measure_steps,
                },
                result_path,
                manifest_cache_path: probe_manifest_cache_path,
                discovery_summary_path: probe_discovery_summary_path.clone(),
                discovery_index_path: probe_discovery_index_path.clone(),
            })),
        }),
        (Some(kind), Some(candidate_microbatch), None, Some(results_path), Some(attempts)) => {
            Ok(TrainCli {
                config_path: Some(config_path.ok_or_else(|| usage(&program))?),
                list_devices: false,
                preflight: None,
                delta_q_promotion: false,
                delta_q_baseline_checkpoint: None,
                probe_only: None,
                probe_child: Some(ProbeChildRequest::Batch(ProbeBatchChildRequest {
                    request: ProbeCliRequest {
                        kind,
                        candidate_microbatch,
                        warmup_steps,
                        measure_steps,
                    },
                    attempts,
                    results_path,
                    manifest_cache_path: probe_manifest_cache_path,
                    discovery_summary_path: probe_discovery_summary_path,
                    discovery_index_path: probe_discovery_index_path,
                })),
            })
        }
        _ => Err(format!(
            "{}\nprobe mode requires both --probe-kind and --probe-candidate-microbatch",
            usage(&program)
        )),
    }
}

pub fn read_config(path: &Path) -> Result<TrainConfig, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read config {}: {err}", path.display()))?;
    match path.extension().and_then(OsStr::to_str) {
        Some("yaml" | "yml") => {
            let precision_omitted = precision_mode_is_omitted(&raw)
                .map_err(|err| format!("failed to parse yaml config {}: {err}", path.display()))?;
            serde_yaml::from_str::<TrainConfig>(&raw)
                .map(|config| resolve_omitted_precision_mode(config, precision_omitted))
                .map_err(|err| format!("failed to parse yaml config {}: {err}", path.display()))
        }
        _ => Err(format!(
            "unsupported config extension for {}; use .yaml",
            path.display()
        )),
    }
}

fn resolve_omitted_precision_mode(mut config: TrainConfig, precision_omitted: bool) -> TrainConfig {
    if config.precision_mode == PrecisionMode::Fp32 && precision_omitted {
        let delta_q_active = config
            .advanced_loss
            .as_ref()
            .and_then(|loss| loss.delta_q)
            .is_some_and(|weight| weight > 0.0);
        if config.device.starts_with("cuda") && config.rl.is_none() && !delta_q_active {
            config.precision_mode = PrecisionMode::Bf16Autocast;
        }
    }
    config
}

fn precision_mode_is_omitted(raw: &str) -> Result<bool, serde_yaml::Error> {
    match serde_yaml::from_str::<serde_yaml::Value>(raw)? {
        serde_yaml::Value::Mapping(mapping) => {
            Ok(!mapping.contains_key(serde_yaml::Value::String("precision_mode".to_string())))
        }
        _ => Ok(true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pf_repetitions_aliases_required_successes() {
        let cli = parse_args(vec![
            "train".to_string(),
            "--preflight".to_string(),
            "--pf-repetitions".to_string(),
            "5".to_string(),
            "--pf-candidate-tuples".to_string(),
            "1024:2:1:1,2048:4:2:2".to_string(),
        ])
        .expect("pf repetitions should parse");
        let preflight = cli.preflight.expect("preflight options should be present");
        assert_eq!(preflight.preflight_config.required_successes, 5);
        assert_eq!(preflight.preflight_config.bench_candidate_tuples.len(), 2);
    }
}
