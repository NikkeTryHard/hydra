#![allow(
    missing_docs,
    reason = "moved train execution support preserves existing public surface"
)]

pub const DEFAULT_RL_MICROBATCH_SIZE: usize = 128;

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use colored::Colorize;

use hydra_train_types::phase::PipelineState;

use super::artifacts::atomic_write_text;
use hydra_train_runtime::config::{
    EffectivePrecision, PrecisionMode, RlPhaseConfig, RlTrainConfig, TrainConfig,
    train_microbatch_size, validation_microbatch_size,
};

use hydra_train_runtime::preflight::requested_precision_signature;
#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq)]
pub struct RuntimeResumeContract {
    pub batch_size: usize,
    pub train_microbatch_size: usize,
    pub validation_microbatch_size: usize,
    pub accum_steps: usize,
    /// Legacy requested-precision field retained for current resume-state compatibility.
    pub precision_mode: PrecisionMode,
    pub requested_precision: PrecisionMode,
    pub effective_precision: EffectivePrecision,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeResumeContractYaml {
    batch_size: usize,
    train_microbatch_size: usize,
    validation_microbatch_size: usize,
    accum_steps: usize,
    precision_mode: PrecisionMode,
    requested_precision: Option<PrecisionMode>,
    effective_precision: Option<EffectivePrecision>,
}

impl<'de> serde::Deserialize<'de> for RuntimeResumeContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RuntimeResumeContractYaml::deserialize(deserializer)?;
        let requested_precision = raw.requested_precision.unwrap_or(raw.precision_mode);
        let effective_precision = raw
            .effective_precision
            .unwrap_or_else(|| effective_precision_for_request(requested_precision));
        Ok(Self {
            batch_size: raw.batch_size,
            train_microbatch_size: raw.train_microbatch_size,
            validation_microbatch_size: raw.validation_microbatch_size,
            accum_steps: raw.accum_steps,
            precision_mode: raw.precision_mode,
            requested_precision,
            effective_precision,
        })
    }
}

const fn effective_precision_for_request(requested_precision: PrecisionMode) -> EffectivePrecision {
    match requested_precision {
        PrecisionMode::Fp32 => EffectivePrecision::Fp32,
        PrecisionMode::Bf16Autocast => EffectivePrecision::Bf16Amp,
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BestValidation {
    pub policy_loss: f64,
    pub agreement: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum ResumeSemantics {
    RestoreOptimizerSkipSeenSamples,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BcResumeState {
    pub schema_version: u32,
    pub resume_semantics: ResumeSemantics,
    pub next_epoch: usize,
    pub skip_optimizer_steps_in_epoch: usize,
    pub global_step: usize,
    pub best_validation: Option<BestValidation>,
    pub runtime: RuntimeResumeContract,
    pub saved_at_unix_s: u64,
}

pub struct ResumeContext {
    pub checkpoint_base: Option<PathBuf>,
    pub state: Option<BcResumeState>,
    pub optimizer_base: Option<PathBuf>,
    pub session_start_global_step: usize,
    pub start_epoch: usize,
}

#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq)]
pub struct RlRuntimeResumeContract {
    pub games_per_batch: usize,
    pub microbatch_size: usize,
    pub phase: RlPhaseConfig,
    /// Legacy requested-precision field retained for current resume-state compatibility.
    pub precision_mode: PrecisionMode,
    pub requested_precision: PrecisionMode,
    pub effective_precision: EffectivePrecision,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RlRuntimeResumeContractYaml {
    games_per_batch: usize,
    microbatch_size: usize,
    phase: RlPhaseConfig,
    precision_mode: PrecisionMode,
    requested_precision: Option<PrecisionMode>,
    effective_precision: Option<EffectivePrecision>,
}

impl<'de> serde::Deserialize<'de> for RlRuntimeResumeContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RlRuntimeResumeContractYaml::deserialize(deserializer)?;
        let requested_precision = raw.requested_precision.unwrap_or(raw.precision_mode);
        let effective_precision = raw
            .effective_precision
            .unwrap_or_else(|| effective_precision_for_request(requested_precision));
        Ok(Self {
            games_per_batch: raw.games_per_batch,
            microbatch_size: raw.microbatch_size,
            phase: raw.phase,
            precision_mode: raw.precision_mode,
            requested_precision,
            effective_precision,
        })
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RlResumeSemantics {
    RestoreOptimizerFreshSelfPlay,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RlResumeState {
    pub schema_version: u32,
    pub resume_semantics: RlResumeSemantics,
    pub global_step: usize,
    pub pipeline_state: PipelineState,
    pub runtime: RlRuntimeResumeContract,
    pub saved_at_unix_s: u64,
}

pub struct RlResumeContext {
    pub checkpoint_base: Option<PathBuf>,
    pub state: Option<RlResumeState>,
    pub optimizer_base: Option<PathBuf>,
    pub session_start_global_step: usize,
}

impl ResumeContext {
    pub fn load(config: &TrainConfig) -> Result<Self, String> {
        let checkpoint_base = config
            .resume_checkpoint
            .as_ref()
            .map(|path| checkpoint_base_from_path(path));
        let state = checkpoint_base
            .as_ref()
            .and_then(|base| latest_state_path_for_checkpoint_base(base))
            .filter(|path| path.exists())
            .map(|path| read_resume_state(&path))
            .transpose()?;
        let optimizer_base = checkpoint_base
            .as_ref()
            .and_then(|base| latest_optimizer_base_for_checkpoint_base(base))
            .filter(|path| path.with_extension("bin").exists());
        let session_start_global_step = state.as_ref().map(|state| state.global_step).unwrap_or(0);
        let start_epoch = state.as_ref().map(|state| state.next_epoch).unwrap_or(0);
        Ok(Self {
            checkpoint_base,
            state,
            optimizer_base,
            session_start_global_step,
            start_epoch,
        })
    }

    pub fn best_validation(&self) -> Option<BestValidation> {
        self.state.as_ref().and_then(|state| state.best_validation)
    }

    pub fn steps_to_skip_for_epoch(&self, epoch: usize) -> usize {
        self.state
            .as_ref()
            .filter(|state| state.next_epoch == epoch)
            .map(|state| state.skip_optimizer_steps_in_epoch)
            .unwrap_or(0)
    }

    pub fn print_banner_with_effective_runtime(
        &self,
        effective_runtime: Option<RuntimeResumeContract>,
    ) {
        if let Some(state) = self.state.as_ref() {
            println!(
                "{}",
                timestamped(format!(
                    "{} {}",
                    "Resume:".bold().cyan(),
                    resume_banner_message(state, effective_runtime).yellow(),
                ))
            );
        }
    }
}

impl RlResumeContext {
    pub fn load(config: &TrainConfig) -> Result<Self, String> {
        let checkpoint_base = config
            .resume_checkpoint
            .as_ref()
            .map(|path| checkpoint_base_from_path(path));
        let state = checkpoint_base
            .as_ref()
            .and_then(|base| latest_state_path_for_checkpoint_base(base))
            .filter(|path| path.exists())
            .map(|path| read_rl_resume_state(&path))
            .transpose()?;
        let optimizer_base = checkpoint_base
            .as_ref()
            .and_then(|base| latest_optimizer_base_for_checkpoint_base(base))
            .filter(|path| path.with_extension("bin").exists());
        let session_start_global_step = state.as_ref().map(|state| state.global_step).unwrap_or(0);
        Ok(Self {
            checkpoint_base,
            state,
            optimizer_base,
            session_start_global_step,
        })
    }

    pub fn restores_optimizer_state(&self) -> bool {
        self.state.is_some()
    }

    pub fn print_banner(&self) {
        if let Some(state) = self.state.as_ref() {
            println!(
                "{}",
                timestamped(format!(
                    "{} {}",
                    "RL Resume:".bold().cyan(),
                    rl_resume_banner_message(state).yellow(),
                ))
            );
        }
    }
}

impl ResumeContext {
    pub fn restores_optimizer_state(&self) -> bool {
        self.state.as_ref().is_some_and(|state| {
            matches!(
                state.resume_semantics,
                ResumeSemantics::RestoreOptimizerSkipSeenSamples
            )
        })
    }
}

pub struct EpochContinuation {
    pub next_epoch: usize,
    pub skip_optimizer_steps_in_epoch: usize,
    pub epoch_completed: bool,
}

pub fn current_timestamp_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn checkpoint_base_from_path(path: &Path) -> PathBuf {
    if path.extension() == Some(OsStr::new("mpk")) {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

pub fn latest_state_path_for_checkpoint_base(checkpoint_base: &Path) -> Option<PathBuf> {
    (checkpoint_base.file_name() == Some(OsStr::new("latest_model")))
        .then(|| checkpoint_base.with_file_name("latest_state.yaml"))
}

pub fn latest_optimizer_base_for_checkpoint_base(checkpoint_base: &Path) -> Option<PathBuf> {
    (checkpoint_base.file_name() == Some(OsStr::new("latest_model")))
        .then(|| checkpoint_base.with_file_name("latest_optimizer"))
}

pub fn read_resume_state(path: &Path) -> Result<BcResumeState, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read resume state {}: {err}", path.display()))?;
    let state: BcResumeState = serde_yaml::from_str(&raw)
        .map_err(|err| format!("failed to parse resume state {}: {err}", path.display()))?;
    if state.schema_version != 3 {
        return Err(format!(
            "unsupported resume schema_version {} in {}; expected 3",
            state.schema_version,
            path.display()
        ));
    }
    if state.resume_semantics != ResumeSemantics::RestoreOptimizerSkipSeenSamples {
        return Err(format!(
            "unsupported resume semantics {:?} in {}; expected RestoreOptimizerSkipSeenSamples",
            state.resume_semantics,
            path.display()
        ));
    }
    Ok(state)
}

pub fn read_rl_resume_state(path: &Path) -> Result<RlResumeState, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read RL resume state {}: {err}", path.display()))?;
    let state: RlResumeState = serde_yaml::from_str(&raw)
        .map_err(|err| format!("failed to parse RL resume state {}: {err}", path.display()))?;
    if state.schema_version != 1 {
        return Err(format!(
            "unsupported RL resume schema_version {} in {}; expected 1",
            state.schema_version,
            path.display()
        ));
    }
    Ok(state)
}

pub fn runtime_resume_contract(config: &TrainConfig) -> RuntimeResumeContract {
    let train_microbatch_size = train_microbatch_size(config);
    RuntimeResumeContract {
        batch_size: config.batch_size,
        train_microbatch_size,
        validation_microbatch_size: validation_microbatch_size(config),
        accum_steps: config.batch_size.div_ceil(train_microbatch_size).max(1),
        precision_mode: config.precision_mode,
        requested_precision: config.precision_mode,
        effective_precision: config.effective_precision(),
    }
}

pub fn rl_runtime_resume_contract(rl: &RlTrainConfig) -> RlRuntimeResumeContract {
    RlRuntimeResumeContract {
        games_per_batch: rl.games_per_batch,
        microbatch_size: rl.microbatch_size.unwrap_or(DEFAULT_RL_MICROBATCH_SIZE),
        phase: rl.phase,
        precision_mode: PrecisionMode::Fp32,
        requested_precision: PrecisionMode::Fp32,
        effective_precision: EffectivePrecision::Fp32,
    }
}

pub fn validate_rl_resume_runtime_compatibility(
    state: &RlResumeState,
    current: RlRuntimeResumeContract,
) -> Result<(), String> {
    if state.runtime != current {
        return Err(format!(
            "RL resume runtime mismatch: checkpoint games_per_batch={} microbatch_size={} phase={:?} current games_per_batch={} microbatch_size={} phase={:?}",
            state.runtime.games_per_batch,
            state.runtime.microbatch_size,
            state.runtime.phase,
            current.games_per_batch,
            current.microbatch_size,
            current.phase,
        ));
    }
    Ok(())
}

pub fn validate_resume_runtime_compatibility(
    state: &BcResumeState,
    current: RuntimeResumeContract,
) -> Result<(), String> {
    if state.runtime.batch_size != current.batch_size {
        return Err(format!(
            "resume batch_size mismatch: checkpoint={} current={}",
            state.runtime.batch_size, current.batch_size
        ));
    }

    if state.skip_optimizer_steps_in_epoch > 0 && state.runtime != current {
        return Err(format!(
            "partial-epoch resume requires identical runtime contract; checkpoint train_mb={} val_mb={} accum_steps={} requested_precision={} effective_precision={} current train_mb={} val_mb={} accum_steps={} requested_precision={} effective_precision={}",
            state.runtime.train_microbatch_size,
            state.runtime.validation_microbatch_size,
            state.runtime.accum_steps,
            requested_precision_signature(state.runtime.requested_precision),
            state.runtime.effective_precision,
            current.train_microbatch_size,
            current.validation_microbatch_size,
            current.accum_steps,
            requested_precision_signature(current.requested_precision),
            current.effective_precision,
        ));
    }

    Ok(())
}

pub fn test_runtime_resume_contract(
    batch_size: usize,
    train_microbatch_size: usize,
    validation_microbatch_size: usize,
) -> RuntimeResumeContract {
    RuntimeResumeContract {
        batch_size,
        train_microbatch_size,
        validation_microbatch_size,
        accum_steps: batch_size.div_ceil(train_microbatch_size).max(1),
        precision_mode: PrecisionMode::Fp32,
        requested_precision: PrecisionMode::Fp32,
        effective_precision: EffectivePrecision::Fp32,
    }
}

pub fn write_resume_state(path: &Path, state: &BcResumeState) -> Result<(), String> {
    let yaml = serde_yaml::to_string(state)
        .map_err(|err| format!("failed to serialize resume state {}: {err}", path.display()))?;
    atomic_write_text(path, &yaml, "resume state")
}

/// Writes an RL resume state as YAML using atomic sibling replacement.
pub fn write_rl_resume_state(path: &Path, state: &RlResumeState) -> Result<(), String> {
    let yaml = serde_yaml::to_string(state).map_err(|err| {
        format!(
            "failed to serialize RL resume state {}: {err}",
            path.display()
        )
    })?;
    atomic_write_text(path, &yaml, "RL resume state")
}

pub fn build_resume_state(
    next_epoch: usize,
    skip_optimizer_steps_in_epoch: usize,
    global_step: usize,
    best_validation: Option<BestValidation>,
    runtime: RuntimeResumeContract,
) -> BcResumeState {
    BcResumeState {
        schema_version: 3,
        resume_semantics: ResumeSemantics::RestoreOptimizerSkipSeenSamples,
        next_epoch,
        skip_optimizer_steps_in_epoch,
        global_step,
        best_validation,
        runtime,
        saved_at_unix_s: current_timestamp_s(),
    }
}

pub fn build_rl_resume_state(
    global_step: usize,
    pipeline_state: PipelineState,
    runtime: RlRuntimeResumeContract,
) -> RlResumeState {
    RlResumeState {
        schema_version: 1,
        resume_semantics: RlResumeSemantics::RestoreOptimizerFreshSelfPlay,
        global_step,
        pipeline_state,
        runtime,
        saved_at_unix_s: current_timestamp_s(),
    }
}

pub fn paused_training_message(continuation: &EpochContinuation) -> String {
    format!(
        "resume_epoch={} skipped_optimizer_steps_in_epoch={} optimizer_state=restored sample_cursor=reconstructed_from_logical_batch_count partial_epoch_requires_matching_runtime",
        continuation.next_epoch + 1,
        continuation.skip_optimizer_steps_in_epoch
    )
}

pub fn resume_banner_message(
    state: &BcResumeState,
    effective_runtime: Option<RuntimeResumeContract>,
) -> String {
    if state.skip_optimizer_steps_in_epoch > 0 {
        format!(
            "global_step={} semantics={:?} skipping {} completed optimizer steps worth of samples in epoch {} before new updates runtime=train_mb:{} val_mb:{} accum_steps:{} requested_precision={} effective_precision={}",
            state.global_step,
            state.resume_semantics,
            state.skip_optimizer_steps_in_epoch,
            state.next_epoch + 1,
            state.runtime.train_microbatch_size,
            state.runtime.validation_microbatch_size,
            state.runtime.accum_steps,
            requested_precision_signature(state.runtime.requested_precision),
            state.runtime.effective_precision,
        )
    } else {
        let base = format!(
            "global_step={} semantics={:?} resuming at epoch {} with new updates immediately runtime=train_mb:{} val_mb:{} accum_steps:{} requested_precision={} effective_precision={}",
            state.global_step,
            state.resume_semantics,
            state.next_epoch + 1,
            state.runtime.train_microbatch_size,
            state.runtime.validation_microbatch_size,
            state.runtime.accum_steps,
            requested_precision_signature(state.runtime.requested_precision),
            state.runtime.effective_precision,
        );
        match effective_runtime.filter(|runtime| runtime != &state.runtime) {
            Some(runtime) => format!(
                "{} effective_runtime=train_mb:{} val_mb:{} accum_steps:{} requested_precision={} effective_precision={}",
                base,
                runtime.train_microbatch_size,
                runtime.validation_microbatch_size,
                runtime.accum_steps,
                requested_precision_signature(runtime.requested_precision),
                runtime.effective_precision,
            ),
            None => base,
        }
    }
}

pub fn rl_resume_banner_message(state: &RlResumeState) -> String {
    format!(
        "global_step={} semantics={:?} phase={:?} games={} samples={} runtime=games_per_batch:{} microbatch_size:{} requested_precision={} effective_precision={}",
        state.global_step,
        state.resume_semantics,
        state.pipeline_state.phase,
        state.pipeline_state.total_games,
        state.pipeline_state.total_samples,
        state.runtime.games_per_batch,
        state.runtime.microbatch_size,
        requested_precision_signature(state.runtime.requested_precision),
        state.runtime.effective_precision,
    )
}
fn timestamped(message: impl std::fmt::Display) -> String {
    message.to_string()
}

#[cfg(test)]
mod tests;
