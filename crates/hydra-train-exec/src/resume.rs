#![allow(
    missing_docs,
    reason = "moved train execution support preserves existing public surface"
)]

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use colored::Colorize;

use super::artifacts::atomic_write_text;
use hydra_train_runtime::config::{EffectivePrecision, PrecisionMode, TrainConfig};
use hydra_train_runtime::preflight::requested_precision_signature;
pub use hydra_train_runtime::resume::{
    BcResumeState, BestValidation, ResumeSemantics, RlResumeSemantics, RlResumeState,
    RlRuntimeResumeContract, RuntimeResumeContract, rl_runtime_resume_contract,
    runtime_resume_contract, validate_resume_runtime_compatibility,
    validate_rl_resume_runtime_compatibility,
};
use hydra_train_types::phase::PipelineState;

pub struct ResumeContext {
    pub checkpoint_base: Option<PathBuf>,
    pub state: Option<BcResumeState>,
    pub optimizer_base: Option<PathBuf>,
    pub session_start_global_step: usize,
    pub start_epoch: usize,
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
