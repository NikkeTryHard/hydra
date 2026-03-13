use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use colored::Colorize;

use super::config::{TrainConfig, train_microbatch_size, validation_microbatch_size};

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeResumeContract {
    pub(crate) batch_size: usize,
    pub(crate) train_microbatch_size: usize,
    pub(crate) validation_microbatch_size: usize,
    pub(crate) accum_steps: usize,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BestValidation {
    pub(crate) policy_loss: f64,
    pub(crate) agreement: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) enum ResumeSemantics {
    RestoreOptimizerSkipSeenSamples,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct BcResumeState {
    pub(crate) schema_version: u32,
    pub(crate) resume_semantics: ResumeSemantics,
    pub(crate) next_epoch: usize,
    pub(crate) skip_optimizer_steps_in_epoch: usize,
    pub(crate) global_step: usize,
    pub(crate) best_validation: Option<BestValidation>,
    pub(crate) runtime: RuntimeResumeContract,
    pub(crate) saved_at_unix_s: u64,
}

pub(crate) struct ResumeContext {
    pub(crate) checkpoint_base: Option<PathBuf>,
    pub(crate) state: Option<BcResumeState>,
    pub(crate) optimizer_base: Option<PathBuf>,
    pub(crate) session_start_global_step: usize,
    pub(crate) start_epoch: usize,
}

impl ResumeContext {
    pub(crate) fn load(config: &TrainConfig) -> Result<Self, String> {
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

    pub(crate) fn best_validation(&self) -> Option<BestValidation> {
        self.state.as_ref().and_then(|state| state.best_validation)
    }

    pub(crate) fn steps_to_skip_for_epoch(&self, epoch: usize) -> usize {
        self.state
            .as_ref()
            .filter(|state| state.next_epoch == epoch)
            .map(|state| state.skip_optimizer_steps_in_epoch)
            .unwrap_or(0)
    }

    pub(crate) fn print_banner(&self) {
        if let Some(state) = self.state.as_ref() {
            println!(
                "{}",
                timestamped(format!(
                    "{} {}",
                    "Resume:".bold().cyan(),
                    resume_banner_message(state).yellow(),
                ))
            );
        }
    }
}

impl ResumeContext {
    pub(crate) fn restores_optimizer_state(&self) -> bool {
        self.state.as_ref().is_some_and(|state| {
            matches!(
                state.resume_semantics,
                ResumeSemantics::RestoreOptimizerSkipSeenSamples
            )
        })
    }
}

pub(crate) struct EpochContinuation {
    pub(crate) next_epoch: usize,
    pub(crate) skip_optimizer_steps_in_epoch: usize,
    pub(crate) epoch_completed: bool,
}

pub(crate) fn current_timestamp_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(crate) fn checkpoint_base_from_path(path: &Path) -> PathBuf {
    if path.extension() == Some(OsStr::new("mpk")) {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

pub(crate) fn latest_state_path_for_checkpoint_base(checkpoint_base: &Path) -> Option<PathBuf> {
    (checkpoint_base.file_name() == Some(OsStr::new("latest_model")))
        .then(|| checkpoint_base.with_file_name("latest_state.yaml"))
}

pub(crate) fn latest_optimizer_base_for_checkpoint_base(checkpoint_base: &Path) -> Option<PathBuf> {
    (checkpoint_base.file_name() == Some(OsStr::new("latest_model")))
        .then(|| checkpoint_base.with_file_name("latest_optimizer"))
}

pub(crate) fn read_resume_state(path: &Path) -> Result<BcResumeState, String> {
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

pub(crate) fn runtime_resume_contract(config: &TrainConfig) -> RuntimeResumeContract {
    let train_microbatch_size = train_microbatch_size(config);
    RuntimeResumeContract {
        batch_size: config.batch_size,
        train_microbatch_size,
        validation_microbatch_size: validation_microbatch_size(config),
        accum_steps: config.batch_size.div_ceil(train_microbatch_size).max(1),
    }
}

pub(crate) fn validate_resume_runtime_compatibility(
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
            "partial-epoch resume requires identical runtime contract; checkpoint train_mb={} val_mb={} accum_steps={} current train_mb={} val_mb={} accum_steps={}",
            state.runtime.train_microbatch_size,
            state.runtime.validation_microbatch_size,
            state.runtime.accum_steps,
            current.train_microbatch_size,
            current.validation_microbatch_size,
            current.accum_steps,
        ));
    }

    Ok(())
}

#[cfg(test)]
pub(crate) fn test_runtime_resume_contract(
    batch_size: usize,
    train_microbatch_size: usize,
    validation_microbatch_size: usize,
) -> RuntimeResumeContract {
    RuntimeResumeContract {
        batch_size,
        train_microbatch_size,
        validation_microbatch_size,
        accum_steps: batch_size.div_ceil(train_microbatch_size).max(1),
    }
}

pub(crate) fn write_resume_state(path: &Path, state: &BcResumeState) -> Result<(), String> {
    let yaml = serde_yaml::to_string(state)
        .map_err(|err| format!("failed to serialize resume state {}: {err}", path.display()))?;
    fs::write(path, yaml)
        .map_err(|err| format!("failed to write resume state {}: {err}", path.display()))
}

pub(crate) fn build_resume_state(
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

pub(crate) fn paused_training_message(continuation: &EpochContinuation) -> String {
    format!(
        "resume_epoch={} skipped_optimizer_steps_in_epoch={} optimizer_state=restored sample_cursor=reconstructed_from_logical_batch_count partial_epoch_requires_matching_runtime",
        continuation.next_epoch + 1,
        continuation.skip_optimizer_steps_in_epoch
    )
}

pub(crate) fn resume_banner_message(state: &BcResumeState) -> String {
    if state.skip_optimizer_steps_in_epoch > 0 {
        format!(
            "global_step={} semantics={:?} skipping {} completed optimizer steps worth of samples in epoch {} before new updates runtime=train_mb:{} val_mb:{} accum_steps:{}",
            state.global_step,
            state.resume_semantics,
            state.skip_optimizer_steps_in_epoch,
            state.next_epoch + 1,
            state.runtime.train_microbatch_size,
            state.runtime.validation_microbatch_size,
            state.runtime.accum_steps,
        )
    } else {
        format!(
            "global_step={} semantics={:?} resuming at epoch {} with new updates immediately runtime=train_mb:{} val_mb:{} accum_steps:{}",
            state.global_step,
            state.resume_semantics,
            state.next_epoch + 1,
            state.runtime.train_microbatch_size,
            state.runtime.validation_microbatch_size,
            state.runtime.accum_steps,
        )
    }
}
use super::presentation::timestamped;
