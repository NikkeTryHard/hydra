//! Pure runtime resume-state DTOs and compatibility checks.
//!
//! This module intentionally contains no filesystem IO. Execution crates own checkpoint paths,
//! atomic writes, and user-facing resume banners; runtime owns the serialized contract shape and
//! hard-fail compatibility rules.

use hydra_train_types::config::DEFAULT_RL_MICROBATCH_SIZE;
use hydra_train_types::phase::PipelineState;

use crate::config::{
    EffectivePrecision, PrecisionMode, RlPhaseConfig, RlTrainConfig, TrainConfig,
    train_microbatch_size, validation_microbatch_size,
};
use crate::preflight::requested_precision_signature;

/// Best BC validation snapshot preserved in resume state.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BestValidation {
    /// Best validation policy loss seen so far.
    pub policy_loss: f64,
    /// Best validation policy agreement seen so far.
    pub agreement: f64,
}

/// BC resume semantics encoded in persisted resume state.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum ResumeSemantics {
    /// Restore optimizer state and skip already-consumed optimizer steps in a partial epoch.
    RestoreOptimizerSkipSeenSamples,
}

/// Strict BC runtime tuple persisted with checkpoint resume state.
#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq)]
pub struct RuntimeResumeContract {
    /// Logical BC batch size.
    pub batch_size: usize,
    /// Physical train microbatch size.
    pub train_microbatch_size: usize,
    /// Physical validation microbatch size.
    pub validation_microbatch_size: usize,
    /// Gradient accumulation steps derived from logical batch and train microbatch.
    pub accum_steps: usize,
    /// Legacy requested-precision field retained for current resume-state compatibility.
    pub precision_mode: PrecisionMode,
    /// Requested precision from YAML after omitted-default resolution.
    pub requested_precision: PrecisionMode,
    /// Actual precision behavior for the resolved runtime.
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
            .unwrap_or_else(|| legacy_effective_precision_for_missing_field(requested_precision));
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

const fn legacy_effective_precision_for_missing_field(
    requested_precision: PrecisionMode,
) -> EffectivePrecision {
    match requested_precision {
        PrecisionMode::Fp32 => EffectivePrecision::Fp32,
        PrecisionMode::Bf16Autocast => EffectivePrecision::Fp32NoopForBf16Request,
    }
}

/// Persisted BC resume state payload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BcResumeState {
    /// Resume-state schema version.
    pub schema_version: u32,
    /// Resume semantics expected by current runtime.
    pub resume_semantics: ResumeSemantics,
    /// Next epoch index to run.
    pub next_epoch: usize,
    /// Already-completed optimizer steps to skip within `next_epoch`.
    pub skip_optimizer_steps_in_epoch: usize,
    /// Global optimizer step at checkpoint save time.
    pub global_step: usize,
    /// Best validation snapshot, if any.
    pub best_validation: Option<BestValidation>,
    /// Runtime contract persisted at checkpoint save time.
    pub runtime: RuntimeResumeContract,
    /// Wall-clock checkpoint save timestamp.
    pub saved_at_unix_s: u64,
}

/// Strict RL runtime tuple persisted with checkpoint resume state.
#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq)]
pub struct RlRuntimeResumeContract {
    /// RL games generated per optimizer step.
    pub games_per_batch: usize,
    /// RL train microbatch size.
    pub microbatch_size: usize,
    /// RL phase contract.
    pub phase: RlPhaseConfig,
    /// Legacy requested-precision field retained for current resume-state compatibility.
    pub precision_mode: PrecisionMode,
    /// Requested precision. RL currently persists FP32 only.
    pub requested_precision: PrecisionMode,
    /// Effective precision. RL currently persists FP32 only.
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
            .unwrap_or_else(|| legacy_effective_precision_for_missing_field(requested_precision));
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

/// RL resume semantics encoded in persisted resume state.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum RlResumeSemantics {
    /// Restore optimizer state while regenerating future self-play data.
    RestoreOptimizerFreshSelfPlay,
}

/// Persisted RL resume state payload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RlResumeState {
    /// Resume-state schema version.
    pub schema_version: u32,
    /// Resume semantics expected by current runtime.
    pub resume_semantics: RlResumeSemantics,
    /// Global optimizer step at checkpoint save time.
    pub global_step: usize,
    /// Persisted self-play pipeline state.
    pub pipeline_state: PipelineState,
    /// Runtime contract persisted at checkpoint save time.
    pub runtime: RlRuntimeResumeContract,
    /// Wall-clock checkpoint save timestamp.
    pub saved_at_unix_s: u64,
}

/// Builds the current BC resume runtime contract from train config.
#[must_use]
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

/// Builds the current RL resume runtime contract from RL config.
#[must_use]
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

/// Validates strict RL resume runtime compatibility.
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

/// Validates BC resume runtime compatibility.
///
/// Batch size is always strict. Other runtime changes are allowed only at epoch boundaries; partial
/// epoch resumes must replay the exact persisted runtime contract.
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

/// Test helper for constructing an FP32 BC resume runtime contract.
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
        precision_mode: PrecisionMode::Fp32,
        requested_precision: PrecisionMode::Fp32,
        effective_precision: EffectivePrecision::Fp32,
    }
}

#[cfg(test)]
mod tests;
