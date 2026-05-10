//! Epoch-runner execution helpers shared by the train binary.
//!
//! This module owns BC epoch execution seams without depending on the
//! `hydra-train` binary crate.

use crate::bc_fixed_shape::{FixedShapeTrainConfig, run_train_logical_batch_fixed_chunks};
use burn::backend::libtorch::{LibTorchDevice, TchTensor};
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use colored::Colorize;
use hydra_bc_shards::{BcShardHostBatch, BcShardSplit, load_bc_shard_reader};
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::NUM_CHANNELS;
use hydra_model::amp::maybe_autocast;
use hydra_train_runtime::bc_metrics::{
    BatchMetricSums, batch_metric_sums_from_outputs, batch_stats_from_metric_sums,
};
use hydra_train_runtime::bc_runtime::{BcExitConfig, gated_bc_context, maybe_add_exit_loss};
use hydra_train_runtime::data::sample::{MjaiBcBatch, MjaiSample, collate_samples_bc_owned};
use hydra_train_runtime::head_gates::HeadActivationController;
use hydra_train_runtime::losses::HydraLoss;
use hydra_train_runtime::model::{HydraModel, HydraTrainModelExt};
use hydra_train_runtime::nvtx;
use hydra_train_runtime::preflight::{
    PROFILING_STAGE_BACKWARD, PROFILING_STAGE_BC_EPOCH, PROFILING_STAGE_BC_INTERVAL,
    PROFILING_STAGE_CHECKPOINT, PROFILING_STAGE_COLLATION, PROFILING_STAGE_FORWARD,
    PROFILING_STAGE_H2D_PAGEABLE_TO_PINNED, PROFILING_STAGE_H2D_STREAM_SYNC,
    PROFILING_STAGE_H2D_TENSOR_MATERIALIZE, PROFILING_STAGE_H2D_TRANSFER, PROFILING_STAGE_LOGGING,
    PROFILING_STAGE_LOSS, PROFILING_STAGE_METRIC_READBACK, PROFILING_STAGE_OPTIMIZER_STEP,
    PROFILING_STAGE_PRODUCER_WAIT, PROFILING_STAGE_TRAIN, PROFILING_STAGE_VALIDATION,
    ProfilingEnvelope,
};
use hydra_train_runtime::progress::{
    BatchStats, RareActionMetrics, ScalarAverages, TrainSubStageTiming,
};
use hydra_train_runtime::schedule::lr_status_message;
use hydra_train_runtime::status::{display_validation_scope_label, session_steps_completed};
use hydra_train_runtime::validation::{ValidationRunConfig, ValidationRunLimits};
use hydra_train_types::head_gates::{AdvancedHead, TargetPresence};
use hydra_train_types::losses::{HydraTargets, LossBreakdown};
use indicatif::MultiProgress;
use std::sync::mpsc;
use std::time::Instant;
use tboard::EventWriter;

use crate::advisory::IntervalTimingInput;
use crate::artifacts::{
    BcArtifactPaths, JsonlAppender, PersistedDeltaQPromotionArtifact,
    PersistedValidationGateArtifact, append_training_log_to_writer, log_tensorboard,
    save_checkpoint, write_delta_q_promotion_artifact, write_validation_gate_artifact,
};
use crate::data_pipeline::{DataManifest, StreamingLoaderConfig, TrainValidationLoader};
use crate::presentation::{phase_label, timestamped};
use crate::progress::EpochLogEntry;
use crate::resume::{BestValidation, EpochContinuation};
use crate::validation::{ValidationSummary, evaluate_validation_gates, is_better_validation};
use crate::validation_runner::{ValidationContext, ValidationRuntime, run_validation};

type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

/// Minimal config fields needed for session-relative epoch cadence decisions.
pub trait EpochCadenceConfig {
    /// Log interval in optimizer steps.
    fn log_every_n_steps(&self) -> usize;
    /// Validation interval in optimizer steps.
    fn validate_every_n_steps(&self) -> usize;
    /// Checkpoint interval in optimizer steps.
    fn checkpoint_every_n_steps(&self) -> usize;
    /// Optional session-relative optimizer-step budget.
    fn max_train_steps(&self) -> Option<usize>;
}

/// Value object for epoch cadence decisions.
pub struct EpochCadenceInput {
    /// Log interval in optimizer steps.
    pub log_every_n_steps: usize,
    /// Validation interval in optimizer steps.
    pub validate_every_n_steps: usize,
    /// Checkpoint interval in optimizer steps.
    pub checkpoint_every_n_steps: usize,
    /// Optional session-relative optimizer-step budget.
    pub max_train_steps: Option<usize>,
}

impl EpochCadenceConfig for EpochCadenceInput {
    fn log_every_n_steps(&self) -> usize {
        self.log_every_n_steps
    }

    fn validate_every_n_steps(&self) -> usize {
        self.validate_every_n_steps
    }

    fn checkpoint_every_n_steps(&self) -> usize {
        self.checkpoint_every_n_steps
    }

    fn max_train_steps(&self) -> Option<usize> {
        self.max_train_steps
    }
}

impl From<&hydra_train_runtime::config::TrainConfig> for EpochCadenceInput {
    fn from(config: &hydra_train_runtime::config::TrainConfig) -> Self {
        Self {
            log_every_n_steps: config.log_every_n_steps,
            validate_every_n_steps: config.validate_every_n_steps,
            checkpoint_every_n_steps: config.checkpoint_every_n_steps,
            max_train_steps: config.max_train_steps,
        }
    }
}

/// Per-logical-batch BC execution config.
pub struct TrainLogicalBatchConfig<'a, B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    /// Number of samples per microbatch.
    pub microbatch_size: usize,
    /// Whether data augmentation is enabled.
    pub augment: bool,
    /// Device receiving collated tensors.
    pub train_device: &'a LibTorchDevice,
    /// Active BC loss function.
    pub loss_fn: &'a HydraLoss<B>,
    /// Optional ExIt loss config.
    pub bc_exit_cfg: &'a BcExitConfig,
    /// Effective optimizer learning rate for this step.
    pub lr: f64,
    /// Whether bf16 autocast is enabled.
    pub use_amp: bool,
}

/// Context needed to run a step-boundary validation pass.
pub struct ValidationStepContext<'a, B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    /// Multiprogress sink used for step validation status lines.
    pub multi: &'a MultiProgress,
    /// Runtime train configuration.
    pub config: &'a hydra_train_runtime::config::TrainConfig,
    /// Streaming validation loader configuration.
    pub loader_config: &'a StreamingLoaderConfig,
    /// Data manifest used by validation.
    pub manifest: &'a DataManifest,
    /// Device used for validation tensors.
    pub train_device: &'a LibTorchDevice,
    /// Validation loss function on the inner backend.
    pub valid_loss_fn: &'a HydraLoss<ValidBackendOf<B>>,
    /// Optional ExIt validation loss config.
    pub bc_exit_cfg: &'a BcExitConfig,
    /// Artifact paths for validation gate and best-checkpoint writes.
    pub artifacts: &'a BcArtifactPaths,
    /// Global step at session start, for session-relative labels.
    pub session_start_global_step: usize,
    /// Optional cached validation samples.
    pub cached_validation_samples: Option<&'a [Box<[MjaiSample]>]>,
}

/// Context needed to run an epoch-end validation pass.
pub struct EpochEndValidationContext<'a, B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    /// Runtime train configuration.
    pub config: &'a hydra_train_runtime::config::TrainConfig,
    /// Streaming validation loader configuration.
    pub loader_config: &'a StreamingLoaderConfig,
    /// Data manifest used by validation.
    pub manifest: &'a DataManifest,
    /// Device used for validation tensors.
    pub train_device: &'a LibTorchDevice,
    /// Validation loss function on the inner backend.
    pub valid_loss_fn: &'a HydraLoss<ValidBackendOf<B>>,
    /// Optional ExIt validation loss config.
    pub bc_exit_cfg: &'a BcExitConfig,
    /// Artifact paths for validation gate and best-checkpoint writes.
    pub artifacts: &'a BcArtifactPaths,
    /// Optional cached validation samples.
    pub cached_validation_samples: Option<&'a [Box<[MjaiSample]>]>,
}

/// Event produced by a completed validation pass.
#[derive(Clone)]
pub struct ValidationEvent {
    /// Step or epoch index associated with the validation result.
    pub global_step: usize,
    /// Validation summary.
    pub summary: ValidationSummary,
}

/// Pluggable validation executor used by tests and production validation paths.
pub trait ValidationExecutor<B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    /// Runs one validation pass.
    fn run_validation(
        &mut self,
        model: &HydraModel<B>,
        context: ValidationContext<
            '_,
            hydra_train_runtime::config::TrainConfig,
            TrainValidationLoader<'_>,
            ValidBackendOf<B>,
        >,
        runtime: ValidationRuntime<'_>,
    ) -> Result<ValidationSummary, String>;
}

/// Production validation executor.
#[derive(Default)]
pub struct DefaultValidationExecutor;

impl<B> ValidationExecutor<B> for DefaultValidationExecutor
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    fn run_validation(
        &mut self,
        model: &HydraModel<B>,
        context: ValidationContext<
            '_,
            hydra_train_runtime::config::TrainConfig,
            TrainValidationLoader<'_>,
            ValidBackendOf<B>,
        >,
        runtime: ValidationRuntime<'_>,
    ) -> Result<ValidationSummary, String> {
        run_validation(model, context, runtime)
    }
}

/// Context needed to finalize a completed validation pass.
pub struct CompletedValidationContext<'a, B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    /// Model to checkpoint if this validation result becomes best.
    pub model: &'a HydraModel<B>,
    /// Artifact paths for checkpoint/gate/promotion writes.
    pub artifacts: &'a BcArtifactPaths,
    /// Runtime train configuration.
    pub config: &'a hydra_train_runtime::config::TrainConfig,
    /// Mutable best-validation state.
    pub best_validation: &'a mut Option<BestValidation>,
    /// Step or epoch index written to validation artifacts/checkpoint metadata.
    pub checkpoint_index: usize,
    /// Loss value written to best-checkpoint metadata.
    pub checkpoint_loss: f64,
    /// Artifact scope string for DeltaQ/gate records.
    pub delta_q_scope: &'static str,
}

fn validation_loader(config: &StreamingLoaderConfig) -> TrainValidationLoader<'_> {
    TrainValidationLoader { config }
}

fn validation_delta_q_suffix(summary: &ValidationSummary) -> colored::ColoredString {
    summary
        .delta_q_promotion_snapshot
        .as_ref()
        .map(|report| {
            format!(
            " val_dq_lift={:.4} val_dq_regret={:.4}/{:.4} val_dq_win={:.2}% val_dq_offline_gate={}",
            report.mean_decision_lift,
            report.candidate_mean_regret,
            report.baseline_mean_regret,
            report.regret_beats_baseline_rate * 100.0,
            report.passed
        )
        })
        .unwrap_or_default()
        .yellow()
}

/// Applies validation gates, writes validation artifacts, and updates best checkpoint state.
pub fn finalize_completed_validation<B>(
    context: CompletedValidationContext<'_, B>,
    summary: ValidationSummary,
) -> Result<ValidationSummary, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    let CompletedValidationContext {
        model,
        config,
        artifacts,
        best_validation,
        checkpoint_index,
        checkpoint_loss,
        delta_q_scope,
    } = context;
    let previous_best = *best_validation;
    let run_config = ValidationRunConfig::from_config(config);
    let gate_decision = evaluate_validation_gates(
        &run_config.gates,
        run_config.advanced_loss.as_ref(),
        &summary.scalar_summary(),
        previous_best,
    );
    if gate_decision.enabled {
        write_validation_gate_artifact(
            &artifacts.validation_gate_path,
            &PersistedValidationGateArtifact {
                scope: delta_q_scope,
                step_or_epoch: checkpoint_index,
                decision: &gate_decision,
                samples: summary.samples,
                policy_loss: summary.policy_loss,
                policy_agreement: summary.agreement,
                best_policy_loss: previous_best.map(|best| best.policy_loss),
                best_policy_agreement: previous_best.map(|best| best.agreement),
            },
        )?;
    }
    if is_better_validation(&summary.scalar_summary(), *best_validation) && gate_decision.passed {
        *best_validation = Some(BestValidation {
            policy_loss: summary.policy_loss,
            agreement: summary.agreement,
        });
        let _checkpoint_scope = nvtx::scope(PROFILING_STAGE_CHECKPOINT);
        save_checkpoint(
            model,
            &artifacts.best_model_base,
            checkpoint_index,
            checkpoint_loss,
            Some(&summary),
        )?;
    } else if gate_decision.enabled
        && !gate_decision.passed
        && config.validation_gates.fail_training_on_gate_failure
    {
        return Err(format!(
            "validation gate failed: {}",
            gate_decision.failed_names().join(",")
        ));
    }
    if let (Some(report), Some(result)) = (
        summary.delta_q_promotion.as_ref(),
        summary.delta_q_promotion_result.as_ref(),
    ) {
        write_delta_q_promotion_artifact(
            &artifacts.delta_q_promotion_path,
            &PersistedDeltaQPromotionArtifact {
                scope: delta_q_scope,
                step_or_epoch: checkpoint_index,
                recommendation: result.recommendation(),
                stage: "offline_gate",
                arena_confirmation: None,
                arena_decision: None,
                arena_report: None,
                report,
                result,
                policy_transfer: summary.delta_q_policy_transfer.as_ref(),
                policy_transfer_result: summary.delta_q_policy_transfer_result.as_ref(),
            },
        )?;
    }

    Ok(summary)
}

/// Runs step-boundary validation when the configured cadence is due.
pub fn maybe_run_interval_validation<B>(
    context: ValidationStepContext<'_, B>,
    model: &HydraModel<B>,
    head_controller: Option<&mut HeadActivationController>,
    best_validation: &mut Option<BestValidation>,
    global_step: usize,
    step_window_total_loss: f64,
) -> Result<Option<ValidationSummary>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    maybe_run_interval_validation_with_executor(
        context,
        model,
        head_controller,
        best_validation,
        global_step,
        step_window_total_loss,
        &mut DefaultValidationExecutor,
    )
}

/// Runs step-boundary validation with an injected executor when cadence is due.
pub fn maybe_run_interval_validation_with_executor<B, E>(
    context: ValidationStepContext<'_, B>,
    model: &HydraModel<B>,
    head_controller: Option<&mut HeadActivationController>,
    best_validation: &mut Option<BestValidation>,
    global_step: usize,
    step_window_total_loss: f64,
    validation_executor: &mut E,
) -> Result<Option<ValidationSummary>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    E: ValidationExecutor<B>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    let ValidationStepContext {
        multi,
        config,
        loader_config,
        manifest,
        train_device,
        valid_loss_fn,
        bc_exit_cfg,
        artifacts,
        session_start_global_step,
        cached_validation_samples,
    } = context;
    let session_step = session_steps_completed(global_step, session_start_global_step);
    if session_step == 0 || !session_step.is_multiple_of(config.validate_every_n_steps) {
        return Ok(None);
    }

    multi
        .println(timestamped(format!(
            "{} {}",
            display_validation_scope_label(
                global_step,
                session_start_global_step,
                config.max_train_steps,
            )
            .bold()
            .magenta(),
            ValidationRunLimits::from_config(config)
                .target_samples_label()
                .yellow(),
        )))
        .map_err(|err| format!("failed to print validation start summary: {err}"))?;

    let summary = {
        let _validation_scope = nvtx::scope(PROFILING_STAGE_VALIDATION);
        validation_executor.run_validation(
            model,
            ValidationContext {
                config,
                loader: &validation_loader(loader_config),
                manifest,
                cached_samples: cached_validation_samples,
                device: train_device,
                loss_fn: valid_loss_fn,
                exit_cfg: bc_exit_cfg,
            },
            ValidationRuntime {
                head_controller,
                progress: None,
            },
        )?
    };
    let summary = finalize_completed_validation(
        CompletedValidationContext {
            model,
            config,
            artifacts,
            best_validation,
            checkpoint_index: global_step,
            checkpoint_loss: step_window_total_loss,
            delta_q_scope: "step_validation",
        },
        summary,
    )?;

    multi
        .println(timestamped(format!(
            "{} {} {} {} {}{}",
            display_validation_scope_label(
                global_step,
                session_start_global_step,
                config.max_train_steps,
            )
            .bold()
            .magenta(),
            format!("val_samples={}", summary.samples).yellow(),
            format!("val_policy_ce={:.4}", summary.policy_loss).yellow(),
            format!("val_total={:.4}", summary.total_loss).yellow(),
            format!("val_agree={:.2}%", summary.agreement * 100.0).yellow(),
            validation_delta_q_suffix(&summary),
        )))
        .map_err(|err| format!("failed to print validation summary: {err}"))?;

    Ok(Some(summary))
}

/// Runs epoch-end validation when the configured epoch cadence is due.
pub fn run_epoch_end_validation<B>(
    epoch: usize,
    model: &HydraModel<B>,
    context: EpochEndValidationContext<'_, B>,
    head_controller: Option<&mut HeadActivationController>,
    best_validation: &mut Option<BestValidation>,
    train_total_loss: f64,
) -> Result<Option<ValidationSummary>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    run_epoch_end_validation_with_executor(
        epoch,
        model,
        context,
        head_controller,
        best_validation,
        train_total_loss,
        &mut DefaultValidationExecutor,
    )
}

/// Runs epoch-end validation with an injected executor when epoch cadence is due.
pub fn run_epoch_end_validation_with_executor<B, E>(
    epoch: usize,
    model: &HydraModel<B>,
    context: EpochEndValidationContext<'_, B>,
    head_controller: Option<&mut HeadActivationController>,
    best_validation: &mut Option<BestValidation>,
    train_total_loss: f64,
    validation_executor: &mut E,
) -> Result<Option<ValidationSummary>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    E: ValidationExecutor<B>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    let EpochEndValidationContext {
        config,
        loader_config,
        manifest,
        train_device,
        valid_loss_fn,
        bc_exit_cfg,
        artifacts,
        cached_validation_samples,
    } = context;
    if !should_run_epoch_end_validation(epoch, config.num_epochs, config.validation_every_n_epochs)
    {
        return Ok(None);
    }

    if !benchmark_quiet() {
        println!(
            "{}",
            timestamped(format!(
                "{} {}",
                "validation @ epoch end".bold().magenta(),
                ValidationRunLimits::from_config(config)
                    .target_samples_label()
                    .yellow(),
            ))
        );
    }
    let summary = {
        let _validation_scope = nvtx::scope(PROFILING_STAGE_VALIDATION);
        validation_executor.run_validation(
            model,
            ValidationContext {
                config,
                loader: &validation_loader(loader_config),
                manifest,
                cached_samples: cached_validation_samples,
                device: train_device,
                loss_fn: valid_loss_fn,
                exit_cfg: bc_exit_cfg,
            },
            ValidationRuntime {
                head_controller,
                progress: None,
            },
        )?
    };
    let summary = finalize_completed_validation(
        CompletedValidationContext {
            model,
            config,
            artifacts,
            best_validation,
            checkpoint_index: epoch + 1,
            checkpoint_loss: train_total_loss,
            delta_q_scope: "epoch_validation",
        },
        summary,
    )?;
    if !benchmark_quiet() {
        println!(
            "{}",
            timestamped(format!(
                "{} {} {} {} {}{}",
                "validation @ epoch end".bold().magenta(),
                format!("val_samples={}", summary.samples).yellow(),
                format!("val_policy_ce={:.4}", summary.policy_loss).yellow(),
                format!("val_total={:.4}", summary.total_loss).yellow(),
                format!("val_agree={:.2}%", summary.agreement * 100.0).yellow(),
                validation_delta_q_suffix(&summary),
            ))
        );
    }
    Ok(Some(summary))
}

/// Returns true when a periodic checkpoint should be saved at `global_step`.
#[must_use]
pub fn should_save_periodic_checkpoint(
    config: &impl EpochCadenceConfig,
    global_step: usize,
    session_start_global_step: usize,
) -> bool {
    let interval = config.checkpoint_every_n_steps();
    interval > 0
        && hydra_train_runtime::status::session_steps_completed(
            global_step,
            session_start_global_step,
        ) > 0
        && hydra_train_runtime::status::session_steps_completed(
            global_step,
            session_start_global_step,
        )
        .is_multiple_of(interval)
}

/// Returns true when the train progress message should be refreshed.
#[must_use]
pub fn should_refresh_train_progress_message(
    config: &impl EpochCadenceConfig,
    global_step: usize,
    session_start_global_step: usize,
) -> bool {
    let session_step = hydra_train_runtime::status::session_steps_completed(
        global_step,
        session_start_global_step,
    );
    if session_step == 0 {
        return false;
    }
    session_step == 1
        || config.log_every_n_steps() > 0 && session_step.is_multiple_of(config.log_every_n_steps())
        || config.validate_every_n_steps() > 0
            && session_step.is_multiple_of(config.validate_every_n_steps())
        || should_save_periodic_checkpoint(config, global_step, session_start_global_step)
        || hydra_train_runtime::status::reached_session_step_budget(
            global_step,
            session_start_global_step,
            config.max_train_steps(),
        )
}

/// Returns true when epoch-end validation is due.
#[must_use]
pub fn should_run_epoch_end_validation(
    epoch: usize,
    num_epochs: usize,
    every_n_epochs: usize,
) -> bool {
    (epoch + 1).is_multiple_of(every_n_epochs) || epoch + 1 == num_epochs
}

/// Builds the persisted resume continuation for a completed or paused epoch.
#[must_use]
pub fn build_epoch_continuation(
    epoch: usize,
    epoch_completed: bool,
    epoch_optimizer_steps: usize,
) -> EpochContinuation {
    EpochContinuation {
        next_epoch: if epoch_completed { epoch + 1 } else { epoch },
        skip_optimizer_steps_in_epoch: if epoch_completed {
            0
        } else {
            epoch_optimizer_steps
        },
        epoch_completed,
    }
}

/// Merges a profiling tree into an optional accumulator.
pub fn merge_optional_profiling(
    target: &mut Option<ProfilingEnvelope>,
    source: Option<&ProfilingEnvelope>,
) {
    let Some(source) = source.cloned() else {
        return;
    };
    if let Some(target) = target.as_mut() {
        target.merge_assign(&source);
    } else {
        *target = Some(source);
    }
}

/// Builds an interval-level profiling tree.
#[must_use]
pub fn bc_interval_profiling(
    train_seconds: f64,
    sub_timing: &TrainSubStageTiming,
    validation: Option<ProfilingEnvelope>,
    checkpoint_seconds: f64,
) -> ProfilingEnvelope {
    ProfilingEnvelope::from_children(
        PROFILING_STAGE_BC_INTERVAL,
        vec![
            ProfilingEnvelope::nested(
                PROFILING_STAGE_TRAIN,
                train_seconds,
                sub_timing.to_profiling_children(),
            ),
            validation.unwrap_or_else(|| ProfilingEnvelope::leaf(PROFILING_STAGE_VALIDATION, 0.0)),
            ProfilingEnvelope::leaf(PROFILING_STAGE_CHECKPOINT, checkpoint_seconds),
        ],
    )
}

/// Builds an epoch-level profiling tree.
#[must_use]
pub fn bc_epoch_profiling(
    train_seconds: f64,
    sub_timing: &TrainSubStageTiming,
    validation: Option<ProfilingEnvelope>,
    checkpoint_seconds: f64,
    logging_seconds: f64,
) -> ProfilingEnvelope {
    ProfilingEnvelope::from_children(
        PROFILING_STAGE_BC_EPOCH,
        vec![
            ProfilingEnvelope::nested(
                PROFILING_STAGE_TRAIN,
                train_seconds,
                sub_timing.to_profiling_children(),
            ),
            validation.unwrap_or_else(|| ProfilingEnvelope::leaf(PROFILING_STAGE_VALIDATION, 0.0)),
            ProfilingEnvelope::leaf(PROFILING_STAGE_CHECKPOINT, checkpoint_seconds),
            ProfilingEnvelope::leaf(PROFILING_STAGE_LOGGING, logging_seconds),
        ],
    )
}

/// Returns elapsed seconds for a direct profiling child stage.
#[must_use]
pub fn child_elapsed_seconds(profiling: &ProfilingEnvelope, stage: &str) -> f64 {
    profiling
        .children
        .iter()
        .find(|child| child.stage == stage)
        .map(|child| child.elapsed_seconds)
        .unwrap_or(0.0)
}
/// Config fields needed to emit epoch-finalization outputs.
pub trait EpochFinalizeConfig {
    /// Number of configured epochs in the BC training run.
    fn num_epochs(&self) -> usize;
}

impl EpochFinalizeConfig for hydra_train_runtime::config::TrainConfig {
    fn num_epochs(&self) -> usize {
        self.num_epochs
    }
}

/// Trainer fields needed to render final learning-rate status.
pub trait EpochFinalizeTrainerConfig {
    /// Number of warmup optimizer steps configured for BC training.
    fn warmup_steps(&self) -> usize;
}

impl EpochFinalizeTrainerConfig for hydra_train_types::config::BCTrainerConfig {
    fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }
}

/// Inputs consumed when emitting final epoch logs and TensorBoard scalars.
pub struct EpochFinalizeContext<
    'a,
    C,
    T,
    V,
    D = crate::validation::DeltaQPromotionSnapshot,
    A = crate::advisory::RuntimeAdvisory,
> where
    C: EpochFinalizeConfig,
    T: EpochFinalizeTrainerConfig,
{
    /// Epoch-finalization config view.
    pub config: &'a C,
    /// BC trainer schedule config view.
    pub train_cfg: &'a T,
    /// Zero-based epoch index being finalized.
    pub epoch: usize,
    /// Global optimizer step after the epoch.
    pub global_step: usize,
    /// Aggregated train metrics for this epoch.
    pub train_stats: ScalarAverages,
    /// Optional validation summary for the epoch.
    pub val_summary: Option<V>,
    /// Best validation metrics after validation gates/checkpointing.
    pub best_validation: Option<crate::resume::BestValidation>,
    /// Effective learning rate at final `global_step`.
    pub final_lr: f64,
    /// Optional profiling tree to persist with the epoch log entry.
    pub profiling: Option<ProfilingEnvelope>,
    _delta_q: std::marker::PhantomData<D>,
    _advisory: std::marker::PhantomData<A>,
}

impl<'a, C, T, V, D, A> EpochFinalizeContext<'a, C, T, V, D, A>
where
    C: EpochFinalizeConfig,
    T: EpochFinalizeTrainerConfig,
{
    /// Constructs epoch-finalization inputs without exposing marker fields.
    #[allow(
        clippy::too_many_arguments,
        reason = "constructor mirrors immutable epoch-finalization DTO fields"
    )]
    pub fn new(
        config: &'a C,
        train_cfg: &'a T,
        epoch: usize,
        global_step: usize,
        train_stats: ScalarAverages,
        val_summary: Option<V>,
        best_validation: Option<crate::resume::BestValidation>,
        final_lr: f64,
        profiling: Option<ProfilingEnvelope>,
    ) -> Self {
        Self {
            config,
            train_cfg,
            epoch,
            global_step,
            train_stats,
            val_summary,
            best_validation,
            final_lr,
            profiling,
            _delta_q: std::marker::PhantomData,
            _advisory: std::marker::PhantomData,
        }
    }
}

/// Validation summary view needed by epoch-finalization logging.
pub trait EpochFinalValidationSummary<D> {
    /// Validation rare-action metrics for JSONL output.
    fn rare_actions(&self) -> Option<RareActionMetrics>;
    /// Validation total loss for JSONL output.
    fn total_loss(&self) -> f64;
    /// Validation policy cross-entropy for JSONL/TensorBoard output.
    fn policy_loss(&self) -> f64;
    /// Validation policy agreement for JSONL/TensorBoard output.
    fn agreement(&self) -> f64;
    /// Validation sample count for human-readable epoch summary.
    fn samples(&self) -> usize;
    /// Delta-Q promotion snapshot for JSONL output.
    fn delta_q_promotion_snapshot(&self) -> Option<D>;
}

fn benchmark_quiet() -> bool {
    std::env::var_os("HYDRA_BENCHMARK_QUIET").is_some()
}

impl EpochFinalValidationSummary<crate::validation::DeltaQPromotionSnapshot>
    for crate::validation::ValidationSummary
{
    fn rare_actions(&self) -> Option<RareActionMetrics> {
        Some(self.rare_actions)
    }

    fn total_loss(&self) -> f64 {
        self.total_loss
    }

    fn policy_loss(&self) -> f64 {
        self.policy_loss
    }

    fn agreement(&self) -> f64 {
        self.agreement
    }

    fn samples(&self) -> usize {
        self.samples
    }

    fn delta_q_promotion_snapshot(&self) -> Option<crate::validation::DeltaQPromotionSnapshot> {
        self.delta_q_promotion_snapshot
    }
}

/// Emits final epoch TensorBoard scalars, console summary, and epoch JSONL entry.
pub fn finalize_epoch_outputs<W, C, T, A>(
    tb: &mut Option<EventWriter<W>>,
    training_log: &mut JsonlAppender,
    context: EpochFinalizeContext<
        '_,
        C,
        T,
        crate::validation::ValidationSummary,
        crate::validation::DeltaQPromotionSnapshot,
        A,
    >,
) -> Result<(), String>
where
    W: std::io::Write,
    C: EpochFinalizeConfig,
    T: EpochFinalizeTrainerConfig,
    A: serde::Serialize,
{
    let _logging_scope = nvtx::scope(PROFILING_STAGE_LOGGING);
    let EpochFinalizeContext {
        config,
        train_cfg,
        epoch,
        global_step,
        train_stats,
        val_summary,
        best_validation,
        final_lr,
        mut profiling,
        _delta_q,
        _advisory,
    } = context;
    let logging_started = Instant::now();
    if let Some(ref mut tb_writer) = tb.as_mut() {
        log_tensorboard(
            tb_writer,
            epoch + 1,
            &train_stats,
            val_summary.as_ref(),
            final_lr,
            best_validation,
        )?;
    }

    let lr_message = lr_status_message(global_step, train_cfg.warmup_steps(), final_lr);
    if !benchmark_quiet() {
        println!(
            "{}",
            timestamped(format!(
                "{} {} {} {} {} {}",
                phase_label("epoch", epoch, config.num_epochs())
                    .bold()
                    .cyan(),
                format!("train_loss={:.4}", train_stats.total_loss).green(),
                format!("train_agree={:.2}%", train_stats.policy_agreement * 100.0).green(),
                if let Some(val_summary) = val_summary.as_ref() {
                    format!(
                        "val_ce={:.4} val_agree={:.2}% val_samples={}",
                        val_summary.policy_loss(),
                        val_summary.agreement() * 100.0,
                        val_summary.samples()
                    )
                } else {
                    "val=skipped".to_string()
                }
                .bold()
                .yellow(),
                if let Some(best_validation) = best_validation {
                    format!(
                        "best_ce={:.4} best_agree={:.2}%",
                        best_validation.policy_loss,
                        best_validation.agreement * 100.0
                    )
                } else {
                    "best=n/a".to_string()
                }
                .bold()
                .magenta(),
                lr_message.white(),
            ))
        );
    }

    let logging_seconds = logging_started.elapsed().as_secs_f64();
    if let Some(existing) = profiling.as_mut() {
        existing.merge_assign(&ProfilingEnvelope::from_children(
            existing.stage.clone(),
            vec![ProfilingEnvelope::leaf(
                PROFILING_STAGE_LOGGING,
                logging_seconds,
            )],
        ));
    } else {
        profiling = Some(ProfilingEnvelope::from_children(
            PROFILING_STAGE_BC_EPOCH,
            vec![ProfilingEnvelope::leaf(
                PROFILING_STAGE_LOGGING,
                logging_seconds,
            )],
        ));
    }

    let entry = EpochLogEntry::<crate::validation::DeltaQPromotionSnapshot, A> {
        epoch: epoch + 1,
        global_step,
        lr: final_lr,
        train_total_loss: train_stats.total_loss,
        train_policy_agreement: train_stats.policy_agreement,
        train_loss_policy: train_stats.loss_policy,
        train_loss_value: train_stats.loss_value,
        train_loss_grp: train_stats.loss_grp,
        train_loss_tenpai: train_stats.loss_tenpai,
        train_loss_danger: train_stats.loss_danger,
        train_loss_opp_next: train_stats.loss_opp_next,
        train_loss_score_pdf: train_stats.loss_score_pdf,
        train_loss_score_cdf: train_stats.loss_score_cdf,
        train_rare_actions: train_stats.rare_actions,
        val_rare_actions: val_summary
            .as_ref()
            .and_then(EpochFinalValidationSummary::rare_actions),
        val_total_loss: val_summary
            .as_ref()
            .map(EpochFinalValidationSummary::total_loss),
        val_policy_loss: val_summary
            .as_ref()
            .map(EpochFinalValidationSummary::policy_loss),
        val_policy_agreement: val_summary
            .as_ref()
            .map(EpochFinalValidationSummary::agreement),
        val_delta_q_promotion: val_summary
            .as_ref()
            .and_then(EpochFinalValidationSummary::delta_q_promotion_snapshot),
        profiling,
        advisories: Vec::new(),
        best_val_policy_loss: best_validation.map(|best| best.policy_loss),
        best_val_agreement: best_validation.map(|best| best.agreement),
        num_batches: train_stats.num_batches,
    };
    append_training_log_to_writer(training_log, &entry)?;

    Ok(())
}

/// Returns elapsed seconds for a child under the train profiling node.
#[must_use]
pub fn train_child_elapsed_seconds(profiling: &ProfilingEnvelope, stage: &str) -> f64 {
    profiling
        .children
        .iter()
        .find(|child| child.stage == PROFILING_STAGE_TRAIN)
        .map(|train| child_elapsed_seconds(train, stage))
        .unwrap_or(0.0)
}

/// Returns elapsed seconds for a nested child below a train sub-stage.
#[must_use]
pub fn train_nested_child_elapsed_seconds(
    profiling: &ProfilingEnvelope,
    parent_stage: &str,
    child_stage: &str,
) -> f64 {
    profiling
        .children
        .iter()
        .find(|child| child.stage == PROFILING_STAGE_TRAIN)
        .and_then(|train| {
            train
                .children
                .iter()
                .find(|child| child.stage == parent_stage)
        })
        .map(|parent| child_elapsed_seconds(parent, child_stage))
        .unwrap_or(0.0)
}

/// Converts interval profiling into advisory timing input.
#[must_use]
pub fn interval_timing_input(
    device: &str,
    kernel_launch_count: Option<u64>,
    tiny_kernel_fraction: Option<f64>,
    cuda_runtime_launch_seconds: Option<f64>,
    profiling: &ProfilingEnvelope,
    window_steps: usize,
) -> IntervalTimingInput {
    let device = device.trim().to_ascii_lowercase();
    IntervalTimingInput {
        producer_wait_seconds: train_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_PRODUCER_WAIT,
        ),
        collation_seconds: train_child_elapsed_seconds(profiling, PROFILING_STAGE_COLLATION),
        h2d_transfer_seconds: train_child_elapsed_seconds(profiling, PROFILING_STAGE_H2D_TRANSFER),
        h2d_pageable_to_pinned_seconds: train_nested_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_H2D_TRANSFER,
            PROFILING_STAGE_H2D_PAGEABLE_TO_PINNED,
        ),
        h2d_tensor_materialize_seconds: train_nested_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_H2D_TRANSFER,
            PROFILING_STAGE_H2D_TENSOR_MATERIALIZE,
        ),
        h2d_stream_sync_seconds: train_nested_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_H2D_TRANSFER,
            PROFILING_STAGE_H2D_STREAM_SYNC,
        ),
        metric_readback_seconds: train_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_METRIC_READBACK,
        ),
        forward_seconds: train_child_elapsed_seconds(profiling, PROFILING_STAGE_FORWARD),
        backward_seconds: train_child_elapsed_seconds(profiling, PROFILING_STAGE_BACKWARD),
        optimizer_step_seconds: train_child_elapsed_seconds(
            profiling,
            PROFILING_STAGE_OPTIMIZER_STEP,
        ),
        kernel_launch_count,
        tiny_kernel_fraction,
        cuda_runtime_launch_seconds,
        validation_seconds: child_elapsed_seconds(profiling, PROFILING_STAGE_VALIDATION),
        checkpoint_seconds: child_elapsed_seconds(profiling, PROFILING_STAGE_CHECKPOINT),
        logging_seconds: child_elapsed_seconds(profiling, PROFILING_STAGE_LOGGING),
        total_seconds: profiling.elapsed_seconds,
        steps: window_steps,
        is_cuda: device == "cuda" || device.starts_with("cuda:"),
    }
}

/// Device-resident BC shard batch materialized from a backend-agnostic host batch.
pub struct BcShardDeviceBatch<B: Backend> {
    /// Observation tensor, shape `[batch, 192, 34]`.
    pub obs: Tensor<B, 3>,
    /// Minimal BC batch targets consumed by ExIt-aware loss code.
    pub batch: MjaiBcBatch<B>,
    /// Full Hydra loss targets.
    pub targets: HydraTargets<B>,
}

/// Materializes a backend-agnostic BC shard host batch onto a Burn device.
#[must_use]
pub fn materialize_host_batch_owned<B: Backend>(
    host: BcShardHostBatch,
    device: &B::Device,
) -> BcShardDeviceBatch<B> {
    let target_presence = target_presence_from_host_batch(&host, host.batch_size);
    materialize_host_parts_owned::<B>(
        host.batch_size,
        host.obs_flat,
        host.actions,
        host.legal_mask_flat,
        host.value_target,
        host.grp_target_flat,
        host.oracle_target_flat,
        host.oracle_target_mask,
        host.tenpai_flat,
        host.danger_flat,
        host.danger_mask_flat,
        host.opp_next_flat,
        host.score_pdf_flat,
        host.score_cdf_flat,
        host.safety_target_flat,
        host.safety_mask_flat,
        host.exit_target_flat,
        host.exit_mask_flat,
        host.delta_q_target_flat,
        host.delta_q_mask_flat,
        target_presence,
        device,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "materializes a flat shard record without regrouping hot-path owned buffers"
)]
fn materialize_host_parts_owned<B: Backend>(
    batch_size: usize,
    obs_flat: Vec<f32>,
    actions: Vec<i64>,
    legal_mask_flat: Vec<f32>,
    value_target: Vec<f32>,
    grp_target_flat: Vec<f32>,
    oracle_target_flat: Vec<f32>,
    oracle_target_mask: Vec<f32>,
    tenpai_flat: Vec<f32>,
    danger_flat: Vec<f32>,
    danger_mask_flat: Vec<f32>,
    opp_next_flat: Vec<f32>,
    score_pdf_flat: Vec<f32>,
    score_cdf_flat: Vec<f32>,
    safety_target_flat: Option<Vec<f32>>,
    safety_mask_flat: Option<Vec<f32>>,
    exit_target_flat: Option<Vec<f32>>,
    exit_mask_flat: Option<Vec<f32>>,
    delta_q_target_flat: Option<Vec<f32>>,
    delta_q_mask_flat: Option<Vec<f32>>,
    target_presence: TargetPresence,
    device: &B::Device,
) -> BcShardDeviceBatch<B> {
    let b = batch_size;

    let obs = Tensor::<B, 3>::from_data(TensorData::new(obs_flat, [b, NUM_CHANNELS, 34]), device);
    let policy_target = policy_target_from_action_slice::<B>(actions.as_slice(), b, device);
    let actions_tensor = Tensor::<B, 1, Int>::from_data(TensorData::new(actions, [b]), device);
    let legal_mask = Tensor::<B, 2>::from_data(
        TensorData::new(legal_mask_flat, [b, HYDRA_ACTION_SPACE]),
        device,
    );
    let value_target = Tensor::<B, 1>::from_data(TensorData::new(value_target, [b]), device);
    let grp_target = Tensor::<B, 2>::from_data(TensorData::new(grp_target_flat, [b, 24]), device);
    let oracle_target =
        Tensor::<B, 2>::from_data(TensorData::new(oracle_target_flat, [b, 4]), device);
    let oracle_target_mask =
        Tensor::<B, 1>::from_data(TensorData::new(oracle_target_mask, [b]), device);
    let tenpai_target = Tensor::<B, 2>::from_data(TensorData::new(tenpai_flat, [b, 3]), device);
    let danger_target = Tensor::<B, 3>::from_data(TensorData::new(danger_flat, [b, 3, 34]), device);
    let danger_mask =
        Tensor::<B, 3>::from_data(TensorData::new(danger_mask_flat, [b, 3, 34]), device);
    let opp_next_target =
        Tensor::<B, 3>::from_data(TensorData::new(opp_next_flat, [b, 3, 34]), device);
    let score_pdf_target =
        Tensor::<B, 2>::from_data(TensorData::new(score_pdf_flat, [b, 64]), device);
    let score_cdf_target =
        Tensor::<B, 2>::from_data(TensorData::new(score_cdf_flat, [b, 64]), device);

    let exit_target_tensor = exit_target_flat.map(|buf| {
        Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
    });
    let exit_mask_tensor = exit_mask_flat.map(|buf| {
        Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
    });

    let batch = MjaiBcBatch {
        actions: actions_tensor,
        exit_target: exit_target_tensor,
        exit_mask: exit_mask_tensor,
    };

    let targets = HydraTargets {
        policy_target,
        legal_mask,
        value_target,
        grp_target,
        tenpai_target,
        danger_target,
        danger_mask,
        opp_next_target,
        score_pdf_target,
        score_cdf_target,
        oracle_target: Some(oracle_target),
        belief_fields_target: None,
        belief_fields_mask: None,
        mixture_weight_target: None,
        mixture_weight_mask: None,
        opponent_hand_type_target: None,
        delta_q_target: delta_q_target_flat.map(|buf| {
            Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
        }),
        delta_q_mask: delta_q_mask_flat.map(|buf| {
            Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
        }),
        safety_residual_target: safety_target_flat.map(|buf| {
            Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
        }),
        safety_residual_mask: safety_mask_flat.map(|buf| {
            Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
        }),
        oracle_guidance_mask: Some(oracle_target_mask),
        target_presence: Some(target_presence),
    };

    BcShardDeviceBatch {
        obs,
        batch,
        targets,
    }
}

fn policy_target_from_action_slice<B: Backend>(
    actions: &[i64],
    batch_size: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut flat = vec![0.0f32; batch_size * HYDRA_ACTION_SPACE];
    for (row, &action) in actions.iter().take(batch_size).enumerate() {
        if action >= 0 {
            let idx = action as usize;
            if idx < HYDRA_ACTION_SPACE {
                flat[row * HYDRA_ACTION_SPACE + idx] = 1.0;
            }
        }
    }
    Tensor::<B, 2>::from_data(
        TensorData::new(flat, [batch_size, HYDRA_ACTION_SPACE]),
        device,
    )
}

/// Derives advanced-head target presence counts from a backend-agnostic host batch.
pub fn target_presence_from_host_batch(
    host: &BcShardHostBatch,
    batch_size: usize,
) -> TargetPresence {
    let mut presence = TargetPresence::with_batch_size(batch_size);
    presence.counts[AdvancedHead::OracleCritic.index()] = host
        .oracle_target_mask
        .iter()
        .take(batch_size)
        .filter(|&&value| value > 0.0)
        .count();
    if let (Some(_target), Some(mask)) = (&host.safety_target_flat, &host.safety_mask_flat) {
        presence.counts[AdvancedHead::SafetyResidual.index()] =
            count_nonzero_action_rows(mask, batch_size, HYDRA_ACTION_SPACE);
    }
    if let (Some(_target), Some(mask)) = (&host.delta_q_target_flat, &host.delta_q_mask_flat) {
        let (rows, actions) =
            count_nonzero_action_rows_and_entries(mask, batch_size, HYDRA_ACTION_SPACE);
        presence.counts[AdvancedHead::DeltaQ.index()] = rows;
        presence.delta_q_actions_present = actions;
    }
    presence
}

fn count_nonzero_action_rows(mask: &[f32], batch_size: usize, action_space: usize) -> usize {
    mask.chunks_exact(action_space)
        .take(batch_size)
        .filter(|row| row.iter().any(|&value| value > 0.0))
        .count()
}

fn count_nonzero_action_rows_and_entries(
    mask: &[f32],
    batch_size: usize,
    action_space: usize,
) -> (usize, usize) {
    let mut rows = 0usize;
    let mut entries = 0usize;
    for row in mask.chunks_exact(action_space).take(batch_size) {
        let row_entries = row.iter().filter(|&&value| value > 0.0).count();
        if row_entries > 0 {
            rows += 1;
            entries += row_entries;
        }
    }
    (rows, entries)
}

/// Runs forward/backward/optimizer for one raw replay logical batch.
pub fn train_logical_batch<B, O>(
    logical_batch: &[MjaiSample],
    config: TrainLogicalBatchConfig<'_, B>,
    head_controller: &mut HeadActivationController,
    model_slot: &mut Option<HydraModel<B>>,
    optimizer: &mut O,
) -> Result<(Vec<BatchStats>, TrainSubStageTiming), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    <B as AutodiffBackend>::InnerBackend: Backend<Device = LibTorchDevice>,
    O: Optimizer<HydraModel<B>, B>,
{
    let TrainLogicalBatchConfig {
        microbatch_size,
        use_amp,
        augment,
        train_device,
        loss_fn,
        bc_exit_cfg,
        lr,
    } = config;
    if logical_batch.is_empty() {
        return Ok((Vec::new(), TrainSubStageTiming::default()));
    }

    let mut accumulator: GradientsAccumulator<HydraModel<B>> = GradientsAccumulator::new();
    let logical_batch_len = logical_batch.len().max(1) as f32;
    let mut total_samples = 0usize;
    let mut microbatch_count = 0usize;
    let mut metric_sums: Option<BatchMetricSums<B>> = None;

    if let Some(fixed_shape) = run_train_logical_batch_fixed_chunks(FixedShapeTrainConfig {
        logical_batch,
        augment,
        microbatch_size: microbatch_size.max(1),
        train_device,
        loss_fn,
        bc_exit_cfg,
        head_controller,
        model: epoch_model(model_slot)?,
        use_amp,
    })? {
        let optimizer_started = Instant::now();
        let _optimizer_scope = nvtx::scope(PROFILING_STAGE_OPTIMIZER_STEP);
        let model = model_slot
            .take()
            .ok_or_else(|| "epoch runner model slot should stay populated".to_string())?;
        *model_slot = Some(optimizer.step(lr, model, fixed_shape.grads));
        head_controller.tick_warmup();
        let mut sub_timing = fixed_shape.sub_stage_timing;
        sub_timing.optimizer_step_seconds += optimizer_started.elapsed().as_secs_f64();
        return Ok((vec![fixed_shape.batch_stats], sub_timing));
    }

    let mut sub_timing_fallback = TrainSubStageTiming::default();

    for chunk in logical_batch.chunks(microbatch_size.max(1)) {
        let t = Instant::now();
        let collated = {
            let _collation_scope = nvtx::scope(PROFILING_STAGE_COLLATION);
            collate_samples_bc_owned::<B>(chunk, augment, train_device)
                .map_err(|err| format!("training collation failed: {err}"))?
        };
        sub_timing_fallback.collation_seconds += t.elapsed().as_secs_f64();
        let Some((obs, batch, targets)) = collated else {
            continue;
        };
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let model = epoch_model(model_slot)?;
        let t = Instant::now();
        let output = {
            let _forward_scope = nvtx::scope(PROFILING_STAGE_FORWARD);
            maybe_autocast(use_amp, || {
                model.forward_with_warmup_train(obs, &active_loss_fn.config, &warmup_heads)
            })
        };
        sub_timing_fallback.forward_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (breakdown, total) = {
            let _loss_scope = nvtx::scope(PROFILING_STAGE_LOSS);
            let breakdown: LossBreakdown<B> = active_loss_fn.total_loss(&output, &targets);
            let total = maybe_add_exit_loss(
                breakdown.total.clone(),
                output.policy_logits.clone(),
                batch.exit_target.as_ref(),
                batch.exit_mask.as_ref(),
                bc_exit_cfg,
            );
            (breakdown, total)
        };
        sub_timing_fallback.loss_seconds += t.elapsed().as_secs_f64();

        let chunk_weight = chunk.len() as f32 / logical_batch_len;
        let weighted_chunk_total = total.clone() * chunk_weight;
        let chunk_metric_sums = batch_metric_sums_from_outputs(
            chunk.len(),
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
            total,
            &breakdown,
        );

        metric_sums = Some(match metric_sums.take() {
            Some(existing) => existing.accumulate(chunk_metric_sums),
            None => chunk_metric_sums,
        });
        total_samples += chunk.len();
        microbatch_count += 1;

        {
            let t = Instant::now();
            let _backward_scope = nvtx::scope(PROFILING_STAGE_BACKWARD);
            let grads = weighted_chunk_total.backward();
            let grads = GradientsParams::from_grads(grads, model);
            accumulator.accumulate(model, grads);
            sub_timing_fallback.backward_seconds += t.elapsed().as_secs_f64();
        }
    }

    let batch_stats = if let Some(metric_sums) = metric_sums {
        let metric_started = Instant::now();
        let stats = vec![batch_stats_from_metric_sums(
            total_samples,
            microbatch_count,
            metric_sums,
        )];
        sub_timing_fallback.metric_readback_seconds += metric_started.elapsed().as_secs_f64();
        stats
    } else {
        Vec::new()
    };

    if !batch_stats.is_empty() {
        let t = Instant::now();
        let _optimizer_scope = nvtx::scope(PROFILING_STAGE_OPTIMIZER_STEP);
        let grads = accumulator.grads();
        let model = model_slot
            .take()
            .ok_or_else(|| "epoch runner model slot should stay populated".to_string())?;
        *model_slot = Some(optimizer.step(lr, model, grads));
        head_controller.tick_warmup();
        sub_timing_fallback.optimizer_step_seconds += t.elapsed().as_secs_f64();
    }

    Ok((batch_stats, sub_timing_fallback))
}

/// Runs forward/backward/optimizer for one device-resident BC shard batch.
pub fn train_device_batch<B, O>(
    device_batch: BcShardDeviceBatch<B>,
    sample_count: usize,
    mut sub_timing: TrainSubStageTiming,
    config: TrainLogicalBatchConfig<'_, B>,
    head_controller: &mut HeadActivationController,
    model_slot: &mut Option<HydraModel<B>>,
    optimizer: &mut O,
) -> Result<(Vec<BatchStats>, TrainSubStageTiming), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    <B as AutodiffBackend>::InnerBackend: Backend<Device = LibTorchDevice>,
    O: Optimizer<HydraModel<B>, B>,
{
    let TrainLogicalBatchConfig {
        microbatch_size: _,
        use_amp,
        augment: _,
        train_device: _,
        loss_fn,
        bc_exit_cfg,
        lr,
    } = config;
    let batch_size = device_batch.batch.actions.dims()[0];
    if batch_size == 0 {
        return Ok((Vec::new(), sub_timing));
    }

    let (active_loss_fn, warmup_heads) =
        gated_bc_context(Some(head_controller), loss_fn, &device_batch.targets);
    let model = epoch_model(model_slot)?;
    let t = Instant::now();
    let output = {
        let _forward_scope = nvtx::scope(PROFILING_STAGE_FORWARD);
        maybe_autocast(use_amp, || {
            model.forward_with_warmup_train(device_batch.obs, &active_loss_fn.config, &warmup_heads)
        })
    };
    sub_timing.forward_seconds += t.elapsed().as_secs_f64();

    let t = Instant::now();
    let (breakdown, total) = {
        let _loss_scope = nvtx::scope(PROFILING_STAGE_LOSS);
        let breakdown: LossBreakdown<B> = active_loss_fn.total_loss(&output, &device_batch.targets);
        let total = maybe_add_exit_loss(
            breakdown.total.clone(),
            output.policy_logits.clone(),
            device_batch.batch.exit_target.as_ref(),
            device_batch.batch.exit_mask.as_ref(),
            bc_exit_cfg,
        );
        (breakdown, total)
    };
    sub_timing.loss_seconds += t.elapsed().as_secs_f64();

    let metric_sums = batch_metric_sums_from_outputs(
        sample_count,
        output.policy_logits.clone(),
        device_batch.targets.legal_mask.clone(),
        device_batch.batch.actions.clone(),
        total.clone(),
        &breakdown,
    );

    let t = Instant::now();
    let _backward_scope = nvtx::scope(PROFILING_STAGE_BACKWARD);
    let grads = total.backward();
    let grads = GradientsParams::from_grads(grads, model);
    sub_timing.backward_seconds += t.elapsed().as_secs_f64();

    let t = Instant::now();
    let stats = vec![batch_stats_from_metric_sums(sample_count, 1, metric_sums)];
    sub_timing.metric_readback_seconds += t.elapsed().as_secs_f64();

    let t = Instant::now();
    let _optimizer_scope = nvtx::scope(PROFILING_STAGE_OPTIMIZER_STEP);
    let model = model_slot
        .take()
        .ok_or_else(|| "epoch runner model slot should stay populated".to_string())?;
    *model_slot = Some(optimizer.step(lr, model, grads));
    head_controller.tick_warmup();
    sub_timing.optimizer_step_seconds += t.elapsed().as_secs_f64();

    Ok((stats, sub_timing))
}

/// Runs forward/backward for one host BC-shard batch without mutating model parameters.
#[cfg(feature = "cuda-graph")]
pub fn probe_logical_batch_from_host_batch<B>(
    host_batch: BcShardHostBatch,
    config: TrainLogicalBatchConfig<'_, B>,
    head_controller: &mut HeadActivationController,
    model: &HydraModel<B>,
    staging: Option<&mut (
        crate::pinned_transfer::PinnedStagingArea,
        crate::pinned_transfer::AsyncH2DContext,
        crate::pinned_transfer::PreallocatedDeviceTensors,
    )>,
) -> Result<(Vec<BatchStats>, TrainSubStageTiming), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<FloatTensorPrimitive = TchTensor, IntTensorPrimitive = TchTensor>,
{
    let TrainLogicalBatchConfig {
        microbatch_size,
        use_amp,
        augment: _,
        train_device,
        loss_fn,
        bc_exit_cfg,
        lr: _,
    } = config;

    let mut sub_timing = TrainSubStageTiming::default();
    let t = Instant::now();
    let shard_batch = {
        let _h2d_scope = nvtx::scope(PROFILING_STAGE_H2D_TRANSFER);
        if let Some((pinned_staging, h2d_ctx, gpu_tensors)) = staging {
            let (shard_batch, h2d_timing) = crate::pinned_transfer::materialize_staged_reuse::<B>(
                &host_batch,
                pinned_staging,
                h2d_ctx,
                train_device,
                gpu_tensors,
            );
            sub_timing.h2d_pageable_to_pinned_seconds += h2d_timing.pageable_to_pinned_seconds;
            sub_timing.h2d_tensor_materialize_seconds += h2d_timing.tensor_materialize_seconds;
            sub_timing.h2d_stream_sync_seconds += h2d_timing.stream_sync_seconds;
            shard_batch
        } else {
            let t_materialize = Instant::now();
            let shard_batch = materialize_host_batch_owned::<B>(host_batch, train_device);
            sub_timing.h2d_tensor_materialize_seconds += t_materialize.elapsed().as_secs_f64();
            shard_batch
        }
    };
    sub_timing.h2d_transfer_seconds += t.elapsed().as_secs_f64();

    let (stats, mut compute_timing) = probe_device_batch_compute(
        shard_batch,
        microbatch_size,
        use_amp,
        loss_fn,
        bc_exit_cfg,
        head_controller,
        model,
    )?;
    compute_timing.h2d_transfer_seconds = sub_timing.h2d_transfer_seconds;
    compute_timing.h2d_pageable_to_pinned_seconds = sub_timing.h2d_pageable_to_pinned_seconds;
    compute_timing.h2d_tensor_materialize_seconds = sub_timing.h2d_tensor_materialize_seconds;
    compute_timing.h2d_stream_sync_seconds = sub_timing.h2d_stream_sync_seconds;
    Ok((stats, compute_timing))
}

/// Runs forward/backward for one device-resident BC-shard batch without mutating model parameters.
#[cfg(feature = "cuda-graph")]
pub fn probe_device_batch_compute<B>(
    shard_batch: BcShardDeviceBatch<B>,
    microbatch_size: usize,
    use_amp: bool,
    loss_fn: &HydraLoss<B>,
    bc_exit_cfg: &BcExitConfig,
    head_controller: &mut HeadActivationController,
    model: &HydraModel<B>,
) -> Result<(Vec<BatchStats>, TrainSubStageTiming), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<FloatTensorPrimitive = TchTensor, IntTensorPrimitive = TchTensor>,
{
    let mut sub_timing = TrainSubStageTiming::default();
    let obs = shard_batch.obs;
    let batch = shard_batch.batch;
    let targets = shard_batch.targets;
    let batch_size = batch.actions.dims()[0];
    if batch_size == 0 {
        return Ok((Vec::new(), sub_timing));
    }

    let logical_batch_len = batch_size.max(1) as f32;
    let mut total_samples = 0usize;
    let mut microbatch_count = 0usize;
    let mut metric_sums: Option<BatchMetricSums<B>> = None;
    let effective_microbatch = microbatch_size.max(1);
    if effective_microbatch >= batch_size {
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let t = Instant::now();
        let output = {
            let _forward_scope = nvtx::scope(PROFILING_STAGE_FORWARD);
            maybe_autocast(use_amp, || {
                model.forward_with_warmup_train(obs.clone(), &active_loss_fn.config, &warmup_heads)
            })
        };
        sub_timing.forward_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (breakdown, total) = {
            let _loss_scope = nvtx::scope(PROFILING_STAGE_LOSS);
            let breakdown = active_loss_fn.total_loss(&output, &targets);
            let total = maybe_add_exit_loss(
                breakdown.total.clone(),
                output.policy_logits.clone(),
                batch.exit_target.as_ref(),
                batch.exit_mask.as_ref(),
                bc_exit_cfg,
            );
            (breakdown, total)
        };
        sub_timing.loss_seconds += t.elapsed().as_secs_f64();
        metric_sums = Some(batch_metric_sums_from_outputs(
            batch_size,
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
            total.clone(),
            &breakdown,
        ));
        total_samples = batch_size;
        microbatch_count = 1;
        let t = Instant::now();
        let _backward_scope = nvtx::scope(PROFILING_STAGE_BACKWARD);
        let _grads = total.backward();
        sub_timing.backward_seconds += t.elapsed().as_secs_f64();
    } else {
        for start in (0..batch_size).step_by(effective_microbatch) {
            let end = (start + effective_microbatch).min(batch_size);
            let chunk_len = end - start;
            #[allow(
                clippy::single_range_in_vec_init,
                reason = "Burn slice API expects a one-element range slice"
            )]
            let r = [start..end];
            let obs_chunk = obs.clone().slice(r.clone());
            let batch_chunk = MjaiBcBatch {
                actions: batch.actions.clone().slice(r.clone()),
                exit_target: batch
                    .exit_target
                    .as_ref()
                    .map(|t| t.clone().slice(r.clone())),
                exit_mask: batch.exit_mask.as_ref().map(|t| t.clone().slice(r.clone())),
            };
            let targets_chunk = targets.slice_batch(start, end);
            let (active_loss_fn, warmup_heads) =
                gated_bc_context(Some(head_controller), loss_fn, &targets_chunk);
            let t = Instant::now();
            let output = {
                let _forward_scope = nvtx::scope(PROFILING_STAGE_FORWARD);
                maybe_autocast(use_amp, || {
                    model.forward_with_warmup_train(
                        obs_chunk,
                        &active_loss_fn.config,
                        &warmup_heads,
                    )
                })
            };
            sub_timing.forward_seconds += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let (breakdown, total) = {
                let _loss_scope = nvtx::scope(PROFILING_STAGE_LOSS);
                let breakdown = active_loss_fn.total_loss(&output, &targets_chunk);
                let total = maybe_add_exit_loss(
                    breakdown.total.clone(),
                    output.policy_logits.clone(),
                    batch_chunk.exit_target.as_ref(),
                    batch_chunk.exit_mask.as_ref(),
                    bc_exit_cfg,
                );
                (breakdown, total)
            };
            sub_timing.loss_seconds += t.elapsed().as_secs_f64();
            let chunk_metric_sums = batch_metric_sums_from_outputs(
                chunk_len,
                output.policy_logits.clone(),
                targets_chunk.legal_mask.clone(),
                batch_chunk.actions.clone(),
                total.clone(),
                &breakdown,
            );
            metric_sums = Some(match metric_sums.take() {
                Some(existing) => existing.accumulate(chunk_metric_sums),
                None => chunk_metric_sums,
            });
            total_samples += chunk_len;
            microbatch_count += 1;
            let t = Instant::now();
            let _backward_scope = nvtx::scope(PROFILING_STAGE_BACKWARD);
            let weighted_total = total * (chunk_len as f32 / logical_batch_len);
            let _grads = weighted_total.backward();
            sub_timing.backward_seconds += t.elapsed().as_secs_f64();
        }
    }
    let stats = if let Some(metric_sums) = metric_sums {
        let metric_started = Instant::now();
        let stats = vec![batch_stats_from_metric_sums(
            total_samples,
            microbatch_count,
            metric_sums,
        )];
        sub_timing.metric_readback_seconds += metric_started.elapsed().as_secs_f64();
        stats
    } else {
        Vec::new()
    };
    Ok((stats, sub_timing))
}

/// Runs forward/backward for one full device-resident BC-shard batch without metrics.
#[cfg(feature = "cuda-graph")]
pub fn probe_device_batch_compute_no_stats<B>(
    shard_batch: BcShardDeviceBatch<B>,
    microbatch_size: usize,
    use_amp: bool,
    loss_fn: &HydraLoss<B>,
    bc_exit_cfg: &BcExitConfig,
    head_controller: &mut HeadActivationController,
    model: &HydraModel<B>,
) -> Result<TrainSubStageTiming, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<FloatTensorPrimitive = TchTensor, IntTensorPrimitive = TchTensor>,
{
    let mut sub_timing = TrainSubStageTiming::default();
    let obs = shard_batch.obs;
    let batch = shard_batch.batch;
    let targets = shard_batch.targets;
    let batch_size = batch.actions.dims()[0];
    if batch_size == 0 {
        return Ok(sub_timing);
    }

    let effective_microbatch = microbatch_size.max(1);
    if effective_microbatch < batch_size {
        return Err("CUDA graph capture probe requires full-batch microbatch to avoid capture-unsafe slice/materialization ops".to_string());
    }
    let (active_loss_fn, warmup_heads) = gated_bc_context(Some(head_controller), loss_fn, &targets);
    let t = Instant::now();
    let output = {
        let _forward_scope = nvtx::scope(PROFILING_STAGE_FORWARD);
        maybe_autocast(use_amp, || {
            model.forward_with_warmup_train(obs, &active_loss_fn.config, &warmup_heads)
        })
    };
    sub_timing.forward_seconds += t.elapsed().as_secs_f64();
    let t = Instant::now();
    let breakdown = active_loss_fn.total_loss(&output, &targets);
    let total = maybe_add_exit_loss(
        breakdown.total,
        output.policy_logits,
        batch.exit_target.as_ref(),
        batch.exit_mask.as_ref(),
        bc_exit_cfg,
    );
    sub_timing.loss_seconds += t.elapsed().as_secs_f64();
    let t = Instant::now();
    let _backward_scope = nvtx::scope(PROFILING_STAGE_BACKWARD);
    let _grads = total.backward();
    sub_timing.backward_seconds += t.elapsed().as_secs_f64();
    Ok(sub_timing)
}

/// Runs the forward/backward/optimizer step from a pre-built host batch.
///
/// The host batch was already collated on the CPU producer thread. This function
/// materializes it onto the device, then runs the same microbatch accumulation
/// and optimizer step as the raw replay path.
///
/// When `staging` is `Some`, the host batch is staged into pinned memory and
/// the H2D transfer is issued on a dedicated copy stream with event-based
/// synchronization.
pub fn train_logical_batch_from_host_batch<B, O>(
    host_batch: BcShardHostBatch,
    config: TrainLogicalBatchConfig<'_, B>,
    head_controller: &mut HeadActivationController,
    model_slot: &mut Option<HydraModel<B>>,
    optimizer: &mut O,
    #[cfg(feature = "cuda-graph")] staging: Option<&mut (
        crate::pinned_transfer::PinnedStagingArea,
        crate::pinned_transfer::AsyncH2DContext,
        crate::pinned_transfer::PreallocatedDeviceTensors,
    )>,
) -> Result<
    (
        Vec<BatchStats>,
        TrainSubStageTiming,
        Option<BcShardHostBatch>,
    ),
    String,
>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<FloatTensorPrimitive = TchTensor, IntTensorPrimitive = TchTensor>,
    O: Optimizer<HydraModel<B>, B>,
{
    let TrainLogicalBatchConfig {
        microbatch_size,
        use_amp,
        augment: _,
        train_device,
        loss_fn,
        bc_exit_cfg,
        lr,
    } = config;

    let mut sub_timing = TrainSubStageTiming::default();

    let t = Instant::now();
    let (shard_batch, recycled_host_batch) = {
        let _h2d_scope = nvtx::scope(PROFILING_STAGE_H2D_TRANSFER);
        #[cfg(feature = "cuda-graph")]
        {
            if let Some((pinned_staging, h2d_ctx, gpu_tensors)) = staging {
                let (shard_batch, h2d_timing) = crate::pinned_transfer::materialize_staged_reuse::<B>(
                    &host_batch,
                    pinned_staging,
                    h2d_ctx,
                    train_device,
                    gpu_tensors,
                );
                sub_timing.h2d_pageable_to_pinned_seconds += h2d_timing.pageable_to_pinned_seconds;
                sub_timing.h2d_tensor_materialize_seconds += h2d_timing.tensor_materialize_seconds;
                sub_timing.h2d_stream_sync_seconds += h2d_timing.stream_sync_seconds;
                (shard_batch, None)
            } else {
                let t_materialize = Instant::now();
                let shard_batch = materialize_host_batch_owned::<B>(host_batch, train_device);
                sub_timing.h2d_tensor_materialize_seconds += t_materialize.elapsed().as_secs_f64();
                (shard_batch, None)
            }
        }
        #[cfg(not(feature = "cuda-graph"))]
        {
            let t_materialize = Instant::now();
            let shard_batch = materialize_host_batch_owned::<B>(host_batch, train_device);
            sub_timing.h2d_tensor_materialize_seconds += t_materialize.elapsed().as_secs_f64();
            (shard_batch, None)
        }
    };
    sub_timing.h2d_transfer_seconds += t.elapsed().as_secs_f64();

    let obs = shard_batch.obs;
    let batch = shard_batch.batch;
    let targets = shard_batch.targets;
    let batch_size = batch.actions.dims()[0];

    if batch_size == 0 {
        return Ok((Vec::new(), sub_timing, recycled_host_batch));
    }

    let logical_batch_len = batch_size.max(1) as f32;
    let mut accumulator: GradientsAccumulator<HydraModel<B>> = GradientsAccumulator::new();
    let mut total_samples = 0usize;
    let mut microbatch_count = 0usize;
    let mut metric_sums: Option<BatchMetricSums<B>> = None;
    let effective_microbatch = microbatch_size.max(1);

    if effective_microbatch >= batch_size {
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let model = epoch_model(model_slot)?;
        let t = Instant::now();
        let output = {
            let _forward_scope = nvtx::scope(PROFILING_STAGE_FORWARD);
            maybe_autocast(use_amp, || {
                model.forward_with_warmup_train(obs, &active_loss_fn.config, &warmup_heads)
            })
        };
        sub_timing.forward_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (breakdown, total) = {
            let _loss_scope = nvtx::scope(PROFILING_STAGE_LOSS);
            let breakdown = active_loss_fn.total_loss(&output, &targets);
            let total = maybe_add_exit_loss(
                breakdown.total.clone(),
                output.policy_logits.clone(),
                batch.exit_target.as_ref(),
                batch.exit_mask.as_ref(),
                bc_exit_cfg,
            );
            (breakdown, total)
        };
        sub_timing.loss_seconds += t.elapsed().as_secs_f64();
        metric_sums = Some(batch_metric_sums_from_outputs(
            batch_size,
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
            total.clone(),
            &breakdown,
        ));
        total_samples = batch_size;
        microbatch_count = 1;
        let t = Instant::now();
        {
            let _backward_scope = nvtx::scope(PROFILING_STAGE_BACKWARD);
            let grads = total.backward();
            let grads = GradientsParams::from_grads(grads, model);
            accumulator.accumulate(model, grads);
        }
        sub_timing.backward_seconds += t.elapsed().as_secs_f64();
    } else {
        for start in (0..batch_size).step_by(effective_microbatch) {
            let end = (start + effective_microbatch).min(batch_size);
            let chunk_len = end - start;
            #[allow(
                clippy::single_range_in_vec_init,
                reason = "Burn slice API expects a one-element range slice"
            )]
            let r = [start..end];
            let obs_chunk = obs.clone().slice(r.clone());
            let batch_chunk = MjaiBcBatch {
                actions: batch.actions.clone().slice(r.clone()),
                exit_target: batch
                    .exit_target
                    .as_ref()
                    .map(|t| t.clone().slice(r.clone())),
                exit_mask: batch.exit_mask.as_ref().map(|t| t.clone().slice(r.clone())),
            };
            let targets_chunk = targets.slice_batch(start, end);
            let (active_loss_fn, warmup_heads) =
                gated_bc_context(Some(head_controller), loss_fn, &targets_chunk);
            let model = epoch_model(model_slot)?;
            let t = Instant::now();
            let output = {
                let _forward_scope = nvtx::scope(PROFILING_STAGE_FORWARD);
                maybe_autocast(use_amp, || {
                    model.forward_with_warmup_train(
                        obs_chunk,
                        &active_loss_fn.config,
                        &warmup_heads,
                    )
                })
            };
            sub_timing.forward_seconds += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let (breakdown, total) = {
                let _loss_scope = nvtx::scope(PROFILING_STAGE_LOSS);
                let breakdown = active_loss_fn.total_loss(&output, &targets_chunk);
                let total = maybe_add_exit_loss(
                    breakdown.total.clone(),
                    output.policy_logits.clone(),
                    batch_chunk.exit_target.as_ref(),
                    batch_chunk.exit_mask.as_ref(),
                    bc_exit_cfg,
                );
                (breakdown, total)
            };
            sub_timing.loss_seconds += t.elapsed().as_secs_f64();
            let chunk_weight = chunk_len as f32 / logical_batch_len;
            let weighted_total = total.clone() * chunk_weight;
            let chunk_metric_sums = batch_metric_sums_from_outputs(
                chunk_len,
                output.policy_logits.clone(),
                targets_chunk.legal_mask.clone(),
                batch_chunk.actions.clone(),
                total,
                &breakdown,
            );
            metric_sums = Some(match metric_sums.take() {
                Some(existing) => existing.accumulate(chunk_metric_sums),
                None => chunk_metric_sums,
            });
            total_samples += chunk_len;
            microbatch_count += 1;
            let t = Instant::now();
            {
                let _backward_scope = nvtx::scope(PROFILING_STAGE_BACKWARD);
                let grads = weighted_total.backward();
                let grads = GradientsParams::from_grads(grads, model);
                accumulator.accumulate(model, grads);
            }
            sub_timing.backward_seconds += t.elapsed().as_secs_f64();
        }
    }

    let optimizer_started = Instant::now();
    let _optimizer_scope = nvtx::scope(PROFILING_STAGE_OPTIMIZER_STEP);
    let model = model_slot
        .take()
        .ok_or_else(|| "epoch runner model slot should stay populated".to_string())?;
    *model_slot = Some(optimizer.step(lr, model, accumulator.grads()));
    head_controller.tick_warmup();
    sub_timing.optimizer_step_seconds += optimizer_started.elapsed().as_secs_f64();

    let stats = if let Some(metric_sums) = metric_sums {
        let metric_started = Instant::now();
        let stats = vec![batch_stats_from_metric_sums(
            total_samples,
            microbatch_count,
            metric_sums,
        )];
        sub_timing.metric_readback_seconds += metric_started.elapsed().as_secs_f64();
        stats
    } else {
        Vec::new()
    };
    Ok((stats, sub_timing, recycled_host_batch))
}

fn epoch_model<B>(model_slot: &Option<HydraModel<B>>) -> Result<&HydraModel<B>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    model_slot
        .as_ref()
        .ok_or_else(|| "epoch runner model slot should stay populated".to_string())
}

/// Records drained batch stats into epoch and step-window accumulators.
pub fn record_drained_batch_stats(
    drained: Vec<BatchStats>,
    stats: &mut ScalarAverages,
    step_window: &mut ScalarAverages,
) {
    for batch_stats in drained {
        stats.record_batch(batch_stats);
        step_window.record_batch(batch_stats);
    }
}

/// One host batch received from the BC shard prefetch producer.
pub struct BcShardPrefetchBatch {
    /// Collated host batch ready for device materialization.
    pub host_batch: BcShardHostBatch,
    /// Number of samples collated into this batch.
    pub sample_count: usize,
    /// Seconds spent waiting for the producer to provide this batch.
    pub producer_wait_seconds: f64,
}

/// Producer/consumer prefetcher for contiguous BC shard training batches.
pub struct BcShardPrefetcher {
    rx: mpsc::Receiver<Result<(BcShardHostBatch, usize), String>>,
    recycle_tx: mpsc::SyncSender<BcShardHostBatch>,
    producer_handle: Option<std::thread::JoinHandle<()>>,
}

impl BcShardPrefetcher {
    /// Starts a bounded producer thread reading train-split shard batches.
    pub fn spawn(
        manifest_path: &std::path::Path,
        batch_size: usize,
        augment: bool,
        start_index: usize,
        total_rows: usize,
        prefetch_depth: usize,
    ) -> Result<Self, String> {
        let reader = load_bc_shard_reader(manifest_path, BcShardSplit::Train)?;
        let producer_start_index = start_index.min(total_rows);
        let (tx, rx) =
            mpsc::sync_channel::<Result<(BcShardHostBatch, usize), String>>(prefetch_depth);
        let (recycle_tx, recycle_rx) = mpsc::sync_channel::<BcShardHostBatch>(prefetch_depth + 1);
        let producer_handle = std::thread::Builder::new()
            .name("bc-shard-prefetch".into())
            .spawn(move || {
                let mut scratch = reader.new_scratch(batch_size);
                let mut idx = producer_start_index;
                while idx < total_rows {
                    let take = batch_size.min(total_rows - idx);
                    let result = reader
                        .collate_host_batch_range_into(idx, take, augment, &mut scratch)
                        .map(|()| {
                            let batch = if let Ok(mut recycled) = recycle_rx.try_recv() {
                                scratch.swap_batch(&mut recycled)
                            } else {
                                scratch.take_batch()
                            };
                            (batch, take)
                        });
                    if tx.send(result).is_err() {
                        break;
                    }
                    idx += take;
                }
            })
            .map_err(|err| format!("failed to spawn bc-shard-prefetch thread: {err}"))?;

        Ok(Self {
            rx,
            recycle_tx,
            producer_handle: Some(producer_handle),
        })
    }

    /// Receives the next prefetched host batch, including producer wait timing.
    pub fn recv(&self) -> Result<Option<BcShardPrefetchBatch>, String> {
        let recv_started = Instant::now();
        let recv_result = match self.rx.recv() {
            Ok(result) => result,
            Err(_) => return Ok(None),
        };
        let producer_wait_seconds = recv_started.elapsed().as_secs_f64();
        let (host_batch, sample_count) = recv_result?;
        Ok(Some(BcShardPrefetchBatch {
            host_batch,
            sample_count,
            producer_wait_seconds,
        }))
    }

    /// Returns a consumed host batch to the producer for allocation reuse.
    pub fn recycle(&self, host_batch: BcShardHostBatch) {
        let _ = self.recycle_tx.try_send(host_batch);
    }

    /// Stops the prefetcher and propagates producer panics.
    pub fn join(mut self) -> Result<(), String> {
        drop(self.rx);
        drop(self.recycle_tx);
        if let Some(handle) = self.producer_handle.take() {
            handle
                .join()
                .map_err(|_| "bc-shard-prefetch thread panicked".to_string())?;
        }
        Ok(())
    }
}
