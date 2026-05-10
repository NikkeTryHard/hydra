use std::collections::VecDeque;
use std::io::Write;
use std::sync::mpsc;
use std::time::Instant;

use burn::backend::libtorch::{LibTorchDevice, TchTensor};
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use colored::Colorize;
use indicatif::{MultiProgress, ProgressBar};
use tboard::EventWriter;

use hydra_train::amp::maybe_autocast;
#[cfg(feature = "cuda-graph")]
use hydra_train::data::bc_shards::BcShardBatch;
use hydra_train::data::bc_shards::{BcShardHostBatch, BcShardSplit, load_bc_shard_reader};
use hydra_train::data::pipeline::{DataManifest, StreamingLoaderConfig, stream_train_epoch};
use hydra_train::data::sample::{MjaiBcBatch, MjaiSample};
use hydra_train::model::{HydraModel, HydraTrainModelExt};
#[cfg(test)]
use hydra_train::preflight::PROFILING_STAGE_PRODUCER_WAIT;
use hydra_train::preflight::{
    PROFILING_STAGE_BACKWARD, PROFILING_STAGE_BC_EPOCH, PROFILING_STAGE_BC_INTERVAL,
    PROFILING_STAGE_CHECKPOINT, PROFILING_STAGE_FORWARD, PROFILING_STAGE_H2D_TRANSFER,
    PROFILING_STAGE_LOGGING, PROFILING_STAGE_LOSS, PROFILING_STAGE_OPTIMIZER_STEP,
    PROFILING_STAGE_TRAIN, PROFILING_STAGE_VALIDATION, ProfilingEnvelope,
};
use hydra_train::training::bc::{BcExitConfig, gated_bc_context, maybe_add_exit_loss};
use hydra_train::training::head_gates::HeadActivationController;
use hydra_train::training::losses::HydraLoss;
use hydra_train_types::config::BCTrainerConfig;

use super::TrainBackend;
use super::advisory::{
    AdvisoryEvent, IntervalTimingInput, RuntimeAdvisory, interval_runtime_advisories,
};
use super::artifacts::{
    BcArtifactPaths, JsonlAppender, LatestCheckpointState, PersistedDeltaQPromotionArtifact,
    PersistedValidationGateArtifact, append_advisory_event_to_writer, append_step_log_to_writer,
    append_training_log_to_writer, log_tensorboard, save_checkpoint,
    save_latest_checkpoint_and_state, write_delta_q_promotion_artifact,
    write_validation_gate_artifact,
};
use super::config::{TrainConfig, shard_prefetch_depth};
use super::nvtx;
use super::presentation::{
    format_progress_message, make_bar, make_spinner, phase_label, timestamped,
};
use super::progress::{
    BatchMetricSums, BatchStats, EpochLogEntry, ScalarAverages, StepLogEntry,
    batch_metric_sums_from_outputs, batch_stats_from_metric_sums,
};
use super::resume::{
    BestValidation, EpochContinuation, RuntimeResumeContract, paused_training_message,
};
use super::schedule::{effective_lr, lr_status_message, steps_per_second};
use super::status::{
    display_step_label, display_validation_scope_label, epoch_progress_message_with_rate,
    estimate_epoch_progress, reached_session_step_budget, session_steps_completed,
};
use super::validation::{
    ValidationContext, ValidationRuntime, ValidationSummary, evaluate_validation_gates,
    is_better_validation, run_validation, validation_loader,
};
use hydra_train_exec::epoch_runner as exec_epoch;
pub(super) use hydra_train_exec::progress::TrainSubStageTiming;
use hydra_train_runtime::validation::ValidationRunLimits;

type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

pub(super) struct EpochRunnerContext<'a, B = TrainBackend>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    pub(super) epoch: usize,
    pub(super) config: &'a TrainConfig,
    pub(super) manifest: &'a DataManifest,
    pub(super) loader_config: &'a StreamingLoaderConfig,
    pub(super) artifacts: &'a BcArtifactPaths,
    pub(super) train_cfg: &'a BCTrainerConfig,
    pub(super) loss_fn: &'a HydraLoss<B>,
    pub(super) valid_loss_fn: &'a HydraLoss<ValidBackendOf<B>>,
    pub(super) bc_exit_cfg: &'a BcExitConfig,
    pub(super) train_device: &'a LibTorchDevice,
    pub(super) session_start_global_step: usize,
    pub(super) steps_to_skip: usize,
    pub(super) microbatch_size: usize,
    pub(super) use_amp: bool,
    pub(super) total_steps: usize,
    pub(super) current_runtime: RuntimeResumeContract,
    pub(super) run_start: &'a Instant,
    pub(super) head_controller: &'a mut HeadActivationController,
    pub(super) cached_validation_samples: Option<&'a [Box<[MjaiSample]>]>,
}

pub(super) struct EpochRuntimeMut<'a, O, W, B = TrainBackend>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    O: Optimizer<HydraModel<B>, B>,
    W: Write,
{
    pub(super) model: &'a mut Option<HydraModel<B>>,
    pub(super) optimizer: &'a mut O,
    pub(super) global_step: &'a mut usize,
    pub(super) best_validation: &'a mut Option<BestValidation>,
    pub(super) tb: &'a mut Option<EventWriter<W>>,
    pub(super) training_log: &'a mut JsonlAppender,
    pub(super) step_log: &'a mut JsonlAppender,
    pub(super) last_log_step: &'a mut usize,
    pub(super) last_log_time: &'a mut Instant,
}

pub(super) struct EpochRunOutcome {
    pub(super) stop_after_epoch: bool,
}

pub(super) type TrainLogicalBatchConfig<'a, B = TrainBackend> =
    exec_epoch::TrainLogicalBatchConfig<'a, B>;

struct ValidationStepContext<'a, B = TrainBackend>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    multi: &'a MultiProgress,
    config: &'a TrainConfig,
    loader_config: &'a StreamingLoaderConfig,
    manifest: &'a DataManifest,
    train_device: &'a LibTorchDevice,
    valid_loss_fn: &'a HydraLoss<ValidBackendOf<B>>,
    bc_exit_cfg: &'a BcExitConfig,
    artifacts: &'a BcArtifactPaths,
    session_start_global_step: usize,
    cached_validation_samples: Option<&'a [Box<[MjaiSample]>]>,
}

struct IntervalStepSummaryContext<'a> {
    manifest: &'a DataManifest,
    config: &'a TrainConfig,
    session_start_global_step: usize,
    global_step: usize,
    epoch: usize,
    lr: f64,
    best_validation: Option<BestValidation>,
    val_summary: Option<ValidationSummary>,
    seen_samples: usize,
    assumed_games_seen: usize,
    epoch_optimizer_steps: usize,
    window_stats: ScalarAverages,
    step_rate: f64,
    profiling: Option<ProfilingEnvelope>,
    advisories: Vec<RuntimeAdvisory>,
}

struct PeriodicCheckpointContext<'a> {
    config: &'a TrainConfig,
    artifacts: &'a BcArtifactPaths,
    epoch: usize,
    session_start_global_step: usize,
    current_runtime: RuntimeResumeContract,
}

struct PeriodicCheckpointState {
    global_step: usize,
    epoch_optimizer_steps: usize,
    total_loss: f64,
    best_validation: Option<BestValidation>,
}

fn epoch_model<B>(model_slot: &Option<HydraModel<B>>) -> Result<&HydraModel<B>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    model_slot
        .as_ref()
        .ok_or_else(|| "epoch model missing after logical batch execution".to_string())
}

fn should_save_periodic_checkpoint(
    config: &TrainConfig,
    global_step: usize,
    session_start_global_step: usize,
) -> bool {
    exec_epoch::should_save_periodic_checkpoint(
        &exec_epoch::EpochCadenceInput::from(config),
        global_step,
        session_start_global_step,
    )
}

fn should_refresh_train_progress_message(
    config: &TrainConfig,
    global_step: usize,
    session_start_global_step: usize,
) -> bool {
    exec_epoch::should_refresh_train_progress_message(
        &exec_epoch::EpochCadenceInput::from(config),
        global_step,
        session_start_global_step,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "progress message call mirrors training context"
)]
fn update_train_progress_message(
    train_pb: &ProgressBar,
    config: &TrainConfig,
    train_cfg: &BCTrainerConfig,
    global_step: usize,
    session_start_global_step: usize,
    run_start: Instant,
    lr: f64,
    stats: ScalarAverages,
) {
    if !should_refresh_train_progress_message(config, global_step, session_start_global_step) {
        return;
    }
    let lr_message = lr_status_message(global_step, train_cfg.warmup_steps, lr);
    train_pb.set_message(format_progress_message(
        stats.total_loss,
        stats.policy_agreement,
        &lr_message,
        steps_per_second(
            session_steps_completed(global_step, session_start_global_step),
            run_start.elapsed(),
        ),
    ));
}

struct EpochEndValidationContext<'a, B = TrainBackend>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    config: &'a TrainConfig,
    loader_config: &'a StreamingLoaderConfig,
    manifest: &'a DataManifest,
    train_device: &'a LibTorchDevice,
    valid_loss_fn: &'a HydraLoss<ValidBackendOf<B>>,
    bc_exit_cfg: &'a BcExitConfig,
    artifacts: &'a BcArtifactPaths,
    cached_validation_samples: Option<&'a [Box<[MjaiSample]>]>,
}

#[derive(Clone)]
struct ValidationEvent {
    global_step: usize,
    summary: ValidationSummary,
}

trait ValidationExecutor<B>
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
            TrainConfig,
            super::validation::TrainValidationLoader<'_>,
            ValidBackendOf<B>,
        >,
        runtime: ValidationRuntime<'_>,
    ) -> Result<ValidationSummary, String>;
}

struct DefaultValidationExecutor;

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
            TrainConfig,
            super::validation::TrainValidationLoader<'_>,
            ValidBackendOf<B>,
        >,
        runtime: ValidationRuntime<'_>,
    ) -> Result<ValidationSummary, String> {
        run_validation(model, context, runtime)
    }
}

struct EpochFinalizeContext<'a> {
    config: &'a TrainConfig,
    train_cfg: &'a BCTrainerConfig,
    epoch: usize,
    global_step: usize,
    train_stats: ScalarAverages,
    val_summary: Option<ValidationSummary>,
    best_validation: Option<BestValidation>,
    final_lr: f64,
    profiling: Option<ProfilingEnvelope>,
}

struct CompletedValidationContext<'a, B = TrainBackend>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<
            Device = LibTorchDevice,
            FloatTensorPrimitive = TchTensor,
            IntTensorPrimitive = TchTensor,
        >,
{
    model: &'a HydraModel<B>,
    artifacts: &'a BcArtifactPaths,
    config: &'a TrainConfig,
    best_validation: &'a mut Option<BestValidation>,
    checkpoint_index: usize,
    checkpoint_loss: f64,
    delta_q_scope: &'static str,
}

fn should_run_epoch_end_validation(epoch: usize, num_epochs: usize, every_n_epochs: usize) -> bool {
    exec_epoch::should_run_epoch_end_validation(epoch, num_epochs, every_n_epochs)
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

fn finalize_completed_validation<B>(
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
    let run_config = hydra_train_runtime::validation::ValidationRunConfig::from_config(config);
    let gate_decision = evaluate_validation_gates(&run_config, &summary, previous_best);
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
    if is_better_validation(&summary, *best_validation) && gate_decision.passed {
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

fn build_epoch_continuation(
    epoch: usize,
    epoch_completed: bool,
    epoch_optimizer_steps: usize,
) -> EpochContinuation {
    exec_epoch::build_epoch_continuation(epoch, epoch_completed, epoch_optimizer_steps)
}

fn merge_optional_profiling(
    target: &mut Option<ProfilingEnvelope>,
    source: Option<&ProfilingEnvelope>,
) {
    exec_epoch::merge_optional_profiling(target, source);
}

fn bc_interval_profiling(
    train_seconds: f64,
    sub_timing: &TrainSubStageTiming,
    validation: Option<ProfilingEnvelope>,
    checkpoint_seconds: f64,
) -> ProfilingEnvelope {
    exec_epoch::bc_interval_profiling(train_seconds, sub_timing, validation, checkpoint_seconds)
}

fn bc_epoch_profiling(
    train_seconds: f64,
    sub_timing: &TrainSubStageTiming,
    validation: Option<ProfilingEnvelope>,
    checkpoint_seconds: f64,
    logging_seconds: f64,
) -> ProfilingEnvelope {
    exec_epoch::bc_epoch_profiling(
        train_seconds,
        sub_timing,
        validation,
        checkpoint_seconds,
        logging_seconds,
    )
}

pub(super) fn train_logical_batch<B, O>(
    logical_batch: &[MjaiSample],
    config: exec_epoch::TrainLogicalBatchConfig<'_, B>,
    head_controller: &mut HeadActivationController,
    model_slot: &mut Option<HydraModel<B>>,
    optimizer: &mut O,
) -> Result<(Vec<BatchStats>, TrainSubStageTiming), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
    O: Optimizer<HydraModel<B>, B>,
{
    exec_epoch::train_logical_batch(
        logical_batch,
        config,
        head_controller,
        model_slot,
        optimizer,
    )
}

fn record_drained_batch_stats(
    drained: Vec<BatchStats>,
    stats: &mut ScalarAverages,
    step_window: &mut ScalarAverages,
) {
    exec_epoch::record_drained_batch_stats(drained, stats, step_window);
}

fn maybe_run_interval_validation<B>(
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

fn maybe_run_interval_validation_with_executor<B, E>(
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

#[cfg(test)]
fn child_elapsed_seconds(profiling: &ProfilingEnvelope, stage: &str) -> f64 {
    exec_epoch::child_elapsed_seconds(profiling, stage)
}

fn interval_timing_input(
    config: &TrainConfig,
    profiling: &ProfilingEnvelope,
    window_steps: usize,
) -> IntervalTimingInput {
    exec_epoch::interval_timing_input(
        &config.device,
        config
            .nsight_trace
            .as_ref()
            .and_then(|trace| trace.kernel_launch_count),
        config
            .nsight_trace
            .as_ref()
            .and_then(|trace| trace.tiny_kernel_fraction),
        config
            .nsight_trace
            .as_ref()
            .and_then(|trace| trace.cuda_runtime_launch_seconds),
        profiling,
        window_steps,
    )
}

fn emit_interval_step_summary<W>(
    multi: &MultiProgress,
    tb: &mut Option<EventWriter<W>>,
    step_log: &mut JsonlAppender,
    context: IntervalStepSummaryContext<'_>,
) -> Result<(), String>
where
    W: Write,
{
    let _logging_scope = nvtx::scope(PROFILING_STAGE_LOGGING);
    let IntervalStepSummaryContext {
        manifest,
        config,
        session_start_global_step,
        global_step,
        epoch,
        lr,
        best_validation,
        val_summary,
        seen_samples,
        assumed_games_seen,
        epoch_optimizer_steps,
        window_stats,
        step_rate,
        mut profiling,
        advisories,
    } = context;
    let logging_started = Instant::now();
    multi
        .println(timestamped(format!(
            "{} {} {} {} {} {} {} {}",
            display_step_label(
                global_step,
                session_start_global_step,
                config.max_train_steps
            )
            .bold()
            .cyan(),
            format!("train_loss={:.4}", window_stats.total_loss).green(),
            format!("train_agree={:.2}%", window_stats.policy_agreement * 100.0).green(),
            if let Some(val_summary) = val_summary.as_ref() {
                format!(
                    "val_ce={:.4} val_agree={:.2}%",
                    val_summary.policy_loss,
                    val_summary.agreement * 100.0
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
            epoch_progress_message_with_rate(
                estimate_epoch_progress(
                    manifest,
                    seen_samples,
                    assumed_games_seen,
                    epoch_optimizer_steps,
                    config.batch_size,
                ),
                Some(step_rate),
            )
            .white(),
            format!("steps/s={step_rate:.2}").white(),
            lr_status_message(global_step, config.bc.warmup_steps, lr).white(),
        )))
        .map_err(|err| format!("failed to print train summary: {err}"))?;

    if let Some(ref mut tb_writer) = tb.as_mut() {
        log_tensorboard(
            tb_writer,
            global_step,
            &window_stats,
            val_summary.as_ref(),
            lr,
            best_validation,
        )?;
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
            PROFILING_STAGE_BC_INTERVAL,
            vec![ProfilingEnvelope::leaf(
                PROFILING_STAGE_LOGGING,
                logging_seconds,
            )],
        ));
    }

    let step_entry = StepLogEntry {
        global_step,
        epoch: epoch + 1,
        lr,
        train_total_loss: window_stats.total_loss,
        train_policy_agreement: window_stats.policy_agreement,
        train_loss_policy: window_stats.loss_policy,
        train_loss_value: window_stats.loss_value,
        train_loss_grp: window_stats.loss_grp,
        train_loss_tenpai: window_stats.loss_tenpai,
        train_loss_danger: window_stats.loss_danger,
        train_loss_opp_next: window_stats.loss_opp_next,
        train_loss_score_pdf: window_stats.loss_score_pdf,
        train_loss_score_cdf: window_stats.loss_score_cdf,
        train_rare_actions: window_stats.rare_actions,
        val_rare_actions: val_summary.as_ref().map(|summary| summary.rare_actions),
        val_total_loss: val_summary.as_ref().map(|summary| summary.total_loss),
        val_policy_loss: val_summary.as_ref().map(|summary| summary.policy_loss),
        val_policy_agreement: val_summary.as_ref().map(|summary| summary.agreement),
        val_delta_q_promotion: val_summary
            .as_ref()
            .and_then(|summary| summary.delta_q_promotion_snapshot),
        profiling,
        advisories,
        best_val_policy_loss: best_validation.map(|best| best.policy_loss),
        best_val_agreement: best_validation.map(|best| best.agreement),
    };
    append_step_log_to_writer(step_log, &step_entry)?;
    if !step_entry.advisories.is_empty() {
        append_advisory_event_to_writer(
            step_log,
            &AdvisoryEvent::interval(&step_entry.advisories),
        )?;
    }
    Ok(())
}

fn maybe_save_periodic_checkpoint<B, O>(
    model: &HydraModel<B>,
    optimizer: &O,
    context: PeriodicCheckpointContext<'_>,
    state: PeriodicCheckpointState,
) -> Result<f64, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    O: Optimizer<HydraModel<B>, B>,
{
    let PeriodicCheckpointContext {
        config,
        artifacts,
        epoch,
        session_start_global_step,
        current_runtime,
    } = context;
    let PeriodicCheckpointState {
        global_step,
        epoch_optimizer_steps,
        total_loss,
        best_validation,
    } = state;
    let session_step = session_steps_completed(global_step, session_start_global_step);
    if session_step == 0 || !session_step.is_multiple_of(config.checkpoint_every_n_steps) {
        return Ok(0.0);
    }

    let continuation = EpochContinuation {
        next_epoch: epoch,
        skip_optimizer_steps_in_epoch: epoch_optimizer_steps,
        epoch_completed: false,
    };
    let checkpoint_started = Instant::now();
    {
        let _checkpoint_scope = nvtx::scope(PROFILING_STAGE_CHECKPOINT);
        save_latest_checkpoint_and_state(
            artifacts,
            model,
            optimizer,
            LatestCheckpointState {
                global_step,
                train_loss: total_loss,
                best_validation,
                continuation: &continuation,
                runtime: current_runtime,
            },
        )?;
    }
    Ok(checkpoint_started.elapsed().as_secs_f64())
}

fn emit_paused_training_message(continuation: &EpochContinuation) {
    if !benchmark_quiet() {
        println!(
            "{}",
            timestamped(format!(
                "{} {}",
                "Paused BC training".bold().cyan(),
                paused_training_message(continuation).yellow(),
            ))
        );
    }
}

fn benchmark_quiet() -> bool {
    std::env::var_os("HYDRA_BENCHMARK_QUIET").is_some()
}

fn run_epoch_end_validation<B>(
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

fn run_epoch_end_validation_with_executor<B, E>(
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

fn finalize_epoch_outputs<W>(
    tb: &mut Option<EventWriter<W>>,
    training_log: &mut JsonlAppender,
    context: EpochFinalizeContext<'_>,
) -> Result<(), String>
where
    W: Write,
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

    let lr_message = lr_status_message(global_step, train_cfg.warmup_steps, final_lr);
    if !benchmark_quiet() {
        println!(
            "{}",
            timestamped(format!(
                "{} {} {} {} {} {}",
                phase_label("epoch", epoch, config.num_epochs).bold().cyan(),
                format!("train_loss={:.4}", train_stats.total_loss).green(),
                format!("train_agree={:.2}%", train_stats.policy_agreement * 100.0).green(),
                if let Some(val_summary) = val_summary.as_ref() {
                    format!(
                        "val_ce={:.4} val_agree={:.2}% val_samples={}",
                        val_summary.policy_loss,
                        val_summary.agreement * 100.0,
                        val_summary.samples
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

    let entry = EpochLogEntry {
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
        val_rare_actions: val_summary.as_ref().map(|summary| summary.rare_actions),
        val_total_loss: val_summary.as_ref().map(|summary| summary.total_loss),
        val_policy_loss: val_summary.as_ref().map(|summary| summary.policy_loss),
        val_policy_agreement: val_summary.as_ref().map(|summary| summary.agreement),
        val_delta_q_promotion: val_summary
            .as_ref()
            .and_then(|summary| summary.delta_q_promotion_snapshot),
        profiling,
        advisories: Vec::new(),
        best_val_policy_loss: best_validation.map(|best| best.policy_loss),
        best_val_agreement: best_validation.map(|best| best.agreement),
        num_batches: train_stats.num_batches,
    };
    append_training_log_to_writer(training_log, &entry)?;

    Ok(())
}

pub(super) fn run_epoch<B, O, W>(
    context: EpochRunnerContext<'_, B>,
    runtime: EpochRuntimeMut<'_, O, W, B>,
) -> Result<EpochRunOutcome, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<FloatTensorPrimitive = TchTensor, IntTensorPrimitive = TchTensor>,
    O: Optimizer<HydraModel<B>, B>,
    W: Write,
{
    if context.config.bc_shards_manifest_path.is_some() {
        return run_epoch_from_shards(context, runtime);
    }
    let _epoch_scope = nvtx::scope(PROFILING_STAGE_BC_EPOCH);
    let EpochRunnerContext {
        epoch,
        config,
        manifest,
        loader_config,
        artifacts,
        train_cfg,
        loss_fn,
        valid_loss_fn,
        bc_exit_cfg,
        train_device,
        session_start_global_step,
        steps_to_skip,
        microbatch_size,
        use_amp,
        total_steps,
        current_runtime,
        run_start,
        head_controller,
        cached_validation_samples,
    } = context;
    let EpochRuntimeMut {
        model: model_slot,
        optimizer,
        global_step,
        best_validation,
        tb,
        training_log,
        step_log,
        last_log_step,
        last_log_time,
    } = runtime;

    let multi = MultiProgress::new();
    let load_label = phase_label("load", epoch, config.num_epochs);
    let train_label = phase_label("train", epoch, config.num_epochs);
    let load_pb = if manifest.counts_exact {
        multi.add(make_bar(
            manifest.train_count as u64,
            &format!("[{load_label}] [{{bar:30.cyan/blue}}] {{pos}}/{{len}} games {{msg}}"),
        )?)
    } else {
        multi.add(make_spinner(&format!(
            "[{load_label}] {{spinner:.cyan}} games={{pos}} {{msg}}"
        ))?)
    };
    let train_pb = if let Some(max_train_steps) = config.max_train_steps {
        multi.add(make_bar(
            max_train_steps as u64,
            &format!("[{train_label}] [{{bar:30.green/black}}] {{pos}}/{{len}} steps {{msg}}"),
        )?)
    } else {
        multi.add(make_spinner(&format!(
            "[{train_label}] {{spinner:.green}} steps={{pos}} {{msg}}"
        ))?)
    };

    let mut stats = ScalarAverages::default();
    let mut step_window = ScalarAverages::default();
    let mut pending_samples = VecDeque::new();
    let samples_to_skip = steps_to_skip.saturating_mul(config.batch_size);
    let mut samples_skipped = 0usize;
    let mut seen_samples = 0usize;
    let mut epoch_completed = true;
    let mut assumed_games_seen = 0usize;
    let mut remaining_games = manifest.train_count;
    let mut epoch_optimizer_steps = steps_to_skip;
    let mut last_interval_validation: Option<ValidationEvent> = None;
    let epoch_started = Instant::now();
    let mut step_window_train_seconds = 0.0;
    let mut step_window_checkpoint_seconds = 0.0;
    let mut step_window_validation_profiling: Option<ProfilingEnvelope> = None;
    let mut epoch_train_seconds = 0.0;
    let mut epoch_checkpoint_seconds = 0.0;
    let mut epoch_validation_profiling: Option<ProfilingEnvelope> = None;
    let mut step_window_sub_timing = TrainSubStageTiming::default();
    let mut epoch_sub_timing = TrainSubStageTiming::default();

    for buffer_result in stream_train_epoch(manifest, loader_config, epoch, Some(&load_pb)) {
        let buffer = buffer_result.map_err(|err| format!("training stream failed: {err}"))?;
        if manifest.counts_exact {
            let assumed_games = remaining_games.min(config.buffer_games);
            remaining_games = remaining_games.saturating_sub(assumed_games);
            assumed_games_seen += assumed_games;
        }
        seen_samples += buffer.len();
        if manifest.counts_exact && assumed_games_seen > 0 {
            let estimated_steps = estimate_epoch_progress(
                manifest,
                seen_samples,
                assumed_games_seen,
                epoch_optimizer_steps,
                config.batch_size,
            )
            .map(|progress| progress.estimated_total_optimizer_steps)
            .unwrap_or(1);
            if config.max_train_steps.is_none() {
                train_pb.set_length(estimated_steps as u64);
            }
        } else if !manifest.counts_exact {
            load_pb.set_message(format!(
                "samples={} steps={}",
                seen_samples, epoch_optimizer_steps
            ));
        }

        pending_samples.extend(buffer);
        if samples_skipped < samples_to_skip {
            let skip_now = (samples_to_skip - samples_skipped).min(pending_samples.len());
            pending_samples.drain(..skip_now);
            samples_skipped += skip_now;
        }

        while pending_samples.len() >= config.batch_size {
            let lr = effective_lr(train_cfg, *global_step, total_steps);
            let logical_batch: Vec<MjaiSample> =
                pending_samples.drain(..config.batch_size).collect();
            let train_started = Instant::now();
            let (drained, batch_sub_timing) = {
                let _train_scope = nvtx::scope(PROFILING_STAGE_TRAIN);
                train_logical_batch(
                    &logical_batch,
                    TrainLogicalBatchConfig {
                        microbatch_size,
                        use_amp,
                        augment: config.augment,
                        train_device,
                        loss_fn,
                        bc_exit_cfg,
                        lr,
                    },
                    head_controller,
                    model_slot,
                    optimizer,
                )?
            };
            let train_seconds = train_started.elapsed().as_secs_f64();

            record_drained_batch_stats(drained, &mut stats, &mut step_window);
            step_window_train_seconds += train_seconds;
            epoch_train_seconds += train_seconds;
            step_window_sub_timing.accumulate(&batch_sub_timing);
            epoch_sub_timing.accumulate(&batch_sub_timing);
            epoch_optimizer_steps += 1;
            *global_step += 1;
            train_pb.inc(1);
            if should_refresh_train_progress_message(
                config,
                *global_step,
                session_start_global_step,
            ) {
                update_train_progress_message(
                    &train_pb,
                    config,
                    train_cfg,
                    *global_step,
                    session_start_global_step,
                    *run_start,
                    lr,
                    stats.finalize(),
                );
            }

            let session_step = session_steps_completed(*global_step, session_start_global_step);
            let val_summary = maybe_run_interval_validation(
                ValidationStepContext {
                    multi: &multi,
                    config,
                    loader_config,
                    manifest,
                    train_device,
                    valid_loss_fn,
                    bc_exit_cfg,
                    artifacts,
                    session_start_global_step,
                    cached_validation_samples,
                },
                epoch_model(model_slot)?,
                Some(head_controller),
                best_validation,
                *global_step,
                step_window.finalize().total_loss,
            )?;
            if let Some(summary) = val_summary.clone() {
                merge_optional_profiling(
                    &mut step_window_validation_profiling,
                    summary.profiling.as_ref(),
                );
                merge_optional_profiling(
                    &mut epoch_validation_profiling,
                    summary.profiling.as_ref(),
                );
                last_interval_validation = Some(ValidationEvent {
                    global_step: *global_step,
                    summary,
                });
            }

            if session_step > 0 && session_step.is_multiple_of(config.log_every_n_steps) {
                let window_stats = std::mem::take(&mut step_window).finalize();
                let window_steps = (*global_step).saturating_sub(*last_log_step);
                let step_rate = steps_per_second(window_steps, last_log_time.elapsed());
                *last_log_step = *global_step;
                *last_log_time = Instant::now();
                let interval_profiling = bc_interval_profiling(
                    step_window_train_seconds,
                    &step_window_sub_timing,
                    step_window_validation_profiling.take(),
                    step_window_checkpoint_seconds,
                );
                step_window_train_seconds = 0.0;
                step_window_checkpoint_seconds = 0.0;
                step_window_sub_timing = TrainSubStageTiming::default();

                emit_interval_step_summary(
                    &multi,
                    tb,
                    step_log,
                    IntervalStepSummaryContext {
                        manifest,
                        config,
                        session_start_global_step,
                        global_step: *global_step,
                        epoch,
                        lr,
                        best_validation: *best_validation,
                        val_summary,
                        seen_samples,
                        assumed_games_seen,
                        epoch_optimizer_steps,
                        window_stats,
                        step_rate,
                        profiling: Some(interval_profiling.clone()),
                        advisories: interval_runtime_advisories(interval_timing_input(
                            config,
                            &interval_profiling,
                            window_steps,
                        )),
                    },
                )?;
            }

            let periodic_checkpoint_seconds =
                if should_save_periodic_checkpoint(config, *global_step, session_start_global_step)
                {
                    maybe_save_periodic_checkpoint(
                        epoch_model(model_slot)?,
                        optimizer,
                        PeriodicCheckpointContext {
                            config,
                            artifacts,
                            epoch,
                            session_start_global_step,
                            current_runtime,
                        },
                        PeriodicCheckpointState {
                            global_step: *global_step,
                            epoch_optimizer_steps,
                            total_loss: stats.finalize().total_loss,
                            best_validation: *best_validation,
                        },
                    )?
                } else {
                    0.0
                };
            step_window_checkpoint_seconds += periodic_checkpoint_seconds;
            epoch_checkpoint_seconds += periodic_checkpoint_seconds;

            if reached_session_step_budget(
                *global_step,
                session_start_global_step,
                config.max_train_steps,
            ) {
                epoch_completed = false;
                break;
            }
        }

        if reached_session_step_budget(
            *global_step,
            session_start_global_step,
            config.max_train_steps,
        ) {
            epoch_completed = false;
            break;
        }
    }

    if !pending_samples.is_empty() && epoch_completed {
        let lr = effective_lr(train_cfg, *global_step, total_steps);
        let logical_batch: Vec<MjaiSample> = pending_samples.drain(..).collect();
        let train_started = Instant::now();
        let (drained, batch_sub_timing) = {
            let _train_scope = nvtx::scope(PROFILING_STAGE_TRAIN);
            train_logical_batch(
                &logical_batch,
                TrainLogicalBatchConfig {
                    microbatch_size,
                    use_amp,
                    augment: config.augment,
                    train_device,
                    loss_fn,
                    bc_exit_cfg,
                    lr,
                },
                head_controller,
                model_slot,
                optimizer,
            )?
        };
        let train_seconds = train_started.elapsed().as_secs_f64();
        record_drained_batch_stats(drained, &mut stats, &mut step_window);
        epoch_train_seconds += train_seconds;
        epoch_sub_timing.accumulate(&batch_sub_timing);
        epoch_optimizer_steps += 1;
        *global_step += 1;
        train_pb.inc(1);
    }

    load_pb.finish_with_message("training data stream complete".to_string());
    let train_stats = stats.finalize();
    let final_steps = config.max_train_steps.unwrap_or(*global_step).max(1) as u64;
    let final_lr = effective_lr(train_cfg, *global_step, total_steps);
    train_pb.set_length(final_steps);
    train_pb.finish_with_message(format_progress_message(
        train_stats.total_loss,
        train_stats.policy_agreement,
        &lr_status_message(*global_step, train_cfg.warmup_steps, final_lr),
        steps_per_second(
            session_steps_completed(*global_step, session_start_global_step),
            run_start.elapsed(),
        ),
    ));

    let continuation = build_epoch_continuation(epoch, epoch_completed, epoch_optimizer_steps);
    let checkpoint_started = Instant::now();
    {
        let _checkpoint_scope = nvtx::scope(PROFILING_STAGE_CHECKPOINT);
        save_latest_checkpoint_and_state(
            artifacts,
            epoch_model(model_slot)?,
            optimizer,
            LatestCheckpointState {
                global_step: *global_step,
                train_loss: train_stats.total_loss,
                best_validation: *best_validation,
                continuation: &continuation,
                runtime: current_runtime,
            },
        )?;
    }
    let checkpoint_seconds = checkpoint_started.elapsed().as_secs_f64();

    if !continuation.epoch_completed {
        emit_paused_training_message(&continuation);
        return Ok(EpochRunOutcome {
            stop_after_epoch: true,
        });
    }

    let reused_interval_validation =
        last_interval_validation
            .as_ref()
            .is_some_and(|last_validation| {
                last_validation.global_step == *global_step
                    && should_run_epoch_end_validation(
                        epoch,
                        config.num_epochs,
                        config.validation_every_n_epochs,
                    )
            });
    let val_summary = if reused_interval_validation {
        last_interval_validation
            .as_ref()
            .map(|last_validation| last_validation.summary.clone())
    } else {
        run_epoch_end_validation(
            epoch,
            epoch_model(model_slot)?,
            EpochEndValidationContext {
                config,
                loader_config,
                manifest,
                train_device,
                valid_loss_fn,
                bc_exit_cfg,
                artifacts,
                cached_validation_samples,
            },
            Some(head_controller),
            best_validation,
            train_stats.total_loss,
        )?
    };
    let epoch_elapsed_seconds = epoch_started.elapsed().as_secs_f64();
    if !reused_interval_validation {
        merge_optional_profiling(
            &mut epoch_validation_profiling,
            val_summary
                .as_ref()
                .and_then(|summary| summary.profiling.as_ref()),
        );
    }
    let mut epoch_profiling = bc_epoch_profiling(
        epoch_train_seconds,
        &epoch_sub_timing,
        epoch_validation_profiling,
        epoch_checkpoint_seconds + checkpoint_seconds,
        0.0,
    );
    epoch_profiling.elapsed_seconds = epoch_elapsed_seconds;

    finalize_epoch_outputs(
        tb,
        training_log,
        EpochFinalizeContext {
            config,
            train_cfg,
            epoch,
            global_step: *global_step,
            train_stats,
            val_summary,
            best_validation: *best_validation,
            final_lr,
            profiling: Some(epoch_profiling),
        },
    )?;

    Ok(EpochRunOutcome {
        stop_after_epoch: reached_session_step_budget(
            *global_step,
            session_start_global_step,
            config.max_train_steps,
        ),
    })
}

/// Default prefetch queue depth for the CPU producer thread.
///
/// Depth 2 keeps at most two host batches resident in memory while the GPU
/// processes the current one. User config may raise this conservatively.
fn run_epoch_from_shards<B, O, W>(
    context: EpochRunnerContext<'_, B>,
    runtime: EpochRuntimeMut<'_, O, W, B>,
) -> Result<EpochRunOutcome, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<FloatTensorPrimitive = TchTensor, IntTensorPrimitive = TchTensor>,
    O: Optimizer<HydraModel<B>, B>,
    W: Write,
{
    let _epoch_scope = nvtx::scope(PROFILING_STAGE_BC_EPOCH);
    let EpochRunnerContext {
        epoch,
        config,
        manifest,
        loader_config,
        artifacts,
        train_cfg,
        loss_fn,
        valid_loss_fn,
        bc_exit_cfg,
        train_device,
        session_start_global_step,
        steps_to_skip,
        microbatch_size,
        use_amp,
        total_steps,
        current_runtime,
        run_start,
        head_controller,
        cached_validation_samples,
    } = context;
    let EpochRuntimeMut {
        model: model_slot,
        optimizer,
        global_step,
        best_validation,
        tb,
        training_log,
        step_log,
        last_log_step,
        last_log_time,
    } = runtime;

    let shard_manifest_path = config
        .bc_shards_manifest_path
        .as_ref()
        .ok_or_else(|| "bc_shards_manifest_path missing for shard epoch path".to_string())?;
    let reader = load_bc_shard_reader(shard_manifest_path, BcShardSplit::Train)?;
    let total_rows = reader.sample_count();
    let samples_to_skip = steps_to_skip
        .saturating_mul(config.batch_size)
        .min(total_rows);

    let multi = MultiProgress::new();
    let train_label = phase_label("train", epoch, config.num_epochs);
    let estimated_steps =
        ((total_rows.saturating_sub(samples_to_skip)) / config.batch_size.max(1)).max(1);
    let train_pb = if let Some(max_train_steps) = config.max_train_steps {
        multi.add(make_bar(
            max_train_steps as u64,
            &format!("[{train_label}] [{{bar:30.green/black}}] {{pos}}/{{len}} steps {{msg}}"),
        )?)
    } else {
        multi.add(make_bar(
            estimated_steps as u64,
            &format!("[{train_label}] [{{bar:30.green/black}}] {{pos}}/{{len}} steps {{msg}}"),
        )?)
    };

    let mut stats = ScalarAverages::default();
    let mut step_window = ScalarAverages::default();
    let mut epoch_completed = true;
    let mut epoch_optimizer_steps = steps_to_skip;
    let mut last_interval_validation: Option<ValidationEvent> = None;
    let mut step_window_train_seconds = 0.0;
    let mut step_window_checkpoint_seconds = 0.0;
    let mut step_window_validation_profiling: Option<ProfilingEnvelope> = None;
    let mut epoch_train_seconds = 0.0;
    let mut epoch_checkpoint_seconds = 0.0;
    let mut epoch_validation_profiling: Option<ProfilingEnvelope> = None;
    let mut step_window_sub_timing = TrainSubStageTiming::default();
    let mut epoch_sub_timing = TrainSubStageTiming::default();
    let mut seen_samples = samples_to_skip;

    // -- async H2D staging for pinned memory + dedicated copy stream --
    #[cfg(feature = "cuda-graph")]
    let mut staging_context = match train_device {
        LibTorchDevice::Cuda(idx) => {
            let device_index = *idx as i64;
            Some((
                super::pinned_transfer::PinnedStagingArea::new(config.batch_size),
                super::pinned_transfer::AsyncH2DContext::new(device_index),
                super::pinned_transfer::PreallocatedDeviceTensors::new(
                    config.batch_size,
                    train_device,
                ),
            ))
        }
        _ => None,
    };

    // -- producer/consumer pipeline for CPU host-batch prefetch --
    let batch_size = config.batch_size;
    let augment = config.augment;
    let producer_start_index = samples_to_skip;
    let prefetch_depth = shard_prefetch_depth(config);
    let (tx, rx) = mpsc::sync_channel::<Result<(BcShardHostBatch, usize), String>>(prefetch_depth);
    // Recycle channel: consumer returns consumed batches so the producer
    // can swap their heap capacity back into the scratch, eliminating
    // 18+ per-batch allocations (including ~1.6MB for obs_flat).
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

    loop {
        let recv_started = Instant::now();
        let recv_result = match rx.recv() {
            Ok(result) => result,
            Err(_) => break,
        };
        let producer_wait_seconds = recv_started.elapsed().as_secs_f64();
        let (host_batch, take) = recv_result?;
        let lr = effective_lr(train_cfg, *global_step, total_steps);
        let train_started = Instant::now();
        let (drained, batch_sub_timing, recycled_host_batch) = {
            let _train_scope = nvtx::scope(PROFILING_STAGE_TRAIN);
            train_logical_batch_from_host_batch(
                host_batch,
                TrainLogicalBatchConfig {
                    microbatch_size,
                    use_amp,
                    augment: config.augment,
                    train_device,
                    loss_fn,
                    bc_exit_cfg,
                    lr,
                },
                head_controller,
                model_slot,
                optimizer,
                #[cfg(feature = "cuda-graph")]
                staging_context.as_mut(),
            )?
        };
        let train_seconds = train_started.elapsed().as_secs_f64();
        if let Some(host_batch) = recycled_host_batch {
            let _ = recycle_tx.try_send(host_batch);
        }

        record_drained_batch_stats(drained, &mut stats, &mut step_window);
        step_window_train_seconds += train_seconds;
        epoch_train_seconds += train_seconds;
        let mut batch_sub_timing = batch_sub_timing;
        batch_sub_timing.producer_wait_seconds += producer_wait_seconds;
        step_window_sub_timing.accumulate(&batch_sub_timing);
        epoch_sub_timing.accumulate(&batch_sub_timing);
        epoch_optimizer_steps += 1;
        *global_step += 1;
        seen_samples += take;
        train_pb.inc(1);

        if should_refresh_train_progress_message(config, *global_step, session_start_global_step) {
            update_train_progress_message(
                &train_pb,
                config,
                train_cfg,
                *global_step,
                session_start_global_step,
                *run_start,
                lr,
                stats.finalize(),
            );
        }

        let session_step = session_steps_completed(*global_step, session_start_global_step);
        let val_summary = maybe_run_interval_validation(
            ValidationStepContext {
                multi: &multi,
                config,
                loader_config,
                manifest,
                train_device,
                valid_loss_fn,
                bc_exit_cfg,
                artifacts,
                session_start_global_step,
                cached_validation_samples,
            },
            epoch_model(model_slot)?,
            Some(head_controller),
            best_validation,
            *global_step,
            step_window.finalize().total_loss,
        )?;
        if let Some(summary) = val_summary.clone() {
            merge_optional_profiling(
                &mut step_window_validation_profiling,
                summary.profiling.as_ref(),
            );
            merge_optional_profiling(&mut epoch_validation_profiling, summary.profiling.as_ref());
            last_interval_validation = Some(ValidationEvent {
                global_step: *global_step,
                summary,
            });
        }

        if session_step > 0 && session_step.is_multiple_of(config.log_every_n_steps) {
            let window_stats = std::mem::take(&mut step_window).finalize();
            let window_steps = (*global_step).saturating_sub(*last_log_step);
            let step_rate = steps_per_second(window_steps, last_log_time.elapsed());
            *last_log_step = *global_step;
            *last_log_time = Instant::now();
            let interval_profiling = bc_interval_profiling(
                step_window_train_seconds,
                &step_window_sub_timing,
                step_window_validation_profiling.take(),
                step_window_checkpoint_seconds,
            );
            step_window_train_seconds = 0.0;
            step_window_checkpoint_seconds = 0.0;
            step_window_sub_timing = TrainSubStageTiming::default();

            emit_interval_step_summary(
                &multi,
                tb,
                step_log,
                IntervalStepSummaryContext {
                    manifest,
                    config,
                    session_start_global_step,
                    global_step: *global_step,
                    epoch,
                    lr,
                    best_validation: *best_validation,
                    val_summary,
                    seen_samples,
                    assumed_games_seen: 0,
                    epoch_optimizer_steps,
                    window_stats,
                    step_rate,
                    profiling: Some(interval_profiling.clone()),
                    advisories: interval_runtime_advisories(interval_timing_input(
                        config,
                        &interval_profiling,
                        window_steps,
                    )),
                },
            )?;
        }

        let periodic_checkpoint_seconds =
            if should_save_periodic_checkpoint(config, *global_step, session_start_global_step) {
                maybe_save_periodic_checkpoint(
                    epoch_model(model_slot)?,
                    optimizer,
                    PeriodicCheckpointContext {
                        config,
                        artifacts,
                        epoch,
                        session_start_global_step,
                        current_runtime,
                    },
                    PeriodicCheckpointState {
                        global_step: *global_step,
                        epoch_optimizer_steps,
                        total_loss: stats.finalize().total_loss,
                        best_validation: *best_validation,
                    },
                )?
            } else {
                0.0
            };
        step_window_checkpoint_seconds += periodic_checkpoint_seconds;
        epoch_checkpoint_seconds += periodic_checkpoint_seconds;

        if reached_session_step_budget(
            *global_step,
            session_start_global_step,
            config.max_train_steps,
        ) {
            epoch_completed = false;
            break;
        }
    }

    drop(rx);
    drop(recycle_tx);

    // Join the producer thread; if it panicked, propagate as an error.
    producer_handle
        .join()
        .map_err(|_| "bc-shard-prefetch thread panicked".to_string())?;

    let train_total_loss = stats.finalize().total_loss;
    let continuation = build_epoch_continuation(epoch, epoch_completed, epoch_optimizer_steps);
    let checkpoint_started = Instant::now();
    {
        let _checkpoint_scope = nvtx::scope(PROFILING_STAGE_CHECKPOINT);
        save_latest_checkpoint_and_state(
            artifacts,
            epoch_model(model_slot)?,
            optimizer,
            LatestCheckpointState {
                global_step: *global_step,
                train_loss: train_total_loss,
                best_validation: *best_validation,
                continuation: &continuation,
                runtime: current_runtime,
            },
        )?;
    }
    let checkpoint_seconds = checkpoint_started.elapsed().as_secs_f64();
    epoch_checkpoint_seconds += checkpoint_seconds;
    if !continuation.epoch_completed {
        emit_paused_training_message(&continuation);
        return Ok(EpochRunOutcome {
            stop_after_epoch: true,
        });
    }

    let reused_interval_validation =
        last_interval_validation
            .as_ref()
            .is_some_and(|last_validation| {
                last_validation.global_step == *global_step
                    && should_run_epoch_end_validation(
                        epoch,
                        config.num_epochs,
                        config.validation_every_n_epochs,
                    )
            });
    let final_validation = if reused_interval_validation {
        last_interval_validation
            .as_ref()
            .map(|last_validation| last_validation.summary.clone())
    } else {
        run_epoch_end_validation(
            epoch,
            epoch_model(model_slot)?,
            EpochEndValidationContext {
                config,
                loader_config,
                manifest,
                train_device,
                valid_loss_fn,
                bc_exit_cfg,
                artifacts,
                cached_validation_samples,
            },
            Some(head_controller),
            best_validation,
            train_total_loss,
        )?
    };
    if !reused_interval_validation {
        merge_optional_profiling(
            &mut epoch_validation_profiling,
            final_validation
                .as_ref()
                .and_then(|summary| summary.profiling.as_ref()),
        );
    }

    let logging_started = Instant::now();
    let final_lr = effective_lr(train_cfg, *global_step, total_steps);
    let final_stats = std::mem::take(&mut stats).finalize();
    let epoch_profiling = bc_epoch_profiling(
        epoch_train_seconds,
        &epoch_sub_timing,
        epoch_validation_profiling,
        epoch_checkpoint_seconds,
        logging_started.elapsed().as_secs_f64(),
    );
    finalize_epoch_outputs(
        tb,
        training_log,
        EpochFinalizeContext {
            config,
            train_cfg,
            epoch,
            global_step: *global_step,
            train_stats: final_stats,
            val_summary: final_validation,
            best_validation: *best_validation,
            final_lr,
            profiling: Some(epoch_profiling),
        },
    )?;

    Ok(EpochRunOutcome {
        stop_after_epoch: !epoch_completed,
    })
}

/// Runs the forward/backward/optimizer step from a pre-built host batch.
///
/// The host batch was already collated on the CPU producer thread.
/// This function materializes it onto the device, then runs the same
/// microbatch accumulation + optimizer step as the original shard path.
///
/// When `staging` is `Some`, the host batch is staged into pinned memory
/// and the H2D transfer is issued on a dedicated copy stream with
/// event-based synchronization.
pub(super) fn train_logical_batch_from_host_batch<B, O>(
    host_batch: BcShardHostBatch,
    config: TrainLogicalBatchConfig<'_, B>,
    head_controller: &mut HeadActivationController,
    model_slot: &mut Option<HydraModel<B>>,
    optimizer: &mut O,
    #[cfg(feature = "cuda-graph")] staging: Option<&mut (
        super::pinned_transfer::PinnedStagingArea,
        super::pinned_transfer::AsyncH2DContext,
        super::pinned_transfer::PreallocatedDeviceTensors,
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
                let (shard_batch, h2d_timing) = super::pinned_transfer::materialize_staged_reuse::<B>(
                    &host_batch,
                    pinned_staging,
                    h2d_ctx,
                    train_device,
                    gpu_tensors,
                );
                sub_timing.h2d_pageable_to_pinned_seconds += h2d_timing.pageable_to_pinned_seconds;
                sub_timing.h2d_tensor_materialize_seconds += h2d_timing.tensor_materialize_seconds;
                sub_timing.h2d_stream_sync_seconds += h2d_timing.stream_sync_seconds;
                (shard_batch, Some(host_batch))
            } else {
                let t_materialize = Instant::now();
                let shard_batch = host_batch.materialize_owned::<B>(train_device);
                sub_timing.h2d_tensor_materialize_seconds += t_materialize.elapsed().as_secs_f64();
                (shard_batch, None)
            }
        }
        #[cfg(not(feature = "cuda-graph"))]
        {
            let t_materialize = Instant::now();
            let shard_batch = host_batch.materialize_owned::<B>(train_device);
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

#[cfg(feature = "cuda-graph")]
pub(super) fn probe_logical_batch_from_host_batch<B>(
    host_batch: BcShardHostBatch,
    config: TrainLogicalBatchConfig<'_, B>,
    head_controller: &mut HeadActivationController,
    model: &HydraModel<B>,
    #[cfg(feature = "cuda-graph")] staging: Option<&mut (
        super::pinned_transfer::PinnedStagingArea,
        super::pinned_transfer::AsyncH2DContext,
        super::pinned_transfer::PreallocatedDeviceTensors,
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
        #[cfg(feature = "cuda-graph")]
        {
            if let Some((pinned_staging, h2d_ctx, gpu_tensors)) = staging {
                let (shard_batch, h2d_timing) = super::pinned_transfer::materialize_staged_reuse::<B>(
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
                let shard_batch = host_batch.materialize_owned::<B>(train_device);
                sub_timing.h2d_tensor_materialize_seconds += t_materialize.elapsed().as_secs_f64();
                shard_batch
            }
        }
        #[cfg(not(feature = "cuda-graph"))]
        {
            let t_materialize = Instant::now();
            let shard_batch = host_batch.materialize_owned::<B>(train_device);
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

#[cfg(feature = "cuda-graph")]
pub(super) fn probe_device_batch_compute<B>(
    shard_batch: BcShardBatch<B>,
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

#[cfg(feature = "cuda-graph")]
pub(super) fn probe_device_batch_compute_no_stats<B>(
    shard_batch: BcShardBatch<B>,
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::path::{Path, PathBuf};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn::backend::libtorch::LibTorchDevice;
    use burn::optim::AdamConfig;
    use hydra_train::model::HydraModelConfig;
    use hydra_train::preflight::PreflightConfig;
    use hydra_train::training::head_gates::{HeadActivationConfig, HeadActivationController};
    use hydra_train::training::losses::HydraLossConfig;

    use crate::config::{BcHyperparamConfig, TrainConfig};
    use crate::resume::read_resume_state;

    type TestValidBackend = ValidBackendOf<TrainBackend>;

    fn batch_stats(sample_count: usize, total_loss: f64, policy_agreement: f64) -> BatchStats {
        BatchStats {
            sample_count,
            batch_count: 1,
            total_loss,
            policy_agreement,
            loss_policy: total_loss + 0.1,
            loss_value: total_loss + 0.2,
            loss_grp: total_loss + 0.3,
            loss_tenpai: total_loss + 0.4,
            loss_danger: total_loss + 0.5,
            loss_opp_next: total_loss + 0.6,
            loss_score_pdf: total_loss + 0.7,
            loss_score_cdf: total_loss + 0.8,
            rare_actions: crate::progress::RareActionMetrics::default(),
        }
    }

    fn dummy_config() -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/data"),
            output_dir: PathBuf::from("/output"),
            num_epochs: 5,
            batch_size: 16,
            microbatch_size: Some(4),
            validation_microbatch_size: Some(4),
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            bc_shards_manifest_path: None,
            shard_prefetch_depth: None,
            train_fraction: 0.9,
            source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            validation_gates: crate::config::ValidationGateConfig::default(),
            rl: None,
            bc: BcHyperparamConfig::default(),
            nsight_trace: None,
            device: "cpu".to_string(),
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: Some(2),
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 2,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 5,
            validate_every_n_steps: 4,
            checkpoint_every_n_steps: 5,
            max_train_steps: Some(20),
            max_validation_batches: None,
            max_validation_samples: None,
            preflight: PreflightConfig::default(),
            precision_mode: crate::config::PrecisionMode::Fp32,
        }
    }

    fn dummy_manifest(counts_exact: bool) -> DataManifest {
        DataManifest {
            sources: Vec::new(),
            total_games: 24,
            train_count: 18,
            val_count: 6,
            counts_exact,
        }
    }

    fn dummy_validation_summary(policy_loss: f64, agreement: f64) -> ValidationSummary {
        ValidationSummary {
            total_loss: policy_loss + 0.5,
            policy_loss,
            agreement,
            samples: 64,
            rare_actions: crate::progress::RareActionMetrics::default(),
            saw_exit_targets: false,
            saw_delta_q_targets: false,
            profiling: None,
            delta_q_promotion: None,
            delta_q_promotion_result: None,
            delta_q_promotion_snapshot: None,
            delta_q_policy_transfer: None,
            delta_q_policy_transfer_result: None,
            delta_q_policy_transfer_snapshot: None,
        }
    }

    struct FakeValidationExecutor {
        calls: usize,
        summary: ValidationSummary,
    }

    impl ValidationExecutor<TrainBackend> for FakeValidationExecutor {
        fn run_validation(
            &mut self,
            _model: &HydraModel<TrainBackend>,
            _context: ValidationContext<
                '_,
                TrainConfig,
                crate::validation::TrainValidationLoader<'_>,
                ValidBackendOf<TrainBackend>,
            >,
            _runtime: ValidationRuntime<'_>,
        ) -> Result<ValidationSummary, String> {
            self.calls += 1;
            Ok(self.summary.clone())
        }
    }

    fn dummy_runtime_resume_contract() -> RuntimeResumeContract {
        RuntimeResumeContract {
            batch_size: 16,
            train_microbatch_size: 4,
            validation_microbatch_size: 4,
            accum_steps: 4,
            precision_mode: crate::config::PrecisionMode::Fp32,
        }
    }

    fn temp_dir_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("hydra_epoch_runner_{label}_{unique}"))
    }

    fn test_artifacts(label: &str) -> BcArtifactPaths {
        let output_dir = temp_dir_path(label);
        fs::create_dir_all(&output_dir).expect("create test output dir");
        let artifacts = BcArtifactPaths::new(&output_dir, 0);
        artifacts.create_root_dir().expect("create artifacts root");
        artifacts
    }

    fn tiny_dummy_model(device: &LibTorchDevice) -> HydraModel<TrainBackend> {
        HydraModelConfig::new(1)
            .with_input_channels(hydra_train::config::INPUT_CHANNELS)
            .with_hidden_channels(4)
            .with_num_groups(4)
            .with_se_bottleneck(1)
            .init::<TrainBackend>(device)
    }

    fn dummy_train_sample(action: u8) -> MjaiSample {
        let mut legal_mask = [0.0f32; hydra_core::action::HYDRA_ACTION_SPACE];
        legal_mask[action as usize] = 1.0;
        legal_mask[45] = 1.0;
        MjaiSample {
            obs: [0.1f32; hydra_core::encoder::OBS_SIZE],
            action,
            legal_mask,
            placement: 0,
            score_delta: 0,
            grp_label: 0,
            oracle_target: None,
            tenpai: [0.0; 3],
            opp_next: [0, 1, 255],
            danger: [0.0; 102],
            danger_mask: [1.0; 102],
            safety_residual: None,
            safety_residual_mask: None,
            exit_target: None,
            exit_mask: None,
            delta_q_target: None,
            delta_q_mask: None,
            belief_fields: None,
            mixture_weights: None,
            belief_fields_present: false,
            mixture_weights_present: false,
        }
    }

    fn dummy_valid_loss() -> HydraLoss<TestValidBackend> {
        HydraLoss::<TestValidBackend>::new(HydraLossConfig::new())
    }

    fn dummy_train_loss() -> HydraLoss<TrainBackend> {
        HydraLoss::<TrainBackend>::new(HydraLossConfig::new())
    }

    fn read_jsonl_entry(path: &Path) -> serde_json::Value {
        let raw = fs::read_to_string(path).expect("read jsonl file");
        let line = raw.lines().next().expect("jsonl entry line");
        serde_json::from_str(line).expect("parse jsonl entry")
    }

    fn modified_time(path: &Path) -> SystemTime {
        fs::metadata(path)
            .expect("read file metadata")
            .modified()
            .expect("read file modified time")
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn train_logical_batch_empty_keeps_model_slot_populated() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch: Vec<MjaiSample> = Vec::new();

        let (drained, _sub_timing) = train_logical_batch(
            &logical_batch,
            TrainLogicalBatchConfig {
                microbatch_size: 1,
                use_amp: false,
                augment: false,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                lr: 1.0e-4,
            },
            &mut head_controller,
            &mut model_slot,
            &mut optimizer,
        )
        .expect("empty logical batch should succeed");

        assert!(drained.is_empty());
        assert!(model_slot.is_some());
    }

    #[test]
    fn train_logical_batch_reports_clear_error_when_model_slot_is_empty() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = None;
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![dummy_train_sample(0)];

        let result = train_logical_batch(
            &logical_batch,
            TrainLogicalBatchConfig {
                microbatch_size: 1,
                use_amp: false,
                augment: false,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                lr: 1.0e-4,
            },
            &mut head_controller,
            &mut model_slot,
            &mut optimizer,
        );

        let err = match result {
            Ok(_) => panic!("empty model slot should return a clear error"),
            Err(err) => err,
        };

        assert!(err.contains("epoch runner model slot should stay populated"));
        assert!(model_slot.is_none());
    }

    #[test]
    fn train_logical_batch_non_empty_keeps_model_slot_populated_and_returns_stats() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];

        let (drained, _sub_timing) = train_logical_batch(
            &logical_batch,
            TrainLogicalBatchConfig {
                microbatch_size: 1,
                use_amp: false,
                augment: false,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                lr: 1.0e-4,
            },
            &mut head_controller,
            &mut model_slot,
            &mut optimizer,
        )
        .expect("train logical batch with samples");

        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].sample_count, 2);
        assert_eq!(drained[0].batch_count, 2);
        assert!(drained.iter().all(|stats| stats.total_loss.is_finite()));
        assert!(
            drained
                .iter()
                .all(|stats| stats.policy_agreement.is_finite())
        );
        assert!(model_slot.is_some());
    }

    #[test]
    fn train_logical_batch_full_microbatch_keeps_model_slot_populated() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];

        let (drained, _sub_timing) = train_logical_batch(
            &logical_batch,
            TrainLogicalBatchConfig {
                microbatch_size: logical_batch.len(),
                use_amp: false,
                augment: false,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                lr: 1.0e-4,
            },
            &mut head_controller,
            &mut model_slot,
            &mut optimizer,
        )
        .expect("full microbatch path should succeed");

        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].sample_count, logical_batch.len());
        assert_eq!(drained[0].batch_count, 1);
        assert!(drained.iter().all(|stats| stats.total_loss.is_finite()));
        assert!(
            drained
                .iter()
                .all(|stats| stats.policy_agreement.is_finite())
        );
        assert!(model_slot.is_some());
    }

    #[test]
    fn train_logical_batch_records_microbatch_sub_stage_scope_order() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];

        let (_, events) = crate::nvtx::with_test_recorder(|| {
            train_logical_batch(
                &logical_batch,
                TrainLogicalBatchConfig {
                    microbatch_size: 1,
                    use_amp: false,
                    augment: false,
                    train_device: &device,
                    loss_fn: &train_loss_fn,
                    bc_exit_cfg: &BcExitConfig::default(),
                    lr: 1.0e-4,
                },
                &mut head_controller,
                &mut model_slot,
                &mut optimizer,
            )
            .expect("train logical batch with NVTX recording");
        });

        assert!(
            events.contains(&"push:collation".to_string()),
            "should record collation sub-stage"
        );
        assert!(
            events.contains(&"push:forward".to_string()),
            "should record forward sub-stage"
        );
        assert!(
            events.contains(&"push:loss".to_string()),
            "should record loss sub-stage"
        );
        assert!(
            events.contains(&"push:backward".to_string()),
            "should record backward sub-stage"
        );
        assert!(
            events.contains(&"push:optimizer_step".to_string()),
            "should record optimizer_step sub-stage"
        );

        for push_event in events.iter().filter(|e| e.starts_with("push:")) {
            let stage = push_event.strip_prefix("push:").unwrap();
            let pop = format!("pop:{stage}");
            assert!(
                events.contains(&pop),
                "every push should have a matching pop: {push_event}"
            );
            let push_idx = events.iter().position(|e| e == push_event).unwrap();
            let pop_idx = events.iter().position(|e| e == &pop).unwrap();
            assert!(pop_idx > push_idx, "pop should come after push for {stage}");
        }
    }

    #[test]
    fn train_logical_batch_sub_stage_timing_has_nonzero_values() {
        let device = LibTorchDevice::Cpu;
        let mut model_slot = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];

        let (_drained, sub_timing) = train_logical_batch(
            &logical_batch,
            TrainLogicalBatchConfig {
                microbatch_size: 1,
                use_amp: false,
                augment: false,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                lr: 1.0e-4,
            },
            &mut head_controller,
            &mut model_slot,
            &mut optimizer,
        )
        .expect("train logical batch for sub-timing check");

        assert!(
            sub_timing.collation_seconds > 0.0,
            "collation should have measurable time"
        );
        assert!(
            sub_timing.forward_seconds > 0.0,
            "forward should have measurable time"
        );
        assert!(
            sub_timing.loss_seconds > 0.0,
            "loss should have measurable time"
        );
        assert!(
            sub_timing.backward_seconds > 0.0,
            "backward should have measurable time"
        );
        assert!(
            sub_timing.optimizer_step_seconds > 0.0,
            "optimizer_step should have measurable time"
        );
        assert!(
            sub_timing.metric_readback_seconds > 0.0,
            "metric_readback should have measurable time"
        );

        let children = sub_timing.to_profiling_children();
        assert_eq!(children.len(), 8);
        assert!(
            children.iter().all(|c| {
                c.stage == PROFILING_STAGE_PRODUCER_WAIT
                    || c.stage == PROFILING_STAGE_H2D_TRANSFER
                    || c.elapsed_seconds > 0.0
            }),
            "active profiling children should have positive elapsed_seconds"
        );
    }

    #[test]
    fn emit_interval_step_summary_records_logging_scope_order() {
        let artifacts = test_artifacts("nvtx_interval_logging_scope");
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let multi = MultiProgress::new();
        let config = dummy_config();
        let manifest = dummy_manifest(true);
        let validation_summary = dummy_validation_summary(0.4, 0.7);

        let (_, events) = crate::nvtx::with_test_recorder(|| {
            emit_interval_step_summary(
                &multi,
                &mut tb,
                &mut step_log,
                IntervalStepSummaryContext {
                    manifest: &manifest,
                    config: &config,
                    session_start_global_step: 0,
                    global_step: 5,
                    epoch: 1,
                    lr: 1.0e-4,
                    best_validation: Some(BestValidation {
                        policy_loss: 0.5,
                        agreement: 0.6,
                    }),
                    val_summary: Some(validation_summary),
                    seen_samples: 16,
                    assumed_games_seen: 4,
                    epoch_optimizer_steps: 5,
                    window_stats: ScalarAverages::default().finalize(),
                    step_rate: 12.0,
                    profiling: None,
                    advisories: Vec::new(),
                },
            )
            .expect("emit interval step summary should succeed");
        });

        assert_eq!(events, vec!["push:logging", "pop:logging"]);
    }

    #[test]
    fn epoch_end_validation_runs_on_interval_or_final_epoch() {
        assert!(should_run_epoch_end_validation(0, 3, 1));
        assert!(!should_run_epoch_end_validation(0, 3, 2));
        assert!(should_run_epoch_end_validation(1, 3, 2));
        assert!(should_run_epoch_end_validation(2, 3, 5));
    }

    #[test]
    fn epoch_end_validation_skips_non_boundary_epochs() {
        assert!(!should_run_epoch_end_validation(0, 5, 3));
        assert!(!should_run_epoch_end_validation(1, 5, 3));
        assert!(should_run_epoch_end_validation(2, 5, 3));
        assert!(!should_run_epoch_end_validation(3, 5, 3));
    }

    #[test]
    fn epoch_end_validation_uses_injected_executor_on_boundary() {
        let mut config = dummy_config();
        config.num_epochs = 5;
        config.validation_every_n_epochs = 2;
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("epoch_validation_executor_seam");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = None;
        let mut executor = FakeValidationExecutor {
            calls: 0,
            summary: dummy_validation_summary(0.33, 0.77),
        };

        let summary = run_epoch_end_validation_with_executor(
            1,
            &model,
            EpochEndValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                cached_validation_samples: None,
            },
            None,
            &mut best_validation,
            2.5,
            &mut executor,
        )
        .expect("epoch-end validation through fake executor")
        .expect("epoch boundary returns validation summary");

        assert_eq!(executor.calls, 1);
        assert_eq!(summary.policy_loss, 0.33);
        assert_eq!(best_validation.map(|best| best.policy_loss), Some(0.33));
    }

    #[test]
    fn epoch_end_validation_skip_does_not_call_injected_executor() {
        let mut config = dummy_config();
        config.num_epochs = 5;
        config.validation_every_n_epochs = 3;
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("epoch_validation_executor_skip");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = None;
        let mut executor = FakeValidationExecutor {
            calls: 0,
            summary: dummy_validation_summary(0.33, 0.77),
        };

        let summary = run_epoch_end_validation_with_executor(
            0,
            &model,
            EpochEndValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                cached_validation_samples: None,
            },
            None,
            &mut best_validation,
            2.5,
            &mut executor,
        )
        .expect("skip epoch-end validation through fake executor");

        assert!(summary.is_none());
        assert_eq!(executor.calls, 0);
        assert_eq!(best_validation, None);
    }

    #[test]
    fn build_epoch_continuation_matches_completion_state() {
        let completed = build_epoch_continuation(2, true, 99);
        assert_eq!(completed.next_epoch, 3);
        assert_eq!(completed.skip_optimizer_steps_in_epoch, 0);
        assert!(completed.epoch_completed);

        let partial = build_epoch_continuation(2, false, 99);
        assert_eq!(partial.next_epoch, 2);
        assert_eq!(partial.skip_optimizer_steps_in_epoch, 99);
        assert!(!partial.epoch_completed);
    }

    #[test]
    fn build_epoch_continuation_resets_skip_count_after_empty_completed_epoch() {
        let continuation = build_epoch_continuation(7, true, 0);

        assert_eq!(continuation.next_epoch, 8);
        assert_eq!(continuation.skip_optimizer_steps_in_epoch, 0);
        assert!(continuation.epoch_completed);
    }

    #[test]
    fn epoch_end_validation_runs_on_final_epoch_even_when_interval_is_larger() {
        assert!(should_run_epoch_end_validation(4, 5, 10));
        assert!(!should_run_epoch_end_validation(3, 5, 10));
    }

    #[test]
    fn record_drained_batch_stats_updates_both_accumulators_with_weighted_values() {
        let mut stats = ScalarAverages::default();
        let mut window = ScalarAverages::default();

        record_drained_batch_stats(
            vec![batch_stats(2, 1.5, 0.25), batch_stats(3, 4.0, 0.75)],
            &mut stats,
            &mut window,
        );

        let stats = stats.finalize();
        let window = window.finalize();

        for aggregate in [stats, window] {
            assert_eq!(aggregate.num_batches, 2);
            assert_eq!(aggregate.num_samples, 5);
            assert!((aggregate.total_loss - 3.0).abs() < 1e-12);
            assert!((aggregate.policy_agreement - 0.55).abs() < 1e-12);
            assert!((aggregate.loss_policy - 3.1).abs() < 1e-12);
            assert!((aggregate.loss_value - 3.2).abs() < 1e-12);
            assert!((aggregate.loss_grp - 3.3).abs() < 1e-12);
            assert!((aggregate.loss_tenpai - 3.4).abs() < 1e-12);
            assert!((aggregate.loss_danger - 3.5).abs() < 1e-12);
            assert!((aggregate.loss_opp_next - 3.6).abs() < 1e-12);
            assert!((aggregate.loss_score_pdf - 3.7).abs() < 1e-12);
            assert!((aggregate.loss_score_cdf - 3.8).abs() < 1e-12);
        }
    }

    #[test]
    fn record_drained_batch_stats_ignores_empty_drains() {
        let mut stats = ScalarAverages::default();
        let mut window = ScalarAverages::default();

        record_drained_batch_stats(Vec::new(), &mut stats, &mut window);

        let stats = stats.finalize();
        let window = window.finalize();

        assert_eq!(stats.num_batches, 0);
        assert_eq!(stats.num_samples, 0);
        assert_eq!(stats.total_loss, 0.0);
        assert_eq!(window.num_batches, 0);
        assert_eq!(window.num_samples, 0);
        assert_eq!(window.total_loss, 0.0);
    }

    #[test]
    fn record_drained_batch_stats_preserves_zero_weight_guard_for_both_accumulators() {
        let mut stats = ScalarAverages::default();
        let mut window = ScalarAverages::default();

        record_drained_batch_stats(
            vec![batch_stats(0, 99.0, 0.99), batch_stats(4, 2.5, 0.4)],
            &mut stats,
            &mut window,
        );

        let stats = stats.finalize();
        let window = window.finalize();

        for aggregate in [stats, window] {
            assert_eq!(aggregate.num_batches, 1);
            assert_eq!(aggregate.num_samples, 4);
            assert!((aggregate.total_loss - 2.5).abs() < 1e-12);
            assert!((aggregate.policy_agreement - 0.4).abs() < 1e-12);
            assert!((aggregate.loss_policy - 2.6).abs() < 1e-12);
        }
    }

    #[test]
    fn interval_validation_uses_injected_executor_on_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_executor_seam");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.8,
            agreement: 0.6,
        });
        let mut executor = FakeValidationExecutor {
            calls: 0,
            summary: dummy_validation_summary(0.25, 0.75),
        };
        let multi = MultiProgress::new();

        let summary = maybe_run_interval_validation_with_executor(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            14,
            1.5,
            &mut executor,
        )
        .expect("interval validation through fake executor")
        .expect("boundary returns validation summary");

        assert_eq!(executor.calls, 1);
        assert_eq!(summary.samples, 64);
        assert_eq!(summary.policy_loss, 0.25);
        assert_eq!(best_validation.map(|best| best.policy_loss), Some(0.25));
    }

    #[test]
    fn interval_validation_skip_does_not_call_injected_executor() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_executor_skip");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = None;
        let mut executor = FakeValidationExecutor {
            calls: 0,
            summary: dummy_validation_summary(0.25, 0.75),
        };
        let multi = MultiProgress::new();

        let summary = maybe_run_interval_validation_with_executor(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            13,
            1.5,
            &mut executor,
        )
        .expect("skip interval validation through fake executor");

        assert!(summary.is_none());
        assert_eq!(executor.calls, 0);
        assert_eq!(best_validation, None);
    }
    #[test]
    fn maybe_run_interval_validation_skips_until_step_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_skip");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.8,
            agreement: 0.6,
        });
        let multi = MultiProgress::new();

        let zero_step = maybe_run_interval_validation(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            10,
            1.5,
        )
        .expect("skip zero session step validation");

        let off_interval = maybe_run_interval_validation(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            13,
            1.5,
        )
        .expect("skip non-boundary validation");

        assert!(zero_step.is_none());
        assert!(off_interval.is_none());
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.8,
                agreement: 0.6,
            })
        );
        assert!(!artifacts.best_model_base.with_extension("mpk").exists());
    }

    #[test]
    fn progress_message_refreshes_only_on_display_boundaries() {
        let mut config = dummy_config();
        config.log_every_n_steps = 10;
        config.validate_every_n_steps = 4;
        config.checkpoint_every_n_steps = 6;
        config.max_train_steps = Some(11);

        assert!(!should_refresh_train_progress_message(&config, 100, 100));
        assert!(should_refresh_train_progress_message(&config, 101, 100));
        assert!(!should_refresh_train_progress_message(&config, 103, 100));
        assert!(should_refresh_train_progress_message(&config, 104, 100));
        assert!(should_refresh_train_progress_message(&config, 106, 100));
        assert!(should_refresh_train_progress_message(&config, 110, 100));
        assert!(should_refresh_train_progress_message(&config, 111, 100));
    }

    #[test]
    fn checkpoint_boundary_helper_matches_session_relative_cadence() {
        let mut config = dummy_config();
        config.checkpoint_every_n_steps = 5;

        assert!(!should_save_periodic_checkpoint(&config, 100, 100));
        assert!(!should_save_periodic_checkpoint(&config, 104, 100));
        assert!(should_save_periodic_checkpoint(&config, 105, 100));
        assert!(!should_save_periodic_checkpoint(&config, 109, 100));
        assert!(should_save_periodic_checkpoint(&config, 110, 100));
    }

    #[test]
    fn maybe_save_periodic_checkpoint_skips_when_session_step_is_not_on_boundary() {
        let config = dummy_config();
        let artifacts = test_artifacts("periodic_checkpoint_skip");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let optimizer = AdamConfig::new().init();

        maybe_save_periodic_checkpoint(
            &model,
            &optimizer,
            PeriodicCheckpointContext {
                config: &config,
                artifacts: &artifacts,
                epoch: 3,
                session_start_global_step: 10,
                current_runtime: dummy_runtime_resume_contract(),
            },
            PeriodicCheckpointState {
                global_step: 10,
                epoch_optimizer_steps: 4,
                total_loss: 1.25,
                best_validation: None,
            },
        )
        .expect("skip checkpoint at session start");

        maybe_save_periodic_checkpoint(
            &model,
            &optimizer,
            PeriodicCheckpointContext {
                config: &config,
                artifacts: &artifacts,
                epoch: 3,
                session_start_global_step: 10,
                current_runtime: dummy_runtime_resume_contract(),
            },
            PeriodicCheckpointState {
                global_step: 14,
                epoch_optimizer_steps: 4,
                total_loss: 1.25,
                best_validation: None,
            },
        )
        .expect("skip checkpoint off interval");

        assert!(!artifacts.latest_state_path.exists());
        assert!(!artifacts.latest_model_base.with_extension("mpk").exists());
        assert!(
            !artifacts
                .latest_optimizer_base
                .with_extension("bin")
                .exists()
        );
    }

    #[test]
    fn maybe_save_periodic_checkpoint_persists_resume_state_on_boundary() {
        let config = dummy_config();
        let artifacts = test_artifacts("periodic_checkpoint_save");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let optimizer = AdamConfig::new().init();
        let expected_best = Some(BestValidation {
            policy_loss: 0.7,
            agreement: 0.8,
        });

        maybe_save_periodic_checkpoint(
            &model,
            &optimizer,
            PeriodicCheckpointContext {
                config: &config,
                artifacts: &artifacts,
                epoch: 3,
                session_start_global_step: 10,
                current_runtime: dummy_runtime_resume_contract(),
            },
            PeriodicCheckpointState {
                global_step: 15,
                epoch_optimizer_steps: 7,
                total_loss: 1.25,
                best_validation: expected_best,
            },
        )
        .expect("save checkpoint on interval");

        let state = read_resume_state(&artifacts.latest_state_path).expect("read resume state");
        assert_eq!(state.next_epoch, 3);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 7);
        assert_eq!(state.global_step, 15);
        assert_eq!(state.best_validation, expected_best);
        assert_eq!(state.runtime, dummy_runtime_resume_contract());
        assert!(artifacts.latest_state_path.exists());
        assert!(artifacts.latest_model_base.with_extension("mpk").exists());
        assert!(
            artifacts
                .latest_model_base
                .with_extension("meta.json")
                .exists()
        );
        assert!(
            artifacts
                .latest_optimizer_base
                .with_extension("bin")
                .exists()
        );
    }

    #[test]
    fn maybe_save_periodic_checkpoint_preserves_absent_best_validation_on_boundary() {
        let config = dummy_config();
        let artifacts = test_artifacts("periodic_checkpoint_save_without_best");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let optimizer = AdamConfig::new().init();

        maybe_save_periodic_checkpoint(
            &model,
            &optimizer,
            PeriodicCheckpointContext {
                config: &config,
                artifacts: &artifacts,
                epoch: 1,
                session_start_global_step: 0,
                current_runtime: dummy_runtime_resume_contract(),
            },
            PeriodicCheckpointState {
                global_step: 5,
                epoch_optimizer_steps: 2,
                total_loss: 0.5,
                best_validation: None,
            },
        )
        .expect("save checkpoint without best validation");

        let state = read_resume_state(&artifacts.latest_state_path).expect("read resume state");
        assert_eq!(state.next_epoch, 1);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 2);
        assert_eq!(state.global_step, 5);
        assert_eq!(state.best_validation, None);
    }

    #[test]
    fn maybe_run_interval_validation_updates_best_and_saves_checkpoint_on_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_save");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.8,
            agreement: 0.6,
        });
        let multi = MultiProgress::new();

        let summary = maybe_run_interval_validation(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            14,
            1.5,
        )
        .expect("run interval validation on boundary");

        let summary = summary.expect("validation summary on boundary");
        assert_eq!(summary.samples, 0);
        assert_eq!(summary.total_loss, 0.0);
        assert_eq!(summary.policy_loss, 0.0);
        assert_eq!(summary.agreement, 0.0);
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.0,
                agreement: 0.0,
            })
        );
        assert!(artifacts.best_model_base.with_extension("mpk").exists());
        assert!(
            artifacts
                .best_model_base
                .with_extension("meta.json")
                .exists()
        );
    }

    #[test]
    fn maybe_run_interval_validation_keeps_existing_best_when_summary_is_not_better() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_keep_best");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: -0.1,
            agreement: 0.9,
        });
        let multi = MultiProgress::new();

        let summary = maybe_run_interval_validation(
            ValidationStepContext {
                multi: &multi,
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                session_start_global_step: 10,
                cached_validation_samples: None,
            },
            &model,
            None,
            &mut best_validation,
            14,
            1.5,
        )
        .expect("run interval validation without best update");

        let summary = summary.expect("validation summary on boundary");
        assert_eq!(summary.samples, 0);
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: -0.1,
                agreement: 0.9,
            })
        );
        assert!(!artifacts.best_model_base.with_extension("mpk").exists());
    }

    #[test]
    fn emit_interval_step_summary_writes_skipped_validation_step_log() {
        let config = dummy_config();
        let artifacts = test_artifacts("step_summary_skipped_validation");
        let manifest = dummy_manifest(false);
        let multi = MultiProgress::new();
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let window_stats = ScalarAverages::default();

        emit_interval_step_summary(
            &multi,
            &mut tb,
            &mut step_log,
            IntervalStepSummaryContext {
                manifest: &manifest,
                config: &config,
                session_start_global_step: 0,
                global_step: 9,
                epoch: 1,
                lr: 1.0e-4,
                best_validation: None,
                val_summary: None,
                seen_samples: 32,
                assumed_games_seen: 0,
                epoch_optimizer_steps: 2,
                window_stats,
                step_rate: 12.5,
                profiling: None,
                advisories: Vec::new(),
            },
        )
        .expect("emit skipped validation interval summary");

        let entry = read_jsonl_entry(&artifacts.step_log_path);
        assert_eq!(entry["global_step"].as_u64(), Some(9));
        assert_eq!(entry["epoch"].as_u64(), Some(2));
        assert_close(entry["lr"].as_f64().expect("step log lr"), 1.0e-4);
        assert_eq!(entry["profiling"]["stage"].as_str(), Some("bc_interval"));
        assert!(entry["profiling"]["elapsed_seconds"].as_f64().is_some());
        assert!(entry["profiling"]["children"].is_array());
        assert_eq!(entry["val_total_loss"], serde_json::Value::Null);
        assert_eq!(entry["val_policy_loss"], serde_json::Value::Null);
        assert_eq!(entry["best_val_policy_loss"], serde_json::Value::Null);
    }

    #[test]
    fn emit_interval_step_summary_writes_validation_and_best_metrics() {
        let config = dummy_config();
        let artifacts = test_artifacts("step_summary_validation_metrics");
        let manifest = dummy_manifest(true);
        let multi = MultiProgress::new();
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut window_stats = ScalarAverages::default();
        window_stats.record_batch(batch_stats(4, 2.5, 0.4));
        let window_stats = window_stats.finalize();
        let val_summary = dummy_validation_summary(0.9, 0.65);

        emit_interval_step_summary(
            &multi,
            &mut tb,
            &mut step_log,
            IntervalStepSummaryContext {
                manifest: &manifest,
                config: &config,
                session_start_global_step: 5,
                global_step: 11,
                epoch: 2,
                lr: 2.5e-4,
                best_validation: Some(BestValidation {
                    policy_loss: 0.8,
                    agreement: 0.7,
                }),
                val_summary: Some(val_summary.clone()),
                seen_samples: 48,
                assumed_games_seen: 6,
                epoch_optimizer_steps: 3,
                window_stats,
                step_rate: 3.0,
                profiling: None,
                advisories: Vec::new(),
            },
        )
        .expect("emit interval summary with validation");

        let entry = read_jsonl_entry(&artifacts.step_log_path);
        assert_eq!(entry["global_step"].as_u64(), Some(11));
        assert_eq!(entry["epoch"].as_u64(), Some(3));
        assert_close(
            entry["train_total_loss"]
                .as_f64()
                .expect("train total loss"),
            2.5,
        );
        assert_close(
            entry["train_policy_agreement"]
                .as_f64()
                .expect("train policy agreement"),
            0.4,
        );
        assert_close(
            entry["val_total_loss"].as_f64().expect("val total loss"),
            val_summary.total_loss,
        );
        assert_close(
            entry["val_policy_loss"].as_f64().expect("val policy loss"),
            val_summary.policy_loss,
        );
        assert_close(
            entry["val_policy_agreement"]
                .as_f64()
                .expect("val policy agreement"),
            val_summary.agreement,
        );
        assert_close(
            entry["best_val_policy_loss"]
                .as_f64()
                .expect("best val policy loss"),
            0.8,
        );
        assert_close(
            entry["best_val_agreement"]
                .as_f64()
                .expect("best val agreement"),
            0.7,
        );
        assert_eq!(entry["profiling"]["stage"].as_str(), Some("bc_interval"));
        assert!(entry["profiling"]["elapsed_seconds"].as_f64().is_some());
        assert!(entry["profiling"]["children"].is_array());
    }

    #[test]
    fn run_epoch_end_validation_returns_none_when_epoch_is_not_a_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("epoch_end_validation_skip");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.6,
            agreement: 0.75,
        });

        let summary = run_epoch_end_validation(
            0,
            &model,
            EpochEndValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                cached_validation_samples: None,
            },
            None,
            &mut best_validation,
            1.2,
        )
        .expect("skip epoch-end validation");

        assert!(summary.is_none());
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.6,
                agreement: 0.75,
            })
        );
        assert!(!artifacts.best_model_base.with_extension("mpk").exists());
    }

    #[test]
    fn run_epoch_end_validation_updates_best_and_saves_checkpoint_on_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("epoch_end_validation_save");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let valid_loss_fn = dummy_valid_loss();
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.7,
            agreement: 0.8,
        });

        let summary = run_epoch_end_validation(
            1,
            &model,
            EpochEndValidationContext {
                config: &config,
                loader_config: &loader_config,
                manifest: &manifest,
                train_device: &device,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                artifacts: &artifacts,
                cached_validation_samples: None,
            },
            None,
            &mut best_validation,
            1.2,
        )
        .expect("run epoch-end validation on boundary");

        let summary = summary.expect("epoch-end validation summary");
        assert_eq!(summary.samples, 0);
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.0,
                agreement: 0.0,
            })
        );
        assert!(artifacts.best_model_base.with_extension("mpk").exists());
        assert!(
            artifacts
                .best_model_base
                .with_extension("meta.json")
                .exists()
        );
    }

    #[test]
    fn finalize_epoch_outputs_writes_training_log_with_validation_metrics() {
        let config = dummy_config();
        let artifacts = test_artifacts("finalize_epoch_outputs");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut train_stats = ScalarAverages::default();
        train_stats.record_batch(batch_stats(4, 3.5, 0.55));
        let train_stats = train_stats.finalize();
        let val_summary = dummy_validation_summary(0.95, 0.68);

        finalize_epoch_outputs(
            &mut tb,
            &mut training_log,
            EpochFinalizeContext {
                config: &config,
                train_cfg: &train_cfg,
                epoch: 2,
                global_step: 17,
                train_stats,
                val_summary: Some(val_summary.clone()),
                best_validation: Some(BestValidation {
                    policy_loss: 0.9,
                    agreement: 0.7,
                }),
                final_lr: 2.0e-4,
                profiling: None,
            },
        )
        .expect("finalize epoch outputs");

        let entry = read_jsonl_entry(&artifacts.training_log_path);
        assert_eq!(entry["epoch"].as_u64(), Some(3));
        assert_eq!(entry["global_step"].as_u64(), Some(17));
        assert_eq!(entry["num_batches"].as_u64(), Some(1));
        assert_close(
            entry["train_total_loss"]
                .as_f64()
                .expect("train total loss"),
            3.5,
        );
        assert_close(
            entry["val_total_loss"].as_f64().expect("val total loss"),
            val_summary.total_loss,
        );
        assert_close(
            entry["val_policy_loss"].as_f64().expect("val policy loss"),
            val_summary.policy_loss,
        );
        assert_close(
            entry["best_val_policy_loss"]
                .as_f64()
                .expect("best val policy loss"),
            0.9,
        );
        assert_close(
            entry["best_val_agreement"]
                .as_f64()
                .expect("best val agreement"),
            0.7,
        );
        assert_eq!(entry["profiling"]["stage"].as_str(), Some("bc_epoch"));
        assert!(entry["profiling"]["elapsed_seconds"].as_f64().is_some());
        assert!(entry["profiling"]["children"].is_array());
    }

    #[test]
    fn finalize_epoch_outputs_preserves_train_sub_stage_children_in_json() {
        let config = dummy_config();
        let artifacts = test_artifacts("finalize_epoch_outputs_sub_stages");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut train_stats = ScalarAverages::default();
        train_stats.record_batch(batch_stats(4, 3.5, 0.55));
        let train_stats = train_stats.finalize();

        let sub_timing = TrainSubStageTiming {
            producer_wait_seconds: 0.04,
            collation_seconds: 0.01,
            h2d_transfer_seconds: 0.06,
            h2d_pageable_to_pinned_seconds: 0.01,
            h2d_tensor_materialize_seconds: 0.04,
            h2d_stream_sync_seconds: 0.01,
            forward_seconds: 0.5,
            loss_seconds: 0.02,
            backward_seconds: 0.3,
            metric_readback_seconds: 0.02,
            optimizer_step_seconds: 0.05,
        };
        let profiling = bc_epoch_profiling(0.88, &sub_timing, None, 0.1, 0.0);

        finalize_epoch_outputs(
            &mut tb,
            &mut training_log,
            EpochFinalizeContext {
                config: &config,
                train_cfg: &train_cfg,
                epoch: 0,
                global_step: 5,
                train_stats,
                val_summary: None,
                best_validation: None,
                final_lr: 1.0e-4,
                profiling: Some(profiling),
            },
        )
        .expect("finalize epoch outputs with sub-stage profiling");

        let entry = read_jsonl_entry(&artifacts.training_log_path);
        let profiling = &entry["profiling"];
        assert_eq!(profiling["stage"].as_str(), Some("bc_epoch"));

        let children = profiling["children"]
            .as_array()
            .expect("profiling should have children array");
        let train_child = children
            .iter()
            .find(|c| c["stage"].as_str() == Some("train"))
            .expect("should have a 'train' child");
        let train_sub_children = train_child["children"]
            .as_array()
            .expect("train child should have sub-stage children");

        let expected_sub_stages = ["collation", "forward", "loss", "backward", "optimizer_step"];
        for stage_name in &expected_sub_stages {
            let found = train_sub_children
                .iter()
                .find(|c| c["stage"].as_str() == Some(stage_name));
            assert!(
                found.is_some(),
                "train sub-stage '{}' should be present in JSON",
                stage_name
            );
            let elapsed = found.unwrap()["elapsed_seconds"].as_f64();
            assert!(
                elapsed.is_some() && elapsed.unwrap() > 0.0,
                "train sub-stage '{}' should have positive elapsed_seconds",
                stage_name
            );
        }

        let h2d_child = train_sub_children
            .iter()
            .find(|c| c["stage"].as_str() == Some("h2d_transfer"))
            .expect("h2d transfer stage should be present in JSON");
        let h2d_sub_children = h2d_child["children"]
            .as_array()
            .expect("h2d child should have materialization sub-stage children");
        for stage_name in &[
            "h2d_pageable_to_pinned",
            "h2d_tensor_materialize",
            "h2d_stream_sync",
        ] {
            let elapsed = h2d_sub_children
                .iter()
                .find(|c| c["stage"].as_str() == Some(stage_name))
                .and_then(|c| c["elapsed_seconds"].as_f64())
                .expect("h2d sub-stage should carry elapsed seconds");
            assert!(
                elapsed > 0.0,
                "h2d sub-stage '{stage_name}' should be positive"
            );
        }
    }

    #[test]
    fn bc_interval_profiling_records_checkpoint_separately_from_logging() {
        let sub_timing = TrainSubStageTiming::default();

        let profiling = bc_interval_profiling(1.0, &sub_timing, None, 0.25);

        assert_eq!(
            child_elapsed_seconds(&profiling, PROFILING_STAGE_CHECKPOINT),
            0.25
        );
        assert_eq!(
            child_elapsed_seconds(&profiling, PROFILING_STAGE_LOGGING),
            0.0
        );
        let input = interval_timing_input(&dummy_config(), &profiling, 4);
        assert_eq!(input.checkpoint_seconds, 0.25);
        assert_eq!(input.logging_seconds, 0.0);
    }

    #[test]
    fn finalize_epoch_outputs_writes_skipped_validation_epoch_log() {
        let config = dummy_config();
        let artifacts = test_artifacts("finalize_epoch_outputs_skipped_validation");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");

        finalize_epoch_outputs(
            &mut tb,
            &mut training_log,
            EpochFinalizeContext {
                config: &config,
                train_cfg: &train_cfg,
                epoch: 0,
                global_step: 3,
                train_stats: ScalarAverages::default(),
                val_summary: None,
                best_validation: None,
                final_lr: 5.0e-5,
                profiling: None,
            },
        )
        .expect("finalize epoch outputs without validation");

        let entry = read_jsonl_entry(&artifacts.training_log_path);
        assert_eq!(entry["epoch"].as_u64(), Some(1));
        assert_eq!(entry["global_step"].as_u64(), Some(3));
        assert_eq!(entry["num_batches"].as_u64(), Some(0));
        assert_close(entry["lr"].as_f64().expect("epoch log lr"), 5.0e-5);
        assert_eq!(entry["val_total_loss"], serde_json::Value::Null);
        assert_eq!(entry["val_policy_loss"], serde_json::Value::Null);
        assert_eq!(entry["best_val_policy_loss"], serde_json::Value::Null);
        assert_eq!(entry["best_val_agreement"], serde_json::Value::Null);
        assert_eq!(entry["profiling"]["stage"].as_str(), Some("bc_epoch"));
        assert!(entry["profiling"]["elapsed_seconds"].as_f64().is_some());
        assert!(entry["profiling"]["children"].is_array());
    }

    #[test]
    fn run_epoch_empty_manifest_finalizes_and_advances_epoch() {
        let mut config = dummy_config();
        config.num_epochs = 3;
        config.validation_every_n_epochs = 2;
        config.max_train_steps = Some(20);
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("run_epoch_empty_manifest_complete");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let train_loss_fn = dummy_train_loss();
        let valid_loss_fn = dummy_valid_loss();
        let device = LibTorchDevice::Cpu;
        let mut model = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut global_step = 7usize;
        let mut best_validation = None;
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut last_log_step = 0usize;
        let mut last_log_time = Instant::now();
        let run_start = Instant::now();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));

        let outcome = run_epoch(
            EpochRunnerContext {
                epoch: 1,
                config: &config,
                manifest: &manifest,
                loader_config: &loader_config,
                artifacts: &artifacts,
                train_cfg: &train_cfg,
                loss_fn: &train_loss_fn,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                train_device: &device,
                session_start_global_step: 0,
                steps_to_skip: 3,
                microbatch_size: 4,
                use_amp: false,
                total_steps: 100,
                current_runtime: dummy_runtime_resume_contract(),
                run_start: &run_start,
                head_controller: &mut head_controller,
                cached_validation_samples: None,
            },
            EpochRuntimeMut {
                model: &mut model,
                optimizer: &mut optimizer,
                global_step: &mut global_step,
                best_validation: &mut best_validation,
                tb: &mut tb,
                training_log: &mut training_log,
                step_log: &mut step_log,
                last_log_step: &mut last_log_step,
                last_log_time: &mut last_log_time,
            },
        )
        .expect("run epoch with empty manifest");

        assert!(!outcome.stop_after_epoch);
        assert_eq!(global_step, 7);
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.0,
                agreement: 0.0,
            })
        );

        let state = read_resume_state(&artifacts.latest_state_path).expect("read latest state");
        assert_eq!(state.next_epoch, 2);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 0);
        assert_eq!(state.global_step, 7);
        assert_eq!(state.best_validation, None);
        assert_eq!(state.runtime, dummy_runtime_resume_contract());
        assert!(artifacts.best_model_base.with_extension("mpk").exists());
        assert!(artifacts.training_log_path.exists());

        let entry = read_jsonl_entry(&artifacts.training_log_path);
        assert_eq!(entry["epoch"].as_u64(), Some(2));
        assert_eq!(entry["global_step"].as_u64(), Some(7));
        assert_close(
            entry["val_total_loss"]
                .as_f64()
                .expect("epoch val total loss"),
            0.0,
        );
        assert_close(
            entry["best_val_policy_loss"]
                .as_f64()
                .expect("epoch best validation loss"),
            0.0,
        );
    }

    #[test]
    fn run_epoch_empty_manifest_stops_when_session_budget_is_already_exhausted() {
        let mut config = dummy_config();
        config.num_epochs = 4;
        config.validation_every_n_epochs = 3;
        config.max_train_steps = Some(0);
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(false);
        let artifacts = test_artifacts("run_epoch_empty_manifest_budget_stop");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let train_loss_fn = dummy_train_loss();
        let valid_loss_fn = dummy_valid_loss();
        let device = LibTorchDevice::Cpu;
        let mut model = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut global_step = 12usize;
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.4,
            agreement: 0.5,
        });
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut last_log_step = 11usize;
        let mut last_log_time = Instant::now();
        let run_start = Instant::now();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));

        let outcome = run_epoch(
            EpochRunnerContext {
                epoch: 0,
                config: &config,
                manifest: &manifest,
                loader_config: &loader_config,
                artifacts: &artifacts,
                train_cfg: &train_cfg,
                loss_fn: &train_loss_fn,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                train_device: &device,
                session_start_global_step: 12,
                steps_to_skip: 0,
                microbatch_size: 4,
                use_amp: false,
                total_steps: 100,
                current_runtime: dummy_runtime_resume_contract(),
                run_start: &run_start,
                head_controller: &mut head_controller,
                cached_validation_samples: None,
            },
            EpochRuntimeMut {
                model: &mut model,
                optimizer: &mut optimizer,
                global_step: &mut global_step,
                best_validation: &mut best_validation,
                tb: &mut tb,
                training_log: &mut training_log,
                step_log: &mut step_log,
                last_log_step: &mut last_log_step,
                last_log_time: &mut last_log_time,
            },
        )
        .expect("run epoch with exhausted session budget");

        assert!(outcome.stop_after_epoch);
        assert_eq!(global_step, 12);
        assert_eq!(
            best_validation,
            Some(BestValidation {
                policy_loss: 0.4,
                agreement: 0.5,
            })
        );

        let state = read_resume_state(&artifacts.latest_state_path).expect("read latest state");
        assert_eq!(state.next_epoch, 1);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 0);
        assert_eq!(state.global_step, 12);
        assert_eq!(state.best_validation, best_validation);

        let entry = read_jsonl_entry(&artifacts.training_log_path);
        assert_eq!(entry["epoch"].as_u64(), Some(1));
        assert_eq!(entry["global_step"].as_u64(), Some(12));
        assert_eq!(entry["val_total_loss"], serde_json::Value::Null);
        assert_eq!(entry["best_val_policy_loss"].as_f64(), Some(0.4));
        assert!(!artifacts.best_model_base.with_extension("mpk").exists());
    }

    #[test]
    fn run_epoch_empty_manifest_completes_with_latest_state_and_best_checkpoint() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(false);
        let artifacts = test_artifacts("run_epoch_empty_manifest_checkpoint_contract");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let train_loss_fn = dummy_train_loss();
        let valid_loss_fn = dummy_valid_loss();
        let device = LibTorchDevice::Cpu;
        let mut model = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut global_step = 7usize;
        let mut best_validation = None;
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut last_log_step = 0usize;
        let mut last_log_time = Instant::now();
        let run_start = Instant::now();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));

        run_epoch(
            EpochRunnerContext {
                epoch: 1,
                config: &config,
                manifest: &manifest,
                loader_config: &loader_config,
                artifacts: &artifacts,
                train_cfg: &train_cfg,
                loss_fn: &train_loss_fn,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                train_device: &device,
                session_start_global_step: 0,
                steps_to_skip: 3,
                microbatch_size: 4,
                use_amp: false,
                total_steps: 100,
                current_runtime: dummy_runtime_resume_contract(),
                run_start: &run_start,
                head_controller: &mut head_controller,
                cached_validation_samples: None,
            },
            EpochRuntimeMut {
                model: &mut model,
                optimizer: &mut optimizer,
                global_step: &mut global_step,
                best_validation: &mut best_validation,
                tb: &mut tb,
                training_log: &mut training_log,
                step_log: &mut step_log,
                last_log_step: &mut last_log_step,
                last_log_time: &mut last_log_time,
            },
        )
        .expect("run epoch with empty manifest");

        let state = read_resume_state(&artifacts.latest_state_path).expect("read latest state");
        assert_eq!(state.next_epoch, 2);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 0);
        assert_eq!(state.global_step, 7);
        assert_eq!(state.best_validation, None);
        assert!(artifacts.latest_model_base.with_extension("mpk").exists());
        assert!(
            artifacts
                .latest_model_base
                .with_extension("meta.json")
                .exists()
        );
        assert!(
            artifacts
                .latest_optimizer_base
                .with_extension("bin")
                .exists()
        );
        assert!(artifacts.best_model_base.with_extension("mpk").exists());
        assert!(
            artifacts
                .best_model_base
                .with_extension("meta.json")
                .exists()
        );
    }

    #[test]
    fn latest_checkpoint_can_refresh_state_without_rewriting_payload_files() {
        let artifacts = test_artifacts("latest_checkpoint_refreshes_state_only");
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let optimizer = AdamConfig::new().init();

        save_latest_checkpoint_and_state(
            &artifacts,
            &model,
            &optimizer,
            LatestCheckpointState {
                global_step: 15,
                train_loss: 1.25,
                best_validation: Some(BestValidation {
                    policy_loss: 0.7,
                    agreement: 0.8,
                }),
                continuation: &EpochContinuation {
                    next_epoch: 3,
                    skip_optimizer_steps_in_epoch: 7,
                    epoch_completed: false,
                },
                runtime: dummy_runtime_resume_contract(),
            },
        )
        .expect("initial latest checkpoint save");

        let latest_model_path = artifacts.latest_model_base.with_extension("mpk");
        let latest_meta_path = artifacts.latest_model_base.with_extension("meta.json");
        let latest_optimizer_path = artifacts.latest_optimizer_base.with_extension("bin");
        let model_before = modified_time(&latest_model_path);
        let meta_before = modified_time(&latest_meta_path);
        let optimizer_before = modified_time(&latest_optimizer_path);
        let state_before = modified_time(&artifacts.latest_state_path);
        thread::sleep(std::time::Duration::from_millis(1100));

        save_latest_checkpoint_and_state(
            &artifacts,
            &model,
            &optimizer,
            LatestCheckpointState {
                global_step: 15,
                train_loss: 1.25,
                best_validation: Some(BestValidation {
                    policy_loss: 0.7,
                    agreement: 0.8,
                }),
                continuation: &EpochContinuation {
                    next_epoch: 4,
                    skip_optimizer_steps_in_epoch: 0,
                    epoch_completed: true,
                },
                runtime: dummy_runtime_resume_contract(),
            },
        )
        .expect("refresh latest checkpoint state without rewriting payloads");

        let state = read_resume_state(&artifacts.latest_state_path).expect("read latest state");
        assert_eq!(state.next_epoch, 4);
        assert_eq!(state.skip_optimizer_steps_in_epoch, 0);
        assert_eq!(state.global_step, 15);
        assert_eq!(modified_time(&latest_model_path), model_before);
        assert_eq!(modified_time(&latest_meta_path), meta_before);
        assert_eq!(modified_time(&latest_optimizer_path), optimizer_before);
        assert!(modified_time(&artifacts.latest_state_path) > state_before);
    }

    #[test]
    fn run_epoch_empty_manifest_records_bc_epoch_scope_order() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(false);
        let artifacts = test_artifacts("nvtx_bc_epoch_scope");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let train_loss_fn = dummy_train_loss();
        let valid_loss_fn = dummy_valid_loss();
        let device = LibTorchDevice::Cpu;
        let mut model = Some(tiny_dummy_model(&device));
        let mut optimizer = AdamConfig::new().init();
        let mut global_step = 4usize;
        let mut best_validation = None;
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut training_log =
            crate::artifacts::open_training_log_appender(&artifacts.training_log_path)
                .expect("open training log appender");
        let mut step_log = crate::artifacts::open_step_log_appender(&artifacts.step_log_path)
            .expect("open step log appender");
        let mut last_log_step = 0usize;
        let mut last_log_time = Instant::now();
        let run_start = Instant::now();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));

        let (_, events) = crate::nvtx::with_test_recorder(|| {
            run_epoch(
                EpochRunnerContext {
                    epoch: 1,
                    config: &config,
                    manifest: &manifest,
                    loader_config: &loader_config,
                    artifacts: &artifacts,
                    train_cfg: &train_cfg,
                    loss_fn: &train_loss_fn,
                    valid_loss_fn: &valid_loss_fn,
                    bc_exit_cfg: &BcExitConfig::default(),
                    train_device: &device,
                    session_start_global_step: 0,
                    steps_to_skip: 0,
                    microbatch_size: 4,
                    use_amp: false,
                    total_steps: 100,
                    current_runtime: dummy_runtime_resume_contract(),
                    run_start: &run_start,
                    head_controller: &mut head_controller,
                    cached_validation_samples: None,
                },
                EpochRuntimeMut {
                    model: &mut model,
                    optimizer: &mut optimizer,
                    global_step: &mut global_step,
                    best_validation: &mut best_validation,
                    tb: &mut tb,
                    training_log: &mut training_log,
                    step_log: &mut step_log,
                    last_log_step: &mut last_log_step,
                    last_log_time: &mut last_log_time,
                },
            )
            .expect("run epoch with empty manifest should succeed");
        });

        assert_eq!(
            events,
            vec![
                "push:bc_epoch",
                "push:checkpoint",
                "pop:checkpoint",
                "push:validation",
                "pop:validation",
                "push:checkpoint",
                "pop:checkpoint",
                "push:logging",
                "pop:logging",
                "pop:bc_epoch",
            ]
        );
    }
}
