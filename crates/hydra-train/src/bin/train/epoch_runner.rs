use std::collections::VecDeque;
use std::io::Write;
use std::time::Instant;

use burn::backend::libtorch::LibTorchDevice;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use colored::Colorize;
use indicatif::MultiProgress;
use tboard::EventWriter;

use hydra_train::data::pipeline::{stream_train_epoch, DataManifest, StreamingLoaderConfig};
use hydra_train::data::sample::{collate_batch_samples, MjaiSample};
use hydra_train::model::HydraModel;
use hydra_train::training::bc::{
    bc_total_with_exit, gated_bc_context, policy_agreement, target_actions_from_policy_target,
    BCTrainerConfig, BcExitConfig,
};
use hydra_train::training::head_gates::HeadActivationController;
use hydra_train::training::losses::HydraLoss;

use super::artifacts::{
    append_step_log, append_training_log, log_tensorboard, save_checkpoint,
    save_latest_checkpoint_and_state, write_delta_q_promotion_artifact, BcArtifactPaths,
    LatestCheckpointState, PersistedDeltaQPromotionArtifact,
};
use super::config::{validation_sample_limit, TrainConfig};
use super::presentation::{
    format_progress_message, make_bar, make_spinner, phase_label, timestamped,
};
use super::progress::{
    batch_stats_from_breakdown, BatchStats, EpochLogEntry, ScalarAverages, StepLogEntry,
};
use super::resume::{
    paused_training_message, BestValidation, EpochContinuation, RuntimeResumeContract,
};
use super::schedule::{effective_lr, lr_status_message, steps_per_second};
use super::status::{
    display_step_label, display_validation_scope_label, epoch_progress_message_with_rate,
    estimate_epoch_progress, reached_session_step_budget, session_steps_completed,
};
use super::validation::{
    is_better_validation, run_validation, ValidationContext, ValidationRuntime, ValidationSummary,
};
use super::{TrainBackend, ValidBackend};

pub(super) struct EpochRunnerContext<'a> {
    pub(super) epoch: usize,
    pub(super) config: &'a TrainConfig,
    pub(super) manifest: &'a DataManifest,
    pub(super) loader_config: &'a StreamingLoaderConfig,
    pub(super) artifacts: &'a BcArtifactPaths,
    pub(super) train_cfg: &'a BCTrainerConfig,
    pub(super) loss_fn: &'a HydraLoss<TrainBackend>,
    pub(super) valid_loss_fn: &'a HydraLoss<ValidBackend>,
    pub(super) bc_exit_cfg: &'a BcExitConfig,
    pub(super) train_device: &'a LibTorchDevice,
    pub(super) session_start_global_step: usize,
    pub(super) steps_to_skip: usize,
    pub(super) microbatch_size: usize,
    pub(super) total_steps: usize,
    pub(super) current_runtime: RuntimeResumeContract,
    pub(super) run_start: &'a Instant,
    pub(super) head_controller: &'a mut HeadActivationController,
}

pub(super) struct EpochRuntimeMut<'a, O, W>
where
    O: Optimizer<HydraModel<TrainBackend>, TrainBackend>,
    W: Write,
{
    pub(super) model: &'a mut HydraModel<TrainBackend>,
    pub(super) optimizer: &'a mut O,
    pub(super) global_step: &'a mut usize,
    pub(super) best_validation: &'a mut Option<BestValidation>,
    pub(super) tb: &'a mut Option<EventWriter<W>>,
    pub(super) last_log_step: &'a mut usize,
    pub(super) last_log_time: &'a mut Instant,
}

pub(super) struct EpochRunOutcome {
    pub(super) stop_after_epoch: bool,
}

struct TrainLogicalBatchConfig<'a> {
    microbatch_size: usize,
    augment: bool,
    train_device: &'a LibTorchDevice,
    loss_fn: &'a HydraLoss<TrainBackend>,
    bc_exit_cfg: &'a BcExitConfig,
    lr: f64,
}

struct ValidationStepContext<'a> {
    multi: &'a MultiProgress,
    config: &'a TrainConfig,
    loader_config: &'a StreamingLoaderConfig,
    manifest: &'a DataManifest,
    train_device: &'a LibTorchDevice,
    valid_loss_fn: &'a HydraLoss<ValidBackend>,
    bc_exit_cfg: &'a BcExitConfig,
    artifacts: &'a BcArtifactPaths,
    session_start_global_step: usize,
}

struct IntervalStepSummaryContext<'a> {
    artifacts: &'a BcArtifactPaths,
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

struct EpochEndValidationContext<'a> {
    config: &'a TrainConfig,
    loader_config: &'a StreamingLoaderConfig,
    manifest: &'a DataManifest,
    train_device: &'a LibTorchDevice,
    valid_loss_fn: &'a HydraLoss<ValidBackend>,
    bc_exit_cfg: &'a BcExitConfig,
    artifacts: &'a BcArtifactPaths,
}

struct EpochFinalizeContext<'a> {
    artifacts: &'a BcArtifactPaths,
    config: &'a TrainConfig,
    train_cfg: &'a BCTrainerConfig,
    epoch: usize,
    global_step: usize,
    train_stats: ScalarAverages,
    val_summary: Option<ValidationSummary>,
    best_validation: Option<BestValidation>,
    final_lr: f64,
}

fn should_run_epoch_end_validation(epoch: usize, num_epochs: usize, every_n_epochs: usize) -> bool {
    (epoch + 1).is_multiple_of(every_n_epochs) || epoch + 1 == num_epochs
}

fn build_epoch_continuation(
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

fn train_logical_batch<O>(
    logical_batch: &[MjaiSample],
    config: TrainLogicalBatchConfig<'_>,
    head_controller: &mut HeadActivationController,
    model: &mut HydraModel<TrainBackend>,
    optimizer: &mut O,
) -> Result<Vec<BatchStats>, String>
where
    O: Optimizer<HydraModel<TrainBackend>, TrainBackend>,
{
    let TrainLogicalBatchConfig {
        microbatch_size,
        augment,
        train_device,
        loss_fn,
        bc_exit_cfg,
        lr,
    } = config;
    if logical_batch.is_empty() {
        return Ok(Vec::new());
    }

    let mut accumulator: GradientsAccumulator<HydraModel<TrainBackend>> =
        GradientsAccumulator::new();
    let mut batch_stats = Vec::new();
    let logical_batch_len = logical_batch.len().max(1) as f32;

    for chunk in logical_batch.chunks(microbatch_size.max(1)) {
        let Some((obs, batch)) =
            collate_batch_samples::<TrainBackend>(chunk, augment, train_device)
                .map_err(|err| format!("training collation failed: {err}"))?
        else {
            continue;
        };
        let targets = batch.to_hydra_targets();
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let output = model.forward_with_warmup(obs.clone(), &active_loss_fn.config, &warmup_heads);
        let agreement = policy_agreement(
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            target_actions_from_policy_target(targets.policy_target.clone()),
        );
        let breakdown = active_loss_fn.total_loss(&output, &targets);
        let total = bc_total_with_exit(&output, &batch, &targets, &active_loss_fn, bc_exit_cfg);
        batch_stats.push(batch_stats_from_breakdown(
            chunk.len(),
            agreement,
            &breakdown,
        ));

        let chunk_weight = chunk.len() as f32 / logical_batch_len;
        let grads = (total * chunk_weight).backward();
        let grads = GradientsParams::from_grads(grads, model);
        accumulator.accumulate(model, grads);
    }

    if !batch_stats.is_empty() {
        let grads = accumulator.grads();
        *model = optimizer.step(lr, model.clone(), grads);
        head_controller.tick_warmup();
    }

    Ok(batch_stats)
}

fn record_drained_batch_stats(
    drained: Vec<BatchStats>,
    stats: &mut ScalarAverages,
    step_window: &mut ScalarAverages,
) {
    for batch_stats in drained {
        stats.record_batch(batch_stats);
        step_window.record_batch(batch_stats);
    }
}

fn maybe_run_interval_validation(
    context: ValidationStepContext<'_>,
    model: &HydraModel<TrainBackend>,
    head_controller: Option<&mut HeadActivationController>,
    best_validation: &mut Option<BestValidation>,
    global_step: usize,
    step_window_total_loss: f64,
) -> Result<Option<ValidationSummary>, String> {
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
            match validation_sample_limit(config) {
                Some(limit) => format!("target_samples={limit}").yellow(),
                None => "target_samples=all".yellow(),
            }
        )))
        .map_err(|err| format!("failed to print validation start summary: {err}"))?;

    let summary = run_validation(
        model,
        ValidationContext {
            config,
            loader_config,
            manifest,
            device: train_device,
            loss_fn: valid_loss_fn,
            exit_cfg: bc_exit_cfg,
        },
        ValidationRuntime {
            head_controller,
            progress: None,
        },
    )?;
    if is_better_validation(&summary, *best_validation) {
        *best_validation = Some(BestValidation {
            policy_loss: summary.policy_loss,
            agreement: summary.agreement,
        });
        save_checkpoint(
            model,
            &artifacts.best_model_base,
            global_step,
            step_window_total_loss,
            Some(&summary),
        )?;
    }

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
            summary
                .delta_q_promotion_snapshot
                .as_ref()
                .map(|report| format!(
                    " val_dq_lift={:.4} val_dq_regret={:.4}/{:.4} val_dq_win={:.2}% val_dq_offline_gate={}",
                    report.mean_decision_lift,
                    report.candidate_mean_regret,
                    report.baseline_mean_regret,
                    report.regret_beats_baseline_rate * 100.0,
                    report.passed
                ))
                .unwrap_or_default()
                .yellow(),
        )))
        .map_err(|err| format!("failed to print validation summary: {err}"))?;

    if let (Some(report), Some(result)) = (
        summary.delta_q_promotion.as_ref(),
        summary.delta_q_promotion_result.as_ref(),
    ) {
        write_delta_q_promotion_artifact(
            &artifacts.delta_q_promotion_path,
            &PersistedDeltaQPromotionArtifact {
                scope: "step_validation",
                step_or_epoch: global_step,
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

    Ok(Some(summary))
}

fn emit_interval_step_summary<W>(
    multi: &MultiProgress,
    tb: &mut Option<EventWriter<W>>,
    context: IntervalStepSummaryContext<'_>,
) -> Result<(), String>
where
    W: Write,
{
    let IntervalStepSummaryContext {
        artifacts,
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
    } = context;
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
        val_total_loss: val_summary.as_ref().map(|summary| summary.total_loss),
        val_policy_loss: val_summary.as_ref().map(|summary| summary.policy_loss),
        val_policy_agreement: val_summary.as_ref().map(|summary| summary.agreement),
        val_delta_q_promotion: val_summary
            .as_ref()
            .and_then(|summary| summary.delta_q_promotion_snapshot),
        best_val_policy_loss: best_validation.map(|best| best.policy_loss),
        best_val_agreement: best_validation.map(|best| best.agreement),
    };
    append_step_log(&artifacts.step_log_path, &step_entry)?;
    Ok(())
}

fn maybe_save_periodic_checkpoint<O>(
    model: &HydraModel<TrainBackend>,
    optimizer: &O,
    context: PeriodicCheckpointContext<'_>,
    state: PeriodicCheckpointState,
) -> Result<(), String>
where
    O: Optimizer<HydraModel<TrainBackend>, TrainBackend>,
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
        return Ok(());
    }

    let continuation = EpochContinuation {
        next_epoch: epoch,
        skip_optimizer_steps_in_epoch: epoch_optimizer_steps,
        epoch_completed: false,
    };
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
    )
}

fn emit_paused_training_message(continuation: &EpochContinuation) {
    println!(
        "{}",
        timestamped(format!(
            "{} {}",
            "Paused BC training".bold().cyan(),
            paused_training_message(continuation).yellow(),
        ))
    );
}

fn run_epoch_end_validation(
    epoch: usize,
    model: &HydraModel<TrainBackend>,
    context: EpochEndValidationContext<'_>,
    head_controller: Option<&mut HeadActivationController>,
    best_validation: &mut Option<BestValidation>,
    train_total_loss: f64,
) -> Result<Option<ValidationSummary>, String> {
    let EpochEndValidationContext {
        config,
        loader_config,
        manifest,
        train_device,
        valid_loss_fn,
        bc_exit_cfg,
        artifacts,
    } = context;
    if !should_run_epoch_end_validation(epoch, config.num_epochs, config.validation_every_n_epochs)
    {
        return Ok(None);
    }

    println!(
        "{}",
        timestamped(format!(
            "{} {}",
            "validation @ epoch end".bold().magenta(),
            match validation_sample_limit(config) {
                Some(limit) => format!("target_samples={limit}").yellow(),
                None => "target_samples=all".yellow(),
            }
        ))
    );
    let summary = run_validation(
        model,
        ValidationContext {
            config,
            loader_config,
            manifest,
            device: train_device,
            loss_fn: valid_loss_fn,
            exit_cfg: bc_exit_cfg,
        },
        ValidationRuntime {
            head_controller,
            progress: None,
        },
    )?;
    if is_better_validation(&summary, *best_validation) {
        *best_validation = Some(BestValidation {
            policy_loss: summary.policy_loss,
            agreement: summary.agreement,
        });
        save_checkpoint(
            model,
            &artifacts.best_model_base,
            epoch + 1,
            train_total_loss,
            Some(&summary),
        )?;
    }
    println!(
        "{}",
        timestamped(format!(
            "{} {} {} {} {}{}",
            "validation @ epoch end".bold().magenta(),
            format!("val_samples={}", summary.samples).yellow(),
            format!("val_policy_ce={:.4}", summary.policy_loss).yellow(),
            format!("val_total={:.4}", summary.total_loss).yellow(),
            format!("val_agree={:.2}%", summary.agreement * 100.0).yellow(),
            summary
                .delta_q_promotion_snapshot
                .as_ref()
                .map(|report| format!(
                    " val_dq_lift={:.4} val_dq_regret={:.4}/{:.4} val_dq_win={:.2}% val_dq_offline_gate={}",
                    report.mean_decision_lift,
                    report.candidate_mean_regret,
                    report.baseline_mean_regret,
                    report.regret_beats_baseline_rate * 100.0,
                    report.passed
                ))
                .unwrap_or_default()
                .yellow(),
        ))
    );
    if let (Some(report), Some(result)) = (
        summary.delta_q_promotion.as_ref(),
        summary.delta_q_promotion_result.as_ref(),
    ) {
        write_delta_q_promotion_artifact(
            &artifacts.delta_q_promotion_path,
            &PersistedDeltaQPromotionArtifact {
                scope: "epoch_validation",
                step_or_epoch: epoch + 1,
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
    Ok(Some(summary))
}

fn finalize_epoch_outputs<W>(
    tb: &mut Option<EventWriter<W>>,
    context: EpochFinalizeContext<'_>,
) -> Result<(), String>
where
    W: Write,
{
    let EpochFinalizeContext {
        artifacts,
        config,
        train_cfg,
        epoch,
        global_step,
        train_stats,
        val_summary,
        best_validation,
        final_lr,
    } = context;
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
        val_total_loss: val_summary.as_ref().map(|summary| summary.total_loss),
        val_policy_loss: val_summary.as_ref().map(|summary| summary.policy_loss),
        val_policy_agreement: val_summary.as_ref().map(|summary| summary.agreement),
        val_delta_q_promotion: val_summary
            .as_ref()
            .and_then(|summary| summary.delta_q_promotion_snapshot),
        best_val_policy_loss: best_validation.map(|best| best.policy_loss),
        best_val_agreement: best_validation.map(|best| best.agreement),
        num_batches: train_stats.num_batches,
    };
    append_training_log(&artifacts.training_log_path, &entry)?;

    let lr_message = lr_status_message(global_step, train_cfg.warmup_steps, final_lr);
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

    Ok(())
}

pub(super) fn run_epoch<O, W>(
    context: EpochRunnerContext<'_>,
    runtime: EpochRuntimeMut<'_, O, W>,
) -> Result<EpochRunOutcome, String>
where
    O: Optimizer<HydraModel<TrainBackend>, TrainBackend>,
    W: Write,
{
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
        total_steps,
        current_runtime,
        run_start,
        head_controller,
    } = context;
    let EpochRuntimeMut {
        model,
        optimizer,
        global_step,
        best_validation,
        tb,
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
            let drained = train_logical_batch(
                &logical_batch,
                TrainLogicalBatchConfig {
                    microbatch_size,
                    augment: config.augment,
                    train_device,
                    loss_fn,
                    bc_exit_cfg,
                    lr,
                },
                head_controller,
                model,
                optimizer,
            )?;

            record_drained_batch_stats(drained, &mut stats, &mut step_window);
            epoch_optimizer_steps += 1;
            *global_step += 1;
            train_pb.inc(1);
            let running_stats = stats.finalize();
            let lr_message = lr_status_message(*global_step, train_cfg.warmup_steps, lr);
            train_pb.set_message(format_progress_message(
                running_stats.total_loss,
                running_stats.policy_agreement,
                &lr_message,
                steps_per_second(
                    session_steps_completed(*global_step, session_start_global_step),
                    run_start.elapsed(),
                ),
            ));

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
                },
                model,
                Some(head_controller),
                best_validation,
                *global_step,
                step_window.finalize().total_loss,
            )?;

            if session_step > 0 && session_step.is_multiple_of(config.log_every_n_steps) {
                let window_stats = std::mem::take(&mut step_window).finalize();
                let window_steps = (*global_step).saturating_sub(*last_log_step);
                let step_rate = steps_per_second(window_steps, last_log_time.elapsed());
                *last_log_step = *global_step;
                *last_log_time = Instant::now();

                emit_interval_step_summary(
                    &multi,
                    tb,
                    IntervalStepSummaryContext {
                        artifacts,
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
                    },
                )?;
            }

            maybe_save_periodic_checkpoint(
                model,
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
            )?;

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
        let drained = train_logical_batch(
            &logical_batch,
            TrainLogicalBatchConfig {
                microbatch_size,
                augment: config.augment,
                train_device,
                loss_fn,
                bc_exit_cfg,
                lr,
            },
            head_controller,
            model,
            optimizer,
        )?;
        record_drained_batch_stats(drained, &mut stats, &mut step_window);
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
    save_latest_checkpoint_and_state(
        artifacts,
        model,
        optimizer,
        LatestCheckpointState {
            global_step: *global_step,
            train_loss: train_stats.total_loss,
            best_validation: *best_validation,
            continuation: &continuation,
            runtime: current_runtime,
        },
    )?;

    if !continuation.epoch_completed {
        emit_paused_training_message(&continuation);
        return Ok(EpochRunOutcome {
            stop_after_epoch: true,
        });
    }

    let val_summary = run_epoch_end_validation(
        epoch,
        model,
        EpochEndValidationContext {
            config,
            loader_config,
            manifest,
            train_device,
            valid_loss_fn,
            bc_exit_cfg,
            artifacts,
        },
        Some(head_controller),
        best_validation,
        train_stats.total_loss,
    )?;

    finalize_epoch_outputs(
        tb,
        EpochFinalizeContext {
            artifacts,
            config,
            train_cfg,
            epoch,
            global_step: *global_step,
            train_stats,
            val_summary,
            best_validation: *best_validation,
            final_lr,
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn::backend::libtorch::LibTorchDevice;
    use burn::optim::AdamConfig;
    use hydra_train::model::HydraModelConfig;
    use hydra_train::preflight::PreflightConfig;
    use hydra_train::training::head_gates::{HeadActivationConfig, HeadActivationController};
    use hydra_train::training::losses::HydraLossConfig;

    use crate::config::{BcHyperparamConfig, TrainConfig};
    use crate::resume::read_resume_state;

    fn batch_stats(sample_count: usize, total_loss: f64, policy_agreement: f64) -> BatchStats {
        BatchStats {
            sample_count,
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
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 0,
            advanced_loss: None,
            rl: None,
            bc: BcHyperparamConfig::default(),
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
            delta_q_promotion: None,
            delta_q_promotion_result: None,
            delta_q_promotion_snapshot: None,
            delta_q_policy_transfer: None,
            delta_q_policy_transfer_result: None,
            delta_q_policy_transfer_snapshot: None,
        }
    }

    fn dummy_runtime_resume_contract() -> RuntimeResumeContract {
        RuntimeResumeContract {
            batch_size: 16,
            train_microbatch_size: 4,
            validation_microbatch_size: 4,
            accum_steps: 4,
        }
    }

    fn temp_dir_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        PathBuf::from("/home/nikketryhard/tmp").join(format!("hydra_epoch_runner_{label}_{unique}"))
    }

    fn test_artifacts(label: &str) -> BcArtifactPaths {
        let output_dir = temp_dir_path(label);
        fs::create_dir_all(&output_dir).expect("create test output dir");
        let artifacts = BcArtifactPaths::new(&output_dir, 0);
        artifacts.create_root_dir().expect("create artifacts root");
        artifacts
    }

    fn dummy_model(device: &LibTorchDevice) -> HydraModel<TrainBackend> {
        HydraModelConfig::learner().init::<TrainBackend>(device)
    }

    fn dummy_valid_loss() -> HydraLoss<ValidBackend> {
        HydraLoss::<ValidBackend>::new(HydraLossConfig::new())
    }

    fn dummy_train_loss() -> HydraLoss<TrainBackend> {
        HydraLoss::<TrainBackend>::new(HydraLossConfig::new())
    }

    fn read_jsonl_entry(path: &Path) -> serde_json::Value {
        let raw = fs::read_to_string(path).expect("read jsonl file");
        let line = raw.lines().next().expect("jsonl entry line");
        serde_json::from_str(line).expect("parse jsonl entry")
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
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
    fn maybe_run_interval_validation_skips_until_step_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_skip");
        let device = LibTorchDevice::Cpu;
        let model = dummy_model(&device);
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
    fn maybe_save_periodic_checkpoint_skips_when_session_step_is_not_on_boundary() {
        let config = dummy_config();
        let artifacts = test_artifacts("periodic_checkpoint_skip");
        let device = LibTorchDevice::Cpu;
        let model = dummy_model(&device);
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
        assert!(!artifacts
            .latest_optimizer_base
            .with_extension("bin")
            .exists());
    }

    #[test]
    fn maybe_save_periodic_checkpoint_persists_resume_state_on_boundary() {
        let config = dummy_config();
        let artifacts = test_artifacts("periodic_checkpoint_save");
        let device = LibTorchDevice::Cpu;
        let model = dummy_model(&device);
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
        assert!(artifacts
            .latest_model_base
            .with_extension("meta.json")
            .exists());
        assert!(artifacts
            .latest_optimizer_base
            .with_extension("bin")
            .exists());
    }

    #[test]
    fn maybe_run_interval_validation_updates_best_and_saves_checkpoint_on_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_save");
        let device = LibTorchDevice::Cpu;
        let model = dummy_model(&device);
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
        assert!(artifacts
            .best_model_base
            .with_extension("meta.json")
            .exists());
    }

    #[test]
    fn maybe_run_interval_validation_keeps_existing_best_when_summary_is_not_better() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("interval_validation_keep_best");
        let device = LibTorchDevice::Cpu;
        let model = dummy_model(&device);
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
        let window_stats = ScalarAverages::default();

        emit_interval_step_summary(
            &multi,
            &mut tb,
            IntervalStepSummaryContext {
                artifacts: &artifacts,
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
            },
        )
        .expect("emit skipped validation interval summary");

        let entry = read_jsonl_entry(&artifacts.step_log_path);
        assert_eq!(entry["global_step"].as_u64(), Some(9));
        assert_eq!(entry["epoch"].as_u64(), Some(2));
        assert_close(entry["lr"].as_f64().expect("step log lr"), 1.0e-4);
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
        let mut window_stats = ScalarAverages::default();
        window_stats.record_batch(batch_stats(4, 2.5, 0.4));
        let window_stats = window_stats.finalize();
        let val_summary = dummy_validation_summary(0.9, 0.65);

        emit_interval_step_summary(
            &multi,
            &mut tb,
            IntervalStepSummaryContext {
                artifacts: &artifacts,
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
    }

    #[test]
    fn run_epoch_end_validation_returns_none_when_epoch_is_not_a_boundary() {
        let config = dummy_config();
        let loader_config = StreamingLoaderConfig::default();
        let manifest = dummy_manifest(true);
        let artifacts = test_artifacts("epoch_end_validation_skip");
        let device = LibTorchDevice::Cpu;
        let model = dummy_model(&device);
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
        let model = dummy_model(&device);
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
        assert!(artifacts
            .best_model_base
            .with_extension("meta.json")
            .exists());
    }

    #[test]
    fn finalize_epoch_outputs_writes_training_log_with_validation_metrics() {
        let config = dummy_config();
        let artifacts = test_artifacts("finalize_epoch_outputs");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
        let mut train_stats = ScalarAverages::default();
        train_stats.record_batch(batch_stats(4, 3.5, 0.55));
        let train_stats = train_stats.finalize();
        let val_summary = dummy_validation_summary(0.95, 0.68);

        finalize_epoch_outputs(
            &mut tb,
            EpochFinalizeContext {
                artifacts: &artifacts,
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
    }

    #[test]
    fn finalize_epoch_outputs_writes_skipped_validation_epoch_log() {
        let config = dummy_config();
        let artifacts = test_artifacts("finalize_epoch_outputs_skipped_validation");
        let train_cfg = BCTrainerConfig::new(HydraModelConfig::learner());
        let mut tb: Option<EventWriter<Vec<u8>>> = None;

        finalize_epoch_outputs(
            &mut tb,
            EpochFinalizeContext {
                artifacts: &artifacts,
                config: &config,
                train_cfg: &train_cfg,
                epoch: 0,
                global_step: 3,
                train_stats: ScalarAverages::default(),
                val_summary: None,
                best_validation: None,
                final_lr: 5.0e-5,
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
        let mut model = dummy_model(&device);
        let mut optimizer = AdamConfig::new().init();
        let mut global_step = 7usize;
        let mut best_validation = None;
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
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
                total_steps: 100,
                current_runtime: dummy_runtime_resume_contract(),
                run_start: &run_start,
                head_controller: &mut head_controller,
            },
            EpochRuntimeMut {
                model: &mut model,
                optimizer: &mut optimizer,
                global_step: &mut global_step,
                best_validation: &mut best_validation,
                tb: &mut tb,
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
        let mut model = dummy_model(&device);
        let mut optimizer = AdamConfig::new().init();
        let mut global_step = 12usize;
        let mut best_validation = Some(BestValidation {
            policy_loss: 0.4,
            agreement: 0.5,
        });
        let mut tb: Option<EventWriter<Vec<u8>>> = None;
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
                total_steps: 100,
                current_runtime: dummy_runtime_resume_contract(),
                run_start: &run_start,
                head_controller: &mut head_controller,
            },
            EpochRuntimeMut {
                model: &mut model,
                optimizer: &mut optimizer,
                global_step: &mut global_step,
                best_validation: &mut best_validation,
                tb: &mut tb,
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
}
