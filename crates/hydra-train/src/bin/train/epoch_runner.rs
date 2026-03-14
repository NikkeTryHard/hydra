use std::collections::VecDeque;
use std::io::Write;
use std::time::Instant;

use burn::backend::libtorch::LibTorchDevice;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use colored::Colorize;
use indicatif::MultiProgress;
use tboard::EventWriter;

use hydra_train::data::pipeline::{DataManifest, StreamingLoaderConfig, stream_train_epoch};
use hydra_train::data::sample::{MjaiSample, collate_batch_samples};
use hydra_train::model::HydraModel;
use hydra_train::training::bc::{
    BCTrainerConfig, BcExitConfig, bc_total_with_exit, policy_agreement,
    target_actions_from_policy_target,
};
use hydra_train::training::losses::HydraLoss;

use super::artifacts::{
    BcArtifactPaths, append_step_log, append_training_log, log_tensorboard, save_checkpoint,
    save_latest_checkpoint_and_state,
};
use super::config::{TrainConfig, validation_sample_limit};
use super::presentation::{
    format_progress_message, make_bar, make_spinner, phase_label, timestamped,
};
use super::progress::{
    BatchStats, EpochLogEntry, ScalarAverages, StepLogEntry, batch_stats_from_breakdown,
};
use super::resume::{
    BestValidation, EpochContinuation, RuntimeResumeContract, paused_training_message,
};
use super::schedule::{effective_lr, lr_status_message, steps_per_second};
use super::status::{
    display_step_label, display_validation_scope_label, epoch_progress_message_with_rate,
    estimate_epoch_progress, reached_session_step_budget, session_steps_completed,
};
use super::validation::{is_better_validation, run_validation};
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
    microbatch_size: usize,
    augment: bool,
    train_device: &LibTorchDevice,
    loss_fn: &HydraLoss<TrainBackend>,
    bc_exit_cfg: &BcExitConfig,
    model: &mut HydraModel<TrainBackend>,
    optimizer: &mut O,
    lr: f64,
) -> Result<Vec<BatchStats>, String>
where
    O: Optimizer<HydraModel<TrainBackend>, TrainBackend>,
{
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
        else {
            continue;
        };
        let targets = batch.to_hydra_targets();
        let output = model.forward(obs.clone());
        let agreement = policy_agreement(
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            target_actions_from_policy_target(targets.policy_target.clone()),
        );
        let breakdown = loss_fn.total_loss(&output, &targets);
        let total = bc_total_with_exit(&output, &batch, &targets, loss_fn, bc_exit_cfg);
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

fn maybe_run_interval_validation<O>(
    multi: &MultiProgress,
    model: &HydraModel<TrainBackend>,
    config: &TrainConfig,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    valid_loss_fn: &HydraLoss<ValidBackend>,
    bc_exit_cfg: &BcExitConfig,
    artifacts: &BcArtifactPaths,
    best_validation: &mut Option<BestValidation>,
    global_step: usize,
    session_start_global_step: usize,
    step_window_total_loss: f64,
) -> Result<Option<super::validation::ValidationSummary>, String>
where
    O: Optimizer<HydraModel<TrainBackend>, TrainBackend>,
{
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
        config,
        loader_config,
        manifest,
        train_device,
        valid_loss_fn,
        bc_exit_cfg,
        None,
    )?;
    if is_better_validation(summary, *best_validation) {
        *best_validation = Some(BestValidation {
            policy_loss: summary.policy_loss,
            agreement: summary.agreement,
        });
        save_checkpoint(
            model,
            &artifacts.best_model_base,
            global_step,
            step_window_total_loss,
            Some(summary),
        )?;
    }

    multi
        .println(timestamped(format!(
            "{} {} {} {} {}",
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
        )))
        .map_err(|err| format!("failed to print validation summary: {err}"))?;

    Ok(Some(summary))
}

fn emit_interval_step_summary<W>(
    multi: &MultiProgress,
    tb: &mut Option<EventWriter<W>>,
    artifacts: &BcArtifactPaths,
    manifest: &DataManifest,
    config: &TrainConfig,
    session_start_global_step: usize,
    global_step: usize,
    epoch: usize,
    lr: f64,
    best_validation: Option<BestValidation>,
    val_summary: Option<super::validation::ValidationSummary>,
    seen_samples: usize,
    assumed_games_seen: usize,
    epoch_optimizer_steps: usize,
    window_stats: ScalarAverages,
    step_rate: f64,
) -> Result<(), String>
where
    W: Write,
{
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
            val_summary,
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
        val_total_loss: val_summary.map(|summary| summary.total_loss),
        val_policy_loss: val_summary.map(|summary| summary.policy_loss),
        val_policy_agreement: val_summary.map(|summary| summary.agreement),
        best_val_policy_loss: best_validation.map(|best| best.policy_loss),
        best_val_agreement: best_validation.map(|best| best.agreement),
    };
    append_step_log(&artifacts.step_log_path, &step_entry)?;
    Ok(())
}

fn maybe_save_periodic_checkpoint<O>(
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    model: &HydraModel<TrainBackend>,
    optimizer: &O,
    global_step: usize,
    epoch: usize,
    epoch_optimizer_steps: usize,
    total_loss: f64,
    best_validation: Option<BestValidation>,
    current_runtime: RuntimeResumeContract,
    session_start_global_step: usize,
) -> Result<(), String>
where
    O: Optimizer<HydraModel<TrainBackend>, TrainBackend>,
{
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
        global_step,
        total_loss,
        best_validation,
        &continuation,
        current_runtime,
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
    config: &TrainConfig,
    loader_config: &StreamingLoaderConfig,
    manifest: &DataManifest,
    train_device: &LibTorchDevice,
    valid_loss_fn: &HydraLoss<ValidBackend>,
    bc_exit_cfg: &BcExitConfig,
    artifacts: &BcArtifactPaths,
    best_validation: &mut Option<BestValidation>,
    train_total_loss: f64,
) -> Result<Option<super::validation::ValidationSummary>, String> {
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
        config,
        loader_config,
        manifest,
        train_device,
        valid_loss_fn,
        bc_exit_cfg,
        None,
    )?;
    if is_better_validation(summary, *best_validation) {
        *best_validation = Some(BestValidation {
            policy_loss: summary.policy_loss,
            agreement: summary.agreement,
        });
        save_checkpoint(
            model,
            &artifacts.best_model_base,
            epoch + 1,
            train_total_loss,
            Some(summary),
        )?;
    }
    println!(
        "{}",
        timestamped(format!(
            "{} {} {} {} {}",
            "validation @ epoch end".bold().magenta(),
            format!("val_samples={}", summary.samples).yellow(),
            format!("val_policy_ce={:.4}", summary.policy_loss).yellow(),
            format!("val_total={:.4}", summary.total_loss).yellow(),
            format!("val_agree={:.2}%", summary.agreement * 100.0).yellow(),
        ))
    );
    Ok(Some(summary))
}

fn finalize_epoch_outputs<W>(
    tb: &mut Option<EventWriter<W>>,
    artifacts: &BcArtifactPaths,
    config: &TrainConfig,
    train_cfg: &BCTrainerConfig,
    epoch: usize,
    global_step: usize,
    train_stats: ScalarAverages,
    val_summary: Option<super::validation::ValidationSummary>,
    best_validation: Option<BestValidation>,
    final_lr: f64,
) -> Result<(), String>
where
    W: Write,
{
    if let Some(ref mut tb_writer) = tb.as_mut() {
        log_tensorboard(
            tb_writer,
            epoch + 1,
            &train_stats,
            val_summary,
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
                microbatch_size,
                config.augment,
                train_device,
                loss_fn,
                bc_exit_cfg,
                model,
                optimizer,
                lr,
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
            let val_summary = maybe_run_interval_validation::<O>(
                &multi,
                model,
                config,
                loader_config,
                manifest,
                train_device,
                valid_loss_fn,
                bc_exit_cfg,
                artifacts,
                best_validation,
                *global_step,
                session_start_global_step,
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
                    artifacts,
                    manifest,
                    config,
                    session_start_global_step,
                    *global_step,
                    epoch,
                    lr,
                    *best_validation,
                    val_summary,
                    seen_samples,
                    assumed_games_seen,
                    epoch_optimizer_steps,
                    window_stats,
                    step_rate,
                )?;
            }

            maybe_save_periodic_checkpoint(
                config,
                artifacts,
                model,
                optimizer,
                *global_step,
                epoch,
                epoch_optimizer_steps,
                stats.finalize().total_loss,
                *best_validation,
                current_runtime,
                session_start_global_step,
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
            microbatch_size,
            config.augment,
            train_device,
            loss_fn,
            bc_exit_cfg,
            model,
            optimizer,
            lr,
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
        *global_step,
        train_stats.total_loss,
        *best_validation,
        &continuation,
        current_runtime,
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
        config,
        loader_config,
        manifest,
        train_device,
        valid_loss_fn,
        bc_exit_cfg,
        artifacts,
        best_validation,
        train_stats.total_loss,
    )?;

    finalize_epoch_outputs(
        tb,
        artifacts,
        config,
        train_cfg,
        epoch,
        *global_step,
        train_stats,
        val_summary,
        *best_validation,
        final_lr,
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

    #[test]
    fn epoch_end_validation_runs_on_interval_or_final_epoch() {
        assert!(should_run_epoch_end_validation(0, 3, 1));
        assert!(!should_run_epoch_end_validation(0, 3, 2));
        assert!(should_run_epoch_end_validation(1, 3, 2));
        assert!(should_run_epoch_end_validation(2, 3, 5));
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
}
