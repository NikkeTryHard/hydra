use colored::Colorize;
use std::path::PathBuf;

use burn::prelude::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::ProbeKind;
use hydra_train::training::delta_q_promotion::DeltaQPromotionRecommendation;

use super::artifacts::{
    write_delta_q_promotion_artifact, BcArtifactPaths, PersistedDeltaQPromotionArtifact,
};
use super::bootstrap::{initialize_rl_training_bootstrap, RlTrainingBootstrap, RlTrainingRuntime};
use super::bootstrap::{initialize_training_bootstrap, TrainingBootstrap, TrainingRuntime};
use super::config::{configure_threads, device_label, validate_config, TrainConfig};
use super::epoch_runner::{run_epoch, EpochRunnerContext, EpochRuntimeMut};
use super::preflight_runtime::{run_preflight, run_probe_ladder_only, run_rl_preflight};
use super::presentation::{
    explicit_preflight_recommendation, explicit_preflight_summary, format_preflight_selection_line,
    format_preflight_summary_line, format_probe_results_table, format_status_line,
    format_warning_line, print_banner, print_preflight_banner, timestamped,
};
use super::probe_request::ProbeRequest;
use super::probe_summary::{best_probe_summary, format_probe_selection_summary, probe_kind_name};
use super::resume::checkpoint_base_from_path;
use super::rl_runner::run_rl_training_loop;
use super::validation::{
    run_validation_with_policy_baseline, ValidationContext, ValidationRuntime,
};

pub(super) fn handle_preflight_mode(
    config_path: &std::path::Path,
    config: &TrainConfig,
) -> Result<(), String> {
    validate_config(config)?;
    configure_threads(config.num_threads)?;
    if config.rl.is_some() {
        let train_device = super::config::train_device(&config.device)?;
        let device_name = device_label(&config.device);
        print_preflight_banner("Hydra RL preflight", config, &device_name);
        let preflight = run_rl_preflight(config_path, config, &train_device)?;
        println!(
            "{}",
            format_preflight_summary_line(
                "Preflight:",
                format!(
                    "selected rl.games_per_batch={} rl.microbatch_size={}",
                    preflight.selected_games_per_batch, preflight.selected_microbatch_size,
                )
            )
        );
        print_probe_table(
            "RL preflight games table",
            ProbeKind::RlGames,
            &preflight.rl_games_probe_results,
            preflight.selected_games_per_batch,
        );
        print_probe_table(
            "RL preflight microbatch table",
            ProbeKind::RlMicrobatch,
            &preflight.rl_microbatch_probe_results,
            preflight.selected_microbatch_size,
        );
        return Ok(());
    }
    let artifacts = BcArtifactPaths::new(&config.output_dir, 0);
    artifacts.create_root_dir()?;
    let device_name = device_label(&config.device);
    print_preflight_banner("Hydra preflight", config, &device_name);
    let preflight = run_preflight(
        config_path,
        config,
        &HydraModelConfig::learner(),
        &device_name,
        &artifacts,
    )?;
    println!(
        "{}",
        format_preflight_summary_line(
            "Preflight:",
            explicit_preflight_summary(preflight.runtime, preflight.explicit)
        )
    );
    print_probe_table(
        "Preflight train table",
        ProbeKind::Train,
        &preflight.train_probe_results,
        preflight.runtime.selected.train_microbatch_size,
    );
    print_probe_table(
        "Preflight validation table",
        ProbeKind::Validation,
        &preflight.validation_probe_results,
        preflight.runtime.selected.validation_microbatch_size,
    );
    Ok(())
}

pub(super) fn handle_probe_mode(
    config_path: &std::path::Path,
    config: &TrainConfig,
    request: ProbeRequest,
) -> Result<(), String> {
    validate_config(config)?;
    configure_threads(config.num_threads)?;
    let artifacts = BcArtifactPaths::new(&config.output_dir, 0);
    artifacts.create_root_dir()?;
    print_preflight_banner("Hydra probe-only", config, &device_label(&config.device));
    println!(
        "{}",
        format_status_line(
            "Probe-only:",
            format!(
                "kind={} candidate_mb={} warmup_steps={} measure_steps= {}",
                probe_kind_name(request.kind),
                request.candidate_microbatch,
                request.warmup_steps,
                request.measure_steps,
            )
            .replace("measure_steps= ", "measure_steps=")
        )
    );
    let (selected, results) = run_probe_ladder_only(config_path, config, &artifacts, request)?;
    let selected_summary = best_probe_summary(&results).ok_or_else(|| {
        format!(
            "no stable {} probe result found",
            probe_kind_name(request.kind)
        )
    })?;
    println!(
        "{}",
        format_preflight_selection_line(format_probe_selection_summary(
            request.kind,
            &selected_summary,
        ))
    );
    println!(
        "{}",
        format_status_line(
            "Probe best candidate:",
            format!("{}={}", probe_kind_name(request.kind), selected)
        )
    );
    print_probe_table("Probe final table", request.kind, &results, selected);
    Ok(())
}

pub(super) fn handle_training_mode(
    config_path: &std::path::Path,
    config: TrainConfig,
) -> Result<(), String> {
    println!(
        "{}",
        format_warning_line(explicit_preflight_recommendation())
    );
    if let Some(rl_cfg) = config.rl.clone() {
        let (bootstrap, runtime) = initialize_rl_training_bootstrap(config_path, config, rl_cfg)?;
        let RlTrainingBootstrap {
            config: _,
            rl_config,
            artifacts,
            model_config,
            device_name,
            ..
        } = &bootstrap;
        println!(
            "{}",
            timestamped(format!(
                "{} mode=rl phase={:?} games_per_batch={} device={} artifacts={} model={} ",
                "Hydra RL training".bold().cyan(),
                rl_config.phase,
                rl_config.games_per_batch,
                device_name,
                artifacts.root.display(),
                if model_config.is_learner() {
                    "learner"
                } else {
                    "actor"
                },
            ))
        );
        let _runtime: RlTrainingRuntime = runtime;
        return run_rl_training_loop(bootstrap, _runtime);
    }
    let (bootstrap, runtime) = initialize_training_bootstrap(config_path, config)?;
    let TrainingBootstrap {
        config,
        resume,
        artifacts,
        loader_config,
        manifest,
        train_cfg,
        model_config,
        device_name,
        train_device,
        current_runtime,
        session_start_global_step,
        total_steps,
        microbatch_size,
        banner_stats,
        loss_fn,
        valid_loss_fn,
        bc_exit_cfg,
    } = bootstrap;
    let TrainingRuntime {
        mut model,
        mut optimizer,
        mut best_validation,
        mut global_step,
        run_start,
        mut last_log_step,
        mut last_log_time,
        mut tb,
        mut head_controller,
    } = runtime;

    print_banner(
        &model_config,
        &config,
        &artifacts,
        &device_name,
        &banner_stats,
        &train_cfg,
    );
    resume.print_banner();

    for epoch in resume.start_epoch..config.num_epochs {
        let outcome = run_epoch(
            EpochRunnerContext {
                epoch,
                config: &config,
                manifest: &manifest,
                loader_config: &loader_config,
                artifacts: &artifacts,
                train_cfg: &train_cfg,
                loss_fn: &loss_fn,
                valid_loss_fn: &valid_loss_fn,
                bc_exit_cfg: &bc_exit_cfg,
                train_device: &train_device,
                session_start_global_step,
                steps_to_skip: resume.steps_to_skip_for_epoch(epoch),
                microbatch_size,
                total_steps,
                current_runtime,
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
        )?;
        if outcome.stop_after_epoch {
            break;
        }
    }

    println!(
        "{}",
        timestamped(format!(
            "{} {}",
            "Finished BC training. Best validation policy CE:"
                .bold()
                .cyan(),
            if let Some(best_validation) = best_validation {
                format!(
                    "{:.4} (agree {:.2}%)",
                    best_validation.policy_loss,
                    best_validation.agreement * 100.0
                )
            } else {
                "n/a".to_string()
            }
            .bold()
            .green()
        ))
    );

    Ok(())
}

pub(super) fn handle_delta_q_promotion_mode(
    config_path: &std::path::Path,
    config: TrainConfig,
    baseline_checkpoint: Option<PathBuf>,
) -> Result<(), String> {
    let (bootstrap, runtime) = initialize_training_bootstrap(config_path, config)?;
    let TrainingBootstrap {
        config,
        artifacts,
        loader_config,
        manifest,
        model_config,
        device_name,
        train_device,
        valid_loss_fn,
        bc_exit_cfg,
        ..
    } = bootstrap;
    let TrainingRuntime {
        model,
        mut head_controller,
        ..
    } = runtime;
    let baseline_model = if let Some(path) = baseline_checkpoint.as_ref() {
        let checkpoint_base = checkpoint_base_from_path(path);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        HydraModelConfig::learner()
            .init::<super::TrainBackend>(&train_device)
            .load_file(&checkpoint_base, &recorder, &train_device)
            .map_err(|err| {
                format!(
                    "failed to load delta_q baseline checkpoint {}: {err}",
                    checkpoint_base.display()
                )
            })?
    } else {
        model.clone()
    };

    println!(
        "{}",
        timestamped(format!(
            "{} device={} artifacts={} model={}",
            "Hydra DeltaQ offline/transfer gate".bold().cyan(),
            device_name,
            artifacts.root.display(),
            if model_config.is_learner() {
                "learner"
            } else {
                "actor"
            },
        ))
    );

    let summary = run_validation_with_policy_baseline(
        &model,
        &baseline_model,
        ValidationContext {
            config: &config,
            loader_config: &loader_config,
            manifest: &manifest,
            device: &train_device,
            loss_fn: &valid_loss_fn,
            exit_cfg: &bc_exit_cfg,
        },
        ValidationRuntime {
            head_controller: Some(&mut head_controller),
            progress: None,
        },
    )?;

    let (Some(report), Some(result), Some(snapshot), transfer_result) = (
        summary.delta_q_promotion.as_ref(),
        summary.delta_q_promotion_result.as_ref(),
        summary.delta_q_promotion_snapshot,
        summary.delta_q_policy_transfer_result.as_ref(),
    ) else {
        return Err(
            "delta_q promotion mode requires active delta_q targets in validation batches"
                .to_string(),
        );
    };
    let final_recommendation = if result.passed && transfer_result.map(|r| r.passed).unwrap_or(true)
    {
        DeltaQPromotionRecommendation::RequiresArenaConfirmation
    } else {
        DeltaQPromotionRecommendation::RejectAtOfflineGate
    };

    write_delta_q_promotion_artifact(
        &artifacts.delta_q_promotion_path,
        &PersistedDeltaQPromotionArtifact {
            scope: "promotion_mode",
            step_or_epoch: 0,
            recommendation: final_recommendation,
            stage: "offline_and_policy_transfer_gate",
            arena_confirmation: (final_recommendation
                == DeltaQPromotionRecommendation::RequiresArenaConfirmation)
                .then_some(Default::default()),
            arena_report: None,
            report,
            result,
            policy_transfer: summary.delta_q_policy_transfer.as_ref(),
            policy_transfer_result: transfer_result,
        },
    )?;

    println!(
        "{}",
        timestamped(format!(
            "{} samples={} compared={} dq_lift={:.4} dq_regret={:.4}/{:.4} dq_win={:.2}% dq_offline_gate={} next={} arena_req='{}' artifact={}",
            "DeltaQ offline gate".bold().magenta(),
            summary.samples,
            snapshot.compared_states,
            snapshot.mean_decision_lift,
            snapshot.candidate_mean_regret,
            snapshot.baseline_mean_regret,
            snapshot.regret_beats_baseline_rate * 100.0,
            snapshot.passed,
            final_recommendation,
            if final_recommendation == DeltaQPromotionRecommendation::RequiresArenaConfirmation {
                hydra_train::training::delta_q_promotion::DeltaQArenaConfirmationRequest::default()
                    .summary()
            } else {
                "n/a".to_string()
            },
            artifacts.delta_q_promotion_path.display(),
        ))
    );
    if let Some(transfer) = summary.delta_q_policy_transfer_snapshot {
        println!(
            "{}",
            timestamped(format!(
                "{} compared={} policy_regret={:.4}/{:.4} policy_top1={:.2}%/{:.2}% policy_beats_baseline={:.2}% candidate_worse_rate={:.2}%",
                "DeltaQ policy-vs-teacher holdout".bold().blue(),
                transfer.compared_states,
                transfer.candidate_policy_mean_teacher_regret,
                transfer.baseline_policy_mean_teacher_regret,
                transfer.candidate_policy_top1_to_teacher * 100.0,
                transfer.baseline_policy_top1_to_teacher * 100.0,
                transfer.candidate_beats_baseline_rate * 100.0,
                transfer.negative_transfer_fraction * 100.0,
            ))
        );
    }
    if let Some(transfer_result) = transfer_result {
        println!(
            "{}",
            timestamped(format!(
                "{} pass={} next={}",
                "DeltaQ policy transfer gate".bold().blue(),
                transfer_result.passed,
                transfer_result.recommendation(),
            ))
        );
    }

    Ok(())
}

fn print_probe_table(
    title: &str,
    kind: ProbeKind,
    results: &[hydra_train::preflight::ProbeResult],
    selected: usize,
) {
    println!(
        "{}",
        timestamped(format!(
            "{}\n{}",
            title.bold().cyan(),
            format_probe_results_table(kind, results, Some(selected))
        ))
    );
}
