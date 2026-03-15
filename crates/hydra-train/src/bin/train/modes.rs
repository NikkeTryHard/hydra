use colored::Colorize;

use hydra_train::model::HydraModelConfig;
use hydra_train::preflight::ProbeKind;

use super::artifacts::BcArtifactPaths;
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
use super::rl_runner::run_rl_training_loop;

pub(super) fn handle_preflight_mode(
    config_path: &std::path::Path,
    config: &TrainConfig,
) -> Result<(), String> {
    validate_config(config)?;
    configure_threads(config.num_threads)?;
    if config.rl.is_some() {
        let train_device = super::config::train_device(&config.device);
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
