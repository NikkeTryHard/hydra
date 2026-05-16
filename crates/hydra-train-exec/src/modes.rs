//! Train binary mode dispatch facade.
//!
//! This module owns the CLI mode selection order and default train-mode bodies
//! without depending on the compatibility `hydra-train` crate.

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::backend::libtorch::{LibTorchDevice, TchTensor};
use burn::tensor::backend::{AutodiffBackend, Backend};
use colored::Colorize;
use hydra_model::model::HydraModelConfig;
use hydra_train_runtime::config::{
    TrainCli, TrainConfig, display_num_threads, require_explicit_preflight_tuning_mode,
};

use crate::config_runtime::{configure_threads, device_label, train_device, validate_config};
use hydra_train_runtime::preflight::{
    EffectiveRuntimeConfig, ExplicitSettings, ProbeKind, ProbeResult,
};
use hydra_train_runtime::probe_request::{ProbeRequest, probe_request_from_cli};
use hydra_train_types::config::BCTrainerConfig;

use crate::advisory::{AdvisoryDeduper, AdvisoryEvent, startup_runtime_advisories};
use crate::artifacts::{BcArtifactPaths, append_advisory_event_to_writer};
use crate::bootstrap::{
    RlTrainingBootstrap, RlTrainingRuntime, TrainBackend, TrainingBootstrap, TrainingReaders,
    TrainingRuntime, initialize_rl_training_bootstrap, initialize_training_bootstrap,
};
use crate::data_pipeline::TrainValidationLoader;
use crate::delta_q_promotion::handle_delta_q_promotion_mode as run_delta_q_promotion_mode;
use crate::epoch_runner::{EpochRunnerContext, EpochRuntimeMut, run_epoch};
use crate::preflight_runtime::{run_preflight, run_probe_ladder_only, run_rl_preflight};
use crate::presentation::{
    BcHyperparamSummaryInput, bc_hyperparam_summary, explicit_preflight_recommendation,
    explicit_preflight_summary, format_advisory_line, format_preflight_selection_line,
    format_preflight_summary_line, format_probe_results_table, format_status_line,
    format_timed_phase_message, format_warning_line, precision_runtime_summary, print_banner_field,
    print_header_block, timestamped, unsafe_preflight_math_summary,
};
use crate::probe_summary::{best_probe_summary, format_probe_selection_summary, probe_kind_name};
use crate::resume::BestValidation;
use crate::rl_runner::run_rl_training_loop;
use crate::validation_runner::materialize_validation_samples;

/// Runs explicit preflight mode for BC or RL training.
pub fn handle_preflight_mode(config_path: &Path, config: &TrainConfig) -> Result<(), String> {
    let preflight_wall_start = Instant::now();
    require_explicit_preflight_tuning_mode(config_path)?;
    validate_config(config)?;
    configure_threads(config.num_threads)?;
    if config.rl.is_some() {
        let train_device = train_device(&config.device)?;
        let device_name = device_label(&config.device);
        print_preflight_banner("Hydra RL preflight", config, &device_name);
        let preflight = run_rl_preflight(config_path, config, &train_device)?;
        println!(
            "{}",
            format_rl_preflight_selection_message(preflight.runtime)
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
        println!(
            "{}",
            format_timed_phase_message(
                "preflight_wall_clock",
                "total elapsed including output",
                preflight_wall_start.elapsed().as_secs_f64(),
            )
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
        format_bc_preflight_selection_message(
            preflight.runtime,
            preflight.explicit,
            config.preflight.tuning_mode,
        )
    );
    if let Some(benchmark) = preflight.benchmark.as_ref() {
        println!(
            "{}",
            format_preflight_selection_line(format!(
                "benchmark winner tuning_mode={:?} benchmark_mode={:?} wall_clock_effective={:.2} samples/s train_only={:.2} train_mb={} val_mb={} loader=({}, {}, {}, {:?}) {}",
                config.preflight.tuning_mode,
                benchmark.metadata.mode,
                benchmark.score.wall_clock_samples_per_second,
                benchmark.score.train_only_samples_per_second,
                benchmark.runtime.train_microbatch_size,
                benchmark.runtime.validation_microbatch_size,
                benchmark.runtime.loader.archive_queue_bound,
                benchmark.runtime.loader.buffer_samples,
                benchmark.runtime.loader.buffer_games,
                benchmark.runtime.loader.num_threads,
                precision_runtime_summary(
                    preflight.runtime.requested_precision,
                    preflight.runtime.effective_precision
                ),
            ))
        );
    }
    if let Some(math_summary) = unsafe_preflight_math_summary(preflight.runtime, &config.bc) {
        println!("{}", format_preflight_selection_line(math_summary));
    }
    for advisory in &preflight.advisories {
        println!("{}", format_advisory_line(advisory));
    }
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
    println!(
        "{}",
        format_timed_phase_message(
            "preflight_wall_clock",
            "total elapsed including output",
            preflight_wall_start.elapsed().as_secs_f64(),
        )
    );
    Ok(())
}

/// Runs explicit probe-only mode for BC or RL probe requests.
pub fn handle_probe_mode(
    config_path: &Path,
    config: &TrainConfig,
    request: ProbeRequest,
) -> Result<(), String> {
    validate_config(config)?;
    configure_threads(config.num_threads)?;
    let artifacts = BcArtifactPaths::new(&config.output_dir, 0);
    artifacts.create_root_dir()?;
    print_preflight_banner("Hydra probe-only", config, &device_label(&config.device));
    println!("{}", format_probe_only_status_message(request));
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
            format_probe_best_candidate_detail(request.kind, selected)
        )
    );
    println!(
        "{}",
        format_probe_table_message("Probe final table", request.kind, &results, selected)
    );
    Ok(())
}

/// Prints the preflight banner shared by BC, RL, and probe-only execution.
pub fn print_preflight_banner(title: &str, config: &TrainConfig, device_name: &str) {
    print_header_block(title);
    print_banner_field("Device", device_name.green());
    print_banner_field("Dataset", config.data_dir.display().to_string().green());
    print_banner_field(
        "Optimizer batch",
        format!("{} samples", config.batch_size).yellow(),
    );
    print_banner_field(
        "Runtime defaults",
        format!(
            "train_mb={} val_mb={} threads={} buffer_games={} buffer_samples={} archive_queue_bound={} requested_precision={} effective_precision={}",
            config.microbatch_size.unwrap_or(config.batch_size),
            config
                .validation_microbatch_size
                .unwrap_or(config.microbatch_size.unwrap_or(config.batch_size)),
            display_num_threads(config.num_threads),
            config.buffer_games,
            config.buffer_samples,
            config.archive_queue_bound,
            hydra_train_runtime::preflight::requested_precision_signature(config.precision_mode),
            config.effective_precision(),
        )
        .yellow(),
    );
    println!();
}

/// Formats the RL preflight selected runtime line.
pub fn format_rl_preflight_selection_message(runtime: EffectiveRuntimeConfig) -> String {
    format_preflight_summary_line(
        "Preflight:",
        format!(
            "selected rl.games_per_batch={} rl.microbatch_size={} {}",
            runtime.loader.buffer_games,
            runtime.selected.train_microbatch_size,
            precision_runtime_summary(runtime.requested_precision, runtime.effective_precision),
        ),
    )
}

/// Formats the BC preflight selected runtime line.
pub fn format_bc_preflight_selection_message(
    runtime: EffectiveRuntimeConfig,
    explicit: ExplicitSettings,
    tuning_mode: hydra_train_runtime::preflight::PreflightTuningMode,
) -> String {
    format_preflight_summary_line(
        "Preflight:",
        explicit_preflight_summary(runtime, explicit, tuning_mode),
    )
}

/// Formats a preflight/probe table with title and selected candidate marker.
pub fn format_probe_table_message(
    title: &str,
    kind: ProbeKind,
    results: &[ProbeResult],
    selected: usize,
) -> String {
    timestamped(format!(
        "{}\n{}",
        title.bold().cyan(),
        format_probe_results_table(kind, results, Some(selected))
    ))
}

fn print_probe_table(title: &str, kind: ProbeKind, results: &[ProbeResult], selected: usize) {
    println!(
        "{}",
        format_probe_table_message(title, kind, results, selected)
    );
}

/// Formats probe-only request status detail.
pub fn format_probe_only_status_detail(request: ProbeRequest) -> String {
    format!(
        "kind={} candidate_mb={} warmup_steps={} measure_steps={}",
        probe_kind_name(request.kind),
        request.candidate_microbatch,
        request.warmup_steps,
        request.measure_steps,
    )
}

/// Formats probe-only request status message.
pub fn format_probe_only_status_message(request: ProbeRequest) -> String {
    format_status_line("Probe-only:", format_probe_only_status_detail(request))
}

/// Formats the selected probe candidate detail.
pub fn format_probe_best_candidate_detail(kind: ProbeKind, selected: usize) -> String {
    format!("{}={}", probe_kind_name(kind), selected)
}

type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

fn bc_hyperparam_summary_input(train_cfg: &BCTrainerConfig) -> BcHyperparamSummaryInput {
    BcHyperparamSummaryInput {
        lr: train_cfg.lr,
        min_learning_rate: train_cfg.min_learning_rate,
        weight_decay: train_cfg.weight_decay.into(),
        grad_clip_norm: train_cfg.grad_clip_norm.into(),
        warmup_steps: train_cfg.warmup_steps,
    }
}

fn model_kind(config: &HydraModelConfig) -> &'static str {
    if config.is_learner() {
        "learner"
    } else {
        "actor"
    }
}

fn optimized_path_summary(config: &TrainConfig) -> String {
    let shard_input = config.bc_shards_manifest_path.is_some();
    let pinned_staging = cfg!(feature = "cuda-graph") && shard_input;
    let preallocated_tensors = pinned_staging;
    let copy_compute_overlap = if pinned_staging {
        "unproven-single-buffer"
    } else {
        "off"
    };
    format!(
        "input={} pinned_h2d={} prealloc_gpu_tensors={} cuda_graph_replay={} copy_compute_overlap={}",
        if shard_input {
            "bc_shards"
        } else {
            "raw_replay"
        },
        if pinned_staging { "on" } else { "off" },
        if preallocated_tensors { "on" } else { "off" },
        crate::presentation::cuda_graph_replay_label(),
        copy_compute_overlap,
    )
}

fn print_bc_training_banner(
    model_config: &HydraModelConfig,
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    device_name: &str,
    stats: &hydra_train_runtime::progress::BannerStats,
    train_hyperparams: BcHyperparamSummaryInput,
) {
    print_header_block("Hydra BC trainer");
    print_banner_field(
        "Model",
        format!(
            "{} ({} blocks, {}ch)",
            model_kind(model_config),
            model_config.num_blocks,
            model_config.hidden_channels
        )
        .green(),
    );
    print_banner_field("Device", device_name.green());
    print_banner_field(
        "Dataset",
        if stats.counts_exact {
            format!(
                "{} ({} sources, {} games)",
                config.data_dir.display(),
                stats.total_sources,
                stats.total_games
            )
        } else {
            format!(
                "{} ({} sources, archive counts deferred)",
                config.data_dir.display(),
                stats.total_sources,
            )
        }
        .green(),
    );
    print_banner_field(
        "Train",
        if stats.counts_exact {
            format!(
                "{} games | Val: {} games",
                stats.train_count, stats.val_count
            )
        } else {
            "streaming split, counts estimated while loading".to_string()
        }
        .green(),
    );
    print_banner_field(
        "Buffer",
        format!(
            "{} samples (max {} games, archive_queue_bound={}, threads={})",
            config.buffer_samples,
            config.buffer_games,
            config.archive_queue_bound,
            display_num_threads(config.num_threads)
        )
        .yellow(),
    );
    print_banner_field(
        "Optimizer batch",
        format!(
            "{} ({} x {} accum) requested_precision={} effective_precision={}",
            config.batch_size,
            config.microbatch_size.unwrap_or(config.batch_size),
            stats.accum_steps,
            hydra_train_runtime::preflight::requested_precision_signature(config.precision_mode),
            config.effective_precision(),
        )
        .yellow(),
    );
    print_banner_field("Optimized path", optimized_path_summary(config).yellow());
    print_banner_field(
        "BC hyperparams",
        bc_hyperparam_summary(train_hyperparams).yellow(),
    );
    print_banner_field("Epochs", config.num_epochs.to_string().yellow());
    print_banner_field(
        "Schedule",
        format!(
            "warmup+cosine (warmup_steps={}, max_train_steps={})",
            train_hyperparams.warmup_steps,
            config
                .max_train_steps
                .map(|steps| steps.to_string())
                .unwrap_or_else(|| "epoch-derived".to_string())
        )
        .yellow(),
    );
    print_banner_field("Output", artifacts.root.display().to_string().green());
    print_banner_field(
        "TBoard",
        if config.tensorboard {
            artifacts.tb_session_dir.display().to_string().green()
        } else {
            "disabled".yellow()
        },
    );
    println!();
}

fn run_bc_training_mode_for_backend<B>(
    bootstrap: TrainingBootstrap<B>,
    runtime: TrainingRuntime<B>,
    readers: TrainingReaders,
) -> Result<(), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<FloatTensorPrimitive = TchTensor, IntTensorPrimitive = TchTensor>,
{
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
        microbatch_explicitness,
        session_start_global_step,
        total_steps,
        microbatch_size,
        use_amp,
        banner_stats,
        loss_fn,
        valid_loss_fn,
        bc_exit_cfg,
    } = bootstrap;
    let TrainingRuntime {
        model,
        mut optimizer,
        mut best_validation,
        mut global_step,
        run_start,
        mut last_log_step,
        mut last_log_time,
        mut tb,
        mut training_log,
        mut step_log,
        mut head_controller,
    } = runtime;
    let TrainingReaders = readers;

    print_bc_training_banner(
        &model_config,
        &config,
        &artifacts,
        &device_name,
        &banner_stats,
        bc_hyperparam_summary_input(&train_cfg),
    );
    resume.print_banner_with_effective_runtime(Some(current_runtime));
    let mut advisory_deduper = AdvisoryDeduper::new();
    let startup_advisories =
        advisory_deduper.retain_new(startup_runtime_advisories(&config, microbatch_explicitness));
    for advisory in &startup_advisories {
        println!("{}", format_advisory_line(advisory));
    }
    if !startup_advisories.is_empty() {
        append_advisory_event_to_writer(
            &mut step_log,
            &AdvisoryEvent::startup(&startup_advisories),
        )?;
    }
    let cached_validation_samples = if config.bc_shards_manifest_path.is_some() {
        None
    } else {
        materialize_validation_samples(
            &config,
            &TrainValidationLoader {
                config: &loader_config,
            },
            &manifest,
        )?
    };
    let mut model = Some(model);

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
                use_amp,
                total_steps,
                current_runtime,
                run_start: &run_start,
                head_controller: &mut head_controller,
                cached_validation_samples: cached_validation_samples.as_deref(),
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
        )?;
        if outcome.stop_after_epoch {
            break;
        }
    }

    if std::env::var_os("HYDRA_BENCHMARK_QUIET").is_none() {
        println!(
            "{}",
            timestamped(format!(
                "{} {}",
                "Finished BC training. Best validation policy CE:"
                    .bold()
                    .cyan(),
                format_best_validation_summary(best_validation.as_ref())
                    .bold()
                    .green()
            ))
        );
    }

    Ok(())
}

/// Runs default BC/RL training mode.
pub fn handle_training_mode(config_path: &Path, config: TrainConfig) -> Result<(), String> {
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
        let runtime: RlTrainingRuntime = runtime;
        return run_rl_training_loop(bootstrap, runtime);
    }
    let (bootstrap, runtime, readers) = initialize_training_bootstrap(config_path, config)?;
    run_bc_training_mode_for_backend::<TrainBackend>(bootstrap, runtime, readers)
}

/// Runs Delta-Q promotion mode for the default train backend.
pub fn handle_delta_q_promotion_mode(
    config_path: &Path,
    config: TrainConfig,
    baseline_checkpoint: Option<PathBuf>,
) -> Result<(), String> {
    run_delta_q_promotion_mode::<TrainBackend>(config_path, config, baseline_checkpoint)
}

/// Formats the final best-validation summary.
pub fn format_best_validation_summary(best_validation: Option<&BestValidation>) -> String {
    if let Some(best_validation) = best_validation {
        format!(
            "{:.4} (agree {:.2}%)",
            best_validation.policy_loss,
            best_validation.agreement * 100.0
        )
    } else {
        "n/a".to_string()
    }
}

/// Dispatches the parsed train CLI into the selected execution mode.
///
/// The order preserves the previous train binary behavior:
/// preflight, Delta-Q promotion, probe-only, then default training. Probe-only
/// request defaults are resolved against the already-loaded config here so the
/// binary no longer owns mode selection semantics.
pub fn run_train_modes(cli: TrainCli, config: TrainConfig) -> Result<(), String> {
    if cli.preflight {
        return handle_preflight_mode(&cli.config_path, &config);
    }
    if cli.delta_q_promotion {
        return handle_delta_q_promotion_mode(
            &cli.config_path,
            config,
            cli.delta_q_baseline_checkpoint,
        );
    }
    if let Some(request) = probe_request_from_cli(&config, cli.probe_only)? {
        return handle_probe_mode(&cli.config_path, &config, request);
    }
    handle_training_mode(&cli.config_path, config)
}

#[cfg(test)]
mod tests;
