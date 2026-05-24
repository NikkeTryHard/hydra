//! Train binary mode dispatch facade.
//!
//! This module owns the CLI mode selection order and default train-mode bodies
//! without depending on the compatibility `hydra-train` crate.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::backend::libtorch::{LibTorchDevice, TchTensor};
use burn::tensor::backend::{AutodiffBackend, Backend};
use colored::Colorize;
use hydra_model::model::HydraModelConfig;
use hydra_train_runtime::config::{
    BcBackend, BenchmarkBaselineCliOptions, BenchmarkBaselineSource, ExperimentalTrainBackend,
    PrecisionMode, PreflightCliOptions, PythonLearnerCliOptions, PythonLearnerInput, TrainCli,
    TrainConfig, display_num_threads,
};
use hydra_train_runtime::config_runtime::validate_preflight_config;

use crate::config_runtime::{configure_threads, device_label, validate_config};
use hydra_bc_shards::BcShardSplitMode;
use hydra_replay_loader::mjai_loader::SidecarProvenance;
use hydra_train_runtime::preflight::{ProbeKind, ProbeResult};
use hydra_train_runtime::probe_request::{ProbeRequest, probe_request_from_cli};
use hydra_train_runtime::timing_metrics::{
    TimingMetricsOptions, extract_timing_metrics_from_paths,
};
use hydra_train_types::config::BCTrainerConfig;

use crate::advisory::{AdvisoryDeduper, AdvisoryEvent, startup_runtime_advisories};
use crate::artifacts::{BcArtifactPaths, append_advisory_event_to_writer};
use crate::bc_shard_builder::{BuildBcShardsConfig, build_bc_shards};
use crate::bootstrap::{
    RlTrainingBootstrap, RlTrainingRuntime, TrainBackend, TrainingBootstrap, TrainingReaders,
    TrainingRuntime, initialize_rl_training_bootstrap, initialize_training_bootstrap,
};
use crate::data_pipeline::TrainValidationLoader;
use crate::delta_q_promotion::handle_delta_q_promotion_mode as run_delta_q_promotion_mode;
use crate::epoch_runner::{EpochRunnerContext, EpochRuntimeMut, run_epoch};
use crate::preflight_runtime::{
    run_preflight_bench, run_probe_ladder_only, run_python_preflight_bench,
};
use crate::presentation::{
    BcHyperparamSummaryInput, bc_hyperparam_summary, format_advisory_line,
    format_preflight_bench_markdown_table, format_preflight_selection_line,
    format_probe_results_table, format_status_line, format_timed_phase_message,
    format_train_timing_markdown_table, print_banner_field, print_header_block, timestamped,
};
use crate::probe_summary::{best_probe_summary, format_probe_selection_summary, probe_kind_name};
use crate::resume::BestValidation;
use crate::rl_runner::run_rl_training_loop;
use crate::validation_runner::materialize_validation_samples;

fn benchmark_train_config(
    options: &BenchmarkBaselineCliOptions,
    data_dir: PathBuf,
    output_dir: PathBuf,
    shard_manifest_path: Option<PathBuf>,
) -> TrainConfig {
    let shard_input = shard_manifest_path.is_some();
    TrainConfig {
        data_dir,
        output_dir,
        num_epochs: 1,
        batch_size: options.batch_size,
        microbatch_size: Some(options.microbatch_size),
        validation_microbatch_size: Some(options.validation_microbatch_size),
        bc_shards_manifest_path: shard_manifest_path,
        shard_prefetch_depth: Some(2),
        train_fraction: options.train_fraction,
        augment: true,
        seed: 42,
        device: options.device.clone(),
        precision_mode: PrecisionMode::Bf16Autocast,
        buffer_games: 512,
        buffer_samples: 8192,
        num_threads: Some(if shard_input {
            1
        } else {
            options.train_threads
        }),
        tensorboard: false,
        archive_queue_bound: if shard_input { 8 } else { options.queue_bound },
        validation_every_n_epochs: 999,
        max_skip_logs_per_source: 4,
        log_every_n_steps: 5,
        validate_every_n_steps: 1_000_000,
        checkpoint_every_n_steps: 1_000_000,
        max_train_steps: Some(options.max_train_steps),
        max_validation_samples: Some(1),
        experimental_backbone_profile: options.experimental_backbone_profile.clone(),
        ..TrainConfig::default_preflight_bench()
    }
}

fn write_benchmark_config(path: &Path, config: &TrainConfig) -> Result<(), String> {
    let yaml = serde_yaml::to_string(config).map_err(|err| {
        format!(
            "failed to serialize benchmark config {}: {err}",
            path.display()
        )
    })?;
    fs::write(path, yaml)
        .map_err(|err| format!("failed to write benchmark config {}: {err}", path.display()))
}

struct BenchmarkTrainResult {
    label: &'static str,
    step_log_path: PathBuf,
}

struct BenchmarkRunSummary {
    build_report: Option<String>,
    train_reports: Vec<BenchmarkTrainResult>,
    elapsed_seconds: f64,
}

fn format_optional_f64(value: Option<f64>, decimals: usize) -> String {
    value
        .map(|value| format!("{value:.decimals$}"))
        .unwrap_or_else(|| "--".to_string())
}

fn format_build_report(build: &crate::bc_shard_builder::BcShardBuildOutput) -> Option<String> {
    let report = build.report.as_ref()?;
    Some(format!(
        "| elapsed s | loaded | skipped | samples | samples/s | games/s | input MiB/s | output MiB/s | bytes/sample | manifest |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n| {:.2} | {} | {} | {} | {:.2} | {} | {} | {:.2} | {} | {} |",
        report.elapsed_seconds,
        report.build.loaded_games,
        report.build.skipped_games,
        report.build.total_samples,
        report.rates.samples_per_second,
        format_optional_f64(report.rates.games_per_second, 2),
        format_optional_f64(report.rates.input_mib_per_second, 2),
        report.rates.output_mib_per_second,
        format_optional_f64(report.output.bytes_per_sample, 1),
        build.manifest_path.display(),
    ))
}

struct EnvVarGuard {
    key: &'static str,
    old: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let old = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, old }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(value) = self.old.as_ref() {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
}

fn run_quiet_training(config_path: &Path, config: TrainConfig) -> Result<(), String> {
    let _quiet = EnvVarGuard::set("HYDRA_BENCHMARK_QUIET", "1");
    handle_training_mode(config_path, config)
}

#[cfg(feature = "burn-cuda")]
fn run_burn_cuda_probe(
    config: &TrainConfig,
    options: &BenchmarkBaselineCliOptions,
    manifest_path: &Path,
) -> Result<BurnCudaProbeReport, String> {
    use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
    use burn::prelude::ElementConversion;

    use crate::bootstrap::BurnCudaTrainBackend;
    use crate::config_runtime::{
        burn_cuda_device, trainer_config_from_train_config, validate_burn_cuda_headers,
    };
    use crate::epoch_runner::materialize_host_batch_owned;
    use crate::losses::HydraLoss;
    use crate::model::{HydraModel, HydraModelInit, HydraTrainModelExt};
    use hydra_bc_shards::{BcShardSplit, load_bc_shard_reader};
    use hydra_train_runtime::loss_policy::{build_bc_exit_config, build_loss_config};
    use hydra_train_runtime::schedule::{TrainerScheduleConfig, effective_lr};

    if !matches!(config.precision_mode, PrecisionMode::Fp32) {
        return Err("Burn CUDA probe is FP32-only; set precision_mode: fp32".to_string());
    }
    if !config.device.trim().eq_ignore_ascii_case("cuda")
        && !config.device.trim().starts_with("cuda:")
    {
        return Err("Burn CUDA probe requires device cuda or cuda:<index>".to_string());
    }
    validate_burn_cuda_headers()?;

    let device = burn_cuda_device(&config.device)?;
    let reader = load_bc_shard_reader(manifest_path, BcShardSplit::Train)?;
    if reader.sample_count() < config.batch_size {
        return Err(format!(
            "Burn CUDA probe needs at least one full batch: have {} need {}",
            reader.sample_count(),
            config.batch_size
        ));
    }

    let train_cfg = trainer_config_from_train_config(config);
    let mut model = Some(train_cfg.model_config.init::<BurnCudaTrainBackend>(&device));
    let mut optimizer = train_cfg.optimizer_config().init();
    let loss_fn =
        HydraLoss::<BurnCudaTrainBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let exit_cfg = build_bc_exit_config(config.advanced_loss.as_ref());
    let mut scratch = reader.new_scratch(config.batch_size);

    let steps = options.max_train_steps.max(1);
    let mut losses = Vec::with_capacity(steps);
    let started = Instant::now();
    for step in 0..steps {
        let start = (step * config.batch_size) % (reader.sample_count() - config.batch_size + 1);
        reader.collate_host_batch_range_into(
            start,
            config.batch_size,
            config.augment,
            &mut scratch,
        )?;
        let host_batch = scratch.take_batch();
        let shard_batch = materialize_host_batch_owned::<BurnCudaTrainBackend>(host_batch, &device);
        let schedule = TrainerScheduleConfig::new(
            train_cfg.lr,
            train_cfg.min_learning_rate,
            train_cfg.warmup_steps,
        );
        let lr = effective_lr(schedule, step, steps);
        let batch_size = shard_batch.batch.actions.dims()[0];
        let microbatch_size = options.microbatch_size.min(batch_size).max(1);
        let logical_batch_len = batch_size.max(1) as f32;
        let mut accumulator: GradientsAccumulator<HydraModel<BurnCudaTrainBackend>> =
            GradientsAccumulator::new();
        let mut step_loss = 0.0f64;
        for chunk_start in (0..batch_size).step_by(microbatch_size) {
            let chunk_end = (chunk_start + microbatch_size).min(batch_size);
            #[allow(
                clippy::single_range_in_vec_init,
                reason = "Burn slice API expects a one-element range slice"
            )]
            let range = [chunk_start..chunk_end];
            let obs_chunk = shard_batch.obs.clone().slice(range.clone());
            let batch_chunk = crate::data::sample::MjaiBcBatch {
                actions: shard_batch.batch.actions.clone().slice(range.clone()),
                exit_target: shard_batch
                    .batch
                    .exit_target
                    .as_ref()
                    .map(|tensor| tensor.clone().slice(range.clone())),
                exit_mask: shard_batch
                    .batch
                    .exit_mask
                    .as_ref()
                    .map(|tensor| tensor.clone().slice(range.clone())),
            };
            let targets_chunk = shard_batch.targets.slice_batch(chunk_start, chunk_end);
            let model_ref = model
                .as_ref()
                .ok_or_else(|| "Burn CUDA probe model slot should stay populated".to_string())?;
            let output = model_ref.forward_train_with_warmup_train(obs_chunk, &loss_fn.config, &[]);
            let loss = loss_fn.bc_train_loss(
                &output,
                &targets_chunk,
                batch_chunk.exit_target.as_ref(),
                batch_chunk.exit_mask.as_ref(),
                &exit_cfg,
            );
            let chunk_weight = (chunk_end - chunk_start) as f32 / logical_batch_len;
            let total = loss.total;
            let weighted_total = total.clone() * chunk_weight;
            let chunk_loss = total
                .try_into_scalar()
                .map_err(|err| format!("Burn CUDA probe failed during loss readback: {err}"))?
                .elem::<f64>();
            if !chunk_loss.is_finite() {
                return Err(format!(
                    "Burn CUDA probe produced non-finite loss at step {step}, chunk {chunk_start}..{chunk_end}: {chunk_loss}"
                ));
            }
            step_loss += chunk_loss * f64::from(chunk_weight);
            let grads = weighted_total.backward();
            let grads = GradientsParams::from_grads(grads, model_ref);
            accumulator.accumulate(model_ref, grads);
        }
        let current_model = model
            .take()
            .ok_or_else(|| "Burn CUDA probe model slot should stay populated".to_string())?;
        model = Some(optimizer.step(lr, current_model, accumulator.grads()));
        if !step_loss.is_finite() {
            return Err(format!(
                "Burn CUDA probe produced non-finite weighted loss at step {step}: {step_loss}"
            ));
        }
        losses.push(step_loss);
    }

    Ok(BurnCudaProbeReport {
        steps,
        samples: steps * config.batch_size,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        first_loss: losses[0],
        last_loss: *losses.last().unwrap_or(&losses[0]),
    })
}

#[cfg(feature = "burn-cuda")]
struct BurnCudaProbeReport {
    steps: usize,
    samples: usize,
    elapsed_seconds: f64,
    first_loss: f64,
    last_loss: f64,
}

#[cfg(feature = "burn-cuda")]
fn format_burn_cuda_probe_report(report: &BurnCudaProbeReport) -> String {
    let samples_per_second = report.samples as f64 / report.elapsed_seconds.max(f64::EPSILON);
    format!(
        "| backend | steps | samples | samples/s | first loss | last loss |\n|---|---:|---:|---:|---:|---:|\n| burn-cuda fp32 | {} | {} | {:.2} | {:.6} | {:.6} |",
        report.steps, report.samples, samples_per_second, report.first_loss, report.last_loss,
    )
}

/// Runs configurable no-config benchmark and prints only final Markdown tables.
pub fn handle_benchmark_baseline_mode(options: BenchmarkBaselineCliOptions) -> Result<(), String> {
    let started = Instant::now();
    fs::remove_dir_all(&options.output_dir).ok();
    fs::create_dir_all(&options.output_dir).map_err(|err| {
        format!(
            "failed to create benchmark output dir {}: {err}",
            options.output_dir.display()
        )
    })?;

    let mut summary = BenchmarkRunSummary {
        build_report: None,
        train_reports: Vec::new(),
        elapsed_seconds: 0.0,
    };

    let shard_manifest = match options.source {
        BenchmarkBaselineSource::Mjai | BenchmarkBaselineSource::Both => {
            let data_dir = options.data_dir.clone().ok_or_else(|| {
                "benchmark source mjai/both requires data_dir in parsed options".to_string()
            })?;
            let build = build_bc_shards(&BuildBcShardsConfig {
                input: data_dir,
                output_dir: options.output_dir.join("bc_shards"),
                manifest_name: "bc_shards_manifest.json".to_string(),
                train_fraction: options.train_fraction,
                shard_samples: options.shard_samples,
                split_mode: BcShardSplitMode::Both,
                max_games: Some(options.max_games),
                num_threads: Some(options.num_threads),
                queue_bound: options.queue_bound,
                report_name: Some("report.json".to_string()),
                exit_provenance: SidecarProvenance::default(),
                delta_q_provenance: SidecarProvenance::default(),
                ..BuildBcShardsConfig::default()
            })
            .map_err(|err| format!("benchmark shard build failed: {err}"))?;
            summary.build_report = format_build_report(&build);
            Some(build.manifest_path)
        }
        BenchmarkBaselineSource::BcShards => options.bc_shards_manifest_path.clone(),
    };

    if matches!(
        options.experimental_backend,
        ExperimentalTrainBackend::BurnCuda
    ) {
        #[cfg(feature = "burn-cuda")]
        {
            let manifest_path = shard_manifest.ok_or_else(|| {
                "Burn CUDA probe requires --bench-source bc-shards or both with a shard manifest".to_string()
            })?;
            let data_dir = options
                .data_dir
                .clone()
                .unwrap_or_else(|| options.output_dir.join("bc_shards_input_placeholder"));
            let output_dir = options.output_dir.join("burn_cuda_probe");
            let mut probe_config =
                benchmark_train_config(&options, data_dir, output_dir, Some(manifest_path.clone()));
            probe_config.precision_mode = PrecisionMode::Fp32;
            let report = run_burn_cuda_probe(&probe_config, &options, &manifest_path)?;
            summary.elapsed_seconds = started.elapsed().as_secs_f64();
            println!("# Hydra benchmark results\n");
            if let Some(build_report) = summary.build_report.as_ref() {
                println!("## Shard build\n");
                println!("{build_report}\n");
            }
            println!("## Burn CUDA FP32 BC shard probe\n");
            println!("{}\n", format_burn_cuda_probe_report(&report));
            println!(
                "## Wall clock\n\n| elapsed s | output dir |\n|---:|---|\n| {:.2} | {} |",
                summary.elapsed_seconds,
                options.output_dir.display()
            );
            return Ok(());
        }
        #[cfg(not(feature = "burn-cuda"))]
        {
            return Err(
                "--experimental-backend burn-cuda requires hydra-train feature burn-cuda-probe"
                    .to_string(),
            );
        }
    }

    if matches!(
        options.source,
        BenchmarkBaselineSource::Mjai | BenchmarkBaselineSource::Both
    ) {
        let data_dir = options.data_dir.clone().ok_or_else(|| {
            "benchmark source mjai/both requires data_dir in parsed options".to_string()
        })?;
        let raw_output_dir = options.output_dir.join("raw_gpu");
        let raw_config = benchmark_train_config(&options, data_dir, raw_output_dir.clone(), None);
        let raw_config_path = options.output_dir.join("raw_gpu.yaml");
        write_benchmark_config(&raw_config_path, &raw_config)?;
        run_quiet_training(&raw_config_path, raw_config)?;
        summary.train_reports.push(BenchmarkTrainResult {
            label: "raw MJAI train",
            step_log_path: BcArtifactPaths::new(&raw_output_dir, 0).step_log_path,
        });
    }

    if matches!(
        options.source,
        BenchmarkBaselineSource::BcShards | BenchmarkBaselineSource::Both
    ) {
        let manifest_path = shard_manifest.ok_or_else(|| {
            "benchmark source bc-shards/both requires a shard manifest".to_string()
        })?;
        let data_dir = options
            .data_dir
            .clone()
            .unwrap_or_else(|| options.output_dir.join("bc_shards_input_placeholder"));
        let shard_output_dir = options.output_dir.join("shard_gpu");
        let shard_config = benchmark_train_config(
            &options,
            data_dir,
            shard_output_dir.clone(),
            Some(manifest_path),
        );
        let shard_config_path = options.output_dir.join("shard_gpu.yaml");
        write_benchmark_config(&shard_config_path, &shard_config)?;
        run_quiet_training(&shard_config_path, shard_config)?;
        summary.train_reports.push(BenchmarkTrainResult {
            label: "BC shard train",
            step_log_path: BcArtifactPaths::new(&shard_output_dir, 0).step_log_path,
        });
    }

    summary.elapsed_seconds = started.elapsed().as_secs_f64();

    println!("# Hydra benchmark results\n");
    if let Some(build_report) = summary.build_report.as_ref() {
        println!("## Shard build\n");
        println!("{build_report}\n");
    }
    for train_report in &summary.train_reports {
        let report = extract_timing_metrics_from_paths(
            &[train_report.step_log_path.clone()],
            &[],
            &TimingMetricsOptions {
                run_id: None,
                skip_initial_rows: 1,
                min_global_step: None,
            },
        )?;
        println!("## {}\n", train_report.label);
        println!(
            "{}\n",
            format_train_timing_markdown_table(train_report.label, &report)
        );
    }
    println!(
        "## Wall clock\n\n| elapsed s | output dir |\n|---:|---|\n| {:.2} | {} |",
        summary.elapsed_seconds,
        options.output_dir.display()
    );
    Ok(())
}

fn python_preflight_options(
    preflight: &PreflightCliOptions,
) -> Result<PythonLearnerCliOptions, String> {
    let bc_shards_manifest = preflight
        .bc_shards_manifest_path
        .clone()
        .ok_or_else(|| "Python BC preflight requires --bc-shards-manifest <path>".to_string())?;
    Ok(PythonLearnerCliOptions {
        bc_shards_manifest: bc_shards_manifest.clone(),
        input: PythonLearnerInput::BcShards {
            manifest: bc_shards_manifest,
        },
        output_dir: preflight.output_dir.clone(),
        device: preflight.device.clone(),
        batch_size: 2048,
        microbatch_size: 1024,
        variant: preflight.python_variant,
        residual_profile: hydra_train_runtime::config::PythonResidualProfileConfig::default(),
        hidden: 256,
        blocks: 10,
        bottleneck: 64,
        warmup_steps: preflight.preflight_config.warmup_steps,
        steps: Some(preflight.preflight_config.measure_steps),
        full_epoch: false,
        validation_steps: 0,
        validation_max_samples: None,
        validation_every: 0,
        raw_mjai_validation_augment: false,
        validation_source_mode: "fixed".to_string(),
        checkpoint_out: None,
        resume: None,
        checkpoint_every_steps: 0,
        log_every_steps: hydra_train_runtime::config::default_log_every_n_steps(),
        keep_step_checkpoints: false,
        tensorboard: false,
        launch_tensorboard: false,
        tensorboard_host: hydra_train_runtime::config::default_tensorboard_host(),
        tensorboard_port: hydra_train_runtime::config::default_tensorboard_port(),
        background: false,
        learning_rate: hydra_train_runtime::config::default_bc_learning_rate(),
        min_learning_rate: hydra_train_runtime::config::default_bc_min_learning_rate(),
        lr_warmup_steps: hydra_train_runtime::config::default_bc_warmup_steps(),
        lr_schedule: "cosine".to_string(),
        schedule_total_steps: Some(preflight.preflight_config.measure_steps),
        schedule_target_games: None,
        grad_clip_norm: f64::from(hydra_train_runtime::config::default_bc_grad_clip_norm()),
        weight_decay: f64::from(hydra_train_runtime::config::default_bc_weight_decay()),
        ema_enabled: false,
        ema_decay: hydra_train_runtime::config::default_ema_decay(),
        ema_start_step: 0,
        ema_update_every_steps: hydra_train_runtime::config::default_ema_update_every_steps(),
        ema_device: hydra_train_runtime::config::EmaDeviceConfig::Auto,
        compile_fullgraph_check: true,
        oracle_critic_weight: 0.0,
        safety_residual_weight: 0.0,
    })
}

fn print_python_preflight_recommendation(
    report: &hydra_train_runtime::preflight::PreflightBenchReport,
) {
    let best = report
        .rows
        .iter()
        .filter(|row| row.status == hydra_train_runtime::preflight::PreflightBenchStatus::Pass)
        .filter_map(|row| row.samples_per_second.map(|rate| (row.batch_size, rate)))
        .max_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    if let Some((batch, samples_per_second)) = best {
        println!(
            "\nPython BC preflight best: batch={} microbatch={} samples/s={:.2}",
            batch, batch, samples_per_second
        );
    }
}

/// Runs standalone synthetic benchmark preflight from explicit CLI arguments.
pub fn handle_preflight_mode(preflight: PreflightCliOptions) -> Result<(), String> {
    let preflight_wall_start = Instant::now();
    let preflight_config = &preflight.preflight_config;
    validate_preflight_config(preflight_config)?;
    let config = TrainConfig {
        bc_shards_manifest_path: None,
        bc_backend: Default::default(),
        output_dir: preflight.output_dir.clone(),
        device: preflight.device.clone(),
        ..TrainConfig::default_preflight_bench()
    };
    configure_threads(config.num_threads)?;
    let artifacts = BcArtifactPaths::new(&config.output_dir, 0);
    artifacts.create_root_dir()?;
    let device_name = device_label(&config.device);
    print_preflight_banner("Hydra preflight", &config, &device_name);
    let preflight = if preflight.bc_backend == BcBackend::Python {
        let base = python_preflight_options(&preflight)?;
        run_python_preflight_bench(preflight_config, &base, &device_name)?
    } else {
        run_preflight_bench(&config, preflight_config, &device_name)?
    };
    println!(
        "{}",
        format_preflight_bench_markdown_table(&preflight.report)
    );
    if preflight
        .report
        .rows
        .iter()
        .any(|row| row.mode == hydra_train_runtime::preflight::PreflightBenchMode::PythonBc)
    {
        print_python_preflight_recommendation(&preflight.report);
    }
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
    let default_preflight = hydra_train_runtime::preflight::PreflightConfig::default();
    let (selected, results) =
        run_probe_ladder_only(config_path, config, &artifacts, &default_preflight, request)?;
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
    let cuda_device = {
        let device = config.device.trim().to_ascii_lowercase();
        device == "cuda" || device.starts_with("cuda:")
    };
    let pinned_staging = cfg!(feature = "cuda-graph") && cuda_device;
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

    if std::env::var_os("HYDRA_BENCHMARK_QUIET").is_none() {
        print_bc_training_banner(
            &model_config,
            &config,
            &artifacts,
            &device_name,
            &banner_stats,
            bc_hyperparam_summary_input(&train_cfg),
        );
        resume.print_banner_with_effective_runtime(Some(current_runtime));
    }
    let mut advisory_deduper = AdvisoryDeduper::new();
    let startup_advisories =
        advisory_deduper.retain_new(startup_runtime_advisories(&config, microbatch_explicitness));
    if std::env::var_os("HYDRA_BENCHMARK_QUIET").is_none() {
        for advisory in &startup_advisories {
            println!("{}", format_advisory_line(advisory));
        }
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

/// Formats train device labels for `--list-devices` stdout.
pub fn format_list_devices_stdout() -> &'static str {
    "Hydra train device labels:\n  cpu        supported; always available\n  cuda       supported syntax; equivalent to cuda:0; requires CUDA-capable LibTorch at runtime\n  cuda:<N>   supported syntax; N is zero-based CUDA device index; availability checked when training opens device\n\nHYDRA_TRAIN_DEVICE overrides YAML device with one of: cpu, cuda, cuda:<N>\n"
}

/// Prints train device labels for `--list-devices`.
pub fn handle_list_devices_mode() -> Result<(), String> {
    print!("{}", format_list_devices_stdout());
    Ok(())
}

/// Dispatches the parsed train CLI into the selected execution mode.
///
/// The order preserves the previous train binary behavior:
/// preflight, benchmark, Delta-Q promotion, probe-only, then default training. Probe-only
/// request defaults are resolved against the already-loaded config here so the
/// binary no longer owns mode selection semantics.
pub fn run_train_modes(cli: TrainCli, config: TrainConfig) -> Result<(), String> {
    if cli.list_devices {
        return handle_list_devices_mode();
    }
    if let Some(preflight) = cli.preflight {
        return handle_preflight_mode(preflight);
    }
    if let Some(benchmark) = cli.benchmark_baseline {
        return handle_benchmark_baseline_mode(benchmark);
    }
    let config_path = cli.config_path.as_deref().ok_or_else(|| {
        "config path is required unless --list-devices or --preflight is used".to_string()
    })?;
    if cli.delta_q_promotion {
        return handle_delta_q_promotion_mode(config_path, config, cli.delta_q_baseline_checkpoint);
    }
    if let Some(request) = probe_request_from_cli(&config, cli.probe_only)? {
        return handle_probe_mode(config_path, &config, request);
    }
    handle_training_mode(config_path, config)
}

#[cfg(test)]
mod tests;
