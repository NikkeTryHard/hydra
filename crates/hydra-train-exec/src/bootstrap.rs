use std::fs;
use std::path::Path;
use std::time::Instant;

use burn::backend::libtorch::LibTorchDevice;
use burn::optim::Adam;
use burn::optim::Optimizer;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::prelude::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend};
use colored::Colorize;
use tboard::EventWriter;

use crate::bc_runtime::BcExitConfig;
use crate::bc_shard_adapter::{
    BcShardManifest, BcShardManifestConfigRef, BcShardSplit as ManifestBcShardSplit,
    data_manifest_from_bc_shard_manifest, read_bc_shard_manifest,
    validate_bc_shard_manifest_for_config as validate_bc_shard_manifest_for_config_ref,
};
use crate::data_pipeline::{DataManifest, ReplayTargetProfile, StreamingLoaderConfig};
use crate::losses::HydraLoss;
use hydra_model::model::{HydraModel, HydraModelConfig, HydraModelInit};
use hydra_replay_sidecar::{
    DeltaQSidecarIndex, ExitSidecarIndex, source_net_hash_from_checkpoint_identity,
};
use hydra_train_algo::gae::GaeConfig;
use hydra_train_runtime::config::{RlTrainConfig, TrainConfig};
use hydra_train_runtime::config_runtime::train_microbatch_size;
use hydra_train_runtime::loss_policy::{
    build_bc_exit_config, build_loss_config, build_rl_loss_config,
};

use crate::config_runtime::{
    configure_threads, device_label, rl_config_from_train_config, train_device,
    trainer_config_from_train_config, validate_config,
};
use hydra_train_runtime::head_gates::{HeadActivationConfig, HeadActivationController};
use hydra_train_runtime::progress::BannerStats;
use hydra_train_runtime::schedule::schedule_total_steps;
use hydra_train_types::config::BCTrainerConfig;
use hydra_train_types::config::RlConfig;
use hydra_train_types::phase::PipelineState;

use crate::advisory::MicrobatchExplicitness;
use crate::artifacts::{
    BcArtifactPaths, JsonlAppender, RlArtifactPaths, RlPreflightPaths, load_or_scan_manifest_cache,
    open_rl_step_log_appender, open_step_log_appender, open_training_log_appender,
};
use crate::presentation::timestamped;
use crate::resume::{
    ResumeContext, RlResumeContext, RlRuntimeResumeContract, RuntimeResumeContract,
    rl_runtime_resume_contract, runtime_resume_contract, validate_resume_runtime_compatibility,
    validate_rl_resume_runtime_compatibility,
};

/// LibTorch autodiff backend used by the train binary.
pub type TrainBackend = burn::backend::Autodiff<burn::backend::LibTorch>;

type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

fn validate_bc_shard_manifest_for_config(
    manifest: &BcShardManifest,
    config: &TrainConfig,
) -> Result<(), String> {
    let advanced_loss = config.advanced_loss.as_ref();
    validate_bc_shard_manifest_for_config_ref(
        manifest,
        BcShardManifestConfigRef {
            train_fraction: config.train_fraction,
            source_filters: &config.source_filters,
            exit_sidecar_path: config.exit_sidecar_path.as_deref(),
            delta_q_sidecar_path: config.delta_q_sidecar_path.as_deref(),
            exit_loss_weight: advanced_loss.and_then(|loss| loss.exit),
            delta_q_loss_weight: advanced_loss.and_then(|loss| loss.delta_q),
        },
    )
}

fn apply_cached_bc_runtime_if_matching(
    config: &mut TrainConfig,
    resume: &ResumeContext,
    artifacts: &BcArtifactPaths,
    model_config: &HydraModelConfig,
) -> Result<(), String> {
    crate::preflight_runtime::apply_cached_bc_runtime_if_matching(
        config,
        resume,
        artifacts,
        model_config,
    )
}

/// Prepared BC training bootstrap state owned by the execution crate.
pub struct TrainingBootstrap<B>
where
    B: AutodiffBackend,
{
    /// Effective training configuration after cache/runtime overrides.
    pub config: TrainConfig,
    /// Resume context selected for this session.
    pub resume: ResumeContext,
    /// Output artifact paths.
    pub artifacts: BcArtifactPaths,
    /// Streaming loader configuration.
    pub loader_config: StreamingLoaderConfig,
    /// Data manifest selected by shard manifest or source scan.
    pub manifest: DataManifest,
    /// BC trainer configuration.
    pub train_cfg: BCTrainerConfig,
    /// Model architecture configuration.
    pub model_config: HydraModelConfig,
    /// Human-readable device label.
    pub device_name: String,
    /// LibTorch device for training tensors/checkpoints.
    pub train_device: LibTorchDevice,
    /// Resume/runtime compatibility contract.
    pub current_runtime: RuntimeResumeContract,
    /// Whether microbatch was explicit or inferred.
    pub microbatch_explicitness: MicrobatchExplicitness,
    /// Global optimizer step at session start.
    pub session_start_global_step: usize,
    /// Scheduled total optimizer steps.
    pub total_steps: usize,
    /// Physical microbatch size.
    pub microbatch_size: usize,
    /// Whether AMP is enabled.
    pub use_amp: bool,
    /// Banner statistics for presentation.
    pub banner_stats: BannerStats,
    /// Autodiff training loss.
    pub loss_fn: HydraLoss<B>,
    /// Inner-backend validation loss.
    pub valid_loss_fn: HydraLoss<ValidBackendOf<B>>,
    /// Optional ExIt loss configuration.
    pub bc_exit_cfg: BcExitConfig,
}

/// Mutable BC runtime state created by bootstrap.
pub struct TrainingRuntime<B>
where
    B: AutodiffBackend,
{
    /// Model being trained.
    pub model: HydraModel<B>,
    /// Optimizer, restored when resume state requires it.
    pub optimizer: OptimizerAdaptor<Adam, HydraModel<B>, B>,
    /// Best validation observed from resume/latest state.
    pub best_validation: Option<crate::resume::BestValidation>,
    /// Global optimizer step.
    pub global_step: usize,
    /// Session start timestamp.
    pub run_start: Instant,
    /// Last logged optimizer step.
    pub last_log_step: usize,
    /// Last logged wall-clock timestamp.
    pub last_log_time: Instant,
    /// Optional TensorBoard event writer.
    pub tb: Option<EventWriter<std::fs::File>>,
    /// Epoch training log appender.
    pub training_log: JsonlAppender,
    /// Step training log appender.
    pub step_log: JsonlAppender,
    /// Advanced-head activation controller.
    pub head_controller: HeadActivationController,
}

/// Reader marker reserved for train-mode callsites.
pub struct TrainingReaders;

/// Prepared RL training bootstrap state owned by the execution crate.
pub struct RlTrainingBootstrap {
    /// Effective root training configuration.
    pub config: TrainConfig,
    /// Effective RL configuration after cache overrides.
    pub rl_config: RlTrainConfig,
    /// RL resume context selected for this session.
    pub resume: RlResumeContext,
    /// Output artifact paths.
    pub artifacts: RlArtifactPaths,
    /// Model architecture configuration.
    pub model_config: HydraModelConfig,
    /// Human-readable device label.
    pub device_name: String,
    /// LibTorch device for training tensors/checkpoints.
    pub train_device: LibTorchDevice,
    /// RL resume/runtime compatibility contract.
    pub current_runtime: RlRuntimeResumeContract,
    /// Global optimizer step at session start.
    pub session_start_global_step: usize,
    /// Scheduled total optimizer steps.
    pub total_steps: usize,
    /// Autodiff RL loss.
    pub loss_fn: HydraLoss<TrainBackend>,
    /// RL step configuration.
    pub rl_step_cfg: RlConfig,
    /// GAE configuration.
    pub gae_config: GaeConfig,
}

/// Mutable RL runtime state created by bootstrap.
pub struct RlTrainingRuntime {
    /// Model being trained.
    pub model: HydraModel<TrainBackend>,
    /// Optimizer, restored when resume state requires it.
    pub optimizer: OptimizerAdaptor<Adam, HydraModel<TrainBackend>, TrainBackend>,
    /// Global optimizer step.
    pub global_step: usize,
    /// Session start timestamp.
    pub run_start: Instant,
    /// Last logged optimizer step.
    pub last_log_step: usize,
    /// Last logged wall-clock timestamp.
    pub last_log_time: Instant,
    /// Optional TensorBoard event writer.
    pub tb: Option<EventWriter<std::fs::File>>,
    /// RL step log appender.
    pub step_log: JsonlAppender,
    /// RL pipeline phase/progress state.
    pub pipeline_state: PipelineState,
    /// Advanced-head activation controller.
    pub head_controller: HeadActivationController,
}

/// Initializes BC training bootstrap/runtime with an explicit model config.
pub fn initialize_training_bootstrap_for_backend_with_model_config<B>(
    _config_path: &Path,
    mut config: TrainConfig,
    model_config: HydraModelConfig,
) -> Result<(TrainingBootstrap<B>, TrainingRuntime<B>, TrainingReaders), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    B::InnerBackend: Backend<Device = LibTorchDevice>,
{
    validate_config(&config)?;
    let microbatch_explicitness = MicrobatchExplicitness::from_config(&config);

    let resume = ResumeContext::load(&config)?;
    let session_start_global_step = resume.session_start_global_step;
    let artifacts = BcArtifactPaths::new(&config.output_dir, session_start_global_step);
    artifacts.create_root_dir()?;
    apply_cached_bc_runtime_if_matching(&mut config, &resume, &artifacts, &model_config)?;
    configure_threads(config.num_threads)?;

    let exit_sidecar = if let Some(path) = config.exit_sidecar_path.as_ref() {
        Some(std::sync::Arc::new(
            ExitSidecarIndex::from_jsonl_path(path).map_err(|err| {
                format!(
                    "failed to load replay ExIt sidecar {}: {err}",
                    path.display()
                )
            })?,
        ))
    } else {
        None
    };
    let delta_q_sidecar = if let Some(path) = config.delta_q_sidecar_path.as_ref() {
        Some(std::sync::Arc::new(
            DeltaQSidecarIndex::from_jsonl_path(path).map_err(|err| {
                format!(
                    "failed to load replay delta_q sidecar {}: {err}",
                    path.display()
                )
            })?,
        ))
    } else {
        None
    };

    let loader_config = StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        aggregate_skip_logs: false,
        source_filters: config.source_filters.clone(),
        replay_target_profile: ReplayTargetProfile::minimal_bc(),
        exit_sidecar,
        exit_sidecar_source_net_hash: None,
        exit_sidecar_source_version: None,
        delta_q_sidecar,
        delta_q_sidecar_source_net_hash: None,
        delta_q_sidecar_source_version: None,
        num_threads: config.num_threads,
    };
    let scan_sources_len = if config.bc_shards_manifest_path.is_some() || config.data_dir.is_file()
    {
        1
    } else {
        fs::read_dir(&config.data_dir)
            .map_err(|err| {
                format!(
                    "failed to read data dir {}: {err}",
                    config.data_dir.display()
                )
            })?
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_type().ok().filter(|ft| ft.is_file()))
            .count()
    };
    let scan_pb = crate::presentation::make_bar(
        scan_sources_len as u64,
        "[scan] [{bar:40.cyan/blue}] {pos}/{len} sources {msg}",
    )?;
    scan_pb.set_message("Scanning archives...".to_string());
    let manifest = if let Some(shard_manifest_path) = config.bc_shards_manifest_path.as_ref() {
        let shard_manifest = read_bc_shard_manifest(shard_manifest_path)?;
        validate_bc_shard_manifest_for_config(&shard_manifest, &config)?;
        scan_pb.finish_with_message(format!(
            "using BC shard manifest: {} train / {} val samples",
            shard_manifest
                .splits
                .iter()
                .find(|split| split.split == ManifestBcShardSplit::Train)
                .map(|split| split.sample_count)
                .unwrap_or(0),
            shard_manifest
                .splits
                .iter()
                .find(|split| split.split == ManifestBcShardSplit::Validation)
                .map(|split| split.sample_count)
                .unwrap_or(0)
        ));
        data_manifest_from_bc_shard_manifest(&shard_manifest)
    } else {
        let preflight_paths = crate::artifacts::PreflightPaths::new(&artifacts);
        let manifest = load_or_scan_manifest_cache(
            &preflight_paths.manifest_cache_path,
            &config.data_dir,
            config.train_fraction,
            &config.source_filters,
            Some(&scan_pb),
            "MJAI data",
            |cached| {
                scan_pb.finish_with_message(format!(
                    "reused manifest: {} train / {} val games",
                    cached.manifest.train_count, cached.manifest.val_count
                ));
            },
        )?;
        if !scan_pb.is_finished() && manifest.counts_exact {
            scan_pb.finish_with_message(format!(
                "found {} train / {} val games",
                manifest.train_count, manifest.val_count
            ));
        } else {
            scan_pb.finish_with_message(format!(
                "found {} sources; exact game counts deferred to streaming load",
                manifest.sources.len()
            ));
        }
        manifest
    };

    let train_cfg = trainer_config_from_train_config(&config);
    train_cfg
        .validate()
        .map_err(|err| format!("invalid trainer config: {err}"))?;

    let device_name = device_label(&config.device);
    let current_runtime = runtime_resume_contract(&config);
    if let Some(state) = resume.state.as_ref() {
        validate_resume_runtime_compatibility(state, current_runtime)?;
    }
    let train_device = train_device(&config.device)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let mut model = model_config.init::<B>(&train_device);
    let checkpoint_identity = resume
        .checkpoint_base
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "latest_model".to_string());
    let exit_sidecar_source_net_hash =
        source_net_hash_from_checkpoint_identity(&checkpoint_identity);
    if let Some(checkpoint_base) = resume.checkpoint_base.as_ref() {
        model = model
            .load_file(checkpoint_base, &recorder, &train_device)
            .map_err(|err| {
                format!(
                    "failed to load checkpoint {}: {err}",
                    checkpoint_base.display()
                )
            })?;
    }
    let learner_params = model.num_params();
    let loader_config = StreamingLoaderConfig {
        exit_sidecar_source_net_hash: Some(exit_sidecar_source_net_hash),
        exit_sidecar_source_version: Some(1),
        delta_q_sidecar_source_net_hash: Some(exit_sidecar_source_net_hash),
        delta_q_sidecar_source_version: Some(1),
        ..loader_config
    };

    let optimizer = if resume.restores_optimizer_state() {
        let optimizer_base = resume.optimizer_base.as_ref().ok_or_else(|| {
            let checkpoint = resume
                .checkpoint_base
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            format!(
                "resume state for checkpoint {} requires optimizer sidecar, but none was found next to that checkpoint",
                checkpoint
            )
        })?;
        let optimizer_recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let optimizer_record = optimizer_recorder
            .load(optimizer_base.clone(), &train_device)
            .map_err(|err| {
                format!(
                    "failed to load optimizer state {}: {err}",
                    optimizer_base.display()
                )
            })?;
        train_cfg
            .optimizer_config()
            .init()
            .load_record(optimizer_record)
    } else {
        train_cfg.optimizer_config().init()
    };
    let loss_fn = HydraLoss::<B>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let valid_loss_fn =
        HydraLoss::<ValidBackendOf<B>>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let bc_exit_cfg = build_bc_exit_config(config.advanced_loss.as_ref());
    let total_steps = schedule_total_steps(&config, session_start_global_step);
    let microbatch_size = train_microbatch_size(&config);
    let use_amp = config.use_amp();
    let best_validation = resume.best_validation();
    let global_step = session_start_global_step;
    let run_start = Instant::now();
    let last_log_step = global_step;
    let last_log_time = run_start;
    let tb = if config.tensorboard {
        artifacts.create_tensorboard_dirs()?;
        Some(
            EventWriter::create(&artifacts.tb_session_dir)
                .map_err(|err| format!("tensorboard init: {err}"))?,
        )
    } else {
        None
    };
    let training_log = open_training_log_appender(&artifacts.training_log_path)?;
    let step_log = open_step_log_appender(&artifacts.step_log_path)?;

    let banner_stats = BannerStats {
        total_sources: manifest.sources.len(),
        total_games: manifest.total_games,
        train_count: manifest.train_count,
        val_count: manifest.val_count,
        accum_steps: current_runtime.accum_steps,
        counts_exact: manifest.counts_exact,
    };

    let readers = TrainingReaders;

    Ok((
        TrainingBootstrap {
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
        },
        TrainingRuntime {
            model,
            optimizer,
            best_validation,
            global_step,
            run_start,
            last_log_step,
            last_log_time,
            tb,
            training_log,
            step_log,
            head_controller: HeadActivationController::new(
                HeadActivationConfig::default_with_params(learner_params),
            ),
        },
        readers,
    ))
}

/// Initializes BC training bootstrap/runtime for a backend using the learner model config.
pub fn initialize_training_bootstrap_for_backend<B>(
    config_path: &Path,
    config: TrainConfig,
) -> Result<(TrainingBootstrap<B>, TrainingRuntime<B>, TrainingReaders), String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    B::InnerBackend: Backend<Device = LibTorchDevice>,
{
    initialize_training_bootstrap_for_backend_with_model_config::<B>(
        config_path,
        config,
        HydraModelConfig::learner(),
    )
}

/// Initializes BC training bootstrap/runtime for the default train backend.
pub fn initialize_training_bootstrap(
    config_path: &Path,
    config: TrainConfig,
) -> Result<
    (
        TrainingBootstrap<TrainBackend>,
        TrainingRuntime<TrainBackend>,
        TrainingReaders,
    ),
    String,
> {
    initialize_training_bootstrap_for_backend::<TrainBackend>(config_path, config)
}

/// Initializes RL training bootstrap/runtime for the default train backend.
pub fn initialize_rl_training_bootstrap(
    _config_path: &Path,
    config: TrainConfig,
    mut rl_config: RlTrainConfig,
) -> Result<(RlTrainingBootstrap, RlTrainingRuntime), String> {
    validate_config(&config)?;
    configure_threads(config.num_threads)?;

    let resume = RlResumeContext::load(&config)?;
    let session_start_global_step = resume.session_start_global_step;
    let artifacts = RlArtifactPaths::new(&config.output_dir, session_start_global_step);
    artifacts.create_root_dir()?;

    let device_name = device_label(&config.device);
    let model_config = HydraModelConfig::learner();

    if let Some(cached) =
        crate::preflight_runtime::matching_rl_preflight_cache(&config, &model_config, &artifacts)?
    {
        let tuned_games = cached.runtime.loader.buffer_games;
        let tuned_microbatch = cached.runtime.selected.train_microbatch_size;
        if tuned_games != rl_config.games_per_batch {
            println!(
                "{}",
                timestamped(format!(
                    "{} games_per_batch={} -> {} (from preflight cache)",
                    "RL preflight override:".bold().cyan(),
                    rl_config.games_per_batch,
                    tuned_games,
                ))
            );
            rl_config.games_per_batch = tuned_games;
        }
        if rl_config.microbatch_size != Some(tuned_microbatch) {
            println!(
                "{}",
                timestamped(format!(
                    "{} rl.microbatch_size={:?} -> {} (from preflight cache)",
                    "RL preflight override:".bold().cyan(),
                    rl_config.microbatch_size,
                    tuned_microbatch,
                ))
            );
            rl_config.microbatch_size = Some(tuned_microbatch);
        }
    } else if RlPreflightPaths::new(&artifacts).cache_path.exists() {
        println!(
            "{}",
            timestamped(format!(
                "{} cache fingerprint mismatch, using config games_per_batch={} rl.microbatch_size={:?}",
                "RL preflight skip:".bold().yellow(),
                rl_config.games_per_batch,
                rl_config.microbatch_size,
            ))
        );
    }

    let current_runtime = rl_runtime_resume_contract(&rl_config);
    if let Some(state) = resume.state.as_ref() {
        validate_rl_resume_runtime_compatibility(state, current_runtime)?;
    }
    let train_device = train_device(&config.device)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let mut model = model_config.init::<TrainBackend>(&train_device);
    if let Some(checkpoint_base) = resume.checkpoint_base.as_ref() {
        model = model
            .load_file(checkpoint_base, &recorder, &train_device)
            .map_err(|err| {
                format!(
                    "failed to load RL checkpoint {}: {err}",
                    checkpoint_base.display()
                )
            })?;
    }

    let optimizer = if resume.restores_optimizer_state() {
        let optimizer_base = resume.optimizer_base.as_ref().ok_or_else(|| {
            let checkpoint = resume
                .checkpoint_base
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            format!(
                "RL resume state for checkpoint {} requires optimizer sidecar, but none was found next to that checkpoint",
                checkpoint
            )
        })?;
        let optimizer_recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let optimizer_record = optimizer_recorder
            .load(optimizer_base.clone(), &train_device)
            .map_err(|err| {
                format!(
                    "failed to load RL optimizer state {}: {err}",
                    optimizer_base.display()
                )
            })?;
        BCTrainerConfig::new(model_config.clone())
            .optimizer_config()
            .init()
            .load_record(optimizer_record)
    } else {
        BCTrainerConfig::new(model_config.clone())
            .optimizer_config()
            .init()
    };

    let loss_fn =
        HydraLoss::<TrainBackend>::new(build_rl_loss_config(config.advanced_loss.as_ref())?);
    let total_steps = schedule_total_steps(&config, session_start_global_step);
    let tb = if config.tensorboard {
        artifacts.create_tensorboard_dirs()?;
        Some(
            EventWriter::create(&artifacts.tb_session_dir)
                .map_err(|err| format!("tensorboard init: {err}"))?,
        )
    } else {
        None
    };
    let step_log = open_rl_step_log_appender(&artifacts.step_log_path)?;
    let pipeline_state = resume
        .state
        .as_ref()
        .map(|state| state.pipeline_state)
        .unwrap_or(PipelineState {
            phase: rl_config.phase.to_training_phase(),
            ..PipelineState::default()
        });
    let head_controller = HeadActivationController::new(HeadActivationConfig::default_with_params(
        model_config.estimated_params(),
    ));
    let rl_step_cfg = rl_config_from_train_config(&rl_config);
    let run_start = Instant::now();

    Ok((
        RlTrainingBootstrap {
            config,
            rl_config,
            resume,
            artifacts,
            model_config,
            device_name,
            train_device,
            current_runtime,
            session_start_global_step,
            total_steps,
            loss_fn,
            rl_step_cfg,
            gae_config: GaeConfig::default(),
        },
        RlTrainingRuntime {
            model,
            optimizer,
            global_step: session_start_global_step,
            run_start,
            last_log_step: session_start_global_step,
            last_log_time: run_start,
            tb,
            step_log,
            pipeline_state,
            head_controller,
        },
    ))
}

#[cfg(test)]
mod tests;
