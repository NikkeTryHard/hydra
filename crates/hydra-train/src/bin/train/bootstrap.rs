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

use hydra_train::config::PipelineState;
use hydra_train::data::bc_shards::{
    BcShardManifest, BcShardReader, BcShardSplit, load_bc_shard_reader,
};
use hydra_train::data::pipeline::{DataManifest, DataSource, StreamingLoaderConfig};
use hydra_train::model::{HydraModel, HydraModelConfig};
#[cfg(test)]
use hydra_train::preflight::ManifestCacheEntry;
use hydra_train::selfplay::CooperativeSelfPlayCoordinator;
use hydra_train::training::bc::{BCTrainerConfig, BcExitConfig};
use hydra_train::training::gae::GaeConfig;
use hydra_train::training::head_gates::{HeadActivationConfig, HeadActivationController};
use hydra_train::training::losses::HydraLoss;
use hydra_train::training::replay_delta_q::DeltaQSidecarIndex;
use hydra_train::training::replay_exit::{
    ExitSidecarIndex, source_net_hash_from_checkpoint_identity,
};
use hydra_train::training::rl::RlConfig;

use super::TrainBackend;
use super::advisory::MicrobatchExplicitness;
#[cfg(test)]
use super::artifacts::write_manifest_cache;
use super::artifacts::{
    BcArtifactPaths, RlArtifactPaths, RlPreflightPaths, load_or_scan_manifest_cache,
    read_preflight_cache,
};
use super::config::{
    RlTrainConfig, TrainConfig, configure_threads, device_label, train_device,
    train_microbatch_size, trainer_config_from_train_config, validate_config,
};
use super::config_runtime::rl_config_from_train_config;
use super::loss_policy::{build_bc_exit_config, build_loss_config, build_rl_loss_config};
use super::preflight_fingerprint::preflight_cache_key;
use super::presentation::timestamped;
use super::progress::BannerStats;
use super::resume::{
    ResumeContext, RlResumeContext, RlRuntimeResumeContract, rl_runtime_resume_contract,
    runtime_resume_contract, validate_resume_runtime_compatibility,
    validate_rl_resume_runtime_compatibility,
};
use super::schedule::schedule_total_steps;

type JsonlAppender = super::artifacts::JsonlAppender;

type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

fn data_manifest_from_bc_shard_manifest(manifest: &BcShardManifest) -> DataManifest {
    let train_count = manifest
        .splits
        .iter()
        .find(|split| split.split == BcShardSplit::Train)
        .map(|split| split.sample_count as usize)
        .unwrap_or(0);
    let val_count = manifest
        .splits
        .iter()
        .find(|split| split.split == BcShardSplit::Validation)
        .map(|split| split.sample_count as usize)
        .unwrap_or(0);

    DataManifest {
        sources: vec![DataSource::LooseFile(
            Path::new(&manifest.input).to_path_buf(),
        )],
        total_games: manifest.source_total_games_hint,
        train_count,
        val_count,
        counts_exact: true,
    }
}

fn validate_bc_shard_manifest_for_config(
    manifest: &BcShardManifest,
    config: &TrainConfig,
) -> Result<(), String> {
    if manifest.train_fraction.to_bits() != config.train_fraction.to_bits() {
        return Err(format!(
            "BC shard manifest train_fraction {} does not match config train_fraction {}. Rebuild shards or use matching config.",
            manifest.train_fraction, config.train_fraction
        ));
    }
    if !config.source_filters.is_empty() {
        return Err(
            "BC shard manifest does not record source_filters; shard-backed BC requires empty source_filters or shards rebuilt with an explicit recorded filter contract"
                .to_string(),
        );
    }

    let advanced_loss = config.advanced_loss.as_ref();
    if advanced_loss
        .and_then(|loss| loss.exit)
        .is_some_and(|weight| weight > 0.0)
    {
        let configured = config
            .exit_sidecar_path
            .as_ref()
            .ok_or_else(|| "advanced_loss.exit requires exit_sidecar_path".to_string())?;
        let manifest_sidecar = manifest.exit_sidecar.as_ref().ok_or_else(|| {
            "advanced_loss.exit requires BC shards built with matching ExIt sidecar".to_string()
        })?;
        if manifest_sidecar.path != configured.display().to_string() {
            return Err(format!(
                "BC shard ExIt sidecar {} does not match config exit_sidecar_path {}",
                manifest_sidecar.path,
                configured.display()
            ));
        }
    }
    if advanced_loss
        .and_then(|loss| loss.delta_q)
        .is_some_and(|weight| weight > 0.0)
    {
        let configured = config
            .delta_q_sidecar_path
            .as_ref()
            .ok_or_else(|| "advanced_loss.delta_q requires delta_q_sidecar_path".to_string())?;
        let manifest_sidecar = manifest.delta_q_sidecar.as_ref().ok_or_else(|| {
            "advanced_loss.delta_q requires BC shards built with matching delta_q sidecar"
                .to_string()
        })?;
        if manifest_sidecar.path != configured.display().to_string() {
            return Err(format!(
                "BC shard delta_q sidecar {} does not match config delta_q_sidecar_path {}",
                manifest_sidecar.path,
                configured.display()
            ));
        }
    }
    Ok(())
}

fn apply_cached_bc_runtime_if_matching(
    config: &mut TrainConfig,
    resume: &ResumeContext,
    artifacts: &BcArtifactPaths,
    model_config: &HydraModelConfig,
) -> Result<(), String> {
    let is_epoch_boundary_resume = resume
        .state
        .as_ref()
        .is_some_and(|state| state.skip_optimizer_steps_in_epoch == 0);
    if !is_epoch_boundary_resume {
        return Ok(());
    }

    let preflight_paths = crate::artifacts::PreflightPaths::new(artifacts);
    let cache_key = preflight_cache_key(
        config,
        model_config,
        &config.device,
        super::config::default_num_threads_for_system(),
    );
    let Some(cached) = read_preflight_cache(&preflight_paths.cache_path)? else {
        return Ok(());
    };
    if cached.cache_key != cache_key {
        println!(
            "{}",
            timestamped(format!(
                "{} cache fingerprint mismatch, using config train_microbatch_size={:?} validation_microbatch_size={:?} buffer_games={} buffer_samples={} archive_queue_bound={} num_threads={:?}",
                "BC preflight skip:".bold().yellow(),
                config.microbatch_size,
                config.validation_microbatch_size,
                config.buffer_games,
                config.buffer_samples,
                config.archive_queue_bound,
                config.num_threads,
            ))
        );
        return Ok(());
    }
    let tuned_selected = cached.runtime.selected;
    let original_train = config.microbatch_size;
    let original_validation = config.validation_microbatch_size;
    if original_train != Some(tuned_selected.train_microbatch_size)
        || original_validation != Some(tuned_selected.validation_microbatch_size)
    {
        println!(
            "{}",
            timestamped(format!(
                "{} train_microbatch_size={:?} -> {} validation_microbatch_size={:?} -> {} accum_steps={} (epoch-boundary selected-runtime from preflight cache)",
                "BC preflight override:".bold().cyan(),
                original_train,
                tuned_selected.train_microbatch_size,
                original_validation,
                tuned_selected.validation_microbatch_size,
                tuned_selected.accum_steps,
            ))
        );
    }

    config.microbatch_size = Some(tuned_selected.train_microbatch_size);
    config.validation_microbatch_size = Some(tuned_selected.validation_microbatch_size);
    Ok(())
}

pub(super) struct TrainingBootstrap<B>
where
    B: AutodiffBackend,
{
    pub(super) config: TrainConfig,
    pub(super) resume: ResumeContext,
    pub(super) artifacts: BcArtifactPaths,
    pub(super) loader_config: StreamingLoaderConfig,
    pub(super) manifest: DataManifest,
    pub(super) train_cfg: BCTrainerConfig,
    pub(super) model_config: HydraModelConfig,
    pub(super) device_name: String,
    pub(super) train_device: LibTorchDevice,
    pub(super) current_runtime: super::resume::RuntimeResumeContract,
    pub(super) microbatch_explicitness: MicrobatchExplicitness,
    pub(super) session_start_global_step: usize,
    pub(super) total_steps: usize,
    pub(super) microbatch_size: usize,
    pub(super) use_amp: bool,
    pub(super) banner_stats: BannerStats,
    pub(super) loss_fn: HydraLoss<B>,
    pub(super) valid_loss_fn: HydraLoss<ValidBackendOf<B>>,
    pub(super) bc_exit_cfg: BcExitConfig,
}

pub(super) struct TrainingRuntime<B>
where
    B: AutodiffBackend,
{
    pub(super) model: HydraModel<B>,
    pub(super) optimizer: OptimizerAdaptor<Adam, HydraModel<B>, B>,
    pub(super) best_validation: Option<super::resume::BestValidation>,
    pub(super) global_step: usize,
    pub(super) run_start: Instant,
    pub(super) last_log_step: usize,
    pub(super) last_log_time: Instant,
    pub(super) tb: Option<EventWriter<std::fs::File>>,
    pub(super) training_log: JsonlAppender,
    pub(super) step_log: JsonlAppender,
    pub(super) head_controller: HeadActivationController,
}

pub(super) struct TrainingReaders {
    pub(super) validation_shard_reader: Option<BcShardReader>,
}

pub(super) struct RlTrainingBootstrap {
    pub(super) config: TrainConfig,
    pub(super) rl_config: RlTrainConfig,
    pub(super) resume: RlResumeContext,
    pub(super) artifacts: RlArtifactPaths,
    pub(super) model_config: HydraModelConfig,
    pub(super) device_name: String,
    pub(super) train_device: LibTorchDevice,
    pub(super) current_runtime: RlRuntimeResumeContract,
    pub(super) session_start_global_step: usize,
    pub(super) total_steps: usize,
    pub(super) loss_fn: HydraLoss<TrainBackend>,
    pub(super) rl_step_cfg: RlConfig,
    pub(super) gae_config: GaeConfig,
}

pub(super) struct RlTrainingRuntime {
    pub(super) model: HydraModel<TrainBackend>,
    pub(super) optimizer: OptimizerAdaptor<Adam, HydraModel<TrainBackend>, TrainBackend>,
    pub(super) global_step: usize,
    pub(super) run_start: Instant,
    pub(super) last_log_step: usize,
    pub(super) last_log_time: Instant,
    pub(super) tb: Option<EventWriter<std::fs::File>>,
    pub(super) step_log: JsonlAppender,
    pub(super) pipeline_state: PipelineState,
    pub(super) head_controller: HeadActivationController,
    pub(super) self_play_coordinator: CooperativeSelfPlayCoordinator,
}

fn initialize_training_bootstrap_for_backend_with_model_config<B>(
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
        source_filters: hydra_train::data::pipeline::SourceFilterConfig {
            include_source_patterns: config.source_filters.include_source_patterns.clone(),
            exclude_source_patterns: config.source_filters.exclude_source_patterns.clone(),
        },
        replay_target_profile: hydra_train::data::mjai_loader::ReplayTargetProfile::minimal_bc(),
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
    let scan_pb = super::presentation::make_bar(
        scan_sources_len as u64,
        "[scan] [{bar:40.cyan/blue}] {pos}/{len} sources {msg}",
    )?;
    scan_pb.set_message("Scanning archives...".to_string());
    let manifest = if let Some(shard_manifest_path) = config.bc_shards_manifest_path.as_ref() {
        let shard_manifest =
            hydra_train::data::bc_shards::read_bc_shard_manifest(shard_manifest_path)?;
        validate_bc_shard_manifest_for_config(&shard_manifest, &config)?;
        scan_pb.finish_with_message(format!(
            "using BC shard manifest: {} train / {} val samples",
            shard_manifest
                .splits
                .iter()
                .find(|split| split.split == BcShardSplit::Train)
                .map(|split| split.sample_count)
                .unwrap_or(0),
            shard_manifest
                .splits
                .iter()
                .find(|split| split.split == BcShardSplit::Validation)
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
    let training_log = super::artifacts::open_training_log_appender(&artifacts.training_log_path)?;
    let step_log = super::artifacts::open_step_log_appender(&artifacts.step_log_path)?;

    let banner_stats = BannerStats {
        total_sources: manifest.sources.len(),
        total_games: manifest.total_games,
        train_count: manifest.train_count,
        val_count: manifest.val_count,
        accum_steps: current_runtime.accum_steps,
        counts_exact: manifest.counts_exact,
    };

    let readers = TrainingReaders {
        validation_shard_reader: config
            .bc_shards_manifest_path
            .as_ref()
            .map(|manifest_path| load_bc_shard_reader(manifest_path, BcShardSplit::Validation))
            .transpose()?,
    };

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

fn initialize_training_bootstrap_for_backend<B>(
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

pub(super) fn initialize_training_bootstrap(
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

pub(super) fn initialize_rl_training_bootstrap(
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

    let preflight_paths = RlPreflightPaths::new(&artifacts);
    let cache_key = preflight_cache_key(
        &config,
        &model_config,
        &config.device,
        super::config::default_num_threads_for_system(),
    );
    if let Some(cached) = read_preflight_cache(&preflight_paths.cache_path)? {
        if cached.cache_key == cache_key {
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
        } else {
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
        hydra_train::training::bc::BCTrainerConfig::new(model_config.clone())
            .optimizer_config()
            .init()
            .load_record(optimizer_record)
    } else {
        hydra_train::training::bc::BCTrainerConfig::new(model_config.clone())
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
    let step_log = super::artifacts::open_rl_step_log_appender(&artifacts.step_log_path)?;
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
            run_start: Instant::now(),
            last_log_step: session_start_global_step,
            last_log_time: Instant::now(),
            tb,
            step_log,
            pipeline_state,
            head_controller,
            self_play_coordinator: CooperativeSelfPlayCoordinator::new(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::PreflightPaths;
    use crate::config::{RlPhaseConfig, RlTrainConfig};
    use crate::resume::{
        build_resume_state, build_rl_resume_state, rl_runtime_resume_contract,
        test_runtime_resume_contract, write_resume_state,
    };
    use crate::test_loose_replay_fixtures::tiny_real_mjai_replay;
    use hydra_train::config::PipelineState;
    use hydra_train::data::pipeline::DataSource;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn unique_temp_dir(label: &str) -> PathBuf {
        let base_dir = std::env::temp_dir();
        fs::create_dir_all(&base_dir).expect("create test temp root");
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        base_dir.join(format!(
            "hydra_rl_bootstrap_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn create_empty_dir(path: &Path) {
        fs::create_dir_all(path).expect("create dir");
    }

    fn cleanup_dir(path: &Path) {
        fs::remove_dir_all(path).ok();
    }

    fn latest_model_checkpoint_path(output_dir: &Path) -> PathBuf {
        output_dir.join("latest_model.mpk")
    }

    fn latest_state_path(output_dir: &Path) -> PathBuf {
        output_dir.join("latest_state.yaml")
    }

    fn save_latest_model_checkpoint(output_dir: &Path) {
        save_model_checkpoint_with_config(output_dir, HydraModelConfig::learner());
    }

    fn save_model_checkpoint_with_config(output_dir: &Path, model_config: HydraModelConfig) {
        let checkpoint_base = latest_model_checkpoint_path(output_dir).with_extension("");
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model_config
            .init::<TrainBackend>(&LibTorchDevice::Cpu)
            .save_file(&checkpoint_base, &recorder)
            .expect("save latest model checkpoint");
    }

    fn tiny_test_model_config() -> HydraModelConfig {
        HydraModelConfig::new(1)
            .with_input_channels(hydra_train::config::INPUT_CHANNELS)
            .with_hidden_channels(4)
            .with_num_groups(4)
            .with_se_bottleneck(1)
    }

    fn save_latest_tiny_model_checkpoint(output_dir: &Path) {
        save_model_checkpoint_with_config(output_dir, tiny_test_model_config());
    }

    fn save_latest_tiny_optimizer_sidecar(output_dir: &Path, data_dir: PathBuf) {
        let train_cfg =
            trainer_config_from_train_config(&dummy_bc_config(data_dir, output_dir.to_path_buf()));
        let optimizer = train_cfg
            .optimizer_config()
            .init::<TrainBackend, HydraModel<TrainBackend>>();
        let optimizer_recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        optimizer_recorder
            .record(optimizer.to_record(), output_dir.join("latest_optimizer"))
            .expect("save latest tiny optimizer sidecar");
    }

    fn initialize_training_bootstrap_with_tiny_model(
        output_dir: &Path,
        config: TrainConfig,
    ) -> Result<
        (
            TrainingBootstrap<TrainBackend>,
            TrainingRuntime<TrainBackend>,
            TrainingReaders,
        ),
        String,
    > {
        initialize_training_bootstrap_for_backend_with_model_config::<TrainBackend>(
            output_dir,
            config,
            tiny_test_model_config(),
        )
    }

    fn dummy_rl_config(output_dir: PathBuf) -> TrainConfig {
        TrainConfig {
            data_dir: PathBuf::from("/tmp/data"),
            output_dir,
            num_epochs: 1,
            batch_size: 256,
            microbatch_size: Some(64),
            validation_microbatch_size: Some(32),
            exit_sidecar_path: None,
            delta_q_sidecar_path: None,
            bc_shards_manifest_path: None,
            shard_prefetch_depth: None,
            train_fraction: 0.9,
            source_filters: hydra_train_runtime::config::SourceFilterConfig::default(),
            augment: true,
            resume_checkpoint: None,
            seed: 7,
            advanced_loss: None,
            validation_gates: crate::config::ValidationGateConfig::default(),
            rl: Some(RlTrainConfig::default()),
            bc: Default::default(),
            nsight_trace: None,
            device: "cpu".to_string(),
            precision_mode: crate::config::PrecisionMode::Fp32,
            buffer_games: 16,
            buffer_samples: 128,
            num_threads: Some(1),
            tensorboard: false,
            archive_queue_bound: 8,
            validation_every_n_epochs: 1,
            max_skip_logs_per_source: 4,
            log_every_n_steps: 10,
            validate_every_n_steps: 10,
            checkpoint_every_n_steps: 10,
            max_train_steps: Some(2),
            max_validation_batches: None,
            max_validation_samples: Some(64),
            preflight: Default::default(),
        }
    }

    fn dummy_bc_config(data_dir: PathBuf, output_dir: PathBuf) -> TrainConfig {
        let mut config = dummy_rl_config(output_dir);
        config.data_dir = data_dir;
        config.rl = None;
        config
    }

    #[test]
    fn initialize_rl_training_bootstrap_uses_rl_defaults() {
        let output_dir = unique_temp_dir("defaults");
        fs::create_dir_all(&output_dir).expect("create output dir");
        let config = dummy_rl_config(output_dir.clone());
        let rl_cfg = config.rl.clone().expect("rl config");

        let (bootstrap, runtime) =
            initialize_rl_training_bootstrap(&output_dir, config, rl_cfg).expect("rl bootstrap");

        assert_eq!(bootstrap.rl_config.games_per_batch, 4);
        assert_eq!(bootstrap.rl_config.phase, RlPhaseConfig::DrdaAchSelfPlay);
        assert_eq!(
            runtime.pipeline_state.phase,
            hydra_train::config::TrainingPhase::DrdaAchSelfPlay
        );
        assert_eq!(runtime.global_step, 0);
        cleanup_dir(&output_dir);
    }

    #[test]
    fn rl_bootstrap_applies_preflight_cache_override() {
        use crate::artifacts::{RlPreflightPaths, write_preflight_cache};
        use crate::preflight_fingerprint::preflight_cache_key;
        use hydra_train::preflight::{
            EffectiveRuntimeConfig, LoaderRuntimeConfig, PreflightCacheEntry, SelectedRuntimeConfig,
        };

        let output_dir = unique_temp_dir("cache_override");
        fs::create_dir_all(&output_dir).expect("create output dir");
        let config = dummy_rl_config(output_dir.clone());
        let rl_cfg = config.rl.clone().expect("rl config");

        let rl_artifacts = crate::artifacts::RlArtifactPaths::new(&config.output_dir, 0);
        rl_artifacts.create_root_dir().expect("create rl dir");
        let paths = RlPreflightPaths::new(&rl_artifacts);
        let model_config = HydraModelConfig::learner();
        let key = preflight_cache_key(
            &config,
            &model_config,
            &config.device,
            crate::config::default_num_threads_for_system(),
        );
        let tuned_games = 16;
        let tuned_microbatch = 32;
        write_preflight_cache(
            &paths.cache_path,
            &PreflightCacheEntry {
                cache_key: key,
                runtime: EffectiveRuntimeConfig {
                    selected: SelectedRuntimeConfig {
                        train_microbatch_size: tuned_microbatch,
                        validation_microbatch_size: config
                            .validation_microbatch_size
                            .unwrap_or(config.batch_size),
                        accum_steps: config.batch_size.div_ceil(tuned_microbatch).max(1),
                    },
                    loader: LoaderRuntimeConfig {
                        num_threads: config.num_threads,
                        buffer_games: tuned_games,
                        buffer_samples: config.buffer_samples,
                        archive_queue_bound: config.archive_queue_bound,
                    },
                },
                benchmark: None,
            },
        )
        .expect("write cache");

        let (bootstrap, _runtime) =
            initialize_rl_training_bootstrap(&output_dir, config, rl_cfg).expect("rl bootstrap");

        assert_eq!(
            bootstrap.rl_config.games_per_batch, tuned_games,
            "bootstrap should apply preflight-cached games_per_batch"
        );
        assert_eq!(
            bootstrap.rl_config.microbatch_size,
            Some(tuned_microbatch),
            "bootstrap should apply preflight-cached rl.microbatch_size"
        );
        cleanup_dir(&output_dir);
    }

    #[test]
    fn rl_bootstrap_ignores_stale_preflight_cache() {
        use crate::artifacts::{RlPreflightPaths, write_preflight_cache};
        use hydra_train::preflight::{
            EffectiveRuntimeConfig, HardwareFingerprint, LoaderRuntimeConfig, PreflightCacheEntry,
            PreflightCacheKey, SelectedRuntimeConfig, WorkloadFingerprint,
        };

        let output_dir = unique_temp_dir("cache_stale");
        fs::create_dir_all(&output_dir).expect("create output dir");
        let config = dummy_rl_config(output_dir.clone());
        let rl_cfg = config.rl.clone().expect("rl config");
        let original_games = rl_cfg.games_per_batch;
        let original_microbatch = rl_cfg.microbatch_size;

        let rl_artifacts = crate::artifacts::RlArtifactPaths::new(&config.output_dir, 0);
        rl_artifacts.create_root_dir().expect("create rl dir");
        let paths = RlPreflightPaths::new(&rl_artifacts);
        let stale_key = PreflightCacheKey {
            hardware: HardwareFingerprint {
                device_label: "stale-device".to_string(),
                backend: "burn-libtorch".to_string(),
                cpu_logical_cores: 999,
                total_memory_bytes: None,
            },
            workload: WorkloadFingerprint {
                batch_size: 9999,
                augment: false,
                precision_mode: "fp32".to_string(),
                train_fraction_bits: 0,
                max_skip_logs_per_source: 0,
                max_validation_batches: None,
                max_validation_samples: None,
                model_signature: "stale".to_string(),
                code_signature: "stale".to_string(),
                advanced_loss_signature: "stale".to_string(),
                preflight_config_signature: "stale".to_string(),
                explicit_train_microbatch: None,
                explicit_validation_microbatch: None,
            },
        };
        write_preflight_cache(
            &paths.cache_path,
            &PreflightCacheEntry {
                cache_key: stale_key,
                runtime: EffectiveRuntimeConfig {
                    selected: SelectedRuntimeConfig {
                        train_microbatch_size: 64,
                        validation_microbatch_size: 32,
                        accum_steps: 1,
                    },
                    loader: LoaderRuntimeConfig {
                        num_threads: Some(1),
                        buffer_games: 999,
                        buffer_samples: 128,
                        archive_queue_bound: 8,
                    },
                },
                benchmark: None,
            },
        )
        .expect("write stale cache");

        let (bootstrap, _runtime) =
            initialize_rl_training_bootstrap(&output_dir, config, rl_cfg).expect("rl bootstrap");

        assert_eq!(
            bootstrap.rl_config.games_per_batch, original_games,
            "bootstrap should ignore stale preflight cache"
        );
        assert_eq!(
            bootstrap.rl_config.microbatch_size, original_microbatch,
            "bootstrap should preserve configured microbatch when preflight cache is stale"
        );
        cleanup_dir(&output_dir);
    }

    #[test]
    fn initialize_rl_training_bootstrap_creates_tensorboard_writer_when_enabled() {
        let output_dir = unique_temp_dir("rl_tensorboard_enabled");
        create_empty_dir(&output_dir);

        let mut config = dummy_rl_config(output_dir.clone());
        config.tensorboard = true;
        let rl_cfg = config.rl.clone().expect("rl config");

        let (bootstrap, runtime) = initialize_rl_training_bootstrap(&output_dir, config, rl_cfg)
            .expect("rl bootstrap with tensorboard");

        assert!(runtime.tb.is_some(), "tensorboard writer should be created");
        assert!(bootstrap.artifacts.tb_root.is_dir());
        assert!(bootstrap.artifacts.tb_session_dir.is_dir());
        cleanup_dir(&output_dir);
    }

    #[test]
    fn initialize_training_bootstrap_rejects_invalid_config_early() {
        let root_dir = unique_temp_dir("bc_invalid_config");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        let mut config = dummy_bc_config(data_dir, output_dir.clone());
        config.num_epochs = 0;

        let err = initialize_training_bootstrap(&output_dir, config)
            .err()
            .expect("invalid config should fail before bootstrap work");

        assert_eq!(err, "num_epochs must be greater than 0");
        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_keeps_fresh_run_config_runtime_even_with_matching_preflight_cache()
     {
        use crate::artifacts::{PreflightPaths, write_preflight_cache};
        use crate::preflight_fingerprint::preflight_cache_key;
        use hydra_train::preflight::{
            EffectiveRuntimeConfig, LoaderRuntimeConfig, PreflightCacheEntry, SelectedRuntimeConfig,
        };

        let root_dir = unique_temp_dir("bc_ignore_preflight_cache");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        let config = dummy_bc_config(data_dir, output_dir.clone());
        let original_buffer_games = config.buffer_games;
        let original_buffer_samples = config.buffer_samples;
        let original_archive_queue_bound = config.archive_queue_bound;
        let original_num_threads = config.num_threads;
        let original_microbatch = config.microbatch_size.expect("train microbatch");
        let original_validation_microbatch = config
            .validation_microbatch_size
            .expect("validation microbatch");
        let original_accum_steps = config.batch_size.div_ceil(original_microbatch).max(1);

        let bc_artifacts = crate::artifacts::BcArtifactPaths::new(&config.output_dir, 0);
        bc_artifacts.create_root_dir().expect("create bc dir");
        let paths = PreflightPaths::new(&bc_artifacts);
        let model_config = HydraModelConfig::learner();
        let key = preflight_cache_key(
            &config,
            &model_config,
            &config.device,
            crate::config::default_num_threads_for_system(),
        );

        write_preflight_cache(
            &paths.cache_path,
            &PreflightCacheEntry {
                cache_key: key,
                runtime: EffectiveRuntimeConfig {
                    selected: SelectedRuntimeConfig {
                        train_microbatch_size: 32,
                        validation_microbatch_size: 16,
                        accum_steps: config.batch_size.div_ceil(32).max(1),
                    },
                    loader: LoaderRuntimeConfig {
                        num_threads: Some(1),
                        buffer_games: 999,
                        buffer_samples: 777,
                        archive_queue_bound: 5,
                    },
                },
                benchmark: None,
            },
        )
        .expect("write cache");

        let expected_accum_steps = config.batch_size.div_ceil(32).max(1);
        let (bootstrap, _runtime, _readers) =
            initialize_training_bootstrap(&output_dir, config).expect("bc bootstrap");

        assert_ne!(original_buffer_games, 999);
        assert_ne!(original_buffer_samples, 777);
        assert_ne!(original_archive_queue_bound, 5);
        assert_ne!(original_microbatch, 32);
        assert_ne!(original_validation_microbatch, 16);
        assert_ne!(original_accum_steps, expected_accum_steps);
        assert_eq!(bootstrap.loader_config.buffer_games, original_buffer_games);
        assert_eq!(
            bootstrap.loader_config.buffer_samples,
            original_buffer_samples
        );
        assert_eq!(
            bootstrap.loader_config.archive_queue_bound,
            original_archive_queue_bound
        );
        assert_eq!(bootstrap.microbatch_size, original_microbatch);
        assert_eq!(
            bootstrap.current_runtime.train_microbatch_size,
            original_microbatch
        );
        assert_eq!(
            bootstrap.current_runtime.validation_microbatch_size,
            original_validation_microbatch
        );
        assert_eq!(bootstrap.current_runtime.accum_steps, original_accum_steps);
        assert_eq!(bootstrap.config.num_threads, original_num_threads);
        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_ignores_stale_preflight_cache_at_epoch_boundary() {
        use crate::artifacts::{PreflightPaths, write_preflight_cache};
        use hydra_train::preflight::{
            EffectiveRuntimeConfig, HardwareFingerprint, LoaderRuntimeConfig, PreflightCacheEntry,
            PreflightCacheKey, SelectedRuntimeConfig, WorkloadFingerprint,
        };

        let root_dir = unique_temp_dir("bc_epoch_boundary_stale_preflight_cache");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        save_latest_tiny_model_checkpoint(&output_dir);
        save_latest_tiny_optimizer_sidecar(&output_dir, data_dir.clone());

        let mut config = dummy_bc_config(data_dir, output_dir.clone());
        config.resume_checkpoint = Some(latest_model_checkpoint_path(&output_dir));

        let state = build_resume_state(
            2,
            0,
            17,
            None,
            test_runtime_resume_contract(config.batch_size, 32, 16),
        );
        write_resume_state(&latest_state_path(&output_dir), &state).expect("write resume state");

        let bc_artifacts = crate::artifacts::BcArtifactPaths::new(&config.output_dir, 17);
        bc_artifacts.create_root_dir().expect("create bc dir");
        let paths = PreflightPaths::new(&bc_artifacts);
        let stale_key = PreflightCacheKey {
            hardware: HardwareFingerprint {
                device_label: "stale-device".to_string(),
                backend: "burn-libtorch".to_string(),
                cpu_logical_cores: 999,
                total_memory_bytes: None,
            },
            workload: WorkloadFingerprint {
                batch_size: 9999,
                augment: false,
                precision_mode: "fp32".to_string(),
                train_fraction_bits: 0,
                max_skip_logs_per_source: 0,
                max_validation_batches: None,
                max_validation_samples: None,
                model_signature: "stale".to_string(),
                code_signature: "stale".to_string(),
                advanced_loss_signature: "stale".to_string(),
                preflight_config_signature: "stale".to_string(),
                explicit_train_microbatch: None,
                explicit_validation_microbatch: None,
            },
        };
        write_preflight_cache(
            &paths.cache_path,
            &PreflightCacheEntry {
                cache_key: stale_key,
                runtime: EffectiveRuntimeConfig {
                    selected: SelectedRuntimeConfig {
                        train_microbatch_size: 32,
                        validation_microbatch_size: 16,
                        accum_steps: config.batch_size.div_ceil(32).max(1),
                    },
                    loader: LoaderRuntimeConfig {
                        num_threads: Some(1),
                        buffer_games: 999,
                        buffer_samples: 777,
                        archive_queue_bound: 5,
                    },
                },
                benchmark: None,
            },
        )
        .expect("write stale cache");

        let (bootstrap, runtime, _readers) =
            initialize_training_bootstrap_with_tiny_model(&output_dir, config)
                .expect("stale epoch-boundary cache");

        assert_eq!(bootstrap.current_runtime.train_microbatch_size, 64);
        assert_eq!(bootstrap.current_runtime.validation_microbatch_size, 32);
        assert_eq!(bootstrap.current_runtime.accum_steps, 4);
        assert_eq!(bootstrap.microbatch_size, 64);
        assert_eq!(runtime.global_step, 17);

        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_reuses_only_selected_runtime_from_matching_preflight_cache_at_epoch_boundary()
     {
        use crate::artifacts::{PreflightPaths, write_preflight_cache};
        use crate::preflight_fingerprint::preflight_cache_key;
        use hydra_train::preflight::{
            EffectiveRuntimeConfig, LoaderRuntimeConfig, PreflightCacheEntry, SelectedRuntimeConfig,
        };

        let root_dir = unique_temp_dir("bc_epoch_boundary_preflight_override");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        save_latest_tiny_model_checkpoint(&output_dir);
        save_latest_tiny_optimizer_sidecar(&output_dir, data_dir.clone());

        let mut config = dummy_bc_config(data_dir, output_dir.clone());
        config.resume_checkpoint = Some(latest_model_checkpoint_path(&output_dir));
        let original_num_threads = config.num_threads;
        let original_buffer_games = config.buffer_games;
        let original_buffer_samples = config.buffer_samples;
        let original_archive_queue_bound = config.archive_queue_bound;

        let state = build_resume_state(
            2,
            0,
            17,
            None,
            test_runtime_resume_contract(config.batch_size, 32, 16),
        );
        write_resume_state(&latest_state_path(&output_dir), &state).expect("write resume state");

        let bc_artifacts = crate::artifacts::BcArtifactPaths::new(&config.output_dir, 17);
        bc_artifacts.create_root_dir().expect("create bc dir");
        let paths = PreflightPaths::new(&bc_artifacts);
        let model_config = tiny_test_model_config();
        let key = preflight_cache_key(
            &config,
            &model_config,
            &config.device,
            crate::config::default_num_threads_for_system(),
        );
        write_preflight_cache(
            &paths.cache_path,
            &PreflightCacheEntry {
                cache_key: key,
                runtime: EffectiveRuntimeConfig {
                    selected: SelectedRuntimeConfig {
                        train_microbatch_size: 32,
                        validation_microbatch_size: 16,
                        accum_steps: config.batch_size.div_ceil(32).max(1),
                    },
                    loader: LoaderRuntimeConfig {
                        num_threads: Some(1),
                        buffer_games: 999,
                        buffer_samples: 777,
                        archive_queue_bound: 5,
                    },
                },
                benchmark: None,
            },
        )
        .expect("write cache");

        let (bootstrap, runtime, _readers) =
            initialize_training_bootstrap_with_tiny_model(&output_dir, config)
                .expect("epoch-boundary override");

        assert_eq!(bootstrap.resume.start_epoch, 2);
        assert_eq!(bootstrap.session_start_global_step, 17);
        assert_eq!(bootstrap.resume.steps_to_skip_for_epoch(2), 0);
        assert_eq!(bootstrap.current_runtime.train_microbatch_size, 32);
        assert_eq!(bootstrap.current_runtime.validation_microbatch_size, 16);
        assert_eq!(bootstrap.current_runtime.accum_steps, 8);
        assert_eq!(bootstrap.microbatch_size, 32);
        assert_eq!(bootstrap.loader_config.buffer_games, original_buffer_games);
        assert_eq!(
            bootstrap.loader_config.buffer_samples,
            original_buffer_samples
        );
        assert_eq!(
            bootstrap.loader_config.archive_queue_bound,
            original_archive_queue_bound
        );
        assert_eq!(bootstrap.config.num_threads, original_num_threads);
        assert_eq!(runtime.global_step, 17);

        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_reports_missing_data_dir() {
        let root_dir = unique_temp_dir("bc_missing_data_dir");
        let output_dir = root_dir.join("output");
        create_empty_dir(&output_dir);

        let missing_data_dir = root_dir.join("missing_data");
        let config = dummy_bc_config(missing_data_dir.clone(), output_dir.clone());

        let err = initialize_training_bootstrap(&output_dir, config)
            .err()
            .expect("missing data dir should fail before scan setup completes");

        assert!(
            err.contains(&format!(
                "failed to read data dir {}",
                missing_data_dir.display()
            )),
            "unexpected error: {err}"
        );
        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_rescans_when_manifest_cache_data_dir_mismatches() {
        let root_dir = unique_temp_dir("bc_manifest_cache_mismatch");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);
        let replay_path = data_dir.join("game.mjai.json");
        fs::write(&replay_path, tiny_real_mjai_replay()).expect("write real replay fixture");

        let config = dummy_bc_config(data_dir.clone(), output_dir.clone());
        let bc_artifacts = crate::artifacts::BcArtifactPaths::new(&config.output_dir, 0);
        bc_artifacts.create_root_dir().expect("create bc dir");
        let paths = PreflightPaths::new(&bc_artifacts);
        write_manifest_cache(
            &paths.manifest_cache_path,
            &ManifestCacheEntry {
                data_dir: root_dir.join("stale-data-dir"),
                train_fraction_bits: config.train_fraction.to_bits(),
                include_source_patterns: Vec::new(),
                exclude_source_patterns: Vec::new(),
                manifest: DataManifest {
                    sources: vec![DataSource::LooseFile(root_dir.join("stale.mjai.json"))],
                    total_games: 1,
                    train_count: 1,
                    val_count: 0,
                    counts_exact: true,
                },
            },
        )
        .expect("write stale manifest cache");

        let (bootstrap, _, _readers) = initialize_training_bootstrap(&output_dir, config)
            .expect("bootstrap should rescan real data on stale manifest cache");

        assert_eq!(bootstrap.manifest.sources.len(), 1);
        assert_eq!(
            bootstrap.manifest.sources[0],
            DataSource::LooseFile(replay_path)
        );
        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_rescans_when_manifest_cache_train_fraction_mismatches() {
        let root_dir = unique_temp_dir("bc_manifest_fraction_mismatch");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);
        let replay_path = data_dir.join("game.mjai.json");
        fs::write(&replay_path, tiny_real_mjai_replay()).expect("write real replay fixture");

        let config = dummy_bc_config(data_dir.clone(), output_dir.clone());
        let bc_artifacts = crate::artifacts::BcArtifactPaths::new(&config.output_dir, 0);
        bc_artifacts.create_root_dir().expect("create bc dir");
        let paths = PreflightPaths::new(&bc_artifacts);
        write_manifest_cache(
            &paths.manifest_cache_path,
            &ManifestCacheEntry {
                data_dir: data_dir.clone(),
                train_fraction_bits: 0.0f32.to_bits(),
                include_source_patterns: Vec::new(),
                exclude_source_patterns: Vec::new(),
                manifest: DataManifest {
                    sources: vec![DataSource::LooseFile(root_dir.join("stale.mjai.json"))],
                    total_games: 1,
                    train_count: 0,
                    val_count: 1,
                    counts_exact: true,
                },
            },
        )
        .expect("write stale train-fraction manifest cache");

        let (bootstrap, _, _readers) = initialize_training_bootstrap(&output_dir, config)
            .expect("bootstrap should rescan real data on stale train_fraction manifest cache");

        assert_eq!(bootstrap.manifest.sources.len(), 1);
        assert_eq!(
            bootstrap.manifest.sources[0],
            DataSource::LooseFile(replay_path)
        );
        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_rejects_invalid_exit_sidecar_path_early() {
        let root_dir = unique_temp_dir("bc_invalid_exit_sidecar");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        let mut config = dummy_bc_config(data_dir, output_dir.clone());
        let missing_sidecar = root_dir.join("missing_exit_sidecar.jsonl");
        config.exit_sidecar_path = Some(missing_sidecar.clone());

        let err = initialize_training_bootstrap(&output_dir, config)
            .err()
            .expect("invalid exit sidecar path should fail before scan/runtime work");

        assert!(err.starts_with(&format!(
            "failed to load replay ExIt sidecar {}:",
            missing_sidecar.display()
        )));
        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_rejects_invalid_delta_q_sidecar_path_early() {
        let root_dir = unique_temp_dir("bc_invalid_delta_q_sidecar");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        let mut config = dummy_bc_config(data_dir, output_dir.clone());
        let missing_sidecar = root_dir.join("missing_delta_q_sidecar.jsonl");
        config.delta_q_sidecar_path = Some(missing_sidecar.clone());

        let err = initialize_training_bootstrap(&output_dir, config)
            .err()
            .expect("invalid delta_q sidecar path should fail before scan/runtime work");

        assert!(err.starts_with(&format!(
            "failed to load replay delta_q sidecar {}:",
            missing_sidecar.display()
        )));
        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_requires_optimizer_sidecar_when_resume_state_demands_it() {
        let root_dir = unique_temp_dir("bc_missing_optimizer_sidecar");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        save_latest_tiny_model_checkpoint(&output_dir);

        let mut config = dummy_bc_config(data_dir, output_dir.clone());
        config.resume_checkpoint = Some(latest_model_checkpoint_path(&output_dir));

        let state = build_resume_state(
            1,
            0,
            12,
            None,
            test_runtime_resume_contract(
                config.batch_size,
                config.microbatch_size.expect("train microbatch"),
                config
                    .validation_microbatch_size
                    .expect("validation microbatch"),
            ),
        );
        write_resume_state(&latest_state_path(&output_dir), &state).expect("write resume state");

        let err = initialize_training_bootstrap_with_tiny_model(&output_dir, config)
            .err()
            .expect("resume without optimizer sidecar should fail after checkpoint load");

        assert_eq!(
            err,
            format!(
                "resume state for checkpoint {} requires optimizer sidecar, but none was found next to that checkpoint",
                output_dir.join("latest_model").display()
            )
        );
        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_rejects_resume_batch_size_mismatch() {
        let root_dir = unique_temp_dir("bc_resume_batch_mismatch");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        let mut config = dummy_bc_config(data_dir, output_dir.clone());
        config.resume_checkpoint = Some(latest_model_checkpoint_path(&output_dir));

        let resumed_batch_size = config.batch_size / 2;
        let state = build_resume_state(
            0,
            0,
            11,
            None,
            test_runtime_resume_contract(
                resumed_batch_size,
                config.microbatch_size.expect("train microbatch"),
                config
                    .validation_microbatch_size
                    .expect("validation microbatch"),
            ),
        );
        write_resume_state(&latest_state_path(&output_dir), &state).expect("write resume state");

        let err = initialize_training_bootstrap(&output_dir, config)
            .err()
            .expect("resume batch mismatch should fail before model load");

        assert_eq!(
            err,
            format!(
                "resume batch_size mismatch: checkpoint={} current=256",
                resumed_batch_size
            )
        );
        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_rejects_partial_epoch_runtime_mismatch() {
        let root_dir = unique_temp_dir("bc_partial_epoch_runtime_mismatch");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        let mut config = dummy_bc_config(data_dir, output_dir.clone());
        config.resume_checkpoint = Some(latest_model_checkpoint_path(&output_dir));

        let state = build_resume_state(
            2,
            3,
            17,
            None,
            test_runtime_resume_contract(config.batch_size, 32, 16),
        );
        write_resume_state(&latest_state_path(&output_dir), &state).expect("write resume state");

        let err = initialize_training_bootstrap(&output_dir, config)
            .err()
            .expect("partial epoch mismatch should fail before model load");

        assert_eq!(
            err,
            "partial-epoch resume requires identical runtime contract; checkpoint train_mb=32 val_mb=16 accum_steps=8 current train_mb=64 val_mb=32 accum_steps=4"
        );
        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_rejects_partial_epoch_runtime_mismatch_even_with_matching_preflight_cache()
     {
        use crate::artifacts::{PreflightPaths, write_preflight_cache};
        use crate::preflight_fingerprint::preflight_cache_key;
        use hydra_train::preflight::{
            EffectiveRuntimeConfig, LoaderRuntimeConfig, PreflightCacheEntry, SelectedRuntimeConfig,
        };

        let root_dir = unique_temp_dir("bc_partial_epoch_runtime_mismatch_with_cache");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        let mut config = dummy_bc_config(data_dir, output_dir.clone());
        config.resume_checkpoint = Some(latest_model_checkpoint_path(&output_dir));

        let state = build_resume_state(
            2,
            3,
            17,
            None,
            test_runtime_resume_contract(config.batch_size, 32, 16),
        );
        write_resume_state(&latest_state_path(&output_dir), &state).expect("write resume state");

        let bc_artifacts = crate::artifacts::BcArtifactPaths::new(&config.output_dir, 17);
        bc_artifacts.create_root_dir().expect("create bc dir");
        let paths = PreflightPaths::new(&bc_artifacts);
        let model_config = HydraModelConfig::learner();
        let key = preflight_cache_key(
            &config,
            &model_config,
            &config.device,
            crate::config::default_num_threads_for_system(),
        );
        write_preflight_cache(
            &paths.cache_path,
            &PreflightCacheEntry {
                cache_key: key,
                runtime: EffectiveRuntimeConfig {
                    selected: SelectedRuntimeConfig {
                        train_microbatch_size: 32,
                        validation_microbatch_size: 16,
                        accum_steps: config.batch_size.div_ceil(32).max(1),
                    },
                    loader: LoaderRuntimeConfig {
                        num_threads: Some(1),
                        buffer_games: 999,
                        buffer_samples: 777,
                        archive_queue_bound: 5,
                    },
                },
                benchmark: None,
            },
        )
        .expect("write cache");

        let err = initialize_training_bootstrap(&output_dir, config)
            .err()
            .expect("partial epoch mismatch should still fail with matching cache present");

        assert_eq!(
            err,
            "partial-epoch resume requires identical runtime contract; checkpoint train_mb=32 val_mb=16 accum_steps=8 current train_mb=64 val_mb=32 accum_steps=4"
        );
        cleanup_dir(&root_dir);
    }

    #[test]
    fn initialize_training_bootstrap_allows_epoch_boundary_runtime_change() {
        let root_dir = unique_temp_dir("bc_epoch_boundary_runtime_change");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        save_latest_tiny_model_checkpoint(&output_dir);
        save_latest_tiny_optimizer_sidecar(&output_dir, data_dir.clone());

        let mut config = dummy_bc_config(data_dir, output_dir.clone());
        config.resume_checkpoint = Some(latest_model_checkpoint_path(&output_dir));

        let state = build_resume_state(
            2,
            0,
            17,
            None,
            test_runtime_resume_contract(config.batch_size, 32, 16),
        );
        write_resume_state(&latest_state_path(&output_dir), &state).expect("write resume state");

        let (bootstrap, runtime, _readers) =
            initialize_training_bootstrap_with_tiny_model(&output_dir, config)
                .expect("epoch-boundary resume");

        assert_eq!(bootstrap.resume.start_epoch, 2);
        assert_eq!(bootstrap.session_start_global_step, 17);
        assert_eq!(bootstrap.resume.steps_to_skip_for_epoch(2), 0);
        assert_eq!(bootstrap.current_runtime.train_microbatch_size, 64);
        assert_eq!(bootstrap.current_runtime.validation_microbatch_size, 32);
        assert_eq!(bootstrap.current_runtime.accum_steps, 4);
        assert_eq!(runtime.global_step, 17);

        cleanup_dir(&root_dir);
    }

    #[test]
    fn validate_bc_shard_manifest_for_config_rejects_train_fraction_mismatch() {
        let root_dir = unique_temp_dir("bc_shard_train_fraction_mismatch");
        let output_dir = root_dir.join("output");
        let data_dir = root_dir.join("data");
        create_empty_dir(&output_dir);
        create_empty_dir(&data_dir);

        let mut config = dummy_bc_config(data_dir, output_dir);
        let manifest = BcShardManifest {
            manifest_version: hydra_train::data::bc_shards::BC_SHARD_MANIFEST_VERSION,
            shard_version: hydra_train::data::bc_shards::BC_SHARD_VERSION,
            shard_header_size: hydra_train::data::bc_shards::BC_SHARD_HEADER_SIZE,
            base_record_size: hydra_train::data::bc_shards::BC_BASE_RECORD_SIZE,
            max_record_size: hydra_train::data::bc_shards::BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
            obs_size: hydra_core::encoder::OBS_SIZE,
            num_channels: hydra_core::encoder::NUM_CHANNELS,
            action_space: hydra_core::action::HYDRA_ACTION_SPACE,
            train_fraction: 0.5,
            shard_samples: 10_000,
            augment_runtime: true,
            input: config.data_dir.display().to_string(),
            output_dir: config.output_dir.display().to_string(),
            created_at: "1970-01-01T00:00:00Z".to_string(),
            source_count: 0,
            source_total_games_hint: 0,
            source_train_count_hint: 0,
            source_val_count_hint: 0,
            source_counts_exact: true,
            exit_sidecar: None,
            delta_q_sidecar: None,
            totals: Default::default(),
            splits: Vec::new(),
        };

        let err = validate_bc_shard_manifest_for_config(&manifest, &config)
            .expect_err("train_fraction mismatch should be rejected");
        assert!(err.contains("train_fraction"));
        config.train_fraction = manifest.train_fraction;
        assert!(validate_bc_shard_manifest_for_config(&manifest, &config).is_ok());
        cleanup_dir(&root_dir);
    }
    #[test]
    fn initialize_rl_training_bootstrap_rejects_resume_runtime_mismatch() {
        let output_dir = unique_temp_dir("rl_resume_runtime_mismatch");
        create_empty_dir(&output_dir);

        let mut config = dummy_rl_config(output_dir.clone());
        config.resume_checkpoint = Some(latest_model_checkpoint_path(&output_dir));
        let rl_cfg = config.rl.clone().expect("rl config");
        let mut mismatched_runtime = rl_runtime_resume_contract(&rl_cfg);
        mismatched_runtime.games_per_batch += 3;
        let state = build_rl_resume_state(
            9,
            PipelineState {
                phase: rl_cfg.phase.to_training_phase(),
                ..PipelineState::default()
            },
            mismatched_runtime,
        );
        let state_yaml = serde_yaml::to_string(&state).expect("serialize rl resume state");
        fs::write(latest_state_path(&output_dir), state_yaml).expect("write rl resume state");

        let err = initialize_rl_training_bootstrap(&output_dir, config, rl_cfg)
            .err()
            .expect("rl runtime mismatch should fail before model load");

        assert_eq!(
            err,
            "RL resume runtime mismatch: checkpoint games_per_batch=7 microbatch_size=128 phase=DrdaAchSelfPlay current games_per_batch=4 microbatch_size=128 phase=DrdaAchSelfPlay"
        );
        cleanup_dir(&output_dir);
    }

    #[test]
    fn initialize_rl_training_bootstrap_requires_optimizer_sidecar_when_resume_state_demands_it() {
        let output_dir = unique_temp_dir("rl_missing_optimizer_sidecar");
        create_empty_dir(&output_dir);

        save_latest_model_checkpoint(&output_dir);

        let mut config = dummy_rl_config(output_dir.clone());
        config.resume_checkpoint = Some(latest_model_checkpoint_path(&output_dir));
        let rl_cfg = config.rl.clone().expect("rl config");
        let state = build_rl_resume_state(
            9,
            PipelineState {
                phase: rl_cfg.phase.to_training_phase(),
                ..PipelineState::default()
            },
            rl_runtime_resume_contract(&rl_cfg),
        );
        let state_yaml = serde_yaml::to_string(&state).expect("serialize rl resume state");
        fs::write(latest_state_path(&output_dir), state_yaml).expect("write rl resume state");

        let err = initialize_rl_training_bootstrap(&output_dir, config, rl_cfg)
            .err()
            .expect("rl resume without optimizer sidecar should fail after checkpoint load");

        assert_eq!(
            err,
            format!(
                "RL resume state for checkpoint {} requires optimizer sidecar, but none was found next to that checkpoint",
                output_dir.join("latest_model").display()
            )
        );
        cleanup_dir(&output_dir);
    }
}
