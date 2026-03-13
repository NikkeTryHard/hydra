use std::fs;
use std::path::Path;
use std::time::Instant;

use burn::backend::libtorch::LibTorchDevice;
use burn::optim::Adam;
use burn::optim::Optimizer;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::prelude::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use tboard::EventWriter;

use hydra_train::data::pipeline::{
    DataManifest, StreamingLoaderConfig, scan_data_sources_with_progress,
};
use hydra_train::model::{HydraModel, HydraModelConfig};
use hydra_train::training::bc::BCTrainerConfig;
use hydra_train::training::losses::HydraLoss;

use super::artifacts::BcArtifactPaths;
use super::config::{
    TrainConfig, configure_threads, device_label, train_device, train_microbatch_size,
    trainer_config_from_train_config, validate_config,
};
use super::loss_policy::build_loss_config;
use super::progress::BannerStats;
use super::resume::{
    ResumeContext, runtime_resume_contract, validate_resume_runtime_compatibility,
};
use super::schedule::schedule_total_steps;
use super::{TrainBackend, ValidBackend};

pub(super) struct TrainingBootstrap {
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
    pub(super) session_start_global_step: usize,
    pub(super) total_steps: usize,
    pub(super) microbatch_size: usize,
    pub(super) banner_stats: BannerStats,
    pub(super) loss_fn: HydraLoss<TrainBackend>,
    pub(super) valid_loss_fn: HydraLoss<ValidBackend>,
}

pub(super) struct TrainingRuntime {
    pub(super) model: HydraModel<TrainBackend>,
    pub(super) optimizer: OptimizerAdaptor<Adam, HydraModel<TrainBackend>, TrainBackend>,
    pub(super) best_validation: Option<super::resume::BestValidation>,
    pub(super) global_step: usize,
    pub(super) run_start: Instant,
    pub(super) last_log_step: usize,
    pub(super) last_log_time: Instant,
    pub(super) tb: Option<EventWriter<std::fs::File>>,
}

pub(super) fn initialize_training_bootstrap(
    _config_path: &Path,
    config: TrainConfig,
) -> Result<(TrainingBootstrap, TrainingRuntime), String> {
    validate_config(&config)?;
    configure_threads(config.num_threads)?;

    let resume = ResumeContext::load(&config)?;
    let session_start_global_step = resume.session_start_global_step;
    let artifacts = BcArtifactPaths::new(&config.output_dir, session_start_global_step);
    artifacts.create_root_dir()?;

    let loader_config = StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        aggregate_skip_logs: false,
    };

    let scan_sources_len = if config.data_dir.is_file() {
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
    let manifest =
        scan_data_sources_with_progress(&config.data_dir, config.train_fraction, Some(&scan_pb))
            .map_err(|err| {
                format!(
                    "failed to scan MJAI data from {}: {err}",
                    config.data_dir.display()
                )
            })?;
    if manifest.counts_exact {
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

    let train_cfg = trainer_config_from_train_config(&config);
    train_cfg
        .validate()
        .map_err(|err| format!("invalid trainer config: {err}"))?;

    let device_name = device_label(&config.device);
    let model_config = HydraModelConfig::learner();
    let current_runtime = runtime_resume_contract(&config);
    if let Some(state) = resume.state.as_ref() {
        validate_resume_runtime_compatibility(state, current_runtime)?;
    }
    let train_device = train_device(&config.device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let mut model = model_config.init::<TrainBackend>(&train_device);
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
    let loss_fn = HydraLoss::<TrainBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let valid_loss_fn =
        HydraLoss::<ValidBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let total_steps = schedule_total_steps(&config, session_start_global_step);
    let microbatch_size = train_microbatch_size(&config);
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

    let banner_stats = BannerStats {
        total_sources: manifest.sources.len(),
        total_games: manifest.total_games,
        train_count: manifest.train_count,
        val_count: manifest.val_count,
        accum_steps: current_runtime.accum_steps,
        counts_exact: manifest.counts_exact,
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
            session_start_global_step,
            total_steps,
            microbatch_size,
            banner_stats,
            loss_fn,
            valid_loss_fn,
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
        },
    ))
}
