use std::fs;
use std::path::Path;
use std::time::Instant;

use burn::backend::libtorch::LibTorchDevice;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::Adam;
use burn::optim::Optimizer;
use burn::prelude::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use colored::Colorize;
use tboard::EventWriter;

use hydra_train::config::PipelineState;
use hydra_train::data::pipeline::{
    scan_data_sources_with_progress, DataManifest, StreamingLoaderConfig,
};
use hydra_train::model::{HydraModel, HydraModelConfig};
use hydra_train::training::bc::{BCTrainerConfig, BcExitConfig};
use hydra_train::training::gae::GaeConfig;
use hydra_train::training::head_gates::{HeadActivationConfig, HeadActivationController};
use hydra_train::training::losses::HydraLoss;
use hydra_train::training::replay_exit::{
    source_net_hash_from_checkpoint_identity, ExitSidecarIndex,
};
use hydra_train::training::rl::RlConfig;

use super::artifacts::{read_preflight_cache, BcArtifactPaths, RlArtifactPaths, RlPreflightPaths};
use super::config::{
    configure_threads, device_label, train_device, train_microbatch_size,
    trainer_config_from_train_config, validate_config, RlTrainConfig, TrainConfig,
};
use super::config_runtime::rl_config_from_train_config;
use super::loss_policy::{build_bc_exit_config, build_loss_config, build_rl_loss_config};
use super::preflight_fingerprint::preflight_cache_key;
use super::presentation::timestamped;
use super::progress::BannerStats;
use super::resume::{
    rl_runtime_resume_contract, runtime_resume_contract, validate_resume_runtime_compatibility,
    validate_rl_resume_runtime_compatibility, ResumeContext, RlResumeContext,
    RlRuntimeResumeContract,
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
    pub(super) bc_exit_cfg: BcExitConfig,
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
    pub(super) pipeline_state: PipelineState,
    pub(super) head_controller: HeadActivationController,
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

    let loader_config = StreamingLoaderConfig {
        buffer_games: config.buffer_games,
        buffer_samples: config.buffer_samples,
        train_fraction: config.train_fraction,
        seed: config.seed,
        archive_queue_bound: config.archive_queue_bound,
        max_skip_logs_per_source: config.max_skip_logs_per_source,
        aggregate_skip_logs: false,
        exit_sidecar,
        exit_sidecar_source_net_hash: None,
        exit_sidecar_source_version: None,
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
    let loader_config = StreamingLoaderConfig {
        exit_sidecar_source_net_hash: Some(exit_sidecar_source_net_hash),
        exit_sidecar_source_version: Some(1),
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
    let loss_fn = HydraLoss::<TrainBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let valid_loss_fn =
        HydraLoss::<ValidBackend>::new(build_loss_config(config.advanced_loss.as_ref())?);
    let bc_exit_cfg = build_bc_exit_config(config.advanced_loss.as_ref());
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
        },
    ))
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
        } else {
            println!(
                "{}",
                timestamped(format!(
                    "{} cache fingerprint mismatch, using config games_per_batch={}",
                    "RL preflight skip:".bold().yellow(),
                    rl_config.games_per_batch,
                ))
            );
        }
    }

    let current_runtime = rl_runtime_resume_contract(&rl_config);
    if let Some(state) = resume.state.as_ref() {
        validate_rl_resume_runtime_compatibility(state, current_runtime)?;
    }
    let train_device = train_device(&config.device);
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
            pipeline_state,
            head_controller,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RlPhaseConfig, RlTrainConfig};
    use std::fs;
    use std::path::PathBuf;

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "hydra_rl_bootstrap_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
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
            train_fraction: 0.9,
            augment: true,
            resume_checkpoint: None,
            seed: 7,
            advanced_loss: None,
            rl: Some(RlTrainConfig::default()),
            bc: Default::default(),
            device: "cpu".to_string(),
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
        fs::remove_dir_all(output_dir).ok();
    }

    #[test]
    fn rl_bootstrap_applies_preflight_cache_override() {
        use crate::artifacts::{write_preflight_cache, RlPreflightPaths};
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
        write_preflight_cache(
            &paths.cache_path,
            &PreflightCacheEntry {
                cache_key: key,
                runtime: EffectiveRuntimeConfig {
                    selected: SelectedRuntimeConfig {
                        train_microbatch_size: config.batch_size,
                        validation_microbatch_size: config
                            .validation_microbatch_size
                            .unwrap_or(config.batch_size),
                        accum_steps: 1,
                    },
                    loader: LoaderRuntimeConfig {
                        num_threads: config.num_threads,
                        buffer_games: tuned_games,
                        buffer_samples: config.buffer_samples,
                        archive_queue_bound: config.archive_queue_bound,
                    },
                },
            },
        )
        .expect("write cache");

        let (bootstrap, _runtime) =
            initialize_rl_training_bootstrap(&output_dir, config, rl_cfg).expect("rl bootstrap");

        assert_eq!(
            bootstrap.rl_config.games_per_batch, tuned_games,
            "bootstrap should apply preflight-cached games_per_batch"
        );
        fs::remove_dir_all(output_dir).ok();
    }

    #[test]
    fn rl_bootstrap_ignores_stale_preflight_cache() {
        use crate::artifacts::{write_preflight_cache, RlPreflightPaths};
        use hydra_train::preflight::{
            EffectiveRuntimeConfig, HardwareFingerprint, LoaderRuntimeConfig, PreflightCacheEntry,
            PreflightCacheKey, SelectedRuntimeConfig, WorkloadFingerprint,
        };

        let output_dir = unique_temp_dir("cache_stale");
        fs::create_dir_all(&output_dir).expect("create output dir");
        let config = dummy_rl_config(output_dir.clone());
        let rl_cfg = config.rl.clone().expect("rl config");
        let original_games = rl_cfg.games_per_batch;

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
                train_fraction_bits: 0,
                max_skip_logs_per_source: 0,
                max_validation_batches: None,
                max_validation_samples: None,
                model_signature: "stale".to_string(),
                code_signature: "stale".to_string(),
                advanced_loss_signature: "stale".to_string(),
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
            },
        )
        .expect("write stale cache");

        let (bootstrap, _runtime) =
            initialize_rl_training_bootstrap(&output_dir, config, rl_cfg).expect("rl bootstrap");

        assert_eq!(
            bootstrap.rl_config.games_per_batch, original_games,
            "bootstrap should ignore stale preflight cache"
        );
        fs::remove_dir_all(output_dir).ok();
    }
}
