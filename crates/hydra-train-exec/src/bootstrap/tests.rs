use super::*;
use crate::artifacts::{PreflightPaths, write_manifest_cache, write_preflight_cache};
use crate::resume::{
    build_resume_state, build_rl_resume_state, test_runtime_resume_contract, write_resume_state,
};
use hydra_data_core::{DataManifest, DataSource};
use hydra_train_runtime::config::{
    BcHyperparamConfig, PrecisionMode, RlPhaseConfig, SourceFilterConfig, ValidationGateConfig,
};
use hydra_train_runtime::preflight::{
    EffectiveRuntimeConfig, LoaderRuntimeConfig, ManifestCacheEntry, PreflightCacheEntry,
    PreflightConfig, SelectedRuntimeConfig, preflight_cache_key,
};
use hydra_train_types::phase::TrainingPhase;
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
    save_latest_optimizer_checkpoint(output_dir, HydraModelConfig::learner());
}

fn save_model_checkpoint_with_config(output_dir: &Path, model_config: HydraModelConfig) {
    let checkpoint_base = latest_model_checkpoint_path(output_dir).with_extension("");
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model_config
        .init::<TrainBackend>(&LibTorchDevice::Cpu)
        .save_file(&checkpoint_base, &recorder)
        .expect("save latest model checkpoint");
}

fn save_latest_optimizer_checkpoint(output_dir: &Path, model_config: HydraModelConfig) {
    let optimizer = BCTrainerConfig::new(model_config).optimizer_config().init();
    crate::artifacts::save_optimizer_payload::<TrainBackend, _>(
        &optimizer,
        &output_dir.join("latest_optimizer"),
    )
    .expect("save latest optimizer checkpoint");
}

fn tiny_test_model_config() -> HydraModelConfig {
    HydraModelConfig::new(1)
        .with_input_channels(hydra_core::encoder::NUM_CHANNELS)
        .with_hidden_channels(4)
        .with_num_groups(4)
        .with_se_bottleneck(1)
}

fn dummy_train_config(output_dir: PathBuf, data_dir: PathBuf) -> TrainConfig {
    TrainConfig {
        data_dir,
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
        source_filters: SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        seed: 0,
        advanced_loss: None,
        validation_gates: ValidationGateConfig::default(),
        rl: None,
        bc: BcHyperparamConfig::default(),
        nsight_trace: None,
        device: "cpu".to_string(),
        precision_mode: PrecisionMode::Fp32,
        buffer_games: 16,
        buffer_samples: 128,
        num_threads: None,
        tensorboard: false,
        archive_queue_bound: 8,
        validation_every_n_epochs: 1,
        max_skip_logs_per_source: 4,
        log_every_n_steps: 10,
        validate_every_n_steps: 10,
        checkpoint_every_n_steps: 10,
        max_train_steps: None,
        max_validation_batches: None,
        max_validation_samples: None,
        preflight: PreflightConfig::default(),
    }
}

fn tiny_real_mjai_replay() -> String {
    [
            r#"{"type":"start_game","names":["a","b","c","d"],"id":"game-1"}"#,
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
            r#"{"type":"tsumo","actor":0,"pai":"5p"}"#,
            r#"{"type":"dahai","actor":0,"pai":"1m","tsumogiri":false}"#,
            r#"{"type":"tsumo","actor":1,"pai":"E"}"#,
            r#"{"type":"dahai","actor":1,"pai":"E","tsumogiri":true}"#,
            r#"{"type":"ryukyoku","scores":[25000,25000,25000,25000]}"#,
            r#"{"type":"end_game"}"#,
        ]
        .join("\n")
            + "\n"
}

fn save_latest_tiny_model_checkpoint(output_dir: &Path) {
    save_model_checkpoint_with_config(output_dir, tiny_test_model_config());
}

fn write_bc_resume_state(
    output_dir: &Path,
    state: crate::resume::BcResumeState,
) -> std::path::PathBuf {
    let path = latest_state_path(output_dir);
    write_resume_state(&path, &state).expect("write resume state");
    path
}

fn write_rl_resume_state_for_test(
    output_dir: &Path,
    state: crate::resume::RlResumeState,
) -> std::path::PathBuf {
    let path = latest_state_path(output_dir);
    crate::resume::write_rl_resume_state(&path, &state).expect("write rl resume state");
    path
}

#[test]
fn initialize_rl_training_bootstrap_rejects_runtime_mismatch() {
    let root_dir = unique_temp_dir("runtime_mismatch");
    let output_dir = root_dir.join("output");
    let data_dir = root_dir.join("data");
    create_empty_dir(&output_dir);
    create_empty_dir(&data_dir);
    save_latest_model_checkpoint(&output_dir);
    let resume_checkpoint = latest_model_checkpoint_path(&output_dir);
    write_rl_resume_state_for_test(
        &output_dir,
        build_rl_resume_state(
            9,
            PipelineState::default(),
            RlRuntimeResumeContract {
                games_per_batch: 999,
                microbatch_size: 4,
                phase: RlPhaseConfig::DrdaAchSelfPlay,
                precision_mode: PrecisionMode::Fp32,
            },
        ),
    );

    let mut config = dummy_train_config(output_dir.clone(), data_dir);
    config.resume_checkpoint = Some(resume_checkpoint);
    let rl = RlTrainConfig {
        games_per_batch: 4,
        microbatch_size: Some(4),
        ..RlTrainConfig::default()
    };

    let err = initialize_rl_training_bootstrap(&output_dir, config, rl)
        .err()
        .expect("runtime mismatch should fail before model load");
    assert!(
        err.contains("RL resume runtime mismatch"),
        "unexpected error: {err}"
    );
    cleanup_dir(&root_dir);
}

#[test]
fn initialize_rl_training_bootstrap_restores_pipeline_state() {
    let root_dir = unique_temp_dir("restore_pipeline_state");
    let output_dir = root_dir.join("output");
    let data_dir = root_dir.join("data");
    create_empty_dir(&output_dir);
    create_empty_dir(&data_dir);
    save_latest_model_checkpoint(&output_dir);
    let resume_checkpoint = latest_model_checkpoint_path(&output_dir);
    let resumed = PipelineState {
        phase: TrainingPhase::ExitPondering,
        gpu_hours_used: 42.5,
        total_games: 7,
        total_samples: 11,
        learner_version: 2,
        actor_version: 3,
    };
    write_rl_resume_state_for_test(
        &output_dir,
        build_rl_resume_state(
            11,
            resumed,
            RlRuntimeResumeContract {
                games_per_batch: 4,
                microbatch_size: 4,
                phase: RlPhaseConfig::DrdaAchSelfPlay,
                precision_mode: PrecisionMode::Fp32,
            },
        ),
    );

    let mut config = dummy_train_config(output_dir.clone(), data_dir);
    config.resume_checkpoint = Some(resume_checkpoint);
    let rl = RlTrainConfig {
        games_per_batch: 4,
        microbatch_size: Some(4),
        ..RlTrainConfig::default()
    };

    let (bootstrap, runtime) = initialize_rl_training_bootstrap(&output_dir, config, rl)
        .expect("resume compatible rl bootstrap");
    assert_eq!(bootstrap.session_start_global_step, 11);
    assert_eq!(runtime.global_step, 11);
    assert_eq!(runtime.pipeline_state.phase, resumed.phase);
    assert_eq!(
        runtime.pipeline_state.gpu_hours_used,
        resumed.gpu_hours_used
    );
    assert_eq!(runtime.pipeline_state.total_games, resumed.total_games);
    assert_eq!(runtime.pipeline_state.total_samples, resumed.total_samples);
    assert_eq!(
        runtime.pipeline_state.learner_version,
        resumed.learner_version
    );
    assert_eq!(runtime.pipeline_state.actor_version, resumed.actor_version);
    cleanup_dir(&root_dir);
}

#[test]
fn initialize_rl_training_bootstrap_uses_config_phase_without_resume() {
    let root_dir = unique_temp_dir("config_phase_without_resume");
    let output_dir = root_dir.join("output");
    let data_dir = root_dir.join("data");
    create_empty_dir(&data_dir);

    let config = dummy_train_config(output_dir.clone(), data_dir);
    let rl = RlTrainConfig {
        phase: hydra_train_runtime::config::RlPhaseConfig::ExitPondering,
        games_per_batch: 4,
        microbatch_size: Some(4),
        ..RlTrainConfig::default()
    };

    let (bootstrap, runtime) =
        initialize_rl_training_bootstrap(&output_dir, config, rl).expect("fresh rl bootstrap");
    assert_eq!(bootstrap.session_start_global_step, 0);
    assert_eq!(
        runtime.pipeline_state.phase,
        bootstrap.rl_config.phase.to_training_phase()
    );
    cleanup_dir(&root_dir);
}

#[test]
fn initialize_training_bootstrap_reuses_manifest_cache() {
    let root_dir = unique_temp_dir("manifest_cache");
    let output_dir = root_dir.join("output");
    let data_dir = root_dir.join("data");
    let cache_file = data_dir.join("cached.jsonl");
    create_empty_dir(&data_dir);
    fs::write(&cache_file, tiny_real_mjai_replay()).expect("write tiny replay");
    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    artifacts.create_root_dir().expect("create artifacts");
    let preflight_paths = PreflightPaths::new(&artifacts);
    let manifest = DataManifest {
        sources: vec![DataSource::LooseFile(cache_file.clone())],
        total_games: 1,
        train_count: 1,
        val_count: 0,
        counts_exact: true,
    };
    write_manifest_cache(
        &preflight_paths.manifest_cache_path,
        &ManifestCacheEntry {
            data_dir: data_dir.clone(),
            train_fraction_bits: 0.9f32.to_bits(),
            include_source_patterns: Vec::new(),
            exclude_source_patterns: Vec::new(),
            manifest: manifest.clone(),
        },
    )
    .expect("write manifest cache");

    let mut config = dummy_train_config(output_dir.clone(), data_dir.clone());
    config.batch_size = 1;
    config.microbatch_size = Some(1);

    let (bootstrap, _, _) = initialize_training_bootstrap_for_backend_with_model_config::<
        TrainBackend,
    >(&output_dir, config, tiny_test_model_config())
    .expect("bootstrap with cached manifest");
    assert_eq!(bootstrap.manifest.sources, manifest.sources);
    cleanup_dir(&root_dir);
}

#[test]
fn initialize_training_bootstrap_rejects_resume_runtime_mismatch() {
    let root_dir = unique_temp_dir("bc_runtime_mismatch");
    let output_dir = root_dir.join("output");
    let data_dir = root_dir.join("data");
    create_empty_dir(&output_dir);
    create_empty_dir(&data_dir);
    fs::write(
        data_dir.join("runtime-mismatch.mjai.json"),
        tiny_real_mjai_replay(),
    )
    .expect("write tiny replay");
    save_latest_tiny_model_checkpoint(&output_dir);
    let resume_checkpoint = latest_model_checkpoint_path(&output_dir);
    write_bc_resume_state(
        &output_dir,
        build_resume_state(2, 0, 5, None, test_runtime_resume_contract(8, 4, 4)),
    );

    let mut config = dummy_train_config(output_dir.clone(), data_dir);
    config.batch_size = 4;
    config.microbatch_size = Some(4);
    config.validation_microbatch_size = Some(4);
    config.resume_checkpoint = Some(resume_checkpoint);

    let err = initialize_training_bootstrap_for_backend_with_model_config::<TrainBackend>(
        &output_dir,
        config,
        tiny_test_model_config(),
    )
    .err()
    .expect("resume batch mismatch should fail before model load");
    assert!(
        err.contains("resume batch_size mismatch"),
        "unexpected error: {err}"
    );
    cleanup_dir(&root_dir);
}

fn write_matching_bc_preflight_cache(
    config: &TrainConfig,
    artifacts: &BcArtifactPaths,
    model_config: &HydraModelConfig,
    runtime: EffectiveRuntimeConfig,
) {
    let paths = PreflightPaths::new(artifacts);
    write_preflight_cache(
        &paths.cache_path,
        &PreflightCacheEntry {
            cache_key: preflight_cache_key(
                config,
                &hydra_train_runtime::preflight::ModelFingerprintInput {
                    num_blocks: model_config.num_blocks,
                    input_channels: model_config.input_channels,
                    hidden_channels: model_config.hidden_channels,
                    num_groups: model_config.num_groups,
                    action_space: model_config.action_space,
                    score_bins: model_config.score_bins,
                },
                &config.device,
                hydra_train_runtime::config::default_num_threads_for_system(),
            ),
            runtime,
            benchmark: None,
        },
    )
    .expect("write matching preflight cache");
}

#[test]
fn initialize_training_bootstrap_applies_cached_loader_tuple_on_fresh_start() {
    let root_dir = unique_temp_dir("bc_loader_cache_fresh");
    let output_dir = root_dir.join("output");
    let data_dir = root_dir.join("data");
    let cache_file = data_dir.join("cached.mjai.json");
    create_empty_dir(&data_dir);
    fs::write(&cache_file, tiny_real_mjai_replay()).expect("write tiny replay");

    let mut config = dummy_train_config(output_dir.clone(), data_dir.clone());
    config.batch_size = 128;
    config.microbatch_size = Some(64);
    config.validation_microbatch_size = Some(32);
    config.buffer_games = 16;
    config.buffer_samples = 128;
    config.archive_queue_bound = 8;
    config.num_threads = Some(1);

    let artifacts = BcArtifactPaths::new(&output_dir, 0);
    artifacts.create_root_dir().expect("create artifact root");
    let model_config = tiny_test_model_config();
    write_matching_bc_preflight_cache(
        &config,
        &artifacts,
        &model_config,
        EffectiveRuntimeConfig {
            selected: SelectedRuntimeConfig {
                train_microbatch_size: 32,
                validation_microbatch_size: 16,
                accum_steps: 4,
                unsafe_selected_batch_size: None,
                unsafe_selected_learning_rate: None,
                unsafe_selected_min_learning_rate: None,
                unsafe_selected_warmup_steps: None,
            },
            loader: LoaderRuntimeConfig {
                num_threads: Some(2),
                buffer_games: 7,
                buffer_samples: 64,
                archive_queue_bound: 3,
            },
        },
    );

    let (bootstrap, _, _) = initialize_training_bootstrap_for_backend_with_model_config::<
        TrainBackend,
    >(&output_dir, config, model_config)
    .expect("fresh bootstrap should apply cached safe runtime");

    assert_eq!(bootstrap.config.batch_size, 128);
    assert_eq!(bootstrap.config.microbatch_size, Some(32));
    assert_eq!(bootstrap.config.validation_microbatch_size, Some(16));
    assert_eq!(bootstrap.config.num_threads, Some(2));
    assert_eq!(bootstrap.config.buffer_games, 7);
    assert_eq!(bootstrap.config.buffer_samples, 64);
    assert_eq!(bootstrap.config.archive_queue_bound, 3);
    assert_eq!(bootstrap.loader_config.num_threads, Some(2));
    assert_eq!(bootstrap.loader_config.buffer_games, 7);
    assert_eq!(bootstrap.loader_config.buffer_samples, 64);
    assert_eq!(bootstrap.loader_config.archive_queue_bound, 3);
    cleanup_dir(&root_dir);
}

#[test]
fn initialize_training_bootstrap_applies_cached_loader_tuple_on_epoch_boundary_resume() {
    let root_dir = unique_temp_dir("bc_loader_cache_epoch_resume");
    let output_dir = root_dir.join("output");
    let data_dir = root_dir.join("data");
    let cache_file = data_dir.join("cached.mjai.json");
    create_empty_dir(&output_dir);
    create_empty_dir(&data_dir);
    fs::write(&cache_file, tiny_real_mjai_replay()).expect("write tiny replay");
    save_latest_tiny_model_checkpoint(&output_dir);
    save_latest_optimizer_checkpoint(&output_dir, tiny_test_model_config());
    let resume_checkpoint = latest_model_checkpoint_path(&output_dir);
    write_bc_resume_state(
        &output_dir,
        build_resume_state(1, 0, 7, None, test_runtime_resume_contract(128, 64, 32)),
    );

    let mut config = dummy_train_config(output_dir.clone(), data_dir.clone());
    config.batch_size = 128;
    config.microbatch_size = Some(64);
    config.validation_microbatch_size = Some(32);
    config.buffer_games = 16;
    config.buffer_samples = 128;
    config.archive_queue_bound = 8;
    config.num_threads = Some(1);
    config.resume_checkpoint = Some(resume_checkpoint);

    let artifacts = BcArtifactPaths::new(&output_dir, 7);
    artifacts.create_root_dir().expect("create artifact root");
    let model_config = tiny_test_model_config();
    write_matching_bc_preflight_cache(
        &config,
        &artifacts,
        &model_config,
        EffectiveRuntimeConfig {
            selected: SelectedRuntimeConfig {
                train_microbatch_size: 64,
                validation_microbatch_size: 32,
                accum_steps: 2,
                unsafe_selected_batch_size: None,
                unsafe_selected_learning_rate: None,
                unsafe_selected_min_learning_rate: None,
                unsafe_selected_warmup_steps: None,
            },
            loader: LoaderRuntimeConfig {
                num_threads: Some(2),
                buffer_games: 9,
                buffer_samples: 96,
                archive_queue_bound: 4,
            },
        },
    );

    let (bootstrap, _, _) = initialize_training_bootstrap_for_backend_with_model_config::<
        TrainBackend,
    >(&output_dir, config, model_config)
    .expect("epoch-boundary resume should apply cached safe loader runtime");

    assert_eq!(bootstrap.config.batch_size, 128);
    assert_eq!(bootstrap.config.microbatch_size, Some(64));
    assert_eq!(bootstrap.config.validation_microbatch_size, Some(32));
    assert_eq!(bootstrap.config.num_threads, Some(2));
    assert_eq!(bootstrap.config.buffer_games, 9);
    assert_eq!(bootstrap.config.buffer_samples, 96);
    assert_eq!(bootstrap.config.archive_queue_bound, 4);
    assert_eq!(bootstrap.loader_config.num_threads, Some(2));
    assert_eq!(bootstrap.loader_config.buffer_games, 9);
    assert_eq!(bootstrap.loader_config.buffer_samples, 96);
    assert_eq!(bootstrap.loader_config.archive_queue_bound, 4);
    cleanup_dir(&root_dir);
}

#[test]
fn initialize_training_bootstrap_keeps_loader_tuple_on_partial_resume() {
    let root_dir = unique_temp_dir("bc_loader_cache_partial_resume");
    let output_dir = root_dir.join("output");
    let data_dir = root_dir.join("data");
    let cache_file = data_dir.join("cached.mjai.json");
    create_empty_dir(&output_dir);
    create_empty_dir(&data_dir);
    fs::write(&cache_file, tiny_real_mjai_replay()).expect("write tiny replay");
    save_latest_tiny_model_checkpoint(&output_dir);
    save_latest_optimizer_checkpoint(&output_dir, tiny_test_model_config());
    let resume_checkpoint = latest_model_checkpoint_path(&output_dir);
    write_bc_resume_state(
        &output_dir,
        build_resume_state(1, 3, 7, None, test_runtime_resume_contract(128, 64, 32)),
    );

    let mut config = dummy_train_config(output_dir.clone(), data_dir.clone());
    config.batch_size = 128;
    config.microbatch_size = Some(64);
    config.validation_microbatch_size = Some(32);
    config.buffer_games = 16;
    config.buffer_samples = 128;
    config.archive_queue_bound = 8;
    config.num_threads = Some(1);
    config.resume_checkpoint = Some(resume_checkpoint);

    let artifacts = BcArtifactPaths::new(&output_dir, 7);
    artifacts.create_root_dir().expect("create artifact root");
    let model_config = tiny_test_model_config();
    write_matching_bc_preflight_cache(
        &config,
        &artifacts,
        &model_config,
        EffectiveRuntimeConfig {
            selected: SelectedRuntimeConfig {
                train_microbatch_size: 32,
                validation_microbatch_size: 16,
                accum_steps: 4,
                unsafe_selected_batch_size: None,
                unsafe_selected_learning_rate: None,
                unsafe_selected_min_learning_rate: None,
                unsafe_selected_warmup_steps: None,
            },
            loader: LoaderRuntimeConfig {
                num_threads: Some(2),
                buffer_games: 9,
                buffer_samples: 96,
                archive_queue_bound: 4,
            },
        },
    );

    let (bootstrap, _, _) = initialize_training_bootstrap_for_backend_with_model_config::<
        TrainBackend,
    >(&output_dir, config, model_config)
    .expect("partial resume should ignore cached safe runtime");

    assert_eq!(bootstrap.config.batch_size, 128);
    assert_eq!(bootstrap.config.microbatch_size, Some(64));
    assert_eq!(bootstrap.config.validation_microbatch_size, Some(32));
    assert_eq!(bootstrap.config.num_threads, Some(1));
    assert_eq!(bootstrap.config.buffer_games, 16);
    assert_eq!(bootstrap.config.buffer_samples, 128);
    assert_eq!(bootstrap.config.archive_queue_bound, 8);
    assert_eq!(bootstrap.loader_config.num_threads, Some(1));
    assert_eq!(bootstrap.loader_config.buffer_games, 16);
    assert_eq!(bootstrap.loader_config.buffer_samples, 128);
    assert_eq!(bootstrap.loader_config.archive_queue_bound, 8);
    cleanup_dir(&root_dir);
}
