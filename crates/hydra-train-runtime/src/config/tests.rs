use super::*;
use crate::config_runtime::validate_rl_config;
use std::path::PathBuf;

#[test]
fn parse_pf_repetitions_aliases_required_successes() {
    let cli = parse_args(vec![
        "train".to_string(),
        "--preflight".to_string(),
        "--pf-repetitions".to_string(),
        "5".to_string(),
        "--pf-candidate-tuples".to_string(),
        "1024:2:1:1,2048:4:2:2".to_string(),
    ])
    .expect("pf repetitions should parse");
    let preflight = cli.preflight.expect("preflight options should be present");
    assert_eq!(preflight.preflight_config.required_successes, 5);
    assert_eq!(preflight.preflight_config.bench_candidate_tuples.len(), 2);
}

#[test]
fn python_residual_profiles_serde_and_string_contract_match() {
    let cases = [
        ("mish_se", PythonResidualProfileConfig::MishSe),
        ("silu_se", PythonResidualProfileConfig::SiluSe),
        ("relu_se", PythonResidualProfileConfig::ReluSe),
        ("mish_no_se", PythonResidualProfileConfig::MishNoSe),
        ("mish_eca", PythonResidualProfileConfig::MishEca),
        ("relu_no_se", PythonResidualProfileConfig::ReluNoSe),
        (
            "relu_no_norm_no_se",
            PythonResidualProfileConfig::ReluNoNormNoSe,
        ),
    ];
    assert_eq!(
        PythonResidualProfileConfig::default(),
        PythonResidualProfileConfig::MishSe
    );
    for (text, profile) in cases {
        let parsed: PythonResidualProfileConfig =
            serde_yaml::from_str(text).expect("profile should deserialize");
        assert_eq!(parsed, profile);
        assert_eq!(profile.as_str(), text);
    }
}

#[test]
fn python_model_profiles_serde_and_dimensions_match() {
    let cases = [
        ("default", PythonModelProfileConfig::Default, 256, 10, 64),
        ("balanced", PythonModelProfileConfig::Balanced, 256, 12, 64),
        ("large", PythonModelProfileConfig::Large, 384, 16, 96),
    ];
    assert_eq!(
        PythonModelProfileConfig::default(),
        PythonModelProfileConfig::Large
    );
    for (text, profile, hidden, blocks, bottleneck) in cases {
        let parsed: PythonModelProfileConfig =
            serde_yaml::from_str(text).expect("profile should deserialize");
        assert_eq!(parsed, profile);
        assert_eq!(profile.hidden(), hidden);
        assert_eq!(profile.blocks(), blocks);
        assert_eq!(profile.bottleneck(), bottleneck);
    }
}

#[test]
fn python_backbone_profiles_serde_and_string_contract_match() {
    let cases = [
        ("conv2d_local3", PythonBackboneProfileConfig::Conv2dLocal3),
        (
            "tileformer_bias",
            PythonBackboneProfileConfig::TileformerBias,
        ),
        (
            "convnext_tile_k7",
            PythonBackboneProfileConfig::ConvnextTileK7,
        ),
        (
            "global_pool_bias",
            PythonBackboneProfileConfig::GlobalPoolBias,
        ),
    ];
    assert_eq!(
        PythonBackboneProfileConfig::default(),
        PythonBackboneProfileConfig::Conv2dLocal3
    );
    for (text, profile) in cases {
        let parsed: PythonBackboneProfileConfig =
            serde_yaml::from_str(text).expect("profile should deserialize");
        assert_eq!(parsed, profile);
        assert_eq!(profile.as_str(), text);
    }
}

#[test]
fn python_variants_serde_and_string_contract_match() {
    let cases = [
        ("eager_fp32", PythonLearnerVariant::EagerFp32),
        ("eager_bf16", PythonLearnerVariant::EagerBf16),
        ("compile_default", PythonLearnerVariant::CompileDefault),
        (
            "compile_reduce_overhead",
            PythonLearnerVariant::CompileReduceOverhead,
        ),
        (
            "compile_max_autotune",
            PythonLearnerVariant::CompileMaxAutotune,
        ),
    ];
    assert_eq!(
        PythonLearnerVariant::default(),
        PythonLearnerVariant::CompileMaxAutotune
    );
    for (text, variant) in cases {
        let parsed: PythonLearnerVariant =
            serde_yaml::from_str(text).expect("variant should deserialize");
        assert_eq!(parsed, variant);
        assert_eq!(variant.as_str(), text);
    }
}

#[test]
fn python_raw_mjai_transport_serde_and_default_match() {
    let cases = [
        ("pinned_pyo3", PythonRawMjaiTransportConfig::PinnedPyo3),
        ("stdout", PythonRawMjaiTransportConfig::Stdout),
    ];
    assert_eq!(
        PythonRawMjaiTransportConfig::default(),
        PythonRawMjaiTransportConfig::PinnedPyo3
    );
    for (text, transport) in cases {
        let parsed: PythonRawMjaiTransportConfig =
            serde_yaml::from_str(text).expect("transport should deserialize");
        assert_eq!(parsed, transport);
        assert_eq!(transport.as_str(), text);
    }
}

#[test]
fn ema_config_defaults_on_and_validates() {
    let config: TrainConfig = serde_yaml::from_str(
        r#"
data_dir: data
output_dir: out
num_epochs: 1
ema:
  enabled: true
  decay: 0.99
  start_step: 5
  update_every_steps: 2
  device: cuda
"#,
    )
    .expect("EMA config should parse");
    assert!(config.ema.enabled);
    assert_eq!(config.ema.decay, 0.99);
    assert_eq!(config.ema.start_step, 5);
    assert_eq!(config.ema.update_every_steps, 2);
    assert_eq!(config.ema.device, EmaDeviceConfig::Cuda);
    let default_config: TrainConfig = serde_yaml::from_str(
        r#"
data_dir: data
output_dir: out
num_epochs: 1
"#,
    )
    .expect("default config should parse");
    assert!(default_config.ema.enabled);
    assert_eq!(default_config.ema.device, EmaDeviceConfig::Auto);

    let cpu: EmaDeviceConfig = serde_yaml::from_str("cpu").expect("EMA device should deserialize");
    assert_eq!(cpu.as_str(), "cpu");
}

#[test]
fn repository_example_config_parses_with_ema_device_default_on() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .expect("crate should live under crates/")
        .to_path_buf();
    let config = read_config(&repo_root.join("example.yaml")).expect("example config should parse");
    assert!(config.ema.enabled);
    assert_eq!(config.ema.device, EmaDeviceConfig::Auto);
}

#[test]
fn ema_config_validation_rejects_invalid_values() {
    let mut config = TrainConfig::default_preflight_bench();
    config.num_epochs = 1;
    config.ema.enabled = true;
    config.ema.decay = 1.0;
    let err = validate_config(&config).expect_err("invalid EMA decay should fail");
    assert!(err.contains("ema.decay"));
    config.ema.decay = 0.9;
    config.ema.update_every_steps = 0;
    let err = validate_config(&config).expect_err("invalid EMA cadence should fail");
    assert!(err.contains("ema.update_every_steps"));
}

#[test]
fn usage_lists_all_probe_kinds() {
    let text = usage("train");
    assert!(text.contains("--probe-kind <train|validation|rl_games|rl_microbatch>"));
}

#[test]
fn parse_args_rejects_partial_probe_flags() {
    let args = vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--probe-kind".to_string(),
        "train".to_string(),
    ];
    let err = parse_args(args).expect_err("partial probe args should fail");
    assert!(
        err.contains("probe-only mode requires both --probe-kind and --probe-candidate-microbatch")
    );
}

#[test]
fn parse_args_accepts_rl_probe_kinds_advertised_in_usage() {
    for kind in ["rl_games", "rl_microbatch"] {
        let args = vec![
            "train".to_string(),
            "config.yaml".to_string(),
            "--probe-kind".to_string(),
            kind.to_string(),
            "--probe-candidate-microbatch".to_string(),
            "16".to_string(),
        ];
        let cli = parse_args(args).expect("advertised probe kind should parse");
        assert!(cli.probe_only.is_some());
    }
}

#[test]
fn parse_args_accepts_benchmark_baseline_without_config() {
    let cli = parse_args(vec![
        "train".to_string(),
        "--benchmark-baseline".to_string(),
        "--data-dir".to_string(),
        "data/mjai".to_string(),
        "--output-dir".to_string(),
        "out/bench".to_string(),
        "--device".to_string(),
        "cuda:0".to_string(),
        "--bench-max-games".to_string(),
        "123".to_string(),
        "--bench-steps".to_string(),
        "7".to_string(),
    ])
    .expect("benchmark mode should parse");
    let benchmark = cli
        .benchmark_baseline
        .expect("benchmark options should be present");
    assert_eq!(
        benchmark.data_dir,
        Some(std::path::PathBuf::from("data/mjai"))
    );
    assert_eq!(benchmark.output_dir, std::path::PathBuf::from("out/bench"));
    assert_eq!(benchmark.device, "cuda:0");
    assert_eq!(benchmark.max_games, 123);
    assert_eq!(benchmark.max_train_steps, 7);
    assert_eq!(benchmark.train_threads, 8);
    assert_eq!(benchmark.source, BenchmarkBaselineSource::Both);
    assert!(cli.config_path.is_none());
    assert!(cli.preflight.is_none());
}

#[test]
fn parse_args_accepts_experimental_burn_cuda_backend() {
    let cli = parse_args(vec![
        "train".to_string(),
        "--benchmark-baseline".to_string(),
        "--bench-source".to_string(),
        "bc-shards".to_string(),
        "--bc-shards-manifest".to_string(),
        "shards/manifest.json".to_string(),
        "--experimental-backend".to_string(),
        "burn-cuda".to_string(),
    ])
    .expect("experimental backend should parse");

    assert_eq!(cli.experimental_backend, ExperimentalTrainBackend::BurnCuda);
    assert_eq!(
        cli.benchmark_baseline
            .expect("benchmark options should be present")
            .experimental_backend,
        ExperimentalTrainBackend::BurnCuda
    );
}

#[test]
fn parse_args_accepts_experimental_backbone_profile() {
    let cli = parse_args(vec![
        "train".to_string(),
        "--benchmark-baseline".to_string(),
        "--bench-source".to_string(),
        "bc-shards".to_string(),
        "--bc-shards-manifest".to_string(),
        "shards/manifest.json".to_string(),
        "--experimental-backbone-profile".to_string(),
        "activation=relu,se_every_n=4,norm=first_only,blocks=12,hidden=128".to_string(),
    ])
    .expect("experimental backbone profile should parse");

    let profile = cli
        .benchmark_baseline
        .expect("benchmark options should be present")
        .experimental_backbone_profile
        .expect("profile should be present");
    assert_eq!(profile.activation, BackboneActivationConfig::Relu);
    assert_eq!(profile.se_every_n, 4);
    assert_eq!(profile.norm, BackboneNormConfig::FirstOnly);
    assert_eq!(profile.num_blocks, Some(12));
    assert_eq!(profile.hidden_channels, Some(128));
}

#[test]
fn parse_args_rejects_invalid_experimental_backbone_profile() {
    let err = parse_args(vec![
        "train".to_string(),
        "--benchmark-baseline".to_string(),
        "--bench-source".to_string(),
        "bc-shards".to_string(),
        "--bc-shards-manifest".to_string(),
        "shards/manifest.json".to_string(),
        "--experimental-backbone-profile".to_string(),
        "activation=gelu".to_string(),
    ])
    .expect_err("invalid backbone profile should fail");

    assert!(err.contains("unsupported backbone activation"));
}

#[test]
fn parse_args_rejects_benchmark_config_path() {
    let err = parse_args(vec![
        "train".to_string(),
        "config.yaml".to_string(),
        "--benchmark-baseline".to_string(),
        "--data-dir".to_string(),
        "data/mjai".to_string(),
    ])
    .expect_err("benchmark mode should reject config paths");
    assert!(err.contains("--benchmark-baseline does not accept a config path"));
}

#[test]
fn parse_args_rejects_data_dir_without_benchmark() {
    let err = parse_args(vec![
        "train".to_string(),
        "--data-dir".to_string(),
        "data/mjai".to_string(),
    ])
    .expect_err("data-dir should be benchmark-only");
    assert!(err.contains("--data-dir/--bc-shards-manifest requires --benchmark-baseline"));
}

#[test]
fn parse_args_rejects_benchmark_with_preflight() {
    let err = parse_args(vec![
        "train".to_string(),
        "--benchmark-baseline".to_string(),
        "--preflight".to_string(),
        "--data-dir".to_string(),
        "data/mjai".to_string(),
    ])
    .expect_err("benchmark and preflight should be exclusive");
    assert!(err.contains("--benchmark-baseline cannot be combined with --preflight"));
}

#[test]
fn parse_args_accepts_benchmark_bc_shards_source() {
    let cli = parse_args(vec![
        "train".to_string(),
        "--benchmark-baseline".to_string(),
        "--bench-source".to_string(),
        "bc-shards".to_string(),
        "--bc-shards-manifest".to_string(),
        "out/shards/manifest.json".to_string(),
    ])
    .expect("bc shard benchmark mode should parse");
    let benchmark = cli
        .benchmark_baseline
        .expect("benchmark options should be present");
    assert_eq!(benchmark.source, BenchmarkBaselineSource::BcShards);
    assert_eq!(
        benchmark.bc_shards_manifest_path,
        Some(std::path::PathBuf::from("out/shards/manifest.json"))
    );
    assert!(benchmark.data_dir.is_none());
}

#[test]
fn parse_args_rejects_benchmark_bc_shards_without_manifest() {
    let err = parse_args(vec![
        "train".to_string(),
        "--benchmark-baseline".to_string(),
        "--bench-source".to_string(),
        "bc-shards".to_string(),
    ])
    .expect_err("bc shard benchmark needs manifest");
    assert!(err.contains("--bench-source bc-shards requires --bc-shards-manifest <path>"));
}

#[test]
fn parse_args_accepts_explicit_python_learner_alias() {
    let cli = parse_args(vec![
        "train".to_string(),
        "--experimental-python-learner".to_string(),
        "--bc-shards-manifest".to_string(),
        "output/shards/manifest.json".to_string(),
        "--output-dir".to_string(),
        "out/python".to_string(),
        "--device".to_string(),
        "cuda:0".to_string(),
        "--python-variant".to_string(),
        "compile_default".to_string(),
        "--python-warmup".to_string(),
        "1".to_string(),
        "--python-steps".to_string(),
        "3".to_string(),
        "--python-compile-fullgraph-check".to_string(),
        "--python-residual-profile".to_string(),
        "relu_no_norm_no_se".to_string(),
    ])
    .expect("explicit python learner mode should parse");
    let python = cli
        .python_learner
        .expect("python options should be present");
    assert_eq!(
        python.bc_shards_manifest,
        std::path::PathBuf::from("output/shards/manifest.json")
    );
    assert_eq!(python.output_dir, std::path::PathBuf::from("out/python"));
    assert_eq!(python.device, "cuda:0");
    assert_eq!(python.variant, PythonLearnerVariant::CompileDefault);
    assert_eq!(python.warmup_steps, 1);
    assert_eq!(python.steps, Some(3));
    assert!(python.compile_fullgraph_check);
    assert_eq!(
        python.residual_profile,
        PythonResidualProfileConfig::ReluNoNormNoSe
    );
    assert!(cli.config_path.is_none());
    assert!(cli.benchmark_baseline.is_none());
    assert_eq!(cli.bc_backend, BcBackend::Python);
}

#[test]
fn parse_args_accepts_bc_shards_manifest_as_python_default() {
    let cli = parse_args(vec![
        "train".to_string(),
        "--bc-shards-manifest".to_string(),
        "output/shards/manifest.json".to_string(),
        "--output-dir".to_string(),
        "out/python".to_string(),
        "--python-variant".to_string(),
        "compile_default".to_string(),
    ])
    .expect("bc shard manifest should default to python learner");
    let python = cli
        .python_learner
        .expect("python options should be present");
    assert_eq!(cli.bc_backend, BcBackend::Python);
    assert_eq!(
        python.bc_shards_manifest,
        std::path::PathBuf::from("output/shards/manifest.json")
    );
    assert_eq!(python.variant, PythonLearnerVariant::CompileDefault);
}

#[test]
fn parse_args_bc_shards_manifest_defaults_to_production_python_variant() {
    let cli = parse_args(vec![
        "train".to_string(),
        "--bc-shards-manifest".to_string(),
        "output/shards/manifest.json".to_string(),
        "--output-dir".to_string(),
        "out/python".to_string(),
    ])
    .expect("bc shard manifest should default to python learner");
    let python = cli
        .python_learner
        .expect("python options should be present");
    assert_eq!(python.variant, PythonLearnerVariant::CompileMaxAutotune);
}

#[test]
fn parse_args_explicit_rust_legacy_backend_uses_old_path() {
    let cli = parse_args(vec![
        "train".to_string(),
        "--bc-shards-manifest".to_string(),
        "output/shards/manifest.json".to_string(),
        "--bc-backend".to_string(),
        "rust-burn".to_string(),
    ])
    .expect("explicit rust backend should parse");
    assert_eq!(cli.bc_backend, BcBackend::RustBurn);
    assert!(cli.python_learner.is_none());
    assert!(cli.benchmark_baseline.is_none());
}

#[test]
fn parse_args_rejects_python_learner_without_manifest() {
    let err = parse_args(vec![
        "train".to_string(),
        "--experimental-python-learner".to_string(),
        "--output-dir".to_string(),
        "out/python".to_string(),
    ])
    .expect_err("python learner requires manifest");
    assert!(err.contains("requires --bc-shards-manifest"));
}

#[test]
fn repository_example_config_matches_train_config_contract() {
    let text =
        std::fs::read_to_string("../../example.yaml").expect("example config should be readable");
    let config: TrainConfig = serde_yaml::from_str(&text).expect("example config should parse");

    assert!(config.full_epoch);
    assert_eq!(config.max_train_steps, None);
    assert_eq!(config.python_model_profile, PythonModelProfileConfig::Large);
    assert_eq!(
        config.python_backbone_profile,
        PythonBackboneProfileConfig::Conv2dLocal3
    );
    assert_eq!(
        config.python_residual_profile,
        PythonResidualProfileConfig::MishSe
    );
    assert_eq!(config.batch_size, 3072);
    assert_eq!(config.python_raw_mjai_target_games, Some(6_000_000));
    assert_eq!(config.python_raw_mjai_estimated_samples_per_game, None);
    assert!(config.resume_latest);
    assert!(config.ema.enabled);
    assert_eq!(config.ema.decay, 0.999);
    assert_eq!(config.ema.update_every_steps, 1);
    assert_eq!(config.bc.grad_clip_norm, 1.0);
    assert_eq!(config.max_validation_samples, Some(524_288));
}

#[test]
fn documented_minimal_rl_config_parses_and_validates() {
    let text = r#"data_dir: /data
output_dir: /out
num_epochs: 1
device: cpu
rl:
  games_per_batch: 1024
  temperature: 1.0
  phase: drda_ach_self_play
  microbatch_size: 1024
"#;
    let config: TrainConfig = serde_yaml::from_str(text).expect("documented RL YAML should parse");
    let rl = config.rl.expect("rl config should be present");
    assert_eq!(rl.games_per_batch, 1024);
    assert_eq!(rl.temperature, 1.0);
    assert_eq!(rl.phase, RlPhaseConfig::DrdaAchSelfPlay);
    assert_eq!(rl.microbatch_size, Some(1024));
    validate_rl_config(&rl).expect("documented RL YAML should validate");
}

#[test]
fn documented_removed_rl_field_is_rejected() {
    let text = r#"data_dir: /data
output_dir: /out
num_epochs: 1
rl:
  games_per_batch: 4
  decisions_per_game: 1024
"#;
    let err =
        serde_yaml::from_str::<TrainConfig>(text).expect_err("unsupported RL field should fail");
    assert!(err.to_string().contains("decisions_per_game"), "{err}");
}

#[test]
fn config_accepts_explicit_raw_mjai_data_dirs() {
    let text = r#"data_dir: /fallback
raw_mjai_data_dirs:
  - /dataset/a
  - /dataset/b
output_dir: /out
num_epochs: 1
"#;
    let config: TrainConfig = serde_yaml::from_str(text).expect("config should parse");

    assert_eq!(
        config.raw_mjai_data_dirs,
        vec![PathBuf::from("/dataset/a"), PathBuf::from("/dataset/b")]
    );
}

fn python_guard_config() -> TrainConfig {
    TrainConfig {
        data_dir: PathBuf::from("/tmp/data"),
        raw_mjai_data_dirs: Vec::new(),
        output_dir: PathBuf::from("/tmp/out"),
        stage: None,
        run_name: None,
        num_epochs: 1,
        batch_size: 1024,
        microbatch_size: Some(1024),
        validation_microbatch_size: Some(1024),
        exit_sidecar_path: None,
        delta_q_sidecar_path: None,
        bc_shards_manifest_path: Some(PathBuf::from("/tmp/shards/manifest.json")),
        bc_backend: Default::default(),
        shard_prefetch_depth: None,
        train_fraction: 0.9,
        source_filters: SourceFilterConfig::default(),
        augment: true,
        resume_checkpoint: None,
        resume_latest: false,
        seed: 0,
        advanced_loss: None,
        python_residual_profile: Default::default(),
        python_variant: Default::default(),
        python_model_profile: Default::default(),
        python_backbone_profile: Default::default(),
        python_conv_memory_format: Default::default(),
        bc_head_profile: BcHeadProfile::default(),
        experimental_backbone_profile: None,
        python_raw_mjai_transport: Default::default(),
        python_raw_mjai_target_games: None,
        python_raw_mjai_estimated_samples_per_game: None,
        validation_gates: ValidationGateConfig::default(),
        ema: EmaConfig::default(),
        rl: None,
        bc: BcHyperparamConfig::default(),
        nsight_trace: None,
        device: "cuda:0".to_string(),
        precision_mode: PrecisionMode::Bf16Autocast,
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
        keep_step_checkpoints: false,
        launch_tensorboard: false,
        tensorboard_host: "127.0.0.1".to_string(),
        tensorboard_port: 6006,
        background: false,
        max_train_steps: Some(3),
        full_epoch: false,
        max_validation_batches: None,
        max_validation_samples: None,
    }
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "hydra_train_runtime_{label}_{}_{}",
        std::process::id(),
        nanos
    ))
}

#[test]
fn python_options_from_config_accepts_plain_bc_defaults() {
    let options = python_options_from_config(&python_guard_config())
        .expect("plain BC should route to Python");
    assert_eq!(
        options.bc_shards_manifest,
        PathBuf::from("/tmp/shards/manifest.json")
    );
    assert_eq!(options.batch_size, 1024);
    assert_eq!(options.microbatch_size, 1024);
    assert_eq!(options.steps, Some(3));
    assert_eq!(options.warmup_steps, PYTHON_TIMING_WARMUP_STEPS);
    assert_eq!(options.residual_profile, Default::default());
    assert!(!options.raw_mjai_validation_augment);
    assert_eq!(options.validation_source_mode, "fixed");
    assert_eq!(options.lr_schedule, "cosine");
    assert_eq!(options.schedule_total_steps, Some(3));
    assert_eq!(options.schedule_target_games, None);
    assert_eq!(options.validation_steps, 0);
    assert_eq!(options.validation_max_samples, None);
    assert_eq!(options.ema_device, Default::default());
    assert_eq!(options.exit_weight, 0.0);
    assert_eq!(options.deltaq_weight, 0.0);
}

#[test]
fn python_options_from_config_maps_campaign_root_to_default_bc_run_dir() {
    let config = python_guard_config();

    let options = python_options_from_config(&config).expect("plain BC should route to Python");

    assert_eq!(
        options.output_dir,
        PathBuf::from("/tmp/out/stages/bc_baseline/runs/latest_run")
    );
}

#[test]
fn python_options_from_config_uses_explicit_stage_and_run_name() {
    let mut config = python_guard_config();
    config.stage = Some("bc_custom".to_string());
    config.run_name = Some("run_001".to_string());

    let options = python_options_from_config(&config).expect("plain BC should route to Python");

    assert_eq!(
        options.output_dir,
        PathBuf::from("/tmp/out/stages/bc_custom/runs/run_001")
    );
}

#[test]
fn python_resume_checkpoint_raw_mjai_allows_explicit_resume() {
    let mut config = python_guard_config();
    config.bc_shards_manifest_path = None;
    let checkpoint = PathBuf::from("/tmp/out/checkpoints/latest.pt");
    config.resume_checkpoint = Some(checkpoint.clone());

    let options =
        python_options_from_config(&config).expect("raw MJAI explicit resume should work");
    assert_eq!(options.resume, Some(checkpoint));
}

#[test]
fn python_resume_checkpoint_raw_mjai_resume_latest_without_checkpoint_is_none() {
    let mut config = python_guard_config();
    config.bc_shards_manifest_path = None;
    config.resume_latest = true;

    let options =
        python_options_from_config(&config).expect("raw MJAI resume_latest should be allowed");
    assert_eq!(options.resume, None);
}

#[test]
fn python_resume_checkpoint_raw_mjai_uses_occupied_latest_by_default() {
    let root = unique_temp_dir("raw-mjai-occupied");
    let checkpoint_dir = root
        .join("stages")
        .join("bc_baseline")
        .join("runs")
        .join("latest_run")
        .join("checkpoints");
    std::fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir should be created");
    let latest = checkpoint_dir.join("latest.pt");
    std::fs::write(&latest, b"checkpoint").expect("latest checkpoint should write");
    let mut config = python_guard_config();
    config.bc_shards_manifest_path = None;
    config.output_dir = root;
    config.resume_latest = true;

    let options = python_options_from_config(&config).expect("raw MJAI latest resume should work");
    assert_eq!(options.resume, Some(latest));
}

#[test]
fn python_resume_checkpoint_bc_shards_uses_latest_when_requested() {
    let root = unique_temp_dir("bc-shards-latest");
    let checkpoint_dir = root
        .join("stages")
        .join("bc_baseline")
        .join("runs")
        .join("latest_run")
        .join("checkpoints");
    std::fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir should be created");
    let latest = checkpoint_dir.join("latest.pt");
    std::fs::write(&latest, b"checkpoint").expect("latest checkpoint should write");
    let mut config = python_guard_config();
    config.output_dir = root;
    config.resume_latest = true;

    let options =
        python_options_from_config(&config).expect("BC shards should support latest resume");
    assert_eq!(options.resume, Some(latest));
}

#[test]
fn python_options_from_config_uses_raw_mjai_when_manifest_absent() {
    let mut config = python_guard_config();
    config.bc_shards_manifest_path = None;
    let options = python_options_from_config(&config).expect("plain BC should route to Python");
    match options.input {
        PythonLearnerInput::RawMjai {
            data_dirs,
            train_fraction,
            augment,
            transport,
            ..
        } => {
            assert_eq!(data_dirs, vec![PathBuf::from("/tmp/data")]);
            assert_eq!(train_fraction, 0.9);
            assert!(augment);
            assert!(!options.raw_mjai_validation_augment);
            assert_eq!(transport, PythonRawMjaiTransportConfig::PinnedPyo3);
        }
        PythonLearnerInput::BcShards { .. } => panic!("expected raw MJAI input"),
    }

    config.max_validation_samples = Some(65_536);
    let options = python_options_from_config(&config).expect("plain BC should route to Python");
    assert_eq!(options.validation_steps, 64);
    assert_eq!(options.validation_max_samples, Some(65_536));

    config.max_validation_samples = None;
    config.max_validation_batches = Some(7);
    let options = python_options_from_config(&config).expect("plain BC should route to Python");
    assert_eq!(options.validation_steps, 7);
    assert_eq!(options.validation_max_samples, None);
}

#[test]
fn python_options_full_epoch_without_step_budget_uses_constant_schedule() {
    let mut config = python_guard_config();
    config.bc_shards_manifest_path = None;
    config.max_train_steps = None;
    config.full_epoch = true;
    config.resume_latest = false;

    let options =
        python_options_from_config(&config).expect("full-epoch raw MJAI should route to Python");

    assert_eq!(options.steps, None);
    assert_eq!(options.lr_schedule, "constant");
    assert_eq!(options.schedule_total_steps, None);
    assert_eq!(options.schedule_target_games, None);
}

#[test]
fn python_options_full_epoch_with_raw_mjai_target_games_uses_cosine_schedule() {
    let mut config = python_guard_config();
    config.bc_shards_manifest_path = None;
    config.max_train_steps = None;
    config.full_epoch = true;
    config.resume_latest = false;
    config.batch_size = 1024;
    config.python_raw_mjai_target_games = Some(5_000_000);
    config.python_raw_mjai_estimated_samples_per_game = Some(70);

    let options =
        python_options_from_config(&config).expect("full-epoch raw MJAI should route to Python");

    assert_eq!(options.steps, None);
    assert_eq!(options.lr_schedule, "cosine");
    assert_eq!(options.schedule_total_steps, None);
    assert_eq!(options.schedule_target_games, Some(5_000_000));
}

#[test]
fn python_options_explicit_max_train_steps_overrides_raw_mjai_target_games() {
    let mut config = python_guard_config();
    config.bc_shards_manifest_path = None;
    config.max_train_steps = Some(3);
    config.full_epoch = true;
    config.resume_latest = false;
    config.batch_size = 1024;
    config.python_raw_mjai_target_games = Some(5_000_000);
    config.python_raw_mjai_estimated_samples_per_game = Some(70);

    let options =
        python_options_from_config(&config).expect("full-epoch raw MJAI should route to Python");

    assert_eq!(options.steps, Some(3));
    assert_eq!(options.lr_schedule, "cosine");
    assert_eq!(options.schedule_total_steps, Some(3));
    assert_eq!(options.schedule_target_games, None);
}

#[test]
fn python_options_from_config_uses_explicit_raw_mjai_dirs() {
    let mut config = python_guard_config();
    config.bc_shards_manifest_path = None;
    config.resume_latest = false;
    config.raw_mjai_data_dirs = vec![PathBuf::from("/data/a"), PathBuf::from("/data/b")];

    let options = python_options_from_config(&config).expect("plain BC should route to Python");

    match options.input {
        PythonLearnerInput::RawMjai { data_dirs, .. } => {
            assert_eq!(data_dirs, config.raw_mjai_data_dirs);
        }
        PythonLearnerInput::BcShards { .. } => panic!("expected raw MJAI input"),
    }
}

#[test]
fn python_options_from_config_preserves_profiles_and_variant() {
    let mut config = python_guard_config();
    config.python_residual_profile = PythonResidualProfileConfig::ReluNoSe;
    config.python_variant = PythonLearnerVariant::CompileMaxAutotune;

    let options = python_options_from_config(&config).expect("plain BC should route to Python");

    assert_eq!(
        options.residual_profile,
        PythonResidualProfileConfig::ReluNoSe
    );
    assert_eq!(options.variant, PythonLearnerVariant::CompileMaxAutotune);
}

type PythonGuardCase = (fn(&mut TrainConfig), &'static str);

#[test]
fn python_options_from_config_accepts_exit_sidecar_default_off_and_weighted_with_manifest_provenance()
 {
    let mut config = python_guard_config();
    config.exit_sidecar_path = Some(PathBuf::from("/tmp/exit.jsonl"));
    let options = python_options_from_config(&config)
        .expect("default-off ExIt sidecar should route through shards");
    assert_eq!(options.exit_weight, 0.0);

    let root = unique_temp_dir("exit-provenance");
    std::fs::create_dir_all(&root).expect("temp root should be created");
    let manifest = root.join("manifest.json");
    std::fs::write(
        &manifest,
        r#"{"exit_sidecar":{"path":"/tmp/exit.jsonl","source_net_hash":123,"source_version":7}}"#,
    )
    .expect("manifest should write");
    config.bc_shards_manifest_path = Some(manifest);
    config.advanced_loss = Some(AdvancedLossConfig {
        exit: Some(0.25),
        ..Default::default()
    });
    let options = python_options_from_config(&config)
        .expect("weighted ExIt sidecar should route through shards");
    assert_eq!(options.exit_weight, 0.25);
}

#[test]
fn python_options_from_config_rejects_weighted_exit_without_manifest_provenance() {
    let cases: &[(Option<&str>, bool, &'static str)] = &[
        (None, true, "exit_sidecar provenance"),
        (None, false, "readable BC shard manifest provenance"),
        (
            Some(r#"{"exit_sidecar":{"path":"/tmp/exit.jsonl","source_version":7}}"#),
            true,
            "source_net_hash",
        ),
        (
            Some(r#"{"exit_sidecar":{"path":"/tmp/exit.jsonl","source_net_hash":123}}"#),
            true,
            "source_version",
        ),
        (
            Some(
                r#"{"exit_sidecar":{"path":"/tmp/other.jsonl","source_net_hash":123,"source_version":7}}"#,
            ),
            true,
            "sidecar path",
        ),
    ];
    for (manifest_json, write_manifest, expected) in cases.iter().copied() {
        let root = unique_temp_dir("exit-provenance-reject");
        std::fs::create_dir_all(&root).expect("temp root should be created");
        let manifest = root.join("manifest.json");
        if write_manifest {
            std::fs::write(&manifest, manifest_json.unwrap_or("{}"))
                .expect("manifest should write");
        }
        let mut config = python_guard_config();
        config.bc_shards_manifest_path = Some(manifest);
        config.exit_sidecar_path = Some(PathBuf::from("/tmp/exit.jsonl"));
        config.advanced_loss = Some(AdvancedLossConfig {
            exit: Some(0.1),
            ..Default::default()
        });

        let err = python_options_from_config(&config).expect_err("missing provenance should fail");
        assert!(err.contains(expected), "{err}");
    }
}

#[test]
fn python_options_from_config_rejects_unsupported_advanced_modes() {
    let cases: &[PythonGuardCase] = &[
        (
            |config| {
                config.bc_shards_manifest_path = None;
                config.exit_sidecar_path = Some(PathBuf::from("/tmp/exit.jsonl"));
            },
            "compact BC shards",
        ),
        (
            |config| {
                config.bc_shards_manifest_path = None;
                config.delta_q_sidecar_path = Some(PathBuf::from("/tmp/delta-q.jsonl"));
            },
            "compact BC shards",
        ),
        (
            |config| {
                config.advanced_loss = Some(AdvancedLossConfig {
                    exit: Some(0.1),
                    ..Default::default()
                })
            },
            "ExIt sidecar-backed compact shard labels",
        ),
        (
            |config| {
                config.advanced_loss = Some(AdvancedLossConfig {
                    delta_q: Some(0.1),
                    ..Default::default()
                })
            },
            "DeltaQ sidecar-backed compact shard labels",
        ),
        (
            |config| {
                config.delta_q_sidecar_path = Some(PathBuf::from("/tmp/delta-q.jsonl"));
                config.advanced_loss = Some(AdvancedLossConfig {
                    delta_q: Some(0.1),
                    ..Default::default()
                })
            },
            "delta_q_output_contract_missing",
        ),
        (
            |config| {
                config.advanced_loss = Some(AdvancedLossConfig {
                    belief_fields: Some(0.1),
                    ..Default::default()
                })
            },
            "advanced_loss.belief_fields",
        ),
        (
            |config| {
                config.advanced_loss = Some(AdvancedLossConfig {
                    mixture_weight: Some(0.1),
                    ..Default::default()
                })
            },
            "advanced_loss.mixture_weight",
        ),
        (
            |config| {
                config.advanced_loss = Some(AdvancedLossConfig {
                    opponent_hand_type: Some(0.1),
                    ..Default::default()
                })
            },
            "advanced_loss.opponent_hand_type",
        ),
    ];
    for (configure, expected) in cases.iter().copied() {
        let mut config = python_guard_config();
        configure(&mut config);
        let err = python_options_from_config(&config).expect_err("unsupported mode should fail");
        assert!(err.contains(expected), "{err}");
    }
}

#[test]
fn python_ppo_control_options_from_config_maps_campaign_root_to_default_ppo_run_dir() {
    let mut config = python_guard_config();
    config.rl = Some(RlTrainConfig::default());
    config.python_backbone_profile = PythonBackboneProfileConfig::Conv2dLocal3;
    config.resume_checkpoint = Some(PathBuf::from("/tmp/bc/checkpoints/latest.pt"));
    config.rl.as_mut().expect("rl config").microbatch_size = Some(321);

    let options = python_ppo_control_options_from_config(&config)
        .expect("PPO control should route to Python");

    assert_eq!(
        options.output_dir,
        PathBuf::from("/tmp/out/stages/T1_ppo_control/runs/latest_run")
    );
    assert_eq!(options.microbatch_size, 321);
    assert_eq!(options.ppo_pipeline_depth, 0);
    assert_eq!(options.rollout_device, None);
    config.rl.as_mut().expect("rl config").ppo_pipeline_depth = Some(1);
    let options = python_ppo_control_options_from_config(&config)
        .expect("PPO control pipeline depth should route to Python");
    assert_eq!(options.ppo_pipeline_depth, 1);
    config.rl.as_mut().expect("rl config").ppo_rollout_device = Some("cpu".to_string());
    let options = python_ppo_control_options_from_config(&config)
        .expect("PPO control rollout device should route to Python");
    assert_eq!(options.rollout_device.as_deref(), Some("cpu"));
    config.rl.as_mut().expect("rl config").ppo_pipeline_depth = Some(2);
    let err = python_ppo_control_options_from_config(&config)
        .expect_err("invalid pipeline depth should fail");
    assert!(err.contains("ppo_pipeline_depth"));
}

#[test]
fn rl_stage_helpers_map_default_phases_and_respect_override() {
    assert_eq!(
        rl_stage_for_phase(RlPhaseConfig::PpoControl),
        T1_PPO_CONTROL_STAGE
    );
    assert_eq!(
        rl_stage_for_phase(RlPhaseConfig::DrdaAchSelfPlay),
        T3_DRDA_RESIDUAL_ACH_STAGE
    );
    assert_eq!(
        rl_stage_for_phase(RlPhaseConfig::ExitPondering),
        T5_EXIT_AUXILIARY_STAGE
    );
    assert_eq!(
        RlPhaseConfig::DrdaAchSelfPlay.default_stage(),
        T3_DRDA_RESIDUAL_ACH_STAGE
    );

    let mut config = python_guard_config();
    config.rl = Some(RlTrainConfig {
        phase: RlPhaseConfig::DrdaAchSelfPlay,
        ..Default::default()
    });
    assert_eq!(rl_stage_for_config(&config), T3_DRDA_RESIDUAL_ACH_STAGE);

    config.stage = Some("custom_stage".to_string());
    assert_eq!(rl_stage_for_config(&config), "custom_stage");
}

#[test]
fn python_run_dir_uses_rl_phase_stage_default() {
    let mut config = python_guard_config();
    config.rl = Some(RlTrainConfig {
        phase: RlPhaseConfig::DrdaAchSelfPlay,
        ..Default::default()
    });

    assert_eq!(
        python_run_dir(&config, rl_stage_for_config(&config)),
        PathBuf::from("/tmp/out/stages/T3_drda_residual_ach/runs/latest_run")
    );

    config.rl = Some(RlTrainConfig {
        phase: RlPhaseConfig::ExitPondering,
        ..Default::default()
    });
    config.run_name = Some("run_002".to_string());

    assert_eq!(
        python_run_dir(&config, rl_stage_for_config(&config)),
        PathBuf::from("/tmp/out/stages/T5_exit_auxiliary/runs/run_002")
    );
}
