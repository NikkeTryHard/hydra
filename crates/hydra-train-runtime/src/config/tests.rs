use super::*;

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
        PythonLearnerVariant::CompileDefault
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
    assert_eq!(python.steps, 3);
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
