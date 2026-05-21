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
