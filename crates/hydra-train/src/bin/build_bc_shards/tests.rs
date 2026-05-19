use super::*;

#[test]
fn parse_args_accepts_minimal_required_flags() {
    let cli = parse_args(
        "build_bc_shards",
        vec![
            "build_bc_shards".to_string(),
            "--input".to_string(),
            "replays".to_string(),
            "--output-dir".to_string(),
            "out".to_string(),
        ],
    )
    .expect("args should parse");

    assert_eq!(cli.input, PathBuf::from("replays"));
    assert_eq!(cli.output_dir, PathBuf::from("out"));
    assert_eq!(cli.shard_samples, 10_000);
    assert!((cli.train_fraction - 0.9).abs() < f32::EPSILON);
}

#[test]
fn parse_args_accepts_validate_manifest_without_build_flags() {
    let cli = parse_args(
        "build_bc_shards",
        vec![
            "build_bc_shards".to_string(),
            "--validate-manifest".to_string(),
            "out/bc_shards_manifest.json".to_string(),
        ],
    )
    .expect("validate-only args should parse");

    assert_eq!(
        cli.validate_manifest,
        Some(PathBuf::from("out/bc_shards_manifest.json"))
    );
    assert!(cli.input.as_os_str().is_empty());
    assert!(cli.output_dir.as_os_str().is_empty());
}

#[test]
fn parse_args_rejects_validate_manifest_with_build_flags() {
    let err = parse_args(
        "build_bc_shards",
        vec![
            "build_bc_shards".to_string(),
            "--validate-manifest".to_string(),
            "out/bc_shards_manifest.json".to_string(),
            "--input".to_string(),
            "replays".to_string(),
        ],
    )
    .expect_err("validate-only mode should reject build flags");

    assert!(err.contains("cannot be combined"));
}
#[test]
fn parse_args_rejects_partial_sidecar_provenance() {
    let err = parse_args(
        "build_bc_shards",
        vec![
            "build_bc_shards".to_string(),
            "--input".to_string(),
            "replays".to_string(),
            "--output-dir".to_string(),
            "out".to_string(),
            "--exit-sidecar".to_string(),
            "exit.jsonl".to_string(),
        ],
    )
    .expect_err("partial sidecar flags should fail");

    assert!(err.contains("exit sidecar requires path"));
}

#[test]
fn parse_split_accepts_aliases() {
    assert!(matches!(parse_split("both"), Ok(BcShardSplitMode::Both)));
    assert!(matches!(parse_split("train"), Ok(BcShardSplitMode::Train)));
    assert!(matches!(
        parse_split("validation"),
        Ok(BcShardSplitMode::Validation)
    ));
}

#[test]
fn parses_parallel_resume_report_flags() {
    let cli = parse_args(
        "build_bc_shards",
        vec![
            "build_bc_shards".to_string(),
            "--input".to_string(),
            "replays".to_string(),
            "--output-dir".to_string(),
            "out".to_string(),
            "--num-threads".to_string(),
            "8".to_string(),
            "--queue-bound".to_string(),
            "64".to_string(),
            "--resume".to_string(),
            "--resume-dir".to_string(),
            "resume-state".to_string(),
            "--chunk-games".to_string(),
            "2048".to_string(),
            "--report-name".to_string(),
            "scan.json".to_string(),
            "--progress-jsonl".to_string(),
            "progress.jsonl".to_string(),
            "--max-error-examples".to_string(),
            "7".to_string(),
        ],
    )
    .expect("parallel/resume/report args should parse");

    assert_eq!(cli.num_threads, Some(8));
    assert_eq!(cli.queue_bound, 64);
    assert!(cli.resume);
    assert_eq!(cli.resume_dir, Some(PathBuf::from("resume-state")));
    assert_eq!(cli.chunk_games, 2048);
    assert_eq!(cli.report_name, Some("scan.json".to_string()));
    assert_eq!(cli.progress_jsonl_name, Some("progress.jsonl".to_string()));
    assert_eq!(cli.max_error_examples, 7);
}

#[test]
fn rejects_zero_parallel_bounds() {
    for flag in ["--num-threads", "--queue-bound", "--chunk-games"] {
        let err = parse_args(
            "build_bc_shards",
            vec![
                "build_bc_shards".to_string(),
                "--input".to_string(),
                "replays".to_string(),
                "--output-dir".to_string(),
                "out".to_string(),
                flag.to_string(),
                "0".to_string(),
            ],
        )
        .expect_err("zero bound should fail");

        assert!(err.contains("must be > 0"), "{flag}: {err}");
    }
}

#[test]
fn parses_no_report_and_dry_scan_only() {
    let cli = parse_args(
        "build_bc_shards",
        vec![
            "build_bc_shards".to_string(),
            "--input".to_string(),
            "replays".to_string(),
            "--output-dir".to_string(),
            "out".to_string(),
            "--no-report".to_string(),
            "--dry-scan-only".to_string(),
        ],
    )
    .expect("no-report dry-scan args should parse");

    assert_eq!(cli.report_name, None);
    assert!(cli.dry_scan_only);
}

#[test]
fn parse_max_games_and_max_samples_flags() {
    let cli = parse_args(
        "build_bc_shards",
        vec![
            "build_bc_shards".to_string(),
            "--input".to_string(),
            "replays".to_string(),
            "--output-dir".to_string(),
            "out".to_string(),
            "--max-games".to_string(),
            "17".to_string(),
            "--max-samples".to_string(),
            "101".to_string(),
        ],
    )
    .expect("limit args should parse");

    assert_eq!(cli.max_games, Some(17));
    assert_eq!(cli.max_samples, Some(101));
}

#[test]
fn max_games_zero_is_rejected() {
    let err = parse_args(
        "build_bc_shards",
        vec![
            "build_bc_shards".to_string(),
            "--input".to_string(),
            "replays".to_string(),
            "--output-dir".to_string(),
            "out".to_string(),
            "--max-games".to_string(),
            "0".to_string(),
        ],
    )
    .expect_err("zero max-games should fail");

    assert!(err.contains("--max-games must be > 0"));
}

#[test]
fn max_samples_zero_is_rejected() {
    let err = parse_args(
        "build_bc_shards",
        vec![
            "build_bc_shards".to_string(),
            "--input".to_string(),
            "replays".to_string(),
            "--output-dir".to_string(),
            "out".to_string(),
            "--max-samples".to_string(),
            "0".to_string(),
        ],
    )
    .expect_err("zero max-samples should fail");

    assert!(err.contains("--max-samples must be > 0"));
}
