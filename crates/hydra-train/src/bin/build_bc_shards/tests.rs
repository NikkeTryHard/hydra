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
