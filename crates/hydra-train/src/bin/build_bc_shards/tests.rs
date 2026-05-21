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
fn throughput_summary_formats_rates_and_missing_values() {
    let report = BcShardBuildReport {
        schema_version: 1,
        started_at: "start".to_string(),
        finished_at: "end".to_string(),
        elapsed_seconds: 2.5,
        abi: hydra_train_exec::bc_shard_builder::BcShardAbiReport {
            storage_layout: "compact".to_string(),
            shard_version: 1,
            layout_version: 1,
            manifest_version: 2,
            feature_flags: 0,
            record_size: 1,
            header_size: 32,
            base_record_size: 1,
            max_record_size: 1,
            dense_obs_f32_bytes_per_sample: 26_112,
        },
        disk: hydra_train_exec::bc_shard_builder::BcShardDiskReport {
            output_dir: "out".to_string(),
            existing_output_bytes: 0,
            available_bytes_before: None,
            available_bytes_after: None,
            projected_output_bytes: None,
            projected_sample_count: None,
            projection_source: "unavailable".to_string(),
        },
        plan_splits: Vec::new(),
        command: hydra_train_exec::bc_shard_builder::BcShardBuildCommandReport {
            input: "in".to_string(),
            output_dir: "out".to_string(),
            manifest_name: "manifest.json".to_string(),
            train_fraction: 1.0,
            shard_samples: 1,
            split: "train".to_string(),
            num_threads: None,
            queue_bound: 1,
            resume: false,
            chunk_games: 1,
            limit_max_games: None,
            limit_max_samples: None,
            exit_sidecar: None,
            delta_q_sidecar: None,
        },
        scan: hydra_train_exec::bc_shard_builder::BcShardScanReport {
            source_count: 1,
            source_total_games_hint: 1,
            source_train_count_hint: 1,
            source_val_count_hint: 0,
            source_counts_exact: true,
            input_compressed_bytes: None,
        },
        build: hydra_train_exec::bc_shard_builder::BcShardMaterializationReport {
            loaded_games: 1,
            skipped_games: 0,
            empty_games: 0,
            total_samples: 10,
            train_samples: 10,
            validation_samples: 0,
            limit_reached: false,
            sample_cap_outcome: "not_configured".to_string(),
            samples_per_loaded_non_empty_game: Some(10.0),
            bad_source_examples: Vec::new(),
            resume_chunks_reused: 0,
            resume_chunks_built: 1,
        },
        output: hydra_train_exec::bc_shard_builder::BcShardOutputReport {
            shard_count: 1,
            output_bytes: 10,
            manifest_bytes: 1,
            bytes_per_sample: Some(1.0),
            dense_equivalent_observation_bytes: 261_120,
            dense_equivalent_observation_bytes_per_sample: 26_112,
            savings_ratio_vs_dense_observation: Some(26_112.0),
            splits: Vec::new(),
        },
        rates: hydra_train_exec::bc_shard_builder::BcShardBuildRates {
            games_per_second: None,
            samples_per_second: 4.0,
            output_mib_per_second: 0.5,
            input_mib_per_second: Some(1.25),
        },
        manifest_path: "out/manifest.json".to_string(),
        progress_jsonl_path: None,
    };

    let line = throughput_summary(&report);

    assert!(line.contains("games/s=n/a"));
    assert!(line.contains("samples/s=4.00"));
    assert!(line.contains("input_mib/s=1.25"));
    assert!(line.contains("output_mib/s=0.50"));
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
