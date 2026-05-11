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
