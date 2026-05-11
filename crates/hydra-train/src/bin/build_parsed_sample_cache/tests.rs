use super::*;

#[test]
fn parse_args_accepts_required_flags() {
    let cli = parse_args(
        "build_parsed_sample_cache",
        vec![
            "build_parsed_sample_cache".to_string(),
            "--input".to_string(),
            "replays".to_string(),
            "--output-dir".to_string(),
            "cache".to_string(),
        ],
    )
    .expect("args should parse");

    assert_eq!(cli.input, PathBuf::from("replays"));
    assert_eq!(cli.output_dir, PathBuf::from("cache"));
}

#[test]
fn collect_loose_mjai_files_rejects_archive_input() {
    let err = collect_loose_mjai_files(Path::new("/data/replays.tar.zst"), &mut Vec::new())
        .expect_err("archive input should fail clearly");
    assert!(err.contains("archive input not supported"));
}
