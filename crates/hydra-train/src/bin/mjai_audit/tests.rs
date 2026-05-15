use super::*;
use std::ffi::OsStr;
use std::io::{Cursor, Write};
use std::time::{SystemTime, UNIX_EPOCH};

fn valid_mjai_log() -> String {
    [
        r#"{"type":"start_game","names":["a","b","c","d"],"id":"game-1"}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["6m","6m","6m","7m","8m","9m","1p","2p","3p","4p","5p","6p","7p","8p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["1m","1m","2m","2m","3m","3m","4m","4m","5m","5m","6m","6m","7m"],["1p","1p","2p","2p","3p","3p","4p","4p","5p","5p","6p","6p","7p"]]}"#,
        r#"{"type":"dahai","actor":0,"pai":"8p","tsumogiri":false}"#,
        r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
        r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
        r#"{"type":"ryukyoku"}"#,
        r#"{"type":"end_kyoku"}"#,
    ]
    .join("\n")
}

fn unique_temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hydra_mjai_audit_{name}_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn write_tar_zst_with_entries(path: &Path, entries: &[(&str, &[u8])]) {
    let tar_bytes = {
        let mut builder = tar::Builder::new(Vec::new());
        for (entry_name, entry_data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(entry_data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, *entry_name, Cursor::new(*entry_data))
                .expect("append archive entry");
        }
        builder.into_inner().expect("finish tar builder")
    };

    let file = fs::File::create(path).expect("create zstd archive");
    let mut encoder = zstd::Encoder::new(file, 0).expect("create zstd encoder");
    encoder
        .write_all(&tar_bytes)
        .expect("write tar payload into zstd stream");
    encoder.finish().expect("finish zstd stream");
}

fn cleanup_dir(dir: &Path) {
    if dir.exists() {
        fs::remove_dir_all(dir).expect("temp dir should be removable");
    }
}

fn read_inventory_records(path: &Path) -> Vec<FailureInventoryRecord> {
    fs::read_to_string(path)
        .expect("inventory file should be readable")
        .lines()
        .map(|line| {
            serde_json::from_str::<FailureInventoryRecord>(line)
                .expect("inventory line should parse")
        })
        .collect()
}

#[test]
fn parse_args_uses_defaults_and_accepts_overrides() {
    let cfg = parse_args(vec!["mjai_audit".to_string(), "/data".to_string()])
        .expect("default args should parse");
    assert_eq!(cfg.data_dir, PathBuf::from("/data"));
    assert_eq!(cfg.threads, 16);
    assert_eq!(cfg.failure_examples, 20);
    assert_eq!(cfg.failure_inventory_dir, None);

    let cfg = parse_args(vec![
        "mjai_audit".to_string(),
        "/data".to_string(),
        "--threads".to_string(),
        "8".to_string(),
        "--failure-examples".to_string(),
        "5".to_string(),
        "--failure-inventory-dir".to_string(),
        "/tmp/inventory".to_string(),
    ])
    .expect("override args should parse");
    assert_eq!(cfg.threads, 8);
    assert_eq!(cfg.failure_examples, 5);
    assert_eq!(
        cfg.failure_inventory_dir,
        Some(PathBuf::from("/tmp/inventory"))
    );
}

#[test]
fn parse_args_rejects_bad_values_and_unknown_flags() {
    assert!(parse_args(vec!["mjai_audit".to_string()]).is_err());

    let zero = parse_args(vec![
        "mjai_audit".to_string(),
        "/data".to_string(),
        "--threads".to_string(),
        "0".to_string(),
    ])
    .expect_err("zero threads should fail");
    assert!(zero.contains("greater than 0"));

    let invalid = parse_args(vec![
        "mjai_audit".to_string(),
        "/data".to_string(),
        "--threads".to_string(),
        "abc".to_string(),
    ])
    .expect_err("non numeric threads should fail");
    assert!(invalid.contains("invalid --threads value"));

    let unknown = parse_args(vec![
        "mjai_audit".to_string(),
        "/data".to_string(),
        "--mystery".to_string(),
    ])
    .expect_err("unknown flag should fail");
    assert!(unknown.contains("unknown argument"));
}

#[test]
fn path_classifiers_and_error_summary_match_expected_suffix_rules() {
    assert_eq!(
        usage("audit-bin"),
        "Usage: audit-bin <data-dir> [--threads N] [--failure-examples N] [--failure-inventory-dir DIR]"
    );

    assert!(is_archive_file(Path::new("dataset.tar.zst")));
    assert!(is_archive_file(Path::new("dataset.tar-0001.zst")));
    assert!(!is_archive_file(Path::new("dataset.zip")));

    assert!(is_mjai_archive_entry(Path::new("round.mjai.json")));
    assert!(is_mjai_archive_entry(Path::new("round.mjai.json.gz")));
    assert!(is_mjai_archive_entry(Path::new("round.json")));
    assert!(!is_mjai_archive_entry(Path::new("round.txt")));

    assert_eq!(summarize_error("first line\nsecond line"), "first line");
    assert_eq!(summarize_error("   \n  "), "unknown error");
}

#[test]
fn collect_sources_accepts_single_files_and_directory_archives() {
    let dir = unique_temp_dir("collect_sources");
    let keep_json = dir.join("a.json");
    let keep_archive = dir.join("dataset.tar.zst");
    let keep_gz = dir.join("b.json.gz");
    let skip_txt = dir.join("note.txt");
    fs::write(&keep_json, valid_mjai_log()).expect("write json");
    fs::write(&keep_gz, b"gz").expect("write gz placeholder");
    fs::write(&skip_txt, b"note").expect("write txt");
    write_tar_zst_with_entries(&keep_archive, &[("good.json", valid_mjai_log().as_bytes())]);

    let single = collect_sources(&keep_json).expect("single file should be accepted");
    assert_eq!(single, vec![DataSource::LooseFile(keep_json.clone())]);

    let single_archive = collect_sources(&keep_archive).expect("archive should be accepted");
    assert_eq!(
        single_archive,
        vec![DataSource::Archive(keep_archive.clone())]
    );

    let listed = collect_sources(&dir).expect("directory collection should succeed");
    assert_eq!(
        listed,
        vec![
            DataSource::LooseFile(keep_json.clone()),
            DataSource::LooseFile(keep_gz.clone()),
        ]
    );

    cleanup_dir(&dir);
}

#[test]
fn collect_sources_reports_missing_directory_with_context() {
    let missing = std::env::temp_dir().join(format!(
        "hydra_missing_mjai_audit_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    let err = collect_sources(&missing).expect_err("missing dir should fail");
    assert!(err.contains("failed to scan data dir"));
    assert!(err.contains(&missing.display().to_string()));
}

#[test]
fn archive_and_mjai_entry_classifiers_reject_directory_and_wrong_suffixes() {
    assert!(!is_archive_file(Path::new("dataset.tar")));
    assert!(!is_archive_file(Path::new("dataset.zst")));
    assert!(!is_archive_file(Path::new("dataset.tar.gz")));

    assert!(!is_mjai_archive_entry(Path::new("round.mjai")));
    assert!(!is_mjai_archive_entry(Path::new("round.log")));
    assert!(is_mjai_archive_entry(Path::new("round.json.zst")));
}

#[test]
fn parse_args_requires_flag_values_for_threads_and_failure_examples() {
    let threads_err = parse_args(vec![
        "mjai_audit".to_string(),
        "/data".to_string(),
        "--threads".to_string(),
    ])
    .expect_err("missing threads value should fail");
    assert!(threads_err.contains("missing value for --threads"));

    let examples_err = parse_args(vec![
        "mjai_audit".to_string(),
        "/data".to_string(),
        "--failure-examples".to_string(),
    ])
    .expect_err("missing failure-examples value should fail");
    assert!(examples_err.contains("missing value for --failure-examples"));

    let inventory_err = parse_args(vec![
        "mjai_audit".to_string(),
        "/data".to_string(),
        "--failure-inventory-dir".to_string(),
    ])
    .expect_err("missing failure inventory dir should fail");
    assert!(inventory_err.contains("missing value for --failure-inventory-dir"));
}

#[test]
fn summary_helpers_cover_zero_positive_sorting_and_failure_limits() {
    assert_eq!(throughput_per_second(10, 0.0), 0.0);
    assert_eq!(throughput_per_second(12, 3.0), 4.0);

    let sorted = sort_error_buckets(HashMap::from([
        ("z bucket".to_string(), 2usize),
        ("a bucket".to_string(), 2usize),
        ("mid bucket".to_string(), 5usize),
    ]));
    assert_eq!(sorted[0], ("mid bucket".to_string(), 5));
    assert_eq!(sorted[1], ("a bucket".to_string(), 2));
    assert_eq!(sorted[2], ("z bucket".to_string(), 2));

    assert!(should_record_failure_example(0, 2));
    assert!(should_record_failure_example(1, 2));
    assert!(!should_record_failure_example(2, 2));
    assert!(!should_record_failure_example(0, 0));
}

#[test]
fn report_helpers_cover_totals_rates_and_failure_rendering() {
    let totals = AuditTotals {
        loaded: 3,
        skipped: 2,
        samples: 40,
    };
    assert_eq!(total_files(&totals), 5);

    let rates = compute_audit_rates(&totals, 2.0);
    assert_eq!(rates.elapsed_secs, 2.0);
    assert_eq!(rates.files_per_sec, 2.5);
    assert_eq!(rates.samples_per_sec, 20.0);

    assert_eq!(
        audit_totals_summary_line(&totals),
        "Audit complete: loaded=3 skipped=2 samples=40 total=5"
    );
    assert_eq!(
        audit_speed_summary_line(&rates),
        "Speed: elapsed=2.00s files_per_sec=2.50 samples_per_sec=20.00"
    );

    let none = failure_report_lines(&[], &[]);
    assert_eq!(none, vec!["No failures detected.".to_string()]);

    let lines = failure_report_lines(
        &[("bucket a".to_string(), 3), ("bucket b".to_string(), 1)],
        &[("/tmp/a.json".to_string(), "boom".to_string())],
    );
    assert_eq!(lines[0], "Top failure buckets:");
    assert!(lines.iter().any(|line| line.contains("bucket a")));
    assert!(lines.iter().any(|line| line == "Failure examples:"));
    assert!(lines.iter().any(|line| line.contains("/tmp/a.json")));
}

#[test]
fn exit_code_helper_distinguishes_success_from_failure() {
    assert_eq!(exit_code_for_run_result(&Ok(())), 0);
    assert_eq!(exit_code_for_run_result(&Err("boom".to_string())), 1);
}

#[test]
fn collect_sources_accepts_non_mjai_single_file_inputs_too() {
    let dir = unique_temp_dir("single_file_any_suffix");
    let single = dir.join("notes.txt");
    fs::write(&single, b"hello").expect("single file fixture should write");

    let paths = collect_sources(&single).expect("single file path should be returned as-is");

    assert_eq!(paths, vec![DataSource::LooseFile(single.clone())]);

    cleanup_dir(&dir);
}

#[test]
fn summarize_error_keeps_single_line_messages_intact() {
    assert_eq!(summarize_error("plain error"), "plain error");
}

#[test]
fn run_succeeds_for_single_valid_file_path() {
    let dir = unique_temp_dir("single_valid_run");
    let path = dir.join("game.json");
    fs::write(&path, valid_mjai_log()).expect("valid mjai fixture should write");

    let args = vec![
        "mjai_audit".to_string(),
        path.display().to_string(),
        "--threads".to_string(),
        "1".to_string(),
        "--failure-examples".to_string(),
        "1".to_string(),
    ];

    let previous_args = std::env::args_os().collect::<Vec<_>>();
    let _ = previous_args;
    let result = {
        let parsed = parse_args(args).expect("args should parse");
        let totals = AuditTotals {
            loaded: 1,
            skipped: 0,
            samples: load_game_from_path(&path)
                .expect("fixture should load")
                .num_samples(),
        };
        let rates = compute_audit_rates(&totals, 1.0);
        assert_eq!(
            audit_totals_summary_line(&totals),
            format!(
                "Audit complete: loaded=1 skipped=0 samples={} total=1",
                totals.samples
            )
        );
        assert_eq!(parsed.data_dir, path);
        assert_eq!(parsed.threads, 1);
        assert_eq!(parsed.failure_examples, 1);
        assert_eq!(parsed.failure_inventory_dir, None);
        assert!(audit_speed_summary_line(&rates).contains("files_per_sec=1.00"));
        Ok::<(), String>(())
    };

    assert_eq!(result, Ok(()));

    cleanup_dir(&dir);
}

#[test]
fn run_style_helpers_cover_failed_single_file_load_accounting() {
    let dir = unique_temp_dir("single_invalid_run");
    let path = dir.join("bad.json");
    fs::write(&path, "not-json").expect("invalid fixture should write");

    let err = match load_game_from_path(&path) {
        Ok(_) => panic!("invalid mjai fixture should fail to load"),
        Err(err) => err,
    };
    let summary = summarize_error(&err.to_string());
    let buckets = sort_error_buckets(HashMap::from([(summary.clone(), 1usize)]));
    let examples = vec![(path.display().to_string(), err.to_string())];
    let totals = AuditTotals {
        loaded: 0,
        skipped: 1,
        samples: 0,
    };

    assert_eq!(total_files(&totals), 1);
    assert_eq!(buckets[0], (summary, 1));
    let lines = failure_report_lines(&buckets, &examples);
    assert!(lines.iter().any(|line| line == "Top failure buckets:"));
    assert!(lines.iter().any(|line| line == "Failure examples:"));
    assert!(lines.iter().any(|line| line.contains("bad.json")));

    cleanup_dir(&dir);
}

#[test]
fn archive_open_failure_string_includes_path() {
    let path = unique_temp_dir("missing_archive_open").join("dataset.tar.zst");

    let err = fs::File::open(&path)
        .map(|_| ())
        .map_err(|err| format!("failed to open archive {}: {err}", path.display()))
        .expect_err("missing archive should fail to open");

    assert!(err.contains("failed to open archive"));
    assert!(err.contains(path.to_string_lossy().as_ref()));
}

#[test]
fn archive_decode_failure_string_includes_path() {
    let dir = unique_temp_dir("invalid_archive_decode");
    let path = dir.join("dataset.tar.zst");
    fs::write(&path, b"not a zstd archive").expect("invalid archive fixture should write");

    let file = fs::File::open(&path).expect("fixture should open");
    let mut decoder = zstd::Decoder::new(file).expect("decoder construction should succeed");
    let err = std::io::Read::read_to_end(&mut decoder, &mut Vec::new())
        .map(|_| ())
        .map_err(|err| format!("failed to decode archive {}: {err}", path.display()))
        .expect_err("invalid archive should fail when decoder is read");

    assert!(err.contains("failed to decode archive"));
    assert!(err.contains(path.to_string_lossy().as_ref()));

    cleanup_dir(&dir);
}

#[test]
fn archive_mode_ignores_non_mjai_entries_and_caps_failure_examples() {
    let dir = unique_temp_dir("archive_mode_mixed_entries");
    let path = dir.join("dataset.tar.zst");

    write_tar_zst_with_entries(
        &path,
        &[
            ("good.json", valid_mjai_log().as_bytes()),
            ("bad1.json", b"not-json"),
            ("bad2.json", b"still-not-json"),
            ("notes.txt", b"hello"),
        ],
    );

    let config = AuditConfig {
        data_dir: path.clone(),
        threads: 1,
        failure_examples: 1,
        failure_inventory_dir: None,
    };
    let state = AuditSharedState::default();
    audit_archive_source(&path, &config, &state, &mut None).expect("archive audit should work");

    let totals = state.totals();
    let error_buckets = state.snapshot_error_buckets();
    let failure_examples = state.snapshot_failure_examples();
    assert_eq!(totals.loaded, 1);
    assert_eq!(totals.skipped, 2);
    assert!(totals.samples > 0);
    assert_eq!(failure_examples.len(), 1);
    assert_eq!(error_buckets.values().sum::<usize>(), 2);
    let lines = failure_report_lines(&sort_error_buckets(error_buckets), &failure_examples);
    assert!(lines.iter().any(|line| line == "Top failure buckets:"));
    assert!(lines.iter().any(|line| line == "Failure examples:"));
    assert!(
        lines
            .iter()
            .any(|line| line.contains("dataset.tar.zst/bad1.json")
                || line.contains("dataset.tar.zst/bad2.json"))
    );

    cleanup_dir(&dir);
}

#[test]
fn archive_entry_identity_includes_archive_path() {
    let archive = Path::new("/data/dataset.tar.zst");
    let entry = Path::new("nested/bad.json");

    assert_eq!(
        archive_entry_identity(archive, entry),
        "/data/dataset.tar.zst/nested/bad.json"
    );
}

#[test]
fn failure_inventory_path_uses_source_name_and_hash_suffix() {
    let inventory_dir = Path::new("/tmp/inventory");
    let source = Path::new("/mnt/dev/dataset/dataset.tar.zst");
    let path = failure_inventory_path(inventory_dir, source);

    assert_eq!(path.parent(), Some(inventory_dir));
    let filename = path
        .file_name()
        .and_then(OsStr::to_str)
        .expect("utf8 filename");
    assert!(filename.starts_with("dataset.tar.zst-"));
    assert!(filename.ends_with(".jsonl"));
}

#[test]
fn audit_source_persists_failure_inventory_for_loose_file_and_archive_entries() {
    let dir = unique_temp_dir("failure_inventory");
    let inventory_dir = dir.join("inventory");
    let loose = dir.join("bad.json");
    let archive = dir.join("dataset.tar.zst");
    fs::write(&loose, "not-json").expect("invalid loose fixture should write");
    write_tar_zst_with_entries(
        &archive,
        &[
            ("good.json", valid_mjai_log().as_bytes()),
            ("bad.json", b"not-json"),
        ],
    );
    let expected_loose_error = match load_game_from_path(&loose) {
        Ok(_) => panic!("invalid loose file should fail"),
        Err(err) => err.to_string(),
    };
    let expected_archive_error = {
        let file = fs::File::open(&archive).expect("archive fixture should open");
        let zstd = zstd::Decoder::new(file).expect("archive decoder should open");
        let mut tar = tar::Archive::new(zstd);
        let mut entries = tar.entries().expect("archive entries should iterate");
        let bad_entry = entries
            .find_map(|entry| {
                let entry = entry.expect("archive entry should read");
                let entry_path = entry.path().expect("archive path should read").into_owned();
                (entry_path == std::path::Path::new("bad.json")).then_some(entry)
            })
            .expect("bad archive entry should exist");
        match load_game_from_stream(BufReader::new(bad_entry)) {
            Ok(_) => panic!("invalid archive entry should fail"),
            Err(err) => err.to_string(),
        }
    };

    let config = AuditConfig {
        data_dir: dir.clone(),
        threads: 2,
        failure_examples: 10,
        failure_inventory_dir: Some(inventory_dir.clone()),
    };
    let state = AuditSharedState::default();

    audit_source(&DataSource::LooseFile(loose.clone()), &config, &state)
        .expect("loose source audit should succeed");
    audit_source(&DataSource::Archive(archive.clone()), &config, &state)
        .expect("archive source audit should succeed");

    let loose_inventory = failure_inventory_path(&inventory_dir, &loose);
    let archive_inventory = failure_inventory_path(&inventory_dir, &archive);
    assert!(loose_inventory.exists());
    assert!(archive_inventory.exists());

    let loose_records = read_inventory_records(&loose_inventory);
    assert_eq!(loose_records.len(), 1);
    assert_eq!(loose_records[0].source, loose.display().to_string());
    assert_eq!(loose_records[0].identity, loose.display().to_string());
    assert_eq!(loose_records[0].error, expected_loose_error);

    let archive_records = read_inventory_records(&archive_inventory);
    assert_eq!(archive_records.len(), 1);
    assert_eq!(archive_records[0].source, archive.display().to_string());
    assert_eq!(
        archive_records[0].identity,
        format!("{}/bad.json", archive.display())
    );
    assert_eq!(archive_records[0].error, expected_archive_error);

    cleanup_dir(&dir);
}

#[test]
fn audit_source_keeps_clean_sources_artifact_free() {
    let dir = unique_temp_dir("clean_source_inventory");
    let inventory_dir = dir.join("inventory");
    let loose = dir.join("good.json");
    let archive = dir.join("good-dataset.tar.zst");
    fs::write(&loose, valid_mjai_log()).expect("valid loose fixture should write");
    write_tar_zst_with_entries(&archive, &[("good.json", valid_mjai_log().as_bytes())]);

    let config = AuditConfig {
        data_dir: dir.clone(),
        threads: 2,
        failure_examples: 10,
        failure_inventory_dir: Some(inventory_dir.clone()),
    };
    let state = AuditSharedState::default();

    audit_source(&DataSource::LooseFile(loose.clone()), &config, &state)
        .expect("clean loose source audit should succeed");
    audit_source(&DataSource::Archive(archive.clone()), &config, &state)
        .expect("clean archive source audit should succeed");

    assert!(!failure_inventory_path(&inventory_dir, &loose).exists());
    assert!(!failure_inventory_path(&inventory_dir, &archive).exists());
    assert!(!inventory_dir.exists());

    cleanup_dir(&dir);
}
