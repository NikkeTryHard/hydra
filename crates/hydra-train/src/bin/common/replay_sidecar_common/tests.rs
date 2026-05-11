use super::*;
use serde::Serialize;
use std::cell::RefCell;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_path(name: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "hydra_replay_sidecar_{name}_{}_{}.{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos(),
        ext
    ))
}

#[derive(Serialize)]
struct DummyRecord {
    id: u32,
    label: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct WrapperRecord {
    id: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct WrapperReport {
    labels_emitted: u32,
}

#[test]
fn parse_args_accepts_required_and_optional_flags() {
    let cli = parse_args(
        "build_replay_sidecar",
        vec![
            "build_replay_sidecar".to_string(),
            "--input".to_string(),
            "game.json.gz".to_string(),
            "--checkpoint".to_string(),
            "model_base".to_string(),
            "--output".to_string(),
            "out.jsonl".to_string(),
            "--source-version".to_string(),
            "7".to_string(),
            "--min-visits".to_string(),
            "128".to_string(),
            "--hard-state-threshold".to_string(),
            "0.35".to_string(),
            "--max-kl".to_string(),
            "1.5".to_string(),
        ],
    )
    .expect("args should parse");

    assert_eq!(cli.input, PathBuf::from("game.json.gz"));
    assert_eq!(cli.checkpoint, PathBuf::from("model_base"));
    assert_eq!(cli.output, PathBuf::from("out.jsonl"));
    assert_eq!(cli.source_version, 7);
    assert_eq!(cli.min_visits, Some(128));
    assert_eq!(cli.hard_state_threshold, Some(0.35));
    assert_eq!(cli.max_kl, Some(1.5));
}

#[test]
fn parse_args_requires_source_version() {
    let err = parse_args(
        "build_replay_sidecar",
        vec![
            "build_replay_sidecar".to_string(),
            "--input".to_string(),
            "game.json".to_string(),
            "--checkpoint".to_string(),
            "model_base".to_string(),
            "--output".to_string(),
            "out.jsonl".to_string(),
        ],
    )
    .expect_err("missing source version should fail");

    assert!(err.contains("Usage:"));
}

#[test]
fn parse_args_rejects_invalid_numeric_flags() {
    let err = parse_args(
        "build_replay_sidecar",
        vec![
            "build_replay_sidecar".to_string(),
            "--input".to_string(),
            "game.json".to_string(),
            "--checkpoint".to_string(),
            "model_base".to_string(),
            "--output".to_string(),
            "out.jsonl".to_string(),
            "--source-version".to_string(),
            "abc".to_string(),
        ],
    )
    .expect_err("invalid source-version should fail");

    assert!(err.contains("invalid --source-version"));
}

#[test]
fn build_exit_config_applies_cli_overrides() {
    let cli = ReplaySidecarCli {
        input: PathBuf::from("game.json"),
        checkpoint: PathBuf::from("model_base"),
        output: PathBuf::from("out.jsonl"),
        source_version: 1,
        min_visits: Some(64),
        hard_state_threshold: Some(0.2),
        max_kl: Some(0.75),
    };

    let cfg = build_exit_config(&cli);
    assert_eq!(cfg.min_visits, 64);
    assert_eq!(cfg.hard_state_threshold, 0.2);
    assert_eq!(cfg.safety_valve_max_kl, 0.75);
}

#[test]
fn write_jsonl_and_report_persist_expected_content() {
    let output_path = unique_path("rows", "jsonl");
    let report_output = unique_path("report-base", "jsonl");

    let records = [
        DummyRecord {
            id: 1,
            label: "alpha",
        },
        DummyRecord {
            id: 2,
            label: "beta",
        },
    ];
    write_jsonl(&output_path, &records).expect("jsonl should be written");
    let written = fs::read_to_string(&output_path).expect("jsonl readable");
    assert!(written.contains("\"id\":1"));
    assert!(written.contains("\"label\":\"beta\""));

    let report = DummyRecord {
        id: 9,
        label: "report",
    };
    let report_path = write_report(&report_output, &report).expect("report should be written");
    assert_eq!(report_path, report_output.with_extension("report.json"));
    let report_json = fs::read_to_string(&report_path).expect("report readable");
    assert!(report_json.contains("\"id\": 9"));
    assert!(report_json.contains("\"label\": \"report\""));

    fs::remove_file(output_path).expect("remove jsonl");
    fs::remove_file(report_path).expect("remove report");
}

#[test]
fn source_net_hash_is_stable_for_same_checkpoint_path() {
    let path = PathBuf::from("/tmp/model_base");
    assert_eq!(
        source_net_hash_from_checkpoint(&path),
        source_net_hash_from_checkpoint(&path)
    );
}

#[test]
fn usage_mentions_required_and_optional_flags() {
    let usage_text = usage("build_replay_sidecar");
    assert!(usage_text.contains("--input <replay.json|replay.json.gz>"));
    assert!(usage_text.contains("--checkpoint <model_base>"));
    assert!(usage_text.contains("[--max-kl <f32>]"));
}

#[test]
fn source_identity_from_input_uses_file_name() {
    let input = Path::new("nested/game.json.gz");
    let identity = source_identity_from_input(input).expect("utf-8 file name should work");
    assert_eq!(identity, "game.json.gz");
}

#[cfg(unix)]
#[test]
fn source_identity_from_input_rejects_non_utf8_file_name() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let input = PathBuf::from(OsString::from_vec(vec![0xff, b'.', b'j', b's', b'o', b'n']));
    let err = source_identity_from_input(&input).expect_err("non-utf-8 should fail");
    assert!(err.contains("invalid replay filename"));
}

#[test]
fn source_identity_from_input_rejects_paths_without_file_names() {
    let err =
        source_identity_from_input(Path::new(".")).expect_err("path without file name should fail");
    assert!(err.contains("invalid replay filename ."));
}

#[test]
fn success_message_includes_record_count_output_and_report_path() {
    let output = Path::new("out.jsonl");
    let report = Path::new("out.report.json");
    let message = success_message(7, output, report, "replay ExIt records");

    assert!(message.contains("Wrote 7 replay ExIt records"));
    assert!(message.contains("out.jsonl"));
    assert!(message.contains("out.report.json"));
}

#[test]
fn write_sidecar_with_forwards_wrapper_inputs_and_formats_success_message() {
    let input = Path::new("replays/game.json.gz");
    let checkpoint = Path::new("checkpoints/model_base");
    let output = Path::new("out.jsonl");
    let expected_report_path = output.with_extension("report.json");
    let seen_identity = RefCell::new(None::<String>);
    let seen_hash = RefCell::new(None::<u64>);
    let seen_version = RefCell::new(None::<u32>);
    let seen_jsonl_write = RefCell::new(None::<(PathBuf, Vec<WrapperRecord>)>);
    let seen_report_write = RefCell::new(None::<(PathBuf, WrapperReport)>);

    let summary = write_sidecar_with(
        ReplaySidecarWriteRequest {
            input,
            checkpoint,
            output,
            source_version: 7,
            lane_name: "replay ExIt sidecar",
            record_label: "replay ExIt records",
        },
        |source_identity, source_net_hash, source_version| {
            *seen_identity.borrow_mut() = Some(source_identity.to_string());
            *seen_hash.borrow_mut() = Some(source_net_hash);
            *seen_version.borrow_mut() = Some(source_version);
            Ok((
                vec![WrapperRecord { id: 1 }, WrapperRecord { id: 2 }],
                WrapperReport { labels_emitted: 2 },
            ))
        },
        |path, records| {
            *seen_jsonl_write.borrow_mut() = Some((path.to_path_buf(), records.to_vec()));
            Ok(())
        },
        |path, report| {
            *seen_report_write.borrow_mut() = Some((path.to_path_buf(), report.clone()));
            Ok(expected_report_path.clone())
        },
    )
    .expect("wrapper path should succeed");

    assert_eq!(seen_identity.borrow().as_deref(), Some("game.json.gz"));
    assert_eq!(
        *seen_hash.borrow(),
        Some(source_net_hash_from_checkpoint(checkpoint))
    );
    assert_eq!(*seen_version.borrow(), Some(7));
    assert_eq!(
        *seen_jsonl_write.borrow(),
        Some((
            output.to_path_buf(),
            vec![WrapperRecord { id: 1 }, WrapperRecord { id: 2 }]
        ))
    );
    assert_eq!(
        *seen_report_write.borrow(),
        Some((output.to_path_buf(), WrapperReport { labels_emitted: 2 }))
    );
    assert_eq!(
        summary,
        success_message(
            2,
            output,
            expected_report_path.as_path(),
            "replay ExIt records"
        )
    );
}

#[test]
fn write_sidecar_with_wraps_generator_errors() {
    let err = write_sidecar_with::<WrapperRecord, WrapperReport, _, _, _>(
        ReplaySidecarWriteRequest {
            input: Path::new("game.json.gz"),
            checkpoint: Path::new("model_base"),
            output: Path::new("out.jsonl"),
            source_version: 1,
            lane_name: "replay delta_q sidecar",
            record_label: "replay delta_q records",
        },
        |_source_identity, _source_net_hash, _source_version| {
            Err("kaboom while generating".to_string())
        },
        |_path, _records| panic!("jsonl writer should not run after generator failure"),
        |_path, _report| panic!("report writer should not run after generator failure"),
    )
    .expect_err("generator failure should bubble up");

    assert_eq!(
        err,
        "failed to generate replay delta_q sidecar: kaboom while generating"
    );
}

#[test]
fn write_sidecar_with_propagates_jsonl_write_errors_without_running_report_writer() {
    let report_writer_called = RefCell::new(false);

    let err = write_sidecar_with(
        ReplaySidecarWriteRequest {
            input: Path::new("game.json.gz"),
            checkpoint: Path::new("model_base"),
            output: Path::new("out.jsonl"),
            source_version: 1,
            lane_name: "replay ExIt sidecar",
            record_label: "replay ExIt records",
        },
        |_source_identity, _source_net_hash, _source_version| {
            Ok((
                vec![WrapperRecord { id: 1 }],
                WrapperReport { labels_emitted: 1 },
            ))
        },
        |_path, _records| Err("disk full".to_string()),
        |_path, _report| {
            *report_writer_called.borrow_mut() = true;
            Ok(PathBuf::from("out.report.json"))
        },
    )
    .expect_err("jsonl writer failure should bubble up");

    assert_eq!(err, "disk full");
    assert!(!*report_writer_called.borrow());
}

#[test]
fn write_sidecar_with_propagates_report_write_errors() {
    let err = write_sidecar_with(
        ReplaySidecarWriteRequest {
            input: Path::new("game.json.gz"),
            checkpoint: Path::new("model_base"),
            output: Path::new("out.jsonl"),
            source_version: 1,
            lane_name: "replay ExIt sidecar",
            record_label: "replay ExIt records",
        },
        |_source_identity, _source_net_hash, _source_version| {
            Ok((
                vec![WrapperRecord { id: 1 }],
                WrapperReport { labels_emitted: 1 },
            ))
        },
        |_path, _records| Ok(()),
        |_path, _report| Err("report write failed".to_string()),
    )
    .expect_err("report writer failure should bubble up");

    assert_eq!(err, "report write failed");
}

#[test]
fn write_sidecar_with_formats_zero_record_success_message() {
    let output = Path::new("out.jsonl");
    let expected_report_path = PathBuf::from("out.report.json");

    let summary = write_sidecar_with(
        ReplaySidecarWriteRequest {
            input: Path::new("game.json.gz"),
            checkpoint: Path::new("model_base"),
            output,
            source_version: 1,
            lane_name: "replay ExIt sidecar",
            record_label: "replay ExIt records",
        },
        |_source_identity, _source_net_hash, _source_version| {
            Ok((
                Vec::<WrapperRecord>::new(),
                WrapperReport { labels_emitted: 0 },
            ))
        },
        |_path, records| {
            assert!(records.is_empty());
            Ok(())
        },
        |_path, report| {
            assert_eq!(*report, WrapperReport { labels_emitted: 0 });
            Ok(expected_report_path.clone())
        },
    )
    .expect("empty sidecar should still succeed");

    assert_eq!(
        summary,
        success_message(
            0,
            output,
            expected_report_path.as_path(),
            "replay ExIt records"
        )
    );
}

#[test]
fn parse_args_rejects_unknown_flag_with_usage() {
    let err = parse_args(
        "build_replay_sidecar",
        vec![
            "build_replay_sidecar".to_string(),
            "--bogus".to_string(),
            "value".to_string(),
        ],
    )
    .expect_err("unknown flags should fail");

    assert!(err.starts_with("Usage:"));
}

#[test]
fn parse_args_reports_missing_optional_flag_values() {
    let err = parse_args(
        "build_replay_sidecar",
        vec![
            "build_replay_sidecar".to_string(),
            "--input".to_string(),
            "game.json".to_string(),
            "--checkpoint".to_string(),
            "model_base".to_string(),
            "--output".to_string(),
            "out.jsonl".to_string(),
            "--source-version".to_string(),
            "1".to_string(),
            "--min-visits".to_string(),
        ],
    )
    .expect_err("missing min-visits value should fail");

    assert!(err.contains("missing value for --min-visits"));
}

#[test]
fn build_exit_config_keeps_phase3_defaults_without_overrides() {
    let cli = ReplaySidecarCli {
        input: PathBuf::from("game.json"),
        checkpoint: PathBuf::from("model_base"),
        output: PathBuf::from("out.jsonl"),
        source_version: 1,
        min_visits: None,
        hard_state_threshold: None,
        max_kl: None,
    };

    let cfg = build_exit_config(&cli);
    let default_cfg = ExitConfig::default_phase3();
    assert_eq!(cfg.min_visits, default_cfg.min_visits);
    assert_eq!(cfg.hard_state_threshold, default_cfg.hard_state_threshold);
    assert_eq!(cfg.safety_valve_max_kl, default_cfg.safety_valve_max_kl);
}

#[test]
fn write_jsonl_handles_empty_records_and_read_events_reports_missing_file() {
    let output_path = unique_path("empty-rows", "jsonl");
    write_jsonl::<DummyRecord>(&output_path, &[]).expect("empty jsonl should be created");
    assert_eq!(
        fs::read_to_string(&output_path).expect("empty jsonl readable"),
        ""
    );

    let missing_path = unique_path("missing", "json");
    let err = read_events(&missing_path).expect_err("missing replay should fail");
    assert!(err.contains("failed to open replay"));
    assert!(err.contains(&missing_path.display().to_string()));

    fs::remove_file(output_path).expect("remove empty jsonl");
}

#[test]
fn write_report_surfaces_directory_write_failures() {
    let output_path = unique_path("report-dir", "jsonl");
    let report_path = output_path.with_extension("report.json");
    fs::create_dir(&report_path).expect("create report-path directory");

    let report = DummyRecord { id: 1, label: "x" };
    let err =
        write_report(&output_path, &report).expect_err("directory-backed report path should fail");
    assert!(err.contains("failed to write report"));

    fs::remove_dir(&report_path).expect("remove temp dir");
}
