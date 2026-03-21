use std::path::{Path, PathBuf};

use burn::backend::libtorch::LibTorchDevice;
#[path = "common/replay_sidecar_common.rs"]
mod replay_sidecar_common;

use hydra_train::training::replay_delta_q::replay_delta_q_records_for_identity;
use serde::Serialize;

use self::replay_sidecar_common::{
    build_exit_config, load_model, parse_args, read_events, source_net_hash_from_checkpoint,
    write_jsonl, write_report,
};

fn validate_source_version(source_version: u32) -> Result<(), String> {
    if source_version != 1 {
        return Err(format!(
            "unsupported --source-version {}; replay delta_q sidecars currently require source-version 1 to match train-side lookup",
            source_version
        ));
    }
    Ok(())
}

fn source_identity_from_input(input: &std::path::Path) -> Result<&str, String> {
    input
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid replay filename {}", input.display()))
}

fn success_message(record_count: usize, output: &Path, report_path: &Path) -> String {
    format!(
        "Wrote {record_count} replay delta_q records to {} (report: {})",
        output.display(),
        report_path.display()
    )
}

fn write_sidecar_with<Record, Report, Generate, WriteJsonl, WriteReport>(
    input: &Path,
    checkpoint: &Path,
    output: &Path,
    source_version: u32,
    generate: Generate,
    write_jsonl_fn: WriteJsonl,
    write_report_fn: WriteReport,
) -> Result<String, String>
where
    Record: Serialize,
    Report: Serialize,
    Generate: FnOnce(&str, u64, u32) -> Result<(Vec<Record>, Report), String>,
    WriteJsonl: FnOnce(&Path, &[Record]) -> Result<(), String>,
    WriteReport: FnOnce(&Path, &Report) -> Result<PathBuf, String>,
{
    let source_identity = source_identity_from_input(input)?;
    let source_net_hash = source_net_hash_from_checkpoint(checkpoint);
    let (records, report) = generate(source_identity, source_net_hash, source_version)
        .map_err(|err| format!("failed to generate replay delta_q sidecar: {err}"))?;

    write_jsonl_fn(output, &records)?;
    let report_path = write_report_fn(output, &report)?;

    Ok(success_message(records.len(), output, &report_path))
}

fn run() -> Result<(), String> {
    let cli = parse_args("build_replay_delta_q_sidecar", std::env::args())?;
    validate_source_version(cli.source_version)?;
    let device = LibTorchDevice::Cpu;
    let model = load_model(&cli.checkpoint, &device)?;
    let exit_cfg = build_exit_config(&cli);

    let events = read_events(&cli.input)?;
    let summary = write_sidecar_with(
        &cli.input,
        &cli.checkpoint,
        &cli.output,
        cli.source_version,
        |source_identity, source_net_hash, source_version| {
            replay_delta_q_records_for_identity(
                source_identity,
                &events,
                &model,
                &device,
                &exit_cfg,
                source_net_hash,
                source_version,
            )
            .map_err(|err| err.to_string())
        },
        write_jsonl,
        write_report,
    )?;

    println!("{summary}");
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        source_identity_from_input, success_message, validate_source_version, write_sidecar_with,
    };
    use serde::Serialize;
    use std::cell::RefCell;
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::{Path, PathBuf};

    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct DummyRecord {
        id: u32,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct DummyReport {
        labels_emitted: u32,
    }

    #[test]
    fn delta_q_source_version_one_is_accepted() {
        validate_source_version(1).expect("source-version 1 should be accepted");
    }

    #[test]
    fn delta_q_source_version_other_values_are_rejected() {
        let err = validate_source_version(2).expect_err("non-1 source-version should fail");
        assert!(err.contains("source-version 1"));
    }

    #[test]
    fn source_identity_from_input_returns_filename_for_regular_paths() {
        let path = Path::new("/tmp/replay-001.json");
        assert_eq!(source_identity_from_input(path), Ok("replay-001.json"));
    }

    #[test]
    fn source_identity_from_input_rejects_non_utf8_filename() {
        let non_utf8 = OsString::from_vec(vec![0x66, 0x6f, 0x80]);
        let path = std::path::PathBuf::from(non_utf8);
        let err = source_identity_from_input(&path).expect_err("non-utf8 filename should fail");
        assert!(err.contains("invalid replay filename"));
    }

    #[test]
    fn source_identity_from_input_rejects_paths_without_file_names() {
        let err = source_identity_from_input(Path::new("."))
            .expect_err("path without file name should fail");

        assert!(err.contains("invalid replay filename ."));
    }

    #[test]
    fn success_message_includes_record_count_output_and_report_path() {
        let output = Path::new("out.jsonl");
        let report = Path::new("out.report.json");
        let message = success_message(7, output, report);

        assert!(message.contains("Wrote 7 replay delta_q records"));
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
        let seen_jsonl_write = RefCell::new(None::<(PathBuf, Vec<DummyRecord>)>);
        let seen_report_write = RefCell::new(None::<(PathBuf, DummyReport)>);

        let summary = write_sidecar_with(
            input,
            checkpoint,
            output,
            7,
            |source_identity, source_net_hash, source_version| {
                *seen_identity.borrow_mut() = Some(source_identity.to_string());
                *seen_hash.borrow_mut() = Some(source_net_hash);
                *seen_version.borrow_mut() = Some(source_version);
                Ok((
                    vec![DummyRecord { id: 1 }, DummyRecord { id: 2 }],
                    DummyReport { labels_emitted: 2 },
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
            Some(super::source_net_hash_from_checkpoint(checkpoint))
        );
        assert_eq!(*seen_version.borrow(), Some(7));
        assert_eq!(
            *seen_jsonl_write.borrow(),
            Some((
                output.to_path_buf(),
                vec![DummyRecord { id: 1 }, DummyRecord { id: 2 }]
            ))
        );
        assert_eq!(
            *seen_report_write.borrow(),
            Some((output.to_path_buf(), DummyReport { labels_emitted: 2 }))
        );
        assert_eq!(
            summary,
            success_message(2, output, expected_report_path.as_path())
        );
    }

    #[test]
    fn write_sidecar_with_wraps_generator_errors() {
        let err = write_sidecar_with::<DummyRecord, DummyReport, _, _, _>(
            Path::new("game.json.gz"),
            Path::new("model_base"),
            Path::new("out.jsonl"),
            1,
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
    fn write_sidecar_with_propagates_jsonl_and_report_errors() {
        let report_writer_called = RefCell::new(false);

        let err = write_sidecar_with(
            Path::new("game.json.gz"),
            Path::new("model_base"),
            Path::new("out.jsonl"),
            1,
            |_source_identity, _source_net_hash, _source_version| {
                Ok((
                    vec![DummyRecord { id: 1 }],
                    DummyReport { labels_emitted: 1 },
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

        let err = write_sidecar_with(
            Path::new("game.json.gz"),
            Path::new("model_base"),
            Path::new("out.jsonl"),
            1,
            |_source_identity, _source_net_hash, _source_version| {
                Ok((
                    vec![DummyRecord { id: 1 }],
                    DummyReport { labels_emitted: 1 },
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
            Path::new("game.json.gz"),
            Path::new("model_base"),
            output,
            1,
            |_source_identity, _source_net_hash, _source_version| {
                Ok((Vec::<DummyRecord>::new(), DummyReport { labels_emitted: 0 }))
            },
            |_path, records| {
                assert!(records.is_empty());
                Ok(())
            },
            |_path, report| {
                assert_eq!(*report, DummyReport { labels_emitted: 0 });
                Ok(expected_report_path.clone())
            },
        )
        .expect("empty sidecar should still succeed");

        assert_eq!(
            summary,
            success_message(0, output, expected_report_path.as_path())
        );
    }
}
