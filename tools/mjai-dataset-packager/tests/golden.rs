//! WP-00A golden round-trip coverage: raw JSON compression, `.mjson` naming,
//! precompressed byte-copy, determinism, and silent ignores.

mod common;

use common::{completed_skipped, convert, decode_zstd, fixture_bytes, plan_field, walk_files};
use std::fs;
use tempfile::tempdir;

/// Checklist: raw `.mjai.json` input produces the expected zstd-decoded bytes,
/// and the compressed bytes match the recorded golden frame byte-for-byte.
#[test]
fn raw_mjai_json_compresses_to_recorded_golden_bytes() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["a.mjai.json"], &work.path().join("input/a.mjai.json"));

    let output = convert(work.path(), "input", "out", &[]);
    assert!(output.status.success(), "stderr: {}", common::stderr_of(&output));
    assert_eq!(completed_skipped(&common::stderr_of(&output)), (1, 0));

    let produced = fs::read(work.path().join("out/a.mjai.json.zst")).expect("output");
    let golden = fixture_bytes(&["golden/a.mjai.json.zst"]);
    assert_eq!(
        produced, golden,
        "compressed bytes diverge from recorded golden frame"
    );

    let decoded = decode_zstd(&produced);
    let raw = fixture_bytes(&["a.mjai.json"]);
    assert_eq!(decoded, raw, "decoded output must reproduce the raw input");
}

/// Checklist: raw `.mjson` fixture normalizes to `<stem>.mjai.json.zst`.
#[test]
fn mjson_input_normalizes_to_stem_mjai_json_zst() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["b.mjson"], &work.path().join("input/b.mjson"));

    let output = convert(work.path(), "input", "out", &[]);
    assert!(output.status.success(), "stderr: {}", common::stderr_of(&output));

    let produced = fs::read(work.path().join("out/b.mjai.json.zst")).expect("output");
    assert_eq!(produced, fixture_bytes(&["golden/b.mjai.json.zst"]));
    assert_eq!(decode_zstd(&produced), fixture_bytes(&["b.mjson"]));
    assert!(
        !work.path().join("out/b.mjson.zst").exists(),
        "mjson must not keep its raw extension in the output name"
    );
}

/// Existing-output contract, copy side: a valid `.mjai.json.zst` input is
/// validated and byte-copied, never recompressed.
#[test]
fn precompressed_zstd_input_is_byte_copied_verbatim() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(
        &["c.mjai.json.zst"],
        &work.path().join("input/c.mjai.json.zst"),
    );

    let output = convert(work.path(), "input", "out", &[]);
    assert!(output.status.success(), "stderr: {}", common::stderr_of(&output));

    let produced = fs::read(work.path().join("out/c.mjai.json.zst")).expect("output");
    let input = fixture_bytes(&["c.mjai.json.zst"]);
    assert_eq!(produced, input, "zstd input must be copied, not recompressed");
    assert_eq!(
        decode_zstd(&produced),
        decode_zstd(&input),
        "copy must remain decodable"
    );
}

/// Identical inputs at a fixed level produce identical output bytes across
/// independent runs and output roots.
#[test]
fn conversion_is_deterministic_across_runs() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["a.mjai.json"], &work.path().join("input/a.mjai.json"));

    for root in ["out1", "out2"] {
        let output = convert(work.path(), "input", root, &[]);
        assert!(output.status.success(), "stderr: {}", common::stderr_of(&output));
    }
    let first = fs::read(work.path().join("out1/a.mjai.json.zst")).expect("run 1");
    let second = fs::read(work.path().join("out2/a.mjai.json.zst")).expect("run 2");
    assert_eq!(first, second, "run-to-run bytes differ");
    assert_eq!(first, fixture_bytes(&["golden/a.mjai.json.zst"]));
}

/// Files with unrecognized extensions are silently ignored by directory scans.
#[test]
fn unrecognized_files_are_silently_ignored() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["a.mjai.json"], &work.path().join("input/a.mjai.json"));
    fs::write(work.path().join("input/notes.txt"), b"not a game log\n").expect("notes");

    let output = convert(work.path(), "input", "out", &[]);
    assert!(output.status.success(), "stderr: {}", common::stderr_of(&output));
    assert_eq!(plan_field(&common::stderr_of(&output), "items"), 1);
    assert_eq!(
        walk_files(&work.path().join("out")),
        vec![work.path().join("out/a.mjai.json.zst")],
        "only the recognized input may produce output"
    );
}

/// `preflight` performs the full inspection without creating any output.
#[test]
fn preflight_reports_plan_without_writing() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["a.mjai.json"], &work.path().join("input/a.mjai.json"));

    let output = common::preflight(work.path(), "input", "out");
    assert!(output.status.success(), "stderr: {}", common::stderr_of(&output));
    let stderr = common::stderr_of(&output);
    assert!(stderr.contains("mode=preflight"), "stderr: {stderr}");
    assert_eq!(plan_field(&stderr, "items"), 1);
    assert!(
        !work.path().join("out").exists(),
        "preflight must not create the output root"
    );
}
