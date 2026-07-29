//! WP-00A archive coverage: member layout, traversal hostility, unsupported
//! members, empty archives, and same-stem extracted-directory precedence.

mod common;

use common::{convert, decode_zstd, fixture_bytes, plan_field, walk_files};
use std::fs;
use tempfile::tempdir;

/// Embedded archive payloads; must match the bytes crafted into the fixtures.
const D_G1: &[u8] =
    b"{\"type\":\"start_game\",\"names\":[\"iris\",\"jon\"],\"from\":\"d-archive\"}\n{\"type\":\"end_game\"}\n";
const D_D2: &[u8] = b"{\"type\":\"start_game\",\"names\":[\"kate\",\"liam\"],\"from\":\"d-archive-mjson\"}\n";
const E_WIN_ARCHIVE: &[u8] = b"{\"origin\":\"archive-wins\",\"note\":\"e.tar.zst is authoritative\"}\n";

/// `.tar.zst` with supported and ignored (directory) members: directory entries
/// are skipped, regular members land below the output root under the archive's
/// stem, and decoded bytes match the recorded payloads.
#[test]
fn archive_members_convert_below_their_archive_stem() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["d.tar.zst"], &work.path().join("input/d.tar.zst"));

    let output = convert(work.path(), "input", "out", &[]);
    assert!(output.status.success(), "stderr: {}", common::stderr_of(&output));
    let stderr = common::stderr_of(&output);
    assert_eq!(plan_field(&stderr, "items"), 2, "dir member must be ignored");

    let g1 = fs::read(work.path().join("out/d/games/g1.mjai.json.zst")).expect("g1 output");
    assert_eq!(g1, fixture_bytes(&["golden/d/games/g1.mjai.json.zst"]));
    assert_eq!(decode_zstd(&g1), D_G1);

    let d2 = fs::read(work.path().join("out/d/d2.mjai.json.zst")).expect("d2 output");
    assert_eq!(d2, fixture_bytes(&["golden/d/d2.mjai.json.zst"]));
    assert_eq!(decode_zstd(&d2), D_D2);
}

/// Checklist: member path normalization cannot escape the destination. A
/// `../evil.txt` member fails preflight and leaves zero outputs behind.
#[test]
fn traversal_member_is_rejected_without_writes() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(
        &["hostile.tar.zst"],
        &work.path().join("input/hostile.tar.zst"),
    );

    let output = convert(work.path(), "input", "out", &[]);
    assert!(!output.status.success(), "traversal member must abort");
    let stderr = common::stderr_of(&output);
    assert!(stderr.contains("unsafe archive path"), "stderr: {stderr}");
    assert!(stderr.contains("../evil.txt"), "stderr: {stderr}");
    assert!(
        !work.path().join("out").exists(),
        "no output root may be created"
    );
    assert!(
        !work.path().join("evil.txt").exists(),
        "nothing may escape the destination"
    );
}

/// Checklist: unsupported regular members fail closed and produce no outputs.
#[test]
fn unsupported_regular_member_fails_with_no_outputs() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(
        &["unsupported.tar.zst"],
        &work.path().join("input/unsupported.tar.zst"),
    );

    let output = convert(work.path(), "input", "out", &[]);
    assert!(!output.status.success());
    let stderr = common::stderr_of(&output);
    assert!(
        stderr.contains("unsupported regular archive member"),
        "stderr: {stderr}"
    );
    assert!(stderr.contains("README.txt"), "stderr: {stderr}");
    assert!(!work.path().join("out").exists(), "zero writes expected");
}

/// An archive without any usable MJAI JSON member is rejected outright.
#[test]
fn empty_archive_is_rejected() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["empty.tar.zst"], &work.path().join("input/empty.tar.zst"));

    let output = convert(work.path(), "input", "out", &[]);
    assert!(!output.status.success());
    let stderr = common::stderr_of(&output);
    assert!(
        stderr.contains("contains no MJAI JSON files"),
        "stderr: {stderr}"
    );
    assert!(!work.path().join("out").exists(), "zero writes expected");
}

/// Checklist: same-stem precedence — when `X.tar.zst` exists next to extracted
/// directory `X/`, the ENTIRE directory is ignored in favor of the archive.
#[test]
fn archive_takes_precedence_over_same_stem_extracted_directory() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["e.tar.zst"], &work.path().join("input/e.tar.zst"));
    // Extracted-dir contents deliberately disagree with the archive payload.
    common::copy_fixture(
        &["e/win.mjai.json"],
        &work.path().join("input/e/win.mjai.json"),
    );
    common::copy_fixture(
        &["e/extra.mjai.json"],
        &work.path().join("input/e/extra.mjai.json"),
    );

    let output = convert(work.path(), "input", "out", &[]);
    assert!(output.status.success(), "stderr: {}", common::stderr_of(&output));
    let stderr = common::stderr_of(&output);
    assert_eq!(plan_field(&stderr, "ignored_extracted_files"), 2);
    assert_eq!(plan_field(&stderr, "ignored_extracted_bytes"), 177);
    assert_eq!(plan_field(&stderr, "items"), 1, "extracted dir fully ignored");

    let files = walk_files(&work.path().join("out"));
    assert_eq!(
        files,
        vec![work.path().join("out/e/win.mjai.json.zst")],
        "only the archive-derived output may exist"
    );
    let win = fs::read(&files[0]).expect("win output");
    assert_eq!(win, fixture_bytes(&["golden/e/win.mjai.json.zst"]));
    assert_eq!(
        decode_zstd(&win),
        E_WIN_ARCHIVE,
        "archive content wins over extracted-dir content"
    );
}
