//! WP-00A CLI compatibility contract: help text, collision abort, repeated-run
//! skip semantics, regeneration of corrupt outputs, and bound enforcement.

mod common;

use common::{completed_skipped, convert, fixture_bytes, plan_field};
use std::fs;
use tempfile::tempdir;

/// Checklist: CLI help remains byte-identical to the recorded post-relocation
/// help. The clap usage line contains the bare crate name only, so the bytes
/// are independent of how the executable is addressed.
const EXPECTED_HELP: &str = r#"Restart-safe MJAI JSON zstd dataset packager

Usage: mjai-dataset-packager <COMMAND>

Commands:
  preflight  Inspect all inputs, reject collisions, and check output capacity without writing
  convert    Perform conversion after the same complete preflight
  help       Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
"#;

#[test]
fn help_output_is_byte_identical_to_recorded_help() {
    let elsewhere = tempdir().expect("workdir");
    for cwd in [elsewhere.path(), std::path::Path::new(env!("CARGO_MANIFEST_DIR"))] {
        let output = common::run(cwd, &["--help"]);
        assert!(output.status.success(), "--help must exit 0");
        assert_eq!(
            common::stdout_of(&output),
            EXPECTED_HELP,
            "help text changed; update EXPECTED_HELP only via the compatibility record"
        );
    }
}

#[test]
fn missing_subcommand_exits_with_code_2() {
    let work = tempdir().expect("workdir");
    let output = common::run(work.path(), &[]);
    assert_eq!(
        output.status.code(),
        Some(2),
        "clap usage error must exit 2"
    );
    assert!(common::stderr_of(&output).contains("Usage:"));
}

/// Checklist: same-stem `.mjai.json` + `.mjson` collide on one output name and
/// abort with zero writes.
#[test]
fn same_stem_collision_aborts_without_writes() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(
        &["collision/x.mjai.json"],
        &work.path().join("input/x.mjai.json"),
    );
    common::copy_fixture(&["collision/x.mjson"], &work.path().join("input/x.mjson"));

    let output = convert(work.path(), "input", "out", &[]);
    assert!(!output.status.success(), "collision must fail");
    let stderr = common::stderr_of(&output);
    assert!(stderr.contains("duplicate output collision"), "stderr: {stderr}");
    assert!(stderr.contains("x.mjai.json.zst"), "stderr: {stderr}");
    assert!(!work.path().join("out").exists(), "zero writes expected");
}

/// Checklist: repeated run proves the skip contract — complete outputs (>= 8
/// bytes with zstd magic) are neither rewritten nor counted toward capacity.
#[test]
fn repeated_run_skips_complete_outputs_without_rewriting() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(
        &["c.mjai.json.zst"],
        &work.path().join("input/c.mjai.json.zst"),
    );

    let first = convert(work.path(), "input", "out", &[]);
    assert!(first.status.success(), "stderr: {}", common::stderr_of(&first));
    assert_eq!(completed_skipped(&common::stderr_of(&first)), (1, 0));

    let out_path = work.path().join("out/c.mjai.json.zst");
    let before = fs::read(&out_path).expect("first-run output");
    let before_mtime = fs::metadata(&out_path)
        .expect("stat output")
        .modified()
        .expect("mtime");

    let second = convert(work.path(), "input", "out", &[]);
    assert!(second.status.success(), "stderr: {}", common::stderr_of(&second));
    let stderr = common::stderr_of(&second);
    assert_eq!(completed_skipped(&stderr), (0, 1), "complete output is skipped");
    assert_eq!(
        plan_field(&stderr, "conservative_bytes"),
        0,
        "skipped outputs are excluded from capacity math"
    );
    assert_eq!(fs::read(&out_path).expect("second-run output"), before);
    assert_eq!(
        fs::metadata(&out_path).expect("stat output").modified().expect("mtime"),
        before_mtime,
        "skip must not rewrite the file"
    );
}

/// Existing-output contract, regenerate side: an existing file without the
/// zstd magic is treated as garbage and rebuilt from the input.
#[test]
fn garbage_existing_output_is_regenerated() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(
        &["c.mjai.json.zst"],
        &work.path().join("input/c.mjai.json.zst"),
    );
    fs::create_dir_all(work.path().join("out")).expect("out dir");
    fs::write(work.path().join("out/c.mjai.json.zst"), b"garbage\n").expect("garbage");

    let output = convert(work.path(), "input", "out", &[]);
    assert!(output.status.success(), "stderr: {}", common::stderr_of(&output));
    assert_eq!(completed_skipped(&common::stderr_of(&output)), (1, 0));
    assert_eq!(
        fs::read(work.path().join("out/c.mjai.json.zst")).expect("regenerated"),
        fixture_bytes(&["c.mjai.json.zst"]),
        "garbage output must be replaced by the byte-copy"
    );
}

/// Checklist: conservative preflight bound — an input above --max-item-bytes
/// aborts before anything is written.
#[test]
fn oversized_input_aborts_preflight_with_zero_outputs() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["a.mjai.json"], &work.path().join("input/a.mjai.json"));

    let output = convert(work.path(), "input", "out", &["--max-item-bytes", "8"]);
    assert!(!output.status.success());
    let stderr = common::stderr_of(&output);
    assert!(
        stderr.contains("input exceeds --max-item-bytes"),
        "stderr: {stderr}"
    );
    assert!(!work.path().join("out").exists(), "zero writes expected");

    // The same input passes preflight untouched when the bound is generous.
    let ok = common::preflight(work.path(), "input", "out");
    assert!(ok.status.success(), "stderr: {}", common::stderr_of(&ok));
    assert!(!work.path().join("out").exists());
}

#[test]
fn memory_limit_below_max_item_is_rejected() {
    let work = tempdir().expect("workdir");
    common::copy_fixture(&["a.mjai.json"], &work.path().join("input/a.mjai.json"));

    let output = convert(
        work.path(),
        "input",
        "out",
        &["--max-item-bytes", "64", "--memory-limit-bytes", "32"],
    );
    assert!(!output.status.success());
    let stderr = common::stderr_of(&output);
    assert!(
        stderr.contains("--memory-limit-bytes must be at least --max-item-bytes"),
        "stderr: {stderr}"
    );
    assert!(!work.path().join("out").exists());
}

/// A symlink sitting at an expected output path is refused, never followed.
#[cfg(unix)]
#[test]
fn symlink_output_is_refused() {
    use std::os::unix::fs::symlink;
    let work = tempdir().expect("workdir");
    common::copy_fixture(
        &["c.mjai.json.zst"],
        &work.path().join("input/c.mjai.json.zst"),
    );
    fs::create_dir_all(work.path().join("out")).expect("out dir");
    symlink("/etc/hostname", work.path().join("out/c.mjai.json.zst")).expect("symlink");

    let output = convert(work.path(), "input", "out", &[]);
    assert!(!output.status.success());
    let stderr = common::stderr_of(&output);
    assert!(stderr.contains("refusing symlink output"), "stderr: {stderr}");
    let meta = fs::symlink_metadata(work.path().join("out/c.mjai.json.zst")).expect("link");
    assert!(meta.file_type().is_symlink(), "link must be left untouched");
}
