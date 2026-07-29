//! Shared helpers for the WP-00A golden compatibility suite.
//!
//! Every test drives the real CLI binary (`env!("CARGO_BIN_EXE_*")`) inside a
//! throwaway directory, mirroring how the packager is used in production.
// Each integration-test binary compiles this module independently, so not every
// binary uses every helper.
#![allow(dead_code)]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Absolute path of the packager binary under test.
pub fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_mjai-dataset-packager")
}

/// Path of a committed fixture below `tests/fixtures/`.
pub fn fixture(segments: &[&str]) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("fixtures");
    for segment in segments {
        path.push(segment);
    }
    path
}

/// Bytes of a committed fixture below `tests/fixtures/`.
pub fn fixture_bytes(segments: &[&str]) -> Vec<u8> {
    fs::read(fixture(segments)).expect("fixture file is readable")
}

/// Copy a fixture into `dest` (its full target path, not a directory).
pub fn copy_fixture(segments: &[&str], dest: &Path) {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).expect("staging parent");
    }
    fs::copy(fixture(segments), dest).expect("fixture copy");
}

/// Run the binary with `args` inside `cwd` and capture stdout/stderr.
pub fn run(cwd: &Path, args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .current_dir(cwd)
        .output()
        .expect("spawn mjai-dataset-packager")
}

/// Run `convert <input> <output> [extra...]` inside `cwd`.
pub fn convert(cwd: &Path, input: &str, output: &str, extra: &[&str]) -> Output {
    let mut args = vec!["convert", input, output];
    args.extend_from_slice(extra);
    run(cwd, &args)
}

/// Run `preflight <input> <output>` inside `cwd`.
pub fn preflight(cwd: &Path, input: &str, output: &str) -> Output {
    run(cwd, &["preflight", input, output])
}

pub fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

pub fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Parse the trailing `completed=N, skipped=M` line emitted by `convert`.
pub fn completed_skipped(stderr: &str) -> (u64, u64) {
    const TAG: &str = "completed=";
    let line = stderr
        .lines()
        .rev()
        .find(|line| line.starts_with(TAG))
        .unwrap_or_else(|| panic!("no 'completed=' line in stderr:\n{stderr}"));
    let rest = &line[TAG.len()..];
    let split = rest
        .find(", skipped=")
        .unwrap_or_else(|| panic!("malformed progress line: {line}"));
    let completed: u64 = rest[..split]
        .parse()
        .unwrap_or_else(|error| panic!("bad completed count in {line}: {error}"));
    let skipped: u64 = rest[split + ", skipped=".len()..]
        .parse()
        .unwrap_or_else(|error| panic!("bad skipped count in {line}: {error}"));
    (completed, skipped)
}

/// Extract an integer field from the `mode=...` plan line of preflight/convert.
pub fn plan_field(stderr: &str, field: &str) -> u64 {
    let needle = format!("{field}=");
    let line = stderr
        .lines()
        .find(|line| line.starts_with("mode="))
        .unwrap_or_else(|| panic!("no 'mode=' plan line in stderr:\n{stderr}"));
    let start = line
        .find(&needle)
        .unwrap_or_else(|| panic!("field {field} missing from plan line: {line}"))
        + needle.len();
    let tail = &line[start..];
    let end = tail.find(',').unwrap_or(tail.len());
    tail[..end]
        .trim()
        .parse()
        .unwrap_or_else(|error| panic!("bad value for {field} in {line}: {error}"))
}

/// Decode a zstd frame produced by the packager.
pub fn decode_zstd(frame: &[u8]) -> Vec<u8> {
    let mut decoder = zstd::stream::read::Decoder::new(frame).expect("zstd decoder");
    let mut plain = Vec::new();
    decoder.read_to_end(&mut plain).expect("zstd decode");
    plain
}

/// Recursively list all regular files below `root`, sorted for stable asserts.
pub fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_owned()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("read_dir") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// SHA-256 of a file's bytes, lowercase hex.
pub fn sha256_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut file = fs::File::open(path).expect("open for hashing");
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer).expect("hash read");
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    hasher.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Parses a JSONL manifest into JSON values, one per line.
pub fn manifest_rows(path: &Path) -> Vec<serde_json::Value> {
    let text = fs::read_to_string(path).expect("manifest readable");
    text.lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).expect("manifest row parses"))
    .collect()
}

pub fn zstd_frame(payload: &[u8]) -> Vec<u8> {
    use std::io::Write as _;
    let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 1).expect("encoder");
    encoder.write_all(payload).expect("compress write");
    encoder.finish().expect("finish frame")
}
