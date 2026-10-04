//! rl/package_verify: pure-Rust work-package completion verifier.
//!
//! Resolves plan C02: the documented
//! `pixi run hydra2 work-package verify <WP-ID> --artifact-root ROOT`
//! (BUILD_EXECUTION_PLAN.md §1 line 58) had no implementation. This module is
//! the specified verifier — actual validation, never a mocked command. It
//! checks the universal completion envelope, canonical SHA-256 identity,
//! immutable record paths, output/log ingestion, package run reports,
//! dependency records, the package command matrix, and the exit disposition,
//! then atomically publishes the mutable index. A missing record, schema,
//! report, or dependency is an explicit `blocked`/`failed` verdict, never
//! `passed`.
//!
//! Authority (read, never redefined):
//! - Completion payload schema, envelope
//!   (`artifact_type="hydra2.work_package_completion"`,
//!   `schema_version="1.0.0"`), immutable `work_packages/<WP>/<artifact_id>.json`
//!   paths, mutable `work_packages/index.json`, and the package command matrix
//!   live in BUILD_EXECUTION_PLAN.md §1 (lines 34-74).
//! - Envelope/canonical/SHA/atomic-publication semantics are SPEC §§2.2-2.3.
//!   Identity math reuses `hydra_feed::canon`, digest text reuses
//!   `trajectory`, the root-relative locator guard reuses
//!   `snapshot::join_root`, and the raw-root discipline mirrors
//!   `python/hydra2/training/rust_stream.py:41-54`.
//! - Package run reports are the `tests/conftest.py` writer:
//!   `reports/<WP>/<run-id>/report.json` with
//!   `kind="hydra2.contract_test_report"`.
//! - No prior completion implementation exists in-tree (the old
//!   `src/hydra2/completion.py` cited by plan status notes is absent), so no
//!   second convention is introduced: this file defers to the documents above.
//!
//! Index schema (new, minimal, exactly what §1 line 57 specifies — one
//! current record hash per package plus retained superseded hashes):
//! ```json
//! {"schema_version": "1.0.0",
//!  "packages": {"WP-11": {"current": "sha256:<hex>",
//!                         "superseded": ["sha256:<hex>"]}}}
//! ```
//!
//! Command-log rule (logs are explicit outputs): every `commands[].log_sha256`
//! MUST equal the `sha256` of one listed `outputs[]` entry whose file streams
//! to that digest. A missing or tampered log fails/blocks, never succeeds.
//! Logs are thereby actually verified, never manufactured, using only
//! specified fields.
//!
//! Verdict law:
//! - `failed`: an integrity contradiction (envelope/schema/canonical/file
//!   digest mismatch, path escape, nonzero exit, failing report or test, a
//!   `passed` claim with blockers or without its matrix gate), or a record
//!   that truthfully reports `failed`.
//! - `blocked`: a missing prerequisite (unknown WP id, unsupported entry
//!   authority for the package or one of its dependencies, no record, no
//!   index entry with several candidates, no passing report, missing
//!   dependency record, non-`passed` dependency, missing environment
//!   manifest, empty commands/tests), or a record that truthfully reports
//!   `blocked`. Missing ledger never fabricates qualification.
//! - `passed`: only a structurally valid `passed` record whose every gate is
//!   green. `failed` dominates `blocked` when both are present.
//!
//! Dependency edges (`entry_deps`) come straight from the plan's `**Entry:**`
//! WP lists only (WP-02B-D expands to WP-02B/C/D). Entry text naming no WP
//! ids, and packages with no Entry WP list at all, carry NO edges and verify
//! as `blocked` (plan defect) — never an empty pass. Blocked-as-defect set:
//! WP-05B/C, WP-08B/C, WP-09A (no Entry WP list); WP-09B1-B10 ("Candidate 3
//! frozen control"), WP-09D ("WP-08"), WP-09E ("Wave 8"), WP-12 (registry
//! description). A dependency whose own entry authority is absent blocks its
//! dependents the same way.
//! Dependency checks are depth-1 status gates (entry authority + envelope +
//! filename digest + `status == "passed"`); full verification happens when
//! the dependency itself is verified, which keeps the graph free of
//! recursion cycles.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::rl::snapshot::join_root;
use crate::rl::trajectory::{DIGEST_PREFIX, TapeError, digest_text, parse_digest_text, sha256_raw};

/// Completion envelope identity (BUILD §1 lines 54-56).
pub const COMPLETION_ARTIFACT_TYPE: &str = "hydra2.work_package_completion";
/// Completion envelope version (BUILD §1 line 55).
pub const COMPLETION_SCHEMA_VERSION: &str = "1.0.0";
/// Mutable completion-index version (same closed line as the envelope).
pub const INDEX_SCHEMA_VERSION: &str = "1.0.0";
/// Package run report kind written by `tests/conftest.py`.
pub const REPORT_KIND: &str = "hydra2.contract_test_report";

const CANON_RECORD: &str = "package_verify:record";
const CANON_RECEIPT: &str = "package_verify:receipt";
const CANON_INDEX: &str = "package_verify:index";
const STREAM_CHUNK: usize = 65536;
const RAW_ROOT_KEYS: [&str; 2] = ["HYDRA2_DATA_ROOT", "HYDRA2_TENHOU_MOUNT"];

fn pv_err(detail: String) -> TapeError {
    TapeError(format!("package_verify {detail}"))
}

/// Entry-edge dependencies straight from the plan's `**Entry:**` WP lists.
/// `None` means no usable entry authority (no `**Entry:**` WP list, or entry
/// text that names no WP ids) — a plan defect that blocks, never an empty
/// pass. Blocked-as-defect set: WP-05B/C, WP-08B/C, WP-09A (no Entry WP
/// list); WP-09B1-B10 ("Candidate 3 frozen control"), WP-09D ("WP-08"),
/// WP-09E ("Wave 8"), WP-12 (registry description).
fn entry_deps(work_package: &str) -> Option<&'static [&'static str]> {
    Some(match work_package {
        "WP-00A" => &[],
        "WP-00B" => &["WP-00A"],
        "WP-01" => &["WP-00A"],
        "WP-02A" => &["WP-01"],
        "WP-02B" => &["WP-02A"],
        "WP-02C" => &["WP-02A"],
        "WP-02D" => &["WP-02A", "WP-02B", "WP-02C"],
        "WP-03A" => &["WP-01", "WP-02D"],
        "WP-03B" => &["WP-02B", "WP-02C", "WP-02D"],
        "WP-03C" => &["WP-01", "WP-02D"],
        "WP-04A" => &["WP-03A"],
        "WP-04B" => &["WP-00B", "WP-02D", "WP-03A", "WP-04A"],
        "WP-04C" => &["WP-03C", "WP-04A"],
        "WP-05A" => &["WP-04A", "WP-04B"],
        "WP-05B" | "WP-05C" | "WP-08B" | "WP-08C" | "WP-09A" => return None,
        "WP-06" => &["WP-03B", "WP-04A", "WP-05C"],
        "WP-07A" => &["WP-03B", "WP-04A", "WP-05C"],
        "WP-07B" => &["WP-05C", "WP-06"],
        "WP-08A" => &["WP-02B", "WP-02C", "WP-02D", "WP-04A", "WP-05C", "WP-07A"],
        "WP-09B1" | "WP-09B2" | "WP-09B3" | "WP-09B4" | "WP-09B5" | "WP-09B6" | "WP-09B7"
        | "WP-09B8" | "WP-09B9" | "WP-09B10" | "WP-09D" | "WP-09E" | "WP-12" => return None,
        "WP-09C" => &["WP-09A", "WP-09B9"],
        "WP-10" => &["WP-09C", "WP-09D", "WP-09E", "WP-12"],
        "WP-11" => &["WP-05C", "WP-06"],
        "WP-13" => &["WP-04B", "WP-07A", "WP-10"],
        _ => return None,
    })
}

/// Required test-command tokens from the §1 package command matrix (BUILD
/// lines 61-74). Every token must appear in SOME recorded argv; when
/// `bind_package` holds, the WP id must appear in the same argv (the matrix
/// spells `--package <WP-ID>` on those rows; WP-00A/B use Cargo, WP-01 runs
/// the Pixi tasks, WP-12 runs bare `test-analysis`).
struct MatrixGate {
    tokens: &'static [&'static str],
    bind_package: bool,
}

fn matrix_gate(work_package: &str) -> Option<MatrixGate> {
    let tokens: &'static [&'static str] = match work_package {
        "WP-00A" | "WP-00B" => &["cargo"],
        "WP-01" => &["pixi"],
        "WP-02A" | "WP-02B" | "WP-02C" | "WP-02D" => &["test-contracts"],
        "WP-03A" | "WP-03B" | "WP-03C" | "WP-04A" | "WP-04C" => &["test-conformance"],
        "WP-04B" | "WP-06" => &["test-integration"],
        "WP-05A" | "WP-05B" | "WP-05C" | "WP-07B" | "WP-10" | "WP-11" => &["test-training"],
        "WP-07A" => &["test-unit", "test-integration"],
        "WP-08A" | "WP-08B" | "WP-08C" | "WP-09A" | "WP-09B1" | "WP-09B2" | "WP-09B3"
        | "WP-09B4" | "WP-09B5" | "WP-09B6" | "WP-09B7" | "WP-09B8" | "WP-09B9" | "WP-09B10"
        | "WP-09C" | "WP-09D" | "WP-09E" | "WP-13" => &["test-search"],
        "WP-12" => &["test-analysis"],
        _ => return None,
    };
    let bind_package = !matches!(work_package, "WP-00A" | "WP-00B" | "WP-01" | "WP-12");
    Some(MatrixGate {
        tokens,
        bind_package,
    })
}

/// Verdict class of one receipt check: `Structural`/`Fail` findings force
/// `failed`, `Block` findings force `blocked` (failed dominates).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Gate {
    Structural,
    Fail,
    Block,
}

struct Check {
    name: String,
    gate: Gate,
    ok: bool,
    detail: String,
}

fn note(checks: &mut Vec<Check>, name: String, gate: Gate, ok: bool, detail: String) {
    checks.push(Check {
        name,
        gate,
        ok,
        detail,
    });
}

fn valid_wp_shape(work_package: &str) -> bool {
    if work_package.len() < 5 || work_package.len() > 12 {
        return false;
    }
    let Some(rest) = work_package.strip_prefix("WP-") else {
        return false;
    };
    let (digits, suffix) = match rest.find(|c: char| !c.is_ascii_digit()) {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    if digits.is_empty() || digits.len() > 2 {
        return false;
    }
    match suffix {
        "" | "A" | "B" | "C" | "D" => true,
        _ => {
            suffix.starts_with('B')
                && suffix.len() <= 3
                && suffix[1..].bytes().all(|c| c.is_ascii_digit())
        }
    }
}

fn valid_utc(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() < 20 || bytes[bytes.len() - 1] != b'Z' {
        return false;
    }
    for (i, c) in text.bytes().take(19).enumerate() {
        if matches!(i, 0 | 1 | 2 | 3 | 5 | 6 | 8 | 9 | 11 | 12 | 14 | 15 | 17 | 18) {
            if !c.is_ascii_digit() {
                return false;
            }
        } else {
            let expect = match i {
                4 | 7 => b'-',
                10 => b'T',
                13 | 16 => b':',
                _ => return false,
            };
            if c != expect {
                return false;
            }
        }
    }
    let rest = &text[19..text.len() - 1];
    if rest.is_empty() {
        return true;
    }
    rest.starts_with('.') && rest.len() > 1 && rest[1..].bytes().all(|c| c.is_ascii_digit())
}

fn canon(value: &Value, record: &str) -> Result<Vec<u8>, TapeError> {
    hydra_feed::canon::canonical_bytes_value(value, record)
        .map_err(|e| pv_err(format!("JCS emit failed for {record}: {e:?}")))
}

fn sha_text(bytes: &[u8]) -> String {
    digest_text(&sha256_raw(bytes))
}

/// Stream a file into its `sha256:<hex>` digest without holding the bytes.
fn stream_digest(path: &Path) -> Result<String, TapeError> {
    let file = File::open(path).map_err(|e| pv_err(format!("file unreadable {}: {e}", path.display())))?;
    let mut reader = std::io::BufReader::with_capacity(STREAM_CHUNK, file);
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    let mut chunk = vec![0u8; STREAM_CHUNK];
    loop {
        let n = reader
            .read(&mut chunk)
            .map_err(|e| pv_err(format!("file stream failed {}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
    }
    Ok(digest_text(&hasher.finalize().into()))
}

fn read_json_file(path: &Path) -> Result<(Vec<u8>, Value), TapeError> {
    let bytes = std::fs::read(path).map_err(|e| pv_err(format!("file unreadable {}: {e}", path.display())))?;
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|e| pv_err(format!("file is not JSON {}: {e}", path.display())))?;
    Ok((bytes, value))
}

/// Sorted `*.json` files directly under `dir` (ignores dotfiles and temp
/// files so interrupted writers never read as records). Missing dir is empty.
fn scan_json_files(dir: &Path) -> Result<Vec<PathBuf>, TapeError> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(pv_err(format!(
                "directory unreadable {}: {e}",
                dir.display()
            )));
        }
    };
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| pv_err(format!("directory entry failed {}: {e}", dir.display())))?;
        let path = entry.path();
        let name = path.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned());
        if !name.ends_with(".json") || name.starts_with('.') || name.contains(".tmp") {
            continue;
        }
        if path.is_file() {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

/// Raw-root discipline, mirroring `rust_stream.py:41-54`: the artifact root
/// must sit outside the raw corpus roots in both directions.
fn assert_root_outside_raw_roots(artifact_root: &str) -> Result<(), TapeError> {
    for key in RAW_ROOT_KEYS {
        let raw = std::env::var(key).unwrap_or_default();
        if raw.is_empty() {
            continue;
        }
        if artifact_root.starts_with(raw.as_str()) || raw.starts_with(artifact_root) {
            return Err(pv_err(format!(
                "{key} raw root must sit outside HYDRA2_ARTIFACT_ROOT: artifact={artifact_root} raw={raw}"
            )));
        }
    }
    Ok(())
}

/// Atomic publish: unique same-directory temp (O_CREAT|O_EXCL) + fsync +
/// rename + parent dir fsync (SPEC §2.3 ordering, local copy — the snapshot
/// helper is private to its owner).
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), TapeError> {
    let parent = path
        .parent()
        .ok_or_else(|| pv_err("publish path has no parent".to_string()))?;
    std::fs::create_dir_all(parent)
        .map_err(|e| pv_err(format!("publish dir create failed {}: {e}", parent.display())))?;
    for attempt in 0..16u32 {
        let tmp = path.with_extension(format!("tmp-{}-{attempt}", std::process::id()));
        let mut file = match std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(pv_err(format!("publish temp failed {}: {e}", tmp.display())));
            }
        };
        let write_result = (|| -> Result<(), TapeError> {
            use std::io::Write as _;
            file.write_all(bytes)
                .map_err(|e| pv_err(format!("publish write failed {}: {e}", tmp.display())))?;
            file.sync_all()
                .map_err(|e| pv_err(format!("publish fsync failed {}: {e}", tmp.display())))?;
            drop(file);
            std::fs::rename(&tmp, path)
                .map_err(|e| pv_err(format!("publish rename failed {}: {e}", tmp.display())))?;
            #[cfg(unix)]
            {
                let dir = File::open(parent)
                    .map_err(|e| pv_err(format!("publish dir open failed {}: {e}", parent.display())))?;
                dir.sync_all()
                    .map_err(|e| pv_err(format!("publish dir fsync failed {}: {e}", parent.display())))?;
            }
            Ok(())
        })();
        if write_result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        return write_result;
    }
    Err(pv_err(format!(
        "publish temp allocation failed {}",
        path.display()
    )))
}

fn index_path(root: &Path) -> PathBuf {
    root.join("work_packages").join("index.json")
}

fn record_dir(root: &Path, work_package: &str) -> PathBuf {
    root.join("work_packages").join(work_package)
}

/// Load the mutable index: `None` when absent, `blocked`-class detail when
/// present but malformed (authority unusable, never guessed).
fn load_index(root: &Path) -> Result<Option<Value>, String> {
    let path = index_path(root);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("index unreadable {}: {e}", path.display())),
    };
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("index is not JSON {}: {e}", path.display()))?;
    let doc = value.as_object().ok_or_else(|| format!("index must be an object {}", path.display()))?;
    if doc.get("schema_version").and_then(Value::as_str) != Some(INDEX_SCHEMA_VERSION) {
        return Err(format!("index schema_version must be {INDEX_SCHEMA_VERSION:?}"));
    }
    if !doc.get("packages").is_some_and(Value::is_object) {
        return Err("index packages must be an object".to_string());
    }
    Ok(Some(value))
}

fn index_current(index: &Value, work_package: &str) -> Option<String> {
    index
        .get("packages")?
        .get(work_package)?
        .get("current")?
        .as_str()
        .map(str::to_string)
}

/// Upsert the verified record digest as current (old current retained under
/// `superseded`); other packages pass through verbatim. Atomic + fsynced.
fn upsert_index(root: &Path, index: Option<&Value>, work_package: &str, digest: &str) -> Result<Value, TapeError> {
    let mut packages = Map::new();
    if let Some(doc) = index {
        if let Some(existing) = doc.get("packages").and_then(Value::as_object) {
            for (name, entry) in existing {
                if name != work_package {
                    packages.insert(name.clone(), entry.clone());
                }
            }
        }
    }
    let mut superseded: Vec<Value> = Vec::new();
    if let Some(doc) = index {
        if let Some(entry) = doc.get("packages").and_then(|p| p.get(work_package)) {
            if let Some(current) = entry.get("current").and_then(Value::as_str) {
                if current != digest {
                    superseded.push(Value::String(current.to_string()));
                }
            }
            if let Some(old) = entry.get("superseded").and_then(Value::as_array) {
                for item in old {
                    if item.as_str() != Some(digest) && !superseded.contains(item) {
                        superseded.push(item.clone());
                    }
                }
            }
        }
    }
    let mut entry = Map::new();
    entry.insert("current".to_string(), Value::String(digest.to_string()));
    entry.insert("superseded".to_string(), Value::Array(superseded));
    packages.insert(work_package.to_string(), Value::Object(entry));
    let mut doc = Map::new();
    doc.insert(
        "schema_version".to_string(),
        Value::String(INDEX_SCHEMA_VERSION.to_string()),
    );
    doc.insert("packages".to_string(), Value::Object(packages));
    let bytes = canon(&Value::Object(doc.clone()), CANON_INDEX)?;
    atomic_write(&index_path(root), &bytes)?;
    Ok(Value::Object(doc))
}

struct RecordView<'a> {
    status: &'a str,
    inputs: &'a Vec<Value>,
    outputs: &'a Vec<Value>,
    commands: &'a Vec<Value>,
    tests: &'a Vec<Value>,
    environment_manifest_sha256: &'a str,
    blockers: &'a Vec<Value>,
    deviations: &'a Vec<Value>,
}

/// Full payload schema (BUILD §1 lines 36-49). Digest shapes and timestamp
/// shapes fail here; existence of the referenced bytes is a later gate.
fn validate_payload<'a>(payload: &'a Map<String, Value>, work_package: &str) -> Result<RecordView<'a>, String> {
    let status = payload
        .get("status")
        .and_then(Value::as_str)
        .ok_or_else(|| "payload status must be a string".to_string())?;
    if !["passed", "failed", "blocked"].contains(&status) {
        return Err(format!("payload status must be passed|failed|blocked, got {status:?}"));
    }
    let package = payload
        .get("work_package")
        .and_then(Value::as_str)
        .ok_or_else(|| "payload work_package must be a string".to_string())?;
    if package != work_package {
        return Err(format!("payload work_package {package:?} != requested {work_package:?}"));
    }
    let array = |key: &str| -> Result<&'a Vec<Value>, String> {
        payload
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("payload {key} must be an array"))
    };
    let inputs = array("inputs")?;
    let outputs = array("outputs")?;
    let commands = array("commands")?;
    let tests = array("tests")?;
    let blockers = array("blockers")?;
    let deviations = array("deviations")?;
    for entry in inputs {
        let item = entry.as_object().ok_or_else(|| "inputs[] must be objects".to_string())?;
        if !item.get("id").and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
            return Err("inputs[].id must be a nonempty string".to_string());
        }
        let sha = item.get("sha256").and_then(Value::as_str).unwrap_or("");
        parse_digest_text(sha).map_err(|e| format!("inputs[].sha256 invalid: {}", e.0))?;
    }
    for entry in outputs {
        let item = entry.as_object().ok_or_else(|| "outputs[] must be objects".to_string())?;
        if !item.get("path").and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
            return Err("outputs[].path must be a nonempty string".to_string());
        }
        let sha = item.get("sha256").and_then(Value::as_str).unwrap_or("");
        parse_digest_text(sha).map_err(|e| format!("outputs[].sha256 invalid: {}", e.0))?;
    }
    for entry in commands {
        let item = entry.as_object().ok_or_else(|| "commands[] must be objects".to_string())?;
        let argv = item.get("argv").and_then(Value::as_array).ok_or_else(|| "commands[].argv must be an array".to_string())?;
        if argv.is_empty() || argv.iter().any(|a| !a.as_str().is_some_and(|s| !s.is_empty())) {
            return Err("commands[].argv must be a nonempty string array".to_string());
        }
        item.get("exit_code").and_then(Value::as_u64).ok_or_else(|| "commands[].exit_code must be a u64".to_string())?;
        let log = item.get("log_sha256").and_then(Value::as_str).unwrap_or("");
        parse_digest_text(log).map_err(|e| format!("commands[].log_sha256 invalid: {}", e.0))?;
    }
    for entry in tests {
        let item = entry.as_object().ok_or_else(|| "tests[] must be objects".to_string())?;
        if !item.get("id").and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
            return Err("tests[].id must be a nonempty string".to_string());
        }
        if !item.get("result").and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
            return Err("tests[].result must be a nonempty string".to_string());
        }
    }
    let environment_manifest_sha256 = payload
        .get("environment_manifest_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "payload environment_manifest_sha256 must be a string".to_string())?;
    parse_digest_text(environment_manifest_sha256)
        .map_err(|e| format!("environment_manifest_sha256 invalid: {}", e.0))?;
    for key in ["started_at_utc", "finished_at_utc"] {
        let text = payload.get(key).and_then(Value::as_str).unwrap_or("");
        if !valid_utc(text) {
            return Err(format!("payload {key} must be RFC 3339 UTC, got {text:?}"));
        }
    }
    Ok(RecordView {
        status,
        inputs,
        outputs,
        commands,
        tests,
        environment_manifest_sha256,
        blockers,
        deviations,
    })
}

/// Envelope validation (SPEC §2.2 via BUILD §1 line 55): exact type/version,
/// closed fields for `exact` compatibility, string-keyed object payload.
fn validate_envelope(doc: &Map<String, Value>) -> Result<&Map<String, Value>, String> {
    if doc.get("artifact_type").and_then(Value::as_str) != Some(COMPLETION_ARTIFACT_TYPE) {
        return Err(format!("artifact_type must be {COMPLETION_ARTIFACT_TYPE:?}"));
    }
    if doc.get("schema_version").and_then(Value::as_str) != Some(COMPLETION_SCHEMA_VERSION) {
        return Err(format!("schema_version must be {COMPLETION_SCHEMA_VERSION:?}"));
    }
    let compatibility = doc.get("compatibility").and_then(Value::as_str).unwrap_or("");
    if !["exact", "backward_read"].contains(&compatibility) {
        return Err("compatibility must be exact|backward_read".to_string());
    }
    if compatibility == "exact" {
        let mut unknown: Vec<&String> = doc
            .keys()
            .filter(|k| !["artifact_type", "schema_version", "compatibility", "payload"].contains(&k.as_str()))
            .collect();
        unknown.sort();
        if !unknown.is_empty() {
            return Err(format!("unknown envelope field(s) {unknown:?} under exact compatibility"));
        }
    }
    doc.get("payload")
        .and_then(Value::as_object)
        .ok_or_else(|| "payload must be a string-keyed JSON object".to_string())
}

/// Depth-1 dependency gate: envelope + filename digest + `status == "passed"`.
/// Full verification happens when the dependency itself is verified.
fn check_dependency(root: &Path, index: Option<&Value>, dep: &str) -> Check {
    let name = format!("dep:{dep}");
    let fail = |detail: String| Check {
        name: name.clone(),
        gate: Gate::Structural,
        ok: false,
        detail,
    };
    let block = |detail: String| Check {
        name: name.clone(),
        gate: Gate::Block,
        ok: false,
        detail,
    };
    let current = index.and_then(|doc| index_current(doc, dep));
    let Some(current) = current else {
        return block(format!("no current record for {dep} in work_packages/index.json"));
    };
    if entry_deps(dep).is_none() {
        return block(format!("unsupported prerequisite: no Entry authority for {dep} (plan defect)"));
    }
    if parse_digest_text(&current).is_err() {
        return fail(format!("index current for {dep} is not digest text: {current:?}"));
    }
    let hex = current.strip_prefix(DIGEST_PREFIX).unwrap_or(&current);
    let path = record_dir(root, dep).join(format!("{hex}.json"));
    let (raw, value) = match read_json_file(&path) {
        Ok(pair) => pair,
        Err(e) => return block(format!("dependency record unreadable {}: {}", path.display(), e.0)),
    };
    let doc = match value.as_object() {
        Some(doc) => doc,
        None => return fail(format!("dependency record must be an object {}", path.display())),
    };
    if let Err(detail) = validate_envelope(doc) {
        return fail(format!("dependency envelope invalid for {dep}: {detail}"));
    }
    let canonical = match canon(&value, CANON_RECORD) {
        Ok(bytes) => bytes,
        Err(e) => return fail(format!("dependency JCS failed for {dep}: {}", e.0)),
    };
    if raw != canonical || sha_text(&canonical) != current {
        return fail(format!("dependency digest mismatch for {dep}"));
    }
    let status = doc
        .get("payload")
        .and_then(|p| p.get("status"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if status != "passed" {
        return block(format!("dependency {dep} status is {status:?}, want passed"));
    }
    Check {
        name,
        gate: Gate::Block,
        ok: true,
        detail: format!("{dep} current {current} passed"),
    }
}

struct ReportIngest {
    runs: usize,
    passing: usize,
    problems: Vec<String>,
}

/// Ingest every `reports/<WP>/<run-id>/report.json`: kind/package match,
/// canonical-bytes equality (conftest writes canonical bytes), and the
/// passed/failed/totals shape. Files are read, never manufactured.
fn ingest_reports(root: &Path, work_package: &str) -> ReportIngest {
    let mut out = ReportIngest {
        runs: 0,
        passing: 0,
        problems: Vec::new(),
    };
    let reports = root.join("reports").join(work_package);
    let run_dirs = match std::fs::read_dir(&reports) {
        Ok(entries) => entries,
        Err(_) => return out,
    };
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in run_dirs.flatten() {
        let report = entry.path().join("report.json");
        if report.is_file() {
            paths.push(report);
        }
    }
    paths.sort();
    for path in paths {
        let (raw, value) = match read_json_file(&path) {
            Ok(pair) => pair,
            Err(e) => {
                out.problems.push(format!("{}: {}", path.display(), e.0));
                continue;
            }
        };
        out.runs += 1;
        let doc = match value.as_object() {
            Some(doc) => doc,
            None => {
                out.problems.push(format!("{}: report must be an object", path.display()));
                continue;
            }
        };
        if doc.get("kind").and_then(Value::as_str) != Some(REPORT_KIND) {
            out.problems.push(format!("{}: kind must be {REPORT_KIND:?}", path.display()));
            continue;
        }
        if doc.get("work_package").and_then(Value::as_str) != Some(work_package) {
            out.problems.push(format!("{}: work_package mismatch", path.display()));
            continue;
        }
        match canon(&value, CANON_RECORD) {
            Ok(expect) if expect == raw => {}
            _ => {
                out.problems.push(format!("{}: report bytes are not canonical JSON", path.display()));
                continue;
            }
        }
        let totals = doc.get("totals");
        let failed = totals.and_then(|t| t.get("failed")).and_then(Value::as_u64).unwrap_or(1);
        let fixtures = doc.get("fixtures").and_then(Value::as_object);
        let fixtures_ok = fixtures.is_some_and(|f| !f.is_empty());
        if doc.get("status").and_then(Value::as_str) == Some("passed") && failed == 0 && fixtures_ok {
            out.passing += 1;
        } else {
            out.problems.push(format!("{}: report is not a passing gate", path.display()));
        }
    }
    out
}

fn receipt_bytes(
    work_package: &str,
    verdict: &str,
    record_digest: Option<&str>,
    record_status: Option<&str>,
    index_current: Option<&str>,
    report_runs: u64,
    checks: &[Check],
) -> Result<Vec<u8>, TapeError> {
    let checks_json: Vec<Value> = checks
        .iter()
        .map(|c| {
            let mut item = Map::new();
            item.insert("name".to_string(), Value::String(c.name.clone()));
            item.insert("ok".to_string(), Value::Bool(c.ok));
            item.insert("detail".to_string(), Value::String(c.detail.clone()));
            Value::Object(item)
        })
        .collect();
    let mut doc = Map::new();
    doc.insert("work_package".to_string(), Value::String(work_package.to_string()));
    doc.insert("verdict".to_string(), Value::String(verdict.to_string()));
    doc.insert(
        "record_digest".to_string(),
        record_digest.map_or(Value::Null, |s| Value::String(s.to_owned())),
    );
    doc.insert(
        "record_status".to_string(),
        record_status.map_or(Value::Null, |s| Value::String(s.to_owned())),
    );
    doc.insert(
        "index_current".to_string(),
        index_current.map_or(Value::Null, |s| Value::String(s.to_owned())),
    );
    doc.insert("report_runs".to_string(), Value::Number(report_runs.into()));
    doc.insert("checks".to_string(), Value::Array(checks_json));
    canon(&Value::Object(doc), CANON_RECEIPT)
}

/// Verify one work-package completion record against `artifact_root`.
///
/// `Ok` bytes are always the canonical receipt (`verdict` is
/// passed|failed|blocked); `Err` is fail-closed argument/infra trouble only
/// (bad shapes, relative root, raw-root overlap, index publication IO).
pub fn verify_work_package(work_package: &str, artifact_root: &str) -> Result<Vec<u8>, TapeError> {
    if !valid_wp_shape(work_package) {
        return Err(pv_err(format!("work_package has bad shape: {work_package:?}")));
    }
    if artifact_root.is_empty() || artifact_root.len() > 4096 {
        return Err(pv_err("artifact-root has bad length".to_string()));
    }
    if artifact_root.contains('\\') {
        return Err(pv_err("artifact-root must use '/' separators".to_string()));
    }
    let root = Path::new(artifact_root);
    if !root.is_absolute() {
        return Err(pv_err(format!("artifact-root must be absolute: {artifact_root:?}")));
    }
    assert_root_outside_raw_roots(artifact_root)?;

    let mut checks: Vec<Check> = Vec::new();
    let deps = entry_deps(work_package);
    let gate = matrix_gate(work_package);
    if deps.is_none() || gate.is_none() {
        note(
            &mut checks,
            "work_package_supported".to_string(),
            Gate::Block,
            false,
            format!("{work_package} has no BUILD §1 matrix/entry row (plan defect)"),
        );
        return receipt_bytes(work_package, "blocked", None, None, None, 0, &checks);
    }
    let deps = deps.unwrap_or(&[]);
    let gate = gate.unwrap_or(MatrixGate {
        tokens: &[],
        bind_package: false,
    });

    if !root.is_dir() {
        note(
            &mut checks,
            "artifact_root".to_string(),
            Gate::Block,
            false,
            format!("artifact root is not a directory: {artifact_root:?}"),
        );
        return receipt_bytes(work_package, "blocked", None, None, None, 0, &checks);
    }
    note(
        &mut checks,
        "artifact_root".to_string(),
        Gate::Block,
        true,
        format!("{artifact_root} is an absolute directory outside raw roots"),
    );

    // Mutable index: absent is fine (adoption below); malformed is a Block
    // finding — the authority is unusable, never guessed.
    let mut index_ok = true;
    let index: Option<Value> = match load_index(root) {
        Ok(index) => index,
        Err(detail) => {
            index_ok = false;
            note(&mut checks, "index_readable".to_string(), Gate::Block, false, detail);
            None
        }
    };
    if index_ok {
        note(&mut checks, "index_readable".to_string(), Gate::Block, true, "index.json parses with schema 1.0.0".to_string());
    }

    // Record discovery: the index current wins; without one, exactly one
    // record file is adopted, zero is blocked, several are blocked as
    // ambiguous (the index exists to disambiguate).
    let current = index.as_ref().and_then(|doc| index_current(doc, work_package));
    let record_path: Option<PathBuf> = if let Some(current) = current.as_deref() {
        if parse_digest_text(current).is_err() {
            note(
                &mut checks,
                "record_found".to_string(),
                Gate::Structural,
                false,
                format!("index current for {work_package} is not digest text: {current:?}"),
            );
            return receipt_bytes(work_package, "failed", None, None, current.into(), 0, &checks);
        }
        let hex = current.strip_prefix(DIGEST_PREFIX).unwrap_or(current);
        let path = record_dir(root, work_package).join(format!("{hex}.json"));
        if !path.is_file() {
            note(
                &mut checks,
                "record_found".to_string(),
                Gate::Block,
                false,
                format!("indexed record is missing: {}", path.display()),
            );
            return receipt_bytes(work_package, "blocked", None, None, current.into(), 0, &checks);
        }
        Some(path)
    } else {
        match scan_json_files(&record_dir(root, work_package)) {
            Ok(files) if files.len() == 1 => files.into_iter().next(),
            Ok(files) if files.is_empty() => {
                note(
                    &mut checks,
                    "record_found".to_string(),
                    Gate::Block,
                    false,
                    format!("no completion record under {}", record_dir(root, work_package).display()),
                );
                return receipt_bytes(work_package, "blocked", None, None, None, 0, &checks);
            }
            Ok(files) => {
                note(
                    &mut checks,
                    "record_found".to_string(),
                    Gate::Block,
                    false,
                    format!("{} record files with no index current (ambiguous)", files.len()),
                );
                return receipt_bytes(work_package, "blocked", None, None, None, 0, &checks);
            }
            Err(e) => {
                note(&mut checks, "record_found".to_string(), Gate::Block, false, e.0);
                return receipt_bytes(work_package, "blocked", None, None, None, 0, &checks);
            }
        }
    };
    let record_path = record_path.unwrap_or_default();
    note(
        &mut checks,
        "record_found".to_string(),
        Gate::Block,
        true,
        format!("record at {}", record_path.display()),
    );

    let (raw, value) = match read_json_file(&record_path) {
        Ok(pair) => pair,
        Err(e) => {
            note(&mut checks, "record_json".to_string(), Gate::Structural, false, e.0);
            return receipt_bytes(work_package, "failed", None, None, current.as_deref(), 0, &checks);
        }
    };
    let doc = match value.as_object() {
        Some(doc) => doc,
        None => {
            note(
                &mut checks,
                "record_json".to_string(),
                Gate::Structural,
                false,
                "record must be a JSON object".to_string(),
            );
            return receipt_bytes(work_package, "failed", None, None, current.as_deref(), 0, &checks);
        }
    };
    note(&mut checks, "record_json".to_string(), Gate::Structural, true, "record parses as a JSON object".to_string());

    let payload = match validate_envelope(doc) {
        Ok(payload) => payload,
        Err(detail) => {
            note(&mut checks, "envelope".to_string(), Gate::Structural, false, detail);
            return receipt_bytes(work_package, "failed", None, None, current.as_deref(), 0, &checks);
        }
    };
    note(
        &mut checks,
        "envelope".to_string(),
        Gate::Structural,
        true,
        format!("{COMPLETION_ARTIFACT_TYPE} v{COMPLETION_SCHEMA_VERSION}"),
    );

    // Canonical SHA: raw bytes must equal the re-emitted canonical bytes and
    // hash to the filename digest (tamper fails here); the index current must
    // agree when present.
    let canonical = match canon(&value, CANON_RECORD) {
        Ok(bytes) => bytes,
        Err(e) => {
            note(&mut checks, "record_canonical_sha".to_string(), Gate::Structural, false, e.0.clone());
            return Err(e);
        }
    };
    let digest = sha_text(&canonical);
    let stem = record_path.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned());
    let expect = format!("{DIGEST_PREFIX}{stem}");
    if raw != canonical {
        note(
            &mut checks,
            "record_canonical_sha".to_string(),
            Gate::Structural,
            false,
            "record bytes are not canonical JSON".to_string(),
        );
        return receipt_bytes(work_package, "failed", Some(&digest), None, current.as_deref(), 0, &checks);
    }
    if digest != expect {
        note(
            &mut checks,
            "record_canonical_sha".to_string(),
            Gate::Structural,
            false,
            format!("record digest {digest} != filename {expect}"),
        );
        return receipt_bytes(work_package, "failed", Some(&digest), None, current.as_deref(), 0, &checks);
    }
    if let Some(current) = current.as_deref() {
        if digest != current {
            note(
                &mut checks,
                "record_canonical_sha".to_string(),
                Gate::Structural,
                false,
                format!("record digest {digest} != index current {current}"),
            );
            return receipt_bytes(work_package, "failed", Some(&digest), None, Some(current), 0, &checks);
        }
    }
    note(
        &mut checks,
        "record_canonical_sha".to_string(),
        Gate::Structural,
        true,
        format!("{digest} matches file and index"),
    );

    let view = match validate_payload(payload, work_package) {
        Ok(view) => view,
        Err(detail) => {
            note(&mut checks, "payload_schema".to_string(), Gate::Structural, false, detail);
            return receipt_bytes(work_package, "failed", Some(&digest), None, current.as_deref(), 0, &checks);
        }
    };
    note(&mut checks, "payload_schema".to_string(), Gate::Structural, true, "universal record schema holds".to_string());

    // Outputs: every locator stays under the root (escape fails closed via
    // join_root) and every file streams to its recorded digest.
    let mut output_shas: Vec<String> = Vec::new();
    let mut output_bad: Option<String> = None;
    for entry in view.outputs.iter() {
        let item = entry.as_object().cloned().unwrap_or_default();
        let locator = item.get("path").and_then(Value::as_str).unwrap_or("");
        let recorded = item.get("sha256").and_then(Value::as_str).unwrap_or("");
        let path = match join_root(root, locator) {
            Ok(path) => path,
            Err(e) => {
                output_bad = Some(format!("output {locator:?} escapes the root: {}", e.0));
                break;
            }
        };
        if !path.is_file() {
            output_bad = Some(format!("output is missing: {}", path.display()));
            break;
        }
        match stream_digest(&path) {
            Ok(actual) if actual == recorded => output_shas.push(recorded.to_string()),
            Ok(actual) => {
                output_bad = Some(format!("output diverges {}: file {actual} != recorded {recorded}", path.display()));
                break;
            }
            Err(e) => {
                output_bad = Some(e.0);
                break;
            }
        }
    }
    if let Some(detail) = output_bad {
        note(&mut checks, "outputs".to_string(), Gate::Structural, false, detail);
        return receipt_bytes(work_package, "failed", Some(&digest), Some(view.status), current.as_deref(), 0, &checks);
    }
    note(
        &mut checks,
        "outputs".to_string(),
        Gate::Structural,
        true,
        format!("{} output file(s) stream to their digests", output_shas.len()),
    );

    // Structurally valid records are published to the index (fsync-durable)
    // whatever the verdict: the index tracks identity, not qualification.
    let fresh_index = upsert_index(root, index.as_ref(), work_package, &digest)?;
    let fresh_current = fresh_index
        .get("packages")
        .and_then(|p| p.get(work_package))
        .and_then(|e| e.get("current"))
        .and_then(Value::as_str)
        .unwrap_or(&digest)
        .to_string();
    note(
        &mut checks,
        "index_published".to_string(),
        Gate::Structural,
        true,
        format!("current {fresh_current}"),
    );
    let index_after: Option<Value> = Some(fresh_index);

    // --- gates below only downgrade a `passed` claim; a truthful
    // --- `failed`/`blocked` record keeps its disposition (evidence appended).
    for &dep in deps {
        checks.push(check_dependency(root, index_after.as_ref(), dep));
    }

    let ingest = ingest_reports(root, work_package);
    if ingest.runs == 0 {
        note(
            &mut checks,
            "reports".to_string(),
            Gate::Block,
            false,
            format!("no reports under {}", root.join("reports").join(work_package).display()),
        );
    } else if ingest.passing == 0 {
        note(
            &mut checks,
            "reports".to_string(),
            Gate::Fail,
            false,
            format!("{} report(s) ingested, none a passing gate: {}", ingest.runs, ingest.problems.join("; ")),
        );
    } else {
        note(
            &mut checks,
            "reports".to_string(),
            Gate::Fail,
            true,
            format!("{}/{} report(s) are passing gates", ingest.passing, ingest.runs),
        );
    }

    if view.commands.is_empty() {
        note(&mut checks, "commands".to_string(), Gate::Block, false, "record carries no commands".to_string());
    } else {
        let mut command_bad: Option<String> = None;
        let joined: Vec<String> = view
            .commands
            .iter()
            .filter_map(|c| {
                c.get("argv")?.as_array().map(|argv| {
                    argv.iter().filter_map(|a| a.as_str()).collect::<Vec<_>>().join(" ")
                })
            })
            .collect();
        for entry in view.commands.iter() {
            let item = entry.as_object().cloned().unwrap_or_default();
            let exit_code = item.get("exit_code").and_then(Value::as_u64).unwrap_or(1);
            let log = item.get("log_sha256").and_then(Value::as_str).unwrap_or("");
            if exit_code != 0 {
                command_bad = Some(format!("command exits {exit_code}, want 0"));
                break;
            }
            if !output_shas.iter().any(|sha| sha == log) {
                command_bad = Some(format!("command log {log} is not among verified outputs"));
                break;
            }
        }
        if command_bad.is_none() {
            for &token in gate.tokens {
                let hit = joined.iter().any(|argv| argv.contains(token) && (!gate.bind_package || argv.contains(work_package)));
                if !hit {
                    command_bad = Some(format!("no command runs the required {token} gate for {work_package}"));
                    break;
                }
            }
        }
        match command_bad {
            Some(detail) => note(&mut checks, "commands".to_string(), Gate::Fail, false, detail),
            None => note(
                &mut checks,
                "commands".to_string(),
                Gate::Fail,
                true,
                format!("{} command(s) exit 0 with ingested logs and the matrix gate", view.commands.len()),
            ),
        }
    }

    let env_path = root.join("environment").join("environment-manifest.json");
    if !env_path.is_file() {
        note(
            &mut checks,
            "environment_manifest".to_string(),
            Gate::Block,
            false,
            format!("environment manifest is missing: {}", env_path.display()),
        );
    } else {
        match stream_digest(&env_path) {
            Ok(actual) if actual == view.environment_manifest_sha256 => note(
                &mut checks,
                "environment_manifest".to_string(),
                Gate::Block,
                true,
                format!("manifest streams to {actual}"),
            ),
            Ok(actual) => note(
                &mut checks,
                "environment_manifest".to_string(),
                Gate::Structural,
                false,
                format!("manifest {actual} != recorded {}", view.environment_manifest_sha256),
            ),
            Err(e) => note(&mut checks, "environment_manifest".to_string(), Gate::Structural, false, e.0),
        }
    }

    if view.tests.is_empty() {
        note(&mut checks, "tests".to_string(), Gate::Block, false, "record carries no tests".to_string());
    } else {
        let bad = view.tests.iter().any(|t| t.get("result").and_then(Value::as_str) != Some("passed"));
        if bad {
            note(&mut checks, "tests".to_string(), Gate::Fail, false, "a recorded test did not pass".to_string());
        } else {
            note(
                &mut checks,
                "tests".to_string(),
                Gate::Fail,
                true,
                format!("{} recorded test(s) passed", view.tests.len()),
            );
        }
    }

    if view.status == "passed" && !view.blockers.is_empty() {
        note(&mut checks, "blockers".to_string(), Gate::Fail, false, "passed record carries blockers".to_string());
    } else {
        note(
            &mut checks,
            "blockers".to_string(),
            Gate::Fail,
            true,
            format!("{} blocker(s), {} deviation(s)", view.blockers.len(), view.deviations.len()),
        );
    }

    let verdict = if checks.iter().any(|c| !c.ok && matches!(c.gate, Gate::Structural | Gate::Fail)) {
        "failed"
    } else if view.status == "failed" {
        "failed"
    } else if view.status == "blocked" {
        "blocked"
    } else if checks.iter().any(|c| !c.ok) {
        "blocked"
    } else {
        "passed"
    };
    note(
        &mut checks,
        "disposition".to_string(),
        Gate::Block,
        verdict == view.status,
        format!("record {0} verifies as {verdict}", view.status),
    );
    // A truthful non-passed record keeps its disposition unless integrity
    // failed above; a passed claim stands only when every gate is green.
    let verdict = if matches!(view.status, "failed" | "blocked")
        && !checks.iter().any(|c| !c.ok && matches!(c.gate, Gate::Structural | Gate::Fail))
    {
        view.status
    } else {
        verdict
    };
    receipt_bytes(
        work_package,
        verdict,
        Some(&digest),
        Some(view.status),
        Some(fresh_current.as_str()),
        ingest.runs as u64,
        &checks,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wp_shape_accepts_matrix_ids() {
        for wp in [
            "WP-00A", "WP-01", "WP-02D", "WP-07A", "WP-09B1", "WP-09B9", "WP-09B10", "WP-09C", "WP-10",
            "WP-11", "WP-12", "WP-13",
        ] {
            assert!(valid_wp_shape(wp), "{wp}");
        }
    }

    #[test]
    fn wp_shape_rejects_junk() {
        for wp in ["", "WP-", "wp-11", "WP-9B100", "WP-11/../x", "../WP-11", "WP-11 "] {
            assert!(!valid_wp_shape(wp), "{wp}");
        }
    }

    #[test]
    fn utc_shape_holds() {
        assert!(valid_utc("2026-09-30T00:00:00Z"));
        assert!(valid_utc("2026-09-30T00:00:00.123456Z"));
        assert!(!valid_utc("2026-09-30 00:00:00"));
        assert!(!valid_utc("not-a-time"));
    }

    #[test]
    fn explicit_entry_set_resolves() {
        for wp in [
            "WP-00A", "WP-00B", "WP-01", "WP-02A", "WP-02B", "WP-02C", "WP-02D", "WP-03A", "WP-03B",
            "WP-03C", "WP-04A", "WP-04B", "WP-04C", "WP-05A", "WP-06", "WP-07A",
            "WP-07B", "WP-08A", "WP-09C", "WP-10", "WP-11", "WP-13",
        ] {
            assert!(entry_deps(wp).is_some(), "{wp}");
            assert!(matrix_gate(wp).is_some(), "{wp}");
        }
        assert!(entry_deps("WP-99").is_none());
        assert!(matrix_gate("WP-99").is_none());
    }

    #[test]
    fn ambiguous_entry_set_blocks() {
        // No `**Entry:**` WP list (WP-05B/C, WP-08B/C, WP-09A) or entry text
        // naming no WP ids (WP-09B "Candidate 3 frozen control", WP-09D
        // "WP-08", WP-09E "Wave 8", WP-12 registry description): blocked as
        // a plan defect, never an empty pass. Matrix rows still exist.
        for wp in [
            "WP-05B", "WP-05C", "WP-08B", "WP-08C", "WP-09A", "WP-09B1", "WP-09B9", "WP-09B10",
            "WP-09D", "WP-09E", "WP-12",
        ] {
            assert!(entry_deps(wp).is_none(), "{wp}");
            assert!(matrix_gate(wp).is_some(), "{wp}");
        }
    }
}
