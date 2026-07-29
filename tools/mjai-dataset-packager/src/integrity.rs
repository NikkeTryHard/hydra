//! WP-00B Packager Integrity Authority (transport layer).
//!
//! Produces and verifies the authoritative [`PackagedObjectRow`] for every
//! packaged object. The row is transport-only: it carries NO authorization
//! claim (attestation/purpose/disclosure arrive with WP-04B's `RawObjectRow`).
//!
//! Authority rules implemented here:
//! - Magic bytes alone never authorize reuse; every candidate output is fully
//!   zstd-decoded before it may be skipped or recorded.
//! - `packaged_object_id` is the SHA-256 over the row's canonical JSON bytes
//!   excluding the field itself. Canonical bytes are compact JSON with fields
//!   in the exact order declared by IMPLEMENTATION_SPEC section 12.1.
//! - `canonical_jsonl` / `record_count` are computed from the decoded payload
//!   without ever rewriting it. A line is a record when it parses as JSON;
//!   the payload is canonical JSONL when additionally every line equals the
//!   RFC 8785-style canonical serialization of its parsed value (sorted keys,
//!   compact separators, ECMAScript number formatting via `ryu-js`), there
//!   are no blank lines, and the payload ends with a newline. `record_count`
//!   counts parsed records regardless of canonicality.
//! - Manifest publication is crash-safe: unique O_EXCL temp in the same
//!   directory, write, fsync file, rename, fsync directory. On error only the
//!   owned temp is removed; an existing destination is never deleted.

use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const HASH_BUFFER: usize = 1024 * 1024;

#[derive(Debug)]
pub enum OutputState {
    /// No file at the expected path.
    Absent,
    /// Present with valid magic but unusable: short, wrong magic, or a body
    /// that does not fully decode. Never authorizes reuse.
    Unusable,
    /// Fully decoded; carries the measured facts.
    Verified(OutputFacts),
}

/// Measured facts about a compressed output and its decoded payload.
#[derive(Clone, Debug)]
pub struct OutputFacts {
    pub compressed_sha256: String,
    pub compressed_length: u64,
    pub decoded_sha256: String,
    pub decoded_length: u64,
    pub jsonl: JsonlFacts,
}

/// JSONL statistics computed from the decoded payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonlFacts {
    pub record_count: u64,
    pub canonical_jsonl: bool,
}

/// Measured facts about the source bytes (raw file, precompressed file, or
/// extracted archive member).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceFacts {
    pub sha256: String,
    pub length: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    #[serde(rename = "raw")]
    Raw,
    #[serde(rename = "archive_member")]
    ArchiveMember,
    #[serde(rename = "precompressed")]
    Precompressed,
}

/// Transport-only packaged-object row (IMPLEMENTATION_SPEC section 12.1).
/// Field order below is normative: it defines the canonical serialization.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackagedObjectRow {
    pub packaged_object_id: String,
    pub source_kind: SourceKind,
    pub source_container_sha256: Option<String>,
    pub source_member_path: Option<String>,
    pub source_bytes_sha256: String,
    pub source_bytes_length: u64,
    pub compressed_path: String,
    pub compressed_bytes_sha256: String,
    pub compressed_bytes_length: u64,
    pub decoded_bytes_sha256: String,
    pub decoded_bytes_length: u64,
    pub record_count: u64,
    pub canonical_jsonl: bool,
    pub packager_identity: String,
    pub packager_config_hash: String,
    pub created_at_utc: String,
}

impl PackagedObjectRow {
    /// Canonical JSON bytes of this row. With `include_id = false` the
    /// `packaged_object_id` field is omitted; those bytes are exactly what
    /// the id hashes.
    pub fn canonical_bytes(&self, include_id: bool) -> Vec<u8> {
        let text = |value: &String| serde_json::Value::from(value.as_str());
        let optional_text = |value: &Option<String>| match value {
            Some(text) => serde_json::Value::from(text.as_str()),
            None => serde_json::Value::Null,
        };
        let kind = |value: &SourceKind| serde_json::to_value(value).expect("enum serializes");
        let mut out = Vec::with_capacity(768);
        out.push(b'{');
        let mut first = true;
        let mut field = |out: &mut Vec<u8>, key: &str, value: &serde_json::Value| {
            if !first {
                out.push(b',');
            }
            first = false;
            out.extend_from_slice(&serde_json::to_vec(key).expect("static key"));
            out.push(b':');
            out.extend_from_slice(&serde_json::to_vec(value).expect("row value"));
        };
        if include_id {
            field(&mut out, "packaged_object_id", &text(&self.packaged_object_id));
        }
        field(&mut out, "source_kind", &kind(&self.source_kind));
        field(
            &mut out,
            "source_container_sha256",
            &optional_text(&self.source_container_sha256),
        );
        field(
            &mut out,
            "source_member_path",
            &optional_text(&self.source_member_path),
        );
        field(&mut out, "source_bytes_sha256", &text(&self.source_bytes_sha256));
        field(
            &mut out,
            "source_bytes_length",
            &serde_json::Value::from(self.source_bytes_length),
        );
        field(&mut out, "compressed_path", &text(&self.compressed_path));
        field(
            &mut out,
            "compressed_bytes_sha256",
            &text(&self.compressed_bytes_sha256),
        );
        field(
            &mut out,
            "compressed_bytes_length",
            &serde_json::Value::from(self.compressed_bytes_length),
        );
        field(&mut out, "decoded_bytes_sha256", &text(&self.decoded_bytes_sha256));
        field(
            &mut out,
            "decoded_bytes_length",
            &serde_json::Value::from(self.decoded_bytes_length),
        );
        field(
            &mut out,
            "record_count",
            &serde_json::Value::from(self.record_count),
        );
        field(
            &mut out,
            "canonical_jsonl",
            &serde_json::Value::from(self.canonical_jsonl),
        );
        field(&mut out, "packager_identity", &text(&self.packager_identity));
        field(&mut out, "packager_config_hash", &text(&self.packager_config_hash));
        field(&mut out, "created_at_utc", &text(&self.created_at_utc));
        out.push(b'}');
        out
    }

    /// Computes and stores `packaged_object_id`.
    pub fn seal(&mut self) {
        let bytes = self.canonical_bytes(false);
        self.packaged_object_id = to_hex(&Sha256::digest(&bytes));
    }

    /// Verifies the row's self-hash. Authoritative manifests must consist
    /// entirely of sealed rows.
    pub fn verify_seal(&self) -> Result<()> {
        let expected = {
            let mut probe = self.clone();
            probe.packaged_object_id = String::new();
            to_hex(&Sha256::digest(&probe.canonical_bytes(false)))
        };
        ensure!(
            self.packaged_object_id == expected,
            "transport row failed self-hash: {}",
            self.compressed_path
        );
        Ok(())
    }
}

/// RFC 8785-style canonical JSON writer for `serde_json` values: compact
/// separators, lexicographically sorted object keys, shortest ECMAScript
/// number representation (via `ryu-js`), minimal string escaping. Requires
/// `serde_json`'s default `BTreeMap` object representation.
pub fn write_canonical_value(value: &serde_json::Value, out: &mut Vec<u8>) {
    use serde_json::Value;
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                let _ = write!(out, "{int}");
            } else if let Some(uint) = number.as_u64() {
                let _ = write!(out, "{uint}");
            } else {
                let float = number.as_f64().expect("JSON numbers are finite");
                let mut buffer = ryu_js::Buffer::new();
                out.extend_from_slice(buffer.format_finite(float).as_bytes());
            }
        }
        Value::String(text) => {
            let encoded = serde_json::to_vec(text).expect("string serializes");
            out.extend_from_slice(&encoded);
        }
        Value::Array(items) => {
            out.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                write_canonical_value(item, out);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            out.push(b'{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push(b',');
                }
                let encoded = serde_json::to_vec(key).expect("key serializes");
                out.extend_from_slice(&encoded);
                out.push(b':');
                write_canonical_value(item, out);
            }
            out.push(b'}');
        }
    }
}

pub fn to_hex(digest: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    to_hex(&Sha256::digest(bytes))
}

/// Stable identity of the producing packager build.
pub fn packager_identity() -> String {
    let value = serde_json::json!({
        "crate": env!("CARGO_PKG_NAME"),
        "version": env!("CARGO_PKG_VERSION"),
    });
    let mut canonical = Vec::new();
    write_canonical_value(&value, &mut canonical);
    sha256_hex(&canonical)
}

/// Stable hash of the configuration inputs that govern output bytes.
pub fn packager_config_hash(level: i32, max_item_bytes: u64) -> String {
    let value = serde_json::json!({
        "level": level,
        "max_item_bytes": max_item_bytes,
    });
    let mut canonical = Vec::new();
    write_canonical_value(&value, &mut canonical);
    sha256_hex(&canonical)
}

/// Reads a whole file into the running hash and length.
pub fn sha256_file(path: &Path) -> Result<SourceFacts> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = BufReader::with_capacity(HASH_BUFFER, file);
    let mut hasher = Sha256::new();
    let mut length = 0u64;
    let mut buffer = vec![0u8; HASH_BUFFER];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        length += read as u64;
    }
    Ok(SourceFacts {
        sha256: to_hex(&hasher.finalize()),
        length,
    })
}

/// Incremental JSONL analyzer fed with decoded chunks.
#[derive(Default)]
struct JsonlAnalyzer {
    pending: Vec<u8>,
    record_count: u64,
    canonical: bool,
    started: bool,
}

impl JsonlAnalyzer {
    fn new() -> Self {
        Self {
            canonical: true,
            ..Self::default()
        }
    }

    fn feed(&mut self, chunk: &[u8]) {
        self.started = true;
        let mut rest = chunk;
        while let Some(position) = rest.iter().position(|byte| *byte == b'\n') {
            let (line, tail) = rest.split_at(position);
            self.pending.extend_from_slice(line);
            self.consume_complete_line();
            self.pending.clear();
            rest = &tail[1..];
        }
        self.pending.extend_from_slice(rest);
    }

    fn consume_complete_line(&mut self) {
        let line: &[u8] = &self.pending;
        if line.last() == Some(&b'\r') {
            // A CR before LF survives into the record bytes; canonical
            // serialization would drop it, so this flags non-canonicality.
            self.canonical = false;
        }
        if line.is_empty() {
            self.canonical = false;
            return;
        }
        match line_record(line) {
            Some(record_is_canonical) => {
                self.record_count += 1;
                self.canonical &= record_is_canonical;
            }
            None => self.canonical = false,
        }
    }

    fn finish(mut self) -> JsonlFacts {
        if !self.started {
            return JsonlFacts {
                record_count: 0,
                canonical_jsonl: false,
            };
        }
        if !self.pending.is_empty() {
            // Trailing bytes without a final newline: still analyzed as a
            // potential record, but the payload is not canonical JSONL.
            self.consume_complete_line();
            self.canonical = false;
        }
        JsonlFacts {
            record_count: self.record_count,
            canonical_jsonl: self.canonical,
        }
    }
}

/// Parses one JSONL line; `None` when it is not a JSON record, `Some(canonical)`
/// when it parses, carrying whether the bytes equal the canonical form.
fn line_record(line: &[u8]) -> Option<bool> {
    let text = std::str::from_utf8(line).ok()?;
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    let mut canonical = Vec::with_capacity(line.len());
    write_canonical_value(&value, &mut canonical);
    Some(canonical.as_slice() == line)
}

/// Fully validates a zstd file: measures the compressed bytes, then decodes
/// the ENTIRE frame, hashing the decoded payload and scanning it as JSONL.
/// Decode failures are reported as [`OutputState::Unusable`], never as skip
/// authorization; hard I/O errors propagate.
pub fn inspect_output_verified(path: &Path) -> Result<OutputState> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(OutputState::Absent),
        Err(error) => return Err(error).with_context(|| format!("inspecting {}", path.display())),
    };
    ensure!(
        !metadata.file_type().is_symlink(),
        "refusing symlink output: {}",
        path.display()
    );
    ensure!(
        metadata.is_file(),
        "output path is not a regular file: {}",
        path.display()
    );
    if metadata.len() < 8 {
        return Ok(OutputState::Unusable);
    }

    let (compressed_sha256, compressed_length) = {
        let file = File::open(path).with_context(|| format!("reading {}", path.display()))?;
        let mut reader = BufReader::with_capacity(HASH_BUFFER, file);
        let mut hasher = Sha256::new();
        let mut length = 0u64;
        let mut buffer = vec![0u8; HASH_BUFFER];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            length += read as u64;
        }
        (to_hex(&hasher.finalize()), length)
    };

    let decoded = (|| -> io::Result<OutputFacts> {
        let file = File::open(path)?;
        let decoder = zstd::stream::read::Decoder::new(BufReader::with_capacity(HASH_BUFFER, file))?;
        let mut reader = BufReader::with_capacity(HASH_BUFFER, decoder);
        let mut decoded_hasher = Sha256::new();
        let mut decoded_length = 0u64;
        let mut analyzer = JsonlAnalyzer::new();
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = reader.read_until(b'\n', &mut line)?;
            if read == 0 {
                break;
            }
            decoded_hasher.update(&line);
            decoded_length += read as u64;
            analyzer.feed(&line);
        }
        Ok(OutputFacts {
            compressed_sha256,
            compressed_length,
            decoded_sha256: to_hex(&decoded_hasher.finalize()),
            decoded_length,
            jsonl: analyzer.finish(),
        })
    })();

    match decoded {
        Ok(facts) => Ok(OutputState::Verified(facts)),
        Err(_) => Ok(OutputState::Unusable),
    }
}

/// Loads the transport manifest. A missing file yields an empty authority.
/// Every row must seal-verify; anything else is a hard error (the manifest is
/// crash-safe-published, so partial content means external corruption).
pub fn load_manifest(path: &Path) -> Result<Vec<PackagedObjectRow>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    let text = String::from_utf8(bytes)
        .map_err(|_| anyhow::anyhow!("manifest is not UTF-8: {}", path.display()))?;
    let mut rows = Vec::new();
    for (index, line) in text.split('\n').enumerate() {
        if line.is_empty() {
            // Only the single trailing newline is tolerated.
            ensure!(
                index + 1 == text.split('\n').count(),
                "blank manifest line at {}: {}",
                index + 1,
                path.display()
            );
            continue;
        }
        let row: PackagedObjectRow = serde_json::from_str(line).with_context(|| {
            format!(
                "malformed manifest row at line {}: {}",
                index + 1,
                path.display()
            )
        })?;
        row.verify_seal()
            .with_context(|| format!("corrupt manifest row at line {}", index + 1))?;
        rows.push(row);
    }
    Ok(rows)
}

/// Crash-safe manifest publication: unique O_EXCL temp beside the target,
/// write, fsync file, rename over destination, fsync directory. On failure
/// the owned temp is removed and the previous manifest is left untouched.
pub fn publish_manifest(path: &Path, mut rows: Vec<PackagedObjectRow>) -> Result<()> {
    rows.sort_by(|left, right| {
        left.compressed_path
            .cmp(&right.compressed_path)
            .then_with(|| left.packaged_object_id.cmp(&right.packaged_object_id))
    });
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating manifest directory {}", parent.display()))?;
        }
    }
    let temp = manifest_temp_path(path);
    let result = (|| -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .with_context(|| format!("creating temporary manifest {}", temp.display()))?;
        let mut writer = io::BufWriter::new(file);
        for row in &rows {
            writer.write_all(&row.canonical_bytes(true))?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        writer.get_ref().sync_all().context("fsyncing manifest temp")?;
        fs::rename(&temp, path).with_context(|| {
            format!(
                "atomically publishing manifest {}",
                path.display()
            )
        })?;
        let directory = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        sync_directory(directory)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn manifest_temp_path(path: &Path) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    path.with_file_name(format!(
        ".{name}.tmp.{}.{id}",
        std::process::id()
    ))
}

pub fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("fsyncing directory {}", path.display()))
}

/// Removes interrupted temporaries matching this packager's naming scheme
/// (`.NAME.tmp.PID.COUNTER`) for any planned output `NAME` inside `parent`.
/// Used only in manifest mode so legacy runs remain byte-for-byte
/// behaviorally identical.
pub fn sweep_stale_temps(parent: &Path, names: &HashSet<String>) -> Result<()> {
    let prefixes: Vec<String> = names
        .iter()
        .map(|name| format!(".{name}.tmp."))
        .collect();
    let entries = match fs::read_dir(parent) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("scanning {}", parent.display()))
        }
    };
    for entry in entries {
        let entry = entry.with_context(|| format!("scanning {}", parent.display()))?;
        let candidate = entry.file_name().to_string_lossy().into_owned();
        if prefixes.iter().any(|prefix| candidate.starts_with(prefix.as_str())) {
            fs::remove_file(entry.path()).with_context(|| {
                format!("removing interrupted temporary {}", entry.path().display())
            })?;
        }
    }
    Ok(())
}

/// Formats a `SystemTime` as `YYYY-MM-DDTHH:MM:SSZ` (proleptic Gregorian,
/// UTC) without pulling in a date-time dependency.
pub fn utc_timestamp(time: SystemTime) -> String {
    let duration = time
        .duration_since(UNIX_EPOCH)
        .expect("timestamps precede the epoch");
    let seconds = duration.as_secs();
    let days = (seconds / 86_400) as i64;
    let remainder = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        remainder / 3_600,
        (remainder % 3_600) / 60,
        remainder % 60
    )
}

pub fn utc_now() -> String {
    utc_timestamp(SystemTime::now())
}

/// Howard Hinnant's civil-from-days algorithm.
fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let shifted = days_since_epoch + 719_468;
    let era = if shifted >= 0 { shifted } else { shifted - 146_096 } / 146_097;
    let day_of_era = (shifted - era * 146_097) as u64;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era as i64 + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_pointer = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_pointer + 2) / 5 + 1) as u32;
    let month = if month_pointer < 10 {
        month_pointer + 3
    } else {
        month_pointer - 9
    } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_timestamp_matches_known_epochs() {
        let cases = [
            (0u64, "1970-01-01T00:00:00Z"),
            (951_782_400, "2000-02-29T00:00:00Z"),
            (1_000_000_000, "2001-09-09T01:46:40Z"),
            (1_700_000_000, "2023-11-14T22:13:20Z"),
            (2_147_483_647, "2038-01-19T03:14:07Z"),
            (4_102_444_800, "2100-01-01T00:00:00Z"),
        ];
        for (seconds, expected) in cases {
            let rendered = utc_timestamp(UNIX_EPOCH + std::time::Duration::from_secs(seconds));
            assert_eq!(rendered, expected, "seconds={seconds}");
        }
    }

    #[test]
    fn canonical_writer_sorts_keys_and_formats_numbers_like_ecmascript() {
        let value: serde_json::Value = serde_json::from_str(
            r#"{"b":1,"a":{"z":[true,false,null],"y":0.15625},"c":1e21,"d":0.30000000000000004}"#,
        )
        .expect("parses");
        let mut out = Vec::new();
        write_canonical_value(&value, &mut out);
        let rendered = String::from_utf8(out).expect("utf8");
        let float_1e21 = {
            let mut buffer = ryu_js::Buffer::new();
            buffer.format_finite(1e21).to_owned()
        };
        let expected = format!(
            r#"{{"a":{{"y":0.15625,"z":[true,false,null]}},"b":1,"c":{float_1e21},"d":0.30000000000000004}}"#
        );
        assert_eq!(rendered, expected);
        assert!(float_1e21.starts_with("1e"), "ES6 exponent style: {float_1e21}");
    }

    #[test]
    fn jsonl_analyzer_flags_noncanonical_but_counts_records() {
        let mut analyzer = JsonlAnalyzer::new();
        analyzer.feed(b"{\"type\":\"start_game\",\"names\":[\"alice\",\"bob\"]}\n");
        analyzer.feed(b"{\"type\":\"end_game\"}\n");
        let facts = analyzer.finish();
        assert_eq!(facts.record_count, 2);
        assert!(
            !facts.canonical_jsonl,
            "unsorted keys are records but not canonical"
        );

        let mut analyzer = JsonlAnalyzer::new();
        analyzer.feed(b"{\"names\":[\"alice\",\"bob\"],\"type\":\"start_game\"}\n");
        let facts = analyzer.finish();
        assert_eq!(facts.record_count, 1);
        assert!(facts.canonical_jsonl, "sorted compact line is canonical");
    }

    #[test]
    fn jsonl_analyzer_rejects_blank_lines_and_missing_trailing_newline() {
        let mut analyzer = JsonlAnalyzer::new();
        analyzer.feed(b"{\"a\":1}\n\n{\"a\":2}\n");
        let facts = analyzer.finish();
        assert_eq!(facts.record_count, 2);
        assert!(!facts.canonical_jsonl, "blank line breaks canonicality");

        let mut analyzer = JsonlAnalyzer::new();
        analyzer.feed(b"{\"a\":1}\n{\"a\":2}");
        let facts = analyzer.finish();
        assert_eq!(facts.record_count, 2);
        assert!(!facts.canonical_jsonl, "payload must end with newline");

        let mut analyzer = JsonlAnalyzer::new();
        analyzer.feed(b"{\"a\":1}\r\n");
        let facts = analyzer.finish();
        assert_eq!(facts.record_count, 1);
        assert!(!facts.canonical_jsonl, "CR surviving in line is non-canonical");

        let mut analyzer = JsonlAnalyzer::new();
        analyzer.feed(b"not json\n");
        let facts = analyzer.finish();
        assert_eq!(facts.record_count, 0);
        assert!(!facts.canonical_jsonl);

        let facts = JsonlAnalyzer::new().finish();
        assert_eq!(facts.record_count, 0);
        assert!(!facts.canonical_jsonl, "empty payload is not canonical");
    }

    #[test]
    fn jsonl_analyzer_counts_whitespace_polluted_records_as_noncanonical() {
        let mut analyzer = JsonlAnalyzer::new();
        analyzer.feed(b"{ \"a\": 1 }\n");
        let facts = analyzer.finish();
        assert_eq!(facts.record_count, 1);
        assert!(!facts.canonical_jsonl);
    }

    #[test]
    fn row_seal_is_stable_and_sensitive_to_content() {
        let base = PackagedObjectRow {
            packaged_object_id: String::new(),
            source_kind: SourceKind::Raw,
            source_container_sha256: None,
            source_member_path: None,
            source_bytes_sha256: "aa".repeat(32),
            source_bytes_length: 172,
            compressed_path: "a.mjai.json.zst".into(),
            compressed_bytes_sha256: "bb".repeat(32),
            compressed_bytes_length: 150,
            decoded_bytes_sha256: "cc".repeat(32),
            decoded_bytes_length: 172,
            record_count: 3,
            canonical_jsonl: false,
            packager_identity: "identity".into(),
            packager_config_hash: "config".into(),
            created_at_utc: "2026-08-22T00:00:00Z".into(),
        };
        let mut sealed = base.clone();
        sealed.seal();
        assert_ne!(sealed.packaged_object_id, "");
        sealed.verify_seal().expect("sealed row verifies");

        let mut tampered = sealed.clone();
        tampered.record_count += 1;
        assert!(tampered.verify_seal().is_err(), "tampering breaks the seal");

        let mut other_time = base.clone();
        other_time.created_at_utc = "2026-08-23T00:00:00Z".into();
        other_time.seal();
        assert_ne!(other_time.packaged_object_id, sealed.packaged_object_id);

        let reserialized = sealed.canonical_bytes(true);
        let parsed: PackagedObjectRow =
            serde_json::from_slice(&reserialized).expect("canonical bytes re-parse");
        assert_eq!(parsed, sealed);
        let first_field = &reserialized[2..reserialized.len().min(30)];
        assert!(
            String::from_utf8_lossy(first_field).starts_with("packaged_object_id"),
            "id comes first per spec order: {}",
            String::from_utf8_lossy(&reserialized[..48])
        );
    }

    #[test]
    fn manifest_round_trips_through_load_and_publish() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("nested").join("manifest.jsonl");
        let loaded_empty = load_manifest(&path).expect("missing manifest is empty");
        assert!(loaded_empty.is_empty());

        let mut row = PackagedObjectRow {
            packaged_object_id: String::new(),
            source_kind: SourceKind::ArchiveMember,
            source_container_sha256: Some("dd".repeat(32)),
            source_member_path: Some("games/g1.mjai.json".into()),
            source_bytes_sha256: "ee".repeat(32),
            source_bytes_length: 90,
            compressed_path: "d/games/g1.mjai.json.zst".into(),
            compressed_bytes_sha256: "ff".repeat(32),
            compressed_bytes_length: 80,
            decoded_bytes_sha256: "ab".repeat(32),
            decoded_bytes_length: 90,
            record_count: 2,
            canonical_jsonl: false,
            packager_identity: "identity".into(),
            packager_config_hash: "config".into(),
            created_at_utc: "2026-08-22T12:00:00Z".into(),
        };
        row.seal();
        publish_manifest(&path, vec![row.clone()]).expect("publishes");

        let loaded = load_manifest(&path).expect("loads");
        assert_eq!(loaded, vec![row]);

        let corrupt = path.parent().unwrap().join("corrupt.jsonl");
        std::fs::write(&corrupt, "{\"packaged_object_id\":\"nope\"}\n").expect("write");
        assert!(load_manifest(&corrupt).is_err(), "unsealed rows rejected");
        let garbage = path.parent().unwrap().join("garbage.jsonl");
        std::fs::write(&garbage, "{}\n").expect("write");
        assert!(
            load_manifest(&garbage).is_err(),
            "unknown/missing fields rejected"
        );
    }

    #[test]
    fn inspect_output_verified_classifies_corruption() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let missing = temp.path().join("missing.zst");
        assert!(matches!(
            inspect_output_verified(&missing)?,
            OutputState::Absent
        ));

        let truncated = temp.path().join("truncated.zst");
        std::fs::write(&truncated, b"\x28\xb5\x2f\xfd\x00\x00")?;
        assert!(matches!(
            inspect_output_verified(&truncated)?,
            OutputState::Unusable
        ));

        let good = temp.path().join("good.zst");
        {
            let file = File::create(&good)?;
            let mut encoder = zstd::stream::write::Encoder::new(file, 1)?;
            encoder.write_all(b"{\"a\":1}\n{\"b\":2}\n")?;
            encoder.finish()?;
        }
        match inspect_output_verified(&good)? {
            OutputState::Verified(facts) => {
                assert_eq!(facts.decoded_length, 16);
                assert_eq!(facts.jsonl.record_count, 2);
                assert!(facts.jsonl.canonical_jsonl);
                assert_eq!(
                    facts.decoded_sha256,
                    sha256_hex(b"{\"a\":1}\n{\"b\":2}\n")
                );
            }
            other => panic!("expected verified, got {other:?}"),
        }
        let corrupt_body = temp.path().join("corrupt.zst");
        let mut frame = std::fs::read(&good)?;
        let middle = frame.len() / 2;
        frame[middle] ^= 0xff;
        std::fs::write(&corrupt_body, &frame)?;
        // Checksum-less zstd frames may silently decode flipped bytes into
        // DIFFERENT payload; either the decoder rejects the body outright or
        // the measured decoded hash diverges. Both outcomes deny reuse of
        // the original object's identity; neither can masquerade as intact.
        match inspect_output_verified(&corrupt_body)? {
            OutputState::Unusable => {}
            OutputState::Verified(facts) => assert_ne!(
                facts.decoded_sha256,
                sha256_hex(b"{\"a\":1}\n{\"b\":2}\n"),
                "silent corruption must change the decoded payload hash"
            ),
            OutputState::Absent => panic!("corrupt file vanished"),
        }
        Ok(())
    }

    #[test]
    fn stale_temp_sweep_removes_only_matching_pattern() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let out_dir = temp.path().join("out");
        std::fs::create_dir_all(&out_dir)?;
        let output = out_dir.join("a.mjai.json.zst");
        std::fs::write(&output, b"keep")?;
        std::fs::write(out_dir.join(".a.mjai.json.zst.tmp.123.0"), b"junk")?;
        std::fs::write(out_dir.join(".b.mjai.json.zst.tmp.123.1"), b"other")?;

        sweep_stale_temps(
            &out_dir,
            &HashSet::from(["a.mjai.json.zst".to_string()]),
        )?;
        assert_eq!(std::fs::read(&output)?, b"keep");
        assert!(!out_dir.join(".a.mjai.json.zst.tmp.123.0").exists());
        assert!(out_dir.join(".b.mjai.json.zst.tmp.123.1").exists());
        Ok(())
    }

    #[test]
    fn config_and_identity_hashes_are_stable() {
        let first = packager_config_hash(1, 512 * 1024 * 1024);
        let again = packager_config_hash(1, 512 * 1024 * 1024);
        let changed = packager_config_hash(2, 512 * 1024 * 1024);
        assert_eq!(first, again);
        assert_ne!(first, changed);
        assert_eq!(packager_identity(), packager_identity());
    }
}

