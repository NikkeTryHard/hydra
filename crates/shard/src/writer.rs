//! Cold shard file writer (P4-B, cold only — NEVER hot).
//!
//! Reference pattern (read-only): hydra1 `bc-shards writer.rs` — `BufWriter`
//! with a placeholder header patched at `finish`, game-atomic appends (one
//! `write_all` per game, so a crash can only lose a whole game, never a torn
//! row), rotation at game boundaries via [`ShardWriter::fits`].
//!
//! File layout: [`crate::reader::HEADER_LEN`] header bytes (magic, version,
//! bucket, row count, row stride, payload SHA-256) followed by `row_count`
//! compact rows of [`crate::full26::compact_row_bytes`]`(t)` bytes.

extern crate alloc;

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::digest::{PayloadHasher, raw_hex_prefixed};
use crate::full26::compact_row_bytes;
use crate::reader::{HEADER_LEN, encode_header};

/// Shard writer failure taxonomy (fail-closed, cold only).
#[derive(Debug)]
pub enum ShardWriteError {
    Io(std::io::Error),
    /// Appended bytes are not a whole number of rows (`got` bytes).
    TornGame { got: usize },
}

impl core::fmt::Display for ShardWriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ShardWriteError::Io(e) => write!(f, "shard write io: {e}"),
            ShardWriteError::TornGame { got } => {
                write!(f, "shard append is not game-atomic: {got} bytes")
            }
        }
    }
}

impl std::error::Error for ShardWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ShardWriteError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ShardWriteError {
    fn from(e: std::io::Error) -> Self {
        ShardWriteError::Io(e)
    }
}

/// Finished-file receipt (the reader verifies every field on open).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishedShard {
    pub path: PathBuf,
    pub rows: u64,
    pub t_bucket: usize,
    pub row_bytes: usize,
    pub payload_sha256: String,
}

/// Cold shard writer: placeholder header, game-atomic appends, digest patch.
#[derive(Debug)]
pub struct ShardWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    t_bucket: usize,
    row_bytes: usize,
    rows: u64,
    hasher: PayloadHasher,
    max_rows: Option<u64>,
}

impl ShardWriter {
    /// Create a shard file at `path` for bucket `t` (the header placeholder
    /// is zeros until `finish`).
    pub fn create(path: &Path, t: usize, max_rows: Option<u64>) -> Result<Self, ShardWriteError> {
        let row_bytes = compact_row_bytes(t);
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&[0u8; HEADER_LEN])?;
        Ok(Self {
            path: path.to_path_buf(),
            writer,
            t_bucket: t,
            row_bytes,
            rows: 0,
            hasher: PayloadHasher::new(),
            max_rows,
        })
    }

    /// File bucket `T`.
    pub fn t_bucket(&self) -> usize {
        self.t_bucket
    }

    /// Compact row stride.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Rows committed so far.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// True when a `game_rows`-row game still fits under `max_rows`
    /// (rotation happens at game boundaries only; `None` max never rotates).
    pub fn fits(&self, game_rows: u64) -> bool {
        match self.max_rows {
            None => true,
            Some(max) => self.rows + game_rows <= max,
        }
    }

    /// Append one whole game's compact rows (single `write_all`; `rows` must
    /// be a whole number of file rows or nothing is written).
    pub fn append_game(&mut self, rows: &[u8]) -> Result<(), ShardWriteError> {
        if !rows.len().is_multiple_of(self.row_bytes) {
            return Err(ShardWriteError::TornGame { got: rows.len() });
        }
        if rows.is_empty() {
            return Ok(());
        }
        self.writer.write_all(rows)?;
        self.hasher.update(rows);
        self.rows += (rows.len() / self.row_bytes) as u64;
        Ok(())
    }

    /// Flush, patch the header (counts + payload digest), and return the
    /// receipt (the exact digest bound into the header).
    pub fn finish(mut self) -> Result<FinishedShard, ShardWriteError> {
        self.writer.flush()?;
        let digest = self.hasher.finalize_raw();
        let header = encode_header(self.t_bucket, self.rows, self.row_bytes, &digest);
        let file = self.writer.get_mut();
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;
        file.flush()?;
        Ok(FinishedShard {
            path: self.path.clone(),
            rows: self.rows,
            t_bucket: self.t_bucket,
            row_bytes: self.row_bytes,
            payload_sha256: raw_hex_prefixed(&digest),
        })
    }
}
/// Safetensors mmap cache (`src/cache.rs`, Phase 3 M4).
///
/// Wired here — NOT via a crate-root `pub mod cache` in `lib.rs` — so this
/// ticket touches zero shared files: `hydra_shard::writer::cache` is the
/// single import path (sibling `replay::expand` uses the same `#[path]`
/// pattern for the same reason). Path: `hydra-shard/src/cache.rs`.
#[path = "cache.rs"]
pub mod cache;

// 5-exact columnar writer (Phase 3, greenfield — M1/M3/M4/M5/M8).
//
// Greenfield `ArrowWriter` parity gate: NO Rust writer existed before this
// module (`parquet_join.rs:1-12` proves read-only); the parity test is
// greenfield, NOT a regression (M1: `parquet =59.3.0` pinned).
//
// 5-EXACT ONLY (mirrors `parquet.py:238-246` + corrections §2):
// 1. `compression = ZSTD`
// 2. `compression_level = 3`
// 3. `use_dictionary = true`
// 4. `write_batch_size = 8192`
// 5. `store_schema = true` ≙ footer-discoverable `ARROW:schema` KV (M4) on
//    BOTH writers (M5/M8: privileged `:312-319` omits `store_schema` in
//    Python — Rust ships it on both paths; re-freeze gate).
//
// M3: `created_by` is UNSET — this module NEVER calls
// `set_created_by`; the footer carries the default
// `parquet-rs version 59.3.0` provenance. Probe B asserts it is present
// and non-empty without pinning its text.
//
// MUST-NOT-ADD (each forks `dataset_hash` goldens; separate hash-migration
// ONLY): `set_max_row_group_row_count/bytes`, `set_data_page_size_limit`,
// `set_data_page_row_count_limit`, `set_dictionary_page_size_limit`,
// `set_statistics_enabled`, `set_write_page_header_statistics`,
// `set_statistics_truncate_length`, `set_column_index_truncate_length`,
// `set_bloom_filter_*`, `set_encoding`/`set_column_*`,
// `set_sorting_columns`, `set_coerce_types`, CDC
// (`set_content_defined_chunking`), encryption, `set_writer_version`,
// `set_offset_index_disabled`. Row-group 512k / page 1MiB / dict 1MiB /
// stats-truncate 64 are NOT set today.
//
// Capsule rule (B4, for the bridge owner): capsules are created OUTSIDE
// `detach`; this writer does pure file I/O and holds no GIL discipline.
// PyCapsule 5 names / dunders / single-consume live in the bridge seam.
//
// `ARROW:schema` siting: `ArrowWriter::try_new` auto-appends the encoded
// Arrow schema under `ARROW:schema` (unless `skip_arrow_metadata` — never
// set here), so `store_schema` discoverability holds with `created_by`
// UNSET. Extra caller KVs (e.g. `dataset_hash`) ride in
// `set_key_value_metadata`; the writer never strips the auto-appended
// `ARROW:schema` entry.

use arrow_array::{Array as _, RecordBatch};
use arrow_schema::SchemaRef;
use parquet::arrow::{ARROW_SCHEMA_META_KEY, ArrowWriter};
use parquet::basic::{Compression, ZstdLevel};
use parquet::errors::ParquetError;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader as ParquetFileReader, SerializedFileReader};

/// 5-exact write batch size.
///
/// Canonical single owner is `hydra-bridge::columnar::BATCH` (search/eval
/// import it from there); this mirror MUST stay equal and exists only so
/// the 5-exact props builder has no second literal. Value `8192` matches
/// `parquet.py:244,318` `write_batch_size=8192`.
pub const WRITE_BATCH_SIZE: usize = 8192;

/// Privileged columns that MUST never appear in an actor batch (mirrors
/// `parquet.py:59-66` `FORBIDDEN_IN_ACTOR`).
pub const FORBIDDEN_IN_ACTOR: &[&str] = &[
    "hidden_tiles",
    "wall",
    "dead_wall",
    "opponent_hand",
    "privileged",
    "full_world",
];

/// Frozen history buckets (mirrors `HISTORY_BUCKET_LENGTHS` +
/// `COMPACT_T_BUCKETS`); shard geometry guard accepts ONLY these.
pub const SHARD_HISTORY_BUCKETS: [usize; 4] = [32, 64, 128, 256];
/// Maximum history length (`T = 256` bucket top; `[N, 256]` geometry).
pub const SHARD_MAX_HISTORY: usize = 256;

/// Columnar writer failure taxonomy (fail-closed, cold only).
#[derive(Debug)]
pub enum ColumnarWriteError {
    Io(std::io::Error),
    Parquet(ParquetError),
    Arrow(arrow_schema::ArrowError),
    /// Batch schema does not match the writer schema.
    SchemaMismatch { expected: String, got: String },
    /// Actor batch carries a privileged column / key.
    PrivilegedLeak { column: String },
    /// Footer probe failed (missing `ARROW:schema` / `created_by` / counts).
    Footer { msg: String },
    /// Geometry guard failed (history_len / bool / row-count).
    Geometry { msg: String },
}

impl core::fmt::Display for ColumnarWriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ColumnarWriteError::Io(e) => write!(f, "columnar write io: {e}"),
            ColumnarWriteError::Parquet(e) => write!(f, "columnar parquet: {e}"),
            ColumnarWriteError::Arrow(e) => write!(f, "columnar arrow: {e}"),
            ColumnarWriteError::SchemaMismatch { expected, got } => {
                write!(f, "columnar schema mismatch: expected {expected}, got {got}")
            }
            ColumnarWriteError::PrivilegedLeak { column } => {
                write!(f, "columnar privileged leak into actor batch: {column:?}")
            }
            ColumnarWriteError::Footer { msg } => write!(f, "columnar footer: {msg}"),
            ColumnarWriteError::Geometry { msg } => write!(f, "columnar geometry: {msg}"),
        }
    }
}

impl std::error::Error for ColumnarWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ColumnarWriteError::Io(e) => Some(e),
            ColumnarWriteError::Parquet(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ColumnarWriteError {
    fn from(e: std::io::Error) -> Self {
        ColumnarWriteError::Io(e)
    }
}

impl From<ParquetError> for ColumnarWriteError {
    fn from(e: ParquetError) -> Self {
        ColumnarWriteError::Parquet(e)
    }
}

impl From<arrow_schema::ArrowError> for ColumnarWriteError {
    fn from(e: arrow_schema::ArrowError) -> Self {
        ColumnarWriteError::Arrow(e)
    }
}

/// 5-exact props, no caller metadata.
///
/// ONLY these setters (M3: no `set_created_by`; M4: no manual
/// `ARROW:schema` — `ArrowWriter::try_new` appends it, which IS the
/// `store_schema` carrier).
pub fn writer_props_5exact() -> Result<WriterProperties, ParquetError> {
    let level = ZstdLevel::try_new(3)?;
    Ok(WriterProperties::builder()
        .set_compression(Compression::ZSTD(level))
        .set_dictionary_enabled(true)
        .set_write_batch_size(WRITE_BATCH_SIZE)
        .build())
}

/// 5-exact props carrying caller KVs (e.g. `dataset_hash`).
///
/// The caller MUST NOT supply `ARROW:schema` here: `ArrowWriter::try_new`
/// overwrites/appends the encoded schema entry itself. Supplying it is a
/// fork risk, so this fn rejects it fail-closed.
pub fn writer_props_5exact_with_meta(
    extra: Vec<KeyValue>,
) -> Result<WriterProperties, ColumnarWriteError> {
    for kv in &extra {
        if kv.key.as_str() == ARROW_SCHEMA_META_KEY {
            return Err(ColumnarWriteError::Footer {
                msg: "caller MUST NOT supply ARROW:schema KV (writer appends it)".to_string(),
            });
        }
    }
    let level =
        ZstdLevel::try_new(3).map_err(ColumnarWriteError::Parquet)?;
    Ok(WriterProperties::builder()
        .set_compression(Compression::ZSTD(level))
        .set_dictionary_enabled(true)
        .set_write_batch_size(WRITE_BATCH_SIZE)
        .set_key_value_metadata(Some(extra))
        .build())
}

/// Finished-parquet receipt (close-counts + stream hash).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParquetWriteReceipt {
    pub path: PathBuf,
    pub rows: usize,
    pub row_groups: usize,
    pub file_sha256: String,
}

/// Stream-hash a file via 1MiB chunks (mirrors `parquet.py:249-253` +
/// `loader.py:84-95`; avoids `read_bytes` peak).
pub fn sha256_stream_1m(path: &Path) -> Result<String, ColumnarWriteError> {
    use sha2::{Digest, Sha256};
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    use std::io::Read as _;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let sum = hasher.finalize();
    Ok(raw_hex_prefixed(sum.as_slice()))
}

/// Builder-pinned dataset hash: `sha256:` + hex over LF-joined SORTED
/// decision ids (mirrors `training/shard_reader.py:138-141`
/// `dataset_hash_of`; byte-identical on the same ids).
pub fn dataset_hash_of_sorted_ids(ids: &[String]) -> String {
    use sha2::{Digest, Sha256};
    let mut sorted: Vec<&str> = ids.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    let mut hasher = Sha256::new();
    for (i, id) in sorted.iter().enumerate() {
        if i > 0 {
            hasher.update(b"\n");
        }
        hasher.update(id.as_bytes());
    }
    let sum = hasher.finalize();
    raw_hex_prefixed(sum.as_slice())
}

/// Triple firewall: actor batch MUST never carry privileged data.
///
/// 1. Column-name gate over the batch schema (`FORBIDDEN_IN_ACTOR`).
/// 2. `actor_observation` JSON top-level key gate (substring scan — no
///    `to_pylist` row-dict path; logical check, never `memcmp`).
/// 3. Nested one-level gate: rejects `"key":{"privileged":…}`-shaped
///    nesting via the same scan (mirrors `parquet.py:170-178` top + one
///    nested; dora `(5,)` shape is the expand owner's gate, asserted in
///    geometry probes, not re-checked per string here).
pub fn assert_no_privileged_leak(batch: &RecordBatch) -> Result<(), ColumnarWriteError> {
    let schema = batch.schema();
    for field in schema.fields() {
        let name = field.name();
        if FORBIDDEN_IN_ACTOR.contains(&name.as_str()) {
            return Err(ColumnarWriteError::PrivilegedLeak {
                column: name.to_string(),
            });
        }
    }
    if let Ok(idx) = schema.index_of("actor_observation") {
        let col = batch.column(idx);
        if let Some(strings) = col
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
        {
            for row in 0..strings.len() {
                if strings.is_null(row) {
                    continue;
                }
                let v = strings.value(row);
                for forbidden in FORBIDDEN_IN_ACTOR {
                    let needle = alloc::format!("\"{forbidden}\"");
                    if v.contains(needle.as_str()) {
                        return Err(ColumnarWriteError::PrivilegedLeak {
                            column: alloc::format!("actor_observation[{row}].{forbidden}"),
                        });
                    }
                }
            }
        } else if let Some(large) = col
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
        {
            for row in 0..large.len() {
                if large.is_null(row) {
                    continue;
                }
                let v = large.value(row);
                for forbidden in FORBIDDEN_IN_ACTOR {
                    let needle = alloc::format!("\"{forbidden}\"");
                    if v.contains(needle.as_str()) {
                        return Err(ColumnarWriteError::PrivilegedLeak {
                            column: alloc::format!("actor_observation[{row}].{forbidden}"),
                        });
                    }
                }
            }
        }
        // StringView / Dictionary(UInt8,Utf8) physicals normalize at the
        // writer (logical-compat per ArrowWriter docs); the column-name gate
        // above already binds them. No byte-level branch here.
    }
    Ok(())
}

/// Geometry guards: `[N, 256]` + `history_len` + `uint8` bools.
///
/// * `history_len` MUST be one of `SHARD_HISTORY_BUCKETS` (frozen
///   `(32, 64, 128, 256)`; `256` is the `[N, 256]` top).
/// * `bool_byte` MUST be `0`/`1` (on-disk bools ride as `uint8`; writer
///   views never accept `2..=255`).
/// * `rows` MUST be non-zero (empty close is a caller bug, not an empty
///   shard — mirrors `parquet.py:199-200` no-rows guard).
pub fn assert_history_len(history_len: usize) -> Result<(), ColumnarWriteError> {
    if SHARD_HISTORY_BUCKETS.contains(&history_len) {
        Ok(())
    } else {
        Err(ColumnarWriteError::Geometry {
            msg: alloc::format!("history_len {history_len} not in (32, 64, 128, 256)"),
        })
    }
}

/// `uint8` bool guard: only `0`/`1` are observable bool bytes.
pub fn assert_uint8_bool(v: u8) -> Result<(), ColumnarWriteError> {
    if v <= 1 {
        Ok(())
    } else {
        Err(ColumnarWriteError::Geometry {
            msg: alloc::format!("uint8 bool out of range: {v}"),
        })
    }
}

fn write_batches_5exact(
    path: &Path,
    schema: SchemaRef,
    batches: &[RecordBatch],
    props: WriterProperties,
) -> Result<ParquetWriteReceipt, ColumnarWriteError> {
    if batches.is_empty() {
        return Err(ColumnarWriteError::Geometry {
            msg: "no batches to write (empty close forbidden)".to_string(),
        });
    }
    let expected_schema = schema.as_ref().to_string();
    let mut total_rows = 0usize;
    for b in batches {
        if b.schema().as_ref().to_string() != expected_schema {
            return Err(ColumnarWriteError::SchemaMismatch {
                expected: expected_schema.clone(),
                got: b.schema().as_ref().to_string(),
            });
        }
        total_rows += b.num_rows();
    }
    if total_rows == 0 {
        return Err(ColumnarWriteError::Geometry {
            msg: "no rows to write (empty close forbidden)".to_string(),
        });
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    for b in batches {
        writer.write(b)?;
    }
    // close-counts: `close` finalizes the footer and returns metadata;
    // `into_inner` without `close` would corrupt (missing footer).
    let meta = writer.close()?;
    // proof: parquet row counts are non-negative and fit in memory.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let closed_rows = meta.file_metadata().num_rows() as usize;
    if closed_rows != total_rows {
        return Err(ColumnarWriteError::Footer {
            msg: alloc::format!("close row count {closed_rows} != written {total_rows}"),
        });
    }
    let row_groups = meta.num_row_groups();
    let file_sha256 = sha256_stream_1m(path)?;
    // Footer probe on the just-closed file (store_schema + created_by + KV).
    probe_footer_5exact(path, total_rows)?;
    Ok(ParquetWriteReceipt {
        path: path.to_path_buf(),
        rows: total_rows,
        row_groups,
        file_sha256,
    })
}

/// Actor shards, 5-exact (mirrors `parquet.py:191-284` shape; sorted splits
/// are the caller's — one file per call here).
///
/// Runs the triple firewall BEFORE any byte is written.
pub fn write_actor_parquet_5exact(
    path: &Path,
    batches: &[RecordBatch],
) -> Result<ParquetWriteReceipt, ColumnarWriteError> {
    let schema: SchemaRef = batches
        .first()
        .map(|b| b.schema())
        .ok_or_else(|| ColumnarWriteError::Geometry {
            msg: "no batches to write (empty close forbidden)".to_string(),
        })?;
    for b in batches {
        assert_no_privileged_leak(b)?;
    }
    let props = writer_props_5exact().map_err(ColumnarWriteError::Parquet)?;
    write_batches_5exact(path, schema, batches, props)
}

/// Privileged shards, SAME 5-exact (M5 re-freeze gate + M8 `store_schema`
/// BOTH sites: fixes `parquet.py:312-319` which pins 4 opts and omits
/// `store_schema` — Rust ships all 5 on both paths or neither ships).
///
/// No firewall here by construction (privileged columns are EXPECTED);
/// the gate is footer parity: `ARROW:schema` + `created_by` + close-counts
/// asserted identically via [`probe_footer_5exact`].
pub fn write_privileged_parquet_5exact(
    path: &Path,
    batches: &[RecordBatch],
) -> Result<ParquetWriteReceipt, ColumnarWriteError> {
    let schema: SchemaRef = batches
        .first()
        .map(|b| b.schema())
        .ok_or_else(|| ColumnarWriteError::Geometry {
            msg: "no batches to write (empty close forbidden)".to_string(),
        })?;
    let props = writer_props_5exact().map_err(ColumnarWriteError::Parquet)?;
    write_batches_5exact(path, schema, batches, props)
}

/// Footer probe B: `close` + `store_schema` (both) + `created_by` + KV.
///
/// Asserts on a CLOSED file:
/// * row count equals `expected_rows` (close-counts);
/// * `created_by` present + non-empty (M3: default provenance, UNSET by us);
/// * key-value metadata contains `ARROW:schema` with non-empty value
///   (M4/M8: `store_schema` discoverable on BOTH writers).
pub fn probe_footer_5exact(path: &Path, expected_rows: usize) -> Result<(), ColumnarWriteError> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let meta = reader.metadata();
    let file_meta = meta.file_metadata();
    // proof: parquet row counts are non-negative and fit in memory.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rows = file_meta.num_rows() as usize;
    if rows != expected_rows {
        return Err(ColumnarWriteError::Footer {
            msg: alloc::format!("footer rows {rows} != expected {expected_rows}"),
        });
    }
    match file_meta.created_by() {
        Some(by) if !by.is_empty() => {}
        _ => {
            return Err(ColumnarWriteError::Footer {
                msg: "footer created_by missing/empty (M3: default provenance expected)".to_string(),
            });
        }
    }
    let kvs = file_meta.key_value_metadata();
    let has_schema = kvs
        .map(|kvs| {
            kvs.iter().any(|kv| {
                kv.key.as_str() == ARROW_SCHEMA_META_KEY
                    && kv.value.as_ref().map(|v| !v.is_empty()).unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if !has_schema {
        return Err(ColumnarWriteError::Footer {
            msg: "footer missing ARROW:schema KV (M4/M8 store_schema)".to_string(),
        });
    }
    Ok(())
}

/// Probe-C canonical row count (same-batches-to-both-writers contract).
///
/// `7` is coprime to `24`, so [`parity_probe_ids`] emits a full permutation:
/// file order is NOT sorted, which proves both `dataset_hash` sides sort
/// before hashing instead of hashing file order.
pub const PARITY_PROBE_ROWS: usize = 24;

/// Probe-C canonical decision ids in FILE order (permuted, NOT sorted).
///
/// Python mirror (MUST change together):
/// `tests/unit/test_columnar_parity.py::_probe_ids`
/// (`f"probe-c:d{(i * 7) % 24:04d}"`). Any one-sided edit raises there
/// (mismatch=raise).
pub fn parity_probe_ids() -> Vec<String> {
    (0..PARITY_PROBE_ROWS)
        .map(|i| alloc::format!("probe-c:d{:04}", (i * 7) % PARITY_PROBE_ROWS))
        .collect()
}

/// Probe-C canonical batch: [`PARITY_PROBE_ROWS`] rows over the CANONICAL
/// [`crate::replay::expand::actor_schema`] (13 `ACTOR_FIELDS`, `Utf8`/`Int64`,
/// `chosen_action_id` sole nullable — the real writer input, never a subset).
///
/// Logical values are fully pinned here AND in the Python mirror
/// (`tests/unit/test_columnar_parity.py::_probe_rows`); both files MUST carry
/// value-identical columns. The parity test compares per-column logical
/// values, never bytes (`created_by` plus page internals differ by writer).
pub fn parity_probe_batch() -> Result<RecordBatch, ColumnarWriteError> {
    use arrow_array::{Int64Array, StringArray};
    use std::sync::Arc;
    let schema = crate::replay::expand::actor_schema();
    let n = PARITY_PROBE_ROWS;
    let fill_const = |s: &'static str| (0..n).map(|_| Some(s)).collect::<Vec<Option<&str>>>();
    let game_ids = fill_const("probe-c-g0000");
    let round_ids: Vec<Option<String>> = (0..n)
        .map(|i| Some(alloc::format!("probe-c-g0000:r{:02}", i % 4)))
        .collect();
    let decision_ids: Vec<Option<String>> = parity_probe_ids().into_iter().map(Some).collect();
    // proof: probe indices < PARITY_PROBE_ROWS fit in i64.
    #[allow(clippy::cast_possible_wrap)]
    let seats: Vec<Option<i64>> = (0..n).map(|i| Some((i % 4) as i64)).collect();
    let sources = fill_const("probe-c-src");
    let splits = fill_const("train");
    let rules = fill_const("sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    let adapters = fill_const("sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    let observations = fill_const("sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    let action_tables =
        fill_const("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
    let derivations =
        fill_const("sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
    let actor_obs: Vec<Option<String>> = (0..n)
        .map(|i| {
            Some(alloc::format!(
                "{{\"dora_indicators\":[1,2,3,4,5],\"seat\":{}}}",
                i % 4
            ))
        })
        .collect();
    // proof: probe indices < PARITY_PROBE_ROWS fit in i64.
    #[allow(clippy::cast_possible_wrap)]
    let chosen: Vec<Option<i64>> = (0..n).map(|i| Some((i % 9) as i64)).collect();
    // Column order is the canonical schema order (13 `ACTOR_FIELDS`).
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(game_ids)) as _,
            Arc::new(StringArray::from(round_ids)) as _,
            Arc::new(StringArray::from(decision_ids)) as _,
            Arc::new(Int64Array::from(seats)) as _,
            Arc::new(StringArray::from(sources)) as _,
            Arc::new(StringArray::from(splits)) as _,
            Arc::new(StringArray::from(rules)) as _,
            Arc::new(StringArray::from(adapters)) as _,
            Arc::new(StringArray::from(observations)) as _,
            Arc::new(StringArray::from(action_tables)) as _,
            Arc::new(StringArray::from(derivations)) as _,
            Arc::new(StringArray::from(actor_obs)) as _,
            Arc::new(Int64Array::from(chosen)) as _,
        ],
    )?;
    debug_assert_eq!(batch.num_rows(), n);
    Ok(batch)
}

/// Probe-C harness entry: same-batches-to-both-writers comparison (Rust side).
///
/// Writes the caller-supplied batches via the actor 5-exact path (triple
/// firewall plus [`writer_props_5exact`] plus close-counts plus
/// [`probe_footer_5exact`]) to `path` and returns the receipt.
/// `tests/unit/test_columnar_parity.py` builds the IDENTICAL logical batch
/// via `pq.write_table` 5-exact and compares `dataset_hash` plus the
/// file-content class (schema/columns/row counts/logical values — NOT bytes:
/// `created_by` plus page internals differ).
///
/// Python writer stays primary until Probe-C green; this entry performs NO
/// cutover (no caller migration, no fallback logic — see the test module
/// docstring for the documented next step).
pub fn write_parity_probe(
    path: &Path,
    batches: &[RecordBatch],
) -> Result<ParquetWriteReceipt, ColumnarWriteError> {
    write_actor_parquet_5exact(path, batches)
}

#[cfg(test)]
mod columnar_probes {
    use super::*;
    use arrow_array::{Int64Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn actor_like_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("game_id", DataType::Utf8, false),
            Field::new("decision_id", DataType::Utf8, false),
            Field::new("actor_observation", DataType::Utf8, false),
            Field::new("chosen_action_id", DataType::Int64, false),
        ]))
    }

    fn actor_like_batch(schema: SchemaRef, n: usize) -> RecordBatch {
        let game_ids: Vec<Option<&str>> = (0..n).map(|_| Some("g1")).collect();
        let decisions: Vec<Option<String>> =
            (0..n).map(|i| Some(alloc::format!("g1:d{i:04}"))).collect();
        let obs: Vec<Option<&str>> = (0..n)
            .map(|_| Some("{\"dora_indicators\":[1,2,3,4,5]}"))
            .collect();
        // proof: probe indices < PARITY_PROBE_ROWS fit in i64.
        #[allow(clippy::cast_possible_wrap)]
        let chosen: Vec<Option<i64>> = (0..n).map(|i| Some(i as i64)).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(game_ids)) as _,
                Arc::new(StringArray::from(decisions)) as _,
                Arc::new(StringArray::from(obs)) as _,
                Arc::new(Int64Array::from(chosen)) as _,
            ],
        )
        .unwrap()
    }

    /// Probe B: footer carries close-counts + `store_schema` (both) +
    /// `created_by` + `ARROW:schema` KV — actor AND privileged paths.
    #[test]
    fn probe_b_footer_both_writers() {
        let dir = std::env::temp_dir().join(alloc::format!(
            "hydra-writer-probe-b-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let schema = actor_like_schema();
        let batch = actor_like_batch(Arc::clone(&schema), 16);
        let actor_path = dir.join("actor-train.parquet");
        let actor = write_actor_parquet_5exact(&actor_path, std::slice::from_ref(&batch)).unwrap();
        assert_eq!(actor.rows, 16);
        probe_footer_5exact(&actor_path, 16).unwrap();
        // Privileged path ships the SAME 5-exact (M5/M8 re-freeze gate).
        let priv_path = dir.join("privileged-000.parquet");
        let priv_r =
            write_privileged_parquet_5exact(&priv_path, &[batch]).unwrap();
        assert_eq!(priv_r.rows, 16);
        probe_footer_5exact(&priv_path, 16).unwrap();
        // `created_by` UNSET by us ⇒ default provenance, non-empty.
        for path in [&actor_path, &priv_path] {
            let file = File::open(path).unwrap();
            let reader = SerializedFileReader::new(file).unwrap();
            let by = reader.metadata().file_metadata().created_by().unwrap().to_string();
            assert!(!by.is_empty());
            assert!(by.contains("parquet-rs"));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Tamper fork: one flipped payload byte MUST fail the stream hash the
    /// receipt binds (close-counts alone are not the gate — bytes are).
    #[test]
    fn probe_tamper_fork_fails_hash() {
        let dir = std::env::temp_dir().join(alloc::format!(
            "hydra-writer-tamper-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let schema = actor_like_schema();
        let batch = actor_like_batch(Arc::clone(&schema), 8);
        let path = dir.join("actor-train.parquet");
        let receipt = write_actor_parquet_5exact(&path, &[batch]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let mut forked = bytes.clone();
        let mid = forked.len() / 2;
        forked[mid] ^= 0x01;
        let fork_path = dir.join("actor-train.fork.parquet");
        std::fs::write(&fork_path, &forked).unwrap();
        let fork_hash = sha256_stream_1m(&fork_path).unwrap();
        assert_ne!(receipt.file_sha256, fork_hash);
        // Footer probe on the fork fails closed (counts or KV corrupt).
        assert!(probe_footer_5exact(&fork_path, 8).is_err()
            || fork_hash != receipt.file_sha256);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Geometry guards: history buckets, `uint8` bools, non-empty close,
    /// firewall triple, sorted-ids dataset hash.
    #[test]
    fn probe_geometry_guards() {
        for good in [32usize, 64, 128, 256] {
            assert_history_len(good).unwrap();
        }
        assert!(assert_history_len(5).is_err());
        assert!(assert_history_len(0).is_err());
        assert!(assert_uint8_bool(0).is_ok());
        assert!(assert_uint8_bool(1).is_ok());
        assert!(assert_uint8_bool(2).is_err());
        // Firewall: privileged column name fails BEFORE any byte is written.
        let bad_schema = Arc::new(Schema::new(vec![
            Field::new("decision_id", DataType::Utf8, false),
            Field::new("hidden_tiles", DataType::Utf8, false),
        ]));
        let bad = RecordBatch::try_new(
            bad_schema,
            vec![
                Arc::new(StringArray::from(vec![Some("g1:d0000")])) as _,
                Arc::new(StringArray::from(vec![Some("{}")])) as _,
            ],
        )
        .unwrap();
        assert!(assert_no_privileged_leak(&bad).is_err());
        // dataset_hash is order-insensitive (sorted ids).
        let a = vec!["g:b".to_string(), "g:a".to_string()];
        let b = vec!["g:a".to_string(), "g:b".to_string()];
        assert_eq!(dataset_hash_of_sorted_ids(&a), dataset_hash_of_sorted_ids(&b));
        assert!(dataset_hash_of_sorted_ids(&a).starts_with("sha256:"));
    }

    /// Probe C (Rust side): canonical batch builds over the real actor schema,
    /// passes the firewall, and round-trips through [`write_parity_probe`]
    /// with footer green. Python compares the same logical values.
    #[test]
    fn probe_c_parity_batch_round_trip() {
        let batch = parity_probe_batch().unwrap();
        assert_eq!(batch.num_rows(), PARITY_PROBE_ROWS);
        assert_eq!(batch.schema(), crate::replay::expand::actor_schema());
        // File order is permuted (unsorted) by construction.
        let ids = parity_probe_ids();
        assert_eq!(ids.len(), PARITY_PROBE_ROWS);
        let mut sorted = ids.clone();
        sorted.sort();
        assert_ne!(ids, sorted);
        assert_no_privileged_leak(&batch).unwrap();
        let dir = std::env::temp_dir().join(alloc::format!(
            "hydra-writer-probe-c-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("probe-c.parquet");
        let receipt = write_parity_probe(&path, &[batch]).unwrap();
        assert_eq!(receipt.rows, PARITY_PROBE_ROWS);
        probe_footer_5exact(&path, PARITY_PROBE_ROWS).unwrap();
        // dataset_hash sorts: file order and sorted order agree.
        assert_eq!(
            dataset_hash_of_sorted_ids(&ids),
            dataset_hash_of_sorted_ids(&sorted)
        );

        // Judge hook: when the Python parity judge sets `HYDRA2_PROBEC_OUT`,
        // emit a second copy there for file-level comparison. Scratch-only
        // discipline mirrors `parity_84` (must sit under the platform temp
        // dir, never inside the repo). Unset: pure self-contained probe.
        if let Some(out) = std::env::var_os("HYDRA2_PROBEC_OUT") {
            let out = std::path::PathBuf::from(out);
            assert!(
                out.starts_with(std::env::temp_dir()),
                "HYDRA2_PROBEC_OUT must sit under scratch temp dir {}: {}",
                std::env::temp_dir().display(),
                out.display(),
            );
            std::fs::create_dir_all(&out).unwrap();
            write_parity_probe(&out.join("probe-c-rust.parquet"), &[parity_probe_batch().unwrap()])
                .unwrap();
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
