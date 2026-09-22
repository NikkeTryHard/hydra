//! resume: thin feed/resume/RNG-restore boundary over `hydra-feed`.
//!
//! DAG: this module depends on the feed crate + pyo3 ONLY (same edge as
//! `canon_rng`; no shard/search arrow edge, no new dependency). FORBIDS:
//! parse/encode/hash/pool/slot-fill/plane-fill logic — arg-check + detach +
//! struct return only. All bytes/hash/RNG math lives in
//! `hydra_feed::{canon, digest, rng, manifest}`; this file only validates
//! arguments attached, runs the feed pure functions detached, and wraps the
//! results attached.
//!
//! Scope split (no dual ownership):
//! - Ring slots (pinned fill, `ring_fill_batch`, `bucket_for_length`) live in
//!   sibling-owned `ring.rs` (Phase5ModelLoss). This module NEVER fills slots.
//! - Plane fill (stage-then-commit game expansion) lives in `stream.rs`.
//! - This module owns the resume side of feed: snapshot envelope + version,
//!   buffer-entries shape check, reservoir/prefix blobs, cursor round-trip,
//!   RNG-snapshot restore/words, `epoch_seed`, and state-tree sha helpers.
//!   Checkpoint APPLY stays Python (`load_state_dict` + torch
//!   `_restore_rng_state` are torch APIs and must stay); verify-before-mutate
//!   order stays Python — the helpers here only hash/compare bytes.
//!
//! Snapshot discipline (entries, {key, block, pos}): a resume snapshot is the
//! buffer-entries list (each `{key, path, offset, split, rows}`, mirroring
//! `stream_dataset_buffer.buffer_snapshot`) PLUS one RNG triple
//! `{key, block, pos}` (`feed::rng::StreamSnapshot`: stream identity +
//! block index of the next undrawn word's block + words already consumed in
//! that block, `0..4`). `pos == 4` MUST be normalized to `(block + 1, 0)` by
//! the caller before storing — this boundary rejects `pos >= 4` fail-closed.
//!
//! Version bump + dual-read one release: this module writes snapshot version
//! 2 (`RESUME_SNAPSHOT_VERSION`) and reads 1..=2 (`check_snapshot_version`).
//! Generation-1 snapshots (Python oracle sidecars) decode drain-only for one
//! release; unknown versions are `ValueError` (callers treat it as a
//! snapshot miss and fall back to the normal fill, never coerce). Drop the
//! V1 arm after one release. Same pattern as the feed's own M7 dual-read
//! (`RESERVOIR_BLOB_VERSION = 2` + `V1 = 1`, `SCAN_CACHE_VERSION = 2` +
//! `V1 = 1` in `hydra-feed/src/manifest.rs`).
//!
//! State-tree sha note: the canonical `state_tree` form stays
//! Python-visible (`runtime/checkpoint.py:80-126`); the per-section digest
//! lowers here to Rust `sha2` over re-canonicalized (`serde_jcs`) bytes —
//! bit-identical to `hash_state_tree` (KAT-1/2 gate at verify).
//! `state_tree_sha_diff` returns per-section match bits for
//! resume-mismatch TRIAGE ONLY and never bypasses the fail-closed raise in
//! `load_checkpoint` / `_load_resume_envelope`.
//!
//! RNG-restore note: torch CPU/CUDA states restore via torch APIs in Python
//! (`capture_rng_state` / `_restore_rng_state`); the bridge owns the buffer
//! RNG triple (`RngSnapshot` + `snapshot_words`), `epoch_seed = data_seed +
//! epoch` (wrapping add, mirroring `stream_checkpoint.py` `epoch_seed`), and
//! the prefix-hash chain verify. `_verify_rng_anchors` compare stays Python.
//! NEW streams only here: held-out splits stay `torch.randperm` oracle and
//! sha-Gumbels replicate verbatim elsewhere (Wave5 B1/B2) — never Philox.
//!
//! m2 warmup disambiguation (editing pass, kept as the single note): the
//! three warmup counts are DISTINCT sites, never unified — kbench
//! `warmup_per_bucket = 2` vs manual-graph warmup 3 vs DDP-11. This module
//! performs no warmup; feed drain callers keep their own count.
//!
//! Bytes path, NO DLPack/Arrow here: inputs are `&[u8]` / dicts in,
//! `sha256:<hex>` text + bytes out. No PyCapsule is ever constructed in this
//! module, so the capsule-outside-detach rule is N/A here (compute detached,
//! wrap attached still holds).
//!
//! Concurrency pattern (matches `stream.rs` + `canon_rng.rs`): validate
//! attached, `py.detach(|| ...)` around every batch/blocking section
//! (fan-out/join never holds the interpreter), `MutexExt::lock_py_attached`
//! stats mutex, frozen pyclasses only (`&self` methods). Single-cdylib tree:
//! this module registers as the `hydra2._native.resume` submodule via
//! `register`, mirroring `canon_rng::register`.
//!
//! GIL attestation (`gil_used=false` default ships on PyO3 >=0.28 — spelled
//! once in `canon_rng.rs` wave-wide, no per-submodule annotation): the single
//! `#[pymodule]` entry plus every `PyModule::new` submodule inherits the
//! `Py_MOD_GIL_NOT_USED` default.
//!
//! VERIFIED feed API (read in-tree, never guessed):
//! - `hydra_feed::manifest::encode_reservoir_blob(&[&[u8]], i32)`
//! - `hydra_feed::manifest::decode_reservoir_blob(&[u8])` (dual-read v1+v2)
//! - `hydra_feed::manifest::StreamCursor { file_index, byte_offset,
//!   games_seen, seed, epoch, shuffle_pos }` + `to_dict/from_dict`
//! - `hydra_feed::rng::StreamKey { domain: [u8; 14], len: u32, seed: [u8; 32] }`
//! - `hydra_feed::rng::philox_block(&StreamKey, u64) -> [u32; 4]` (pure words;
//!   snapshot draws replicate `restore_snapshot` + drain through it, pinned by
//!   feed KATs `kat4_block_replay_and_pure_vs_stream` /
//!   `kat4_position_in_block_replay` / `kat4_snapshot_restore_roundtrip` — no
//!   stateful stream, no rand edge in this crate)
//! - `hydra_feed::digest::sha256_hex(&[u8])`
//! - `hydra_feed::canon::parse_canonical_bytes` + `hydra_feed::digest::of_canonical`
//!   (used opaquely — the `serde_json::Value` type is never named here, so no
//!   new bridge dependency).

use std::collections::BTreeMap;
use std::sync::Mutex;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBool, PyBytes, PyDict, PyModule};

/// Snapshot envelope version this module writes (bump; readers dual-read).
pub const RESUME_SNAPSHOT_VERSION: u32 = 2;

/// Snapshot envelope version the Python oracle wrote — readable drain-only
/// for one release (miss, never coerce; drop after one release).
pub const RESUME_SNAPSHOT_VERSION_V1: u32 = 1;

/// zstd level for the reservoir/prefix blob encode path (mirrors the
/// end-state `copy_encode(level=3)` + `sha256:` record; `torch.save` payload
/// bytes stay Python).
pub const RESERVOIR_BLOB_LEVEL: i32 = 3;

/// Cap on `snapshot_words` draws per call (fail-closed bound on one batch).
const MAX_SNAPSHOT_WORDS: usize = 1 << 20;

/// Buffer-entry keys exactly (`stream_dataset_buffer.buffer_snapshot`).
const ENTRY_KEYS: [&str; 5] = ["key", "path", "offset", "split", "rows"];

/// Cumulative judge counters (observability only, never identity).
#[derive(Debug, Default, Clone, Copy)]
struct JudgeStats {
    snapshots: u64,
    blobs: u64,
    restores: u64,
}

static JUDGE_STATS: Mutex<JudgeStats> = Mutex::new(JudgeStats {
    snapshots: 0,
    blobs: 0,
    restores: 0,
});

fn bump_snapshots(py: Python<'_>, n: u64) {
    if let Ok(mut guard) = JUDGE_STATS.lock_py_attached(py) {
        guard.snapshots = guard.snapshots.saturating_add(n);
    }
}

fn bump_blobs(py: Python<'_>, n: u64) {
    if let Ok(mut guard) = JUDGE_STATS.lock_py_attached(py) {
        guard.blobs = guard.blobs.saturating_add(n);
    }
}

fn bump_restores(py: Python<'_>, n: u64) {
    if let Ok(mut guard) = JUDGE_STATS.lock_py_attached(py) {
        guard.restores = guard.restores.saturating_add(n);
    }
}

/// Frozen resume config: snapshot version written + blob level. No `&mut`
/// methods; shared state (if ever added) goes behind a `Mutex`.
#[pyclass(name = "ResumeConfig", frozen, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ResumeConfig {
    /// Snapshot envelope version this writer emits (always 2).
    #[pyo3(get)]
    pub snapshot_version: u32,
    /// zstd level for the blob encode path.
    #[pyo3(get)]
    pub blob_level: i32,
}

#[pymethods]
impl ResumeConfig {
    #[new]
    #[pyo3(signature = (blob_level=None))]
    fn new(blob_level: Option<i32>) -> PyResult<Self> {
        let level = blob_level.unwrap_or(RESERVOIR_BLOB_LEVEL);
        if !(1..=22).contains(&level) {
            return Err(PyValueError::new_err(
                "resume config blob_level must be 1..=22 (zstd range)",
            ));
        }
        Ok(Self {
            snapshot_version: RESUME_SNAPSHOT_VERSION,
            blob_level: level,
        })
    }
}

/// Frozen RNG-snapshot triple `{key, block, pos}`: everything needed for
/// exact continuation of one buffer RNG stream.
///
/// `domain` is the 1..=14 significant tag bytes (zero-padded to `[u8; 14]`
/// like `canon_rng.StreamSpec`); `seed` is exactly 32 bytes; `block` is a
/// BLOCK index (never a word index); `pos` counts words already consumed in
/// `block` (`0..4` — `pos == 4` must be normalized to `(block + 1, 0)` by
/// the caller before storing, rejected here).
#[pyclass(name = "RngSnapshot", frozen, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct RngSnapshot {
    domain: [u8; 14],
    domain_len: u32,
    seed: [u8; 32],
    block: u64,
    pos: u8,
}

#[pymethods]
impl RngSnapshot {
    #[new]
    fn new(domain: Vec<u8>, seed: Vec<u8>, block: u64, pos: u8) -> PyResult<Self> {
        if domain.is_empty() || domain.len() > 14 {
            return Err(PyValueError::new_err(
                "resume snapshot domain must be 1..=14 bytes",
            ));
        }
        if seed.len() != 32 {
            return Err(PyValueError::new_err(
                "resume snapshot seed must be exactly 32 bytes",
            ));
        }
        if pos >= 4 {
            return Err(PyValueError::new_err(
                "resume snapshot pos must be 0..4 (normalize pos == 4 to (block + 1, 0) before storing)",
            ));
        }
        let mut padded_domain = [0u8; 14];
        padded_domain[..domain.len()].copy_from_slice(&domain);
        let mut fixed_seed = [0u8; 32];
        fixed_seed.copy_from_slice(&seed);
        // proof: `domain.len()` in 1..=14 (checked above), fits `u32`.
        #[allow(clippy::cast_possible_truncation)]
        let domain_len: u32 = domain.len() as u32;
        Ok(Self {
            domain: padded_domain,
            domain_len,
            seed: fixed_seed,
            block,
            pos,
        })
    }

    #[getter]
    fn domain_bytes(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.domain[..self.domain_len as usize]).unbind()
    }

    #[getter]
    fn seed_bytes(&self, py: Python<'_>) -> Py<PyBytes> {
        PyBytes::new(py, &self.seed).unbind()
    }

    #[getter]
    fn block(&self) -> u64 {
        self.block
    }

    #[getter]
    fn pos(&self) -> u8 {
        self.pos
    }
}

/// Snapshot version this writer emits (bump note lives at module top).
#[pyfunction]
fn snapshot_version() -> u32 {
    RESUME_SNAPSHOT_VERSION
}

/// Oldest snapshot version readable drain-only this release (dual-read).
#[pyfunction]
fn snapshot_min_reader() -> u32 {
    RESUME_SNAPSHOT_VERSION_V1
}

fn snapshot_readable(version: u32) -> bool {
    version == RESUME_SNAPSHOT_VERSION || version == RESUME_SNAPSHOT_VERSION_V1
}

/// Dual-read gate: accept 1..=2 this release, echo the version; anything
/// else is `ValueError` (callers treat it as a snapshot miss and fall back
/// to the normal fill, never coerce).
#[pyfunction]
fn check_snapshot_version(version: u32) -> PyResult<u32> {
    if !snapshot_readable(version) {
        return Err(PyValueError::new_err(format!(
            "resume snapshot version {version} not readable (dual-read 1..=2 this release)"
        )));
    }
    Ok(version)
}

/// Buffer-entries shape check: each entry carries exactly
/// `{key, path, offset, split, rows}` with non-empty-string
/// key/path/split, non-negative-int offset, and positive-int rows
/// (mirrors `_parse_dataset_buffer_sidecar` entry gates; the row-hash and
/// counter gates stay Python). Returns the entry count. Never raises on
/// valid input; every shape violation is `ValueError`.
#[pyfunction]
fn check_buffer_entries(entries: Vec<Bound<'_, PyDict>>) -> PyResult<usize> {
    let count = entries.len();
    for (idx, entry) in entries.iter().enumerate() {
        let mut unknown: Vec<String> = Vec::new();
        let key_list = entry.keys();
        for key in key_list.iter() {
            let name: String = key.extract().map_err(|_| {
                PyValueError::new_err(format!("buffer entry {idx}: key must be str"))
            })?;
            if !ENTRY_KEYS.contains(&name.as_str()) {
                unknown.push(name);
            }
        }
        if !unknown.is_empty() {
            unknown.sort();
            return Err(PyValueError::new_err(format!(
                "buffer entry {idx} unknown keys {unknown:?}"
            )));
        }
        let non_empty_str = |field: &str| -> PyResult<String> {
            let value = entry.get_item(field).map_err(|e| {
                PyValueError::new_err(format!(
                    "buffer entry {idx} field {field:?} unreadable: {e}"
                ))
            })?;
            let Some(value) = value else {
                return Err(PyValueError::new_err(format!(
                    "buffer entry {idx} field {field:?} missing"
                )));
            };
            let text: String = value.extract().map_err(|_| {
                PyValueError::new_err(format!("buffer entry {idx} field {field:?} must be str"))
            })?;
            if text.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "buffer entry {idx} field {field:?} must be non-empty"
                )));
            }
            Ok(text)
        };
        let non_negative_int = |field: &str| -> PyResult<i64> {
            let value = entry.get_item(field).map_err(|e| {
                PyValueError::new_err(format!(
                    "buffer entry {idx} field {field:?} unreadable: {e}"
                ))
            })?;
            let Some(value) = value else {
                return Err(PyValueError::new_err(format!(
                    "buffer entry {idx} field {field:?} missing"
                )));
            };
            if value.is_instance_of::<PyBool>() {
                return Err(PyValueError::new_err(format!(
                    "buffer entry {idx} field {field:?} must be int, not bool"
                )));
            }
            let number: i64 = value.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "buffer entry {idx} field {field:?} must be a non-negative int"
                ))
            })?;
            if number < 0 {
                return Err(PyValueError::new_err(format!(
                    "buffer entry {idx} field {field:?} must be a non-negative int"
                )));
            }
            Ok(number)
        };
        let _ = non_empty_str("key")?;
        let _ = non_empty_str("path")?;
        let _ = non_empty_str("split")?;
        let _ = non_negative_int("offset")?;
        let rows = non_negative_int("rows")?;
        if rows <= 0 {
            return Err(PyValueError::new_err(format!(
                "buffer entry {idx} field \"rows\" must be a positive int"
            )));
        }
    }
    Ok(count)
}

/// Reservoir-blob encode: per-game raw bytes in, `(bytes, count,
/// uncompressed_bytes, blob_sha256)` out. The whole zstd fan-out runs
/// detached; `level` defaults to 3 (`RESERVOIR_BLOB_LEVEL`; validated
/// 1..=22 attached). Layout and dual-read decode live feed-side
/// (`manifest::encode/decode_reservoir_blob`); `blob_sha256` is the
/// `sha256:` record the sidecar carries.
#[pyfunction]
#[pyo3(signature = (frames, level=None))]
fn encode_reservoir(
    py: Python<'_>,
    frames: Vec<Vec<u8>>,
    level: Option<i32>,
) -> PyResult<(Py<PyBytes>, u32, u64, String)> {
    let want = level.unwrap_or(RESERVOIR_BLOB_LEVEL);
    if !(1..=22).contains(&want) {
        return Err(PyValueError::new_err(
            "resume encode_reservoir level must be 1..=22 (zstd range)",
        ));
    }
    let (bytes, index) = py
        .detach(
            || -> Result<(Vec<u8>, hydra_feed::manifest::ReservoirIndex), String> {
                let refs: Vec<&[u8]> = frames.iter().map(|f| &f[..]).collect();
                hydra_feed::manifest::encode_reservoir_blob(&refs, want)
                    .map_err(|e| format!("reservoir encode failed: {e:?}"))
            },
        )
        .map_err(PyValueError::new_err)?;
    let out = PyBytes::new(py, &bytes).unbind();
    bump_blobs(py, 1);
    Ok((
        out,
        index.count,
        index.uncompressed_bytes,
        index.blob_sha256,
    ))
}

/// Reservoir-blob decode (dual-read both generations): blob bytes in,
/// per-game raw bytes out (buffer order). Bad magic, unknown version,
/// truncated frames, and trailing bytes are `ValueError` — callers treat
/// them as a snapshot miss and fall back to the normal fill.
#[pyfunction]
fn decode_reservoir(py: Python<'_>, data: Vec<u8>) -> PyResult<Vec<Py<PyBytes>>> {
    let raws = py
        .detach(|| {
            hydra_feed::manifest::decode_reservoir_blob(&data)
                .map_err(|e| format!("reservoir decode failed: {e:?}"))
        })
        .map_err(PyValueError::new_err)?;
    let out: Vec<Py<PyBytes>> = raws.iter().map(|r| PyBytes::new(py, r).unbind()).collect();
    bump_blobs(py, 1);
    Ok(out)
}

fn verify_prefix(
    data: &[u8],
    expected_count: usize,
    expected_sha: &str,
) -> Result<Vec<String>, String> {
    if !expected_sha.starts_with("sha256:") || expected_sha == "sha256:" {
        return Err("prefix record digest must be sha256:<hex>".to_string());
    }
    let actual = hydra_feed::digest::sha256_hex(data);
    if actual != expected_sha {
        return Err("prefix blob digest mismatch".to_string());
    }
    let text = std::str::from_utf8(data).map_err(|_| "prefix blob must be UTF-8".to_string())?;
    let lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    if lines.len() != expected_count {
        return Err(format!(
            "prefix count mismatch: {} lines != {expected_count} recorded",
            lines.len()
        ));
    }
    for sha in &lines {
        if !sha.starts_with("sha256:") || sha == "sha256:" {
            return Err("prefix entry must be sha256:<hex>".to_string());
        }
    }
    let mut ordered = lines.clone();
    ordered.sort();
    if lines != ordered {
        return Err("prefix entries must be sorted".to_string());
    }
    Ok(lines)
}

/// Prefix-hash chain verify: blob bytes + recorded `(count, sha256:)` in,
/// entry lines out. Mirrors `_load_prefix_hashes` gates (blob sha, count,
/// per-entry `sha256:` shape, sorted order) — all detached. Tamper raises
/// `ValueError`; callers fail closed before mutating live state.
#[pyfunction]
fn verify_prefix_blob(
    py: Python<'_>,
    data: Vec<u8>,
    expected_count: usize,
    expected_sha: &str,
) -> PyResult<Vec<String>> {
    let owned_sha = expected_sha.to_string();
    let lines = py
        .detach(|| verify_prefix(&data, expected_count, &owned_sha))
        .map_err(PyValueError::new_err)?;
    bump_blobs(py, 1);
    Ok(lines)
}

/// `epoch_seed = data_seed + epoch` (wrapping add, mirroring the sidecar
/// `epoch_seed` in `stream_checkpoint.py`). Pure arithmetic, stays attached.
#[pyfunction]
fn epoch_seed(data_seed: u64, epoch: u64) -> u64 {
    data_seed.wrapping_add(epoch)
}

/// Draw `n` words from `(key, block, pos)` through pure
/// [`philox_block`](hydra_feed::rng::philox_block) words — no stateful
/// stream, no `RngCore` edge (this crate has no rand dep by design).
///
/// Word `(block, pos) + i` is `philox_block(key, block + (pos + i) / 4)
/// [(pos + i) % 4]`, spilling into following blocks. Feed KATs pin pure
/// == stateful exactly (`kat4_block_replay_and_pure_vs_stream`,
/// `kat4_position_in_block_replay`, `kat4_snapshot_restore_roundtrip`),
/// so this replicates `restore_snapshot` + drain word-for-word.
fn draw_snapshot_words(
    domain: &[u8],
    seed: &[u8; 32],
    block: u64,
    pos: u8,
    n: usize,
) -> Result<Vec<u32>, String> {
    if domain.is_empty() || domain.len() > 14 {
        return Err("resume snapshot domain must be 1..=14 bytes".to_string());
    }
    if pos >= 4 {
        return Err(
            "resume snapshot pos must be 0..4 (normalize pos == 4 to (block + 1, 0) before storing)"
                .to_string(),
        );
    }
    if n == 0 || n > MAX_SNAPSHOT_WORDS {
        return Err(format!(
            "resume snapshot draws must be 1..={MAX_SNAPSHOT_WORDS}"
        ));
    }
    let mut padded = [0u8; 14];
    padded[..domain.len()].copy_from_slice(domain);
    // proof: `domain.len()` in 1..=14 (checked above), fits `u32`.
    #[allow(clippy::cast_possible_truncation)]
    let domain_len: u32 = domain.len() as u32;
    let key = hydra_feed::rng::StreamKey::new(padded, domain_len, *seed);
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let at = pos as usize + i;
        let blk = block.wrapping_add((at / 4) as u64);
        let off = at % 4;
        out.push(hydra_feed::rng::philox_block(&key, blk)[off]);
        i += 1;
    }
    Ok(out)
}

/// RNG-restore words: re-open the `(key, block, pos)` triple detached and
/// draw `n` words (1..=2^20). Restoring `(block, 0)` prefixes
/// `philox_block(key, block)` exactly (pinned by the in-module tests);
/// `pos` advance matches word-for-word. Drain recomputation consumes these
/// words through `bounded` (canon-owned Lemire, never `% n`).
#[pyfunction]
fn snapshot_words(py: Python<'_>, snap: &RngSnapshot, n: usize) -> PyResult<Vec<u32>> {
    if n == 0 || n > MAX_SNAPSHOT_WORDS {
        return Err(PyValueError::new_err(format!(
            "resume snapshot_words n must be 1..={MAX_SNAPSHOT_WORDS}"
        )));
    }
    let domain = snap.domain[..snap.domain_len as usize].to_vec();
    let seed = snap.seed;
    let (block, pos) = (snap.block, snap.pos);
    let words = py
        .detach(|| draw_snapshot_words(&domain, &seed, block, pos, n))
        .map_err(PyValueError::new_err)?;
    bump_restores(py, 1);
    Ok(words)
}

fn cursor_from_pairs(pairs: &[(String, i64)]) -> Result<BTreeMap<String, i64>, String> {
    let mut raw = BTreeMap::new();
    for (key, value) in pairs {
        if *value < 0 {
            return Err(format!("cursor field {key:?} must be a non-negative int"));
        }
        raw.insert(key.clone(), *value);
    }
    Ok(raw)
}

/// Cursor pack: six frontier fields in, YAML-resume mapping out (attached;
/// pure, mirrors `StreamCursor::to_dict` key set exactly:
/// `byte_offset/epoch/file_index/games_seen/seed/shuffle_pos`).
#[pyfunction]
#[pyo3(signature = (*, file_index, byte_offset, games_seen, seed, epoch, shuffle_pos))]
fn pack_cursor(
    py: Python<'_>,
    file_index: u64,
    byte_offset: u64,
    games_seen: u64,
    seed: u64,
    epoch: u64,
    shuffle_pos: u64,
) -> PyResult<Py<PyDict>> {
    let cursor = hydra_feed::manifest::StreamCursor {
        file_index,
        byte_offset,
        games_seen,
        seed,
        epoch,
        shuffle_pos,
    };
    let dict = PyDict::new(py);
    for (key, value) in cursor.to_dict() {
        dict.set_item(key, value)
            .map_err(|e| PyOSError::new_err(format!("resume pack_cursor failed: {e}")))?;
    }
    Ok(dict.unbind())
}

/// Cursor unpack: YAML-resume mapping in, six frontier fields out
/// (attached validation, NOT getters-only). Missing/negative/bool/non-int
/// fields are `ValueError` (mirrors `StreamCursor::from_dict` +
/// `stream_read.py` `type(value) is not int or value < 0`).
#[pyfunction]
fn unpack_cursor(raw: Bound<'_, PyDict>) -> PyResult<(u64, u64, u64, u64, u64, u64)> {
    let mut pairs: Vec<(String, i64)> = Vec::with_capacity(6);
    for key in [
        "file_index",
        "byte_offset",
        "games_seen",
        "seed",
        "epoch",
        "shuffle_pos",
    ] {
        let value = raw
            .get_item(key)
            .map_err(|e| PyValueError::new_err(format!("cursor field {key:?} unreadable: {e}")))?;
        let Some(value) = value else {
            return Err(PyValueError::new_err(format!(
                "cursor field {key:?} must be a non-negative int"
            )));
        };
        if value.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(format!(
                "cursor field {key:?} must be a non-negative int, not bool"
            )));
        }
        let number: i64 = value.extract().map_err(|_| {
            PyValueError::new_err(format!("cursor field {key:?} must be a non-negative int"))
        })?;
        if number < 0 {
            return Err(PyValueError::new_err(format!(
                "cursor field {key:?} must be a non-negative int"
            )));
        }
        pairs.push((key.to_string(), number));
    }
    let map = cursor_from_pairs(&pairs).map_err(PyValueError::new_err)?;
    let cursor = hydra_feed::manifest::StreamCursor::from_dict(&map)
        .map_err(|e| PyValueError::new_err(format!("cursor rejected: {e:?}")))?;
    Ok((
        cursor.file_index,
        cursor.byte_offset,
        cursor.games_seen,
        cursor.seed,
        cursor.epoch,
        cursor.shuffle_pos,
    ))
}

fn hash_canonical_section(doc: &[u8], record: &str) -> Result<String, String> {
    let value = hydra_feed::canon::parse_canonical_bytes(doc, record)
        .map_err(|e| format!("state-tree parse rejected: {e:?}"))?;
    hydra_feed::digest::of_canonical(&value).map_err(|e| format!("state-tree digest failed: {e:?}"))
}

/// State-tree section sha: canonical section bytes in, `sha256:<hex>` out
/// (parse + re-canonicalize + hash, all detached). The input MUST already be
/// the canonical `state_tree` form of one payload section
/// (model/optim/sched/training/sampler/rng); the section split stays Python.
#[pyfunction]
fn state_tree_sha(py: Python<'_>, section_json: Vec<u8>) -> PyResult<String> {
    let digest = py
        .detach(|| hash_canonical_section(&section_json, "bridge:state_tree_sha"))
        .map_err(PyValueError::new_err)?;
    bump_snapshots(py, 1);
    Ok(digest)
}

/// State-tree sha diff: parallel old/new canonical section bytes in,
/// per-section match bits out (one `detach` around the whole fan-out).
/// TRIAGE ONLY — a `false` bit names the diverged section for the
/// resume-mismatch report; it never bypasses the fail-closed raise.
/// Length mismatch and section parse rejects are `ValueError`.
#[pyfunction]
fn state_tree_sha_diff(
    py: Python<'_>,
    olds: Vec<Vec<u8>>,
    news: Vec<Vec<u8>>,
) -> PyResult<Vec<bool>> {
    if olds.len() != news.len() {
        return Err(PyValueError::new_err(
            "state_tree_sha_diff needs parallel old/new section lists",
        ));
    }
    let count = olds.len() as u64;
    let bits = py
        .detach(|| -> Result<Vec<bool>, String> {
            let mut out = Vec::with_capacity(olds.len());
            for (idx, (old, new)) in olds.iter().zip(news.iter()).enumerate() {
                let old_digest = hash_canonical_section(old, "bridge:state_tree_sha_diff:old")
                    .map_err(|e| format!("section {idx} old: {e}"))?;
                let new_digest = hash_canonical_section(new, "bridge:state_tree_sha_diff:new")
                    .map_err(|e| format!("section {idx} new: {e}"))?;
                out.push(old_digest == new_digest);
            }
            Ok(out)
        })
        .map_err(PyValueError::new_err)?;
    bump_snapshots(py, count);
    Ok(bits)
}

/// Judge observability: `(snapshots, blobs, restores)` via an
/// interpreter-safe lock.
#[pyfunction]
fn judge_stats(py: Python<'_>) -> PyResult<(u64, u64, u64)> {
    let guard = JUDGE_STATS
        .lock_py_attached(py)
        .map_err(|_| PyOSError::new_err("resume judge stats mutex poisoned"))?;
    Ok((guard.snapshots, guard.blobs, guard.restores))
}

/// Register the `resume` submodule (mirrors `canon_rng::register`): compute
/// detached, wrap attached; single cdylib, no new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "resume")?;
    sub.add_class::<ResumeConfig>()?;
    sub.add_class::<RngSnapshot>()?;
    sub.add_function(wrap_pyfunction!(snapshot_version, &sub)?)?;
    sub.add_function(wrap_pyfunction!(snapshot_min_reader, &sub)?)?;
    sub.add_function(wrap_pyfunction!(check_snapshot_version, &sub)?)?;
    sub.add_function(wrap_pyfunction!(check_buffer_entries, &sub)?)?;
    sub.add_function(wrap_pyfunction!(encode_reservoir, &sub)?)?;
    sub.add_function(wrap_pyfunction!(decode_reservoir, &sub)?)?;
    sub.add_function(wrap_pyfunction!(verify_prefix_blob, &sub)?)?;
    sub.add_function(wrap_pyfunction!(epoch_seed, &sub)?)?;
    sub.add_function(wrap_pyfunction!(snapshot_words, &sub)?)?;
    sub.add_function(wrap_pyfunction!(pack_cursor, &sub)?)?;
    sub.add_function(wrap_pyfunction!(unpack_cursor, &sub)?)?;
    sub.add_function(wrap_pyfunction!(state_tree_sha, &sub)?)?;
    sub.add_function(wrap_pyfunction!(state_tree_sha_diff, &sub)?)?;
    sub.add_function(wrap_pyfunction!(judge_stats, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod resume_tests {
    use super::*;

    const HELLO_WORLD_SHA: &str =
        "sha256:b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";

    fn test_domain() -> Vec<u8> {
        b"resume_v1".to_vec()
    }

    fn test_seed() -> [u8; 32] {
        [0x22u8; 32]
    }

    #[test]
    fn snapshot_version_window_is_dual_read() {
        assert_eq!(RESUME_SNAPSHOT_VERSION, 2);
        assert_eq!(RESUME_SNAPSHOT_VERSION_V1, 1);
        assert!(snapshot_readable(1));
        assert!(snapshot_readable(2));
        assert!(!snapshot_readable(0));
        assert!(!snapshot_readable(3));
        assert!(!snapshot_readable(u32::MAX));
    }

    #[test]
    fn reservoir_v2_round_trip_shapes() {
        let frames: Vec<Vec<u8>> = vec![b"game-one".to_vec(), b"game-two".to_vec()];
        let refs: Vec<&[u8]> = frames.iter().map(|f| &f[..]).collect();
        let (bytes, index) =
            hydra_feed::manifest::encode_reservoir_blob(&refs, RESERVOIR_BLOB_LEVEL)
                .expect("v2 encodes");
        assert_eq!(index.version, 2);
        assert_eq!(index.count, 2);
        assert_eq!(
            index.uncompressed_bytes,
            (b"game-one".len() + b"game-two".len()) as u64
        );
        assert!(index.blob_sha256.starts_with("sha256:"));
        let back = hydra_feed::manifest::decode_reservoir_blob(&bytes).expect("v2 decodes");
        assert_eq!(back, frames);
    }

    #[test]
    fn reservoir_v1_dual_reads_drain_only() {
        // Generation-1 blob via the feed encoder (no new dep: this crate
        // has no `zstd` edge) then patch the version word to v1. Layout is
        // identical across generations except the version, so the patched
        // blob is exactly what the Python oracle wrote; it must decode
        // through the same drain-only reader.
        let frames: Vec<Vec<u8>> = vec![b"v1-game".to_vec()];
        let refs: Vec<&[u8]> = frames.iter().map(|f| &f[..]).collect();
        let (mut v1, _) = hydra_feed::manifest::encode_reservoir_blob(&refs, RESERVOIR_BLOB_LEVEL)
            .expect("v1 base encodes");
        v1[8..12].copy_from_slice(&RESUME_SNAPSHOT_VERSION_V1.to_le_bytes());
        let back = hydra_feed::manifest::decode_reservoir_blob(&v1).expect("v1 dual-reads");
        assert_eq!(back, frames);
    }

    #[test]
    fn reservoir_rejects_bad_magic_version_trailing() {
        let frames: Vec<Vec<u8>> = vec![b"game-one".to_vec()];
        let refs: Vec<&[u8]> = frames.iter().map(|f| &f[..]).collect();
        let (mut bytes, _) =
            hydra_feed::manifest::encode_reservoir_blob(&refs, RESERVOIR_BLOB_LEVEL)
                .expect("encodes");
        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 0xFF;
        assert!(hydra_feed::manifest::decode_reservoir_blob(&bad_magic).is_err());
        let mut bad_version = bytes.clone();
        bad_version[8..12].copy_from_slice(&9u32.to_le_bytes());
        assert!(hydra_feed::manifest::decode_reservoir_blob(&bad_version).is_err());
        bytes.push(0x00);
        assert!(hydra_feed::manifest::decode_reservoir_blob(&bytes).is_err());
        assert!(hydra_feed::manifest::decode_reservoir_blob(b"short").is_err());
    }

    #[test]
    fn prefix_verify_ok_and_tamper_paths() {
        let mut entries = vec![
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
        ];
        entries.sort();
        let blob = entries.join("\n").into_bytes();
        let sha = hydra_feed::digest::sha256_hex(&blob);
        let lines = verify_prefix(&blob, 2, &sha).expect("valid prefix verifies");
        assert_eq!(lines, entries);
        // Count mismatch.
        assert!(verify_prefix(&blob, 3, &sha).is_err());
        // Digest mismatch (tampered blob).
        let mut tampered = blob.clone();
        tampered[0] = b'#';
        assert!(verify_prefix(&tampered, 2, &sha).is_err());
        // Order violation.
        let mut rev = entries.clone();
        rev.reverse();
        let rev_blob = rev.join("\n").into_bytes();
        let rev_sha = hydra_feed::digest::sha256_hex(&rev_blob);
        assert!(verify_prefix(&rev_blob, 2, &rev_sha).is_err());
        // Non-sha entry.
        let bad_blob = b"not-a-digest\n".to_vec();
        let bad_sha = hydra_feed::digest::sha256_hex(&bad_blob);
        assert!(verify_prefix(&bad_blob, 1, &bad_sha).is_err());
        // Malformed record digest.
        assert!(verify_prefix(&blob, 2, "nope").is_err());
    }

    #[test]
    fn rng_snapshot_words_anchor_to_philox_block() {
        // Restoring (block, 0) prefixes philox_block(key, block) exactly
        // (RNG-restore parity shape); pos advance matches word-for-word.
        let domain = test_domain();
        let seed = test_seed();
        let mut padded = [0u8; 14];
        padded[..domain.len()].copy_from_slice(&domain);
        // proof: test domain `b"resume_v1"` len 9, fits `u32`.
        #[allow(clippy::cast_possible_truncation)]
        let domain_len: u32 = domain.len() as u32;
        let key = hydra_feed::rng::StreamKey::new(padded, domain_len, seed);
        let block_words = hydra_feed::rng::philox_block(&key, 7);
        let words = draw_snapshot_words(&domain, &seed, 7, 0, 4).expect("restores");
        assert_eq!(words, block_words.to_vec());
        let tail = draw_snapshot_words(&domain, &seed, 7, 1, 2).expect("pos advances");
        assert_eq!(tail, vec![block_words[1], block_words[2]]);
        // Adjacent blocks differ (counter actually moves).
        let next = draw_snapshot_words(&domain, &seed, 8, 0, 4).expect("next block");
        assert_ne!(words, next);
        // pos == 4 rejected (caller must normalize to (block + 1, 0)).
        assert!(draw_snapshot_words(&domain, &seed, 7, 4, 1).is_err());
        assert!(draw_snapshot_words(&[], &seed, 7, 0, 1).is_err());
        assert!(draw_snapshot_words(&domain, &seed, 7, 0, 0).is_err());
    }

    #[test]
    fn epoch_seed_wraps_like_sidecar() {
        assert_eq!(epoch_seed(100, 5), 105);
        assert_eq!(epoch_seed(0, 0), 0);
        assert_eq!(epoch_seed(u64::MAX, 1), 0);
    }

    #[test]
    fn cursor_dict_round_trip_parity() {
        // Bridge-level parity for the YAML-resume round-trip (Data m9:
        // to_dict/from_dict, not getters-only).
        let cursor = hydra_feed::manifest::StreamCursor {
            file_index: 3,
            byte_offset: 4096,
            games_seen: 128,
            seed: 99,
            epoch: 0,
            shuffle_pos: 64,
        };
        let dict = cursor.to_dict();
        let back = hydra_feed::manifest::StreamCursor::from_dict(&dict).expect("round-trips");
        assert_eq!(cursor, back);
        let mut missing = dict.clone();
        missing.remove("seed");
        assert!(hydra_feed::manifest::StreamCursor::from_dict(&missing).is_err());
        let mut negative = dict.clone();
        negative.insert("epoch".to_string(), -1);
        assert!(hydra_feed::manifest::StreamCursor::from_dict(&negative).is_err());
    }

    #[test]
    fn state_tree_diff_shapes_and_single_bit() {
        // Triage shape: identical sections diff all-true; one changed
        // section flips exactly its bit; length mismatch is Err.
        let a = br#"{"b":2,"a":1}"#.to_vec();
        let b = br#"{"a":1,"b":2}"#.to_vec();
        let c = br#"{"a":1,"b":3}"#.to_vec();
        let da = hash_canonical_section(&a, "t").expect("hashes");
        let db = hash_canonical_section(&b, "t").expect("hashes");
        assert_eq!(da, db, "JCS key order must not move the digest");
        let olds = [a.clone(), a.clone()];
        let news = [b, c];
        let bits: Vec<bool> = olds
            .iter()
            .zip(news.iter())
            .map(|(o, n)| {
                hash_canonical_section(o, "t").unwrap() == hash_canonical_section(n, "t").unwrap()
            })
            .collect();
        assert_eq!(bits, vec![true, false]);
        // Digest path itself is SHA-256 identity (KAT-1 shape for anchors).
        assert_eq!(
            hydra_feed::digest::sha256_hex(b"hello world"),
            HELLO_WORLD_SHA
        );
    }

    #[test]
    fn frozen_surfaces_are_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ResumeConfig>();
        assert_sync::<RngSnapshot>();
    }

    #[test]
    fn detach_smoke() {
        Python::initialize();
        Python::attach(|py| {
            let words = py.detach(|| {
                draw_snapshot_words(&test_domain(), &test_seed(), 0, 0, 4).expect("draws")
            });
            assert_eq!(words.len(), 4);
            let digest = py.detach(|| hydra_feed::digest::sha256_hex(b"hello world"));
            assert_eq!(digest, HELLO_WORLD_SHA);
        });
    }
}
