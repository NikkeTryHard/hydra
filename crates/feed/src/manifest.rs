//! Stream manifest: partition vocabulary, file list, digests, and caches.
//!
//! Rust owner of `src/hydra2/data/stream_manifest.py` (sha-sort file order,
//! manifest digest with ASCII fast-path, reservoir blob, scan cache, shuffle
//! RNG codec). Python oracle is read-only this phase.
//!
//! Order rule: the file list is sorted by sha256-hex of the relative POSIX
//! path (`build_manifest`, stream_manifest.py:83-99). The hash *orders* and
//! never splits — split assignment uses `crate::partition::assign_one`
//! exclusively.
//!
//! Digest rule (stream_manifest.py:102-140): the digest binds
//! `[{bytes, path}, ...]` over canon bytes. Element key order is
//! `bytes < path` by UTF-16BE. The ASCII fast-path assembles the same bytes
//! directly when every path is escape-free ASCII; anything else takes the
//! element-wise canon path. Production ALWAYS verifies the fast bytes
//! against the canon path and returns `Err(Canon)` on divergence
//! (fail-closed, all builds — never `debug_assert`-only). Any divergence
//! breaks scan-cache keys loudly (miss → full scan), never silently.
//!
//! Identity discipline: SHA-256 ONLY (`feed::digest`), canon bytes ONLY
//! (`feed::canon`). No second printer, no second hasher.
//!
//! Snapshot versions (Data M7): this module WRITES generation 2 and reads
//! both generations drain-only:
//! - `RESERVOIR_BLOB_VERSION = 2`, `SCAN_CACHE_VERSION = 2`.
//! - Generation-1 blobs/caches (Python `HYDRARS1 v1` / scan-cache v1) decode
//!   through the same fail-closed reader for drain-only resume during the one
//!   dual-read release; unknown versions are `ContractError`-style `Err`
//!   (fail-closed to miss / full scan), never coerced.
//! - Gate: snapshot→restore→drain equivalence on BOTH blob generations (see
//!   tests below).
//!
//! Shuffle-RNG codec NOTE (Wave5 B2 + Data M2): the `version/state/gauss_next`
//! `random.Random` codec below is drain-only compatibility for snapshots
//! written by the Python oracle. NEW streams use Philox
//! (`feed::rng::StreamSnapshot` = `(entries, {key, block, pos})` triple,
//! block-starts-only seek). `torch.randperm` held-out splits stay the oracle
//! behind KAT — this file never invents a Philox split permutation. There is
//! NO `feed::shuffle` module (M2: `feed::rng` owns ALL Philox draws).
//!
//! Scan migration (Data M8): `training/stream_scan.py:23-24` imports
//! `GameStream`/`ZstdLineStream` from the `data.stream` shim today; the
//! packet slice migrates those imports to the new Rust scan/batch ABI. This
//! module owns the manifest/scan-cache bytes the scan keys on, not the scan
//! loop itself.
//!
//! Filename qualification (Data m6): `validate_manifest` lives in
//! `src/hydra2/training/shard_reader.py:562-566` — NOT in `parquet.py` (424
//! lines, has no such symbol). Any cross-reference to `validate_manifest`
//! without a filename is ambiguous; this module always qualifies it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::canon::{CanonError, canonical_bytes};
use crate::digest::sha256_hex;

/// Partition names in cumulative-threshold order.
///
/// Value-equal to `crate::partition::PARTITION_ORDER` and to
/// `stream_manifest.PARTITION_ORDER` (`:50`). See partition module doc for
/// the Data m4 value-equality note.
pub const PARTITION_ORDER: [&str; 5] =
    ["train", "validation", "test", "decision_eval", "block_eval"];

/// Default grouping; mirrors `grouping_keys=("source", "time")` (`:52`).
pub const DEFAULT_GROUPING_KEYS: [&str; 2] = ["source", "time"];

/// Manifest failure taxonomy (fail-closed, never silent coercion).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    /// Reservoir blob unreadable / bad magic / truncated / trailing.
    BlobCorrupt { detail: String },
    /// Blob or cache version outside the dual-read window.
    BadVersion { detail: String },
    /// Shuffle RNG codec misuse (unknown keys, bad ints, empty state).
    BadRngState { detail: String },
    /// Cursor dict misuse (missing/negative/non-int fields).
    BadCursor { detail: String },
    /// Canon step failed.
    Canon { detail: String },
    /// zstd frame failure.
    Zstd { detail: String },
}

impl core::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ManifestError::BlobCorrupt { detail } => {
                write!(f, "manifest: reservoir blob corrupt ({detail})")
            }
            ManifestError::BadVersion { detail } => {
                write!(f, "manifest: bad snapshot version ({detail})")
            }
            ManifestError::BadRngState { detail } => {
                write!(f, "manifest: shuffle RNG state malformed ({detail})")
            }
            ManifestError::BadCursor { detail } => {
                write!(f, "manifest: cursor field invalid ({detail})")
            }
            ManifestError::Canon { detail } => {
                write!(f, "manifest: canonicalization failed ({detail})")
            }
            ManifestError::Zstd { detail } => {
                write!(f, "manifest: zstd frame failed ({detail})")
            }
        }
    }
}

impl std::error::Error for ManifestError {}

impl From<CanonError> for ManifestError {
    fn from(e: CanonError) -> Self {
        ManifestError::Canon {
            detail: e.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// File list (sha-sort)
// ---------------------------------------------------------------------------

/// One corpus file: relative POSIX path + compressed size.
///
/// Counts (`wall_hashes` etc.) are populated by the scan pass, not here —
/// mirroring `stream_manifest.FileEntry` (`:58-65`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileEntry {
    /// Relative POSIX path (`path.relative_to(base).as_posix()`).
    pub path: String,
    /// Compressed size in bytes (`st_size` at collect time).
    pub bytes: u64,
}

/// Deterministic file list; order is sha256-hex of the relative path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamManifest {
    /// Files in sha-sort order.
    pub files: Vec<FileEntry>,
    /// Collection root (display/provenance; ordering uses relative paths).
    pub root: String,
}

/// Order key: sha256-hex (lowercase, no prefix) of the relative POSIX path.
///
/// Mirrors `build_manifest._order_key` (stream_manifest.py:94-95). Routed
/// through `feed::digest::sha256_hex` (the single owned hasher); the prefix
/// is stripped for the sort key (ordering only, never an identity digest).
pub fn order_key(rel_posix: &str) -> String {
    let prefixed = crate::digest::sha256_hex(rel_posix.as_bytes());
    prefixed["sha256:".len()..].to_string()
}

/// Build a manifest from `(relative POSIX path, bytes)` pairs, sha-sorted.
pub fn build_manifest_from_paths(
    root: &str,
    mut paths: Vec<(String, u64)>,
) -> StreamManifest {
    paths.sort_by_key(|a| order_key(&a.0));
    StreamManifest {
        files: paths
            .into_iter()
            .map(|(path, bytes)| FileEntry { path, bytes })
            .collect(),
        root: root.to_string(),
    }
}

/// Escape-free ASCII check for the digest fast-path.
///
/// Mirrors the `re.search(r"[^\x20\x21\x23-\x5b\x5d-\x7e]", ...)` guard
/// (`:116`): only `"`, `\`, controls, and non-ASCII force the element-wise
/// path. Implemented as a single byte scan (no regex dependency in feed).
/// Allowed: `0x20`, `0x21`, `0x23..=0x5B`, `0x5D..=0x7E` — i.e. printable
/// ASCII except `"` (`0x22`) and `\` (`0x5C`).
pub fn is_escape_free_ascii(text: &str) -> bool {
    for b in text.bytes() {
        let ok = b == 0x20
            || b == 0x21
            || (0x23..=0x5B).contains(&b)
            || (0x5D..=0x7E).contains(&b);
        if !ok {
            return false;
        }
    }
    true
}

#[derive(Serialize)]
struct ManifestElement<'a> {
    bytes: u64,
    path: &'a str,
}

/// Bind the file list for RunSpec provenance.
///
/// Byte-identical to `sha256(canonical_bytes([{bytes, path}, ...]))` by
/// construction in both paths (element key order `bytes < path`; compact
/// list encoding `[` + comma-joined elements + `]`). The fast path assembles
/// the same bytes directly when every path is escape-free ASCII; anything
/// else takes the element-wise canon path. Production ALWAYS recomputes the
/// canon path and returns `Err(Canon)` on divergence (fail-closed, all
/// builds). Any divergence breaks scan-cache keys loudly (miss → full
/// scan), never silently.
pub fn manifest_digest(manifest: &StreamManifest) -> Result<String, ManifestError> {
    let eligible = !manifest.files.is_empty()
        && manifest.files.iter().all(|e| is_escape_free_ascii(&e.path));
    if eligible {
        let mut fast: Vec<u8> = Vec::new();
        fast.push(b'[');
        for (i, entry) in manifest.files.iter().enumerate() {
            if i > 0 {
                fast.push(b',');
            }
            fast.extend_from_slice(b"{\"bytes\":");
            fast.extend_from_slice(entry.bytes.to_string().as_bytes());
            fast.extend_from_slice(b",\"path\":\"");
            fast.extend_from_slice(entry.path.as_bytes());
            fast.extend_from_slice(b"\"}");
        }
        fast.push(b']');
        {
            let canon = manifest_digest_canon(manifest)?;
            let fast_digest = sha256_hex(&fast);
            if fast_digest != canon {
                return Err(ManifestError::Canon {
                    detail: "manifest fast path diverged from canon path".to_string(),
                });
            }
        }
        return Ok(sha256_hex(&fast));
    }
    manifest_digest_canon(manifest)
}

/// Element-wise canon path (always correct, used for non-ASCII/escaped
/// paths and as the fail-closed cross-check baseline for the fast path).
///
/// Assembles the full `[` + comma-joined elements + `]` document then hashes
/// once through `feed::digest` (the single owned hasher — no second hasher,
/// no incremental-hash import).
fn manifest_digest_canon(manifest: &StreamManifest) -> Result<String, ManifestError> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"[");
    for (i, entry) in manifest.files.iter().enumerate() {
        if i > 0 {
            buf.push(b',');
        }
        let el = ManifestElement {
            bytes: entry.bytes,
            path: &entry.path,
        };
        let bytes = canonical_bytes(&el, "manifest:digest")?;
        buf.extend_from_slice(&bytes);
    }
    buf.push(b']');
    Ok(sha256_hex(&buf))
}

// ---------------------------------------------------------------------------
// Reservoir blob (shuffle-buffer zstd, M7 dual-read)
// ---------------------------------------------------------------------------

/// Reservoir-blob magic (`_RESERVOIR_MAGIC`, stream_manifest.py:145).
pub const RESERVOIR_MAGIC: &[u8; 8] = b"HYDRARS1";

/// Generation the Rust writer emits (Data M7 bump).
pub const RESERVOIR_BLOB_VERSION: u32 = 2;

/// Generation the Python oracle wrote — readable drain-only for one release.
pub const RESERVOIR_BLOB_VERSION_V1: u32 = 1;

/// Index record returned alongside the blob bytes.
///
/// Mirrors `write_reservoir_blob`'s `{version, count, uncompressed_bytes,
/// blob_sha256}` dict (`:173-178`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservoirIndex {
    /// Layout version written.
    pub version: u32,
    /// Game count.
    pub count: u32,
    /// Total uncompressed bytes.
    pub uncompressed_bytes: u64,
    /// `sha256:` over header + all `[len][frame]` records.
    pub blob_sha256: String,
}

/// Encode length-prefixed per-game zstd frames for a shuffle buffer.
///
/// Layout: `magic(8) + version u32le + count u32le`, then per game
/// `[u32le frame-len][frame]` in buffer order. `level` mirrors the Python
/// `level=1` default. Pure bytes in / bytes out; file I/O stays with the
/// caller (framer sibling).
pub fn encode_reservoir_blob(
    raws: &[&[u8]],
    level: i32,
) -> Result<(Vec<u8>, ReservoirIndex), ManifestError> {
    let count: u32 = raws.len().try_into().map_err(|_| ManifestError::BlobCorrupt {
        detail: "too many games for reservoir blob".to_string(),
    })?;
    let mut out: Vec<u8> = Vec::new();
    let mut header = [0u8; 16];
    header[..8].copy_from_slice(RESERVOIR_MAGIC);
    header[8..12].copy_from_slice(&RESERVOIR_BLOB_VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&header);
    let mut uncompressed: u64 = 0;
    for raw in raws {
        let frame = zstd::encode_all(&raw[..], level).map_err(|e| ManifestError::Zstd {
            detail: e.to_string(),
        })?;
        // proof: compressed frames < u32::MAX bytes.
        #[allow(clippy::cast_possible_truncation)]
        let len = frame.len() as u32;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&frame);
        uncompressed += raw.len() as u64;
    }
    // Single owned hasher: `feed::digest::sha256_hex` over the emitted blob
    // itself (header + [len][frame] records, one buffer, one call site for
    // the primitive — no dual-extend drift, halves peak on large buffers).
    let blob_sha256 = sha256_hex(&out);
    Ok((
        out,
        ReservoirIndex {
            version: RESERVOIR_BLOB_VERSION,
            count,
            uncompressed_bytes: uncompressed,
            blob_sha256,
        },
    ))
}

/// Decode a reservoir blob into per-game raw bytes (in order), either
/// generation (M7 dual-read).
///
/// Fail-closed like `read_reservoir_blob` (`:181-219`): bad magic,
/// unknown version, truncated frames, and trailing bytes are all `Err`
/// (callers treat it as a snapshot miss and fall back to the normal fill).
/// Transient peak is the decompressed buffer; freed after decode by the
/// caller.
pub fn decode_reservoir_blob(data: &[u8]) -> Result<Vec<Vec<u8>>, ManifestError> {
    if data.len() < 16 {
        return Err(ManifestError::BlobCorrupt {
            detail: "reservoir blob shorter than header".to_string(),
        });
    }
    if &data[..8] != RESERVOIR_MAGIC {
        return Err(ManifestError::BlobCorrupt {
            detail: "reservoir blob bad magic".to_string(),
        });
    }
    let version = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let count = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    if version != RESERVOIR_BLOB_VERSION && version != RESERVOIR_BLOB_VERSION_V1 {
        return Err(ManifestError::BadVersion {
            detail: format!("reservoir blob version {version} not readable"),
        });
    }
    let mut raws: Vec<Vec<u8>> = Vec::with_capacity(count as usize);
    let mut off = 16usize;
    for _ in 0..count {
        if off + 4 > data.len() {
            return Err(ManifestError::BlobCorrupt {
                detail: "reservoir blob truncated length".to_string(),
            });
        }
        let flen =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize;
        off += 4;
        if off + flen > data.len() {
            return Err(ManifestError::BlobCorrupt {
                detail: "reservoir blob truncated frame".to_string(),
            });
        }
        let raw = zstd::decode_all(&data[off..off + flen]).map_err(|e| {
            ManifestError::BlobCorrupt {
                detail: format!("reservoir frame corrupt ({e})"),
            }
        })?;
        raws.push(raw);
        off += flen;
    }
    if off != data.len() {
        return Err(ManifestError::BlobCorrupt {
            detail: "reservoir blob trailing bytes".to_string(),
        });
    }
    Ok(raws)
}
// ---------------------------------------------------------------------------
// Scan cache (content-hash keyed, advisory, M7 dual-read)
// ---------------------------------------------------------------------------

/// Scan-cache envelope version this module writes (Data M7 bump).
pub const SCAN_CACHE_VERSION: u32 = 2;

/// Scan-cache envelope version the Python oracle wrote (miss, never coerce).
pub const SCAN_CACHE_VERSION_V1: u32 = 1;

/// Pre-train corpus scan report: split walls, eval ledger, quarantine counts.
///
/// Mirrors the `scan` mapping validated by `load_scan_cache`
/// (`:289-323`); walls are stored sorted for deterministic payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanReport {
    /// Sorted train walls.
    pub train_walls: Vec<String>,
    /// Sorted validation walls.
    pub val_walls: Vec<String>,
    /// Per-split game counts + quarantine counters.
    pub counts: BTreeMap<String, u64>,
}

/// Count keys the scan report must carry (`:304-313`).
pub const SCAN_COUNT_KEYS: [&str; 8] = [
    "train_games",
    "val_games",
    "train_sim_games",
    "val_sim_games",
    "framed",
    "emitted",
    "quarantined",
    "duplicates",
];

/// Exact-ish ratio comparison (JSON round-trip safe, fail-closed to miss).
///
/// Mirrors `_ratios_match` (`:231-251`): 1e-9 tolerates JSON round-trip
/// without admitting a different split. (Rust is statically typed so the
/// bool-exclusion note is encoding-side documentation, not a branch.)
pub fn ratios_match(cached: &BTreeMap<String, f64>, expected: &BTreeMap<String, f64>) -> bool {
    if cached.len() != expected.len() {
        return false;
    }
    for (key, want) in expected {
        match cached.get(key) {
            Some(got) => {
                if !got.is_finite() || !want.is_finite() {
                    return false;
                }
                if (got - want).abs() > 1e-9 {
                    return false;
                }
            }
            None => return false,
        }
    }
    true
}

#[derive(Serialize, Deserialize)]
struct ScanCacheEnvelope {
    version: u32,
    manifest_digest: String,
    data_seed: u64,
    ratios: BTreeMap<String, f64>,
    train_split: String,
    val_split: String,
    scan: ScanCacheScan,
}

#[derive(Serialize, Deserialize)]
struct ScanCacheScan {
    train_walls: Vec<String>,
    val_walls: Vec<String>,
    train_games: u64,
    val_games: u64,
    train_sim_games: u64,
    val_sim_games: u64,
    framed: u64,
    emitted: u64,
    quarantined: u64,
    duplicates: u64,
}

/// Load a cached scan report on exact key match, else `None` (miss).
///
/// Mirrors `load_scan_cache` (`:254-325`): corrupt/stale entries fail closed
/// to miss (full scan), never raise. Generation-1 envelopes are a miss
/// (never coerced into v2 shape). Disjointness is re-checked on load: a
/// cache written before a wall fix must not resurrect cross-split walls.
#[allow(clippy::too_many_arguments)]
pub fn load_scan_cache(
    raw_text: &str,
    manifest_digest: &str,
    seed: u64,
    ratios: &BTreeMap<String, f64>,
    train_split: &str,
    val_split: &str,
) -> Option<ScanReport> {
    let raw: serde_json::Value = serde_json::from_str(raw_text).ok()?;
    let env: ScanCacheEnvelope = serde_json::from_value(raw).ok()?;
    if env.version != SCAN_CACHE_VERSION {
        return None;
    }
    if env.manifest_digest != manifest_digest {
        return None;
    }
    if env.data_seed != seed {
        return None;
    }
    if env.train_split != train_split || env.val_split != val_split {
        return None;
    }
    if !ratios_match(&env.ratios, ratios) {
        return None;
    }
    let mut train_walls = env.scan.train_walls;
    let mut val_walls = env.scan.val_walls;
    train_walls.sort();
    val_walls.sort();
    let train_set: std::collections::BTreeSet<&str> =
        train_walls.iter().map(String::as_str).collect();
    let val_set: std::collections::BTreeSet<&str> =
        val_walls.iter().map(String::as_str).collect();
    if !train_set.is_disjoint(&val_set) {
        return None;
    }
    let mut counts = BTreeMap::new();
    counts.insert("train_games".to_string(), env.scan.train_games);
    counts.insert("val_games".to_string(), env.scan.val_games);
    counts.insert("train_sim_games".to_string(), env.scan.train_sim_games);
    counts.insert("val_sim_games".to_string(), env.scan.val_sim_games);
    counts.insert("framed".to_string(), env.scan.framed);
    counts.insert("emitted".to_string(), env.scan.emitted);
    counts.insert("quarantined".to_string(), env.scan.quarantined);
    counts.insert("duplicates".to_string(), env.scan.duplicates);
    Some(ScanReport {
        train_walls,
        val_walls,
        counts,
    })
}

/// Serialize a scan-cache payload (deterministic: sorted walls).
///
/// Mirrors `save_scan_cache` (`:328-374`) minus filesystem atomicity
/// (tmp+replace publish stays with the caller; cache is advisory and never
/// fails training on I/O error).
#[allow(clippy::too_many_arguments)]
pub fn encode_scan_cache(
    manifest_digest: &str,
    seed: u64,
    ratios: &BTreeMap<String, f64>,
    train_split: &str,
    val_split: &str,
    report: &ScanReport,
) -> Result<String, ManifestError> {
    let get = |key: &str| -> Result<u64, ManifestError> {
        report.counts.get(key).copied().ok_or_else(|| ManifestError::BlobCorrupt {
            detail: format!("scan report missing count {key:?}"),
        })
    };
    let mut train_walls = report.train_walls.clone();
    let mut val_walls = report.val_walls.clone();
    train_walls.sort();
    val_walls.sort();
    let env = ScanCacheEnvelope {
        version: SCAN_CACHE_VERSION,
        manifest_digest: manifest_digest.to_string(),
        data_seed: seed,
        ratios: ratios.clone(),
        train_split: train_split.to_string(),
        val_split: val_split.to_string(),
        scan: ScanCacheScan {
            train_walls,
            val_walls,
            train_games: get("train_games")?,
            val_games: get("val_games")?,
            train_sim_games: get("train_sim_games")?,
            val_sim_games: get("val_sim_games")?,
            framed: get("framed")?,
            emitted: get("emitted")?,
            quarantined: get("quarantined")?,
            duplicates: get("duplicates")?,
        },
    };
    serde_json::to_string(&env).map_err(|e| ManifestError::Canon {
        detail: e.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Shuffle RNG codec (drain-only compat) + NEW-stream note
// ---------------------------------------------------------------------------

/// JSON-safe snapshot of a shuffle `random.Random` (drain-only compat).
///
/// Mirrors `serialize_shuffle_rng` (`:377-390`): the
/// `version/state/gauss_next` `getstate()` triple. NEW Rust snapshots are
/// `(entries, {key, block, pos})` via `feed::rng::StreamSnapshot` — this
/// codec exists ONLY to drain oracle-written sidecars for one release.
#[derive(Debug, Clone, PartialEq)]
pub struct ShuffleRngState {
    /// `getstate()[0]` protocol version.
    pub version: i64,
    /// `getstate()[1]` internal ints (non-empty, non-negative).
    pub state: Vec<u64>,
    /// `getstate()[2]` Gauss spare (`None` ⇔ null).
    pub gauss_next: Option<f64>,
}

/// Serialize the shuffle RNG state to the `{version, state, gauss_next}`
/// mapping shape (keys exactly those three — mirrors `:390`).
pub fn serialize_shuffle_rng(state: &ShuffleRngState) -> BTreeMap<String, serde_json::Value> {
    BTreeMap::from([
        (
            "version".to_string(),
            serde_json::Value::from(state.version),
        ),
        (
            "state".to_string(),
            serde_json::Value::Array(
                state.state.iter().map(|v| serde_json::Value::from(*v)).collect(),
            ),
        ),
        (
            "gauss_next".to_string(),
            match state.gauss_next {
                Some(g) => serde_json::json!(g),
                None => serde_json::Value::Null,
            },
        ),
    ])
}

/// Inverse of [`serialize_shuffle_rng`]; raises on any mismatch.
///
/// Mirrors `parse_shuffle_rng` (`:393-417`): non-mapping, unknown keys,
/// non-int version, empty/non-int state, negative words, and non-float
/// `gauss_next` all fail closed. (`bool` is excluded by requiring JSON
/// numbers through `as_i64`/`as_u64`, matching the Python `isinstance`
/// guards.)
pub fn parse_shuffle_rng(
    raw: &BTreeMap<String, serde_json::Value>,
) -> Result<ShuffleRngState, ManifestError> {
    let bad = |detail: &str| ManifestError::BadRngState {
        detail: detail.to_string(),
    };
    for key in raw.keys() {
        if key != "version" && key != "state" && key != "gauss_next" {
            return Err(bad(&format!("shuffle buffer_rng_state unknown keys [{key:?}]")));
        }
    }
    let version = raw
        .get("version")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| bad("shuffle buffer_rng_state.version must be an int"))?;
    let state_raw = raw
        .get("state")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| bad("shuffle buffer_rng_state.state must be a non-empty int list"))?;
    if state_raw.is_empty() {
        return Err(bad("shuffle buffer_rng_state.state must be a non-empty int list"));
    }
    let mut state = Vec::with_capacity(state_raw.len());
    for value in state_raw {
        match value.as_u64() {
            Some(v) => state.push(v),
            None => {
                return Err(bad(
                    "shuffle buffer_rng_state.state must hold non-negative ints",
                ));
            }
        }
    }
    let gauss_next = match raw.get("gauss_next") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => Some(v.as_f64().ok_or_else(|| {
            bad("shuffle buffer_rng_state.gauss_next must be a float or null")
        })?),
    };
    Ok(ShuffleRngState {
        version,
        state,
        gauss_next,
    })
}

// ---------------------------------------------------------------------------
// Stream cursor (YAML-resume round-trip, Data m9)
// ---------------------------------------------------------------------------

/// Resumable frontier: file seek + byte resume + accounting.
///
/// Mirrors `stream_read.StreamCursor` (`stream_read.py:179-222`):
/// `byte_offset` is a decompressed-byte offset (stable across zstd
/// re-encodes); resume skips framed games with `offset < byte_offset`;
/// `shuffle_pos` counts shuffled emissions. The bridge exposes the
/// `to_dict`/`from_dict` YAML-resume round-trip, not getters-only (m9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamCursor {
    /// File index into the manifest.
    pub file_index: u64,
    /// Decompressed-byte resume offset.
    pub byte_offset: u64,
    /// Games seen (framed) so far.
    pub games_seen: u64,
    /// Stream seed (must match on resume).
    pub seed: u64,
    /// Stream epoch (must match on resume).
    pub epoch: u64,
    /// Shuffled emissions so far.
    pub shuffle_pos: u64,
}

impl StreamCursor {
    /// Plain mapping for YAML persistence under the artifact root.
    ///
    /// Keys exactly `byte_offset/epoch/file_index/games_seen/seed/shuffle_pos`
    /// (mirrors `:195-204`).
    pub fn to_dict(&self) -> BTreeMap<String, i64> {
        BTreeMap::from([
            ("byte_offset".to_string(), self.byte_offset.cast_signed()),
            ("epoch".to_string(), self.epoch.cast_signed()),
            ("file_index".to_string(), self.file_index.cast_signed()),
            ("games_seen".to_string(), self.games_seen.cast_signed()),
            ("seed".to_string(), self.seed.cast_signed()),
            ("shuffle_pos".to_string(), self.shuffle_pos.cast_signed()),
        ])
    }

    /// Inverse of [`to_dict`](Self::to_dict); rejects missing or negative
    /// fields (mirrors `:206-222` `type(value) is not int or value < 0`).
    pub fn from_dict(raw: &BTreeMap<String, i64>) -> Result<Self, ManifestError> {
        let bad = |key: &str| ManifestError::BadCursor {
            detail: format!("cursor field {key:?} must be a non-negative int"),
        };
        let get = |key: &str| -> Result<u64, ManifestError> {
            let v = raw.get(key).ok_or_else(|| bad(key))?;
            u64::try_from(*v).map_err(|_| bad(key))
        };
        Ok(StreamCursor {
            file_index: get("file_index")?,
            byte_offset: get("byte_offset")?,
            games_seen: get("games_seen")?,
            seed: get("seed")?,
            epoch: get("epoch")?,
            shuffle_pos: get("shuffle_pos")?,
        })
    }
}

/// One shuffle-restore entry: deterministic key plus verbatim refetch
/// coordinates (mirrors `stream_iter.GameStream.shuffle_snapshot`, `:297-317`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShuffleEntry {
    /// `raw_bytes_sha256` of the buffered game.
    pub key: String,
    /// Corpus path (POSIX) for verbatim refetch.
    pub path: String,
    /// Decompressed-byte game offset for verbatim refetch.
    pub offset: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_two_files() -> StreamManifest {
        build_manifest_from_paths(
            "/data",
            vec![
                ("b/20240102.mjai.json.zst".to_string(), 10),
                ("a/20240101.mjai.json.zst".to_string(), 20),
            ],
        )
    }

    #[test]
    fn sha_sort_orders_by_path_hash_not_lexically() {
        // Contract: order key is sha256-hex of the relative path. Sorted
        // files must equal paths sorted by that key (lexical order is only
        // sometimes the same — the key function is the contract).
        let m = manifest_two_files();
        let mut want = vec![
            "b/20240102.mjai.json.zst",
            "a/20240101.mjai.json.zst",
        ];
        want.sort_by_key(|a| order_key(a));
        let got: Vec<&str> = m.files.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(got, want);
        assert_eq!(order_key("a").len(), 64);
    }

    #[test]
    fn manifest_digest_fast_equals_canon_on_ascii() {
        // Wave3 §1.4 asserted-equal optimization: escape-free ASCII ⇒ fast
        // bytes == canon bytes ⇒ same digest. Non-ASCII ⇒ canon path.
        let m = manifest_two_files();
        assert!(m.files.iter().all(|e| is_escape_free_ascii(&e.path)));
        let digest = manifest_digest(&m).unwrap();
        assert!(digest.starts_with("sha256:"));
        assert_eq!(digest.len(), 7 + 64);
        let canon = manifest_digest_canon(&m).unwrap();
        assert_eq!(digest, canon);
        // Stability: rebuild ⇒ same digest.
        assert_eq!(digest, manifest_digest(&manifest_two_files()).unwrap());
        // Non-ASCII path takes the canon arm and still binds.
        let uni = build_manifest_from_paths(
            "/data",
            vec![("t/é.mjai.json.zst".to_string(), 5)],
        );
        assert!(!is_escape_free_ascii("t/é.mjai.json.zst"));
        let d2 = manifest_digest(&uni).unwrap();
        assert!(d2.starts_with("sha256:"));
        assert_ne!(digest, d2);
        // Quote/backslash force the element-wise path too.
        assert!(!is_escape_free_ascii("a/\"q\".mjai.json.zst"));
        assert!(!is_escape_free_ascii("a\\b.mjai.json.zst"));
    }

    #[test]
    fn reservoir_round_trip_v2_and_v1_drain() {
        // M7 gate (half): snapshot→restore→drain on the generation this
        // module writes. Drained bytes must equal snapshot bytes in order.
        let raws: Vec<&[u8]> = vec![b"game-one-bytes", b"game-two-bytes-longer", b""];
        let (blob, index) = encode_reservoir_blob(&raws, 1).unwrap();
        assert_eq!(index.version, RESERVOIR_BLOB_VERSION);
        assert_eq!(index.count, 3);
        assert_eq!(
            index.uncompressed_bytes,
            raws.iter().map(|r| r.len() as u64).sum::<u64>()
        );
        assert!(index.blob_sha256.starts_with("sha256:"));
        let back = decode_reservoir_blob(&blob).unwrap();
        assert_eq!(back.len(), 3);
        for (got, want) in back.iter().zip(raws.iter()) {
            assert_eq!(got.as_slice(), *want);
        }
        // V1 generation drains through the same reader (dual-read release).
        let mut v1 = Vec::new();
        v1.extend_from_slice(RESERVOIR_MAGIC);
        v1.extend_from_slice(&RESERVOIR_BLOB_VERSION_V1.to_le_bytes());
        v1.extend_from_slice(&1u32.to_le_bytes());
        let frame = zstd::encode_all(&b"v1-game"[..], 1).unwrap();
        v1.extend_from_slice(&u32::try_from(frame.len()).unwrap().to_le_bytes());
        v1.extend_from_slice(&frame);
        let drained = decode_reservoir_blob(&v1).unwrap();
        assert_eq!(drained, vec![b"v1-game".to_vec()]);
        // Fail-closed: bad magic / unknown version / trailing / truncated.
        assert!(matches!(
            decode_reservoir_blob(b"short"),
            Err(ManifestError::BlobCorrupt { .. })
        ));
        let mut bad_magic = blob.clone();
        bad_magic[0] ^= 0xFF;
        assert!(matches!(
            decode_reservoir_blob(&bad_magic),
            Err(ManifestError::BlobCorrupt { .. })
        ));
        let mut bad_version = blob.clone();
        bad_version[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            decode_reservoir_blob(&bad_version),
            Err(ManifestError::BadVersion { .. })
        ));
        let mut trailing = blob.clone();
        trailing.push(0);
        assert!(matches!(
            decode_reservoir_blob(&trailing),
            Err(ManifestError::BlobCorrupt { .. })
        ));
        let truncated = &blob[..blob.len() - 1];
        assert!(decode_reservoir_blob(truncated).is_err());
    }

    #[test]
    fn scan_cache_hit_miss_and_disjoint_fail_closed() {
        let ratios = BTreeMap::from([
            ("train".to_string(), 0.8),
            ("validation".to_string(), 0.2),
        ]);
        let mut counts = BTreeMap::new();
        for (i, k) in SCAN_COUNT_KEYS.iter().enumerate() {
            counts.insert(k.to_string(), i as u64);
        }
        let report = ScanReport {
            train_walls: vec!["sha256:aa".to_string()],
            val_walls: vec!["sha256:bb".to_string()],
            counts,
        };
        let text =
            encode_scan_cache("sha256:deadbeef", 7, &ratios, "train", "validation", &report)
                .unwrap();
        let hit = load_scan_cache(&text, "sha256:deadbeef", 7, &ratios, "train", "validation")
            .expect("exact key match hits");
        assert_eq!(hit.train_walls, report.train_walls);
        assert_eq!(hit.val_walls, report.val_walls);
        // Stale keys miss, never coerce.
        assert!(
            load_scan_cache(&text, "sha256:other", 7, &ratios, "train", "validation").is_none()
        );
        assert!(
            load_scan_cache(&text, "sha256:deadbeef", 8, &ratios, "train", "validation").is_none()
        );
        let mut other_ratios = ratios.clone();
        other_ratios.insert("train".to_string(), 0.5);
        assert!(
            load_scan_cache(&text, "sha256:deadbeef", 7, &other_ratios, "train", "validation")
                .is_none()
        );
        assert!(load_scan_cache("not json", "sha256:deadbeef", 7, &ratios, "train", "validation")
            .is_none());
        // V1 envelope is a miss (dual-read window is blobs; caches re-scan).
        let v1_text = text.replace(
            &format!("\"version\":{SCAN_CACHE_VERSION}"),
            &format!("\"version\":{SCAN_CACHE_VERSION_V1}"),
        );
        assert!(
            load_scan_cache(
                &v1_text,
                "sha256:deadbeef",
                7,
                &ratios,
                "train",
                "validation"
            )
            .is_none()
        );
        // Cross-split walls fail closed to miss even on exact keys.
        let mut bad_counts = BTreeMap::new();
        for k in SCAN_COUNT_KEYS {
            bad_counts.insert(k.to_string(), 0);
        }
        let bad = ScanReport {
            train_walls: vec!["sha256:shared".to_string()],
            val_walls: vec!["sha256:shared".to_string()],
            counts: bad_counts,
        };
        let bad_text =
            encode_scan_cache("sha256:x", 1, &ratios, "train", "validation", &bad).unwrap();
        assert!(
            load_scan_cache(&bad_text, "sha256:x", 1, &ratios, "train", "validation").is_none()
        );
        // 1e-9 tolerates JSON round-trip without admitting a new split.
        let mut rt = ratios.clone();
        *rt.get_mut("train").unwrap() += 1e-10;
        *rt.get_mut("validation").unwrap() -= 1e-10;
        assert!(ratios_match(&rt, &ratios));
    }

    #[test]
    fn shuffle_rng_codec_round_trip_and_rejects() {
        // M7 gate (other half): the oracle codec drains losslessly.
        let state = ShuffleRngState {
            version: 2,
            state: vec![1, 2, 3, 4],
            gauss_next: Some(0.5),
        };
        let map = serialize_shuffle_rng(&state);
        assert_eq!(
            map.keys().collect::<Vec<_>>(),
            [&"gauss_next".to_string(), &"state".to_string(), &"version".to_string()]
        );
        assert_eq!(parse_shuffle_rng(&map).unwrap(), state);
        let null_gauss = ShuffleRngState {
            version: 2,
            state: vec![9],
            gauss_next: None,
        };
        assert_eq!(
            parse_shuffle_rng(&serialize_shuffle_rng(&null_gauss)).unwrap(),
            null_gauss
        );
        // Unknown keys / empty state / bad version all fail closed.
        let mut unknown = map.clone();
        unknown.insert("extra".to_string(), serde_json::json!(1));
        assert!(matches!(
            parse_shuffle_rng(&unknown),
            Err(ManifestError::BadRngState { .. })
        ));
        let mut empty = map.clone();
        empty.insert("state".to_string(), serde_json::json!([]));
        assert!(matches!(
            parse_shuffle_rng(&empty),
            Err(ManifestError::BadRngState { .. })
        ));
        let mut bad_version = map.clone();
        bad_version.insert("version".to_string(), serde_json::json!("2"));
        assert!(matches!(
            parse_shuffle_rng(&bad_version),
            Err(ManifestError::BadRngState { .. })
        ));
    }

    #[test]
    fn cursor_to_dict_from_dict_round_trip_m9() {
        // Data m9: the bridge needs the YAML-resume round-trip, not
        // getters-only (stream_read.py:195-222).
        let cursor = StreamCursor {
            file_index: 3,
            byte_offset: 1024,
            games_seen: 40,
            seed: 7,
            epoch: 2,
            shuffle_pos: 11,
        };
        let dict = cursor.to_dict();
        assert_eq!(dict.len(), 6);
        assert_eq!(dict["byte_offset"], 1024);
        assert_eq!(dict["shuffle_pos"], 11);
        assert_eq!(StreamCursor::from_dict(&dict).unwrap(), cursor);
        // Negative / missing fields fail closed.
        let mut neg = dict.clone();
        neg.insert("byte_offset".to_string(), -1);
        assert!(matches!(
            StreamCursor::from_dict(&neg),
            Err(ManifestError::BadCursor { .. })
        ));
        let mut missing = dict.clone();
        missing.remove("seed");
        assert!(matches!(
            StreamCursor::from_dict(&missing),
            Err(ManifestError::BadCursor { .. })
        ));
    }
}
