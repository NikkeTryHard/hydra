//! Thin bridge: `FeedStream` + `PyHydraStream` over `feed::fill` (serial
//! staging in file order; stats and quarantines stay readable post-close).
//!
//! DAG: this crate depends on the feed crate + pyo3 ONLY (no cold-crate edge;
//! CI triple-grep enforces). FORBIDS: parse/encode/hash/pool logic —
//! arg-check + detach + struct return only. The SOLE hot pool lives in
//! `feed::fill`; the bridge creates none and stages serially in file order.
//!
//! Staging contract (breaks the `t_len`/`rows` circle: the stager picks T
//! BEFORE committing rows):
//! each `next_into` call stages whole games (`stage_one`: frame → gate →
//! walk on a thread-local, rollback-free) into the `Scratch` until
//! `batch_rows` staged rows or input exhaustion, then commits via
//! `fill_pinned`, which drains exactly the committed game-aligned prefix
//! and leaves the remainder staged for the next call. A short fill with
//! staged remainder is normal, NOT end-of-stream: exhaustion is
//! `rows == 0 && games_consumed == 0`.
//!
//! The scratch holds ok-games ONLY (rejects are sidecar quarantines), so
//! the drained prefix is all-ok: `games_ok` advances by
//! `FillOut::games_consumed` exactly once per commit.
//!
//! Failure atomicity: `BufferTooSmall` (or any fill error) rolls the WHOLE
//! call back — cursor, scratch rows, quarantine pushes, quarantine counter —
//! pointer fails closed before any staging work.

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use hydra_feed::fill::{FillError, N_PLANES, Scratch};
use hydra_feed::ledger;
use pyo3::exceptions::{PyBufferError, PyOSError, PyValueError};
use pyo3::prelude::*;

// N_PLANES is 26 (canonical plane order #0-25); the FFI surface spells it
// literally so a future plane-count change fails loudly here instead of
// silently widening.
const _: () = assert!(N_PLANES == 26);

// ---------------------------------------------------------------------------
// Closed failure vocabulary (kept 1:1 with the root handoff it replaces,
// plus the plane-carrying buffer error).
// ---------------------------------------------------------------------------

/// Closed stream failure vocabulary (every variant maps to a Python error).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    /// Bad open arguments / unreadable inputs (never a partial stream).
    Open(String),
    /// Action table missing or unverifiable.
    Table(String),
    /// Provenance authority missing or unverifiable (rules manifest).
    Provenance(String),
    /// File read failure mid-stream (fail closed per call).
    Io(String),
    /// A row that is not exactly the hot 13-plane projection (schema fork).
    Envelope(String),
    /// Null caller pointer: nothing staged, nothing drained.
    NullPointer { plane: u8 },
    /// Caller buffer too small: nothing drained, retry with a bigger buffer.
    /// Byte units matching `byte_caps`; the first-short plane is reported.
    BufferTooSmall {
        plane: u8,
        needed: usize,
        capacity: usize,
    },
    /// Use after `close`.
    Closed,
    /// Mutex poison (a prior panic while holding the guard).
    Poisoned,
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Open(msg) => write!(f, "hydra2 replay open: {msg}"),
            StreamError::Table(msg) => write!(f, "hydra2 replay action table: {msg}"),
            StreamError::Provenance(msg) => write!(f, "hydra2 replay provenance: {msg}"),
            StreamError::Io(msg) => write!(f, "hydra2 replay io: {msg}"),
            StreamError::Envelope(msg) => write!(f, "hydra2 replay envelope: {msg}"),
            StreamError::NullPointer { plane } => {
                write!(
                    f,
                    "hydra2 replay next_into got a null buffer pointer for plane {plane}"
                )
            }
            StreamError::BufferTooSmall {
                plane,
                needed,
                capacity,
            } => write!(
                f,
                "hydra2 replay plane {plane} needs {needed} bytes but the caller buffer holds {capacity}"
            ),
            StreamError::Closed => write!(f, "hydra2 replay stream is closed"),
            StreamError::Poisoned => write!(f, "hydra2 replay stream mutex poisoned"),
        }
    }
}

impl std::error::Error for StreamError {}

/// Map a closed failure to its Python error.
///
/// `BufferTooSmall` is the ONLY `PyBufferError` (callers branch on it to
/// grow the slot); everything else is `ValueError`/`OSError` per the
/// long-standing handoff mapping.
pub fn stream_py_err(err: StreamError) -> PyErr {
    match err {
        StreamError::Open(msg)
        | StreamError::Table(msg)
        | StreamError::Provenance(msg)
        | StreamError::Envelope(msg) => PyValueError::new_err(msg),
        StreamError::NullPointer { .. } | StreamError::Closed => {
            PyValueError::new_err(err.to_string())
        }
        StreamError::BufferTooSmall { .. } => PyBufferError::new_err(err.to_string()),
        StreamError::Io(_) | StreamError::Poisoned => PyOSError::new_err(err.to_string()),
    }
}

fn fill_py_err(err: FillError) -> StreamError {
    match err {
        FillError::NullPointer { plane } => StreamError::NullPointer { plane },
        FillError::BufferTooSmall {
            plane,
            needed_bytes,
            capacity_bytes,
        } => StreamError::BufferTooSmall {
            plane,
            needed: needed_bytes,
            capacity: capacity_bytes,
        },
    }
}

/// Cold renderer for a quarantine stub reason (never hot): canonical in
/// [`crate::sink::quarantine_reason_name`] (`reason < 10` → gate names, else
/// walk names; unknown bytes stay `"other"` on both sides).
fn quarantine_reason_name(reason: u8) -> &'static str {
    crate::sink::quarantine_reason_name(reason)
}

// ---------------------------------------------------------------------------
// Pure-Rust serial feed stream: sorted discovery, whole-game staging,
// quarantine capture (never a new reason code: feed verdicts only).
// ---------------------------------------------------------------------------

/// Open parameters (mirrors the PyO3 `open` signature 1:1).
#[derive(Debug, Clone)]
pub struct FeedOpen {
    /// Framed per-game input dirs (sorted discovery, `.jsonl[.zst|.gz]` + `.mjai.json[.zst|.gz]`).
    pub data_dirs: Vec<PathBuf>,
    /// Staging target: games accumulate until this many staged rows.
    pub batch_rows: usize,
    /// `train` or `validation` (bound into quarantines, never into planes).
    pub split: String,
    /// Pinned source/object authority digest (validated non-empty at open).
    pub source_hash: String,
    /// Pinned rules-manifest digest (validated non-empty at open).
    pub rules_hash: String,
    /// Verified action-table CONTENT digest the closed-form walk ids pin to.
    pub action_table_hash: String,
}

/// Cumulative counters. `open_count` is 1 on every live stream (open-once).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeedStats {
    /// 1 on every live stream (open-once; stashed at close, still readable).
    pub open_count: u64,
    /// Games committed to caller buffers (drained prefix is all-ok).
    pub games_ok: u64,
    /// Games quarantined (frame/gate/walk verdicts; skips emit zero rows).
    pub games_quarantined: u64,
    /// Rows committed into caller buffers.
    pub rows_out: u64,
}

/// One quarantined game crossing the boundary (feed reason codes only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedQuar {
    /// Input file stem (string ids live cold-side only).
    pub game_id: String,
    /// Closed vocabulary (`framing`, `turn-order`, …).
    pub reason_code: &'static str,
    /// Cold lineage (`event_idx` + `game_idx`; never parsed hot).
    pub detail: String,
    /// Numeric game key (cold join key; wall digests are cold-side only).
    pub obs_hash: u64,
    /// Offending event index (sink lineage leg; saturated at `u16::MAX`).
    pub event_idx: u16,
    /// Verdict byte (`REASON_*` / `WALK_*`; renders [`Self::reason_code`]).
    pub reason: u8,
}

/// One committed fill: drained row count, per-call game accounting, bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeedFill {
    /// Rows committed into the caller prefix this call.
    pub rows: u32,
    /// Games consumed (staged) this call, ok + quarantined; multi-game files
    /// contribute their chunk count, so `games_ok + games_quarantined` over
    /// drained calls balances against this.
    pub games_consumed: u32,
    /// Games quarantined this call (zero rows from each).
    pub games_quarantined: u32,
    /// History bucket committed (`32|64|128|256`; 0 when no rows staged).
    pub t_len: u32,
}

/// Framed per-game inputs (plain `.jsonl` / `.mjai.json`, or compressed
/// `.jsonl.zst` / `.jsonl.gz` / `.mjai.json.zst` / `.mjai.json.gz`).
/// Decompression is the ingest arena's job; discovery filters names.
fn is_game_file(file_name: &str) -> bool {
    file_name.ends_with(".jsonl")
        || file_name.ends_with(".jsonl.zst")
        || file_name.ends_with(".jsonl.gz")
        || file_name.ends_with(".mjai.json")
        || file_name.ends_with(".mjai.json.zst")
        || file_name.ends_with(".mjai.json.gz")
}

/// Object id without a manifest: the file stem (mirrors the dump CLI
/// fallback for the probe layout).
fn object_id_for(file_name: &str) -> String {
    for suffix in [
        ".jsonl.zst",
        ".jsonl.gz",
        ".mjai.json.zst",
        ".mjai.json.gz",
        ".jsonl",
        ".mjai.json",
    ] {
        if let Some(stem) = file_name.strip_suffix(suffix) {
            return stem.to_string();
        }
    }
    file_name.to_string()
}

/// Numeric stand-in game key (cold lineage only; the wall digest is computed
/// cold-side — feed forbids sha hot — so the hot path never invents one).
fn obs_key_fallback(game_idx: u32) -> u64 {
    u64::from(game_idx).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Pure-Rust serial feed stream over framed per-game inputs.
pub struct FeedStream {
    files: Vec<PathBuf>,
    cursor: usize,
    game_seq: u32,
    batch_rows: usize,
    split: String,
    source_hash: String,
    rules_hash: String,
    action_table_hash: String,
    scratch: Scratch,
    quarantines: Vec<FeedQuar>,
    games_ok: u64,
    games_quarantined: u64,
    rows_out: u64,
    closed: bool,
}

impl FeedStream {
    /// Open-once: validate everything, discover + sort inputs, verify the
    /// action-table pin. No rows flow until the first `next_into`.
    pub fn open(cfg: FeedOpen) -> Result<Self, StreamError> {
        if cfg.data_dirs.is_empty() {
            return Err(StreamError::Open("data_dirs must not be empty".to_string()));
        }
        if cfg.batch_rows == 0 {
            return Err(StreamError::Open("batch must be >= 1".to_string()));
        }
        if cfg.split != "train" && cfg.split != "validation" {
            return Err(StreamError::Open(format!(
                "invalid split {:?}; expected train or validation",
                cfg.split
            )));
        }
        if cfg.source_hash.is_empty() {
            return Err(StreamError::Open(
                "source_hash must not be empty".to_string(),
            ));
        }
        if cfg.rules_hash.is_empty() {
            return Err(StreamError::Open(
                "rules_hash must not be empty".to_string(),
            ));
        }
        if cfg.action_table_hash.is_empty() {
            return Err(StreamError::Open(
                "action_table_hash must not be empty".to_string(),
            ));
        }
        // Table verify: the walk's closed-form ids are pinned entry-wise to
        // this digest by the walk module tests; open fails closed when the
        // pin itself drifts (never a silent id fork).
        if ledger::ACTION_TABLE_DIGEST.is_empty() || ledger::ACTION_TABLE_LEN != 6792 {
            return Err(StreamError::Table(
                "action table pin drifted (digest/len)".to_string(),
            ));
        }
        let mut files = Vec::new();
        for dir in &cfg.data_dirs {
            if !dir.is_dir() {
                return Err(StreamError::Open(format!(
                    "data dir is not a directory: {}",
                    dir.display()
                )));
            }
            let mut names: Vec<String> = fs::read_dir(dir)
                .map_err(|e| StreamError::Open(format!("inputs list {}: {e}", dir.display())))?
                .map(|entry| {
                    entry.map(|e| e.path()).map_err(|e| {
                        StreamError::Open(format!("inputs entry {}: {e}", dir.display()))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .filter_map(|path| {
                    let name = path.file_name()?.to_string_lossy().to_string();
                    is_game_file(&name).then_some(name)
                })
                .collect();
            names.sort();
            for name in names {
                files.push(dir.join(name));
            }
        }
        Ok(Self {
            files,
            cursor: 0,
            game_seq: 0,
            batch_rows: cfg.batch_rows,
            split: cfg.split,
            source_hash: cfg.source_hash,
            rules_hash: cfg.rules_hash,
            action_table_hash: cfg.action_table_hash,
            scratch: Scratch::new(),
            quarantines: Vec::new(),
            games_ok: 0,
            games_quarantined: 0,
            rows_out: 0,
            closed: false,
        })
    }

    /// Stage the file at `index`: read bytes, `stage_file` (split at start
    /// markers via `frame_file(base)`, then gate/walk/commit each sub-game
    /// in file order, wall-less path). Returns the staged-game advance:
    /// every chunk, ok or reject, consumes exactly one index from `base`,
    /// so the caller advances `game_seq` by the count, never by 1 (globally
    /// unique keys across files). Rejects become sidecar quarantines keyed
    /// by their sub-game index and contribute zero rows. `games_ok` is NOT
    /// bumped here: the scratch holds ok-games only, so the commit accounts
    /// them via `FillOut::games_consumed` exactly once at drain time.
    fn stage_at(&mut self, index: usize, game_idx: u32) -> Result<u32, StreamError> {
        let path = self
            .files
            .get(index)
            .ok_or_else(|| StreamError::Io("stream cursor past inputs".to_string()))?;
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let bytes =
            fs::read(path).map_err(|e| StreamError::Io(format!("input read {file_name}: {e}")))?;
        let game_id = object_id_for(&file_name);
        // File-stable object id (cursor position): disjoint `(object_id,
        // game_idx)` lineage keys across files even when `game_idx` ranges
        // overlap; the per-game base still comes from `game_idx`.
        let object_id = u32::try_from(index)
            .map_err(|_| StreamError::Envelope(format!("file index overflow {index}")))?;
        let out = self.scratch.stage_file(&bytes, object_id, game_idx);
        for reject in &out.rejects {
            let stub = reject.stub;
            self.quarantines.push(FeedQuar {
                game_id: game_id.clone(),
                reason_code: quarantine_reason_name(stub.reason),
                detail: format!("event_idx={} game_idx={}", stub.event_idx, stub.game_idx),
                obs_hash: obs_key_fallback(stub.game_idx),
                event_idx: stub.event_idx,
                reason: stub.reason,
            });
        }
        self.games_quarantined += out.rejects.len() as u64;
        Ok(out.advanced)
    }

    /// Commit staged rows into the caller planes.
    ///
    /// `byte_caps` are tensor byte sizes (`tensor.nbytes`); rows commit into
    /// the caller PREFIX exactly, the game-aligned committed prefix drains
    /// from the scratch, and any remainder stays staged for the next call.
    /// Exhaustion is `rows == 0 && games_consumed == 0` (a call that only
    /// stages quarantines reports `games_consumed > 0` with `rows == 0`).
    /// A too-small buffer drains nothing and moves no counter (whole-call
    /// rollback: cursor, scratch rows, quarantine pushes + counter).
    pub fn next_into(
        &mut self,
        ptrs: [u64; N_PLANES],
        byte_caps: [usize; N_PLANES],
    ) -> Result<FeedFill, StreamError> {
        if self.closed {
            return Err(StreamError::Closed);
        }
        // Fail-closed pointer check BEFORE any staging work (the fill
        // re-checks as defense in depth; this copy only saves the walk).
        for (plane, ptr) in ptrs.iter().enumerate() {
            if *ptr == 0 {
                // proof: `plane` is a plane index (< 26), fits `u8`.
                #[allow(clippy::cast_possible_truncation)]
                let plane_u: u8 = plane as u8;
                return Err(StreamError::NullPointer { plane: plane_u });
            }
        }
        // Whole-call snapshot for the BufferTooSmall rollback. The entry
        // scratch is always game-aligned (fills drain game-aligned
        // prefixes), so truncating back to `snap_rows` drops exactly this
        // call's staged games.
        let snap_cursor = self.cursor;
        let snap_seq = self.game_seq;
        let snap_rows = self.scratch.rows();
        let snap_quars = self.quarantines.len();
        let snap_quarantined = self.games_quarantined;

        let mut games_consumed = 0u32;
        while self.scratch.rows() < self.batch_rows {
            if self.cursor >= self.files.len() {
                break;
            }
            let index = self.cursor;
            let game_idx = self.game_seq;
            self.cursor += 1;
            let advance = self.stage_at(index, game_idx)?;
            games_consumed = games_consumed.saturating_add(advance.max(1));
            self.game_seq = self.game_seq.saturating_add(advance.max(1));
        }
        // proof: quarantine-count delta is a small vec-len difference, fits `u32`.
        #[allow(clippy::cast_possible_truncation)]
        let games_quarantined: u32 = (self.quarantines.len() - snap_quars) as u32;

        // Empty staging (fresh exhaustion): no bucket to pick, report zeros
        // without entering the fill (`t_len == 0` means "no rows, no bucket";
        // callers skip the transfer).
        if self.scratch.rows() == 0 {
            return Ok(FeedFill {
                rows: 0,
                games_consumed,
                games_quarantined,
                t_len: 0,
            });
        }
        match self
            .scratch
            .fill_pinned(&ptrs, &byte_caps)
            .map_err(fill_py_err)
        {
            Ok(out) => {
                self.games_ok += u64::from(out.games_consumed);
                self.rows_out += u64::from(out.rows);
                Ok(FeedFill {
                    rows: out.rows,
                    games_consumed,
                    games_quarantined,
                    t_len: u32::try_from(out.t_len).map_err(|_| {
                        StreamError::Envelope(format!("t_len overflow {}", out.t_len))
                    })?,
                })
            }
            Err(err) => {
                // Whole-call rollback: nothing drained, counters unmoved.
                self.cursor = snap_cursor;
                self.game_seq = snap_seq;
                self.scratch
                    .truncate_rows(u32::try_from(snap_rows).map_err(|_| {
                        StreamError::Envelope(format!("staged row count overflow {}", snap_rows))
                    })?);
                self.quarantines.truncate(snap_quars);
                self.games_quarantined = snap_quarantined;
                Err(err)
            }
        }
    }

    /// Cumulative counters (`open_count == 1` on every live stream).
    pub fn stats(&self) -> FeedStats {
        FeedStats {
            open_count: 1,
            games_ok: self.games_ok,
            games_quarantined: self.games_quarantined,
            rows_out: self.rows_out,
        }
    }

    /// Whole-game quarantines in load order (feed reason codes only; the
    /// fill path introduces none of its own).
    pub fn quarantines(&self) -> &[FeedQuar] {
        &self.quarantines
    }

    /// Bound split (`train` | `validation`).
    pub fn split(&self) -> &str {
        &self.split
    }
    /// Pinned source/object authority digest bound at open.
    pub fn source_hash(&self) -> &str {
        &self.source_hash
    }

    /// Pinned rules-manifest digest bound at open.
    pub fn rules_hash(&self) -> &str {
        &self.rules_hash
    }

    /// Verified action-table CONTENT digest bound at open.
    pub fn action_table_hash(&self) -> &str {
        &self.action_table_hash
    }

    /// Idempotent close: staged rows drop, counters/quarantines stay
    /// readable at the Rust level (the PyO3 side stashes them past `take`).
    pub fn close(&mut self) {
        self.closed = true;
        self.scratch.reset();
    }

    /// True once `close` ran (further `next_into` fails `Closed`).
    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

// ---------------------------------------------------------------------------
// PyO3 surface: Mutex-Option + detach + stats objects (live-handoff shape).
// ---------------------------------------------------------------------------

/// `next_into()` surface: committed rows, per-call game accounting, bucket.
#[pyclass(name = "PyFill", skip_from_py_object)]
#[derive(Clone, Copy)]
pub struct PyFill {
    /// Rows committed into the caller prefix this call.
    #[pyo3(get)]
    pub rows: u32,
    /// Games consumed (staged) this call, ok + quarantined.
    #[pyo3(get)]
    pub games_consumed: u32,
    /// Games quarantined this call (zero rows from each).
    #[pyo3(get)]
    pub games_quarantined: u32,
    /// History bucket committed (`32|64|128|256`; 0 when no rows staged).
    #[pyo3(get)]
    pub t_len: u32,
}

impl From<FeedFill> for PyFill {
    fn from(fill: FeedFill) -> Self {
        Self {
            rows: fill.rows,
            games_consumed: fill.games_consumed,
            games_quarantined: fill.games_quarantined,
            t_len: fill.t_len,
        }
    }
}

/// Cumulative counters crossing the boundary.
#[pyclass(name = "PyStats", skip_from_py_object)]
#[derive(Clone, Copy)]
pub struct PyStats {
    /// 1 on every live stream (open-once).
    #[pyo3(get)]
    pub open_count: u64,
    /// Games committed to caller buffers.
    #[pyo3(get)]
    pub games_ok: u64,
    /// Games quarantined.
    #[pyo3(get)]
    pub games_quarantined: u64,
    /// Rows committed into caller buffers.
    #[pyo3(get)]
    pub rows_out: u64,
}

impl From<FeedStats> for PyStats {
    fn from(stats: FeedStats) -> Self {
        Self {
            open_count: stats.open_count,
            games_ok: stats.games_ok,
            games_quarantined: stats.games_quarantined,
            rows_out: stats.rows_out,
        }
    }
}

/// Whole-game quarantine record crossing the boundary (feed reason codes).
#[pyclass(name = "PyQuar", skip_from_py_object)]
#[derive(Clone)]
pub struct PyQuar {
    /// Input file stem.
    #[pyo3(get)]
    pub game_id: String,
    /// Closed vocabulary (`framing`, `turn-order`, …).
    #[pyo3(get)]
    pub reason_code: String,
    /// Cold lineage (`event_idx` + `game_idx`).
    #[pyo3(get)]
    pub detail: String,
    /// Numeric game key (cold join key).
    #[pyo3(get)]
    pub obs_hash: u64,
    /// Offending event index (sink lineage leg).
    #[pyo3(get)]
    pub event_idx: u16,
    /// Verdict byte (`REASON_*` / `WALK_*`).
    #[pyo3(get)]
    pub reason: u8,
}

impl From<&FeedQuar> for PyQuar {
    fn from(q: &FeedQuar) -> Self {
        Self {
            game_id: q.game_id.clone(),
            reason_code: q.reason_code.to_string(),
            detail: q.detail.clone(),
            obs_hash: q.obs_hash,
            event_idx: q.event_idx,
            reason: q.reason,
        }
    }
}

/// Post-close stash: `close` takes the stream, but stats/quarantines stay
/// readable.
#[derive(Debug, Clone, Default)]
struct TailStats {
    games_ok: u64,
    games_quarantined: u64,
    rows_out: u64,
    quars: Vec<FeedQuar>,
}

/// PyO3 handoff: `Mutex<Option<Stream>>` with poison/closed errors, `detach`
/// around every blocking section (file reads + stage + fill). The WHOLE
/// fill runs detached under the SAME mutex guard (serial, never
/// unlock-relock mid-fill).
#[pyclass(name = "PyHydraStream", frozen)]
pub struct PyHydraStream {
    inner: Mutex<Option<FeedStream>>,
    tail: Mutex<TailStats>,
}

#[pymethods]
impl PyHydraStream {
    /// Open-once: validate everything, then detach the sorted discovery +
    /// table verify. No rows flow until the first `next_into`.
    #[staticmethod]
    #[pyo3(signature = (data_dirs, batch, split, source_hash, rules_hash, action_table_hash))]
    fn open(
        py: Python<'_>,
        data_dirs: Vec<PathBuf>,
        batch: usize,
        split: &str,
        source_hash: &str,
        rules_hash: &str,
        action_table_hash: &str,
    ) -> PyResult<Self> {
        if data_dirs.is_empty() {
            return Err(PyValueError::new_err(
                "hydra2 replay data_dirs must not be empty",
            ));
        }
        if batch == 0 {
            return Err(PyValueError::new_err("hydra2 replay batch must be >= 1"));
        }
        let cfg = FeedOpen {
            data_dirs,
            batch_rows: batch,
            split: split.to_string(),
            source_hash: source_hash.to_string(),
            rules_hash: rules_hash.to_string(),
            action_table_hash: action_table_hash.to_string(),
        };
        let stream = py.detach(|| FeedStream::open(cfg)).map_err(stream_py_err)?;
        Ok(Self {
            inner: Mutex::new(Some(stream)),
            tail: Mutex::new(TailStats::default()),
        })
    }

    /// Commit staged rows into the caller planes.
    ///
    /// `ptrs` are 26 pinned-slot base addresses (canonical plane order #0-25),
    /// their byte sizes (`tensor.nbytes`). Rows land in the caller PREFIX
    /// exactly; `t_len` selects the committed history width. The GIL is
    /// released for the entire fill. A null pointer fails closed; a
    /// too-small buffer raises `PyBufferError`, drains nothing, and moves
    /// no counter (retry with a bigger buffer).
    #[pyo3(signature = (ptrs, byte_caps))]
    fn next_into(
        &self,
        py: Python<'_>,
        ptrs: [u64; 26],
        byte_caps: [usize; 26],
    ) -> PyResult<PyFill> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 replay stream mutex poisoned"))?;
        let stream = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("hydra2 replay stream is closed"))?;
        // SAME guard across the detach: serial, never unlock-relock mid-fill.
        py.detach(|| {
            stream
                .next_into(ptrs, byte_caps)
                .map(PyFill::from)
                .map_err(stream_py_err)
        })
    }

    /// Cumulative counters (readable post-close via the close stash).
    fn stats(&self) -> PyResult<PyStats> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 replay stream mutex poisoned"))?;
        if let Some(stream) = guard.as_ref() {
            return Ok(PyStats::from(stream.stats()));
        }
        let tail = self
            .tail
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 replay stream mutex poisoned"))?;
        Ok(PyStats {
            open_count: 1,
            games_ok: tail.games_ok,
            games_quarantined: tail.games_quarantined,
            rows_out: tail.rows_out,
        })
    }

    /// Whole-game quarantines in load order (readable post-close).
    fn quarantines(&self) -> PyResult<Vec<PyQuar>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 replay stream mutex poisoned"))?;
        if let Some(stream) = guard.as_ref() {
            return Ok(stream.quarantines().iter().map(PyQuar::from).collect());
        }
        let tail = self
            .tail
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 replay stream mutex poisoned"))?;
        Ok(tail.quars.iter().map(PyQuar::from).collect())
    }

    /// Idempotent close: stash stats/quarantines, then take + detach the
    /// drop. The second call is a no-op success.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let stream = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 replay stream mutex poisoned"))?
            .take();
        if let Some(stream) = stream {
            let stats = stream.stats();
            let quars = stream.quarantines().to_vec();
            if let Ok(mut tail) = self.tail.lock() {
                tail.games_ok = stats.games_ok;
                tail.games_quarantined = stats.games_quarantined;
                tail.rows_out = stats.rows_out;
                tail.quars = quars;
            }
            py.detach(|| std::mem::drop(stream));
        }
        Ok(())
    }
}
