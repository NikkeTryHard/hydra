//! StreamDriver: Rust-owned Pull>Take over `feed::fill` (batcher+pool+order+take).
//!
//! Inversion backlog prize #2 / D4: Python MUST NOT orchestrate Rust via
//! PyO3. The Python dataset currently owns pull batching
//! (`_pull_streamed_batch`), per-worker chunking, the spawn pool
//! (`_fill_parallel` via `_expand_streamed_chunk`), ordered merging
//! (`_merge_expanded`), contiguous-prefix takes (`_consume_microbatch`),
//! quarantine classes (`_quarantine_class`), the sidecar window hash
//! (`_sidecar_window_hash`), and buffer snapshots. This module moves all of
//! that control flow into Rust; Python keeps file-IO authority (reads files,
//! passes batched raw bytes in), torch collate/H2D (pinned-ring transfer
//! stays caller-side), and serde envelopes (builds `DecisionRow`/
//! `GameRecord` dataclasses from the plain vectors returned here).
//!
//! Boundary (batched-bytes in, envelopes out, cursors via `resume`):
//!
//! In: `DriverJob { bytes, game_id, key, wall }` per game in pull order.
//! `bytes` are the verbatim framed game bytes (`StreamGame.raw`);
//! `game_id`/`key` (`raw_bytes_sha256`) are the file-IO envelope Python
//! already owns; `wall` (`Some` 136 tiles) splices the S7 override in Rust
//! via [`inject_wall`] (no Python re-serialization), `None` walks the
//! embedded content verbatim. Order is input order, never shuffled here:
//! manifest/shuffle/split assignment stays Python (file-IO authority).
//!
//! Pool+order: [`stage_batch`] on the SOLE hot [`FeedPool`] (rayon,
//! bounded channel, indexed slots) expands the batch with deterministic
//! input-order outcomes; [`Scratch::commit`] appends whole ok games in
//! that order. No Python pool, no `pool.map` (one game's failure never
//! cancels the batch tail: per-item failures ride as quarantine data).
//!
//! Take: contiguous-prefix microbatches (`_consume_microbatch` shape):
//! `next_microbatch` fills the caller's 26 planes via raw `data_ptr`
//! handles (row-exact, may cut mid-game like the 5/5/5 takes) with the
//! committed history width (`pick_t_len`), then returns the take envelope
//! (`ids`/`chosen`/`kinds`/`counts`/`classes`/`snapshot`). Consumed rows
//! advance `offset`; whole consumed games compact past
//! `COMPACT_ROWS` (entries stay whole-game aligned, at most one partial
//! game strands, mirroring `_compact`).
//!
//! Out: `ids` are `<game_id>:d<seq:04d>` (matches `_slim_row_dicts`
//! exactly); `chosen` are the closed-form walk ids; `kinds` are the frozen
//! census kind literals (`"unknown"` when unmapped, mirroring
//! `_action_kind_for_id`); `counts` are
//! `(replayed, sim_replayed, quarantined, microbatches)`; `classes` are the
//! normalized quarantine classes (`_quarantine_class` shape, capped at 32
//! plus `_other`); `snapshot` is the buffer frontier (`offset`/`dropped`/
//! `microbatches`/`replayed`/`sim_replayed`/`quarantined`/`total_rows`/
//! `row_hash` + whole-game `(key, rows)` entries). File cursors
//! (`file_index`/`byte_offset`/`shuffle_pos`) stay Python via
//! `resume::pack_cursor`; the RNG triple (`StreamSnapshot`) rides the
//! shuffle snapshot, not these entries.
//!
//! Window hash (`row_hash`) is byte-identical to `_sidecar_window_hash`:
//! per entry `key utf8 + 0x00 + count le64`, then per row `chosen le64
//! signed + kind utf8 + 0x00`, in window order. Per-row `decision_id`
//! strings never enter the digest (identity rides the per-game `key`).
//!
//! Determinism: input order + [`stage_batch`] indexed slots + census order;
//! no wall-clock, no global RNG, no `hash()`. The `seed` is a cursor
//! passthrough (semantic counter-based, e.g. the parity `data_seed`), never
//! consumed as randomness here. Thread count does not perturb order.
//!
//! Failure atomicity: `push_games` quarantines-and-counts per game (fail
//! closed, never a silent drop); only fully expanded games buffer rows.
//! `next_microbatch` validates caps before mutating (too-small drains
//! nothing, reports the first-short plane); exhaustion is an error (never a
//! silent short take). Every shape violation fails closed (`DriverError`,
//! never a default).
//!
//! Feed-only deps: `crate::{census, fill, ledger, quarantine}` + `sha2` +
//! `std` + `rayon` (via `fill`). NEVER `hydra-shard` (DAG one-way), NEVER
//! pyo3/sha2-hot digests (window hash is the cold snapshot path, not the row
//! hot path), NEVER file IO (caller passes bytes).

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::census::frozen_census;
use crate::fill::{
    FeedPool, Scratch, StageJob, checked_slice, inject_wall, pick_t_len, row_bytes,
    stage_batch, FillError, N_PLANES, ROW_BYTES_FIXED,
};
use crate::ledger::{PLANE_CHOSEN, PLANE_HIST_KIND, PLANE_HIST_MASK};
use crate::quarantine::reason_class;

/// Consumed-row prefix dropped at a time (mirrors `_BUFFER_COMPACT_ROWS`:
/// 8192 rows keeps the live buffer near in-flight games only).
pub const COMPACT_ROWS: usize = 8192;

/// Max games per `push_games` call (mirrors `expand_batch_games` max 1024).
pub const MAX_BATCH_GAMES: usize = 1024;

/// Max rows per `next_microbatch` take (same bound, fail-closed).
pub const MAX_TAKE_ROWS: usize = 1024;

/// Max distinct quarantine reason classes retained (mirrors
/// `_QUARANTINE_REASON_CLASSES`; overflow merges into `"_other"`).
pub const QUARANTINE_REASON_CLASSES: usize = 32;

/// Closed driver failure vocabulary (every variant maps to a Python error
/// at the bridge: `BufferTooSmall` is the ONLY `PyBufferError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverError {
    /// Bad argument or drifted buffer (never a partial take).
    InvalidArg(String),
    /// Rayon pool build failure (fail closed at open).
    Pool(String),
    /// Take needs more buffered rows than the driver holds (single-pass:
    /// stream end with updates remaining).
    Exhausted(String),
    /// Caller buffer too small: nothing drained, retry with a bigger buffer.
    BufferTooSmall {
        /// First-short plane (canonical order #0-25).
        plane: u8,
        /// Take footprint for `plane` (bytes).
        needed: usize,
        /// `byte_caps[plane]`.
        capacity: usize,
    },
}

impl core::fmt::Display for DriverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DriverError::InvalidArg(msg) => write!(f, "hydra2 stream driver invalid: {msg}"),
            DriverError::Pool(msg) => write!(f, "hydra2 stream driver pool: {msg}"),
            DriverError::Exhausted(msg) => write!(f, "hydra2 stream driver exhausted: {msg}"),
            DriverError::BufferTooSmall {
                plane,
                needed,
                capacity,
            } => write!(
                f,
                "hydra2 stream driver plane {plane} needs {needed} bytes but the caller buffer holds {capacity}"
            ),
        }
    }
}

impl std::error::Error for DriverError {}

fn fill_err(err: FillError) -> DriverError {
    match err {
        FillError::NullPointer { plane } => DriverError::BufferTooSmall {
            plane,
            needed: 0,
            capacity: 0,
        },
        FillError::BufferTooSmall {
            plane,
            needed_bytes,
            capacity_bytes,
        } => DriverError::BufferTooSmall {
            plane,
            needed: needed_bytes,
            capacity: capacity_bytes,
        },
    }
}

/// One pulled game: batched bytes + file-IO envelope (pull order).
#[derive(Clone, Debug)]
pub struct DriverJob {
    /// Verbatim framed game bytes (`StreamGame.raw`, trailing newline).
    pub bytes: Vec<u8>,
    /// Game identity for `decision_id` (`<game_id>:d<seq:04d>`).
    pub game_id: String,
    /// Whole-game key for the window hash (`raw_bytes_sha256`, `sha256:`).
    pub key: String,
    /// S7 wall override (136 tiles) or `None` for the wall-less sim path.
    pub wall: Option<Vec<u32>>,
}

/// Whole-game buffer index entry (hash-relevant half; `path`/`offset`/
/// `split` stay Python file-IO authority and rejoin there).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriverEntry {
    /// Whole-game key (`raw_bytes_sha256`).
    pub key: String,
    /// Row count contributed by this game (always >= 1).
    pub rows: u32,
}

/// Per-row take metadata (parallel to the staged planes).
#[derive(Clone, Debug, PartialEq, Eq)]
struct RowMeta {
    /// Index into `entries`/`game_ids` (whole-game alignment).
    entry_idx: u32,
    /// Row sequence within the game (`d<seq:04d>`).
    seq: u32,
    /// Closed-form chosen action id.
    chosen: i64,
}

/// Batch push receipt (per-call game accounting).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PushOut {
    /// Games committed to the buffer this call.
    pub games_ok: u32,
    /// Games quarantined this call (zero rows each).
    pub games_quarantined: u32,
    /// Rows added to the buffer this call.
    pub rows_added: u32,
}

/// Microbatch take: plane fill + take envelope (ids/kinds/counts/classes/
/// frontier). Planes land in the caller prefix exactly; `t_len` selects the
/// committed history width. Takes stay `O(take)`: the game-keyed window hash
/// (`sha256:`, K4 shape) rides [`StreamDriver::snapshot`] only (verify at
/// checkpoints, never per take — hashing the live window per take would make
/// a run quadratic in buffered rows).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicrobatchOut {
    /// Rows committed into the caller prefix this take.
    pub rows: u32,
    /// History bucket committed (`32|64|128|256`).
    pub t_len: u32,
    /// Take decision ids (`<game_id>:d<seq:04d>`, take order).
    pub ids: Vec<String>,
    /// Take chosen action ids (take order).
    pub chosen: Vec<i64>,
    /// Take action kinds (frozen census literals, take order).
    pub kinds: Vec<String>,
    /// Games replayed (walled path) lifetime.
    pub replayed: u64,
    /// Games sim-replayed (wall-less path) lifetime.
    pub sim_replayed: u64,
    /// Games quarantined lifetime.
    pub quarantined: u64,
    /// Quarantine reason classes (capped at 32 plus `_other`, sorted).
    pub classes: Vec<(String, u64)>,
    /// Rows consumed (plus tail-dropped) lifetime.
    pub offset: u64,
    /// Compacted prefix length lifetime.
    pub dropped: u64,
    /// Microbatches consumed lifetime.
    pub microbatches: u64,
    /// Buffered rows (`rows_meta.len()`).
    pub total_rows: u64,
    /// Games pushed (ok + quarantined) lifetime.
    pub games_seen: u64,
}

/// Buffer snapshot: whole-game entries + counters + row hash
/// (verify-before-mutate shape; `path`/`offset`/`split` rejoin Python-side).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverSnapshot {
    /// Whole-game `(key, rows)` entries in buffer order.
    pub entries: Vec<DriverEntry>,
    /// Rows consumed lifetime.
    pub offset: u64,
    /// Compacted prefix length lifetime.
    pub dropped: u64,
    /// Microbatches consumed lifetime.
    pub microbatches: u64,
    /// Games replayed lifetime.
    pub replayed: u64,
    /// Games sim-replayed lifetime.
    pub sim_replayed: u64,
    /// Games quarantined lifetime.
    pub quarantined: u64,
    /// Quarantine classes (sorted).
    pub classes: Vec<(String, u64)>,
    /// Window hash over the live buffer.
    pub row_hash: String,
    /// Buffered rows.
    pub total_rows: u64,
}

/// Cumulative counters (observability only, never identity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DriverStats {
    /// Games pushed lifetime.
    pub games_seen: u64,
    /// Games committed lifetime.
    pub games_ok: u64,
    /// Games quarantined lifetime.
    pub games_quarantined: u64,
    /// Rows committed lifetime.
    pub rows_in: u64,
    /// Rows consumed lifetime.
    pub rows_out: u64,
    /// Microbatches consumed lifetime.
    pub microbatches: u64,
}

/// Honest kind for one table index (`"unknown"` when unmapped).
///
/// Mirrors `_action_kind_for_id`: the 6792 census is authoritative
/// (index == action id); out-of-range ids fall back to `"unknown"` (never
/// synthesized, never raised).
fn kind_for_chosen(chosen: i64) -> &'static str {
    if chosen < 0 {
        return "unknown";
    }
    let idx = chosen as usize;
    let census = frozen_census();
    if idx >= census.len() {
        return "unknown";
    }
    let name = census[idx].kind.name();
    if name.is_empty() {
        "unknown"
    } else {
        name
    }
}

/// Normalize a quarantine reason to a stable, game-identity-free class.
///
/// Mirrors `_quarantine_class` exactly (message-based, truncation at 100
/// chars): `game '...'`/`game "..."` collapse to `game '<id>'`,
/// `game-<hex>{4,}` collapses to `game-<id>`, spaceless `"token"` quotes
/// unify to `'token'`, `kyoku N`/`seat N` collapse to `<n>`. Spaced prose
/// quotes are untouched.
fn quarantine_class(msg: &str) -> String {
    let step1 = collapse_game_quotes(msg, '\'');
    let step2 = collapse_game_quotes(&step1, '"');
    let step3 = collapse_game_hex(&step2);
    let step4 = unify_spaceless_quotes(&step3);
    let step5 = collapse_word_number(&step4, "kyoku");
    let step6 = collapse_word_number(&step5, "seat");
    step6.chars().take(100).collect()
}

/// Collapse `game '...'` or `game "..."` (by `quote`) to `game '<id>'`.
fn collapse_game_quotes(text: &str, quote: char) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if text[i..].starts_with("game ")
            && bytes.get(i + 5) == Some(&(quote as u8))
        {
            let mut j = i + 6;
            while j < bytes.len() && bytes[j] != quote as u8 {
                j += 1;
            }
            if j < bytes.len() {
                out.push_str("game '<id>'");
                i = j + 1;
                continue;
            }
        }
        if let Some(ch) = text[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

/// Collapse `game-<lowerhex>{4,}` to `game-<id>`.
fn collapse_game_hex(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if text[i..].starts_with("game-") {
            let mut j = i + 5;
            while j < bytes.len() && is_lower_hex(bytes[j]) {
                j += 1;
            }
            if j - (i + 5) >= 4 {
                out.push_str("game-<id>");
                i = j;
                continue;
            }
        }
        if let Some(ch) = text[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

#[inline]
fn is_lower_hex(b: u8) -> bool {
    b.is_ascii_hexdigit() && !b.is_ascii_uppercase()
}

/// Unify spaceless `"token"` (`[A-Za-z0-9_-]+`) to `'token'`.
fn unify_spaceless_quotes(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            let mut j = i + 1;
            while j < bytes.len() && is_token_byte(bytes[j]) {
                j += 1;
            }
            if j > i + 1 && j < bytes.len() && bytes[j] == b'"' {
                out.push('\'');
                out.push_str(&text[i + 1..j]);
                out.push('\'');
                i = j + 1;
                continue;
            }
        }
        if let Some(ch) = text[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

#[inline]
fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

/// Collapse `<word> <digits>` to `<word> <n>`.
fn collapse_word_number(text: &str, word: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if text[i..].starts_with(word)
            && bytes.get(i + word.len()) == Some(&b' ')
        {
            let mut j = i + word.len() + 1;
            let digits = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > digits {
                out.push_str(word);
                out.push_str(" <n>");
                i = j;
                continue;
            }
        }
        if let Some(ch) = text[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

/// Read one `i64` LE from `src` at `off` (fail-closed, never panics).
fn read_i64_le(src: &[u8], off: usize) -> Result<i64, DriverError> {
    let end = off.saturating_add(8);
    if end > src.len() || off > end {
        return Err(DriverError::InvalidArg(
            "driver chosen plane truncated".to_string(),
        ));
    }
    let slice = src.get(off..end).ok_or_else(|| {
        DriverError::InvalidArg("driver chosen plane truncated".to_string())
    })?;
    let mut buf = [0u8; 8];
    let mut k = 0usize;
    while k < 8 {
        buf[k] = slice[k];
        k += 1;
    }
    Ok(i64::from_le_bytes(buf))
}

/// Game-keyed window hash over the live buffer (K4 sidecar shape).
///
/// Binds each whole-game record (`key` + `rows`) plus that game's row
/// payload (`chosen` + `kind` per row, window order). Byte-identical to
/// `_sidecar_window_hash` (drift fails closed, never coerced).
fn window_hash(entries: &[DriverEntry], metas: &[RowMeta]) -> Result<String, DriverError> {
    let mut total: u64 = 0;
    for entry in entries {
        total = total.saturating_add(u64::from(entry.rows));
    }
    if total != metas.len() as u64 {
        return Err(DriverError::InvalidArg(format!(
            "buffer index drift: {total} indexed rows != {} buffered",
            metas.len()
        )));
    }
    let mut h = Sha256::new();
    let mut pos: usize = 0;
    for entry in entries {
        if entry.key.is_empty() {
            return Err(DriverError::InvalidArg(
                "sidecar window entry key must be a non-empty str".to_string(),
            ));
        }
        if entry.rows == 0 {
            return Err(DriverError::InvalidArg(
                "sidecar window entry rows must be a positive int".to_string(),
            ));
        }
        h.update(entry.key.as_bytes());
        h.update([0u8]);
        h.update(u64::from(entry.rows).to_le_bytes());
        let count = entry.rows as usize;
        let end = pos.saturating_add(count);
        if end > metas.len() || pos > end {
            return Err(DriverError::InvalidArg(
                "buffer index drift: entry slice out of range".to_string(),
            ));
        }
        let mut k = pos;
        while k < end {
            let meta = &metas[k];
            h.update(meta.chosen.to_le_bytes());
            h.update(kind_for_chosen(meta.chosen).as_bytes());
            h.update([0u8]);
            k += 1;
        }
        pos = end;
    }
    let sum = h.finalize();
    let mut out = String::with_capacity(7 + 64);
    out.push_str("sha256:");
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for b in sum {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    Ok(out)
}

/// Rust-owned Pull>Take driver: batcher+pool+order+take over staged planes.
pub struct StreamDriver {
    pool: FeedPool,
    scratch: Scratch,
    rows_meta: Vec<RowMeta>,
    entries: Vec<DriverEntry>,
    game_ids: Vec<String>,
    seed: u64,
    compact_rows: usize,
    offset: u64,
    dropped: u64,
    microbatches: u64,
    replayed: u64,
    sim_replayed: u64,
    quarantined: u64,
    classes: BTreeMap<String, u64>,
    games_seen: u64,
    games_ok: u64,
    rows_in: u64,
    rows_out: u64,
    closed: bool,
}

impl StreamDriver {
    /// Open-once driver: build the SOLE hot pool, pin the cursor seed and
    /// the compact bound. No rows flow until the first `push_games`.
    pub fn open(seed: u64, threads: usize, compact_rows: usize) -> Result<Self, DriverError> {
        if compact_rows == 0 {
            return Err(DriverError::InvalidArg(
                "driver compact_rows must be >= 1".to_string(),
            ));
        }
        let pool = FeedPool::build(threads)
            .map_err(|e| DriverError::Pool(format!("driver pool build: {e}")))?;
        Ok(Self {
            pool,
            scratch: Scratch::new(),
            rows_meta: Vec::new(),
            entries: Vec::new(),
            game_ids: Vec::new(),
            seed,
            compact_rows,
            offset: 0,
            dropped: 0,
            microbatches: 0,
            replayed: 0,
            sim_replayed: 0,
            quarantined: 0,
            classes: BTreeMap::new(),
            games_seen: 0,
            games_ok: 0,
            rows_in: 0,
            rows_out: 0,
            closed: false,
        })
    }

    /// Cursor seed passthrough (semantic counter-based, never wall-clock).
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// True once `close` ran (further `push`/`take` fail `Closed` as
    /// `InvalidArg`).
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Idempotent close: staged rows drop, counters stay readable.
    pub fn close(&mut self) {
        self.closed = true;
        self.scratch.reset();
    }

    fn fail_if_closed(&self) -> Result<(), DriverError> {
        if self.closed {
            return Err(DriverError::InvalidArg(
                "hydra2 stream driver is closed".to_string(),
            ));
        }
        Ok(())
    }

    fn count_quarantine(&mut self, msg: &str) {
        self.quarantined = self.quarantined.saturating_add(1);
        let class = quarantine_class(msg);
        if let Some(count) = self.classes.get_mut(&class) {
            *count = count.saturating_add(1);
            return;
        }
        if self.classes.len() < QUARANTINE_REASON_CLASSES {
            self.classes.insert(class, 1);
        } else {
            let other = self.classes.get("_other").copied().unwrap_or(0);
            self.classes
                .insert("_other".to_string(), other.saturating_add(1));
        }
    }

    /// Pull: expand `jobs` in pull order via the owned pool and buffer whole
    /// ok games (quarantines ride as counted classes, never silent drops).
    ///
    /// `jobs` arrive in pull order (manifest/shuffle/split already applied
    /// Python-side); outcomes commit in that same order. Empty input is a
    /// no-op success (lets takes drain the remainder without a special
    /// case).
    pub fn push_games(&mut self, jobs: Vec<DriverJob>) -> Result<PushOut, DriverError> {
        self.fail_if_closed()?;
        if jobs.len() > MAX_BATCH_GAMES {
            return Err(DriverError::InvalidArg(format!(
                "driver batch games invalid: {} (max {MAX_BATCH_GAMES})",
                jobs.len()
            )));
        }
        if jobs.is_empty() {
            return Ok(PushOut {
                games_ok: 0,
                games_quarantined: 0,
                rows_added: 0,
            });
        }
        // Whole-batch invariant: staged planes and row metadata stay joined
        // (`scratch.rows() == rows_meta.len()`); fail closed on drift before
        // touching either (never a half-joined batch).
        if self.scratch.rows() != self.rows_meta.len() {
            return Err(DriverError::InvalidArg(format!(
                "buffer index drift: {} staged rows != {} buffered",
                self.scratch.rows(),
                self.rows_meta.len()
            )));
        }
        // Per-game pre-gates (mirror `_wall_override` + envelope gates):
        // malformed envelopes quarantine the GAME, never the batch.
        let mut pre_err: Vec<Option<String>> = Vec::with_capacity(jobs.len());
        let mut idxs: Vec<usize> = Vec::with_capacity(jobs.len());
        let mut pos = 0usize;
        while pos < jobs.len() {
            let job = &jobs[pos];
            if job.bytes.is_empty() {
                pre_err.push(Some(format!(
                    "driver game '{}' bytes must be non-empty",
                    job.game_id
                )));
            } else if job.game_id.is_empty() {
                pre_err.push(Some("driver game_id must be a non-empty str".to_string()));
            } else if job.key.is_empty() {
                pre_err.push(Some(
                    "driver game key must be a non-empty str".to_string(),
                ));
            } else if let Some(wall) = job.wall.as_ref() {
                if wall.len() != 136 {
                    pre_err.push(Some(format!(
                        "wall_tiles must carry 136 tiles, got {}",
                        wall.len()
                    )));
                } else {
                    pre_err.push(None);
                    idxs.push(pos);
                }
            } else {
                pre_err.push(None);
                idxs.push(pos);
            }
            pos += 1;
        }
        // StageJobs for the gate-clean positions (game_idx keys the ordered
        // rejoin; object_id stays 0 like the per-game replay path).
        let base_seen = self.games_seen;
        let mut stage_jobs: Vec<StageJob> = Vec::with_capacity(idxs.len());
        let mut k = 0usize;
        while k < idxs.len() {
            let pos = idxs[k];
            let job = &jobs[pos];
            let game_idx = u32::try_from(base_seen.saturating_add(pos as u64)).map_err(|_| {
                DriverError::InvalidArg("driver game index overflow".to_string())
            })?;
            let bytes = match job.wall.as_ref() {
                Some(wall) => inject_wall(&job.bytes, wall),
                None => job.bytes.clone(),
            };
            stage_jobs.push(StageJob {
                game_idx,
                object_id: 0,
                bytes,
                wall_digest: None,
            });
            k += 1;
        }
        // Ordered pool expansion (one GIL release at the bridge; zero Python
        // API in here). `stage_batch` returns input-ordered outcomes split
        // into ok/reject legs; the cursors below merge them back into pull
        // order in one linear pass (both legs arrive ordered, so no map).
        let (games, rejects) = stage_batch(&self.pool, &stage_jobs);
        let mut gi = 0usize;
        let mut ri = 0usize;
        // Pull-order merge (mirrors `_pull_game_batch`): counters, rows,
        // whole-game index, and classes update exactly as the serial pull
        // does; only the Rust handoff was batched.
        let mut games_ok: u32 = 0;
        let mut games_quarantined: u32 = 0;
        let mut rows_added: u32 = 0;
        let mut pos = 0usize;
        while pos < jobs.len() {
            self.games_seen = self.games_seen.saturating_add(1);
            if let Some(msg) = pre_err[pos].as_ref() {
                self.count_quarantine(msg);
                games_quarantined = games_quarantined.saturating_add(1);
                pos += 1;
                continue;
            }
            let game_idx = u32::try_from(base_seen.saturating_add(pos as u64)).map_err(|_| {
                DriverError::InvalidArg("driver game index overflow".to_string())
            })?;
            if games.get(gi).is_some_and(|game| game.game_idx == game_idx) {
                let game = &games[gi];
                gi += 1;
                let job = &jobs[pos];
                let entry_idx = u32::try_from(self.entries.len()).map_err(|_| {
                    DriverError::InvalidArg("driver buffer entry overflow".to_string())
                })?;
                let chosen_plane = game
                    .planes
                    .get(PLANE_CHOSEN)
                    .ok_or_else(|| {
                        DriverError::InvalidArg("driver chosen plane missing".to_string())
                    })?;
                let mut seq: u32 = 0;
                while seq < game.rows {
                    let off = (seq as usize).saturating_mul(8);
                    let chosen = read_i64_le(chosen_plane, off)?;
                    self.rows_meta.push(RowMeta {
                        entry_idx,
                        seq,
                        chosen,
                    });
                    seq = seq.saturating_add(1);
                }
                self.scratch.commit(game);
                self.entries.push(DriverEntry {
                    key: job.key.clone(),
                    rows: game.rows,
                });
                self.game_ids.push(job.game_id.clone());
                if job.wall.is_none() {
                    self.sim_replayed = self.sim_replayed.saturating_add(1);
                } else {
                    self.replayed = self.replayed.saturating_add(1);
                }
                self.games_ok = self.games_ok.saturating_add(1);
                self.rows_in = self.rows_in.saturating_add(u64::from(game.rows));
                games_ok = games_ok.saturating_add(1);
                rows_added = rows_added.saturating_add(game.rows);
            } else if rejects.get(ri).is_some_and(|reject| reject.stub.game_idx == game_idx) {
                let stub = rejects[ri].stub;
                ri += 1;
                let job = &jobs[pos];
                let msg = format!(
                    "rust walk quarantined game '{}': {} at event {}",
                    job.game_id,
                    reason_class(stub.reason),
                    u32::from(stub.event_idx)
                );
                self.count_quarantine(&msg);
                games_quarantined = games_quarantined.saturating_add(1);
            } else {
                return Err(DriverError::InvalidArg(format!(
                    "bridge batch drift: no outcome for batch position {pos}"
                )));
            }
            pos += 1;
        }
        Ok(PushOut {
            games_ok,
            games_quarantined,
            rows_added,
        })
    }

    /// Buffered-but-unconsumed rows (list length minus compacted prefix).
    fn live_count(&self) -> u64 {
        let consumed = self.offset.saturating_sub(self.dropped);
        (self.rows_meta.len() as u64).saturating_sub(consumed)
    }

    /// Take: advance exactly one microbatch through the fill machine.
    ///
    /// Buffers must already hold `batch_size` live rows (push first; the
    /// Python `_fill` loop becomes push-until-live + take). Takes are
    /// contiguous-prefix advances (`offset`), so sampler offset/hash/
    /// counter semantics match `_consume_microbatch` bit-for-bit. Planes
    /// land in the caller prefix exactly with the committed history width;
    /// the remainder stays buffered for the next take.
    pub fn next_microbatch(
        &mut self,
        batch_size: usize,
        ptrs: &[u64; N_PLANES],
        byte_caps: &[usize; N_PLANES],
    ) -> Result<MicrobatchOut, DriverError> {
        self.fail_if_closed()?;
        if batch_size == 0 || batch_size > MAX_TAKE_ROWS {
            return Err(DriverError::InvalidArg(format!(
                "driver batch_size invalid: {batch_size} (int in [1, {MAX_TAKE_ROWS}])"
            )));
        }
        if self.scratch.rows() != self.rows_meta.len() {
            return Err(DriverError::InvalidArg(format!(
                "buffer index drift: {} staged rows != {} buffered",
                self.scratch.rows(),
                self.rows_meta.len()
            )));
        }
        let live = self.live_count();
        if live < batch_size as u64 {
            return Err(DriverError::Exhausted(format!(
                "stream exhausted with {live} buffered rows, need {batch_size} \
                 (single-pass: stream end with updates remaining; \
                 rescope loop.max_updates to supply)"
            )));
        }
        let consumed = self.offset.saturating_sub(self.dropped);
        let start = usize::try_from(consumed).map_err(|_| {
            DriverError::InvalidArg("driver buffer offset overflow".to_string())
        })?;
        let end = start.saturating_add(batch_size);
        if end > self.rows_meta.len() || start > end {
            return Err(DriverError::InvalidArg(
                "driver take slice out of range".to_string(),
            ));
        }
        // Committed history width BEFORE touching caller memory (breaks the
        // `t_len`/`rows` circle like the stager: the take picks T, then the
        // fill commits exactly the take prefix).
        let hist = self.scratch.hist_lens();
        let mut max_hist: u32 = 0;
        let mut r = start;
        while r < end {
            let hl = hist.get(r).copied().unwrap_or(0);
            if hl > max_hist {
                max_hist = hl;
            }
            r += 1;
        }
        let t_len = match pick_t_len(max_hist) {
            Some(t) => t,
            None => {
                return Err(DriverError::BufferTooSmall {
                    plane: PLANE_HIST_KIND as u8,
                    needed: take_hist_bytes(hist, start, end).saturating_mul(8),
                    capacity: byte_caps.get(PLANE_HIST_KIND).copied().unwrap_or(0),
                });
            }
        };
        // Byte caps cover the take footprint (first-short plane reported).
        let mut i = 0usize;
        while i < N_PLANES {
            let need = row_bytes(i, t_len).saturating_mul(batch_size);
            let cap = byte_caps.get(i).copied().unwrap_or(0);
            if need > cap {
                return Err(DriverError::BufferTooSmall {
                    plane: i as u8,
                    needed: need,
                    capacity: cap,
                });
            }
            i += 1;
        }
        for (plane, ptr) in ptrs.iter().enumerate() {
            if *ptr == 0 {
                return Err(DriverError::BufferTooSmall {
                    plane: plane as u8,
                    needed: 0,
                    capacity: 0,
                });
            }
        }
        // Row-exact fill from the take offset (fixed planes are contiguous
        // memcpys; history planes copy the per-row valid prefix + zero tail;
        // dora is memcpy-verbatim, never tail-touched).
        fill_take_into(&self.scratch, start, batch_size, t_len, ptrs, byte_caps)
            .map_err(fill_err)?;
        // Take envelope (take order; game-identity-free hash inputs only).
        let mut ids: Vec<String> = Vec::with_capacity(batch_size);
        let mut chosen_out: Vec<i64> = Vec::with_capacity(batch_size);
        let mut kinds: Vec<String> = Vec::with_capacity(batch_size);
        let mut k = start;
        while k < end {
            let meta = &self.rows_meta[k];
            let game_id = self.game_ids.get(meta.entry_idx as usize).ok_or_else(|| {
                DriverError::InvalidArg("driver game index out of range".to_string())
            })?;
            ids.push(format!("{}:d{:04}", game_id, meta.seq));
            chosen_out.push(meta.chosen);
            kinds.push(kind_for_chosen(meta.chosen).to_string());
            k += 1;
        }
        let classes: Vec<(String, u64)> = self
            .classes
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        let out = MicrobatchOut {
            rows: batch_size as u32,
            t_len: t_len as u32,
            ids,
            chosen: chosen_out,
            kinds,
            replayed: self.replayed,
            sim_replayed: self.sim_replayed,
            quarantined: self.quarantined,
            classes,
            offset: self.offset.saturating_add(batch_size as u64),
            dropped: self.dropped,
            microbatches: self.microbatches.saturating_add(1),
            total_rows: self.rows_meta.len() as u64,
            games_seen: self.games_seen,
        };
        // Commit (fill succeeded): advance the consumed prefix, then reclaim
        // whole consumed games past the bound (entries stay whole-game
        // aligned; at most one partial game strands).
        self.offset = self.offset.saturating_add(batch_size as u64);
        self.microbatches = self.microbatches.saturating_add(1);
        self.rows_out = self.rows_out.saturating_add(batch_size as u64);
        self.compact();
        Ok(out)
    }

    /// Drop whole consumed games once the prefix outgrows the bound.
    ///
    /// Logical counters (`offset` and everything derived) are untouched;
    /// only list storage is reclaimed. Mirrors `_compact` exactly.
    fn compact(&mut self) {
        let consumed = self.offset.saturating_sub(self.dropped);
        if consumed < self.compact_rows as u64 {
            return;
        }
        let mut remaining = consumed;
        let mut drop_games: usize = 0;
        let mut drop_rows: usize = 0;
        for entry in &self.entries {
            let count = u64::from(entry.rows);
            if count <= remaining {
                remaining = remaining.saturating_sub(count);
                drop_games = drop_games.saturating_add(1);
                drop_rows = drop_rows.saturating_add(count as usize);
            } else {
                break;
            }
        }
        if drop_games == 0 || drop_rows == 0 {
            return;
        }
        let mut hist_entries: usize = 0;
        let hist = self.scratch.hist_lens();
        let mut r = 0usize;
        while r < drop_rows {
            hist_entries = hist_entries.saturating_add(hist.get(r).copied().unwrap_or(0) as usize);
            r += 1;
        }
        self.scratch.drain_prefix(drop_rows, drop_games, hist_entries);
        self.rows_meta.drain(..drop_rows);
        self.entries.drain(..drop_games);
        self.game_ids.drain(..drop_games);
        // Rebase per-row entry indices after the front drain.
        for meta in &mut self.rows_meta {
            meta.entry_idx = meta.entry_idx.saturating_sub(drop_games as u32);
        }
        self.dropped = self.dropped.saturating_add(drop_rows as u64);
    }

    /// Fast-resume snapshot: whole-game entries + counters + row hash.
    ///
    /// Entries carry `(key, rows)`; `path`/`offset`/`split` rejoin
    /// Python-side (file-IO authority). Fails closed on drift (never a
    /// partial snapshot).
    pub fn snapshot(&self) -> Result<DriverSnapshot, DriverError> {
        self.fail_if_closed()?;
        if self.scratch.rows() != self.rows_meta.len() {
            return Err(DriverError::InvalidArg(format!(
                "buffer index drift: {} staged rows != {} buffered",
                self.scratch.rows(),
                self.rows_meta.len()
            )));
        }
        Ok(DriverSnapshot {
            entries: self
                .entries
                .iter()
                .map(|e| DriverEntry {
                    key: e.key.clone(),
                    rows: e.rows,
                })
                .collect(),
            offset: self.offset,
            dropped: self.dropped,
            microbatches: self.microbatches,
            replayed: self.replayed,
            sim_replayed: self.sim_replayed,
            quarantined: self.quarantined,
            classes: self
                .classes
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
            row_hash: window_hash(&self.entries, &self.rows_meta)?,
            total_rows: self.rows_meta.len() as u64,
        })
    }

    /// Cumulative counters (readable post-close at the bridge via stash).
    pub fn stats(&self) -> DriverStats {
        DriverStats {
            games_seen: self.games_seen,
            games_ok: self.games_ok,
            games_quarantined: self.quarantined,
            rows_in: self.rows_in,
            rows_out: self.rows_out,
            microbatches: self.microbatches,
        }
    }
}

/// Take history bytes in `[start, end)` (kind plane, pre-`t_len`).
fn take_hist_bytes(hist: &[u32], start: usize, end: usize) -> usize {
    let mut total: usize = 0;
    let mut r = start;
    while r < end {
        total = total.saturating_add(hist.get(r).copied().unwrap_or(0) as usize);
        r += 1;
    }
    total
}

/// Row-exact fill of take rows `[start, start+count)` into caller planes.
///
/// Same caller contract as [`Scratch::fill_pinned`] (flat row-major,
/// `t_len` history width, T-tail zero-clears history kind/mask ONLY), but
/// the source is an interior take window instead of the staged prefix.
/// Nothing drains here; the driver advances `offset`/compacts on success.
///
/// # Safety
/// `ptrs[i]` must be a valid writable allocation of at least
/// `byte_caps[i]` bytes for the borrow duration; the caller keeps both
/// buffers alive across the call.
fn fill_take_into(
    scratch: &Scratch,
    start: usize,
    count: usize,
    t_len: usize,
    ptrs: &[u64; N_PLANES],
    byte_caps: &[usize; N_PLANES],
) -> Result<(), FillError> {
    let planes = scratch.planes();
    let hist = scratch.hist_lens();
    // Fixed planes: single contiguous memcpy per plane (dora verbatim).
    let mut i = 0usize;
    while i < N_PLANES {
        if i == PLANE_HIST_KIND || i == PLANE_HIST_MASK {
            i += 1;
            continue;
        }
        let stride = ROW_BYTES_FIXED.get(i).copied().unwrap_or(0);
        let src_off = start.saturating_mul(stride);
        let need = count.saturating_mul(stride);
        let src = planes
            .get(i)
            .and_then(|p| p.get(src_off..src_off.saturating_add(need)))
            .ok_or(FillError::BufferTooSmall {
                plane: i as u8,
                needed_bytes: need,
                capacity_bytes: byte_caps.get(i).copied().unwrap_or(0),
            })?;
        let dst = unsafe { checked_slice(ptrs[i], need, byte_caps[i]) };
        let dst = dst.map_err(|e| match e {
            FillError::NullPointer { .. } => FillError::NullPointer { plane: i as u8 },
            FillError::BufferTooSmall { .. } => FillError::BufferTooSmall {
                plane: i as u8,
                needed_bytes: need,
                capacity_bytes: byte_caps[i],
            },
        })?;
        dst.copy_from_slice(src);
        i += 1;
    }
    // Source history offset: entries before `start`.
    let mut src_entries: usize = 0;
    let mut r = 0usize;
    while r < start {
        src_entries = src_entries.saturating_add(hist.get(r).copied().unwrap_or(0) as usize);
        r += 1;
    }
    // #4 kinds `[count, t_len]` i64: per-row valid prefix + zero tail.
    {
        let stride = t_len.saturating_mul(8);
        let need = count.saturating_mul(stride);
        let dst = unsafe { checked_slice(ptrs[PLANE_HIST_KIND], need, byte_caps[PLANE_HIST_KIND]) };
        let dst = dst.map_err(|e| match e {
            FillError::NullPointer { .. } => FillError::NullPointer {
                plane: PLANE_HIST_KIND as u8,
            },
            FillError::BufferTooSmall { .. } => FillError::BufferTooSmall {
                plane: PLANE_HIST_KIND as u8,
                needed_bytes: need,
                capacity_bytes: byte_caps[PLANE_HIST_KIND],
            },
        })?;
        let src = planes.get(PLANE_HIST_KIND).ok_or(FillError::BufferTooSmall {
            plane: PLANE_HIST_KIND as u8,
            needed_bytes: need,
            capacity_bytes: byte_caps[PLANE_HIST_KIND],
        })?;
        let mut r = 0usize;
        let mut off = src_entries;
        while r < count {
            let hl = hist.get(start.saturating_add(r)).copied().unwrap_or(0) as usize;
            let base = r.saturating_mul(stride);
            let src_off = off.saturating_mul(8);
            let src_len = hl.saturating_mul(8);
            let chunk = src
                .get(src_off..src_off.saturating_add(src_len))
                .ok_or(FillError::BufferTooSmall {
                    plane: PLANE_HIST_KIND as u8,
                    needed_bytes: need,
                    capacity_bytes: byte_caps[PLANE_HIST_KIND],
                })?;
            let slot = dst
                .get_mut(base..base.saturating_add(hl.saturating_mul(8)))
                .ok_or(FillError::BufferTooSmall {
                    plane: PLANE_HIST_KIND as u8,
                    needed_bytes: need,
                    capacity_bytes: byte_caps[PLANE_HIST_KIND],
                })?;
            slot.copy_from_slice(chunk);
            if let Some(tail) = dst.get_mut(base.saturating_add(hl.saturating_mul(8))..base.saturating_add(stride)) {
                tail.fill(0);
            }
            off = off.saturating_add(hl);
            r += 1;
        }
    }
    // #5 mask `[count, t_len]` bool: per-row ones prefix + zero tail.
    {
        let stride = t_len;
        let need = count.saturating_mul(stride);
        let dst = unsafe { checked_slice(ptrs[PLANE_HIST_MASK], need, byte_caps[PLANE_HIST_MASK]) };
        let dst = dst.map_err(|e| match e {
            FillError::NullPointer { .. } => FillError::NullPointer {
                plane: PLANE_HIST_MASK as u8,
            },
            FillError::BufferTooSmall { .. } => FillError::BufferTooSmall {
                plane: PLANE_HIST_MASK as u8,
                needed_bytes: need,
                capacity_bytes: byte_caps[PLANE_HIST_MASK],
            },
        })?;
        let src = planes.get(PLANE_HIST_MASK).ok_or(FillError::BufferTooSmall {
            plane: PLANE_HIST_MASK as u8,
            needed_bytes: need,
            capacity_bytes: byte_caps[PLANE_HIST_MASK],
        })?;
        let mut r = 0usize;
        let mut off = src_entries;
        while r < count {
            let hl = hist.get(start.saturating_add(r)).copied().unwrap_or(0) as usize;
            let base = r.saturating_mul(stride);
            let chunk = src
                .get(off..off.saturating_add(hl))
                .ok_or(FillError::BufferTooSmall {
                    plane: PLANE_HIST_MASK as u8,
                    needed_bytes: need,
                    capacity_bytes: byte_caps[PLANE_HIST_MASK],
                })?;
            let slot = dst
                .get_mut(base..base.saturating_add(hl))
                .ok_or(FillError::BufferTooSmall {
                    plane: PLANE_HIST_MASK as u8,
                    needed_bytes: need,
                    capacity_bytes: byte_caps[PLANE_HIST_MASK],
                })?;
            slot.copy_from_slice(chunk);
            if let Some(tail) = dst.get_mut(base.saturating_add(hl)..base.saturating_add(stride)) {
                tail.fill(0);
            }
            off = off.saturating_add(hl);
            r += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod driver_tests {
    use super::*;

    #[test]
    fn quarantine_class_pinned() {
        assert_eq!(
            quarantine_class("rust walk quarantined game 'parity-game-4': turn-order at event 5"),
            "rust walk quarantined game '<id>': turn-order at event 5"
        );
        assert_eq!(
            quarantine_class("rust walk quarantined game \"abc\" kyoku 12 seat 3"),
            "rust walk quarantined game '<id>' kyoku <n> seat <n>"
        );
        assert_eq!(
            quarantine_class("wall_tiles must carry 136 tiles, got 5"),
            "wall_tiles must carry 136 tiles, got 5"
        );
        assert_eq!(
            quarantine_class("tile \"5mr\" overused kyoku 2"),
            "tile '5mr' overused kyoku <n>"
        );
        assert_eq!(
            quarantine_class("wall game-abcdef1234 in splits"),
            "wall game-<id> in splits"
        );
    }

    #[test]
    fn kind_for_parity_ids_is_tsumogiri() {
        assert_eq!(kind_for_chosen(192), "tsumogiri");
        assert_eq!(kind_for_chosen(193), "tsumogiri");
        assert_eq!(kind_for_chosen(194), "tsumogiri");
        assert_eq!(kind_for_chosen(-1), "unknown");
        assert_eq!(kind_for_chosen(6792), "unknown");
    }

    #[test]
    fn window_hash_shape_and_drift() {
        let entries = vec![
            DriverEntry {
                key: "sha256:abc".to_string(),
                rows: 1,
            },
            DriverEntry {
                key: "sha256:def".to_string(),
                rows: 1,
            },
        ];
        let metas = vec![
            RowMeta {
                entry_idx: 0,
                seq: 0,
                chosen: 192,
            },
            RowMeta {
                entry_idx: 1,
                seq: 0,
                chosen: 193,
            },
        ];
        let hash = window_hash(&entries, &metas).unwrap();
        assert!(hash.starts_with("sha256:"));
        assert_eq!(hash.len(), 7 + 64);
        let bad = window_hash(&entries[..1], &metas).unwrap_err();
        assert!(matches!(bad, DriverError::InvalidArg(_)));
    }
}
