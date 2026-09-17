//! S4 fill — Scratch-once staging into caller pinned memory (`feed::fill`).
//!
//! Stage-then-commit staging: pick T BEFORE committing rows; whole-game
//! prefix commits only, remainder stays.
//!
//! Staging order (breaks the `t_len`/`rows` circle — the stager picks T BEFORE
//! committing rows):
//! 1. `walk` stages games into [`Scratch`] with history lens recorded
//!    ([`stage_one`] walks thread-local, [`Scratch::commit`] appends whole
//!    games only — a rejected game never touches shared state, so walk-error
//!    prefixes are discarded by construction, never rolled back).
//! 2. `t_len` = smallest bucket in [`T_BUCKETS`] `>=` max staged history len.
//! 3. `rows_cap` = min over planes of `byte_caps[i] / row_bytes(i, t_len)`.
//! 4. If the max history len exceeds every bucket ([`pick_t_len`] is `None`),
//!    fail `BufferTooSmall { plane: 4, .. }` BEFORE any commit (plane:4 rule).
//!
//! [`Scratch::fill_pinned`] is stage-then-commit: it commits the largest
//! whole-game prefix fitting `rows_cap`, drains exactly that prefix, and
//! leaves the remainder staged for the next call. A buffer that cannot fit
//! even the first staged game fails [`FillError::BufferTooSmall`] and drains
//! NOTHING (counters unmoved, [`Scratch`] byte-identical).
//!
//! Caller contract (bridge owns the call site):
//! - `ptrs` are pinned-memory base addresses, `byte_caps` are tensor byte
//!   sizes (`tensor.nbytes`, BYTES in). Each plane buffer is treated as FLAT
//!   row-major with stride [`row_bytes`]`(i, t_len)` — history planes MUST be
//!   shaped `[rows, t_len]` (or flat equivalents) for the `t_len` of that
//!   fill (see [`FillOut::t_len`]).
//! - T-tail `[valid..T)` zero-clears history kind/mask ONLY (`kind = 0`,
//!   `mask = 0`); `dora_indicators` is memcpy-verbatim on every row and NEVER
//!   touched by tail logic (G3). Rows beyond the committed prefix are
//!   entirely untouched.
//! - The whole fill runs detached under ONE guard on the bridge side
//!   (serial, never unlock-relock mid-fill); per-slot `slot_events`
//!   record-after-`.to()`s + `wait_event` is Python-side (bridge owns it).
//!
//! Parallelism: [`FeedPool`] is the SOLE hot rayon pool
//! (`ThreadPoolBuilder::new().num_threads(n).build()` + [`FeedPool::install`];
//! packager `main.rs:546` precedent). Work is per-game batches handed over a
//! bounded [`STAGE_CHANNEL_BOUND`]-deep channel (1 hop/game); the shard reuses
//! this handle, the bridge creates none.
//!
//! Feed-only deps: `crate::{gate, ingest, ledger, walk}` + `std::sync` +
//! `rayon`. NEVER `hydra-shard` (DAG one-way), NEVER pyo3/sha2-hot.

use std::sync::Arc;

use crate::gate::{QuarantineStub, SampleDecision, gate_game};
use crate::ingest::{FramedGame, frame_file, frame_spans};
use crate::ledger::{
    GameCtx, Ledger, N_WALK_PLANES, PLANE_HIST_KIND, PLANE_HIST_MASK, RowSink,
};
use crate::walk::walk_game;

// ---------------------------------------------------------------------------
// Plane geometry (canonical 26-plane order; Rust owns this table, Python NEVER duplicates it)
// ---------------------------------------------------------------------------

/// Hot plane count (canonical order #0-25: 0-12 hot replay + 13-25 row scalars).
/// MUST equal `ledger::N_WALK_PLANES` (that alias keeps the borrow type
/// literal-free; the const assert below pins the equality at compile time).
pub const N_PLANES: usize = 26;

/// Compile-time pin: `N_PLANES` and `N_WALK_PLANES` are the same width, so
/// staged `[Vec<u8>; N_PLANES]` views feed [`RowSink`] directly.
const _: [u8; N_PLANES] = [0; N_WALK_PLANES];
/// Per-plane FIXED row stride in bytes (dtypes match `schema.py`).
/// Slots #4/#5 are T-variable and read `0` here — use [`row_bytes`].
/// #12 packed = 128B `legal_ids[32] int32 LE` + 8B `legal_len int64 LE`.
/// #13/#14 `i64`; #15-21 `i32`; #22 `bool[4]`; #23 `i64[4]`; #24/#25 `bool`.
pub const ROW_BYTES_FIXED: [usize; N_PLANES] = [
    34, 34, 20, 16, 0, 0, 8, 8, 8, 8, 8, 32, 136, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 32, 1, 1,
];

/// Frozen history-length buckets (history planes are `[T]` memcpy).
pub const T_BUCKETS: [usize; 4] = [32, 64, 128, 256];

/// Row stride in bytes for `plane` at history length `t_len`:
/// #4 = `t_len * 8` (`history_event_kind[T] int64`),
/// #5 = `t_len` (`history_mask[T] bool`), else [`ROW_BYTES_FIXED`].
/// Out-of-range planes report `0` (fail-closed; [`Scratch::fill_pinned`]
/// only ever queries `0..N_PLANES`).
#[inline]
pub fn row_bytes(plane: usize, t_len: usize) -> usize {
    if plane == PLANE_HIST_KIND {
        t_len.saturating_mul(8)
    } else if plane == PLANE_HIST_MASK {
        t_len
    } else {
        ROW_BYTES_FIXED.get(plane).copied().unwrap_or(0)
    }
}

/// Smallest bucket in [`T_BUCKETS`] `>= max_hist`; `None` when the history
/// exceeds every bucket (caller fails plane:4 BEFORE any commit).
#[inline]
pub fn pick_t_len(max_hist: u32) -> Option<usize> {
    let h = max_hist as usize;
    let mut k = 0usize;
    while k < T_BUCKETS.len() {
        if T_BUCKETS[k] >= h {
            return Some(T_BUCKETS[k]);
        }
        k += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// Errors (byte units matching `byte_caps`; first-short plane reported)
// ---------------------------------------------------------------------------

/// Fill failure taxonomy: fail-closed, numeric, never a `String` hot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillError {
    /// Null caller pointer (`ptr == 0`). Always plane-attributed by
    /// [`Scratch::fill_pinned`]; raw [`checked_slice`] reports
    /// [`UNKNOWN_PLANE`].
    NullPointer {
        /// Failing plane index (canonical plane order #0-25).
        plane: u8,
    },
    /// Buffer cannot stage the commit. `needed_bytes` is the STAGED byte
    /// length of `plane`, `capacity_bytes` is `byte_caps[plane]` — both in
    /// bytes, matching the caller's `tensor.nbytes` units.
    BufferTooSmall {
        /// First-short plane index (canonical plane order #0-25; lowest index on ties).
        plane: u8,
        /// Staged bytes in `plane` (exact shortfall context).
        needed_bytes: usize,
        /// `byte_caps[plane]`.
        capacity_bytes: usize,
    },
}

/// Plane marker for errors raised without plane context (raw
/// [`checked_slice`]); [`Scratch::fill_pinned`] rewrites it to the real
/// index before returning.
pub const UNKNOWN_PLANE: u8 = u8::MAX;

/// Borrow a caller pinned range as `&mut [u8]` — fail-closed.
///
/// - `ptr == 0` → `Err(NullPointer)` (never a null deref, never silent).
/// - `byte_len > byte_cap` → `Err(BufferTooSmall)` (defense-in-depth;
///   [`Scratch::fill_pinned`] pre-verifies lengths, so this arm is
///   unreachable there and only fires on direct misuse).
///
/// # Safety
/// `ptr` must be a valid writable allocation of at least `byte_len` bytes
/// for the lifetime of the returned borrow; `byte_cap` must be the caller's
/// real capacity for that allocation.
pub unsafe fn checked_slice<'a>(
    ptr: u64,
    byte_len: usize,
    byte_cap: usize,
) -> Result<&'a mut [u8], FillError> {
    if ptr == 0 {
        return Err(FillError::NullPointer {
            plane: UNKNOWN_PLANE,
        });
    }
    if byte_len > byte_cap {
        return Err(FillError::BufferTooSmall {
            plane: UNKNOWN_PLANE,
            needed_bytes: byte_len,
            capacity_bytes: byte_cap,
        });
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, byte_len) })
}

// ---------------------------------------------------------------------------
// Staging types (per-game batches; walk errors never touch shared state)
// ---------------------------------------------------------------------------

/// One owned game payload ready for gate+walk on any worker thread.
/// `wall_digest` is cold-computed (feed FORBIDS sha2-hot); `None` is the
/// wall-less path (SIM mark, never invented).
#[derive(Clone, Debug)]
pub struct StageJob {
    /// Stable game index (sink join key: `base_game_idx + file position`).
    pub game_idx: u32,
    /// File-stable identity for lineage joins (ingest `object_id`).
    /// Independent of `game_idx`: two files may stage identical index ranges
    /// while their `(object_id, game_idx)` keys stay disjoint.
    pub object_id: u32,
    /// Raw payload bytes (full game, trailing newline included).
    pub bytes: Vec<u8>,
    /// Cold-computed wall digest allocation, borrowed (never cloned) by the
    /// walk context for the duration of [`stage_one`].
    pub wall_digest: Option<Arc<str>>,
}

#[derive(Debug)]
pub struct StagedGame {
    /// Stable game index (sink join key).
    pub game_idx: u32,
    /// File-stable identity, propagated from the framed game (lineage key
    /// with `game_idx`).
    pub object_id: u32,
    /// Staged plane bytes in canonical plane order (row-major, valid prefixes only).
    pub planes: [Vec<u8>; N_PLANES],
    /// Per-row history lengths (breaks the `t_len`/`rows` circle).
    pub hist_lens: Vec<u32>,
    /// Staged row count.
    pub rows: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagedReject {
    /// `(game_idx, event_idx, reason)` stub for the sink / lineage leg.
    pub stub: QuarantineStub,
}

/// File-stage receipt: seq advance + committed lineage keys + rejects.
#[derive(Debug, Default)]
pub struct FileStageOut {
    /// Advance for the caller's `game_seq` base: ALWAYS the chunk count
    /// (ok or err) — never 1, never ok-only.
    pub advanced: u32,
    /// `(object_id, game_idx)` lineage keys of committed games, file order.
    /// Disjoint across files whenever `object_id` is file-stable, even when
    /// `game_idx` ranges overlap.
    pub staged: Vec<(u32, u32)>,
    /// Per-game rejects in file order (positions keyed via their stubs).
    pub rejects: Vec<StagedReject>,
}

/// Frame + gate + walk one game into thread-local planes.
///
/// Pure over `job`: safe on any pool worker, touches no shared state, so a
/// rejection discards ONLY this game's thread-local prefix (stage-then-commit
/// by construction — [`Scratch`] only ever receives whole `Ok` games via
/// [`Scratch::commit`]).
pub fn stage_one(job: &StageJob) -> Result<StagedGame, StagedReject> {
    let mut arena = Vec::new();
    let game = match frame_spans(&mut arena, &job.bytes, job.object_id, job.game_idx) {
        Ok(g) => g,
        Err(e) => return Err(StagedReject { stub: e.stub() }),
    };
    gate_and_walk(&game, job.wall_digest.clone())
}

/// Splice an overriding wall into the first event of raw game bytes.
///
/// The game-pull training path binds external (S7) walls per game while the
/// bytes on the wire carry whatever the log embedded. Re-serializing every
/// event in Python to swap one field costs more than the walk itself, so the
/// swap happens here: parse the first line, set `"wall"`, rejoin. The tail
/// stays byte-exact and output ends with exactly one trailing `\n`, matching
/// the Python rebuild it replaces (which emits `head + "\n" + body + "\n"`).
/// Unparseable heads (or non-objects) pass through so `stage_one` rejects
/// them with its own reason — strict decode upstream guarantees parseable
/// heads on the production path, so the fallback never fires there.
pub fn inject_wall(events: &[u8], wall: &[u32]) -> Vec<u8> {
    let (head, rest) = match events.iter().position(|&b| b == b'\n') {
        Some(i) => (&events[..i], &events[i + 1..]),
        None => (events, &[][..]),
    };
    let mut out = match serde_json::from_slice::<serde_json::Value>(head) {
        Ok(mut v) => match v.as_object_mut() {
            Some(o) => {
                o.insert("wall".to_string(), serde_json::Value::from(wall.to_vec()));
                serde_json::to_vec(&v).unwrap_or_else(|_| head.to_vec())
            }
            None => head.to_vec(),
        },
        Err(_) => head.to_vec(),
    };
    out.push(b'\n');
    out.extend_from_slice(rest);
    if !out.ends_with(b"\n") {
        out.push(b'\n');
    }
    out
}

/// Gate + walk one framed game into thread-local planes (shared core behind
/// [`stage_one`] and [`Scratch::stage_file`]).
///
/// Pure over `game` (+ an owned digest refcount): safe on any pool worker,
/// touches no shared state. `wall_digest` is borrowed (never cloned out)
/// by the walk context for the call duration; the owned `Arc` keeps the
/// allocation alive exactly like the caller's `StageJob` does.
fn gate_and_walk(
    game: &FramedGame<'_>,
    wall_digest: Option<Arc<str>>,
) -> Result<StagedGame, StagedReject> {
    let verdict = match gate_game(game.game_idx, &game.events) {
        Ok(v) => v,
        Err(e) => return Err(StagedReject { stub: e.stub() }),
    };
    match verdict {
        SampleDecision::Sample => {}
        SampleDecision::SkipQuarantine(stub) => return Err(StagedReject { stub }),
    }
    // Walled regime derives from CONTENT (the framed 136-permutation wall),
    // never from caller plumbing: the bridge stages raw bytes and never
    // computes digests, so a caller-None digest with a wall-bearing game
    // must still walk unfolded (wall oracle reports true takes). The marker
    // is walledness only — row derivations bind real digests cold-side.
    const WALL_CONTENT_MARKER: &str = "wall-content-present";
    let wall_view: Option<&str> = wall_digest.as_deref().or(if game.wall.is_some() {
        Some(WALL_CONTENT_MARKER)
    } else {
        None
    });
    let ctx = GameCtx {
        game_idx: game.game_idx,
        wall_digest: wall_view,
        digest_arc: None,
    };
    let mut ledger = Ledger::new();
    let mut planes: [Vec<u8>; N_PLANES] = Default::default();
    let mut hist_lens: Vec<u32> = Vec::new();
    let rows = {
        let mut sink = RowSink::new(&mut planes, &mut hist_lens);
        match walk_game(&ctx, &game.events, &mut ledger, &mut sink) {
            Ok(_) => sink.rows,
            Err(w) => {
                return Err(StagedReject {
                    stub: QuarantineStub::new(w.game_idx, w.event_idx as u32, w.reason),
                });
            }
        }
    };
    Ok(StagedGame {
        game_idx: game.game_idx,
        object_id: game.object_id,
        planes,
        hist_lens,
        rows,
    })
}
// ---------------------------------------------------------------------------
// Scratch: alloc-once staging store (len=0 on reset, caps kept)
// ---------------------------------------------------------------------------

/// Alloc-once staging store: concatenated staged rows per plane + game
/// boundaries. Buffers grow once (steady-state zero-alloc); [`reset`]
/// truncates lengths and keeps capacities.
///
/// [`reset`]: Scratch::reset
#[derive(Debug, Default)]
pub struct Scratch {
    /// Staged plane bytes in canonical plane order (row-major, valid prefixes only).
    planes: [Vec<u8>; N_PLANES],
    /// Per-row history lengths, one entry per staged row.
    hist_lens: Vec<u32>,
    /// Staged row count.
    rows: usize,
    /// Staged game count.
    games: usize,
    /// Start row of each staged game (`len == games`; game `g` covers
    /// `[game_starts[g], game_starts[g+1])`, last game ends at `rows`).
    game_starts: Vec<u32>,
    /// Max staged history length (drives [`pick_t_len`]).
    max_hist: u32,
}

impl Scratch {
    /// Empty staging store (no allocation; buffers grow on first commit).
    pub fn new() -> Self {
        Self::default()
    }

    /// Truncate all planes/lens/boundaries (`len = 0`, capacities kept).
    /// T-tail semantics live in [`Scratch::fill_pinned`], not here: reset
    /// clears every plane equally and touches no caller memory.
    pub fn reset(&mut self) {
        let mut i = 0usize;
        while i < N_PLANES {
            self.planes[i].clear();
            i += 1;
        }
        self.hist_lens.clear();
        self.game_starts.clear();
        self.rows = 0;
        self.games = 0;
        self.max_hist = 0;
    }

    /// Staged row count.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Staged game count.
    #[inline]
    pub fn games(&self) -> usize {
        self.games
    }

    /// Max staged history length.
    #[inline]
    pub fn max_hist(&self) -> u32 {
        self.max_hist
    }

    /// Per-row history lengths (staged order).
    #[inline]
    pub fn hist_lens(&self) -> &[u32] {
        &self.hist_lens
    }

    /// Staged plane views in canonical plane order (valid prefixes only).
    #[inline]
    pub fn planes(&self) -> &[Vec<u8>; N_PLANES] {
        &self.planes
    }

    /// Commit one whole walked game (appends rows, records the boundary).
    /// Only `Ok` games reach here — see [`stage_one`].
    pub fn commit(&mut self, g: &StagedGame) {
        self.game_starts.push(self.rows as u32);
        let mut i = 0usize;
        while i < N_PLANES {
            self.planes[i].extend_from_slice(&g.planes[i]);
            i += 1;
        }
        self.hist_lens.extend_from_slice(&g.hist_lens);
        self.rows += g.rows as usize;
        self.games += 1;
        let mut k = 0usize;
        while k < g.hist_lens.len() {
            if g.hist_lens[k] > self.max_hist {
                self.max_hist = g.hist_lens[k];
            }
            k += 1;
        }
    }

    /// Commit walked games in order (boundaries preserved).
    pub fn commit_batch(&mut self, games: &[StagedGame]) {
        let mut k = 0usize;
        while k < games.len() {
            self.commit(&games[k]);
            k += 1;
        }
    }

    /// Stage one whole file: split via `frame_file`, then gate/walk/commit
    /// each game in file order (wall-less path).
    ///
    /// Identity contract: `object_id` is the file-stable lineage identity
    /// (ingest `object_id`, e.g. the file's manifest slot) while
    /// `base_game_idx` keys positions (`game_idx =
    /// base.saturating_add(position)`). The two MUST differ across files
    /// that share `game_idx` ranges — doubling `base_game_idx` as
    /// `object_id` conflates file identity and breaks lineage joins. The
    /// single-game [`stage_one`] path is untouched (same [`gate_and_walk`]
    /// core, one chunk, no special case).
    pub fn stage_file(
        &mut self,
        bytes: &[u8],
        object_id: u32,
        base_game_idx: u32,
    ) -> FileStageOut {
        let mut arena = Vec::new();
        let (results, count) = frame_file(&mut arena, bytes, object_id, base_game_idx);
        let mut out = FileStageOut::default();
        for r in results {
            match r {
                Err(e) => out.rejects.push(StagedReject { stub: e.stub() }),
                Ok(game) => match gate_and_walk(&game, None) {
                    Ok(g) => {
                        out.staged.push((g.object_id, g.game_idx));
                        self.commit(&g);
                    }
                    Err(e) => out.rejects.push(e),
                },
            }
        }
        out.advanced = count;
        out
    }

    /// Roll back to the first `rows` staged rows (whole-call rollback).
    ///
    /// Keeps `planes`/`hist_lens`/`game_starts` consistent: planes truncate
    /// to the kept footprint, `hist_lens` truncates, games ending after the
    /// kept prefix are dropped, `max_hist` is rescanned. Capacities are kept
    /// for reuse. A mid-game `rows` rounds DOWN to the prior game end (so
    /// the whole-game invariant holds even on misuse); `0` drops everything.
    pub fn truncate_rows(&mut self, rows: u32) {
        let want = (rows as usize).min(self.rows);
        // Round down to the largest game end <= want (total even on misuse).
        let mut keep = 0usize;
        if want >= self.rows {
            keep = self.rows;
        } else {
            let mut g = 0usize;
            while g < self.games {
                let end = if g + 1 < self.games {
                    self.game_starts[g + 1] as usize
                } else {
                    self.rows
                };
                if end <= want {
                    keep = end;
                } else {
                    break;
                }
                g += 1;
            }
        }
        let mut hist = 0usize;
        let mut r = 0usize;
        while r < keep {
            hist += self.hist_lens[r] as usize;
            r += 1;
        }
        let mut i = 0usize;
        while i < N_PLANES {
            let len = if i == PLANE_HIST_KIND {
                hist * 8
            } else if i == PLANE_HIST_MASK {
                hist
            } else {
                keep * ROW_BYTES_FIXED[i]
            };
            self.planes[i].truncate(len);
            i += 1;
        }
        self.hist_lens.truncate(keep);
        let mut g = 0usize;
        while g < self.game_starts.len() && (self.game_starts[g] as usize) < keep {
            g += 1;
        }
        self.game_starts.truncate(g);
        self.rows = keep;
        self.games = g;
        let mut m = 0u32;
        let mut r = 0usize;
        while r < self.hist_lens.len() {
            if self.hist_lens[r] > m {
                m = self.hist_lens[r];
            }
            r += 1;
        }
        self.max_hist = m;
    }

    /// Fill caller pinned memory from the stage (stage-then-commit).
    ///
    /// Commits the largest whole-game prefix fitting
    /// `rows_cap = min(byte_caps[i] / row_bytes(i, t_len))`, memcpys staged
    /// bytes, zero-clears the per-row T-tail on history planes ONLY, drains
    /// exactly the committed prefix, and leaves the remainder staged.
    /// `dora_indicators` is copied verbatim and NEVER tail-touched (G3);
    /// rows beyond the committed prefix are entirely untouched.
    pub fn fill_pinned(
        &mut self,
        ptrs: &[u64; N_PLANES],
        byte_caps: &[usize; N_PLANES],
    ) -> Result<FillOut, FillError> {
        let staged = self.rows;
        if staged == 0 {
            return Ok(FillOut {
                rows: 0,
                games_consumed: 0,
                t_len: T_BUCKETS[0],
            });
        }
        // (b) T BEFORE rows. History beyond every bucket fails plane:4
        // BEFORE any commit (needed = staged plane-4 bytes, byte units).
        let t_len = match pick_t_len(self.max_hist) {
            Some(t) => t,
            None => {
                return Err(FillError::BufferTooSmall {
                    plane: PLANE_HIST_KIND as u8,
                    needed_bytes: self.planes[PLANE_HIST_KIND].len(),
                    capacity_bytes: byte_caps[PLANE_HIST_KIND],
                });
            }
        };
        // (c) rows = min over planes of caps/stride; track first-short.
        let mut rows_cap = usize::MAX;
        let mut short = 0usize;
        let mut i = 0usize;
        while i < N_PLANES {
            let cap_rows = byte_caps[i] / row_bytes(i, t_len);
            if cap_rows < rows_cap {
                rows_cap = cap_rows;
                short = i;
            }
            i += 1;
        }
        let target = if rows_cap < staged { rows_cap } else { staged };
        // Commit whole games only: largest prefix with game end <= target.
        let mut commit_games = 0usize;
        while commit_games < self.games {
            let end = if commit_games + 1 < self.games {
                self.game_starts[commit_games + 1] as usize
            } else {
                staged
            };
            if end <= target {
                commit_games += 1;
            } else {
                break;
            }
        }
        // Small buffer drains NOTHING (fail-closed, state untouched).
        if commit_games == 0 {
            return Err(FillError::BufferTooSmall {
                plane: short as u8,
                needed_bytes: self.planes[short].len(),
                capacity_bytes: byte_caps[short],
            });
        }
        let commit_rows = if commit_games == self.games {
            staged
        } else {
            self.game_starts[commit_games] as usize
        };
        // Committed history entries (planes #4/#5 source lengths).
        let mut hist_entries = 0usize;
        let mut r = 0usize;
        while r < commit_rows {
            hist_entries += self.hist_lens[r] as usize;
            r += 1;
        }
        // Fixed planes: single contiguous memcpy (dora verbatim, G3).
        let mut i = 0usize;
        while i < N_PLANES {
            if i == PLANE_HIST_KIND || i == PLANE_HIST_MASK {
                i += 1;
                continue;
            }
            let need = commit_rows * ROW_BYTES_FIXED[i];
            // SAFETY: `ptrs[i]`/`byte_caps[i]` are caller-pinned per the
            // caller-contract block above; `need` is the verified footprint.
            let dst = unsafe { checked_slice(ptrs[i], need, byte_caps[i]) };
            let dst = dst.map_err(|e| self.plane_err(e, i, need))?;
            dst.copy_from_slice(&self.planes[i][..need]);
            i += 1;
        }
        // #4 kinds [commit_rows, t_len] i64: per-row valid prefix + zero tail.
        {
            let stride = t_len * 8;
            let need = commit_rows * stride;
            // SAFETY: caller-pinned plane buffer (see caller contract);
            // `need = commit_rows * stride` is the verified footprint, and
            // the borrow ends before `drain_prefix` below.
            let dst = unsafe { checked_slice(ptrs[PLANE_HIST_KIND], need, byte_caps[PLANE_HIST_KIND]) };
            let dst = dst.map_err(|e| self.plane_err(e, PLANE_HIST_KIND, need))?;
            let src = &self.planes[PLANE_HIST_KIND];
            let mut r = 0usize;
            let mut off = 0usize;
            while r < commit_rows {
                let hl = self.hist_lens[r] as usize;
                let base = r * stride;
                dst[base..base + hl * 8].copy_from_slice(&src[off * 8..off * 8 + hl * 8]);
                dst[base + hl * 8..base + stride].fill(0);
                off += hl;
                r += 1;
            }
        }
        // #5 mask [commit_rows, t_len] bool: per-row ones prefix + zero tail.
        {
            let stride = t_len;
            let need = commit_rows * stride;
            // SAFETY: caller-pinned plane buffer (see caller contract);
            // `need = commit_rows * stride` is the verified footprint, and
            // the borrow ends before `drain_prefix` below.
            let dst = unsafe { checked_slice(ptrs[PLANE_HIST_MASK], need, byte_caps[PLANE_HIST_MASK]) };
            let dst = dst.map_err(|e| self.plane_err(e, PLANE_HIST_MASK, need))?;
            let src = &self.planes[PLANE_HIST_MASK];
            let mut r = 0usize;
            let mut off = 0usize;
            while r < commit_rows {
                let hl = self.hist_lens[r] as usize;
                let base = r * stride;
                dst[base..base + hl].copy_from_slice(&src[off..off + hl]);
                dst[base + hl..base + stride].fill(0);
                off += hl;
                r += 1;
            }
        }
        self.drain_prefix(commit_rows, commit_games, hist_entries);
        Ok(FillOut {
            rows: commit_rows as u32,
            games_consumed: commit_games as u32,
            t_len,
        })
    }

    /// Attach the real plane index to a [`checked_slice`] error raised for
    /// plane `i` (`need` echoes the verified footprint on the unreachable
    /// length arm; the null arm carries the attribution the bridge needs).
    fn plane_err(&self, e: FillError, i: usize, need: usize) -> FillError {
        match e {
            FillError::NullPointer { .. } => FillError::NullPointer { plane: i as u8 },
            FillError::BufferTooSmall {
                needed_bytes,
                capacity_bytes,
                ..
            } => {
                let _ = needed_bytes;
                FillError::BufferTooSmall {
                    plane: i as u8,
                    needed_bytes: need,
                    capacity_bytes,
                }
            }
        }
    }

    /// Remove the committed prefix in place (capacities kept for reuse).
    /// Driver compact path: whole consumed games only, boundaries stay aligned.
    pub(crate) fn drain_prefix(&mut self, commit_rows: usize, commit_games: usize, hist_entries: usize) {
        let mut i = 0usize;
        while i < N_PLANES {
            let consumed = if i == PLANE_HIST_KIND {
                hist_entries * 8
            } else if i == PLANE_HIST_MASK {
                hist_entries
            } else {
                commit_rows * ROW_BYTES_FIXED[i]
            };
            let v = &mut self.planes[i];
            let len = v.len();
            v.copy_within(consumed..len, 0);
            v.truncate(len - consumed);
            i += 1;
        }
        self.hist_lens.drain(..commit_rows);
        self.game_starts.drain(..commit_games);
        let mut g = 0usize;
        while g < self.game_starts.len() {
            self.game_starts[g] -= commit_rows as u32;
            g += 1;
        }
        self.rows -= commit_rows;
        self.games -= commit_games;
        let mut m = 0u32;
        let mut r = 0usize;
        while r < self.hist_lens.len() {
            if self.hist_lens[r] > m {
                m = self.hist_lens[r];
            }
            r += 1;
        }
        self.max_hist = m;
    }
}

// ---------------------------------------------------------------------------
// Fill receipt
// ---------------------------------------------------------------------------

/// Committed-fill receipt (bridge maps this 1:1 to `PyFill`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FillOut {
    /// Committed rows (whole games only; remainder stays staged).
    pub rows: u32,
    /// Fully consumed staged games (stream cursor advance).
    pub games_consumed: u32,
    /// History bucket used for this fill (`32 | 64 | 128 | 256`).
    pub t_len: usize,
}

// ---------------------------------------------------------------------------
// SOLE hot pool + per-game batch staging over a bounded channel
// ---------------------------------------------------------------------------

/// Worker-thread clamp for [`FeedPool::build`]: at least 1, at most the
/// machine parallelism (hydra1 `normalize_worker_threads` idiom).
pub fn normalize_worker_threads(requested: usize) -> usize {
    let avail = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let cap = if avail < 1 { 1 } else { avail };
    if requested < 1 {
        1
    } else if requested > cap {
        cap
    } else {
        requested
    }
}

/// The SOLE hot rayon pool. Built once at stream open via
/// `ThreadPoolBuilder::new().num_threads(n).build()`; driven via
/// [`install`](FeedPool::install). The shard reuses this handle; the bridge
/// creates none.
#[derive(Debug)]
pub struct FeedPool {
    pool: rayon::ThreadPool,
}

impl FeedPool {
    /// Build the pool (thread count normalized via
    /// [`normalize_worker_threads`]).
    pub fn build(threads: usize) -> Result<Self, rayon::ThreadPoolBuildError> {
        let n = normalize_worker_threads(threads);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build()?;
        Ok(Self { pool })
    }

    /// Run `op` on this pool (rayon `install`: current thread joins the
    /// pool's workers for nested parallelism).
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(op)
    }
}

/// Bound for the stage handoff channel (1 hop/game backpressure).
pub const STAGE_CHANNEL_BOUND: usize = 128;

/// Walk `jobs` on `pool` (per-game batches) and return ordered outcomes.
///
/// [`Scratch::commit_batch`] in the returned order.
pub fn stage_batch(pool: &FeedPool, jobs: &[StageJob]) -> (Vec<StagedGame>, Vec<StagedReject>) {
    use rayon::prelude::*;

    let n = jobs.len();
    let (tx, rx) =
        std::sync::mpsc::sync_channel::<(usize, Result<StagedGame, StagedReject>)>(STAGE_CHANNEL_BOUND);
    let mut slots: Vec<Option<Result<StagedGame, StagedReject>>> = Vec::with_capacity(n);
    slots.resize_with(n, || None);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            pool.install(|| {
                jobs.par_iter().enumerate().for_each(|(idx, job)| {
                    let _ = tx.send((idx, stage_one(job)));
                });
            });
        });
        let mut received = 0usize;
        while received < n {
            match rx.recv() {
                Ok((idx, out)) => {
                    if idx < n && slots[idx].is_none() {
                        slots[idx] = Some(out);
                        received += 1;
                    }
                }
                Err(_) => break,
            }
        }
    });
    let mut games = Vec::new();
    let mut rejects = Vec::new();
    for slot in slots {
        match slot {
            Some(Ok(g)) => games.push(g),
            Some(Err(r)) => rejects.push(r),
            None => {}
        }
    }
    (games, rejects)
}

// ---------------------------------------------------------------------------
// Tests (FillBuilder gates: exact fill, plane attribution, drains-nothing)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod fill_tests {
    use super::*;
    use crate::ledger::{PLANE_ACTOR, PLANE_CHOSEN, PLANE_CONCEALED, PLANE_DORA, PLANE_LEGAL};

    const TEHAIS: [&[&str]; 4] = [
        &[
            "1m", "1m", "1m", "1m", "5mr", "5m", "5m", "5m", "9m", "9m", "9m", "9m", "4p",
        ],
        &[
            "2m", "2m", "2m", "2m", "6m", "6m", "6m", "6m", "1p", "1p", "1p", "1p", "4p",
        ],
        &[
            "3m", "3m", "3m", "3m", "7m", "7m", "7m", "7m", "2p", "2p", "2p", "2p", "4p",
        ],
        &[
            "4m", "4m", "4m", "4m", "8m", "8m", "8m", "8m", "3p", "3p", "3p", "3p", "4p",
        ],
    ];

    fn kyoku_line(tehais: &[&[&str]], oya: u8) -> String {
        let mut seats = Vec::new();
        for hand in tehais {
            let tiles: Vec<String> = hand.iter().map(|t| format!("{t:?}")).collect();
            seats.push(format!("[{}]", tiles.join(",")));
        }
        format!(
            "{{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"dora_marker\":\"F\",\"honba\":0,\"kyoku\":1,\"kyotaku\":0,\"oya\":{oya},\"scores\":[25000,25000,25000,25000],\"tehais\":[{}]}}",
            seats.join(",")
        )
    }

    fn tsumo_line(actor: u8, pai: &str) -> String {
        format!("{{\"type\":\"tsumo\",\"actor\":{actor},\"pai\":{pai:?}}}")
    }

    fn dahai_line(actor: u8, pai: &str) -> String {
        format!("{{\"type\":\"dahai\",\"actor\":{actor},\"pai\":{pai:?},\"tsumogiri\":true}}")
    }

    /// F1 shape (wall-less): 5 tsumogiri discards, no hora.
    fn game_text(pais: &[&str]) -> Vec<u8> {
        let mut lines = vec![
            "{\"type\":\"start_game\"}".to_string(),
            kyoku_line(&TEHAIS, 0),
        ];
        for (k, pai) in pais.iter().enumerate() {
            let actor = (k % 4) as u8;
            lines.push(tsumo_line(actor, pai));
            lines.push(dahai_line(actor, pai));
        }
        lines.push("{\"type\":\"end_game\"}".to_string());
        let mut out = lines.join("\n");
        out.push('\n');
        out.into_bytes()
    }
    fn stage_text(game_idx: u32, pais: &[&str]) -> StagedGame {
        let job = StageJob {
            game_idx,
            object_id: 7,
            bytes: game_text(pais),
            wall_digest: None,
        };
        stage_one(&job).expect("stage ok")
    }

    fn rd_i64(plane: &[u8], row: usize) -> i64 {
        let o = row * 8;
        i64::from_le_bytes(plane[o..o + 8].try_into().unwrap())
    }

    fn rd_dora(plane: &[u8], row: usize, slot: usize) -> i32 {
        let o = row * 20 + slot * 4;
        i32::from_le_bytes(plane[o..o + 4].try_into().unwrap())
    }

    #[test]
    fn inject_wall_sets_first_line_tail_exact() {
        let text = game_text(&["5pr", "5p", "5p", "5p", "6p"]);
        let wall: Vec<u32> = (0..136).collect();
        let out = inject_wall(&text, &wall);
        assert!(out.ends_with(b"\n"));
        assert!(!out.ends_with(b"\n\n"));
        // Tail past the first line is byte-exact.
        let first_nl = text.iter().position(|&b| b == b'\n').unwrap();
        let out_nl = out.iter().position(|&b| b == b'\n').unwrap();
        assert_eq!(&out[out_nl + 1..], &text[first_nl + 1..]);
        // First line parses with the bound wall.
        let head: serde_json::Value = serde_json::from_slice(&out[..out_nl]).unwrap();
        let got: Vec<u32> = serde_json::from_value(head["wall"].clone()).unwrap();
        assert_eq!(got, wall);
    }

    #[test]
    fn inject_wall_binds_and_stages() {
        // The bound wall takes effect: walled decisions differ from the
        // wall-less walk (draws unfold from the wall), while the row count
        // holds. If injection silently dropped the wall, chosen would equal
        // the wall-less golden.
        let text = game_text(&["5pr", "5p", "5p", "5p", "6p"]);
        let wall: Vec<u32> = (0..136).collect();
        let job = StageJob {
            game_idx: 3,
            object_id: 7,
            bytes: inject_wall(&text, &wall),
            wall_digest: None,
        };
        let g = stage_one(&job).expect("walled stages");
        let plain = stage_text(3, &["5pr", "5p", "5p", "5p", "6p"]);
        assert_eq!(g.rows, 5);
        assert_eq!(g.rows, plain.rows);
        assert_ne!(g.planes[6], plain.planes[6]);
    }

    #[test]
    fn inject_wall_garbage_passes_through() {
        // Unparseable heads pass through so stage_one rejects them with its
        // own reason (decode upstream guarantees parseable heads, so this
        // only fires on direct bridge misuse).
        let bad = b"not json at all\n".to_vec();
        assert_eq!(inject_wall(&bad, &[1, 2, 3]), bad);
        let no_nl = b"{\"type\":\"start_game\"}".to_vec();
        let out = inject_wall(&no_nl, &[9]);
        assert!(out.ends_with(b"\n"));
        let head: serde_json::Value =
            serde_json::from_slice(&out[..out.len() - 1]).unwrap();
        assert_eq!(head["wall"], serde_json::Value::from(vec![9u32]));
    }

    /// Caller buffers + caps for `rows` staged rows at `t_len`, prefilled
    /// with `0xAB` past the committed footprint (beyond-rows untouched).
    struct CallerBufs {
        bufs: [Vec<u8>; N_PLANES],
        caps: [usize; N_PLANES],
    }

    fn caller_bufs(rows: usize, t_len: usize) -> CallerBufs {
        let mut bufs: [Vec<u8>; N_PLANES] = Default::default();
        let mut caps = [0usize; N_PLANES];
        let mut i = 0usize;
        while i < N_PLANES {
            let cap = rows * row_bytes(i, t_len);
            caps[i] = cap;
            bufs[i] = vec![0xABu8; cap];
            i += 1;
        }
        CallerBufs { bufs, caps }
    }

    fn ptrs_of(c: &CallerBufs) -> [u64; N_PLANES] {
        let mut ptrs = [0u64; N_PLANES];
        let mut i = 0usize;
        while i < N_PLANES {
            ptrs[i] = c.bufs[i].as_ptr() as u64;
            i += 1;
        }
        ptrs
    }

    #[test]
    fn row_bytes_table_matches_plan() {
        assert_eq!(
            ROW_BYTES_FIXED,
            [34, 34, 20, 16, 0, 0, 8, 8, 8, 8, 8, 32, 136, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 32, 1, 1]
        );
        assert_eq!(N_PLANES, 26);
        assert_eq!(row_bytes(0, 64), 34);
        assert_eq!(row_bytes(2, 64), 20);
        assert_eq!(row_bytes(4, 64), 512);
        assert_eq!(row_bytes(5, 64), 64);
        assert_eq!(row_bytes(12, 256), 136);
        assert_eq!(pick_t_len(0), Some(32));
        assert_eq!(pick_t_len(32), Some(32));
        assert_eq!(pick_t_len(33), Some(64));
        assert_eq!(pick_t_len(200), Some(256));
        assert_eq!(pick_t_len(256), Some(256));
        assert_eq!(pick_t_len(257), None);
    }

    #[test]
    fn checked_slice_fail_closed() {
        // Null pointer fails (never a null deref).
        let err = unsafe { checked_slice(0, 8, 8).unwrap_err() };
        assert_eq!(err, FillError::NullPointer { plane: UNKNOWN_PLANE });
        // Over-length fails before any borrow.
        let mut backing = vec![0u8; 8];
        let ptr = backing.as_mut_ptr() as u64;
        let err = unsafe { checked_slice(ptr, 9, 8).unwrap_err() };
        assert_eq!(
            err,
            FillError::BufferTooSmall {
                plane: UNKNOWN_PLANE,
                needed_bytes: 9,
                capacity_bytes: 8,
            }
        );
        // Exact borrow writes through.
        let dst = unsafe { checked_slice(ptr, 8, 8).unwrap() };
        dst.copy_from_slice(&[7u8; 8]);
        assert_eq!(backing, vec![7u8; 8]);
    }

    #[test]
    fn fill_exact_f1_prefix_tail_dora() {
        // F1: 5 tsumogiri discards; golden chosen ids from the walk oracle.
        let g = stage_text(0, &["5pr", "5p", "5p", "5p", "6p"]);
        assert_eq!(g.rows, 5);
        let mut s = Scratch::new();
        s.commit(&g);
        assert_eq!(s.rows(), 5);
        assert_eq!(s.games(), 1);

        let t_len = pick_t_len(s.max_hist()).unwrap();
        assert_eq!(t_len, 32);
        // Oversized caps: short final batch leaves beyond-rows untouched.
        let c = caller_bufs(8, t_len);
        let ptrs = ptrs_of(&c);
        let out = s.fill_pinned(&ptrs, &c.caps).unwrap();
        assert_eq!(
            out,
            FillOut {
                rows: 5,
                games_consumed: 1,
                t_len: 32,
            }
        );
        assert_eq!(s.rows(), 0);
        assert_eq!(s.games(), 0);

        // Valid prefix exact on every plane (vs staged bytes).
        let staged = &g.planes;
        let mut i = 0usize;
        while i < N_PLANES {
            if i == PLANE_HIST_KIND || i == PLANE_HIST_MASK {
                i += 1;
                continue;
            }
            let valid = 5 * ROW_BYTES_FIXED[i];
            assert_eq!(&c.bufs[i][..valid], &staged[i][..], "plane {i}");
            i += 1;
        }
        // T-tail zeros on history ONLY; valid prefix intact.
        let mut off = 0usize;
        for r in 0..5usize {
            let hl = g.hist_lens[r] as usize;
            let kb = r * 32 * 8;
            assert_eq!(&c.bufs[4][kb..kb + hl * 8], &staged[4][off * 8..off * 8 + hl * 8]);
            assert!(c.bufs[4][kb + hl * 8..kb + 32 * 8].iter().all(|&b| b == 0));
            let mb = r * 32;
            assert_eq!(&c.bufs[5][mb..mb + hl], &staged[5][off..off + hl]);
            assert!(c.bufs[5][mb + hl..mb + 32].iter().all(|&b| b == 0));
            assert!(c.bufs[5][mb..mb + hl].iter().all(|&b| b == 1));
            off += hl;
        }
        // Beyond-rows untouched (0xAB sentinel past the committed footprint).
        let mut i = 0usize;
        while i < N_PLANES {
            let valid = 5 * row_bytes(i, 32);
            assert!(
                c.bufs[i][valid..].iter().all(|&b| b == 0xAB),
                "plane {i} beyond-rows touched"
            );
            i += 1;
        }
        // Spot checks: golden chosen ids, dora -1 intact (G3), actor seats.
        let chosen: Vec<i64> = (0..5).map(|r| rd_i64(&c.bufs[PLANE_CHOSEN], r)).collect();
        assert_eq!(chosen, [192, 193, 193, 193, 196]);
        for r in 0..5 {
            for k in 0..5 {
                assert_eq!(rd_dora(&c.bufs[PLANE_DORA], r, k), -1);
            }
        }
        let actors: Vec<i64> = (0..5).map(|r| rd_i64(&c.bufs[PLANE_ACTOR], r)).collect();
        assert_eq!(actors, [0, 1, 2, 3, 0]);
        assert_eq!(c.bufs[PLANE_CONCEALED][0], 4);
    }

    #[test]
    fn fill_exact_f2_distinct_bytes() {
        // F2: same shape, honor draws — distinct bytes, same golden structure.
        let g = stage_text(1, &["E", "E", "E", "E", "S"]);
        assert_eq!(g.rows, 5);
        let mut s = Scratch::new();
        s.commit(&g);
        let c = caller_bufs(5, 32);
        let ptrs = ptrs_of(&c);
        let out = s.fill_pinned(&ptrs, &c.caps).unwrap();
        assert_eq!(out.rows, 5);
        assert_eq!(out.games_consumed, 1);
        assert_eq!(out.t_len, 32);
        // Distinct chosen ids from F1 (honor tsumogiri), dora still -1.
        let chosen: Vec<i64> = (0..5).map(|r| rd_i64(&c.bufs[PLANE_CHOSEN], r)).collect();
        assert_ne!(chosen, [192, 193, 193, 193, 196]);
        for r in 0..5 {
            for k in 0..5 {
                assert_eq!(rd_dora(&c.bufs[PLANE_DORA], r, k), -1);
            }
        }
        // Legal plane packed exactly (128B ids + 8B len per row).
        assert_eq!(c.bufs[PLANE_LEGAL].len(), 5 * 136);
    }

    #[test]
    fn small_buffer_drains_nothing_plane_attributed() {
        let g = stage_text(0, &["5pr", "5p", "5p", "5p", "6p"]);
        let mut s = Scratch::new();
        s.commit(&g);
        let staged_len = s.planes()[PLANE_ACTOR].len();

        // Starve the actor plane: first-short is plane 7.
        let mut c = caller_bufs(5, 32);
        c.caps[PLANE_ACTOR] = 0;
        let ptrs = ptrs_of(&c);
        let err = s.fill_pinned(&ptrs, &c.caps).unwrap_err();
        assert_eq!(
            err,
            FillError::BufferTooSmall {
                plane: PLANE_ACTOR as u8,
                needed_bytes: staged_len,
                capacity_bytes: 0,
            }
        );
        // NOTHING drained: stage byte-identical, retry fills exactly.
        assert_eq!(s.rows(), 5);
        assert_eq!(s.games(), 1);
        assert_eq!(s.planes()[PLANE_ACTOR].len(), staged_len);

        // Plane:4 rule: starve history-kind instead.
        let mut c = caller_bufs(5, 32);
        c.caps[PLANE_HIST_KIND] = 0;
        let ptrs = ptrs_of(&c);
        let err = s.fill_pinned(&ptrs, &c.caps).unwrap_err();
        assert!(matches!(
            err,
            FillError::BufferTooSmall { plane: 4, .. }
        ));
        assert_eq!(s.rows(), 5);

        // Full caps after the failures: exact fill proves no partial commit.
        let c = caller_bufs(5, 32);
        let ptrs = ptrs_of(&c);
        let out = s.fill_pinned(&ptrs, &c.caps).unwrap();
        assert_eq!(out.rows, 5);
        assert_eq!(s.rows(), 0);
    }

    #[test]
    fn prefix_commit_leaves_remainder_staged() {
        let a = stage_text(0, &["5pr", "5p", "5p", "5p", "6p"]);
        let b = stage_text(1, &["E", "E", "E", "E", "S"]);
        let mut s = Scratch::new();
        s.commit(&a);
        s.commit(&b);
        assert_eq!((s.rows(), s.games()), (10, 2));

        // Caps fit exactly one game: commit the whole-game prefix only.
        let c = caller_bufs(5, 32);
        let ptrs = ptrs_of(&c);
        let out = s.fill_pinned(&ptrs, &c.caps).unwrap();
        assert_eq!(out.rows, 5);
        assert_eq!(out.games_consumed, 1);
        assert_eq!((s.rows(), s.games()), (5, 1));

        // Remainder drains on the next call, byte-exact vs game B.
        let c2 = caller_bufs(5, 32);
        let ptrs2 = ptrs_of(&c2);
        let out2 = s.fill_pinned(&ptrs2, &c2.caps).unwrap();
        assert_eq!((out2.rows, out2.games_consumed), (5, 1));
        assert_eq!((s.rows(), s.games()), (0, 0));
        let mut i = 0usize;
        while i < N_PLANES {
            if i == PLANE_HIST_KIND || i == PLANE_HIST_MASK {
                i += 1;
                continue;
            }
            assert_eq!(&c2.bufs[i][..5 * ROW_BYTES_FIXED[i]], &b.planes[i][..]);
            i += 1;
        }
    }

    #[test]
    fn truncate_rows_rolls_back_call() {
        let a = stage_text(0, &["5pr", "5p", "5p", "5p", "6p"]);
        let b = stage_text(1, &["E", "E", "E", "E", "S"]);
        let mut s = Scratch::new();
        s.commit(&a);
        let entry_rows = s.rows() as u32;
        s.commit(&b);
        assert_eq!((s.rows(), s.games()), (10, 2));
        // Whole-call rollback to the entry snapshot: byte-identical to
        // the single-game stage, then fills exactly.
        s.truncate_rows(entry_rows);
        assert_eq!((s.rows(), s.games()), (5, 1));
        assert_eq!(s.hist_lens(), &a.hist_lens[..]);
        let mut i = 0usize;
        while i < N_PLANES {
            assert_eq!(&s.planes()[i][..], &a.planes[i][..], "plane {i}");
            i += 1;
        }
        let c = caller_bufs(5, 32);
        let ptrs = ptrs_of(&c);
        let out = s.fill_pinned(&ptrs, &c.caps).unwrap();
        assert_eq!((out.rows, out.games_consumed), (5, 1));
        assert_eq!((s.rows(), s.games()), (0, 0));
    }

    #[test]
    fn truncate_rows_rounds_down_mid_game() {
        let a = stage_text(0, &["5pr", "5p", "5p", "5p", "6p"]);
        let b = stage_text(1, &["E", "E", "E", "E", "S"]);
        // Mid-game keep (7 of 5+5) rounds down to the prior game end:
        // whole-game invariant holds, byte-identical to game A alone.
        let mut s = Scratch::new();
        s.commit(&a);
        s.commit(&b);
        s.truncate_rows(7);
        assert_eq!((s.rows(), s.games()), (5, 1));
        assert_eq!(s.hist_lens(), &a.hist_lens[..]);
        assert_eq!(&s.planes()[PLANE_CHOSEN][..], &a.planes[PLANE_CHOSEN][..]);
        // Zero drops everything (state reusable after).
        s.commit(&b);
        s.truncate_rows(0);
        assert_eq!((s.rows(), s.games()), (0, 0));
        assert!(s.hist_lens().is_empty());
        assert_eq!(s.max_hist(), 0);
        // Full and over-full keeps are no-ops.
        s.commit(&a);
        s.commit(&b);
        s.truncate_rows(10);
        assert_eq!((s.rows(), s.games()), (10, 2));
        s.truncate_rows(u32::MAX);
        assert_eq!((s.rows(), s.games()), (10, 2));
        assert_eq!(s.max_hist(), 14);
    }
    #[test]
    fn stage_file_multi_game_unique_idx_independent_quarantine() {
        const A: &[&str] = &["5pr", "5p", "5p", "5p", "6p"];
        const C: &[&str] = &["E", "E", "E", "E", "S"];
        // Poison the middle game: one garbage line fails its chunk while
        // the start/end markers still split the file into 3 chunks.
        let mut poison = String::from_utf8(game_text(&["2s", "2s", "2s", "2s", "3s"])).unwrap();
        poison = poison.replacen(
            &tsumo_line(1, "2s"),
            "THIS IS NOT JSON",
            1,
        );
        let mut file = game_text(A);
        file.extend_from_slice(poison.as_bytes());
        file.extend_from_slice(&game_text(C));
        // Reference stages of the two good games (content-identical).
        let a = stage_text(0, A);
        let c = stage_text(2, C);
        // Nonzero file object_id + nonzero base prove caller-supplied
        // identity and keying (bridge manifest slot + game_seq).
        let mut s = Scratch::new();
        let out = s.stage_file(&file, 7, 41);
        // Advance by the staged count (3 chunks), not 1 and not ok-only 2.
        assert_eq!(out.advanced, 3);
        // Committed lineage keys carry the file object_id, not the base.
        assert_eq!(out.staged, [(7, 41), (7, 43)]);
        // Independent quarantine: exactly the middle chunk, keyed 41+1.
        assert_eq!(out.rejects.len(), 1);
        assert_eq!(out.rejects[0].stub.game_idx, 42);
        assert_eq!(out.rejects[0].stub.reason, crate::gate::REASON_FRAMING);
        // Siblings still sample, in file order, with game boundaries.
        assert_eq!((s.rows(), s.games()), (10, 2));
        assert_eq!(s.hist_lens()[..5], a.hist_lens[..]);
        assert_eq!(s.hist_lens()[5..], c.hist_lens[..]);
        let mut i = 0usize;
        while i < N_PLANES {
            if i == PLANE_HIST_KIND || i == PLANE_HIST_MASK {
                i += 1;
                continue;
            }
            let (an, cn) = (a.planes[i].len(), c.planes[i].len());
            assert_eq!(&s.planes()[i][..an], &a.planes[i][..], "plane {i} game A");
            assert_eq!(&s.planes()[i][an..], &c.planes[i][..], "plane {i} game C");
            assert_eq!(s.planes()[i].len(), an + cn);
            i += 1;
        }
        // Drains end-to-end: A rows then C rows.
        let cc = caller_bufs(10, 32);
        let ptrs = ptrs_of(&cc);
        let out = s.fill_pinned(&ptrs, &cc.caps).unwrap();
        assert_eq!((out.rows, out.games_consumed), (10, 2));
        let chosen: Vec<i64> = (0..10).map(|r| rd_i64(&cc.bufs[PLANE_CHOSEN], r)).collect();
        assert_eq!(&chosen[..5], &[192, 193, 193, 193, 196]);
        let c_chosen: Vec<i64> = (0..5).map(|r| rd_i64(&c.planes[PLANE_CHOSEN], r)).collect();
        assert_eq!(&chosen[5..], &c_chosen[..]);
    }

    #[test]
    fn stage_file_disjoint_keys_across_files() {
        // Two files, distinct file-stable object_ids, IDENTICAL game_idx
        // ranges: lineage keys must still be disjoint (no conflation).
        let mut f1 = game_text(&["5pr", "5p", "5p", "5p", "6p"]);
        f1.extend_from_slice(&game_text(&["E", "E", "E", "E", "S"]));
        let mut f2 = game_text(&["5pr", "5p", "5p", "5p", "6p"]);
        f2.extend_from_slice(&game_text(&["E", "E", "E", "E", "S"]));
        let mut s1 = Scratch::new();
        let o1 = s1.stage_file(&f1, 7, 0);
        let mut s2 = Scratch::new();
        let o2 = s2.stage_file(&f2, 8, 0);
        assert_eq!((o1.advanced, o2.advanced), (2, 2));
        assert!(o1.rejects.is_empty() && o2.rejects.is_empty());
        assert_eq!(o1.staged, [(7, 0), (7, 1)]);
        assert_eq!(o2.staged, [(8, 0), (8, 1)]);
        // Pairwise disjoint despite identical game_idx ranges.
        for k1 in &o1.staged {
            assert!(!o2.staged.contains(k1), "key collision on {k1:?}");
        }
        assert_eq!((s1.rows(), s1.games()), (10, 2));
        assert_eq!((s2.rows(), s2.games()), (10, 2));
    }

    #[test]
    fn scratch_reset_keeps_capacity() {
        let g = stage_text(0, &["5pr", "5p", "5p", "5p", "6p"]);
        let mut s = Scratch::new();
        s.commit(&g);
        let caps: [usize; N_PLANES] = std::array::from_fn(|i| s.planes()[i].capacity());
        s.reset();
        assert_eq!((s.rows(), s.games(), s.max_hist()), (0, 0, 0));
        assert!(s.hist_lens().is_empty());
        let mut i = 0usize;
        while i < N_PLANES {
            assert!(s.planes()[i].is_empty());
            assert_eq!(s.planes()[i].capacity(), caps[i]);
            i += 1;
        }
        // Reuse after reset stages byte-identically.
        s.commit(&g);
        assert_eq!(s.rows(), 5);
        assert_eq!(s.planes()[PLANE_CHOSEN], g.planes[PLANE_CHOSEN]);
    }

    #[test]
    fn stage_batch_parallel_ordered_past_channel_bound() {
        let pool = FeedPool::build(4).unwrap();
        // 150 jobs > STAGE_CHANNEL_BOUND(128): exercises bound, proves order.
        let mut jobs = Vec::new();
        let mut k = 0usize;
        while k < 150 {
            jobs.push(StageJob {
                game_idx: k as u32,
                object_id: 7,
                bytes: game_text(&["5pr", "5p", "5p", "5p", "6p"]),
                wall_digest: None,
            });
            k += 1;
        }
        // One poisoned job (empty payload -> framing quarantine).
        jobs.push(StageJob {
            game_idx: 1000,
            object_id: 7,
            bytes: Vec::new(),
            wall_digest: None,
        });
        let (games, rejects) = stage_batch(&pool, &jobs);
        assert_eq!(games.len(), 150);
        assert_eq!(rejects.len(), 1);
        assert_eq!(rejects[0].stub.game_idx, 1000);
        let mut k = 0usize;
        while k < 150 {
            assert_eq!(games[k].game_idx, k as u32);
            assert_eq!(games[k].rows, 5);
            k += 1;
        }
        // Ordered commit preserves input order in the stage.
        let mut s = Scratch::new();
        s.commit_batch(&games);
        assert_eq!((s.rows(), s.games()), (750, 150));
        assert_eq!(s.max_hist(), 14);
    }

    #[test]
    fn pool_build_normalizes() {
        assert!(FeedPool::build(0).is_ok());
        assert!(FeedPool::build(4).is_ok());
        assert_eq!(normalize_worker_threads(0), 1);
        assert!(normalize_worker_threads(usize::MAX) >= 1);
    }
}
