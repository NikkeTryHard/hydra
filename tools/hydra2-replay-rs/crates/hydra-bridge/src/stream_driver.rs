//! Thin bridge: `PyStreamDriver` over `feed::stream_driver` (Pull>Take inversion).
//!
//! DAG: this crate depends on the feed crate + pyo3 ONLY (same edge as
//! `stream.rs`; no shard/search arrow edge, no new dependency). FORBIDS:
//! parse/encode/hash/pool/slot-fill/plane-fill logic — arg-check + detach +
//! struct return only. All batcher/pool/order/take/hash/class math lives in
//! `hydra_feed::stream_driver`; this file only validates arguments attached,
//! runs the driver pure functions detached, and wraps the results attached.
//!
//! Scope split (no dual ownership):
//! - File discovery + serial file staging (`FeedStream::open`/`stage_at`)
//!   stays in sibling-owned `stream.rs`. This driver takes batched raw bytes
//!   Python already read (file-IO authority stays Python) and owns the
//!   expansion batcher + rayon pool + pull-order merge + contiguous-prefix
//!   takes + quarantine classes + window hash + snapshot scalars.
//! - Plane fill (`stage-then-commit` game expansion) stays feed-owned; this
//!   module never fills slots itself beyond forwarding the caller's pinned
//!   `data_ptr` handles (torch collate/H2D stays Python via the pinned-ring
//!   pattern, never a DLPack edge here).
//! - Resume cursors (`file_index`/`byte_offset`/`shuffle_pos`) and the RNG
//!   triple (`RngSnapshot`) stay sibling-owned `resume.rs`; this driver
//!   returns the take frontier (`offset`/`dropped`/`microbatches`/
//!   `games_seen` + `seed` passthrough) that Python packs there.
//!
//! Concurrency pattern (matches `stream.rs` + `resume.rs`): validate
//! attached, `py.detach(|| ...)` around every batch/blocking section
//! (pool fan-out/join never holds the interpreter), frozen pyclasses only
//! (`&self` methods), `Mutex<Option<...>>` + tail stash (stats/snapshot stay
//! readable post-close). The WHOLE push/take runs detached under the SAME
//! mutex guard (serial, never unlock-relock mid-call).
//!
//! GIL attestation inherits the wave-wide `gil_used=false` default (spelled
//! once in `canon_rng.rs`).

use std::sync::Mutex;

use pyo3::exceptions::{PyBufferError, PyOSError, PyValueError};
use pyo3::prelude::*;

use hydra_feed::stream_driver::{
    DriverEntry, DriverError, DriverSnapshot, DriverStats, MicrobatchOut, PushOut, StreamDriver,
    COMPACT_ROWS, MAX_BATCH_GAMES, MAX_TAKE_ROWS,
};
use hydra_feed::fill::N_PLANES;

/// One pulled game crossing the boundary: verbatim framed bytes, game
/// identity, whole-game key, and the S7 wall override (`Some` 136 tiles).
type DriverGame = (Vec<u8>, String, String, Option<Vec<u32>>);

const _: () = assert!(N_PLANES == 26);

/// Map a closed driver failure to its Python error.
///
/// `BufferTooSmall` is the ONLY `PyBufferError` (callers branch on it to
/// grow the slot); exhaustion and bad arguments are `ValueError`; pool
/// build failures are `OSError` (like the stream `Io`/`Poisoned` leg).
fn driver_py_err(err: DriverError) -> PyErr {
    match err {
        DriverError::BufferTooSmall { .. } => PyBufferError::new_err(err.to_string()),
        DriverError::InvalidArg(_) | DriverError::Exhausted(_) => {
            PyValueError::new_err(err.to_string())
        }
        DriverError::Pool(_) => PyOSError::new_err(err.to_string()),
    }
}

/// Batch push receipt crossing the boundary (per-call game accounting).
#[pyclass(name = "PyPushOut", frozen, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct PyPushOut {
    /// Games committed to the buffer this call.
    #[pyo3(get)]
    pub games_ok: u32,
    /// Games quarantined this call (zero rows each).
    #[pyo3(get)]
    pub games_quarantined: u32,
    /// Rows added to the buffer this call.
    #[pyo3(get)]
    pub rows_added: u32,
}

impl From<PushOut> for PyPushOut {
    fn from(out: PushOut) -> Self {
        Self {
            games_ok: out.games_ok,
            games_quarantined: out.games_quarantined,
            rows_added: out.rows_added,
        }
    }
}

/// Microbatch take crossing the boundary: plane fill + take envelope
/// (`ids`/`chosen`/`kinds`/`counts`/`classes`/frontier). The window hash
/// rides `snapshot()` only (takes stay `O(take)`).
#[pyclass(name = "PyMicrobatchOut", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyMicrobatchOut {
    /// Rows committed into the caller prefix this take.
    #[pyo3(get)]
    pub rows: u32,
    /// History bucket committed (`32|64|128|256`).
    #[pyo3(get)]
    pub t_len: u32,
    /// Take decision ids (`<game_id>:d<seq:04d>`, take order).
    #[pyo3(get)]
    pub ids: Vec<String>,
    /// Take chosen action ids (take order).
    #[pyo3(get)]
    pub chosen: Vec<i64>,
    /// Take action kinds (frozen census literals, take order).
    #[pyo3(get)]
    pub kinds: Vec<String>,
    /// Games replayed (walled path) lifetime.
    #[pyo3(get)]
    pub replayed: u64,
    /// Games sim-replayed (wall-less path) lifetime.
    #[pyo3(get)]
    pub sim_replayed: u64,
    /// Games quarantined lifetime.
    #[pyo3(get)]
    pub quarantined: u64,
    /// Quarantine reason classes (`(class, count)`, sorted).
    #[pyo3(get)]
    pub classes: Vec<(String, u64)>,
    /// Rows consumed lifetime.
    #[pyo3(get)]
    pub offset: u64,
    /// Compacted prefix length lifetime.
    #[pyo3(get)]
    pub dropped: u64,
    /// Microbatches consumed lifetime.
    #[pyo3(get)]
    pub microbatches: u64,
    /// Buffered rows.
    #[pyo3(get)]
    pub total_rows: u64,
    /// Games pushed lifetime.
    #[pyo3(get)]
    pub games_seen: u64,
}

impl From<MicrobatchOut> for PyMicrobatchOut {
    fn from(out: MicrobatchOut) -> Self {
        Self {
            rows: out.rows,
            t_len: out.t_len,
            ids: out.ids,
            chosen: out.chosen,
            kinds: out.kinds,
            replayed: out.replayed,
            sim_replayed: out.sim_replayed,
            quarantined: out.quarantined,
            classes: out.classes,
            offset: out.offset,
            dropped: out.dropped,
            microbatches: out.microbatches,
            total_rows: out.total_rows,
            games_seen: out.games_seen,
        }
    }
}

/// Buffer snapshot crossing the boundary (whole-game entries + counters +
/// row hash; `path`/`offset`/`split` rejoin Python-side).
#[pyclass(name = "PyDriverSnapshot", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyDriverSnapshot {
    /// Whole-game `(key, rows)` entries in buffer order.
    #[pyo3(get)]
    pub entries: Vec<(String, u32)>,
    /// Rows consumed lifetime.
    #[pyo3(get)]
    pub offset: u64,
    /// Compacted prefix length lifetime.
    #[pyo3(get)]
    pub dropped: u64,
    /// Microbatches consumed lifetime.
    #[pyo3(get)]
    pub microbatches: u64,
    /// Games replayed lifetime.
    #[pyo3(get)]
    pub replayed: u64,
    /// Games sim-replayed lifetime.
    #[pyo3(get)]
    pub sim_replayed: u64,
    /// Games quarantined lifetime.
    #[pyo3(get)]
    pub quarantined: u64,
    /// Quarantine classes (sorted).
    #[pyo3(get)]
    pub classes: Vec<(String, u64)>,
    /// Window hash over the live buffer.
    #[pyo3(get)]
    pub row_hash: String,
    /// Buffered rows.
    #[pyo3(get)]
    pub total_rows: u64,
}

impl From<DriverSnapshot> for PyDriverSnapshot {
    fn from(snap: DriverSnapshot) -> Self {
        Self {
            entries: snap
                .entries
                .into_iter()
                .map(|e: DriverEntry| (e.key, e.rows))
                .collect(),
            offset: snap.offset,
            dropped: snap.dropped,
            microbatches: snap.microbatches,
            replayed: snap.replayed,
            sim_replayed: snap.sim_replayed,
            quarantined: snap.quarantined,
            classes: snap.classes,
            row_hash: snap.row_hash,
            total_rows: snap.total_rows,
        }
    }
}

/// Cumulative counters crossing the boundary.
#[pyclass(name = "PyDriverStats", frozen, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct PyDriverStats {
    /// Games pushed lifetime.
    #[pyo3(get)]
    pub games_seen: u64,
    /// Games committed lifetime.
    #[pyo3(get)]
    pub games_ok: u64,
    /// Games quarantined lifetime.
    #[pyo3(get)]
    pub games_quarantined: u64,
    /// Rows committed lifetime.
    #[pyo3(get)]
    pub rows_in: u64,
    /// Rows consumed lifetime.
    #[pyo3(get)]
    pub rows_out: u64,
    /// Microbatches consumed lifetime.
    #[pyo3(get)]
    pub microbatches: u64,
}

impl From<DriverStats> for PyDriverStats {
    fn from(stats: DriverStats) -> Self {
        Self {
            games_seen: stats.games_seen,
            games_ok: stats.games_ok,
            games_quarantined: stats.games_quarantined,
            rows_in: stats.rows_in,
            rows_out: stats.rows_out,
            microbatches: stats.microbatches,
        }
    }
}

/// Post-close stash: `close` takes the driver, but stats/snapshot stay
/// readable.
#[derive(Debug, Clone, Default)]
struct TailStats {
    stats: Option<DriverStats>,
    snapshot: Option<DriverSnapshot>,
}

/// PyO3 handoff: `Mutex<Option<Driver>>` with poison/closed errors, `detach`
/// around every blocking section (pool fan-out + stage + fill). The WHOLE
/// push/take runs detached under the SAME mutex guard (serial, never
/// unlock-relock mid-call).
#[pyclass(name = "PyStreamDriver", frozen)]
pub struct PyStreamDriver {
    inner: Mutex<Option<StreamDriver>>,
    tail: Mutex<TailStats>,
}

#[pymethods]
impl PyStreamDriver {
    /// Open-once: build the owned pool, pin the cursor seed and compact
    /// bound. No rows flow until the first `push_games`.
    #[staticmethod]
    #[pyo3(signature = (seed, threads=0, compact_rows=None))]
    fn open(
        py: Python<'_>,
        seed: u64,
        threads: usize,
        compact_rows: Option<usize>,
    ) -> PyResult<Self> {
        if threads > 64 {
            return Err(PyValueError::new_err(format!(
                "hydra2 stream driver threads must be <= 64, got {threads}"
            )));
        }
        let compact = compact_rows.unwrap_or(COMPACT_ROWS);
        if compact == 0 || compact > (1usize << 20) {
            return Err(PyValueError::new_err(format!(
                "hydra2 stream driver compact_rows must be in [1, 1048576], got {compact}"
            )));
        }
        let driver =
            py.detach(|| StreamDriver::open(seed, threads, compact)).map_err(driver_py_err)?;
        Ok(Self {
            inner: Mutex::new(Some(driver)),
            tail: Mutex::new(TailStats::default()),
        })
    }

    /// Pull: expand batched raw games in pull order via the owned pool.
    ///
    /// `games` are `(bytes, game_id, key, wall)` in pull order (`wall`
    /// `Some` splices the 136-tile S7 override in Rust, `None` walks the
    /// embedded content). Returns the per-call push receipt; per-game
    fn push_games(&self, py: Python<'_>, games: Vec<DriverGame>) -> PyResult<PyPushOut> {
        if games.len() > MAX_BATCH_GAMES {
            return Err(PyValueError::new_err(format!(
                "hydra2 stream driver batch games invalid: {} (max {MAX_BATCH_GAMES})",
                games.len()
            )));
        }
        for (pos, (bytes, game_id, key, wall)) in games.iter().enumerate() {
            if bytes.len() > (8usize << 20) {
                return Err(PyValueError::new_err(format!(
                    "hydra2 stream driver game {pos} bytes exceed 8MiB"
                )));
            }
            if game_id.is_empty() || game_id.len() > 1024 {
                return Err(PyValueError::new_err(format!(
                    "hydra2 stream driver game {pos} game_id must be 1..=1024 chars"
                )));
            }
            if key.is_empty() || key.len() > 1024 {
                return Err(PyValueError::new_err(format!(
                    "hydra2 stream driver game {pos} key must be 1..=1024 chars"
                )));
            }
            if let Some(w) = wall
                && w.len() > 1024
            {
                return Err(PyValueError::new_err(format!(
                    "hydra2 stream driver game {pos} wall exceeds 1024 tiles"
                )));
            }
        }
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 stream driver mutex poisoned"))?;
        let driver = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("hydra2 stream driver is closed"))?;
        // SAME guard across the detach: serial, never unlock-relock mid-push.
        py.detach(|| {
            let jobs = games
                .into_iter()
                .map(|(bytes, game_id, key, wall)| hydra_feed::stream_driver::DriverJob {
                    bytes,
                    game_id,
                    key,
                    wall,
                })
                .collect();
            driver.push_games(jobs).map(PyPushOut::from).map_err(driver_py_err)
        })
    }

    /// Take: advance exactly one microbatch through the fill machine.
    ///
    /// `ptrs` are 26 pinned-slot base addresses (canonical plane order
    /// #0-25), `byte_caps` their byte sizes (`tensor.nbytes`). Rows land in
    /// the caller PREFIX exactly (row-exact, may cut mid-game); `t_len`
    /// selects the committed history width. The GIL is released for the
    /// entire fill + envelope build. A null pointer fails closed; a
    /// too-small buffer raises `PyBufferError`, drains nothing, and moves
    /// no counter (retry with a bigger buffer); exhaustion raises
    /// `ValueError` (single-pass: stream end with updates remaining).
    #[pyo3(signature = (ptrs, byte_caps, batch_size))]
    fn next_microbatch(
        &self,
        py: Python<'_>,
        ptrs: [u64; 26],
        byte_caps: [usize; 26],
        batch_size: usize,
    ) -> PyResult<PyMicrobatchOut> {
        if batch_size == 0 || batch_size > MAX_TAKE_ROWS {
            return Err(PyValueError::new_err(format!(
                "hydra2 stream driver batch_size must be in [1, {MAX_TAKE_ROWS}], got {batch_size}"
            )));
        }
        for (plane, ptr) in ptrs.iter().enumerate() {
            if *ptr == 0 {
                return Err(PyValueError::new_err(format!(
                    "hydra2 stream driver next_microbatch got a null buffer pointer for plane {plane}"
                )));
            }
        }
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 stream driver mutex poisoned"))?;
        let driver = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("hydra2 stream driver is closed"))?;
        // SAME guard across the detach: serial, never unlock-relock mid-take.
        py.detach(|| {
            driver
                .next_microbatch(batch_size, &ptrs, &byte_caps)
                .map(PyMicrobatchOut::from)
                .map_err(driver_py_err)
        })
    }

    /// Buffer snapshot: whole-game entries + counters + row hash
    /// (verify-before-mutate shape).
    fn snapshot(&self, py: Python<'_>) -> PyResult<PyDriverSnapshot> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 stream driver mutex poisoned"))?;
        if let Some(driver) = guard.as_mut() {
            return py.detach(|| driver.snapshot().map(PyDriverSnapshot::from).map_err(driver_py_err));
        }
        let tail = self
            .tail
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 stream driver mutex poisoned"))?;
        tail.snapshot
            .clone()
            .map(PyDriverSnapshot::from)
            .ok_or_else(|| PyValueError::new_err("hydra2 stream driver is closed"))
    }

    /// Cumulative counters (readable post-close via the close stash).
    fn stats(&self) -> PyResult<PyDriverStats> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 stream driver mutex poisoned"))?;
        if let Some(driver) = guard.as_ref() {
            return Ok(PyDriverStats::from(driver.stats()));
        }
        let tail = self
            .tail
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 stream driver mutex poisoned"))?;
        tail.stats
            .map(PyDriverStats::from)
            .ok_or_else(|| PyValueError::new_err("hydra2 stream driver is closed"))
    }

    /// Cursor seed passthrough (semantic counter-based, never wall-clock).
    fn seed(&self) -> PyResult<u64> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 stream driver mutex poisoned"))?;
        if let Some(driver) = guard.as_ref() {
            return Ok(driver.seed());
        }
        Err(PyValueError::new_err("hydra2 stream driver is closed"))
    }

    /// Idempotent close: stash stats/snapshot, then take + detach the
    /// drop. The second call is a no-op success.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let driver = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 stream driver mutex poisoned"))?
            .take();
        if let Some(driver) = driver {
            let stats = driver.stats();
            let snapshot = driver.snapshot().ok();
            if let Ok(mut tail) = self.tail.lock() {
                tail.stats = Some(stats);
                tail.snapshot = snapshot;
            }
            py.detach(|| std::mem::drop(driver));
        }
        Ok(())
    }
}

/// Register the stream-driver surface on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStreamDriver>()?;
    m.add_class::<PyPushOut>()?;
    m.add_class::<PyMicrobatchOut>()?;
    m.add_class::<PyDriverSnapshot>()?;
    m.add_class::<PyDriverStats>()?;
    Ok(())
}

#[cfg(test)]
mod driver_bridge_tests {
    use super::*;

    #[test]
    fn driver_error_mapping_is_closed() {
        Python::attach(|py| {
            let too_small = driver_py_err(DriverError::BufferTooSmall {
                plane: 4,
                needed: 10,
                capacity: 5,
            });
            assert!(too_small.is_instance_of::<PyBufferError>(py));
            let exhausted = driver_py_err(DriverError::Exhausted("empty".to_string()));
            assert!(exhausted.is_instance_of::<PyValueError>(py));
            let invalid = driver_py_err(DriverError::InvalidArg("bad".to_string()));
            assert!(invalid.is_instance_of::<PyValueError>(py));
            let pool = driver_py_err(DriverError::Pool("fail".to_string()));
            assert!(pool.is_instance_of::<PyOSError>(py));
        });
    }
}
