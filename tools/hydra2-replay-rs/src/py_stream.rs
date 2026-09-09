//! PyO3 JSON-rows handoff (Slice 4, Phase A: JSON rows, no tensors/buffers).
//!
//! Reference (read-only): `hydra-raw-mjai-pyo3/src/lib.rs` (Mutex-Option
//! stream, open-once/scan-once, `py.detach` around blocking sections,
//! raw-pointer caller buffers, poison/closed errors). Shape reused; the
//! payload here is DecisionRow-JSON (`ReplayRow::to_decision_json`, exactly
//! the 13 `ACTOR_FIELDS`), never parquet bytes or torch pointers.
//!
//! Serial only: `workers > 0` fails closed (`StreamError::WorkersDeferred`)
//! until the Slice-7 worker pool. The Python stub
//! (`src/hydra2/training/rust_stream.py`) re-checks `FORBIDDEN_REPLAY_KEYS`
//! per batch; this module enforces the same key discipline on the Rust side
//! before any byte crosses the FFI boundary.
//!
//! Provenance binding (Slice 5): the stream mints every digest from pinned
//! authorities, never from operator strings. `source_object_id` is the
//! input file stem (the dump CLI additionally honors a manifest; the
//! stream carries none), `split` is the `open` split, `rules_hash` is the
//! pinned rules-manifest digest (published bytes win), `adapter_hash` is
//! the `ENGINE_IDENTITY` digest, and `action_table_hash` is the verified
//! table CONTENT digest (`HYDRA2_REPLAY_TABLE` when set, else the baked v1
//! artifact). `spec_hash` stays accepted for FFI stability but is no longer
//! bound into rows (the S8 cutover re-plumbs the Python stub).

use std::collections::VecDeque;
use std::fmt;
use std::fs;
use std::mem;
use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::exceptions::{PyBufferError, PyOSError, PyValueError};
use pyo3::prelude::*;

use crate::decisions::ActionTable;
use crate::provenance;
use crate::replay_game_text;
use crate::hydra2_row::{ACTOR_FIELDS, Quarantine, RowProvenance};

/// Privileged keys that must never appear in actor rows.
///
/// Mirror of `src/hydra2/data/parquet.py::FORBIDDEN_IN_ACTOR`, enforced here
/// AND in the Python stub (defense in depth, both sides fail closed).
pub const FORBIDDEN_IN_ACTOR: [&str; 6] = [
    "hidden_tiles",
    "wall",
    "dead_wall",
    "opponent_hand",
    "privileged",
    "full_world",
];
/// Decision-row envelope version (W3-A version bump).
///
/// v1 is exactly the 13 `ACTOR_FIELDS` projection rows this gate enforces
/// (see `check_decision_envelope`); the Python bridge expands them into
/// `BRIDGE_FULL_DOC_ENVELOPE_VERSION` full docs downstream. Any row that is
/// not exactly v1 fails closed here, and any unexpanded v1 projection that
/// reaches the encoder fails closed in `dataset.py`
/// (`unexpanded-projection-row`). Bump only with a matching ingress update
/// plus an oracle re-freeze.
pub const DECISION_ENVELOPE_VERSION: &str = "v1";
/// Baked action-table bytes: the published v1 artifact, hermetic default.
/// `HYDRA2_REPLAY_TABLE` overrides with an explicit file (fail closed when
/// unreadable or unverifiable); the verified table CONTENT digest
/// (`table.digest`, never a file-bytes hash) becomes `action_table_hash`.
const BAKED_ACTION_TABLE: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../configs/contracts/action_table_v1.json"));

/// Closed stream failure vocabulary (every variant maps to a Python error).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    /// Bad open arguments / unreadable inputs (never a partial stream).
    Open(String),
    /// Action table missing or unverifiable.
    Table(String),
    /// Provenance authority missing or unverifiable (rules manifest).
    Provenance(String),
    /// File read / decode failure mid-stream (fail closed per call).
    Io(String),
    /// A row that is not exactly the 13 `ACTOR_FIELDS` (leak or schema fork).
    Envelope(String),
    /// Caller buffer too small: nothing drained, retry with a bigger buffer.
    BufferTooSmall { needed: usize, capacity: usize },
    /// `workers > 0`: deferred to the Slice-7 pool, never silently serial.
    WorkersDeferred,
    /// Use after `close`.
    Closed,
    /// Mutex poison (a prior panic while holding the guard).
    Poisoned,
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::Open(msg) => write!(f, "hydra2 replay open: {msg}"),
            StreamError::Table(msg) => write!(f, "hydra2 replay action table: {msg}"),
            StreamError::Provenance(msg) => write!(f, "hydra2 replay provenance: {msg}"),
            StreamError::Io(msg) => write!(f, "hydra2 replay io: {msg}"),
            StreamError::Envelope(msg) => write!(f, "hydra2 replay envelope: {msg}"),
            StreamError::BufferTooSmall { needed, capacity } => write!(
                f,
                "hydra2 replay batch needs {needed} bytes but the caller buffer holds {capacity}"
            ),
            StreamError::WorkersDeferred => write!(
                f,
                "hydra2 replay workers>0 is deferred to Slice 7 (pinned buffers); pass workers=0"
            ),
            StreamError::Closed => write!(f, "hydra2 replay stream is closed"),
            StreamError::Poisoned => write!(f, "hydra2 replay stream mutex poisoned"),
        }
    }
}

impl std::error::Error for StreamError {}

/// Open parameters (mirrors the PyO3 `open` signature 1:1).
#[derive(Debug, Clone)]
pub struct StreamOpen {
    pub data_dirs: Vec<PathBuf>,
    pub batch_rows: usize,
    pub workers: usize,
    pub queue: usize,
    pub split: String,
    /// Legacy operator pin, retained for FFI stability: still required
    /// non-empty at open, but NO LONGER bound into rows (Slice 5 mints
    /// real digests; the S8 cutover re-plumbs the Python stub).
    pub spec_hash: String,
}

/// Cumulative counters. `open_count` is 1 on every live stream (open-once).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamStats {
    pub open_count: u64,
    pub games_ok: u64,
    pub games_quarantined: u64,
    pub rows_out: u64,
}

/// One drained batch: newline-delimited decision-JSON payload description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawNext {
    pub rows: usize,
    pub games_consumed: usize,
    pub bytes_written: usize,
}

/// Envelope gate: the decision value must carry EXACTLY the 13
/// `ACTOR_FIELDS` (no privileged extras, no missing schema fields), the
/// `actor_observation` JSON string must parse to an object with no
/// `FORBIDDEN_IN_ACTOR` keys, and `chosen_action_id` must be an integer or
/// null (mirrors `rust_stream._validated_rows` on the Python side: both
/// sides fail closed on the same mutations).
fn check_decision_envelope(value: &serde_json::Value) -> Result<(), StreamError> {
    let obj = value.as_object().ok_or_else(|| {
        StreamError::Envelope("decision row is not a JSON object".to_string())
    })?;
    for key in obj.keys() {
        if FORBIDDEN_IN_ACTOR.contains(&key.as_str()) {
            return Err(StreamError::Envelope(format!(
                "decision row carries forbidden key {key:?}"
            )));
        }
        if !ACTOR_FIELDS.contains(&key.as_str()) {
            return Err(StreamError::Envelope(format!(
                "decision row carries non-actor key {key:?}"
            )));
        }
    }
    for field in ACTOR_FIELDS {
        if !obj.contains_key(field) {
            return Err(StreamError::Envelope(format!(
                "decision row misses actor field {field:?}"
            )));
        }
    }
    let text = obj
        .get("actor_observation")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            StreamError::Envelope("actor_observation is not a JSON string".to_string())
        })?;
    let doc: serde_json::Value =
        serde_json::from_str(text).map_err(|e| {
            StreamError::Envelope(format!("actor_observation does not parse: {e}"))
        })?;
    let doc_obj = doc.as_object().ok_or_else(|| {
        StreamError::Envelope("actor_observation is not a JSON object".to_string())
    })?;
    for key in doc_obj.keys() {
        if FORBIDDEN_IN_ACTOR.contains(&key.as_str()) {
            return Err(StreamError::Envelope(format!(
                "actor_observation carries forbidden key {key:?}"
            )));
        }
    }
    match obj.get("chosen_action_id") {
        Some(serde_json::Value::Number(n)) if n.is_u64() || n.is_i64() => Ok(()),
        Some(serde_json::Value::Null) => Ok(()),
        _ => Err(StreamError::Envelope(
            "chosen_action_id must be an integer or null".to_string(),
        )),
    }
}

/// Framed per-game inputs, same suffixes as the dump CLI (plain `.jsonl` or
/// compressed `.jsonl.zst` / `.jsonl.gz`).
fn is_game_file(file_name: &str) -> bool {
    file_name.ends_with(".jsonl")
        || file_name.ends_with(".jsonl.zst")
        || file_name.ends_with(".jsonl.gz")
}

/// Object id without a manifest: the file stem (mirrors the dump CLI
/// fallback for the probe layout).
fn object_id_for(file_name: &str) -> String {
    for suffix in [".jsonl.zst", ".jsonl.gz", ".jsonl"] {
        if let Some(stem) = file_name.strip_suffix(suffix) {
            return stem.to_string();
        }
    }
    file_name.to_string()
}

/// Read one framed input, transparently decoding `.zst` / `.gz` to text.
fn read_input_text(path: &std::path::Path, file_name: &str) -> Result<String, StreamError> {
    let bytes =
        fs::read(path).map_err(|e| StreamError::Io(format!("input read {file_name}: {e}")))?;
    let raw: Vec<u8> = if file_name.ends_with(".zst") {
        zstd::decode_all(&bytes[..])
            .map_err(|e| StreamError::Io(format!("input zstd decode {file_name}: {e}")))?
    } else if file_name.ends_with(".gz") {
        use std::io::Read as _;
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&bytes[..])
            .read_to_end(&mut out)
            .map_err(|e| StreamError::Io(format!("input gzip decode {file_name}: {e}")))?;
        out
    } else {
        bytes
    };
    String::from_utf8(raw)
        .map_err(|e| StreamError::Io(format!("input utf8 {file_name}: {e}")))
}

/// Load the action table plus its CONTENT digest (env override else baked).
///
/// The table is fully verified at load (`ActionTable::load_json` mirrors
/// the Python `_table` path); the bound hash is the verified content
/// digest, never a file-bytes hash.
fn load_action_table() -> Result<(ActionTable, String), StreamError> {
    let text = match std::env::var("HYDRA2_REPLAY_TABLE")
        .ok()
        .filter(|v| !v.is_empty())
    {
        Some(path) => fs::read_to_string(&path)
            .map_err(|e| StreamError::Table(format!("HYDRA2_REPLAY_TABLE read {path}: {e}")))?,
        None => BAKED_ACTION_TABLE.to_string(),
    };
    let table = ActionTable::load_json(&text)
        .map_err(|e| StreamError::Table(format!("action table load: {e}")))?;
    let hash = table.digest.clone();
    Ok((table, hash))
}

/// Load the pinned rules manifest (baked hermetic default, fail closed).
fn load_rules_info() -> Result<provenance::RulesInfo, StreamError> {
    provenance::load_baked_rules()
        .map_err(|e| StreamError::Provenance(format!("rules manifest load: {e}")))
}

/// Pure-Rust serial replay stream: sorted file discovery, whole-game loads,
/// quarantine capture (never a new reason code: every game-level failure is
/// a `Quarantine` from the shared `replay_game_text` entry point).
pub struct ReplayStream {
    files: Vec<PathBuf>,
    cursor: usize,
    table: ActionTable,
    action_table_hash: String,
    batch_rows: usize,
    queue: usize,
    split: String,
    rules_hash: String,
    adapter_hash: String,
    pending: VecDeque<Vec<u8>>,
    quarantines: Vec<Quarantine>,
    games_ok: u64,
    games_quarantined: u64,
    rows_out: u64,
    closed: bool,
}

impl ReplayStream {
    /// Open-once: validate everything, discover + sort inputs, load the
    /// table. No rows flow until the first `next_into_raw`.
    pub fn open(cfg: StreamOpen) -> Result<Self, StreamError> {
        if cfg.data_dirs.is_empty() {
            return Err(StreamError::Open(
                "data_dirs must not be empty".to_string(),
            ));
        }
        if cfg.batch_rows == 0 {
            return Err(StreamError::Open("batch must be >= 1".to_string()));
        }
        if cfg.workers > 0 {
            return Err(StreamError::WorkersDeferred);
        }
        if cfg.split != "train" && cfg.split != "validation" {
            return Err(StreamError::Open(format!(
                "invalid split {:?}; expected train or validation",
                cfg.split
            )));
        }
        if cfg.spec_hash.is_empty() {
            return Err(StreamError::Open("spec_hash must not be empty".to_string()));
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
                .map_err(|e| {
                    StreamError::Open(format!("inputs list {}: {e}", dir.display()))
                })?
                .map(|entry| {
                    entry
                        .map(|e| e.path())
                        .map_err(|e| {
                            StreamError::Open(format!("inputs entry {}: {e}", dir.display()))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .filter_map(|path| {
                    let name = path.file_name()?.to_string_lossy().to_string();
                    is_game_file(&name).then(|| (name, path.clone()))
                })
                .map(|(name, _)| name)
                .collect();
            names.sort();
            for name in names {
                files.push(dir.join(name));
            }
        }
        let (table, action_table_hash) = load_action_table()?;
        let rules = load_rules_info()?;
        let adapter_hash = provenance::adapter_hash();
        Ok(Self {
            files,
            cursor: 0,
            table,
            action_table_hash,
            batch_rows: cfg.batch_rows,
            queue: cfg.queue,
            split: cfg.split,
            rules_hash: rules.rules_hash,
            adapter_hash,
            pending: VecDeque::new(),
            quarantines: Vec::new(),
            games_ok: 0,
            games_quarantined: 0,
            rows_out: 0,
            closed: false,
        })
    }

    /// Load the next game (ok rows to `pending`, quarantine to the sidecar
    /// list). `Ok(false)` means the input list is exhausted.
    fn load_next_game(&mut self) -> Result<bool, StreamError> {
        if self.cursor >= self.files.len() {
            return Ok(false);
        }
        let path = self.files[self.cursor].clone();
        self.cursor += 1;
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let text = read_input_text(&path, &file_name)?;
        let object_id = object_id_for(&file_name);
        let provenance = RowProvenance {
            source_object_id: object_id.clone(),
            split: self.split.clone(),
            rules_hash: self.rules_hash.clone(),
            adapter_hash: self.adapter_hash.clone(),
            action_table_hash: self.action_table_hash.clone(),
        };
        match replay_game_text(&text, &object_id, &self.table) {
            Ok(rows) => {
                // Stage-then-commit: every row of the game must pass the
                // envelope gate before any of it becomes visible. Validating
                // inline would leave a quarantined-by-bug game's prefix in
                // `pending` (partial rows) with `games_ok` already bumped.
                // The gate is expected to pass on all driver output (it is a
                // code-bug tripwire, never a game bug); on failure the call
                // fails closed with nothing drained and no counter moved.
                let mut staged: Vec<Vec<u8>> = Vec::with_capacity(rows.len());
                for row in &rows {
                    let decision = row.to_decision_json(&provenance);
                    check_decision_envelope(&decision)?;
                    let mut line = serde_json::to_vec(&decision).map_err(|e| {
                        StreamError::Envelope(format!("decision row encode: {e}"))
                    })?;
                    line.push(b'\n');
                    staged.push(line);
                }
                self.games_ok += 1;
                self.pending.extend(staged);
            }
            Err(quarantine) => {
                self.games_quarantined += 1;
                self.quarantines.push(quarantine);
            }
        }
        Ok(true)
    }

    /// Drain up to `batch_rows` decision lines into the caller buffer.
    /// Exhaustion is `rows == 0 && games_consumed == 0` (a call that only
    /// loads quarantines reports `games_consumed > 0` with `rows == 0`).
    /// A too-small buffer drains nothing: retry with a bigger buffer.
    pub fn next_into_raw(&mut self, dst: &mut [u8]) -> Result<RawNext, StreamError> {
        if self.closed {
            return Err(StreamError::Closed);
        }
        let mut games_consumed = 0usize;
        while self.pending.len() < self.batch_rows {
            if !self.load_next_game()? {
                break;
            }
            games_consumed += 1;
        }
        let take = self.batch_rows.min(self.pending.len());
        let needed: usize = self.pending.iter().take(take).map(|line| line.len()).sum();
        if needed > dst.len() {
            return Err(StreamError::BufferTooSmall {
                needed,
                capacity: dst.len(),
            });
        }
        let mut written = 0usize;
        for _ in 0..take {
            if let Some(line) = self.pending.pop_front() {
                dst[written..written + line.len()].copy_from_slice(&line);
                written += line.len();
            }
        }
        self.rows_out += take as u64;
        Ok(RawNext {
            rows: take,
            games_consumed,
            bytes_written: written,
        })
    }

    /// Cumulative counters (`open_count == 1` on every live stream).
    pub fn stats(&self) -> StreamStats {
        StreamStats {
            open_count: 1,
            games_ok: self.games_ok,
            games_quarantined: self.games_quarantined,
            rows_out: self.rows_out,
        }
    }

    /// Whole-game quarantines in load order (reason codes are the shared
    /// Slice-3 taxonomy; the JSON path introduces none of its own).
    pub fn quarantines(&self) -> &[Quarantine] {
        &self.quarantines
    }

    /// Queue depth as configured (accepted, unused in serial mode).
    pub fn queue(&self) -> usize {
        self.queue
    }

    /// Idempotent close: pending rows drop, counters/quarantines stay
    /// readable at the Rust level.
    pub fn close(&mut self) {
        self.closed = true;
        self.pending.clear();
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

fn stream_py_err(err: StreamError) -> PyErr {
    match err {
        StreamError::Open(msg)
        | StreamError::Table(msg)
        | StreamError::Provenance(msg)
        | StreamError::Envelope(msg) => PyValueError::new_err(msg),
        StreamError::WorkersDeferred | StreamError::Closed => PyValueError::new_err(err.to_string()),
        StreamError::BufferTooSmall { .. } => PyBufferError::new_err(err.to_string()),
        StreamError::Io(_) | StreamError::Poisoned => PyOSError::new_err(err.to_string()),
    }
}

#[pyclass(name = "PyReplayStats", skip_from_py_object)]
#[derive(Clone, Copy)]
struct PyReplayStats {
    #[pyo3(get)]
    open_count: u64,
    #[pyo3(get)]
    games_ok: u64,
    #[pyo3(get)]
    games_quarantined: u64,
    #[pyo3(get)]
    rows_out: u64,
}

impl From<StreamStats> for PyReplayStats {
    fn from(stats: StreamStats) -> Self {
        Self {
            open_count: stats.open_count,
            games_ok: stats.games_ok,
            games_quarantined: stats.games_quarantined,
            rows_out: stats.rows_out,
        }
    }
}

/// Whole-game quarantine record crossing the boundary (Slice-3 reason codes).
#[pyclass(name = "PyQuarantine", skip_from_py_object)]
#[derive(Clone)]
struct PyQuarantine {
    #[pyo3(get)]
    game_id: String,
    #[pyo3(get)]
    reason_code: String,
    #[pyo3(get)]
    detail: String,
}

impl From<&Quarantine> for PyQuarantine {
    fn from(q: &Quarantine) -> Self {
        Self {
            game_id: q.game_id.clone(),
            reason_code: q.reason_code.clone(),
            detail: q.detail.clone(),
        }
    }
}

/// `next_into_json()` surface: drained row count, games loaded by this call,
/// and the valid prefix length of the caller buffer.
#[pyclass(name = "PyReplayNext", skip_from_py_object)]
#[derive(Clone, Copy)]
struct PyReplayNext {
    #[pyo3(get)]
    rows: usize,
    #[pyo3(get)]
    games_consumed: usize,
    #[pyo3(get)]
    bytes_written: usize,
    #[pyo3(get)]
    stats: PyReplayStats,
}

/// PyO3 handoff: `Mutex<Option<Stream>>` with poison/closed errors, `detach`
/// around every blocking section (file reads + the log-order walk).
#[pyclass(name = "PyHydra2ReplayStream")]
pub struct PyHydra2ReplayStream {
    inner: Mutex<Option<ReplayStream>>,
}

#[pymethods]
impl PyHydra2ReplayStream {
    #[staticmethod]
    #[pyo3(signature = (data_dirs, batch, workers, queue, split, spec_hash))]
    fn open(
        py: Python<'_>,
        data_dirs: Vec<PathBuf>,
        batch: usize,
        workers: usize,
        queue: usize,
        split: &str,
        spec_hash: &str,
    ) -> PyResult<Self> {
        if data_dirs.is_empty() {
            return Err(PyValueError::new_err(
                "hydra2 replay data_dirs must not be empty",
            ));
        }
        let cfg = StreamOpen {
            data_dirs,
            batch_rows: batch,
            workers,
            queue,
            split: split.to_string(),
            spec_hash: spec_hash.to_string(),
        };
        let stream = py
            .detach(|| ReplayStream::open(cfg))
            .map_err(stream_py_err)?;
        Ok(Self {
            inner: Mutex::new(Some(stream)),
        })
    }

    #[pyo3(signature = (buf_ptr, capacity))]
    fn next_into_json(
        &self,
        py: Python<'_>,
        buf_ptr: usize,
        capacity: usize,
    ) -> PyResult<PyReplayNext> {
        if buf_ptr == 0 {
            return Err(PyValueError::new_err(
                "hydra2 replay next_into_json got a null buffer pointer",
            ));
        }
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 replay stream mutex poisoned"))?;
        let stream = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("hydra2 replay stream is closed"))?;
        py.detach(|| unsafe {
            let dst = std::slice::from_raw_parts_mut(buf_ptr as *mut u8, capacity);
            let next = stream.next_into_raw(dst).map_err(stream_py_err)?;
            let stats = PyReplayStats::from(stream.stats());
            Ok(PyReplayNext {
                rows: next.rows,
                games_consumed: next.games_consumed,
                bytes_written: next.bytes_written,
                stats,
            })
        })
    }

    fn stats(&self) -> PyResult<PyReplayStats> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 replay stream mutex poisoned"))?;
        let stream = guard
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("hydra2 replay stream is closed"))?;
        Ok(PyReplayStats::from(stream.stats()))
    }

    fn quarantines(&self) -> PyResult<Vec<PyQuarantine>> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 replay stream mutex poisoned"))?;
        let stream = guard
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("hydra2 replay stream is closed"))?;
        Ok(stream.quarantines().iter().map(PyQuarantine::from).collect())
    }

    /// Idempotent close: the second call is a no-op success.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let stream = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("hydra2 replay stream mutex poisoned"))?
            .take();
        if let Some(stream) = stream {
            py.detach(|| mem::drop(stream));
        }
        Ok(())
    }
}

/// Extension module: `import hydra2_replay_rs`.
#[pymodule]
pub fn hydra2_replay_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHydra2ReplayStream>()?;
    m.add_class::<PyReplayNext>()?;
    m.add_class::<PyReplayStats>()?;
    m.add_class::<PyQuarantine>()?;
    Ok(())
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    const TILES: [&str; 34] = [
        "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p",
        "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E",
        "S", "W", "N", "P", "F", "C",
    ];

    /// 52 tehai slots + draws cycle the 34 types (<= 2 copies each, no aka).
    fn tile_at(slot: usize) -> String {
        TILES[slot % TILES.len()].to_string()
    }

    fn cycling_hand(seat: usize) -> Vec<String> {
        (0..13).map(|i| tile_at(seat * 13 + i)).collect()
    }

    /// Winner tenpai for the tail-drawn tile `pai` (one copy held, so the
    /// draw is take #2 of a 2-copy pool: the drained pool discipline
    /// holds, and the draw completes a genuine win for the shape gate).
    /// Honors keep every take pool disjoint from the cycling seats.
    fn winner_tehais(pai: &str) -> Vec<String> {
        // The wins are 44s/EEE/678s/111p/222p (4s), 55s/EEE/678s/111p/222p
        // (5s), 66s/EEE/345s/789s/SSS (6s): each a genuine 4-melds-plus-
        // pair completion of the tail draw.
        let meld_tiles: Vec<&str> = match pai {
            "6s" => vec!["3s", "4s", "5s", "7s", "8s", "9s"],
            _ => vec!["6s", "7s", "8s", "1p", "1p", "1p"],
        };
        let pon_tiles: Vec<&str> = match pai {
            "6s" => vec!["S", "S", "S"],
            _ => vec!["2p", "2p", "2p"],
        };
        let mut hand = vec![pai, "E", "E", "E"];
        hand.extend(meld_tiles);
        hand.extend(pon_tiles);
        assert_eq!(hand.len(), 13);
        hand.iter().map(|t| t.to_string()).collect()
    }

    fn tehais() -> Vec<Vec<String>> {
        (0..4).map(cycling_hand).collect()
    }

    fn tehais_for_winner(seat: u8, pai: &str) -> Vec<Vec<String>> {
        let mut hands: Vec<Vec<String>> = (0..4).map(cycling_hand).collect();
        hands[seat as usize] = winner_tehais(pai);
        // Seat 1's cycling hand holds every simple 4s/5s/6s the winners
        // draw; without this swap the tail draw would be take #3 of a
        // 2-copy pool and break the drained pool discipline.
        if seat != 1 && ["4s", "5s", "6s"].contains(&pai) {
            hands[1] = hands[1]
                .iter()
                .map(|t| if t == pai { "9s".to_string() } else { t.clone() })
                .collect();
        }
        hands
    }

    /// Tail-drawn tsumo win: `draws_before` full turns (seats in rotation
    /// order), then the next seat in rotation draws and wins by tsumo.
    /// The win is genuine (tenpai tehais + completing draw) so the
    /// engine shape gate accepts it exactly like the oracle.
    fn tsumo_win(draws_before: usize) -> String {
        let seat = (draws_before % 4) as u8;
        let pai = tile_at(52 + draws_before);
        let tsumo = format!("{{\"type\":\"tsumo\",\"actor\":{seat},\"pai\":{pai:?}}}");
        let hora = format!(
            "{{\"type\":\"hora\",\"actor\":{seat},\"target\":{seat},\"tsumo\":true,\"deltas\":[8000,-2000,-2000,-4000]}}"
        );
        game_text_with_tail_and_tehais(draws_before, &format!("{tsumo}\n{hora}"), tehais_for_winner(seat, &pai))
    }

    fn game_text_with_tail(draws_before_tail: usize, tail: &str) -> String {
        game_text_with_tail_and_tehais(draws_before_tail, tail, tehais())
    }

    fn game_text_with_tail_and_tehais(
        draws_before_tail: usize,
        tail: &str,
        tehais: Vec<Vec<String>>,
    ) -> String {
        let tehais_json: Vec<String> = tehais
            .iter()
            .map(|hand| {
                let tiles: Vec<String> =
                    hand.iter().map(|t| format!("{t:?}")).collect();
                format!("[{}]", tiles.join(","))
            })
            .collect();
        let mut lines = vec![
            r#"{"type":"start_game"}"#.to_string(),
            format!(
                "{{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"dora_marker\":\"3m\",\"kyoku\":1,\"honba\":0,\"kyotaku\":0,\"oya\":0,\"scores\":[25000,25000,25000,25000],\"tehais\":[{}]}}",
                tehais_json.join(",")
            ),
        ];
        for turn in 0..draws_before_tail {
            let seat = turn % 4;
            let pai = tile_at(52 + turn);
            lines.push(format!(
                "{{\"type\":\"tsumo\",\"actor\":{seat},\"pai\":{pai:?}}}"
            ));
            lines.push(format!(
                "{{\"type\":\"dahai\",\"actor\":{seat},\"pai\":{pai:?},\"tsumogiri\":true}}"
            ));
        }
        lines.push(tail.to_string());
        lines.push(r#"{"type":"end_game"}"#.to_string());
        lines.join("\n") + "\n"
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hydra2-py-stream-test-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn open_stream(dir: &std::path::Path, batch: usize) -> ReplayStream {
        ReplayStream::open(StreamOpen {
            data_dirs: vec![dir.to_path_buf()],
            batch_rows: batch,
            workers: 0,
            queue: 4,
            split: "train".to_string(),
            spec_hash: "spec-test".to_string(),
        })
        .expect("open test stream")
    }

    fn drain(stream: &mut ReplayStream) -> Vec<u8> {
        let mut out = Vec::new();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let next = stream.next_into_raw(&mut buf).expect("next batch");
            out.extend_from_slice(&buf[..next.bytes_written]);
            if next.rows == 0 && next.games_consumed == 0 {
                break;
            }
        }
        out
    }

    #[test]
    fn serial_walk_stats_and_quarantine_taxonomy() {
        let dir = scratch_dir("walk");
        fs::write(dir.join("gA-good.jsonl"), tsumo_win(3)).unwrap();
        fs::write(dir.join("gB-good.jsonl"), tsumo_win(4)).unwrap();
        // Double ron on one discard: prescan quarantines before any row.
        let ron_tail = concat!(
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"E\"}\n",
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"E\",\"tsumogiri\":true}\n",
            "{\"type\":\"hora\",\"actor\":1,\"target\":0,\"deltas\":[8000,8000,-8000,-8000]}\n",
            "{\"type\":\"hora\",\"actor\":2,\"target\":0,\"deltas\":[8000,-8000,8000,-8000]}",
        );
        fs::write(
            dir.join("gC-double-ron.jsonl"),
            game_text_with_tail(0, ron_tail),
        )
        .unwrap();

        let mut stream = open_stream(&dir, 2);
        assert_eq!(stream.queue(), 4);
        let payload = drain(&mut stream);
        let stats = stream.stats();
        assert_eq!(stats.open_count, 1);
        assert_eq!(stats.games_ok, 2);
        assert_eq!(stats.games_quarantined, 1);
        let lines: Vec<&[u8]> = payload.split(|b| *b == b'\n').filter(|l| !l.is_empty()).collect();
        assert_eq!(stats.rows_out as usize, lines.len());
        assert!(!lines.is_empty());
        for line in &lines {
            let row: serde_json::Value = serde_json::from_slice(line).expect("row json");
            check_decision_envelope(&row).expect("envelope gate");
        }
        let codes: Vec<&str> = stream
            .quarantines()
            .iter()
            .map(|q| q.reason_code.as_str())
            .collect();
        assert_eq!(codes, vec!["double-ron"]);
        stream.close();
        assert!(stream.is_closed());
        stream.close();
        assert!(stream.next_into_raw(&mut vec![0u8; 64]).is_err());
    }

    #[test]
    fn two_passes_are_byte_identical() {
        let dir = scratch_dir("determinism");
        fs::write(dir.join("gA-good.jsonl"), tsumo_win(3)).unwrap();
        fs::write(dir.join("gB-good.jsonl"), tsumo_win(4)).unwrap();
        let mut first = open_stream(&dir, 2);
        let pass_one = drain(&mut first);
        let mut second = open_stream(&dir, 2);
        let pass_two = drain(&mut second);
        assert!(!pass_one.is_empty());
        assert_eq!(pass_one, pass_two);
    }

    #[test]
    fn small_buffer_drains_nothing() {
        let dir = scratch_dir("buffer");
        fs::write(dir.join("gA-good.jsonl"), tsumo_win(3)).unwrap();
        let mut stream = open_stream(&dir, 64);
        let mut tiny = vec![0u8; 8];
        let err = stream.next_into_raw(&mut tiny).expect_err("too small");
        assert!(matches!(err, StreamError::BufferTooSmall { .. }));
        let payload = drain(&mut stream);
        assert!(!payload.is_empty());
        assert_eq!(stream.stats().games_ok, 1);
    }

    #[test]
    fn workers_rejected_and_args_validated() {
        let dir = scratch_dir("validate");
        let workers = StreamOpen {
            data_dirs: vec![dir.clone()],
            batch_rows: 2,
            workers: 1,
            queue: 4,
            split: "train".to_string(),
            spec_hash: "spec-test".to_string(),
        };
        assert!(matches!(
            ReplayStream::open(workers),
            Err(StreamError::WorkersDeferred)
        ));
        let empty_dirs = StreamOpen {
            data_dirs: vec![],
            batch_rows: 2,
            workers: 0,
            queue: 4,
            split: "train".to_string(),
            spec_hash: "spec-test".to_string(),
        };
        assert!(matches!(
            ReplayStream::open(empty_dirs),
            Err(StreamError::Open(_))
        ));
    }

    #[test]
    fn envelope_rejects_leak_and_schema_fork() {
        let dir = scratch_dir("envelope");
        fs::write(dir.join("gA-good.jsonl"), tsumo_win(3)).unwrap();
        let mut stream = open_stream(&dir, 64);
        let payload = drain(&mut stream);
        let first: serde_json::Value =
            serde_json::from_slice(payload.split(|b| *b == b'\n').next().unwrap()).unwrap();
        let mut leak = first.clone();
        leak["wall"] = serde_json::json!([1, 2, 3]);
        assert!(matches!(
            check_decision_envelope(&leak),
            Err(StreamError::Envelope(_))
        ));
        let mut fork = first.clone();
        fork.as_object_mut().unwrap().remove("seat");
        assert!(matches!(
            check_decision_envelope(&fork),
            Err(StreamError::Envelope(_))
        ));
        check_decision_envelope(&first).expect("clean row passes");
    }

    /// S4 (G10a): 8-mutation negative battery. Every mutation fails closed
    /// on the Rust gate; the Python gate (`rust_stream._validated_rows`)
    /// rejects the same shapes (exact-13 set, forbidden scan top level +
    /// actor-doc, actor-as-string-to-object, chosen int-or-null). The split
    /// binding is a Python-only extra; rules/adapter/action-table digests
    /// are minted from pinned authorities (Slice 5, `crate::provenance`),
    /// asserted against the oracle pins in `tests/s5_provenance.rs`.
    #[test]
    fn envelope_negative_battery_eight_mutations() {
        let dir = scratch_dir("battery");
        fs::write(dir.join("gA-good.jsonl"), tsumo_win(3)).unwrap();
        let mut stream = open_stream(&dir, 64);
        let payload = drain(&mut stream);
        let clean: serde_json::Value =
            serde_json::from_slice(payload.split(|b| *b == b'\n').next().unwrap()).unwrap();
        check_decision_envelope(&clean).expect("clean row passes");

        // 1. Extra non-actor key.
        let mut mutated = clean.clone();
        mutated["extra_field"] = serde_json::json!(1);
        assert!(check_decision_envelope(&mutated).is_err(), "extra key");

        // 2. Missing actor field.
        let mut mutated = clean.clone();
        mutated.as_object_mut().unwrap().remove("seat");
        assert!(check_decision_envelope(&mutated).is_err(), "missing field");

        // 3. Forbidden key at top level.
        let mut mutated = clean.clone();
        mutated["wall"] = serde_json::json!([1, 2, 3]);
        assert!(check_decision_envelope(&mutated).is_err(), "top-level forbidden");

        // 4. Forbidden key inside the actor-observation document.
        let mut mutated = clean.clone();
        let text = mutated["actor_observation"].as_str().unwrap().to_string();
        let mut doc: serde_json::Value = serde_json::from_str(&text).unwrap();
        doc["wall"] = serde_json::json!([1, 2, 3]);
        mutated["actor_observation"] = serde_json::Value::String(doc.to_string());
        assert!(check_decision_envelope(&mutated).is_err(), "actor-doc forbidden");

        // 5. actor_observation not a string.
        let mut mutated = clean.clone();
        mutated["actor_observation"] = serde_json::json!(7);
        assert!(check_decision_envelope(&mutated).is_err(), "actor not a string");

        // 6. actor_observation string that parses but is not an object.
        let mut mutated = clean.clone();
        mutated["actor_observation"] = serde_json::Value::String("[1,2]".to_string());
        assert!(check_decision_envelope(&mutated).is_err(), "actor not an object");

        // 7. chosen_action_id of the wrong type (string, not int-or-null).
        let mut mutated = clean.clone();
        mutated["chosen_action_id"] = serde_json::json!("7");
        assert!(check_decision_envelope(&mutated).is_err(), "chosen wrong type");

        // 8. actor_observation string that does not parse at all.
        let mut mutated = clean.clone();
        mutated["actor_observation"] = serde_json::Value::String("{not json".to_string());
        assert!(check_decision_envelope(&mutated).is_err(), "actor unparsable");
    }

    /// S4 (G10b): sorted discovery + batch cursor. Files created in reverse
    /// name order still drain in sorted order, and batch=1 vs batch=N agree
    /// byte-for-byte with identical stats.
    #[test]
    fn sorted_discovery_and_batch_sizes_agree() {
        for batch in [1usize, 2, 64] {
            let dir = scratch_dir(&format!("batch-{batch}"));
            // Deliberately created out of order: discovery must sort.
            fs::write(dir.join("z-third.jsonl"), tsumo_win(5)).unwrap();
            fs::write(dir.join("a-first.jsonl"), tsumo_win(3)).unwrap();
            fs::write(dir.join("m-second.jsonl"), tsumo_win(4)).unwrap();
            let mut stream = open_stream(&dir, batch);
            let payload = drain(&mut stream);
            let stats = stream.stats();
            assert_eq!(stats.open_count, 1);
            assert_eq!(stats.games_ok, 3);
            assert_eq!(stats.games_quarantined, 0);
            assert_eq!(stats.rows_out as usize, payload.split(|b| *b == b'\n').filter(|l| !l.is_empty()).count());
            // Sorted order: the first drained row belongs to a-first.jsonl
            // (object id = file stem; game id falls back to game-<sha12>).
            let first: serde_json::Value =
                serde_json::from_slice(payload.split(|b| *b == b'\n').next().unwrap()).unwrap();
            assert_eq!(
                first.get("source_object_id").and_then(|v| v.as_str()),
                Some("a-first"),
                "discovery order is sorted, batch={batch}"
            );
            // Stash the batch=1 payload for the cross-batch comparison.
            std::fs::write(dir.join(format!("payload-{batch}.bin")), &payload).unwrap();
        }
        // Rebuild all three payloads in one place for the equality check.
        let dir = scratch_dir("batch-compare");
        fs::write(dir.join("z-third.jsonl"), tsumo_win(5)).unwrap();
        fs::write(dir.join("a-first.jsonl"), tsumo_win(3)).unwrap();
        fs::write(dir.join("m-second.jsonl"), tsumo_win(4)).unwrap();
        let mut one = open_stream(&dir, 1);
        let payload_one = drain(&mut one);
        let stats_one = one.stats();
        let mut many = open_stream(&dir, 64);
        let payload_many = drain(&mut many);
        let stats_many = many.stats();
        assert!(!payload_one.is_empty());
        assert_eq!(payload_one, payload_many, "batch=1 vs batch=N identical");
        assert_eq!(stats_one, stats_many, "stats agree across batch sizes");
    }

    /// S4 quarantine codebook (game-level, closed taxonomy): every input
    /// below quarantines with its named code and contributes zero rows.
    /// `truncated` (no `end_game`) maps to `framing` — the framer is the
    /// first gate, so a log that never reaches `end_game` never reaches the
    /// walk's terminal check. S7: valid 136-tile walls are ACCEPTED (never
    /// quarantined) — `codebook_game("wall-bearing")` builds a walled-ok
    /// game with zero rows, covered by the walled-ok test below.
    fn codebook_game(code: &str) -> String {
        let tehais = tehais();
        let tehais_json: Vec<String> = tehais
            .iter()
            .map(|hand| {
                let tiles: Vec<String> = hand.iter().map(|t| format!("{t:?}")).collect();
                format!("[{}]", tiles.join(","))
            })
            .collect();
        let kyoku = format!(
            "{{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"dora_marker\":\"3m\",\"kyoku\":1,\"honba\":0,\"kyotaku\":0,\"oya\":0,\"scores\":[25000,25000,25000,25000],\"tehais\":[{}]}}",
            tehais_json.join(",")
        );
        match code {
            "framing" => {
                // Blank line inside the payload.
                format!("{{\"type\":\"start_game\"}}\n\n{kyoku}\n{{\"type\":\"end_game\"}}\n")
            }
            "truncated" => {
                // Never reaches end_game: the framer reports `framing`.
                format!("{{\"type\":\"start_game\"}}\n{kyoku}\n")
            }
            "wall-bearing" => {
                // S7 walled-ok: valid permutation wall, zero decisions.
                let wall: Vec<String> = (0..136).map(|i| i.to_string()).collect();
                format!(
                    "{{\"type\":\"start_game\",\"wall\":[{}]}}\n{kyoku}\n{{\"type\":\"end_game\"}}\n",
                    wall.join(",")
                )
            }
            "bare-dora" => {
                format!(
                    "{{\"type\":\"start_game\"}}\n{kyoku}\n{{\"type\":\"dora\"}}\n{{\"type\":\"end_game\"}}\n"
                )
            }
            "double-ron" => game_text_with_tail(
                0,
                concat!(
                    "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"E\"}\n",
                    "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"E\",\"tsumogiri\":true}\n",
                    "{\"type\":\"hora\",\"actor\":1,\"target\":0,\"deltas\":[8000,8000,-8000,-8000]}\n",
                    "{\"type\":\"hora\",\"actor\":2,\"target\":0,\"deltas\":[8000,-8000,8000,-8000]}",
                ),
            ),
            "turn-order" => {
                // Decision before the first start_kyoku.
                "{\"type\":\"start_game\"}\n{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"1m\"}\n{\"type\":\"end_game\"}\n"
                    .to_string()
            }
            "tile-conservation" => {
                // Draw without a pai string after a valid kyoku header.
                format!(
                    "{{\"type\":\"start_game\"}}\n{kyoku}\n{{\"type\":\"tsumo\",\"actor\":0}}\n{{\"type\":\"end_game\"}}\n"
                )
            }
            "unknown-event" => {
                format!(
                    "{{\"type\":\"start_game\"}}\n{kyoku}\n{{\"type\":\"frobnicate\"}}\n{{\"type\":\"end_game\"}}\n"
                )
            }
            _ => panic!("unknown codebook code {code}"),
        }
    }

    #[test]
    fn quarantine_codebook_pins_codes_and_zero_partial_rows() {
        // (fixture code, expected quarantine reason code). S7: `wall-bearing`
        // is no longer a quarantine — valid walls replay (see
        // `walled_ok_games_replay_with_real_digest`).
        let cases = [
            ("framing", "framing"),
            ("truncated", "framing"),
            ("bare-dora", "bare-dora"),
            ("double-ron", "double-ron"),
            ("turn-order", "turn-order"),
            ("tile-conservation", "tile-conservation"),
            ("unknown-event", "unknown-event"),
        ];
        let dir = scratch_dir("codebook");
        fs::write(dir.join("good.jsonl"), tsumo_win(3)).unwrap();
        // S7 walled-ok companion: accepted with zero rows (not quarantined).
        fs::write(dir.join("walled-ok.jsonl"), codebook_game("wall-bearing")).unwrap();
        for (code, _) in &cases {
            fs::write(dir.join(format!("q-{code}.jsonl")), codebook_game(code)).unwrap();
        }
        let mut stream = open_stream(&dir, 2);
        let payload = drain(&mut stream);
        let stats = stream.stats();
        // good + walled-ok replay; the 7 codebook games quarantine.
        assert_eq!(stats.games_ok, 2);
        assert_eq!(stats.games_quarantined, cases.len() as u64);
        let mut codes: Vec<(String, String)> = stream
            .quarantines()
            .iter()
            .map(|q| (q.game_id.clone(), q.reason_code.clone()))
            .collect();
        codes.sort();
        let mut got: Vec<&str> = codes.iter().map(|(_, c)| c.as_str()).collect();
        got.sort();
        let mut expected: Vec<&str> = cases.iter().map(|(_, c)| *c).collect();
        expected.sort();
        assert_eq!(got, expected, "codebook codes pin 1:1");
        // Zero-partial-rows: every drained row passes the gate and none
        // belongs to a quarantined game.
        let quarantined_ids: std::collections::HashSet<&str> =
            codes.iter().map(|(g, _)| g.as_str()).collect();
        let lines: Vec<&[u8]> =
            payload.split(|b| *b == b'\n').filter(|l| !l.is_empty()).collect();
        assert!(!lines.is_empty(), "good game still emits rows");
        assert_eq!(stats.rows_out as usize, lines.len());
        for line in &lines {
            let row: serde_json::Value = serde_json::from_slice(line).expect("row json");
            check_decision_envelope(&row).expect("envelope gate");
            let game = row.get("game_id").and_then(|v| v.as_str()).unwrap_or("");
            assert!(
                !quarantined_ids.contains(game),
                "quarantined game {game} contributed a row"
            );
        }
        // Idempotent close: second close is a no-op success, reads fail.
        stream.close();
        assert!(stream.is_closed());
        stream.close();
        assert!(stream.next_into_raw(&mut vec![0u8; 64]).is_err());
        // Counters and quarantines stay readable at the Rust level.
        assert_eq!(stream.stats(), stats);
        assert_eq!(stream.quarantines().len(), cases.len());
    }
}
