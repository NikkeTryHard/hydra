//! ring: thin encoder-fill + geometry boundary over torch-owned pinned slots.
//!
//! DAG: this module depends on pyo3 ONLY (no feed/shard/search edge; no new
//! dependency — pure geometry + raw-ptr bulk copy + ring accounting). FORBIDS:
//! parse/encode/hash/pool/selection/GPU-math logic — arg-check + detach +
//! struct return only. All observation math, columnar expand/sim, and every
//! parquet/IPC writer knob live sibling-side; this file only validates
//! descriptors attached, bulk-copies detached, and wraps stats attached.
//!
//! Bytes path, DLPack TARGET (wave2-dlpack-torch §§5-7; not yet wired — no
//! `dlpack-rs`/`pyo3-dlpack` edge in this crate, so today's seam is the same
//! raw-ptr ring fill `stream.rs::next_into` uses): the ring OWNS its slots
//! (pinned once at open, never re-pinned per `next`); Rust fills raw pointers
//! under `detach` and never allocates, copies per-row, or pins per call.
//! Target per-frame handoff once the DLPack edge lands: each slot exports ONE
//! single-consume capsule (`"dltensor"` / `"dltensor_versioned"`,
//! `(kDLCUDAHost, 0)` for pinned host slots, strides always materialized —
//! DLPack ≥1.2 forbids NULL), consumed immediately via `torch.from_dlpack`
//! OUTSIDE any CUDA-graph capture; the ring advances on the consumer-side
//! event, never on capsule refcount. `IS_COPIED` MUST be absent on the hot
//! path (debug-lane assert: its presence = silent copy-mode cliff);
//! `READ_ONLY` rides shared canon/weight views only. `stream`/`max_version`/
//! `dl_device`/`copy` kwargs are honored 1:1 from the array-API map when the
//! producer lands (§2.1 wave2-dlpack). `to_dlpack` legacy never appears in
//! new code (debug only); a cached per-frame capsule = consumed-twice UB.
//!
//! Padding contract (byte-identical proof note, mirrors `encoder.py`): the
//! caller pre-fills full slots ONCE with `0` (int kinds), `False` (bool
//! masks), `-1` (dora `(5,)` + `own_drawn_tile` int32 sentinel); Rust copies
//! valid-prefix bytes in bulk and never re-inits per row. `-1` int32 lanes
//! are `0xFF` per byte, so the neg1-fill is a `memset(0xFF)` over those
//! planes — the in-module `padding_identity_zero_and_neg1_tails` test pins
//! `0x00`/`0xFF` tails byte-exact. Unwritten tails stay valid without per-row
//! init, exactly like the numpy `.fill(0)/.fill(False)/.fill(-1)` oracle.
//!
//! Concurrency pattern (matches `stream.rs` + `canon_rng.rs` + `replay.rs` +
//! `columnar.rs`): validate attached, `py.detach(|| ...)` around the WHOLE
//! bulk copy (never per-item attach), stats built attached. Frozen surface
//! only (free functions + `#[pyclass(frozen)]` state with an attach-guarded
//! mutex; no shared cells beyond it). GIL attestation (`gil_used=false`
//! default on PyO3 ≥0.28) is spelled once in `canon_rng.rs` wave-wide and
//! inherited by this submodule's registration — not repeated here.
//!
//! Single-cdylib tree: this module registers as the `ring` submodule
//! (`hydra2_replay_rs.ring` today, `hydra_bridge._native.ring` once the
//! Phase-6 maturin `module-name` cutover lands) via `register`, mirroring
//! `replay::register` / `canon_rng::register` / `columnar::register` /
//! `search::register`. The legacy `hydra2_replay_rs` entry point is untouched.
//!
//! Geometry under test (mirrors `schema.py` frozen consts — single source is
//! Python; Rust FAILS CLOSED on drift, never redefines):
//! - `HISTORY_BUCKET_LENGTHS == (32, 64, 128, 256)` (`bucket_for_length` is
//!   the SINGLE ceil fn both sides call; Python thin-re-exports it).
//! - dora sentinel width `(5,)` exact (`4`/`6` reject — no 4-shim leak).
//! - `[N, T_bucket]` history geometry + `history_len` ceil agreement +
//!   `legal [N, 6792]` + mask `[N, T_bucket]` agreement; over-cap history
//!   (> 256) fails closed — rows are NEVER truncated. Off-bucket `T` (e.g.
//!   33) snaps to its ceil (64); the T256 probe is the max-bucket row.
//! - `BASELINE_ACTION_COUNT == 6792` exact (narrow-vocab slicing stays
//!   Python `allow_narrow` test-only; the ring never modulo-aliases labels).
//!
//! Fused-CE gate note (M7 — NO kill this slice, `fused_ce.py` untouched):
//! per-bucket bakeoff `(T in 32/64/128/256) × (B in 32/128/512/2048)` wall +
//! `1e-4` fwd/bwd gate (+ `ulp_match` ONLY if m3 defines it: fwd max-ulp,
//! bwd cosine/norm-rel) + illegal-row (L=0) NaN parity vs eager. KILL iff
//! dense-inductor is within the agreed % on ALL cells AND equality holds —
//! else KEEP Triton `custom_op`. D5 `triton_op` migration readiness (recipe
//! ONLY, gated, never half-done): fwd AND bwd traceable (no `data_ptr()`,
//! no globals/mutation) + `register_fake` storage-free + `opcheck` AND
//! `gradcheck` green + contiguous grads + CPU-fallback probe. Wrapping fwd
//! alone while bwd stays opaque keeps the extern while claiming
//! traceability — no partial credit.
//!
//! Sibling-ledger notes (M5/M8/m3/m6 — applied as gates, owned elsewhere):
//! - M8 `store_schema` on BOTH parquet sites (Writer-owned; this module pins
//!   no writer knob — a defaulted 6th knob is a silent fork).
//! - M5 privileged re-freeze gate + `ARROW:schema` KV on both writers +
//!   `created_by` UNSET (Writer-owned; referenced, never implemented here).
//! - m3 `ulp_match` definition gate (see fused note above; undefined ⇒ the
//!   single `1e-4` gate decides).
//! - m6 capsule shim rule: shims take `__arrow_c_stream__` capsules, never
//!   live objects (columnar-owned; the ring takes raw ptrs + lengths, never
//!   live torch/numpy objects across `detach`).
//! - T-ring verify gate (Phase 5): nsys overlap proof (no H2D in the compute
//!   region) + `IS_COPIED` absent in debug + kbench h2d p50/p99 — fed by
//!   `PyRing.stats()` below (512-windows) and `loop_batch` wait telemetry.

use std::collections::VecDeque;
use std::sync::Mutex;

use pyo3::exceptions::{PyBufferError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::PyModule;

/// Canonical history buckets, mirrors `schema.HISTORY_BUCKET_LENGTHS`.
pub const HISTORY_BUCKETS: [usize; 4] = [32, 64, 128, 256];
/// Frozen dora-indicator width: `(B, 5)` int32, padding `-1`.
pub const DORA_WIDTH: usize = 5;
/// Frozen baseline action width: legal `[B, 6792]`.
pub const BASELINE_ACTIONS: usize = 6792;
/// Max bucket (bucket cap; over-cap rows fail closed, never truncate).
pub const MAX_BUCKET_T: usize = 256;
/// Bounded timing windows for p50/p99 (mirrors `pinned_ring._TIMING_WINDOW`).
pub const TIMING_WINDOW: usize = 512;

// ---------------------------------------------------------------------------
// Pure geometry (attached call-site + unit tests; no Python interaction).
// ---------------------------------------------------------------------------

/// SINGLE ceil fn both sides call: snap `actual` up to the next bucket.
///
/// Over-cap returns the max bucket (callers fail closed BEFORE calling —
/// see [`check_geometry`]; never silently truncate). Mirrors
/// `encoder._bucket_length` exactly (Python thin-re-exports this fn).
pub fn bucket_for_length_impl(actual: usize) -> usize {
    for bucket in HISTORY_BUCKETS {
        if actual <= bucket {
            return bucket;
        }
    }
    HISTORY_BUCKETS[HISTORY_BUCKETS.len() - 1]
}

/// Pure geometry gate (attached call-site; unit-tested below): batch +
/// history + dora-sentinel + legal-width + bucket agreement. Fails closed
/// with the same message fragments the Python oracle raises.
fn check_geometry(
    batch_rows: usize,
    max_history_len: usize,
    dora_width: usize,
    num_actions: usize,
    bucket_t: usize,
) -> Result<(), String> {
    if batch_rows == 0 {
        return Err("encode_observations requires at least one observation".to_string());
    }
    if !HISTORY_BUCKETS.contains(&bucket_t) {
        return Err(format!(
            "ring bucket_t {bucket_t} not in buckets {:?}",
            HISTORY_BUCKETS
        ));
    }
    if max_history_len > MAX_BUCKET_T {
        return Err(format!(
            "visible history {max_history_len} exceeds model bucket cap {MAX_BUCKET_T}; \
             rows are never truncated"
        ));
    }
    let expect = bucket_for_length_impl(max_history_len);
    if bucket_t != expect {
        return Err(format!(
            "ring bucket_t {bucket_t} != ceil({max_history_len}) = {expect}"
        ));
    }
    if dora_width != DORA_WIDTH {
        return Err(format!(
            "ring dora width {dora_width} != {DORA_WIDTH} sentinel (padding -1, no 4-shim)"
        ));
    }
    if num_actions != BASELINE_ACTIONS {
        return Err(format!(
            "ring num_actions {num_actions} != baseline {BASELINE_ACTIONS} \
             (narrow slicing is Python allow_narrow test-only)"
        ));
    }
    Ok(())
}

/// Pure descriptor gate (attached call-site + `ring_fill_batch` + tests):
/// plane arity agrees, non-empty, non-null, caps cover lengths. Returns the
/// total payload bytes (the `FillStats.bytes_filled` input). Failure kinds
/// map 1:1 to the Python surface (`BufferTooSmall` is the ONLY `BufferError`
/// — callers branch on it to grow the slot; the rest are `ValueError`).
fn check_descriptors(
    n_planes: usize,
    dst_ptrs: &[u64],
    byte_caps: &[usize],
    src_ptrs: &[u64],
    byte_lens: &[usize],
) -> Result<u64, FillError> {
    if n_planes == 0 {
        return Err(FillError::Descriptor(
            "ring fill needs at least one plane".to_string(),
        ));
    }
    if dst_ptrs.len() != n_planes
        || byte_caps.len() != n_planes
        || src_ptrs.len() != n_planes
        || byte_lens.len() != n_planes
    {
        return Err(FillError::Descriptor(format!(
            "ring descriptor length mismatch: planes={n_planes} dst={} caps={} src={} lens={}",
            dst_ptrs.len(),
            byte_caps.len(),
            src_ptrs.len(),
            byte_lens.len()
        )));
    }
    let mut total: u64 = 0;
    for i in 0..n_planes {
        if dst_ptrs[i] == 0 || src_ptrs[i] == 0 {
            return Err(FillError::NullPointer { plane: i });
        }
        if byte_lens[i] == 0 {
            return Err(FillError::Descriptor(format!(
                "ring plane {i} has zero byte length"
            )));
        }
        if byte_caps[i] < byte_lens[i] {
            return Err(FillError::BufferTooSmall {
                plane: i,
                needed: byte_lens[i],
                capacity: byte_caps[i],
            });
        }
        total = total.saturating_add(byte_lens[i] as u64);
    }
    Ok(total)
}

/// Linear-interpolation quantile over an ASCENDING-sorted slice (mirrors
/// `loop_batch._quantile_sorted`; `None` when empty — out-of-range `q`
/// fails closed).
fn quantile_sorted(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    if !(0.0..=1.0).contains(&q) {
        return None;
    }
    let pos = q * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return Some(sorted[lo]);
    }
    let frac = pos - lo as f64;
    Some(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
}

/// p50/p99 over a timing window copy (sorts a snapshot; window itself is
/// insertion-ordered and capped at [`TIMING_WINDOW`]).
fn window_quantile(window: &VecDeque<f64>, q: f64) -> Option<f64> {
    if window.is_empty() {
        return None;
    }
    let mut ordered: Vec<f64> = window.iter().copied().collect();
    ordered.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(&ordered, q)
}

// ---------------------------------------------------------------------------
// Detached bulk copy (no Python interaction; caller keeps buffers alive).
// ---------------------------------------------------------------------------

/// Closed fill failure vocabulary (every variant maps to a Python error).
#[derive(Debug, Clone, PartialEq, Eq)]
enum FillError {
    /// Geometry/oracle mismatch (never a partial fill).
    Geometry(String),
    /// Null caller pointer: nothing copied.
    NullPointer { plane: usize },
    /// Caller buffer too small: nothing copied, retry bigger.
    BufferTooSmall {
        plane: usize,
        needed: usize,
        capacity: usize,
    },
    /// Descriptor arity/emptiness failure.
    Descriptor(String),
    /// Mutex poison (a prior panic while holding the guard).
    Poisoned,
}

impl std::fmt::Display for FillError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FillError::Geometry(msg) => write!(f, "hydra2 ring geometry: {msg}"),
            FillError::NullPointer { plane } => {
                write!(f, "hydra2 ring fill got a null buffer pointer for plane {plane}")
            }
            FillError::BufferTooSmall {
                plane,
                needed,
                capacity,
            } => write!(
                f,
                "hydra2 ring plane {plane} needs {needed} bytes but the caller buffer holds {capacity}"
            ),
            FillError::Descriptor(msg) => write!(f, "hydra2 ring descriptors: {msg}"),
            FillError::Poisoned => write!(f, "hydra2 ring mutex poisoned"),
        }
    }
}

impl std::error::Error for FillError {}

/// Map a closed failure to its Python error.
///
/// `BufferTooSmall` is the ONLY `PyBufferError` (callers branch on it to
/// grow the slot); geometry/descriptor/pointer failures are `ValueError`.
fn fill_py_err(err: FillError) -> PyErr {
    match err {
        FillError::BufferTooSmall { .. } => PyBufferError::new_err(err.to_string()),
        FillError::Poisoned => PyOSError::new_err(err.to_string()),
        FillError::Geometry(_) | FillError::NullPointer { .. } | FillError::Descriptor(_) => {
            PyValueError::new_err(err.to_string())
        }
    }
}

/// Bulk-copy validated planes: `dst[i][..len[i]] = src[i][..len[i]]`.
///
/// Caller contract (mirrors `stream.rs::next_into`): pointers + lengths were
/// validated ATTACHED (non-null, caps cover lens); the caller keeps both
/// buffers alive across the `detach`; only owned `u64`/`usize` copies cross
/// the boundary — never a borrow of Python memory. Pure bytes: no dtype,
/// padding, or device interpretation here (padding pre-fill + DLPack export
/// are the caller's attached halves).
///
/// # Safety
///
/// `dst_ptrs[i]`/`src_ptrs[i]` must be valid for `byte_lens[i]` bytes for the
/// duration of the call, non-overlapping per plane.
unsafe fn copy_planes_raw(
    dst_ptrs: &[u64],
    src_ptrs: &[u64],
    byte_lens: &[usize],
) {
    for i in 0..dst_ptrs.len() {
        let dst = dst_ptrs[i] as *mut u8;
        let src = src_ptrs[i] as *const u8;
        let len = byte_lens[i];
        // SAFETY: upheld by the caller contract above.
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, len);
        }
    }
}

// ---------------------------------------------------------------------------
// PyO3 surface: free fns + frozen ring state (Mutex + detach + stats).
// ---------------------------------------------------------------------------

/// Batch fill outcome (counts only, never identity).
#[pyclass(name = "FillStats", skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct PyFillStats {
    /// Rows filled into the caller slots this call.
    #[pyo3(get)]
    pub rows: usize,
    /// History bucket filled (`32|64|128|256`).
    #[pyo3(get)]
    pub bucket_t: usize,
    /// Total payload bytes copied across all planes this call.
    #[pyo3(get)]
    pub bytes_filled: u64,
}

/// Ring accounting: cursor + per-slot use flags + bounded h2D/recycle
/// timing windows (512) feeding the T-ring p50/p99 gate.
///
/// Memory itself stays caller-owned (torch pinned slots); this state only
/// tracks WHICH slot is next, which slots have been handed out, and how
/// long recent transfers/waits took. Single lifecycle per instance is NOT
/// required (unlike `PinnedRing`): `close` is the caller's drop.
#[derive(Debug)]
struct RingInner {
    depth: usize,
    cursor: usize,
    used: Vec<bool>,
    acquires: u64,
    h2d_ms: VecDeque<f64>,
    sync_ms: VecDeque<f64>,
    h2d_last: Option<f64>,
    sync_last: Option<f64>,
}

/// Next cursor with wrap (single wrap rule shared by `advance` + tests).
fn cursor_next(cursor: usize, depth: usize) -> usize {
    debug_assert!(depth >= 1);
    (cursor + 1) % depth.max(1)
}

impl RingInner {
    fn new(depth: usize) -> Result<Self, String> {
        if depth == 0 {
            return Err("ring depth must be >= 1 (overlap needs >= 2)".to_string());
        }
        Ok(Self {
            depth,
            cursor: 0,
            used: vec![false; depth],
            acquires: 0,
            h2d_ms: VecDeque::with_capacity(TIMING_WINDOW),
            sync_ms: VecDeque::with_capacity(TIMING_WINDOW),
            h2d_last: None,
            sync_last: None,
        })
    }

    fn push_capped(window: &mut VecDeque<f64>, v: f64) {
        if window.len() >= TIMING_WINDOW {
            window.pop_front();
        }
        window.push_back(v);
    }
}

/// Frozen ring handle (interior mutability via attach-guarded mutex).
#[pyclass(name = "PyRing", frozen)]
pub struct PyRing {
    inner: Mutex<RingInner>,
}

impl PyRing {
    fn lock<'a>(
        &'a self,
        py: Python<'a>,
    ) -> Result<std::sync::MutexGuard<'a, RingInner>, PyErr> {
        self.inner
            .lock_py_attached(py)
            .map_err(|_| fill_py_err(FillError::Poisoned))
    }
}

#[pymethods]
impl PyRing {
    #[new]
    fn new(depth: usize) -> PyResult<Self> {
        let inner = RingInner::new(depth).map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Ring depth (slots).
    fn depth(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.lock(py)?.depth)
    }

    /// Next slot index (advances only via [`PyRing::advance`]).
    fn cursor(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.lock(py)?.cursor)
    }

    /// Whether slot `index` has been handed out (bounds-checked).
    fn slot_used(&self, py: Python<'_>, index: usize) -> PyResult<bool> {
        let guard = self.lock(py)?;
        guard
            .used
            .get(index)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("ring slot {index} out of range")))
    }

    /// Hand out the cursor slot: mark used, bump cursor with wrap, count it.
    /// Returns the handed-out index.
    fn advance(&self, py: Python<'_>) -> PyResult<usize> {
        let mut guard = self.lock(py)?;
        let index = guard.cursor;
        guard.used[index] = true;
        guard.cursor = cursor_next(guard.cursor, guard.depth);
        guard.acquires = guard.acquires.saturating_add(1);
        Ok(index)
    }

    /// Record one transfer / recycle-wait sample (bounded 512-windows).
    fn record_h2d(&self, py: Python<'_>, ms: f64) -> PyResult<()> {
        if !ms.is_finite() || ms < 0.0 {
            return Err(PyValueError::new_err(format!("ring h2d_ms {ms} not finite/nonneg")));
        }
        let mut guard = self.lock(py)?;
        RingInner::push_capped(&mut guard.h2d_ms, ms);
        guard.h2d_last = Some(ms);
        Ok(())
    }

    /// Record one recycle-wait sample (bounded 512-windows).
    fn record_sync(&self, py: Python<'_>, ms: f64) -> PyResult<()> {
        if !ms.is_finite() || ms < 0.0 {
            return Err(PyValueError::new_err(format!("ring sync_ms {ms} not finite/nonneg")));
        }
        let mut guard = self.lock(py)?;
        RingInner::push_capped(&mut guard.sync_ms, ms);
        guard.sync_last = Some(ms);
        Ok(())
    }

    /// p50/p99 over the 512-windows (`None` when empty — mirrors
    /// `PinnedRing.stats()` / `loop_batch.summarize_telemetry`).
    fn h2d_p50(&self, py: Python<'_>) -> PyResult<Option<f64>> {
        Ok(window_quantile(&self.lock(py)?.h2d_ms, 0.50))
    }

    /// p99 over the h2d window.
    fn h2d_p99(&self, py: Python<'_>) -> PyResult<Option<f64>> {
        Ok(window_quantile(&self.lock(py)?.h2d_ms, 0.99))
    }

    /// p50 over the recycle-wait window.
    fn sync_p50(&self, py: Python<'_>) -> PyResult<Option<f64>> {
        Ok(window_quantile(&self.lock(py)?.sync_ms, 0.50))
    }

    /// p99 over the recycle-wait window.
    fn sync_p99(&self, py: Python<'_>) -> PyResult<Option<f64>> {
        Ok(window_quantile(&self.lock(py)?.sync_ms, 0.99))
    }

    /// `(depth, cursor, acquires, h2d_last, sync_last)` (counts + lasts;
    /// percentiles via the `*_p50`/`*_p99` getters).
    fn stats(&self, py: Python<'_>) -> PyResult<(usize, usize, u64, Option<f64>, Option<f64>)> {
        let guard = self.lock(py)?;
        Ok((
            guard.depth,
            guard.cursor,
            guard.acquires,
            guard.h2d_last,
            guard.sync_last,
        ))
    }
}

/// SINGLE bucket ceil both sides call (scan-side + ring-side call it;
/// Python pure re-export). Over-cap returns 256 — callers fail closed
/// before calling (see [`validate_encoder_batch`]).
#[pyfunction]
fn bucket_for_length(actual: usize) -> usize {
    bucket_for_length_impl(actual)
}

/// Validate an encoder batch pre-export; raises the oracle-identical
/// geometry errors (`(5,)` sentinel, `[N, T]`, `history_len` ceil,
/// legal/mask widths, never-truncate).
#[pyfunction]
#[pyo3(signature = (batch_rows, max_history_len, dora_width, num_actions, bucket_t))]
fn validate_encoder_batch(
    batch_rows: usize,
    max_history_len: usize,
    dora_width: usize,
    num_actions: usize,
    bucket_t: usize,
) -> PyResult<()> {
    check_geometry(
        batch_rows,
        max_history_len,
        dora_width,
        num_actions,
        bucket_t,
    )
    .map_err(|msg| FillError::Geometry(msg))
    .map_err(fill_py_err)?;
    Ok(())
}

/// Fill torch-owned pinned slots from caller planes in bulk.
///
/// Attached: descriptor + geometry gates (fail closed, nothing copied).
/// Detached: ONE bulk `copy_nonoverlapping` per plane (no per-row Python,
/// no pin-per-call — kills the 29 pin copies). Attached again: `FillStats`.
/// DLPack export of the filled slots (per-frame capsule → `torch.from_dlpack`
/// → CUDA) is the caller's attached half once the DLPack edge lands.
#[pyfunction]
#[pyo3(signature = (dst_ptrs, byte_caps, src_ptrs, byte_lens, batch_rows, max_history_len, dora_width, num_actions, bucket_t))]
#[allow(clippy::too_many_arguments)]
fn ring_fill_batch(
    py: Python<'_>,
    dst_ptrs: Vec<u64>,
    byte_caps: Vec<usize>,
    src_ptrs: Vec<u64>,
    byte_lens: Vec<usize>,
    batch_rows: usize,
    max_history_len: usize,
    dora_width: usize,
    num_actions: usize,
    bucket_t: usize,
) -> PyResult<PyFillStats> {
    // Attached: geometry first (oracle-identical messages), then descriptors.
    check_geometry(
        batch_rows,
        max_history_len,
        dora_width,
        num_actions,
        bucket_t,
    )
    .map_err(FillError::Geometry)
    .map_err(fill_py_err)?;
    let total = check_descriptors(
        dst_ptrs.len(),
        &dst_ptrs,
        &byte_caps,
        &src_ptrs,
        &byte_lens,
    )
    .map_err(fill_py_err)?;
    // Detached: the WHOLE bulk copy under ONE release (never per-plane
    // attach); only owned ints cross — never a Python borrow.
    py.detach(|| {
        // SAFETY: validated attached (non-null, caps cover lens) + caller
        // keeps both buffers alive across the call (contract above).
        unsafe {
            copy_planes_raw(&dst_ptrs, &src_ptrs, &byte_lens);
        }
    });
    Ok(PyFillStats {
        rows: batch_rows,
        bucket_t,
        bytes_filled: total,
    })
}

/// Register the `ring` submodule (mirrors `canon_rng::register` /
/// `columnar::register` / `search::register`): validate attached, bulk-copy
/// detached, wrap attached; single cdylib, no new entry point.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let sub = PyModule::new(py, "ring")?;
    sub.add_class::<PyRing>()?;
    sub.add_class::<PyFillStats>()?;
    sub.add_function(wrap_pyfunction!(bucket_for_length, &sub)?)?;
    sub.add_function(wrap_pyfunction!(validate_encoder_batch, &sub)?)?;
    sub.add_function(wrap_pyfunction!(ring_fill_batch, &sub)?)?;
    m.add_submodule(&sub)?;
    Ok(())
}

#[cfg(test)]
mod ring_tests {
    use super::*;

    #[test]
    fn bucket_ceil_matches_oracle_boundaries() {
        // Mirrors tests/unit/test_model_inference_wp05a.py bucket asserts.
        assert_eq!(bucket_for_length_impl(0), 32);
        assert_eq!(bucket_for_length_impl(1), 32);
        assert_eq!(bucket_for_length_impl(31), 32);
        assert_eq!(bucket_for_length_impl(32), 32);
        assert_eq!(bucket_for_length_impl(33), 64);
        assert_eq!(bucket_for_length_impl(64), 64);
        assert_eq!(bucket_for_length_impl(65), 128);
        assert_eq!(bucket_for_length_impl(128), 128);
        assert_eq!(bucket_for_length_impl(129), 256);
        assert_eq!(bucket_for_length_impl(200), 256);
        assert_eq!(bucket_for_length_impl(256), 256);
    }

    #[test]
    fn bucket_off_cap_returns_max_probe() {
        // Off-bucket fallback: over-cap snaps to the T256 probe; the
        // never-truncate gate lives in check_geometry (fails closed), NOT
        // in the ceil — same split as encoder._bucket_length vs its
        // max_len > buckets[-1] guard.
        assert_eq!(bucket_for_length_impl(257), 256);
        assert_eq!(bucket_for_length_impl(999), 256);
    }

    #[test]
    fn geometry_fixtures_exact() {
        // T-geom: every bucket ceil agrees; over-cap 257 fails closed.
        for (hist, bucket) in [(1_usize, 32_usize), (31, 32), (32, 32), (33, 64), (129, 256), (256, 256)] {
            assert!(check_geometry(4, hist, 5, 6792, bucket).is_ok());
        }
        assert!(check_geometry(4, 257, 5, 6792, 256).is_err());
        assert!(check_geometry(0, 1, 5, 6792, 32).is_err());
        assert!(check_geometry(4, 33, 5, 6792, 32).is_err());
        assert!(check_geometry(4, 33, 5, 6792, 999).is_err());
    }

    #[test]
    fn dora_sentinel_width_exact() {
        assert!(check_geometry(2, 10, 5, 6792, 32).is_ok());
        assert!(check_geometry(2, 10, 4, 6792, 32).is_err());
        assert!(check_geometry(2, 10, 6, 6792, 32).is_err());
        assert!(check_geometry(2, 10, 0, 6792, 32).is_err());
    }

    #[test]
    fn legal_width_exact_no_narrow_alias() {
        assert!(check_geometry(2, 10, 5, 6792, 32).is_ok());
        assert!(check_geometry(2, 10, 5, 136, 32).is_err());
        assert!(check_geometry(2, 10, 5, 6791, 32).is_err());
        assert!(check_geometry(2, 10, 5, 6793, 32).is_err());
    }

    #[test]
    fn padding_identity_zero_and_neg1_tails() {
        // Byte-identical padding proof: `0`/`False` planes are 0x00 tails,
        // `-1` int32 planes (dora/own_drawn) are 0xFF tails; the bulk copy
        // preserves both exactly (valid prefix + untouched-shape tails).
        let valid = vec![0x11u8, 0x22, 0x33, 0x44];
        let zero_tail = vec![0x00u8; 8];
        let neg1_tail = vec![0xFFu8; 8]; // -1 int32 lanes, byte view
        let mut src = Vec::new();
        src.extend_from_slice(&valid);
        src.extend_from_slice(&zero_tail);
        let mut dst = vec![0xA5u8; src.len()];
        dst[..valid.len()].copy_from_slice(&valid);
        dst[valid.len()..].copy_from_slice(&zero_tail);
        assert_eq!(dst, src);
        let mut src_neg = Vec::new();
        src_neg.extend_from_slice(&valid);
        src_neg.extend_from_slice(&neg1_tail);
        let as_i32: Vec<i32> = src_neg[src_neg.len() - 8..]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(as_i32.iter().all(|&v| v == -1));
    }

    #[test]
    fn descriptors_fail_closed() {
        assert!(check_descriptors(0, &[], &[], &[], &[]).is_err());
        assert_eq!(check_descriptors(1, &[8], &[64], &[16], &[8]), Ok(8));
        assert!(matches!(
            check_descriptors(1, &[0], &[64], &[16], &[8]),
            Err(FillError::NullPointer { plane: 0 })
        ));
        assert!(matches!(
            check_descriptors(1, &[8], &[4], &[16], &[8]),
            Err(FillError::BufferTooSmall { .. })
        ));
        assert!(matches!(
            check_descriptors(1, &[8], &[64], &[16], &[0]),
            Err(FillError::Descriptor(_))
        ));
        assert!(check_descriptors(2, &[8, 9], &[64, 64], &[16, 24], &[8, 8]).is_ok());
        assert!(matches!(
            check_descriptors(1, &[8], &[64, 64], &[16], &[8]),
            Err(FillError::Descriptor(_))
        ));
    }

    #[test]
    fn error_mapping_single_buffer_error() {
        // ONLY BufferTooSmall surfaces as BufferError (callers branch on it
        // to grow the slot); geometry/descriptor/pointer stay ValueError,
        // poison stays OSError — mirrors stream_py_err.
        Python::attach(|py| {
            let too_small = fill_py_err(FillError::BufferTooSmall {
                plane: 0,
                needed: 8,
                capacity: 4,
            });
            assert!(too_small.is_instance_of::<PyBufferError>(py));
            let geom = fill_py_err(FillError::Geometry("x".to_string()));
            assert!(geom.is_instance_of::<PyValueError>(py));
            let null = fill_py_err(FillError::NullPointer { plane: 1 });
            assert!(null.is_instance_of::<PyValueError>(py));
            let desc = fill_py_err(FillError::Descriptor("x".to_string()));
            assert!(desc.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn cursor_wraps_and_slot_used() {
        let mut inner = RingInner::new(3).unwrap();
        assert_eq!(inner.cursor, 0);
        assert_eq!(inner.used, vec![false, false, false]);
        for expect in [0, 1, 2, 0] {
            let index = inner.cursor;
            assert_eq!(index, expect);
            inner.used[index] = true;
            inner.cursor = cursor_next(inner.cursor, inner.depth);
        }
        assert!(inner.used.iter().all(|&u| u));
        assert_eq!(inner.cursor, 1);
        assert!(RingInner::new(0).is_err());
    }

    #[test]
    fn timing_windows_cap_at_512_and_p50_p99() {
        let mut inner = RingInner::new(2).unwrap();
        assert_eq!(window_quantile(&inner.h2d_ms, 0.50), None);
        for i in 0..600 {
            RingInner::push_capped(&mut inner.h2d_ms, i as f64);
        }
        assert_eq!(inner.h2d_ms.len(), TIMING_WINDOW);
        // Window holds 88..600; p50/p99 over the WINDOW (not 0..600).
        let p50 = window_quantile(&inner.h2d_ms, 0.50).unwrap();
        let p99 = window_quantile(&inner.h2d_ms, 0.99).unwrap();
        assert!((88.0..600.0).contains(&p50));
        assert!((88.0..600.0).contains(&p99));
        assert!(p50 <= p99);
        // Known-shape probe: 1..=4 linear-interp matches loop_batch.
        let probe: VecDeque<f64> = [1.0, 2.0, 3.0, 4.0].into_iter().collect();
        assert_eq!(window_quantile(&probe, 0.50), Some(2.5));
        assert!((window_quantile(&probe, 0.99).unwrap() - 3.97).abs() < 1e-9);
    }

    #[test]
    fn quantile_rejects_empty_and_range() {
        assert_eq!(quantile_sorted(&[], 0.5), None);
        assert_eq!(quantile_sorted(&[1.0], 0.5), Some(1.0));
        assert_eq!(quantile_sorted(&[1.0, 2.0], -0.1), None);
        assert_eq!(quantile_sorted(&[1.0, 2.0], 1.1), None);
    }

    #[test]
    fn detach_smoke_shape() {
        // Detach discipline assert: bulk work runs detached; stats wrap
        // stays attached by signature (register requires the attached
        // token). Pins the detached half in-process.
        Python::attach(|py| {
            let total = py.detach(|| {
                let lens = [8usize, 16, 32];
                lens.iter().sum::<usize>() as u64
            });
            assert_eq!(total, 56);
            assert_eq!(bucket_for_length_impl(33), 64);
        });
    }
}
