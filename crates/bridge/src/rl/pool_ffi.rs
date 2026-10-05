//! rl/pool_ffi: thin batch-only PyO3 boundary over `super::pool`.
//!
//! DAG: pyo3 ONLY (+ `super::pool` synchronous calls). No torch/CUDA/libcuda
//! dependency: CUDA-event completion is observed by calling the owner-supplied
//! event objects' `query()` method (attached, batch-boundary only), never by
//! linking CUDA. No per-row Python callbacks; workers never touch Python.
//!
//! Contract (MAIN's GPU-owner bindings):
//! - `NativePool.open(profile_json, context_json, derivation_hex)`.
//! - `register_buffers(T, slot_id, host_input, host_output)`: any
//!   buffer-protocol object (`numpy.uint8`, `bytearray`, writable
//!   `memoryview`); exact-capacity sizes matched via `buffer_geometry`
//!   (aligned totals are not divisible per row, so sizes match exactly);
//!   strong `Py<PyAny>` owners retained until `close`; slot ranges must not
//!   overlap.
//! - `request_ready_fill(slot_id) -> n_valid` (compat: admit + blocking
//!   take); `admit_fill(slot_id)` + `take_fill(slot_id) -> ticket | None`
//!   for the hot loop (dispatched native encode overlaps GPU work).
//!   Staging is a worker-side vector, then one bounded copy into the
//!   registered buffer (attached). Never a borrow of Python memory across
//!   detach.
//! - `peek_ready() -> lease | None` (nonblocking, work-conserving).
//! - lease properties `slot_id, bucket_t, n_valid, input_bytes,
//!   output_bytes, input_layout`; layout entries are
//!   `(name, dtype_string, shape_tuple, byte_offset)` over the ACTIVE prefix
//!   with dtype-aligned bases (`uniform` f64 last). Cached torch views wrap
//!   these offsets with the real active B.
//! - `lease_descriptor(lease) -> JSON` (staged rows + legal signatures;
//!   scheduling/verification only, never the hot loop).
//! - `mark_h2d_issued/mark_inference_queued/mark_d2h_issued` record owner
//!   phases; `mark_host_ready(compute_event, d2h_event)` queries BOTH actual
//!   events (attached) and records HOST_READY only when both report complete.
//! - `commit_actions(lease) -> rows_captured`: copies the active output
//!   planes out (attached), validates + captures + retains + dispatches
//!   applies under detach, releases the slot immediately, and returns the
//!   count (never waits for engine applies). Output is three contiguous
//!   planes (`selected int64[B] @ 0`, `logp float32[B] @ 8B`, `value4
//!   float32[B,4] @ 12B`, total `28B`); consumers must use byte-stride-aware
//!   reads.
//! - `buffer_geometry(T, capacity) -> (input_bytes, output_bytes)` cold helper.
//! - `poll/wait_for_progress` report poison, leases, pending applies, and
//!   collection counts (`captured_rows`, `terminal_games`,
//!   `retained_ready_rows`, `last_tape_handle`) for the seal loop.
//! - `completions() -> JSON` (per-game terminal/captured/outcome facts);
//!   `export_tape() -> u64` (generation-end tape handle for the seal; rows
//!   never cross Python).
//! - `quiesce/drain/snapshot/restore/refill_staged/close` are owner-level;
//!   `quiesce` joins admitted applies while retaining READY; `snapshot`
//!   persists full captured rows + retained READY reservations;
//!   `close` rejects live leases/work, joins workers, and only then
//!   releases the registered host owners.

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::pool::{
    GameSnapshot, Lease, NativePool, PoolProfile, PoolSnapshot, SlotPhase, SlotSnapshot,
    StagedRowSnapshot, StoredAction, buffer_geometry as pool_buffer_geometry,
};

/// Fail-closed Python error for every pool boundary rejection.
fn pool_error(detail: impl Into<String>) -> PyErr {
    PyValueError::new_err(format!("rl pool rejected: {}", detail.into()))
}
/// Parse the exact three-key live tape binding (attached, before detach):
/// digest-shaped `pipeline_spec_id`/`behavior_policy_id plus an integer
/// `generation` (bools reject: `as_u64` yields None on Bool).
fn parse_live_binding(json: &[u8]) -> PyResult<super::trajectory::LiveTapeBinding> {
    let value: serde_json::Value = serde_json::from_slice(json)
        .map_err(|e| pool_error(format!("live binding not JSON: {e}")))?;
    let map = value
        .as_object()
        .ok_or_else(|| pool_error("live binding must be an object"))?;
    for key in ["pipeline_spec_id", "generation", "behavior_policy_id"] {
        if !map.contains_key(key) {
            return Err(pool_error(format!("live binding lacks {key}")));
        }
    }
    if map.len() != 3 {
        return Err(pool_error(
            "live binding carries exactly pipeline_spec_id/generation/behavior_policy_id",
        ));
    }
    let pipeline_spec_id = map
        .get("pipeline_spec_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| pool_error("live binding pipeline_spec_id must be a string"))?;
    let behavior_policy_id = map
        .get("behavior_policy_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| pool_error("live binding behavior_policy_id must be a string"))?;
    super::trajectory::parse_digest_text(pipeline_spec_id).map_err(|e| pool_error(e.0))?;
    super::trajectory::parse_digest_text(behavior_policy_id).map_err(|e| pool_error(e.0))?;
    let generation = map
        .get("generation")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| pool_error("live binding generation must be a non-bool integer"))?;
    Ok(super::trajectory::LiveTapeBinding {
        pipeline_spec_id: pipeline_spec_id.to_string(),
        generation,
        behavior_policy_id: behavior_policy_id.to_string(),
    })
}

/// Whether two host spans overlap (exact byte ranges, overflow-safe).
/// Empty spans occupy no bytes and never overlap.
fn ranges_overlap(a_ptr: usize, a_len: usize, b_ptr: usize, b_len: usize) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let Some(a_end) = a_ptr.checked_add(a_len) else {
        return true;
    };
    let Some(b_end) = b_ptr.checked_add(b_len) else {
        return true;
    };
    a_ptr < b_end && b_ptr < a_end
}
/// Registered host buffer pair (strong owners; exact-capacity geometry
/// validated at registration against the aligned `buffer_geometry`).
struct RegisteredBuffers {
    input: Py<PyAny>,
    output: Py<PyAny>,
    input_len: usize,
    output_len: usize,
    input_ptr: usize,
    output_ptr: usize,
    bucket_t: usize,
}

/// Thin pool handle: the native pool plus registered host owners.
#[pyclass]
pub struct PyNativePool {
    pool: Arc<NativePool>,
    buffers: HashMap<u64, RegisteredBuffers>,
    closed: bool,
}

/// Byte length + base address of a `uint8` buffer-protocol object
/// (`PyBuffer::<u8>::get` already rejects non-byte formats). Index `[0]`
/// (not `&[]`): `get_ptr` needs exactly `ndim` indices, and a 1-D host
/// slab has `ndim == 1`; `&[]` reads out of bounds and segfaults inside
/// `PyBuffer_GetPointer`.
fn buffer_span(obj: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
    let buffer = PyBuffer::<u8>::get(obj)
        .map_err(|e| pool_error(format!("buffer protocol rejected: {e}")))?;
    if !buffer.is_c_contiguous() {
        return Err(pool_error("host buffers must be C-contiguous"));
    }
    if buffer.readonly() {
        return Err(pool_error("host buffers must be writable"));
    }
    let len = buffer.len_bytes();
    if len == 0 {
        return Ok((0, 0));
    }
    Ok((buffer.get_ptr(&[0]) as usize, len))
}

/// Bounded active-prefix write through the validated raw u8 span.
/// `copy_from_slice` demands exact buffer length, but narrow-B active
/// prefixes are shorter than registered capacity by construction, so the
/// owner contract is prefix writes, never full-capacity transfers.
fn write_prefix(buffer: &PyBuffer<u8>, active: &[u8]) -> PyResult<()> {
    if buffer.readonly() {
        return Err(pool_error("host buffer is read-only"));
    }
    if !buffer.is_c_contiguous() {
        return Err(pool_error("host buffer must be C-contiguous"));
    }
    if active.len() > buffer.len_bytes() {
        return Err(pool_error("active prefix exceeds host buffer"));
    }
    if active.is_empty() {
        return Ok(());
    }
    // `[0]` for the 1-D base address (see `buffer_span`).
    let dst = buffer.get_ptr(&[0]) as *mut u8;
    // SAFETY: `dst..dst+len` lies inside the validated C-contiguous span;
    // plain u8 copy; the GIL is held (attached) and the strong buffer
    // owner is retained by the caller, so no Python resize or free can
    // interleave.
    unsafe {
        std::ptr::copy_nonoverlapping(active.as_ptr(), dst, active.len());
    }
    Ok(())
}

/// Bounded active-prefix read through the validated raw u8 span (output
/// slabs: `need` is the live `n_valid` plane total, often narrower than
/// registered capacity).
fn read_prefix(buffer: &PyBuffer<u8>, need: usize) -> PyResult<Vec<u8>> {
    if !buffer.is_c_contiguous() {
        return Err(pool_error("host buffer must be C-contiguous"));
    }
    if need > buffer.len_bytes() {
        return Err(pool_error("host buffer shrank after registration"));
    }
    let mut out = vec![0u8; need];
    if need == 0 {
        return Ok(out);
    }
    // `[0]` for the 1-D base address (see `buffer_span`).
    let src = buffer.get_ptr(&[0]) as *const u8;
    // SAFETY: same stability argument as `write_prefix`.
    unsafe {
        std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), need);
    }
    Ok(out)
}


#[pymethods]
impl PyNativePool {
    /// Open + cold-check the cohort (`derivation_hex` is the frozen
    /// row-derivation digest, caller-owned recipe, `sha256:`-prefixed hex).
    #[staticmethod]
    fn open(profile_json: &str, context_json: &str, derivation_hex: &str) -> PyResult<Self> {
        let profile =
            PoolProfile::parse(profile_json, context_json).map_err(|e| pool_error(e.0))?;
        let derivation = parse_hex32(derivation_hex)?;
        let pool = NativePool::open(profile, derivation).map_err(|e| pool_error(e.0))?;
        Ok(Self { pool, buffers: HashMap::new(), closed: false })
    }
    /// Live open: shares `open`, then installs the immutable live collector
    /// binding (generation/behavior compared against the profile here;
    /// pipeline stored and re-compared at `drain_tape`).
    #[staticmethod]
    fn open_live(
        profile_json: &str,
        context_json: &str,
        derivation_hex: &str,
        live_binding_json: Vec<u8>,
    ) -> PyResult<Self> {
        let profile =
            PoolProfile::parse(profile_json, context_json).map_err(|e| pool_error(e.0))?;
        let derivation = parse_hex32(derivation_hex)?;
        let binding = parse_live_binding(&live_binding_json)?;
        let pool =
            NativePool::open_live(profile, derivation, binding).map_err(|e| pool_error(e.0))?;
        Ok(Self { pool, buffers: HashMap::new(), closed: false })
    }

    /// Restore a pool from a JSON snapshot (slots adopted with epochs;
    /// buffers re-registered by the owner afterwards).
    #[staticmethod]
    fn restore(
        profile_json: &str,
        context_json: &str,
        snapshot_json: &str,
        derivation_hex: &str,
    ) -> PyResult<Self> {
        let profile =
            PoolProfile::parse(profile_json, context_json).map_err(|e| pool_error(e.0))?;
        let snapshot = parse_snapshot(snapshot_json)?;
        let derivation = parse_hex32(derivation_hex)?;
        let pool =
            NativePool::restore(profile, &snapshot, derivation).map_err(|e| pool_error(e.0))?;
        Ok(Self { pool, buffers: HashMap::new(), closed: false })
    }

    /// Register exact-capacity host buffers for a `(T, slot_id)` lane.
    fn register_buffers(
        &mut self,
        bucket_t: usize,
        slot_id: u64,
        host_input: Bound<'_, PyAny>,
        host_output: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let (in_ptr, in_len) = buffer_span(&host_input)?;
        let (out_ptr, out_len) = buffer_span(&host_output)?;
        if ranges_overlap(in_ptr, in_len, out_ptr, out_len) {
            return Err(pool_error("input/output spans must not overlap"));
        }
        for (other_id, other) in &self.buffers {
            if *other_id == slot_id {
                continue;
            }
            if ranges_overlap(in_ptr, in_len, other.input_ptr, other.input_len)
                || ranges_overlap(in_ptr, in_len, other.output_ptr, other.output_len)
                || ranges_overlap(out_ptr, out_len, other.input_ptr, other.input_len)
                || ranges_overlap(out_ptr, out_len, other.output_ptr, other.output_len)
            {
                return Err(pool_error(format!(
                    "slot {slot_id} spans overlap registered slot {other_id}"
                )));
            }
        }
        // Capacity from exact-size match (aligned input totals are not a
        // multiple of a per-row stride, so division is unsound; the match
        // is unique because aligned totals grow strictly with capacity).
        let capacity = capacity_for(bucket_t, in_len, out_len)?;
        self.pool.attach_slot(slot_id, bucket_t, capacity).map_err(|e| pool_error(e.0))?;
        self.buffers.insert(slot_id, RegisteredBuffers {
            input: host_input.unbind(),
            output: host_output.unbind(),
            input_len: in_len,
            output_len: out_len,
            input_ptr: in_ptr,
            output_ptr: out_ptr,
            bucket_t,
        });
        Ok(())
    }

    /// Admission-scheduled typed fill; returns staged row count (compat
    /// wrapper: admit + blocking take; prefer `admit_fill` + `take_fill`
    /// on the hot loop so native encoding overlaps GPU work).
    fn request_ready_fill(&mut self, py: Python<'_>, slot_id: u64) -> PyResult<usize> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let registered = self.buffers.get(&slot_id).ok_or_else(|| {
            pool_error(format!("slot {slot_id} has no registered buffers"))
        })?;
        let input_len = registered.input_len;
        let output_capacity = registered.output_len;
        // Admit attached (fast lock ops + thread spawn), then block on the
        // worker detached into staging (pure Rust, no Python borrows).
        let pool = Arc::clone(&self.pool);
        pool.admit_fill(slot_id, output_capacity).map_err(|e| pool_error(e.0))?;
        let mut staging = vec![0u8; input_len];
        let ticket = py
            .detach(|| pool.take_fill_blocking(slot_id, &mut staging))
            .map_err(|e| pool_error(e.0))?;
        let active = super::observation::active_input_bytes(ticket.bucket_t, ticket.n_valid);
        // One bounded copy into the registered buffer (attached).
        let registered = self.buffers.get(&slot_id).ok_or_else(|| {
            pool_error(format!("slot {slot_id} buffers released before fill copy"))
        })?;
        let input_obj = registered.input.bind(py);
        let buffer = PyBuffer::<u8>::get(&input_obj)
            .map_err(|e| pool_error(format!("input buffer lost: {e}")))?;
        write_prefix(&buffer, &staging[..active])?;
        Ok(ticket.n_valid)
    }

    /// Admit a dispatched fill for one lane (nonblocking; the worker
    /// encodes off the CUDA owner thread). Collect with `take_fill`.
    fn admit_fill(&self, slot_id: u64) -> PyResult<()> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let registered = self.buffers.get(&slot_id).ok_or_else(|| {
            pool_error(format!("slot {slot_id} has no registered buffers"))
        })?;
        self.pool
            .admit_fill(slot_id, registered.output_len)
            .map_err(|e| pool_error(e.0))
    }

    /// Nonblocking take of the admitted fill: `None` while the worker runs
    /// (do GPU work and retry), otherwise the ticket descriptor
    /// (`n_valid`, `chunk`, `logical_round`, `bucket_t`, `slot_id`).
    fn take_fill(&mut self, py: Python<'_>, slot_id: u64) -> PyResult<Option<Py<PyAny>>> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        // Readiness gate BEFORE any allocation: a None poll costs one
        // lock, never a megabyte memset on the CUDA-owner thread.
        if !self.pool.fill_ready(slot_id).map_err(|e| pool_error(e.0))? {
            self.pool.bump_take_miss();
            return Ok(None);
        }
        let registered = self.buffers.get(&slot_id).ok_or_else(|| {
            pool_error(format!("slot {slot_id} has no registered buffers"))
        })?;
        let input_len = registered.input_len;
        let pool = Arc::clone(&self.pool);
        let mut staging = vec![0u8; input_len];
        let taken = pool
            .take_fill(slot_id, &mut staging)
            .map(|taken| taken.map(|ticket| (ticket, staging)))
            .map_err(|e| pool_error(e.0))?;
        let (ticket, staging) = match taken {
            None => {
                self.pool.bump_take_miss();
                return Ok(None);
            }
            Some(pair) => pair,
        };
        self.pool.bump_take_hit();
        let active = super::observation::active_input_bytes(ticket.bucket_t, ticket.n_valid);
        let registered = self.buffers.get(&slot_id).ok_or_else(|| {
            pool_error(format!("slot {slot_id} buffers released before fill copy"))
        })?;
        let input_obj = registered.input.bind(py);
        let buffer = PyBuffer::<u8>::get(&input_obj)
            .map_err(|e| pool_error(format!("input buffer lost: {e}")))?;
        write_prefix(&buffer, &staging[..active])?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("slot_id", ticket.slot_id)?;
        dict.set_item("bucket_t", ticket.bucket_t)?;
        dict.set_item("n_valid", ticket.n_valid)?;
        dict.set_item("chunk", ticket.chunk)?;
        dict.set_item("logical_round", ticket.logical_round)?;
        Ok(Some(dict.into_any().unbind()))
    }

    /// Blocking take (tooling only; never the hot loop).
    fn take_fill_blocking(&mut self, py: Python<'_>, slot_id: u64) -> PyResult<Py<PyAny>> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let registered = self.buffers.get(&slot_id).ok_or_else(|| {
            pool_error(format!("slot {slot_id} has no registered buffers"))
        })?;
        let input_len = registered.input_len;
        let pool = Arc::clone(&self.pool);
        let mut staging = vec![0u8; input_len];
        let ticket = py
            .detach(|| pool.take_fill_blocking(slot_id, &mut staging))
            .map_err(|e| pool_error(e.0))?;
        self.pool.bump_take_blocking();
        let active = super::observation::active_input_bytes(ticket.bucket_t, ticket.n_valid);
        let registered = self.buffers.get(&slot_id).ok_or_else(|| {
            pool_error(format!("slot {slot_id} buffers released before fill copy"))
        })?;
        let input_obj = registered.input.bind(py);
        let buffer = PyBuffer::<u8>::get(&input_obj)
            .map_err(|e| pool_error(format!("input buffer lost: {e}")))?;
        write_prefix(&buffer, &staging[..active])?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("slot_id", ticket.slot_id)?;
        dict.set_item("bucket_t", ticket.bucket_t)?;
        dict.set_item("n_valid", ticket.n_valid)?;
        dict.set_item("chunk", ticket.chunk)?;
        dict.set_item("logical_round", ticket.logical_round)?;
        Ok(dict.into_any().unbind())
    }

    /// Re-encode retained staged rows after restore (byte-identical inputs
    /// for adopted READY lanes; reserves no new sampler words).
    fn refill_staged(&mut self, py: Python<'_>, slot_id: u64) -> PyResult<usize> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let registered = self.buffers.get(&slot_id).ok_or_else(|| {
            pool_error(format!("slot {slot_id} has no registered buffers"))
        })?;
        let input_len = registered.input_len;
        let pool = Arc::clone(&self.pool);
        let mut staging = vec![0u8; input_len];
        let staged = py
            .detach(|| pool.refill_staged(slot_id, &mut staging))
            .map_err(|e| pool_error(e.0))?;
        let registered = self.buffers.get(&slot_id).ok_or_else(|| {
            pool_error(format!("slot {slot_id} buffers released before refill copy"))
        })?;
        let active = super::observation::active_input_bytes(registered.bucket_t, staged);
        let input_obj = registered.input.bind(py);
        let buffer = PyBuffer::<u8>::get(&input_obj)
            .map_err(|e| pool_error(format!("input buffer lost: {e}")))?;
        write_prefix(&buffer, &staging[..active])?;
        Ok(staged)
    }

    /// Nonblocking scan of ALL READY tickets (work-conserving eligibility).
    fn peek_ready(&self) -> PyResult<Option<PyLease>> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let lease = self.pool.peek_ready().map_err(|e| pool_error(e.0))?;
        Ok(lease.map(|lease| PyLease { pool: Arc::clone(&self.pool), lease }))
    }

    fn return_ready_lease(&self, lease: &PyLease) -> PyResult<()> {
        if self.closed || !Arc::ptr_eq(&self.pool, &lease.pool) {
            return Err(pool_error("READY lease does not belong to this open pool"));
        }
        self.pool
            .return_ready_lease(&lease.lease)
            .map_err(|e| pool_error(e.0))
    }

    /// Staged-row descriptor for one lease (JSON string; scheduling and
    /// verification only, never the hot loop).
    fn lease_descriptor(&self, py: Python<'_>, lease: &PyLease) -> PyResult<String> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let pool = Arc::clone(&self.pool);
        let lease_inner = lease.lease.clone();
        let descriptor =
            py.detach(move || pool.lease_descriptor(&lease_inner)).map_err(|e| pool_error(e.0))?;
        serde_json::to_string(&descriptor).map_err(|e| pool_error(e.to_string()))
    }

    /// Commit GPU outputs from the registered slab; returns rows captured.
    /// The slot releases immediately (no engine-apply wait); completions
    /// surface on `poll`/`drain`, joined at `quiesce`.
    fn commit_actions(&mut self, py: Python<'_>, lease: &PyLease) -> PyResult<usize> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let registered = self.buffers.get(&lease.lease.slot_id).ok_or_else(|| {
            pool_error(format!("slot {} has no registered buffers", lease.lease.slot_id))
        })?;
        let need = lease.lease.n_valid * super::pool::OUTPUT_ROW_STRIDE;
        let output_obj = registered.output.bind(py);
        let buffer = PyBuffer::<u8>::get(&output_obj)
            .map_err(|e| pool_error(format!("output buffer lost: {e}")))?;
        let output = read_prefix(&buffer, need)?;
        let pool = Arc::clone(&self.pool);
        let lease_inner = lease.lease.clone();
        let count = py
            .detach(move || pool.commit_actions(&lease_inner, &output))
            .map_err(|e| pool_error(e.0))?;
        Ok(count)
    }

    fn mark_h2d_issued(&self, lease: &PyLease) -> PyResult<()> {
        self.pool
            .mark_phase(&lease.lease, SlotPhase::H2dIssued)
            .map_err(|e| pool_error(e.0))
    }

    fn mark_inference_queued(&self, lease: &PyLease) -> PyResult<()> {
        self.pool
            .mark_phase(&lease.lease, SlotPhase::InferQueued)
            .map_err(|e| pool_error(e.0))
    }

    fn mark_d2h_issued(&self, lease: &PyLease) -> PyResult<()> {
        self.pool
            .mark_phase(&lease.lease, SlotPhase::D2hIssued)
            .map_err(|e| pool_error(e.0))
    }

    /// Record HOST_READY only when BOTH actual CUDA events report complete
    /// (owner batch-boundary query; never caller-supplied bools).
    fn mark_host_ready(
        &self,
        lease: &PyLease,
        compute_event: Bound<'_, PyAny>,
        d2h_event: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        check_events_complete(lease.lease.lease_id, &compute_event, &d2h_event)?;
        self.pool
            .mark_phase(&lease.lease, SlotPhase::HostReady)
            .map_err(|e| pool_error(e.0))
    }

    fn poll(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let status = self.pool.poll().map_err(|e| pool_error(e.0))?;
        status_dict(py, &status)
    }

    fn wait_for_progress(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let pool = Arc::clone(&self.pool);
        let status =
            py.detach(move || pool.wait_for_progress()).map_err(|e| pool_error(e.0))?;
        status_dict(py, &status)
    }

    fn quiesce(&self) -> PyResult<()> {
        self.pool.quiesce().map_err(|e| pool_error(e.0))
    }

    /// Resume admission after quiesce + snapshot/control durable (no
    /// reset, no resample; retained READY keeps flowing). Zero-issued
    /// state required (no leases/applies/in-flight fill).
    fn resume_admission(&self) -> PyResult<()> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        self.pool.resume_admission().map_err(|e| pool_error(e.0))
    }

    /// Collector fill choice + admission in one call: the native router
    /// picks the eligible bucket among the owner's FREE lanes (yield-aware
    /// rotation, starves no bucket) and admits its fill. Returns the
    /// chosen slot, or `None` when no lane is free/quiesced (expected
    /// stall, never poison). Collect with `take_fill`.
    fn admit_next_fill(&mut self, free_slot_ids: Vec<u64>) -> PyResult<Option<u64>> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let chosen = self.pool.advise_fill(&free_slot_ids).map_err(|e| pool_error(e.0))?;
        let slot_id = match chosen {
            Some(slot_id) => slot_id,
            None => return Ok(None),
        };
        let registered = self.buffers.get(&slot_id).ok_or_else(|| {
            pool_error(format!("slot {slot_id} has no registered buffers"))
        })?;
        self.pool
            .admit_fill(slot_id, registered.output_len)
            .map_err(|e| pool_error(e.0))?;
        Ok(Some(slot_id))
    }

    fn drain(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let status = self.pool.drain_status().map_err(|e| pool_error(e.0))?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("retained_ready", status.retained_ready.len())?;
        dict.set_item("retained_rows", status.retained_ready.iter().map(|(_, n)| n).sum::<usize>())?;
        dict.set_item("live_leases", status.live_leases.len())?;
        dict.set_item("pending_applies", status.pending_applies)?;
        dict.set_item("poison", status.poison)?;
        Ok(dict.into_any().unbind())
    }

    /// Per-game completion descriptors (JSON string; terminal flags,
    /// captured counts, outcome facts for the seal loop).
    fn completions(&self) -> PyResult<String> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let completions = self.pool.game_completions().map_err(|e| pool_error(e.0))?;
        let docs: Vec<serde_json::Value> = completions
            .iter()
            .map(|c| {
                serde_json::json!({
                    "wall_id": c.wall_id,
                    "game_id": c.game_id,
                    "terminal": c.terminal,
                    "captured": c.captured,
                    "outcome": c.outcome.as_ref().map(|o| serde_json::json!({
                        "final_scores": o.final_scores,
                        "ranks": o.ranks,
                        "point_deltas": o.point_deltas,
                        "utility_g4": o.utility_g4,
                    })),
                })
            })
            .collect();
        serde_json::to_string(&docs).map_err(|e| pool_error(e.to_string()))
    }

    /// Generation-end tape handoff: registers the owned tape (production
    /// rows UNBOUND with utilities set; Main's GPU freezes exactly once)
    /// and returns its process handle for the ArtifactStorage seal. Rows
    /// never cross Python.
    fn export_tape(&self, py: Python<'_>) -> PyResult<u64> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let pool = Arc::clone(&self.pool);
        py.detach(move || pool.export_tape()).map_err(|e| pool_error(e.0))
    }

    /// Destructive live tape handoff: moves completed arenas/rows into a
    /// registered live tape with the verified source binding and returns
    /// its process handle. Rows never cross Python.
    fn drain_tape(&self, py: Python<'_>, live_binding_json: Vec<u8>) -> PyResult<u64> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let binding = parse_live_binding(&live_binding_json)?;
        let pool = Arc::clone(&self.pool);
        py.detach(move || pool.drain_tape(binding)).map_err(|e| pool_error(e.0))
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<String> {
        if self.closed {
            return Err(pool_error("pool is closed"));
        }
        let pool = Arc::clone(&self.pool);
        let snap = py.detach(move || pool.snapshot()).map_err(|e| pool_error(e.0))?;
        serde_json::to_string(&snap.to_json()).map_err(|e| pool_error(e.to_string()))
    }

    /// Close: rejects issued work (leases, leased/Filling lanes, applies,
    /// in-flight fills) but accepts retained unissued READY (snapshot
    /// first); joins workers, then releases host owners.
    fn close(&mut self) -> PyResult<()> {
        if self.closed {
            return Ok(());
        }
        self.pool.close().map_err(|e| pool_error(e.0))?;
        self.buffers.clear();
        self.closed = true;
        Ok(())
    }
}

/// Full poll descriptor for the Main seal loop (completion detection).
fn status_dict(py: Python<'_>, status: &super::pool::PoolStatus) -> PyResult<Py<PyAny>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("poison", status.poison.clone())?;
    dict.set_item("live_leases", status.live_leases)?;
    dict.set_item("pending_applies", status.pending_applies)?;
    dict.set_item("retained_ready_rows", status.retained_ready_rows)?;
    dict.set_item("captured_rows", status.captured_rows)?;
    dict.set_item("terminal_games", status.terminal_games)?;
    dict.set_item("last_tape_handle", status.last_tape_handle)?;
    dict.set_item("admit_ok", status.admit_ok)?;
    dict.set_item("advise_none_quiesced", status.advise_none_quiesced)?;
    dict.set_item("advise_none_owner_empty", status.advise_none_owner_empty)?;
    dict.set_item("advise_none_no_truly_free", status.advise_none_no_truly_free)?;
    dict.set_item("take_miss", status.take_miss)?;
    dict.set_item("take_hit", status.take_hit)?;
    dict.set_item("take_blocking", status.take_blocking)?;
    dict.set_item("fill_inflight", status.fill_inflight)?;
    dict.set_item("fill_result_ready", status.fill_result_ready)?;
    dict.set_item("quiesced", status.quiesced)?;
    Ok(dict.into_any().unbind())
}

fn check_events_complete(
    lease_id: u64,
    compute_event: &Bound<'_, PyAny>,
    d2h_event: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let compute_done: bool = compute_event
        .call_method0("query")
        .map_err(|e| pool_error(format!("compute event query failed: {e}")))?
        .extract()
        .map_err(|e| pool_error(format!("compute event query not a bool: {e}")))?;
    let d2h_done: bool = d2h_event
        .call_method0("query")
        .map_err(|e| pool_error(format!("d2h event query failed: {e}")))?
        .extract()
        .map_err(|e| pool_error(format!("d2h event query not a bool: {e}")))?;
    if !compute_done || !d2h_done {
        return Err(pool_error(format!(
            "lease {lease_id} HOST_READY refused: compute_done={compute_done} d2h_done={d2h_done}"
        )));
    }
    Ok(())
}

/// Opaque lease: pool binding + slot epoch + generation/policy + geometry.
/// Method calls validate phase order; the FFI never forges completion.
#[pyclass]
pub struct PyLease {
    pool: Arc<NativePool>,
    lease: Lease,
}

#[pymethods]
impl PyLease {
    #[getter]
    fn lease_id(&self) -> u64 {
        self.lease.lease_id
    }

    #[getter]
    fn slot_id(&self) -> u64 {
        self.lease.slot_id
    }

    #[getter]
    fn bucket_t(&self) -> usize {
        self.lease.bucket_t
    }

    #[getter]
    fn n_valid(&self) -> usize {
        self.lease.n_valid
    }

    #[getter]
    fn input_bytes(&self) -> usize {
        super::observation::active_input_bytes(self.lease.bucket_t, self.lease.n_valid)
    }

    #[getter]
    fn output_bytes(&self) -> usize {
        self.lease.n_valid * super::pool::OUTPUT_ROW_STRIDE
    }

    /// Active-prefix layout entries `(name, dtype, shape, byte_offset)` for
    /// cached torch views over the real active B (`uniform` f64 last).
    #[getter]
    fn input_layout(&self) -> Vec<(String, String, Vec<usize>, usize)> {
        super::observation::active_layout(self.lease.bucket_t, self.lease.n_valid, true)
    }

    fn mark_h2d_issued(&self) -> PyResult<()> {
        self.pool
            .mark_phase(&self.lease, SlotPhase::H2dIssued)
            .map_err(|e| pool_error(e.0))
    }

    fn mark_inference_queued(&self) -> PyResult<()> {
        self.pool
            .mark_phase(&self.lease, SlotPhase::InferQueued)
            .map_err(|e| pool_error(e.0))
    }

    fn mark_d2h_issued(&self) -> PyResult<()> {
        self.pool
            .mark_phase(&self.lease, SlotPhase::D2hIssued)
            .map_err(|e| pool_error(e.0))
    }

    fn mark_host_ready(
        &self,
        compute_event: Bound<'_, PyAny>,
        d2h_event: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        check_events_complete(self.lease.lease_id, &compute_event, &d2h_event)?;
        self.pool
            .mark_phase(&self.lease, SlotPhase::HostReady)
            .map_err(|e| pool_error(e.0))
    }
}

/// Cold geometry helper: exact `(input_bytes, output_bytes)` for registration.
#[pyfunction]
fn buffer_geometry(bucket_t: usize, capacity: usize) -> PyResult<(usize, usize)> {
    pool_buffer_geometry(bucket_t, capacity).map_err(|e| pool_error(e.0))
}

/// Register the batch-only surface onto the `rl` submodule (registrar-owned
/// `rl/mod.rs` calls this; no new entry point).
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNativePool>()?;
    m.add_class::<PyLease>()?;
    m.add_function(pyo3::wrap_pyfunction!(buffer_geometry, m)?)?;
    Ok(())
}

fn parse_hex32(text: &str) -> PyResult<[u8; 32]> {
    let noprefix = text.trim_start_matches("sha256:");
    if noprefix.len() != 64 {
        return Err(pool_error("digest must be 32 bytes hex"));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&noprefix[2 * i..2 * i + 2], 16)
            .map_err(|_| pool_error("digest must be 32 bytes hex"))?;
    }
    Ok(out)
}

/// Exact-capacity match for one lane (aligned input totals are not a
/// multiple of a per-row stride, so division is unsound; totals grow
/// strictly with capacity, so the match is unique).
fn capacity_for(bucket_t: usize, in_len: usize, out_len: usize) -> PyResult<usize> {
    for capacity in 1..=super::pool::TICKET_CAP {
        if let Ok((expect_in, expect_out)) = buffer_geometry(bucket_t, capacity) {
            if expect_in == in_len && expect_out == out_len {
                return Ok(capacity);
            }
        }
    }
    Err(pool_error(format!(
        "buffer sizes ({in_len},{out_len}) match no capacity of T={bucket_t}"
    )))
}

fn parse_snapshot(snapshot_json: &str) -> PyResult<PoolSnapshot> {
    use super::core::GameRow;
    let value: serde_json::Value =
        serde_json::from_str(snapshot_json).map_err(|e| pool_error(e.to_string()))?;
    let mut games = vec![];
    for game in value.get("games").and_then(|g| g.as_array()).cloned().unwrap_or_default() {
        let mut applied = vec![];
        for action in game.get("applied").and_then(|a| a.as_array()).cloned().unwrap_or_default() {
            applied.push(StoredAction::from_json(&action).map_err(|e| pool_error(e.0))?);
        }
        let mut completed = vec![];
        for row in game.get("completed").and_then(|r| r.as_array()).cloned().unwrap_or_default() {
            completed.push(
                GameRow::from_json(&row)
                    .map_err(|e| pool_error(format!("snapshot row rejected: {e}")))?,
            );
        }
        let digest_hex = game.get("digest").and_then(|d| d.as_str()).unwrap_or("");
        // Per-seat counters (new shape); a legacy scalar migrates by
        // broadcast (skip-ahead, never word reuse: each seat resumes at or
        // past every word its game ever issued).
        let sample_counters = match game.get("sample_counters").and_then(|c| c.as_array()) {
            Some(counters) if counters.len() == 4 => {
                let mut out = [0u64; 4];
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = counters[i].as_u64().ok_or_else(|| {
                        pool_error("snapshot sample_counters must hold u64s".to_string())
                    })?;
                }
                out
            }
            _ => {
                let legacy = game.get("sample_counter").and_then(|c| c.as_u64()).ok_or_else(|| {
                    pool_error("snapshot game lacks sample_counters".to_string())
                })?;
                [legacy; 4]
            }
        };
        games.push(GameSnapshot {
            wall_id: game.get("wall_id").and_then(|w| w.as_str()).unwrap_or("").to_string(),
            ordinal: game.get("ordinal").and_then(|o| o.as_u64()).unwrap_or(0),
            applied,
            digest: parse_hex32(digest_hex)?,
            captured: game.get("captured").and_then(|c| c.as_u64()).unwrap_or(0),
            sample_counters,
            terminal: game.get("terminal").and_then(|t| t.as_bool()).unwrap_or(false),
            completed,
        });
    }
    let mut slots = vec![];
    for slot in value.get("slots").and_then(|s| s.as_array()).cloned().unwrap_or_default() {
        let phase = match slot.get("phase").and_then(|p| p.as_str()).unwrap_or("Free") {
            "Ready" => SlotPhase::Ready,
            "Free" => SlotPhase::Free,
            other => {
                return Err(pool_error(format!("snapshot slot phase {other:?} unrestorable")));
            }
        };
        let mut staged = vec![];
        for row in slot.get("staged").and_then(|s| s.as_array()).cloned().unwrap_or_default() {
            staged.push(StagedRowSnapshot {
                wall_id: row.get("wall_id").and_then(|w| w.as_str()).unwrap_or("").to_string(),
                actor: row.get("actor").and_then(|a| a.as_u64()).unwrap_or(0) as u8,
                uniform_bits: row
                    .get("uniform_bits")
                    .and_then(|u| u.as_u64())
                    .ok_or_else(|| pool_error("snapshot staged row lacks uniform_bits".to_string()))?,
                cursor_before: row
                    .get("cursor_before")
                    .and_then(|c| c.as_u64())
                    .ok_or_else(|| pool_error("snapshot staged row lacks cursor_before".to_string()))?,
                obs_hash: parse_hex32(
                    row.get("obs_hash").and_then(|h| h.as_str()).unwrap_or(""),
                )?,
            });
        }
        slots.push(SlotSnapshot {
            id: slot.get("id").and_then(|i| i.as_u64()).unwrap_or(0),
            epoch: slot.get("epoch").and_then(|e| e.as_u64()).unwrap_or(0),
            phase,
            bucket_t: slot.get("bucket_t").and_then(|b| b.as_u64()).unwrap_or(32) as usize,
            capacity: slot.get("capacity").and_then(|c| c.as_u64()).unwrap_or(0) as usize,
            chunk: slot.get("chunk").and_then(|c| c.as_u64()).unwrap_or(0),
            logical_round: slot.get("logical_round").and_then(|l| l.as_u64()).unwrap_or(0),
            staged,
        });
    }
    let mut leases = vec![];
    for lease in value.get("leases").and_then(|l| l.as_array()).cloned().unwrap_or_default() {
        leases.push(Lease {
            lease_id: lease.get("lease_id").and_then(|l| l.as_u64()).unwrap_or(0),
            slot_id: lease.get("slot_id").and_then(|s| s.as_u64()).unwrap_or(0),
            slot_epoch: lease.get("slot_epoch").and_then(|s| s.as_u64()).unwrap_or(0),
            generation: lease.get("generation").and_then(|g| g.as_u64()).unwrap_or(0),
            policy_id: lease.get("policy_id").and_then(|p| p.as_str()).unwrap_or("").to_string(),
            bucket_t: lease.get("bucket_t").and_then(|b| b.as_u64()).unwrap_or(32) as usize,
            chunk: lease.get("chunk").and_then(|c| c.as_u64()).unwrap_or(0),
            n_valid: lease.get("n_valid").and_then(|n| n.as_u64()).unwrap_or(0) as usize,
            logical_round: lease.get("logical_round").and_then(|l| l.as_u64()).unwrap_or(0),
        });
    }
    Ok(PoolSnapshot {
        pool_id: value.get("pool_id").and_then(|p| p.as_str()).unwrap_or("").to_string(),
        generation: value.get("generation").and_then(|g| g.as_u64()).unwrap_or(0),
        games,
        slots,
        leases,
        poison: value.get("poison").and_then(|p| p.as_str()).map(str::to_string),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_byte_math_matches_geometry() {
        let (input, output) = buffer_geometry(256, 128).unwrap();
        assert_eq!(input, 128 * (7254 + 9 * 256) + 1024);
        assert_eq!(output, 128 * 28);
    }
}
