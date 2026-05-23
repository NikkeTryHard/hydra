//! C ABI prototype for direct raw-MJAI fill into caller-owned pinned host buffers.
#![allow(
    clippy::missing_safety_doc,
    reason = "C ABI functions document pointer contracts in struct and function comments"
)]

use std::ffi::{CStr, CString, c_char};
use std::mem;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::ptr;
use std::slice;
use std::sync::Mutex;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;
use hydra_train_exec::raw_mjai_stream::{
    RawMjaiBatchStreamConfig, RawMjaiPinnedBatchView, RawMjaiPinnedStream,
    RawMjaiPinnedStreamStats, RawMjaiStreamSplit,
};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

const GRP_CLASS_COUNT: usize = 24;
const PLAYER_COUNT: usize = 4;
const OPPONENT_COUNT: usize = 3;
const SPATIAL_TARGET_SIZE: usize = 102;
const SCORE_BINS: usize = 64;

/// FFI config for one raw-MJAI batch fill.
#[repr(C)]
pub struct HydraRawMjaiConfig {
    /// UTF-8 input path.
    pub input_utf8: *const c_char,
    /// `0` train, `1` validation.
    pub split: u32,
    /// Train fraction used by deterministic split.
    pub train_fraction: f32,
    /// Batch rows to fill.
    pub batch_size: usize,
    /// `0` means unset.
    pub max_games: usize,
    /// `0` means unset.
    pub max_samples: usize,
    /// `0` means Rayon default.
    pub num_threads: usize,
    /// Work queue bound.
    pub queue_bound: usize,
    /// Enable suit augmentation.
    pub augment: bool,
}

/// FFI destination view. All pointers must address writable contiguous CPU memory.
#[repr(C)]
pub struct HydraRawMjaiBatchView {
    /// `[capacity_rows, 192, 34]` float32.
    pub obs_f32: *mut f32,
    /// `[capacity_rows]` int64.
    pub actions_i64: *mut i64,
    /// `[capacity_rows, 46]` bool/uint8, written as 0/1.
    pub legal_u8: *mut u8,
    /// `[capacity_rows]` float32.
    pub value_f32: *mut f32,
    /// `[capacity_rows, 24]` float32.
    pub grp_f32: *mut f32,
    /// `[capacity_rows, 4]` float32.
    pub oracle_f32: *mut f32,
    /// `[capacity_rows]` float32.
    pub oracle_mask_f32: *mut f32,
    /// `[capacity_rows, 3]` float32.
    pub tenpai_f32: *mut f32,
    /// `[capacity_rows, 102]` float32.
    pub opp_next_f32: *mut f32,
    /// `[capacity_rows, 102]` float32.
    pub danger_f32: *mut f32,
    /// `[capacity_rows, 102]` float32.
    pub danger_mask_f32: *mut f32,
    /// `[capacity_rows, 64]` float32.
    pub score_pdf_f32: *mut f32,
    /// `[capacity_rows, 64]` float32.
    pub score_cdf_f32: *mut f32,
    /// Row capacity shared by all fields.
    pub capacity_rows: usize,
}

/// FFI totals copied from Rust stream accounting.
#[repr(C)]
#[derive(Default)]
pub struct HydraRawMjaiTotals {
    /// Games accepted by the stream.
    pub loaded_games: u64,
    /// Games skipped by parsing/materialization.
    pub skipped_games: u64,
    /// Rows written.
    pub samples: u64,
    /// Batches written.
    pub batches: u64,
    /// Max-games cap stopped planning.
    pub max_games_reached: bool,
    /// Max-samples cap stopped fill.
    pub max_samples_reached: bool,
}

/// FFI-owned error string. Free with `hydra_raw_mjai_error_free`.
#[repr(C)]
#[derive(Default)]
pub struct HydraRawMjaiError {
    /// Negative error code.
    pub code: i32,
    /// UTF-8 error text, or null.
    pub message_utf8: *mut c_char,
}

/// FFI counters/timers for persistent stream instrumentation.
#[repr(C)]
#[derive(Default)]
pub struct HydraRawMjaiStats {
    /// Number of stream opens; must stay 1 for a handle.
    pub open_count: u64,
    /// Open scan+plan+worker-start wall time.
    pub open_scan_plan_ms: f64,
    /// Last next_into wall time.
    pub last_next_fill_ms: f64,
    /// Last next_into wait-for-materialized-game time.
    pub last_queue_wait_ms: f64,
    /// Bytes filled by last next_into.
    pub last_bytes_filled: u64,
    /// Games consumed by last next_into.
    pub last_games_consumed: u64,
}

/// Persistent Rust stream handle. Opaque to Python.
pub struct HydraRawMjaiStreamHandle {
    stream: RawMjaiPinnedStream,
}

/// Open a persistent raw-MJAI pinned stream. Scan/plan happens exactly once here.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hydra_raw_mjai_stream_open(
    cfg: *const HydraRawMjaiConfig,
    out_handle: *mut *mut HydraRawMjaiStreamHandle,
    out_stats: *mut HydraRawMjaiStats,
    err: *mut HydraRawMjaiError,
) -> i32 {
    clear_error(err);
    if out_handle.is_null() {
        return set_error(err, -1, "out_handle is null");
    }
    unsafe { *out_handle = ptr::null_mut() };
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        open_inner(cfg, out_handle, out_stats)
    }));
    match result {
        Ok(Ok(())) => 1,
        Ok(Err(message)) => set_error(err, -2, &message),
        Err(_) => set_error(err, -3, "raw MJAI persistent stream open panicked"),
    }
}

/// Fill the next batch into Python-owned pinned host buffers using an existing stream handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hydra_raw_mjai_stream_next_into(
    handle: *mut HydraRawMjaiStreamHandle,
    dst: *mut HydraRawMjaiBatchView,
    out_rows: *mut usize,
    out_totals: *mut HydraRawMjaiTotals,
    out_stats: *mut HydraRawMjaiStats,
    err: *mut HydraRawMjaiError,
) -> i32 {
    clear_error(err);
    if out_rows.is_null() {
        return set_error(err, -1, "out_rows is null");
    }
    unsafe { *out_rows = 0 };
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        next_into_inner(handle, dst, out_rows, out_totals, out_stats)
    }));
    match result {
        Ok(Ok(0)) => 0,
        Ok(Ok(_)) => 1,
        Ok(Err(message)) => set_error(err, -2, &message),
        Err(_) => set_error(err, -3, "raw MJAI persistent stream next_into panicked"),
    }
}

/// Close and free a persistent raw-MJAI stream handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hydra_raw_mjai_stream_close(handle: *mut HydraRawMjaiStreamHandle) {
    if handle.is_null() {
        return;
    }
    unsafe { drop(Box::from_raw(handle)) };
}

/// Free an FFI error message allocated by this crate.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hydra_raw_mjai_error_free(err: *mut HydraRawMjaiError) {
    if err.is_null() {
        return;
    }
    let message = unsafe { (*err).message_utf8 };
    if !message.is_null() {
        unsafe { drop(CString::from_raw(message)) };
    }
    unsafe {
        (*err).code = 0;
        (*err).message_utf8 = ptr::null_mut();
    }
}

unsafe fn open_inner(
    cfg: *const HydraRawMjaiConfig,
    out_handle: *mut *mut HydraRawMjaiStreamHandle,
    out_stats: *mut HydraRawMjaiStats,
) -> Result<(), String> {
    if cfg.is_null() {
        return Err("config is null".to_owned());
    }
    let cfg = unsafe { &*cfg };
    let config = rust_config_from_ffi(cfg)?;
    let stream = RawMjaiPinnedStream::open(config).map_err(|err| err.to_string())?;
    if !out_stats.is_null() {
        unsafe { *out_stats = ffi_stats(stream.stats()) };
    }
    let handle = Box::new(HydraRawMjaiStreamHandle { stream });
    unsafe { *out_handle = Box::into_raw(handle) };
    Ok(())
}

unsafe fn next_into_inner(
    handle: *mut HydraRawMjaiStreamHandle,
    dst: *mut HydraRawMjaiBatchView,
    out_rows: *mut usize,
    out_totals: *mut HydraRawMjaiTotals,
    out_stats: *mut HydraRawMjaiStats,
) -> Result<usize, String> {
    if handle.is_null() {
        return Err("stream handle is null".to_owned());
    }
    if dst.is_null() {
        return Err("batch view is null".to_owned());
    }
    let handle = unsafe { &mut *handle };
    let dst = unsafe { &mut *dst };
    let view = unsafe { pinned_view_from_ffi(dst)? };
    let next = handle
        .stream
        .next_into(view)
        .map_err(|err| err.to_string())?;
    unsafe { *out_rows = next.rows };
    if !out_totals.is_null() {
        unsafe { *out_totals = ffi_totals(next.totals) };
    }
    if !out_stats.is_null() {
        unsafe { *out_stats = ffi_stats(next.stats) };
    }
    Ok(next.rows)
}

fn ffi_totals(
    totals: hydra_train_exec::raw_mjai_stream::RawMjaiBatchStreamTotals,
) -> HydraRawMjaiTotals {
    HydraRawMjaiTotals {
        loaded_games: totals.loaded_games,
        skipped_games: totals.skipped_games,
        samples: totals.samples,
        batches: totals.batches,
        max_games_reached: totals.max_games_reached,
        max_samples_reached: totals.max_samples_reached,
    }
}

fn ffi_stats(stats: RawMjaiPinnedStreamStats) -> HydraRawMjaiStats {
    HydraRawMjaiStats {
        open_count: stats.open_count,
        open_scan_plan_ms: stats.open_scan_plan_ms,
        last_next_fill_ms: stats.last_next_fill_ms,
        last_queue_wait_ms: stats.last_queue_wait_ms,
        last_bytes_filled: stats.last_bytes_filled,
        last_games_consumed: stats.last_games_consumed,
    }
}

fn rust_config_from_ffi(cfg: &HydraRawMjaiConfig) -> Result<RawMjaiBatchStreamConfig, String> {
    if cfg.input_utf8.is_null() {
        return Err("input_utf8 is null".to_owned());
    }
    // SAFETY: `input_utf8` is non-null and must point to a NUL-terminated C string.
    let input = unsafe { CStr::from_ptr(cfg.input_utf8) }
        .to_str()
        .map_err(|err| format!("input path is not UTF-8: {err}"))?;
    let split = match cfg.split {
        0 => RawMjaiStreamSplit::Train,
        1 => RawMjaiStreamSplit::Validation,
        other => {
            return Err(format!(
                "invalid split {other}; expected 0 train or 1 validation"
            ));
        }
    };
    Ok(RawMjaiBatchStreamConfig {
        inputs: vec![PathBuf::from(input)],
        split,
        train_fraction: cfg.train_fraction,
        batch_size: cfg.batch_size,
        max_games: nonzero(cfg.max_games),
        max_samples: nonzero(cfg.max_samples),
        num_threads: nonzero(cfg.num_threads),
        queue_bound: cfg.queue_bound,
        augment: cfg.augment,
        source_manifest: None,
    })
}

unsafe fn pinned_view_from_ffi<'a>(
    dst: &'a mut HydraRawMjaiBatchView,
) -> Result<RawMjaiPinnedBatchView<'a>, String> {
    let rows = dst.capacity_rows;
    Ok(RawMjaiPinnedBatchView {
        obs: checked_slice(dst.obs_f32, rows * OBS_SIZE, "obs_f32")?,
        actions: checked_slice(dst.actions_i64, rows, "actions_i64")?,
        legal: checked_slice(dst.legal_u8, rows * HYDRA_ACTION_SPACE, "legal_u8")?,
        value: checked_slice(dst.value_f32, rows, "value_f32")?,
        grp: checked_slice(dst.grp_f32, rows * GRP_CLASS_COUNT, "grp_f32")?,
        oracle: checked_slice(dst.oracle_f32, rows * PLAYER_COUNT, "oracle_f32")?,
        oracle_mask: checked_slice(dst.oracle_mask_f32, rows, "oracle_mask_f32")?,
        tenpai: checked_slice(dst.tenpai_f32, rows * OPPONENT_COUNT, "tenpai_f32")?,
        opp_next: checked_slice(dst.opp_next_f32, rows * SPATIAL_TARGET_SIZE, "opp_next_f32")?,
        danger: checked_slice(dst.danger_f32, rows * SPATIAL_TARGET_SIZE, "danger_f32")?,
        danger_mask: checked_slice(
            dst.danger_mask_f32,
            rows * SPATIAL_TARGET_SIZE,
            "danger_mask_f32",
        )?,
        score_pdf: checked_slice(dst.score_pdf_f32, rows * SCORE_BINS, "score_pdf_f32")?,
        score_cdf: checked_slice(dst.score_cdf_f32, rows * SCORE_BINS, "score_cdf_f32")?,
        capacity_rows: rows,
    })
}

fn nonzero(value: usize) -> Option<usize> {
    if value == 0 { None } else { Some(value) }
}

fn checked_slice<'a, T>(ptr: *mut T, len: usize, name: &str) -> Result<&'a mut [T], String> {
    if len == 0 {
        return Ok(&mut []);
    }
    if ptr.is_null() {
        return Err(format!("{name} is null"));
    }
    // SAFETY: caller guarantees `ptr` points to `len` contiguous initialized writable elements.
    Ok(unsafe { slice::from_raw_parts_mut(ptr, len) })
}

fn clear_error(err: *mut HydraRawMjaiError) {
    if err.is_null() {
        return;
    }
    // SAFETY: non-null error pointer is caller-owned result storage.
    unsafe {
        (*err).code = 0;
        (*err).message_utf8 = ptr::null_mut();
    }
}

fn set_error(err: *mut HydraRawMjaiError, code: i32, message: &str) -> i32 {
    if !err.is_null() {
        let sanitized = message.replace('\0', " ");
        if let Ok(c_string) = CString::new(sanitized) {
            // SAFETY: non-null error pointer is caller-owned result storage.
            unsafe {
                (*err).code = code;
                (*err).message_utf8 = c_string.into_raw();
            }
        }
    }
    code
}

#[pyclass(name = "RawMjaiStream")]
struct PyRawMjaiStream {
    inner: Mutex<Option<RawMjaiPinnedStream>>,
}

#[pymethods]
impl PyRawMjaiStream {
    #[new]
    #[allow(
        clippy::too_many_arguments,
        reason = "PyO3 constructor mirrors Python keyword API for explicit stream config"
    )]
    #[pyo3(signature = (data_dirs, batch_size=2048, train_fraction=0.9, worker_threads=20, queue_bound=8, max_games=None, max_samples=None, augment=false, split="train"))]
    fn new(
        py: Python<'_>,
        data_dirs: RawMjaiPyInputs,
        batch_size: usize,
        train_fraction: f32,
        worker_threads: usize,
        queue_bound: usize,
        max_games: Option<usize>,
        max_samples: Option<usize>,
        augment: bool,
        split: &str,
    ) -> PyResult<Self> {
        let split = parse_py_split(split)?;
        let config = RawMjaiBatchStreamConfig {
            inputs: data_dirs.paths,
            split,
            train_fraction,
            batch_size,
            max_games,
            max_samples,
            num_threads: Some(worker_threads),
            queue_bound,
            augment,
            source_manifest: None,
        };
        let stream = py
            .detach(|| RawMjaiPinnedStream::open(config))
            .map_err(io_py_err)?;
        Ok(Self {
            inner: Mutex::new(Some(stream)),
        })
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "PyO3 hot path accepts fixed field pointers to avoid per-batch object allocation"
    )]
    fn next_into(
        &mut self,
        py: Python<'_>,
        obs_ptr: usize,
        actions_ptr: usize,
        legal_ptr: usize,
        value_ptr: usize,
        grp_ptr: usize,
        oracle_ptr: usize,
        oracle_mask_ptr: usize,
        tenpai_ptr: usize,
        opp_next_ptr: usize,
        danger_ptr: usize,
        danger_mask_ptr: usize,
        score_pdf_ptr: usize,
        score_cdf_ptr: usize,
        capacity_rows: usize,
    ) -> PyResult<PyRawMjaiNext> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("raw MJAI stream mutex poisoned"))?;
        let stream = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("raw MJAI stream is closed"))?;
        py.detach(|| unsafe {
            let mut view = HydraRawMjaiBatchView {
                obs_f32: obs_ptr as *mut f32,
                actions_i64: actions_ptr as *mut i64,
                legal_u8: legal_ptr as *mut u8,
                value_f32: value_ptr as *mut f32,
                grp_f32: grp_ptr as *mut f32,
                oracle_f32: oracle_ptr as *mut f32,
                oracle_mask_f32: oracle_mask_ptr as *mut f32,
                tenpai_f32: tenpai_ptr as *mut f32,
                opp_next_f32: opp_next_ptr as *mut f32,
                danger_f32: danger_ptr as *mut f32,
                danger_mask_f32: danger_mask_ptr as *mut f32,
                score_pdf_f32: score_pdf_ptr as *mut f32,
                score_cdf_f32: score_cdf_ptr as *mut f32,
                capacity_rows,
            };
            let view = pinned_view_from_ffi(&mut view).map_err(PyValueError::new_err)?;
            stream.next_into(view).map_err(io_py_err)
        })
        .map(PyRawMjaiNext::from)
    }

    fn stats(&self) -> PyResult<PyRawMjaiStats> {
        let guard = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("raw MJAI stream mutex poisoned"))?;
        let stream = guard
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("raw MJAI stream is closed"))?;
        Ok(PyRawMjaiStats::from(stream.stats()))
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let stream = self
            .inner
            .lock()
            .map_err(|_| PyOSError::new_err("raw MJAI stream mutex poisoned"))?
            .take();
        if let Some(stream) = stream {
            py.detach(|| mem::drop(stream));
        }
        Ok(())
    }
}

struct RawMjaiPyInputs {
    paths: Vec<PathBuf>,
}

impl<'a, 'py> FromPyObject<'a, 'py> for RawMjaiPyInputs {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(path) = obj.extract::<String>() {
            return Ok(Self {
                paths: vec![PathBuf::from(path)],
            });
        }
        let values = obj.extract::<Vec<String>>()?;
        if values.is_empty() {
            return Err(PyValueError::new_err(
                "raw MJAI data_dirs must not be empty",
            ));
        }
        Ok(Self {
            paths: values.into_iter().map(PathBuf::from).collect(),
        })
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone, Copy)]
struct PyRawMjaiStats {
    #[pyo3(get)]
    open_count: u64,
    #[pyo3(get)]
    open_scan_plan_ms: f64,
    #[pyo3(get)]
    last_next_fill_ms: f64,
    #[pyo3(get)]
    last_queue_wait_ms: f64,
    #[pyo3(get)]
    last_bytes_filled: u64,
    #[pyo3(get)]
    last_games_consumed: u64,
}

impl From<RawMjaiPinnedStreamStats> for PyRawMjaiStats {
    fn from(stats: RawMjaiPinnedStreamStats) -> Self {
        Self {
            open_count: stats.open_count,
            open_scan_plan_ms: stats.open_scan_plan_ms,
            last_next_fill_ms: stats.last_next_fill_ms,
            last_queue_wait_ms: stats.last_queue_wait_ms,
            last_bytes_filled: stats.last_bytes_filled,
            last_games_consumed: stats.last_games_consumed,
        }
    }
}

#[pyclass]
struct PyRawMjaiNext {
    #[pyo3(get)]
    rows: usize,
    #[pyo3(get)]
    loaded_games: u64,
    #[pyo3(get)]
    skipped_games: u64,
    #[pyo3(get)]
    samples: u64,
    #[pyo3(get)]
    batches: u64,
    #[pyo3(get)]
    max_games_reached: bool,
    #[pyo3(get)]
    max_samples_reached: bool,
    #[pyo3(get)]
    stats: PyRawMjaiStats,
}

impl From<hydra_train_exec::raw_mjai_stream::RawMjaiPinnedNext> for PyRawMjaiNext {
    fn from(next: hydra_train_exec::raw_mjai_stream::RawMjaiPinnedNext) -> Self {
        Self {
            rows: next.rows,
            loaded_games: next.totals.loaded_games,
            skipped_games: next.totals.skipped_games,
            samples: next.totals.samples,
            batches: next.totals.batches,
            max_games_reached: next.totals.max_games_reached,
            max_samples_reached: next.totals.max_samples_reached,
            stats: next.stats.into(),
        }
    }
}

fn parse_py_split(split: &str) -> PyResult<RawMjaiStreamSplit> {
    match split {
        "train" => Ok(RawMjaiStreamSplit::Train),
        "validation" => Ok(RawMjaiStreamSplit::Validation),
        other => Err(PyValueError::new_err(format!(
            "invalid split {other:?}; expected train or validation"
        ))),
    }
}

fn io_py_err(err: std::io::Error) -> PyErr {
    PyOSError::new_err(err.to_string())
}

#[pymodule]
fn hydra_raw_mjai(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRawMjaiStream>()?;
    module.add_class::<PyRawMjaiStats>()?;
    module.add_class::<PyRawMjaiNext>()?;
    Ok(())
}
