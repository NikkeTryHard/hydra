//! PyO3 extension for direct raw-MJAI fill into caller-owned pinned host buffers.

use std::mem;
use std::path::PathBuf;
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
struct HydraRawMjaiBatchView {
    obs_f32: *mut f32,
    actions_i64: *mut i64,
    legal_u8: *mut u8,
    value_f32: *mut f32,
    grp_f32: *mut f32,
    oracle_f32: *mut f32,
    oracle_mask_f32: *mut f32,
    tenpai_f32: *mut f32,
    opp_next_f32: *mut f32,
    danger_f32: *mut f32,
    danger_mask_f32: *mut f32,
    score_pdf_f32: *mut f32,
    score_cdf_f32: *mut f32,
    capacity_rows: usize,
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
            let view = pinned_view_from_py(&mut view).map_err(PyValueError::new_err)?;
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
unsafe fn pinned_view_from_py<'a>(
    dst: &'a mut HydraRawMjaiBatchView,
) -> Result<RawMjaiPinnedBatchView<'a>, String> {
    let rows = dst.capacity_rows;
    Ok(RawMjaiPinnedBatchView {
        obs: checked_slice(
            dst.obs_f32,
            checked_field_len(rows, OBS_SIZE, "obs_f32")?,
            "obs_f32",
        )?,
        actions: checked_slice(dst.actions_i64, rows, "actions_i64")?,
        legal: checked_slice(
            dst.legal_u8,
            checked_field_len(rows, HYDRA_ACTION_SPACE, "legal_u8")?,
            "legal_u8",
        )?,
        value: checked_slice(dst.value_f32, rows, "value_f32")?,
        grp: checked_slice(
            dst.grp_f32,
            checked_field_len(rows, GRP_CLASS_COUNT, "grp_f32")?,
            "grp_f32",
        )?,
        oracle: checked_slice(
            dst.oracle_f32,
            checked_field_len(rows, PLAYER_COUNT, "oracle_f32")?,
            "oracle_f32",
        )?,
        oracle_mask: checked_slice(dst.oracle_mask_f32, rows, "oracle_mask_f32")?,
        tenpai: checked_slice(
            dst.tenpai_f32,
            checked_field_len(rows, OPPONENT_COUNT, "tenpai_f32")?,
            "tenpai_f32",
        )?,
        opp_next: checked_slice(
            dst.opp_next_f32,
            checked_field_len(rows, SPATIAL_TARGET_SIZE, "opp_next_f32")?,
            "opp_next_f32",
        )?,
        danger: checked_slice(
            dst.danger_f32,
            checked_field_len(rows, SPATIAL_TARGET_SIZE, "danger_f32")?,
            "danger_f32",
        )?,
        danger_mask: checked_slice(
            dst.danger_mask_f32,
            checked_field_len(rows, SPATIAL_TARGET_SIZE, "danger_mask_f32")?,
            "danger_mask_f32",
        )?,
        score_pdf: checked_slice(
            dst.score_pdf_f32,
            checked_field_len(rows, SCORE_BINS, "score_pdf_f32")?,
            "score_pdf_f32",
        )?,
        score_cdf: checked_slice(
            dst.score_cdf_f32,
            checked_field_len(rows, SCORE_BINS, "score_cdf_f32")?,
            "score_cdf_f32",
        )?,
        capacity_rows: rows,
    })
}

fn checked_field_len(rows: usize, width: usize, name: &str) -> Result<usize, String> {
    rows.checked_mul(width)
        .ok_or_else(|| format!("{name} length overflows usize"))
}

fn checked_slice<'a, T>(ptr: *mut T, len: usize, name: &str) -> Result<&'a mut [T], String> {
    if len == 0 {
        return Ok(&mut []);
    }
    if ptr.is_null() {
        return Err(format!("{name} is null"));
    }
    // SAFETY: PyTorch owns a writable contiguous tensor storage with at least `len` elements.
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

fn io_py_err(err: std::io::Error) -> PyErr {
    PyOSError::new_err(err.to_string())
}

#[pymodule]
fn hydra_raw_mjai_pyo3(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRawMjaiStream>()?;
    module.add_class::<PyRawMjaiStats>()?;
    module.add_class::<PyRawMjaiNext>()?;
    Ok(())
}
