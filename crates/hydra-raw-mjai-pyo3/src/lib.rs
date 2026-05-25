//! PyO3 extension for direct raw-MJAI fill into caller-owned pinned host buffers.

use std::mem;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use hydra_core::action::{HYDRA_ACTION_SPACE, riichienv_to_hydra};
use hydra_core::arena::compute_placements;
use hydra_core::encoder::OBS_SIZE;
use hydra_core::game_loop::{ActionDecision, ActionSelector, GameRunner, StepOutcome};
use hydra_model::onnx_policy::{OnnxPolicyDevice, OnnxPolicyRuntime};
use hydra_train_exec::raw_mjai_stream::{
    RawMjaiBatchStreamConfig, RawMjaiPinnedBatchView, RawMjaiPinnedStream,
    RawMjaiPinnedStreamStats, RawMjaiStreamSplit,
};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use riichienv_core::action::Action;

const GRP_CLASS_COUNT: usize = 24;
const PLAYER_COUNT: usize = 4;
const OPPONENT_COUNT: usize = 3;
const SPATIAL_TARGET_SIZE: usize = 102;
const SCORE_BINS: usize = 64;

#[derive(Default)]
struct NativeArenaTiming {
    pending: Duration,
    infer: Duration,
    sample: Duration,
    step: Duration,
    completed: Duration,
    loops: u64,
    requests: u64,
    inference_batches: u64,
    worker_threads: usize,
}

impl NativeArenaTiming {
    fn add_to_dict<'py>(&self, dict: &Bound<'py, PyDict>) -> PyResult<()> {
        let total = self.pending + self.infer + self.sample + self.step + self.completed;
        let py_timing = PyDict::new(dict.py());
        py_timing.set_item("pending_ms", duration_ms(self.pending))?;
        py_timing.set_item("infer_ms", duration_ms(self.infer))?;
        py_timing.set_item("sample_ms", duration_ms(self.sample))?;
        py_timing.set_item("step_ms", duration_ms(self.step))?;
        py_timing.set_item("completed_ms", duration_ms(self.completed))?;
        py_timing.set_item("total_profiled_ms", duration_ms(total))?;
        py_timing.set_item("requests", self.requests)?;
        py_timing.set_item("inference_batches", self.inference_batches)?;
        py_timing.set_item("loops", self.loops)?;
        py_timing.set_item("worker_threads", self.worker_threads)?;
        dict.set_item("timing", py_timing)?;
        Ok(())
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
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
    #[pyo3(signature = (data_dirs, batch_size=2048, train_fraction=0.9, worker_threads=20, queue_bound=8, max_games=None, max_samples=None, skip_games=0, augment=false, split="train"))]
    fn new(
        py: Python<'_>,
        data_dirs: RawMjaiPyInputs,
        batch_size: usize,
        train_fraction: f32,
        worker_threads: usize,
        queue_bound: usize,
        max_games: Option<usize>,
        max_samples: Option<usize>,
        skip_games: usize,
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
            skip_games,
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

#[pyclass(skip_from_py_object)]
struct PyPairedArenaMetrics {
    #[pyo3(get)]
    games: usize,
    #[pyo3(get)]
    candidate_winrate: f64,
    #[pyo3(get)]
    baseline_winrate: f64,
    #[pyo3(get)]
    candidate_avg_rank: f64,
    #[pyo3(get)]
    baseline_avg_rank: f64,
    #[pyo3(get)]
    candidate_mean_placement: f64,
    #[pyo3(get)]
    baseline_mean_placement: f64,
    #[pyo3(get)]
    candidate_top2: f64,
    #[pyo3(get)]
    baseline_top2: f64,
    #[pyo3(get)]
    candidate_fourth: f64,
    #[pyo3(get)]
    baseline_fourth: f64,
    #[pyo3(get)]
    candidate_avg_score: f64,
    #[pyo3(get)]
    baseline_avg_score: f64,
    #[pyo3(get)]
    score_delta: f64,
    #[pyo3(get)]
    pt_delta: f64,
}

impl PyPairedArenaMetrics {
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("games", self.games)?;
        dict.set_item("candidate_winrate", self.candidate_winrate)?;
        dict.set_item("baseline_winrate", self.baseline_winrate)?;
        dict.set_item("candidate_avg_rank", self.candidate_avg_rank)?;
        dict.set_item("baseline_avg_rank", self.baseline_avg_rank)?;
        dict.set_item("candidate_mean_placement", self.candidate_mean_placement)?;
        dict.set_item("baseline_mean_placement", self.baseline_mean_placement)?;
        dict.set_item("candidate_top2", self.candidate_top2)?;
        dict.set_item("baseline_top2", self.baseline_top2)?;
        dict.set_item("candidate_fourth", self.candidate_fourth)?;
        dict.set_item("baseline_fourth", self.baseline_fourth)?;
        dict.set_item("candidate_avg_score", self.candidate_avg_score)?;
        dict.set_item("baseline_avg_score", self.baseline_avg_score)?;
        dict.set_item("score_delta", self.score_delta)?;
        dict.set_item("pt_delta", self.pt_delta)?;
        Ok(dict)
    }
}

#[derive(Default)]
struct ArenaSideStats {
    games: usize,
    wins: usize,
    top2: usize,
    fourth: usize,
    placement_sum: u64,
    score_sum: i64,
}

impl ArenaSideStats {
    fn add(&mut self, score: i32, placement: u8) {
        self.games += 1;
        if placement == 0 {
            self.wins += 1;
        }
        if placement <= 1 {
            self.top2 += 1;
        }
        if placement == 3 {
            self.fourth += 1;
        }
        self.placement_sum += u64::from(placement) + 1;
        self.score_sum += i64::from(score);
    }

    fn rate(&self, count: usize) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            count as f64 / self.games as f64
        }
    }

    fn mean_placement(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.placement_sum as f64 / self.games as f64
        }
    }

    fn avg_score(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.score_sum as f64 / self.games as f64
        }
    }
}

struct PairedArenaSelector<'py> {
    py: Python<'py>,
    infer: Bound<'py, PyAny>,
    temperature: f32,
    rng: StdRng,
    candidate_seats: [bool; PLAYER_COUNT],
    candidate_model_count: usize,
    pending_action: Option<u8>,
    pending_error: Option<PyErr>,
}

impl<'py> PairedArenaSelector<'py> {
    fn model_id_for(&self, seat: u8) -> usize {
        if self.candidate_seats[seat as usize] {
            (seat as usize) % self.candidate_model_count.max(1)
        } else {
            self.candidate_model_count
        }
    }

    fn callback_action_id(
        &mut self,
        obs: &[f32; OBS_SIZE],
        legal_mask: &[bool; HYDRA_ACTION_SPACE],
        model_id: usize,
        seat_id: u8,
    ) -> PyResult<u8> {
        let obs_batch = PyList::new(self.py, [PyList::new(self.py, obs.iter().copied())?])?;
        let legal_batch = PyList::new(
            self.py,
            [PyList::new(
                self.py,
                legal_mask.iter().map(|&v| u8::from(v)),
            )?],
        )?;
        let model_ids = PyList::new(self.py, [model_id])?;
        let seat_ids = PyList::new(self.py, [seat_id])?;
        let result = self
            .infer
            .call1((obs_batch, legal_batch, model_ids, seat_ids))?;

        if let Ok(actions) = result.extract::<Vec<u8>>() {
            let Some(&action) = actions.first() else {
                return Err(PyValueError::new_err(
                    "arena inference returned empty action batch",
                ));
            };
            return Ok(action);
        }
        if let Ok(actions) = result.extract::<Vec<i64>>() {
            let Some(&action) = actions.first() else {
                return Err(PyValueError::new_err(
                    "arena inference returned empty action batch",
                ));
            };
            return u8::try_from(action)
                .map_err(|_| PyValueError::new_err(format!("invalid action id {action}")));
        }
        if let Ok(matrix) = result.extract::<Vec<Vec<f32>>>() {
            let Some(row) = matrix.first() else {
                return Err(PyValueError::new_err(
                    "arena inference returned empty logits batch",
                ));
            };
            return self.sample_from_scores(row, legal_mask);
        }
        if let Ok(row) = result.extract::<Vec<f32>>() {
            return self.sample_from_scores(&row, legal_mask);
        }

        Err(PyValueError::new_err(
            "arena inference must return action ids or logits/probs",
        ))
    }

    fn sample_from_scores(
        &mut self,
        scores: &[f32],
        legal_mask: &[bool; HYDRA_ACTION_SPACE],
    ) -> PyResult<u8> {
        if scores.len() != HYDRA_ACTION_SPACE {
            return Err(PyValueError::new_err(format!(
                "expected {HYDRA_ACTION_SPACE} logits/probs, got {}",
                scores.len()
            )));
        }
        let mut max_score = f32::NEG_INFINITY;
        for (&score, &legal) in scores.iter().zip(legal_mask.iter()) {
            if legal && score.is_finite() && score > max_score {
                max_score = score;
            }
        }
        if !max_score.is_finite() {
            return first_legal_action_id(legal_mask)
                .ok_or_else(|| PyValueError::new_err("no legal actions in arena decision"));
        }

        let temp = self.temperature.max(1e-3);
        let mut total = 0.0f32;
        let mut weights = [0.0f32; HYDRA_ACTION_SPACE];
        for (idx, (&score, &legal)) in scores.iter().zip(legal_mask.iter()).enumerate() {
            if legal && score.is_finite() {
                let weight = ((score - max_score) / temp).exp();
                weights[idx] = weight;
                total += weight;
            }
        }
        if total <= 0.0 || !total.is_finite() {
            return first_legal_action_id(legal_mask)
                .ok_or_else(|| PyValueError::new_err("no legal actions in arena decision"));
        }

        let mut draw = self.rng.random::<f32>() * total;
        for (idx, &weight) in weights.iter().enumerate() {
            if weight == 0.0 {
                continue;
            }
            if draw <= weight {
                return Ok(idx as u8);
            }
            draw -= weight;
        }
        first_legal_action_id(legal_mask)
            .ok_or_else(|| PyValueError::new_err("no legal actions in arena decision"))
    }
}

impl ActionSelector for PairedArenaSelector<'_> {
    fn observe_decision(&mut self, decision: ActionDecision<'_>) {
        let model_id = self.model_id_for(decision.seat_id);
        match self.callback_action_id(
            decision.obs,
            decision.legal_mask,
            model_id,
            decision.seat_id,
        ) {
            Ok(action_id) => {
                self.pending_action = Some(action_id);
                self.pending_error = None;
            }
            Err(err) => {
                self.pending_action = None;
                self.pending_error = Some(err);
            }
        }
    }

    fn select_action(&mut self, _player: u8, legal_actions: &[Action]) -> Action {
        let Some(action_id) = self.pending_action.take() else {
            return legal_actions[0];
        };
        if let Some(action) = legal_actions.iter().copied().find(|action| {
            riichienv_to_hydra(action)
                .map(|hydra| hydra.id() == action_id)
                .unwrap_or(false)
        }) {
            return action;
        }
        self.pending_error = Some(PyValueError::new_err(format!(
            "arena inference returned illegal action id {action_id}"
        )));
        legal_actions[0]
    }
}

fn first_legal_action_id(legal_mask: &[bool; HYDRA_ACTION_SPACE]) -> Option<u8> {
    legal_mask
        .iter()
        .position(|&legal| legal)
        .and_then(|idx| u8::try_from(idx).ok())
}

struct ArenaGame {
    runner: GameRunner,
    rng: StdRng,
    candidate_seats: [bool; PLAYER_COUNT],
}

struct ArenaRequest {
    game_idx: usize,
    model_id: usize,
    seat_id: u8,
    obs: [f32; OBS_SIZE],
    legal_mask: [bool; HYDRA_ACTION_SPACE],
}

struct ShardRequest {
    shard_idx: usize,
    local_game_idx: usize,
    model_id: usize,
    obs: [f32; OBS_SIZE],
    legal_mask: [bool; HYDRA_ACTION_SPACE],
}

struct GameShard {
    active: Vec<ArenaGame>,
    candidate: ArenaSideStats,
    baseline: ArenaSideStats,
    score_delta_sum: f64,
    pt_delta_sum: f64,
}

struct ShardStep {
    requests: Vec<ShardRequest>,
    completed: usize,
}

fn normalize_worker_threads(requested: usize, games: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let default_threads = available.saturating_sub(2).clamp(1, 16);
    let wanted = if requested == 0 {
        default_threads
    } else {
        requested
    };
    wanted.max(1).min(games.max(1))
}

fn build_game_shards(games_per_seat: usize, seed: u64, worker_threads: usize) -> Vec<GameShard> {
    let mut shards = (0..worker_threads)
        .map(|_| GameShard {
            active: Vec::new(),
            candidate: ArenaSideStats::default(),
            baseline: ArenaSideStats::default(),
            score_delta_sum: 0.0,
            pt_delta_sum: 0.0,
        })
        .collect::<Vec<_>>();
    for seat in 0..PLAYER_COUNT {
        for game_idx in 0..games_per_seat {
            let sequence = seat * games_per_seat + game_idx;
            let game_seed = seed.wrapping_add(sequence as u64);
            let mut candidate_seats = [false; PLAYER_COUNT];
            candidate_seats[seat] = true;
            shards[sequence % worker_threads].active.push(ArenaGame {
                runner: GameRunner::new(Some(game_seed), 0),
                rng: StdRng::seed_from_u64(game_seed ^ 0x9e37_79b9_7f4a_7c15),
                candidate_seats,
            });
        }
    }
    shards
}

fn collect_shard_requests(
    shard_idx: usize,
    shard: &mut GameShard,
    candidate_model_count: usize,
    max_requests: usize,
) -> PyResult<ShardStep> {
    let mut requests = Vec::with_capacity(max_requests);
    let mut completed = 0usize;
    let mut game_idx = 0usize;
    while game_idx < shard.active.len() && requests.len() < max_requests {
        match shard.active[game_idx].runner.pending_decisions() {
            Ok(decisions) => {
                for decision in decisions {
                    requests.push(ShardRequest {
                        shard_idx,
                        local_game_idx: game_idx,
                        model_id: model_id_for_seat(
                            &shard.active[game_idx].candidate_seats,
                            candidate_model_count,
                            decision.seat_id,
                        ),
                        obs: decision.obs,
                        legal_mask: decision.legal_mask,
                    });
                }
                game_idx += 1;
            }
            Err(StepOutcome::Advanced) => {}
            Err(StepOutcome::Complete) => {
                add_completed_game(
                    &shard.active[game_idx],
                    &mut shard.candidate,
                    &mut shard.baseline,
                    &mut shard.score_delta_sum,
                    &mut shard.pt_delta_sum,
                );
                shard.active.swap_remove(game_idx);
                completed += 1;
            }
            Err(outcome) => {
                return Err(PyRuntimeError::new_err(format!(
                    "arena game did not complete: {outcome:?}"
                )));
            }
        }
    }
    Ok(ShardStep {
        requests,
        completed,
    })
}

fn apply_shard_actions(shard: &mut GameShard, action_rows: &[Vec<u8>]) -> PyResult<()> {
    for (game_idx, action_ids) in action_rows.iter().enumerate() {
        if action_ids.is_empty() || game_idx >= shard.active.len() {
            continue;
        }
        let outcome = shard.active[game_idx]
            .runner
            .step_with_hydra_action_ids(action_ids);
        if matches!(
            outcome,
            StepOutcome::NoLegalAction { .. } | StepOutcome::StepLimitExceeded
        ) {
            return Err(PyRuntimeError::new_err(format!(
                "arena game did not complete: {outcome:?}"
            )));
        }
    }
    Ok(())
}
fn model_id_for_seat(
    candidate_seats: &[bool; PLAYER_COUNT],
    candidate_model_count: usize,
    seat: u8,
) -> usize {
    if candidate_seats[seat as usize] {
        (seat as usize) % candidate_model_count.max(1)
    } else {
        candidate_model_count
    }
}

fn validate_arena_inputs(
    games_per_seat: usize,
    temperature: f32,
    candidate_model_count: usize,
    batch_decisions: usize,
) -> PyResult<()> {
    if games_per_seat == 0 {
        return Err(PyValueError::new_err("games_per_seat must be > 0"));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(PyValueError::new_err("temperature must be finite and > 0"));
    }
    if candidate_model_count == 0 {
        return Err(PyValueError::new_err("candidate_model_count must be > 0"));
    }
    if batch_decisions == 0 {
        return Err(PyValueError::new_err("batch_decisions must be > 0"));
    }
    Ok(())
}

fn py_infer_batch<'py>(
    py: Python<'py>,
    infer: &Bound<'py, PyAny>,
    requests: &[ArenaRequest],
) -> PyResult<Bound<'py, PyAny>> {
    let obs_rows = PyList::empty(py);
    let legal_rows = PyList::empty(py);
    let model_ids = PyList::empty(py);
    let seat_ids = PyList::empty(py);
    for request in requests {
        obs_rows.append(PyList::new(py, request.obs.iter().copied())?)?;
        legal_rows.append(PyList::new(
            py,
            request.legal_mask.iter().map(|&v| u8::from(v)),
        )?)?;
        model_ids.append(request.model_id)?;
        seat_ids.append(request.seat_id)?;
    }
    infer.call1((obs_rows, legal_rows, model_ids, seat_ids))
}

fn parse_action_batch(
    result: Bound<'_, PyAny>,
    requests: &mut [ArenaRequest],
    games: &mut [ArenaGame],
    temperature: f32,
) -> PyResult<Vec<Vec<u8>>> {
    let mut actions = vec![Vec::new(); games.len()];
    if let Ok(ids) = result.extract::<Vec<u8>>() {
        if ids.len() != requests.len() {
            return Err(PyValueError::new_err("arena action batch length mismatch"));
        }
        for (request, action_id) in requests.iter().zip(ids) {
            actions[request.game_idx].push(action_id);
        }
        return Ok(actions);
    }
    if let Ok(ids) = result.extract::<Vec<i64>>() {
        if ids.len() != requests.len() {
            return Err(PyValueError::new_err("arena action batch length mismatch"));
        }
        for (request, action_id) in requests.iter().zip(ids) {
            actions[request.game_idx]
                .push(u8::try_from(action_id).map_err(|_| {
                    PyValueError::new_err(format!("invalid action id {action_id}"))
                })?);
        }
        return Ok(actions);
    }
    let matrix = result.extract::<Vec<Vec<f32>>>()?;
    if matrix.len() != requests.len() {
        return Err(PyValueError::new_err("arena logits batch length mismatch"));
    }
    for (request, scores) in requests.iter_mut().zip(matrix.iter()) {
        let action_id = sample_from_scores_with_rng(
            scores,
            &request.legal_mask,
            temperature,
            &mut games[request.game_idx].rng,
        )?;
        actions[request.game_idx].push(action_id);
    }
    Ok(actions)
}

fn sample_from_scores_with_rng(
    scores: &[f32],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    temperature: f32,
    rng: &mut StdRng,
) -> PyResult<u8> {
    if scores.len() != HYDRA_ACTION_SPACE {
        return Err(PyValueError::new_err(format!(
            "expected {HYDRA_ACTION_SPACE} logits/probs, got {}",
            scores.len()
        )));
    }
    let mut max_score = f32::NEG_INFINITY;
    for (&score, &legal) in scores.iter().zip(legal_mask.iter()) {
        if legal && score.is_finite() && score > max_score {
            max_score = score;
        }
    }
    if !max_score.is_finite() {
        return first_legal_action_id(legal_mask)
            .ok_or_else(|| PyValueError::new_err("no legal actions in arena decision"));
    }

    let temp = temperature.max(1e-3);
    let mut total = 0.0f32;
    let mut weights = [0.0f32; HYDRA_ACTION_SPACE];
    for (idx, (&score, &legal)) in scores.iter().zip(legal_mask.iter()).enumerate() {
        if legal && score.is_finite() {
            let weight = ((score - max_score) / temp).exp();
            weights[idx] = weight;
            total += weight;
        }
    }
    if total <= 0.0 || !total.is_finite() {
        return first_legal_action_id(legal_mask)
            .ok_or_else(|| PyValueError::new_err("no legal actions in arena decision"));
    }

    let mut draw = rng.random::<f32>() * total;
    for (idx, &weight) in weights.iter().enumerate() {
        if weight == 0.0 {
            continue;
        }
        if draw <= weight {
            return Ok(idx as u8);
        }
        draw -= weight;
    }
    first_legal_action_id(legal_mask)
        .ok_or_else(|| PyValueError::new_err("no legal actions in arena decision"))
}

fn add_completed_game(
    game: &ArenaGame,
    candidate: &mut ArenaSideStats,
    baseline: &mut ArenaSideStats,
    score_delta_sum: &mut f64,
    pt_delta_sum: &mut f64,
) {
    let scores = game.runner.scores();
    let placements = compute_placements(scores);
    let mut candidate_score_sum = 0i64;
    let mut candidate_count = 0usize;
    let mut baseline_score_sum = 0i64;
    let mut baseline_count = 0usize;
    for seat in 0..PLAYER_COUNT {
        if game.candidate_seats[seat] {
            candidate.add(scores[seat], placements[seat]);
            candidate_score_sum += i64::from(scores[seat]);
            candidate_count += 1;
        } else {
            baseline.add(scores[seat], placements[seat]);
            baseline_score_sum += i64::from(scores[seat]);
            baseline_count += 1;
        }
    }
    if candidate_count > 0 && baseline_count > 0 {
        let candidate_avg = candidate_score_sum as f64 / candidate_count as f64;
        let baseline_avg = baseline_score_sum as f64 / baseline_count as f64;
        let delta = candidate_avg - baseline_avg;
        *score_delta_sum += delta;
        *pt_delta_sum += delta / 1000.0;
    }
}

fn metrics_dict<'py>(
    py: Python<'py>,
    games: usize,
    candidate: ArenaSideStats,
    baseline: ArenaSideStats,
    score_delta_sum: f64,
    pt_delta_sum: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let metrics = PyPairedArenaMetrics {
        games,
        candidate_winrate: candidate.rate(candidate.wins),
        baseline_winrate: baseline.rate(baseline.wins),
        candidate_avg_rank: candidate.mean_placement(),
        baseline_avg_rank: baseline.mean_placement(),
        candidate_mean_placement: candidate.mean_placement(),
        baseline_mean_placement: baseline.mean_placement(),
        candidate_top2: candidate.rate(candidate.top2),
        baseline_top2: baseline.rate(baseline.top2),
        candidate_fourth: candidate.rate(candidate.fourth),
        baseline_fourth: baseline.rate(baseline.fourth),
        candidate_avg_score: candidate.avg_score(),
        baseline_avg_score: baseline.avg_score(),
        score_delta: score_delta_sum / games as f64,
        pt_delta: pt_delta_sum / games as f64,
    };
    metrics.to_dict(py)
}

#[pyfunction]
#[pyo3(signature = (games_per_seat, seed, temperature, candidate_model_count, batch_decisions, infer))]
fn run_paired_arena_batched<'py>(
    py: Python<'py>,
    games_per_seat: usize,
    seed: u64,
    temperature: f32,
    candidate_model_count: usize,
    batch_decisions: usize,
    infer: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    validate_arena_inputs(
        games_per_seat,
        temperature,
        candidate_model_count,
        batch_decisions,
    )?;
    if !infer.is_callable() {
        return Err(PyValueError::new_err("infer must be callable"));
    }

    let mut active = Vec::with_capacity(games_per_seat * PLAYER_COUNT);
    for seat in 0..PLAYER_COUNT {
        for game_idx in 0..games_per_seat {
            let game_seed = seed.wrapping_add((seat * games_per_seat + game_idx) as u64);
            let mut candidate_seats = [false; PLAYER_COUNT];
            candidate_seats[seat] = true;
            active.push(ArenaGame {
                runner: GameRunner::new(Some(game_seed), 0),
                rng: StdRng::seed_from_u64(game_seed ^ 0x9e37_79b9_7f4a_7c15),
                candidate_seats,
            });
        }
    }

    let total_games = active.len();
    let mut candidate = ArenaSideStats::default();
    let mut baseline = ArenaSideStats::default();
    let mut score_delta_sum = 0.0f64;
    let mut pt_delta_sum = 0.0f64;

    while !active.is_empty() {
        let mut requests = Vec::with_capacity(batch_decisions);
        let mut game_idx = 0usize;
        while game_idx < active.len() && requests.len() < batch_decisions {
            match active[game_idx].runner.pending_decisions() {
                Ok(decisions) => {
                    for decision in decisions {
                        requests.push(ArenaRequest {
                            game_idx,
                            model_id: model_id_for_seat(
                                &active[game_idx].candidate_seats,
                                candidate_model_count,
                                decision.seat_id,
                            ),
                            seat_id: decision.seat_id,
                            obs: decision.obs,
                            legal_mask: decision.legal_mask,
                        });
                    }
                    game_idx += 1;
                }
                Err(StepOutcome::Advanced) => {}
                Err(StepOutcome::Complete) => {
                    add_completed_game(
                        &active[game_idx],
                        &mut candidate,
                        &mut baseline,
                        &mut score_delta_sum,
                        &mut pt_delta_sum,
                    );
                    active.swap_remove(game_idx);
                }
                Err(outcome) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "arena game did not complete: {outcome:?}"
                    )));
                }
            }
        }
        if requests.is_empty() {
            continue;
        }
        let result = py_infer_batch(py, &infer, &requests)?;
        let actions = parse_action_batch(result, &mut requests, &mut active, temperature)?;
        for (idx, action_ids) in actions.iter().enumerate() {
            if action_ids.is_empty() || idx >= active.len() {
                continue;
            }
            let outcome = active[idx].runner.step_with_hydra_action_ids(action_ids);
            if matches!(
                outcome,
                StepOutcome::NoLegalAction { .. } | StepOutcome::StepLimitExceeded
            ) {
                return Err(PyRuntimeError::new_err(format!(
                    "arena game did not complete: {outcome:?}"
                )));
            }
        }
    }

    metrics_dict(
        py,
        total_games,
        candidate,
        baseline,
        score_delta_sum,
        pt_delta_sum,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 arena API keeps explicit positional fields for Python extension boundary"
)]
#[pyfunction]
#[pyo3(signature = (games_per_seat, seed, temperature, candidate_model_paths, baseline_model_path, batch_decisions, device = "cuda:0", worker_threads = 0))]
fn run_paired_arena_rust_native<'py>(
    py: Python<'py>,
    games_per_seat: usize,
    seed: u64,
    temperature: f32,
    candidate_model_paths: Vec<PathBuf>,
    baseline_model_path: PathBuf,
    batch_decisions: usize,
    device: &str,
    worker_threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    hydra_model::ort_init::init_ort_from_env()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    validate_arena_inputs(
        games_per_seat,
        temperature,
        candidate_model_paths.len(),
        batch_decisions,
    )?;
    let worker_threads = normalize_worker_threads(worker_threads, games_per_seat * PLAYER_COUNT);
    let device =
        OnnxPolicyDevice::parse(device).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let mut models = Vec::with_capacity(candidate_model_paths.len() + 1);
    for path in &candidate_model_paths {
        models.push(
            OnnxPolicyRuntime::load_dir_with_device(path, device)
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        );
    }
    models.push(
        OnnxPolicyRuntime::load_dir_with_device(&baseline_model_path, device)
            .map_err(|err| PyValueError::new_err(err.to_string()))?,
    );
    let mut timing = NativeArenaTiming {
        worker_threads,
        ..NativeArenaTiming::default()
    };
    run_paired_arena_native_models(
        py,
        games_per_seat,
        seed,
        temperature,
        candidate_model_paths.len(),
        batch_decisions,
        &mut models,
        worker_threads,
        &mut timing,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "Native arena driver threads validated CLI fields through narrow Rust boundary"
)]
fn run_paired_arena_native_models<'py>(
    py: Python<'py>,
    games_per_seat: usize,
    seed: u64,
    temperature: f32,
    candidate_model_count: usize,
    batch_decisions: usize,
    models: &mut [OnnxPolicyRuntime],
    worker_threads: usize,
    timing: &mut NativeArenaTiming,
) -> PyResult<Bound<'py, PyDict>> {
    let total_games = games_per_seat * PLAYER_COUNT;
    let mut shards = build_game_shards(games_per_seat, seed, worker_threads);
    let mut active_games = total_games;
    let arena_start = Instant::now();
    while active_games > 0 {
        timing.loops += 1;
        let pending_start = Instant::now();
        let per_shard_limit = batch_decisions.div_ceil(worker_threads).max(1);
        let shard_steps = shards
            .par_iter_mut()
            .enumerate()
            .map(|(shard_idx, shard)| {
                collect_shard_requests(shard_idx, shard, candidate_model_count, per_shard_limit)
            })
            .collect::<Vec<_>>();
        let mut requests = Vec::with_capacity(batch_decisions);
        for step in shard_steps {
            let step = step?;
            active_games = active_games.saturating_sub(step.completed);
            requests.extend(step.requests);
        }
        timing.pending += pending_start.elapsed();
        if requests.is_empty() {
            continue;
        }
        timing.requests += requests.len() as u64;
        let infer_start = Instant::now();
        let actions = infer_native_actions(&requests, &mut shards, models, temperature, timing)?;
        timing.infer += infer_start.elapsed();
        let step_start = Instant::now();
        let mut actions_by_shard = shards
            .iter()
            .map(|shard| vec![Vec::<u8>::new(); shard.active.len()])
            .collect::<Vec<_>>();
        for (request, action_id) in requests.iter().zip(actions) {
            if request.local_game_idx < actions_by_shard[request.shard_idx].len() {
                actions_by_shard[request.shard_idx][request.local_game_idx].push(action_id);
            }
        }
        shards
            .par_iter_mut()
            .zip(actions_by_shard)
            .try_for_each(|(shard, action_rows)| apply_shard_actions(shard, &action_rows))?;
        timing.step += step_start.elapsed();
    }
    timing.completed += arena_start.elapsed().saturating_sub(
        timing.pending + timing.infer + timing.sample + timing.step + timing.completed,
    );
    let mut candidate = ArenaSideStats::default();
    let mut baseline = ArenaSideStats::default();
    let mut score_delta_sum = 0.0f64;
    let mut pt_delta_sum = 0.0f64;
    for shard in shards {
        candidate.games += shard.candidate.games;
        candidate.wins += shard.candidate.wins;
        candidate.top2 += shard.candidate.top2;
        candidate.fourth += shard.candidate.fourth;
        candidate.score_sum += shard.candidate.score_sum;
        candidate.placement_sum += shard.candidate.placement_sum;
        baseline.games += shard.baseline.games;
        baseline.wins += shard.baseline.wins;
        baseline.top2 += shard.baseline.top2;
        baseline.fourth += shard.baseline.fourth;
        baseline.score_sum += shard.baseline.score_sum;
        baseline.placement_sum += shard.baseline.placement_sum;
        score_delta_sum += shard.score_delta_sum;
        pt_delta_sum += shard.pt_delta_sum;
    }
    let metrics = metrics_dict(
        py,
        total_games,
        candidate,
        baseline,
        score_delta_sum,
        pt_delta_sum,
    )?;
    timing.add_to_dict(&metrics)?;
    Ok(metrics)
}

fn infer_native_actions(
    requests: &[ShardRequest],
    shards: &mut [GameShard],
    models: &mut [OnnxPolicyRuntime],
    temperature: f32,
    timing: &mut NativeArenaTiming,
) -> PyResult<Vec<u8>> {
    let mut actions = vec![0u8; requests.len()];
    let mut by_model = vec![Vec::<usize>::new(); models.len()];
    for (idx, request) in requests.iter().enumerate() {
        let Some(bucket) = by_model.get_mut(request.model_id) else {
            return Err(PyValueError::new_err(format!(
                "model id {} out of range",
                request.model_id
            )));
        };
        bucket.push(idx);
    }
    for (model_id, indices) in by_model.iter().enumerate() {
        if indices.is_empty() {
            continue;
        }
        let mut obs = Vec::with_capacity(indices.len() * OBS_SIZE);
        for &idx in indices {
            obs.extend_from_slice(&requests[idx].obs);
        }
        timing.inference_batches += 1;
        let logits = models[model_id]
            .policy_logits_batch(&obs)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        for (&request_idx, row) in indices.iter().zip(logits.iter()) {
            let request = &requests[request_idx];
            let sample_start = Instant::now();
            let action_id = sample_from_scores_with_rng(
                row,
                &request.legal_mask,
                temperature,
                &mut shards[request.shard_idx].active[request.local_game_idx].rng,
            )?;
            actions[request_idx] = action_id;
            timing.sample += sample_start.elapsed();
        }
    }
    Ok(actions)
}

#[pyfunction]
#[pyo3(signature = (games, seed, temperature, candidate_seats, candidate_model_count, infer))]
fn run_paired_arena<'py>(
    py: Python<'py>,
    games: usize,
    seed: u64,
    temperature: f32,
    candidate_seats: Vec<usize>,
    candidate_model_count: usize,
    infer: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    if games == 0 {
        return Err(PyValueError::new_err("games must be > 0"));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(PyValueError::new_err("temperature must be finite and > 0"));
    }
    if candidate_model_count == 0 {
        return Err(PyValueError::new_err("candidate_model_count must be > 0"));
    }
    if !infer.is_callable() {
        return Err(PyValueError::new_err("infer must be callable"));
    }

    let mut candidate_flags = [false; PLAYER_COUNT];
    if candidate_seats.is_empty() {
        candidate_flags[0] = true;
    } else {
        for seat in candidate_seats {
            if seat >= PLAYER_COUNT {
                return Err(PyValueError::new_err(format!(
                    "candidate seat {seat} outside 0..{PLAYER_COUNT}"
                )));
            }
            candidate_flags[seat] = true;
        }
    }
    if !candidate_flags.iter().any(|&is_candidate| is_candidate) {
        return Err(PyValueError::new_err(
            "candidate_seats must include at least one seat",
        ));
    }

    let mut candidate = ArenaSideStats::default();
    let mut baseline = ArenaSideStats::default();
    let mut score_delta_sum = 0.0f64;
    let mut pt_delta_sum = 0.0f64;

    for game_idx in 0..games {
        let game_seed = seed.wrapping_add(game_idx as u64);
        let mut selector = PairedArenaSelector {
            py,
            infer: infer.clone(),
            temperature,
            rng: StdRng::seed_from_u64(game_seed ^ 0x9e37_79b9_7f4a_7c15),
            candidate_seats: candidate_flags,
            candidate_model_count,
            pending_action: None,
            pending_error: None,
        };
        let mut runner = GameRunner::new(Some(game_seed), 0);
        let outcome = runner.run_to_completion(&mut selector);
        if let Some(err) = selector.pending_error.take() {
            return Err(err);
        }
        if !runner.is_done() {
            return Err(PyRuntimeError::new_err(format!(
                "arena game {game_idx} did not complete: {outcome:?}"
            )));
        }

        let scores = runner.scores();
        let placements = compute_placements(scores);
        let mut candidate_score_sum = 0i64;
        let mut candidate_count = 0usize;
        let mut baseline_score_sum = 0i64;
        let mut baseline_count = 0usize;
        for seat in 0..PLAYER_COUNT {
            if candidate_flags[seat] {
                candidate.add(scores[seat], placements[seat]);
                candidate_score_sum += i64::from(scores[seat]);
                candidate_count += 1;
            } else {
                baseline.add(scores[seat], placements[seat]);
                baseline_score_sum += i64::from(scores[seat]);
                baseline_count += 1;
            }
        }
        if candidate_count > 0 && baseline_count > 0 {
            let candidate_avg = candidate_score_sum as f64 / candidate_count as f64;
            let baseline_avg = baseline_score_sum as f64 / baseline_count as f64;
            let delta = candidate_avg - baseline_avg;
            score_delta_sum += delta;
            pt_delta_sum += delta / 1000.0;
        }
    }

    let metrics = PyPairedArenaMetrics {
        games,
        candidate_winrate: candidate.rate(candidate.wins),
        baseline_winrate: baseline.rate(baseline.wins),
        candidate_avg_rank: candidate.mean_placement(),
        baseline_avg_rank: baseline.mean_placement(),
        candidate_mean_placement: candidate.mean_placement(),
        baseline_mean_placement: baseline.mean_placement(),
        candidate_top2: candidate.rate(candidate.top2),
        baseline_top2: baseline.rate(baseline.top2),
        candidate_fourth: candidate.rate(candidate.fourth),
        baseline_fourth: baseline.rate(baseline.fourth),
        candidate_avg_score: candidate.avg_score(),
        baseline_avg_score: baseline.avg_score(),
        score_delta: score_delta_sum / games as f64,
        pt_delta: pt_delta_sum / games as f64,
    };
    metrics.to_dict(py)
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    #[test]
    fn arena_side_stats_aggregates_rank_metrics() {
        let mut stats = ArenaSideStats::default();
        stats.add(35_000, 0);
        stats.add(25_000, 1);
        stats.add(15_000, 3);

        assert_eq!(stats.games, 3);
        assert_eq!(stats.rate(stats.wins), 1.0 / 3.0);
        assert_eq!(stats.rate(stats.top2), 2.0 / 3.0);
        assert_eq!(stats.rate(stats.fourth), 1.0 / 3.0);
        assert_eq!(stats.mean_placement(), 7.0 / 3.0);
        assert_eq!(stats.avg_score(), 25_000.0);
    }

    #[test]
    fn first_legal_action_id_returns_lowest_legal() {
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[7] = true;
        mask[3] = true;
        assert_eq!(first_legal_action_id(&mask), Some(3));
    }
}

#[pymodule]
fn hydra_raw_mjai_pyo3(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRawMjaiStream>()?;
    module.add_class::<PyRawMjaiStats>()?;
    module.add_class::<PyPairedArenaMetrics>()?;
    module.add_function(wrap_pyfunction!(run_paired_arena, module)?)?;
    module.add_function(wrap_pyfunction!(run_paired_arena_batched, module)?)?;
    module.add_function(wrap_pyfunction!(run_paired_arena_rust_native, module)?)?;
    module.add_class::<PyRawMjaiNext>()?;
    Ok(())
}
