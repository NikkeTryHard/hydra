//! PyO3 extension for direct raw-MJAI fill into caller-owned pinned host buffers.

use std::mem;
use std::path::PathBuf;
use std::sync::Mutex;

use hydra_core::action::{HYDRA_ACTION_SPACE, riichienv_to_hydra};
use hydra_core::arena::compute_placements;
use hydra_core::encoder::OBS_SIZE;
use hydra_core::game_loop::{ActionDecision, ActionSelector, GameRunner};
use hydra_train_exec::raw_mjai_stream::{
    RawMjaiBatchStreamConfig, RawMjaiPinnedBatchView, RawMjaiPinnedStream,
    RawMjaiPinnedStreamStats, RawMjaiStreamSplit,
};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use riichienv_core::action::Action;

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
    module.add_class::<PyRawMjaiNext>()?;
    Ok(())
}
