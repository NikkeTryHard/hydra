use std::path::PathBuf;
use std::time::{Duration, Instant};

use hydra_core::encoder::OBS_SIZE;
use hydra_core::game_loop::{GameRunner, StepOutcome};
use hydra_model::onnx_policy::{OnnxPolicyDevice, OnnxPolicyRuntime};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::{PLAYER_COUNT, duration_ms, normalize_worker_threads};

use super::batched::validate_paired_inputs;
use super::metrics::{ArenaSideStats, add_completed_game, metrics_dict};
use super::sampling::{model_id_for_seat, sample_from_scores_with_rng};
use super::shared::{ArenaGame, ShardRequest};

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

#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 arena API keeps explicit positional fields for Python extension boundary"
)]
#[pyfunction]
#[pyo3(signature = (games_per_seat, seed, temperature, candidate_model_paths, baseline_model_path, batch_decisions, device = "cuda:0", worker_threads = 0))]
pub(crate) fn run_paired_arena_rust_native<'py>(
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
    validate_paired_inputs(
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
