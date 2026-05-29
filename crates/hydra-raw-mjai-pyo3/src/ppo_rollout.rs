//! Native PPO rollout collection for Python PPO control.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::arena::compute_placements;
use hydra_core::encoder::OBS_SIZE;
use hydra_core::game_loop::{GameRunner, StepOutcome};
use hydra_model::onnx_policy::{OnnxPolicyDevice, OnnxPolicyRuntime};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::{PLAYER_COUNT, duration_ms, normalize_worker_threads};
use rayon::prelude::*;

struct RolloutGame {
    runner: GameRunner,
    rng: StdRng,
    game_id: u64,
    seed: u64,
    rows: Vec<RolloutRow>,
}

struct RolloutRequest {
    shard_idx: usize,
    local_game_idx: usize,
    game_id: u64,
    seed: u64,
    decision: hydra_core::game_loop::PendingDecision,
}

struct RolloutShard {
    active: Vec<RolloutGame>,
    completed_games: Vec<CompletedRolloutGame>,
    next_game_idx: usize,
}

struct RolloutShardStep {
    requests: Vec<RolloutRequest>,
    completed: usize,
}

#[derive(Default)]
struct RolloutTiming {
    pending: Duration,
    infer: Duration,
    sample: Duration,
    step: Duration,
    completed: Duration,
    gather: Duration,
    callback_build: Duration,
    callback_python: Duration,
    callback_parse: Duration,
    action_alloc: Duration,
    action_sample_store: Duration,
    action_apply: Duration,
    pack: Duration,
    sort: Duration,
    loops: u64,
    max_requests_per_batch: u64,
    requests: u64,
    inference_batches: u64,
    min_requests_per_batch: u64,
    small_batches: u64,
}

struct RolloutRow {
    game_id: u64,
    seed: u64,
    obs: [f32; OBS_SIZE],
    legal_mask: [bool; HYDRA_ACTION_SPACE],
    action: u8,
    legal_count: u8,
    player_id: u8,
    seat_id: u8,
    turn: u32,
    old_logits: Option<[f32; HYDRA_ACTION_SPACE]>,
    value_old: Option<f32>,
    old_logprob: Option<f32>,
}

struct CompletedRolloutGame {
    terminal: RolloutTerminal,
    rows: Vec<RolloutRow>,
}

struct RolloutTerminal {
    game_id: u64,
    seed: u64,
    final_scores: [i32; PLAYER_COUNT],
    placements: [u8; PLAYER_COUNT],
}

struct RolloutPolicyOutput {
    logits: Vec<[f32; HYDRA_ACTION_SPACE]>,
    values: Option<Vec<f32>>,
}

trait RolloutPolicyInfer {
    fn logits_batch(
        &mut self,
        requests: &[RolloutRequest],
        timing: &mut RolloutTiming,
    ) -> PyResult<RolloutPolicyOutput>;
}

struct OnnxRolloutPolicy<'a> {
    model: &'a mut OnnxPolicyRuntime,
}

impl RolloutPolicyInfer for OnnxRolloutPolicy<'_> {
    fn logits_batch(
        &mut self,
        requests: &[RolloutRequest],
        timing: &mut RolloutTiming,
    ) -> PyResult<RolloutPolicyOutput> {
        let gather_start = Instant::now();
        let mut obs = Vec::with_capacity(requests.len() * OBS_SIZE);
        for request in requests {
            obs.extend_from_slice(&request.decision.obs);
        }
        timing.gather += gather_start.elapsed();
        let python_start = Instant::now();
        let logits = self
            .model
            .policy_logits_batch(&obs)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        timing.callback_python += python_start.elapsed();
        Ok(RolloutPolicyOutput {
            logits,
            values: None,
        })
    }
}

struct CallbackRolloutPolicy<'py> {
    py: Python<'py>,
    infer: Bound<'py, PyAny>,
}

impl RolloutPolicyInfer for CallbackRolloutPolicy<'_> {
    fn logits_batch(
        &mut self,
        requests: &[RolloutRequest],
        timing: &mut RolloutTiming,
    ) -> PyResult<RolloutPolicyOutput> {
        let gather_start = Instant::now();
        let mut obs = Vec::with_capacity(requests.len() * OBS_SIZE);
        for request in requests {
            obs.extend_from_slice(&request.decision.obs);
        }
        timing.gather += gather_start.elapsed();
        let bytes_start = Instant::now();
        let obs_bytes = PyByteArray::new(self.py, bytemuck::cast_slice(&obs));
        timing.callback_build += bytes_start.elapsed();
        let python_start = Instant::now();
        let raw = self.infer.call1((obs_bytes, requests.len()))?;
        timing.callback_python += python_start.elapsed();
        let parse_start = Instant::now();
        let parsed = parse_policy_output(raw, requests.len());
        timing.callback_parse += parse_start.elapsed();
        parsed
    }
}
fn parse_policy_output(raw: Bound<'_, PyAny>, rows: usize) -> PyResult<RolloutPolicyOutput> {
    if let Ok(buffer) = PyBuffer::<f32>::get(&raw)
        && let Some(values) = buffer.as_slice(raw.py())
    {
        let expected_with_values = rows * (HYDRA_ACTION_SPACE + 1);
        if values.len() == expected_with_values {
            let mut logits = Vec::with_capacity(rows);
            let mut value_old = Vec::with_capacity(rows);
            for row_idx in 0..rows {
                let offset = row_idx * (HYDRA_ACTION_SPACE + 1);
                let mut row_logits = [0.0f32; HYDRA_ACTION_SPACE];
                for action_idx in 0..HYDRA_ACTION_SPACE {
                    row_logits[action_idx] = values[offset + action_idx].get();
                }
                logits.push(row_logits);
                value_old.push(values[offset + HYDRA_ACTION_SPACE].get());
            }
            return Ok(RolloutPolicyOutput {
                logits,
                values: Some(value_old),
            });
        }
        let expected_logits = rows * HYDRA_ACTION_SPACE;
        if values.len() == expected_logits {
            let mut logits = Vec::with_capacity(rows);
            for row_idx in 0..rows {
                let offset = row_idx * HYDRA_ACTION_SPACE;
                let mut row_logits = [0.0f32; HYDRA_ACTION_SPACE];
                for action_idx in 0..HYDRA_ACTION_SPACE {
                    row_logits[action_idx] = values[offset + action_idx].get();
                }
                logits.push(row_logits);
            }
            return Ok(RolloutPolicyOutput {
                logits,
                values: None,
            });
        }
    }
    if let Ok(torch) = raw.py().import("torch")
        && let Ok(tensor_type) = torch.getattr("Tensor")
        && raw.is_instance(&tensor_type)?
    {
        let cpu = raw.call_method1("to", ("cpu",))?;
        let contiguous = cpu.call_method0("contiguous")?;
        let raw_bytes = contiguous.call_method0("numpy")?.call_method0("tobytes")?;
        return parse_policy_output(raw_bytes, rows);
    }
    if let Ok(bytes) = raw.extract::<&[u8]>() {
        let expected_with_values = rows * (HYDRA_ACTION_SPACE + 1) * std::mem::size_of::<f32>();
        if bytes.len() == expected_with_values {
            let values = bytemuck::try_cast_slice::<u8, f32>(bytes).map_err(|err| {
                PyValueError::new_err(format!(
                    "PPO inference callback output must be f32 bytes: {err}"
                ))
            })?;
            let mut logits = Vec::with_capacity(rows);
            let mut value_old = Vec::with_capacity(rows);
            for row in values.chunks_exact(HYDRA_ACTION_SPACE + 1) {
                let mut row_logits = [0.0f32; HYDRA_ACTION_SPACE];
                row_logits.copy_from_slice(&row[..HYDRA_ACTION_SPACE]);
                logits.push(row_logits);
                value_old.push(row[HYDRA_ACTION_SPACE]);
            }
            return Ok(RolloutPolicyOutput {
                logits,
                values: Some(value_old),
            });
        }
    }
    let logits = parse_logits(raw, rows)?;
    Ok(RolloutPolicyOutput {
        logits,
        values: None,
    })
}

fn parse_logits(raw: Bound<'_, PyAny>, rows: usize) -> PyResult<Vec<[f32; HYDRA_ACTION_SPACE]>> {
    if let Ok(bytes) = raw.extract::<&[u8]>() {
        let expected_bytes = rows * HYDRA_ACTION_SPACE * std::mem::size_of::<f32>();
        if bytes.len() != expected_bytes {
            return Err(PyValueError::new_err(format!(
                "PPO inference callback returned {} logit bytes, expected {expected_bytes}",
                bytes.len()
            )));
        }
        let values = bytemuck::try_cast_slice::<u8, f32>(bytes).map_err(|err| {
            PyValueError::new_err(format!(
                "PPO inference callback logits must be f32 bytes: {err}"
            ))
        })?;
        let mut out = Vec::with_capacity(rows);
        for row in values.chunks_exact(HYDRA_ACTION_SPACE) {
            let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
            logits.copy_from_slice(row);
            out.push(logits);
        }
        return Ok(out);
    }

    let values = raw.extract::<Vec<Vec<f32>>>()?;
    if values.len() != rows {
        return Err(PyValueError::new_err(format!(
            "PPO inference callback returned {} rows, expected {rows}",
            values.len()
        )));
    }
    let mut out = Vec::with_capacity(rows);
    for row in values {
        if row.len() != HYDRA_ACTION_SPACE {
            return Err(PyValueError::new_err(format!(
                "PPO inference callback row has {} logits, expected {HYDRA_ACTION_SPACE}",
                row.len()
            )));
        }
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits.copy_from_slice(&row);
        out.push(logits);
    }
    Ok(out)
}

#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 rollout API keeps explicit operator fields across language boundary"
)]
#[pyfunction]
#[pyo3(signature = (games, seed, policy_model_path, batch_decisions, device = "cuda:0", worker_threads = 0, temperature = 1.0))]
pub(crate) fn collect_ppo_rollouts_rust_native<'py>(
    py: Python<'py>,
    games: usize,
    seed: u64,
    policy_model_path: PathBuf,
    batch_decisions: usize,
    device: &str,
    worker_threads: usize,
    temperature: f32,
) -> PyResult<Bound<'py, PyDict>> {
    hydra_model::ort_init::init_ort_from_env()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    validate_rollout_inputs(games, temperature, batch_decisions)?;
    let worker_threads = normalize_worker_threads(worker_threads, games);
    let device =
        OnnxPolicyDevice::parse(device).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let session_load_start = Instant::now();
    let mut model = OnnxPolicyRuntime::load_dir_with_device(&policy_model_path, device)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let session_load = session_load_start.elapsed();
    let mut policy = OnnxRolloutPolicy { model: &mut model };
    let result = collect_ppo_rollouts_native_inner(
        py,
        games,
        seed,
        batch_decisions,
        worker_threads,
        temperature,
        &mut policy,
    )?;
    let timing = result
        .get_item("timing")?
        .ok_or_else(|| PyRuntimeError::new_err("native rollout result missing timing"))?;
    timing
        .cast::<PyDict>()?
        .set_item("session_load_ms", duration_ms(session_load))?;
    Ok(result)
}

fn validate_rollout_inputs(games: usize, temperature: f32, batch_decisions: usize) -> PyResult<()> {
    if games == 0 {
        return Err(PyValueError::new_err("games must be > 0"));
    }
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(PyValueError::new_err("temperature must be finite and > 0"));
    }
    if batch_decisions == 0 {
        return Err(PyValueError::new_err("batch_decisions must be > 0"));
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "PyO3 rollout callback API keeps explicit fields across language boundary"
)]
#[pyfunction]
#[pyo3(signature = (games, seed, batch_decisions, worker_threads, temperature, infer))]
pub(crate) fn collect_ppo_rollouts_with_callback<'py>(
    py: Python<'py>,
    games: usize,
    seed: u64,
    batch_decisions: usize,
    worker_threads: usize,
    temperature: f32,
    infer: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    validate_rollout_inputs(games, temperature, batch_decisions)?;
    let worker_threads = normalize_worker_threads(worker_threads, games);
    let mut policy = CallbackRolloutPolicy { py, infer };
    collect_ppo_rollouts_native_inner(
        py,
        games,
        seed,
        batch_decisions,
        worker_threads,
        temperature,
        &mut policy,
    )
}

fn collect_ppo_rollouts_native_inner<'py>(
    py: Python<'py>,
    games: usize,
    seed: u64,
    batch_decisions: usize,
    worker_threads: usize,
    temperature: f32,
    policy: &mut dyn RolloutPolicyInfer,
) -> PyResult<Bound<'py, PyDict>> {
    let mut shards = build_rollout_shards(games, seed, worker_threads);
    let mut active_games = games;
    let mut timing = RolloutTiming::default();
    let started = Instant::now();
    while active_games > 0 {
        timing.loops += 1;
        let pending_start = Instant::now();
        let per_shard_limit = batch_decisions.div_ceil(worker_threads).max(1);
        let shard_steps = shards
            .par_iter_mut()
            .enumerate()
            .map(|(shard_idx, shard)| {
                collect_rollout_shard_requests(shard_idx, shard, per_shard_limit)
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
        let requests_count = requests.len() as u64;
        timing.requests += requests_count;
        timing.max_requests_per_batch = timing.max_requests_per_batch.max(requests_count);
        timing.min_requests_per_batch = if timing.min_requests_per_batch == 0 {
            requests_count
        } else {
            timing.min_requests_per_batch.min(requests_count)
        };
        if requests.len() < batch_decisions.min(games) / 2 {
            timing.small_batches += 1;
        }
        let infer_start = Instant::now();
        timing.inference_batches += 1;
        let policy_output = policy.logits_batch(&requests, &mut timing)?;
        timing.infer += infer_start.elapsed();
        let action_alloc_start = Instant::now();

        let step_start = Instant::now();
        let mut actions_by_shard = (0..shards.len())
            .map(|_| Vec::<(usize, u8)>::new())
            .collect::<Vec<_>>();
        timing.action_alloc += action_alloc_start.elapsed();
        let action_sample_start = Instant::now();
        for (request_idx, (request, scores)) in
            requests.iter().zip(policy_output.logits.iter()).enumerate()
        {
            let (action, old_logprob) = sample_action_and_logprob(
                scores,
                &request.decision.legal_mask,
                temperature,
                &mut shards[request.shard_idx].active[request.local_game_idx].rng,
            )?;
            actions_by_shard[request.shard_idx].push((request.local_game_idx, action));
            let game = &mut shards[request.shard_idx].active[request.local_game_idx];
            game.rows.push(RolloutRow {
                game_id: request.game_id,
                seed: request.seed,
                obs: request.decision.obs,
                legal_mask: request.decision.legal_mask,
                action,
                legal_count: request.decision.legal_count,
                player_id: request.decision.player_id,
                seat_id: request.decision.seat_id,
                turn: request.decision.turn,
                old_logits: Some(*scores),
                value_old: policy_output
                    .values
                    .as_ref()
                    .map(|values| values[request_idx]),
                old_logprob: Some(old_logprob),
            });
        }
        timing.action_sample_store += action_sample_start.elapsed();
        timing.sample += action_sample_start.elapsed();
        let action_apply_start = Instant::now();
        shards
            .par_iter_mut()
            .zip(actions_by_shard)
            .try_for_each(|(shard, action_rows)| {
                apply_rollout_shard_actions(shard, &action_rows)
            })?;
        timing.action_apply += action_apply_start.elapsed();
        timing.step += step_start.elapsed();
    }
    timing.completed += started
        .elapsed()
        .saturating_sub(timing.pending + timing.infer + timing.sample + timing.step);

    let sort_start = Instant::now();
    let mut rows_by_game = (0..games)
        .map(|_| None)
        .collect::<Vec<Option<Vec<RolloutRow>>>>();
    let mut terminals_by_game = (0..games)
        .map(|_| None)
        .collect::<Vec<Option<RolloutTerminal>>>();
    for shard in shards {
        for completed in shard.completed_games {
            let game_idx = completed.terminal.game_id as usize;
            if game_idx >= games {
                return Err(PyRuntimeError::new_err(
                    "native rollout completed game_id out of range",
                ));
            }
            if rows_by_game[game_idx].is_some() {
                return Err(PyRuntimeError::new_err(
                    "native rollout duplicate completed game",
                ));
            }
            rows_by_game[game_idx] = Some(completed.rows);
            terminals_by_game[game_idx] = Some(completed.terminal);
        }
    }
    let total_rows = rows_by_game
        .iter()
        .map(|rows| rows.as_ref().map_or(0, Vec::len))
        .sum();
    let mut terminals = Vec::<RolloutTerminal>::with_capacity(games);
    let mut game_spans = Vec::<(usize, usize)>::with_capacity(games);
    let mut row_refs = Vec::<&RolloutRow>::with_capacity(total_rows);
    for game_idx in 0..games {
        let game_rows = rows_by_game[game_idx]
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("native rollout missing rows"))?;
        let start = row_refs.len();
        row_refs.extend(game_rows.iter());
        game_spans.push((start, row_refs.len()));
        terminals.push(
            terminals_by_game[game_idx]
                .take()
                .ok_or_else(|| PyRuntimeError::new_err("native rollout missing terminal"))?,
        );
    }
    if row_refs.len() != total_rows {
        return Err(PyRuntimeError::new_err("native rollout row count mismatch"));
    }
    timing.sort += sort_start.elapsed();

    let result = PyDict::new(py);
    result.set_item("schema_version", 1)?;
    result.set_item("contract_version", "ppo_native_rollout_v1")?;
    result.set_item("seed", seed)?;
    result.set_item("games_requested", games)?;
    result.set_item("worker_threads", worker_threads)?;
    result.set_item("elapsed_ms", duration_ms(started.elapsed()))?;
    let py_timing = PyDict::new(py);
    py_timing.set_item("pending_ms", duration_ms(timing.pending))?;
    py_timing.set_item("gather_ms", duration_ms(timing.gather))?;
    py_timing.set_item("callback_build_ms", duration_ms(timing.callback_build))?;
    py_timing.set_item("callback_python_ms", duration_ms(timing.callback_python))?;
    py_timing.set_item("callback_parse_ms", duration_ms(timing.callback_parse))?;
    py_timing.set_item("action_alloc_ms", duration_ms(timing.action_alloc))?;
    py_timing.set_item(
        "action_sample_store_ms",
        duration_ms(timing.action_sample_store),
    )?;
    py_timing.set_item("action_apply_ms", duration_ms(timing.action_apply))?;
    py_timing.set_item("infer_ms", duration_ms(timing.infer))?;
    py_timing.set_item("sample_ms", duration_ms(timing.sample))?;
    py_timing.set_item("step_ms", duration_ms(timing.step))?;
    py_timing.set_item("completed_ms", duration_ms(timing.completed))?;
    py_timing.set_item("requests", timing.requests)?;
    py_timing.set_item("inference_batches", timing.inference_batches)?;
    py_timing.set_item("loops", timing.loops)?;
    py_timing.set_item("sort_ms", duration_ms(timing.sort))?;
    py_timing.set_item("min_requests_per_batch", timing.min_requests_per_batch)?;
    py_timing.set_item("small_batches", timing.small_batches)?;
    py_timing.set_item("max_requests_per_batch", timing.max_requests_per_batch)?;
    result.set_item("timing", &py_timing)?;
    let pack_start = Instant::now();
    let row_count = row_refs.len();
    let mut obs_values = vec![0.0f32; row_count * OBS_SIZE];
    let mut legal_bytes = vec![0u8; row_count * HYDRA_ACTION_SPACE];
    let mut actions = vec![0u8; row_count];
    let mut legal_counts = vec![0u8; row_count];
    let mut player_ids = vec![0u8; row_count];
    let mut seat_ids = vec![0u8; row_count];
    let mut game_ids = vec![0u8; row_count * std::mem::size_of::<u64>()];
    let mut turns = vec![0u8; row_count * std::mem::size_of::<u32>()];
    let mut game_row_starts = vec![0u8; games * std::mem::size_of::<u64>()];
    let mut game_row_ends = vec![0u8; games * std::mem::size_of::<u64>()];
    let mut placements_bytes = vec![0u8; games * PLAYER_COUNT];
    for (game_idx, ((start, end), terminal)) in game_spans.iter().zip(terminals.iter()).enumerate()
    {
        let offset = game_idx * std::mem::size_of::<u64>();
        game_row_starts[offset..offset + std::mem::size_of::<u64>()]
            .copy_from_slice(&(*start as u64).to_ne_bytes());
        game_row_ends[offset..offset + std::mem::size_of::<u64>()]
            .copy_from_slice(&(*end as u64).to_ne_bytes());
        let placement_offset = game_idx * PLAYER_COUNT;
        placements_bytes[placement_offset..placement_offset + PLAYER_COUNT]
            .copy_from_slice(&terminal.placements);
    }
    let cached_policy_scalars = row_refs
        .iter()
        .all(|row| row.value_old.is_some() && row.old_logprob.is_some());
    let cached_policy_logits = row_refs.iter().all(|row| row.old_logits.is_some());
    let mut old_logits_values = if cached_policy_logits {
        vec![0.0f32; row_count * HYDRA_ACTION_SPACE]
    } else {
        Vec::new()
    };
    let mut value_old_values = if cached_policy_scalars {
        vec![0.0f32; row_count]
    } else {
        Vec::new()
    };
    let mut old_logprob_values = if cached_policy_scalars {
        vec![0.0f32; row_count]
    } else {
        Vec::new()
    };
    let mut raw_advantage_values = if cached_policy_scalars {
        vec![0.0f32; row_count]
    } else {
        Vec::new()
    };
    let mut return_values = if cached_policy_scalars {
        vec![0.0f32; row_count]
    } else {
        Vec::new()
    };
    if cached_policy_scalars {
        fill_terminal_gae_by_game(
            &rows_by_game,
            &terminals,
            &game_spans,
            &mut raw_advantage_values,
            &mut return_values,
        )?;
    }
    obs_values
        .par_chunks_mut(OBS_SIZE)
        .zip(legal_bytes.par_chunks_mut(HYDRA_ACTION_SPACE))
        .zip(actions.par_iter_mut())
        .zip(legal_counts.par_iter_mut())
        .zip(player_ids.par_iter_mut())
        .zip(seat_ids.par_iter_mut())
        .zip(game_ids.par_chunks_mut(std::mem::size_of::<u64>()))
        .zip(turns.par_chunks_mut(std::mem::size_of::<u32>()))
        .zip(row_refs.par_iter())
        .for_each(
            |(
                (
                    (
                        (
                            ((((obs_out, legal_out), action_out), legal_count_out), player_id_out),
                            seat_id_out,
                        ),
                        game_id_out,
                    ),
                    turn_out,
                ),
                &row,
            )| {
                obs_out.copy_from_slice(&row.obs);
                for (dst, &legal) in legal_out.iter_mut().zip(row.legal_mask.iter()) {
                    *dst = u8::from(legal);
                }
                *action_out = row.action;
                *legal_count_out = row.legal_count;
                *player_id_out = row.player_id;
                *seat_id_out = row.seat_id;
                game_id_out.copy_from_slice(&row.game_id.to_ne_bytes());
                turn_out.copy_from_slice(&row.turn.to_ne_bytes());
            },
        );
    if cached_policy_scalars {
        value_old_values
            .par_iter_mut()
            .zip(old_logprob_values.par_iter_mut())
            .zip(row_refs.par_iter())
            .for_each(|((value_out, logprob_out), &row)| {
                *value_out = row.value_old.expect("checked cached value");
                *logprob_out = row.old_logprob.expect("checked cached old logprob");
            });
    }
    if cached_policy_logits {
        old_logits_values
            .par_chunks_mut(HYDRA_ACTION_SPACE)
            .zip(row_refs.par_iter())
            .for_each(|(logits_out, &row)| {
                logits_out.copy_from_slice(&row.old_logits.expect("checked cached logits"));
            });
    }
    result.set_item(
        "obs_f32_le",
        PyBytes::new(py, bytemuck::cast_slice(&obs_values)),
    )?;
    result.set_item("legal_mask_u8", PyBytes::new(py, &legal_bytes))?;
    result.set_item("actions", PyBytes::new(py, &actions))?;
    result.set_item("legal_counts", PyBytes::new(py, &legal_counts))?;
    result.set_item("player_ids", PyBytes::new(py, &player_ids))?;
    result.set_item("seat_ids", PyBytes::new(py, &seat_ids))?;
    result.set_item("game_ids_u64_le", PyBytes::new(py, &game_ids))?;
    result.set_item("turns_u32_le", PyBytes::new(py, &turns))?;
    result.set_item("game_row_starts_u64_le", PyBytes::new(py, &game_row_starts))?;
    result.set_item("game_row_ends_u64_le", PyBytes::new(py, &game_row_ends))?;
    result.set_item("placements_u8", PyBytes::new(py, &placements_bytes))?;
    if cached_policy_logits {
        result.set_item(
            "old_logits_f32_le",
            PyBytes::new(py, bytemuck::cast_slice(&old_logits_values)),
        )?;
    }
    if cached_policy_scalars {
        result.set_item(
            "value_old_f32_le",
            PyBytes::new(py, bytemuck::cast_slice(&value_old_values)),
        )?;
        result.set_item(
            "old_logprob_f32_le",
            PyBytes::new(py, bytemuck::cast_slice(&old_logprob_values)),
        )?;
        result.set_item(
            "raw_advantages_f32_le",
            PyBytes::new(py, bytemuck::cast_slice(&raw_advantage_values)),
        )?;
        result.set_item(
            "returns_f32_le",
            PyBytes::new(py, bytemuck::cast_slice(&return_values)),
        )?;
    }
    result.set_item("row_count", row_count)?;
    timing.pack += pack_start.elapsed();
    py_timing.set_item("pack_ms", duration_ms(timing.pack))?;
    let py_rows = PyList::empty(py);
    if row_count <= 256 {
        for (row_idx, row) in row_refs.iter().enumerate() {
            let item = PyDict::new(py);
            item.set_item("row_idx", row_idx)?;
            item.set_item("game_id", row.game_id)?;
            item.set_item("seed", row.seed)?;
            item.set_item("obs", PyList::new(py, row.obs.iter().copied())?)?;
            item.set_item(
                "legal_mask",
                PyList::new(py, row.legal_mask.iter().copied())?,
            )?;
            item.set_item("action", row.action)?;
            item.set_item("legal_count", row.legal_count)?;
            item.set_item("player_id", row.player_id)?;
            item.set_item("seat_id", row.seat_id)?;
            item.set_item("turn", row.turn)?;
            py_rows.append(item)?;
        }
    }
    let py_games = PyList::empty(py);
    for terminal in terminals {
        let item = PyDict::new(py);
        item.set_item("game_id", terminal.game_id)?;
        item.set_item("seed", terminal.seed)?;
        item.set_item(
            "final_scores",
            PyList::new(py, terminal.final_scores.iter().copied())?,
        )?;
        item.set_item(
            "placements",
            PyList::new(py, terminal.placements.iter().copied())?,
        )?;
        py_games.append(item)?;
    }
    result.set_item("rows", py_rows)?;
    result.set_item("games", py_games)?;
    Ok(result)
}
const PPO_GAE_GAMMA: f32 = 0.995;
const PPO_GAE_LAMBDA: f32 = 0.95;
const PPO_PLACEMENT_UTILITY: [f32; PLAYER_COUNT] = [1.0, 0.3, -0.3, -1.0];

fn fill_terminal_gae_by_game(
    rows_by_game: &[Option<Vec<RolloutRow>>],
    terminals: &[RolloutTerminal],
    game_spans: &[(usize, usize)],
    raw_advantages: &mut [f32],
    returns: &mut [f32],
) -> PyResult<()> {
    let discount = PPO_GAE_GAMMA * PPO_GAE_LAMBDA;
    for (game_idx, (terminal, &(start, _end))) in
        terminals.iter().zip(game_spans.iter()).enumerate()
    {
        let game_rows = rows_by_game[game_idx]
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("native rollout missing rows"))?;
        for player in 0..PLAYER_COUNT {
            let mut running = 0.0f32;
            let mut next_value = 0.0f32;
            let mut has_next = false;
            let reward = PPO_PLACEMENT_UTILITY[usize::from(terminal.placements[player])];
            for (local_idx, row) in game_rows.iter().enumerate().rev() {
                if usize::from(row.player_id) != player {
                    continue;
                }
                let value = row.value_old.ok_or_else(|| {
                    PyRuntimeError::new_err("native rollout missing cached value")
                })?;
                let delta = (if has_next { 0.0 } else { reward })
                    + if has_next {
                        PPO_GAE_GAMMA * next_value
                    } else {
                        0.0
                    }
                    - value;
                running = delta + if has_next { discount * running } else { 0.0 };
                let row_idx = start + local_idx;
                raw_advantages[row_idx] = running;
                returns[row_idx] = running + value;
                next_value = value;
                has_next = true;
            }
        }
    }
    Ok(())
}

fn build_rollout_shards(games: usize, seed: u64, worker_threads: usize) -> Vec<RolloutShard> {
    let mut shards = (0..worker_threads)
        .map(|_| RolloutShard {
            active: Vec::new(),
            completed_games: Vec::new(),
            next_game_idx: 0,
        })
        .collect::<Vec<_>>();
    for game_idx in 0..games {
        let game_seed = seed.wrapping_add(game_idx as u64);
        shards[game_idx % worker_threads].active.push(RolloutGame {
            runner: GameRunner::new(Some(game_seed), 0),
            rng: StdRng::seed_from_u64(game_seed ^ 0x9e37_79b9_7f4a_7c15),
            game_id: game_idx as u64,
            seed: game_seed,
            rows: Vec::new(),
        });
    }
    shards
}

fn collect_rollout_shard_requests(
    shard_idx: usize,
    shard: &mut RolloutShard,
    max_requests: usize,
) -> PyResult<RolloutShardStep> {
    let mut requests = Vec::with_capacity(max_requests);
    let mut completed = 0usize;
    if shard.active.is_empty() {
        return Ok(RolloutShardStep {
            requests,
            completed,
        });
    }
    let mut inspected = 0usize;
    shard.next_game_idx %= shard.active.len();
    while !shard.active.is_empty()
        && requests.len() < max_requests
        && inspected < shard.active.len()
    {
        let game_idx = shard.next_game_idx % shard.active.len();
        match shard.active[game_idx].runner.pending_decisions() {
            Ok(decisions) => {
                for decision in decisions {
                    if requests.len() >= max_requests {
                        break;
                    }
                    requests.push(RolloutRequest {
                        shard_idx,
                        local_game_idx: game_idx,
                        game_id: shard.active[game_idx].game_id,
                        seed: shard.active[game_idx].seed,
                        decision,
                    });
                }
                shard.next_game_idx = (game_idx + 1) % shard.active.len();
                inspected += 1;
            }
            Err(StepOutcome::Advanced) => {}
            Err(StepOutcome::Complete) => {
                let completed_game = shard.active.swap_remove(game_idx);
                let scores = completed_game.runner.scores();
                let placements = compute_placements(scores);
                shard.completed_games.push(CompletedRolloutGame {
                    terminal: RolloutTerminal {
                        game_id: completed_game.game_id,
                        seed: completed_game.seed,
                        final_scores: scores,
                        placements,
                    },
                    rows: completed_game.rows,
                });
                completed += 1;
                if shard.active.is_empty() {
                    shard.next_game_idx = 0;
                } else {
                    shard.next_game_idx = game_idx % shard.active.len();
                }
            }
            Err(outcome) => {
                return Err(PyRuntimeError::new_err(format!(
                    "PPO rollout game did not complete: {outcome:?}"
                )));
            }
        }
    }
    Ok(RolloutShardStep {
        requests,
        completed,
    })
}

fn first_legal_rollout_action_id(legal_mask: &[bool; HYDRA_ACTION_SPACE]) -> Option<u8> {
    legal_mask
        .iter()
        .position(|&legal| legal)
        .map(|idx| idx as u8)
}

fn sample_action_and_logprob(
    scores: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    temperature: f32,
    rng: &mut StdRng,
) -> PyResult<(u8, f32)> {
    let mut max_score = f32::NEG_INFINITY;
    for (&score, &legal) in scores.iter().zip(legal_mask.iter()) {
        if legal && score.is_finite() && score > max_score {
            max_score = score;
        }
    }
    if !max_score.is_finite() {
        let action = first_legal_rollout_action_id(legal_mask)
            .ok_or_else(|| PyValueError::new_err("no legal actions in arena decision"))?;
        return Ok((action, 0.0));
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
        let action = first_legal_rollout_action_id(legal_mask)
            .ok_or_else(|| PyValueError::new_err("no legal actions in arena decision"))?;
        return Ok((action, 0.0));
    }

    let mut draw = rng.random::<f32>() * total;
    for (idx, &weight) in weights.iter().enumerate() {
        if weight == 0.0 {
            continue;
        }
        if draw <= weight {
            return Ok((idx as u8, (scores[idx] - max_score) / temp - total.ln()));
        }
        draw -= weight;
    }
    let action = first_legal_rollout_action_id(legal_mask)
        .ok_or_else(|| PyValueError::new_err("no legal actions in arena decision"))?;
    let action_idx = action as usize;
    let old_logprob = if scores[action_idx].is_finite() {
        (scores[action_idx] - max_score) / temp - total.ln()
    } else {
        0.0
    };
    Ok((action, old_logprob))
}

#[cfg(test)]
fn masked_action_logprob(
    scores: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    action: u8,
    temperature: f32,
) -> PyResult<f32> {
    let action_idx = action as usize;
    if action_idx >= HYDRA_ACTION_SPACE || !legal_mask[action_idx] {
        return Err(PyRuntimeError::new_err(
            "PPO rollout sampled illegal action",
        ));
    }
    let mut max_score = f32::NEG_INFINITY;
    for (&score, &legal) in scores.iter().zip(legal_mask.iter()) {
        if legal && score.is_finite() && score > max_score {
            max_score = score;
        }
    }
    if !max_score.is_finite() {
        return Ok(0.0);
    }
    let temp = temperature.max(1e-3);
    let mut total = 0.0f32;
    for (&score, &legal) in scores.iter().zip(legal_mask.iter()) {
        if legal && score.is_finite() {
            total += ((score - max_score) / temp).exp();
        }
    }
    if total <= 0.0 || !total.is_finite() || !scores[action_idx].is_finite() {
        return Ok(0.0);
    }
    Ok((scores[action_idx] - max_score) / temp - total.ln())
}

fn apply_rollout_shard_actions(
    shard: &mut RolloutShard,
    action_rows: &[(usize, u8)],
) -> PyResult<()> {
    let mut cursor = action_rows.len();
    while cursor > 0 {
        cursor -= 1;
        let game_idx = action_rows[cursor].0;
        let mut action_ids = Vec::<u8>::with_capacity(4);
        action_ids.push(action_rows[cursor].1);
        while cursor > 0 && action_rows[cursor - 1].0 == game_idx {
            cursor -= 1;
            action_ids.push(action_rows[cursor].1);
        }
        action_ids.reverse();
        if game_idx >= shard.active.len() {
            continue;
        }
        let outcome = shard.active[game_idx]
            .runner
            .step_with_hydra_action_ids(&action_ids);
        if matches!(
            outcome,
            StepOutcome::NoLegalAction { .. } | StepOutcome::StepLimitExceeded
        ) {
            return Err(PyRuntimeError::new_err(format!(
                "PPO rollout game did not complete: {outcome:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use hydra_core::game_loop::StepOutcome;
    #[test]
    fn rollout_shards_assign_all_games_once() {
        let shards = build_rollout_shards(10, 42, 3);
        let mut game_ids = shards
            .iter()
            .flat_map(|shard| shard.active.iter().map(|game| game.game_id))
            .collect::<Vec<_>>();
        game_ids.sort_unstable();
        assert_eq!(game_ids, (0..10).collect::<Vec<_>>());
        assert_eq!(
            shards[0]
                .active
                .iter()
                .map(|game| game.game_id)
                .collect::<Vec<_>>(),
            vec![0, 3, 6, 9]
        );
        assert_eq!(
            shards[1]
                .active
                .iter()
                .map(|game| game.game_id)
                .collect::<Vec<_>>(),
            vec![1, 4, 7]
        );
        assert_eq!(
            shards[2]
                .active
                .iter()
                .map(|game| game.game_id)
                .collect::<Vec<_>>(),
            vec![2, 5, 8]
        );
    }

    #[test]
    fn rollout_rows_preserve_game_then_local_decision_order() {
        let rows_by_game = [
            Some(vec![test_row(0, 0, 0, 0), test_row(0, 0, 2, 0)]),
            Some(vec![test_row(1, 0, 1, 1), test_row(1, 0, 0, 2)]),
        ];
        let row_refs = rows_by_game
            .iter()
            .flat_map(|rows| rows.as_ref().expect("rows").iter())
            .collect::<Vec<_>>();
        assert_eq!(
            row_refs
                .iter()
                .enumerate()
                .map(|(row_idx, row)| (row.game_id, row.turn, row.player_id, row_idx))
                .collect::<Vec<_>>(),
            vec![(0, 0, 0, 0), (0, 2, 0, 1), (1, 1, 1, 2), (1, 0, 2, 3)]
        );
    }

    #[test]
    fn fused_sample_preserves_action_and_logprob() {
        let mut scores = [0.0f32; HYDRA_ACTION_SPACE];
        for (idx, score) in scores.iter_mut().enumerate() {
            *score = (idx as f32 * 0.037).sin() * 3.0;
        }
        scores[2] = f32::NAN;
        scores[7] = f32::INFINITY;
        scores[11] = f32::NEG_INFINITY;
        let mut legal_mask = [false; HYDRA_ACTION_SPACE];
        for idx in [0usize, 3, 5, 8, 13, 21, 34, 42] {
            legal_mask[idx] = true;
        }
        let mut sample_rng = StdRng::seed_from_u64(12345);
        let mut fused_rng = StdRng::seed_from_u64(12345);
        let action = crate::arena::sampling::sample_from_scores_with_rng(
            &scores,
            &legal_mask,
            0.7,
            &mut sample_rng,
        )
        .expect("sample action");
        let old_logprob =
            masked_action_logprob(&scores, &legal_mask, action, 0.7).expect("logprob");
        let (fused_action, fused_logprob) =
            sample_action_and_logprob(&scores, &legal_mask, 0.7, &mut fused_rng)
                .expect("fused sample");
        assert_eq!(fused_action, action);
        assert!((fused_logprob - old_logprob).abs() <= 1e-6);
    }

    #[test]
    fn fused_sample_keeps_no_finite_scores_fallback() {
        let scores = [f32::NAN; HYDRA_ACTION_SPACE];
        let mut legal_mask = [false; HYDRA_ACTION_SPACE];
        legal_mask[4] = true;
        legal_mask[9] = true;
        let mut rng = StdRng::seed_from_u64(1);
        let (action, old_logprob) = sample_action_and_logprob(&scores, &legal_mask, 1.0, &mut rng)
            .expect("fallback sample");
        assert_eq!(action, 4);
        assert_eq!(old_logprob, 0.0);
    }

    #[test]
    fn first_legal_rollout_completes_without_model_inference() {
        let mut shards = build_rollout_shards(1, 2026052705, 1);
        let mut active_games = 1usize;
        let mut loops = 0usize;
        while active_games > 0 {
            loops += 1;
            assert!(loops < 1_000, "rollout should complete without spinning");
            let step = collect_rollout_shard_requests(0, &mut shards[0], 1).expect("requests");
            active_games = active_games.saturating_sub(step.completed);
            let mut actions = Vec::<(usize, u8)>::new();
            for request in step.requests {
                let action = request
                    .decision
                    .legal_mask
                    .iter()
                    .position(|&legal| legal)
                    .expect("legal action") as u8;
                actions.push((request.local_game_idx, action));
            }
            apply_rollout_shard_actions(&mut shards[0], &actions).expect("apply actions");
        }
        assert_eq!(shards[0].completed_games.len(), 1);
        assert!(loops > 20);
    }
    #[test]
    fn wait_act_actions_advance_game_runner() {
        let mut game = GameRunner::new(Some(20260527), 0);
        for _ in 0..8 {
            match game.pending_decisions() {
                Ok(decisions) => {
                    assert_eq!(decisions.len(), 1);
                    let action = decisions[0]
                        .legal_mask
                        .iter()
                        .position(|&legal| legal)
                        .expect("legal action exists") as u8;
                    assert!(matches!(
                        game.step_with_hydra_action_ids(&[action]),
                        StepOutcome::Advanced
                    ));
                    return;
                }
                Err(StepOutcome::Advanced) => continue,
                Err(outcome) => panic!("unexpected startup outcome {outcome:?}"),
            }
        }
        panic!("runner did not reach WaitAct decision");
    }

    fn test_row(game_id: u64, _local_idx: u64, turn: u32, player_id: u8) -> RolloutRow {
        RolloutRow {
            game_id,
            seed: game_id,
            obs: [0.0; OBS_SIZE],
            legal_mask: [true; HYDRA_ACTION_SPACE],
            action: 0,
            legal_count: HYDRA_ACTION_SPACE as u8,
            player_id,
            seat_id: player_id,
            turn,
            old_logits: None,
            value_old: None,
            old_logprob: None,
        }
    }
}
