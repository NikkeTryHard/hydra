//! Native PPO rollout collection for Python PPO control.

use std::path::PathBuf;
use std::time::Instant;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::arena::compute_placements;
use hydra_core::encoder::OBS_SIZE;
use hydra_core::game_loop::{GameRunner, StepOutcome};
use hydra_model::onnx_policy::{OnnxPolicyDevice, OnnxPolicyRuntime};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::arena::sample_from_scores_with_rng;
use crate::{PLAYER_COUNT, duration_ms, normalize_worker_threads};

struct RolloutGame {
    runner: GameRunner,
    rng: StdRng,
    game_id: u64,
    seed: u64,
}

struct RolloutRow {
    row_idx: u64,
    game_id: u64,
    seed: u64,
    obs: [f32; OBS_SIZE],
    legal_mask: [bool; HYDRA_ACTION_SPACE],
    action: u8,
    legal_count: u8,
    player_id: u8,
    seat_id: u8,
    turn: u32,
}

struct RolloutTerminal {
    game_id: u64,
    seed: u64,
    final_scores: [i32; PLAYER_COUNT],
    placements: [u8; PLAYER_COUNT],
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
    let mut model = OnnxPolicyRuntime::load_dir_with_device(&policy_model_path, device)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    collect_ppo_rollouts_native_inner(
        py,
        games,
        seed,
        batch_decisions,
        worker_threads,
        temperature,
        &mut model,
    )
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

fn collect_ppo_rollouts_native_inner<'py>(
    py: Python<'py>,
    games: usize,
    seed: u64,
    batch_decisions: usize,
    worker_threads: usize,
    temperature: f32,
    model: &mut OnnxPolicyRuntime,
) -> PyResult<Bound<'py, PyDict>> {
    let mut active = (0..games)
        .map(|game_idx| {
            let game_seed = seed.wrapping_add(game_idx as u64);
            RolloutGame {
                runner: GameRunner::new(Some(game_seed), 0),
                rng: StdRng::seed_from_u64(game_seed ^ 0x9e37_79b9_7f4a_7c15),
                game_id: game_idx as u64,
                seed: game_seed,
            }
        })
        .collect::<Vec<_>>();
    let mut rows = Vec::<RolloutRow>::new();
    let mut terminals = Vec::<RolloutTerminal>::with_capacity(games);
    let mut row_idx = 0u64;
    let started = Instant::now();
    while !active.is_empty() {
        let mut requests = Vec::with_capacity(batch_decisions);
        let mut game_idx = 0usize;
        while game_idx < active.len() && requests.len() < batch_decisions {
            match active[game_idx].runner.pending_decisions() {
                Ok(decisions) => {
                    for decision in decisions {
                        requests.push((game_idx, decision));
                    }
                    game_idx += 1;
                }
                Err(StepOutcome::Advanced) => {}
                Err(StepOutcome::Complete) => {
                    let scores = active[game_idx].runner.scores();
                    let placements = compute_placements(scores);
                    terminals.push(RolloutTerminal {
                        game_id: active[game_idx].game_id,
                        seed: active[game_idx].seed,
                        final_scores: scores,
                        placements,
                    });
                    active.swap_remove(game_idx);
                }
                Err(outcome) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "PPO rollout game did not complete: {outcome:?}"
                    )));
                }
            }
        }
        if requests.is_empty() {
            continue;
        }
        let mut obs = Vec::with_capacity(requests.len() * OBS_SIZE);
        for (_, decision) in &requests {
            obs.extend_from_slice(&decision.obs);
        }
        let logits = model
            .policy_logits_batch(&obs)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut actions_by_game = vec![Vec::<u8>::new(); active.len()];
        for ((game_idx, decision), scores) in requests.iter().zip(logits.iter()) {
            let action = sample_from_scores_with_rng(
                scores,
                &decision.legal_mask,
                temperature,
                &mut active[*game_idx].rng,
            )?;
            actions_by_game[*game_idx].push(action);
            rows.push(RolloutRow {
                row_idx,
                game_id: active[*game_idx].game_id,
                seed: active[*game_idx].seed,
                obs: decision.obs,
                legal_mask: decision.legal_mask,
                action,
                legal_count: decision.legal_count,
                player_id: decision.player_id,
                seat_id: decision.seat_id,
                turn: decision.turn,
            });
            row_idx += 1;
        }
        for (idx, action_ids) in actions_by_game.iter().enumerate() {
            if action_ids.is_empty() || idx >= active.len() {
                continue;
            }
            let outcome = active[idx].runner.step_with_hydra_action_ids(action_ids);
            if matches!(
                outcome,
                StepOutcome::NoLegalAction { .. } | StepOutcome::StepLimitExceeded
            ) {
                return Err(PyRuntimeError::new_err(format!(
                    "PPO rollout game did not complete: {outcome:?}"
                )));
            }
        }
    }
    rows.sort_by_key(|row| row.row_idx);
    terminals.sort_by_key(|game| game.game_id);
    let result = PyDict::new(py);
    result.set_item("schema_version", 1)?;
    result.set_item("contract_version", "ppo_native_rollout_v1")?;
    result.set_item("seed", seed)?;
    result.set_item("games_requested", games)?;
    result.set_item("worker_threads", worker_threads)?;
    result.set_item("elapsed_ms", duration_ms(started.elapsed()))?;
    let py_rows = PyList::empty(py);
    for row in rows {
        let item = PyDict::new(py);
        item.set_item("row_idx", row.row_idx)?;
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
