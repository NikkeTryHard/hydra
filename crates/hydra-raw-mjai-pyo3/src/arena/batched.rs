use hydra_core::arena::compute_placements;
use hydra_core::game_loop::{GameRunner, StepOutcome};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::PLAYER_COUNT;

use super::metrics::{ArenaSideStats, add_completed_game, add_completed_scores, metrics_dict};
use super::sampling::{PairedArenaSelector, model_id_for_seat, sample_from_scores_with_rng};
use super::shared::{ArenaGame, ArenaRequest};

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

pub(super) fn validate_paired_inputs(
    games_per_seat: usize,
    temperature: f32,
    candidate_model_count: usize,
    batch_decisions: usize,
) -> PyResult<()> {
    validate_arena_inputs(
        games_per_seat,
        temperature,
        candidate_model_count,
        batch_decisions,
    )
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

#[pyfunction]
#[pyo3(signature = (games_per_seat, seed, temperature, candidate_model_count, batch_decisions, infer))]
pub(crate) fn run_paired_arena_batched<'py>(
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

#[pyfunction]
#[pyo3(signature = (games, seed, temperature, candidate_seats, candidate_model_count, infer))]
pub(crate) fn run_paired_arena<'py>(
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
        add_completed_scores(
            &scores,
            &placements,
            &candidate_flags,
            &mut candidate,
            &mut baseline,
            &mut score_delta_sum,
            &mut pt_delta_sum,
        );
    }

    metrics_dict(
        py,
        games,
        candidate,
        baseline,
        score_delta_sum,
        pt_delta_sum,
    )
}
