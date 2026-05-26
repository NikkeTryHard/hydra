use hydra_core::action::{HYDRA_ACTION_SPACE, riichienv_to_hydra};
use hydra_core::encoder::OBS_SIZE;
use hydra_core::game_loop::{ActionDecision, ActionSelector};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::Rng;
use rand::rngs::StdRng;
use riichienv_core::action::Action;

use crate::PLAYER_COUNT;

pub(crate) struct PairedArenaSelector<'py> {
    pub(crate) py: Python<'py>,
    pub(crate) infer: Bound<'py, PyAny>,
    pub(crate) temperature: f32,
    pub(crate) rng: StdRng,
    pub(crate) candidate_seats: [bool; PLAYER_COUNT],
    pub(crate) candidate_model_count: usize,
    pub(crate) pending_action: Option<u8>,
    pub(crate) pending_error: Option<PyErr>,
}

impl<'py> PairedArenaSelector<'py> {
    fn model_id_for(&self, seat: u8) -> usize {
        model_id_for_seat(&self.candidate_seats, self.candidate_model_count, seat)
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
        sample_from_scores_with_rng(scores, legal_mask, self.temperature, &mut self.rng)
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

pub(crate) fn first_legal_action_id(legal_mask: &[bool; HYDRA_ACTION_SPACE]) -> Option<u8> {
    legal_mask
        .iter()
        .position(|&legal| legal)
        .and_then(|idx| u8::try_from(idx).ok())
}

pub(crate) fn model_id_for_seat(
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

pub(crate) fn sample_from_scores_with_rng(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_legal_action_id_returns_lowest_legal() {
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[7] = true;
        mask[3] = true;
        assert_eq!(first_legal_action_id(&mask), Some(3));
    }
}
