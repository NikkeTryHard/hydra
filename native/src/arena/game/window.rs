//! Claim-window buffering: one answer per responder, then one engine step.
//!
//! riichienv-core treats an absent responder as a pass, so staging one
//! responder at a time requires this buffer. The engine sees the complete map
//! only after every responder has answered.

use std::collections::HashMap;

use hydra_rules::rl::actions::CanonicalAction;
use riichienv_core::action::{Action, Phase};
use riichienv_core::state::legal_actions::GameStateLegalActions;

use super::Game;

pub(super) struct WindowState {
    responders: Vec<u8>,
    answers: Vec<Option<(Action, CanonicalAction)>>,
}

impl Game {
    /// Initialize or refresh the current window and return the next responder.
    pub(super) fn sync_window(&mut self) -> Result<Option<u8>, String> {
        if self.state.phase != Phase::WaitResponse {
            self.window = None;
            return Ok(None);
        }

        let mut responders = Vec::new();
        for &seat in &self.state.active_players {
            if usize::from(seat) >= 4 {
                return Err("response window contains a seat outside 0..4".to_owned());
            }
            if !self.state._get_legal_actions_internal(seat).is_empty() {
                responders.push(seat);
            }
        }
        responders.sort_unstable();
        responders.dedup();
        if responders.is_empty() {
            return Err("response window has no legal responder".to_owned());
        }

        if self
            .window
            .as_ref()
            .is_none_or(|window| window.responders != responders)
        {
            self.window = Some(WindowState {
                responders: responders.clone(),
                answers: vec![None; responders.len()],
            });
        }

        let window = self
            .window
            .as_ref()
            .ok_or_else(|| "claim window was not initialized".to_owned())?;
        let index = window
            .answers
            .iter()
            .position(|answer| answer.is_none())
            .ok_or_else(|| "claim window has no unresolved responder".to_owned())?;
        Ok(Some(window.responders[index]))
    }

    pub(super) fn apply_window(
        &mut self,
        seat: u8,
        action: &CanonicalAction,
    ) -> Result<(), String> {
        let expected = self
            .sync_window()?
            .ok_or_else(|| "claim window has no pending responder".to_owned())?;
        if expected != seat {
            return Err(format!(
                "seat {seat} is not the next pending responder {expected}"
            ));
        }
        let engine = self.resolve_engine_action(seat, action)?;
        let window = self
            .window
            .as_mut()
            .ok_or_else(|| "claim window was not initialized".to_owned())?;
        let index = window
            .responders
            .iter()
            .position(|responder| *responder == seat)
            .ok_or_else(|| format!("seat {seat} is not in the claim window"))?;
        if window.answers[index].is_some() {
            return Err(format!("seat {seat} already answered the claim window"));
        }
        window.answers[index] = Some((engine, action.clone()));

        if window.answers.iter().any(|answer| answer.is_none()) {
            return Ok(());
        }

        let window = self
            .window
            .take()
            .ok_or_else(|| "complete claim window disappeared".to_owned())?;
        let mut engine_actions = HashMap::with_capacity(window.responders.len());
        let mut canonicals = Vec::with_capacity(window.responders.len());
        for (responder, answer) in window.responders.into_iter().zip(window.answers) {
            let (engine, canonical) =
                answer.ok_or_else(|| "claim window ended with a missing answer".to_owned())?;
            engine_actions.insert(responder, engine);
            canonicals.push(canonical);
        }
        let applied: Vec<&CanonicalAction> = canonicals.iter().collect();
        self.state.step(&engine_actions);
        self.history.update(&self.state, &applied)?;
        self.finish_step()
    }
}
