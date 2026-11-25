//! One scheduled match driven directly by the pinned riichienv-core state.
//!
//! The engine owns scoring, wall shuffling, hand progression, and settlement.
//! This adapter selects pending decisions, maps canonical actions back to
//! engine actions, and buffers a complete claim window before one engine step.

mod actions;
mod history;
mod window;

use std::cmp::Reverse;
use std::collections::HashMap;

use hydra_rules::census::ActionKind;
use hydra_rules::rl::actions::CanonicalAction;
use riichienv_core::action::{Action, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

use self::history::History;
use self::window::WindowState;
use crate::view::{EventArena, Observation};

pub(super) struct Game {
    state: GameState,
    history: History,
    window: Option<WindowState>,
}

impl Game {
    pub(super) fn open(seed: u64) -> Result<Self, String> {
        let state = GameState::new(2, false, Some(seed), 0, GameRule::default_tenhou());
        let history =
            History::new(&state).map_err(|error| format!("seed {seed:016x} history: {error}"))?;
        let mut game = Self {
            state,
            history,
            window: None,
        };
        game.advance_rounds()?;
        Ok(game)
    }

    pub(super) fn is_terminal(&self) -> bool {
        self.state.is_done
    }

    /// Next seat with a decision. Claim windows expose one responder at a
    /// time in ascending seat order until the window can be resolved.
    pub(super) fn pending_seat(&mut self) -> Result<u8, String> {
        self.advance_rounds()?;
        if self.state.is_done {
            return Err("terminal game has no pending decision".to_owned());
        }
        match self.state.phase {
            Phase::WaitAct => {
                let seat = self.state.current_player;
                if usize::from(seat) >= 4 {
                    return Err("draw decision has an invalid current player".to_owned());
                }
                self.window = None;
                Ok(seat)
            }
            Phase::WaitResponse => self
                .sync_window()?
                .ok_or_else(|| "claim window has no pending responder".to_owned()),
        }
    }

    pub(super) fn observe(
        &self,
        seat: u8,
        legal: &[(u32, CanonicalAction)],
    ) -> Result<Observation, String> {
        self.history.observe(&self.state, seat, legal)
    }

    pub(super) fn arena(&self) -> &EventArena {
        self.history.arena()
    }

    pub(super) fn apply(&mut self, seat: u8, action: &CanonicalAction) -> Result<(), String> {
        if self.state.is_done {
            return Err("terminal game cannot apply an action".to_owned());
        }
        if action.actor != seat {
            return Err(format!(
                "action actor {} does not match seat {seat}",
                action.actor
            ));
        }
        self.advance_rounds()?;
        if self.state.is_done {
            return Err("game ended before the staged action".to_owned());
        }
        match self.state.phase {
            Phase::WaitAct => {
                if self.state.current_player != seat {
                    return Err(format!(
                        "seat {seat} is not the current actor {}",
                        self.state.current_player
                    ));
                }
                if action.kind == ActionKind::RiichiDiscard {
                    return self.apply_riichi(seat, action);
                }
                let engine = self.resolve_engine_action(seat, action)?;
                let mut step = HashMap::with_capacity(1);
                step.insert(seat, engine);
                self.state.step(&step);
                self.history.update(&self.state, &[action])?;
                self.window = None;
                self.finish_step()
            }
            Phase::WaitResponse => self.apply_window(seat, action),
        }
    }

    /// Terminal outcome in original absolute seat order. Engine scores are
    /// ranked with the reference east1-seat tie break.
    pub(super) fn outcome_row(
        &self,
        slot: usize,
    ) -> Result<Option<(i32, Vec<i32>, Vec<i32>)>, String> {
        if !self.state.is_done {
            return Ok(None);
        }
        if let Some(error) = &self.state.last_error {
            return Err(error.clone());
        }
        let slot = i32::try_from(slot).map_err(|_| "arena slot exceeds int32".to_owned())?;
        let scores: Vec<i32> = self
            .state
            .players
            .iter()
            .map(|player| player.score)
            .collect();
        let mut seats = [0usize, 1, 2, 3];
        seats.sort_unstable_by_key(|&seat| (Reverse(scores[seat]), seat));
        let mut ranks = vec![0i32; 4];
        for (position, seat) in seats.into_iter().enumerate() {
            ranks[seat] = i32::try_from(position + 1)
                .map_err(|_| "rank position exceeds int32".to_owned())?;
        }
        Ok(Some((slot, scores, ranks)))
    }

    fn advance_rounds(&mut self) -> Result<(), String> {
        loop {
            if let Some(error) = &self.state.last_error {
                return Err(error.clone());
            }
            if self.state.is_done || !self.state.needs_initialize_next_round {
                return Ok(());
            }
            let empty: HashMap<u8, Action> = HashMap::new();
            self.state.step(&empty);
            self.history.update(&self.state, &[])?;
        }
    }

    fn finish_step(&mut self) -> Result<(), String> {
        self.advance_rounds()
    }
}
