//! Game loop runner with proper phase handling and safety tracking.
//!
//! Provides `GameRunner` which orchestrates the full game loop:
//! WaitAct/WaitResponse handling, SafetyInfo updates, and
//! policy-driven action selection.

use std::collections::HashMap;

use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

use crate::safety::SafetyInfo;
use crate::seeding::SessionRng;

/// Trait for action selection policies.
/// Implemented by random agents, NN inference, etc.
pub trait ActionSelector {
    /// Select an action from the given legal actions.
    /// `player`: the player who must act (0-3)
    /// `legal_actions`: the available actions
    fn select_action(&mut self, player: u8, legal_actions: &[Action]) -> Action;
}

/// Simple policy that always picks the first legal action.
pub struct FirstActionSelector;

impl ActionSelector for FirstActionSelector {
    fn select_action(&mut self, _player: u8, legal_actions: &[Action]) -> Action {
        legal_actions[0].clone()
    }
}

/// Runs a complete game with proper phase handling and safety tracking.
pub struct GameRunner {
    state: GameState,
    safety: [SafetyInfo; 4], // one SafetyInfo per player perspective
    total_actions: u32,
    rounds_played: u32,
}

impl GameRunner {
    /// Create a new game runner.
    pub fn new(seed: Option<u64>, game_mode: u8) -> Self {
        let rule = GameRule::default_tenhou();
        let state = GameState::new(game_mode, true, seed, 0, rule);
        Self {
            state,
            safety: std::array::from_fn(|_| SafetyInfo::new()),
            total_actions: 0,
            rounds_played: 1,
        }
    }

    /// Create a new game runner using Hydra's deterministic seeding.
    ///
    /// Derives a game seed from the session RNG via SHA-256 KDF,
    /// then passes it to riichienv-core's GameState.
    pub fn new_with_session(session: &mut SessionRng, game_mode: u8) -> Self {
        let game_seed = session.next_game_seed();
        // Convert first 8 bytes of the 32-byte seed to u64 for riichienv
        let seed_u64 = u64::from_le_bytes({
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&game_seed[..8]);
            buf
        });
        let rule = GameRule::default_tenhou();
        let state = GameState::new(game_mode, true, Some(seed_u64), 0, rule);
        Self {
            state,
            safety: std::array::from_fn(|_| SafetyInfo::new()),
            total_actions: 0,
            rounds_played: 1,
        }
    }

    pub fn is_done(&self) -> bool {
        self.state.is_done
    }

    pub fn total_actions(&self) -> u32 {
        self.total_actions
    }

    pub fn rounds_played(&self) -> u32 {
        self.rounds_played
    }

    pub fn scores(&self) -> [i32; 4] {
        std::array::from_fn(|i| self.state.players[i].score)
    }

    /// Get safety info from a specific player's perspective.
    pub fn safety(&self, player: u8) -> &SafetyInfo {
        &self.safety[player as usize]
    }
}

const MAX_STEPS: u32 = 50_000;

impl GameRunner {
    /// Advance the game by one step. Returns false if game is over.
    pub fn step_once(&mut self, selector: &mut dyn ActionSelector) -> bool {
        if self.state.is_done || self.total_actions >= MAX_STEPS {
            return false;
        }

        // Handle round transitions
        if self.state.needs_initialize_next_round {
            self.state.step(&HashMap::new());
            self.rounds_played += 1;
            // Reset safety for new round
            for s in &mut self.safety {
                s.reset();
            }
            return !self.state.is_done;
        }

        let mut actions = HashMap::new();

        match self.state.phase {
            Phase::WaitAct => {
                let pid = self.state.current_player;
                let obs = self.state.get_observation(pid);
                let legal = obs.legal_actions_method();
                if legal.is_empty() { return false; }
                let chosen = selector.select_action(pid, &legal);
                self.track_action(pid, &chosen);
                actions.insert(pid, chosen);
            }
            Phase::WaitResponse => {
                for &pid in &self.state.active_players.clone() {
                    let obs = self.state.get_observation(pid);
                    let legal = obs.legal_actions_method();
                    if legal.is_empty() { continue; }
                    let chosen = selector.select_action(pid, &legal);
                    self.track_action(pid, &chosen);
                    actions.insert(pid, chosen);
                }
            }
        }

        self.state.step(&actions);
        self.total_actions += 1;
        !self.state.is_done
    }
}

impl GameRunner {
    /// Update safety info when an action is taken.
    fn track_action(&mut self, actor: u8, action: &Action) {
        match action.action_type {
            ActionType::Discard => {
                if let Some(tile136) = action.tile {
                    let tile_type = tile136 / 4;
                    // Update safety from each OTHER player's perspective
                    for observer in 0..4u8 {
                        if observer == actor { continue; }
                        let opp_idx = ((actor + 4 - observer) % 4).wrapping_sub(1) as usize;
                        if opp_idx < 3 {
                            self.safety[observer as usize]
                                .on_discard(tile_type, opp_idx, false);
                        }
                    }
                }
            }
            ActionType::Chi | ActionType::Pon | ActionType::Daiminkan => {
                let tile_types: Vec<u8> = action.consume_tiles.iter()
                    .map(|&t| t / 4)
                    .collect();
                for s in &mut self.safety {
                    s.on_call(&tile_types);
                }
            }
            ActionType::Riichi => {
                for observer in 0..4u8 {
                    if observer == actor { continue; }
                    let opp_idx = ((actor + 4 - observer) % 4).wrapping_sub(1) as usize;
                    if opp_idx < 3 {
                        self.safety[observer as usize].on_riichi(opp_idx);
                    }
                }
            }
            _ => {}
        }
    }

    /// Run the full game to completion.
    pub fn run_to_completion(&mut self, selector: &mut dyn ActionSelector) {
        while self.step_once(selector) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_completes_with_first_action() {
        let mut runner = GameRunner::new(Some(42), 0);
        let mut selector = FirstActionSelector;
        runner.run_to_completion(&mut selector);
        assert!(runner.is_done());
        assert!(runner.total_actions() > 20,
            "expected realistic action count, got {}", runner.total_actions());
    }

    #[test]
    fn safety_updated_during_game() {
        let mut runner = GameRunner::new(Some(42), 0);
        let mut selector = FirstActionSelector;
        for _ in 0..20 {
            if !runner.step_once(&mut selector) { break; }
        }
        let has_safety_data = (0..4).any(|p| {
            let s = runner.safety(p);
            s.visible_counts.iter().any(|&c| c > 0)
        });
        assert!(has_safety_data, "safety should be updated during play");
    }

    #[test]
    fn scores_are_plausible() {
        let mut runner = GameRunner::new(Some(99), 0);
        let mut selector = FirstActionSelector;
        runner.run_to_completion(&mut selector);
        let sum: i32 = runner.scores().iter().sum();
        assert!((90_000..=110_000).contains(&sum),
            "score sum {} outside plausible range", sum);
    }

    #[test]
    fn session_seeded_games_are_deterministic() {
        let mut session_a = crate::seeding::SessionRng::new([42u8; 32]);
        let mut session_b = crate::seeding::SessionRng::new([42u8; 32]);

        let mut runner_a = GameRunner::new_with_session(&mut session_a, 0);
        let mut runner_b = GameRunner::new_with_session(&mut session_b, 0);

        let mut sel_a = FirstActionSelector;
        let mut sel_b = FirstActionSelector;

        runner_a.run_to_completion(&mut sel_a);
        runner_b.run_to_completion(&mut sel_b);

        assert_eq!(runner_a.scores(), runner_b.scores());
        assert_eq!(runner_a.total_actions(), runner_b.total_actions());
}
