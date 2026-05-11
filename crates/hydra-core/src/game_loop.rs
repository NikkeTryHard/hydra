//! Game loop runner with proper phase handling and safety tracking.
//!
//! Provides `GameRunner` which orchestrates the full game loop:
//! WaitAct/WaitResponse handling, SafetyInfo updates, and
//! policy-driven action selection.

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
        legal_actions[0]
    }
}

/// Runs a complete game with proper phase handling and safety tracking.
pub struct GameRunner {
    state: GameState,
    safety: [SafetyInfo; 4],
    total_actions: u32,
    rounds_played: u32,
    actions: [Option<Action>; 4],
    legal_buf: Vec<Action>,
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
            actions: [None; 4],
            legal_buf: Vec::with_capacity(46),
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
            actions: [None; 4],
            legal_buf: Vec::with_capacity(46),
        }
    }

    pub fn reset_for_new_game(&mut self, seed: Option<u64>) {
        self.state.reset_for_new_game(seed);
        for safety in &mut self.safety {
            safety.reset();
        }
        self.total_actions = 0;
        self.rounds_played = 1;
        self.actions = [None; 4];
        self.legal_buf.clear();
    }

    #[inline]
    pub fn is_done(&self) -> bool {
        self.state.is_done
    }

    #[inline]
    pub fn total_actions(&self) -> u32 {
        self.total_actions
    }

    #[inline]
    pub fn rounds_played(&self) -> u32 {
        self.rounds_played
    }

    #[inline]
    pub fn scores(&self) -> [i32; 4] {
        std::array::from_fn(|i| self.state.players[i].score)
    }

    /// Get safety info from a specific player's perspective.
    #[inline]
    pub fn safety(&self, player: u8) -> &SafetyInfo {
        &self.safety[player as usize]
    }
}

const MAX_STEPS: u32 = 50_000;

impl GameRunner {
    /// Advance the game by one step. Returns false if game is over.
    pub fn step_once<S: ActionSelector>(&mut self, selector: &mut S) -> bool {
        if self.state.is_done || self.total_actions >= MAX_STEPS {
            return false;
        }

        // Handle round transitions
        if self.state.needs_initialize_next_round {
            self.state.step_unchecked(&[None; 4]);
            self.rounds_played += 1;
            // Reset safety for new round
            for s in &mut self.safety {
                s.reset();
            }
            return !self.state.is_done;
        }

        self.actions = [None; 4];

        match self.state.phase {
            Phase::WaitAct => {
                let pid = self.state.current_player;
                self.state.get_legal_actions_into(pid, &mut self.legal_buf);
                if self.legal_buf.is_empty() {
                    return false;
                }
                let chosen = selector.select_action(pid, &self.legal_buf);
                self.track_action(pid, &chosen);
                self.actions[pid as usize] = Some(chosen);
            }
            Phase::WaitResponse => {
                let n = self.state.active_player_count as usize;
                let mut pids = [0u8; 4];
                pids[..n].copy_from_slice(self.state.active_player_slice());
                for &pid in &pids[..n] {
                    self.state.get_legal_actions_into(pid, &mut self.legal_buf);
                    if self.legal_buf.is_empty() {
                        continue;
                    }
                    let chosen = selector.select_action(pid, &self.legal_buf);
                    self.track_action(pid, &chosen);
                    self.actions[pid as usize] = Some(chosen);
                }
            }
        }

        self.state.step_unchecked(&self.actions);
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
                    // Tedashi = discarded from hand (not the just-drawn tile).
                    // drawn_tile is still set here because track_action runs
                    // BEFORE state.step() clears it.
                    let is_tsumogiri = self.state.drawn_tile == Some(tile136);
                    let is_tedashi = !is_tsumogiri;
                    // Update safety from each OTHER player's perspective
                    for observer in 0..4u8 {
                        if observer == actor {
                            continue;
                        }
                        // Relative-opponent slots are ordered as left-to-right opponents
                        // from the observer's perspective: [observer+1, +2, +3] mod 4.
                        let opp_idx = ((actor + 4 - observer) % 4).wrapping_sub(1) as usize;
                        if opp_idx < 3 {
                            self.safety[observer as usize]
                                .on_discard(tile_type, opp_idx, is_tedashi);
                        }
                    }
                }
            }
            ActionType::Chi | ActionType::Pon | ActionType::Daiminkan => {
                let mut tile_types = [0u8; 4];
                let count = action.consume_count as usize;
                for (i, &t) in action.consume_slice().iter().enumerate() {
                    tile_types[i] = t / 4;
                }
                for s in &mut self.safety {
                    s.on_call(&tile_types[..count]);
                }
            }
            ActionType::Riichi => {
                for observer in 0..4u8 {
                    if observer == actor {
                        continue;
                    }
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
    pub fn run_to_completion<S: ActionSelector>(&mut self, selector: &mut S) {
        while self.step_once(selector) {}
    }
}

#[cfg(test)]
mod tests;
