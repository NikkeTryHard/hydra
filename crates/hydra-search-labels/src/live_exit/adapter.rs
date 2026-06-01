use hydra_core::bridge::encode_observation_ref;
use hydra_core::encoder::{OBS_SIZE, ObservationEncoder};
use hydra_core::safety::SafetyInfo;
use riichienv_core::action::{Action, ActionType};
use riichienv_core::observation::Observation;
use riichienv_core::state::GameState;

use super::hash::obs_hash;

/// Adapter trait for generating child public observations after a discard.
///
/// Implementors must produce the public observation tensor that the value
/// head would see after the root player discards a given tile.  This is the
/// main blocked surface identified by Agent 22 -- callers must provide a
/// concrete implementation that clones the game state, applies the discard,
/// and re-encodes without leaking hidden state.
pub trait ExitSearchAdapter {
    /// Returns the info-state hash for the root player at the current state.
    fn root_hash(&self, state: &GameState, player: u8, obs_encoded: &[f32; OBS_SIZE]) -> u64;

    /// Produces the public observation after the root player discards `action`.
    ///
    /// Returns `None` if the child observation cannot be constructed (e.g.
    /// the action is invalid or the state cannot be cloned safely).
    fn child_public_obs_after_discard(
        &mut self,
        state: &GameState,
        obs: &Observation,
        player: u8,
        action: u8,
        safety: &SafetyInfo,
    ) -> Option<[f32; OBS_SIZE]>;
}

/// Concrete [`ExitSearchAdapter`] for self-play that reconstructs child
/// observations by cloning the game state, applying a discard, and
/// re-encoding from the root player's public perspective.
///
/// Hidden-state-contingent opponent actions are NOT rolled through.
/// The observation is taken immediately after the discard resolves,
/// giving the value head the root player's public view of the
/// post-discard state.
pub struct SelfPlayExitAdapter {
    encoder: ObservationEncoder,
    scratch_state: Option<GameState>,
}

impl SelfPlayExitAdapter {
    pub fn new() -> Self {
        Self {
            encoder: ObservationEncoder::new(),
            scratch_state: None,
        }
    }

    pub fn reset(&mut self) {
        self.scratch_state = None;
    }

    pub fn child_public_obs_after_discard_ref(
        &mut self,
        state: &GameState,
        player: u8,
        action: u8,
        safety: &SafetyInfo,
    ) -> Option<[f32; OBS_SIZE]> {
        if action > 33 {
            return None;
        }

        let hand = state.players[player as usize].hand_slice();
        let tile136 = hand.iter().find(|&&t| t / 4 == action)?;
        let riichienv_action = Action::new(ActionType::Discard, Some(*tile136), &[], None);

        let child_state = self.scratch_state.get_or_insert_with(|| state.clone());
        child_state.clone_from(state);
        child_state.skip_mjai_logging = true;

        let mut actions = [None; 4];
        actions[player as usize] = Some(riichienv_action);
        child_state.step_unchecked(&actions);

        let child_obs = child_state.observe(player);
        let encoded = encode_observation_ref(&mut self.encoder, &child_obs, safety);

        Some(encoded)
    }
}

impl Default for SelfPlayExitAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl ExitSearchAdapter for SelfPlayExitAdapter {
    fn root_hash(&self, _state: &GameState, _player: u8, obs_encoded: &[f32; OBS_SIZE]) -> u64 {
        obs_hash(obs_encoded)
    }

    fn child_public_obs_after_discard(
        &mut self,
        state: &GameState,
        _obs: &Observation,
        player: u8,
        action: u8,
        safety: &SafetyInfo,
    ) -> Option<[f32; OBS_SIZE]> {
        self.child_public_obs_after_discard_ref(state, player, action, safety)
    }
}
