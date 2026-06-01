use crate::action::HYDRA_ACTION_SPACE;
use crate::encoder::OBS_SIZE;
use riichienv_core::action::Action;

pub struct ActionDecision<'a> {
    pub player: u8,
    pub seat_id: u8,
    pub obs: &'a [f32; OBS_SIZE],
    pub legal_mask: &'a [bool; HYDRA_ACTION_SPACE],
    pub legal_actions: &'a [Action],
    pub turn: u32,
}

/// Trait for action selection policies.
/// Implemented by random agents, NN inference, etc.
pub trait ActionSelector {
    /// Observe the encoded decision context before `select_action` is called.
    fn observe_decision(&mut self, _decision: ActionDecision<'_>) {}

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
