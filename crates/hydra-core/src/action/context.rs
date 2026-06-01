/// Tracks two-phase composite actions (riichi tile select, kan tile select).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionPhase {
    /// Normal action selection from full 46-action space.
    Normal,
    /// Selecting which tile to discard for riichi (subset of 0-36).
    RiichiSelect,
    /// Selecting which tile for kan (subset of 0-36).
    KanSelect,
}

/// Context from the game state needed to resolve certain Hydra actions
/// into complete riichienv-core Actions.
#[derive(Debug, Clone)]
pub struct GameContext {
    /// The last discarded tile (136-format) -- needed for chi/pon calls
    pub last_discard: Option<u8>,
    /// Current game phase -- needed to distinguish tsumo vs ron
    pub phase: ActionPhase,
    /// Tiles in the acting player's hand (136-format) -- needed for chi consume_tiles
    pub hand: [u8; 14],
    /// Number of valid tiles currently in `hand`.
    pub hand_len: u8,
}
