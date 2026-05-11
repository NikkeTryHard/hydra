use crate::rule::GameRule;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameSubMode {
    Single = 0,
    East = 1,
    Half = 2,
}

/// 4P-only game mode configuration.
/// 3P games use `state_3p::GameState3P` instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameModeConfig {
    pub sub_mode: GameSubMode,
    pub rule: GameRule,
}

impl GameModeConfig {
    pub fn from_game_mode(mode: u8, rule: GameRule) -> Self {
        let sub_mode = match mode {
            0 => GameSubMode::Single,
            1 => GameSubMode::East,
            2 => GameSubMode::Half,
            _ => GameSubMode::East,
        };
        GameModeConfig { sub_mode, rule }
    }

    pub fn num_players(&self) -> u8 {
        4
    }

    pub fn starting_score(&self) -> i32 {
        25000
    }

    pub fn tenpai_pool(&self) -> i32 {
        3000
    }

    pub fn get_next_dora_tile(&self, tile: u8) -> u8 {
        standard_next_dora_tile(tile)
    }

    pub fn rule(&self) -> &GameRule {
        &self.rule
    }

    pub fn game_mode_id(&self) -> u8 {
        self.sub_mode as u8
    }

    pub fn sub_mode(&self) -> &GameSubMode {
        &self.sub_mode
    }
}

// Re-export shared utilities from types module
pub use crate::types::{is_sanma_excluded_tile, standard_next_dora_tile};

#[cfg(test)]
mod tests;
