use crate::types::{is_sanma_excluded_tile, standard_next_dora_tile};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameSubMode3P {
    Single = 0,
    East = 1,
    Half = 2,
}

impl GameSubMode3P {
    pub fn from_game_mode(mode: u8) -> Self {
        match mode {
            3 => GameSubMode3P::Single,
            4 => GameSubMode3P::East,
            5 => GameSubMode3P::Half,
            _ => GameSubMode3P::East,
        }
    }

    pub fn game_mode_id(&self) -> u8 {
        3 + *self as u8
    }
}

/// 3P fixed configuration (no enum dispatch needed).
pub fn num_players() -> u8 {
    3
}

pub fn starting_score() -> i32 {
    35000
}

pub fn tile_set() -> Vec<u8> {
    (0..136u8).filter(|&t| !is_sanma_excluded_tile(t)).collect()
}

pub fn tenpai_pool() -> i32 {
    2000
}

/// Get the next dora tile for a given indicator tile (tile type 0-33).
/// In sanma, manzu wraps 1m(0)->9m(8) and 9m(8)->1m(0) directly.
pub fn get_next_dora_tile(tile: u8) -> u8 {
    if tile < 9 {
        // Manzu suit in sanma: only 0 (1m) and 8 (9m) exist
        if tile == 0 {
            8
        } else if tile == 8 {
            0
        } else {
            tile
        }
    } else {
        standard_next_dora_tile(tile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_mode_mapping_and_ids_follow_sanma_modes() {
        assert_eq!(GameSubMode3P::from_game_mode(3), GameSubMode3P::Single);
        assert_eq!(GameSubMode3P::from_game_mode(4), GameSubMode3P::East);
        assert_eq!(GameSubMode3P::from_game_mode(5), GameSubMode3P::Half);
        assert_eq!(GameSubMode3P::from_game_mode(99), GameSubMode3P::East);

        assert_eq!(GameSubMode3P::Single.game_mode_id(), 3);
        assert_eq!(GameSubMode3P::East.game_mode_id(), 4);
        assert_eq!(GameSubMode3P::Half.game_mode_id(), 5);
    }

    #[test]
    fn sanma_constants_and_tile_set_match_expected_rules() {
        assert_eq!(num_players(), 3);
        assert_eq!(starting_score(), 35000);
        assert_eq!(tenpai_pool(), 2000);

        let tiles = tile_set();
        assert_eq!(tiles.len(), 108);
        assert!(!tiles.iter().any(|&tile| is_sanma_excluded_tile(tile)));
        assert!(tiles.contains(&0));
        assert!(tiles.contains(&135));
    }

    #[test]
    fn sanma_dora_wraps_manzu_edges_and_uses_standard_for_other_tiles() {
        assert_eq!(get_next_dora_tile(0), 8);
        assert_eq!(get_next_dora_tile(8), 0);
        assert_eq!(get_next_dora_tile(4), 4);
        assert_eq!(get_next_dora_tile(9), 10);
        assert_eq!(get_next_dora_tile(30), 27);
    }
}
