use super::*;
use crate::rule::GameRule;

#[test]
fn test_game_mode_config_four_player() {
    let mode = GameModeConfig::from_game_mode(2, GameRule::default_tenhou());
    assert_eq!(mode.num_players(), 4);
    assert_eq!(mode.starting_score(), 25000);
    assert_eq!(mode.tenpai_pool(), 3000);
    assert_eq!(mode.game_mode_id(), 2);
}

#[test]
fn test_sanma_excluded_tiles() {
    assert!(!is_sanma_excluded_tile(0));
    assert!(!is_sanma_excluded_tile(3));
    assert!(is_sanma_excluded_tile(4));
    assert!(is_sanma_excluded_tile(7));
    assert!(is_sanma_excluded_tile(28));
    assert!(is_sanma_excluded_tile(31));
    assert!(!is_sanma_excluded_tile(32));
    assert!(!is_sanma_excluded_tile(35));
    assert!(!is_sanma_excluded_tile(36));
    assert!(!is_sanma_excluded_tile(72));
    assert!(!is_sanma_excluded_tile(108));
    assert!(!is_sanma_excluded_tile(135));
}

#[test]
fn test_four_player_dora_wrapping() {
    let mode = GameModeConfig::from_game_mode(0, GameRule::default_tenhou());
    assert_eq!(mode.get_next_dora_tile(0), 1); // 1m -> 2m
    assert_eq!(mode.get_next_dora_tile(8), 0); // 9m -> 1m
    assert_eq!(mode.get_next_dora_tile(27), 28); // E -> S
    assert_eq!(mode.get_next_dora_tile(30), 27); // N -> E
    assert_eq!(mode.get_next_dora_tile(31), 32); // Haku -> Hatsu
    assert_eq!(mode.get_next_dora_tile(33), 31); // Chun -> Haku
}
