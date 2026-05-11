use crate::state_3p::game_mode::{
    get_next_dora_tile, num_players, starting_score, tenpai_pool, tile_set, GameSubMode3P,
};
use crate::types::is_sanma_excluded_tile;

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
