use super::*;

fn counts(tiles: &[usize]) -> [u8; TILE_MAX] {
    let mut hand = [0u8; TILE_MAX];
    for &tile in tiles {
        hand[tile] += 1;
    }
    hand
}

#[test]
fn special_hand_shanten_helpers_match_known_shapes() {
    let chiitoi_tenpai = counts(&[0, 0, 1, 1, 9, 9, 10, 10, 18, 18, 27, 27, 31]);
    assert_eq!(calc_chitoi(&chiitoi_tenpai), 0);

    let kokushi_tenpai = counts(&[0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 32]);
    assert_eq!(calc_kokushi(&kokushi_tenpai), 0);
}

#[test]
fn combined_shanten_prefers_special_hands_when_better() {
    let chiitoi_tenpai = counts(&[0, 0, 1, 1, 9, 9, 10, 10, 18, 18, 27, 27, 31]);
    assert_eq!(calc_shanten_from_counts(&chiitoi_tenpai, 4), 0);

    let kokushi_tenpai = counts(&[0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 32]);
    assert_eq!(calc_shanten_from_counts(&kokushi_tenpai, 4), 0);
}

#[test]
fn calculate_shanten_from_136_tiles_matches_count_based_version() {
    let hand_tiles = vec![0, 1, 2, 4, 5, 6, 36, 37, 38, 72, 73, 74, 108];
    let hand_counts = counts(&[0, 0, 0, 1, 1, 1, 9, 9, 9, 18, 18, 18, 27]);

    assert_eq!(
        calculate_shanten(&hand_tiles),
        calc_shanten_from_counts(&hand_counts, 4) as i32
    );
}

#[test]
fn three_player_shanten_ignores_missing_manzu_tiles_for_chiitoi() {
    let hand = counts(&[0, 0, 8, 8, 9, 9, 18, 18, 27, 27, 31, 31, 32]);

    assert_eq!(calc_chitoi_3p(&hand), 0);
    assert_eq!(calc_shanten_from_counts_3p(&hand, 4), 0);
}

#[test]
fn three_player_normal_shanten_handles_manzu_as_honor_like_tiles() {
    let hand = counts(&[0, 0, 0, 8, 8, 9, 10, 11, 18, 19, 20, 27, 27]);
    let shanten = calc_normal_3p(&hand, 4);

    assert!(shanten >= -1);
    assert!(shanten <= 6);
    assert_eq!(
        calc_shanten_from_counts_3p(&hand, 4),
        shanten.min(calc_kokushi(&hand))
    );
}

#[test]
fn effective_tile_helpers_return_nonnegative_counts_for_tenpai_hands() {
    let hand_tiles = vec![0, 1, 2, 4, 5, 6, 36, 37, 38, 72, 73, 74, 108];
    assert!(calculate_effective_tiles(&hand_tiles) > 0);
    assert!(calculate_best_ukeire(&hand_tiles, &[]) <= 4 * 34);

    let sanma_tiles = vec![0, 1, 32, 33, 36, 37, 72, 73, 108, 109, 124, 125, 128];
    assert!(calculate_effective_tiles_3p(&sanma_tiles) > 0);
    assert!(calculate_best_ukeire_3p(&sanma_tiles, &[]) <= 4 * SANMA_VALID_TILE_TYPES.len() as u32);
}

#[test]
fn best_ukeire_saturates_when_visible_and_hand_copies_exhaust_tile_pool() {
    let hand = vec![0, 1, 2, 4, 5, 6, 36, 37, 38, 72, 73, 74, 108];
    let visible = vec![108, 109, 110, 111, 112, 113, 114, 115];

    let ukeire = calculate_best_ukeire(&hand, &visible);

    assert!(ukeire <= 4 * 34);
}

#[test]
fn best_ukeire_3p_saturates_when_visible_and_hand_copies_exhaust_tile_pool() {
    let hand = vec![0, 1, 32, 33, 36, 37, 72, 73, 108, 109, 124, 125, 128];
    let visible = vec![108, 109, 110, 111, 124, 125, 126, 127];

    let ukeire = calculate_best_ukeire_3p(&hand, &visible);

    assert!(ukeire <= 4 * SANMA_VALID_TILE_TYPES.len() as u32);
}
