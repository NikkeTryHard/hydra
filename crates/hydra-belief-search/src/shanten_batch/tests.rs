use super::*;
use riichienv_core::shanten::calc_shanten_from_counts;

/// Verify batch matches the naive per-tile approach.
fn assert_batch_matches_naive(hand: &[u8; 34]) {
    let total: u8 = hand.iter().sum();
    let len_div3 = total / 3;
    let batch = batch_discard_shanten(hand, len_div3);
    let naive_base = calc_shanten_from_counts(hand, len_div3);
    assert_eq!(batch.base, naive_base, "base mismatch");

    let after_len_div3 = (total - 1) / 3;
    let mut tmp = *hand;
    for t in 0..34 {
        if hand[t] == 0 {
            assert!(batch.discard[t].is_none(), "tile {t} not in hand");
        } else {
            tmp[t] -= 1;
            let naive = calc_shanten_from_counts(&tmp, after_len_div3);
            assert_eq!(batch.discard[t], Some(naive), "tile {t}");
            tmp[t] += 1;
        }
    }
}

#[test]
fn batch_complete_hand() {
    // 123m 456p 789s 11z + pair 22z = agari
    let mut hand = [0u8; 34];
    hand[0..3].fill(1); // 1-3m
    hand[12..15].fill(1); // 4-6p
    hand[24..27].fill(1); // 7-9s
    hand[27] = 2;
    hand[28] = 2; // 11z 22z
    assert_batch_matches_naive(&hand);
}

#[test]
fn batch_tenpai_hand() {
    // 123m 456m 789m 11p + 2p = tenpai
    let mut hand = [0u8; 34];
    hand[0..9].fill(1); // 1-9m
    hand[9] = 2; // 11p
    hand[10] = 1; // 2p
    hand[11] = 1; // 3p
    hand[12] = 1; // 4p
    assert_batch_matches_naive(&hand);
}

#[test]
fn batch_iishanten_hand() {
    // Scattered hand
    let mut hand = [0u8; 34];
    hand[0] = 2;
    hand[3] = 1;
    hand[5] = 1;
    hand[9] = 1;
    hand[12] = 2;
    hand[18] = 1;
    hand[21] = 1;
    hand[27] = 2;
    hand[28] = 1;
    hand[29] = 1;
    assert_batch_matches_naive(&hand);
}

#[test]
fn batch_kokushi_tenpai() {
    // All 13 terminals/honors, one pair
    let mut hand = [0u8; 34];
    let terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    for &t in &terminals {
        hand[t] = 1;
    }
    hand[0] = 2; // pair on 1m
    assert_batch_matches_naive(&hand);
}

#[test]
fn batch_chiitoi_tenpai() {
    // 6 pairs + 1 lone tile
    let mut hand = [0u8; 34];
    hand[0] = 2;
    hand[3] = 2;
    hand[9] = 2;
    hand[12] = 2;
    hand[18] = 2;
    hand[21] = 2;
    hand[27] = 1;
    assert_batch_matches_naive(&hand);
}

#[test]
fn batch_empty_hand_returns_no_discards() {
    let hand = [0u8; 34];
    let batch = batch_discard_shanten(&hand, 0);
    assert_eq!(batch.base, calc_shanten_from_counts(&hand, 0));
    assert!(batch.discard.iter().all(Option::is_none));
}

#[test]
fn batch_single_tile_hand_matches_naive() {
    let mut hand = [0u8; 34];
    hand[5] = 1;
    assert_batch_matches_naive(&hand);
}

#[test]
fn batch_four_of_a_kind_matches_naive() {
    let mut hand = [0u8; 34];
    hand[0] = 4;
    assert_batch_matches_naive(&hand);
}

fn assert_draw_batch_matches_naive(hand: &[u8; 34]) {
    let total: u8 = hand.iter().sum();
    let len_div3 = total / 3;
    let batch = batch_draw_shanten(hand, len_div3);
    let naive_base = calc_shanten_from_counts(hand, len_div3);
    assert_eq!(batch.base, naive_base, "base mismatch");

    let after_len_div3 = (total + 1) / 3;
    let mut tmp = *hand;
    for t in 0..34 {
        if hand[t] >= 4 {
            assert!(
                batch.draw[t].is_none(),
                "tile {t} already exhausted in hand"
            );
        } else {
            tmp[t] += 1;
            let naive = calc_shanten_from_counts(&tmp, after_len_div3);
            assert_eq!(batch.draw[t], Some(naive), "draw tile {t}");
            tmp[t] -= 1;
        }
    }
}

#[test]
fn batch_draw_complete_hand_matches_naive() {
    let mut hand = [0u8; 34];
    hand[0..3].fill(1);
    hand[12..15].fill(1);
    hand[24..27].fill(1);
    hand[27] = 2;
    hand[28] = 2;
    assert_draw_batch_matches_naive(&hand);
}

#[test]
fn batch_draw_single_tile_matches_naive() {
    let mut hand = [0u8; 34];
    hand[5] = 1;
    assert_draw_batch_matches_naive(&hand);
}

#[test]
fn batch_draw_four_of_a_kind_skips_full_tile() {
    let mut hand = [0u8; 34];
    hand[0] = 4;
    let batch = batch_draw_shanten(&hand, 1);
    assert!(batch.draw[0].is_none());
    assert_draw_batch_matches_naive(&hand);
}
