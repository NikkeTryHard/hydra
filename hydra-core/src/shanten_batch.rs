//! Batch shanten computation with hierarchical hash caching.
//!
//! Computes base shanten + all 34 discard-shanten values in a single pass
//! by caching intermediate suit hashes and only rehashing the affected suit.
//! This avoids redundant rehashing of unchanged suits across discard candidates.

use riichienv_core::shanten::{
    calc_chitoi, calc_kokushi, hash_shupai, hash_zipai, KEYS1, KEYS2, KEYS3, SHUPAI_KEYS,
    ZIPAI_KEYS,
};
use riichienv_core::types::TILE_MAX;

/// Result of batch shanten computation.
#[derive(Debug, Clone)]
pub struct BatchShantenResult {
    /// Base shanten of the full hand.
    pub base: i8,
    /// Shanten after discarding each tile. `None` if tile not in hand.
    pub discard: [Option<i8>; 34],
}

/// Chain the four cached suit keys into a normal-form shanten value.
#[inline]
fn chain_normal(k0_m: usize, k0_p: usize, k0_s: usize, k0_z: usize, m: usize) -> i8 {
    let k1 = KEYS1[k0_m * 126 + k0_p] as usize;
    let k2 = KEYS2[k1 * 126 + k0_s] as usize;
    (KEYS3[(k2 * 55 + k0_z) * 5 + m] as i8) - 1
}

/// Combine normal, chiitoi, and kokushi into final shanten.
#[inline]
fn combined_shanten(normal: i8, tiles: &[u8; TILE_MAX], len_div3: u8) -> i8 {
    let mut sh = normal;
    if sh <= 0 || len_div3 < 4 {
        return sh;
    }
    sh = sh.min(calc_chitoi(tiles));
    if sh > 0 {
        sh.min(calc_kokushi(tiles))
    } else {
        sh
    }
}

/// Compute base shanten and all 34 discard-shanten values efficiently.
///
/// Caches per-suit hashes so each discard only rehashes the affected suit.
/// For a typical 14-tile hand this reduces table lookups from ~530 to ~80.
pub fn batch_discard_shanten(hand: &[u8; TILE_MAX], len_div3: u8) -> BatchShantenResult {
    // 1. Compute base per-suit keys
    let k0_m = SHUPAI_KEYS[hash_shupai(&hand[0..9])] as usize;
    let k0_p = SHUPAI_KEYS[hash_shupai(&hand[9..18])] as usize;
    let k0_s = SHUPAI_KEYS[hash_shupai(&hand[18..27])] as usize;
    let k0_z = ZIPAI_KEYS[hash_zipai(&hand[27..34])] as usize;
    let m_base = len_div3 as usize;

    // 2. Base shanten (normal + chitoi + kokushi)
    let base_normal = chain_normal(k0_m, k0_p, k0_s, k0_z, m_base);
    let base = combined_shanten(base_normal, hand, len_div3);

    // 3. Pre-compute cached chain intermediates for reuse
    let k1_mp = KEYS1[k0_m * 126 + k0_p] as usize;
    let k2_mps = KEYS2[k1_mp * 126 + k0_s] as usize;

    let mut result = BatchShantenResult {
        base,
        discard: [None; 34],
    };

    let total: u8 = hand.iter().sum();
    let m_after = ((total - 1) / 3) as usize;
    let after_len_div3 = (total - 1) / 3;
    let mut tmp = *hand;

    // 4a. Manzu discards (tiles 0..9): rehash manzu, reuse k0_p, k0_s, k0_z
    for t in 0..9 {
        if tmp[t] == 0 {
            continue;
        }
        tmp[t] -= 1;
        let new_k0_m = SHUPAI_KEYS[hash_shupai(&tmp[0..9])] as usize;
        let normal = chain_normal(new_k0_m, k0_p, k0_s, k0_z, m_after);
        result.discard[t] = Some(combined_shanten(normal, &tmp, after_len_div3));
        tmp[t] += 1;
    }

    // 4b. Pinzu discards (tiles 9..18): rehash pinzu, reuse k0_m, k0_s, k0_z
    for t in 9..18 {
        if tmp[t] == 0 {
            continue;
        }
        tmp[t] -= 1;
        let new_k0_p = SHUPAI_KEYS[hash_shupai(&tmp[9..18])] as usize;
        let normal = chain_normal(k0_m, new_k0_p, k0_s, k0_z, m_after);
        result.discard[t] = Some(combined_shanten(normal, &tmp, after_len_div3));
        tmp[t] += 1;
    }

    // 4c. Souzu discards (tiles 18..27): rehash souzu, reuse k1_mp (cached), k0_z
    for t in 18..27 {
        if tmp[t] == 0 {
            continue;
        }
        tmp[t] -= 1;
        let new_k0_s = SHUPAI_KEYS[hash_shupai(&tmp[18..27])] as usize;
        let new_k2 = KEYS2[k1_mp * 126 + new_k0_s] as usize;
        let normal = (KEYS3[(new_k2 * 55 + k0_z) * 5 + m_after] as i8) - 1;
        result.discard[t] = Some(combined_shanten(normal, &tmp, after_len_div3));
        tmp[t] += 1;
    }

    // 4d. Honor discards (tiles 27..34): rehash honors, reuse k2_mps (cached)
    for t in 27..34 {
        if tmp[t] == 0 {
            continue;
        }
        tmp[t] -= 1;
        let new_k0_z = ZIPAI_KEYS[hash_zipai(&tmp[27..34])] as usize;
        let normal = (KEYS3[(k2_mps * 55 + new_k0_z) * 5 + m_after] as i8) - 1;
        result.discard[t] = Some(combined_shanten(normal, &tmp, after_len_div3));
        tmp[t] += 1;
    }

    result
}

#[cfg(test)]
mod tests {
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
        hand[0..3].fill(1);               // 1-3m
        hand[12..15].fill(1);             // 4-6p
        hand[24..27].fill(1);             // 7-9s
        hand[27] = 2; hand[28] = 2;        // 11z 22z
        assert_batch_matches_naive(&hand);
    }

    #[test]
    fn batch_tenpai_hand() {
        // 123m 456m 789m 11p + 2p = tenpai
        let mut hand = [0u8; 34];
        hand[0..9].fill(1);               // 1-9m
        hand[9] = 2;                        // 11p
        hand[10] = 1;                       // 2p
        hand[11] = 1;                       // 3p
        hand[12] = 1;                       // 4p
        assert_batch_matches_naive(&hand);
    }

    #[test]
    fn batch_iishanten_hand() {
        // Scattered hand
        let mut hand = [0u8; 34];
        hand[0] = 2; hand[3] = 1; hand[5] = 1;
        hand[9] = 1; hand[12] = 2;
        hand[18] = 1; hand[21] = 1;
        hand[27] = 2; hand[28] = 1; hand[29] = 1;
        assert_batch_matches_naive(&hand);
    }

    #[test]
    fn batch_kokushi_tenpai() {
        // All 13 terminals/honors, one pair
        let mut hand = [0u8; 34];
        let terminals = [0,8,9,17,18,26,27,28,29,30,31,32,33];
        for &t in &terminals { hand[t] = 1; }
        hand[0] = 2; // pair on 1m
        assert_batch_matches_naive(&hand);
    }

    #[test]
    fn batch_chiitoi_tenpai() {
        // 6 pairs + 1 lone tile
        let mut hand = [0u8; 34];
        hand[0] = 2; hand[3] = 2; hand[9] = 2;
        hand[12] = 2; hand[18] = 2; hand[21] = 2;
        hand[27] = 1;
        assert_batch_matches_naive(&hand);
    }
}
