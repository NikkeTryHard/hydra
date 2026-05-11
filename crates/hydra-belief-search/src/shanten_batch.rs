//! Batch shanten computation with hierarchical hash caching.
//!
//! Computes base shanten + all 34 discard-shanten values in a single pass
//! by caching intermediate suit hashes and only rehashing the affected suit.
//! This avoids redundant rehashing of unchanged suits across discard candidates.

use riichienv_core::shanten::{
    KEYS1, KEYS2, KEYS3, SHUPAI_KEYS, ZIPAI_KEYS, calc_chitoi, calc_kokushi, hash_shupai,
    hash_zipai,
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

#[derive(Debug, Clone)]
pub struct BatchDrawShantenResult {
    pub base: i8,
    pub draw: [Option<i8>; 34],
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
    if total == 0 {
        return result;
    }
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

pub fn batch_draw_shanten(hand: &[u8; TILE_MAX], len_div3: u8) -> BatchDrawShantenResult {
    let k0_m = SHUPAI_KEYS[hash_shupai(&hand[0..9])] as usize;
    let k0_p = SHUPAI_KEYS[hash_shupai(&hand[9..18])] as usize;
    let k0_s = SHUPAI_KEYS[hash_shupai(&hand[18..27])] as usize;
    let k0_z = ZIPAI_KEYS[hash_zipai(&hand[27..34])] as usize;
    let m_base = len_div3 as usize;

    let base_normal = chain_normal(k0_m, k0_p, k0_s, k0_z, m_base);
    let base = combined_shanten(base_normal, hand, len_div3);

    let k1_mp = KEYS1[k0_m * 126 + k0_p] as usize;
    let k2_mps = KEYS2[k1_mp * 126 + k0_s] as usize;

    let mut result = BatchDrawShantenResult {
        base,
        draw: [None; 34],
    };

    let total: u8 = hand.iter().sum();
    let m_after = ((total + 1) / 3) as usize;
    let after_len_div3 = (total + 1) / 3;
    let mut tmp = *hand;

    for t in 0..9 {
        if tmp[t] >= 4 {
            continue;
        }
        tmp[t] += 1;
        let new_k0_m = SHUPAI_KEYS[hash_shupai(&tmp[0..9])] as usize;
        let normal = chain_normal(new_k0_m, k0_p, k0_s, k0_z, m_after);
        result.draw[t] = Some(combined_shanten(normal, &tmp, after_len_div3));
        tmp[t] -= 1;
    }

    for t in 9..18 {
        if tmp[t] >= 4 {
            continue;
        }
        tmp[t] += 1;
        let new_k0_p = SHUPAI_KEYS[hash_shupai(&tmp[9..18])] as usize;
        let normal = chain_normal(k0_m, new_k0_p, k0_s, k0_z, m_after);
        result.draw[t] = Some(combined_shanten(normal, &tmp, after_len_div3));
        tmp[t] -= 1;
    }

    for t in 18..27 {
        if tmp[t] >= 4 {
            continue;
        }
        tmp[t] += 1;
        let new_k0_s = SHUPAI_KEYS[hash_shupai(&tmp[18..27])] as usize;
        let new_k2 = KEYS2[k1_mp * 126 + new_k0_s] as usize;
        let normal = (KEYS3[(new_k2 * 55 + k0_z) * 5 + m_after] as i8) - 1;
        result.draw[t] = Some(combined_shanten(normal, &tmp, after_len_div3));
        tmp[t] -= 1;
    }

    for t in 27..34 {
        if tmp[t] >= 4 {
            continue;
        }
        tmp[t] += 1;
        let new_k0_z = ZIPAI_KEYS[hash_zipai(&tmp[27..34])] as usize;
        let normal = (KEYS3[(k2_mps * 55 + new_k0_z) * 5 + m_after] as i8) - 1;
        result.draw[t] = Some(combined_shanten(normal, &tmp, after_len_div3));
        tmp[t] -= 1;
    }

    result
}

#[cfg(test)]
mod tests;
