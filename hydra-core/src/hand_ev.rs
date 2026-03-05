//! Hand-EV oracle features: per-discard tenpai/win probability and ukeire.

use crate::tile::NUM_TILE_TYPES;

pub struct HandEvFeatures {
    pub tenpai_prob: [[f32; 3]; NUM_TILE_TYPES],
    pub win_prob: [[f32; 3]; NUM_TILE_TYPES],
    pub expected_score: [f32; NUM_TILE_TYPES],
    pub ukeire: [[f32; NUM_TILE_TYPES]; NUM_TILE_TYPES],
}

impl Default for HandEvFeatures {
    fn default() -> Self {
        Self {
            tenpai_prob: [[0.0; 3]; NUM_TILE_TYPES],
            win_prob: [[0.0; 3]; NUM_TILE_TYPES],
            expected_score: [0.0; NUM_TILE_TYPES],
            ukeire: [[0.0; NUM_TILE_TYPES]; NUM_TILE_TYPES],
        }
    }
}

pub fn compute_ukeire(
    hand: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
    shanten_fn: &dyn Fn(&[u8; NUM_TILE_TYPES]) -> i8,
) -> [f32; NUM_TILE_TYPES] {
    let base_shanten = shanten_fn(hand);
    let mut ukeire = [0.0f32; NUM_TILE_TYPES];
    for t in 0..NUM_TILE_TYPES {
        if remaining[t] <= 0.0 {
            continue;
        }
        let mut test_hand = *hand;
        test_hand[t] += 1;
        let new_shanten = shanten_fn(&test_hand);
        if new_shanten < base_shanten {
            ukeire[t] = remaining[t];
        }
    }
    ukeire
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ukeire_zero_when_no_improvement() {
        let hand = [0u8; NUM_TILE_TYPES];
        let remaining = [4.0f32; NUM_TILE_TYPES];
        let always_same = |_: &[u8; NUM_TILE_TYPES]| -> i8 { 6 };
        let uke = compute_ukeire(&hand, &remaining, &always_same);
        assert!(uke.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn ukeire_counts_improving_tiles() {
        let hand = [0u8; NUM_TILE_TYPES];
        let remaining = [4.0f32; NUM_TILE_TYPES];
        let improves_on_tile_0 = |h: &[u8; NUM_TILE_TYPES]| -> i8 {
            if h[0] > 0 {
                0
            } else {
                1
            }
        };
        let uke = compute_ukeire(&hand, &remaining, &improves_on_tile_0);
        assert!((uke[0] - 4.0).abs() < 1e-5);
        assert!(uke[1..].iter().all(|&v| v == 0.0));
    }
}
