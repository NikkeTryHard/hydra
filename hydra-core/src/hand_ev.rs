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

pub fn hand_tile_count(hand: &[u8; NUM_TILE_TYPES]) -> u8 {
    hand.iter().sum()
}

pub fn safe_tiles(
    hand: &[u8; NUM_TILE_TYPES],
    danger_scores: &[f32; NUM_TILE_TYPES],
    threshold: f32,
) -> Vec<u8> {
    (0..NUM_TILE_TYPES)
        .filter(|&t| hand[t] > 0 && danger_scores[t] < threshold)
        .map(|t| t as u8)
        .collect()
}

pub fn most_dangerous_tile(danger_scores: &[f32; NUM_TILE_TYPES]) -> u8 {
    danger_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u8)
        .unwrap_or(0)
}

pub fn safest_discard(
    hand: &[u8; NUM_TILE_TYPES],
    danger_scores: &[f32; NUM_TILE_TYPES],
) -> Option<u8> {
    let mut safest = None;
    let mut min_danger = f32::INFINITY;
    for t in 0..NUM_TILE_TYPES {
        if hand[t] == 0 {
            continue;
        }
        if danger_scores[t] < min_danger {
            min_danger = danger_scores[t];
            safest = Some(t as u8);
        }
    }
    safest
}

pub fn shanten_improvement_count(
    hand: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
    shanten_fn: &dyn Fn(&[u8; NUM_TILE_TYPES]) -> i8,
) -> usize {
    compute_ukeire(hand, remaining, shanten_fn)
        .iter()
        .filter(|&&v| v > 0.0)
        .count()
}

pub fn danger_from_particles(particles: &[crate::ct_smc::Particle], tile: u8, opponent: u8) -> f32 {
    if particles.is_empty() || tile >= 34 || opponent >= 3 {
        return 0.0;
    }
    let count: usize = particles
        .iter()
        .filter(|p| p.allocation[tile as usize][opponent as usize] > 0)
        .count();
    count as f32 / particles.len() as f32
}

pub fn total_ukeire(
    hand: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
    shanten_fn: &dyn Fn(&[u8; NUM_TILE_TYPES]) -> i8,
) -> f32 {
    compute_ukeire(hand, remaining, shanten_fn).iter().sum()
}

pub fn best_discard_by_ukeire(
    hand: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
    shanten_fn: &dyn Fn(&[u8; NUM_TILE_TYPES]) -> i8,
) -> Option<u8> {
    let mut best = None;
    let mut best_acc = -1.0f32;
    for t in 0..NUM_TILE_TYPES {
        if hand[t] == 0 {
            continue;
        }
        let mut after = *hand;
        after[t] -= 1;
        let uke = compute_ukeire(&after, remaining, shanten_fn);
        let acc: f32 = uke.iter().sum();
        if acc > best_acc {
            best_acc = acc;
            best = Some(t as u8);
        }
    }
    best
}

pub fn compute_hand_ev(
    hand: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
    shanten_fn: &dyn Fn(&[u8; NUM_TILE_TYPES]) -> i8,
) -> HandEvFeatures {
    let mut features = HandEvFeatures::default();
    let total_remaining: f32 = remaining.iter().sum();
    if total_remaining <= 0.0 {
        return features;
    }

    for discard in 0..NUM_TILE_TYPES {
        if hand[discard] == 0 {
            continue;
        }
        let mut after_discard = *hand;
        after_discard[discard] -= 1;
        let uke = compute_ukeire(&after_discard, remaining, shanten_fn);
        features.ukeire[discard] = uke;
        let shanten_after = shanten_fn(&after_discard);
        let acceptance: f32 = uke.iter().sum();
        if total_remaining > 0.0 {
            let p_draw = acceptance / total_remaining;
            features.tenpai_prob[discard][0] = if shanten_after <= 0 { 1.0 } else { p_draw };
            features.tenpai_prob[discard][1] = features.tenpai_prob[discard][0].min(1.0);
            features.tenpai_prob[discard][2] = features.tenpai_prob[discard][1].min(1.0);
            features.win_prob[discard][0] = if shanten_after < 0 { 1.0 } else { p_draw * 0.5 };
            features.win_prob[discard][1] = features.win_prob[discard][0].min(1.0);
            features.win_prob[discard][2] = features.win_prob[discard][1].min(1.0);
        }
        features.expected_score[discard] = acceptance * 1000.0;
    }
    features
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

    #[test]
    fn tenpai_hand_has_high_p_tenpai() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 3;
        hand[1] = 1;
        let remaining = [4.0f32; NUM_TILE_TYPES];
        let shanten_fn = |h: &[u8; NUM_TILE_TYPES]| -> i8 {
            let total: u8 = h.iter().sum();
            if total >= 4 {
                0
            } else {
                1
            }
        };
        let features = compute_hand_ev(&hand, &remaining, &shanten_fn);
        assert!(
            features.tenpai_prob[1][0] > 0.0,
            "discarding tile 1 should have positive tenpai prob"
        );
    }

    #[test]
    fn ukeire_sums_match_acceptance() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 2;
        hand[1] = 1;
        let remaining = [3.0f32; NUM_TILE_TYPES];
        let shanten_fn = |h: &[u8; NUM_TILE_TYPES]| -> i8 {
            if h[0] >= 3 {
                -1
            } else {
                0
            }
        };
        let uke = compute_ukeire(&hand, &remaining, &shanten_fn);
        let acceptance: f32 = uke.iter().sum();
        assert!((acceptance - 3.0).abs() < 1e-5, "tile 0 has 3 remaining");
    }

    #[test]
    fn compute_hand_ev_empty_hand_returns_defaults() {
        let hand = [0u8; NUM_TILE_TYPES];
        let remaining = [4.0f32; NUM_TILE_TYPES];
        let shanten_fn = |_: &[u8; NUM_TILE_TYPES]| -> i8 { 8 };
        let features = compute_hand_ev(&hand, &remaining, &shanten_fn);
        assert!(features
            .tenpai_prob
            .iter()
            .all(|p| p.iter().all(|&v| v == 0.0)));
        assert!(features.expected_score.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn compute_hand_ev_no_remaining_returns_defaults() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 3;
        let remaining = [0.0f32; NUM_TILE_TYPES];
        let shanten_fn = |_: &[u8; NUM_TILE_TYPES]| -> i8 { 0 };
        let features = compute_hand_ev(&hand, &remaining, &shanten_fn);
        assert!(features.expected_score[0] == 0.0);
    }

    #[test]
    fn danger_from_particles_basic() {
        use crate::ct_smc::Particle;
        let mut p1 = Particle {
            allocation: [[0; 4]; 34],
            log_weight: 0.0,
        };
        p1.allocation[5][0] = 2;
        let mut p2 = Particle {
            allocation: [[0; 4]; 34],
            log_weight: 0.0,
        };
        p2.allocation[5][1] = 1;
        let particles = vec![p1, p2];
        let d = danger_from_particles(&particles, 5, 0);
        assert!(
            (d - 0.5).abs() < 1e-5,
            "1/2 particles have tile 5 for opp 0"
        );
    }
}
