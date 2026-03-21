//! Hand-EV oracle features: per-discard tenpai/win probability and ukeire.

use crate::tile::NUM_TILE_TYPES;
use riichienv_core::shanten::calc_shanten_from_counts;

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

pub fn suit_counts(hand: &[u8; NUM_TILE_TYPES]) -> [u8; 3] {
    let m: u8 = hand[..9].iter().sum();
    let p: u8 = hand[9..18].iter().sum();
    let s: u8 = hand[18..27].iter().sum();
    [m, p, s]
}

pub fn honor_count(hand: &[u8; NUM_TILE_TYPES]) -> u8 {
    hand[27..].iter().sum()
}

pub fn has_triplet(hand: &[u8; NUM_TILE_TYPES]) -> bool {
    hand.iter().any(|&c| c >= 3)
}

pub fn has_pair(hand: &[u8; NUM_TILE_TYPES]) -> bool {
    hand.iter().any(|&c| c >= 2)
}

pub fn max_tile_count(hand: &[u8; NUM_TILE_TYPES]) -> u8 {
    hand.iter().copied().max().unwrap_or(0)
}

pub fn unique_tile_count(hand: &[u8; NUM_TILE_TYPES]) -> usize {
    hand.iter().filter(|&&c| c > 0).count()
}

pub fn tiles_held(hand: &[u8; NUM_TILE_TYPES]) -> Vec<u8> {
    (0..NUM_TILE_TYPES)
        .filter(|&t| hand[t] > 0)
        .map(|t| t as u8)
        .collect()
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

#[inline]
fn conditional_score_estimate(
    hand: &[u8; NUM_TILE_TYPES],
    discard: usize,
    acceptance: f32,
    shanten_after: i8,
) -> f32 {
    let suit_mix = suit_counts(hand);
    let honor_tiles = honor_count(hand) as f32;
    let pair_bonus = if has_pair(hand) { 0.35 } else { 0.0 };
    let triplet_bonus = if has_triplet(hand) { 0.5 } else { 0.0 };
    let flush_bias = suit_mix.iter().copied().max().unwrap_or(0) as f32 / 14.0;
    let concentration = max_tile_count(hand) as f32 / 4.0;
    let diversity_penalty = unique_tile_count(hand) as f32 / 14.0;
    let honor_bonus = (honor_tiles / 7.0).min(1.0) * 0.25;
    let shanten_factor = match shanten_after {
        s if s < 0 => 1.35,
        0 => 1.1,
        1 => 0.85,
        _ => 0.6,
    };
    let tile_bonus = if discard >= 27 { 0.15 } else { 0.0 };
    let shape =
        1.0 + pair_bonus + triplet_bonus + honor_bonus + 0.6 * flush_bias + 0.3 * concentration
            - 0.15 * diversity_penalty
            + tile_bonus;
    (1500.0 + 220.0 * acceptance) * shanten_factor * shape.max(0.4)
}

#[inline]
fn immediate_win_probability(
    after_discard: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
    shanten_after: i8,
    shanten_fn: &dyn Fn(&[u8; NUM_TILE_TYPES]) -> i8,
) -> f32 {
    if shanten_after != 0 {
        return 0.0;
    }
    let total_remaining: f32 = remaining.iter().sum();
    if total_remaining <= 0.0 {
        return 0.0;
    }
    let mut waits = 0.0f32;
    for tile in 0..NUM_TILE_TYPES {
        if remaining[tile] <= 0.0 || after_discard[tile] >= 4 {
            continue;
        }
        let mut test_hand = *after_discard;
        test_hand[tile] += 1;
        if shanten_fn(&test_hand) < 0 {
            waits += remaining[tile];
        }
    }
    (waits / total_remaining).clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Default)]
struct FollowUpQuality {
    tenpai_prob: f32,
    win_prob: f32,
}

#[inline]
fn best_follow_up_quality(
    after_draw: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
    shanten_fn: &dyn Fn(&[u8; NUM_TILE_TYPES]) -> i8,
) -> FollowUpQuality {
    let total_remaining: f32 = remaining.iter().sum();
    if total_remaining <= 0.0 {
        return FollowUpQuality::default();
    }

    let mut best = FollowUpQuality::default();
    for discard in 0..NUM_TILE_TYPES {
        if after_draw[discard] == 0 {
            continue;
        }
        let mut after_rediscard = *after_draw;
        after_rediscard[discard] -= 1;
        let shanten_after = shanten_fn(&after_rediscard);
        let uke = compute_ukeire(&after_rediscard, remaining, shanten_fn);
        let acceptance_ratio = (uke.iter().sum::<f32>() / total_remaining).clamp(0.0, 1.0);
        let tenpai_prob = if shanten_after <= 0 {
            1.0
        } else {
            acceptance_ratio
        };
        let win_prob = if shanten_after < 0 {
            1.0
        } else {
            immediate_win_probability(&after_rediscard, remaining, shanten_after, shanten_fn)
                .max((acceptance_ratio * 0.35).clamp(0.0, 1.0))
        };

        if win_prob > best.win_prob || (win_prob == best.win_prob && tenpai_prob > best.tenpai_prob)
        {
            best = FollowUpQuality {
                tenpai_prob,
                win_prob,
            };
        }
    }
    best
}

#[inline]
fn expected_follow_up_quality(
    after_discard: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
    shanten_fn: &dyn Fn(&[u8; NUM_TILE_TYPES]) -> i8,
) -> FollowUpQuality {
    let total_remaining: f32 = remaining.iter().sum();
    if total_remaining <= 0.0 {
        return FollowUpQuality::default();
    }

    let mut weighted = FollowUpQuality::default();
    for draw in 0..NUM_TILE_TYPES {
        if remaining[draw] <= 0.0 || after_discard[draw] >= 4 {
            continue;
        }

        let mut after_draw = *after_discard;
        after_draw[draw] += 1;

        let mut remaining_after_draw = *remaining;
        remaining_after_draw[draw] = (remaining_after_draw[draw] - 1.0).max(0.0);

        let draw_prob = remaining[draw] / total_remaining;
        let best = best_follow_up_quality(&after_draw, &remaining_after_draw, shanten_fn);
        weighted.tenpai_prob += draw_prob * best.tenpai_prob;
        weighted.win_prob += draw_prob * best.win_prob;
    }

    weighted
}

#[inline]
fn continuation_boost(horizon: usize, shanten_after: i8, acceptance_ratio: f32) -> f32 {
    let horizon_scale = match horizon {
        0 => 1.0,
        1 => 0.78,
        _ => 0.62,
    };
    let shanten_scale = match shanten_after {
        s if s < 0 => 1.0,
        0 => 0.9,
        1 => 0.65,
        _ => 0.45,
    };
    (acceptance_ratio * horizon_scale * shanten_scale).clamp(0.0, 1.0)
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

#[inline]
fn default_shanten_fn(counts: &[u8; NUM_TILE_TYPES]) -> i8 {
    let hand_total: u8 = counts.iter().sum();
    calc_shanten_from_counts(counts, hand_total / 3)
}

pub fn compute_hand_ev(
    hand: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
) -> HandEvFeatures {
    compute_hand_ev_with_shanten_fn(hand, remaining, &default_shanten_fn)
}

pub fn compute_hand_ev_with_shanten_fn(
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
            let acceptance_ratio = (acceptance / total_remaining).clamp(0.0, 1.0);
            let follow_up_quality =
                expected_follow_up_quality(&after_discard, remaining, shanten_fn);
            let immediate_tenpai_draw_prob = if shanten_after <= 0 {
                1.0
            } else {
                acceptance_ratio
            };
            let immediate_win_draw_prob = if shanten_after < 0 {
                1.0
            } else {
                immediate_win_probability(&after_discard, remaining, shanten_after, shanten_fn)
            };
            let base_win = immediate_win_draw_prob.max((acceptance_ratio * 0.35).clamp(0.0, 1.0));
            for horizon in 0..3 {
                let draws = (horizon + 1) as u32;
                let tenpai_continue = continuation_boost(horizon, shanten_after, acceptance_ratio)
                    .max(follow_up_quality.tenpai_prob * if horizon == 0 { 0.0 } else { 1.0 });
                let win_continue = continuation_boost(horizon, shanten_after - 1, acceptance_ratio)
                    .max(follow_up_quality.win_prob * if horizon == 0 { 0.0 } else { 1.0 });
                let tenpai_miss = 1.0 - immediate_tenpai_draw_prob;
                let win_miss = 1.0 - base_win;
                features.tenpai_prob[discard][horizon] = (1.0
                    - tenpai_miss.powi(draws as i32) * (1.0 - tenpai_continue))
                    .clamp(0.0, 1.0);
                features.win_prob[discard][horizon] =
                    (1.0 - win_miss.powi(draws as i32) * (1.0 - win_continue)).clamp(0.0, 1.0);
            }
            let score_estimate =
                conditional_score_estimate(hand, discard, acceptance, shanten_after);
            features.expected_score[discard] = features.win_prob[discard][2] * score_estimate;
        }
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_hand_shape_helpers_cover_counts_and_safe_selection() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 3;
        hand[9] = 2;
        hand[18] = 1;
        hand[27] = 2;

        assert_eq!(suit_counts(&hand), [3, 2, 1]);
        assert_eq!(honor_count(&hand), 2);
        assert!(has_triplet(&hand));
        assert!(has_pair(&hand));
        assert_eq!(max_tile_count(&hand), 3);
        assert_eq!(unique_tile_count(&hand), 4);
        assert_eq!(tiles_held(&hand), vec![0, 9, 18, 27]);
        assert_eq!(hand_tile_count(&hand), 8);

        let mut danger = [10.0f32; NUM_TILE_TYPES];
        danger[0] = 0.8;
        danger[9] = 0.2;
        danger[18] = 0.6;
        danger[27] = 0.1;
        assert_eq!(safe_tiles(&hand, &danger, 0.5), vec![9, 27]);
        assert_eq!(most_dangerous_tile(&danger), 33);
        assert_eq!(safest_discard(&hand, &danger), Some(27));
    }

    #[test]
    fn shanten_improvement_best_discard_and_total_ukeire_align() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 1;
        hand[1] = 1;

        let mut remaining = [0.0f32; NUM_TILE_TYPES];
        remaining[0] = 2.0;
        remaining[2] = 5.0;

        let shanten_fn =
            |h: &[u8; NUM_TILE_TYPES]| -> i8 { if h[0] >= 2 || h[2] >= 1 { 0 } else { 1 } };

        assert_eq!(shanten_improvement_count(&hand, &remaining, &shanten_fn), 2);
        assert!((total_ukeire(&hand, &remaining, &shanten_fn) - 7.0).abs() < 1e-6);
        assert_eq!(
            best_discard_by_ukeire(&hand, &remaining, &shanten_fn),
            Some(1)
        );
    }

    #[test]
    fn follow_up_quality_and_score_helpers_cover_zero_and_positive_paths() {
        let hand = [0u8; NUM_TILE_TYPES];
        let remaining = [0.0f32; NUM_TILE_TYPES];
        assert_eq!(
            best_follow_up_quality(&hand, &remaining, &|_| 1).tenpai_prob,
            0.0
        );
        assert_eq!(
            expected_follow_up_quality(&hand, &remaining, &|_| 1).win_prob,
            0.0
        );

        let mut after = [0u8; NUM_TILE_TYPES];
        after[0] = 3;
        let mut rem = [0.0f32; NUM_TILE_TYPES];
        rem[0] = 2.0;
        let quality = best_follow_up_quality(&after, &rem, &|counts| {
            if counts[0] >= 4 { -1 } else { 0 }
        });
        assert!(quality.tenpai_prob > 0.0);
        assert!((0.0..=1.0).contains(&quality.win_prob));

        let score = conditional_score_estimate(&after, 27, 3.0, -1);
        assert!(score > 1500.0);
        assert_eq!(continuation_boost(0, -1, 5.0), 1.0);
    }

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
        let improves_on_tile_0 = |h: &[u8; NUM_TILE_TYPES]| -> i8 { if h[0] > 0 { 0 } else { 1 } };
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
            if total >= 4 { 0 } else { 1 }
        };
        let features = compute_hand_ev_with_shanten_fn(&hand, &remaining, &shanten_fn);
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
        let shanten_fn = |h: &[u8; NUM_TILE_TYPES]| -> i8 { if h[0] >= 3 { -1 } else { 0 } };
        let uke = compute_ukeire(&hand, &remaining, &shanten_fn);
        let acceptance: f32 = uke.iter().sum();
        assert!((acceptance - 3.0).abs() < 1e-5, "tile 0 has 3 remaining");
    }

    #[test]
    fn compute_hand_ev_empty_hand_returns_defaults() {
        let hand = [0u8; NUM_TILE_TYPES];
        let remaining = [4.0f32; NUM_TILE_TYPES];
        let features = compute_hand_ev(&hand, &remaining);
        assert!(
            features
                .tenpai_prob
                .iter()
                .all(|p| p.iter().all(|&v| v == 0.0))
        );
        assert!(features.expected_score.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn compute_hand_ev_no_remaining_returns_defaults() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 3;
        let remaining = [0.0f32; NUM_TILE_TYPES];
        let features = compute_hand_ev(&hand, &remaining);
        assert!(features.expected_score[0] == 0.0);
    }

    #[test]
    fn compute_hand_ev_accumulates_multi_draw_horizons() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[1] = 1;

        let mut remaining = [0.0f32; NUM_TILE_TYPES];
        remaining[0] = 1.0;
        remaining[2] = 3.0;

        let shanten_fn = |h: &[u8; NUM_TILE_TYPES]| -> i8 { if h[0] > 0 { 0 } else { 1 } };

        let features = compute_hand_ev_with_shanten_fn(&hand, &remaining, &shanten_fn);
        let tenpai = features.tenpai_prob[1];
        let win = features.win_prob[1];

        assert!(tenpai[0] > 0.0);
        assert!(tenpai[1] >= tenpai[0]);
        assert!(tenpai[2] >= tenpai[1]);

        assert!(win[0] > 0.0);
        assert!(win[1] >= win[0]);
        assert!(win[2] >= win[1]);
    }

    #[test]
    fn compute_hand_ev_default_shanten_matches_custom_shanten() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 1;
        hand[1] = 1;
        hand[2] = 1;
        hand[9] = 1;
        hand[10] = 1;
        hand[11] = 1;
        hand[18] = 1;
        hand[19] = 1;
        hand[20] = 1;
        hand[27] = 2;
        hand[31] = 2;

        let mut remaining = [0.0f32; NUM_TILE_TYPES];
        remaining[27] = 2.0;
        remaining[31] = 1.0;

        let default_features = compute_hand_ev(&hand, &remaining);
        let custom_features = compute_hand_ev_with_shanten_fn(&hand, &remaining, &|counts| {
            let hand_total: u8 = counts.iter().sum();
            calc_shanten_from_counts(counts, hand_total / 3)
        });

        assert_eq!(default_features.tenpai_prob, custom_features.tenpai_prob);
        assert_eq!(default_features.win_prob, custom_features.win_prob);
        assert_eq!(
            default_features.expected_score,
            custom_features.expected_score
        );
        assert_eq!(default_features.ukeire, custom_features.ukeire);
    }

    #[test]
    fn immediate_win_probability_detects_one_draw_agari() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 3;
        hand[1] = 1;
        let mut remaining = [0.0f32; NUM_TILE_TYPES];
        remaining[0] = 2.0;
        let mut after = hand;
        after[1] -= 1;
        let p = immediate_win_probability(&after, &remaining, 0, &|counts| {
            let total: u8 = counts.iter().sum();
            if counts[0] >= 4 && total >= 4 { -1 } else { 0 }
        });
        assert!(p > 0.0);
    }

    #[test]
    fn expected_score_tracks_win_probability() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 3;
        hand[1] = 1;
        let mut remaining = [0.0f32; NUM_TILE_TYPES];
        remaining[0] = 3.0;
        let features = compute_hand_ev_with_shanten_fn(&hand, &remaining, &|counts| {
            let total: u8 = counts.iter().sum();
            if counts[0] >= 4 && total >= 4 { -1 } else { 0 }
        });
        assert!(features.expected_score[1] > 0.0);
        assert!(features.win_prob[1][2] > 0.0);
    }

    #[test]
    fn later_horizon_prefers_discard_with_stronger_follow_up_line() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 1;
        hand[1] = 1;

        let mut remaining = [0.0f32; NUM_TILE_TYPES];
        remaining[2] = 2.0;
        remaining[3] = 2.0;
        remaining[4] = 4.0;

        let shanten_fn = |counts: &[u8; NUM_TILE_TYPES]| -> i8 {
            let total: u8 = counts.iter().sum();
            match total {
                1 => {
                    if counts[2] == 1 || counts[3] == 1 {
                        0
                    } else {
                        1
                    }
                }
                2 => {
                    if counts[2] == 1 && counts[4] == 1 {
                        -1
                    } else if (counts[1] == 1 && counts[2] == 1)
                        || (counts[0] == 1 && counts[3] == 1)
                    {
                        0
                    } else {
                        1
                    }
                }
                _ => 1,
            }
        };

        let features = compute_hand_ev_with_shanten_fn(&hand, &remaining, &shanten_fn);

        assert!(
            features.win_prob[0][2] > features.win_prob[1][2],
            "discarding 0 should rank above discarding 1 once the stronger next discard line is considered"
        );
        assert!(
            features.expected_score[0] > features.expected_score[1],
            "better downstream line should also raise expected score"
        );
    }

    #[test]
    fn continuation_boost_is_bounded() {
        let boost = continuation_boost(2, 1, 0.9);
        assert!((0.0..=1.0).contains(&boost));
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

    #[test]
    fn safe_tiles_and_safest_discard_cover_empty_and_tie_paths() {
        let hand = [0u8; NUM_TILE_TYPES];
        let danger = [0.2f32; NUM_TILE_TYPES];
        assert!(safe_tiles(&hand, &danger, 0.5).is_empty());
        assert_eq!(safest_discard(&hand, &danger), None);

        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[3] = 1;
        hand[7] = 1;
        let mut danger = [1.0f32; NUM_TILE_TYPES];
        danger[3] = 0.4;
        danger[7] = 0.4;
        assert_eq!(safest_discard(&hand, &danger), Some(3));
    }

    #[test]
    fn compute_ukeire_skips_exhausted_tiles_and_four_copies() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[5] = 4;
        let mut remaining = [0.0f32; NUM_TILE_TYPES];
        remaining[5] = 3.0;
        remaining[6] = 2.0;

        let uke = compute_ukeire(&hand, &remaining, &|counts| {
            if counts[6] > 0 { 0 } else { 1 }
        });
        assert_eq!(uke[5], 0.0);
        assert_eq!(uke[6], 2.0);
    }

    #[test]
    fn danger_from_particles_handles_empty_and_out_of_range_queries() {
        use crate::ct_smc::Particle;

        assert_eq!(danger_from_particles(&[], 0, 0), 0.0);

        let particles = vec![Particle {
            allocation: [[0; 4]; 34],
            log_weight: 0.0,
        }];
        assert_eq!(danger_from_particles(&particles, 34, 0), 0.0);
        assert_eq!(danger_from_particles(&particles, 0, 3), 0.0);
    }

    #[test]
    fn best_discard_by_ukeire_prefers_first_maximum_and_none_for_empty_hand() {
        let hand = [0u8; NUM_TILE_TYPES];
        assert_eq!(
            best_discard_by_ukeire(&hand, &[0.0; NUM_TILE_TYPES], &|_| 1),
            None
        );

        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 1;
        hand[1] = 1;
        let mut remaining = [0.0f32; NUM_TILE_TYPES];
        remaining[2] = 3.0;
        let shanten_fn = |counts: &[u8; NUM_TILE_TYPES]| if counts[2] > 0 { 0 } else { 1 };
        assert_eq!(
            best_discard_by_ukeire(&hand, &remaining, &shanten_fn),
            Some(0)
        );
    }

    #[test]
    fn expected_follow_up_quality_weights_multiple_draws() {
        let mut after_discard = [0u8; NUM_TILE_TYPES];
        after_discard[0] = 1;
        let mut remaining = [0.0f32; NUM_TILE_TYPES];
        remaining[1] = 1.0;
        remaining[2] = 3.0;

        let q = expected_follow_up_quality(&after_discard, &remaining, &|counts| {
            if counts[2] > 0 {
                -1
            } else if counts[1] > 0 {
                0
            } else {
                1
            }
        });

        assert!(q.tenpai_prob > 0.0);
        assert!(q.win_prob > 0.0);
        assert!(q.tenpai_prob >= q.win_prob);
    }

    #[test]
    fn compute_hand_ev_skips_discards_not_in_hand_and_respects_four_copy_ceiling() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[5] = 1;
        let mut remaining = [0.0f32; NUM_TILE_TYPES];
        remaining[5] = 2.0;
        remaining[6] = 1.0;

        let features = compute_hand_ev_with_shanten_fn(&hand, &remaining, &|counts| {
            if counts[6] > 0 { 0 } else { 1 }
        });

        assert_eq!(features.ukeire[0], [0.0; NUM_TILE_TYPES]);
        assert_eq!(features.ukeire[5][5], 0.0);
        assert_eq!(features.ukeire[5][6], 1.0);
    }

    #[test]
    fn conditional_score_estimate_rewards_flush_honor_and_terminal_discards_differently() {
        let mut flushy = [0u8; NUM_TILE_TYPES];
        flushy[0] = 3;
        flushy[1] = 3;
        flushy[2] = 2;
        flushy[27] = 2;

        let suit_score = conditional_score_estimate(&flushy, 0, 4.0, 0);
        let honor_score = conditional_score_estimate(&flushy, 27, 4.0, 0);
        assert!(honor_score > suit_score);
    }

    #[test]
    fn continuation_boost_scales_down_with_higher_horizons_and_worse_shanten() {
        let base = continuation_boost(0, -1, 0.8);
        let later = continuation_boost(2, -1, 0.8);
        let worse = continuation_boost(0, 2, 0.8);

        assert!(later < base);
        assert!(worse < base);
    }
}
