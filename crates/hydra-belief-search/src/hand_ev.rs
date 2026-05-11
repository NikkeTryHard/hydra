//! Hand-EV oracle features: per-discard tenpai/win probability and ukeire.

use std::collections::HashMap;

use crate::shanten_batch::{BatchDrawShantenResult, batch_discard_shanten, batch_draw_shanten};
use hydra_runtime_types::tile::NUM_TILE_TYPES;

#[derive(Clone)]
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

#[inline]
fn compute_ukeire_from_batch(
    remaining: &[f32; NUM_TILE_TYPES],
    batch: &BatchDrawShantenResult,
) -> [f32; NUM_TILE_TYPES] {
    let mut ukeire = [0.0f32; NUM_TILE_TYPES];
    for t in 0..NUM_TILE_TYPES {
        if remaining[t] <= 0.0 {
            continue;
        }
        if let Some(new_shanten) = batch.draw[t]
            && new_shanten < batch.base
        {
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

#[inline]
fn immediate_win_probability_from_batch(
    remaining: &[f32; NUM_TILE_TYPES],
    batch: &BatchDrawShantenResult,
) -> f32 {
    if batch.base != 0 {
        return 0.0;
    }
    let total_remaining: f32 = remaining.iter().sum();
    if total_remaining <= 0.0 {
        return 0.0;
    }
    let waits: f32 = batch
        .draw
        .iter()
        .enumerate()
        .filter_map(|(tile, shanten)| match shanten {
            Some(value) if *value < 0 && remaining[tile] > 0.0 => Some(remaining[tile]),
            _ => None,
        })
        .sum();
    (waits / total_remaining).clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Default)]
struct FollowUpQuality {
    tenpai_prob: f32,
    win_prob: f32,
}

#[derive(Default)]
struct HandEvLocalCache {
    draw_batches_13: HashMap<[u8; NUM_TILE_TYPES], BatchDrawShantenResult>,
}

impl HandEvLocalCache {
    #[inline]
    fn draw_batch_13(&mut self, hand: &[u8; NUM_TILE_TYPES]) -> BatchDrawShantenResult {
        self.draw_batches_13
            .entry(*hand)
            .or_insert_with(|| batch_draw_shanten(hand, hand.iter().sum::<u8>() / 3))
            .clone()
    }
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
fn best_follow_up_quality_default(
    after_draw: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
    cache: &mut HandEvLocalCache,
) -> FollowUpQuality {
    let total_remaining: f32 = remaining.iter().sum();
    if total_remaining <= 0.0 {
        return FollowUpQuality::default();
    }

    let discard_batch = batch_discard_shanten(after_draw, after_draw.iter().sum::<u8>() / 3);
    let mut best = FollowUpQuality::default();
    for discard in 0..NUM_TILE_TYPES {
        if after_draw[discard] == 0 {
            continue;
        }
        let Some(shanten_after) = discard_batch.discard[discard] else {
            continue;
        };
        let mut after_rediscard = *after_draw;
        after_rediscard[discard] -= 1;
        let draw_batch = cache.draw_batch_13(&after_rediscard);
        let uke = compute_ukeire_from_batch(remaining, &draw_batch);
        let acceptance_ratio = (uke.iter().sum::<f32>() / total_remaining).clamp(0.0, 1.0);
        let tenpai_prob = if shanten_after <= 0 {
            1.0
        } else {
            acceptance_ratio
        };
        let win_prob = if shanten_after < 0 {
            1.0
        } else {
            immediate_win_probability_from_batch(remaining, &draw_batch)
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
fn expected_follow_up_quality_default(
    after_discard: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
    cache: &mut HandEvLocalCache,
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
        let best = best_follow_up_quality_default(&after_draw, &remaining_after_draw, cache);
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

pub fn compute_hand_ev(
    hand: &[u8; NUM_TILE_TYPES],
    remaining: &[f32; NUM_TILE_TYPES],
) -> HandEvFeatures {
    let mut features = HandEvFeatures::default();
    let mut cache = HandEvLocalCache::default();
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
        let shanten_batch =
            batch_draw_shanten(&after_discard, after_discard.iter().sum::<u8>() / 3);
        let uke = compute_ukeire_from_batch(remaining, &shanten_batch);
        features.ukeire[discard] = uke;
        let shanten_after = shanten_batch.base;
        let acceptance: f32 = uke.iter().sum();
        let acceptance_ratio = (acceptance / total_remaining).clamp(0.0, 1.0);
        let follow_up_quality =
            expected_follow_up_quality_default(&after_discard, remaining, &mut cache);
        let immediate_tenpai_draw_prob = if shanten_after <= 0 {
            1.0
        } else {
            acceptance_ratio
        };
        let immediate_win_draw_prob = if shanten_after < 0 {
            1.0
        } else {
            immediate_win_probability_from_batch(remaining, &shanten_batch)
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
            features.tenpai_prob[discard][horizon] =
                (1.0 - tenpai_miss.powi(draws as i32) * (1.0 - tenpai_continue)).clamp(0.0, 1.0);
            features.win_prob[discard][horizon] =
                (1.0 - win_miss.powi(draws as i32) * (1.0 - win_continue)).clamp(0.0, 1.0);
        }
        let score_estimate = conditional_score_estimate(hand, discard, acceptance, shanten_after);
        features.expected_score[discard] = features.win_prob[discard][2] * score_estimate;
    }

    features
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
mod tests;
