<combined_run_record run_id="answer_17" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 17 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_17_IMPLEMENT_HAND_EV_SEMANTIC_REPAIR.md">
  <![CDATA[# Prompt 17 — Hand-EV semantic repair blueprint

<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest exact blueprint for repairing the semantics of Hand-EV.

We want a detailed answer that makes clear:
- what the current quantities really mean
- what is semantically broken or misleading
- what the clean repaired meanings should be
- what should stay exact, what should stay approximate, and what should be dropped or demoted
- how to implement the repair with minimal guesswork

Use the artifacts below to derive your conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- include reasoning
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)
- distinguish direct artifact support from your own inference
- use search/browse to find the original paper, then inspect the full PDF with skill; use abstracts or summaries only for discovery, not as the final evidence base
- use the bash tool to run Python for calculations, math checks, and validation when rigor matters
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>
Embedded artifact/context dump intentionally omitted during normalization. Use the paired answer body plus the repository source files for audit and follow-up.
</artifacts>

]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_17.md">
  <![CDATA[# Hand-EV semantic repair blueprint

```rust
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
                let tenpai_continue = continuation_boost(horizon, shanten_after, acceptance_ratio);
                let win_continue = continuation_boost(horizon, shanten_after - 1, acceptance_ratio);
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
        let features = compute_hand_ev(&hand, &remaining);
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

        let shanten_fn = |h: &[u8; NUM_TILE_TYPES]| -> i8 {
            if h[0] > 0 {
                0
            } else {
                1
            }
        };

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
            if counts[0] >= 4 && total >= 4 {
                -1
            } else {
                0
            }
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
            if counts[0] >= 4 && total >= 4 {
                -1
            } else {
                0
            }
        });
        assert!(features.expected_score[1] > 0.0);
        assert!(features.win_prob[1][2] > 0.0);
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
}
```

Artifact B — bridge integration excerpt:

```rust
/// Extract game metadata from an Observation.
///
/// Computes shanten from the observer's hand counts. All player-relative
/// fields (riichi, scores) are rotated so index 0 = observer,
/// index 1 = shimocha, etc.
#[inline]
pub fn extract_metadata(obs: &Observation, hand_counts: &[u8; NUM_TILE_TYPES]) -> GameMetadata {
    let observer = obs.player_id as usize;

    // Compute shanten: len_div3 is based on the closed hand tile count.
    // A 13-tile hand has len_div3=4, a 14-tile hand also has len_div3=4.
    let hand_total: u8 = hand_counts.iter().sum();
    let len_div3 = hand_total / 3;
    let shanten = calc_shanten_from_counts(hand_counts, len_div3);

    GameMetadata {
        riichi: std::array::from_fn(|i| obs.riichi_declared[(observer + i) % 4]),
        scores: std::array::from_fn(|i| obs.scores[(observer + i) % 4]),
        shanten,
        kyoku_index: obs.kyoku_index,
        honba: obs.honba,
        kyotaku: obs.riichi_sticks.min(255) as u8,
    }
}

/// Compute public-state remaining tile counts for the observer.
///
/// This subtracts all tiles visible to the observer: their concealed hand,
/// all open melds, all discards, and visible dora indicators. This is a safe
/// bridge-side approximation for Hand-EV features until belief-weighted
/// remaining counts from CT-SMC are threaded into the encoder path.
#[inline]
pub fn extract_public_remaining_counts(
    hand: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
) -> [f32; NUM_TILE_TYPES] {
    let mut remaining = [4.0f32; NUM_TILE_TYPES];

    for (tile, &count) in hand.iter().enumerate() {
        remaining[tile] -= count as f32;
    }
    for player_discards in discards {
        for entry in player_discards
            .discards
            .iter()
            .take(player_discards.len as usize)
        {
            remaining[entry.tile as usize] -= 1.0;
        }
    }
    for player_melds in melds {
        for meld in player_melds.melds.iter().take(player_melds.len as usize) {
            for &tile in meld.tiles.iter().take(meld.tile_count as usize) {
                remaining[tile as usize] -= 1.0;
            }
        }
    }
    for &indicator in dora.indicators.iter().take(dora.indicator_count as usize) {
        remaining[indicator as usize] -= 1.0;
    }

    for value in &mut remaining {
        *value = value.max(0.0);
    }
    remaining
}

/// Compute bridge-side Hand-EV features from public-state remaining counts.
#[inline]
pub fn compute_public_hand_ev(
    hand: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
) -> HandEvFeatures {
    let remaining = extract_public_remaining_counts(hand, discards, melds, dora);
    compute_hand_ev(hand, &remaining)
}

/// Compute belief-weighted remaining tile counts from a CT-SMC posterior.
#[inline]
pub fn extract_ct_smc_remaining_counts(ct_smc: &CtSmc) -> [f32; NUM_TILE_TYPES] {
    let mut remaining = [0.0f32; NUM_TILE_TYPES];
    if ct_smc.is_empty() {
        return remaining;
    }
    for (tile, slot) in remaining.iter_mut().enumerate() {
        *slot = (0..4)
            .map(|col| ct_smc.weighted_mean_tile_count(tile as u8, col as u8))
            .sum();
    }
    remaining
}

/// Compute bridge-side Hand-EV features from CT-SMC belief-weighted counts.
#[inline]
pub fn compute_ct_smc_hand_ev(hand: &[u8; NUM_TILE_TYPES], ct_smc: &CtSmc) -> HandEvFeatures {
    let remaining = extract_ct_smc_remaining_counts(ct_smc);
    compute_hand_ev(hand, &remaining)
}

#[inline]
fn compute_hand_ev_from_context(
    hand: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
    search_context: &SearchContext<'_>,
) -> HandEvFeatures {
    if let Some(ct_smc) = search_context.ct_smc
        && !ct_smc.is_empty()
    {
        return compute_ct_smc_hand_ev(hand, ct_smc);
    }
    compute_public_hand_ev(hand, discards, melds, dora)
}

/// Encode a full observation into the fixed-superset tensor with optional Group C runtime context.
#[inline]
pub fn encode_observation_with_search_context(
    encoder: &mut ObservationEncoder,
    obs: &Observation,
    safety: &SafetyInfo,
    drawn_tile: Option<u8>,
    search_context: &SearchContext<'_>,
) -> [f32; OBS_SIZE] {
    let hand = extract_hand(obs);
    let discards = extract_discards(obs);
    let melds = extract_melds(obs);
    let open_meld_counts = extract_observer_meld_counts(obs);
    let dora = extract_dora(obs);
    let meta = extract_metadata(obs, &hand);
    let hand_ev = compute_hand_ev_from_context(&hand, &discards, &melds, &dora, search_context);
    let search_features = build_search_features(safety, search_context);

    let slice = encoder.encode_with_context(
        &hand,
        drawn_tile,
        &open_meld_counts,
        &discards,
        &melds,
        &dora,
        &meta,
        safety,
        Some(&search_features),
        Some(&hand_ev),
    );
    *slice
}
```

Artifact C — architecture/doctrine excerpts:

```text
Hand-EV oracle features (~34-68 planes, CPU-precomputed): For each discard candidate a (34 tile types), pre-compute exact look-ahead analysis:
- P_tenpai^(d)(a): probability of reaching tenpai within d in {1,2,3} self-draws.
- P_win^(d)(a): probability of winning within d draws.
- E[score | win, a]: expected hand value if we win after discarding a.
- Ukeire vector: 34-element effective tile acceptance weighted by remaining counts.
```

```text
Main consensus:
- AFBS should be selective and specialist, not the default path everywhere.
- Hand-EV is worth moving earlier than deeper AFBS expansion.
```

```text
Recommendation: Rework Hand-EV realism before deeper AFBS expansion.
Repo verification status:
- hand_ev exists, but expected score and win modeling remain heuristic
- bridge already threads Hand-EV into encoder paths
```

Artifact D — critique targets:

```text
- expected_score is currently assigned as win_prob[horizon=2] * score_estimate
- immediate_win_probability only behaves as a clean one-step quantity in shanten_after == 0 states
- continuation_boost is openly heuristic
- public remaining counts and CT-SMC weighted counts both exist as upstream count sources
- the module-level comment says “oracle features,” but internal semantics mix exact and heuristic pieces
```

Artifact E — encoder-side Hand-EV channel sink:

```rust
/// Encode fixed-shape Group D Hand-EV context planes.
#[inline]
pub fn encode_hand_ev_features(&mut self, hand_ev: &HandEvFeatures) {
    self.clear_range(CH_HAND_EV, CH_HAND_EV + HAND_EV_CHANNELS);

    for discard in 0..NUM_TILES {
        for horizon in 0..3 {
            self.set(
                CH_HAND_EV_TENPAI + horizon,
                discard,
                hand_ev.tenpai_prob[discard][horizon],
            );
            self.set(
                CH_HAND_EV_WIN + horizon,
                discard,
                hand_ev.win_prob[discard][horizon],
            );
        }
        self.set(CH_HAND_EV_SCORE, discard, hand_ev.expected_score[discard]);
    }
    for draw_tile in 0..NUM_TILES {
        for discard in 0..NUM_TILES {
            self.set(
                CH_HAND_EV_UKEIRE + draw_tile,
                discard,
                hand_ev.ukeire[discard][draw_tile],
            );
        }
    }
    self.fill_channel(CH_HAND_EV_MASK, 1.0);
}

/// Encode a complete observation plus optional Group C / Group D context.
#[allow(clippy::too_many_arguments)]
pub fn encode_with_context(
    &mut self,
    hand: &[u8; NUM_TILES],
    drawn_tile: Option<u8>,
    open_meld_counts: &[u8; NUM_TILES],
    discards: &[PlayerDiscards; NUM_PLAYERS],
    melds: &[PlayerMelds; NUM_PLAYERS],
    dora: &DoraInfo,
    meta: &GameMetadata,
    safety: &SafetyInfo,
    search_features: Option<&SearchFeaturePlanes>,
    hand_ev: Option<&HandEvFeatures>,
) -> &[f32; OBS_SIZE] {
    self.encode(
        hand,
        drawn_tile,
        open_meld_counts,
        discards,
        melds,
        dora,
        meta,
        safety,
    );
    if let Some(features) = search_features {
        self.encode_search_features(features);
    }
    if let Some(features) = hand_ev {
        self.encode_hand_ev_features(features);
    }
    self.as_slice()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct DirtyFlags(pub u16);

impl DirtyFlags {
    pub const HAND: Self = Self(1 << 0);
    pub const OPEN_MELD: Self = Self(1 << 1);
    pub const DRAWN: Self = Self(1 << 2);
    pub const SHANTEN: Self = Self(1 << 3);
    pub const DISCARDS: Self = Self(1 << 4);
    pub const MELDS: Self = Self(1 << 5);
    pub const DORA: Self = Self(1 << 6);
    pub const META: Self = Self(1 << 7);
    pub const SAFETY: Self = Self(1 << 8);
    pub const SEARCH: Self = Self(1 << 9);
    pub const HAND_EV: Self = Self(1 << 10);
    pub const ALL: Self = Self(0x7FF);

    pub const AFTER_DRAW: Self = Self(
        Self::HAND.0
            | Self::DRAWN.0
            | Self::SHANTEN.0
            | Self::META.0
            | Self::SEARCH.0
            | Self::HAND_EV.0,
    );
    pub const AFTER_DISCARD: Self = Self(
        Self::HAND.0
            | Self::DRAWN.0
            | Self::SHANTEN.0
            | Self::DISCARDS.0
            | Self::META.0
            | Self::SAFETY.0
            | Self::SEARCH.0
            | Self::HAND_EV.0,
    );
    pub const AFTER_CALL: Self = Self(
        Self::HAND.0
            | Self::OPEN_MELD.0
            | Self::SHANTEN.0
            | Self::DISCARDS.0
            | Self::MELDS.0
            | Self::META.0
            | Self::SAFETY.0
            | Self::SEARCH.0
            | Self::HAND_EV.0,
    );
    pub const NEW_ROUND: Self = Self::ALL;
}
```

Artifact F — engine documentation excerpts:

```text
Canonical SSOT note: the original 85 x 34 tensor now describes the baseline prefix of the live encoder, not the full live encoder. The current implementation is already a fixed-shape 192 x 34 superset with Groups C/D plus presence-mask channels.
```

```text
Each observation is a 192 x 34 float tensor (6,528 values). The first 85 channels retain the baseline public+safety encoding; the remaining channels provide fixed-shape search/belief and Hand-EV context with zero-fill plus explicit presence masks when dynamic features are unavailable.
```

```text
Hand-EV oracle features (~34-68 planes, CPU-precomputed):
- P_tenpai^(d)(a): probability of reaching tenpai within d in {1,2,3} self-draws.
- P_win^(d)(a): probability of winning within d draws.
- E[score | win, a]: expected hand value if we win after discarding a.
- Ukeire vector: 34-element effective tile acceptance weighted by remaining counts.
```

```text
The bridge-side approximation for Hand-EV features uses public-state remaining counts until belief-weighted remaining counts from CT-SMC are threaded into the encoder path.
```

```text
Hydra should become a strong learned policy/value system with better public-belief-quality features and selective search layered on top only where it clearly pays. Hand-EV realism is a cheaper, earlier lever than deeper AFBS expansion.
```

Artifact G — benchmark and testing artifacts:

```rust
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use hydra_core::bridge;
use hydra_core::encoder::ObservationEncoder;
use hydra_core::game_loop::{FirstActionSelector, GameRunner};
use hydra_core::safety::SafetyInfo;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

fn bench_encoder(c: &mut Criterion) {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(42), 0, rule);
    let obs = state.get_observation(0);
    let safety = SafetyInfo::new();
    let mut encoder = ObservationEncoder::new();

    c.bench_function("encode_observation", |b| {
        b.iter(|| {
            bridge::encode_observation(&mut encoder, &obs, &safety, None);
            black_box(encoder.as_slice());
        });
    });
}
```

```text
golden_encoder.rs: regression tests for the encoder. Compares encoder output against saved golden snapshots. Catches silent encoding drift when any channel logic changes.
```

```text
Property-based tests verify encoder channel bounds and tile count conservation.
```

Artifact H — additional encoder/runtime documentation excerpts:

```text
Observation Encoder tensor shape: 192 x 34 float tensor (6,528 values). The first 85 channels retain the baseline public+safety encoding; the remaining channels provide fixed-shape search/belief and Hand-EV context with zero-fill plus explicit presence masks when dynamic features are unavailable.
```

```text
Group D Hand-EV context is fixed-shape and already part of the live encoder path, not a future extension.
```

```text
DirtyFlags includes HAND_EV in AFTER_DRAW, AFTER_DISCARD, and AFTER_CALL presets. That means the runtime already assumes Hand-EV is part of the incremental encoding contract.
```

```text
Hand-EV should become: a semantically clean local evaluator with exact one-step anchors, not a broad offensive oracle project first.
```

```text
Bridge extractors compute public remaining counts by subtracting visible hand, melds, discards, and dora indicators from a 4-copy prior. CT-SMC weighted counts can override that path when context is present.
```

Artifact I — doctrinal validation-gate excerpts:

```text
Every “guarantee-like” claim must be either a theorem, a bound with explicit constants, or an empirical gate with a measurable pass/fail threshold.
```

```text
High-impact drift only; not a full doc rewrite.
Keep the strongest old ideas in a reserve shelf so they are not lost.
Define the clearest version of Hydra to build right now.
```

```text
Best combined reading:
- stronger target generation is a better immediate lever than a giant search rewrite
- AFBS should be selective and specialist, not the default path everywhere
- Hand-EV is worth moving earlier than deeper AFBS expansion
```

Artifact J — additional engine/runtime documentation excerpts:

```text
Observation Encoder baseline prefix channel layout includes 23 safety channels in channels 62-84, and the current implementation extends beyond the old baseline with fixed-shape search/belief and Hand-EV context.
```

```text
ObservationEncoder holds a pre-allocated [f32; 192 * 34] buffer marked repr(C) for predictable memory layout.
```

```text
Incremental encoding matters for performance: after draw, discard, and call events, only selected channel groups are recomputed; Hand-EV is included in those dirty-flag presets.
```

```text
The bridge module is the translation layer from observation structs into encoder-ready data. Hand-EV is computed before encode_with_context injects it into the fixed superset tensor.
```

```text
Testing includes golden encoder regression, replay validation, property-based invariants, and end-to-end game-loop integration. Silent encoding drift is treated as a real failure mode.
```

Artifact K — additional bridge excerpts:

```rust
/// Compute public-state remaining tile counts from a zero-copy observation.
#[inline]
pub fn extract_public_remaining_counts_ref(
    hand: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
) -> [f32; NUM_TILE_TYPES] {
    extract_public_remaining_counts(hand, discards, melds, dora)
}

/// Compute bridge-side Hand-EV features from a zero-copy observation path.
#[inline]
pub fn compute_public_hand_ev_ref(
    hand: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
) -> HandEvFeatures {
    compute_public_hand_ev(hand, discards, melds, dora)
}

#[inline]
pub fn encode_observation(
    encoder: &mut ObservationEncoder,
    obs: &Observation,
    safety: &SafetyInfo,
    drawn_tile: Option<u8>,
) -> [f32; OBS_SIZE] {
    let search_context = SearchContext::default();
    encode_observation_with_search_context(encoder, obs, safety, drawn_tile, &search_context)
}
```

Artifact L — more doctrine excerpts:

```text
Fix semantic wrongness before adding broader complexity.
```

```text
Selective search layered on top only where it clearly pays. Not a giant search project first.
```

```text
Hand-EV realism improvements are a cheaper, higher-ROI upgrade than broader search.
```


## 0. Rename the thing before changing any math

**Replace the module-level label `oracle features` with `local self-draw hand-progress features`.**

Reason:

* The current module is **not** an oracle module in the usual RL / imperfect-information sense. In the oracle-guiding literature, *oracle observation* means information unavailable at execution time and available only in training or hindsight. The VLOG paper describes Suphx’s oracle-guiding method as a heuristic use of oracle observation during training, not as an online runtime feature family. ([OpenReview][1])
* The original Suphx paper’s look-ahead features are also **not** the current repo’s `P_tenpai / P_win / E[score|win] + ukeire` summary. Suphx describes **100+** look-ahead vectors built by DFS, keyed by winning-score/replacement-depth conditions, under explicit simplifications. It is a broader feature family than the current module. ([arXiv][2])

**New top-level sentence:**

```rust
//! Hand-EV local self-draw features: exact one-step hand-progress anchors
//! under a supplied draw-mass model, plus reserved slots for future exact
//! multi-draw local projection.
```

Do **not** call the live module `oracle` again unless it actually consumes perfect hidden information at runtime.

---

## 1. What the current quantities really mean

Everything below is directly supported by Artifact A/B/E/K unless explicitly marked as inference.

### 1.1 Core state used by the current module

For a discard candidate `a`:

* `after_a = hand - e_a`
* `s_a = shanten(after_a)`
* `R_t = remaining[t]`
* `R = sum_t R_t`

The current code treats `remaining` as a nonnegative tile-weight vector. It may be integer public unseen-copy counts or fractional CT-SMC weighted counts.

### 1.2 `ukeire[discard][draw_tile]` — current meaning

Current code:

[
U_a(t) = R_t \cdot \mathbf{1}\big[\operatorname{shanten}(after_a + e_t) < s_a\big]
]

This is **exact**, conditional on:

* the supplied `shanten_fn`
* the supplied `remaining` vector

So the current `ukeire` is **not** “general hand improvement,” **not** “effective tiles” in the broad Mortal/Akochan sense, and **not** a probability vector. It is:

> **one-self-draw shanten-lowering live-copy mass by tile type**

That meaning is good and should be kept.

Immediate corollaries:

* `sum_t U_a(t)` is the mass of draws that lower shanten by at least 1.
* If `remaining` is normalized into a draw distribution, `sum_t U_a(t) / R` is the exact probability that the next self-draw lowers shanten.

### 1.3 `acceptance = sum(ukeire)` — current meaning

Current code uses:

[
A_a = \sum_t U_a(t)
\qquad
p^\downarrow_a = A_a / R
]

`acceptance_ratio` is therefore:

> **probability that the next self-draw lowers shanten by at least 1**, under the supplied draw-mass model.

That is exact.

It is **not** automatically:

* probability of reaching tenpai
* probability of winning
* probability of “good shape”
* generic offensive EV

### 1.4 `tenpai_prob[discard][h]` — current meaning

Current code computes:

[
q_a =
\begin{cases}
1 & s_a \le 0 \
A_a / R & s_a > 0
\end{cases}
]

[
c^{tenpai}_{a,h} = \texttt{continuation_boost}(h, s_a, A_a/R)
]

[
\texttt{tenpai_prob}[a][h]
==========================

1 - (1-q_a)^{h+1}(1-c^{tenpai}_{a,h})
]

This is **not** semantically “probability of reaching tenpai within `h+1` self-draws” except in the trivial already-tenpai case.

Useful special cases:

* If `s_a = 1`, then `A_a / R` is the **exact one-draw tenpai probability**, but current horizon-0 output is

[
1 - (1-p)(1-0.65p) = 1.65p - 0.65p^2
]

not `p`.

* If `s_a \ge 2`, then `A_a / R` is only the probability of moving from `s_a` to a lower shanten. Current horizon-0 output becomes

[
1 - (1-p)(1-0.45p) = 1.45p - 0.45p^2
]

which is not a tenpai probability at all.

### 1.5 `win_prob[discard][h]` — current meaning

Current code first computes `immediate_win_probability` only when `s_a == 0`:

[
w_a =
\frac{\sum_t R_t \cdot \mathbf{1}[\operatorname{shanten}(after_a + e_t) < 0]}{R}
]

That is an exact **one-draw structural completion probability** conditional on:

* `after_a` being tenpai
* `remaining` being a valid next-draw mass model
* “`shanten < 0`” being accepted as the win predicate

Then current code replaces this with

[
b_a = \max(w_a,\ 0.35 A_a/R)
]

and then applies the same continuation transform:

[
\texttt{win_prob}[a][h]
=======================

1 - (1-b_a)^{h+1}(1-c^{win}_{a,h})
]

So current `win_prob` is:

> a heuristic monotone offensive proxy derived from either exact one-draw structural wait mass (only in tenpai) or a `0.35 * acceptance_ratio` fallback

It is **not** a clean probability of winning.

Special case:

* If `s_a = 0` and `w_a = p`, then current horizon-0 output is

[
1 - (1-p)^2 = 2p - p^2
]

not `p`.

Also, current code is only modeling **self-draw** addition of one tile. So even if the field were exact, it would still be **tsumo-only local win probability**, not generic “win probability.”

### 1.6 `expected_score[discard]` — current meaning

Current code computes:

[
\texttt{expected_score}[a]
==========================

\texttt{win_prob}[a][2] \cdot \texttt{conditional_score_estimate}(hand, a, A_a, s_a)
]

This field is currently:

> **a 3-draw offensive utility heuristic in score-like units**

It is **not** any of these:

* not (E[\text{score} \mid \text{win}, a])
* not (E[\text{score} \cdot \mathbf{1}{\text{win}} \mid a]) under any exact model
* not conditional on actual winning hands
* not computed from `after_discard`
* not computed from a legal scoring engine

It is especially broken because `conditional_score_estimate(...)`:

* uses the **original hand**, not `after_discard`
* uses raw `acceptance` count, not a probability
* includes arbitrary hand-shape bonuses
* adds a hardcoded bonus for `discard >= 27` (honor discard)
* never evaluates actual winning hand distributions

This field should not be kept live under the current name.

### 1.7 Upstream `remaining` sources — current meaning

Observed:

* public path: starts from `4.0` and subtracts visible hand, discards, meld tiles, dora indicators
* CT-SMC path: sums `weighted_mean_tile_count(tile, col)` over all four columns and passes that as `remaining`

So the current module is being fed **a generic hidden-tile mass vector**, not a clearly named “wall posterior.”

That is fine as an input, but the input name must stop implying “exact remaining wall counts.”

---

## 2. What is semantically broken or misleading

### 2.1 The field names overclaim

Broken labels:

* `oracle`
* `win_prob`
* `expected_score`
* `P_tenpai^(d)` for `d in {1,2,3}` as if all are exact local probabilities
* `E[score | win, a]`

### 2.2 Multi-draw semantics are unspecified because follow-up policy is unspecified

For any exact `d > 1` projection, you must specify:

* what happens on later self-draws
* which discard policy is used after each later draw
* whether the objective is tenpai probability, tsumo probability, conditional score, or EV
* whether riichi/dama choices are part of the policy

Current code avoids this by using `continuation_boost`. That makes the outputs heuristic, not probabilistic.

### 2.3 The “win” field is really “self-draw structural completion under one specific simplification”

The current `immediate_win_probability` only tests adding one self-drawn tile. It ignores:

* ron
* calls
* opponent actions
* explicit yaku legality unless `shanten < 0` happens to coincide with legal agari

So even the exact anchor inside the current module is narrower than the field name.

### 2.4 The score field is unsalvageable as a live semantic claim

Do not rename it to “approximate EV” and keep it in the main path. It is too contaminated:

* heuristic win probability
* heuristic score proxy
* original-hand features instead of post-discard / winning-hand features
* arbitrary discard-type bonus
* no scoring context
* no explicit riichi policy

Move it off the live path.

### 2.5 The upstream count sources are being mixed with stronger wording than they support

Keep this distinction:

* **exact given the supplied draw-mass vector**
* **approximate because the draw-mass vector itself is approximate**

Public unseen counts and CT-SMC hidden-tile weighted counts can be valid **draw-mass models**. They are not automatically exact live-wall counts.

### 2.6 Concrete distortions from the current code

Using Artifact A formulas:

* Example A: 2-shanten after discard, `20/70` hidden mass lowers shanten by 1.

  * current `tenpai_prob[0] = 0.377551`
  * exact repaired one-draw tenpai probability = `0.0`

* Example B: tenpai after discard, `8/70` are winning waits.

  * current `win_prob[0] = 0.215510`
  * exact one-draw tsumo probability = `8/70 = 0.114286`

Those are not calibration errors. They are semantic errors.

---

## 3. Repaired meanings

## 3.1 Live semantic contract

The repaired live module should mean exactly this:

> For each discard candidate `a`, compute exact one-step local self-draw anchors under an explicit draw-mass model. Anything beyond that is either reserved or moved to a separate exact local projection module.

### Keep exact

#### A. `ukeire[a][t]`

Keep current meaning exactly:

[
U_a(t) = m_t \cdot \mathbf{1}\big[\operatorname{shanten}(after_a + e_t) < \operatorname{shanten}(after_a)\big]
]

where `m_t` is the supplied draw mass.

Name in docs:

> exact one-draw shanten-lowering draw mass by tile type

#### B. `tenpai_prob[a][0]`

Redefine slot 0 only:

[
P^{(1)}_{\text{tenpai}}(a)
==========================

\begin{cases}
1 & \operatorname{shanten}(after_a) \le 0 \
\frac{\sum_t m_t \cdot \mathbf{1}[\operatorname{shanten}(after_a + e_t) \le 0]}{\sum_t m_t \cdot \mathbf{1}[after_a[t] < 4]} & \text{otherwise}
\end{cases}
]

Meaning:

> exact probability of being in tenpai within one further self-draw, under the supplied draw-mass model

This is stronger and cleaner than current semantics.

#### C. `win_prob[a][0]`

Redefine slot 0 only:

[
P^{(1)}_{\text{tsumo}}(a)
=========================

\frac{\sum_t m_t \cdot \mathbf{1}[\text{legal_tsumo}(after_a + e_t,\ t,\ ctx)]}{\sum_t m_t \cdot \mathbf{1}[after_a[t] < 4]}
]

Meaning:

> exact probability of tsumo within one further self-draw, under the supplied draw-mass model and a legal-win evaluator

Call it **tsumo probability** in docs, even if the Rust field name stays `win_prob` for compatibility.

### Keep approximate, but say so explicitly

#### D. The draw-mass source

Two acceptable approximate upstream models:

* **Public unseen exchangeable model**
  Use public unseen-copy counts and normalize them. This is an approximation to next-draw probability, not an observed wall posterior.

* **CT-SMC hidden-mass model**
  If you only have total hidden-copy expectations, normalize those and say so.
  If CT-SMC can expose **wall marginals**, prefer that instead.

The exactness claim for `ukeire`, `tenpai_prob[...,0]`, and `win_prob[...,0]` is always:

> exact **given the supplied draw-mass model**

### Drop / demote from the live path

#### E. `tenpai_prob[a][1]`, `tenpai_prob[a][2]`

#### F. `win_prob[a][1]`, `win_prob[a][2]`

Do not populate them with heuristics.

Set them to `0.0` and mark them **reserved** until an exact local multi-draw projection exists with an explicit follow-up policy.

#### G. `expected_score[a]`

Immediate recommendation:

* set to `0.0`
* mark **reserved**

Reason: exact score semantics require all of the following to be specified:

* legal tsumo predicate
* scoring context
* whether value means raw hand value, round gain, or total delta
* whether honba / kyotaku are included
* explicit riichi/dama policy
* if riichi is allowed, how ura expectation is treated

Until that policy exists, the score slot must not carry live meaning.

---

## 4. The exact shape-preserving implementation

Do **not** change the tensor shape now. Change the semantics and zero-fill reserved slots.

## 4.1 Keep the existing struct, rewrite its comments

```rust
pub struct HandEvFeatures {
    /// [discard][0] = exact P(tenpai within 1 self-draw | discard, draw-mass model)
    /// [discard][1], [discard][2] = reserved, currently 0.0
    pub tenpai_prob: [[f32; 3]; NUM_TILE_TYPES],

    /// [discard][0] = exact P(tsumo within 1 self-draw | discard, draw-mass model, win evaluator)
    /// [discard][1], [discard][2] = reserved, currently 0.0
    pub win_prob: [[f32; 3]; NUM_TILE_TYPES],

    /// Reserved. Currently 0.0.
    pub expected_score: [f32; NUM_TILE_TYPES],

    /// Exact one-draw shanten-lowering draw mass by tile type.
    pub ukeire: [[f32; NUM_TILE_TYPES]; NUM_TILE_TYPES],
}
```

## 4.2 Add explicit input semantics

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawMassProvenance {
    PublicUnseenExchangeable,
    CtSmcHiddenMassExchangeable,
    CtSmcWallPosterior,
}

#[derive(Clone, Copy, Debug)]
pub struct DrawMass {
    pub mass: [f32; NUM_TILE_TYPES],
    pub provenance: DrawMassProvenance,
}
```

Reason: stop pretending every `[f32; 34]` is “remaining counts.”

## 4.3 Add a win evaluator trait instead of hardcoding scoring logic in Hand-EV

`riichienv-core` already exposes agari detection, hand evaluation, yaku, and score calculation under Apache-2.0, so the core Hand-EV module should call into a narrow callback/trait rather than grow its own scoring semantics. ([Docs.rs][3])

```rust
pub trait TsumoEvaluator {
    /// Exact legal tsumo predicate under the chosen rule/policy context.
    fn is_legal_tsumo(
        &self,
        hand_after_draw: &[u8; NUM_TILE_TYPES],
        winning_tile: u8,
    ) -> bool;
}
```

Phase 1 keeps score reserved, so `point_gain(...)` is not needed yet.

---

## 5. Core computation patch

Delete these functions from the live path:

* `conditional_score_estimate`
* `immediate_win_probability`
* `continuation_boost`

Keep them only in a legacy shelf if you want offline ablations.

### 5.1 Replace `compute_hand_ev_with_shanten_fn` with an exact one-step implementation

```rust
pub fn compute_hand_ev_exact_1step(
    hand: &[u8; NUM_TILE_TYPES],
    draw_mass: &DrawMass,
    shanten_fn: &dyn Fn(&[u8; NUM_TILE_TYPES]) -> i8,
    tsumo_eval: Option<&dyn TsumoEvaluator>,
) -> HandEvFeatures {
    let mut out = HandEvFeatures::default();

    for discard in 0..NUM_TILE_TYPES {
        if hand[discard] == 0 {
            continue;
        }

        let mut after = *hand;
        after[discard] -= 1;
        let base_shanten = shanten_fn(&after);

        let mut eligible_mass = 0.0f32;
        let mut tenpai_mass = 0.0f32;
        let mut tsumo_mass = 0.0f32;
        let mut uke = [0.0f32; NUM_TILE_TYPES];

        for t in 0..NUM_TILE_TYPES {
            let m = draw_mass.mass[t];
            if m <= 0.0 {
                continue;
            }
            if after[t] >= 4 {
                continue; // impossible fifth copy; remove from denominator
            }

            eligible_mass += m;

            let mut next = after;
            next[t] += 1;
            let next_shanten = shanten_fn(&next);

            if next_shanten < base_shanten {
                uke[t] = m;
            }
            if next_shanten <= 0 {
                tenpai_mass += m;
            }
            if let Some(eval) = tsumo_eval {
                if eval.is_legal_tsumo(&next, t as u8) {
                    tsumo_mass += m;
                }
            }
        }

        out.ukeire[discard] = uke;

        if eligible_mass > 0.0 {
            out.tenpai_prob[discard][0] = if base_shanten <= 0 {
                1.0
            } else {
                (tenpai_mass / eligible_mass).clamp(0.0, 1.0)
            };

            out.win_prob[discard][0] = if tsumo_eval.is_some() {
                (tsumo_mass / eligible_mass).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }

        // reserved slots
        out.tenpai_prob[discard][1] = 0.0;
        out.tenpai_prob[discard][2] = 0.0;
        out.win_prob[discard][1] = 0.0;
        out.win_prob[discard][2] = 0.0;
        out.expected_score[discard] = 0.0;
    }

    out
}
```

### 5.2 Important denominator rule

Use

[
M_a = \sum_t m_t \cdot \mathbf{1}[after_a[t] < 4]
]

not blindly `sum(draw_mass)`.

Reason: once `draw_mass` becomes a model object, you want robust behavior even if the upstream model is slightly inconsistent.

### 5.3 Keep `compute_ukeire` or fold it into the same loop

Either is fine.

Recommendation: **fold it into the same loop** to avoid duplicated shanten calls.

---

## 6. Bridge patch

## 6.1 Rename extractors to stop overstating their meaning

Rename:

* `extract_public_remaining_counts` -> `extract_public_draw_mass`
* `extract_ct_smc_remaining_counts` -> `extract_ct_smc_draw_mass`

### Public path

Current implementation is acceptable as a **public unseen exchangeable** draw-mass model:

```rust
pub fn extract_public_draw_mass(...) -> DrawMass {
    DrawMass {
        mass: existing_logic(...),
        provenance: DrawMassProvenance::PublicUnseenExchangeable,
    }
}
```

### CT-SMC path

Current implementation should be renamed to reflect what it actually produces.

If the four CT-SMC columns do **not** mean “live wall only,” then current code is not extracting wall counts; it is extracting **total hidden mass**.

So:

```rust
pub fn extract_ct_smc_draw_mass(ct_smc: &CtSmc) -> DrawMass {
    DrawMass {
        mass: existing_sum_all_columns_logic(ct_smc),
        provenance: DrawMassProvenance::CtSmcHiddenMassExchangeable,
    }
}
```

If CT-SMC can expose direct wall marginals, add:

```rust
pub fn extract_ct_smc_wall_posterior(ct_smc: &CtSmc) -> DrawMass {
    DrawMass {
        mass: ct_smc.expected_wall_tile_counts(),
        provenance: DrawMassProvenance::CtSmcWallPosterior,
    }
}
```

and prefer that path.

## 6.2 Bridge scoring / win legality

If you wire exact one-step `win_prob[...,0]`, do it in the bridge, where rule context already exists.

The dependency stack already contains the primitives for agari detection and scoring. Use those there; keep Hand-EV core generic. ([Docs.rs][3])

---

## 7. Encoder mapping

Keep the channel layout. Change only the semantic contract.

### New Group D meaning

* `CH_HAND_EV_TENPAI + 0` = exact one-step tenpai probability
* `CH_HAND_EV_TENPAI + 1` = reserved zero
* `CH_HAND_EV_TENPAI + 2` = reserved zero
* `CH_HAND_EV_WIN + 0` = exact one-step tsumo probability
* `CH_HAND_EV_WIN + 1` = reserved zero
* `CH_HAND_EV_WIN + 2` = reserved zero
* `CH_HAND_EV_SCORE` = reserved zero
* `CH_HAND_EV_UKEIRE + draw_tile` = exact one-step shanten-lowering draw mass by tile type
* `CH_HAND_EV_MASK` = `1.0` when Group D is present

### Add explicit version constants

```rust
pub const HAND_EV_SEMANTICS_VERSION: u32 = 2;
pub const HAND_EV_VALID_TENPAI_HORIZONS: usize = 1;
pub const HAND_EV_VALID_WIN_HORIZONS: usize = 1;
pub const HAND_EV_SCORE_VALID: bool = false;
```

This is the cheapest way to stop future readers from inferring false semantics from legacy slot count.

---

## 8. Documentation patch

Replace every documentation bullet that currently says:

* `P_tenpai^(d)(a): probability ... within d in {1,2,3}`
* `P_win^(d)(a): probability ... within d draws`
* `E[score | win, a]`
* `oracle features`

with:

> Hand-EV local self-draw features:
>
> * exact `P_tenpai^(1)(a)` in slot 0
> * exact `P_tsumo^(1)(a)` in slot 0 when a legal-win evaluator is provided
> * exact one-step shanten-lowering draw-mass vector (`ukeire`)
> * additional legacy horizon slots reserved for future exact local projection and currently zero-filled

Also add one sentence:

> The public and CT-SMC bridge paths provide approximate draw-mass models; Hand-EV values are exact **conditional on the supplied draw-mass model**.

---

## 9. Validation gates

These are required. No “should probably work” language.

## 9.1 Exactness gate for one-step semantics

For a brute-force enumerator over valid draw tiles, sampled over random states:

[
\max | \text{computed} - \text{enumerated} | \le 10^{-6}
]

Apply to:

* `ukeire[a][t]`
* `tenpai_prob[a][0]`
* `win_prob[a][0]` when `TsumoEvaluator` is wired

## 9.2 Required unit tests

Add or replace tests with these exact semantic checks:

1. `ukeire_is_exact_shanten_lowering_mass`
2. `one_step_tenpai_prob_zero_for_two_shanten_without_direct_tenpai_draws`
3. `one_step_tenpai_prob_equals_wait_mass_for_one_shanten`
4. `one_step_tenpai_prob_is_one_when_after_discard_already_tenpai`
5. `one_step_tsumo_prob_equals_wait_mass_for_tenpai_hand`
6. `reserved_horizon_slots_are_zero`
7. `reserved_score_slot_is_zero`
8. `win_prob_never_exceeds_tenpai_prob`
9. `impossible_fifth_copy_mass_is_excluded_from_denominator`

## 9.3 Property tests

For random valid states:

* `0 <= tenpai_prob[a][0] <= 1`
* `0 <= win_prob[a][0] <= 1`
* `win_prob[a][0] <= tenpai_prob[a][0]`
* if `ukeire[a][t] > 0`, then `shanten(after_a + e_t) < shanten(after_a)`
* if `shanten(after_a) > 1` and no direct tenpai draws exist, `tenpai_prob[a][0] == 0`
* reserved slots stay exactly zero

## 9.4 Golden encoder gate

Because the channel semantics change, all golden encoder snapshots must be regenerated.

This is **intentional high-impact drift**, not a silent drift.

## 9.5 Model compatibility gate

Do not reuse checkpoints trained on old Group D semantics.

Bump model metadata / checkpoint naming with `HAND_EV_SEMANTICS_VERSION=2`.

## 9.6 Runtime gate

Use the existing Criterion bench and fail the change if median `encode_observation` latency regresses by more than **25%** in the no-score, one-step-only version.

That threshold is explicit and cheap to enforce.

---

## 10. Reserve shelf: what to keep, but not live

Move the current heuristic ideas into `hand_ev_legacy.rs` or a feature flag:

* `continuation_boost`
* `conditional_score_estimate`
* current `expected_score`
* any legacy 2/3-draw monotone offense proxy

Mark them:

> legacy offensive heuristics, not semantically exact probabilities or EV

This preserves the old ideas without contaminating the live encoder contract.

---

## 11. Phase 2, only if you want the 2/3-draw slots to mean something exact

Do **not** fill those slots heuristically.

Build a separate exact local projection module with an explicit follow-up policy.

### State

For exact local self-draw projection, the recursive state must include at least:

* 13-tile hand after a discard
* draw-mass / wall state
* remaining self-draw budget `d`
* menzen / open state
* legal-win context
* explicit riichi policy if score is involved

### Recurrence

For a chosen follow-up policy `π`:

[
V^\pi_d(h, m)
=============

\sum_t \Pr(t \mid m), G^\pi_d(h+t, m-e_t)
]

where `G` either:

* records immediate tsumo if legal, or
* applies the next discard chosen by `π` and recurses with `d-1`

You may optimize different objectives with different policies:

* maximize tenpai probability
* maximize tsumo probability
* maximize expected point gain

Those are **not the same policy** in general. That is exactly why current multi-draw semantics are underspecified.

### If you implement Phase 2

Then and only then:

* populate horizon-2 / horizon-3 slots
* define whether they are “under policy π_EV”, “under policy π_tsumo”, or “under current policy net”
* reintroduce a score slot with an exact definition such as:

[
E[\Delta\text{score} \mid \text{tsumo within horizon}, a, \pi]
]

Do not ship a score channel until that definition is written down first.

---

## 12. External references to validate against

### Original Suphx paper

Use it for **provenance and scope only**:

* Suphx look-ahead features are a much broader family than the current repo summary.
* They are built with DFS under explicit simplifications.
* They include score/depth-conditioned feature vectors, not just a small scalar summary. ([arXiv][2])

### Oracle-guiding clarification

Use it to justify removing the word `oracle` from the live runtime feature label unless you truly mean hidden-information training-time guidance. ([OpenReview][1])

### riichienv-core

Use it as the implementation substrate for exact one-step legal tsumo checks if you wire them now:

* hand evaluation
* agari detection
* yaku
* score calculation
* Apache-2.0 licensing ([Docs.rs][3])

### Mortal

Use it as a **design reference only**, not source to copy blindly:

* official docs say the project started as a Suphx reproduction
* official repo is AGPL-3.0 licensed ([Mortal][4])

---

## 13. Final live contract to implement now

Implement **this** and stop:

1. Rename the feature family to **local self-draw hand-progress features**.
2. Rename `remaining` semantics to **draw mass**.
3. Keep `ukeire` exact as one-step shanten-lowering draw mass.
4. Set `tenpai_prob[...,0]` to exact one-step tenpai probability.
5. Set `win_prob[...,0]` to exact one-step tsumo probability if a legal-win evaluator is provided; otherwise `0.0`.
6. Set `tenpai_prob[...,1..] = 0.0`.
7. Set `win_prob[...,1..] = 0.0`.
8. Set `expected_score[...] = 0.0`.
9. Move all current continuation / score heuristics to a legacy shelf.
10. Regenerate goldens, bump semantic version, retrain downstream models.

That is the strongest exact repair with the least guesswork.

[1]: https://openreview.net/pdf?id=pjqqxepwoMy "https://openreview.net/pdf?id=pjqqxepwoMy"
[2]: https://arxiv.org/pdf/2003.13590.pdf "https://arxiv.org/pdf/2003.13590.pdf"
[3]: https://docs.rs/riichienv-core/latest/riichienv_core/all.html "https://docs.rs/riichienv-core/latest/riichienv_core/all.html"
[4]: https://mortal.ekyu.moe/ref/meta.html "https://mortal.ekyu.moe/ref/meta.html"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
