use super::*;
use riichienv_core::shanten::calc_shanten_from_counts;

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
            1 if counts[2] == 1 || counts[3] == 1 => 0,
            1 => 1,
            2 => {
                if counts[2] == 1 && counts[4] == 1 {
                    -1
                } else if (counts[1] == 1 && counts[2] == 1) || (counts[0] == 1 && counts[3] == 1) {
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
