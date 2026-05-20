use hydra_core::encoder::OBS_SIZE;
use hydra_data_core::{
    COMPACT_ADVANCED_TAIL_LEN, COMPACT_BASELINE_CHANNELS, GRP_PERM_TABLE, SCORE_BINS,
    score_delta_to_bin, score_delta_to_cdf, score_delta_to_pdf, score_delta_to_value,
    score_to_placement, score_to_placements, scores_to_grp_index,
};

#[test]
fn grp_table_has_24_unique_permutations() {
    let mut seen = std::collections::HashSet::new();
    for perm in &GRP_PERM_TABLE {
        assert!(seen.insert(*perm), "duplicate perm {perm:?}");
    }
    assert_eq!(seen.len(), 24);
}

#[test]
fn grp_index_uses_score_descending_and_seat_tie_breaks() {
    assert_eq!(scores_to_grp_index([40_000, 30_000, 20_000, 10_000]), Ok(0));

    let reversed = scores_to_grp_index([10_000, 20_000, 30_000, 40_000]).unwrap();
    assert!(reversed < 24);
    assert_ne!(reversed, 0);

    let tied = scores_to_grp_index([25_000, 25_000, 25_000, 25_000]).unwrap();
    assert_eq!(GRP_PERM_TABLE[tied as usize], [0, 1, 2, 3]);
}

#[test]
fn score_bin_pdf_cdf_and_value_clamp_edges() {
    assert_eq!(score_delta_to_bin(-50_000), 0);
    assert_eq!(score_delta_to_bin(60_000), SCORE_BINS - 1);
    let mid = score_delta_to_bin(5_000);
    assert!(mid > 0 && mid < SCORE_BINS - 1);

    let pdf = score_delta_to_pdf(5_000);
    assert_eq!(pdf.iter().filter(|&&v| v > 0.0).count(), 1);
    assert!((pdf.iter().sum::<f32>() - 1.0).abs() < 1e-5);

    let cdf = score_delta_to_cdf(5_000);
    for window in cdf.windows(2) {
        assert!(window[1] >= window[0]);
    }
    assert_eq!(cdf[score_delta_to_bin(5_000)], 1.0);
    assert_eq!(cdf[SCORE_BINS - 1], 1.0);

    assert_eq!(score_delta_to_value(100_000), 1.0);
    assert_eq!(score_delta_to_value(-100_000), -1.0);
    assert_eq!(score_delta_to_value(0), 0.0);
}

#[test]
fn placement_helpers_are_tie_stable_and_checked() {
    assert_eq!(
        score_to_placements([40_000, 30_000, 20_000, 10_000]),
        [0, 1, 2, 3]
    );
    assert_eq!(
        score_to_placements([25_000, 25_000, 25_000, 25_000]),
        [0, 1, 2, 3]
    );
    assert_eq!(
        score_to_placements([10_000, 40_000, 40_000, 20_000]),
        [3, 0, 1, 2]
    );

    assert_eq!(
        score_to_placement([40_000, 30_000, 20_000, 10_000], 0),
        Some(0)
    );
    assert_eq!(
        score_to_placement([40_000, 30_000, 20_000, 10_000], 3),
        Some(3)
    );
    assert_eq!(
        score_to_placement([40_000, 30_000, 20_000, 10_000], 4),
        None
    );
    assert_eq!(
        score_to_placement([40_000, 30_000, 20_000, 10_000], u8::MAX),
        None
    );
}

#[test]
fn compact_observation_shape_invariants_hold() {
    assert_eq!(OBS_SIZE, 192 * 34);
    assert_eq!(COMPACT_BASELINE_CHANNELS, 85);
    assert_eq!(
        COMPACT_ADVANCED_TAIL_LEN,
        OBS_SIZE - COMPACT_BASELINE_CHANNELS * 34
    );
}
