use super::*;

#[test]
fn new_safety_info_is_zeroed() {
    let si = SafetyInfo::new();
    for opp in 0..NUM_OPPONENTS {
        assert_eq!(si.genbutsu_all[opp], 0);
        assert_eq!(si.genbutsu_tedashi[opp], 0);
        assert_eq!(si.genbutsu_riichi_era[opp], 0);
        assert!(si.suji[opp].iter().all(|&v| v == 0.0));
        assert!(!si.opponent_riichi[opp]);
        assert_eq!(si.cached_tenpai_prob[opp], 0.0);
        assert_eq!(si.half_suji[opp], 0);
        assert!(si.matagi[opp].iter().all(|&v| v == 0.0));
    }
    assert_eq!(si.kabe, 0);
    assert_eq!(si.one_chance, 0);
    assert!(si.visible_counts.iter().all(|&v| v == 0));
}

#[test]
fn on_discard_sets_genbutsu_all() {
    let mut si = SafetyInfo::new();
    si.on_discard(5, 0, false); // 6m tsumogiri by opponent 0
    assert!(bit_test(si.genbutsu_all[0], 5));
    assert!(!bit_test(si.genbutsu_tedashi[0], 5));
    assert!(!bit_test(si.genbutsu_all[1], 5)); // other opponents unaffected
}

#[test]
fn on_discard_tedashi_sets_both_flags() {
    let mut si = SafetyInfo::new();
    si.on_discard(10, 1, true); // 2p tedashi by opponent 1
    assert!(bit_test(si.genbutsu_all[1], 10));
    assert!(bit_test(si.genbutsu_tedashi[1], 10));
}

#[test]
fn riichi_then_discard_sets_riichi_era() {
    let mut si = SafetyInfo::new();
    si.on_riichi(2);
    si.on_discard(0, 2, false); // 1m after opponent 2's riichi
    assert!(bit_test(si.genbutsu_riichi_era[2], 0));
    // Before riichi, should not be set
    assert!(!bit_test(si.genbutsu_riichi_era[0], 0));
}

#[test]
fn suji_from_4m_discard() {
    // 4m = index 3. Suji targets: 1m (index 0) and 7m (index 6)
    let mut si = SafetyInfo::new();
    si.on_discard(3, 0, false);
    assert_eq!(si.suji[0][0], 1.0); // 1m gets suji
    assert_eq!(si.suji[0][6], 1.0); // 7m gets suji
    assert_eq!(si.suji[0][3], 0.0); // 4m itself has no suji from this
}

#[test]
fn suji_honors_produce_none() {
    let mut si = SafetyInfo::new();
    si.on_discard(27, 0, false); // East wind (first honor)
    for i in 0..NUM_TILES {
        assert_eq!(si.suji[0][i], 0.0);
    }
}

#[test]
fn kabe_at_four_visible() {
    let mut si = SafetyInfo::new();
    for _ in 0..3 {
        si.on_discard(15, 0, false); // discard 7p three times
    }
    assert!(!bit_test(si.kabe, 15));
    assert!(bit_test(si.one_chance, 15));
    si.on_discard(15, 1, false); // 4th copy
    assert!(bit_test(si.kabe, 15));
    assert!(!bit_test(si.one_chance, 15)); // no longer one-chance at 4
}

#[test]
fn on_call_updates_visible_counts() {
    let mut si = SafetyInfo::new();
    si.on_call(&[0, 1, 2]); // chi 1m-2m-3m
    assert_eq!(si.visible_counts[0], 1);
    assert_eq!(si.visible_counts[1], 1);
    assert_eq!(si.visible_counts[2], 1);
}

#[test]
fn on_dora_revealed_updates_visible() {
    let mut si = SafetyInfo::new();
    si.on_dora_revealed(33); // last honor tile
    assert_eq!(si.visible_counts[33], 1);
}

#[test]
fn reset_clears_everything() {
    let mut si = SafetyInfo::new();
    si.on_discard(5, 0, true);
    si.on_riichi(1);
    si.set_tenpai_prediction(2, 0.9);
    si.on_dora_revealed(20);
    si.reset();
    assert!(!bit_test(si.genbutsu_all[0], 5));
    assert!(!bit_test(si.genbutsu_tedashi[0], 5));
    assert!(!si.opponent_riichi[1]);
    assert_eq!(si.cached_tenpai_prob[2], 0.0);
    assert_eq!(si.visible_counts[20], 0);
}

#[test]
fn tenpai_hint_activates_from_cached_prediction() {
    let mut si = SafetyInfo::new();
    assert!(!si.tenpai_hint_active(0));
    si.set_tenpai_prediction(0, 0.6);
    assert!(si.tenpai_hint_active(0));
}

#[test]
fn tenpai_hint_clamps_predictions() {
    let mut si = SafetyInfo::new();
    si.set_tenpai_prediction(1, 10.0);
    si.set_tenpai_prediction(2, -5.0);
    assert_eq!(si.cached_tenpai_prob[1], 1.0);
    assert_eq!(si.cached_tenpai_prob[2], 0.0);
}

#[test]
fn out_of_bounds_ignored() {
    let mut si = SafetyInfo::new();
    // Should not panic
    si.on_discard(34, 0, false); // tile out of bounds
    si.on_discard(0, 3, false); // opponent out of bounds
    si.on_dora_revealed(255); // way out of bounds
    si.on_call(&[35, 100]); // tiles out of bounds
}

#[test]
fn half_suji_center_tile_one_partner() {
    // Discard 1m (index 0) -> 4m (index 3) gets suji.
    // 4m is center tile with partners 1m(0) and 7m(6).
    // Only 1m is genbutsu -> half suji (0.5).
    let mut si = SafetyInfo::new();
    si.on_discard(0, 0, false); // 1m genbutsu
    assert_eq!(si.suji[0][3], 0.5); // 4m half suji
    assert!(bit_test(si.half_suji[0], 3));
}

#[test]
fn half_suji_center_tile_both_partners() {
    // Discard both 1m and 7m -> 4m gets full suji.
    let mut si = SafetyInfo::new();
    si.on_discard(0, 0, false); // 1m genbutsu
    si.on_discard(6, 0, false); // 7m genbutsu
    assert_eq!(si.suji[0][3], 1.0); // 4m full suji
    assert!(!bit_test(si.half_suji[0], 3)); // not half
}

#[test]
fn half_suji_edge_tile_unaffected() {
    // 1m (index 0) is edge tile, only partner is 4m.
    // Discarding 4m -> 1m gets full 1.0, not half.
    let mut si = SafetyInfo::new();
    si.on_discard(3, 0, false); // 4m genbutsu
    assert_eq!(si.suji[0][0], 1.0); // 1m full suji
    assert!(!bit_test(si.half_suji[0], 0)); // edge tile never half
}

#[test]
fn matagi_from_tedashi() {
    // Tedashi discard of 5m (index 4) marks 4m and 6m as matagi.
    let mut si = SafetyInfo::new();
    si.on_discard(4, 1, true); // 5m tedashi by opp1
    assert_eq!(si.matagi[1][3], 1.0); // 4m matagi
    assert_eq!(si.matagi[1][5], 1.0); // 6m matagi
    assert_eq!(si.matagi[1][4], 0.0); // 5m itself no matagi
}

#[test]
fn matagi_not_from_tsumogiri() {
    // Tsumogiri should NOT set matagi.
    let mut si = SafetyInfo::new();
    si.on_discard(4, 0, false); // 5m tsumogiri
    assert_eq!(si.matagi[0][3], 0.0);
    assert_eq!(si.matagi[0][5], 0.0);
}

#[test]
fn matagi_edge_tiles() {
    // 1m (index 0) tedashi -> only 2m (index 1) gets matagi (no -1).
    let mut si = SafetyInfo::new();
    si.on_discard(0, 0, true); // 1m tedashi
    assert_eq!(si.matagi[0][1], 1.0); // 2m matagi
    // 9m (index 8) tedashi -> only 8m (index 7) gets matagi (no +1).
    si.on_discard(8, 0, true); // 9m tedashi
    assert_eq!(si.matagi[0][7], 1.0); // 8m matagi
}

#[test]
fn matagi_honors_ignored() {
    // Honor tile tedashi should NOT produce matagi.
    let mut si = SafetyInfo::new();
    si.on_discard(27, 0, true); // East wind tedashi
    for i in 0..NUM_TILES {
        assert_eq!(si.matagi[0][i], 0.0);
    }
}
