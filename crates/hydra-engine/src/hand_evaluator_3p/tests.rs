use super::*;

#[test]
fn hand_from_text_and_wait_helpers_work_for_simple_tenpai_hand() {
    let eval = HandEvaluator3P::hand_from_text("123m123p123s111z2z")
        .expect("hand text should parse into evaluator");

    assert!(eval.is_tenpai());
    assert_eq!(eval.get_waits_u8(), vec![28]);
    assert_eq!(eval.get_waits(), vec![28u32]);

    let mut buf = [0u8; 34];
    let count = eval.get_waits_u8_into(&mut buf);
    assert_eq!(count, 1);
    assert_eq!(&buf[..count as usize], &[28]);
}

#[test]
fn waits_are_empty_when_tile_count_is_not_thirteen() {
    let eval = HandEvaluator3P::new(&[0, 4, 8], &[]);
    assert!(!eval.is_tenpai());
    assert!(eval.get_waits_u8().is_empty());

    let mut buf = [0u8; 34];
    assert_eq!(eval.get_waits_u8_into(&mut buf), 0);
}

#[test]
fn calc_returns_non_win_for_incomplete_hand() {
    let eval = HandEvaluator3P::new(&[0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 112, 116, 120], &[]);
    let result = eval.calc(124, &[], &[], None);

    assert!(!result.is_win);
    assert!(!result.has_win_shape);
    assert_eq!(result.han, 0);
    assert_eq!(result.fu, 0);
}

#[test]
fn calc_counts_sanma_dora_ura_and_kita() {
    let eval =
        HandEvaluator3P::hand_from_text("111m123p123s111z22z").expect("winning hand should parse");
    let conditions = Conditions {
        player_wind: Wind::South,
        round_wind: Wind::East,
        kita_count: 2,
        ..Default::default()
    };

    let result = eval.calc(28 * 4, &[27 * 4], &[27 * 4], Some(conditions));

    assert!(result.is_win);
    assert!(result.yaku[..result.yaku_count as usize].contains(&yaku::ID_DORA));
    assert!(result.yaku[..result.yaku_count as usize].contains(&yaku::ID_URADORA));
    assert!(result.yaku[..result.yaku_count as usize].contains(&yaku::ID_NUKIDORA));
    assert!(result.han >= 5, "expected yakuhai + dora + ura + nukidora");
}

#[test]
fn riichi_candidate_helper_returns_discard_that_leaves_tenpai() {
    let candidates = check_riichi_candidates_3p(vec![
        0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 110, 112, 116,
    ]);

    assert!(candidates.contains(&116));
    assert!(!candidates.is_empty());
}

#[test]
fn sanma_dora_mapping_wraps_manzu_and_cycles_honors() {
    assert_eq!(get_next_tile_sanma(0), 8);
    assert_eq!(get_next_tile_sanma(8), 0);
    assert_eq!(get_next_tile_sanma(4), 4);
    assert_eq!(get_next_tile_sanma(27), 28);
    assert_eq!(get_next_tile_sanma(30), 27);
    assert_eq!(get_next_tile_sanma(31), 32);
    assert_eq!(get_next_tile_sanma(33), 31);
}
