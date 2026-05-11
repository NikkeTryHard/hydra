use super::*;
use crate::types::{Conditions, Meld, MeldType, Wind};

fn active_yaku_ids(result: &WinResult) -> &[u32] {
    &result.yaku[..result.yaku_count as usize]
}

#[test]
fn hand_from_text_and_wait_helpers_work_for_simple_tenpai_hand() {
    let eval = HandEvaluator::hand_from_text("123m123p123s111z2z")
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
    let eval = HandEvaluator::new(&[0, 4, 8], &[]);

    assert!(!eval.is_tenpai());
    assert!(eval.get_waits_u8().is_empty());
    assert!(eval.get_waits().is_empty());

    let mut buf = [0u8; 34];
    assert_eq!(eval.get_waits_u8_into(&mut buf), 0);
}

#[test]
fn new_sorts_chi_meld_and_counts_aka_tiles_from_hand_and_meld() {
    let meld = Meld::new(MeldType::Chi, &[20, 12, 16], true, 3, Some(16));
    let eval = HandEvaluator::new(&[0, 4, 52], &[meld]);

    assert_eq!(eval.meld_count, 1);
    assert_eq!(eval.aka_dora_count, 2);
    assert_eq!(eval.melds_slice()[0].tiles_slice(), &[3, 4, 5]);
    assert!(eval.melds_slice()[0].opened);
    assert_eq!(eval.hand.counts[0], 1);
    assert_eq!(eval.hand.counts[1], 1);
    assert_eq!(eval.hand.counts[13], 1);
}

#[test]
fn calc_returns_non_win_for_incomplete_hand() {
    let eval = HandEvaluator::new(&[0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 112, 116, 120], &[]);
    let result = eval.calc(124, &[], &[], None);

    assert!(!result.is_win);
    assert!(!result.has_win_shape);
    assert_eq!(result.han, 0);
    assert_eq!(result.fu, 0);
}

#[test]
fn calc_rejects_dora_only_hand_even_when_shape_is_complete() {
    let eval =
        HandEvaluator::hand_from_text("123m456m789p123s5z").expect("tenpai hand should parse");
    let result = eval.calc(124, &[132], &[], Some(Conditions::default()));

    assert!(!result.is_win);
    assert!(result.has_win_shape);
    assert_eq!(
        result.han, 2,
        "pair wait picks up two dora on white dragons only"
    );
    assert_eq!(result.fu, 40);
    assert!(active_yaku_ids(&result).contains(&yaku::ID_DORA));
    assert!(!active_yaku_ids(&result).contains(&yaku::ID_AKADORA));
    assert!(!active_yaku_ids(&result).contains(&yaku::ID_URADORA));
}

#[test]
fn calc_counts_dora_ura_and_aka_on_red_tsumo_scoring_path() {
    let eval =
        HandEvaluator::hand_from_text("123m456m789m234p5s").expect("winning hand should parse");
    let conditions = Conditions {
        tsumo: true,
        riichi: true,
        honba: 1,
        player_wind: Wind::South,
        round_wind: Wind::East,
        ..Default::default()
    };

    let result = eval.calc(88, &[84], &[84], Some(conditions));

    assert!(result.is_win);
    assert!(!result.yakuman);
    assert_eq!(result.han, 9);
    assert_eq!(result.fu, 30);
    assert_eq!(result.tsumo_agari_oya, 8100);
    assert_eq!(result.tsumo_agari_ko, 4100);
    assert_eq!(result.ron_agari, 0);
    assert!(active_yaku_ids(&result).contains(&yaku::ID_RIICHI));
    assert!(active_yaku_ids(&result).contains(&yaku::ID_TSUMO));
    assert!(active_yaku_ids(&result).contains(&yaku::ID_DORA));
    assert!(active_yaku_ids(&result).contains(&yaku::ID_AKADORA));
    assert!(active_yaku_ids(&result).contains(&yaku::ID_URADORA));
    assert!(active_yaku_ids(&result).contains(&yaku::ID_ITTSU));
}

#[test]
fn riichi_candidate_helper_returns_discard_that_leaves_tenpai() {
    let candidates = check_riichi_candidates(vec![
        0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 110, 112, 116,
    ]);

    assert!(candidates.contains(&116));
    assert!(!candidates.is_empty());
}

#[test]
fn get_next_tile_wraps_each_suit_and_honor_group() {
    assert_eq!(get_next_tile(0), 1);
    assert_eq!(get_next_tile(8), 0);
    assert_eq!(get_next_tile(9), 10);
    assert_eq!(get_next_tile(17), 9);
    assert_eq!(get_next_tile(18), 19);
    assert_eq!(get_next_tile(26), 18);
    assert_eq!(get_next_tile(27), 28);
    assert_eq!(get_next_tile(30), 27);
    assert_eq!(get_next_tile(31), 32);
    assert_eq!(get_next_tile(33), 31);
}
