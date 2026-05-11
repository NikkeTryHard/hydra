use super::*;
use crate::agari::Mentsu;
use crate::types::MeldType;

fn hand_with_tiles(tiles: &[u8]) -> Hand {
    Hand::new(Some(tiles.to_vec()))
}

fn meld(meld_type: MeldType, tiles: &[u8], opened: bool) -> Meld {
    Meld::new(meld_type, tiles, opened, -1, None)
}

#[test]
fn apply_static_yaku_adds_all_enabled_flags() {
    let mut res = YakuResult::default();
    let ctx = YakuContext3P {
        is_menzen: true,
        is_reach: true,
        is_ippatsu: true,
        is_tsumo: true,
        is_haitei: true,
        is_houtei: false,
        is_rinshan: true,
        is_chankan: false,
        is_tsumo_first_turn: false,
        is_daburu_reach: false,
        dora_count: 2,
        aka_dora: 1,
        ura_dora_count: 3,
        nukidora_count: 2,
        round_wind: 27,
        seat_wind: 28,
    };

    apply_static_yaku(&mut res, &ctx);

    assert_eq!(res.han, 13);
    assert!(res.yaku_ids.contains(&ID_RIICHI));
    assert!(res.yaku_ids.contains(&ID_IPPATSU));
    assert!(res.yaku_ids.contains(&ID_TSUMO));
    assert!(res.yaku_ids.contains(&ID_HAITEI));
    assert!(res.yaku_ids.contains(&ID_RINSHAN));
    assert!(res.yaku_ids.contains(&ID_DORA));
    assert!(res.yaku_ids.contains(&ID_AKADORA));
    assert!(res.yaku_ids.contains(&ID_URADORA));
    assert!(res.yaku_ids.contains(&ID_NUKIDORA));
}

#[test]
fn apply_static_yaku_prefers_double_reach_and_non_tsumo_branches() {
    let mut res = YakuResult::default();
    let ctx = YakuContext3P {
        is_menzen: false,
        is_reach: true,
        is_ippatsu: false,
        is_tsumo: false,
        is_haitei: false,
        is_houtei: true,
        is_rinshan: false,
        is_chankan: true,
        is_tsumo_first_turn: false,
        is_daburu_reach: true,
        dora_count: 0,
        aka_dora: 0,
        ura_dora_count: 0,
        nukidora_count: 0,
        round_wind: 27,
        seat_wind: 27,
    };

    apply_static_yaku(&mut res, &ctx);

    assert_eq!(res.han, 4);
    assert!(res.yaku_ids.contains(&ID_DOUBLE_RIICHI));
    assert!(!res.yaku_ids.contains(&ID_RIICHI));
    assert!(res.yaku_ids.contains(&ID_HOUTEI));
    assert!(res.yaku_ids.contains(&ID_CHANKAN));
}

#[test]
fn calculate_fu_with_waiting_handles_tanki_and_rounding() {
    let div = Division {
        head: 27,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(9),
            Mentsu::Koutsu(31),
        ],
    };
    let ctx = YakuContext3P {
        is_menzen: true,
        is_tsumo: false,
        round_wind: 27,
        seat_wind: 28,
        ..Default::default()
    };

    let fu = calculate_fu_with_waiting(&div, &[], &ctx, None, 27);
    assert_eq!(fu, 50);
}

#[test]
fn check_pinfu_accepts_closed_ryanmen_and_rejects_yakuhai_pair() {
    let good_div = Division {
        head: 1,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(9),
            Mentsu::Shuntsu(18),
        ],
    };
    let good_ctx = YakuContext3P {
        is_menzen: true,
        round_wind: 27,
        seat_wind: 28,
        ..Default::default()
    };
    assert!(check_pinfu(&good_div, &[], &good_ctx, Some(0), 0));

    let bad_div = Division {
        head: 27,
        body: good_div.body.clone(),
    };
    assert!(!check_pinfu(&bad_div, &[], &good_ctx, Some(0), 2));
}

#[test]
fn suit_and_terminal_helpers_distinguish_flush_and_terminal_patterns() {
    let honitsu_hand = hand_with_tiles(&[0, 1, 2, 3, 4, 5, 27, 27, 28, 28, 29, 29, 31, 31]);
    let chinitsu_hand = hand_with_tiles(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 3]);
    let honroutou_hand = hand_with_tiles(&[0, 0, 8, 8, 9, 9, 17, 17, 18, 18, 26, 26, 27, 27]);
    let tanyao_hand = hand_with_tiles(&[1, 1, 2, 2, 10, 10, 11, 11, 19, 19, 20, 20, 21, 21]);

    assert!(is_honitsu(&honitsu_hand, &[]));
    assert!(!is_chinitsu(&honitsu_hand, &[]));
    assert!(is_chinitsu(&chinitsu_hand, &[]));
    assert!(is_honroutou(&honroutou_hand, &[]));
    assert!(is_tanyao(&tanyao_hand, &[]));
    assert!(!is_tanyao(&honroutou_hand, &[]));
}

#[test]
fn sequence_and_triplet_helpers_detect_cross_suit_patterns() {
    let doujun = Division {
        head: 31,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(9),
            Mentsu::Shuntsu(18),
            Mentsu::Shuntsu(3),
        ],
    };
    let doukou = Division {
        head: 31,
        body: vec![
            Mentsu::Koutsu(0),
            Mentsu::Koutsu(9),
            Mentsu::Koutsu(18),
            Mentsu::Shuntsu(3),
        ],
    };
    let ittsu = Division {
        head: 31,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(6),
            Mentsu::Koutsu(31),
        ],
    };

    assert!(is_sanshoku_doujun(&doujun, &[]));
    assert!(is_sanshoku_doukou(&doukou, &[]));
    assert!(check_ittsu(&ittsu, &[]));
}

#[test]
fn chuuren_helpers_distinguish_base_and_nine_wait_shapes() {
    let base = hand_with_tiles(&[0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 8]);
    let nine_wait = hand_with_tiles(&[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]);

    assert!(is_chuuren_poutou(&base));
    assert!(is_chuuren_9_wait(&base, 4));
    assert!(is_chuuren_poutou(&nine_wait));
    assert!(is_chuuren_9_wait(&nine_wait, 0));
}

#[test]
fn calculate_fu_with_waiting_counts_open_and_closed_meld_fu() {
    let div = Division {
        head: 1,
        body: vec![
            Mentsu::Koutsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(9),
            Mentsu::Shuntsu(18),
        ],
    };
    let melds = [Meld::new(
        MeldType::Ankan,
        &[31, 31, 31, 31],
        false,
        -1,
        None,
    )];
    let ctx = YakuContext3P {
        is_menzen: false,
        is_tsumo: true,
        round_wind: 27,
        seat_wind: 28,
        ..Default::default()
    };

    let fu = calculate_fu_with_waiting(&div, &melds, &ctx, Some(0), 0);
    assert_eq!(fu, 70);
}

#[test]
fn outside_and_yakuman_shape_helpers_detect_expected_patterns() {
    let tsuuiisou = hand_with_tiles(&[27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 31, 31, 31]);
    let chinroutou = hand_with_tiles(&[0, 0, 0, 8, 8, 8, 9, 9, 9, 17, 17, 17, 18, 18]);
    let ryuuiisou = hand_with_tiles(&[19, 19, 20, 20, 21, 21, 23, 23, 25, 25, 32, 32, 32, 32]);

    assert!(is_tsuu_iisou(&tsuuiisou, &[]));
    assert!(is_chinroutou(&chinroutou, &[]));
    assert!(is_ryuu_iisou(&ryuuiisou, &[]));
    assert!(!is_tsuu_iisou(&chinroutou, &[]));
    assert!(!is_chinroutou(&tsuuiisou, &[]));
    assert!(!is_ryuu_iisou(&chinroutou, &[]));
}

#[test]
fn junchan_and_chantai_distinguish_number_terminals_from_honor_mix() {
    let junchan_div = Division {
        head: 0,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(6),
            Mentsu::Shuntsu(9),
            Mentsu::Koutsu(26),
        ],
    };
    let chantai_div = Division {
        head: 27,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(6),
            Mentsu::Koutsu(8),
            Mentsu::Koutsu(31),
        ],
    };
    let bad_div = Division {
        head: 1,
        body: vec![
            Mentsu::Shuntsu(1),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(9),
            Mentsu::Koutsu(18),
        ],
    };

    assert!(is_junchan(&junchan_div, &[]));
    assert!(is_chantai(&chantai_div, &[]));
    assert!(!is_junchan(&chantai_div, &[]));
    assert!(!is_chantai(&bad_div, &[]));
}

#[test]
fn apply_yakuman_detects_tenhou_and_double_wind_yakuman() {
    let hand = hand_with_tiles(&[27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33]);
    let mut res = YakuResult::default();
    let div = Division {
        head: 27,
        body: vec![
            Mentsu::Koutsu(28),
            Mentsu::Koutsu(29),
            Mentsu::Koutsu(30),
            Mentsu::Koutsu(31),
        ],
    };
    let ctx = YakuContext3P {
        is_menzen: true,
        is_tsumo: true,
        is_tsumo_first_turn: true,
        seat_wind: 27,
        round_wind: 27,
        ..Default::default()
    };

    apply_yakuman(&mut res, &hand, &[], &ctx, &div, None, 27);

    assert!(res.yakuman_count >= 3);
    assert_eq!(res.han, 13 * res.yakuman_count);
    assert!(res.yaku_ids.contains(&ID_TENHO));
    assert!(res.yaku_ids.contains(&ID_SHOUSUUSHI));
    assert!(res.yaku_ids.contains(&ID_TSUISO));
}

#[test]
fn apply_yakuman_detects_suuankou_tanki_shape() {
    let hand = hand_with_tiles(&[31, 31, 32, 32, 32, 33, 33, 33, 0, 0, 0, 9, 9, 9]);
    let mut res = YakuResult::default();
    let div = Division {
        head: 31,
        body: vec![
            Mentsu::Koutsu(32),
            Mentsu::Koutsu(33),
            Mentsu::Koutsu(0),
            Mentsu::Koutsu(9),
        ],
    };
    let ctx = YakuContext3P {
        is_menzen: true,
        is_tsumo: false,
        ..Default::default()
    };

    apply_yakuman(&mut res, &hand, &[], &ctx, &div, None, 31);

    assert!(res.yakuman_count >= 2);
    assert_eq!(res.han, 13 * res.yakuman_count);
    assert!(res.yaku_ids.contains(&ID_SUANKO_TANKI));
    assert!(!res.yaku_ids.is_empty());
}

#[test]
fn apply_yakuman_detects_daisangen_and_daisuushii_variants() {
    let dragon_hand = hand_with_tiles(&[31, 31, 31, 32, 32, 32, 33, 33, 33, 0, 1, 2, 27, 27]);
    let dragon_div = Division {
        head: 27,
        body: vec![
            Mentsu::Koutsu(31),
            Mentsu::Koutsu(32),
            Mentsu::Koutsu(33),
            Mentsu::Shuntsu(0),
        ],
    };
    let mut dragon_res = YakuResult::default();
    apply_yakuman(
        &mut dragon_res,
        &dragon_hand,
        &[],
        &YakuContext3P::default(),
        &dragon_div,
        Some(3),
        2,
    );
    assert!(dragon_res.yaku_ids.contains(&ID_DAISANGEN));

    let wind_hand = hand_with_tiles(&[27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31]);
    let wind_div = Division {
        head: 31,
        body: vec![
            Mentsu::Koutsu(27),
            Mentsu::Koutsu(28),
            Mentsu::Koutsu(29),
            Mentsu::Koutsu(30),
        ],
    };
    let mut wind_res = YakuResult::default();
    apply_yakuman(
        &mut wind_res,
        &wind_hand,
        &[],
        &YakuContext3P::default(),
        &wind_div,
        Some(0),
        27,
    );
    assert!(wind_res.yaku_ids.contains(&ID_DAISUUSHI));
}

#[test]
fn calculate_yaku_3p_surfaces_daisangen_and_daisuushii_results() {
    let dragon_hand = hand_with_tiles(&[31, 31, 31, 32, 32, 32, 33, 33, 33, 0, 1, 2, 27, 27]);
    let dragon_res = calculate_yaku_3p(&dragon_hand, &[], &YakuContext3P::default(), 2);
    assert!(dragon_res.han >= 13);
    assert!(dragon_res.yaku_ids.contains(&ID_DAISANGEN));

    let wind_hand = hand_with_tiles(&[27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31]);
    let wind_res = calculate_yaku_3p(&wind_hand, &[], &YakuContext3P::default(), 31);
    assert!(wind_res.han >= 26);
    assert!(wind_res.yaku_ids.contains(&ID_DAISUUSHI));
}

#[test]
fn calculate_yaku_3p_distinguishes_kokushi_wait_variants() {
    let thirteen_wait = hand_with_tiles(&[0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 33]);
    let thirteen_wait_res = calculate_yaku_3p(&thirteen_wait, &[], &YakuContext3P::default(), 33);

    assert_eq!(thirteen_wait_res.han, 26);
    assert_eq!(thirteen_wait_res.yakuman_count, 2);
    assert!(thirteen_wait_res.yaku_ids.contains(&ID_KOKUSHI_13));

    let regular = hand_with_tiles(&[0, 0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]);
    let regular_res = calculate_yaku_3p(&regular, &[], &YakuContext3P::default(), 33);

    assert_eq!(regular_res.han, 13);
    assert_eq!(regular_res.yakuman_count, 1);
    assert!(regular_res.yaku_ids.contains(&ID_KOKUSHI));
}

#[test]
fn calculate_yaku_3p_scores_chiitoitsu_chinitsu_path() {
    let hand = hand_with_tiles(&[1, 1, 2, 2, 4, 4, 5, 5, 7, 7, 8, 8, 6, 6]);

    let res = calculate_yaku_3p(&hand, &[], &YakuContext3P::default(), 7);

    assert_eq!(res.han, 8);
    assert_eq!(res.fu, 25);
    assert!(res.yaku_ids.contains(&ID_CHITOITSU));
    assert!(res.yaku_ids.contains(&ID_CHINITSU));
    assert!(!res.yaku_ids.contains(&ID_HONITSU));
    assert!(!res.yaku_ids.contains(&ID_HONROUTO));
}

#[test]
fn calculate_yaku_3p_accumulates_standard_triplet_yaku_and_double_winds() {
    let hand = hand_with_tiles(&[0, 0, 0, 27, 27, 27, 31, 31, 31, 32, 32, 32, 33, 33]);
    let ctx = YakuContext3P {
        is_menzen: true,
        is_tsumo: false,
        round_wind: 27,
        seat_wind: 27,
        ..Default::default()
    };

    let res = calculate_yaku_3p(&hand, &[], &ctx, 0);

    assert_eq!(res.yakuman_count, 0);
    assert!(res.han >= 12);
    assert!(res.yaku_ids.contains(&ID_HAKU));
    assert!(res.yaku_ids.contains(&ID_HATSU));
    assert!(res.yaku_ids.contains(&ID_BAKAZE));
    assert!(res.yaku_ids.contains(&ID_JIKAZE));
    assert!(res.yaku_ids.contains(&ID_SHOSANGEN));
    assert!(res.yaku_ids.contains(&ID_TOITOI));
    assert!(res.yaku_ids.contains(&ID_SANANKOU));
    assert!(res.yaku_ids.contains(&ID_HONITSU));
    assert!(res.yaku_ids.contains(&ID_HONROUTO));
}

#[test]
fn check_pinfu_rejects_open_triplet_and_bad_wait_shapes() {
    let ctx = YakuContext3P {
        is_menzen: true,
        round_wind: 27,
        seat_wind: 28,
        ..Default::default()
    };
    let div = Division {
        head: 1,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(6),
            Mentsu::Shuntsu(18),
        ],
    };
    let triplet_div = Division {
        head: 1,
        body: vec![
            Mentsu::Koutsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(6),
            Mentsu::Shuntsu(18),
        ],
    };

    assert!(!check_pinfu(
        &div,
        &[],
        &YakuContext3P {
            is_menzen: false,
            ..ctx
        },
        Some(0),
        0,
    ));
    assert!(!check_pinfu(
        &div,
        &[meld(MeldType::Chi, &[0, 1, 2], true)],
        &ctx,
        Some(0),
        0,
    ));
    assert!(!check_pinfu(&triplet_div, &[], &ctx, Some(0), 0));
    assert!(!check_pinfu(&div, &[], &ctx, Some(0), 1));
    assert!(!check_pinfu(&div, &[], &ctx, Some(0), 2));
    assert!(!check_pinfu(&div, &[], &ctx, Some(2), 6));
}

#[test]
fn calculate_fu_with_waiting_distinguishes_ryanmen_kanchan_and_penchan() {
    let div = Division {
        head: 1,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(6),
            Mentsu::Shuntsu(18),
        ],
    };
    let ctx = YakuContext3P {
        is_menzen: true,
        is_tsumo: false,
        round_wind: 27,
        seat_wind: 28,
        ..Default::default()
    };

    assert_eq!(calculate_fu_with_waiting(&div, &[], &ctx, Some(1), 3), 30);
    assert_eq!(calculate_fu_with_waiting(&div, &[], &ctx, Some(1), 4), 40);
    assert_eq!(calculate_fu_with_waiting(&div, &[], &ctx, Some(0), 2), 40);
    assert_eq!(calculate_fu_with_waiting(&div, &[], &ctx, Some(2), 6), 40);
}

#[test]
fn calculate_fu_with_waiting_promotes_open_twenty_fu_to_thirty() {
    let div = Division {
        head: 1,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(9),
            Mentsu::Shuntsu(18),
        ],
    };
    let ctx = YakuContext3P {
        is_menzen: false,
        is_tsumo: false,
        round_wind: 27,
        seat_wind: 28,
        ..Default::default()
    };

    let fu = calculate_fu_with_waiting(&div, &[], &ctx, Some(1), 3);

    assert_eq!(fu, 30);
}

#[test]
fn calculate_fu_with_waiting_stacks_round_seat_and_dragon_pair_fu() {
    let div = Division {
        head: 31,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(9),
            Mentsu::Shuntsu(18),
        ],
    };
    let ctx = YakuContext3P {
        is_menzen: true,
        is_tsumo: false,
        round_wind: 31,
        seat_wind: 31,
        ..Default::default()
    };

    let fu = calculate_fu_with_waiting(&div, &[], &ctx, None, 31);

    assert_eq!(fu, 40);
}

#[test]
fn sequence_and_triplet_helpers_consider_meld_types() {
    let ittsu_div = Division {
        head: 31,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Koutsu(31),
            Mentsu::Koutsu(32),
        ],
    };
    let doujun_div = Division {
        head: 31,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(9),
            Mentsu::Koutsu(31),
            Mentsu::Koutsu(32),
        ],
    };
    let doukou_div = Division {
        head: 31,
        body: vec![
            Mentsu::Koutsu(0),
            Mentsu::Koutsu(9),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(12),
        ],
    };

    assert!(check_ittsu(
        &ittsu_div,
        &[meld(MeldType::Chi, &[6, 7, 8], true)]
    ));
    assert!(is_sanshoku_doujun(
        &doujun_div,
        &[meld(MeldType::Chi, &[18, 19, 20], true)],
    ));
    assert!(!is_sanshoku_doujun(
        &doujun_div,
        &[meld(MeldType::Pon, &[18, 18, 18], true)],
    ));
    assert!(is_sanshoku_doukou(
        &doukou_div,
        &[meld(MeldType::Pon, &[18, 18, 18], true)],
    ));
    assert!(!is_sanshoku_doukou(
        &doukou_div,
        &[meld(MeldType::Chi, &[18, 19, 20], true)],
    ));
}

#[test]
fn outside_helpers_consider_meld_terminal_and_honor_rules() {
    let junchan_div = Division {
        head: 0,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(6),
            Mentsu::Shuntsu(9),
            Mentsu::Koutsu(26),
        ],
    };

    assert!(is_junchan(
        &junchan_div,
        &[meld(MeldType::Pon, &[8, 8, 8], true)]
    ));
    assert!(!is_junchan(
        &junchan_div,
        &[meld(MeldType::Pon, &[31, 31, 31], true)]
    ));
    assert!(!is_chantai(&junchan_div, &[]));
    assert!(is_chantai(
        &junchan_div,
        &[meld(MeldType::Pon, &[31, 31, 31], true)],
    ));
}

#[test]
fn apply_yakuman_detects_chiihou_without_other_yakuman() {
    let hand = hand_with_tiles(&[0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 31, 31]);
    let div = Division {
        head: 31,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(9),
            Mentsu::Shuntsu(18),
        ],
    };
    let ctx = YakuContext3P {
        is_menzen: true,
        is_tsumo: true,
        is_tsumo_first_turn: true,
        seat_wind: 28,
        ..Default::default()
    };
    let mut res = YakuResult::default();

    apply_yakuman(&mut res, &hand, &[], &ctx, &div, Some(0), 2);

    assert_eq!(res.han, 13);
    assert_eq!(res.yakuman_count, 1);
    assert!(res.yaku_ids.contains(&ID_CHIHO));
}

#[test]
fn apply_yakuman_detects_suukantsu_from_four_kans() {
    let hand = hand_with_tiles(&[1, 1]);
    let melds = [
        meld(MeldType::Ankan, &[0, 0, 0, 0], false),
        meld(MeldType::Daiminkan, &[9, 9, 9, 9], true),
        meld(MeldType::Kakan, &[18, 18, 18, 18], true),
        meld(MeldType::Ankan, &[27, 27, 27, 27], false),
    ];
    let mut res = YakuResult::default();

    apply_yakuman(
        &mut res,
        &hand,
        &melds,
        &YakuContext3P::default(),
        &Division {
            head: 1,
            body: Vec::new(),
        },
        None,
        1,
    );

    assert_eq!(res.han, 13);
    assert_eq!(res.yakuman_count, 1);
    assert!(res.yaku_ids.contains(&ID_SUKANTSU));
}

#[test]
fn apply_yakuman_distinguishes_chuuren_and_junsei_variants() {
    let pure_hand = hand_with_tiles(&[0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]);
    let pure_div = Division {
        head: 1,
        body: vec![
            Mentsu::Koutsu(0),
            Mentsu::Shuntsu(2),
            Mentsu::Shuntsu(5),
            Mentsu::Koutsu(8),
        ],
    };
    let ctx = YakuContext3P {
        is_menzen: true,
        ..Default::default()
    };
    let mut pure_res = YakuResult::default();

    apply_yakuman(&mut pure_res, &pure_hand, &[], &ctx, &pure_div, Some(1), 1);

    assert_eq!(pure_res.han, 26);
    assert_eq!(pure_res.yakuman_count, 2);
    assert!(pure_res.yaku_ids.contains(&ID_JUNSEI_CHUUREN));

    let regular_hand = hand_with_tiles(&[0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8]);
    let mut regular_res = YakuResult::default();

    apply_yakuman(
        &mut regular_res,
        &regular_hand,
        &[],
        &ctx,
        &pure_div,
        Some(1),
        7,
    );

    assert_eq!(regular_res.han, 13);
    assert_eq!(regular_res.yakuman_count, 1);
    assert!(regular_res.yaku_ids.contains(&ID_CHUUREN));
}

#[test]
fn tile_classifier_helpers_distinguish_terminals_honors_and_yakuhai() {
    let ctx = YakuContext3P {
        round_wind: 28,
        seat_wind: 30,
        ..Default::default()
    };

    assert!(is_terminal(0));
    assert!(is_terminal(33));
    assert!(is_number_terminal(8));
    assert!(!is_number_terminal(27));
    assert!(is_honor(31));
    assert!(!is_honor(9));
    assert!(is_yakuhai_tile(31, &ctx));
    assert!(is_yakuhai_tile(28, &ctx));
    assert!(is_yakuhai_tile(30, &ctx));
    assert!(!is_yakuhai_tile(29, &ctx));
    assert!(!is_yakuhai_tile(8, &ctx));
}

#[test]
fn flush_and_outside_helpers_reject_mixed_inside_shapes() {
    let mixed_flush = hand_with_tiles(&[0, 1, 2, 9, 10, 11, 27, 27, 28, 28, 29, 29, 31, 31]);
    let inside_shape = Division {
        head: 1,
        body: vec![
            Mentsu::Shuntsu(1),
            Mentsu::Shuntsu(4),
            Mentsu::Shuntsu(10),
            Mentsu::Koutsu(18),
        ],
    };

    assert!(!is_honitsu(&mixed_flush, &[]));
    assert!(!is_chinitsu(&mixed_flush, &[]));
    assert!(!is_honroutou(&mixed_flush, &[]));
    assert!(!is_junchan(&inside_shape, &[]));
    assert!(!is_chantai(&inside_shape, &[]));
}

#[test]
fn calculate_yaku_3p_returns_empty_result_for_non_agari_shape() {
    let hand = hand_with_tiles(&[0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 18, 19, 27, 31]);
    let res = calculate_yaku_3p(&hand, &[], &YakuContext3P::default(), 31);

    assert_eq!(res.han, 0);
    assert_eq!(res.fu, 0);
    assert_eq!(res.yakuman_count, 0);
    assert!(res.yaku_names.is_empty() || res.yaku_ids.len() == res.yaku_names.len());
}

#[test]
fn yakuhai_and_terminal_helpers_reject_non_matching_tiles() {
    let ctx = YakuContext3P {
        round_wind: 27,
        seat_wind: 28,
        ..Default::default()
    };

    assert!(!is_terminal(1));
    assert!(!is_number_terminal(1));
    assert!(!is_honor(8));
    assert!(!is_yakuhai_tile(29, &ctx));
    assert!(!is_yakuhai_tile(0, &ctx));
}

#[test]
fn tsuuiisou_chinroutou_and_ryuuiisou_reject_mixed_hands() {
    let mixed = hand_with_tiles(&[0, 0, 8, 8, 27, 27, 31, 31, 19, 19, 20, 20, 25, 25]);
    assert!(!is_tsuu_iisou(&mixed, &[]));
    assert!(!is_chinroutou(&mixed, &[]));
    assert!(!is_ryuu_iisou(&mixed, &[]));
}

#[test]
fn chuuren_helpers_reject_honor_and_multi_suit_hands() {
    let honors = hand_with_tiles(&[27, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 33, 33]);
    let mixed = hand_with_tiles(&[0, 0, 0, 1, 2, 3, 9, 9, 9, 10, 11, 12, 18, 18]);

    assert!(!is_chuuren_poutou(&honors));
    assert!(!is_chuuren_poutou(&mixed));
    assert!(!is_chuuren_9_wait(&mixed, 0));
}

#[test]
fn apply_yakuman_leaves_result_empty_for_non_yakuman_hand() {
    let hand = hand_with_tiles(&[0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27]);
    let div = Division {
        head: 27,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(9),
            Mentsu::Shuntsu(18),
        ],
    };
    let mut res = YakuResult::default();

    apply_yakuman(
        &mut res,
        &hand,
        &[],
        &YakuContext3P::default(),
        &div,
        Some(0),
        2,
    );

    assert_eq!(res.han, 0);
    assert_eq!(res.yakuman_count, 0);
    assert!(res.yaku_names.is_empty() || res.yaku_ids.len() == res.yaku_names.len());
}
