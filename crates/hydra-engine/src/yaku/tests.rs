use super::*;
use crate::agari::Mentsu;
use crate::types::MeldType;

fn hand(tiles: &[u8]) -> Hand {
    Hand::new(Some(tiles.to_vec()))
}

#[test]
fn get_yaku_by_id_returns_expected_entry_and_none_for_unknown_id() {
    let riichi = get_yaku_by_id(ID_RIICHI).expect("riichi should exist in yaku table");
    assert_eq!(riichi.id, ID_RIICHI);
    assert_eq!(riichi.name, "立直");
    assert_eq!(riichi.name_en, "Riichi");
    assert_eq!(riichi.tenhou_id, 1);
    assert_eq!(riichi.mjsoul_id, 2);

    assert!(get_yaku_by_id(46).is_none());
    assert!(get_yaku_by_id(999).is_none());
}

#[test]
fn yaku_result_push_yaku_id_tracks_slice_length() {
    let mut result = YakuResult::default();

    result.push_yaku_id(ID_TANYAO);
    result.push_yaku_id(ID_PINFU);

    assert_eq!(result.yaku_id_count, 2);
    assert_eq!(result.yaku_ids_slice(), &[ID_TANYAO, ID_PINFU]);
}

#[test]
fn calculate_yaku_detects_kokushi_and_kokushi_thirteen_wait() {
    let standard_kokushi = hand(&[0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 33]);
    let standard_result = calculate_yaku(&standard_kokushi, &[], &YakuContext::default(), 0);
    assert_eq!(standard_result.han, 13);
    assert_eq!(standard_result.yakuman_count, 1);
    assert_eq!(standard_result.yaku_ids_slice(), &[ID_KOKUSHI]);
    assert_eq!(standard_result.yaku_names, vec!["Kokushi Musou"]);

    let thirteen_wait = hand(&[0, 0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]);
    let thirteen_wait_result = calculate_yaku(&thirteen_wait, &[], &YakuContext::default(), 0);
    assert_eq!(thirteen_wait_result.han, 26);
    assert_eq!(thirteen_wait_result.yakuman_count, 2);
    assert_eq!(thirteen_wait_result.yaku_ids_slice(), &[ID_KOKUSHI_13]);
    assert_eq!(
        thirteen_wait_result.yaku_names,
        vec!["Kokushi Musou 13-wait"]
    );
}

#[test]
fn calculate_yaku_for_chiitoitsu_applies_compound_hand_and_static_yaku() {
    let chiitoi_honitsu = hand(&[0, 0, 8, 8, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31]);
    let ctx = YakuContext {
        is_reach: true,
        is_ippatsu: true,
        dora_count: 2,
        aka_dora: 1,
        ura_dora_count: 1,
        ..Default::default()
    };

    let result = calculate_yaku(&chiitoi_honitsu, &[], &ctx, 31);

    assert_eq!(result.fu, 25);
    assert_eq!(result.han, 13);
    assert_eq!(
        result.yaku_ids_slice(),
        &[
            ID_CHITOITSU,
            ID_HONITSU,
            ID_HONROUTO,
            ID_RIICHI,
            ID_IPPATSU,
            ID_DORA,
            ID_AKADORA,
            ID_URADORA,
        ]
    );
    assert!(result.yaku_names.iter().any(|name| name == "Chiitoitsu"));
    assert!(result.yaku_names.iter().any(|name| name == "Honitsu"));
    assert!(result.yaku_names.iter().any(|name| name == "Honroutou"));
}

#[test]
fn calculate_fu_with_waiting_covers_wait_and_meld_fu_cases() {
    let closed_tanki = Division {
        head: 31,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(6),
            Mentsu::Koutsu(27),
        ],
    };
    let closed_ctx = YakuContext {
        is_menzen: true,
        is_tsumo: false,
        round_wind: 31,
        seat_wind: 27,
        ..Default::default()
    };
    assert_eq!(
        calculate_fu_with_waiting(&closed_tanki, &[], &closed_ctx, None, 31),
        50
    );

    let kanchan = Division {
        head: 1,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(6),
            Mentsu::Shuntsu(9),
        ],
    };
    assert_eq!(
        calculate_fu_with_waiting(&kanchan, &[], &YakuContext::default(), Some(3), 10),
        40
    );

    let open_hand = Division {
        head: 1,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(6),
            Mentsu::Shuntsu(9),
        ],
    };
    let open_melds = [
        Meld::new(MeldType::Pon, &[27, 27, 27], true, 1, Some(27)),
        Meld::new(MeldType::Ankan, &[0, 0, 0, 0], false, -1, None),
    ];
    let open_ctx = YakuContext {
        is_menzen: false,
        is_tsumo: false,
        ..Default::default()
    };
    assert_eq!(
        calculate_fu_with_waiting(&open_hand, &open_melds, &open_ctx, Some(0), 0),
        60
    );
}

#[test]
fn helper_detectors_cover_terminal_flush_and_sequence_patterns() {
    let terminal_hand = hand(&[0, 0, 8, 8, 9, 9, 17, 17, 18, 18, 26, 26, 27, 27]);
    assert!(is_honroutou(&terminal_hand, &[]));
    assert!(!is_chinitsu(&terminal_hand, &[]));
    assert!(!is_honitsu(&terminal_hand, &[]));

    let half_flush = hand(&[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 27, 31, 31]);
    assert!(is_honitsu(&half_flush, &[]));
    assert!(!is_chinitsu(&half_flush, &[]));

    let full_flush = hand(&[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]);
    assert!(is_chinitsu(&full_flush, &[]));
    assert!(!is_honitsu(&full_flush, &[]));

    let mixed = hand(&[0, 0, 1, 1, 2, 2, 9, 9, 10, 10, 11, 11, 27, 27]);
    assert!(!is_chinitsu(&mixed, &[]));
    assert!(!is_honitsu(&mixed, &[]));

    let junchan = Division {
        head: 8,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(6),
            Mentsu::Shuntsu(9),
            Mentsu::Shuntsu(15),
        ],
    };
    assert!(is_junchan(&junchan, &[]));
    assert!(!is_chantai(&junchan, &[]));

    let chantai = Division {
        head: 27,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(6),
            Mentsu::Koutsu(31),
            Mentsu::Koutsu(8),
        ],
    };
    assert!(is_chantai(&chantai, &[]));
    assert!(!is_junchan(&chantai, &[]));

    let pattern_div = Division {
        head: 27,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(6),
            Mentsu::Koutsu(31),
        ],
    };
    assert!(check_ittsu(&pattern_div, &[]));

    let sanshoku_doujun = Division {
        head: 27,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(9),
            Mentsu::Shuntsu(18),
            Mentsu::Koutsu(31),
        ],
    };
    assert!(is_sanshoku_doujun(&sanshoku_doujun, &[]));

    let sanshoku_doukou = Division {
        head: 27,
        body: vec![
            Mentsu::Koutsu(0),
            Mentsu::Koutsu(9),
            Mentsu::Koutsu(18),
            Mentsu::Shuntsu(3),
        ],
    };
    assert!(is_sanshoku_doukou(&sanshoku_doukou, &[]));
}

#[test]
fn pinfu_and_yakuhai_helpers_distinguish_waits_and_value_pairs() {
    let ryanmen = Division {
        head: 1,
        body: vec![
            Mentsu::Shuntsu(0),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(6),
            Mentsu::Shuntsu(9),
        ],
    };
    assert!(check_pinfu(
        &ryanmen,
        &[],
        &YakuContext::default(),
        Some(0),
        0,
    ));
    assert!(!check_pinfu(
        &ryanmen,
        &[],
        &YakuContext::default(),
        Some(0),
        2,
    ));

    let with_triplet = Division {
        head: 1,
        body: vec![
            Mentsu::Koutsu(27),
            Mentsu::Shuntsu(3),
            Mentsu::Shuntsu(6),
            Mentsu::Shuntsu(9),
        ],
    };
    assert!(!check_pinfu(
        &with_triplet,
        &[],
        &YakuContext::default(),
        Some(1),
        3,
    ));

    let yakuhai_ctx = YakuContext {
        round_wind: 27,
        seat_wind: 28,
        ..Default::default()
    };
    assert!(is_yakuhai_tile(31, &yakuhai_ctx));
    assert!(is_yakuhai_tile(27, &yakuhai_ctx));
    assert!(is_yakuhai_tile(28, &yakuhai_ctx));
    assert!(!is_yakuhai_tile(1, &yakuhai_ctx));
}

#[test]
fn yakuman_helpers_detect_special_hands_and_nine_gates_wait_shape() {
    let all_honors = hand(&[27, 27, 27, 28, 28, 28, 29, 29, 29, 31, 31, 31, 33, 33]);
    assert!(is_tsuu_iisou(&all_honors, &[]));

    let all_terminals = hand(&[0, 0, 0, 8, 8, 8, 9, 9, 9, 17, 17, 17, 18, 18]);
    assert!(is_chinroutou(&all_terminals, &[]));

    let all_green = hand(&[19, 19, 19, 20, 20, 20, 21, 21, 21, 23, 23, 23, 32, 32]);
    assert!(is_ryuu_iisou(&all_green, &[]));

    let chuuren = hand(&[0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 8]);
    assert!(is_chuuren_poutou(&chuuren));
    assert!(is_chuuren_9_wait(&chuuren, 4));
    assert!(!is_chuuren_9_wait(&chuuren, 0));

    let not_chuuren = hand(&[0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8, 8, 27]);
    assert!(!is_chuuren_poutou(&not_chuuren));
    assert!(!is_chuuren_9_wait(&chuuren, 31));
}

#[test]
fn apply_static_and_yakuman_helpers_stack_expected_ids_and_counts() {
    let mut static_result = YakuResult::default();
    let static_ctx = YakuContext {
        is_menzen: true,
        is_reach: true,
        is_ippatsu: true,
        is_tsumo: true,
        is_haitei: true,
        is_rinshan: true,
        dora_count: 2,
        aka_dora: 1,
        ura_dora_count: 1,
        ..Default::default()
    };
    apply_static_yaku(&mut static_result, &static_ctx);
    assert_eq!(static_result.han, 9);
    assert_eq!(
        static_result.yaku_ids_slice(),
        &[
            ID_RIICHI, ID_IPPATSU, ID_TSUMO, ID_HAITEI, ID_RINSHAN, ID_DORA, ID_AKADORA,
            ID_URADORA,
        ]
    );

    let mut yakuman_result = YakuResult::default();
    let yakuman_hand = hand(&[27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31]);
    let yakuman_div = Division {
        head: 31,
        body: vec![
            Mentsu::Koutsu(27),
            Mentsu::Koutsu(28),
            Mentsu::Koutsu(29),
            Mentsu::Koutsu(30),
        ],
    };
    let yakuman_ctx = YakuContext {
        is_menzen: true,
        is_tsumo: true,
        is_tsumo_first_turn: true,
        seat_wind: 27,
        ..Default::default()
    };
    apply_yakuman(
        &mut yakuman_result,
        &yakuman_hand,
        &[],
        &yakuman_ctx,
        &yakuman_div,
        None,
        31,
    );

    assert_eq!(yakuman_result.han, 78);
    assert_eq!(yakuman_result.yakuman_count, 6);
    assert!(yakuman_result.yaku_ids_slice().contains(&ID_TSUISO));
    assert!(yakuman_result.yaku_ids_slice().contains(&ID_DAISUUSHI));
    assert!(yakuman_result.yaku_ids_slice().contains(&ID_TENHO));
    assert!(yakuman_result.yaku_ids_slice().contains(&ID_SUANKO_TANKI));
}
