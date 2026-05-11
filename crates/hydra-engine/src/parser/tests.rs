use super::*;
use crate::types::MeldType;

#[test]
fn parse_tile_supports_basic_tiles_and_red_fives() {
    assert_eq!(parse_tile("1m").expect("1m should parse"), 0);
    assert_eq!(parse_tile("5m").expect("5m should parse"), 17);
    assert_eq!(parse_tile("0m").expect("red 5m should parse"), 16);
    assert_eq!(parse_tile("7z").expect("honor should parse"), 132);
}

#[test]
fn parse_tile_rejects_empty_multi_tile_and_meld_inputs() {
    assert!(parse_tile("").is_err());
    assert!(parse_tile("12m").is_err());
    assert!(parse_tile("(p5m1)").is_err());
}

#[test]
fn parse_hand_internal_parses_plain_tiles_and_chi_meld() {
    let (tiles, melds) = parse_hand_internal("123m(p5m1)").expect("hand should parse");

    assert_eq!(tiles.len(), 3);
    assert_eq!(melds.len(), 1);
    assert_eq!(melds[0].meld_type, MeldType::Pon);
    assert_eq!(melds[0].tile_count, 3);
    assert!(melds[0].opened);
}

#[test]
fn parse_hand_internal_rejects_pending_digits_and_bad_meld_suit() {
    let err = parse_hand_internal("123").expect_err("pending digits should fail");
    match err {
        crate::errors::RiichiError::Parse { message, .. } => {
            assert!(message.contains("Pending digits without suit"));
        }
        other => panic!("unexpected error: {other:?}"),
    }

    let err = parse_hand_internal("(123x)").expect_err("invalid meld suit should fail");
    match err {
        crate::errors::RiichiError::Parse { message, .. } => {
            assert!(message.contains("Invalid suit in meld"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn tid_to_mjai_and_mjai_to_tid_cover_reds_honors_and_regular_tiles() {
    assert_eq!(tid_to_mjai(16), "5mr");
    assert_eq!(tid_to_mjai(52), "5pr");
    assert_eq!(tid_to_mjai(88), "5sr");
    assert_eq!(tid_to_mjai(0), "1m");
    assert_eq!(tid_to_mjai(108), "E");

    assert_eq!(mjai_to_tid("5mr"), Some(16));
    assert_eq!(mjai_to_tid("5pr"), Some(52));
    assert_eq!(mjai_to_tid("5sr"), Some(88));
    assert_eq!(mjai_to_tid("1m"), Some(0));
    assert_eq!(mjai_to_tid("5m"), Some(17));
    assert_eq!(mjai_to_tid("E"), Some(108));
    assert_eq!(mjai_to_tid("0x"), None);
    assert_eq!(mjai_to_tid(""), None);
}

#[test]
fn parse_hand_returns_u32_tiles_matching_internal_parser() {
    let (tiles_u8, melds_internal) = parse_hand_internal("406p").expect("internal parse");
    let (tiles_u32, melds) = parse_hand("406p").expect("public parse");

    assert_eq!(
        tiles_u32,
        tiles_u8.iter().map(|&t| t as u32).collect::<Vec<_>>()
    );
    assert_eq!(melds.len(), melds_internal.len());
}

#[test]
fn parse_hand_internal_supports_ankan_and_kakan_meld_kinds() {
    let (_, ankan_melds) = parse_hand_internal("(k5m)").expect("ankan should parse");
    assert_eq!(ankan_melds.len(), 1);
    assert_eq!(ankan_melds[0].meld_type, MeldType::Ankan);
    assert!(!ankan_melds[0].opened);
    assert_eq!(ankan_melds[0].tile_count, 4);

    let (_, kakan_melds) = parse_hand_internal("(s5p)").expect("kakan should parse");
    assert_eq!(kakan_melds.len(), 1);
    assert_eq!(kakan_melds[0].meld_type, MeldType::Kakan);
    assert!(kakan_melds[0].opened);
    assert_eq!(kakan_melds[0].tile_count, 4);
}

#[test]
fn parse_hand_internal_rejects_bad_chi_and_tile_overflow() {
    let err = parse_hand_internal("(1234m)").expect_err("chi meld requires exactly 3 digits");
    match err {
        crate::errors::RiichiError::Parse { message, .. } => {
            assert!(message.contains("Chi meld requires 3 digits"));
        }
        other => panic!("unexpected error: {other:?}"),
    }

    let err = parse_hand_internal("55555m").expect_err("fifth copy of 5m should fail");
    match err {
        crate::errors::RiichiError::Parse { message, .. } => {
            assert!(message.contains("No more copies of tile 4"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn parse_hand_internal_rejects_melds_without_enough_copies() {
    let err = parse_hand_internal("(k5m)5m").expect_err("no fifth 5m copy should exist");
    match err {
        crate::errors::RiichiError::Parse { message, .. } => {
            assert!(
                message.contains("No more copies of tile 4")
                    || message.contains("Not enough tiles for meld of 4")
            );
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
