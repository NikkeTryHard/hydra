use super::TileConverter;

#[test]
fn parse_tile_variants_preserve_suit_and_red_flags() {
    assert_eq!(TileConverter::parse_tile("1m"), (0, false));
    assert_eq!(TileConverter::parse_tile_34("0m"), (4, true));
    assert_eq!(TileConverter::parse_tile_34("5p"), (13, false));
    assert_eq!(TileConverter::parse_tile_34("7z"), (33, false));
    assert_eq!(TileConverter::parse_tile(""), (0, false));
}

#[test]
fn parse_tile_136_handles_red_fives_and_normal_fives() {
    assert_eq!(TileConverter::parse_tile_136("0m"), 16);
    assert_eq!(TileConverter::parse_tile_136("0p"), 52);
    assert_eq!(TileConverter::parse_tile_136("0s"), 88);

    assert_eq!(TileConverter::parse_tile_136("5m"), 17);
    assert_eq!(TileConverter::parse_tile_136("5p"), 53);
    assert_eq!(TileConverter::parse_tile_136("5s"), 89);
    assert_eq!(TileConverter::parse_tile_136("1z"), 108);
}

#[test]
fn to_string_roundtrips_regular_and_red_tiles() {
    assert_eq!(TileConverter::to_string(0), "1m");
    assert_eq!(TileConverter::to_string(16), "0m");
    assert_eq!(TileConverter::to_string(52), "0p");
    assert_eq!(TileConverter::to_string(88), "0s");
    assert_eq!(TileConverter::to_string(108), "1z");
    assert_eq!(TileConverter::to_string(255), "?");
}

#[test]
fn match_and_remove_prefers_exact_match_then_same_tile_class() {
    let mut hand = vec![16, 17, 53, 108];
    assert!(TileConverter::match_and_remove_u8(&mut hand, 16));
    assert_eq!(hand, vec![17, 53, 108]);

    assert!(TileConverter::match_and_remove_u8(&mut hand, 52));
    assert_eq!(hand, vec![17, 108]);

    assert!(!TileConverter::match_and_remove_u8(&mut hand, 88));
    assert_eq!(hand, vec![17, 108]);
}
