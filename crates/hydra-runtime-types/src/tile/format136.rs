use super::constants::*;
use super::kind::TileType;

/// Converts a 136-format tile id (0-135) to its tile type (0-33).
#[inline]
pub const fn tile136_to_type(tile136: u8) -> TileType {
    // Each tile type has 4 copies: type = tile136 / 4
    TileType::new(tile136 / 4).unwrap()
}

/// Returns `true` if the 136-format tile is a red five (aka-dora).
///
/// Convention: index 16 = red 5m, 52 = red 5p, 88 = red 5s.
#[inline]
pub const fn tile136_is_aka(tile136: u8) -> bool {
    matches!(tile136, AKA_MANZU_136 | AKA_PINZU_136 | AKA_SOUZU_136)
}

/// Strips the aka flag from an extended tile type index (34-36 -> base type).
/// Normal tile types (0-33) pass through unchanged.
#[inline]
pub const fn deaka(tile: u8) -> u8 {
    match tile {
        AKA_MANZU_TYPE => FIVE_MANZU,
        AKA_PINZU_TYPE => FIVE_PINZU,
        AKA_SOUZU_TYPE => FIVE_SOUZU,
        other => other,
    }
}

/// If `tile` is the base type for a 5 in the given suit, returns the aka
/// extended index (34/35/36). Otherwise returns the tile unchanged.
#[inline]
pub const fn re_akaize(tile: u8, was_aka: bool) -> u8 {
    if !was_aka {
        return tile;
    }
    match tile {
        FIVE_MANZU => AKA_MANZU_TYPE,
        FIVE_PINZU => AKA_PINZU_TYPE,
        FIVE_SOUZU => AKA_SOUZU_TYPE,
        other => other,
    }
}

/// Returns `true` if the extended tile type (0-36) represents an aka-dora.
#[inline]
pub const fn is_aka_type(tile: u8) -> bool {
    matches!(tile, AKA_MANZU_TYPE | AKA_PINZU_TYPE | AKA_SOUZU_TYPE)
}
