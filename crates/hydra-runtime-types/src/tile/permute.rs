use super::constants::*;
use super::format136::{deaka, is_aka_type, re_akaize};

/// All 6 permutations of the 3 suits [manzu, pinzu, souzu].
/// Each entry maps [manzu_target, pinzu_target, souzu_target].
pub const ALL_PERMUTATIONS: [[u8; 3]; 6] = [
    [0, 1, 2], // identity
    [0, 2, 1], // swap pin-sou
    [1, 0, 2], // swap man-pin
    [1, 2, 0], // rotate right
    [2, 0, 1], // rotate left
    [2, 1, 0], // swap man-sou
];

/// Permutes a tile type (0-33) according to the given suit permutation.
/// Honor tiles pass through unchanged.
///
/// `perm[i]` = which output suit original suit `i` maps to.
#[inline]
pub const fn permute_tile_type(tile_type: u8, perm: &[u8; 3]) -> u8 {
    if tile_type >= JIHAI_START {
        return tile_type;
    }
    let suit = tile_type / NUM_SUIT_TILES as u8;
    let num = tile_type % NUM_SUIT_TILES as u8;
    perm[suit as usize] * NUM_SUIT_TILES as u8 + num
}

/// Permutes a 136-format tile, preserving its copy index (and thus aka status).
/// Honor tiles pass through unchanged.
#[inline]
pub const fn permute_tile136(tile136: u8, perm: &[u8; 3]) -> u8 {
    let tile_type = tile136 / 4;
    if tile_type >= JIHAI_START {
        return tile136;
    }
    let copy = tile136 % 4;
    let new_type = permute_tile_type(tile_type, perm);
    new_type * 4 + copy
}

/// Permutes an extended tile type (0-36, where 34-36 are aka).
/// Strips aka, permutes the base, then re-applies aka if needed.
#[inline]
pub const fn permute_tile_extended(tile: u8, perm: &[u8; 3]) -> u8 {
    let aka = is_aka_type(tile);
    let base = deaka(tile);
    let permuted = permute_tile_type(base, perm);
    re_akaize(permuted, aka)
}
