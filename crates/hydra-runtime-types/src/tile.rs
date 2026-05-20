//! Tile representation and suit permutation for data augmentation.
//!
//! Provides the 34-tile type system, aka-dora handling, 136-format conversion,
//! and suit permutation (6 permutations of manzu/pinzu/souzu) used to 6x
//! training data without changing game semantics.

use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Total number of distinct tile types (0-33).
pub const NUM_TILE_TYPES: usize = 34;

/// Number of tiles per suited category (1-9).
pub const NUM_SUIT_TILES: usize = 9;

/// Total physical tiles in a standard mahjong set.
pub const NUM_TILES_136: usize = 136;

/// First manzu tile type index.
pub const MANZU_START: u8 = 0;
/// First pinzu tile type index.
pub const PINZU_START: u8 = 9;
/// First souzu tile type index.
pub const SOUZU_START: u8 = 18;
/// First honor (jihai) tile type index.
pub const JIHAI_START: u8 = 27;

pub(crate) const FIVE_MANZU: u8 = MANZU_START + 4;
pub(crate) const FIVE_PINZU: u8 = PINZU_START + 4;
pub(crate) const FIVE_SOUZU: u8 = SOUZU_START + 4;

/// East wind tile type index.
pub const EAST: u8 = 27;
/// South wind tile type index.
pub const SOUTH: u8 = 28;
/// West wind tile type index.
pub const WEST: u8 = 29;
/// North wind tile type index.
pub const NORTH: u8 = 30;
/// White dragon tile type index.
pub const HAKU: u8 = 31;
/// Green dragon tile type index.
pub const HATSU: u8 = 32;
/// Red dragon tile type index.
pub const CHUN: u8 = 33;

// ---------------------------------------------------------------------------
// Aka-dora constants (136-format indices for red fives)
// ---------------------------------------------------------------------------

/// Red 5m in 136-format. The 0th copy of tile type `FIVE_MANZU` is red.
pub const AKA_MANZU_136: u8 = FIVE_MANZU * 4;
/// Red 5p in 136-format. The 0th copy of tile type `FIVE_PINZU` is red.
pub const AKA_PINZU_136: u8 = FIVE_PINZU * 4;
/// Red 5s in 136-format. The 0th copy of tile type `FIVE_SOUZU` is red.
pub const AKA_SOUZU_136: u8 = FIVE_SOUZU * 4;

/// Extended tile type indices for aka-dora (used in action encoding).
pub const AKA_MANZU_TYPE: u8 = 34;
/// Extended tile type index for red five of pinzu (aka 5p).
pub const AKA_PINZU_TYPE: u8 = 35;
/// Extended tile type index for red five of souzu (aka 5s).
pub const AKA_SOUZU_TYPE: u8 = 36;

// ---------------------------------------------------------------------------
// Suit
// ---------------------------------------------------------------------------

/// The four tile categories in Riichi Mahjong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Suit {
    /// Characters suit.
    Manzu = 0,
    /// Dots/circles suit.
    Pinzu = 1,
    /// Bamboo suit.
    Souzu = 2,
    /// Honor tiles.
    Jihai = 3,
}

impl Suit {
    /// Returns the starting tile type index for this suit.
    #[inline]
    pub const fn start(self) -> u8 {
        match self {
            Suit::Manzu => MANZU_START,
            Suit::Pinzu => PINZU_START,
            Suit::Souzu => SOUZU_START,
            Suit::Jihai => JIHAI_START,
        }
    }
}

// ---------------------------------------------------------------------------
// TileType newtype
// ---------------------------------------------------------------------------

/// A tile type in the range 0-33. Wraps a `u8` for type safety.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileType(u8);

impl TileType {
    /// Creates a `TileType` if `id` is in range 0..34.
    #[inline]
    pub const fn new(id: u8) -> Option<Self> {
        if id < NUM_TILE_TYPES as u8 {
            Some(TileType(id))
        } else {
            None
        }
    }

    /// Raw numeric id (0-33).
    #[inline]
    pub const fn id(self) -> u8 {
        self.0
    }

    /// Which suit this tile belongs to.
    #[inline]
    pub const fn suit(self) -> Suit {
        match self.0 {
            0..9 => Suit::Manzu,
            9..18 => Suit::Pinzu,
            18..27 => Suit::Souzu,
            _ => Suit::Jihai,
        }
    }

    /// 1-based number within the suit (1-9), or `None` for honor tiles.
    #[inline]
    pub const fn number(self) -> Option<u8> {
        if self.0 < JIHAI_START {
            Some((self.0 % NUM_SUIT_TILES as u8) + 1)
        } else {
            None
        }
    }

    /// True for 1 or 9 of any suit.
    #[inline]
    pub const fn is_terminal(self) -> bool {
        if self.0 >= JIHAI_START {
            return false;
        }
        let num = self.0 % NUM_SUIT_TILES as u8;
        num == 0 || num == 8
    }

    /// True for wind or dragon tiles (indices 27-33).
    #[inline]
    pub const fn is_honor(self) -> bool {
        self.0 >= JIHAI_START
    }

    /// True for terminals or honors (yaochuuhai).
    #[inline]
    pub const fn is_terminal_or_honor(self) -> bool {
        self.is_terminal() || self.is_honor()
    }

    /// True for manzu, pinzu, or souzu (not jihai).
    #[inline]
    pub const fn is_suited(self) -> bool {
        self.0 < JIHAI_START
    }
}

impl fmt::Debug for TileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TileType({}={})", self.0, tile_type_to_mjai(self.0))
    }
}

impl fmt::Display for TileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(tile_type_to_mjai(self.0))
    }
}

// ---------------------------------------------------------------------------
// 136-format conversion and aka-dora
// ---------------------------------------------------------------------------

/// Converts a 136-format tile id (0-135) to its tile type (0-33).
#[inline]
pub const fn tile136_to_type(tile136: u8) -> TileType {
    // Each tile type has 4 copies: type = tile136 / 4
    TileType(tile136 / 4)
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

// ---------------------------------------------------------------------------
// Suit permutation
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Display / debug helpers
// ---------------------------------------------------------------------------

/// MJAI-style string names for tile types.
const TILE_NAMES: [&str; NUM_TILE_TYPES] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p",
    "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N", "P", "F",
    "C",
];

/// Returns the MJAI-style name for a tile type (0-33).
/// Out-of-range values return "??".
#[inline]
pub fn tile_type_to_mjai(tile_type: u8) -> &'static str {
    TILE_NAMES.get(tile_type as usize).copied().unwrap_or("??")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
