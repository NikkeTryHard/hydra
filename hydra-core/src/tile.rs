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

// Suit range starts (tile type indices).
pub const MANZU_START: u8 = 0;
pub const PINZU_START: u8 = 9;
pub const SOUZU_START: u8 = 18;
pub const JIHAI_START: u8 = 27;

// Named honor tile indices for readability.
pub const EAST: u8 = 27;
pub const SOUTH: u8 = 28;
pub const WEST: u8 = 29;
pub const NORTH: u8 = 30;
pub const HAKU: u8 = 31;
pub const HATSU: u8 = 32;
pub const CHUN: u8 = 33;

// ---------------------------------------------------------------------------
// Aka-dora constants (136-format indices for red fives)
// ---------------------------------------------------------------------------

/// Red 5m in 136-format. The 0th copy of tile type 4 (5m) is red.
pub const AKA_MANZU_136: u8 = 16;
/// Red 5p in 136-format. The 0th copy of tile type 13 (5p) is red.
pub const AKA_PINZU_136: u8 = 52;
/// Red 5s in 136-format. The 0th copy of tile type 22 (5s) is red.
pub const AKA_SOUZU_136: u8 = 88;

/// Extended tile type indices for aka-dora (used in action encoding).
pub const AKA_MANZU_TYPE: u8 = 34;
pub const AKA_PINZU_TYPE: u8 = 35;
pub const AKA_SOUZU_TYPE: u8 = 36;

// ---------------------------------------------------------------------------
// Suit
// ---------------------------------------------------------------------------

/// The four tile categories in Riichi Mahjong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Suit {
    Manzu = 0,
    Pinzu = 1,
    Souzu = 2,
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
        AKA_MANZU_TYPE => 4,  // 5m base
        AKA_PINZU_TYPE => 13, // 5p base
        AKA_SOUZU_TYPE => 22, // 5s base
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
        4 => AKA_MANZU_TYPE,
        13 => AKA_PINZU_TYPE,
        22 => AKA_SOUZU_TYPE,
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
mod tests {
    use super::*;

    #[test]
    fn tile_type_new_valid() {
        for i in 0..34u8 {
            assert!(
                TileType::new(i).is_some(),
                "TileType::new({i}) should be Some"
            );
        }
        assert!(TileType::new(34).is_none());
        assert!(TileType::new(255).is_none());
    }

    #[test]
    fn suit_classification() {
        // Manzu 0-8
        for i in 0..9u8 {
            let t = TileType::new(i).unwrap();
            assert_eq!(t.suit(), Suit::Manzu, "tile {i} should be Manzu");
            assert!(t.is_suited());
            assert!(!t.is_honor());
        }
        // Pinzu 9-17
        for i in 9..18u8 {
            let t = TileType::new(i).unwrap();
            assert_eq!(t.suit(), Suit::Pinzu, "tile {i} should be Pinzu");
        }
        // Souzu 18-26
        for i in 18..27u8 {
            let t = TileType::new(i).unwrap();
            assert_eq!(t.suit(), Suit::Souzu, "tile {i} should be Souzu");
        }
        // Jihai 27-33
        for i in 27..34u8 {
            let t = TileType::new(i).unwrap();
            assert_eq!(t.suit(), Suit::Jihai, "tile {i} should be Jihai");
            assert!(t.is_honor());
            assert!(!t.is_suited());
        }
    }

    #[test]
    fn tile_number() {
        // Suited tiles have 1-based numbers
        assert_eq!(TileType::new(0).unwrap().number(), Some(1)); // 1m
        assert_eq!(TileType::new(8).unwrap().number(), Some(9)); // 9m
        assert_eq!(TileType::new(9).unwrap().number(), Some(1)); // 1p
        assert_eq!(TileType::new(22).unwrap().number(), Some(5)); // 5s
                                                                  // Honors have no number
        assert_eq!(TileType::new(27).unwrap().number(), None);
        assert_eq!(TileType::new(33).unwrap().number(), None);
    }

    #[test]
    fn terminal_detection() {
        let terminals = [0, 8, 9, 17, 18, 26]; // 1m,9m,1p,9p,1s,9s
        for &i in &terminals {
            let t = TileType::new(i).unwrap();
            assert!(t.is_terminal(), "tile {i} should be terminal");
            assert!(t.is_terminal_or_honor());
        }
        // Middle tiles are not terminal
        let middles = [1, 4, 10, 14, 19, 23];
        for &i in &middles {
            let t = TileType::new(i).unwrap();
            assert!(!t.is_terminal(), "tile {i} should NOT be terminal");
        }
        // Honors are not terminal but are terminal_or_honor
        for i in 27..34u8 {
            let t = TileType::new(i).unwrap();
            assert!(!t.is_terminal());
            assert!(t.is_terminal_or_honor());
        }
    }

    #[test]
    fn tile136_to_type_correct() {
        // Each group of 4 consecutive 136-tiles maps to one type
        for t in 0..34u8 {
            for copy in 0..4u8 {
                let t136 = t * 4 + copy;
                assert_eq!(tile136_to_type(t136).id(), t);
            }
        }
    }

    #[test]
    fn aka_detection_136() {
        assert!(tile136_is_aka(16)); // red 5m
        assert!(tile136_is_aka(52)); // red 5p
        assert!(tile136_is_aka(88)); // red 5s
                                     // Non-aka copies of the same tile types
        assert!(!tile136_is_aka(17)); // normal 5m
        assert!(!tile136_is_aka(18)); // normal 5m
        assert!(!tile136_is_aka(53)); // normal 5p
        assert!(!tile136_is_aka(0)); // 1m
    }

    #[test]
    fn deaka_strips_aka() {
        assert_eq!(deaka(34), 4); // aka 5m -> 5m
        assert_eq!(deaka(35), 13); // aka 5p -> 5p
        assert_eq!(deaka(36), 22); // aka 5s -> 5s
                                   // Non-aka pass through
        assert_eq!(deaka(0), 0);
        assert_eq!(deaka(4), 4);
        assert_eq!(deaka(33), 33);
    }

    #[test]
    fn re_akaize_roundtrip() {
        // Aka types roundtrip through deaka -> re_akaize
        for aka in [AKA_MANZU_TYPE, AKA_PINZU_TYPE, AKA_SOUZU_TYPE] {
            let base = deaka(aka);
            assert_eq!(re_akaize(base, true), aka);
        }
        // Non-aka tiles are unaffected
        assert_eq!(re_akaize(4, false), 4);
        assert_eq!(re_akaize(0, false), 0);
    }

    #[test]
    fn permutation_identity() {
        let identity = &ALL_PERMUTATIONS[0];
        for i in 0..34u8 {
            assert_eq!(permute_tile_type(i, identity), i);
        }
    }

    #[test]
    fn all_permutations_produce_valid_types() {
        for perm in &ALL_PERMUTATIONS {
            for i in 0..34u8 {
                let result = permute_tile_type(i, perm);
                assert!(
                    result < NUM_TILE_TYPES as u8,
                    "permute_tile_type({i}, {perm:?}) = {result} out of range"
                );
            }
        }
    }

    #[test]
    fn permutation_honors_unchanged() {
        for perm in &ALL_PERMUTATIONS {
            for i in 27..34u8 {
                assert_eq!(
                    permute_tile_type(i, perm),
                    i,
                    "honor tile {i} should not be affected by permutation {perm:?}"
                );
            }
        }
    }

    #[test]
    fn permutation_swap_man_pin() {
        let perm = &ALL_PERMUTATIONS[2]; // [1, 0, 2] = swap man-pin
        assert_eq!(permute_tile_type(0, perm), 9); // 1m -> 1p
        assert_eq!(permute_tile_type(9, perm), 0); // 1p -> 1m
        assert_eq!(permute_tile_type(18, perm), 18); // 1s -> 1s
        assert_eq!(permute_tile_type(27, perm), 27); // E -> E
    }

    #[test]
    fn permute_tile136_preserves_aka() {
        let perm = &ALL_PERMUTATIONS[2]; // swap man-pin
                                         // Red 5m (136-idx 16) -> should become red 5p (136-idx 52)
        let result = permute_tile136(AKA_MANZU_136, perm);
        assert_eq!(result, AKA_PINZU_136);
        assert!(tile136_is_aka(result), "aka status should be preserved");

        // Red 5p -> red 5m
        let result2 = permute_tile136(AKA_PINZU_136, perm);
        assert_eq!(result2, AKA_MANZU_136);
        assert!(tile136_is_aka(result2));
    }

    #[test]
    fn permute_extended_aka() {
        let perm = &ALL_PERMUTATIONS[2]; // swap man-pin
        assert_eq!(permute_tile_extended(AKA_MANZU_TYPE, perm), AKA_PINZU_TYPE);
        assert_eq!(permute_tile_extended(AKA_PINZU_TYPE, perm), AKA_MANZU_TYPE);
        assert_eq!(permute_tile_extended(AKA_SOUZU_TYPE, perm), AKA_SOUZU_TYPE);
        // Non-aka pass through
        assert_eq!(permute_tile_extended(0, perm), 9);
    }

    #[test]
    fn all_permutations_are_bijections() {
        // Each permutation should be a bijection on the 34 tile types
        for perm in &ALL_PERMUTATIONS {
            let mut seen = [false; NUM_TILE_TYPES];
            for i in 0..34u8 {
                let out = permute_tile_type(i, perm) as usize;
                assert!(!seen[out], "duplicate output {out} for perm {perm:?}");
                seen[out] = true;
            }
            assert!(seen.iter().all(|&s| s), "perm {perm:?} is not surjective");
        }
    }

    #[test]
    fn tile_type_display() {
        assert_eq!(format!("{}", TileType::new(0).unwrap()), "1m");
        assert_eq!(format!("{}", TileType::new(8).unwrap()), "9m");
        assert_eq!(format!("{}", TileType::new(27).unwrap()), "E");
        assert_eq!(format!("{}", TileType::new(33).unwrap()), "C");
    }

    #[test]
    fn mjai_names() {
        assert_eq!(tile_type_to_mjai(0), "1m");
        assert_eq!(tile_type_to_mjai(9), "1p");
        assert_eq!(tile_type_to_mjai(18), "1s");
        assert_eq!(tile_type_to_mjai(27), "E");
        assert_eq!(tile_type_to_mjai(31), "P");
        assert_eq!(tile_type_to_mjai(32), "F");
        assert_eq!(tile_type_to_mjai(33), "C");
        assert_eq!(tile_type_to_mjai(99), "??");
    }
}
