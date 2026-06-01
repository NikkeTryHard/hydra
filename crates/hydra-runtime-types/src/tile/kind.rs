use std::fmt;

use super::constants::*;

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
