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
