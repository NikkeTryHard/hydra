use super::layout::*;

/// Fixed-shape Group C search/belief context planes.
#[derive(Debug, Clone)]
pub struct SearchFeaturePlanes {
    pub belief_fields: [[f32; NUM_TILES]; SEARCH_BELIEF_CHANNELS],
    pub mixture_weights: [f32; SEARCH_MIXTURE_WEIGHT_CHANNELS],
    pub mixture_entropy: f32,
    pub mixture_ess: f32,
    pub delta_q: [f32; NUM_TILES],
    pub opponent_risk: [[f32; NUM_TILES]; SEARCH_RISK_CHANNELS],
    pub opponent_stress: [f32; SEARCH_STRESS_CHANNELS],
    pub belief_features_present: bool,
    pub search_features_present: bool,
    pub robust_features_present: bool,
    pub context_features_present: bool,
}

impl Default for SearchFeaturePlanes {
    fn default() -> Self {
        Self {
            belief_fields: [[0.0; NUM_TILES]; SEARCH_BELIEF_CHANNELS],
            mixture_weights: [0.0; SEARCH_MIXTURE_WEIGHT_CHANNELS],
            mixture_entropy: 0.0,
            mixture_ess: 0.0,
            delta_q: [0.0; NUM_TILES],
            opponent_risk: [[0.0; NUM_TILES]; SEARCH_RISK_CHANNELS],
            opponent_stress: [0.0; SEARCH_STRESS_CHANNELS],
            belief_features_present: false,
            search_features_present: false,
            robust_features_present: false,
            context_features_present: false,
        }
    }
}
// ---------------------------------------------------------------------------
// Discard info input type
// ---------------------------------------------------------------------------

/// A single discard event for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscardEntry {
    /// Tile type (0-33).
    pub tile: u8,
    /// True if discarded from hand (not tsumogiri).
    pub is_tedashi: bool,
    /// 0-based turn index when this discard happened.
    pub turn: u16,
}

/// Per-player discard history for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerDiscards {
    /// Fixed-size array of discards (oldest first).
    pub discards: [DiscardEntry; 30],
    /// Number of valid entries in `discards`.
    pub len: u8,
}

impl Default for PlayerDiscards {
    fn default() -> Self {
        Self::new()
    }
}

impl PlayerDiscards {
    /// Create an empty discard history.
    #[inline]
    pub fn new() -> Self {
        Self {
            discards: [DiscardEntry {
                tile: 0,
                is_tedashi: false,
                turn: 0,
            }; 30],
            len: 0,
        }
    }

    /// Append a discard entry. Silently drops if at capacity (30).
    #[inline]
    pub fn push(&mut self, entry: DiscardEntry) {
        let i = self.len as usize;
        if i < 30 {
            self.discards[i] = entry;
            self.len += 1;
        }
    }

    /// Return a slice of valid entries.
    #[inline]
    pub fn as_slice(&self) -> &[DiscardEntry] {
        &self.discards[..self.len as usize]
    }
}
// ---------------------------------------------------------------------------
// Meld info input type
// ---------------------------------------------------------------------------

/// Type of meld for encoding purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeldType {
    /// Chi (sequence call).
    Chi,
    /// Pon (triplet call).
    Pon,
    /// Kan (any kan: ankan, daiminkan, kakan).
    Kan,
}

/// A single meld for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeldInfo {
    /// Tile types present in the meld (0-33 each). Up to 4 tiles.
    pub tiles: [u8; 4],
    /// Number of valid tiles in `tiles`.
    pub tile_count: u8,
    /// What kind of meld this is.
    pub meld_type: MeldType,
}

/// Per-player meld collection for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerMelds {
    /// Fixed-size array of melds (max 4 per player).
    pub melds: [MeldInfo; 4],
    /// Number of valid melds.
    pub len: u8,
}

impl Default for PlayerMelds {
    fn default() -> Self {
        Self::new()
    }
}

impl PlayerMelds {
    /// Create an empty meld collection.
    #[inline]
    pub fn new() -> Self {
        Self {
            melds: [MeldInfo {
                tiles: [0; 4],
                tile_count: 0,
                meld_type: MeldType::Chi,
            }; 4],
            len: 0,
        }
    }

    /// Append a meld. Silently drops if at capacity (4).
    #[inline]
    pub fn push(&mut self, meld: MeldInfo) {
        let i = self.len as usize;
        if i < 4 {
            self.melds[i] = meld;
            self.len += 1;
        }
    }

    /// Return a slice of valid melds.
    #[inline]
    pub fn as_slice(&self) -> &[MeldInfo] {
        &self.melds[..self.len as usize]
    }
}
// ---------------------------------------------------------------------------
// Encoding: dora (channels 35-39) and aka (channels 40-42)
// ---------------------------------------------------------------------------

/// Dora information for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DoraInfo {
    /// Dora indicator tile types (0-33). Fixed array, up to 5 kan dora.
    pub indicators: [u8; 5],
    /// Number of valid indicators.
    pub indicator_count: u8,
    /// Aka dora flags: `[has_aka_5m, has_aka_5p, has_aka_5s]`.
    pub aka_flags: [bool; 3],
}
// ---------------------------------------------------------------------------
// Game metadata input type
// ---------------------------------------------------------------------------

/// Game metadata for encoding channels 43-61.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct GameMetadata {
    /// Riichi status for all 4 players (relative to observer). Index 0 = self.
    pub riichi: [bool; 4],
    /// Scores for all 4 players (relative to observer). Raw point values.
    pub scores: [i32; 4],
    /// Observer's shanten number (from calc_shanten_from_counts).
    pub shanten: i8,
    /// Round index (0-7: East 1 = 0, South 4 = 7).
    pub kyoku_index: u8,
    /// Honba (repeat) counter.
    pub honba: u8,
    /// Number of riichi sticks deposited on the table.
    pub kyotaku: u8,
}
