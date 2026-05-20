//! Fixed-superset observation tensor encoder for neural network input.
//!
//! Encodes the full game state into a flat `[f32; NUM_CHANNELS * 34]` array
//! (row-major) that serves as input to the Hydra SE-ResNet model.
//!
//! The currently implemented baseline channels remain intact in the first 85
//! planes. Additional Group C / Group D planes provide a fixed-shape superset
//! for search/belief and Hand-EV context, with zero-filled planes plus
//! presence-mask channels when those dynamic features are unavailable.
//!
//! Channels are grouped:
//!
//! - 0..3:   closed hand (thresholded tile counts)
//! - 4..7:   open meld hand counts (thresholded)
//! - 8:      drawn tile one-hot
//! - 9..10:  shanten masks (keep / next)
//! - 11..22: discards per player (presence, tedashi, temporal)
//! - 23..34: melds per player (chi, pon, kan)
//! - 35..39: dora indicator thermometer
//! - 40..42: aka dora flags (per suit plane)
//! - 43..61: game metadata (riichi, scores, gaps, shanten, round, honba, kyotaku)
//! - 62..84: safety channels (genbutsu, suji, kabe, one-chance, tenpai)
//! - 85..149: Group C search/belief context + presence masks + reserved slots
//! - 150..191: Group D Hand-EV context + presence mask
use hydra_belief_search::hand_ev::HandEvFeatures;
use hydra_belief_search::shanten_batch::{self, BatchShantenResult};
use hydra_runtime_types::tile::NUM_TILE_TYPES;
use hydra_safety::{self, SafetyInfo};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Baseline observation channels (public + safety).
pub const BASELINE_CHANNELS: usize = 85;

/// Group C search/belief context channels.
pub const SEARCH_CONTEXT_CHANNELS: usize = 65;

/// Group D Hand-EV channels.
pub const HAND_EV_CHANNELS: usize = 42;

/// First Group C search/belief channel.
pub const SEARCH_CHANNEL_START: usize = BASELINE_CHANNELS;

/// First Group C belief-field channel.
pub const SEARCH_BELIEF_CHANNEL_START: usize = SEARCH_CHANNEL_START;

/// First Group D Hand-EV channel.
pub const HAND_EV_CHANNEL_START: usize = SEARCH_CHANNEL_START + SEARCH_CONTEXT_CHANNELS;

/// Group C discard-level delta-Q channel.
pub const SEARCH_DELTA_Q_CHANNEL: usize = SEARCH_CHANNEL_START + 22;

/// Group C mixture-entropy scalar channel.
pub const SEARCH_MIXTURE_ENTROPY_CHANNEL: usize = SEARCH_CHANNEL_START + 20;

/// Group C mixture-ESS scalar channel.
pub const SEARCH_MIXTURE_ESS_CHANNEL: usize = SEARCH_CHANNEL_START + 21;

/// First Group C opponent-risk channel.
pub const SEARCH_RISK_CHANNEL_START: usize = SEARCH_CHANNEL_START + 23;

/// First Group C opponent-stress channel.
pub const SEARCH_STRESS_CHANNEL_START: usize = SEARCH_CHANNEL_START + 26;

/// First Group C presence-mask channel.
pub const SEARCH_MASK_CHANNEL_START: usize = SEARCH_CHANNEL_START + 29;

/// Final Group D Hand-EV presence-mask channel.
pub const HAND_EV_MASK_CHANNEL: usize = HAND_EV_CHANNEL_START + HAND_EV_CHANNELS - 1;

/// Total observation channels.
pub const NUM_CHANNELS: usize = BASELINE_CHANNELS + SEARCH_CONTEXT_CHANNELS + HAND_EV_CHANNELS;

/// Tiles per channel (one per tile type).
pub const NUM_TILES: usize = NUM_TILE_TYPES; // 34

/// Total elements in the flat observation buffer.
pub const OBS_SIZE: usize = NUM_CHANNELS * NUM_TILES;

// -- Channel group starts --

const CH_HAND: usize = 0; // 0..3   (4 channels)
const CH_OPEN_MELD: usize = 4; // 4..7   (4 channels)
const CH_DRAWN: usize = 8; // 8      (1 channel)
const CH_SHANTEN_MASK: usize = 9; // 9..10  (2 channels)
const CH_DISCARDS: usize = 11; // 11..22 (12 channels: 3 per player)
const CH_MELDS: usize = 23; // 23..34 (12 channels: 3 per player)
const CH_DORA: usize = 35; // 35..39 (5 channels)
const CH_AKA: usize = 40; // 40..42 (3 channels)
const CH_META: usize = 43; // 43..61 (19 channels)
const CH_SAFETY: usize = 62; // 62..84 (23 channels)
const CH_SEARCH: usize = SEARCH_CHANNEL_START; // 85..149 (65 channels)
const CH_HAND_EV: usize = HAND_EV_CHANNEL_START; // 150..191 (42 channels)

const SEARCH_BELIEF_CHANNELS: usize = 16;
const SEARCH_MIXTURE_WEIGHT_CHANNELS: usize = 4;
const SEARCH_RISK_CHANNELS: usize = 3;
const SEARCH_STRESS_CHANNELS: usize = 3;
const SEARCH_MASK_CHANNELS: usize = 4;
const SEARCH_RESERVED_CHANNELS: usize = 32;

const CH_SEARCH_BELIEF: usize = CH_SEARCH; // 85..100
const CH_SEARCH_MIXTURE_WEIGHT: usize = CH_SEARCH_BELIEF + SEARCH_BELIEF_CHANNELS; // 101..104
const CH_SEARCH_MIXTURE_ENTROPY: usize = CH_SEARCH_MIXTURE_WEIGHT + SEARCH_MIXTURE_WEIGHT_CHANNELS; // 105
const CH_SEARCH_MIXTURE_ESS: usize = CH_SEARCH_MIXTURE_ENTROPY + 1; // 106
const CH_SEARCH_DELTA_Q: usize = CH_SEARCH_MIXTURE_ESS + 1; // 107
const CH_SEARCH_RISK: usize = CH_SEARCH_DELTA_Q + 1; // 108..110
const CH_SEARCH_STRESS: usize = CH_SEARCH_RISK + SEARCH_RISK_CHANNELS; // 111..113
const CH_SEARCH_MASKS: usize = CH_SEARCH_STRESS + SEARCH_STRESS_CHANNELS; // 114..117
const CH_SEARCH_RESERVED: usize = CH_SEARCH_MASKS + SEARCH_MASK_CHANNELS; // 118..149

const CH_HAND_EV_TENPAI: usize = CH_HAND_EV; // 150..152
const CH_HAND_EV_WIN: usize = CH_HAND_EV_TENPAI + 3; // 153..155
const CH_HAND_EV_SCORE: usize = CH_HAND_EV_WIN + 3; // 156
const CH_HAND_EV_UKEIRE: usize = CH_HAND_EV_SCORE + 1; // 157..190
const CH_HAND_EV_MASK: usize = CH_HAND_EV_UKEIRE + NUM_TILES; // 191

/// Number of players at the table.
const NUM_PLAYERS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChannelRange {
    start: usize,
    end: usize,
}

const BASELINE_LAYOUT: [ChannelRange; 9] = [
    ChannelRange {
        start: CH_HAND,
        end: CH_HAND + 4,
    },
    ChannelRange {
        start: CH_OPEN_MELD,
        end: CH_OPEN_MELD + 4,
    },
    ChannelRange {
        start: CH_DRAWN,
        end: CH_DRAWN + 1,
    },
    ChannelRange {
        start: CH_SHANTEN_MASK,
        end: CH_SHANTEN_MASK + 2,
    },
    ChannelRange {
        start: CH_DISCARDS,
        end: CH_DISCARDS + 12,
    },
    ChannelRange {
        start: CH_MELDS,
        end: CH_MELDS + 12,
    },
    ChannelRange {
        start: CH_DORA,
        end: CH_AKA + 3,
    },
    ChannelRange {
        start: CH_META,
        end: CH_META + 19,
    },
    ChannelRange {
        start: CH_SAFETY,
        end: BASELINE_CHANNELS,
    },
];

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
// ObservationEncoder
// ---------------------------------------------------------------------------

/// Pre-allocated encoder buffer for the fixed-superset observation tensor.
///
/// Reuse across turns to avoid per-turn allocation. Call [`clear`] then
/// the individual `encode_*` methods, or use [`encode`] as the one-shot
/// entry point.
#[derive(Clone)]
#[repr(C)]
pub struct ObservationEncoder {
    /// Flat buffer: `NUM_CHANNELS` channels x 34 tiles, row-major.
    buffer: [f32; OBS_SIZE],
}

impl ObservationEncoder {
    /// Create a new encoder with a zeroed buffer.
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: [0.0; OBS_SIZE],
        }
    }

    /// Zero the entire buffer.
    #[inline]
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
    }

    /// Zero only the channels in range `[start_ch, end_ch)` (exclusive end).
    #[inline]
    pub fn clear_range(&mut self, start_ch: usize, end_ch: usize) {
        let start = start_ch * NUM_TILES;
        let end = end_ch * NUM_TILES;
        self.buffer[start..end].fill(0.0);
    }

    /// Read-only view of the flat observation buffer.
    #[inline]
    pub fn as_slice(&self) -> &[f32; OBS_SIZE] {
        &self.buffer
    }

    /// Set a single cell: `buffer[channel * 34 + tile] = value`.
    #[inline]
    fn set(&mut self, channel: usize, tile: usize, value: f32) {
        self.buffer[channel * NUM_TILES + tile] = value;
    }

    /// Fill an entire channel with a uniform value.
    #[inline]
    fn fill_channel(&mut self, channel: usize, value: f32) {
        let start = channel * NUM_TILES;
        self.buffer[start..start + NUM_TILES].fill(value);
    }

    #[inline]
    fn copy_channel(&mut self, channel: usize, values: &[f32; NUM_TILES]) {
        let start = channel * NUM_TILES;
        self.buffer[start..start + NUM_TILES].copy_from_slice(values);
    }
}

impl Default for ObservationEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Encoding: closed hand (channels 0-3)
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode the observer's closed hand tile counts into channels 0-3.
    ///
    /// Binary thresholded planes:
    /// - Ch 0: count >= 1
    /// - Ch 1: count >= 2
    /// - Ch 2: count >= 3
    /// - Ch 3: count == 4
    #[inline]
    pub fn encode_hand(&mut self, hand_counts: &[u8; NUM_TILES]) {
        let row0 = CH_HAND * NUM_TILES;
        let row1 = row0 + NUM_TILES;
        let row2 = row1 + NUM_TILES;
        let row3 = row2 + NUM_TILES;
        for (tile, &count) in hand_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            self.buffer[row0 + tile] = 1.0;
            if count >= 2 {
                self.buffer[row1 + tile] = 1.0;
            }
            if count >= 3 {
                self.buffer[row2 + tile] = 1.0;
            }
            if count == 4 {
                self.buffer[row3 + tile] = 1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding: open meld hand counts (channels 4-7)
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode tile counts contributed by open melds into channels 4-7.
    ///
    /// Same thermometer encoding as the closed hand:
    /// - Ch 4: count >= 1
    /// - Ch 5: count >= 2
    /// - Ch 6: count >= 3
    /// - Ch 7: count == 4
    #[inline]
    pub fn encode_open_meld_hand(&mut self, counts: &[u8; NUM_TILES]) {
        let row0 = CH_OPEN_MELD * NUM_TILES;
        let row1 = row0 + NUM_TILES;
        let row2 = row1 + NUM_TILES;
        let row3 = row2 + NUM_TILES;
        for (tile, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            self.buffer[row0 + tile] = 1.0;
            if count >= 2 {
                self.buffer[row1 + tile] = 1.0;
            }
            if count >= 3 {
                self.buffer[row2 + tile] = 1.0;
            }
            if count == 4 {
                self.buffer[row3 + tile] = 1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding: drawn tile (channel 8)
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode the drawn tile as a one-hot on channel 8.
    /// `None` means no tile was drawn (e.g. first turn or after a call).
    #[inline]
    pub fn encode_drawn_tile(&mut self, tile: Option<u8>) {
        if let Some(t) = tile {
            let idx = t as usize;
            if idx < NUM_TILES {
                self.set(CH_DRAWN, idx, 1.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding: shanten masks (channels 9-10)
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode shanten-based discard masks into channels 9-10.
    ///
    /// - Ch 9 (keep-shanten): 1.0 for tiles whose discard does not increase shanten.
    /// - Ch 10 (next-shanten): 1.0 for tiles whose discard decreases shanten.
    ///
    /// `hand` is the full hand including drawn tile (typically 14 tiles).
    #[inline]
    pub fn encode_shanten_masks(&mut self, hand: &[u8; NUM_TILES]) {
        let total: u8 = hand.iter().sum();
        let len_div3 = total / 3;
        let batch = shanten_batch::batch_discard_shanten(hand, len_div3);
        self.encode_shanten_masks_from_batch(&batch);
    }

    #[inline]
    pub fn encode_shanten_masks_from_batch(&mut self, batch: &BatchShantenResult) {
        for tile in 0..NUM_TILES {
            if let Some(after) = batch.discard[tile] {
                if after <= batch.base {
                    self.set(CH_SHANTEN_MASK, tile, 1.0);
                }
                if after < batch.base {
                    self.set(CH_SHANTEN_MASK + 1, tile, 1.0);
                }
            }
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
// Encoding: discards (channels 11-22)
// ---------------------------------------------------------------------------

/// Precomputed `exp(-DISCARD_DECAY * i)` for `i` in `0..=30`.
/// DISCARD_DECAY is 0.2, so entry `i` = `exp(-0.2 * i)`.
#[allow(
    clippy::excessive_precision,
    reason = "table entries preserve the measured decay curve"
)]
const DISCARD_EXP_TABLE: [f32; 31] = [
    1.0,           // exp(0.0)
    0.818_730_8,   // exp(-0.2)
    0.670_320_0,   // exp(-0.4)
    0.548_811_6,   // exp(-0.6)
    0.449_329_0,   // exp(-0.8)
    0.367_879_5,   // exp(-1.0)
    0.301_194_2,   // exp(-1.2)
    0.246_597_0,   // exp(-1.4)
    0.201_896_5,   // exp(-1.6)
    0.165_298_9,   // exp(-1.8)
    0.135_335_3,   // exp(-2.0)
    0.110_803_2,   // exp(-2.2)
    0.090_717_96,  // exp(-2.4)
    0.074_273_58,  // exp(-2.6)
    0.060_810_06,  // exp(-2.8)
    0.049_787_07,  // exp(-3.0)
    0.040_762_20,  // exp(-3.2)
    0.033_373_27,  // exp(-3.4)
    0.027_323_72,  // exp(-3.6)
    0.022_370_77,  // exp(-3.8)
    0.018_315_64,  // exp(-4.0)
    0.014_995_58,  // exp(-4.2)
    0.012_277_34,  // exp(-4.4)
    0.010_051_84,  // exp(-4.6)
    0.008_229_747, // exp(-4.8)
    0.006_737_947, // exp(-5.0)
    0.005_516_564, // exp(-5.2)
    0.004_516_581, // exp(-5.4)
    0.003_697_864, // exp(-5.6)
    0.003_027_555, // exp(-5.8)
    0.002_478_752, // exp(-6.0)
];
impl ObservationEncoder {
    /// Encode discard info for all 4 players into channels 11-22.
    ///
    /// Per player (3 channels each):
    /// - presence:  binary 1.0 if tile was discarded by this player
    /// - tedashi:   binary 1.0 if that discard was from hand (not tsumogiri)
    /// - temporal:  exp(-0.2 * (t_max - t_discard))
    #[inline]
    pub fn encode_discards(&mut self, discards: &[PlayerDiscards; NUM_PLAYERS]) {
        for (p, pd) in discards.iter().enumerate() {
            let ch_base = CH_DISCARDS + 3 * p;
            let sl = pd.as_slice();
            let t_max = sl.last().map(|d| d.turn).unwrap_or(0);
            let row_presence = ch_base * NUM_TILES;
            let row_tedashi = row_presence + NUM_TILES;
            let row_temporal = row_tedashi + NUM_TILES;
            for d in sl {
                let t = d.tile as usize;
                if t >= NUM_TILES {
                    continue;
                }
                self.buffer[row_presence + t] = 1.0;
                if d.is_tedashi {
                    self.buffer[row_tedashi + t] = 1.0;
                }
                let dt = (t_max - d.turn).min(30) as usize;
                let w = DISCARD_EXP_TABLE[dt];
                let idx = row_temporal + t;
                if w > self.buffer[idx] {
                    self.buffer[idx] = w;
                }
            }
        }
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
// Encoding: melds (channels 23-34)
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode melds for all 4 players into channels 23-34.
    ///
    /// Per player (3 channels each):
    /// - chi tiles
    /// - pon tiles
    /// - kan tiles
    #[inline]
    pub fn encode_melds(&mut self, melds: &[PlayerMelds; NUM_PLAYERS]) {
        for (p, player_melds) in melds.iter().enumerate() {
            let ch_base = CH_MELDS + 3 * p;
            for meld in player_melds.as_slice() {
                let ch_offset = match meld.meld_type {
                    MeldType::Chi => 0,
                    MeldType::Pon => 1,
                    MeldType::Kan => 2,
                };
                let row = (ch_base + ch_offset) * NUM_TILES;
                match meld.meld_type {
                    MeldType::Chi => {
                        for &tile in &meld.tiles[..meld.tile_count as usize] {
                            let t = tile as usize;
                            if t < NUM_TILES {
                                self.buffer[row + t] = 1.0;
                            }
                        }
                    }
                    MeldType::Pon | MeldType::Kan => {
                        let t = meld.tiles[0] as usize;
                        if t < NUM_TILES {
                            self.buffer[row + t] = 1.0;
                        }
                    }
                }
            }
        }
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

impl ObservationEncoder {
    /// Encode dora indicators as a thermometer into channels 35-39.
    ///
    /// Counts how many indicators point to each tile type, then thresholds:
    /// - Ch 35: count >= 1
    /// - Ch 36: count >= 2
    /// - Ch 37: count >= 3
    /// - Ch 38: count >= 4
    /// - Ch 39: count >= 5
    #[inline]
    pub fn encode_dora(&mut self, dora: &DoraInfo) {
        let mut counts = [0u8; NUM_TILES];
        let mut touched = [0usize; 5];
        let mut touched_len = 0usize;
        for &ind in &dora.indicators[..dora.indicator_count as usize] {
            let i = ind as usize;
            if i < NUM_TILES {
                if counts[i] == 0 {
                    touched[touched_len] = i;
                    touched_len += 1;
                }
                counts[i] = counts[i].saturating_add(1);
            }
        }
        let row0 = CH_DORA * NUM_TILES;
        let row1 = row0 + NUM_TILES;
        let row2 = row1 + NUM_TILES;
        let row3 = row2 + NUM_TILES;
        let row4 = row3 + NUM_TILES;
        for &tile in &touched[..touched_len] {
            let c = counts[tile];
            if c >= 1 {
                self.buffer[row0 + tile] = 1.0;
            }
            if c >= 2 {
                self.buffer[row1 + tile] = 1.0;
            }
            if c >= 3 {
                self.buffer[row2 + tile] = 1.0;
            }
            if c >= 4 {
                self.buffer[row3 + tile] = 1.0;
            }
            if c >= 5 {
                self.buffer[row4 + tile] = 1.0;
            }
        }
    }

    /// Encode aka dora flags into channels 40-42 (one plane per suit).
    ///
    /// Each channel is fully filled with 1.0 if the corresponding aka is present.
    /// - Ch 40: has red 5m
    /// - Ch 41: has red 5p
    /// - Ch 42: has red 5s
    #[inline]
    pub fn encode_aka(&mut self, dora: &DoraInfo) {
        for (suit, &has_aka) in dora.aka_flags.iter().enumerate() {
            if has_aka {
                self.fill_channel(CH_AKA + suit, 1.0);
            }
        }
    }
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

// ---------------------------------------------------------------------------
// Encoding: game metadata (channels 43-61)
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode game metadata into channels 43-61.
    ///
    /// Layout:
    /// - Ch 43-46: riichi flags (4 players)
    /// - Ch 47-50: scores / 100000.0 (4 players)
    /// - Ch 51-54: relative score gaps (my - their) / 30000.0 (4 players)
    /// - Ch 55-58: shanten one-hot (0=tenpai, 1, 2, 3+)
    /// - Ch 59: round number (kyoku_index / 8.0)
    /// - Ch 60: honba / 10.0
    /// - Ch 61: kyotaku / 10.0
    #[inline]
    pub fn encode_metadata(&mut self, meta: &GameMetadata) {
        // Riichi flags (ch 43-46)
        for (i, &r) in meta.riichi.iter().enumerate() {
            if r {
                self.fill_channel(CH_META + i, 1.0);
            }
        }

        // Scores normalized (ch 47-50)
        for (i, &score) in meta.scores.iter().enumerate() {
            self.fill_channel(CH_META + 4 + i, score as f32 / 100_000.0);
        }

        // Relative score gaps (ch 51-54): (my_score - their_score) / 30000
        let my_score = meta.scores[0];
        for (i, &their_score) in meta.scores.iter().enumerate() {
            let gap = (my_score - their_score) as f32 / 30_000.0;
            self.fill_channel(CH_META + 8 + i, gap);
        }

        // Shanten one-hot (ch 55-58): 0=tenpai, 1, 2, 3+
        let sh = meta.shanten.clamp(0, 3) as usize;
        self.fill_channel(CH_META + 12 + sh, 1.0);

        // Round number (ch 59)
        self.fill_channel(CH_META + 16, meta.kyoku_index as f32 / 8.0);

        // Honba (ch 60)
        self.fill_channel(CH_META + 17, meta.honba as f32 / 10.0);

        // Kyotaku (ch 61)
        self.fill_channel(CH_META + 18, meta.kyotaku as f32 / 10.0);
    }
}

// ---------------------------------------------------------------------------
// Encoding: safety channels (62-84)
// ---------------------------------------------------------------------------

/// Number of opponents for safety channels.
const NUM_OPPS: usize = hydra_safety::NUM_OPPONENTS; // 3

/// Iterate over set bits in a `u64` bitmask using trailing-zeros extraction.
///
/// More efficient than scanning all 34 positions when the mask is sparse
/// (typically 5-10 bits set).
#[inline]
fn for_each_set_bit(mut bits: u64, mut f: impl FnMut(usize)) {
    while bits != 0 {
        let idx = bits.trailing_zeros() as usize;
        f(idx);
        bits &= bits - 1; // clear lowest set bit
    }
}

impl ObservationEncoder {
    /// Encode safety info into channels 62-84 (23 channels total).
    ///
    /// Layout:
    /// - Ch 62-64: genbutsu_all (per opponent)
    /// - Ch 65-67: genbutsu_tedashi (per opponent)
    /// - Ch 68-70: genbutsu_riichi_era (per opponent)
    /// - Ch 71-73: suji (per opponent, float 0.0-1.0)
    /// - Ch 74-76: half-suji indicator (per opponent)
    /// - Ch 77-79: matagi-suji danger (per opponent)
    /// - Ch 80: kabe
    /// - Ch 81: one-chance
    /// - Ch 82-84: tenpai hints (riichi or cached tenpai prediction > 0.5)
    #[inline]
    pub fn encode_safety(&mut self, safety: &SafetyInfo) {
        for opp in 0..NUM_OPPS {
            for_each_set_bit(safety.genbutsu_all[opp], |tile| {
                self.set(CH_SAFETY + opp, tile, 1.0);
            });
            for_each_set_bit(safety.genbutsu_tedashi[opp], |tile| {
                self.set(CH_SAFETY + NUM_OPPS + opp, tile, 1.0);
            });
            for_each_set_bit(safety.genbutsu_riichi_era[opp], |tile| {
                self.set(CH_SAFETY + 2 * NUM_OPPS + opp, tile, 1.0);
            });
            self.copy_channel(CH_SAFETY + 3 * NUM_OPPS + opp, &safety.suji[opp]);
        }

        // Ch 74-76: half-suji indicator per opponent
        for opp in 0..NUM_OPPS {
            for_each_set_bit(safety.half_suji[opp], |tile| {
                self.set(CH_SAFETY + 12 + opp, tile, 1.0);
            });
        }

        // Ch 77-79: matagi-suji danger per opponent
        for opp in 0..NUM_OPPS {
            self.copy_channel(CH_SAFETY + 15 + opp, &safety.matagi[opp]);
        }

        for_each_set_bit(safety.kabe, |tile| {
            self.set(CH_SAFETY + 18, tile, 1.0);
        });
        for_each_set_bit(safety.one_chance, |tile| {
            self.set(CH_SAFETY + 19, tile, 1.0);
        });

        // Ch 82-84: tenpai hints per opponent.
        for opp in 0..NUM_OPPS {
            if safety.tenpai_hint_active(opp) {
                self.fill_channel(CH_SAFETY + 20 + opp, 1.0);
            }
        }
    }

    /// Encode fixed-shape Group C search/belief context planes.
    #[inline]
    pub fn encode_search_features(&mut self, features: &SearchFeaturePlanes) {
        self.clear_range(CH_SEARCH, CH_SEARCH + SEARCH_CONTEXT_CHANNELS);

        for (idx, plane) in features.belief_fields.iter().enumerate() {
            self.copy_channel(CH_SEARCH_BELIEF + idx, plane);
        }
        for (idx, &weight) in features.mixture_weights.iter().enumerate() {
            self.fill_channel(CH_SEARCH_MIXTURE_WEIGHT + idx, weight);
        }
        self.fill_channel(CH_SEARCH_MIXTURE_ENTROPY, features.mixture_entropy);
        self.fill_channel(CH_SEARCH_MIXTURE_ESS, features.mixture_ess);
        self.copy_channel(CH_SEARCH_DELTA_Q, &features.delta_q);
        for (idx, plane) in features.opponent_risk.iter().enumerate() {
            self.copy_channel(CH_SEARCH_RISK + idx, plane);
        }
        for (idx, &stress) in features.opponent_stress.iter().enumerate() {
            self.fill_channel(CH_SEARCH_STRESS + idx, stress);
        }
        if features.belief_features_present {
            self.fill_channel(CH_SEARCH_MASKS, 1.0);
        }
        if features.search_features_present {
            self.fill_channel(CH_SEARCH_MASKS + 1, 1.0);
        }
        if features.robust_features_present {
            self.fill_channel(CH_SEARCH_MASKS + 2, 1.0);
        }
        if features.context_features_present {
            self.fill_channel(CH_SEARCH_MASKS + 3, 1.0);
        }

        let _ = CH_SEARCH_RESERVED;
        let _ = SEARCH_RESERVED_CHANNELS;
    }

    /// Encode fixed-shape Group D Hand-EV context planes.
    #[inline]
    pub fn encode_hand_ev_features(&mut self, hand_ev: &HandEvFeatures) {
        self.clear_range(CH_HAND_EV, CH_HAND_EV + HAND_EV_CHANNELS);

        let tenpai0 = (CH_HAND_EV_TENPAI) * NUM_TILES;
        let tenpai1 = tenpai0 + NUM_TILES;
        let tenpai2 = tenpai1 + NUM_TILES;
        let win0 = (CH_HAND_EV_WIN) * NUM_TILES;
        let win1 = win0 + NUM_TILES;
        let win2 = win1 + NUM_TILES;
        let score = CH_HAND_EV_SCORE * NUM_TILES;

        for discard in 0..NUM_TILES {
            let tenpai = hand_ev.tenpai_prob[discard];
            let win = hand_ev.win_prob[discard];
            self.buffer[tenpai0 + discard] = tenpai[0];
            self.buffer[tenpai1 + discard] = tenpai[1];
            self.buffer[tenpai2 + discard] = tenpai[2];
            self.buffer[win0 + discard] = win[0];
            self.buffer[win1 + discard] = win[1];
            self.buffer[win2 + discard] = win[2];
            self.buffer[score + discard] = hand_ev.expected_score[discard];
        }
        for draw_tile in 0..NUM_TILES {
            let row = (CH_HAND_EV_UKEIRE + draw_tile) * NUM_TILES;
            for discard in 0..NUM_TILES {
                self.buffer[row + discard] = hand_ev.ukeire[discard][draw_tile];
            }
        }
        self.fill_channel(CH_HAND_EV_MASK, 1.0);
    }

    /// Encode a complete observation plus optional Group C / Group D context.
    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    pub fn encode_with_context(
        &mut self,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
        search_features: Option<&SearchFeaturePlanes>,
        hand_ev: Option<&HandEvFeatures>,
    ) -> &[f32; OBS_SIZE] {
        self.encode(
            hand,
            drawn_tile,
            open_meld_counts,
            discards,
            melds,
            dora,
            meta,
            safety,
        );
        if let Some(features) = search_features {
            self.encode_search_features(features);
        }
        if let Some(features) = hand_ev {
            self.encode_hand_ev_features(features);
        }
        self.as_slice()
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    pub fn encode_with_context_and_shanten_batch(
        &mut self,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
        shanten_batch: &BatchShantenResult,
        search_features: Option<&SearchFeaturePlanes>,
        hand_ev: Option<&HandEvFeatures>,
    ) -> &[f32; OBS_SIZE] {
        self.clear();
        self.encode_baseline_prefix_from_batch(
            hand,
            drawn_tile,
            open_meld_counts,
            discards,
            melds,
            dora,
            meta,
            safety,
            shanten_batch,
        );
        if let Some(features) = search_features {
            self.encode_search_features(features);
        }
        if let Some(features) = hand_ev {
            self.encode_hand_ev_features(features);
        }
        self.as_slice()
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    fn encode_baseline_prefix(
        &mut self,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
    ) {
        let total: u8 = hand.iter().sum();
        let len_div3 = total / 3;
        let shanten_batch = shanten_batch::batch_discard_shanten(hand, len_div3);
        self.encode_baseline_prefix_from_batch(
            hand,
            drawn_tile,
            open_meld_counts,
            discards,
            melds,
            dora,
            meta,
            safety,
            &shanten_batch,
        );
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    fn encode_baseline_prefix_from_batch(
        &mut self,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
        shanten_batch: &BatchShantenResult,
    ) {
        #[cfg(debug_assertions)]
        {
            let mut cursor = 0;
            for range in BASELINE_LAYOUT {
                debug_assert_eq!(range.start, cursor);
                cursor = range.end;
            }
            debug_assert_eq!(cursor, BASELINE_CHANNELS);
        }
        self.encode_hand(hand);
        self.encode_open_meld_hand(open_meld_counts);
        self.encode_drawn_tile(drawn_tile);
        self.encode_shanten_masks_from_batch(shanten_batch);
        self.encode_discards(discards);
        self.encode_melds(melds);
        self.encode_dora(dora);
        self.encode_aka(dora);
        self.encode_metadata(meta);
        self.encode_safety(safety);
    }
}

// ---------------------------------------------------------------------------
// Incremental encoding: selective channel updates
// ---------------------------------------------------------------------------

/// Dirty flags indicating which channel groups need re-encoding.
///
/// Use with [`ObservationEncoder::encode_incremental`] to avoid
/// recomputing the full observation tensor when only a few groups changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct DirtyFlags(pub u16);

impl DirtyFlags {
    /// Closed hand channels (0-3) need re-encoding.
    pub const HAND: Self = Self(1 << 0); // Ch 0-3
    /// Open meld hand channels (4-7) need re-encoding.
    pub const OPEN_MELD: Self = Self(1 << 1); // Ch 4-7
    /// Drawn tile channel (8) needs re-encoding.
    pub const DRAWN: Self = Self(1 << 2); // Ch 8
    /// Shanten mask channels (9-10) need re-encoding.
    pub const SHANTEN: Self = Self(1 << 3); // Ch 9-10
    /// Discard channels (11-22) need re-encoding.
    pub const DISCARDS: Self = Self(1 << 4); // Ch 11-22
    /// Meld channels (23-34) need re-encoding.
    pub const MELDS: Self = Self(1 << 5); // Ch 23-34
    /// Dora channels (35-42) need re-encoding.
    pub const DORA: Self = Self(1 << 6); // Ch 35-42
    /// Game metadata channels (43-61) need re-encoding.
    pub const META: Self = Self(1 << 7); // Ch 43-61
    /// Safety channels (62-84) need re-encoding.
    pub const SAFETY: Self = Self(1 << 8); // Ch 62-84
    /// Search/belief Group C channels (85-149) need re-encoding.
    pub const SEARCH: Self = Self(1 << 9);
    /// Hand-EV Group D channels (150-191) need re-encoding.
    pub const HAND_EV: Self = Self(1 << 10);
    /// All channels need re-encoding.
    pub const ALL: Self = Self(0x7FF);

    /// After a draw: hand, drawn tile, shanten, metadata.
    pub const AFTER_DRAW: Self = Self(
        Self::HAND.0
            | Self::DRAWN.0
            | Self::SHANTEN.0
            | Self::META.0
            | Self::SEARCH.0
            | Self::HAND_EV.0,
    );
    /// After a discard: hand, drawn, shanten, discards, metadata, safety.
    pub const AFTER_DISCARD: Self = Self(
        Self::HAND.0
            | Self::DRAWN.0
            | Self::SHANTEN.0
            | Self::DISCARDS.0
            | Self::META.0
            | Self::SAFETY.0
            | Self::SEARCH.0
            | Self::HAND_EV.0,
    );
    /// After a call: hand, open melds, shanten, discards, melds, meta, safety.
    pub const AFTER_CALL: Self = Self(
        Self::HAND.0
            | Self::OPEN_MELD.0
            | Self::SHANTEN.0
            | Self::DISCARDS.0
            | Self::MELDS.0
            | Self::META.0
            | Self::SAFETY.0
            | Self::SEARCH.0
            | Self::HAND_EV.0,
    );
    /// New round: everything.
    pub const NEW_ROUND: Self = Self::ALL;

    #[inline]
    /// Check whether all flags in `other` are set in `self`.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    #[inline]
    /// Return a new `DirtyFlags` with all bits from both `self` and `other`.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

// ---------------------------------------------------------------------------
// Full encode entry point
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode a complete observation from explicit game state components.
    ///
    /// Clears the buffer, then calls each sub-encoder in order.
    /// Returns a reference to the filled observation buffer.
    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    pub fn encode(
        &mut self,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
    ) -> &[f32; OBS_SIZE] {
        self.clear();
        self.encode_baseline_prefix(
            hand,
            drawn_tile,
            open_meld_counts,
            discards,
            melds,
            dora,
            meta,
            safety,
        );
        self.as_slice()
    }
}

impl ObservationEncoder {
    /// Incrementally re-encode only the channel groups marked dirty.
    ///
    /// Unlike [`encode`], this does NOT clear the entire buffer. It only
    /// clears and rewrites the specific channel ranges that changed.
    /// Use [`DirtyFlags`] presets (`AFTER_DRAW`, `AFTER_DISCARD`,
    /// `AFTER_CALL`) for common game events.
    ///
    /// For a new round or first encode, use `DirtyFlags::ALL` or [`encode`].
    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    pub fn encode_incremental(
        &mut self,
        dirty: DirtyFlags,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
    ) -> &[f32; OBS_SIZE] {
        if dirty.contains(DirtyFlags::HAND) {
            self.clear_range(CH_HAND, CH_HAND + 4);
            self.encode_hand(hand);
        }
        if dirty.contains(DirtyFlags::OPEN_MELD) {
            self.clear_range(CH_OPEN_MELD, CH_OPEN_MELD + 4);
            self.encode_open_meld_hand(open_meld_counts);
        }
        if dirty.contains(DirtyFlags::DRAWN) {
            self.clear_range(CH_DRAWN, CH_DRAWN + 1);
            self.encode_drawn_tile(drawn_tile);
        }
        if dirty.contains(DirtyFlags::SHANTEN) {
            self.clear_range(CH_SHANTEN_MASK, CH_SHANTEN_MASK + 2);
            self.encode_shanten_masks(hand);
        }
        if dirty.contains(DirtyFlags::DISCARDS) {
            self.clear_range(CH_DISCARDS, CH_DISCARDS + 12);
            self.encode_discards(discards);
        }
        if dirty.contains(DirtyFlags::MELDS) {
            self.clear_range(CH_MELDS, CH_MELDS + 12);
            self.encode_melds(melds);
        }
        if dirty.contains(DirtyFlags::DORA) {
            self.clear_range(CH_DORA, CH_AKA + 3);
            self.encode_dora(dora);
            self.encode_aka(dora);
        }
        if dirty.contains(DirtyFlags::META) {
            self.clear_range(CH_META, CH_META + 19);
            self.encode_metadata(meta);
        }
        if dirty.contains(DirtyFlags::SAFETY) {
            self.clear_range(CH_SAFETY, CH_SAFETY + 23);
            self.encode_safety(safety);
        }
        self.as_slice()
    }

    /// Incrementally re-encode baseline plus optional Group C / Group D context.
    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    pub fn encode_incremental_with_context(
        &mut self,
        dirty: DirtyFlags,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
        search_features: Option<&SearchFeaturePlanes>,
        hand_ev: Option<&HandEvFeatures>,
    ) -> &[f32; OBS_SIZE] {
        self.encode_incremental(
            dirty,
            hand,
            drawn_tile,
            open_meld_counts,
            discards,
            melds,
            dora,
            meta,
            safety,
        );
        if dirty.contains(DirtyFlags::SEARCH) {
            self.clear_range(CH_SEARCH, CH_SEARCH + SEARCH_CONTEXT_CHANNELS);
            if let Some(features) = search_features {
                self.encode_search_features(features);
            }
        }
        if dirty.contains(DirtyFlags::HAND_EV) {
            self.clear_range(CH_HAND_EV, CH_HAND_EV + HAND_EV_CHANNELS);
            if let Some(features) = hand_ev {
                self.encode_hand_ev_features(features);
            }
        }
        self.as_slice()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
