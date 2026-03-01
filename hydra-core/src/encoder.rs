//! 85x34 observation tensor encoder for neural network input.
//!
//! Encodes the full game state into a flat `[f32; 85 * 34]` array (row-major)
//! that serves as input to the Hydra SE-ResNet model. Channels are grouped:
//!
//! - 0..3:   self hand (thresholded tile counts)
//! - 4..15:  discards per player (presence, tedashi, temporal)
//! - 16..27: melds per player (open, closed kan, kakan)
//! - 28..31: dora info (indicators, actual dora, aka, ura)
//! - 32..61: game metadata (winds, riichi, scores, shanten, etc.)
//! - 62..84: safety channels (genbutsu, suji, kabe, one-chance, tenpai)

use crate::safety::SafetyInfo;
use crate::tile::NUM_TILE_TYPES;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Total observation channels.
pub const NUM_CHANNELS: usize = 85;

/// Tiles per channel (one per tile type).
pub const NUM_TILES: usize = NUM_TILE_TYPES; // 34

/// Total elements in the flat observation buffer.
pub const OBS_SIZE: usize = NUM_CHANNELS * NUM_TILES; // 2890

// -- Channel group starts --

const CH_HAND: usize = 0; // 0..3   (4 channels)
const CH_DISCARDS: usize = 4; // 4..15  (12 channels: 3 per player)
const CH_MELDS: usize = 16; // 16..27 (12 channels: 3 per player)
const CH_DORA: usize = 28; // 28..31 (4 channels)
const CH_META: usize = 32; // 32..61 (30 channels)
const CH_SAFETY: usize = 62; // 62..84 (23 channels)

/// Number of players at the table.
const NUM_PLAYERS: usize = 4;

// ---------------------------------------------------------------------------
// ObservationEncoder
// ---------------------------------------------------------------------------

/// Pre-allocated encoder buffer for the 85x34 observation tensor.
///
/// Reuse across turns to avoid per-turn allocation. Call [`clear`] then
/// the individual `encode_*` methods, or use [`encode`] as the one-shot
/// entry point.
#[derive(Clone)]
pub struct ObservationEncoder {
    /// Flat buffer: 85 channels x 34 tiles, row-major.
    buffer: [f32; OBS_SIZE],
}

impl ObservationEncoder {
    /// Create a new encoder with a zeroed buffer.
    pub fn new() -> Self {
        Self {
            buffer: [0.0; OBS_SIZE],
        }
    }

    /// Zero the entire buffer.
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
    }

    /// Read-only view of the flat observation buffer.
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
}

impl Default for ObservationEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Encoding: self hand (channels 0-3)
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode the observer's hand tile counts into channels 0-3.
    ///
    /// Binary thresholded planes:
    /// - Ch 0: count >= 1
    /// - Ch 1: count >= 2
    /// - Ch 2: count >= 3
    /// - Ch 3: count == 4
    pub fn encode_hand(&mut self, hand_counts: &[u8; NUM_TILES], drawn_tile: Option<u8>) {
        for (tile, &count) in hand_counts.iter().enumerate() {
            if count >= 1 { self.set(CH_HAND, tile, 1.0); }
            if count >= 2 { self.set(CH_HAND + 1, tile, 1.0); }
            if count >= 3 { self.set(CH_HAND + 2, tile, 1.0); }
            if count == 4 { self.set(CH_HAND + 3, tile, 1.0); }
        }
        // Channel 56 (CH_META + 24): drawn tile indicator (one-hot for tsumo tile)
        if let Some(dt) = drawn_tile {
            let t = dt as usize;
            if t < NUM_TILES {
                self.set(56, t, 1.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Discard info input type
// ---------------------------------------------------------------------------

/// A single discard event for encoding.
#[derive(Debug, Clone, Copy)]
pub struct DiscardEntry {
    /// Tile type (0-33).
    pub tile: u8,
    /// True if discarded from hand (not tsumogiri).
    pub is_tedashi: bool,
    /// 0-based turn index when this discard happened.
    pub turn: u16,
}

/// Per-player discard history for encoding.
#[derive(Debug, Clone)]
pub struct PlayerDiscards {
    /// Ordered list of discards (oldest first).
    pub discards: Vec<DiscardEntry>,
}

// ---------------------------------------------------------------------------
// Encoding: discards (channels 4-15)
// ---------------------------------------------------------------------------

/// Temporal decay factor for discard recency weighting.
const DISCARD_DECAY: f32 = 0.2;

impl ObservationEncoder {
    /// Encode discard info for all 4 players into channels 4-15.
    ///
    /// Per player (3 channels each):
    /// - presence:  binary 1.0 if tile was discarded by this player
    /// - tedashi:   binary 1.0 if that discard was from hand (not tsumogiri)
    /// - temporal:  exp(-0.2 * (t_max - t_discard))
    pub fn encode_discards(&mut self, discards: &[PlayerDiscards; NUM_PLAYERS]) {
        for (p, pd) in discards.iter().enumerate() {
            let ch_base = CH_DISCARDS + 3 * p;
            // Find max turn for temporal weighting
            let t_max = pd.discards.iter().map(|d| d.turn).max().unwrap_or(0);
            for d in &pd.discards {
                let t = d.tile as usize;
                if t >= NUM_TILES { continue; }
                // Presence
                self.set(ch_base, t, 1.0);
                // Tedashi
                if d.is_tedashi {
                    self.set(ch_base + 1, t, 1.0);
                }
                // Temporal weight (most recent = 1.0, older decays)
                let dt = (t_max - d.turn) as f32;
                let w = (-DISCARD_DECAY * dt).exp();
                // Keep the highest weight if a tile was discarded multiple times
                let idx = (ch_base + 2) * NUM_TILES + t;
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
    /// Chi, pon, or daiminkan (open).
    Open,
    /// Closed kan (ankan).
    ClosedKan,
    /// Added kan (kakan) -- the single added tile.
    Kakan,
}

/// A single meld for encoding.
#[derive(Debug, Clone)]
pub struct MeldInfo {
    /// Tile types present in the meld (0-33 each).
    pub tiles: Vec<u8>,
    /// What kind of meld this is.
    pub meld_type: MeldType,
}

// ---------------------------------------------------------------------------
// Encoding: melds (channels 16-27)
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode melds for all 4 players into channels 16-27.
    ///
    /// Per player (3 channels each):
    /// - open meld tiles (chi/pon/daiminkan)
    /// - closed kan tiles
    /// - kakan (added tile)
    pub fn encode_melds(&mut self, melds: &[Vec<MeldInfo>; NUM_PLAYERS]) {
        for (p, player_melds) in melds.iter().enumerate() {
            let ch_base = CH_MELDS + 3 * p;
            for meld in player_melds {
                let ch_offset = match meld.meld_type {
                    MeldType::Open => 0,
                    MeldType::ClosedKan => 1,
                    MeldType::Kakan => 2,
                };
                for &tile in &meld.tiles {
                    let t = tile as usize;
                    if t < NUM_TILES {
                        self.set(ch_base + ch_offset, t, 1.0);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding: dora (channels 28-31)
// ---------------------------------------------------------------------------

/// Dora information for encoding.
#[derive(Debug, Clone)]
pub struct DoraInfo {
    /// Dora indicator tile types (0-33). Up to 5 kan dora.
    pub indicators: Vec<u8>,
    /// Aka dora flags: `[has_aka_5m, has_aka_5p, has_aka_5s]`.
    pub aka_flags: [bool; 3],
}

/// Map a dora indicator tile type to the actual dora tile type.
/// Wraps within suit: 9m indicator -> 1m dora, 9p -> 1p, 9s -> 1s.
/// Wraps within winds (E-S-W-N) and dragons (haku-hatsu-chun).
fn indicator_to_dora(indicator: u8) -> u8 {
    let i = indicator as usize;
    if i >= NUM_TILES { return 0; }
    match i {
        // Suited tiles: next in suit, wrap 9->1
        0..27 => {
            let suit_start = (i / 9) * 9;
            let num = i - suit_start;
            (suit_start + (num + 1) % 9) as u8
        }
        // Winds: E->S->W->N->E (27-30)
        27..31 => ((i - 27 + 1) % 4 + 27) as u8,
        // Dragons: haku->hatsu->chun->haku (31-33)
        31..34 => ((i - 31 + 1) % 3 + 31) as u8,
        _ => 0,
    }
}

/// Aka-dora tile type indices (5m=4, 5p=13, 5s=22).
const AKA_TILE_TYPES: [usize; 3] = [4, 13, 22];

impl ObservationEncoder {
    /// Encode dora information into channels 28-31.
    ///
    /// - Ch 28: dora indicator tiles (binary)
    /// - Ch 29: actual dora tiles (indicator + 1, the tiles that score)
    /// - Ch 30: aka dora flags (red fives)
    /// - Ch 31: ura dora (all zeros during play, filled post-win)
    pub fn encode_dora(&mut self, dora: &DoraInfo) {
        for &ind in &dora.indicators {
            let i = ind as usize;
            if i < NUM_TILES {
                self.set(CH_DORA, i, 1.0);
                let actual = indicator_to_dora(ind) as usize;
                self.set(CH_DORA + 1, actual, 1.0);
            }
        }
        // Aka dora flags
        for (slot, &has_aka) in dora.aka_flags.iter().enumerate() {
            if has_aka {
                self.set(CH_DORA + 2, AKA_TILE_TYPES[slot], 1.0);
            }
        }
        // Ch 31 (ura dora) stays zero during play
    }
}

// ---------------------------------------------------------------------------
// Game metadata input type
// ---------------------------------------------------------------------------

/// Game metadata for encoding channels 32-61.
#[derive(Debug, Clone)]
pub struct GameMetadata {
    /// Round wind: 0=East, 1=South, 2=West, 3=North.
    pub round_wind: u8,
    /// Observer's seat wind: 0=East, 1=South, 2=West, 3=North.
    pub seat_wind: u8,
    /// True if observer is the dealer.
    pub is_dealer: bool,
    /// Riichi status for all 4 players (relative to observer). Index 0 = self.
    pub riichi: [bool; NUM_PLAYERS],
    /// Honba (repeat) counter.
    pub honba: u8,
    /// Number of riichi sticks deposited on the table.
    pub riichi_sticks: u8,
    /// Tiles remaining in the wall.
    pub tiles_remaining: u8,
    /// Scores for all 4 players (relative to observer). Raw point values.
    pub scores: [i32; NUM_PLAYERS],
}

// ---------------------------------------------------------------------------
// Encoding: game metadata (channels 32-61)
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode game metadata into channels 32-61.
    ///
    /// Layout:
    /// - Ch 32-35: round wind one-hot
    /// - Ch 36-39: seat wind one-hot
    /// - Ch 40: dealer flag
    /// - Ch 41: self riichi
    /// - Ch 42-44: opponent riichi
    /// - Ch 45: honba (normalized / 8.0)
    /// - Ch 46: riichi sticks (normalized / 4.0)
    /// - Ch 47: wall remaining (normalized / 70.0)
    /// - Ch 48-51: player scores (normalized / 100000.0)
    /// - Ch 52-61: reserved (zeros)
    pub fn encode_metadata(&mut self, meta: &GameMetadata) {
        // Round wind one-hot (ch 32-35)
        let rw = meta.round_wind as usize;
        if rw < 4 { self.fill_channel(CH_META + rw, 1.0); }

        // Seat wind one-hot (ch 36-39)
        let sw = meta.seat_wind as usize;
        if sw < 4 { self.fill_channel(CH_META + 4 + sw, 1.0); }

        // Dealer flag (ch 40)
        if meta.is_dealer { self.fill_channel(CH_META + 8, 1.0); }

        // Riichi flags (ch 41 = self, ch 42-44 = opponents)
        for (i, &r) in meta.riichi.iter().enumerate() {
            if r && i < NUM_PLAYERS {
                self.fill_channel(CH_META + 9 + i, 1.0);
            }
        }

        // Honba normalized (ch 45)
        self.fill_channel(CH_META + 13, meta.honba as f32 / 8.0);

        // Riichi sticks normalized (ch 46)
        self.fill_channel(CH_META + 14, meta.riichi_sticks as f32 / 4.0);

        // Wall remaining normalized (ch 47)
        self.fill_channel(CH_META + 15, meta.tiles_remaining as f32 / 70.0);

        // Player scores normalized (ch 48-51)
        for (i, &score) in meta.scores.iter().enumerate() {
            self.fill_channel(CH_META + 16 + i, score as f32 / 100_000.0);
        }
        // Ch 52-61 reserved (already zero from clear)
    }
}

// ---------------------------------------------------------------------------
// Encoding: safety channels (62-84)
// ---------------------------------------------------------------------------

/// Number of opponents for safety channels.
const NUM_OPPS: usize = crate::safety::NUM_OPPONENTS; // 3

impl ObservationEncoder {
    /// Encode safety info into channels 62-84 (23 channels total).
    ///
    /// Layout:
    /// - Ch 62-64: genbutsu_all (per opponent)
    /// - Ch 65-67: genbutsu_tedashi (per opponent)
    /// - Ch 68-70: genbutsu_riichi_era (per opponent)
    /// - Ch 71-73: suji (per opponent, float 0.0-1.0)
    /// - Ch 74-79: reserved suji context (zeros)
    /// - Ch 80: kabe
    /// - Ch 81: one-chance
    /// - Ch 82-84: reserved tenpai hints (zeros)
    pub fn encode_safety(&mut self, safety: &SafetyInfo) {
        for opp in 0..NUM_OPPS {
            for tile in 0..NUM_TILES {
                // Genbutsu (ch 62-64)
                if safety.genbutsu_all[opp][tile] {
                    self.set(CH_SAFETY + opp, tile, 1.0);
                }
                // Tedashi genbutsu (ch 65-67)
                if safety.genbutsu_tedashi[opp][tile] {
                    self.set(CH_SAFETY + NUM_OPPS + opp, tile, 1.0);
                }
                // Riichi-era genbutsu (ch 68-70)
                if safety.genbutsu_riichi_era[opp][tile] {
                    self.set(CH_SAFETY + 2 * NUM_OPPS + opp, tile, 1.0);
                }
                // Suji scores (ch 71-73)
                let suji = safety.suji[opp][tile];
                if suji > 0.0 {
                    self.set(CH_SAFETY + 3 * NUM_OPPS + opp, tile, suji);
                }
            }
        }
        // Kabe (ch 80) and one-chance (ch 81)
        for tile in 0..NUM_TILES {
            if safety.kabe[tile] {
                self.set(CH_SAFETY + 18, tile, 1.0);
            }
            if safety.one_chance[tile] {
                self.set(CH_SAFETY + 19, tile, 1.0);
            }
        }
        // Ch 74-79, 82-84 reserved (already zero)
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
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        hand_counts: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[Vec<MeldInfo>; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
    ) -> &[f32; OBS_SIZE] {
        self.clear();
        self.encode_hand(hand_counts, drawn_tile);
        self.encode_discards(discards);
        self.encode_melds(melds);
        self.encode_dora(dora);
        self.encode_metadata(meta);
        self.encode_safety(safety);
        self.as_slice()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: read a single cell from the obs buffer.
    fn get(enc: &ObservationEncoder, ch: usize, tile: usize) -> f32 {
        enc.as_slice()[ch * NUM_TILES + tile]
    }

    #[test]
    fn new_encoder_is_zeroed() {
        let enc = ObservationEncoder::new();
        assert!(enc.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn clear_resets_buffer() {
        let mut enc = ObservationEncoder::new();
        enc.buffer[0] = 42.0;
        enc.buffer[OBS_SIZE - 1] = 99.0;
        enc.clear();
        assert!(enc.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn hand_single_tile() {
        let mut enc = ObservationEncoder::new();
        let mut hand = [0u8; NUM_TILES];
        hand[0] = 1; // one copy of 1m
        enc.encode_hand(&hand, None);
        assert_eq!(get(&enc, 0, 0), 1.0); // ch0: count >= 1
        assert_eq!(get(&enc, 1, 0), 0.0); // ch1: count >= 2
    }

    #[test]
    fn hand_four_copies() {
        let mut enc = ObservationEncoder::new();
        let mut hand = [0u8; NUM_TILES];
        hand[5] = 4; // four copies of 6m
        enc.encode_hand(&hand, None);
        assert_eq!(get(&enc, 0, 5), 1.0); // >= 1
        assert_eq!(get(&enc, 1, 5), 1.0); // >= 2
        assert_eq!(get(&enc, 2, 5), 1.0); // >= 3
        assert_eq!(get(&enc, 3, 5), 1.0); // == 4
    }

    #[test]
    fn hand_three_copies_not_four() {
        let mut enc = ObservationEncoder::new();
        let mut hand = [0u8; NUM_TILES];
        hand[10] = 3;
        enc.encode_hand(&hand, None);
        assert_eq!(get(&enc, 2, 10), 1.0); // >= 3
        assert_eq!(get(&enc, 3, 10), 0.0); // == 4 should be off
    }

    #[test]
    fn drawn_tile_encoded() {
        let mut enc = ObservationEncoder::new();
        let hand = [0u8; NUM_TILES];
        enc.encode_hand(&hand, Some(5)); // drew 6m
        assert_eq!(get(&enc, 56, 5), 1.0); // channel 56, tile 5
        assert_eq!(get(&enc, 56, 0), 0.0); // other tiles zero
    }

    #[test]
    fn drawn_tile_none_leaves_channel_zero() {
        let mut enc = ObservationEncoder::new();
        let hand = [0u8; NUM_TILES];
        enc.encode_hand(&hand, None);
        for t in 0..NUM_TILES {
            assert_eq!(get(&enc, 56, t), 0.0);
        }
    }

    // -- Discard tests --

    fn empty_discards() -> [PlayerDiscards; NUM_PLAYERS] {
        [
            PlayerDiscards { discards: vec![] },
            PlayerDiscards { discards: vec![] },
            PlayerDiscards { discards: vec![] },
            PlayerDiscards { discards: vec![] },
        ]
    }

    #[test]
    fn discard_presence_and_tedashi() {
        let mut enc = ObservationEncoder::new();
        let mut discards = empty_discards();
        discards[0].discards.push(DiscardEntry { tile: 5, is_tedashi: true, turn: 0 });
        discards[1].discards.push(DiscardEntry { tile: 10, is_tedashi: false, turn: 0 });
        enc.encode_discards(&discards);
        // Player 0: ch_base=4, presence=ch4, tedashi=ch5
        assert_eq!(get(&enc, 4, 5), 1.0);  // presence
        assert_eq!(get(&enc, 5, 5), 1.0);  // tedashi
        // Player 1: ch_base=7, presence=ch7, tedashi=ch8
        assert_eq!(get(&enc, 7, 10), 1.0); // presence
        assert_eq!(get(&enc, 8, 10), 0.0); // NOT tedashi (tsumogiri)
    }

    #[test]
    fn discard_temporal_decay() {
        let mut enc = ObservationEncoder::new();
        let mut discards = empty_discards();
        // Two discards by player 0 at different turns
        discards[0].discards.push(DiscardEntry { tile: 0, is_tedashi: false, turn: 0 });
        discards[0].discards.push(DiscardEntry { tile: 1, is_tedashi: false, turn: 5 });
        enc.encode_discards(&discards);
        // Tile at turn 5 (t_max=5): weight = exp(-0.2*(5-5)) = 1.0
        assert!((get(&enc, 6, 1) - 1.0).abs() < 1e-6);
        // Tile at turn 0: weight = exp(-0.2*(5-0)) = exp(-1.0)
        let expected = (-1.0f32).exp();
        assert!((get(&enc, 6, 0) - expected).abs() < 1e-6);
    }

    // -- Meld tests --

    fn empty_melds() -> [Vec<MeldInfo>; NUM_PLAYERS] {
        [vec![], vec![], vec![], vec![]]
    }

    #[test]
    fn meld_open_chi() {
        let mut enc = ObservationEncoder::new();
        let mut melds = empty_melds();
        melds[0].push(MeldInfo {
            tiles: vec![0, 1, 2], // 1m-2m-3m chi
            meld_type: MeldType::Open,
        });
        enc.encode_melds(&melds);
        // Player 0 open meld = ch 16
        assert_eq!(get(&enc, 16, 0), 1.0);
        assert_eq!(get(&enc, 16, 1), 1.0);
        assert_eq!(get(&enc, 16, 2), 1.0);
        assert_eq!(get(&enc, 17, 0), 0.0); // closed kan channel empty
    }

    #[test]
    fn meld_closed_kan() {
        let mut enc = ObservationEncoder::new();
        let mut melds = empty_melds();
        melds[2].push(MeldInfo {
            tiles: vec![27, 27, 27, 27], // East ankan by player 2
            meld_type: MeldType::ClosedKan,
        });
        enc.encode_melds(&melds);
        // Player 2: ch_base = 16 + 3*2 = 22, closed_kan = ch 23
        assert_eq!(get(&enc, 23, 27), 1.0);
    }

    // -- Dora tests --

    #[test]
    fn dora_indicator_and_actual() {
        let mut enc = ObservationEncoder::new();
        let dora = DoraInfo {
            indicators: vec![0], // 1m indicator -> 2m is dora
            aka_flags: [false, false, false],
        };
        enc.encode_dora(&dora);
        assert_eq!(get(&enc, 28, 0), 1.0); // indicator on ch 28
        assert_eq!(get(&enc, 29, 1), 1.0); // actual dora (2m) on ch 29
    }

    #[test]
    fn dora_9m_wraps_to_1m() {
        // indicator 9m (idx 8) -> dora is 1m (idx 0)
        assert_eq!(indicator_to_dora(8), 0);
    }

    #[test]
    fn dora_wind_wrap() {
        // North (30) -> East (27)
        assert_eq!(indicator_to_dora(30), 27);
        // East (27) -> South (28)
        assert_eq!(indicator_to_dora(27), 28);
    }

    #[test]
    fn dora_dragon_wrap() {
        // Chun (33) -> Haku (31)
        assert_eq!(indicator_to_dora(33), 31);
        // Haku (31) -> Hatsu (32)
        assert_eq!(indicator_to_dora(31), 32);
    }

    #[test]
    fn aka_dora_flags() {
        let mut enc = ObservationEncoder::new();
        let dora = DoraInfo {
            indicators: vec![],
            aka_flags: [true, false, true], // has red 5m and red 5s
        };
        enc.encode_dora(&dora);
        assert_eq!(get(&enc, 30, 4), 1.0);  // 5m on ch30
        assert_eq!(get(&enc, 30, 13), 0.0); // 5p NOT set
        assert_eq!(get(&enc, 30, 22), 1.0); // 5s on ch30
    }

    // -- Metadata tests --

    fn test_metadata() -> GameMetadata {
        GameMetadata {
            round_wind: 0, // East
            seat_wind: 1,  // South
            is_dealer: false,
            riichi: [true, false, false, false],
            honba: 2,
            riichi_sticks: 1,
            tiles_remaining: 35,
            scores: [25000, 25000, 25000, 25000],
        }
    }

    #[test]
    fn metadata_winds_one_hot() {
        let mut enc = ObservationEncoder::new();
        enc.encode_metadata(&test_metadata());
        // Round wind: East (idx 0) -> ch 32 filled with 1.0
        assert_eq!(get(&enc, 32, 0), 1.0);
        assert_eq!(get(&enc, 33, 0), 0.0); // South channel empty
        // Seat wind: South (idx 1) -> ch 37 filled with 1.0
        assert_eq!(get(&enc, 37, 0), 1.0);
    }

    #[test]
    fn metadata_riichi_and_scores() {
        let mut enc = ObservationEncoder::new();
        enc.encode_metadata(&test_metadata());
        // Self riichi = ch 41 (CH_META + 9)
        assert_eq!(get(&enc, 41, 0), 1.0);
        // Opponent 1 riichi = ch 42 -- NOT set
        assert_eq!(get(&enc, 42, 0), 0.0);
        // Honba = 2/8 = 0.25 on ch 45
        assert!((get(&enc, 45, 0) - 0.25).abs() < 1e-6);
        // Score ch 48 = 25000/100000 = 0.25
        assert!((get(&enc, 48, 0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn metadata_wall_remaining() {
        let mut enc = ObservationEncoder::new();
        enc.encode_metadata(&test_metadata());
        // tiles_remaining=35, normalized=35/70=0.5 on ch 47
        assert!((get(&enc, 47, 0) - 0.5).abs() < 1e-6);
    }

    // -- Safety tests --

    #[test]
    fn safety_genbutsu_channels() {
        let mut enc = ObservationEncoder::new();
        let mut si = SafetyInfo::new();
        si.genbutsu_all[0][5] = true;
        si.genbutsu_tedashi[1][10] = true;
        si.genbutsu_riichi_era[2][20] = true;
        enc.encode_safety(&si);
        // genbutsu_all: ch 62 + opp
        assert_eq!(get(&enc, 62, 5), 1.0);  // opp 0
        // tedashi: ch 65 + opp
        assert_eq!(get(&enc, 66, 10), 1.0); // opp 1, ch 65+1=66
        // riichi_era: ch 68 + opp
        assert_eq!(get(&enc, 70, 20), 1.0); // opp 2, ch 68+2=70
    }

    #[test]
    fn safety_suji_channel() {
        let mut enc = ObservationEncoder::new();
        let mut si = SafetyInfo::new();
        si.suji[0][0] = 1.0; // 1m has suji for opp 0
        enc.encode_safety(&si);
        // suji: ch 71 + opp
        assert_eq!(get(&enc, 71, 0), 1.0);
    }

    #[test]
    fn safety_kabe_and_one_chance() {
        let mut enc = ObservationEncoder::new();
        let mut si = SafetyInfo::new();
        si.kabe[15] = true;
        si.one_chance[20] = true;
        enc.encode_safety(&si);
        // kabe = ch 80 (62+18), one_chance = ch 81 (62+19)
        assert_eq!(get(&enc, 80, 15), 1.0);
        assert_eq!(get(&enc, 81, 20), 1.0);
    }

    // -- Full encode test --

    #[test]
    fn full_encode_returns_correct_size() {
        let mut enc = ObservationEncoder::new();
        let hand = [0u8; NUM_TILES];
        let discards = empty_discards();
        let melds = empty_melds();
        let dora = DoraInfo { indicators: vec![], aka_flags: [false; 3] };
        let meta = test_metadata();
        let safety = SafetyInfo::new();
        let obs = enc.encode(&hand, None, &discards, &melds, &dora, &meta, &safety);
        assert_eq!(obs.len(), OBS_SIZE);
    }

    #[test]
    fn full_encode_clears_between_calls() {
        let mut enc = ObservationEncoder::new();
        // First encode with some hand data
        let mut hand = [0u8; NUM_TILES];
        hand[0] = 3;
        let discards = empty_discards();
        let melds = empty_melds();
        let dora = DoraInfo { indicators: vec![], aka_flags: [false; 3] };
        let meta = test_metadata();
        let safety = SafetyInfo::new();
        enc.encode(&hand, None, &discards, &melds, &dora, &meta, &safety);
        assert_eq!(get(&enc, 2, 0), 1.0); // ch2 tile 0 set

        // Second encode with empty hand -- should be cleared
        let empty_hand = [0u8; NUM_TILES];
        enc.encode(&empty_hand, None, &discards, &melds, &dora, &meta, &safety);
        assert_eq!(get(&enc, 2, 0), 0.0); // cleared
    }
}
