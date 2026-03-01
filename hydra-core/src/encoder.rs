//! 85x34 observation tensor encoder for neural network input.
//!
//! Encodes the full game state into a flat `[f32; 85 * 34]` array (row-major)
//! that serves as input to the Hydra SE-ResNet model. Channels are grouped:
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

use crate::safety::SafetyInfo;
use crate::tile::NUM_TILE_TYPES;
use riichienv_core::shanten::calc_shanten_from_counts;

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
const CH_OPEN_MELD: usize = 4; // 4..7   (4 channels)
const CH_DRAWN: usize = 8; // 8      (1 channel)
const CH_SHANTEN_MASK: usize = 9; // 9..10  (2 channels)
const CH_DISCARDS: usize = 11; // 11..22 (12 channels: 3 per player)
const CH_MELDS: usize = 23; // 23..34 (12 channels: 3 per player)
const CH_DORA: usize = 35; // 35..39 (5 channels)
const CH_AKA: usize = 40; // 40..42 (3 channels)
const CH_META: usize = 43; // 43..61 (19 channels)
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
    pub fn encode_hand(&mut self, hand_counts: &[u8; NUM_TILES]) {
        for (tile, &count) in hand_counts.iter().enumerate() {
            if count >= 1 { self.set(CH_HAND, tile, 1.0); }
            if count >= 2 { self.set(CH_HAND + 1, tile, 1.0); }
            if count >= 3 { self.set(CH_HAND + 2, tile, 1.0); }
            if count == 4 { self.set(CH_HAND + 3, tile, 1.0); }
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
    pub fn encode_open_meld_hand(&mut self, counts: &[u8; NUM_TILES]) {
        for (tile, &count) in counts.iter().enumerate() {
            if count >= 1 { self.set(CH_OPEN_MELD, tile, 1.0); }
            if count >= 2 { self.set(CH_OPEN_MELD + 1, tile, 1.0); }
            if count >= 3 { self.set(CH_OPEN_MELD + 2, tile, 1.0); }
            if count == 4 { self.set(CH_OPEN_MELD + 3, tile, 1.0); }
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding: drawn tile (channel 8)
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode the drawn tile as a one-hot on channel 8.
    /// `None` means no tile was drawn (e.g. first turn or after a call).
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
    pub fn encode_shanten_masks(&mut self, hand: &[u8; NUM_TILES]) {
        let total: u8 = hand.iter().sum();
        let len_div3 = total / 3;
        let base = calc_shanten_from_counts(hand, len_div3);
        let mut tmp = *hand;
        for tile in 0..NUM_TILES {
            if tmp[tile] == 0 { continue; }
            tmp[tile] -= 1;
            let after_len_div3 = (total - 1) / 3;
            let after = calc_shanten_from_counts(&tmp, after_len_div3);
            if after <= base {
                self.set(CH_SHANTEN_MASK, tile, 1.0);
            }
            if after < base {
                self.set(CH_SHANTEN_MASK + 1, tile, 1.0);
            }
            tmp[tile] += 1;
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
// Encoding: discards (channels 11-22)
// ---------------------------------------------------------------------------

/// Temporal decay factor for discard recency weighting.
const DISCARD_DECAY: f32 = 0.2;

impl ObservationEncoder {
    /// Encode discard info for all 4 players into channels 11-22.
    ///
    /// Per player (3 channels each):
    /// - presence:  binary 1.0 if tile was discarded by this player
    /// - tedashi:   binary 1.0 if that discard was from hand (not tsumogiri)
    /// - temporal:  exp(-0.2 * (t_max - t_discard))
    pub fn encode_discards(&mut self, discards: &[PlayerDiscards; NUM_PLAYERS]) {
        for (p, pd) in discards.iter().enumerate() {
            let ch_base = CH_DISCARDS + 3 * p;
            let t_max = pd.discards.iter().map(|d| d.turn).max().unwrap_or(0);
            for d in &pd.discards {
                let t = d.tile as usize;
                if t >= NUM_TILES { continue; }
                self.set(ch_base, t, 1.0);
                if d.is_tedashi {
                    self.set(ch_base + 1, t, 1.0);
                }
                let dt = (t_max - d.turn) as f32;
                let w = (-DISCARD_DECAY * dt).exp();
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
    /// Chi (sequence call).
    Chi,
    /// Pon (triplet call).
    Pon,
    /// Kan (any kan: ankan, daiminkan, kakan).
    Kan,
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
// Encoding: melds (channels 23-34)
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode melds for all 4 players into channels 23-34.
    ///
    /// Per player (3 channels each):
    /// - chi tiles
    /// - pon tiles
    /// - kan tiles
    pub fn encode_melds(&mut self, melds: &[Vec<MeldInfo>; NUM_PLAYERS]) {
        for (p, player_melds) in melds.iter().enumerate() {
            let ch_base = CH_MELDS + 3 * p;
            for meld in player_melds {
                let ch_offset = match meld.meld_type {
                    MeldType::Chi => 0,
                    MeldType::Pon => 1,
                    MeldType::Kan => 2,
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
// Encoding: dora (channels 35-39) and aka (channels 40-42)
// ---------------------------------------------------------------------------

/// Dora information for encoding.
#[derive(Debug, Clone)]
pub struct DoraInfo {
    /// Dora indicator tile types (0-33). Up to 5 kan dora.
    pub indicators: Vec<u8>,
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
    pub fn encode_dora(&mut self, dora: &DoraInfo) {
        let mut counts = [0u8; NUM_TILES];
        for &ind in &dora.indicators {
            let i = ind as usize;
            if i < NUM_TILES {
                counts[i] = counts[i].saturating_add(1);
            }
        }
        for (tile, &c) in counts.iter().enumerate() {
            if c >= 1 { self.set(CH_DORA, tile, 1.0); }
            if c >= 2 { self.set(CH_DORA + 1, tile, 1.0); }
            if c >= 3 { self.set(CH_DORA + 2, tile, 1.0); }
            if c >= 4 { self.set(CH_DORA + 3, tile, 1.0); }
            if c >= 5 { self.set(CH_DORA + 4, tile, 1.0); }
        }
    }

    /// Encode aka dora flags into channels 40-42 (one plane per suit).
    ///
    /// Each channel is fully filled with 1.0 if the corresponding aka is present.
    /// - Ch 40: has red 5m
    /// - Ch 41: has red 5p
    /// - Ch 42: has red 5s
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
#[derive(Debug, Clone)]
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
    pub fn encode_metadata(&mut self, meta: &GameMetadata) {
        // Riichi flags (ch 43-46)
        for (i, &r) in meta.riichi.iter().enumerate() {
            if r { self.fill_channel(CH_META + i, 1.0); }
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
                if safety.genbutsu_all[opp][tile] {
                    self.set(CH_SAFETY + opp, tile, 1.0);
                }
                if safety.genbutsu_tedashi[opp][tile] {
                    self.set(CH_SAFETY + NUM_OPPS + opp, tile, 1.0);
                }
                if safety.genbutsu_riichi_era[opp][tile] {
                    self.set(CH_SAFETY + 2 * NUM_OPPS + opp, tile, 1.0);
                }
                let suji = safety.suji[opp][tile];
                if suji > 0.0 {
                    self.set(CH_SAFETY + 3 * NUM_OPPS + opp, tile, suji);
                }
            }
        }
        for tile in 0..NUM_TILES {
            if safety.kabe[tile] {
                self.set(CH_SAFETY + 18, tile, 1.0);
            }
            if safety.one_chance[tile] {
                self.set(CH_SAFETY + 19, tile, 1.0);
            }
        }
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
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[Vec<MeldInfo>; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
    ) -> &[f32; OBS_SIZE] {
        self.clear();
        self.encode_hand(hand);
        self.encode_open_meld_hand(open_meld_counts);
        self.encode_drawn_tile(drawn_tile);
        self.encode_shanten_masks(hand);
        self.encode_discards(discards);
        self.encode_melds(melds);
        self.encode_dora(dora);
        self.encode_aka(dora);
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

    // -- Hand tests (ch 0-3) --

    #[test]
    fn hand_single_tile() {
        let mut enc = ObservationEncoder::new();
        let mut hand = [0u8; NUM_TILES];
        hand[0] = 1;
        enc.encode_hand(&hand);
        assert_eq!(get(&enc, 0, 0), 1.0);
        assert_eq!(get(&enc, 1, 0), 0.0);
    }

    #[test]
    fn hand_four_copies() {
        let mut enc = ObservationEncoder::new();
        let mut hand = [0u8; NUM_TILES];
        hand[5] = 4;
        enc.encode_hand(&hand);
        assert_eq!(get(&enc, 0, 5), 1.0);
        assert_eq!(get(&enc, 1, 5), 1.0);
        assert_eq!(get(&enc, 2, 5), 1.0);
        assert_eq!(get(&enc, 3, 5), 1.0);
    }

    #[test]
    fn hand_three_copies_not_four() {
        let mut enc = ObservationEncoder::new();
        let mut hand = [0u8; NUM_TILES];
        hand[10] = 3;
        enc.encode_hand(&hand);
        assert_eq!(get(&enc, 2, 10), 1.0);
        assert_eq!(get(&enc, 3, 10), 0.0);
    }

    // -- Open meld hand tests (ch 4-7) --

    #[test]
    fn open_meld_hand_thermometer() {
        let mut enc = ObservationEncoder::new();
        let mut counts = [0u8; NUM_TILES];
        counts[0] = 3; // 3 tiles of type 0 from melds
        counts[9] = 1; // 1 tile of type 9
        enc.encode_open_meld_hand(&counts);
        // tile 0: ch4=1, ch5=1, ch6=1, ch7=0
        assert_eq!(get(&enc, 4, 0), 1.0);
        assert_eq!(get(&enc, 5, 0), 1.0);
        assert_eq!(get(&enc, 6, 0), 1.0);
        assert_eq!(get(&enc, 7, 0), 0.0);
        // tile 9: ch4=1, ch5=0
        assert_eq!(get(&enc, 4, 9), 1.0);
        assert_eq!(get(&enc, 5, 9), 0.0);
    }

    #[test]
    fn open_meld_hand_four() {
        let mut enc = ObservationEncoder::new();
        let mut counts = [0u8; NUM_TILES];
        counts[27] = 4; // kan of East
        enc.encode_open_meld_hand(&counts);
        assert_eq!(get(&enc, 7, 27), 1.0);
    }

    // -- Drawn tile tests (ch 8) --

    #[test]
    fn drawn_tile_encoded() {
        let mut enc = ObservationEncoder::new();
        enc.encode_drawn_tile(Some(5));
        assert_eq!(get(&enc, 8, 5), 1.0);
        assert_eq!(get(&enc, 8, 0), 0.0);
    }

    #[test]
    fn drawn_tile_none_leaves_channel_zero() {
        let mut enc = ObservationEncoder::new();
        enc.encode_drawn_tile(None);
        for t in 0..NUM_TILES {
            assert_eq!(get(&enc, 8, t), 0.0);
        }
    }

    // -- Shanten mask tests (ch 9-10) --

    #[test]
    fn shanten_masks_complete_hand() {
        // Complete 14-tile hand: 123m 456m 789m 123p 11s
        // 4 sequences + 1 pair = agari (shanten = -1)
        let mut enc = ObservationEncoder::new();
        let mut hand = [0u8; NUM_TILES];
        hand[..9].fill(1); // 1-9m
        hand[9] = 1; hand[10] = 1; hand[11] = 1; // 1-3p
        hand[18] = 2; // 1s pair
        // 14 tiles, len_div3=4, shanten=-1
        enc.encode_shanten_masks(&hand);
        // After discarding any tile, shanten goes from -1 to 0 (worsens).
        // So next-shanten (ch10) should have NO tiles set.
        for t in 0..NUM_TILES {
            assert_eq!(get(&enc, 10, t), 0.0);
        }
    }

    #[test]
    fn shanten_masks_one_away() {
        // Simple iishanten hand: 1m,2m,3m, 4m,5m,6m, 7m,8m,9m, 1p,1p,1p, 2p, drawn 5s
        let mut enc = ObservationEncoder::new();
        let mut hand = [0u8; NUM_TILES];
        hand[0] = 1; hand[1] = 1; hand[2] = 1; // 123m
        hand[3] = 1; hand[4] = 1; hand[5] = 1; // 456m
        hand[6] = 1; hand[7] = 1; hand[8] = 1; // 789m
        hand[9] = 3; // 1p x3
        hand[10] = 1; // 2p
        hand[22] = 1; // 5s (drawn tile)
        // 14 tiles. This is tenpai (waiting on 2p or 5s-related).
        // Actually 123m 456m 789m 111p + 2p 5s = tenpai waiting on 3p
        // shanten = 0 (tenpai)
        enc.encode_shanten_masks(&hand);
        // Discarding 2p or 5s keeps tenpai (shanten stays 0), so ch9 should be set
        // The exact tiles depend on shanten calc, but at minimum some tiles on ch9
        let ch9_sum: f32 = (0..NUM_TILES).map(|t| get(&enc, 9, t)).sum();
        assert!(ch9_sum > 0.0, "keep-shanten mask should have some tiles set");
    }

    // -- Discard tests (ch 11-22) --

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
        // Player 0: ch_base=11, presence=ch11, tedashi=ch12
        assert_eq!(get(&enc, 11, 5), 1.0);
        assert_eq!(get(&enc, 12, 5), 1.0);
        // Player 1: ch_base=14, presence=ch14, tedashi=ch15
        assert_eq!(get(&enc, 14, 10), 1.0);
        assert_eq!(get(&enc, 15, 10), 0.0); // tsumogiri
    }

    #[test]
    fn discard_temporal_decay() {
        let mut enc = ObservationEncoder::new();
        let mut discards = empty_discards();
        discards[0].discards.push(DiscardEntry { tile: 0, is_tedashi: false, turn: 0 });
        discards[0].discards.push(DiscardEntry { tile: 1, is_tedashi: false, turn: 5 });
        enc.encode_discards(&discards);
        // temporal ch = 11 + 2 = 13
        assert!((get(&enc, 13, 1) - 1.0).abs() < 1e-6);
        let expected = (-1.0f32).exp();
        assert!((get(&enc, 13, 0) - expected).abs() < 1e-6);
    }

    // -- Meld tests (ch 23-34) --

    fn empty_melds() -> [Vec<MeldInfo>; NUM_PLAYERS] {
        [vec![], vec![], vec![], vec![]]
    }

    #[test]
    fn meld_chi() {
        let mut enc = ObservationEncoder::new();
        let mut melds = empty_melds();
        melds[0].push(MeldInfo {
            tiles: vec![0, 1, 2],
            meld_type: MeldType::Chi,
        });
        enc.encode_melds(&melds);
        // Player 0 chi = ch 23
        assert_eq!(get(&enc, 23, 0), 1.0);
        assert_eq!(get(&enc, 23, 1), 1.0);
        assert_eq!(get(&enc, 23, 2), 1.0);
        assert_eq!(get(&enc, 24, 0), 0.0); // pon channel empty
    }

    #[test]
    fn meld_pon() {
        let mut enc = ObservationEncoder::new();
        let mut melds = empty_melds();
        melds[2].push(MeldInfo {
            tiles: vec![27, 27, 27],
            meld_type: MeldType::Pon,
        });
        enc.encode_melds(&melds);
        // Player 2: ch_base = 23 + 3*2 = 29, pon = ch 30
        assert_eq!(get(&enc, 30, 27), 1.0);
    }

    #[test]
    fn meld_kan() {
        let mut enc = ObservationEncoder::new();
        let mut melds = empty_melds();
        melds[1].push(MeldInfo {
            tiles: vec![31, 31, 31, 31],
            meld_type: MeldType::Kan,
        });
        enc.encode_melds(&melds);
        // Player 1: ch_base = 23 + 3*1 = 26, kan = ch 28
        assert_eq!(get(&enc, 28, 31), 1.0);
    }

    // -- Dora tests (ch 35-39) --

    #[test]
    fn dora_indicator_thermometer_single() {
        let mut enc = ObservationEncoder::new();
        let dora = DoraInfo {
            indicators: vec![0], // one indicator on 1m
            aka_flags: [false; 3],
        };
        enc.encode_dora(&dora);
        assert_eq!(get(&enc, 35, 0), 1.0); // >= 1
        assert_eq!(get(&enc, 36, 0), 0.0); // >= 2 not set
    }

    #[test]
    fn dora_indicator_thermometer_multiple_same() {
        let mut enc = ObservationEncoder::new();
        let dora = DoraInfo {
            indicators: vec![5, 5, 5], // three indicators on 6m
            aka_flags: [false; 3],
        };
        enc.encode_dora(&dora);
        assert_eq!(get(&enc, 35, 5), 1.0); // >= 1
        assert_eq!(get(&enc, 36, 5), 1.0); // >= 2
        assert_eq!(get(&enc, 37, 5), 1.0); // >= 3
        assert_eq!(get(&enc, 38, 5), 0.0); // >= 4 not set
    }

    #[test]
    fn dora_indicator_thermometer_different() {
        let mut enc = ObservationEncoder::new();
        let dora = DoraInfo {
            indicators: vec![0, 10], // 1m and 2p indicators
            aka_flags: [false; 3],
        };
        enc.encode_dora(&dora);
        assert_eq!(get(&enc, 35, 0), 1.0);
        assert_eq!(get(&enc, 35, 10), 1.0);
        assert_eq!(get(&enc, 36, 0), 0.0);
        assert_eq!(get(&enc, 36, 10), 0.0);
    }

    // -- Aka tests (ch 40-42) --

    #[test]
    fn aka_dora_plane_fill() {
        let mut enc = ObservationEncoder::new();
        let dora = DoraInfo {
            indicators: vec![],
            aka_flags: [true, false, true],
        };
        enc.encode_aka(&dora);
        // Ch 40 (5m): entire channel filled
        assert_eq!(get(&enc, 40, 0), 1.0);
        assert_eq!(get(&enc, 40, 33), 1.0);
        // Ch 41 (5p): not set
        assert_eq!(get(&enc, 41, 0), 0.0);
        // Ch 42 (5s): entire channel filled
        assert_eq!(get(&enc, 42, 0), 1.0);
    }

    // -- Metadata tests (ch 43-61) --

    fn test_metadata() -> GameMetadata {
        GameMetadata {
            riichi: [true, false, false, false],
            scores: [25000, 25000, 25000, 25000],
            shanten: 1,
            kyoku_index: 0,
            honba: 2,
            kyotaku: 1,
        }
    }

    #[test]
    fn metadata_riichi_and_scores() {
        let mut enc = ObservationEncoder::new();
        enc.encode_metadata(&test_metadata());
        // Self riichi = ch 43 filled
        assert_eq!(get(&enc, 43, 0), 1.0);
        // Opponent 1 riichi = ch 44 -- NOT set
        assert_eq!(get(&enc, 44, 0), 0.0);
        // Score ch 47 = 25000/100000 = 0.25
        assert!((get(&enc, 47, 0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn metadata_score_gaps() {
        let mut enc = ObservationEncoder::new();
        let mut meta = test_metadata();
        meta.scores = [30000, 25000, 20000, 25000];
        enc.encode_metadata(&meta);
        // gap[0] = (30000-30000)/30000 = 0.0
        assert!((get(&enc, 51, 0) - 0.0).abs() < 1e-6);
        // gap[1] = (30000-25000)/30000 = 0.1667
        assert!((get(&enc, 52, 0) - 5000.0 / 30000.0).abs() < 1e-4);
        // gap[2] = (30000-20000)/30000 = 0.3333
        assert!((get(&enc, 53, 0) - 10000.0 / 30000.0).abs() < 1e-4);
    }

    #[test]
    fn metadata_shanten_one_hot() {
        // shanten = 0 (tenpai) -> ch 55
        let mut enc = ObservationEncoder::new();
        let mut meta = test_metadata();
        meta.shanten = 0;
        enc.encode_metadata(&meta);
        assert_eq!(get(&enc, 55, 0), 1.0);
        assert_eq!(get(&enc, 56, 0), 0.0);

        // shanten = 2 -> ch 57
        enc.clear();
        meta.shanten = 2;
        enc.encode_metadata(&meta);
        assert_eq!(get(&enc, 57, 0), 1.0);
        assert_eq!(get(&enc, 55, 0), 0.0);

        // shanten = 5 (clamped to 3+) -> ch 58
        enc.clear();
        meta.shanten = 5;
        enc.encode_metadata(&meta);
        assert_eq!(get(&enc, 58, 0), 1.0);
    }

    #[test]
    fn metadata_round_honba_kyotaku() {
        let mut enc = ObservationEncoder::new();
        let mut meta = test_metadata();
        meta.kyoku_index = 4; // South 1
        meta.honba = 3;
        meta.kyotaku = 2;
        enc.encode_metadata(&meta);
        // Ch 59: 4/8 = 0.5
        assert!((get(&enc, 59, 0) - 0.5).abs() < 1e-6);
        // Ch 60: 3/10 = 0.3
        assert!((get(&enc, 60, 0) - 0.3).abs() < 1e-6);
        // Ch 61: 2/10 = 0.2
        assert!((get(&enc, 61, 0) - 0.2).abs() < 1e-6);
    }

    // -- Safety tests (ch 62-84, unchanged) --

    #[test]
    fn safety_genbutsu_channels() {
        let mut enc = ObservationEncoder::new();
        let mut si = SafetyInfo::new();
        si.genbutsu_all[0][5] = true;
        si.genbutsu_tedashi[1][10] = true;
        si.genbutsu_riichi_era[2][20] = true;
        enc.encode_safety(&si);
        assert_eq!(get(&enc, 62, 5), 1.0);
        assert_eq!(get(&enc, 66, 10), 1.0);
        assert_eq!(get(&enc, 70, 20), 1.0);
    }

    #[test]
    fn safety_suji_channel() {
        let mut enc = ObservationEncoder::new();
        let mut si = SafetyInfo::new();
        si.suji[0][0] = 1.0;
        enc.encode_safety(&si);
        assert_eq!(get(&enc, 71, 0), 1.0);
    }

    #[test]
    fn safety_kabe_and_one_chance() {
        let mut enc = ObservationEncoder::new();
        let mut si = SafetyInfo::new();
        si.kabe[15] = true;
        si.one_chance[20] = true;
        enc.encode_safety(&si);
        assert_eq!(get(&enc, 80, 15), 1.0);
        assert_eq!(get(&enc, 81, 20), 1.0);
    }

    // -- Full encode test --

    #[test]
    fn full_encode_returns_correct_size() {
        let mut enc = ObservationEncoder::new();
        let hand = [0u8; NUM_TILES];
        let open_meld = [0u8; NUM_TILES];
        let discards = empty_discards();
        let melds = empty_melds();
        let dora = DoraInfo { indicators: vec![], aka_flags: [false; 3] };
        let meta = test_metadata();
        let safety = SafetyInfo::new();
        let obs = enc.encode(
            &hand, None, &open_meld, &discards, &melds, &dora, &meta, &safety,
        );
        assert_eq!(obs.len(), OBS_SIZE);
    }

    #[test]
    fn full_encode_clears_between_calls() {
        let mut enc = ObservationEncoder::new();
        let mut hand = [0u8; NUM_TILES];
        hand[0] = 3;
        let open_meld = [0u8; NUM_TILES];
        let discards = empty_discards();
        let melds = empty_melds();
        let dora = DoraInfo { indicators: vec![], aka_flags: [false; 3] };
        let meta = test_metadata();
        let safety = SafetyInfo::new();
        enc.encode(&hand, None, &open_meld, &discards, &melds, &dora, &meta, &safety);
        assert_eq!(get(&enc, 2, 0), 1.0);

        let empty_hand = [0u8; NUM_TILES];
        enc.encode(&empty_hand, None, &open_meld, &discards, &melds, &dora, &meta, &safety);
        assert_eq!(get(&enc, 2, 0), 0.0);
    }
}