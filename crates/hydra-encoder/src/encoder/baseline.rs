use hydra_belief_search::shanten_batch::{self, BatchShantenResult};
use hydra_safety::{self, SafetyInfo};

use super::ObservationEncoder;
use super::layout::*;
use super::types::*;

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
pub(super) fn for_each_set_bit(mut bits: u64, mut f: impl FnMut(usize)) {
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
    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    pub(super) fn encode_baseline_prefix(
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
    pub(super) fn encode_baseline_prefix_from_batch(
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
