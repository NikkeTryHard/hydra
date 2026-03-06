//! Safety tile calculations: genbutsu, suji, kabe, one-chance.
//!
//! These feed into the 23 safety channels (62-84) of the fixed-superset observation tensor.
//! Updated incrementally on each discard/call/kan event.

/// Number of opponents to track safety against.
pub const NUM_OPPONENTS: usize = 3;

/// Inference-time threshold for activating tenpai hint channels.
pub const TENPAI_HINT_THRESHOLD: f32 = 0.5;

/// Number of tile types.
const NUM_TILES: usize = 34;

/// Set bit `idx` (0-33) in a u64 bitfield.
#[inline]
pub fn bit_set(field: &mut u64, idx: usize) {
    *field |= 1u64 << idx;
}

/// Test bit `idx` (0-33) in a u64 bitfield.
#[inline]
pub fn bit_test(field: u64, idx: usize) -> bool {
    (field >> idx) & 1 != 0
}

/// Safety information for the current player against all opponents.
/// Updated incrementally as the game progresses.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct SafetyInfo {
    /// Genbutsu (safe tiles): 3 sub-channels per opponent.
    /// `bit_test(genbutsu_all[opp], tile)` = true if tile is 100% safe against opponent.
    pub genbutsu_all: [u64; NUM_OPPONENTS],
    /// `bit_test(genbutsu_tedashi[opp], tile)` = true if opponent discarded this tile from hand (not tsumogiri).
    pub genbutsu_tedashi: [u64; NUM_OPPONENTS],
    /// `bit_test(genbutsu_riichi_era[opp], tile)` = true if tile was discarded after opponent's riichi.
    pub genbutsu_riichi_era: [u64; NUM_OPPONENTS],

    /// Suji (2-away inference): float 0.0-1.0 per tile.
    ///
    /// e.g., if 4m is genbutsu, then 1m and 7m get suji safety.
    pub suji: [[f32; NUM_TILES]; NUM_OPPONENTS],

    /// Half-suji: tile has exactly 1 of 2 suji partners genbutsu (center tiles only).
    /// Only meaningful for center tiles (4,5,6 of each suit).
    pub half_suji: [u64; NUM_OPPONENTS],

    /// Matagi-suji: tedashi straddle signal.
    /// When opponent discards tile T from hand, tiles T-1 and T+1 get matagi danger.
    pub matagi: [[f32; NUM_TILES]; NUM_OPPONENTS],

    /// Kabe (wall block): all 4 copies visible.
    pub kabe: u64,

    /// One-chance: 3 copies visible, 1 remaining.
    pub one_chance: u64,

    /// Visible tile counts (for kabe/one-chance calculation).
    pub visible_counts: [u8; NUM_TILES],

    /// Opponent riichi status (for genbutsu_riichi_era tracking).
    pub opponent_riichi: [bool; NUM_OPPONENTS],

    /// Cached tenpai head probabilities from the previous decision step.
    ///
    /// These feed channels 82-84 at inference time via
    /// `max(riichi_status, cached_tenpai_prob > 0.5)`.
    pub cached_tenpai_prob: [f32; NUM_OPPONENTS],
}

impl SafetyInfo {
    /// Create a new `SafetyInfo` with all fields zeroed/false.
    #[inline]
    pub fn new() -> Self {
        Self {
            genbutsu_all: [0; NUM_OPPONENTS],
            genbutsu_tedashi: [0; NUM_OPPONENTS],
            genbutsu_riichi_era: [0; NUM_OPPONENTS],
            suji: [[0.0; NUM_TILES]; NUM_OPPONENTS],
            half_suji: [0; NUM_OPPONENTS],
            matagi: [[0.0; NUM_TILES]; NUM_OPPONENTS],
            kabe: 0,
            one_chance: 0,
            visible_counts: [0; NUM_TILES],
            opponent_riichi: [false; NUM_OPPONENTS],
            cached_tenpai_prob: [0.0; NUM_OPPONENTS],
        }
    }

    /// Reset all safety data to initial state.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for SafetyInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetyInfo {
    /// Returns whether the tenpai hint channel should be active for `opponent_idx`.
    #[inline]
    pub fn tenpai_hint_active(&self, opponent_idx: usize) -> bool {
        opponent_idx < NUM_OPPONENTS
            && (self.opponent_riichi[opponent_idx]
                || self.cached_tenpai_prob[opponent_idx] > TENPAI_HINT_THRESHOLD)
    }

    /// Caches a tenpai head probability for use on the next decision step.
    #[inline]
    pub fn set_tenpai_prediction(&mut self, opponent_idx: usize, probability: f32) {
        if opponent_idx < NUM_OPPONENTS {
            self.cached_tenpai_prob[opponent_idx] = probability.clamp(0.0, 1.0);
        }
    }

    /// Replaces all cached tenpai probabilities at once.
    #[inline]
    pub fn set_tenpai_predictions(&mut self, probabilities: [f32; NUM_OPPONENTS]) {
        for (idx, probability) in probabilities.into_iter().enumerate() {
            self.set_tenpai_prediction(idx, probability);
        }
    }

    /// Update safety info when a tile is discarded.
    ///
    /// `tile_type`: 0-33 tile type index
    /// `opponent_idx`: which opponent (0-2) discarded it, relative to the observing player
    /// `is_tedashi`: true if discarded from hand (not tsumogiri)
    #[inline]
    pub fn on_discard(&mut self, tile_type: u8, opponent_idx: usize, is_tedashi: bool) {
        let t = tile_type as usize;
        if t >= NUM_TILES || opponent_idx >= NUM_OPPONENTS {
            return;
        }

        // Update genbutsu
        bit_set(&mut self.genbutsu_all[opponent_idx], t);
        if is_tedashi {
            bit_set(&mut self.genbutsu_tedashi[opponent_idx], t);
        }
        if self.opponent_riichi[opponent_idx] {
            bit_set(&mut self.genbutsu_riichi_era[opponent_idx], t);
        }

        // Update visible counts and kabe/one-chance
        self.visible_counts[t] = self.visible_counts[t].saturating_add(1);
        self.update_kabe_one_chance(t);

        // Update suji for this opponent
        self.update_suji(opponent_idx, t);

        // Update matagi-suji: tedashi straddle signal
        // When opponent discards tile T from hand, tiles T-1 and T+1 get matagi danger.
        // Only applies to suited tiles (0-26).
        if is_tedashi && t < 27 {
            let suit_pos = t % 9;
            if suit_pos > 0 {
                self.matagi[opponent_idx][t - 1] = 1.0;
            }
            if suit_pos < 8 {
                self.matagi[opponent_idx][t + 1] = 1.0;
            }
        }
    }
}

impl SafetyInfo {
    /// Update suji safety for an opponent when a tile becomes genbutsu.
    ///
    /// Suji pattern: if tile N is safe, tiles N-3 and N+3 get suji inference.
    /// Only applies to suited tiles (indices 0-26). Honors have no suji.
    #[inline]
    fn update_suji(&mut self, opponent_idx: usize, tile: usize) {
        // Suji only applies to suited tiles (0-26)
        if tile >= 27 {
            return;
        }

        let suit_offset = (tile / 9) * 9;
        let number = tile - suit_offset; // 0-indexed: 0=1m, 1=2m, ..., 8=9m

        // If tile N is safe, then N-3 and N+3 get suji inference
        // 1-4-7, 2-5-8, 3-6-9 pattern
        if number >= 3 {
            let suji_tile = suit_offset + number - 3;
            self.suji[opponent_idx][suji_tile] = self.suji[opponent_idx][suji_tile].max(1.0);
        }
        if number + 3 < 9 {
            let suji_tile = suit_offset + number + 3;
            self.suji[opponent_idx][suji_tile] = self.suji[opponent_idx][suji_tile].max(1.0);
        }

        // Now fix center tiles (4,5,6 of each suit) for half-suji.
        // Center tiles have TWO suji partners. If only one partner is genbutsu,
        // suji should be 0.5 (half-suji), not 1.0.
        // Also update the half_suji bitfield.
        self.recompute_center_suji(opponent_idx);
    }

    /// Recompute suji values for center tiles (4,5,6 of each suit) for one opponent.
    ///
    /// Center tiles have two suji partners. Edge tiles (1-3, 7-9) have only one.
    /// For center tiles: 1 of 2 partners safe -> 0.5 (half-suji), 2 of 2 -> 1.0.
    #[inline]
    fn recompute_center_suji(&mut self, opp: usize) {
        // Center tile indices in 34-format: 3,4,5 (man), 12,13,14 (pin), 21,22,23 (sou)
        const CENTER_TILES: [usize; 9] = [3, 4, 5, 12, 13, 14, 21, 22, 23];
        for &tile in &CENTER_TILES {
            let partner_low = tile - 3; // e.g. 4m(3) -> 1m(0)
            let partner_high = tile + 3; // e.g. 4m(3) -> 7m(6)
            let p_low = bit_test(self.genbutsu_all[opp], partner_low);
            let p_high = bit_test(self.genbutsu_all[opp], partner_high);
            match (p_low, p_high) {
                (true, true) => {
                    self.suji[opp][tile] = 1.0;
                    // Clear half_suji bit
                    self.half_suji[opp] &= !(1u64 << tile);
                }
                (true, false) | (false, true) => {
                    self.suji[opp][tile] = 0.5;
                    bit_set(&mut self.half_suji[opp], tile);
                }
                (false, false) => {
                    self.suji[opp][tile] = 0.0;
                    self.half_suji[opp] &= !(1u64 << tile);
                }
            }
        }
    }

    /// Update kabe and one-chance flags based on visible tile counts.
    #[inline]
    fn update_kabe_one_chance(&mut self, tile: usize) {
        if self.visible_counts[tile] >= 4 {
            bit_set(&mut self.kabe, tile);
        }
        if self.visible_counts[tile] == 3 {
            bit_set(&mut self.one_chance, tile);
        } else {
            self.one_chance &= !(1u64 << tile);
        }
    }
}

impl SafetyInfo {
    /// Update when an opponent declares riichi.
    #[inline]
    pub fn on_riichi(&mut self, opponent_idx: usize) {
        if opponent_idx < NUM_OPPONENTS {
            self.opponent_riichi[opponent_idx] = true;
        }
    }

    /// Update visible counts when tiles are revealed via call (chi/pon/kan).
    #[inline]
    pub fn on_call(&mut self, tiles: &[u8]) {
        for &t in tiles {
            let idx = t as usize;
            if idx < NUM_TILES {
                self.visible_counts[idx] = self.visible_counts[idx].saturating_add(1);
                self.update_kabe_one_chance(idx);
            }
        }
    }

    /// Update visible counts when a dora indicator is revealed.
    #[inline]
    pub fn on_dora_revealed(&mut self, tile_type: u8) {
        let idx = tile_type as usize;
        if idx < NUM_TILES {
            self.visible_counts[idx] = self.visible_counts[idx].saturating_add(1);
            self.update_kabe_one_chance(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_safety_info_is_zeroed() {
        let si = SafetyInfo::new();
        for opp in 0..NUM_OPPONENTS {
            assert_eq!(si.genbutsu_all[opp], 0);
            assert_eq!(si.genbutsu_tedashi[opp], 0);
            assert_eq!(si.genbutsu_riichi_era[opp], 0);
            assert!(si.suji[opp].iter().all(|&v| v == 0.0));
            assert!(!si.opponent_riichi[opp]);
            assert_eq!(si.cached_tenpai_prob[opp], 0.0);
            assert_eq!(si.half_suji[opp], 0);
            assert!(si.matagi[opp].iter().all(|&v| v == 0.0));
        }
        assert_eq!(si.kabe, 0);
        assert_eq!(si.one_chance, 0);
        assert!(si.visible_counts.iter().all(|&v| v == 0));
    }

    #[test]
    fn on_discard_sets_genbutsu_all() {
        let mut si = SafetyInfo::new();
        si.on_discard(5, 0, false); // 6m tsumogiri by opponent 0
        assert!(bit_test(si.genbutsu_all[0], 5));
        assert!(!bit_test(si.genbutsu_tedashi[0], 5));
        assert!(!bit_test(si.genbutsu_all[1], 5)); // other opponents unaffected
    }

    #[test]
    fn on_discard_tedashi_sets_both_flags() {
        let mut si = SafetyInfo::new();
        si.on_discard(10, 1, true); // 2p tedashi by opponent 1
        assert!(bit_test(si.genbutsu_all[1], 10));
        assert!(bit_test(si.genbutsu_tedashi[1], 10));
    }

    #[test]
    fn riichi_then_discard_sets_riichi_era() {
        let mut si = SafetyInfo::new();
        si.on_riichi(2);
        si.on_discard(0, 2, false); // 1m after opponent 2's riichi
        assert!(bit_test(si.genbutsu_riichi_era[2], 0));
        // Before riichi, should not be set
        assert!(!bit_test(si.genbutsu_riichi_era[0], 0));
    }

    #[test]
    fn suji_from_4m_discard() {
        // 4m = index 3. Suji targets: 1m (index 0) and 7m (index 6)
        let mut si = SafetyInfo::new();
        si.on_discard(3, 0, false);
        assert_eq!(si.suji[0][0], 1.0); // 1m gets suji
        assert_eq!(si.suji[0][6], 1.0); // 7m gets suji
        assert_eq!(si.suji[0][3], 0.0); // 4m itself has no suji from this
    }

    #[test]
    fn suji_honors_produce_none() {
        let mut si = SafetyInfo::new();
        si.on_discard(27, 0, false); // East wind (first honor)
        for i in 0..NUM_TILES {
            assert_eq!(si.suji[0][i], 0.0);
        }
    }

    #[test]
    fn kabe_at_four_visible() {
        let mut si = SafetyInfo::new();
        for _ in 0..3 {
            si.on_discard(15, 0, false); // discard 7p three times
        }
        assert!(!bit_test(si.kabe, 15));
        assert!(bit_test(si.one_chance, 15));
        si.on_discard(15, 1, false); // 4th copy
        assert!(bit_test(si.kabe, 15));
        assert!(!bit_test(si.one_chance, 15)); // no longer one-chance at 4
    }

    #[test]
    fn on_call_updates_visible_counts() {
        let mut si = SafetyInfo::new();
        si.on_call(&[0, 1, 2]); // chi 1m-2m-3m
        assert_eq!(si.visible_counts[0], 1);
        assert_eq!(si.visible_counts[1], 1);
        assert_eq!(si.visible_counts[2], 1);
    }

    #[test]
    fn on_dora_revealed_updates_visible() {
        let mut si = SafetyInfo::new();
        si.on_dora_revealed(33); // last honor tile
        assert_eq!(si.visible_counts[33], 1);
    }

    #[test]
    fn reset_clears_everything() {
        let mut si = SafetyInfo::new();
        si.on_discard(5, 0, true);
        si.on_riichi(1);
        si.set_tenpai_prediction(2, 0.9);
        si.on_dora_revealed(20);
        si.reset();
        assert!(!bit_test(si.genbutsu_all[0], 5));
        assert!(!bit_test(si.genbutsu_tedashi[0], 5));
        assert!(!si.opponent_riichi[1]);
        assert_eq!(si.cached_tenpai_prob[2], 0.0);
        assert_eq!(si.visible_counts[20], 0);
    }

    #[test]
    fn tenpai_hint_activates_from_cached_prediction() {
        let mut si = SafetyInfo::new();
        assert!(!si.tenpai_hint_active(0));
        si.set_tenpai_prediction(0, 0.6);
        assert!(si.tenpai_hint_active(0));
    }

    #[test]
    fn tenpai_hint_clamps_predictions() {
        let mut si = SafetyInfo::new();
        si.set_tenpai_prediction(1, 10.0);
        si.set_tenpai_prediction(2, -5.0);
        assert_eq!(si.cached_tenpai_prob[1], 1.0);
        assert_eq!(si.cached_tenpai_prob[2], 0.0);
    }

    #[test]
    fn out_of_bounds_ignored() {
        let mut si = SafetyInfo::new();
        // Should not panic
        si.on_discard(34, 0, false); // tile out of bounds
        si.on_discard(0, 3, false); // opponent out of bounds
        si.on_dora_revealed(255); // way out of bounds
        si.on_call(&[35, 100]); // tiles out of bounds
    }

    #[test]
    fn half_suji_center_tile_one_partner() {
        // Discard 1m (index 0) -> 4m (index 3) gets suji.
        // 4m is center tile with partners 1m(0) and 7m(6).
        // Only 1m is genbutsu -> half suji (0.5).
        let mut si = SafetyInfo::new();
        si.on_discard(0, 0, false); // 1m genbutsu
        assert_eq!(si.suji[0][3], 0.5); // 4m half suji
        assert!(bit_test(si.half_suji[0], 3));
    }

    #[test]
    fn half_suji_center_tile_both_partners() {
        // Discard both 1m and 7m -> 4m gets full suji.
        let mut si = SafetyInfo::new();
        si.on_discard(0, 0, false); // 1m genbutsu
        si.on_discard(6, 0, false); // 7m genbutsu
        assert_eq!(si.suji[0][3], 1.0); // 4m full suji
        assert!(!bit_test(si.half_suji[0], 3)); // not half
    }

    #[test]
    fn half_suji_edge_tile_unaffected() {
        // 1m (index 0) is edge tile, only partner is 4m.
        // Discarding 4m -> 1m gets full 1.0, not half.
        let mut si = SafetyInfo::new();
        si.on_discard(3, 0, false); // 4m genbutsu
        assert_eq!(si.suji[0][0], 1.0); // 1m full suji
        assert!(!bit_test(si.half_suji[0], 0)); // edge tile never half
    }

    #[test]
    fn matagi_from_tedashi() {
        // Tedashi discard of 5m (index 4) marks 4m and 6m as matagi.
        let mut si = SafetyInfo::new();
        si.on_discard(4, 1, true); // 5m tedashi by opp1
        assert_eq!(si.matagi[1][3], 1.0); // 4m matagi
        assert_eq!(si.matagi[1][5], 1.0); // 6m matagi
        assert_eq!(si.matagi[1][4], 0.0); // 5m itself no matagi
    }

    #[test]
    fn matagi_not_from_tsumogiri() {
        // Tsumogiri should NOT set matagi.
        let mut si = SafetyInfo::new();
        si.on_discard(4, 0, false); // 5m tsumogiri
        assert_eq!(si.matagi[0][3], 0.0);
        assert_eq!(si.matagi[0][5], 0.0);
    }

    #[test]
    fn matagi_edge_tiles() {
        // 1m (index 0) tedashi -> only 2m (index 1) gets matagi (no -1).
        let mut si = SafetyInfo::new();
        si.on_discard(0, 0, true); // 1m tedashi
        assert_eq!(si.matagi[0][1], 1.0); // 2m matagi
        // 9m (index 8) tedashi -> only 8m (index 7) gets matagi (no +1).
        si.on_discard(8, 0, true); // 9m tedashi
        assert_eq!(si.matagi[0][7], 1.0); // 8m matagi
    }

    #[test]
    fn matagi_honors_ignored() {
        // Honor tile tedashi should NOT produce matagi.
        let mut si = SafetyInfo::new();
        si.on_discard(27, 0, true); // East wind tedashi
        for i in 0..NUM_TILES {
            assert_eq!(si.matagi[0][i], 0.0);
        }
    }
}
