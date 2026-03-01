//! Safety tile calculations: genbutsu, suji, kabe, one-chance.
//!
//! These feed into the 23 safety channels (62-84) of the 85x34 observation tensor.
//! Updated incrementally on each discard/call/kan event.

/// Number of opponents to track safety against.
pub const NUM_OPPONENTS: usize = 3;

/// Number of tile types.
const NUM_TILES: usize = 34;

/// Safety information for the current player against all opponents.
/// Updated incrementally as the game progresses.
#[derive(Debug, Clone)]
pub struct SafetyInfo {
    // -- Genbutsu (safe tiles): 3 sub-channels per opponent --
    // genbutsu_all[opp][tile] = true if tile is 100% safe against opponent
    pub genbutsu_all: [[bool; NUM_TILES]; NUM_OPPONENTS],
    // genbutsu_tedashi[opp][tile] = true if opponent discarded this tile from hand (not tsumogiri)
    pub genbutsu_tedashi: [[bool; NUM_TILES]; NUM_OPPONENTS],
    // genbutsu_riichi_era[opp][tile] = true if tile was discarded after opponent's riichi
    pub genbutsu_riichi_era: [[bool; NUM_TILES]; NUM_OPPONENTS],

    // -- Suji (2-away inference): float 0.0-1.0 per tile --
    // e.g., if 4m is genbutsu, then 1m and 7m get suji safety
    pub suji: [[f32; NUM_TILES]; NUM_OPPONENTS],

    // -- Kabe (wall block): all 4 copies visible --
    pub kabe: [bool; NUM_TILES],

    // -- One-chance: 3 copies visible, 1 remaining --
    pub one_chance: [bool; NUM_TILES],

    // -- Visible tile counts (for kabe/one-chance calculation) --
    pub visible_counts: [u8; NUM_TILES],

    // -- Opponent riichi status (for genbutsu_riichi_era tracking) --
    pub opponent_riichi: [bool; NUM_OPPONENTS],
}

impl SafetyInfo {
    /// Create a new `SafetyInfo` with all fields zeroed/false.
    pub fn new() -> Self {
        Self {
            genbutsu_all: [[false; NUM_TILES]; NUM_OPPONENTS],
            genbutsu_tedashi: [[false; NUM_TILES]; NUM_OPPONENTS],
            genbutsu_riichi_era: [[false; NUM_TILES]; NUM_OPPONENTS],
            suji: [[0.0; NUM_TILES]; NUM_OPPONENTS],
            kabe: [false; NUM_TILES],
            one_chance: [false; NUM_TILES],
            visible_counts: [0; NUM_TILES],
            opponent_riichi: [false; NUM_OPPONENTS],
        }
    }

    /// Reset all safety data to initial state.
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
    /// Update safety info when a tile is discarded.
    ///
    /// `tile_type`: 0-33 tile type index
    /// `opponent_idx`: which opponent (0-2) discarded it, relative to the observing player
    /// `is_tedashi`: true if discarded from hand (not tsumogiri)
    pub fn on_discard(&mut self, tile_type: u8, opponent_idx: usize, is_tedashi: bool) {
        let t = tile_type as usize;
        if t >= NUM_TILES || opponent_idx >= NUM_OPPONENTS {
            return;
        }

        // Update genbutsu
        self.genbutsu_all[opponent_idx][t] = true;
        if is_tedashi {
            self.genbutsu_tedashi[opponent_idx][t] = true;
        }
        if self.opponent_riichi[opponent_idx] {
            self.genbutsu_riichi_era[opponent_idx][t] = true;
        }

        // Update visible counts and kabe/one-chance
        self.visible_counts[t] = self.visible_counts[t].saturating_add(1);
        self.update_kabe_one_chance(t);

        // Update suji for this opponent
        self.update_suji(opponent_idx, t);
    }
}

impl SafetyInfo {
    /// Update suji safety for an opponent when a tile becomes genbutsu.
    ///
    /// Suji pattern: if tile N is safe, tiles N-3 and N+3 get suji inference.
    /// Only applies to suited tiles (indices 0-26). Honors have no suji.
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
            self.suji[opponent_idx][suji_tile] =
                self.suji[opponent_idx][suji_tile].max(1.0);
        }
        if number + 3 < 9 {
            let suji_tile = suit_offset + number + 3;
            self.suji[opponent_idx][suji_tile] =
                self.suji[opponent_idx][suji_tile].max(1.0);
        }
    }

    /// Update kabe and one-chance flags based on visible tile counts.
    fn update_kabe_one_chance(&mut self, tile: usize) {
        self.kabe[tile] = self.visible_counts[tile] >= 4;
        self.one_chance[tile] = self.visible_counts[tile] == 3;
    }
}

impl SafetyInfo {
    /// Update when an opponent declares riichi.
    pub fn on_riichi(&mut self, opponent_idx: usize) {
        if opponent_idx < NUM_OPPONENTS {
            self.opponent_riichi[opponent_idx] = true;
        }
    }

    /// Update visible counts when tiles are revealed via call (chi/pon/kan).
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
            assert!(si.genbutsu_all[opp].iter().all(|&v| !v));
            assert!(si.genbutsu_tedashi[opp].iter().all(|&v| !v));
            assert!(si.genbutsu_riichi_era[opp].iter().all(|&v| !v));
            assert!(si.suji[opp].iter().all(|&v| v == 0.0));
            assert!(!si.opponent_riichi[opp]);
        }
        assert!(si.kabe.iter().all(|&v| !v));
        assert!(si.one_chance.iter().all(|&v| !v));
        assert!(si.visible_counts.iter().all(|&v| v == 0));
    }

    #[test]
    fn on_discard_sets_genbutsu_all() {
        let mut si = SafetyInfo::new();
        si.on_discard(5, 0, false); // 6m tsumogiri by opponent 0
        assert!(si.genbutsu_all[0][5]);
        assert!(!si.genbutsu_tedashi[0][5]);
        assert!(!si.genbutsu_all[1][5]); // other opponents unaffected
    }

    #[test]
    fn on_discard_tedashi_sets_both_flags() {
        let mut si = SafetyInfo::new();
        si.on_discard(10, 1, true); // 2p tedashi by opponent 1
        assert!(si.genbutsu_all[1][10]);
        assert!(si.genbutsu_tedashi[1][10]);
    }

    #[test]
    fn riichi_then_discard_sets_riichi_era() {
        let mut si = SafetyInfo::new();
        si.on_riichi(2);
        si.on_discard(0, 2, false); // 1m after opponent 2's riichi
        assert!(si.genbutsu_riichi_era[2][0]);
        // Before riichi, should not be set
        assert!(!si.genbutsu_riichi_era[0][0]);
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
        assert!(!si.kabe[15]);
        assert!(si.one_chance[15]);
        si.on_discard(15, 1, false); // 4th copy
        assert!(si.kabe[15]);
        assert!(!si.one_chance[15]); // no longer one-chance at 4
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
        si.on_dora_revealed(20);
        si.reset();
        assert!(!si.genbutsu_all[0][5]);
        assert!(!si.genbutsu_tedashi[0][5]);
        assert!(!si.opponent_riichi[1]);
        assert_eq!(si.visible_counts[20], 0);
    }

    #[test]
    fn out_of_bounds_ignored() {
        let mut si = SafetyInfo::new();
        // Should not panic
        si.on_discard(34, 0, false); // tile out of bounds
        si.on_discard(0, 3, false);  // opponent out of bounds
        si.on_dora_revealed(255);     // way out of bounds
        si.on_call(&[35, 100]);       // tiles out of bounds
    }
}
