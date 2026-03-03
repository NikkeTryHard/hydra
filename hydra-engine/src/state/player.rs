use crate::types::Meld;
/// Per-player state for a 4-player mahjong game.
#[derive(Debug, Clone)]
pub struct PlayerState {
    /// Sorted hand tiles (136-format), valid up to `hand_len`.
    pub hand: [u8; 14],
    /// Number of tiles currently in hand.
    pub hand_len: u8,
    /// Called melds (pon, chi, kan), valid up to `meld_count`.
    pub melds: [Meld; 4],
    /// Number of active melds.
    pub meld_count: u8,
    /// Discarded tiles in order, valid up to `discard_len`.
    pub discards: [u8; 30],
    /// Whether each discard came from the hand (true) or was tsumogiri (false).
    pub discard_from_hand: [bool; 30],
    /// Whether each discard was a riichi declaration discard.
    pub discard_is_riichi: [bool; 30],
    /// Number of discards made this round.
    pub discard_len: u8,
    /// Index into the discards array where riichi was declared, if any.
    pub riichi_declaration_index: Option<usize>,
    /// Current score in points.
    pub score: i32,
    /// Score change from the most recent scoring event.
    pub score_delta: i32,
    /// Whether riichi has been declared and accepted.
    pub riichi_declared: bool,
    /// Whether a riichi declaration is pending acceptance.
    pub riichi_stage: bool,
    /// Whether double riichi was declared on the first turn.
    pub double_riichi_declared: bool,
    /// Whether a ron opportunity was missed while in riichi (permanent furiten).
    pub missed_agari_riichi: bool,
    /// Whether a ron opportunity was missed this turn (temporary furiten).
    pub missed_agari_doujun: bool,
    /// Whether this player is still eligible for nagashi mangan.
    pub nagashi_eligible: bool,
    /// Whether the ippatsu window is active after riichi.
    pub ippatsu_cycle: bool,
    /// Pao (liability) entries mapping yaku id to liable player.
    pub pao: [(u8, u8); 3],
    /// Number of active pao entries.
    pub pao_count: u8,
    /// Tiles forbidden from being discarded after a call.
    pub forbidden_discards: [u8; 6],
    /// Number of active forbidden discard entries.
    pub forbidden_discard_count: u8,
    /// Per-player MJAI event log for observation diffing.
    pub mjai_log: Vec<String>,
}

impl PlayerState {
    /// Create a new player state with the given starting score.
    pub fn new(starting_score: i32) -> Self {
        Self {
            hand: [0; 14],
            hand_len: 0,
            melds: [Meld::default(); 4],
            meld_count: 0,
            discards: [0; 30],
            discard_from_hand: [false; 30],
            discard_is_riichi: [false; 30],
            discard_len: 0,
            riichi_declaration_index: None,
            score: starting_score,
            score_delta: 0,
            riichi_declared: false,
            riichi_stage: false,
            double_riichi_declared: false,
            missed_agari_riichi: false,
            missed_agari_doujun: false,
            nagashi_eligible: true,
            ippatsu_cycle: false,
            pao: [(0, 0); 3],
            pao_count: 0,
            forbidden_discards: [0; 6],
            forbidden_discard_count: 0,
            mjai_log: Vec::new(),
        }
    }

    /// Returns the hand as a slice.
    #[inline]
    pub fn hand_slice(&self) -> &[u8] {
        &self.hand[..self.hand_len as usize]
    }

    /// Returns the hand as a mutable slice.
    #[inline]
    pub fn hand_slice_mut(&mut self) -> &mut [u8] {
        &mut self.hand[..self.hand_len as usize]
    }

    /// Appends a tile to the end of the hand (caller sorts separately).
    #[inline]
    pub fn push_hand(&mut self, tile: u8) {
        self.hand[self.hand_len as usize] = tile;
        self.hand_len += 1;
    }

    /// Removes the tile at `idx`, shifting remaining elements left.
    #[inline]
    pub fn remove_hand(&mut self, idx: usize) -> u8 {
        let val = self.hand[idx];
        let len = self.hand_len as usize;
        self.hand[idx..len].rotate_left(1);
        self.hand_len -= 1;
        val
    }

    /// Returns the melds as a slice.
    #[inline]
    pub fn melds_slice(&self) -> &[Meld] {
        &self.melds[..self.meld_count as usize]
    }

    /// Returns the melds as a mutable slice.
    #[inline]
    pub fn melds_slice_mut(&mut self) -> &mut [Meld] {
        &mut self.melds[..self.meld_count as usize]
    }

    /// Appends a meld.
    #[inline]
    pub fn push_meld(&mut self, m: Meld) {
        self.melds[self.meld_count as usize] = m;
        self.meld_count += 1;
    }

    /// Returns the discards as a slice.
    #[inline]
    pub fn discards_slice(&self) -> &[u8] {
        &self.discards[..self.discard_len as usize]
    }

    /// Appends a discard with associated metadata.
    #[inline]
    pub fn push_discard(&mut self, tile: u8, from_hand: bool, is_riichi: bool) {
        let i = self.discard_len as usize;
        self.discards[i] = tile;
        self.discard_from_hand[i] = from_hand;
        self.discard_is_riichi[i] = is_riichi;
        self.discard_len += 1;
    }

    /// Reset all round-specific state for the start of a new round.
    pub fn reset_round(&mut self) {
        self.hand_len = 0;
        self.meld_count = 0;
        self.discard_len = 0;
        self.riichi_declaration_index = None;
        self.riichi_declared = false;
        self.riichi_stage = false;
        self.double_riichi_declared = false;
        self.missed_agari_riichi = false;
        self.missed_agari_doujun = false;
        self.nagashi_eligible = true;
        self.ippatsu_cycle = false;
        self.clear_forbidden();
        self.mjai_log.clear();
        self.pao_clear();
    }

    // ---- pao helpers ----

    /// Looks up a pao entry by tile key.
    #[inline]
    pub fn pao_get(&self, key: u8) -> Option<u8> {
        for i in 0..self.pao_count as usize {
            if self.pao[i].0 == key {
                return Some(self.pao[i].1);
            }
        }
        None
    }

    /// Inserts or updates a pao entry.
    #[inline]
    pub fn pao_insert(&mut self, key: u8, val: u8) {
        for i in 0..self.pao_count as usize {
            if self.pao[i].0 == key {
                self.pao[i].1 = val;
                return;
            }
        }
        self.pao[self.pao_count as usize] = (key, val);
        self.pao_count += 1;
    }

    /// Clears all pao entries.
    #[inline]
    pub fn pao_clear(&mut self) {
        self.pao_count = 0;
    }


    // ---- forbidden_discards helpers ----

    /// Returns the current forbidden discards as a slice.
    #[inline]
    pub fn forbidden_slice(&self) -> &[u8] {
        &self.forbidden_discards[..self.forbidden_discard_count as usize]
    }

    /// Appends a tile to the forbidden discards list.
    #[inline]
    pub fn push_forbidden(&mut self, tile: u8) {
        self.forbidden_discards[self.forbidden_discard_count as usize] = tile;
        self.forbidden_discard_count += 1;
    }

    /// Clears all forbidden discards.
    #[inline]
    pub fn clear_forbidden(&mut self) {
        self.forbidden_discard_count = 0;
    }

}
