#![allow(
    clippy::useless_conversion,
    reason = "matches upstream parse and scoring conversions"
)]
use crate::agari;
use crate::errors::RiichiResult;
use crate::score;
use crate::types::{Conditions, Hand, Meld, MeldType, WinResult, Wind};
use crate::yaku;

/// Evaluate a mahjong hand for agari, tenpai, waits, and scoring.
pub struct HandEvaluator {
    /// Normalised tile counts for agari detection (melds reduced).
    pub hand: Hand,
    /// Full tile counts including meld tiles, used for dora and yaku lookup.
    pub full_hand: Hand,
    /// Fixed-size array of declared melds (up to 4).
    pub melds: [Meld; 4],
    /// Number of active melds in the `melds` array.
    pub meld_count: u8,
    /// Count of aka-dora (red five) tiles in hand and melds.
    pub aka_dora_count: u8,
}

impl HandEvaluator {
    /// Returns the active melds as a slice.
    #[inline]
    pub fn melds_slice(&self) -> &[Meld] {
        &self.melds[..self.meld_count as usize]
    }

    /// Parse an MPSZ text string into a `HandEvaluator`.
    pub fn hand_from_text(text: &str) -> RiichiResult<Self> {
        let (tiles, melds) = crate::parser::parse_hand_internal(text)?;
        Ok(Self::new(&tiles, &melds))
    }

    /// Create a new evaluator from 136-format tiles and declared melds.
    pub fn new(tiles_136: &[u8], melds: &[Meld]) -> Self {
        let mut aka_dora_count = 0;

        // Build hand directly without intermediate Vec
        let mut full_hand = Hand::default();
        for &t in tiles_136 {
            if t == 16 || t == 52 || t == 88 {
                aka_dora_count += 1;
            }
            full_hand.add(t / 4);
        }

        let mut hand = full_hand.clone();

        let mut internal_melds = [Meld::default(); 4];
        let mut meld_count = 0u8;

        for meld in melds {
            let mut new_meld = *meld;

            if new_meld.meld_type == MeldType::Daiminkan
                || new_meld.meld_type == MeldType::Ankan
                || new_meld.meld_type == MeldType::Kakan
            {
                let t_34 = meld.tiles[0] / 4;
                if hand.counts[t_34 as usize] == 4 {
                    hand.counts[t_34 as usize] = 3;
                }
            }

            for (i, &t) in meld.tiles_slice().iter().enumerate() {
                if t == 16 || t == 52 || t == 88 {
                    aka_dora_count += 1;
                }
                let t_34 = t / 4;
                new_meld.tiles[i] = t_34;
                full_hand.add(t_34);
            }
            if new_meld.meld_type == MeldType::Chi {
                new_meld.tiles_slice_mut().sort();
            }
            internal_melds[meld_count as usize] = new_meld;
            meld_count += 1;
        }

        Self {
            hand,
            full_hand,
            melds: internal_melds,
            meld_count,
            aka_dora_count,
        }
    }

    /// Calculate the win result for the given winning tile and conditions.
    #[inline]
    pub fn calc(
        &self,
        win_tile: u8,
        dora_indicators: &[u8],
        ura_indicators: &[u8],
        conditions: Option<Conditions>,
    ) -> WinResult {
        let win_tile_136 = win_tile;
        let conditions = conditions.unwrap_or_default();
        let win_tile_34 = win_tile_136 / 4;

        let mut hand_14 = self.hand.clone();
        let mut full_hand_14 = self.full_hand.clone();

        let current_total: u8 = hand_14.counts.iter().sum::<u8>() + (self.meld_count * 3);

        if current_total == 13 {
            hand_14.add(win_tile_34);
            full_hand_14.add(win_tile_34);
        }

        let is_agari = agari::is_agari(&mut hand_14);

        if !is_agari {
            return WinResult::new(false, false, 0, 0, 0, [0; 16], 0, 0, 0, None, false);
        }

        let mut dora_count = 0;
        for &indicator_136 in dora_indicators {
            let next_tile_34 = get_next_tile(indicator_136 / 4);
            dora_count += full_hand_14.counts[next_tile_34 as usize];
        }

        let mut ura_dora_count = 0;
        for &indicator_136 in ura_indicators {
            let next_tile_34 = get_next_tile(indicator_136 / 4);
            ura_dora_count += full_hand_14.counts[next_tile_34 as usize];
        }

        let mut aka_dora = self.aka_dora_count;
        if current_total == 13 && (win_tile_136 == 16 || win_tile_136 == 52 || win_tile_136 == 88) {
            aka_dora += 1;
        }

        let ctx = yaku::YakuContext {
            is_tsumo: conditions.tsumo,
            is_reach: conditions.riichi,
            is_daburu_reach: conditions.double_riichi,
            is_ippatsu: conditions.ippatsu,
            is_haitei: conditions.haitei,
            is_houtei: conditions.houtei,
            is_rinshan: conditions.rinshan,
            is_chankan: conditions.chankan,
            is_tsumo_first_turn: conditions.tsumo_first_turn,
            dora_count,
            aka_dora,
            ura_dora_count,
            round_wind: 27 + conditions.round_wind as u8,
            seat_wind: 27 + conditions.player_wind as u8,
            is_menzen: self.melds_slice().iter().all(|m| !m.opened),
        };

        let _divisions = agari::find_divisions(&hand_14);
        let yaku_res = yaku::calculate_yaku(&hand_14, self.melds_slice(), &ctx, win_tile_34);

        let is_oya = conditions.player_wind == Wind::East;
        // Kazoe yakuman: cap han at 13 for scoring (single yakuman)
        let scoring_han = if yaku_res.yakuman_count == 0 && yaku_res.han >= 13 {
            13
        } else {
            yaku_res.han
        };
        let score_res = score::calculate_score(
            scoring_han,
            yaku_res.fu,
            is_oya,
            conditions.tsumo,
            conditions.honba,
            4, // Always 4 players
        );
        let has_yaku = yaku_res
            .yaku_ids_slice()
            .iter()
            .any(|&id| id != yaku::ID_DORA && id != yaku::ID_AKADORA && id != yaku::ID_URADORA);

        WinResult {
            is_win: (has_yaku || yaku_res.yakuman_count > 0) && yaku_res.han >= 1,
            yakuman: yaku_res.yakuman_count > 0,
            ron_agari: score_res.pay_ron,
            tsumo_agari_oya: score_res.pay_tsumo_oya,
            tsumo_agari_ko: score_res.pay_tsumo_ko,
            yaku: yaku_res.yaku_ids,
            yaku_count: yaku_res.yaku_id_count,
            han: yaku_res.han as u32,
            fu: yaku_res.fu as u32,
            pao_payer: None,
            has_win_shape: true,
        }
    }

    /// Return `true` if the hand is tenpai (one tile away from winning).
    #[inline]
    pub fn is_tenpai(&self) -> bool {
        let current_total: u8 = self.hand.counts.iter().sum::<u8>() + (self.meld_count * 3);
        if current_total != 13 {
            return false;
        }
        let mut hand_14 = self.hand.clone();
        for i in 0..crate::types::TILE_MAX {
            if hand_14.counts[i] < 4 {
                hand_14.add(i as u8);
                if agari::is_agari(&mut hand_14) {
                    return true;
                }
                hand_14.remove(i as u8);
            }
        }
        false
    }

    /// Return the list of winning tile types (34-format) as a `Vec<u8>`.
    #[inline]
    pub fn get_waits_u8(&self) -> Vec<u8> {
        let mut waits = Vec::new();
        let current_total: u8 = self.hand.counts.iter().sum::<u8>() + (self.meld_count * 3);
        if current_total != 13 {
            return waits;
        }
        let mut hand_14 = self.hand.clone();
        for i in 0..crate::types::TILE_MAX {
            if hand_14.counts[i] < 4 {
                hand_14.add(i as u8);
                if crate::agari::is_agari(&mut hand_14) {
                    waits.push(i as u8);
                }
                hand_14.remove(i as u8);
            }
        }
        waits
    }

    /// Get winning tile types into a caller-provided buffer.
    /// Returns the number of waits written.
    #[inline]
    pub fn get_waits_u8_into(&self, buf: &mut [u8; 34]) -> u8 {
        let current_total: u8 = self.hand.counts.iter().sum::<u8>() + (self.meld_count * 3);
        if current_total != 13 {
            return 0;
        }
        let mut count = 0u8;
        let mut hand_14 = self.hand.clone();
        for i in 0..crate::types::TILE_MAX {
            if hand_14.counts[i] < 4 {
                hand_14.add(i as u8);
                if crate::agari::is_agari(&mut hand_14) {
                    buf[count as usize] = i as u8;
                    count += 1;
                }
                hand_14.remove(i as u8);
            }
        }
        count
    }

    /// Return the list of winning tile types as `Vec<u32>`.
    pub fn get_waits(&self) -> Vec<u32> {
        self.get_waits_u8().iter().map(|&x| x as u32).collect()
    }

    #[inline]
    pub fn get_waits_mask(&self) -> u64 {
        let mut waits_buf = [0u8; 34];
        let waits_count = self.get_waits_u8_into(&mut waits_buf);
        let mut mask = 0u64;
        for &tile in &waits_buf[..waits_count as usize] {
            mask |= 1u64 << tile;
        }
        mask
    }
}

/// Return the 136-format tiles that can be discarded to reach tenpai.
pub fn check_riichi_candidates(tiles_136: Vec<u8>) -> Vec<u32> {
    let mut candidates = Vec::new();
    // Convert to 34-tile hand
    let mut tiles_34 = Vec::with_capacity(tiles_136.len());
    for t in &tiles_136 {
        tiles_34.push(t / 4);
    }

    for (i, &t_discard) in tiles_136.iter().enumerate() {
        let mut hand = crate::types::Hand::default();
        for (j, &t) in tiles_34.iter().enumerate() {
            if i != j {
                hand.add(t);
            }
        }

        if agari::is_tenpai(&mut hand) {
            candidates.push(t_discard as u32);
        }
    }
    candidates
}

fn get_next_tile(tile: u8) -> u8 {
    if tile < 9 {
        if tile == 8 {
            0
        } else {
            tile + 1
        }
    } else if tile < 18 {
        if tile == 17 {
            9
        } else {
            tile + 1
        }
    } else if tile < 27 {
        if tile == 26 {
            18
        } else {
            tile + 1
        }
    } else if tile < 31 {
        if tile == 30 {
            27
        } else {
            tile + 1
        }
    } else if tile == 33 {
        31
    } else {
        tile + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Conditions, Meld, MeldType, Wind};

    fn active_yaku_ids(result: &WinResult) -> &[u32] {
        &result.yaku[..result.yaku_count as usize]
    }

    #[test]
    fn hand_from_text_and_wait_helpers_work_for_simple_tenpai_hand() {
        let eval = HandEvaluator::hand_from_text("123m123p123s111z2z")
            .expect("hand text should parse into evaluator");

        assert!(eval.is_tenpai());
        assert_eq!(eval.get_waits_u8(), vec![28]);
        assert_eq!(eval.get_waits(), vec![28u32]);

        let mut buf = [0u8; 34];
        let count = eval.get_waits_u8_into(&mut buf);
        assert_eq!(count, 1);
        assert_eq!(&buf[..count as usize], &[28]);
    }

    #[test]
    fn waits_are_empty_when_tile_count_is_not_thirteen() {
        let eval = HandEvaluator::new(&[0, 4, 8], &[]);

        assert!(!eval.is_tenpai());
        assert!(eval.get_waits_u8().is_empty());
        assert!(eval.get_waits().is_empty());

        let mut buf = [0u8; 34];
        assert_eq!(eval.get_waits_u8_into(&mut buf), 0);
    }

    #[test]
    fn new_sorts_chi_meld_and_counts_aka_tiles_from_hand_and_meld() {
        let meld = Meld::new(MeldType::Chi, &[20, 12, 16], true, 3, Some(16));
        let eval = HandEvaluator::new(&[0, 4, 52], &[meld]);

        assert_eq!(eval.meld_count, 1);
        assert_eq!(eval.aka_dora_count, 2);
        assert_eq!(eval.melds_slice()[0].tiles_slice(), &[3, 4, 5]);
        assert!(eval.melds_slice()[0].opened);
        assert_eq!(eval.hand.counts[0], 1);
        assert_eq!(eval.hand.counts[1], 1);
        assert_eq!(eval.hand.counts[13], 1);
    }

    #[test]
    fn calc_returns_non_win_for_incomplete_hand() {
        let eval = HandEvaluator::new(&[0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 112, 116, 120], &[]);
        let result = eval.calc(124, &[], &[], None);

        assert!(!result.is_win);
        assert!(!result.has_win_shape);
        assert_eq!(result.han, 0);
        assert_eq!(result.fu, 0);
    }

    #[test]
    fn calc_rejects_dora_only_hand_even_when_shape_is_complete() {
        let eval =
            HandEvaluator::hand_from_text("123m456m789p123s5z").expect("tenpai hand should parse");
        let result = eval.calc(124, &[132], &[], Some(Conditions::default()));

        assert!(!result.is_win);
        assert!(result.has_win_shape);
        assert_eq!(
            result.han, 2,
            "pair wait picks up two dora on white dragons only"
        );
        assert_eq!(result.fu, 40);
        assert!(active_yaku_ids(&result).contains(&yaku::ID_DORA));
        assert!(!active_yaku_ids(&result).contains(&yaku::ID_AKADORA));
        assert!(!active_yaku_ids(&result).contains(&yaku::ID_URADORA));
    }

    #[test]
    fn calc_counts_dora_ura_and_aka_on_red_tsumo_scoring_path() {
        let eval =
            HandEvaluator::hand_from_text("123m456m789m234p5s").expect("winning hand should parse");
        let conditions = Conditions {
            tsumo: true,
            riichi: true,
            honba: 1,
            player_wind: Wind::South,
            round_wind: Wind::East,
            ..Default::default()
        };

        let result = eval.calc(88, &[84], &[84], Some(conditions));

        assert!(result.is_win);
        assert!(!result.yakuman);
        assert_eq!(result.han, 9);
        assert_eq!(result.fu, 30);
        assert_eq!(result.tsumo_agari_oya, 8100);
        assert_eq!(result.tsumo_agari_ko, 4100);
        assert_eq!(result.ron_agari, 0);
        assert!(active_yaku_ids(&result).contains(&yaku::ID_RIICHI));
        assert!(active_yaku_ids(&result).contains(&yaku::ID_TSUMO));
        assert!(active_yaku_ids(&result).contains(&yaku::ID_DORA));
        assert!(active_yaku_ids(&result).contains(&yaku::ID_AKADORA));
        assert!(active_yaku_ids(&result).contains(&yaku::ID_URADORA));
        assert!(active_yaku_ids(&result).contains(&yaku::ID_ITTSU));
    }

    #[test]
    fn riichi_candidate_helper_returns_discard_that_leaves_tenpai() {
        let candidates = check_riichi_candidates(vec![
            0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 110, 112, 116,
        ]);

        assert!(candidates.contains(&116));
        assert!(!candidates.is_empty());
    }

    #[test]
    fn get_next_tile_wraps_each_suit_and_honor_group() {
        assert_eq!(get_next_tile(0), 1);
        assert_eq!(get_next_tile(8), 0);
        assert_eq!(get_next_tile(9), 10);
        assert_eq!(get_next_tile(17), 9);
        assert_eq!(get_next_tile(18), 19);
        assert_eq!(get_next_tile(26), 18);
        assert_eq!(get_next_tile(27), 28);
        assert_eq!(get_next_tile(30), 27);
        assert_eq!(get_next_tile(31), 32);
        assert_eq!(get_next_tile(33), 31);
    }
}
