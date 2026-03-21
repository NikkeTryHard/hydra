use crate::action::{Action, ActionType};
use crate::parser::tid_to_mjai;
use crate::types::{Conditions, Wind};
use serde_json::Value;

use super::GameState3P;

impl GameState3P {
    pub fn handle_kita(&mut self, pid: u8, act: &Action) {
        let tile = act
            .tile
            .unwrap_or_else(|| act.consume_slice().first().copied().unwrap_or(0));
        let p_idx = pid as usize;

        // Remove North tile from hand
        if let Some(idx) = self.players[p_idx]
            .hand_slice()
            .iter()
            .position(|&t| t == tile)
        {
            self.players[p_idx].remove_hand(idx);
        }

        // Add to kita_tiles
        self.players[p_idx].push_kita(tile);

        // Kita declaration breaks first-turn status (invalidates Tenhou/Chiihou)
        self.is_first_turn = false;

        // NOTE: Don't break ippatsu here (before chankan ron check).
        // MjSoul awards ippatsu for ron on kita tiles, so the check below
        // must use the pre-kita ippatsu state.  Ippatsu is broken later:
        //  - in the "no ron" branch below (before rinshan draw), or
        //  - when "all pass" resolves pending_kan (mod.rs).

        // Log kita event
        if !self.skip_mjai_logging {
            let mut ev = serde_json::Map::new();
            ev.insert("type".to_string(), Value::String("kita".to_string()));
            ev.insert("actor".to_string(), Value::Number(pid.into()));
            ev.insert("pai".to_string(), Value::String(tid_to_mjai(tile)));
            self._push_mjai_event(Value::Object(ev));
        }

        // Reveal any pending kan dora (e.g. from a prior kakan/daiminkan) before
        // checking for ron on kita.  Kita acts like a discard for dora timing
        // purposes, so the pending dora must be active when evaluating and
        // scoring the ron.  Without this, a kakan→kita→ron sequence would
        // miss the kakan dora in the scoring.
        while self.wall.pending_kan_dora_count > 0 {
            self.wall.pending_kan_dora_count -= 1;
            self._reveal_kan_dora();
        }

        // Check other players for chankan-style ron on kita
        let np: u8 = 3;
        let mut chankan_ronners = Vec::new();
        for i in 0..np {
            if i == pid {
                continue;
            }
            let hand = self.players[i as usize].hand_slice();
            let melds = self.players[i as usize].melds_slice();

            // Furiten check
            let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(hand, melds);
            let waits = calc.get_waits_u8();
            let mut is_furiten = false;
            for &w in &waits {
                if self.players[i as usize]
                    .discards_slice()
                    .iter()
                    .any(|&d| d / 4 == w)
                {
                    is_furiten = true;
                    break;
                }
            }
            if self.players[i as usize].missed_agari_riichi
                || self.players[i as usize].missed_agari_doujun
            {
                is_furiten = true;
            }

            if is_furiten {
                continue;
            }

            let p_wind = (i + np - self.oya) % np;
            let cond = Conditions {
                tsumo: false,
                riichi: self.players[i as usize].riichi_declared,
                double_riichi: self.players[i as usize].double_riichi_declared,
                ippatsu: self.players[i as usize].ippatsu_cycle,
                chankan: false, // Kita does not award chankan yaku
                player_wind: Wind::from(p_wind),
                round_wind: Wind::from(self.round_wind),
                riichi_sticks: self.riichi_sticks,
                honba: self.honba as u32,
                is_sanma: true,
                num_players: np,
                kita_count: self.players[i as usize].kita_count,
                ..Default::default()
            };

            let res = calc.calc(tile, self.wall.dora_indicator_slice(), &[], Some(cond));

            if res.is_win && (res.yakuman || res.han >= 1) {
                chankan_ronners.push(i);
                self.push_claim(
                    i as usize,
                    Action::new(ActionType::Ron, Some(tile), &[], Some(i)),
                );
            }
        }

        if !chankan_ronners.is_empty() {
            // Offer ron to opponents (chankan-style)
            self.phase = crate::action::Phase::WaitResponse;
            self.set_active_players_from_slice(&chankan_ronners);
            self.last_discard = Some((pid, tile));
            // Store kita as pending kan for resolution
            self.pending_kan = Some((pid, *act));
        } else {
            // No ron - break ippatsu for all players, then draw from rinshan
            for p in &mut self.players {
                p.ippatsu_cycle = false;
            }
            self.resolve_kita_rinshan(pid);
        }
    }

    pub fn get_kita_legal_actions(&self, pid: u8) -> Vec<Action> {
        // Must have a drawn tile (it's player's turn to act)
        if self.drawn_tile.is_none() {
            return Vec::new();
        }

        // Must have tiles left in wall (enough for rinshan draw)
        if self.wall.remaining() <= 14 {
            return Vec::new();
        }

        let p_idx = pid as usize;
        let mut actions = Vec::new();

        // Find North tiles (type 30, IDs 120-123) in hand
        for &tile in self.players[p_idx].hand_slice() {
            if tile / 4 == 30 {
                // North wind
                actions.push(Action::new(ActionType::Kita, Some(tile), &[], Some(pid)));
            }
        }

        actions
    }

    pub fn resolve_kita_rinshan(&mut self, pid: u8) {
        let p_idx = pid as usize;

        if self.wall.remaining() > 14 {
            // Reveal any pending kan dora (e.g. from a prior kakan/daiminkan)
            while self.wall.pending_kan_dora_count > 0 {
                self.wall.pending_kan_dora_count -= 1;
                self._reveal_kan_dora();
            }

            // Draw from rinshan via cursor (no memmove)
            let t = self.wall.draw_rinshan();
            self.players[p_idx].push_hand(t);
            self.drawn_tile = Some(t);
            self.wall.rinshan_draw_count += 1;
            self.is_rinshan_flag = true;

            // NO new dora indicator for kita (confirmed Tenhou rule)

            if !self.skip_mjai_logging {
                let mut t_ev = serde_json::Map::new();
                t_ev.insert("type".to_string(), Value::String("tsumo".to_string()));
                t_ev.insert("actor".to_string(), Value::Number(pid.into()));
                t_ev.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
                self._push_mjai_event(Value::Object(t_ev));
            }

            self.phase = crate::action::Phase::WaitAct;
            self.set_single_active_player(pid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Phase;
    use crate::rule::GameRule;

    fn empty_state() -> GameState3P {
        let mut state = GameState3P::new(3, true, Some(7), 0, GameRule::default_mortal_sanma());
        for player in &mut state.players {
            player.reset_round();
            player.score = 35_000;
        }
        state.phase = Phase::WaitAct;
        state.current_player = 0;
        state.active_players = [0; 4];
        state.active_player_count = 1;
        state.is_done = false;
        state.needs_tsumo = false;
        state.is_first_turn = true;
        state.is_rinshan_flag = false;
        state.pending_kan = None;
        state.last_discard = None;
        state.drawn_tile = None;
        state.current_claim_counts = [0; 3];
        state
    }

    #[test]
    fn get_kita_legal_actions_requires_draw_and_wall_headroom() {
        let mut state = empty_state();
        state.players[0].push_hand(120);
        state.players[0].push_hand(121);

        assert!(state.get_kita_legal_actions(0).is_empty());

        state.drawn_tile = Some(0);
        state.wall.tile_count = 14;
        state.wall.draw_cursor = 0;
        assert!(state.get_kita_legal_actions(0).is_empty());

        state.wall.tile_count = 20;
        let actions = state.get_kita_legal_actions(0);
        assert_eq!(actions.len(), 2);
        assert!(actions
            .iter()
            .all(|action| action.action_type == ActionType::Kita));
        assert_eq!(actions[0].tile, Some(120));
        assert_eq!(actions[1].tile, Some(121));
    }

    #[test]
    fn resolve_kita_rinshan_draws_tile_and_reveals_pending_dora() {
        let mut state = empty_state();
        state.wall.tile_count = 20;
        state.wall.draw_cursor = 0;
        state.wall.tiles[0] = 44;
        state.wall.dora_indicator_tiles[1] = 88;
        state.wall.pending_kan_dora_count = 1;

        state.resolve_kita_rinshan(0);

        assert_eq!(state.drawn_tile, Some(44));
        assert_eq!(state.players[0].hand_slice(), &[44]);
        assert_eq!(state.wall.draw_cursor, 1);
        assert_eq!(state.wall.rinshan_draw_count, 1);
        assert_eq!(state.wall.pending_kan_dora_count, 0);
        assert_eq!(
            state.wall.dora_indicator_slice(),
            &[state.wall.dora_indicator_tiles[0], 88]
        );
        assert!(state.is_rinshan_flag);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_slice(), &[0]);
    }

    #[test]
    fn handle_kita_without_ron_removes_tile_and_breaks_ippatsu() {
        let mut state = empty_state();
        state.wall.tile_count = 20;
        state.wall.tiles[0] = 40;
        state.wall.draw_cursor = 0;
        state.drawn_tile = Some(120);
        state.players[0].push_hand(120);
        state.players[0].push_hand(4);
        state.players[1].ippatsu_cycle = true;
        state.players[2].ippatsu_cycle = true;

        let act = Action::new(ActionType::Kita, Some(120), &[], Some(0));
        state.handle_kita(0, &act);

        assert_eq!(state.players[0].kita_slice(), &[120]);
        assert!(!state.players[0].hand_slice().contains(&120));
        assert!(!state.is_first_turn);
        assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
        assert_eq!(state.drawn_tile, Some(40));
        assert_eq!(state.phase, Phase::WaitAct);
    }

    #[test]
    fn handle_kita_with_claim_window_preserves_ippatsu_and_tracks_pending_kita() {
        let mut state = empty_state();
        state.drawn_tile = Some(120);
        state.players[0].push_hand(0);
        state.players[0].push_hand(120);
        state.players[0].hand_slice_mut().sort();
        state.players[1].hand[..13]
            .copy_from_slice(&[36, 40, 44, 72, 76, 80, 96, 100, 104, 108, 109, 110, 121]);
        state.players[1].hand_len = 13;
        state.players[1].riichi_declared = true;
        state.players[1].ippatsu_cycle = true;
        state.players[2].ippatsu_cycle = true;
        state.wall.tile_count = 20;
        state.wall.draw_cursor = 0;

        let act = Action::new(ActionType::Kita, Some(120), &[], Some(0));
        state.handle_kita(0, &act);

        assert_eq!(state.players[0].kita_slice(), &[120]);
        assert_eq!(state.phase, Phase::WaitResponse);
        assert_eq!(state.active_player_slice(), &[1]);
        assert_eq!(state.last_discard, Some((0, 120)));
        assert_eq!(state.pending_kan, Some((0, act)));
        assert!(
            state.players[1].ippatsu_cycle,
            "kita ron window should preserve pre-kita ippatsu state"
        );
        assert!(state.players[2].ippatsu_cycle);
        assert_eq!(state.current_claim_counts[1], 1);
        assert_eq!(state.current_claims[1][0].action_type, ActionType::Ron);
        assert_eq!(state.current_claims[1][0].tile, Some(120));
    }

    #[test]
    fn resolve_kita_rinshan_is_noop_without_wall_headroom() {
        let mut state = empty_state();
        state.drawn_tile = Some(120);
        state.phase = Phase::WaitResponse;
        state.active_players = [1, 2, 0, 0];
        state.active_player_count = 2;
        state.wall.tile_count = 14;
        state.wall.draw_cursor = 0;
        state.wall.pending_kan_dora_count = 1;

        state.resolve_kita_rinshan(0);

        assert_eq!(state.drawn_tile, Some(120));
        assert_eq!(state.wall.draw_cursor, 0);
        assert_eq!(state.wall.rinshan_draw_count, 0);
        assert_eq!(state.wall.pending_kan_dora_count, 1);
        assert_eq!(state.phase, Phase::WaitResponse);
        assert_eq!(state.active_player_slice(), &[1, 2]);
    }
}
