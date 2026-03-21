use crate::action::{Action, ActionType, Phase};
use crate::state::GameState;
use crate::types::{is_terminal_tile, Conditions, Meld, MeldType, Wind};

pub trait GameStateLegalActions {
    fn _get_legal_actions_internal(&self, pid: u8) -> Vec<Action>;
    fn _get_legal_actions_into(&self, pid: u8, buf: &mut Vec<Action>);
    fn _get_claim_actions_for_player(&self, i: u8, pid: u8, tile: u8) -> (Vec<Action>, bool);
}

impl GameStateLegalActions for GameState {
    fn _get_legal_actions_internal(&self, pid: u8) -> Vec<Action> {
        let mut legals = Vec::new();
        let pid_us = pid as usize;
        let mut hand = self.players[pid_us].hand_slice().to_vec();
        hand.sort();

        if self.is_done {
            return legals;
        }

        if self.phase == Phase::WaitAct {
            if pid != self.current_player {
                return legals;
            }

            // 1. Tsumo
            if let Some(tile) = self.drawn_tile {
                if !self.players[pid_us].riichi_stage {
                    let cond = Conditions {
                        tsumo: true,
                        riichi: self.players[pid_us].riichi_declared,
                        double_riichi: self.players[pid_us].double_riichi_declared,
                        ippatsu: self.players[pid_us].ippatsu_cycle,
                        player_wind: Wind::from((pid + 4 - self.oya) % 4),
                        round_wind: Wind::from(self.round_wind),
                        chankan: false,
                        haitei: self.wall.remaining() <= 14 && !self.is_rinshan_flag,
                        houtei: false,
                        rinshan: self.is_rinshan_flag,
                        tsumo_first_turn: self.is_first_turn
                            && (self.players[pid_us].discard_len == 0),
                        riichi_sticks: self.riichi_sticks,
                        honba: self.honba as u32,
                        ..Default::default()
                    };
                    let mut hand = self.players[pid_us].hand_slice().to_vec();
                    if let Some(idx) = hand.iter().rposition(|&t| t == tile) {
                        hand.remove(idx);
                    }
                    let calc = crate::hand_evaluator::HandEvaluator::new(
                        &hand,
                        self.players[pid_us].melds_slice(),
                    );
                    let res = calc.calc(tile, self.wall.dora_indicator_slice(), &[], Some(cond));
                    if res.is_win && (res.yakuman || res.han >= 1) {
                        legals.push(Action::new(ActionType::Tsumo, Some(tile), &[], Some(pid)));
                    }
                }
            }

            // 2. Discard / Riichi
            let declaration_turn = if self.players[pid_us].riichi_declared {
                if let Some(idx) = self.players[pid_us].riichi_declaration_index {
                    self.players[pid_us].discard_len as usize <= idx
                } else {
                    false
                }
            } else {
                false
            };

            if !self.players[pid_us].riichi_declared || declaration_turn {
                let mut forbidden_set = [false; 34];
                for &f in self.players[pid_us].forbidden_slice() {
                    forbidden_set[(f / 4) as usize] = true;
                }
                for &t in self.players[pid_us].hand_slice().iter() {
                    if !forbidden_set[(t / 4) as usize] {
                        legals.push(Action::new(ActionType::Discard, Some(t), &[], Some(pid)));
                    }
                }

                // Riichi check (Only if not already declared)
                if !self.players[pid_us].riichi_declared
                    && self.players[pid_us].score >= 1000
                    && self.wall.remaining() >= 18
                    && self.players[pid_us].melds_slice().iter().all(|m| !m.opened)
                    && !self.players[pid_us].riichi_stage
                {
                    let indices: Vec<usize> = (0..self.players[pid_us].hand_len as usize).collect();
                    let mut can_riichi = false;

                    for &skip_idx in &indices {
                        let mut temp_hand = self.players[pid_us].hand_slice().to_vec();
                        temp_hand.remove(skip_idx);
                        let calc = crate::hand_evaluator::HandEvaluator::new(
                            &temp_hand,
                            self.players[pid_us].melds_slice(),
                        );
                        if calc.is_tenpai() {
                            can_riichi = true;
                            break;
                        }
                    }
                    if can_riichi {
                        legals.push(Action::new(ActionType::Riichi, None, &[], Some(pid)));
                    }
                }
            } else if let Some(dt) = self.drawn_tile {
                legals.push(Action::new(ActionType::Discard, Some(dt), &[], Some(pid)));
            }

            // 3. Kan (Ankan / Kakan)
            if self.wall.remaining() > 14 && self.drawn_tile.is_some() {
                let mut counts = [0; 34];
                for &t in self.players[pid_us].hand_slice() {
                    let idx = t as usize / 4;
                    counts[idx] += 1;
                }

                if !self.players[pid_us].riichi_declared && !self.players[pid_us].riichi_stage {
                    // Ankan
                    for (t_val, &c) in counts.iter().enumerate() {
                        if c == 4 {
                            let lowest = (t_val * 4) as u8;
                            let consume = [lowest, lowest + 1, lowest + 2, lowest + 3];
                            legals.push(Action::new(
                                ActionType::Ankan,
                                Some(lowest),
                                &consume,
                                Some(pid),
                            ));
                        }
                    }
                    // Kakan
                    for m in self.players[pid_us].melds_slice() {
                        if m.meld_type == MeldType::Pon {
                            let target = m.tiles[0] / 4;
                            for &t in self.players[pid_us].hand_slice() {
                                if t / 4 == target {
                                    legals.push(Action::new(
                                        ActionType::Kakan,
                                        Some(t),
                                        m.tiles_slice(),
                                        Some(pid),
                                    ));
                                }
                            }
                        }
                    }
                } else if self.players[pid_us].riichi_declared {
                    // Ankan is only allowed after riichi is declared (not during riichi_stage)
                    // and only if it doesn't change the waits
                    if let Some(t) = self.drawn_tile {
                        let t34 = t / 4;
                        if counts[t34 as usize] == 4 {
                            // Check waits
                            let mut hand_pre = self.players[pid_us].hand_slice().to_vec();
                            if let Some(pos) = hand_pre.iter().position(|&x| x == t) {
                                hand_pre.remove(pos);
                            }
                            let calc_pre = crate::hand_evaluator::HandEvaluator::new(
                                &hand_pre,
                                self.players[pid_us].melds_slice(),
                            );
                            let mut waits_pre = calc_pre.get_waits();
                            waits_pre.sort();

                            let mut hand_post = self.players[pid_us].hand_slice().to_vec();
                            hand_post.retain(|&x| x / 4 != t34);
                            let mut melds_post = self.players[pid_us].melds_slice().to_vec();
                            let lowest = t34 * 4;
                            melds_post.push(Meld::new(
                                MeldType::Ankan,
                                &[lowest, lowest + 1, lowest + 2, lowest + 3],
                                false,
                                -1,
                                None,
                            ));
                            let calc_post =
                                crate::hand_evaluator::HandEvaluator::new(&hand_post, &melds_post);
                            let mut waits_post = calc_post.get_waits();
                            waits_post.sort();

                            if waits_pre == waits_post && !waits_pre.is_empty() {
                                let consume = [lowest, lowest + 1, lowest + 2, lowest + 3];
                                legals.push(Action::new(
                                    ActionType::Ankan,
                                    Some(lowest),
                                    &consume,
                                    Some(pid),
                                ));
                            }
                        }
                    }
                }
            }

            // 4. Kyushu Kyuhai (Abortive Draw)
            // Simplified check: Check if all melds of all players are empty? No, Kyusyu Kyuhai is usually only valid if NO ONE has called.
            // But here we emulate generic rules.
            // Original code: if self.is_first_turn && self.melds.iter().all(|m| m.is_empty()) -> This meant check all players' melds?
            // In original GameState, melds was [Vec<Meld>; 4]. so self.melds.iter().all... checked all 4 vectors.
            let no_calls = self.players.iter().all(|p| p.meld_count == 0);

            if self.is_first_turn && no_calls && !self.players[pid_us].riichi_stage {
                let mut terminal_bits: u64 = 0;
                for &t in self.players[pid_us].hand_slice() {
                    if is_terminal_tile(t) {
                        terminal_bits |= 1u64 << (t / 4);
                    }
                }
                if terminal_bits.count_ones() >= 9 {
                    legals.push(Action::new(ActionType::KyushuKyuhai, None, &[], Some(pid)));
                }
            }
        } else if self.phase == Phase::WaitResponse {
            let claims = self.claims_slice(pid as usize);
            if !claims.is_empty() {
                legals.extend(claims.iter().copied());
            }
            // Always offer Pass
            legals.push(Action::new(ActionType::Pass, None, &[], Some(pid)));
        }
        legals
    }

    fn _get_legal_actions_into(&self, pid: u8, buf: &mut Vec<Action>) {
        let pid_us = pid as usize;

        if self.is_done {
            return;
        }

        if self.phase == Phase::WaitAct {
            if pid != self.current_player {
                return;
            }

            // 1. Tsumo
            if let Some(tile) = self.drawn_tile {
                if !self.players[pid_us].riichi_stage {
                    let cond = Conditions {
                        tsumo: true,
                        riichi: self.players[pid_us].riichi_declared,
                        double_riichi: self.players[pid_us].double_riichi_declared,
                        ippatsu: self.players[pid_us].ippatsu_cycle,
                        player_wind: Wind::from((pid + 4 - self.oya) % 4),
                        round_wind: Wind::from(self.round_wind),
                        chankan: false,
                        haitei: self.wall.remaining() <= 14 && !self.is_rinshan_flag,
                        houtei: false,
                        rinshan: self.is_rinshan_flag,
                        tsumo_first_turn: self.is_first_turn
                            && (self.players[pid_us].discard_len == 0),
                        riichi_sticks: self.riichi_sticks,
                        honba: self.honba as u32,
                        ..Default::default()
                    };
                    // Build hand without drawn tile on stack (no clone)
                    let hand = self.players[pid_us].hand_slice();
                    let mut temp = [0u8; 14];
                    let mut temp_len = 0usize;
                    let mut skipped = false;
                    for &t in hand.iter() {
                        if !skipped && t == tile {
                            skipped = true;
                            continue;
                        }
                        temp[temp_len] = t;
                        temp_len += 1;
                    }
                    let calc = crate::hand_evaluator::HandEvaluator::new(
                        &temp[..temp_len],
                        self.players[pid_us].melds_slice(),
                    );
                    let res = calc.calc(tile, self.wall.dora_indicator_slice(), &[], Some(cond));
                    if res.is_win && (res.yakuman || res.han >= 1) {
                        buf.push(Action::new(ActionType::Tsumo, Some(tile), &[], Some(pid)));
                    }
                }
            }

            // 2. Discard / Riichi
            let declaration_turn = if self.players[pid_us].riichi_declared {
                if let Some(idx) = self.players[pid_us].riichi_declaration_index {
                    self.players[pid_us].discard_len as usize <= idx
                } else {
                    false
                }
            } else {
                false
            };

            if !self.players[pid_us].riichi_declared || declaration_turn {
                let mut forbidden_set = [false; 34];
                for &f in self.players[pid_us].forbidden_slice() {
                    forbidden_set[(f / 4) as usize] = true;
                }
                for &t in self.players[pid_us].hand_slice().iter() {
                    if !forbidden_set[(t / 4) as usize] {
                        buf.push(Action::new(ActionType::Discard, Some(t), &[], Some(pid)));
                    }
                }

                // Riichi check (Only if not already declared)
                if !self.players[pid_us].riichi_declared
                    && self.players[pid_us].score >= 1000
                    && self.wall.remaining() >= 18
                    && self.players[pid_us].melds_slice().iter().all(|m| !m.opened)
                    && !self.players[pid_us].riichi_stage
                {
                    let hand = self.players[pid_us].hand_slice();
                    let hand_len = hand.len();
                    let mut can_riichi = false;

                    // In-place remove/restore to avoid clone
                    let mut temp = [0u8; 14];
                    temp[..hand_len].copy_from_slice(hand);
                    for skip_idx in 0..hand_len {
                        // Build hand without tile at skip_idx
                        let mut check = [0u8; 13];
                        let mut ci = 0;
                        for (i, &t) in temp[..hand_len].iter().enumerate() {
                            if i == skip_idx {
                                continue;
                            }
                            check[ci] = t;
                            ci += 1;
                        }
                        let calc = crate::hand_evaluator::HandEvaluator::new(
                            &check[..ci],
                            self.players[pid_us].melds_slice(),
                        );
                        if calc.is_tenpai() {
                            can_riichi = true;
                            break;
                        }
                    }
                    if can_riichi {
                        buf.push(Action::new(ActionType::Riichi, None, &[], Some(pid)));
                    }
                }
            } else if let Some(dt) = self.drawn_tile {
                buf.push(Action::new(ActionType::Discard, Some(dt), &[], Some(pid)));
            }

            // 3. Kan (Ankan / Kakan)
            if self.wall.remaining() > 14 && self.drawn_tile.is_some() {
                let mut counts = [0; 34];
                for &t in self.players[pid_us].hand_slice() {
                    let idx = t as usize / 4;
                    counts[idx] += 1;
                }

                if !self.players[pid_us].riichi_declared && !self.players[pid_us].riichi_stage {
                    // Ankan
                    for (t_val, &c) in counts.iter().enumerate() {
                        if c == 4 {
                            let lowest = (t_val * 4) as u8;
                            let consume = [lowest, lowest + 1, lowest + 2, lowest + 3];
                            buf.push(Action::new(
                                ActionType::Ankan,
                                Some(lowest),
                                &consume,
                                Some(pid),
                            ));
                        }
                    }
                    // Kakan
                    for m in self.players[pid_us].melds_slice() {
                        if m.meld_type == MeldType::Pon {
                            let target = m.tiles[0] / 4;
                            for &t in self.players[pid_us].hand_slice() {
                                if t / 4 == target {
                                    buf.push(Action::new(
                                        ActionType::Kakan,
                                        Some(t),
                                        m.tiles_slice(),
                                        Some(pid),
                                    ));
                                }
                            }
                        }
                    }
                } else if self.players[pid_us].riichi_declared {
                    // Ankan after riichi: only if it doesn't change waits
                    if let Some(t) = self.drawn_tile {
                        let t34 = t / 4;
                        if counts[t34 as usize] == 4 {
                            // Build hand_pre (hand without drawn tile) on stack
                            let hand = self.players[pid_us].hand_slice();
                            let hand_len = hand.len();
                            let mut pre = [0u8; 14];
                            let mut pre_len = 0usize;
                            let mut skipped = false;
                            for &x in hand.iter() {
                                if !skipped && x == t {
                                    skipped = true;
                                    continue;
                                }
                                pre[pre_len] = x;
                                pre_len += 1;
                            }
                            let calc_pre = crate::hand_evaluator::HandEvaluator::new(
                                &pre[..pre_len],
                                self.players[pid_us].melds_slice(),
                            );
                            let mut waits_pre = calc_pre.get_waits();
                            waits_pre.sort();

                            // Build hand_post (hand without tiles of this type)
                            let mut post = [0u8; 14];
                            let mut post_len = 0usize;
                            for &tile_val in &hand[..hand_len] {
                                if tile_val / 4 != t34 {
                                    post[post_len] = tile_val;
                                    post_len += 1;
                                }
                            }
                            let mut melds_post = self.players[pid_us].melds_slice().to_vec();
                            let lowest = t34 * 4;
                            melds_post.push(Meld::new(
                                MeldType::Ankan,
                                &[lowest, lowest + 1, lowest + 2, lowest + 3],
                                false,
                                -1,
                                None,
                            ));
                            let calc_post = crate::hand_evaluator::HandEvaluator::new(
                                &post[..post_len],
                                &melds_post,
                            );
                            let mut waits_post = calc_post.get_waits();
                            waits_post.sort();

                            if waits_pre == waits_post && !waits_pre.is_empty() {
                                let consume = [lowest, lowest + 1, lowest + 2, lowest + 3];
                                buf.push(Action::new(
                                    ActionType::Ankan,
                                    Some(lowest),
                                    &consume,
                                    Some(pid),
                                ));
                            }
                        }
                    }
                }
            }

            // 4. Kyushu Kyuhai (Abortive Draw)
            let no_calls = self.players.iter().all(|p| p.meld_count == 0);

            if self.is_first_turn && no_calls && !self.players[pid_us].riichi_stage {
                let mut terminal_bits: u64 = 0;
                for &t in self.players[pid_us].hand_slice() {
                    if is_terminal_tile(t) {
                        terminal_bits |= 1u64 << (t / 4);
                    }
                }
                if terminal_bits.count_ones() >= 9 {
                    buf.push(Action::new(ActionType::KyushuKyuhai, None, &[], Some(pid)));
                }
            }
        } else if self.phase == Phase::WaitResponse {
            let claims = self.claims_slice(pid as usize);
            if !claims.is_empty() {
                buf.extend(claims.iter().copied());
            }
            // Always offer Pass
            buf.push(Action::new(ActionType::Pass, None, &[], Some(pid)));
        }
    }
    fn _get_claim_actions_for_player(&self, i: u8, pid: u8, tile: u8) -> (Vec<Action>, bool) {
        let mut legals = Vec::new();
        let mut missed_agari = false;
        let i_us = i as usize;
        let hand = self.players[i_us].hand_slice();
        let melds = self.players[i_us].melds_slice();

        // 1. Ron
        let tile_class = tile / 4;
        let in_discards = self.players[i_us]
            .discards_slice()
            .iter()
            .any(|&d| d / 4 == tile_class);
        let in_missed = self.players[i_us].missed_agari_doujun
            || (self.players[i_us].riichi_declared && self.players[i_us].missed_agari_riichi);

        if !in_discards && !in_missed {
            let calc = crate::hand_evaluator::HandEvaluator::new(hand, melds);
            let p_wind = (i + 4 - self.oya) % 4;
            let cond = Conditions {
                tsumo: false,
                riichi: self.players[i_us].riichi_declared,
                double_riichi: self.players[i_us].double_riichi_declared,
                ippatsu: self.players[i_us].ippatsu_cycle,
                player_wind: Wind::from(p_wind),
                round_wind: Wind::from(self.round_wind),
                chankan: false,
                haitei: false,
                houtei: self.wall.remaining() <= 14 && !self.is_rinshan_flag,
                rinshan: false,
                tsumo_first_turn: false,
                riichi_sticks: self.riichi_sticks,
                honba: self.honba as u32,
                ..Default::default()
            };

            let mut is_furiten = false;
            let waits = calc.get_waits_u8();
            let mut discard_set = [false; 34];
            for &d in &self.players[i_us].discards {
                discard_set[(d / 4) as usize] = true;
            }
            for &w in &waits {
                if discard_set[w as usize] {
                    is_furiten = true;
                    break;
                }
            }
            if self.players[i_us].missed_agari_riichi || self.players[i_us].missed_agari_doujun {
                is_furiten = true;
            }

            if !is_furiten {
                let res = calc.calc(tile, self.wall.dora_indicator_slice(), &[], Some(cond));
                if res.is_win {
                    legals.push(Action::new(ActionType::Ron, Some(tile), &[], Some(i)));
                } else if res.has_win_shape {
                    missed_agari = true;
                }
            }
        }

        // 2. Pon / Kan
        if !self.players[i_us].riichi_declared && self.wall.remaining() > 14 {
            let count = hand.iter().filter(|&&t| t / 4 == tile / 4).count();
            if count >= 2 && hand.len() >= 3 {
                let check_pon_kuikae = |consumes: &[u8]| -> bool {
                    let forbidden_tile: Option<u8> = if self.rule.kuikae_forbidden {
                        Some(tile / 4)
                    } else {
                        None
                    };
                    let (mut used_0, mut used_1) = (false, false);
                    for &t in hand.iter() {
                        let mut consumed_this = false;
                        if !used_0 && consumes[0] == t {
                            used_0 = true;
                            consumed_this = true;
                        } else if !used_1 && consumes[1] == t {
                            used_1 = true;
                            consumed_this = true;
                        }
                        if consumed_this {
                            continue;
                        }
                        if forbidden_tile != Some(t / 4) {
                            return true;
                        }
                    }
                    false
                };

                // Generate all distinct pon consume pairs.
                // When a player has 3 copies of a tile (e.g. red 5m + 5m + 5m),
                // we need separate pon options with and without the red five.
                let mut matching = [0u8; 3];
                let mut matching_len = 0u8;
                for &t in hand.iter() {
                    if t / 4 == tile / 4 {
                        matching[matching_len as usize] = t;
                        matching_len += 1;
                    }
                }
                let mut seen_pairs = [(0u8, 0u8); 3];
                let mut seen_len = 0u8;
                for a in 0..matching_len as usize {
                    for b in (a + 1)..matching_len as usize {
                        let pair = (matching[a], matching[b]);
                        if !seen_pairs[..seen_len as usize].contains(&pair) {
                            seen_pairs[seen_len as usize] = pair;
                            seen_len += 1;
                            let consumes = [pair.0, pair.1];
                            if check_pon_kuikae(&consumes) {
                                legals.push(Action::new(
                                    ActionType::Pon,
                                    Some(tile),
                                    &consumes,
                                    Some(i),
                                ));
                            }
                        }
                    }
                }
            }
            if count >= 3 {
                let mut consumes = [0u8; 3];
                let mut ci = 0usize;
                for &t in hand.iter() {
                    if t / 4 == tile / 4 {
                        consumes[ci] = t;
                        ci += 1;
                        if ci == 3 {
                            break;
                        }
                    }
                }
                legals.push(Action::new(
                    ActionType::Daiminkan,
                    Some(tile),
                    &consumes,
                    Some(i),
                ));
            }
        }

        // 3. Chi
        let is_shimocha = i == (pid + 1) % 4;
        if !self.players[i_us].riichi_declared
            && self.wall.remaining() > 14
            && is_shimocha
            && hand.len() >= 3
        {
            let t_val = tile / 4;
            if t_val < 27 {
                let check_chi_kuikae = |c1: u8, c2: u8| -> bool {
                    let mut chi_forbidden = [0u8; 2];
                    let mut chi_forbidden_len = 0u8;
                    if self.rule.kuikae_forbidden {
                        chi_forbidden[chi_forbidden_len as usize] = t_val;
                        chi_forbidden_len += 1;
                        let mut cons_34 = [c1 / 4, c2 / 4];
                        cons_34.sort();
                        if cons_34[0] == t_val + 1 && cons_34[1] == t_val + 2 {
                            if t_val % 9 <= 5 {
                                chi_forbidden[chi_forbidden_len as usize] = t_val + 3;
                                chi_forbidden_len += 1;
                            }
                        } else if t_val >= 2
                            && cons_34[1] == t_val - 1
                            && cons_34[0] == t_val - 2
                            && t_val % 9 >= 3
                        {
                            chi_forbidden[chi_forbidden_len as usize] = t_val - 3;
                            chi_forbidden_len += 1;
                        }
                    }
                    let mut used_c1 = false;
                    let mut used_c2 = false;
                    for &t in hand.iter() {
                        if !used_c1 && t == c1 {
                            used_c1 = true;
                            continue;
                        }
                        if !used_c2 && t == c2 {
                            used_c2 = true;
                            continue;
                        }
                        if !chi_forbidden[..chi_forbidden_len as usize].contains(&(t / 4)) {
                            return true;
                        }
                    }
                    false
                };

                // Pattern 1: t-2, t-1, t
                if t_val % 9 >= 2 {
                    let mut c1_opts = [0u8; 4];
                    let mut c1_len = 0u8;
                    let mut c2_opts = [0u8; 4];
                    let mut c2_len = 0u8;
                    for &t in hand.iter() {
                        if t / 4 == t_val - 2 {
                            c1_opts[c1_len as usize] = t;
                            c1_len += 1;
                        } else if t / 4 == t_val - 1 {
                            c2_opts[c2_len as usize] = t;
                            c2_len += 1;
                        }
                    }
                    for &c1 in &c1_opts[..c1_len as usize] {
                        for &c2 in &c2_opts[..c2_len as usize] {
                            if check_chi_kuikae(c1, c2) {
                                legals.push(Action::new(
                                    ActionType::Chi,
                                    Some(tile),
                                    &[c1, c2],
                                    Some(i),
                                ));
                            }
                        }
                    }
                }
                // Pattern 2: t-1, t, t+1
                if t_val % 9 >= 1 && t_val % 9 <= 7 {
                    let mut c1_opts = [0u8; 4];
                    let mut c1_len = 0u8;
                    let mut c2_opts = [0u8; 4];
                    let mut c2_len = 0u8;
                    for &t in hand.iter() {
                        if t / 4 == t_val - 1 {
                            c1_opts[c1_len as usize] = t;
                            c1_len += 1;
                        } else if t / 4 == t_val + 1 {
                            c2_opts[c2_len as usize] = t;
                            c2_len += 1;
                        }
                    }
                    for &c1 in &c1_opts[..c1_len as usize] {
                        for &c2 in &c2_opts[..c2_len as usize] {
                            if check_chi_kuikae(c1, c2) {
                                legals.push(Action::new(
                                    ActionType::Chi,
                                    Some(tile),
                                    &[c1, c2],
                                    Some(i),
                                ));
                            }
                        }
                    }
                }
                // Pattern 3: t, t+1, t+2
                if t_val % 9 <= 6 {
                    let mut c1_opts = [0u8; 4];
                    let mut c1_len = 0u8;
                    let mut c2_opts = [0u8; 4];
                    let mut c2_len = 0u8;
                    for &t in hand.iter() {
                        if t / 4 == t_val + 1 {
                            c1_opts[c1_len as usize] = t;
                            c1_len += 1;
                        } else if t / 4 == t_val + 2 {
                            c2_opts[c2_len as usize] = t;
                            c2_len += 1;
                        }
                    }
                    for &c1 in &c1_opts[..c1_len as usize] {
                        for &c2 in &c2_opts[..c2_len as usize] {
                            if check_chi_kuikae(c1, c2) {
                                legals.push(Action::new(
                                    ActionType::Chi,
                                    Some(tile),
                                    &[c1, c2],
                                    Some(i),
                                ));
                            }
                        }
                    }
                }
            }
        }

        (legals, missed_agari)
    }
}

impl GameState {
    /// Get claim actions for a player, writing directly into current_claims.
    /// Returns (action_count, missed_agari).
    pub fn _get_claim_actions_into_claims(&mut self, i: u8, pid: u8, tile: u8) -> (usize, bool) {
        self.current_claim_counts[i as usize] = 0;
        let mut missed_agari = false;
        let i_us = i as usize;

        // Copy player data into local buffers to avoid borrow conflicts with &mut self
        let hand_buf = self.players[i_us].hand;
        let hand_len = self.players[i_us].hand_len as usize;
        let hand = &hand_buf[..hand_len];
        let melds_buf = self.players[i_us].melds;
        let meld_count = self.players[i_us].meld_count as usize;
        let melds = &melds_buf[..meld_count];
        let discards_buf = self.players[i_us].discards;
        let discard_len = self.players[i_us].discard_len as usize;
        let riichi_declared = self.players[i_us].riichi_declared;
        let double_riichi_declared = self.players[i_us].double_riichi_declared;
        let ippatsu_cycle = self.players[i_us].ippatsu_cycle;
        let missed_agari_doujun = self.players[i_us].missed_agari_doujun;
        let missed_agari_riichi = self.players[i_us].missed_agari_riichi;
        let kuikae_forbidden = self.rule.kuikae_forbidden;

        // 1. Ron
        let tile_class = tile / 4;
        let in_discards = discards_buf[..discard_len]
            .iter()
            .any(|&d| d / 4 == tile_class);
        let in_missed = missed_agari_doujun || (riichi_declared && missed_agari_riichi);

        if !in_discards && !in_missed {
            let calc = crate::hand_evaluator::HandEvaluator::new(hand, melds);
            let p_wind = (i + 4 - self.oya) % 4;
            let cond = Conditions {
                tsumo: false,
                riichi: riichi_declared,
                double_riichi: double_riichi_declared,
                ippatsu: ippatsu_cycle,
                player_wind: Wind::from(p_wind),
                round_wind: Wind::from(self.round_wind),
                chankan: false,
                haitei: false,
                houtei: self.wall.remaining() <= 14 && !self.is_rinshan_flag,
                rinshan: false,
                tsumo_first_turn: false,
                riichi_sticks: self.riichi_sticks,
                honba: self.honba as u32,
                ..Default::default()
            };

            let mut is_furiten = false;
            let mut waits_buf = [0u8; 34];
            let waits_count = calc.get_waits_u8_into(&mut waits_buf);
            let waits = &waits_buf[..waits_count as usize];
            let mut discard_set = [false; 34];
            for &d in &discards_buf[..discard_len] {
                discard_set[(d / 4) as usize] = true;
            }
            for &w in waits {
                if discard_set[w as usize] {
                    is_furiten = true;
                    break;
                }
            }
            if missed_agari_riichi || missed_agari_doujun {
                is_furiten = true;
            }

            if !is_furiten {
                let res = calc.calc(tile, self.wall.dora_indicator_slice(), &[], Some(cond));
                if res.is_win {
                    self.push_claim(i_us, Action::new(ActionType::Ron, Some(tile), &[], Some(i)));
                } else if res.has_win_shape {
                    missed_agari = true;
                }
            }
        }

        // 2. Pon / Kan
        if !riichi_declared && self.wall.remaining() > 14 {
            let count = hand.iter().filter(|&&t| t / 4 == tile / 4).count();
            if count >= 2 && hand.len() >= 3 {
                let check_pon_kuikae = |consumes: &[u8]| -> bool {
                    let forbidden_tile: Option<u8> = if kuikae_forbidden {
                        Some(tile / 4)
                    } else {
                        None
                    };
                    let (mut used_0, mut used_1) = (false, false);
                    for &t in hand.iter() {
                        let mut consumed_this = false;
                        if !used_0 && consumes[0] == t {
                            used_0 = true;
                            consumed_this = true;
                        } else if !used_1 && consumes[1] == t {
                            used_1 = true;
                            consumed_this = true;
                        }
                        if consumed_this {
                            continue;
                        }
                        if forbidden_tile != Some(t / 4) {
                            return true;
                        }
                    }
                    false
                };

                let mut matching = [0u8; 3];
                let mut matching_len = 0u8;
                for &t in hand.iter() {
                    if t / 4 == tile / 4 {
                        matching[matching_len as usize] = t;
                        matching_len += 1;
                    }
                }
                let mut seen_pairs = [(0u8, 0u8); 3];
                let mut seen_len = 0u8;
                for a in 0..matching_len as usize {
                    for b in (a + 1)..matching_len as usize {
                        let pair = (matching[a], matching[b]);
                        if !seen_pairs[..seen_len as usize].contains(&pair) {
                            seen_pairs[seen_len as usize] = pair;
                            seen_len += 1;
                            let consumes = [pair.0, pair.1];
                            if check_pon_kuikae(&consumes) {
                                self.push_claim(
                                    i_us,
                                    Action::new(ActionType::Pon, Some(tile), &consumes, Some(i)),
                                );
                            }
                        }
                    }
                }
            }
            if count >= 3 {
                let mut consumes = [0u8; 3];
                let mut ci = 0usize;
                for &t in hand.iter() {
                    if t / 4 == tile / 4 {
                        consumes[ci] = t;
                        ci += 1;
                        if ci == 3 {
                            break;
                        }
                    }
                }
                self.push_claim(
                    i_us,
                    Action::new(ActionType::Daiminkan, Some(tile), &consumes, Some(i)),
                );
            }
        }

        // 3. Chi
        let is_shimocha = i == (pid + 1) % 4;
        if !riichi_declared && self.wall.remaining() > 14 && is_shimocha && hand.len() >= 3 {
            let t_val = tile / 4;
            if t_val < 27 {
                let check_chi_kuikae = |c1: u8, c2: u8| -> bool {
                    let mut chi_forbidden = [0u8; 2];
                    let mut chi_forbidden_len = 0u8;
                    if kuikae_forbidden {
                        chi_forbidden[chi_forbidden_len as usize] = t_val;
                        chi_forbidden_len += 1;
                        let mut cons_34 = [c1 / 4, c2 / 4];
                        cons_34.sort();
                        if cons_34[0] == t_val + 1 && cons_34[1] == t_val + 2 {
                            if t_val % 9 <= 5 {
                                chi_forbidden[chi_forbidden_len as usize] = t_val + 3;
                                chi_forbidden_len += 1;
                            }
                        } else if t_val >= 2
                            && cons_34[1] == t_val - 1
                            && cons_34[0] == t_val - 2
                            && t_val % 9 >= 3
                        {
                            chi_forbidden[chi_forbidden_len as usize] = t_val - 3;
                            chi_forbidden_len += 1;
                        }
                    }
                    let mut used_c1 = false;
                    let mut used_c2 = false;
                    for &t in hand.iter() {
                        if !used_c1 && t == c1 {
                            used_c1 = true;
                            continue;
                        }
                        if !used_c2 && t == c2 {
                            used_c2 = true;
                            continue;
                        }
                        if !chi_forbidden[..chi_forbidden_len as usize].contains(&(t / 4)) {
                            return true;
                        }
                    }
                    false
                };

                // Pattern 1: t-2, t-1, t
                if t_val % 9 >= 2 {
                    let mut c1_opts = [0u8; 4];
                    let mut c1_len = 0u8;
                    let mut c2_opts = [0u8; 4];
                    let mut c2_len = 0u8;
                    for &t in hand.iter() {
                        if t / 4 == t_val - 2 {
                            c1_opts[c1_len as usize] = t;
                            c1_len += 1;
                        } else if t / 4 == t_val - 1 {
                            c2_opts[c2_len as usize] = t;
                            c2_len += 1;
                        }
                    }
                    for &c1 in &c1_opts[..c1_len as usize] {
                        for &c2 in &c2_opts[..c2_len as usize] {
                            if check_chi_kuikae(c1, c2) {
                                self.push_claim(
                                    i_us,
                                    Action::new(ActionType::Chi, Some(tile), &[c1, c2], Some(i)),
                                );
                            }
                        }
                    }
                }
                // Pattern 2: t-1, t, t+1
                if t_val % 9 >= 1 && t_val % 9 <= 7 {
                    let mut c1_opts = [0u8; 4];
                    let mut c1_len = 0u8;
                    let mut c2_opts = [0u8; 4];
                    let mut c2_len = 0u8;
                    for &t in hand.iter() {
                        if t / 4 == t_val - 1 {
                            c1_opts[c1_len as usize] = t;
                            c1_len += 1;
                        } else if t / 4 == t_val + 1 {
                            c2_opts[c2_len as usize] = t;
                            c2_len += 1;
                        }
                    }
                    for &c1 in &c1_opts[..c1_len as usize] {
                        for &c2 in &c2_opts[..c2_len as usize] {
                            if check_chi_kuikae(c1, c2) {
                                self.push_claim(
                                    i_us,
                                    Action::new(ActionType::Chi, Some(tile), &[c1, c2], Some(i)),
                                );
                            }
                        }
                    }
                }
                // Pattern 3: t, t+1, t+2
                if t_val % 9 <= 6 {
                    let mut c1_opts = [0u8; 4];
                    let mut c1_len = 0u8;
                    let mut c2_opts = [0u8; 4];
                    let mut c2_len = 0u8;
                    for &t in hand.iter() {
                        if t / 4 == t_val + 1 {
                            c1_opts[c1_len as usize] = t;
                            c1_len += 1;
                        } else if t / 4 == t_val + 2 {
                            c2_opts[c2_len as usize] = t;
                            c2_len += 1;
                        }
                    }
                    for &c1 in &c1_opts[..c1_len as usize] {
                        for &c2 in &c2_opts[..c2_len as usize] {
                            if check_chi_kuikae(c1, c2) {
                                self.push_claim(
                                    i_us,
                                    Action::new(ActionType::Chi, Some(tile), &[c1, c2], Some(i)),
                                );
                            }
                        }
                    }
                }
            }
        }

        (self.current_claim_counts[i_us] as usize, missed_agari)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse_hand, parse_tile};
    use crate::rule::GameRule;

    fn empty_state() -> GameState {
        let mut state = GameState::new(0, true, Some(7), 0, GameRule::default_tenhou());
        for player in &mut state.players {
            player.reset_round();
            player.score = 25_000;
        }
        state.is_done = false;
        state.needs_tsumo = false;
        state.needs_initialize_next_round = false;
        state.pending_oya_won = false;
        state.pending_is_draw = false;
        state.riichi_sticks = 0;
        state.phase = Phase::WaitAct;
        state.current_player = 0;
        state.active_players = [0; 4];
        state.active_player_count = 1;
        state.last_discard = None;
        state.current_claim_counts = [0; 4];
        state.pending_kan = None;
        state.oya = 0;
        state.honba = 0;
        state.round_wind = 0;
        state.is_rinshan_flag = false;
        state.is_first_turn = false;
        state.riichi_pending_acceptance = None;
        state.drawn_tile = None;
        state.is_after_kan = false;
        state
    }

    fn set_wall_remaining(state: &mut GameState, remaining: u8) {
        state.wall.tile_count = remaining;
        state.wall.draw_cursor = 0;
    }

    fn add_hand_from_text(state: &mut GameState, pid: usize, text: &str) {
        let (hand, melds) = parse_hand(text).unwrap();
        for tile in hand {
            state.players[pid].push_hand(tile as u8);
        }
        for meld in melds {
            state.players[pid].push_meld(meld);
        }
    }

    fn tile(text: &str) -> u8 {
        parse_tile(text).unwrap()
    }

    fn action_key(action: &Action) -> String {
        format!(
            "{:?}:{:?}:{:?}:{:?}",
            action.action_type,
            action.tile,
            action.consume_slice(),
            action.actor
        )
    }

    fn assert_actions_match(expected: &[Action], actual: &[Action]) {
        assert_eq!(expected.len(), actual.len());
        assert_eq!(
            expected.iter().map(action_key).collect::<Vec<_>>(),
            actual.iter().map(action_key).collect::<Vec<_>>()
        );
    }

    fn assert_legals_match(state: &GameState, pid: u8) -> Vec<Action> {
        let expected = state._get_legal_actions_internal(pid);
        let mut buf = Vec::new();
        state._get_legal_actions_into(pid, &mut buf);
        assert_actions_match(&expected, &buf);
        expected
    }

    fn assert_claim_actions_match(
        state: &mut GameState,
        i: u8,
        pid: u8,
        tile: u8,
    ) -> (Vec<Action>, bool) {
        let (expected, missed_agari) = state._get_claim_actions_for_player(i, pid, tile);
        let (count, missed_agari_into) = state._get_claim_actions_into_claims(i, pid, tile);
        let actual = state.current_claims[i as usize][..count].to_vec();
        assert_actions_match(&expected, &actual);
        assert_eq!(missed_agari, missed_agari_into);
        (expected, missed_agari)
    }

    fn has_action<F>(actions: &[Action], action_type: ActionType, predicate: F) -> bool
    where
        F: Fn(&Action) -> bool,
    {
        actions
            .iter()
            .any(|action| action.action_type == action_type && predicate(action))
    }

    #[test]
    fn wait_act_returns_empty_for_non_current_player() {
        let state = empty_state();
        assert!(state._get_legal_actions_internal(1).is_empty());
    }

    #[test]
    fn wait_act_offers_discards_for_non_forbidden_tiles() {
        let mut state = empty_state();
        state.players[0].push_hand(0);
        state.players[0].push_hand(4);
        state.players[0].push_hand(8);
        state.players[0].push_forbidden(4);

        let legals = state._get_legal_actions_internal(0);
        let discard_tiles: Vec<u8> = legals
            .iter()
            .filter(|action| action.action_type == ActionType::Discard)
            .filter_map(|action| action.tile)
            .collect();
        assert_eq!(discard_tiles, vec![0, 8]);
    }

    #[test]
    fn wait_response_offers_claims_plus_pass() {
        let mut state = empty_state();
        state.phase = Phase::WaitResponse;
        state.current_claim_counts[2] = 2;
        state.current_claims[2][0] = Action::new(ActionType::Pon, Some(12), &[13, 14], Some(2));
        state.current_claims[2][1] = Action::new(ActionType::Ron, Some(12), &[], Some(2));

        let legals = state._get_legal_actions_internal(2);
        assert_eq!(legals.len(), 3);
        assert!(legals
            .iter()
            .any(|action| action.action_type == ActionType::Pon));
        assert!(legals
            .iter()
            .any(|action| action.action_type == ActionType::Ron));
        assert!(legals
            .iter()
            .any(|action| action.action_type == ActionType::Pass));
    }

    #[test]
    fn done_state_has_no_legal_actions() {
        let mut state = empty_state();
        state.is_done = true;
        assert!(state._get_legal_actions_internal(0).is_empty());
    }

    #[test]
    fn get_legal_actions_into_matches_internal_builder() {
        let mut state = empty_state();
        state.players[0].push_hand(0);
        state.players[0].push_hand(4);

        let expected = state._get_legal_actions_internal(0);
        let mut buf = Vec::new();
        state._get_legal_actions_into(0, &mut buf);
        assert_actions_match(&expected, &buf);
    }

    #[test]
    fn wait_response_without_claims_still_offers_pass() {
        let mut state = empty_state();
        state.phase = Phase::WaitResponse;

        let legals = assert_legals_match(&state, 3);
        assert_eq!(legals.len(), 1);
        assert_eq!(legals[0].action_type, ActionType::Pass);
    }

    #[test]
    fn wait_act_winning_draw_offers_tsumo_and_riichi() {
        let mut state = empty_state();
        set_wall_remaining(&mut state, 40);
        add_hand_from_text(&mut state, 0, "123456m12223p123s");
        state.drawn_tile = Some(tile("6m"));

        let legals = assert_legals_match(&state, 0);
        assert!(legals
            .iter()
            .any(|action| action.action_type == ActionType::Tsumo));
        assert!(legals
            .iter()
            .any(|action| action.action_type == ActionType::Riichi));
    }

    #[test]
    fn wait_act_after_riichi_only_allows_drawn_discard_outside_declaration_turn() {
        let mut state = empty_state();
        state.drawn_tile = Some(8);
        state.players[0].push_hand(0);
        state.players[0].push_hand(4);
        state.players[0].push_hand(8);
        state.players[0].riichi_declared = true;
        state.players[0].riichi_declaration_index = Some(0);
        state.players[0].push_discard(40, true, true);
        state.players[0].push_discard(44, true, false);

        let legals = assert_legals_match(&state, 0);
        assert_eq!(legals.len(), 1);
        assert_eq!(legals[0].action_type, ActionType::Discard);
        assert_eq!(legals[0].tile, Some(8));
    }

    #[test]
    fn wait_act_offers_ankan_and_kakan_before_riichi() {
        let mut state = empty_state();
        set_wall_remaining(&mut state, 40);
        state.drawn_tile = Some(0);
        for tile in [0, 1, 2, 3, 8, 55] {
            state.players[0].push_hand(tile);
        }
        state.players[0].push_meld(Meld::new(MeldType::Pon, &[52, 53, 54], true, 1, Some(52)));

        let legals = assert_legals_match(&state, 0);
        assert!(has_action(&legals, ActionType::Ankan, |action| {
            action.tile == Some(0) && action.consume_slice() == [0, 1, 2, 3]
        }));
        assert!(has_action(&legals, ActionType::Kakan, |action| {
            action.tile == Some(55) && action.consume_slice() == [52, 53, 54]
        }));
    }

    #[test]
    fn wait_act_post_riichi_allows_ankan_when_waits_are_unchanged() {
        let mut state = empty_state();
        set_wall_remaining(&mut state, 40);
        add_hand_from_text(&mut state, 0, "1111234m234p55s66z");
        state.drawn_tile = Some(tile("1m"));
        state.players[0].riichi_declared = true;
        state.players[0].riichi_declaration_index = Some(0);
        state.players[0].push_discard(tile("9p"), true, true);

        let legals = assert_legals_match(&state, 0);
        assert!(has_action(&legals, ActionType::Ankan, |action| {
            action.tile == Some(0) && action.consume_slice() == [0, 1, 2, 3]
        }));
    }

    #[test]
    fn wait_act_does_not_offer_kyushu_kyuhai_after_any_call_or_in_riichi_stage() {
        let mut called_state = empty_state();
        called_state.is_first_turn = true;
        called_state.drawn_tile = Some(0);
        for tile in [0, 32, 36, 68, 72, 104, 108, 112, 116] {
            called_state.players[0].push_hand(tile);
        }
        called_state.players[1].push_meld(Meld::new(MeldType::Pon, &[4, 5, 6], true, 0, Some(4)));

        let called_legals = assert_legals_match(&called_state, 0);
        assert!(!called_legals
            .iter()
            .any(|action| action.action_type == ActionType::KyushuKyuhai));

        let mut riichi_stage_state = empty_state();
        riichi_stage_state.is_first_turn = true;
        riichi_stage_state.drawn_tile = Some(0);
        riichi_stage_state.players[0].riichi_stage = true;
        for tile in [0, 32, 36, 68, 72, 104, 108, 112, 116] {
            riichi_stage_state.players[0].push_hand(tile);
        }

        let riichi_stage_legals = assert_legals_match(&riichi_stage_state, 0);
        assert!(!riichi_stage_legals
            .iter()
            .any(|action| action.action_type == ActionType::KyushuKyuhai));
    }

    #[test]
    fn claim_actions_offer_ron_for_non_furiten_winning_tile() {
        let mut state = empty_state();
        set_wall_remaining(&mut state, 40);
        add_hand_from_text(&mut state, 1, "12345m12223p123s");

        let (legals, missed_agari) = assert_claim_actions_match(&mut state, 1, 0, tile("6m"));
        assert!(!missed_agari);
        assert!(legals
            .iter()
            .any(|action| action.action_type == ActionType::Ron));
    }

    #[test]
    fn claim_actions_mark_missed_agari_for_yakuless_win_shape() {
        let mut state = empty_state();
        set_wall_remaining(&mut state, 40);
        for tile in [0, 1, 2, 60, 64, 72, 73, 74, 76, 80, 96, 100, 104] {
            state.players[1].push_hand(tile);
        }

        let (legals, missed_agari) = assert_claim_actions_match(&mut state, 1, 0, 56);
        assert!(!legals
            .iter()
            .any(|action| action.action_type == ActionType::Ron));
        assert!(missed_agari);
    }

    #[test]
    fn claim_actions_block_ron_when_furiten_flag_or_discard_applies() {
        let mut discarded_state = empty_state();
        set_wall_remaining(&mut discarded_state, 40);
        add_hand_from_text(&mut discarded_state, 1, "12345m12223p123s");
        discarded_state.players[1].push_discard(tile("6m"), true, false);

        let (discarded_legals, discarded_missed) =
            assert_claim_actions_match(&mut discarded_state, 1, 0, tile("6m"));
        assert!(!discarded_legals
            .iter()
            .any(|action| action.action_type == ActionType::Ron));
        assert!(!discarded_missed);

        let mut missed_state = empty_state();
        set_wall_remaining(&mut missed_state, 40);
        add_hand_from_text(&mut missed_state, 1, "12345m12223p123s");
        missed_state.players[1].riichi_declared = true;
        missed_state.players[1].missed_agari_riichi = true;

        let (missed_legals, missed_agari) =
            assert_claim_actions_match(&mut missed_state, 1, 0, tile("6m"));
        assert!(!missed_legals
            .iter()
            .any(|action| action.action_type == ActionType::Ron));
        assert!(!missed_agari);
    }

    #[test]
    fn claim_actions_block_pon_when_kuikae_forbids_every_remaining_discard() {
        let mut state = empty_state();
        set_wall_remaining(&mut state, 40);
        for tile in [16, 17, 18] {
            state.players[1].push_hand(tile);
        }

        let (legals, missed_agari) = assert_claim_actions_match(&mut state, 1, 0, 19);
        assert!(!missed_agari);
        assert!(!legals
            .iter()
            .any(|action| action.action_type == ActionType::Pon));
        assert!(legals
            .iter()
            .any(|action| action.action_type == ActionType::Daiminkan));
    }

    #[test]
    fn claim_actions_generate_distinct_red_five_pon_variants_and_daiminkan() {
        let mut state = empty_state();
        set_wall_remaining(&mut state, 40);
        for tile in [16, 17, 18, 68] {
            state.players[1].push_hand(tile);
        }

        let (legals, missed_agari) = assert_claim_actions_match(&mut state, 1, 0, 19);
        let pon_actions = legals
            .iter()
            .filter(|action| action.action_type == ActionType::Pon)
            .collect::<Vec<_>>();

        assert!(!missed_agari);
        assert_eq!(pon_actions.len(), 3);
        assert!(legals
            .iter()
            .any(|action| action.action_type == ActionType::Daiminkan));
    }

    #[test]
    fn claim_actions_generate_all_three_chi_patterns_for_shimocha() {
        let mut state = empty_state();
        set_wall_remaining(&mut state, 40);
        for tile in [8, 12, 13, 20, 21, 24, 28, 68] {
            state.players[1].push_hand(tile);
        }

        let (legals, missed_agari) = assert_claim_actions_match(&mut state, 1, 0, 19);
        assert!(!missed_agari);
        assert!(has_action(&legals, ActionType::Chi, |action| {
            action
                .consume_slice()
                .iter()
                .map(|t| t / 4)
                .collect::<Vec<_>>()
                == vec![2, 3]
        }));
        assert!(has_action(&legals, ActionType::Chi, |action| {
            action
                .consume_slice()
                .iter()
                .map(|t| t / 4)
                .collect::<Vec<_>>()
                == vec![3, 5]
        }));
        assert!(has_action(&legals, ActionType::Chi, |action| {
            action
                .consume_slice()
                .iter()
                .map(|t| t / 4)
                .collect::<Vec<_>>()
                == vec![5, 6]
        }));
    }
}
