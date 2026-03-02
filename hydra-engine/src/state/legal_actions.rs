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
                    let res =
                        calc.calc(tile, self.wall.dora_indicator_slice(), &[], Some(cond));
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
            let no_calls = self.players.iter().all(|p| p.meld_count == 0 );

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
                    let res =
                        calc.calc(tile, self.wall.dora_indicator_slice(), &[], Some(cond));
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
                            let calc_post =
                                crate::hand_evaluator::HandEvaluator::new(&post[..post_len], &melds_post);
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
            let no_calls = self.players.iter().all(|p| p.meld_count == 0 );

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
        let in_discards = self.players[i_us].discards_slice().iter()
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
