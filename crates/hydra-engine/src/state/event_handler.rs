use super::sorted_insert_arr;
use crate::action::Phase;
use crate::parser::mjai_to_tid;
use crate::replay::{Action as LogAction, MjaiEvent};
use crate::state::GameState;
use crate::types::{Meld, MeldType, Wind};

fn parse_mjai_tile(s: &str) -> u8 {
    mjai_to_tid(s).unwrap_or(0)
}

fn mjai_tile_has_explicit_copy(s: &str) -> bool {
    matches!(s, "5mr" | "5pr" | "5sr")
}

fn remove_replay_hand_tile_by_mjai(player: &mut super::player::PlayerState, tile: u8, mjai: &str) {
    let idx = if mjai_tile_has_explicit_copy(mjai) {
        player.hand_slice().iter().position(|&t| t == tile)
    } else {
        let tile_type = tile / 4;
        player.hand_slice().iter().position(|&t| t / 4 == tile_type)
    };

    if let Some(idx) = idx {
        player.remove_hand(idx);
    }
}

fn alloc_start_kyoku_tile(tile_counts: &mut [u8; 34], tile_str: &str) -> u8 {
    let tile = parse_mjai_tile(tile_str);
    let tile_type = (tile / 4) as usize;

    if mjai_tile_has_explicit_copy(tile_str) {
        tile_counts[tile_type] = tile_counts[tile_type].max(1);
        return tile;
    }

    let mut copy = tile_counts[tile_type];
    if matches!(tile_type, 4 | 13 | 22) {
        copy = copy.max(1);
    }
    tile_counts[tile_type] = copy.saturating_add(1);
    tile_type as u8 * 4 + copy
}

pub trait GameStateEventHandler {
    fn apply_mjai_event(&mut self, event: MjaiEvent);
    fn apply_log_action(&mut self, action: &LogAction);
}

impl GameStateEventHandler for GameState {
    fn apply_mjai_event(&mut self, event: MjaiEvent) {
        match event {
            MjaiEvent::StartKyoku {
                bakaze,
                honba,
                kyoutaku,
                scores,
                dora_marker,
                tehais,
                oya,
                ..
            } => {
                // Initialize round state from event
                self.honba = honba;
                self.riichi_sticks = kyoutaku as u32;
                self.kyoku_idx = oya;
                self.players.iter_mut().enumerate().for_each(|(i, p)| {
                    p.reset_round();
                    p.score = scores[i];
                });
                self.round_wind = match bakaze.as_str() {
                    "E" => Wind::East as u8,
                    "S" => Wind::South as u8,
                    "W" => Wind::West as u8,
                    "N" => Wind::North as u8,
                    _ => Wind::East as u8,
                };
                self.oya = oya;
                self.wall
                    .set_dora_indicators_single(parse_mjai_tile(&dora_marker));
                self.wall.tile_count = 136 - (13 * 4);
                self.wall.rinshan_draw_count = 0;
                self.wall.pending_kan_dora_count = 0;
                self.wall.draw_cursor = 0;
                self.clear_claims();
                self.clear_active_players();
                self.pending_kan = None;
                self.pending_oya_won = false;
                self.pending_is_draw = false;
                self.needs_initialize_next_round = false;
                self.turn_count = 0;
                self.riichi_pending_acceptance = None;
                self.is_rinshan_flag = false;
                self.is_first_turn = true;
                self.is_after_kan = false;
                self.last_discard = None;
                self.last_error = None;
                self.win_results = Default::default();
                self.last_win_results = Default::default();
                self.riichi_sutehais = [None; 4];
                self.last_tedashis = [None; 4];

                // Set hands
                for (i, hand_strs) in tehais.iter().enumerate() {
                    let mut tile_counts = [0u8; 34];
                    for tile_str in hand_strs {
                        self.players[i]
                            .push_hand(alloc_start_kyoku_tile(&mut tile_counts, tile_str));
                    }
                    self.players[i].hand_slice_mut().sort();
                }

                self.drawn_tile = None;
                self.current_player = self.oya; // Oya starts
                self.phase = Phase::WaitAct;
                self.set_single_active_player(self.current_player);
                self.needs_tsumo = true;
                self.is_done = false;
            }
            MjaiEvent::Tsumo { actor, pai } => {
                let tile = parse_mjai_tile(&pai);
                self.current_player = actor as u8;
                self.drawn_tile = Some(tile);
                sorted_insert_arr(
                    &mut self.players[actor].hand,
                    &mut self.players[actor].hand_len,
                    tile,
                );
                if self.wall.tile_count > 0 {
                    self.wall.draw_back();
                }
                self.needs_tsumo = false;
            }
            MjaiEvent::Dahai { actor, pai, .. } => {
                let tile = parse_mjai_tile(&pai);
                self.current_player = actor as u8;
                remove_replay_hand_tile_by_mjai(&mut self.players[actor], tile, &pai);
                self.players[actor].push_discard(tile, false, false);
                self.last_discard = Some((actor as u8, tile));
                self.drawn_tile = None;

                if self.players[actor].riichi_stage {
                    self.players[actor].riichi_declared = true;
                }
                self.needs_tsumo = true;
            }
            MjaiEvent::Pon {
                actor,
                pai,
                consumed,
                ..
            } => {
                let tile = parse_mjai_tile(&pai);
                self.current_player = actor as u8;
                let c1 = parse_mjai_tile(&consumed[0]);
                let c2 = parse_mjai_tile(&consumed[1]);
                let form_tiles = vec![tile, c1, c2];

                for t in &[c1, c2] {
                    let mjai = if *t == c1 { &consumed[0] } else { &consumed[1] };
                    remove_replay_hand_tile_by_mjai(&mut self.players[actor], *t, mjai);
                }

                self.players[actor].push_meld(Meld::new(
                    MeldType::Pon,
                    &form_tiles,
                    true,
                    -1,
                    Some(tile),
                ));
                self.drawn_tile = None;
                self.needs_tsumo = false;
            }
            MjaiEvent::Chi {
                actor,
                pai,
                consumed,
                ..
            } => {
                let tile = parse_mjai_tile(&pai);
                self.current_player = actor as u8;
                let c1 = parse_mjai_tile(&consumed[0]);
                let c2 = parse_mjai_tile(&consumed[1]);
                let form_tiles = vec![tile, c1, c2];

                for t in &[c1, c2] {
                    let mjai = if *t == c1 { &consumed[0] } else { &consumed[1] };
                    remove_replay_hand_tile_by_mjai(&mut self.players[actor], *t, mjai);
                }

                self.players[actor].push_meld(Meld::new(
                    MeldType::Chi,
                    &form_tiles,
                    true,
                    -1,
                    Some(tile),
                ));
                self.drawn_tile = None;
                self.needs_tsumo = false;
            }
            MjaiEvent::Kan {
                actor,
                pai,
                consumed,
                ..
            } => {
                let tile = parse_mjai_tile(&pai);
                self.current_player = actor as u8;
                let mut tiles = vec![tile];
                for c in &consumed {
                    tiles.push(parse_mjai_tile(c));
                }

                for c in &consumed {
                    let tv = parse_mjai_tile(c);
                    remove_replay_hand_tile_by_mjai(&mut self.players[actor], tv, c);
                }

                self.players[actor].push_meld(Meld::new(
                    MeldType::Daiminkan,
                    &tiles,
                    true,
                    -1,
                    Some(tile),
                ));
                self.needs_tsumo = true;
            }
            MjaiEvent::Ankan { actor, consumed } => {
                let mut tiles = Vec::new();
                for c in &consumed {
                    let t = parse_mjai_tile(c);
                    tiles.push(t);
                    remove_replay_hand_tile_by_mjai(&mut self.players[actor], t, c);
                }
                self.players[actor].push_meld(Meld::new(MeldType::Ankan, &tiles, false, -1, None));
                self.needs_tsumo = true;
            }
            MjaiEvent::Kakan { actor, pai } => {
                let tile = parse_mjai_tile(&pai);
                remove_replay_hand_tile_by_mjai(&mut self.players[actor], tile, &pai);
                for m in self.players[actor].melds_slice_mut().iter_mut() {
                    if m.meld_type == MeldType::Pon && m.tiles[0] / 4 == tile / 4 {
                        m.meld_type = MeldType::Kakan;
                        m.push_tile(tile);
                        m.tiles_slice_mut().sort();
                        break;
                    }
                }
                self.needs_tsumo = true;
            }
            MjaiEvent::Reach { actor } => {
                self.players[actor].riichi_stage = true;
            }
            MjaiEvent::ReachAccepted { actor } => {
                self.players[actor].riichi_declared = true;
                self.riichi_sticks += 1;
                self.players[actor].score -= 1000;
            }
            MjaiEvent::Dora { dora_marker } => {
                let tile = parse_mjai_tile(&dora_marker);
                self.wall.push_dora_indicator(tile);
            }
            MjaiEvent::Kita { .. } => {
                // Kita is 3P only; ignored in 4P event handler
            }
            MjaiEvent::Hora { .. } | MjaiEvent::Ryukyoku { .. } | MjaiEvent::EndKyoku => {
                self.is_done = true;
            }
            _ => {}
        }
    }

    fn apply_log_action(&mut self, action: &LogAction) {
        match action {
            LogAction::DiscardTile {
                seat,
                tile,
                is_liqi,
                is_wliqi,
                ..
            } => {
                let s = *seat;
                let t = *tile;
                let is_tsumogiri = if let Some(dt) = self.drawn_tile {
                    dt == t
                } else {
                    false
                };

                if let Some(idx) = self.players[s].hand_slice().iter().position(|&x| x == t) {
                    self.players[s].remove_hand(idx);
                }
                self.players[s].push_discard(t, !is_tsumogiri, *is_liqi || *is_wliqi);
                self.last_discard = Some((s as u8, t));
                self.drawn_tile = None;
                // Track nagashi eligibility: discard must be terminal/honor
                self.players[s].nagashi_eligible &= crate::types::is_terminal_tile(t);

                if *is_liqi || *is_wliqi {
                    if !self.players[s].riichi_declared {
                        self.players[s].riichi_declared = true;
                        if *is_wliqi {
                            self.players[s].double_riichi_declared = true;
                        }
                        // Defer the 1000 deposit; it gets voided if this
                        // discard is ronned, otherwise finalized on the next
                        // DealTile / ChiPengGang.
                        self.riichi_pending_acceptance = Some(s as u8);
                    }
                    self.players[s].riichi_declaration_index =
                        Some(self.players[s].discard_len as usize - 1);
                }
                self.current_player = (s as u8 + 1) % 4;
                self.phase = Phase::WaitAct;
                self.set_single_active_player(self.current_player);
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = false;
            }
            LogAction::DealTile { seat, tile, .. } => {
                // Finalize pending riichi deposit (discard was not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }
                sorted_insert_arr(
                    &mut self.players[*seat].hand,
                    &mut self.players[*seat].hand_len,
                    *tile,
                );
                self.drawn_tile = Some(*tile);
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.set_single_active_player(self.current_player);
                self.is_rinshan_flag = self.is_after_kan && *seat == self.current_player as usize;
                self.needs_tsumo = false;
                self.is_after_kan = false;
                if self.wall.tile_count > 0 {
                    self.wall.draw_back();
                }
            }
            LogAction::ChiPengGang {
                seat,
                meld_type,
                tiles,
                froms,
            } => {
                // Finalize pending riichi deposit (discard was claimed, not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }
                // Discard was called -> discarder loses nagashi eligibility
                if let Some((discarder_pid, _)) = self.last_discard {
                    self.players[discarder_pid as usize].nagashi_eligible = false;
                }
                // Remove tiles from hand
                for (i, t) in tiles.iter().enumerate() {
                    if i < froms.len() && froms[i] == *seat {
                        if let Some(idx) = self.players[*seat]
                            .hand_slice()
                            .iter()
                            .position(|&x| x == *t)
                        {
                            self.players[*seat].remove_hand(idx);
                        }
                    }
                }

                let from_who = froms
                    .iter()
                    .find(|&&f| f != *seat)
                    .map(|&f| f as i8)
                    .unwrap_or(-1);
                let ct = tiles
                    .iter()
                    .zip(froms.iter())
                    .find(|(_, &f)| f != *seat)
                    .map(|(&t, _)| t);
                let discarder = from_who.max(0) as u8;
                self.players[*seat].push_meld(Meld::new(*meld_type, tiles, true, from_who, ct));

                // PAO detection: daisangen (3 dragon melds) or daisuushii (4 wind melds)
                if *meld_type == MeldType::Pon || *meld_type == MeldType::Daiminkan {
                    if let Some(&called) = ct.as_ref() {
                        let tile_val = called / 4;
                        if (31..=33).contains(&tile_val) {
                            let dragon_melds = self.players[*seat]
                                .melds
                                .iter()
                                .filter(|m| {
                                    let t = m.tiles[0] / 4;
                                    (31..=33).contains(&t) && m.meld_type != MeldType::Chi
                                })
                                .count();
                            if dragon_melds == 3 {
                                self.players[*seat].pao_insert(37, discarder);
                            }
                        } else if (27..=30).contains(&tile_val) {
                            let wind_melds = self.players[*seat]
                                .melds
                                .iter()
                                .filter(|m| {
                                    let t = m.tiles[0] / 4;
                                    (27..=30).contains(&t) && m.meld_type != MeldType::Chi
                                })
                                .count();
                            if wind_melds == 4 {
                                self.players[*seat].pao_insert(50, discarder);
                            }
                        }
                    }
                }

                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.set_single_active_player(self.current_player);
                let is_gang = *meld_type == MeldType::Daiminkan;
                self.needs_tsumo = is_gang;
                self.is_first_turn = false;
                self.is_after_kan = is_gang;
            }
            LogAction::AnGangAddGang {
                seat,
                meld_type,
                tiles,
                ..
            } => {
                if *meld_type == MeldType::Ankan {
                    let t_val = tiles[0] / 4;
                    for _ in 0..4 {
                        if let Some(idx) = self.players[*seat]
                            .hand
                            .iter()
                            .position(|&x| x / 4 == t_val)
                        {
                            self.players[*seat].remove_hand(idx);
                        }
                    }
                    let mut m_tiles = vec![t_val * 4, t_val * 4 + 1, t_val * 4 + 2, t_val * 4 + 3];
                    if t_val == 4 {
                        m_tiles = vec![16, 17, 18, 19];
                    } else if t_val == 13 {
                        m_tiles = vec![52, 53, 54, 55];
                    } else if t_val == 22 {
                        m_tiles = vec![88, 89, 90, 91];
                    }

                    self.players[*seat].push_meld(Meld::new(*meld_type, &m_tiles, false, -1, None));
                } else {
                    // Kakan
                    let tile = tiles[0];
                    if let Some(idx) = self.players[*seat]
                        .hand_slice()
                        .iter()
                        .position(|&x| x == tile)
                    {
                        self.players[*seat].remove_hand(idx);
                    }
                    for m in self.players[*seat].melds_slice_mut().iter_mut() {
                        if m.meld_type == MeldType::Pon && m.tiles[0] / 4 == tile / 4 {
                            m.meld_type = MeldType::Kakan;
                            m.push_tile(tile);
                            m.tiles_slice_mut().sort();
                            break;
                        }
                    }
                    // Set last_discard so chankan ron targets the kakan player
                    self.last_discard = Some((*seat as u8, tile));
                }
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.set_single_active_player(self.current_player);
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = true;
                // Also record ankan for kokushi chankan (kokushi can ron on closed kan)
                if *meld_type == MeldType::Ankan {
                    self.last_discard = Some((*seat as u8, tiles[0]));
                }
            }
            LogAction::Dora { dora_marker } => {
                self.wall.push_dora_indicator(*dora_marker);
            }
            LogAction::Hule { hules } => {
                // If a riichi deposit is pending and this is a ron, the deposit
                // is voided (MjSoul does not deduct it when the discard is ronned).
                let first_is_ron = hules.first().is_some_and(|h| !h.zimo);
                if first_is_ron {
                    self.riichi_pending_acceptance = None;
                }

                let honba = self.honba;
                let riichi_on_table = self.riichi_sticks;
                let mut honba_taken = false;

                for h in hules {
                    let winner = h.seat;
                    let is_tsumo = h.zimo;

                    if is_tsumo {
                        let is_oya = (winner as u8) == self.oya;

                        // Check PAO (sekinin barai): for yakuman tsumo, the PAO
                        // player pays the full amount for all non-winning
                        // players. We detect PAO from the player's pao map
                        // which was populated when dragon/wind melds were claimed.
                        let mut pao_payer = None;
                        let mut pao_yakuman_val: i32 = 0;
                        let mut total_yakuman_val: i32 = 0;

                        if h.yiman {
                            // Daisangen = yaku 37, Daisuushii = yaku 50
                            // Double yakuman IDs: 47, 48, 49, 50
                            for &yid in &h.fans {
                                let val: i32 = if [47, 48, 49, 50].contains(&yid) {
                                    2
                                } else {
                                    1
                                };
                                total_yakuman_val += val;
                                if let Some(liable) = self.players[winner].pao_get(yid as u8) {
                                    pao_yakuman_val += val;
                                    pao_payer = Some(liable);
                                }
                            }
                        }

                        if pao_yakuman_val > 0 {
                            // PAO: liable player pays the PAO portion entirely
                            let unit: i32 = if is_oya { 48000 } else { 32000 };
                            let pao_amt = pao_yakuman_val * unit;
                            let non_pao_yakuman_val = total_yakuman_val - pao_yakuman_val;
                            let non_pao_amt = non_pao_yakuman_val * unit;

                            if let Some(pp) = pao_payer {
                                self.players[pp as usize].score -= pao_amt;
                                self.players[winner].score += pao_amt;
                            }

                            // Non-PAO part split normally
                            if non_pao_amt > 0 {
                                if is_oya {
                                    let share = non_pao_amt / 3;
                                    for i in 0..4 {
                                        if i != winner {
                                            self.players[i].score -= share;
                                            self.players[winner].score += share;
                                        }
                                    }
                                } else {
                                    for i in 0..4 {
                                        if i != winner {
                                            let share = if (i as u8) == self.oya {
                                                non_pao_amt / 2
                                            } else {
                                                non_pao_amt / 4
                                            };
                                            self.players[i].score -= share;
                                            self.players[winner].score += share;
                                        }
                                    }
                                }
                            }

                            // Add honba bonus (paid by PAO player)
                            if let Some(pp) = pao_payer {
                                let honba_total = honba as i32 * 300;
                                self.players[pp as usize].score -= honba_total;
                                self.players[winner].score += honba_total;
                            }
                        } else {
                            // Standard tsumo distribution
                            for i in 0..4 {
                                if i != winner {
                                    let base_pay = if is_oya {
                                        h.point_zimo_xian
                                    } else if (i as u8) == self.oya {
                                        h.point_zimo_qin
                                    } else {
                                        h.point_zimo_xian
                                    };
                                    let pay = base_pay as i32 + honba as i32 * 100;
                                    self.players[i].score -= pay;
                                    self.players[winner].score += pay;
                                }
                            }
                        }
                    } else if let Some((discarder, _)) = self.last_discard {
                        // Only the first ron winner gets the honba bonus
                        let ron_honba = if !honba_taken {
                            honba_taken = true;
                            honba
                        } else {
                            0
                        };

                        // Check PAO for ron yakuman: target pays half,
                        // PAO player pays the other half.
                        let mut pao_payer_ron: Option<u8> = None;
                        if h.yiman {
                            for &yid in &h.fans {
                                if let Some(liable) = self.players[winner].pao_get(yid as u8) {
                                    pao_payer_ron = Some(liable);
                                    break;
                                }
                            }
                        }

                        if let Some(pp) = pao_payer_ron {
                            let half = h.point_rong as i32 / 2;
                            let honba_pts = ron_honba as i32 * 300;
                            self.players[pp as usize].score -= half + honba_pts;
                            self.players[discarder as usize].score -= half;
                            self.players[winner].score += h.point_rong as i32 + honba_pts;
                        } else {
                            let pay = h.point_rong as i32 + ron_honba as i32 * 300;
                            self.players[discarder as usize].score -= pay;
                            self.players[winner].score += pay;
                        }
                    }
                }

                // Distribute riichi sticks to first winner
                if !hules.is_empty() {
                    let winner = hules[0].seat;
                    self.players[winner].score += riichi_on_table as i32 * 1000;
                    self.riichi_sticks = 0;
                }

                self.is_done = true;
            }
            LogAction::NoTile => {
                // Finalize pending riichi deposit (exhaustive draw, not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }

                // Check for nagashi mangan first
                let np = 4usize;
                let mut nagashi_winners = Vec::new();
                for (i, p) in self.players.iter().enumerate() {
                    if p.nagashi_eligible {
                        nagashi_winners.push(i as u8);
                    }
                }

                if !nagashi_winners.is_empty() {
                    // Nagashi mangan: apply mangan tsumo payment (no honba)
                    for &w in &nagashi_winners {
                        let is_oya = w == self.oya;
                        let score_res =
                            crate::score::calculate_score(5, 30, is_oya, true, 0, np as u8);
                        if is_oya {
                            for i in 0..np {
                                if i as u8 != w {
                                    self.players[i].score -= score_res.pay_tsumo_ko as i32;
                                    self.players[w as usize].score += score_res.pay_tsumo_ko as i32;
                                }
                            }
                        } else {
                            for i in 0..np {
                                if i as u8 != w {
                                    let pay = if i as u8 == self.oya {
                                        score_res.pay_tsumo_oya as i32
                                    } else {
                                        score_res.pay_tsumo_ko as i32
                                    };
                                    self.players[i].score -= pay;
                                    self.players[w as usize].score += pay;
                                }
                            }
                        }
                    }
                } else {
                    // Compute tenpai/noten payments
                    let mut tenpai = [false; 4];
                    for (i, p) in self.players.iter().enumerate() {
                        if i < 4 {
                            let calc = crate::hand_evaluator::HandEvaluator::new(&p.hand, &p.melds);
                            tenpai[i] = calc.is_tenpai();
                        }
                    }
                    let num_tp = tenpai.iter().filter(|&&t| t).count();
                    if num_tp > 0 && num_tp < 4 {
                        let pk = 3000 / num_tp as i32;
                        let pn = 3000 / (4 - num_tp) as i32;
                        for (i, tp) in tenpai.iter().enumerate() {
                            let delta = if *tp { pk } else { -pn };
                            self.players[i].score += delta;
                        }
                    }
                }
                self.is_done = true;
            }
            LogAction::LiuJu { .. } => {
                // Finalize pending riichi deposit
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }
                // Abortive draw - no score changes
                self.is_done = true;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{Action, ActionType};
    use crate::rule::GameRule;
    use std::collections::HashSet;

    fn start_kyoku_event() -> MjaiEvent {
        MjaiEvent::StartKyoku {
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 2,
            kyoutaku: 1,
            oya: 2,
            scores: vec![25_000, 24_000, 26_000, 25_000],
            dora_marker: "4m".to_string(),
            tehais: vec![
                vec![
                    "1m".to_string(),
                    "2m".to_string(),
                    "3m".to_string(),
                    "4m".to_string(),
                    "5m".to_string(),
                    "6m".to_string(),
                    "7m".to_string(),
                    "8m".to_string(),
                    "9m".to_string(),
                    "1p".to_string(),
                    "2p".to_string(),
                    "3p".to_string(),
                    "4p".to_string(),
                ],
                vec![
                    "1s".to_string(),
                    "2s".to_string(),
                    "3s".to_string(),
                    "4s".to_string(),
                    "5s".to_string(),
                    "6s".to_string(),
                    "7s".to_string(),
                    "8s".to_string(),
                    "9s".to_string(),
                    "E".to_string(),
                    "S".to_string(),
                    "W".to_string(),
                    "N".to_string(),
                ],
                vec![
                    "P".to_string(),
                    "F".to_string(),
                    "C".to_string(),
                    "1m".to_string(),
                    "1m".to_string(),
                    "2m".to_string(),
                    "2m".to_string(),
                    "3m".to_string(),
                    "3m".to_string(),
                    "4m".to_string(),
                    "4m".to_string(),
                    "5m".to_string(),
                    "5m".to_string(),
                ],
                vec![
                    "6p".to_string(),
                    "6p".to_string(),
                    "7p".to_string(),
                    "7p".to_string(),
                    "8p".to_string(),
                    "8p".to_string(),
                    "9p".to_string(),
                    "9p".to_string(),
                    "1s".to_string(),
                    "1s".to_string(),
                    "2s".to_string(),
                    "2s".to_string(),
                    "3s".to_string(),
                ],
            ],
        }
    }

    #[test]
    fn start_kyoku_replay_resets_round_scoped_state() {
        let rule = GameRule::default_tenhou();
        let mut state = GameState::new(0, true, Some(7), 0, rule);

        state.wall.tile_count = 3;
        state.wall.draw_cursor = 5;
        state.wall.rinshan_draw_count = 2;
        state.wall.pending_kan_dora_count = 1;
        state.current_player = 1;
        state.phase = Phase::WaitResponse;
        state.active_players = [0, 1, 2, 3];
        state.active_player_count = 4;
        state.pending_kan = Some((1, Action::new(ActionType::Kakan, Some(42), &[], Some(1))));
        state.pending_oya_won = true;
        state.pending_is_draw = true;
        state.needs_initialize_next_round = true;
        state.turn_count = 99;
        state.riichi_pending_acceptance = Some(3);
        state.is_rinshan_flag = true;
        state.is_first_turn = false;
        state.is_after_kan = true;
        state.last_discard = Some((1, 16));
        state.riichi_sutehais = [Some(1), Some(2), Some(3), Some(4)];
        state.last_tedashis = [Some(5), Some(6), Some(7), Some(8)];
        state.players[0].riichi_declared = true;
        state.players[1].riichi_stage = true;
        state.players[2].push_discard(0, true, false);
        state.players[3].push_meld(Meld::new(MeldType::Pon, &[0, 1, 2], true, 0, Some(2)));

        state.apply_mjai_event(start_kyoku_event());

        assert_eq!(state.honba, 2);
        assert_eq!(state.riichi_sticks, 1);
        assert_eq!(state.oya, 2);
        assert_eq!(state.current_player, 2);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_count, 1);
        assert_eq!(state.active_players[0], 2);
        assert_eq!(state.wall.tile_count, 84);
        assert_eq!(state.wall.draw_cursor, 0);
        assert_eq!(state.wall.rinshan_draw_count, 0);
        assert_eq!(state.wall.pending_kan_dora_count, 0);
        assert_eq!(state.pending_kan, None);
        assert!(!state.pending_oya_won);
        assert!(!state.pending_is_draw);
        assert!(!state.needs_initialize_next_round);
        assert_eq!(state.turn_count, 0);
        assert_eq!(state.riichi_pending_acceptance, None);
        assert!(!state.is_rinshan_flag);
        assert!(state.is_first_turn);
        assert!(!state.is_after_kan);
        assert_eq!(state.last_discard, None);
        assert_eq!(state.riichi_sutehais, [None; 4]);
        assert_eq!(state.last_tedashis, [None; 4]);
        assert!(!state.players[0].riichi_declared);
        assert!(!state.players[1].riichi_stage);
        assert_eq!(state.players[2].discard_len, 0);
        assert_eq!(state.players[3].meld_count, 0);
        assert_eq!(state.players[0].score, 25_000);
        assert_eq!(state.players[1].score, 24_000);
        assert_eq!(state.players[2].score, 26_000);
        assert_eq!(state.players[3].score, 25_000);
    }

    #[test]
    fn replay_kakan_removes_matching_tile_class_from_hand() {
        let rule = GameRule::default_tenhou();
        let mut state = GameState::new(0, true, Some(7), 0, rule);
        state.apply_mjai_event(MjaiEvent::StartKyoku {
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 0,
            kyoutaku: 0,
            oya: 0,
            scores: vec![25_000, 25_000, 25_000, 25_000],
            dora_marker: "1m".to_string(),
            tehais: vec![
                vec![
                    "1m".to_string(),
                    "2m".to_string(),
                    "3m".to_string(),
                    "4m".to_string(),
                    "5m".to_string(),
                    "6m".to_string(),
                    "7m".to_string(),
                    "8m".to_string(),
                    "9m".to_string(),
                    "1p".to_string(),
                    "2p".to_string(),
                    "3p".to_string(),
                    "4p".to_string(),
                    "4p".to_string(),
                ],
                vec![
                    "1s".to_string(),
                    "2s".to_string(),
                    "3s".to_string(),
                    "4s".to_string(),
                    "5s".to_string(),
                    "6s".to_string(),
                    "7s".to_string(),
                    "8s".to_string(),
                    "9s".to_string(),
                    "E".to_string(),
                    "S".to_string(),
                    "W".to_string(),
                    "N".to_string(),
                ],
                vec![
                    "P".to_string(),
                    "F".to_string(),
                    "C".to_string(),
                    "1m".to_string(),
                    "1m".to_string(),
                    "2m".to_string(),
                    "2m".to_string(),
                    "3m".to_string(),
                    "3m".to_string(),
                    "4m".to_string(),
                    "4m".to_string(),
                    "5m".to_string(),
                    "5m".to_string(),
                ],
                vec![
                    "6p".to_string(),
                    "6p".to_string(),
                    "7p".to_string(),
                    "7p".to_string(),
                    "8p".to_string(),
                    "8p".to_string(),
                    "9p".to_string(),
                    "9p".to_string(),
                    "1s".to_string(),
                    "1s".to_string(),
                    "2s".to_string(),
                    "2s".to_string(),
                    "3s".to_string(),
                ],
            ],
        });

        state.apply_mjai_event(MjaiEvent::Pon {
            actor: 0,
            target: 0,
            pai: "4p".to_string(),
            consumed: vec!["4p".to_string(), "4p".to_string()],
        });
        state.apply_mjai_event(MjaiEvent::Tsumo {
            actor: 0,
            pai: "4p".to_string(),
        });

        let four_p_count_before = state.players[0]
            .hand_slice()
            .iter()
            .filter(|&&tile| tile / 4 == 12)
            .count();
        assert_eq!(four_p_count_before, 1);

        state.apply_mjai_event(MjaiEvent::Kakan {
            actor: 0,
            pai: "4p".to_string(),
        });

        let four_p_count_after = state.players[0]
            .hand_slice()
            .iter()
            .filter(|&&tile| tile / 4 == 12)
            .count();
        assert_eq!(four_p_count_after, 0);
        assert!(state.players[0]
            .melds_slice()
            .iter()
            .any(|meld| meld.meld_type == MeldType::Kakan && meld.tiles[0] / 4 == 12));
    }

    #[test]
    fn replay_start_kyoku_assigns_unique_tile_ids_for_duplicate_plain_tiles() {
        let rule = GameRule::default_tenhou();
        let mut state = GameState::new(0, true, Some(7), 0, rule);

        state.apply_mjai_event(MjaiEvent::StartKyoku {
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 0,
            kyoutaku: 0,
            oya: 0,
            scores: vec![25_000, 25_000, 25_000, 25_000],
            dora_marker: "1m".to_string(),
            tehais: vec![
                vec![
                    "6m".to_string(),
                    "6m".to_string(),
                    "6m".to_string(),
                    "7m".to_string(),
                    "8m".to_string(),
                    "9m".to_string(),
                    "1p".to_string(),
                    "2p".to_string(),
                    "3p".to_string(),
                    "4p".to_string(),
                    "5p".to_string(),
                    "6p".to_string(),
                    "7p".to_string(),
                ],
                vec![
                    "1s".to_string(),
                    "2s".to_string(),
                    "3s".to_string(),
                    "4s".to_string(),
                    "5s".to_string(),
                    "6s".to_string(),
                    "7s".to_string(),
                    "8s".to_string(),
                    "9s".to_string(),
                    "E".to_string(),
                    "S".to_string(),
                    "W".to_string(),
                    "N".to_string(),
                ],
                vec![
                    "P".to_string(),
                    "F".to_string(),
                    "C".to_string(),
                    "1m".to_string(),
                    "1m".to_string(),
                    "2m".to_string(),
                    "2m".to_string(),
                    "3m".to_string(),
                    "3m".to_string(),
                    "4m".to_string(),
                    "4m".to_string(),
                    "5m".to_string(),
                    "5m".to_string(),
                ],
                vec![
                    "6p".to_string(),
                    "6p".to_string(),
                    "7p".to_string(),
                    "7p".to_string(),
                    "8p".to_string(),
                    "8p".to_string(),
                    "9p".to_string(),
                    "9p".to_string(),
                    "1s".to_string(),
                    "1s".to_string(),
                    "2s".to_string(),
                    "2s".to_string(),
                    "3s".to_string(),
                ],
            ],
        });

        let six_m_tiles: Vec<u8> = state.players[0]
            .hand_slice()
            .iter()
            .copied()
            .filter(|tile| tile / 4 == 5)
            .collect();
        assert_eq!(six_m_tiles.len(), 3);
        assert_eq!(six_m_tiles.iter().copied().collect::<HashSet<_>>().len(), 3);
    }
}
