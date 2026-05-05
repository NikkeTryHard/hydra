use crate::action::Phase;
use crate::hand_evaluator_3p::HandEvaluator3P;
use crate::parser::mjai_to_tid;
use crate::replay::{Action as LogAction, MjaiEvent};
use crate::state_3p::GameState3P;
use crate::types::{Meld, MeldType, Wind};

fn parse_mjai_tile(s: &str) -> u8 {
    mjai_to_tid(s).unwrap_or(0)
}

fn mjai_tile_has_explicit_copy(s: &str) -> bool {
    matches!(s, "5mr" | "5pr" | "5sr")
}

fn remove_replay_hand_tile_by_mjai(
    player: &mut super::player::PlayerState3P,
    tile: u8,
    mjai: &str,
) {
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

pub trait GameState3PEventHandler {
    fn apply_mjai_event(&mut self, event: MjaiEvent);
    fn apply_log_action(&mut self, action: &LogAction);
}

impl GameState3PEventHandler for GameState3P {
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
                self.wall.tile_count = 108 - (13 * 3);
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
                self.win_results = [None; 3];
                self.last_win_results = [None; 3];
                self.riichi_sutehais = [None; 3];
                self.last_tedashis = [None; 3];

                for (i, hand_strs) in tehais.iter().enumerate() {
                    let mut tile_counts = [0u8; 34];
                    for tile_str in hand_strs {
                        self.players[i]
                            .push_hand(alloc_start_kyoku_tile(&mut tile_counts, tile_str));
                    }
                    self.players[i].hand_slice_mut().sort();
                }

                self.drawn_tile = None;
                self.current_player = self.oya;
                self.phase = Phase::WaitAct;
                self.set_single_active_player(self.current_player);
                self.needs_tsumo = true;
                let oya_idx = self.oya as usize;
                if self.players[oya_idx].hand_len == 14 {
                    self.drawn_tile = self.players[oya_idx].hand_slice().last().copied();
                    self.needs_tsumo = false;
                }
                self.is_done = false;
            }
            MjaiEvent::Tsumo { actor, pai } => {
                let tile = parse_mjai_tile(&pai);
                self.current_player = actor as u8;
                self.drawn_tile = Some(tile);
                self.players[actor].push_hand(tile);
                self.players[actor].hand_slice_mut().sort();
                if self.wall.tile_count > 0 {
                    self.wall.draw_back();
                }
                self.phase = Phase::WaitAct;
                self.set_single_active_player(actor as u8);
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
                // Chi shouldn't happen in 3P, but handle gracefully
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
            MjaiEvent::ReachAccepted { actor }
                if self.riichi_pending_acceptance == Some(actor as u8)
                    || !self.players[actor].riichi_declared =>
            {
                self.players[actor].riichi_declared = true;
                self.riichi_sticks += 1;
                self.players[actor].score -= 1000;
                self.riichi_pending_acceptance = None;
            }
            MjaiEvent::Dora { dora_marker } => {
                let tile = parse_mjai_tile(&dora_marker);
                self.wall.push_dora_indicator(tile);
            }
            MjaiEvent::Kita { actor } => {
                let north_id = 30;
                if let Some(idx) = self.players[actor]
                    .hand
                    .iter()
                    .position(|&t| t / 4 == north_id)
                {
                    let tile = self.players[actor].remove_hand(idx);
                    self.players[actor].push_kita(tile);
                }
                self.needs_tsumo = true;
            }
            MjaiEvent::Hora { .. } | MjaiEvent::Ryukyoku { .. } | MjaiEvent::EndKyoku => {
                self.is_done = true;
            }
            _ => {}
        }
    }

    fn apply_log_action(&mut self, action: &LogAction) {
        let np: u8 = 3;
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
                self.players[s].hand_slice_mut().sort();
                self.players[s].push_discard(t, !is_tsumogiri, *is_liqi || *is_wliqi);
                self.last_discard = Some((s as u8, t));
                self.drawn_tile = None;

                self.players[s].riichi_declared =
                    self.players[s].riichi_declared || *is_liqi || *is_wliqi;
                if *is_wliqi {
                    self.players[s].double_riichi_declared = true;
                }
                if *is_liqi || *is_wliqi {
                    self.players[s].riichi_declaration_index =
                        Some(self.players[s].discard_len as usize - 1);
                    self.riichi_pending_acceptance = Some(s as u8);
                }
                // Track nagashi eligibility: discard must be terminal/honor
                self.players[s].nagashi_eligible &= crate::types::is_terminal_tile(t);
                self.current_player = (s as u8 + 1) % np;
                self.phase = Phase::WaitAct;
                self.set_single_active_player(self.current_player);
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = false;
            }
            LogAction::DealTile { seat, tile, .. } => {
                // Accept pending riichi deposit (discard was not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }
                self.players[*seat].push_hand(*tile);
                self.drawn_tile = Some(*tile);
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.set_single_active_player(self.current_player);
                self.is_rinshan_flag = self.is_after_kan && *seat == self.current_player as usize;
                self.needs_tsumo = false;
                self.is_after_kan = false;
                self.players[*seat].hand_slice_mut().sort();
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
                // Accept pending riichi deposit (discard was not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }
                // Discard was called → discarder loses nagashi eligibility
                if let Some((discarder_pid, _)) = self.last_discard {
                    self.players[discarder_pid as usize].nagashi_eligible = false;
                }
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
                self.players[*seat].hand_slice_mut().sort();

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
                }
                self.players[*seat].hand_slice_mut().sort();
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.set_single_active_player(self.current_player);
                self.needs_tsumo = true;
                self.is_first_turn = false;
                self.is_after_kan = true;
                // Record as last_discard so chankan ron (Hule) can identify
                // the kan declarer as the payer (e.g. kokushi chankan on ankan).
                self.last_discard = Some((*seat as u8, tiles[0]));
            }
            LogAction::Dora { dora_marker } => {
                self.wall.push_dora_indicator(*dora_marker);
            }
            LogAction::BaBei { seat, .. } => {
                // Accept pending riichi deposit (discard was not ronned)
                if let Some(rp) = self.riichi_pending_acceptance.take() {
                    self.players[rp as usize].score -= 1000;
                    self.riichi_sticks += 1;
                }
                // Remove a North tile from hand and add to kita_tiles
                let north_34: u8 = 30; // 4z = North
                if let Some(idx) = self.players[*seat]
                    .hand
                    .iter()
                    .position(|&x| x / 4 == north_34)
                {
                    let tile = self.players[*seat].remove_hand(idx);
                    self.players[*seat].push_kita(tile);
                    // Record as last_discard so ron-on-kita (Hule) can identify
                    // the kita declarer as the payer.
                    self.last_discard = Some((*seat as u8, tile));
                }
                self.players[*seat].hand_slice_mut().sort();
                self.current_player = *seat as u8;
                self.phase = Phase::WaitAct;
                self.set_single_active_player(self.current_player);
                self.needs_tsumo = true;
                self.is_first_turn = false;
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

                        // Check PAO (sekinin barai) for yakuman tsumo
                        let mut pao_payer = None;
                        let mut pao_yakuman_val: i32 = 0;
                        let mut total_yakuman_val: i32 = 0;

                        if h.yiman {
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
                            // PAO tsumo: pao payer pays the full tsumo total
                            // for the liable portion.
                            let tsumo_total: i32 = if is_oya {
                                // Oya: each ko pays xian, total = xian * (np-1)
                                h.point_zimo_xian as i32 * (np as i32 - 1)
                            } else {
                                // Ko: oya pays qin, each other ko pays xian
                                h.point_zimo_qin as i32 + h.point_zimo_xian as i32 * (np as i32 - 2)
                            };
                            let pao_amt = if total_yakuman_val > 0 {
                                tsumo_total * pao_yakuman_val / total_yakuman_val
                            } else {
                                tsumo_total
                            };
                            let non_pao_amt = tsumo_total - pao_amt;

                            if let Some(pp) = pao_payer {
                                self.players[pp as usize].score -= pao_amt;
                                self.players[winner].score += pao_amt;
                            }

                            // Non-PAO part split normally
                            if non_pao_amt > 0 {
                                for i in 0..np as usize {
                                    if i != winner {
                                        let share = if is_oya {
                                            non_pao_amt / (np as i32 - 1)
                                        } else if (i as u8) == self.oya {
                                            // Approximate: qin share
                                            h.point_zimo_qin as i32 * non_pao_amt / tsumo_total
                                        } else {
                                            h.point_zimo_xian as i32 * non_pao_amt / tsumo_total
                                        };
                                        self.players[i].score -= share;
                                        self.players[winner].score += share;
                                    }
                                }
                            }

                            // Honba paid by pao payer
                            if let Some(pp) = pao_payer {
                                let honba_total = honba as i32 * (np as i32 - 1) * 100;
                                self.players[pp as usize].score -= honba_total;
                                self.players[winner].score += honba_total;
                            }
                        } else {
                            // Standard tsumo distribution (no pao)
                            for i in 0..np as usize {
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
                        // Ron
                        let ron_honba = if !honba_taken {
                            honba_taken = true;
                            honba
                        } else {
                            0
                        };

                        // Check PAO for ron
                        let mut pao_payer = None;
                        let mut pao_yakuman_val: i32 = 0;
                        let mut total_yakuman_val: i32 = 0;

                        if h.yiman {
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
                            let pp = pao_payer.unwrap_or(discarder);
                            let ron_total = h.point_rong as i32;
                            let pao_amt = ron_total * pao_yakuman_val / total_yakuman_val;
                            let honba_pts = ron_honba as i32 * (np as i32 - 1) * 100;

                            // PAO ron: split between pao payer and discarder
                            let pao_share = pao_amt / 2 + honba_pts;
                            let discarder_share = ron_total - pao_amt / 2;

                            self.players[pp as usize].score -= pao_share;
                            self.players[discarder as usize].score -= discarder_share;
                            self.players[winner].score += pao_share + discarder_share;
                        } else {
                            // Standard ron
                            let pay =
                                h.point_rong as i32 + ron_honba as i32 * (np as i32 - 1) * 100;
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
                        let score_res = crate::score::calculate_score(5, 30, is_oya, true, 0, np);
                        if is_oya {
                            for i in 0..np as usize {
                                if i as u8 != w {
                                    self.players[i].score -= score_res.pay_tsumo_ko as i32;
                                    self.players[w as usize].score += score_res.pay_tsumo_ko as i32;
                                }
                            }
                        } else {
                            for i in 0..np as usize {
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
                    // Regular tenpai/noten payments (pool = 2000 in 3P)
                    let mut tenpai = [false; 3];
                    for (i, p) in self.players.iter().enumerate() {
                        if i < 3 {
                            let calc = HandEvaluator3P::new(&p.hand, &p.melds);
                            tenpai[i] = calc.is_tenpai();
                        }
                    }
                    let num_tp = tenpai.iter().filter(|&&t| t).count();
                    if num_tp > 0 && num_tp < 3 {
                        let pk = 2000 / num_tp as i32;
                        let pn = 2000 / (3 - num_tp) as i32;
                        for (i, tp) in tenpai.iter().enumerate() {
                            let delta = if *tp { pk } else { -pn };
                            self.players[i].score += delta;
                        }
                    }
                }
                self.is_done = true;
            }
            LogAction::LiuJu { .. } => {
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
    use crate::replay::HuleData;
    use crate::rule::GameRule;

    fn start_kyoku_event() -> MjaiEvent {
        MjaiEvent::StartKyoku {
            bakaze: "E".to_string(),
            kyoku: 1,
            honba: 1,
            kyoutaku: 2,
            oya: 1,
            scores: vec![35_000, 30_000, 35_000],
            dora_marker: "4p".to_string(),
            tehais: vec![
                vec![
                    "1p".to_string(),
                    "2p".to_string(),
                    "3p".to_string(),
                    "4p".to_string(),
                    "5p".to_string(),
                    "6p".to_string(),
                    "7p".to_string(),
                    "8p".to_string(),
                    "9p".to_string(),
                    "E".to_string(),
                    "S".to_string(),
                    "W".to_string(),
                    "N".to_string(),
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
                    "P".to_string(),
                    "F".to_string(),
                    "C".to_string(),
                    "1m".to_string(),
                ],
                vec![
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
                    "5p".to_string(),
                ],
            ],
        }
    }

    fn new_test_state() -> GameState3P {
        let rule = GameRule::default_tenhou();
        GameState3P::new(4, true, Some(9), 0, rule)
    }

    fn set_backing_hand_tiles(state: &mut GameState3P, seat: usize, tiles: &[u8]) {
        state.players[seat].reset_round();
        for &tile in tiles {
            state.players[seat].push_hand(tile);
        }
        state.players[seat].hand_slice_mut().sort();
    }

    #[test]
    fn start_kyoku_replay_resets_round_scoped_state() {
        let rule = GameRule::default_tenhou();
        let mut state = GameState3P::new(4, true, Some(9), 0, rule);

        state.wall.tile_count = 2;
        state.wall.draw_cursor = 4;
        state.wall.rinshan_draw_count = 3;
        state.wall.pending_kan_dora_count = 1;
        state.current_player = 2;
        state.phase = Phase::WaitResponse;
        state.active_players = [0, 1, 2, 0];
        state.active_player_count = 3;
        state.pending_kan = Some((1, Action::new(ActionType::Kakan, Some(18), &[], Some(1))));
        state.pending_oya_won = true;
        state.pending_is_draw = true;
        state.needs_initialize_next_round = true;
        state.turn_count = 33;
        state.riichi_pending_acceptance = Some(2);
        state.is_rinshan_flag = true;
        state.is_first_turn = false;
        state.is_after_kan = true;
        state.last_discard = Some((2, 52));
        state.riichi_sutehais = [Some(1), Some(2), Some(3)];
        state.last_tedashis = [Some(4), Some(5), Some(6)];
        state.players[0].riichi_declared = true;
        state.players[1].riichi_stage = true;
        state.players[1].push_discard(0, true, false);
        state.players[2].push_meld(Meld::new(MeldType::Pon, &[0, 1, 2], true, 0, Some(2)));

        state.apply_mjai_event(start_kyoku_event());

        assert_eq!(state.honba, 1);
        assert_eq!(state.riichi_sticks, 2);
        assert_eq!(state.oya, 1);
        assert_eq!(state.current_player, 1);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_count, 1);
        assert_eq!(state.active_players[0], 1);
        assert_eq!(state.wall.tile_count, 69);
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
        assert_eq!(state.riichi_sutehais, [None; 3]);
        assert_eq!(state.last_tedashis, [None; 3]);
        assert!(!state.players[0].riichi_declared);
        assert!(!state.players[1].riichi_stage);
        assert_eq!(state.players[1].discard_len, 0);
        assert_eq!(state.players[2].meld_count, 0);
        assert_eq!(state.players[0].score, 35_000);
        assert_eq!(state.players[1].score, 30_000);
        assert_eq!(state.players[2].score, 35_000);
    }

    #[test]
    fn reach_accepted_deposits_points_and_marks_player_riichi() {
        let rule = GameRule::default_tenhou();
        let mut state = GameState3P::new(4, true, Some(9), 0, rule);
        state.players[1].score = 30_000;
        state.riichi_sticks = 2;

        state.apply_mjai_event(MjaiEvent::ReachAccepted { actor: 1 });

        assert!(state.players[1].riichi_declared);
        assert_eq!(state.players[1].score, 29_000);
        assert_eq!(state.riichi_sticks, 3);
    }

    #[test]
    fn reach_accepted_is_idempotent_after_replay_pass_resolution() {
        let rule = GameRule::default_tenhou();
        let mut state = GameState3P::new(4, true, Some(9), 0, rule);
        state.players[0].reset_round();
        state.players[1].score = 25_000;
        state.players[1].riichi_declared = true;
        state.riichi_pending_acceptance = Some(1);

        state.apply_log_action(&LogAction::DealTile {
            seat: 0,
            tile: 4,
            doras: None,
            left_tile_count: None,
        });
        assert!(state.players[1].riichi_declared);
        assert_eq!(state.players[1].score, 24_000);
        assert_eq!(state.riichi_sticks, 1);
        assert_eq!(state.phase, Phase::WaitAct);
        assert!(state.riichi_pending_acceptance.is_none());

        state.apply_mjai_event(MjaiEvent::ReachAccepted { actor: 1 });
        assert!(state.players[1].riichi_declared);
        assert_eq!(state.players[1].score, 24_000);
        assert_eq!(state.riichi_sticks, 1);
        assert!(state.riichi_pending_acceptance.is_none());
    }

    #[test]
    fn dora_event_appends_indicator() {
        let rule = GameRule::default_tenhou();
        let mut state = GameState3P::new(4, true, Some(9), 0, rule);
        state.wall.set_dora_indicators_single(40);

        state.apply_mjai_event(MjaiEvent::Dora {
            dora_marker: "5s".to_string(),
        });

        assert_eq!(state.wall.dora_indicator_slice(), &[40, 89]);
    }

    #[test]
    fn kita_event_moves_north_tile_to_kita_and_requests_draw() {
        let mut state = new_test_state();
        state.players[0].reset_round();
        state.players[0].push_hand(120);
        state.players[0].push_hand(0);
        state.players[0].hand_slice_mut().sort();
        state.needs_tsumo = false;

        state.apply_mjai_event(MjaiEvent::Kita { actor: 0 });

        assert_eq!(state.players[0].kita_slice(), &[120]);
        assert_eq!(state.players[0].hand_slice(), &[0]);
        assert!(state.needs_tsumo);
    }

    #[test]
    fn discard_tile_tracks_riichi_metadata_and_terminal_only_nagashi() {
        let mut state = new_test_state();
        state.players[0].reset_round();
        state.players[0].push_hand(0);
        state.players[0].push_hand(1);
        state.players[0].hand_slice_mut().sort();
        state.drawn_tile = Some(0);
        state.current_player = 2;
        state.phase = Phase::WaitResponse;
        state.is_first_turn = true;
        state.is_after_kan = true;

        state.apply_log_action(&LogAction::DiscardTile {
            seat: 0,
            tile: 0,
            is_liqi: false,
            is_wliqi: true,
            doras: None,
        });

        assert_eq!(state.players[0].hand_slice(), &[1]);
        assert_eq!(state.players[0].discards_slice(), &[0]);
        assert!(!state.players[0].discard_from_hand[0]);
        assert!(state.players[0].discard_is_riichi[0]);
        assert!(state.players[0].riichi_declared);
        assert!(state.players[0].double_riichi_declared);
        assert_eq!(state.players[0].riichi_declaration_index, Some(0));
        assert!(state.players[0].nagashi_eligible);
        assert_eq!(state.riichi_pending_acceptance, Some(0));
        assert_eq!(state.last_discard, Some((0, 0)));
        assert_eq!(state.drawn_tile, None);
        assert_eq!(state.current_player, 1);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_count, 1);
        assert_eq!(state.active_players[0], 1);
        assert!(state.needs_tsumo);
        assert!(!state.is_first_turn);
        assert!(!state.is_after_kan);

        state.players[1].reset_round();
        state.players[1].push_hand(4);
        state.players[1].push_hand(8);
        state.players[1].hand_slice_mut().sort();
        state.drawn_tile = Some(4);

        state.apply_log_action(&LogAction::DiscardTile {
            seat: 1,
            tile: 4,
            is_liqi: false,
            is_wliqi: false,
            doras: None,
        });

        assert_eq!(state.players[1].hand_slice(), &[8]);
        assert!(!state.players[1].nagashi_eligible);
        assert_eq!(state.players[1].discards_slice(), &[4]);
        assert!(!state.players[1].discard_from_hand[0]);
        assert!(!state.players[1].discard_is_riichi[0]);
    }

    #[test]
    fn deal_tile_accepts_pending_riichi_and_sets_rinshan_flag_after_kan() {
        let mut state = new_test_state();
        state.players[2].score = 25_000;
        state.players[1].reset_round();
        state.players[1].push_hand(9);
        state.riichi_pending_acceptance = Some(2);
        state.riichi_sticks = 1;
        state.is_after_kan = true;
        state.needs_tsumo = true;
        state.wall.tile_count = 5;
        state.current_player = 0;

        state.apply_log_action(&LogAction::DealTile {
            seat: 1,
            tile: 7,
            doras: None,
            left_tile_count: None,
        });

        assert_eq!(state.players[2].score, 24_000);
        assert_eq!(state.riichi_sticks, 2);
        assert_eq!(state.riichi_pending_acceptance, None);
        assert_eq!(state.players[1].hand_slice(), &[7, 9]);
        assert_eq!(state.drawn_tile, Some(7));
        assert_eq!(state.current_player, 1);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_count, 1);
        assert_eq!(state.active_players[0], 1);
        assert!(state.is_rinshan_flag);
        assert!(!state.needs_tsumo);
        assert!(!state.is_after_kan);
        assert_eq!(state.wall.tile_count, 4);
    }

    #[test]
    fn ankan_log_action_builds_full_meld_and_sets_chankan_tracking_flags() {
        let mut state = new_test_state();
        state.players[0].reset_round();
        state.players[0].push_hand(16);
        state.players[0].push_hand(17);
        state.players[0].push_hand(18);
        state.players[0].push_hand(19);
        state.players[0].push_hand(0);
        state.players[0].hand_slice_mut().sort();
        state.phase = Phase::WaitResponse;
        state.active_player_count = 3;
        state.is_first_turn = true;
        state.is_after_kan = false;

        state.apply_log_action(&LogAction::AnGangAddGang {
            seat: 0,
            meld_type: MeldType::Ankan,
            tiles: vec![16, 17, 18, 19],
            tile_raw_id: 16,
            doras: None,
        });

        assert_eq!(state.players[0].hand_slice(), &[0]);
        assert_eq!(state.players[0].meld_count, 1);
        let meld = &state.players[0].melds_slice()[0];
        assert_eq!(meld.meld_type, MeldType::Ankan);
        assert_eq!(meld.tiles_slice(), &[16, 17, 18, 19]);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_count, 1);
        assert_eq!(state.active_players[0], 0);
        assert!(state.needs_tsumo);
        assert!(!state.is_first_turn);
        assert!(state.is_after_kan);
        assert_eq!(state.last_discard, Some((0, 16)));
    }

    #[test]
    fn ba_bei_accepts_pending_riichi_and_moves_north_to_kita() {
        let mut state = new_test_state();
        state.players[2].score = 25_000;
        state.players[0].reset_round();
        state.players[0].push_hand(120);
        state.players[0].push_hand(0);
        state.players[0].hand_slice_mut().sort();
        state.riichi_pending_acceptance = Some(2);
        state.riichi_sticks = 0;
        state.phase = Phase::WaitResponse;
        state.active_player_count = 3;
        state.is_first_turn = true;

        state.apply_log_action(&LogAction::BaBei {
            seat: 0,
            moqie: false,
        });

        assert_eq!(state.players[2].score, 24_000);
        assert_eq!(state.riichi_sticks, 1);
        assert_eq!(state.riichi_pending_acceptance, None);
        assert_eq!(state.players[0].hand_slice(), &[0]);
        assert_eq!(state.players[0].kita_slice(), &[120]);
        assert_eq!(state.last_discard, Some((0, 120)));
        assert_eq!(state.current_player, 0);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_count, 1);
        assert_eq!(state.active_players[0], 0);
        assert!(state.needs_tsumo);
        assert!(!state.is_first_turn);
    }

    #[test]
    fn hule_ron_voids_pending_riichi_and_awards_first_winner_riichi_sticks() {
        let mut state = new_test_state();
        state.players[0].score = 25_000;
        state.players[1].score = 25_000;
        state.players[2].score = 25_000;
        state.honba = 2;
        state.riichi_sticks = 3;
        state.riichi_pending_acceptance = Some(2);
        state.last_discard = Some((1, 33));

        state.apply_log_action(&LogAction::Hule {
            hules: vec![HuleData {
                seat: 0,
                hu_tile: 33,
                zimo: false,
                count: 0,
                fu: 30,
                fans: vec![1],
                li_doras: None,
                yiman: false,
                point_rong: 3_900,
                point_zimo_qin: 0,
                point_zimo_xian: 0,
            }],
        });

        assert_eq!(state.players[0].score, 32_300);
        assert_eq!(state.players[1].score, 20_700);
        assert_eq!(state.players[2].score, 25_000);
        assert_eq!(state.riichi_pending_acceptance, None);
        assert_eq!(state.riichi_sticks, 0);
        assert!(state.is_done);
    }

    #[test]
    fn no_tile_pays_nagashi_mangan_and_finalizes_pending_riichi() {
        let mut state = new_test_state();
        state.oya = 0;
        state.players[0].score = 25_000;
        state.players[1].score = 25_000;
        state.players[2].score = 25_000;
        state.players[0].nagashi_eligible = true;
        state.players[1].nagashi_eligible = false;
        state.players[2].nagashi_eligible = false;
        state.players[2].score = 26_000;
        state.riichi_pending_acceptance = Some(2);
        state.riichi_sticks = 1;

        state.apply_log_action(&LogAction::NoTile);

        assert_eq!(state.players[0].score, 33_000);
        assert_eq!(state.players[1].score, 21_000);
        assert_eq!(state.players[2].score, 21_000);
        assert_eq!(state.riichi_pending_acceptance, None);
        assert_eq!(state.riichi_sticks, 2);
        assert!(state.is_done);
    }

    #[test]
    fn replay_mjai_claim_and_round_end_events_cover_direct_branches() {
        let mut state = new_test_state();

        state.apply_mjai_event(MjaiEvent::StartKyoku {
            bakaze: "W".to_string(),
            kyoku: 3,
            honba: 0,
            kyoutaku: 0,
            oya: 0,
            scores: vec![25_000, 25_000, 25_000],
            dora_marker: "5mr".to_string(),
            tehais: vec![
                vec![
                    "4p".to_string(),
                    "4p".to_string(),
                    "5p".to_string(),
                    "5p".to_string(),
                    "1m".to_string(),
                    "2m".to_string(),
                    "3m".to_string(),
                    "7m".to_string(),
                    "7m".to_string(),
                    "7m".to_string(),
                    "E".to_string(),
                    "E".to_string(),
                    "E".to_string(),
                ],
                vec![
                    "2m".to_string(),
                    "3m".to_string(),
                    "4m".to_string(),
                    "5m".to_string(),
                    "6m".to_string(),
                    "7m".to_string(),
                    "8m".to_string(),
                    "1p".to_string(),
                    "2p".to_string(),
                    "3p".to_string(),
                    "4p".to_string(),
                    "6p".to_string(),
                    "6p".to_string(),
                ],
                vec![
                    "9s".to_string(),
                    "9s".to_string(),
                    "9s".to_string(),
                    "1p".to_string(),
                    "2p".to_string(),
                    "3p".to_string(),
                    "4p".to_string(),
                    "5p".to_string(),
                    "6p".to_string(),
                    "7p".to_string(),
                    "8p".to_string(),
                    "9p".to_string(),
                    "S".to_string(),
                ],
            ],
        });

        assert_eq!(state.round_wind, Wind::West as u8);
        assert_eq!(state.wall.dora_indicator_slice(), &[16]);

        state.apply_mjai_event(MjaiEvent::Tsumo {
            actor: 0,
            pai: "6p".to_string(),
        });
        assert_eq!(state.current_player, 0);
        assert_eq!(state.drawn_tile, Some(56));
        assert_eq!(state.players[0].hand_len, 14);
        assert_eq!(state.wall.tile_count, 68);
        assert!(!state.needs_tsumo);

        state.apply_mjai_event(MjaiEvent::Reach { actor: 0 });
        assert!(state.players[0].riichi_stage);

        state.apply_mjai_event(MjaiEvent::Dahai {
            actor: 0,
            pai: "6p".to_string(),
            tsumogiri: true,
        });
        assert!(state.players[0].riichi_declared);
        assert_eq!(state.players[0].discards_slice(), &[56]);
        assert_eq!(state.last_discard, Some((0, 56)));
        assert_eq!(state.drawn_tile, None);
        assert!(state.needs_tsumo);

        state.apply_mjai_event(MjaiEvent::Pon {
            actor: 0,
            target: 1,
            pai: "4p".to_string(),
            consumed: vec!["4p".to_string(), "4p".to_string()],
        });
        assert_eq!(state.players[0].meld_count, 1);
        let pon = &state.players[0].melds_slice()[0];
        assert_eq!(pon.meld_type, MeldType::Pon);
        assert_eq!(pon.tiles_slice().len(), 3);
        assert!(pon.tiles_slice().iter().all(|&tile| tile / 4 == 12));
        assert!(!state.needs_tsumo);

        state.apply_mjai_event(MjaiEvent::Chi {
            actor: 1,
            target: 0,
            pai: "1m".to_string(),
            consumed: vec!["2m".to_string(), "3m".to_string()],
        });
        assert_eq!(state.players[1].meld_count, 1);
        assert_eq!(state.players[1].melds_slice()[0].meld_type, MeldType::Chi);
        assert_eq!(state.players[1].melds_slice()[0].tiles_slice(), &[0, 4, 8]);
        assert!(!state.needs_tsumo);

        state.apply_mjai_event(MjaiEvent::Kan {
            actor: 2,
            target: 1,
            pai: "9s".to_string(),
            consumed: vec!["9s".to_string(), "9s".to_string(), "9s".to_string()],
        });
        assert_eq!(state.players[2].meld_count, 1);
        assert_eq!(
            state.players[2].melds_slice()[0].meld_type,
            MeldType::Daiminkan
        );
        assert_eq!(state.players[2].melds_slice()[0].tiles_slice().len(), 4);
        assert!(state.players[2].melds_slice()[0]
            .tiles_slice()
            .iter()
            .all(|&tile| tile / 4 == 26));
        assert!(state.needs_tsumo);

        state.apply_mjai_event(MjaiEvent::Ankan {
            actor: 0,
            consumed: vec![
                "E".to_string(),
                "E".to_string(),
                "E".to_string(),
                "E".to_string(),
            ],
        });
        assert_eq!(state.players[0].meld_count, 2);
        assert_eq!(state.players[0].melds_slice()[1].meld_type, MeldType::Ankan);
        assert!(!state.players[0].melds_slice()[1].opened);
        assert!(state.needs_tsumo);

        state.players[1].push_meld(Meld::new(MeldType::Pon, &[56, 57, 58], true, 0, Some(56)));
        state.players[1].push_hand(59);
        state.players[1].hand_slice_mut().sort();

        state.apply_mjai_event(MjaiEvent::Kakan {
            actor: 1,
            pai: "6p".to_string(),
        });
        let kakan = &state.players[1].melds_slice()[1];
        assert_eq!(kakan.meld_type, MeldType::Kakan);
        assert_eq!(kakan.tiles_slice(), &[56, 56, 57, 58]);
        assert!(state.needs_tsumo);

        state.is_done = false;
        state.apply_mjai_event(MjaiEvent::Hora {
            actor: 1,
            target: 0,
            pai: Some("6p".to_string()),
            uradora_markers: None,
            yaku: None,
            fu: None,
            han: None,
            scores: None,
            delta: None,
        });
        assert!(state.is_done);

        state.is_done = false;
        state.apply_mjai_event(MjaiEvent::Ryukyoku {
            reason: Some("yao9".to_string()),
            tehais: None,
            delta: None,
            scores: None,
        });
        assert!(state.is_done);

        state.is_done = false;
        state.apply_mjai_event(MjaiEvent::EndKyoku);
        assert!(state.is_done);
    }

    #[test]
    fn chi_peng_gang_applies_pending_riichi_nagashi_and_pao_branches() {
        let mut state = new_test_state();
        state.players[0].score = 25_000;
        state.players[0].nagashi_eligible = true;
        state.players[1].reset_round();
        for tile in [124, 125, 126, 0] {
            state.players[1].push_hand(tile);
        }
        state.players[1].hand_slice_mut().sort();
        state.players[1].push_meld(Meld::new(
            MeldType::Pon,
            &[128, 129, 130],
            true,
            0,
            Some(128),
        ));
        state.players[1].push_meld(Meld::new(
            MeldType::Pon,
            &[132, 133, 134],
            true,
            0,
            Some(132),
        ));
        state.players[1].score = 25_000;
        state.riichi_pending_acceptance = Some(0);
        state.riichi_sticks = 1;
        state.last_discard = Some((0, 124));
        state.phase = Phase::WaitResponse;
        state.active_player_count = 3;

        state.apply_log_action(&LogAction::ChiPengGang {
            seat: 1,
            meld_type: MeldType::Daiminkan,
            tiles: vec![124, 125, 126, 127],
            froms: vec![1, 1, 1, 0],
        });

        assert_eq!(state.players[0].score, 24_000);
        assert_eq!(state.riichi_sticks, 2);
        assert_eq!(state.riichi_pending_acceptance, None);
        assert!(!state.players[0].nagashi_eligible);
        assert_eq!(state.players[1].hand_slice(), &[0]);
        assert_eq!(state.players[1].meld_count, 3);
        let meld = &state.players[1].melds_slice()[2];
        assert_eq!(meld.meld_type, MeldType::Daiminkan);
        assert_eq!(meld.from_who, 0);
        assert_eq!(meld.called_tile, Some(127));
        assert_eq!(state.players[1].pao_get(37), Some(0));
        assert_eq!(state.current_player, 1);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_count, 1);
        assert_eq!(state.active_players[0], 1);
        assert!(state.needs_tsumo);
        assert!(!state.is_first_turn);
        assert!(state.is_after_kan);
    }

    #[test]
    fn log_kakan_upgrades_existing_pon_and_keeps_kan_flags() {
        let mut state = new_test_state();
        state.players[2].reset_round();
        state.players[2].push_hand(7);
        state.players[2].push_hand(40);
        state.players[2].hand_slice_mut().sort();
        state.players[2].push_meld(Meld::new(MeldType::Pon, &[4, 5, 6], true, 0, Some(4)));
        state.phase = Phase::WaitResponse;
        state.active_player_count = 3;

        state.apply_log_action(&LogAction::AnGangAddGang {
            seat: 2,
            meld_type: MeldType::Kakan,
            tiles: vec![7],
            tile_raw_id: 7,
            doras: None,
        });

        assert_eq!(state.players[2].hand_slice(), &[40]);
        assert_eq!(state.players[2].meld_count, 1);
        let meld = &state.players[2].melds_slice()[0];
        assert_eq!(meld.meld_type, MeldType::Kakan);
        assert_eq!(meld.tiles_slice(), &[4, 5, 6, 7]);
        assert_eq!(state.current_player, 2);
        assert_eq!(state.phase, Phase::WaitAct);
        assert_eq!(state.active_player_count, 1);
        assert_eq!(state.active_players[0], 2);
        assert!(state.needs_tsumo);
        assert!(!state.is_first_turn);
        assert!(state.is_after_kan);
        assert_eq!(state.last_discard, Some((2, 7)));
    }

    #[test]
    fn hule_standard_tsumo_distributes_honba_and_riichi_sticks_in_sanma() {
        let mut state = new_test_state();
        state.oya = 0;
        state.honba = 1;
        state.riichi_sticks = 2;
        state.players[0].score = 24_000;
        state.players[1].score = 25_000;
        state.players[2].score = 25_000;

        state.apply_log_action(&LogAction::Hule {
            hules: vec![HuleData {
                seat: 2,
                hu_tile: 33,
                zimo: true,
                count: 0,
                fu: 40,
                fans: vec![3],
                li_doras: None,
                yiman: false,
                point_rong: 0,
                point_zimo_qin: 4_000,
                point_zimo_xian: 2_000,
            }],
        });

        assert_eq!(state.players[0].score, 19_900);
        assert_eq!(state.players[1].score, 22_900);
        assert_eq!(state.players[2].score, 33_200);
        assert_eq!(state.riichi_sticks, 0);
        assert!(state.is_done);
    }

    #[test]
    fn hule_ron_pao_splits_payment_between_discarder_and_liable_player_in_sanma() {
        let mut state = new_test_state();
        state.honba = 1;
        state.last_discard = Some((1, 33));
        state.players[2].score = 25_000;
        state.players[1].score = 25_000;
        state.players[0].score = 25_000;
        state.players[2].pao_insert(37, 0);

        state.apply_log_action(&LogAction::Hule {
            hules: vec![HuleData {
                seat: 2,
                hu_tile: 33,
                zimo: false,
                count: 0,
                fu: 0,
                fans: vec![37],
                li_doras: None,
                yiman: true,
                point_rong: 32_000,
                point_zimo_qin: 0,
                point_zimo_xian: 0,
            }],
        });

        assert_eq!(state.players[0].score, 8_800);
        assert_eq!(state.players[1].score, 9_000);
        assert_eq!(state.players[2].score, 57_200);
        assert!(state.is_done);
    }

    #[test]
    fn no_tile_without_nagashi_or_detected_tenpai_leaves_scores_unchanged() {
        let mut state = new_test_state();
        set_backing_hand_tiles(&mut state, 1, &[]);
        set_backing_hand_tiles(&mut state, 2, &[]);
        for player in &mut state.players {
            player.score = 25_000;
            player.nagashi_eligible = false;
        }

        state.apply_log_action(&LogAction::NoTile);

        assert_eq!(state.players[0].score, 25_000);
        assert_eq!(state.players[1].score, 25_000);
        assert_eq!(state.players[2].score, 25_000);
        assert!(state.is_done);
    }

    #[test]
    fn liuju_marks_round_done_without_touching_pending_riichi_or_scores() {
        let mut state = new_test_state();
        state.players[0].score = 24_000;
        state.players[1].score = 25_000;
        state.players[2].score = 26_000;
        state.riichi_pending_acceptance = Some(1);
        state.riichi_sticks = 2;

        state.apply_log_action(&LogAction::LiuJu {
            lj_type: 1,
            seat: 1,
            tiles: vec![0, 4, 8],
        });

        assert_eq!(state.players[0].score, 24_000);
        assert_eq!(state.players[1].score, 25_000);
        assert_eq!(state.players[2].score, 26_000);
        assert_eq!(state.riichi_pending_acceptance, Some(1));
        assert_eq!(state.riichi_sticks, 2);
        assert!(state.is_done);
    }
}
