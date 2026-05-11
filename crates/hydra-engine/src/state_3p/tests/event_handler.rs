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
