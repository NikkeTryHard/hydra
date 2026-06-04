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

fn start_kyoku_with_tehais(tehais: [Vec<&str>; 4]) -> MjaiEvent {
    MjaiEvent::StartKyoku {
        bakaze: "E".to_string(),
        kyoku: 1,
        honba: 0,
        kyoutaku: 0,
        oya: 0,
        scores: vec![25_000, 25_000, 25_000, 25_000],
        dora_marker: "1m".to_string(),
        tehais: tehais
            .into_iter()
            .map(|tiles| tiles.into_iter().map(|tile| tile.to_string()).collect())
            .collect(),
    }
}

fn set_backing_hand_tiles(state: &mut GameState, seat: usize, tiles: &[u8]) {
    state.players[seat].reset_round();
    state.players[seat].hand = [255; 14];
    for (idx, &tile) in tiles.iter().enumerate() {
        state.players[seat].hand[idx] = tile;
    }
    state.players[seat].hand_len = tiles.len() as u8;
}

fn make_hule(
    seat: usize,
    zimo: bool,
    yiman: bool,
    fans: Vec<u32>,
    point_rong: u32,
    point_zimo_qin: u32,
    point_zimo_xian: u32,
) -> crate::replay::HuleData {
    crate::replay::HuleData {
        seat,
        hu_tile: 0,
        zimo,
        count: 0,
        fu: 30,
        fans,
        li_doras: None,
        yiman,
        point_rong,
        point_zimo_qin,
        point_zimo_xian,
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
    assert_eq!(state.wall.remaining(), 84);
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
fn riichi_handler_ignores_second_reach_while_declaration_pending() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.skip_mjai_logging = false;
    state.phase = Phase::WaitAct;
    state.current_player = 0;
    state.active_players = [0; 4];
    state.active_player_count = 1;
    state.wall.tile_count = 40;
    state.players[0].score = 25_000;

    state._handle_riichi(0, Action::new(ActionType::Riichi, None, &[], Some(0)));
    let event_count = state.mjai_log.len();
    assert!(state.players[0].riichi_stage);

    state._handle_riichi(0, Action::new(ActionType::Riichi, None, &[], Some(0)));
    assert_eq!(state.mjai_log.len(), event_count);
    assert!(state.players[0].riichi_stage);
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
fn replay_ankan_matching_open_pon_applies_as_kakan() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.players[0].reset_round();
    state.players[0].push_hand(7);
    state.players[0].push_hand(40);
    state.players[0].hand_slice_mut().sort();
    state.players[0].push_meld(Meld::new(MeldType::Pon, &[4, 5, 6], true, 1, Some(4)));

    state.apply_mjai_event(MjaiEvent::Ankan {
        actor: 0,
        consumed: vec![
            "2m".to_string(),
            "2m".to_string(),
            "2m".to_string(),
            "2m".to_string(),
        ],
    });

    assert_eq!(state.players[0].hand_slice(), &[40]);
    assert_eq!(state.players[0].meld_count, 1);
    assert_eq!(state.players[0].melds_slice()[0].meld_type, MeldType::Kakan);
    assert_eq!(
        state.players[0].melds_slice()[0].tiles_slice(),
        &[4, 5, 6, 7]
    );
    assert_eq!(state.last_discard, Some((0, 4)));
    assert_eq!(state.current_player, 0);
    assert!(state.needs_tsumo);
    assert!(state.is_after_kan);
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

#[test]
fn replay_start_kyoku_allows_three_plain_fives_and_red_five() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);

    state.apply_mjai_event(start_kyoku_with_tehais([
        vec![
            "4m", "5m", "5m", "5m", "5mr", "9m", "4p", "7p", "8p", "5s", "6s", "7s", "P",
        ],
        vec![
            "1m", "3m", "6m", "5pr", "7p", "9p", "8s", "8s", "E", "E", "W", "N", "P",
        ],
        vec![
            "1m", "1m", "2m", "6m", "8m", "8m", "1p", "3p", "5p", "6p", "4s", "5sr", "S",
        ],
        vec![
            "2m", "2m", "4m", "1p", "2p", "3p", "5p", "6p", "9p", "2s", "4s", "6s", "C",
        ],
    ]));

    let five_m_tiles: Vec<u8> = state.players[0]
        .hand_slice()
        .iter()
        .copied()
        .filter(|tile| tile / 4 == 4)
        .collect();
    assert_eq!(five_m_tiles, vec![16, 17, 18, 19]);
}

#[test]
fn tsumo_and_dahai_update_turn_state_and_discards() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.apply_mjai_event(start_kyoku_event());

    let hand_len_before = state.players[2].hand_len;
    state.apply_mjai_event(MjaiEvent::Tsumo {
        actor: 2,
        pai: "6m".to_string(),
    });

    assert_eq!(state.current_player, 2);
    assert_eq!(state.drawn_tile, Some(parse_mjai_tile("6m").unwrap()));
    assert_eq!(state.players[2].hand_len, hand_len_before + 1);
    assert!(!state.needs_tsumo);

    state.players[2].riichi_stage = true;
    state.apply_mjai_event(MjaiEvent::Dahai {
        actor: 2,
        pai: "6m".to_string(),
        tsumogiri: false,
    });

    assert_eq!(state.drawn_tile, None);
    assert!(state.needs_tsumo);
    assert_eq!(
        state.last_discard,
        Some((2, parse_mjai_tile("6m").unwrap()))
    );
    assert_eq!(state.players[2].discard_len, 1);
    assert!(state.players[2].riichi_declared);
}

#[test]
fn replay_dahai_populates_response_claims_for_other_players() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.apply_mjai_event(start_kyoku_with_tehais([
        vec![
            "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
        ],
        vec![
            "5m", "5m", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S",
        ],
        vec![
            "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s",
        ],
        vec![
            "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N",
        ],
    ]));

    state.apply_mjai_event(MjaiEvent::Dahai {
        actor: 0,
        pai: "5m".to_string(),
        tsumogiri: false,
    });

    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.active_player_slice(), &[1]);
    let claims = state.claims_slice(1);
    assert!(claims
        .iter()
        .any(|action| action.action_type == ActionType::Pon));
    let mut legal = Vec::new();
    state.get_legal_actions_into(1, &mut legal);
    assert!(legal
        .iter()
        .any(|action| action.action_type == ActionType::Pon));
    assert!(legal
        .iter()
        .any(|action| action.action_type == ActionType::Pass));
}

#[test]
fn replay_tsumo_prefers_a_hand_unique_non_red_copy_for_plain_tiles() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.apply_mjai_event(start_kyoku_with_tehais([
        vec![
            "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
        ],
        vec![
            "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N",
        ],
        vec![
            "P", "F", "C", "1m", "1m", "2m", "2m", "3m", "3m", "4m", "4m", "5m", "5m",
        ],
        vec![
            "6p", "6p", "7p", "7p", "8p", "8p", "9p", "9p", "1s", "1s", "2s", "2s", "3s",
        ],
    ]));

    let existing_4p = parse_mjai_tile("4p").unwrap();
    assert!(state.players[0].hand_slice().contains(&existing_4p));

    state.apply_mjai_event(MjaiEvent::Tsumo {
        actor: 0,
        pai: "4p".to_string(),
    });

    assert_ne!(state.drawn_tile, Some(existing_4p));
    assert_eq!(state.drawn_tile.map(|t| t / 4), Some(existing_4p / 4));
    let four_p_copies = state.players[0]
        .hand_slice()
        .iter()
        .filter(|&&tile| tile / 4 == existing_4p / 4)
        .count();
    assert_eq!(four_p_copies, 2);
}

#[test]
fn replay_tsumo_clears_forbidden_discards_like_live_draw_path() {
    let mut rule = GameRule::default_tenhou();
    rule.kuikae_forbidden = true;
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.apply_mjai_event(start_kyoku_with_tehais([
        vec![
            "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
        ],
        vec![
            "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p",
        ],
        vec![
            "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N",
        ],
        vec![
            "1m", "1m", "1p", "1p", "1s", "1s", "E", "E", "S", "S", "W", "W", "N",
        ],
    ]));

    state.apply_mjai_event(MjaiEvent::Chi {
        actor: 1,
        target: 0,
        pai: "1m".to_string(),
        consumed: vec!["2m".to_string(), "3m".to_string()],
    });
    assert!(!state.players[1].forbidden_slice().is_empty());

    state.apply_mjai_event(MjaiEvent::Tsumo {
        actor: 1,
        pai: "6p".to_string(),
    });

    assert!(state.players[1].forbidden_slice().is_empty());
}

#[test]
fn replay_dahai_clears_riichi_stage_after_declared_discard() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.apply_mjai_event(start_kyoku_event());
    state.players[2].riichi_stage = true;

    state.apply_mjai_event(MjaiEvent::Dahai {
        actor: 2,
        pai: "5m".to_string(),
        tsumogiri: false,
    });

    assert!(state.players[2].riichi_declared);
    assert!(!state.players[2].riichi_stage);
}

#[test]
fn replay_dahai_without_claims_matches_live_wait_act_turn_transition() {
    let rule = GameRule::default_tenhou();
    let mut replay = GameState::new(0, true, Some(7), 0, rule);
    replay.apply_mjai_event(start_kyoku_with_tehais([
        vec![
            "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p",
        ],
        vec![
            "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N",
        ],
        vec![
            "P", "F", "C", "1m", "1m", "2m", "2m", "3m", "3m", "4m", "4m", "5m", "5m",
        ],
        vec![
            "6p", "6p", "7p", "7p", "8p", "8p", "9p", "9p", "1s", "1s", "2s", "2s", "3s",
        ],
    ]));
    replay.apply_mjai_event(MjaiEvent::Tsumo {
        actor: 0,
        pai: "5p".to_string(),
    });
    replay.apply_mjai_event(MjaiEvent::Dahai {
        actor: 0,
        pai: "4p".to_string(),
        tsumogiri: false,
    });

    assert_eq!(replay.phase, Phase::WaitAct);
    assert_eq!(replay.current_player, 1);
    assert_eq!(replay.active_player_slice(), &[1]);
    assert!(replay.drawn_tile.is_none());
    assert!(replay.needs_tsumo);
}

#[test]
fn replay_chi_sets_kuikae_forbidden_discards() {
    let mut rule = GameRule::default_tenhou();
    rule.kuikae_forbidden = true;
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.apply_mjai_event(start_kyoku_with_tehais([
        vec![
            "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s",
        ],
        vec![
            "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p",
        ],
        vec![
            "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N",
        ],
        vec![
            "1m", "1m", "1p", "1p", "1s", "1s", "E", "E", "S", "S", "W", "W", "N",
        ],
    ]));

    state.apply_mjai_event(MjaiEvent::Chi {
        actor: 1,
        target: 0,
        pai: "1m".to_string(),
        consumed: vec!["2m".to_string(), "3m".to_string()],
    });

    let forbidden = state.players[1].forbidden_slice();
    assert!(forbidden.contains(&parse_mjai_tile("1m").unwrap()));
    assert!(forbidden.contains(&parse_mjai_tile("4m").unwrap()));
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[1]);
}

#[test]
fn reach_accept_dora_and_terminal_events_flip_expected_flags() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.apply_mjai_event(start_kyoku_event());

    let score_before = state.players[1].score;
    let sticks_before = state.riichi_sticks;
    state.apply_mjai_event(MjaiEvent::Reach { actor: 1 });
    assert!(state.players[1].riichi_stage);

    state.apply_mjai_event(MjaiEvent::ReachAccepted { actor: 1 });
    assert!(state.players[1].riichi_declared);
    assert_eq!(state.players[1].score, score_before - 1000);
    assert_eq!(state.riichi_sticks, sticks_before + 1);

    let old_dora_count = state.wall.dora_indicator_count;
    state.apply_mjai_event(MjaiEvent::Dora {
        dora_marker: "5s".to_string(),
    });
    assert_eq!(state.wall.dora_indicator_count, old_dora_count + 1);

    state.apply_mjai_event(MjaiEvent::Kita { actor: 0 });
    assert_eq!(state.wall.dora_indicator_count, old_dora_count + 1);
    assert!(!state.is_done);

    state.apply_mjai_event(MjaiEvent::EndKyoku);
    assert!(state.is_done);
}

#[test]
fn reach_accepted_is_idempotent_after_replay_pass_resolution() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.players[1].score = 25_000;
    state.players[1].riichi_stage = true;
    state.riichi_pending_acceptance = Some(1);
    state.phase = Phase::WaitResponse;
    state.active_players = [2, 3, 0, 0];
    state.active_player_count = 2;

    state.resolve_replay_all_passes();
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
fn log_actions_track_riichi_claim_and_rinshan_bookkeeping() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);

    state.players[0].reset_round();
    state.players[0].push_hand(0);
    state.players[0].push_hand(4);
    state.players[0].hand_slice_mut().sort();
    state.players[1].reset_round();
    state.players[1].push_hand(1);
    state.players[1].push_hand(2);
    state.players[1].push_hand(3);
    state.players[1].push_hand(8);
    state.players[1].hand_slice_mut().sort();
    state.drawn_tile = Some(0);
    state.current_player = 3;
    state.phase = Phase::WaitResponse;
    state.is_first_turn = true;
    state.is_after_kan = true;
    state.wall.tile_count = 5;

    state.apply_log_action(&LogAction::DiscardTile {
        seat: 0,
        tile: 0,
        is_liqi: false,
        is_wliqi: true,
        doras: None,
    });

    assert_eq!(state.players[0].hand_slice(), &[4]);
    assert_eq!(state.players[0].discards_slice(), &[0]);
    assert!(!state.players[0].discard_from_hand[0]);
    assert!(state.players[0].discard_is_riichi[0]);
    assert!(state.players[0].riichi_declared);
    assert!(state.players[0].double_riichi_declared);
    assert_eq!(state.players[0].riichi_declaration_index, Some(0));
    assert!(state.players[0].nagashi_eligible);
    assert_eq!(state.riichi_pending_acceptance, Some(0));
    assert_eq!(state.last_discard, Some((0, 0)));
    assert_eq!(state.current_player, 1);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_count, 1);
    assert_eq!(state.active_players[0], 1);
    assert!(state.needs_tsumo);
    assert!(!state.is_first_turn);
    assert!(!state.is_after_kan);

    state.apply_log_action(&LogAction::ChiPengGang {
        seat: 1,
        meld_type: MeldType::Daiminkan,
        tiles: vec![0, 1, 2, 3],
        froms: vec![0, 1, 1, 1],
    });

    assert_eq!(state.players[0].score, 24_000);
    assert_eq!(state.riichi_sticks, 1);
    assert_eq!(state.riichi_pending_acceptance, None);
    assert!(!state.players[0].nagashi_eligible);
    assert_eq!(state.players[1].hand_slice(), &[8]);
    assert_eq!(state.players[1].meld_count, 1);
    let meld = &state.players[1].melds_slice()[0];
    assert_eq!(meld.meld_type, MeldType::Daiminkan);
    assert_eq!(meld.tiles_slice(), &[0, 1, 2, 3]);
    assert_eq!(state.current_player, 1);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_count, 1);
    assert_eq!(state.active_players[0], 1);
    assert!(state.needs_tsumo);
    assert!(state.is_after_kan);

    state.apply_log_action(&LogAction::DealTile {
        seat: 1,
        tile: 12,
        doras: None,
        left_tile_count: None,
    });

    assert_eq!(state.players[1].hand_slice(), &[8, 12]);
    assert_eq!(state.drawn_tile, Some(12));
    assert_eq!(state.current_player, 1);
    assert!(state.is_rinshan_flag);
    assert!(!state.needs_tsumo);
    assert!(!state.is_after_kan);
    assert_eq!(state.wall.tile_count, 4);
}

#[test]
fn log_discard_then_deal_tile_finalizes_riichi_and_breaks_nagashi() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);

    state.players[0].reset_round();
    state.players[0].push_hand(4);
    state.players[0].push_hand(8);
    state.players[0].hand_slice_mut().sort();
    state.players[1].reset_round();
    state.players[1].push_hand(16);
    state.players[1].hand_slice_mut().sort();
    state.drawn_tile = Some(4);
    state.wall.tile_count = 2;

    state.apply_log_action(&LogAction::DiscardTile {
        seat: 0,
        tile: 4,
        is_liqi: true,
        is_wliqi: false,
        doras: None,
    });

    assert_eq!(state.players[0].hand_slice(), &[8]);
    assert!(!state.players[0].nagashi_eligible);
    assert!(state.players[0].riichi_declared);
    assert_eq!(state.riichi_pending_acceptance, Some(0));

    state.apply_log_action(&LogAction::DealTile {
        seat: 1,
        tile: 12,
        doras: None,
        left_tile_count: None,
    });

    assert_eq!(state.players[0].score, 24_000);
    assert_eq!(state.riichi_sticks, 1);
    assert_eq!(state.riichi_pending_acceptance, None);
    assert_eq!(state.players[1].hand_slice(), &[12, 16]);
    assert_eq!(state.drawn_tile, Some(12));
    assert_eq!(state.current_player, 1);
    assert!(!state.is_rinshan_flag);
    assert_eq!(state.wall.tile_count, 1);
}

#[test]
fn replay_claim_and_kan_events_consume_tiles_and_set_melds() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.apply_mjai_event(start_kyoku_with_tehais([
        vec![
            "4p", "4p", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1s", "2s",
        ],
        vec![
            "2m", "3m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s",
        ],
        vec![
            "9s", "9s", "9s", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p",
        ],
        vec![
            "E", "E", "E", "E", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
        ],
    ]));

    state.apply_mjai_event(MjaiEvent::Pon {
        actor: 0,
        target: 3,
        pai: "4p".to_string(),
        consumed: vec!["4p".to_string(), "4p".to_string()],
    });
    assert_eq!(state.current_player, 0);
    assert_eq!(state.players[0].meld_count, 1);
    assert_eq!(state.players[0].melds_slice()[0].meld_type, MeldType::Pon);
    assert!(!state.needs_tsumo);

    state.apply_mjai_event(MjaiEvent::Chi {
        actor: 1,
        target: 0,
        pai: "1m".to_string(),
        consumed: vec!["2m".to_string(), "3m".to_string()],
    });
    assert_eq!(state.current_player, 1);
    assert_eq!(state.players[1].meld_count, 1);
    assert_eq!(state.players[1].melds_slice()[0].meld_type, MeldType::Chi);
    assert!(!state.needs_tsumo);

    state.apply_mjai_event(MjaiEvent::Kan {
        actor: 2,
        target: 1,
        pai: "9s".to_string(),
        consumed: vec!["9s".to_string(), "9s".to_string(), "9s".to_string()],
    });
    assert_eq!(state.current_player, 2);
    assert_eq!(state.players[2].meld_count, 1);
    assert_eq!(
        state.players[2].melds_slice()[0].meld_type,
        MeldType::Daiminkan
    );
    assert!(state.needs_tsumo);

    state.apply_mjai_event(MjaiEvent::Ankan {
        actor: 3,
        consumed: vec![
            "E".to_string(),
            "E".to_string(),
            "E".to_string(),
            "E".to_string(),
        ],
    });
    assert_eq!(state.players[3].meld_count, 1);
    let meld = &state.players[3].melds_slice()[0];
    assert_eq!(meld.meld_type, MeldType::Ankan);
    assert!(!meld.opened);
    assert!(state.needs_tsumo);
}

#[test]
fn log_ankan_and_kakan_update_melds_and_last_discard() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);

    state.players[0].reset_round();
    for tile in [16, 17, 18, 19, 40] {
        state.players[0].push_hand(tile);
    }
    state.players[0].hand_slice_mut().sort();

    state.apply_log_action(&LogAction::AnGangAddGang {
        seat: 0,
        meld_type: MeldType::Ankan,
        tiles: vec![16, 17, 18, 19],
        tile_raw_id: 16,
        doras: None,
    });

    assert_eq!(state.players[0].hand_slice(), &[40]);
    assert_eq!(state.players[0].meld_count, 1);
    assert_eq!(state.players[0].melds_slice()[0].meld_type, MeldType::Ankan);
    assert_eq!(
        state.players[0].melds_slice()[0].tiles_slice(),
        &[16, 17, 18, 19]
    );
    assert_eq!(state.last_discard, Some((0, 16)));
    assert_eq!(state.current_player, 0);
    assert!(state.needs_tsumo);
    assert!(state.is_after_kan);

    state.players[1].reset_round();
    state.players[1].push_hand(7);
    state.players[1].push_hand(40);
    state.players[1].hand_slice_mut().sort();
    state.players[1].push_meld(Meld::new(MeldType::Pon, &[4, 5, 6], true, 0, Some(4)));

    state.apply_log_action(&LogAction::AnGangAddGang {
        seat: 1,
        meld_type: MeldType::Kakan,
        tiles: vec![7],
        tile_raw_id: 7,
        doras: None,
    });

    assert_eq!(state.players[1].hand_slice(), &[40]);
    assert_eq!(state.players[1].meld_count, 1);
    assert_eq!(state.players[1].melds_slice()[0].meld_type, MeldType::Kakan);
    assert_eq!(
        state.players[1].melds_slice()[0].tiles_slice(),
        &[4, 5, 6, 7]
    );
    assert_eq!(state.last_discard, Some((1, 7)));
    assert_eq!(state.current_player, 1);
    assert!(state.needs_tsumo);
    assert!(state.is_after_kan);
}

#[test]
fn chi_peng_gang_sets_pao_for_final_dragon_and_wind_melds() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);

    state.players[0].reset_round();
    state.players[0].push_hand(133);
    state.players[0].push_hand(134);
    state.players[0].push_meld(Meld::new(
        MeldType::Pon,
        &[124, 125, 126],
        true,
        2,
        Some(124),
    ));
    state.players[0].push_meld(Meld::new(
        MeldType::Pon,
        &[128, 129, 130],
        true,
        3,
        Some(128),
    ));
    state.last_discard = Some((1, 132));

    state.apply_log_action(&LogAction::ChiPengGang {
        seat: 0,
        meld_type: MeldType::Pon,
        tiles: vec![132, 133, 134],
        froms: vec![1, 0, 0],
    });

    assert_eq!(state.players[0].hand_len, 0);
    assert_eq!(state.players[0].meld_count, 3);
    assert_eq!(state.players[0].pao_get(37), Some(1));
    assert!(!state.players[1].nagashi_eligible);

    state.players[2].reset_round();
    state.players[2].push_hand(121);
    state.players[2].push_hand(122);
    state.players[2].push_hand(123);
    state.players[2].push_meld(Meld::new(
        MeldType::Pon,
        &[108, 109, 110],
        true,
        0,
        Some(108),
    ));
    state.players[2].push_meld(Meld::new(
        MeldType::Pon,
        &[112, 113, 114],
        true,
        0,
        Some(112),
    ));
    state.players[2].push_meld(Meld::new(
        MeldType::Pon,
        &[116, 117, 118],
        true,
        1,
        Some(116),
    ));
    state.last_discard = Some((3, 120));

    state.apply_log_action(&LogAction::ChiPengGang {
        seat: 2,
        meld_type: MeldType::Daiminkan,
        tiles: vec![120, 121, 122, 123],
        froms: vec![3, 2, 2, 2],
    });

    assert_eq!(state.players[2].hand_len, 0);
    assert_eq!(state.players[2].meld_count, 4);
    assert_eq!(state.players[2].pao_get(50), Some(3));
    assert_eq!(state.current_player, 2);
    assert!(state.needs_tsumo);
    assert!(state.is_after_kan);
}

#[test]
fn hule_standard_tsumo_applies_honba_and_riichi_sticks() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);

    state.oya = 0;
    state.honba = 1;
    state.riichi_sticks = 1;
    state.players[0].score = 24_000;
    state.players[1].score = 25_000;
    state.players[2].score = 25_000;
    state.players[3].score = 25_000;

    state.apply_log_action(&LogAction::Hule {
        hules: vec![make_hule(2, true, false, vec![3], 0, 4_000, 2_000)],
    });

    assert_eq!(state.players[0].score, 19_900);
    assert_eq!(state.players[1].score, 22_900);
    assert_eq!(state.players[2].score, 34_300);
    assert_eq!(state.players[3].score, 22_900);
    assert_eq!(state.riichi_sticks, 0);
    assert!(state.is_done);
}

#[test]
fn hule_ron_pao_splits_payment_between_discarder_and_liable_player() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);

    state.honba = 1;
    state.last_discard = Some((1, 33));
    state.players[2].pao_insert(37, 3);

    state.apply_log_action(&LogAction::Hule {
        hules: vec![make_hule(2, false, true, vec![37], 32_000, 0, 0)],
    });

    assert_eq!(state.players[0].score, 25_000);
    assert_eq!(state.players[1].score, 9_000);
    assert_eq!(state.players[2].score, 57_300);
    assert_eq!(state.players[3].score, 8_700);
    assert!(state.is_done);
}

#[test]
fn no_tile_without_nagashi_or_tenpai_keeps_scores_unchanged() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);

    set_backing_hand_tiles(&mut state, 0, &[]);
    set_backing_hand_tiles(&mut state, 1, &[]);
    set_backing_hand_tiles(&mut state, 2, &[]);
    set_backing_hand_tiles(&mut state, 3, &[]);
    for player in &mut state.players {
        player.score = 25_000;
        player.nagashi_eligible = false;
    }

    state.apply_log_action(&LogAction::NoTile);

    assert_eq!(state.players[0].score, 25_000);
    assert_eq!(state.players[1].score, 25_000);
    assert_eq!(state.players[2].score, 25_000);
    assert_eq!(state.players[3].score, 25_000);
    assert!(state.is_done);
}

#[test]
fn hule_multi_ron_awards_honba_once_and_riichi_to_first_winner() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);

    state.honba = 2;
    state.riichi_sticks = 2;
    state.riichi_pending_acceptance = Some(3);
    state.last_discard = Some((1, 33));
    state.players[0].score = 24_000;
    state.players[1].score = 25_000;
    state.players[2].score = 25_000;
    state.players[3].score = 24_000;

    state.apply_log_action(&LogAction::Hule {
        hules: vec![
            crate::replay::HuleData {
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
            },
            crate::replay::HuleData {
                seat: 2,
                hu_tile: 33,
                zimo: false,
                count: 0,
                fu: 40,
                fans: vec![3],
                li_doras: None,
                yiman: false,
                point_rong: 8_000,
                point_zimo_qin: 0,
                point_zimo_xian: 0,
            },
        ],
    });

    assert_eq!(state.players[0].score, 30_500);
    assert_eq!(state.players[1].score, 12_500);
    assert_eq!(state.players[2].score, 33_000);
    assert_eq!(state.players[3].score, 24_000);
    assert_eq!(state.riichi_pending_acceptance, None);
    assert_eq!(state.riichi_sticks, 0);
    assert!(state.is_done);
}

#[test]
fn no_tile_applies_nagashi_mangan_and_accepts_pending_riichi() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);

    state.oya = 0;
    state.players[0].score = 25_000;
    state.players[1].score = 25_000;
    state.players[2].score = 25_000;
    state.players[3].score = 26_000;
    state.players[0].nagashi_eligible = true;
    state.players[1].nagashi_eligible = false;
    state.players[2].nagashi_eligible = false;
    state.players[3].nagashi_eligible = false;
    state.riichi_pending_acceptance = Some(3);
    state.honba = 2;

    state.apply_log_action(&LogAction::NoTile);

    assert_eq!(state.players[0].score, 37_600);
    assert_eq!(state.players[1].score, 20_800);
    assert_eq!(state.players[2].score, 20_800);
    assert_eq!(state.players[3].score, 20_800);
    assert_eq!(state.riichi_pending_acceptance, None);
    assert_eq!(state.riichi_sticks, 1);
    assert!(state.is_done);
}

#[test]
fn try_apply_mjai_event_rejects_missing_discard_tile() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.apply_mjai_event(start_kyoku_event());

    let err = state
        .try_apply_mjai_event(MjaiEvent::Dahai {
            actor: 2,
            pai: "6m".to_string(),
            tsumogiri: false,
        })
        .expect_err("discarding absent tile should reject");

    assert!(err.to_string().contains("tile missing from hand"));
}

#[test]
fn try_apply_mjai_event_rejects_kakan_without_upgrade_tile() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);
    state.apply_mjai_event(start_kyoku_event());
    state.players[0].reset_round();
    state.players[0].push_meld(Meld::new(MeldType::Pon, &[4, 5, 6], true, 1, Some(4)));

    let err = state
        .try_apply_mjai_event(MjaiEvent::Kakan {
            actor: 0,
            pai: "2m".to_string(),
        })
        .expect_err("kakan without fourth hand tile should reject");

    assert!(err.to_string().contains("upgrade tile"));
}

#[test]
fn liuju_finalizes_pending_riichi_without_other_score_changes() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(7), 0, rule);

    state.players[0].score = 24_000;
    state.players[1].score = 25_000;
    state.players[2].score = 26_000;
    state.players[3].score = 25_000;
    state.riichi_pending_acceptance = Some(1);
    state.riichi_sticks = 2;

    state.apply_log_action(&LogAction::LiuJu {
        lj_type: 1,
        seat: 1,
        tiles: vec![0, 4, 8],
    });

    assert_eq!(state.players[0].score, 24_000);
    assert_eq!(state.players[1].score, 24_000);
    assert_eq!(state.players[2].score, 26_000);
    assert_eq!(state.players[3].score, 25_000);
    assert_eq!(state.riichi_pending_acceptance, None);
    assert_eq!(state.riichi_sticks, 3);
    assert!(state.is_done);
}
