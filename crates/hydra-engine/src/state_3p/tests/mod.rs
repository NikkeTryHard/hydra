use super::*;
use crate::test_support::{parsed_tile, tiles_to_u32};
mod event_handler;
mod game_mode;
mod legal_actions;
mod sanma;

fn test_state_with_mode(game_mode: u8, skip_mjai_logging: bool) -> GameState3P {
    GameState3P::new(
        game_mode,
        skip_mjai_logging,
        Some(7),
        0,
        GameRule::default_tenhou(),
    )
}

fn test_state(skip_mjai_logging: bool) -> GameState3P {
    test_state_with_mode(5, skip_mjai_logging)
}
fn direct_state(rule: GameRule) -> GameState3P {
    let mut state = GameState3P::new(5, true, Some(7), 0, rule);
    for player in &mut state.players {
        player.reset_round();
        player.score = 35_000;
        player.score_delta = 0;
    }
    state.phase = Phase::WaitAct;
    state.current_player = 0;
    state.active_players = [0; 4];
    state.active_player_count = 1;
    state.is_done = false;
    state.needs_tsumo = false;
    state.needs_initialize_next_round = false;
    state.pending_oya_won = false;
    state.pending_is_draw = false;
    state.riichi_sticks = 0;
    state.last_discard = None;
    state.pending_kan = None;
    state.is_rinshan_flag = false;
    state.is_first_turn = false;
    state.riichi_pending_acceptance = None;
    state.drawn_tile = None;
    state.current_claim_counts = [0; 3];
    state.win_results = [None; NP];
    state.last_win_results = [None; NP];
    state.wall.tile_count = 20;
    state.wall.draw_cursor = 0;
    state
}

fn set_closed_hand(state: &mut GameState3P, pid: usize, text: &str) {
    let (tiles, melds) = crate::parser::parse_hand_internal(text)
        .expect("test hand should parse into a direct sanma state");
    assert!(
        melds.is_empty(),
        "expected closed hand text without meld syntax"
    );

    state.players[pid].hand = [0; 14];
    state.players[pid].hand_len = 0;
    state.players[pid].melds = [Meld::default(); 4];
    state.players[pid].meld_count = 0;

    for tile in tiles {
        state.players[pid].push_hand(tile);
    }
    state.players[pid].hand_slice_mut().sort();
}
fn test_start_kyoku_event() -> Value {
    serde_json::json!({
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [35000, 35000, 35000],
        "dora_marker": "1p",
        "tehais": [
            ["1p", "2p", "3p"],
            ["4p", "5p", "6p"],
            ["7p", "8p", "9p"]
        ]
    })
}

#[test]
fn helper_methods_manage_active_players_and_claims() {
    let mut state = test_state(true);

    state.set_single_active_player(2);
    assert_eq!(state.active_player_slice(), &[2]);

    state.set_active_players_from_slice(&[1, 2]);
    assert_eq!(state.active_player_slice(), &[1, 2]);

    state.clear_active_players();
    assert!(state.active_player_slice().is_empty());

    let ron = Action::new(ActionType::Ron, Some(88), &[], Some(1));
    let pon = Action::new(ActionType::Pon, Some(88), &[84, 85], Some(1));
    state.push_claim(1, ron);
    state.push_claim(1, pon);
    assert_eq!(state.claims_slice(1), &[ron, pon]);

    let many_claims = vec![Action::new(ActionType::Pass, None, &[], Some(1)); 60];
    state.set_claims_from_vec(1, &many_claims);
    assert_eq!(state.current_claim_counts[1], 54);
    assert_eq!(state.claims_slice(1).len(), 54);
    assert!(state
        .claims_slice(1)
        .iter()
        .all(|claim| claim.action_type == ActionType::Pass));

    state.clear_claims();
    assert_eq!(state.current_claim_counts, [0; 3]);
    assert!(state.claims_slice(1).is_empty());
}

#[test]
fn reset_rebuilds_logging_buffers_when_logging_enabled() {
    let mut state = test_state(false);
    state.reset();

    state.mjai_log.push("stale".to_string());
    state.mjai_log_per_player[0].push("p0".to_string());
    state.mjai_log_per_player[1].push("p1".to_string());
    state.player_event_counts = [3, 2, 1];

    state.reset();

    assert_eq!(state.player_event_counts, [0; 3]);
    assert_eq!(state.mjai_log.len(), 1);
    assert!(state.mjai_log[0].contains("start_game"));
    assert_eq!(state.mjai_log_per_player[0].len(), 1);
    assert_eq!(state.mjai_log_per_player[1].len(), 1);
    assert_eq!(state.mjai_log_per_player[2].len(), 1);
}

#[test]
fn reset_drops_logs_without_readding_events_when_logging_is_disabled() {
    let mut state = test_state(true);
    state.mjai_log.push("stale".to_string());
    state.mjai_log_per_player[0].push("p0".to_string());
    state.player_event_counts = [4, 5, 6];

    state.reset();

    assert!(state.mjai_log.is_empty());
    assert!(state
        .mjai_log_per_player
        .iter()
        .all(|events| events.is_empty()));
    assert_eq!(state.player_event_counts, [0; 3]);
}

#[test]
fn reset_for_new_game_recreates_state_with_new_seed() {
    let mut state = test_state(true);
    state.current_player = 2;
    state.turn_count = 9;
    state.pending_oya_won = true;
    state.pending_is_draw = true;
    state.needs_initialize_next_round = true;
    state.active_players = [2, 1, 0, 0];
    state.active_player_count = 3;
    state.last_discard = Some((1, 40));
    state.riichi_sticks = 4;

    state.reset_for_new_game(Some(99));

    assert_eq!(state.seed, Some(99));
    assert_eq!(state.round_wind, 0);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.turn_count, 0);
    assert!(!state.pending_oya_won);
    assert!(!state.pending_is_draw);
    assert!(!state.needs_initialize_next_round);
    assert_eq!(state.active_player_slice(), &[0]);
    assert_eq!(state.last_discard, None);
    assert_eq!(state.riichi_sticks, 0);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.game_mode, 5);
    assert!(state.skip_mjai_logging);
}

#[test]
fn initialize_next_round_keeps_oya_and_increments_honba_after_oya_win() {
    let mut state = test_state(true);
    state.oya = 1;
    state.honba = 2;
    state.round_wind = 0;
    state.current_player = 2;
    state.phase = Phase::WaitResponse;
    state.active_player_count = 0;

    state._initialize_next_round(true, false);

    assert_eq!(state.oya, 1);
    assert_eq!(state.honba, 3);
    assert_eq!(state.round_wind, 0);
    assert_eq!(state.current_player, 1);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[1]);
    assert!(state.drawn_tile.is_some());
}

#[test]
fn initialize_next_round_draw_wraps_oya_and_advances_round_wind() {
    let mut state = test_state(true);
    state.oya = 2;
    state.honba = 1;
    state.round_wind = 0;

    state._initialize_next_round(false, true);

    assert_eq!(state.oya, 0);
    assert_eq!(state.honba, 2);
    assert_eq!(state.round_wind, 1);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.active_player_slice(), &[0]);
}

#[test]
fn initialize_next_round_loss_resets_honba_and_rotates_oya() {
    let mut state = test_state(true);
    state.oya = 1;
    state.honba = 3;
    state.round_wind = 0;

    state._initialize_next_round(false, false);

    assert_eq!(state.oya, 2);
    assert_eq!(state.honba, 0);
    assert_eq!(state.round_wind, 0);
    assert_eq!(state.current_player, 2);
    assert_eq!(state.active_player_slice(), &[2]);
}

#[test]
fn reveal_kan_dora_stops_after_five_indicators() {
    let mut state = test_state(true);

    for _ in 0..10 {
        state._reveal_kan_dora();
    }

    assert_eq!(state.wall.dora_indicator_count, 5);
    assert_eq!(state.wall.dora_indicator_slice().len(), 5);
}

#[test]
fn reveal_kan_dora_logs_marker_and_ura_helpers_follow_indicator_count() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();

    let expected_marker = tid_to_mjai(state.wall.dora_indicator_tiles[1]);
    let expected_ura = vec![
        tid_to_mjai(state.wall.ura_indicator_tiles[0]),
        tid_to_mjai(state.wall.ura_indicator_tiles[1]),
    ];
    let expected_ura_ids = vec![
        state.wall.ura_indicator_tiles[0],
        state.wall.ura_indicator_tiles[1],
    ];

    state._reveal_kan_dora();

    assert_eq!(state.wall.dora_indicator_count, 2);
    assert_eq!(state._get_ura_markers(), expected_ura);
    assert_eq!(state._get_ura_indicators(), expected_ura_ids);

    let event: Value = serde_json::from_str(state.mjai_log.last().unwrap()).unwrap();
    assert_eq!(event["type"], Value::String("dora".to_string()));
    assert_eq!(event["dora_marker"], Value::String(expected_marker));
}

#[test]
fn push_mjai_event_masks_start_kyoku_hands_for_other_players() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();

    state._push_mjai_event(test_start_kyoku_event());

    let p0: Value = serde_json::from_str(&state.mjai_log_per_player[0][0]).unwrap();
    let p1: Value = serde_json::from_str(&state.mjai_log_per_player[1][0]).unwrap();

    assert_eq!(p0["tehais"][0], serde_json::json!(["1p", "2p", "3p"]));
    assert_eq!(p0["tehais"][1], serde_json::json!(["?", "?", "?"]));
    assert_eq!(p1["tehais"][0], serde_json::json!(["?", "?", "?"]));
    assert_eq!(p1["tehais"][1], serde_json::json!(["4p", "5p", "6p"]));
}

#[test]
fn push_mjai_event_start_kyoku_masks_missing_tehai_lengths_to_default_13() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state._push_mjai_event(serde_json::json!({
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "dora_marker": "1m",
        "tehais": [
            ["1p", "2p", "3p"],
            serde_json::Value::Null,
            ["7p"]
        ]
    }));

    let p0: Value = serde_json::from_str(&state.mjai_log_per_player[0][0]).unwrap();
    let p1: Value = serde_json::from_str(&state.mjai_log_per_player[1][0]).unwrap();
    let p2: Value = serde_json::from_str(&state.mjai_log_per_player[2][0]).unwrap();

    assert_eq!(p0["tehais"][0], serde_json::json!(["1p", "2p", "3p"]));
    assert_eq!(p0["tehais"][1].as_array().unwrap().len(), 13);
    assert_eq!(p0["tehais"][2], serde_json::json!(["?"]));

    assert_eq!(p1["tehais"][0], serde_json::json!(["?", "?", "?"]));
    assert_eq!(p1["tehais"][1], serde_json::Value::Null);
    assert_eq!(p1["tehais"][2], serde_json::json!(["?"]));

    assert_eq!(p2["tehais"][0], serde_json::json!(["?", "?", "?"]));
    assert_eq!(p2["tehais"][1].as_array().unwrap().len(), 13);
    assert_eq!(p2["tehais"][2], serde_json::json!(["7p"]));
}

#[test]
fn push_mjai_event_masks_tsumo_tile_for_non_actor_players() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();

    state._push_mjai_event(serde_json::json!({
        "type": "tsumo",
        "actor": 1,
        "pai": "5p"
    }));

    let p0: Value = serde_json::from_str(&state.mjai_log_per_player[0][0]).unwrap();
    let p1: Value = serde_json::from_str(&state.mjai_log_per_player[1][0]).unwrap();

    assert_eq!(p0["pai"], Value::String("?".to_string()));
    assert_eq!(p1["pai"], Value::String("5p".to_string()));
}

#[test]
fn push_mjai_event_keeps_tsumo_tile_visible_when_actor_is_unknown() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();

    state._push_mjai_event(serde_json::json!({
        "type": "tsumo",
        "pai": "5p"
    }));

    let p0: Value = serde_json::from_str(&state.mjai_log_per_player[0][0]).unwrap();
    let p1: Value = serde_json::from_str(&state.mjai_log_per_player[1][0]).unwrap();
    let p2: Value = serde_json::from_str(&state.mjai_log_per_player[2][0]).unwrap();

    assert_eq!(p0["pai"], Value::String("5p".to_string()));
    assert_eq!(p1["pai"], Value::String("5p".to_string()));
    assert_eq!(p2["pai"], Value::String("5p".to_string()));
}

#[test]
fn push_mjai_event_is_noop_when_logging_disabled() {
    let mut state = test_state(true);

    state._push_mjai_event(serde_json::json!({ "type": "tsumo", "actor": 0, "pai": "1p" }));

    assert!(state.mjai_log.is_empty());
    assert!(state
        .mjai_log_per_player
        .iter()
        .all(|events| events.is_empty()));
}

#[test]
fn get_observation_masks_other_hands_and_drains_only_new_events() {
    let mut state = test_state(true);
    state.mjai_log_per_player[0] = vec![
        "already-seen".to_string(),
        "fresh-event-a".to_string(),
        "fresh-event-b".to_string(),
    ];
    state.player_event_counts[0] = 1;
    state.riichi_sutehais[0] = Some(12);
    state.last_tedashis[0] = Some(16);
    state.last_discard = Some((40, 2));

    let expected_hand = tiles_to_u32(state.players[0].hand_slice());

    let obs = state.get_observation(0);

    assert_eq!(obs.hands[0], expected_hand);
    assert!(obs.hands[1].is_empty());
    assert!(obs.hands[2].is_empty());
    assert_eq!(obs.new_events(), vec!["fresh-event-a", "fresh-event-b"]);
    assert_eq!(obs.riichi_sutehais[0], Some(12));
    assert_eq!(obs.last_tedashis[0], Some(16));
    assert_eq!(obs.last_discard, Some(40));
    assert_eq!(state.player_event_counts[0], 3);

    let obs_again = state.get_observation(0);
    assert!(obs_again.new_events().is_empty());
}

#[test]
fn get_observation_limits_legal_actions_to_visible_turn_owners() {
    let mut state = test_state(true);

    let current_obs = state.get_observation(0);
    assert!(!current_obs.legal_actions_method().is_empty());

    let hidden_obs = state.get_observation(1);
    assert!(hidden_obs.legal_actions_method().is_empty());

    state.phase = Phase::WaitResponse;
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Ron, Some(12), &[], Some(1));

    let response_obs = state.get_observation(1);
    let response_legals = response_obs.legal_actions_method();
    assert!(response_legals
        .iter()
        .any(|action| action.action_type == ActionType::Ron));
    assert!(response_legals
        .iter()
        .any(|action| action.action_type == ActionType::Pass));

    state.is_done = true;
    let done_obs = state.get_observation(1);
    assert!(done_obs.legal_actions_method().is_empty());
}

#[test]
fn process_end_game_marks_done_and_emits_end_game_only_when_logging_enabled() {
    let mut logged = test_state(false);
    logged.mjai_log.clear();
    logged.mjai_log_per_player = Default::default();

    logged._process_end_game();

    assert!(logged.is_done);
    let logged_event: Value = serde_json::from_str(logged.mjai_log.last().unwrap()).unwrap();
    assert_eq!(logged_event["type"], Value::String("end_game".to_string()));
    assert_eq!(logged.mjai_log_per_player[0].len(), 1);
    assert_eq!(logged.mjai_log_per_player[1].len(), 1);
    assert_eq!(logged.mjai_log_per_player[2].len(), 1);

    let mut silent = test_state(true);
    silent._process_end_game();

    assert!(silent.is_done);
    assert!(silent.mjai_log.is_empty());
    assert!(silent
        .mjai_log_per_player
        .iter()
        .all(|events| events.is_empty()));
}

#[test]
fn check_abortive_draw_triggers_suukansansen_only_with_four_kans_by_multiple_players() {
    let mut triggered = test_state(true);
    triggered.players[0].push_meld(Meld::new(MeldType::Ankan, &[0, 1, 2, 3], false, -1, None));
    triggered.players[0].push_meld(Meld::new(MeldType::Kakan, &[4, 5, 6, 7], true, -1, None));
    triggered.players[1].push_meld(Meld::new(
        MeldType::Daiminkan,
        &[8, 9, 10, 11],
        true,
        0,
        Some(8),
    ));
    triggered.players[2].push_meld(Meld::new(
        MeldType::Ankan,
        &[12, 13, 14, 15],
        false,
        -1,
        None,
    ));
    let scores_before: Vec<i32> = triggered
        .players
        .iter()
        .map(|player| player.score)
        .collect();

    assert!(triggered.check_abortive_draw());
    assert!(!triggered.needs_tsumo);
    assert!(triggered.drawn_tile.is_some());
    assert_eq!(triggered.phase, Phase::WaitAct);
    assert_eq!(triggered.current_player, triggered.oya);
    assert_eq!(triggered.active_player_slice(), &[triggered.oya]);
    let scores_after: Vec<i32> = triggered
        .players
        .iter()
        .map(|player| player.score)
        .collect();
    assert_eq!(scores_after, scores_before);
    assert!(triggered
        .players
        .iter()
        .all(|player| player.score_delta == 0));

    let mut same_owner = test_state(true);
    for meld_tiles in [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]] {
        same_owner.players[0].push_meld(Meld::new(MeldType::Ankan, &meld_tiles, false, -1, None));
    }

    assert!(!same_owner.check_abortive_draw());

    let mut not_enough = test_state(true);
    not_enough.players[0].push_meld(Meld::new(MeldType::Ankan, &[0, 1, 2, 3], false, -1, None));
    not_enough.players[1].push_meld(Meld::new(MeldType::Kakan, &[4, 5, 6, 7], true, -1, None));
    not_enough.players[2].push_meld(Meld::new(
        MeldType::Daiminkan,
        &[8, 9, 10, 11],
        true,
        0,
        Some(8),
    ));

    assert!(!not_enough.check_abortive_draw());
}

#[test]
fn trigger_ryukyoku_illegal_action_penalizes_non_oya_offender_and_keeps_renchan() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.oya = 0;
    state.current_player = 1;
    state.phase = Phase::WaitResponse;

    state._trigger_ryukyoku("Error: Illegal Action by Player 1");

    assert_eq!(state.players[0].score, 39000);
    assert_eq!(state.players[0].score_delta, 4000);
    assert_eq!(state.players[1].score, 29000);
    assert_eq!(state.players[1].score_delta, -6000);
    assert_eq!(state.players[2].score, 37000);
    assert_eq!(state.players[2].score_delta, 2000);
    assert_eq!(state.oya, 0);
    assert_eq!(state.honba, 1);
    assert_eq!(state.round_wind, 0);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);

    let ryukyoku_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"ryukyoku\""))
        .expect("ryukyoku event should be logged");
    let ryukyoku_event: Value = serde_json::from_str(&state.mjai_log[ryukyoku_idx]).unwrap();
    assert_eq!(
        ryukyoku_event["reason"],
        Value::String("Error: Illegal Action by Player 1".to_string())
    );
    assert_eq!(
        ryukyoku_event["deltas"],
        serde_json::json!([4000, -6000, 2000])
    );
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
}

#[test]
fn initialize_next_round_ends_game_on_negative_score_before_restarting() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players[2].score = -100;
    state.oya = 1;
    state.honba = 2;
    state.round_wind = 0;

    state._initialize_next_round(false, false);

    assert!(state.is_done);
    assert_eq!(state.oya, 1);
    assert_eq!(state.honba, 2);
    assert_eq!(state.round_wind, 0);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));
    assert!(!state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
}

#[test]
fn initialize_next_round_single_mode_ends_game_immediately() {
    let mut state = test_state_with_mode(3, false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.oya = 2;
    state.honba = 1;
    state.round_wind = 0;

    state._initialize_next_round(false, true);

    assert!(state.is_done);
    assert_eq!(state.oya, 2);
    assert_eq!(state.honba, 1);
    assert_eq!(state.round_wind, 0);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));
    assert!(!state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
}

#[test]
fn initialize_next_round_east_mode_continues_into_south_if_nobody_has_30000() {
    let mut state = test_state_with_mode(4, false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players[0].score = 29000;
    state.players[1].score = 28000;
    state.players[2].score = 27000;
    state.oya = 2;
    state.honba = 1;
    state.round_wind = 0;

    state._initialize_next_round(false, true);

    assert!(!state.is_done);
    assert_eq!(state.oya, 0);
    assert_eq!(state.honba, 2);
    assert_eq!(state.round_wind, 1);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
    assert!(!state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));
}

#[test]
fn initialize_next_round_east_mode_ends_when_south_starts_with_30000_leader() {
    let mut state = test_state_with_mode(4, false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players[0].score = 31000;
    state.players[1].score = 25000;
    state.players[2].score = 24000;
    state.oya = 2;
    state.honba = 1;
    state.round_wind = 0;

    state._initialize_next_round(false, true);

    assert!(state.is_done);
    assert_eq!(state.oya, 2);
    assert_eq!(state.honba, 1);
    assert_eq!(state.round_wind, 0);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));
    assert!(!state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
}

#[test]
fn initialize_round_uses_provided_wall_scores_and_resets_round_state() {
    let mut state = test_state(true);
    state.phase = Phase::WaitResponse;
    state.is_done = true;
    state.pending_kan = Some((1, Action::new(ActionType::Kakan, Some(16), &[], Some(1))));
    state.is_rinshan_flag = true;
    state.wall.rinshan_draw_count = 3;
    state.wall.pending_kan_dora_count = 2;
    state.is_first_turn = false;
    state.riichi_pending_acceptance = Some(2);
    state.turn_count = 8;
    state.needs_tsumo = false;
    state.needs_initialize_next_round = true;
    state.pending_oya_won = true;
    state.pending_is_draw = true;
    state.last_discard = Some((2, 44));
    state.win_results[0] = Some(WinResult::new(
        false, false, 0, 0, 0, [0u32; 16], 0, 0, 0, None, false,
    ));
    state.last_win_results[1] = Some(WinResult::new(
        false, false, 0, 0, 0, [0u32; 16], 0, 0, 0, None, false,
    ));
    state.riichi_sutehais = [Some(12), Some(16), Some(20)];
    state.last_tedashis = [Some(24), Some(28), Some(32)];
    state.active_players = [2, 1, 0, 0];
    state.active_player_count = 3;
    state.players[0].riichi_declared = true;
    state.players[1].double_riichi_declared = true;
    state.players[2].missed_agari_doujun = true;
    state.players[0].nagashi_eligible = false;
    state.players[1].ippatsu_cycle = true;
    state.players[2].push_forbidden(60);
    state.players[0].pao_insert(37, 2);

    state._initialize_round(
        1,
        1,
        2,
        3,
        Some((0..108).collect()),
        Some(vec![11000, 22000, 33000]),
    );

    assert_eq!(state.oya, 1);
    assert_eq!(state.kyoku_idx, 1);
    assert_eq!(state.current_player, 1);
    assert_eq!(state.honba, 2);
    assert_eq!(state.riichi_sticks, 3);
    assert_eq!(state.round_wind, 1);
    assert!(!state.is_done);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[1]);
    assert!(state.pending_kan.is_none());
    assert!(!state.is_rinshan_flag);
    assert_eq!(state.wall.rinshan_draw_count, 0);
    assert_eq!(state.wall.pending_kan_dora_count, 0);
    assert!(state.is_first_turn);
    assert!(state.riichi_pending_acceptance.is_none());
    assert_eq!(state.turn_count, 0);
    assert!(!state.needs_tsumo);
    assert!(!state.needs_initialize_next_round);
    assert!(!state.pending_oya_won);
    assert!(!state.pending_is_draw);
    assert_eq!(state.last_discard, None);
    assert!(state.win_results.iter().all(Option::is_none));
    assert!(state.last_win_results.iter().all(Option::is_none));
    assert_eq!(state.riichi_sutehais, [None; 3]);
    assert_eq!(state.last_tedashis, [None; 3]);
    assert_eq!(state.players[0].score, 11000);
    assert_eq!(state.players[1].score, 22000);
    assert_eq!(state.players[2].score, 33000);
    assert!(!state.players[0].riichi_declared);
    assert!(!state.players[1].double_riichi_declared);
    assert!(!state.players[2].missed_agari_doujun);
    assert!(state.players[0].nagashi_eligible);
    assert!(!state.players[1].ippatsu_cycle);
    assert!(state.players[2].forbidden_slice().is_empty());
    assert_eq!(state.players[0].pao_count, 0);
    assert_eq!(state.players[0].hand_slice().len(), 13);
    assert_eq!(state.players[1].hand_slice().len(), 14);
    assert_eq!(state.players[2].hand_slice().len(), 13);
    assert_eq!(state.wall.tile_count, 68);
    assert_eq!(state.wall.draw_cursor, 0);
    assert_eq!(state.wall.dora_indicator_slice(), &[99]);
    assert_eq!(state.drawn_tile, Some(39));
}

#[test]
fn deal_next_draws_from_wall_and_clears_forbidden_discards() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.current_player = 1;
    state.phase = Phase::WaitResponse;
    state.needs_tsumo = true;
    state.is_rinshan_flag = true;
    state.drawn_tile = None;
    state.players[1].push_forbidden(44);

    let expected_tile = state.wall.tiles[state.wall.tile_count as usize - 1];
    let hand_len_before = state.players[1].hand_slice().len();

    state._deal_next();

    assert!(!state.is_rinshan_flag);
    assert_eq!(state.drawn_tile, Some(expected_tile));
    assert_eq!(state.players[1].hand_slice().len(), hand_len_before + 1);
    assert!(!state.needs_tsumo);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[1]);
    assert!(state.players[1].forbidden_slice().is_empty());

    let event: Value = serde_json::from_str(state.mjai_log.last().unwrap()).unwrap();
    assert_eq!(event["type"], Value::String("tsumo".to_string()));
    assert_eq!(event["actor"], Value::Number(1.into()));
    assert_eq!(event["pai"], Value::String(tid_to_mjai(expected_tile)));
}

#[test]
fn deal_next_exhaustive_draw_with_oya_nagashi_keeps_renchan_and_scores() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.oya = 2;
    state.current_player = 1;
    state.honba = 0;
    state.round_wind = 0;
    state.wall.tile_count = 14;
    state.wall.draw_cursor = 0;
    state.players[0].nagashi_eligible = false;
    state.players[1].nagashi_eligible = false;
    state.players[2].nagashi_eligible = true;

    let nagashi_score = crate::score::calculate_score(5, 30, true, true, 0, 3);

    state._deal_next();

    assert_eq!(
        state.players[0].score,
        35000 - nagashi_score.pay_tsumo_ko as i32
    );
    assert_eq!(
        state.players[1].score,
        35000 - nagashi_score.pay_tsumo_ko as i32
    );
    assert_eq!(
        state.players[2].score,
        35000 + 2 * nagashi_score.pay_tsumo_ko as i32
    );
    assert_eq!(state.oya, 2);
    assert_eq!(state.honba, 1);
    assert_eq!(state.round_wind, 0);
    assert_eq!(state.current_player, 2);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[2]);
    assert!(state.drawn_tile.is_some());

    let ryukyoku_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"ryukyoku\""))
        .expect("ryukyoku event should be logged");
    let ryukyoku_event: Value = serde_json::from_str(&state.mjai_log[ryukyoku_idx]).unwrap();
    assert_eq!(
        ryukyoku_event["reason"],
        Value::String("nagashimangan".to_string())
    );
    assert_eq!(
        ryukyoku_event["deltas"],
        serde_json::json!([
            -(nagashi_score.pay_tsumo_ko as i32),
            -(nagashi_score.pay_tsumo_ko as i32),
            2 * nagashi_score.pay_tsumo_ko as i32,
        ])
    );
}

#[test]
fn trigger_ryukyoku_nagashi_with_multiple_winners_stacks_payments_and_renchan_depends_on_oya() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.oya = 0;
    state.honba = 0;
    state.round_wind = 0;
    state.players.iter_mut().for_each(|player| {
        player.score = 35_000;
        player.score_delta = 0;
        player.nagashi_eligible = false;
    });
    state.players[0].nagashi_eligible = true;
    state.players[2].nagashi_eligible = true;

    let oya_nagashi = crate::score::calculate_score(5, 30, true, true, 0, 3);
    let ko_nagashi = crate::score::calculate_score(5, 30, false, true, 0, 3);

    state._trigger_ryukyoku("exhaustive_draw");

    assert_eq!(
        state.players[0].score,
        35_000 + 2 * oya_nagashi.pay_tsumo_ko as i32 - ko_nagashi.pay_tsumo_oya as i32
    );
    assert_eq!(
        state.players[1].score,
        35_000 - oya_nagashi.pay_tsumo_ko as i32 - ko_nagashi.pay_tsumo_ko as i32
    );
    assert_eq!(
        state.players[2].score,
        35_000 - oya_nagashi.pay_tsumo_ko as i32
            + ko_nagashi.pay_tsumo_oya as i32
            + ko_nagashi.pay_tsumo_ko as i32
    );
    assert_eq!(state.oya, 0);
    assert_eq!(state.honba, 1);
    assert_eq!(state.round_wind, 0);
    let ryukyoku_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"ryukyoku\""))
        .expect("ryukyoku event should be logged");
    let ryukyoku_event: Value = serde_json::from_str(&state.mjai_log[ryukyoku_idx]).unwrap();
    assert_eq!(
        ryukyoku_event["reason"],
        Value::String("nagashimangan".to_string())
    );
    assert_eq!(
        ryukyoku_event["deltas"],
        serde_json::json!([
            2 * oya_nagashi.pay_tsumo_ko as i32 - ko_nagashi.pay_tsumo_oya as i32,
            -(oya_nagashi.pay_tsumo_ko as i32) - ko_nagashi.pay_tsumo_ko as i32,
            -(oya_nagashi.pay_tsumo_ko as i32)
                + ko_nagashi.pay_tsumo_oya as i32
                + ko_nagashi.pay_tsumo_ko as i32,
        ])
    );
}

#[test]
fn trigger_ryukyoku_accepts_pending_riichi_before_scoring_and_carries_stick_forward() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.oya = 0;
    state.riichi_pending_acceptance = Some(2);

    state._trigger_ryukyoku("Error: Illegal Action by Player 1");

    assert_eq!(state.players[0].score, 39000);
    assert_eq!(state.players[1].score, 29000);
    assert_eq!(state.players[2].score, 36000);
    assert_eq!(state.riichi_sticks, 1);
    assert!(state.riichi_pending_acceptance.is_none());
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"reach_accepted\"")));
    let reach_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"reach_accepted\""))
        .expect("reach_accepted should be logged");
    let ryukyoku_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"ryukyoku\""))
        .expect("ryukyoku should be logged");
    assert!(reach_idx < ryukyoku_idx);
}

#[test]
fn trigger_ryukyoku_illegal_action_penalizes_oya_offender() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.oya = 0;

    state._trigger_ryukyoku("Error: Illegal Action by Player 0");

    assert_eq!(state.players[0].score, 27000);
    assert_eq!(state.players[0].score_delta, -8000);
    assert_eq!(state.players[1].score, 39000);
    assert_eq!(state.players[1].score_delta, 4000);
    assert_eq!(state.players[2].score, 39000);
    assert_eq!(state.players[2].score_delta, 4000);
    assert_eq!(state.oya, 0);
    assert_eq!(state.honba, 1);

    let ryukyoku_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"ryukyoku\""))
        .expect("ryukyoku event should be logged");
    let ryukyoku_event: Value = serde_json::from_str(&state.mjai_log[ryukyoku_idx]).unwrap();
    assert_eq!(
        ryukyoku_event["deltas"],
        serde_json::json!([-8000, 4000, 4000])
    );
}

#[test]
fn trigger_ryukyoku_illegal_action_penalizes_non_oya_offender() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.oya = 0;

    state._trigger_ryukyoku("Error: Illegal Action by Player 1");

    assert_eq!(state.players[0].score, 39000);
    assert_eq!(state.players[0].score_delta, 4000);
    assert_eq!(state.players[1].score, 29000);
    assert_eq!(state.players[1].score_delta, -6000);
    assert_eq!(state.players[2].score, 37000);
    assert_eq!(state.players[2].score_delta, 2000);
    let ryukyoku_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"ryukyoku\""))
        .expect("ryukyoku event should be logged");
    let ryukyoku_event: Value = serde_json::from_str(&state.mjai_log[ryukyoku_idx]).unwrap();
    assert_eq!(
        ryukyoku_event["deltas"],
        serde_json::json!([4000, -6000, 2000])
    );
}

#[test]
fn replay_observation_retries_discard_after_temporarily_clearing_riichi() {
    let mut state = test_state(true);
    let drawn_tile = state
        .drawn_tile
        .expect("test state should start with a drawn tile");
    let retry_tile = state.players[0]
        .hand_slice()
        .iter()
        .copied()
        .find(|&tile| tile != drawn_tile)
        .expect("hand should contain a non-drawn tile to discard");
    state.players[0].riichi_declared = true;

    let obs = state
        .get_observation_for_replay(
            0,
            &Action::new(ActionType::Discard, Some(retry_tile), &[], Some(0)),
            "{\"type\":\"dahai\"}",
        )
        .expect("replay discard should succeed after riichi retry path");

    assert!(
        obs.legal_actions_method()
            .iter()
            .any(|action| action.action_type == ActionType::Discard
                && action.tile == Some(retry_tile))
    );
    assert!(!state.players[0].riichi_declared);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
}

#[test]
fn push_mjai_event_keeps_tsumo_tile_visible_when_actor_is_missing() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();

    state._push_mjai_event(serde_json::json!({
        "type": "tsumo",
        "pai": "5p"
    }));

    for pid in 0..3 {
        let event: Value = serde_json::from_str(&state.mjai_log_per_player[pid][0]).unwrap();
        assert_eq!(event["pai"], Value::String("5p".to_string()));
    }
}

#[test]
fn replay_observation_temporarily_injects_claims_and_restores_state_on_success() {
    let mut state = test_state(true);
    state.active_players = [2, 1, 0, 0];
    state.active_player_count = 2;
    state.current_claim_counts[2] = 1;
    state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));

    let obs = state
        .get_observation_for_replay(
            1,
            &Action::new(ActionType::Ron, Some(48), &[], Some(1)),
            "{\"type\":\"hora\"}",
        )
        .expect("ron replay action should be exposed as legal");

    let legal_actions = obs.legal_actions_method();
    assert!(legal_actions
        .iter()
        .any(|action| action.action_type == ActionType::Ron));
    assert!(legal_actions
        .iter()
        .any(|action| action.action_type == ActionType::Pass));

    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_players, [2, 1, 0, 0]);
    assert_eq!(state.active_player_count, 2);
    assert_eq!(state.current_claim_counts, [0, 0, 1]);
    assert_eq!(state.current_claims[2][0].action_type, ActionType::Pass);
}

#[test]
fn replay_observation_restores_state_after_invalid_action_error() {
    let mut state = test_state(true);
    state.active_players = [0, 2, 0, 0];
    state.active_player_count = 2;
    state.current_claim_counts[2] = 1;
    state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));

    let err = state
        .get_observation_for_replay(
            1,
            &Action::new(ActionType::Discard, Some(0), &[], Some(1)),
            "{\"type\":\"dahai\"}",
        )
        .expect_err("non-active player discard should stay illegal in replay observation");

    assert!(matches!(err, RiichiError::InvalidState { .. }));
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_players, [0, 2, 0, 0]);
    assert_eq!(state.active_player_count, 2);
    assert_eq!(state.current_claim_counts, [0, 0, 1]);
    assert_eq!(state.current_claims[2][0].action_type, ActionType::Pass);
}

#[test]
fn step_array_unchecked_initializes_pending_round_before_processing_actions() {
    let mut state = test_state(true);
    state.oya = 1;
    state.honba = 2;
    state.round_wind = 0;
    state.phase = Phase::WaitResponse;
    state.needs_initialize_next_round = true;
    state.pending_oya_won = false;
    state.pending_is_draw = false;
    state.last_discard = Some((0, 44));

    let actions = [
        Some(Action::new(
            ActionType::Discard,
            state.drawn_tile,
            &[],
            Some(state.current_player),
        )),
        None,
        None,
    ];

    state.step_array_unchecked(&actions);

    assert!(!state.needs_initialize_next_round);
    assert_eq!(state.oya, 2);
    assert_eq!(state.honba, 0);
    assert_eq!(state.round_wind, 0);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.current_player, 2);
    assert_eq!(state.active_player_slice(), &[2]);
    assert_eq!(state.last_discard, None);
}

#[test]
fn step_array_unchecked_initializes_pending_draw_round_before_processing_actions() {
    let mut state = test_state(true);
    state.oya = 2;
    state.honba = 1;
    state.round_wind = 0;
    state.phase = Phase::WaitResponse;
    state.needs_initialize_next_round = true;
    state.pending_oya_won = false;
    state.pending_is_draw = true;
    state.last_discard = Some((1, 48));

    state.step_array_unchecked(&[None, None, None]);

    assert!(!state.needs_initialize_next_round);
    assert_eq!(state.oya, 0);
    assert_eq!(state.honba, 2);
    assert_eq!(state.round_wind, 1);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.active_player_slice(), &[0]);
    assert_eq!(state.last_discard, None);
}

#[test]
fn step_returns_immediately_when_game_is_done() {
    let mut state = test_state(true);
    state.is_done = true;
    state.last_error = Some("existing error".to_string());
    state.current_player = 2;
    state.phase = Phase::WaitResponse;
    state.last_discard = Some((1, 48));

    let mut actions = std::collections::HashMap::new();
    actions.insert(
        0,
        Action::new(ActionType::Discard, Some(parsed_tile("1p")), &[], Some(0)),
    );

    state.step(&actions);

    assert!(state.is_done);
    assert_eq!(state.last_error.as_deref(), Some("existing error"));
    assert_eq!(state.current_player, 2);
    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.last_discard, Some((1, 48)));
}

#[test]
fn step_initializes_pending_round_before_validating_actions() {
    let mut state = test_state(true);
    state.oya = 2;
    state.honba = 1;
    state.round_wind = 0;
    state.phase = Phase::WaitResponse;
    state.needs_initialize_next_round = true;
    state.pending_oya_won = false;
    state.pending_is_draw = true;
    state.last_discard = Some((1, 48));

    let mut actions = std::collections::HashMap::new();
    actions.insert(2, Action::new(ActionType::Ron, None, &[], Some(2)));

    state.step(&actions);

    assert!(!state.needs_initialize_next_round);
    assert_eq!(state.oya, 0);
    assert_eq!(state.honba, 2);
    assert_eq!(state.round_wind, 1);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.active_player_slice(), &[0]);
    assert_eq!(state.last_discard, None);
    assert!(state.last_error.is_none());
}

#[test]
fn step_rejects_illegal_action_and_records_last_error() {
    let mut state = test_state(true);
    state.oya = 0;
    state.current_player = 1;
    state.phase = Phase::WaitResponse;
    state.last_discard = Some((2, 52));

    let mut actions = std::collections::HashMap::new();
    actions.insert(
        1,
        Action::new(ActionType::Discard, Some(parsed_tile("1p")), &[], Some(1)),
    );

    state.step(&actions);

    assert_eq!(
        state.last_error.as_deref(),
        Some("Error: Illegal Action by Player 1")
    );
    assert_ne!(state.players[1].score, 35_000);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert_eq!(state.last_discard, None);
}

#[test]
fn replay_ankan_matcher_accepts_same_tile_class_with_different_copy_ids() {
    let legal = Action::new(ActionType::Ankan, Some(16), &[16, 17, 18, 19], Some(0));
    let replay = Action::new(ActionType::Ankan, Some(17), &[17, 17, 17, 17], Some(0));

    assert!(GameState3P::replay_action_matches_legal(&legal, &replay));
}

#[test]
fn replay_kakan_matcher_accepts_same_tile_class_with_different_copy_ids() {
    let legal = Action::new(ActionType::Kakan, Some(16), &[16, 17, 18], Some(0));
    let replay = Action::new(ActionType::Kakan, Some(17), &[], Some(0));

    assert!(GameState3P::replay_action_matches_legal(&legal, &replay));
}

#[test]
fn replay_kan_matcher_rejects_different_tile_classes() {
    let legal = Action::new(ActionType::Ankan, Some(16), &[16, 17, 18, 19], Some(0));
    let replay = Action::new(ActionType::Ankan, Some(20), &[20, 20, 20, 20], Some(0));

    assert!(!GameState3P::replay_action_matches_legal(&legal, &replay));
}

#[test]
fn replay_matcher_accepts_tileless_special_actions() {
    let legal = Action::new(ActionType::Kita, Some(120), &[], Some(0));
    let replay = Action::new(ActionType::Kita, None, &[], Some(0));

    assert!(GameState3P::replay_action_matches_legal(&legal, &replay));
}

#[test]
fn replay_matcher_accepts_matching_kan_consumes_even_without_matching_tiles() {
    let legal = Action::new(ActionType::Kakan, Some(16), &[16, 17, 18], Some(0));
    let replay = Action::new(ActionType::Kakan, Some(52), &[16, 17, 18], Some(0));

    assert!(GameState3P::replay_action_matches_legal(&legal, &replay));
}

#[test]
fn replay_matcher_rejects_tileless_non_contextual_actions_but_allows_ron_in_sanma() {
    let legal_ron = Action::new(ActionType::Ron, Some(16), &[], Some(0));
    let replay_ron = Action::new(ActionType::Ron, None, &[], Some(0));
    assert!(GameState3P::replay_action_matches_legal(
        &legal_ron,
        &replay_ron
    ));

    let legal_chi = Action::new(ActionType::Chi, Some(16), &[12, 20], Some(0));
    let replay_chi = Action::new(ActionType::Chi, None, &[12, 20], Some(0));
    assert!(!GameState3P::replay_action_matches_legal(
        &legal_chi,
        &replay_chi
    ));

    let legal_discard = Action::new(ActionType::Discard, Some(16), &[], Some(0));
    let replay_discard = Action::new(ActionType::Discard, None, &[], Some(0));
    assert!(!GameState3P::replay_action_matches_legal(
        &legal_discard,
        &replay_discard
    ));
}

#[test]
fn replay_matcher_rejects_consume_mismatches_for_non_kan_calls_in_sanma() {
    let legal_pon = Action::new(ActionType::Pon, Some(48), &[49, 50], Some(1));
    let replay_pon = Action::new(ActionType::Pon, Some(48), &[49, 51], Some(1));

    assert!(!GameState3P::replay_action_matches_legal(
        &legal_pon,
        &replay_pon
    ));
}

#[test]
fn initialize_next_round_returns_immediately_when_game_already_done() {
    let mut state = test_state(true);
    state.is_done = true;
    state.oya = 2;
    state.honba = 1;
    state.round_wind = 1;

    state._initialize_next_round(false, false);

    assert!(state.is_done);
    assert_eq!(state.oya, 2);
    assert_eq!(state.honba, 1);
    assert_eq!(state.round_wind, 1);
}

#[test]
fn initialize_next_round_ends_game_immediately_when_any_player_is_negative() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players[0].score = -100;
    state.players[1].score = 20_000;
    state.players[2].score = 40_100;
    state.oya = 1;
    state.honba = 2;
    state.round_wind = 0;

    state._initialize_next_round(true, false);

    assert!(state.is_done);
    assert_eq!(state.oya, 1);
    assert_eq!(state.honba, 2);
    assert_eq!(state.round_wind, 0);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));
    assert!(!state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
}

#[test]
fn initialize_next_round_east_mode_continues_before_south_then_ends_at_south_with_30000() {
    let mut continue_state = test_state_with_mode(4, false);
    continue_state.mjai_log.clear();
    continue_state.mjai_log_per_player = Default::default();
    continue_state.players[0].score = 29_000;
    continue_state.players[1].score = 28_000;
    continue_state.players[2].score = 27_000;
    continue_state.oya = 1;
    continue_state.honba = 3;
    continue_state.round_wind = 0;

    continue_state._initialize_next_round(false, false);

    assert!(!continue_state.is_done);
    assert_eq!(continue_state.oya, 2);
    assert_eq!(continue_state.honba, 0);
    assert_eq!(continue_state.round_wind, 0);
    assert_eq!(continue_state.current_player, 2);
    assert_eq!(continue_state.phase, Phase::WaitAct);
    assert_eq!(continue_state.active_player_slice(), &[2]);
    assert!(continue_state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
    assert!(!continue_state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));

    let mut end_state = test_state_with_mode(4, false);
    end_state.mjai_log.clear();
    end_state.mjai_log_per_player = Default::default();
    end_state.players[0].score = 30_000;
    end_state.players[1].score = 29_000;
    end_state.players[2].score = 28_000;
    end_state.oya = 2;
    end_state.honba = 1;
    end_state.round_wind = 0;

    end_state._initialize_next_round(false, false);

    assert!(end_state.is_done);
    assert!(end_state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));
    assert!(!end_state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
}

#[test]
fn initialize_next_round_single_mode_always_ends_after_current_hand() {
    let mut state = test_state_with_mode(3, false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players[0].score = 26_000;
    state.players[1].score = 35_000;
    state.players[2].score = 44_000;
    state.oya = 0;
    state.honba = 1;
    state.round_wind = 0;

    state._initialize_next_round(true, true);

    assert!(state.is_done);
    assert_eq!(state.oya, 0);
    assert_eq!(state.honba, 1);
    assert_eq!(state.round_wind, 0);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));
    assert!(!state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
}

#[test]
fn deal_next_exhaustive_draw_without_nagashi_keeps_scores_even_and_renchan_depends_on_oya_tenpai() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.oya = 1;
    state.current_player = 2;
    state.honba = 0;
    state.round_wind = 0;
    state.wall.tile_count = 14;
    state.wall.draw_cursor = 0;
    state.players.iter_mut().for_each(|player| {
        player.nagashi_eligible = false;
        player.score = 35_000;
        player.score_delta = 0;
        player.hand = [0; 14];
        player.hand_len = 0;
        player.melds = [Meld::default(); 4];
        player.meld_count = 0;
    });

    state.players[0].hand[..13]
        .copy_from_slice(&[0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 110, 112]);
    state.players[0].hand_len = 13;
    state.players[1].hand[..13]
        .copy_from_slice(&[0, 5, 9, 36, 41, 45, 72, 77, 81, 108, 112, 116, 120]);
    state.players[1].hand_len = 13;
    state.players[2].hand[..13]
        .copy_from_slice(&[1, 6, 10, 37, 42, 46, 73, 78, 82, 109, 113, 117, 121]);
    state.players[2].hand_len = 13;

    state._deal_next();

    assert_eq!(state.players[0].score, 35_000);
    assert_eq!(state.players[1].score, 35_000);
    assert_eq!(state.players[2].score, 35_000);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"ryukyoku\"")));
    assert_eq!(state.oya, 2);
    assert_eq!(state.honba, 1);
    assert_eq!(state.round_wind, 0);
}

#[test]
fn initialize_next_round_half_mode_continues_into_west_when_no_one_has_30000() {
    let mut state = test_state_with_mode(5, false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players[0].score = 29_000;
    state.players[1].score = 28_000;
    state.players[2].score = 27_000;
    state.oya = 2;
    state.honba = 1;
    state.round_wind = 1;

    state._initialize_next_round(false, true);

    assert!(!state.is_done);
    assert_eq!(state.oya, 0);
    assert_eq!(state.honba, 2);
    assert_eq!(state.round_wind, 2);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
    assert!(!state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));
}

#[test]
fn initialize_next_round_half_mode_stays_in_south_before_west_even_with_30000_leader() {
    let mut state = test_state_with_mode(5, false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players[0].score = 31_000;
    state.players[1].score = 28_000;
    state.players[2].score = 26_000;
    state.oya = 1;
    state.honba = 2;
    state.round_wind = 1;

    state._initialize_next_round(false, false);

    assert!(!state.is_done);
    assert_eq!(state.oya, 2);
    assert_eq!(state.honba, 0);
    assert_eq!(state.round_wind, 1);
    assert_eq!(state.current_player, 2);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[2]);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
    assert!(!state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));
}

#[test]
fn initialize_next_round_half_mode_ends_once_west_wraps_past_limit_without_30000_leader() {
    let mut state = test_state_with_mode(5, false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players[0].score = 29_000;
    state.players[1].score = 28_000;
    state.players[2].score = 27_000;
    state.oya = 2;
    state.honba = 0;
    state.round_wind = 2;

    state._initialize_next_round(false, false);

    assert!(state.is_done);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));
    assert!(!state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_kyoku\"")));
}

#[test]
fn process_end_game_is_idempotent_for_done_flag_and_logging_shape() {
    let mut logged = test_state(false);
    logged.mjai_log.clear();
    logged.mjai_log_per_player = Default::default();

    logged._process_end_game();
    logged._process_end_game();

    assert!(logged.is_done);
    assert_eq!(logged.mjai_log.len(), 2);
    assert!(logged
        .mjai_log
        .iter()
        .all(|event| event.contains("\"type\":\"end_game\"")));

    let mut silent = test_state(true);
    silent._process_end_game();
    silent._process_end_game();
    assert!(silent.is_done);
    assert!(silent.mjai_log.is_empty());
}

#[test]
fn replay_matcher_rejects_mismatched_action_kinds_and_tiles() {
    let discard = Action::new(ActionType::Discard, Some(16), &[], Some(0));
    let riichi = Action::new(ActionType::Riichi, Some(16), &[], Some(0));
    assert!(!GameState3P::replay_action_matches_legal(&discard, &riichi));

    let legal_ron = Action::new(ActionType::Ron, Some(16), &[], Some(0));
    let replay_with_wrong_tile = Action::new(ActionType::Ron, Some(52), &[], Some(0));
    assert!(!GameState3P::replay_action_matches_legal(
        &legal_ron,
        &replay_with_wrong_tile
    ));
}

#[test]
fn accept_riichi_is_noop_without_pending_player_and_logs_when_present() {
    let mut silent = test_state(true);
    let score_before = silent.players[0].score;
    silent._accept_riichi();
    assert_eq!(silent.players[0].score, score_before);
    assert_eq!(silent.riichi_sticks, 0);

    let mut logged = test_state(false);
    logged.mjai_log.clear();
    logged.mjai_log_per_player = Default::default();
    logged.riichi_pending_acceptance = Some(1);
    logged._accept_riichi();
    assert_eq!(logged.players[1].score, 34_000);
    assert_eq!(logged.players[1].score_delta, -1000);
    assert_eq!(logged.riichi_sticks, 1);
    assert!(logged.players[1].riichi_declared);
    assert!(logged.players[1].ippatsu_cycle);
    assert!(logged.riichi_pending_acceptance.is_none());
    assert!(logged
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"reach_accepted\"")));
}

#[test]
fn replay_observation_allows_pass_for_active_response_player_and_restores_state() {
    let mut state = test_state(true);
    state.phase = Phase::WaitResponse;
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Ron, Some(48), &[], Some(1));

    let obs = state
        .get_observation_for_replay(
            1,
            &Action::new(ActionType::Pass, None, &[], Some(1)),
            "{\"type\":\"none\"}",
        )
        .expect("pass should be exposed as legal during response replay");

    assert!(obs
        .legal_actions_method()
        .iter()
        .any(|action| action.action_type == ActionType::Pass));
    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.active_players, [1, 0, 0, 0]);
    assert_eq!(state.active_player_count, 1);
    assert_eq!(state.current_claim_counts[1], 1);
}

#[test]
fn replay_observation_exposes_call_action_for_response_player_and_restores_claims() {
    let mut state = test_state(true);
    state.phase = Phase::WaitResponse;
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Pon, Some(48), &[49, 50], Some(1));

    let obs = state
        .get_observation_for_replay(
            1,
            &Action::new(ActionType::Pon, Some(48), &[49, 50], Some(1)),
            "{\"type\":\"pon\"}",
        )
        .expect("pon should be exposed as legal during response replay");

    assert!(obs
        .legal_actions_method()
        .iter()
        .any(|action| action.action_type == ActionType::Pon));
    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.active_players, [1, 0, 0, 0]);
    assert_eq!(state.current_claim_counts[1], 1);
}

#[test]
fn replay_observation_restores_riichi_after_failed_discard_retry() {
    let mut state = test_state(true);
    let drawn_tile = state
        .drawn_tile
        .expect("test state should start with a drawn tile");
    let invalid_tile = (0..108u8)
        .find(|&tile| tile != drawn_tile && !state.players[0].hand_slice().contains(&tile))
        .expect("there should be a tile outside player 0's hand");
    state.players[0].riichi_declared = true;

    let err = state
        .get_observation_for_replay(
            0,
            &Action::new(ActionType::Discard, Some(invalid_tile), &[], Some(0)),
            "{\"type\":\"dahai\"}",
        )
        .expect_err("illegal replay discard should stay illegal after riichi retry");

    assert!(matches!(err, RiichiError::InvalidState { .. }));
    assert!(state.players[0].riichi_declared);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
}

#[test]
fn wait_response_marks_missed_ron_and_riichi_when_player_passes_on_win() {
    let mut state = test_state(true);
    state.phase = Phase::WaitResponse;
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.current_claim_counts[1] = 2;
    state.current_claims[1][0] = Action::new(ActionType::Ron, Some(48), &[], Some(1));
    state.current_claims[1][1] = Action::new(ActionType::Pass, None, &[], Some(1));
    state.players[1].riichi_declared = true;

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Pass, None, &[], Some(1))),
        None,
    ]);

    assert!(state.players[1].missed_agari_doujun);
    assert!(state.players[1].missed_agari_riichi);
}

#[test]
fn wait_response_marks_only_doujun_when_non_riichi_player_passes_on_win() {
    let mut state = test_state(true);
    state.phase = Phase::WaitResponse;
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.current_claim_counts[1] = 2;
    state.current_claims[1][0] = Action::new(ActionType::Ron, Some(48), &[], Some(1));
    state.current_claims[1][1] = Action::new(ActionType::Pass, None, &[], Some(1));
    state.players[1].riichi_declared = false;

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Pass, None, &[], Some(1))),
        None,
    ]);

    assert!(state.players[1].missed_agari_doujun);
    assert!(!state.players[1].missed_agari_riichi);
}

#[test]
fn wait_response_all_pass_accepts_riichi_and_advances_turn_out_of_first_cycle() {
    let mut state = test_state(true);
    state.phase = Phase::WaitResponse;
    state.current_player = 2;
    state.turn_count = NP as u32 - 1;
    state.is_first_turn = true;
    state.riichi_pending_acceptance = Some(1);
    if state.players[0].hand_len > 0 {
        state.players[0].hand_len -= 1;
    }
    state.drawn_tile = None;

    state._handle_wait_response(&[None, None, None]);

    assert!(state.riichi_pending_acceptance.is_none());
    assert_eq!(state.players[1].score, 34_000);
    assert!(state.players[1].riichi_declared);
    assert!(state.players[1].ippatsu_cycle);
    assert_eq!(state.riichi_sticks, 1);
    assert_eq!(state.turn_count, NP as u32);
    assert!(!state.is_first_turn);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state.drawn_tile.is_some());
}

#[test]
fn wait_response_all_pass_before_first_cycle_keeps_first_turn_true_and_clears_claims() {
    let mut state = test_state(true);
    state.phase = Phase::WaitResponse;
    state.current_player = 0;
    state.turn_count = 0;
    state.is_first_turn = true;
    state.active_players = [1, 2, 0, 0];
    state.active_player_count = 2;
    state.current_claim_counts[1] = 1;
    state.current_claim_counts[2] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Pass, None, &[], Some(1));
    state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));
    if state.players[1].hand_len > 0 {
        state.players[1].hand_len -= 1;
    }
    state.drawn_tile = None;

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Pass, None, &[], Some(1))),
        Some(Action::new(ActionType::Pass, None, &[], Some(2))),
    ]);

    assert_eq!(state.turn_count, 1);
    assert!(state.is_first_turn);
    assert_eq!(state.current_player, 1);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[1]);
    assert_eq!(state.current_claim_counts, [0, 0, 0]);
    assert!(state.drawn_tile.is_some());
}

#[test]
fn wait_response_resolves_pending_kakan_after_all_players_pass() {
    let mut state = test_state(true);
    state.phase = Phase::WaitResponse;
    state.current_player = 0;
    state.last_discard = Some((0, 48));
    state.active_players = [1, 2, 0, 0];
    state.active_player_count = 2;
    state.current_claim_counts[1] = 1;
    state.current_claim_counts[2] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Pass, None, &[], Some(1));
    state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));
    state.pending_kan = Some((
        0,
        Action::new(ActionType::Kakan, Some(16), &[16, 17, 18], Some(0)),
    ));
    if state.players[0].hand_len > 0 {
        state.players[0].hand_len -= 1;
    }
    state.drawn_tile = None;
    let rinshan_before = state.wall.rinshan_draw_count;
    let pending_before = state.wall.pending_kan_dora_count;

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Pass, None, &[], Some(1))),
        Some(Action::new(ActionType::Pass, None, &[], Some(2))),
    ]);

    assert!(state.pending_kan.is_none());
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state.drawn_tile.is_some());
    assert!(state.is_rinshan_flag);
    assert_eq!(state.wall.rinshan_draw_count, rinshan_before + 1);
    assert_eq!(state.wall.pending_kan_dora_count, pending_before + 1);
}

#[test]
fn resolve_discard_sets_riichi_pending_and_logs_after_tedashi() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.current_player = 0;
    let drawn = state
        .drawn_tile
        .expect("test state should start with a drawn tile");
    state.players[0].riichi_stage = true;
    state.players[0].nagashi_eligible = true;

    state._resolve_discard(0, drawn, false);

    assert_eq!(state.last_discard, Some((0, drawn)));
    assert!(state.players[0].riichi_declared);
    assert_eq!(state.last_tedashis[0], Some(drawn));
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"dahai\"")));
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"reach_accepted\"")));
}

#[test]
fn replay_observation_temporarily_injects_daiminkan_and_restores_state_on_success() {
    let mut state = test_state(true);
    state.phase = Phase::WaitResponse;
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Pass, None, &[], Some(1));

    let obs = state
        .get_observation_for_replay(
            1,
            &Action::new(ActionType::Daiminkan, Some(48), &[48, 49, 50], Some(1)),
            "{\"type\":\"daiminkan\"}",
        )
        .expect("daiminkan replay action should be exposed as legal");

    assert!(obs
        .legal_actions_method()
        .iter()
        .any(|action| action.action_type == ActionType::Daiminkan));
    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.active_players, [1, 0, 0, 0]);
    assert_eq!(state.active_player_count, 1);
    assert_eq!(state.current_claim_counts[1], 1);
    assert_eq!(state.current_claims[1][0].action_type, ActionType::Pass);
}

#[test]
fn abortive_draw_disabled_rules_stay_false_in_sanma() {
    let mut four_winds_like = test_state(true);
    for player in &mut four_winds_like.players {
        player.discards[0] = 108;
        player.discard_len = 1;
        player.meld_count = 0;
    }
    assert!(!four_winds_like.check_abortive_draw());

    let mut all_riichi = test_state(true);
    all_riichi
        .players
        .iter_mut()
        .for_each(|player| player.riichi_declared = true);
    assert!(!all_riichi.check_abortive_draw());
}

#[test]
fn initialize_round_without_oya_draw_leaves_needs_tsumo_true() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();

    state._initialize_round(0, 0, 0, 0, Some((0..39).collect()), Some(vec![35_000; 3]));

    assert!(state.drawn_tile.is_none());
    assert!(state.needs_tsumo);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"start_kyoku\"")));
    assert!(!state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"tsumo\"")));
}

#[test]
fn wrapper_methods_delegate_into_event_and_log_handlers() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.apply_mjai_event(MjaiEvent::Reach { actor: 0 });
    assert!(state.players[0].riichi_stage);

    let mut replay = test_state(false);
    replay.mjai_log.clear();
    replay.mjai_log_per_player = Default::default();
    let action = LogAction::DiscardTile {
        seat: 0,
        tile: replay
            .drawn_tile
            .expect("test state should start with a drawn tile"),
        is_liqi: false,
        is_wliqi: false,
        doras: None,
    };
    replay.apply_log_action(&action);
    assert!(replay.last_discard.is_some());
}

#[test]
fn wait_response_resolves_pending_kita_after_all_players_pass_and_breaks_ippatsu() {
    let mut state = test_state(true);
    let kita_tile = state
        .drawn_tile
        .expect("test state should start with a drawn tile");
    let kita_idx = state.players[0]
        .hand_slice()
        .iter()
        .position(|&tile| tile == kita_tile)
        .expect("drawn tile should still be present in hand");
    state.players[0].remove_hand(kita_idx);
    state.phase = Phase::WaitResponse;
    state.current_player = 0;
    state.last_discard = Some((0, kita_tile));
    state.active_players = [1, 2, 0, 0];
    state.active_player_count = 2;
    state.current_claim_counts[1] = 1;
    state.current_claim_counts[2] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Pass, None, &[], Some(1));
    state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));
    state.pending_kan = Some((
        0,
        Action::new(ActionType::Kita, Some(kita_tile), &[], Some(0)),
    ));
    state
        .players
        .iter_mut()
        .for_each(|player| player.ippatsu_cycle = true);
    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Pass, None, &[], Some(1))),
        Some(Action::new(ActionType::Pass, None, &[], Some(2))),
    ]);

    assert!(state.pending_kan.is_none());
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state.drawn_tile.is_some());
    assert!(state.is_rinshan_flag);
    assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
}

#[test]
fn wait_response_pon_claim_sets_forbidden_and_clears_missed_agari_without_rinshan() {
    let mut state = test_state(true);
    state.phase = Phase::WaitResponse;
    state.current_player = 0;
    state.last_discard = Some((0, 48));
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Pon, Some(48), &[49, 50], Some(1));
    state.players[1].hand[..2].copy_from_slice(&[49, 50]);
    state.players[1].hand_len = 2;
    state.players[1].missed_agari_doujun = true;
    state.players[0].nagashi_eligible = true;
    state
        .players
        .iter_mut()
        .for_each(|player| player.ippatsu_cycle = true);

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Pon, Some(48), &[49, 50], Some(1))),
        None,
    ]);

    assert_eq!(state.current_player, 1);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[1]);
    assert_eq!(state.players[1].meld_count, 1);
    assert_eq!(state.players[1].melds[0].meld_type, MeldType::Pon);
    assert!(!state.players[1].missed_agari_doujun);
    assert!(!state.players[0].nagashi_eligible);
    assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
    assert!(state.players[1].forbidden_slice().contains(&48));
    assert!(!state.needs_tsumo);
    assert!(state.drawn_tile.is_none());
}

#[test]
fn wait_response_daiminkan_claim_accepts_riichi_and_resolves_kan_immediately() {
    let mut state = test_state(true);
    state.phase = Phase::WaitResponse;
    state.current_player = 0;
    state.last_discard = Some((0, 48));
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] =
        Action::new(ActionType::Daiminkan, Some(48), &[49, 50, 51], Some(1));
    state.players[1].hand[..3].copy_from_slice(&[49, 50, 51]);
    state.players[1].hand_len = 3;
    state.riichi_pending_acceptance = Some(2);
    state
        .players
        .iter_mut()
        .for_each(|player| player.ippatsu_cycle = true);
    let rinshan_before = state.wall.rinshan_draw_count;
    let pending_before = state.wall.pending_kan_dora_count;

    state._handle_wait_response(&[
        None,
        Some(Action::new(
            ActionType::Daiminkan,
            Some(48),
            &[49, 50, 51],
            Some(1),
        )),
        None,
    ]);

    assert_eq!(state.players[2].score, 34_000);
    assert!(state.riichi_pending_acceptance.is_none());
    assert_eq!(state.riichi_sticks, 1);
    assert_eq!(state.current_player, 1);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[1]);
    assert!(state.is_rinshan_flag);
    assert!(state.drawn_tile.is_some());
    assert_eq!(state.wall.rinshan_draw_count, rinshan_before + 1);
    assert_eq!(state.wall.pending_kan_dora_count, pending_before + 1);
    assert_eq!(state.players[1].meld_count, 1);
    assert_eq!(state.players[1].melds[0].meld_type, MeldType::Daiminkan);
    assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
}

#[test]
fn claim_and_active_player_helpers_round_trip_state_in_sanma() {
    let mut state = test_state(false);
    let ron = Action::new(ActionType::Ron, Some(48), &[], Some(1));
    let pass = Action::new(ActionType::Pass, None, &[], Some(1));

    state.clear_active_players();
    assert!(state.active_player_slice().is_empty());

    state.set_single_active_player(2);
    assert_eq!(state.active_player_slice(), &[2]);

    state.set_active_players_from_slice(&[1, 2]);
    assert_eq!(state.active_player_slice(), &[1, 2]);

    state.push_claim(1, ron);
    state.push_claim(1, pass);
    assert_eq!(state.claims_slice(1), &[ron, pass]);

    state.set_claims_from_vec(2, &[ron]);
    assert_eq!(state.claims_slice(2), &[ron]);

    state.clear_claims();
    assert!(state.claims_slice(1).is_empty());
    assert!(state.claims_slice(2).is_empty());
}

#[test]
fn handle_ankan_without_chankan_claims_resolves_kan_immediately() {
    let mut rule = GameRule::default_tenhou_sanma();
    rule.allows_ron_on_ankan_for_kokushi_musou = true;
    let mut state = test_state(true);
    state.rule = rule;
    state.phase = Phase::WaitAct;
    state.current_player = 0;
    state.active_players = [0, 0, 0, 0];
    state.active_player_count = 1;
    state.last_discard = None;
    state.players[0].hand = [0; 14];
    state.players[0].hand_len = 0;
    for tile in [0u8, 1, 2, 3] {
        state.players[0].push_hand(tile);
    }
    state.players[1].hand = [0; 14];
    state.players[1].hand_len = 0;
    state.players[2].hand = [0; 14];
    state.players[2].hand_len = 0;
    let rinshan_before = state.wall.rinshan_draw_count;

    state._handle_ankan(
        0,
        Action::new(ActionType::Ankan, Some(0), &[0, 1, 2, 3], Some(0)),
    );

    assert!(state.pending_kan.is_none());
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state.drawn_tile.is_some());
    assert!(state.is_rinshan_flag);
    assert_eq!(state.wall.rinshan_draw_count, rinshan_before + 1);
    assert_eq!(state.players[0].meld_count, 1);
    assert_eq!(state.players[0].melds[0].meld_type, MeldType::Ankan);
    assert!(state.current_claim_counts.iter().all(|count| *count == 0));
}

#[test]
fn wait_response_ron_with_pao_liability_only_splits_between_pao_and_discarder() {
    let mut rule = GameRule::default_mjsoul_sanma();
    rule.yakuman_pao_is_liability_only = true;
    let mut state = direct_state(rule);
    let win_tile = parsed_tile("4z");
    state.oya = 2;
    state.current_player = 0;
    state.honba = 1;
    state.riichi_sticks = 1;
    state.phase = Phase::WaitResponse;
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.last_discard = Some((0, win_tile));
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Ron, Some(win_tile), &[], Some(1));
    set_closed_hand(&mut state, 1, "11122233344455z");
    state.players[1].pao_insert(50, 2);

    let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(
        state.players[1].hand_slice(),
        state.players[1].melds_slice(),
    );
    let result = calc.calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: false,
            player_wind: Wind::West,
            round_wind: Wind::East,
            honba: 1,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    assert!(result.is_win);
    assert!(result.yakuman);
    assert!(result.yaku_slice().contains(&50));
    let pao_yakuman_val = if state.rule.is_daisuushii_double {
        2
    } else {
        1
    };
    let expected_pao_amt = (pao_yakuman_val * 32_000) / 2 + 200;

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(1))),
        None,
    ]);

    assert_eq!(state.players[1].score_delta, result.ron_agari as i32 + 1000);
    assert_eq!(state.players[2].score_delta, -expected_pao_amt);
    assert_eq!(
        state.players[0].score_delta,
        -(result.ron_agari as i32 - expected_pao_amt)
    );
    assert_eq!(state.riichi_sticks, 0);
    assert_eq!(
        state.win_results[1].as_ref().and_then(|res| res.pao_payer),
        Some(2)
    );
}

#[test]
fn wait_response_multi_ron_gives_honba_and_riichi_to_first_winner_only() {
    let mut state = direct_state(GameRule::default_mjsoul_sanma());
    let win_tile = parsed_tile("4z");
    state.oya = 0;
    state.current_player = 0;
    state.honba = 2;
    state.riichi_sticks = 2;
    state.phase = Phase::WaitResponse;
    state.active_players = [2, 1, 0, 0];
    state.active_player_count = 2;
    state.last_discard = Some((0, win_tile));
    state.current_claim_counts[1] = 1;
    state.current_claim_counts[2] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Ron, Some(win_tile), &[], Some(1));
    state.current_claims[2][0] = Action::new(ActionType::Ron, Some(win_tile), &[], Some(2));
    set_closed_hand(&mut state, 1, "123p123s789s555z4z");
    set_closed_hand(&mut state, 2, "123p123s789s555z4z");

    let winner_one = crate::hand_evaluator_3p::HandEvaluator3P::new(
        state.players[1].hand_slice(),
        state.players[1].melds_slice(),
    )
    .calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: false,
            player_wind: Wind::South,
            round_wind: Wind::East,
            honba: 2,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    let winner_two = crate::hand_evaluator_3p::HandEvaluator3P::new(
        state.players[2].hand_slice(),
        state.players[2].melds_slice(),
    )
    .calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: false,
            player_wind: Wind::West,
            round_wind: Wind::East,
            honba: 0,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    assert!(winner_one.is_win);
    assert!(winner_two.is_win);

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(1))),
        Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(2))),
    ]);

    assert_eq!(
        state.players[1].score_delta,
        winner_one.ron_agari as i32 + 2000
    );
    assert_eq!(state.players[2].score_delta, winner_two.ron_agari as i32);
    assert_eq!(
        state.players[0].score_delta,
        -(winner_one.ron_agari as i32 + winner_two.ron_agari as i32)
    );
    assert_eq!(state.riichi_sticks, 0);
}

#[test]
fn handle_tsumo_caps_daisuushii_when_double_is_disabled() {
    let mut disabled_rule = GameRule::default_mjsoul_sanma();
    disabled_rule.is_daisuushii_double = false;
    let mut disabled = direct_state(disabled_rule);
    let mut enabled = direct_state(GameRule::default_mjsoul_sanma());
    let win_tile = parsed_tile("5z");
    for state in [&mut disabled, &mut enabled] {
        state.oya = 2;
        state.current_player = 1;
        state.phase = Phase::WaitAct;
        state.active_players = [0, 1, 0, 0];
        state.active_player_count = 1;
        state.drawn_tile = Some(win_tile);
        set_closed_hand(state, 1, "11122233344455z");
    }

    disabled._handle_tsumo(1);
    enabled._handle_tsumo(1);

    let disabled_result = disabled.win_results[1]
        .as_ref()
        .expect("disabled-rule tsumo win result should exist");
    let enabled_result = enabled.win_results[1]
        .as_ref()
        .expect("enabled-rule tsumo win result should exist");
    assert!(disabled_result.is_win);
    assert!(enabled_result.is_win);
    assert!(disabled_result.yakuman);
    assert!(enabled_result.yakuman);
    assert!(disabled_result.yaku_slice().contains(&50));
    assert!(enabled_result.yaku_slice().contains(&50));
    assert!(disabled_result.han < enabled_result.han);
    assert!(disabled_result.tsumo_agari_oya <= enabled_result.tsumo_agari_oya);
    assert!(disabled_result.tsumo_agari_ko <= enabled_result.tsumo_agari_ko);
    assert_eq!(disabled_result.pao_payer, None);
    assert_eq!(enabled_result.pao_payer, None);
    assert_eq!(
        disabled.players[0].score_delta
            + disabled.players[1].score_delta
            + disabled.players[2].score_delta,
        0
    );
    assert_eq!(
        enabled.players[0].score_delta
            + enabled.players[1].score_delta
            + enabled.players[2].score_delta,
        0
    );
}

#[test]
fn wait_response_caps_daisuushii_ron_when_double_is_disabled() {
    let mut disabled_rule = GameRule::default_mjsoul_sanma();
    disabled_rule.is_daisuushii_double = false;
    let mut disabled = direct_state(disabled_rule);
    let mut enabled = direct_state(GameRule::default_mjsoul_sanma());
    let win_tile = parsed_tile("5z");

    for state in [&mut disabled, &mut enabled] {
        state.oya = 2;
        state.current_player = 0;
        state.phase = Phase::WaitResponse;
        state.active_players = [1, 0, 0, 0];
        state.active_player_count = 1;
        state.last_discard = Some((0, win_tile));
        state.current_claim_counts[1] = 1;
        state.current_claims[1][0] = Action::new(ActionType::Ron, Some(win_tile), &[], Some(1));
        set_closed_hand(state, 1, "11122233344455z");
    }

    disabled._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(1))),
        None,
    ]);
    enabled._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(1))),
        None,
    ]);

    let disabled_result = disabled.win_results[1]
        .as_ref()
        .expect("disabled-rule ron win result should exist");
    let enabled_result = enabled.win_results[1]
        .as_ref()
        .expect("enabled-rule ron win result should exist");
    assert!(disabled_result.is_win);
    assert!(enabled_result.is_win);
    assert!(disabled_result.yakuman);
    assert!(enabled_result.yakuman);
    assert!(disabled_result.yaku_slice().contains(&50));
    assert!(enabled_result.yaku_slice().contains(&50));
    assert!(disabled_result.han <= enabled_result.han);

    assert_eq!(
        disabled.players[1].score_delta,
        disabled_result.ron_agari as i32
    );
    assert_eq!(
        disabled.players[0].score_delta,
        -(disabled_result.ron_agari as i32)
    );
    assert_eq!(
        enabled.players[1].score_delta,
        enabled_result.ron_agari as i32
    );
    assert_eq!(
        enabled.players[0].score_delta,
        -(enabled_result.ron_agari as i32)
    );
    assert!(disabled.players[1].score_delta <= enabled.players[1].score_delta);
    assert_eq!(
        disabled.players[0].score_delta
            + disabled.players[1].score_delta
            + disabled.players[2].score_delta,
        0
    );
    assert_eq!(
        enabled.players[0].score_delta
            + enabled.players[1].score_delta
            + enabled.players[2].score_delta,
        0
    );
}

#[test]
fn handle_tsumo_with_pao_liability_only_charges_pao_and_non_pao_shares() {
    let mut rule = GameRule::default_mjsoul_sanma();
    rule.yakuman_pao_is_liability_only = true;
    let mut state = direct_state(rule);
    let win_tile = parsed_tile("5z");
    state.oya = 2;
    state.current_player = 1;
    state.honba = 1;
    state.riichi_sticks = 1;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 1, 0, 0];
    state.active_player_count = 1;
    state.drawn_tile = Some(win_tile);
    set_closed_hand(&mut state, 1, "11122233344455z");
    state.players[1].pao_insert(50, 0);

    let result = crate::hand_evaluator_3p::HandEvaluator3P::new(
        state.players[1].hand_slice(),
        state.players[1].melds_slice(),
    )
    .calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: true,
            player_wind: Wind::West,
            round_wind: Wind::East,
            honba: 1,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    assert!(result.is_win);
    assert!(result.yakuman);
    assert!(result.yaku_slice().contains(&50));

    let mut total_yakuman_val = 0i32;
    let mut pao_yakuman_val = 0i32;
    for &yid in result.yaku_slice() {
        let val = match yid {
            47 if state.rule.is_junsei_chuurenpoutou_double => 2,
            48 if state.rule.is_suuankou_tanki_double => 2,
            49 if state.rule.is_kokushi_musou_13machi_double => 2,
            50 if state.rule.is_daisuushii_double => 2,
            _ => 1,
        };
        total_yakuman_val += val;
        if state.players[1].pao_get(yid as u8).is_some() {
            pao_yakuman_val += val;
        }
    }
    let non_pao_yakuman_val = total_yakuman_val - pao_yakuman_val;
    let pao_amt = pao_yakuman_val * 24_000 + 200;
    let oya_share = non_pao_yakuman_val * 16_000;
    let ko_share = non_pao_yakuman_val * 8_000;

    state._handle_tsumo(1);

    assert_eq!(state.players[0].score_delta, -(pao_amt + ko_share));
    assert_eq!(state.players[2].score_delta, -oya_share);
    assert_eq!(
        state.players[1].score_delta,
        pao_amt + ko_share + oya_share + 1000
    );
    assert_eq!(state.riichi_sticks, 0);
    assert_eq!(
        state.win_results[1].as_ref().and_then(|res| res.pao_payer),
        Some(0)
    );
}

#[test]
fn handle_tsumo_with_full_pao_makes_liable_player_pay_everything() {
    let mut rule = GameRule::default_mjsoul_sanma();
    rule.yakuman_pao_is_liability_only = false;
    let mut state = direct_state(rule);
    let win_tile = parsed_tile("5z");
    state.oya = 2;
    state.current_player = 1;
    state.honba = 1;
    state.riichi_sticks = 1;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 1, 0, 0];
    state.active_player_count = 1;
    state.drawn_tile = Some(win_tile);
    set_closed_hand(&mut state, 1, "11122233344455z");
    state.players[1].pao_insert(50, 0);

    let result = crate::hand_evaluator_3p::HandEvaluator3P::new(
        state.players[1].hand_slice(),
        state.players[1].melds_slice(),
    )
    .calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: true,
            player_wind: Wind::West,
            round_wind: Wind::East,
            honba: 1,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    assert!(result.is_win);
    assert!(result.yakuman);
    assert!(result.yaku_slice().contains(&50));

    let total_yakuman_val: i32 = result
        .yaku_slice()
        .iter()
        .map(|&yid| match yid {
            47 if state.rule.is_junsei_chuurenpoutou_double => 2,
            48 if state.rule.is_suuankou_tanki_double => 2,
            49 if state.rule.is_kokushi_musou_13machi_double => 2,
            50 if state.rule.is_daisuushii_double => 2,
            _ => 1,
        })
        .sum();
    let full_amt = total_yakuman_val * 24_000 + 200;

    state._handle_tsumo(1);

    assert_eq!(state.players[0].score_delta, -full_amt);
    assert_eq!(state.players[2].score_delta, 0);
    assert_eq!(state.players[1].score_delta, full_amt + 1000);
    assert_eq!(state.riichi_sticks, 0);
    assert_eq!(
        state.win_results[1].as_ref().and_then(|res| res.pao_payer),
        Some(0)
    );
}

#[test]
fn handle_tsumo_for_oya_charges_both_ko_players_and_collects_riichi_sticks() {
    let mut state = direct_state(GameRule::default_mjsoul_sanma());
    let win_tile = parsed_tile("4z");
    state.oya = 1;
    state.current_player = 1;
    state.honba = 1;
    state.riichi_sticks = 2;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 1, 0, 0];
    state.active_player_count = 1;
    state.drawn_tile = Some(win_tile);
    set_closed_hand(&mut state, 1, "123p123s789s555z4z");

    let result = crate::hand_evaluator_3p::HandEvaluator3P::new(
        state.players[1].hand_slice(),
        state.players[1].melds_slice(),
    )
    .calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: true,
            player_wind: Wind::East,
            round_wind: Wind::East,
            honba: 1,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    assert!(result.is_win);
    assert!(!result.yakuman);

    state._handle_tsumo(1);

    assert_eq!(
        state.players[0].score_delta,
        -(result.tsumo_agari_ko as i32)
    );
    assert_eq!(
        state.players[2].score_delta,
        -(result.tsumo_agari_ko as i32)
    );
    assert_eq!(
        state.players[1].score_delta,
        2 * result.tsumo_agari_ko as i32 + 2000
    );
    assert_eq!(state.riichi_sticks, 0);
    assert_eq!(state.honba, 2);
    assert_eq!(state.current_player, 1);
}

#[test]
fn handle_tsumo_for_ko_charges_oya_and_ko_different_amounts() {
    let mut state = direct_state(GameRule::default_mjsoul_sanma());
    let win_tile = parsed_tile("4z");
    state.oya = 0;
    state.current_player = 1;
    state.honba = 1;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 1, 0, 0];
    state.active_player_count = 1;
    state.drawn_tile = Some(win_tile);
    set_closed_hand(&mut state, 1, "123p123s789s555z4z");

    let result = crate::hand_evaluator_3p::HandEvaluator3P::new(
        state.players[1].hand_slice(),
        state.players[1].melds_slice(),
    )
    .calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: true,
            player_wind: Wind::South,
            round_wind: Wind::East,
            honba: 1,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    assert!(result.is_win);
    assert!(!result.yakuman);

    state._handle_tsumo(1);

    assert_eq!(
        state.players[0].score_delta,
        -(result.tsumo_agari_oya as i32)
    );
    assert_eq!(
        state.players[2].score_delta,
        -(result.tsumo_agari_ko as i32)
    );
    assert_eq!(
        state.players[1].score_delta,
        result.tsumo_agari_oya as i32 + result.tsumo_agari_ko as i32
    );
    assert_eq!(state.riichi_sticks, 0);
    assert_eq!(state.honba, 0);
    assert_eq!(state.current_player, 1);
}

#[test]
fn handle_tsumo_non_win_advances_turn_and_deals_next_tile() {
    let mut state = direct_state(GameRule::default_mjsoul_sanma());
    let attempted_tile = parsed_tile("5s");
    let next_draw = parsed_tile("7p");
    state.oya = 0;
    state.current_player = 1;
    state.honba = 2;
    state.riichi_sticks = 1;
    state.is_first_turn = true;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 1, 0, 0];
    state.active_player_count = 1;
    state.drawn_tile = Some(attempted_tile);
    state.wall.tile_count = 20;
    state.wall.draw_cursor = 0;
    state.wall.tiles[19] = next_draw;
    set_closed_hand(&mut state, 1, "123p456p789p123s45s");

    state._handle_tsumo(1);

    assert!(state.win_results.iter().all(Option::is_none));
    assert_eq!(state.current_player, 2);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[2]);
    assert_eq!(state.drawn_tile, Some(next_draw));
    assert_eq!(
        state.players[2].hand_slice().last().copied(),
        Some(next_draw)
    );
    assert_eq!(state.players[0].score_delta, 0);
    assert_eq!(state.players[1].score_delta, 0);
    assert_eq!(state.players[2].score_delta, 0);
    assert_eq!(state.wall.tile_count, 19);
    assert!(!state.needs_tsumo);
    assert_eq!(state.honba, 2);
    assert_eq!(state.riichi_sticks, 1);
    assert!(state.is_first_turn);
}

#[test]
fn handle_tsumo_riichi_logs_ura_markers_on_win() {
    let mut state = GameState3P::new(5, false, Some(7), 0, GameRule::default_mjsoul_sanma());
    for player in &mut state.players {
        player.reset_round();
        player.score = 35_000;
        player.score_delta = 0;
    }
    let win_tile = parsed_tile("5z");
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.oya = 0;
    state.current_player = 1;
    state.phase = Phase::WaitAct;
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.drawn_tile = Some(win_tile);
    state.players[1].riichi_declared = true;
    state.riichi_sticks = 1;
    state.wall.dora_indicator_count = 2;
    state.wall.ura_indicator_tiles[0] = parsed_tile("1s");
    state.wall.ura_indicator_tiles[1] = parsed_tile("2s");
    set_closed_hand(&mut state, 1, "11122233344455z");

    state._handle_tsumo(1);

    assert_eq!(state.riichi_sticks, 0);
    let hora_event = state
        .mjai_log
        .iter()
        .find(|event| event.contains("\"type\":\"hora\""))
        .expect("hora event should be logged");
    assert!(hora_event.contains("\"tsumo\":true"));
    assert!(hora_event.contains("\"ura_markers\""));
    assert!(hora_event.contains("1s"));
    assert!(hora_event.contains("2s"));
}

#[test]
fn wait_response_kita_ron_scores_without_chankan_han() {
    let mut state = direct_state(GameRule::default_mjsoul_sanma());
    let win_tile = parsed_tile("4z");
    state.oya = 2;
    state.phase = Phase::WaitResponse;
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.last_discard = Some((0, win_tile));
    state.pending_kan = Some((
        0,
        Action::new(ActionType::Kita, Some(win_tile), &[], Some(0)),
    ));
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Ron, Some(win_tile), &[], Some(1));
    set_closed_hand(&mut state, 1, "123p123s789s555z4z");

    let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(
        state.players[1].hand_slice(),
        state.players[1].melds_slice(),
    );
    let no_chankan = calc.calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: false,
            chankan: false,
            player_wind: Wind::West,
            round_wind: Wind::East,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    let with_chankan = calc.calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: false,
            chankan: true,
            player_wind: Wind::West,
            round_wind: Wind::East,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    assert!(no_chankan.is_win);
    assert!(with_chankan.han > no_chankan.han);

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(1))),
        None,
    ]);

    assert_eq!(state.players[1].score_delta, no_chankan.ron_agari as i32);
    assert_eq!(state.players[0].score_delta, -(no_chankan.ron_agari as i32));
    assert_ne!(state.players[1].score_delta, with_chankan.ron_agari as i32);
}

#[test]
fn wait_response_kakan_ron_scores_with_chankan_han() {
    let mut state = direct_state(GameRule::default_mjsoul_sanma());
    let win_tile = parsed_tile("4z");
    state.oya = 2;
    state.phase = Phase::WaitResponse;
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.last_discard = Some((0, win_tile));
    state.pending_kan = Some((
        0,
        Action::new(ActionType::Kakan, Some(win_tile), &[win_tile; 3], Some(0)),
    ));
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Ron, Some(win_tile), &[], Some(1));
    set_closed_hand(&mut state, 1, "123p123s789s555z4z");

    let calc = crate::hand_evaluator_3p::HandEvaluator3P::new(
        state.players[1].hand_slice(),
        state.players[1].melds_slice(),
    );
    let no_chankan = calc.calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: false,
            chankan: false,
            player_wind: Wind::West,
            round_wind: Wind::East,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    let with_chankan = calc.calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: false,
            chankan: true,
            player_wind: Wind::West,
            round_wind: Wind::East,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    assert!(no_chankan.is_win);
    assert!(with_chankan.is_win);
    assert!(with_chankan.han > no_chankan.han);

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(1))),
        None,
    ]);

    assert_eq!(state.players[1].score_delta, with_chankan.ron_agari as i32);
    assert_eq!(
        state.players[0].score_delta,
        -(with_chankan.ron_agari as i32)
    );
    assert_ne!(state.players[1].score_delta, no_chankan.ron_agari as i32);
}

#[test]
fn kakan_furiten_suppresses_chankan_claims_and_resolves_kan() {
    let mut state = direct_state(GameRule::default_tenhou_sanma());
    let north_tiles = [
        parsed_tile("4z"),
        parsed_tile("4z"),
        parsed_tile("4z"),
        parsed_tile("4z"),
    ];
    state.current_player = 0;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 0, 0, 0];
    state.active_player_count = 1;
    state.wall.tiles[0] = parsed_tile("2p");
    state.players[0].melds[0] = Meld::new(
        MeldType::Pon,
        &north_tiles[..3],
        true,
        1,
        Some(north_tiles[0]),
    );
    state.players[0].meld_count = 1;
    state.players[0].hand[0] = north_tiles[3];
    state.players[0].hand_len = 1;
    set_closed_hand(&mut state, 1, "123p123s789s555z4z");
    state.players[1].discards[0] = north_tiles[0];
    state.players[1].discard_len = 1;

    state._handle_kakan(
        0,
        Action::new(
            ActionType::Kakan,
            Some(north_tiles[3]),
            &north_tiles[..3],
            Some(0),
        ),
    );

    assert!(state.pending_kan.is_none());
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state.drawn_tile.is_some());
    assert!(state.is_rinshan_flag);
    assert_eq!(state.players[0].meld_count, 1);
    assert_eq!(state.players[0].melds[0].meld_type, MeldType::Kakan);
    assert_eq!(state.players[0].melds[0].tiles_slice().len(), 4);
    assert_eq!(state.current_claim_counts[1], 0);
    assert_eq!(state.wall.pending_kan_dora_count, 1);
}

#[test]
fn kakan_reveals_pending_kan_doras_before_resolving_without_chankan() {
    let mut state = direct_state(GameRule::default_tenhou_sanma());
    let north_tiles = [
        parsed_tile("4z"),
        parsed_tile("4z"),
        parsed_tile("4z"),
        parsed_tile("4z"),
    ];
    state.current_player = 0;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 0, 0, 0];
    state.active_player_count = 1;
    state.wall.tiles[0] = parsed_tile("2p");
    state.wall.pending_kan_dora_count = 2;
    state.wall.dora_indicator_count = 1;
    state.wall.dora_indicators[0] = parsed_tile("1p");
    state.wall.dora_indicator_tiles[1] = parsed_tile("2p");
    state.wall.dora_indicator_tiles[2] = parsed_tile("3p");
    state.players[0].melds[0] = Meld::new(
        MeldType::Pon,
        &north_tiles[..3],
        true,
        1,
        Some(north_tiles[0]),
    );
    state.players[0].meld_count = 1;
    state.players[0].hand[0] = north_tiles[3];
    state.players[0].hand_len = 1;
    state.players[1].hand = [0; 14];
    state.players[1].hand_len = 0;
    state.players[2].hand = [0; 14];
    state.players[2].hand_len = 0;

    state._handle_kakan(
        0,
        Action::new(
            ActionType::Kakan,
            Some(north_tiles[3]),
            &north_tiles[..3],
            Some(0),
        ),
    );

    assert!(state.pending_kan.is_none());
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state.drawn_tile.is_some());
    assert!(state.is_rinshan_flag);
    assert_eq!(state.players[0].melds[0].meld_type, MeldType::Kakan);
    assert_eq!(state.players[0].melds[0].tiles_slice().len(), 4);
    assert_eq!(state.wall.pending_kan_dora_count, 1);
    assert_eq!(
        state.wall.dora_indicator_slice(),
        &[parsed_tile("1p"), parsed_tile("2p"), parsed_tile("3p")]
    );
}

#[test]
fn resolve_kan_at_dead_wall_threshold_skips_rinshan_draw_and_keeps_pending_state_unchanged() {
    let mut state = direct_state(GameRule::default_tenhou_sanma());
    state.current_player = 0;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 0, 0, 0];
    state.active_player_count = 1;
    state.drawn_tile = None;
    state.is_rinshan_flag = false;
    state.wall.tile_count = 34;
    state.wall.draw_cursor = 20;
    state.wall.pending_kan_dora_count = 2;
    state.players[0].ippatsu_cycle = true;
    for tile in [0u8, 1, 2, 3] {
        state.players[0].push_hand(tile);
    }

    state._resolve_kan(
        0,
        Action::new(ActionType::Ankan, Some(0), &[0, 1, 2, 3], Some(0)),
    );

    assert_eq!(state.players[0].meld_count, 1);
    assert_eq!(state.players[0].melds[0].meld_type, MeldType::Ankan);
    assert_eq!(state.players[0].hand_len, 0);
    assert_eq!(state.wall.remaining(), 14);
    assert_eq!(state.wall.rinshan_draw_count, 0);
    assert_eq!(state.wall.pending_kan_dora_count, 2);
    assert_eq!(state.drawn_tile, None);
    assert!(!state.is_rinshan_flag);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
}

#[test]
fn resolve_kan_ankan_reveals_old_pending_doras_and_new_kan_dora_immediately() {
    let mut state = direct_state(GameRule::default_tenhou_sanma());
    state.current_player = 0;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 0, 0, 0];
    state.active_player_count = 1;
    state.wall.tile_count = 20;
    state.wall.draw_cursor = 0;
    state.wall.pending_kan_dora_count = 2;
    state.wall.dora_indicator_count = 1;
    state.wall.dora_indicators[0] = parsed_tile("1p");
    state.wall.dora_indicator_tiles[1] = parsed_tile("2p");
    state.wall.dora_indicator_tiles[2] = parsed_tile("3p");
    state.wall.dora_indicator_tiles[3] = parsed_tile("4p");
    for tile in [0u8, 1, 2, 3] {
        state.players[0].push_hand(tile);
    }

    state._resolve_kan(
        0,
        Action::new(ActionType::Ankan, Some(0), &[0, 1, 2, 3], Some(0)),
    );

    assert_eq!(state.players[0].meld_count, 1);
    assert_eq!(state.players[0].melds[0].meld_type, MeldType::Ankan);
    assert_eq!(state.wall.pending_kan_dora_count, 0);
    assert_eq!(state.wall.dora_indicator_count, 4);
    assert_eq!(
        state.wall.dora_indicator_slice(),
        &[
            parsed_tile("1p"),
            parsed_tile("2p"),
            parsed_tile("3p"),
            parsed_tile("4p"),
        ]
    );
    assert!(state.is_rinshan_flag);
    assert!(state.drawn_tile.is_some());
}

#[test]
fn resolve_kan_ankan_logs_pai_from_consumed_tiles_when_action_tile_is_missing() {
    let mut state = GameState3P::new(5, false, Some(7), 0, GameRule::default_tenhou_sanma());
    for player in &mut state.players {
        player.reset_round();
        player.score = 35_000;
        player.score_delta = 0;
    }
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.current_player = 0;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 0, 0, 0];
    state.active_player_count = 1;
    state.wall.tile_count = 20;
    state.wall.draw_cursor = 0;
    for tile in [0u8, 1, 2, 3] {
        state.players[0].push_hand(tile);
    }

    state._resolve_kan(
        0,
        Action::new(ActionType::Ankan, None, &[0, 1, 2, 3], Some(0)),
    );

    let ankan_event = state
        .mjai_log
        .iter()
        .find(|event| event.contains("\"type\":\"ankan\""))
        .expect("ankan event should be logged");
    assert!(ankan_event.contains("\"pai\":\"1m\""));
    assert!(ankan_event.contains("\"consumed\":[\"1m\",\"1m\",\"1m\",\"1m\"]"));
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"tsumo\"")));
}

#[test]
fn resolve_kan_daiminkan_adds_final_wind_meld_and_records_pao_before_dead_wall() {
    let mut state = direct_state(GameRule::default_tenhou_sanma());
    let north_tiles = [
        parsed_tile("4z"),
        parsed_tile("4z"),
        parsed_tile("4z"),
        parsed_tile("4z"),
    ];
    state.current_player = 1;
    state.oya = 0;
    state.phase = Phase::WaitAct;
    state.active_players = [1, 0, 0, 0];
    state.active_player_count = 1;
    state.wall.tile_count = 34;
    state.wall.draw_cursor = 20;
    state.last_discard = Some((2, north_tiles[0]));
    state.players[1].melds[0] = Meld::new(MeldType::Pon, &[108, 109, 110], true, 0, Some(108));
    state.players[1].melds[1] = Meld::new(MeldType::Pon, &[112, 113, 114], true, 0, Some(112));
    state.players[1].melds[2] = Meld::new(MeldType::Pon, &[116, 117, 118], true, 0, Some(116));
    state.players[1].meld_count = 3;
    state.players[1].hand[..3].copy_from_slice(&north_tiles[1..]);
    state.players[1].hand_len = 3;

    state._resolve_kan(
        1,
        Action::new(
            ActionType::Daiminkan,
            Some(north_tiles[0]),
            &north_tiles[1..],
            Some(1),
        ),
    );

    assert_eq!(state.players[1].meld_count, 4);
    assert_eq!(state.players[1].melds[3].meld_type, MeldType::Daiminkan);
    assert_eq!(state.players[1].melds[3].tiles_slice().len(), 4);
    assert_eq!(state.players[1].hand_len, 0);
    assert_eq!(state.players[1].pao_get(50), Some(2));
    assert_eq!(state.wall.remaining(), 14);
    assert_eq!(state.wall.rinshan_draw_count, 0);
    assert_eq!(state.drawn_tile, None);
    assert!(!state.is_rinshan_flag);
}

#[test]
fn resolve_discard_reveals_pending_kan_dora_before_dahai_for_mortal_sanma() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.rule = GameRule::default_mortal_sanma();
    state.current_player = 0;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 0, 0, 0];
    state.active_player_count = 1;
    state.wall.pending_kan_dora_count = 1;
    state.wall.dora_indicator_count = 1;
    state.wall.dora_indicators[0] = parsed_tile("1p");
    state.wall.dora_indicator_tiles[1] = parsed_tile("2p");
    state.players[1].hand = [0; 14];
    state.players[1].hand_len = 0;
    state.players[1].melds = [Meld::default(); 4];
    state.players[1].meld_count = 0;
    state.players[2].hand = [0; 14];
    state.players[2].hand_len = 0;
    state.players[2].melds = [Meld::default(); 4];
    state.players[2].meld_count = 0;
    let discard_tile = state
        .drawn_tile
        .expect("test state should start with a drawn tile");

    state._resolve_discard(0, discard_tile, true);

    assert_eq!(state.wall.pending_kan_dora_count, 0);
    assert_eq!(
        state.wall.dora_indicator_slice(),
        &[parsed_tile("1p"), parsed_tile("2p")]
    );
    let dora_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"dora\""))
        .expect("dora reveal should be logged");
    let dahai_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"dahai\""))
        .expect("discard should be logged");
    assert!(dora_idx < dahai_idx);
}

#[test]
fn resolve_discard_clears_stale_kan_state_and_enters_wait_response_for_claims() {
    let mut state = direct_state(GameRule::default_mjsoul_sanma());
    let discard_tile = parsed_tile("4z");
    state.current_player = 0;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 0, 0, 0];
    state.active_player_count = 1;
    state.drawn_tile = Some(discard_tile);
    state.pending_kan = Some((
        2,
        Action::new(
            ActionType::Kakan,
            Some(discard_tile),
            &[discard_tile; 3],
            Some(2),
        ),
    ));
    state.is_rinshan_flag = true;
    state.players[0].ippatsu_cycle = true;
    state.players[0].missed_agari_doujun = true;
    state.players[0].nagashi_eligible = true;
    state.players[1].riichi_declared = true;
    set_closed_hand(&mut state, 1, "123p123s789s555z4z");
    state.players[2].hand = [0; 14];
    state.players[2].hand_len = 0;
    state.players[2].melds = [Meld::default(); 4];
    state.players[2].meld_count = 0;

    state._resolve_discard(0, discard_tile, true);

    assert!(state.pending_kan.is_none());
    assert!(!state.is_rinshan_flag);
    assert!(!state.players[0].ippatsu_cycle);
    assert!(!state.players[0].missed_agari_doujun);
    assert_eq!(
        state.players[0].nagashi_eligible,
        crate::types::is_terminal_tile(discard_tile)
    );
    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.active_player_slice(), &[1]);
    assert_eq!(state.last_discard, Some((0, discard_tile)));
    assert_eq!(state.drawn_tile, None);
    assert!(state.needs_tsumo);
    let claims = state.claims_slice(1);
    assert!(claims
        .iter()
        .any(|action| action.action_type == ActionType::Ron));
    let visible_legals = state._get_legal_actions_internal(1);
    assert!(visible_legals
        .iter()
        .any(|action| action.action_type == ActionType::Pass));
}

#[test]
fn resolve_discard_mjsoul_reveals_pending_kan_dora_after_dahai_and_accepts_riichi() {
    let mut state = test_state(false);
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.rule = GameRule::default_mjsoul_sanma();
    state.current_player = 2;
    state.phase = Phase::WaitAct;
    state.active_players = [2, 0, 0, 0];
    state.active_player_count = 1;
    state.turn_count = NP as u32 - 1;
    state.is_first_turn = true;
    state.wall.pending_kan_dora_count = 1;
    state.wall.dora_indicator_count = 1;
    state.wall.dora_indicators[0] = parsed_tile("1p");
    state.wall.dora_indicator_tiles[1] = parsed_tile("2p");
    state.pending_kan = Some((
        1,
        Action::new(ActionType::Kakan, Some(16), &[16, 17, 18], Some(1)),
    ));
    state.is_rinshan_flag = true;
    state.players[2].riichi_stage = true;
    state.players[2].ippatsu_cycle = true;
    state.players[0].hand = [0; 14];
    state.players[0].hand_len = 0;
    state.players[0].melds = [Meld::default(); 4];
    state.players[0].meld_count = 0;
    state.players[1].hand = [0; 14];
    state.players[1].hand_len = 0;
    state.players[1].melds = [Meld::default(); 4];
    state.players[1].meld_count = 0;
    let discard_tile = state
        .drawn_tile
        .expect("test state should start with a drawn tile");

    state._resolve_discard(2, discard_tile, false);

    assert!(state.pending_kan.is_none());
    assert!(!state.is_rinshan_flag);
    assert!(state.riichi_pending_acceptance.is_none());
    assert!(state.players[2].riichi_declared);
    assert!(state.players[2].ippatsu_cycle);
    assert_eq!(state.players[2].score, 34_000);
    assert_eq!(state.riichi_sticks, 1);
    assert_eq!(state.last_tedashis[2], Some(discard_tile));
    assert_eq!(state.wall.pending_kan_dora_count, 0);
    assert_eq!(
        state.wall.dora_indicator_slice(),
        &[parsed_tile("1p"), parsed_tile("2p")]
    );
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.active_player_slice(), &[0]);
    assert_eq!(state.turn_count, NP as u32);
    assert!(!state.is_first_turn);
    let dahai_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"dahai\""))
        .expect("discard should be logged");
    let dora_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"dora\""))
        .expect("dora reveal should be logged");
    let accepted_idx = state
        .mjai_log
        .iter()
        .position(|event| event.contains("\"type\":\"reach_accepted\""))
        .expect("riichi acceptance should be logged");
    assert!(dahai_idx < dora_idx);
    assert!(dora_idx < accepted_idx);
}

#[test]
fn wait_response_prefers_ron_over_competing_call_claims() {
    let mut state = direct_state(GameRule::default_mjsoul_sanma());
    let win_tile = parsed_tile("4z");
    state.oya = 2;
    state.current_player = 0;
    state.honba = 1;
    state.phase = Phase::WaitResponse;
    state.active_players = [1, 2, 0, 0];
    state.active_player_count = 2;
    state.last_discard = Some((0, win_tile));
    state.current_claim_counts[1] = 1;
    state.current_claim_counts[2] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Ron, Some(win_tile), &[], Some(1));
    state.current_claims[2][0] = Action::new(ActionType::Pon, Some(win_tile), &[49, 50], Some(2));
    set_closed_hand(&mut state, 1, "123p123s789s555z4z");
    state.players[2].hand[..2].copy_from_slice(&[49, 50]);
    state.players[2].hand_len = 2;

    let ron_result = crate::hand_evaluator_3p::HandEvaluator3P::new(
        state.players[1].hand_slice(),
        state.players[1].melds_slice(),
    )
    .calc(
        win_tile,
        state.wall.dora_indicator_slice(),
        &[],
        Some(Conditions {
            tsumo: false,
            player_wind: Wind::West,
            round_wind: Wind::East,
            honba: 1,
            is_sanma: true,
            num_players: 3,
            ..Default::default()
        }),
    );
    assert!(ron_result.is_win);

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(1))),
        Some(Action::new(
            ActionType::Pon,
            Some(win_tile),
            &[49, 50],
            Some(2),
        )),
    ]);

    assert_eq!(state.players[1].score_delta, ron_result.ron_agari as i32);
    assert_eq!(state.players[0].score_delta, -(ron_result.ron_agari as i32));
    assert_eq!(state.players[2].score_delta, 0);
    assert_eq!(state.players[2].meld_count, 0);
    assert_eq!(state.honba, 0);
    assert_eq!(state.oya, 0);
}

#[test]
fn wait_response_keeps_first_competing_call_claim_when_no_ron_occurs() {
    let mut state = test_state(true);
    let called_tile = parsed_tile("4z");
    state.phase = Phase::WaitResponse;
    state.current_player = 0;
    state.last_discard = Some((0, called_tile));
    state.active_players = [1, 2, 0, 0];
    state.active_player_count = 2;
    state.current_claim_counts[1] = 1;
    state.current_claim_counts[2] = 1;
    state.current_claims[1][0] =
        Action::new(ActionType::Pon, Some(called_tile), &[49, 50], Some(1));
    state.current_claims[2][0] = Action::new(
        ActionType::Daiminkan,
        Some(called_tile),
        &[49, 50, 51],
        Some(2),
    );
    state.players[1].hand[..2].copy_from_slice(&[49, 50]);
    state.players[1].hand_len = 2;
    state.players[2].hand[..3].copy_from_slice(&[49, 50, 51]);
    state.players[2].hand_len = 3;

    state._handle_wait_response(&[
        None,
        Some(Action::new(
            ActionType::Pon,
            Some(called_tile),
            &[49, 50],
            Some(1),
        )),
        Some(Action::new(
            ActionType::Daiminkan,
            Some(called_tile),
            &[49, 50, 51],
            Some(2),
        )),
    ]);

    assert_eq!(state.current_player, 1);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[1]);
    assert_eq!(state.players[1].meld_count, 1);
    assert_eq!(state.players[1].melds[0].meld_type, MeldType::Pon);
    assert_eq!(state.players[1].hand_len, 0);
    assert!(state.players[1].forbidden_slice().contains(&called_tile));
    assert_eq!(state.players[2].meld_count, 0);
    assert_eq!(state.players[2].hand_len, 3);
    assert!(!state.needs_tsumo);
    assert_eq!(state.drawn_tile, None);
}

#[test]
fn initialize_next_round_draw_rotates_oya_and_carries_scores_and_sticks_into_next_hand() {
    let mut state = test_state(true);
    state.game_mode = 5;
    state.oya = 1;
    state.round_wind = 0;
    state.honba = 2;
    state.riichi_sticks = 3;
    state.players[0].score = 31_000;
    state.players[1].score = 28_000;
    state.players[2].score = 46_000;
    state.players[0].score_delta = 1200;
    state.players[1].score_delta = -1200;
    state.players[2].score_delta = 0;
    state.pending_kan = Some((
        0,
        Action::new(ActionType::Ankan, Some(0), &[0, 1, 2, 3], Some(0)),
    ));
    state.last_discard = Some((1, parsed_tile("5p")));
    state.drawn_tile = Some(parsed_tile("6p"));
    state.players[0].riichi_declared = true;
    state.players[1].double_riichi_declared = true;
    state.players[2].ippatsu_cycle = true;
    state.win_results[0] = Some(WinResult::new(
        false, false, 0, 0, 0, [0u32; 16], 0, 0, 0, None, false,
    ));
    state.last_win_results[1] = Some(WinResult::new(
        false, false, 0, 0, 0, [0u32; 16], 0, 0, 0, None, false,
    ));
    state.wall.pending_kan_dora_count = 2;
    state.wall.rinshan_draw_count = 1;
    state.current_claim_counts = [1, 1, 1];
    state.riichi_sutehais = [
        Some(parsed_tile("1p")),
        Some(parsed_tile("2p")),
        Some(parsed_tile("3p")),
    ];
    state.last_tedashis = [
        Some(parsed_tile("4p")),
        Some(parsed_tile("5p")),
        Some(parsed_tile("6p")),
    ];

    state._initialize_next_round(false, true);

    assert!(!state.is_done);
    assert_eq!(state.oya, 2);
    assert_eq!(state.current_player, 2);
    assert_eq!(state.kyoku_idx, 2);
    assert_eq!(state.round_wind, 0);
    assert_eq!(state.honba, 3);
    assert_eq!(state.riichi_sticks, 3);
    assert_eq!(state.players[0].score, 31_000);
    assert_eq!(state.players[1].score, 28_000);
    assert_eq!(state.players[2].score, 46_000);
    assert!(state.pending_kan.is_none());
    assert_eq!(state.last_discard, None);
    assert_eq!(state.current_claim_counts, [0; 3]);
    assert!(state.drawn_tile.is_some());
    assert!(!state.needs_tsumo);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[2]);
    assert!(state.win_results.iter().all(Option::is_none));
    assert!(state.last_win_results.iter().all(Option::is_none));
    assert!(state.players.iter().all(|player| !player.riichi_declared));
    assert!(state
        .players
        .iter()
        .all(|player| !player.double_riichi_declared));
    assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
    assert_eq!(state.wall.pending_kan_dora_count, 0);
    assert_eq!(state.wall.rinshan_draw_count, 0);
    assert_eq!(state.riichi_sutehais, [None; 3]);
    assert_eq!(state.last_tedashis, [None; 3]);
}

#[test]
fn initialize_next_round_default_mode_ends_once_east_wraps() {
    let mut state = direct_state(GameRule::default_tenhou_sanma());
    state.game_mode = 0;
    state.oya = 2;
    state.round_wind = 0;
    state.honba = 1;
    state.players[0].score = 29_000;
    state.players[1].score = 35_000;
    state.players[2].score = 41_000;

    state._initialize_next_round(false, false);

    assert!(state.is_done);
    assert_eq!(state.oya, 2);
    assert_eq!(state.round_wind, 0);
    assert_eq!(state.honba, 1);
}

#[test]
fn initialize_next_round_east_mode_ends_when_round_wind_would_advance_past_south() {
    let mut state = direct_state(GameRule::default_tenhou_sanma());
    state.game_mode = 4;
    state.oya = 2;
    state.round_wind = 1;
    state.honba = 2;
    state.players[0].score = 29_000;
    state.players[1].score = 29_500;
    state.players[2].score = 29_800;

    state._initialize_next_round(false, false);

    assert!(state.is_done);
    assert_eq!(state.oya, 2);
    assert_eq!(state.round_wind, 1);
    assert_eq!(state.honba, 2);
    assert_eq!(state.players[0].score, 29_000);
    assert_eq!(state.players[1].score, 29_500);
    assert_eq!(state.players[2].score, 29_800);
}
