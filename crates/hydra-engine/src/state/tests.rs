use super::*;
use crate::test_support::{parsed_tile, tiles_to_u32};
mod game_mode;
mod legal_actions;

fn fresh_state() -> GameState {
    GameState::new(1, false, Some(7), 0, GameRule::default_tenhou())
}
#[test]
fn sorted_insert_helpers_keep_tiles_ordered_across_edge_positions() {
    let mut hand = [0u8; 14];
    hand[..4].copy_from_slice(&[4, 12, 20, 28]);
    let mut hand_len = 4;

    sorted_insert_arr(&mut hand, &mut hand_len, 2);
    sorted_insert_arr(&mut hand, &mut hand_len, 16);
    sorted_insert_arr(&mut hand, &mut hand_len, 40);

    assert_eq!(&hand[..hand_len as usize], &[2, 4, 12, 16, 20, 28, 40]);

    let (insert_front, front_len) = copy_and_sorted_insert(&[8, 12, 16, 20], 4);
    let (insert_middle, middle_len) = copy_and_sorted_insert(&[8, 12, 16, 20], 14);
    let (insert_back, back_len) = copy_and_sorted_insert(&[8, 12, 16, 20], 24);

    assert_eq!(&insert_front[..front_len], &[4, 8, 12, 16, 20]);
    assert_eq!(&insert_middle[..middle_len], &[8, 12, 14, 16, 20]);
    assert_eq!(&insert_back[..back_len], &[8, 12, 16, 20, 24]);
}

#[test]
fn helper_methods_manage_active_players_and_claims() {
    let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());

    state.set_single_active_player(3);
    assert_eq!(state.active_player_slice(), &[3]);

    state.set_active_players_from_slice(&[1, 2, 3]);
    assert_eq!(state.active_player_slice(), &[1, 2, 3]);

    state.clear_active_players();
    assert!(state.active_player_slice().is_empty());

    let ron = Action::new(ActionType::Ron, Some(88), &[], Some(1));
    let pon = Action::new(ActionType::Pon, Some(88), &[84, 85], Some(1));
    state.push_claim(1, ron);
    state.push_claim(1, pon);
    assert_eq!(state.claims_slice(1), &[ron, pon]);

    state.clear_claims();
    assert_eq!(state.current_claim_counts, [0; 4]);
    assert!(state.claims_slice(1).is_empty());
}

#[test]
fn replay_ankan_matcher_accepts_same_tile_class_with_different_copy_ids() {
    let legal = Action::new(ActionType::Ankan, Some(16), &[16, 17, 18, 19], Some(0));
    let replay = Action::new(ActionType::Ankan, Some(17), &[17, 17, 17, 17], Some(0));

    assert!(GameState::replay_action_matches_legal(&legal, &replay));
}

#[test]
fn replay_kakan_matcher_accepts_same_tile_class_with_different_copy_ids() {
    let legal = Action::new(ActionType::Kakan, Some(16), &[16, 17, 18], Some(0));
    let replay = Action::new(ActionType::Kakan, Some(17), &[], Some(0));

    assert!(GameState::replay_action_matches_legal(&legal, &replay));
}

#[test]
fn replay_discard_matcher_accepts_same_tile_class_with_different_copy_ids() {
    let legal = Action::new(ActionType::Discard, Some(44), &[], Some(0));
    let replay = Action::new(ActionType::Discard, Some(46), &[], Some(0));

    assert!(GameState::replay_action_matches_legal(&legal, &replay));
}

#[test]
fn replay_discard_matcher_distinguishes_plain_and_red_fives() {
    let legal_red = Action::new(ActionType::Discard, Some(52), &[], Some(0));
    let replay_plain = Action::new(ActionType::Discard, Some(53), &[], Some(0));
    let legal_plain = Action::new(ActionType::Discard, Some(54), &[], Some(0));
    let replay_red = Action::new(ActionType::Discard, Some(52), &[], Some(0));

    assert!(!GameState::replay_action_matches_legal(
        &legal_red,
        &replay_plain
    ));
    assert!(GameState::replay_action_matches_legal(
        &legal_plain,
        &replay_plain
    ));
    assert!(!GameState::replay_action_matches_legal(
        &legal_plain,
        &replay_red
    ));
}

#[test]
fn replay_kan_matcher_rejects_different_tile_classes() {
    let legal = Action::new(ActionType::Ankan, Some(16), &[16, 17, 18, 19], Some(0));
    let replay = Action::new(ActionType::Ankan, Some(20), &[20, 20, 20, 20], Some(0));

    assert!(!GameState::replay_action_matches_legal(&legal, &replay));
}

#[test]
fn replay_matcher_accepts_context_implied_actions_and_kans_with_matching_consumes() {
    let legal_riichi = Action::new(ActionType::Riichi, Some(16), &[], Some(0));
    let replay_riichi = Action::new(ActionType::Riichi, None, &[], Some(0));
    assert!(GameState::replay_action_matches_legal(
        &legal_riichi,
        &replay_riichi
    ));

    let legal_kita = Action::new(ActionType::Kita, Some(120), &[], Some(0));
    let replay_kita = Action::new(ActionType::Kita, None, &[], Some(0));
    assert!(GameState::replay_action_matches_legal(
        &legal_kita,
        &replay_kita
    ));

    let legal_ankan = Action::new(ActionType::Ankan, Some(16), &[16, 17, 18, 19], Some(0));
    let replay_ankan = Action::new(ActionType::Ankan, Some(99), &[16, 17, 18, 19], Some(0));
    assert!(GameState::replay_action_matches_legal(
        &legal_ankan,
        &replay_ankan
    ));
}

#[test]
fn reset_and_reset_for_new_game_restore_logging_and_seeded_state() {
    let mut state = fresh_state();
    state.mjai_log.push("junk".to_string());
    state.mjai_log_per_player[0].push("junk".to_string());
    state.player_event_counts = [3, 2, 1, 0];
    state.mjai_events.clear();
    state.reset();

    assert_eq!(state.player_event_counts, [0; 4]);
    assert_eq!(state.mjai_log.len(), 1);
    assert_eq!(state.mjai_log_per_player[0].len(), 1);
    assert_eq!(state.mjai_events.len(), 1);

    state.players[0].score = 1234;
    state.turn_count = 99;
    state.reset_for_new_game(Some(99));
    assert_eq!(state.seed, Some(99));
    assert_eq!(state.turn_count, 0);
    assert_eq!(state.players[0].score, state.mode.starting_score());
    assert!(!state.mjai_log.is_empty());
}

#[test]
fn get_observation_masks_other_hands_and_drains_only_new_events() {
    let mut state = fresh_state();
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
    assert!(obs.hands[3].is_empty());
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
    let mut state = fresh_state();

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
fn observe_and_public_legal_action_wrappers_match_internal_state() {
    let mut state = fresh_state();
    state.players[0].score = 31_500;
    state.players[1].score = 22_100;
    state.players[2].score = 24_200;
    state.players[3].score = 22_200;
    state.players[0].riichi_declared = true;
    state.players[1].riichi_declared = true;
    state.players[0].push_discard(parsed_tile("1m"), true, false);
    state.players[1].push_discard(parsed_tile("2p"), false, true);
    state.honba = 2;
    state.riichi_sticks = 1;
    state.round_wind = 1;
    state.oya = 3;
    state.kyoku_idx = 2;
    state.current_player = 0;

    let expected_legals = state._get_legal_actions_internal(0);
    assert_eq!(state.get_legal_actions(0), expected_legals);

    let mut buf = vec![Action::new(ActionType::Pass, None, &[], Some(3))];
    state.get_legal_actions_into(0, &mut buf);
    assert_eq!(buf, expected_legals);

    let obs = state.observe(0);
    assert_eq!(obs.player_id, 0);
    assert_eq!(obs.observer_hand, state.players[0].hand_slice());
    assert_eq!(obs.melds[0].len(), state.players[0].melds_slice().len());
    assert_eq!(obs.melds[1].len(), state.players[1].melds_slice().len());
    assert_eq!(obs.discards[0], state.players[0].discards_slice());
    assert_eq!(obs.discards[1], state.players[1].discards_slice());
    assert_eq!(obs.dora_indicators, state.wall.dora_indicator_slice());
    assert_eq!(obs.scores, [31_500, 22_100, 24_200, 22_200]);
    assert_eq!(obs.riichi_declared, [true, true, false, false]);
    assert_eq!(obs.honba, 2);
    assert_eq!(obs.riichi_sticks, 1);
    assert_eq!(obs.round_wind, 1);
    assert_eq!(obs.oya, 3);
    assert_eq!(obs.kyoku_index, 2);
    assert_eq!(obs.current_player, 0);
    assert_eq!(obs.drawn_tile, state.drawn_tile);
    assert!(!obs.is_done);
}

#[test]
fn ura_helpers_and_kan_dora_reveal_use_dead_wall_layout() {
    let mut state = fresh_state();
    state.wall.tile_count = 20;
    state.wall.tiles[4] = 16;
    state.wall.tiles[5] = 52;
    state.wall.tiles[6] = 88;
    state.wall.tiles[7] = 108;
    state.wall.dora_indicator_count = 1;

    assert_eq!(state._get_ura_indicators(), vec![52]);
    assert_eq!(state._get_ura_markers(), vec!["5pr".to_string()]);

    state._reveal_kan_dora();
    assert_eq!(state.wall.dora_indicator_slice(), &[4, 88]);
}

#[test]
fn initialize_next_round_ends_single_round_games_and_rotates_east_games() {
    let mut single = GameState::new(0, true, Some(1), 0, GameRule::default_tenhou());
    single.is_done = false;
    single._initialize_next_round(false, false);
    assert!(single.is_done);

    let mut east = GameState::new(1, true, Some(1), 0, GameRule::default_tenhou());
    east.is_done = false;
    east.oya = 3;
    east.honba = 2;
    east.players.iter_mut().for_each(|p| p.score = 25_000);
    east._initialize_next_round(false, true);
    assert!(!east.is_done);
    assert_eq!(east.oya, 0);
    assert_eq!(east.round_wind, 1);
    assert_eq!(east.honba, 3);
}

#[test]
fn initialize_next_round_east_mode_ends_when_south_starts_with_30000_leader() {
    let mut state = GameState::new(1, false, Some(1), 0, GameRule::default_tenhou());
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players[0].score = 30_000;
    state.players[1].score = 25_000;
    state.players[2].score = 24_000;
    state.players[3].score = 21_000;
    state.oya = 3;
    state.honba = 1;
    state.round_wind = 0;

    state._initialize_next_round(false, true);

    assert!(state.is_done);
    assert_eq!(state.oya, 3);
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
fn initialize_next_round_half_mode_stays_alive_in_west_before_limit_even_with_30000_leader() {
    let mut state = GameState::new(2, false, Some(1), 0, GameRule::default_tenhou());
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players[0].score = 31_000;
    state.players[1].score = 24_000;
    state.players[2].score = 23_000;
    state.players[3].score = 22_000;
    state.oya = 2;
    state.honba = 2;
    state.round_wind = 1;

    state._initialize_next_round(false, false);

    assert!(!state.is_done);
    assert_eq!(state.oya, 3);
    assert_eq!(state.honba, 0);
    assert_eq!(state.round_wind, 1);
    assert_eq!(state.current_player, 3);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[3]);
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
fn initialize_next_round_half_mode_ends_after_west_wrap_without_30000_leader() {
    let mut state = GameState::new(2, false, Some(1), 0, GameRule::default_tenhou());
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players[0].score = 29_000;
    state.players[1].score = 28_000;
    state.players[2].score = 27_000;
    state.players[3].score = 26_000;
    state.oya = 3;
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
fn trigger_ryukyoku_handles_illegal_action_penalties_and_nagashi() {
    let mut state = GameState::new(1, true, Some(1), 0, GameRule::default_tenhou());
    state.players.iter_mut().for_each(|p| {
        p.score = 25_000;
        p.score_delta = 0;
    });
    state.oya = 0;
    state._trigger_ryukyoku("Error: Illegal Action by Player 1");
    assert_eq!(state.players[0].score_delta, 4000);
    assert_eq!(state.players[1].score_delta, -8000);
    assert_eq!(state.players[2].score_delta, 2000);
    assert_eq!(state.players[3].score_delta, 2000);

    let mut nagashi = GameState::new(1, true, Some(1), 0, GameRule::default_tenhou());
    nagashi.players.iter_mut().for_each(|p| {
        p.score = 25_000;
        p.score_delta = 0;
        p.nagashi_eligible = false;
    });
    nagashi.players[0].nagashi_eligible = true;
    nagashi.oya = 0;
    nagashi._trigger_ryukyoku("exhaustive_draw");
    assert!(nagashi.players[0].score > 25_000);
    assert!(nagashi.players[1].score < 25_000);
    assert_eq!(nagashi.honba, 1);
}

#[test]
fn trigger_ryukyoku_illegal_action_penalizes_dealer_offender_and_keeps_renchan() {
    let mut state = GameState::new(1, false, Some(1), 0, GameRule::default_tenhou());
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players.iter_mut().for_each(|player| {
        player.score = 25_000;
        player.score_delta = 0;
    });
    state.oya = 0;

    state._trigger_ryukyoku("Error: Illegal Action by Player 0");

    assert_eq!(state.players[0].score, 13_000);
    assert_eq!(state.players[0].score_delta, -12_000);
    assert_eq!(state.players[1].score, 29_000);
    assert_eq!(state.players[1].score_delta, 4_000);
    assert_eq!(state.players[2].score, 29_000);
    assert_eq!(state.players[2].score_delta, 4_000);
    assert_eq!(state.players[3].score, 29_000);
    assert_eq!(state.players[3].score_delta, 4_000);
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
        serde_json::json!([-12000, 4000, 4000, 4000])
    );
    assert_eq!(
        ryukyoku_event["reason"],
        serde_json::json!("Error: Illegal Action by Player 0")
    );
}

#[test]
fn push_mjai_event_masks_start_kyoku_hands_and_other_players_draws() {
    let mut state = fresh_state();
    let mut start = serde_json::Map::new();
    start.insert("type".to_string(), Value::String("start_kyoku".to_string()));
    start.insert(
        "tehais".to_string(),
        serde_json::json!([["1m", "2m"], ["3m", "4m"], ["5m", "6m"], ["7m", "8m"]]),
    );
    state._push_mjai_event(Value::Object(start));
    let masked_start = state.mjai_log_per_player[1]
        .last()
        .expect("masked start_kyoku event should exist");
    assert!(masked_start.contains("?"));
    assert!(!masked_start.contains("1m"));
    assert!(!masked_start.contains("2m"));

    let mut tsumo = serde_json::Map::new();
    tsumo.insert("type".to_string(), Value::String("tsumo".to_string()));
    tsumo.insert("actor".to_string(), Value::Number(0.into()));
    tsumo.insert("pai".to_string(), Value::String("5pr".to_string()));
    state._push_mjai_event(Value::Object(tsumo));
    let masked_tsumo = state.mjai_log_per_player[1]
        .last()
        .expect("masked tsumo event should exist");
    let actor_tsumo = state.mjai_log_per_player[0]
        .last()
        .expect("actor tsumo event should exist");
    assert!(masked_tsumo.contains("\"pai\":\"?\""));
    assert!(actor_tsumo.contains("5pr"));
}

#[test]
fn push_mjai_event_start_kyoku_masks_missing_tehai_lengths_to_default_13_in_4p() {
    let mut state = fresh_state();
    state._push_mjai_event(serde_json::json!({
        "type": "start_kyoku",
        "tehais": [
            ["1m", "2m"],
            serde_json::Value::Null,
            ["5m"],
            ["7m", "8m", "9m", "1p"]
        ]
    }));

    let p0: Value = serde_json::from_str(state.mjai_log_per_player[0].last().unwrap()).unwrap();
    let p1: Value = serde_json::from_str(state.mjai_log_per_player[1].last().unwrap()).unwrap();
    let p2: Value = serde_json::from_str(state.mjai_log_per_player[2].last().unwrap()).unwrap();
    let p3: Value = serde_json::from_str(state.mjai_log_per_player[3].last().unwrap()).unwrap();

    assert_eq!(p0["tehais"][0], serde_json::json!(["1m", "2m"]));
    assert_eq!(p0["tehais"][1].as_array().unwrap().len(), 13);
    assert_eq!(p0["tehais"][2], serde_json::json!(["?"]));
    assert_eq!(p0["tehais"][3], serde_json::json!(["?", "?", "?", "?"]));

    assert_eq!(p1["tehais"][0], serde_json::json!(["?", "?"]));
    assert_eq!(p1["tehais"][1], serde_json::Value::Null);

    assert_eq!(p2["tehais"][2], serde_json::json!(["5m"]));
    assert_eq!(p3["tehais"][3], serde_json::json!(["7m", "8m", "9m", "1p"]));
}

#[test]
fn push_mjai_event_keeps_tsumo_tile_visible_when_actor_is_unknown_in_4p() {
    let mut state = fresh_state();
    state._push_mjai_event(serde_json::json!({
        "type": "tsumo",
        "pai": "5pr"
    }));

    for pid in 0..NP {
        let event = state.mjai_log_per_player[pid]
            .last()
            .expect("tsumo event should be logged for every player");
        assert!(event.contains("5pr"));
        assert!(!event.contains("\"pai\":\"?\""));
    }
}

#[test]
fn push_mjai_event_start_kyoku_keeps_known_tehai_lengths_for_actor_with_empty_others() {
    let mut state = fresh_state();
    state._push_mjai_event(serde_json::json!({
        "type": "start_kyoku",
        "tehais": [
            ["1m"],
            [],
            ["5m", "6m", "7m"],
            []
        ]
    }));

    let p2: Value = serde_json::from_str(state.mjai_log_per_player[2].last().unwrap()).unwrap();

    assert_eq!(p2["tehais"][0], serde_json::json!(["?"]));
    assert_eq!(p2["tehais"][1], serde_json::json!([]));
    assert_eq!(p2["tehais"][2], serde_json::json!(["5m", "6m", "7m"]));
    assert_eq!(p2["tehais"][3], serde_json::json!([]));
}

#[test]
fn process_end_game_idempotent_when_logging_disabled() {
    let mut state = GameState::new(1, true, Some(1), 0, GameRule::default_tenhou());

    state._process_end_game();
    state._process_end_game();

    assert!(state.is_done);
    assert!(state.mjai_log.is_empty());
    assert!(state
        .mjai_log_per_player
        .iter()
        .all(|events| events.is_empty()));
}

#[test]
fn initialize_next_round_returns_immediately_when_game_is_already_done() {
    let mut state = fresh_state();
    state.is_done = true;
    state.oya = 3;
    state.honba = 2;
    state.round_wind = 1;

    state._initialize_next_round(false, false);

    assert!(state.is_done);
    assert_eq!(state.oya, 3);
    assert_eq!(state.honba, 2);
    assert_eq!(state.round_wind, 1);
}

#[test]
fn deal_next_exhaustive_draw_without_nagashi_keeps_scores_even_and_renchan_depends_on_oya_tenpai() {
    let mut state = GameState::new(1, false, Some(1), 0, GameRule::default_tenhou());
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.oya = 2;
    state.current_player = 1;
    state.honba = 0;
    state.round_wind = 0;
    state.wall.tile_count = 14;
    state.wall.draw_cursor = 0;
    state.players.iter_mut().for_each(|player| {
        player.nagashi_eligible = false;
        player.score = 25_000;
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
        .copy_from_slice(&[1, 5, 9, 37, 41, 45, 73, 77, 81, 113, 117, 121, 125]);
    state.players[1].hand_len = 13;
    state.players[2].hand[..13]
        .copy_from_slice(&[0, 1, 2, 36, 37, 38, 72, 73, 74, 108, 109, 110, 112]);
    state.players[2].hand_len = 13;
    state.players[3].hand[..13]
        .copy_from_slice(&[2, 6, 10, 38, 42, 46, 74, 78, 82, 114, 118, 122, 126]);
    state.players[3].hand_len = 13;

    state._deal_next();

    assert_eq!(state.players[0].score, 25_000);
    assert_eq!(state.players[1].score, 25_000);
    assert_eq!(state.players[2].score, 25_000);
    assert_eq!(state.players[3].score, 25_000);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"ryukyoku\"")));
    assert_eq!(state.oya, 3);
    assert_eq!(state.honba, 1);
    assert_eq!(state.round_wind, 0);
}

#[test]
fn process_end_game_is_idempotent_for_done_flag_and_logging_shape() {
    let mut logged = fresh_state();
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

    let mut silent = GameState::new(1, true, Some(1), 0, GameRule::default_tenhou());
    silent._process_end_game();
    silent._process_end_game();
    assert!(silent.is_done);
    assert!(silent.mjai_log.is_empty());
}

#[test]
fn check_abortive_draw_triggers_suufon_renda_and_rejects_mixed_winds() {
    let mut triggered = fresh_state();
    for (idx, player) in triggered.players.iter_mut().enumerate() {
        player.discards[0] = [108, 109, 110, 111][idx];
        player.discard_len = 1;
        player.meld_count = 0;
    }

    assert!(triggered.check_abortive_draw());
    assert!(triggered
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"ryukyoku\"")));

    let mut mixed = fresh_state();
    mixed.players[0].discards[0] = 108;
    mixed.players[1].discards[0] = 112;
    mixed.players[2].discards[0] = 109;
    mixed.players[3].discards[0] = 110;
    for player in &mut mixed.players {
        player.discard_len = 1;
        player.meld_count = 0;
    }
    assert!(!mixed.check_abortive_draw());
}

#[test]
fn check_abortive_draw_triggers_suucha_riichi_when_all_players_declared() {
    let mut state = fresh_state();
    state.players.iter_mut().for_each(|player| {
        player.riichi_declared = true;
        player.score_delta = 0;
    });

    assert!(state.check_abortive_draw());
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"reason\":\"suucha_riichi\"")));
}

#[test]
fn check_abortive_draw_triggers_suukansansen_only_with_four_kans_by_multiple_players() {
    let mut triggered = fresh_state();
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

    assert!(triggered.check_abortive_draw());
    assert!(triggered
        .mjai_log
        .iter()
        .any(|event| event.contains("\"reason\":\"suukansansen\"")));

    let mut same_owner = fresh_state();
    for meld_tiles in [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]] {
        same_owner.players[0].push_meld(Meld::new(MeldType::Ankan, &meld_tiles, false, -1, None));
    }

    assert!(!same_owner.check_abortive_draw());

    let mut not_enough = fresh_state();
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
fn handle_wait_response_triple_ron_uses_sanchaho_abortive_draw_when_enabled() {
    let mut rule = GameRule::default_tenhou();
    rule.sanchaho_is_draw = true;
    let mut state = GameState::new(1, false, Some(7), 0, rule);
    let win_tile = 48;
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.phase = Phase::WaitResponse;
    state.current_player = 0;
    state.last_discard = Some((0, win_tile));
    state.active_players = [1, 2, 3, 0];
    state.active_player_count = 3;
    for pid in 1..4usize {
        state.current_claim_counts[pid] = 1;
        state.current_claims[pid][0] =
            Action::new(ActionType::Ron, Some(win_tile), &[], Some(pid as u8));
    }

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(1))),
        Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(2))),
        Some(Action::new(ActionType::Ron, Some(win_tile), &[], Some(3))),
    ]);

    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"reason\":\"sanchaho\"")));
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.current_claim_counts, [0, 0, 0, 0]);
    assert!(state.is_first_turn);
}

#[test]
fn initialize_next_round_ends_game_immediately_when_any_player_is_bankrupt() {
    let mut state = fresh_state();
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    let oya_before = state.oya;
    let honba_before = state.honba;
    let round_wind_before = state.round_wind;
    state.players[2].score = -1;

    state._initialize_next_round(false, false);

    assert!(state.is_done);
    assert_eq!(state.oya, oya_before);
    assert_eq!(state.honba, honba_before);
    assert_eq!(state.round_wind, round_wind_before);
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"end_game\"")));
}

#[test]
fn accept_riichi_is_noop_without_pending_player_and_logs_when_present() {
    let mut silent = fresh_state();
    let score_before = silent.players[0].score;
    silent._accept_riichi();
    assert_eq!(silent.players[0].score, score_before);
    assert_eq!(silent.riichi_sticks, 0);

    let mut logged = fresh_state();
    logged.mjai_log.clear();
    logged.mjai_log_per_player = Default::default();
    logged.riichi_pending_acceptance = Some(2);
    logged._accept_riichi();
    assert_eq!(logged.players[2].score, 24_000);
    assert_eq!(logged.players[2].score_delta, -1000);
    assert_eq!(logged.riichi_sticks, 1);
    assert!(logged.players[2].riichi_declared);
    assert!(logged.players[2].ippatsu_cycle);
    assert!(logged.riichi_pending_acceptance.is_none());
    assert!(logged
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"reach_accepted\"")));
}

#[test]
fn deal_next_draws_tile_from_back_and_clears_needs_tsumo() {
    let mut state = fresh_state();
    state.drawn_tile = None;
    state.needs_tsumo = true;
    state.current_player = 1;
    state.wall.tile_count = 40;
    let hand_len_before = state.players[1].hand_len;

    state._deal_next();

    assert!(state.drawn_tile.is_some());
    assert!(!state.needs_tsumo);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[1]);
    assert_eq!(state.players[1].hand_len, hand_len_before + 1);
    assert!(state.players[1]
        .hand_slice()
        .contains(&state.drawn_tile.expect("drawn tile should be recorded")));
}

#[test]
fn resolve_kan_at_dead_wall_threshold_skips_rinshan_draw_and_keeps_pending_dora_count() {
    let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
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
    state.players[0].hand = [0; 14];
    state.players[0].hand_len = 0;
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
fn resolve_discard_reveals_pending_kan_dora_before_dahai_for_mortal_rules() {
    let mut state = fresh_state();
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.rule = GameRule::default_mortal();
    state.current_player = 0;
    state.phase = Phase::WaitAct;
    state.active_players = [0, 0, 0, 0];
    state.active_player_count = 1;
    state.wall.pending_kan_dora_count = 1;
    state.wall.dora_indicator_count = 1;
    state.wall.tiles[6] = parsed_tile("2p");
    for pid in 1..4 {
        state.players[pid].hand = [0; 14];
        state.players[pid].hand_len = 0;
        state.players[pid].melds = [Meld::default(); 4];
        state.players[pid].meld_count = 0;
    }
    let discard_tile = state
        .drawn_tile
        .expect("fresh state should start with a drawn tile");

    state._resolve_discard(0, discard_tile, true);

    assert_eq!(state.wall.pending_kan_dora_count, 0);
    assert_eq!(
        state.wall.dora_indicator_slice(),
        &[state.wall.tiles[4], parsed_tile("2p")]
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
fn resolve_discard_sets_riichi_side_effects_and_logs_dahai() {
    let mut state = fresh_state();
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.current_player = 0;
    let drawn = state
        .drawn_tile
        .expect("fresh state should start with a drawn tile");
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
fn initialize_round_applies_scores_and_logs_start_and_initial_tsumo() {
    let mut state = fresh_state();
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();

    state._initialize_round(2, 1, 3, 4, None, Some(vec![21_000, 22_000, 23_000, 24_000]));

    assert_eq!(state.oya, 2);
    assert_eq!(state.round_wind, 1);
    assert_eq!(state.honba, 3);
    assert_eq!(state.riichi_sticks, 4);
    assert_eq!(state.players[0].score, 21_000);
    assert_eq!(state.players[3].score, 24_000);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[2]);
    assert!(state.drawn_tile.is_some());
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"start_kyoku\"")));
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"tsumo\"")));
}

#[test]
fn initialize_round_without_oya_draw_leaves_needs_tsumo_true() {
    let mut state = fresh_state();
    let wall = vec![0; 52];
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();

    state._initialize_round(0, 0, 0, 0, Some(wall), Some(vec![25_000; 4]));

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
fn apply_mjai_event_and_log_action_wrappers_delegate_to_handlers() {
    let mut state = fresh_state();
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.apply_mjai_event(MjaiEvent::Reach { actor: 0 });
    assert!(state.players[0].riichi_stage);

    let mut replay = fresh_state();
    replay.mjai_log.clear();
    replay.mjai_log_per_player = Default::default();
    replay.last_tedashis = [None; NP];
    let action = LogAction::DiscardTile {
        seat: 0,
        tile: 16,
        is_liqi: false,
        is_wliqi: false,
        doras: None,
    };
    replay.apply_log_action(&action);
    assert_eq!(replay.last_discard.map(|(_, t)| t / 4), Some(4));
}

#[test]
fn replay_matcher_rejects_mismatched_action_kinds_and_tiles() {
    let discard = Action::new(ActionType::Discard, Some(16), &[], Some(0));
    let riichi = Action::new(ActionType::Riichi, Some(16), &[], Some(0));
    assert!(!GameState::replay_action_matches_legal(&discard, &riichi));

    let legal_ron = Action::new(ActionType::Ron, Some(16), &[], Some(0));
    let replay_with_wrong_tile = Action::new(ActionType::Ron, Some(20), &[], Some(0));
    assert!(!GameState::replay_action_matches_legal(
        &legal_ron,
        &replay_with_wrong_tile
    ));
}

#[test]
fn replay_matcher_accepts_kakan_context_actions_and_red_five_rules() {
    let legal_kakan = Action::new(ActionType::Kakan, Some(16), &[17, 18, 19], Some(0));
    let replay_kakan = Action::new(ActionType::Kakan, Some(16), &[], Some(0));
    assert!(GameState::replay_action_matches_legal(
        &legal_kakan,
        &replay_kakan
    ));

    let legal_ankan = Action::new(ActionType::Ankan, Some(20), &[20, 21, 22, 23], Some(0));
    let replay_ankan = Action::new(ActionType::Ankan, Some(23), &[20, 21, 22, 23], Some(0));
    assert!(GameState::replay_action_matches_legal(
        &legal_ankan,
        &replay_ankan
    ));

    let legal_riichi = Action::new(ActionType::Riichi, Some(16), &[], Some(0));
    let replay_riichi = Action::new(ActionType::Riichi, None, &[], Some(0));
    assert!(GameState::replay_action_matches_legal(
        &legal_riichi,
        &replay_riichi
    ));

    assert!(GameState::replay_tile_matches_mjai_semantics(16, 16));
    assert!(!GameState::replay_tile_matches_mjai_semantics(16, 17));
    assert!(GameState::replay_tile_matches_mjai_semantics(0, 3));
}

#[test]
fn replay_observation_accepts_sparse_kakan_action_when_drawn_tile_matches() {
    let rule = GameRule::default_mjsoul();
    let mut state = GameState::new(0, false, Some(1), 0, rule);
    state.apply_mjai_event(MjaiEvent::StartKyoku {
        bakaze: "E".to_string(),
        kyoku: 1,
        honba: 0,
        kyoutaku: 0,
        oya: 0,
        dora_marker: "1p".to_string(),
        scores: vec![25000, 25000, 25000, 25000],
        tehais: vec![
            vec![
                "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            vec![
                "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "S", "W", "N",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            vec![
                "1p", "1p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "E", "S",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            vec![
                "6m", "6m", "6m", "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "P",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        ],
    });
    state.apply_mjai_event(MjaiEvent::Pon {
        actor: 3,
        target: 0,
        pai: "6m".to_string(),
        consumed: vec!["6m".to_string(), "6m".to_string()],
    });
    state.apply_mjai_event(MjaiEvent::Tsumo {
        actor: 3,
        pai: "6m".to_string(),
    });

    let replay_kakan = Action::new(ActionType::Kakan, Some(20), &[], Some(3));
    let obs = state
        .get_observation_for_replay(3, &replay_kakan, r#"{"actor":3,"pai":"6m","type":"kakan"}"#)
        .expect("sparse kakan replay action should be accepted");

    assert!(obs
        .legal_actions_ref()
        .iter()
        .any(|action| action.action_type == ActionType::Kakan));
}

#[test]
fn replay_matcher_accepts_context_implied_tile_less_actions_and_consume_matched_kan_upgrades() {
    let legal_tsumo = Action::new(ActionType::Tsumo, Some(48), &[], Some(0));
    let replay_tsumo = Action::new(ActionType::Tsumo, None, &[], Some(0));
    assert!(GameState::replay_action_matches_legal(
        &legal_tsumo,
        &replay_tsumo
    ));

    let legal_kyushu = Action::new(ActionType::KyushuKyuhai, Some(0), &[], Some(0));
    let replay_kyushu = Action::new(ActionType::KyushuKyuhai, None, &[], Some(0));
    assert!(GameState::replay_action_matches_legal(
        &legal_kyushu,
        &replay_kyushu
    ));

    let legal_kakan = Action::new(ActionType::Kakan, Some(16), &[17, 18, 19], Some(0));
    let replay_kakan = Action::new(ActionType::Kakan, Some(64), &[17, 18, 19], Some(0));
    assert!(GameState::replay_action_matches_legal(
        &legal_kakan,
        &replay_kakan
    ));

    let legal_discard = Action::new(ActionType::Discard, Some(16), &[], Some(0));
    let replay_discard = Action::new(ActionType::Discard, None, &[], Some(0));
    assert!(!GameState::replay_action_matches_legal(
        &legal_discard,
        &replay_discard
    ));
}

#[test]
fn sorted_hand_helpers_cover_front_middle_end_and_copy_clamp_edges() {
    let mut hand = [0u8; 14];
    hand[..3].copy_from_slice(&[8, 16, 24]);
    let mut len = 3;

    sorted_insert_arr(&mut hand, &mut len, 0);
    sorted_insert_arr(&mut hand, &mut len, 20);
    sorted_insert_arr(&mut hand, &mut len, 32);

    assert_eq!(len, 6);
    assert_eq!(&hand[..len as usize], &[0, 8, 16, 20, 24, 32]);

    let (buf, copied_len) = copy_and_sorted_insert(&[12, 20, 28, 36, 44], 24);
    assert_eq!(copied_len, 5);
    assert_eq!(&buf[..copied_len], &[12, 20, 24, 28, 36]);
}

#[test]
fn claim_and_active_player_helpers_round_trip_state() {
    let mut state = fresh_state();
    let ron = Action::new(ActionType::Ron, Some(48), &[], Some(1));
    let pass = Action::new(ActionType::Pass, None, &[], Some(1));

    state.clear_active_players();
    assert!(state.active_player_slice().is_empty());

    state.set_single_active_player(2);
    assert_eq!(state.active_player_slice(), &[2]);

    state.set_active_players_from_slice(&[1, 3]);
    assert_eq!(state.active_player_slice(), &[1, 3]);

    state.push_claim(1, ron);
    state.push_claim(1, pass);
    assert_eq!(state.claims_slice(1), &[ron, pass]);

    state.push_claim(2, ron);
    assert_eq!(state.claims_slice(2), &[ron]);

    state.clear_claims();
    assert!(state.claims_slice(1).is_empty());
    assert!(state.claims_slice(2).is_empty());
}

#[test]
fn replay_matcher_rejects_tileless_non_contextual_actions_but_allows_ron() {
    let legal_ron = Action::new(ActionType::Ron, Some(16), &[], Some(0));
    let replay_ron = Action::new(ActionType::Ron, None, &[], Some(0));
    assert!(GameState::replay_action_matches_legal(
        &legal_ron,
        &replay_ron
    ));

    let legal_chi = Action::new(ActionType::Chi, Some(16), &[12, 20], Some(0));
    let replay_chi = Action::new(ActionType::Chi, None, &[12, 20], Some(0));
    assert!(!GameState::replay_action_matches_legal(
        &legal_chi,
        &replay_chi
    ));
}

#[test]
fn replay_observation_allows_pass_for_active_response_player_and_restores_state() {
    let mut state = fresh_state();
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
        ._legal_actions
        .iter()
        .any(|action| action.action_type == ActionType::Pass));
    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.active_players, [1, 0, 0, 0]);
    assert_eq!(state.active_player_count, 1);
    assert_eq!(state.current_claim_counts[1], 1);
}

#[test]
fn replay_observation_exposes_call_action_for_response_player_and_restores_claims() {
    let mut state = fresh_state();
    state.phase = Phase::WaitResponse;
    state.active_players = [2, 0, 0, 0];
    state.active_player_count = 1;
    state.current_claim_counts[2] = 1;
    state.current_claims[2][0] = Action::new(ActionType::Pon, Some(48), &[49, 50], Some(2));

    let obs = state
        .get_observation_for_replay(
            2,
            &Action::new(ActionType::Pon, Some(48), &[49, 50], Some(2)),
            "{\"type\":\"pon\"}",
        )
        .expect("pon should be exposed as legal during response replay");

    assert!(obs
        ._legal_actions
        .iter()
        .any(|action| action.action_type == ActionType::Pon));
    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.active_players, [2, 0, 0, 0]);
    assert_eq!(state.current_claim_counts[2], 1);
}

#[test]
fn replay_observation_restores_riichi_after_failed_discard_retry() {
    let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
    let drawn_tile = state
        .drawn_tile
        .expect("fresh state should start with a drawn tile");
    let invalid_tile = (0..136u8)
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
fn replay_observation_retries_discard_after_temporarily_clearing_riichi() {
    let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
    let drawn_tile = state
        .drawn_tile
        .expect("fresh state should start with a drawn tile");
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
fn replay_observation_rejects_wait_act_discard_that_is_only_tile_semantic_match_in_hand() {
    let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
    let replay_tile = state.players[0]
        .hand_slice()
        .iter()
        .copied()
        .find(|&tile| {
            !state.players[0]
                .forbidden_slice()
                .iter()
                .any(|&forbidden| forbidden / 4 == tile / 4)
        })
        .expect("fresh state should have at least one discardable hand tile");
    state.players[0].push_forbidden(replay_tile);

    let baseline_obs = state.get_observation(0);
    assert!(baseline_obs
        ._legal_actions
        .iter()
        .filter(|action| action.action_type == ActionType::Discard)
        .all(|action| action.tile.is_some_and(|tile| tile / 4 != replay_tile / 4)));

    let err = state
        .get_observation_for_replay(
            0,
            &Action::new(ActionType::Discard, Some(replay_tile), &[], Some(0)),
            "{\"type\":\"dahai\"}",
        )
        .expect_err(
            "replay discard should stay illegal when only a hand tile semantically matches",
        );

    assert!(matches!(err, RiichiError::InvalidState { .. }));
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state.players[0]
        .forbidden_slice()
        .iter()
        .any(|&forbidden| forbidden / 4 == replay_tile / 4));
}

#[test]
fn replay_observation_rejects_wait_act_discard_for_non_current_player_even_with_drawn_tile() {
    let mut state = fresh_state();
    let replay_tile = state.players[1]
        .hand_slice()
        .iter()
        .copied()
        .find(|&tile| {
            state.players[1]
                .forbidden_slice()
                .iter()
                .all(|&forbidden| forbidden / 4 != tile / 4)
        })
        .expect("player 1 should have at least one non-forbidden tile in hand");

    let err = state
        .get_observation_for_replay(
            1,
            &Action::new(ActionType::Discard, Some(replay_tile), &[], Some(1)),
            "{\"type\":\"dahai\"}",
        )
        .expect_err("non-current player discard should stay illegal during wait-act replay");

    assert!(matches!(err, RiichiError::InvalidState { .. }));
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.active_player_slice(), &[0]);
}

#[test]
fn wait_response_marks_missed_ron_and_riichi_when_player_passes_on_win() {
    let mut state = fresh_state();
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
        None,
    ]);

    assert!(state.players[1].missed_agari_doujun);
    assert!(state.players[1].missed_agari_riichi);
}

#[test]
fn wait_response_resolves_pending_kakan_after_all_players_pass() {
    let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
    state.phase = Phase::WaitResponse;
    state.current_player = 0;
    state.last_discard = Some((0, 48));
    state.active_players = [1, 2, 3, 0];
    state.active_player_count = 3;
    state.current_claim_counts[1] = 1;
    state.current_claim_counts[2] = 1;
    state.current_claim_counts[3] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Pass, None, &[], Some(1));
    state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));
    state.current_claims[3][0] = Action::new(ActionType::Pass, None, &[], Some(3));
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
        Some(Action::new(ActionType::Pass, None, &[], Some(3))),
    ]);

    assert!(state.pending_kan.is_none());
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state.drawn_tile.is_some());
    assert!(state.is_rinshan_flag);
    assert_eq!(state.wall.rinshan_draw_count, rinshan_before + 1);
    assert_eq!(state.wall.pending_kan_dora_count, pending_before + 1);
    assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
}

#[test]
fn wait_response_all_pass_accepts_riichi_and_advances_turn_out_of_first_cycle() {
    let mut state = GameState::new(1, true, Some(7), 0, GameRule::default_tenhou());
    state.phase = Phase::WaitResponse;
    state.current_player = 3;
    state.turn_count = NP as u32 - 1;
    state.is_first_turn = true;
    state.riichi_pending_acceptance = Some(2);
    if state.players[0].hand_len > 0 {
        state.players[0].hand_len -= 1;
    }
    state.drawn_tile = None;

    state._handle_wait_response(&[None, None, None, None]);

    assert!(state.riichi_pending_acceptance.is_none());
    assert_eq!(state.players[2].score, 24_000);
    assert!(state.players[2].riichi_declared);
    assert!(state.players[2].ippatsu_cycle);
    assert_eq!(state.riichi_sticks, 1);
    assert_eq!(state.turn_count, NP as u32);
    assert!(!state.is_first_turn);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state.drawn_tile.is_some());
}

#[test]
fn step_array_initializes_pending_round_before_processing_actions() {
    let mut state = fresh_state();
    state.oya = 1;
    state.honba = 2;
    state.round_wind = 0;
    state.phase = Phase::WaitResponse;
    state.needs_initialize_next_round = true;
    state.pending_oya_won = false;
    state.pending_is_draw = false;
    state.last_discard = Some((0, 44));
    state.last_error = None;

    let actions = [
        None,
        None,
        Some(Action::new(ActionType::Discard, Some(255), &[], Some(2))),
        None,
    ];

    state.step_array(&actions);

    assert!(!state.needs_initialize_next_round);
    assert_eq!(state.oya, 2);
    assert_eq!(state.honba, 0);
    assert_eq!(state.round_wind, 0);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.current_player, 2);
    assert_eq!(state.active_player_slice(), &[2]);
    assert_eq!(state.last_discard, None);
    assert!(state.last_error.is_none());
}

#[test]
fn step_rejects_tileless_discard_and_records_illegal_action_error() {
    let mut state = GameState::new(1, false, Some(1), 0, GameRule::default_tenhou());
    state.mjai_log.clear();
    state.mjai_log_per_player = Default::default();
    state.players.iter_mut().for_each(|player| {
        player.score = 25_000;
        player.score_delta = 0;
    });

    let mut actions = std::collections::HashMap::new();
    actions.insert(
        state.current_player,
        Action::new(ActionType::Discard, None, &[], Some(state.current_player)),
    );

    state.step(&actions);

    assert_eq!(
        state.last_error.as_deref(),
        Some("Error: Illegal Action by Player 0")
    );
    assert_eq!(state.players[0].score, 13_000);
    assert_eq!(state.players[0].score_delta, -12_000);
    assert_eq!(state.players[1].score, 29_000);
    assert_eq!(state.players[1].score_delta, 4_000);
    assert_eq!(state.players[2].score, 29_000);
    assert_eq!(state.players[2].score_delta, 4_000);
    assert_eq!(state.players[3].score, 29_000);
    assert_eq!(state.players[3].score_delta, 4_000);
    assert_eq!(state.oya, 0);
    assert_eq!(state.honba, 1);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.current_player, 0);
    assert_eq!(state.active_player_slice(), &[0]);
    assert!(state.drawn_tile.is_some());
    assert!(state
        .mjai_log
        .iter()
        .any(|event| event.contains("\"type\":\"ryukyoku\"")));
}

#[test]
fn step_array_unchecked_initializes_pending_round_before_processing_actions() {
    let mut state = fresh_state();
    state.oya = 1;
    state.honba = 2;
    state.round_wind = 0;
    state.phase = Phase::WaitResponse;
    state.needs_initialize_next_round = true;
    state.pending_oya_won = false;
    state.pending_is_draw = false;
    state.last_discard = Some((0, 44));
    state.last_error = None;

    let actions = [
        None,
        None,
        Some(Action::new(ActionType::Riichi, None, &[], Some(2))),
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
    assert!(state.last_error.is_none());
    assert!(!state.players[2].riichi_stage);
}

#[test]
fn wait_response_prioritizes_pon_over_existing_chi_claim() {
    let mut state = fresh_state();
    state.phase = Phase::WaitResponse;
    state.last_discard = Some((0, 12));
    state.active_players = [1, 2, 0, 0];
    state.active_player_count = 2;
    state.current_claim_counts[1] = 1;
    state.current_claim_counts[2] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Chi, Some(12), &[8, 16], Some(1));
    state.current_claims[2][0] = Action::new(ActionType::Pon, Some(12), &[13, 14], Some(2));
    state.players[1].hand[..2].copy_from_slice(&[8, 16]);
    state.players[1].hand_len = 2;
    state.players[2].hand[..2].copy_from_slice(&[13, 14]);
    state.players[2].hand_len = 2;

    state._handle_wait_response(&[
        None,
        Some(Action::new(ActionType::Chi, Some(12), &[8, 16], Some(1))),
        Some(Action::new(ActionType::Pon, Some(12), &[13, 14], Some(2))),
        None,
    ]);

    assert_eq!(state.current_player, 2);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[2]);
    assert_eq!(state.players[2].meld_count, 1);
    assert_eq!(state.players[2].melds[0].meld_type, MeldType::Pon);
    assert_eq!(state.players[1].meld_count, 0);
}

#[test]
fn initialize_round_clears_pending_round_and_kan_riichi_transients() {
    let mut state = fresh_state();
    state.pending_kan = Some((
        0,
        Action::new(ActionType::Kakan, Some(16), &[16, 17, 18], Some(0)),
    ));
    state.is_rinshan_flag = true;
    state.wall.rinshan_draw_count = 2;
    state.wall.pending_kan_dora_count = 1;
    state.riichi_pending_acceptance = Some(1);
    state.needs_initialize_next_round = true;
    state.pending_oya_won = true;
    state.pending_is_draw = true;
    state.last_discard = Some((0, 16));
    state.win_results[0] = Some(WinResult::new(
        false, false, 0, 0, 0, [0u32; 16], 0, 0, 0, None, false,
    ));
    state.riichi_sutehais[0] = Some(16);
    state.last_tedashis[0] = Some(16);
    state.players[0].ippatsu_cycle = true;

    state._initialize_round(0, 0, 0, 0, Some(vec![0; 52]), Some(vec![25_000; 4]));

    assert!(state.pending_kan.is_none());
    assert!(!state.is_rinshan_flag);
    assert_eq!(state.wall.rinshan_draw_count, 0);
    assert_eq!(state.wall.pending_kan_dora_count, 0);
    assert!(state.riichi_pending_acceptance.is_none());
    assert!(!state.needs_initialize_next_round);
    assert!(!state.pending_oya_won);
    assert!(!state.pending_is_draw);
    assert!(state.last_discard.is_none());
    assert!(state.win_results.iter().all(Option::is_none));
    assert_eq!(state.riichi_sutehais, [None; NP]);
    assert_eq!(state.last_tedashis, [None; NP]);
    assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
}

#[test]
fn replay_tile_semantics_reject_different_tile_classes_and_plain_red_mismatch() {
    assert!(!GameState::replay_tile_matches_mjai_semantics(0, 4));
    assert!(!GameState::replay_tile_matches_mjai_semantics(52, 53));
    assert!(GameState::replay_tile_matches_mjai_semantics(53, 54));
}

#[test]
fn replay_tile_semantics_accepts_same_non_red_tile_copies() {
    assert!(GameState::replay_tile_matches_mjai_semantics(17, 18));
    assert!(GameState::replay_tile_matches_mjai_semantics(89, 91));
}

#[test]
fn replay_matcher_rejects_consume_mismatches_for_non_kan_calls() {
    let legal_pon = Action::new(ActionType::Pon, Some(16), &[17, 18], Some(1));
    let replay_wrong_consume = Action::new(ActionType::Pon, Some(16), &[17, 19], Some(1));
    assert!(!GameState::replay_action_matches_legal(
        &legal_pon,
        &replay_wrong_consume
    ));

    let legal_chi = Action::new(ActionType::Chi, Some(16), &[12, 20], Some(1));
    let replay_empty_consume = Action::new(ActionType::Chi, Some(16), &[], Some(1));
    assert!(!GameState::replay_action_matches_legal(
        &legal_chi,
        &replay_empty_consume
    ));
}

#[test]
fn replay_observation_error_restores_original_response_state_on_illegal_action() {
    let mut state = fresh_state();
    state.phase = Phase::WaitResponse;
    state.active_players = [0, 2, 3, 1];
    state.active_player_count = 2;
    state.current_claim_counts[2] = 1;
    state.current_claims[2][0] = Action::new(ActionType::Pass, None, &[], Some(2));

    let err = state
        .get_observation_for_replay(
            1,
            &Action::new(ActionType::Discard, Some(parsed_tile("1m")), &[], Some(1)),
            "{\"type\":\"dahai\"}",
        )
        .expect_err("inactive response player should reject unrelated replay discard");

    assert!(matches!(err, RiichiError::InvalidState { .. }));
    let message = match err {
        RiichiError::InvalidState { message } => message,
        other => panic!("expected InvalidState replay error, got {other:?}"),
    };
    assert!(message.contains("Replay desync"));
    assert!(message.contains("Log action: {\"type\":\"dahai\"}"));
    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.active_players, [0, 2, 3, 1]);
    assert_eq!(state.active_player_count, 2);
    assert_eq!(state.current_claim_counts[2], 1);
    assert!(state.claims_slice(1).is_empty());
}
