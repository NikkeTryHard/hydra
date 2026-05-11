use super::*;
use crate::rule::GameRule;

fn empty_state() -> GameState3P {
    let mut state = GameState3P::new(3, true, Some(7), 0, GameRule::default_mortal_sanma());
    for player in &mut state.players {
        player.reset_round();
        player.score = 35_000;
    }
    state.is_done = false;
    state.phase = Phase::WaitAct;
    state.current_player = 0;
    state.active_players = [0; 4];
    state.active_player_count = 1;
    state.last_discard = None;
    state.pending_kan = None;
    state.drawn_tile = None;
    state.is_first_turn = false;
    state.is_rinshan_flag = false;
    state.current_claim_counts = [0; 3];
    state
}

#[test]
fn wait_act_returns_empty_for_non_current_player_and_done_state() {
    let mut state = empty_state();
    assert!(state._get_legal_actions_internal(1).is_empty());

    state.is_done = true;
    assert!(state._get_legal_actions_internal(0).is_empty());
}

#[test]
fn wait_act_offers_discards_and_kita_for_allowed_tiles() {
    let mut state = empty_state();
    state.drawn_tile = Some(0);
    state.wall.tile_count = 20;
    state.players[0].push_hand(0);
    state.players[0].push_hand(4);
    state.players[0].push_hand(120);
    state.players[0].push_forbidden(4);

    let legals = state._get_legal_actions_internal(0);
    let discard_tiles: Vec<u8> = legals
        .iter()
        .filter(|action| action.action_type == ActionType::Discard)
        .filter_map(|action| action.tile)
        .collect();
    assert_eq!(discard_tiles, vec![0, 120]);
    assert!(legals
        .iter()
        .any(|action| action.action_type == ActionType::Kita));
}

#[test]
fn wait_response_offers_claims_plus_pass() {
    let mut state = empty_state();
    state.phase = Phase::WaitResponse;
    state.current_claim_counts[1] = 1;
    state.current_claims[1][0] = Action::new(ActionType::Ron, Some(12), &[], Some(1));

    let legals = state._get_legal_actions_internal(1);
    assert_eq!(legals.len(), 2);
    assert!(legals
        .iter()
        .any(|action| action.action_type == ActionType::Ron));
    assert!(legals
        .iter()
        .any(|action| action.action_type == ActionType::Pass));
}

#[test]
fn claim_actions_for_player_generate_pon_and_daiminkan_without_chi() {
    let mut state = empty_state();
    state.wall.tile_count = 20;
    state.players[1].push_hand(120);
    state.players[1].push_hand(121);
    state.players[1].push_hand(122);
    state.players[1].push_hand(0);

    let (legals, missed_agari) = state._get_claim_actions_for_player(1, 0, 123);
    assert!(!missed_agari);
    assert!(legals
        .iter()
        .any(|action| action.action_type == ActionType::Pon));
    assert!(legals
        .iter()
        .any(|action| action.action_type == ActionType::Daiminkan));
    assert!(!legals
        .iter()
        .any(|action| action.action_type == ActionType::Chi));
}

#[test]
fn first_turn_kyushu_kyuhai_is_offered_with_nine_distinct_terminals() {
    let mut state = empty_state();
    state.is_first_turn = true;
    state.drawn_tile = Some(0);
    state.players[0].push_hand(0);
    state.players[0].push_hand(32);
    state.players[0].push_hand(36);
    state.players[0].push_hand(68);
    state.players[0].push_hand(72);
    state.players[0].push_hand(104);
    state.players[0].push_hand(108);
    state.players[0].push_hand(112);
    state.players[0].push_hand(116);

    let legals = state._get_legal_actions_internal(0);
    assert!(legals
        .iter()
        .any(|action| action.action_type == ActionType::KyushuKyuhai));
}
