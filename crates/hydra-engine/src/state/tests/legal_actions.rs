use crate::action::{Action, ActionType, Phase};
use crate::parser::{parse_hand, parse_tile};
use crate::rule::GameRule;
use crate::state::legal_actions::GameStateLegalActions;
use crate::state::GameState;
use crate::types::{Meld, MeldType};

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
