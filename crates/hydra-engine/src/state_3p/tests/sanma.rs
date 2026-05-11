use super::*;
use crate::action::Phase;
use crate::rule::GameRule;

fn empty_state() -> GameState3P {
    let mut state = GameState3P::new(3, true, Some(7), 0, GameRule::default_mortal_sanma());
    for player in &mut state.players {
        player.reset_round();
        player.score = 35_000;
    }
    state.phase = Phase::WaitAct;
    state.current_player = 0;
    state.active_players = [0; 4];
    state.active_player_count = 1;
    state.is_done = false;
    state.needs_tsumo = false;
    state.is_first_turn = true;
    state.is_rinshan_flag = false;
    state.pending_kan = None;
    state.last_discard = None;
    state.drawn_tile = None;
    state.current_claim_counts = [0; 3];
    state
}

#[test]
fn get_kita_legal_actions_requires_draw_and_wall_headroom() {
    let mut state = empty_state();
    state.players[0].push_hand(120);
    state.players[0].push_hand(121);

    assert!(state.get_kita_legal_actions(0).is_empty());

    state.drawn_tile = Some(0);
    state.wall.tile_count = 14;
    state.wall.draw_cursor = 0;
    assert!(state.get_kita_legal_actions(0).is_empty());

    state.wall.tile_count = 20;
    let actions = state.get_kita_legal_actions(0);
    assert_eq!(actions.len(), 2);
    assert!(actions
        .iter()
        .all(|action| action.action_type == ActionType::Kita));
    assert_eq!(actions[0].tile, Some(120));
    assert_eq!(actions[1].tile, Some(121));
}

#[test]
fn resolve_kita_rinshan_draws_tile_and_reveals_pending_dora() {
    let mut state = empty_state();
    state.wall.tile_count = 20;
    state.wall.draw_cursor = 0;
    state.wall.tiles[0] = 44;
    state.wall.dora_indicator_tiles[1] = 88;
    state.wall.pending_kan_dora_count = 1;

    state.resolve_kita_rinshan(0);

    assert_eq!(state.drawn_tile, Some(44));
    assert_eq!(state.players[0].hand_slice(), &[44]);
    assert_eq!(state.wall.draw_cursor, 1);
    assert_eq!(state.wall.rinshan_draw_count, 1);
    assert_eq!(state.wall.pending_kan_dora_count, 0);
    assert_eq!(
        state.wall.dora_indicator_slice(),
        &[state.wall.dora_indicator_tiles[0], 88]
    );
    assert!(state.is_rinshan_flag);
    assert_eq!(state.phase, Phase::WaitAct);
    assert_eq!(state.active_player_slice(), &[0]);
}

#[test]
fn handle_kita_without_ron_removes_tile_and_breaks_ippatsu() {
    let mut state = empty_state();
    state.wall.tile_count = 20;
    state.wall.tiles[0] = 40;
    state.wall.draw_cursor = 0;
    state.drawn_tile = Some(120);
    state.players[0].push_hand(120);
    state.players[0].push_hand(4);
    state.players[1].ippatsu_cycle = true;
    state.players[2].ippatsu_cycle = true;

    let act = Action::new(ActionType::Kita, Some(120), &[], Some(0));
    state.handle_kita(0, &act);

    assert_eq!(state.players[0].kita_slice(), &[120]);
    assert!(!state.players[0].hand_slice().contains(&120));
    assert!(!state.is_first_turn);
    assert!(state.players.iter().all(|player| !player.ippatsu_cycle));
    assert_eq!(state.drawn_tile, Some(40));
    assert_eq!(state.phase, Phase::WaitAct);
}

#[test]
fn handle_kita_with_claim_window_preserves_ippatsu_and_tracks_pending_kita() {
    let mut state = empty_state();
    state.drawn_tile = Some(120);
    state.players[0].push_hand(0);
    state.players[0].push_hand(120);
    state.players[0].hand_slice_mut().sort();
    state.players[1].hand[..13]
        .copy_from_slice(&[36, 40, 44, 72, 76, 80, 96, 100, 104, 108, 109, 110, 121]);
    state.players[1].hand_len = 13;
    state.players[1].riichi_declared = true;
    state.players[1].ippatsu_cycle = true;
    state.players[2].ippatsu_cycle = true;
    state.wall.tile_count = 20;
    state.wall.draw_cursor = 0;

    let act = Action::new(ActionType::Kita, Some(120), &[], Some(0));
    state.handle_kita(0, &act);

    assert_eq!(state.players[0].kita_slice(), &[120]);
    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.active_player_slice(), &[1]);
    assert_eq!(state.last_discard, Some((0, 120)));
    assert_eq!(state.pending_kan, Some((0, act)));
    assert!(
        state.players[1].ippatsu_cycle,
        "kita ron window should preserve pre-kita ippatsu state"
    );
    assert!(state.players[2].ippatsu_cycle);
    assert_eq!(state.current_claim_counts[1], 1);
    assert_eq!(state.current_claims[1][0].action_type, ActionType::Ron);
    assert_eq!(state.current_claims[1][0].tile, Some(120));
}

#[test]
fn resolve_kita_rinshan_is_noop_without_wall_headroom() {
    let mut state = empty_state();
    state.drawn_tile = Some(120);
    state.phase = Phase::WaitResponse;
    state.active_players = [1, 2, 0, 0];
    state.active_player_count = 2;
    state.wall.tile_count = 14;
    state.wall.draw_cursor = 0;
    state.wall.pending_kan_dora_count = 1;

    state.resolve_kita_rinshan(0);

    assert_eq!(state.drawn_tile, Some(120));
    assert_eq!(state.wall.draw_cursor, 0);
    assert_eq!(state.wall.rinshan_draw_count, 0);
    assert_eq!(state.wall.pending_kan_dora_count, 1);
    assert_eq!(state.phase, Phase::WaitResponse);
    assert_eq!(state.active_player_slice(), &[1, 2]);
}
