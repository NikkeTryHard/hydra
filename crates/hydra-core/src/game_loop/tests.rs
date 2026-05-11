use super::*;
use crate::safety::bit_test;
use riichienv_core::action::{Action, ActionType};

#[test]
fn game_completes_with_first_action() {
    let mut runner = GameRunner::new(Some(42), 0);
    let mut selector = FirstActionSelector;
    runner.run_to_completion(&mut selector);
    assert!(runner.is_done());
    assert!(
        runner.total_actions() > 20,
        "expected realistic action count, got {}",
        runner.total_actions()
    );
}

#[test]
fn safety_updated_during_game() {
    let mut runner = GameRunner::new(Some(42), 0);
    let mut selector = FirstActionSelector;
    for _ in 0..20 {
        if !runner.step_once(&mut selector) {
            break;
        }
    }
    let has_safety_data = (0..4).any(|p| {
        let s = runner.safety(p);
        s.visible_counts.iter().any(|&c| c > 0)
    });
    assert!(has_safety_data, "safety should be updated during play");
}

#[test]
fn scores_are_plausible() {
    let mut runner = GameRunner::new(Some(99), 0);
    let mut selector = FirstActionSelector;
    runner.run_to_completion(&mut selector);
    let sum: i32 = runner.scores().iter().sum();
    assert!(
        (90_000..=110_000).contains(&sum),
        "score sum {} outside plausible range",
        sum
    );
}

#[test]
fn session_seeded_games_are_deterministic() {
    let mut session_a = crate::seeding::SessionRng::new([42u8; 32]);
    let mut session_b = crate::seeding::SessionRng::new([42u8; 32]);

    let mut runner_a = GameRunner::new_with_session(&mut session_a, 0);
    let mut runner_b = GameRunner::new_with_session(&mut session_b, 0);

    let mut sel_a = FirstActionSelector;
    let mut sel_b = FirstActionSelector;

    runner_a.run_to_completion(&mut sel_a);
    runner_b.run_to_completion(&mut sel_b);

    assert_eq!(runner_a.scores(), runner_b.scores());
    assert_eq!(runner_a.total_actions(), runner_b.total_actions());
}

#[test]
fn tedashi_detected_during_game() {
    // Run a game for enough steps that some tedashi discards happen.
    // FirstActionSelector usually picks the first legal action, which
    // for WaitAct is often the first discard -- typically NOT the drawn tile.
    let mut runner = GameRunner::new(Some(42), 0);
    let mut selector = FirstActionSelector;
    for _ in 0..50 {
        if !runner.step_once(&mut selector) {
            break;
        }
    }
    let has_tedashi = (0..4).any(|p| {
        let s = runner.safety(p);
        s.genbutsu_tedashi.iter().any(|&bits| bits != 0)
    });
    assert!(
        has_tedashi,
        "at least one tedashi should be detected after 50 steps"
    );
}

#[test]
fn track_action_maps_actor_to_relative_opponent_slot() {
    let mut runner = GameRunner::new(Some(7), 0);
    runner.state.drawn_tile = None;
    let discard = Action::new(ActionType::Discard, Some(6 * 4), &[], None);

    runner.track_action(2, &discard);

    assert!(bit_test(runner.safety(0).genbutsu_all[1], 6));
    assert!(bit_test(runner.safety(1).genbutsu_all[0], 6));
    assert!(bit_test(runner.safety(3).genbutsu_all[2], 6));
    assert!(!bit_test(runner.safety(0).genbutsu_all[0], 6));
    assert!(!bit_test(runner.safety(0).genbutsu_all[2], 6));
}
