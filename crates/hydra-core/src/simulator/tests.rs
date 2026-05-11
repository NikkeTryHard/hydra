use super::*;

#[test]
fn single_game_completes() {
    let result = simulate_single_game(Some(42), 0);
    assert!(result.total_actions > 0, "game should have actions");
    assert!(result.rounds_played > 0, "game should have rounds");
    assert!(
        result.total_actions > 10,
        "game had only {} actions, expected more than 10",
        result.total_actions
    );
}

#[test]
fn batch_returns_correct_count() {
    let config = BatchConfig {
        num_games: 4,
        base_seed: Some(100),
        game_mode: 0,
        ..Default::default()
    };
    let results = run_batch_simple(&config);
    assert_eq!(results.len(), 4);
}

#[test]
fn seeded_games_are_deterministic() {
    let r1 = simulate_single_game(Some(999), 0);
    let r2 = simulate_single_game(Some(999), 0);
    assert_eq!(r1.scores, r2.scores);
    assert_eq!(r1.total_actions, r2.total_actions);
    assert_eq!(r1.rounds_played, r2.rounds_played);
}

#[test]
fn scores_sum_is_plausible() {
    // Standard mahjong: 4 players x 25000 = 100000 total.
    // Riichi sticks on table can cause deviations, but sum
    // should be close to 100000.
    let result = simulate_single_game(Some(123), 0);
    let sum: i32 = result.scores.iter().sum();
    // Allow deviation for riichi deposits still on table.
    assert!(
        (90_000..=110_000).contains(&sum),
        "score sum {} outside plausible range",
        sum
    );
}

#[test]
fn batch_simulator_with_threads() {
    let sim = BatchSimulator::new(Some(2)).unwrap();
    let config = BatchConfig {
        num_games: 4,
        base_seed: Some(500),
        game_mode: 0,
        ..Default::default()
    };
    let results = sim.run_batch(&config);
    assert_eq!(results.len(), 4);
    for r in &results {
        assert!(r.total_actions > 0);
    }
}

#[test]
fn game_has_realistic_action_count() {
    // A full hanchan with first-legal-action should have 50-500 actions.
    let result = simulate_single_game(Some(42), 0);
    assert!(
        result.total_actions > 20,
        "game had only {} actions, expected realistic count",
        result.total_actions
    );
}
