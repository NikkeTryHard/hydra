//! Integration tests for the game_loop module (public API).
//!
//! Verifies that GameRunner, ActionSelector, and FirstActionSelector
//! are accessible from the crate's public API after the module was
//! added to lib.rs.

use hydra_core::game_loop::{FirstActionSelector, GameRunner};

#[test]
fn game_runner_accessible_and_completes() {
    let mut runner = GameRunner::new(Some(42), 0);
    let mut selector = FirstActionSelector;
    runner.run_to_completion(&mut selector);
    assert!(runner.is_done());
    assert!(runner.total_actions() > 20);
    assert!(runner.rounds_played() > 0);
}

#[test]
fn game_runner_safety_accessible() {
    let mut runner = GameRunner::new(Some(42), 0);
    let mut selector = FirstActionSelector;
    for _ in 0..30 {
        if !runner.step_once(&mut selector) {
            break;
        }
    }
    // Safety data should be accessible via public API
    let safety = runner.safety(0);
    let has_data = safety.visible_counts.iter().any(|&c| c > 0);
    assert!(has_data, "safety data should be populated");
}

#[test]
fn game_runner_scores_accessible() {
    let mut runner = GameRunner::new(Some(99), 0);
    let mut selector = FirstActionSelector;
    runner.run_to_completion(&mut selector);
    let scores = runner.scores();
    let sum: i32 = scores.iter().sum();
    assert!((80_000..=100_000).contains(&sum));
}
