//! Integration tests for the game_loop module (public API).
//!
//! Verifies that GameRunner, ActionSelector, and FirstActionSelector
//! are accessible from the crate's public API after the module was
//! added to lib.rs.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::arena::compute_placements;
use hydra_core::encoder::OBS_SIZE;
use hydra_core::game_loop::{DecisionRecord, FirstActionSelector, GameRunner};

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

#[test]
fn game_runner_recording_boundary_produces_real_legal_rows_and_terminal_placements() {
    let mut runner = GameRunner::new(Some(7), 0);
    let mut selector = FirstActionSelector;
    let mut records = Vec::new();

    while !runner.is_done() {
        let outcome = runner.step_once_recording(&mut selector, &mut |record| records.push(record));
        assert_ne!(
            outcome,
            hydra_core::game_loop::StepOutcome::StepLimitExceeded
        );
        if !outcome.advanced() {
            break;
        }
    }

    assert!(runner.is_done());
    assert!(!records.is_empty());
    for record in &records {
        assert_record_contract(record);
    }
    let placements = compute_placements(runner.scores());
    let mut seen = [false; 4];
    for &placement in &placements {
        assert!(placement < 4);
        seen[placement as usize] = true;
    }
    assert!(seen.into_iter().all(|value| value));
}

#[test]
fn game_runner_recording_boundary_is_seed_deterministic() {
    fn collect(seed: u64) -> Vec<(u8, u8, u32, u8, [bool; HYDRA_ACTION_SPACE])> {
        let mut runner = GameRunner::new(Some(seed), 0);
        let mut selector = FirstActionSelector;
        let mut rows = Vec::new();
        while !runner.is_done() {
            let outcome =
                runner.step_once_recording(&mut selector, &mut |record: DecisionRecord| {
                    rows.push((
                        record.player_id,
                        record.action,
                        record.turn,
                        record.legal_count,
                        record.legal_mask,
                    ));
                });
            if !outcome.advanced() {
                break;
            }
        }
        rows
    }

    let first = collect(123);
    let second = collect(123);
    assert!(!first.is_empty());
    assert_eq!(first, second);
}

fn assert_record_contract(record: &DecisionRecord) {
    assert_eq!(record.obs.len(), OBS_SIZE);
    assert_eq!(record.legal_mask.len(), HYDRA_ACTION_SPACE);
    assert!(record.player_id < 4);
    assert!(record.seat_id < 4);
    assert!(usize::from(record.action) < HYDRA_ACTION_SPACE);
    assert!(record.legal_mask[record.action as usize]);
    let legal_count = record.legal_mask.iter().filter(|&&legal| legal).count();
    assert_eq!(usize::from(record.legal_count), legal_count);
    assert!(legal_count > 0);
    assert!(record.obs.iter().all(|value| value.is_finite()));
}
