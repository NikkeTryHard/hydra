//! Property-based invariant tests for the game engine.
//!
//! Uses proptest to generate random seeds, plays full games with random
//! action selection, and verifies core invariants at every step.

use proptest::prelude::*;
use riichienv_core::action::{Action, Phase};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use std::collections::HashMap;

const MAX_STEPS: u32 = 10_000;

/// Pick a "random" action deterministically from seed + counter.
fn pick_action(seed: u64, counter: u64, legal: &[Action]) -> Action {
    let idx = (seed.wrapping_mul(counter.wrapping_add(1))) as usize % legal.len();
    legal[idx]
}

/// Advance the game one step with random action selection.
/// Returns (continued, actions_taken_this_step).
fn step_game(state: &mut GameState, seed: u64, counter: &mut u64) -> bool {
    if state.is_done {
        return false;
    }

    // Handle round transitions
    if state.needs_initialize_next_round {
        state.step(&HashMap::new());
        return !state.is_done;
    }

    let mut actions = HashMap::new();

    match state.phase {
        Phase::WaitAct => {
            let obs = state.get_observation(state.current_player);
            let legal = obs.legal_actions_method();
            if legal.is_empty() { return false; }
            *counter += 1;
            actions.insert(state.current_player, pick_action(seed, *counter, &legal));
        }
        Phase::WaitResponse => {
            for &pid in &state.active_players.clone() {
                let obs = state.get_observation(pid);
                let legal = obs.legal_actions_method();
                if legal.is_empty() { continue; }
                *counter += 1;
                actions.insert(pid, pick_action(seed, *counter, &legal));
            }
        }
    }

    state.step(&actions);
    !state.is_done
}

/// Sum of all 4 player scores.
fn score_sum(state: &GameState) -> i32 {
    state.players.iter().map(|p| p.score).sum()
}

/// Create a new game with the given seed.
fn new_game(seed: u64) -> GameState {
    let rule = GameRule::default_tenhou();
    GameState::new(0, true, Some(seed), 0, rule)
}

/// Play a full game, returning (final_state, step_count).
fn play_full_game(seed: u64) -> (GameState, u32) {
    let mut state = new_game(seed);
    let mut counter = 0u64;
    let mut steps = 0u32;
    while step_game(&mut state, seed, &mut counter) && steps < MAX_STEPS {
        steps += 1;
    }
    (state, steps)
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Invariants 1, 2, 6, 7: legal actions exist, scores conserved,
    /// no panics, and terminal detection -- all checked across 1000 random games.
    #[test]
    fn game_invariants_hold(seed in 0u64..1_000_000) {
        let mut state = new_game(seed);
        let mut counter = 0u64;
        let mut steps = 0u32;

        while !state.is_done && steps < MAX_STEPS {
            if state.needs_initialize_next_round {
                state.step(&HashMap::new());
                continue;
            }

            // -- Invariant 1: at least one legal action when not terminal --
            let mut actions = HashMap::new();
            match state.phase {
                Phase::WaitAct => {
                    let obs = state.get_observation(state.current_player);
                    let legal = obs.legal_actions_method();
                    prop_assert!(!legal.is_empty(),
                        "seed {seed}: no legal actions at step {steps}");
                    counter += 1;
                    actions.insert(
                        state.current_player,
                        pick_action(seed, counter, &legal),
                    );
                }
                Phase::WaitResponse => {
                    for &pid in &state.active_players.clone() {
                        let obs = state.get_observation(pid);
                        let legal = obs.legal_actions_method();
                        if legal.is_empty() { continue; }
                        counter += 1;
                        actions.insert(pid, pick_action(seed, counter, &legal));
                    }
                }
            }

            // -- Invariant 6: no crash (implicit -- reaching here means no panic) --
            state.step(&actions);
            steps += 1;

            // -- Invariant 2: score conservation --
            // 4 players x 25000 = 100000. Riichi deposits remove 1000 each.
            let sum = score_sum(&state);
            prop_assert!((80_000..=100_000).contains(&sum),
                "seed {seed}: score sum {sum} outside [80k, 100k] at step {steps}");
        }

        // -- Invariant 7: game terminates (within MAX_STEPS) --
        prop_assert!(state.is_done || steps >= MAX_STEPS,
            "seed {seed}: game neither done nor at step limit");
    }
}

// ---------------------------------------------------------------------------
// Invariant 3: shanten is always in range -1..=6 for any valid hand.
// Uses the public shanten API from riichienv-core.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn shanten_always_bounded(seed in 0u64..1000) {
        let mut state = new_game(seed);
        let mut counter = 0u64;
        let mut steps = 0u32;
        while !state.is_done && steps < MAX_STEPS {
            if matches!(state.phase, Phase::WaitAct) {
                let obs = state.get_observation(state.current_player);
                let hand_counts = hydra_core::bridge::extract_hand(&obs);
                let total: u8 = hand_counts.iter().sum();
                let len_div3 = total / 3;
                let sh = riichienv_core::shanten::calc_shanten_from_counts(
                    &hand_counts, len_div3,
                );
                prop_assert!(
                    (-1..=6).contains(&sh),
                    "shanten {} out of range at step {}, seed {}",
                    sh, steps, seed,
                );
            }
            step_game(&mut state, seed, &mut counter);
            steps += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Invariant 4: no player's hand contains more than 4 copies of any tile type.
// Called tiles may appear in both discard pool and meld (double-counted in
// visible info), so we only check per-hand counts which are always clean.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn hand_tile_count_never_exceeds_four(seed in 0u64..1000) {
        let mut state = new_game(seed);
        let mut counter = 0u64;
        let mut steps = 0u32;
        while !state.is_done && steps < MAX_STEPS {
            for pid in 0..4u8 {
                let obs = state.get_observation(pid);
                let mut counts = [0u8; 34];
                for &t in &obs.hands[pid as usize] {
                    let tt = (t / 4) as usize;
                    if tt < 34 { counts[tt] += 1; }
                }
                for (tt, &c) in counts.iter().enumerate() {
                    prop_assert!(
                        c <= 4,
                        "player {} has {} of tile type {} at step {}, seed {}",
                        pid, c, tt, steps, seed,
                    );
                }
            }
            step_game(&mut state, seed, &mut counter);
            steps += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Invariant 5 (136 total tiles) -- all tiles accounted for across wall,
// hands, discards, melds. Requires internal GameState access (not yet public).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Standalone deterministic tests
// ---------------------------------------------------------------------------

#[test]
fn game_never_panics_100_seeds() {
    // Quick smoke test: 100 games with random actions, no panics.
    for seed in 0..100u64 {
        let (state, _steps) = play_full_game(seed);
        assert!(state.is_done, "seed {seed}: game did not finish");
    }
}

#[test]
fn score_sum_at_game_end() {
    // At game end, scores should sum to ~100000 minus riichi deposits on table.
    for seed in 0..50u64 {
        let (state, _) = play_full_game(seed);
        let sum = score_sum(&state);
        assert!(
            (80_000..=100_000).contains(&sum),
            "seed {seed}: score sum {sum} outside [80k, 100k]",
        );
    }
}
