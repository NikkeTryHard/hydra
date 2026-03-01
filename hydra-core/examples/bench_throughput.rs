use std::collections::HashMap;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use riichienv_core::action::Phase;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

/// Simulate one complete game with random action selection.
/// Correctly handles WaitAct (single actor) and WaitResponse (multiple responders).
fn simulate_game_random(seed: u64) -> (u32, [i32; 4]) {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(seed), 0, rule);
    let mut rng = rand::rngs::StdRng::from_seed({
        let mut s = [0u8; 32];
        s[..8].copy_from_slice(&seed.to_le_bytes());
        s
    });
    let mut step_count = 0u32;

    while !state.is_done && step_count < 50_000 {
        // Handle round transitions
        if state.needs_initialize_next_round {
            state.step(&HashMap::new());
            step_count += 1;
            continue;
        }

        let mut actions = HashMap::new();

        match state.phase {
            Phase::WaitAct => {
                // Single active player draws/discards
                let obs = state.get_observation(state.current_player);
                let legal = obs.legal_actions_method();
                if legal.is_empty() {
                    break;
                }
                let idx = rng.random_range(0..legal.len());
                actions.insert(state.current_player, legal[idx].clone());
            }
            Phase::WaitResponse => {
                // All active players respond (chi/pon/ron/pass)
                for &pid in &state.active_players.clone() {
                    let obs = state.get_observation(pid);
                    let legal = obs.legal_actions_method();
                    if legal.is_empty() {
                        continue;
                    }
                    // For benchmarking: pick random action (often Pass)
                    let idx = rng.random_range(0..legal.len());
                    actions.insert(pid, legal[idx].clone());
                }
            }
        }

        state.step(&actions);
        step_count += 1;
    }

    let scores = [
        state.players[0].score,
        state.players[1].score,
        state.players[2].score,
        state.players[3].score,
    ];
    (step_count, scores)
}

fn main() {
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // --- Single-core warmup + benchmark ---
    let _ = simulate_game_random(0); // warmup

    let start = Instant::now();
    let mut total_steps = 0u64;
    let n_single = 100;
    for i in 0..n_single {
        let (steps, _) = simulate_game_random(i);
        total_steps += steps as u64;
    }
    let elapsed = start.elapsed();
    let gps = n_single as f64 / elapsed.as_secs_f64();

    println!("=== SINGLE CORE ({} games) ===", n_single);
    println!("Time:             {:.2?}", elapsed);
    println!(
        "Avg steps/game:   {:.0}",
        total_steps as f64 / n_single as f64
    );
    println!("Games/sec:        {:.1}", gps);
    println!("Games/hour:       {:.0}", gps * 3600.0);
    println!();

    // --- Multi-core benchmark ---
    let n_multi: u64 = 1000;

    // warmup
    let _: Vec<_> = (0..num_cpus as u64)
        .into_par_iter()
        .map(|i| simulate_game_random(i + 90000))
        .collect();

    let start = Instant::now();
    let results: Vec<(u32, [i32; 4])> = (0..n_multi)
        .into_par_iter()
        .map(|i| simulate_game_random(i))
        .collect();
    let elapsed = start.elapsed();

    let total_steps: u64 = results.iter().map(|(s, _)| *s as u64).sum();
    let avg_steps = total_steps as f64 / results.len() as f64;
    let gps = results.len() as f64 / elapsed.as_secs_f64();

    println!(
        "=== ALL CORES ({} threads, {} games) ===",
        num_cpus, n_multi
    );
    println!("Time:             {:.2?}", elapsed);
    println!("Avg steps/game:   {:.0}", avg_steps);
    println!("Games/sec:        {:.1}", gps);
    println!("Games/hour:       {:.0}", gps * 3600.0);
    println!("Per-core/hour:    {:.0}", gps * 3600.0 / num_cpus as f64);
    println!();
    println!("Target: 100,000 games/hour/core");
}
