use super::GameResult;

/// Simulate a single complete game with first-legal-action selection.
/// Used for benchmarking throughput -- real training uses NN policy.
#[cfg(test)]
pub(super) fn simulate_single_game(seed: Option<u64>, game_mode: u8) -> GameResult {
    let mut runner = crate::game_loop::GameRunner::new(None, game_mode);
    simulate_single_game_with_runner(&mut runner, seed)
}

pub(super) fn simulate_single_game_with_runner(
    runner: &mut crate::game_loop::GameRunner,
    seed: Option<u64>,
) -> GameResult {
    let mut selector = crate::game_loop::FirstActionSelector;
    runner.reset_for_new_game(seed);
    let outcome = runner.run_to_completion(&mut selector);
    assert_eq!(
        outcome,
        crate::game_loop::StepOutcome::Complete,
        "batch simulator stopped before game completion: {outcome:?}"
    );
    GameResult {
        scores: runner.scores(),
        rounds_played: runner.rounds_played(),
        total_actions: runner.total_actions(),
        seed,
    }
}
