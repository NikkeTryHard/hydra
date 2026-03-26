use criterion::{Criterion, black_box, criterion_group, criterion_main};
use hydra_core::bridge;
use hydra_core::encoder::ObservationEncoder;
use hydra_core::game_loop::{FirstActionSelector, GameRunner};
use hydra_core::safety::SafetyInfo;
use hydra_core::simulator::{BatchConfig, run_batch_simple};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

fn bench_single_game(c: &mut Criterion) {
    c.bench_function("single_game_first_action", |b| {
        b.iter(|| {
            let mut runner = GameRunner::new(Some(42), 0);
            let mut selector = FirstActionSelector;
            runner.run_to_completion(&mut selector);
            black_box(runner.scores())
        });
    });

    c.bench_function("single_game_first_action_reuse", |b| {
        let mut runner = GameRunner::new(None, 0);
        let mut selector = FirstActionSelector;
        let mut game_idx = 0u64;
        b.iter(|| {
            runner.reset_for_new_game(Some(game_idx));
            runner.run_to_completion(&mut selector);
            game_idx = game_idx.wrapping_add(1);
            black_box(runner.scores())
        });
    });
}

fn bench_batch_100(c: &mut Criterion) {
    c.bench_function("batch_100_games", |b| {
        b.iter(|| {
            let results: Vec<[i32; 4]> = run_batch_simple(&BatchConfig {
                num_games: 100,
                base_seed: Some(0),
                num_threads: None,
                game_mode: 0,
            })
            .into_iter()
            .map(|result| result.scores)
            .collect();
            black_box(results)
        });
    });
}

fn bench_encoder(c: &mut Criterion) {
    // Set up a game state to encode
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(42), 0, rule);
    let obs = state.get_observation(0);
    let obs_ref = state.observe(0);
    let safety = SafetyInfo::new();
    let mut encoder = ObservationEncoder::new();

    c.bench_function("encode_observation", |b| {
        b.iter(|| {
            bridge::encode_observation(&mut encoder, &obs, &safety, None);
            black_box(encoder.as_slice());
        });
    });

    c.bench_function("encode_observation_ref", |b| {
        b.iter(|| {
            bridge::encode_observation_ref(&mut encoder, &obs_ref, &safety);
            black_box(encoder.as_slice());
        });
    });
}

criterion_group!(benches, bench_single_game, bench_batch_100, bench_encoder);
criterion_main!(benches);
