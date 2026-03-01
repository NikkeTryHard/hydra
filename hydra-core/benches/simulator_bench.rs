use criterion::{criterion_group, criterion_main, Criterion};
use hydra_core::bridge;
use hydra_core::encoder::ObservationEncoder;
use hydra_core::game_loop::{FirstActionSelector, GameRunner};
use hydra_core::safety::SafetyInfo;
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

fn bench_single_game(c: &mut Criterion) {
    c.bench_function("single_game_first_action", |b| {
        b.iter(|| {
            let mut runner = GameRunner::new(Some(42), 0);
            let mut selector = FirstActionSelector;
            runner.run_to_completion(&mut selector);
            runner.scores()
        });
    });
}

fn bench_batch_100(c: &mut Criterion) {
    use rayon::prelude::*;
    c.bench_function("batch_100_games", |b| {
        b.iter(|| {
            let results: Vec<[i32; 4]> = (0..100u64)
                .into_par_iter()
                .map(|i| {
                    let mut r = GameRunner::new(Some(i), 0);
                    let mut s = FirstActionSelector;
                    r.run_to_completion(&mut s);
                    r.scores()
                })
                .collect();
            results
        });
    });
}

fn bench_encoder(c: &mut Criterion) {
    // Set up a game state to encode
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(42), 0, rule);
    let obs = state.get_observation(0);
    let safety = SafetyInfo::new();
    let mut encoder = ObservationEncoder::new();

    c.bench_function("encode_observation_1000x", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                bridge::encode_observation(&mut encoder, &obs, &safety, None);
            }
        });
    });
}

criterion_group!(benches, bench_single_game, bench_batch_100, bench_encoder);
criterion_main!(benches);
