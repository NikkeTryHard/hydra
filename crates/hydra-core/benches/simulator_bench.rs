use criterion::{Criterion, black_box, criterion_group, criterion_main};
use hydra_core::bridge;
use hydra_core::bridge::BridgeEncodeProfile;
use hydra_core::encoder::ObservationEncoder;
use hydra_core::game_loop::{FirstActionSelector, GameRunner};
use hydra_core::hand_ev::compute_hand_ev;
use hydra_core::safety::SafetyInfo;
use hydra_core::shanten_batch::{batch_discard_shanten, batch_draw_shanten};
use hydra_core::simulator::{BatchConfig, run_batch_simple};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

fn canonical_iishanten_hand() -> [u8; 34] {
    let mut hand = [0u8; 34];
    hand[0] = 2;
    hand[3] = 1;
    hand[5] = 1;
    hand[9] = 1;
    hand[12] = 2;
    hand[18] = 1;
    hand[21] = 1;
    hand[27] = 2;
    hand[28] = 1;
    hand[29] = 1;
    hand
}

fn canonical_kokushi_tenpai_hand() -> [u8; 34] {
    let mut hand = [0u8; 34];
    let terminals = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    for &t in &terminals {
        hand[t] = 1;
    }
    hand[0] = 2;
    hand
}

fn bench_hand_ev(c: &mut Criterion) {
    let mut hand = [0u8; 34];
    hand[0] = 1;
    hand[1] = 1;
    hand[2] = 1;
    hand[9] = 1;
    hand[10] = 1;
    hand[11] = 1;
    hand[18] = 1;
    hand[19] = 1;
    hand[20] = 1;
    hand[27] = 2;
    hand[31] = 2;

    let mut remaining = [0.0f32; 34];
    remaining[0] = 2.0;
    remaining[3] = 4.0;
    remaining[9] = 3.0;
    remaining[12] = 2.0;
    remaining[18] = 4.0;
    remaining[21] = 3.0;
    remaining[27] = 2.0;
    remaining[28] = 1.0;
    remaining[31] = 1.0;

    c.bench_function("compute_hand_ev", |b| {
        b.iter(|| {
            let features = compute_hand_ev(black_box(&hand), black_box(&remaining));
            black_box(features)
        });
    });

    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(42), 0, rule);
    let obs = state.get_observation(0);
    let hand_from_obs = bridge::extract_hand(&obs);
    let discards = bridge::extract_discards(&obs);
    let melds = bridge::extract_melds(&obs);
    let dora = bridge::extract_dora(&obs);
    let remaining_from_obs =
        bridge::extract_public_remaining_counts(&hand_from_obs, &discards, &melds, &dora);

    c.bench_function("compute_hand_ev_from_observation_fixture", |b| {
        b.iter(|| {
            let features =
                compute_hand_ev(black_box(&hand_from_obs), black_box(&remaining_from_obs));
            black_box(features)
        });
    });

    let iishanten_hand = canonical_iishanten_hand();
    let iishanten_len_div3 = iishanten_hand.iter().sum::<u8>() / 3;
    let kokushi_hand = canonical_kokushi_tenpai_hand();
    let kokushi_len_div3 = kokushi_hand.iter().sum::<u8>() / 3;

    c.bench_function("batch_draw_shanten_iishanten", |b| {
        b.iter(|| {
            let batch =
                batch_draw_shanten(black_box(&iishanten_hand), black_box(iishanten_len_div3));
            black_box(batch)
        });
    });

    c.bench_function("batch_discard_shanten_iishanten", |b| {
        b.iter(|| {
            let batch =
                batch_discard_shanten(black_box(&iishanten_hand), black_box(iishanten_len_div3));
            black_box(batch)
        });
    });

    c.bench_function("batch_draw_shanten_kokushi_tenpai", |b| {
        b.iter(|| {
            let batch = batch_draw_shanten(black_box(&kokushi_hand), black_box(kokushi_len_div3));
            black_box(batch)
        });
    });

    c.bench_function("batch_discard_shanten_kokushi_tenpai", |b| {
        b.iter(|| {
            let batch =
                batch_discard_shanten(black_box(&kokushi_hand), black_box(kokushi_len_div3));
            black_box(batch)
        });
    });
}

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

    c.bench_function("encode_observation_bc_minimal", |b| {
        b.iter(|| {
            bridge::encode_observation_with_profile(
                &mut encoder,
                &obs,
                &safety,
                None,
                BridgeEncodeProfile::bc_minimal(),
            );
            black_box(encoder.as_slice());
        });
    });

    c.bench_function("encode_observation_ref", |b| {
        b.iter(|| {
            bridge::encode_observation_ref(&mut encoder, &obs_ref, &safety);
            black_box(encoder.as_slice());
        });
    });

    c.bench_function("encode_observation_ref_bc_minimal", |b| {
        b.iter(|| {
            bridge::encode_observation_ref_with_profile(
                &mut encoder,
                &obs_ref,
                &safety,
                BridgeEncodeProfile::bc_minimal(),
            );
            black_box(encoder.as_slice());
        });
    });
}

criterion_group!(
    benches,
    bench_single_game,
    bench_batch_100,
    bench_encoder,
    bench_hand_ev
);
criterion_main!(benches);
