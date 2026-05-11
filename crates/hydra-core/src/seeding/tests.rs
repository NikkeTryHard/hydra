use super::*;

const TEST_SEED: [u8; 32] = [
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
];

#[test]
fn wall_determinism_same_inputs() {
    let wall_a = generate_wall(&TEST_SEED, 42, 0, 0);
    let wall_b = generate_wall(&TEST_SEED, 42, 0, 0);
    assert_eq!(wall_a, wall_b, "same inputs must produce identical walls");
}

#[test]
fn wall_different_kyoku() {
    let wall_0 = generate_wall(&TEST_SEED, 0, 0, 0);
    let wall_1 = generate_wall(&TEST_SEED, 0, 1, 0);
    assert_ne!(
        wall_0, wall_1,
        "different kyoku must produce different walls"
    );
}

#[test]
fn wall_different_honba() {
    let wall_0 = generate_wall(&TEST_SEED, 0, 0, 0);
    let wall_1 = generate_wall(&TEST_SEED, 0, 0, 1);
    assert_ne!(
        wall_0, wall_1,
        "different honba must produce different walls"
    );
}

#[test]
fn wall_contains_all_tiles() {
    let wall = generate_wall(&TEST_SEED, 0, 0, 0);
    let mut counts = [0u32; WALL_SIZE];
    for &tile in &wall {
        counts[tile as usize] += 1;
    }
    for (id, &count) in counts.iter().enumerate() {
        assert_eq!(
            count, 1,
            "tile ID {id} appears {count} times, expected exactly 1"
        );
    }
}

#[test]
fn wall_is_shuffled() {
    // A sorted wall would be [0, 1, 2, ..., 135]. Any real shuffle should
    // differ (probability of identity permutation is 1/136! ~ 0).
    let wall = generate_wall(&TEST_SEED, 0, 0, 0);
    let sorted: Vec<u8> = (0..136).collect();
    assert_ne!(
        wall.as_slice(),
        sorted.as_slice(),
        "wall should be shuffled, not sorted"
    );
}

#[test]
fn session_rng_determinism() {
    let mut rng_a = SessionRng::new(TEST_SEED);
    let mut rng_b = SessionRng::new(TEST_SEED);

    let seeds_a: Vec<[u8; 32]> = (0..10).map(|_| rng_a.next_game_seed()).collect();
    let seeds_b: Vec<[u8; 32]> = (0..10).map(|_| rng_b.next_game_seed()).collect();

    assert_eq!(
        seeds_a, seeds_b,
        "same initial seed must produce same sequence"
    );
}

#[test]
fn session_rng_different_games() {
    let mut rng = SessionRng::new(TEST_SEED);
    let seed_0 = rng.next_game_seed();
    let seed_1 = rng.next_game_seed();
    assert_ne!(
        seed_0, seed_1,
        "different game indices must produce different seeds"
    );
}

#[test]
fn session_rng_advances_index() {
    let mut rng = SessionRng::new(TEST_SEED);
    assert_eq!(rng.game_index(), 0);
    let _ = rng.next_game_seed();
    assert_eq!(rng.game_index(), 1);
    let _ = rng.next_game_seed();
    assert_eq!(rng.game_index(), 2);
}

#[test]
fn fisher_yates_determinism() {
    let seed = [0xABu8; 32];
    let mut data_a: Vec<u32> = (0..100).collect();
    let mut data_b: Vec<u32> = (0..100).collect();

    let mut rng_a = ChaCha8Rng::from_seed(seed);
    let mut rng_b = ChaCha8Rng::from_seed(seed);

    fisher_yates_shuffle(&mut data_a, &mut rng_a);
    fisher_yates_shuffle(&mut data_b, &mut rng_b);

    assert_eq!(data_a, data_b, "same RNG seed must produce same shuffle");
}

#[test]
fn fisher_yates_empty_and_single() {
    let mut rng = ChaCha8Rng::from_seed([0u8; 32]);

    // Empty slice: no-op, should not panic
    let mut empty: Vec<u8> = vec![];
    fisher_yates_shuffle(&mut empty, &mut rng);
    assert!(empty.is_empty());

    // Single element: no-op, should not panic
    let mut single = vec![42u8];
    fisher_yates_shuffle(&mut single, &mut rng);
    assert_eq!(single, vec![42]);
}

#[test]
fn derive_kyoku_seed_determinism() {
    let a = derive_kyoku_seed(&TEST_SEED, 0, 0, 0);
    let b = derive_kyoku_seed(&TEST_SEED, 0, 0, 0);
    assert_eq!(a, b, "same inputs must produce same seed");
}

#[test]
fn derive_kyoku_seed_sensitivity() {
    let base = derive_kyoku_seed(&TEST_SEED, 0, 0, 0);

    // Changing any single parameter should produce a different seed
    let diff_nonce = derive_kyoku_seed(&TEST_SEED, 1, 0, 0);
    let diff_kyoku = derive_kyoku_seed(&TEST_SEED, 0, 1, 0);
    let diff_honba = derive_kyoku_seed(&TEST_SEED, 0, 0, 1);

    assert_ne!(base, diff_nonce, "different nonce should differ");
    assert_ne!(base, diff_kyoku, "different kyoku should differ");
    assert_ne!(base, diff_honba, "different honba should differ");
}
