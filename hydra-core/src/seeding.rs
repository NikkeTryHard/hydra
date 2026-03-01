//! Deterministic RNG seeding and wall shuffle.
//!
//! Implements the seeding hierarchy from `research/SEEDING.md`:
//! - Session-level RNG with per-game seed derivation
//! - Per-kyoku wall seed via SHA-256 KDF
//! - Vendored Fisher-Yates shuffle for cross-version determinism
//! - Deterministic 136-tile wall generation
//!
//! # Determinism guarantee
//!
//! Given the same `(seed, kyoku, honba)` tuple, `generate_wall` produces an
//! identical 136-tile wall on any platform, any Rust version, any thread count.
//!
//! # RNG choice
//!
//! Uses `ChaCha8Rng` from `rand_chacha` as specified in SEEDING.md.
//! ChaCha8 is ~33% faster than ChaCha12 (StdRng) with the same determinism
//! guarantees. Both use the same `[u8; 32]` seed format and `SeedableRng`.

use rand_chacha::ChaCha8Rng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

/// Number of tiles in a standard 4-player Riichi Mahjong wall.
/// 34 tile types x 4 copies each = 136 tiles.
pub const WALL_SIZE: usize = 136;

/// Vendored Fisher-Yates shuffle for cross-version determinism.
///
/// Does NOT depend on `rand::seq::SliceRandom` which may change its internal
/// distribution algorithm across rand versions. By vendoring, we guarantee
/// identical shuffle output for the same RNG state across all Hydra versions.
pub fn fisher_yates_shuffle<T>(slice: &mut [T], rng: &mut impl Rng) {
    for i in (1..slice.len()).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

/// Derive a deterministic seed for a specific kyoku within a game.
///
/// Uses SHA-256 as a KDF: `SHA-256(session_seed || nonce_le || kyoku || honba)`
/// produces a 32-byte seed suitable for `ChaCha8Rng::from_seed`.
///
/// This is the foundation of the `(seed, kyoku, honba) -> wall` determinism
/// contract described in SEEDING.md.
pub fn derive_kyoku_seed(session_seed: &[u8; 32], nonce: u64, kyoku: u8, honba: u8) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(session_seed);
    hasher.update(nonce.to_le_bytes());
    hasher.update([kyoku]);
    hasher.update([honba]);
    hasher.finalize().into()
}

/// Generate a deterministic wall shuffle for a specific kyoku.
///
/// Given session seed bytes, a nonce, kyoku number, and honba count,
/// produces an identical 136-tile wall on any platform.
///
/// The wall is an array of 136-format tile IDs (0..135), where each ID
/// represents one physical tile copy. Tile type = `id / 4`, copy = `id % 4`.
///
/// # Algorithm
///
/// 1. Derive a kyoku-specific seed via `SHA-256(session_seed || nonce || kyoku || honba)`
/// 2. Seed a fresh `ChaCha8Rng` from that hash
/// 3. Initialize a sorted wall `[0, 1, 2, ..., 135]`
/// 4. Apply vendored Fisher-Yates shuffle
pub fn generate_wall(session_seed: &[u8; 32], nonce: u64, kyoku: u8, honba: u8) -> [u8; 136] {
    let seed = derive_kyoku_seed(session_seed, nonce, kyoku, honba);
    let mut rng = ChaCha8Rng::from_seed(seed);

    let mut wall = [0u8; WALL_SIZE];
    for (i, tile) in wall.iter_mut().enumerate() {
        // Safe: WALL_SIZE=136 fits in u8 (max 135)
        *tile = i as u8;
    }

    fisher_yates_shuffle(&mut wall, &mut rng);
    wall
}

/// A deterministic session RNG that produces per-game seeds.
///
/// Each call to `next_game_seed` derives a unique 32-byte seed via
/// `SHA-256(session_seed || game_index_le)` and advances the internal counter.
/// This gives 2^64 independent game seeds from a single session seed.
///
/// # Example
///
/// ```
/// use hydra_core::seeding::SessionRng;
///
/// let mut session = SessionRng::new([0u8; 32]);
/// let seed_0 = session.next_game_seed();
/// let seed_1 = session.next_game_seed();
/// assert_ne!(seed_0, seed_1);
/// ```
pub struct SessionRng {
    seed: [u8; 32],
    game_index: u64,
}

impl SessionRng {
    /// Create a new session RNG from a 32-byte seed.
    pub fn new(seed: [u8; 32]) -> Self {
        Self {
            seed,
            game_index: 0,
        }
    }

    /// Get the current game index (number of seeds generated so far).
    pub fn game_index(&self) -> u64 {
        self.game_index
    }

    /// Get the seed for the next game and advance the counter.
    ///
    /// Derives via `SHA-256(session_seed || game_index_le_bytes)`.
    pub fn next_game_seed(&mut self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.seed);
        hasher.update(self.game_index.to_le_bytes());
        let result: [u8; 32] = hasher.finalize().into();
        self.game_index += 1;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SEED: [u8; 32] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E,
        0x1F, 0x20,
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
}
