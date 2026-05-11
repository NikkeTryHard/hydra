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

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
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
mod tests;
