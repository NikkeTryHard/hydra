use rand::prelude::*;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};

/// Wall state for 4-player mahjong (136 tiles).
#[derive(Debug, Clone)]
pub struct WallState {
    /// Fixed-size array holding all 136 wall tiles.
    pub tiles: [u8; 136],
    /// Number of tiles currently in the wall.
    pub tile_count: u8,
    /// Revealed dora indicator tiles (up to 5).
    pub dora_indicators: [u8; 5],
    /// Number of revealed dora indicators.
    pub dora_indicator_count: u8,
    /// Number of rinshan (replacement) tiles drawn so far.
    pub rinshan_draw_count: u8,
    /// Number of kan dora reveals deferred until after discard.
    pub pending_kan_dora_count: u8,
    /// SHA-256 digest of the shuffled wall for verification.
    pub wall_digest: String,
    /// Random salt used in the wall digest computation.
    pub salt: String,
    /// Optional deterministic seed for reproducible shuffles.
    pub seed: Option<u64>,
    /// Monotonic hand counter used for per-hand seed derivation.
    pub hand_index: u64,
    /// Cursor for rinshan draws from the front of the wall.
    pub draw_cursor: usize,
}

impl WallState {
    /// Create a new empty wall state with the given seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            tiles: [0; 136],
            tile_count: 0,
            dora_indicators: [0; 5],
            dora_indicator_count: 0,
            rinshan_draw_count: 0,
            pending_kan_dora_count: 0,
            wall_digest: String::new(),
            salt: String::new(),
            seed,
            hand_index: 0,
            draw_cursor: 0,
        }
    }

    /// Shuffle the wall tiles and initialize dora indicators.
    pub fn shuffle(&mut self, skip_digest: bool) {
        for i in 0..136u8 {
            self.tiles[i as usize] = i;
        }
        let mut rng = if let Some(episode_seed) = self.seed {
            let hand_seed = splitmix64(episode_seed.wrapping_add(self.hand_index));
            self.hand_index = self.hand_index.wrapping_add(1);
            StdRng::seed_from_u64(hand_seed)
        } else {
            self.hand_index = self.hand_index.wrapping_add(1);
            StdRng::from_entropy()
        };

        self.tiles.shuffle(&mut rng);
        if skip_digest {
            self.salt.clear();
            self.wall_digest.clear();
        } else {
            self.salt = format!("{:016x}", rng.next_u64());
            let mut hasher = Sha256::new();
            hasher.update(self.salt.as_bytes());
            for &t in &self.tiles {
                hasher.update([t]);
            }
            self.wall_digest = format!("{:x}", hasher.finalize());
        }

        self.tiles.reverse();
        self.tile_count = 136;
        self.draw_cursor = 0;

        self.dora_indicators[0] = self.tiles[4];
        self.dora_indicator_count = 1;
        self.rinshan_draw_count = 0;
        self.pending_kan_dora_count = 0;
    }

    /// Returns the number of remaining drawable tiles in the wall.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.tile_count as usize - self.draw_cursor
    }

    /// Draws the next rinshan tile from the front of the wall via cursor.
    #[inline]
    pub fn draw_rinshan(&mut self) -> u8 {
        let t = self.tiles[self.draw_cursor];
        self.draw_cursor += 1;
        t
    }

    /// Load a pre-built wall from a tile vector, replacing the current wall.
    pub fn load_wall(&mut self, tiles: Vec<u8>) {
        let len = tiles.len().min(136);
        self.tiles[..len].copy_from_slice(&tiles[..len]);
        // Reverse in place
        self.tiles[..len].reverse();
        self.tile_count = len as u8;
        self.dora_indicator_count = 0;
        if len > 5 {
            self.dora_indicators[0] = self.tiles[4];
            self.dora_indicator_count = 1;
        }
        self.rinshan_draw_count = 0;
        self.pending_kan_dora_count = 0;
        self.draw_cursor = 0;
    }

    /// Draws the next tile from the back of the wall (equivalent to Vec::pop).
    #[inline]
    pub fn draw_back(&mut self) -> Option<u8> {
        if self.tile_count == 0 {
            return None;
        }
        self.tile_count -= 1;
        Some(self.tiles[self.tile_count as usize])
    }

    /// Returns the current dora indicators as a slice.
    #[inline]
    pub fn dora_indicator_slice(&self) -> &[u8] {
        &self.dora_indicators[..self.dora_indicator_count as usize]
    }

    /// Pushes a new dora indicator.
    #[inline]
    pub fn push_dora_indicator(&mut self, tile: u8) {
        if (self.dora_indicator_count as usize) < 5 {
            self.dora_indicators[self.dora_indicator_count as usize] = tile;
            self.dora_indicator_count += 1;
        }
    }

    /// Resets dora indicators to a single tile.
    #[inline]
    pub fn set_dora_indicators_single(&mut self, tile: u8) {
        self.dora_indicators[0] = tile;
        self.dora_indicator_count = 1;
    }
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
