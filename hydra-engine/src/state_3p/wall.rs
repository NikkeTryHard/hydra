use rand::prelude::*;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};

use crate::types::is_sanma_excluded_tile;

/// Wall state for 3-player mahjong (108 tiles, sanma hardcoded).
#[derive(Debug, Clone)]
pub struct WallState3P {
    pub tiles: [u8; 136],
    pub tile_count: u8,
    pub dora_indicators: [u8; 5],
    pub dora_indicator_count: u8,
    /// Pre-extracted dora indicator tiles (omote) in order D1..D5.
    pub dora_indicator_tiles: [u8; 5],
    /// Pre-extracted ura dora indicator tiles in order U1..U5.
    pub ura_indicator_tiles: [u8; 5],
    pub rinshan_draw_count: u8,
    pub pending_kan_dora_count: u8,
    pub wall_digest: String,
    pub salt: String,
    pub seed: Option<u64>,
    pub hand_index: u64,
    /// Cursor for rinshan draws from the front of the wall.
    pub draw_cursor: usize,
}

impl WallState3P {
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            tiles: [0; 136],
            tile_count: 0,
            dora_indicators: [0; 5],
            dora_indicator_count: 0,
            dora_indicator_tiles: [0; 5],
            ura_indicator_tiles: [0; 5],
            rinshan_draw_count: 0,
            pending_kan_dora_count: 0,
            wall_digest: String::new(),
            salt: String::new(),
            seed,
            hand_index: 0,
            draw_cursor: 0,
        }
    }

    pub fn shuffle(&mut self, skip_digest: bool) {
        // 3P: 108 tiles (no 2m-8m). Filter into temp, then copy to fixed array.
        let mut count = 0u8;
        for i in 0..136u8 {
            if !is_sanma_excluded_tile(i) {
                self.tiles[count as usize] = i;
                count += 1;
            }
        }
        self.tile_count = count;

        let mut rng = if let Some(episode_seed) = self.seed {
            let hand_seed = splitmix64(episode_seed.wrapping_add(self.hand_index));
            self.hand_index = self.hand_index.wrapping_add(1);
            StdRng::seed_from_u64(hand_seed)
        } else {
            self.hand_index = self.hand_index.wrapping_add(1);
            StdRng::from_entropy()
        };

        self.tiles[..count as usize].shuffle(&mut rng);
        if skip_digest {
            self.salt.clear();
            self.wall_digest.clear();
        } else {
            self.salt = format!("{:016x}", rng.next_u64());
            let mut hasher = Sha256::new();
            hasher.update(self.salt.as_bytes());
            for &t in &self.tiles[..count as usize] {
                hasher.update([t]);
            }
            self.wall_digest = format!("{:x}", hasher.finalize());
        }

        self.tiles[..count as usize].reverse();

        // Pre-extract dora/ura indicators from standard layout.
        // After reversal: D_i omote at tiles[4+2i], ura at tiles[5+2i].
        for i in 0..5 {
            self.dora_indicator_tiles[i] = self.tiles[4 + 2 * i];
            self.ura_indicator_tiles[i] = self.tiles[5 + 2 * i];
        }

        self.dora_indicators[0] = self.dora_indicator_tiles[0];
        self.dora_indicator_count = 1;
        self.rinshan_draw_count = 0;
        self.pending_kan_dora_count = 0;
        self.draw_cursor = 0;
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

    pub fn load_wall(&mut self, tiles: Vec<u8>) {
        let len = tiles.len().min(136);

        // MjSoul 3P dead wall layout (positions 94-107):
        //   Positions 94-99: dora stacks 1-3 (each pair [X,X+1] = ura,omote)
        //     Stack1=[98,99]  Stack2=[96,97]  Stack3=[94,95]
        //   Positions 100-107: rinshan draw area (8 tiles for up to 8 draws: kans+kitas)
        // Dora stacks 4-5 extend into the live wall area (positions 90-93):
        //     Stack4=[92,93]  Stack5=[90,91]
        // These are pre-extracted before any draws, so it's safe even if
        // those live wall positions are later drawn during normal play.
        if len == 108 {
            // D1..D5 omote indicators
            self.dora_indicator_tiles = [tiles[99], tiles[97], tiles[95], tiles[93], tiles[91]];
            // U1..U5 ura indicators
            self.ura_indicator_tiles = [tiles[98], tiles[96], tiles[94], tiles[92], tiles[90]];
        }

        self.tiles[..len].copy_from_slice(&tiles[..len]);
        self.tiles[..len].reverse();
        self.tile_count = len as u8;
        self.dora_indicators[0] = self.dora_indicator_tiles[0];
        self.dora_indicator_count = 1;
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
