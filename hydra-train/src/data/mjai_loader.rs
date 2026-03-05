//! MJAI .mjson.gz data loader for behavioral cloning.
//!
//! Requires `flate2` and `serde_json` dependencies to be added to Cargo.toml
//! for actual file parsing. The types and replay logic are defined here.

use crate::data::sample::MjaiSample;

pub struct MjaiGame {
    pub samples: Vec<MjaiSample>,
    pub final_scores: [i32; 4],
}

pub struct MjaiDataset {
    pub games: Vec<MjaiGame>,
    pub train_fraction: f32,
}

impl MjaiDataset {
    pub fn new(train_fraction: f32) -> Self {
        Self {
            games: Vec::new(),
            train_fraction,
        }
    }

    pub fn add_game(&mut self, game: MjaiGame) {
        self.games.push(game);
    }

    pub fn num_samples(&self) -> usize {
        self.games.iter().map(|g| g.samples.len()).sum()
    }

    pub fn num_games(&self) -> usize {
        self.games.len()
    }

    pub fn summary(&self) -> String {
        format!(
            "dataset(games={}, samples={})",
            self.num_games(),
            self.num_samples()
        )
    }

    pub fn train_split(&self) -> (&[MjaiGame], &[MjaiGame]) {
        let n = (self.games.len() as f32 * self.train_fraction) as usize;
        (&self.games[..n], &self.games[n..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_dataset() {
        let ds = MjaiDataset::new(0.95);
        assert_eq!(ds.num_samples(), 0);
        let (train, eval) = ds.train_split();
        assert!(train.is_empty());
        assert!(eval.is_empty());
    }
}
