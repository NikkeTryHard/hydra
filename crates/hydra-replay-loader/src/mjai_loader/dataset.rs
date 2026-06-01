use super::*;

pub struct MjaiGame {
    pub samples: Vec<MjaiSample>,
    pub final_scores: [i32; 4],
}

impl MjaiGame {
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
pub struct MjaiDataset {
    pub games: Vec<MjaiGame>,
    pub train_fraction: f32,
}

#[inline]
pub fn normalized_train_fraction(train_fraction: f32) -> f32 {
    if train_fraction.is_finite() {
        train_fraction.clamp(0.0, 1.0)
    } else {
        0.0
    }
}
pub fn load_dataset_from_paths<P: AsRef<Path>>(
    paths: &[P],
    train_fraction: f32,
) -> io::Result<MjaiDataset> {
    let mut dataset = MjaiDataset::new(train_fraction);
    for path in paths {
        dataset.add_game(load_game_from_path(path)?);
    }
    Ok(dataset)
}

impl MjaiDataset {
    pub fn new(train_fraction: f32) -> Self {
        Self {
            games: Vec::new(),
            train_fraction: normalized_train_fraction(train_fraction),
        }
    }

    pub fn add_game(&mut self, game: MjaiGame) {
        self.games.push(game);
    }

    pub fn num_samples(&self) -> usize {
        self.games.iter().map(MjaiGame::num_samples).sum()
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
        let fraction = normalized_train_fraction(self.train_fraction);
        let n = (self.games.len() as f32 * fraction) as usize;
        (&self.games[..n], &self.games[n..])
    }
}
