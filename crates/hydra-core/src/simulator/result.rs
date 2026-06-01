#[derive(Debug, Clone)]
#[repr(C)]
pub struct GameResult {
    /// Final scores for each player (4 players).
    pub scores: [i32; 4],
    /// Number of rounds (kyoku) played.
    pub rounds_played: u32,
    /// Total number of actions taken across all rounds.
    pub total_actions: u32,
    /// The seed used for this game.
    pub seed: Option<u64>,
}
