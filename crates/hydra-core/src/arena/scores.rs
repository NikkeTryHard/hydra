pub fn games_played(scores: &[[i32; 4]]) -> usize {
    scores.len()
}

pub fn total_score_sum(scores: &[[i32; 4]]) -> i64 {
    scores
        .iter()
        .flat_map(|s| s.iter())
        .map(|&s| s as i64)
        .sum()
}

pub fn score_std(scores: &[[i32; 4]], player: u8) -> f32 {
    let mean = avg_score(scores, player);
    if scores.is_empty() {
        return 0.0;
    }
    let var: f32 = scores
        .iter()
        .map(|s| (s[player as usize] as f32 - mean).powi(2))
        .sum::<f32>()
        / scores.len() as f32;
    var.sqrt()
}

pub fn avg_score(scores: &[[i32; 4]], player: u8) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    scores
        .iter()
        .map(|s| s[player as usize] as f32)
        .sum::<f32>()
        / scores.len() as f32
}

pub fn top_two_rate(scores: &[[i32; 4]], player: u8) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let top2 = scores
        .iter()
        .filter(|s| compute_placements(**s)[player as usize] <= 1)
        .count();
    top2 as f32 / scores.len() as f32
}

pub fn fourth_place_rate(scores: &[[i32; 4]], player: u8) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let fourths = scores
        .iter()
        .filter(|s| compute_placements(**s)[player as usize] == 3)
        .count();
    fourths as f32 / scores.len() as f32
}

pub fn win_rate_from_scores(scores: &[[i32; 4]], player: u8) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let wins = scores
        .iter()
        .filter(|s| compute_placements(**s)[player as usize] == 0)
        .count();
    wins as f32 / scores.len() as f32
}

pub fn mean_placement_from_scores(scores: &[[i32; 4]], player: u8) -> f32 {
    if scores.is_empty() {
        return 2.5;
    }
    let sum: f32 = scores
        .iter()
        .map(|s| compute_placements(*s)[player as usize] as f32 + 1.0)
        .sum();
    sum / scores.len() as f32
}

pub fn compute_placements(scores: [i32; 4]) -> [u8; 4] {
    let mut indexed: [(i32, u8); 4] = [
        (scores[0], 0),
        (scores[1], 1),
        (scores[2], 2),
        (scores[3], 3),
    ];
    indexed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut placements = [0u8; 4];
    for (rank, &(_, player)) in indexed.iter().enumerate() {
        placements[player as usize] = rank as u8;
    }
    placements
}
