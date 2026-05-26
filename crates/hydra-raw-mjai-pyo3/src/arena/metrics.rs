use hydra_core::arena::compute_placements;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::PLAYER_COUNT;

use super::shared::ArenaGame;

#[pyclass(skip_from_py_object)]
pub(crate) struct PyPairedArenaMetrics {
    #[pyo3(get)]
    games: usize,
    #[pyo3(get)]
    candidate_winrate: f64,
    #[pyo3(get)]
    baseline_winrate: f64,
    #[pyo3(get)]
    candidate_avg_rank: f64,
    #[pyo3(get)]
    baseline_avg_rank: f64,
    #[pyo3(get)]
    candidate_mean_placement: f64,
    #[pyo3(get)]
    baseline_mean_placement: f64,
    #[pyo3(get)]
    candidate_top2: f64,
    #[pyo3(get)]
    baseline_top2: f64,
    #[pyo3(get)]
    candidate_fourth: f64,
    #[pyo3(get)]
    baseline_fourth: f64,
    #[pyo3(get)]
    candidate_avg_score: f64,
    #[pyo3(get)]
    baseline_avg_score: f64,
    #[pyo3(get)]
    score_delta: f64,
    #[pyo3(get)]
    pt_delta: f64,
}

impl PyPairedArenaMetrics {
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("games", self.games)?;
        dict.set_item("candidate_winrate", self.candidate_winrate)?;
        dict.set_item("baseline_winrate", self.baseline_winrate)?;
        dict.set_item("candidate_avg_rank", self.candidate_avg_rank)?;
        dict.set_item("baseline_avg_rank", self.baseline_avg_rank)?;
        dict.set_item("candidate_mean_placement", self.candidate_mean_placement)?;
        dict.set_item("baseline_mean_placement", self.baseline_mean_placement)?;
        dict.set_item("candidate_top2", self.candidate_top2)?;
        dict.set_item("baseline_top2", self.baseline_top2)?;
        dict.set_item("candidate_fourth", self.candidate_fourth)?;
        dict.set_item("baseline_fourth", self.baseline_fourth)?;
        dict.set_item("candidate_avg_score", self.candidate_avg_score)?;
        dict.set_item("baseline_avg_score", self.baseline_avg_score)?;
        dict.set_item("score_delta", self.score_delta)?;
        dict.set_item("pt_delta", self.pt_delta)?;
        Ok(dict)
    }
}

#[derive(Default)]
pub(crate) struct ArenaSideStats {
    pub(crate) games: usize,
    pub(crate) wins: usize,
    pub(crate) top2: usize,
    pub(crate) fourth: usize,
    pub(crate) placement_sum: u64,
    pub(crate) score_sum: i64,
}

impl ArenaSideStats {
    pub(crate) fn add(&mut self, score: i32, placement: u8) {
        self.games += 1;
        if placement == 0 {
            self.wins += 1;
        }
        if placement <= 1 {
            self.top2 += 1;
        }
        if placement == 3 {
            self.fourth += 1;
        }
        self.placement_sum += u64::from(placement) + 1;
        self.score_sum += i64::from(score);
    }

    fn rate(&self, count: usize) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            count as f64 / self.games as f64
        }
    }

    fn mean_placement(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.placement_sum as f64 / self.games as f64
        }
    }

    fn avg_score(&self) -> f64 {
        if self.games == 0 {
            0.0
        } else {
            self.score_sum as f64 / self.games as f64
        }
    }
}

pub(crate) fn add_completed_game(
    game: &ArenaGame,
    candidate: &mut ArenaSideStats,
    baseline: &mut ArenaSideStats,
    score_delta_sum: &mut f64,
    pt_delta_sum: &mut f64,
) {
    let scores = game.runner.scores();
    let placements = compute_placements(scores);
    add_completed_scores(
        &scores,
        &placements,
        &game.candidate_seats,
        candidate,
        baseline,
        score_delta_sum,
        pt_delta_sum,
    );
}

pub(crate) fn add_completed_scores(
    scores: &[i32; PLAYER_COUNT],
    placements: &[u8; PLAYER_COUNT],
    candidate_seats: &[bool; PLAYER_COUNT],
    candidate: &mut ArenaSideStats,
    baseline: &mut ArenaSideStats,
    score_delta_sum: &mut f64,
    pt_delta_sum: &mut f64,
) {
    let mut candidate_score_sum = 0i64;
    let mut candidate_count = 0usize;
    let mut baseline_score_sum = 0i64;
    let mut baseline_count = 0usize;
    for seat in 0..PLAYER_COUNT {
        if candidate_seats[seat] {
            candidate.add(scores[seat], placements[seat]);
            candidate_score_sum += i64::from(scores[seat]);
            candidate_count += 1;
        } else {
            baseline.add(scores[seat], placements[seat]);
            baseline_score_sum += i64::from(scores[seat]);
            baseline_count += 1;
        }
    }
    if candidate_count > 0 && baseline_count > 0 {
        let candidate_avg = candidate_score_sum as f64 / candidate_count as f64;
        let baseline_avg = baseline_score_sum as f64 / baseline_count as f64;
        let delta = candidate_avg - baseline_avg;
        *score_delta_sum += delta;
        *pt_delta_sum += delta / 1000.0;
    }
}

pub(crate) fn metrics_dict<'py>(
    py: Python<'py>,
    games: usize,
    candidate: ArenaSideStats,
    baseline: ArenaSideStats,
    score_delta_sum: f64,
    pt_delta_sum: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let metrics = PyPairedArenaMetrics {
        games,
        candidate_winrate: candidate.rate(candidate.wins),
        baseline_winrate: baseline.rate(baseline.wins),
        candidate_avg_rank: candidate.mean_placement(),
        baseline_avg_rank: baseline.mean_placement(),
        candidate_mean_placement: candidate.mean_placement(),
        baseline_mean_placement: baseline.mean_placement(),
        candidate_top2: candidate.rate(candidate.top2),
        baseline_top2: baseline.rate(baseline.top2),
        candidate_fourth: candidate.rate(candidate.fourth),
        baseline_fourth: baseline.rate(baseline.fourth),
        candidate_avg_score: candidate.avg_score(),
        baseline_avg_score: baseline.avg_score(),
        score_delta: score_delta_sum / games as f64,
        pt_delta: pt_delta_sum / games as f64,
    };
    metrics.to_dict(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_side_stats_aggregates_rank_metrics() {
        let mut stats = ArenaSideStats::default();
        stats.add(35_000, 0);
        stats.add(25_000, 1);
        stats.add(15_000, 3);

        assert_eq!(stats.games, 3);
        assert_eq!(stats.rate(stats.wins), 1.0 / 3.0);
        assert_eq!(stats.rate(stats.top2), 2.0 / 3.0);
        assert_eq!(stats.rate(stats.fourth), 1.0 / 3.0);
        assert_eq!(stats.mean_placement(), 7.0 / 3.0);
        assert_eq!(stats.avg_score(), 25_000.0);
    }
}
