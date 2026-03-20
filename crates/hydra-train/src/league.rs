//! Population league: agent roster, matchmaking, and Elo tracking.

use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub enum AgentType {
    Current,
    Checkpoint(u32),
    BcAnchor,
    Exploiter,
}

#[derive(Debug, Clone)]
pub struct LeagueAgent {
    pub weights_path: PathBuf,
    pub agent_type: AgentType,
    pub elo: f32,
}

pub struct LeagueSnapshot {
    pub agents: Vec<(String, f32)>,
    pub total_games: u64,
}

pub struct League {
    pub agents: Vec<LeagueAgent>,
    pub total_matches: u64,
}

impl League {
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
            total_matches: 0,
        }
    }

    pub fn snapshot(&self) -> LeagueSnapshot {
        LeagueSnapshot {
            agents: self
                .agents
                .iter()
                .map(|a| (a.weights_path.display().to_string(), a.elo))
                .collect(),
            total_games: self.total_matches,
        }
    }

    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    pub fn standard_roster(&mut self, current_path: PathBuf, checkpoints: &[PathBuf]) {
        self.add_agent(LeagueAgent {
            weights_path: current_path,
            agent_type: AgentType::Current,
            elo: 1500.0,
        });
        for (i, path) in checkpoints.iter().enumerate() {
            self.add_agent(LeagueAgent {
                weights_path: path.clone(),
                agent_type: AgentType::Checkpoint(i as u32),
                elo: 1500.0,
            });
        }
    }

    pub fn best_agent_by_elo(&self) -> Option<usize> {
        self.agents
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.elo
                    .partial_cmp(&b.elo)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    pub fn mean_elo(&self) -> f32 {
        if self.agents.is_empty() {
            return 0.0;
        }
        self.agents.iter().map(|a| a.elo).sum::<f32>() / self.agents.len() as f32
    }

    pub fn elo_spread(&self) -> f32 {
        if self.agents.is_empty() {
            return 0.0;
        }
        let max = self
            .agents
            .iter()
            .map(|a| a.elo)
            .fold(f32::NEG_INFINITY, f32::max);
        let min = self
            .agents
            .iter()
            .map(|a| a.elo)
            .fold(f32::INFINITY, f32::min);
        max - min
    }

    pub fn worst_agent_by_elo(&self) -> Option<usize> {
        self.agents
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.elo
                    .partial_cmp(&b.elo)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    pub fn elo_range(&self) -> (f32, f32) {
        let max = self
            .agents
            .iter()
            .map(|a| a.elo)
            .fold(f32::NEG_INFINITY, f32::max);
        let min = self
            .agents
            .iter()
            .map(|a| a.elo)
            .fold(f32::INFINITY, f32::min);
        (min, max)
    }

    pub fn top_k_agents(&self, k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self
            .agents
            .iter()
            .enumerate()
            .map(|(i, a)| (i, a.elo))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }

    pub fn total_elo(&self) -> f32 {
        self.agents.iter().map(|a| a.elo).sum()
    }

    pub fn elo_of(&self, idx: usize) -> f32 {
        self.agents.get(idx).map_or(1500.0, |a| a.elo)
    }

    pub fn current_agent(&self) -> Option<usize> {
        self.agents
            .iter()
            .position(|a| a.agent_type == AgentType::Current)
    }

    pub fn agents_of_type(&self, agent_type: &AgentType) -> Vec<usize> {
        self.agents
            .iter()
            .enumerate()
            .filter(|(_, a)| &a.agent_type == agent_type)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn replace_weakest(&mut self, new_agent: LeagueAgent) {
        if let Some(idx) = self.worst_agent_by_elo() {
            self.agents[idx] = new_agent;
        }
    }

    pub fn remove_agent(&mut self, idx: usize) -> Option<LeagueAgent> {
        if idx < self.agents.len() {
            Some(self.agents.remove(idx))
        } else {
            None
        }
    }

    pub fn add_agent(&mut self, agent: LeagueAgent) {
        self.agents.push(agent);
    }

    pub fn select_opponents(&self, num: usize, rng_val: f32) -> Vec<usize> {
        if self.agents.is_empty() {
            return Vec::new();
        }
        let n = self.agents.len();
        (0..num)
            .map(|i| ((rng_val * 1000.0) as usize + i) % n)
            .collect()
    }

    pub fn update_elo(&mut self, winner: usize, loser: usize, k: f32) {
        let r_w = self.agents[winner].elo;
        let r_l = self.agents[loser].elo;
        let e_w = 1.0 / (1.0 + 10.0f32.powf((r_l - r_w) / 400.0));
        self.agents[winner].elo += k * (1.0 - e_w);
        self.agents[loser].elo += k * (0.0 - (1.0 - e_w));
        self.total_matches += 1;
    }

    pub fn update_elo_4p(&mut self, placements: [usize; 4], k: f32) {
        for i in 0..4 {
            for j in (i + 1)..4 {
                if placements[i] < placements[j] {
                    self.update_elo(placements[i], placements[j], k / 6.0);
                } else if placements[j] < placements[i] {
                    self.update_elo(placements[j], placements[i], k / 6.0);
                }
            }
        }
    }
}

impl League {
    pub fn summary(&self) -> String {
        format!(
            "agents={} matches={} spread={:.0}",
            self.num_agents(),
            self.total_matches,
            self.elo_spread()
        )
    }
}

impl Default for League {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(path: &str, agent_type: AgentType, elo: f32) -> LeagueAgent {
        LeagueAgent {
            weights_path: PathBuf::from(path),
            agent_type,
            elo,
        }
    }

    #[test]
    fn default_league_helpers_cover_empty_roster_behavior() {
        let league = League::default();
        assert_eq!(league.num_agents(), 0);
        assert_eq!(league.mean_elo(), 0.0);
        assert_eq!(league.elo_spread(), 0.0);
        assert_eq!(league.best_agent_by_elo(), None);
        assert_eq!(league.worst_agent_by_elo(), None);
        assert_eq!(league.current_agent(), None);
        assert_eq!(league.elo_of(99), 1500.0);
        assert_eq!(league.select_opponents(3, 0.5), Vec::<usize>::new());
        assert_eq!(league.snapshot().agents.len(), 0);
        assert_eq!(league.snapshot().total_games, 0);
        assert_eq!(league.summary(), "agents=0 matches=0 spread=0");
    }

    #[test]
    fn roster_snapshot_and_type_helpers_track_agent_metadata() {
        let mut league = League::new();
        league.standard_roster(
            PathBuf::from("current.bin"),
            &[PathBuf::from("cp0.bin"), PathBuf::from("cp1.bin")],
        );

        assert_eq!(league.num_agents(), 3);
        assert_eq!(league.current_agent(), Some(0));
        assert_eq!(league.agents_of_type(&AgentType::Checkpoint(1)), vec![2]);

        let snapshot = league.snapshot();
        assert_eq!(snapshot.total_games, 0);
        assert_eq!(snapshot.agents[0], ("current.bin".to_string(), 1500.0));
        assert_eq!(snapshot.agents[2], ("cp1.bin".to_string(), 1500.0));
    }

    #[test]
    fn ranking_and_range_helpers_return_expected_indices() {
        let mut league = League::new();
        league.add_agent(agent("a.bin", AgentType::Current, 1510.0));
        league.add_agent(agent("b.bin", AgentType::Checkpoint(0), 1475.0));
        league.add_agent(agent("c.bin", AgentType::Exploiter, 1600.0));
        league.add_agent(agent("d.bin", AgentType::BcAnchor, 1490.0));

        assert_eq!(league.best_agent_by_elo(), Some(2));
        assert_eq!(league.worst_agent_by_elo(), Some(1));
        assert_eq!(league.top_k_agents(2), vec![2, 0]);
        assert_eq!(league.top_k_agents(10), vec![2, 0, 3, 1]);
        assert_eq!(league.elo_range(), (1475.0, 1600.0));
        assert_eq!(league.total_elo(), 6075.0);
        assert!((league.mean_elo() - 1518.75).abs() < 1e-6);
        assert_eq!(league.elo_spread(), 125.0);
        assert_eq!(league.elo_of(3), 1490.0);
        assert_eq!(league.summary(), "agents=4 matches=0 spread=125");
    }

    #[test]
    fn replace_and_remove_helpers_mutate_roster_safely() {
        let mut league = League::new();
        league.add_agent(agent("weak.bin", AgentType::Checkpoint(0), 1400.0));
        league.add_agent(agent("mid.bin", AgentType::Checkpoint(1), 1500.0));
        league.add_agent(agent("best.bin", AgentType::Current, 1600.0));

        league.replace_weakest(agent("new.bin", AgentType::Exploiter, 1550.0));
        assert_eq!(league.worst_agent_by_elo(), Some(1));
        assert_eq!(league.agents[0].weights_path, PathBuf::from("new.bin"));

        let removed = league.remove_agent(1).expect("agent at idx 1 should exist");
        assert_eq!(removed.weights_path, PathBuf::from("mid.bin"));
        assert_eq!(league.num_agents(), 2);
        assert!(league.remove_agent(99).is_none());
    }

    #[test]
    fn elo_updates_correctly() {
        let mut league = League::new();
        league.add_agent(LeagueAgent {
            weights_path: PathBuf::from("a.bin"),
            agent_type: AgentType::Current,
            elo: 1500.0,
        });
        league.add_agent(LeagueAgent {
            weights_path: PathBuf::from("b.bin"),
            agent_type: AgentType::Checkpoint(1),
            elo: 1500.0,
        });
        league.update_elo(0, 1, 32.0);
        assert!(league.agents[0].elo > 1500.0);
        assert!(league.agents[1].elo < 1500.0);
    }

    #[test]
    fn league_matchmaking() {
        let mut league = League::new();
        for i in 0..5 {
            league.add_agent(LeagueAgent {
                weights_path: PathBuf::from(format!("{i}.bin")),
                agent_type: AgentType::Checkpoint(i),
                elo: 1500.0,
            });
        }
        let opps = league.select_opponents(3, 0.5);
        assert_eq!(opps.len(), 3);
        for &idx in &opps {
            assert!(idx < 5);
        }
    }

    #[test]
    fn league_matchmaking_covers_all_agents() {
        let mut league = League::new();
        for i in 0..4 {
            league.add_agent(LeagueAgent {
                weights_path: PathBuf::from(format!("{i}.bin")),
                agent_type: AgentType::Checkpoint(i),
                elo: 1500.0,
            });
        }
        let mut seen = [false; 4];
        for r in 0..400 {
            let opps = league.select_opponents(1, r as f32 / 400.0);
            seen[opps[0]] = true;
        }
        let coverage = seen.iter().filter(|&&s| s).count();
        assert!(
            coverage >= 3,
            "should cover most agents, covered {coverage}/4"
        );
    }

    #[test]
    fn elo_conserved_after_update() {
        let mut league = League::new();
        league.add_agent(LeagueAgent {
            weights_path: PathBuf::from("a.bin"),
            agent_type: AgentType::Current,
            elo: 1500.0,
        });
        league.add_agent(LeagueAgent {
            weights_path: PathBuf::from("b.bin"),
            agent_type: AgentType::Checkpoint(1),
            elo: 1500.0,
        });
        let total_before: f32 = league.agents.iter().map(|a| a.elo).sum();
        league.update_elo(0, 1, 32.0);
        let total_after: f32 = league.agents.iter().map(|a| a.elo).sum();
        assert!(
            (total_before - total_after).abs() < 0.01,
            "Elo should be zero-sum: before={total_before}, after={total_after}"
        );
    }

    #[test]
    fn elo_4p_conserved() {
        let mut league = League::new();
        for i in 0..4 {
            league.add_agent(LeagueAgent {
                weights_path: PathBuf::from(format!("{i}.bin")),
                agent_type: AgentType::Checkpoint(i as u32),
                elo: 1500.0,
            });
        }
        let before: f32 = league.agents.iter().map(|a| a.elo).sum();
        league.update_elo_4p([0, 1, 2, 3], 32.0);
        let after: f32 = league.agents.iter().map(|a| a.elo).sum();
        assert!((before - after).abs() < 0.1, "4p Elo: {before} vs {after}");
        assert!(league.agents[0].elo > 1500.0, "1st place should gain Elo");
    }
}
