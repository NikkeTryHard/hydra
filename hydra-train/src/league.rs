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

    pub fn agents_of_type(&self, agent_type: &AgentType) -> Vec<usize> {
        self.agents
            .iter()
            .enumerate()
            .filter(|(_, a)| &a.agent_type == agent_type)
            .map(|(i, _)| i)
            .collect()
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
