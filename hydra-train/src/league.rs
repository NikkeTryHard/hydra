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

pub struct League {
    pub agents: Vec<LeagueAgent>,
}

impl League {
    pub fn new() -> Self {
        Self { agents: Vec::new() }
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
}
