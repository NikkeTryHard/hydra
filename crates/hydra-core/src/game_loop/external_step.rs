use crate::action::riichienv_to_hydra;
use riichienv_core::action::{Action, Phase};
use std::time::Instant;

use super::outcome::StepOutcome;
use super::pending::{
    CachedLegalActions, PendingDecision, PendingDecisionTiming, legal_action_map,
};
use super::runner::GameRunner;

impl GameRunner {
    fn cached_action_for_hydra_id(cached: &CachedLegalActions, action_id: u8) -> Option<Action> {
        cached
            .by_hydra_id
            .get(usize::from(action_id))
            .copied()
            .flatten()
    }
    pub fn pending_decisions(&mut self) -> Result<Vec<PendingDecision>, StepOutcome> {
        self.pending_decisions_with_timing(None)
    }

    pub fn pending_decisions_with_timing(
        &mut self,
        mut timing: Option<&mut PendingDecisionTiming>,
    ) -> Result<Vec<PendingDecision>, StepOutcome> {
        if let Some(timing) = timing.as_deref_mut() {
            timing.calls += 1;
        }
        if let Err(outcome) = self.begin_action_step() {
            match outcome {
                StepOutcome::Complete => {
                    if let Some(timing) = timing.as_deref_mut() {
                        timing.complete += 1;
                    }
                }
                StepOutcome::Advanced => {
                    if let Some(timing) = timing.as_deref_mut() {
                        timing.advanced += 1;
                    }
                }
                _ => {}
            }
            return Err(outcome);
        }

        let mut decisions = Vec::with_capacity(4);
        match self.state.phase {
            Phase::WaitAct => {
                if let Some(timing) = timing.as_deref_mut() {
                    timing.wait_act += 1;
                }
                let pid = self.state.current_player;
                self.legal_buf.clear();
                let legal_start = Instant::now();
                self.state.get_legal_actions_into(pid, &mut self.legal_buf);
                if let Some(timing) = timing.as_deref_mut() {
                    timing.legal_actions += legal_start.elapsed();
                }
                if self.legal_buf.is_empty() {
                    return Err(StepOutcome::NoLegalAction { player: pid });
                }
                decisions.push(self.pending_decision_for_player(pid, timing.as_deref_mut()));
            }
            Phase::WaitResponse => {
                if let Some(timing) = timing.as_deref_mut() {
                    timing.wait_response += 1;
                }
                let (pids, n) = self.active_player_ids();
                for &pid in &pids[..n] {
                    self.legal_buf.clear();
                    let legal_start = Instant::now();
                    self.state.get_legal_actions_into(pid, &mut self.legal_buf);
                    if let Some(timing) = timing.as_deref_mut() {
                        timing.legal_actions += legal_start.elapsed();
                    }
                    if self.legal_buf.is_empty() {
                        continue;
                    }
                    decisions.push(self.pending_decision_for_player(pid, timing.as_deref_mut()));
                }
            }
        }
        if let Some(timing) = timing {
            timing.decisions += decisions.len() as u64;
        }
        Ok(decisions)
    }

    pub fn step_with_hydra_action_ids(&mut self, action_ids: &[u8]) -> StepOutcome {
        if let Err(outcome) = self.begin_action_step() {
            return outcome;
        }
        self.clear_pending_actions();
        let mut cursor = 0usize;
        match self.state.phase {
            Phase::WaitAct => {
                let pid = self.state.current_player;
                self.legal_buf.clear();
                self.state.get_legal_actions_into(pid, &mut self.legal_buf);
                if self.legal_buf.is_empty() {
                    return StepOutcome::NoLegalAction { player: pid };
                }
                let Some(&action_id) = action_ids.get(cursor) else {
                    return StepOutcome::NoLegalAction { player: pid };
                };
                let Some(chosen) = self.action_for_hydra_id(action_id) else {
                    return StepOutcome::NoLegalAction { player: pid };
                };
                self.track_action(pid, &chosen);
                self.actions[pid as usize] = Some(chosen);
            }
            Phase::WaitResponse => {
                let (pids, n) = self.active_player_ids();
                for &pid in &pids[..n] {
                    self.legal_buf.clear();
                    self.state.get_legal_actions_into(pid, &mut self.legal_buf);
                    if self.legal_buf.is_empty() {
                        continue;
                    }
                    let Some(&action_id) = action_ids.get(cursor) else {
                        return StepOutcome::NoLegalAction { player: pid };
                    };
                    cursor += 1;
                    let Some(chosen) = self.action_for_hydra_id(action_id) else {
                        return StepOutcome::NoLegalAction { player: pid };
                    };
                    self.track_action(pid, &chosen);
                    self.actions[pid as usize] = Some(chosen);
                }
            }
        }
        self.apply_pending_actions()
    }

    pub fn step_with_cached_legal_actions(
        &mut self,
        decisions: &[(CachedLegalActions, u8)],
    ) -> StepOutcome {
        if let Err(outcome) = self.begin_action_step() {
            return outcome;
        }
        self.clear_pending_actions();
        let mut cursor = 0usize;
        match self.state.phase {
            Phase::WaitAct => {
                let pid = self.state.current_player;
                let Some((cached, action_id)) = decisions.get(cursor) else {
                    return StepOutcome::NoLegalAction { player: pid };
                };
                if cached.turn != self.total_actions || cached.player_id != pid {
                    return StepOutcome::NoLegalAction { player: pid };
                }
                let Some(chosen) = Self::cached_action_for_hydra_id(cached, *action_id) else {
                    return StepOutcome::NoLegalAction { player: pid };
                };
                self.track_action(pid, &chosen);
                self.actions[pid as usize] = Some(chosen);
            }
            Phase::WaitResponse => {
                for (cached, action_id) in decisions {
                    let pid = cached.player_id;
                    if cached.turn != self.total_actions {
                        return StepOutcome::NoLegalAction { player: pid };
                    }
                    let Some(chosen) = Self::cached_action_for_hydra_id(cached, *action_id) else {
                        return StepOutcome::NoLegalAction { player: pid };
                    };
                    self.track_action(pid, &chosen);
                    self.actions[pid as usize] = Some(chosen);
                    cursor += 1;
                }
                if cursor == 0 {
                    return StepOutcome::NoLegalAction {
                        player: self.state.current_player,
                    };
                }
            }
        }
        self.apply_pending_actions()
    }

    fn pending_decision_for_player(
        &mut self,
        player: u8,
        mut timing: Option<&mut PendingDecisionTiming>,
    ) -> PendingDecision {
        let (obs, legal_mask) = self.encode_decision(player, timing.as_deref_mut());
        let legal_pack_start = Instant::now();
        let legal_count = legal_mask
            .iter()
            .filter(|&&legal| legal)
            .count()
            .min(u8::MAX as usize) as u8;
        let legal_actions = CachedLegalActions {
            turn: self.total_actions,
            player_id: player,
            by_hydra_id: legal_action_map(&self.legal_buf),
        };
        if let Some(timing) = timing {
            timing.legal_pack += legal_pack_start.elapsed();
        }
        PendingDecision {
            obs,
            legal_mask,
            legal_count,
            player_id: player,
            seat_id: player,
            turn: self.total_actions,
            legal_actions,
        }
    }

    fn action_for_hydra_id(&self, action_id: u8) -> Option<Action> {
        self.legal_buf.iter().copied().find(|action| {
            riichienv_to_hydra(action)
                .map(|hydra| hydra.id() == action_id)
                .unwrap_or(false)
        })
    }
}
