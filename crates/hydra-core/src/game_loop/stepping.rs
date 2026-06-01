use riichienv_core::action::Phase;

use super::outcome::StepOutcome;
use super::record::{DecisionRecord, DecisionRecorder, NoopDecisionRecorder};
use super::runner::GameRunner;
use super::selector::{ActionDecision, ActionSelector};

impl GameRunner {
    /// Advance the game by one step. Returns false if game is over or cannot advance.
    pub fn step_once<S: ActionSelector>(&mut self, selector: &mut S) -> bool {
        self.step_once_checked(selector).advanced()
    }

    /// Advance one step and report why iteration stopped.
    pub fn step_once_checked<S: ActionSelector>(&mut self, selector: &mut S) -> StepOutcome {
        let mut recorder = NoopDecisionRecorder;
        self.step_once_checked_with_recorder(selector, &mut recorder)
    }

    /// Advance one step and record every player decision before it is applied.
    pub fn step_once_recording<S, R>(&mut self, selector: &mut S, recorder: &mut R) -> StepOutcome
    where
        S: ActionSelector,
        R: FnMut(DecisionRecord),
    {
        self.step_once_checked_with_recorder(selector, recorder)
    }

    fn step_once_checked_with_recorder<S, R>(
        &mut self,
        selector: &mut S,
        recorder: &mut R,
    ) -> StepOutcome
    where
        S: ActionSelector,
        R: DecisionRecorder,
    {
        if let Err(outcome) = self.begin_action_step() {
            return outcome;
        }

        self.clear_pending_actions();

        match self.state.phase {
            Phase::WaitAct => {
                let pid = self.state.current_player;
                self.legal_buf.clear();
                self.state.get_legal_actions_into(pid, &mut self.legal_buf);
                if self.legal_buf.is_empty() {
                    return StepOutcome::NoLegalAction { player: pid };
                }
                let (encoded, legal_mask) = self.encode_decision(pid, None);
                selector.observe_decision(ActionDecision {
                    player: pid,
                    seat_id: pid,
                    obs: &encoded,
                    legal_mask: &legal_mask,
                    legal_actions: &self.legal_buf,
                    turn: self.total_actions,
                });
                let chosen = selector.select_action(pid, &self.legal_buf);
                self.record_encoded_decision(pid, encoded, legal_mask, &chosen, recorder);
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
                    let (encoded, legal_mask) = self.encode_decision(pid, None);
                    selector.observe_decision(ActionDecision {
                        player: pid,
                        seat_id: pid,
                        obs: &encoded,
                        legal_mask: &legal_mask,
                        legal_actions: &self.legal_buf,
                        turn: self.total_actions,
                    });
                    let chosen = selector.select_action(pid, &self.legal_buf);
                    self.record_encoded_decision(pid, encoded, legal_mask, &chosen, recorder);
                    self.track_action(pid, &chosen);
                    self.actions[pid as usize] = Some(chosen);
                }
            }
        }

        self.apply_pending_actions()
    }

    /// Run the full game to completion.
    pub fn run_to_completion<S: ActionSelector>(&mut self, selector: &mut S) -> StepOutcome {
        loop {
            match self.step_once_checked(selector) {
                StepOutcome::Advanced => {}
                outcome => return outcome,
            }
        }
    }
}
