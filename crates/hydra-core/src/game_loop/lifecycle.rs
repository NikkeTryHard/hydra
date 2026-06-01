use super::MAX_STEPS;
use super::outcome::StepOutcome;
use super::runner::GameRunner;

impl GameRunner {
    /// Check common terminal/limit/round-transition state before a public step API
    /// prepares player actions.
    pub(super) fn begin_action_step(&mut self) -> Result<(), StepOutcome> {
        if self.state.is_done {
            return Err(StepOutcome::Complete);
        }
        if self.total_actions >= MAX_STEPS {
            return Err(StepOutcome::StepLimitExceeded);
        }
        if self.state.needs_initialize_next_round {
            self.state.step_unchecked(&[None; 4]);
            self.rounds_played += 1;
            self.reset_round_scoped_tracking();
            return Err(if self.state.is_done {
                StepOutcome::Complete
            } else {
                StepOutcome::Advanced
            });
        }
        Ok(())
    }

    pub(super) fn reset_round_scoped_tracking(&mut self) {
        for safety in &mut self.safety {
            safety.reset();
        }
    }

    #[inline]
    pub(super) fn clear_pending_actions(&mut self) {
        self.actions = [None; 4];
    }

    pub(super) fn active_player_ids(&self) -> ([u8; 4], usize) {
        let n = self.state.active_player_count as usize;
        let mut pids = [0u8; 4];
        pids[..n].copy_from_slice(self.state.active_player_slice());
        (pids, n)
    }

    pub(super) fn apply_pending_actions(&mut self) -> StepOutcome {
        self.state.step_unchecked(&self.actions);
        self.total_actions += 1;
        if self.state.is_done {
            StepOutcome::Complete
        } else {
            StepOutcome::Advanced
        }
    }
}
