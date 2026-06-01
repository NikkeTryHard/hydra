use crate::action::{HYDRA_ACTION_SPACE, build_legal_mask, riichienv_to_hydra};
use crate::bridge::encode_observation_ref;
use crate::encoder::OBS_SIZE;
use riichienv_core::action::Action;
use std::time::Instant;

use super::pending::PendingDecisionTiming;
use super::record::{DecisionRecord, DecisionRecorder};
use super::runner::GameRunner;

impl GameRunner {
    pub(super) fn encode_decision(
        &mut self,
        player: u8,
        mut timing: Option<&mut PendingDecisionTiming>,
    ) -> ([f32; OBS_SIZE], [bool; HYDRA_ACTION_SPACE]) {
        let observe_start = Instant::now();
        let obs = self.state.observe(player);
        if let Some(timing) = timing.as_deref_mut() {
            timing.observe += observe_start.elapsed();
        }
        let encode_start = Instant::now();
        let encoded =
            encode_observation_ref(&mut self.encoder, &obs, &self.safety[player as usize]);
        if let Some(timing) = timing.as_deref_mut() {
            timing.encode += encode_start.elapsed();
        }
        let legal_pack_start = Instant::now();
        let legal_mask = build_legal_mask(&self.legal_buf, crate::action::ActionPhase::Normal);
        if let Some(timing) = timing {
            timing.legal_pack += legal_pack_start.elapsed();
        }
        (encoded, legal_mask)
    }

    pub(super) fn record_encoded_decision<R: DecisionRecorder>(
        &mut self,
        player: u8,
        encoded: [f32; OBS_SIZE],
        legal_mask: [bool; HYDRA_ACTION_SPACE],
        action: &Action,
        recorder: &mut R,
    ) {
        let legal_count = legal_mask
            .iter()
            .filter(|&&legal| legal)
            .count()
            .min(u8::MAX as usize) as u8;
        let Ok(hydra_action) = riichienv_to_hydra(action) else {
            return;
        };
        recorder.record(DecisionRecord {
            obs: encoded,
            legal_mask,
            action: hydra_action.id(),
            legal_count,
            player_id: player,
            seat_id: player,
            turn: self.total_actions,
        });
    }
}
