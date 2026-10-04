//! rl/core/tiles: exact tile resolution for event translation.
//!
//! The engine reports discards by river position and melds by current tiles;
//! this module attributes each envelope to one exact physical tile id
//! (0..=135, red fives distinct by id). Resolution order per discard is:
//! live river cursor, then exactly-one-missing-tile diff of the pre-step hand
//! versus the current hand, then the mjai-string fallback for discards only.
//! Claims read the last river tile; meld envelopes read the engine's latest
//! meld tiles. A missing responder, a missing river tile, or a missing meld
//! fails closed. No wall law, no RNG, no Python.

use hydra_feed::census::ActionKind;

use crate::rl::actions::{CanonicalAction, canonical_id};

use super::CoreError;

impl super::NativeGame {
    pub(super) fn stamp_if(&self, actor: u8, kind: ActionKind) -> Option<u32> {
        let stamp = self.stamp.as_ref()?;
        if stamp.actor != actor {
            return None;
        }
        if stamp.kind == kind
            || matches!(
                (stamp.kind, kind),
                (ActionKind::Discard, ActionKind::Tsumogiri)
                    | (ActionKind::Tsumogiri, ActionKind::Discard)
            )
            || (stamp.kind == ActionKind::RiichiDiscard
                && matches!(kind, ActionKind::Discard | ActionKind::Tsumogiri))
        {
            return Some(stamp.id);
        }
        None
    }

    pub(super) fn accepted_claim_id(&self, actor: u8) -> Result<u32, CoreError> {
        if let Some(buffered) = self.buffered.get(&actor) {
            return Ok(buffered.canonical_id);
        }
        for kind in [ActionKind::Chi, ActionKind::Pon, ActionKind::Daiminkan] {
            if let Some(id) = self.stamp_if(actor, kind) {
                return Ok(id);
            }
        }
        Err(CoreError(format!("claim event for seat {actor} without a buffered decision")))
    }

    pub(super) fn fallback_discard_id(&self, actor: u8, tile: u8) -> Result<u32, CoreError> {
        let probe = CanonicalAction {
            kind: ActionKind::Discard,
            actor,
            tile: Some(tile),
            called_tile: None,
            consumed: vec![],
            source: None,
            declares_riichi: false,
            prior_pon_id: None,
        };
        canonical_id(&probe).map_err(CoreError::from)
    }

    pub(super) fn next_discard(&mut self, engine_pid: u8, pai: &str) -> Result<u8, CoreError> {
        let river_len = self.state.players[engine_pid as usize].discards.len();
        let index = self.dahai_cursor[engine_pid as usize];
        if index < river_len {
            self.dahai_cursor[engine_pid as usize] = index + 1;
            return Ok(self.state.players[engine_pid as usize].discards[index]);
        }
        if let Some(pre) = self.pre_step_hands.get(&engine_pid).cloned()
            && let Some(single) = hand_diff_single(&pre, &self.state.players[engine_pid as usize].hand)
        {
            return Ok(single);
        }
        hydra_shard::tile::physical_of(pai)
            .map_err(|e| CoreError(format!("discard fallback rejected {pai:?}: {e:?}")))
    }

    pub(super) fn peek_discard(&self, engine_pid: u8, pai: &str) -> Result<u8, CoreError> {
        let river = &self.state.players[engine_pid as usize].discards;
        let index = self.dahai_cursor[engine_pid as usize];
        if index < river.len() {
            return Ok(river[index]);
        }
        if let Some(pre) = self.pre_step_hands.get(&engine_pid)
            && let Some(single) =
                hand_diff_single(pre, &self.state.players[engine_pid as usize].hand)
        {
            return Ok(single);
        }
        Err(CoreError(format!(
            "cannot resolve declaration discard of {pai:?} on engine seat {engine_pid}"
        )))
    }

    pub(super) fn last_river_tile(&self, engine_pid: u8) -> Result<u8, CoreError> {
        self.state.players[engine_pid as usize]
            .discards
            .last()
            .copied()
            .ok_or_else(|| CoreError(format!("seat {engine_pid} has no discard to claim")))
    }

    pub(super) fn latest_meld_tiles(&self, engine_pid: u8) -> Result<Vec<u8>, CoreError> {
        self.state.players[engine_pid as usize]
            .melds
            .last()
            .map(|m| m.tiles.clone())
            .ok_or_else(|| CoreError(format!("seat {engine_pid} has no meld to translate")))
    }
}

/// Exactly-one-missing-tile diff of pre-step vs current hand (exact physical
/// ids; mjai strings cannot disambiguate plain five copies).
fn hand_diff_single(pre: &[u8], post: &[u8]) -> Option<u8> {
    let mut rest = {
        let mut sorted = post.to_vec();
        sorted.sort_unstable();
        sorted
    };
    let mut missing = vec![];
    for t in pre {
        if let Some(pos) = rest.iter().position(|x| x == t) {
            rest.remove(pos);
        } else {
            missing.push(*t);
        }
    }
    if missing.len() == 1 {
        Some(missing[0])
    } else {
        None
    }
}
