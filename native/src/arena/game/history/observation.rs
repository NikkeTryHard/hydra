use hydra_rules::census::ActionKind;
use hydra_rules::rl::actions::CanonicalAction;
use riichienv_core::action::Phase;
use riichienv_core::state::GameState;

use super::History;
use crate::view::{HISTORY_EVENT_CAP, Observation};

impl History {
    pub(in crate::arena::game) fn observe(
        &self,
        state: &GameState,
        seat: u8,
        legal: &[(u32, CanonicalAction)],
    ) -> Result<Observation, String> {
        let slot = usize::from(seat);
        let player = state
            .players
            .get(slot)
            .ok_or("observation seat outside 0..4")?;
        let phase = match state.phase {
            Phase::WaitAct => "draw_decision",
            Phase::WaitResponse if state.pending_kan.is_some() => "kan_response",
            Phase::WaitResponse => "discard_response",
        };
        let mut concealed_hand = player.hand.clone();
        // Only a real current draw moves from concealed to drawn. After a
        // call, the previous private draw may still be in the concealed hand.
        if state.phase == Phase::WaitAct
            && let Some(tile) = state.drawn_tile
        {
            concealed_hand.retain(|&id| id != tile);
        }
        let history_indices = self
            .events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| event.visible_to(seat).then_some(index))
            .collect::<Vec<_>>();
        if history_indices.len() > HISTORY_EVENT_CAP {
            return Err(format!(
                "visible history exceeds {HISTORY_EVENT_CAP} events"
            ));
        }
        let has = |kind| legal.iter().any(|(_, action)| action.kind == kind);
        Ok(Observation {
            actor: seat,
            round_index: self.round_index,
            round_wind: 27 + i64::from(state.round_wind),
            hand_number: u32::from(state.kyoku_idx) + 1,
            seat_winds: std::array::from_fn(|i| 27 + ((i + 4 - usize::from(state.oya)) % 4) as i64),
            honba: u32::from(state.honba),
            riichi_sticks: state.riichi_sticks,
            dealer: state.oya,
            scores: std::array::from_fn(|i| state.players[i].score),
            turn_actor: state.current_player,
            phase,
            live_wall_remaining: u32::from(state.wall.drawable_count),
            kan_count: self.kans,
            ippatsu_active: std::array::from_fn(|i| state.players[i].ippatsu_cycle),
            actor_furiten: if player.missed_agari_riichi {
                "riichi"
            } else if player.missed_agari_doujun {
                "temporary"
            } else {
                "none"
            },
            actor_can_tsumo: has(ActionKind::Tsumo),
            actor_can_riichi: has(ActionKind::RiichiDiscard),
            concealed_hand,
            own_drawn_tile: self.draws[slot],
            meld_counts: std::array::from_fn(|i| state.players[i].melds.len()),
            riichi_states: self.riichi,
            dora_indicators: self.dora,
            history_indices,
        })
    }
}
