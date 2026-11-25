//! Canonical-to-engine action bridge and legal-view construction.

use std::collections::HashMap;

use hydra_rules::census::ActionKind;
use hydra_rules::rl::actions::{
    CanonicalAction, canonical_id, engine_matches_canonical, expand_engine_legals,
};
use riichienv_core::action::{Action, ActionType, Phase};
use riichienv_core::check_riichi_candidates;
use riichienv_core::state::GameState;
use riichienv_core::state::legal_actions::GameStateLegalActions;
use riichienv_core::types::MeldType;

use super::Game;

impl Game {
    pub(in crate::arena) fn legal_view(
        &self,
        seat: u8,
    ) -> Result<Vec<(u32, CanonicalAction)>, String> {
        let player = self
            .state
            .players
            .get(usize::from(seat))
            .ok_or_else(|| format!("legal view seat {seat} is outside 0..4"))?;
        let engine = self.state._get_legal_actions_internal(seat);
        let offered = match self.state.phase {
            Phase::WaitResponse => Some(
                self.state
                    .last_discard
                    .map(|(source, _)| source)
                    .ok_or_else(|| "response window has no discard source".to_owned())?,
            ),
            Phase::WaitAct => None,
        };
        let melds = actor_melds(&self.state, seat);
        let candidates = if engine
            .iter()
            .any(|action| action.action_type == ActionType::Riichi)
        {
            check_riichi_candidates(player.hand.clone())
        } else {
            Vec::new()
        };
        let expanded = expand_engine_legals(
            &engine,
            seat,
            phase_name(&self.state),
            self.state.drawn_tile,
            &player.hand,
            offered,
            &melds,
            &candidates,
        )
        .map_err(|error| error.to_string())?;
        sorted_legal_actions(expanded)
    }

    pub(super) fn resolve_engine_action(
        &self,
        seat: u8,
        wanted: &CanonicalAction,
    ) -> Result<Action, String> {
        if wanted.actor != seat {
            return Err(format!(
                "canonical actor {} does not match seat {seat}",
                wanted.actor
            ));
        }
        let candidates = if wanted.kind == ActionKind::RiichiDiscard {
            check_riichi_candidates(self.state.players[usize::from(seat)].hand.clone())
        } else {
            Vec::new()
        };
        self.state
            ._get_legal_actions_internal(seat)
            .into_iter()
            .find(|engine| engine_matches_canonical(engine, wanted, &candidates))
            .ok_or_else(|| {
                format!(
                    "canonical {:?} has no engine counterpart for seat {seat}",
                    wanted.kind
                )
            })
    }

    /// A riichi declaration needs the engine's declaration step followed by
    /// its discard step. The engine's legal declaration carries no tile and
    /// rejects a tile payload, so history observes both logs once here.
    pub(super) fn apply_riichi(
        &mut self,
        seat: u8,
        action: &CanonicalAction,
    ) -> Result<(), String> {
        if self.state.current_player != seat {
            return Err(format!("seat {seat} is not the riichi actor"));
        }
        let tile = action
            .tile
            .ok_or_else(|| "riichi discard has no tile".to_owned())?;
        let hand = self
            .state
            .players
            .get(usize::from(seat))
            .ok_or_else(|| format!("riichi seat {seat} is outside 0..4"))?
            .hand
            .clone();
        if !check_riichi_candidates(hand).contains(&u32::from(tile)) {
            return Err(format!(
                "tile {tile} is not a riichi candidate for seat {seat}"
            ));
        }
        if !self
            .state
            ._get_legal_actions_internal(seat)
            .iter()
            .any(|engine| engine.action_type == ActionType::Riichi)
        {
            return Err(format!("engine does not offer riichi to seat {seat}"));
        }

        let declaration = Action::new(ActionType::Riichi, None, Vec::new(), Some(seat));
        let mut step = HashMap::with_capacity(1);
        step.insert(seat, declaration);
        self.state.step(&step);
        if let Some(error) = &self.state.last_error {
            return Err(error.clone());
        }

        let Some(discard) = self
            .state
            ._get_legal_actions_internal(seat)
            .into_iter()
            .find(|engine| engine.action_type == ActionType::Discard && engine.tile == Some(tile))
        else {
            return Err(format!(
                "riichi declaration left no discard for tile {tile}"
            ));
        };
        step.clear();
        step.insert(seat, discard);
        self.state.step(&step);
        self.history.update(&self.state, &[action])?;
        self.window = None;
        self.finish_step()
    }
}

fn sorted_legal_actions(
    expanded: Vec<CanonicalAction>,
) -> Result<Vec<(u32, CanonicalAction)>, String> {
    let mut actions = Vec::with_capacity(expanded.len());
    for action in expanded {
        let id = canonical_id(&action).map_err(|error| error.to_string())?;
        actions.push((id, action));
    }
    actions.sort_unstable_by_key(|(id, _)| *id);
    for pair in actions.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(format!("duplicate canonical action id {}", pair[0].0));
        }
    }
    Ok(actions)
}

fn phase_name(state: &GameState) -> &'static str {
    match state.phase {
        Phase::WaitAct => "draw_decision",
        Phase::WaitResponse if state.pending_kan.is_some() => "kan_response",
        Phase::WaitResponse => "discard_response",
    }
}

fn actor_melds(state: &GameState, seat: u8) -> Vec<(String, Vec<u8>)> {
    state
        .players
        .get(usize::from(seat))
        .map_or_else(Vec::new, |player| {
            player
                .melds
                .iter()
                .map(|meld| (meld_name(meld.meld_type).to_owned(), meld.tiles.clone()))
                .collect()
        })
}

fn meld_name(kind: MeldType) -> &'static str {
    match kind {
        MeldType::Chi => "chi",
        MeldType::Pon => "pon",
        MeldType::Daiminkan => "daiminkan",
        MeldType::Ankan => "ankan",
        MeldType::Kakan => "kakan",
    }
}
