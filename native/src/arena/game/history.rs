//! Translate the engine's current-hand progression without inspecting opponents' hands.

mod observation;

use hydra_rules::census::ActionKind;
use hydra_rules::rl::actions::{CanonicalAction, canonical_id};
use riichienv_core::state::GameState;
use serde::Deserialize;

use crate::view::{EventArena, TypedEvent, Visibility};

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "type", borrow)]
    kind: &'a str,
    actor: Option<u8>,
}

#[derive(Default)]
pub(super) struct History {
    events: EventArena,
    cursor: usize,
    round_index: u64,
    draws: [Option<u8>; 4],
    riichi: [&'static str; 4],
    dora: [i32; 5],
    dora_count: usize,
    kans: u32,
}

impl History {
    pub(super) fn new(state: &GameState) -> Result<Self, String> {
        let mut history = Self::default();
        history.update(state, &[])?;
        Ok(history)
    }

    pub(super) fn update(
        &mut self,
        state: &GameState,
        applied: &[&CanonicalAction],
    ) -> Result<(), String> {
        let mut entries = state
            .mjai_log
            .get(self.cursor..)
            .ok_or_else(|| "engine log cursor is outside the log".to_owned())?;
        // The engine can settle a hand and deal the next one in one step.
        // Drop the settled hand's tail before consulting the new tile state.
        if state.wall.hand_index > self.round_index + 1 {
            let start = entries
                .iter()
                .enumerate()
                .rev()
                .find_map(|(i, entry)| {
                    let record: Record<'_> = serde_json::from_str(entry).ok()?;
                    (record.kind == "start_kyoku").then_some(i)
                })
                .ok_or("engine advanced without a hand start")?;
            entries = &entries[start..];
        }
        for entry in entries {
            let record: Record<'_> = serde_json::from_str(entry).map_err(|e| e.to_string())?;
            let actor = || {
                record
                    .actor
                    .filter(|&seat| seat < 4)
                    .ok_or_else(|| format!("{} has no valid actor", record.kind))
            };
            match record.kind {
                "start_game" => {}
                "start_kyoku" => self.start_hand(state)?,
                // Terminal actions can open a new hand in the same step. Do not
                // use the new hand's tile state to reconstruct the finished one.
                "hora" | "ryukyoku" | "end_kyoku" | "end_game" => {}
                "tsumo" => {
                    let seat = actor()?;
                    let tile = state.drawn_tile.ok_or("draw has no physical tile")?;
                    self.events
                        .push(TypedEvent::public("turn_advance", Some(seat)));
                    let mut event = TypedEvent::public("draw_tile", Some(seat));
                    event.visibility = Visibility::ActorPrivate;
                    event.tile = Some(tile);
                    self.events.push(event);
                    self.draws[usize::from(seat)] = Some(tile);
                }
                "dora" => self.reveal_dora(state)?,
                "reach" | "dahai" => {
                    let seat = actor()?;
                    let action = find_action(applied, seat, |kind| {
                        matches!(
                            kind,
                            ActionKind::Discard | ActionKind::Tsumogiri | ActionKind::RiichiDiscard
                        )
                    })?;
                    let kind = if record.kind == "reach" {
                        self.riichi[usize::from(seat)] = "declared";
                        "riichi_declared"
                    } else {
                        "discard"
                    };
                    self.events.push(action_event(kind, action)?);
                }
                "reach_accepted" => {
                    let seat = actor()?;
                    self.riichi[usize::from(seat)] = "accepted";
                    self.events
                        .push(TypedEvent::public("riichi_accepted", Some(seat)));
                }
                "chi" | "pon" | "daiminkan" | "ankan" | "kakan" => {
                    let seat = actor()?;
                    let (kind, tag) = match record.kind {
                        "chi" => (ActionKind::Chi, "chi"),
                        "pon" => (ActionKind::Pon, "pon"),
                        "daiminkan" => (ActionKind::Daiminkan, "daiminkan"),
                        "ankan" => (ActionKind::Ankan, "ankan"),
                        _ => (ActionKind::Kakan, "kakan"),
                    };
                    let action = find_action(applied, seat, |offered| offered == kind)?;
                    let event = action_event(tag, action)?;
                    self.events.push(event);
                    if matches!(
                        kind,
                        ActionKind::Daiminkan | ActionKind::Ankan | ActionKind::Kakan
                    ) {
                        self.kans += 1;
                    }
                }
                other => return Err(format!("unexpected engine event {other:?}")),
            }
        }
        self.cursor = state.mjai_log.len();
        Ok(())
    }

    fn start_hand(&mut self, state: &GameState) -> Result<(), String> {
        self.events.clear();
        self.round_index = state
            .wall
            .hand_index
            .checked_sub(1)
            .ok_or("no seeded hand")?;
        self.draws = [None; 4];
        self.riichi = ["none"; 4];
        self.dora = [-1; 5];
        self.dora_count = 0;
        self.kans = 0;
        if self.round_index == 0 {
            self.events.push(TypedEvent::public("game_start", None));
        }
        self.events
            .push(TypedEvent::public("round_start", Some(state.oya)));
        self.reveal_dora(state)
    }

    fn reveal_dora(&mut self, state: &GameState) -> Result<(), String> {
        let tile = *state
            .wall
            .dora_indicators
            .get(self.dora_count)
            .ok_or("dora reveal has no physical tile")?;
        *self
            .dora
            .get_mut(self.dora_count)
            .ok_or("more than five dora indicators")? = i32::from(tile);
        self.dora_count += 1;
        let mut event = TypedEvent::public("dora_revealed", None);
        event.tile = Some(tile);
        self.events.push(event);
        Ok(())
    }
}

fn find_action<'a>(
    applied: &[&'a CanonicalAction],
    seat: u8,
    matches: impl Fn(ActionKind) -> bool,
) -> Result<&'a CanonicalAction, String> {
    applied
        .iter()
        .copied()
        .find(|a| a.actor == seat && matches(a.kind))
        .ok_or_else(|| format!("event for seat {seat} has no matching applied action"))
}

fn action_event(kind: &'static str, action: &CanonicalAction) -> Result<TypedEvent, String> {
    let mut event = TypedEvent::public(kind, Some(action.actor));
    event.tile = action.tile.or(action.called_tile);
    event.action_id = Some(canonical_id(action).map_err(|error| error.to_string())?);
    event.source_seat = action.source;
    event.consumed_tiles.clone_from(&action.consumed);
    Ok(event)
}
