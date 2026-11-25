//! Synthetic observation, action, and wire-buffer fixtures.

use crate::view::{EventArena, Observation as NativeObservation, Snapshot};
use hydra_rules::census::ActionKind;
use hydra_rules::rl::actions::{CanonicalAction, canonical_id};

use super::super::super::{ACTION, TOKENS, WIDTH, encode};

pub(crate) fn observation(actor: u8, phase: &'static str) -> NativeObservation {
    NativeObservation {
        actor,
        round_index: 0,
        round_wind: 27,
        hand_number: 1,
        seat_winds: [27, 28, 29, 30],
        honba: 0,
        riichi_sticks: 0,
        dealer: 0,
        scores: [25_000; 4],
        turn_actor: actor,
        phase,
        live_wall_remaining: 70,
        kan_count: 0,
        ippatsu_active: [false; 4],
        actor_furiten: "none",
        actor_can_tsumo: false,
        actor_can_riichi: false,
        concealed_hand: Vec::new(),
        own_drawn_tile: None,
        meld_counts: [0; 4],
        riichi_states: ["none"; 4],
        dora_indicators: [-1; 5],
        history_indices: Vec::new(),
    }
}

pub(crate) fn action(
    kind: ActionKind,
    tile: Option<u8>,
    called: Option<u8>,
    consumed: &[u8],
    source: Option<u8>,
) -> CanonicalAction {
    CanonicalAction {
        kind,
        actor: 0,
        tile,
        called_tile: called,
        consumed: consumed.to_vec(),
        source,
        declares_riichi: kind == ActionKind::RiichiDiscard,
        prior_pon_id: None,
    }
}

pub(crate) fn legal(actions: Vec<CanonicalAction>) -> Vec<(u32, CanonicalAction)> {
    actions
        .into_iter()
        .map(|action| {
            let id = canonical_id(&action)
                .unwrap_or_else(|error| panic!("test action absent from census: {error}"));
            (id, action)
        })
        .collect()
}

pub(crate) fn encode_pair(
    obs: &NativeObservation,
    arena: &EventArena,
    legal: &[(u32, CanonicalAction)],
) -> (Vec<f32>, Vec<i32>) {
    let snapshot = Snapshot {
        observation: obs.clone(),
        events: arena,
    };
    let mut tokens = vec![0.0; TOKENS * WIDTH];
    let mut candidates = vec![0; legal.len() * ACTION];
    encode(&snapshot, legal, &mut tokens, &mut candidates)
        .unwrap_or_else(|error| panic!("encode failed: {error}"));
    (tokens, candidates)
}

pub(crate) fn tile(tokens: &[f32], face: usize, column: usize) -> f32 {
    tokens[face * WIDTH + column]
}

pub(crate) fn record(tokens: &[f32], face: usize, slot: usize, field: usize) -> f32 {
    tokens[face * WIDTH + 11 + slot * 9 + field]
}

pub(crate) fn descriptor(candidates: &[i32], row: usize, column: usize) -> i32 {
    candidates[row * ACTION + column]
}
