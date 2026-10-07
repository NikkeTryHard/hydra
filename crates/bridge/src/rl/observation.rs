//! Actor-visible observations with two validation gates.
//!
//! This module is the public front for observation construction, validation,
//! tensor-plane geometry, per-row fill, and seal encoding. The heavy pieces
//! live in the child modules and are re-exported here so every public path
//! stays exactly where downstream users expect it:
//!
//! - `layout`: the ordered 27-plane table, the aligned batch geometry, and
//!   the splitter that carves one caller slab into per-plane views.
//! - `fill`: writing one validated observation into one row of plane views.
//! - `seal`: rebuilding the sealed document for a captured row and encoding
//!   it to the exact sealed feature bytes, by either the document path or
//!   the direct integer-facts path (both agree byte for byte).
//!
//! An observation carries only what the acting seat may see: its own
//! concealed tiles and drawn tile, public rivers and melds, revealed dora
//! indicators, per-seat riichi states, scores, counters, the visible history
//! slice, and the full-width legal mask. Anything concealed from the actor
//! (other seats' hidden tiles, wall order, unrevealed dora, private events,
//! opponent masks) is never stored here.
//!
//! Every observation passes two gates. First, the builder gate checks closed
//! slots: non-empty ids, hash digests with their prefix, seat winds forming
//! one exact permutation of the four winds, an ascending concealed hand, a
//! revealed-prefix dora shape, capped kan count, a full-width legal mask with
//! at least one legal action, a history within the 256-event cap whose every
//! event is visible to the actor. Second, the independent gate re-derives the
//! identity hash from the canonical identity bytes and requires equality, so
//! a tampered or misbound observation cannot pass.
//!
//! The 26 feature planes encode one row as exactly `7254 + 9 * T` bytes for
//! history bucket `T` (32, 64, 128, or 256). The live input slab appends one
//! trailing uniform plane after the frozen 26; sealed bytes never include it.
//! Over-cap histories fail closed; rows are never truncated. All failures
//! return descriptive text errors and substitute nothing.

mod fill;
mod identity;
mod layout;
pub mod packed;
mod seal;
#[cfg(test)]
mod tests;

pub use self::fill::fill_row;
pub use self::layout::{
    PlaneBuffers, PLANE_COUNT, active_feature_bytes, active_input_bytes, active_layout,
    plane_layout, split_planes,
};
pub use self::packed::{PACKED_CAP_QUANTUM, PACKED_MAX_ROWS, PACKED_MAX_TOKENS, packed_bytes, packed_layout};
pub use self::seal::{encode_canonical_doc, encode_row_facts, materialize_canonical};

use super::events::{EventArena, EventError, TypedEvent, visible_to_actor};
use crate::rl::actions::BASELINE_ACTIONS;

/// Native failure (fail-closed text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationError(pub String);

impl std::fmt::Display for ObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rl observation: {}", self.0)
    }
}

impl std::error::Error for ObservationError {}

impl From<EventError> for ObservationError {
    fn from(e: EventError) -> Self {
        ObservationError(e.0)
    }
}

/// History cap: at most 256 visible history events per row. Longer histories
/// fail closed instead of truncating.
pub const HISTORY_EVENT_CAP: usize = 256;
/// Dora width: exactly 5 indicator slots per row.
pub const DORA_WIDTH: usize = 5;
/// Unrevealed dora sentinel: -1 marks indicator slots with no revealed tile.
pub const DORA_SENTINEL: i32 = -1;

/// Exact 26-plane bytes per row: int64 planes cover actor/furiten/seats/
/// dealer/phase/round_wind/turn_actor scalars (7x8) plus the history-kind
/// prefix (8T) plus seat winds and riichi states (2x32); int32 planes cover
/// concealed counts (136) plus dora (20) plus 7 scalars (x4) plus scores (16)
/// plus visible counts (136); bool planes cover the two can-flags plus the
/// history mask (T) plus ippatsu (4) plus the legal mask (6792).
pub const fn row_bytes_26(bucket_t: usize) -> usize {
    7254 + 9 * bucket_t
}

/// Numeric id for a phase literal. The frozen ids are 0 round_start,
/// 1 draw_decision, 2 discard_response, 3 kan_response, 4 round_end,
/// 5 game_end. Unknown literals fail closed.
pub fn phase_id(phase: &str) -> Result<i64, ObservationError> {
    match phase {
        "round_start" => Ok(0),
        "draw_decision" => Ok(1),
        "discard_response" => Ok(2),
        "kan_response" => Ok(3),
        "round_end" => Ok(4),
        "game_end" => Ok(5),
        _ => Err(ObservationError(format!("unknown phase {phase:?}"))),
    }
}

/// Phase literal for a frozen id (inverse of `phase_id`). Ids outside 0..5
/// fail closed.
pub fn phase_name(id: u8) -> Result<&'static str, ObservationError> {
    match id {
        0 => Ok("round_start"),
        1 => Ok("draw_decision"),
        2 => Ok("discard_response"),
        3 => Ok("kan_response"),
        4 => Ok("round_end"),
        5 => Ok("game_end"),
        _ => Err(ObservationError(format!("phase id {id} out of range"))),
    }
}

/// Numeric id for an actor furiten state literal: 0 none, 1 temporary,
/// 2 riichi, 3 discard. Unknown literals fail closed.
pub fn furiten_id(state: &str) -> Result<i64, ObservationError> {
    match state {
        "none" => Ok(0),
        "temporary" => Ok(1),
        "riichi" => Ok(2),
        "discard" => Ok(3),
        _ => Err(ObservationError(format!("unknown furiten {state:?}"))),
    }
}

/// Furiten literal for a frozen id (inverse of `furiten_id`). Ids outside
/// 0..3 fail closed.
pub fn furiten_name(id: u8) -> Result<&'static str, ObservationError> {
    match id {
        0 => Ok("none"),
        1 => Ok("temporary"),
        2 => Ok("riichi"),
        3 => Ok("discard"),
        _ => Err(ObservationError(format!("furiten id {id} out of range"))),
    }
}

/// Numeric id for a riichi state literal: 0 none, 1 declared, 2 accepted.
/// Unknown literals fail closed.
pub fn riichi_id(state: &str) -> Result<i64, ObservationError> {
    match state {
        "none" => Ok(0),
        "declared" => Ok(1),
        "accepted" => Ok(2),
        _ => Err(ObservationError(format!("unknown riichi state {state:?}"))),
    }
}

/// Riichi literal for a frozen id (inverse of `riichi_id`). Ids outside 0..2
/// fail closed.
pub fn riichi_name(id: u8) -> Result<&'static str, ObservationError> {
    match id {
        0 => Ok("none"),
        1 => Ok("declared"),
        2 => Ok("accepted"),
        _ => Err(ObservationError(format!("riichi id {id} out of range"))),
    }
}

/// Wind tile 27..30 to seat-relative id 0..3. Tiles outside 27..30 fail
/// closed.
pub fn wind_id(tile: i64) -> Result<i64, ObservationError> {
    match tile {
        27 => Ok(0),
        28 => Ok(1),
        29 => Ok(2),
        30 => Ok(3),
        _ => Err(ObservationError(format!("wind tile {tile} not in 27..30"))),
    }
}

/// One visible meld in canonical-seat space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeMeld {
    pub kind: String,
    pub owner: u8,
    pub source: Option<u8>,
    pub called_tile: Option<u8>,
    pub tiles: Vec<u8>,
}

/// Actor-visible observation at one decision: game and decision identity,
/// digest bindings, round state, the actor's own hand, public discards and
/// melds, riichi states, the revealed dora prefix, the visible history slice,
/// and the full-width legal mask.
#[derive(Debug, Clone)]
pub struct NativeObservation {
    pub game_id: String,
    pub decision_id: String,
    pub sequence: u64,
    pub actor: u8,
    pub rules_id: String,
    pub rules_hash: String,
    pub action_table_hash: String,
    pub event_schema_hash: String,
    pub observation_schema_hash: String,
    pub packet_boundary_hash: String,
    pub round_index: u64,
    pub round_wind: i64,
    pub hand_number: u32,
    pub seat_winds: [i64; 4],
    pub honba: u32,
    pub riichi_sticks: u32,
    pub dealer: u8,
    pub scores: [i32; 4],
    pub turn_actor: u8,
    pub phase: String,
    pub live_wall_remaining: u32,
    pub kan_count: u32,
    pub ippatsu_active: [bool; 4],
    pub actor_furiten: String,
    pub actor_can_tsumo: bool,
    pub actor_can_riichi: bool,
    pub pending_declaration_discard: Option<u8>,
    pub concealed_hand: Vec<u8>,
    pub own_drawn_tile: Option<u8>,
    pub visible_discards: [Vec<u8>; 4],
    pub visible_melds: [Vec<NativeMeld>; 4],
    pub riichi_states: [String; 4],
    pub dora_indicators: Vec<i32>,
    pub history_indices: Vec<usize>,
    pub legal_mask: Vec<bool>,
    pub observation_hash: Option<[u8; 32]>,
}

impl NativeObservation {
    /// Builder-gate validation: every closed slot is checked against the
    /// actor boundary (non-empty ids, digest prefixes, wind permutation,
    /// ascending concealed hand, dora revealed-prefix shape, kan cap,
    /// full-width non-empty legal mask, history cap with per-event
    /// visibility). Any violation fails closed.
    pub fn validate(&self, arena: &EventArena) -> Result<(), ObservationError> {
        if self.game_id.is_empty() || self.decision_id.is_empty() || self.rules_id.is_empty() {
            return Err(ObservationError("game/decision/rules ids must be non-empty".to_string()));
        }
        for name in [
            &self.rules_hash,
            &self.action_table_hash,
            &self.event_schema_hash,
            &self.observation_schema_hash,
            &self.packet_boundary_hash,
        ] {
            if !name.starts_with("sha256:") {
                return Err(ObservationError(format!("digest {name:?} must carry sha256:")));
            }
        }
        let mut winds = self.seat_winds.to_vec();
        winds.sort_unstable();
        if winds != vec![27, 28, 29, 30] {
            return Err(ObservationError(
                "seat_winds must permute East/South/West/North aligned by seat".to_string(),
            ));
        }
        let mut sorted = self.concealed_hand.clone();
        sorted.sort_unstable();
        if sorted != self.concealed_hand {
            return Err(ObservationError(
                "concealed_hand must be ascending by physical TileId".to_string(),
            ));
        }
        // Dora contiguity: revealed prefix then sentinels only.
        let mut seen_sentinel = false;
        for d in &self.dora_indicators {
            if *d == DORA_SENTINEL {
                seen_sentinel = true;
            } else if seen_sentinel {
                return Err(ObservationError("dora indicators must be a revealed prefix".to_string()));
            }
        }
        if self.kan_count > 4 {
            return Err(ObservationError("kan_count above cap 4".to_string()));
        }
        if self.legal_mask.len() != BASELINE_ACTIONS {
            return Err(ObservationError(format!(
                "legal_mask length {} != baseline {BASELINE_ACTIONS}",
                self.legal_mask.len()
            )));
        }
        if !self.legal_mask.iter().any(|f| *f) {
            return Err(ObservationError(
                "legal_mask must contain at least one True at a decision".to_string(),
            ));
        }
        if self.history_indices.len() > HISTORY_EVENT_CAP {
            return Err(ObservationError(format!(
                "visible history {} exceeds model bucket cap {HISTORY_EVENT_CAP}; rows are never truncated",
                self.history_indices.len()
            )));
        }
        // Per-seat history filtering: every referenced event must be visible.
        for index in &self.history_indices {
            let event = arena.get(*index).ok_or_else(|| {
                ObservationError(format!("history index {index} outside arena"))
            })?;
            if !visible_to_actor(event, self.actor) {
                return Err(ObservationError(format!(
                    "history holds a {} event this actor may not see",
                    event.visibility.as_str()
                )));
            }
        }
        Ok(())
    }

    /// CURRENT non-history identity fields without the legal mask, shared by
    /// the full-document oracle (`identity_json` via `identity_nonhistory_map`)
    /// and the cached-bytes hash emitter. Same constructions, same `Value`
    /// shapes; the emitter must never drift from this map.
    /// `visible_history` is excluded (history reuses immutable cached envelope
    /// subbytes instead of rebuilt docs); `legal_mask` is excluded here so the
    /// emitter never allocates the full-width `Value` array.
    pub(crate) fn identity_base_map(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut meld_rows = Vec::with_capacity(4);
        for row in &self.visible_melds {
            let mut docs = Vec::with_capacity(row.len());
            for m in row {
                docs.push(super::events::meld_doc_value(
                    &m.kind,
                    m.owner,
                    m.source,
                    m.called_tile,
                    &m.tiles,
                ));
            }
            meld_rows.push(serde_json::Value::Array(docs));
        }
        let mut map = serde_json::Map::with_capacity(35);
        // Insertion order is irrelevant to the hash (`serde_jcs` sorts keys);
        // values below match the `identity_json` oracle exactly.
        map.insert("game_id".to_string(), serde_json::Value::String(self.game_id.clone()));
        map.insert("decision_id".to_string(), serde_json::Value::String(self.decision_id.clone()));
        map.insert("sequence".to_string(), serde_json::json!(self.sequence));
        map.insert("actor".to_string(), serde_json::json!(u32::from(self.actor)));
        map.insert("rules_id".to_string(), serde_json::Value::String(self.rules_id.clone()));
        map.insert("rules_hash".to_string(), serde_json::Value::String(self.rules_hash.clone()));
        map.insert("action_table_hash".to_string(), serde_json::Value::String(self.action_table_hash.clone()));
        map.insert("event_schema_hash".to_string(), serde_json::Value::String(self.event_schema_hash.clone()));
        map.insert("observation_schema_hash".to_string(), serde_json::Value::String(self.observation_schema_hash.clone()));
        map.insert("packet_boundary_hash".to_string(), serde_json::Value::String(self.packet_boundary_hash.clone()));
        map.insert("round_index".to_string(), serde_json::json!(self.round_index));
        map.insert("round_wind".to_string(), serde_json::json!(self.round_wind));
        map.insert("hand_number".to_string(), serde_json::json!(self.hand_number));
        map.insert("seat_winds".to_string(), serde_json::json!(self.seat_winds.to_vec()));
        map.insert("honba".to_string(), serde_json::json!(self.honba));
        map.insert("riichi_sticks".to_string(), serde_json::json!(self.riichi_sticks));
        map.insert("dealer".to_string(), serde_json::json!(u32::from(self.dealer)));
        map.insert("scores".to_string(), serde_json::json!(self.scores.to_vec()));
        map.insert("turn_actor".to_string(), serde_json::json!(u32::from(self.turn_actor)));
        map.insert("phase".to_string(), serde_json::Value::String(self.phase.clone()));
        map.insert("live_wall_tiles_remaining".to_string(), serde_json::json!(self.live_wall_remaining));
        map.insert("kan_count".to_string(), serde_json::json!(self.kan_count));
        map.insert("ippatsu_active".to_string(), serde_json::json!(self.ippatsu_active.to_vec()));
        map.insert("actor_furiten".to_string(), serde_json::Value::String(self.actor_furiten.clone()));
        map.insert("actor_can_tsumo".to_string(), serde_json::Value::Bool(self.actor_can_tsumo));
        map.insert("actor_can_riichi".to_string(), serde_json::Value::Bool(self.actor_can_riichi));
        map.insert("pending_declaration_discard".to_string(), serde_json::json!(self.pending_declaration_discard.map(u32::from)));
        map.insert("concealed_hand".to_string(), serde_json::json!(self.concealed_hand.iter().map(|t| u32::from(*t)).collect::<Vec<_>>()));
        map.insert("own_drawn_tile".to_string(), serde_json::json!(self.own_drawn_tile.map(u32::from)));
        map.insert("visible_discards".to_string(), serde_json::json!(self.visible_discards.iter().map(|r| r.iter().map(|t| u32::from(*t)).collect::<Vec<_>>()).collect::<Vec<_>>()));
        map.insert("visible_melds".to_string(), serde_json::Value::Array(meld_rows));
        map.insert("riichi_states".to_string(), serde_json::json!(self.riichi_states.to_vec()));
        map.insert("dora_indicators".to_string(), serde_json::json!(self.dora_indicators));
        map
    }

    /// CURRENT non-history identity fields: exact `identity_base_map` plus
    /// the CURRENT `legal_mask` (`Value` bool array, same shape as the
    /// `identity_json` oracle below). The hash emitter uses the base map plus
    /// memo-fed legal bytes instead of this array allocation.
    pub(crate) fn identity_nonhistory_map(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = self.identity_base_map();
        map.insert("legal_mask".to_string(), serde_json::json!(self.legal_mask));
        map
    }

    /// Identity document in the exact hash-input field order (the bound hash
    /// itself is excluded from its own input). Exact full-document
    /// oracle/export path: non-history fields come from the shared helper so
    /// the emitter cannot drift, history docs are rebuilt here (never
    /// optimized).
    pub fn identity_json(&self, arena: &EventArena) -> Result<serde_json::Value, ObservationError> {
        let mut map = self.identity_nonhistory_map();
        let mut history = Vec::with_capacity(self.history_indices.len());
        for index in &self.history_indices {
            let event: &TypedEvent = arena.get(*index).ok_or_else(|| {
                ObservationError(format!("history index {index} outside arena"))
            })?;
            history.push(event.to_json_value());
        }
        map.insert(
            "visible_history".to_string(),
            serde_json::Value::Array(history),
        );
        Ok(serde_json::Value::Object(map))
    }

/// Bind `observation_hash`: sha256 over the streamed CURRENT identity root
/// (`identity::hash_current_via_cache`): CURRENT scalar values plus CURRENT
/// legal bits through authority-derived atom fragments plus CURRENT
/// referenced immutable event bytes, rehashed from scratch through a fresh
/// `Sha256`. The stored hash is what the independent gate re-derives and
/// compares. A CURRENT legal mutation (even one keeping some mask `True`)
/// rehashes from the mutated mask and therefore mismatches the old bound
/// hash.
    pub fn bind_hash(&mut self, arena: &EventArena) -> Result<[u8; 32], ObservationError> {
        let hash = identity::hash_current_via_cache(self, arena)?;
        self.observation_hash = Some(hash);
        Ok(hash)
    }
}

/// Independent-gate validation: run the builder gate, then require a bound
/// hash and recompute the CURRENT identity root, rejecting any mismatch.
/// Every history event is re-checked against the actor boundary through the
/// builder gate, so a second independent pass over the same observation
/// cannot admit actor-invisible history. The recomputation re-derives
/// CURRENT fields plus cached subbytes via `identity::hash_current_via_cache`
/// — a second independent full-stream hash, never a reused
/// whole-observation digest. A CURRENT legal mutation (even one keeping some
/// mask `True`) rehashes from the mutated mask and therefore mismatches the
/// old bound hash.
pub fn independent_validate(
    obs: &NativeObservation,
    arena: &EventArena,
) -> Result<(), ObservationError> {
    obs.validate(arena)?;
    let recorded = obs.observation_hash.ok_or_else(|| {
        ObservationError("observation_hash must be bound before validation".to_string())
    })?;
    let recomputed: [u8; 32] = identity::hash_current_via_cache(obs, arena)?;
    if recorded != recomputed {
        return Err(ObservationError("observation_hash mismatch on re-derivation".to_string()));
    }
    Ok(())
}
/// Revealed dora indicators at one arena prefix, hand-scoped and in reveal
/// order with `-1` padding to width 5. The live `observe` path and the seal
/// `materialize` path MUST share this exact replay (same scope, same order,
/// same pad): the sealed bytes and the live slab planes agree byte for byte
/// only when both derive dora from `dora_revealed` events, never from the
/// wall (which already holds future, unrevealed indicators).
pub(crate) fn revealed_dora(arena: &EventArena, prefix: usize) -> Vec<i32> {
    let prefix = prefix.min(arena.len());
    let start = arena.hand_window_start(prefix);
    let mut dora = Vec::with_capacity(DORA_WIDTH);
    for index in start..prefix {
        if let Some(tile) = arena
            .get(index)
            .filter(|event| event.kind.as_str() == "dora_revealed")
            .and_then(|event| event.tile)
        {
            dora.push(i32::from(tile));
        }
    }
    while dora.len() < DORA_WIDTH {
        dora.push(DORA_SENTINEL);
    }
    dora
}
