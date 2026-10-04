//! rl/events: typed canonical event arena, envelope JSON, visibility gates.
//!
//! DAG: `serde_json` (parsed mjai records + envelope docs) + `hydra_feed::canon`
//! (JCS) + `sha2` (digests). No engine, no Python, no wall law.
//! FORBIDS: mjai-string tile guessing (exact ids arrive from `super::core`
//! river/hand state), per-row history clones (rows ref prefix lengths).
//!
//! TABLE ported here (mirrors `contracts/event_envelope.py` +
//! `engines/riichienv/events.py` + adapter handlers):
//! - envelope/payload/delta key order <- `to_json` (`event_envelope.py:83-84,
//!   156-167, 230-242`); meld doc keys <- `meld_delta_value` (`events.py:82-100`)
//!   with `meld_id = kind:sorted.tiles`.
//! - visibility matrix <- `_validate_visibility_matrix` (`:253-267`);
//!   `visible_to_actor`/`filter_events_for_actor` <- (`:513-531`).
//! - per-kind payload shape <- `_validate_kind_shape` (`:296-470`).
//! - reason classes <- `reason_kind`/`ABORTIVE_REASONS`/`DRAW_END_REASONS`
//!   (`events.py:46-75`); stream grammar (monotonic seq, single game,
//!   `game_start` first) <- `validate_event_stream` (`:487-510`).
//! - history kind ids 0..20 <- `encoder.rs::event_kind_id` (unknown -> 0).

mod arena;

pub use arena::EventArena;

/// Native failure (fail-closed text).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventError(pub String);

impl std::fmt::Display for EventError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rl events: {}", self.0)
    }
}

impl std::error::Error for EventError {}

/// Canonical history kind ids (mirrors `encoder.rs::event_kind_id`).
pub fn history_kind_id(kind: &str) -> i64 {
    match kind {
        "game_start" => 0,
        "round_start" => 1,
        "turn_advance" => 2,
        "draw_tile" => 3,
        "discard" => 4,
        "riichi_declared" => 5,
        "riichi_accepted" => 6,
        "call_window" => 7,
        "call_resolved" => 8,
        "chi" => 9,
        "pon" => 10,
        "daiminkan" => 11,
        "ankan" => 12,
        "kakan" => 13,
        "dora_revealed" => 14,
        "ron" => 15,
        "tsumo" => 16,
        "draw_end" => 17,
        "abortive_draw" => 18,
        "round_end" => 19,
        "game_end" => 20,
        _ => 0,
    }
}

/// Event visibility (closed vocabulary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Public,
    ActorPrivate,
    ServerPrivate,
}

impl Visibility {
    pub fn as_str(self) -> &'static str {
        match self {
            Visibility::Public => "public",
            Visibility::ActorPrivate => "actor_private",
            Visibility::ServerPrivate => "server_private",
        }
    }
}

/// One public-state delta (path/op/value; JSON order path, operation, value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateDelta {
    pub path: Vec<DeltaPathSeg>,
    pub operation: String,
    pub value: serde_json::Value,
}

/// Delta path segment (string key or seat index).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaPathSeg {
    Key(String),
    Index(u8),
}

/// One typed canonical event (append-only arena element).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedEvent {
    pub game_id: String,
    pub sequence: u64,
    pub kind: String,
    pub actor: Option<u8>,
    pub visibility: Visibility,
    pub visible_to: Vec<u8>,
    pub tile: Option<u8>,
    pub action_id: Option<u32>,
    pub source_seat: Option<u8>,
    pub consumed_tiles: Vec<u8>,
    pub offered_action_ids: Vec<u32>,
    pub accepted_action_ids: Vec<u32>,
    pub round_index: Option<u64>,
    pub scores: Option<[i32; 4]>,
    pub reason: Option<String>,
    pub public_delta: Vec<StateDelta>,
    pub rules_hash: String,
    pub schema_hash: String,
}

impl TypedEvent {
    /// Visibility matrix gate (mirrors `_validate_visibility_matrix`).
    pub fn check_visibility(&self) -> Result<(), EventError> {
        match self.visibility {
            Visibility::Public => {
                if self.visible_to != vec![0, 1, 2, 3] {
                    return Err(EventError(format!(
                        "{}: public events are visible to exactly (0, 1, 2, 3)",
                        self.kind
                    )));
                }
            }
            Visibility::ActorPrivate => {
                if self.visible_to.len() != 1 {
                    return Err(EventError(format!(
                        "{}: actor_private events must name exactly one seat",
                        self.kind
                    )));
                }
                if self.kind == "draw_tile" && Some(self.visible_to[0]) != self.actor {
                    return Err(EventError(
                        "draw_tile must be actor_private addressed to the drawing actor only"
                            .to_string(),
                    ));
                }
            }
            Visibility::ServerPrivate => {
                if !self.visible_to.is_empty() {
                    return Err(EventError(format!(
                        "{}: server_private events must have empty visible_to",
                        self.kind
                    )));
                }
                if self.kind == "call_resolved" {
                    let offered = &self.offered_action_ids;
                    let accepted = &self.accepted_action_ids;
                    if !offered.is_empty() {
                        let mut distinct = offered.clone();
                        distinct.sort_unstable();
                        distinct.dedup();
                        if distinct.len() != offered.len() {
                            return Err(EventError(
                                "call_resolved offered_action_ids must be distinct".to_string(),
                            ));
                        }
                        if accepted.len() != 1 || !offered.contains(&accepted[0]) {
                            return Err(EventError(
                                "call_resolved accepts exactly one offered action id".to_string(),
                            ));
                        }
                    } else if !accepted.is_empty() {
                        return Err(EventError(
                            "call_resolved pass resolution accepts no actions".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Per-kind payload composition gate (mirrors `_validate_kind_shape`).
    pub fn check_kind_shape(&self) -> Result<(), EventError> {
        let forbid_scalar = |names: &[&str]| -> Result<(), EventError> {
            for name in names {
                let present = match *name {
                    "actor" => self.actor.is_some(),
                    "tile" => self.tile.is_some(),
                    "action_id" => self.action_id.is_some(),
                    "source_seat" => self.source_seat.is_some(),
                    "round_index" => self.round_index.is_some(),
                    "scores" => self.scores.is_some(),
                    "reason" => self.reason.is_some(),
                    _ => false,
                };
                if present {
                    return Err(EventError(format!("{}: {name} must be null", self.kind)));
                }
            }
            Ok(())
        };
        let forbid_tuples = |names: &[&str]| -> Result<(), EventError> {
            for name in names {
                let len = match *name {
                    "consumed_tiles" => self.consumed_tiles.len(),
                    "offered_action_ids" => self.offered_action_ids.len(),
                    "accepted_action_ids" => self.accepted_action_ids.len(),
                    _ => 0,
                };
                if len > 0 {
                    return Err(EventError(format!("{}: {name} must be empty", self.kind)));
                }
            }
            Ok(())
        };
        let tuples = [
            "consumed_tiles",
            "offered_action_ids",
            "accepted_action_ids",
        ];
        match self.kind.as_str() {
            "turn_advance" => {
                self.require_actor()?;
                forbid_scalar(&[
                    "tile",
                    "action_id",
                    "source_seat",
                    "round_index",
                    "scores",
                    "reason",
                ])?;
                forbid_tuples(&tuples)?;
            }
            "draw_tile" => {
                self.require_actor()?;
                if self.tile.is_none() {
                    return Err(EventError("draw_tile: tile is required".to_string()));
                }
                forbid_scalar(&["action_id", "source_seat", "round_index", "scores", "reason"])?;
                forbid_tuples(&tuples)?;
            }
            // Discards and riichi declarations carry the actor's tile plus
            // the stamped canonical action; the call window resolves the
            // source seat later, so it stays null here.
            "discard" | "riichi_declared" => {
                self.require_actor()?;
                if self.tile.is_none() || self.action_id.is_none() {
                    return Err(EventError(format!(
                        "{}: actor, tile, action_id are required",
                        self.kind
                    )));
                }
                forbid_scalar(&["source_seat", "round_index", "scores", "reason"])?;
                forbid_tuples(&tuples)?;
            }
            "riichi_accepted" => {
                self.require_actor()?;
                forbid_scalar(&[
                    "tile",
                    "action_id",
                    "source_seat",
                    "round_index",
                    "scores",
                    "reason",
                ])?;
                forbid_tuples(&tuples)?;
            }
            "call_window" => {
                forbid_scalar(&[
                    "actor",
                    "tile",
                    "action_id",
                    "source_seat",
                    "round_index",
                    "scores",
                    "reason",
                ])?;
                forbid_tuples(&tuples)?;
            }
            "call_resolved" => {
                forbid_scalar(&[
                    "actor",
                    "tile",
                    "action_id",
                    "source_seat",
                    "round_index",
                    "scores",
                    "reason",
                    "consumed_tiles",
                ])?;
            }
            "chi" | "pon" => {
                self.require_actor()?;
                if self.tile.is_none() || self.action_id.is_none() || self.source_seat.is_none() {
                    return Err(EventError(format!(
                        "{}: actor, tile, action_id, source_seat are required",
                        self.kind
                    )));
                }
                if self.consumed_tiles.len() != 2 {
                    return Err(EventError(format!(
                        "{} consumes exactly two hand tiles",
                        self.kind
                    )));
                }
                forbid_scalar(&["round_index", "scores", "reason"])?;
                forbid_tuples(&["offered_action_ids", "accepted_action_ids"])?;
            }
            "daiminkan" => {
                self.require_actor()?;
                if self.tile.is_none() || self.action_id.is_none() || self.source_seat.is_none() {
                    return Err(EventError(
                        "daiminkan: actor, tile, action_id, source_seat are required".to_string(),
                    ));
                }
                if self.consumed_tiles.len() != 3 {
                    return Err(EventError(
                        "daiminkan consumes exactly three hand tiles".to_string(),
                    ));
                }
                forbid_scalar(&["round_index", "scores", "reason"])?;
                forbid_tuples(&["offered_action_ids", "accepted_action_ids"])?;
            }
            "ankan" => {
                self.require_actor()?;
                if self.action_id.is_none() {
                    return Err(EventError("ankan: actor, action_id are required".to_string()));
                }
                if self.consumed_tiles.len() != 4 {
                    return Err(EventError(
                        "ankan consumes exactly four concealed tiles".to_string(),
                    ));
                }
                forbid_scalar(&[
                    "tile",
                    "source_seat",
                    "round_index",
                    "scores",
                    "reason",
                ])?;
                forbid_tuples(&["offered_action_ids", "accepted_action_ids"])?;
            }
            "kakan" => {
                self.require_actor()?;
                if self.tile.is_none() || self.action_id.is_none() {
                    return Err(EventError("kakan: actor, tile, action_id are required".to_string()));
                }
                forbid_scalar(&[
                    "source_seat",
                    "consumed_tiles",
                    "round_index",
                    "scores",
                    "reason",
                ])?;
                forbid_tuples(&["offered_action_ids", "accepted_action_ids"])?;
            }
            "dora_revealed" => {
                if self.tile.is_none() {
                    return Err(EventError("dora_revealed: tile is required".to_string()));
                }
                forbid_scalar(&[
                    "actor",
                    "action_id",
                    "source_seat",
                    "round_index",
                    "scores",
                    "reason",
                ])?;
                forbid_tuples(&tuples)?;
            }
            "ron" => {
                self.require_actor()?;
                if self.tile.is_none() || self.action_id.is_none() || self.source_seat.is_none() {
                    return Err(EventError(
                        "ron: actor, tile, action_id, source_seat are required".to_string(),
                    ));
                }
                forbid_scalar(&["round_index", "scores", "reason"])?;
                forbid_tuples(&tuples)?;
            }
            "tsumo" => {
                self.require_actor()?;
                if self.tile.is_none() || self.action_id.is_none() {
                    return Err(EventError(
                        "tsumo: actor, tile, action_id are required".to_string(),
                    ));
                }
                forbid_scalar(&["source_seat", "round_index", "scores", "reason"])?;
                forbid_tuples(&tuples)?;
            }
            "game_start" => {
                if self.round_index.is_none() || self.scores.is_none() {
                    return Err(EventError(
                        "game_start: round_index, scores are required".to_string(),
                    ));
                }
                forbid_scalar(&["actor", "tile", "action_id", "source_seat", "reason"])?;
                forbid_tuples(&tuples)?;
            }
            "round_start" => {
                self.require_actor()?;
                if self.round_index.is_none() || self.scores.is_none() {
                    return Err(EventError(
                        "round_start: actor, round_index, scores are required".to_string(),
                    ));
                }
                forbid_scalar(&["tile", "action_id", "source_seat", "reason"])?;
                forbid_tuples(&tuples)?;
            }
            "round_end" => {
                if self.round_index.is_none() || self.scores.is_none() {
                    return Err(EventError(
                        "round_end: round_index, scores are required".to_string(),
                    ));
                }
                forbid_scalar(&["actor", "tile", "action_id", "source_seat", "reason"])?;
                forbid_tuples(&tuples)?;
            }
            // Hand terminations without a winner: abortives carry the hand
            // ordinal plus the classified reason; plain draws carry only
            // scores and the reason.
            "abortive_draw" => {
                if self.round_index.is_none() || self.scores.is_none() || self.reason.is_none() {
                    return Err(EventError(
                        "abortive_draw: round_index, scores, reason are required".to_string(),
                    ));
                }
                forbid_scalar(&["actor", "tile", "action_id", "source_seat"])?;
                forbid_tuples(&tuples)?;
            }
            "draw_end" => {
                if self.scores.is_none() || self.reason.is_none() {
                    return Err(EventError(
                        "draw_end: scores, reason are required".to_string(),
                    ));
                }
                forbid_scalar(&["actor", "tile", "action_id", "source_seat", "round_index"])?;
                forbid_tuples(&tuples)?;
            }
            "game_end" => {
                if self.round_index.is_none() || self.scores.is_none() || self.reason.is_none() {
                    return Err(EventError(
                        "game_end: round_index, scores, reason are required".to_string(),
                    ));
                }
                forbid_scalar(&["actor", "tile", "action_id", "source_seat"])?;
                forbid_tuples(&tuples)?;
            }
            other => return Err(EventError(format!("unhandled event kind {other:?}"))),
        }
        Ok(())
    }

    fn require_actor(&self) -> Result<(), EventError> {
        if self.actor.is_none() {
            return Err(EventError(format!("{}: actor is required", self.kind)));
        }
        Ok(())
    }

    /// Envelope JSON in exact `to_json` key order (payload + delta nested).
    pub fn to_json_value(&self) -> serde_json::Value {
        let payload = serde_json::json!({
            "kind": self.kind,
            "actor": self.actor.map(u32::from),
            "tile": self.tile.map(u32::from),
            "action_id": self.action_id,
            "source_seat": self.source_seat.map(u32::from),
            "consumed_tiles": self.consumed_tiles.iter().map(|t| u32::from(*t)).collect::<Vec<_>>(),
            "offered_action_ids": self.offered_action_ids,
            "accepted_action_ids": self.accepted_action_ids,
            "round_index": self.round_index,
            "scores": self.scores.map(|s| s.to_vec()),
            "reason": self.reason,
        });
        let deltas = self.public_delta.iter().map(StateDelta::to_json_value).collect();
        let mut document = serde_json::json!({
            "game_id": self.game_id,
            "sequence": self.sequence,
            "kind": self.kind,
            "actor": self.actor.map(u32::from),
            "visibility": self.visibility.as_str(),
            "visible_to": self.visible_to.iter().map(|s| u32::from(*s)).collect::<Vec<_>>(),
            "payload": null,
            "public_delta": null,
            "rules_hash": self.rules_hash,
            "schema_hash": self.schema_hash,
        });
        // Moving staged subtrees preserves exact fields without reserializing
        // and copying every nested value on each history materialization.
        document["payload"] = payload;
        document["public_delta"] = serde_json::Value::Array(deltas);
        document
    }
}

impl StateDelta {
    pub fn to_json_value(&self) -> serde_json::Value {
        let path: Vec<serde_json::Value> = self
            .path
            .iter()
            .map(|seg| match seg {
                DeltaPathSeg::Key(k) => serde_json::Value::String(k.clone()),
                DeltaPathSeg::Index(i) => serde_json::Value::from(u32::from(*i)),
            })
            .collect();
        let mut document = serde_json::json!({
            "path": null,
            "operation": self.operation,
            "value": self.value,
        });
        // Moving the staged path avoids copying key strings a second time
        // during repeated history projection without changing delta identity.
        document["path"] = serde_json::Value::Array(path);
        document
    }
}

/// Meld document keys in `meld_delta_value` order.
pub fn meld_doc_value(
    kind: &str,
    owner: u8,
    source_seat: Option<u8>,
    called_tile: Option<u8>,
    tiles: &[u8],
) -> serde_json::Value {
    let mut ordered = tiles.to_vec();
    ordered.sort_unstable();
    serde_json::json!({
        "meld_id": super::actions::meld_id(kind, &ordered),
        "kind": kind,
        "owner": owner,
        "source_seat": source_seat.map(u32::from),
        "called_tile": called_tile.map(u32::from),
        "tiles": ordered.iter().map(|t| u32::from(*t)).collect::<Vec<_>>(),
    })
}

/// Whether `event` may enter `actor`'s history (mirrors `visible_to_actor`).
pub fn visible_to_actor(event: &TypedEvent, actor: u8) -> bool {
    match event.visibility {
        Visibility::Public => true,
        Visibility::ActorPrivate => {
            event.visible_to.len() == 1 && event.visible_to[0] == actor
        }
        Visibility::ServerPrivate => false,
    }
}

/// Ryukyoku reason classes (mirrors `reason_kind`).
pub fn reason_kind(reason: &str) -> Result<&'static str, EventError> {
    match reason {
        "exhaustive_draw" | "nagashi_mangan" => Ok("draw_end"),
        "kyushu_kyuhai" | "kyuushu_kyuuhai" | "suucha_riichi" | "sanchaho" | "sanchahou"
        | "suukaikan" | "suukansansen" | "suufon_renda" | "sufuurenta" => Ok("abortive_draw"),
        other => Err(EventError(format!(
            "unmapped ryukyoku reason {other:?}; engine reports a hand termination the v1 event grammar cannot classify"
        ))),
    }
}

/// Canonical abortive reason literal (mirrors `ABORTIVE_REASONS`).
pub fn abortive_literal(reason: &str) -> Option<&'static str> {
    match reason {
        "kyushu_kyuhai" | "kyuushu_kyuuhai" => Some("kyuushu_kyuuhai"),
        "suucha_riichi" => Some("suucha_riichi"),
        "sanchaho" | "sanchahou" => Some("sanchahou"),
        "suukaikan" | "suukansansen" => Some("suukaikan"),
        "suufon_renda" | "sufuurenta" => Some("suufon_renda"),
        _ => None,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn public_event(kind: &str, seq: u64) -> TypedEvent {
        TypedEvent {
            game_id: "g".to_string(),
            sequence: seq,
            kind: kind.to_string(),
            actor: None,
            visibility: Visibility::Public,
            visible_to: vec![0, 1, 2, 3],
            tile: None,
            action_id: None,
            source_seat: None,
            consumed_tiles: vec![],
            offered_action_ids: vec![],
            accepted_action_ids: vec![],
            round_index: Some(0),
            scores: Some([25000, 25000, 25000, 25000]),
            reason: None,
            public_delta: vec![],
            rules_hash: "sha256:0".to_string(),
            schema_hash: "sha256:0".to_string(),
        }
    }

    #[test]
    fn stream_grammar_and_visibility_matrix() {
        let mut arena = EventArena::new();
        assert!(arena.admit(public_event("round_start", 0)).is_err());
        arena.admit(public_event("game_start", 0)).unwrap();
        assert!(arena.admit(public_event("game_start", 0)).is_err());
        let mut private = public_event("draw_tile", 1);
        private.visibility = Visibility::ActorPrivate;
        private.visible_to = vec![2];
        private.actor = Some(2);
        private.tile = Some(17);
        private.round_index = None;
        private.scores = None;
        arena.admit(private).unwrap();
        assert!(visible_to_actor(arena.get(1).unwrap(), 2));
        assert!(!visible_to_actor(arena.get(1).unwrap(), 0));
        let resolved = TypedEvent {
            kind: "call_resolved".to_string(),
            visibility: Visibility::ServerPrivate,
            visible_to: vec![],
            offered_action_ids: vec![5, 9],
            accepted_action_ids: vec![9],
            sequence: 2,
            // Resolution carries only the offered/accepted id sets (same
            // shape the flush path emits); the helper's round/scores
            // defaults belong to boundary events, not resolutions.
            round_index: None,
            scores: None,
            ..public_event("call_resolved", 2)
        };
        arena.admit(resolved).unwrap();
        assert!(!visible_to_actor(arena.get(2).unwrap(), 0));
    }

    #[test]
    fn reason_classes_match_oracle() {
        assert_eq!(reason_kind("exhaustive_draw").unwrap(), "draw_end");
        assert_eq!(reason_kind("nagashi_mangan").unwrap(), "draw_end");
        assert_eq!(reason_kind("suukaikan").unwrap(), "abortive_draw");
        assert_eq!(abortive_literal("sanchaho"), Some("sanchahou"));
        assert!(reason_kind("bogus").is_err());
    }

    #[test]
    fn meld_id_composition() {
        assert_eq!(super::super::actions::meld_id("pon", &[20, 22, 21]), "pon:20.21.22");
    }
}
