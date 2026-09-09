//! Decision-row schema: the CLI/JSON surface of the wall-less replay driver.
//!
//! Reference (read-only): `src/hydra2/data/parquet.py` `ACTOR_FIELDS`
//! (actor-visible field discipline) and `src/hydra2/contracts/observation.py`
//! (`ActorObservation` closed slot set). The privileged fields have no slot
//! here by construction: wall/dead-wall, opponent concealed tiles,
//! unrevealed dora/ura, RNG, future events, server-private events, opponent
//! legal masks, and privileged labels are unrepresentable in `ReplayRow`.

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::decisions::ActionTable;

/// Canonical chosen action at one decision point (actor-visible subset).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChosenAction {
    Discard {
        tile: u8,
    },
    Tsumogiri {
        tile: u8,
    },
    RiichiDiscard {
        tile: u8,
    },
    Chi {
        called: u8,
        consumed: Vec<u8>,
    },
    Pon {
        called: u8,
        consumed: Vec<u8>,
        source: u8,
    },
    Daiminkan {
        called: u8,
        consumed: Vec<u8>,
        source: u8,
    },
    Ankan {
        consumed: Vec<u8>,
    },
    Kakan {
        tile: u8,
    },
    Ron {
        tile: u8,
        source: u8,
    },
    Tsumo {
        tile: u8,
    },
    Pass,
}

impl ChosenAction {
    pub fn kind_str(&self) -> &'static str {
        match self {
            ChosenAction::Discard { .. } => "discard",
            ChosenAction::Tsumogiri { .. } => "tsumogiri",
            ChosenAction::RiichiDiscard { .. } => "riichi_discard",
            ChosenAction::Chi { .. } => "chi",
            ChosenAction::Pon { .. } => "pon",
            ChosenAction::Daiminkan { .. } => "daiminkan",
            ChosenAction::Ankan { .. } => "ankan",
            ChosenAction::Kakan { .. } => "kakan",
            ChosenAction::Ron { .. } => "ron",
            ChosenAction::Tsumo { .. } => "tsumo",
            ChosenAction::Pass => "pass",
        }
    }

    /// Resolve the canonical template id over the published table.
    /// `seat` is the acting seat (offsets are responder-relative).
    pub fn lookup(&self, table: &ActionTable, seat: u8) -> Option<u32> {
        match self {
            ChosenAction::Discard { tile }
            | ChosenAction::Tsumogiri { tile }
            | ChosenAction::RiichiDiscard { tile } => table.lookup(
                self.kind_str(),
                Some(*tile),
                None,
                &[],
                None,
                matches!(self, ChosenAction::RiichiDiscard { .. }),
                false,
            ),
            ChosenAction::Chi { called, consumed } => table.lookup(
                "chi",
                None,
                Some(*called),
                consumed,
                Some(-1),
                false,
                false,
            ),
            ChosenAction::Pon { called, consumed, source }
            | ChosenAction::Daiminkan { called, consumed, source } => {
                let delta = (source + 4 - seat) % 4;
                let offset = if delta == 3 { -1 } else { delta as i8 };
                table.lookup(self.kind_str(), None, Some(*called), consumed, Some(offset), false, false)
            }
            ChosenAction::Ankan { consumed } => {
                table.lookup("ankan", None, None, consumed, None, false, false)
            }
            ChosenAction::Kakan { tile } => {
                table.lookup("kakan", Some(*tile), None, &[], None, false, true)
            }
            ChosenAction::Ron { tile, source } => {
                let delta = (source + 4 - seat) % 4;
                let offset = if delta == 3 { -1 } else { delta as i8 };
                table.lookup("ron", Some(*tile), None, &[], Some(offset), false, false)
            }
            ChosenAction::Tsumo { tile } => {
                table.lookup("tsumo", Some(*tile), None, &[], None, false, false)
            }
            ChosenAction::Pass => table.lookup("pass", None, None, &[], None, false, false),
        }
    }
}

/// One actor decision row: the JSON object the dump CLI emits per decision.
///
/// String-level observation projection (physical-copy folding is pinned by
/// design: every copy of one MJAI string reports the same canonical id, so
/// concealed/drawn/dora content compares at string level; red fives stay
/// distinct by string — `5mr`/`0m` report the aka copy 16/52/88, plain `5x`
/// reports the non-red pool — and kakan/ankan meld copies resolve to the
/// pool copy absent from the prior pon, exactly like the engine path's
/// deterministic copy resolution):
/// - `concealed_hand`: MJAI strings in ascending physical-id order, drawn
///   tile excluded (drawn reported separately);
/// - `drawn_tile`: acting seat's most recent drawn tile (persists across
///   turns until the next draw, mirroring the builder);
/// - `dora_indicators`: five slots of revealed indicator strings (`None`
///   tail for unrevealed; the initial marker is never revealed in v1);
/// - `legal_mask`: sorted true action-table ids (engine answers, never
///   guessed);
/// - `history_kinds`: per-seat visible envelope kinds in emission order
///   (public events plus the seat's own draws);
/// - `wall_digest`: `None` on the wall-less path (SIM mark), `Some` real
///   wall digest on the walled path (never a placeholder; the SIM mark is
///   retired for walled rows).
#[derive(Debug, Clone, Serialize)]
pub struct ReplayRow {
    pub game_id: String,
    pub round_id: String,
    pub decision_id: String,
    pub seat: u8,
    pub phase: String,
    pub turn_actor: u8,
    pub chosen: ChosenAction,
    pub chosen_action_id: Option<u32>,
    pub chosen_unresolved: Option<String>,
    pub legal_mask: Vec<u32>,
    pub history_kinds: Vec<String>,
    pub concealed_hand: Vec<String>,
    pub drawn_tile: Option<String>,
    pub dora_indicators: Vec<Option<String>>,
    pub wall_remaining: u32,
    /// Real wall digest for walled rows; `None` for wall-less (SIM mark).
    /// Skipped in serde so row JSON stays stable; it binds `derivation_hash`
    /// and the `walled-v1` projection tag, never a separate envelope field.
    #[serde(skip)]
    pub wall_digest: Option<String>,
}

/// Whole-game quarantine record (mirrors the oracle's quarantine table).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Quarantine {
    pub game_id: String,
    pub reason_code: String,
    pub detail: String,
}

// ---------------------------------------------------------------------------
// DecisionRow-JSON emission (Slice 3 + Slice 7 walled binding).
// ---------------------------------------------------------------------------
//
// Reference (read-only): `src/hydra2/data/parquet.py` `ACTOR_FIELDS` (13
// actor-namespace fields), `src/hydra2/engines/riichienv/log_replay.py`
// `_capture_row` (wall-less derivation = `of_canonical` over game/decision/
// observation-hash/chosen-id/`wall_digest: None`/adapter-hash/`SIM mark`)
// and `src/hydra2/data/replay_expand.py` `_emit_row` (walled derivation =
// `of_canonical` over game/decision/observation-hash/chosen-id/real
// `wall_digest`/adapter-hash, no `derivation` key). The parquet layout
// owner stays Python: Rust never writes parquet, it only emits the same
// 13-field envelope as JSON so the frozen-hash fixture
// (`scripts/freeze_row_hashes.py`) can key whole-row identity per
// `decision_id`.
//
/// Wall-less derivation marker bound into `derivation_hash` instead of a
/// wall digest (placeholder digests are never bound; the SIM mark rides ONLY
/// wall-less rows — walled rows bind the real digest with no mark).
pub const SIM_DERIVATION_MARK: &str = "sim-replay-wall-less-v1";
/// Walled observation projection tag (retires the SIM mark for walled rows).
pub const WALLED_PROJECTION: &str = "walled-v1";
/// Wall-less observation projection tag.
pub const WALL_LESS_PROJECTION: &str = "wall-less-v1";
/// Bridge full-doc envelope version (W3-A).
///
/// The Rust handoff stays v1 projections (bridge INPUT, tagged
/// `walled-v1` / `wall-less-v1` above); the Python bridge
/// (`src/hydra2/training/rust_observations.py::assemble_game_rows`)
/// expands them into versioned full-doc rows carrying this marker
/// out-of-band, so `src/hydra2/training/dataset.py` ingress accepts
/// versioned full docs and rejects unexpanded projections fail-closed
/// (`unexpanded-projection-row`, never synthesized). Bumping this marker
/// requires a matching ingress update plus an oracle re-freeze.
pub const BRIDGE_FULL_DOC_ENVELOPE_VERSION: &str = "full-doc-v1";

/// The 13 actor-namespace fields, in `ACTOR_FIELDS` order. `privileged_label_ref`
/// is intentionally absent: it is a privileged-join pointer, not an actor field.
pub const ACTOR_FIELDS: [&str; 13] = [
    "game_id",
    "round_id",
    "decision_id",
    "seat",
    "source_object_id",
    "split",
    "rules_hash",
    "adapter_hash",
    "observation_hash",
    "action_table_hash",
    "derivation_hash",
    "actor_observation",
    "chosen_action_id",
];

/// Provenance passthroughs the wall-less driver cannot mint itself.
///
/// Supplied by the oracle materializer at emission time: the Tenhou
/// object id, the dataset split, and the content hashes of the rules,
/// the adapter, and the published action table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowProvenance {
    pub source_object_id: String,
    pub split: String,
    pub rules_hash: String,
    pub adapter_hash: String,
    pub action_table_hash: String,
}

/// RFC 8785-shaped canonical bytes: object keys sorted, no whitespace.
///
/// Byte-identical to `hydra2.artifacts.canonical.canonical_bytes` for the
/// values row docs carry (null/bool/int/string/array/object; keys compared
/// by UTF-16BE code units per RFC 8785 §3.2.3, which is byte order for the
/// ASCII keys used here; string escapes match `_SHORT_ESCAPES` plus
/// `\u00XX` for other controls, `\\`/`\"`, raw UTF-8 otherwise).
/// Floats never occur in row docs; they render as JSON numbers.
pub fn canonical_json_bytes(value: &serde_json::Value) -> Vec<u8> {
    let mut out = String::new();
    write_canonical(&mut out, value);
    out.into_bytes()
}

fn write_canonical(out: &mut String, value: &serde_json::Value) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(true) => out.push_str("true"),
        serde_json::Value::Bool(false) => out.push_str("false"),
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::String(s) => push_canonical_string(out, s),
        serde_json::Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(out, item);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by(|a, b| {
                let (au, bu): (Vec<u16>, Vec<u16>) =
                    (a.encode_utf16().collect(), b.encode_utf16().collect());
                au.cmp(&bu)
            });
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_canonical_string(out, key);
                out.push(':');
                write_canonical(out, &map[*key]);
            }
            out.push('}');
        }
    }
}

fn push_canonical_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '\u{08}' => out.push_str("\\b"),
            '\u{09}' => out.push_str("\\t"),
            '\u{0A}' => out.push_str("\\n"),
            '\u{0C}' => out.push_str("\\f"),
            '\u{0D}' => out.push_str("\\r"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub(crate) fn digest_text(bytes: &[u8]) -> String {
    let sum = Sha256::digest(bytes);
    let mut text = String::with_capacity(7 + 64);
    text.push_str("sha256:");
    for b in sum {
        text.push_str(&format!("{b:02x}"));
    }
    text
}

/// `observation_hash` over the actor-observation projection document.
pub fn observation_hash_for(actor_observation: &serde_json::Value) -> String {
    digest_text(&canonical_json_bytes(actor_observation))
}

/// `derivation_hash` mirroring the oracle's `of_canonical` derivation doc:
/// game, decision, observation hash, chosen id (`null` when unresolved),
/// `wall_digest: None` always (never invented), adapter hash, SIM mark.
/// Wall-less ONLY: walled rows use [`derivation_hash_for_walled`] (real
/// digest, no mark). Placeholder digests are never bound on either path.
pub fn derivation_hash_for(
    game_id: &str,
    decision_id: &str,
    observation_hash: &str,
    chosen_action_id: Option<u32>,
    adapter_hash: &str,
) -> String {
    let doc = serde_json::json!({
        "game_id": game_id,
        "decision_id": decision_id,
        "observation_hash": observation_hash,
        "chosen_action_id": chosen_action_id,
        "wall_digest": serde_json::Value::Null,
        "adapter_hash": adapter_hash,
        "derivation": SIM_DERIVATION_MARK,
    });
    digest_text(&canonical_json_bytes(&doc))
}

/// Walled `derivation_hash` mirroring `replay_expand._emit_row`: game,
/// decision, observation hash, chosen id, REAL `wall_digest`, adapter hash —
/// with NO `derivation` mark (the SIM mark is retired for walled rows).
/// `wall_digest` must be a non-empty `sha256:` digest (the schedule digest);
/// anything else is a caller bug and panics in debug (the walker always
/// mints it via [`wall_schedule_digest`]).
pub fn derivation_hash_for_walled(
    game_id: &str,
    decision_id: &str,
    observation_hash: &str,
    chosen_action_id: Option<u32>,
    wall_digest: &str,
    adapter_hash: &str,
) -> String {
    debug_assert!(
        wall_digest.starts_with("sha256:") && wall_digest.len() == 7 + 64,
        "walled derivation needs a real wall digest, got {wall_digest:?}"
    );
    let doc = serde_json::json!({
        "game_id": game_id,
        "decision_id": decision_id,
        "observation_hash": observation_hash,
        "chosen_action_id": chosen_action_id,
        "wall_digest": wall_digest,
        "adapter_hash": adapter_hash,
    });
    digest_text(&canonical_json_bytes(&doc))
}

/// Real wall digest binding the derivation and `wall_id` (mirrors
/// `protocol.wall_schedule_digest`: sha256 over canonical bytes of
/// `{schedule_id, physical_tiles}` with `schedule_id =
/// replay-{game_id}`). Byte-identical to Python on the same inputs;
/// placeholder digests are never minted here.
pub fn wall_schedule_digest(schedule_id: &str, physical_tiles: &[u8]) -> String {
    debug_assert_eq!(physical_tiles.len(), 136, "wall must carry 136 tiles");
    let tiles: Vec<serde_json::Value> = physical_tiles
        .iter()
        .map(|t| serde_json::Value::Number(serde_json::Number::from(*t as u64)))
        .collect();
    let doc = serde_json::json!({
        "schedule_id": schedule_id,
        "physical_tiles": tiles,
    });
    digest_text(&canonical_json_bytes(&doc))
}

/// Schedule id for one game (mirrors `_schedule_for`: `replay-{game_id}`).
pub fn schedule_id_for(game_id: &str) -> String {
    format!("replay-{game_id}")
}

impl ReplayRow {
    /// Whether this row is walled (real digest) or wall-less (SIM mark).
    pub fn is_walled(&self) -> bool {
        self.wall_digest.is_some()
    }

    /// Actor-observation projection: the row's string-level observation
    /// content as one JSON document.
    ///
    /// Honest subset, never the engine truth: concealed/drawn/dora strings
    /// (pool-first copy folding, smallest-copy everywhere; red fives stay
    /// string-distinct), the countdown `live_wall_tiles_remaining`
    /// (actor-visible), per-seat visible envelope kinds, the sorted
    /// legal-mask ids, phase/turn/seat routing, and the chosen action.
    /// Engine-owned surfaces (scores, winds, furiten, riichi state, meld
    /// records, envelope payloads, the 6792-wide bool mask) have no slot
    /// here by construction. The `projection` tag keeps this doc
    /// distinguishable from an engine `ActorObservation` AND marks the
    /// wall binding: `walled-v1` (real digest in the derivation) vs
    /// `wall-less-v1` (SIM mark in the derivation).
    pub fn actor_observation_doc(&self) -> serde_json::Value {
        let mut legal_mask = self.legal_mask.clone();
        legal_mask.sort_unstable();
        legal_mask.dedup();
        let projection = if self.is_walled() {
            WALLED_PROJECTION
        } else {
            WALL_LESS_PROJECTION
        };
        serde_json::json!({
            "projection": projection,
            "game_id": self.game_id,
            "decision_id": self.decision_id,
            "seat": self.seat,
            "phase": self.phase,
            "turn_actor": self.turn_actor,
            "concealed_hand": self.concealed_hand,
            "drawn_tile": self.drawn_tile,
            "dora_indicators": self.dora_indicators,
            "live_wall_tiles_remaining": self.wall_remaining,
            "history_kinds": self.history_kinds,
            "legal_mask": legal_mask,
            "chosen": self.chosen,
        })
    }

    /// Emit EXACTLY the 13 `ACTOR_FIELDS` as a JSON object.
    ///
    /// `actor_observation` rides as a JSON string (the parquet column
    /// type); `derivation_hash` binds the wall: wall-less rows bind
    /// `SIM_DERIVATION_MARK` with `wall_digest: None` (never invented),
    /// walled rows bind the REAL wall digest with no mark (the SIM mark is
    /// retired for walled rows); `chosen_action_id` is `null` exactly
    /// when `chosen_unresolved` is set (the parquet schema owner stays
    /// Python — Rust never writes parquet). Provenance hashes pass
    /// through from the oracle materializer.
    pub fn to_decision_json(&self, provenance: &RowProvenance) -> serde_json::Value {
        // S4 pin: `chosen_action_id` is null exactly when `chosen_unresolved`
        // is set. The walker builds both together (`decisions.rs`
        // `capture_row`: `Some` id with `None` reason, or `None` id with
        // `Some("action-id-unresolved:{kind}")`); the envelope carries only
        // the id, so a mismatch here would silently fork the two sides.
        // Fail closed in debug; the `s4_gates` integration test asserts the
        // iff over every fixture row in release too.
        debug_assert_eq!(
            self.chosen_action_id.is_none(),
            self.chosen_unresolved.is_some(),
            "chosen null iff unresolved: id={:?} reason={:?}",
            self.chosen_action_id,
            self.chosen_unresolved,
        );
        // S7 pin: the SIM mark rides ONLY wall-less rows. Walled rows must
        // carry a real digest (fail closed in debug when missing).
        debug_assert!(
            self.wall_digest.as_ref().map_or(true, |d| d.starts_with("sha256:") && d.len() == 7 + 64),
            "walled rows need a real wall digest, got {:?}",
            self.wall_digest,
        );
        let actor_doc = self.actor_observation_doc();
        let actor_text =
            serde_json::to_string(&actor_doc).expect("actor projection must encode");
        let observation_hash = observation_hash_for(&actor_doc);
        let derivation_hash = match &self.wall_digest {
            None => derivation_hash_for(
                &self.game_id,
                &self.decision_id,
                &observation_hash,
                self.chosen_action_id,
                &provenance.adapter_hash,
            ),
            Some(digest) => derivation_hash_for_walled(
                &self.game_id,
                &self.decision_id,
                &observation_hash,
                self.chosen_action_id,
                digest,
                &provenance.adapter_hash,
            ),
        };
        let mut decision = serde_json::Map::with_capacity(ACTOR_FIELDS.len());
        decision.insert("game_id".to_string(), serde_json::Value::String(self.game_id.clone()));
        decision.insert("round_id".to_string(), serde_json::Value::String(self.round_id.clone()));
        decision.insert(
            "decision_id".to_string(),
            serde_json::Value::String(self.decision_id.clone()),
        );
        decision.insert(
            "seat".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.seat as u64)),
        );
        decision.insert(
            "source_object_id".to_string(),
            serde_json::Value::String(provenance.source_object_id.clone()),
        );
        decision.insert("split".to_string(), serde_json::Value::String(provenance.split.clone()));
        decision.insert(
            "rules_hash".to_string(),
            serde_json::Value::String(provenance.rules_hash.clone()),
        );
        decision.insert(
            "adapter_hash".to_string(),
            serde_json::Value::String(provenance.adapter_hash.clone()),
        );
        decision.insert(
            "observation_hash".to_string(),
            serde_json::Value::String(observation_hash),
        );
        decision.insert(
            "action_table_hash".to_string(),
            serde_json::Value::String(provenance.action_table_hash.clone()),
        );
        decision.insert(
            "derivation_hash".to_string(),
            serde_json::Value::String(derivation_hash),
        );
        decision.insert(
            "actor_observation".to_string(),
            serde_json::Value::String(actor_text),
        );
        decision.insert(
            "chosen_action_id".to_string(),
            match self.chosen_action_id {
                Some(id) => serde_json::Value::Number(serde_json::Number::from(id as u64)),
                None => serde_json::Value::Null,
            },
        );
        serde_json::Value::Object(decision)
    }
}

#[cfg(test)]
mod decision_json_tests {
    use super::*;
    use crate::decision_ids;

    fn sample_row() -> ReplayRow {
        ReplayRow {
            game_id: "game-23efeea403a7".to_string(),
            round_id: decision_ids::round_id("game-23efeea403a7", 0),
            decision_id: decision_ids::decision_id("game-23efeea403a7", 0),
            seat: 0,
            phase: "draw_decision".to_string(),
            turn_actor: 0,
            chosen: ChosenAction::Discard { tile: 116 },
            chosen_action_id: Some(120),
            chosen_unresolved: None,
            legal_mask: vec![52, 8, 36, 28],
            history_kinds: vec![
                "game_start".to_string(),
                "round_start".to_string(),
                "turn_advance".to_string(),
                "draw_tile".to_string(),
            ],
            concealed_hand: vec!["2m".to_string(), "7m".to_string()],
            drawn_tile: Some("F".to_string()),
            dora_indicators: vec![None, None, None, None, None],
            wall_remaining: 69,
            wall_digest: None,
        }
    }

    fn provenance() -> RowProvenance {
        RowProvenance {
            source_object_id: "2012010100gm-00a9-0000-8b952992".to_string(),
            split: "train".to_string(),
            rules_hash: "sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b"
                .to_string(),
            adapter_hash: "sha256:61f4d0328b5fbf46c0bdd9331e573b11e0166b357e93e218e96a5152c7c71c33"
                .to_string(),
            action_table_hash: "sha256:7b55693428384713f6a6ab7292f57657259c2ca9f05139944d4c8c6197ae76e8"
                .to_string(),
        }
    }

    #[test]
    fn envelope_carries_exactly_actor_fields() {
        let decision = sample_row().to_decision_json(&provenance());
        let map = decision.as_object().expect("decision object");
        let mut keys: Vec<&String> = map.keys().collect();
        keys.sort();
        let mut expected: Vec<String> = ACTOR_FIELDS.iter().map(|s| s.to_string()).collect();
        expected.sort();
        assert_eq!(keys, expected.iter().collect::<Vec<_>>());
        assert!(!map.contains_key("privileged_label_ref"));
        assert!(!map.contains_key("wall_id"));
    }

    #[test]
    fn actor_observation_rides_as_string_without_privilege() {
        let decision = sample_row().to_decision_json(&provenance());
        let text = decision
            .get("actor_observation")
            .and_then(|v| v.as_str())
            .expect("actor_observation string");
        for forbidden in [
            "hidden_tiles",
            "dead_wall",
            "opponent_hand",
            "privileged",
            "full_world",
            "wall_id",
        ] {
            assert!(!text.contains(forbidden), "leak: {forbidden}");
        }
        let doc: serde_json::Value = serde_json::from_str(text).expect("actor doc parses");
        assert_eq!(doc.get("seat").and_then(|v| v.as_u64()), Some(0));
        assert_eq!(doc.get("phase").and_then(|v| v.as_str()), Some("draw_decision"));
        // Legal ids sort at the emission boundary even when the row is not.
        assert_eq!(
            doc.get("legal_mask").expect("mask"),
            &serde_json::json!([8, 28, 36, 52])
        );
    }

    #[test]
    fn hashes_bind_mark_and_wall_none() {
        let row = sample_row();
        let decision = row.to_decision_json(&provenance());
        let obs_hash = decision
            .get("observation_hash")
            .and_then(|v| v.as_str())
            .expect("observation hash");
        assert!(obs_hash.starts_with("sha256:"));
        let recomputed = derivation_hash_for(
            &row.game_id,
            &row.decision_id,
            obs_hash,
            Some(120),
            &provenance().adapter_hash,
        );
        assert_eq!(
            decision.get("derivation_hash").and_then(|v| v.as_str()),
            Some(recomputed.as_str())
        );
        // The derivation doc carries the SIM mark with no wall digest; the
        // canonical form sorts keys with no whitespace.
        let doc = serde_json::json!({
            "game_id": "g",
            "decision_id": "g:d0000",
            "observation_hash": "sha256:00",
            "chosen_action_id": 7,
            "wall_digest": serde_json::Value::Null,
            "adapter_hash": "sha256:11",
            "derivation": SIM_DERIVATION_MARK,
        });
        let bytes = canonical_json_bytes(&doc);
        let text = String::from_utf8(bytes).expect("ascii");
        assert!(text.starts_with("{\"adapter_hash\":"));
        assert!(text.contains("\"derivation\":\"sim-replay-wall-less-v1\""));
        assert!(text.contains("\"wall_digest\":null"));
    }

    #[test]
    fn unresolved_chosen_emits_null() {
        let mut row = sample_row();
        row.chosen_action_id = None;
        row.chosen_unresolved = Some("action-id-unresolved:pass".to_string());
        let decision = row.to_decision_json(&provenance());
        assert!(decision.get("chosen_action_id").map(|v| v.is_null()).unwrap_or(false));
    }

    #[test]
    fn resolved_chosen_emits_int_iff_unresolved_absent() {
        // Resolved side of the iff: an id rides as a JSON integer and the
        // unresolved reason is absent.
        let row = sample_row();
        assert!(row.chosen_action_id.is_some());
        assert!(row.chosen_unresolved.is_none());
        let decision = row.to_decision_json(&provenance());
        let chosen = decision.get("chosen_action_id").expect("chosen field");
        assert_eq!(chosen.as_u64(), Some(120));
        // Unresolved side: null id with the reason set (constructed together
        // by the walker; the envelope carries only the id).
        let mut unresolved = sample_row();
        unresolved.chosen_action_id = None;
        unresolved.chosen_unresolved = Some("action-id-unresolved:pass".to_string());
        assert_eq!(
            unresolved.chosen_action_id.is_none(),
            unresolved.chosen_unresolved.is_some()
        );
        let decision = unresolved.to_decision_json(&provenance());
        assert!(decision.get("chosen_action_id").map(|v| v.is_null()).unwrap_or(false));
    }

    #[test]
    fn canonical_matches_rfc8785_shapes() {
        let nested = serde_json::json!({"b": [true, null, 7], "a": {"z": "é\"\n"}});
        let text = String::from_utf8(canonical_json_bytes(&nested)).expect("utf8");
        assert_eq!(text, "{\"a\":{\"z\":\"é\\\"\\n\"},\"b\":[true,null,7]}");
    }

    /// S4 (R8a): `canonical_json_bytes` is byte-identical to Python
    /// `hydra2.artifacts.canonical.canonical_bytes` over the value corpus
    /// row docs carry (null/bool/int/string/array/object). Expected strings
    /// below were generated by the Python authority
    /// (`canonicalize` over these values) and are pinned verbatim: any fork
    /// in key order, escapes, or UTF-8 handling fails here, not downstream.
    /// Floats never occur in row docs and are excluded from the corpus.
    #[test]
    fn canonical_matches_python_authority_corpus() {
        let cases: Vec<(serde_json::Value, &str)> = vec![
            (serde_json::json!(null), "null"),
            (serde_json::json!(true), "true"),
            (serde_json::json!(false), "false"),
            (serde_json::json!(0), "0"),
            (serde_json::json!(-1), "-1"),
            (serde_json::json!(9007199254740991i64), "9007199254740991"),
            (serde_json::json!(""), "\"\""),
            (serde_json::json!("a"), "\"a\""),
            (serde_json::json!("é"), "\"é\""),
            (serde_json::json!("\"\\"), "\"\\\"\\\\\""),
            (serde_json::json!("\u{08}\u{09}\u{0A}\u{0C}\u{0D}"), "\"\\b\\t\\n\\f\\r\""),
            // Controls < 0x20 ride as `\u00XX`; DEL (0x7f) rides raw UTF-8.
            (serde_json::json!("\u{0}\u{1F}\u{7F}"), "\"\\u0000\\u001f\u{7F}\""),
            (serde_json::json!([]), "[]"),
            (serde_json::json!([true, null, 7]), "[true,null,7]"),
            (serde_json::json!({}), "{}"),
            (
                serde_json::json!({"b": [true, null, 7], "a": {"z": "é\"\n"}}),
                "{\"a\":{\"z\":\"é\\\"\\n\"},\"b\":[true,null,7]}",
            ),
            // Derivation doc: keys sort ASCII-bytewise with no whitespace.
            (
                serde_json::json!({
                    "game_id": "g",
                    "decision_id": "g:d0000",
                    "observation_hash": "sha256:00",
                    "chosen_action_id": 7,
                    "wall_digest": serde_json::Value::Null,
                    "adapter_hash": "sha256:11",
                    "derivation": SIM_DERIVATION_MARK,
                }),
                "{\"adapter_hash\":\"sha256:11\",\"chosen_action_id\":7,\"decision_id\":\"g:d0000\",\"derivation\":\"sim-replay-wall-less-v1\",\"game_id\":\"g\",\"observation_hash\":\"sha256:00\",\"wall_digest\":null}",
            ),
            // Non-ASCII keys sort by UTF-16 code units (byte order for ASCII;
            // é U+00E9 sorts after z U+007A per RFC 8785 §3.2.3).
            (
                serde_json::json!({"z": 1, "a": 2, "é": 3}),
                "{\"a\":2,\"z\":1,\"é\":3}",
            ),
            (
                serde_json::json!({"nested": {"list": [1, {"k": "v"}]}, "empty": {}}),
                "{\"empty\":{},\"nested\":{\"list\":[1,{\"k\":\"v\"}]}}",
            ),
        ];
        for (value, expected) in &cases {
            let bytes = canonical_json_bytes(value);
            assert_eq!(
                bytes,
                expected.as_bytes(),
                "canonical drift for {}",
                serde_json::to_string(value).unwrap_or_default()
            );
        }
    }
}
