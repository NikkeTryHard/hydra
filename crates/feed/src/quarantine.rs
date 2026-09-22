//! Quarantine — invalid-game records + lineage (`feed::quarantine`).
//!
//! Rust owner of `src/hydra2/data/quarantine.py::quarantine_invalid`
//! (lines 44-87) + `write_quarantine_manifest` (lines 90-109): every invalid
//! outcome becomes a record carrying its lineage; a missing outcome is a
//! hard error, never a silent skip (`:59-63`).
//!
//! Taxonomy is CLOSED (trap-24): `REASON_* 0-8 ∪ WALK_* 10+`, unknown →
//! `"other"` on both sides. No new strings are minted here —
//! [`reason_class`] reuses `gate::reason_name` (framing family) +
//! `ledger::walk_reason_name` (walk family); [`normalize_error_class`]
//! admits the validate/decode closed sets and folds everything else to
//! `"other"`.
//!
//! Morsel discipline (trap-15): [`quarantine_invalid`] builds its id lookup
//! locally per call; the caller merges batches in sequence order. No shared
//! dict under `par_iter`.
//!
//! Allocation: lineage (and its checks clone) allocates ONLY on the invalid
//! path — valid rows are skipped with zero allocation beyond the local
//! lookup table.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::gate::reason_name;
use crate::ledger::walk_reason_name;
use crate::validate::Outcome;

/// Lineage keys, EXACT (`quarantine.py:74-83`).
pub const LINEAGE_KEYS: [&str; 8] = [
    "object_id",
    "packaged_object_id",
    "confidential_source_id",
    "authorization_attestation_id",
    "permitted_purpose",
    "parent_ids",
    "game_events",
    "validation_checks",
];

/// Quarantine manifest schema version (`quarantine.py:92`).
pub const MANIFEST_SCHEMA_VERSION: &str = "1.0.0";

/// Closed vocabulary join: numeric reason → cold string.
///
/// `0-8` reuse the `REASON_*` framing family, `10+` the `WALK_*` walk
/// family; the gap (`9`) and anything unmapped render `"other"` (never a
/// silent pass — trap-24).
pub const fn reason_class(code: u8) -> &'static str {
    if code <= 8 {
        reason_name(code)
    } else if code >= 10 {
        walk_reason_name(code)
    } else {
        "other"
    }
}

/// Admit a closed error-class string, fold anything else to `"other"`.
///
/// Closed set = validate taxonomy (`structure|event_order|`
/// `tile_conservation|red_identity|legality|dora_shape`) + decode taxonomy
/// (`utf8|trailing_newline|blank_line|json|shape|start_end`) + the
/// `gate`/`walk` cold renders + `"other"` itself.
pub fn normalize_error_class(class: &str) -> &'static str {
    match class {
        // validate.rs closed set.
        "structure" | "event_order" | "tile_conservation" | "red_identity" | "legality"
        | "dora_shape" => class_as_static(class),
        // decode.rs closed set.
        "utf8" | "trailing_newline" | "blank_line" | "json" | "shape" | "start_end" => {
            class_as_static(class)
        }
        // gate + walk cold renders (shared members folded into one arm —
        // "unknown-event" / "bare-dora" / "double-ron" / "turn-order" render
        // on both sides with identical strings).
        "ok" | "framing" | "unknown-event" | "wall-bearing" | "bare-dora" | "double-ron"
        | "turn-order" | "tile-conservation" | "claim-no-offer" | "draw-past-wall"
        | "kyushu-ambiguous" | "unmapped-ryukyoku-reason" | "engine-desync"
        | "action-id-unresolved" | "past-terminal" => class_as_static(class),
        "other" => "other",
        _ => "other",
    }
}

/// Map a closed-vocabulary member back to its static render (`match` arms
/// above already proved membership; the fallback is unreachable-by-proof
/// but total — never panics).
fn class_as_static(class: &str) -> &'static str {
    match class {
        "structure" => "structure",
        "event_order" => "event_order",
        "tile_conservation" => "tile_conservation",
        "red_identity" => "red_identity",
        "legality" => "legality",
        "dora_shape" => "dora_shape",
        "utf8" => "utf8",
        "trailing_newline" => "trailing_newline",
        "blank_line" => "blank_line",
        "json" => "json",
        "shape" => "shape",
        "start_end" => "start_end",
        "ok" => "ok",
        "framing" => "framing",
        "unknown-event" => "unknown-event",
        "wall-bearing" => "wall-bearing",
        "bare-dora" => "bare-dora",
        "double-ron" => "double-ron",
        "turn-order" => "turn-order",
        "tile-conservation" => "tile-conservation",
        "claim-no-offer" => "claim-no-offer",
        "draw-past-wall" => "draw-past-wall",
        "kyushu-ambiguous" => "kyushu-ambiguous",
        "unmapped-ryukyoku-reason" => "unmapped-ryukyoku-reason",
        "engine-desync" => "engine-desync",
        "action-id-unresolved" => "action-id-unresolved",
        "past-terminal" => "past-terminal",
        _ => "other",
    }
}

/// Raw-object identity row (the `RawObjectRow` join fields quarantine
/// lineage needs: `rows.py:269-280`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawRow {
    /// Object id (join key, `sha256:<hex>`).
    pub object_id: String,
    /// Packaged object id (`sha256:<hex>`).
    pub packaged_object_id: String,
    /// Confidential source id.
    pub confidential_source_id: String,
    /// Authorization attestation id.
    pub authorization_attestation_id: String,
    /// Permitted purposes.
    pub permitted_purpose: Vec<String>,
    /// Parent ids.
    pub parent_ids: Vec<String>,
}

/// One outcome inside the current morsel: lookup key + verdict + decoded
/// event count (`None` when the game never decoded — `game_events: 0`,
/// `quarantine.py:81`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MorselOutcome<'a> {
    /// Object id (lookup key).
    pub object_id: &'a str,
    /// Validation verdict.
    pub outcome: &'a Outcome,
    /// Decoded event count, when decoded.
    pub game_events: Option<usize>,
}

/// Provenance carried on every quarantined record (`quarantine.py:74-83`).
///
/// Allocated ONLY on the invalid path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lineage {
    /// Confidential source id.
    pub confidential_source_id: String,
    /// Authorization attestation id.
    pub authorization_attestation_id: String,
    /// Permitted purposes.
    pub permitted_purpose: Vec<String>,
    /// Parent ids.
    pub parent_ids: Vec<String>,
    /// Decoded event count (`0` when undecodable).
    pub game_events: usize,
    /// Frozen validation checks clone.
    pub validation_checks: Vec<(String, String)>,
}

/// Quarantined record (mirrors `QuarantinedRecord`, `:27-35`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarantinedRecord {
    /// Object id.
    pub object_id: String,
    /// Packaged object id.
    pub packaged_object_id: String,
    /// Game id (`None` when the game never decoded).
    pub game_id: Option<String>,
    /// Closed error class (normalized at construction).
    pub error_class: String,
    /// Offending event index, when the failure names an event.
    pub error_event_index: Option<u32>,
    /// Provenance.
    pub lineage: Lineage,
    /// Validation seal, when validated.
    pub validation_hash: Option<String>,
}

impl QuarantinedRecord {
    /// Decode-failure record (framer-side): no game id, no seal, zero
    /// events; the class is normalized closed (`"other"` on drift).
    pub fn decode_failure(
        raw: &RawRow,
        error_class: &str,
        event_index: Option<u32>,
    ) -> Self {
        QuarantinedRecord {
            object_id: raw.object_id.clone(),
            packaged_object_id: raw.packaged_object_id.clone(),
            game_id: None,
            error_class: normalize_error_class(error_class).to_owned(),
            error_event_index: event_index,
            lineage: Lineage {
                confidential_source_id: raw.confidential_source_id.clone(),
                authorization_attestation_id: raw.authorization_attestation_id.clone(),
                permitted_purpose: raw.permitted_purpose.clone(),
                parent_ids: raw.parent_ids.clone(),
                game_events: 0,
                validation_checks: Vec::new(),
            },
            validation_hash: None,
        }
    }
}

/// Quarantine build failure: missing outcome is a hard error, never a
/// silent skip (`quarantine.py:59-63`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuarantineErr {
    /// No outcome for this object id (would be a silent skip).
    MissingOutcome(String),
}

impl core::fmt::Display for QuarantineErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QuarantineErr::MissingOutcome(id) => {
                write!(f, "missing validation outcome for {id} (would be silent skip)")
            }
        }
    }
}

impl std::error::Error for QuarantineErr {}

/// Build the quarantine list for invalid games, preserving lineage
/// (`quarantine.py:44-87`).
///
/// Iterates `rows` in order (output order = input order; the manifest sorts
/// separately). Valid outcomes are skipped allocation-free; every invalid
/// outcome becomes a record. A row with no outcome is a hard
/// [`QuarantineErr::MissingOutcome`].
pub fn quarantine_invalid(
    rows: &[RawRow],
    outcomes: &[MorselOutcome<'_>],
) -> Result<Vec<QuarantinedRecord>, QuarantineErr> {
    // Morsel-local lookup (trap-15): built per call, merged by the caller
    // in sequence order — never a shared dict under `par_iter`.
    let mut by_id: HashMap<&str, (&Outcome, Option<usize>)> =
        HashMap::with_capacity(outcomes.len());
    for m in outcomes {
        by_id.insert(m.object_id, (m.outcome, m.game_events));
    }
    let mut quarantined = Vec::new();
    for raw in rows {
        let (outcome, game_events) = by_id
            .get(raw.object_id.as_str())
            .copied()
            .ok_or_else(|| QuarantineErr::MissingOutcome(raw.object_id.clone()))?;
        if outcome.valid {
            continue;
        }
        let (class, event_index) = match &outcome.error {
            Some(e) => (e.class.as_str(), e.event_index),
            None => ("other", None),
        };
        quarantined.push(QuarantinedRecord {
            object_id: raw.object_id.clone(),
            packaged_object_id: raw.packaged_object_id.clone(),
            game_id: Some(outcome.game_id.clone()),
            error_class: normalize_error_class(class).to_owned(),
            error_event_index: event_index,
            lineage: Lineage {
                confidential_source_id: raw.confidential_source_id.clone(),
                authorization_attestation_id: raw.authorization_attestation_id.clone(),
                permitted_purpose: raw.permitted_purpose.clone(),
                parent_ids: raw.parent_ids.clone(),
                game_events: game_events.unwrap_or(0),
                validation_checks: outcome
                    .checks
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), v.clone()))
                    .collect(),
            },
            validation_hash: outcome.validation_hash.clone(),
        });
    }
    Ok(quarantined)
}

/// Serialize one record (manifest object shape, `quarantine.py:93-103`).
fn record_value(r: &QuarantinedRecord) -> Value {
    let mut lineage = serde_json::Map::with_capacity(LINEAGE_KEYS.len());
    lineage.insert(
        "object_id".to_owned(),
        Value::String(r.object_id.clone()),
    );
    lineage.insert(
        "packaged_object_id".to_owned(),
        Value::String(r.packaged_object_id.clone()),
    );
    lineage.insert(
        "confidential_source_id".to_owned(),
        Value::String(r.lineage.confidential_source_id.clone()),
    );
    lineage.insert(
        "authorization_attestation_id".to_owned(),
        Value::String(r.lineage.authorization_attestation_id.clone()),
    );
    lineage.insert(
        "permitted_purpose".to_owned(),
        Value::Array(
            r.lineage
                .permitted_purpose
                .iter()
                .map(|s| Value::String(s.clone()))
                .collect(),
        ),
    );
    lineage.insert(
        "parent_ids".to_owned(),
        Value::Array(
            r.lineage
                .parent_ids
                .iter()
                .map(|s| Value::String(s.clone()))
                .collect(),
        ),
    );
    lineage.insert(
        "game_events".to_owned(),
        Value::from(r.lineage.game_events as u64),
    );
    let mut checks = serde_json::Map::with_capacity(r.lineage.validation_checks.len());
    for (k, v) in r.lineage.validation_checks.iter() {
        checks.insert(k.clone(), Value::String(v.clone()));
    }
    lineage.insert("validation_checks".to_owned(), Value::Object(checks));

    let mut o = serde_json::Map::with_capacity(7);
    o.insert("object_id".to_owned(), Value::String(r.object_id.clone()));
    o.insert(
        "packaged_object_id".to_owned(),
        Value::String(r.packaged_object_id.clone()),
    );
    o.insert(
        "game_id".to_owned(),
        r.game_id.clone().map(Value::String).unwrap_or(Value::Null),
    );
    o.insert(
        "error_class".to_owned(),
        Value::String(r.error_class.clone()),
    );
    o.insert(
        "error_event_index".to_owned(),
        r.error_event_index.map(|i| Value::from(i as u64)).unwrap_or(Value::Null),
    );
    o.insert("lineage".to_owned(), Value::Object(lineage));
    o.insert(
        "validation_hash".to_owned(),
        r.validation_hash
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    Value::Object(o)
}

/// Write the quarantine manifest: records sorted by `object_id`
/// (`quarantine.py:103`), digest `sha256(canonical(payload))` (`:106`),
/// atomic tmp+rename publish (`:108` via `atomic_replace_bytes`).
///
/// Returns the `sha256:<hex>` digest.
pub fn write_quarantine_manifest(
    path: &Path,
    records: &[QuarantinedRecord],
) -> std::io::Result<String> {
    let mut sorted: Vec<&QuarantinedRecord> = records.iter().collect();
    sorted.sort_by(|a, b| a.object_id.cmp(&b.object_id));
    let arr: Vec<Value> = sorted.iter().map(|r| record_value(r)).collect();
    let mut payload = serde_json::Map::with_capacity(2);
    payload.insert(
        "schema_version".to_owned(),
        Value::String(MANIFEST_SCHEMA_VERSION.to_owned()),
    );
    payload.insert("quarantined".to_owned(), Value::Array(arr));
    let payload_v = Value::Object(payload);
    let digest = crate::digest::of_canonical(&payload_v).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("seal failed: {e}"))
    })?;
    let mut with = payload_v.as_object().map_or_else(serde_json::Map::new, Clone::clone);
    with.insert("digest".to_owned(), Value::String(digest.clone()));
    let bytes = crate::canon::canonical_bytes(&Value::Object(with), "quarantine:manifest")
        .map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("seal failed: {e}"))
        })?;
    // Atomic publish: tmp sibling + rename (truncate-before-reuse + single
    // rename — trap-11 discipline).
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp_path = Path::new(&tmp).to_path_buf();
    std::fs::write(&tmp_path, &bytes)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::{ErrorClass, Outcome, ValidationError};

    fn raw(id: &str) -> RawRow {
        RawRow {
            object_id: id.to_owned(),
            packaged_object_id: format!("pkg-{id}"),
            confidential_source_id: "src-1".to_owned(),
            authorization_attestation_id: "att-1".to_owned(),
            permitted_purpose: vec!["train".to_owned()],
            parent_ids: vec!["sha256:00".to_owned()],
        }
    }

    fn invalid_outcome(game: &str, class: ErrorClass, idx: Option<u32>) -> Outcome {
        Outcome {
            game_id: game.to_owned(),
            object_id: "o".to_owned(),
            valid: false,
            error: Some(ValidationError {
                class,
                event_index: idx,
                message: "m".to_owned(),
            }),
            validation_hash: None,
            checks: vec![("structure", "ok".to_owned())]
                .into_iter()
                .map(|(k, v): (&str, String)| (k, v))
                .collect(),
        }
    }

    fn valid_outcome(game: &str) -> Outcome {
        Outcome {
            game_id: game.to_owned(),
            object_id: "o".to_owned(),
            valid: true,
            error: None,
            validation_hash: Some("sha256:ab".to_owned()),
            checks: vec![("structure", "ok".to_owned())],
        }
    }

    /// KAT: invalid-mix → records in row order with EXACT lineage keys;
    /// valid rows skipped allocation-free (no records); checks cloned.
    #[test]
    fn quarantine_mix_keeps_row_order_and_lineage() {
        let rows = [raw("b"), raw("a"), raw("c")];
        let bad_b = invalid_outcome("gb", ErrorClass::Structure, Some(0));
        let bad_a = invalid_outcome("ga", ErrorClass::DoraShape, Some(1));
        let good_c = valid_outcome("gc");
        let outcomes = [
            MorselOutcome {
                object_id: "b",
                outcome: &bad_b,
                game_events: Some(7),
            },
            MorselOutcome {
                object_id: "a",
                outcome: &bad_a,
                game_events: Some(3),
            },
            MorselOutcome {
                object_id: "c",
                outcome: &good_c,
                game_events: Some(5),
            },
        ];
        let q = quarantine_invalid(&rows, &outcomes).unwrap();
        assert_eq!(q.len(), 2);
        // Row order (manifest sorts separately).
        assert_eq!(q[0].object_id, "b");
        assert_eq!(q[1].object_id, "a");
        assert_eq!(q[0].game_id.as_deref(), Some("gb"));
        assert_eq!(q[0].error_class, "structure");
        assert_eq!(q[0].error_event_index, Some(0));
        assert_eq!(q[1].error_class, "dora_shape");
        assert_eq!(q[1].error_event_index, Some(1));
        // Lineage keys EXACT (8 keys, oracle order).
        assert_eq!(LINEAGE_KEYS.len(), 8);
        let l = &q[0].lineage;
        assert_eq!(l.confidential_source_id, "src-1");
        assert_eq!(l.authorization_attestation_id, "att-1");
        assert_eq!(l.permitted_purpose, ["train"]);
        assert_eq!(l.parent_ids, ["sha256:00"]);
        assert_eq!(l.game_events, 7);
        assert_eq!(
            l.validation_checks,
            [("structure".to_owned(), "ok".to_owned())]
        );
        assert!(q[0].validation_hash.is_none());
    }

    /// KAT: missing outcome is a hard error (no silent skip).
    #[test]
    fn quarantine_missing_outcome_hard_errors() {
        let rows = [raw("a"), raw("ghost")];
        let bad = invalid_outcome("ga", ErrorClass::Legality, None);
        let outcomes = [MorselOutcome {
            object_id: "a",
            outcome: &bad,
            game_events: None,
        }];
        let e = quarantine_invalid(&rows, &outcomes).unwrap_err();
        assert_eq!(e, QuarantineErr::MissingOutcome("ghost".to_owned()));
    }

    /// KAT: taxonomy closed — `REASON_* 0-8` reuse gate strings, `WALK_*`
    /// `10+` reuse walk strings, the gap + unknowns fold to `"other"`.
    #[test]
    fn quarantine_reason_taxonomy_closed() {
        assert_eq!(
            (0..=8u8).map(reason_class).collect::<Vec<_>>(),
            [
                "ok",
                "framing",
                "unknown-event",
                "framing",
                "framing",
                "wall-bearing",
                "bare-dora",
                "double-ron",
                "turn-order"
            ]
        );
        assert_eq!(reason_class(9), "other");
        assert_eq!(
            (10..=21u8).map(reason_class).collect::<Vec<_>>(),
            [
                "turn-order",
                "tile-conservation",
                "claim-no-offer",
                "draw-past-wall",
                "kyushu-ambiguous",
                "unmapped-ryukyoku-reason",
                "engine-desync",
                "unknown-event",
                "bare-dora",
                "double-ron",
                "action-id-unresolved",
                "past-terminal"
            ]
        );
        assert_eq!(reason_class(22), "other");
        assert_eq!(reason_class(255), "other");
        // normalize: closed sets pass through, drift folds.
        assert_eq!(normalize_error_class("tile_conservation"), "tile_conservation");
        assert_eq!(normalize_error_class("blank_line"), "blank_line");
        assert_eq!(normalize_error_class("wall-bearing"), "wall-bearing");
        assert_eq!(normalize_error_class("past-terminal"), "past-terminal");
        assert_eq!(normalize_error_class("nope"), "other");
        assert_eq!(normalize_error_class(""), "other");
    }

    /// KAT: decode-failure record lineage — no game id, zero events, empty
    /// checks, normalized class.
    #[test]
    fn quarantine_decode_failure_lineage() {
        let r = RawRow {
            object_id: "o".to_owned(),
            packaged_object_id: "p".to_owned(),
            confidential_source_id: "s".to_owned(),
            authorization_attestation_id: "a".to_owned(),
            permitted_purpose: vec![],
            parent_ids: vec![],
        };
        let q = QuarantinedRecord::decode_failure(&r, "blank_line", Some(2));
        assert!(q.game_id.is_none());
        assert_eq!(q.error_class, "blank_line");
        assert_eq!(q.error_event_index, Some(2));
        assert_eq!(q.lineage.game_events, 0);
        assert!(q.lineage.validation_checks.is_empty());
        assert!(q.validation_hash.is_none());
        let q = QuarantinedRecord::decode_failure(&r, "drifted-class", None);
        assert_eq!(q.error_class, "other");
    }

    /// KAT: manifest sorted by `object_id` + digest verifies against the
    /// canon seal; write is atomic (tmp gone, file parses).
    #[test]
    fn quarantine_manifest_sorted_and_digest() {
        let dir = std::env::temp_dir().join("hydra-feed-quarantine-kat");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("q.json");
        let r2 = QuarantinedRecord::decode_failure(&raw("b"), "json", Some(0));
        let bad = invalid_outcome("ga", ErrorClass::RedIdentity, Some(4));
        let rows = [raw("a")];
        let outcomes = [MorselOutcome {
            object_id: "a",
            outcome: &bad,
            game_events: Some(9),
        }];
        let mut recs = quarantine_invalid(&rows, &outcomes).unwrap();
        recs.push(r2);
        // Inserted b-after-a… push order is [a, b]; reverse to prove sorting.
        recs.reverse();
        let digest = write_quarantine_manifest(&path, &recs).unwrap();
        assert!(digest.starts_with("sha256:"));
        let bytes = std::fs::read(&path).unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        let arr = v["quarantined"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["object_id"], Value::from("a"));
        assert_eq!(arr[1]["object_id"], Value::from("b"));
        assert_eq!(v["schema_version"], Value::from(MANIFEST_SCHEMA_VERSION));
        // Digest verifies: seal over the payload WITHOUT the digest field.
        let mut payload = v.as_object().unwrap().clone();
        payload.remove("digest");
        let expect = crate::digest::of_canonical(&Value::Object(payload)).unwrap();
        assert_eq!(digest, expect);
        assert_eq!(v["digest"], Value::from(digest.as_str()));
        // Atomic publish: no tmp sibling left behind.
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        assert!(!Path::new(&tmp).exists());
        let _ = std::fs::remove_file(&path);
    }
}
