//! Framed MJAI event vocabulary for the wall-less replay driver.
//!
//! Reference (read-only): `src/hydra2/data/replay_expand.py` vocabulary sets
//! (`_START_TYPES`, `_END_TYPES`, `_SKIP_TYPES`, `_ROW_TYPES`, `_CLAIM_TYPES`)
//! and `src/hydra2/data/decode.py` framing rules. Order reused; code fresh.
//!
//! Tenhou MJAI carries no wall field; rows are seat-filtered projections of
//! these log events. Anything outside the allowlist quarantines the game
//! (fail closed, never skipped silently).

use serde::Deserialize;

/// Framing aliases accepted at the game boundaries (mirrors decode.py).
pub const START_TYPES: [&str; 4] = ["start_game", "startGame", "game_start", "start"];
pub const END_TYPES: [&str; 4] = ["end_game", "endGame", "game_end", "end"];

/// Non-decision events: consumed for ordering/round tracking, never rows.
pub const SKIP_TYPES: [&str; 10] = [
    "start_kyoku",
    "tsumo",
    "dora",
    "reach_accepted",
    "ryukyoku",
    "end_kyoku",
    "start_game",
    "startGame",
    "game_start",
    "start",
];

/// One row per occurrence (`reach` collapses with its following `dahai`).
pub const ROW_TYPES: [&str; 7] = [
    "dahai",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "hora",
];

/// Row kinds that claim the live discard offer.
pub const CLAIM_TYPES: [&str; 3] = ["chi", "pon", "daiminkan"];

/// Log event kinds skipped when looking behind/ahead through a kyoku.
pub const TRANSPARENT_KINDS: [&str; 2] = ["dora", "reach_accepted"];

/// Envelope kinds emitted into actor-visible histories (v1 wall-less path).
/// `call_resolved` is server-private and never visible; it is therefore
/// absent here by construction.
pub const HISTORY_PUBLIC_KINDS: [&str; 18] = [
    "game_start",
    "round_start",
    "turn_advance",
    "draw_tile",
    "discard",
    "call_window",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "dora_revealed",
    "riichi_accepted",
    "ron",
    "tsumo",
    "draw_end",
    "abortive_draw",
    "round_end",
];

/// One framed MJAI log event. Unknown extra fields are tolerated (the
/// decoder is liberal in what it accepts); missing/invalid required fields
/// fail closed at use sites with named reasons.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MjaiEvent {
    /// Event discriminator (`type` in JSON).
    #[serde(rename = "type")]
    pub type_: String,
    pub actor: Option<serde_json::Value>,
    pub target: Option<serde_json::Value>,
    pub pai: Option<String>,
    pub consumed: Option<Vec<String>>,
    #[serde(default)]
    pub tsumogiri: bool,
    #[serde(default)]
    pub tsumo: bool,
    pub bakaze: Option<String>,
    pub kyoku: Option<serde_json::Value>,
    pub honba: Option<serde_json::Value>,
    pub kyotaku: Option<serde_json::Value>,
    pub oya: Option<serde_json::Value>,
    pub scores: Option<Vec<serde_json::Value>>,
    /// `end_game` score alias read by the privileged ranks path
    /// (`_final_scores` checks `scores`, then `final_scores`).
    pub final_scores: Option<Vec<serde_json::Value>>,
    pub tehais: Option<Vec<Vec<String>>>,
    pub dora_marker: Option<String>,
    pub deltas: Option<Vec<serde_json::Value>>,
    pub reason: Option<String>,
    pub game_id: Option<String>,
    #[serde(rename = "gameId")]
    pub game_id_alt: Option<String>,
}

impl MjaiEvent {
    /// Validated seat 0..3. JSON bools and out-of-range ints are hard
    /// failures naming the event kind (mirrors `_require_actor`).
    pub fn actor_seat(&self, what: &str) -> Result<u8, String> {
        seat_of(&self.actor, &self.type_, what)
    }

    /// Validated optional target seat 0..3.
    pub fn target_seat(&self) -> Result<u8, String> {
        seat_of(&self.target, &self.type_, "target")
    }
}

fn seat_of(value: &Option<serde_json::Value>, kind: &str, what: &str) -> Result<u8, String> {
    match value {
        Some(serde_json::Value::Number(n)) => {
            if let Some(seat) = n.as_u64()
                && seat <= 3
            {
                return u8::try_from(seat).map_err(|_| format!("{kind}: {what} must be 0..3"));
            }
            Err(format!("{kind}: {what} must be 0..3"))
        }
        _ => Err(format!("{kind}: {what} must be 0..3")),
    }
}

fn is_start(kind: &str) -> bool {
    START_TYPES.contains(&kind)
}

fn is_end(kind: &str) -> bool {
    END_TYPES.contains(&kind)
}

pub fn is_row_kind(kind: &str) -> bool {
    ROW_TYPES.contains(&kind)
}

pub fn is_claim_kind(kind: &str) -> bool {
    CLAIM_TYPES.contains(&kind)
}

pub fn is_transparent(kind: &str) -> bool {
    TRANSPARENT_KINDS.contains(&kind)
}

/// Strict framing check mirroring `decode_game_object` line rules:
/// non-empty payload, no blank lines, every line a JSON object with a
/// `type`, first line a start alias, last line an end alias, exactly one
/// start and one end marker.
pub fn frame_events(text: &str, object_id: &str) -> Result<Vec<MjaiEvent>, String> {
    if !text.ends_with('\n') {
        return Err(format!(
            "framing {object_id}: payload must end with newline"
        ));
    }
    let body = &text[..text.len() - 1];
    if body.is_empty() {
        return Err(format!("framing {object_id}: empty payload"));
    }
    let mut events = Vec::new();
    for (idx, line) in body.split('\n').enumerate() {
        if line.trim().is_empty() {
            return Err(format!(
                "framing {object_id}: blank line at index {idx}"
            ));
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| format!("framing {object_id}: line {idx} invalid JSON: {e}"))?;
        let obj = value
            .as_object()
            .ok_or_else(|| format!("framing {object_id}: line {idx} must be a JSON object"))?;
        if !obj.contains_key("type") {
            return Err(format!(
                "framing {object_id}: line {idx} missing 'type' field"
            ));
        }
        let event: MjaiEvent = serde_json::from_value(value)
            .map_err(|e| format!("framing {object_id}: line {idx} malformed: {e}"))?;
        if event.type_.is_empty() {
            return Err(format!("framing {object_id}: line {idx} without a string type"));
        }
        events.push(event);
    }
    let first = events.first().map(|e| e.type_.as_str()).unwrap_or("");
    let last = events.last().map(|e| e.type_.as_str()).unwrap_or("");
    if !is_start(first) {
        return Err(format!(
            "framing {object_id}: first record must be start_game, got {first:?}"
        ));
    }
    if !is_end(last) {
        return Err(format!(
            "framing {object_id}: last record must be end_game, got {last:?}"
        ));
    }
    // decode.py counts the canonical spellings only (aliases pass the
    // boundary check but do not satisfy the exactly-one rule).
    let starts = events
        .iter()
        .filter(|e| matches!(e.type_.as_str(), "start_game" | "startGame" | "game_start"))
        .count();
    let ends = events
        .iter()
        .filter(|e| matches!(e.type_.as_str(), "end_game" | "endGame" | "game_end"))
        .count();
    if starts != 1 || ends != 1 {
        return Err(format!(
            "framing {object_id}: exactly one start_game and one end_game required, got {starts}/{ends}"
        ));
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn framed(lines: &[&str]) -> String {
        let mut s = lines.join("\n");
        s.push('\n');
        s
    }

    #[test]
    fn accepts_minimal_game() {
        let text = framed(&[
            r#"{"type":"start_game"}"#,
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"3m","tehais":[["1m"],["2m"],["3m"],["4m"]]}"#,
            r#"{"type":"end_kyoku"}"#,
            r#"{"type":"end_game"}"#,
        ]);
        let events = frame_events(&text, "probe").unwrap();
        assert_eq!(events.len(), 4);
        assert_eq!(events[1].tehais.as_ref().unwrap().len(), 4);
    }

    #[test]
    fn rejects_fail_closed() {
        // Missing trailing newline.
        assert!(frame_events(r#"{"type":"start_game"}"#, "x").is_err());
        // Blank line.
        assert!(frame_events("{\"type\":\"start_game\"}\n\n{\"type\":\"end_game\"}\n", "x").is_err());
        // Non-object line.
        assert!(frame_events("[1]\n{\"type\":\"end_game\"}\n", "x").is_err());
        // Missing type.
        assert!(frame_events("{}\n{\"type\":\"end_game\"}\n", "x").is_err());
        // Wrong boundaries.
        assert!(
            frame_events(
                "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"1m\"}\n{\"type\":\"end_game\"}\n",
                "x"
            )
            .is_err()
        );
        // Doubled start.
        assert!(
            frame_events(
                "{\"type\":\"start_game\"}\n{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n",
                "x"
            )
            .is_err()
        );
    }

    #[test]
    fn actor_validation_rejects_bools() {
        let ev: MjaiEvent = serde_json::from_str(r#"{"type":"dahai","actor":true}"#).unwrap();
        assert!(ev.actor_seat("dahai").is_err());
        let ev: MjaiEvent = serde_json::from_str(r#"{"type":"dahai","actor":4}"#).unwrap();
        assert!(ev.actor_seat("dahai").is_err());
        let ev: MjaiEvent = serde_json::from_str(r#"{"type":"dahai","actor":2}"#).unwrap();
        assert_eq!(ev.actor_seat("dahai").unwrap(), 2);
    }
    // --- Slice S1 pins: vocabulary + framing corpus vs the Python oracle ---
    //
    // Reference literals (read-only):
    // - `src/hydra2/data/replay_expand.py`: `_START_TYPES`, `_END_TYPES`,
    //   `_SKIP_TYPES` (6 kyoku kinds + every `_START_TYPES` alias),
    //   `_ROW_TYPES`, `_CLAIM_TYPES`.
    // - `src/hydra2/data/decode.py`: `decode_game_object` line rules.
    // - `replay_expand._event_kind`: empty/non-string `type` fails there, so
    //   the combined (decode + kind) oracle verdict is quarantine — matching
    //   this module's earlier framing rejection of those lines.

    #[test]
    fn vocabulary_matches_replay_expand() {
        let mut start = START_TYPES.to_vec();
        start.sort_unstable();
        assert_eq!(start, ["game_start", "start", "startGame", "start_game"]);
        let mut end = END_TYPES.to_vec();
        end.sort_unstable();
        assert_eq!(end, ["end", "endGame", "end_game", "game_end"]);
        let mut skip = SKIP_TYPES.to_vec();
        skip.sort_unstable();
        assert_eq!(
            skip,
            [
                "dora",
                "end_kyoku",
                "game_start",
                "reach_accepted",
                "ryukyoku",
                "start",
                "startGame",
                "start_game",
                "start_kyoku",
                "tsumo",
            ]
        );
        let mut row = ROW_TYPES.to_vec();
        row.sort_unstable();
        assert_eq!(
            row,
            ["ankan", "chi", "dahai", "daiminkan", "hora", "kakan", "pon"]
        );
        let mut claim = CLAIM_TYPES.to_vec();
        claim.sort_unstable();
        assert_eq!(claim, ["chi", "daiminkan", "pon"]);
        // `reach` collapses with its following `dahai`: a row trigger, never
        // a row kind itself, on both sides.
        assert!(!ROW_TYPES.contains(&"reach"));
        assert!(!SKIP_TYPES.contains(&"reach"));
    }

    #[test]
    fn boundary_alias_matrix_matches_decode() {
        // decode.py counts only the three canonical spellings toward the
        // exactly-one rule; the bare `start`/`end` aliases pass the boundary
        // check but satisfy no count. Oracle verdicts (cross-checked):
        // ("start","end") -> quarantine 0/0, ("start","end_game") -> 0/1.
        let canonical_start = ["start_game", "startGame", "game_start"];
        let canonical_end = ["end_game", "endGame", "game_end"];
        for s in canonical_start {
            for e in canonical_end {
                let text = framed(&[
                    &format!(r#"{{"type":"{s}"}}"#),
                    &format!(r#"{{"type":"{e}"}}"#),
                ]);
                assert!(frame_events(&text, "probe").is_ok(), "{s}/{e}");
            }
        }
        for (s, e) in [
            ("start", "end"),
            ("start", "end_game"),
            ("start_game", "end"),
        ] {
            let text = framed(&[
                &format!(r#"{{"type":"{s}"}}"#),
                &format!(r#"{{"type":"{e}"}}"#),
            ]);
            let err = frame_events(&text, "probe").expect_err("bare alias must quarantine");
            assert!(err.contains("exactly one"), "{s}/{e}: {err}");
        }
        // A bare `start` alongside the canonical marker is tolerated by both
        // sides (the count is still exactly one).
        let text = framed(&[
            r#"{"type":"start"}"#,
            r#"{"type":"start_game"}"#,
            r#"{"type":"end_game"}"#,
        ]);
        assert!(frame_events(&text, "probe").is_ok());
    }

    #[test]
    fn framing_corpus_matches_oracle_verdicts() {
        // (name, payload, expect_ok). Oracle verdicts from
        // `decode_game_object` (+ `_event_kind` for the `type_*` rows),
        // cross-checked against the live oracle. Unknown kinds and `reach`
        // pass framing on both sides (unknown kinds quarantine later at row
        // dispatch, outside this module).
        let cases: &[(&str, String, bool)] = &[
            ("minimal", framed(&[r#"{"type":"start_game"}"#, r#"{"type":"end_game"}"#]), true),
            ("unknown_kind", framed(&[r#"{"type":"start_game"}"#, r#"{"type":"foobar"}"#, r#"{"type":"end_game"}"#]), true),
            ("reach_passthrough", framed(&[r#"{"type":"start_game"}"#, r#"{"type":"reach","actor":0}"#, r#"{"type":"end_game"}"#]), true),
            ("double_start", framed(&[r#"{"type":"start_game"}"#, r#"{"type":"start_game"}"#, r#"{"type":"end_game"}"#]), false),
            ("double_end", framed(&[r#"{"type":"start_game"}"#, r#"{"type":"end_game"}"#, r#"{"type":"end_game"}"#]), false),
            ("missing_type", framed(&[r#"{"type":"start_game"}"#, r#"{}"#, r#"{"type":"end_game"}"#]), false),
            ("non_object", framed(&[r#"{"type":"start_game"}"#, r#"[1]"#, r#"{"type":"end_game"}"#]), false),
            ("invalid_json", framed(&[r#"{"type":"start_game"}"#, r#"{bad}"#, r#"{"type":"end_game"}"#]), false),
            ("wrong_first", framed(&[r#"{"type":"tsumo","actor":0,"pai":"1m"}"#, r#"{"type":"end_game"}"#]), false),
            ("wrong_last", framed(&[r#"{"type":"start_game"}"#, r#"{"type":"tsumo","actor":0,"pai":"1m"}"#]), false),
            ("truncated_no_end", framed(&[r#"{"type":"start_game"}"#, r#"{"type":"tsumo","actor":0,"pai":"1m"}"#]), false),
            ("blank_middle", "{\"type\":\"start_game\"}\n\n{\"type\":\"end_game\"}\n".to_string(), false),
            ("blank_whitespace", "{\"type\":\"start_game\"}\n   \n{\"type\":\"end_game\"}\n".to_string(), false),
            ("trailing_blank", "{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n\n".to_string(), false),
            ("missing_newline", "{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}".to_string(), false),
            ("empty_payload", "\n".to_string(), false),
            // Framing-strict superset: decode.py accepts these middle lines
            // but `_event_kind` rejects them, so the combined oracle verdict
            // is quarantine — matching this module's earlier rejection.
            ("empty_type", framed(&[r#"{"type":"start_game"}"#, r#"{"type":""}"#, r#"{"type":"end_game"}"#]), false),
            ("non_string_type", framed(&[r#"{"type":"start_game"}"#, r#"{"type":123}"#, r#"{"type":"end_game"}"#]), false),
        ];
        for (name, text, expect_ok) in cases {
            let verdict = frame_events(text, "probe").is_ok();
            assert_eq!(verdict, *expect_ok, "corpus case {name}");
        }
    }

    #[test]
    fn framing_reasons_name_the_fault() {
        let err =
            frame_events("{\"type\":\"start_game\"}\n\n{\"type\":\"end_game\"}\n", "x").unwrap_err();
        assert!(err.contains("blank line"), "{err}");
        let err = frame_events("{}\n{\"type\":\"end_game\"}\n", "x").unwrap_err();
        assert!(err.contains("missing 'type'"), "{err}");
        let err = frame_events("[1]\n{\"type\":\"end_game\"}\n", "x").unwrap_err();
        assert!(err.contains("must be a JSON object"), "{err}");
        let err =
            frame_events("{\"type\":\"start_game\"}\n{bad}\n{\"type\":\"end_game\"}\n", "x")
                .unwrap_err();
        assert!(err.contains("invalid JSON"), "{err}");
        let err = frame_events("{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}", "x").unwrap_err();
        assert!(err.contains("must end with newline"), "{err}");
        let err = frame_events(
            "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"1m\"}\n{\"type\":\"end_game\"}\n",
            "x",
        )
        .unwrap_err();
        assert!(err.contains("first record must be"), "{err}");
        let err = frame_events(
            "{\"type\":\"start_game\"}\n{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"1m\"}\n",
            "x",
        )
        .unwrap_err();
        assert!(err.contains("last record must be"), "{err}");
        let err = frame_events(
            "{\"type\":\"start_game\"}\n{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n",
            "x",
        )
        .unwrap_err();
        assert!(err.contains("exactly one"), "{err}");
        let err = frame_events(
            "{\"type\":\"start_game\"}\n{\"type\":123}\n{\"type\":\"end_game\"}\n",
            "x",
        )
        .unwrap_err();
        assert!(err.contains("malformed"), "{err}");
    }
}
