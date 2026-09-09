//! S4 gates over the vendored fixture corpus (no probe env needed).
//!
//! `tests/fixtures/s4/` holds two good wall-less games plus one game per
//! quarantine code. These tests pin the Slice-4 acceptance set in CI
//! without `HYDRA2_REPLAY_PROBE`:
//! - batch=1 vs batch=N byte-identical streams with equal stats;
//! - quarantine multiset + per-file codes, and the zero-partial-rows
//!   invariant (no drained row belongs to a quarantined game);
//! - the 13-field envelope + actor-string + chosen-int-or-null shape on
//!   every drained row, and the chosen-null-iff-unresolved iff on every
//!   driver row (via the public `replay_game_text` entry point).

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use hydra2_replay_rs::{
    ACTOR_FIELDS, ActionTable, ReplayStream, StreamOpen, replay_game_text,
};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("s4")
}

/// Sorted `q-*.jsonl` stems with their pinned reason codes, in discovery
/// order. `q-truncated` (no `end_game`) maps to `framing`: the framer is
/// the first gate, so a log that never reaches `end_game` never reaches
/// the walk's terminal check. S7: `q-wall-bearing` is NO LONGER a
/// quarantine — valid 136-tile walls replay (zero rows here) with a real
/// digest; it stays vendored as the walled-ok companion and is excluded
/// from this list.
fn expected_quarantines_in_order() -> Vec<(&'static str, &'static str)> {
    vec![
        ("q-bare-dora", "bare-dora"),
        ("q-double-ron", "double-ron"),
        ("q-framing", "framing"),
        ("q-tile-conservation", "tile-conservation"),
        ("q-truncated", "framing"),
        ("q-turn-order", "turn-order"),
        ("q-unknown-event", "unknown-event"),
    ]
}

fn open_stream(batch_rows: usize) -> ReplayStream {
    ReplayStream::open(StreamOpen {
        data_dirs: vec![fixture_dir()],
        batch_rows,
        workers: 0,
        queue: 4,
        split: "train".to_string(),
        spec_hash: "s4-fixture-v1".to_string(),
    })
    .expect("open fixture stream")
}

fn drain(stream: &mut ReplayStream) -> Vec<u8> {
    let mut out = Vec::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let next = stream.next_into_raw(&mut buf).expect("next batch");
        out.extend_from_slice(&buf[..next.bytes_written]);
        if next.rows == 0 && next.games_consumed == 0 {
            break;
        }
    }
    out
}

fn action_table() -> ActionTable {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs/contracts/action_table_v1.json");
    let text = std::fs::read_to_string(&path).expect("baked action table read");
    ActionTable::load_json(&text).expect("action table load")
}

#[test]
fn s4_batch_sizes_agree_stats_and_quarantine_codes() {
    let mut one = open_stream(1);
    let payload_one = drain(&mut one);
    let stats_one = one.stats();
    let mut many = open_stream(64);
    let payload_many = drain(&mut many);
    let stats_many = many.stats();

    assert!(!payload_one.is_empty(), "good fixtures must emit rows");
    assert_eq!(
        payload_one, payload_many,
        "batch=1 vs batch=N must drain identical bytes"
    );
    assert_eq!(stats_one.open_count, 1);
    // S7: 2 wall-less goods + 1 walled-ok (`q-wall-bearing`, zero rows).
    assert_eq!(stats_one.games_ok, 3);
    assert_eq!(
        stats_one.games_quarantined,
        expected_quarantines_in_order().len() as u64
    );
    assert_eq!(stats_one.rows_out as usize, payload_one.split(|b| *b == b'\n').filter(|l| !l.is_empty()).count());

    // Per-file codes in sorted discovery order (goods sort before q-*).
    let ordered: Vec<String> = many
        .quarantines()
        .iter()
        .map(|q| q.reason_code.clone())
        .collect();
    let expected: Vec<String> = expected_quarantines_in_order()
        .iter()
        .map(|(_, code)| code.to_string())
        .collect();
    assert_eq!(ordered, expected, "quarantine codes pin per file in order");

    // Zero-partial-rows: no drained row belongs to a quarantined game.
    let quarantined: HashSet<&str> = many
        .quarantines()
        .iter()
        .map(|q| q.game_id.as_str())
        .collect();
    for line in payload_one.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        let row: serde_json::Value = serde_json::from_slice(line).expect("row json");
        let game = row.get("game_id").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            !quarantined.contains(game),
            "quarantined game {game} contributed a row"
        );
    }
}

#[test]
fn s4_fixture_rows_carry_exact_envelope_and_chosen_iff() {
    let mut stream = open_stream(64);
    let payload = drain(&mut stream);
    let mut actor_fields = ACTOR_FIELDS.to_vec();
    actor_fields.sort_unstable();
    for line in payload.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        let row: serde_json::Value = serde_json::from_slice(line).expect("row json");
        let obj = row.as_object().expect("row object");
        let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(keys, actor_fields, "exactly the 13 ACTOR_FIELDS");
        // actor_observation rides as a JSON string decoding to an object.
        let text = row
            .get("actor_observation")
            .and_then(|v| v.as_str())
            .expect("actor_observation string");
        let doc: serde_json::Value = serde_json::from_str(text).expect("actor doc parses");
        assert!(doc.is_object(), "actor_observation decodes to an object");
        // chosen rides as int or null (never a string, bool, or float).
        match row.get("chosen_action_id") {
            Some(serde_json::Value::Number(n)) if n.is_u64() || n.is_i64() => {}
            Some(serde_json::Value::Null) => {}
            other => panic!("chosen_action_id must be int or null, got {other:?}"),
        }
    }

    // Chosen-null-iff-unresolved over every driver row of the good games.
    let table = action_table();
    let mut rows_seen = 0u64;
    for file in ["good-a.jsonl", "good-b.jsonl"] {
        let text =
            std::fs::read_to_string(fixture_dir().join(file)).expect("good fixture read");
        let rows = replay_game_text(&text, file.trim_end_matches(".jsonl"), &table)
            .expect("good fixture replays clean");
        for row in &rows {
            rows_seen += 1;
            assert_eq!(
                row.chosen_action_id.is_none(),
                row.chosen_unresolved.is_some(),
                "chosen null iff unresolved at {}",
                row.decision_id,
            );
        }
    }
    assert!(rows_seen > 0);
    // The quarantined fixtures yield zero rows each (whole-game quarantine).
    let mut quarantined_rows = BTreeMap::new();
    for (stem, _) in expected_quarantines_in_order() {
        let text =
            std::fs::read_to_string(fixture_dir().join(format!("{stem}.jsonl")))
                .expect("quarantine fixture read");
        match replay_game_text(&text, stem, &table) {
            Ok(rows) => {
                quarantined_rows.insert(stem, rows.len());
            }
            Err(_) => {}
        }
    }
    assert!(
        quarantined_rows.is_empty(),
        "quarantined fixtures must yield no rows, got {quarantined_rows:?}"
    );
}
