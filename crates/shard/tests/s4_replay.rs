#![allow(clippy::expect_used, clippy::unwrap_used)] // integration harness: fixture-load expects are the test idiom
//! S4 replay pins over the vendored fixture corpus (no probe env needed).
//!
//! Ported from the deleted root-handoff battery (`tests/s4_gates.rs`): the
//! two halves below exercise the live `replay_game_text` entry point
//! unchanged — the batch-agreement / envelope / per-file-code halves died
//! with the JSON `ReplayStream` (deleted drain path; per-file gate codes
//! now live in `feed::gate`'s verdict table + the bridge quarantine leg).
//!
//! - chosen-null-iff-unresolved on every driver row of the good games;
//! - the quarantined fixtures yield zero rows each (whole-game quarantine).

use std::collections::BTreeMap;
use std::path::PathBuf;

use hydra_shard::{ActionTable, replay_game_text};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/s4")
}

/// Sorted `q-*.jsonl` stems (discovery order). `q-wall-bearing` stays
/// vendored as the walled-ok companion (valid 136-tile walls replay, zero
/// rows here with a real digest) and is excluded from this list.
fn quarantined_stems() -> Vec<&'static str> {
    vec![
        "q-bare-dora",
        "q-double-ron",
        "q-framing",
        "q-tile-conservation",
        "q-truncated",
        "q-turn-order",
        "q-unknown-event",
    ]
}

fn action_table() -> ActionTable {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs/contracts/action_table_v1.json");
    let text = std::fs::read_to_string(&path).expect("baked action table read");
    ActionTable::load_json(&text).expect("action table load")
}

#[test]
fn s4_chosen_null_iff_unresolved_on_good_games() {
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
}

#[test]
fn s4_quarantined_fixtures_yield_no_rows() {
    let table = action_table();
    let mut quarantined_rows = BTreeMap::new();
    for stem in quarantined_stems() {
        let text =
            std::fs::read_to_string(fixture_dir().join(format!("{stem}.jsonl")))
                .expect("quarantine fixture read");
        if let Ok(rows) = replay_game_text(&text, stem, &table) {
            quarantined_rows.insert(stem, rows.len());
        }
    }
    assert!(
        quarantined_rows.is_empty(),
        "quarantined fixtures must yield no rows, got {quarantined_rows:?}"
    );
}
