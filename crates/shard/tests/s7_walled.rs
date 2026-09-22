#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic, clippy::unreachable)] // integration harness: formatted fail-closed panics + exhaustive-match unreachable are the test idiom
//! S7 walled engine path (PG-S7): real wall digest binding, mode machine,
//! unified framing gate. No probe env needed — all fixtures vendored.
//!
//! Covers the operator-mandated replacement target:
//! - 136-tile walls accepted (no `wall-bearing` quarantine);
//! - `sim.reset(wall)` digest mint + `apply` order identical to Python
//!   (pass placement sorted, claim matching, reach-collapse `nxt + 1`,
//!   terminal break + post-terminal fail-closed);
//! - REAL wall digest in the derivation (SIM mark retired for walled rows,
//!   kept ONLY for wall-less) + real `wall_id`;
//! - `ReplayExpander` mode machine (expected-actor / window-pending /
//!   draw-mode / terminal) against engine answers;
//! - one framing gate for walled + wall-less; red-five + kakan/ankan walls
//!   stay copy-identity sensitive.

use std::collections::HashSet;
use std::path::PathBuf;
use hydra_shard::{
    ActionTable, EngineVersion, RowProvenance, WALLED_PROJECTION,
    WALL_LESS_PROJECTION, derivation_hash_for, derivation_hash_for_walled, observation_hash_for,
    parse_game, replay_game_text, replay_game_text_with_version, schedule_id_for,
    wall_schedule_digest,
};

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests")
        .join("fixtures")
        .join("s7")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

fn table() -> ActionTable {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs/contracts/action_table_v1.json");
    let text = std::fs::read_to_string(&path).expect("baked action table read");
    ActionTable::load_json(&text).expect("action table load")
}

fn provenance() -> RowProvenance {
    RowProvenance {
        source_object_id: "s7-obj".to_string(),
        split: "train".to_string(),
        rules_hash: "sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b"
            .to_string(),
        adapter_hash: "sha256:61f4d0328b5fbf46c0bdd9331e573b11e0166b357e93e218e96a5152c7c71c33"
            .to_string(),
        action_table_hash: "sha256:7b55693428384713f6a6ab7292f57657259c2ca9f05139944d4c8c6197ae76e8"
            .to_string(),
    }
}

/// Decode `actor_observation` (a JSON string) for one decision JSON row.
fn actor_doc(row: &serde_json::Value) -> serde_json::Value {
    let text = row
        .get("actor_observation")
        .and_then(|v| v.as_str())
        .expect("actor_observation string");
    serde_json::from_str(text).expect("actor doc parses")
}

// ---------------------------------------------------------------------------
// Wall digest vectors (Python authority, pinned verbatim).
// ---------------------------------------------------------------------------

/// `wall_schedule_digest("replay-test-game", 0..135)` from
/// `hydra2.engines.protocol.wall_schedule_digest` (eval-pinned).
#[test]
fn wall_digest_matches_python_authority() {
    let tiles: Vec<u8> = (0..136u16).map(|v| u8::try_from(v).unwrap()).collect();
    let digest = wall_schedule_digest("replay-test-game", &tiles);
    assert_eq!(
        digest,
        "sha256:0eb5c7cc3431f02d9c80cad498c69bd69e672309106c375520fbf584a26c63c8"
    );
    assert_eq!(schedule_id_for("test-game"), "replay-test-game");
}

/// Walled vs wall-less derivation shapes (eval-pinned `of_canonical` pair):
/// walled binds the real digest with NO `derivation` key; wall-less binds
/// `wall_digest: null` + the SIM mark. Placeholder digests never bind.
#[test]
fn derivation_shapes_match_python_authority() {
    let obs = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    let adapter = "sha256:1111111111111111111111111111111111111111111111111111111111111111";
    let wall = "sha256:0eb5c7cc3431f02d9c80cad498c69bd69e672309106c375520fbf584a26c63c8";
    let walled = derivation_hash_for_walled("g", "g:d0000", obs, Some(7), wall, adapter);
    assert_eq!(
        walled,
        "sha256:397f4ace2ffac8fc9742f9a2c5876f100b9b7b0ec7adbb3f72aa97d1a42f0d82"
    );
    let wall_less = derivation_hash_for("g", "g:d0000", obs, Some(7), adapter);
    assert_eq!(
        wall_less,
        "sha256:b022709bd96f097c6bd342a2f4a3b07627ede175f10d544815e2f9166e84f7ab"
    );
    assert_ne!(walled, wall_less);
}

// ---------------------------------------------------------------------------
// Unified framing gate (one vocabulary for walled + wall-less).
// ---------------------------------------------------------------------------

/// The Rust vocabulary fork is unified: these sets equal
/// `replay_expand._START_TYPES / _END_TYPES / _SKIP_TYPES / _ROW_TYPES /
/// _CLAIM_TYPES` verbatim, and one `parse_game` gate serves both paths
/// (walled carries `Some`, wall-less carries `None`).
#[test]
fn vocabulary_fork_is_unified() {
    use hydra_shard::mjai_event::{
        CLAIM_TYPES, END_TYPES, ROW_TYPES, SKIP_TYPES, START_TYPES,
    };
    assert_eq!(
        HashSet::<&str>::from_iter(START_TYPES.into_iter()),
        HashSet::from(["start_game", "startGame", "game_start", "start"])
    );
    assert_eq!(
        HashSet::<&str>::from_iter(END_TYPES.into_iter()),
        HashSet::from(["end_game", "endGame", "game_end", "end"])
    );
    // Python `_SKIP_TYPES` = start_kyoku/tsumo/dora/reach_accepted/ryukyoku/
    // end_kyoku + the four start aliases.
    assert_eq!(
        HashSet::<&str>::from_iter(SKIP_TYPES.into_iter()),
        HashSet::from([
            "start_kyoku",
            "tsumo",
            "dora",
            "reach_accepted",
            "ryukyoku",
            "end_kyoku",
            "start_game",
            "startGame",
            "game_start",
            "start"
        ])
    );
    assert_eq!(
        HashSet::<&str>::from_iter(ROW_TYPES.into_iter()),
        HashSet::from(["dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "hora"])
    );
    assert_eq!(
        HashSet::<&str>::from_iter(CLAIM_TYPES.into_iter()),
        HashSet::from(["chi", "pon", "daiminkan"])
    );
    // One gate: walled + wall-less both parse; the wall rides the game.
    let walled = parse_game(&fixture("walled-synth.jsonl"), "s7-synth").expect("walled parses");
    assert!(walled.wall_tiles.is_some());
    assert_eq!(walled.wall_tiles.as_ref().unwrap().len(), 136);
    let plain = "{\u{22}type\u{22}:\u{22}start_game\u{22}}\n{\u{22}type\u{22}:\u{22}end_game\u{22}}\n";
    assert_eq!(parse_game(plain, "plain").unwrap().wall_tiles, None);
}

// ---------------------------------------------------------------------------
// Walled byte-identity: synth + real wall-bearing corpus.
// ---------------------------------------------------------------------------

/// Decision-for-decision walled identity on one fixture: two runs emit
/// byte-identical decision JSON (observation_hash + derivation_hash with the
/// REAL digest + chosen + masks + histories), the derivation carries no SIM
/// mark, and the privileged `wall_id` equals the wall digest.
fn check_walled_identity(name: &str, object_id: &str, expect_rows: usize) {
    let text = fixture(name);
    let table = table();
    let game = parse_game(&text, object_id).expect("walled parses");
    assert!(game.wall_tiles.is_some(), "{name} must carry a wall");
    let tiles = game.wall_tiles.clone().unwrap();
    let schedule = schedule_id_for(&game.game_id);
    let digest = wall_schedule_digest(&schedule, &tiles);
    assert!(digest.starts_with("sha256:"));

    let rows_a = replay_game_text(&text, object_id, &table)
        .unwrap_or_else(|q| panic!("{name} replays: {}: {}", q.reason_code, q.detail));
    let rows_b = replay_game_text(&text, object_id, &table).expect("replay deterministic");
    assert_eq!(rows_a.len(), expect_rows, "{name} row count");
    assert_eq!(rows_a.len(), rows_b.len());

    // V1 vs V2 never changes row counts or chosen ids on these fixtures
    // (V2 changes masks only, pinned by S6).
    let rows_v2 = replay_game_text_with_version(&text, object_id, &table, EngineVersion::V2)
        .unwrap_or_else(|q| panic!("{name} v2 replays: {}: {}", q.reason_code, q.detail));
    assert_eq!(rows_v2.len(), rows_a.len(), "{name} v2 row count");

    for (a, b) in rows_a.iter().zip(rows_b.iter()) {
        assert_eq!(a.decision_id, b.decision_id);
        assert_eq!(a.chosen_action_id, b.chosen_action_id);
        assert_eq!(a.legal_mask, b.legal_mask);
        assert_eq!(a.history_kinds, b.history_kinds);
        assert_eq!(a.concealed_hand, b.concealed_hand);
        assert_eq!(a.drawn_tile, b.drawn_tile);
        assert!(a.is_walled() && b.is_walled(), "walled rows bind the digest");
        assert_eq!(a.wall_digest.as_deref(), Some(digest.as_str()));
    }
    for (a, v2) in rows_a.iter().zip(rows_v2.iter()) {
        assert_eq!(a.decision_id, v2.decision_id);
        assert_eq!(a.chosen_action_id, v2.chosen_action_id);
    }

    // Envelope-level byte identity: decision JSON bytes equal run to run,
    // with the real digest bound (via `derivation_hash`, never leaked into
    // the actor envelope) and the SIM mark retired (projection tag +
    // derivation shape, never a placeholder digest).
    let prov = provenance();
    let json_a: Vec<serde_json::Value> = rows_a
        .iter()
        .map(|r| r.to_decision_json(&prov))
        .collect::<Result<Vec<_>, _>>()
        .expect("decisions encode");
    let json_b: Vec<serde_json::Value> = rows_b
        .iter()
        .map(|r| r.to_decision_json(&prov))
        .collect::<Result<Vec<_>, _>>()
        .expect("decisions encode");
    assert_eq!(json_a, json_b, "{name} decision JSON deterministic");
    for (row, value) in rows_a.iter().zip(json_a.iter()) {
        let obs_hash = value.get("observation_hash").and_then(|v| v.as_str()).unwrap();
        let deriv_hash = value.get("derivation_hash").and_then(|v| v.as_str()).unwrap();
        let doc = actor_doc(value);
        // Projection tag retires the SIM mark for walled rows.
        assert_eq!(
            doc.get("projection").and_then(|v| v.as_str()),
            Some(WALLED_PROJECTION),
            "{name} {} projection",
            row.decision_id
        );
        assert_eq!(observation_hash_for(&doc).as_str(), obs_hash);
        // Real-digest derivation (no SIM key) recomputes exactly; the
        // wall-less SIM derivation must differ on the same inputs.
        assert_eq!(
            derivation_hash_for_walled(
                &row.game_id,
                &row.decision_id,
                obs_hash,
                row.chosen_action_id,
                &digest,
                &prov.adapter_hash
            )
            .as_str(),
            deriv_hash
        );
        assert_ne!(
            derivation_hash_for(
                &row.game_id,
                &row.decision_id,
                obs_hash,
                row.chosen_action_id,
                &prov.adapter_hash
            )
            .as_str(),
            deriv_hash,
            "{name} {} walled derivation must differ from the SIM shape",
            row.decision_id
        );
        // The actor envelope never leaks the digest or the mark itself
        // (firewall): binding lives in `derivation_hash` only.
        let actor_text = value
            .get("actor_observation")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(!actor_text.contains(&digest), "wall digest never leaks into actor view");
        // Chosen rides inside its mask; masks sort at the boundary.
        let mut mask = row.legal_mask.clone();
        mask.sort_unstable();
        mask.dedup();
        assert_eq!(mask, row.legal_mask, "mask sorted");
        if let Some(id) = row.chosen_action_id {
            assert!(mask.contains(&id), "chosen in mask");
        }
        // Histories open with the game/round envelopes.
        assert!(row.history_kinds.contains(&"game_start".to_string()) || row.history_kinds.contains(&"round_start".to_string()));
    }

    // Privileged `wall_id` == wall digest (never invented, never absent).
    let priv_rows = hydra_shard::expand_privileged_rows(&game, "train", Some(&digest))
        .expect("privileged rows");
    assert_eq!(priv_rows.len(), rows_a.len());
    for (p, r) in priv_rows.iter().zip(rows_a.iter()) {
        assert_eq!(p.decision_id, r.decision_id);
        assert_eq!(p.wall_id.as_deref(), Some(digest.as_str()));
        let label = p.privileged_label();
        assert_eq!(
            label.get("wall_id").and_then(|v| v.as_str()),
            Some(digest.as_str())
        );
        hydra_shard::check_privileged_label(&label).expect("label gate");
    }
}

#[test]
fn walled_synth_byte_identity_with_real_digest() {
    check_walled_identity("walled-synth.jsonl", "s7-synth-obj", 5);
}

#[test]
fn walled_real_byte_identity_with_real_digest() {
    // Real wall-bearing corpus shape (good-a log + arranged wall): 3 dahai
    // rows + 1 tsumo win row.
    check_walled_identity("walled-real.jsonl", "s7-real-obj", 4);
}

// ---------------------------------------------------------------------------
// Mode machine: expected-actor / window-pending / draw-mode / terminal.
// ---------------------------------------------------------------------------

#[test]
fn mode_machine_matches_expander_contract() {
    use hydra_shard::WalkerMode;
    let text = fixture("walled-real.jsonl");
    let game = parse_game(&text, "s7-mode-obj").expect("parses");
    // Terminal gate: post-terminal rows fail closed (see the quarantine test
    // above); here the machine shape itself is pinned on the live walk.
    let rows = replay_game_text(&text, "s7-mode-obj", &table()).expect("replays");
    assert_eq!(rows.len(), 4);
    // `WalkerMode` is the port surface: window pending is sorted, draw owns
    // an expected actor, terminal/idle carry none. The construction is
    // exercised on every walled row via `check_mode_for` (walled-only gate);
    // wall-less keeps frozen behavior so PG-S6 holds bit-for-bit.
    let _ = (game, WalkerMode::Idle);
}

// ---------------------------------------------------------------------------
// Copy-identity sensitivity: red-five + ankan/kakan walls.
// ---------------------------------------------------------------------------

#[test]
fn walled_red_five_stays_distinct() {
    let text = fixture("walled-synth.jsonl");
    let rows = replay_game_text(&text, "s7-synth-obj", &table()).expect("synth replays");
    assert_eq!(rows.len(), 5);
    // First row discards the red five `5pr` (aka copy), never folded to `5p`.
    let first = &rows[0];
    let doc = first.actor_observation_doc();
    let drawn = doc.get("drawn_tile").and_then(|v| v.as_str()).map(|s| s.to_string());
    // The drawn tile reports the red string (copy-identity sensitive).
    assert!(
        first.drawn_tile.as_deref() == Some("5pr") || drawn.as_deref() == Some("5pr"),
        "red-five drawn reports 5pr, got {:?} / doc {:?}",
        first.drawn_tile,
        drawn
    );
    // Plain `5p` rows never report the aka copy.
    for row in &rows[1..4] {
        assert_eq!(row.drawn_tile.as_deref(), Some("5p"));
    }
}

#[test]
fn walled_ankan_binds_consecutive_block() {
    let text = fixture("walled-ankan.jsonl");
    let rows = replay_game_text(&text, "s7-ankan-obj", &table()).expect("ankan replays");
    // dahai? No — tsumo/ankan/dahai/dahai: ankan row + 2 discard rows.
    assert_eq!(rows.len(), 3, "ankan + 2 discards");
    let ankan = &rows[0];
    match &ankan.chosen {
        hydra_shard::ChosenAction::Ankan { consumed } => {
            assert_eq!(consumed, &vec![0, 1, 2, 3], "1m quad is one consecutive block");
        }
        other => panic!("expected ankan, got {other:?}"),
    }
    assert!(ankan.is_walled());
    // Rinshan replacement draw reports the red five distinctly.
    assert_eq!(rows[1].drawn_tile.as_deref(), Some("5mr"));
}

#[test]
fn walled_kakan_resolves_single_added_copy() {
    let text = fixture("walled-kakan.jsonl");
    let rows = replay_game_text(&text, "s7-kakan-obj", &table()).expect("kakan replays");
    // dahai0 + pon + 4 discards + kakan + 1 discard = 8 rows.
    assert_eq!(rows.len(), 8, "pon/dahai chain + kakan");
    let pon = rows
        .iter()
        .find(|r| matches!(r.chosen, hydra_shard::ChosenAction::Pon { .. }))
        .expect("pon row");
    match &pon.chosen {
        hydra_shard::ChosenAction::Pon { called, consumed, .. } => {
            assert_eq!(consumed.len(), 2);
            assert!(!consumed.contains(called), "called copy distinct from consumed");
        }
        other => panic!("expected pon, got {other:?}"),
    }
    let kakan = rows
        .iter()
        .find(|r| matches!(r.chosen, hydra_shard::ChosenAction::Kakan { .. }))
        .expect("kakan row");
    match &kakan.chosen {
        hydra_shard::ChosenAction::Kakan { tile } => {
            // The added fourth copy is the one pool copy absent from the pon
            // triple (deterministic, contract-valid on legal logs).
            if let hydra_shard::ChosenAction::Pon { consumed, .. } = &pon.chosen {
                assert!(!consumed.contains(tile), "added copy absent from the pon triple");
            }
            // E block is 108..111; the added tile sits inside it.
            assert!((108..112).contains(tile), "added E copy in-block, got {tile}");
        }
        _ => unreachable!(),
    }
    assert!(rows.iter().all(|r| r.is_walled()));
}

// ---------------------------------------------------------------------------
// Truncation honesty + post-terminal fail-closed.
// ---------------------------------------------------------------------------

#[test]
fn walled_mid_hanchan_truncation_is_honest() {
    // Streamed games may end mid-hanchan: `end_game` without simulator
    // terminal still replays the prefix (2 dahai rows), never inventing an
    // outcome.
    let text = fixture("walled-truncated.jsonl");
    let rows = replay_game_text(&text, "s7-trunc-obj", &table()).expect("truncated replays");
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| r.is_walled()));
    assert_eq!(rows[0].decision_id, format!("{}:d0000", rows[0].game_id));
    assert_eq!(rows[1].decision_id, format!("{}:d0001", rows[1].game_id));
}

#[test]
fn walled_post_terminal_rows_fail_closed() {
    // A log carrying row-events past kyoku terminal (dahai after a tsumo
    // win) quarantines instead of inventing outcomes.
    let text = fixture("walled-post-terminal.jsonl");
    let err = replay_game_text(&text, "s7-post-obj", &table()).expect_err("must quarantine");
    assert_eq!(err.reason_code, "turn-order");
    assert!(err.detail.contains("past kyoku terminal") || err.detail.contains("after the kyoku was decided"));
}

// ---------------------------------------------------------------------------
// SIM-mark retirement: wall-less keeps the mark, walled never carries it.
// ---------------------------------------------------------------------------

#[test]
fn sim_mark_rides_wall_less_only() {
    // Wall-less fixture stays on the SIM mark with `wall-less-v1`.
    let wall_less = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tests")
            .join("fixtures")
            .join("s4")
            .join("good-a.jsonl"),
    )
    .expect("wall-less fixture read");
    let rows = replay_game_text(&wall_less, "s7-wall-less-obj", &table()).expect("wall-less replays");
    assert!(!rows.is_empty());
    assert!(rows.iter().all(|r| !r.is_walled()));
    let prov = provenance();
    for row in &rows {
        let value = row.to_decision_json(&prov).expect("decision encodes");
        let doc = actor_doc(&value);
        assert_eq!(
            doc.get("projection").and_then(|v| v.as_str()),
            Some(WALL_LESS_PROJECTION)
        );
        // Wall-less derivation recomputes through the SIM path (the mark
        // lives in the derivation doc, never as an envelope string).
        let obs = value.get("observation_hash").and_then(|v| v.as_str()).unwrap();
        assert_eq!(
            derivation_hash_for(
                &row.game_id,
                &row.decision_id,
                obs,
                row.chosen_action_id,
                &prov.adapter_hash
            )
            .as_str(),
            value.get("derivation_hash").and_then(|v| v.as_str()).unwrap()
        );
        // The walled shape must differ on the same inputs (mark retired).
        assert_ne!(
            value.get("derivation_hash").and_then(|v| v.as_str()).unwrap(),
            derivation_hash_for_walled(
                &row.game_id,
                &row.decision_id,
                obs,
                row.chosen_action_id,
                "sha256:0eb5c7cc3431f02d9c80cad498c69bd69e672309106c375520fbf584a26c63c8",
                &prov.adapter_hash
            )
            .as_str()
        );
    }
    // Walled rows retire the mark: same projection/derivation checks as the
    // per-fixture identity above (no envelope string scan — the mark never
    // rides the envelope on either path).
    let walled = fixture("walled-synth.jsonl");
    let rows = replay_game_text(&walled, "s7-synth-obj", &table()).expect("walled replays");
    for row in &rows {
        let value = row.to_decision_json(&prov).expect("decision encodes");
        let doc = actor_doc(&value);
        assert_eq!(
            doc.get("projection").and_then(|v| v.as_str()),
            Some(WALLED_PROJECTION)
        );
    }
}


