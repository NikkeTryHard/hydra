//! S6 engine-answer bridge tests (no probe env needed).
//!
//! Pins the native v2 rule implementations against corpus-observed cases:
//! - chi type patterns cover all three positions (pinned by
//!   game-01cc80726381:d0282: called 56 offers pos2 + pos1 + pos0);
//! - kuikae withholding covers same-type and adjacent takes of the
//!   just-claimed chi (never pon/kan, never distance 2+);
//! - kyushu counts distinct terminals/honors (9+ offers);
//! - the v1/v2 backend selection never changes row counts or chosen ids
//!   (v2 changes masks only), on the vendored s4 good fixtures.

use std::path::PathBuf;

use hydra_shard::engine::{complete_chi, distinct_yaochu, kuikae_withheld};
use hydra_shard::{ActionTable, EngineVersion, replay_game_text_with_version};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests")
        .join("fixtures")
}

fn table() -> ActionTable {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../configs/contracts/action_table_v1.json");
    let text = std::fs::read_to_string(&path).expect("baked action table read");
    ActionTable::load_json(&text).expect("action table load")
}
/// game-01cc80726381:d0282 (called 56): all three positions (single-copy
/// takes, min == max).
#[test]
fn complete_chi_offers_all_three_positions() {
    let hand = vec![0, 36, 40, 44, 48, 53, 56, 57, 60, 64];
    assert_eq!(
        complete_chi(&hand, 56),
        vec![vec![48, 53], vec![53, 60], vec![60, 64]]
    );
}

/// Live takes sort descending-first: multi-copy types resolve to the
/// max-held take (d0190 type12 {48,51} -> 51; d0492 type16 {64,67} -> 67).
#[test]
fn complete_chi_uses_max_held_takes() {
    assert_eq!(complete_chi(&[44, 48, 51], 40), vec![vec![44, 51]]);
    assert_eq!(
        complete_chi(&[56, 64, 67, 68], 60),
        vec![vec![56, 67], vec![67, 68]]
    );
}

#[test]
fn complete_chi_honors_never() {
    assert!(complete_chi(&[108, 109, 112], 108).is_empty());
}

#[test]
fn kuikae_ignores_other_suits_and_honors() {
    // Chi in suit 0 (types 1,2,3); suit-1 and honor discards are kept.
    let chi = vec![4, 8, 12];
    let discards = vec![36, 40, 108, 4, 0];
    let mut got = kuikae_withheld(&chi, &discards);
    got.sort_unstable();
    assert_eq!(got, vec![0, 4]);
}

/// game-ef19c589332a:d0290 concealed (13) + drawn 60: 9+ distinct yaochu.
#[test]
fn kyushu_counts_distinct_terminals_and_honors() {
    let hand14 = vec![4, 17, 32, 36, 68, 72, 96, 100, 104, 108, 112, 120, 128, 60];
    assert!(distinct_yaochu(&hand14) >= 9, "kyushu hand must count 9+");
    // Closed simple-heavy hand: no yaochu.
    let plain = vec![4, 5, 8, 9, 12, 13, 20, 21, 24, 25, 40, 41, 44, 45];
    assert!(distinct_yaochu(&plain) < 9);
}

/// V2 backend changes masks only: same row counts and chosen ids as V1 on
/// the vendored good fixtures (neither backend quarantines them).
#[test]
fn v2_backend_preserves_rows_and_chosen() {
    let table = table();
    for name in ["s4/good-a.jsonl", "s4/good-b.jsonl"] {
        let text = std::fs::read_to_string(fixture_dir().join(name)).expect("fixture");
        let v1 =
            replay_game_text_with_version(&text, name, &table, EngineVersion::V1).expect("v1 rows");
        let v2 =
            replay_game_text_with_version(&text, name, &table, EngineVersion::V2).expect("v2 rows");
        assert_eq!(v1.len(), v2.len(), "row count for {name}");
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert_eq!(a.decision_id, b.decision_id);
            assert_eq!(
                a.chosen_action_id, b.chosen_action_id,
                "chosen for {}",
                a.decision_id
            );
        }
    }
}

/// A logged tsumo win without a winning shape quarantines the whole game
/// as `engine-desync` (fail-closed net, never emits). The detail carries
/// the oracle's `game + kyoku + step` vocabulary.
#[test]
fn shapeless_logged_win_quarantines_engine_desync() {
    let table = table();
    let text =
        std::fs::read_to_string(fixture_dir().join("s6/q-engine-desync.jsonl")).expect("fixture");
    for version in [EngineVersion::V1, EngineVersion::V2] {
        let err = replay_game_text_with_version(&text, "q-engine-desync", &table, version)
            .expect_err("shapeless win must quarantine");
        assert_eq!(err.reason_code, "engine-desync");
        assert!(err.detail.contains("kyoku 0"), "detail: {}", err.detail);
        assert!(err.detail.contains("hora"), "detail: {}", err.detail);
    }
}
