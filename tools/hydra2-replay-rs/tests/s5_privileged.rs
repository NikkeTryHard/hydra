//! S5 privileged battery (PG-S5, no engine): privileged rows join actor rows.
//!
//! `tests/fixtures/s5/` holds four small wall-less games: `scores` quad,
//! `final_scores`-alias quad, missing quad (fail closed), tied quad (fail
//! closed). Per game the battery asserts: privileged length joins actor
//! length, decision ids pair 1:1, ranks equal the terminal quad, `wall_id`
//! is absent wall-less (never a placeholder) and passes through verbatim
//! when walled, and the actor-privileged firewall holds both directions.

use std::collections::HashSet;
use std::path::PathBuf;

use hydra2_replay_rs::{
    ACTOR_FIELDS, ActionTable, FORBIDDEN_IN_ACTOR, PrivilegedRow, RowProvenance,
    adapter_hash, check_privileged_label, count_row_decisions, expand_privileged_rows,
    load_baked_rules, parse_game, replay_game_text,
};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("s5")
}

fn real_table() -> ActionTable {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../configs/contracts/action_table_v1.json");
    let text = std::fs::read_to_string(&path).expect("pinned action table read");
    ActionTable::load_json(&text).expect("pinned action table verifies")
}

fn fixture_text(stem: &str) -> String {
    std::fs::read_to_string(fixture_dir().join(format!("{stem}.jsonl"))).expect("fixture read")
}

fn provenance() -> RowProvenance {
    RowProvenance {
        source_object_id: "s5-battery".to_string(),
        split: "train".to_string(),
        rules_hash: load_baked_rules().expect("baked rules").rules_hash,
        adapter_hash: adapter_hash(),
        action_table_hash: real_table().digest,
    }
}

/// Actor rows + privileged rows + engine-free count for one fixture.
fn joined(stem: &str, wall_id: Option<&str>) -> (Vec<String>, Vec<PrivilegedRow>) {
    let text = fixture_text(stem);
    let table = real_table();
    let actor = replay_game_text(&text, stem, &table).expect("s5 fixture replays clean");
    let game = parse_game(&text, stem).expect("s5 fixture parses");
    assert_eq!(
        count_row_decisions(&game).expect("count"),
        actor.len(),
        "engine-free count joins the walk on {stem}"
    );
    let privileged =
        expand_privileged_rows(&game, "train", wall_id).expect("s5 privileged expands");
    let actor_ids: Vec<String> = actor.iter().map(|row| row.decision_id.clone()).collect();
    (actor_ids, privileged)
}

#[test]
fn s5_lengths_join_and_ids_pair_one_to_one() {
    for stem in ["priv-a", "priv-alias"] {
        let (actor_ids, privileged) = joined(stem, None);
        assert_eq!(actor_ids.len(), privileged.len(), "lengths join on {stem}");
        assert!(!privileged.is_empty(), "s5 fixtures must emit rows");
        // Positional ids share the parsed game stem (explicit event id
        // else the `game-<sha12(object_id)>` fallback).
        let text = fixture_text(stem);
        let game_id = parse_game(&text, stem).expect("parse").game_id;
        for (seq, (actor_id, priv_row)) in actor_ids.iter().zip(privileged.iter()).enumerate() {
            assert_eq!(actor_id, &priv_row.decision_id, "ids pair up on {stem}");
            assert_eq!(
                priv_row.decision_id,
                format!("{game_id}:d{seq:04}"),
                "ids stay positional on {stem}"
            );
        }
    }
}
#[test]
fn s5_ranks_equal_terminal_quad() {
    let (_, privileged) = joined("priv-a", None);
    assert!(privileged.iter().all(|row| row.ranks == [1, 2, 3, 4]));
    let (_, privileged) = joined("priv-alias", None);
    assert!(privileged.iter().all(|row| row.ranks == [4, 3, 2, 1]));
}

#[test]
fn s5_wall_less_wall_id_absent_never_placeholder() {
    for stem in ["priv-a", "priv-alias"] {
        let (_, privileged) = joined(stem, None);
        for row in &privileged {
            assert!(row.wall_id.is_none(), "wall-less rows bind no wall");
            let label = row.privileged_label();
            let keys: HashSet<&str> = label
                .as_object()
                .expect("label object")
                .keys()
                .map(|k| k.as_str())
                .collect();
            assert_eq!(keys, HashSet::from(["ranks", "split"]));
            assert!(
                !label.as_object().unwrap().contains_key("wall_id"),
                "wall_id key OMITS, never null/empty"
            );
            check_privileged_label(&label).expect("wall-less label gates");
        }
    }
}

#[test]
fn s5_walled_wall_id_passes_through_verbatim() {
    // The walled digest itself is minted once S6 wall support lands; the
    // S5 rule is exact passthrough (never invented, never normalized).
    let digest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let (_, privileged) = joined("priv-a", Some(digest));
    assert!(!privileged.is_empty());
    for row in &privileged {
        assert_eq!(row.wall_id.as_deref(), Some(digest));
        let label = row.privileged_label();
        assert_eq!(
            label.get("wall_id").and_then(|v| v.as_str()),
            Some(digest)
        );
        check_privileged_label(&label).expect("walled label gates");
    }
    // Empty wall / empty split fail closed (mirrors the oracle messages).
    let text = fixture_text("priv-a");
    let game = parse_game(&text, "priv-a").expect("parse");
    assert!(expand_privileged_rows(&game, "train", Some("")).unwrap_err().contains("wall_id"));
    assert!(expand_privileged_rows(&game, "", None).unwrap_err().contains("split"));
}

#[test]
fn s5_missing_and_tied_quads_fail_closed() {
    for (stem, needle) in [("priv-missing", "no 4-score quad"), ("priv-tie", "distinct")] {
        let text = fixture_text(stem);
        // Actor replay stays clean: only the privileged path fails.
        let table = real_table();
        assert!(replay_game_text(&text, stem, &table).is_ok(), "{stem} is actor-clean");
        let game = parse_game(&text, stem).expect("parse");
        let err = expand_privileged_rows(&game, "train", None).unwrap_err();
        assert!(err.contains(needle), "{stem}: unexpected error {err:?}");
    }
}

#[test]
fn s5_leakage_scan_passes_both_directions() {
    let text = fixture_text("priv-a");
    let table = real_table();
    let actor = replay_game_text(&text, "priv-a", &table).expect("replay");
    let game = parse_game(&text, "priv-a").expect("parse");
    let privileged = expand_privileged_rows(&game, "train", None).expect("expand");
    let provenance = provenance();

    // Actor -> privileged: envelope is exactly the 13 ACTOR_FIELDS and the
    // actor-observation document carries zero forbidden keys.
    let mut actor_fields = ACTOR_FIELDS.to_vec();
    actor_fields.sort_unstable();
    for row in &actor {
        let decision = row.to_decision_json(&provenance);
        let mut keys: Vec<&str> = decision
            .as_object()
            .expect("decision object")
            .keys()
            .map(|k| k.as_str())
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, actor_fields, "exactly the 13 ACTOR_FIELDS");
        let text = decision["actor_observation"].as_str().expect("actor string");
        let doc: serde_json::Value = serde_json::from_str(text).expect("actor parses");
        for key in doc.as_object().expect("actor object").keys() {
            assert!(
                !FORBIDDEN_IN_ACTOR.contains(&key.as_str()),
                "actor doc leaks {key:?}"
            );
        }
        for key in ["ranks", "privileged_label", "wall_id"] {
            assert!(!decision.as_object().unwrap().contains_key(key), "actor leaks {key}");
        }
    }

    // Privileged -> actor: labels gate, and privileged JSON is exactly
    // {decision_id, privileged_label} with matching ids.
    assert_eq!(actor.len(), privileged.len());
    for (actor_row, priv_row) in actor.iter().zip(privileged.iter()) {
        assert_eq!(actor_row.decision_id, priv_row.decision_id);
        let doc = priv_row.to_privileged_json();
        let mut keys: Vec<&str> = doc
            .as_object()
            .expect("privileged object")
            .keys()
            .map(|k| k.as_str())
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, ["decision_id", "privileged_label"]);
        check_privileged_label(&doc["privileged_label"]).expect("label gates");
        // No actor observation content crosses into the privileged row.
        assert!(!doc.to_string().contains("concealed"));
    }
}
