//! S5 provenance battery (PG-S5, no engine): digests byte-equal the oracle.
//!
//! Rules/adapter/action-table digests minted by Rust must equal the Python
//! oracle values on the pinned artifacts, and every tampered authority
//! must fail closed (wrong digest or hard error, never a silent fallback).
//! The stream battery proves drained rows bind the minted digests.

use std::path::PathBuf;

use hydra2_replay_rs::{
    ACTOR_FIELDS, ActionTable, PINNED_ACTION_TABLE_HASH, PINNED_ADAPTER_HASH,
    PINNED_RULES_HASH, ReplayStream, StreamOpen, adapter_hash, load_baked_rules,
    load_rules_manifest,
};

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn pinned_table_text() -> String {
    std::fs::read_to_string(crate_dir().join("../../configs/contracts/action_table_v1.json"))
        .expect("pinned table read")
}

#[test]
fn s5_adapter_digest_byte_equals_oracle() {
    assert_eq!(adapter_hash(), PINNED_ADAPTER_HASH);
}

#[test]
fn s5_rules_digest_byte_equals_oracle() {
    let info = load_baked_rules().expect("baked rules load");
    assert_eq!(info.rules_id, "tenhou_4p_hanchan_v1");
    assert_eq!(info.rules_hash, PINNED_RULES_HASH);
    // Same bytes from disk (baked == published file).
    let text = std::fs::read_to_string(
        crate_dir().join("../../configs/rules/tenhou_4p_hanchan_v1.json"),
    )
    .expect("rules file read");
    assert_eq!(load_rules_manifest(&text).expect("disk rules load").rules_hash, PINNED_RULES_HASH);
}

#[test]
fn s5_action_table_digest_byte_equals_oracle() {
    let table = ActionTable::load_json(&pinned_table_text()).expect("pinned table verifies");
    assert_eq!(table.len, 6792);
    assert_eq!(table.digest, PINNED_ACTION_TABLE_HASH);
}

#[test]
fn s5_tampered_manifest_is_evident_and_structural_tamper_fails() {
    // One semantic byte changed, envelope + id intact: the loader stays
    // faithful (new digest) instead of echoing the pin — tampering is
    // always evident at the pin comparison, never silently accepted.
    let text = std::fs::read_to_string(
        crate_dir().join("tests/fixtures/s5/rules-tampered.json"),
    )
    .expect("tampered fixture read");
    let info = load_rules_manifest(&text).expect("tampered still parses");
    assert_eq!(info.rules_id, "tenhou_4p_hanchan_v1");
    assert_ne!(info.rules_hash, PINNED_RULES_HASH);
    // Structural tampering fails closed with an error.
    assert!(load_rules_manifest("not json").is_err());
    let baked = hydra2_replay_rs::provenance::BAKED_RULES_MANIFEST;
    assert!(load_rules_manifest(&baked[..baked.len() / 2]).is_err());
    assert!(
        load_rules_manifest(&baked.replacen("tenhou_4p_hanchan_v1", "other_rules_v9", 1)).is_err()
    );
}

#[test]
fn s5_tampered_table_fails_closed() {
    let text = pinned_table_text();
    // One template byte flipped: declared-vs-recomputed mismatch.
    let tampered = text.replacen("\"tile\":0", "\"tile\":1", 1);
    assert_ne!(tampered, text);
    let err = ActionTable::load_json(&tampered).unwrap_err();
    assert!(err.contains("declared digest"), "unexpected: {err}");
    // Declared digest flipped instead: same mismatch, other direction.
    let tampered = text.replacen(PINNED_ACTION_TABLE_HASH, &format!("sha256:{}", "0".repeat(64)), 1);
    assert!(ActionTable::load_json(&tampered).unwrap_err().contains("declared digest"));
    // Truncation and wrong envelopes fail closed.
    assert!(ActionTable::load_json(&text[..4096]).is_err());
    assert!(
        ActionTable::load_json(&text.replacen("hydra2.action_table", "hydra2.other", 1)).is_err()
    );
    assert!(ActionTable::load_json("{\"payload\":{\"actions\":[]}}").is_err());
}

#[test]
fn s5_stream_rows_bind_minted_digests() {
    let mut stream = ReplayStream::open(StreamOpen {
        data_dirs: vec![crate_dir().join("tests/fixtures/s5")],
        batch_rows: 64,
        workers: 0,
        queue: 4,
        split: "train".to_string(),
        spec_hash: "s5-legacy-pin".to_string(),
    })
    .expect("open s5 stream");
    let mut buf = vec![0u8; 1 << 20];
    let mut out = Vec::new();
    loop {
        let next = stream.next_into_raw(&mut buf).expect("next batch");
        out.extend_from_slice(&buf[..next.bytes_written]);
        if next.rows == 0 && next.games_consumed == 0 {
            break;
        }
    }
    // All four s5 games are actor-clean (privileged-only failures).
    assert_eq!(stream.stats().games_ok, 4);
    assert_eq!(stream.stats().games_quarantined, 0);
    let mut actor_fields = ACTOR_FIELDS.to_vec();
    actor_fields.sort_unstable();
    let mut rows = 0u64;
    for line in out.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        let row: serde_json::Value = serde_json::from_slice(line).expect("row json");
        let mut keys: Vec<&str> = row
            .as_object()
            .expect("row object")
            .keys()
            .map(|k| k.as_str())
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, actor_fields, "envelope stays exactly ACTOR_FIELDS");
        assert_eq!(row["split"].as_str(), Some("train"));
        // Minted authorities bind rows — the legacy spec_hash echo is gone.
        assert_eq!(row["rules_hash"].as_str(), Some(PINNED_RULES_HASH));
        assert_eq!(row["adapter_hash"].as_str(), Some(PINNED_ADAPTER_HASH));
        assert_eq!(row["action_table_hash"].as_str(), Some(PINNED_ACTION_TABLE_HASH));
        assert_ne!(row["rules_hash"].as_str(), Some("s5-legacy-pin"));
        rows += 1;
    }
    assert!(rows > 0, "s5 fixtures must drain rows");
    assert_eq!(stream.stats().rows_out, rows);
}
