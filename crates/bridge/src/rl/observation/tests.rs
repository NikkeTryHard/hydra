//! Observation tests: layout totals, id maps, encoder lengths, and the
//! rebuild/encode parity that guards the sealed byte law.
//!
//! The parity test builds a small arena, seals one row through both the
//! rebuild-then-encode path and the direct facts path, and requires byte
//! equality at every bucket. It then opens a second hand and requires both
//! paths to drop every first-hand fact identically.

use super::layout::{dtype_width, plane_spec};
use super::{
    PLANE_COUNT, active_input_bytes, active_layout, encode_canonical_doc, encode_row_facts,
    furiten_id, furiten_name, materialize_canonical, phase_id, phase_name, riichi_id, riichi_name,
    row_bytes_26, wind_id,
};
use crate::rl::core::GameRow;
use crate::rl::events::{EventArena, TypedEvent, Visibility, history_kind_id};

#[test]
fn layout_totals_match_s_of_t() {
    for t in [32usize, 64, 128, 256] {
        assert_eq!(row_bytes_26(t), 7254 + 9 * t);
        // Tight per-row spec sum still equals S(T) exactly (artifact law).
        let tight: usize = plane_spec(t)
            .iter()
            .filter(|(name, _, _)| name != "uniform")
            .map(|(_, dtype, per_row)| per_row * dtype_width(dtype))
            .sum();
        assert_eq!(tight, row_bytes_26(t));
        // Cap128 carries zero padding (every region a multiple of 8).
        assert_eq!(active_input_bytes(t, 128), 128 * row_bytes_26(t) + 128 * 8);
        assert_eq!(active_input_bytes(t, 8), 8 * row_bytes_26(t) + 8 * 8);
        for b in [1usize, 3, 5, 127, 128] {
            let full = active_layout(t, b, true);
            assert_eq!(full.len(), PLANE_COUNT);
            // Every base satisfies its dtype alignment (cached Torch
            // typed views); total is the last-region end (no tail pad).
            for (_, dtype, _, off) in &full {
                assert_eq!(off % dtype_width(dtype), 0, "B={b} T={t} {dtype}");
            }
            let last = full.last().expect("nonempty layout");
            let end = last.3
                + last.2.iter().product::<usize>() * dtype_width(&last.1);
            assert_eq!(end, active_input_bytes(t, b));
            // Regions are disjoint and ordered (split_planes relies on it).
            let mut prev = 0usize;
            for (_, dtype, shape, off) in &full {
                assert!(*off >= prev);
                prev = off + shape.iter().product::<usize>() * dtype_width(dtype);
            }
        }
        // Odd B pads (bounded, deterministic); uniform stays 8-aligned last.
        let odd = active_layout(t, 3, true);
        let tight3 = 3 * row_bytes_26(t) + 3 * 8;
        assert!(active_input_bytes(t, 3) >= tight3);
        let uniform = odd.last().expect("uniform last");
        assert_eq!(uniform.0, "uniform");
        assert_eq!(uniform.3 % 8, 0);
    }
}

#[test]
fn id_maps_match_oracle() {
    assert_eq!(phase_id("draw_decision").unwrap(), 1);
    assert_eq!(phase_name(2).unwrap(), "discard_response");
    assert_eq!(furiten_id("riichi").unwrap(), 2);
    assert_eq!(furiten_name(3).unwrap(), "discard");
    assert_eq!(riichi_id("accepted").unwrap(), 2);
    assert_eq!(riichi_name(1).unwrap(), "declared");
    assert_eq!(wind_id(30).unwrap(), 3);
    assert!(phase_id("bogus").is_err());
    assert!(phase_name(9).is_err());
    assert_eq!(history_kind_id("game_start"), 0);
    assert_eq!(history_kind_id("unknown_kind"), 0);
}

#[test]
fn encode_doc_round_trip_length() {
    let doc = serde_json::json!({
        "actor": 0, "actor_can_riichi": false, "actor_can_tsumo": true,
        "actor_furiten": "none", "dealer": 0,
        "concealed_hand": [0, 1, 2],
        "dora_indicators": [4, -1, -1, -1, -1],
        "hand_number": 1, "visible_history": [{"kind": "round_start"}],
        "honba": 0, "ippatsu_active": [false, false, false, false],
        "kan_count": 0, "legal_mask": vec![true].into_iter().chain(std::iter::repeat(false)).take(6792).collect::<Vec<_>>(),
        "live_wall_tiles_remaining": 70, "own_drawn_tile": null,
        "phase": "draw_decision", "riichi_states": ["none", "none", "none", "none"],
        "riichi_sticks": 0, "round_index": 0, "round_wind": 27,
        "scores": [25000, 25000, 25000, 25000],
        "seat_winds": [27, 28, 29, 30], "turn_actor": 0,
        "visible_discards": [[], [], [], []],
        "visible_melds": [[], [], [], []],
    });
    for t in [32usize, 64, 128, 256] {
        let bytes = encode_canonical_doc(&doc, t).unwrap();
        assert_eq!(bytes.len(), 7254 + 9 * t);
    }
}

fn parity_event(kind: &str, seq: u64, actor: Option<u8>, tile: Option<u8>) -> TypedEvent {
    let (visibility, visible_to) = if kind == "draw_tile" {
        (Visibility::ActorPrivate, vec![actor.expect("drawer")])
    } else {
        (Visibility::Public, vec![0, 1, 2, 3])
    };
    TypedEvent {
        game_id: "g".to_string(),
        sequence: seq,
        kind: kind.to_string(),
        actor,
        visibility,
        visible_to,
        tile,
        action_id: None,
        source_seat: None,
        consumed_tiles: vec![],
        offered_action_ids: vec![],
        accepted_action_ids: vec![],
        round_index: Some(0),
        scores: Some([25000, 25000, 25000, 25000]),
        reason: None,
        public_delta: vec![],
        rules_hash: "sha256:00".to_string(),
        schema_hash: "sha256:02".to_string(),
    }
}

#[test]
fn facts_path_matches_doc_path_byte_for_byte() {
    let mut arena = EventArena::new();
    arena.admit(parity_event("game_start", 0, None, None)).unwrap();
    let mut round = parity_event("round_start", 1, Some(0), None);
    round.actor = Some(0);
    arena.admit(round).unwrap();
    let mut dora = parity_event("dora_revealed", 2, None, Some(4));
    dora.round_index = None;
    dora.scores = None;
    arena.admit(dora).unwrap();
    let mut advance = parity_event("turn_advance", 3, Some(1), None);
    advance.round_index = None;
    advance.scores = None;
    arena.admit(advance).unwrap();
    let mut drawn = parity_event("draw_tile", 4, Some(1), Some(17));
    drawn.round_index = None;
    drawn.scores = None;
    arena.admit(drawn).unwrap();
    let mut discard = parity_event("discard", 5, Some(1), Some(18));
    discard.action_id = Some(61);
    discard.round_index = None;
    discard.scores = None;
    arena.admit(discard).unwrap();
    let row = GameRow {
        wall_id: "w".to_string(),
        game_id: "g".to_string(),
        round_id: "g:0".to_string(),
        trajectory_step: 0,
        actor: 2,
        decision_id: "g:5".to_string(),
        sequence: 5,
        row_id: [0u8; 32],
        round_index: 0,
        round_wind: 27,
        honba: 0,
        kyotaku: 0,
        dealer: 0,
        scores: [25000, 25000, 25000, 25000],
        live_remaining: 65,
        hand_number: 1,
        turn_actor: 2,
        phase_id: 1,
        concealed: vec![0, 1, 2, 4, 5],
        drawn: Some(16),
        dora5: [4, -1, -1, -1, -1],
        ippatsu_bits: 0,
        furiten_id: 0,
        can_tsumo: true,
        can_riichi: false,
        legal_ids: vec![0, 61],
        legal_mask_bits: vec![],
        selected: 61,
        selected_logp: -0.5,
        value4: [0.1, 0.2, 0.3, 0.4],
        cursor_before: 0,
        cursor_after: 8,
        obs_hash: [0u8; 32],
        event_prefix_len: arena.len() as u32,
        derivation: [0u8; 32],
        action_table_hash: [0u8; 32],
        action_table_hash_hex: "sha256:01".to_string(),
        behavior_ckpt: [0u8; 32],
        transition_provenance: [0u8; 32],
        rules_id: "test-rules".to_string(),
        rules_hash: "sha256:00".to_string(),
        event_schema_hash: "sha256:02".to_string(),
        observation_schema_hash: "sha256:03".to_string(),
        packet_boundary_hash: "sha256:04".to_string(),
        return_g4: None,
        raw_advantage_f32: None,
    };
    for t in [32usize, 64, 128, 256] {
        let doc = materialize_canonical(&row, &arena).unwrap();
        let via_doc = encode_canonical_doc(&doc, t).unwrap();
        let via_facts = encode_row_facts(&row, &arena, t).unwrap();
        assert_eq!(via_facts, via_doc);
        assert_eq!(via_facts.len(), 7254 + 9 * t);
    }
    // Second hand: the hand-scoped window (after the latest round-end) drops
    // every first-hand fact on both paths identically.
    arena.admit(parity_event("round_end", 6, None, None)).unwrap();
    let mut round2 = parity_event("round_start", 7, Some(0), None);
    round2.actor = Some(0);
    arena.admit(round2).unwrap();
    let mut discard2 = parity_event("discard", 8, Some(3), Some(99));
    discard2.action_id = Some(62);
    discard2.round_index = None;
    discard2.scores = None;
    arena.admit(discard2).unwrap();
    let mut row2 = row.clone();
    row2.actor = 0;
    row2.event_prefix_len = arena.len() as u32;
    for t in [32usize, 64, 128, 256] {
        let doc = materialize_canonical(&row2, &arena).unwrap();
        let via_doc = encode_canonical_doc(&doc, t).unwrap();
        let via_facts = encode_row_facts(&row2, &arena, t).unwrap();
        assert_eq!(via_facts, via_doc);
        // First-hand discard (seat 1, tile 18) and first-hand dora stay out.
        let rivers = doc.get("visible_discards").and_then(|v| v.as_array()).unwrap();
        for river in rivers {
            let tiles: Vec<u64> =
                river.as_array().unwrap().iter().filter_map(|v| v.as_u64()).collect();
            assert!(!tiles.contains(&18));
        }
        let history = doc.get("visible_history").and_then(|v| v.as_array()).unwrap();
        assert!(history.iter().all(|e| {
            e.get("sequence").and_then(|s| s.as_u64()).unwrap_or(0) >= 7
        }));
    }
    // A kakan replaces the matching pon without reordering other melds;
    // appending it leaves stale source metadata and double-counts tiles.
    let mut pon = parity_event("pon", 9, Some(2), Some(70));
    pon.action_id = Some(300);
    pon.round_index = None;
    pon.scores = None;
    pon.source_seat = Some(3);
    pon.consumed_tiles = vec![68, 69];
    arena.admit(pon).unwrap();
    let mut kakan = parity_event("kakan", 10, Some(2), Some(71));
    kakan.action_id = Some(301);
    kakan.round_index = None;
    kakan.scores = None;
    arena.admit(kakan).unwrap();
    row2.event_prefix_len = arena.len() as u32;
    let doc = materialize_canonical(&row2, &arena).unwrap();
    assert_eq!(doc["visible_melds"][2], serde_json::json!([{
        "kind": "kakan", "owner": 2, "source_seat": null,
        "called_tile": 70, "tiles": [68, 69, 70, 71],
        "meld_id": "kakan:68.69.70.71",
    }]));
    assert_eq!(doc["kan_count"], 1);
    assert_eq!(encode_row_facts(&row2, &arena, 32).unwrap(), encode_canonical_doc(&doc, 32).unwrap());
}
