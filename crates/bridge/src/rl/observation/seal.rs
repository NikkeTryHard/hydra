//! Canonical observation materialization and seal encoding.
//!
//! This module rebuilds the sealed view of one captured row. The capture row
//! carries scalars (scores, winds, counters, the concealed hand, the drawn
//! tile, the legal ids) plus the length of the event prefix it was captured
//! at; everything else (rivers, melds, revealed dora, riichi states, the
//! visible history slice) is replayed deterministically from that event
//! prefix. Identical inputs always yield byte-identical outputs.
//!
//! Replay is scoped to the current hand: it starts after the latest round-end
//! event below the prefix, so no prior-hand fact can leak in. The first hand
//! keeps the game-start event because no round-end precedes it.
//!
//! Two encoders produce the sealed feature bytes, and they agree byte for
//! byte: one maps the rebuilt canonical document to bytes, the other encodes
//! directly from the row plus the event prefix with integer-only replay. Both
//! emit exactly `7254 + 9 * T` bytes for bucket `T` and fail closed on any
//! length or range violation instead of substituting values.

use super::{
    DORA_WIDTH, ObservationError, furiten_id, furiten_name, phase_id, phase_name, riichi_id,
    row_bytes_26, wind_id,
};
use crate::rl::actions::BASELINE_ACTIONS;
use crate::rl::core::GameRow;
use crate::rl::events::{EventArena, history_kind_id, meld_doc_value, visible_to_actor};

/// Rebuild one bounded canonical observation document from a captured row
/// plus its event prefix. Rivers, melds, dora, riichi states, and the visible
/// history slice are replayed from the prefix scoped to the current hand;
/// scalars ride in the row. The document field order matches the identity
/// document order, and identical inputs give byte-identical documents.
pub fn materialize_canonical(
    row: &GameRow,
    arena: &EventArena,
) -> Result<serde_json::Value, ObservationError> {
    let prefix = row.event_prefix_len as usize;
    if prefix > arena.len() {
        return Err(ObservationError(format!(
            "row prefix {prefix} exceeds arena {}",
            arena.len()
        )));
    }
    let mut rivers: [Vec<u32>; 4] = [vec![], vec![], vec![], vec![]];
    let mut melds: [Vec<serde_json::Value>; 4] = [vec![], vec![], vec![], vec![]];
    let mut history: Vec<serde_json::Value> = vec![];
    let mut riichi = vec!["none".to_string(); 4];
    let mut kan_count = 0u32;
    // Hand-scoped replay: start after the latest round-end below the prefix so
    // rivers, melds, dora, riichi, kan, and history never carry prior-hand
    // facts. The first hand keeps the game-start event because no round-end
    // precedes it.
    let start = arena.hand_window_start(prefix);
    for index in start..prefix {
        let event = arena.get(index).ok_or_else(|| {
            ObservationError(format!("prefix index {index} outside arena"))
        })?;
        if visible_to_actor(event, row.actor) {
            history.push(event.to_json_value());
        }
        match event.kind.as_str() {
            "discard" => {
                if let (Some(actor), Some(tile)) = (event.actor, event.tile) {
                    rivers[actor as usize].push(u32::from(tile));
                }
            }
            "riichi_declared" => {
                if let Some(actor) = event.actor {
                    riichi[actor as usize] = "declared".to_string();
                }
            }
            "riichi_accepted" => {
                if let Some(actor) = event.actor {
                    riichi[actor as usize] = "accepted".to_string();
                }
            }
            "chi" | "pon" | "daiminkan" | "ankan" | "kakan" => {
                if let Some(actor) = event.actor {
                    kan_count += u32::from(matches!(
                        event.kind.as_str(),
                        "daiminkan" | "ankan" | "kakan"
                    ));
                    if event.kind == "kakan" {
                        // Added kan upgrades one existing pon in place; appending
                        // a new meld duplicates tiles and loses the called tile.
                        let added = event.tile.ok_or_else(|| ObservationError("kakan lacks added tile".to_string()))?;
                        let prior = melds[actor as usize].iter_mut().find(|meld| {
                            meld.get("kind").and_then(serde_json::Value::as_str) == Some("pon")
                                && meld.get("tiles").and_then(serde_json::Value::as_array)
                                    .is_some_and(|tiles| tiles.iter().all(|tile| tile.as_u64().is_some_and(|tile| tile / 4 == u64::from(added / 4))))
                        }).ok_or_else(|| ObservationError("kakan lacks matching prior pon".to_string()))?;
                        let mut tiles: Vec<u8> = prior.get("tiles").and_then(serde_json::Value::as_array)
                            .ok_or_else(|| ObservationError("prior pon tiles missing".to_string()))?
                            .iter().map(|tile| tile.as_u64().map(|tile| tile as u8)
                                .ok_or_else(|| ObservationError("prior pon tile malformed".to_string())))
                            .collect::<Result<_, _>>()?;
                        tiles.push(added);
                        let called = prior.get("called_tile").and_then(serde_json::Value::as_u64).map(|tile| tile as u8);
                        *prior = meld_doc_value("kakan", actor, None, called, &tiles);
                        continue;
                    }
                    let mut tiles: Vec<u32> = event.consumed_tiles.iter().map(|t| u32::from(*t)).collect();
                    if let Some(called) = event.tile {
                        tiles.push(u32::from(called));
                    }
                    tiles.sort_unstable();
                    melds[actor as usize].push(meld_doc_value(
                        &event.kind,
                        actor,
                        event.source_seat,
                        event.tile,
                        &tiles.iter().map(|t| *t as u8).collect::<Vec<_>>(),
                    ));
                }
            }
            _ => {}
        }
    }
    // Shared replay (see `revealed_dora`): live and seal dora agree.
    let dora = super::revealed_dora(arena, prefix);
    let mut mask = vec![false; BASELINE_ACTIONS];
    for id in &row.legal_ids {
        *mask.get_mut(*id as usize).ok_or_else(|| {
            ObservationError(format!("legal id {id} outside census"))
        })? = true;
    }
    // Hoisted: `json!` parses `[0, 1, 2, 3]` as a JSON array literal, so a
    // Rust array `.map` chain cannot appear inline (macro sees `.` after `]`).
    let ippatsu: Vec<bool> = [0u8, 1, 2, 3]
        .iter()
        .map(|i| row.ippatsu_bits & (1 << i) != 0)
        .collect();
    // Canonical seat winds derive from the canonical dealer: seat `s` holds
    // tile `27 + ((dealer + s) % 4)` (East/South/West/North permutation).
    let seat_winds: Vec<i64> = [0u8, 1, 2, 3]
        .iter()
        .map(|o| 27 + ((u16::from(row.dealer) + u16::from(*o)) % 4) as i64)
        .collect();
    // The seal independently rehashes this rebuilt document; its captured
    // binding must be present or a valid native row cannot be admitted.
    let mut observation_hash = String::with_capacity(71);
    observation_hash.push_str("sha256:");
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in row.obs_hash {
        observation_hash.push(char::from(HEX[usize::from(byte >> 4)]));
        observation_hash.push(char::from(HEX[usize::from(byte & 15)]));
    }
    let mut document = serde_json::json!({
        "game_id": row.game_id,
        "decision_id": row.decision_id,
        "sequence": row.sequence,
        "actor": u32::from(row.actor),
        "rules_id": row.rules_id,
        "rules_hash": row.rules_hash,
        "action_table_hash": row.action_table_hash_hex,
        "event_schema_hash": row.event_schema_hash,
        "observation_schema_hash": row.observation_schema_hash,
        "packet_boundary_hash": row.packet_boundary_hash,
        "observation_hash": observation_hash,
        "round_index": row.round_index,
        "round_wind": row.round_wind,
        "hand_number": row.hand_number,
        "seat_winds": seat_winds,
        "honba": row.honba,
        "riichi_sticks": row.kyotaku,
        "dealer": u32::from(row.dealer),
        "scores": row.scores.to_vec(),
        "turn_actor": u32::from(row.turn_actor),
        "phase": phase_name(row.phase_id)?,
        "live_wall_tiles_remaining": row.live_remaining,
        "kan_count": kan_count,
        "ippatsu_active": ippatsu,
        "actor_furiten": furiten_name(row.furiten_id)?,
        "actor_can_tsumo": row.can_tsumo,
        "actor_can_riichi": row.can_riichi,
        "pending_declaration_discard": null,
        "concealed_hand": row.concealed.iter().map(|t| u32::from(*t)).collect::<Vec<_>>(),
        "own_drawn_tile": row.drawn.map(u32::from),
        "visible_discards": rivers,
        "visible_melds": melds,
        "riichi_states": riichi,
        "dora_indicators": dora,
        "visible_history": null,
        "legal_mask": null,
    });
    // History values are already staged; serializing them through json! clones
    // every event tree. Moving them preserves the exact document without copies.
    document["visible_history"] = serde_json::Value::Array(history);
    document["legal_mask"] = serde_json::Value::Array(
        mask.into_iter().map(serde_json::Value::Bool).collect(),
    );
    Ok(document)
}

/// Map one canonical observation document plus bucket `T` to exactly
/// `7254 + 9 * T` feature bytes, failing closed on any length mismatch.
/// Reads the same shapes the identity document emits: tile ids as numbers,
/// phase, riichi, and furiten states as literals, winds as 27..30, dora with
/// `-1` padding, and the legal mask as booleans.
pub fn encode_canonical_doc(
    doc: &serde_json::Value,
    bucket_t: usize,
) -> Result<Vec<u8>, ObservationError> {
    let get_u64 = |key: &str| -> Result<u64, ObservationError> {
        doc.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
            ObservationError(format!("doc missing int field {key:?}"))
        })
    };
    let get_i64 = |key: &str| -> Result<i64, ObservationError> {
        doc.get(key).and_then(|v| v.as_i64()).ok_or_else(|| {
            ObservationError(format!("doc missing int field {key:?}"))
        })
    };
    let get_str = |key: &str| -> Result<&str, ObservationError> {
        doc.get(key).and_then(|v| v.as_str()).ok_or_else(|| {
            ObservationError(format!("doc missing str field {key:?}"))
        })
    };
    let get_seq = |key: &str| -> Result<Vec<serde_json::Value>, ObservationError> {
        doc.get(key).and_then(|v| v.as_array()).cloned().ok_or_else(|| {
            ObservationError(format!("doc missing seq field {key:?}"))
        })
    };
    let num = |v: &serde_json::Value| -> Result<u64, ObservationError> {
        v.as_u64().ok_or_else(|| ObservationError("doc holds a non-int tile".to_string()))
    };
    let mut out = vec![0u8; row_bytes_26(bucket_t)];
    let mut at = 0usize;
    let put_i64 = |out: &mut Vec<u8>, at: &mut usize, v: i64| {
        out[*at..*at + 8].copy_from_slice(&v.to_le_bytes());
        *at += 8;
    };
    let put_i32 = |out: &mut Vec<u8>, at: &mut usize, v: i32| {
        out[*at..*at + 4].copy_from_slice(&v.to_le_bytes());
        *at += 4;
    };
    let put_bool = |out: &mut Vec<u8>, at: &mut usize, v: bool| {
        out[*at] = u8::from(v);
        *at += 1;
    };
    put_i64(&mut out, &mut at, get_u64("actor")? as i64);
    put_bool(&mut out, &mut at, doc.get("actor_can_riichi").and_then(|v| v.as_bool()).unwrap_or(false));
    put_bool(&mut out, &mut at, doc.get("actor_can_tsumo").and_then(|v| v.as_bool()).unwrap_or(false));
    put_i64(&mut out, &mut at, furiten_id(get_str("actor_furiten")?)?);
    put_i64(&mut out, &mut at, get_u64("actor")? as i64);
    let mut concealed_counts = [0i32; 34];
    for t in get_seq("concealed_hand")? {
        let kind = (num(&t)? as usize) / 4;
        *concealed_counts.get_mut(kind).ok_or_else(|| {
            ObservationError("concealed kind out of range".to_string())
        })? += 1;
    }
    for c in concealed_counts {
        put_i32(&mut out, &mut at, c);
    }
    put_i64(&mut out, &mut at, get_u64("dealer")? as i64);
    let mut dora = get_seq("dora_indicators")?;
    while dora.len() < DORA_WIDTH {
        dora.push(serde_json::Value::from(-1));
    }
    for d in dora.iter().take(DORA_WIDTH) {
        put_i32(&mut out, &mut at, d.as_i64().unwrap_or(-1) as i32);
    }
    put_i32(&mut out, &mut at, get_u64("hand_number")? as i32);
    let history = get_seq("visible_history")?;
    if history.len() > bucket_t {
        return Err(ObservationError(format!(
            "history {} exceeds bucket {bucket_t}",
            history.len()
        )));
    }
    let mut kinds = vec![0i64; bucket_t];
    let mut mask = vec![false; bucket_t];
    for (pos, event) in history.iter().enumerate() {
        let kind = event.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        kinds[pos] = history_kind_id(kind);
        mask[pos] = true;
    }
    for k in kinds {
        put_i64(&mut out, &mut at, k);
    }
    for m in mask {
        put_bool(&mut out, &mut at, m);
    }
    put_i32(&mut out, &mut at, get_u64("honba")? as i32);
    for v in get_seq("ippatsu_active")? {
        put_bool(&mut out, &mut at, v.as_bool().unwrap_or(false));
    }
    put_i32(&mut out, &mut at, get_i64("kan_count")? as i32);
    let legal = get_seq("legal_mask")?;
    if legal.len() != BASELINE_ACTIONS {
        return Err(ObservationError(format!(
            "legal_mask length {} != baseline {BASELINE_ACTIONS}",
            legal.len()
        )));
    }
    for flag in legal {
        put_bool(&mut out, &mut at, flag.as_bool().unwrap_or(false));
    }
    put_i32(&mut out, &mut at, get_u64("live_wall_tiles_remaining")? as i32);
    put_i32(
        &mut out,
        &mut at,
        doc.get("own_drawn_tile").and_then(|v| v.as_u64()).map_or(-1, |t| t as i32),
    );
    put_i64(&mut out, &mut at, phase_id(get_str("phase")?)?);
    for s in get_seq("riichi_states")? {
        put_i64(&mut out, &mut at, riichi_id(s.as_str().unwrap_or("none"))?);
    }
    put_i32(&mut out, &mut at, get_u64("riichi_sticks")? as i32);
    put_i32(&mut out, &mut at, get_u64("round_index")? as i32);
    put_i64(&mut out, &mut at, wind_id(get_i64("round_wind")?)?);
    for s in get_seq("scores")? {
        put_i32(&mut out, &mut at, s.as_i64().unwrap_or(0) as i32);
    }
    for w in get_seq("seat_winds")? {
        put_i64(&mut out, &mut at, wind_id(w.as_i64().unwrap_or(27))?);
    }
    put_i64(&mut out, &mut at, get_u64("turn_actor")? as i64);
    let mut visible_counts = [0i32; 34];
    for river in get_seq("visible_discards")? {
        for t in river.as_array().cloned().unwrap_or_default() {
            let kind = (num(&t)? as usize) / 4;
            if let Some(slot) = visible_counts.get_mut(kind) {
                *slot += 1;
            }
        }
    }
    for row_melds in get_seq("visible_melds")? {
        for m in row_melds.as_array().cloned().unwrap_or_default() {
            for t in m.get("tiles").and_then(|v| v.as_array()).cloned().unwrap_or_default() {
                let kind = (num(&t)? as usize) / 4;
                if let Some(slot) = visible_counts.get_mut(kind) {
                    *slot += 1;
                }
            }
        }
    }
    for slot in visible_counts.iter_mut() {
        if *slot > 4 {
            *slot = 4;
        }
    }
    for c in visible_counts {
        put_i32(&mut out, &mut at, c);
    }
    if at != row_bytes_26(bucket_t) {
        return Err(ObservationError(format!(
            "encode length {at} != S(T)={}",
            row_bytes_26(bucket_t)
        )));
    }
    Ok(out)
}

/// Encode directly from the capture row plus its event prefix to exactly
/// `7254 + 9 * T` bytes, with no document serialize or parse step. The fact
/// math matches the rebuild-then-encode path above (a parity test asserts
/// byte equality); prefix replay is integer-only. Rows are sealed in order
/// with growing prefixes, so per-call work is linear in the prefix length.
pub fn encode_row_facts(
    row: &GameRow,
    arena: &EventArena,
    bucket_t: usize,
) -> Result<Vec<u8>, ObservationError> {
    let prefix = row.event_prefix_len as usize;
    if prefix > arena.len() {
        return Err(ObservationError(format!(
            "row prefix {prefix} exceeds arena {}",
            arena.len()
        )));
    }
    if row.phase_id > 5 {
        return Err(ObservationError(format!("phase id {} out of range", row.phase_id)));
    }
    if row.furiten_id > 3 {
        return Err(ObservationError(format!("furiten id {} out of range", row.furiten_id)));
    }
    let mut out = vec![0u8; row_bytes_26(bucket_t)];
    let mut at = 0usize;
    let put_i64 = |out: &mut Vec<u8>, at: &mut usize, v: i64| {
        out[*at..*at + 8].copy_from_slice(&v.to_le_bytes());
        *at += 8;
    };
    let put_i32 = |out: &mut Vec<u8>, at: &mut usize, v: i32| {
        out[*at..*at + 4].copy_from_slice(&v.to_le_bytes());
        *at += 4;
    };
    let put_bool = |out: &mut Vec<u8>, at: &mut usize, v: bool| {
        out[*at] = u8::from(v);
        *at += 1;
    };
    put_i64(&mut out, &mut at, i64::from(row.actor));
    put_bool(&mut out, &mut at, row.can_riichi);
    put_bool(&mut out, &mut at, row.can_tsumo);
    put_i64(&mut out, &mut at, i64::from(row.furiten_id));
    put_i64(&mut out, &mut at, i64::from(row.actor));
    let mut concealed_counts = [0i32; 34];
    for t in &row.concealed {
        let kind = (*t as usize) / 4;
        *concealed_counts.get_mut(kind).ok_or_else(|| {
            ObservationError("concealed kind out of range".to_string())
        })? += 1;
    }
    for c in concealed_counts {
        put_i32(&mut out, &mut at, c);
    }
    put_i64(&mut out, &mut at, i64::from(row.dealer));
    // Event-sourced like the document path (see `revealed_dora`): the row's
    // wall snapshot may hold indicators the prefix never revealed, and the
    // learner must see exactly what the live slab showed.
    for d in super::revealed_dora(arena, prefix) {
        put_i32(&mut out, &mut at, d);
    }
    put_i32(&mut out, &mut at, row.hand_number as i32);
    let mut kinds: Vec<i64> = Vec::new();
    let mut riichi = [0i64; 4];
    let mut kan_count = 0i32;
    let mut visible_counts = [0i32; 34];
    // Hand-scoped replay, same window law as the rebuild path above.
    let start = arena.hand_window_start(prefix);
    for index in start..prefix {
        let event = arena.get(index).ok_or_else(|| {
            ObservationError(format!("prefix index {index} outside arena"))
        })?;
        if visible_to_actor(event, row.actor) {
            kinds.push(history_kind_id(&event.kind));
        }
        match event.kind.as_str() {
            "discard" => {
                if let Some(tile) = event.tile {
                    let kind = (tile as usize) / 4;
                    if let Some(slot) = visible_counts.get_mut(kind) {
                        *slot += 1;
                    }
                }
            }
            "riichi_declared" => {
                if let Some(actor) = event.actor {
                    riichi[actor as usize] = 1;
                }
            }
            "riichi_accepted" => {
                if let Some(actor) = event.actor {
                    riichi[actor as usize] = 2;
                }
            }
            "chi" | "pon" | "daiminkan" | "ankan" | "kakan" => {
                if matches!(event.kind.as_str(), "daiminkan" | "ankan" | "kakan") {
                    kan_count += 1;
                }
                for t in &event.consumed_tiles {
                    let kind = (*t as usize) / 4;
                    if let Some(slot) = visible_counts.get_mut(kind) {
                        *slot += 1;
                    }
                }
                if let Some(called) = event.tile {
                    let kind = (called as usize) / 4;
                    if let Some(slot) = visible_counts.get_mut(kind) {
                        *slot += 1;
                    }
                }
            }
            _ => {}
        }
    }
    if kinds.len() > bucket_t {
        return Err(ObservationError(format!(
            "history {} exceeds bucket {bucket_t}",
            kinds.len()
        )));
    }
    for _ in kinds.len()..bucket_t {
        kinds.push(0);
    }
    for k in &kinds {
        put_i64(&mut out, &mut at, *k);
    }
    for pos in 0..bucket_t {
        put_bool(&mut out, &mut at, pos < row_history_len(row, arena));
    }
    put_i32(&mut out, &mut at, row.honba as i32);
    for i in 0..4 {
        put_bool(&mut out, &mut at, row.ippatsu_bits & (1 << i) != 0);
    }
    put_i32(&mut out, &mut at, kan_count);
    let mut legal = vec![false; BASELINE_ACTIONS];
    for id in &row.legal_ids {
        *legal.get_mut(*id as usize).ok_or_else(|| {
            ObservationError(format!("legal id {id} outside census"))
        })? = true;
    }
    for flag in &legal {
        put_bool(&mut out, &mut at, *flag);
    }
    put_i32(&mut out, &mut at, row.live_remaining as i32);
    put_i32(&mut out, &mut at, row.drawn.map_or(-1, |t| i32::from(t)));
    put_i64(&mut out, &mut at, i64::from(row.phase_id));
    for r in riichi {
        put_i64(&mut out, &mut at, r);
    }
    put_i32(&mut out, &mut at, row.kyotaku as i32);
    put_i32(&mut out, &mut at, row.round_index as i32);
    put_i64(&mut out, &mut at, wind_id(row.round_wind)?);
    for s in row.scores {
        put_i32(&mut out, &mut at, s);
    }
    for o in 0..4u8 {
        put_i64(&mut out, &mut at, wind_id(27 + ((i64::from(row.dealer) + i64::from(o)) % 4))?);
    }
    put_i64(&mut out, &mut at, i64::from(row.turn_actor));
    for slot in visible_counts.iter_mut() {
        if *slot > 4 {
            *slot = 4;
        }
    }
    for c in visible_counts {
        put_i32(&mut out, &mut at, c);
    }
    if at != row_bytes_26(bucket_t) {
        return Err(ObservationError(format!(
            "encode length {at} != S(T)={}",
            row_bytes_26(bucket_t)
        )));
    }
    Ok(out)
}

/// Actor-visible history length for one row: the hand-scoped prefix slice
/// filtered by the actor visibility boundary.
fn row_history_len(row: &GameRow, arena: &EventArena) -> usize {
    let prefix = (row.event_prefix_len as usize).min(arena.len());
    let start = arena.hand_window_start(prefix);
    (start..prefix)
        .filter(|i| {
            arena
                .get(*i)
                .is_some_and(|e| visible_to_actor(e, row.actor))
        })
        .count()
}
