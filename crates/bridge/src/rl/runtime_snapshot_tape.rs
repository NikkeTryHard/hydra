//! rl/runtime_snapshot_tape: full-fidelity native tape persistence.
//!
//! DAG: `super::{runtime_snapshot, core, events, observation, trajectory}`
//! + `serde_json`. FORBIDS: PyO3, wall-clock, RNG, recomputing sealed
//! advantage (stored `A`/`G4` are preserved verbatim and verified, never
//! re-derived), opaque core RNG serialization (replay stays canonical
//! action prefixes + stream counters in the pool blob; the tape carries
//! only typed rows + arenas + frozen terminal facts).
//!
//! Two CAS blobs per save (both via `runtime_snapshot::write_runtime_blob`):
//! - event blob (`rl.tape_events/v1`): per-game typed arenas as envelope
//!   JSON (the exact `TypedEvent::to_json_value` shape, re-admitted through
//!   `EventArena::admit` on restore so stream grammar is re-verified).
//! - tape blob (`rl.tape/v1`): per-game rows (`GameRow::to_json`, including
//!   frozen `raw_advantage`/`return_g4`, LP/value/selected/cursors/hashes)
//!   plus the event blob's `{digest, locator}` ref.
//! Observations re-materialize through the existing pure
//! `observation::materialize_canonical` / `encode_row_facts` helpers bound
//! at registration (the seal path asserts identical row ids across its two
//! passes, so byte-identity is enforced downstream, not duplicated here).
//!
//! Lock discipline: per-game short `with_tape` clones (many small holds,
use crate::rl::trajectory::{TapeError, TapeGame, TapeHandle, with_tape};

use serde_json::{Map, Value};

use crate::rl::core::GameRow;
use crate::rl::events::{DeltaPathSeg, EventArena, StateDelta, TypedEvent, Visibility};
use crate::rl::runtime_snapshot::{
    RUNTIME_EVENT_SCHEMA, RUNTIME_TAPE_SCHEMA, canon_bytes, parse_cas_ref, parse_json,
    publish_runtime_canonical, read_verified_blob, reject_escaped_refs, rt_err,
};

/// Decode one envelope document (the exact `to_json_value` shape) back
/// into a `TypedEvent`. Fails closed on any shape drift.
pub fn typed_event_from_envelope(envelope: &Value) -> Result<TypedEvent, TapeError> {
    let fail = |what: &str| rt_err(format!("tape event envelope rejected: {what}"));
    let map = envelope.as_object().ok_or_else(|| fail("must be an object"))?;
    let str_of = |key: &str| {
        map.get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| fail(key))
    };
    let opt_u8 = |obj: &Map<String, Value>, key: &str| -> Result<Option<u8>, TapeError> {
        match obj.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v.as_u64().and_then(|n| u8::try_from(n).ok()).map_or_else(
                || Err(fail(key)),
                |b| Ok(Some(b)),
            ),
        }
    };
    let opt_u32 = |obj: &Map<String, Value>, key: &str| -> Result<Option<u32>, TapeError> {
        match obj.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v.as_u64().and_then(|n| u32::try_from(n).ok()).map_or_else(
                || Err(fail(key)),
                |n| Ok(Some(n)),
            ),
        }
    };
    let opt_u64 = |obj: &Map<String, Value>, key: &str| -> Result<Option<u64>, TapeError> {
        match obj.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v.as_u64().map_or_else(|| Err(fail(key)), |n| Ok(Some(n))),
        }
    };
    let u8_vec = |obj: &Map<String, Value>, key: &str| -> Result<Vec<u8>, TapeError> {
        obj.get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| fail(key))?
            .iter()
            .map(|t| t.as_u64().and_then(|n| u8::try_from(n).ok()).ok_or_else(|| fail(key)))
            .collect()
    };
    let u32_vec = |obj: &Map<String, Value>, key: &str| -> Result<Vec<u32>, TapeError> {
        obj.get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| fail(key))?
            .iter()
            .map(|t| t.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| fail(key)))
            .collect()
    };
    let payload = map
        .get("payload")
        .and_then(Value::as_object)
        .ok_or_else(|| fail("payload"))?;
    let visibility = match map.get("visibility").and_then(Value::as_str) {
        Some("public") => Visibility::Public,
        Some("actor_private") => Visibility::ActorPrivate,
        Some("server_private") => Visibility::ServerPrivate,
        other => return Err(fail(&format!("visibility {other:?}"))),
    };
    let visible_to = u8_vec(map, "visible_to")?;
    let public_delta = map
        .get("public_delta")
        .and_then(Value::as_array)
        .ok_or_else(|| fail("public_delta"))?
        .iter()
        .map(|d| {
            let obj = d.as_object().ok_or_else(|| fail("delta"))?;
            let path = obj
                .get("path")
                .and_then(Value::as_array)
                .ok_or_else(|| fail("delta.path"))?
                .iter()
                .map(|seg| match seg {
                    Value::String(k) => Ok(DeltaPathSeg::Key(k.clone())),
                    Value::Number(n) => n.as_u64().and_then(|i| u8::try_from(i).ok()).map_or_else(
                        || Err(fail("delta.path index")),
                        |i| Ok(DeltaPathSeg::Index(i)),
                    ),
                    _ => Err(fail("delta.path segment")),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let operation = obj
                .get("operation")
                .and_then(Value::as_str)
                .ok_or_else(|| fail("delta.operation"))?
                .to_string();
            let value = obj.get("value").cloned().ok_or_else(|| fail("delta.value"))?;
            Ok(StateDelta {
                path,
                operation,
                value,
            })
        })
        .collect::<Result<Vec<_>, TapeError>>()?;
    let scores = match payload.get("scores") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let items = v.as_array().ok_or_else(|| fail("scores"))?;
            if items.len() != 4 {
                return Err(fail("scores"));
            }
            let mut out = [0i32; 4];
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = items[i].as_i64().and_then(|n| i32::try_from(n).ok()).ok_or_else(|| fail("scores"))?;
            }
            Some(out)
        }
    };
    Ok(TypedEvent {
        game_id: str_of("game_id")?,
        sequence: map.get("sequence").and_then(Value::as_u64).ok_or_else(|| fail("sequence"))?,
        kind: str_of("kind")?,
        actor: opt_u8(payload, "actor")?,
        visibility,
        visible_to,
        tile: opt_u8(payload, "tile")?,
        action_id: opt_u32(payload, "action_id")?,
        source_seat: opt_u8(payload, "source_seat")?,
        consumed_tiles: u8_vec(payload, "consumed_tiles")?,
        offered_action_ids: u32_vec(payload, "offered_action_ids")?,
        accepted_action_ids: u32_vec(payload, "accepted_action_ids")?,
        round_index: opt_u64(payload, "round_index")?,
        scores,
        reason: match payload.get("reason") {
            None | Some(Value::Null) => None,
            Some(v) => Some(v.as_str().ok_or_else(|| fail("reason"))?.to_string()),
        },
        public_delta,
        rules_hash: str_of("rules_hash")?,
        schema_hash: str_of("schema_hash")?,
    })
}

/// Verify one restored row against its game's frozen utilities + arena
/// (stored facts preserved verbatim; the frozen recipe is CHECKED, never
/// re-applied).
fn verify_row(row: &GameRow, utilities: [f64; 4], arena_len: usize) -> Result<(), TapeError> {
    if row.actor > 3 {
        return Err(rt_err(format!("row {} actor outside four seats", row.decision_id)));
    }
    if !row.selected_logp.is_finite() || row.selected_logp > 0.0 {
        return Err(rt_err(format!("row {} logp not a finite log-prob", row.decision_id)));
    }
    if row.value4.iter().any(|v| !v.is_finite()) {
        return Err(rt_err(format!("row {} value4 nonfinite", row.decision_id)));
    }
    if !row.legal_ids.contains(&row.selected) {
        return Err(rt_err(format!("row {} selected outside legal ids", row.decision_id)));
    }
    if row.cursor_after < row.cursor_before {
        return Err(rt_err(format!("row {} cursor regresses", row.decision_id)));
    }
    if row.event_prefix_len as usize > arena_len {
        return Err(rt_err(format!(
            "row {} prefix {} escapes its arena of {arena_len}",
            row.decision_id, row.event_prefix_len
        )));
    }
    let g4 = row.return_g4.ok_or_else(|| {
        rt_err(format!("row {} sealed before terminal bind", row.decision_id))
    })?;
    if g4.iter().any(|v| !v.is_finite()) {
        return Err(rt_err(format!("row {} G4 nonfinite", row.decision_id)));
    }
    if g4 != utilities {
        return Err(rt_err(format!(
            "row {} G4 diverges from its game utilities",
            row.decision_id
        )));
    }
    let stored = row.raw_advantage_f32.ok_or_else(|| {
        rt_err(format!("row {} lacks frozen advantage", row.decision_id))
    })?;
    if !stored.is_finite() {
        return Err(rt_err(format!("row {} advantage nonfinite", row.decision_id)));
    }
    let expected = (g4[row.actor as usize] as f32) - row.value4[row.actor as usize];
    if stored != expected {
        return Err(rt_err(format!(
            "row {} advantage diverges from the frozen recipe",
            row.decision_id
        )));
    }
    Ok(())
}

/// Clone one game's tape data under a short registry hold.
fn clone_game(handle: TapeHandle, index: usize) -> Result<TapeGame, TapeError> {
    with_tape(handle, |tape| {
        tape.games
            .get(index)
            .cloned()
            .ok_or_else(|| TapeError(format!("tape game index {index} outside cohort")))
    })
}

fn tape_game_count(handle: TapeHandle) -> Result<usize, TapeError> {
    with_tape(handle, |tape| Ok(tape.games.len()))
}

/// Canonical fragments retain array order and the two-GiB blob limit;
/// checked appends reject overflow before an oversized buffer is allocated.
fn append_fragment(out: &mut Vec<u8>, fragment: &[u8]) -> Result<(), TapeError> {
    let size = out.len().checked_add(fragment.len())
        .ok_or_else(|| rt_err("blob document has a bad length"))?;
    if size > (1usize << 31) {
        return Err(rt_err("blob document has a bad length"));
    }
    out.extend_from_slice(fragment);
    Ok(())
}

/// Per-game object validation preserves escaping-reference rejection;
/// retaining only its canonical bytes avoids a generation-wide JSON tree.
fn append_game(out: &mut Vec<u8>, game: Map<String, Value>) -> Result<(), TapeError> {
    let value = Value::Object(game);
    reject_escaped_refs(&value)?;
    append_fragment(out, &canon_bytes(&value, "rl:tape-game")?)
}

/// Persist the FULL owned tape behind `handle` as two CAS blobs (event
/// arenas + rows with frozen terminal facts). Post-freeze only: every
/// row-bearing game must carry utilities and every row its frozen `A`/`G4`
/// (unbound tapes fail closed here rather than being sealed with invented
/// facts). Returns the tape receipt bytes (`digest`, `locator`,
/// `row_count`, `game_count`).
pub fn save_tape(
    root: &std::path::Path,
    handle: TapeHandle,
) -> Result<Vec<u8>, TapeError> {
    let game_count = tape_game_count(handle)?;
    if game_count == 0 {
        return Err(rt_err("cannot persist an empty tape"));
    }
    let mut event_doc = format!("{{\"game_count\":{game_count},\"games\":[").into_bytes();
    let mut tape_games = Vec::new();
    let mut row_count: u64 = 0;
    for index in 0..game_count {
        let game = clone_game(handle, index)?;
        let utilities = game.utilities.ok_or_else(|| {
            rt_err(format!("tape game {} lacks terminal utilities", game.game_id))
        })?;
        if utilities.iter().any(|v| !v.is_finite()) {
            return Err(rt_err(format!(
                "tape game {} utilities nonfinite",
                game.game_id
            )));
        }
        if game.rows.is_empty() {
            return Err(rt_err(format!("tape game {} carries no rows", game.game_id)));
        }
        let mut envelopes = Vec::with_capacity(game.arena.len());
        for i in 0..game.arena.len() {
            let event = game
                .arena
                .get(i)
                .ok_or_else(|| rt_err(format!("tape game {} arena gap at {i}", game.game_id)))?;
            if event.game_id != game.game_id {
                return Err(rt_err(format!(
                    "tape game {} arena mixes games",
                    game.game_id
                )));
            }
            envelopes.push(event.to_json_value());
        }
        if envelopes.is_empty() {
            return Err(rt_err(format!("tape game {} arena is empty", game.game_id)));
        }
        let mut rows = Vec::with_capacity(game.rows.len());
        for row in &game.rows {
            if row.wall_id != game.wall_id || row.game_id != game.game_id {
                return Err(rt_err(format!(
                    "row {} identity escapes game {}",
                    row.decision_id, game.game_id
                )));
            }
            verify_row(row, utilities, game.arena.len())?;
            rows.push(row.to_json());
            row_count = row_count
                .checked_add(1)
                .ok_or_else(|| rt_err("tape row count overflows"))?;
        }
        let mut event_game = Map::with_capacity(3);
        event_game.insert("wall_id".into(), Value::String(game.wall_id.clone()));
        event_game.insert("game_id".into(), Value::String(game.game_id.clone()));
        event_game.insert("events".into(), Value::Array(envelopes));
        if index != 0 {
            append_fragment(&mut event_doc, b",")?;
            append_fragment(&mut tape_games, b",")?;
        }
        append_game(&mut event_doc, event_game)?;
        let mut tape_game = Map::with_capacity(4);
        tape_game.insert("wall_id".into(), Value::String(game.wall_id));
        tape_game.insert("game_id".into(), Value::String(game.game_id));
        tape_game.insert("utilities".into(), serde_json::json!(utilities));
        tape_game.insert("rows".into(), Value::Array(rows));
        append_game(&mut tape_games, tape_game)?;
    }
    // Fixed ASCII root keys are emitted in RFC8785 order; per-game fragments
    // retain their authoritative canonical encoding instead of being rebuilt.
    append_fragment(&mut event_doc, format!("],\"schema\":\"{RUNTIME_EVENT_SCHEMA}\",\"schema_version\":\"1.0.0\"}}").as_bytes())?;
    let event_receipt = publish_runtime_canonical(root, &event_doc)?;
    drop(event_doc);
    let event_value: Value = parse_json(&event_receipt, "event receipt")?;
    let event_ref = serde_json::json!({
        "digest": event_value.get("digest").cloned().unwrap_or(Value::Null),
        "locator": event_value.get("locator").cloned().unwrap_or(Value::Null),
    });
    let mut tape_doc = b"{\"event_ref\":".to_vec();
    reject_escaped_refs(&event_ref)?;
    append_fragment(&mut tape_doc, &canon_bytes(&event_ref, "rl:event-ref")?)?;
    append_fragment(&mut tape_doc, format!(",\"game_count\":{game_count},\"games\":[").as_bytes())?;
    append_fragment(&mut tape_doc, &tape_games)?;
    drop(tape_games);
    append_fragment(&mut tape_doc, format!("],\"row_count\":{row_count},\"schema\":\"{RUNTIME_TAPE_SCHEMA}\",\"schema_version\":\"1.0.0\"}}").as_bytes())?;
    let tape_receipt = publish_runtime_canonical(root, &tape_doc)?;
    let mut receipt: Value = parse_json(&tape_receipt, "tape receipt")?;
    if let Some(map) = receipt.as_object_mut() {
        map.insert("row_count".to_string(), Value::from(row_count));
        map.insert("game_count".to_string(), Value::from(game_count as u64));
    }
    canon_bytes(&receipt, "rl:tape-ref")
}

/// Tape ref shape: single-owned by `super::runtime_snapshot::parse_tape_ref`
/// (the exact fixed shape `save_tape` returns). Re-exported here for
/// callers resolving through the tape module.
pub use super::runtime_snapshot::parse_tape_ref;

/// Restore a verified tape blob into a fresh registry handle (stored facts
/// preserved verbatim; materialize/encode are the existing pure native
/// observation helpers). The handle is unstamped here; `restore_runtime`
/// re-applies the tip-committed artifact digest after the full-chain
/// verification. Returns the new handle.
pub fn restore_tape(
    root: &std::path::Path,
    ref_doc: &[u8],
) -> Result<TapeHandle, TapeError> {
    let value = parse_json(ref_doc, "tape ref")?;
    let tape_ref = parse_tape_ref(&value, "tape ref")?;
    let (digest, locator, want_rows, want_games) = (
        tape_ref.digest.as_str(),
        tape_ref.locator.as_str(),
        tape_ref.row_count,
        tape_ref.game_count,
    );
    let bytes = read_verified_blob(root, &digest, &locator)?;
    let doc: Value = parse_json(&bytes, "tape blob")?;
    let map = doc
        .as_object()
        .ok_or_else(|| rt_err("tape blob must be an object"))?;
    match map.get("schema").and_then(Value::as_str) {
        Some(RUNTIME_TAPE_SCHEMA) => {}
        other => {
            return Err(rt_err(format!("tape blob schema must be {RUNTIME_TAPE_SCHEMA:?}, got {other:?}")));
        }
    }
    let event_ref = parse_cas_ref(
        map.get("event_ref").unwrap_or(&Value::Null),
        "tape event_ref",
    )?;
    let event_bytes = read_verified_blob(root, &event_ref.digest, &event_ref.locator)?;
    let event_doc: Value = parse_json(&event_bytes, "event blob")?;
    let event_map = event_doc
        .as_object()
        .ok_or_else(|| rt_err("event blob must be an object"))?;
    match event_map.get("schema").and_then(Value::as_str) {
        Some(RUNTIME_EVENT_SCHEMA) => {}
        other => {
            return Err(rt_err(format!(
                "event blob schema must be {RUNTIME_EVENT_SCHEMA:?}, got {other:?}"
            )));
        }
    }
    let row_count = map.get("row_count").and_then(Value::as_u64).ok_or_else(|| {
        rt_err("tape blob lacks row_count")
    })?;
    let game_count = map.get("game_count").and_then(Value::as_u64).ok_or_else(|| {
        rt_err("tape blob lacks game_count")
    })?;
    if row_count != want_rows || game_count != want_games {
        return Err(rt_err("tape blob counts diverge from its ref"));
    }
    let tape_games = map
        .get("games")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err("tape blob lacks games"))?;
    let event_games = event_map
        .get("games")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err("event blob lacks games"))?;
    if tape_games.len() as u64 != game_count || event_games.len() as u64 != game_count {
        return Err(rt_err("tape/event game arrays diverge from game_count"));
    }
    let mut games = Vec::with_capacity(tape_games.len());
    let mut restored_rows: u64 = 0;
    for (index, (tape_game, event_game)) in tape_games.iter().zip(event_games.iter()).enumerate() {
        let tmap = tape_game
            .as_object()
            .ok_or_else(|| rt_err(format!("tape games[{index}] must be an object")))?;
        let emap = event_game
            .as_object()
            .ok_or_else(|| rt_err(format!("event games[{index}] must be an object")))?;
        let wall_id = tmap
            .get("wall_id")
            .and_then(Value::as_str)
            .ok_or_else(|| rt_err(format!("tape games[{index}] lacks wall_id")))?;
        let game_id = tmap
            .get("game_id")
            .and_then(Value::as_str)
            .ok_or_else(|| rt_err(format!("tape games[{index}] lacks game_id")))?;
        if emap.get("wall_id").and_then(Value::as_str) != Some(wall_id)
            || emap.get("game_id").and_then(Value::as_str) != Some(game_id)
        {
            return Err(rt_err(format!("tape/event games[{index}] identity diverges")));
        }
        let utilities_value = tmap.get("utilities").and_then(Value::as_array).ok_or_else(|| {
            rt_err(format!("tape games[{index}] lacks utilities"))
        })?;
        if utilities_value.len() != 4 {
            return Err(rt_err(format!("tape games[{index}] utilities must hold four seats")));
        }
        let mut utilities = [0.0f64; 4];
        for (i, seat) in utilities_value.iter().enumerate() {
            utilities[i] = seat
                .as_f64()
                .ok_or_else(|| rt_err(format!("tape games[{index}] utilities must be numbers")))?;
        }
        if utilities.iter().any(|v| !v.is_finite()) {
            return Err(rt_err(format!("tape games[{index}] utilities nonfinite")));
        }
        let mut arena = EventArena::new();
        for envelope in emap
            .get("events")
            .and_then(Value::as_array)
            .ok_or_else(|| rt_err(format!("event games[{index}] lacks events")))?
        {
            let event = typed_event_from_envelope(envelope)?;
            if event.game_id != game_id {
                return Err(rt_err(format!("event games[{index}] mixes games")));
            }
            arena
                .admit(event)
                .map_err(|e| rt_err(format!("event games[{index}] grammar rejected: {e}")))?;
        }
        if arena.is_empty() {
            return Err(rt_err(format!("event games[{index}] arena is empty")));
        }
        let mut rows = Vec::new();
        for row_doc in tmap
            .get("rows")
            .and_then(Value::as_array)
            .ok_or_else(|| rt_err(format!("tape games[{index}] lacks rows")))?
        {
            let row = GameRow::from_json(row_doc)
                .map_err(|e| rt_err(format!("tape games[{index}] row rejected: {e}")))?;
            if row.wall_id != wall_id || row.game_id != game_id {
                return Err(rt_err(format!(
                    "tape games[{index}] row identity escapes its game"
                )));
            }
            verify_row(&row, utilities, arena.len())?;
            rows.push(row);
            restored_rows = restored_rows
                .checked_add(1)
                .ok_or_else(|| rt_err("restored row count overflows"))?;
        }
        if rows.is_empty() {
            return Err(rt_err(format!("tape games[{index}] carries no rows")));
        }
        games.push(TapeGame {
            wall_id: wall_id.to_string(),
            game_id: game_id.to_string(),
            arena,
            rows,
            utilities: Some(utilities),
        });
    }
    if restored_rows != row_count {
        return Err(rt_err("restored rows diverge from the tape row_count"));
    }
    let materialize = |row: &GameRow, arena: &EventArena| {
        crate::rl::observation::materialize_canonical(row, arena)
            .map_err(|e| crate::rl::trajectory::TapeError(e.0))
    };
    let encode = |row: &GameRow, arena: &EventArena, t: u32| {
        crate::rl::observation::encode_row_facts(row, arena, t as usize)
            .map_err(|e| crate::rl::trajectory::TapeError(e.0))
    };
    crate::rl::trajectory::register_tape(games, Box::new(materialize), Box::new(encode))
}
