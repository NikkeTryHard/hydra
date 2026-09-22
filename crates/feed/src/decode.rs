//! Packet decode — simd prefilter + serde typed-de (`feed::decode`).
//!
//! Rust owner of `src/hydra2/data/decode.py::decode_game_object` (lines
//! 49-135): one-game-per-object JSONL decode. Batch entry point mirrors the
//! `stream_decode.py` spawn-16 ordered-merge worker
//! (`_decode_frames_worker`, lines 88-143) minus the spawn: order-preserving,
//! index-aligned output ready for a `rayon::par_iter` inside `detach`.
//!
//! Pipeline per line (M9 exact): `simd_json::serde::from_slice` prefilter
//! scan → `serde_json::from_slice` typed decode as the authoritative value.
//! A simd-reject falls back to serde EXACT once (compat probe); failure
//! still rejects — never coerced. `serde_json` here is parse/reject ONLY:
//! this module NEVER emits canonical bytes (canon phase owns bytes; trap-5).
//! `raw_bytes_sha256` hashes the verbatim input bytes via
//! `crate::digest::sha256_hex`, exactly like Python
//! `hashlib.sha256(decoded_bytes)` (`decode.py:126`).
//!
//! Oracle-exact rules carried over:
//! - UTF-8 whole-payload check (`:63-66`) → [`DecodeErr::Utf8`].
//! - Trailing `\n` required (`:67-70`) → [`DecodeErr::TrailingNewline`].
//! - ANY blank line → reject, never skip (`:75-80`) → [`DecodeErr::BlankLine`].
//! - Per line: must be a JSON object with a `type` key (`:81-93`).
//! - First type ∈ 4 start aliases, last ∈ 4 end aliases (`:95-100`).
//! - Exactly-one counts cover the CANONICAL 3 spellings only (`:101-109`):
//!   a bare `start`/`end` boundary passes the edge gate but fails the count
//!   (mirrors `ingest::kinds::boundary_class`).
//! - `game_id`: first `game_id`|`gameId` non-empty string, else
//!   `"game-" + sha256(object_id)[:12]` (`:36-46`).
//! - Wall: first `wall`|`wall_tiles`|`tiles` list of 136 ints (`:114-125`),
//!   stored as `[i64; 136]` so out-of-range ints stay representable and fail
//!   in validation (permutation), exactly like the oracle.
//!
//! Traps applied: 13 (batch API for `par_iter`-inside-`detach`, ordered merge
//! replaces spawn ordered-merge; pure-Rust closures, never `with_gil`), 15
//! (no shared dict here — batch output is index-aligned; checks-dict
//! locality lives in `crate::validate::validate_batch`).
//!
//! Allocation: one `Vec<u8>` probe copy per line for the simd scan plus the
//! owned `serde_json::Value` events. Framing (`feed::ingest`) stays the
//! zero-alloc hot path; decode is the cold-side batch path.

use serde_json::Value;

use crate::digest::sha256_hex;
use crate::framer::Frame;

/// Start aliases (`decode.py:97`).
pub const START_TYPES: [&str; 4] = ["start_game", "startGame", "game_start", "start"];
/// End aliases (`decode.py:99`).
pub const END_TYPES: [&str; 4] = ["end_game", "endGame", "game_end", "end"];
/// Canonical start spellings counted by the exactly-one rule (`decode.py:101-103`).
const START_CANON: [&str; 3] = ["start_game", "startGame", "game_start"];
/// Canonical end spellings counted by the exactly-one rule (`decode.py:104`).
const END_CANON: [&str; 3] = ["end_game", "endGame", "game_end"];
/// Wall-carrying keys, first hit wins (`decode.py:116-120`).
const WALL_KEYS: [&str; 3] = ["wall", "wall_tiles", "tiles"];
/// Wall length (`decode.py:122`).
pub const WALL_LEN: usize = 136;

/// Decode failure. [`DecodeErr::error_class`] renders the closed taxonomy
/// string (`utf8|trailing_newline|blank_line|json|shape|start_end`, wave3 §2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeErr {
    /// Payload is not UTF-8 (`decode.py:63-66`).
    Utf8,
    /// Payload does not end with `\n` (`decode.py:67-70`).
    TrailingNewline,
    /// Blank line at event index (`decode.py:75-80`).
    BlankLine(u32),
    /// Line is not valid JSON (`decode.py:83-86`).
    Json(u32),
    /// Line is not a JSON object or carries no `type` key (`decode.py:87-92`).
    Shape(u32),
    /// First/last boundary or exactly-one count violated (`decode.py:95-109`).
    StartEnd,
}

impl DecodeErr {
    /// Closed taxonomy string, byte-exact vs fixtures.
    pub const fn error_class(&self) -> &'static str {
        match self {
            DecodeErr::Utf8 => "utf8",
            DecodeErr::TrailingNewline => "trailing_newline",
            DecodeErr::BlankLine(_) => "blank_line",
            DecodeErr::Json(_) => "json",
            DecodeErr::Shape(_) => "shape",
            DecodeErr::StartEnd => "start_end",
        }
    }

    /// Offending event index, when the failure names a line.
    pub const fn event_index(&self) -> Option<u32> {
        match self {
            DecodeErr::BlankLine(i) | DecodeErr::Json(i) | DecodeErr::Shape(i) => Some(*i),
            DecodeErr::Utf8 | DecodeErr::TrailingNewline | DecodeErr::StartEnd => None,
        }
    }
}

impl core::fmt::Display for DecodeErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodeErr::Utf8 => write!(f, "decoded bytes not utf-8"),
            DecodeErr::TrailingNewline => write!(f, "payload must end with newline"),
            DecodeErr::BlankLine(i) => write!(f, "blank line at index {i}"),
            DecodeErr::Json(i) => write!(f, "line {i} invalid JSON"),
            DecodeErr::Shape(i) => write!(f, "line {i} must be JSON object with 'type'"),
            DecodeErr::StartEnd => write!(f, "start/end boundary or count violated"),
        }
    }
}

impl std::error::Error for DecodeErr {}

/// Decoded game record (mirrors `GameRecord`, `decode.py:25-33`).
///
/// `source` (`{"type": first_type}`, `:134`) is derivable from
/// `events[0]` and is not stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameRecord {
    /// First explicit `game_id`|`gameId`, else `game-<12hex>` fallback.
    pub game_id: String,
    /// Caller-supplied object id (stem).
    pub object_id: String,
    /// Caller-supplied packaged object id.
    pub packaged_object_id: String,
    /// Typed-decoded events in log order (authoritative `serde_json` values).
    pub events: Vec<Value>,
    /// `sha256:<hex>` over the VERBATIM input bytes (`decode.py:126`).
    pub raw_bytes_sha256: String,
    /// First 136-int list under `wall`|`wall_tiles`|`tiles`, if any.
    pub wall_tiles: Option<[i64; WALL_LEN]>,
}
/// JSON integer as `i64` (floats/strings are NOT ints; bools ARE —
/// Python `isinstance(True, int)` is `True`, so `True` counts as tile `1`
/// in the wall-less overflow scan and as `action_id` `1` in the bounds
/// check; corpus payloads never use bools there, so the paths agree).
pub(crate) fn json_int(v: &Value) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(i64::from(*b)),
        _ => v
            .as_i64()
            .or_else(|| v.as_u64().and_then(|u| i64::try_from(u).ok())),
    }
}

/// Event `type` when present AND a string.
///
/// Non-string types match nothing (Python `str(...)` of a non-string never
/// hits the alias sets, and the exactly-one counts use the same coercion).
pub(crate) fn ev_type(ev: &Value) -> Option<&str> {
    ev.get("type")?.as_str()
}

/// True iff the typed value is a JSON object carrying a `type` key
/// (`decode.py:87-92`: object + presence, any value).
fn is_object_with_type(v: &Value) -> bool {
    v.as_object().is_some_and(|o| o.contains_key("type"))
}

/// ASCII-blank line (`decode.py:77`: `line.strip() == ""`).
///
/// Byte-level `is_ascii_whitespace` covers space/tab/CR/LF. A line of only
/// non-ASCII whitespace (e.g. U+00A0) is blank to Python but parses-or-fails
/// here as `Json` instead of `BlankLine`: both reject, the class differs on
/// inputs the corpus never carries.
fn is_blank(line: &[u8]) -> bool {
    line.iter().all(|b| b.is_ascii_whitespace())
}

/// Parse one line: M9 simd prefilter → authoritative serde typed decode.
///
/// simd-reject falls back to serde EXACT once (compat probe); failure still
/// rejects — never coerced (wave3 §2 `decode_game_object` step 4).
fn parse_line(line: &[u8], idx: u32) -> Result<Value, DecodeErr> {
    if is_blank(line) {
        return Err(DecodeErr::BlankLine(idx));
    }
    // M9 prefilter scan: simd owns the fast path, serde owns the definition
    // (trap-12 pattern: escape-free fast path is an optimization, never the
    // definition — same discipline as the manifest ASCII-fast rule).
    let mut probe = line.to_vec();
    let pre: Option<Value> = simd_json::serde::from_slice(&mut probe).ok();
    if pre.as_ref().is_some_and(is_object_with_type) {
        // Authoritative typed decode of the prefiltered line.
        return serde_json::from_slice::<Value>(line).map_err(|_| DecodeErr::Json(idx));
    }
    // simd-reject (or non-object): compat probe, serde EXACT once.
    match serde_json::from_slice::<Value>(line) {
        Ok(v) if is_object_with_type(&v) => Ok(v),
        Ok(_) => Err(DecodeErr::Shape(idx)),
        Err(_) => Err(DecodeErr::Json(idx)),
    }
}

/// `game_id` derivation (`decode.py:36-46`): first non-empty
/// `game_id`|`gameId` string, else `"game-" + sha256(object_id)[:12]`.
fn game_id_from_events(events: &[Value], object_id: &str) -> String {
    for ev in events {
        for key in ["game_id", "gameId"] {
            if let Some(gid) = ev.get(key).and_then(Value::as_str)
                && !gid.is_empty() {
                    return gid.to_owned();
                }
        }
    }
    // `sha256_hex` renders `sha256:<64hex>`; strip the prefix and keep the
    // first 12 hex chars (first 6 digest bytes), exactly like
    // `hashlib.sha256(object_id.encode()).hexdigest()[:12]`.
    let hex = sha256_hex(object_id.as_bytes());
    let mut out = String::from("game-");
    out.push_str(&hex["sha256:".len()..][..12]);
    out
}

/// Wall extraction (`decode.py:114-125`): first 136-int list under
/// `wall`|`wall_tiles`|`tiles`. Stored `i64` so oracle-accepted
/// out-of-range ints stay representable and fail in validation.
fn wall_from_events(events: &[Value]) -> Option<[i64; WALL_LEN]> {
    for ev in events {
        for key in WALL_KEYS {
            if let Some(Value::Array(a)) = ev.get(key)
                && a.len() == WALL_LEN && a.iter().all(|x| json_int(x).is_some()) {
                    let mut wall = [0i64; WALL_LEN];
                    for (i, x) in a.iter().enumerate() {
                        // Candidacy already proved `Some`; default is unreachable.
                        wall[i] = json_int(x).unwrap_or(0);
                    }
                    return Some(wall);
                }
        }
    }
    None
}

/// Decode exactly one game per object (`decode.py:49-135`).
///
/// `object_id` feeds ONLY the synthetic `game_id` fallback; the identity
/// hash covers the verbatim bytes.
pub fn decode_game_object(
    object_id: &str,
    packaged_object_id: &str,
    bytes: &[u8],
) -> Result<GameRecord, DecodeErr> {
    std::str::from_utf8(bytes).map_err(|_| DecodeErr::Utf8)?;
    if !bytes.ends_with(b"\n") {
        return Err(DecodeErr::TrailingNewline);
    }
    // `memchr` `\n`-split over the body (final `\n` stripped). A body that
    // ends with `\n` (payload `...\n\n`) leaves a trailing empty segment =
    // blank line, exactly like `splitlines` (`decode.py:72-80`).
    let body = &bytes[..bytes.len() - 1];
    let mut events: Vec<Value> = Vec::new();
    let mut start = 0usize;
    let mut idx = 0u32;
    if body.is_empty() {
        return Err(DecodeErr::BlankLine(0));
    }
    for nl in memchr::memchr_iter(b'\n', body) {
        events.push(parse_line(&body[start..nl], idx)?);
        idx = idx.saturating_add(1);
        start = nl + 1;
    }
    if start >= body.len() {
        return Err(DecodeErr::BlankLine(idx));
    }
    events.push(parse_line(&body[start..], idx)?);

    // Boundary gates (`decode.py:95-109`).
    let first = events.first().and_then(ev_type).unwrap_or("");
    let last = events.last().and_then(ev_type).unwrap_or("");
    if !START_TYPES.contains(&first) || !END_TYPES.contains(&last) {
        return Err(DecodeErr::StartEnd);
    }
    let starts = events
        .iter()
        .filter_map(ev_type)
        .filter(|t| START_CANON.contains(t))
        .count();
    let ends = events
        .iter()
        .filter_map(ev_type)
        .filter(|t| END_CANON.contains(t))
        .count();
    if starts != 1 || ends != 1 {
        return Err(DecodeErr::StartEnd);
    }

    let game_id = game_id_from_events(&events, object_id);
    let wall_tiles = wall_from_events(&events);
    Ok(GameRecord {
        game_id,
        object_id: object_id.to_owned(),
        packaged_object_id: packaged_object_id.to_owned(),
        events,
        raw_bytes_sha256: sha256_hex(bytes),
        wall_tiles,
    })
}

/// Decode a batch of frames with caller-supplied stems, output index-aligned
/// with the input (M8 scan ABI / trap-13 ordered merge).
///
/// Stems are parallel params (NOT synthesized): the Python oracle passes
/// `stem_of(fpath)` as BOTH `object_id` and `packaged_object_id`
/// (`stream_decode.py:123-126`, `stream_read.py:78-81`). `object_id` feeds the
/// synthetic `game_id` fallback (`game-<12hex>` of the stem) when no explicit
/// id exists — synthesizing `frame-{idx}-{offset}` here would fork game
/// identity vs the oracle. Callers pass `stems.len() == frames.len()`.
pub fn decode_frames_batch(
    frames: &[Frame],
    stems: &[(&str, &str)],
) -> Vec<Result<GameRecord, DecodeErr>> {
    debug_assert_eq!(frames.len(), stems.len());
    frames
        .iter()
        .zip(stems.iter())
        .map(|(f, (object_id, packaged_object_id))| {
            decode_game_object(object_id, packaged_object_id, &f.bytes)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frame bytes from JSON values (test-only framing; `serde_json` emits
    /// test bytes here, never identity bytes — identity hashes cover these
    /// exact bytes self-consistently on both sides of each assert).
    fn game_bytes(events: &[Value]) -> Vec<u8> {
        let mut v = Vec::new();
        for e in events {
            v.extend_from_slice(serde_json::to_string(e).unwrap().as_bytes());
            v.push(b'\n');
        }
        v
    }

    fn start(id: &str) -> Value {
        serde_json::json!({"type": "start_game", "game_id": id})
    }

    fn end() -> Value {
        serde_json::json!({"type": "end_game"})
    }

    fn tsumo() -> Value {
        serde_json::json!({"type": "tsumo", "actor": 0})
    }

    /// KAT: batch accept — walled + wall-less games decode with caller order,
    /// game_id passthrough, wall binding, verbatim raw sha.
    #[test]
    fn decode_batch_accept_and_order() {
        let mut wall_ev = serde_json::json!({"type": "start_game", "game_id": "walled"});
        wall_ev["wall"] = Value::Array((0..136).map(|t: i64| Value::from(t)).collect());
        let walled = game_bytes(&[wall_ev, tsumo(), end()]);
        let plain = game_bytes(&[start("plain"), tsumo(), end()]);
        let walled_len = walled.len() as u64;
        let plain_len = plain.len() as u64;
        let frames = [
            Frame { file_idx: 0, offset: 0, end: walled_len, bytes: walled },
            Frame { file_idx: 0, offset: 100, end: 100 + plain_len, bytes: plain },
        ];
        let stems = [("a", "pa"), ("b", "pb")];
        let out = decode_frames_batch(&frames, &stems);
        assert_eq!(out.len(), 2);
        let g0 = out[0].as_ref().unwrap();
        let g1 = out[1].as_ref().unwrap();
        // Caller order preserved.
        assert_eq!(g0.game_id, "walled");
        assert_eq!(g1.game_id, "plain");
        assert_eq!(g0.object_id, "a");
        assert_eq!(g0.packaged_object_id, "pa");
        // Wall bound from the first 136-int list.
        let wall = g0.wall_tiles.unwrap();
        assert_eq!(wall[0], 0);
        assert_eq!(wall[135], 135);
        assert!(g1.wall_tiles.is_none());
        // raw sha covers the verbatim bytes.
        assert_eq!(g0.raw_bytes_sha256, sha256_hex(&frames[0].bytes));
        assert_eq!(g1.raw_bytes_sha256, sha256_hex(&frames[1].bytes));
        assert_eq!(g0.events.len(), 3);
    }

    /// KAT: reject taxonomy strings + offending line index, byte-exact.
    #[test]
    fn decode_taxonomy_strings_and_indices() {
        // utf8.
        let e = decode_game_object("o", "p", b"\xff\xfe\n").unwrap_err();
        assert_eq!(e.error_class(), "utf8");
        assert_eq!(e.event_index(), None);
        // trailing_newline.
        let e = decode_game_object("o", "p", b"{\"type\":\"start_game\"}").unwrap_err();
        assert_eq!(e.error_class(), "trailing_newline");
        assert_eq!(e.event_index(), None);
        // blank_line at index 1.
        let mut b = game_bytes(&[start("g")]);
        b.extend_from_slice(b"   \n");
        b.extend_from_slice(game_bytes(&[end()]).as_slice());
        let e = decode_game_object("o", "p", &b).unwrap_err();
        assert_eq!(e.error_class(), "blank_line");
        assert_eq!(e.event_index(), Some(1));
        // json at index 1.
        let mut b = game_bytes(&[start("g")]);
        b.extend_from_slice(b"{oops\n");
        b.extend_from_slice(game_bytes(&[end()]).as_slice());
        let e = decode_game_object("o", "p", &b).unwrap_err();
        assert_eq!(e.error_class(), "json");
        assert_eq!(e.event_index(), Some(1));
        // shape: non-object at index 1.
        let b = game_bytes(&[start("g"), Value::from(5), end()]);
        let e = decode_game_object("o", "p", &b).unwrap_err();
        assert_eq!(e.error_class(), "shape");
        assert_eq!(e.event_index(), Some(1));
        // shape: missing type at index 1.
        let b = game_bytes(&[start("g"), serde_json::json!({"actor": 0}), end()]);
        let e = decode_game_object("o", "p", &b).unwrap_err();
        assert_eq!(e.error_class(), "shape");
        assert_eq!(e.event_index(), Some(1));
        // start_end: wrong first.
        let b = game_bytes(&[tsumo(), end()]);
        let e = decode_game_object("o", "p", &b).unwrap_err();
        assert_eq!(e.error_class(), "start_end");
        // start_end: wrong last.
        let b = game_bytes(&[start("g"), tsumo()]);
        let e = decode_game_object("o", "p", &b).unwrap_err();
        assert_eq!(e.error_class(), "start_end");
        // start_end: doubled canonical start (counts == 2/1).
        let b = game_bytes(&[start("g"), start("g"), end()]);
        let e = decode_game_object("o", "p", &b).unwrap_err();
        assert_eq!(e.error_class(), "start_end");
    }

    /// KAT: bare-`start` passes the edge gate but fails the canonical count
    /// (`decode.py:97` vs `:101-103`) — oracle-exact alias accounting.
    #[test]
    fn decode_bare_alias_fails_canonical_count() {
        let b = game_bytes(&[
            serde_json::json!({"type": "start"}),
            tsumo(),
            serde_json::json!({"type": "end"}),
        ]);
        let e = decode_game_object("o", "p", &b).unwrap_err();
        assert_eq!(e.error_class(), "start_end");
    }

    /// KAT: synthetic game_id fallback = `game-` + first 12 hex of
    /// `sha256(object_id)` (`decode.py:46`).
    #[test]
    fn decode_game_id_fallback() {
        let b = game_bytes(&[
            serde_json::json!({"type": "start_game"}),
            tsumo(),
            end(),
        ]);
        let g = decode_game_object("stem-7", "p", &b).unwrap();
        let hex = sha256_hex(b"stem-7");
        assert_eq!(g.game_id, format!("game-{}", &hex["sha256:".len()..][..12]));
        // Explicit gameId alias also wins.
        let b = game_bytes(&[
            serde_json::json!({"type": "start_game", "gameId": "gid-9"}),
            end(),
        ]);
        let g = decode_game_object("stem-7", "p", &b).unwrap();
        assert_eq!(g.game_id, "gid-9");
    }

    /// KAT: M9 compat probe — `\u`-escaped `type` key survives the prefilter
    /// (escaped boundaries must not be dropped).
    #[test]
    fn decode_escaped_type_key_survives_prefilter() {
        let raw = b"{\"\\u0074ype\":\"start_game\",\"game_id\":\"esc\"}\n{\"type\":\"end_game\"}\n";
        let g = decode_game_object("o", "p", raw).unwrap();
        assert_eq!(g.game_id, "esc");
        assert_eq!(g.events.len(), 2);
    }

    /// KAT: batch order under mixed ok/err (trap-13 ordered merge) —
    /// failures occupy their slot, siblings never shift.
    #[test]
    fn decode_batch_mixed_keeps_positions() {
        let good = game_bytes(&[start("g0"), end()]);
        let bad = b"{\"type\":\"start_game\"}\n".to_vec(); // missing end
        let good2 = game_bytes(&[start("g2"), end()]);
        // Stems mirror the oracle: `stem_of(fpath)` for both ids (per-file stem,
        // NOT per-frame synthetic — game_id fallback hashes the stem).
        let frames = [
            Frame { file_idx: 0, offset: 0, end: good.len() as u64, bytes: good },
            Frame { file_idx: 0, offset: 10, end: 20, bytes: bad },
            Frame { file_idx: 0, offset: 30, end: 40, bytes: good2 },
        ];
        // Reborrow byte slices out of the owned frames for the stems view.
        let stems = [("stem-a", "stem-a"), ("stem-a", "stem-a"), ("stem-a", "stem-a")];
        let out = decode_frames_batch(&frames, &stems);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].as_ref().unwrap().game_id, "g0");
        assert_eq!(
            out[1].as_ref().unwrap_err().error_class(),
            "start_end"
        );
        assert_eq!(out[2].as_ref().unwrap().game_id, "g2");
    }

    /// KAT: wall candidacy takes the FIRST 136-int list under any of the
    /// three keys; non-136 lists are skipped, not taken.
    #[test]
    fn decode_wall_first_candidate_wins() {
        let mut s = serde_json::json!({"type": "start_game", "game_id": "w"});
        s["tiles"] = Value::Array(vec![Value::from(1); 5]);
        s["wall_tiles"] = Value::Array((0..136).map(|t: i64| Value::from(t)).collect());
        let b = game_bytes(&[s, end()]);
        let g = decode_game_object("o", "p", &b).unwrap();
        // `wall` absent, `wall_tiles` first valid candidate (key order
        // wall|wall_tiles|tiles, `decode.py:116-120`).
        assert_eq!(g.wall_tiles.unwrap()[7], 7);
    }
}
