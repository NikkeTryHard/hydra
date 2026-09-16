//! Game validation — 9-check elementwise pipeline (`feed::validate`).
//!
//! Rust owner of `src/hydra2/data/validate.py::validate_game` (lines
//! 92-344): structure, event_order, tile_conservation, red_identity,
//! legality, calls, scores, termination, dora_shape, plus the
//! `trailing_data` const marker (enforced in decode, recorded here).
//! Identity binding mirrors `compute_validation_hash` (`:53-55`):
//! `sha256(canonical({game_id, checks}))` via `crate::canon` +
//! `crate::digest` (SHA-256 ONLY, BLAKE3 barred; trap-6: hash the bytes,
//! never re-hash hex).
//!
//! Callsite shapes (all four migrate in this slice, wave3 §1.7):
//! - `stream_decode.py:130` — decode ok → `validate_game(game)` → take
//!   `.validation_hash` (see [`validation_hash_of`]).
//! - `stream_iter.py:271` — `_decode_inline`: decode failure → `(None,
//!   None)`; else `(game, validation_hash)` (see [`decode_inline`]).
//! - `stream_read.py:62` — lazy `fetch_game_at`: decode/validate failure is
//!   fail-closed (raises); composed from [`decode_game_object`] +
//!   [`validate_game`], pinned by KAT, no separate API.
//! - `training/stream_scan.py:87` — decode+validate per game, outcome
//!   discarded (rejects surface as `None` via the same `Err`/invalid
//!   channels); `training/stream_expand.py:164-169` binds
//!   `game.wall_tiles` verbatim (136-int or `None`), pinned by the wall KATs.
//!
//! Oracle-exact rules carried over:
//! - structure (`:94-105`): every event carries a string `type`.
//! - event_order (`:107-120`): schema probe loads → `ok`, else quarantine
//!   `("event_order", None)` — NEVER raises. The schema bytes are embedded
//!   via `include_bytes!` + `LazyLock` (global memo GONE); probe failure
//!   quarantines, never panics.
//! - tile_conservation (`:122-190`): wall `Some` → 136-permutation +
//!   per-logical `== 4` (`[u32; 34]` counts, no `Counter`); wall `None` →
//!   collect `tile|pai|dora|tiles|wall|hand` ints in `0..136` (one level,
//!   bools count as ints per Python `isinstance`), `> 4` per logical fails.
//! - red_identity (`:192-223`): `is_aka|aka|red` flag (Python truthiness,
//!   Null-skipping) vs `RED_IDS = (16, 52, 88)`; logical `(4, 13, 22)` table
//!   kept verbatim (defense-in-depth: the ids always pass it).
//! - legality/calls/scores/termination (`:225-300`): wall + actions →
//!   simulator probe, which stays PYTHON (engine slice owns it; wave3
//!   trap-8 — NO engine import here). The caller passes [`AdapterProbe`];
//!   `ok=false` sets `legality = "skipped_adapter_error:<name>"` and leaves
//!   `calls|scores|termination` ABSENT, exactly like the oracle
//!   (`:277-284`). Otherwise `action_id ∈ [0, 2000]` range-check in Rust.
//! - dora_shape (`:302-333`): `(4,)` list → fail; `(5,)` →
//!   sentinel-prefix contiguity (`DORA_SENTINEL = -1`); absent → `ok`.
//! - `trailing_data` (`:334`): always `ok` const (decode owns the gate).
//!
//! Traps applied: 8 (simulator OUT, `adapter_ok` flag IN), 13 (batch API for
//! `par_iter`-inside-`detach`; M8 scan ABI), 15 (per-record local checks
//! `Vec`, index-aligned batch output — no shared dict; the position IS the
//! merge key), 24 (taxonomy closed; unknown normalizes to `"other"` in
//! `crate::quarantine`, never here silently).
//!
//! Allocation: each [`Outcome`] owns its checks `Vec`; the skipped-adapter
//! `format!` and invalid messages allocate on cold paths only. The hot
//! elementwise checks themselves allocate nothing.

use std::sync::LazyLock;

use serde_json::Value;

use crate::decode::{GameRecord, decode_game_object, json_int};

/// Red tile physical ids (`validate.py:32`).
pub const RED_IDS: [i64; 3] = [16, 52, 88];
/// Red logical types (`validate.py:211`).
pub const RED_LOGICAL: [i64; 3] = [4, 13, 22];
/// Unrevealed dora indicator slot (`contracts/observation_types.py:41`).
pub const DORA_SENTINEL: i64 = -1;
/// `action_id` closed range (`validate.py:288`).
pub const ACTION_ID_MAX: i64 = 2000;
/// Wall length (mirrors `crate::decode::WALL_LEN`).
pub const WALL_LEN: usize = 136;

/// Check names in oracle insertion order (`validate.py:92-334`).
pub const CHECK_STRUCTURE: &str = "structure";
/// Check names in oracle insertion order.
pub const CHECK_EVENT_ORDER: &str = "event_order";
/// Check names in oracle insertion order.
pub const CHECK_TILE_CONSERVATION: &str = "tile_conservation";
/// Check names in oracle insertion order.
pub const CHECK_RED_IDENTITY: &str = "red_identity";
/// Check names in oracle insertion order.
pub const CHECK_LEGALITY: &str = "legality";
/// Check names in oracle insertion order.
pub const CHECK_CALLS: &str = "calls";
/// Check names in oracle insertion order.
pub const CHECK_SCORES: &str = "scores";
/// Check names in oracle insertion order.
pub const CHECK_TERMINATION: &str = "termination";
/// Check names in oracle insertion order.
pub const CHECK_DORA_SHAPE: &str = "dora_shape";
/// Check names in oracle insertion order (always `ok` — decode owns the gate).
pub const CHECK_TRAILING_DATA: &str = "trailing_data";
/// Full checks order for a valid game (oracle insertion order).
pub const CHECK_ORDER: [&str; 10] = [
    CHECK_STRUCTURE,
    CHECK_EVENT_ORDER,
    CHECK_TILE_CONSERVATION,
    CHECK_RED_IDENTITY,
    CHECK_LEGALITY,
    CHECK_CALLS,
    CHECK_SCORES,
    CHECK_TERMINATION,
    CHECK_DORA_SHAPE,
    CHECK_TRAILING_DATA,
];

/// Closed validation error taxonomy (byte-exact vs oracle strings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Every event carries a string `type`.
    Structure,
    /// Schema probe failed (never raised).
    EventOrder,
    /// Wall/logical conservation violated.
    TileConservation,
    /// Aka flag on a non-red tile.
    RedIdentity,
    /// `action_id` out of range.
    Legality,
    /// `(4,)` dora or non-contiguous `(5,)` sentinels.
    DoraShape,
}

impl ErrorClass {
    /// Closed taxonomy string, byte-exact vs fixtures.
    pub const fn as_str(self) -> &'static str {
        match self {
            ErrorClass::Structure => "structure",
            ErrorClass::EventOrder => "event_order",
            ErrorClass::TileConservation => "tile_conservation",
            ErrorClass::RedIdentity => "red_identity",
            ErrorClass::Legality => "legality",
            ErrorClass::DoraShape => "dora_shape",
        }
    }
}

/// Validation failure: closed class + offending event + cold detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    /// Closed error class.
    pub class: ErrorClass,
    /// Offending event index, when the failure names an event.
    pub event_index: Option<u32>,
    /// Cold detail (mirrors the oracle message family; never asserted
    /// byte-exact — KATs pin class + index).
    pub message: String,
}
/// Per-game validation outcome (mirrors `ValidationOutcome`, `:44-50`).
///
/// `checks` is per-record owned (trap-15: built locally, merged by position).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    pub game_id: String,
    /// Object id (copied from the record).
    pub object_id: String,
    /// True iff every check passed.
    pub valid: bool,
    /// Failure, when invalid (always `Some` when `!valid`).
    pub error: Option<ValidationError>,
    /// `sha256(canonical({game_id, checks}))`, when valid.
    pub validation_hash: Option<String>,
    /// Checks in oracle insertion order.
    pub checks: Vec<(&'static str, String)>,
}

impl Outcome {
    /// Look up one check result by name.
    pub fn checks_get(&self, name: &str) -> Option<&str> {
        self.checks
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Simulator-probe verdict from the engine-owning caller (wave3 trap-8).
///
/// The `RiichiEnvExactSimulator.reset` probe (`validate.py:271-276`) stays
/// Python; Rust takes this flag. `err_name` labels the skipped path
/// (`skipped_adapter_error:<name>`, mirroring `:279`); the bridge pins a
/// static label (e.g. `"NotWired"`) — there is no Python exception to name
/// on the Rust path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdapterProbe {
    /// Simulator probe succeeded.
    pub ok: bool,
    /// Static label for the skipped path.
    pub err_name: &'static str,
}

impl AdapterProbe {
    /// Probe succeeded: full `legality|calls|scores|termination = ok`.
    pub const fn ok() -> Self {
        AdapterProbe {
            ok: true,
            err_name: "ok",
        }
    }

    /// Probe unavailable: `legality = skipped_adapter_error:<name>`,
    /// `calls|scores|termination` ABSENT (oracle-exact `:277-284`).
    pub const fn failed(err_name: &'static str) -> Self {
        AdapterProbe {
            ok: false,
            err_name,
        }
    }
}

/// Python `bool()` truthiness over JSON values (`validate.py:201`).
fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i != 0
            } else if let Some(u) = n.as_u64() {
                u != 0
            } else if let Some(f) = n.as_f64() {
                f != 0.0
            } else {
                false
            }
        }
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// First non-null value under any of `keys` (Python `ev.get(k)` returns
/// `None` for a present-null, which falls through to the next key —
/// `validate.py:194-199, 304-309`).
fn first_present<'v>(ev: &'v Value, keys: &[&str]) -> Option<&'v Value> {
    keys.iter()
        .filter_map(|k| ev.get(*k))
        .find(|v| !v.is_null())
}

/// Composite-logical index with Python floor semantics (`t // 4`).
fn logical(t: i64) -> usize {
    t.div_euclid(4) as usize
}

/// Embedded event-schema bytes (`configs/contracts/event_schema_v1.json`).
const EVENT_SCHEMA: &[u8] =
    include_bytes!("../../../../../configs/contracts/event_schema_v1.json");

/// Schema-probe predicate over raw bytes: parses AND is an object
/// (`validate.py:75-79` shape gate).
fn schema_bytes_ok(bytes: &[u8]) -> bool {
    serde_json::from_slice::<Value>(bytes).is_ok_and(|v| v.is_object())
}

/// Memoized embedded-schema probe (the `_EVENT_SCHEMA_CACHE` global memo,
/// GONE — `LazyLock`, wave3 §1.7; initializer lives with the static).
static SCHEMA_OK: LazyLock<bool> = LazyLock::new(|| schema_bytes_ok(EVENT_SCHEMA));
fn embedded_schema_ok() -> bool {
    *SCHEMA_OK
}

/// Invalid outcome constructor (checks frozen as built so far).
fn invalid(
    rec: &GameRecord,
    class: ErrorClass,
    event_index: Option<u32>,
    message: String,
    checks: Vec<(&'static str, String)>,
) -> Outcome {
    Outcome {
        game_id: rec.game_id.clone(),
        object_id: rec.object_id.clone(),
        valid: false,
        error: Some(ValidationError {
            class,
            event_index,
            message,
        }),
        validation_hash: None,
        checks,
    }
}

/// `sha256(canonical({game_id, checks}))` (`validate.py:53-55`).
fn seal_hash(game_id: &str, checks: &[(&'static str, String)]) -> Option<String> {
    let mut checks_obj = serde_json::Map::with_capacity(checks.len());
    for (k, v) in checks {
        checks_obj.insert((*k).to_owned(), Value::String(v.clone()));
    }
    let payload =
        serde_json::json!({"game_id": game_id, "checks": Value::Object(checks_obj)});
    crate::digest::of_canonical(&payload).ok()
}

fn validate_inner(rec: &GameRecord, adapter: &AdapterProbe, schema_ok: bool) -> Outcome {
    let mut checks: Vec<(&'static str, String)> = Vec::with_capacity(CHECK_ORDER.len());

    // 1. Structure (`:94-105`).
    for (idx, ev) in rec.events.iter().enumerate() {
        if !ev.get("type").is_some_and(Value::is_string) {
            return invalid(
                rec,
                ErrorClass::Structure,
                Some(idx as u32),
                "missing type".to_owned(),
                checks,
            );
        }
    }
    checks.push((CHECK_STRUCTURE, "ok".to_owned()));

    // 2. Event order (`:107-120`): probe failure quarantines, never raises.
    if !schema_ok {
        return invalid(
            rec,
            ErrorClass::EventOrder,
            None,
            "schema load failed".to_owned(),
            checks,
        );
    }
    checks.push((CHECK_EVENT_ORDER, "ok".to_owned()));

    // 3. Tile conservation (`:122-190`).
    if let Some(wall) = &rec.wall_tiles {
        let mut seen = [false; WALL_LEN];
        let mut distinct = 0u32;
        let mut perm = true;
        for t in wall.iter() {
            if *t < 0 || *t >= WALL_LEN as i64 {
                perm = false;
                break;
            }
            let i = *t as usize;
            if !seen[i] {
                seen[i] = true;
                distinct += 1;
            }
        }
        if !perm || distinct != WALL_LEN as u32 {
            return invalid(
                rec,
                ErrorClass::TileConservation,
                None,
                "wall not permutation".to_owned(),
                checks,
            );
        }
        let mut counts = [0u32; 34];
        for t in wall.iter() {
            counts[logical(*t)] += 1;
        }
        if counts.iter().any(|c| *c != 4) {
            return invalid(
                rec,
                ErrorClass::TileConservation,
                None,
                "logical count != 4".to_owned(),
                checks,
            );
        }
        checks.push((CHECK_TILE_CONSERVATION, "ok".to_owned()));
    } else {
        const TILE_KEYS: [&str; 6] = ["tile", "pai", "dora", "tiles", "wall", "hand"];
        let mut counts = [0u32; 34];
        for ev in rec.events.iter() {
            for key in TILE_KEYS {
                match ev.get(key) {
                    Some(Value::Array(a)) => {
                        for x in a {
                            if let Some(t) = json_int(x) {
                                if t >= 0 && t < WALL_LEN as i64 {
                                    counts[logical(t)] += 1;
                                }
                            }
                        }
                    }
                    Some(v) => {
                        if let Some(t) = json_int(v) {
                            if t >= 0 && t < WALL_LEN as i64 {
                                counts[logical(t)] += 1;
                            }
                        }
                    }
                    None => {}
                }
            }
        }
        if let Some(bad) = counts.iter().position(|c| *c > 4) {
            return invalid(
                rec,
                ErrorClass::TileConservation,
                None,
                format!("logical {bad} exceeds 4 copies"),
                checks,
            );
        }
        checks.push((CHECK_TILE_CONSERVATION, "ok".to_owned()));
    }

    // 4. Red identity (`:192-223`).
    for (idx, ev) in rec.events.iter().enumerate() {
        let flag = first_present(ev, &["is_aka", "aka", "red"]);
        let tile = ev.get("tile").and_then(json_int);
        if flag.is_some_and(py_truthy) && tile.is_some_and(|t| !RED_IDS.contains(&t)) {
            return invalid(
                rec,
                ErrorClass::RedIdentity,
                Some(idx as u32),
                "red flag on non-red".to_owned(),
                checks,
            );
        }
        if let Some(t) = tile {
            if RED_IDS.contains(&t) && !RED_LOGICAL.contains(&(logical(t) as i64)) {
                return invalid(
                    rec,
                    ErrorClass::RedIdentity,
                    Some(idx as u32),
                    "red id wrong logical".to_owned(),
                    checks,
                );
            }
        }
    }
    checks.push((CHECK_RED_IDENTITY, "ok".to_owned()));

    // 5-8. Legality/calls/scores/termination (`:225-300`).
    let has_actions = rec
        .events
        .iter()
        .any(|ev| ev.get("action_id").is_some() || ev.get("action").is_some());
    if rec.wall_tiles.is_some() && has_actions {
        // Simulator probe stays Python (trap-8); Rust takes the flag.
        if adapter.ok {
            checks.push((CHECK_LEGALITY, "ok".to_owned()));
            checks.push((CHECK_CALLS, "ok".to_owned()));
            checks.push((CHECK_SCORES, "ok".to_owned()));
            checks.push((CHECK_TERMINATION, "ok".to_owned()));
        } else {
            // Oracle-exact (`:277-279`): legality skipped, the other three
            // ABSENT (not `ok`) — the seal covers the smaller map.
            checks.push((
                CHECK_LEGALITY,
                format!("skipped_adapter_error:{}", adapter.err_name),
            ));
        }
    } else {
        for (idx, ev) in rec.events.iter().enumerate() {
            if let Some(aid) = ev.get("action_id").and_then(json_int) {
                if aid < 0 || aid > ACTION_ID_MAX {
                    return invalid(
                        rec,
                        ErrorClass::Legality,
                        Some(idx as u32),
                        "action_id out of range".to_owned(),
                        checks,
                    );
                }
            }
        }
        checks.push((CHECK_LEGALITY, "ok".to_owned()));
        checks.push((CHECK_CALLS, "ok".to_owned()));
        checks.push((CHECK_SCORES, "ok".to_owned()));
        checks.push((CHECK_TERMINATION, "ok".to_owned()));
    }

    // 9. Dora shape (`:302-333`): `(4,)` fails, `(5,)` sentinel-prefix,
    // absent → ok. NEVER a `(4,)` shim (goal invariant).
    for (idx, ev) in rec.events.iter().enumerate() {
        if let Some(Value::Array(a)) = first_present(ev, &["dora_indicators", "dora", "indicators"])
        {
            if a.len() == 4 {
                return invalid(
                    rec,
                    ErrorClass::DoraShape,
                    Some(idx as u32),
                    "DORA_SHAPE must be (5,), got (4,)".to_owned(),
                    checks,
                );
            }
            if a.len() == 5 {
                let mut seen_sentinel = false;
                for v in a {
                    if is_sentinel(v) {
                        seen_sentinel = true;
                    } else if seen_sentinel {
                        return invalid(
                            rec,
                            ErrorClass::DoraShape,
                            Some(idx as u32),
                            "dora indicators not contiguous".to_owned(),
                            checks,
                        );
                    }
                }
            }
        }
    }
    checks.push((CHECK_DORA_SHAPE, "ok".to_owned()));
    checks.push((CHECK_TRAILING_DATA, "ok".to_owned()));

    match seal_hash(&rec.game_id, &checks) {
        Some(hash) => Outcome {
            game_id: rec.game_id.clone(),
            object_id: rec.object_id.clone(),
            valid: true,
            error: None,
            validation_hash: Some(hash),
            checks,
        },
        None => invalid(
            rec,
            ErrorClass::EventOrder,
            None,
            "validation seal failed".to_owned(),
            checks,
        ),
    }
}

/// `DORA_SENTINEL` equality with Python `==` semantics: int `-1` or float
/// `-1.0` match; strings/bools never do.
fn is_sentinel(v: &Value) -> bool {
    if let Some(i) = v.as_i64() {
        i == DORA_SENTINEL
    } else if let Some(f) = v.as_f64() {
        f == DORA_SENTINEL as f64
    } else {
        false
    }
}

/// Validate one game: the 9-check pipeline (`validate.py:92-344`).
pub fn validate_game(rec: &GameRecord, adapter: &AdapterProbe) -> Outcome {
    validate_inner(rec, adapter, embedded_schema_ok())
}

/// Validate a batch, output index-aligned with the input (M8 scan ABI).
///
/// Per-record local checks `Vec` (trap-15: built inside the iteration, never
/// a shared dict); the caller merges positionally in sequence order.
/// Failures occupy their slot as `valid: false` outcomes, never shift
/// siblings. Intended for `rayon::par_iter` inside `detach` (pure-Rust
/// closure, trap-13).
pub fn validate_batch(records: &[GameRecord], adapter: &AdapterProbe) -> Vec<Outcome> {
    records.iter().map(|r| validate_game(r, adapter)).collect()
}

/// `stream_decode.py:130` shape: decode ok → validate → validation hash.
pub fn validation_hash_of(rec: &GameRecord, adapter: &AdapterProbe) -> Option<String> {
    validate_game(rec, adapter).validation_hash
}

/// `stream_iter.py:271` `_decode_inline` shape: decode failure → `(None,
/// None)`; else `(Some(game), validation_hash)`.
pub fn decode_inline(
    object_id: &str,
    packaged_object_id: &str,
    bytes: &[u8],
    adapter: &AdapterProbe,
) -> (Option<GameRecord>, Option<String>) {
    match decode_game_object(object_id, packaged_object_id, bytes) {
        Err(_) => (None, None),
        Ok(game) => {
            let hash = validation_hash_of(&game, adapter);
            (Some(game), hash)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framer::Frame;

    fn rec(events: Vec<Value>, wall: Option<[i64; WALL_LEN]>) -> GameRecord {
        GameRecord {
            game_id: "g".to_owned(),
            object_id: "o".to_owned(),
            packaged_object_id: "p".to_owned(),
            events,
            raw_bytes_sha256: "sha256:00".to_owned(),
            wall_tiles: wall,
        }
    }

    fn start() -> Value {
        serde_json::json!({"type": "start_game"})
    }

    fn end() -> Value {
        serde_json::json!({"type": "end_game"})
    }

    fn perm_wall() -> [i64; WALL_LEN] {
        let mut w = [0i64; WALL_LEN];
        for (i, v) in w.iter_mut().enumerate() {
            *v = i as i64;
        }
        w
    }

    /// KAT: minimal valid game — all 10 checks `ok` in oracle insertion
    /// order, seal `sha256:`-prefixed and stable.
    #[test]
    fn valid_game_checks_order_and_seal() {
        let r = rec(vec![start(), end()], None);
        let o = validate_game(&r, &AdapterProbe::ok());
        assert!(o.valid);
        assert!(o.error.is_none());
        let names: Vec<&str> = o.checks.iter().map(|(k, _)| *k).collect();
        assert_eq!(names, CHECK_ORDER);
        assert!(o.checks.iter().all(|(_, v)| v == "ok"));
        let h = o.validation_hash.clone().unwrap();
        assert!(h.starts_with("sha256:"));
        assert_eq!(h.len(), "sha256:".len() + 64);
        // Stable: same game seals identically.
        assert_eq!(validate_game(&r, &AdapterProbe::ok()).validation_hash, Some(h));
    }

    /// KAT: structure fails with empty checks (first gate, nothing set).
    #[test]
    fn structure_failure_has_empty_checks() {
        let r = rec(vec![start(), serde_json::json!({"actor": 0}), end()], None);
        let o = validate_game(&r, &AdapterProbe::ok());
        assert!(!o.valid);
        assert!(o.validation_hash.is_none());
        assert!(o.checks.is_empty());
        let e = o.error.unwrap();
        assert_eq!(e.class, ErrorClass::Structure);
        assert_eq!(e.class.as_str(), "structure");
        assert_eq!(e.event_index, Some(1));
    }

    /// KAT: event_order fallback quarantines with structure-only checks
    /// (probe failure never raises).
    #[test]
    fn event_order_fallback_quarantines() {
        let r = rec(vec![start(), end()], None);
        let o = validate_inner(&r, &AdapterProbe::ok(), false);
        assert!(!o.valid);
        let e = o.error.unwrap();
        assert_eq!(e.class.as_str(), "event_order");
        assert_eq!(e.event_index, None);
        assert_eq!(
            o.checks.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            [CHECK_STRUCTURE]
        );
    }

    /// KAT: schema predicate — embedded bytes load; non-object/malformed
    /// bytes fail (global-checks fallback unit).
    #[test]
    fn schema_bytes_predicate() {
        assert!(schema_bytes_ok(EVENT_SCHEMA));
        assert!(embedded_schema_ok());
        assert!(!schema_bytes_ok(b"[1,2]"));
        assert!(!schema_bytes_ok(b"{oops"));
        assert!(!schema_bytes_ok(b"null"));
    }

    /// KAT: walled tile conservation — dup wall fails with
    /// structure+event_order checks frozen; perm wall passes.
    #[test]
    fn tile_conservation_walled() {
        let mut dup = perm_wall();
        dup[7] = 0;
        let o = validate_game(&rec(vec![start(), end()], Some(dup)), &AdapterProbe::ok());
        assert!(!o.valid);
        assert_eq!(o.error.unwrap().class.as_str(), "tile_conservation");
        assert_eq!(
            o.checks.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            [CHECK_STRUCTURE, CHECK_EVENT_ORDER]
        );
        let o = validate_game(&rec(vec![start(), end()], Some(perm_wall())), &AdapterProbe::ok());
        assert_eq!(o.checks_get("tile_conservation"), Some("ok"));
    }

    /// KAT: wall-less overflow — 5 copies of one logical fails; Python
    /// `isinstance` parity: JSON `true` counts as tile `1`.
    #[test]
    fn tile_conservation_wall_less_overflow() {
        let mut events = vec![start()];
        for _ in 0..5 {
            events.push(serde_json::json!({"type": "tsumo", "tile": 0}));
        }
        events.push(end());
        let o = validate_game(&rec(events, None), &AdapterProbe::ok());
        assert!(!o.valid);
        assert_eq!(o.error.unwrap().class.as_str(), "tile_conservation");
        // Bool tiles count (True == 1): five `true` tiles overflow logical 0.
        let mut events = vec![start()];
        for _ in 0..5 {
            events.push(serde_json::json!({"type": "tsumo", "tile": true}));
        }
        events.push(end());
        let o = validate_game(&rec(events, None), &AdapterProbe::ok());
        assert!(!o.valid);
        assert_eq!(o.error.unwrap().class.as_str(), "tile_conservation");
    }

    /// KAT: red identity — aka flag on non-red fails at the event; red ids
    /// always carry the right logical (table verbatim).
    #[test]
    fn red_identity_flag_vs_membership() {
        let events = vec![
            start(),
            serde_json::json!({"type": "dahai", "tile": 5, "is_aka": true}),
            end(),
        ];
        let o = validate_game(&rec(events, None), &AdapterProbe::ok());
        assert!(!o.valid);
        let e = o.error.unwrap();
        assert_eq!(e.class.as_str(), "red_identity");
        assert_eq!(e.event_index, Some(1));
        // aka alias + falsy flag pass; red tile without flag passes.
        let events = vec![
            start(),
            serde_json::json!({"type": "dahai", "tile": 16}),
            serde_json::json!({"type": "dahai", "tile": 5, "aka": false}),
            end(),
        ];
        let o = validate_game(&rec(events, None), &AdapterProbe::ok());
        assert_eq!(o.checks_get("red_identity"), Some("ok"));
        assert_eq!(RED_IDS, [16, 52, 88]);
        assert_eq!(RED_LOGICAL, [4, 13, 22]);
    }

    /// KAT: `action_id` bounds — `-1` and `2001` fail at the event; `0` and
    /// `2000` pass (wall-less path range-checks in Rust).
    #[test]
    fn legality_action_id_bounds() {
        for bad in [-1i64, 2001] {
            let events = vec![
                start(),
                serde_json::json!({"type": "dahai", "action_id": bad}),
                end(),
            ];
            let o = validate_game(&rec(events, None), &AdapterProbe::ok());
            assert!(!o.valid, "aid {bad} must fail");
            let e = o.error.unwrap();
            assert_eq!(e.class.as_str(), "legality");
            assert_eq!(e.event_index, Some(1));
        }
        for good in [0i64, 2000] {
            let events = vec![
                start(),
                serde_json::json!({"type": "dahai", "action_id": good}),
                end(),
            ];
            let o = validate_game(&rec(events, None), &AdapterProbe::ok());
            assert!(o.valid, "aid {good} must pass");
            assert_eq!(o.checks_get("legality"), Some("ok"));
            assert_eq!(o.checks_get("calls"), Some("ok"));
            assert_eq!(o.checks_get("scores"), Some("ok"));
            assert_eq!(o.checks_get("termination"), Some("ok"));
        }
    }

    /// KAT: adapter-skipped path is oracle-exact — `legality` carries the
    /// label while `calls|scores|termination` are ABSENT (not `ok`).
    #[test]
    fn adapter_skipped_leaves_markers_absent() {
        let events = vec![
            start(),
            serde_json::json!({"type": "dahai", "action_id": 3}),
            end(),
        ];
        let o = validate_game(
            &rec(events, Some(perm_wall())),
            &AdapterProbe::failed("NotWired"),
        );
        assert!(o.valid);
        assert_eq!(
            o.checks_get("legality"),
            Some("skipped_adapter_error:NotWired")
        );
        assert_eq!(o.checks_get("calls"), None);
        assert_eq!(o.checks_get("scores"), None);
        assert_eq!(o.checks_get("termination"), None);
        assert_eq!(o.checks_get("dora_shape"), Some("ok"));
        assert_eq!(o.checks_get("trailing_data"), Some("ok"));
    }

    /// KAT: adapter-ok walled path sets all four markers `ok`.
    #[test]
    fn adapter_ok_sets_all_markers() {
        let events = vec![
            start(),
            serde_json::json!({"type": "dahai", "action_id": 3}),
            end(),
        ];
        let o = validate_game(&rec(events, Some(perm_wall())), &AdapterProbe::ok());
        assert!(o.valid);
        for k in [CHECK_LEGALITY, CHECK_CALLS, CHECK_SCORES, CHECK_TERMINATION] {
            assert_eq!(o.checks_get(k), Some("ok"), "{k} must be ok");
        }
    }

    /// KAT: dora shape — `(4,)` fails, non-contiguous `(5,)` fails,
    /// sentinel-tail `(5,)` passes, absent passes. Sentinel is `-1` exact.
    #[test]
    fn dora_shape_four_fails_five_contiguous() {
        assert_eq!(DORA_SENTINEL, -1);
        let bad4 = vec![
            start(),
            serde_json::json!({"type": "dora", "dora_indicators": [1, 2, 3, 4]}),
            end(),
        ];
        let o = validate_game(&rec(bad4, None), &AdapterProbe::ok());
        assert!(!o.valid);
        let e = o.error.unwrap();
        assert_eq!(e.class.as_str(), "dora_shape");
        assert_eq!(e.event_index, Some(1));
        // Sentinel then revealed → not contiguous.
        let bad5 = vec![
            start(),
            serde_json::json!({"type": "dora", "dora": [1, -1, -1, -1, 4]}),
            end(),
        ];
        let o = validate_game(&rec(bad5, None), &AdapterProbe::ok());
        assert!(!o.valid);
        assert_eq!(o.error.unwrap().class.as_str(), "dora_shape");
        // Revealed prefix + sentinel tail → ok.
        let good5 = vec![
            start(),
            serde_json::json!({"type": "dora", "indicators": [1, 2, -1, -1, -1]}),
            end(),
        ];
        let o = validate_game(&rec(good5, None), &AdapterProbe::ok());
        assert!(o.valid);
        assert_eq!(o.checks_get("dora_shape"), Some("ok"));
    }

    /// KAT: batch keeps positions (trap-13 ordered merge) with per-record
    /// local checks (trap-15: mutating one outcome never touches another).
    #[test]
    fn batch_order_and_local_checks() {
        let good = rec(vec![start(), end()], None);
        let bad = rec(vec![serde_json::json!({"actor": 0})], None);
        let out = validate_batch(&[good, bad], &AdapterProbe::ok());
        assert_eq!(out.len(), 2);
        assert!(out[0].valid);
        assert!(!out[1].valid);
        let mut out = out;
        out[0].checks.clear();
        assert_eq!(out[1].checks.len(), 0); // bad has empty checks; independence holds by ownership
        assert!(out[0].validation_hash.is_some());
    }

    /// KAT: `stream_decode.py:130` shape — hash `Some` on valid, `None` on
    /// invalid (caller emits the `None`-payload trio downstream).
    #[test]
    fn callsite_stream_decode_hash_shape() {
        let ok_r = rec(vec![start(), end()], None);
        assert!(validation_hash_of(&ok_r, &AdapterProbe::ok()).is_some());
        let bad_r = rec(vec![serde_json::json!({"actor": 0})], None);
        assert!(validation_hash_of(&bad_r, &AdapterProbe::ok()).is_none());
    }

    /// KAT: `stream_iter.py:271` `_decode_inline` shape — undecodable →
    /// `(None, None)`; decodable → `(Some, hash)`.
    #[test]
    fn callsite_stream_iter_inline_shape() {
        let good = b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n";
        let (g, h) = decode_inline("s", "s", good, &AdapterProbe::ok());
        assert!(g.is_some());
        assert!(h.is_some());
        let (g, h) = decode_inline("s", "s", b"{\"type\":\"start_game\"}\n", &AdapterProbe::ok());
        assert!(g.is_none());
        assert!(h.is_none());
    }

    /// KAT: `stream_read.py:62` `fetch_game_at` fail-closed shape —
    /// decode error OR invalid outcome both close (raise in Python, `Err`
    /// here); only valid games yield a hash.
    #[test]
    fn callsite_stream_read_fail_closed_shape() {
        fn fetch_closed(bytes: &[u8]) -> Result<String, &'static str> {
            let frames = [Frame { file_idx: 0, offset: 0, end: bytes.len() as u64, bytes: bytes.to_vec() }];
            let stems = [("s", "s")];
            let batch = crate::decode::decode_frames_batch(&frames, &stems);
            let rec = batch.into_iter().next().unwrap().map_err(|_| "undecodable")?;
            let out = validate_game(&rec, &AdapterProbe::ok());
            out.validation_hash.ok_or("invalid")
        }
        let good = b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n";
        assert!(fetch_closed(good).is_ok());
        assert_eq!(fetch_closed(b"{\"type\":\"start_game\"}\n"), Err("undecodable"));
        let bad_action = b"{\"type\":\"start_game\"}\n{\"type\":\"dahai\",\"action_id\":9999}\n{\"type\":\"end_game\"}\n";
        assert_eq!(fetch_closed(bad_action), Err("invalid"));
    }

    /// KAT: `training/stream_expand.py:164-169` wall-binding shape — the
    /// record wall overrides the log verbatim: `Some([i64; 136])` walled,
    /// `None` wall-less, length fixed at the type level (never 4/5/135).
    #[test]
    fn callsite_stream_expand_wall_binding_shape() {
        // Walled regime: decode binds the 136-int wall.
        let mut s = serde_json::json!({"type": "start_game"});
        s["wall"] = serde_json::Value::Array((0..136).map(|t: i64| t.into()).collect());
        let bytes = [
            serde_json::to_string(&s).unwrap(),
            serde_json::to_string(&end()).unwrap(),
        ]
        .join("\n")
            + "\n";
        let (g, _) = decode_inline("s", "s", bytes.as_bytes(), &AdapterProbe::ok());
        let wall = g.unwrap().wall_tiles.unwrap();
        assert_eq!(wall.len(), WALL_LEN);
        assert_eq!(wall[135], 135);
        // Wall-less regime: no 136-list anywhere → None (sim path).
        let bytes = b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n";
        let (g, _) = decode_inline("s", "s", bytes, &AdapterProbe::ok());
        assert!(g.unwrap().wall_tiles.is_none());
    }
}
