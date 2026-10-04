//! rl/runtime_snapshot_pool: pool-snapshot (core-snapshot full blob) persistence.
//!
//! DAG: `super::{runtime_snapshot, core, pool}` + `serde_json`. FORBIDS:
//! PyO3, wall-clock, RNG, alternate snapshot formats (the persisted bytes
//! are the actual `PyNativePool.snapshot` payload, canonicalized; row and
//! action facts are verified with the existing `GameRow::from_json` and
//! `StoredAction` parsers, and full engine restore stays the
//! `PyNativePool.restore` path).
//!
//! The pool blob is the `CoreSnapshotFullBlob` domain the commit path
//! references: replay prefixes, full LP/V captures, per-seat sampler
//! cursors, and retained READY rows — never opaque RNG. `save_pool`
//! derives the control-facing collection facts (frontier, terminal set,
//! tickets, captured rows) from the verified snapshot and returns them
use crate::rl::runtime_snapshot::{
    canon_bytes, parse_json, parse_pool_ref, publish_runtime_canonical, read_verified_blob,
    reject_escaped_refs, rt_err,
};
use std::sync::{LazyLock, Mutex};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde_json::{Map, Value};

use crate::rl::core::GameRow;
use crate::rl::pool::StoredAction;
use crate::rl::trajectory::TapeError;

/// Bare 64-hex digest (pool snapshot digests are unprefixed hex).
fn parse_hex_digest(text: &str, what: &str) -> Result<[u8; 32], TapeError> {
    if text.len() != 64 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(rt_err(format!("{what} must be 32 bytes hex")));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[2 * i..2 * i + 2], 16)
            .map_err(|_| rt_err(format!("{what} must be 32 bytes hex")))?;
    }
    Ok(out)
}

fn req_str(map: &Map<String, Value>, key: &str, what: &str) -> Result<String, TapeError> {
    map.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| rt_err(format!("{what} lacks string {key}")))
}

fn req_u64(map: &Map<String, Value>, key: &str, what: &str) -> Result<u64, TapeError> {
    map.get(key).and_then(Value::as_u64).ok_or_else(|| {
        rt_err(format!("{what} lacks integer {key}"))
    })
}

/// Verified pool snapshot view (borrowed shapes needed downstream).
pub struct VerifiedPool {
    /// Canonical snapshot bytes (re-emitted, stored in the blob).
    pub canonical: Vec<u8>,
    pub pool_id: String,
    pub generation: u64,
    pub frontier: Value,
    pub terminal_set: Vec<String>,
    pub tickets: Value,
    pub captured_rows: u64,
    /// Wall ids in cohort order.
    pub wall_ids: Vec<String>,
}
/// Collection facts without the 400MB canonical bytes (cacheable; every
/// repeat check of one digest reuses these instead of re-parsing rows).
/// Bytes stay content-addressed on disk: a cache hit still requires the
/// file to hash to the claimed digest first, so corrupted or swapped
/// bytes can never hit.
#[derive(Clone, Debug)]
pub struct PoolFacts {
    pub pool_id: String,
    pub generation: u64,
    pub frontier: Value,
    pub terminal_set: Vec<String>,
    pub tickets: Value,
    pub captured_rows: u64,
    /// Wall ids in cohort order.
    pub wall_ids: Vec<String>,
}

impl From<&VerifiedPool> for PoolFacts {
    fn from(verified: &VerifiedPool) -> Self {
        Self {
            pool_id: verified.pool_id.clone(),
            generation: verified.generation,
            frontier: verified.frontier.clone(),
            terminal_set: verified.terminal_set.clone(),
            tickets: verified.tickets.clone(),
            captured_rows: verified.captured_rows,
            wall_ids: verified.wall_ids.clone(),
        }
    }
}

/// At most two pool digests are live per run (current + one future); 16
/// bounds multi-run processes (tests) with clear-on-overflow. Digest-keyed
/// only: same digest means byte-identical content by CAS law, so no
/// invalidation on quiesce/save/restore. Poisoned snapshots never verify,
/// hence never enter.
static POOL_FACTS_CACHE: LazyLock<Mutex<std::collections::HashMap<String, PoolFacts>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

fn cache_lookup(digest: &str) -> Option<PoolFacts> {
    POOL_FACTS_CACHE.lock().ok()?.get(digest).cloned()
}

fn cache_store(digest: String, facts: PoolFacts) {
    if let Ok(mut cache) = POOL_FACTS_CACHE.lock() {
        if cache.len() >= 16 {
            cache.clear();
        }
        cache.insert(digest, facts);
    }
}

/// Verify a `PyNativePool.snapshot` payload with the existing row/action
/// parsers plus closed structural gates. Returns the verified view with
/// derived collection facts. No engine replay happens here (that is the
/// `NativePool::restore` path); every fact below is actual snapshot data.
pub fn verify_pool_snapshot(doc: &[u8]) -> Result<VerifiedPool, TapeError> {
    let value = parse_json(doc, "pool snapshot")?;
    let map = value
        .as_object()
        .ok_or_else(|| rt_err("pool snapshot must be an object"))?;
    if map.len() != 6
        || !["pool_id", "generation", "games", "slots", "leases", "poison"]
            .iter()
            .all(|k| map.contains_key(*k))
    {
        return Err(rt_err(
            "pool snapshot must carry exactly pool_id+generation+games+slots+leases+poison",
        ));
    }
    let pool_id = req_str(map, "pool_id", "pool")?;
    if pool_id.is_empty() {
        return Err(rt_err("pool snapshot pool_id must be nonempty"));
    }
    let generation = req_u64(map, "generation", "pool")?;
    match map.get("poison").unwrap_or(&Value::Null) {
        Value::Null => {}
        Value::String(_) => return Err(rt_err("pool snapshot carries worker poison")),
        _ => return Err(rt_err("pool snapshot poison has a bad shape")),
    }
    let games = map
        .get("games")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err("pool lacks games array"))?;
    if games.is_empty() {
        return Err(rt_err("pool cohort must be nonempty"));
    }
    let mut wall_ids = Vec::with_capacity(games.len());
    let mut seen_walls = HashSet::new();
    let mut terminal_set = Vec::new();
    let mut captured_rows: u64 = 0;
    for (i, game) in games.iter().enumerate() {
        let what = format!("pool games[{i}]");
        let gmap = game
            .as_object()
            .ok_or_else(|| rt_err(format!("{what} must be an object")))?;
        if gmap.len() != 8
            || ![
                "wall_id",
                "ordinal",
                "applied",
                "digest",
                "captured",
                "sample_counters",
                "terminal",
                "completed",
            ]
            .iter()
            .all(|k| gmap.contains_key(*k))
        {
            return Err(rt_err(format!("{what} has a bad shape")));
        }
        let wall_id = req_str(gmap, "wall_id", &what)?;
        if !seen_walls.insert(wall_id.clone()) {
            return Err(rt_err(format!("{what} duplicate wall {wall_id:?}")));
        }
        wall_ids.push(wall_id);
        req_u64(gmap, "ordinal", &what)?;
        let digest_hex = req_str(gmap, "digest", &what)?;
        parse_hex_digest(&digest_hex, &format!("{what}.digest"))?;
        let captured = req_u64(gmap, "captured", &what)?;
        captured_rows = captured_rows
            .checked_add(captured)
            .ok_or_else(|| rt_err("pool captured rows overflow"))?;
        let counters = gmap
            .get("sample_counters")
            .and_then(Value::as_array)
            .ok_or_else(|| rt_err(format!("{what} lacks per-seat sample_counters")))?;
        if counters.len() != 4
            || counters.iter().any(|c| c.as_u64().is_none())
        {
            return Err(rt_err(format!(
                "{what} sample_counters must hold four u64 seats"
            )));
        }
        let terminal = gmap
            .get("terminal")
            .and_then(Value::as_bool)
            .ok_or_else(|| rt_err(format!("{what} lacks boolean terminal")))?;
        if terminal {
            terminal_set.push(
                gmap.get("wall_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            );
        }
        for action in gmap
            .get("applied")
            .and_then(Value::as_array)
            .ok_or_else(|| rt_err(format!("{what} lacks applied array")))?
        {
            // Existing pool action parser (kind vocabulary + shapes).
            let stored = StoredAction::from_json(action)
                .map_err(|e| rt_err(format!("{what} applied action rejected: {e}")))?;
            stored
                .to_canonical()
                .map_err(|e| rt_err(format!("{what} applied action unrestorable: {e}")))?;
        }
        let mut completed_count: u64 = 0;
        for row_doc in gmap
            .get("completed")
            .and_then(Value::as_array)
            .ok_or_else(|| rt_err(format!("{what} lacks completed array")))?
        {
            // Existing core row parser (full compact facts: LP, value,
            // selected, cursors, hashes, lineage, frozen terminal facts).
            let row = GameRow::from_json(row_doc)
                .map_err(|e| rt_err(format!("{what} captured row rejected: {e}")))?;
            if row.actor > 3 {
                return Err(rt_err(format!("{what} row actor outside four seats")));
            }
            if !row.selected_logp.is_finite() || row.selected_logp > 0.0 {
                return Err(rt_err(format!("{what} row logp not a finite log-prob")));
            }
            if row.value4.iter().any(|v| !v.is_finite()) {
                return Err(rt_err(format!("{what} row value4 nonfinite")));
            }
            if !row.legal_ids.contains(&row.selected) {
                return Err(rt_err(format!("{what} row selected outside legal ids")));
            }
            completed_count = completed_count
                .checked_add(1)
                .ok_or_else(|| rt_err("pool completed rows overflow"))?;
        }
        if completed_count != captured {
            return Err(rt_err(format!(
                "{what} completed rows {completed_count} != captured counter {captured}"
            )));
        }
    }
    terminal_set.sort();
    // Slots: quiesced lanes only (Free/Ready); staged rows keep uniform
    // reservations + cursors + observation hashes verbatim.
    let slots = map
        .get("slots")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err("pool lacks slots array"))?;
    let mut seen_slots = HashSet::new();
    let mut tickets = Vec::with_capacity(slots.len());
    let mut max_round: u64 = 0;
    let mut staged_by_slot: HashMap<u64, usize> = HashMap::new();
    for (i, slot) in slots.iter().enumerate() {
        let what = format!("pool slots[{i}]");
        let smap = slot
            .as_object()
            .ok_or_else(|| rt_err(format!("{what} must be an object")))?;
        if smap.len() != 8
            || !["id", "epoch", "phase", "bucket_t", "capacity", "chunk", "logical_round", "staged"]
                .iter()
                .all(|k| smap.contains_key(*k))
        {
            return Err(rt_err(format!("{what} has a bad shape")));
        }
        let id = req_u64(smap, "id", &what)?;
        if !seen_slots.insert(id) {
            return Err(rt_err(format!("{what} duplicate slot {id}")));
        }
        let epoch = req_u64(smap, "epoch", &what)?;
        let phase = req_str(smap, "phase", &what)?;
        let ticket_phase = match phase.as_str() {
            "Free" => "free",
            "Ready" => "ready",
            _ => {
                return Err(rt_err(format!("{what} phase {phase:?} unrestorable (settle first)")));
            }
        };
        let bucket_t = req_u64(smap, "bucket_t", &what)?;
        if ![32, 64, 128, 256].contains(&bucket_t) {
            return Err(rt_err(format!("{what} bucket {bucket_t} outside 32/64/128/256")));
        }
        let capacity = req_u64(smap, "capacity", &what)?;
        if capacity == 0 || capacity > 128 {
            return Err(rt_err(format!("{what} capacity {capacity} outside 1..=128")));
        }
        let chunk = req_u64(smap, "chunk", &what)?;
        let logical_round = req_u64(smap, "logical_round", &what)?;
        max_round = max_round.max(logical_round);
        let staged = smap
            .get("staged")
            .and_then(Value::as_array)
            .ok_or_else(|| rt_err(format!("{what} lacks staged array")))?;
        if staged.len() as u64 > capacity {
            return Err(rt_err(format!("{what} staged rows exceed capacity")));
        }
        let mut cursor: u64 = 0;
        for (j, row) in staged.iter().enumerate() {
            let rwhat = format!("{what}.staged[{j}]");
            let rmap = row
                .as_object()
                .ok_or_else(|| rt_err(format!("{rwhat} must be an object")))?;
            if rmap.len() != 5
                || !["wall_id", "actor", "uniform_bits", "cursor_before", "obs_hash"]
                    .iter()
                    .all(|k| rmap.contains_key(*k))
            {
                return Err(rt_err(format!("{rwhat} has a bad shape")));
            }
            let wall = req_str(rmap, "wall_id", &rwhat)?;
            if !wall_ids.contains(&wall) {
                return Err(rt_err(format!("{rwhat} wall escapes the cohort")));
            }
            let actor = req_u64(rmap, "actor", &rwhat)?;
            if actor > 3 {
                return Err(rt_err(format!("{rwhat} actor outside four seats")));
            }
            let bits = req_u64(rmap, "uniform_bits", &rwhat)?;
            let uniform = f64::from_bits(bits);
            if !(0.0..1.0).contains(&uniform) {
                return Err(rt_err(format!("{rwhat} uniform escapes [0, 1)")));
            }
            cursor = cursor.max(req_u64(rmap, "cursor_before", &rwhat)?);
            let obs = req_str(rmap, "obs_hash", &rwhat)?;
            parse_hex_digest(&obs, &format!("{rwhat}.obs_hash"))?;
        }
        staged_by_slot.insert(id, staged.len());
        tickets.push(serde_json::json!({
            "ticket_descriptor": format!("{pool_id}/slot-{id}/t-{bucket_t}/chunk-{chunk}/round-{logical_round}"),
            "phase": ticket_phase,
            "slot_epoch": epoch,
            "stream_cursor": cursor,
        }));
    }
    tickets.sort_by(|a: &Value, b: &Value| {
        a.get("ticket_descriptor")
            .and_then(Value::as_str)
            .cmp(&b.get("ticket_descriptor").and_then(Value::as_str))
    });
    // Leases must bind adopted slots (epoch + row-count agreement); the
    // owner re-registers buffers and commits against the same slab.
    let leases = map
        .get("leases")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err("pool lacks leases array"))?;
    for (i, lease) in leases.iter().enumerate() {
        let what = format!("pool leases[{i}]");
        let lmap = lease
            .as_object()
            .ok_or_else(|| rt_err(format!("{what} must be an object")))?;
        for key in [
            "lease_id",
            "slot_id",
            "slot_epoch",
            "generation",
            "policy_id",
            "bucket_t",
            "chunk",
            "n_valid",
            "logical_round",
        ] {
            if lmap.get(key).is_none() {
                return Err(rt_err(format!("{what} lacks {key}")));
            }
        }
        let slot_id = req_u64(lmap, "slot_id", &what)?;
        let staged = staged_by_slot.get(&slot_id).ok_or_else(|| {
            rt_err(format!("{what} slot {slot_id} unrestorable"))
        })?;
        if req_u64(lmap, "generation", &what)? != generation {
            return Err(rt_err(format!("{what} generation drifts from the pool")));
        }
        if req_u64(lmap, "n_valid", &what)? as usize != *staged {
            return Err(rt_err(format!("{what} row count drifts from its slot")));
        }
    }
    let frontier = serde_json::json!({
        "generation": generation,
        "round": max_round,
        "epoch": 0u64,
    });
    // Validate references on the existing tree; reparsing canonical bytes here
    // duplicates the full snapshot allocation and stalls GPU dispatch.
    reject_escaped_refs(&value)?;
    let canonical = canon_bytes(&value, "rl:pool-snapshot")?;
    Ok(VerifiedPool {
        canonical,
        pool_id,
        generation,
        frontier,
        terminal_set,
        tickets: Value::Array(tickets),
        captured_rows,
        wall_ids,
    })
}

/// Persist an actual `PyNativePool.snapshot` payload as a CAS pool blob.
/// Returns canonical bytes
/// (`digest`, `locator`, `frontier`, `terminal_set`, `tickets`,
/// `captured_rows`).
pub fn save_pool(root: &Path, snapshot_json: &str) -> Result<Vec<u8>, TapeError> {
    let verified = verify_pool_snapshot(snapshot_json.as_bytes())?;
    let receipt = publish_runtime_canonical(root, &verified.canonical)?;
    let receipt_value: Value = parse_json(&receipt, "pool receipt")?;
    let digest = receipt_value
        .get("digest")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    // Seed the repeat-check cache: runtime_control verifies this same digest
    // seconds later and must not re-parse 400MB of rows to learn the same
    // facts. Small clones only; the bytes stay content-addressed on disk.
    if !digest.is_empty() {
        cache_store(digest, PoolFacts::from(&verified));
    }
    let out = serde_json::json!({
        "digest": receipt_value.get("digest").cloned().unwrap_or(Value::Null),
        "locator": receipt_value.get("locator").cloned().unwrap_or(Value::Null),
        "frontier": verified.frontier,
        "terminal_set": verified.terminal_set,
        "tickets": verified.tickets,
        "captured_rows": verified.captured_rows,
    });
    canon_bytes(&out, "rl:pool-ref")
}

/// Re-derive the collection facts from a pool blob and require equality
/// with the claimed ref (callers cannot invent progress). Repeat checks of
/// one digest reuse cached facts; the file still hashes to the digest on
/// every call, so swapped bytes can never hit. Returns facts only: no
/// caller needs the 400MB canonical bytes from a check.
pub fn check_pool_ref(
    root: &Path,
    digest: &str,
    locator: &str,
    frontier: &Value,
    terminal_set: &[String],
    tickets: &Value,
    captured_rows: u64,
) -> Result<PoolFacts, TapeError> {
    let bytes = read_verified_blob(root, digest, locator)?;
    if let Some(cached) = cache_lookup(digest) {
        if cached.frontier != *frontier {
            return Err(rt_err("pool blob frontier diverges from its ref"));
        }
        if cached.terminal_set != terminal_set {
            return Err(rt_err("pool blob terminal set diverges from its ref"));
        }
        if cached.tickets != *tickets {
            return Err(rt_err("pool blob tickets diverge from its ref"));
        }
        if cached.captured_rows != captured_rows {
            return Err(rt_err("pool blob captured rows diverge from its ref"));
        }
        return Ok(cached);
    }
    let verified = verify_pool_snapshot(&bytes)?;
    if verified.frontier != *frontier {
        return Err(rt_err("pool blob frontier diverges from its ref"));
    }
    if verified.terminal_set != terminal_set {
        return Err(rt_err("pool blob terminal set diverges from its ref"));
    }
    if verified.tickets != *tickets {
        return Err(rt_err("pool blob tickets diverge from its ref"));
    }
    if verified.captured_rows != captured_rows {
        return Err(rt_err("pool blob captured rows diverge from its ref"));
    }
    let facts = PoolFacts::from(&verified);
    cache_store(digest.to_string(), facts.clone());
    Ok(facts)
}

/// Return the verified pool snapshot bytes for `PyNativePool.restore`.
/// Takes the FULL `save_pool` ref object (same shape Main stores in
/// `state.pool_ref`: digest + locator + frontier + terminal_set + tickets
/// + captured_rows); the claimed collection facts are re-derived from the
/// blob and required equal before the bytes are returned.
pub fn restore_pool_doc(root: &Path, ref_doc: &[u8]) -> Result<Vec<u8>, TapeError> {
    let value = parse_json(ref_doc, "pool ref")?;
    let pool_ref = parse_pool_ref(&value, "pool ref")?;
    let bytes = read_verified_blob(root, &pool_ref.digest, &pool_ref.locator)?;
    // Cache hit: bytes already hashed to the digest above, and the cached
    // facts came from a full verification of identical bytes. Equality
    // against the claimed ref is rechecked; only the row re-parse is
    // skipped. The file holds canonical bytes, so returning them preserves
    // the previous `verified.canonical` contract exactly.
    if let Some(cached) = cache_lookup(&pool_ref.digest) {
        if cached.frontier != pool_ref.frontier
            || cached.terminal_set != pool_ref.terminal_set
            || cached.tickets != pool_ref.tickets
            || cached.captured_rows != pool_ref.captured_rows
        {
            return Err(rt_err("pool blob diverges from its ref"));
        }
        return Ok(bytes);
    }
    let verified = verify_pool_snapshot(&bytes)?;
    if verified.frontier != pool_ref.frontier
        || verified.terminal_set != pool_ref.terminal_set
        || verified.tickets != pool_ref.tickets
        || verified.captured_rows != pool_ref.captured_rows
    {
        return Err(rt_err("pool blob diverges from its ref"));
    }
    cache_store(pool_ref.digest.clone(), PoolFacts::from(&verified));
    Ok(verified.canonical)
}

#[cfg(test)]
mod pool_facts_cache_tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static POOL_CACHE_TMP: AtomicU64 = AtomicU64::new(0);

    fn tmp_root() -> PathBuf {
        let tag = POOL_CACHE_TMP.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "rl-pool-cache-test-{}-{tag}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("test tmp");
        dir
    }

    /// Minimal verifiable snapshot: one empty game, one free lane, no
    /// leases, no poison. Exercises the cache paths without a 400MB
    /// corpus; full-row validation stays covered by the seal corpus.
    fn minimal_snapshot(pool_id: &str) -> String {
        serde_json::json!({
            "pool_id": pool_id,
            "generation": 0u64,
            "games": [{
                "wall_id": "w0",
                "ordinal": 0u64,
                "applied": [],
                "digest": "ab".repeat(32),
                "captured": 0u64,
                "sample_counters": [0u64, 0, 0, 0],
                "terminal": false,
                "completed": [],
            }],
            "slots": [{
                "id": 0u64,
                "epoch": 0u64,
                "phase": "Free",
                "bucket_t": 32u64,
                "capacity": 8u64,
                "chunk": 0u64,
                "logical_round": 0u64,
                "staged": [],
            }],
            "leases": [],
            "poison": Value::Null,
        })
        .to_string()
    }

    fn ref_parts(pool_ref: &[u8]) -> (String, String, Value, Vec<String>, Value, u64) {
        let value: Value = serde_json::from_slice(pool_ref).expect("pool ref json");
        let str_of = |key: &str| {
            value
                .get(key)
                .and_then(Value::as_str)
                .map_or(String::new(), str::to_string)
        };
        (
            str_of("digest"),
            str_of("locator"),
            value.get("frontier").cloned().map_or(Value::Null, |v| v),
            value
                .get("terminal_set")
                .and_then(Value::as_array)
                .map_or(Vec::new(), |items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                }),
            value.get("tickets").cloned().map_or(Value::Null, |v| v),
            value.get("captured_rows").and_then(Value::as_u64).map_or(0, |n| n),
        )
    }

    #[test]
    fn repeat_checks_reuse_facts_but_reject_lies_and_corruption() {
        let root = tmp_root();
        let pool_ref = save_pool(&root, &minimal_snapshot("cache-test")).expect("save");
        let (digest, locator, frontier, terminal_set, tickets, captured) = ref_parts(&pool_ref);
        assert!(!digest.is_empty());
        // First check verifies; second must hit the cache and agree.
        let first = check_pool_ref(&root, &digest, &locator, &frontier, &terminal_set, &tickets, captured)
            .expect("first check");
        let second = check_pool_ref(&root, &digest, &locator, &frontier, &terminal_set, &tickets, captured)
            .expect("cached check");
        assert_eq!(first.wall_ids, vec!["w0".to_string()]);
        assert_eq!(second.wall_ids, first.wall_ids);
        // A forged frontier against cached facts fails with the exact
        // pre-cache message (no silent acceptance on the hit path).
        let mut forged = frontier.clone();
        forged["generation"] = Value::from(99u64);
        assert!(check_pool_ref(&root, &digest, &locator, &forged, &terminal_set, &tickets, captured)
            .is_err());
        // Corrupted bytes never hit: the digest gate fires first.
        let path = root.join(&locator);
        let mut bytes = std::fs::read(&path).expect("blob bytes");
        bytes[10] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("corrupt blob");
        assert!(check_pool_ref(&root, &digest, &locator, &frontier, &terminal_set, &tickets, captured)
            .is_err());
    }
}
