//! rl/runtime_snapshot: content-addressed runtime persistence core.
//!
//! DAG: `super::{snapshot, trajectory}` + `serde_json` +
//! `hydra_feed::canon` + `sha2`. FORBIDS: PyO3, wall-clock, RNG, Torch,
//! second schema conventions (control vocabulary stays `config` +
//! `lifecycle_control`; the runtime state below is an internal versioned
//! blob, never new control vocabulary).
//!
//! Layout (all root-relative, immutable once published):
//! - `runtime/blob_<hex>.json`: any canonical JSON document (tape, event,
//!   pool snapshot, index, state, receipt) keyed by sha256 over its
//!   RFC 8785 canonical bytes. One CAS namespace; typed locators are not
//!   needed because the digest IS the identity (no ad-hoc identity law).
//! - `runtime/pointer.json` (+ `pointer.bak.json`): last-committed runtime
//!   tip mirroring the control-tip law (fsync + atomic rename, one backup).
//!   The control tip (`control/tip.json`, owned by `snapshot`) stays the
//!   commit authority; the runtime pointer must agree with it on
//!   (control digest, generation, phase) or restore fails closed.
//!
//! The validated-state schema (`validate_runtime_state`) is defined here
//! once and shared by `runtime_snapshot_control` (commit path) and restore
//! (verification path): no defaults, no guessing, every id/count actual.

use std::fs::File;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::rl::snapshot::join_root;
use crate::rl::trajectory::{DIGEST_HEX_LEN, DIGEST_PREFIX, TapeError};

/// Internal runtime-state schema version (this module's output contract).
pub const RUNTIME_STATE_VERSION: &str = "1.0.0";
/// Marker object key for the state schema.
pub const RUNTIME_STATE_SCHEMA: &str = "rl.runtime_state/v1";
/// Tape blob document marker.
pub const RUNTIME_TAPE_SCHEMA: &str = "rl.tape/v1";
/// Event blob document marker (tape's arena projection).
pub const RUNTIME_EVENT_SCHEMA: &str = "rl.tape_events/v1";
/// Pointer file carrying the last-committed runtime tip.
pub const RUNTIME_POINTER_RELPATH: &str = "runtime/pointer.json";
/// Pointer backup (one-backup law, mirrors the control tip).
pub const RUNTIME_POINTER_BAK_RELPATH: &str = "runtime/pointer.bak.json";

pub fn rt_err(detail: impl Into<String>) -> TapeError {
    TapeError(format!("runtime_snapshot rejected: {}", detail.into()))
}

/// Bare-hex sha256 of bytes.
pub fn sha_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    let sum = sha2::Sha256::digest(bytes);
    // `GenericArray` has no `LowerHex`; format each byte explicitly.
    sum.iter().map(|b| format!("{b:02x}")).collect()
}

/// Digest text (`sha256:<hex>`) of canonical bytes.
pub fn blob_digest(canonical: &[u8]) -> String {
    format!("{DIGEST_PREFIX}{}", sha_hex(canonical))
}

/// True when `text` is `sha256:<64 lowercase hex>`.
pub fn is_digest_text(text: &str) -> bool {
    let hex = match text.strip_prefix(DIGEST_PREFIX) {
        Some(hex) => hex,
        None => return false,
    };
    hex.len() == DIGEST_HEX_LEN && hex.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Canonical bytes of a JSON value through the shared JCS printer.
pub fn canon_bytes(value: &Value, record: &str) -> Result<Vec<u8>, TapeError> {
    hydra_feed::canon::canonical_bytes_value(value, record)
        .map_err(|e| rt_err(format!("JCS emit failed for {record}: {e:?}")))
}

/// Parse JSON bytes into a `Value` (fail-closed).
pub fn parse_json(doc: &[u8], what: &str) -> Result<Value, TapeError> {
    serde_json::from_slice(doc).map_err(|e| rt_err(format!("{what} is not JSON: {e}")))
}

fn fsync_file(file: &File) -> Result<(), TapeError> {
    file.sync_all()
        .map_err(|e| rt_err(format!("file fsync failed: {e}")))
}

fn fsync_dir(dir: &Path) -> Result<(), TapeError> {
    #[cfg(unix)]
    {
        let handle = File::open(dir).map_err(|e| rt_err(format!("dir open failed: {e}")))?;
        handle
            .sync_all()
            .map_err(|e| rt_err(format!("dir fsync failed: {e}")))?;
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
    Ok(())
}

/// Atomic publish: temp + fsync + rename + dir fsync (mirrors the snapshot
/// tip law; local copy so this module never edits `snapshot.rs`).
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), TapeError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| rt_err(format!("dir create failed: {e}")))?;
        }
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(|e| rt_err(format!("temp write failed: {e}")))?;
    let handle = File::open(&tmp).map_err(|e| rt_err(format!("temp open failed: {e}")))?;
    fsync_file(&handle)?;
    drop(handle);
    std::fs::rename(&tmp, path).map_err(|e| rt_err(format!("atomic rename failed: {e}")))?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fsync_dir(parent)?;
        }
    }
    Ok(())
}

/// Fsync an already-written immutable file plus its directory.
pub fn commit_path(path: &Path) -> Result<(), TapeError> {
    let handle = File::open(path).map_err(|e| rt_err(format!("blob open failed: {e}")))?;
    fsync_file(&handle)?;
    drop(handle);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fsync_dir(parent)?;
        }
    }
    Ok(())
}

/// Reject unsafe locators anywhere a `*locator*`/`*path` key carries a
/// string: absolute paths, parent escapes, backslashes, empties. Returns
/// the count of locator strings scanned.
pub fn reject_escaped_refs(value: &Value) -> Result<usize, TapeError> {
    fn walk(value: &Value, key: Option<&str>, seen: &mut usize) -> Result<(), TapeError> {
        match value {
            Value::Object(map) => {
                for (k, v) in map {
                    walk(v, Some(k), seen)?;
                }
                Ok(())
            }
            Value::Array(items) => {
                for item in items {
                    walk(item, key, seen)?;
                }
                Ok(())
            }
            Value::String(text) => {
                let is_locator = key.is_some_and(|k| {
                    k == "locator" || k.ends_with("_locator") || k == "path" || k.ends_with("_path")
                });
                if is_locator {
                    *seen += 1;
                    if text.is_empty() || text.len() > 4096 || text.contains('\\') {
                        return Err(rt_err(format!("ref escapes the run root: {text:?}")));
                    }
                    let rel = Path::new(text);
                    if rel.is_absolute() {
                        return Err(rt_err(format!("ref must be root-relative: {text:?}")));
                    }
                    for component in rel.components() {
                        if !matches!(component, std::path::Component::Normal(_)) {
                            return Err(rt_err(format!("ref escapes the run root: {text:?}")));
                        }
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
    let mut seen = 0usize;
    walk(value, None, &mut seen)?;
    Ok(seen)
}

/// Content-addressed publish of one canonical JSON document: parse,
/// re-emit canonical (idempotent for canonical inputs), scan refs,
/// digest, atomic immutable publication. Existing files with identical
/// bytes are idempotent; differing bytes under the same digest are a
/// conflict (never overwritten). Returns canonical receipt bytes
/// (`digest`, `locator`).
pub fn write_runtime_blob(root: &Path, document: &[u8]) -> Result<Vec<u8>, TapeError> {
    if document.is_empty() || document.len() > (1usize << 31) {
        return Err(rt_err("blob document has a bad length"));
    }
    let value = parse_json(document, "blob document")?;
    write_runtime_value(root, &value)
}

/// Native structured documents are checked and canonicalized exactly once;
/// untrusted byte callers still parse first, so identity/ref guards stay intact.
pub(super) fn write_runtime_value(root: &Path, value: &Value) -> Result<Vec<u8>, TapeError> {
    if !value.is_object() {
        return Err(rt_err("blob document must be a JSON object"));
    }
    reject_escaped_refs(value)?;
    let canonical = canon_bytes(value, "rl:runtime-blob")?;
    publish_runtime_canonical(root, &canonical)
}

/// Only native emitters that already checked object framing and references
/// may publish canonical bytes; untrusted callers must parse and validate first.
pub(super) fn publish_runtime_canonical(root: &Path, canonical: &[u8]) -> Result<Vec<u8>, TapeError> {
    if canonical.is_empty() || canonical.len() > (1usize << 31) {
        return Err(rt_err("blob document has a bad length"));
    }
    let digest = blob_digest(canonical);
    let hex = digest
        .strip_prefix(DIGEST_PREFIX)
        .unwrap_or(&digest)
        .to_string();
    let locator = format!("runtime/blob_{hex}.json");
    let path = join_root(root, &locator)?;
    if path.exists() {
        let existing =
            std::fs::read(&path).map_err(|e| rt_err(format!("blob re-read failed: {e}")))?;
        if existing != canonical {
            return Err(rt_err(format!(
                "blob {locator:?} conflicts with different bytes under one digest"
            )));
        }
    } else {
        atomic_write(&path, canonical)?;
    }
    commit_path(&path)?;
    let mut receipt = Map::with_capacity(2);
    receipt.insert("digest".to_string(), Value::String(digest));
    receipt.insert("locator".to_string(), Value::String(locator));
    canon_bytes(&Value::Object(receipt), "rl:runtime-receipt")
}

/// Read one CAS blob and verify its bytes hash to the expected digest.
/// Returns the canonical bytes.
pub fn read_verified_blob(root: &Path, digest: &str, locator: &str) -> Result<Vec<u8>, TapeError> {
    if !is_digest_text(digest) {
        return Err(rt_err(format!("blob digest has a bad shape: {digest:?}")));
    }
    let path = join_root(root, locator)?;
    let bytes = std::fs::read(&path).map_err(|e| rt_err(format!("blob unreadable: {e}")))?;
    let actual = blob_digest(&bytes);
    if actual != digest {
        return Err(rt_err(format!("blob {locator:?} diverges from its digest")));
    }
    Ok(bytes)
}

/// Parse a `{digest, locator}` CAS ref (closed, exactly two keys).
#[derive(Debug, Clone)]
pub struct CasRef {
    pub digest: String,
    pub locator: String,
}

pub fn parse_cas_ref(value: &Value, name: &str) -> Result<CasRef, TapeError> {
    let map = value
        .as_object()
        .ok_or_else(|| rt_err(format!("{name} must be an object")))?;
    if map.len() != 2 || !map.contains_key("digest") || !map.contains_key("locator") {
        return Err(rt_err(format!("{name} must carry exactly digest+locator")));
    }
    let digest = map
        .get("digest")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err(format!("{name} lacks digest")))?;
    if !is_digest_text(digest) {
        return Err(rt_err(format!("{name} digest has a bad shape")));
    }
    let locator = map
        .get("locator")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err(format!("{name} lacks locator")))?;
    join_root(Path::new("."), locator)
        .map_err(|e| rt_err(format!("{name} locator rejected: {e}")))?;
    Ok(CasRef {
        digest: digest.to_string(),
        locator: locator.to_string(),
    })
}

/// One collector side of the runtime state: `spec` + `pool_ref` always
/// actual; the sealed unit (receipt + order + index_ref + tape_ref) is
/// all-present or all-null (a cold/initial collector carries the actual
/// pool only, never a fabricated artifact).
#[derive(Debug, Clone)]
pub struct StateSide {
    pub spec: Value,
    pub receipt: Option<Value>,
    pub order: Option<Value>,
    pub index_ref: Option<CasRef>,
    pub tape_ref: Option<TapeRef>,
    pub pool_ref: PoolRef,
}

/// Pool ref as returned by `save_pool` (digest + locator + derived
/// collection facts, all verified against the pool blob on use).
#[derive(Debug, Clone)]
pub struct PoolRef {
    pub digest: String,
    pub locator: String,
    pub frontier: Value,
    pub terminal_set: Vec<String>,
    pub tickets: Value,
    pub captured_rows: u64,
}

/// Validated runtime state: every id/count actual, no defaults.
#[derive(Debug, Clone)]
pub struct ValidatedState {
    pub generation: u64,
    pub batch_cursor: u64,
    pub completed_updates: u64,
    pub behavior_id: String,
    pub generation_initial_id: String,
    pub publication_phase: String,
    pub stopped: bool,
    pub current: StateSide,
    pub future: Option<StateSide>,
}

fn req_u64(map: &Map<String, Value>, key: &str, what: &str) -> Result<u64, TapeError> {
    map.get(key).and_then(Value::as_u64).ok_or_else(|| {
        rt_err(format!("{what} lacks integer {key}"))
    })
}

fn req_digest(map: &Map<String, Value>, key: &str, what: &str) -> Result<String, TapeError> {
    let text = map
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err(format!("{what} lacks {key}")))?;
    if !is_digest_text(text) {
        return Err(rt_err(format!("{what}.{key} has a bad digest shape")));
    }
    Ok(text.to_string())
}

/// Validate the sealed-artifact receipt shape (ArtifactStorage seal keys).
fn parse_seal_receipt(value: &Value, name: &str) -> Result<(String, u64), TapeError> {
    let map = value
        .as_object()
        .ok_or_else(|| rt_err(format!("{name} must be an object")))?;
    if map.len() != 4
        || !["artifact_digest", "row_count", "compressed_bytes", "uncompressed_bytes"]
            .iter()
            .all(|k| map.contains_key(*k))
    {
        return Err(rt_err(format!("{name} must carry exactly the seal receipt keys")));
    }
    let digest = req_digest(map, "artifact_digest", name)?;
    let rows = req_u64(map, "row_count", name)?;
    if rows == 0 {
        return Err(rt_err(format!("{name} row_count must be nonzero")));
    }
    req_u64(map, "compressed_bytes", name)?;
    req_u64(map, "uncompressed_bytes", name)?;
    Ok((digest, rows))
}

/// Validate the frozen order descriptor shape (replay-index keys) and
/// return (artifact_digest, order_digest, row_count, batch_count).
fn parse_order_desc(value: &Value, name: &str) -> Result<(String, String, u64, usize), TapeError> {
    let map = value
        .as_object()
        .ok_or_else(|| rt_err(format!("{name} must be an object")))?;
    if map.len() != 4
        || !["artifact_digest", "order_digest", "row_count", "batches"]
            .iter()
            .all(|k| map.contains_key(*k))
    {
        return Err(rt_err(format!("{name} must carry exactly the order descriptor keys")));
    }
    let artifact = req_digest(map, "artifact_digest", name)?;
    let order = req_digest(map, "order_digest", name)?;
    let rows = req_u64(map, "row_count", name)?;
    let batches = map
        .get("batches")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err(format!("{name} lacks batches array")))?;
    if batches.is_empty() {
        return Err(rt_err(format!("{name} batches must be nonempty")));
    }
    let mut counted: u64 = 0;
    for (i, batch) in batches.iter().enumerate() {
        let obj = batch
            .as_object()
            .ok_or_else(|| rt_err(format!("{name} batches[{i}] must be an object")))?;
        if obj.len() != 2 || !obj.contains_key("t") || !obj.contains_key("count") {
            return Err(rt_err(format!("{name} batches[{i}] must carry exactly t+count")));
        }
        let t = obj.get("t").and_then(Value::as_u64).ok_or_else(|| {
            rt_err(format!("{name} batches[{i}].t must be an integer"))
        })?;
        if ![32, 64, 128, 256].contains(&t) {
            return Err(rt_err(format!("{name} batches[{i}].t={t} outside 32/64/128/256")));
        }
        let count = obj.get("count").and_then(Value::as_u64).ok_or_else(|| {
            rt_err(format!("{name} batches[{i}].count must be an integer"))
        })?;
        if count == 0 || count > 4096 {
            return Err(rt_err(format!(
                "{name} batches[{i}].count={count} outside 1..=4096"
            )));
        }
        counted = counted
            .checked_add(count)
            .ok_or_else(|| rt_err(format!("{name} batch counts overflow")))?;
    }
    if counted != rows {
        return Err(rt_err(format!(
            "{name} batch counts sum {counted} != row_count {rows}"
        )));
    }
    Ok((artifact, order, rows, batches.len()))
}

/// Validate the collector spec minimum shape (RlLifecycle-owned; we check
/// identity linkage + presence, never its internals).
fn parse_collector_spec(
    value: &Value,
    name: &str,
    want_generation: u64,
    want_policy: &str,
) -> Result<(), TapeError> {
    let map = value
        .as_object()
        .ok_or_else(|| rt_err(format!("{name} must be an object")))?;
    match map.get("lifecycle_version").and_then(Value::as_str) {
        Some("1.0.0") => {}
        other => {
            return Err(rt_err(format!(
                "{name} lifecycle_version must be '1.0.0', got {other:?}"
            )));
        }
    }
    let generation = req_u64(map, "generation", name)?;
    if generation != want_generation {
        return Err(rt_err(format!(
            "{name} generation {generation} != state side {want_generation}"
        )));
    }
    let policy = map
        .get("policy_checkpoint_id")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err(format!("{name} lacks policy_checkpoint_id")))?;
    if policy != want_policy {
        return Err(rt_err(format!("{name} policy is not the side behavior id")));
    }
    for key in ["profile", "context", "header", "shuffle_key"] {
        if !map.get(key).is_some_and(|v| v.is_object()) {
            return Err(rt_err(format!("{name} lacks object {key}")));
        }
    }
    let ledger = map
        .get("source_ledger")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err(format!("{name} lacks source_ledger array")))?;
    if ledger.is_empty() || ledger.iter().any(|w| !w.as_str().is_some_and(|s| !s.is_empty())) {
        return Err(rt_err(format!("{name} source_ledger must hold nonempty wall ids")));
    }
    for key in ["artifact_locator", "metas_locator"] {
        if !map.get(key).and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
            return Err(rt_err(format!("{name} lacks {key}")));
        }
    }
    Ok(())
}

/// Validate the `save_pool` ref shape against its closed keys.
pub fn parse_pool_ref(value: &Value, name: &str) -> Result<PoolRef, TapeError> {
    let map = value
        .as_object()
        .ok_or_else(|| rt_err(format!("{name} must be an object")))?;
    if map.len() != 6
        || !["digest", "locator", "frontier", "terminal_set", "tickets", "captured_rows"]
            .iter()
            .all(|k| map.contains_key(*k))
    {
        return Err(rt_err(format!(
            "{name} must carry exactly digest+locator+frontier+terminal_set+tickets+captured_rows"
        )));
    }
    let digest = req_digest(map, "digest", name)?;
    let locator = map
        .get("locator")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err(format!("{name} lacks locator")))?;
    join_root(Path::new("."), locator)
        .map_err(|e| rt_err(format!("{name} locator rejected: {e}")))?;
    let frontier = map.get("frontier").cloned().unwrap_or(Value::Null);
    let fmap = frontier
        .as_object()
        .ok_or_else(|| rt_err(format!("{name} frontier must be an object")))?;
    for key in ["generation", "round", "epoch"] {
        if fmap.get(key).and_then(Value::as_u64).is_none() {
            return Err(rt_err(format!("{name} frontier lacks integer {key}")));
        }
    }
    let terminal_set = map
        .get("terminal_set")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err(format!("{name} lacks terminal_set array")))?
        .iter()
        .map(|w| {
            w.as_str()
                .map(str::to_string)
                .ok_or_else(|| rt_err(format!("{name} terminal_set holds non-strings")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tickets = map.get("tickets").cloned().unwrap_or(Value::Null);
    if !tickets.is_array() {
        return Err(rt_err(format!("{name} tickets must be an array")));
    }
    let captured_rows = req_u64(map, "captured_rows", name)?;
    Ok(PoolRef {
        digest,
        locator: locator.to_string(),
        frontier,
        terminal_set,
        tickets,
        captured_rows,
    })
}
fn opt_cas_ref(map: &Map<String, Value>, key: &str, what: &str) -> Result<Option<CasRef>, TapeError> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => parse_cas_ref(value, &format!("{what}.{key}")).map(Some),
    }
}
/// Tape ref: the exact fixed shape `save_tape` returns and every consumer
/// parses (single owner: digest + locator + sealed counts). The counts
/// bind the tape to its sealed artifact scope at every use.
#[derive(Debug, Clone)]
pub struct TapeRef {
    pub digest: String,
    pub locator: String,
    pub row_count: u64,
    pub game_count: u64,
}

/// Validate the tape ref shape (closed: digest+locator+row_count+game_count).
pub fn parse_tape_ref(value: &Value, name: &str) -> Result<TapeRef, TapeError> {
    let map = value
        .as_object()
        .ok_or_else(|| rt_err(format!("{name} must be an object")))?;
    if map.len() != 4
        || !["digest", "locator", "row_count", "game_count"]
            .iter()
            .all(|k| map.contains_key(*k))
    {
        return Err(rt_err(format!(
            "{name} must carry exactly digest+locator+row_count+game_count"
        )));
    }
    let digest = req_digest(map, "digest", name)?;
    let locator = map
        .get("locator")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err(format!("{name} lacks locator")))?;
    join_root(Path::new("."), locator)
        .map_err(|e| rt_err(format!("{name} locator rejected: {e}")))?;
    let row_count = req_u64(map, "row_count", name)?;
    let game_count = req_u64(map, "game_count", name)?;
    Ok(TapeRef {
        digest,
        locator: locator.to_string(),
        row_count,
        game_count,
    })
}

fn opt_tape_ref(map: &Map<String, Value>, key: &str, what: &str) -> Result<Option<TapeRef>, TapeError> {
    match map.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => parse_tape_ref(value, &format!("{what}.{key}")).map(Some),
    }
}

fn parse_side(
    value: &Value,
    name: &str,
    want_generation: u64,
    want_policy: &str,
) -> Result<StateSide, TapeError> {
    let map = value
        .as_object()
        .ok_or_else(|| rt_err(format!("{name} must be an object")))?;
    if map.len() != 6
        || !["spec", "receipt", "order", "index_ref", "tape_ref", "pool_ref"]
            .iter()
            .all(|k| map.contains_key(*k))
    {
        return Err(rt_err(format!(
            "{name} must carry exactly spec+receipt+order+index_ref+tape_ref+pool_ref"
        )));
    }
    let spec = map.get("spec").cloned().unwrap_or(Value::Null);
    parse_collector_spec(&spec, &format!("{name}.spec"), want_generation, want_policy)?;
    let receipt = match map.get("receipt") {
        None => return Err(rt_err(format!("{name} lacks receipt slot"))),
        Some(Value::Null) => None,
        Some(v) => {
            parse_seal_receipt(v, &format!("{name}.receipt"))?;
            Some(v.clone())
        }
    };
    let order = match map.get("order") {
        None => return Err(rt_err(format!("{name} lacks order slot"))),
        Some(Value::Null) => None,
        Some(v) => {
            parse_order_desc(v, &format!("{name}.order"))?;
            Some(v.clone())
        }
    };
    if map.get("index_ref").is_none() || map.get("tape_ref").is_none() {
        return Err(rt_err(format!("{name} lacks sealed binding slots")));
    }
    let index_ref = opt_cas_ref(map, "index_ref", name)?;
    let tape_ref = opt_tape_ref(map, "tape_ref", name)?;
    // Sealed unit is atomic: receipt + order + index_ref + tape_ref are
    // all present or all null. A cold collector carries the actual pool
    // only; a sealed receipt without frozen bindings (or vice versa) is
    // a partial fabrication and fails closed here, not at publish.
    let sealed_count = [
        receipt.is_some(),
        order.is_some(),
        index_ref.is_some(),
        tape_ref.is_some(),
    ]
    .iter()
    .filter(|present| **present)
    .count();
    if sealed_count != 0 && sealed_count != 4 {
        return Err(rt_err(format!(
            "{name} sealed unit is partial (receipt+order+index_ref+tape_ref must agree)"
        )));
    }
    if let (Some(r), Some(o)) = (receipt.as_ref(), order.as_ref()) {
        let (r_digest, r_rows) = parse_seal_receipt(r, &format!("{name}.receipt"))?;
        let (o_artifact, _, o_rows, _) = parse_order_desc(o, &format!("{name}.order"))?;
        if o_artifact != r_digest || o_rows != r_rows {
            return Err(rt_err(format!(
                "{name} receipt and order disagree on artifact/rows"
            )));
        }
    }
    let pool_ref = parse_pool_ref(
        map.get("pool_ref").unwrap_or(&Value::Null),
        &format!("{name}.pool_ref"),
    )?;
    Ok(StateSide {
        spec,
        receipt,
        order,
        index_ref,
        tape_ref,
        pool_ref,
    })
}

/// Validate the full runtime state document (closed schema, all linkage
/// actual). Returns the validated state for the commit/restore paths.
pub fn validate_runtime_state(doc: &[u8]) -> Result<ValidatedState, TapeError> {
    let value = parse_json(doc, "runtime state")?;
    let map = value
        .as_object()
        .ok_or_else(|| rt_err("runtime state must be an object"))?;
    if map.len() != 10
        || ![
            "schema",
            "generation",
            "batch_cursor",
            "completed_updates",
            "behavior_id",
            "generation_initial_id",
            "current",
            "future",
            "publication_phase",
            "stopped",
        ]
        .iter()
        .all(|k| map.contains_key(*k))
    {
        return Err(rt_err(
            "runtime state must carry exactly schema+generation+batch_cursor+completed_updates+behavior_id+generation_initial_id+current+future+publication_phase+stopped",
        ));
    }
    match map.get("schema").and_then(Value::as_str) {
        Some(RUNTIME_STATE_SCHEMA) => {}
        other => {
            return Err(rt_err(format!(
                "runtime state schema must be {RUNTIME_STATE_SCHEMA:?}, got {other:?}"
            )));
        }
    }
    let generation = req_u64(map, "generation", "state")?;
    let batch_cursor = req_u64(map, "batch_cursor", "state")?;
    let completed_updates = req_u64(map, "completed_updates", "state")?;
    let behavior_id = req_digest(map, "behavior_id", "state")?;
    let generation_initial_id = req_digest(map, "generation_initial_id", "state")?;
    let phase = map
        .get("publication_phase")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err("state lacks publication_phase"))?;
    if ![
        "not_captured",
        "captured_at_update1",
        "checkpoint_durable",
        "actor_activated",
    ]
    .contains(&phase)
    {
        return Err(rt_err(format!("state publication_phase {phase:?} unknown")));
    }
    let stopped = map
        .get("stopped")
        .and_then(Value::as_bool)
        .ok_or_else(|| rt_err("state lacks boolean stopped"))?;
    let current = parse_side(
        map.get("current").unwrap_or(&Value::Null),
        "state.current",
        generation,
        &behavior_id,
    )?;
    if current.receipt.is_none() {
        // Cold collector: no sealed artifact exists yet, so no optimizer
        // progress can exist either (cursors are 0 by construction).
        if batch_cursor != 0 || completed_updates != 0 {
            return Err(rt_err(
                "unsealed current cannot carry optimizer progress",
            ));
        }
    } else if let Some(order) = current.order.as_ref() {
        let (_, _, _, batches) = parse_order_desc(order, "state.current.order")?;
        if batch_cursor > batches as u64 {
            return Err(rt_err(format!(
                "state batch_cursor {batch_cursor} exceeds {} order batches",
                batches
            )));
        }
    }
    let future = match map.get("future").unwrap_or(&Value::Null) {
        Value::Null => None,
        value => {
            let fmap = value
                .as_object()
                .ok_or_else(|| rt_err("state.future must be an object or null"))?;
            let fgen = fmap
                .get("spec")
                .and_then(|s| s.as_object())
                .and_then(|s| s.get("generation"))
                .and_then(Value::as_u64)
                .ok_or_else(|| rt_err("state.future.spec lacks generation"))?;
            if fgen != generation && fgen != generation + 1 {
                return Err(rt_err(format!(
                    "state.future generation {fgen} must be current or next"
                )));
            }
            let fpolicy = fmap
                .get("spec")
                .and_then(|s| s.as_object())
                .and_then(|s| s.get("policy_checkpoint_id"))
                .and_then(Value::as_str)
                .ok_or_else(|| rt_err("state.future.spec lacks policy_checkpoint_id"))?;
            if !is_digest_text(fpolicy) {
                return Err(rt_err("state.future policy has a bad digest shape"));
            }
            Some(parse_side(value, "state.future", fgen, fpolicy)?)
        }
    };
    Ok(ValidatedState {
        generation,
        batch_cursor,
        completed_updates,
        behavior_id,
        generation_initial_id,
        publication_phase: phase.to_string(),
        stopped,
        current,
        future,
    })
}

/// Read the runtime pointer (last-committed tip with one-backup fallback).
/// Returns (state_digest, state_locator, control_digest, generation, phase,
/// learner_receipt_locator, actor_receipt_locator).
#[allow(clippy::type_complexity)]
pub fn read_runtime_pointer(
    root: &Path,
) -> Result<(String, String, String, u64, String, String, String), TapeError> {
    let tip_path = root.join(RUNTIME_POINTER_RELPATH);
    let bak_path = root.join(RUNTIME_POINTER_BAK_RELPATH);
    let mut last = rt_err("no committed runtime pointer");
    for candidate in [&tip_path, &bak_path] {
        match read_one_pointer(candidate) {
            Ok(parts) => return Ok(parts),
            Err(e) => last = e,
        }
    }
    Err(last)
}

#[allow(clippy::type_complexity)]
fn read_one_pointer(
    path: &Path,
) -> Result<(String, String, String, u64, String, String, String), TapeError> {
    let bytes = std::fs::read(path).map_err(|e| rt_err(format!("runtime pointer unreadable: {e}")))?;
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|e| rt_err(format!("pointer not JSON: {e}")))?;
    let map = value
        .as_object()
        .ok_or_else(|| rt_err("runtime pointer must be an object"))?;
    let str_of = |key: &str| {
        map.get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| rt_err(format!("pointer lacks {key}")))
    };
    let state_digest = str_of("state_digest")?;
    let state_locator = str_of("state_locator")?;
    let control_digest = str_of("control_digest")?;
    let generation = map
        .get("generation")
        .and_then(Value::as_u64)
        .ok_or_else(|| rt_err("pointer lacks generation"))?;
    let phase = str_of("phase")?;
    let learner_locator = str_of("learner_receipt_locator")?;
    let actor_locator = str_of("actor_receipt_locator")?;
    if !is_digest_text(&state_digest) || !is_digest_text(&control_digest) {
        return Err(rt_err("pointer carries bad digest shapes"));
    }
    Ok((
        state_digest,
        state_locator,
        control_digest,
        generation,
        phase,
        learner_locator,
        actor_locator,
    ))
}

/// Write the runtime pointer atomically (backup-then-replace, mirrors the
/// control tip law). Returns the canonical pointer bytes.
#[allow(clippy::too_many_arguments)]
pub fn write_runtime_pointer(
    root: &Path,
    state_digest: &str,
    state_locator: &str,
    control_digest: &str,
    control_locator: &str,
    generation: u64,
    phase: &str,
    learner_receipt_locator: &str,
    actor_receipt_locator: &str,
) -> Result<Vec<u8>, TapeError> {
    let doc = serde_json::json!({
        "schema_version": "1.0.0",
        "state_digest": state_digest,
        "state_locator": state_locator,
        "control_digest": control_digest,
        "control_locator": control_locator,
        "generation": generation,
        "phase": phase,
        "learner_receipt_locator": learner_receipt_locator,
        "actor_receipt_locator": actor_receipt_locator,
    });
    let bytes = canon_bytes(&doc, "rl:runtime-pointer")?;
    let tip_path = root.join(RUNTIME_POINTER_RELPATH);
    if tip_path.exists() {
        let bak = root.join(RUNTIME_POINTER_BAK_RELPATH);
        let current =
            std::fs::read(&tip_path).map_err(|e| rt_err(format!("pointer backup failed: {e}")))?;
        atomic_write(&bak, &current)?;
    }
    atomic_write(&tip_path, &bytes)?;
    Ok(bytes)
}

/// Native `checkpoint_sections`: project the generation bindings'
/// deterministic skeletons onto the true completed cursors for one
/// ordinary snapshot (Main-owned Torch payload shape; native owns the
/// counters). `batch_cursor` is local within `max_updates` (next batch
/// index); `completed_updates` is GLOBAL cumulative. Micro accumulations
/// are always quiesced at optimizer boundaries, so `micro_offset` is 0.
/// Returns canonical `{sampler_state, training_state}` bytes. No async
/// collector metadata ever enters either section.
pub fn checkpoint_sections(
    bindings_json: &[u8],
    batch_cursor: u64,
    completed_updates: u64,
) -> Result<Vec<u8>, TapeError> {
    let bindings: Value =
        serde_json::from_slice(bindings_json).map_err(|e| rt_err(format!("bindings not JSON: {e}")))?;
    let bind = bindings
        .as_object()
        .ok_or_else(|| rt_err("bindings must be an object"))?;
    match bind.get("lifecycle_version").and_then(Value::as_str) {
        Some("1.0.0") => {}
        other => {
            return Err(rt_err(format!(
                "bindings version must be '1.0.0', got {other:?}"
            )));
        }
    }
    let generation = bind
        .get("generation")
        .and_then(Value::as_u64)
        .ok_or_else(|| rt_err("bindings lack generation"))?;
    let max_updates = bind
        .get("max_updates")
        .and_then(Value::as_u64)
        .ok_or_else(|| rt_err("bindings lack max_updates"))?;
    if batch_cursor > max_updates {
        return Err(rt_err(format!(
            "batch_cursor {batch_cursor} exceeds max_updates {max_updates}"
        )));
    }
    let row_count = bind
        .get("row_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| rt_err("bindings lack row_count"))?;
    let artifact_digest = bind
        .get("artifact_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err("bindings lack artifact_digest"))?;
    if !is_digest_text(artifact_digest) {
        return Err(rt_err("bindings artifact_digest has a bad shape"));
    }
    let sampler = bind
        .get("sampler_skeleton")
        .and_then(|v| v.as_object())
        .ok_or_else(|| rt_err("bindings lack sampler_skeleton"))?;
    let training = bind
        .get("training_skeleton")
        .and_then(|v| v.as_object())
        .ok_or_else(|| rt_err("bindings lack training_skeleton"))?;
    // Immutable source/order linkage: the skeleton must name this
    // generation's sealed artifact + row count + pipeline; the batch
    // views must sum to the sealed count (same law as `prepare_generation`,
    // re-checked here so a swapped bindings blob fails closed).
    let source_hash = sampler
        .get("source_hash")
        .and_then(Value::as_str)
        .ok_or_else(|| rt_err("sampler skeleton lacks source_hash"))?;
    if source_hash != artifact_digest {
        return Err(rt_err("sampler source_hash is not the bound artifact"));
    }
    if sampler.get("row_count").and_then(Value::as_u64) != Some(row_count) {
        return Err(rt_err("sampler row_count is not the bound count"));
    }
    if sampler.get("generation").and_then(Value::as_u64) != Some(generation) {
        return Err(rt_err("sampler generation drifts from the bindings"));
    }
    let views = sampler
        .get("ordered_batches")
        .and_then(Value::as_array)
        .ok_or_else(|| rt_err("sampler lacks ordered_batches"))?;
    if views.len() as u64 != max_updates {
        return Err(rt_err("sampler batch views disagree with max_updates"));
    }
    if training.get("generation").and_then(Value::as_u64) != Some(generation) {
        return Err(rt_err("training generation drifts from the bindings"));
    }
    let mut sampler_state = sampler.clone();
    sampler_state.insert(
        "optimizer_cursor".to_string(),
        Value::from(batch_cursor),
    );
    sampler_state.insert("micro_offset".to_string(), Value::from(0u64));
    let mut training_state = training.clone();
    training_state.insert(
        "completed_updates".to_string(),
        Value::from(completed_updates),
    );
    let out = serde_json::json!({
        "sampler_state": Value::Object(sampler_state),
        "training_state": Value::Object(training_state),
    });
    canon_bytes(&out, "rl:checkpoint-sections")
}

/// Locate a file under the root (re-export for the FFI layer).
pub fn locate_blob(root: &Path, locator: &str) -> Result<PathBuf, TapeError> {
    join_root(root, locator)
}
