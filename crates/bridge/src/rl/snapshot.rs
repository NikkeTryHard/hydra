//! Control/native-snapshot publication: atomic compressed-artifact commit,
//! ordinary-checkpoint receipts, closed canonical control envelopes, and
//! last-committed-tip authority.
//!
//! Layering (plan section 7): the Torch ordinary checkpoint payload is
//! Main-owned — this module NEVER reads Torch bytes. It validates the
//! structural manifest (exact 19 fields, FULL-JCS manifest id with no
//! exclusion), streams the landed file for transport SHA + length, and
//! binds the receipt into the control transaction. Native async progress
//! (tape/frontier/cursors/phases) lives ONLY in the control envelope, never
//! in ordinary policy identity (no seed/hash cycles: control digests never
//! feed RNG, shuffle, or wall derivation).
//!
//! Envelope conformance reuses the registered schema vocabulary
//! (`config::RL_CONTROL_FIELDS`, ticket/nested vocabularies, publication and
//! ticket phases) plus `check_envelope_keys` and
//! `identity_of_payload_without_digest` — no second schema convention.
//!
//! Durable sequence: immutable blobs (`commit_blob` fsync) -> ordinary
//! receipt (verified at `publish_control` against the on-disk locator, SHA,
//! length, and manifest id) -> canonical control record -> fsync + atomic
//! tip (`tip.json` with one-backup `tip.bak.json`). A crash before the tip
//! rename leaves the previous tip authoritative; unreferenced files are
//! orphans, never committed. Stale/corrupt tips are rejected, never
//! repaired.

use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde_json::{Map, Value};

use crate::rl::artifact::CHECKPOINT_MANIFEST_KEYS;
use crate::rl::config::{
    RL_CONTROL_ARTIFACT, RL_CONTROL_FIELDS, RL_CONTROL_FREE_MAPS, RL_NATIVE_SNAPSHOT_ARTIFACT,
    RL_NATIVE_SNAPSHOT_FIELDS, RL_PUBLICATION_PHASES, RL_TICKET_PHASES, check_envelope_keys,
    identity_of_payload_without_digest,
};
use crate::rl::trajectory::{DIGEST_PREFIX, TapeError, digest_text, hex_of, parse_digest_text};

/// Supported ordinary checkpoint versions (runtime `checkpoint.py:33`).
pub const CHECKPOINT_VERSIONS: [&str; 1] = ["1.0.0"];
/// Control envelope schema version (registered `1.0.0`, exact).
pub const CONTROL_SCHEMA_VERSION: &str = "1.0.0";
/// File read chunk for streaming transport hashes.
pub const RECEIPT_CHUNK: usize = 1 << 20;

fn snap_err(detail: String) -> TapeError {
    TapeError(detail)
}

/// Root-relative locator guard: rejects absolute paths, parent escapes,
/// empty locators, and backslash separators. Returns the joined path.
/// Confidential absolute source paths can never enter a receipt or tip.
pub fn join_root(root: &Path, locator: &str) -> Result<PathBuf, TapeError> {
    if locator.is_empty() || locator.len() > 4096 {
        return Err(snap_err("locator has bad length".to_string()));
    }
    if locator.contains('\\') {
        return Err(snap_err("locator must use '/' separators".to_string()));
    }
    let rel = Path::new(locator);
    if rel.is_absolute() {
        return Err(snap_err(format!("locator must be root-relative: {locator:?}")));
    }
    for component in rel.components() {
        match component {
            Component::Normal(_) => {}
            _ => {
                return Err(snap_err(format!(
                    "locator escapes the run root: {locator:?}"
                )));
            }
        }
    }
    Ok(root.join(rel))
}

fn fsync_file(file: &File) -> Result<(), TapeError> {
    file.sync_all()
        .map_err(|e| snap_err(format!("file fsync failed: {e}")))
}

fn fsync_dir(dir: &Path) -> Result<(), TapeError> {
    #[cfg(unix)]
    {
        let handle =
            File::open(dir).map_err(|e| snap_err(format!("dir open failed: {e}")))?;
        handle
            .sync_all()
            .map_err(|e| snap_err(format!("dir fsync failed: {e}")))?;
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
    Ok(())
}

/// Atomic publish: temp + fsync + rename + dir fsync. Readers see the old
/// or the new file, never a partial one.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), TapeError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| snap_err(format!("dir create failed: {e}")))?;
        }
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(|e| snap_err(format!("temp write failed: {e}")))?;
    let handle =
        File::open(&tmp).map_err(|e| snap_err(format!("temp open failed: {e}")))?;
    fsync_file(&handle)?;
    drop(handle);
    std::fs::rename(&tmp, path).map_err(|e| snap_err(format!("atomic rename failed: {e}")))?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fsync_dir(parent)?;
        }
    }
    Ok(())
}

/// Durable blob step: fsync an already-written immutable file plus its
/// directory (no rename, no mutation — the seal wrote the bytes).
pub fn commit_blob(path: &Path) -> Result<(), TapeError> {
    let handle =
        File::open(path).map_err(|e| snap_err(format!("blob open failed: {e}")))?;
    fsync_file(&handle)?;
    drop(handle);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fsync_dir(parent)?;
        }
    }
    Ok(())
}

/// Stream a file into `(transport_sha256_text, byte_len)` without holding
/// the bytes (Torch payloads stay on disk; only the digest crosses).
fn stream_file_digest(path: &Path) -> Result<(String, u64), TapeError> {
    let file =
        File::open(path).map_err(|e| snap_err(format!("receipt file unreadable: {e}")))?;
    let mut reader = std::io::BufReader::with_capacity(RECEIPT_CHUNK, file);
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    let mut len: u64 = 0;
    let mut chunk = vec![0u8; RECEIPT_CHUNK];
    loop {
        let n = reader
            .read(&mut chunk)
            .map_err(|e| snap_err(format!("receipt stream failed: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
        len = len.saturating_add(n as u64);
    }
    Ok((digest_text(&hasher.finalize().into()), len))
}

/// Cold receipt for one landed ordinary checkpoint: enforces the exact 19
/// manifest fields, the supported version, the dataset-XOR-rollout source
/// rule, and digest shapes; hashes the FULL JCS manifest (no digest-field
/// exclusion — the manifest carries none); streams the landed file for
/// transport SHA + length. Returns canonical receipt bytes
/// (`manifest_id`, `transport_sha256`, `byte_len`, `locator`).
pub fn ordinary_checkpoint_receipt(
    root: &Path,
    locator: &str,
    manifest_doc: &[u8],
) -> Result<Vec<u8>, TapeError> {
    let manifest: Value = serde_json::from_slice(manifest_doc)
        .map_err(|e| snap_err(format!("manifest is not JSON: {e}")))?;
    let map = manifest
        .as_object()
        .ok_or_else(|| snap_err("manifest must be an object".to_string()))?;
    if map.len() != CHECKPOINT_MANIFEST_KEYS.len()
        || !CHECKPOINT_MANIFEST_KEYS
            .iter()
            .all(|k| map.contains_key(*k))
    {
        return Err(snap_err(
            "manifest must carry exactly the 19 CheckpointManifest fields".to_string(),
        ));
    }
    let version = map
        .get("checkpoint_version")
        .and_then(Value::as_str)
        .ok_or_else(|| snap_err("manifest lacks checkpoint_version".to_string()))?;
    if !CHECKPOINT_VERSIONS.contains(&version) {
        return Err(snap_err(format!(
            "unsupported checkpoint_version {version:?}"
        )));
    }
    for key in CHECKPOINT_MANIFEST_KEYS {
        if key == "checkpoint_version" {
            continue;
        }
        let value = map.get(key).ok_or_else(|| {
            snap_err(format!("manifest lacks {key}"))
        })?;
        let optional = matches!(
            key,
            "dataset_manifest_hash" | "rollout_artifact_hash" | "parent_checkpoint_hash"
        );
        match value {
            Value::Null if optional => {}
            Value::String(text) => {
                parse_digest_text(text)?;
            }
            _ => {
                return Err(snap_err(format!("manifest field {key} has a bad shape")));
            }
        }
    }
    let dataset = map.get("dataset_manifest_hash").unwrap_or(&Value::Null);
    let rollout = map.get("rollout_artifact_hash").unwrap_or(&Value::Null);
    if (dataset == &Value::Null) == (rollout == &Value::Null) {
        return Err(snap_err(
            "manifest must declare exactly one source: dataset XOR rollout".to_string(),
        ));
    }
    // FULL-JCS manifest id (the manifest carries no digest field to exclude).
    let manifest_bytes =
        hydra_feed::canon::canonical_bytes_value(&manifest, "rl:checkpoint-manifest")
            .map_err(|e| snap_err(format!("manifest JCS emit failed: {e:?}")))?;
    use sha2::Digest as _;
    let manifest_id = digest_text(&sha2::Sha256::digest(&manifest_bytes).into());
    let path = join_root(root, locator)?;
    let (transport_sha256, byte_len) = stream_file_digest(&path)?;
    let mut receipt = Map::with_capacity(4);
    receipt.insert("manifest_id".to_string(), Value::String(manifest_id));
    receipt.insert(
        "transport_sha256".to_string(),
        Value::String(transport_sha256),
    );
    receipt.insert("byte_len".to_string(), Value::from(byte_len));
    receipt.insert("locator".to_string(), Value::String(locator.to_string()));
    hydra_feed::canon::canonical_bytes_value(&Value::Object(receipt), "rl:checkpoint-receipt")
        .map_err(|e| snap_err(format!("receipt JCS emit failed: {e:?}")))
}

/// Re-verify a receipt against the on-disk file (locator, SHA, length).
/// Returns the bound manifest id on success.
pub fn verify_receipt_against_disk(root: &Path, receipt_doc: &[u8]) -> Result<String, TapeError> {
    let receipt: Value = serde_json::from_slice(receipt_doc)
        .map_err(|e| snap_err(format!("receipt is not JSON: {e}")))?;
    let map = receipt
        .as_object()
        .ok_or_else(|| snap_err("receipt must be an object".to_string()))?;
    let locator = map
        .get("locator")
        .and_then(Value::as_str)
        .ok_or_else(|| snap_err("receipt lacks locator".to_string()))?;
    let recorded_sha = map
        .get("transport_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| snap_err("receipt lacks transport_sha256".to_string()))?;
    let recorded_len = map
        .get("byte_len")
        .and_then(Value::as_u64)
        .ok_or_else(|| snap_err("receipt lacks byte_len".to_string()))?;
    let manifest_id = map
        .get("manifest_id")
        .and_then(Value::as_str)
        .ok_or_else(|| snap_err("receipt lacks manifest_id".to_string()))?
        .to_string();
    parse_digest_text(&manifest_id)?;
    parse_digest_text(recorded_sha)?;
    let path = join_root(root, locator)?;
    let (actual_sha, actual_len) = stream_file_digest(&path)?;
    if actual_sha != recorded_sha || actual_len != recorded_len {
        return Err(snap_err(format!(
            "ordinary checkpoint at {locator:?} diverges from its receipt"
        )));
    }
    Ok(manifest_id)
}

fn nested_keys(doc: &Map<String, Value>) -> Vec<(String, Vec<String>)> {
    let mut nested = Vec::new();
    for name in ["cohort_membership", "frontier_current", "frontier_next"] {
        if let Some(obj) = doc.get(name).and_then(Value::as_object) {
            nested.push((name.to_string(), obj.keys().cloned().collect()));
        }
    }
    if let Some(tickets) = doc.get("tickets").and_then(Value::as_array) {
        let mut union: Vec<String> = Vec::new();
        for entry in tickets {
            if let Some(obj) = entry.as_object() {
                for key in obj.keys() {
                    if !union.contains(key) {
                        union.push(key.clone());
                    }
                }
            }
        }
        nested.push(("tickets".to_string(), union));
    }
    nested
}

/// Publish one control snapshot: closed envelope validation, ordinary
/// receipt binding (both roles), blob verification, fixed-parent check,
/// digest verification, phase monotonicity, then fsync + atomic tip.
/// Returns canonical tip bytes.
///
/// `receipts` carries the learner + actor ordinary receipts (matched by
/// locator); `expected_parent` is the generation-init id every snapshot of
/// the generation must name; `blobs` carries `(digest, locator)` pairs for
/// the tape/event/native/order blobs referenced by digest.
pub fn publish_control(
    root: &Path,
    control_doc: &[u8],
    receipts: &[Vec<u8>],
    expected_parent: Option<&str>,
    blobs: &[(String, String)],
) -> Result<Vec<u8>, TapeError> {
    let control: Value = serde_json::from_slice(control_doc)
        .map_err(|e| snap_err(format!("control is not JSON: {e}")))?;
    let map = control
        .as_object()
        .ok_or_else(|| snap_err("control must be an object".to_string()))?;
    let keys: Vec<String> = map
        .keys()
        .filter(|key| key.as_str() != "digest")
        .cloned()
        .collect();
    check_envelope_keys(RL_CONTROL_ARTIFACT, &keys, &nested_keys(map))
        .map_err(snap_err)?;
    for required in RL_CONTROL_FIELDS {
        if required != "digest" && !map.contains_key(required) {
            return Err(snap_err(format!("control lacks {required}")));
        }
    }
    // Nested cohort/frontier vocabulary (free maps skip enumeration, but the
    // closed nested keys must be present with the right shapes).
    for dotted in [
        "cohort_membership.generations",
        "cohort_membership.games_per_generation",
        "frontier_current.generation",
        "frontier_current.round",
        "frontier_current.epoch",
        "frontier_next.generation",
        "frontier_next.round",
        "frontier_next.epoch",
    ] {
        let mut parts = dotted.splitn(2, '.');
        let parent = parts.next().unwrap_or_default();
        let child = parts.next().unwrap_or_default();
        let present = map
            .get(parent)
            .and_then(Value::as_object)
            .is_some_and(|obj| obj.contains_key(child));
        if !present {
            return Err(snap_err(format!("control lacks {dotted}")));
        }
    }
    let phase = map
        .get("publication_phase")
        .and_then(Value::as_str)
        .ok_or_else(|| snap_err("control lacks publication_phase".to_string()))?;
    if !RL_PUBLICATION_PHASES.contains(&phase) {
        return Err(snap_err(format!("unknown publication_phase {phase:?}")));
    }
    let tickets = map
        .get("tickets")
        .and_then(Value::as_array)
        .ok_or_else(|| snap_err("control tickets must be an array".to_string()))?;
    for entry in tickets {
        let obj = entry
            .as_object()
            .ok_or_else(|| snap_err("control ticket must be an object".to_string()))?;
        let ticket_phase = obj
            .get("phase")
            .and_then(Value::as_str)
            .ok_or_else(|| snap_err("control ticket lacks phase".to_string()))?;
        if !RL_TICKET_PHASES.contains(&ticket_phase) {
            return Err(snap_err(format!("unknown ticket phase {ticket_phase:?}")));
        }
        if obj.get("slot_epoch").and_then(Value::as_u64).is_none() {
            return Err(snap_err("control ticket lacks slot_epoch".to_string()));
        }
        if !obj.contains_key("ticket_descriptor") || !obj.contains_key("stream_cursor") {
            return Err(snap_err("control ticket lacks descriptor/cursor".to_string()));
        }
    }
    let generation = map
        .get("generation")
        .and_then(Value::as_u64)
        .ok_or_else(|| snap_err("control lacks generation".to_string()))?;
    let parent = map
        .get("parent_checkpoint_hash")
        .and_then(Value::as_str)
        .ok_or_else(|| snap_err("control lacks parent_checkpoint_hash".to_string()))?;
    parse_digest_text(parent)?;
    if let Some(want) = expected_parent {
        if parent != want {
            return Err(snap_err(
                "control parent is not the generation-init id".to_string(),
            ));
        }
    }
    // Ordinary receipt binding for both roles (locator match, manifest id
    // match, transport match, on-disk re-verification).
    for role in ["learner", "actor"] {
        let manifest_key = format!("{role}_manifest_id");
        let sha_key = format!("{role}_transport_sha");
        let locator_key = format!("{role}_locator");
        let want_manifest = map
            .get(&manifest_key)
            .and_then(Value::as_str)
            .ok_or_else(|| snap_err(format!("control lacks {manifest_key}")))?;
        let want_sha = map
            .get(&sha_key)
            .and_then(Value::as_str)
            .ok_or_else(|| snap_err(format!("control lacks {sha_key}")))?;
        let want_locator = map
            .get(&locator_key)
            .and_then(Value::as_str)
            .ok_or_else(|| snap_err(format!("control lacks {locator_key}")))?;
        parse_digest_text(want_manifest)?;
        parse_digest_text(want_sha)?;
        let mut matched = false;
        for receipt_doc in receipts {
            let receipt: Value = serde_json::from_slice(receipt_doc)
                .map_err(|e| snap_err(format!("receipt is not JSON: {e}")))?;
            let robj = receipt
                .as_object()
                .ok_or_else(|| snap_err("receipt must be an object".to_string()))?;
            if robj.get("locator").and_then(Value::as_str) != Some(want_locator) {
                continue;
            }
            matched = true;
            if robj.get("manifest_id").and_then(Value::as_str) != Some(want_manifest) {
                return Err(snap_err(format!("{role} receipt manifest id mismatches control")));
            }
            if robj.get("transport_sha256").and_then(Value::as_str) != Some(want_sha) {
                return Err(snap_err(format!("{role} receipt transport SHA mismatches control")));
            }
            let disk_manifest = verify_receipt_against_disk(root, receipt_doc)?;
            if disk_manifest != want_manifest {
                return Err(snap_err(format!("{role} on-disk manifest id mismatches control")));
            }
        }
        if !matched {
            return Err(snap_err(format!("no receipt covers {role} locator")));
        }
    }
    // Referenced blobs stream to their recorded digests.
    for (digest, locator) in blobs {
        parse_digest_text(digest)?;
        let path = join_root(root, locator)?;
        let (actual, _) = stream_file_digest(&path)?;
        if &actual != digest {
            return Err(snap_err(format!("blob {locator:?} diverges from its digest")));
        }
    }
    // Envelope digest verification (recorded `digest` must match the JCS
    // payload without it — no self-hash cycle, the digest is verified, not
    // trusted).
    let (digest, _) = identity_of_payload_without_digest(&control).map_err(snap_err)?;
    // Phase monotonicity within one generation (cross-generation resets).
    if let Ok((tip_control, tip_phase, tip_generation)) = read_tip_parts(root) {
        let _ = tip_control;
        if tip_generation == generation {
            let old_idx = RL_PUBLICATION_PHASES
                .iter()
                .position(|p| *p == tip_phase)
                .unwrap_or(0);
            let new_idx = RL_PUBLICATION_PHASES
                .iter()
                .position(|p| *p == phase)
                .unwrap_or(0);
            if new_idx < old_idx {
                return Err(snap_err(format!(
                    "publication_phase regresses {tip_phase} -> {phase} in generation {generation}"
                )));
            }
        }
    }
    let hex = digest
        .strip_prefix(DIGEST_PREFIX)
        .unwrap_or(&digest)
        .to_string();
    let control_path = root.join("control").join(format!("control_{hex}.json"));
    let canonical = hydra_feed::canon::canonical_bytes_value(&control, "rl:control")
        .map_err(|e| snap_err(format!("control JCS emit failed: {e:?}")))?;
    atomic_write(&control_path, &canonical)?;
    commit_blob(&control_path)?;
    let control_locator = format!("control/control_{hex}.json");
    let mut tip = Map::with_capacity(4);
    tip.insert("control_digest".to_string(), Value::String(digest.clone()));
    tip.insert(
        "control_locator".to_string(),
        Value::String(control_locator),
    );
    tip.insert("generation".to_string(), Value::from(generation));
    tip.insert("phase".to_string(), Value::String(phase.to_string()));
    let tip_bytes = hydra_feed::canon::canonical_bytes_value(&Value::Object(tip), "rl:tip")
        .map_err(|e| snap_err(format!("tip JCS emit failed: {e:?}")))?;
    let tip_path = root.join("control").join("tip.json");
    if tip_path.exists() {
        let bak = root.join("control").join("tip.bak.json");
        let current = std::fs::read(&tip_path)
            .map_err(|e| snap_err(format!("tip backup read failed: {e}")))?;
        atomic_write(&bak, &current)?;
    }
    atomic_write(&tip_path, &tip_bytes)?;
    Ok(tip_bytes)
}

/// Read the authoritative tip: `tip.json` preferred, one-backup `tip.bak.json`
/// on corruption, hard error when neither validates. Returns the control
/// document bytes (digest-verified against the stored file).
pub fn read_tip(root: &Path) -> Result<Vec<u8>, TapeError> {
    let (control_bytes, _, _) = read_tip_parts(root)?;
    Ok(control_bytes)
}

fn read_tip_parts(root: &Path) -> Result<(Vec<u8>, String, u64), TapeError> {
    let tip_path = root.join("control").join("tip.json");
    let bak_path = root.join("control").join("tip.bak.json");
    let mut last_err = snap_err("no committed tip".to_string());
    for candidate in [&tip_path, &bak_path] {
        match read_one_tip(root, candidate) {
            Ok(parts) => return Ok(parts),
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

fn read_one_tip(root: &Path, tip_path: &Path) -> Result<(Vec<u8>, String, u64), TapeError> {
    let tip_bytes = std::fs::read(tip_path)
        .map_err(|e| snap_err(format!("tip unreadable: {e}")))?;
    let tip: Value = serde_json::from_slice(&tip_bytes)
        .map_err(|e| snap_err(format!("tip is not JSON: {e}")))?;
    let map = tip
        .as_object()
        .ok_or_else(|| snap_err("tip must be an object".to_string()))?;
    let control_digest = map
        .get("control_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| snap_err("tip lacks control_digest".to_string()))?
        .to_string();
    let control_locator = map
        .get("control_locator")
        .and_then(Value::as_str)
        .ok_or_else(|| snap_err("tip lacks control_locator".to_string()))?
        .to_string();
    let generation = map
        .get("generation")
        .and_then(Value::as_u64)
        .ok_or_else(|| snap_err("tip lacks generation".to_string()))?;
    let phase = map
        .get("phase")
        .and_then(Value::as_str)
        .ok_or_else(|| snap_err("tip lacks phase".to_string()))?
        .to_string();
    parse_digest_text(&control_digest)?;
    if !RL_PUBLICATION_PHASES.contains(&phase.as_str()) {
        return Err(snap_err("tip carries an unknown phase".to_string()));
    }
    let control_path = join_root(root, &control_locator)?;
    let control_bytes = std::fs::read(&control_path)
        .map_err(|e| snap_err(format!("control file unreadable: {e}")))?;
    let control: Value = serde_json::from_slice(&control_bytes)
        .map_err(|e| snap_err(format!("control file is not JSON: {e}")))?;
    let (verified, _) =
        identity_of_payload_without_digest(&control).map_err(snap_err)?;
    if verified != control_digest {
        return Err(snap_err("tip control digest diverges from the stored file".to_string()));
    }
    Ok((control_bytes, phase, generation))
}

/// Stage one per-game native snapshot envelope: closed key check, required
/// shapes, digest computed here (input MUST NOT carry `digest` — no
/// self-hash games), content-addressed write. Returns canonical receipt
/// bytes (`digest`, `locator`). Value shapes inside `pending_responses` /
/// `stream_cursor` stay consumer-owned; presence plus the digest-bound
/// identity fields are enforced here.
pub fn stage_native_snapshot(root: &Path, game_doc: &[u8]) -> Result<Vec<u8>, TapeError> {
    let doc: Value = serde_json::from_slice(game_doc)
        .map_err(|e| snap_err(format!("native snapshot is not JSON: {e}")))?;
    let map = doc
        .as_object()
        .ok_or_else(|| snap_err("native snapshot must be an object".to_string()))?;
    if map.contains_key("digest") {
        return Err(snap_err(
            "native snapshot must exclude digest (computed at stage)".to_string(),
        ));
    }
    let keys: Vec<String> = map.keys().cloned().collect();
    check_envelope_keys(RL_NATIVE_SNAPSHOT_ARTIFACT, &keys, &[]).map_err(snap_err)?;
    for required in RL_NATIVE_SNAPSHOT_FIELDS {
        if required != "digest" && !map.contains_key(required) {
            return Err(snap_err(format!("native snapshot lacks {required}")));
        }
    }
    for name in ["wall_id", "game_id"] {
        let text = map
            .get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| snap_err(format!("native snapshot lacks {name}")))?;
        if text.is_empty() || text.len() > 1024 {
            return Err(snap_err(format!("native snapshot {name} has bad length")));
        }
    }
    let seats = map
        .get("seat_policy_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| snap_err("seat_policy_ids must be an array".to_string()))?;
    if seats.len() != 4
        || seats
            .iter()
            .any(|s| s.as_str().is_none_or(|text| text.is_empty()))
    {
        return Err(snap_err("seat_policy_ids must hold four policy ids".to_string()));
    }
    if map.get("ordinal").and_then(Value::as_u64).is_none() {
        return Err(snap_err("native snapshot lacks ordinal".to_string()));
    }
    for name in ["core_carry_hash", "action_prefix_digest", "state_digest"] {
        let text = map
            .get(name)
            .and_then(Value::as_str)
            .ok_or_else(|| snap_err(format!("native snapshot lacks {name}")))?;
        parse_digest_text(text)?;
    }
    if map.get("decision_count").and_then(Value::as_u64).is_none() {
        return Err(snap_err("native snapshot lacks decision_count".to_string()));
    }
    if map.get("pending_responses").and_then(Value::as_array).is_none() {
        return Err(snap_err("pending_responses must be an array".to_string()));
    }
    if !map.contains_key("stream_cursor") {
        return Err(snap_err("native snapshot lacks stream_cursor".to_string()));
    }
    let bytes = hydra_feed::canon::canonical_bytes_value(&doc, "rl:native-snapshot")
        .map_err(|e| snap_err(format!("native snapshot JCS emit failed: {e:?}")))?;
    use sha2::Digest as _;
    let digest = digest_text(&sha2::Sha256::digest(&bytes).into());
    let hex = digest
        .strip_prefix(DIGEST_PREFIX)
        .unwrap_or(&digest)
        .to_string();
    let mut full = map.clone();
    full.insert("digest".to_string(), Value::String(digest.clone()));
    let full_bytes =
        hydra_feed::canon::canonical_bytes_value(&Value::Object(full), "rl:native-snapshot-full")
            .map_err(|e| snap_err(format!("native snapshot full JCS emit failed: {e:?}")))?;
    let locator = format!("native/{hex}.json");
    let path = join_root(root, &locator)?;
    atomic_write(&path, &full_bytes)?;
    commit_blob(&path)?;
    let mut receipt = Map::with_capacity(3);
    receipt.insert("digest".to_string(), Value::String(digest));
    receipt.insert("locator".to_string(), Value::String(locator));
    receipt.insert(
        "game_id".to_string(),
        Value::String(
            map.get("game_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        ),
    );
    hydra_feed::canon::canonical_bytes_value(&Value::Object(receipt), "rl:native-receipt")
        .map_err(|e| snap_err(format!("native receipt JCS emit failed: {e:?}")))
}

/// Read one staged native snapshot by digest (cross-checked against the
/// requested game id). Returns the canonical document bytes.
pub fn read_native_snapshot(
    root: &Path,
    game_id: &str,
    expect_digest: &str,
) -> Result<Vec<u8>, TapeError> {
    let raw = parse_digest_text(expect_digest)?;
    let locator = format!("native/{}.json", hex_of(&raw));
    let path = join_root(root, &locator)?;
    let bytes =
        std::fs::read(&path).map_err(|e| snap_err(format!("native snapshot unreadable: {e}")))?;
    let doc: Value = serde_json::from_slice(&bytes)
        .map_err(|e| snap_err(format!("native snapshot is not JSON: {e}")))?;
    let (verified, _) = identity_of_payload_without_digest(&doc).map_err(snap_err)?;
    if verified != expect_digest {
        return Err(snap_err("native snapshot digest diverges".to_string()));
    }
    let got = doc
        .get("game_id")
        .and_then(Value::as_str)
        .ok_or_else(|| snap_err("native snapshot lacks game_id".to_string()))?;
    if got != game_id {
        return Err(snap_err("native snapshot game id mismatch".to_string()));
    }
    Ok(bytes)
}

/// Names of the free content-reference maps (re-export for FFI/tooling).
pub fn control_free_maps() -> &'static [&'static str] {
    &RL_CONTROL_FREE_MAPS
}

/// Names of the closed control fields (re-export for FFI/tooling).
pub fn control_fields() -> &'static [&'static str] {
    &RL_CONTROL_FIELDS
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SNAP_TMP: AtomicU64 = AtomicU64::new(0);

    fn tmp_root() -> PathBuf {
        let tag = SNAP_TMP.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir()
            .join(format!("rl-snapshot-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("test tmp");
        dir
    }

    fn manifest_doc(version: &str, dataset: bool) -> Vec<u8> {
        let mut map = Map::with_capacity(19);
        for key in CHECKPOINT_MANIFEST_KEYS {
            let value = if key == "checkpoint_version" {
                Value::String(version.to_string())
            } else if key == "dataset_manifest_hash" {
                if dataset {
                    Value::String(format!("{DIGEST_PREFIX}{}", "d".repeat(64)))
                } else {
                    Value::Null
                }
            } else if key == "rollout_artifact_hash" {
                if dataset {
                    Value::Null
                } else {
                    Value::String(format!("{DIGEST_PREFIX}{}", "a".repeat(64)))
                }
            } else if key == "parent_checkpoint_hash" {
                Value::String(format!("{DIGEST_PREFIX}{}", "c".repeat(64)))
            } else {
                Value::String(format!("{DIGEST_PREFIX}{}", "e".repeat(64)))
            };
            map.insert(key.to_string(), value);
        }
        serde_json::to_vec(&Value::Object(map)).expect("test manifest json")
    }

    fn write_ordinary(root: &Path, locator: &str, payload: &[u8]) {
        let path = root.join(locator);
        std::fs::create_dir_all(path.parent().expect("test parent")).expect("test dir");
        std::fs::write(&path, payload).expect("test payload");
    }

    fn control_doc(
        learner: (&str, &str, &str),
        actor: (&str, &str, &str),
        parent: &str,
        phase: &str,
        generation: u64,
    ) -> Vec<u8> {
        let mut map = Map::new();
        map.insert(
            "schema_version".to_string(),
            Value::String(CONTROL_SCHEMA_VERSION.to_string()),
        );
        map.insert(
            "pipeline_spec_id".to_string(),
            Value::String(format!("{DIGEST_PREFIX}{}", "0".repeat(64))),
        );
        map.insert(
            "learner_manifest_id".to_string(),
            Value::String(learner.0.to_string()),
        );
        map.insert(
            "learner_transport_sha".to_string(),
            Value::String(learner.1.to_string()),
        );
        map.insert(
            "learner_locator".to_string(),
            Value::String(learner.2.to_string()),
        );
        map.insert(
            "actor_manifest_id".to_string(),
            Value::String(actor.0.to_string()),
        );
        map.insert(
            "actor_transport_sha".to_string(),
            Value::String(actor.1.to_string()),
        );
        map.insert(
            "actor_locator".to_string(),
            Value::String(actor.2.to_string()),
        );
        map.insert("generation".to_string(), Value::from(generation));
        map.insert(
            "generation_statuses".to_string(),
            Value::Object(Map::new()),
        );
        map.insert("source_refs".to_string(), Value::Object(Map::new()));
        map.insert(
            "order_ref".to_string(),
            Value::String(format!("{DIGEST_PREFIX}{}", "1".repeat(64))),
        );
        map.insert("ledger_refs".to_string(), Value::Object(Map::new()));
        for name in [
            "native_snapshot_digest",
            "tape_blob_digest",
            "event_blob_digest",
        ] {
            map.insert(
                name.to_string(),
                Value::String(format!("{DIGEST_PREFIX}{}", "2".repeat(64))),
            );
        }
        let mut cohort = Map::new();
        cohort.insert("generations".to_string(), Value::from(8u64));
        cohort.insert("games_per_generation".to_string(), Value::from(128u64));
        map.insert("cohort_membership".to_string(), Value::Object(cohort));
        for name in ["frontier_current", "frontier_next"] {
            let mut frontier = Map::new();
            frontier.insert("generation".to_string(), Value::from(generation));
            frontier.insert("round".to_string(), Value::from(0u64));
            frontier.insert("epoch".to_string(), Value::from(0u64));
            map.insert(name.to_string(), Value::Object(frontier));
        }
        map.insert("terminal_set".to_string(), Value::Array(vec![]));
        let mut ticket = Map::new();
        ticket.insert(
            "ticket_descriptor".to_string(),
            Value::String("slot-0:epoch-1".to_string()),
        );
        ticket.insert("phase".to_string(), Value::String("ready".to_string()));
        ticket.insert("slot_epoch".to_string(), Value::from(1u64));
        let mut cursor = Map::new();
        cursor.insert("before".to_string(), Value::from(0u64));
        cursor.insert("after".to_string(), Value::from(8u64));
        ticket.insert("stream_cursor".to_string(), Value::Object(cursor));
        map.insert("tickets".to_string(), Value::Array(vec![Value::Object(ticket)]));
        map.insert("committed_row_offset".to_string(), Value::from(0u64));
        map.insert("committed_update_offset".to_string(), Value::from(0u64));
        map.insert(
            "publication_phase".to_string(),
            Value::String(phase.to_string()),
        );
        map.insert(
            "parent_checkpoint_hash".to_string(),
            Value::String(parent.to_string()),
        );
        // Digest computed here (test helper mirrors the staging rule).
        let bytes = hydra_feed::canon::canonical_bytes_value(&Value::Object(map.clone()), "test:ctl")
            .expect("test control JCS");
        use sha2::Digest as _;
        let digest = digest_text(&sha2::Sha256::digest(&bytes).into());
        map.insert("digest".to_string(), Value::String(digest));
        hydra_feed::canon::canonical_bytes_value(&Value::Object(map), "test:ctl-full")
            .expect("test control full JCS")
    }

    #[test]
    fn receipt_round_trips_and_rejects() {
        let root = tmp_root();
        write_ordinary(&root, "ckpt/l0.pt", b"fake-torch-bytes");
        let manifest = manifest_doc("1.0.0", false);
        let receipt = ordinary_checkpoint_receipt(&root, "ckpt/l0.pt", &manifest).expect("receipt");
        let back = verify_receipt_against_disk(&root, &receipt).expect("verify");
        let receipt_value: Value = serde_json::from_slice(&receipt).expect("receipt json");
        assert_eq!(
            receipt_value.get("manifest_id").and_then(Value::as_str),
            Some(back.as_str())
        );
        // Corrupt the file: receipt no longer verifies.
        write_ordinary(&root, "ckpt/l0.pt", b"tampered");
        assert!(verify_receipt_against_disk(&root, &receipt).is_err());
        // Absolute + escape locators rejected without touching disk.
        assert!(ordinary_checkpoint_receipt(&root, "/abs/evil.pt", &manifest).is_err());
        assert!(ordinary_checkpoint_receipt(&root, "../evil.pt", &manifest).is_err());
        // Wrong version, both-sources, and non-19 manifests rejected. A
        // dataset-sourced manifest is VALID here (training checkpoints
        // reference their dataset; the gate requires exactly one source,
        // not a particular one).
        assert!(ordinary_checkpoint_receipt(&root, "ckpt/l0.pt", &manifest_doc("2.0.0", false)).is_err());
        let dataset_receipt =
            ordinary_checkpoint_receipt(&root, "ckpt/l0.pt", &manifest_doc("1.0.0", true))
                .expect("dataset-sourced ordinary manifest receipts");
        assert!(verify_receipt_against_disk(&root, &dataset_receipt).is_ok());
        let both = {
            let mut value: Value = serde_json::from_slice(&manifest_doc("1.0.0", false)).expect("json");
            value["dataset_manifest_hash"] =
                Value::String(format!("{DIGEST_PREFIX}{}", "d".repeat(64)));
            serde_json::to_vec(&value).expect("json")
        };
        assert!(ordinary_checkpoint_receipt(&root, "ckpt/l0.pt", &both).is_err());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn control_publish_tip_authority_and_phases() {
        let root = tmp_root();
        write_ordinary(&root, "ckpt/learner.pt", b"learner-payload");
        write_ordinary(&root, "ckpt/actor.pt", b"actor-payload");
        let learner_manifest = manifest_doc("1.0.0", false);
        let actor_manifest = manifest_doc("1.0.0", false);
        let learner_receipt =
            ordinary_checkpoint_receipt(&root, "ckpt/learner.pt", &learner_manifest).expect("lr");
        let actor_receipt =
            ordinary_checkpoint_receipt(&root, "ckpt/actor.pt", &actor_manifest).expect("ar");
        let lr: Value = serde_json::from_slice(&learner_receipt).expect("json");
        let ar: Value = serde_json::from_slice(&actor_receipt).expect("json");
        let lmid = lr.get("manifest_id").and_then(Value::as_str).expect("id").to_string();
        let lsha = lr.get("transport_sha256").and_then(Value::as_str).expect("sha").to_string();
        let amid = ar.get("manifest_id").and_then(Value::as_str).expect("id").to_string();
        let asha = ar.get("transport_sha256").and_then(Value::as_str).expect("sha").to_string();
        let parent = format!("{DIGEST_PREFIX}{}", "c".repeat(64));
        let receipts = vec![learner_receipt, actor_receipt];
        // First publication at captured_at_update1.
        let doc1 = control_doc(
            (&lmid, &lsha, "ckpt/learner.pt"),
            (&amid, &asha, "ckpt/actor.pt"),
            &parent,
            "captured_at_update1",
            2,
        );
        let tip1 = publish_control(&root, &doc1, &receipts, Some(&parent), &[]).expect("pub1");
        assert!(!tip1.is_empty());
        let back = read_tip(&root).expect("read tip");
        let back_value: Value = serde_json::from_slice(&back).expect("json");
        assert_eq!(
            back_value.get("publication_phase").and_then(Value::as_str),
            Some("captured_at_update1")
        );
        // Phase regression in the same generation is rejected.
        let doc0 = control_doc(
            (&lmid, &lsha, "ckpt/learner.pt"),
            (&amid, &asha, "ckpt/actor.pt"),
            &parent,
            "not_captured",
            2,
        );
        assert!(publish_control(&root, &doc0, &receipts, Some(&parent), &[]).is_err());
        // Forward publication succeeds; wrong parent is rejected.
        let doc2 = control_doc(
            (&lmid, &lsha, "ckpt/learner.pt"),
            (&amid, &asha, "ckpt/actor.pt"),
            &parent,
            "checkpoint_durable",
            2,
        );
        publish_control(&root, &doc2, &receipts, Some(&parent), &[]).expect("pub2");
        let evil_parent = format!("{DIGEST_PREFIX}{}", "f".repeat(64));
        assert!(publish_control(&root, &doc2, &receipts, Some(&evil_parent), &[]).is_err());
        // Corrupt the live tip: the backup (phase captured_at_update1)
        // becomes authoritative.
        std::fs::write(root.join("control").join("tip.json"), b"{broken").expect("corrupt");
        let fallback = read_tip(&root).expect("fallback tip");
        let fallback_value: Value = serde_json::from_slice(&fallback).expect("json");
        assert_eq!(
            fallback_value.get("publication_phase").and_then(Value::as_str),
            Some("captured_at_update1")
        );
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn native_snapshot_stage_and_read() {
        let root = tmp_root();
        let mut map = Map::new();
        map.insert(
            "schema_version".to_string(),
            Value::String(CONTROL_SCHEMA_VERSION.to_string()),
        );
        map.insert("wall_id".to_string(), Value::String("wall-1".to_string()));
        map.insert("game_id".to_string(), Value::String("game-a".to_string()));
        map.insert(
            "seat_policy_ids".to_string(),
            Value::Array(vec![
                Value::String("ckpt-c0".to_string()),
                Value::String("ckpt-c0".to_string()),
                Value::String("ckpt-c0".to_string()),
                Value::String("ckpt-c0".to_string()),
            ]),
        );
        map.insert("ordinal".to_string(), Value::from(3u64));
        for name in ["core_carry_hash", "action_prefix_digest", "state_digest"] {
            map.insert(
                name.to_string(),
                Value::String(format!("{DIGEST_PREFIX}{}", "7".repeat(64))),
            );
        }
        map.insert("decision_count".to_string(), Value::from(41u64));
        map.insert("pending_responses".to_string(), Value::Array(vec![]));
        let mut cursor = Map::new();
        cursor.insert("seed_hex".to_string(), Value::String("00".repeat(32)));
        cursor.insert("cursor".to_string(), Value::from(88u64));
        map.insert("stream_cursor".to_string(), Value::Object(cursor));
        let doc = hydra_feed::canon::canonical_bytes_value(&Value::Object(map), "test:native")
            .expect("test native JCS");
        let receipt = stage_native_snapshot(&root, &doc).expect("stage");
        let receipt_value: Value = serde_json::from_slice(&receipt).expect("json");
        let digest = receipt_value
            .get("digest")
            .and_then(Value::as_str)
            .expect("digest")
            .to_string();
        let back = read_native_snapshot(&root, "game-a", &digest).expect("read");
        assert!(!back.is_empty());
        assert!(read_native_snapshot(&root, "game-B", &digest).is_err());
        std::fs::remove_dir_all(&root).ok();
    }
}
