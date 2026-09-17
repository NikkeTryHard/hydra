//! `ismcts_driver`: unified tiny transition + deduped vectors + batch descent.
//!
//! Unifies `ismcts_search.py:95-136` (`_apply_action`) with
//! `gumbel_core.py:371-411` (`exact_transition`): both consume one live tile,
//! rotate turn, bump step, and stamp `last_action`; only the snapshot domain
//! differs (`ismcts:` vs `gumbel:`). One Rust transition keeps both salts via
//! an explicit `domain` arg, so parity goldens never fork on `world_id`.
//!
//! Dedups the three vector-copy sites with salts kept:
//! `ismcts_core.py:264-286` (`{wid}:{candidate1}:leaf`, `{wid}:terminal`),
//! `gumbel_core.py:254-286` (same shape, salt `candidate6`), and
//! `local_abstraction.py:141-164` (same shape, salt `leaf_kind`). One pair of
//! Rust fns takes the salt as a param; the tiny-domain math is identical.
//!
//! Batch descent owns the tree arena (`BTreeMap` digest -> `IsNode`, never
//! `hash()`), the `ismcts_search.py:333-350` descent/backup loop, and the
//! `ismcts_core.py:510-554` UCT call. Python precomputes the batch once per
//! search (sampled worlds, per-step info keys, per-step policy directions,
//! CTR floats) and Rust replays them with no per-step Python calls. Leaf
//! vectors default to the deduped stub fns; an optional override map lets
//! Python supply read-only torch vectors in batch (no per-leaf callback).
//! All floats are f64 with stated eps; tie-breaks never use `hash()`.
//!
//! Firewall: tree keys are `sha256:` digests only (info-keys, never world
//! blobs); `FORBIDDEN_IN_TREE_KEY` (17 names, `ismcts_core.py:135-155`) is
//! enforced on info-key docs and on key strings; world refs stay opaque
//! strings; continuation inputs are scalars (direction + float), never
//! `FullWorld`. Every shape violation fails closed (`SearchError`, never a
//! default). There is no clock, no global RNG, and no Python API here.

use std::collections::BTreeMap;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::SearchError;
use crate::ismcts::{IsNode, IsmctsParams, TieBreak, check_legal, uct_select};

/// Forbidden tree-key fields (`ismcts_core.py:135-155`): no world id, hidden
/// tiles, walls, snapshots, refs, or server-private state may appear in an
/// info-key document or key string. Presence fails closed, never filtered.
pub const FORBIDDEN_IN_TREE_KEY: &[&str] = &[
    "world_id",
    "simulator_snapshot",
    "hidden_tiles",
    "wall",
    "dead_wall",
    "opponent_hand",
    "full_world",
    "privileged",
    "privileged_label",
    "world_ref",
    "parent_id",
    "latent_state_hidden",
    "server_private",
    "engine_rng_state",
    "future_events",
    "opponent_concealed",
    "unrevealed_dora",
];

/// Final-selection tie window (`ismcts_search.py:454-457`): means within
/// `1e-12` are ties for the tie-break arm; larger gaps win outright.
pub const FINAL_EPS: f64 = 1e-12;
/// Continuation bias (`ismcts_core.py:431-461`): two-action tilt `0.5 +- 0.2`.
pub const CONTINUATION_BIAS: f64 = 0.2;
/// Tile id ceiling (`contracts` TileId `0..135`): ids `>= 136` fail closed.
pub const TILE_CEILING: u32 = 136;
/// Snapshot domain salt for the ISMCTS transition prefix.
pub const DOMAIN_ISMCTS: &str = "ismcts";
/// Snapshot domain salt for the Gumbel transition prefix.
pub const DOMAIN_GUMBEL: &str = "gumbel";

/// Validate `sha256:<64 lowercase hex>` shape (world ids, tree keys, hashes).
pub(crate) fn check_digest_shape(text: &str) -> Result<(), SearchError> {
    const PREFIX: &str = "sha256:";
    let hex = match text.strip_prefix(PREFIX) {
        Some(hex) => hex,
        None => {
            return Err(SearchError::InvalidArg { detail: "digest must start with sha256:" });
        }
    };
    if hex.len() != 64 {
        return Err(SearchError::InvalidArg { detail: "digest must hold 64 hex chars" });
    }
    let mut i = 0;
    while i < hex.len() {
        let byte = hex.as_bytes()[i];
        let ok = (b'0'..=b'9').contains(&byte) || (b'a'..=b'f').contains(&byte);
        if !ok {
            return Err(SearchError::InvalidArg {
                detail: "digest hex must be lowercase 0-9a-f",
            });
        }
        i += 1;
    }
    Ok(())
}

/// Validate a tree key: digest shape plus no forbidden substring
/// (`ismcts_core.py:311-325` best-effort guard on key strings).
pub fn check_tree_key(key: &str) -> Result<(), SearchError> {
    check_digest_shape(key)?;
    let mut i = 0;
    while i < FORBIDDEN_IN_TREE_KEY.len() {
        let bad = FORBIDDEN_IN_TREE_KEY[i];
        if key.contains(bad) {
            return Err(SearchError::InvalidArg {
                detail: "tree key leaks forbidden field",
            });
        }
        i += 1;
    }
    Ok(())
}

/// Validate one tile id (`0..135`).
fn check_tile(tile: u32) -> Result<(), SearchError> {
    if tile >= TILE_CEILING {
        return Err(SearchError::InvalidArg { detail: "tile id must be 0..135" });
    }
    Ok(())
}

/// Raw SHA-256 bytes (vector salts hash short ASCII payloads; the digest
/// itself is the draw, never a hex round-trip).
fn sha_raw(payload: &[u8]) -> [u8; 32] {
    let sum = Sha256::digest(payload);
    let bytes: &[u8] = sum.as_slice();
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = bytes[i];
        i += 1;
    }
    out
}

/// Deduped model vector (`ismcts_core.py:264-286` shape, salt kept):
/// `sha256(f"{wid}:{candidate}:leaf")`, first 4 bytes `(b % 100) / 100.0`.
/// `candidate_id` non-empty (keeps `candidate1` vs `candidate6` vs
/// `leaf_kind` salts distinct); `world_id` digest-shaped.
pub fn model_vector(world_id: &str, candidate_id: &str) -> Result<[f64; 4], SearchError> {
    check_digest_shape(world_id)?;
    if candidate_id.is_empty() {
        return Err(SearchError::InvalidArg { detail: "candidate_id must be non-empty" });
    }
    let mut payload = Vec::with_capacity(world_id.len() + candidate_id.len() + 8);
    payload.extend_from_slice(world_id.as_bytes());
    payload.extend_from_slice(b":");
    payload.extend_from_slice(candidate_id.as_bytes());
    payload.extend_from_slice(b":leaf");
    let digest = sha_raw(payload.as_slice());
    let mut out = [0.0f64; 4];
    let mut i = 0;
    while i < 4 {
        let value = (f64::from(digest[i] % 100)) / 100.0;
        if !value.is_finite() {
            return Err(SearchError::NonFinite { context: "ismcts model vector" });
        }
        out[i] = value;
        i += 1;
    }
    Ok(out)
}

/// Deduped terminal vector (`ismcts_core.py:289-308` shape):
/// `sha256(f"{wid}:terminal")`, scores `(b % 50) - 25`, then `/ 50.0 + 0.5`
/// in that order (bit-identical to the oracle float ops).
pub fn terminal_vector(world_id: &str) -> Result<[f64; 4], SearchError> {
    check_digest_shape(world_id)?;
    let mut payload = Vec::with_capacity(world_id.len() + 9);
    payload.extend_from_slice(world_id.as_bytes());
    payload.extend_from_slice(b":terminal");
    let digest = sha_raw(payload.as_slice());
    let mut out = [0.0f64; 4];
    let mut i = 0;
    while i < 4 {
        let score = i32::from(digest[i] % 50) - 25;
        let value = (f64::from(score)) / 50.0 + 0.5;
        if !value.is_finite() {
            return Err(SearchError::NonFinite { context: "ismcts terminal vector" });
        }
        out[i] = value;
        i += 1;
    }
    Ok(out)
}

/// Neumaier compensated sum over four seats (matches the oracle `sum()`
/// bit-exact on this toolchain: the low-order compensation decides the last
/// ulp of zero-sum centering, and naive left-to-right addition rounds the
/// other way).
fn neumaier_sum4(values: [f64; 4]) -> f64 {
    let mut sum = 0.0f64;
    let mut compensation = 0.0f64;
    let mut i = 0;
    while i < 4 {
        let next = values[i];
        let pending = sum + next;
        if sum.abs() >= next.abs() {
            compensation += (sum - pending) + next;
        } else {
            compensation += (next - pending) + sum;
        }
        sum = pending;
        i += 1;
    }
    sum + compensation
}

/// Deduped local model vector (`local_abstraction.py:141-164`, salt kept):
/// `sha256(wid + ":" + leaf_kind)`, four `u16` little-endian words
/// `((r / 65535.0) * 2.0 - 1.0)`, then zero-sum centered (`v - s / 4.0`).
/// Empty `world_id` reads as `"world_unknown"` (oracle `getattr` default);
/// out-of-`+-2.0` or non-finite values fail closed. `leaf_kind` is the salt
/// (`model` in prod, never defaulted here).
pub fn local_model_vector(world_id: &str, leaf_kind: &str) -> Result<[f64; 4], SearchError> {
    let wid = if world_id.is_empty() { "world_unknown" } else { world_id };
    let mut payload = Vec::with_capacity(wid.len() + leaf_kind.len() + 1);
    payload.extend_from_slice(wid.as_bytes());
    payload.extend_from_slice(b":");
    payload.extend_from_slice(leaf_kind.as_bytes());
    let digest = sha_raw(payload.as_slice());
    let mut vals = [0.0f64; 4];
    let mut i = 0;
    while i < 4 {
        let word = u16::from_le_bytes([digest[i * 2], digest[i * 2 + 1]]);
        let value = ((f64::from(word)) / 65535.0) * 2.0 - 1.0;
        if !value.is_finite() {
            return Err(SearchError::NonFinite { context: "ismcts local model vector" });
        }
        vals[i] = value;
        i += 1;
    }
    let sum = neumaier_sum4(vals);
    let mut out = [0.0f64; 4];
    let mut k = 0;
    while k < 4 {
        let value = vals[k] - sum / 4.0;
        if !value.is_finite() || value.abs() > 2.0 {
            return Err(SearchError::NonFinite { context: "ismcts local model centered" });
        }
        out[k] = value;
        k += 1;
    }
    Ok(out)
}

/// Deduped local terminal vector (`local_abstraction.py:167-204`): exact
/// settlement from hands + wall (no hash). Seat sums centered (`/ 10.0`),
/// wall influence added (`+ wall_sum * (0.1 | -0.03)`), re-centered. Exactly
/// four seats required (tiny frozen); non-finite fails closed.
pub fn local_terminal_vector(
    hands: &[Vec<u32>],
    live: &[u32],
) -> Result<[f64; 4], SearchError> {
    if hands.len() != 4 {
        return Err(SearchError::InvalidArg { detail: "local terminal needs 4 seats" });
    }
    let mut sums = [0.0f64; 4];
    let mut seat = 0;
    while seat < 4 {
        let mut acc: u64 = 0;
        let mut t = 0;
        while t < hands[seat].len() {
            let tile = hands[seat][t];
            check_tile(tile)?;
            acc = acc.checked_add(u64::from(tile)).ok_or(SearchError::InvalidArg {
                detail: "hand sum overflow",
            })?;
            t += 1;
        }
        sums[seat] = acc as f64;
        seat += 1;
    }
    let total = neumaier_sum4(sums);
    let mean = total / 4.0;
    let mut centered = [0.0f64; 4];
    let mut j = 0;
    while j < 4 {
        centered[j] = (sums[j] - mean) / 10.0;
        j += 1;
    }
    let mut wall_acc: u64 = 0;
    let mut w = 0;
    while w < live.len() {
        check_tile(live[w])?;
        wall_acc = wall_acc.checked_add(u64::from(live[w])).ok_or(SearchError::InvalidArg {
            detail: "wall sum overflow",
        })?;
        w += 1;
    }
    let wall_sum = (wall_acc as f64) / 100.0;
    let mut k = 0;
    while k < 4 {
        let push = if k == 0 { 0.1 } else { -0.03 };
        centered[k] += wall_sum * push;
        k += 1;
    }
    let total2 = neumaier_sum4(centered);
    let mean2 = total2 / 4.0;
    let mut out = [0.0f64; 4];
    let mut n = 0;
    while n < 4 {
        let value = centered[n] - mean2;
        if !value.is_finite() {
            return Err(SearchError::NonFinite { context: "ismcts local terminal vector" });
        }
        out[n] = value;
        n += 1;
    }
    Ok(out)
}

/// Exhaustive path offset (`local_strategy.py:263-268`): per-step
/// `(int(sha256(f"{wid}:{aid}:{d}").hexdigest()[:4], 16) % 100) / 1000.0 -
/// 0.05` summed over the path (first nibbles `u16` big-endian == hex `[:4]`).
pub fn exhaustive_path_offset(world_id: &str, path: &[u32]) -> Result<f64, SearchError> {
    if world_id.is_empty() {
        return Err(SearchError::InvalidArg { detail: "world_id must be non-empty" });
    }
    let mut off = 0.0f64;
    let mut d: u32 = 0;
    while (d as usize) < path.len() {
        let aid = path[d as usize];
        let payload = format!("{world_id}:{aid}:{d}");
        if payload.is_empty() {
            return Err(SearchError::InvalidArg { detail: "offset payload empty" });
        }
        let digest = sha_raw(payload.as_bytes());
        let head = u16::from_be_bytes([digest[0], digest[1]]);
        off += (f64::from(head % 100)) / 1000.0 - 0.05;
        d = d.checked_add(1).ok_or(SearchError::InvalidArg { detail: "path too long" })?;
    }
    if !off.is_finite() {
        return Err(SearchError::NonFinite { context: "ismcts path offset" });
    }
    Ok(off)
}

/// Model placeholder over a non-digest tag (`unvisited:{aid}`, `fallback`):
/// same salt math as `model_vector` but the tag is not a digest (kept salt,
/// no belief lookup).
fn placeholder_vector(tag: &str, candidate_id: &str) -> Result<[f64; 4], SearchError> {
    if tag.is_empty() || candidate_id.is_empty() {
        return Err(SearchError::InvalidArg { detail: "placeholder tag must be non-empty" });
    }
    let mut payload = Vec::with_capacity(tag.len() + candidate_id.len() + 8);
    payload.extend_from_slice(tag.as_bytes());
    payload.extend_from_slice(b":");
    payload.extend_from_slice(candidate_id.as_bytes());
    payload.extend_from_slice(b":leaf");
    let digest = sha_raw(payload.as_slice());
    let mut out = [0.0f64; 4];
    let mut i = 0;
    while i < 4 {
        let value = (f64::from(digest[i] % 100)) / 100.0;
        if !value.is_finite() {
            return Err(SearchError::NonFinite { context: "ismcts placeholder vector" });
        }
        out[i] = value;
        i += 1;
    }
    Ok(out)
}

/// Info key from a full observation JSON doc (`ismcts_core.py:197-242`):
/// parse, drop `legal_mask` + `observation_hash`, reject forbidden keys,
/// canon via the single feed site, hash. Input is the `ActorObservation`
/// `to_json` doc (never a `FullWorld`); `FullWorld` fields fail closed here.
pub fn info_key_from_doc(doc_bytes: &[u8]) -> Result<String, SearchError> {
    if doc_bytes.is_empty() {
        return Err(SearchError::InvalidArg { detail: "obs doc must be non-empty" });
    }
    let mut value: serde_json::Value = serde_json::from_slice(doc_bytes)
        .map_err(|_| SearchError::InvalidArg { detail: "obs doc must be JSON object" })?;
    let obj = match value.as_object_mut() {
        Some(obj) => obj,
        None => {
            return Err(SearchError::InvalidArg { detail: "obs doc must be JSON object" });
        }
    };
    obj.remove("legal_mask");
    obj.remove("observation_hash");
    let mut i = 0;
    while i < FORBIDDEN_IN_TREE_KEY.len() {
        let bad = FORBIDDEN_IN_TREE_KEY[i];
        if obj.contains_key(bad) {
            return Err(SearchError::InvalidArg {
                detail: "forbidden field in tree key document",
            });
        }
        i += 1;
    }
    let bytes = hydra_feed::canon::canonical_bytes_value(&value, "search:info_key")
        .map_err(|err| SearchError::Canon { detail: err.to_string() })?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

// ---------------------------------------------------------------------------
// Tiny world + unified transition
// ---------------------------------------------------------------------------

/// One tiny-domain world blob (arena-side copy, never a Python object).
///
/// Hands are 4 seats x 2 tiles in seat order (stored order, not sorted);
/// walls hold tile ids `0..135`. `step`/`turn`/`last_action`/`corpus_idx`
/// are `None` when absent from `latent_state` (corpus worlds carry only
/// `corpus_idx`; successors add `step`/`turn`/`last_action`). Absent keys
/// stay absent from the hashed `latent_state` map, so parent/child world ids
/// match the oracle exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TinyWorld {
    /// `sha256:` world id (hash of the identity doc).
    pub world_id: String,
    /// Four seats x two tiles in seat order.
    pub hands: [[u32; 2]; 4],
    /// Live wall (popped from the front per transition).
    pub live: Vec<u32>,
    /// Dead wall (carried through, usually empty in tiny).
    pub dead: Vec<u32>,
    /// `latent_state.step` when present.
    pub step: Option<u32>,
    /// `latent_state.turn` when present (`0..3`).
    pub turn: Option<u32>,
    /// `latent_state.last_action` when present.
    pub last_action: Option<u32>,
    /// `latent_state.corpus_idx` when present (preserved across transitions).
    pub corpus_idx: Option<u32>,
    /// Passthrough rules hash (`sha256:`).
    pub rules_hash: String,
    /// Passthrough observation hash (`sha256:`).
    pub observation_hash: String,
    /// Simulator snapshot (domain-prefixed chain link).
    pub snapshot: String,
}

impl TinyWorld {
    /// Validate all fields (digest shapes, tile ranges, seat ranges).
    pub fn validate(&self) -> Result<(), SearchError> {
        check_digest_shape(&self.world_id)?;
        check_digest_shape(&self.rules_hash)?;
        check_digest_shape(&self.observation_hash)?;
        if self.snapshot.is_empty() {
            return Err(SearchError::InvalidArg { detail: "snapshot must be non-empty" });
        }
        let mut seat = 0;
        while seat < 4 {
            let mut k = 0;
            while k < 2 {
                check_tile(self.hands[seat][k])?;
                k += 1;
            }
            seat += 1;
        }
        let mut i = 0;
        while i < self.live.len() {
            check_tile(self.live[i])?;
            i += 1;
        }
        let mut j = 0;
        while j < self.dead.len() {
            check_tile(self.dead[j])?;
            j += 1;
        }
        if let Some(turn) = self.turn {
            if turn >= 4 {
                return Err(SearchError::InvalidSeat { seat: turn });
            }
        }
        Ok(())
    }

    /// Actor to move (`gumbel_core.py:329-341`): `turn` when present and
    /// `0..3`, else `step.unwrap_or(0) % 4`.
    pub fn actor_to_move(&self) -> u32 {
        if let Some(turn) = self.turn {
            if turn < 4 {
                return turn;
            }
        }
        self.step.unwrap_or(0) % 4
    }

    /// Terminal (`gumbel_core.py:344-353`): step bound or live exhausted.
    /// The caller passes the search `max_depth`; the world step tracks the
    /// latent counter (kept separate so corpus worlds without `step` read 0).
    pub fn is_terminal(&self, max_depth: u32) -> bool {
        if self.step.unwrap_or(0) >= max_depth {
            return true;
        }
        if self.live.is_empty() {
            return true;
        }
        false
    }
}

/// World-id bytes for a (possibly successor) world: the `make_full_world`
/// identity doc (`belief/world.py:111-148`) canonicalized via the single
/// feed site. Only `Some` latent keys enter the map, so absent stays absent.
pub(crate) fn world_id_for(
    hands: &[[u32; 2]; 4],
    live: &[u32],
    dead: &[u32],
    step: Option<u32>,
    turn: Option<u32>,
    last_action: Option<u32>,
    corpus_idx: Option<u32>,
    rules_hash: &str,
    observation_hash: &str,
    snapshot: &str,
) -> Result<String, SearchError> {
    let mut latent = BTreeMap::new();
    if let Some(idx) = corpus_idx {
        latent.insert("corpus_idx".to_string(), serde_json::json!(idx));
    }
    if let Some(value) = step {
        latent.insert("step".to_string(), serde_json::json!(value));
    }
    if let Some(value) = turn {
        latent.insert("turn".to_string(), serde_json::json!(value));
    }
    if let Some(value) = last_action {
        latent.insert("last_action".to_string(), serde_json::json!(value));
    }
    let mut hands_json = Vec::with_capacity(4);
    let mut seat = 0;
    while seat < 4 {
        hands_json.push(serde_json::json!([hands[seat][0], hands[seat][1]]));
        seat += 1;
    }
    let mut doc = BTreeMap::new();
    doc.insert("concealed_hands".to_string(), serde_json::Value::Array(hands_json));
    doc.insert("dead_wall".to_string(), serde_json::json!(dead));
    doc.insert(
        "latent_state".to_string(),
        serde_json::Value::Object(latent.into_iter().collect()),
    );
    doc.insert("live_wall".to_string(), serde_json::json!(live));
    doc.insert("observation_hash".to_string(), serde_json::json!(observation_hash));
    doc.insert("rules_hash".to_string(), serde_json::json!(rules_hash));
    doc.insert("simulator_snapshot".to_string(), serde_json::json!(snapshot));
    let bytes = hydra_feed::canon::canonical_bytes(&doc, "search:world")
        .map_err(|err| SearchError::Canon { detail: err.to_string() })?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

/// Unified exact transition (`ismcts_search.py:95-136` + `gumbel_core.py:371-411`).
///
/// Pops one live tile (no-op when empty), keeps hands/dead, bumps
/// `step = parent.step.unwrap_or(0) + 1`, rotates `turn = (actor + 1) % 4`,
/// stamps `last_action = action`, preserves `corpus_idx`, passes hashes
/// through, and links `snapshot = f"{domain}:{parent_wid}:{action}:{step}"`.
/// `domain` keeps the salts (`ismcts` vs `gumbel`); anything else fails
/// closed so goldens never silently fork.
pub fn exact_transition(
    parent: &TinyWorld,
    actor: u32,
    action_id: u32,
    domain: &str,
) -> Result<TinyWorld, SearchError> {
    parent.validate()?;
    if actor >= 4 {
        return Err(SearchError::InvalidSeat { seat: actor });
    }
    if domain != DOMAIN_ISMCTS && domain != DOMAIN_GUMBEL {
        return Err(SearchError::InvalidArg { detail: "domain must be ismcts|gumbel" });
    }
    let base_step = parent.step.unwrap_or(0);
    let new_step =
        base_step.checked_add(1).ok_or(SearchError::InvalidArg { detail: "step overflow" })?;
    let new_turn = (actor + 1) % 4;
    let mut new_live = Vec::with_capacity(parent.live.len());
    let mut i = 0;
    while i < parent.live.len() {
        if i > 0 {
            new_live.push(parent.live[i]);
        }
        i += 1;
    }
    let snapshot = format!("{domain}:{}:{action_id}:{new_step}", parent.world_id);
    let world_id = world_id_for(
        &parent.hands,
        &new_live,
        &parent.dead,
        Some(new_step),
        Some(new_turn),
        Some(action_id),
        parent.corpus_idx,
        &parent.rules_hash,
        &parent.observation_hash,
        &snapshot,
    )?;
    Ok(TinyWorld {
        world_id,
        hands: parent.hands,
        live: new_live,
        dead: parent.dead.clone(),
        step: Some(new_step),
        turn: Some(new_turn),
        last_action: Some(action_id),
        corpus_idx: parent.corpus_idx,
        rules_hash: parent.rules_hash.clone(),
        observation_hash: parent.observation_hash.clone(),
        snapshot,
    })
}

/// Continuation sample (`ismcts_core.py:436-507`): two-action tilt by
/// `direction` (`0` favors `legal[0]` at `0.7`, `1` flips), longer sets
/// uniform; categorical walk over `rng_float` in `[0,1)` with last-action
/// fallback. `legal` must be sorted-unique (checked); single action still
/// consumes the float (oracle parity) and returns it.
pub fn continuation_sample(
    legal: &[u32],
    direction: u32,
    rng_float: f64,
) -> Result<u32, SearchError> {
    let sorted = check_legal(legal)?;
    if direction > 1 {
        return Err(SearchError::InvalidArg { detail: "direction must be 0|1" });
    }
    if !rng_float.is_finite() || rng_float < 0.0 || rng_float >= 1.0 {
        return Err(SearchError::InvalidArg { detail: "rng float must be in [0,1)" });
    }
    if sorted.len() == 1 {
        return Ok(sorted[0]);
    }
    if sorted.len() == 2 {
        let first_prob =
            if direction == 0 { 0.5 + CONTINUATION_BIAS } else { 0.5 - CONTINUATION_BIAS };
        if rng_float < first_prob {
            return Ok(sorted[0]);
        }
        return Ok(sorted[1]);
    }
    let width = sorted.len() as f64;
    if !width.is_finite() || width <= 0.0 {
        return Err(SearchError::NonFinite { context: "ismcts continuation width" });
    }
    let weight = 1.0 / width;
    let mut cumulative = 0.0f64;
    let mut idx = 0;
    while idx < sorted.len() {
        cumulative += weight;
        if rng_float < cumulative {
            return Ok(sorted[idx]);
        }
        idx += 1;
    }
    match sorted.last() {
        Some(last) => Ok(*last),
        None => Err(SearchError::EmptyLegal),
    }
}

// ---------------------------------------------------------------------------
// Batch JSON shape (one py call per search; no per-step crossing)
// ---------------------------------------------------------------------------

/// One batch world in the descent JSON envelope.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct BatchWorldJson {
    /// `sha256:` world id.
    pub(crate) world_id: String,
    /// Four seats x two tiles in seat order.
    pub(crate) hands: Vec<Vec<u32>>,
    /// Live wall tile ids.
    pub(crate) live: Vec<u32>,
    /// Dead wall tile ids (usually empty).
    pub(crate) dead: Vec<u32>,
    /// `latent_state.step` (`None` when absent).
    pub(crate) step: Option<u32>,
    /// `latent_state.turn` (`None` when absent).
    pub(crate) turn: Option<u32>,
    /// `latent_state.corpus_idx` (`None` when absent).
    pub(crate) corpus_idx: Option<u32>,
    /// True simulator snapshot (chain link, e.g. `tiny:{target}:{idx}`).
    /// Carried verbatim for fidelity (never re-hashed into the kept
    /// `world_id`); the successor snapshot derives from the parent
    /// `world_id`, never from this string.
    pub(crate) snapshot: String,
}

/// One leaf override (read-only torch vector in batch, keyed by world id).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct LeafOverrideJson {
    /// `sha256:` world id needing the override.
    pub(crate) world_id: String,
    /// Four-seat override vector (finite).
    pub(crate) vector: Vec<f64>,
}

/// Full descent batch envelope: precomputed worlds, per-step keys/dirs,
/// CTR floats, frozen params, and optional leaf overrides.
#[derive(Debug, Clone, Deserialize)]
struct DescentBatchJson {
    /// One world per simulation (`len == max_sims`, oracle sample order).
    worlds: Vec<BatchWorldJson>,
    /// Passthrough rules hash for successors.
    rules_hash: String,
    /// Passthrough observation hash for successors.
    observation_hash: String,
    /// Root info key (`sha256:`).
    root_key: String,
    /// Root legal ids (sorted-unique checked).
    root_legal: Vec<u32>,
    /// Root seat `0..3`.
    root_seat: u32,
    /// Per-sim per-step info keys (`S x D`; `""` where actor != root).
    step_keys: Vec<Vec<String>>,
    /// Per-sim per-step policy directions (`S x D`; `0|1`).
    policy_dirs: Vec<Vec<u32>>,
    /// CTR floats in consumption order (`len >= S * D`).
    rng_floats: Vec<f64>,
    /// UCT constant (finite `> 0`).
    uct_c: f64,
    /// Descent depth `1..=32`.
    max_depth: u32,
    /// Simulation budget (`>= 1`, equals `worlds.len()`).
    max_sims: u32,
    /// Transition cap (`None` = unbounded).
    max_transitions: Option<u64>,
    /// Model-call cap (`None` = unbounded; exhaustion falls back to terminal).
    max_model_calls: Option<u64>,
    /// Leaf salt (`candidate1` in ISMCTS; kept, never defaulted).
    candidate_id: String,
    /// Snapshot salt (`ismcts` or `gumbel`; kept, never defaulted).
    domain: String,
    /// Tie-break arm name.
    tie_break: String,
    /// Optional read-only leaf overrides (possibly empty).
    leaf_overrides: Vec<LeafOverrideJson>,
}

/// Batch descent outcome (owned, detach-safe): selection, per-candidate
/// means/visits in sorted-candidate order, canon tree digest, counters, and
/// a key-ordered tree dump for golden debug.
#[derive(Debug, Clone, PartialEq)]
pub struct DescentOut {
    /// Selected action id (max scalarized mean, `1e-12` eps + tie arm).
    pub selected_id: u32,
    /// Candidate ids in sorted order.
    pub candidate_ids: Vec<u32>,
    /// Mean 4-vectors per candidate (unvisited carry the model placeholder).
    pub value_vectors: Vec<[f64; 4]>,
    /// Visits per candidate.
    pub visits: Vec<u64>,
    /// Canon digest over the sorted tree dump.
    pub tree_digest: String,
    /// Completed simulations.
    pub sims_run: u64,
    /// Completed transitions.
    pub transitions: u64,
    /// Model leaf calls (terminal fallbacks excluded).
    pub model_calls: u64,
    /// Tree node count.
    pub tree_nodes: u64,
    /// CTR floats consumed.
    pub floats_used: u64,
    /// Debug dump: `(key, visits, [(action, visits, sum)])`, key-ordered.
    pub tree: Vec<(String, u64, Vec<(u32, u64, [f64; 4])>)>,
}

impl BatchWorldJson {
    /// Materialize one validated arena world (hands `4 x 2`, tiles `< 136`).
    pub(crate) fn materialize(&self) -> Result<TinyWorld, SearchError> {
        check_digest_shape(&self.world_id)?;
        if self.hands.len() != 4 {
            return Err(SearchError::InvalidArg { detail: "hands must hold 4 seats" });
        }
        let mut hands = [[0u32; 2]; 4];
        let mut seat = 0;
        while seat < 4 {
            if self.hands[seat].len() != 2 {
                return Err(SearchError::InvalidArg { detail: "each hand must hold 2 tiles" });
            }
            check_tile(self.hands[seat][0])?;
            check_tile(self.hands[seat][1])?;
            hands[seat][0] = self.hands[seat][0];
            hands[seat][1] = self.hands[seat][1];
            seat += 1;
        }
        let mut i = 0;
        while i < self.live.len() {
            check_tile(self.live[i])?;
            i += 1;
        }
        let mut j = 0;
        while j < self.dead.len() {
            check_tile(self.dead[j])?;
            j += 1;
        }
        if let Some(turn) = self.turn {
            if turn >= 4 {
                return Err(SearchError::InvalidSeat { seat: turn });
            }
        }
        if self.snapshot.is_empty() {
            return Err(SearchError::InvalidArg { detail: "snapshot must be non-empty" });
        }
        Ok(TinyWorld {
            world_id: self.world_id.clone(),
            hands,
            live: self.live.clone(),
            dead: self.dead.clone(),
            step: self.step,
            turn: self.turn,
            last_action: None,
            corpus_idx: self.corpus_idx,
            rules_hash: String::new(),
            observation_hash: String::new(),
            snapshot: self.snapshot.clone(),
        })
    }
}

/// Canon tree digest over the sorted dump (byte-ordered keys, never `hash()`).
fn tree_digest(tree: &BTreeMap<String, IsNode>) -> Result<String, SearchError> {
    let mut doc = BTreeMap::new();
    for (key, node) in tree {
        let mut arms = BTreeMap::new();
        let mut i = 0;
        while i < node.arms.len() {
            let (action, stats) = node.arms[i];
            let mut entry = BTreeMap::new();
            entry.insert("visits".to_string(), serde_json::json!(stats.visits));
            entry.insert(
                "value_sum".to_string(),
                serde_json::json!([
                    stats.value_sum[0],
                    stats.value_sum[1],
                    stats.value_sum[2],
                    stats.value_sum[3],
                ]),
            );
            arms.insert(action.to_string(), serde_json::Value::Object(entry.into_iter().collect()));
            i += 1;
        }
        let mut node_doc = BTreeMap::new();
        node_doc.insert("visits".to_string(), serde_json::json!(node.visits));
        node_doc.insert("arms".to_string(), serde_json::Value::Object(arms.into_iter().collect()));
        doc.insert(key.clone(), serde_json::Value::Object(node_doc.into_iter().collect()));
    }
    let bytes = hydra_feed::canon::canonical_bytes(&doc, "search:ismcts_tree")
        .map_err(|err| SearchError::Canon { detail: err.to_string() })?;
    Ok(hydra_feed::digest::sha256_hex(&bytes))
}

/// Final root selection (`ismcts_search.py:442-480`): max scalarized mean
/// with `1e-12` eps; `lowest_action_id` breaks ties by min id, other arms
/// keep the first best (oracle parity). Unvisited candidates carry the model
/// placeholder over the `unvisited:{aid}` tag (kept salt, no belief lookup).
fn final_select(
    root: &IsNode,
    candidates: &[u32],
    root_seat: u32,
    tie: TieBreak,
    candidate_id: &str,
) -> Result<(u32, Vec<[f64; 4]>, Vec<u64>), SearchError> {
    if root_seat >= 4 {
        return Err(SearchError::InvalidSeat { seat: root_seat });
    }
    if candidates.is_empty() {
        return Err(SearchError::EmptyLegal);
    }
    let mut best_id = candidates[0];
    let mut best_q = f64::NEG_INFINITY;
    let mut seen = false;
    let mut i = 0;
    while i < candidates.len() {
        let aid = candidates[i];
        if let Some(vector) = root.mean_vector(aid) {
            let q = vector[root_seat as usize];
            if !q.is_finite() {
                return Err(SearchError::NonFinite { context: "ismcts final q" });
            }
            if !seen {
                best_id = aid;
                best_q = q;
                seen = true;
            } else if q > best_q + FINAL_EPS {
                best_id = aid;
                best_q = q;
            } else if (q - best_q).abs() <= FINAL_EPS
                && tie == TieBreak::LowestActionId
                && aid < best_id
            {
                best_id = aid;
                best_q = q;
            }
        }
        i += 1;
    }
    if !seen {
        best_id = candidates[0];
    }
    let mut vectors = Vec::with_capacity(candidates.len());
    let mut visits = Vec::with_capacity(candidates.len());
    let mut k = 0;
    while k < candidates.len() {
        let aid = candidates[k];
        match root.mean_vector(aid) {
            Some(vector) => {
                for value in vector {
                    if !value.is_finite() {
                        return Err(SearchError::NonFinite { context: "ismcts final mean" });
                    }
                }
                vectors.push(vector);
            }
            None => {
                let tag = format!("unvisited:{aid}");
                vectors.push(placeholder_vector(&tag, candidate_id)?);
            }
        }
        let n = root.stats(aid).map(|stats| stats.visits).unwrap_or(0);
        visits.push(n);
        k += 1;
    }
    Ok((best_id, vectors, visits))
}

/// Root-missing fallback (`ismcts_search.py:432-441` shape): first legal
/// wins with model placeholders over the `unvisited:{aid}` tags and zero
/// visits. The oracle would consult belief here; the arena stays total by
/// hashing the kept tags instead (documented, fail-closed elsewhere).
fn fallback_select(
    candidates: &[u32],
    candidate_id: &str,
) -> Result<(u32, Vec<[f64; 4]>, Vec<u64>), SearchError> {
    if candidates.is_empty() {
        return Err(SearchError::EmptyLegal);
    }
    let mut vectors = Vec::with_capacity(candidates.len());
    let mut visits = Vec::with_capacity(candidates.len());
    let mut i = 0;
    while i < candidates.len() {
        let tag = format!("unvisited:{}", candidates[i]);
        vectors.push(placeholder_vector(&tag, candidate_id)?);
        visits.push(0);
        i += 1;
    }
    Ok((candidates[0], vectors, visits))
}

/// Run one batch ISMCTS search: pure replay of precomputed worlds/keys/dirs/
/// floats through UCT descent, unified transitions, and deduped leaf vectors
/// with backup (`ismcts_search.py:276-350`), then scalarized selection.
pub fn ismcts_search_batch(batch_json: &[u8]) -> Result<DescentOut, SearchError> {
    if batch_json.is_empty() {
        return Err(SearchError::InvalidArg { detail: "batch JSON must be non-empty" });
    }
    let batch: DescentBatchJson = serde_json::from_slice(batch_json)
        .map_err(|_| SearchError::InvalidArg { detail: "batch JSON malformed" })?;
    let sorted_legal = check_legal(&batch.root_legal)?;
    if batch.root_seat >= 4 {
        return Err(SearchError::InvalidSeat { seat: batch.root_seat });
    }
    if !batch.uct_c.is_finite() || batch.uct_c <= 0.0 {
        return Err(SearchError::InvalidArg { detail: "uct_c must be finite >0" });
    }
    if batch.max_depth == 0 || batch.max_depth > 32 {
        return Err(SearchError::InvalidArg { detail: "max_depth must be 1..32" });
    }
    if batch.max_sims == 0 {
        return Err(SearchError::InvalidArg { detail: "max_sims must be >= 1" });
    }
    if batch.candidate_id.is_empty() {
        return Err(SearchError::InvalidArg { detail: "candidate_id must be non-empty" });
    }
    if batch.domain != DOMAIN_ISMCTS && batch.domain != DOMAIN_GUMBEL {
        return Err(SearchError::InvalidArg { detail: "domain must be ismcts|gumbel" });
    }
    let tie = TieBreak::parse(&batch.tie_break)?;
    check_digest_shape(&batch.rules_hash)?;
    check_digest_shape(&batch.observation_hash)?;
    check_tree_key(&batch.root_key)?;
    let sims = batch.max_sims as usize;
    if batch.worlds.len() != sims {
        return Err(SearchError::InvalidArg { detail: "worlds len must equal max_sims" });
    }
    if batch.step_keys.len() != sims {
        return Err(SearchError::InvalidArg { detail: "step_keys rows must equal max_sims" });
    }
    if batch.policy_dirs.len() != sims {
        return Err(SearchError::InvalidArg { detail: "policy_dirs rows must equal max_sims" });
    }
    let depth = batch.max_depth as usize;
    let mut s = 0;
    while s < sims {
        if batch.step_keys[s].len() != depth {
            return Err(SearchError::InvalidArg {
                detail: "step_keys cols must equal max_depth",
            });
        }
        if batch.policy_dirs[s].len() != depth {
            return Err(SearchError::InvalidArg {
                detail: "policy_dirs cols must equal max_depth",
            });
        }
        let mut k = 0;
        while k < depth {
            if batch.policy_dirs[s][k] > 1 {
                return Err(SearchError::InvalidArg { detail: "direction must be 0|1" });
            }
            k += 1;
        }
        s += 1;
    }
    // Float layout: one CTR draw per policy (non-root) step, in `(sim,
    // step)` consumption order, NO placeholders for root steps or belief
    // draws. The exact demand is counted below by a dry run over the
    // materialized worlds (mirrors the replay loop step-for-step, including
    // sample-then-gate order and the transitions cap); the per-read
    // `exhausted` check inside replay stays the ultimate backstop.
    let params = IsmctsParams {
        uct_c: batch.uct_c,
        max_depth: batch.max_depth,
        max_sims: batch.max_sims,
        tie,
    };
    params.validate()?;
    for value in &batch.rng_floats {
        if !value.is_finite() || *value < 0.0 || *value >= 1.0 {
            return Err(SearchError::InvalidArg { detail: "rng float must be in [0,1)" });
        }
    }
    // Leaf overrides: validated once, looked up by world id (read-only).
    let mut overrides: BTreeMap<String, [f64; 4]> = BTreeMap::new();
    let mut o = 0;
    while o < batch.leaf_overrides.len() {
        let entry = &batch.leaf_overrides[o];
        check_digest_shape(&entry.world_id)?;
        if entry.vector.len() != 4 {
            return Err(SearchError::InvalidArg { detail: "override vector must hold 4" });
        }
        let mut vector = [0.0f64; 4];
        let mut k = 0;
        while k < 4 {
            let value = entry.vector[k];
            if !value.is_finite() {
                return Err(SearchError::NonFinite { context: "ismcts override vector" });
            }
            vector[k] = value;
            k += 1;
        }
        if overrides.contains_key(&entry.world_id) {
            return Err(SearchError::InvalidArg { detail: "duplicate leaf override" });
        }
        overrides.insert(entry.world_id.clone(), vector);
        o += 1;
    }
    // Materialize worlds with shared passthrough hashes.
    let mut worlds: Vec<TinyWorld> = Vec::with_capacity(sims);
    let mut w = 0;
    while w < sims {
        let mut world = batch.worlds[w].materialize()?;
        world.rules_hash = batch.rules_hash.clone();
        world.observation_hash = batch.observation_hash.clone();
        world.validate()?;
        worlds.push(world);
        w += 1;
    }
    // Exact float demand: dry-run the consumption walk over the materialized
    // worlds, mirroring the replay loop below step-for-step — same
    // `is_terminal` stop, same sample-then-gate order (a policy step counts
    // its float even when the transitions gate breaks right after, and the
    // breaking read still counts in `floats_used`), same live-deplete and
    // `step+1`/`turn+1` advance, same post-cap stop. Shapes that end before
    // `max_depth` (live empties at 4 in tiny, so 48x6 needs 144 not 240)
    // demand only what they consume; the per-read `exhausted` check below
    // stays the ultimate backstop if walk and replay ever drift.
    let mut need_floats: usize = 0;
    let mut walked_trans: u64 = 0;
    let mut ws = 0;
    while ws < sims {
        let start = &worlds[ws];
        let mut actor = start.actor_to_move();
        let mut live_len = start.live.len();
        let mut stepped = start.step.unwrap_or(0);
        let mut wd = 0;
        while wd < depth {
            if stepped >= batch.max_depth || live_len == 0 {
                break;
            }
            if actor >= 4 {
                return Err(SearchError::InvalidSeat { seat: actor });
            }
            if actor == batch.root_seat {
                if batch.step_keys[ws][wd].is_empty() {
                    return Err(SearchError::InvalidArg {
                        detail: "root step needs info key",
                    });
                }
            } else {
                need_floats = need_floats.checked_add(1).ok_or(SearchError::InvalidArg {
                    detail: "batch dims overflow",
                })?;
            }
            if let Some(cap) = batch.max_transitions {
                if walked_trans >= cap {
                    break;
                }
            }
            if live_len > 0 {
                live_len -= 1;
            }
            stepped = stepped.checked_add(1).ok_or(SearchError::InvalidArg {
                detail: "step overflow",
            })?;
            actor = (actor + 1) % 4;
            walked_trans = walked_trans.checked_add(1).ok_or(SearchError::InvalidArg {
                detail: "transition counter overflow",
            })?;
            // Post-transition cap break mirrors replay below: stop the descent
            // immediately once the cap is hit, so the next policy float is
            // NOT counted (deadline shape needs 7 floats, not 8).
            if let Some(cap) = batch.max_transitions {
                if walked_trans >= cap {
                    break;
                }
            }
            wd += 1;
        }
        if let Some(cap) = batch.max_transitions {
            if walked_trans >= cap {
                break;
            }
        }
        ws += 1;
    }
    if batch.rng_floats.len() < need_floats {
        return Err(SearchError::InvalidArg { detail: "rng_floats too short" });
    }
    let mut tree: BTreeMap<String, IsNode> = BTreeMap::new();
    let mut transitions: u64 = 0;
    let mut model_calls: u64 = 0;
    let mut sims_run: u64 = 0;
    let mut float_ptr: usize = 0;
    let mut sim = 0;
    while sim < sims {
        let mut current = worlds[sim].clone();
        // Path holds only root visits `(key, action)`; policy steps hold no
        // node, so backup skips them exactly like the oracle.
        let mut path: Vec<(String, u32)> = Vec::new();
        let mut step_idx: usize = 0;
        while step_idx < depth && !current.is_terminal(batch.max_depth) {
            let actor = current.actor_to_move();
            if actor >= 4 {
                return Err(SearchError::InvalidSeat { seat: actor });
            }
            let action = if actor == batch.root_seat {
                let key = batch.step_keys[sim][step_idx].clone();
                if key.is_empty() {
                    return Err(SearchError::InvalidArg {
                        detail: "root step needs info key",
                    });
                }
                check_tree_key(&key)?;
                let node = tree.entry(key.clone()).or_insert_with(IsNode::new);
                let picked = uct_select(node, &sorted_legal, batch.root_seat, &params)?;
                path.push((key, picked));
                picked
            } else {
                let direction = batch.policy_dirs[sim][step_idx];
                if float_ptr >= batch.rng_floats.len() {
                    return Err(SearchError::InvalidArg { detail: "rng_floats exhausted" });
                }
                let float = batch.rng_floats[float_ptr];
                float_ptr += 1;
                continuation_sample(&sorted_legal, direction, float)?
            };
            // Budget gate before the transition (oracle parity: break leaves
            // the partial path for backup with the leaf below).
            if let Some(cap) = batch.max_transitions {
                if transitions >= cap {
                    break;
                }
            }
            current = exact_transition(&current, actor, action, &batch.domain)?;
            transitions =
                transitions.checked_add(1).ok_or(SearchError::InvalidArg {
                    detail: "transition counter overflow",
                })?;
            step_idx += 1;
            if let Some(cap) = batch.max_transitions {
                if transitions >= cap {
                    break;
                }
            }
        }
        // Leaf: terminal when stopped terminal; else model unless the call
        // budget is exhausted (then terminal fallback). Overrides win when
        // the leaf world id is pinned (read-only torch path).
        let leaf_is_terminal = current.is_terminal(batch.max_depth);
        let allow_model = match batch.max_model_calls {
            Some(cap) => model_calls < cap,
            None => true,
        };
        let vector = match overrides.get(&current.world_id) {
            Some(vector) => *vector,
            None => {
                if leaf_is_terminal || !allow_model {
                    terminal_vector(&current.world_id)?
                } else {
                    let vector = model_vector(&current.world_id, &batch.candidate_id)?;
                    model_calls =
                        model_calls.checked_add(1).ok_or(SearchError::InvalidArg {
                            detail: "model counter overflow",
                        })?;
                    vector
                }
            }
        };
        for value in vector {
            if !value.is_finite() {
                return Err(SearchError::NonFinite { context: "ismcts backup vector" });
            }
        }
        let mut p = 0;
        while p < path.len() {
            let (key, action) = &path[p];
            match tree.get_mut(key) {
                Some(node) => {
                    node.backup(*action, vector);
                }
                None => {
                    return Err(SearchError::InvalidArg { detail: "backup node missing" });
                }
            }
            p += 1;
        }
        sims_run =
            sims_run.checked_add(1).ok_or(SearchError::InvalidArg {
                detail: "sim counter overflow",
            })?;
        // Post-sim budget: transitions stop the search (oracle parity);
        // model exhaustion never stops it (terminal fallbacks continue).
        if let Some(cap) = batch.max_transitions {
            if transitions >= cap {
                break;
            }
        }
        sim += 1;
    }
    let (selected_id, value_vectors, visits) = match tree.get(&batch.root_key) {
        Some(node) if !node.arms.is_empty() => {
            final_select(node, &sorted_legal, batch.root_seat, tie, &batch.candidate_id)?
        }
        _ => fallback_select(&sorted_legal, &batch.candidate_id)?,
    };
    let digest = tree_digest(&tree)?;
    let tree_nodes = tree.len() as u64;
    let mut dump: Vec<(String, u64, Vec<(u32, u64, [f64; 4])>)> = Vec::new();
    for (key, node) in &tree {
        let mut arms: Vec<(u32, u64, [f64; 4])> = Vec::new();
        let mut i = 0;
        while i < node.arms.len() {
            let (action, stats) = node.arms[i];
            arms.push((action, stats.visits, stats.value_sum));
            i += 1;
        }
        arms.sort_by(|left, right| left.0.cmp(&right.0));
        dump.push((key.clone(), node.visits, arms));
    }
    let floats_used = float_ptr as u64;
    Ok(DescentOut {
        selected_id,
        candidate_ids: sorted_legal,
        value_vectors,
        visits,
        tree_digest: digest,
        sims_run,
        transitions,
        model_calls,
        tree_nodes,
        floats_used,
        tree: dump,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARENT_WID: &str =
        "sha256:b60a48e634b665241d1e1f753189ab2b59c59ce05a6ce73f5a7b265fa9cfa457";
    const RULES: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OBS: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const ROOT_KEY: &str =
        "sha256:0962179d2a50471795ff4cff24fbd6566c67f94bf12270c0d66684c860c472d1";

    fn parent_world() -> TinyWorld {
        TinyWorld {
            world_id: PARENT_WID.to_string(),
            hands: [[0, 1], [2, 3], [4, 5], [6, 7]],
            live: vec![8, 9, 10, 11],
            dead: vec![],
            step: None,
            turn: None,
            last_action: None,
            corpus_idx: Some(0),
            rules_hash: RULES.to_string(),
            observation_hash: OBS.to_string(),
            snapshot: "snap_test".to_string(),
        }
    }

    #[test]
    fn model_vector_golden() {
        let vector = model_vector(PARENT_WID, "candidate1").expect("model");
        assert_eq!(vector, [0.45, 0.52, 0.49, 0.32]);
    }

    #[test]
    fn terminal_vector_golden() {
        let vector = terminal_vector(PARENT_WID).expect("terminal");
        assert_eq!(vector, [0.06, 0.08000000000000002, 0.5, 0.9199999999999999]);
    }

    #[test]
    fn transition_unifies_with_domain_salts() {
        let ismcts = exact_transition(&parent_world(), 0, 2, "ismcts").expect("ismcts");
        assert_eq!(
            ismcts.snapshot,
            "ismcts:sha256:b60a48e634b665241d1e1f753189ab2b59c59ce05a6ce73f5a7b265fa9cfa457:2:1"
        );
        assert_eq!(
            ismcts.world_id,
            "sha256:8924de4cf663decec01c4f1e454422040a3fd3ce297267ebf65ac6ce3d28ffd0"
        );
        assert_eq!(ismcts.live, vec![9, 10, 11]);
        assert_eq!((ismcts.step, ismcts.turn, ismcts.last_action), (Some(1), Some(1), Some(2)));
        let gumbel = exact_transition(&parent_world(), 0, 2, "gumbel").expect("gumbel");
        assert_eq!(
            gumbel.world_id,
            "sha256:1b9107f18cc23ce5d29807da9945de68f84adba9e73e52dfb65a71f50d515344"
        );
        assert_ne!(ismcts.world_id, gumbel.world_id);
        assert_eq!(gumbel.live, ismcts.live);
    }

    #[test]
    fn actor_and_terminal_match_oracle() {
        let parent = parent_world();
        assert_eq!(parent.actor_to_move(), 0);
        assert!(!parent.is_terminal(6));
        let child = exact_transition(&parent, 0, 2, "ismcts").expect("child");
        assert_eq!(child.actor_to_move(), 1);
        assert!(!child.is_terminal(6));
        let mut deep = child;
        let mut step = 1;
        while step < 4 {
            let actor = deep.actor_to_move();
            deep = exact_transition(&deep, actor, 0, "ismcts").expect("step");
            step += 1;
        }
        assert!(deep.live.is_empty());
        assert!(deep.is_terminal(6));
    }

    #[test]
    fn continuation_tilt_matches_oracle() {
        assert_eq!(continuation_sample(&[0, 2], 0, 0.69).expect("tilt"), 0);
        assert_eq!(continuation_sample(&[0, 2], 0, 0.71).expect("tilt"), 2);
        assert_eq!(continuation_sample(&[0, 2], 1, 0.29).expect("tilt"), 0);
        assert_eq!(continuation_sample(&[0, 2], 1, 0.31).expect("tilt"), 2);
        assert_eq!(continuation_sample(&[7], 0, 0.99).expect("single"), 7);
    }

    #[test]
    fn forbidden_tree_key_rejected() {
        assert!(check_tree_key("sha256:world_id_leak").is_err());
        assert!(check_tree_key("not-a-digest").is_err());
    }

    #[test]
    fn bad_domain_and_tiles_fail_closed() {
        assert!(exact_transition(&parent_world(), 0, 2, "nope").is_err());
        assert!(exact_transition(&parent_world(), 4, 2, "ismcts").is_err());
        assert!(model_vector(PARENT_WID, "").is_err());
        assert!(model_vector("nope", "candidate1").is_err());
    }

    #[test]
    fn local_shapes_match_oracle() {
        let zero = local_model_vector(PARENT_WID, "model").expect("local model");
        assert_eq!(
            zero,
            [
                -0.3271839475089646,
                -0.2868390936140993,
                0.033752956435492476,
                0.5802700846875715
            ]
        );
        let mut sum = 0.0f64;
        let mut i = 0;
        while i < 4 {
            sum += zero[i];
            i += 1;
        }
        assert!(sum.abs() < 1e-12);
        let settle = local_terminal_vector(
            &[vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
            &[8, 9, 10, 11],
        )
        .expect("local terminal");
        assert_eq!(settle, [-0.56295, -0.21235000000000004, 0.18764999999999998, 0.58765]);
        let off = exhaustive_path_offset(PARENT_WID, &[2, 0]).expect("offset");
        assert_eq!(off, -0.010000000000000009);
        assert!(local_terminal_vector(&[vec![0]], &[8]).is_err());
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn batch_descent_runs_with_counters() {
        let world = serde_json::json!({
            "world_id": PARENT_WID,
            "hands": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "live": [8, 9, 10, 11],
            "dead": [],
            "step": null,
            "turn": null,
            "corpus_idx": 0,
            "snapshot": "snap_test"
        });
        let worlds = vec![world.clone(), world.clone(), world.clone(), world];
        let keys_row = vec![ROOT_KEY.to_string(); 4];
        let batch = serde_json::json!({
            "worlds": worlds,
            "rules_hash": RULES,
            "observation_hash": OBS,
            "root_key": ROOT_KEY,
            "root_legal": [0, 2],
            "root_seat": 0,
            "step_keys": [keys_row.clone(), keys_row.clone(), keys_row.clone(), keys_row],
            "policy_dirs": [vec![0, 0, 0, 0], vec![1, 1, 1, 1], vec![0, 0, 0, 0], vec![1, 1, 1, 1]],
            "rng_floats": vec![0.1; 32],
            "uct_c": 1.41421356237,
            "max_depth": 4,
            "max_sims": 4,
            "max_transitions": 64,
            "max_model_calls": 24,
            "candidate_id": "candidate1",
            "domain": "ismcts",
            "tie_break": "lowest_action_id",
            "leaf_overrides": []
        });
        let bytes = serde_json::to_vec(&batch).expect("json");
        let out = ismcts_search_batch(&bytes).expect("descent");
        assert_eq!(out.sims_run, 4);
        assert_eq!(out.transitions, 16);
        assert_eq!(out.tree_nodes, 1);
        assert!(out.tree_digest.starts_with("sha256:"));
        assert!([0, 2].contains(&out.selected_id));
        assert_eq!(out.candidate_ids, vec![0, 2]);
        assert_eq!(out.value_vectors.len(), 2);
        assert_eq!(out.visits.iter().sum::<u64>(), 4);
    }
    #[test]
    fn early_terminal_sims_demand_no_floats() {
        // Empty live wall: every sim is terminal at step 0, so the exact
        // dry-run demand is 0 floats (the 48x6 shape needs 144, not 240,
        // for the same live-depletes-at-4 reason).
        let world = serde_json::json!({
            "world_id": PARENT_WID,
            "hands": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "live": [],
            "dead": [],
            "step": null,
            "turn": null,
            "corpus_idx": 0,
            "snapshot": "snap_test"
        });
        let keys_row = vec![ROOT_KEY.to_string(); 6];
        let dirs_row = vec![0; 6];
        let batch = serde_json::json!({
            "worlds": [world],
            "rules_hash": RULES,
            "observation_hash": OBS,
            "root_key": ROOT_KEY,
            "root_legal": [0, 2],
            "root_seat": 0,
            "step_keys": [keys_row],
            "policy_dirs": [dirs_row],
            "rng_floats": [],
            "uct_c": 1.41421356237,
            "max_depth": 6,
            "max_sims": 1,
            "max_transitions": 256,
            "max_model_calls": 48,
            "candidate_id": "candidate1",
            "domain": "ismcts",
            "tie_break": "lowest_action_id",
            "leaf_overrides": []
        });
        let bytes = serde_json::to_vec(&batch).expect("json");
        let out = ismcts_search_batch(&bytes).expect("descent");
        assert_eq!((out.sims_run, out.transitions, out.floats_used), (1, 0, 0));
        assert_eq!(out.tree_nodes, 0);
    }
    #[test]
    fn deadline_cap_counts_breaking_float_only_once() {
        // Deadline shape (depth 8, trans cap 10, live 4 per world): sim0
        // draws 3 floats / 4 transitions, sim1 draws 3 / 4 (total 8), sim2
        // draws 1 (root + 1 policy) then the post-transition cap break
        // stops the descent — exact demand is 7 floats, not 8. The dry run
        // must mirror the replay's inner post-increment break.
        let world = serde_json::json!({
            "world_id": PARENT_WID,
            "hands": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "live": [8, 9, 10, 11],
            "dead": [],
            "step": null,
            "turn": null,
            "corpus_idx": 0,
            "snapshot": "snap_test"
        });
        let worlds = vec![world.clone(), world.clone(), world];
        let mut keys_row = vec![String::new(); 8];
        keys_row[0] = ROOT_KEY.to_string();
        let dirs_row = vec![0; 8];
        let batch = serde_json::json!({
            "worlds": worlds,
            "rules_hash": RULES,
            "observation_hash": OBS,
            "root_key": ROOT_KEY,
            "root_legal": [0, 2],
            "root_seat": 0,
            "step_keys": [keys_row.clone(), keys_row.clone(), keys_row],
            "policy_dirs": [dirs_row.clone(), dirs_row.clone(), dirs_row],
            "rng_floats": vec![0.1; 7],
            "uct_c": 1.41421356237,
            "max_depth": 8,
            "max_sims": 3,
            "max_transitions": 10,
            "max_model_calls": 48,
            "candidate_id": "candidate1",
            "domain": "ismcts",
            "tie_break": "lowest_action_id",
            "leaf_overrides": []
        });
        let bytes = serde_json::to_vec(&batch).expect("json");
        let out = ismcts_search_batch(&bytes).expect("descent");
        assert_eq!((out.sims_run, out.transitions, out.floats_used), (3, 10, 7));
        assert_eq!(out.tree_nodes, 1);
    }
}
