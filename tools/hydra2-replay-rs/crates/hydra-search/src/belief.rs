//! `belief`: deterministic packet-successor enumeration + CTR-exact sampling kernels.
//!
//! Ports the three belief hot cores behind firewall-safe scalar/digest shapes
//! (info-keys only — no world blobs, hands, or walls ever cross):
//!
//! - `belief/kernel.py:100-116` (`_cached_successor_refs`): successor/delta
//!   digest shapes (`world_succ:sha(parent:tile:aid)[:16]`,
//!   `delta:sha(delta:parent:tile)[:16]`).
//! - `belief/kernel.py:119-130` (`_cached_observation_hash`): observation hash
//!   over canon `{"packet_seq","tile"}`.
//! - `belief/kernel.py:155-260` (`enumerate_next`): exactly 2 successors per
//!   `(parent, action)` (`tile = 8+idx`, `seq = 100+idx`,
//!   `actor = (root+1+idx) % 4`, `prob = 0.5`, `log_phys = ln(0.5)`,
//!   `log_policy = 0`); packet/chain/observation hashes over canon bytes via
//!   the single `feed::canon` site (canon-wins B3).
//! - `belief/natural.py:298-336` (`sample_natural` index loop): `K == 1`
//!   yields `0` consuming no stream, else one rejection-sampled `below(K)`
//!   per draw.
//! - `belief/sampled_kernel.py:95-166` (`enumerate_sampled` draw loop):
//!   `u = random_float()` over the frame categorical, draw `i` of packet `e`
//!   carries `raw_weight = P(e) / L`.
//! - `contracts/randomness.py:398-468` (`RandomStream` CTR): `DOMAIN =
//!   b"hydra2_ctr_v1\x00"`, `BLOCK = 32`, `block(i) = sha256(DOMAIN ||
//!   len(seed)BE32 || seed || iBE64)`; `random_float = u64BE / 2^64`;
//!   `random_below` is rejection sampling with `nbytes =
//!   (bound-1).bit_length // 8 + 1`, `limit = 256^nbytes // bound * bound`.
//!
//! RNG discipline: `rng.rs` Lemire `below` stays the ONLY path for NEW Philox
//! segment/scenario streams (B1/B2 — never Gumbels, never splits). This module
//! replicates the CTR oracle VERBATIM (like the B1 sha-Gumbels) so
//! bridge==oracle bit-exact on belief draws; it NEVER draws a Gumbel and
//! NEVER splits a corpus.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::SearchError;

/// CTR domain tag (`RandomStream._DOMAIN`, `randomness.py:405`).
pub const CTR_DOMAIN: &[u8] = b"hydra2_ctr_v1\x00";
/// CTR block width in bytes (`RandomStream._BLOCK`, `randomness.py:406`).
pub const CTR_BLOCK: u64 = 32;
/// `2^64` as `f64` (matches the `random_float` divisor, `randomness.py:449`).
pub const CTR_U64_DENOM: f64 = 18_446_744_073_709_551_616.0;
/// First discard tile (`kernel.py:198`: `tile = 8 + idx`).
pub const PACKET_TILE_BASE: u32 = 8;
/// First packet sequence (`kernel.py:199`: `seq = 100 + idx`).
pub const PACKET_SEQ_BASE: u32 = 100;
/// Successors per `(parent, action)` (`kernel.py:197`: exactly 2).
pub const PACKET_SUCCESSORS: u32 = 2;
/// Per-successor physical mass (`kernel.py:225`: uniform `0.5` each).
pub const PACKET_PROB: f64 = 0.5;
/// Packet mass gate (`NaturalPacketKernel` default `kernel_tolerance = 1e-9`).
pub const PACKET_MASS_TOL: f64 = 1e-9;
/// Synthetic game id (`_make_public_discard_event` default `game_tiny_001`).
pub const PACKET_GAME_ID: &str = "game_tiny_001";
/// Synthetic schema hash (`kernel.py:191`: `"sha256:" + "c" * 64`).
pub const PACKET_SCHEMA_HASH: &str = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

/// Lowercase hex of a digest (nibble loop, no allocation beyond the string).
fn hex_of(digest: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(digest.len() * 2);
    let mut i = 0;
    while i < digest.len() {
        let byte = digest[i];
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0F) as usize] as char);
        i += 1;
    }
    out
}

/// Canonical `sha256:<hex>` of a staged JSON map (canon-wins: single site).
fn canon_digest(map: &BTreeMap<String, serde_json::Value>, record: &str) -> Result<String, SearchError> {
    let bytes = hydra_feed::canon::canonical_bytes(map, record)
        .map_err(|err| SearchError::Canon { detail: err.to_string() })?;
    Ok(format!("sha256:{}", hex_of(Sha256::digest(bytes.as_slice()).as_slice())))
}

// ---------------------------------------------------------------------------
// Counter stream: verbatim `RandomStream` replication (seekable, replayable)
// ---------------------------------------------------------------------------

/// One CTR block (`_block_at`, `randomness.py:462-468`).
pub fn ctr_block(seed: &[u8], index: u64) -> Result<[u8; 32], SearchError> {
    if seed.is_empty() {
        return Err(SearchError::InvalidArg { detail: "ctr seed must be non-empty" });
    }
    let len = u32::try_from(seed.len())
        .map_err(|_| SearchError::InvalidArg { detail: "ctr seed too long" })?;
    let mut hasher = Sha256::new();
    hasher.update(CTR_DOMAIN);
    hasher.update(len.to_be_bytes());
    hasher.update(seed);
    hasher.update(index.to_be_bytes());
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_slice());
    Ok(out)
}

/// Seekable CTR reader over one semantic seed (`RandomStream` cursor shape).
#[derive(Debug, Clone)]
pub struct CtrReader {
    seed: Vec<u8>,
    cursor: u64,
}

impl CtrReader {
    /// Open a reader at `cursor` (empty seed fails closed, like the oracle).
    pub fn open(seed: &[u8], cursor: u64) -> Result<CtrReader, SearchError> {
        if seed.is_empty() {
            return Err(SearchError::InvalidArg { detail: "ctr seed must be non-empty" });
        }
        Ok(CtrReader { seed: seed.to_vec(), cursor })
    }

    /// Byte position (advances only through `get_bytes` draws).
    pub fn cursor(&self) -> u64 {
        self.cursor
    }

    /// Next `count` stream bytes (`get_bytes`, `randomness.py:436-445`).
    pub fn get_bytes(&mut self, count: u64) -> Result<Vec<u8>, SearchError> {
        let offset = self.cursor % CTR_BLOCK;
        let span = offset
            .checked_add(count)
            .ok_or(SearchError::InvalidArg { detail: "ctr span overflow" })?;
        let needed = span
            .checked_add(CTR_BLOCK - 1)
            .ok_or(SearchError::InvalidArg { detail: "ctr span overflow" })?
            / CTR_BLOCK;
        let first = self.cursor / CTR_BLOCK;
        let mut out = Vec::new();
        let mut i = 0u64;
        while i < needed {
            let block = first
                .checked_add(i)
                .ok_or(SearchError::InvalidArg { detail: "ctr block overflow" })?;
            let bytes = ctr_block(&self.seed, block)?;
            out.extend_from_slice(bytes.as_slice());
            i += 1;
        }
        self.cursor = self
            .cursor
            .checked_add(count)
            .ok_or(SearchError::InvalidArg { detail: "ctr cursor overflow" })?;
        let start = offset as usize;
        let end = start
            .checked_add(count as usize)
            .ok_or(SearchError::InvalidArg { detail: "ctr slice overflow" })?;
        if end > out.len() {
            return Err(SearchError::InvalidArg { detail: "ctr slice out of range" });
        }
        Ok(out[start..end].to_vec())
    }

    /// Uniform `f64` in `[0,1)` from the next 64 stream bits
    /// (`random_float`, `randomness.py:447-449`).
    pub fn random_float(&mut self) -> Result<f64, SearchError> {
        let bytes = self.get_bytes(8)?;
        let mut raw = [0u8; 8];
        raw.copy_from_slice(bytes.as_slice());
        Ok((u64::from_be_bytes(raw) as f64) / CTR_U64_DENOM)
    }

    /// Uniform integer in `[0, bound)` via rejection sampling (`random_below`,
    /// `randomness.py:451-460`): `nbytes = (bound-1).bit_length // 8 + 1`,
    /// `limit = 256^nbytes // bound * bound`.
    pub fn random_below(&mut self, bound: u32) -> Result<u32, SearchError> {
        if bound < 2 {
            return Err(SearchError::InvalidArg { detail: "ctr bound must be >= 2" });
        }
        let bitlen = 64 - (u64::from(bound) - 1).leading_zeros();
        let nbytes = bitlen / 8 + 1;
        let power = 256u64
            .checked_pow(nbytes)
            .ok_or(SearchError::InvalidArg { detail: "ctr width overflow" })?;
        let wide = u64::from(bound);
        let limit = power / wide * wide;
        loop {
            let bytes = self.get_bytes(u64::from(nbytes))?;
            let mut value: u64 = 0;
            let mut i = 0;
            while i < bytes.len() {
                value = (value << 8) | u64::from(bytes[i]);
                i += 1;
            }
            if value < limit {
                return Ok((value % wide) as u32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Natural index sampling (`sample_natural` hot loop, `natural.py:298-336`)
// ---------------------------------------------------------------------------

/// Deterministic corpus indices for `count` natural draws over `K` worlds.
///
/// `K == 1` yields `0` per draw consuming NO stream (`natural.py:313`); else
/// one rejection-sampled `below(K)` per draw. Returns `(indices, end_cursor)`
/// so callers replay exactly (`checkpoint`/`jump_to` shape).
pub fn natural_indices(
    k: u32,
    count: u32,
    seed: &[u8],
    cursor: u64,
) -> Result<(Vec<u32>, u64), SearchError> {
    if k == 0 {
        return Err(SearchError::InvalidArg { detail: "natural corpus size must be >= 1" });
    }
    if count == 0 {
        return Err(SearchError::InvalidArg { detail: "natural count must be >= 1" });
    }
    if k == 1 {
        return Ok((vec![0; count as usize], cursor));
    }
    let mut reader = CtrReader::open(seed, cursor)?;
    let mut indices = Vec::with_capacity(count as usize);
    let mut drawn = 0u32;
    while drawn < count {
        indices.push(reader.random_below(k)?);
        drawn += 1;
    }
    Ok((indices, reader.cursor()))
}

// ---------------------------------------------------------------------------
// Sampled draws (`enumerate_sampled` categorical loop, `sampled_kernel.py`)
// ---------------------------------------------------------------------------

/// Categorical draws over the exhaustive frame law (`sampled_kernel.py:137-147`).
///
/// `u = random_float()` per draw walks `cumulative += prob / total` in frame
/// order (last packet is the default); draw `i` of packet `e` carries
/// `raw_weight = P(e) / draws` (`sampled_kernel.py:158`). Returns
/// `(chosen, raw_weights, end_cursor)`.
pub fn sampled_draws(
    probs: &[f64],
    draws: u32,
    seed: &[u8],
    cursor: u64,
) -> Result<(Vec<u32>, Vec<f64>, u64), SearchError> {
    if probs.is_empty() {
        return Err(SearchError::InvalidArg { detail: "sampled frame must be non-empty" });
    }
    if draws == 0 {
        return Err(SearchError::InvalidArg { detail: "sampled draws must be >= 1" });
    }
    for prob in probs {
        if !prob.is_finite() || *prob < 0.0 {
            return Err(SearchError::NonFinite { context: "sampled frame probability" });
        }
    }
    let mut total = 0.0f64;
    let mut i = 0;
    while i < probs.len() {
        total += probs[i];
        i += 1;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(SearchError::ZeroMass);
    }
    let last = match u32::try_from(probs.len() - 1) {
        Ok(last) => last,
        Err(_) => return Err(SearchError::InvalidArg { detail: "sampled frame too large" }),
    };
    let mut reader = CtrReader::open(seed, cursor)?;
    let mut chosen = Vec::with_capacity(draws as usize);
    let mut weights = Vec::with_capacity(draws as usize);
    let mut drawn = 0u32;
    while drawn < draws {
        let u = reader.random_float()?;
        if !(0.0..1.0).contains(&u) {
            return Err(SearchError::NonFinite { context: "sampled uniform draw" });
        }
        let mut cumulative = 0.0f64;
        let mut pick = last;
        let mut idx = 0u32;
        while (idx as usize) < probs.len() {
            cumulative += probs[idx as usize] / total;
            if u < cumulative {
                pick = idx;
                break;
            }
            idx += 1;
        }
        chosen.push(pick);
        weights.push(probs[pick as usize] / f64::from(draws));
        drawn += 1;
    }
    Ok((chosen, weights, reader.cursor()))
}

// ---------------------------------------------------------------------------
// Packet successor enumeration (`enumerate_next`, `kernel.py:155-260`)
// ---------------------------------------------------------------------------

/// One enumerated successor: packet identity + successor refs + split mass.
///
/// `packet_id` is bare hex (`compute_packet_id`: no `sha256:` prefix);
/// every other digest is `sha256:<hex>` text.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefSuccessor {
    /// `world_succ:<16>` digest (`_cached_successor_refs`).
    pub successor_world_ref: String,
    /// `delta:<16>` digest (`_cached_successor_refs`).
    pub successor_delta: String,
    /// Discard tile (`8 + idx`).
    pub tile: u32,
    /// Packet sequence (`100 + idx`).
    pub seq: u32,
    /// Discarding seat (`(root + 1 + idx) % 4`).
    pub actor: u32,
    /// Combined mass (`0.5`).
    pub probability: f64,
    /// `ln(0.5)` (physical split applied once).
    pub log_physical: f64,
    /// `0.0` (deterministic policy applied once).
    pub log_policy: f64,
    /// `sha256:` observation hash after (`_cached_observation_hash`).
    pub observation_hash: String,
    /// Bare-hex packet id (`compute_packet_id` over the identity doc).
    pub packet_id: String,
    /// `sha256:` chain hash after folding the single public event.
    pub chain_after: String,
}

/// Successor/delta refs (`_cached_successor_refs`, `kernel.py:100-116`).
pub fn successor_refs(parent_ref: &str, tile: u32, action_id: u32) -> (String, String) {
    let succ_input = format!("{parent_ref}:{tile}:{action_id}");
    let delta_input = format!("delta:{parent_ref}:{tile}");
    let succ = format!(
        "world_succ:{}",
        &hex_of(Sha256::digest(succ_input.as_bytes()).as_slice())[..16]
    );
    let delta = format!(
        "delta:{}",
        &hex_of(Sha256::digest(delta_input.as_bytes()).as_slice())[..16]
    );
    (succ, delta)
}

/// Observation hash after (`_cached_observation_hash`, `kernel.py:119-130`):
/// `sha256:` over canon `{"packet_seq","tile"}`.
pub fn observation_hash(tile: u32, seq: u32) -> Result<String, SearchError> {
    let mut map = BTreeMap::new();
    map.insert("packet_seq".to_string(), serde_json::json!(seq));
    map.insert("tile".to_string(), serde_json::json!(tile));
    canon_digest(&map, "belief:observation_hash")
}

/// Synthetic public-discard envelope (`_make_public_discard_event`,
/// `kernel.py:62-96` with `game_tiny_001`, `action_id = 0`, empty deltas).
fn envelope_value(seq: u32, actor: u32, tile: u32, rules_hash: &str) -> serde_json::Value {
    serde_json::json!({
        "game_id": PACKET_GAME_ID,
        "sequence": seq,
        "kind": "discard",
        "actor": actor,
        "visibility": "public",
        "visible_to": [0, 1, 2, 3],
        "payload": {
            "kind": "discard",
            "actor": actor,
            "tile": tile,
            "action_id": 0,
            "source_seat": null,
            "consumed_tiles": [],
            "offered_action_ids": [],
            "accepted_action_ids": [],
            "round_index": null,
            "scores": null,
            "reason": null
        },
        "public_delta": [],
        "rules_hash": rules_hash,
        "schema_hash": PACKET_SCHEMA_HASH
    })
}

/// Empty chain digest (`_EMPTY_CHAIN_DIGEST`: `sha256(sha256(b""))`).
fn empty_chain() -> String {
    format!("sha256:{}", hex_of(Sha256::digest(b"").as_slice()))
}

/// One public fold (`_fold_public_hash`: canon `{"prefix","event"}`).
fn fold_public_hash(prefix: &str, event: &serde_json::Value) -> Result<String, SearchError> {
    let mut map = BTreeMap::new();
    map.insert("event".to_string(), event.clone());
    map.insert("prefix".to_string(), serde_json::Value::String(prefix.to_string()));
    canon_digest(&map, "belief:chain_fold")
}

/// Bare-hex packet id (`compute_packet_id`: sha256 over the identity doc
/// WITHOUT `packet_id`, hex WITHOUT the `sha256:` prefix).
fn packet_id_hex(
    actor_view: u32,
    seq: u32,
    event: &serde_json::Value,
    before: &str,
    after: &str,
    obs: &str,
) -> Result<String, SearchError> {
    let mut map = BTreeMap::new();
    map.insert("actor_view".to_string(), serde_json::json!(actor_view));
    map.insert("events".to_string(), serde_json::Value::Array(vec![event.clone()]));
    map.insert("observation_hash_after".to_string(), serde_json::Value::String(obs.to_string()));
    map.insert(
        "public_state_hash_after".to_string(),
        serde_json::Value::String(after.to_string()),
    );
    map.insert(
        "public_state_hash_before".to_string(),
        serde_json::Value::String(before.to_string()),
    );
    map.insert("source_sequence_end".to_string(), serde_json::json!(seq));
    map.insert("source_sequence_start".to_string(), serde_json::json!(seq));
    let bytes = hydra_feed::canon::canonical_bytes(&map, "belief:packet_id")
        .map_err(|err| SearchError::Canon { detail: err.to_string() })?;
    Ok(hex_of(Sha256::digest(bytes.as_slice()).as_slice()))
}

/// Exhaustive next-packet enumeration (`enumerate_next`, `kernel.py:155-260`).
///
/// Exactly 2 disjoint successors per `(parent_ref, action_id)`; mass sums to
/// one within `1e-9`; `probability == exp(log_phys + log_policy)` exactly.
/// `rules_hash` is the epoch digest text (`sha256:<hex>`); only digest/text
/// shapes cross (info-keys, never worlds).
pub fn enumerate_next(
    parent_ref: &str,
    action_id: u32,
    root_actor: u32,
    rules_hash: &str,
) -> Result<Vec<BeliefSuccessor>, SearchError> {
    if parent_ref.is_empty() {
        return Err(SearchError::InvalidArg { detail: "belief parent_ref must be non-empty" });
    }
    if root_actor >= 4 {
        return Err(SearchError::InvalidSeat { seat: root_actor });
    }
    if rules_hash.is_empty() {
        return Err(SearchError::InvalidArg { detail: "belief rules_hash must be non-empty" });
    }
    let log_physical = 0.5f64.ln();
    if !log_physical.is_finite() {
        return Err(SearchError::NonFinite { context: "belief log mass" });
    }
    let before = empty_chain();
    let mut out = Vec::with_capacity(PACKET_SUCCESSORS as usize);
    let mut idx = 0u32;
    while idx < PACKET_SUCCESSORS {
        let tile = PACKET_TILE_BASE + idx;
        let seq = PACKET_SEQ_BASE + idx;
        let actor = (root_actor + 1 + idx) % 4;
        let (successor_world_ref, successor_delta) = successor_refs(parent_ref, tile, action_id);
        let obs = observation_hash(tile, seq)?;
        let event = envelope_value(seq, actor, tile, rules_hash);
        let chain_after = fold_public_hash(&before, &event)?;
        let packet_id = packet_id_hex(root_actor, seq, &event, &before, &chain_after, &obs)?;
        out.push(BeliefSuccessor {
            successor_world_ref,
            successor_delta,
            tile,
            seq,
            actor,
            probability: PACKET_PROB,
            log_physical,
            log_policy: 0.0,
            observation_hash: obs,
            packet_id,
            chain_after,
        });
        idx += 1;
    }
    let first = match out.first() {
        Some(first) => first,
        None => return Err(SearchError::BadPartition),
    };
    let second = match out.get(1) {
        Some(second) => second,
        None => return Err(SearchError::BadPartition),
    };
    if first.packet_id == second.packet_id {
        return Err(SearchError::BadPartition);
    }
    let total = first.probability + second.probability;
    if (total - 1.0).abs() > PACKET_MASS_TOL {
        return Err(SearchError::BadPartition);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctr_block_is_deterministic_per_index() {
        let seed = b"belief-test-seed";
        let a = ctr_block(seed, 7).unwrap();
        let b = ctr_block(seed, 7).unwrap();
        let c = ctr_block(seed, 8).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn ctr_reader_advances_cursor_by_draws() {
        let seed = b"belief-test-seed";
        let mut reader = CtrReader::open(seed, 0).unwrap();
        let _ = reader.random_float().unwrap();
        assert_eq!(reader.cursor(), 8);
        let _ = reader.random_below(4).unwrap();
        assert!(reader.cursor() > 8);
    }

    #[test]
    fn natural_singleton_consumes_no_stream() {
        let (indices, end) = natural_indices(1, 5, b"belief-test-seed", 41).unwrap();
        assert_eq!(indices, vec![0, 0, 0, 0, 0]);
        assert_eq!(end, 41);
    }

    #[test]
    fn natural_indices_replay_from_cursor() {
        let (first, mid) = natural_indices(4, 8, b"belief-test-seed", 0).unwrap();
        assert_eq!(first.len(), 8);
        assert!(first.iter().all(|idx| *idx < 4));
        let (rest, end) = natural_indices(4, 4, b"belief-test-seed", mid).unwrap();
        let (full, _) = natural_indices(4, 12, b"belief-test-seed", 0).unwrap();
        assert_eq!(&full[..8], &first[..]);
        assert_eq!(&full[8..], &rest[..]);
        assert!(end > mid);
    }

    #[test]
    fn sampled_draws_weights_match_frame_over_draws() {
        let (chosen, weights, _) =
            sampled_draws(&[0.5, 0.5], 4, b"belief-test-seed", 0).unwrap();
        assert_eq!(chosen.len(), 4);
        for (pick, weight) in chosen.iter().zip(weights.iter()) {
            assert!(*pick < 2);
            assert_eq!(*weight, 0.5 / 4.0);
        }
    }

    #[test]
    fn packet_successors_form_unit_partition() {
        let out = enumerate_next("world:test:001", 3, 0, PACKET_SCHEMA_HASH).unwrap();
        assert_eq!(out.len(), 2);
        assert_ne!(out[0].packet_id, out[1].packet_id);
        assert_eq!(out[0].tile, 8);
        assert_eq!(out[1].tile, 9);
        assert_eq!(out[0].seq, 100);
        assert_eq!(out[1].seq, 101);
        assert!((out[0].probability + out[1].probability - 1.0).abs() <= PACKET_MASS_TOL);
        for succ in &out {
            assert!(succ.successor_world_ref.starts_with("world_succ:"));
            assert!(succ.successor_delta.starts_with("delta:"));
            assert!(succ.observation_hash.starts_with("sha256:"));
            assert!(succ.chain_after.starts_with("sha256:"));
            assert_eq!(succ.packet_id.len(), 64);
        }
    }
}
