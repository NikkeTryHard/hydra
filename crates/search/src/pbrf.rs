//! `pbrf`: forest carry + delta-verify.
//!
//! Parity: `pbrf_forest.py:61-86` (`_conditional_carry_logps`: `Z = sum(raw)`,
//! finite `Z > 0` else `ContractError`; emit `ln(raw/Z)` for BOTH density
//! fields so the ratio stays one by construction; uniform `-ln(N)`
//! FORBIDDEN) + `pbrf_forest.py:112-178` (`_verify_delta_reconstruction`:
//! `succ = "world_succ:"+sha(parent:tile:aid)[:16]`,
//! `delta = "delta:"+sha("delta:"+parent:tile)[:16]`; stored tile
//! authoritative; `None` tile probes the legacy `0..19` window plus the
//! canonical parent+delta escape hatch) + `pbrf_forest.py:244-527`
//! (`parent_count = 16` frozen parents, fixed `allocate` batches `<= 64`,
//! keys `(action_id, packet_id)` ONLY).
//!
//! Commit: hit -> carry `b+` (conditional law, never fresh naturals);
//! miss -> fresh; stale epoch (`!= forest.epoch + 1`) rejects; ESS-gated.

use sha2::{Digest, Sha256};

use crate::SearchError;

/// Frozen PBRF parent count (forest roots sampled natural).
pub const PBRF_PARENTS: u32 = 16;
/// Fixed-allocate batch ceiling (deterministic sizes, no RNG).
pub const PBRF_MAX_BATCHES: u32 = 64;
/// Legacy tile probe window (`0..19` covers stub tiles 8/9).
pub const PBRF_LEGACY_TILES: u32 = 20;
/// Digest tail length (`[:16]` hex chars).
pub const PBRF_TAIL: usize = 16;

/// One forest child entry: one particle's contribution to a specific
/// `(action, packet)` child (`pbrf_partition.py:152-174` vocabulary:
/// `parent_id` text, `successor_world_ref` + `successor_delta` digest
/// texts, `raw_weight`, `target_id` digest text, `epoch`, `ancestors`,
/// `tile`). `u64` arena fields (`action_id`, `packet_id`, forest `epoch`)
/// ride OUTSIDE this struct as `(action_id, packet_id)` keys + the
/// `commit_gate` epoch args (keys-`(action,packet)`-ONLY rule).
#[derive(Debug, Clone)]
pub struct ChildEntry {
    /// Parent id text (non-empty; `make_parent_id` shape).
    pub parent_id: String,
    /// Successor digest text (`"world_succ:"+hex[:16]`).
    pub successor_world_ref: String,
    /// Delta digest text (`"delta:"+hex[:16]`).
    pub successor_delta: String,
    /// Raw (unnormalized) weight (`> 0`, finite).
    pub raw_weight: f64,
    /// Forest target digest text (`sha256:`-prefixed).
    pub target_id: String,
    /// Forest epoch at enumeration.
    pub epoch: u64,
    /// Ancestor parent-id texts.
    pub ancestors: Vec<String>,
    /// Originating transition tile (authoritative when `Some`).
    pub tile: Option<u32>,
}

/// Conditional carry log-densities (`pbrf_forest.py:61-86` parity):
/// `ln(raw_i / Z)` with `Z = sum(raw)`; zero/nonfinite `Z` or entry mass
/// is `ZeroMass` (caller takes the MISS path). Uniform `-ln(N)` FORBIDDEN
/// (would smuggle a hidden importance ratio).
pub fn conditional_carry_logps(entries: &[ChildEntry]) -> Result<Vec<f64>, SearchError> {
    if entries.is_empty() {
        return Err(SearchError::ZeroMass);
    }
    let mut total = 0.0;
    for entry in entries {
        if !entry.raw_weight.is_finite() {
            return Err(SearchError::NonFinite {
                context: "pbrf carry raw",
            });
        }
        total += entry.raw_weight;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(SearchError::ZeroMass);
    }
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let w = entry.raw_weight / total;
        if !w.is_finite() || w <= 0.0 {
            return Err(SearchError::ZeroMass);
        }
        out.push(w.ln());
    }
    Ok(out)
}

/// Expected successor digest (`"world_succ:"+sha(parent:tile:aid)[:16]`).
pub fn expected_successor(parent: &str, tile: u32, action: u32) -> String {
    let digest = Sha256::digest(format!("{parent}:{tile}:{action}").as_bytes());
    let hex = hex_of(digest.as_slice());
    format!("world_succ:{}", &hex[..PBRF_TAIL])
}

/// Expected delta digest (`"delta:"+sha("delta:"+parent:tile)[:16]`).
pub fn expected_delta(parent: &str, tile: u32) -> String {
    let digest = Sha256::digest(format!("delta:{parent}:{tile}").as_bytes());
    let hex = hex_of(digest.as_slice());
    format!("delta:{}", &hex[..PBRF_TAIL])
}

/// Lowercase hex of a digest.
fn hex_of(digest: &[u8]) -> String {
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Delta-verify (`pbrf_forest.py:112-178` parity): with `Some(tile)`, BOTH
/// reconstructions MUST match (stored tile authoritative; mismatch falls
/// through to the canonical parent+delta escape hatch for synthetic
/// kernels, else `false`). With `None`, probe the legacy `0..19` window,
/// then the canonical hatch. Empty refs are `false` (never panic).
pub fn verify_delta(
    parent: &str,
    successor: &str,
    delta: &str,
    action: u32,
    tile: Option<u32>,
) -> bool {
    if parent.is_empty() || successor.is_empty() || delta.is_empty() {
        return false;
    }
    match tile {
        Some(known) => {
            if expected_successor(parent, known, action) == successor
                && expected_delta(parent, known) == delta
            {
                return true;
            }
            canonical_hatch(parent, successor, delta)
        }
        None => {
            let mut probe = 0;
            while probe < PBRF_LEGACY_TILES {
                if expected_successor(parent, probe, action) == successor
                    && expected_delta(parent, probe) == delta
                {
                    return true;
                }
                probe += 1;
            }
            canonical_hatch(parent, successor, delta)
        }
    }
}

/// Canonical parent+delta escape hatch: `world_succ:sha256(canonical_bytes(
/// {parent, delta}))[:16]` (synthetic kernels bypassing tile hashing).
/// Canon-wins: serializes through `feed::canon`.
fn canonical_hatch(parent: &str, successor: &str, delta: &str) -> bool {
    let mut map = std::collections::BTreeMap::new();
    map.insert("delta".to_string(), serde_json::json!(delta));
    map.insert("parent".to_string(), serde_json::json!(parent));
    let bytes = match hydra_feed::canon::canonical_bytes(&map, "pbrf:delta_hatch") {
        Ok(bytes) => bytes,
        Err(_) => return false,
    };
    let digest = Sha256::digest(bytes.as_slice());
    let hex = hex_of(digest.as_slice());
    successor == format!("world_succ:{}", &hex[..PBRF_TAIL])
}

/// Entry-shaped delta-verify: oracle-arg-shape wrapper over [`verify_delta`].
///
/// `_verify_delta_reconstruction` reads the refs off one [`ChildEntry`]
/// (`parent_id`, `successor_world_ref`, `successor_delta`, `tile`) plus the
/// acting `action_id` key; `packet_id` is intentionally NOT an input (the
/// `(action_id, packet_id)` keys select the entry, never the bytes).
pub fn verify_entry(entry: &ChildEntry, action_id: u32) -> bool {
    verify_delta(
        entry.parent_id.as_str(),
        entry.successor_world_ref.as_str(),
        entry.successor_delta.as_str(),
        action_id,
        entry.tile,
    )
}

/// Target-compat gate (`_is_target_compatible` epoch-increment shape):
/// entries non-empty, share one `(target_id, epoch)`, and the authoritative
/// epoch is exactly `forest_epoch + 1` (stale children reject).
pub fn is_target_compatible(entries: &[ChildEntry], authoritative_epoch: u64) -> bool {
    if entries.is_empty() {
        return false;
    }
    let first = &entries[0];
    for entry in entries {
        if entry.target_id != first.target_id || entry.epoch != first.epoch {
            return false;
        }
    }
    authoritative_epoch == first.epoch + 1
}

/// Commit disposition for one forest commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitDisposition {
    /// Hit: carry `b+` (conditional law, never fresh naturals).
    HitCarry,
    /// Miss: sample fresh naturals.
    MissFresh,
    /// Stale epoch (`packet_epoch != forest_epoch + 1`): reject.
    StaleReject,
}

/// Commit gate: stale epoch rejects; `hit` selects carry vs fresh.
/// ESS gating is the caller's numeric check (`ess_below`).
pub fn commit_gate(forest_epoch: u64, packet_epoch: u64, hit: bool) -> CommitDisposition {
    if packet_epoch != forest_epoch + 1 {
        return CommitDisposition::StaleReject;
    }
    if hit {
        CommitDisposition::HitCarry
    } else {
        CommitDisposition::MissFresh
    }
}

/// ESS gate: `ess < threshold * parents` forces a fresh sample even on hit.
pub fn ess_gate(ess: f64, parents: u32, threshold: f64) -> Result<bool, SearchError> {
    if !ess.is_finite() || ess < 0.0 {
        return Err(SearchError::NonFinite {
            context: "pbrf ess",
        });
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(SearchError::InvalidArg {
            detail: "ess threshold must be finite >=0",
        });
    }
    Ok(ess < threshold * parents as f64)
}

/// Deterministic leaf vector for ONE `(action, packet)` child
/// (`pbrf_search.py:252-294` parity): `z = sum(raw)` (pack order); `z <= 0`
/// reads as the zero vector (the oracle's early return — note `NaN <= 0`
/// is false, so a `NaN` mass flows through to a `NaN` scalar exactly as
/// the oracle, and the caller fails it closed on finiteness). Otherwise
/// each entry contributes `u32BE(sha256(canon({action, packet, parent,
/// target}))[:4]) / 0xFFFFFFFF * (raw / z)` with `parent`/`target`
/// truncated to 8 chars caller-side, summed in entry order; the scalar
/// expands to the 4-seat vector `(s, (1-s)*0.3, (1-s)*0.3, (1-s)*0.4)`
/// with the oracle's op order.
///
/// Canon-wins (B3): the payload serializes through `feed::canon`
/// (`BTreeMap` discipline matches the oracle's `canonical_bytes`;
/// `action` crosses as a JSON number, matching the oracle's int).
pub fn child_value(
    parent8s: &[String],
    target8s: &[String],
    raw_weights: &[f64],
    aid: u32,
    packet_id: &str,
) -> Result<(f64, f64, f64, f64), SearchError> {
    if parent8s.len() != target8s.len() || parent8s.len() != raw_weights.len() {
        return Err(SearchError::InvalidArg {
            detail: "pbrf child row length mismatch",
        });
    }
    if parent8s.is_empty() {
        return Ok((0.0, 0.0, 0.0, 0.0));
    }
    let z = crate::builtin_sum(raw_weights);
    if z <= 0.0 {
        return Ok((0.0, 0.0, 0.0, 0.0));
    }
    let mut terms = Vec::with_capacity(parent8s.len());
    let mut idx = 0;
    while idx < parent8s.len() {
        let mut map = std::collections::BTreeMap::new();
        map.insert("action".to_string(), serde_json::json!(aid));
        map.insert("packet".to_string(), serde_json::json!(packet_id));
        map.insert("parent".to_string(), serde_json::json!(parent8s[idx]));
        map.insert("target".to_string(), serde_json::json!(target8s[idx]));
        let bytes =
            hydra_feed::canon::canonical_bytes(&map, "pbrf:child_value").map_err(|err| {
                SearchError::Canon {
                    detail: err.to_string(),
                }
            })?;
        let digest = Sha256::digest(bytes.as_slice());
        let word = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
        let val = word as f64 / crate::despot::HASH_UNIT;
        terms.push(val * (raw_weights[idx] / z));
        idx += 1;
    }
    let scalar = crate::builtin_sum(&terms);
    Ok((
        scalar,
        (1.0 - scalar) * 0.3,
        (1.0 - scalar) * 0.3,
        (1.0 - scalar) * 0.4,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn honest_entry(tile: u32, raw: f64) -> ChildEntry {
        let parent = "sha256:parent0001";
        ChildEntry {
            parent_id: parent.to_string(),
            successor_world_ref: expected_successor(parent, tile, 4),
            successor_delta: expected_delta(parent, tile),
            raw_weight: raw,
            target_id: format!("sha256:{}", "t".repeat(64)),
            epoch: 1,
            ancestors: Vec::new(),
            tile: Some(tile),
        }
    }

    #[test]
    fn carry_is_normalized_log_not_uniform() {
        // weights 1:3 -> ln(0.25), ln(0.75); uniform -ln(2) FORBIDDEN.
        let entries = [honest_entry(8, 1.0), honest_entry(9, 3.0)];
        let logps = conditional_carry_logps(&entries).expect("carry");
        assert!((logps[0] - 0.25f64.ln()).abs() < 1e-12);
        assert!((logps[1] - 0.75f64.ln()).abs() < 1e-12);
        assert!(
            (logps[0] - (-2f64.ln())).abs() > 1e-9,
            "must not be uniform"
        );
    }

    #[test]
    fn carry_zero_mass_takes_miss_path() {
        assert_eq!(conditional_carry_logps(&[]), Err(SearchError::ZeroMass));
        let zero = [honest_entry(8, 0.0)];
        assert_eq!(conditional_carry_logps(&zero), Err(SearchError::ZeroMass));
    }

    #[test]
    fn delta_verify_honest_tiles_pass() {
        let parent = "sha256:parent0001";
        let succ = expected_successor(parent, 8, 4);
        let delta = expected_delta(parent, 8);
        assert!(verify_delta(parent, &succ, &delta, 4, Some(8)));
        // Wrong stored tile -> genuine failure (canonical hatch cannot
        // rescue a tile-hashed successor built for tile 8).
        assert!(!verify_delta(parent, &succ, &delta, 4, Some(9)));
    }

    #[test]
    fn delta_verify_legacy_window_probes() {
        // `None` tile falls back to the 0..19 probe: stub tiles 8/9 verify.
        let parent = "sha256:parent0002";
        let succ = expected_successor(parent, 9, 2);
        let delta = expected_delta(parent, 9);
        assert!(verify_delta(parent, &succ, &delta, 2, None));
        // Tampered delta fails everywhere.
        assert!(!verify_delta(
            parent,
            &succ,
            "delta:deadbeefdeadbeef",
            2,
            None
        ));
    }

    #[test]
    fn delta_verify_empty_refs_false() {
        assert!(!verify_delta("", "x", "y", 0, Some(1)));
        assert!(!verify_delta("p", "", "y", 0, None));
    }

    #[test]
    fn commit_gate_epochs() {
        assert_eq!(commit_gate(5, 6, true), CommitDisposition::HitCarry);
        assert_eq!(commit_gate(5, 6, false), CommitDisposition::MissFresh);
        assert_eq!(commit_gate(5, 7, true), CommitDisposition::StaleReject);
        assert_eq!(commit_gate(5, 5, false), CommitDisposition::StaleReject);
    }

    #[test]
    fn ess_gate_threshold() {
        assert!(ess_gate(4.0, 16, 0.5).expect("ess"));
        assert!(!ess_gate(12.0, 16, 0.5).expect("ess"));
        assert!(ess_gate(f64::NAN, 16, 0.5).is_err());
    }

    #[test]
    fn persistence_golden_per_arm_shape() {
        // Persistence-golden family anchor (B4 lands fully with EvalControl):
        // per-arm (selected, vectors, counters) digest inputs are stable.
        // Here: carry logps for 16 equal parents are EXACTLY -ln(16).
        let entries: Vec<ChildEntry> = (0..16)
            .map(|i| ChildEntry {
                parent_id: "sha256:golden".to_string(),
                successor_world_ref: expected_successor("sha256:golden", 8 + (i % 2), i),
                successor_delta: expected_delta("sha256:golden", 8 + (i % 2)),
                raw_weight: 1.0,
                target_id: format!("sha256:{}", "g".repeat(64)),
                epoch: 0,
                ancestors: Vec::new(),
                tile: Some(8 + (i % 2)),
            })
            .collect();
        let logps = conditional_carry_logps(&entries).expect("golden carry");
        assert_eq!(logps.len(), 16);
        for lp in logps {
            assert!((lp - (-16f64.ln())).abs() < 1e-12);
        }
    }

    #[test]
    fn verify_entry_matches_string_form() {
        // Oracle-arg-shape wrapper agrees with the string form on honest
        // entries (tile 8, action 4).
        let entry = honest_entry(8, 1.0);
        assert!(verify_entry(&entry, 4));
        assert!(!verify_entry(&entry, 5));
    }

    #[test]
    fn target_compat_epoch_increment() {
        let entries = [honest_entry(8, 1.0), honest_entry(9, 2.0)];
        assert!(is_target_compatible(&entries, 2));
        assert!(!is_target_compatible(&entries, 3));
        assert!(!is_target_compatible(&[], 2));
    }
}
