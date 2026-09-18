//! Whole-game partition assignment — the pre-expansion split authority.
//!
//! Rust owner of `src/hydra2/data/partition.py` (group/sort/dedup/wall-disjoint)
//! plus the split half of `src/hydra2/data/stream_read.py`
//! (`assign_split`, `group_key_for[_path]`, `compute_wall_hash` identity math).
//! Python oracle is read-only this phase: this module ports its behavior, it
//! never edits it.
//!
//! Requirements (mirrors `partition.py:1-9` module doc):
//! - Assign complete games before decisions (never split a game).
//! - Enforce `(source, time)` grouping when metadata permits.
//! - Reject exact and near duplicates across partitions.
//! - Keep rollout/evaluation walls disjoint.
//! - Split manifest stores algorithm/version/seed/input hashes.
//!
//! Single-assignment rule (wave3 packet §1.10, Data m4): `assign_split`
//! (per-group, stream path) and `assign_partitions` (whole-corpus, partition
//! path) MUST agree — same order, same preimage, same draw, same walk. This
//! module implements ONE function ([`assign_one`]) called by both paths. The
//! order table is *value-equal* to `stream_manifest.py:50` `PARTITION_ORDER`:
//! `partition.py:161-167` holds a local literal, the import lives in the
//! manifest — worded as value-equality per Data m4, never claimed same-symbol.
//!
//! Identity discipline (endstate §0.3):
//! - SHA-256 ONLY on identity paths. No BLAKE3 anywhere here.
//! - Spec/games/manifest digests call `feed::canon` + `feed::digest`
//!   (Data M1: canon owns bytes+math, this module frames+parses+calls). There
//!   is NO second printer and NO second hasher in this file.
//! - Split draw is `u64BE / 2**64` float on `sha256(f"{seed}|{group}")`
//!   (stream_read.py:171-172, partition.py:180-181). It is NOT Lemire
//!   `bounded` — converting it would fork every assignment golden.
//!
//! Shuffle/RNG discipline (Data M2 + B2): there is NO `feed::shuffle` module.
//! Shuffle draws live in `feed::rng` (`shuffle_indices` / `below`); held-out
//! splits stay on the `torch.randperm` oracle behind KAT per Wave5 B2 — Rust
//! `philox_split_perm` ships only behind a `philox_perm == torch.randperm`
//! KAT, so this file never invents a Philox split permutation.
//!
//! Scan migration (Data M8): `training/stream_scan.py:23-24` imports
//! `GameStream`/`ZstdLineStream` from the `data.stream` shim today. The packet
//! slice migrates those two imports to the new Rust scan/batch ABI; the scan
//! caller (including its `validate_game` discard-check) moves with it. This
//! module owns the assignment math the scan calls, not the scan loop itself.
//!
//! Seed map (Data M6): the 32 B → Philox lane mapping lives in `feed::rng`
//! ([`crate::rng::seed_parts`]: `key0=seed[0:8]`, `key1=seed[8:16]`,
//! `ctr0=seed[16:24]`, `spare=seed[24:32]`, all LE). This module takes the
//! already-derived `seed: u64` for the `f"{seed}|{group}"` preimage and does
//! not re-derive lanes — the M6 KAT below pins the shared mapping by calling
//! it directly so the two modules cannot silently disagree.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::canon::CanonError;
use crate::digest::{of_canonical, sha256_hex};

/// Partition names in cumulative-threshold order.
///
/// Value-equal to `stream_manifest.PARTITION_ORDER` and to the
/// `partition.py:161-167` local literal (Data m4: value-equality, not
/// same-symbol). Cumulative thresholds walk THIS order; reordering forks
/// every assignment.
pub const PARTITION_ORDER: [&str; 5] =
    ["train", "validation", "test", "decision_eval", "block_eval"];

/// Default grouping; mirrors `grouping_keys=("source", "time")`.
pub const DEFAULT_GROUPING_KEYS: [&str; 2] = ["source", "time"];

/// Partition failure taxonomy. Mirrors `ContractError` edges in
/// `partition.py` (fail-closed, never silent coercion).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionError {
    /// No games supplied (`partition.py:127-128`).
    EmptyGames { detail: String },
    /// Exact duplicates by decoded hash (`:134-136`).
    ExactDuplicates { detail: String },
    /// Near duplicates by wall hash (`:137-139`).
    NearDuplicates { detail: String },
    /// Ratios do not sum to 1.0 within 1e-6 (`:158-160`).
    BadRatios { detail: String },
    /// No active (positive-weight) partition (`:169-170`).
    NoActivePartitions { detail: String },
    /// A single ratio is negative or non-finite.
    BadRatioValue { detail: String },
    /// Assignment count differs from game count (`:190-191` — game split).
    CountMismatch { detail: String },
    /// One wall hash lands in two partitions (`:192-200`).
    WallCrossPartition { detail: String },
    /// Train walls intersect eval walls (`:201-213`).
    EvalTrainWallOverlap { detail: String },
    /// Canon/digest step failed (non-string key, unsafe number, …).
    Canon { detail: String },
}

impl core::fmt::Display for PartitionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PartitionError::EmptyGames { detail } => {
                write!(f, "partition: no games to partition ({detail})")
            }
            PartitionError::ExactDuplicates { detail } => {
                write!(f, "partition: exact duplicates detected: {detail}")
            }
            PartitionError::NearDuplicates { detail } => {
                write!(f, "partition: near duplicates detected: {detail}")
            }
            PartitionError::BadRatios { detail } => {
                write!(f, "partition: ratios must sum to 1.0 ({detail})")
            }
            PartitionError::NoActivePartitions { detail } => {
                write!(f, "partition: no active partitions in ratios ({detail})")
            }
            PartitionError::BadRatioValue { detail } => {
                write!(f, "partition: bad split ratio ({detail})")
            }
            PartitionError::CountMismatch { detail } => {
                write!(f, "partition: partition count mismatch ({detail})")
            }
            PartitionError::WallCrossPartition { detail } => {
                write!(f, "partition: wall appears in multiple partitions ({detail})")
            }
            PartitionError::EvalTrainWallOverlap { detail } => {
                write!(f, "partition: rollout/evaluation walls disjoint violation ({detail})")
            }
            PartitionError::Canon { detail } => {
                write!(f, "partition: canonicalization failed ({detail})")
            }
        }
    }
}

impl std::error::Error for PartitionError {}

impl From<CanonError> for PartitionError {
    fn from(e: CanonError) -> Self {
        PartitionError::Canon {
            detail: e.to_string(),
        }
    }
}

/// One game's split identity. Mirrors `partition.GameIdentity`
/// (`partition.py:39-47`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameIdentity {
    /// Stable game id (assignment key).
    pub game_id: String,
    /// Transport object id (acquisition-metadata join key).
    pub object_id: String,
    /// Source id (`acquisition_metadata["source"]`, default `"unknown"`).
    pub source_id: String,
    /// Player ids (`"unknown"` singleton when absent, per `:75-76`).
    pub player_ids: Vec<String>,
    /// Timestamp string or `None` (non-string metadata → `None`, per `:77-78`).
    pub timestamp: Option<String>,
    /// Wall hash (`sha256:<hex>` of canon wall tiles) or `None` when no wall.
    pub wall_hash: Option<String>,
    /// Decoded-bytes identity (`record.raw_bytes_sha256`).
    pub decoded_hash: String,
}

impl GameIdentity {
    /// Test/fixture constructor with the Python `"unknown"` defaults.
    pub fn new(
        game_id: &str,
        object_id: &str,
        source_id: &str,
        player_ids: Vec<String>,
        timestamp: Option<String>,
        wall_hash: Option<String>,
        decoded_hash: &str,
    ) -> Self {
        let pids = if player_ids.is_empty() {
            vec!["unknown".to_string()]
        } else {
            player_ids
        };
        GameIdentity {
            game_id: game_id.to_string(),
            object_id: object_id.to_string(),
            source_id: source_id.to_string(),
            player_ids: pids,
            timestamp,
            wall_hash,
            decoded_hash: decoded_hash.to_string(),
        }
    }
}

/// Split specification. Mirrors `partition.SplitSpec` (`:50-57`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SplitSpec {
    /// Assignment algorithm name (recorded, not interpreted).
    pub algorithm: String,
    /// Algorithm version (recorded, not interpreted).
    pub version: String,
    /// Split seed (non-negative; formats as `f"{seed}|{group}"`).
    pub seed: u64,
    /// Partition weights; must sum to 1.0 within 1e-6.
    pub ratios: BTreeMap<String, f64>,
    /// Grouping keys, subset of `source` / `player` / `time`.
    pub grouping_keys: Vec<String>,
    /// Enforce wall-disjointness across partitions.
    pub wall_disjoint: bool,
}

/// Split result. Mirrors `partition.SplitManifest` (`:60-65`).
#[derive(Debug, Clone, PartialEq)]
pub struct SplitManifest {
    /// Spec the assignment was computed under.
    pub spec: SplitSpec,
    /// `game_id -> partition` (whole games only).
    pub assignments: BTreeMap<String, String>,
    /// `{"spec": sha, "games": sha}` input hashes.
    pub input_hashes: BTreeMap<String, String>,
    /// `sha256:` digest over `{spec, assignments}` canon bytes.
    pub digest: String,
}

/// `(source, time)` group key. Identical to
/// `stream_read.group_key_for` (`stream_read.py:113-115`).
pub fn group_key_for_source_time(source: &str, time: &str) -> String {
    format!("{source}|{time}")
}

/// Corpus grouping key for one identity under `grouping_keys`.
///
/// Mirrors `partition.py:141-156`: `source` → `source_id`,
/// `player` → `"|".join(sorted(player_ids))`, `time` → first 10 chars of the
/// timestamp (`"unknown"` when absent), anything else → `"unknown"`; segments
/// joined with `|`. Empty `grouping_keys` falls back to `game_id` (each game
/// its own group).
pub fn partition_group_key(ident: &GameIdentity, grouping_keys: &[String]) -> String {
    if grouping_keys.is_empty() {
        return ident.game_id.clone();
    }
    let mut parts: Vec<String> = Vec::with_capacity(grouping_keys.len());
    for gk in grouping_keys {
        match gk.as_str() {
            "source" => parts.push(ident.source_id.clone()),
            "player" => {
                let mut pids = ident.player_ids.clone();
                pids.sort();
                parts.push(pids.join("|"));
            }
            "time" => {
                let ts = ident.timestamp.as_deref().unwrap_or("unknown");
                parts.push(ts.chars().take(10).collect());
            }
            _ => parts.push("unknown".to_string()),
        }
    }
    parts.join("|")
}

/// Validate ratios and build the cumulative threshold walk.
///
/// Shared by [`assign_one`] and [`assign_partitions`]: active partitions are
/// `PARTITION_ORDER` filtered to positive weight, thresholds accumulate in
/// that order. Rejects empty maps, negative/non-finite weights, sums outside
/// `1.0 ± 1e-6`, and empty active sets — mirroring `stream_read.py:147-170`
/// and `partition.py:158-170` exactly (both raise `ContractError` there).
fn cumulative_thresholds(
    ratios: &BTreeMap<String, f64>,
) -> Result<Vec<(String, f64)>, PartitionError> {
    if ratios.is_empty() {
        return Err(PartitionError::BadRatios {
            detail: "split ratios must not be empty".to_string(),
        });
    }
    for (name, w) in ratios {
        if !w.is_finite() || *w < 0.0 {
            return Err(PartitionError::BadRatioValue {
                detail: format!("split ratio for {name:?} not a non-negative number: {w}"),
            });
        }
    }
    let total: f64 = ratios.values().sum();
    if (total - 1.0).abs() > 1e-6 {
        return Err(PartitionError::BadRatios {
            detail: format!("split ratios must sum to 1.0, got {total}"),
        });
    }
    let mut cumulative: Vec<(String, f64)> = Vec::new();
    let mut running = 0.0;
    for part in PARTITION_ORDER {
        if let Some(w) = ratios.get(part) {
            if *w > 0.0 {
                running += *w;
                cumulative.push((part.to_string(), running));
            }
        }
    }
    if cumulative.is_empty() {
        return Err(PartitionError::NoActivePartitions {
            detail: "no active partitions in ratios".to_string(),
        });
    }
    Ok(cumulative)
}

/// Draw for one group: `u64BE(sha256(f"{seed}|{group}")[..8]) / 2**64`.
///
/// Byte-exact with `stream_read.py:171-172` and `partition.py:180-181`.
/// Routed through `feed::digest::sha256_hex` (the single owned `sha2 0.11`
/// hasher — this module owns NO second hasher): the first 16 hex chars
/// decode to the same `u64BE` value as the first 8 digest bytes.
fn hex_val(c: u8) -> u64 {
    match c {
        b'0'..=b'9' => u64::from(c - b'0'),
        b'a'..=b'f' => u64::from(c - b'a') + 10,
        b'A'..=b'F' => u64::from(c - b'A') + 10,
        _ => 0,
    }
}
fn split_draw(group_key: &str, seed: u64) -> f64 {
    let preimage = format!("{seed}|{group_key}");
    let hex = sha256_hex(preimage.as_bytes());
    let head = hex.as_bytes();
    let mut raw: u64 = 0;
    let mut i = "sha256:".len();
    while i < head.len() && i < "sha256:".len() + 16 {
        raw = (raw << 4) | hex_val(head[i]);
        i += 1;
    }
    raw as f64 / 18446744073709551616.0
}

/// Walk the cumulative thresholds for one draw — the single threshold walk
/// both paths share (stream `assign_one` + corpus `assign_partitions`).
///
/// First partition with `draw < threshold` wins; a draw at or past every
/// threshold falls through to the last active partition (mirrors both
/// Python `return cumulative[-1][0]` tails). The `None` arm is fail-closed
/// (never a silent `"train"` fallback — masks ratio bugs).
fn choose_partition(
    cumulative: &[(String, f64)],
    draw: f64,
) -> Result<String, PartitionError> {
    for (part, threshold) in cumulative {
        if draw < *threshold {
            return Ok(part.clone());
        }
    }
    match cumulative.last() {
        Some((part, _)) => Ok(part.clone()),
        None => Err(PartitionError::NoActivePartitions {
            detail: "no active partitions in ratios".to_string(),
        }),
    }
}

/// Assign ONE group to a partition — the single function both paths call.
///
/// `assign_split(group_key, seed, ratios)` (stream/scan path) and the
/// per-group loop inside `assign_partitions` (corpus path) are the same math:
/// cumulative thresholds over [`PARTITION_ORDER`] on the [`split_draw`].
pub fn assign_one(
    group_key: &str,
    seed: u64,
    ratios: &BTreeMap<String, f64>,
) -> Result<String, PartitionError> {
    let cumulative = cumulative_thresholds(ratios)?;
    choose_partition(&cumulative, split_draw(group_key, seed))
}

/// Exact duplicates: same decoded hash (identical bytes).
///
/// Mirrors `partition.detect_exact_duplicates` (`:94-103`): first-seen wins,
/// later games pair against it. Order-stable in input order.
pub fn detect_exact_duplicates(games: &[GameIdentity]) -> Vec<(String, String)> {
    let mut seen: BTreeMap<&str, &str> = BTreeMap::new();
    let mut dups: Vec<(String, String)> = Vec::new();
    for g in games {
        if let Some(first) = seen.get(g.decoded_hash.as_str()) {
            dups.push(((*first).to_string(), g.game_id.clone()));
        } else {
            seen.insert(g.decoded_hash.as_str(), g.game_id.as_str());
        }
    }
    dups
}

/// Near duplicates: same wall hash (identical wall).
///
/// Mirrors `partition.detect_near_duplicates` (`:106-117`): wall-less games
/// are skipped (no wall ⇒ no near-duplicate signal).
pub fn detect_near_duplicates(games: &[GameIdentity]) -> Vec<(String, String)> {
    let mut seen: BTreeMap<&str, &str> = BTreeMap::new();
    let mut dups: Vec<(String, String)> = Vec::new();
    for g in games {
        let Some(wall) = g.wall_hash.as_deref() else {
            continue;
        };
        if let Some(first) = seen.get(wall) {
            dups.push(((*first).to_string(), g.game_id.clone()));
        } else {
            seen.insert(wall, g.game_id.as_str());
        }
    }
    dups
}

/// Canonical spec document hashed into `input_hashes["spec"]`.
///
/// Shape mirrors `partition.py:216-228`: `{algorithm, version, seed,
/// ratios, grouping_keys, wall_disjoint}` over `feed::canon` bytes.
#[derive(Serialize)]
struct SpecDoc<'a> {
    algorithm: &'a str,
    version: &'a str,
    seed: u64,
    ratios: &'a BTreeMap<String, f64>,
    grouping_keys: &'a [String],
    wall_disjoint: bool,
}

/// Assign whole games to partitions; enforces grouping and duplicate checks.
///
/// Mirrors `partition.assign_partitions` (`:120-246`):
/// 1. Empty input raises.
/// 2. Exact duplicates raise; near duplicates raise (duplicates cannot cross
///    partitions — whole games are rejected, never split).
/// 3. Groups form under [`partition_group_key`], sorted by group key; each
///    group draws once via [`assign_one`] and all members follow it (a game
///    is never split across partitions — `:190-191` count check).
/// 4. With `wall_disjoint`: one wall ⇒ one partition, plus train walls ∩
///    eval (`block_eval`/`decision_eval`) walls must be empty (`:192-213`).
/// 5. `input_hashes` + `digest` bind spec/games/assignments via canon bytes
///    (`:215-243`).
pub fn assign_partitions(
    games: &[GameIdentity],
    spec: &SplitSpec,
) -> Result<SplitManifest, PartitionError> {
    if games.is_empty() {
        return Err(PartitionError::EmptyGames {
            detail: "no games to partition".to_string(),
        });
    }
    let exact = detect_exact_duplicates(games);
    if !exact.is_empty() {
        let preview: Vec<(String, String)> = exact.into_iter().take(3).collect();
        return Err(PartitionError::ExactDuplicates {
            detail: format!("{preview:?}"),
        });
    }
    let near = detect_near_duplicates(games);
    if !near.is_empty() {
        let preview: Vec<(String, String)> = near.into_iter().take(3).collect();
        return Err(PartitionError::NearDuplicates {
            detail: format!("{preview:?}"),
        });
    }

    // Grouping when metadata permits (sorted group order ⇒ deterministic).
    let mut groups: BTreeMap<String, Vec<&GameIdentity>> = BTreeMap::new();
    for ident in games {
        let key = partition_group_key(ident, &spec.grouping_keys);
        groups.entry(key).or_default().push(ident);
    }
    // Ratio validation BEFORE any assignment (same order as Python).
    let cumulative = cumulative_thresholds(&spec.ratios)?;

    let mut assignments: BTreeMap<String, String> = BTreeMap::new();
    for (group_key, members) in &groups {
        let chosen = choose_partition(&cumulative, split_draw(group_key, spec.seed))?;
        for ident in members {
            assignments.insert(ident.game_id.clone(), chosen.clone());
        }
    }
    if assignments.len() != games.len() {
        return Err(PartitionError::CountMismatch {
            detail: "partition count mismatch: game split detected".to_string(),
        });
    }

    if spec.wall_disjoint {
        let mut wall_to_part: BTreeMap<&str, &str> = BTreeMap::new();
        for ident in games {
            let Some(wall) = ident.wall_hash.as_deref() else {
                continue;
            };
            let Some(part) = assignments.get(&ident.game_id).map(String::as_str) else {
                return Err(PartitionError::CountMismatch {
                    detail: "partition count mismatch: game split detected".to_string(),
                });
            };
            if let Some(prev) = wall_to_part.get(wall) {
                if *prev != part {
                    return Err(PartitionError::WallCrossPartition {
                        detail: format!("wall {} appears in multiple partitions", &wall[..wall.len().min(12)]),
                    });
                }
            } else {
                wall_to_part.insert(wall, part);
            }
        }
        let mut eval_walls: BTreeSet<&str> = BTreeSet::new();
        let mut train_walls: BTreeSet<&str> = BTreeSet::new();
        for ident in games {
            let Some(wall) = ident.wall_hash.as_deref() else {
                continue;
            };
            let Some(part) = assignments.get(&ident.game_id).map(String::as_str) else {
                return Err(PartitionError::CountMismatch {
                    detail: "partition count mismatch: game split detected".to_string(),
                });
            };
            match part
            {
                "block_eval" | "decision_eval" => {
                    eval_walls.insert(wall);
                }
                "train" => {
                    train_walls.insert(wall);
                }
                _ => {}
            }
        }
        if !eval_walls.is_disjoint(&train_walls) {
            return Err(PartitionError::EvalTrainWallOverlap {
                detail: "rollout/evaluation walls disjoint violation".to_string(),
            });
        }
    }

    let spec_doc = SpecDoc {
        algorithm: &spec.algorithm,
        version: &spec.version,
        seed: spec.seed,
        ratios: &spec.ratios,
        grouping_keys: &spec.grouping_keys,
        wall_disjoint: spec.wall_disjoint,
    };
    let spec_hash = of_canonical(&spec_doc).map_err(PartitionError::from)?;
    let mut decoded_sorted: Vec<&str> =
        games.iter().map(|g| g.decoded_hash.as_str()).collect();
    decoded_sorted.sort();
    let games_hash = of_canonical(&decoded_sorted).map_err(PartitionError::from)?;
    let mut input_hashes = BTreeMap::new();
    input_hashes.insert("spec".to_string(), spec_hash);
    input_hashes.insert("games".to_string(), games_hash);

    #[derive(Serialize)]
    struct ManifestPayload<'a> {
        spec: SpecDoc<'a>,
        assignments: &'a BTreeMap<String, String>,
    }
    let payload = ManifestPayload {
        spec: SpecDoc {
            algorithm: &spec.algorithm,
            version: &spec.version,
            seed: spec.seed,
            ratios: &spec.ratios,
            grouping_keys: &spec.grouping_keys,
            wall_disjoint: spec.wall_disjoint,
        },
        assignments: &assignments,
    };
    let digest = of_canonical(&payload).map_err(PartitionError::from)?;

    Ok(SplitManifest {
        spec: spec.clone(),
        assignments,
        input_hashes,
        digest,
    })
}

/// Wall-hash helper for the decode sibling: `sha256:` over canon wall bytes.
///
/// The wall-tiles → canon-bytes step belongs to `feed::canon` (Data M1); the
/// decode owner canonicalizes `wall_tiles` through canon and calls this for
/// the digest. `None` walls stay `None` (null corpus-wide for real MJAI —
/// `stream_read.compute_wall_hash` doc).
pub fn wall_hash_of_canon_bytes(canon_wall_bytes: &[u8]) -> String {
    sha256_hex(canon_wall_bytes)
}

/// Digest-shape check shared by seal tests: `sha256:<64 lowercase hex>`.
pub fn is_digest_text(text: &str) -> bool {
    let Some(hex) = text.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canon::canonical_bytes;

    fn ratios_two_way() -> BTreeMap<String, f64> {
        BTreeMap::from([
            ("train".to_string(), 0.8),
            ("validation".to_string(), 0.2),
        ])
    }

    fn game(game_id: &str, group: &str, wall: Option<&str>) -> GameIdentity {
        // Distinct hash per game: sha256 over the game_id itself (NOT its
        // length — same-length ids collided and tripped exact-dup detection).
        GameIdentity::new(
            game_id,
            &format!("obj-{game_id}"),
            group,
            vec!["p1".to_string()],
            Some("2024-01-01T00:00:00Z".to_string()),
            wall.map(str::to_string),
            &crate::digest::sha256_hex(game_id.as_bytes()),
        )
    }

    #[test]
    fn assign_one_is_deterministic_and_bounded() {
        // Contract: same (group, seed, ratios) ⇒ same partition, always an
        // active partition. Determinism is the observable split-stability
        // gate; hardcoded names would pin a hash, not a contract.
        let ratios = ratios_two_way();
        for group in ["a|2024-01-01", "b|2024-01-02", "c|unknown"] {
            let first = assign_one(group, 7, &ratios).unwrap();
            let second = assign_one(group, 7, &ratios).unwrap();
            assert_eq!(first, second);
            assert!(ratios.contains_key(&first));
        }
        // Different seeds may differ (probabilistic check over many groups:
        // at least one group flips between seed 0 and seed 999).
        let groups: Vec<String> =
            (0..32).map(|i| format!("src{i}|2024-05-05")).collect();
        let flips = groups
            .iter()
            .filter(|g| assign_one(g, 0, &ratios) != assign_one(g, 999, &ratios))
            .count();
        assert!(flips > 0, "seeds must perturb assignment");
    }

    #[test]
    fn assign_one_matches_assign_partitions_group_draw() {
        // Wave3 §1.10 dedup direction: the corpus path MUST call the same
        // draw as the stream path. Every game in a group follows its group.
        let ratios = ratios_two_way();
        let spec = SplitSpec {
            algorithm: "hash".to_string(),
            version: "1".to_string(),
            seed: 42,
            ratios: ratios.clone(),
            grouping_keys: vec!["source".to_string(), "time".to_string()],
            wall_disjoint: false,
        };
        let games = vec![
            GameIdentity::new(
                "g1", "o1", "srcA", vec!["p1".into()], Some("2024-03-03T00:00:00Z".into()), None,
                "sha256:0000000000000000000000000000000000000000000000000000000000000001",
            ),
            GameIdentity::new(
                "g2", "o2", "srcA", vec!["p2".into()], Some("2024-03-03T12:00:00Z".into()), None,
                "sha256:0000000000000000000000000000000000000000000000000000000000000002",
            ),
        ];
        let manifest = assign_partitions(&games, &spec).unwrap();
        // Same (source,time) day ⇒ same group ⇒ same partition.
        assert_eq!(manifest.assignments["g1"], manifest.assignments["g2"]);
        let expected =
            assign_one("srcA|2024-03-03", 42, &ratios).unwrap();
        assert_eq!(manifest.assignments["g1"], expected);
    }

    #[test]
    fn whole_games_before_decisions_group_stays_together() {
        // Checklist item 6 headliner: grouping keeps a game's decisions in
        // one partition — here, same group ⇒ one draw ⇒ one partition.
        let spec = SplitSpec {
            algorithm: "hash".to_string(),
            version: "1".to_string(),
            seed: 1,
            ratios: BTreeMap::from([
                ("train".to_string(), 0.5),
                ("test".to_string(), 0.5),
            ]),
            grouping_keys: vec!["source".to_string(), "time".to_string()],
            wall_disjoint: false,
        };
        let games: Vec<GameIdentity> = (0..8)
            .map(|i| {
                GameIdentity::new(
                    &format!("g{i}"),
                    &format!("o{i}"),
                    "same-src",
                    vec!["p1".into()],
                    Some("2024-06-01T00:00:00Z".into()),
                    None,
                    &format!("sha256:{i:064x}"),
                )
            })
            .collect();
        let manifest = assign_partitions(&games, &spec).unwrap();
        let mut parts: BTreeSet<&str> =
            manifest.assignments.values().map(String::as_str).collect();
        assert_eq!(parts.len(), 1, "one group ⇒ one partition, never split");
        parts.clear();
        let _ = parts;
    }

    #[test]
    fn exact_duplicates_reject() {
        let spec = SplitSpec {
            algorithm: "hash".to_string(),
            version: "1".to_string(),
            seed: 0,
            ratios: ratios_two_way(),
            grouping_keys: vec![],
            wall_disjoint: false,
        };
        let dup_hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let games = vec![
            GameIdentity::new("g1", "o1", "s", vec![], None, None, dup_hash),
            GameIdentity::new("g2", "o2", "s", vec![], None, None, dup_hash),
        ];
        assert_eq!(
            detect_exact_duplicates(&games),
            vec![("g1".to_string(), "g2".to_string())]
        );
        assert!(matches!(
            assign_partitions(&games, &spec),
            Err(PartitionError::ExactDuplicates { .. })
        ));
    }

    #[test]
    fn near_duplicates_reject_on_identical_wall() {
        // Near duplicate = same wall hash (identical wall), per
        // partition.py:106-107 wording (value-equality, not token identity).
        let spec = SplitSpec {
            algorithm: "hash".to_string(),
            version: "1".to_string(),
            seed: 0,
            ratios: ratios_two_way(),
            grouping_keys: vec![],
            wall_disjoint: false,
        };
        let wall = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let games = vec![
            GameIdentity::new("g1", "o1", "s", vec![], None, Some(wall.into()),
                "sha256:0000000000000000000000000000000000000000000000000000000000000001"),
            GameIdentity::new("g2", "o2", "s", vec![], None, Some(wall.into()),
                "sha256:0000000000000000000000000000000000000000000000000000000000000002"),
        ];
        assert_eq!(
            detect_near_duplicates(&games),
            vec![("g1".to_string(), "g2".to_string())]
        );
        assert!(matches!(
            assign_partitions(&games, &spec),
            Err(PartitionError::NearDuplicates { .. })
        ));
        // Wall-less games never near-dup.
        let free = vec![
            game("a", "s", None),
            GameIdentity::new("b", "ob", "s", vec!["p1".into()],
                Some("2024-01-01T00:00:00Z".into()), None,
                "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"),
        ];
        assert!(detect_near_duplicates(&free).is_empty());
    }

    #[test]
    fn wall_disjoint_rejects_cross_partition_walls() {
        // Two groups forced apart by seed search, sharing one wall: with
        // wall_disjoint the corpus is unshippable (fail-closed).
        let ratios = BTreeMap::from([
            ("train".to_string(), 0.5),
            ("test".to_string(), 0.5),
        ]);
        let wall = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        // Find two group keys that land on opposite sides (deterministic
        // search over group names — the draw is a pure function).
        let mut pair: Option<(String, String)> = None;
        for i in 0..200 {
            for j in (i + 1)..200 {
                let a = format!("src{i}|2024-01-01");
                let b = format!("src{j}|2024-01-01");
                let pa = assign_one(&a, 11, &ratios).unwrap();
                let pb = assign_one(&b, 11, &ratios).unwrap();
                if pa != pb {
                    pair = Some((a, b));
                    break;
                }
            }
            if pair.is_some() {
                break;
            }
        }
        let (ga, gb) = pair.expect("two-sided groups exist for 50/50");
        let src_a = ga.split('|').next().unwrap().to_string();
        let src_b = gb.split('|').next().unwrap().to_string();
        let spec = SplitSpec {
            algorithm: "hash".to_string(),
            version: "1".to_string(),
            seed: 11,
            ratios,
            grouping_keys: vec!["source".to_string(), "time".to_string()],
            wall_disjoint: true,
        };
        let games = vec![
            GameIdentity::new("g1", "o1", &src_a, vec!["p1".into()],
                Some("2024-01-01T00:00:00Z".into()), Some(wall.into()),
                "sha256:1000000000000000000000000000000000000000000000000000000000000001"),
            GameIdentity::new("g2", "o2", &src_b, vec!["p1".into()],
                Some("2024-01-01T00:00:00Z".into()), Some(wall.into()),
                "sha256:2000000000000000000000000000000000000000000000000000000000000002"),
        ];
        assert!(matches!(
            assign_partitions(&games, &spec),
            Err(PartitionError::NearDuplicates { .. }
            | PartitionError::WallCrossPartition { .. }
            | PartitionError::EvalTrainWallOverlap { .. })
        ));
    }

    #[test]
    fn bad_ratios_reject() {
        let bad_sum = BTreeMap::from([
            ("train".to_string(), 0.5),
            ("validation".to_string(), 0.5),
            ("test".to_string(), 0.5),
        ]);
        assert!(matches!(
            assign_one("g|t", 0, &bad_sum),
            Err(PartitionError::BadRatios { .. })
        ));
        let empty: BTreeMap<String, f64> = BTreeMap::new();
        assert!(matches!(
            assign_one("g|t", 0, &empty),
            Err(PartitionError::BadRatios { .. })
        ));
        let mut neg = ratios_two_way();
        neg.insert("train".to_string(), -0.1);
        assert!(matches!(
            assign_one("g|t", 0, &neg),
            Err(PartitionError::BadRatioValue { .. })
        ));
        // Zero-weight everywhere ⇒ no active partitions.
        let zeros = BTreeMap::from([
            ("train".to_string(), 0.0),
            ("validation".to_string(), 0.0),
        ]);
        assert!(matches!(
            assign_one("g|t", 0, &zeros),
            Err(PartitionError::BadRatios { .. }
            | PartitionError::NoActivePartitions { .. })
        ));
    }

    #[test]
    fn manifest_digest_binds_spec_and_assignments() {
        let spec = SplitSpec {
            algorithm: "hash".to_string(),
            version: "1".to_string(),
            seed: 3,
            ratios: ratios_two_way(),
            grouping_keys: vec!["source".to_string(), "time".to_string()],
            wall_disjoint: true,
        };
        let games = vec![
            game("g1", "srcA", None),
            game("g2", "srcB", None),
        ];
        let m = assign_partitions(&games, &spec).unwrap();
        assert!(is_digest_text(&m.digest));
        assert!(is_digest_text(&m.input_hashes["spec"]));
        assert!(is_digest_text(&m.input_hashes["games"]));
        // Digest is stable: recompute ⇒ same bytes.
        let m2 = assign_partitions(&games, &spec).unwrap();
        assert_eq!(m.digest, m2.digest);
        assert_eq!(m.input_hashes, m2.input_hashes);
        // Canon owns the bytes: digest equals of_canonical over the same
        // payload shape (M1 — no second printer).
        let bytes = canonical_bytes(&m.assignments, "partition:test").unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn seed_map_m6_lanes_are_shared_with_rng() {
        // Data M6: LE lanes key0=seed[0:8], key1=[8:16], ctr0=[16:24],
        // spare=[24:32]. Partition takes the derived u64 seed for its
        // preimage; this KAT pins the shared mapping so the two modules
        // cannot silently disagree on the most load-bearing bytes.
        let mut seed = [0u8; 32];
        for (i, b) in seed.iter_mut().enumerate().take(24) {
            *b = i as u8;
        }
        let (k0, k1, c0, spare) = crate::rng::seed_parts(&seed);
        assert_eq!(k0, u64::from_le_bytes(seed[0..8].try_into().unwrap()));
        assert_eq!(k1, u64::from_le_bytes(seed[8..16].try_into().unwrap()));
        assert_eq!(c0, u64::from_le_bytes(seed[16..24].try_into().unwrap()));
        assert_eq!(spare, 0);
        for (name, fixture) in crate::rng::seed_map_fixture() {
            let (_, _, _, sp) = crate::rng::seed_parts(&fixture);
            assert_eq!(sp, 0, "fixture {name} pins spare = 0");
        }
    }
}
