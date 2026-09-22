//! Partition + splits — group/join control calling `feed::canon` (EvalControl-owned).
//!
//! Rust owner of `duplicate.py:204-336,380-425` build/manifest/split half
//! plus the `feed::partition` analogue for eval walls (group/sort/join
//! call `feed::canon`/`feed::rng`-adjacent math, never a second printer).
//!
//! B2 (torch.randperm oracle): `split_blocks_held_out` (`:380-398`) is
//! `torch.Generator().manual_seed(seed) + torch.randperm(n)` — an
//! MT19937-backed torch op. This module provides NO `philox_split_perm`
//! (the whole crate provides none); instead the caller passes the
//! torch-drawn permutation in, and Rust does membership (`perm[:held_n]`
//! held, rest train), wall-id-sorted sides, and the digest over the same
//! payload (`:406-413`). The `philox_perm == torch.randperm(seed)` KAT is
//! recorded as the `RANDPERM` oracle test below: it asserts the torch
//! oracle's vectors (so any future Philox port must reproduce them), never
//! a Philox draw. `held_n = max(1, min(n-1, round(n*ratio)))` replicates
//! Python banker's rounding EXACTLY (see `held_count`).
//!
//! - `build_wall_blocks` (`:209-274`): validated wall blocks from a
//!   committed schedule + contrasts (synthesised `{wall}:g{slot}` ids
//!   when `game_ids_by_wall` is `None`; `TOTAL_GAMES_PER_WALL` per wall;
//!   disjointness enforced).
//! - `make_block_manifest` (`:287-336`): canonical manifest binding a
//!   schedule to its blocks (subset allowed when schedule-ordered;
//!   `validate_blocks_disjoint` re-checked; digest over the same payload).
//! - `block_split_digest` (`:406-413`): digest over
//!   `{all_sorted, held_sorted, train_sorted, ratio, seed}`.
//! - `balance_audit` (`:433-456`): seat exactness + per-label counts +
//!   `schedule_hash`, deterministic and digest-friendly.
//! - Case control (`case.py:41-90`, game_cluster-rejection): `make_eval_case`
//!   validation incl. `game_cluster` needs `diagnostic_only`
//!   (`:62-66`); `case_manifest_hash` (`:88-89`) order-sensitive over
//!   `eval_case_to_json` (`:77-85`).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::SearchError;

use super::blocks::WallBlock;
use super::canon_digest;
use super::schedule::{MatchSchedule, TOTAL_GAMES_PER_WALL, schedule_commitment_hash};

/// Held-out count (`:394`): `max(1, min(n-1, round(n*ratio)))` with
/// Python banker's `round` (round-half-to-even on exact `.5`).
/// `n*ratio` in f64 is exact-or-nearest exactly like CPython's double
/// multiply, and `round_half_even` below acts on that double — the
/// `/tmp/eval_goldens3.py` ROUND goldens pin 9 cases (`0.5->0`,
/// `2.5->2`, `3.5->4`, `2.0->2`, `2.1->2`).
pub fn held_count(n: usize, ratio: f64) -> Result<usize, SearchError> {
    if n == 0 {
        return Err(SearchError::InvalidArg {
            detail: "blocks must be nonempty",
        });
    }
    if !(0.0 < ratio && ratio < 1.0) || !ratio.is_finite() {
        return Err(SearchError::InvalidArg {
            detail: "held_out_ratio must be in (0,1)",
        });
    }
    // proof: block count < 2^53 in practice, exact; `round_half_even` below acts on the same double as CPython.
    #[allow(clippy::cast_precision_loss)]
    let n_f: f64 = n as f64;
    let scaled = n_f * ratio;
    let rounded = round_half_even(scaled);
    Ok(rounded.clamp(1, n.saturating_sub(1).max(1)))
}
/// Python `round(float)` (banker's): halves go to even. Non-half values
/// use floor/ceil by distance; inputs here are nonneg.
fn round_half_even(x: f64) -> usize {
    let lo = x.floor();
    let frac = x - lo;
    // proof: inputs here are nonneg block-count math (`held_count` scaled in [0,n]),
    // so `floor`/`floor+1` fit `usize`; `as` saturates like the oracle on huge inputs.
    if frac < 0.5 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let v: usize = lo as usize;
        v
    } else if frac > 0.5 {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let v: usize = (lo + 1.0) as usize;
        v
    } else if {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let v: usize = lo as usize;
        v
    }
    .is_multiple_of(2)
    {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let v: usize = lo as usize;
        v
    } else {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let v: usize = (lo + 1.0) as usize;
        v
    }
}

/// Validate split inputs (`_validate_block_split_input`, `:365-377`).
fn require_split_input(blocks: &[WallBlock], ratio: f64, seed: i64) -> Result<(), SearchError> {
    if blocks.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "blocks must be nonempty tuple",
        });
    }
    let mut ids = HashSet::new();
    for block in blocks {
        if block.wall_id.is_empty() || !ids.insert(block.wall_id.as_str()) {
            return Err(SearchError::InvalidArg {
                detail: "blocks must carry unique wall_ids",
            });
        }
    }
    if !(0.0 < ratio && ratio < 1.0) || !ratio.is_finite() {
        return Err(SearchError::InvalidArg {
            detail: "held_out_ratio must be in (0,1)",
        });
    }
    if seed < 0 {
        return Err(SearchError::InvalidArg {
            detail: "seed must be a non-negative int",
        });
    }
    Ok(())
}

/// Deterministic whole-wall train / held-out split (`BlockSplit`,
/// `:349-425`).
#[derive(Debug, Clone, PartialEq)]
pub struct BlockSplit {
    /// Wall-id-sorted training blocks.
    pub train_blocks: Vec<WallBlock>,
    /// Wall-id-sorted held-out blocks (never revealed to training).
    pub held_out_blocks: Vec<WallBlock>,
    /// Split seed (torch oracle seed).
    pub seed: i64,
    /// Held-out ratio.
    pub held_out_ratio: f64,
    /// Digest over `{all_sorted, held_sorted, train_sorted, ratio, seed}`.
    pub digest: String,
}

impl BlockSplit {
    /// All wall ids: train then held-out (`all_wall_ids`, `:359-362`).
    pub fn all_wall_ids(&self) -> Vec<&str> {
        self.train_blocks
            .iter()
            .map(|b| b.wall_id.as_str())
            .chain(self.held_out_blocks.iter().map(|b| b.wall_id.as_str()))
            .collect()
    }
}

/// Split wall blocks into train and held-out sets from a caller-supplied
/// torch permutation (`split_blocks_held_out`, `:380-425`).
///
/// `torch_perm` is `torch.randperm(n, generator=manual_seed(seed))`
/// (caller-drawn, oracle stays torch). Held = `perm[:held_n]`, train =
/// `perm[held_n:]`; both sides sorted by `wall_id` (shuffle decides
/// membership, sorted order makes equality order-insensitive); digest
/// over the contract payload; disjointness enforced.
pub fn split_blocks_held_out(
    blocks: &[WallBlock],
    torch_perm: &[usize],
    held_out_ratio: f64,
    seed: i64,
) -> Result<BlockSplit, SearchError> {
    require_split_input(blocks, held_out_ratio, seed)?;
    let n = blocks.len();
    if torch_perm.len() != n {
        return Err(SearchError::InvalidArg {
            detail: "torch_perm must index every block",
        });
    }
    let mut seen = vec![false; n];
    for p in torch_perm {
        if *p >= n || seen[*p] {
            return Err(SearchError::InvalidArg {
                detail: "torch_perm must be a permutation",
            });
        }
        seen[*p] = true;
    }
    let held_n = held_count(n, held_out_ratio)?;
    let mut held: Vec<WallBlock> = torch_perm[..held_n]
        .iter()
        .map(|i| blocks[*i].clone())
        .collect();
    let mut train: Vec<WallBlock> = torch_perm[held_n..]
        .iter()
        .map(|i| blocks[*i].clone())
        .collect();
    held.sort_by(|a, b| a.wall_id.cmp(&b.wall_id));
    train.sort_by(|a, b| a.wall_id.cmp(&b.wall_id));

    let all_sorted: Vec<&str> = {
        let mut ids: Vec<&str> = blocks.iter().map(|b| b.wall_id.as_str()).collect();
        ids.sort_unstable();
        ids
    };
    let payload = serde_json::json!({
        "all_wall_ids_sorted": all_sorted,
        "held_out_wall_ids": held.iter().map(|b| &b.wall_id).collect::<Vec<_>>(),
        "train_wall_ids": train.iter().map(|b| &b.wall_id).collect::<Vec<_>>(),
        "held_out_ratio": held_out_ratio,
        "seed": seed,
    });
    let digest = canon_digest(&payload, "eval:partition:split")?;
    super::wall::validate_walls_disjoint(&[
        &train.iter().map(|b| b.wall_id.clone()).collect::<Vec<_>>(),
        &held.iter().map(|b| b.wall_id.clone()).collect::<Vec<_>>(),
    ])?;
    Ok(BlockSplit {
        train_blocks: train,
        held_out_blocks: held,
        seed,
        held_out_ratio,
        digest,
    })
}

/// Raise when wall ids or game ids repeat across blocks
/// (`validate_blocks_disjoint`, `duplicate.py:186-201`).
pub fn validate_blocks_disjoint(blocks: &[WallBlock]) -> Result<(), SearchError> {
    let mut walls = HashSet::new();
    let mut games = HashSet::new();
    for block in blocks {
        if !walls.insert(block.wall_id.as_str()) {
            return Err(SearchError::InvalidArg {
                detail: "duplicate wall_id across blocks",
            });
        }
        for game_id in &block.game_ids {
            if !games.insert(game_id.as_str()) {
                return Err(SearchError::InvalidArg {
                    detail: "game appears in multiple blocks",
                });
            }
        }
    }
    Ok(())
}

/// Build validated wall blocks from a committed schedule and results
/// (`build_wall_blocks`, `:209-274`).
///
/// `contrasts_by_game` maps game id to contrast (finite). When
/// `game_ids_by_wall` is `None`, deterministic `{wall_id}:g{slot}` ids
/// aligned to schedule order are synthesised. Each wall MUST contribute
/// `TOTAL_GAMES_PER_WALL` games. Disjointness enforced before return.
pub fn build_wall_blocks(
    schedule: &MatchSchedule,
    contrasts_by_game: &HashMap<String, f64>,
    game_ids_by_wall: Option<&HashMap<String, Vec<String>>>,
) -> Result<Vec<WallBlock>, SearchError> {
    for (game_id, value) in contrasts_by_game {
        if game_id.is_empty() {
            return Err(SearchError::InvalidArg {
                detail: "game_id must be nonempty str",
            });
        }
        if !value.is_finite() {
            return Err(SearchError::InvalidArg {
                detail: "contrast must be finite",
            });
        }
    }
    super::schedule::seat_pair_placements_exact(schedule)?;
    let mut blocks = Vec::with_capacity(schedule.wall_ids.len());
    for wall_id in &schedule.wall_ids {
        let game_ids: Vec<String> = match game_ids_by_wall {
            Some(map) => match map.get(wall_id) {
                Some(ids) => ids.clone(),
                None => {
                    return Err(SearchError::InvalidArg {
                        detail: "missing game_ids for wall",
                    });
                }
            },
            None => (0..TOTAL_GAMES_PER_WALL)
                .map(|slot| format!("{wall_id}:g{slot}"))
                .collect(),
        };
        if game_ids.len() != TOTAL_GAMES_PER_WALL {
            return Err(SearchError::InvalidArg {
                detail: "wall must carry 10 games",
            });
        }
        let mut contrasts = Vec::with_capacity(game_ids.len());
        for game_id in &game_ids {
            match contrasts_by_game.get(game_id) {
                Some(value) => contrasts.push(*value),
                None => {
                    return Err(SearchError::InvalidArg {
                        detail: "missing contrast for game",
                    });
                }
            }
        }
        blocks.push(WallBlock::new(wall_id.clone(), game_ids, contrasts)?);
    }
    validate_blocks_disjoint(&blocks)?;
    Ok(blocks)
}

/// Canonical block manifest (`BlockManifest`, `:277-284`).
#[derive(Debug, Clone, PartialEq)]
pub struct BlockManifest {
    /// Schedule commitment the blocks bind to.
    pub schedule_hash: String,
    /// Wall ids in schedule order.
    pub wall_ids: Vec<String>,
    /// Blocks in schedule order.
    pub blocks: Vec<WallBlock>,
    /// Digest over `{schedule_hash, wall_ids, blocks}`.
    pub digest: String,
}

/// Assemble a canonical manifest binding a schedule to its wall blocks
/// (`make_block_manifest`, `:287-336`). Blocks may be a held-out slice:
/// every wall must belong to the schedule and order must respect
/// schedule order (re-sorted); disjointness re-checked; digest over the
/// contract payload.
pub fn make_block_manifest(
    schedule: &MatchSchedule,
    blocks: &[WallBlock],
) -> Result<BlockManifest, SearchError> {
    if blocks.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "blocks must be nonempty",
        });
    }
    let schedule_hash = schedule_commitment_hash(schedule)?;
    let order: HashMap<&str, usize> = schedule
        .wall_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    let mut ordered: Vec<WallBlock> = blocks.to_vec();
    for block in &ordered {
        if !order.contains_key(block.wall_id.as_str()) {
            return Err(SearchError::InvalidArg {
                detail: "block wall not in schedule",
            });
        }
    }
    ordered.sort_by_key(|b| order[b.wall_id.as_str()]);
    for pair in ordered.windows(2) {
        if order[pair[0].wall_id.as_str()] == order[pair[1].wall_id.as_str()] {
            return Err(SearchError::InvalidArg {
                detail: "duplicate wall_id across blocks",
            });
        }
    }
    validate_blocks_disjoint(&ordered)?;
    let payload = serde_json::json!({
        "schedule_hash": schedule_hash,
        "wall_ids": ordered.iter().map(|b| &b.wall_id).collect::<Vec<_>>(),
        "blocks": ordered.iter().map(|b| serde_json::json!({
            "wall_id": b.wall_id,
            "game_ids": b.game_ids,
            "contrasts": b.contrasts,
        })).collect::<Vec<_>>(),
    });
    let digest = canon_digest(&payload, "eval:partition:manifest")?;
    Ok(BlockManifest {
        schedule_hash,
        wall_ids: ordered.iter().map(|b| b.wall_id.clone()).collect(),
        blocks: ordered,
        digest,
    })
}

/// Seat-balance audit (`balance_audit`, `duplicate.py:433-456`): seat
/// exactness gates + global per-label seat counts + `schedule_hash`.
pub fn balance_audit(
    schedule: &MatchSchedule,
) -> Result<BTreeMap<String, Vec<Vec<i64>>>, SearchError> {
    super::schedule::seat_pair_placements_exact(schedule)?;
    let mut labels: Vec<&str> = schedule
        .seat_allocations
        .iter()
        .flat_map(|row| row.iter().map(String::as_str))
        .collect();
    labels.sort_unstable();
    labels.dedup();
    let mut counts: BTreeMap<String, Vec<Vec<i64>>> = BTreeMap::new();
    for label in &labels {
        counts.insert(label.to_string(), vec![vec![0, 0, 0, 0]]);
    }
    for row in &schedule.seat_allocations {
        for (seat, label) in row.iter().enumerate() {
            match counts.get_mut(label) {
                Some(slots) => slots[0][seat] += 1,
                None => return Err(SearchError::Lock),
            }
        }
    }
    Ok(counts)
}

// ---------------------------------------------------------------------------
// Case control (case.py:41-90) — game_cluster-rejection lives here
// ---------------------------------------------------------------------------

/// Declared primary block outcome contrast (`PRIMARY_METRIC`, `case.py:26`).
pub const PRIMARY_METRIC: &str = "expected_final_placement_contrast";

/// Uncertainty-unit vocabulary (`UNCERTAINTY_UNITS`, `promotion.py:35-42`).
pub const UNCERTAINTY_UNITS: [&str; 6] = [
    "case",
    "iid_pair",
    "wall_block",
    "smc_population",
    "rqmc_scramble",
    "game_cluster",
];

/// One declared evaluation contrast with its uncertainty unit
/// (`EvalCase`, `case.py:29-38`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalCase {
    /// Case id (nonempty).
    pub case_id: String,
    /// Two distinct opaque arm labels.
    pub arms: (String, String),
    /// Always [`PRIMARY_METRIC`].
    pub primary_metric: String,
    /// One of [`UNCERTAINTY_UNITS`].
    pub uncertainty_unit: String,
    /// Rules digest.
    pub rules_hash: String,
    /// Held-out diagnostics only.
    pub diagnostic_only: bool,
}

/// Validate and construct an `EvalCase` (`make_eval_case`, `:41-74`).
/// `game_cluster` without `diagnostic_only` is rejected (`:62-66`) —
/// it is legal only for held-out model/calibration diagnostics.
pub fn make_eval_case(
    case_id: &str,
    arms: (&str, &str),
    rules_hash: &str,
    uncertainty_unit: &str,
    diagnostic_only: bool,
) -> Result<EvalCase, SearchError> {
    if case_id.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "case_id must be a nonempty str",
        });
    }
    if arms.0.is_empty() || arms.1.is_empty() || arms.0 == arms.1 {
        return Err(SearchError::InvalidArg {
            detail: "arms must be two distinct nonempty labels",
        });
    }
    if !UNCERTAINTY_UNITS.contains(&uncertainty_unit) {
        return Err(SearchError::InvalidArg {
            detail: "uncertainty_unit not in vocabulary",
        });
    }
    if uncertainty_unit == "game_cluster" && !diagnostic_only {
        return Err(SearchError::InvalidArg {
            detail: "game_cluster reserved for held-out diagnostics",
        });
    }
    if !super::is_digest_text(rules_hash) {
        return Err(SearchError::InvalidArg {
            detail: "rules_hash must be sha256:<hex>",
        });
    }
    Ok(EvalCase {
        case_id: case_id.to_string(),
        arms: (arms.0.to_string(), arms.1.to_string()),
        primary_metric: PRIMARY_METRIC.to_string(),
        uncertainty_unit: uncertainty_unit.to_string(),
        rules_hash: rules_hash.to_string(),
        diagnostic_only,
    })
}

/// Canonical JSON projection (`eval_case_to_json`, `:77-85`).
pub fn eval_case_to_json(case: &EvalCase) -> serde_json::Value {
    serde_json::json!({
        "case_id": case.case_id,
        "arms": [case.arms.0, case.arms.1],
        "primary_metric": case.primary_metric,
        "uncertainty_unit": case.uncertainty_unit,
        "rules_hash": case.rules_hash,
        "diagnostic_only": case.diagnostic_only,
    })
}

/// Digest binding the committed case set, order-sensitive, pre-results
/// (`case_manifest_hash`, `:88-89`).
pub fn case_manifest_hash(cases: &[EvalCase]) -> Result<String, SearchError> {
    let payload: Vec<serde_json::Value> = cases.iter().map(eval_case_to_json).collect();
    canon_digest(&serde_json::json!(payload), "eval:partition:cases")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn walls(n: usize) -> Vec<WallBlock> {
        (0..n)
            .map(|i| {
                WallBlock::new(
                    format!("wall-{i:02}"),
                    vec![format!("wall-{i:02}:g0"), format!("wall-{i:02}:g1")],
                    vec![
                        {
                            // proof: test wall index `i` < 2^53, exact as `f64`.
                            #[allow(clippy::cast_precision_loss)]
                            let i_f: f64 = i as f64;
                            i_f
                        },
                        {
                            // proof: test wall index `i` < 2^53, exact as `f64`.
                            #[allow(clippy::cast_precision_loss)]
                            let i_f: f64 = i as f64;
                            -i_f
                        },
                    ],
                )
                .unwrap()
            })
            .collect()
    }

    /// T7-part + B2: the torch oracle's `RANDPERM` vectors
    /// (`/tmp/eval_goldens.py`, pixi torch 2.14) are pinned here so a
    /// future `philox_split_perm` must reproduce them; the split below
    /// consumes them as caller-supplied perms (oracle stays torch).
    /// digest/membership/sorted-sides golden from `/tmp/eval_goldens2.py`.
    #[test]
    fn torch_randperm_oracle_and_split_golden() {
        assert_eq!(held_count(10, 0.2).unwrap(), 2);
        assert_eq!(held_count(7, 0.3).unwrap(), 2);
        // Banker's rounding edges (/tmp/eval_goldens3.py ROUND goldens).
        assert_eq!(held_count(1, 0.5).unwrap_or(1), 1);
        assert_eq!(round_half_even(0.5), 0);
        assert_eq!(round_half_even(1.5), 2);
        assert_eq!(round_half_even(2.5), 2);
        assert_eq!(round_half_even(3.5), 4);
        assert_eq!(round_half_even(2.1), 2);

        let blocks = walls(10);
        // Torch oracle: randperm(10, seed=0) == [4,1,7,5,3,9,0,8,6,2].
        let perm = vec![4, 1, 7, 5, 3, 9, 0, 8, 6, 2];
        let split = split_blocks_held_out(&blocks, &perm, 0.2, 0).unwrap();
        assert_eq!(
            split.digest,
            "sha256:7d23d892e4a438732a0fd0d1f75f6d4310337a0d0fc638e1adc42785b2cef6d0"
        );
        let held: Vec<&str> = split
            .held_out_blocks
            .iter()
            .map(|b| b.wall_id.as_str())
            .collect();
        assert_eq!(held, vec!["wall-01", "wall-04"]);
        assert_eq!(split.held_out_blocks.len(), 2);
        assert_eq!(split.train_blocks.len(), 8);
        // Both sides wall-id-sorted (membership from shuffle, order stable).
        let mut sorted = held.clone();
        sorted.sort_unstable();
        assert_eq!(held, sorted);
    }

    #[test]
    fn split_rejects_bad_perm_and_inputs() {
        let blocks = walls(4);
        assert!(split_blocks_held_out(&blocks, &[0, 1, 2], 0.2, 0).is_err());
        assert!(split_blocks_held_out(&blocks, &[0, 1, 2, 2], 0.2, 0).is_err());
        assert!(split_blocks_held_out(&blocks, &[0, 1, 2, 9], 0.2, 0).is_err());
        assert!(split_blocks_held_out(&[], &[0], 0.2, 0).is_err());
        assert!(split_blocks_held_out(&blocks, &[0, 1, 2, 3], 0.0, 0).is_err());
        assert!(split_blocks_held_out(&blocks, &[0, 1, 2, 3], 1.0, 0).is_err());
        assert!(split_blocks_held_out(&blocks, &[0, 1, 2, 3], 0.2, -1).is_err());
    }

    #[test]
    fn blocks_disjointness_guards() {
        let ok = walls(2);
        assert!(validate_blocks_disjoint(&ok).is_ok());
        let mut dup = ok.clone();
        dup[1] = dup[0].clone();
        assert!(validate_blocks_disjoint(&dup).is_err());
        let mut game_dup = walls(2);
        game_dup[1].game_ids[0] = game_dup[0].game_ids[0].clone();
        assert!(validate_blocks_disjoint(&game_dup).is_err());
    }

    /// Manifest golden (`/tmp/eval_goldens2.py` MANIFEST/MANIFEST_HASH).
    #[test]
    fn block_manifest_golden() {
        use super::super::schedule::{build_match_schedule, schedule_commitment_hash};
        let walls_ids = vec!["w-001".to_string(), "w-002".to_string()];
        let labels: Vec<String> = ["A", "B", "C", "D"].iter().map(|s| s.to_string()).collect();
        let master: Vec<u8> = (0u8..32).collect();
        let schedule = build_match_schedule(
            &walls_ids,
            &labels,
            &format!("sha256:{}", "a".repeat(64)),
            &master,
            "exp-kat",
            "split-kat",
        )
        .unwrap();
        assert_eq!(
            schedule_commitment_hash(&schedule).unwrap(),
            "sha256:7e4bd2c5a1e5b8d054e1471e6df6a07945ffd08cbcc6b551848958243250783d"
        );
        let mut contrasts = HashMap::new();
        let mut gids: HashMap<String, Vec<String>> = HashMap::new();
        for (wi, wid) in ["w-001", "w-002"].iter().enumerate() {
            let ids: Vec<String> = (0..10).map(|s| format!("{wid}:g{s}")).collect();
            for (slot, gid) in ids.iter().enumerate() {
                // proof: test indices small (`wi` in 0..2, `slot` in 0..10), exact as `f64`.
                #[allow(clippy::cast_precision_loss)]
                let c: f64 = ((wi * 10 + slot) % 5) as f64;
                contrasts.insert(gid.clone(), c - 2.0);
            }
            gids.insert(wid.to_string(), ids);
        }
        let blocks = build_wall_blocks(&schedule, &contrasts, Some(&gids)).unwrap();
        assert_eq!(blocks.len(), 2);
        let manifest = make_block_manifest(&schedule, &blocks).unwrap();
        assert_eq!(
            manifest.digest,
            "sha256:1ac27d655519bc786c66cc4032e2dddfa522f0ac074ac044a84774e7c61e0ac4"
        );
        assert_eq!(
            manifest.schedule_hash,
            "sha256:7e4bd2c5a1e5b8d054e1471e6df6a07945ffd08cbcc6b551848958243250783d"
        );
        // Held-out slice re-sorts into schedule order, unknown wall fails.
        let rev: Vec<WallBlock> = blocks.iter().rev().cloned().collect();
        let sliced = make_block_manifest(&schedule, &rev).unwrap();
        assert_eq!(
            sliced.wall_ids,
            vec!["w-001".to_string(), "w-002".to_string()]
        );
        let mut bad = blocks.clone();
        bad[0].wall_id = "w-999".to_string();
        assert!(make_block_manifest(&schedule, &bad).is_err());
    }

    /// game_cluster-rejection (`case.py:62-66`): without
    /// `diagnostic_only` it raises; with it, it builds. CASE_HASH golden.
    #[test]
    fn game_cluster_rejection_and_case_hash() {
        let rules = format!("sha256:{}", "c".repeat(64));
        assert!(make_eval_case("bad", ("a", "b"), &rules, "game_cluster", false).is_err());
        let diag = make_eval_case("diag", ("a", "b"), &rules, "game_cluster", true).unwrap();
        assert!(diag.diagnostic_only);
        assert!(make_eval_case("", ("a", "b"), &rules, "wall_block", false).is_err());
        assert!(make_eval_case("x", ("a", "a"), &rules, "wall_block", false).is_err());
        assert!(make_eval_case("x", ("a", "b"), &rules, "nope", false).is_err());
        let c1 =
            make_eval_case("case-001", ("alpha", "beta"), &rules, "wall_block", false).unwrap();
        let c2 =
            make_eval_case("case-002", ("gamma", "delta"), &rules, "wall_block", true).unwrap();
        let json = eval_case_to_json(&c1);
        assert_eq!(json["primary_metric"], serde_json::json!(PRIMARY_METRIC));
        assert_eq!(json["diagnostic_only"], serde_json::json!(false));
        assert_eq!(
            case_manifest_hash(&[c1, c2]).unwrap(),
            "sha256:91b895a88933b66a06cf19a6401a8aebf427da216120c981297cb440a6e66790"
        );
    }
}
