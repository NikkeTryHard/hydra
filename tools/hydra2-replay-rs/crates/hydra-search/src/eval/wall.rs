//! Wall identity: hash / fingerprint / sorts(136) (EvalControl-owned).
//!
//! Rust owner of `duplicate.py:79-107,110-184` wall-identity half.
//! B3 canon-wins: digests call `feed::canon` + `feed::digest` (the single
//! identity site) via `super::canon_digest`. No second printer here.
//!
//! - `wall_hash_from_tiles` (`:79-91`): order-SENSITIVE digest over the
//!   136 tile ids. Any reordering changes the hash.
//! - `wall_fingerprint` (`:94-107`): order-INSENSITIVE digest over the
//!   sorted multiset (`sort_unstable`, 136 small ints — same bytes as
//!   Python `sorted()`; parity test pins 10k-wall agreement via goldens).
//! - `find_exact_duplicates` (`:110-127`) / `find_near_duplicates`
//!   (`:130-152`): first-seen discovery order, same pairs as Python.
//! - `validate_walls_disjoint` (`:166-184`): wall sets must be disjoint
//!   (e.g. train vs evaluation ledger reuse).
//!
//! B2: no split permutation lives here. Splits stay the `torch.randperm`
//! oracle (`super::partition` takes caller-supplied perms).

use std::collections::HashMap;

use crate::SearchError;

use super::{canon_digest, is_digest_text};

/// Physical wall length (tiles `0..136`).
pub const WALL_TILES: usize = 136;

/// Digest of a 136-length wall (physical tile ids `0..135`).
///
/// Order-sensitive: any reordering changes the hash. Mirrors
/// `wall_hash_from_tiles` (`duplicate.py:79-91`).
pub fn wall_hash_from_tiles(tiles: &[u32]) -> Result<String, SearchError> {
    require_wall_tiles(tiles)?;
    let value = serde_json::json!(tiles);
    canon_digest(&value, "eval:wall:hash")
}

/// Logical fingerprint: digest over the sorted tile multiset.
///
/// Two walls with identical multisets in different dealing orders share
/// the fingerprint (near duplicate). Mirrors `wall_fingerprint`
/// (`duplicate.py:94-107`).
pub fn wall_fingerprint(tiles: &[u32]) -> Result<String, SearchError> {
    require_wall_tiles(tiles)?;
    let mut sorted = tiles.to_vec();
    sorted.sort_unstable();
    let value = serde_json::json!(sorted);
    canon_digest(&value, "eval:wall:fingerprint")
}

/// Validate wall shape: exactly 136 ids in `[0,136)`.
fn require_wall_tiles(tiles: &[u32]) -> Result<(), SearchError> {
    if tiles.len() != WALL_TILES {
        return Err(SearchError::InvalidArg { detail: "wall must have 136 tiles" });
    }
    for t in tiles {
        if *t >= WALL_TILES as u32 {
            return Err(SearchError::InvalidArg {
                detail: "tile ids must be int in [0,136)",
            });
        }
    }
    Ok(())
}

/// Nonempty wall-id gate (mirrors `_require_wall_id`, `:67-70`).
fn require_wall_id(wall_id: &str) -> Result<(), SearchError> {
    if wall_id.is_empty() {
        return Err(SearchError::InvalidArg { detail: "wall_id must be nonempty str" });
    }
    Ok(())
}

/// Digest-shape gate (mirrors `_require_digest_value` + `validate_digest`,
/// `:73-76`).
fn require_digest_value(digest: &str) -> Result<(), SearchError> {
    if !is_digest_text(digest) {
        return Err(SearchError::InvalidArg { detail: "digest must be sha256:<64 hex>" });
    }
    Ok(())
}

/// Exact duplicates: two distinct wall ids sharing one digest.
///
/// Returns `(first_id, second_id)` pairs in discovery order. Mirrors
/// `find_exact_duplicates` (`:110-127`).
pub fn find_exact_duplicates(
    wall_hashes: &[(String, String)],
) -> Result<Vec<(String, String)>, SearchError> {
    let mut seen: HashMap<&str, &str> = HashMap::new();
    let mut dups = Vec::new();
    for (wall_id, digest) in wall_hashes {
        require_wall_id(wall_id)?;
        require_digest_value(digest)?;
        if let Some(first) = seen.get(digest.as_str()) {
            dups.push(((*first).to_string(), wall_id.clone()));
        } else {
            seen.insert(digest.as_str(), wall_id.as_str());
        }
    }
    Ok(dups)
}

/// Near duplicates: walls sharing the logical fingerprint.
///
/// Input maps wall id to its 136 tile ids. Mirrors `find_near_duplicates`
/// (`:130-152`).
pub fn find_near_duplicates(
    walls: &[(String, Vec<u32>)],
) -> Result<Vec<(String, String)>, SearchError> {
    let mut first: HashMap<String, String> = HashMap::new();
    let mut dups = Vec::new();
    for (wall_id, tiles) in walls {
        require_wall_id(wall_id)?;
        let fingerprint = wall_fingerprint(tiles)?;
        if let Some(prev) = first.get(&fingerprint) {
            dups.push((prev.clone(), wall_id.clone()));
        } else {
            first.insert(fingerprint, wall_id.clone());
        }
    }
    Ok(dups)
}

/// Result of exact/near duplicate checks over a wall ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DuplicateReport {
    /// Exact-duplicate pairs (byte-identical wall order).
    pub exact_duplicates: Vec<(String, String)>,
    /// Near-duplicate pairs (same sorted multiset).
    pub near_duplicates: Vec<(String, String)>,
}

impl DuplicateReport {
    /// Clean iff no exact AND no near duplicates (`:162-163`).
    pub fn is_clean(&self) -> bool {
        self.exact_duplicates.is_empty() && self.near_duplicates.is_empty()
    }
}

/// Raise when a wall id appears in more than one collection.
///
/// Each collection is one partition (e.g. train vs evaluation). Overlap
/// across any two partitions is forbidden. Mirrors
/// `validate_walls_disjoint` (`:166-184`).
pub fn validate_walls_disjoint(collections: &[&[String]]) -> Result<(), SearchError> {
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for (index, collection) in collections.iter().enumerate() {
        for wall_id in collection.iter() {
            require_wall_id(wall_id)?;
            if seen.contains_key(wall_id.as_str()) {
                return Err(SearchError::InvalidArg {
                    detail: "wall sets must be disjoint",
                });
            }
            seen.insert(wall_id.as_str(), index);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ordered_wall() -> Vec<u32> {
        (0..136).collect()
    }

    /// WALL goldens (`/tmp/eval_goldens.py`): ordered hash, ordered
    /// fingerprint (equal — ordered input is already sorted), reversed
    /// hash (differs), reversed fingerprint (equal — multiset).
    #[test]
    fn wall_hash_and_fingerprint_goldens() {
        let tiles = ordered_wall();
        assert_eq!(
            wall_hash_from_tiles(&tiles).unwrap(),
            "sha256:87ef3e03a99fdd08632d1e74dda2c6287b549293c6d781c0693789f1095ae25c"
        );
        assert_eq!(
            wall_fingerprint(&tiles).unwrap(),
            "sha256:87ef3e03a99fdd08632d1e74dda2c6287b549293c6d781c0693789f1095ae25c"
        );
        let mut rev = tiles.clone();
        rev.reverse();
        assert_eq!(
            wall_hash_from_tiles(&rev).unwrap(),
            "sha256:5fe74e996adfae3f2f3f44525e13c4367dc696fe8f5e4fc9f9f5a8373c157adc"
        );
        assert_eq!(
            wall_fingerprint(&rev).unwrap(),
            "sha256:87ef3e03a99fdd08632d1e74dda2c6287b549293c6d781c0693789f1095ae25c"
        );
    }

    /// sorts(136): any permutation shares the fingerprint but (almost
    /// surely) not the hash. `sort_unstable` == `sorted()` output on
    /// 136 small ints.
    #[test]
    fn sorts_136_parity() {
        let mut perm = ordered_wall();
        let mut i = 0;
        while i < 136 {
            let j = (i * 37 + 11) % 136;
            perm.swap(i, j);
            i += 1;
        }
        assert_eq!(wall_fingerprint(&perm).unwrap(), wall_fingerprint(&ordered_wall()).unwrap());
        assert_ne!(
            wall_hash_from_tiles(&perm).unwrap(),
            wall_hash_from_tiles(&ordered_wall()).unwrap()
        );
    }

    #[test]
    fn wall_shape_rejects() {
        assert!(wall_hash_from_tiles(&vec![0u32; 135]).is_err());
        assert!(wall_hash_from_tiles(&vec![0u32; 137]).is_err());
        let mut bad = ordered_wall();
        bad[0] = 136;
        assert!(wall_hash_from_tiles(&bad).is_err());
        assert!(wall_fingerprint(&bad).is_err());
    }

    #[test]
    fn exact_duplicates_discovery_order() {
        let pairs = vec![
            ("w-a".to_string(), "sha256:".to_string() + &"a".repeat(64)),
            ("w-b".to_string(), "sha256:".to_string() + &"b".repeat(64)),
            ("w-c".to_string(), "sha256:".to_string() + &"a".repeat(64)),
        ];
        assert_eq!(
            find_exact_duplicates(&pairs).unwrap(),
            vec![("w-a".to_string(), "w-c".to_string())]
        );
        assert!(find_exact_duplicates(&[("".to_string(), "sha256:".to_string() + &"a".repeat(64))]).is_err());
        assert!(find_exact_duplicates(&[("w-a".to_string(), "not-a-digest".to_string())]).is_err());
    }

    #[test]
    fn near_duplicates_share_fingerprint() {
        let tiles = ordered_wall();
        let mut rev = tiles.clone();
        rev.reverse();
        let walls = vec![
            ("w-a".to_string(), tiles),
            ("w-b".to_string(), rev),
        ];
        assert_eq!(
            find_near_duplicates(&walls).unwrap(),
            vec![("w-a".to_string(), "w-b".to_string())]
        );
    }

    /// T7-part: overlap across partitions raises (fail-closed); disjoint
    /// passes; empty ids raise.
    #[test]
    fn walls_disjointness() {
        let a = vec!["w-001".to_string(), "w-002".to_string()];
        let b = vec!["w-003".to_string()];
        assert!(validate_walls_disjoint(&[&a, &b]).is_ok());
        let c = vec!["w-002".to_string()];
        assert!(validate_walls_disjoint(&[&a, &c]).is_err());
        let bad = vec!["".to_string()];
        assert!(validate_walls_disjoint(&[&bad]).is_err());
    }
}
