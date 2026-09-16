//! `keys::Categories` interns digests to `u32` (never hashes; search owns
//! tables + selection, canon owns bytes — B3).
//!
//! All hashing happens caller-side via `feed::canon`/`feed::digest` (shared
//! crate, single canon site). This table maps already-hashed digest bytes to
//! compact ids for arena nodes and strategy tables. It NEVER hashes itself:
//! the forward map is a `BTreeMap` (byte-wise order, no hasher at all;
//! PYTHONHASHSEED trap avoided — the map key IS the digest bytes).

use std::collections::BTreeMap;

use crate::SearchError;

/// Interned info-key table: digest bytes <-> `u32` id.
///
/// Ids are dense (`0..len`) in first-intern order. Deterministic across runs
/// given the same insertion order (insertion order in the arena is the
/// deterministic `0..T` segment order plus sorted-legal descent, never a
/// hashmap iteration).
#[derive(Debug, Default)]
pub struct Categories {
    /// Digest bytes -> id (byte-ordered, never hashed).
    forward: BTreeMap<Vec<u8>, u32>,
    /// Id -> digest bytes (index = id).
    reverse: Vec<Vec<u8>>,
}

impl Categories {
    /// Empty table.
    pub fn new() -> Categories {
        Categories { forward: BTreeMap::new(), reverse: Vec::new() }
    }

    /// Intern digest bytes, returning the stable id.
    ///
    /// Same bytes -> same id (never re-hashes, never re-assigns). Empty
    /// input is rejected (a digest is never empty).
    pub fn intern(&mut self, digest: &[u8]) -> Result<u32, SearchError> {
        if digest.is_empty() {
            return Err(SearchError::InvalidArg { detail: "digest must be non-empty" });
        }
        if let Some(found) = self.forward.get(digest) {
            return Ok(*found);
        }
        let id = self.reverse.len() as u32;
        self.forward.insert(digest.to_vec(), id);
        self.reverse.push(digest.to_vec());
        Ok(id)
    }

    /// Look up the id for digest bytes, if interned.
    pub fn lookup(&self, digest: &[u8]) -> Option<u32> {
        self.forward.get(digest).copied()
    }

    /// Resolve an id back to its digest bytes, if assigned.
    pub fn resolve(&self, id: u32) -> Option<&[u8]> {
        self.reverse.get(id as usize).map(Vec::as_slice)
    }

    /// Number of interned digests.
    pub fn len(&self) -> usize {
        self.reverse.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.reverse.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_is_stable_and_dedupes() {
        let mut cats = Categories::new();
        let a = cats.intern(b"sha256:aaaa").expect("intern");
        let b = cats.intern(b"sha256:bbbb").expect("intern");
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        // Same bytes -> same id, no growth.
        assert_eq!(cats.intern(b"sha256:aaaa").expect("re-intern"), 0);
        assert_eq!(cats.len(), 2);
    }

    #[test]
    fn lookup_resolve_round_trip() {
        let mut cats = Categories::new();
        let id = cats.intern(b"sha256:0123").expect("intern");
        assert_eq!(cats.lookup(b"sha256:0123"), Some(id));
        assert_eq!(cats.lookup(b"sha256:nope"), None);
        assert_eq!(cats.resolve(id), Some(b"sha256:0123".as_slice()));
        assert_eq!(cats.resolve(99), None);
    }

    #[test]
    fn empty_digest_rejected() {
        let mut cats = Categories::new();
        assert_eq!(
            cats.intern(b""),
            Err(SearchError::InvalidArg { detail: "digest must be non-empty" })
        );
    }

    #[test]
    fn insertion_order_pins_ids() {
        // Golden: same insertion order in two tables -> identical ids.
        let digests = [b"sha256:0001".as_slice(), b"sha256:0002".as_slice(), b"sha256:0003".as_slice()];
        let mut first = Categories::new();
        let mut second = Categories::new();
        for d in digests {
            assert_eq!(
                first.intern(d).expect("intern"),
                second.intern(d).expect("intern")
            );
        }
    }

    #[test]
    fn never_hashes_distinct_bytes_stay_distinct() {
        // Near-identical digests that would collide under a weak hash stay
        // distinct: the table keys on the bytes themselves.
        let mut cats = Categories::new();
        let a = cats.intern(b"sha256:abc0").expect("intern");
        let b = cats.intern(b"sha256:abc1").expect("intern");
        assert_ne!(a, b);
    }
}
