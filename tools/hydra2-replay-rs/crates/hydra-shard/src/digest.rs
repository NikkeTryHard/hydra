//! SHA-256 parity for the cold shard (P4-B, cold only — NEVER hot).
//!
//! Feed FORBIDS hashing on the hot path (see `hydra-feed` manifest notes); all
//! digests here are minted cold: the shard-file payload digest bound into the
//! file header (writer mints, reader verifies) and per-row parity hashes for
//! the G2 cold-vs-hot comparison. Digest text shape (`sha256:` + lowercase
//! hex) matches the [`crate::parity`] authorities.

use sha2::{Digest, Sha256};

/// Digest of `bytes` as `sha256:<hex>` (matches `DigestText` style).
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let sum = h.finalize();
    raw_hex_prefixed(sum.as_slice())
}

/// Render pre-computed digest bytes as `sha256:<hex>` (no re-hash).
pub fn raw_hex_prefixed(raw: &[u8]) -> String {
    let mut out = String::with_capacity(7 + 2 * raw.len());
    out.push_str("sha256:");
    let mut i = 0usize;
    while i < raw.len() {
        out.push(hex_of(raw[i] >> 4));
        out.push(hex_of(raw[i] & 0xF));
        i += 1;
    }
    out
}

#[inline]
fn hex_of(nibble: u8) -> char {
    (b'0' + nibble + if nibble >= 10 { b'a' - b'0' - 10 } else { 0 }) as char
}

/// Incremental payload hasher for the shard writer (header-placeholder
/// pattern: rows stream through this, the digest patches the header at
/// `finish`).
#[derive(Debug, Clone)]
pub struct PayloadHasher {
    inner: Sha256,
    len: u64,
}

impl PayloadHasher {
    /// Empty hasher.
    pub fn new() -> Self {
        Self {
            inner: Sha256::new(),
            len: 0,
        }
    }

    /// Feed one game-atomic byte run (already a multiple of the row stride).
    pub fn update(&mut self, bytes: &[u8]) {
        self.inner.update(bytes);
        self.len += bytes.len() as u64;
    }

    /// Streamed byte count.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// True when nothing has been streamed.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Finalize into the raw 32-byte digest (stored in the file header).
    pub fn finalize_raw(self) -> [u8; 32] {
        let sum = self.inner.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&sum);
        out
    }
}

impl Default for PayloadHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod digest_tests {
    use super::*;

    #[test]
    fn empty_digest_matches_sha256() {
        assert_eq!(
            sha256_hex(b""),
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn incremental_matches_oneshot() {
        let mut h = PayloadHasher::new();
        assert!(h.is_empty());
        h.update(b"row0-bytes");
        h.update(b"row1-bytes");
        assert_eq!(h.len(), 20);
        let raw = h.finalize_raw();
        assert_eq!(sha256_hex(b"row0-bytesrow1-bytes"), raw_hex_prefixed(&raw));
    }
}
