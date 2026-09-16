//! Eval CPU control — shared helpers (EvalControl-owned).
//!
//! B3b: there is NO `hydra-eval` crate. Eval control lives as
//! `hydra-search::eval::{wall,blocks,schedule,telemetry,partition,statistics}`
//! submodules declared here; the `pub mod eval;` line itself is
//! arena-appended to `src/lib.rs` (disjoint ownership — EvalControl never
//! touches `lib.rs` or the manifest).
//!
//! B3 canon-wins: every digest below calls `feed::canon` / `feed::digest`
//! (the single identity site). This module owns NO printer and NO hasher;
//! raw SHA-256 goes through `sha2 0.11`, the same crate feed uses.
//!
//! B2: held-out splits stay the `torch.randperm` oracle. This crate provides
//! NO `philox_split_perm`; `partition` takes caller-supplied (torch-drawn)
//! permutations and only does membership + sorted sides + digest.
//!
//! M2 (measure-first): numpy PCG64 eager stays the bootstrap draw source.
//! Rust owns block-means + percentile-index math + validation + the draw-free
//! hedged-CS path (see `statistics`). No PCG64 reimplementation lives here.
//!
//! M3: `%3` latency draws stay `%3` verbatim (see `schedule`); Lemire
//! `below` is ONLY for new streams, never identity draws.
//!
//! Legacy-CTR replication: `RandomStream` (`hydra2_ctr_v1` sha256-CTR,
//! `randomness.py:358-429`) stays the Python oracle behind KAT. The
//! [`CtrReader`] below replicates its bytes bit-exact so schedule `%3`
//! draws and score-selection seeds agree; it is NOT a new stream.

pub mod blocks;
pub mod partition;
pub mod schedule;
pub mod statistics;
pub mod telemetry;
pub mod wall;

use crate::SearchError;

/// `sha256:<64 lowercase hex>` shape check shared by seal tests.
pub fn is_digest_text(text: &str) -> bool {
    if text.len() != 7 + 64 || !text.starts_with("sha256:") {
        return false;
    }
    text.bytes()
        .skip(7)
        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// Lowercase hex of raw bytes (no `sha256:` prefix).
pub fn hex_of(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(char::from_digit(u32::from(*b >> 4), 16).unwrap_or('0'));
        out.push(char::from_digit(u32::from(*b & 0x0F), 16).unwrap_or('0'));
    }
    out
}

/// SHA-256 over raw bytes (`sha2 0.11`, the feed-identity hasher).
pub fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    use sha2::Digest as _;
    let sum = sha2::Sha256::digest(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(&sum);
    out
}

/// Canon+hash fused over a JSON value (mirrors Python `of_canonical`).
pub fn canon_digest(value: &serde_json::Value, record: &str) -> Result<String, SearchError> {
    hydra_feed::digest::of_canonical(value)
        .map_err(|e| SearchError::Canon { detail: format!("{record}: {e}") })
}

/// Canon bytes for a JSON value through the single `feed::canon` site.
pub fn canon_bytes_of(
    value: &serde_json::Value,
    record: &str,
) -> Result<Vec<u8>, SearchError> {
    hydra_feed::canon::canonical_bytes(value, record)
        .map_err(|e| SearchError::Canon { detail: format!("{record}: {e}") })
}

/// Neumaier compensated sum — matches `math.fsum` bit-exact on the
/// adversarial golden set (`wall.rs`/`blocks.rs` parity tests pin five
/// cancellation vectors). `math.fsum` is correctly rounded; the naive
/// left fold is NOT, so the naive fold is barred here.
pub fn neumaier_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut corr = 0.0;
    for v in values {
        let next = sum + v;
        if sum.abs() >= v.abs() {
            corr += (sum - next) + v;
        } else {
            corr += (v - next) + sum;
        }
        sum = next;
    }
    sum + corr
}

/// Reject non-finite contrasts (mirrors the `isinstance` + `isfinite`
/// preambles; `bool` has no Rust analogue at `f64`).
pub(crate) fn checked_finite(values: &[f64], context: &'static str) -> Result<(), SearchError> {
    for v in values {
        if !v.is_finite() {
            return Err(SearchError::NonFinite { context });
        }
    }
    Ok(())
}

/// Collapse one block to ONE number (wall-block atomicity: games inside a
/// wall NEVER separate downstream).
pub fn block_mean(values: &[f64]) -> Result<f64, SearchError> {
    if values.is_empty() {
        return Err(SearchError::InvalidArg { detail: "empty wall block has no games" });
    }
    checked_finite(values, "block contrast")?;
    Ok(neumaier_sum(values) / values.len() as f64)
}

// ---------------------------------------------------------------------------
// Legacy sha256-CTR stream (`randomness.py:358-429`, oracle stays Python)
// ---------------------------------------------------------------------------

/// Fixed domain tag (`randomness.py:366`): 13 chars + NUL.
pub(crate) const CTR_DOMAIN: [u8; 14] = *b"hydra2_ctr_v1\x00";

/// CTR block width in bytes (`randomness.py:367`).
pub(crate) const CTR_BLOCK: u64 = 32;

/// Block `index` of the legacy stream: `sha256(domain || len(seed)BE32 ||
/// seed || index BE64)`. Byte position `p` depends only on
/// `(seed, p / 32)` — no sequential state, no call-order sensitivity.
pub fn ctr_block(seed: &[u8], index: u64) -> [u8; 32] {
    let mut pre = Vec::with_capacity(14 + 4 + seed.len() + 8);
    pre.extend_from_slice(&CTR_DOMAIN);
    pre.extend_from_slice(&(seed.len() as u32).to_be_bytes());
    pre.extend_from_slice(seed);
    pre.extend_from_slice(&index.to_be_bytes());
    sha256_bytes(&pre)
}
/// Cursor-based reader over the legacy CTR stream (replicates
/// `RandomStream.get_bytes/random_below/random_float`, `:397-410`).
pub struct CtrReader<'a> {
    seed: &'a [u8],
    cursor: u64,
}

impl<'a> CtrReader<'a> {
    /// Open at `cursor`; empty seed fails closed (`:370-371`).
    pub fn new(seed: &'a [u8], cursor: u64) -> Result<Self, SearchError> {
        if seed.is_empty() {
            return Err(SearchError::InvalidArg { detail: "seed must be nonempty bytes" });
        }
        Ok(CtrReader { seed, cursor })
    }

    /// Next `count` stream bytes, advancing the cursor (`:397-406`).
    pub fn get_bytes(&mut self, count: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(count);
        let mut pos = self.cursor;
        while out.len() < count {
            let block = ctr_block(self.seed, pos / CTR_BLOCK);
            let off = (pos % CTR_BLOCK) as usize;
            let take = (count - out.len()).min(CTR_BLOCK as usize - off);
            out.extend_from_slice(&block[off..off + take]);
            pos += take as u64;
        }
        self.cursor = pos;
        out
    }

    /// Uniform `u64` in `[0, bound)` via rejection sampling (`:412-421`).
    /// `bound < 2` is `Err` (M14), never `0`. Lemire-`bounded` is barred
    /// here — this replicates the oracle loop verbatim.
    pub fn random_below(&mut self, bound: u64) -> Result<u64, SearchError> {
        if bound < 2 {
            return Err(SearchError::InvalidArg { detail: "bound must be an int >= 2" });
        }
        let bits = 64 - (bound - 1).leading_zeros();
        let nbytes = bits / 8 + 1;
        let full = 1u128 << (8 * nbytes);
        let limit = full / u128::from(bound) * u128::from(bound);
        loop {
            let raw = self.get_bytes(nbytes as usize);
            let mut value: u128 = 0;
            for b in raw {
                value = (value << 8) | u128::from(b);
            }
            if value < limit {
                return Ok((value % u128::from(bound)) as u64);
            }
        }
    }

    /// Uniform in `[0,1)` from the next 64 stream bits (`:408-410`).
    pub fn random_float(&mut self) -> f64 {
        let raw = self.get_bytes(8);
        let mut n = 0u64;
        for b in raw {
            n = (n << 8) | u64::from(b);
        }
        n as f64 / 18_446_744_073_709_551_616.0
    }
}

/// `semantic_seed` (`randomness.py:328-339`): `sha256` over the canonical
/// `{protocol, master_seed.hex, key}` payload. `key` carries all 17 SPEC-13
/// fields (nulls preserved); JCS sorts, so field order cannot fork.
pub fn semantic_seed(
    master_seed: &[u8],
    key: &serde_json::Value,
) -> Result<[u8; 32], SearchError> {
    if master_seed.is_empty() {
        return Err(SearchError::InvalidArg { detail: "master_seed must be nonempty bytes" });
    }
    let payload = serde_json::json!({
        "protocol": "hydra2_rng_v1",
        "master_seed": hex_of(master_seed),
        "key": key,
    });
    let bytes = canon_bytes_of(&payload, "eval:rng:semantic_seed")?;
    Ok(sha256_bytes(&bytes))
}

/// `score_selection` domain separation (`statistics.py:553`):
/// `sha256("score-selection-v1:{seed}")`.
pub fn score_selection_seed(seed: i64) -> [u8; 32] {
    sha256_bytes(format!("score-selection-v1:{seed}").as_bytes())
}

/// Python `repr`-style list rendering for single-quoted field names
/// (`f"... {missing}"` over `list[str]`). Field names never contain quotes,
/// so the simple form is exact.
pub fn py_list_repr(items: &[&str]) -> String {
    let mut out = String::from("[");
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('\'');
        out.push_str(item);
        out.push('\'');
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// STREAM goldens (`/tmp/eval_goldens4.py`, pixi torch 2.14/numpy 2.5.3):
    /// first two `get_bytes(8)` draws + rejection draws + unit float off the
    /// `score-selection-v1:0` seed. Any CTR reinvention fails here.
    #[test]
    fn legacy_ctr_stream_parity() {
        let seed = score_selection_seed(0);
        assert_eq!(
            hex_of(&seed),
            "54aebf5c2fb762c6673fb0a9385afd9efc77cc5627bb96d4df768022b10466e6"
        );
        let mut r = CtrReader::new(&seed, 0).unwrap();
        assert_eq!(hex_of(&r.get_bytes(8)), "0c775d7b33a5bf12");
        assert_eq!(hex_of(&r.get_bytes(8)), "842a911a640c7b90");
        let mut r = CtrReader::new(&seed, 0).unwrap();
        let draws: Vec<u64> = (0..6).map(|_| r.random_below(10).unwrap()).collect();
        assert_eq!(draws, vec![2, 9, 3, 3, 1, 5]);
        let mut r = CtrReader::new(&seed, 0).unwrap();
        let bits: Vec<u64> = (0..8).map(|_| r.random_below(2).unwrap()).collect();
        assert_eq!(bits, vec![0, 1, 1, 1, 1, 1, 1, 0]);
        let mut r = CtrReader::new(&seed, 0).unwrap();
        assert_eq!(r.random_float(), 0.04869636781554386);
    }

    #[test]
    fn ctr_reader_rejects_empty_seed_and_small_bounds() {
        assert!(CtrReader::new(&[], 0).is_err());
        let seed = score_selection_seed(7);
        let mut r = CtrReader::new(&seed, 0).unwrap();
        assert!(r.random_below(0).is_err());
        assert!(r.random_below(1).is_err());
        assert!(semantic_seed(&[], &serde_json::json!({})).is_err());
    }

    #[test]
    fn digest_shape_check() {
        assert!(is_digest_text("sha256:87ef3e03a99fdd08632d1e74dda2c6287b549293c6d781c0693789f1095ae25c"));
        assert!(!is_digest_text("87ef3e03"));
        assert!(!is_digest_text("sha256:ZZZZ"));
        assert!(!is_digest_text("sha256:87ef3e03"));
    }
}
