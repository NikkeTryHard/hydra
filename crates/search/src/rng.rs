//! `rng`: Philox NEW streams only (never Gumbels, never splits — B1/B2).
//!
//! sha-Gumbels replicate `gumbel_core.py:117-156` bytes VERBATIM in
//! `gumbel.rs` (B1); splits stay the `torch.randperm` oracle behind KAT (B2).
//! This module owns ONLY fresh segment/scenario substreams: each worker
//! segment opens one stream from a hashed 64-bit seed
//! (`rand_philox::Philox::from_u64_seed` splits adjacent int keys through
//! splitmix64, avoiding the verbatim-key correlation trap), and scenario
//! packs (`despot`) derive per-idx independence via counter seeks (block
//! index = scenario idx, NEVER a word index — the 4x-shift trap).
//!
//! ONLY sanctioned `[0,n)` path is `below` (`n >= 2`, Lemire `bounded`);
//! `% n` and `int(u*f)` are barred.

use rand_philox::Philox;

use crate::SearchError;

/// One worker's stream: position is implicit in the Philox counter
/// (counter = BLOCK index; seeks land on block starts only).
#[derive(Debug, Clone)]
pub struct SegmentStream {
    /// Underlying counter stream.
    stream: Philox,
}

impl SegmentStream {
    /// Open a NEW stream for one segment: `from_u64_seed` (splitmix64 key
    /// derivation; adjacent int seeds do NOT correlate).
    pub fn open(seed: u64) -> SegmentStream {
        SegmentStream {
            stream: Philox::from_u64_seed(seed),
        }
    }

    /// Raw 32-bit word.
    pub fn next_u32(&mut self) -> u32 {
        self.stream.next_u32()
    }

    /// Raw 64-bit word: `lo | (hi << 32)` word order (canon §7.1).
    pub fn next_u64(&mut self) -> u64 {
        self.stream.next_u64()
    }

    /// Uniform `f64` in `[0,1)`: `(next_u64 as f64) / 2^64` exact
    /// (no `%`, no `int(u*f)`; the `+0.5` map lives ONLY in `gumbel.rs`).
    pub fn next_f64(&mut self) -> f64 {
        // proof: u64 word / 2^64 is in [0,1); 1ulp on the unit interval is oracle-tolerable.
        #[allow(clippy::cast_precision_loss)]
        let f: f64 = self.next_u64() as f64;
        f / 18_446_744_073_709_551_616.0
    }

    /// ONLY sanctioned `[0,n)` draw: Lemire `bounded`, `n >= 2`.
    pub fn below(&mut self, n: u32) -> Result<u32, SearchError> {
        if n < 2 {
            return Err(SearchError::InvalidArg {
                detail: "below() bound must be >= 2",
            });
        }
        Ok(self.stream.bounded(n))
    }

    /// Reposition at a block start (block-starts-only; `pos` words consumed
    /// after the seek is the caller's snapshot business).
    pub fn seek_block(&mut self, block: u128) {
        self.stream.seek(block);
    }
}

/// Open per-idx independent scenario streams for a DESPOT pack: each idx
/// seeks the shared-key stream to a distinct block (`block = base + idx`),
/// so draws are reproducible under thread splits AND KAT-pinned.
///
/// `base_seed` is the pack seed; `count` is `K` (16 in prod).
pub fn scenario_streams(base_seed: u64, count: u32) -> Vec<SegmentStream> {
    let mut out = Vec::with_capacity(count as usize);
    let mut idx = 0;
    while idx < count {
        let mut stream = SegmentStream::open(base_seed);
        stream.seek_block(idx as u128);
        out.push(stream);
        idx += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_rejects_small_bounds() {
        let mut stream = SegmentStream::open(7);
        assert!(stream.below(0).is_err());
        assert!(stream.below(1).is_err());
        assert!(stream.below(2).is_ok());
    }

    #[test]
    fn same_seed_same_sequence() {
        let mut left = SegmentStream::open(42);
        let mut right = SegmentStream::open(42);
        let mut i = 0;
        while i < 16 {
            assert_eq!(left.next_u32(), right.next_u32());
            i += 1;
        }
    }

    #[test]
    fn adjacent_seeds_do_not_alias() {
        // `from_u64_seed` splitmix derivation: adjacent int seeds diverge
        // on the first word (the verbatim-key correlation trap is avoided).
        let mut left = SegmentStream::open(100);
        let mut right = SegmentStream::open(101);
        assert_ne!(left.next_u32(), right.next_u32());
    }

    #[test]
    fn scenario_streams_are_per_idx_independent() {
        let streams = scenario_streams(9, 16);
        assert_eq!(streams.len(), 16);
        // Distinct blocks -> distinct first words across the pack.
        let mut firsts: Vec<u32> = streams
            .into_iter()
            .map(|mut stream| stream.next_u32())
            .collect();
        firsts.sort_unstable();
        firsts.dedup();
        assert!(firsts.len() > 1);
    }

    #[test]
    fn next_f64_stays_unit() {
        let mut stream = SegmentStream::open(5);
        let mut i = 0;
        while i < 64 {
            let v = stream.next_f64();
            assert!((0.0..1.0).contains(&v), "f64 out of [0,1): {v}");
            i += 1;
        }
    }

    #[test]
    fn seek_block_is_reproducible() {
        let mut left = SegmentStream::open(3);
        left.seek_block(11);
        let first = left.next_u32();
        let mut right = SegmentStream::open(3);
        right.seek_block(11);
        assert_eq!(right.next_u32(), first);
    }
}
