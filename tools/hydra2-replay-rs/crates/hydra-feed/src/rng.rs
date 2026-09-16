//! Counter-stream RNG: Philox4x32-10 block words + Lemire draws (KAT-pinned).
//!
//! Mirrors `src/hydra2/contracts/randomness.py` discipline without copying its
//! bytes: the legacy `RandomStream` (`hydra2_ctr_v1` sha256-CTR) stays the Python
//! oracle behind KAT; NEW shuffle/sample/split streams use Philox + Lemire here.
//! sha-Gumbels NEVER become Philox-Gumbels (B1); `torch.randperm` splits stay the
//! oracle behind KAT (B2).
//!
//! Conventions (wave2 §7.1-7.3, Wave5 M6/M10/M14):
//! - `StreamKey.domain` is `[u8;14]` (M10); the canonical tag is
//!   `b"hydra2_ctr_v1\x00"` (13 chars + NUL, `randomness.py:366`).
//! - `StreamKey.len` is the SIGNIFICANT domain-tag length (`1..=14`); the tag is
//!   zero-padded to `[u8;14]` and [`philox_block`] mixes only `domain[..len]`
//!   (padding beyond `len` is provably ignored — KAT-pinned).
//! - The counter is a BLOCK index, never a word index (4x-shift trap); seeks
//!   position at block starts only.
//! - ONLY sanctioned `[0,n)` path is [`below`] → `Philox::bounded` (exact Lemire
//!   with rejection in Rust); `% n` and `int(u*f)` are barred. [`bounded`] is the
//!   single-word Lemire hi-bits primitive for KAT pinning, not a draw path.
//! - [`below`] mirrors `RandomStream.random_below` (`randomness.py:412-421`,
//!   `bound >= 2`): `below(0)`/`below(1)` are `Err` (M14), never `0`.
//! - Snapshot discipline: streams persist as `(entries, {key, block, pos})` —
//!   [`StreamSnapshot`] — and restore as `seek(block)` + consume `pos` words;
//!   callers record `(key, base, n-sequence)` so resharding replays exactly.
//!
//! Hash-free by construction: this module imports `rand_philox` only — no `sha2`,
//! no `serde_jcs` (those live in `feed::canon`/`feed::digest`).

use rand_philox::{Philox, philox4x32_10};

/// Canonical legacy domain tag: 13 chars + NUL (`randomness.py:366`).
pub const LEGACY_DOMAIN_TAG: [u8; 14] = *b"hydra2_ctr_v1\x00";

/// Compile-time domain-width pin (M10): fails the build if the tag is not 14 B.
const _: [u8; 14] = LEGACY_DOMAIN_TAG;

// ---------------------------------------------------------------------------
// Errors (fail-closed, numeric, never a `String`)
// ---------------------------------------------------------------------------

/// RNG failure taxonomy: mirrors `ContractError` edges, never a panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngError {
    /// [`below`] called with `bound < 2` (mirrors `random_below` M14).
    BoundTooSmall {
        /// Rejected bound (`0` or `1`).
        bound: u32,
    },
    /// [`restore_snapshot`] called with `pos >= 4` (a block holds 4 words).
    InvalidPosition {
        /// Rejected position-in-block.
        pos: u8,
    },
}

impl core::fmt::Display for RngError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            RngError::BoundTooSmall { bound } => {
                write!(f, "below() bound must be >= 2, got {bound}")
            }
            RngError::InvalidPosition { pos } => {
                write!(f, "snapshot pos must be < 4, got {pos}")
            }
        }
    }
}

impl std::error::Error for RngError {}

// ---------------------------------------------------------------------------
// Stream key (M10 width, M6 LE lanes)
// ---------------------------------------------------------------------------

/// Counter-stream key: 14 B domain tag + significant tag length + 32 B seed.
///
/// - `domain`: stream tag, zero-padded to 14 B; only `domain[..len]` is
///   significant (see [`philox_block`]).
/// - `len`: significant domain-tag length (`1..=14`); `len` bounds stream
///   identity, never perturbs the word sequence beyond tag selection.
/// - `seed`: 32 B stream seed; LE lanes per M6 are
///   `key0 = seed[0:8]`, `key1 = seed[8:16]`, `ctr0 = seed[16:24]`,
///   `spare = seed[24:32]` (see [`seed_parts`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamKey {
    /// Zero-padded domain tag; significant prefix length is [`StreamKey::len`].
    pub domain: [u8; 14],
    /// Significant domain-tag length (`1..=14`).
    pub len: u32,
    /// 32 B stream seed (M6 LE lanes).
    pub seed: [u8; 32],
}

impl StreamKey {
    /// Build a key; `len` is the significant tag length (`1..=14`).
    pub fn new(domain: [u8; 14], len: u32, seed: [u8; 32]) -> StreamKey {
        StreamKey { domain, len, seed }
    }
}

/// M6 LE lanes of a 32 B seed: `(key0, key1, ctr0, spare)`.
///
/// `key0 = seed[0:8]`, `key1 = seed[8:16]`, `ctr0 = seed[16:24]`,
/// `spare = seed[24:32]`. The Philox mapping consumes `seed[0:8]` as the
/// `[u32;2]` key verbatim, `u128::from_le_bytes(seed[8:24])` as the counter
/// base, and folds `spare` into the counter high lane (spare policy: perturbs,
/// never ignored; fixtures pin `spare = 0`).
pub fn seed_parts(seed: &[u8; 32]) -> (u64, u64, u64, u64) {
    let lane = |o: usize| {
        u64::from_le_bytes([
            seed[o],
            seed[o + 1],
            seed[o + 2],
            seed[o + 3],
            seed[o + 4],
            seed[o + 5],
            seed[o + 6],
            seed[o + 7],
        ])
    };
    (lane(0), lane(8), lane(16), lane(24))
}

/// Recorded seed fixtures (M6): 3+ named 32 B seeds with `spare = 0`.
///
/// Vectors only — the `(key0, key1, ctr0)` LE derivation is pinned by
/// [`seed_parts`] tests, and stream replay by the KAT-4 tests below.
pub fn seed_map_fixture() -> Vec<(String, [u8; 32])> {
    // First 24 B distinct per fixture; spare lane [24..32] pinned to zero
    // so `seed_parts(..).3 == 0` holds for every vector.
    let mut seq = [0u8; 32];
    let mut i = 0;
    while i < 24 {
        seq[i] = i as u8;
        i += 1;
    }
    let mut pat5a = [0u8; 32];
    let mut j = 0;
    while j < 24 {
        pat5a[j] = 0x5A;
        j += 1;
    }
    Vec::from([
        ("hydra2_rng_fixture_zero".to_string(), [0u8; 32]),
        ("hydra2_rng_fixture_seq".to_string(), seq),
        ("hydra2_rng_fixture_5a".to_string(), pat5a),
    ])
}

// ---------------------------------------------------------------------------
// Philox mapping (KAT-pinned derivation, block-starts-only)
// ---------------------------------------------------------------------------

/// Philox `[u32;2]` key for a stream: `seed[0:8]` verbatim, folded with the
/// significant domain bytes `domain[..len]` (padding beyond `len` ignored).
fn philox_key(key: &StreamKey) -> [u32; 2] {
    let s = &key.seed;
    let mut k0 = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
    let mut k1 = u32::from_le_bytes([s[4], s[5], s[6], s[7]]);
    let take = key.len.min(14) as usize;
    let mut i = 0;
    while i < take {
        let b = key.domain[i] as u32;
        if (i / 4) % 2 == 0 {
            k0 ^= b << ((i % 4) as u32 * 8);
        } else {
            k1 ^= b << ((i % 4) as u32 * 8);
        }
        i += 1;
    }
    [k0, k1]
}

/// Absolute Philox block counter: `u128 LE seed[8:24]`, plus `block`, plus the
/// `spare` lane folded into the high 64 (M6 spare policy).
fn block_counter(key: &StreamKey, block: u64) -> u128 {
    let s = &key.seed;
    let base = u128::from_le_bytes([
        s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15], s[16], s[17], s[18],
        s[19], s[20], s[21], s[22], s[23],
    ]);
    let spare = u64::from_le_bytes([
        s[24], s[25], s[26], s[27], s[28], s[29], s[30], s[31],
    ]) as u128;
    base
        .wrapping_add(block as u128)
        .wrapping_add(spare << 64)
}

/// Block words as a pure function of `(key, block)` (KAT-pinned).
///
/// `word j = philox4x32_10(ctr_le(base + block), key)[j]`; position-in-block
/// selects `block_words[i % 4]`. `block` is a BLOCK index — never a word index.
pub fn philox_block(key: &StreamKey, block: u64) -> [u32; 4] {
    let ctr = block_counter(key, block).to_le_bytes();
    let words = |o: usize| u32::from_le_bytes([ctr[o], ctr[o + 1], ctr[o + 2], ctr[o + 3]]);
    philox4x32_10([words(0), words(4), words(8), words(12)], philox_key(key))
}

/// Open a stateful stream positioned at the start of `block`.
pub fn open_block_stream(key: &StreamKey, block: u64) -> Philox {
    Philox::new(philox_key(key), block_counter(key, block))
}

/// Reposition a stream at the start of `block` (block-starts-only).
///
/// The stream MUST have been opened from the same `key`: `seek` sets the
/// absolute counter `base(key) + block`, so seeking a foreign stream silently
/// re-keys it — callers keep `(key, block)` together (see [`StreamSnapshot`]).
pub fn seek_to_block(stream: &mut Philox, key: &StreamKey, block: u64) {
    stream.seek(block_counter(key, block));
}

// ---------------------------------------------------------------------------
// Snapshot discipline (entries, {key, block, pos})
// ---------------------------------------------------------------------------

/// Opaque checkpoint: everything needed for exact continuation.
///
/// `pos` counts words already consumed in `block` (`0..4`); a snapshot with
/// `pos == 4` MUST be normalized to `(block + 1, 0)` by the caller before
/// storing. Snapshots are caller-maintained triples — the Philox counter is
/// private, so capture records `(key, block, pos)` at draw boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamSnapshot {
    /// Stream identity.
    pub key: StreamKey,
    /// Block index of the next undrawn word's block.
    pub block: u64,
    /// Words already consumed in `block` (`0..4`).
    pub pos: u8,
}

/// Restore a snapshot: `seek(block)` then consume `pos` words.
pub fn restore_snapshot(snap: &StreamSnapshot) -> Result<Philox, RngError> {
    if snap.pos >= 4 {
        return Err(RngError::InvalidPosition { pos: snap.pos });
    }
    let mut stream = open_block_stream(&snap.key, snap.block);
    let mut i = 0;
    while i < snap.pos {
        let _ = stream.next_u32();
        i += 1;
    }
    Ok(stream)
}

// ---------------------------------------------------------------------------
// Draws: Lemire ONLY
// ---------------------------------------------------------------------------

/// Single-word Lemire hi-bits primitive: `((word * n) >> 32)`.
///
/// KAT-pinning primitive — NEVER `% n`, NEVER float. This is NOT the sanctioned
/// draw path: one word cannot serve the Lemire rejection loop (rare slow path
/// needs extra words). Sanctioned `[0,n)` draws go through [`below`].
/// `bounded(_, 0)` is defined as `0` (empty product) but meaningless — callers
/// MUST use [`below`], which rejects `n < 2`.
pub fn bounded(word: u32, n: u32) -> u32 {
    (((word as u64) * (n as u64)) >> 32) as u32
}

/// ONLY sanctioned `[0,n)` path: thin wrapper over `Philox::bounded`
/// (exact Lemire, byte-identical to CommonStats, wave2 §7.2).
///
/// Mirrors `random_below` (`bound >= 2`): `n < 2` is `Err` (M14), never `0`.
pub fn below(rng: &mut Philox, n: u32) -> Result<u32, RngError> {
    if n < 2 {
        return Err(RngError::BoundTooSmall { bound: n });
    }
    Ok(rng.bounded(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> StreamKey {
        StreamKey::new(LEGACY_DOMAIN_TAG, 14, [0x11u8; 32])
    }

    #[test]
    fn kat4_legacy_tag_bytes() {
        assert_eq!(LEGACY_DOMAIN_TAG, *b"hydra2_ctr_v1\x00");
        assert_eq!(LEGACY_DOMAIN_TAG.len(), 14);
    }

    #[test]
    fn kat4_seed_parts_le_offsets() {
        let mut seed = [0u8; 32];
        seed[0] = 0x01;
        seed[8] = 0x02;
        seed[16] = 0x03;
        seed[24] = 0x04;
        let (k0, k1, c0, spare) = seed_parts(&seed);
        assert_eq!((k0, k1, c0, spare), (0x01, 0x02, 0x03, 0x04));
    }

    #[test]
    fn kat4_seed_map_fixtures() {
        let fixtures = seed_map_fixture();
        assert!(fixtures.len() >= 3);
        // Distinct seeds, spare lane pinned to zero.
        assert_ne!(fixtures[0].1, fixtures[1].1);
        assert_ne!(fixtures[1].1, fixtures[2].1);
        for (_, seed) in &fixtures {
            assert_eq!(seed_parts(seed).3, 0);
        }
        // Distinct seeds open distinct streams.
        let words: Vec<[u32; 4]> = fixtures
            .iter()
            .map(|(_, s)| philox_block(&StreamKey::new(LEGACY_DOMAIN_TAG, 14, *s), 0))
            .collect();
        assert_ne!(words[0], words[1]);
        assert_ne!(words[1], words[2]);
    }

    #[test]
    fn kat4_block_replay_and_pure_vs_stream() {
        let key = test_key();
        let a = philox_block(&key, 7);
        let b = philox_block(&key, 7);
        assert_eq!(a, b);
        // Pure block == stateful drain of the same (key, block).
        let mut stream = open_block_stream(&key, 7);
        let drained = [
            stream.next_u32(),
            stream.next_u32(),
            stream.next_u32(),
            stream.next_u32(),
        ];
        assert_eq!(a, drained);
        // Adjacent blocks differ; block index is not a word index.
        assert_ne!(a, philox_block(&key, 8));
    }

    #[test]
    fn kat4_position_in_block_replay() {
        let key = test_key();
        // word j of block == pure block [j]: (key, block, pos) replay.
        let mut stream = open_block_stream(&key, 3);
        let words = philox_block(&key, 3);
        for pos in 0..4 {
            assert_eq!(stream.next_u32(), words[pos]);
        }
    }

    #[test]
    fn kat4_next_u64_word_order() {
        let key = test_key();
        let mut r1 = open_block_stream(&key, 0);
        let lo = r1.next_u32();
        let hi = r1.next_u32();
        let mut r2 = open_block_stream(&key, 0);
        assert_eq!(r2.next_u64(), (lo as u64) | ((hi as u64) << 32));
    }

    #[test]
    fn kat4_domain_separation_and_padding_ignored() {
        let seed = [0x22u8; 32];
        let full = StreamKey::new(LEGACY_DOMAIN_TAG, 14, seed);
        let mut other_tag = LEGACY_DOMAIN_TAG;
        other_tag[0] ^= 0xFF;
        let other = StreamKey::new(other_tag, 14, seed);
        assert_ne!(philox_block(&full, 0), philox_block(&other, 0));
        // Short tag: bytes beyond `len` are provably ignored.
        let mut short = [0u8; 14];
        short[0] = b's';
        short[1] = b's';
        let mut padded = short;
        padded[5] = 0xAB;
        padded[13] = 0xCD;
        let a = StreamKey::new(short, 2, seed);
        let b = StreamKey::new(padded, 2, seed);
        assert_eq!(philox_block(&a, 0), philox_block(&b, 0));
    }

    #[test]
    fn kat4_seek_block_starts_replay() {
        let key = test_key();
        let mut stream = open_block_stream(&key, 0);
        let first = [stream.next_u32(), stream.next_u32()];
        seek_to_block(&mut stream, &key, 5);
        let at_five = [stream.next_u32(), stream.next_u32()];
        assert_eq!(at_five, philox_block(&key, 5)[..2]);
        seek_to_block(&mut stream, &key, 0);
        assert_eq!([stream.next_u32(), stream.next_u32()], first);
    }

    #[test]
    fn kat4_snapshot_restore_roundtrip() {
        let key = test_key();
        let snap = StreamSnapshot { key, block: 4, pos: 2 };
        let mut restored = restore_snapshot(&snap).unwrap();
        let words = philox_block(&key, 4);
        assert_eq!(restored.next_u32(), words[2]);
        assert_eq!(restored.next_u32(), words[3]);
        assert!(restore_snapshot(&StreamSnapshot { key, block: 0, pos: 4 }).is_err());
    }

    #[test]
    fn kat4_bounded_lemire_hi_bits() {
        // Single-word Lemire candidate, never `% n`.
        assert_eq!(bounded(0xFFFF_FFFF, 2), 1);
        assert_eq!(bounded(0x8000_0000, 2), 1);
        assert_eq!(bounded(0x7FFF_FFFF, 2), 0);
        assert_eq!(bounded(0x1234_5678, 1), 0);
        assert_eq!(bounded(0xFFFF_FFFF, 1), 0);
    }

    #[test]
    fn kat4_below_edge_m14() {
        let key = test_key();
        let mut rng = open_block_stream(&key, 0);
        assert_eq!(below(&mut rng, 0), Err(RngError::BoundTooSmall { bound: 0 }));
        assert_eq!(below(&mut rng, 1), Err(RngError::BoundTooSmall { bound: 1 }));
        // `Philox::bounded(1) == 0` is defined upstream (wave2 §7.1); our
        // guarded `below` deliberately extends it to `Err` per M14.
        assert_eq!(rng.bounded(1), 0);
    }

    #[test]
    fn kat4_below_range_and_determinism() {
        let key = test_key();
        let mut r1 = open_block_stream(&key, 9);
        let mut r2 = open_block_stream(&key, 9);
        for n in [2u32, 3, 5, 136] {
            for _ in 0..500 {
                let a = below(&mut r1, n).unwrap();
                let b = below(&mut r2, n).unwrap();
                assert_eq!(a, b);
                assert!(a < n);
            }
        }
    }
}
