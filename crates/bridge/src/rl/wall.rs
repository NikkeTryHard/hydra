//! rl/wall: pure-Rust continuation-wall law (NumPy 2.5.3 byte discipline).
//!
//! Thin exact port of the pinned oracle path in
//! `python/hydra2/engines/riichienv/walls.py::derive_hand_wall` (NumPy fast
//! path): named continuation seed (SHA-256 first 8 bytes, big-endian) ->
//! SeedSequence (pool 4) -> PCG64-XSL-RR-128/64 -> masked inclusive interval
//! -> descending Fisher-Yates over 136 tiles. Hand 0 is rejected here: the
//! supplied initial schedule tiles are authoritative and never generated.
//! The engine seed uses the separate `rng:{ordinal}` key suffix.
//!
//! Source pins (never substitute):
//! - SeedSequence mixing + `generate_state` + masked `random_interval`:
//!   NumPy v2.5.3 (`numpy/random/bit_generator.pyx`,
//!   `numpy/random/src/distributions/distributions.c`), BSD-3-Clause,
//!   Copyright (c) 2005-2024 NumPy Developers. Constants `INIT_A/MULT_A`,
//!   `INIT_B/MULT_B`, `MIX_MULT_L/MIX_MULT_R`, `XSHIFT = 32/2` reproduced
//!   verbatim; LE limb order (`zero -> [0]`), in-place off-diagonal pool
//!   mixing, trailing-entropy tail, and LE u32-pairing into u64 kept.
//! - PCG seeding/step/XSL-RR output + low-half-first u32 cache: NumPy's
//!   vendored PCG (`numpy/random/src/pcg64/pcg64.h`, `pcg64.c`), Apache-2.0 /
//!   MIT (Melissa O'Neill / NumPy Developers). Multiplier high
//!   2549297995355413924 / low 4865540595714422341; `inc = (seq << 1) | 1`;
//!   `state = 0; step; state += initstate; step`; `next64` steps FIRST then
//!   `rotr64(high ^ low, high >> 58)`.
//!
//! DAG: this module depends on `sha2` ONLY for the continuation-key digest
//! (same pinned `sha2 = "=0.11"` as the bridge root). No `rand/getrandom`,
//! no Philox (walls are NumPy-law, never the semantic counter stream), no
//! Lemire, no SHA-fallback keystream, no StdRng. The semantic CTR stream
//! (`hydra_feed::digest::ctr_block`) is reused ONLY for the batched u53
//! reserve helper below, never for wall bytes.
//!
//! FORBIDS: hand-0 generation, per-row Python callbacks (batch FFI lives in
//! `rl/ffi.rs`), float-clamped/repaired sampling (u53 endpoints exact).

use sha2::{Digest, Sha256};
use std::fmt;

/// Named semantic stream for continuation walls AND engine seeds
/// (mirrors `walls.py::WALL_STREAM_NAME`; part of snapshot identity).
pub const WALL_STREAM_NAME: &str = "hydra2.wall_continuation_v1";

/// Tiles per wall: a permutation of physical ids `0..136`.
pub const WALL_TILES: usize = 136;

/// SeedSequence pool width (NumPy `DEFAULT_POOL_SIZE`).
pub const SEED_POOL: usize = 4;

/// Words consumed per u53 reserve: exactly 8 bytes per real decision.
pub const U53_BYTES: usize = 8;

/// Exact `2^-53` scale for the u53 endpoint law.
pub const U53_SCALE: f64 = 1.0 / 9007199254740992.0;

/// CTR block width of the reused semantic counter stream.
const CTR_BLOCK: u64 = 32;

// NumPy SeedSequence source constants (`bit_generator.pyx:56-63`).
const INIT_A: u32 = 0x43b0d7e5;
const MULT_A: u32 = 0x931e8875;
const INIT_B: u32 = 0x8b51f9dd;
const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd;
const MIX_MULT_R: u32 = 0x4973f715;
const XSHIFT: u32 = 16;

// PCG default 128-bit multiplier (`pcg64.h:97-101`).
const PCG_MULT: u128 = ((2549297995355413924u128) << 64) | 4865540595714422341u128;

/// Fail-closed wall errors (no defaults, no silent reseeding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WallError {
    /// Hand 0 must consume the supplied schedule tiles, never a derivation.
    HandZero,
    /// The u53 reserve needs a nonempty semantic seed.
    EmptySeed,
    /// `cursor + draws * 8` overflows the byte window.
    WindowOverflow,
    /// The CTR block counter overflows `u64` while covering the window.
    BlockOverflow,
}

impl fmt::Display for WallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WallError::HandZero => {
                write!(f, "hand 0 uses the supplied schedule tiles (never derived)")
            }
            WallError::EmptySeed => write!(f, "u53 seed must be nonempty bytes"),
            WallError::WindowOverflow => write!(f, "u53 byte window overflows"),
            WallError::BlockOverflow => write!(f, "u53 block counter overflows u64"),
        }
    }
}

impl std::error::Error for WallError {}

/// Continuation key digest: `SHA256(UTF8(stream|schedule|digest|hand))`,
/// first 8 bytes as big-endian `u64` (mirrors `walls.py:89-90`).
pub fn continuation_seed(schedule_id: &str, digest_noprefix: &str, hand: u64) -> u64 {
    let mut key = String::with_capacity(
        WALL_STREAM_NAME.len() + schedule_id.len() + digest_noprefix.len() + 24,
    );
    key.push_str(WALL_STREAM_NAME);
    key.push('|');
    key.push_str(schedule_id);
    key.push('|');
    key.push_str(digest_noprefix);
    key.push('|');
    key.push_str(&hand.to_string());
    let sum = Sha256::digest(key.as_bytes());
    u64::from_be_bytes([
        sum[0], sum[1], sum[2], sum[3], sum[4], sum[5], sum[6], sum[7],
    ])
}

/// Engine seed for an ordinal: same stream family, separate `rng:{hand}`
/// last component (mirrors `adapter_step.py::_hand_rng_seed`). Infallible
/// for every ordinal including 0; never a wall permutation input.
pub fn hand_rng_seed(schedule_id: &str, digest_noprefix: &str, hand: u64) -> u64 {
    let mut key = String::with_capacity(
        WALL_STREAM_NAME.len() + schedule_id.len() + digest_noprefix.len() + 28,
    );
    key.push_str(WALL_STREAM_NAME);
    key.push('|');
    key.push_str(schedule_id);
    key.push('|');
    key.push_str(digest_noprefix);
    key.push_str("|rng:");
    key.push_str(&hand.to_string());
    let sum = Sha256::digest(key.as_bytes());
    u64::from_be_bytes([
        sum[0], sum[1], sum[2], sum[3], sum[4], sum[5], sum[6], sum[7],
    ])
}

/// Seed `u64` to least-significant-first `u32` limbs (`zero -> [0]`,
/// mirrors `_int_to_uint32_array`). Returns the limbs plus the used length
/// (1 limb when the seed fits `u32`, else 2); no allocation.
pub fn seed_limbs(seed: u64) -> ([u32; 2], usize) {
    // proof: low-word truncation is the LE limb split by construction.
    #[allow(clippy::cast_possible_truncation)]
    let lo = seed as u32;
    // proof: shift-then-truncate keeps the high limb; no sign loss (u64).
    #[allow(clippy::cast_possible_truncation)]
    let hi = (seed >> 32) as u32;
    if hi == 0 {
        ([lo, 0], 1)
    } else {
        ([lo, hi], 2)
    }
}

/// One `hashmix` step; the multiplier cell is input-output exactly like the
/// NumPy source (`bit_generator.pyx:158-164`). Wrapping `u32` throughout.
fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^= v >> XSHIFT;
    v
}

/// One `mix` step (`bit_generator.pyx:166-169`). Wrapping `u32` throughout.
fn mix(x: u32, y: u32) -> u32 {
    let mut r = (MIX_MULT_L.wrapping_mul(x)).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^= r >> XSHIFT;
    r
}

/// SeedSequence pool for a `u64` seed: zeroed pool of 4, entropy hashed in
/// up to the pool size, off-diagonal mixing so late bits affect early bits,
/// then the (here empty for `u64` seeds) trailing-entropy tail
/// (`bit_generator.pyx:351-384`). The tail loop is retained verbatim: `u64`
/// seeds always fit 2 limbs, so it covers zero iterations by construction.
pub fn seedsequence_pool(seed: u64) -> [u32; SEED_POOL] {
    let (limbs, n_limbs) = seed_limbs(seed);
    let mut mixer = [0u32; SEED_POOL];
    let mut hash_const = INIT_A;
    let mut i = 0usize;
    while i < SEED_POOL {
        let e = if i < n_limbs { limbs[i] } else { 0 };
        mixer[i] = hashmix(e, &mut hash_const);
        i += 1;
    }
    let mut i_src = 0usize;
    while i_src < SEED_POOL {
        let mut i_dst = 0usize;
        while i_dst < SEED_POOL {
            if i_src != i_dst {
                let h = hashmix(mixer[i_src], &mut hash_const);
                mixer[i_dst] = mix(mixer[i_dst], h);
            }
            i_dst += 1;
        }
        i_src += 1;
    }
    let mut i_src = SEED_POOL;
    while i_src < n_limbs {
        let mut i_dst = 0usize;
        while i_dst < SEED_POOL {
            let h = hashmix(limbs[i_src], &mut hash_const);
            mixer[i_dst] = mix(mixer[i_dst], h);
            i_dst += 1;
        }
        i_src += 1;
    }
    mixer
}

/// `generate_state(4, uint64)`: 8 `u32` results cycling the pool, then
/// little-endian u32 pairs into 4 `u64` (`bit_generator.pyx:437-460`).
/// The input-seed BE convention and this internal LE pairing are distinct.
pub fn generate_state_u64(pool: [u32; SEED_POOL]) -> [u64; 4] {
    let mut hash_const = INIT_B;
    let mut words = [0u32; 8];
    let mut k = 0usize;
    while k < 8 {
        let mut v = pool[k % SEED_POOL];
        v ^= hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hash_const);
        v ^= v >> XSHIFT;
        words[k] = v;
        k += 1;
    }
    let mut out = [0u64; 4];
    let mut i = 0usize;
    while i < 4 {
        // proof: u32 -> u64 widens; shift/or assembles the LE pair exactly.
        out[i] = (words[2 * i] as u64) | ((words[2 * i + 1] as u64) << 32);
        i += 1;
    }
    out
}

/// PCG64-XSL-RR-128/64 with the exact NumPy source seed, step, output, and
/// cached-u32 discipline (`pcg64.h` + `pcg64.c`). `u128` state/increment;
/// `wrapping_*` so debug builds match the C wraparound bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_u32: bool,
    cached: u32,
}

impl Pcg64 {
    /// Source seeding (`pcg_setseq_128_srandom_r`): `state = 0`,
    /// `inc = (initseq << 1) | 1`, step, `state += initstate`, step.
    /// `initstate` is words 0/1 as high/low, `initseq` words 2/3.
    pub fn srandom(initstate: u128, initseq: u128) -> Self {
        let mut rng = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            has_u32: false,
            cached: 0,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    /// Seed from a continuation `u64`: pool -> 4 state words -> `srandom`.
    /// Words 0/1 are the high/low `u128` state, words 2/3 the sequence
    /// (mirrors `_pcg64.pyx:132-135` on little-endian pairing).
    pub fn from_seed(seed: u64) -> Self {
        let pool = seedsequence_pool(seed);
        let words = generate_state_u64(pool);
        let initstate = ((words[0] as u128) << 64) | (words[1] as u128);
        let initseq = ((words[2] as u128) << 64) | (words[3] as u128);
        Pcg64::srandom(initstate, initseq)
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULT)
            .wrapping_add(self.inc);
    }

    /// `next64`: step FIRST, then XSL-RR of the NEW state
    /// (`pcg_setseq_128_xsl_rr_64_random_r` non-Windows path).
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        // proof: `>> 64` / low-word truncation keep hi/lo exact (u128 -> u64).
        #[allow(clippy::cast_possible_truncation)]
        let hi = (self.state >> 64) as u64;
        #[allow(clippy::cast_possible_truncation)]
        let lo = self.state as u64;
        // proof: `hi >> 58 < 64` always fits `u32` (rotation amount).
        #[allow(clippy::cast_possible_truncation)]
        let rot = (hi >> 58) as u32;
        (hi ^ lo).rotate_right(rot)
    }

    /// `next32`: low half first, cached high half second; the cache starts
    /// empty (`pcg64_next32`, `has_uint32 = 0` at seed time).
    pub fn next_u32(&mut self) -> u32 {
        if self.has_u32 {
            self.has_u32 = false;
            return self.cached;
        }
        let v = self.next_u64();
        self.has_u32 = true;
        // proof: shifts/truncation split the u64 into exact halves.
        #[allow(clippy::cast_possible_truncation)]
        let high = (v >> 32) as u32;
        #[allow(clippy::cast_possible_truncation)]
        let low = v as u32;
        self.cached = high;
        low
    }

    /// `(state_hi, state_lo, inc_hi, inc_lo)` for diagnostics/snapshots.
    pub fn state_words(&self) -> (u64, u64, u64, u64) {
        // proof: shifts/truncation split each u128 into exact halves.
        #[allow(clippy::cast_possible_truncation)]
        let (shi, slo) = ((self.state >> 64) as u64, self.state as u64);
        #[allow(clippy::cast_possible_truncation)]
        let (ihi, ilo) = ((self.inc >> 64) as u64, self.inc as u64);
        (shi, slo, ihi, ilo)
    }

    /// Whether a cached high half is pending (diagnostic only).
    pub fn has_cached(&self) -> bool {
        self.has_u32
    }

    /// The pending cached high half (diagnostic only).
    pub fn cached(&self) -> u32 {
        self.cached
    }
}

/// Masked inclusive interval (`random_interval`, `distributions.c:1084`):
/// smallest all-ones mask `>= max`, then `next & mask` until `<= max`.
/// `max == 0` returns 0 consuming nothing; `max <= u32::MAX` rides
/// `next_u32`, wider bounds ride `next_u64`. Never Lemire, never modulo.
pub fn interval(rng: &mut Pcg64, max: u64) -> u64 {
    if max == 0 {
        return 0;
    }
    let mut mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    if max <= u64::from(u32::MAX) {
        loop {
            let v = u64::from(rng.next_u32()) & mask;
            if v <= max {
                return v;
            }
        }
    } else {
        loop {
            let v = rng.next_u64() & mask;
            if v <= max {
                return v;
            }
        }
    }
}

/// Descending Fisher-Yates over one wall: raw `0..136` array shuffled
/// `i = 135..1` with `interval(i)` (the 8-byte `int64` permutation path in
/// `_shuffle_raw`: `permutation` copies `arange(136)` then `shuffle` walks i
/// down to 1, position 0 never drawn). Shared by the keyed derivation and
/// the boundary-seed diagnostic so the two can never diverge.
pub fn shuffle136(rng: &mut Pcg64) -> [u8; WALL_TILES] {
    let mut wall = [0u8; WALL_TILES];
    let mut i = 0usize;
    while i < WALL_TILES {
        // proof: `i < 136` always fits `u8`.
        #[allow(clippy::cast_possible_truncation)]
        let t = i as u8;
        wall[i] = t;
        i += 1;
    }
    let mut pos: usize = WALL_TILES - 1;
    loop {
        // proof: both bounds stay `< 136` (`pos` descends from 135,
        // `interval` returns `<= pos`).
        #[allow(clippy::cast_possible_truncation)]
        let bound = pos as u64;
        #[allow(clippy::cast_possible_truncation)]
        let j = interval(rng, bound) as usize;
        wall.swap(pos, j);
        if pos == 1 {
            break;
        }
        pos -= 1;
    }
    wall
}

/// Continuation wall for `hand >= 1`: [`shuffle136`] driven by the keyed
/// [`Pcg64`]. Hand 0 is `Err(HandZero)`; the caller selects the supplied
/// schedule tiles instead.
pub fn derive_hand_wall(
    schedule_id: &str,
    digest_noprefix: &str,
    hand: u64,
) -> Result<[u8; WALL_TILES], WallError> {
    if hand == 0 {
        return Err(WallError::HandZero);
    }
    let seed = continuation_seed(schedule_id, digest_noprefix, hand);
    let mut rng = Pcg64::from_seed(seed);
    Ok(shuffle136(&mut rng))
}

/// One u53 draw from a raw 64-bit stream word: `(word >> 11) * 2^-53`.
/// Both endpoints are exact (`0 -> 0.0`, `u64::MAX -> 0.9999999999999999`,
/// bit `0x3fefffffffffffff`); no modulo, no clamp, no repair.
pub fn ctr_u53_word(word: u64) -> f64 {
    // proof: `word >> 11 < 2^53` is exactly representable in `f64`.
    #[allow(clippy::cast_precision_loss)]
    let mantissa = (word >> 11) as f64;
    mantissa * U53_SCALE
}

/// Batched u53 reserves over the EXISTING native semantic counter stream
/// (`hydra_feed::digest::ctr_block`, same bytes as
/// `contracts::ctr_stream_bytes`): draw `k` reads the 8 bytes at
/// `[cursor + 8k, cursor + 8k + 8)` as big-endian `u64` through
/// [`ctr_u53_word`]. Checked arithmetic throughout; the seed-length lane
/// and cursor advance stay caller-owned (mirrors `RandomStream.get_bytes`
/// slicing without the live cursor).
pub fn ctr_u53_window(
    seed: &[u8],
    cursor: u64,
    draws: usize,
) -> Result<Vec<f64>, WallError> {
    if seed.is_empty() {
        return Err(WallError::EmptySeed);
    }
    let total = draws
        .checked_mul(U53_BYTES)
        .ok_or(WallError::WindowOverflow)?;
    let total_u64 = u64::try_from(total).map_err(|_| WallError::WindowOverflow)?;
    let _end = cursor
        .checked_add(total_u64)
        .ok_or(WallError::WindowOverflow)?;
    let first_block = cursor / CTR_BLOCK;
    // proof: remainder `< 32` fits `usize` on every target.
    #[allow(clippy::cast_possible_truncation)]
    let offset = (cursor % CTR_BLOCK) as usize;
    let window = offset.checked_add(total).ok_or(WallError::WindowOverflow)?;
    // proof: `CTR_BLOCK` is the 32-byte const, fits `usize` everywhere.
    #[allow(clippy::cast_possible_truncation)]
    let block_bytes = CTR_BLOCK as usize;
    let needed = window.div_ceil(block_bytes);
    let capacity = needed
        .checked_mul(block_bytes)
        .ok_or(WallError::WindowOverflow)?;
    let mut bytes: Vec<u8> = Vec::with_capacity(capacity);
    let mut block = first_block;
    let mut n = 0usize;
    while n < needed {
        bytes.extend_from_slice(&hydra_feed::digest::ctr_block(seed, block));
        block = block.checked_add(1).ok_or(WallError::BlockOverflow)?;
        n += 1;
    }
    let end = offset.checked_add(total).ok_or(WallError::WindowOverflow)?;
    let slice = bytes.get(offset..end).ok_or(WallError::WindowOverflow)?;
    let mut out = Vec::with_capacity(draws);
    for chunk in slice.chunks_exact(U53_BYTES) {
        // `chunks_exact(8)` guarantees the 8-byte shape; indices 0..8 exact.
        let word = u64::from_be_bytes([
            chunk[0],
            chunk[1],
            chunk[2],
            chunk[3],
            chunk[4],
            chunk[5],
            chunk[6],
            chunk[7],
        ]);
        out.push(ctr_u53_word(word));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    const SCHEDULE_ID: &str = "sched-wave3-001";
    const DIGEST_NO_PREFIX: &str =
        "75c313f2c3f348c33075c056e3f011a274a03579675cc08591a64d733e6e09a1";
    // Hand-1 continuation seed of that schedule (SHA first-8 BE).
    const HAND1_SEED: u64 = 4707380933110928792;

    /// SHA-256 hex of wall bytes (what the `*_digests_match_oracle` pins hold).
    fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn continuation_seed_matches_oracle_hands1_to_8() {
        // Seeds recomputed from the pinned NumPy 2.5.3 oracle key law.
        let expected: [u64; 8] = [
            4707380933110928792,
            2983181260496288213,
            12463996856400830553,
            14194551210969859827,
            15919114772721524256,
            15875094157928757932,
            10397959175035869155,
            7877327388269815412,
        ];
        let mut hand = 1u64;
        while hand <= 8 {
            #[allow(clippy::cast_possible_truncation)]
            let slot = (hand - 1) as usize;
            assert_eq!(
                continuation_seed(SCHEDULE_ID, DIGEST_NO_PREFIX, hand),
                expected[slot],
                "hand {hand}"
            );
            hand += 1;
        }
    }

    #[test]
    fn engine_seed_uses_separate_rng_scope() {
        // `rng:{ordinal}` keys, pinned against the oracle hashlib law.
        assert_eq!(
            hand_rng_seed(SCHEDULE_ID, DIGEST_NO_PREFIX, 0),
            13683845269432369280
        );
        assert_eq!(
            hand_rng_seed(SCHEDULE_ID, DIGEST_NO_PREFIX, 1),
            9583203488292165949
        );
        assert_eq!(
            hand_rng_seed(SCHEDULE_ID, DIGEST_NO_PREFIX, 2),
            2600304583356499433
        );
        assert_eq!(
            hand_rng_seed(SCHEDULE_ID, DIGEST_NO_PREFIX, 8),
            10993005885823952421
        );
        // Wall and engine scopes never collide on the same ordinal.
        assert_ne!(
            hand_rng_seed(SCHEDULE_ID, DIGEST_NO_PREFIX, 1),
            continuation_seed(SCHEDULE_ID, DIGEST_NO_PREFIX, 1)
        );
    }

    #[test]
    fn seed_limbs_are_le_with_zero_singleton() {
        assert_eq!(seed_limbs(0), ([0, 0], 1));
        assert_eq!(seed_limbs(1), ([1, 0], 1));
        assert_eq!(seed_limbs(u64::from(u32::MAX)), ([u32::MAX, 0], 1));
        assert_eq!(seed_limbs(0x1_0000_0000), ([0, 1], 2));
        assert_eq!(seed_limbs(u64::MAX), ([u32::MAX, u32::MAX], 2));
        assert_eq!(seed_limbs(HAND1_SEED), ([2491815320, 1096022532], 2));
    }

    #[test]
    fn seedsequence_pool_matches_numpy() {
        assert_eq!(
            seedsequence_pool(HAND1_SEED),
            [0x5ed67fd1, 0x2d83bbc7, 0xba5e91d0, 0xf0de49f1]
        );
        // Zero-seed pool pins the `zero -> [0]` limb convention end to end.
        assert_eq!(
            seedsequence_pool(0),
            [0xfe40eb07, 0x4f363a36, 0x4eb2009d, 0xc89a7aa7]
        );
        assert_eq!(
            generate_state_u64(seedsequence_pool(0)),
            [
                0xdb2cd7e7b0f478be,
                0xabf4641a2c71ba49,
                0x20c6ed6d9d7b8d41,
                0x2c4099de223c39d4,
            ]
        );
        assert_ne!(seedsequence_pool(0), seedsequence_pool(1));
    }

    #[test]
    fn generate_state_pairs_le_into_u64() {
        assert_eq!(
            generate_state_u64(seedsequence_pool(HAND1_SEED)),
            [
                0x5f80f616f0b3659f,
                0xceaf8adf75494b7c,
                0x203847febaaf5473,
                0x374cedecb1e82bdd,
            ]
        );
        // LE pairing of MIXED words (not raw input): generate_state hashes
        // every pool word through the INIT_B/MULT_B chain before LE-pairing,
        // so [0x11223344, 0xAABBCCDD, ...] yields 0x77066ed23b583429
        // (verified against an independent Python transcription of NumPy).
        let pair = generate_state_u64([0x11223344, 0xAABBCCDD, 0, 0]);
        assert_eq!(pair[0], 0x77066ed23b583429);
    }

    #[test]
    fn pcg_srandom_state_inc_and_raw_match() {
        let mut rng = Pcg64::from_seed(HAND1_SEED);
        assert_eq!(
            rng.state_words(),
            (
                0xb45798c72f7e2edb,
                0x79ffc393c3642f8e,
                0x40708ffd755ea8e6,
                0x6e99dbd963d057bb,
            )
        );
        assert!(!rng.has_cached());
        let expected: [u64; 6] = [
            0x42bd46c86d7df9f4,
            0x39eaf0fbdc10030f,
            0x68f35ce3619d92dd,
            0x6f985d84daaa9270,
            0x4b4cb88a25cd6374,
            0x860af0b6ce2d2868,
        ];
        for want in expected {
            assert_eq!(rng.next_u64(), want);
        }
    }

    #[test]
    fn pcg_next32_returns_low_then_cached_high() {
        let mut rng = Pcg64::from_seed(HAND1_SEED);
        // Raw u64 0 is 0x42bd46c86d7df9f4: low half first, high half cached.
        assert_eq!(rng.next_u32(), 0x6d7df9f4);
        assert!(rng.has_cached());
        assert_eq!(rng.cached(), 0x42bd46c8);
        assert_eq!(rng.next_u32(), 0x42bd46c8);
        assert!(!rng.has_cached());
        // Next pair rides raw u64 1 (0x39eaf0fbdc10030f).
        assert_eq!(rng.next_u32(), 0xdc10030f);
        assert_eq!(rng.next_u32(), 0x39eaf0fb);
        assert_eq!(rng.next_u32(), 0x619d92dd);
        assert_eq!(rng.next_u32(), 0x68f35ce3);
        assert_eq!(rng.next_u32(), 0xdaaa9270);
        assert_eq!(rng.next_u32(), 0x6f985d84);
    }

    #[test]
    fn interval_zero_consumes_nothing() {
        let mut rng = Pcg64::from_seed(HAND1_SEED);
        let before = rng.state_words();
        assert_eq!(interval(&mut rng, 0), 0);
        assert_eq!(rng.state_words(), before);
        assert!(!rng.has_cached());
    }

    #[test]
    fn interval_first_draw_and_bounds_match_oracle() {
        // Fresh-generator transcript head: i=135 rides mask 0xff, first
        // accepted draw is 15 (two rejections consumed before it).
        let mut rng = Pcg64::from_seed(HAND1_SEED);
        assert_eq!(interval(&mut rng, 135), 15);
        // Full-width single-draw paths pin the active word width.
        let mut rng = Pcg64::from_seed(HAND1_SEED);
        assert_eq!(interval(&mut rng, u64::from(u32::MAX)), 0x6d7df9f4);
        let mut rng = Pcg64::from_seed(HAND1_SEED);
        assert_eq!(interval(&mut rng, u64::MAX), 0x42bd46c86d7df9f4);
        // Bounds hold across the whole shuffle range on a live generator.
        let mut rng = Pcg64::from_seed(HAND1_SEED);
        let mut i = 135u64;
        while i >= 1 {
            let v = interval(&mut rng, i);
            assert!(v <= i, "interval({i}) = {v}");
            if i == 1 {
                break;
            }
            i -= 1;
        }
    }

    #[test]
    fn hand_zero_is_rejected() {
        assert_eq!(
            derive_hand_wall(SCHEDULE_ID, DIGEST_NO_PREFIX, 0),
            Err(WallError::HandZero)
        );
    }

    #[test]
    fn hand1_wall_is_exact_permutation() {
        let wall = derive_hand_wall(SCHEDULE_ID, DIGEST_NO_PREFIX, 1).unwrap();
        let expected: [u8; WALL_TILES] = [
            92, 86, 77, 87, 61, 128, 16, 130, 101, 53, 3, 8, 24, 44, 37, 10, 1,
            48, 114, 118, 31, 39, 95, 38, 110, 71, 32, 72, 17, 109, 115, 94,
            62, 120, 74, 67, 117, 51, 76, 60, 96, 80, 18, 99, 0, 97, 54, 93,
            119, 11, 6, 64, 90, 84, 21, 107, 35, 123, 100, 121, 25, 65, 85,
            47, 82, 28, 129, 75, 79, 30, 106, 43, 57, 12, 98, 134, 113, 105,
            13, 14, 103, 26, 5, 46, 55, 4, 102, 50, 70, 22, 42, 59, 7, 63,
            126, 36, 23, 45, 69, 29, 127, 34, 131, 81, 73, 83, 78, 135, 66,
            56, 124, 91, 68, 88, 52, 111, 122, 40, 20, 58, 33, 133, 41, 125,
            19, 49, 2, 108, 89, 9, 27, 104, 116, 132, 112, 15,
        ];
        assert_eq!(wall, expected);
    }

    #[test]
    fn hands1_to_8_digests_match_oracle() {
        // SHA-256 over the 136 wall bytes, pinned against NumPy 2.5.3.
        let expected = [
            "9138cb9e45994a2ff0b1f607e50ed24ab95d8b9d10961ba72b9d1cc1ff71e82a",
            "05420c92508eeaf54b415f6b6c833dc29394a675ed10d81c7d33f49455209cd6",
            "b9d24558a46c2cc9dccb85a0c4d0284f2bd05399004b3da9ebcced91c4354e42",
            "a74beec58c3f92c3864d9bd95d0ce4b882ef3a2aee0eaf4e8f042f23af80492c",
            "4b4baedb24ee3659cab1447814bb3244a2cabe1d25fcac55c3b798ef0a896379",
            "343167c687a75803fbc0196247d3ba148c7947fd68e1df1fa16608c8c8848731",
            "531ebb70d94e46168df5172be75abf574b55b234e001509e9f50a577beed25fc",
            "d4538739efa2557d1bf410d0d0606130cf507767949a4a9e1dd2dc181157bb20",
        ];
        let mut hand = 1u64;
        while hand <= 8 {
            let wall = derive_hand_wall(SCHEDULE_ID, DIGEST_NO_PREFIX, hand).unwrap();
            #[allow(clippy::cast_possible_truncation)]
            let slot = (hand - 1) as usize;
            assert_eq!(sha256_hex(&wall), expected[slot], "hand {hand}");
            // Every wall is a permutation of 0..136 (sorted check).
            let mut sorted = wall;
            sorted.sort_unstable();
            let mut i = 0usize;
            while i < WALL_TILES {
                #[allow(clippy::cast_possible_truncation)]
                let want = i as u8;
                assert_eq!(sorted[i], want, "hand {hand} position {i}");
                i += 1;
            }
            hand += 1;
        }
    }

    #[test]
    fn boundary_seeds_match_oracle() {
        // Direct continuation seeds (boundary inputs, no schedule key).
        // The shuffle below intentionally inlines the Fisher-Yates walk
        // instead of calling `shuffle136`, so this test stays an
        // independent re-derivation against the oracle goldens.
        let cases: [(u64, &str, [u8; 8], [u8; 4]); 6] = [
            (
                0,
                "6efc0b8fdab87abec12674af06894740c04ac0381fd241549713e4b8c6352bfc",
                [93, 110, 71, 54, 107, 87, 99, 101],
                [59, 29, 33, 95],
            ),
            (
                1,
                "9e9d238a51fec8b3577e288be17f90e4c496c0ad0e51450cef519a7b4eb6a9c1",
                [105, 20, 73, 107, 22, 110, 8, 24],
                [36, 69, 72, 102],
            ),
            (
                u64::MAX,
                "834e84b3aab836cda414bc85d84cb3d77a7f794e8b37af9727cda1b8cd331016",
                [79, 63, 109, 85, 123, 91, 70, 16],
                [25, 74, 89, 122],
            ),
            (
                0x8000_0000_0000_0000,
                "96debfdeac0dc71e034f1e8da5c4b9f9e8f7327d9c08c6d11cd9f5c70958371e",
                [2, 60, 7, 108, 131, 34, 135, 118],
                [37, 100, 11, 20],
            ),
            (
                0x1_0000_0000,
                "b1b4eda3053a51a5eba716375e43b742eaa6a679c6ff7839344879a8a0846230",
                [2, 76, 88, 96, 16, 48, 62, 87],
                [51, 82, 104, 37],
            ),
            (
                42,
                "de65ef524fd287e75619808ff71398686779975a7f53c0c2b21ed41e67d64b80",
                [88, 122, 24, 25, 101, 116, 4, 2],
                [113, 14, 36, 97],
            ),
        ];
        for (seed, digest, head, tail) in cases {
            let mut rng = Pcg64::from_seed(seed);
            let mut wall = [0u8; WALL_TILES];
            let mut i = 0usize;
            while i < WALL_TILES {
                #[allow(clippy::cast_possible_truncation)]
                let t = i as u8;
                wall[i] = t;
                i += 1;
            }
            let mut pos: usize = WALL_TILES - 1;
            loop {
                #[allow(clippy::cast_possible_truncation)]
                let bound = pos as u64;
                #[allow(clippy::cast_possible_truncation)]
                let j = interval(&mut rng, bound) as usize;
                wall.swap(pos, j);
                if pos == 1 {
                    break;
                }
                pos -= 1;
            }
            assert_eq!(sha256_hex(&wall), digest, "seed {seed}");
            assert_eq!(&wall[..8], &head, "seed {seed} head");
            assert_eq!(&wall[WALL_TILES - 4..], &tail, "seed {seed} tail");
        }
    }

    #[test]
    fn u53_endpoints_are_exact() {
        assert_eq!(ctr_u53_word(0).to_bits(), 0x0);
        assert_eq!(ctr_u53_word(u64::MAX).to_bits(), 0x3fefffffffffffff);
        assert_eq!(ctr_u53_word(u64::MAX), 0.9999999999999999);
    }

    #[test]
    fn u53_window_matches_semantic_ctr_stream() {
        // Seed bytes are the raw wave3 schedule digest (no prefix on this path).
        let seed: [u8; 32] = [
            0x75, 0xc3, 0x13, 0xf2, 0xc3, 0xf3, 0x48, 0xc3, 0x30, 0x75, 0xc0,
            0x56, 0xe3, 0xf0, 0x11, 0xa2, 0x74, 0xa0, 0x35, 0x79, 0x67, 0x5c,
            0xc0, 0x85, 0x91, 0xa6, 0x4d, 0x73, 0x3e, 0x6e, 0x09, 0xa1,
        ];
        let at = |cursor: u64| ctr_u53_window(&seed, cursor, 1).unwrap()[0].to_bits();
        assert_eq!(at(0), 0x3fa89ef25c942a80);
        assert_eq!(at(8), 0x3fc0a492ba1f45d4);
        assert_eq!(at(24), 0x3fcc7fcd061f3fa8);
        // Multi-draw batching equals repeated single draws; 8 bytes each.
        let batch = ctr_u53_window(&seed, 0, 4).unwrap();
        assert_eq!(batch.len(), 4);
        assert_eq!(batch[0].to_bits(), at(0));
        assert_eq!(batch[1].to_bits(), at(8));
        assert_eq!(batch[2].to_bits(), at(16));
        assert_eq!(batch[3].to_bits(), at(24));
        // Fail-closed edges.
        assert_eq!(ctr_u53_window(&[], 0, 1), Err(WallError::EmptySeed));
        assert_eq!(
            ctr_u53_window(&seed, u64::MAX, 1),
            Err(WallError::WindowOverflow)
        );
    }
}
