//! Compact bitpack for the cold shard (P4-B, cold only — NEVER hot).
//!
//! Reference pattern (read-only): hydra1 `bc-shards compact.rs` — bit pack
//! the wide boolean planes on disk, expand to `u8` at collate. Widths are
//! re-derived for hydra2 (6792-action table, `T` buckets `{32,64,128,256}`).
//!
//! On-disk legal field: `LEGAL_BITS = 6792` bits LSB-first in
//! [`LEGAL_BIT_BYTES`] `= 849` bytes plus one reserved zero byte, i.e. exactly
//! [`LEGAL_PACKED_BYTES`] `= 850`B on disk. The reserved byte is verified on
//! read (nonzero fails closed) so a torn write cannot alias a valid mask.
//! History masks pack the same way: `(T+7)/8` bytes per row.
//!
//! Everything else in a compact row is byte-aligned; kinds narrow `i64 → u8`
//! (ids `0..=19` fit), counts stay `u8`, scalars narrow with range checks.
//! Collate widens back (`u8 → int32` counts with the G2 post-cast note — see
//! `crate::collate` — and bit `→ u8` legal/mask expansion).

use core::fmt;

/// Legal width: the frozen v1 action table
/// (`configs/contracts/action_table_v1.json`, 6792 entries).
pub const LEGAL_BITS: usize = 6792;
/// Bit bytes for the legal mask (`6792 / 8` exact, no padding bits).
pub const LEGAL_BIT_BYTES: usize = 849;
/// On-disk legal field: bit bytes + one reserved zero byte (verified on read).
pub const LEGAL_PACKED_BYTES: usize = 850;

/// Compact failure taxonomy: fail-closed, numeric, never a `String` hot
/// (this module is cold-only, but the shape stays hot-compatible).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactError {
    /// Mask/slice length mismatch (`expected`, `got`).
    Length { expected: usize, got: usize },
    /// A mask element is not `0`/`1` (`offset` = element index).
    NotBit { offset: usize },
    /// The reserved trailing byte of the packed legal field is nonzero.
    ReservedNonzero { got: u8 },
    /// A narrowed value does not fit (`offset` = element index).
    NarrowRange { offset: usize },
}

impl fmt::Display for CompactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            CompactError::Length { expected, got } => {
                write!(f, "compact length: expected {expected}, got {got}")
            }
            CompactError::NotBit { offset } => {
                write!(f, "compact mask element {offset} is not 0/1")
            }
            CompactError::ReservedNonzero { got } => {
                write!(f, "compact legal reserved byte nonzero: {got}")
            }
            CompactError::NarrowRange { offset } => {
                write!(f, "compact narrow range at element {offset}")
            }
        }
    }
}

impl std::error::Error for CompactError {}

/// LSB-first bit writer over an owned byte buffer (hydra1 `BitWriter` idiom:
/// bit 0 of the mask is bit 0 of byte 0).
#[derive(Debug, Default)]
pub struct BitWriter {
    buf: Vec<u8>,
    acc: u8,
    nbits: u8,
}

impl BitWriter {
    /// Empty writer (no allocation until the first full byte flushes).
    pub fn new() -> Self {
        Self::default()
    }

    /// Empty writer with capacity for `bytes` flushed bytes.
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buf: Vec::with_capacity(bytes),
            acc: 0,
            nbits: 0,
        }
    }

    /// Push one bit (`false` = 0, `true` = 1).
    #[inline]
    pub fn push(&mut self, bit: bool) {
        if bit {
            self.acc |= 1 << self.nbits;
        }
        self.nbits += 1;
        if self.nbits == 8 {
            self.buf.push(self.acc);
            self.acc = 0;
            self.nbits = 0;
        }
    }

    /// Push `bits[..n]` as `0`/`1` bytes; fails closed on non-bit elements.
    pub fn push_mask(&mut self, bits: &[u8], n: usize) -> Result<(), CompactError> {
        if bits.len() < n {
            return Err(CompactError::Length {
                expected: n,
                got: bits.len(),
            });
        }
        let mut i = 0usize;
        while i < n {
            let b = bits[i];
            if b > 1 {
                return Err(CompactError::NotBit { offset: i });
            }
            self.push(b == 1);
            i += 1;
        }
        Ok(())
    }

    /// Flush the partial tail byte (zero-padded) and return the bytes.
    pub fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.buf.push(self.acc);
        }
        self.buf
    }
}

/// LSB-first bit reader over a borrowed byte slice (hydra1 `BitReader` idiom).
#[derive(Debug, Clone, Copy)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    /// Reader over `bytes` starting at bit 0.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    /// Total bits available.
    #[inline]
    pub fn total_bits(&self) -> usize {
        self.bytes.len() * 8
    }

    /// Bits remaining.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.total_bits().saturating_sub(self.bit_pos)
    }

    /// Read one bit (`None` past the end).
    #[inline]
    pub fn next_bit(&mut self) -> Option<bool> {
        if self.bit_pos >= self.total_bits() {
            return None;
        }
        let byte = self.bytes[self.bit_pos / 8];
        let bit = (byte >> (self.bit_pos % 8)) & 1 == 1;
        self.bit_pos += 1;
        Some(bit)
    }

    /// Expand `n` bits into `out` as `0`/`1` bytes (the collate expansion).
    pub fn expand_into(&mut self, out: &mut [u8], n: usize) -> Result<(), CompactError> {
        if out.len() < n {
            return Err(CompactError::Length {
                expected: n,
                got: out.len(),
            });
        }
        if self.remaining() < n {
            return Err(CompactError::Length {
                expected: n,
                got: self.remaining(),
            });
        }
        let mut i = 0usize;
        while i < n {
            // `remaining` check above pins `Some`; index math stays in-bounds.
            let bit = self.next_bit().unwrap_or(false);
            out[i] = u8::from(bit);
            i += 1;
        }
        Ok(())
    }
}

/// Pack a 6792-element `0`/`1` legal mask into the 850B on-disk field.
pub fn pack_legal_bits(mask: &[u8]) -> Result<[u8; LEGAL_PACKED_BYTES], CompactError> {
    if mask.len() != LEGAL_BITS {
        return Err(CompactError::Length {
            expected: LEGAL_BITS,
            got: mask.len(),
        });
    }
    let mut w = BitWriter::with_capacity(LEGAL_BIT_BYTES);
    w.push_mask(mask, LEGAL_BITS)?;
    let bytes = w.finish();
    // 6792 is a multiple of 8: no tail byte, exactly 849 flushed bytes.
    if bytes.len() != LEGAL_BIT_BYTES {
        return Err(CompactError::Length {
            expected: LEGAL_BIT_BYTES,
            got: bytes.len(),
        });
    }
    let mut out = [0u8; LEGAL_PACKED_BYTES];
    out[..LEGAL_BIT_BYTES].copy_from_slice(&bytes);
    out[LEGAL_BIT_BYTES] = 0; // reserved, verified on read
    Ok(out)
}

/// Expand the 850B on-disk legal field into 6792 `0`/`1` bytes (collate).
pub fn unpack_legal_bits(
    packed: &[u8; LEGAL_PACKED_BYTES],
    out: &mut [u8],
) -> Result<(), CompactError> {
    if out.len() != LEGAL_BITS {
        return Err(CompactError::Length {
            expected: LEGAL_BITS,
            got: out.len(),
        });
    }
    if packed[LEGAL_BIT_BYTES] != 0 {
        return Err(CompactError::ReservedNonzero {
            got: packed[LEGAL_BIT_BYTES],
        });
    }
    let mut r = BitReader::new(&packed[..LEGAL_BIT_BYTES]);
    r.expand_into(out, LEGAL_BITS)
}

/// Pack a `0`/`1` history mask prefix into `(t_len+7)/8` bytes (zero-padded tail).
pub fn pack_hist_mask(mask: &[u8], t_len: usize) -> Result<Vec<u8>, CompactError> {
    let mut w = BitWriter::with_capacity(t_len.div_ceil(8));
    w.push_mask(mask, t_len)?;
    Ok(w.finish())
}

/// Expand packed history-mask bytes into `t_len` `0`/`1` bytes.
pub fn unpack_hist_mask(
    packed: &[u8],
    out: &mut [u8],
    t_len: usize,
) -> Result<(), CompactError> {
    if out.len() < t_len {
        return Err(CompactError::Length {
            expected: t_len,
            got: out.len(),
        });
    }
    if packed.len() != t_len.div_ceil(8) {
        return Err(CompactError::Length {
            expected: t_len.div_ceil(8),
            got: packed.len(),
        });
    }
    let mut r = BitReader::new(packed);
    r.expand_into(out, t_len)
}

#[cfg(test)]
mod compact_tests {
    use super::*;

    #[test]
    fn legal_size_is_850_on_disk() {
        assert_eq!(LEGAL_BITS, 6792);
        assert_eq!(LEGAL_BIT_BYTES, 849);
        assert_eq!(LEGAL_PACKED_BYTES, 850);
        assert_eq!(LEGAL_BIT_BYTES, LEGAL_BITS / 8);
        assert_eq!(LEGAL_BITS % 8, 0);
    }

    #[test]
    fn legal_pack_unpack_round_trip() {
        let mut mask = vec![0u8; LEGAL_BITS];
        // Sparse + dense + edge-bit pattern.
        mask[0] = 1;
        mask[7] = 1;
        mask[8] = 1;
        mask[140] = 1;
        mask[6791] = 1;
        let mut i = 0usize;
        while i < LEGAL_BITS {
            if i.is_multiple_of(97) {
                mask[i] = 1;
            }
            i += 1;
        }
        let packed = pack_legal_bits(&mask).expect("pack ok");
        assert_eq!(packed.len(), LEGAL_PACKED_BYTES);
        assert_eq!(packed[LEGAL_PACKED_BYTES - 1], 0);
        let mut out = vec![0u8; LEGAL_BITS];
        unpack_legal_bits(&packed, &mut out).expect("unpack ok");
        assert_eq!(out, mask);
    }

    #[test]
    fn legal_all_zeros_and_all_ones() {
        for fill in [0u8, 1u8] {
            let mask = vec![fill; LEGAL_BITS];
            let packed = pack_legal_bits(&mask).expect("pack ok");
            let expect = if fill == 1 { 0xFFu8 } else { 0x00u8 };
            assert!(packed[..LEGAL_BIT_BYTES].iter().all(|b| *b == expect));
            let mut out = vec![9u8; LEGAL_BITS];
            unpack_legal_bits(&packed, &mut out).expect("unpack ok");
            assert!(out.iter().all(|b| *b == fill));
        }
    }

    #[test]
    fn legal_rejects_fail_closed() {
        // Wrong length.
        assert_eq!(
            pack_legal_bits(&[0u8; 32]).unwrap_err(),
            CompactError::Length {
                expected: LEGAL_BITS,
                got: 32
            }
        );
        // Non-bit element.
        let mut mask = vec![0u8; LEGAL_BITS];
        mask[6790] = 2;
        assert_eq!(
            pack_legal_bits(&mask).unwrap_err(),
            CompactError::NotBit { offset: 6790 }
        );
        // Reserved byte tamper.
        let mask = vec![1u8; LEGAL_BITS];
        let mut packed = pack_legal_bits(&mask).expect("pack ok");
        packed[LEGAL_PACKED_BYTES - 1] = 0xA5;
        let mut out = vec![0u8; LEGAL_BITS];
        assert_eq!(
            unpack_legal_bits(&packed, &mut out).unwrap_err(),
            CompactError::ReservedNonzero { got: 0xA5 }
        );
        // Short output.
        let packed = pack_legal_bits(&mask).expect("pack ok");
        let mut short = vec![0u8; 100];
        assert!(unpack_legal_bits(&packed, &mut short).is_err());
    }

    #[test]
    fn hist_mask_round_trip_all_buckets() {
        for t in [32usize, 64, 128, 256] {
            let mut mask = vec![1u8; t];
            // Valid prefix zeros past a short history are the caller's; pack
            // the full bucket width (collate slices the valid prefix).
            mask[t - 1] = 0;
            let packed = pack_hist_mask(&mask, t).expect("pack ok");
            assert_eq!(packed.len(), t.div_ceil(8));
            let mut out = vec![9u8; t];
            unpack_hist_mask(&packed, &mut out, t).expect("unpack ok");
            assert_eq!(out, mask);
        }
    }

    #[test]
    fn bit_reader_past_end_is_none() {
        let mut r = BitReader::new(&[0b1010_0101u8]);
        let mut bits = [false; 8];
        let mut i = 0usize;
        while i < 8 {
            bits[i] = r.next_bit().expect("bit");
            i += 1;
        }
        assert_eq!(bits, [true, false, true, false, false, true, false, true]);
        assert_eq!(r.next_bit(), None);
    }
}
