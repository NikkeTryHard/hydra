//! Zero-alloc MJAI tile codec: const LUTs only.
//!
//! Physical layout (Tenhou order, mirrors `tile.rs` + `tiles.py` without
//! copying either): suited blocks `m` = 0, `p` = 36, `s` = 72 (nine ranks
//! x four copies); honors `E/S/W/N/P/F/C` from 108. The red five is the
//! FIRST copy of each suited-five block (16/52/88); a plain `5x` is the
//! SECOND copy (17/53/89); `"0x"` aliases the red copy.
//!
//! No `match` ladders on hot paths, no `format!`, no `Vec`: one 256-byte
//! class table (2-3 L1 loads per tile) plus two const reverse tables.

/// Red-five physical ids: first copy of each suited five block.
pub const AKA_IDS: [u8; 3] = [16, 52, 88];

/// Fail-closed tile error: static reason only, never allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileError(pub &'static str);

impl core::fmt::Display for TileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for TileError {}

// --- Char classes (single LUT byte per ASCII char) --------------------------
// `0x00`        = `0` red-alias rank
// `0x01..=0x09` = digit rank 1..=9
// `0x10..=0x12` = suits m/p/s
// `0x20..=0x26` = honors E/S/W/N/P/F/C
// `0x30`        = `r` red suffix
// `0xFF`        = not a tile char
const fn build_lut() -> [u8; 256] {
    let mut t = [0xFFu8; 256];
    t[b'0' as usize] = 0x00;
    let mut d: u8 = 1;
    while d <= 9 {
        t[(b'0' + d) as usize] = d;
        d += 1;
    }
    t[b'm' as usize] = 0x10;
    t[b'p' as usize] = 0x11;
    t[b's' as usize] = 0x12;
    t[b'E' as usize] = 0x20;
    t[b'S' as usize] = 0x21;
    t[b'W' as usize] = 0x22;
    t[b'N' as usize] = 0x23;
    t[b'P' as usize] = 0x24;
    t[b'F' as usize] = 0x25;
    t[b'C' as usize] = 0x26;
    t[b'r' as usize] = 0x30;
    t
}

/// Char-class table: tile-char roles, `0xFF` for anything else.
pub const TILE_LUT: [u8; 256] = build_lut();

/// Suit block base for a suit class byte, `None` for non-suits.
const fn suit_base(class: u8) -> Option<u8> {
    match class {
        0x10 => Some(0),
        0x11 => Some(36),
        0x12 => Some(72),
        _ => None,
    }
}

const SUFFIX_OF_SUIT: [u8; 3] = *b"mps";
const HONOR_LETTER: [u8; 7] = *b"ESWNPFC";

/// Physical id -> (canonical MJAI bytes, len): red fives render `5xr`
/// (len 3), plain fives render `5x` (len 2), suits `Nx`, honors `X`.
const fn build_rev() -> [([u8; 4], u8); 136] {
    let mut t = [([0u8; 4], 0u8); 136];
    let mut id: usize = 0;
    while id < 136 {
        if id < 108 {
            let suit = id / 36;
            let rank = (id % 36) / 4 + 1; // 1..=9
            let copy = (id % 36) % 4; // 0 == aka slot for fives
            if rank == 5 && copy == 0 {
                t[id] = ([b'5', SUFFIX_OF_SUIT[suit], b'r', 0], 3);
            } else {
                // proof: rank 1..=9, byte value <256.
                #[allow(clippy::cast_possible_truncation)]
                {
                    t[id] = ([b'0' + rank as u8, SUFFIX_OF_SUIT[suit], 0, 0], 2);
                }
            }
        } else {
            t[id] = ([HONOR_LETTER[(id - 108) / 4], 0, 0, 0], 1);
        }
        id += 1;
    }
    t
}

/// Canonical rendering table: `TILE_REV[id]` is the bytes + len.
pub const TILE_REV: [([u8; 4], u8); 136] = build_rev();

/// Copy-pool table: red copies pool to the single aka id; plain-five
/// non-aka copies pool to the three non-aka ids; every other id pools
/// to its full four-copy block.
const fn build_pool() -> [([u8; 4], u8); 136] {
    let mut t = [([0u8; 4], 0u8); 136];
    let mut id: usize = 0;
    while id < 136 {
        let base = (id / 4) * 4;
        let is_aka_slot = id < 108 && (id % 36) / 4 + 1 == 5 && (id % 36).is_multiple_of(4);
        let is_plain_five = id < 108 && (id % 36) / 4 + 1 == 5 && !is_aka_slot;
        if is_aka_slot {
            // proof: table id <136 fits in u8.
            #[allow(clippy::cast_possible_truncation)]
            {
                t[id] = ([id as u8, 0, 0, 0], 1);
            }
        } else if is_plain_five {
            // `id` is base+1/base+2/base+3 with base the aka slot.
            // proof: base+1..3 <136 fits in u8.
            #[allow(clippy::cast_possible_truncation)]
            {
                t[id] = ([(base + 1) as u8, (base + 2) as u8, (base + 3) as u8, 0], 3);
            }
        } else {
            // proof: block base 0..136 fits in u8.
            #[allow(clippy::cast_possible_truncation)]
            {
                t[id] = (
                    [
                        base as u8,
                        (base + 1) as u8,
                        (base + 2) as u8,
                        (base + 3) as u8,
                    ],
                    4,
                );
            }
        }
        id += 1;
    }
    t
}

/// Copy-pool table: `TILE_POOL[id]` is the ordered copy ids + len.
pub const TILE_POOL: [([u8; 4], u8); 136] = build_pool();

/// Const tile codec: pure LUT loads, const refs out, never `Vec`.
pub struct TileCodec;

impl TileCodec {
    /// Parse one MJAI tile span into its exact physical id (0..135).
    ///
    /// `"0x"` / `"5xr"` resolve to the FIRST copy (16/52/88); plain
    /// `"5x"` resolves to the SECOND copy (17/53/89); anything else
    /// fails closed with [`TileError`].
    pub fn physical(span: &[u8]) -> Result<u8, TileError> {
        const E: TileError = TileError("bad-tile");
        match span.len() {
            1 => match TILE_LUT[span[0] as usize] {
                c if (0x20..=0x26).contains(&c) => Ok(108 + (c - 0x20) * 4),
                _ => Err(E),
            },
            2 => {
                let rank = TILE_LUT[span[0] as usize];
                let Some(base) = suit_base(TILE_LUT[span[1] as usize]) else {
                    return Err(E);
                };
                if rank == 0x00 {
                    return Ok(base + 16); // "0x" red alias
                }
                if (1..=9).contains(&rank) {
                    return Ok(if rank == 5 {
                        base + 17 // plain five skips the aka copy
                    } else {
                        base + 4 * (rank - 1)
                    });
                }
                Err(E)
            }
            3 => {
                if TILE_LUT[span[0] as usize] != 5 {
                    return Err(E); // red suffix on non-five
                }
                let Some(base) = suit_base(TILE_LUT[span[1] as usize]) else {
                    return Err(E);
                };
                if TILE_LUT[span[2] as usize] != 0x30 {
                    return Err(E);
                }
                Ok(base + 16)
            }
            _ => Err(E),
        }
    }

    /// Canonical rendering as a const ref + len (never `Vec`).
    pub fn bytes_of(tile: u8) -> Option<(&'static [u8; 4], u8)> {
        TILE_REV.get(tile as usize).map(|entry| (&entry.0, entry.1))
    }

    /// Ordered copy pool as a const ref + len (never `Vec`).
    pub fn pool_of(tile: u8) -> Option<(&'static [u8; 4], u8)> {
        TILE_POOL.get(tile as usize).map(|entry| (&entry.0, entry.1))
    }

    /// Tile type index 0..34 (aka shares its five type).
    pub fn tile_type(tile: u8) -> u8 {
        tile / 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_classes_cover_tile_chars() {
        for d in 1u8..=9 {
            assert_eq!(TILE_LUT[(b'0' + d) as usize], d);
        }
        assert_eq!(TILE_LUT[b'0' as usize], 0x00);
        assert_eq!(TILE_LUT[b'm' as usize], 0x10);
        assert_eq!(TILE_LUT[b'p' as usize], 0x11);
        assert_eq!(TILE_LUT[b's' as usize], 0x12);
        assert_eq!(TILE_LUT[b'r' as usize], 0x30);
        assert_eq!(TILE_LUT[b'X' as usize], 0xFF);
        assert_eq!(TILE_LUT[b' ' as usize], 0xFF);
    }

    #[test]
    fn aka_and_plain_five_ids() {
        assert_eq!(TileCodec::physical(b"0m"), Ok(16));
        assert_eq!(TileCodec::physical(b"0p"), Ok(52));
        assert_eq!(TileCodec::physical(b"0s"), Ok(88));
        assert_eq!(TileCodec::physical(b"5mr"), Ok(16));
        assert_eq!(TileCodec::physical(b"5pr"), Ok(52));
        assert_eq!(TileCodec::physical(b"5sr"), Ok(88));
        assert_eq!(TileCodec::physical(b"5m"), Ok(17));
        assert_eq!(TileCodec::physical(b"5p"), Ok(53));
        assert_eq!(TileCodec::physical(b"5s"), Ok(89));
    }

    #[test]
    fn full_codec_round_trip() {
        for id in 0u8..=135u8 {
            let (bytes, len) = TileCodec::bytes_of(id).unwrap();
            let back = TileCodec::physical(&bytes[..len as usize]).unwrap();
            if AKA_IDS.contains(&id) {
                assert_eq!(len, 3);
                assert_eq!(bytes[2], b'r');
                assert_eq!(back, id);
            } else if [4u8, 13u8, 22u8].contains(&TileCodec::tile_type(id)) {
                assert_eq!(back, (id / 4) * 4 + 1);
            } else {
                assert_eq!(back, (id / 4) * 4);
            }
        }
        assert!(TileCodec::bytes_of(136).is_none());
    }

    #[test]
    fn canonical_renderings_match_oracle() {
        assert_eq!(TileCodec::bytes_of(16).unwrap(), (&[b'5', b'm', b'r', 0], 3));
        assert_eq!(TileCodec::bytes_of(52).unwrap(), (&[b'5', b'p', b'r', 0], 3));
        assert_eq!(TileCodec::bytes_of(88).unwrap(), (&[b'5', b's', b'r', 0], 3));
        assert_eq!(TileCodec::bytes_of(0).unwrap(), (&[b'1', b'm', 0, 0], 2));
        assert_eq!(TileCodec::bytes_of(104).unwrap(), (&[b'9', b's', 0, 0], 2));
        assert_eq!(TileCodec::bytes_of(108).unwrap(), (&[b'E', 0, 0, 0], 1));
        assert_eq!(TileCodec::bytes_of(135).unwrap(), (&[b'C', 0, 0, 0], 1));
    }

    #[test]
    fn copy_pools_partition_the_wall() {
        let mut seen = [false; 136];
        let mut distinct = 0usize;
        for id in 0u8..=135u8 {
            let (pool, n) = TileCodec::pool_of(id).unwrap();
            // Aka renderings own the aka copies: only count the pool of the
            // canonical id for each distinct rendering.
            let (canon, _) = TileCodec::bytes_of(id).unwrap();
            let mut is_canon_owner = true;
            for earlier in 0..id {
                let (eb, _) = TileCodec::bytes_of(earlier).unwrap();
                if eb == canon {
                    is_canon_owner = false;
                    break;
                }
            }
            if !is_canon_owner {
                continue;
            }
            for copy in &pool[..n as usize] {
                assert!(!seen[*copy as usize]);
                seen[*copy as usize] = true;
                distinct += 1;
            }
        }
        assert_eq!(distinct, 136);
        // Spot pools match the `_copies_of_string` oracle matrix.
        assert_eq!(TileCodec::pool_of(16).unwrap(), (&[16, 0, 0, 0], 1));
        assert_eq!(TileCodec::pool_of(17).unwrap(), (&[17, 18, 19, 0], 3));
        assert_eq!(TileCodec::pool_of(0).unwrap(), (&[0, 1, 2, 3], 4));
        assert_eq!(TileCodec::pool_of(132).unwrap(), (&[132, 133, 134, 135], 4));
        assert!(TileCodec::pool_of(136).is_none());
    }

    #[test]
    fn rejects_fail_closed() {
        for bad in [
            &b""[..],
            b"10m",
            b"0x",
            b"3mr",
            b"X",
            b"1mr",
            b"9pr",
            b"5mrr",
            b"5",
            b"m",
            b"0m ",
            b"5r",
            b"5E",
            b"EE",
        ] {
            assert!(TileCodec::physical(bad).is_err());
        }
    }
}
