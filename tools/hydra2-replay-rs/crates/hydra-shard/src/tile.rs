//! Physical tile id <-> MJAI-string codec (Tenhou wall-less replay core).
//!
//! Reference (read-only, never copied wholesale):
//! `src/hydra2/engines/riichienv/tiles.py` (`physical_of`, `mjai_string_of`)
//! plus the red-normalization map and copy-pool order from
//! `src/hydra2/engines/riichienv/log_replay.py`
//! (`_RED_NORM`, `_copies_of_string`, `_YAOCHU`).
//!
//! Order reused, implementation written fresh: suits are sequential
//! four-copy blocks with the red five FIRST ({16, 52, 88}); honors follow
//! E/S/W/N/haku/hatsu/chun from 108.

/// Fail-closed tile codec error with a named reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileError(pub String);

impl std::fmt::Display for TileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tile codec: {}", self.0)
    }
}

impl std::error::Error for TileError {}

/// Red-five physical ids (first copy of each suited five block).
pub const AKA_IDS: [u8; 3] = [16, 52, 88];

fn suit_base(suit: u8) -> Option<u8> {
    match suit {
        b'm' => Some(0),
        b'p' => Some(36),
        b's' => Some(72),
        _ => None,
    }
}

fn honor_base(letter: u8) -> Option<u8> {
    match letter {
        b'E' => Some(108),
        b'S' => Some(112),
        b'W' => Some(116),
        b'N' => Some(120),
        b'P' => Some(124),
        b'F' => Some(128),
        b'C' => Some(132),
        _ => None,
    }
}

/// Parse one MJAI tile string into its exact physical id (0..135).
///
/// The unsuffixed `"5x"` resolves to the SECOND copy (17/53/89); the red
/// five is `"5xr"` (or the `"0x"` alias) resolving to the FIRST copy
/// (16/52/88).
pub fn physical_of(mjai_tile: &str) -> Result<u8, TileError> {
    let err = || TileError(format!("invalid mjai tile {mjai_tile:?}"));
    match mjai_tile {
        "0m" => return Ok(16),
        "0p" => return Ok(52),
        "0s" => return Ok(88),
        _ => {}
    }
    let b = mjai_tile.as_bytes();
    if b.len() == 3 && b[2] == b'r' {
        if b[0] != b'5' {
            return Err(TileError(format!(
                "red suffix on non-five tile {mjai_tile:?}"
            )));
        }
        let base = suit_base(b[1]).ok_or_else(err)?;
        return Ok(base + 16);
    }
    if b.len() == 2 {
        if let Some(base) = suit_base(b[1]) {
            let number = (b[0] as char).to_digit(10).ok_or_else(err)?;
            if !(1..=9).contains(&number) {
                return Err(err());
            }
            if number == 5 {
                return Ok(base + 17);
            }
            return Ok(base + 4 * (number as u8 - 1));
        }
    }
    if b.len() == 1 {
        if let Some(base) = honor_base(b[0]) {
            return Ok(base);
        }
    }
    Err(err())
}

/// Render one physical id as its canonical MJAI string (red fives marked).
pub fn mjai_string_of(tile: u8) -> Result<String, TileError> {
    if tile > 135 {
        return Err(TileError(format!("physical tile out of range: {tile}")));
    }
    for (suffix, base) in [("m", 0u8), ("p", 36u8), ("s", 72u8)] {
        if (base..base + 36).contains(&tile) {
            let number = (tile - base) / 4 + 1;
            if number == 5 && tile % 4 == 0 {
                return Ok(format!("5{suffix}r"));
            }
            return Ok(format!("{number}{suffix}"));
        }
    }
    for (letter, base) in [
        ("E", 108u8),
        ("S", 112),
        ("W", 116),
        ("N", 120),
        ("P", 124),
        ("F", 128),
        ("C", 132),
    ] {
        if (base..base + 4).contains(&tile) {
            return Ok(letter.to_string());
        }
    }
    Err(TileError(format!("no mjai rendering for tile {tile}")))
}

/// Ordered physical copies for one MJAI string (red-aware).
///
/// Mirrors `_copies_of_string`: the red aliases collapse to the single aka
/// copy; a plain five excludes the red copy; every other string spans its
/// full four-copy block.
pub fn copies_of_string(pai: &str) -> Result<Vec<u8>, TileError> {
    if pai == "5mr" || pai == "0m" {
        return Ok(vec![16]);
    }
    if pai == "5pr" || pai == "0p" {
        return Ok(vec![52]);
    }
    if pai == "5sr" || pai == "0s" {
        return Ok(vec![88]);
    }
    let first = physical_of(pai)?;
    let base = (first / 4) * 4;
    if first == base + 1 && pai.as_bytes().first() == Some(&b'5') {
        return Ok(vec![base + 1, base + 2, base + 3]);
    }
    Ok(vec![base, base + 1, base + 2, base + 3])
}

/// Red-normalized MJAI string (`5pr` folds to `5p`).
pub fn norm_pai(pai: &str) -> &str {
    match pai {
        "5mr" | "0m" => "5m",
        "5pr" | "0p" => "5p",
        "5sr" | "0s" => "5s",
        other => other,
    }
}

/// Tile type index 0..34 (aka shares its five type).
pub fn tile_type(tile: u8) -> u8 {
    tile / 4
}

/// Whether a red-NORMALIZED kind string is terminal or honor.
pub fn is_yaochu_norm(normed: &str) -> bool {
    matches!(
        normed,
        "1m" | "9m" | "1p" | "9p" | "1s" | "9s" | "E" | "S" | "W" | "N" | "P" | "F" | "C"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn aka_ids_are_first_copies() {
        assert_eq!(physical_of("0m").unwrap(), 16);
        assert_eq!(physical_of("0p").unwrap(), 52);
        assert_eq!(physical_of("0s").unwrap(), 88);
        assert_eq!(physical_of("5mr").unwrap(), 16);
        assert_eq!(physical_of("5pr").unwrap(), 52);
        assert_eq!(physical_of("5sr").unwrap(), 88);
        // Unsuffixed five resolves to the second copy.
        assert_eq!(physical_of("5m").unwrap(), 17);
        assert_eq!(physical_of("5p").unwrap(), 53);
        assert_eq!(physical_of("5s").unwrap(), 89);
    }

    #[test]
    fn full_codec_round_trip() {
        // Every physical id renders and re-parses (aka renders red-marked).
        for tile in 0u8..=135u8 {
            let rendered = mjai_string_of(tile).unwrap();
            let back = physical_of(&rendered).unwrap();
            if AKA_IDS.contains(&tile) {
                assert_eq!(rendered.chars().nth(2), Some('r'));
                assert_eq!(back, tile);
            } else if [4u8, 13u8, 22u8].contains(&tile_type(tile)) {
                // Plain fives render unsuffixed and re-parse to the second
                // copy (base + 1).
                assert_eq!(back, (tile / 4) * 4 + 1, "id {tile} via {rendered}");
            } else {
                // Every other unsuffixed string resolves to its block base.
                assert_eq!(back, (tile / 4) * 4, "id {tile} via {rendered}");
            }
        }
    }

    #[test]
    fn copy_pools_partition_the_wall() {
        // Canonical string per copy: pools over every distinct rendering
        // cover all 136 ids exactly once (red strings own the aka copies).
        let mut seen = HashSet::new();
        let mut strings: Vec<String> = (0u8..=135u8)
            .map(|t| mjai_string_of(t).unwrap())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        strings.sort();
        for s in &strings {
            for copy in copies_of_string(s).unwrap() {
                assert!(seen.insert(copy), "copy {copy} shared by pools");
            }
        }
        assert_eq!(seen.len(), 136);
    }

    #[test]
    fn rejects_fail_closed() {
        assert!(physical_of("").is_err());
        assert!(physical_of("10m").is_err());
        assert!(physical_of("0x").is_err());
        assert!(physical_of("3mr").is_err());
        assert!(physical_of("X").is_err());
        assert!(mjai_string_of(136).is_err());
    }
    // --- Slice S1 pins: codec matrices vs tiles.py + _copies_of_string ---
    //
    // Reference literals (read-only):
    // - `src/hydra2/engines/riichienv/tiles.py`: `_RED_ALIASES`
    //   (`0m`->16, `0p`->52, `0s`->88), unsuffixed `5x` -> second copy
    //   (17/53/89), red five (`5xr`) -> first copy (16/52/88).
    // - `log_replay.py` + `single_pass.py` `_copies_of_string` (agree
    //   byte-for-byte; cross-checked): red aliases -> single aka copy,
    //   plain five -> three non-aka copies, else the full four-copy block.
    // - `log_replay.py` `_RED_NORM` / `_YAOCHU`.

    #[test]
    fn red_alias_matrix_matches_engine() {
        // Canonical renderings of the three aka copies.
        for (id, rendered) in [(16u8, "5mr"), (52, "5pr"), (88, "5sr")] {
            assert_eq!(mjai_string_of(id).unwrap(), rendered);
        }
        // Every red spelling resolves to the first copy of its block.
        for (alias, id) in [
            ("0m", 16u8),
            ("0p", 52),
            ("0s", 88),
            ("5mr", 16),
            ("5pr", 52),
            ("5sr", 88),
            ("5m", 17),
            ("5p", 53),
            ("5s", 89),
        ] {
            assert_eq!(physical_of(alias).unwrap(), id, "{alias}");
        }
        // Unsuffixed fives render plain and re-parse to the second copy.
        for (plain, second) in [("5m", 17u8), ("5p", 53), ("5s", 89)] {
            assert_eq!(mjai_string_of(second).unwrap(), plain);
            assert_eq!(mjai_string_of(second + 1).unwrap(), plain);
            assert_eq!(mjai_string_of(second + 2).unwrap(), plain);
        }
    }

    #[test]
    fn copies_matrix_matches_copies_of_string() {
        // Byte-exact pools from both oracle implementations.
        let cases: &[(&str, &[u8])] = &[
            ("5mr", &[16]),
            ("0m", &[16]),
            ("5pr", &[52]),
            ("0p", &[52]),
            ("5sr", &[88]),
            ("0s", &[88]),
            ("5m", &[17, 18, 19]),
            ("5p", &[53, 54, 55]),
            ("5s", &[89, 90, 91]),
            ("1m", &[0, 1, 2, 3]),
            ("9m", &[32, 33, 34, 35]),
            ("1p", &[36, 37, 38, 39]),
            ("9s", &[104, 105, 106, 107]),
            ("E", &[108, 109, 110, 111]),
            ("C", &[132, 133, 134, 135]),
            ("2s", &[76, 77, 78, 79]),
        ];
        for (pai, pool) in cases {
            assert_eq!(copies_of_string(pai).unwrap(), pool.to_vec(), "{pai}");
        }
        // Plain-five pools exclude the aka copy everywhere.
        for (plain, aka) in [("5m", 16u8), ("5p", 52), ("5s", 88)] {
            assert!(!copies_of_string(plain).unwrap().contains(&aka));
        }
        // Unknown strings fail closed instead of inventing a pool.
        assert!(copies_of_string("bogus").is_err());
    }

    #[test]
    fn norm_and_yaochu_match_oracle() {
        for (alias, normed) in [
            ("5mr", "5m"),
            ("0m", "5m"),
            ("5pr", "5p"),
            ("0p", "5p"),
            ("5sr", "5s"),
            ("0s", "5s"),
        ] {
            assert_eq!(norm_pai(alias), normed);
        }
        for plain in ["5m", "5p", "5s", "1m", "9s", "E", "2m"] {
            assert_eq!(norm_pai(plain), plain);
        }
        for kind in [
            "1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C",
        ] {
            assert!(is_yaochu_norm(kind), "{kind}");
        }
        for kind in ["2m", "5m", "8p", "3s", "5p", "5s", "4m"] {
            assert!(!is_yaochu_norm(kind), "{kind}");
        }
    }

    #[test]
    fn rejects_more_malformed() {
        // Red suffix on a non-five, overlong strings, and bare ranks.
        assert!(physical_of("1mr").is_err());
        assert!(physical_of("9pr").is_err());
        assert!(physical_of("5mrr").is_err());
        assert!(physical_of("5").is_err());
        assert!(physical_of("m").is_err());
        assert!(physical_of("0m ").is_err());
    }
}
