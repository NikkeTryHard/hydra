#![allow(
    clippy::useless_conversion,
    reason = "matches upstream parse and scoring conversions"
)]
use crate::errors::{RiichiError, RiichiResult};
use crate::types::{Meld, MeldType};
use std::iter::Peekable;
use std::str::Chars;

struct TileManager {
    used: [[bool; 4]; 34],
}

impl TileManager {
    fn new() -> Self {
        Self {
            used: [[false; 4]; 34],
        }
    }

    fn get_tile(&mut self, tile_34: usize, is_red: bool) -> Result<u8, String> {
        if tile_34 >= 34 {
            return Err(format!("Invalid tile ID: {}", tile_34));
        }

        let is_5 = tile_34 == 4 || tile_34 == 13 || tile_34 == 22;

        let search_indices: &[usize] = match (is_5, is_red) {
            (true, true) => &[0],
            (true, false) => &[1, 2, 3, 0],
            (false, _) => &[0, 1, 2, 3],
        };

        let target_idx = search_indices
            .iter()
            .find(|&&idx| !self.used[tile_34][idx])
            .copied()
            .ok_or_else(|| format!("No more copies of tile {}", tile_34))?;
        self.used[tile_34][target_idx] = true;
        Ok(((tile_34 * 4) + target_idx) as u8)
    }
}

/// Parse an MPSZ hand string into 136-format tile IDs and melds.
pub fn parse_hand_internal(text: &str) -> RiichiResult<(Vec<u8>, Vec<Meld>)> {
    let mut tm = TileManager::new();
    let mut tiles_136 = Vec::new();
    let mut melds = Vec::new();

    let mut chars = text.chars().peekable();
    let mut pending_digits: Vec<char> = Vec::new();

    while let Some(&c) = chars.peek() {
        if c == '(' {
            chars.next();
            let meld = parse_meld(&mut chars, &mut tm)?;
            melds.push(meld);
        } else if c.is_ascii_digit() {
            chars.next();
            pending_digits.push(c);
        } else if is_suit(c) {
            chars.next();
            let suit_offset = match c {
                'm' => 0,
                'p' => 9,
                's' => 18,
                'z' => 27,
                _ => unreachable!(),
            };
            for d in &pending_digits {
                let val = d.to_digit(10).ok_or_else(|| RiichiError::Parse {
                    input: text.to_string(),
                    message: format!("Invalid digit: {}", d),
                })? as usize;
                let (tile_34, is_red) = if val == 0 {
                    (suit_offset + 4, true)
                } else {
                    (suit_offset + val - 1, false)
                };
                let tid = tm
                    .get_tile(tile_34, is_red)
                    .map_err(|e| RiichiError::Parse {
                        input: text.to_string(),
                        message: e,
                    })?;
                tiles_136.push(tid);
            }
            pending_digits.clear();
        } else {
            chars.next();
        }
    }

    if !pending_digits.is_empty() {
        return Err(RiichiError::Parse {
            input: text.to_string(),
            message: "Pending digits without suit".to_string(),
        });
    }

    Ok((tiles_136, melds))
}

/// Parse an MPSZ hand string into 32-bit tile IDs and melds.
pub fn parse_hand(text: &str) -> RiichiResult<(Vec<u32>, Vec<Meld>)> {
    let (tiles, melds) = parse_hand_internal(text)?;
    Ok((tiles.iter().map(|&x| x as u32).collect(), melds))
}

/// Parse a single tile from an MPSZ string and return its 136-format ID.
pub fn parse_tile(text: &str) -> RiichiResult<u8> {
    let (tiles, melds) = parse_hand_internal(text)?;
    if !melds.is_empty() {
        return Err(RiichiError::Parse {
            input: text.to_string(),
            message: "parse_tile expects a single tile, but found meld syntax in input".to_string(),
        });
    }
    if tiles.is_empty() {
        return Err(RiichiError::Parse {
            input: text.to_string(),
            message: "No tile found in string".to_string(),
        });
    }
    if tiles.len() != 1 {
        return Err(RiichiError::Parse {
            input: text.to_string(),
            message: format!(
                "Expected exactly one tile, but found {} tiles in string",
                tiles.len()
            ),
        });
    }
    Ok(tiles[0])
}

fn is_suit(c: char) -> bool {
    matches!(c, 'm' | 'p' | 's' | 'z')
}

fn parse_meld(chars: &mut Peekable<Chars>, tm: &mut TileManager) -> RiichiResult<Meld> {
    let mut content = String::new();
    while let Some(&c) = chars.peek() {
        if c == ')' {
            chars.next();
            break;
        }
        content.push(c);
        chars.next();
    }

    let (prefix, rest) = if let Some(stripped) = content.strip_prefix('p') {
        ('p', stripped)
    } else if let Some(stripped) = content.strip_prefix('k') {
        ('k', stripped)
    } else if let Some(stripped) = content.strip_prefix('s') {
        ('s', stripped)
    } else {
        (' ', content.as_str())
    };

    let mut digits = Vec::new();
    let remaining_str = rest;
    let mut suit_char = ' ';

    let mut idx = 0;
    let chars_vec: Vec<char> = remaining_str.chars().collect();
    while idx < chars_vec.len() && chars_vec[idx].is_ascii_digit() {
        digits.push(chars_vec[idx]);
        idx += 1;
    }

    if idx < chars_vec.len() {
        suit_char = chars_vec[idx];
        idx += 1;
    }

    let _call_idx = if idx < chars_vec.len() {
        let c = chars_vec[idx];
        if c.is_ascii_digit() {
            c.to_digit(10).ok_or_else(|| RiichiError::Parse {
                input: content.clone(),
                message: format!("Invalid digit: {}", c),
            })?
        } else {
            0
        }
    } else {
        0
    };

    let suit_offset = match suit_char {
        'm' => 0,
        'p' => 9,
        's' => 18,
        'z' => 27,
        _ => {
            return Err(RiichiError::Parse {
                input: content.clone(),
                message: format!("Invalid suit in meld: {}", suit_char),
            })
        }
    };

    let mut tiles_136 = Vec::new();

    if prefix == ' ' {
        // Chi
        if digits.len() != 3 {
            return Err(RiichiError::Parse {
                input: content.clone(),
                message: "Chi meld requires 3 digits".to_string(),
            });
        }
        for d in digits {
            let val = d.to_digit(10).ok_or_else(|| RiichiError::Parse {
                input: content.clone(),
                message: format!("Invalid digit: {}", d),
            })? as usize;
            let (tile_34, is_red) = if val == 0 {
                (suit_offset + 4, true)
            } else {
                (suit_offset + val - 1, false)
            };
            tiles_136.push(
                tm.get_tile(tile_34, is_red)
                    .map_err(|e| RiichiError::Parse {
                        input: content.clone(),
                        message: e,
                    })?,
            );
        }
        tiles_136.sort();
        Ok(Meld::new(MeldType::Chi, &tiles_136, true, -1, None))
    } else {
        let val_d = digits[0].to_digit(10).ok_or_else(|| RiichiError::Parse {
            input: content.clone(),
            message: format!("Invalid digit: {}", digits[0]),
        })? as usize;
        let (base_34, is_red_indicated) = if val_d == 0 {
            (suit_offset + 4, true)
        } else {
            (suit_offset + val_d - 1, false)
        };

        let count = match prefix {
            'p' => 3,
            'k' | 's' => 4,
            _ => 3,
        };

        let mut got_red = false;
        if is_red_indicated {
            tiles_136.push(tm.get_tile(base_34, true).map_err(|e| RiichiError::Parse {
                input: content.clone(),
                message: e,
            })?);
            got_red = true;
        }

        while tiles_136.len() < count {
            if let Ok(t) = tm.get_tile(base_34, false) {
                tiles_136.push(t);
            } else if !got_red {
                if let Ok(t) = tm.get_tile(base_34, true) {
                    tiles_136.push(t);
                    got_red = true;
                } else {
                    return Err(RiichiError::Parse {
                        input: content.clone(),
                        message: format!("Not enough tiles for meld of {}", base_34),
                    });
                }
            } else {
                return Err(RiichiError::Parse {
                    input: content.clone(),
                    message: format!("Not enough tiles for meld of {}", base_34),
                });
            }
        }

        tiles_136.sort();

        let mtype = match prefix {
            'p' => MeldType::Pon,
            'k' => {
                if _call_idx == 0 {
                    MeldType::Ankan
                } else {
                    MeldType::Daiminkan
                }
            }
            's' => MeldType::Kakan,
            _ => unreachable!(),
        };

        let opened = mtype != MeldType::Ankan;

        Ok(Meld::new(mtype, &tiles_136, opened, -1, None))
    }
}

/// Convert a 136-format tile ID to its MJAI string representation.
pub fn tid_to_mjai(tid: u8) -> String {
    // Check Red 5s
    if tid == 16 {
        return "5mr".to_string();
    }
    if tid == 52 {
        return "5pr".to_string();
    }
    if tid == 88 {
        return "5sr".to_string();
    }

    let kind = tid / 36;
    if kind < 3 {
        let suit_char = match kind {
            0 => "m",
            1 => "p",
            2 => "s",
            _ => unreachable!(),
        };
        let offset = tid % 36;
        let num = offset / 4 + 1;
        format!("{}{}", num, suit_char)
    } else {
        let offset = tid - 108;
        let num = offset / 4 + 1;
        let honors = ["E", "S", "W", "N", "P", "F", "C"];
        if (1..=7).contains(&num) {
            honors[num as usize - 1].to_string()
        } else {
            format!("{}z", num)
        }
    }
}

/// Convert an MJAI tile string to its 136-format tile ID.
pub fn mjai_to_tid(mjai: &str) -> Option<u8> {
    // Honors
    let honors = ["E", "S", "W", "N", "P", "F", "C"];
    if let Some(pos) = honors.iter().position(|&h| h == mjai) {
        return Some(108 + (pos as u8) * 4);
    }

    // Red 5s
    if mjai == "5mr" {
        return Some(16);
    }
    if mjai == "5pr" {
        return Some(52);
    }
    if mjai == "5sr" {
        return Some(88);
    }

    // MPS
    if mjai.len() < 2 {
        return None;
    }
    let num_char = mjai.chars().next()?;
    let suit_char = mjai.chars().nth(1)?;
    let num = num_char.to_digit(10)? as u8;
    if num == 0 {
        let suit_idx = match suit_char {
            'm' => 0,
            'p' => 1,
            's' => 2,
            _ => return None,
        };
        return Some(suit_idx * 36 + 16);
    }
    if !(1..=9).contains(&num) {
        return None;
    }
    let suit_idx = match suit_char {
        'm' => 0,
        'p' => 1,
        's' => 2,
        'z' => {
            return Some(108 + (num - 1) * 4);
        }
        _ => return None,
    };

    let base = suit_idx * 36 + (num - 1) * 4;
    if num == 5 {
        Some(base + 1)
    } else {
        Some(base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MeldType;

    #[test]
    fn parse_tile_supports_basic_tiles_and_red_fives() {
        assert_eq!(parse_tile("1m").expect("1m should parse"), 0);
        assert_eq!(parse_tile("5m").expect("5m should parse"), 17);
        assert_eq!(parse_tile("0m").expect("red 5m should parse"), 16);
        assert_eq!(parse_tile("7z").expect("honor should parse"), 132);
    }

    #[test]
    fn parse_tile_rejects_empty_multi_tile_and_meld_inputs() {
        assert!(parse_tile("").is_err());
        assert!(parse_tile("12m").is_err());
        assert!(parse_tile("(p5m1)").is_err());
    }

    #[test]
    fn parse_hand_internal_parses_plain_tiles_and_chi_meld() {
        let (tiles, melds) = parse_hand_internal("123m(p5m1)").expect("hand should parse");

        assert_eq!(tiles.len(), 3);
        assert_eq!(melds.len(), 1);
        assert_eq!(melds[0].meld_type, MeldType::Pon);
        assert_eq!(melds[0].tile_count, 3);
        assert!(melds[0].opened);
    }

    #[test]
    fn parse_hand_internal_rejects_pending_digits_and_bad_meld_suit() {
        let err = parse_hand_internal("123").expect_err("pending digits should fail");
        match err {
            crate::errors::RiichiError::Parse { message, .. } => {
                assert!(message.contains("Pending digits without suit"));
            }
            other => panic!("unexpected error: {other:?}"),
        }

        let err = parse_hand_internal("(123x)").expect_err("invalid meld suit should fail");
        match err {
            crate::errors::RiichiError::Parse { message, .. } => {
                assert!(message.contains("Invalid suit in meld"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn tid_to_mjai_and_mjai_to_tid_cover_reds_honors_and_regular_tiles() {
        assert_eq!(tid_to_mjai(16), "5mr");
        assert_eq!(tid_to_mjai(52), "5pr");
        assert_eq!(tid_to_mjai(88), "5sr");
        assert_eq!(tid_to_mjai(0), "1m");
        assert_eq!(tid_to_mjai(108), "E");

        assert_eq!(mjai_to_tid("5mr"), Some(16));
        assert_eq!(mjai_to_tid("5pr"), Some(52));
        assert_eq!(mjai_to_tid("5sr"), Some(88));
        assert_eq!(mjai_to_tid("1m"), Some(0));
        assert_eq!(mjai_to_tid("5m"), Some(17));
        assert_eq!(mjai_to_tid("E"), Some(108));
        assert_eq!(mjai_to_tid("0x"), None);
        assert_eq!(mjai_to_tid(""), None);
    }

    #[test]
    fn parse_hand_returns_u32_tiles_matching_internal_parser() {
        let (tiles_u8, melds_internal) = parse_hand_internal("406p").expect("internal parse");
        let (tiles_u32, melds) = parse_hand("406p").expect("public parse");

        assert_eq!(
            tiles_u32,
            tiles_u8.iter().map(|&t| t as u32).collect::<Vec<_>>()
        );
        assert_eq!(melds.len(), melds_internal.len());
    }

    #[test]
    fn parse_hand_internal_supports_ankan_and_kakan_meld_kinds() {
        let (_, ankan_melds) = parse_hand_internal("(k5m)").expect("ankan should parse");
        assert_eq!(ankan_melds.len(), 1);
        assert_eq!(ankan_melds[0].meld_type, MeldType::Ankan);
        assert!(!ankan_melds[0].opened);
        assert_eq!(ankan_melds[0].tile_count, 4);

        let (_, kakan_melds) = parse_hand_internal("(s5p)").expect("kakan should parse");
        assert_eq!(kakan_melds.len(), 1);
        assert_eq!(kakan_melds[0].meld_type, MeldType::Kakan);
        assert!(kakan_melds[0].opened);
        assert_eq!(kakan_melds[0].tile_count, 4);
    }

    #[test]
    fn parse_hand_internal_rejects_bad_chi_and_tile_overflow() {
        let err = parse_hand_internal("(1234m)").expect_err("chi meld requires exactly 3 digits");
        match err {
            crate::errors::RiichiError::Parse { message, .. } => {
                assert!(message.contains("Chi meld requires 3 digits"));
            }
            other => panic!("unexpected error: {other:?}"),
        }

        let err = parse_hand_internal("55555m").expect_err("fifth copy of 5m should fail");
        match err {
            crate::errors::RiichiError::Parse { message, .. } => {
                assert!(message.contains("No more copies of tile 4"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn parse_hand_internal_rejects_melds_without_enough_copies() {
        let err = parse_hand_internal("(k5m)5m").expect_err("no fifth 5m copy should exist");
        match err {
            crate::errors::RiichiError::Parse { message, .. } => {
                assert!(
                    message.contains("No more copies of tile 4")
                        || message.contains("Not enough tiles for meld of 4")
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
