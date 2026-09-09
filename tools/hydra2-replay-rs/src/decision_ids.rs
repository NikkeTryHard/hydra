//! Positional decision/round ids: `{game}:d{seq:04}` / `{game}:h{idx:02}`.
//!
//! Reference (read-only): `src/hydra2/engines/riichienv/log_replay.py`
//! `_capture_row` (`decision_id = f"{game_id}:d{seq:04}"`, `round_id =
//! f"{game_id}:h{round_idx:02}"`). Ids stay positional over every decision
//! so seat-filtered replays still join privileged labels, and the frozen
//! row-hash fixture (`scripts/freeze_row_hashes.py`) keys on `decision_id`.

/// Format one decision id: positional sequence over every decision in the game.
///
/// Minimum width 4, zero-padded (`:d0000`); wider sequences keep all digits,
/// exactly like the oracle's `f"{game}:d{seq:04}"`.
pub fn decision_id(game_id: &str, seq: u32) -> String {
    format!("{game_id}:d{seq:04}")
}

/// Format one round id: two-digit round index within the game.
///
/// Minimum width 2, zero-padded (`:h00`), exactly like the oracle's
/// `f"{game}:h{round_idx:02}"`.
pub fn round_id(game_id: &str, round_idx: u32) -> String {
    format!("{game_id}:h{round_idx:02}")
}

/// Split `...:dNNNN` back into `(game_id, seq)`; `None` when malformed.
///
/// Accepts the minimum-width-4 form and wider sequences; rejects a missing
/// `:d` tag, an empty game stem, or non-digit sequences.
pub fn parse_decision_id(id: &str) -> Option<(&str, u32)> {
    let (game, seq) = id.rsplit_once(':')?;
    let digits = seq.strip_prefix('d')?;
    if game.is_empty() || digits.len() < 4 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u32>().ok().map(|n| (game, n))
}

/// Split `...:hNN` back into `(game_id, round_idx)`; `None` when malformed.
///
/// Accepts the minimum-width-2 form and wider indexes; rejects a missing
/// `:h` tag, an empty game stem, or non-digit indexes.
pub fn parse_round_id(id: &str) -> Option<(&str, u32)> {
    let (game, idx) = id.rsplit_once(':')?;
    let digits = idx.strip_prefix('h')?;
    if game.is_empty() || digits.len() < 2 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u32>().ok().map(|n| (game, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_format_matches_oracle() {
        assert_eq!(decision_id("game-23efeea403a7", 0), "game-23efeea403a7:d0000");
        assert_eq!(decision_id("g", 42), "g:d0042");
        // Minimum width only: wider sequences keep every digit.
        assert_eq!(decision_id("g", 12345), "g:d12345");
    }

    #[test]
    fn round_format_matches_oracle() {
        assert_eq!(round_id("game-23efeea403a7", 0), "game-23efeea403a7:h00");
        assert_eq!(round_id("g", 9), "g:h09");
    }

    #[test]
    fn parse_roundtrips() {
        let (game, seq) = parse_decision_id("game-23efeea403a7:d0042").expect("parse");
        assert_eq!((game, seq), ("game-23efeea403a7", 42));
        let (game, idx) = parse_round_id("game-23efeea403a7:h09").expect("parse");
        assert_eq!((game, idx), ("game-23efeea403a7", 9));
    }

    #[test]
    fn parse_rejects_malformed() {
        assert!(parse_decision_id("game-23efeea403a7").is_none());
        assert!(parse_decision_id("game-23efeea403a7:d042").is_none());
        assert!(parse_decision_id("game-23efeea403a7:d00ab").is_none());
        assert!(parse_decision_id("game-23efeea403a7:h00").is_none());
        assert!(parse_decision_id(":d0000").is_none());
        assert!(parse_round_id("game-23efeea403a7").is_none());
        assert!(parse_round_id("game-23efeea403a7:h0").is_none());
        assert!(parse_round_id("game-23efeea403a7:d0000").is_none());
        assert!(parse_round_id(":h00").is_none());
    }

    #[test]
    fn sequences_stay_positional() {
        // Join discipline: decision ids strictly increase with the walk
        // sequence, so the frozen-hash fixture keys stay total per game.
        let ids: Vec<String> = (0..500).map(|s| decision_id("g", s)).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }
    // --- Slice S1 pins: format widths + wider acceptance vs the oracle ---
    //
    // Reference (read-only): `log_replay.py`/`single_pass.py`/`replay_expand.py`
    // `f"{game}:d{seq:04d}"` / `f"{game}:h{idx:02d}"` — minimum widths 4/2,
    // wider values keep every digit.

    #[test]
    fn vectors_match_python_fstrings() {
        assert_eq!(decision_id("g", 0), "g:d0000");
        assert_eq!(decision_id("g", 42), "g:d0042");
        assert_eq!(decision_id("g", 9999), "g:d9999");
        assert_eq!(decision_id("g", 10000), "g:d10000");
        assert_eq!(decision_id("g", 12000), "g:d12000");
        assert_eq!(decision_id("g", 12345), "g:d12345");
        assert_eq!(round_id("g", 0), "g:h00");
        assert_eq!(round_id("g", 9), "g:h09");
        assert_eq!(round_id("g", 99), "g:h99");
        assert_eq!(round_id("g", 100), "g:h100");
        assert_eq!(round_id("g", 120), "g:h120");
    }

    #[test]
    fn parse_accepts_wide_ids() {
        for seq in [0u32, 1, 42, 9999, 10000, 12000, 12345] {
            let id = decision_id("game-23efeea403a7", seq);
            assert_eq!(parse_decision_id(&id), Some(("game-23efeea403a7", seq)), "{id}");
        }
        for idx in [0u32, 1, 9, 99, 100, 120] {
            let id = round_id("game-23efeea403a7", idx);
            assert_eq!(parse_round_id(&id), Some(("game-23efeea403a7", idx)), "{id}");
        }
        // Game stems may themselves contain colons; the tag split is rightmost.
        assert_eq!(parse_decision_id("a:b:d0007"), Some(("a:b", 7)));
        assert_eq!(parse_round_id("a:b:h03"), Some(("a:b", 3)));
    }

    #[test]
    fn parse_rejects_more_malformed() {
        for bad in [
            "g:d",
            "g:d1",
            "g:d12",
            "g:d123",
            "g:D0000",
            "g:dd0000",
            "g:d 000",
            "g:d99999999999",
        ] {
            assert!(parse_decision_id(bad).is_none(), "{bad}");
        }
        for bad in ["g:h", "g:h1", "g:H00", "g:hh00", "g:h 0", "g:h99999999999"] {
            assert!(parse_round_id(bad).is_none(), "{bad}");
        }
        // Cross-tagged ids never parse as the other kind.
        assert!(parse_decision_id("g:h09").is_none());
        assert!(parse_round_id("g:d0042").is_none());
    }
}
