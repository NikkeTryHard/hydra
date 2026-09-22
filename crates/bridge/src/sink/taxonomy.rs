use std::path::Path;

use hydra_feed::gate::{self, QuarantineStub};
use hydra_feed::ledger;

// ---------------------------------------------------------------------------
// Reason renderer + closed sink taxonomy.
// ---------------------------------------------------------------------------

/// Cold renderer for a quarantine stub reason (never hot).
///
/// Gate buckets (`0..=8`) render via `gate::reason_name`; walk codes
/// (`WALK_*`, `10..`) via `ledger::walk_reason_name`. The split mirrors the
/// two constant blocks; unknown bytes stay `"other"` on both sides.
pub fn quarantine_reason_name(reason: u8) -> &'static str {
    if reason < 10 {
        gate::reason_name(reason)
    } else {
        ledger::walk_reason_name(reason)
    }
}

/// Sink bucket: framing (S1 payload/boundary family).
pub const SINK_FRAMING: &str = "framing";
/// Sink bucket: vocabulary (unknown kinds / unmapped reasons / ids).
pub const SINK_VOCAB: &str = "vocab";
/// Sink bucket: tile conservation (ledger/copy/claim/offer family, incl the
/// post-claim reject — reused code, no taxonomy churn).
pub const SINK_CONSERVATION: &str = "conservation";
/// Sink bucket: hora mismatch (shape / turn-order / desync / kyushu family).
pub const SINK_HORA_MISMATCH: &str = "hora-mismatch";
/// Sink bucket: wall permutation / exhaustion family.
pub const SINK_WALL_PERM: &str = "wall-perm";
/// Sink bucket: history capacity (S4 T-bucket overflow family).
pub const SINK_HISTORY_CAP: &str = "history-cap";

/// Closed sink vocabulary: every emitted bucket is a member; `"other"` is
/// the canary for unassigned bytes (same fail-loud rule as the renderer).
pub const SINK_VOCABULARY: [&str; 6] = [
    SINK_FRAMING,
    SINK_VOCAB,
    SINK_CONSERVATION,
    SINK_HORA_MISMATCH,
    SINK_WALL_PERM,
    SINK_HISTORY_CAP,
];

/// Coarsen a verdict byte to its closed sink bucket.
///
/// Mapping (every `REASON_*` / `WALK_*` const is covered):
/// - framing ← payload shape / blank line / boundary (1, 3, 4);
/// - vocab ← unknown-event (2, 17), unmapped ryukyoku (15), unresolved id (20);
/// - conservation ← bare-dora (6, 18), double-ron (7, 19), tile ledger (11),
///   claim-no-offer (12);
/// - hora-mismatch ← hora shape (8), turn-order (10), kyushu-ambiguous (14),
///   engine-desync (16), past-terminal (21: hora with trailing row/end
///   events and no closing end_kyoku — the wall oracle goes terminal);
/// - wall-perm ← wall-bearing (5), draw-past-wall (13).
///   `history-cap` is reserved for S4 T-bucket-overflow quarantines (today that
///   overflow surfaces as retryable `BufferTooSmall`, never a quarantine, so
///   the bucket pins vocabulary-closed with zero current emitters).
///   Unassigned bytes (0 `ok`, 9, 22+) → `"other"` (fail-loud canary).
pub const fn sink_bucket(reason: u8) -> &'static str {
    match reason {
        gate::REASON_FRAMING | gate::REASON_BLANK_LINE | gate::REASON_BOUNDARY => SINK_FRAMING,
        gate::REASON_UNKNOWN_KIND
        | ledger::WALK_UNMAPPED_RYUKYOKU
        | ledger::WALK_UNKNOWN_EVENT
        | ledger::WALK_ACTION_UNRESOLVED => SINK_VOCAB,
        gate::REASON_BARE_DORA
        | gate::REASON_DOUBLE_RON
        | ledger::WALK_TILE_CONSERVATION
        | ledger::WALK_CLAIM_NO_OFFER
        | ledger::WALK_BARE_DORA
        | ledger::WALK_DOUBLE_RON => SINK_CONSERVATION,
        gate::REASON_HORA_SHAPE
        | ledger::WALK_TURN_ORDER
        | ledger::WALK_KYUSHU_AMBIGUOUS
        | ledger::WALK_ENGINE_DESYNC
        | ledger::WALK_PAST_TERMINAL => SINK_HORA_MISMATCH,
        gate::REASON_WALL_PERM | ledger::WALK_DRAW_PAST_WALL => SINK_WALL_PERM,
        _ => "other",
    }
}

// ---------------------------------------------------------------------------
// Lineage file: {identity, event_idx, reason, obs_hash u64} + game join.
// ---------------------------------------------------------------------------

/// One quarantined game's cold lineage: file-stem identity, offending event,
/// verdict byte, numeric game key. [`reason_code`](Self::reason_code) renders
/// the G5 histogram string; [`bucket`](Self::bucket) the sink taxonomy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuarantineLineage {
    /// Input file stem (string ids live cold-side only).
    pub game_id: String,
    /// Offending event index (`u16::MAX` saturates oversized games).
    pub event_idx: u16,
    /// Verdict byte (`REASON_*` / `WALK_*`).
    pub reason: u8,
    /// Numeric game key (cold join key).
    pub obs_hash: u64,
}

impl QuarantineLineage {
    /// Build from a hot stub plus the bridge's cold identity/key.
    pub fn from_parts(game_id: &str, stub: QuarantineStub, obs_hash: u64) -> Self {
        Self {
            game_id: game_id.to_string(),
            event_idx: stub.event_idx,
            reason: stub.reason,
            obs_hash,
        }
    }

    /// G5 histogram string (oracle vocabulary).
    pub fn reason_code(&self) -> &'static str {
        quarantine_reason_name(self.reason)
    }

    /// Closed sink bucket.
    pub fn bucket(&self) -> &'static str {
        sink_bucket(self.reason)
    }
}

/// Lineage file failure (closed: IO vs parse).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageError {
    /// File read/write failure (carries the OS message, never the payload).
    Io(String),
    /// Malformed lineage line (carries `line N: <cause>`, never panics).
    Parse(String),
}

impl std::fmt::Display for LineageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "quarantine lineage io: {msg}"),
            Self::Parse(msg) => write!(f, "quarantine lineage parse: {msg}"),
        }
    }
}

impl std::error::Error for LineageError {}

fn escape_game_id(out: &mut String, id: &str) {
    for c in id.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
}

/// Encode one record as a JSON line (hand-rolled: this crate carries no
/// serde; the writer emits exactly what [`decode_lineage_line`] accepts).
fn encode_lineage_line(r: &QuarantineLineage) -> String {
    let mut out = String::from("{\"game_id\":\"");
    escape_game_id(&mut out, &r.game_id);
    out.push_str(&format!(
        "\",\"event_idx\":{},\"reason\":{},\"obs_hash\":{}}}",
        r.event_idx, r.reason, r.obs_hash
    ));
    out
}

fn parse_hex4(s: &[u8], pos: usize) -> Option<(u32, usize)> {
    if pos + 4 > s.len() {
        return None;
    }
    let mut v: u32 = 0;
    let mut k = 0usize;
    while k < 4 {
        let d = s[pos + k];
        let digit = if d.is_ascii_digit() {
            (d - b'0') as u32
        } else if (b'a'..=b'f').contains(&d) {
            (d - b'a' + 10) as u32
        } else if (b'A'..=b'F').contains(&d) {
            (d - b'A' + 10) as u32
        } else {
            return None;
        };
        v = v * 16 + digit;
        k += 1;
    }
    Some((v, pos + 4))
}

fn parse_u64(s: &[u8], mut pos: usize) -> Option<(u64, usize)> {
    let start = pos;
    let mut v: u64 = 0;
    while pos < s.len() && s[pos].is_ascii_digit() {
        let d = (s[pos] - b'0') as u64;
        v = v.checked_mul(10)?.checked_add(d)?;
        pos += 1;
    }
    if pos == start {
        return None;
    }
    Some((v, pos))
}

/// Strict decode of one [`encode_lineage_line`] line (1-based `line_no` for
/// error context).
fn decode_lineage_line(line: &str, line_no: usize) -> Result<QuarantineLineage, LineageError> {
    let fail = |cause: &str| LineageError::Parse(format!("line {line_no}: {cause}"));
    let s = line.as_bytes();
    let mut pos = 0usize;
    let expect = |pos: &mut usize, lit: &[u8], what: &str| -> Result<(), LineageError> {
        if s.len() >= *pos + lit.len() && &s[*pos..*pos + lit.len()] == lit {
            *pos += lit.len();
            Ok(())
        } else {
            Err(fail(what))
        }
    };
    expect(&mut pos, b"{\"game_id\":\"", "missing game_id header")?;
    let mut id_bytes: Vec<u8> = Vec::new();
    loop {
        let c = *s.get(pos).ok_or_else(|| fail("truncated game_id"))?;
        if c == b'"' {
            pos += 1;
            break;
        }
        if c != b'\\' {
            if c < 0x20 {
                return Err(fail("raw control in game_id"));
            }
            id_bytes.push(c);
            pos += 1;
            continue;
        }
        let e = *s.get(pos + 1).ok_or_else(|| fail("truncated escape"))?;
        match e {
            b'"' => {
                id_bytes.push(b'"');
                pos += 2;
            }
            b'\\' => {
                id_bytes.push(b'\\');
                pos += 2;
            }
            b'n' => {
                id_bytes.push(b'\n');
                pos += 2;
            }
            b't' => {
                id_bytes.push(b'\t');
                pos += 2;
            }
            b'u' => {
                let (v, next) = parse_hex4(s, pos + 2).ok_or_else(|| fail("bad \\u escape"))?;
                let ch = char::from_u32(v).ok_or_else(|| fail("bad \\u scalar"))?;
                let mut buf = [0u8; 4];
                id_bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                pos = next;
            }
            _ => return Err(fail("unknown escape")),
        }
    }
    let game_id = String::from_utf8(id_bytes).map_err(|_| fail("game_id not utf-8"))?;
    expect(&mut pos, b",\"event_idx\":", "missing event_idx")?;
    let (event_idx, next) = parse_u64(s, pos).ok_or_else(|| fail("bad event_idx"))?;
    if event_idx > u16::MAX as u64 {
        return Err(fail("event_idx overflow"));
    }
    pos = next;
    expect(&mut pos, b",\"reason\":", "missing reason")?;
    let (reason, next) = parse_u64(s, pos).ok_or_else(|| fail("bad reason"))?;
    if reason > u8::MAX as u64 {
        return Err(fail("reason overflow"));
    }
    pos = next;
    expect(&mut pos, b",\"obs_hash\":", "missing obs_hash")?;
    let (obs_hash, next) = parse_u64(s, pos).ok_or_else(|| fail("bad obs_hash"))?;
    pos = next;
    expect(&mut pos, b"}", "missing close")?;
    if pos != s.len() {
        return Err(fail("trailing bytes"));
    }
    // proof: both range-checked above (`> u16::MAX`/`> u8::MAX` fail), fit.
    #[allow(clippy::cast_possible_truncation)]
    let event_u: u16 = event_idx as u16;
    #[allow(clippy::cast_possible_truncation)]
    let reason_u: u8 = reason as u8;
    Ok(QuarantineLineage {
        game_id,
        event_idx: event_u,
        reason: reason_u,
        obs_hash,
    })
}

/// Write quarantine lineage, one JSON record per line.
pub fn write_lineage_file(path: &Path, records: &[QuarantineLineage]) -> Result<(), LineageError> {
    let mut out = String::new();
    for r in records {
        out.push_str(&encode_lineage_line(r));
        out.push('\n');
    }
    std::fs::write(path, out).map_err(|e| LineageError::Io(e.to_string()))
}

/// Read back a [`write_lineage_file`] file (empty lines skipped; anything
/// else malformed fails closed).
pub fn read_lineage_file(path: &Path) -> Result<Vec<QuarantineLineage>, LineageError> {
    let text = std::fs::read_to_string(path).map_err(|e| LineageError::Io(e.to_string()))?;
    let mut out = Vec::new();
    for (k, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        out.push(decode_lineage_line(line, k + 1)?);
    }
    Ok(out)
}

/// Game lineage join: match lineage records back to consumed games by
/// `(game_id, obs_hash)`, in game order (`None` = game walked clean).
/// Duplicate record keys keep the first (stems are unique per stream).
pub fn join_lineage<'a>(
    games: &'a [(String, u64)],
    records: &'a [QuarantineLineage],
) -> Vec<(&'a (String, u64), Option<&'a QuarantineLineage>)> {
    games
        .iter()
        .map(|g| {
            let hit = records
                .iter()
                .find(|r| r.game_id == g.0 && r.obs_hash == g.1);
            (g, hit)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Accounting assert path (G5 identity).
// ---------------------------------------------------------------------------

/// Accounting mismatch: `ok + quarantined + staged != consumed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccountingMismatch {
    /// Games committed to caller buffers.
    pub ok: u64,
    /// Games quarantined (zero rows each).
    pub quarantined: u64,
    /// Games staged but not yet drained.
    pub staged: u64,
    /// Input files consumed (cursor).
    pub consumed: u64,
}

impl std::fmt::Display for AccountingMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "quarantine accounting broke: ok={} quarantined={} staged={} consumed={}",
            self.ok, self.quarantined, self.staged, self.consumed
        )
    }
}

impl std::error::Error for AccountingMismatch {}

/// Assert the G5 identity `ok + quarantined + staged == consumed`
/// (overflow-safe: any overflow is a mismatch, never a wrap). At full drain
/// `staged == 0` and this is `games_quarantined + games_ok == games`.
pub fn check_accounting(
    ok: u64,
    quarantined: u64,
    staged: u64,
    consumed: u64,
) -> Result<(), AccountingMismatch> {
    let balanced = ok
        .checked_add(quarantined)
        .and_then(|s| s.checked_add(staged))
        == Some(consumed);
    if balanced {
        Ok(())
    } else {
        Err(AccountingMismatch {
            ok,
            quarantined,
            staged,
            consumed,
        })
    }
}

#[cfg(test)]
mod taxonomy_tests {
    use super::*;

    fn stub(reason: u8) -> QuarantineStub {
        QuarantineStub::new(7, 5, reason)
    }

    #[test]
    fn reason_renderer_covers_both_code_spaces() {
        // Gate buckets 0..=8.
        let gate = [
            (0, "ok"),
            (1, "framing"),
            (2, "unknown-event"),
            (3, "framing"),
            (4, "framing"),
            (5, "wall-bearing"),
            (6, "bare-dora"),
            (7, "double-ron"),
            (8, "turn-order"),
        ];
        for (code, name) in gate {
            assert_eq!(quarantine_reason_name(code), name, "gate {code}");
        }
        // Walk buckets 10..=21.
        let walk = [
            (10, "turn-order"),
            (11, "tile-conservation"),
            (12, "claim-no-offer"),
            (13, "draw-past-wall"),
            (14, "kyushu-ambiguous"),
            (15, "unmapped-ryukyoku-reason"),
            (16, "engine-desync"),
            (17, "unknown-event"),
            (18, "bare-dora"),
            (19, "double-ron"),
            (20, "action-id-unresolved"),
            (21, "past-terminal"),
        ];
        for (code, name) in walk {
            assert_eq!(quarantine_reason_name(code), name, "walk {code}");
        }
        // Unassigned bytes are the fail-loud canary, never silent.
        for code in [9u8, 22, 100, 255] {
            assert_eq!(quarantine_reason_name(code), "other", "canary {code}");
        }
    }

    #[test]
    fn sink_buckets_are_closed_over_assigned_codes() {
        // Every assigned code lands in the 6-bucket vocabulary (never other).
        let assigned = [
            1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        ];
        for code in assigned {
            let bucket = sink_bucket(code);
            assert!(
                SINK_VOCABULARY.contains(&bucket),
                "code {code} escaped to {bucket}"
            );
            assert_ne!(bucket, "other", "code {code} hit the canary");
        }
        // Spot pins per family.
        assert_eq!(sink_bucket(1), "framing");
        assert_eq!(sink_bucket(4), "framing");
        assert_eq!(sink_bucket(2), "vocab");
        assert_eq!(sink_bucket(15), "vocab");
        assert_eq!(sink_bucket(20), "vocab");
        assert_eq!(sink_bucket(6), "conservation");
        assert_eq!(sink_bucket(11), "conservation");
        assert_eq!(sink_bucket(12), "conservation");
        assert_eq!(sink_bucket(8), "hora-mismatch");
        assert_eq!(sink_bucket(10), "hora-mismatch");
        assert_eq!(sink_bucket(16), "hora-mismatch");
        assert_eq!(sink_bucket(21), "hora-mismatch");
        assert_eq!(sink_bucket(5), "wall-perm");
        assert_eq!(sink_bucket(13), "wall-perm");
        // history-cap is reserved (zero current emitters); canary intact.
        assert_eq!(SINK_VOCABULARY.len(), 6);
        assert_eq!(sink_bucket(0), "other");
        assert_eq!(sink_bucket(9), "other");
        assert_eq!(sink_bucket(255), "other");
    }

    #[test]
    fn lineage_record_renders_both_vocabularies() {
        let r = QuarantineLineage::from_parts("q-tile-conservation", stub(11), 0x9E37);
        assert_eq!(r.reason_code(), "tile-conservation");
        assert_eq!(r.bucket(), "conservation");
        assert_eq!(r.event_idx, 5);
        assert_eq!(r.obs_hash, 0x9E37);
    }

    #[test]
    fn lineage_line_round_trips_with_escapes() {
        let records = [
            QuarantineLineage::from_parts("plain-stem", stub(4), 1),
            QuarantineLineage::from_parts("quote\"back\\slash\ttab", stub(11), u64::MAX),
            QuarantineLineage::from_parts("", stub(2), 0),
        ];
        for r in &records {
            let line = encode_lineage_line(r);
            let back = decode_lineage_line(&line, 1).expect("round trip");
            assert_eq!(&back, r);
        }
        // Malformed lines fail closed, never panic.
        for bad in [
            "",
            "{}",
            "{\"game_id\":\"x\"}",
            "{\"game_id\":\"x\",\"event_idx\":70000,\"reason\":1,\"obs_hash\":0}",
            "{\"game_id\":\"x\",\"event_idx\":1,\"reason\":256,\"obs_hash\":0}",
            "{\"game_id\":\"x\",\"event_idx\":1,\"reason\":1,\"obs_hash\":0} ",
            "{\"game_id\":\"x\\q\",\"event_idx\":1,\"reason\":1,\"obs_hash\":0}",
        ] {
            if bad.is_empty() {
                continue;
            }
            assert!(decode_lineage_line(bad, 3).is_err(), "must reject {bad:?}");
        }
    }

    #[test]
    fn lineage_file_round_trips_and_joins() {
        let dir = std::env::temp_dir().join(format!("hydra-sink-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("quarantine.lineage");
        let records = vec![
            QuarantineLineage::from_parts("good-a", stub(11), 11),
            QuarantineLineage::from_parts("q-framing", stub(1), 22),
        ];
        write_lineage_file(&path, &records).expect("write");
        let back = read_lineage_file(&path).expect("read");
        assert_eq!(back, records);
        // Game lineage join: consumed games hit, clean games miss.
        let games = vec![
            ("good-a".to_string(), 11u64),
            ("clean-b".to_string(), 33u64),
            ("q-framing".to_string(), 22u64),
        ];
        let joined = join_lineage(&games, &back);
        assert_eq!(joined.len(), 3);
        assert!(joined[0].1.is_some());
        assert!(joined[1].1.is_none());
        assert!(joined[2].1.is_some());
        assert_eq!(joined[0].1.expect("hit").reason, 11);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn accounting_identity_holds_and_breaks_loudly() {
        assert!(check_accounting(0, 0, 0, 0).is_ok());
        assert!(check_accounting(8, 2, 0, 10).is_ok());
        assert!(check_accounting(8, 2, 3, 13).is_ok());
        let err = check_accounting(8, 2, 0, 9).expect_err("must break");
        assert_eq!(
            err.to_string(),
            "quarantine accounting broke: ok=8 quarantined=2 staged=0 consumed=9"
        );
        // Overflow is a mismatch, never a wrap.
        assert!(check_accounting(u64::MAX, 1, 0, 0).is_err());
        assert!(check_accounting(u64::MAX, u64::MAX, u64::MAX, u64::MAX).is_err());
    }
}
