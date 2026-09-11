//! S6 sink — quarantine lineage + accounting + terminal-hora han/fu (cold leg).
//!
//! OWNER: SinkBuilder (P4-A). Cold only: everything here runs AFTER the hot
//! verdict (post-close lineage IO, accounting asserts, rare terminal-hora
//! recompute). The hot path never calls in: gate/walk/fill take no dependency
//! on this module. Same-crate `stream.rs` calls only [`quarantine_reason_name`]
//! while capturing quarantines, never the file/score legs.
//!
//! Shape (plan §6.6 + P4-A + Wave-C post-claim rule):
//! - reason renderer: [`quarantine_reason_name`] (`reason < 10` → gate names,
//!   else walk names; unknown bytes stay `"other"` and MUST grow the
//!   vocabulary, never pass silently — the histogram test asserts `"other"`
//!   is absent on the corpus).
//! - closed sink taxonomy: [`sink_bucket`] over [`SINK_VOCABULARY`] (6
//!   buckets: framing/vocab/conservation/hora-mismatch/wall-perm/history-cap).
//! - lineage file per quarantine: [`QuarantineLineage`]
//!   `{identity, event_idx, reason, obs_hash u64}` JSON lines +
//!   [`join_lineage`] back to consumed games.
//! - accounting assert path: [`check_accounting`] (`ok + quarantined + staged
//!   == consumed`; G5 `games_quarantined + games_ok == games` at drain).
//! - han/fu recompute HERE only, terminal hora only (rare): [`score_hora`] +
//!   [`reconcile_hora`] (G4 scores-vs-delta). Report-only: `Mismatch` goes to
//!   cold review, never into hot accept/reject (the hot walk keeps its
//!   shape-only hora check by design).

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
/// Wave-C post-claim reject — reused code, no taxonomy churn).
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
/// Mapping (hub-agreed; every `REASON_*` / `WALK_*` const is covered):
/// - framing ← payload shape / blank line / boundary (1, 3, 4);
/// - vocab ← unknown-event (2, 17), unmapped ryukyoku (15), unresolved id (20);
/// - conservation ← bare-dora (6, 18), double-ron (7, 19), tile ledger (11),
///   claim-no-offer (12);
/// - hora-mismatch ← hora shape (8), turn-order (10), kyushu-ambiguous (14),
///   engine-desync (16), past-terminal (21: hora with trailing row/end
///   events and no closing end_kyoku — the wall oracle goes terminal);
/// - wall-perm ← wall-bearing (5), draw-past-wall (13).
/// `history-cap` is reserved for S4 T-bucket-overflow quarantines (today that
/// overflow surfaces as retryable `BufferTooSmall`, never a quarantine, so
/// the bucket pins vocabulary-closed with zero current emitters).
/// Unassigned bytes (0 `ok`, 9, 22+) → `"other"` (fail-loud canary).
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
    Ok(QuarantineLineage {
        game_id,
        event_idx: event_idx as u16,
        reason: reason as u8,
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

// ---------------------------------------------------------------------------
// Terminal-hora han/fu recompute (S6-cold, G4; rare leg, never hot).
// ---------------------------------------------------------------------------

/// Tile types for winds/honors (tile id `/ 4`).
pub const TILE_E: u8 = 27;
/// South wind type.
pub const TILE_S: u8 = 28;
/// West wind type.
pub const TILE_W: u8 = 29;
/// North wind type.
pub const TILE_N: u8 = 30;
/// White dragon type.
pub const TILE_HAKU: u8 = 31;
/// Green dragon type.
pub const TILE_HATSU: u8 = 32;
/// Red dragon type.
pub const TILE_CHUN: u8 = 33;

/// Single yakuman in han units.
pub const HAN_YAKUMAN: u8 = 13;

fn is_terminal_type(t: u8) -> bool {
    matches!(t, 0 | 8 | 9 | 17 | 18 | 26)
}

fn is_honor(t: u8) -> bool {
    t >= 27
}

fn is_simple(t: u8) -> bool {
    t < 27 && !is_terminal_type(t)
}

/// How the winning tile was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinKind {
    /// Self-draw.
    Tsumo,
    /// Discard claim.
    Ron,
}

/// Claimed wait shape (cold caller supplies it; the hot walk never derives it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitKind {
    /// Open-ended (no fu).
    Ryanmen,
    /// Closed (middle) wait (+2 fu).
    Kanchan,
    /// Edge (3 on 12 / 7 on 89) wait (+2 fu).
    Penchan,
    /// Double-pair wait (no fu).
    Shanpon,
    /// Single wait on the pair (+2 fu).
    Tanki,
}

/// Open meld kind (closed quads ride here too: `Ankan` counts closed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeldKind {
    /// Open sequence.
    Chi,
    /// Open triplet.
    Pon,
    /// Open quad (claimed).
    Daiminkan,
    /// Closed quad (counts closed for sanankou/menzen).
    Ankan,
    /// Added quad (open).
    Kakan,
}

/// One meld outside the concealed takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenMeld {
    /// Meld kind.
    pub kind: MeldKind,
    /// Tile type (`0..34`) of the meld's base (sequence start / triplet type).
    pub tile_type: u8,
}

impl OpenMeld {
    /// Takes bound in this meld (3 for chi/pon, 4 for quads).
    pub const fn takes(self) -> u8 {
        match self.kind {
            MeldKind::Chi | MeldKind::Pon => 3,
            MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => 4,
        }
    }

    /// True for concealed-equivalent melds (ankan only).
    pub const fn is_closed(self) -> bool {
        matches!(self.kind, MeldKind::Ankan)
    }
}

/// Terminal-hora recompute input (cold-assembled; all takes accounted:
/// `concealed.sum() + open takes == 14`, winning tile INCLUDED in concealed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoraInput {
    /// Closed takes by type, winning tile included.
    pub concealed: [u8; 34],
    /// Winning tile type (`0..34`, present in `concealed`).
    pub win_type: u8,
    /// Tsumo or ron.
    pub win: WinKind,
    /// Ron target seat (required for ron, absent for tsumo).
    pub ron_target: Option<u8>,
    /// Claimed wait shape.
    pub wait: WaitKind,
    /// Melds outside the concealed takes.
    pub open: Vec<OpenMeld>,
    /// Riichi declared.
    pub riichi: bool,
    /// Double riichi declared.
    pub double_riichi: bool,
    /// Ippatsu (one-shot).
    pub ippatsu: bool,
    /// Total dora han (indicators + red fives + ura, cold-summed).
    pub dora: u8,
    /// Win on a rinshan replacement (tsumo only).
    pub rinshan: bool,
    /// Win on a kan rob (ron only).
    pub chankan: bool,
    /// Last-tile win, self-draw (tsumo only).
    pub haitei: bool,
    /// Last-tile win, discard claim (ron only).
    pub houtei: bool,
    /// Winner's seat wind (`27..=30`).
    pub seat_wind: u8,
    /// Round wind (`27..=30`).
    pub round_wind: u8,
    /// Winner is dealer (payment side).
    pub dealer: bool,
    /// Dealer seat (`0..=3`): tsumo distribution side; reconcile checks
    /// `(winner == dealer_seat) == dealer` fail-loud.
    pub dealer_seat: u8,
}

/// Scored han/fu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HanFu {
    /// Han (13/26/39 = single/double/triple yakuman; 13 = counted).
    pub han: u8,
    /// Fu (25 for chiitoi/kokushi; payment ignores fu at mangan+).
    pub fu: u16,
}

/// Score payment in points (caller reads the side it needs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Payment {
    /// Ron receipt (loser pays).
    pub ron: u32,
    /// Tsumo receipt from each non-dealer.
    pub tsumo_ko: u32,
    /// Tsumo receipt from the dealer.
    pub tsumo_oya: u32,
}

/// Recompute outcome: scored, or manual review (fail-loud, never a guess).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScoreOutcome {
    /// Fully classified hand.
    Known {
        /// Scored han/fu (best interpretation by winner receipt).
        han_fu: HanFu,
        /// Points for the winner's side.
        payment: Payment,
    },
    /// Unclassifiable here (bad input, no yaku found, unimplemented shape):
    /// cold review, never a hot verdict.
    NeedsManual {
        /// Stable machine-readable cause (`bad-input` / `no-yaku` / `exotic`).
        why: &'static str,
    },
}

/// G4 reconciliation outcome (report-only; `Mismatch` goes to cold review,
/// never into hot accept/reject).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reconcile {
    /// Recomputed paymentTransfer equals the logged `deltas`.
    Match {
        /// Scored han.
        han: u8,
        /// Scored fu.
        fu: u16,
    },
    /// Logged `deltas` differ from every interpretation: cold review.
    Mismatch {
        /// Scored han.
        han: u8,
        /// Scored fu.
        fu: u16,
        /// Expected score movement (winner-indexed `[s0, s1, s2, s3]`).
        expected: [i32; 4],
    },
    /// Hand needs manual review (see [`ScoreOutcome::NeedsManual`]).
    NeedsManual {
        /// Stable machine-readable cause.
        why: &'static str,
    },
}

fn ceil100(v: u32) -> u32 {
    v.div_ceil(100) * 100
}

fn base_points(han: u8, fu: u16) -> u32 {
    if han >= 39 {
        24000
    } else if han >= 26 {
        16000
    } else if han >= HAN_YAKUMAN {
        8000
    } else if han >= 11 {
        6000
    } else if han >= 8 {
        4000
    } else if han >= 6 {
        3000
    } else if han >= 5 {
        2000
    } else {
        let b = u32::from(fu) * (1u32 << (u32::from(han) + 2));
        b.min(2000)
    }
}

/// Standard payment table for `(han, fu)` (fu ignored at mangan+).
pub fn hora_payment(han: u8, fu: u16, dealer: bool, tsumo: bool) -> Payment {
    let base = base_points(han, fu);
    if tsumo {
        let ko = ceil100(base);
        let oya = ceil100(base * 2);
        Payment {
            ron: 0,
            tsumo_ko: ko,
            tsumo_oya: oya,
        }
    } else if dealer {
        Payment {
            ron: ceil100(base * 6),
            tsumo_ko: 0,
            tsumo_oya: 0,
        }
    } else {
        Payment {
            ron: ceil100(base * 4),
            tsumo_ko: 0,
            tsumo_oya: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClosedShape {
    Shuntsu,
    Koutsu,
}

#[derive(Debug, Clone)]
struct Decomp {
    pair: u8,
    melds: Vec<(u8, ClosedShape)>,
}

fn split_into(
    counts: &mut [u8; 34],
    pair: Option<u8>,
    melds: &mut Vec<(u8, ClosedShape)>,
    target: usize,
    out: &mut Vec<Decomp>,
) {
    if out.len() >= 64 {
        return;
    }
    let mut t = 0usize;
    while t < 34 && counts[t] == 0 {
        t += 1;
    }
    if t == 34 {
        if melds.len() == target {
            if let Some(p) = pair {
                out.push(Decomp {
                    pair: p,
                    melds: melds.clone(),
                });
            }
        }
        return;
    }
    if pair.is_none() && counts[t] >= 2 {
        counts[t] -= 2;
        split_into(counts, Some(t as u8), melds, target, out);
        counts[t] += 2;
    }
    if melds.len() < target && counts[t] >= 3 {
        counts[t] -= 3;
        melds.push((t as u8, ClosedShape::Koutsu));
        split_into(counts, pair, melds, target, out);
        melds.pop();
        counts[t] += 3;
    }
    if melds.len() < target && t < 27 && t % 9 <= 6 && counts[t + 1] > 0 && counts[t + 2] > 0 {
        counts[t] -= 1;
        counts[t + 1] -= 1;
        counts[t + 2] -= 1;
        melds.push((t as u8, ClosedShape::Shuntsu));
        split_into(counts, pair, melds, target, out);
        melds.pop();
        counts[t] += 1;
        counts[t + 1] += 1;
        counts[t + 2] += 1;
    }
}

/// Standard-form decompositions of the concealed takes into pair + `target`
/// closed melds (no chiitoi/kokushi here; callers check those first).
fn decompose(counts: &[u8; 34], target: usize) -> Vec<Decomp> {
    let mut counts = *counts;
    let mut melds = Vec::with_capacity(4);
    let mut out = Vec::new();
    split_into(&mut counts, None, &mut melds, target, &mut out);
    out
}

fn is_chiitoi(counts: &[u8; 34]) -> bool {
    let mut pairs = 0u8;
    for t in 0..34 {
        match counts[t] {
            2 => pairs += 1,
            0 => {}
            _ => return false,
        }
    }
    pairs == 7
}

fn is_kokushi(counts: &[u8; 34]) -> bool {
    const NEED: [u8; 13] = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
    let mut sum = 0u32;
    for t in 0..34 {
        sum += u32::from(counts[t]);
    }
    if sum != 14 {
        return false;
    }
    let mut doubled = 0u8;
    for t in 0..34 {
        let c = counts[t];
        if NEED.contains(&(t as u8)) {
            if c == 2 {
                doubled += 1;
            } else if c != 1 {
                return false;
            }
        } else if c != 0 {
            return false;
        }
    }
    doubled == 1
}

fn is_chuuren(counts: &[u8; 34]) -> bool {
    let mut sum = 0u32;
    for t in 0..34 {
        sum += u32::from(counts[t]);
    }
    if sum != 14 {
        return false;
    }
    for suit in 0..3u8 {
        let b = (suit * 9) as usize;
        let c = &counts[b..b + 9];
        if c.iter().all(|&x| x == 0) {
            continue;
        }
        if c[0] >= 3 && c[8] >= 3 && c[1..8].iter().all(|&x| x >= 1) {
            return true;
        }
        return false;
    }
    false
}

fn is_all_green(counts: &[u8; 34]) -> bool {
    const GREEN: [u8; 6] = [19, 20, 21, 23, 25, 32];
    let mut sum = 0u32;
    for t in 0..34 {
        let c = counts[t];
        sum += u32::from(c);
        if c > 0 && !GREEN.contains(&(t as u8)) {
            return false;
        }
    }
    sum == 14
}

fn pair_value_han(pair: u8, input: &HoraInput) -> u8 {
    let mut han = 0u8;
    if pair >= TILE_HAKU {
        han += 1;
    }
    if pair == input.seat_wind {
        han += 1;
    }
    if pair == input.round_wind {
        han += 1;
    }
    han
}

fn meld_koutsu_fu(tile_type: u8, closed: bool) -> u16 {
    let big = is_terminal_type(tile_type) || is_honor(tile_type);
    match (closed, big) {
        (true, true) => 8,
        (true, false) => 4,
        (false, true) => 4,
        (false, false) => 2,
    }
}

fn open_meld_fu(m: OpenMeld) -> u16 {
    let big = is_terminal_type(m.tile_type) || is_honor(m.tile_type);
    match m.kind {
        MeldKind::Chi => 0,
        MeldKind::Pon => {
            if big {
                4
            } else {
                2
            }
        }
        MeldKind::Daiminkan | MeldKind::Kakan => {
            if big {
                16
            } else {
                8
            }
        }
        MeldKind::Ankan => {
            if big {
                32
            } else {
                16
            }
        }
    }
}

/// Yaku + fu for one standard interpretation. `exclude_triplet` names a
/// closed triplet of the winning type treated as the ron-open meld for the
/// sanankou count (`None` = count all closed). Returns `None` when the
/// interpretation carries no yaku.
fn eval_standard(
    decomp: &Decomp,
    input: &HoraInput,
    exclude_triplet: Option<u8>,
    hand_open: bool,
) -> Option<HanFu> {
    let closed = &decomp.melds;
    // Yakuman scan (closed melds + open melds + pair).
    let mut closed_quads = 0u8;
    let mut quads = 0u8;
    for m in &input.open {
        match m.kind {
            MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                quads += 1;
                if m.is_closed() {
                    closed_quads += 1;
                }
            }
            MeldKind::Chi | MeldKind::Pon => {}
        }
    }
    let mut closed_triplets = 0u8;
    for (t, shape) in closed {
        if *shape == ClosedShape::Koutsu {
            if Some(*t) == exclude_triplet {
                continue;
            }
            closed_triplets += 1;
        }
    }
    closed_triplets += closed_quads;
    // Wind triplet tallies (open or closed).
    let mut wind3 = [0u8; 4];
    let mut drag3 = [0u8; 3];
    let mut all_meld_koutsu = true;
    for (t, shape) in closed {
        if *shape == ClosedShape::Shuntsu {
            all_meld_koutsu = false;
        } else if *t >= TILE_E && *t <= TILE_N {
            wind3[(t - TILE_E) as usize] += 1;
        } else if *t >= TILE_HAKU {
            drag3[(t - TILE_HAKU) as usize] += 1;
        }
    }
    for m in &input.open {
        match m.kind {
            MeldKind::Chi => all_meld_koutsu = false,
            MeldKind::Pon | MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                if m.tile_type >= TILE_E && m.tile_type <= TILE_N {
                    wind3[(m.tile_type - TILE_E) as usize] += 1;
                } else if m.tile_type >= TILE_HAKU {
                    drag3[(m.tile_type - TILE_HAKU) as usize] += 1;
                }
            }
        }
    }
    let mut yakuman = 0u8;
    if drag3 == [1, 1, 1] {
        yakuman += 1;
    }
    if closed_triplets + quads - closed_quads >= 4 && input.win == WinKind::Tsumo && quads == closed_quads
    {
        // Four closed triplets/quads on tsumo (quads all ankan).
        yakuman += 1;
        if input.wait == WaitKind::Tanki {
            yakuman += 1;
        }
    }
    if quads == 4 {
        yakuman += 1;
    }
    if wind3 == [1, 1, 1, 1] {
        yakuman += 1;
    }
    if (wind3[0] + wind3[1] + wind3[2] + wind3[3] == 3)
        && (decomp.pair >= TILE_E && decomp.pair <= TILE_N)
    {
        yakuman += 1;
    }
    // All-terminals / all-honors / all-green over concealed + open takes.
    let mut concealed_all_term = true;
    let mut concealed_all_honor = true;
    for t in 0..34u8 {
        if input.concealed[t as usize] == 0 {
            continue;
        }
        if !is_terminal_type(t) {
            concealed_all_term = false;
        }
        if !is_honor(t) {
            concealed_all_honor = false;
        }
    }
    for m in &input.open {
        match m.kind {
            MeldKind::Chi => {
                concealed_all_term = false;
                concealed_all_honor = false;
            }
            MeldKind::Pon | MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                if !is_terminal_type(m.tile_type) {
                    concealed_all_term = false;
                }
                if !is_honor(m.tile_type) {
                    concealed_all_honor = false;
                }
            }
        }
    }
    if concealed_all_term && input.open.iter().all(|m| m.kind != MeldKind::Chi) {
        yakuman += 1;
    }
    if concealed_all_honor {
        yakuman += 1;
    }
    if yakuman > 0 {
        return Some(HanFu {
            han: HAN_YAKUMAN.saturating_mul(yakuman),
            fu: 25,
        });
    }

    // Normal yaku.
    let mut han: u8 = 0;
    let closed_hand = !hand_open;
    // Pinfu first (fu depends on it).
    let pair_valueless = pair_value_han(decomp.pair, input) == 0;
    let all_seq = closed.iter().all(|(_, s)| *s == ClosedShape::Shuntsu)
        && input
            .open
            .iter()
            .all(|m| m.kind == MeldKind::Chi);
    let pinfu = closed_hand && all_seq && pair_valueless && input.wait == WaitKind::Ryanmen;
    if pinfu {
        han += 1;
    }
    if input.win == WinKind::Tsumo && closed_hand {
        han += 1;
    }
    if input.riichi && closed_hand {
        han += if input.double_riichi { 2 } else { 1 };
    }
    if input.ippatsu && closed_hand && (input.riichi || input.double_riichi) {
        han += 1;
    }
    // Tanyao: every set all-simples.
    let mut meld_simple = true;
    for (t, _) in closed {
        if !is_simple(*t) {
            meld_simple = false;
        }
    }
    for m in &input.open {
        match m.kind {
            MeldKind::Chi => {
                if !is_simple(m.tile_type) || !is_simple(m.tile_type + 1) || !is_simple(m.tile_type + 2)
                {
                    meld_simple = false;
                }
            }
            MeldKind::Pon | MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                if !is_simple(m.tile_type) {
                    meld_simple = false;
                }
            }
        }
    }
    if meld_simple && is_simple(decomp.pair) {
        han += 1;
    }
    // Yakuhai triplets/quads (open or closed).
    for (t, shape) in closed {
        if *shape == ClosedShape::Koutsu {
            han += pair_value_han(*t, input);
        }
    }
    for m in &input.open {
        match m.kind {
            MeldKind::Chi => {}
            MeldKind::Pon | MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                han += pair_value_han(m.tile_type, input);
            }
        }
    }
    // Toitoi / sanankou / sankantsu.
    if all_meld_koutsu {
        han += 2;
    }
    if closed_triplets == 3 {
        han += 2;
    }
    if quads == 3 {
        han += 2;
    }
    // Sanshoku doujun (open or closed sequences count; han drops when open).
    let mut seq_present = [[false; 7]; 3];
    for (t, shape) in closed {
        if *shape == ClosedShape::Shuntsu && *t < 27 {
            seq_present[(t / 9) as usize][(t % 9) as usize] = true;
        }
    }
    for m in &input.open {
        if m.kind == MeldKind::Chi && m.tile_type < 27 {
            seq_present[(m.tile_type / 9) as usize][(m.tile_type % 9) as usize] = true;
        }
    }
    let mut sanshoku = false;
    for n in 0..7usize {
        if seq_present[0][n] && seq_present[1][n] && seq_present[2][n] {
            sanshoku = true;
        }
    }
    if sanshoku {
        han += if closed_hand { 2 } else { 1 };
    }
    // Ittsu.
    let mut ittsu = false;
    for s in 0..3u8 {
        if seq_present[s as usize][0] && seq_present[s as usize][3] && seq_present[s as usize][6] {
            ittsu = true;
        }
    }
    if ittsu {
        han += if closed_hand { 2 } else { 1 };
    }
    // Chanta / junchan / honroutou.
    let set_has_term_honor = |t: u8, seq: bool| -> bool {
        if seq {
            is_terminal_type(t) || is_terminal_type(t + 2)
        } else {
            is_terminal_type(t) || is_honor(t)
        }
    };
    let set_all_term = |t: u8, seq: bool| -> bool {
        if seq {
            (t % 9 == 0) || (t % 9 == 6)
        } else {
            is_terminal_type(t)
        }
    };
    let mut chanta = set_has_term_honor(decomp.pair, false);
    let mut junchan = is_terminal_type(decomp.pair);
    let mut honroutou = is_terminal_type(decomp.pair) || is_honor(decomp.pair);
    for (t, shape) in closed {
        let seq = *shape == ClosedShape::Shuntsu;
        chanta &= set_has_term_honor(*t, seq);
        junchan &= set_all_term(*t, seq);
        honroutou &= is_terminal_type(*t) || is_honor(*t);
        if seq {
            honroutou = false;
        }
    }
    for m in &input.open {
        match m.kind {
            MeldKind::Chi => {
                chanta &= set_has_term_honor(m.tile_type, true);
                junchan &= set_all_term(m.tile_type, true);
                honroutou = false;
            }
            MeldKind::Pon | MeldKind::Daiminkan | MeldKind::Ankan | MeldKind::Kakan => {
                chanta &= set_has_term_honor(m.tile_type, false);
                junchan &= set_all_term(m.tile_type, false);
                honroutou &= is_terminal_type(m.tile_type) || is_honor(m.tile_type);
            }
        }
    }
    if honroutou {
        han += 2;
        chanta = true;
        junchan = false;
    }
    if junchan {
        han += if closed_hand { 3 } else { 2 };
    } else if chanta {
        han += if closed_hand { 2 } else { 1 };
    }
    // Iipeikou / ryanpeikou (closed sequences only).
    if closed_hand {
        let mut seq_counts = [0u8; 27];
        for (t, shape) in closed {
            if *shape == ClosedShape::Shuntsu {
                seq_counts[*t as usize] += 1;
            }
        }
        let mut pairs2 = 0u8;
        let mut single = false;
        for t in 0..27 {
            if seq_counts[t] == 2 {
                pairs2 += 1;
            } else if seq_counts[t] == 4 {
                pairs2 += 2;
            } else if seq_counts[t] > 2 {
                single = true;
            }
        }
        if pairs2 == 2 && !single {
            han += 3;
        } else if pairs2 == 1 || single {
            han += 1;
        }
    }
    // Honitsu / chinitsu.
    let mut suits = [false; 4];
    for t in 0..34u8 {
        if input.concealed[t as usize] > 0 {
            suits[if t < 27 { (t / 9) as usize } else { 3 }] = true;
        }
    }
    for m in &input.open {
        if m.kind == MeldKind::Chi {
            suits[(m.tile_type / 9) as usize] = true;
        } else if is_honor(m.tile_type) {
            suits[3] = true;
        } else {
            suits[(m.tile_type / 9) as usize] = true;
        }
    }
    let suit_count = suits[0] as u8 + suits[1] as u8 + suits[2] as u8;
    if suit_count == 1 && suits[3] {
        han += if closed_hand { 3 } else { 2 };
    } else if suit_count == 1 {
        han += if closed_hand { 6 } else { 5 };
    }
    // Shousangen: two dragon triplets + dragon pair.
    let mut drag_pongs = 0u8;
    for d in 0..3u8 {
        if drag3[d as usize] > 0 {
            drag_pongs += 1;
        }
    }
    if drag_pongs == 2 && decomp.pair >= TILE_HAKU {
        han += 2;
    }
    // Situational 1-han.
    if input.rinshan {
        han += 1;
    }
    if input.chankan {
        han += 1;
    }
    if input.haitei {
        han += 1;
    }
    if input.houtei {
        han += 1;
    }
    han += input.dora;
    if han == 0 {
        return None;
    }
    if han >= HAN_YAKUMAN {
        return Some(HanFu { han: HAN_YAKUMAN, fu: 25 });
    }

    // Fu.
    let mut fu: u16 = 20;
    if input.win == WinKind::Ron && closed_hand {
        fu += 10;
    }
    if input.win == WinKind::Tsumo && !(pinfu && closed_hand) {
        fu += 2;
    }
    for (t, shape) in closed {
        if *shape == ClosedShape::Koutsu {
            let ron_open = Some(*t) == exclude_triplet;
            fu += meld_koutsu_fu(*t, !ron_open);
        }
    }
    for m in &input.open {
        fu += open_meld_fu(*m);
    }
    fu += u16::from(pair_value_han(decomp.pair, input)) * 2;
    match input.wait {
        WaitKind::Ryanmen | WaitKind::Shanpon => {}
        WaitKind::Kanchan | WaitKind::Penchan | WaitKind::Tanki => fu += 2,
    }
    fu = ((fu + 9) / 10) * 10;
    if fu < 30 {
        fu = 30;
    }
    Some(HanFu { han, fu })
}

/// Chiitoi scoring (closed only): 25 fu + 2 han + combinable subset
/// (tanyao / honitsu / chinitsu / riichi family / tsumo / dora).
fn eval_chiitoi(input: &HoraInput) -> Option<HanFu> {
    if !input.open.is_empty() {
        return None;
    }
    let mut han: u8 = 2;
    let mut all_simple = true;
    let mut suits = [false; 4];
    for t in 0..34u8 {
        if input.concealed[t as usize] > 0 {
            if !is_simple(t) {
                all_simple = false;
            }
            suits[if t < 27 { (t / 9) as usize } else { 3 }] = true;
        }
    }
    if all_simple {
        han += 1;
    }
    let suit_count = suits[0] as u8 + suits[1] as u8 + suits[2] as u8;
    if suit_count == 1 && suits[3] {
        han += 3;
    } else if suit_count == 1 {
        han += 6;
    }
    if input.win == WinKind::Tsumo {
        han += 1;
    }
    if input.riichi {
        han += if input.double_riichi { 2 } else { 1 };
    }
    if input.ippatsu && (input.riichi || input.double_riichi) {
        han += 1;
    }
    if input.rinshan || input.chankan || input.haitei || input.houtei {
        han += 1;
    }
    han += input.dora;
    if han >= HAN_YAKUMAN {
        return Some(HanFu { han: HAN_YAKUMAN, fu: 25 });
    }
    Some(HanFu { han, fu: 25 })
}

fn winner_receipt(payment: Payment, dealer: bool, tsumo: bool) -> u32 {
    if !tsumo {
        payment.ron
    } else if dealer {
        payment.tsumo_oya * 3
    } else {
        payment.tsumo_ko * 2 + payment.tsumo_oya
    }
}

/// Cold terminal-hora recompute: enumerate closed-form interpretations of
/// the concealed takes and score the best by winner receipt. `NeedsManual`
/// on bad input, yaku-less hands, or unimplemented (exotic) shapes —
/// fail-loud, never a guess.
pub fn score_hora(input: &HoraInput) -> ScoreOutcome {
    if input.win_type >= 34
        || input.open.len() > 4
        || input.dealer_seat > 3
        || !(TILE_E..=TILE_N).contains(&input.seat_wind)
        || !(TILE_E..=TILE_N).contains(&input.round_wind)
        || input.concealed[input.win_type as usize] == 0
    {
        return ScoreOutcome::NeedsManual { why: "bad-input" };
    }
    if (input.haitei && input.win != WinKind::Tsumo)
        || (input.houtei && input.win != WinKind::Ron)
        || (input.rinshan && input.win != WinKind::Tsumo)
        || (input.chankan && input.win != WinKind::Ron)
    {
        return ScoreOutcome::NeedsManual { why: "bad-input" };
    }
    if (input.win == WinKind::Tsumo && input.ron_target.is_some())
        || (input.win == WinKind::Ron
            && input.ron_target.map(|s| s > 3).unwrap_or(true))
    {
        return ScoreOutcome::NeedsManual { why: "bad-input" };
    }
    let mut open_takes = 0u32;
    for m in &input.open {
        if m.tile_type >= 34 {
            return ScoreOutcome::NeedsManual { why: "bad-input" };
        }
        open_takes += u32::from(m.takes());
    }
    let mut concealed_sum = 0u32;
    for t in 0..34 {
        let c = input.concealed[t];
        if c > 4 {
            return ScoreOutcome::NeedsManual { why: "bad-input" };
        }
        concealed_sum += u32::from(c);
    }
    if concealed_sum + open_takes != 14 {
        return ScoreOutcome::NeedsManual { why: "bad-input" };
    }
    let hand_open = !input.open.is_empty();

    // Kokushi (closed thirteen-orphans): immediate yakuman.
    if !hand_open && is_kokushi(&input.concealed) {
        let payment = hora_payment(HAN_YAKUMAN, 25, input.dealer, input.win == WinKind::Tsumo);
        return ScoreOutcome::Known {
            han_fu: HanFu {
                han: HAN_YAKUMAN,
                fu: 25,
            },
            payment,
        };
    }
    // Chuuren (closed nine-gates): immediate yakuman.
    if !hand_open && is_chuuren(&input.concealed) {
        let payment = hora_payment(HAN_YAKUMAN, 25, input.dealer, input.win == WinKind::Tsumo);
        return ScoreOutcome::Known {
            han_fu: HanFu {
                han: HAN_YAKUMAN,
                fu: 25,
            },
            payment,
        };
    }
    // All-green (ryuuiisou): immediate yakuman.
    if is_all_green(&input.concealed)
        && input.open.iter().all(|m| {
            m.kind == MeldKind::Chi
                || ((m.tile_type == 19
                    || m.tile_type == 20
                    || m.tile_type == 21
                    || m.tile_type == 23
                    || m.tile_type == 25
                    || m.tile_type == TILE_HATSU)
                    && m.kind != MeldKind::Chi)
        })
    {
        let payment = hora_payment(HAN_YAKUMAN, 25, input.dealer, input.win == WinKind::Tsumo);
        return ScoreOutcome::Known {
            han_fu: HanFu {
                han: HAN_YAKUMAN,
                fu: 25,
            },
            payment,
        };
    }
    // Chiitoi (closed seven-pairs).
    if !hand_open && is_chiitoi(&input.concealed) {
        if let Some(han_fu) = eval_chiitoi(input) {
            let payment = hora_payment(han_fu.han, han_fu.fu, input.dealer, input.win == WinKind::Tsumo);
            return ScoreOutcome::Known { han_fu, payment };
        }
        return ScoreOutcome::NeedsManual { why: "no-yaku" };
    }

    // Standard form: closed melds + pair.
    let target = 4usize.saturating_sub(input.open.len());
    let decomps = decompose(&input.concealed, target);
    if decomps.is_empty() {
        return ScoreOutcome::NeedsManual { why: "exotic" };
    }
    let tsumo = input.win == WinKind::Tsumo;
    let mut best: Option<(HanFu, Payment)> = None;
    for decomp in &decomps {
        // Ron openness variants for the sanankou count: the winning tile
        // may complete a triplet (open) or a sequence (triplets all closed).
        let mut variants = vec![None];
        if !tsumo {
            for (t, shape) in &decomp.melds {
                if *shape == ClosedShape::Koutsu && *t == input.win_type {
                    variants.push(Some(*t));
                    break;
                }
            }
        }
        for exclude in variants {
            if let Some(han_fu) = eval_standard(decomp, input, exclude, hand_open) {
                let payment = hora_payment(han_fu.han, han_fu.fu, input.dealer, tsumo);
                let receipt = winner_receipt(payment, input.dealer, tsumo);
                let replace = match best {
                    None => true,
                    Some((b, p)) => {
                        receipt > winner_receipt(p, input.dealer, tsumo) || (receipt == winner_receipt(p, input.dealer, tsumo) && han_fu.fu > b.fu)
                    }
                };
                if replace {
                    best = Some((han_fu, payment));
                }
            }
        }
    }
    match best {
        Some((han_fu, payment)) => ScoreOutcome::Known { han_fu, payment },
        None => ScoreOutcome::NeedsManual { why: "no-yaku" },
    }
}

/// G4 scores-vs-delta reconciliation (cold, report-only): recompute the
/// terminal hora and compare the winner-indexed score movement against the
/// logged `deltas` (honba/kyotaku sticks applied standardly). `Mismatch` is
/// cold review, never a hot verdict.
pub fn reconcile_hora(
    input: &HoraInput,
    winner: usize,
    deltas: [i32; 4],
    honba: u32,
    kyotaku: u32,
) -> Reconcile {
    if winner > 3 || (winner == usize::from(input.dealer_seat)) != input.dealer {
        return Reconcile::NeedsManual { why: "bad-input" };
    }
    let (han_fu, payment) = match score_hora(input) {
        ScoreOutcome::Known { han_fu, payment } => (han_fu, payment),
        ScoreOutcome::NeedsManual { why } => return Reconcile::NeedsManual { why },
    };
    let tsumo = input.win == WinKind::Tsumo;
    let mut expected = [0i32; 4];
    if tsumo {
        for p in 0..4 {
            if p == winner {
                continue;
            }
            let base = if p == usize::from(input.dealer_seat) {
                payment.tsumo_oya
            } else {
                payment.tsumo_ko
            };
            let pay = base + 100 * honba;
            expected[p] -= pay as i32;
            expected[winner] += pay as i32;
        }
        expected[winner] += (1000 * kyotaku) as i32;
    } else {
        let target = match input.ron_target {
            Some(s) if (s as usize) < 4 && (s as usize) != winner => s as usize,
            _ => return Reconcile::NeedsManual { why: "bad-input" },
        };
        let gain = payment.ron + 300 * honba + 1000 * kyotaku;
        expected[target] -= gain as i32;
        expected[winner] += gain as i32;
    }
    if expected == deltas {
        Reconcile::Match {
            han: han_fu.han,
            fu: han_fu.fu,
        }
    } else {
        Reconcile::Mismatch {
            han: han_fu.han,
            fu: han_fu.fu,
            expected,
        }
    }
}

#[cfg(test)]
mod sink_tests {
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
            assert!(
                decode_lineage_line(bad, 3).is_err(),
                "must reject {bad:?}"
            );
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

    // -- han/fu pins (standard scoring anchors) --

    fn base_input() -> HoraInput {
        HoraInput {
            concealed: [0u8; 34],
            win_type: 0,
            win: WinKind::Ron,
            ron_target: Some(1),
            wait: WaitKind::Ryanmen,
            open: Vec::new(),
            riichi: false,
            double_riichi: false,
            ippatsu: false,
            dora: 0,
            rinshan: false,
            chankan: false,
            haitei: false,
            houtei: false,
            seat_wind: TILE_S,
            round_wind: TILE_E,
            dealer: false,
            dealer_seat: 1,
        }
    }

    fn set_counts(input: &mut HoraInput, types: &[u8]) {
        for t in types {
            input.concealed[*t as usize] += 1;
        }
    }

    #[test]
    fn pinfu_closed_ron_scores_1han_30fu() {
        // 234m 123p 345s 678s + pair 5m, ron on 8s ryanmen.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[1, 2, 3, 9, 10, 11, 20, 21, 22, 23, 24, 25, 4, 4],
        );
        input.win_type = 25;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 1, fu: 30 });
                assert_eq!(payment.ron, 1000);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
        // Same hand + 2 dora = 3 han 30 fu ron = 3900.
        input.dora = 2;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 3, fu: 30 });
                assert_eq!(payment.ron, 3900);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
    }

    #[test]
    fn tanyao_closed_tsumo_scores_2han_30fu() {
        // 234m 345p 555s 678s + pair 6p, tsumo on 8s ryanmen.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[1, 2, 3, 10, 11, 12, 21, 21, 21, 23, 24, 25, 14, 14],
        );
        input.win_type = 25;
        input.win = WinKind::Tsumo;
        input.ron_target = None;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 2, fu: 30 });
                assert_eq!(payment.tsumo_ko, 500);
                assert_eq!(payment.tsumo_oya, 1000);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
    }

    #[test]
    fn yakuhai_triplet_scores_fu_and_han() {
        // Haku pon + 234m + 345p + 567s + pair 6p, ron on 7s.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[31, 31, 31, 1, 2, 3, 10, 11, 12, 22, 23, 24, 14, 14],
        );
        input.win_type = 24;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 1, fu: 40 });
                assert_eq!(payment.ron, 1300);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
    }

    #[test]
    fn chiitoi_scores_fixed_25fu() {
        // Seven pairs incl honors, ron tanki.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[0, 0, 4, 4, 10, 10, 14, 14, 20, 20, 24, 24, 27, 27],
        );
        input.win_type = 27;
        input.wait = WaitKind::Tanki;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 2, fu: 25 });
                assert_eq!(payment.ron, 1600);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
    }
    #[test]
    fn reconcile_tsumo_applies_honba_and_kyotaku() {
        // Chiitoi tsumo, no riichi: 2 + tsumo = 3 han 25 fu -> ko 800 /
        // oya 1600. Winner seat 0 (child), dealer seat 1, honba 1 (+100 per
        // payer), kyotaku 2 (+2000 to winner).
        let mut input = base_input();
        set_counts(
            &mut input,
            &[0, 0, 4, 4, 10, 10, 14, 14, 20, 20, 24, 24, 27, 27],
        );
        input.win_type = 27;
        input.wait = WaitKind::Tanki;
        input.win = WinKind::Tsumo;
        input.ron_target = None;
        assert_eq!(
            reconcile_hora(&input, 0, [5500, -1700, -900, -900], 1, 2),
            Reconcile::Match { han: 3, fu: 25 }
        );
        // Dealer/consistency mismatch fails loud.
        match reconcile_hora(&input, 1, [5500, -1700, -900, -900], 1, 2) {
            Reconcile::NeedsManual { why } => assert_eq!(why, "bad-input"),
            other => panic!("must reject, got {other:?}"),
        }
    }

    #[test]
    fn toitoi_ron_downgrade_scores_sanankou_mangan() {
        // Four closed triplets + pair, ron: suuankou shape downgrades to
        // sanankou + toitoi = 4 han; fu: 20+10+(8+4+4)+4 ron-open -> 50 -> mangan 8000.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[0, 0, 0, 10, 10, 10, 20, 20, 20, 27, 27, 27, 5, 5],
        );
        input.win_type = 27;
        input.wait = WaitKind::Shanpon;
        input.seat_wind = TILE_S;
        input.round_wind = TILE_N;
        match score_hora(&input) {
            ScoreOutcome::Known { han_fu, payment } => {
                assert_eq!(han_fu, HanFu { han: 4, fu: 50 });
                assert_eq!(payment.ron, 8000);
            }
            ScoreOutcome::NeedsManual { why } => panic!("must score: {why}"),
        }
    }

    #[test]
    fn payment_table_pins_mangan_family() {
        assert_eq!(hora_payment(3, 60, false, false).ron, 7700);
        assert_eq!(hora_payment(3, 70, false, false).ron, 8000);
        assert_eq!(hora_payment(5, 30, false, false).ron, 8000);
        assert_eq!(hora_payment(5, 30, true, false).ron, 12000);
        assert_eq!(hora_payment(6, 40, false, false).ron, 12000);
        assert_eq!(hora_payment(8, 30, false, false).ron, 16000);
        assert_eq!(hora_payment(11, 30, false, false).ron, 24000);
        assert_eq!(hora_payment(13, 25, false, false).ron, 32000);
        assert_eq!(hora_payment(13, 25, true, false).ron, 48000);
        let mangan_child = hora_payment(3, 70, false, true);
        assert_eq!((mangan_child.tsumo_ko, mangan_child.tsumo_oya), (2000, 4000));
        let mangan_dealer = hora_payment(4, 40, true, true);
        assert_eq!((mangan_dealer.tsumo_ko, mangan_dealer.tsumo_oya), (2000, 4000));
    }

    #[test]
    fn reconcile_matches_logged_deltas() {
        // Yakuhai vector above: child ron 1300 off seat 1.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[31, 31, 31, 1, 2, 3, 10, 11, 12, 22, 23, 24, 14, 14],
        );
        input.win_type = 24;
        assert_eq!(
            reconcile_hora(&input, 0, [1300, -1300, 0, 0], 0, 0),
            Reconcile::Match { han: 1, fu: 40 }
        );
        // Wrong deltas -> Mismatch with the expected movement.
        match reconcile_hora(&input, 0, [8000, -8000, 0, 0], 0, 0) {
            Reconcile::Mismatch { han, fu, expected } => {
                assert_eq!((han, fu), (1, 40));
                assert_eq!(expected, [1300, -1300, 0, 0]);
            }
            other => panic!("must mismatch, got {other:?}"),
        }
    }

    #[test]
    fn open_yakuless_hand_needs_manual() {
        // Open pinfu shape, no dora, no yaku: fail-loud, never a guess.
        let mut input = base_input();
        set_counts(&mut input, &[1, 2, 3, 10, 11, 12, 23, 24, 25, 4, 4]);
        input.win_type = 25;
        input.open.push(OpenMeld {
            kind: MeldKind::Pon,
            tile_type: TILE_E,
        });
        input.seat_wind = TILE_S;
        input.round_wind = TILE_N;
        match score_hora(&input) {
            ScoreOutcome::NeedsManual { why } => assert_eq!(why, "no-yaku"),
            ScoreOutcome::Known { han_fu, .. } => panic!("must not score: {han_fu:?}"),
        }
        match reconcile_hora(&input, 0, [1000, -1000, 0, 0], 0, 0) {
            Reconcile::NeedsManual { why } => assert_eq!(why, "no-yaku"),
            other => panic!("must propagate manual, got {other:?}"),
        }
    }

    #[test]
    fn bad_inputs_need_manual_never_panic() {
        let mut input = base_input();
        // Concealed sum != 14.
        set_counts(&mut input, &[1, 2, 3]);
        input.win_type = 3;
        match score_hora(&input) {
            ScoreOutcome::NeedsManual { why } => assert_eq!(why, "bad-input"),
            other => panic!("must reject, got {other:?}"),
        }
        // Win tile absent from concealed.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[1, 2, 3, 9, 10, 11, 20, 21, 22, 23, 24, 25, 4, 4],
        );
        input.win_type = 5;
        match score_hora(&input) {
            ScoreOutcome::NeedsManual { why } => assert_eq!(why, "bad-input"),
            other => panic!("must reject, got {other:?}"),
        }
        // Winner seat out of range reconciles manual.
        let mut input = base_input();
        set_counts(
            &mut input,
            &[1, 2, 3, 9, 10, 11, 20, 21, 22, 23, 24, 25, 4, 4],
        );
        input.win_type = 25;
        match reconcile_hora(&input, 9, [0, 0, 0, 0], 0, 0) {
            Reconcile::NeedsManual { why } => assert_eq!(why, "bad-input"),
            other => panic!("must reject, got {other:?}"),
        }
    }
}
