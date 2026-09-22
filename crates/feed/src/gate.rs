//! S2 gate — strict + sample fused (`feed::gate`).
//!
//! Replaces, verdict-identical to the oracles:
//! - `decisions::prescan` (bare-dora / double-ron whole-game quarantine),
//! - the hydra1 strict-validator port (`decode_game_object` boundary +
//!   canonical start/end counts, `_count_row_decisions` unknown-event rule),
//! - `py_stream.rs:153-202` `check_decision_envelope` re-parse (deleted on the
//!   hot path: the envelope JSON round-trip is gone; walked rows flow straight
//!   into planes via S3/S4, never re-serialized).
//!
//! Hot/cold split (normative):
//! - HOT (this module): one walk over stacked events — strict start/end count
//!   (oracle-exact, including bare `start`/`end` alias accounting),
//!   type-present (`KIND_OTHER` → InvalidData), [`should_sample`] over the 7
//!   row kinds, bare-dora + double-ron prescan, hora `deltas`-quad SHAPE check.
//!   Zero-alloc: [`QuarantineStub`] (`game_idx u32, event_idx u16, reason u8`)
//!   and [`GateReject`] carry no `String` on the hot path.
//! - COLD (never S2/S3): hora scores-vs-delta reconciliation and any han/fu
//!   recompute (S6 quarantine leg / `shard::parity`, G4). The gate checks that
//!   a 4-int `deltas` quad EXISTS and is well-shaped; it never interprets it.
//!
//! Decision channels:
//! - `Ok(SampleDecision::Sample)` — walk the game. Quarantined games never
//!   take this path, so quarantined rows can never pollute the decisions/sec
//!   numerator (G5: `games_quarantined + games_ok == games`, skips emit zero).
//! - `Ok(SampleDecision::SkipQuarantine(stub))` — semantic quarantine
//!   (reasons 6–8). Skip-and-count hot.
//! - `Err(GateReject)` — framing/vocab family (reasons 1–5) via
//!   `From<GateReject> for std::io::Error` as `InvalidData` (`Other` →
//!   `InvalidData`). `feed::ingest::frame_spans` surfaces its own framing
//!   failures through the same type via the `empty`/`blank_line`/…
//!   constructors.
//!
//! Reason codes (u8 taxonomy; [`reason_name`] renders the cold
//! string for sink / lineage / histogram joins):
//! - 0 `ok` — accepted (never emitted as a verdict).
//! - 1 `framing` — `Err` (payload shape; ingest constructors).
//! - 2 `unknown-event` — `Err` (`KIND_OTHER` → InvalidData).
//! - 3 `framing` — `Err` (blank line; raised by ingest, reserved here).
//! - 4 `framing` — `Err` (boundary / start-end count).
//! - 5 `wall-bearing` — `Err` (wall permutation; raised by ingest via
//!   [`is_wall_perm136`]).
//! - 6 `bare-dora` — `Skip` (kan-dora indicator unrecoverable).
//! - 7 `double-ron` — `Skip` (two ron winners on one discard).
//! - 8 `turn-order` — `Skip` (hora `deltas`-quad shape; matches the oracle
//!   message family — both oracles report a missing/malformed quad under
//!   turn-order, so the string join agrees even though the u8 bucket is new).
//!
//! G1 note: verdicts are a pure function of `(game_idx, kinds, spans)` —
//! deterministic, no map iteration, no time/hash input.
//!
//! Allocation audit: the hot path ([`gate_game`], [`should_sample`], the span
//! scanners, [`is_wall_perm136`]) uses no `String`/`Vec`/`format!`/`Box`.
//! `String`-adjacent machinery (`Display`, `Error`, `io::Error` conversion)
//! is cold lineage only and never runs inside the sample loop.

use crate::ingest::{
    StackedEvent, KIND_DAHAI, KIND_DORA, KIND_END, KIND_HORA, KIND_OTHER, KIND_REACH_ACCEPTED,
    KIND_START, KIND_START_KYOKU,
};

/// Verdict bucket: accepted (never emitted; `Sample` covers it).
pub const REASON_OK: u8 = 0;
/// Verdict bucket: payload shape (`Err`, ingest constructors).
pub const REASON_FRAMING: u8 = 1;
/// Verdict bucket: unknown event kind (`Err`, Other → InvalidData).
pub const REASON_UNKNOWN_KIND: u8 = 2;
/// Verdict bucket: blank line (`Err`, raised by ingest).
pub const REASON_BLANK_LINE: u8 = 3;
/// Verdict bucket: boundary / start-end count (`Err`).
pub const REASON_BOUNDARY: u8 = 4;
/// Verdict bucket: wall permutation (`Err`, raised by ingest).
pub const REASON_WALL_PERM: u8 = 5;
/// Verdict bucket: bare kan-dora marker (`Skip`).
///
/// Strict-oracle behavior (`log_replay._do_dora` + `prescan` fail closed:
/// "the indicator is unrecoverable from the log"). NOTE: the bench F1 smoke
/// file pads 37 bare `{"type":"dora"}` no-ops for the EXPANDER path (which
/// skips dora verbatim and counts 5 rows); the strict oracle — and this gate
/// — quarantine that same content. Path-dependent by design, not a mismatch.
pub const REASON_BARE_DORA: u8 = 6;
/// Verdict bucket: double ron on one discard (`Skip`).
pub const REASON_DOUBLE_RON: u8 = 7;
/// Verdict bucket: hora `deltas`-quad shape (`Skip`, renders `turn-order`).
pub const REASON_HORA_SHAPE: u8 = 8;

/// Cold string for the sink / lineage / histogram join (G5 closed vocabulary;
/// unknown bytes bucket to `"other"` and MUST grow the vocabulary, never pass
/// silently — the histogram test asserts `"other" not in codes`).
pub const fn reason_name(reason: u8) -> &'static str {
    match reason {
        REASON_OK => "ok",
        REASON_FRAMING | REASON_BLANK_LINE | REASON_BOUNDARY => "framing",
        REASON_UNKNOWN_KIND => "unknown-event",
        REASON_WALL_PERM => "wall-bearing",
        REASON_BARE_DORA => "bare-dora",
        REASON_DOUBLE_RON => "double-ron",
        REASON_HORA_SHAPE => "turn-order",
        _ => "other",
    }
}

/// Zero-alloc quarantine record for the hot skip path (game_idx + event_idx + `REASON_*` bucket).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct QuarantineStub {
    /// Stable game index (sink join key).
    pub game_idx: u32,
    /// Offending event index, saturated at `u16::MAX`.
    pub event_idx: u16,
    /// Reason bucket (`REASON_*`).
    pub reason: u8,
}

/// Framing/vocab failure for the `Err` channel (`GateReject`: same game_idx/event_idx/reason triple).
///
/// Same triple as [`QuarantineStub`]; converts to `std::io::ErrorKind::InvalidData`
/// (cold only) and back to a stub for unified sink accounting.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GateReject {
    /// Stable game index (sink join key).
    pub game_idx: u32,
    /// Offending event index, saturated at `u16::MAX`.
    pub event_idx: u16,
    /// Reason bucket (`REASON_*`, framing family 1–5).
    pub reason: u8,
}

impl QuarantineStub {
    /// Build a stub, saturating oversized indices (games past 64K events).
    pub const fn new(game_idx: u32, event_idx: u32, reason: u8) -> Self {
        Self {
            game_idx,
            event_idx: saturate_idx(event_idx),
            reason,
        }
    }
}

impl GateReject {
    /// Build a reject, saturating oversized indices (games past 64K events).
    pub const fn new(game_idx: u32, event_idx: u32, reason: u8) -> Self {
        Self {
            game_idx,
            event_idx: saturate_idx(event_idx),
            reason,
        }
    }

    /// Unified accounting: every `Err` converts losslessly to a sink stub.
    pub const fn stub(self) -> QuarantineStub {
        QuarantineStub {
            game_idx: self.game_idx,
            event_idx: self.event_idx,
            reason: self.reason,
        }
    }

    // -- Ingest framing constructors (all const, zero-alloc).
    // -- Reason mapping: shape → 1, unusable type → 2, blank → 3,
    // -- boundary/count → 4, wall → 5.
    /// Empty payload (no records at all).
    pub const fn empty(game_idx: u32) -> Self {
        Self::new(game_idx, 0, REASON_FRAMING)
    }
    /// Payload missing its trailing newline.
    pub const fn missing_newline(game_idx: u32) -> Self {
        Self::new(game_idx, 0, REASON_FRAMING)
    }
    /// Blank line at `line` (decode.py: blank lines forbidden).
    pub const fn blank_line(game_idx: u32, line: u32) -> Self {
        Self::new(game_idx, line, REASON_BLANK_LINE)
    }
    /// Payload not valid UTF-8.
    pub const fn utf8(game_idx: u32) -> Self {
        Self::new(game_idx, 0, REASON_FRAMING)
    }
    /// Compressed payload fails to decode.
    pub const fn decompress(game_idx: u32) -> Self {
        Self::new(game_idx, 0, REASON_FRAMING)
    }
    /// Line `line` is not valid JSON / not a JSON object.
    pub const fn invalid_json(game_idx: u32, line: u32) -> Self {
        Self::new(game_idx, line, REASON_FRAMING)
    }
    /// Line `line` carries no `type` field (unusable → unknown-kind family).
    pub const fn missing_type(game_idx: u32, line: u32) -> Self {
        Self::new(game_idx, line, REASON_UNKNOWN_KIND)
    }
    /// Line `line` carries a non-string, empty, or unmapped `type`.
    pub const fn bad_type(game_idx: u32, line: u32) -> Self {
        Self::new(game_idx, line, REASON_UNKNOWN_KIND)
    }
    /// First/last record is not a start/end alias.
    pub const fn bad_boundary(game_idx: u32, line: u32) -> Self {
        Self::new(game_idx, line, REASON_BOUNDARY)
    }
    /// Start/end canonical counts are not exactly one each.
    pub const fn start_end_count(game_idx: u32) -> Self {
        Self::new(game_idx, 0, REASON_BOUNDARY)
    }
    /// Malformed tile value on line `line`.
    pub const fn bad_tile(game_idx: u32, line: u32) -> Self {
        Self::new(game_idx, line, REASON_FRAMING)
    }
    /// Wall field fails the 136-permutation check.
    pub const fn bad_wall(game_idx: u32) -> Self {
        Self::new(game_idx, 0, REASON_WALL_PERM)
    }
}

const fn saturate_idx(idx: u32) -> u16 {
    // proof: guarded idx <= u16::MAX above, fits in u16.
    #[allow(clippy::cast_possible_truncation)]
    if idx > u16::MAX as u32 {
        u16::MAX
    } else {
        idx as u16
    }
}

impl core::fmt::Display for GateReject {
    /// Cold lineage detail (never called on the sample hot path).
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "gate game {} event {}: {}",
            self.game_idx,
            self.event_idx,
            reason_name(self.reason)
        )
    }
}

impl std::error::Error for GateReject {}

impl From<GateReject> for std::io::Error {
    /// Other → InvalidData (cold surfacing only; the box alloc never runs hot).
    fn from(r: GateReject) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidData, r)
    }
}

/// Per-game sample verdict (walk the game, or skip-and-count with a quarantine stub).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SampleDecision {
    /// Walk the game into rows.
    Sample,
    /// Skip-and-count hot: the game contributes zero rows (never numerator).
    SkipQuarantine(QuarantineStub),
}

impl SampleDecision {
    /// True for walkable games.
    pub const fn is_sample(self) -> bool {
        matches!(self, Self::Sample)
    }

    /// The quarantine record, if skipped.
    pub const fn quarantine(self) -> Option<QuarantineStub> {
        match self {
            Self::Sample => None,
            Self::SkipQuarantine(stub) => Some(stub),
        }
    }
}

/// Per-event row predicate: exactly the 7 `_ROW_TYPES` kinds
/// (`dahai/chi/pon/daiminkan/ankan/kakan/hora`, LUT codes 2–8).
#[inline]
pub const fn should_sample(kind: u8) -> bool {
    matches!(
        kind,
        KIND_DAHAI | KIND_CHI_M | KIND_PON_M | KIND_DAIMINKAN_M | KIND_ANKAN_M | KIND_KAKAN_M | KIND_HORA
    )
}

// Re-exported here under gate-local aliases so the predicate reads without a
// second import block at use sites; values are ingest's LUT, verbatim.
use crate::ingest::{
    KIND_ANKAN as KIND_ANKAN_M, KIND_CHI as KIND_CHI_M, KIND_DAIMINKAN as KIND_DAIMINKAN_M,
    KIND_KAKAN as KIND_KAKAN_M, KIND_PON as KIND_PON_M,
};

/// Strict + sample fused verdict over one framed game.
///
/// Single walk: boundary → per-event unknown-kind / bare-dora / double-ron /
/// hora-quad checks → canonical start/end counts. First offender wins.
///
/// `game_idx` has no other source on the stacked slice (`FramedGame` carries it,
/// not per event), so it is an explicit parameter: every reject/skip triple
/// needs it for sink accounting.
///
/// Self-draw heuristic: a hora with `target == actor` is treated as tsumo
/// (mirrors the oracle fallback for Tenhou-real wins, which carry no flag and
/// no pai); only `actor != target` rons arm the double-ron pending slot.
/// Ingest normalizes an explicit `tsumo:true` hora with `target != actor` to
/// `target = actor` when stacking, so the pending slot never sees a flagged
/// tsumo as a ron candidate.
pub fn gate_game(game_idx: u32, ev: &[StackedEvent<'_>]) -> Result<SampleDecision, GateReject> {
    let Some(first) = ev.first() else {
        return Err(GateReject::empty(game_idx));
    };
    // `first` exists, so `last` exists; index directly.
    let last = &ev[ev.len() - 1];
    if first.kind != KIND_START {
        return Err(GateReject::bad_boundary(game_idx, 0));
    }
    if last.kind != KIND_END {
        return Err(GateReject::bad_boundary(
            game_idx,
            u32::try_from(ev.len())
                .unwrap_or(u32::MAX)
                .saturating_sub(1),
        ));
    }

    let mut starts: u32 = 0;
    let mut ends: u32 = 0;
    // Double-ron pending slot: (claimer, discarder) of the last ron on the
    // still-live discard, exactly like `prescan`/`log_replay`.
    let mut pending_ron: Option<(u8, u8)> = None;

    for (idx, e) in ev.iter().enumerate() {
        let kind = e.kind;
        if kind == KIND_OTHER {
            return Err(GateReject::new(
                game_idx,
                u32::try_from(idx).unwrap_or(u32::MAX),
                REASON_UNKNOWN_KIND,
            ));
        }
        if kind == KIND_START {
            // Oracle-exact count: bare `start` passes the boundary check but
            // does NOT satisfy the exactly-one rule (decode.py counts the
            // three canonical spellings only).
            if !span_is_bare_start_alias(e.span) {
                starts += 1;
            }
        } else if kind == KIND_END
            && !span_is_bare_end_alias(e.span) {
                ends += 1;
            }

        if kind == KIND_DORA {
            // Bare kan-dora marker quarantines the whole game (fail closed
            // before any row is built). Ingest normalizes absent-or-empty to
            // `None`; the `is_empty` arm is defense-in-depth.
            let bare = match e.dora_span {
                None => true,
                Some(s) => s.is_empty(),
            };
            if bare {
                return Ok(SampleDecision::SkipQuarantine(QuarantineStub::new(
                    game_idx,
                    u32::try_from(idx).unwrap_or(u32::MAX),
                    REASON_BARE_DORA,
                )));
            }
            // `dora` is transparent for ron-pending (live `continue`).
        } else if kind == KIND_HORA {
            let a = e.actor;
            let t = e.target;
            if a <= 3 && t <= 3 && a != t {
                if let Some((prev_a, prev_t)) = pending_ron
                    && prev_t == t
                    && prev_a != a
                {
                    return Ok(SampleDecision::SkipQuarantine(QuarantineStub::new(
                        game_idx,
                        u32::try_from(idx).unwrap_or(u32::MAX),
                        REASON_DOUBLE_RON,
                    )));
                }
                pending_ron = Some((a, t));
            } else {
                // Self-draw (`target == actor`) or unusable seats: mirror the
                // live fallthrough (reset the slot, arm nothing). Seat-shape
                // failures are the walk's turn-order reject, never the gate's.
                pending_ron = None;
            }
            // Quad SHAPE only (scores-vs-delta + han/fu are S6-cold, G4).
            if !hora_span_has_quad(e.span) {
                return Ok(SampleDecision::SkipQuarantine(QuarantineStub::new(
                    game_idx,
                    u32::try_from(idx).unwrap_or(u32::MAX),
                    REASON_HORA_SHAPE,
                )));
            }
        } else if kind == KIND_START_KYOKU {
            pending_ron = None;
        } else if kind == KIND_REACH_ACCEPTED {
            // Transparent: the discard offer (and any ron pending) survives.
        } else {
            // Every other significant event — discards, claims, draws,
            // kyoku terminals, and reserved-transparent 16 — ends ron pending,
            // exactly like the live prescan's trailing `pending_ron = None`.
            pending_ron = None;
        }
    }

    if starts != 1 || ends != 1 {
        return Err(GateReject::start_end_count(game_idx));
    }
    Ok(SampleDecision::Sample)
}

/// 136-permutation check for ingest's inline wall capture (stack-only,
/// zero-alloc): exactly the ids `0..135`, each once. Physical ids group in
/// fours per logical tile, so the permutation implies the 4-each logical
/// counts the oracle checks.
pub fn is_wall_perm136(ids: &[u8; 136]) -> bool {
    let mut seen = [false; 136];
    for &v in ids.iter() {
        if v >= 136 || seen[v as usize] {
            return false;
        }
        seen[v as usize] = true;
    }
    true
}

// ---------------------------------------------------------------------------
// Zero-alloc span scanners (byte loops only; no parse, no heap).
// ---------------------------------------------------------------------------

/// The event's `"type"` string value, if structurally present.
fn span_type_value(span: &[u8]) -> Option<&[u8]> {
    let mut i = 0usize;
    while i + 6 <= span.len() {
        if &span[i..i + 6] == b"\"type\"" {
            let mut j = skip_ws(span, i + 6);
            if j < span.len() && span[j] == b':' {
                j = skip_ws(span, j + 1);
                if j < span.len() && span[j] == b'"' {
                    j += 1;
                    let s = j;
                    while j < span.len() && span[j] != b'"' {
                        if span[j] == b'\\' {
                            j += 1;
                        }
                        j += 1;
                    }
                    if j < span.len() {
                        return Some(&span[s..j]);
                    }
                    return None;
                }
            }
        }
        i += 1;
    }
    None
}

/// True iff the start event uses the bare `start` alias (boundary-ok, but
/// excluded from the oracle's exactly-one canonical count).
fn span_is_bare_start_alias(span: &[u8]) -> bool {
    matches!(span_type_value(span), Some(v) if v == b"start")
}

/// True iff the end event uses the bare `end` alias (same rule).
fn span_is_bare_end_alias(span: &[u8]) -> bool {
    matches!(span_type_value(span), Some(v) if v == b"end")
}

#[inline]
fn skip_ws(s: &[u8], mut j: usize) -> usize {
    while j < s.len() && (s[j] == b' ' || s[j] == b'\t' || s[j] == b'\n' || s[j] == b'\r') {
        j += 1;
    }
    j
}

/// Parses `-?(0|[1-9][0-9]{0,18})` at `s[j..]`; returns the end position.
/// The 19-digit cap keeps accepted magnitudes inside i64 (mirrors the
/// oracle's `int(d)` + `as_i64` fail-closed edge); the leading-zero rule
/// mirrors JSON (the oracle never observes `007` — decode rejects it first).
fn parse_json_int(s: &[u8], mut j: usize) -> Option<usize> {
    if j < s.len() && s[j] == b'-' {
        j += 1;
    }
    let d0 = j;
    while j < s.len() && s[j].is_ascii_digit() {
        j += 1;
    }
    let run = j - d0;
    if run == 0 || run > 19 {
        return None;
    }
    if run > 1 && s[d0] == b'0' {
        return None;
    }
    Some(j)
}

/// True iff `pos` starts a well-shaped 4-int `deltas` array value.
fn match_deltas_quad_at(span: &[u8], pos: usize) -> bool {
    let mut j = skip_ws(span, pos);
    if j >= span.len() || span[j] != b'[' {
        return false;
    }
    j = skip_ws(span, j + 1);
    for k in 0..4 {
        let Some(nj) = parse_json_int(span, j) else {
            return false;
        };
        j = skip_ws(span, nj);
        if k < 3 {
            if j >= span.len() || span[j] != b',' {
                return false;
            }
            j = skip_ws(span, j + 1);
        }
    }
    j < span.len() && span[j] == b']'
}

/// True iff the hora line carries a `"deltas":[i,i,i,i]` quad (live
/// `deltas.len() == 4 && all as_i64`, zero-alloc). Scanning continues past a
/// non-matching occurrence so a `"deltas"` substring embedded in some other
/// string value can never false-quarantine a well-formed line.
fn hora_span_has_quad(span: &[u8]) -> bool {
    let mut i = 0usize;
    while i + 8 <= span.len() {
        if &span[i..i + 8] == b"\"deltas\"" {
            let mut j = skip_ws(span, i + 8);
            if j < span.len() && span[j] == b':' {
                j = skip_ws(span, j + 1);
                if match_deltas_quad_at(span, j) {
                    return true;
                }
            }
        }
        i += 1;
    }
    false
}

// ---------------------------------------------------------------------------
// Tests (unit scope only; oracle histogram agreement runs over real ingest
// output — these pin every gate-owned rule + the F4 table).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::{
        KIND_ANKAN, KIND_CHI, KIND_DAHAI, KIND_DAIMINKAN, KIND_END, KIND_HORA, KIND_KAKAN,
        KIND_PON, KIND_REACH, KIND_REACH_ACCEPTED, KIND_RYUKYOKU, KIND_START, KIND_START_KYOKU,
        KIND_TSUMO, NO_PAI, NO_SEAT,
    };

    fn tev(kind: u8, span: &'static [u8]) -> StackedEvent<'static> {
        StackedEvent {
            kind,
            actor: NO_SEAT,
            target: NO_SEAT,
            pai: NO_PAI,
            consumed: ([0; 4], 0),
            tsumogiri: false,
            span,
            dora_span: None,
            wall_span: None,
        }
    }

    fn hora(actor: u8, target: u8, span: &'static [u8]) -> StackedEvent<'static> {
        StackedEvent {
            kind: KIND_HORA,
            actor,
            target,
            pai: NO_PAI,
            consumed: ([0; 4], 0),
            tsumogiri: false,
            span,
            dora_span: None,
            wall_span: None,
        }
    }

    fn dora(span: &'static [u8], marker: Option<&'static [u8]>) -> StackedEvent<'static> {
        StackedEvent {
            kind: KIND_DORA,
            actor: NO_SEAT,
            target: NO_SEAT,
            pai: NO_PAI,
            consumed: ([0; 4], 0),
            tsumogiri: false,
            span,
            dora_span: marker,
            wall_span: None,
        }
    }

    const START: &[u8] = br#"{"type":"start_game"}"#;
    const START_ALIAS: &[u8] = br#"{"type":"start"}"#;
    const END: &[u8] = br#"{"type":"end_game"}"#;
    const END_ALIAS: &[u8] = br#"{"type":"end"}"#;
    const KYOKU: &[u8] = br#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1m"],["2m"],["3m"],["4m"]]}"#;
    const TSUMO: &[u8] = br#"{"type":"tsumo","actor":0,"pai":"E"}"#;
    const DAHAI: &[u8] = br#"{"type":"dahai","actor":0,"pai":"E","tsumogiri":true}"#;
    const DORA_BARE: &[u8] = br#"{"type":"dora"}"#;
    const DORA_MARKED: &[u8] = br#"{"type":"dora","dora_marker":"4m"}"#;
    const HORA_RON_A: &[u8] =
        br#"{"type":"hora","actor":1,"target":0,"deltas":[8000,8000,-8000,-8000]}"#;
    const HORA_RON_B: &[u8] =
        br#"{"type":"hora","actor":2,"target":0,"deltas":[8000,-8000,8000,-8000]}"#;
    const HORA_RON_C: &[u8] =
        br#"{"type":"hora","actor":1,"target":3,"deltas":[0,0,0,0]}"#;
    const HORA_TSUMO: &[u8] =
        br#"{"type":"hora","actor":0,"target":0,"deltas":[-8000,8000,0,0]}"#;
    const OTHER: &[u8] = br#"{"type":"frobnicate"}"#;

    fn sample_game() -> [StackedEvent<'static>; 3] {
        [tev(KIND_START, START), tev(KIND_START_KYOKU, KYOKU), tev(KIND_END, END)]
    }

    #[test]
    fn kind_values_match_hub_lut() {
        use crate::ingest::{
            KIND_DORA as D, KIND_END_KYOKU as EK, KIND_OTHER as O, KIND_REACH_ACCEPTED as RA,
            KIND_RYUKYOKU as R, KIND_START as S, KIND_START_KYOKU as SK,
            KIND_TRANSPARENT_OTHER as T, KIND_TSUMO as TS,
        };
        assert_eq!((S, KIND_END, KIND_DAHAI, KIND_CHI, KIND_PON, KIND_DAIMINKAN), (0, 1, 2, 3, 4, 5));
        assert_eq!((KIND_ANKAN, KIND_KAKAN, KIND_HORA, D), (6, 7, 8, 9));
        assert_eq!((KIND_REACH, RA, TS, SK, EK, R, T, O), (10, 11, 12, 13, 14, 15, 16, 17));
    }

    #[test]
    fn should_sample_is_row_types_7_incl_hora() {
        let mut n = 0;
        for kind in 0..=17u8 {
            let want = matches!(kind, 2..=8);
            assert_eq!(should_sample(kind), want, "kind {kind}");
            n += want as u32;
        }
        assert_eq!(n, 7, "exactly dahai/chi/pon/daiminkan/ankan/kakan/hora");
        assert!(should_sample(KIND_HORA));
        assert!(!should_sample(KIND_REACH));
        assert!(!should_sample(KIND_OTHER));
    }

    #[test]
    fn reason_names_cover_closed_vocab() {
        let names: Vec<_> = (0..=8u8).map(reason_name).collect();
        assert_eq!(
            names,
            [
                "ok",
                "framing",
                "unknown-event",
                "framing",
                "framing",
                "wall-bearing",
                "bare-dora",
                "double-ron",
                "turn-order"
            ]
        );
        assert_eq!(reason_name(255), "other");
    }

    #[test]
    fn minimal_game_samples() {
        assert_eq!(gate_game(0, &sample_game()), Ok(SampleDecision::Sample));
    }

    #[test]
    fn empty_rejects_framing() {
        assert_eq!(gate_game(9, &[]), Err(GateReject::new(9, 0, REASON_FRAMING)));
    }

    #[test]
    fn boundary_blames_first_and_last() {
        let bad_first = [tev(KIND_TSUMO, TSUMO), tev(KIND_END, END)];
        assert_eq!(
            gate_game(0, &bad_first),
            Err(GateReject::new(0, 0, REASON_BOUNDARY))
        );
        let bad_last = [tev(KIND_START, START), tev(KIND_TSUMO, TSUMO)];
        assert_eq!(
            gate_game(0, &bad_last),
            Err(GateReject::new(0, 1, REASON_BOUNDARY))
        );
    }

    #[test]
    fn start_end_counts_oracle_exact() {
        // Doubled canonical start.
        let doubled = [
            tev(KIND_START, START),
            tev(KIND_START, START),
            tev(KIND_END, END),
        ];
        assert_eq!(gate_game(0, &doubled), Err(GateReject::start_end_count(0)));
        // Doubled canonical end.
        let doubled_end = [
            tev(KIND_START, START),
            tev(KIND_END, END),
            tev(KIND_END, END),
        ];
        assert_eq!(gate_game(0, &doubled_end), Err(GateReject::start_end_count(0)));
        // Bare `start` alias alone: boundary-ok, count fails (oracle rejects).
        let bare_only = [
            tev(KIND_START, START_ALIAS),
            tev(KIND_START_KYOKU, KYOKU),
            tev(KIND_END, END),
        ];
        assert_eq!(gate_game(0, &bare_only), Err(GateReject::start_end_count(0)));
        // Bare `start` + canonical `start_game`: oracle ACCEPTS — gate agrees.
        let bare_plus_canon = [
            tev(KIND_START, START_ALIAS),
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            tev(KIND_END, END),
        ];
        assert_eq!(gate_game(0, &bare_plus_canon), Ok(SampleDecision::Sample));
        // Bare `end` alias alone: count fails.
        let bare_end = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            tev(KIND_END, END_ALIAS),
        ];
        assert_eq!(gate_game(0, &bare_end), Err(GateReject::start_end_count(0)));
    }

    #[test]
    fn unknown_kind_rejects_invalid_data() {
        let ev = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            tev(KIND_OTHER, OTHER),
            tev(KIND_END, END),
        ];
        let err = gate_game(4, &ev).unwrap_err();
        assert_eq!(err, GateReject::new(4, 2, REASON_UNKNOWN_KIND));
        let io: std::io::Error = err.into();
        assert_eq!(io.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn bare_dora_skips_whole_game() {
        // Absent marker (q-bare-dora shape).
        let ev = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            dora(DORA_BARE, None),
            tev(KIND_END, END),
        ];
        assert_eq!(
            gate_game(7, &ev),
            Ok(SampleDecision::SkipQuarantine(QuarantineStub::new(
                7,
                2,
                REASON_BARE_DORA
            )))
        );
        // Empty marker normalizes the same (defense-in-depth).
        let ev_empty = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            dora(DORA_BARE, Some(b"")),
            tev(KIND_END, END),
        ];
        assert_eq!(
            gate_game(7, &ev_empty),
            Ok(SampleDecision::SkipQuarantine(QuarantineStub::new(
                7,
                2,
                REASON_BARE_DORA
            )))
        );
        // Present marker samples.
        let ok = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            dora(DORA_MARKED, Some(b"4m")),
            tev(KIND_END, END),
        ];
        assert_eq!(gate_game(7, &ok), Ok(SampleDecision::Sample));
    }

    #[test]
    fn double_ron_skips_at_second_winner() {
        // q-double-ron shape: ron(1→0), ron(2→0) on the live discard.
        let ev = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            tev(KIND_TSUMO, TSUMO),
            tev(KIND_DAHAI, DAHAI),
            hora(1, 0, HORA_RON_A),
            hora(2, 0, HORA_RON_B),
            tev(KIND_END, END),
        ];
        assert_eq!(
            gate_game(3, &ev),
            Ok(SampleDecision::SkipQuarantine(QuarantineStub::new(
                3,
                5,
                REASON_DOUBLE_RON
            )))
        );
    }

    #[test]
    fn ron_pending_matches_live_transparency() {
        let base = |tail: Vec<StackedEvent<'static>>| {
            let mut ev = vec![
                tev(KIND_START, START),
                tev(KIND_START_KYOKU, KYOKU),
                tev(KIND_TSUMO, TSUMO),
                tev(KIND_DAHAI, DAHAI),
                hora(1, 0, HORA_RON_A),
            ];
            ev.extend(tail);
            ev.push(tev(KIND_END, END));
            ev
        };
        // Single ron samples.
        assert_eq!(gate_game(0, &base(vec![])), Ok(SampleDecision::Sample));
        // Rons on different discards sample.
        let diff_target = base(vec![hora(2, 3, HORA_RON_C)]);
        assert_eq!(gate_game(0, &diff_target), Ok(SampleDecision::Sample));
        // Self-draw (target == actor) resets the slot: ron, tsumo, ron samples.
        let tsumo_between = base(vec![
            hora(0, 0, HORA_TSUMO),
            hora(2, 0, HORA_RON_B),
        ]);
        assert_eq!(gate_game(0, &tsumo_between), Ok(SampleDecision::Sample));
        // reach_accepted is transparent: ron, accepted, ron still double-ron.
        let accepted_between = base(vec![
            tev(KIND_REACH_ACCEPTED, br#"{"type":"reach_accepted","actor":1}"#),
            hora(2, 0, HORA_RON_B),
        ]);
        assert!(matches!(
            gate_game(0, &accepted_between),
            Ok(SampleDecision::SkipQuarantine(s)) if s.reason == REASON_DOUBLE_RON
        ));
        // A discard between rons clears the slot.
        let dahai_between = base(vec![
            tev(KIND_DAHAI, DAHAI),
            hora(2, 0, HORA_RON_B),
        ]);
        assert_eq!(gate_game(0, &dahai_between), Ok(SampleDecision::Sample));
        // New kyoku clears the slot.
        let kyoku_between = base(vec![
            tev(KIND_START_KYOKU, KYOKU),
            hora(2, 0, HORA_RON_B),
        ]);
        assert_eq!(gate_game(0, &kyoku_between), Ok(SampleDecision::Sample));
        // Ryukyoku clears the slot.
        let ryu_between = base(vec![
            tev(KIND_RYUKYOKU, br#"{"type":"ryukyoku","reason":"yama9"}"#),
            hora(2, 0, HORA_RON_B),
        ]);
        assert_eq!(gate_game(0, &ryu_between), Ok(SampleDecision::Sample));
    }

    #[test]
    fn hora_quad_shape_table() {
        // (span, well-shaped?)
        let cases: &[(&[u8], bool)] = &[
            (br#"{"type":"hora","actor":1,"target":0,"deltas":[8000,8000,-8000,-8000]}"#, true),
            (br#"{"type":"hora","actor":0,"target":0,"deltas": [ 0 , 0 , 0 , 0 ]}"#, true),
            (br#"{"type":"hora","actor":1,"target":0}"#, false),
            (br#"{"type":"hora","actor":1,"target":0,"deltas":[1,2,3]}"#, false),
            (br#"{"type":"hora","actor":1,"target":0,"deltas":[1,2,3,4,5]}"#, false),
            (br#"{"type":"hora","actor":1,"target":0,"deltas":[1,2,"3",4]}"#, false),
            (br#"{"type":"hora","actor":1,"target":0,"deltas":[1.0,2,3,4]}"#, false),
            (br#"{"type":"hora","actor":1,"target":0,"deltas":[]}"#, false),
            (br#"{"type":"hora","actor":1,"target":0,"deltas":"x"}"#, false),
        ];
        for (span, want) in cases {
            assert_eq!(hora_span_has_quad(span), *want, "{}", String::from_utf8_lossy(span));
        }
    }

    #[test]
    fn hora_quad_failure_skips_not_errors() {
        let ev = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            hora(1, 0, br#"{"type":"hora","actor":1,"target":0}"#),
            tev(KIND_END, END),
        ];
        assert_eq!(
            gate_game(0, &ev),
            Ok(SampleDecision::SkipQuarantine(QuarantineStub::new(
                0,
                2,
                REASON_HORA_SHAPE
            )))
        );
    }

    #[test]
    fn wall_perm136_exact() {
        let mut ids = [0u8; 136];
        for (i, v) in ids.iter_mut().enumerate() {
            *v = u8::try_from(i).unwrap();
        }
        assert!(is_wall_perm136(&ids));
        let mut dup = ids;
        dup[135] = 0;
        assert!(!is_wall_perm136(&dup));
        let mut big = ids;
        big[0] = 200;
        assert!(!is_wall_perm136(&big));
    }

    #[test]
    fn reject_stub_round_trip() {
        let r = GateReject::bad_tile(11, 40_000);
        assert_eq!(r.stub(), QuarantineStub::new(11, 40_000, REASON_FRAMING));
        // Saturation at u16::MAX (games past 64K events).
        let big = GateReject::new(1, u32::MAX, REASON_BOUNDARY);
        assert_eq!(big.event_idx, u16::MAX);
        assert_eq!(big.stub().event_idx, u16::MAX);
    }

    #[test]
    fn ingest_constructors_map_to_framing_family() {
        let g = 5u32;
        assert_eq!(GateReject::empty(g).reason, REASON_FRAMING);
        assert_eq!(GateReject::missing_newline(g).reason, REASON_FRAMING);
        assert_eq!(GateReject::blank_line(g, 1).reason, REASON_BLANK_LINE);
        assert_eq!(GateReject::blank_line(g, 1).event_idx, 1);
        assert_eq!(GateReject::utf8(g).reason, REASON_FRAMING);
        assert_eq!(GateReject::decompress(g).reason, REASON_FRAMING);
        assert_eq!(GateReject::invalid_json(g, 2).reason, REASON_FRAMING);
        assert_eq!(GateReject::missing_type(g, 2).reason, REASON_UNKNOWN_KIND);
        assert_eq!(GateReject::bad_type(g, 2).reason, REASON_UNKNOWN_KIND);
        assert_eq!(GateReject::bad_boundary(g, 0).reason, REASON_BOUNDARY);
        assert_eq!(GateReject::start_end_count(g).reason, REASON_BOUNDARY);
        assert_eq!(GateReject::bad_tile(g, 3).reason, REASON_FRAMING);
        assert_eq!(GateReject::bad_wall(g).reason, REASON_WALL_PERM);
        for r in [
            GateReject::empty(g),
            GateReject::missing_newline(g),
            GateReject::blank_line(g, 1),
            GateReject::utf8(g),
            GateReject::decompress(g),
            GateReject::invalid_json(g, 2),
            GateReject::missing_type(g, 2),
            GateReject::bad_type(g, 2),
            GateReject::bad_boundary(g, 0),
            GateReject::start_end_count(g),
            GateReject::bad_tile(g, 3),
            GateReject::bad_wall(g),
        ] {
            let io: std::io::Error = r.into();
            assert_eq!(io.kind(), std::io::ErrorKind::InvalidData);
        }
    }

    /// F4 gate-verdict table (mirrors
    /// `crates/tests/fixtures/s4/*.jsonl` kind sequences;
    /// blank-line framing belongs to ingest and is covered by its contract).
    /// Gate quarantines 2 + framing-errors 3 here; the walk adds turn-order +
    /// tile-conservation for the oracle's 7 total (G5 histogram agreement).
    #[test]
    fn f4_gate_verdict_table() {
        // good-a / good-b: rows present → Sample (row COUNT is the walk's job).
        let good = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            tev(KIND_TSUMO, TSUMO),
            tev(KIND_DAHAI, DAHAI),
            tev(KIND_END, END),
        ];
        assert_eq!(gate_game(0, &good), Ok(SampleDecision::Sample));
        // q-wall-bearing: valid wall-less-shape game, zero decisions → Sample.
        let wall_bearing = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            tev(KIND_END, END),
        ];
        assert_eq!(gate_game(1, &wall_bearing), Ok(SampleDecision::Sample));
        // q-turn-order (tsumo before start_kyoku): gate Samples; walk rejects.
        let turn_order = [
            tev(KIND_START, START),
            tev(KIND_TSUMO, TSUMO),
            tev(KIND_END, END),
        ];
        assert_eq!(gate_game(2, &turn_order), Ok(SampleDecision::Sample));
        // q-tile-conservation (tsumo without pai): gate Samples; walk rejects.
        let conservation = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            tev(KIND_TSUMO, br#"{"type":"tsumo","actor":0}"#),
            tev(KIND_END, END),
        ];
        assert_eq!(gate_game(3, &conservation), Ok(SampleDecision::Sample));
        // q-bare-dora → Skip(6) at event 2.
        let bare = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            dora(DORA_BARE, None),
            tev(KIND_END, END),
        ];
        assert_eq!(
            gate_game(4, &bare),
            Ok(SampleDecision::SkipQuarantine(QuarantineStub::new(
                4,
                2,
                REASON_BARE_DORA
            )))
        );
        // q-double-ron → Skip(7) at event 5.
        let dron = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            tev(KIND_TSUMO, TSUMO),
            tev(KIND_DAHAI, DAHAI),
            hora(1, 0, HORA_RON_A),
            hora(2, 0, HORA_RON_B),
            tev(KIND_END, END),
        ];
        assert_eq!(
            gate_game(5, &dron),
            Ok(SampleDecision::SkipQuarantine(QuarantineStub::new(
                5,
                5,
                REASON_DOUBLE_RON
            )))
        );
        // q-truncated (no end_game) → Err boundary at last index.
        let truncated = [tev(KIND_START, START), tev(KIND_START_KYOKU, KYOKU)];
        assert_eq!(
            gate_game(6, &truncated),
            Err(GateReject::new(6, 1, REASON_BOUNDARY))
        );
        // q-unknown-event → Err unknown-kind at event 2.
        let unknown = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            tev(KIND_OTHER, OTHER),
            tev(KIND_END, END),
        ];
        assert_eq!(
            gate_game(7, &unknown),
            Err(GateReject::new(7, 2, REASON_UNKNOWN_KIND))
        );
    }

    #[test]
    fn hot_path_needs_no_heap_inputs() {
        // Every input below is `&'static` bytes + stack structs: the Sample /
        // Skip verdicts are reachable with zero heap allocation live at the
        // call. (Compile-time shape proof; the grep audit pins no alloc ops
        // in non-test code.)
        let ev = sample_game();
        assert!(gate_game(0, &ev).unwrap().is_sample());
        let skip = [
            tev(KIND_START, START),
            tev(KIND_START_KYOKU, KYOKU),
            dora(DORA_BARE, None),
            tev(KIND_END, END),
        ];
        assert!(gate_game(0, &skip).unwrap().quarantine().is_some());
    }
}
