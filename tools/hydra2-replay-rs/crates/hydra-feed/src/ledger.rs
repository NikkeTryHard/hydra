//! S3 walk state — u8 ledger DIRECT to planes (`feed::ledger`).
//!
//! u8 ledger DIRECT to minimal hot planes (§6.3 shapes, verbatim names;
//! aka shares its type, offers/shapes/furiten are type-level reads).
//!
//! Kills (vs `hydra-shard/src/decisions.rs`, read-only reference):
//! - `reported_hand` string roundtrip (:617-646, 2× `HashMap<String,..>` +
//!   `sort_unstable` per row, re-run per `capture_row`/`seat_view`) →
//!   [`Ledger::hands`] copy-slot presence + [`collapse present`](collapse_slot)
//!   canonical pool-first derivation (2-3 L1 loads, no heap).
//! - `format!(d/h)` decision/round ids (:810-811) + `game_id.clone` →
//!   numeric keys only ([`GameCtx::game_idx`] `u32`, `walk_game` seq `u16`).
//! - `wall_digest.clone` + `histories.clone` + `mask.to_vec` (:870-877) →
//!   [`GameCtx`] borrowed `Arc<str>` digest (never cloned/row) + interned
//!   `u8` history kinds + stack `[u32; 32]` legal.
//! - per-row `Vec<String>` concealed/drawn/dora renders → `u8`/`i8` planes.
//!
//! Copy-slot model (oracle-exact, see `walk.rs` for the proof sketch):
//! `hands[t][c]` is presence (0/1) of copy `c` of type `t` (`t = id / 4`).
//! Takes are allocated monotonically per string class
//! ([`Ledger::used_n`]/[`used_aka`], never reused — mirrors
//! `KyokuTrack::take_next` pool order), so true-take identity
//! ([`Ledger::drawn_live`], riichi-forced discards) and pool-first collapse
//! (reports, offers) both agree with the string ledger on every legal log.
//! Akaillis: `"5mr"`/`"0m"` occupy slot 0 of types 4/13/22; plain `"5x"`
//! occupy slots 1..3 in order; every other string fills slots 0..3.
//! Norm folding (aka shares its type) makes offers, shapes, and furiten
//! pure type-level reads.
//!
//! Feed-only deps: `crate::{ingest, gate, tiles}` + `std::sync::Arc` for the
//! borrowed digest. NEVER `hydra-shard` (DAG one-way), NEVER sha2/hash hot.

use std::sync::Arc;

/// Action-table artifact this walk's closed-form ids are pinned to
/// (`configs/contracts/action_table_v1.json`): 6792 entries.
pub const ACTION_TABLE_LEN: u32 = 6792;
/// Pinned content digest of the artifact above (payload.digest).
pub const ACTION_TABLE_DIGEST: &str =
    "sha256:7b55693428384713f6a6ab7292f57657259c2ca9f05139944d4c8c6197ae76e8";

// ---------------------------------------------------------------------------
// Action-kind codes (numeric; replace `ChosenAction::kind_str()`).
// Order matches `ACTION_KIND_ORDINALS` in the cold reference.
// ---------------------------------------------------------------------------

/// Action kind: `pass`.
pub const ACT_PASS: u8 = 0;
/// Action kind: `discard`.
pub const ACT_DISCARD: u8 = 1;
/// Action kind: `tsumogiri`.
pub const ACT_TSUMOGIRI: u8 = 2;
/// Action kind: `riichi_discard`.
pub const ACT_RIICHI_DISCARD: u8 = 3;
/// Action kind: `chi`.
pub const ACT_CHI: u8 = 4;
/// Action kind: `pon`.
pub const ACT_PON: u8 = 5;
/// Action kind: `daiminkan`.
pub const ACT_DAIMINKAN: u8 = 6;
/// Action kind: `ankan`.
pub const ACT_ANKAN: u8 = 7;
/// Action kind: `kakan`.
pub const ACT_KAKAN: u8 = 8;
/// Action kind: `ron`.
pub const ACT_RON: u8 = 9;
/// Action kind: `tsumo`.
pub const ACT_TSUMO: u8 = 10;

// ---------------------------------------------------------------------------
// Walk reject: zero-alloc quarantine triple (sink-join compatible).
// ---------------------------------------------------------------------------

/// Walk failure reason: `turn-order` (drawer/sequence/header violations).
pub const WALK_TURN_ORDER: u8 = 10;
/// Walk failure reason: `tile-conservation` (ledger/copy violations).
pub const WALK_TILE_CONSERVATION: u8 = 11;
/// Walk failure reason: `claim-no-offer` (claim/ron outside a live offer).
pub const WALK_CLAIM_NO_OFFER: u8 = 12;
/// Walk failure reason: `draw-past-wall` (draw at/after wall exhaustion).
pub const WALK_DRAW_PAST_WALL: u8 = 13;
/// Walk failure reason: `kyushu-ambiguous` (reason-less ryukyoku, no shape).
pub const WALK_KYUSHU_AMBIGUOUS: u8 = 14;
/// Walk failure reason: `unmapped-ryukyoku-reason` (unknown reason string).
pub const WALK_UNMAPPED_RYUKYOKU: u8 = 15;
/// Walk failure reason: `engine-desync` (logged action not offered).
pub const WALK_ENGINE_DESYNC: u8 = 16;
/// Walk failure reason: `unknown-event` (unmapped stacked kind).
pub const WALK_UNKNOWN_EVENT: u8 = 17;
/// Walk failure reason: `bare-dora` (defense-in-depth; gate pre-empts).
pub const WALK_BARE_DORA: u8 = 18;
/// Walk failure reason: `double-ron` (defense-in-depth; gate pre-empts).
pub const WALK_DOUBLE_RON: u8 = 19;
/// Walk failure reason: `action-id-unresolved` (chosen not in table).
pub const WALK_ACTION_UNRESOLVED: u8 = 20;
/// Walk failure reason: `past-terminal` (row/end events past a hora with no
/// closing end_kyoku; the wall oracle goes terminal there).
pub const WALK_PAST_TERMINAL: u8 = 21;

/// Cold render of a walk reason for sink / lineage / histogram joins (G5
/// closed vocabulary; never called on the row hot path).
pub const fn walk_reason_name(reason: u8) -> &'static str {
    match reason {
        WALK_TURN_ORDER => "turn-order",
        WALK_TILE_CONSERVATION => "tile-conservation",
        WALK_CLAIM_NO_OFFER => "claim-no-offer",
        WALK_DRAW_PAST_WALL => "draw-past-wall",
        WALK_KYUSHU_AMBIGUOUS => "kyushu-ambiguous",
        WALK_UNMAPPED_RYUKYOKU => "unmapped-ryukyoku-reason",
        WALK_ENGINE_DESYNC => "engine-desync",
        WALK_UNKNOWN_EVENT => "unknown-event",
        WALK_BARE_DORA => "bare-dora",
        WALK_DOUBLE_RON => "double-ron",
        WALK_ACTION_UNRESOLVED => "action-id-unresolved",
        WALK_PAST_TERMINAL => "past-terminal",
        _ => "other",
    }
}

/// Classified walk failure: numeric triple only, never a `String` hot.
///
/// Joins the sink on `(game_idx, event_idx, reason)` exactly like
/// [`crate::gate::QuarantineStub`]; [`walk_reason_name`] renders the cold
/// oracle string (`turn-order`, `tile-conservation`, …) for G5 agreement.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WalkReject {
    /// Stable game index (sink join key).
    pub game_idx: u32,
    /// Offending event index, saturated at `u16::MAX`.
    pub event_idx: u16,
    /// Reason bucket (`WALK_*`).
    pub reason: u8,
}

impl WalkReject {
    /// Build a reject, saturating oversized indices.
    pub const fn new(game_idx: u32, event_idx: usize, reason: u8) -> Self {
        let idx = if event_idx > u16::MAX as usize {
            u16::MAX
        } else {
            event_idx as u16
        };
        Self {
            game_idx,
            event_idx: idx,
            reason,
        }
    }

    /// Cold lineage string for the quarantine leg (never hot).
    pub const fn name(self) -> &'static str {
        walk_reason_name(self.reason)
    }
}
/// One open meld: numeric kind (`ACT_*`), owner seat, sorted takes + len.
///
/// `Copy` so responder scans and mask builds never allocate. A kakan never
/// installs a record (the prior pon stands, mirroring the oracle); ankan
/// keeps its full sorted block.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Meld {
    /// Meld kind (`ACT_CHI/PON/DAIMINKAN/ANKAN`).
    pub kind: u8,
    /// Owning seat.
    pub owner: u8,
    /// Sorted takes + length.
    pub tiles: ([u8; 4], u8),
}

impl Meld {
    /// Empty slot (len 0, never emitted).
    pub const EMPTY: Self = Self {
        kind: 0xFF,
        owner: 0xFF,
        tiles: ([0u8; 4], 0),
    };
}

/// S3 u8 ledger: whole table state, zero heap.
///
/// Plan-named fields keep plan order/types; `*_len` counters and the
/// take-allocation / turn-machine fields are the minimal extras the walk
/// needs (documented at each site). One `Ledger` walks one game; kyoku
/// boundaries overwrite (never realloc).
#[derive(Clone, Debug)]
pub struct Ledger {
/// `hands` note: plan §6.3 sketches `hands[[u8;4];34]` — the per-seat
/// slice. The ledger holds one slice per seat (`hands[seat][type][copy]`,
/// stored take id + 1 with 0 == empty); a seat-less ledger could not walk
/// four holdings, while `melds`/`rivers` are already per-seat in the plan.
/// Stored takes are the TRUE allocated copies (never pool-first twins), so
/// offers/chosen report the same ids the oracle encodes. Global take
/// allocation (conservation) lives in `used_n`/`used_aka`, monotonic and
    /// Concealed takes per seat: `hands[seat][type][copy]`, stored take id + 1 (0 == empty).
    pub hands: [[[u8; 4]; 34]; 4],
    /// Report folding: wall-less rows render pool-first canonical takes (the
    /// SIM oracle reports canonical copies); walled rows report true takes
    /// (the wall oracle encodes held copies). State (allocation, removal,
    /// conservation) is true takes in both modes; only reports fold.
    pub fold_reports: bool,
    /// Open melds per seat (at most 4).
    pub melds: [[Meld; 4]; 4],
    /// Meld count per seat.
    pub meld_lens: [u8; 4],
 /// Discard rivers (true take ids) per seat.
    pub rivers: [[u8; 32]; 4],
    pub rivers_len: [u8; 4],
    /// Live-wall draws this kyoku (wall gate reads `70 - draws`).
    pub draws: u8,
    /// Dora indicator physical ids, `-1` tail (NEVER padded otherwise).
    pub dora: [i8; 5],
    /// Revealed indicator count (extras past 5 are row-invisible).
    pub dora_len: u8,
    /// Tracked scores (kyoku install, −1000 per riichi, hora/ryukyoku deltas).
    pub scores: [i32; 4],
    /// Seat winds as wind tile types (27=E … 30=N).
    pub seat_wind: [u8; 4],
    /// Round wind as wind tile type.
    pub round_wind: u8,
    /// Riichi declared per seat (0/1).
    pub riichi: [u8; 4],
    /// Dealer seat (oya).
    pub dealer: u8,
    /// Last emitted row phase (`PHASE_*`).
    pub phase: u8,
    // -- turn machine (kyoku-scoped; reset per `start_kyoku`) --
    /// Seat owning the next draw decision (`None` until the first draw).
    pub drawer: Option<u8>,
    /// Expected drawer after kan (rinshan target).
    pub exp_drawer: Option<u8>,
    /// A kan is pending its replacement draw.
    pub kan_pending: bool,
    /// Live kyoku installed (guards pre-kyoku decisions).
    pub kyoku_active: bool,
    // -- draw memory --
 /// True live take per seat (offers/chosen; stale across turns).
    pub drawn_col: [Option<u8>; 4],
    /// True live take per seat (riichi-forced answers; cleared on act).
    pub drawn_live: [Option<u8>; 4],
    // -- take allocation (monotonic per string class; mirrors take_next) --
    /// Non-aka takes consumed per type (cap 4; five-types cap 3 plain).
    pub used_n: [u8; 34],
    /// Aka takes consumed per five suit (m/p/s → 0/1/2; cap 1).
    pub used_aka: [u8; 3],
    // -- kyushu inference memory --
    /// Tehai type counts at install (norm-folded; aka shares its type).
    pub tehai_cnt: [u8; 34],
    /// First drawn take per seat (kyushu rule reads its type).
    pub first_draw: [Option<u8>; 4],
    /// Tsumo count per seat (kyushu needs exactly the first draw).
    pub tsumo_cnt: [u32; 4],
    /// Window snapshot scores (latest draw row's observed scores).
    pub snap_scores: [i32; 4],
    /// Window snapshot sticks (latest draw row's observed sticks).
    pub snap_sticks: u8,
    /// Honba count from the kyoku header.
    pub honba: u8,
    /// Riichi sticks on the table (header kyotaku + accepts this kyoku).
    pub sticks: u8,
    /// Hand number from the kyoku header (`kyoku` key).
    pub hand_number: u8,
    /// Kyoku ordinal counter (game-scoped; `round_index` reads pre-increment).
    pub kyoku_ordinal: u8,
    /// Round index for the current kyoku (0-based ordinal).
    pub round_index: u8,
    /// Reach accepted per seat (game-scoped? no: kyoku-scoped, reset per kyoku).
    pub accepted: [u8; 4],
    /// Ippatsu window open per seat.
    pub ippatsu: [u8; 4],
    /// Ron-pass marks per seat (temporary furiten).
    pub missed: [u8; 4],
    /// Kan calls this kyoku (ankan/daiminkan/kakan declarations).
    pub kan_count: u8,
    /// Last offer built a riichi candidate (can_riichi stash for mint).
    pub riichi_offered: bool,
    /// Last offer built a tsumo answer (can_tsumo stash for mint).
    pub tsumo_offered: bool,
}
impl Ledger {
    /// Zeroed ledger (caller installs kyoku state via the walk).
    pub const fn new() -> Self {
        Self {
            hands: [[[0u8; 4]; 34]; 4],
            fold_reports: false,
            melds: [[Meld::EMPTY; 4]; 4],
            meld_lens: [0u8; 4],
            rivers: [[0u8; 32]; 4],
            rivers_len: [0u8; 4],
            draws: 0,
            dora: [-1i8; 5],
            dora_len: 0,
            scores: [0i32; 4],
            seat_wind: [27u8; 4],
            round_wind: 27,
            riichi: [0u8; 4],
            dealer: 0,
            phase: 0,
            drawer: None,
            exp_drawer: None,
            kan_pending: false,
            kyoku_active: false,
            drawn_col: [None; 4],
            drawn_live: [None; 4],
            used_n: [0u8; 34],
            used_aka: [0u8; 3],
            tehai_cnt: [0u8; 34],
            first_draw: [None; 4],
            tsumo_cnt: [0u32; 4],
            snap_scores: [0i32; 4],
            snap_sticks: 0,
            honba: 0,
            sticks: 0,
            hand_number: 0,
            kyoku_ordinal: 0,
            round_index: 0,
            accepted: [0u8; 4],
            ippatsu: [0u8; 4],
            missed: [0u8; 4],
            kan_count: 0,
            riichi_offered: false,
            tsumo_offered: false,
        }
    }

    /// Live-wall countdown at decision time (`70 - draws`, saturating).
    #[inline]
    pub const fn live_wall(self: &Ledger) -> u32 {
        crate::walk::LIVE_WALL_BASE.saturating_sub(self.draws as u32)
    }
}

impl Default for Ledger {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-game walk context: numeric key + borrowed wall binding.
///
/// The digest is computed ONCE per game on the cold side (feed FORBIDS
/// sha2-hot) and borrowed here: `digest_arc` keeps the allocation alive,
/// `wall_digest` is the `&str` view rows bind. `None` is the wall-less
/// path (SIM mark, never invented). The walk never clones either field —
/// wall agreement (F3) is a context property, not a per-row recompute.
pub struct GameCtx<'a> {
    /// Stable game index (sink join key; replaces `format!` ids).
    pub game_idx: u32,
    /// Wall digest view (`None` wall-less). Borrowed, never cloned hot.
    pub wall_digest: Option<&'a str>,
    /// Owner keeping the digest allocation alive for the walk.
    pub digest_arc: Option<Arc<str>>,
}

impl<'a> GameCtx<'a> {
    /// Wall-less context (SIM-mark path).
    pub const fn wall_less(game_idx: u32) -> Self {
        Self {
            game_idx,
            wall_digest: None,
            digest_arc: None,
        }
    }

    /// Walled context borrowing a cold-computed digest (`digest_arc` keeps
    /// it alive; `wall_digest` must borrow from it or equivalent storage).
    pub const fn walled(game_idx: u32, wall_digest: &'a str, digest_arc: Option<Arc<str>>) -> Self {
        Self {
            game_idx,
            wall_digest: Some(wall_digest),
            digest_arc,
        }
    }

    /// True on the walled path (mode gate active, SIM mark retired).
    #[inline]
    pub const fn is_walled(self: &GameCtx<'a>) -> bool {
        self.wall_digest.is_some()
    }
}

/// Hot plane count (§7 order #0-25). MUST equal `feed::fill::N_PLANES`
/// (P3-A owns that const; this alias keeps the borrow type literal-free).
pub const N_WALK_PLANES: usize = 26;

/// Hot plane names in §7 index order. MUST match the Python
/// ``_HOT_PLANES`` order (``training/rust_stream.py``) and the staged
/// ``[Vec<u8>; N_PLANES]`` order: the per-game bridge serves these names so
/// Python assembles without positional coupling. Index 12 serves packed
/// bytes; callers split ids/len like the fill path does.
pub const PLANE_NAMES: [&str; N_WALK_PLANES] = [
    "concealed_hand_counts",
    "visible_discards_counts",
    "dora_indicators",
    "scores",
    "history_event_kind",
    "history_mask",
    "chosen_action_id",
    "actor",
    "dealer",
    "round_wind",
    "phase",
    "seat_winds",
    "legal_packed",
    "turn_actor",
    "actor_furiten",
    "honba",
    "riichi_sticks",
    "live_wall_tiles_remaining",
    "kan_count",
    "round_index",
    "hand_number",
    "own_drawn_tile",
    "ippatsu_active",
    "riichi_states",
    "actor_can_riichi",
    "actor_can_tsumo",
];

/// Row sink: row counter + borrowed Scratch plane views (§7 order).
///
/// Each plane buffer holds raw LE bytes appended row-major; T-variable
/// planes (#4 kinds `i64`, #5 mask `u8`) append `hist_lens.last()` entries
/// per row and the fill stage buckets them (T-tail clears history ONLY —
/// dora is never touched, G3). `rows` counts committed rows; `hist_lens`
/// records per-row history lengths (breaks the `t_len`/`rows` circle).
pub struct RowSink<'b> {
    /// Committed row count (join key with `(game_idx, seq)`).
    pub rows: u32,
    /// Scratch plane buffers in §7 order (raw LE bytes, row-major).
    pub planes: &'b mut [Vec<u8>; N_WALK_PLANES],
    /// Per-row history lengths (one entry per committed row).
    pub hist_lens: &'b mut Vec<u32>,
}

impl<'b> RowSink<'b> {
    /// Borrow Scratch views (buffers must be empty or reset; kept across games).
    pub const fn new(planes: &'b mut [Vec<u8>; N_WALK_PLANES], hist_lens: &'b mut Vec<u32>) -> Self {
        Self {
            rows: 0,
            planes,
            hist_lens,
        }
    }
}

/// Plane #13: `turn_actor i64` (decision turn owner).
pub const PLANE_TURN_ACTOR: usize = 13;
/// Plane #14: `actor_furiten i64` (0=none 1=temporary 2=riichi 3=discard).
pub const PLANE_FURITEN: usize = 14;
/// Plane #15: `honba i32`.
pub const PLANE_HONBA: usize = 15;
/// Plane #16: `riichi_sticks i32`.
pub const PLANE_STICKS: usize = 16;
/// Plane #17: `live_wall_tiles_remaining i32`.
pub const PLANE_LIVE_WALL: usize = 17;
/// Plane #18: `kan_count i32`.
pub const PLANE_KAN_COUNT: usize = 18;
/// Plane #19: `round_index i32`.
pub const PLANE_ROUND_INDEX: usize = 19;
/// Plane #20: `hand_number i32`.
pub const PLANE_HAND_NUMBER: usize = 20;
/// Plane #21: `own_drawn_tile i32` (`-1` when none).
pub const PLANE_OWN_DRAWN: usize = 21;
/// Plane #22: `ippatsu_active[4] bool`.
pub const PLANE_IPPATSU: usize = 22;
/// Plane #23: `riichi_states[4] i64` (0=none 1=declared 2=accepted).
pub const PLANE_RIICHI_STATES: usize = 23;
/// Plane #24: `actor_can_riichi bool`.
pub const PLANE_CAN_RIICHI: usize = 24;
/// Plane #25: `actor_can_tsumo bool`.
pub const PLANE_CAN_TSUMO: usize = 25;

// ---------------------------------------------------------------------------
// Plane order + categorical ids (mirror the Python authorities, no heap).
// ---------------------------------------------------------------------------

/// Plane #0: `concealed_hand_counts[34] u8`.
pub const PLANE_CONCEALED: usize = 0;
/// Plane #1: `visible_discards_counts[34] u8` (rivers + meld tiles, clamp 4).
pub const PLANE_VISIBLE: usize = 1;
/// Plane #2: `dora_indicators[5] i32` (`-1` tail).
pub const PLANE_DORA: usize = 2;
/// Plane #3: `scores[4] i32`.
pub const PLANE_SCORES: usize = 3;
/// Plane #4: `history_event_kind[T] i64` (`EVENT_KINDS` index).
pub const PLANE_HIST_KIND: usize = 4;
/// Plane #5: `history_mask[T] bool` (all ones for the valid prefix).
pub const PLANE_HIST_MASK: usize = 5;
/// Plane #6: `chosen_action_id i64`.
pub const PLANE_CHOSEN: usize = 6;
/// Plane #7: `actor i64`.
pub const PLANE_ACTOR: usize = 7;
/// Plane #8: `dealer i64`.
pub const PLANE_DEALER: usize = 8;
/// Plane #9: `round_wind i64` (0=E 1=S 2=W 3=N).
pub const PLANE_ROUND_WIND: usize = 9;
/// Plane #10: `phase i64` (`PHASE_*`).
pub const PLANE_PHASE: usize = 10;
/// Plane #11: `seat_winds[4] i64` (0=E 1=S 2=W 3=N, aligned by seat).
pub const PLANE_SEAT_WINDS: usize = 11;
/// Plane #12: `legal_ids[32] i32` + `legal_len i64` (packed 128+8).
pub const PLANE_LEGAL: usize = 12;

/// Phase: `draw_decision` (`PHASES[1]`).
pub const PHASE_DRAW: u8 = 1;
/// Phase: `discard_response` (`PHASES[2]`).
pub const PHASE_RESPONSE: u8 = 2;
/// Phase: `kan_response` (`PHASES[3]`).
pub const PHASE_KAN_RESPONSE: u8 = 3;

/// History kind: `game_start` (`EVENT_KINDS[0]`).
pub const H_GAME_START: u8 = 0;
/// History kind: `round_start` (`EVENT_KINDS[1]`).
pub const H_ROUND_START: u8 = 1;
/// History kind: `turn_advance` (`EVENT_KINDS[2]`).
pub const H_TURN_ADVANCE: u8 = 2;
/// History kind: `draw_tile` (`EVENT_KINDS[3]`, drawing seat only).
pub const H_DRAW_TILE: u8 = 3;
/// History kind: `discard` (`EVENT_KINDS[4]`).
pub const H_DISCARD: u8 = 4;
/// History kind: `riichi_accepted` (`EVENT_KINDS[6]`).
pub const H_RIICHI_ACCEPTED: u8 = 6;
/// History kind: `call_window` (`EVENT_KINDS[7]`).
pub const H_CALL_WINDOW: u8 = 7;
/// History kind: `chi` (`EVENT_KINDS[9]`).
pub const H_CHI: u8 = 9;
/// History kind: `pon` (`EVENT_KINDS[10]`).
pub const H_PON: u8 = 10;
/// History kind: `daiminkan` (`EVENT_KINDS[11]`).
pub const H_DAIMINKAN: u8 = 11;
/// History kind: `ankan` (`EVENT_KINDS[12]`).
pub const H_ANKAN: u8 = 12;
/// History kind: `kakan` (`EVENT_KINDS[13]`).
pub const H_KAKAN: u8 = 13;
/// History kind: `dora_revealed` (`EVENT_KINDS[14]`).
pub const H_DORA: u8 = 14;
/// History kind: `ron` (`EVENT_KINDS[15]`).
pub const H_RON: u8 = 15;
/// History kind: `tsumo` (`EVENT_KINDS[16]`).
pub const H_TSUMO: u8 = 16;
/// History kind: `draw_end` (`EVENT_KINDS[17]`).
pub const H_DRAW_END: u8 = 17;
/// History kind: `abortive_draw` (`EVENT_KINDS[18]`).
pub const H_ABORTIVE: u8 = 18;
/// History kind: `round_end` (`EVENT_KINDS[19]`).
pub const H_ROUND_END: u8 = 19;

// ---------------------------------------------------------------------------
// Tile-class helpers (the string domain folded to u8; aka-aware).
// ---------------------------------------------------------------------------

/// String class of one held take: normal (0), aka red-five string (1),
/// plain-five string (2). Norm folding makes everything else type-level.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum TileClass {
    /// Ordinary string (full four-copy block).
    Norm,
    /// Red-five string (`5xr`/`0x`: the aka copy only).
    Aka,
    /// Plain-five string (`5x`: the three non-aka copies).
    PlainFive,
}

/// Five types whose copy 0 is the aka slot (5m/5p/5s).
#[inline]
pub const fn is_five_type(t: u8) -> bool {
    t == 4 || t == 13 || t == 22
}

/// Aka-suit index for a five type (m/p/s → 0/1/2).
#[inline]
pub const fn aka_suit(t: u8) -> usize {
    (t / 9) as usize
}

/// True iff the physical id is an aka copy (16/52/88).
#[inline]
pub const fn is_aka_id(id: u8) -> bool {
    id == 16 || id == 52 || id == 88
}

/// String class of a canonical ingest id (`TileCodec::physical` output:
/// aka → 16/52/88, plain five → base+1, else block base … +0).
#[inline]
pub const fn class_of_canonical(id: u8) -> (u8, TileClass) {
    let t = id / 4;
    if is_aka_id(id) {
        (t, TileClass::Aka)
    } else if is_five_type(t) {
        (t, TileClass::PlainFive)
    } else {
        (t, TileClass::Norm)
    }
}

/// String class of a TRUE take id (slot copy).
#[inline]
pub const fn class_of_take(id: u8) -> (u8, TileClass) {
    let t = id / 4;
    let base = (t as u16) * 4;
    if is_five_type(t) {
        if id as u16 == base {
            (t, TileClass::Aka)
        } else {
            (t, TileClass::PlainFive)
        }
    } else {
        (t, TileClass::Norm)
    }
}

/// Pool-first (canonical report/offer) id for a held string class:
/// aka → the aka copy; plain five → base+1; else the block base.
/// Mirrors `copies_of_string(pai).first()` without touching a string.
#[inline]
pub const fn pool_first_of_class(t: u8, class: TileClass) -> u8 {
    let base = t * 4;
    match class {
        TileClass::Aka => base,
        TileClass::PlainFive => base + 1,
        TileClass::Norm => base,
    }
}

/// Norm pool-first id for a TYPE (responder shape gate): plain-five
/// representative is base+1 (mirrors `copies_of_string("5x").first()`),
/// aka folds in; everything else is the block base.
#[inline]
pub const fn norm_pool_first(t: u8) -> u8 {
    if is_five_type(t) { t * 4 + 1 } else { t * 4 }
}

/// True held takes of one present type row, slot order.
///
/// Cells store take id + 1 (0 == empty); emits the stored takes ascending
/// by slot (allocation is monotonic without reuse, so slot order is take
/// order within each class and the aka singleton sorts first). Replaces
/// the old pool-first collapse: the oracle reports true copies and the
/// walk must offer/encode the same ids.
pub fn held_takes(row: &[u8; 4], _t: u8, out: &mut [u8; 8]) -> usize {
    let mut n = 0usize;
    let mut c = 0usize;
    while c < 4 {
        let v = row[c];
        if v != 0 {
            out[n] = v - 1;
            n += 1;
        }
        c += 1;
    }
    n
}
