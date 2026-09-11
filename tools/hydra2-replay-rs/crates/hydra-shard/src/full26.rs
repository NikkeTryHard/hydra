//! Full-26 cold build (P4-B, cold only — NEVER hot).
//!
//! The hot path keeps 13 planes (plan §7: counts as `uint8`, legal as
//! packed `legal_ids[32] + legal_len`); the full 26-field baseline
//! (`src/hydra2/models/schema.py::_BASELINE_FIELDS`, alphabetical) is rebuilt
//! HERE, cold-only, from one hot row plus a [`ColdSidecar`]:
//!
//! - `uint8 → int32` count widening (concealed/visible): the hot `u8` bytes
//!   widen to `int32` LE at collate. G2 compares **post-cast**: the hot `u8`
//!   is cast to `int32` first, then compared EXACT (no float path — the
//!   schema carries no float fields, so the `allclose 1e-6` leg is vacuous
//!   and the integer leg is byte-exact).
//! - `legal[6792]`: the hot prefix carries at most 32 ids; the full mask
//!   rides the sidecar (neutral = exactly the hot ids; production = the cold
//!   oracle walk's full offer set — same follow-up wiring as han/fu, P4-A).
//! - Dropped scalars (`hand_number`, `honba`, `live_wall`, …) ride the
//!   sidecar; `actor_seats` is the `actor` dup (encoder-verbatim).
//!
//! Compact row layout (LE, fixed stride per file `T` bucket; all offsets
//! from [`compact_row_bytes`]):
//! ```text
//! u16 hist_len | u32 game_idx | u32 seq
//! u8[34] concealed | u8[34] visible | i32[5] dora | i32[4] scores
//! u8[T] hist_kind (valid prefix, zero tail)
//! u8[(T+7)/8] hist_mask packed (full bucket, zero tail)
//! i64 chosen | u8 actor dealer round_wind phase seat_winds[4]
//! u8[850] legal packed (6792b + reserved zero byte)
//! sidecar[34]: can_riichi can_tsumo furiten turn_actor
//!   hand_number i32 honba i32 ippatsu_mask u8 kan_count u8
//!   live_wall i32 own_drawn i32 riichi[4] u8 riichi_sticks i32 round_index i32
//! ```
//!
//! Measured shape (this layout, `T = 256`): compact row
//! [`compact_row_bytes`](compact_row_bytes)`(256)` `= 1302`B vs naive dense
//! full-26 [`dense_full26_row_bytes`](dense_full26_row_bytes)`(256)` `= 9558`B
//! (≈7.3×; the legal bitpack alone saves `6792 − 850 = 5942`B/row).

use core::fmt;

use hydra_feed::fill::{N_PLANES, ROW_BYTES_FIXED, StagedGame};
use hydra_feed::ledger::{
    PLANE_ACTOR, PLANE_CHOSEN, PLANE_CONCEALED, PLANE_DEALER, PLANE_DORA, PLANE_HIST_KIND,
    PLANE_HIST_MASK, PLANE_LEGAL, PLANE_PHASE, PLANE_ROUND_WIND, PLANE_SCORES, PLANE_SEAT_WINDS,
    PLANE_VISIBLE,
};

use crate::compact::{
    LEGAL_BITS, LEGAL_PACKED_BYTES, pack_hist_mask, pack_legal_bits, unpack_hist_mask,
    unpack_legal_bits,
};

/// Frozen history buckets (matches `HISTORY_BUCKET_LENGTHS` + feed `T_BUCKETS`).
pub const COMPACT_T_BUCKETS: [usize; 4] = [32, 64, 128, 256];

/// Full-26 dtype (schema dtypes; `bool` stores as one `0`/`1` byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Full26Dtype {
    I64,
    I32,
    Bool,
}

/// Full-26 plane shape class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Full26Shape {
    Scalar,
    Fixed(usize),
    /// History bucket width `T`.
    Hist,
}

/// One full-26 field: schema-alphabetical order (matches `_BASELINE_FIELDS`).
#[derive(Debug, Clone, Copy)]
pub struct Full26Field {
    pub name: &'static str,
    pub dtype: Full26Dtype,
    pub shape: Full26Shape,
}

/// The 26 baseline fields in schema-alphabetical order.
pub const FULL26_FIELDS: [Full26Field; 26] = [
    Full26Field { name: "actor", dtype: Full26Dtype::I64, shape: Full26Shape::Scalar },
    Full26Field { name: "actor_can_riichi", dtype: Full26Dtype::Bool, shape: Full26Shape::Scalar },
    Full26Field { name: "actor_can_tsumo", dtype: Full26Dtype::Bool, shape: Full26Shape::Scalar },
    Full26Field { name: "actor_furiten", dtype: Full26Dtype::I64, shape: Full26Shape::Scalar },
    Full26Field { name: "actor_seats", dtype: Full26Dtype::I64, shape: Full26Shape::Scalar },
    Full26Field { name: "concealed_hand_counts", dtype: Full26Dtype::I32, shape: Full26Shape::Fixed(34) },
    Full26Field { name: "dealer", dtype: Full26Dtype::I64, shape: Full26Shape::Scalar },
    Full26Field { name: "dora_indicators", dtype: Full26Dtype::I32, shape: Full26Shape::Fixed(5) },
    Full26Field { name: "hand_number", dtype: Full26Dtype::I32, shape: Full26Shape::Scalar },
    Full26Field { name: "history_event_kind", dtype: Full26Dtype::I64, shape: Full26Shape::Hist },
    Full26Field { name: "history_mask", dtype: Full26Dtype::Bool, shape: Full26Shape::Hist },
    Full26Field { name: "honba", dtype: Full26Dtype::I32, shape: Full26Shape::Scalar },
    Full26Field { name: "ippatsu_active", dtype: Full26Dtype::Bool, shape: Full26Shape::Fixed(4) },
    Full26Field { name: "kan_count", dtype: Full26Dtype::I32, shape: Full26Shape::Scalar },
    Full26Field { name: "legal_mask", dtype: Full26Dtype::Bool, shape: Full26Shape::Fixed(6792) },
    Full26Field { name: "live_wall_tiles_remaining", dtype: Full26Dtype::I32, shape: Full26Shape::Scalar },
    Full26Field { name: "own_drawn_tile", dtype: Full26Dtype::I32, shape: Full26Shape::Scalar },
    Full26Field { name: "phase", dtype: Full26Dtype::I64, shape: Full26Shape::Scalar },
    Full26Field { name: "riichi_states", dtype: Full26Dtype::I64, shape: Full26Shape::Fixed(4) },
    Full26Field { name: "riichi_sticks", dtype: Full26Dtype::I32, shape: Full26Shape::Scalar },
    Full26Field { name: "round_index", dtype: Full26Dtype::I32, shape: Full26Shape::Scalar },
    Full26Field { name: "round_wind", dtype: Full26Dtype::I64, shape: Full26Shape::Scalar },
    Full26Field { name: "scores", dtype: Full26Dtype::I32, shape: Full26Shape::Fixed(4) },
    Full26Field { name: "seat_winds", dtype: Full26Dtype::I64, shape: Full26Shape::Fixed(4) },
    Full26Field { name: "turn_actor", dtype: Full26Dtype::I64, shape: Full26Shape::Scalar },
    Full26Field { name: "visible_discards_counts", dtype: Full26Dtype::I32, shape: Full26Shape::Fixed(34) },
];

/// Plane index of a full-26 field by name (`None` when unknown).
pub fn full26_plane(name: &str) -> Option<usize> {
    let mut i = 0usize;
    while i < FULL26_FIELDS.len() {
        if FULL26_FIELDS[i].name == name {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Row stride of one full-26 plane at bucket `t_len`.
pub fn full26_plane_row_bytes(field: usize, t_len: usize) -> usize {
    let f = &FULL26_FIELDS[field];
    let width = match f.dtype {
        Full26Dtype::I64 => 8,
        Full26Dtype::I32 => 4,
        Full26Dtype::Bool => 1,
    };
    let count = match f.shape {
        Full26Shape::Scalar => 1,
        Full26Shape::Fixed(n) => n,
        Full26Shape::Hist => t_len,
    };
    width * count
}

/// Naive dense full-26 row bytes at bucket `t_len` (the un-packed baseline
/// the compact shape beats: legal as 6792 `u8`, counts as `int32`, kinds as
/// `int64`). `t = 256` → 9558B.
pub fn dense_full26_row_bytes(t_len: usize) -> usize {
    let mut total = 0usize;
    let mut i = 0usize;
    while i < FULL26_FIELDS.len() {
        total += full26_plane_row_bytes(i, t_len);
        i += 1;
    }
    total
}

// ---------------------------------------------------------------------------
// Compact row geometry
// ---------------------------------------------------------------------------

/// Fixed compact prefix: `u16 hist_len + u32 game_idx + u32 seq`
/// + `34 + 34 + 20 + 16` (concealed/visible/dora/scores).
pub const COMPACT_PREFIX: usize = 2 + 4 + 4 + 34 + 34 + 20 + 16;
/// Trailing actor block: `i64 chosen + u8 actor/dealer/round_wind/phase + u8[4] seats`.
pub const COMPACT_ACTOR_BLOCK: usize = 8 + 8;
/// Cold sidecar tail (see module docs).
pub const COMPACT_SIDECAR: usize = 34;

/// Packed history-mask bytes for bucket `t`.
#[inline]
pub fn compact_mask_bytes(t: usize) -> usize {
    (t + 7) / 8
}

/// Compact row stride at bucket `t` (`T = 256` → 1302B).
pub fn compact_row_bytes(t: usize) -> usize {
    COMPACT_PREFIX + t + compact_mask_bytes(t) + COMPACT_ACTOR_BLOCK + LEGAL_PACKED_BYTES + COMPACT_SIDECAR
}

/// Offset of the packed history mask at bucket `t`.
#[inline]
pub fn off_hist_mask(t: usize) -> usize {
    COMPACT_PREFIX + t
}

/// Offset of the actor block (`chosen` first) at bucket `t`.
#[inline]
pub fn off_actor(t: usize) -> usize {
    COMPACT_PREFIX + t + compact_mask_bytes(t)
}

/// Offset of the packed legal field at bucket `t`.
#[inline]
pub fn off_legal(t: usize) -> usize {
    off_actor(t) + COMPACT_ACTOR_BLOCK
}

/// Offset of the sidecar tail at bucket `t`.
#[inline]
pub fn off_sidecar(t: usize) -> usize {
    off_legal(t) + LEGAL_PACKED_BYTES
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Full-26 build failure taxonomy (fail-closed, cold only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Full26Error {
    /// Hot plane footprint mismatch (`plane`, `expected`, `got`).
    HotLength { plane: usize, expected: usize, got: usize },
    /// History length exceeds the file bucket (`len`, `bucket`).
    HistoryCap { len: usize, bucket: usize },
    /// A narrowed value is out of range (`field` static, raw value).
    NarrowRange { field: &'static str, value: i64 },
    /// A hot legal id is out of range (`id`).
    BadLegalId { id: i64 },
    /// Sidecar legal mask has the wrong length (`got`).
    BadLegalLen { got: usize },
    /// Compact row has the wrong length (`expected`, `got`).
    RowLength { expected: usize, got: usize },
    /// Bitpack round-trip failure (message-free wrap; see `crate::compact`).
    Pack,
}

impl fmt::Display for Full26Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Full26Error::HotLength { plane, expected, got } => {
                write!(f, "hot plane {plane}: expected {expected}B, got {got}B")
            }
            Full26Error::HistoryCap { len, bucket } => {
                write!(f, "history {len} exceeds bucket {bucket}")
            }
            Full26Error::NarrowRange { field, value } => {
                write!(f, "narrow range: {field} = {value}")
            }
            Full26Error::BadLegalId { id } => write!(f, "hot legal id out of range: {id}"),
            Full26Error::BadLegalLen { got } => {
                write!(f, "sidecar legal mask: expected {LEGAL_BITS}, got {got}")
            }
            Full26Error::RowLength { expected, got } => {
                write!(f, "compact row: expected {expected}B, got {got}B")
            }
            Full26Error::Pack => write!(f, "compact bitpack failure"),
        }
    }
}

impl std::error::Error for Full26Error {}

// ---------------------------------------------------------------------------
// Hot row view (borrowed §7 planes, one row)
// ---------------------------------------------------------------------------

/// One decoded hot row: borrowed byte slices plus decoded scalars.
#[derive(Debug, Clone, Copy)]
pub struct HotRow<'a> {
    pub concealed: &'a [u8],
    pub visible: &'a [u8],
    pub dora: &'a [u8],
    pub scores: &'a [u8],
    pub hist_kind: &'a [u8],
    pub hist_mask: &'a [u8],
    pub hist_len: usize,
    pub chosen: i64,
    pub actor: i64,
    pub dealer: i64,
    pub round_wind: i64,
    pub phase: i64,
    pub seat_winds: &'a [u8],
    pub legal_ids: &'a [u8],
    pub legal_len: i64,
}

fn rd_i64(bytes: &[u8]) -> Option<i64> {
    let arr: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
    Some(i64::from_le_bytes(arr))
}

/// Borrow row `row` of staged hot planes (`hist_lens` breaks the `t_len`
/// circle the same way `feed::fill` does).
pub fn hot_row_at<'a>(
    planes: &'a [Vec<u8>; N_PLANES],
    hist_lens: &'a [u32],
    row: usize,
) -> Result<HotRow<'a>, Full26Error> {
    let hl = *hist_lens.get(row).ok_or(Full26Error::HotLength {
        plane: PLANE_HIST_KIND,
        expected: row + 1,
        got: hist_lens.len(),
    })? as usize;
    // Fixed-plane offsets.
    let mut kind_off = 0usize;
    let mut mask_off = 0usize;
    let mut r = 0usize;
    while r < row {
        let l = *hist_lens.get(r).ok_or(Full26Error::HotLength {
            plane: PLANE_HIST_KIND,
            expected: row + 1,
            got: hist_lens.len(),
        })? as usize;
        kind_off += l * 8;
        mask_off += l;
        r += 1;
    }
    let fixed = |plane: usize| -> Result<&[u8], Full26Error> {
        let stride = ROW_BYTES_FIXED[plane];
        let need = (row + 1) * stride;
        let buf = &planes[plane];
        if buf.len() < need {
            return Err(Full26Error::HotLength {
                plane,
                expected: need,
                got: buf.len(),
            });
        }
        Ok(&buf[row * stride..(row + 1) * stride])
    };
    let var = |plane: usize, off: usize, len: usize| -> Result<&[u8], Full26Error> {
        let buf = &planes[plane];
        if buf.len() < off + len {
            return Err(Full26Error::HotLength {
                plane,
                expected: off + len,
                got: buf.len(),
            });
        }
        Ok(&buf[off..off + len])
    };
    let chosen_b = fixed(PLANE_CHOSEN)?;
    let actor_b = fixed(PLANE_ACTOR)?;
    let dealer_b = fixed(PLANE_DEALER)?;
    let wind_b = fixed(PLANE_ROUND_WIND)?;
    let phase_b = fixed(PLANE_PHASE)?;
    let legal_b = fixed(PLANE_LEGAL)?;
    let chosen = rd_i64(chosen_b).ok_or(Full26Error::HotLength {
        plane: PLANE_CHOSEN,
        expected: 8,
        got: chosen_b.len(),
    })?;
    let actor = rd_i64(actor_b).ok_or(Full26Error::HotLength {
        plane: PLANE_ACTOR,
        expected: 8,
        got: actor_b.len(),
    })?;
    let dealer = rd_i64(dealer_b).ok_or(Full26Error::HotLength {
        plane: PLANE_DEALER,
        expected: 8,
        got: dealer_b.len(),
    })?;
    let round_wind = rd_i64(wind_b).ok_or(Full26Error::HotLength {
        plane: PLANE_ROUND_WIND,
        expected: 8,
        got: wind_b.len(),
    })?;
    let phase = rd_i64(phase_b).ok_or(Full26Error::HotLength {
        plane: PLANE_PHASE,
        expected: 8,
        got: phase_b.len(),
    })?;
    let legal_len = rd_i64(&legal_b[128..136]).ok_or(Full26Error::HotLength {
        plane: PLANE_LEGAL,
        expected: 136,
        got: legal_b.len(),
    })?;
    Ok(HotRow {
        concealed: fixed(PLANE_CONCEALED)?,
        visible: fixed(PLANE_VISIBLE)?,
        dora: fixed(PLANE_DORA)?,
        scores: fixed(PLANE_SCORES)?,
        hist_kind: var(PLANE_HIST_KIND, kind_off, hl * 8)?,
        hist_mask: var(PLANE_HIST_MASK, mask_off, hl)?,
        hist_len: hl,
        chosen,
        actor,
        dealer,
        round_wind,
        phase,
        seat_winds: fixed(PLANE_SEAT_WINDS)?,
        legal_ids: legal_b.get(..128).ok_or(Full26Error::HotLength {
            plane: PLANE_LEGAL,
            expected: 136,
            got: legal_b.len(),
        })?,
        legal_len,
    })
}

/// Row iterator over one staged game (running kind/mask offsets, no re-scan).
pub struct HotRowIter<'a> {
    planes: &'a [Vec<u8>; N_PLANES],
    hist_lens: &'a [u32],
    row: usize,
    kind_off: usize,
    mask_off: usize,
}

impl<'a> HotRowIter<'a> {
    /// Iterate rows of `game` in emission order.
    pub fn new(game: &'a StagedGame) -> Self {
        Self {
            planes: &game.planes,
            hist_lens: &game.hist_lens,
            row: 0,
            kind_off: 0,
            mask_off: 0,
        }
    }
}

impl<'a> Iterator for HotRowIter<'a> {
    type Item = Result<HotRow<'a>, Full26Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.hist_lens.len() {
            return None;
        }
        let row = self.row;
        let out = hot_row_at(self.planes, self.hist_lens, row);
        if let Ok(hot) = &out {
            self.kind_off += hot.hist_len * 8;
            self.mask_off += hot.hist_len;
        }
        self.row += 1;
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// Cold sidecar (everything the hot 13 do not carry)
// ---------------------------------------------------------------------------

/// Per-row cold context for the full-26 build.
///
/// `neutral_from_hot` derives a deterministic sidecar from the hot row alone
/// (legal = exactly the hot ids; kyoku scalars = documented neutral defaults)
/// for hot-parity (G2) tests. Production fills every field from the cold
/// oracle walk — the same follow-up wiring as han/fu (P4-A owns that leg).
#[derive(Debug, Clone)]
pub struct ColdSidecar {
    pub actor_can_riichi: bool,
    pub actor_can_tsumo: bool,
    /// Furiten state id (`none/temporary/riichi/discard` = `0..=3`).
    pub actor_furiten: u8,
    pub turn_actor: u8,
    pub hand_number: i32,
    pub honba: i32,
    /// One bit per seat (`ippatsu_active[4]`).
    pub ippatsu_mask: u8,
    pub kan_count: i32,
    pub live_wall: i32,
    /// `-1` = none (matches the encoder `-1` fill).
    pub own_drawn: i32,
    /// Per-seat riichi state id (`none/declared/accepted` = `0..=2`).
    pub riichi: [u8; 4],
    pub riichi_sticks: i32,
    pub round_index: i32,
    /// Full 6792-element `0`/`1` legal mask.
    pub legal_full: Vec<u8>,
}

impl ColdSidecar {
    /// Deterministic neutral sidecar from the hot row alone.
    pub fn neutral_from_hot(hot: &HotRow<'_>) -> Result<Self, Full26Error> {
        if hot.legal_len < 0 || hot.legal_len > 32 {
            return Err(Full26Error::BadLegalId { id: hot.legal_len });
        }
        let mut legal_full = vec![0u8; LEGAL_BITS];
        let n = hot.legal_len as usize;
        let mut k = 0usize;
        while k < n {
            let id = i32::from_le_bytes(
                hot.legal_ids
                    .get(k * 4..k * 4 + 4)
                    .ok_or(Full26Error::HotLength {
                        plane: PLANE_LEGAL,
                        expected: 128,
                        got: hot.legal_ids.len(),
                    })?
                    .try_into()
                    .map_err(|_| Full26Error::HotLength {
                        plane: PLANE_LEGAL,
                        expected: 128,
                        got: hot.legal_ids.len(),
                    })?,
            ) as i64;
            if id < 0 || id >= LEGAL_BITS as i64 {
                return Err(Full26Error::BadLegalId { id });
            }
            legal_full[id as usize] = 1;
            k += 1;
        }
        let turn_actor = narrow_u8("turn_actor", hot.actor)?;
        Ok(Self {
            actor_can_riichi: false,
            actor_can_tsumo: false,
            actor_furiten: 0,
            turn_actor,
            hand_number: 0,
            honba: 0,
            ippatsu_mask: 0,
            kan_count: 0,
            // Neutral default: full live wall (no draw information hot).
            live_wall: 70,
            own_drawn: -1,
            riichi: [0; 4],
            riichi_sticks: 0,
            round_index: 0,
            legal_full,
        })
    }
}

fn narrow_u8(field: &'static str, v: i64) -> Result<u8, Full26Error> {
    if v < 0 || v > 255 {
        return Err(Full26Error::NarrowRange { field, value: v });
    }
    Ok(v as u8)
}

fn narrow_seat(field: &'static str, v: i64) -> Result<u8, Full26Error> {
    if v < 0 || v > 3 {
        return Err(Full26Error::NarrowRange { field, value: v });
    }
    Ok(v as u8)
}

// ---------------------------------------------------------------------------
// Encode: hot row + sidecar → compact bytes
// ---------------------------------------------------------------------------

/// Encode one compact row at bucket `t` (appends exactly
/// [`compact_row_bytes`](compact_row_bytes)`(t)` bytes).
pub fn encode_compact_row(
    game_idx: u32,
    seq: u32,
    hot: &HotRow<'_>,
    cold: &ColdSidecar,
    t: usize,
    out: &mut Vec<u8>,
) -> Result<(), Full26Error> {
    if hot.hist_len > t {
        return Err(Full26Error::HistoryCap {
            len: hot.hist_len,
            bucket: t,
        });
    }
    if cold.legal_full.len() != LEGAL_BITS {
        return Err(Full26Error::BadLegalLen {
            got: cold.legal_full.len(),
        });
    }
    if hot.hist_kind.len() != hot.hist_len * 8 || hot.hist_mask.len() != hot.hist_len {
        return Err(Full26Error::HotLength {
            plane: PLANE_HIST_KIND,
            expected: hot.hist_len,
            got: hot.hist_kind.len().min(hot.hist_mask.len()),
        });
    }
    // Narrow checks BEFORE any byte is appended (all-or-nothing).
    let actor = narrow_seat("actor", hot.actor)?;
    let dealer = narrow_seat("dealer", hot.dealer)?;
    let round_wind = narrow_seat("round_wind", hot.round_wind)?;
    let phase = narrow_u8("phase", hot.phase)?;
    let mut seats = [0u8; 4];
    let mut i = 0usize;
    while i < 4 {
        seats[i] = narrow_seat(
            "seat_winds",
            rd_i64(&hot.seat_winds[i * 8..]).ok_or(Full26Error::HotLength {
                plane: PLANE_SEAT_WINDS,
                expected: 32,
                got: hot.seat_winds.len(),
            })?,
        )?;
        i += 1;
    }
    let mut kinds = Vec::with_capacity(t);
    let mut k = 0usize;
    while k < hot.hist_len {
        let kind = rd_i64(&hot.hist_kind[k * 8..]).ok_or(Full26Error::HotLength {
            plane: PLANE_HIST_KIND,
            expected: hot.hist_len * 8,
            got: hot.hist_kind.len(),
        })?;
        kinds.push(narrow_u8("history_event_kind", kind)?);
        k += 1;
    }
    if cold.actor_furiten > 3 {
        return Err(Full26Error::NarrowRange {
            field: "actor_furiten",
            value: i64::from(cold.actor_furiten),
        });
    }
    if cold.turn_actor > 3 {
        return Err(Full26Error::NarrowRange {
            field: "turn_actor",
            value: i64::from(cold.turn_actor),
        });
    }
    if cold.kan_count < 0 || cold.kan_count > 4 {
        return Err(Full26Error::NarrowRange {
            field: "kan_count",
            value: i64::from(cold.kan_count),
        });
    }
    let mut r = 0usize;
    while r < 4 {
        if cold.riichi[r] > 2 {
            return Err(Full26Error::NarrowRange {
                field: "riichi_states",
                value: i64::from(cold.riichi[r]),
            });
        }
        r += 1;
    }
    let packed_legal = pack_legal_bits(&cold.legal_full).map_err(|_| Full26Error::Pack)?;
    // Full-bucket mask (valid prefix + zero tail), then bit-pack.
    let mut mask_full = vec![0u8; t];
    mask_full[..hot.hist_len].copy_from_slice(hot.hist_mask);
    let packed_mask = pack_hist_mask(&mask_full, t).map_err(|_| Full26Error::Pack)?;

    out.extend_from_slice(&(hot.hist_len as u16).to_le_bytes());
    out.extend_from_slice(&game_idx.to_le_bytes());
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(hot.concealed);
    out.extend_from_slice(hot.visible);
    out.extend_from_slice(hot.dora);
    out.extend_from_slice(hot.scores);
    // Kinds: valid prefix + zero tail.
    out.extend_from_slice(&kinds);
    out.resize(out.len() + (t - hot.hist_len), 0);
    out.extend_from_slice(&packed_mask);
    out.extend_from_slice(&hot.chosen.to_le_bytes());
    out.push(actor);
    out.push(dealer);
    out.push(round_wind);
    out.push(phase);
    out.extend_from_slice(&seats);
    out.extend_from_slice(&packed_legal);
    // Sidecar tail.
    out.push(u8::from(cold.actor_can_riichi));
    out.push(u8::from(cold.actor_can_tsumo));
    out.push(cold.actor_furiten);
    out.push(cold.turn_actor);
    out.extend_from_slice(&cold.hand_number.to_le_bytes());
    out.extend_from_slice(&cold.honba.to_le_bytes());
    out.push(cold.ippatsu_mask & 0x0F);
    out.push(cold.kan_count as u8);
    out.extend_from_slice(&cold.live_wall.to_le_bytes());
    out.extend_from_slice(&cold.own_drawn.to_le_bytes());
    out.extend_from_slice(&cold.riichi);
    out.extend_from_slice(&cold.riichi_sticks.to_le_bytes());
    out.extend_from_slice(&cold.round_index.to_le_bytes());
    Ok(())
}

// ---------------------------------------------------------------------------
// Decode: compact bytes → one row per full-26 plane (the collate expansion)
// ---------------------------------------------------------------------------

/// Decode one compact row into the 26 collated plane buffers plus the label.
#[doc = "Appends exactly one row per plane (counts widen `u8 -> int32` LE,"]
#[doc = "legal/mask bits expand to `0`/`1` bytes, kinds widen `u8 -> i64`)"]
#[doc = "and one `i64` LE label into `label_chosen`. `chosen_action_id` is the"]
#[doc = "training label, not a baseline input field: it rides alongside, never"]
#[doc = "inside, the collated planes."]
pub fn decode_row_into_planes(
    row: &[u8],
    t: usize,
    planes: &mut [Vec<u8>; 26],
    label_chosen: &mut Vec<u8>,
) -> Result<(), Full26Error> {
    let want = compact_row_bytes(t);
    if row.len() != want {
        return Err(Full26Error::RowLength {
            expected: want,
            got: row.len(),
        });
    }
    let hl = u16::from_le_bytes([row[0], row[1]]) as usize;
    if hl > t {
        return Err(Full26Error::HistoryCap { len: hl, bucket: t });
    }
    let concealed = &row[10..44];
    let visible = &row[44..78];
    let dora = &row[78..98];
    let scores = &row[98..114];
    let kinds = &row[COMPACT_PREFIX..COMPACT_PREFIX + t];
    let mask_packed = &row[off_hist_mask(t)..off_hist_mask(t) + compact_mask_bytes(t)];
    let actor_off = off_actor(t);
    let chosen = i64::from_le_bytes(row[actor_off..actor_off + 8].try_into().map_err(|_| {
        Full26Error::RowLength {
            expected: want,
            got: row.len(),
        }
    })?);
    let actor = row[actor_off + 8];
    let dealer = row[actor_off + 9];
    let round_wind = row[actor_off + 10];
    let phase = row[actor_off + 11];
    let seats = &row[actor_off + 12..actor_off + 16];
    let mut legal_packed = [0u8; LEGAL_PACKED_BYTES];
    legal_packed.copy_from_slice(&row[off_legal(t)..off_legal(t) + LEGAL_PACKED_BYTES]);
    let sc = &row[off_sidecar(t)..off_sidecar(t) + COMPACT_SIDECAR];
    let can_riichi = sc[0];
    let can_tsumo = sc[1];
    let furiten = sc[2];
    let turn_actor = sc[3];

    // Expand bitpacks.
    let mut legal = vec![0u8; LEGAL_BITS];
    unpack_legal_bits(&legal_packed, &mut legal).map_err(|_| Full26Error::Pack)?;
    let mut mask_full = vec![0u8; t];
    unpack_hist_mask(mask_packed, &mut mask_full, t).map_err(|_| Full26Error::Pack)?;

    let put_i64 = |planes: &mut [Vec<u8>; 26], p: usize, v: i64| {
        planes[p].extend_from_slice(&v.to_le_bytes());
    };
    let put_i32 = |planes: &mut [Vec<u8>; 26], p: usize, v: i32| {
        planes[p].extend_from_slice(&v.to_le_bytes());
    };
    let widen_u8_to_i32 = |planes: &mut [Vec<u8>; 26], p: usize, xs: &[u8]| {
        for b in xs {
            planes[p].extend_from_slice(&i32::from(*b).to_le_bytes());
        }
    };

    // Schema-alphabetical plane order (indices match FULL26_FIELDS).
    put_i64(planes, 0, i64::from(actor)); // actor
    planes[1].push(can_riichi); // actor_can_riichi
    planes[2].push(can_tsumo); // actor_can_tsumo
    put_i64(planes, 3, i64::from(furiten)); // actor_furiten
    put_i64(planes, 4, i64::from(actor)); // actor_seats = actor dup (encoder-verbatim)
    widen_u8_to_i32(planes, 5, concealed); // concealed u8→int32 (G2 post-cast)
    put_i64(planes, 6, i64::from(dealer)); // dealer
    planes[7].extend_from_slice(dora); // dora i32 verbatim
    put_i32(
        planes,
        8,
        i32::from_le_bytes([sc[4], sc[5], sc[6], sc[7]]),
    ); // hand_number
    // history_event_kind: valid prefix widened u8→i64, zero tail.
    for b in &kinds[..hl] {
        put_i64(planes, 9, i64::from(*b));
    }
    for _ in hl..t {
        put_i64(planes, 9, 0);
    }
    // history_mask: valid prefix bytes, zero tail.
    planes[10].extend_from_slice(&mask_full[..hl]);
    planes[10].resize(planes[10].len() + (t - hl), 0);
    put_i32(
        planes,
        11,
        i32::from_le_bytes([sc[8], sc[9], sc[10], sc[11]]),
    ); // honba
    for b in [sc[12] & 1, (sc[12] >> 1) & 1, (sc[12] >> 2) & 1, (sc[12] >> 3) & 1] {
        planes[12].push(b);
    } // ippatsu_active[4]
    put_i32(planes, 13, i32::from(sc[13])); // kan_count
    planes[14].extend_from_slice(&legal); // legal_mask expanded u8
    put_i32(
        planes,
        15,
        i32::from_le_bytes([sc[14], sc[15], sc[16], sc[17]]),
    ); // live_wall
    put_i32(
        planes,
        16,
        i32::from_le_bytes([sc[18], sc[19], sc[20], sc[21]]),
    ); // own_drawn
    put_i64(planes, 17, i64::from(phase)); // phase
    for b in &sc[22..26] {
        put_i64(planes, 18, i64::from(*b));
    } // riichi_states[4] u8→i64
    put_i32(
        planes,
        19,
        i32::from_le_bytes([sc[26], sc[27], sc[28], sc[29]]),
    ); // riichi_sticks
    put_i32(
        planes,
        20,
        i32::from_le_bytes([sc[30], sc[31], sc[32], sc[33]]),
    ); // round_index
    put_i64(planes, 21, i64::from(round_wind)); // round_wind
    planes[22].extend_from_slice(scores); // scores[4] i32 verbatim
    for b in seats {
        put_i64(planes, 23, i64::from(*b));
    } // seat_winds[4] u8→i64
    put_i64(planes, 24, i64::from(turn_actor)); // turn_actor
    widen_u8_to_i32(planes, 25, visible); // visible u8→int32 (G2 post-cast)
    label_chosen.extend_from_slice(&chosen.to_le_bytes()); // training label
    Ok(())
}
