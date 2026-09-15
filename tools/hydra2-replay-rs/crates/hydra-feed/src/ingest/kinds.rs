/// Event kind: start aliases (`start_game`/`startGame`/`game_start`/`start`).
pub const KIND_START: u8 = 0;
/// Event kind: end aliases (`end_game`/`endGame`/`game_end`/`end`).
pub const KIND_END: u8 = 1;
/// Event kind: `dahai` (sampled).
pub const KIND_DAHAI: u8 = 2;
/// Event kind: `chi` (sampled).
pub const KIND_CHI: u8 = 3;
/// Event kind: `pon` (sampled).
pub const KIND_PON: u8 = 4;
/// Event kind: `daiminkan` (sampled).
pub const KIND_DAIMINKAN: u8 = 5;
/// Event kind: `ankan` (sampled).
pub const KIND_ANKAN: u8 = 6;
/// Event kind: `kakan` (sampled).
pub const KIND_KAKAN: u8 = 7;
/// Event kind: `hora` (sampled).
pub const KIND_HORA: u8 = 8;
/// Event kind: `dora`.
pub const KIND_DORA: u8 = 9;
/// Event kind: `reach` (declaration; collapses with its dahai, never a row).
pub const KIND_REACH: u8 = 10;
/// Event kind: `reach_accepted` (transparent for ron-pending).
pub const KIND_REACH_ACCEPTED: u8 = 11;
/// Event kind: `tsumo` (draw).
pub const KIND_TSUMO: u8 = 12;
/// Event kind: `start_kyoku` (kyoku boundary; resets ron-pending).
pub const KIND_START_KYOKU: u8 = 13;
/// Event kind: `end_kyoku`.
pub const KIND_END_KYOKU: u8 = 14;
/// Event kind: `ryukyoku`.
pub const KIND_RYUKYOKU: u8 = 15;
/// Event kind: reserved transparent extension (nothing maps here today).
pub const KIND_TRANSPARENT_OTHER: u8 = 16;
/// Event kind: unknown / unusable type (gate rejects → InvalidData).
pub const KIND_OTHER: u8 = 17;

/// Seat sentinel: `0xFF` = field absent or out of range.
pub const NO_SEAT: u8 = 0xFF;
/// Tile sentinel: `0xFF` = no tile.
pub const NO_PAI: u8 = 0xFF;

/// Framing vocabulary: game-boundary aliases (mirrors `decode.py`).
pub const START_TYPES: [&str; 4] = ["start_game", "startGame", "game_start", "start"];
/// Framing vocabulary: game-end aliases (mirrors `decode.py`).
pub const END_TYPES: [&str; 4] = ["end_game", "endGame", "game_end", "end"];
/// Non-decision events: ordering/round tracking only, never rows.
pub const SKIP_TYPES: [&str; 10] = [
    "start_kyoku",
    "tsumo",
    "dora",
    "reach_accepted",
    "ryukyoku",
    "end_kyoku",
    "start_game",
    "startGame",
    "game_start",
    "start",
];
/// One row per occurrence (`reach` collapses with its following `dahai`).
pub const ROW_TYPES: [&str; 7] = ["dahai", "chi", "pon", "daiminkan", "ankan", "kakan", "hora"];
/// Row kinds that claim the live discard offer.
pub const CLAIM_TYPES: [&str; 3] = ["chi", "pon", "daiminkan"];
/// Log event kinds skipped when looking behind/ahead through a kyoku.
pub const TRANSPARENT_KINDS: [&str; 2] = ["dora", "reach_accepted"];

/// Interned kind code for a decoded `type` string (unknown → [`KIND_OTHER`]).
pub fn kind_from_bytes(ty: &[u8]) -> u8 {
    if ty == b"start_game" || ty == b"startGame" || ty == b"game_start" || ty == b"start" {
        KIND_START
    } else if ty == b"end_game" || ty == b"endGame" || ty == b"game_end" || ty == b"end" {
        KIND_END
    } else if ty == b"dahai" {
        KIND_DAHAI
    } else if ty == b"chi" {
        KIND_CHI
    } else if ty == b"pon" {
        KIND_PON
    } else if ty == b"daiminkan" {
        KIND_DAIMINKAN
    } else if ty == b"ankan" {
        KIND_ANKAN
    } else if ty == b"kakan" {
        KIND_KAKAN
    } else if ty == b"hora" {
        KIND_HORA
    } else if ty == b"dora" {
        KIND_DORA
    } else if ty == b"reach" {
        KIND_REACH
    } else if ty == b"reach_accepted" {
        KIND_REACH_ACCEPTED
    } else if ty == b"tsumo" {
        KIND_TSUMO
    } else if ty == b"start_kyoku" {
        KIND_START_KYOKU
    } else if ty == b"end_kyoku" {
        KIND_END_KYOKU
    } else if ty == b"ryukyoku" {
        KIND_RYUKYOKU
    } else {
        KIND_OTHER
    }
}

/// Boundary class: 1 = canonical start, 2 = bare `start` alias,
/// 3 = canonical end, 4 = bare `end` alias, 0 = not a boundary.
/// The exactly-one rule counts canonical spellings only (1/3), mirroring
/// both `decode.py` and the legacy `frame_events`.
pub(crate) fn boundary_class(ty: &[u8]) -> u8 {
    if ty == b"start_game" || ty == b"startGame" || ty == b"game_start" {
        1
    } else if ty == b"start" {
        2
    } else if ty == b"end_game" || ty == b"endGame" || ty == b"game_end" {
        3
    } else if ty == b"end" {
        4
    } else {
        0
    }
}

/// One stacked MJAI event over arena spans (kind + seats + tile/consumed/tsumogiri + raw spans).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackedEvent<'a> {
    /// Kind code (`KIND_*`).
    pub kind: u8,
    /// Actor seat `0..=3`, else [`NO_SEAT`].
    pub actor: u8,
    /// Target seat `0..=3`, else [`NO_SEAT`].
    pub target: u8,
    /// Physical tile id, else [`NO_PAI`].
    pub pai: u8,
    /// Claimed tiles + length.
    pub consumed: ([u8; 4], u8),
    /// Dahai tsumogiri flag.
    pub tsumogiri: bool,
    /// Raw line bytes (gate scans `deltas` / bare-alias type here, zero-alloc).
    pub span: &'a [u8],
    /// `dora_marker` bytes when present and non-empty.
    pub dora_span: Option<&'a [u8]>,
    /// Wall-list bytes when the line carries one.
    pub wall_span: Option<&'a [u8]>,
}

/// One framed game borrowing the ingest arena (stable index + stacked events + optional wall).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramedGame<'a> {
    /// Stable game index (sink join key).
    pub game_idx: u32,
    /// Hashed object id (numeric; string ids live cold-side only).
    pub object_id: u32,
    /// Stacked events in log order.
    pub events: Vec<StackedEvent<'a>>,
    /// Validated 136-permutation wall, when the payload carries one.
    pub wall: Option<[u8; 136]>,
    /// Opaque game key (seed material, never a display id).
    pub game_key: u64,
}
