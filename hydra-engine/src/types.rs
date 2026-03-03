#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// The total number of distinct tile types in mahjong (0-33).
pub const TILE_MAX: usize = 34;

/// A hand representation using a histogram of tile types (0-33).
#[derive(Debug, Clone)]
pub struct Hand {
    /// The tile-type histogram (index = tile type 0-33, value = count).
    pub counts: [u8; TILE_MAX],
}

impl Hand {
    /// Creates a new hand, optionally populated from a list of tile types.
    pub fn new(tiles: Option<Vec<u8>>) -> Self {
        let mut h = Hand {
            counts: [0; TILE_MAX],
        };
        if let Some(ts) = tiles {
            for t in ts {
                h.add(t);
            }
        }
        h
    }

    /// Adds a tile type to the hand, incrementing its count.
    pub fn add(&mut self, t: u8) {
        if (t as usize) < TILE_MAX {
            self.counts[t as usize] += 1;
        }
    }

    /// Removes a tile type from the hand, decrementing its count if present.
    pub fn remove(&mut self, t: u8) {
        if (t as usize) < TILE_MAX && self.counts[t as usize] > 0 {
            self.counts[t as usize] -= 1;
        }
    }

    fn __str__(&self) -> String {
        format!("Hand(counts={:?})", &self.counts[..])
    }
}

impl Default for Hand {
    fn default() -> Self {
        Hand {
            counts: [0; TILE_MAX],
        }
    }
}

/// The type of meld (open/closed group of tiles).
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeldType {
    /// A sequence of three consecutive suited tiles claimed from the left player.
    Chi = 0,
    /// A triplet formed by claiming another player's discard.
    Pon = 1,
    /// An open quad formed by claiming another player's discard.
    Daiminkan = 2,
    /// A concealed quad declared from four identical tiles in hand.
    Ankan = 3,
    /// An added quad formed by upgrading a pon with the fourth tile.
    Kakan = 4,
}

/// Represents wind directions in mahjong, used for player seats and round wind.
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Wind {
    #[default]
    /// East wind (ton).
    East = 0,
    /// South wind (nan).
    South = 1,
    /// West wind (shaa).
    West = 2,
    /// North wind (pei).
    North = 3,
}

impl From<u8> for Wind {
    fn from(val: u8) -> Self {
        match val % 4 {
            0 => Wind::East,
            1 => Wind::South,
            2 => Wind::West,
            3 => Wind::North,
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Wind {
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

/// A declared meld (chi, pon, or kan) consisting of up to 4 tiles.
#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Meld {
    /// The type of this meld (chi, pon, daiminkan, ankan, kakan).
    pub meld_type: MeldType,
    /// The tile IDs in this meld (136-format), padded with zeros.
    pub tiles: [u8; 4],
    /// The number of active tiles in the `tiles` array.
    pub tile_count: u8,
    /// Whether this meld is open (visible to other players).
    pub opened: bool,
    /// The relative seat of the player the tile was claimed from (-1 if N/A).
    pub from_who: i8,
    /// The tile claimed from another player's discard (for chi/pon/daiminkan).
    /// None for ankan/kakan or melds not involving a discard claim.
    pub called_tile: Option<u8>,
}

impl Default for Meld {
    fn default() -> Self {
        Self {
            meld_type: MeldType::Chi,
            tiles: [0; 4],
            tile_count: 0,
            opened: false,
            from_who: 0,
            called_tile: None,
        }
    }
}

impl Meld {
    /// Create a new Meld from a tile slice (max 4 tiles).
    pub fn new(
        meld_type: MeldType,
        tiles: &[u8],
        opened: bool,
        from_who: i8,
        called_tile: Option<u8>,
    ) -> Self {
        let mut arr = [0u8; 4];
        let count = tiles.len().min(4);
        arr[..count].copy_from_slice(&tiles[..count]);
        Self { meld_type, tiles: arr, tile_count: count as u8, opened, from_who, called_tile }
    }

    /// Get the active tiles as a slice.
    #[inline]
    pub fn tiles_slice(&self) -> &[u8] {
        &self.tiles[..self.tile_count as usize]
    }

    /// Get the active tiles as a mutable slice.
    #[inline]
    pub fn tiles_slice_mut(&mut self) -> &mut [u8] {
        &mut self.tiles[..self.tile_count as usize]
    }

    /// Push a tile (for kakan upgrade: 3->4).
    #[inline]
    pub fn push_tile(&mut self, tile: u8) {
        self.tiles[self.tile_count as usize] = tile;
        self.tile_count += 1;
    }

    /// Return tiles as u32 vec (for Python compatibility).
    pub fn tiles_as_u32(&self) -> Vec<u32> {
        self.tiles_slice().iter().map(|&t| t as u32).collect()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Meld {
    #[new]
    #[pyo3(signature = (meld_type, tiles, opened, from_who=-1, called_tile=None))]
    pub fn py_new(
        meld_type: MeldType,
        tiles: Vec<u8>,
        opened: bool,
        from_who: i8,
        called_tile: Option<u8>,
    ) -> Self {
        Self::new(meld_type, &tiles, opened, from_who, called_tile)
    }

    #[getter]
    pub fn get_meld_type(&self) -> MeldType {
        self.meld_type
    }

    #[setter]
    pub fn set_meld_type(&mut self, meld_type: MeldType) {
        self.meld_type = meld_type;
    }

    #[getter]
    pub fn get_tiles(&self) -> Vec<u32> {
        self.tiles_as_u32()
    }

    #[setter]
    pub fn set_tiles(&mut self, tiles: Vec<u8>) {
        let count = tiles.len().min(4);
        self.tiles = [0u8; 4];
        self.tiles[..count].copy_from_slice(&tiles[..count]);
        self.tile_count = count as u8;
    }

    #[getter]
    pub fn get_opened(&self) -> bool {
        self.opened
    }

    #[setter]
    pub fn set_opened(&mut self, opened: bool) {
        self.opened = opened;
    }

    #[getter]
    pub fn get_from_who(&self) -> i8 {
        self.from_who
    }

    #[setter]
    pub fn set_from_who(&mut self, from_who: i8) {
        self.from_who = from_who;
    }

    #[getter]
    pub fn get_called_tile(&self) -> Option<u8> {
        self.called_tile
    }
}

/// Contextual conditions for hand evaluation (win method, special states, round info).
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone)]
pub struct Conditions {
    /// Whether the win was by self-draw (tsumo).
    pub tsumo: bool,
    /// Whether the player declared riichi.
    pub riichi: bool,
    /// Whether the player declared double riichi (riichi on first turn).
    pub double_riichi: bool,
    /// Whether the win occurred within one turn of declaring riichi.
    pub ippatsu: bool,
    /// Whether the win was on the last drawable tile (haitei raoyue).
    pub haitei: bool,
    /// Whether the win was on the last discarded tile (houtei raoyui).
    pub houtei: bool,
    /// Whether the winning tile was drawn from the dead wall after a kan.
    pub rinshan: bool,
    /// The player's seat wind.
    pub player_wind: Wind,
    /// The prevailing round wind.
    pub round_wind: Wind,
    /// Whether the win was by robbing a kan (chankan).
    pub chankan: bool,
    /// Whether the win occurred on the very first uninterrupted turn.
    pub tsumo_first_turn: bool,
    /// The number of riichi sticks on the table.
    pub riichi_sticks: u32,
    /// The current honba (repeat counter) for the round.
    pub honba: u32,
    /// The number of kita (north tile) declarations by this player (sanma).
    pub kita_count: u8,
    /// Whether the game is three-player (sanma) mode.
    pub is_sanma: bool,
    /// The number of players in the game (3 or 4).
    pub num_players: u8,
}

impl Default for Conditions {
    fn default() -> Self {
        Self {
            tsumo: false,
            riichi: false,
            double_riichi: false,
            ippatsu: false,
            haitei: false,
            houtei: false,
            rinshan: false,
            player_wind: Wind::East,
            round_wind: Wind::East,
            chankan: false,
            tsumo_first_turn: false,
            riichi_sticks: 0,
            honba: 0,
            kita_count: 0,
            is_sanma: false,
            num_players: 4,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Conditions {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (tsumo=false, riichi=false, double_riichi=false, ippatsu=false, haitei=false, houtei=false, rinshan=false, chankan=false, tsumo_first_turn=false, player_wind=Wind::East, round_wind=Wind::East, riichi_sticks=0, honba=0, kita_count=0, is_sanma=false, num_players=4))]
    pub fn py_new(
        tsumo: bool,
        riichi: bool,
        double_riichi: bool,
        ippatsu: bool,
        haitei: bool,
        houtei: bool,
        rinshan: bool,
        chankan: bool,
        tsumo_first_turn: bool,
        player_wind: Wind,
        round_wind: Wind,
        riichi_sticks: u32,
        honba: u32,
        kita_count: u8,
        is_sanma: bool,
        num_players: u8,
    ) -> Self {
        Self {
            tsumo,
            riichi,
            double_riichi,
            ippatsu,
            haitei,
            houtei,
            rinshan,
            chankan,
            tsumo_first_turn,
            player_wind,
            round_wind,
            riichi_sticks,
            honba,
            kita_count,
            is_sanma,
            num_players,
        }
    }
}

/// The result of evaluating a hand for a win, including score and yaku breakdown.
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[derive(Debug, Clone, Copy)]
pub struct WinResult {
    /// Whether the hand is a valid winning hand.
    pub is_win: bool,
    /// Whether the hand qualifies as yakuman.
    pub yakuman: bool,
    /// The ron payment amount (from the discarder).
    pub ron_agari: u32,
    /// The tsumo payment from the dealer (oya) when a non-dealer wins.
    pub tsumo_agari_oya: u32,
    /// The tsumo payment from each non-dealer (ko).
    pub tsumo_agari_ko: u32,
    /// Fixed-size array of yaku IDs present in the winning hand.
    pub yaku: [u32; 16],
    /// The number of active entries in the `yaku` array.
    pub yaku_count: u8,
    /// The total han (doubles) count.
    pub han: u32,
    /// The fu (minipoints) count.
    pub fu: u32,
    /// The player index responsible for pao (liability), if any.
    pub pao_payer: Option<u8>,
    /// Whether the hand has a valid winning tile grouping (regardless of yaku).
    pub has_win_shape: bool,
}

impl WinResult {
    /// Creates a new win result from all scoring components.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        is_win: bool,
        yakuman: bool,
        ron_agari: u32,
        tsumo_agari_oya: u32,
        tsumo_agari_ko: u32,
        yaku: [u32; 16],
        yaku_count: u8,
        han: u32,
        fu: u32,
        pao_payer: Option<u8>,
        has_win_shape: bool,
    ) -> Self {
        Self {
            is_win,
            yakuman,
            ron_agari,
            tsumo_agari_oya,
            tsumo_agari_ko,
            yaku,
            yaku_count,
            han,
            fu,
            pao_payer,
            has_win_shape,
        }
    }

    /// Get the active yaku IDs as a slice.
    #[inline]
    pub fn yaku_slice(&self) -> &[u32] {
        &self.yaku[..self.yaku_count as usize]
    }

    /// Push a yaku ID into the fixed array.
    #[inline]
    pub fn push_yaku(&mut self, id: u32) {
        self.yaku[self.yaku_count as usize] = id;
        self.yaku_count += 1;
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl WinResult {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (is_win, yakuman=false, ron_agari=0, tsumo_agari_oya=0, tsumo_agari_ko=0, yaku=vec![], han=0, fu=0, pao_payer=None, has_win_shape=false))]
    pub fn py_new(
        is_win: bool,
        yakuman: bool,
        ron_agari: u32,
        tsumo_agari_oya: u32,
        tsumo_agari_ko: u32,
        yaku: Vec<u32>,
        han: u32,
        fu: u32,
        pao_payer: Option<u8>,
        has_win_shape: bool,
    ) -> Self {
        let mut arr = [0u32; 16];
        let count = yaku.len().min(16);
        for (i, &id) in yaku.iter().take(16).enumerate() {
            arr[i] = id;
        }
        Self::new(
            is_win,
            yakuman,
            ron_agari,
            tsumo_agari_oya,
            tsumo_agari_ko,
            arr,
            count as u8,
            han,
            fu,
            pao_payer,
            has_win_shape,
        )
    }

    /// Returns the list of resolved yaku objects for this win.
    pub fn yaku_list(&self) -> Vec<crate::yaku::Yaku> {
        self.yaku_slice()
            .iter()
            .filter_map(|&id| crate::yaku::get_yaku_by_id(id))
            .collect()
    }
}

/// Returns whether a tile ID (0-135) is a terminal or honor tile.
pub fn is_terminal_tile(t: u8) -> bool {
    let t_type = t / 4;
    let rank = t_type % 9;
    let suit = t_type / 9;
    suit == 3 || rank == 0 || rank == 8
}

/// Check if a tile ID (0-135) is excluded in sanma (2m through 8m).
/// Manzu tiles: type 0-8 (IDs 0-35)
/// 1m = type 0 (IDs 0-3) - KEPT
/// 2m = type 1 (IDs 4-7) - EXCLUDED
/// ...
/// 8m = type 7 (IDs 28-31) - EXCLUDED
/// 9m = type 8 (IDs 32-35) - KEPT
pub fn is_sanma_excluded_tile(tile_id: u8) -> bool {
    let tile_type = tile_id / 4;
    // Manzu 2-8 (types 1-7)
    (1..=7).contains(&tile_type)
}

/// Standard dora wrapping for 4-player mahjong.
/// Input/output are tile types (0-33), not tile IDs.
pub fn standard_next_dora_tile(tile: u8) -> u8 {
    match tile {
        // Manzu (0-8): 1m->2m->...->9m->1m
        0..=8 => (tile + 1) % 9,
        // Pinzu (9-17): 1p->2p->...->9p->1p
        9..=17 => 9 + (tile - 9 + 1) % 9,
        // Souzu (18-26): 1s->2s->...->9s->1s
        18..=26 => 18 + (tile - 18 + 1) % 9,
        // Winds (27-30): E->S->W->N->E
        27..=30 => 27 + (tile - 27 + 1) % 4,
        // Dragons (31-33): Haku->Hatsu->Chun->Haku
        31..=33 => 31 + (tile - 31 + 1) % 3,
        _ => tile,
    }
}
