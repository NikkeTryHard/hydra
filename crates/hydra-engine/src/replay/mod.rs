/*
 * replay/mod.rs: Utilities for replaying games to verify the agari calculator.
 */
#![allow(
    clippy::useless_conversion,
    reason = "matches upstream replay conversion behavior"
)]

use std::sync::Arc;

use crate::types::MeldType;

pub mod mjai_replay;
pub mod mjsoul_replay;

pub use mjai_replay::{
    load_mjai_events_from_path, mjai_event_actor, mjai_event_to_action, read_mjai_events, MjaiEvent,
};

#[derive(Clone, Debug)]
pub enum Action {
    DiscardTile {
        seat: usize,
        tile: u8,
        is_liqi: bool,
        is_wliqi: bool,
        doras: Option<Vec<u8>>,
    },
    DealTile {
        seat: usize,
        tile: u8,
        doras: Option<Vec<u8>>,
        left_tile_count: Option<u8>,
    },
    ChiPengGang {
        seat: usize,
        meld_type: MeldType,
        tiles: Vec<u8>,
        froms: Vec<usize>,
    },
    AnGangAddGang {
        seat: usize,
        meld_type: MeldType,
        tiles: Vec<u8>,
        tile_raw_id: u8,
        doras: Option<Vec<u8>>,
    },
    Dora {
        dora_marker: u8,
    },
    Hule {
        hules: Vec<HuleData>,
    },
    NoTile,
    BaBei {
        seat: usize,
        moqie: bool,
    },
    LiuJu {
        lj_type: u8,
        seat: usize,
        tiles: Vec<u8>,
    },
    Other(String),
}

#[derive(Clone, Debug)]
pub struct HuleData {
    pub seat: usize,
    pub hu_tile: u8,
    pub zimo: bool,
    pub count: u32,
    pub fu: u32,
    pub fans: Vec<u32>,
    pub li_doras: Option<Vec<u8>>,
    pub yiman: bool,
    pub point_rong: u32,
    pub point_zimo_qin: u32,
    pub point_zimo_xian: u32,
}

#[derive(Clone)]
pub struct LogKyoku {
    pub scores: Vec<i32>,
    pub doras: Vec<u8>,
    pub ura_doras: Vec<u8>,
    pub hands: Vec<Vec<u8>>,
    pub chang: u8,
    pub ju: u8,
    pub ben: u8,
    pub liqibang: u8,
    pub left_tile_count: u8,
    pub end_scores: Vec<i32>,
    pub wliqi: Vec<bool>,
    pub paishan: Option<String>,
    #[allow(
        dead_code,
        reason = "replay fixtures keep parsed actions for later inspection"
    )]
    pub(crate) actions: Arc<[Action]>,
    pub rule: crate::rule::GameRule,
    pub game_end_scores: Option<Vec<i32>>,
}

pub struct TileConverter {}

impl TileConverter {
    pub fn parse_tile(t: &str) -> (u8, bool) {
        if t.is_empty() {
            return (0, false);
        }
        let (num_str, suit) = t.split_at(1);
        let num: u8 = num_str.parse().unwrap_or(0);
        let is_aka = num == 0;
        let num = if is_aka { 5 } else { num };

        let id_34 = match suit {
            "m" => num - 1,
            "p" => 9 + num - 1,
            "s" => 18 + num - 1,
            "z" => 27 + num - 1,
            _ => 0,
        };

        (id_34, is_aka)
    }

    pub fn parse_tile_34(t: &str) -> (u8, bool) {
        Self::parse_tile(t)
    }

    pub fn parse_tile_136(t: &str) -> u8 {
        let (id_34, is_aka) = Self::parse_tile(t);
        if is_aka {
            match id_34 {
                4 => 16,
                13 => 52,
                22 => 88,
                _ => id_34 * 4,
            }
        } else if id_34 == 4 || id_34 == 13 || id_34 == 22 {
            id_34 * 4 + 1
        } else {
            id_34 * 4
        }
    }

    pub fn to_string(tile: u8) -> String {
        let t34 = tile / 4;
        let is_red = tile == 16 || tile == 52 || tile == 88;
        let suit_idx = t34 / 9;
        let num = t34 % 9 + 1;
        let suit = match suit_idx {
            0 => "m",
            1 => "p",
            2 => "s",
            3 => "z",
            _ => return "?".to_string(),
        };
        if is_red {
            return format!("0{}", suit);
        }
        let res = format!("{}{}", num, suit);
        res
    }

    pub fn match_and_remove_u8(hand: &mut Vec<u8>, target: u8) -> bool {
        if let Some(pos) = hand.iter().position(|x| *x == target) {
            hand.remove(pos);
            return true;
        }
        // Try other 136-ids of the same 34-tile if not found (for robustness)
        let target_34 = target / 4;
        if let Some(pos) = hand.iter().position(|x| *x / 4 == target_34) {
            hand.remove(pos);
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests;
