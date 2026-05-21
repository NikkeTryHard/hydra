//! MJAI replay parsing helpers and event definitions.

use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::action::{Action as EnvAction, ActionType};
use crate::errors::{RiichiError, RiichiResult};
use crate::parser::mjai_to_tid;

use serde::{Deserialize, Serialize};

pub fn parse_mjai_tile_checked(tile: &str) -> RiichiResult<u8> {
    mjai_to_tid(tile).ok_or_else(|| RiichiError::Parse {
        input: tile.to_string(),
        message: "invalid MJAI tile".to_string(),
    })
}

fn parse_consumed_tiles(consumed: &[String]) -> RiichiResult<([u8; 4], usize)> {
    if consumed.len() > 4 {
        return Err(RiichiError::InvalidAction {
            message: format!("MJAI meld consumed too many tiles: {}", consumed.len()),
        });
    }

    let mut tiles = [0u8; 4];
    for (idx, tile) in consumed.iter().enumerate() {
        tiles[idx] = parse_mjai_tile_checked(tile)?;
    }
    Ok((tiles, consumed.len()))
}

// MJAI Event Definitions
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum MjaiEvent {
    #[serde(rename = "start_game")]
    StartGame {
        names: Option<Vec<String>>,
        id: Option<String>,
    },
    #[serde(rename = "start_kyoku")]
    StartKyoku {
        bakaze: String,
        kyoku: u8,
        honba: u8,
        #[serde(alias = "kyotaku")]
        kyoutaku: u8,
        oya: u8,
        scores: Vec<i32>,
        dora_marker: String,
        tehais: Vec<Vec<String>>,
    },
    #[serde(rename = "tsumo")]
    Tsumo { actor: usize, pai: String },
    #[serde(rename = "dahai")]
    Dahai {
        actor: usize,
        pai: String,
        tsumogiri: bool,
    },
    #[serde(rename = "pon")]
    Pon {
        actor: usize,
        target: usize,
        pai: String,
        consumed: Vec<String>,
    },
    #[serde(rename = "chi")]
    Chi {
        actor: usize,
        target: usize,
        pai: String,
        consumed: Vec<String>,
    },
    #[serde(rename = "kan", alias = "daiminkan")]
    Kan {
        actor: usize,
        target: usize,
        pai: String,
        consumed: Vec<String>,
    },
    #[serde(rename = "kakan")]
    Kakan { actor: usize, pai: String },
    #[serde(rename = "ankan")]
    Ankan { actor: usize, consumed: Vec<String> },
    #[serde(rename = "dora")]
    Dora { dora_marker: String },
    #[serde(rename = "reach")]
    Reach { actor: usize },
    #[serde(rename = "reach_accepted")]
    ReachAccepted { actor: usize },
    #[serde(rename = "hora")]
    Hora {
        actor: usize,
        target: usize,
        pai: Option<String>, // Winning tile (optional in some logs)
        #[serde(alias = "ura_markers")]
        uradora_markers: Option<Vec<String>>,
        #[serde(default)]
        yaku: Option<Vec<(String, u32)>>, // List of [yaku_name, han_value]
        fu: Option<u32>,
        han: Option<u32>,
        #[serde(default)]
        scores: Option<Vec<i32>>, // Scores AFTER hor
        #[serde(alias = "deltas")]
        delta: Option<Vec<i32>>,
    },
    #[serde(rename = "ryukyoku")]
    Ryukyoku {
        reason: Option<String>,
        tehais: Option<Vec<Vec<String>>>, // Revealed hands
        #[serde(alias = "deltas")]
        delta: Option<Vec<i32>>,
        scores: Option<Vec<i32>>,
    },
    #[serde(rename = "kita")]
    Kita { actor: usize },
    #[serde(rename = "end_game")]
    EndGame,
    #[serde(rename = "end_kyoku")]
    EndKyoku,
    #[serde(other)]
    Other,
}

/// Returns the acting seat for an actionable MJAI event.
#[inline]
pub fn mjai_event_actor(event: &MjaiEvent) -> Option<usize> {
    match event {
        MjaiEvent::Dahai { actor, .. }
        | MjaiEvent::Pon { actor, .. }
        | MjaiEvent::Chi { actor, .. }
        | MjaiEvent::Kan { actor, .. }
        | MjaiEvent::Kakan { actor, .. }
        | MjaiEvent::Ankan { actor, .. }
        | MjaiEvent::Reach { actor }
        | MjaiEvent::Hora { actor, .. } => Some(*actor),
        _ => None,
    }
}

/// Converts an actionable MJAI event into the equivalent engine action.
pub fn mjai_event_to_action(event: &MjaiEvent) -> RiichiResult<Option<EnvAction>> {
    let action = match event {
        MjaiEvent::Dahai { actor, pai, .. } => EnvAction::new(
            ActionType::Discard,
            Some(parse_mjai_tile_checked(pai)?),
            &[],
            Some(*actor as u8),
        ),
        MjaiEvent::Pon {
            actor,
            pai,
            consumed,
            ..
        } => {
            let (tiles, len) = parse_consumed_tiles(consumed)?;
            EnvAction::new(
                ActionType::Pon,
                Some(parse_mjai_tile_checked(pai)?),
                &tiles[..len],
                Some(*actor as u8),
            )
        }
        MjaiEvent::Chi {
            actor,
            pai,
            consumed,
            ..
        } => {
            let (tiles, len) = parse_consumed_tiles(consumed)?;
            EnvAction::new(
                ActionType::Chi,
                Some(parse_mjai_tile_checked(pai)?),
                &tiles[..len],
                Some(*actor as u8),
            )
        }
        MjaiEvent::Kan {
            actor,
            pai,
            consumed,
            ..
        } => {
            let (tiles, len) = parse_consumed_tiles(consumed)?;
            EnvAction::new(
                ActionType::Daiminkan,
                Some(parse_mjai_tile_checked(pai)?),
                &tiles[..len],
                Some(*actor as u8),
            )
        }
        MjaiEvent::Kakan { actor, pai } => EnvAction::new(
            ActionType::Kakan,
            Some(parse_mjai_tile_checked(pai)?),
            &[],
            Some(*actor as u8),
        ),
        MjaiEvent::Ankan { actor, consumed } => {
            let (tiles, len) = parse_consumed_tiles(consumed)?;
            EnvAction::new(
                ActionType::Ankan,
                consumed
                    .first()
                    .map(|tile| parse_mjai_tile_checked(tile))
                    .transpose()?,
                &tiles[..len],
                Some(*actor as u8),
            )
        }
        MjaiEvent::Reach { actor } => {
            EnvAction::new(ActionType::Riichi, None, &[], Some(*actor as u8))
        }
        MjaiEvent::Hora {
            actor, target, pai, ..
        } => EnvAction::new(
            if actor == target {
                ActionType::Tsumo
            } else {
                ActionType::Ron
            },
            pai.as_deref().map(parse_mjai_tile_checked).transpose()?,
            &[],
            Some(*actor as u8),
        ),
        _ => return Ok(None),
    };
    Ok(Some(action))
}

/// Reads line-delimited MJAI events from a buffered reader.
pub fn read_mjai_events<R: BufRead>(reader: R) -> RiichiResult<Vec<MjaiEvent>> {
    let mut events = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|err| RiichiError::Serialization {
            message: format!("failed to read MJAI line {}: {err}", line_no + 1),
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let event = parse_mjai_line(trimmed).map_err(|msg| RiichiError::Parse {
            input: format!("line {}", line_no + 1),
            message: msg,
        })?;
        events.push(event);
    }
    Ok(events)
}

#[cfg(feature = "fast-json-ingest")]
#[inline]
fn parse_mjai_line(line: &str) -> Result<MjaiEvent, String> {
    sonic_rs::from_str(line).map_err(|err| err.to_string())
}

#[cfg(not(feature = "fast-json-ingest"))]
#[inline]
fn parse_mjai_line(line: &str) -> Result<MjaiEvent, String> {
    serde_json::from_str(line).map_err(|err| err.to_string())
}

/// Loads line-delimited MJAI events from a file path, transparently handling gzip.
pub fn load_mjai_events_from_path(path: impl AsRef<Path>) -> RiichiResult<Vec<MjaiEvent>> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|err| RiichiError::Serialization {
        message: format!("failed to open {}: {err}", path.display()),
    })?;
    let mut reader = BufReader::new(file);
    let is_gzip = {
        let buf = reader
            .fill_buf()
            .map_err(|err| RiichiError::Serialization {
                message: format!("failed to inspect {}: {err}", path.display()),
            })?;
        buf.starts_with(&[0x1f, 0x8b])
    };
    if is_gzip {
        return read_mjai_events(BufReader::new(GzDecoder::new(reader)));
    }
    read_mjai_events(reader)
}
