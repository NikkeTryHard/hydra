//! MJAI replay parsing helpers and event definitions.

use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::action::{Action as EnvAction, ActionType};
use crate::errors::{RiichiError, RiichiResult};
use crate::parser::mjai_to_tid;

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(feature = "python")]
use std::sync::Arc;

#[cfg(feature = "python")]
use crate::replay::{Action, HuleData, LogKyoku};
#[cfg(feature = "python")]
use crate::types::MeldType;

fn parse_mjai_tile_checked(tile: &str) -> RiichiResult<u8> {
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

#[cfg(feature = "python")]
fn parse_mjai_tile(s: &str) -> u8 {
    mjai_to_tid(s).unwrap_or(0)
}

#[cfg(feature = "python")]
#[pyclass]
pub struct MjaiReplay {
    pub rounds: Vec<LogKyoku>,
}

#[cfg(feature = "python")]
#[derive(Debug)]
#[pyclass]
pub struct KyokuIterator {
    game: Py<MjaiReplay>,
    index: usize,
    len: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl KyokuIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<LogKyoku> {
        if slf.index >= slf.len {
            return None;
        }

        let kyoku = {
            let game = slf.game.borrow(slf.py());
            game.rounds[slf.index].clone()
        };
        slf.index += 1;

        Some(kyoku)
    }
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
        let event = serde_json::from_str(trimmed).map_err(|err| RiichiError::Parse {
            input: format!("line {}", line_no + 1),
            message: err.to_string(),
        })?;
        events.push(event);
    }
    Ok(events)
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

#[cfg(feature = "python")]
struct KyokuBuilder {
    actions: Vec<Action>,
    scores: Vec<i32>,
    end_scores: Vec<i32>,
    hands: Vec<Vec<u8>>,
    doras: Vec<u8>,
    chang: u8,
    ju: u8,
    ben: u8,
    liqibang: u8,
    left_tile_count: u8,
    ura_doras: Vec<u8>,

    // Internal tracking
    liqi_flags: Vec<bool>, // Who has declared reach (to set `is_liqi` on discard)
    wliqi_flags: Vec<bool>, // Who has effectively achieved Double Riichi
    reach_accepted: Vec<bool>,
    reached: Vec<bool>, // Tracks if player declared reach (for riichi cost in end_scores)
    first_discard: Vec<bool>,
    has_calls: bool,
    pending_hule: Vec<HuleData>, // Buffer for batching consecutive hora events (double/triple ron)
}

#[cfg(feature = "python")]
impl KyokuBuilder {
    fn new(
        bakaze: String,
        kyoku: u8,
        honba: u8,
        kyoutaku: u8,
        scores: Vec<i32>,
        dora_marker: String,
        tehais: Vec<Vec<String>>,
    ) -> Self {
        let chang = match bakaze.as_str() {
            "S" => 1,
            "W" => 2,
            "N" => 3,
            _ => 0, // "E" or default
        };
        let ju = kyoku - 1;

        let mut hands = vec![Vec::new(); 4];
        for (i, tehai_strs) in tehais.iter().enumerate() {
            if i < 4 {
                hands[i] = tehai_strs.iter().map(|s| parse_mjai_tile(s)).collect();
            }
        }

        let first_dora = parse_mjai_tile(&dora_marker);
        let end_scores = scores.clone();

        KyokuBuilder {
            actions: Vec::new(),
            scores,
            end_scores,
            hands,
            doras: vec![first_dora],
            chang,
            ju,
            ben: honba,
            liqibang: kyoutaku,
            left_tile_count: 70, // Standard starting count?
            ura_doras: Vec::new(),
            liqi_flags: vec![false; 4],
            wliqi_flags: vec![false; 4],
            reach_accepted: vec![false; 4],
            reached: vec![false; 4],
            first_discard: vec![true; 4],
            has_calls: false,
            pending_hule: Vec::new(),
        }
    }

    fn flush_pending_hule(&mut self) {
        if !self.pending_hule.is_empty() {
            let hules = std::mem::take(&mut self.pending_hule);
            self.actions.push(Action::Hule { hules });
        }
    }

    fn build(mut self) -> LogKyoku {
        self.flush_pending_hule();
        LogKyoku {
            scores: self.scores,
            end_scores: self.end_scores,
            doras: self.doras,
            ura_doras: self.ura_doras,
            hands: self.hands,
            chang: self.chang,
            ju: self.ju,
            ben: self.ben,
            liqibang: self.liqibang,
            left_tile_count: self.left_tile_count,
            wliqi: self.wliqi_flags,
            paishan: None, // MJAI usually doesn't have full paishan
            actions: Arc::from(self.actions),
            rule: crate::rule::GameRule::default_mortal(),
            game_end_scores: None,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl MjaiReplay {
    #[staticmethod]
    pub fn from_jsonl(path: String) -> PyResult<Self> {
        let file = File::open(&path)
            .map_err(|e| PyValueError::new_err(format!("Failed to open file: {}", e)))?;
        let mut buf_reader = BufReader::new(file);

        // Detect gzip by magic bytes (0x1f 0x8b) instead of extension
        let is_gzip = {
            let buf = buf_reader
                .fill_buf()
                .map_err(|e| PyValueError::new_err(format!("Failed to peek file: {}", e)))?;
            buf.len() >= 2 && buf[0] == 0x1f && buf[1] == 0x8b
        };

        let reader: Box<dyn BufRead> = if is_gzip {
            let decoder = GzDecoder::new(buf_reader);
            Box::new(BufReader::new(decoder))
        } else {
            Box::new(buf_reader)
        };

        let mut rounds = Vec::new();
        let mut builder: Option<KyokuBuilder> = None;

        for line in reader.lines() {
            let line = line.map_err(|e| PyValueError::new_err(format!("Read error: {}", e)))?;
            if line.trim().is_empty() {
                continue;
            }
            let event: MjaiEvent = serde_json::from_str(&line)
                .map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;

            match event {
                MjaiEvent::StartKyoku {
                    bakaze,
                    kyoku,
                    honba,
                    kyoutaku,
                    scores,
                    dora_marker,
                    tehais,
                    ..
                } => {
                    // Start new LogKyoku
                    if let Some(b) = builder.take() {
                        rounds.push(b.build());
                    }
                    builder = Some(KyokuBuilder::new(
                        bakaze,
                        kyoku,
                        honba,
                        kyoutaku,
                        scores,
                        dora_marker,
                        tehais,
                    ));
                }
                MjaiEvent::EndKyoku | MjaiEvent::EndGame => {
                    if let Some(b) = builder.take() {
                        rounds.push(b.build());
                    }
                }
                _ => {
                    if let Some(ref mut b) = builder {
                        Self::process_event(b, event);
                    }
                }
            }
        }

        // Final flush if unexpected end
        if let Some(b) = builder.take() {
            rounds.push(b.build());
        }

        Ok(MjaiReplay { rounds })
    }

    fn num_rounds(&self) -> usize {
        self.rounds.len()
    }

    fn take_kyokus(slf: Py<Self>, py: Python<'_>) -> PyResult<KyokuIterator> {
        let logs_len = slf.borrow(py).rounds.len();
        Ok(KyokuIterator {
            game: slf,
            index: 0,
            len: logs_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::{Cursor, Write};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn read_mjai_events_parses_jsonl() {
        let log = concat!(
            "{\"type\":\"start_game\"}\n",
            "{\"type\":\"reach\",\"actor\":2}\n",
            "{\"type\":\"dahai\",\"actor\":2,\"pai\":\"5pr\",\"tsumogiri\":true}\n"
        );
        let events = read_mjai_events(Cursor::new(log)).expect("read events");
        assert_eq!(events.len(), 3);
        assert_eq!(mjai_event_actor(&events[1]), Some(2));
    }

    #[test]
    fn load_mjai_events_from_gzip_path_parses_jsonl() {
        let path = std::env::temp_dir().join(format!(
            "hydra_engine_mjai_events_{}_{}.json.gz",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        let file = File::create(&path).expect("create gzip log");
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder
            .write_all(b"{\"type\":\"start_game\"}\n{\"type\":\"end_game\"}\n")
            .expect("write gzip log");
        encoder.finish().expect("finish gzip log");

        let events = load_mjai_events_from_path(&path).expect("load gz events");
        std::fs::remove_file(&path).expect("remove temp log");

        assert_eq!(events.len(), 2);
    }

    #[test]
    fn mjai_event_to_action_preserves_tiles_and_actor() {
        let discard = MjaiEvent::Dahai {
            actor: 1,
            pai: "5pr".to_string(),
            tsumogiri: true,
        };
        let action = mjai_event_to_action(&discard)
            .expect("convert discard")
            .expect("discard action");
        assert_eq!(action.actor, Some(1));
        assert_eq!(action.action_type, ActionType::Discard);
        assert_eq!(action.tile, Some(52));

        let hora = MjaiEvent::Hora {
            actor: 3,
            target: 1,
            pai: Some("C".to_string()),
            uradora_markers: None,
            yaku: None,
            fu: None,
            han: None,
            scores: None,
            delta: None,
        };
        let action = mjai_event_to_action(&hora)
            .expect("convert hora")
            .expect("hora action");
        assert_eq!(action.actor, Some(3));
        assert_eq!(action.action_type, ActionType::Ron);
        assert_eq!(action.tile, Some(132));
    }
}

#[cfg(feature = "python")]
impl MjaiReplay {
    fn process_event(builder: &mut KyokuBuilder, event: MjaiEvent) {
        // Flush pending hora batch before any non-Hora event
        if !matches!(event, MjaiEvent::Hora { .. }) {
            builder.flush_pending_hule();
        }

        match event {
            MjaiEvent::Tsumo { actor, pai } => {
                let tile = parse_mjai_tile(&pai);
                builder.actions.push(Action::DealTile {
                    seat: actor,
                    tile,
                    doras: None,
                    left_tile_count: None, // Could decrement builder.left_tile_count?
                });
                if builder.left_tile_count > 0 {
                    builder.left_tile_count -= 1;
                }
            }
            MjaiEvent::Dahai {
                actor,
                pai,
                tsumogiri: _,
            } => {
                let tile = parse_mjai_tile(&pai);
                let is_liqi = builder.liqi_flags[actor];

                let is_wliqi = is_liqi && builder.first_discard[actor] && !builder.has_calls;
                if is_wliqi {
                    builder.wliqi_flags[actor] = true;
                }

                builder.actions.push(Action::DiscardTile {
                    seat: actor,
                    tile,
                    is_liqi,
                    is_wliqi,
                    doras: None,
                });

                builder.first_discard[actor] = false;

                if is_liqi {
                    builder.liqibang += 1;
                    builder.liqi_flags[actor] = false;
                }
            }
            MjaiEvent::Reach { actor } => {
                builder.liqi_flags[actor] = true;
                builder.reached[actor] = true;
            }
            MjaiEvent::ReachAccepted { actor } => {
                builder.reach_accepted[actor] = true;
            }
            MjaiEvent::Chi {
                actor,
                pai,
                consumed,
                ..
            } => {
                builder.has_calls = true;
                let mut tiles = vec![parse_mjai_tile(&pai)];
                for c in consumed {
                    tiles.push(parse_mjai_tile(&c));
                }
                builder.actions.push(Action::ChiPengGang {
                    seat: actor,
                    meld_type: MeldType::Chi,
                    tiles,
                    froms: vec![],
                });
            }
            MjaiEvent::Pon {
                actor,
                pai,
                consumed,
                ..
            } => {
                builder.has_calls = true;
                let mut tiles = vec![parse_mjai_tile(&pai)];
                for c in consumed {
                    tiles.push(parse_mjai_tile(&c));
                }
                builder.actions.push(Action::ChiPengGang {
                    seat: actor,
                    meld_type: MeldType::Pon,
                    tiles,
                    froms: vec![],
                });
            }
            MjaiEvent::Kan {
                actor,
                pai,
                consumed,
                ..
            } => {
                builder.has_calls = true;
                // Daiminkan
                let mut tiles = vec![parse_mjai_tile(&pai)];
                for c in consumed {
                    tiles.push(parse_mjai_tile(&c));
                }
                builder.actions.push(Action::ChiPengGang {
                    seat: actor,
                    meld_type: MeldType::Daiminkan,
                    tiles,
                    froms: vec![],
                });
            }
            MjaiEvent::Ankan { actor, consumed } => {
                builder.has_calls = true;
                let tiles: Vec<u8> = consumed.iter().map(|s| parse_mjai_tile(s)).collect();
                builder.actions.push(Action::AnGangAddGang {
                    seat: actor,
                    meld_type: MeldType::Ankan,
                    tiles,
                    tile_raw_id: 0,
                    doras: None,
                });
            }
            MjaiEvent::Kakan { actor, pai } => {
                builder.has_calls = true;
                let tile = parse_mjai_tile(&pai);
                builder.actions.push(Action::AnGangAddGang {
                    seat: actor,
                    meld_type: MeldType::Kakan,
                    tiles: vec![tile],
                    tile_raw_id: 0,
                    doras: None,
                });
            }
            MjaiEvent::Dora { dora_marker } => {
                let marker = parse_mjai_tile(&dora_marker);
                builder.doras.push(marker);
                builder.actions.push(Action::Dora {
                    dora_marker: marker,
                });
            }
            MjaiEvent::Hora {
                actor,
                target,
                pai,
                uradora_markers,
                yaku: _,
                fu,
                han,
                scores,
                delta,
            } => {
                let hu_tile_id = if let Some(p) = pai {
                    parse_mjai_tile(&p)
                } else {
                    // Try to infer from last action
                    // If Tsumo (actor == target), last action should be DealTile for actor
                    // If Ron (actor != target), last action should be DiscardTile
                    // Simplifying assumption: look at last action
                    if let Some(last_action) = builder.actions.last() {
                        match last_action {
                            Action::DealTile { tile, .. } => *tile,
                            Action::DiscardTile { tile, .. } => *tile,
                            // Check for AddGang too (Chankan)?
                            Action::AnGangAddGang { tiles, .. } => tiles[0], // AddGang
                            _ => 0, // Fallback, though ideally shouldn't happen
                        }
                    } else {
                        0
                    }
                };

                let mut hule_data = HuleData {
                    seat: actor,
                    hu_tile: hu_tile_id,
                    zimo: actor == target, // If actor is target, it's Tsumo
                    count: han.unwrap_or(0),
                    fu: fu.unwrap_or(0),
                    fans: Vec::new(),
                    li_doras: None,
                    yiman: false,
                    point_rong: 0,
                    point_zimo_qin: 0,
                    point_zimo_xian: 0,
                };

                if let Some(uras) = uradora_markers {
                    let ud: Vec<u8> = uras.iter().map(|s| parse_mjai_tile(s)).collect();
                    builder.ura_doras = ud.clone();
                    hule_data.li_doras = Some(ud);
                }

                // Update end_scores: accumulate deltas for double/triple ron
                if let Some(s) = scores {
                    builder.end_scores = s;
                } else if let Some(d) = delta {
                    let is_first_hora = builder.pending_hule.is_empty();
                    for (i, val) in d.iter().enumerate() {
                        if i < builder.end_scores.len() {
                            if is_first_hora {
                                // First hora: initialize from starting scores.
                                // Use reach_accepted (not reached) because riichi
                                // that was ronned on the declaration tile is never
                                // accepted and the 1000 deposit is not paid.
                                let riichi_cost = if builder.reach_accepted[i] { 1000 } else { 0 };
                                builder.end_scores[i] = builder.scores[i] + val - riichi_cost;
                            } else {
                                // Subsequent hora: add delta to existing end_scores
                                builder.end_scores[i] += val;
                            }
                        }
                    }
                }

                // Buffer the hora for batching (double/triple ron)
                builder.pending_hule.push(hule_data);
            }
            MjaiEvent::Kita { actor: _ } => {
                // Kita is treated as a special action; no separate Action variant needed
                // The event handler handles it via MjaiEvent directly
            }
            MjaiEvent::Ryukyoku { delta, scores, .. } => {
                if let Some(s) = scores {
                    builder.end_scores = s;
                } else if let Some(d) = delta {
                    for (i, val) in d.iter().enumerate() {
                        if i < builder.end_scores.len() {
                            let riichi_cost = if builder.reached[i] { 1000 } else { 0 };
                            builder.end_scores[i] = builder.scores[i] + val - riichi_cost;
                        }
                    }
                }
                builder.actions.push(Action::NoTile);
            }
            _ => {}
        }
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    // Test disabled due to linking issues in CI environment (missing python symbols).
    // To run locally, ensure binding env is set up.
    // #[test]
    fn test_mjai_parsing() {
        let json_data = r#"
{"type":"start_game"}
{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyoutaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1s","1s","1s","2s","3s","4s","5s","6s","7s","8s","9s","9s","9s"],["1s","1s","1s","2s","3s","4s","5s","6s","7s","8s","9s","9s","9s"],["1s","1s","1s","2s","3s","4s","5s","6s","7s","8s","9s","9s","9s"],["1s","1s","1s","2s","3s","4s","5s","6s","7s","8s","9s","9s","9s"]]}
{"type":"tsumo","actor":0,"pai":"2m"}
{"type":"dahai","actor":0,"pai":"2m","tsumogiri":false}
{"type":"ryukyoku","reason":"fanpai"}
{"type":"end_kyoku"}
{"type":"end_game"}
"#;
        let mut path = std::env::temp_dir();
        path.push("test_mjai.jsonl");
        let mut file = File::create(&path).unwrap();
        writeln!(file, "{}", json_data.trim()).unwrap();

        let path_str = path.to_str().unwrap().to_string();

        let replay = MjaiReplay::from_jsonl(path_str.clone()).expect("Failed to parse MJAI");
        assert_eq!(replay.rounds.len(), 1);
        let kyoku = &replay.rounds[0];
        assert_eq!(kyoku.actions.len(), 3);

        // ... (assertions)

        let _ = std::fs::remove_file(path);
    }
}
*/
