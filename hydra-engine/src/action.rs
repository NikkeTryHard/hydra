#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyDict, PyDictMethods};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{RiichiError, RiichiResult};
use crate::parser::tid_to_mjai;

/// The number of distinct actions in the 4-player action space.
pub const ACTION_SPACE_4P: usize = 82;
/// The number of distinct actions in the 3-player (sanma) action space.
pub const ACTION_SPACE_3P: usize = 60;

const TILE34_TO_COMPACT: [u8; 34] = [
    0, // type  0: 1m
    255, 255, 255, 255, 255, 255, 255, // type 1-7: 2m-8m (invalid)
    1,   // type  8: 9m
    2, 3, 4, 5, 6, 7, 8, 9, 10, // type  9-17: 1p-9p
    11, 12, 13, 14, 15, 16, 17, 18, 19, // type 18-26: 1s-9s
    20, 21, 22, 23, // type 27-30: ESWN
    24, 25, 26, // type 31-33: PFC
];

/// The current phase of the game turn.
#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", eq, eq_int)
)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    /// Waiting for the active player to choose an action (discard, riichi, kan, tsumo).
    WaitAct = 0,
    /// Waiting for other players to respond to a discard (chi, pon, kan, ron, pass).
    WaitResponse = 1,
}

#[cfg(feature = "python")]
#[pymethods]
impl Phase {
    fn __hash__(&self) -> i32 {
        *self as i32
    }
}

/// The type of action a player can take during a mahjong game.
#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", eq, eq_int)
)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    /// Discard a tile from hand.
    Discard = 0,
    /// Claim a sequence (chi) from the left player's discard.
    Chi = 1,
    /// Claim a triplet (pon) from any player's discard.
    Pon = 2,
    /// Claim an open quad (daiminkan) from any player's discard.
    Daiminkan = 3,
    /// Declare a win on another player's discard.
    Ron = 4,
    /// Declare riichi (ready hand) before discarding.
    Riichi = 5,
    /// Declare a win by self-draw.
    Tsumo = 6,
    /// Pass on a claim opportunity.
    Pass = 7,
    /// Declare a concealed quad (ankan) from four identical tiles in hand.
    Ankan = 8,
    /// Upgrade a pon to a quad (kakan) with the fourth tile.
    Kakan = 9,
    /// Abort the round with nine different terminals/honors on the first turn.
    KyushuKyuhai = 10,
    /// Declare kita (north tile set-aside) in three-player mode.
    Kita = 11,
}

#[cfg(feature = "python")]
#[pymethods]
impl ActionType {
    fn __hash__(&self) -> i32 {
        *self as i32
    }
}

/// A player action in a mahjong game (discard, meld, win declaration, etc.).
#[cfg_attr(feature = "python", pyclass(module = "riichienv._riichienv"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Action {
    /// The type of action being performed.
    pub action_type: ActionType,
    /// The primary tile ID (136-format) involved in this action, if any.
    pub tile: Option<u8>,
    /// Tiles consumed from hand for this action (sorted, padded with zeros).
    pub consume_tiles: [u8; 4],
    /// The number of active entries in `consume_tiles`.
    pub consume_count: u8,
    /// The player index who performed this action, if known.
    pub actor: Option<u8>,
}

impl Default for Action {
    fn default() -> Self {
        Self {
            action_type: ActionType::Pass,
            tile: None,
            consume_tiles: [0; 4],
            consume_count: 0,
            actor: None,
        }
    }
}

impl Serialize for Action {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Action", 4)?;
        s.serialize_field("action_type", &self.action_type)?;
        s.serialize_field("tile", &self.tile)?;
        s.serialize_field("consume_tiles", self.consume_slice())?;
        s.serialize_field("actor", &self.actor)?;
        s.end()
    }
}

impl<'de> Deserialize<'de> for Action {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct ActionHelper {
            action_type: ActionType,
            tile: Option<u8>,
            consume_tiles: Vec<u8>,
            actor: Option<u8>,
        }
        let h = ActionHelper::deserialize(deserializer)?;
        Ok(Action::new(
            h.action_type,
            h.tile,
            &h.consume_tiles,
            h.actor,
        ))
    }
}

impl Action {
    /// Creates a new action with the given type, tile, consumed tiles, and actor.
    pub fn new(r#type: ActionType, tile: Option<u8>, consume: &[u8], actor: Option<u8>) -> Self {
        let count = consume.len().min(4);
        let mut tiles = [0u8; 4];
        tiles[..count].copy_from_slice(&consume[..count]);
        // Sort the active portion
        tiles[..count].sort();
        Self {
            action_type: r#type,
            tile,
            consume_tiles: tiles,
            consume_count: count as u8,
            actor,
        }
    }

    /// Get the active consume tiles as a slice.
    pub fn consume_slice(&self) -> &[u8] {
        &self.consume_tiles[..self.consume_count as usize]
    }

    /// Converts this action to an MJAI protocol JSON string.
    pub fn to_mjai(&self) -> String {
        let type_str = match self.action_type {
            ActionType::Discard => "dahai",
            ActionType::Chi => "chi",
            ActionType::Pon => "pon",
            ActionType::Daiminkan => "daiminkan",
            ActionType::Ankan => "ankan",
            ActionType::Kakan => "kakan",
            ActionType::Riichi => "reach",
            ActionType::Tsumo | ActionType::Ron => "hora",
            ActionType::KyushuKyuhai => "ryukyoku",
            ActionType::Kita => "kita",
            ActionType::Pass => "none",
        };

        let mut data = serde_json::Map::new();
        data.insert("type".to_string(), Value::String(type_str.to_string()));

        if let Some(actor) = self.actor {
            data.insert(
                "actor".to_string(),
                Value::Number(serde_json::Number::from(actor)),
            );
        }

        if let Some(t) = self.tile {
            if self.action_type != ActionType::Tsumo
                && self.action_type != ActionType::Ron
                && self.action_type != ActionType::Riichi
            {
                data.insert("pai".to_string(), Value::String(tid_to_mjai(t)));
            }
        }

        if self.consume_count > 0 {
            let cons: Vec<String> = self
                .consume_slice()
                .iter()
                .map(|&t| tid_to_mjai(t))
                .collect();
            // SAFETY: serialization of Vec<String> never fails
            data.insert("consumed".to_string(), serde_json::to_value(cons).unwrap());
        }

        Value::Object(data).to_string()
    }

    /// Returns a debug-friendly string representation of this action.
    pub fn repr(&self) -> String {
        format!(
            "Action(action_type={:?}, tile={:?}, consume_tiles={:?}, actor={:?})",
            self.action_type,
            self.tile,
            self.consume_slice(),
            self.actor
        )
    }

    /// Encodes this action as a 4-player action space index (0-81).
    pub fn encode(&self) -> RiichiResult<i32> {
        match self.action_type {
            ActionType::Discard => {
                if let Some(tile) = self.tile {
                    Ok((tile as i32) / 4)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Discard action requires a tile".to_string(),
                    })
                }
            }
            ActionType::Riichi => Ok(37),
            ActionType::Chi => {
                if let Some(target) = self.tile {
                    let target_34 = (target as i32) / 4;
                    let mut tiles_34: Vec<i32> = self
                        .consume_slice()
                        .iter()
                        .map(|&x| (x as i32) / 4)
                        .collect();
                    tiles_34.push(target_34);
                    tiles_34.sort();
                    tiles_34.dedup();

                    if tiles_34.len() != 3 {
                        return Err(RiichiError::InvalidAction {
                            message: format!(
                                "Invalid Chi tiles: target={}, consumed={:?}",
                                target,
                                self.consume_slice()
                            ),
                        });
                    }

                    if target_34 == tiles_34[0] {
                        Ok(38) // Low
                    } else if target_34 == tiles_34[1] {
                        Ok(39) // Mid
                    } else {
                        Ok(40) // High
                    }
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Chi action requires a target tile".to_string(),
                    })
                }
            }
            ActionType::Pon => Ok(41),
            ActionType::Daiminkan => {
                if let Some(tile) = self.tile {
                    Ok(42 + (tile as i32) / 4)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Daiminkan action requires a tile".to_string(),
                    })
                }
            }
            ActionType::Ankan | ActionType::Kakan => {
                if self.consume_count > 0 {
                    Ok(42 + (self.consume_tiles[0] as i32) / 4)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Ankan/Kakan action requires consumed tiles".to_string(),
                    })
                }
            }
            ActionType::Ron | ActionType::Tsumo => Ok(79),
            ActionType::KyushuKyuhai => Ok(80),
            ActionType::Pass => Ok(81),
            ActionType::Kita => Err(RiichiError::InvalidAction {
                message: "Kita action is not valid in 4-player mode".to_string(),
            }),
        }
    }
}

/// Separate encoder objects for 4P (default) and 3P action spaces.
///
/// `Action::encode()` always returns the 4P encoding (82 IDs, 0-81).
/// For 3P compact encoding (60 IDs), use `ActionEncoder::ThreePlayer`.
#[derive(Debug, Clone, Copy)]
pub enum ActionEncoder {
    /// Encoder for the standard 4-player action space (82 IDs).
    FourPlayer,
    /// Encoder for the 3-player (sanma) compact action space (60 IDs).
    ThreePlayer,
}

impl ActionEncoder {
    /// Creates an encoder appropriate for the given player count.
    pub fn from_num_players(n: u8) -> Self {
        match n {
            3 => Self::ThreePlayer,
            _ => Self::FourPlayer,
        }
    }

    /// Returns the total number of actions in this encoder's action space.
    pub fn action_space_size(&self) -> usize {
        match self {
            Self::FourPlayer => ACTION_SPACE_4P,
            Self::ThreePlayer => ACTION_SPACE_3P,
        }
    }

    /// Encodes an action into this encoder's action space index.
    pub fn encode(&self, action: &Action) -> RiichiResult<i32> {
        match self {
            Self::FourPlayer => action.encode(),
            Self::ThreePlayer => Self::encode_3p(action),
        }
    }

    fn encode_3p(action: &Action) -> RiichiResult<i32> {
        match action.action_type {
            ActionType::Discard => {
                if let Some(tile) = action.tile {
                    let tile_type = (tile / 4) as usize;
                    if tile_type >= 34 {
                        return Err(RiichiError::InvalidAction {
                            message: format!("Invalid tile type {} for 3P encode", tile_type),
                        });
                    }
                    let compact = TILE34_TO_COMPACT[tile_type];
                    if compact == 255 {
                        return Err(RiichiError::InvalidAction {
                            message: format!(
                                "Tile type {} (manzu 2-8) is not valid in 3P mode",
                                tile_type
                            ),
                        });
                    }
                    Ok(compact as i32)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Discard action requires a tile".to_string(),
                    })
                }
            }
            ActionType::Riichi => Ok(27),
            ActionType::Chi => Err(RiichiError::InvalidAction {
                message: "Chi is not allowed in 3P mode".to_string(),
            }),
            ActionType::Pon => Ok(28),
            ActionType::Daiminkan => {
                if let Some(tile) = action.tile {
                    let tile_type = (tile / 4) as usize;
                    if tile_type >= 34 {
                        return Err(RiichiError::InvalidAction {
                            message: format!("Invalid tile type {} for 3P encode", tile_type),
                        });
                    }
                    let compact = TILE34_TO_COMPACT[tile_type];
                    if compact == 255 {
                        return Err(RiichiError::InvalidAction {
                            message: format!(
                                "Tile type {} (manzu 2-8) is not valid in 3P mode",
                                tile_type
                            ),
                        });
                    }
                    Ok(29 + compact as i32)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Daiminkan action requires a tile".to_string(),
                    })
                }
            }
            ActionType::Ankan | ActionType::Kakan => {
                if action.consume_count > 0 {
                    let tile_type = (action.consume_tiles[0] / 4) as usize;
                    if tile_type >= 34 {
                        return Err(RiichiError::InvalidAction {
                            message: format!("Invalid tile type {} for 3P encode", tile_type),
                        });
                    }
                    let compact = TILE34_TO_COMPACT[tile_type];
                    if compact == 255 {
                        return Err(RiichiError::InvalidAction {
                            message: format!(
                                "Tile type {} (manzu 2-8) is not valid in 3P mode",
                                tile_type
                            ),
                        });
                    }
                    Ok(29 + compact as i32)
                } else {
                    Err(RiichiError::InvalidAction {
                        message: "Ankan/Kakan action requires consumed tiles".to_string(),
                    })
                }
            }
            ActionType::Ron | ActionType::Tsumo => Ok(56),
            ActionType::KyushuKyuhai => Ok(57),
            ActionType::Pass => Ok(58),
            ActionType::Kita => Ok(59),
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Action {
    #[new]
    #[pyo3(signature = (r#type=ActionType::Pass, tile=None, consume_tiles=vec![], actor=None))]
    pub fn py_new(
        r#type: ActionType,
        tile: Option<u8>,
        consume_tiles: Vec<u8>,
        actor: Option<u8>,
    ) -> Self {
        Self::new(r#type, tile, &consume_tiles, actor)
    }

    #[pyo3(name = "to_dict")]
    pub fn to_dict_py<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("type", self.action_type as i32)?;
        dict.set_item("tile", self.tile)?;

        let cons: Vec<u32> = self.consume_slice().iter().map(|&x| x as u32).collect();
        dict.set_item("consume_tiles", cons)?;
        dict.set_item("actor", self.actor)?;
        Ok(dict.unbind().into())
    }

    #[pyo3(name = "to_mjai")]
    pub fn to_mjai_py(&self) -> PyResult<String> {
        Ok(self.to_mjai())
    }

    fn __repr__(&self) -> String {
        self.repr()
    }

    fn __str__(&self) -> String {
        self.repr()
    }

    #[getter]
    fn get_action_type(&self) -> ActionType {
        self.action_type
    }

    #[setter]
    fn set_action_type(&mut self, action_type: ActionType) {
        self.action_type = action_type;
    }

    #[getter]
    fn get_tile(&self) -> Option<u8> {
        self.tile
    }

    #[setter]
    fn set_tile(&mut self, tile: Option<u8>) {
        self.tile = tile;
    }

    #[getter]
    fn get_consume_tiles(&self) -> Vec<u32> {
        self.consume_slice().iter().map(|&x| x as u32).collect()
    }

    #[setter]
    fn set_consume_tiles(&mut self, value: Vec<u8>) {
        let count = value.len().min(4);
        self.consume_tiles = [0u8; 4];
        self.consume_tiles[..count].copy_from_slice(&value[..count]);
        self.consume_count = count as u8;
    }

    #[getter]
    fn get_actor(&self) -> Option<u8> {
        self.actor
    }

    #[setter]
    fn set_actor(&mut self, actor: Option<u8>) {
        self.actor = actor;
    }

    #[pyo3(name = "encode")]
    pub fn encode_py(&self) -> PyResult<i32> {
        self.encode().map_err(Into::into)
    }
}
