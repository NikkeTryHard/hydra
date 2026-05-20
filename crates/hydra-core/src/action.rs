//! Hydra 46-action space mapping, Mortal-compatible.
//!
//! Maps between Hydra's compact 46-action representation and
//! riichienv-core's ActionType/Action structs.

use anyhow::{Result, bail};
use riichienv_core::action::{Action, ActionType};

use hydra_runtime_types::tile::{AKA_MANZU_136, AKA_PINZU_136, AKA_SOUZU_136};

pub use hydra_runtime_types::action::{
    AGARI, AKA_5M, AKA_5P, AKA_5S, CHI_LEFT, CHI_MID, CHI_RIGHT, DISCARD_END, DISCARD_START,
    HYDRA_ACTION_SPACE, HydraAction, KAN, PASS, PON, RIICHI, RYUUKYOKU,
};

/// Tracks two-phase composite actions (riichi tile select, kan tile select).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionPhase {
    /// Normal action selection from full 46-action space.
    Normal,
    /// Selecting which tile to discard for riichi (subset of 0-36).
    RiichiSelect,
    /// Selecting which tile for kan (subset of 0-36).
    KanSelect,
}

/// Find a 136-format tile of the given type in a hand.
fn find_tile_in_hand(hand: &[u8], tile_type: u8) -> Result<u8> {
    hand.iter()
        .find(|&&t| t / 4 == tile_type)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("tile type {} not in hand", tile_type))
}

fn chi_consume_type(called_type: u8, offset: i8) -> Result<u8> {
    if called_type >= 27 {
        bail!("chi requires suited tile, got tile type {called_type}");
    }
    let consume_type = i16::from(called_type) + i16::from(offset);
    if !(0..=26).contains(&consume_type) || consume_type / 9 != i16::from(called_type / 9) {
        bail!("chi variant out of suit range: called tile type {called_type}, offset {offset}");
    }
    Ok(consume_type as u8)
}

/// Context from the game state needed to resolve certain Hydra actions
/// into complete riichienv-core Actions.
#[derive(Debug, Clone)]
pub struct GameContext {
    /// The last discarded tile (136-format) -- needed for chi/pon calls
    pub last_discard: Option<u8>,
    /// Current game phase -- needed to distinguish tsumo vs ron
    pub phase: ActionPhase,
    /// Tiles in the acting player's hand (136-format) -- needed for chi consume_tiles
    pub hand: [u8; 14],
    /// Number of valid tiles currently in `hand`.
    pub hand_len: u8,
}

// ---------------------------------------------------------------------------
// Hydra -> riichienv conversion
// ---------------------------------------------------------------------------

/// Convert a HydraAction to a riichienv Action using game context.
///
/// For discard actions, converts tile type (0-33) to 136-format. For 5m/5p/5s
/// (types 4,13,22), uses copy 1 to avoid the aka slot. Aka discards (34-36)
/// use the known aka 136-indices directly.
///
/// Chi, kan, and agari actions use the `GameContext` to resolve the full
/// action details (consume tiles, kan type, tsumo vs ron).
pub fn hydra_to_riichienv(hydra: HydraAction, ctx: &GameContext) -> Result<Action> {
    let id = hydra.id();
    match id {
        // Normal discards: tile type 0-33 -> 136-format = type * 4 + copy
        // For 5m/5p/5s (types 4,13,22), copy 0 is the aka tile.
        // Use copy 1 for normal discards to avoid the collision.
        0..=33 => {
            let copy = if matches!(id, 4 | 13 | 22) { 1 } else { 0 };
            Ok(Action::new(
                ActionType::Discard,
                Some(id * 4 + copy),
                &[],
                None,
            ))
        }
        // Aka 5m discard
        34 => Ok(Action::new(
            ActionType::Discard,
            Some(AKA_MANZU_136),
            &[],
            None,
        )),
        // Aka 5p discard
        35 => Ok(Action::new(
            ActionType::Discard,
            Some(AKA_PINZU_136),
            &[],
            None,
        )),
        // Aka 5s discard
        36 => Ok(Action::new(
            ActionType::Discard,
            Some(AKA_SOUZU_136),
            &[],
            None,
        )),
        // Riichi declaration (tile selection is a separate phase)
        37 => Ok(Action::new(ActionType::Riichi, None, &[], None)),
        // Chi variants -- resolved using last_discard and hand from context
        38..=40 => {
            let called = ctx
                .last_discard
                .ok_or_else(|| anyhow::anyhow!("chi requires last_discard"))?;
            let called_type = called / 4;
            let (offset_a, offset_b) = match id {
                38 => (1i8, 2i8),  // left: called is lowest
                39 => (-1i8, 1i8), // mid: called is middle
                _ => (-2i8, -1i8), // right: called is highest
            };
            let type_a = chi_consume_type(called_type, offset_a)?;
            let type_b = chi_consume_type(called_type, offset_b)?;
            let tile_a = find_tile_in_hand(&ctx.hand[..ctx.hand_len as usize], type_a)?;
            let tile_b = find_tile_in_hand(&ctx.hand[..ctx.hand_len as usize], type_b)?;
            Ok(Action::new(
                ActionType::Chi,
                Some(called),
                &[tile_a, tile_b],
                None,
            ))
        }
        // Pon
        41 => Ok(Action::new(ActionType::Pon, None, &[], None)),
        // Kan -- resolved from game phase
        42 => {
            let action_type = match ctx.phase {
                ActionPhase::Normal => ActionType::Ankan,
                _ => ActionType::Daiminkan,
            };
            Ok(Action::new(action_type, None, &[], None))
        }
        // Agari -- tsumo during own turn, ron during response
        43 => {
            let action_type = match ctx.phase {
                ActionPhase::Normal => ActionType::Tsumo,
                _ => ActionType::Ron,
            };
            Ok(Action::new(action_type, None, &[], None))
        }
        // Kyushu kyuhai (abortive draw)
        44 => Ok(Action::new(ActionType::KyushuKyuhai, None, &[], None)),
        // Pass
        45 => Ok(Action::new(ActionType::Pass, None, &[], None)),
        _ => bail!("invalid HydraAction id: {id}"),
    }
}

// ---------------------------------------------------------------------------
// riichienv -> Hydra conversion
// ---------------------------------------------------------------------------

/// Convert a riichienv Action to a HydraAction.
///
/// Uses 136-format tile IDs from the Action to determine the correct Hydra
/// action index. Aka tiles (136-indices 16, 52, 88) map to Hydra 34-36.
pub fn riichienv_to_hydra(action: &Action) -> Result<HydraAction> {
    let id = match action.action_type {
        ActionType::Discard => {
            let tile = action
                .tile
                .ok_or_else(|| anyhow::anyhow!("Discard action missing tile"))?;
            // Check if this is an aka tile in 136-format
            match tile {
                AKA_MANZU_136 => AKA_5M,
                AKA_PINZU_136 => AKA_5P,
                AKA_SOUZU_136 => AKA_5S,
                _ => tile / 4, // 136-format -> 34-format tile type
            }
        }
        ActionType::Riichi => RIICHI,
        ActionType::Chi => {
            // Determine chi variant from called tile position among sorted tiles
            let target = action
                .tile
                .ok_or_else(|| anyhow::anyhow!("Chi action missing target tile"))?;
            let target_34 = target / 4;
            let slice = action.consume_slice();
            let mut tiles_34 = [0u8; 4];
            for (i, &t) in slice.iter().enumerate() {
                tiles_34[i] = t / 4;
            }
            let tile_count = slice.len();
            tiles_34[tile_count] = target_34;
            let total = tile_count + 1;
            let used = &mut tiles_34[..total];
            used.sort();
            if target_34 == used[0] {
                CHI_LEFT // called tile is lowest
            } else if target_34 == used[1] {
                CHI_MID // called tile is middle
            } else {
                CHI_RIGHT // called tile is highest
            }
        }
        ActionType::Pon => PON,
        ActionType::Daiminkan | ActionType::Ankan | ActionType::Kakan => KAN,
        ActionType::Ron | ActionType::Tsumo => AGARI,
        ActionType::KyushuKyuhai => RYUUKYOKU,
        ActionType::Pass => PASS,
        ActionType::Kita => bail!("Kita not supported in Hydra 4-player action space"),
    };
    HydraAction::new(id).ok_or_else(|| anyhow::anyhow!("computed invalid HydraAction id: {id}"))
}

// ---------------------------------------------------------------------------
// Legal action mask
// ---------------------------------------------------------------------------

/// Builds a boolean mask over the 46-action space from riichienv legal actions.
///
/// Each entry in the returned array is `true` if the corresponding Hydra
/// action index is legal. Actions that fail conversion (e.g. Kita) are skipped.
pub fn build_legal_mask(
    legal_actions: &[Action],
    phase: ActionPhase,
) -> [bool; HYDRA_ACTION_SPACE] {
    let mut mask = [false; HYDRA_ACTION_SPACE];
    for action in legal_actions {
        if let Ok(hydra) = riichienv_to_hydra(action) {
            let idx = hydra.id() as usize;
            if idx >= HYDRA_ACTION_SPACE {
                continue;
            }
            match phase {
                ActionPhase::Normal => {
                    mask[idx] = true;
                }
                ActionPhase::RiichiSelect | ActionPhase::KanSelect => {
                    if hydra.is_discard() {
                        mask[idx] = true;
                    }
                }
            }
        }
    }
    mask
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
