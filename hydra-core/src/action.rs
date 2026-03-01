//! Hydra 46-action space mapping, Mortal-compatible.
//!
//! Maps between Hydra's compact 46-action representation and
//! riichienv-core's ActionType/Action structs.

use anyhow::{bail, Result};
use riichienv_core::action::{Action, ActionType};

use crate::tile::{AKA_MANZU_136, AKA_PINZU_136, AKA_SOUZU_136};

/// Total number of distinct actions in Hydra's action space.
pub const HYDRA_ACTION_SPACE: usize = 46;

// Discard actions: 0-33 = base tile types, 34-36 = aka (red five) discards
pub const DISCARD_START: u8 = 0;
pub const DISCARD_END: u8 = 36;
pub const AKA_5M: u8 = 34;
pub const AKA_5P: u8 = 35;
pub const AKA_5S: u8 = 36;

// Non-discard actions
pub const RIICHI: u8 = 37;
pub const CHI_LEFT: u8 = 38;
pub const CHI_MID: u8 = 39;
pub const CHI_RIGHT: u8 = 40;
pub const PON: u8 = 41;
pub const KAN: u8 = 42;
pub const AGARI: u8 = 43;
pub const RYUUKYOKU: u8 = 44;
pub const PASS: u8 = 45;

/// A validated action in Hydra's 46-action space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HydraAction(u8);

impl HydraAction {
    /// Create from raw index. Returns None if out of range.
    pub fn new(id: u8) -> Option<Self> {
        if (id as usize) < HYDRA_ACTION_SPACE {
            Some(Self(id))
        } else {
            None
        }
    }

    pub fn id(self) -> u8 {
        self.0
    }

    pub fn is_discard(self) -> bool {
        self.0 <= DISCARD_END
    }

    pub fn is_aka_discard(self) -> bool {
        matches!(self.0, 34..=36)
    }

    /// For discard actions, returns the base tile type (0-33).
    /// Aka discards map back: 34->4(5m), 35->13(5p), 36->22(5s).
    pub fn discard_tile_type(self) -> Option<u8> {
        match self.0 {
            0..=33 => Some(self.0),
            34 => Some(4),  // aka 5m -> 5m
            35 => Some(13), // aka 5p -> 5p
            36 => Some(22), // aka 5s -> 5s
            _ => None,
        }
    }
}

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

// ---------------------------------------------------------------------------
// Hydra -> riichienv conversion
// ---------------------------------------------------------------------------

/// Convert a HydraAction to a riichienv Action.
///
/// For discard actions, converts tile type (0-33) to 136-format by multiplying
/// by 4 (picks copy 0). Aka discards (34-36) use the known aka 136-indices.
///
/// Some actions (chi variants, kan) need game context to fully resolve.
/// This function produces a best-effort Action that captures the action type.
pub fn hydra_to_riichienv(hydra: HydraAction) -> Result<Action> {
    let id = hydra.id();
    match id {
        // Normal discards: tile type 0-33 -> 136-format = type * 4
        0..=33 => Ok(Action::new(
            ActionType::Discard,
            Some(id * 4),
            vec![],
            None,
        )),
        // Aka 5m discard
        34 => Ok(Action::new(
            ActionType::Discard,
            Some(AKA_MANZU_136),
            vec![],
            None,
        )),
        // Aka 5p discard
        35 => Ok(Action::new(
            ActionType::Discard,
            Some(AKA_PINZU_136),
            vec![],
            None,
        )),
        // Aka 5s discard
        36 => Ok(Action::new(
            ActionType::Discard,
            Some(AKA_SOUZU_136),
            vec![],
            None,
        )),
        // Riichi declaration (tile selection is a separate phase)
        37 => Ok(Action::new(ActionType::Riichi, None, vec![], None)),
        // Chi variants -- consume_tiles need game context to resolve
        // TODO: caller must fill in tile + consume_tiles from game state
        38 => Ok(Action::new(ActionType::Chi, None, vec![], None)),
        39 => Ok(Action::new(ActionType::Chi, None, vec![], None)),
        40 => Ok(Action::new(ActionType::Chi, None, vec![], None)),
        // Pon
        41 => Ok(Action::new(ActionType::Pon, None, vec![], None)),
        // Kan -- defaults to Daiminkan; caller should resolve ankan/kakan from context
        42 => Ok(Action::new(ActionType::Daiminkan, None, vec![], None)),
        // Agari -- defaults to Tsumo; caller should resolve tsumo vs ron from phase
        43 => Ok(Action::new(ActionType::Tsumo, None, vec![], None)),
        // Kyushu kyuhai (abortive draw)
        44 => Ok(Action::new(ActionType::KyushuKyuhai, None, vec![], None)),
        // Pass
        45 => Ok(Action::new(ActionType::Pass, None, vec![], None)),
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
            let tile = action.tile.ok_or_else(|| {
                anyhow::anyhow!("Discard action missing tile")
            })?;
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
            let target = action.tile.ok_or_else(|| {
                anyhow::anyhow!("Chi action missing target tile")
            })?;
            let target_34 = target / 4;
            let mut tiles_34: Vec<u8> = action
                .consume_tiles
                .iter()
                .map(|&x| x / 4)
                .collect();
            tiles_34.push(target_34);
            tiles_34.sort();
            tiles_34.dedup();
            if target_34 == tiles_34[0] {
                CHI_LEFT  // called tile is lowest
            } else if target_34 == tiles_34[1] {
                CHI_MID   // called tile is middle
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
    HydraAction::new(id)
        .ok_or_else(|| anyhow::anyhow!("computed invalid HydraAction id: {id}"))
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
    _phase: ActionPhase,
) -> [bool; HYDRA_ACTION_SPACE] {
    let mut mask = [false; HYDRA_ACTION_SPACE];
    for action in legal_actions {
        if let Ok(hydra) = riichienv_to_hydra(action) {
            let idx = hydra.id() as usize;
            if idx < HYDRA_ACTION_SPACE {
                mask[idx] = true;
            }
        }
    }
    mask
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydra_action_valid_range() {
        for i in 0..46u8 {
            assert!(HydraAction::new(i).is_some(), "id {i} should be valid");
        }
        assert!(HydraAction::new(46).is_none());
        assert!(HydraAction::new(255).is_none());
    }

    #[test]
    fn discard_tile_type_normal() {
        for i in 0..34u8 {
            let a = HydraAction::new(i).unwrap();
            assert!(a.is_discard());
            assert!(!a.is_aka_discard());
            assert_eq!(a.discard_tile_type(), Some(i));
        }
    }

    #[test]
    fn discard_tile_type_aka() {
        let a34 = HydraAction::new(34).unwrap();
        assert!(a34.is_discard());
        assert!(a34.is_aka_discard());
        assert_eq!(a34.discard_tile_type(), Some(4)); // 5m

        let a35 = HydraAction::new(35).unwrap();
        assert_eq!(a35.discard_tile_type(), Some(13)); // 5p

        let a36 = HydraAction::new(36).unwrap();
        assert_eq!(a36.discard_tile_type(), Some(22)); // 5s
    }

    #[test]
    fn non_discard_has_no_tile_type() {
        for i in 37..46u8 {
            let a = HydraAction::new(i).unwrap();
            assert!(!a.is_discard());
            assert_eq!(a.discard_tile_type(), None);
        }
    }

    #[test]
    fn roundtrip_pass() {
        let pass = Action::new(ActionType::Pass, None, vec![], None);
        let hydra = riichienv_to_hydra(&pass).unwrap();
        assert_eq!(hydra.id(), PASS);
        let back = hydra_to_riichienv(hydra).unwrap();
        assert_eq!(back.action_type, ActionType::Pass);
    }

    #[test]
    fn roundtrip_agari() {
        // Tsumo -> AGARI -> Tsumo (default)
        let tsumo = Action::new(ActionType::Tsumo, None, vec![], None);
        let hydra = riichienv_to_hydra(&tsumo).unwrap();
        assert_eq!(hydra.id(), AGARI);

        // Ron also maps to AGARI
        let ron = Action::new(ActionType::Ron, None, vec![], None);
        let hydra_ron = riichienv_to_hydra(&ron).unwrap();
        assert_eq!(hydra_ron.id(), AGARI);
    }

    #[test]
    fn discard_normal_roundtrip() {
        // Discard 1m (type 0, 136-format = 0)
        let discard = Action::new(ActionType::Discard, Some(0), vec![], None);
        let hydra = riichienv_to_hydra(&discard).unwrap();
        assert_eq!(hydra.id(), 0);
    }

    #[test]
    fn discard_aka_roundtrip() {
        // Aka 5m: 136-index 16 -> Hydra 34
        let discard = Action::new(ActionType::Discard, Some(16), vec![], None);
        let hydra = riichienv_to_hydra(&discard).unwrap();
        assert_eq!(hydra.id(), AKA_5M);
        assert!(hydra.is_aka_discard());

        // Roundtrip back: Hydra 34 -> 136-index 16
        let back = hydra_to_riichienv(hydra).unwrap();
        assert_eq!(back.tile, Some(AKA_MANZU_136));
    }

    #[test]
    fn chi_variant_encoding() {
        // Called tile is lowest (left chi): e.g. call 3m, consume 4m+5m
        // 3m=type2, 4m=type3, 5m=type4 -> sorted [2,3,4], target=2 -> CHI_LEFT
        let chi = Action::new(
            ActionType::Chi,
            Some(2 * 4), // 3m in 136-format
            vec![3 * 4, 4 * 4],
            None,
        );
        let hydra = riichienv_to_hydra(&chi).unwrap();
        assert_eq!(hydra.id(), CHI_LEFT);

        // Called tile is middle
        let chi_mid = Action::new(
            ActionType::Chi,
            Some(3 * 4), // 4m
            vec![2 * 4, 4 * 4],
            None,
        );
        assert_eq!(riichienv_to_hydra(&chi_mid).unwrap().id(), CHI_MID);

        // Called tile is highest
        let chi_right = Action::new(
            ActionType::Chi,
            Some(4 * 4), // 5m
            vec![2 * 4, 3 * 4],
            None,
        );
        assert_eq!(riichienv_to_hydra(&chi_right).unwrap().id(), CHI_RIGHT);
    }

    #[test]
    fn kan_variants_all_map_to_kan() {
        let daiminkan = Action::new(ActionType::Daiminkan, Some(0), vec![], None);
        assert_eq!(riichienv_to_hydra(&daiminkan).unwrap().id(), KAN);

        let ankan = Action::new(ActionType::Ankan, None, vec![0, 1, 2, 3], None);
        assert_eq!(riichienv_to_hydra(&ankan).unwrap().id(), KAN);

        let kakan = Action::new(ActionType::Kakan, None, vec![0], None);
        assert_eq!(riichienv_to_hydra(&kakan).unwrap().id(), KAN);
    }

    #[test]
    fn legal_mask_basic() {
        let actions = vec![
            Action::new(ActionType::Discard, Some(0), vec![], None),  // 1m -> idx 0
            Action::new(ActionType::Discard, Some(16), vec![], None), // aka 5m -> idx 34
            Action::new(ActionType::Pass, None, vec![], None),        // -> idx 45
        ];
        let mask = build_legal_mask(&actions, ActionPhase::Normal);
        assert!(mask[0]);
        assert!(mask[34]);
        assert!(mask[45]);
        // Everything else should be false
        assert!(!mask[1]);
        assert!(!mask[37]);
    }
}
