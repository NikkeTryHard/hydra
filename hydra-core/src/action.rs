//! Hydra 46-action space mapping, Mortal-compatible.
//!
//! Maps between Hydra's compact 46-action representation and
//! riichienv-core's ActionType/Action structs.

use anyhow::{bail, Result};
use riichienv_core::action::{Action, ActionType};

use crate::tile::{AKA_MANZU_136, AKA_PINZU_136, AKA_SOUZU_136};

/// Total number of distinct actions in Hydra's action space.
pub const HYDRA_ACTION_SPACE: usize = 46;

/// Discard actions: 0-33 = base tile types, 34-36 = aka (red five) discards.
pub const DISCARD_START: u8 = 0;
pub const DISCARD_END: u8 = 36;
pub const AKA_5M: u8 = 34;
pub const AKA_5P: u8 = 35;
pub const AKA_5S: u8 = 36;

/// Non-discard actions.
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
    #[inline]
    pub const fn new(id: u8) -> Option<Self> {
        if (id as usize) < HYDRA_ACTION_SPACE {
            Some(Self(id))
        } else {
            None
        }
    }

    #[inline]
    pub const fn id(self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn is_discard(self) -> bool {
        self.0 <= DISCARD_END
    }

    #[inline]
    pub const fn is_aka_discard(self) -> bool {
        matches!(self.0, 34..=36)
    }

    /// For discard actions, returns the base tile type (0-33).
    /// Aka discards map back: 34->4(5m), 35->13(5p), 36->22(5s).
    #[inline]
    pub const fn discard_tile_type(self) -> Option<u8> {
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

/// Find a 136-format tile of the given type in a hand.
fn find_tile_in_hand(hand: &[u8], tile_type: u8) -> Result<u8> {
    hand.iter()
        .find(|&&t| t / 4 == tile_type)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("tile type {} not in hand", tile_type))
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
            let type_a = (called_type as i8 + offset_a) as u8;
            let type_b = (called_type as i8 + offset_b) as u8;
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
mod tests {
    use super::*;

    fn dummy_ctx() -> GameContext {
        GameContext {
            last_discard: Some(0),
            phase: ActionPhase::Normal,
            hand: [0u8; 14],
            hand_len: 0,
        }
    }

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
        let pass = Action::new(ActionType::Pass, None, &[], None);
        let hydra = riichienv_to_hydra(&pass).unwrap();
        assert_eq!(hydra.id(), PASS);
        let back = hydra_to_riichienv(hydra, &dummy_ctx()).unwrap();
        assert_eq!(back.action_type, ActionType::Pass);
    }

    #[test]
    fn roundtrip_agari() {
        // Tsumo -> AGARI -> Tsumo (default)
        let tsumo = Action::new(ActionType::Tsumo, None, &[], None);
        let hydra = riichienv_to_hydra(&tsumo).unwrap();
        assert_eq!(hydra.id(), AGARI);

        // Ron also maps to AGARI
        let ron = Action::new(ActionType::Ron, None, &[], None);
        let hydra_ron = riichienv_to_hydra(&ron).unwrap();
        assert_eq!(hydra_ron.id(), AGARI);
    }

    #[test]
    fn discard_normal_roundtrip() {
        // Discard 1m (type 0, 136-format = 0)
        let discard = Action::new(ActionType::Discard, Some(0), &[], None);
        let hydra = riichienv_to_hydra(&discard).unwrap();
        assert_eq!(hydra.id(), 0);
    }

    #[test]
    fn discard_aka_roundtrip() {
        // Aka 5m: 136-index 16 -> Hydra 34
        let discard = Action::new(ActionType::Discard, Some(16), &[], None);
        let hydra = riichienv_to_hydra(&discard).unwrap();
        assert_eq!(hydra.id(), AKA_5M);
        assert!(hydra.is_aka_discard());

        // Roundtrip back: Hydra 34 -> 136-index 16
        let back = hydra_to_riichienv(hydra, &dummy_ctx()).unwrap();
        assert_eq!(back.tile, Some(AKA_MANZU_136));
    }

    #[test]
    fn chi_variant_encoding() {
        // Called tile is lowest (left chi): e.g. call 3m, consume 4m+5m
        // 3m=type2, 4m=type3, 5m=type4 -> sorted [2,3,4], target=2 -> CHI_LEFT
        let chi = Action::new(
            ActionType::Chi,
            Some(2 * 4), // 3m in 136-format
            &[3 * 4, 4 * 4],
            None,
        );
        let hydra = riichienv_to_hydra(&chi).unwrap();
        assert_eq!(hydra.id(), CHI_LEFT);

        // Called tile is middle
        let chi_mid = Action::new(
            ActionType::Chi,
            Some(3 * 4), // 4m
            &[2 * 4, 4 * 4],
            None,
        );
        assert_eq!(riichienv_to_hydra(&chi_mid).unwrap().id(), CHI_MID);

        // Called tile is highest
        let chi_right = Action::new(
            ActionType::Chi,
            Some(4 * 4), // 5m
            &[2 * 4, 3 * 4],
            None,
        );
        assert_eq!(riichienv_to_hydra(&chi_right).unwrap().id(), CHI_RIGHT);
    }

    #[test]
    fn kan_variants_all_map_to_kan() {
        let daiminkan = Action::new(ActionType::Daiminkan, Some(0), &[], None);
        assert_eq!(riichienv_to_hydra(&daiminkan).unwrap().id(), KAN);

        let ankan = Action::new(ActionType::Ankan, None, &[0, 1, 2, 3], None);
        assert_eq!(riichienv_to_hydra(&ankan).unwrap().id(), KAN);

        let kakan = Action::new(ActionType::Kakan, None, &[0], None);
        assert_eq!(riichienv_to_hydra(&kakan).unwrap().id(), KAN);
    }

    #[test]
    fn legal_mask_basic() {
        let actions = vec![
            Action::new(ActionType::Discard, Some(0), &[], None), // 1m -> idx 0
            Action::new(ActionType::Discard, Some(16), &[], None), // aka 5m -> idx 34
            Action::new(ActionType::Pass, None, &[], None),       // -> idx 45
        ];
        let mask = build_legal_mask(&actions, ActionPhase::Normal);
        assert!(mask[0]);
        assert!(mask[34]);
        assert!(mask[45]);
        // Everything else should be false
        assert!(!mask[1]);
        assert!(!mask[37]);
    }

    #[test]
    fn discard_5m_is_not_aka() {
        // Normal 5m (Hydra id=4) must NOT map to the aka 136-index (16)
        let hydra = HydraAction::new(4).unwrap();
        let action = hydra_to_riichienv(hydra, &dummy_ctx()).unwrap();
        let tile136 = action.tile.unwrap();
        assert_ne!(tile136, 16, "normal 5m must not use aka 136-index");
        assert_eq!(tile136 / 4, 4, "must still be tile type 5m");
    }

    #[test]
    fn discard_aka_5m_is_aka() {
        // Aka 5m (Hydra id=34) MUST map to aka 136-index (16)
        let hydra = HydraAction::new(34).unwrap();
        let action = hydra_to_riichienv(hydra, &dummy_ctx()).unwrap();
        assert_eq!(action.tile.unwrap(), 16);
    }

    #[test]
    fn all_five_tiles_avoid_aka_collision() {
        // Check 5m, 5p, 5s normal discards
        for (hydra_id, aka_136) in [(4u8, 16u8), (13, 52), (22, 88)] {
            let hydra = HydraAction::new(hydra_id).unwrap();
            let action = hydra_to_riichienv(hydra, &dummy_ctx()).unwrap();
            let tile136 = action.tile.unwrap();
            assert_ne!(
                tile136, aka_136,
                "normal discard of type {} must not use aka 136-index {}",
                hydra_id, aka_136,
            );
            assert_eq!(
                tile136 / 4,
                hydra_id,
                "tile type must still be {}",
                hydra_id,
            );
        }
    }

    #[test]
    fn legal_mask_riichi_select_filters_non_discards() {
        let actions = vec![
            Action::new(ActionType::Discard, Some(0), &[], None),
            Action::new(ActionType::Discard, Some(16), &[], None),
            Action::new(ActionType::Pass, None, &[], None),
            Action::new(ActionType::Tsumo, None, &[], None),
        ];
        let mask = build_legal_mask(&actions, ActionPhase::RiichiSelect);
        assert!(mask[0]); // discard 1m allowed
        assert!(mask[34]); // discard aka 5m allowed
        assert!(!mask[45]); // pass NOT allowed in riichi select
        assert!(!mask[43]); // agari NOT allowed in riichi select
    }

    #[test]
    fn chi_left_resolves_consume_tiles() {
        // Chi left (38): called tile is lowest (1m=type0), need type1 + type2 from hand
        // Discard: tile 0 (type 0, 1m). Hand has tiles 4 (type 1, 2m) and 8 (type 2, 3m)
        let mut hand = [0u8; 14];
        hand[..4].copy_from_slice(&[4, 8, 20, 24]);
        let ctx = GameContext {
            last_discard: Some(0),
            phase: ActionPhase::Normal,
            hand,
            hand_len: 4,
        };
        let hydra = HydraAction::new(CHI_LEFT).unwrap();
        let action = hydra_to_riichienv(hydra, &ctx).unwrap();
        assert_eq!(action.action_type, ActionType::Chi);
        assert_eq!(action.tile, Some(0));
        assert_eq!(action.consume_slice(), &[4, 8]);
    }

    #[test]
    fn agari_resolves_to_tsumo_in_normal_phase() {
        let ctx = GameContext {
            last_discard: None,
            phase: ActionPhase::Normal,
            hand: [0u8; 14],
            hand_len: 0,
        };
        let hydra = HydraAction::new(AGARI).unwrap();
        let action = hydra_to_riichienv(hydra, &ctx).unwrap();
        assert_eq!(action.action_type, ActionType::Tsumo);
    }

    #[test]
    fn agari_resolves_to_ron_in_response_phase() {
        let ctx = GameContext {
            last_discard: Some(0),
            phase: ActionPhase::RiichiSelect, // non-Normal = response
            hand: [0u8; 14],
            hand_len: 0,
        };
        let hydra = HydraAction::new(AGARI).unwrap();
        let action = hydra_to_riichienv(hydra, &ctx).unwrap();
        assert_eq!(action.action_type, ActionType::Ron);
    }
}
