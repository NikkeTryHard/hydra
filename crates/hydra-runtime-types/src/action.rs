//! Pure Hydra action-space runtime types.

/// Total number of distinct actions in Hydra's action space.
pub const HYDRA_ACTION_SPACE: usize = 46;

/// Discard actions: 0-33 = base tile types, 34-36 = aka (red five) discards.
pub const DISCARD_START: u8 = 0;
/// End of the discard action range (inclusive).
pub const DISCARD_END: u8 = 36;
/// Discard action for red five of manzu (aka 5m).
pub const AKA_5M: u8 = 34;
/// Discard action for red five of pinzu (aka 5p).
pub const AKA_5P: u8 = 35;
/// Discard action for red five of souzu (aka 5s).
pub const AKA_5S: u8 = 36;

/// Declare riichi.
pub const RIICHI: u8 = 37;
/// Declare chi consuming the left tile of a sequence.
pub const CHI_LEFT: u8 = 38;
/// Declare chi consuming the middle tile of a sequence.
pub const CHI_MID: u8 = 39;
/// Declare chi consuming the right tile of a sequence.
pub const CHI_RIGHT: u8 = 40;
/// Declare pon (triplet call).
pub const PON: u8 = 41;
/// Declare kan (quad call or extension).
pub const KAN: u8 = 42;
/// Declare agari (tsumo or ron win).
pub const AGARI: u8 = 43;
/// Declare ryuukyoku (abortive draw).
pub const RYUUKYOKU: u8 = 44;
/// Pass on an optional action (skip chi/pon/kan/ron).
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

    /// Return the raw action index (0-45).
    #[inline]
    pub const fn id(self) -> u8 {
        self.0
    }

    /// Check whether this action is a discard (index 0-36).
    #[inline]
    pub const fn is_discard(self) -> bool {
        self.0 <= DISCARD_END
    }

    /// Check whether this action is an aka (red five) discard.
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
            34 => Some(4),
            35 => Some(13),
            36 => Some(22),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_space_and_key_indices_are_frozen() {
        assert_eq!(HYDRA_ACTION_SPACE, 46);
        assert_eq!(DISCARD_START, 0);
        assert_eq!(DISCARD_END, 36);
        assert_eq!(AKA_5M, 34);
        assert_eq!(AKA_5P, 35);
        assert_eq!(AKA_5S, 36);
        assert_eq!(RIICHI, 37);
        assert_eq!(CHI_LEFT, 38);
        assert_eq!(CHI_MID, 39);
        assert_eq!(CHI_RIGHT, 40);
        assert_eq!(PON, 41);
        assert_eq!(KAN, 42);
        assert_eq!(AGARI, 43);
        assert_eq!(RYUUKYOKU, 44);
        assert_eq!(PASS, 45);
    }

    #[test]
    fn action_validation_and_discard_mapping_match_abi() {
        for id in 0..HYDRA_ACTION_SPACE as u8 {
            let action = HydraAction::new(id).expect("action id should be valid");
            assert_eq!(action.id(), id);
            assert_eq!(action.is_discard(), id <= DISCARD_END);
            assert_eq!(
                action.is_aka_discard(),
                matches!(id, AKA_5M | AKA_5P | AKA_5S)
            );
        }

        assert!(HydraAction::new(HYDRA_ACTION_SPACE as u8).is_none());
        assert_eq!(HydraAction::new(0).unwrap().discard_tile_type(), Some(0));
        assert_eq!(HydraAction::new(33).unwrap().discard_tile_type(), Some(33));
        assert_eq!(
            HydraAction::new(AKA_5M).unwrap().discard_tile_type(),
            Some(4)
        );
        assert_eq!(
            HydraAction::new(AKA_5P).unwrap().discard_tile_type(),
            Some(13)
        );
        assert_eq!(
            HydraAction::new(AKA_5S).unwrap().discard_tile_type(),
            Some(22)
        );
        assert_eq!(HydraAction::new(RIICHI).unwrap().discard_tile_type(), None);
        assert_eq!(HydraAction::new(PASS).unwrap().discard_tile_type(), None);
    }
}
