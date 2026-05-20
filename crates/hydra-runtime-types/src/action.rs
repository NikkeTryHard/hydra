//! Pure Hydra action-space runtime types.

use crate::tile::{FIVE_MANZU, FIVE_PINZU, FIVE_SOUZU};

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
        matches!(self.0, AKA_5M | AKA_5P | AKA_5S)
    }

    /// For discard actions, returns the base tile type (0-33).
    /// Aka discards map back to their base five tile types.
    #[inline]
    pub const fn discard_tile_type(self) -> Option<u8> {
        match self.0 {
            DISCARD_START..=33 => Some(self.0),
            AKA_5M => Some(FIVE_MANZU),
            AKA_5P => Some(FIVE_PINZU),
            AKA_5S => Some(FIVE_SOUZU),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
