use hydra_belief_search::hand_ev::HandEvFeatures;
use hydra_safety::SafetyInfo;

use super::ObservationEncoder;
use super::layout::*;
use super::types::*;

// ---------------------------------------------------------------------------
// Incremental encoding: selective channel updates
// ---------------------------------------------------------------------------

/// Dirty flags indicating which channel groups need re-encoding.
///
/// Use with [`ObservationEncoder::encode_incremental`] to avoid
/// recomputing the full observation tensor when only a few groups changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct DirtyFlags(pub u16);

impl DirtyFlags {
    /// Closed hand channels (0-3) need re-encoding.
    pub const HAND: Self = Self(1 << 0); // Ch 0-3
    /// Open meld hand channels (4-7) need re-encoding.
    pub const OPEN_MELD: Self = Self(1 << 1); // Ch 4-7
    /// Drawn tile channel (8) needs re-encoding.
    pub const DRAWN: Self = Self(1 << 2); // Ch 8
    /// Shanten mask channels (9-10) need re-encoding.
    pub const SHANTEN: Self = Self(1 << 3); // Ch 9-10
    /// Discard channels (11-22) need re-encoding.
    pub const DISCARDS: Self = Self(1 << 4); // Ch 11-22
    /// Meld channels (23-34) need re-encoding.
    pub const MELDS: Self = Self(1 << 5); // Ch 23-34
    /// Dora channels (35-42) need re-encoding.
    pub const DORA: Self = Self(1 << 6); // Ch 35-42
    /// Game metadata channels (43-61) need re-encoding.
    pub const META: Self = Self(1 << 7); // Ch 43-61
    /// Safety channels (62-84) need re-encoding.
    pub const SAFETY: Self = Self(1 << 8); // Ch 62-84
    /// Search/belief Group C channels (85-149) need re-encoding.
    pub const SEARCH: Self = Self(1 << 9);
    /// Hand-EV Group D channels (150-191) need re-encoding.
    pub const HAND_EV: Self = Self(1 << 10);
    /// All channels need re-encoding.
    pub const ALL: Self = Self(0x7FF);

    /// After a draw: hand, drawn tile, shanten, metadata.
    pub const AFTER_DRAW: Self = Self(
        Self::HAND.0
            | Self::DRAWN.0
            | Self::SHANTEN.0
            | Self::META.0
            | Self::SEARCH.0
            | Self::HAND_EV.0,
    );
    /// After a discard: hand, drawn, shanten, discards, metadata, safety.
    pub const AFTER_DISCARD: Self = Self(
        Self::HAND.0
            | Self::DRAWN.0
            | Self::SHANTEN.0
            | Self::DISCARDS.0
            | Self::META.0
            | Self::SAFETY.0
            | Self::SEARCH.0
            | Self::HAND_EV.0,
    );
    /// After a call: hand, open melds, shanten, discards, melds, meta, safety.
    pub const AFTER_CALL: Self = Self(
        Self::HAND.0
            | Self::OPEN_MELD.0
            | Self::SHANTEN.0
            | Self::DISCARDS.0
            | Self::MELDS.0
            | Self::META.0
            | Self::SAFETY.0
            | Self::SEARCH.0
            | Self::HAND_EV.0,
    );
    /// New round: everything.
    pub const NEW_ROUND: Self = Self::ALL;

    #[inline]
    /// Check whether all flags in `other` are set in `self`.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
    #[inline]
    /// Return a new `DirtyFlags` with all bits from both `self` and `other`.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}
impl ObservationEncoder {
    /// Incrementally re-encode only the channel groups marked dirty.
    ///
    /// Unlike [`encode`], this does NOT clear the entire buffer. It only
    /// clears and rewrites the specific channel ranges that changed.
    /// Use [`DirtyFlags`] presets (`AFTER_DRAW`, `AFTER_DISCARD`,
    /// `AFTER_CALL`) for common game events.
    ///
    /// For a new round or first encode, use `DirtyFlags::ALL` or [`encode`].
    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    pub fn encode_incremental(
        &mut self,
        dirty: DirtyFlags,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
    ) -> &[f32; OBS_SIZE] {
        if dirty.contains(DirtyFlags::HAND) {
            self.clear_range(CH_HAND, CH_HAND + 4);
            self.encode_hand(hand);
        }
        if dirty.contains(DirtyFlags::OPEN_MELD) {
            self.clear_range(CH_OPEN_MELD, CH_OPEN_MELD + 4);
            self.encode_open_meld_hand(open_meld_counts);
        }
        if dirty.contains(DirtyFlags::DRAWN) {
            self.clear_range(CH_DRAWN, CH_DRAWN + 1);
            self.encode_drawn_tile(drawn_tile);
        }
        if dirty.contains(DirtyFlags::SHANTEN) {
            self.clear_range(CH_SHANTEN_MASK, CH_SHANTEN_MASK + 2);
            self.encode_shanten_masks(hand);
        }
        if dirty.contains(DirtyFlags::DISCARDS) {
            self.clear_range(CH_DISCARDS, CH_DISCARDS + 12);
            self.encode_discards(discards);
        }
        if dirty.contains(DirtyFlags::MELDS) {
            self.clear_range(CH_MELDS, CH_MELDS + 12);
            self.encode_melds(melds);
        }
        if dirty.contains(DirtyFlags::DORA) {
            self.clear_range(CH_DORA, CH_AKA + 3);
            self.encode_dora(dora);
            self.encode_aka(dora);
        }
        if dirty.contains(DirtyFlags::META) {
            self.clear_range(CH_META, CH_META + 19);
            self.encode_metadata(meta);
        }
        if dirty.contains(DirtyFlags::SAFETY) {
            self.clear_range(CH_SAFETY, CH_SAFETY + 23);
            self.encode_safety(safety);
        }
        self.as_slice()
    }

    /// Incrementally re-encode baseline plus optional Group C / Group D context.
    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    pub fn encode_incremental_with_context(
        &mut self,
        dirty: DirtyFlags,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
        search_features: Option<&SearchFeaturePlanes>,
        hand_ev: Option<&HandEvFeatures>,
    ) -> &[f32; OBS_SIZE] {
        self.encode_incremental(
            dirty,
            hand,
            drawn_tile,
            open_meld_counts,
            discards,
            melds,
            dora,
            meta,
            safety,
        );
        if dirty.contains(DirtyFlags::SEARCH) {
            self.clear_range(CH_SEARCH, CH_SEARCH + SEARCH_CONTEXT_CHANNELS);
            if let Some(features) = search_features {
                self.encode_search_features(features);
            }
        }
        if dirty.contains(DirtyFlags::HAND_EV) {
            self.clear_range(CH_HAND_EV, CH_HAND_EV + HAND_EV_CHANNELS);
            if let Some(features) = hand_ev {
                self.encode_hand_ev_features(features);
            }
        }
        self.as_slice()
    }
}
