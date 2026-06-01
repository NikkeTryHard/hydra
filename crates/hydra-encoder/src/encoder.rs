//! Fixed-superset observation tensor encoder for neural network input.
//!
//! Encodes the full game state into a flat `[f32; NUM_CHANNELS * 34]` array
//! (row-major) that serves as input to the Hydra SE-ResNet model.
//!
//! The currently implemented baseline channels remain intact in the first 85
//! planes. Additional Group C / Group D planes provide a fixed-shape superset
//! for search/belief and Hand-EV context, with zero-filled planes plus
//! presence-mask channels when those dynamic features are unavailable.
//!
//! Channels are grouped:
//!
//! - 0..3:   closed hand (thresholded tile counts)
//! - 4..7:   open meld hand counts (thresholded)
//! - 8:      drawn tile one-hot
//! - 9..10:  shanten masks (keep / next)
//! - 11..22: discards per player (presence, tedashi, temporal)
//! - 23..34: melds per player (chi, pon, kan)
//! - 35..39: dora indicator thermometer
//! - 40..42: aka dora flags (per suit plane)
//! - 43..61: game metadata (riichi, scores, gaps, shanten, round, honba, kyotaku)
//! - 62..84: safety channels (genbutsu, suji, kabe, one-chance, tenpai)
//! - 85..149: Group C search/belief context + presence masks + reserved slots
//! - 150..191: Group D Hand-EV context + presence mask

mod baseline;
mod context;
mod dirty;
mod layout;
mod types;

pub use dirty::DirtyFlags;
pub use layout::{
    BASELINE_CHANNELS, HAND_EV_CHANNEL_START, HAND_EV_CHANNELS, HAND_EV_MASK_CHANNEL, NUM_CHANNELS,
    NUM_TILES, OBS_SIZE, SEARCH_BELIEF_CHANNEL_START, SEARCH_CHANNEL_START,
    SEARCH_CONTEXT_CHANNELS, SEARCH_DELTA_Q_CHANNEL, SEARCH_MASK_CHANNEL_START,
    SEARCH_MIXTURE_ENTROPY_CHANNEL, SEARCH_MIXTURE_ESS_CHANNEL, SEARCH_RISK_CHANNEL_START,
    SEARCH_STRESS_CHANNEL_START,
};
pub use types::{
    DiscardEntry, DoraInfo, GameMetadata, MeldInfo, MeldType, PlayerDiscards, PlayerMelds,
    SearchFeaturePlanes,
};

use hydra_safety::SafetyInfo;

use self::layout::*;

// ---------------------------------------------------------------------------
// ObservationEncoder
// ---------------------------------------------------------------------------

/// Pre-allocated encoder buffer for the fixed-superset observation tensor.
///
/// Reuse across turns to avoid per-turn allocation. Call [`clear`] then
/// the individual `encode_*` methods, or use [`encode`] as the one-shot
/// entry point.
#[derive(Clone)]
#[repr(C)]
pub struct ObservationEncoder {
    /// Flat buffer: `NUM_CHANNELS` channels x 34 tiles, row-major.
    buffer: [f32; OBS_SIZE],
}

impl ObservationEncoder {
    /// Create a new encoder with a zeroed buffer.
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: [0.0; OBS_SIZE],
        }
    }

    /// Zero the entire buffer.
    #[inline]
    pub fn clear(&mut self) {
        self.buffer.fill(0.0);
    }

    /// Zero only the channels in range `[start_ch, end_ch)` (exclusive end).
    #[inline]
    pub fn clear_range(&mut self, start_ch: usize, end_ch: usize) {
        let start = start_ch * NUM_TILES;
        let end = end_ch * NUM_TILES;
        self.buffer[start..end].fill(0.0);
    }

    /// Read-only view of the flat observation buffer.
    #[inline]
    pub fn as_slice(&self) -> &[f32; OBS_SIZE] {
        &self.buffer
    }

    /// Set a single cell: `buffer[channel * 34 + tile] = value`.
    #[inline]
    pub(super) fn set(&mut self, channel: usize, tile: usize, value: f32) {
        self.buffer[channel * NUM_TILES + tile] = value;
    }

    /// Fill an entire channel with a uniform value.
    #[inline]
    pub(super) fn fill_channel(&mut self, channel: usize, value: f32) {
        let start = channel * NUM_TILES;
        self.buffer[start..start + NUM_TILES].fill(value);
    }

    #[inline]
    pub(super) fn copy_channel(&mut self, channel: usize, values: &[f32; NUM_TILES]) {
        let start = channel * NUM_TILES;
        self.buffer[start..start + NUM_TILES].copy_from_slice(values);
    }
}

impl Default for ObservationEncoder {
    fn default() -> Self {
        Self::new()
    }
}
// ---------------------------------------------------------------------------
// Full encode entry point
// ---------------------------------------------------------------------------

impl ObservationEncoder {
    /// Encode a complete observation from explicit game state components.
    ///
    /// Clears the buffer, then calls each sub-encoder in order.
    /// Returns a reference to the filled observation buffer.
    #[allow(
        clippy::too_many_arguments,
        reason = "encoder API mirrors the fixed observation layout"
    )]
    pub fn encode(
        &mut self,
        hand: &[u8; NUM_TILES],
        drawn_tile: Option<u8>,
        open_meld_counts: &[u8; NUM_TILES],
        discards: &[PlayerDiscards; NUM_PLAYERS],
        melds: &[PlayerMelds; NUM_PLAYERS],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
    ) -> &[f32; OBS_SIZE] {
        self.clear();
        self.encode_baseline_prefix(
            hand,
            drawn_tile,
            open_meld_counts,
            discards,
            melds,
            dora,
            meta,
            safety,
        );
        self.as_slice()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
