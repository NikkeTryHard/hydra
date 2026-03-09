//! Zero-copy observation view into GameState.
//!
//! [`ObservationRef`] borrows directly from [`GameState`] fields,
//! avoiding the ~15 Vec allocations that [`Observation::new`] performs.

use crate::types::Meld;

/// Zero-copy view into GameState for a specific player's perspective.
///
/// Every `Vec` in the original [`Observation`] becomes a borrowed slice.
/// Copy types (`u8`, `i32`, `bool`) are copied directly.
pub struct ObservationRef<'a> {
    /// The player whose perspective this observation represents.
    pub player_id: u8,
    /// Observer's hand in 136-format (sorted). Opponents' hands are hidden.
    pub observer_hand: &'a [u8],
    /// Open melds for each player (absolute index, not relative).
    pub melds: [&'a [Meld]; 4],
    /// Discard piles for each player in 136-format.
    pub discards: [&'a [u8]; 4],
    /// Dora indicator tiles in 136-format.
    pub dora_indicators: &'a [u8],
    /// Current scores for each player.
    pub scores: [i32; 4],
    /// Whether each player has declared riichi.
    pub riichi_declared: [bool; 4],
    /// Honba counter.
    pub honba: u8,
    /// Riichi sticks on the table.
    pub riichi_sticks: u32,
    /// Round wind (0=East, 1=South, 2=West, 3=North).
    pub round_wind: u8,
    /// Dealer seat index.
    pub oya: u8,
    /// Kyoku index within the current wind.
    pub kyoku_index: u8,
    /// Seat of the player who is currently acting.
    pub current_player: u8,
    /// The tile just drawn (136-format), if any.
    pub drawn_tile: Option<u8>,
    /// Whether the game has ended.
    pub is_done: bool,
}
