//! Bridge between riichienv-core game state and Hydra's observation encoder.
//!
//! Converts riichienv [`Observation`] data into the encoder's input types,
//! then runs the full 85x34 encoding pipeline. This is the critical glue
//! between the game engine and the neural network.

use riichienv_core::observation::Observation;
use riichienv_core::shanten::calc_shanten_from_counts;
use riichienv_core::types::MeldType as RiichiMeldType;
use riichienv_core::observation_ref::ObservationRef;

use crate::encoder::{
    DiscardEntry, DoraInfo, GameMetadata, MeldInfo, MeldType, ObservationEncoder, PlayerDiscards,
    OBS_SIZE,
};
use crate::safety::SafetyInfo;
use crate::tile::NUM_TILE_TYPES;

// -- Aka dora tile IDs in 136-format --
const AKA_5M: u32 = 16;
const AKA_5P: u32 = 52;
const AKA_5S: u32 = 88;

/// Convert a 136-format tile ID (u32) to its 34-format tile type (u8).
#[inline]
fn tile136_to_type(tile136: u32) -> u8 {
    (tile136 / 4) as u8
}

/// Extract hand tile counts from an Observation.
///
/// Only the observer's own hand is meaningful (opponents' hands are hidden).
/// Converts from 136-format `Vec<u32>` to 34-bin histogram `[u8; 34]`.
#[inline]
pub fn extract_hand(obs: &Observation) -> [u8; NUM_TILE_TYPES] {
    let observer = obs.player_id as usize;
    let mut counts = [0u8; NUM_TILE_TYPES];
    for &tile136 in &obs.hands[observer] {
        let t = tile136_to_type(tile136) as usize;
        if t < NUM_TILE_TYPES {
            counts[t] = counts[t].saturating_add(1);
        }
    }
    counts
}

/// Extract discard info for all 4 players from an Observation.
///
/// Player indices are RELATIVE to the observer (index 0 = observer).
/// Uses `tsumogiri_flags` to determine tedashi (tedashi = !tsumogiri).
#[inline]
pub fn extract_discards(obs: &Observation) -> [PlayerDiscards; 4] {
    let observer = obs.player_id as usize;
    std::array::from_fn(|relative_idx| {
        let abs = (observer + relative_idx) % 4;
        let disc = &obs.discards[abs];
        let tsumogiri = &obs.tsumogiri_flags[abs];
        let entries: Vec<DiscardEntry> = disc
            .iter()
            .enumerate()
            .map(|(turn, &tile136)| {
                let is_tsumogiri = tsumogiri.get(turn).copied().unwrap_or(false);
                DiscardEntry {
                    tile: tile136_to_type(tile136),
                    is_tedashi: !is_tsumogiri,
                    turn: turn as u16,
                }
            })
            .collect();
        PlayerDiscards { discards: entries }
    })
}

/// Extract meld info for all 4 players from an Observation.
///
/// Maps riichienv `MeldType` variants to the encoder's three-category system:
/// - Chi -> `MeldType::Chi`
/// - Pon -> `MeldType::Pon`
/// - Daiminkan/Ankan/Kakan -> `MeldType::Kan` (all kan variants merged)
///
/// Meld tile IDs are converted from 136-format (u8) to 34-format tile types.
#[inline]
pub fn extract_melds(obs: &Observation) -> [Vec<MeldInfo>; 4] {
    let observer = obs.player_id as usize;
    std::array::from_fn(|relative_idx| {
        let abs = (observer + relative_idx) % 4;
        obs.melds[abs]
            .iter()
            .map(|meld| {
                let tiles: Vec<u8> = meld.tiles.iter().map(|&t| t / 4).collect();
                let meld_type = match meld.meld_type {
                    RiichiMeldType::Chi => MeldType::Chi,
                    RiichiMeldType::Pon => MeldType::Pon,
                    RiichiMeldType::Daiminkan
                    | RiichiMeldType::Ankan
                    | RiichiMeldType::Kakan => MeldType::Kan,
                };
                MeldInfo { tiles, meld_type }
            })
            .collect()
    })
}

/// Count tile types across the observer's melds for channel 4-7 encoding.
///
/// Returns a 34-element histogram where each entry is the number of tiles
/// of that type present in the observer's open/called melds.
#[inline]
pub fn extract_observer_meld_counts(obs: &Observation) -> [u8; NUM_TILE_TYPES] {
    let observer = obs.player_id as usize;
    let mut counts = [0u8; NUM_TILE_TYPES];
    for meld in &obs.melds[observer] {
        for &tile in &meld.tiles {
            let t = (tile / 4) as usize;
            if t < NUM_TILE_TYPES {
                counts[t] = counts[t].saturating_add(1);
            }
        }
    }
    counts
}

/// Extract dora information from an Observation.
///
/// Converts dora indicator tile IDs from 136-format to 34-format tile types.
/// Scans the observer's hand for aka dora (red fives) at 136-format
/// indices 16 (5m), 52 (5p), 88 (5s).
#[inline]
pub fn extract_dora(obs: &Observation) -> DoraInfo {
    let indicators: Vec<u8> = obs
        .dora_indicators
        .iter()
        .map(|&t| tile136_to_type(t))
        .collect();

    // Scan observer's hand for aka dora tiles
    let observer = obs.player_id as usize;
    let hand = &obs.hands[observer];
    let aka_flags = [
        hand.contains(&AKA_5M),
        hand.contains(&AKA_5P),
        hand.contains(&AKA_5S),
    ];

    DoraInfo {
        indicators,
        aka_flags,
    }
}

/// Extract game metadata from an Observation.
///
/// Computes shanten from the observer's hand counts. All player-relative
/// fields (riichi, scores) are rotated so index 0 = observer,
/// index 1 = shimocha, etc.
#[inline]
pub fn extract_metadata(obs: &Observation, hand_counts: &[u8; NUM_TILE_TYPES]) -> GameMetadata {
    let observer = obs.player_id as usize;

    // Compute shanten: len_div3 is based on the closed hand tile count.
    // A 13-tile hand has len_div3=4, a 14-tile hand also has len_div3=4.
    let hand_total: u8 = hand_counts.iter().sum();
    let len_div3 = hand_total / 3;
    let shanten = calc_shanten_from_counts(hand_counts, len_div3);

    GameMetadata {
        riichi: std::array::from_fn(|i| {
            obs.riichi_declared[(observer + i) % 4]
        }),
        scores: std::array::from_fn(|i| {
            obs.scores[(observer + i) % 4]
        }),
        shanten,
        kyoku_index: obs.kyoku_index,
        honba: obs.honba,
        kyotaku: obs.riichi_sticks.min(255) as u8,
    }
}

/// Encode a full observation into the 85x34 tensor.
///
/// This is the main bridge entry point. Extracts all components from
/// a riichienv [`Observation`], feeds them through the encoder pipeline,
/// and returns a reference to the filled `[f32; 2890]` buffer.
///
/// # Drawn tile limitation
///
/// The drawn tile cannot be reliably determined from `Observation` alone.
/// Encode a full observation into the 85x34 tensor.
///
/// `drawn_tile` should be `Some(tile_type)` when the observer just drew a
/// tile (obtain from `GameState.drawn_tile` mapped to tile type via `/ 4`).
/// Pass `None` when no draw occurred or the information is unavailable.
#[inline]
pub fn encode_observation(
    encoder: &mut ObservationEncoder,
    obs: &Observation,
    safety: &SafetyInfo,
    drawn_tile: Option<u8>,
) -> [f32; OBS_SIZE] {
    let hand = extract_hand(obs);
    let discards = extract_discards(obs);
    let melds = extract_melds(obs);
    let open_meld_counts = extract_observer_meld_counts(obs);
    let dora = extract_dora(obs);
    let meta = extract_metadata(obs, &hand);

    let slice = encoder.encode(
        &hand, drawn_tile, &open_meld_counts, &discards, &melds, &dora, &meta, safety,
    );
    *slice
}

// ---------------------------------------------------------------------------
// ObservationRef extractors (zero-copy path)
// ---------------------------------------------------------------------------

/// Aka dora tile IDs in 136-format (u8 variant for ObservationRef).
const AKA_5M_U8: u8 = 16;
const AKA_5P_U8: u8 = 52;
const AKA_5S_U8: u8 = 88;

/// Extract hand tile counts from an ObservationRef.
///
/// Converts from 136-format `&[u8]` to 34-bin histogram.
#[inline]
pub fn extract_hand_ref(obs: &ObservationRef<'_>) -> [u8; NUM_TILE_TYPES] {
    let mut counts = [0u8; NUM_TILE_TYPES];
    for &tile136 in obs.observer_hand {
        let t = (tile136 / 4) as usize;
        if t < NUM_TILE_TYPES {
            counts[t] = counts[t].saturating_add(1);
        }
    }
    counts
}

/// Extract discard info for all 4 players from an ObservationRef.
///
/// Player indices are RELATIVE to the observer (index 0 = observer).
/// Note: tsumogiri flags are not available on ObservationRef, so all
/// discards default to tedashi=true (conservative for safety encoding).
#[inline]
pub fn extract_discards_ref(obs: &ObservationRef<'_>) -> [PlayerDiscards; 4] {
    let observer = obs.player_id as usize;
    std::array::from_fn(|relative_idx| {
        let abs = (observer + relative_idx) % 4;
        let disc = obs.discards[abs];
        let entries: Vec<DiscardEntry> = disc
            .iter()
            .enumerate()
            .map(|(turn, &tile136)| DiscardEntry {
                tile: (tile136 / 4),
                is_tedashi: true,
                turn: turn as u16,
            })
            .collect();
        PlayerDiscards { discards: entries }
    })
}

/// Extract meld info for all 4 players from an ObservationRef.
#[inline]
pub fn extract_melds_ref(obs: &ObservationRef<'_>) -> [Vec<MeldInfo>; 4] {
    let observer = obs.player_id as usize;
    std::array::from_fn(|relative_idx| {
        let abs = (observer + relative_idx) % 4;
        obs.melds[abs]
            .iter()
            .map(|meld| {
                let tiles: Vec<u8> = meld.tiles.iter().map(|&t| t / 4).collect();
                let meld_type = match meld.meld_type {
                    RiichiMeldType::Chi => MeldType::Chi,
                    RiichiMeldType::Pon => MeldType::Pon,
                    RiichiMeldType::Daiminkan
                    | RiichiMeldType::Ankan
                    | RiichiMeldType::Kakan => MeldType::Kan,
                };
                MeldInfo { tiles, meld_type }
            })
            .collect()
    })
}

/// Count tile types across the observer's melds from an ObservationRef.
#[inline]
pub fn extract_observer_meld_counts_ref(obs: &ObservationRef<'_>) -> [u8; NUM_TILE_TYPES] {
    let observer = obs.player_id as usize;
    let mut counts = [0u8; NUM_TILE_TYPES];
    for meld in obs.melds[observer] {
        for &tile in &meld.tiles {
            let t = (tile / 4) as usize;
            if t < NUM_TILE_TYPES {
                counts[t] = counts[t].saturating_add(1);
            }
        }
    }
    counts
}

/// Extract dora information from an ObservationRef.
#[inline]
pub fn extract_dora_ref(obs: &ObservationRef<'_>) -> DoraInfo {
    let indicators: Vec<u8> = obs
        .dora_indicators
        .iter()
        .map(|&t| t / 4)
        .collect();

    let aka_flags = [
        obs.observer_hand.contains(&AKA_5M_U8),
        obs.observer_hand.contains(&AKA_5P_U8),
        obs.observer_hand.contains(&AKA_5S_U8),
    ];

    DoraInfo {
        indicators,
        aka_flags,
    }
}

/// Extract game metadata from an ObservationRef.
#[inline]
pub fn extract_metadata_ref(
    obs: &ObservationRef<'_>,
    hand_counts: &[u8; NUM_TILE_TYPES],
) -> GameMetadata {
    let observer = obs.player_id as usize;
    let hand_total: u8 = hand_counts.iter().sum();
    let len_div3 = hand_total / 3;
    let shanten = calc_shanten_from_counts(hand_counts, len_div3);

    GameMetadata {
        riichi: std::array::from_fn(|i| {
            obs.riichi_declared[(observer + i) % 4]
        }),
        scores: std::array::from_fn(|i| {
            obs.scores[(observer + i) % 4]
        }),
        shanten,
        kyoku_index: obs.kyoku_index,
        honba: obs.honba,
        kyotaku: obs.riichi_sticks.min(255) as u8,
    }
}

/// Encode directly from a zero-copy observation reference.
///
/// This bypasses `get_observation()` and its ~15 Vec allocations.
/// The `drawn_tile` from `ObservationRef` is automatically converted
/// from 136-format to tile type (/ 4).
#[inline]
pub fn encode_observation_ref(
    encoder: &mut ObservationEncoder,
    obs: &ObservationRef<'_>,
    safety: &SafetyInfo,
) -> [f32; OBS_SIZE] {
    let hand = extract_hand_ref(obs);
    let discards = extract_discards_ref(obs);
    let melds = extract_melds_ref(obs);
    let open_meld_counts = extract_observer_meld_counts_ref(obs);
    let dora = extract_dora_ref(obs);
    let meta = extract_metadata_ref(obs, &hand);
    let drawn_tile = obs.drawn_tile.map(|t| t / 4);

    let slice = encoder.encode(
        &hand, drawn_tile, &open_meld_counts, &discards, &melds, &dora, &meta, safety,
    );
    *slice
}

#[cfg(test)]
mod tests {
    use super::*;
    use riichienv_core::rule::GameRule;
    use riichienv_core::state::GameState;

    /// Create a fresh observation from a newly dealt game.
    fn fresh_obs() -> Observation {
        let rule = GameRule::default_tenhou();
        let mut state = GameState::new(0, true, Some(42), 0, rule);
        state.get_observation(0)
    }

    #[test]
    fn extract_hand_has_13_or_14_tiles() {
        let obs = fresh_obs();
        let hand = extract_hand(&obs);
        let total: u8 = hand.iter().sum();
        assert!(
            (13..=14).contains(&total),
            "hand has {total} tiles, expected 13 or 14",
        );
    }

    #[test]
    fn extract_discards_initially_empty() {
        let obs = fresh_obs();
        let discards = extract_discards(&obs);
        for pd in &discards {
            assert!(pd.discards.is_empty());
        }
    }

    #[test]
    fn extract_melds_initially_empty() {
        let obs = fresh_obs();
        let melds = extract_melds(&obs);
        for player_melds in &melds {
            assert!(player_melds.is_empty());
        }
    }

    #[test]
    fn extract_dora_has_one_indicator() {
        let obs = fresh_obs();
        let dora = extract_dora(&obs);
        assert_eq!(dora.indicators.len(), 1, "initial game has 1 dora indicator");
        assert!(dora.indicators[0] < 34, "tile type must be 0-33");
    }

    #[test]
    fn extract_metadata_sane_values() {
        let obs = fresh_obs();
        let hand = extract_hand(&obs);
        let meta = extract_metadata(&obs, &hand);
        assert_eq!(meta.kyoku_index, obs.kyoku_index);
        assert_eq!(meta.honba, 0);
        assert_eq!(meta.kyotaku, 0);
        // Shanten for a dealt hand should be reasonable (-1 to 8)
        assert!(
            (-1..=8).contains(&meta.shanten),
            "shanten {} out of range",
            meta.shanten,
        );
    }

    #[test]
    fn extract_observer_meld_counts_initially_zero() {
        let obs = fresh_obs();
        let counts = extract_observer_meld_counts(&obs);
        assert_eq!(counts.iter().sum::<u8>(), 0, "no melds at game start");
    }

    #[test]
    fn encode_observation_produces_nonzero() {
        let obs = fresh_obs();
        let safety = SafetyInfo::new();
        let mut encoder = ObservationEncoder::new();
        let result = encode_observation(&mut encoder, &obs, &safety, None);
        let nonzero = result.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero > 0, "encoded observation should have nonzero values");
    }

    #[test]
    fn tile136_to_type_basics() {
        assert_eq!(tile136_to_type(0), 0);   // 1m copy 0
        assert_eq!(tile136_to_type(3), 0);   // 1m copy 3
        assert_eq!(tile136_to_type(4), 1);   // 2m copy 0
        assert_eq!(tile136_to_type(135), 33); // chun copy 3
    }
}
