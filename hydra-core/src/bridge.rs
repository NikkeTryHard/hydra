//! Bridge between riichienv-core game state and Hydra's observation encoder.
//!
//! Converts riichienv [`Observation`] data into the encoder's input types,
//! then runs the full 85x34 encoding pipeline. This is the critical glue
//! between the game engine and the neural network.

use riichienv_core::observation::Observation;
use riichienv_core::types::MeldType as RiichiMeldType;

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
/// Maps riichienv `MeldType` variants to encoder's three-category system:
/// - Chi/Pon/Daiminkan -> `MeldType::Open`
/// - Ankan -> `MeldType::ClosedKan`
/// - Kakan -> `MeldType::Kakan`
///
/// Meld tile IDs are converted from 136-format (u8) to 34-format tile types.
pub fn extract_melds(obs: &Observation) -> [Vec<MeldInfo>; 4] {
    let observer = obs.player_id as usize;
    std::array::from_fn(|relative_idx| {
        let abs = (observer + relative_idx) % 4;
        obs.melds[abs]
            .iter()
            .map(|meld| {
                let tiles: Vec<u8> = meld.tiles.iter().map(|&t| t / 4).collect();
                let meld_type = match meld.meld_type {
                    RiichiMeldType::Chi
                    | RiichiMeldType::Pon
                    | RiichiMeldType::Daiminkan => MeldType::Open,
                    RiichiMeldType::Ankan => MeldType::ClosedKan,
                    RiichiMeldType::Kakan => MeldType::Kakan,
                };
                MeldInfo { tiles, meld_type }
            })
            .collect()
    })
}

/// Extract dora information from an Observation.
///
/// Converts dora indicator tile IDs from 136-format to 34-format tile types.
/// Scans the observer's hand for aka dora (red fives) at 136-format
/// indices 16 (5m), 52 (5p), 88 (5s).
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

/// Estimate tiles remaining in the wall from observable information.
///
/// Standard 4-player: 136 total - 14 dead wall - 52 dealt (13x4) = 70 drawable.
/// Each discard or kan draw reduces this count.
fn estimate_tiles_remaining(obs: &Observation) -> u8 {
    let total_discards: usize = obs.discards.iter().map(|d| d.len()).sum();
    // Each open meld (chi/pon) consumed 1 tile from discards (already counted),
    // but kan calls draw from the dead wall, reducing drawable tiles by 1 each.
    let kan_count: usize = obs.melds.iter().flat_map(|m| m.iter()).filter(|m| {
        matches!(
            m.meld_type,
            RiichiMeldType::Daiminkan | RiichiMeldType::Ankan | RiichiMeldType::Kakan
        )
    }).count();
    70u8.saturating_sub((total_discards + kan_count) as u8)
}

/// Extract game metadata from an Observation.
///
/// All player-relative fields (riichi, scores) are rotated so
/// index 0 = observer, index 1 = shimocha, etc.
pub fn extract_metadata(obs: &Observation) -> GameMetadata {
    let observer = obs.player_id as usize;
    GameMetadata {
        round_wind: obs.round_wind,
        seat_wind: (obs.player_id + 4 - obs.oya) % 4,
        is_dealer: obs.player_id == obs.oya,
        riichi: std::array::from_fn(|i| {
            obs.riichi_declared[(observer + i) % 4]
        }),
        honba: obs.honba,
        riichi_sticks: obs.riichi_sticks.min(255) as u8,
        tiles_remaining: estimate_tiles_remaining(obs),
        scores: std::array::from_fn(|i| {
            obs.scores[(observer + i) % 4]
        }),
    }
}

/// Encode a full observation into the 85x34 tensor.
///
/// This is the main bridge entry point. Extracts all components from
/// a riichienv [`Observation`], feeds them through the encoder pipeline,
/// and returns a reference to the filled `[f32; 2890]` buffer.
pub fn encode_observation(
    encoder: &mut ObservationEncoder,
    obs: &Observation,
    safety: &SafetyInfo,
) -> [f32; OBS_SIZE] {
    let hand = extract_hand(obs);
    let discards = extract_discards(obs);
    let melds = extract_melds(obs);
    let dora = extract_dora(obs);
    let meta = extract_metadata(obs);

    let slice = encoder.encode(&hand, None, &discards, &melds, &dora, &meta, safety);
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
        let meta = extract_metadata(&obs);
        assert_eq!(meta.round_wind, 0, "first round is East");
        assert!(meta.seat_wind < 4);
        assert!(meta.tiles_remaining <= 70);
        assert_eq!(meta.honba, 0);
    }

    #[test]
    fn encode_observation_produces_nonzero() {
        let obs = fresh_obs();
        let safety = SafetyInfo::new();
        let mut encoder = ObservationEncoder::new();
        let result = encode_observation(&mut encoder, &obs, &safety);
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
