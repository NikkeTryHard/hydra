//! Bridge between riichienv-core game state and Hydra's observation encoder.
//!
//! Converts riichienv [`Observation`] data into the encoder's input types,
//! then runs the full fixed-superset encoding pipeline. This is the critical glue
//! between the game engine and the neural network.

use riichienv_core::observation::Observation;
use riichienv_core::observation_ref::ObservationRef;
use riichienv_core::shanten::calc_shanten_from_counts;
use riichienv_core::types::MeldType as RiichiMeldType;

use crate::afbs::{AfbsTree, NodeIdx};
use crate::ct_smc::CtSmc;
use crate::encoder::{
    DiscardEntry, DoraInfo, GameMetadata, MeldInfo, MeldType, OBS_SIZE, ObservationEncoder,
    PlayerDiscards, PlayerMelds, SearchFeaturePlanes,
};
use crate::hand_ev::{HandEvFeatures, compute_hand_ev};
use crate::safety::SafetyInfo;
use crate::sinkhorn::MixtureSib;
use crate::tile::NUM_TILE_TYPES;

const NUM_OPPONENTS: usize = 3;
const NUM_MIXTURE_COMPONENTS: usize = 4;
const NUM_BELIEF_ZONES: usize = 4;

/// Optional runtime context used to populate Group C search/belief channels.
///
/// This allows the fixed-superset encoder to consume real belief/search features
/// when they are available, while preserving a backward-safe path when they are not.
#[derive(Clone, Copy, Default)]
pub struct SearchContext<'a> {
    /// Optional Mixture-SIB belief state.
    pub mixture: Option<&'a MixtureSib>,
    /// Optional CT-SMC posterior used for belief-weighted Hand-EV counts.
    pub ct_smc: Option<&'a CtSmc>,
    /// Optional AFBS search tree.
    pub afbs_tree: Option<&'a AfbsTree>,
    /// Optional AFBS root node corresponding to `afbs_tree`.
    pub afbs_root: Option<NodeIdx>,
    /// Optional externally produced per-opponent tile risk planes.
    pub opponent_risk: Option<&'a [[f32; NUM_TILE_TYPES]; NUM_OPPONENTS]>,
    /// Optional externally produced per-opponent scalar stress values.
    pub opponent_stress: Option<&'a [f32; NUM_OPPONENTS]>,
}

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
        let mut pd = PlayerDiscards::new();
        for (turn, &tile136) in disc.iter().enumerate() {
            let is_tsumogiri = tsumogiri.get(turn).copied().unwrap_or(false);
            pd.push(DiscardEntry {
                tile: tile136_to_type(tile136),
                is_tedashi: !is_tsumogiri,
                turn: turn as u16,
            });
        }
        pd
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
pub fn extract_melds(obs: &Observation) -> [PlayerMelds; 4] {
    let observer = obs.player_id as usize;
    std::array::from_fn(|relative_idx| {
        let abs = (observer + relative_idx) % 4;
        let mut pm = PlayerMelds::new();
        for meld in &obs.melds[abs] {
            let mut tiles = [0u8; 4];
            let tile_count = meld.tile_count;
            for (i, &t) in meld.tiles_slice().iter().enumerate() {
                tiles[i] = t / 4;
            }
            let meld_type = match meld.meld_type {
                RiichiMeldType::Chi => MeldType::Chi,
                RiichiMeldType::Pon => MeldType::Pon,
                RiichiMeldType::Daiminkan | RiichiMeldType::Ankan | RiichiMeldType::Kakan => {
                    MeldType::Kan
                }
            };
            pm.push(MeldInfo {
                tiles,
                tile_count,
                meld_type,
            });
        }
        pm
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
        for &tile in meld.tiles_slice() {
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
    let mut indicators = [0u8; 5];
    let indicator_count = obs.dora_indicators.len().min(5) as u8;
    for (i, &t) in obs.dora_indicators.iter().enumerate().take(5) {
        indicators[i] = tile136_to_type(t);
    }

    // Single-pass aka dora detection
    let observer = obs.player_id as usize;
    let mut aka_flags = [false; 3];
    for &t in &obs.hands[observer] {
        match t {
            16 => aka_flags[0] = true,
            52 => aka_flags[1] = true,
            88 => aka_flags[2] = true,
            _ => {}
        }
    }

    DoraInfo {
        indicators,
        indicator_count,
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
        riichi: std::array::from_fn(|i| obs.riichi_declared[(observer + i) % 4]),
        scores: std::array::from_fn(|i| obs.scores[(observer + i) % 4]),
        shanten,
        kyoku_index: obs.kyoku_index,
        honba: obs.honba,
        kyotaku: obs.riichi_sticks.min(255) as u8,
    }
}

/// Compute public-state remaining tile counts for the observer.
///
/// This subtracts all tiles visible to the observer: their concealed hand,
/// all open melds, all discards, and visible dora indicators. This is a safe
/// bridge-side approximation for Hand-EV features until belief-weighted
/// remaining counts from CT-SMC are threaded into the encoder path.
#[inline]
pub fn extract_public_remaining_counts(
    hand: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
) -> [f32; NUM_TILE_TYPES] {
    let mut remaining = [4.0f32; NUM_TILE_TYPES];

    for (tile, &count) in hand.iter().enumerate() {
        remaining[tile] -= count as f32;
    }
    for player_discards in discards {
        for entry in player_discards
            .discards
            .iter()
            .take(player_discards.len as usize)
        {
            remaining[entry.tile as usize] -= 1.0;
        }
    }
    for player_melds in melds {
        for meld in player_melds.melds.iter().take(player_melds.len as usize) {
            for &tile in meld.tiles.iter().take(meld.tile_count as usize) {
                remaining[tile as usize] -= 1.0;
            }
        }
    }
    for &indicator in dora.indicators.iter().take(dora.indicator_count as usize) {
        remaining[indicator as usize] -= 1.0;
    }

    for value in &mut remaining {
        *value = value.max(0.0);
    }
    remaining
}

/// Compute bridge-side Hand-EV features from public-state remaining counts.
#[inline]
pub fn compute_public_hand_ev(
    hand: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
) -> HandEvFeatures {
    let remaining = extract_public_remaining_counts(hand, discards, melds, dora);
    compute_hand_ev(hand, &remaining)
}

/// Compute belief-weighted remaining tile counts from a CT-SMC posterior.
#[inline]
pub fn extract_ct_smc_remaining_counts(ct_smc: &CtSmc) -> [f32; NUM_TILE_TYPES] {
    let mut remaining = [0.0f32; NUM_TILE_TYPES];
    if ct_smc.is_empty() {
        return remaining;
    }
    for (tile, slot) in remaining.iter_mut().enumerate() {
        *slot = (0..4)
            .map(|col| ct_smc.weighted_mean_tile_count(tile as u8, col as u8))
            .sum();
    }
    remaining
}

/// Compute bridge-side Hand-EV features from CT-SMC belief-weighted counts.
#[inline]
pub fn compute_ct_smc_hand_ev(hand: &[u8; NUM_TILE_TYPES], ct_smc: &CtSmc) -> HandEvFeatures {
    let remaining = extract_ct_smc_remaining_counts(ct_smc);
    compute_hand_ev(hand, &remaining)
}

#[inline]
fn compute_hand_ev_from_context(
    hand: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
    search_context: &SearchContext<'_>,
) -> HandEvFeatures {
    if let Some(ct_smc) = search_context.ct_smc
        && !ct_smc.is_empty()
    {
        return compute_ct_smc_hand_ev(hand, ct_smc);
    }
    compute_public_hand_ev(hand, discards, melds, dora)
}

/// Build fixed-shape Group C search/belief planes from available runtime context.
///
/// Current sources:
/// - Mixture-SIB -> belief fields, weights, entropy, ESS
/// - AFBS root -> discard-level delta-Q summary for expanded discard actions
/// - safety/opponent model cache -> per-opponent stress and matagi danger fallback
/// - explicit robust risk/stress overrides when provided
#[inline]
pub fn build_search_features(
    safety: &SafetyInfo,
    context: &SearchContext<'_>,
) -> SearchFeaturePlanes {
    let mut features = SearchFeaturePlanes::default();

    if let Some(mixture) = context.mixture {
        let weights = mixture.weights();
        let mut ranked: Vec<(usize, f64)> = weights.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (rank, (component_idx, weight)) in ranked
            .iter()
            .take(NUM_MIXTURE_COMPONENTS)
            .copied()
            .enumerate()
        {
            features.mixture_weights[rank] = weight as f32;
            for zone in 0..NUM_BELIEF_ZONES {
                let channel = rank * NUM_BELIEF_ZONES + zone;
                for tile in 0..NUM_TILE_TYPES {
                    features.belief_fields[channel][tile] = mixture.components[component_idx].belief
                        [tile * NUM_BELIEF_ZONES + zone]
                        as f32;
                }
            }
        }

        features.mixture_entropy = mixture.weight_entropy() as f32;
        features.mixture_ess = mixture.ess() as f32;
        features.belief_features_present = true;
        features.context_features_present = true;
    }

    if let (Some(tree), Some(root)) = (context.afbs_tree, context.afbs_root) {
        let root_q = tree.node_q_value(root);
        let mut any_delta_q = false;
        for action in 0..NUM_TILE_TYPES as u8 {
            if let Some(child) = tree.find_child_by_action(root, action) {
                features.delta_q[action as usize] = tree.node_q_value(child) - root_q;
                any_delta_q = true;
            }
        }
        if any_delta_q {
            features.search_features_present = true;
            features.context_features_present = true;
        }
    }

    for opp in 0..NUM_OPPONENTS {
        features.opponent_risk[opp] = safety.matagi[opp];
        features.opponent_stress[opp] = if safety.opponent_riichi[opp] {
            1.0
        } else {
            safety.cached_tenpai_prob[opp]
        };
    }

    if let Some(risk) = context.opponent_risk {
        features.opponent_risk = *risk;
    }
    if let Some(stress) = context.opponent_stress {
        features.opponent_stress = *stress;
    }

    let robust_signal_present = features
        .opponent_risk
        .iter()
        .flat_map(|plane| plane.iter())
        .any(|&v| v != 0.0)
        || features.opponent_stress.iter().any(|&v| v != 0.0);
    if robust_signal_present {
        features.robust_features_present = true;
        features.context_features_present = true;
    }

    features
}

/// Encode a full observation into the fixed-superset tensor with optional Group C runtime context.
#[inline]
pub fn encode_observation_with_search_context(
    encoder: &mut ObservationEncoder,
    obs: &Observation,
    safety: &SafetyInfo,
    drawn_tile: Option<u8>,
    search_context: &SearchContext<'_>,
) -> [f32; OBS_SIZE] {
    let hand = extract_hand(obs);
    let discards = extract_discards(obs);
    let melds = extract_melds(obs);
    let open_meld_counts = extract_observer_meld_counts(obs);
    let dora = extract_dora(obs);
    let meta = extract_metadata(obs, &hand);
    let hand_ev = compute_hand_ev_from_context(&hand, &discards, &melds, &dora, search_context);
    let search_features = build_search_features(safety, search_context);

    let slice = encoder.encode_with_context(
        &hand,
        drawn_tile,
        &open_meld_counts,
        &discards,
        &melds,
        &dora,
        &meta,
        safety,
        Some(&search_features),
        Some(&hand_ev),
    );
    *slice
}

/// Encode a full observation into the fixed-superset tensor.
///
/// This is the main bridge entry point. Extracts all components from
/// a riichienv [`Observation`], feeds them through the encoder pipeline,
/// and returns a reference to the filled `[f32; 2890]` buffer.
///
/// # Drawn tile limitation
///
/// The drawn tile cannot be reliably determined from `Observation` alone.
/// Encode a full observation into the fixed-superset tensor.
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
    let search_context = SearchContext::default();
    encode_observation_with_search_context(encoder, obs, safety, drawn_tile, &search_context)
}

// ---------------------------------------------------------------------------
// ObservationRef extractors (zero-copy path)
// ---------------------------------------------------------------------------

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
        let mut pd = PlayerDiscards::new();
        for (turn, &tile136) in disc.iter().enumerate() {
            pd.push(DiscardEntry {
                tile: (tile136 / 4),
                is_tedashi: true,
                turn: turn as u16,
            });
        }
        pd
    })
}

/// Extract meld info for all 4 players from an ObservationRef.
#[inline]
pub fn extract_melds_ref(obs: &ObservationRef<'_>) -> [PlayerMelds; 4] {
    let observer = obs.player_id as usize;
    std::array::from_fn(|relative_idx| {
        let abs = (observer + relative_idx) % 4;
        let mut pm = PlayerMelds::new();
        for meld in obs.melds[abs] {
            let mut tiles = [0u8; 4];
            let tile_count = meld.tile_count;
            for (i, &t) in meld.tiles_slice().iter().enumerate() {
                tiles[i] = t / 4;
            }
            let meld_type = match meld.meld_type {
                RiichiMeldType::Chi => MeldType::Chi,
                RiichiMeldType::Pon => MeldType::Pon,
                RiichiMeldType::Daiminkan | RiichiMeldType::Ankan | RiichiMeldType::Kakan => {
                    MeldType::Kan
                }
            };
            pm.push(MeldInfo {
                tiles,
                tile_count,
                meld_type,
            });
        }
        pm
    })
}

/// Count tile types across the observer's melds from an ObservationRef.
#[inline]
pub fn extract_observer_meld_counts_ref(obs: &ObservationRef<'_>) -> [u8; NUM_TILE_TYPES] {
    let observer = obs.player_id as usize;
    let mut counts = [0u8; NUM_TILE_TYPES];
    for meld in obs.melds[observer] {
        for &tile in meld.tiles_slice() {
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
    let mut indicators = [0u8; 5];
    let indicator_count = obs.dora_indicators.len().min(5) as u8;
    for (i, &t) in obs.dora_indicators.iter().enumerate().take(5) {
        indicators[i] = t / 4;
    }

    let mut aka_flags = [false; 3];
    for &t in obs.observer_hand {
        match t {
            16 => aka_flags[0] = true,
            52 => aka_flags[1] = true,
            88 => aka_flags[2] = true,
            _ => {}
        }
    }

    DoraInfo {
        indicators,
        indicator_count,
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
        riichi: std::array::from_fn(|i| obs.riichi_declared[(observer + i) % 4]),
        scores: std::array::from_fn(|i| obs.scores[(observer + i) % 4]),
        shanten,
        kyoku_index: obs.kyoku_index,
        honba: obs.honba,
        kyotaku: obs.riichi_sticks.min(255) as u8,
    }
}

/// Compute public-state remaining tile counts from a zero-copy observation.
#[inline]
pub fn extract_public_remaining_counts_ref(
    hand: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
) -> [f32; NUM_TILE_TYPES] {
    extract_public_remaining_counts(hand, discards, melds, dora)
}

/// Compute bridge-side Hand-EV features from a zero-copy observation path.
#[inline]
pub fn compute_public_hand_ev_ref(
    hand: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
) -> HandEvFeatures {
    let remaining = extract_public_remaining_counts_ref(hand, discards, melds, dora);
    compute_hand_ev(hand, &remaining)
}

#[inline]
fn compute_hand_ev_from_context_ref(
    hand: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
    search_context: &SearchContext<'_>,
) -> HandEvFeatures {
    if let Some(ct_smc) = search_context.ct_smc
        && !ct_smc.is_empty()
    {
        return compute_ct_smc_hand_ev(hand, ct_smc);
    }
    compute_public_hand_ev_ref(hand, discards, melds, dora)
}

/// Encode a zero-copy observation into the fixed-superset tensor with optional Group C runtime context.
#[inline]
pub fn encode_observation_ref_with_search_context(
    encoder: &mut ObservationEncoder,
    obs: &ObservationRef<'_>,
    safety: &SafetyInfo,
    search_context: &SearchContext<'_>,
) -> [f32; OBS_SIZE] {
    let hand = extract_hand_ref(obs);
    let discards = extract_discards_ref(obs);
    let melds = extract_melds_ref(obs);
    let open_meld_counts = extract_observer_meld_counts_ref(obs);
    let dora = extract_dora_ref(obs);
    let meta = extract_metadata_ref(obs, &hand);
    let drawn_tile = obs.drawn_tile.map(|t| t / 4);
    let hand_ev = compute_hand_ev_from_context_ref(&hand, &discards, &melds, &dora, search_context);
    let search_features = build_search_features(safety, search_context);

    let slice = encoder.encode_with_context(
        &hand,
        drawn_tile,
        &open_meld_counts,
        &discards,
        &melds,
        &dora,
        &meta,
        safety,
        Some(&search_features),
        Some(&hand_ev),
    );
    *slice
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
    let search_context = SearchContext::default();
    encode_observation_ref_with_search_context(encoder, obs, safety, &search_context)
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
            assert_eq!(pd.len, 0);
        }
    }

    #[test]
    fn extract_melds_initially_empty() {
        let obs = fresh_obs();
        let melds = extract_melds(&obs);
        for player_melds in &melds {
            assert_eq!(player_melds.len, 0);
        }
    }

    #[test]
    fn extract_dora_has_one_indicator() {
        let obs = fresh_obs();
        let dora = extract_dora(&obs);
        assert_eq!(dora.indicator_count, 1, "initial game has 1 dora indicator");
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
        assert!(
            nonzero > 0,
            "encoded observation should have nonzero values"
        );
    }

    #[test]
    fn public_remaining_counts_subtract_visible_tiles() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 2;
        hand[1] = 1;

        let mut discards = std::array::from_fn(|_| PlayerDiscards::new());
        discards[0].push(DiscardEntry {
            tile: 0,
            is_tedashi: true,
            turn: 0,
        });

        let mut melds = std::array::from_fn(|_| PlayerMelds::new());
        melds[1].push(MeldInfo {
            tiles: [1, 1, 1, 0],
            tile_count: 3,
            meld_type: MeldType::Pon,
        });

        let dora = DoraInfo {
            indicators: [0, 0, 0, 0, 0],
            indicator_count: 1,
            aka_flags: [false; 3],
        };

        let remaining = extract_public_remaining_counts(&hand, &discards, &melds, &dora);
        assert_eq!(
            remaining[0], 0.0,
            "2 in hand + 1 discard + 1 dora indicator exhaust tile 0"
        );
        assert_eq!(remaining[1], 0.0, "1 in hand + pon exhaust tile 1");
        assert_eq!(
            remaining[2], 4.0,
            "unseen tile should keep full remaining count"
        );
    }

    #[test]
    fn compute_public_hand_ev_on_real_observation_has_signal() {
        let obs = fresh_obs();
        let hand = extract_hand(&obs);
        let discards = extract_discards(&obs);
        let melds = extract_melds(&obs);
        let dora = extract_dora(&obs);
        let hand_ev = compute_public_hand_ev(&hand, &discards, &melds, &dora);

        let any_tenpai = hand_ev
            .tenpai_prob
            .iter()
            .flat_map(|p| p.iter())
            .any(|&v| v > 0.0);
        let any_ukeire = hand_ev
            .ukeire
            .iter()
            .flat_map(|u| u.iter())
            .any(|&v| v > 0.0);

        assert!(
            any_tenpai || any_ukeire,
            "public Hand-EV should expose some nonzero signal"
        );
    }

    #[test]
    fn extract_ct_smc_remaining_counts_sums_weighted_hidden_columns() {
        let mut smc = CtSmc::new(crate::ct_smc::CtSmcConfig::default().with_particles(2));
        smc.particles = vec![
            crate::ct_smc::Particle {
                allocation: {
                    let mut allocation = [[0u8; 4]; 34];
                    allocation[3] = [1, 1, 0, 0];
                    allocation[7] = [0, 0, 1, 0];
                    allocation
                },
                log_weight: 0.0,
            },
            crate::ct_smc::Particle {
                allocation: {
                    let mut allocation = [[0u8; 4]; 34];
                    allocation[3] = [0, 1, 1, 1];
                    allocation
                },
                log_weight: 0.0,
            },
        ];

        let remaining = extract_ct_smc_remaining_counts(&smc);
        assert!((remaining[3] - 2.5).abs() < 1e-6);
        assert!((remaining[7] - 0.5).abs() < 1e-6);
        assert_eq!(remaining[2], 0.0);
    }

    #[test]
    fn compute_ct_smc_hand_ev_uses_weighted_remaining_counts() {
        let mut hand = [0u8; NUM_TILE_TYPES];
        hand[0] = 1;
        hand[1] = 1;

        let mut smc = CtSmc::new(crate::ct_smc::CtSmcConfig::default().with_particles(2));
        smc.particles = vec![
            crate::ct_smc::Particle {
                allocation: {
                    let mut allocation = [[0u8; 4]; 34];
                    allocation[0] = [1, 0, 0, 0];
                    allocation
                },
                log_weight: 0.0,
            },
            crate::ct_smc::Particle {
                allocation: {
                    let mut allocation = [[0u8; 4]; 34];
                    allocation[0] = [0, 0, 0, 1];
                    allocation
                },
                log_weight: 0.0,
            },
        ];

        let features = compute_ct_smc_hand_ev(&hand, &smc);
        assert!(features.ukeire[1][0] > 0.0);
        assert!(features.expected_score[1] > 0.0);
    }

    #[test]
    fn build_search_features_from_mixture_populates_belief_and_weights() {
        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0f64; 4];
        let mut mixture = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
        mixture.bayesian_update(&[1.5, 0.5, -0.5, -1.5]);

        let mut safety = SafetyInfo::new();
        safety.set_tenpai_prediction(0, 0.6);
        safety.on_discard(5, 1, true);

        let context = SearchContext {
            mixture: Some(&mixture),
            ..SearchContext::default()
        };
        let features = build_search_features(&safety, &context);

        assert!(features.belief_features_present);
        assert!(features.context_features_present);
        assert!(features.mixture_weights.iter().any(|&v| v > 0.0));
        assert!(features.mixture_entropy > 0.0);
        assert!(features.mixture_ess > 0.0);
        assert!(features.belief_fields.iter().flatten().any(|&v| v > 0.0));
        assert!(features.opponent_risk[1][4] > 0.0 || features.opponent_risk[1][6] > 0.0);
        assert!((features.opponent_stress[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn build_search_features_from_afbs_populates_delta_q() {
        let mut tree = AfbsTree::new();
        let root = tree.add_node(100, 1.0, false);
        tree.nodes[root as usize].visit_count = 10;
        tree.nodes[root as usize].total_value = 4.0; // q = 0.4

        let child_a = tree.add_node(101, 0.6, false);
        tree.nodes[child_a as usize].visit_count = 4;
        tree.nodes[child_a as usize].total_value = 3.2; // q = 0.8

        let child_b = tree.add_node(105, 0.4, false);
        tree.nodes[child_b as usize].visit_count = 4;
        tree.nodes[child_b as usize].total_value = 0.4; // q = 0.1

        tree.nodes[root as usize].children = vec![(0, child_a), (5, child_b)].into();

        let context = SearchContext {
            afbs_tree: Some(&tree),
            afbs_root: Some(root),
            ..SearchContext::default()
        };
        let features = build_search_features(&SafetyInfo::new(), &context);

        assert!(features.search_features_present);
        assert!(features.context_features_present);
        assert!((features.delta_q[0] - 0.4).abs() < 1e-6);
        assert!((features.delta_q[5] + 0.3).abs() < 1e-6);
    }

    #[test]
    fn encode_observation_populates_hand_ev_planes() {
        let obs = fresh_obs();
        let safety = SafetyInfo::new();
        let mut encoder = ObservationEncoder::new();
        let result = encode_observation(&mut encoder, &obs, &safety, None);

        let mask_offset = crate::encoder::HAND_EV_MASK_CHANNEL * NUM_TILE_TYPES;
        assert_eq!(
            result[mask_offset], 1.0,
            "Hand-EV presence mask should be enabled"
        );

        let hand_ev_payload =
            &result[crate::encoder::HAND_EV_CHANNEL_START * NUM_TILE_TYPES..mask_offset];
        let nonzero = hand_ev_payload.iter().filter(|&&v| v != 0.0).count();
        assert!(
            nonzero > 0,
            "encoded observation should contain nonzero Hand-EV payload"
        );
    }

    #[test]
    fn encode_observation_with_search_context_populates_group_c_planes() {
        let obs = fresh_obs();
        let mut safety = SafetyInfo::new();
        safety.set_tenpai_prediction(0, 0.7);
        safety.on_discard(5, 1, true);

        let kernel = [1.0f64; 136];
        let row_sums = [4.0f64; 34];
        let col_sums = [34.0f64; 4];
        let mut mixture = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
        mixture.bayesian_update(&[1.0, 0.0, -0.5, -1.0]);

        let mut tree = AfbsTree::new();
        let root = tree.add_node(7, 1.0, false);
        tree.nodes[root as usize].visit_count = 10;
        tree.nodes[root as usize].total_value = 2.0;
        let child = tree.add_node(11, 1.0, false);
        tree.nodes[child as usize].visit_count = 5;
        tree.nodes[child as usize].total_value = 3.0;
        tree.nodes[root as usize].children = vec![(0, child)].into();

        let context = SearchContext {
            mixture: Some(&mixture),
            afbs_tree: Some(&tree),
            afbs_root: Some(root),
            ..SearchContext::default()
        };

        let mut encoder = ObservationEncoder::new();
        let result =
            encode_observation_with_search_context(&mut encoder, &obs, &safety, None, &context);

        let belief_mask = crate::encoder::SEARCH_MASK_CHANNEL_START * NUM_TILE_TYPES;
        let search_mask = (crate::encoder::SEARCH_MASK_CHANNEL_START + 1) * NUM_TILE_TYPES;
        let robust_mask = (crate::encoder::SEARCH_MASK_CHANNEL_START + 2) * NUM_TILE_TYPES;
        assert_eq!(result[belief_mask], 1.0);
        assert_eq!(result[search_mask], 1.0);
        assert_eq!(result[robust_mask], 1.0);

        let belief_payload = result[crate::encoder::SEARCH_BELIEF_CHANNEL_START * NUM_TILE_TYPES
            ..crate::encoder::SEARCH_DELTA_Q_CHANNEL * NUM_TILE_TYPES]
            .iter()
            .filter(|&&v| v != 0.0)
            .count();
        let delta_q_payload = result[crate::encoder::SEARCH_DELTA_Q_CHANNEL * NUM_TILE_TYPES];
        assert!(
            belief_payload > 0,
            "belief/search payload should be nonzero"
        );
        assert!(
            delta_q_payload > 0.0,
            "delta-q channel should reflect AFBS context"
        );
    }

    #[test]
    fn encode_observation_with_ct_smc_context_uses_belief_weighted_hand_ev() {
        let obs = fresh_obs();
        let safety = SafetyInfo::new();
        let hand = extract_hand(&obs);

        let mut smc = CtSmc::new(crate::ct_smc::CtSmcConfig::default().with_particles(2));
        smc.particles = vec![
            crate::ct_smc::Particle {
                allocation: {
                    let mut allocation = [[0u8; 4]; 34];
                    for tile in 0..NUM_TILE_TYPES {
                        if hand[tile] == 0 {
                            allocation[tile][3] = 1;
                        }
                    }
                    allocation
                },
                log_weight: 0.0,
            },
            crate::ct_smc::Particle {
                allocation: {
                    let mut allocation = [[0u8; 4]; 34];
                    for tile in 0..NUM_TILE_TYPES {
                        if hand[tile] == 0 && tile % 2 == 0 {
                            allocation[tile][2] = 1;
                        }
                    }
                    allocation
                },
                log_weight: 0.0,
            },
        ];

        let context = SearchContext {
            ct_smc: Some(&smc),
            ..SearchContext::default()
        };

        let mut encoder = ObservationEncoder::new();
        let result =
            encode_observation_with_search_context(&mut encoder, &obs, &safety, None, &context);

        let hand_ev_payload = &result[crate::encoder::HAND_EV_CHANNEL_START * NUM_TILE_TYPES
            ..crate::encoder::HAND_EV_MASK_CHANNEL * NUM_TILE_TYPES];
        let nonzero = hand_ev_payload.iter().filter(|&&v| v != 0.0).count();
        assert!(
            nonzero > 0,
            "CT-SMC context should produce nonzero Hand-EV payload"
        );
    }

    #[test]
    fn tile136_to_type_basics() {
        assert_eq!(tile136_to_type(0), 0); // 1m copy 0
        assert_eq!(tile136_to_type(3), 0); // 1m copy 3
        assert_eq!(tile136_to_type(4), 1); // 2m copy 0
        assert_eq!(tile136_to_type(135), 33); // chun copy 3
    }
}
