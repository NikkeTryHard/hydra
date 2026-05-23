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
use crate::shanten_batch::{BatchShantenResult, batch_discard_shanten};
use crate::sinkhorn::MixtureSib;
use crate::tile::NUM_TILE_TYPES;

#[derive(Clone, Debug)]
pub struct ExtractedObservationFacts {
    pub hand: [u8; NUM_TILE_TYPES],
    pub drawn_tile: Option<u8>,
    pub open_meld_counts: [u8; NUM_TILE_TYPES],
    pub discards: [PlayerDiscards; 4],
    pub melds: [PlayerMelds; 4],
    pub dora: DoraInfo,
    pub meta: GameMetadata,
    pub shanten_batch: BatchShantenResult,
}

const NUM_OPPONENTS: usize = 3;
const NUM_MIXTURE_COMPONENTS: usize = 4;
const NUM_BELIEF_ZONES: usize = 4;

/// Bridge encoding feature switches.
///
/// `bc_minimal` is the replay/plain-BC baseline profile: it skips Hand-EV
/// planes, but still permits search planes when explicit runtime context is
/// supplied. With an empty `SearchContext`, search planes are cleared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BridgeEncodeProfile {
    pub include_search_features: bool,
    pub include_hand_ev: bool,
}

impl BridgeEncodeProfile {
    pub const fn full() -> Self {
        Self {
            include_search_features: true,
            include_hand_ev: true,
        }
    }

    /// Replay/plain-BC baseline: no Hand-EV; search planes only from explicit runtime context.
    pub const fn bc_minimal() -> Self {
        Self {
            include_search_features: true,
            include_hand_ev: false,
        }
    }
}

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

#[inline]
fn aka_flags_from_tiles<I>(tiles: I) -> [bool; 3]
where
    I: IntoIterator<Item = u8>,
{
    let mut aka_flags = [false; 3];
    for tile in tiles {
        match tile {
            16 => aka_flags[0] = true,
            52 => aka_flags[1] = true,
            88 => aka_flags[2] = true,
            _ => {}
        }
    }
    aka_flags
}

#[inline]
fn dora_info_from_parts<I, J>(indicator_tiles: I, observer_tiles: J) -> DoraInfo
where
    I: IntoIterator<Item = u8>,
    J: IntoIterator<Item = u8>,
{
    let mut indicators = [0u8; 5];
    let mut indicator_count = 0u8;
    for (idx, tile) in indicator_tiles.into_iter().take(5).enumerate() {
        indicators[idx] = tile;
        indicator_count += 1;
    }

    DoraInfo {
        indicators,
        indicator_count,
        aka_flags: aka_flags_from_tiles(observer_tiles),
    }
}

#[inline]
fn metadata_from_parts(
    observer: usize,
    riichi_declared: &[bool; 4],
    scores: &[i32; 4],
    kyoku_index: u8,
    honba: u8,
    kyotaku: u8,
    hand_counts: &[u8; NUM_TILE_TYPES],
) -> GameMetadata {
    let hand_total: u8 = hand_counts.iter().sum();
    let len_div3 = hand_total / 3;
    let shanten = calc_shanten_from_counts(hand_counts, len_div3);
    metadata_from_parts_with_shanten(
        observer,
        riichi_declared,
        scores,
        kyoku_index,
        honba,
        kyotaku,
        shanten,
    )
}

#[inline]
fn metadata_from_parts_with_shanten(
    observer: usize,
    riichi_declared: &[bool; 4],
    scores: &[i32; 4],
    kyoku_index: u8,
    honba: u8,
    kyotaku: u8,
    shanten: i8,
) -> GameMetadata {
    GameMetadata {
        riichi: std::array::from_fn(|i| riichi_declared[(observer + i) % 4]),
        scores: std::array::from_fn(|i| scores[(observer + i) % 4]),
        shanten,
        kyoku_index,
        honba,
        kyotaku,
    }
}

#[inline]
fn search_context_has_runtime_planes(context: &SearchContext<'_>) -> bool {
    context.mixture.is_some()
        || (context.afbs_tree.is_some() && context.afbs_root.is_some())
        || context.opponent_risk.is_some()
        || context.opponent_stress.is_some()
}

#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "encoder API mirrors the fixed observation layout"
)]
fn encode_extracted_observation_with_profile(
    encoder: &mut ObservationEncoder,
    hand: &[u8; NUM_TILE_TYPES],
    drawn_tile: Option<u8>,
    open_meld_counts: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
    meta: &GameMetadata,
    shanten_batch: &BatchShantenResult,
    safety: &SafetyInfo,
    search_context: &SearchContext<'_>,
    profile: BridgeEncodeProfile,
) -> [f32; OBS_SIZE] {
    let hand_ev = if profile.include_hand_ev {
        Some(compute_hand_ev_from_context(
            hand,
            discards,
            melds,
            dora,
            search_context,
        ))
    } else {
        None
    };
    let search_features =
        if profile.include_search_features && search_context_has_runtime_planes(search_context) {
            Some(build_search_features(safety, search_context))
        } else {
            None
        };
    let slice = encoder.encode_with_context_and_shanten_batch(
        hand,
        drawn_tile,
        open_meld_counts,
        discards,
        melds,
        dora,
        meta,
        safety,
        shanten_batch,
        search_features.as_ref(),
        hand_ev.as_ref(),
    );
    *slice
}

#[inline]
#[allow(
    clippy::too_many_arguments,
    reason = "encoder API mirrors the fixed observation layout"
)]
fn encode_extracted_observation(
    encoder: &mut ObservationEncoder,
    hand: &[u8; NUM_TILE_TYPES],
    drawn_tile: Option<u8>,
    open_meld_counts: &[u8; NUM_TILE_TYPES],
    discards: &[PlayerDiscards; 4],
    melds: &[PlayerMelds; 4],
    dora: &DoraInfo,
    meta: &GameMetadata,
    safety: &SafetyInfo,
    search_context: &SearchContext<'_>,
) -> [f32; OBS_SIZE] {
    let hand_total: u8 = hand.iter().sum();
    let shanten_batch = batch_discard_shanten(hand, hand_total / 3);
    encode_extracted_observation_with_profile(
        encoder,
        hand,
        drawn_tile,
        open_meld_counts,
        discards,
        melds,
        dora,
        meta,
        &shanten_batch,
        safety,
        search_context,
        BridgeEncodeProfile::full(),
    )
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
    let observer = obs.player_id as usize;
    dora_info_from_parts(
        obs.dora_indicators.iter().copied().map(tile136_to_type),
        obs.hands[observer].iter().copied().map(|tile| tile as u8),
    )
}

/// Extract game metadata from an Observation.
///
/// Computes shanten from the observer's hand counts. All player-relative
/// fields (riichi, scores) are rotated so index 0 = observer,
/// index 1 = shimocha, etc.
#[inline]
pub fn extract_metadata(obs: &Observation, hand_counts: &[u8; NUM_TILE_TYPES]) -> GameMetadata {
    let observer = obs.player_id as usize;
    metadata_from_parts(
        observer,
        &obs.riichi_declared,
        &obs.scores,
        obs.kyoku_index,
        obs.honba,
        obs.riichi_sticks.min(255) as u8,
        hand_counts,
    )
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

/// Compute wall-weighted remaining tile counts from a CT-SMC posterior.
#[inline]
pub fn extract_ct_smc_remaining_counts(ct_smc: &CtSmc) -> [f32; NUM_TILE_TYPES] {
    let mut remaining = [0.0f32; NUM_TILE_TYPES];
    if ct_smc.is_empty() {
        return remaining;
    }
    for (tile, slot) in remaining.iter_mut().enumerate() {
        *slot = ct_smc.weighted_mean_tile_count(tile as u8, 3);
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
        let mut ranked: [usize; NUM_MIXTURE_COMPONENTS] = std::array::from_fn(|idx| idx);
        ranked.sort_by(|&a, &b| {
            weights[b]
                .partial_cmp(&weights[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (rank, component_idx) in ranked.iter().copied().enumerate() {
            let weight = weights[component_idx];
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
    encode_extracted_observation(
        encoder,
        &hand,
        drawn_tile,
        &open_meld_counts,
        &discards,
        &melds,
        &dora,
        &meta,
        safety,
        search_context,
    )
}

/// Encode a full observation into the fixed-superset tensor.
///
/// This is the main bridge entry point. Extracts all components from
/// a riichienv [`Observation`], feeds them through the encoder pipeline,
/// and returns a reference to the filled fixed-superset observation buffer
/// (`[f32; OBS_SIZE]`, currently `192 x 34`).
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

#[inline]
pub fn extract_observation_facts(
    obs: &Observation,
    drawn_tile: Option<u8>,
) -> ExtractedObservationFacts {
    let hand = extract_hand(obs);
    let discards = extract_discards(obs);
    let melds = extract_melds(obs);
    let open_meld_counts = extract_observer_meld_counts(obs);
    let dora = extract_dora(obs);
    let shanten_batch = batch_discard_shanten(&hand, hand.iter().sum::<u8>() / 3);
    let meta = metadata_from_parts_with_shanten(
        obs.player_id as usize,
        &obs.riichi_declared,
        &obs.scores,
        obs.kyoku_index,
        obs.honba,
        obs.riichi_sticks.min(255) as u8,
        shanten_batch.base,
    );
    ExtractedObservationFacts {
        hand,
        drawn_tile,
        open_meld_counts,
        discards,
        melds,
        dora,
        meta,
        shanten_batch,
    }
}

#[inline]
pub fn encode_extracted_observation_facts_with_profile(
    encoder: &mut ObservationEncoder,
    facts: &ExtractedObservationFacts,
    safety: &SafetyInfo,
    profile: BridgeEncodeProfile,
) -> [f32; OBS_SIZE] {
    let search_context = SearchContext::default();
    encode_extracted_observation_with_profile(
        encoder,
        &facts.hand,
        facts.drawn_tile,
        &facts.open_meld_counts,
        &facts.discards,
        &facts.melds,
        &facts.dora,
        &facts.meta,
        &facts.shanten_batch,
        safety,
        &search_context,
        profile,
    )
}

#[inline]
pub fn encode_observation_with_profile(
    encoder: &mut ObservationEncoder,
    obs: &Observation,
    safety: &SafetyInfo,
    drawn_tile: Option<u8>,
    profile: BridgeEncodeProfile,
) -> [f32; OBS_SIZE] {
    let search_context = SearchContext::default();
    let hand = extract_hand(obs);
    let discards = extract_discards(obs);
    let melds = extract_melds(obs);
    let open_meld_counts = extract_observer_meld_counts(obs);
    let dora = extract_dora(obs);
    let shanten_batch = batch_discard_shanten(&hand, hand.iter().sum::<u8>() / 3);
    let meta = metadata_from_parts_with_shanten(
        obs.player_id as usize,
        &obs.riichi_declared,
        &obs.scores,
        obs.kyoku_index,
        obs.honba,
        obs.riichi_sticks.min(255) as u8,
        shanten_batch.base,
    );
    encode_extracted_observation_with_profile(
        encoder,
        &hand,
        drawn_tile,
        &open_meld_counts,
        &discards,
        &melds,
        &dora,
        &meta,
        &shanten_batch,
        safety,
        &search_context,
        profile,
    )
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
#[inline]
pub fn extract_discards_ref(obs: &ObservationRef<'_>) -> [PlayerDiscards; 4] {
    let observer = obs.player_id as usize;
    std::array::from_fn(|relative_idx| {
        let abs = (observer + relative_idx) % 4;
        let disc = obs.discards[abs];
        let from_hand = obs.tsumogiri_flags[abs];
        let mut pd = PlayerDiscards::new();
        for (turn, &tile136) in disc.iter().enumerate() {
            pd.push(DiscardEntry {
                tile: (tile136 / 4),
                is_tedashi: from_hand.get(turn).copied().unwrap_or(false),
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
    dora_info_from_parts(
        obs.dora_indicators.iter().copied().map(|tile| tile / 4),
        obs.observer_hand.iter().copied(),
    )
}

/// Extract game metadata from an ObservationRef.
#[inline]
pub fn extract_metadata_ref(
    obs: &ObservationRef<'_>,
    hand_counts: &[u8; NUM_TILE_TYPES],
) -> GameMetadata {
    metadata_from_parts(
        obs.player_id as usize,
        &obs.riichi_declared,
        &obs.scores,
        obs.kyoku_index,
        obs.honba,
        obs.riichi_sticks.min(255) as u8,
        hand_counts,
    )
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
    compute_public_hand_ev(hand, discards, melds, dora)
}

#[inline]
pub fn extract_observation_facts_ref(obs: &ObservationRef<'_>) -> ExtractedObservationFacts {
    let hand = extract_hand_ref(obs);
    let discards = extract_discards_ref(obs);
    let melds = extract_melds_ref(obs);
    let open_meld_counts = extract_observer_meld_counts_ref(obs);
    let dora = extract_dora_ref(obs);
    let shanten_batch = batch_discard_shanten(&hand, hand.iter().sum::<u8>() / 3);
    let meta = metadata_from_parts_with_shanten(
        obs.player_id as usize,
        &obs.riichi_declared,
        &obs.scores,
        obs.kyoku_index,
        obs.honba,
        obs.riichi_sticks.min(255) as u8,
        shanten_batch.base,
    );
    ExtractedObservationFacts {
        hand,
        drawn_tile: obs.drawn_tile.map(|t| t / 4),
        open_meld_counts,
        discards,
        melds,
        dora,
        meta,
        shanten_batch,
    }
}

#[inline]
pub fn encode_observation_ref_with_search_context_and_profile(
    encoder: &mut ObservationEncoder,
    obs: &ObservationRef<'_>,
    safety: &SafetyInfo,
    search_context: &SearchContext<'_>,
    profile: BridgeEncodeProfile,
) -> [f32; OBS_SIZE] {
    let facts = extract_observation_facts_ref(obs);
    encode_extracted_observation_with_profile(
        encoder,
        &facts.hand,
        facts.drawn_tile,
        &facts.open_meld_counts,
        &facts.discards,
        &facts.melds,
        &facts.dora,
        &facts.meta,
        &facts.shanten_batch,
        safety,
        search_context,
        profile,
    )
}

#[inline]
pub fn encode_observation_ref_with_search_context(
    encoder: &mut ObservationEncoder,
    obs: &ObservationRef<'_>,
    safety: &SafetyInfo,
    search_context: &SearchContext<'_>,
) -> [f32; OBS_SIZE] {
    encode_observation_ref_with_search_context_and_profile(
        encoder,
        obs,
        safety,
        search_context,
        BridgeEncodeProfile::full(),
    )
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

#[inline]
pub fn encode_observation_ref_with_profile(
    encoder: &mut ObservationEncoder,
    obs: &ObservationRef<'_>,
    safety: &SafetyInfo,
    profile: BridgeEncodeProfile,
) -> [f32; OBS_SIZE] {
    let search_context = SearchContext::default();
    encode_observation_ref_with_search_context_and_profile(
        encoder,
        obs,
        safety,
        &search_context,
        profile,
    )
}

#[cfg(test)]
mod tests;
