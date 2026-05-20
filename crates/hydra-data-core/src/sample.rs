//! Pure MJAI sample DTO and score target helpers.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{
    DoraInfo, GameMetadata, MeldType, OBS_SIZE, PlayerDiscards, PlayerMelds,
};
use hydra_core::safety::SafetyInfo;

/// Sentinel used when a compact observation fact has no tile value.
pub const COMPACT_MISSING_TILE: u8 = 255;
/// Sentinel used when a discard shanten result is unavailable for a tile.
pub const COMPACT_MISSING_SHANTEN: i8 = 127;
/// Number of baseline planes that compact replay facts can reconstruct losslessly.
pub const COMPACT_BASELINE_CHANNELS: usize = 85;
/// Number of non-baseline planes in the current fixed-superset observation.
pub const COMPACT_ADVANCED_TAIL_LEN: usize = OBS_SIZE - COMPACT_BASELINE_CHANNELS * 34;

/// Compact replay-derived observation facts for lossless shard storage.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactObservationFacts {
    pub hand_counts: [u8; 34],
    pub open_meld_counts: [u8; 34],
    pub drawn_tile: u8,
    pub shanten_base: i8,
    pub shanten_discard: [i8; 34],
    pub discards: [CompactPlayerDiscards; 4],
    pub melds: [CompactPlayerMelds; 4],
    pub dora_indicators: [u8; 5],
    pub dora_indicator_count: u8,
    pub aka_flags: [bool; 3],
    pub riichi: [bool; 4],
    pub scores: [i32; 4],
    pub kyoku_index: u8,
    pub honba: u8,
    pub kyotaku: u8,
    pub safety: CompactSafetyFacts,
    /// Exact non-baseline planes (channels 85..192) when runtime/search features are present.
    pub advanced_tail: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactDiscardEntry {
    pub tile: u8,
    pub is_tedashi: bool,
    pub turn: u16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactPlayerDiscards {
    pub discards: [CompactDiscardEntry; 30],
    pub len: u8,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompactMeldType {
    #[default]
    Chi,
    Pon,
    Kan,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactMeldInfo {
    pub tiles: [u8; 4],
    pub tile_count: u8,
    pub meld_type: CompactMeldType,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactPlayerMelds {
    pub melds: [CompactMeldInfo; 4],
    pub len: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompactSafetyFacts {
    pub genbutsu_all: [u64; 3],
    pub genbutsu_tedashi: [u64; 3],
    pub genbutsu_riichi_era: [u64; 3],
    pub suji: [[f32; 34]; 3],
    pub half_suji: [u64; 3],
    pub matagi: [[f32; 34]; 3],
    pub kabe: u64,
    pub one_chance: u64,
    pub visible_counts: [u8; 34],
    pub opponent_riichi: [bool; 3],
    pub cached_tenpai_prob: [f32; 3],
}

impl CompactObservationFacts {
    #[allow(clippy::too_many_arguments, reason = "DTO mirrors encoder inputs")]
    pub fn from_encoder_inputs(
        hand_counts: [u8; 34],
        open_meld_counts: [u8; 34],
        drawn_tile: Option<u8>,
        shanten_base: i8,
        shanten_discard: [Option<i8>; 34],
        discards: &[PlayerDiscards; 4],
        melds: &[PlayerMelds; 4],
        dora: &DoraInfo,
        meta: &GameMetadata,
        safety: &SafetyInfo,
        obs: &[f32; OBS_SIZE],
        preserve_advanced_tail: bool,
    ) -> Self {
        let mut compact_shanten = [COMPACT_MISSING_SHANTEN; 34];
        for (dst, src) in compact_shanten.iter_mut().zip(shanten_discard) {
            if let Some(value) = src {
                *dst = value;
            }
        }
        let advanced_tail =
            preserve_advanced_tail.then(|| obs[COMPACT_BASELINE_CHANNELS * 34..].to_vec());
        Self {
            hand_counts,
            open_meld_counts,
            drawn_tile: drawn_tile.unwrap_or(COMPACT_MISSING_TILE),
            shanten_base,
            shanten_discard: compact_shanten,
            discards: compact_discards(discards),
            melds: compact_melds(melds),
            dora_indicators: dora.indicators,
            dora_indicator_count: dora.indicator_count,
            aka_flags: dora.aka_flags,
            riichi: meta.riichi,
            scores: meta.scores,
            kyoku_index: meta.kyoku_index,
            honba: meta.honba,
            kyotaku: meta.kyotaku,
            safety: CompactSafetyFacts::from(safety),
            advanced_tail,
        }
    }
}

fn compact_discards(discards: &[PlayerDiscards; 4]) -> [CompactPlayerDiscards; 4] {
    std::array::from_fn(|player| {
        let src = &discards[player];
        let mut dst = CompactPlayerDiscards {
            len: src.len,
            ..Default::default()
        };
        for (dst_entry, src_entry) in dst.discards.iter_mut().zip(src.discards) {
            *dst_entry = CompactDiscardEntry {
                tile: src_entry.tile,
                is_tedashi: src_entry.is_tedashi,
                turn: src_entry.turn,
            };
        }
        dst
    })
}

fn compact_melds(melds: &[PlayerMelds; 4]) -> [CompactPlayerMelds; 4] {
    std::array::from_fn(|player| {
        let src = &melds[player];
        let mut dst = CompactPlayerMelds {
            len: src.len,
            ..Default::default()
        };
        for (dst_meld, src_meld) in dst.melds.iter_mut().zip(src.melds) {
            let meld_type = match src_meld.meld_type {
                MeldType::Chi => CompactMeldType::Chi,
                MeldType::Pon => CompactMeldType::Pon,
                MeldType::Kan => CompactMeldType::Kan,
            };
            *dst_meld = CompactMeldInfo {
                tiles: src_meld.tiles,
                tile_count: src_meld.tile_count,
                meld_type,
            };
        }
        dst
    })
}

impl From<&SafetyInfo> for CompactSafetyFacts {
    fn from(safety: &SafetyInfo) -> Self {
        Self {
            genbutsu_all: safety.genbutsu_all,
            genbutsu_tedashi: safety.genbutsu_tedashi,
            genbutsu_riichi_era: safety.genbutsu_riichi_era,
            suji: safety.suji,
            half_suji: safety.half_suji,
            matagi: safety.matagi,
            kabe: safety.kabe,
            one_chance: safety.one_chance,
            visible_counts: safety.visible_counts,
            opponent_riichi: safety.opponent_riichi,
            cached_tenpai_prob: safety.cached_tenpai_prob,
        }
    }
}

/// One encoded MJAI decision sample plus optional auxiliary targets.
#[derive(Clone)]
pub struct MjaiSample {
    /// Encoded observation planes flattened as `[NUM_CHANNELS * 34]`.
    pub obs: [f32; OBS_SIZE],
    /// Replay-derived compact facts for shard storage; real replay samples populate this.
    pub compact_facts: Option<CompactObservationFacts>,
    /// Hydra action id in the 46-action policy space.
    pub action: u8,
    /// Legal-action mask over the Hydra policy space.
    pub legal_mask: [f32; HYDRA_ACTION_SPACE],
    /// Final placement for the acting player, where `0` is first place.
    pub placement: u8,
    /// Final score delta for the acting player.
    pub score_delta: i32,
    /// Global rank permutation class label.
    pub grp_label: u8,
    /// Optional oracle policy distribution over four coarse choices.
    pub oracle_target: Option<[f32; 4]>,
    /// Opponent tenpai targets in seat order.
    pub tenpai: [f32; 3],
    /// Opponent next-danger tile ids in seat order, or sentinel values.
    pub opp_next: [u8; 3],
    /// Opponent/tile danger targets flattened as `3 * 34`.
    pub danger: [f32; 102],
    /// Mask for `danger`.
    pub danger_mask: [f32; 102],
    /// Optional safety residual target over actions.
    pub safety_residual: Option<[f32; HYDRA_ACTION_SPACE]>,
    /// Optional mask for `safety_residual`.
    pub safety_residual_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
    /// Optional ExIt target over actions.
    pub exit_target: Option<[f32; HYDRA_ACTION_SPACE]>,
    /// Optional mask for `exit_target`.
    pub exit_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
    /// Optional delta-Q target over actions.
    pub delta_q_target: Option<[f32; HYDRA_ACTION_SPACE]>,
    /// Optional mask for `delta_q_target`.
    pub delta_q_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
    /// Optional belief targets flattened as `16 * 34`.
    pub belief_fields: Option<[f32; 16 * 34]>,
    /// Optional mixture weights for belief supervision.
    pub mixture_weights: Option<[f32; 4]>,
    /// Whether belief-field supervision is present.
    pub belief_fields_present: bool,
    /// Whether mixture-weight supervision is present.
    pub mixture_weights_present: bool,
}

/// Minimum score delta represented by binned score targets.
const SCORE_BIN_MIN: f32 = -50000.0;
/// Maximum score delta represented by binned score targets.
const SCORE_BIN_MAX: f32 = 60000.0;
/// Number of score-delta bins.
pub const SCORE_BINS: usize = 64;

/// Converts final scores into the global rank permutation class.
pub fn scores_to_grp_index(scores: [i32; 4]) -> Result<u8, &'static str> {
    let mut indexed: [(i32, u8); 4] = [
        (scores[0], 0),
        (scores[1], 1),
        (scores[2], 2),
        (scores[3], 3),
    ];
    indexed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let ranking = [indexed[0].1, indexed[1].1, indexed[2].1, indexed[3].1];
    GRP_PERM_TABLE
        .iter()
        .position(|p| *p == ranking)
        .map(|i| i as u8)
        .ok_or("invalid ranking permutation")
}

/// Table of all four-player rank permutations.
pub const GRP_PERM_TABLE: [[u8; 4]; 24] = generate_perm_table();

const fn generate_perm_table() -> [[u8; 4]; 24] {
    let mut table = [[0u8; 4]; 24];
    let mut idx = 0;
    let mut a = 0u8;
    while a < 4 {
        let mut b = 0u8;
        while b < 4 {
            if b != a {
                let mut c = 0u8;
                while c < 4 {
                    if c != a && c != b {
                        let d = 6 - a - b - c;
                        table[idx] = [a, b, c, d];
                        idx += 1;
                    }
                    c += 1;
                }
            }
            b += 1;
        }
        a += 1;
    }
    table
}

/// Converts a score delta into a clamped score-bin index.
#[inline]
pub fn score_delta_to_bin(score_delta: i32) -> usize {
    const RANGE_INV: f32 = 1.0 / (SCORE_BIN_MAX - SCORE_BIN_MIN);
    let normalized = (score_delta as f32 - SCORE_BIN_MIN) * RANGE_INV;
    let bin = (normalized * SCORE_BINS as f32) as usize;
    bin.min(SCORE_BINS - 1)
}

/// Converts a score delta into the normalized scalar value target.
#[inline]
pub fn score_delta_to_value(score_delta: i32) -> f32 {
    const INV_100K: f32 = 1.0 / 100_000.0;
    (score_delta as f32 * INV_100K).clamp(-1.0, 1.0)
}

/// Converts a score delta into a one-hot score-bin PDF target.
pub fn score_delta_to_pdf(score_delta: i32) -> [f32; SCORE_BINS] {
    let mut pdf = [0.0f32; SCORE_BINS];
    pdf[score_delta_to_bin(score_delta)] = 1.0;
    pdf
}

/// Converts a score delta into a cumulative score-bin CDF target.
pub fn score_delta_to_cdf(score_delta: i32) -> [f32; SCORE_BINS] {
    let bin = score_delta_to_bin(score_delta);
    let mut cdf = [0.0f32; SCORE_BINS];
    for v in &mut cdf[bin..] {
        *v = 1.0;
    }
    cdf
}

/// Returns one player's final placement, where `0` is first place.
///
/// Returns `None` when `player` is outside the four-player score array.
pub fn score_to_placement(scores: [i32; 4], player: u8) -> Option<u8> {
    score_to_placements(scores).get(player as usize).copied()
}

/// Returns final placements for all players, where `0` is first place.
pub fn score_to_placements(scores: [i32; 4]) -> [u8; 4] {
    let mut indexed: [(i32, u8); 4] = [
        (scores[0], 0),
        (scores[1], 1),
        (scores[2], 2),
        (scores[3], 3),
    ];
    indexed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut placements = [3u8; 4];
    for (placement, (_, player)) in indexed.into_iter().enumerate() {
        placements[player as usize] = placement as u8;
    }
    placements
}

/// Builds a one-hot action vector with out-of-range actions left all zero.
pub fn one_hot_action(action: u8, num_classes: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; num_classes];
    if (action as usize) < num_classes {
        v[action as usize] = 1.0;
    }
    v
}
