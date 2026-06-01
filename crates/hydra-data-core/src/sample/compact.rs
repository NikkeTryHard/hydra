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
