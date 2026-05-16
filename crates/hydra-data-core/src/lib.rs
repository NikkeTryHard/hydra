//! Pure data sample contracts and scoring helpers.

pub mod manifest;

pub mod sample;

pub use manifest::{
    DataManifest, DataSource, DiscoveryManifest, DiscoveryMode, DiscoverySummary, GameLocator,
    SourceFilterConfig,
};
pub use sample::{
    COMPACT_ADVANCED_TAIL_LEN, COMPACT_BASELINE_CHANNELS, COMPACT_MISSING_SHANTEN,
    COMPACT_MISSING_TILE, CompactDiscardEntry, CompactMeldInfo, CompactMeldType,
    CompactObservationFacts, CompactPlayerDiscards, CompactPlayerMelds, CompactSafetyFacts,
    GRP_PERM_TABLE, MjaiSample, SCORE_BINS, one_hot_action, score_delta_to_bin, score_delta_to_cdf,
    score_delta_to_pdf, score_delta_to_value, score_to_placement, score_to_placements,
    scores_to_grp_index,
};
