//! Pure MJAI sample DTO and score target helpers.

mod compact;
mod placement;
mod score;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;

pub use compact::{
    COMPACT_ADVANCED_TAIL_LEN, COMPACT_BASELINE_CHANNELS, COMPACT_MISSING_SHANTEN,
    COMPACT_MISSING_TILE, CompactDiscardEntry, CompactMeldInfo, CompactMeldType,
    CompactObservationFacts, CompactPlayerDiscards, CompactPlayerMelds, CompactSafetyFacts,
};
pub use placement::{GRP_PERM_TABLE, score_to_placement, score_to_placements, scores_to_grp_index};
pub use score::{
    SCORE_BINS, score_delta_to_bin, score_delta_to_cdf, score_delta_to_pdf, score_delta_to_value,
};

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

/// Builds a one-hot action vector with out-of-range actions left all zero.
pub fn one_hot_action(action: u8, num_classes: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; num_classes];
    if (action as usize) < num_classes {
        v[action as usize] = 1.0;
    }
    v
}
