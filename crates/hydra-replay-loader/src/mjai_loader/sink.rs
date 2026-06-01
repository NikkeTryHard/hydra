use super::*;

/// Decoded training sample passed to streaming replay sinks.
pub struct ReplaySampleRecord {
    /// Encoded observation planes flattened as `[NUM_CHANNELS * 34]`.
    pub obs: [f32; OBS_SIZE],
    /// Replay-derived compact facts for shard storage.
    pub compact_facts: CompactObservationFacts,
    /// Hydra action id in the 46-action policy space.
    pub action: u8,
    /// Legal-action mask over the Hydra policy space.
    pub legal_mask: [f32; HYDRA_ACTION_SPACE],
    /// Final placement for the acting player.
    pub placement: u8,
    /// Final score delta for the acting player.
    pub score_delta: i32,
    /// Global rank permutation class label.
    pub grp_label: u8,
    /// Optional oracle policy distribution.
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

impl ReplaySampleRecord {
    pub(super) fn into_sample(self) -> MjaiSample {
        MjaiSample {
            obs: self.obs,
            compact_facts: Some(self.compact_facts),
            action: self.action,
            legal_mask: self.legal_mask,
            placement: self.placement,
            score_delta: self.score_delta,
            grp_label: self.grp_label,
            oracle_target: self.oracle_target,
            tenpai: self.tenpai,
            opp_next: self.opp_next,
            danger: self.danger,
            danger_mask: self.danger_mask,
            safety_residual: self.safety_residual,
            safety_residual_mask: self.safety_residual_mask,
            exit_target: self.exit_target,
            exit_mask: self.exit_mask,
            delta_q_target: self.delta_q_target,
            delta_q_mask: self.delta_q_mask,
            belief_fields: self.belief_fields,
            mixture_weights: self.mixture_weights,
            belief_fields_present: self.belief_fields_present,
            mixture_weights_present: self.mixture_weights_present,
        }
    }
}

/// Streaming destination for replay materialization.
pub trait ReplaySampleSink {
    /// Accepts one decoded sample in replay order.
    fn push_sample(&mut self, sample: ReplaySampleRecord) -> io::Result<()>;
}

pub(super) struct VecReplaySampleSink {
    pub(super) samples: Vec<MjaiSample>,
}

impl VecReplaySampleSink {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
        }
    }
}

impl ReplaySampleSink for VecReplaySampleSink {
    fn push_sample(&mut self, sample: ReplaySampleRecord) -> io::Result<()> {
        self.samples.push(sample.into_sample());
        Ok(())
    }
}
