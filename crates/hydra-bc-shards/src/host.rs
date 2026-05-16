//! Backend-agnostic host batches and reusable scratch buffers for BC shard rows.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;
use hydra_data_core::sample::SCORE_BINS;

use crate::manifest::{
    BC_BASE_RECORD_SIZE, BELIEF_FIELDS_BYTES, FLAG_BELIEF_FIELDS, FLAG_DELTA_Q, FLAG_EXIT,
    FLAG_MIXTURE_WEIGHTS, FLAG_SAFETY_RESIDUAL, MIXTURE_WEIGHTS_BYTES, OPPONENT_COUNT,
    OPTIONAL_ACTION_FLOAT32_BYTES, OPTIONAL_ACTION_MASK_BYTES, PLAYER_COUNT, SPATIAL_TARGET_SIZE,
};

/// Number of GRP permutation classes.
pub const GRP_CLASS_COUNT: usize = 24;

/// CPU-side host batch ready to cross a thread boundary.
///
/// All expensive shard I/O, parsing, and augmentation is already done. Burn materialization remains
/// in `hydra-train`.
pub struct BcShardHostBatch {
    /// Batch row count.
    pub batch_size: usize,
    /// Flat observation buffer `[batch, OBS_SIZE]`.
    pub obs_flat: Vec<f32>,
    /// Action id per row.
    pub actions: Vec<i64>,
    /// Flat legal mask `[batch, HYDRA_ACTION_SPACE]`.
    pub legal_mask_flat: Vec<f32>,
    /// Scalar value target per row.
    pub value_target: Vec<f32>,
    /// Flat GRP target `[batch, GRP_CLASS_COUNT]`.
    pub grp_target_flat: Vec<f32>,
    /// Flat oracle target `[batch, PLAYER_COUNT]`.
    pub oracle_target_flat: Vec<f32>,
    /// Oracle target presence per row.
    pub oracle_target_mask: Vec<f32>,
    /// Flat tenpai target `[batch, OPPONENT_COUNT]`.
    pub tenpai_flat: Vec<f32>,
    /// Flat danger target `[batch, SPATIAL_TARGET_SIZE]`.
    pub danger_flat: Vec<f32>,
    /// Flat danger-mask target `[batch, SPATIAL_TARGET_SIZE]`.
    pub danger_mask_flat: Vec<f32>,
    /// Flat opponent-next target `[batch, SPATIAL_TARGET_SIZE]`.
    pub opp_next_flat: Vec<f32>,
    /// Flat score PDF target `[batch, SCORE_BINS]`.
    pub score_pdf_flat: Vec<f32>,
    /// Flat score CDF target `[batch, SCORE_BINS]`.
    pub score_cdf_flat: Vec<f32>,
    /// Optional safety residual action target.
    pub safety_target_flat: Option<Vec<f32>>,
    /// Optional safety residual mask.
    pub safety_mask_flat: Option<Vec<f32>>,
    /// Optional ExIt action target.
    pub exit_target_flat: Option<Vec<f32>>,
    /// Optional ExIt action mask.
    pub exit_mask_flat: Option<Vec<f32>>,
    /// Optional delta-Q action target.
    pub delta_q_target_flat: Option<Vec<f32>>,
    /// Optional delta-Q action mask.
    pub delta_q_mask_flat: Option<Vec<f32>>,
}

// SAFETY: all fields are plain vecs of Copy types -- trivially Send + Sync.
unsafe impl Send for BcShardHostBatch {}
unsafe impl Sync for BcShardHostBatch {}

impl BcShardHostBatch {
    /// Creates an empty host batch.
    pub fn empty() -> Self {
        Self {
            batch_size: 0,
            obs_flat: Vec::new(),
            actions: Vec::new(),
            legal_mask_flat: Vec::new(),
            value_target: Vec::new(),
            grp_target_flat: Vec::new(),
            oracle_target_flat: Vec::new(),
            oracle_target_mask: Vec::new(),
            tenpai_flat: Vec::new(),
            danger_flat: Vec::new(),
            danger_mask_flat: Vec::new(),
            opp_next_flat: Vec::new(),
            score_pdf_flat: Vec::new(),
            score_cdf_flat: Vec::new(),
            safety_target_flat: None,
            safety_mask_flat: None,
            exit_target_flat: None,
            exit_mask_flat: None,
            delta_q_target_flat: None,
            delta_q_mask_flat: None,
        }
    }
}

/// Reusable scratch buffers for the BC shard producer path.
///
/// Mirrors every field in [`BcShardHostBatch`] but is designed to be reset and refilled across
/// batches without reallocating.
pub struct BcShardHostScratch {
    /// Batch row count.
    pub batch_size: usize,
    /// Flat observation scratch.
    pub obs_flat: Vec<f32>,
    /// Action scratch.
    pub actions: Vec<i64>,
    /// Legal-mask scratch.
    pub legal_mask_flat: Vec<f32>,
    /// Value-target scratch.
    pub value_target: Vec<f32>,
    /// GRP-target scratch.
    pub grp_target_flat: Vec<f32>,
    /// Oracle-target scratch.
    pub oracle_target_flat: Vec<f32>,
    /// Oracle-mask scratch.
    pub oracle_target_mask: Vec<f32>,
    /// Tenpai scratch.
    pub tenpai_flat: Vec<f32>,
    /// Danger scratch.
    pub danger_flat: Vec<f32>,
    /// Danger-mask scratch.
    pub danger_mask_flat: Vec<f32>,
    /// Opponent-next scratch.
    pub opp_next_flat: Vec<f32>,
    /// Score PDF scratch.
    pub score_pdf_flat: Vec<f32>,
    /// Score CDF scratch.
    pub score_cdf_flat: Vec<f32>,
    /// Optional safety residual scratch.
    pub safety_target_flat: Option<Vec<f32>>,
    /// Optional safety mask scratch.
    pub safety_mask_flat: Option<Vec<f32>>,
    /// Optional ExIt target scratch.
    pub exit_target_flat: Option<Vec<f32>>,
    /// Optional ExIt mask scratch.
    pub exit_mask_flat: Option<Vec<f32>>,
    /// Optional delta-Q target scratch.
    pub delta_q_target_flat: Option<Vec<f32>>,
    /// Optional delta-Q mask scratch.
    pub delta_q_mask_flat: Option<Vec<f32>>,
}

// SAFETY: same plain-vec-of-Copy argument as BcShardHostBatch.
unsafe impl Send for BcShardHostScratch {}

impl BcShardHostScratch {
    /// Pre-allocate scratch buffers for a given batch size and feature flags.
    pub fn new(batch_size: usize, need_safety: bool, need_exit: bool, need_delta_q: bool) -> Self {
        let action_space = HYDRA_ACTION_SPACE;
        Self {
            batch_size,
            obs_flat: vec![0.0f32; batch_size * OBS_SIZE],
            actions: vec![0i64; batch_size],
            legal_mask_flat: vec![0.0f32; batch_size * action_space],
            value_target: vec![0.0f32; batch_size],
            grp_target_flat: vec![0.0f32; batch_size * GRP_CLASS_COUNT],
            oracle_target_flat: vec![0.0f32; batch_size * PLAYER_COUNT],
            oracle_target_mask: vec![0.0f32; batch_size],
            tenpai_flat: vec![0.0f32; batch_size * OPPONENT_COUNT],
            danger_flat: vec![0.0f32; batch_size * SPATIAL_TARGET_SIZE],
            danger_mask_flat: vec![0.0f32; batch_size * SPATIAL_TARGET_SIZE],
            opp_next_flat: vec![0.0f32; batch_size * SPATIAL_TARGET_SIZE],
            score_pdf_flat: vec![0.0f32; batch_size * SCORE_BINS],
            score_cdf_flat: vec![0.0f32; batch_size * SCORE_BINS],
            safety_target_flat: need_safety.then(|| vec![0.0f32; batch_size * action_space]),
            safety_mask_flat: need_safety.then(|| vec![0.0f32; batch_size * action_space]),
            exit_target_flat: need_exit.then(|| vec![0.0f32; batch_size * action_space]),
            exit_mask_flat: need_exit.then(|| vec![0.0f32; batch_size * action_space]),
            delta_q_target_flat: need_delta_q.then(|| vec![0.0f32; batch_size * action_space]),
            delta_q_mask_flat: need_delta_q.then(|| vec![0.0f32; batch_size * action_space]),
        }
    }

    /// Zero-fill all buffers and resize to `batch_size` without deallocating when possible.
    pub fn reset(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
        resize_uninit_f32(&mut self.obs_flat, batch_size * OBS_SIZE);
        resize_uninit_i64(&mut self.actions, batch_size);
        resize_uninit_f32(&mut self.legal_mask_flat, batch_size * HYDRA_ACTION_SPACE);
        resize_uninit_f32(&mut self.value_target, batch_size);
        resize_zeroed(&mut self.grp_target_flat, batch_size * GRP_CLASS_COUNT);
        resize_uninit_f32(&mut self.oracle_target_flat, batch_size * PLAYER_COUNT);
        resize_zeroed(&mut self.oracle_target_mask, batch_size);
        resize_uninit_f32(&mut self.tenpai_flat, batch_size * OPPONENT_COUNT);
        resize_uninit_f32(&mut self.danger_flat, batch_size * SPATIAL_TARGET_SIZE);
        resize_uninit_f32(&mut self.danger_mask_flat, batch_size * SPATIAL_TARGET_SIZE);
        resize_zeroed(&mut self.opp_next_flat, batch_size * SPATIAL_TARGET_SIZE);
        resize_zeroed(&mut self.score_pdf_flat, batch_size * SCORE_BINS);
        resize_zeroed(&mut self.score_cdf_flat, batch_size * SCORE_BINS);
        if let Some(buf) = self.safety_target_flat.as_mut() {
            resize_zeroed(buf, batch_size * HYDRA_ACTION_SPACE);
        }
        if let Some(buf) = self.safety_mask_flat.as_mut() {
            resize_zeroed(buf, batch_size * HYDRA_ACTION_SPACE);
        }
        if let Some(buf) = self.exit_target_flat.as_mut() {
            resize_zeroed(buf, batch_size * HYDRA_ACTION_SPACE);
        }
        if let Some(buf) = self.exit_mask_flat.as_mut() {
            resize_zeroed(buf, batch_size * HYDRA_ACTION_SPACE);
        }
        if let Some(buf) = self.delta_q_target_flat.as_mut() {
            resize_zeroed(buf, batch_size * HYDRA_ACTION_SPACE);
        }
        if let Some(buf) = self.delta_q_mask_flat.as_mut() {
            resize_zeroed(buf, batch_size * HYDRA_ACTION_SPACE);
        }
    }

    /// Swap the filled buffers out of this scratch into an owned [`BcShardHostBatch`].
    pub fn take_batch(&mut self) -> BcShardHostBatch {
        BcShardHostBatch {
            batch_size: self.batch_size,
            obs_flat: std::mem::take(&mut self.obs_flat),
            actions: std::mem::take(&mut self.actions),
            legal_mask_flat: std::mem::take(&mut self.legal_mask_flat),
            value_target: std::mem::take(&mut self.value_target),
            grp_target_flat: std::mem::take(&mut self.grp_target_flat),
            oracle_target_flat: std::mem::take(&mut self.oracle_target_flat),
            oracle_target_mask: std::mem::take(&mut self.oracle_target_mask),
            tenpai_flat: std::mem::take(&mut self.tenpai_flat),
            danger_flat: std::mem::take(&mut self.danger_flat),
            danger_mask_flat: std::mem::take(&mut self.danger_mask_flat),
            opp_next_flat: std::mem::take(&mut self.opp_next_flat),
            score_pdf_flat: std::mem::take(&mut self.score_pdf_flat),
            score_cdf_flat: std::mem::take(&mut self.score_cdf_flat),
            safety_target_flat: self.safety_target_flat.as_mut().map(std::mem::take),
            safety_mask_flat: self.safety_mask_flat.as_mut().map(std::mem::take),
            exit_target_flat: self.exit_target_flat.as_mut().map(std::mem::take),
            exit_mask_flat: self.exit_mask_flat.as_mut().map(std::mem::take),
            delta_q_target_flat: self.delta_q_target_flat.as_mut().map(std::mem::take),
            delta_q_mask_flat: self.delta_q_mask_flat.as_mut().map(std::mem::take),
        }
    }

    /// Extract a batch while recycling a previously-consumed batch's heap allocations back into the scratch.
    pub fn swap_batch(&mut self, recycled: &mut BcShardHostBatch) -> BcShardHostBatch {
        std::mem::swap(&mut self.obs_flat, &mut recycled.obs_flat);
        std::mem::swap(&mut self.actions, &mut recycled.actions);
        std::mem::swap(&mut self.legal_mask_flat, &mut recycled.legal_mask_flat);
        std::mem::swap(&mut self.value_target, &mut recycled.value_target);
        std::mem::swap(&mut self.grp_target_flat, &mut recycled.grp_target_flat);
        std::mem::swap(
            &mut self.oracle_target_flat,
            &mut recycled.oracle_target_flat,
        );
        std::mem::swap(
            &mut self.oracle_target_mask,
            &mut recycled.oracle_target_mask,
        );
        std::mem::swap(&mut self.tenpai_flat, &mut recycled.tenpai_flat);
        std::mem::swap(&mut self.danger_flat, &mut recycled.danger_flat);
        std::mem::swap(&mut self.danger_mask_flat, &mut recycled.danger_mask_flat);
        std::mem::swap(&mut self.opp_next_flat, &mut recycled.opp_next_flat);
        std::mem::swap(&mut self.score_pdf_flat, &mut recycled.score_pdf_flat);
        std::mem::swap(&mut self.score_cdf_flat, &mut recycled.score_cdf_flat);
        if let (Some(s), Some(r)) = (
            self.safety_target_flat.as_mut(),
            recycled.safety_target_flat.as_mut(),
        ) {
            std::mem::swap(s, r);
        }
        if let (Some(s), Some(r)) = (
            self.safety_mask_flat.as_mut(),
            recycled.safety_mask_flat.as_mut(),
        ) {
            std::mem::swap(s, r);
        }
        if let (Some(s), Some(r)) = (
            self.exit_target_flat.as_mut(),
            recycled.exit_target_flat.as_mut(),
        ) {
            std::mem::swap(s, r);
        }
        if let (Some(s), Some(r)) = (
            self.exit_mask_flat.as_mut(),
            recycled.exit_mask_flat.as_mut(),
        ) {
            std::mem::swap(s, r);
        }
        if let (Some(s), Some(r)) = (
            self.delta_q_target_flat.as_mut(),
            recycled.delta_q_target_flat.as_mut(),
        ) {
            std::mem::swap(s, r);
        }
        if let (Some(s), Some(r)) = (
            self.delta_q_mask_flat.as_mut(),
            recycled.delta_q_mask_flat.as_mut(),
        ) {
            std::mem::swap(s, r);
        }
        recycled.batch_size = self.batch_size;

        let mut out = BcShardHostBatch::empty();
        std::mem::swap(&mut out, recycled);
        out
    }
}

/// Returns the exact BC record byte size for feature flags.
pub const fn record_size_for_flags(flags: u32) -> u32 {
    let mut size = BC_BASE_RECORD_SIZE;
    if flags & FLAG_SAFETY_RESIDUAL != 0 {
        size += OPTIONAL_ACTION_FLOAT32_BYTES as u32 + OPTIONAL_ACTION_MASK_BYTES as u32;
    }
    if flags & FLAG_EXIT != 0 {
        size += OPTIONAL_ACTION_FLOAT32_BYTES as u32 + OPTIONAL_ACTION_MASK_BYTES as u32;
    }
    if flags & FLAG_DELTA_Q != 0 {
        size += OPTIONAL_ACTION_FLOAT32_BYTES as u32 + OPTIONAL_ACTION_MASK_BYTES as u32;
    }
    if flags & FLAG_BELIEF_FIELDS != 0 {
        size += BELIEF_FIELDS_BYTES as u32;
    }
    if flags & FLAG_MIXTURE_WEIGHTS != 0 {
        size += MIXTURE_WEIGHTS_BYTES as u32;
    }
    size
}

#[inline]
fn resize_zeroed(buf: &mut Vec<f32>, len: usize) {
    buf.clear();
    buf.resize(len, 0.0);
}

#[inline]
fn resize_uninit_f32(buf: &mut Vec<f32>, len: usize) {
    buf.clear();
    if buf.capacity() < len {
        buf.reserve(len);
    }
    // SAFETY: capacity >= len after the branch above; caller writes all elements.
    unsafe { buf.set_len(len) };
}

#[inline]
fn resize_uninit_i64(buf: &mut Vec<i64>, len: usize) {
    buf.clear();
    if buf.capacity() < len {
        buf.reserve(len);
    }
    // SAFETY: capacity >= len after the branch above; caller writes all elements.
    unsafe { buf.set_len(len) };
}
