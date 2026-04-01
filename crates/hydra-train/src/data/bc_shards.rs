use std::fs;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

use burn::prelude::*;
use burn::tensor::backend::Backend;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use memmap2::{Advice, Mmap};
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::data::augment::{
    augment_action_vector_suit_into,
    augment_obs_suit_from_le_bytes, permutation_tables,
};
use crate::data::mjai_loader::{
    MjaiGame, SidecarProvenance, invalid_data, load_game_from_path, load_game_from_path_with_sidecar,
    load_game_from_stream, load_game_from_stream_with_sidecar,
};
use crate::data::pipeline::{DataSource, is_train_game, scan_data_sources};
use crate::data::sample::{MjaiBcBatch, score_delta_to_bin, score_delta_to_value};
use crate::training::head_gates::{AdvancedHead, TargetPresence};
use crate::training::losses::HydraTargets;
use crate::training::replay_delta_q::DeltaQSidecarIndex;
use crate::training::replay_exit::ExitSidecarIndex;

const OPPONENT_COUNT: usize = 3;
const PLAYER_COUNT: usize = 4;
const TILE_COUNT: usize = 34;
const SPATIAL_TARGET_SIZE: usize = OPPONENT_COUNT * TILE_COUNT;
const GRP_CLASS_COUNT: usize = 24;
const SCORE_BINS: usize = 64;

pub const BC_SHARD_MAGIC: [u8; 8] = *b"HYBCS2\0\0";
pub const BC_SHARD_VERSION: u32 = 2;
pub const BC_SHARD_MANIFEST_VERSION: u32 = 2;
pub const BC_SHARD_HEADER_SIZE: u32 = 80;

pub const FLAG_SAFETY_RESIDUAL: u32 = 1 << 0;
pub const FLAG_EXIT: u32 = 1 << 1;
pub const FLAG_DELTA_Q: u32 = 1 << 2;

pub const OBS_F32_BYTES: usize = OBS_SIZE * 4;
pub const LEGAL_MASK_BYTES: usize = HYDRA_ACTION_SPACE;
pub const ORACLE_FLOAT32_BYTES: usize = PLAYER_COUNT * 4;
pub const ORACLE_MASK_BYTES: usize = 1;
pub const TENPAI_BYTES: usize = OPPONENT_COUNT;
pub const OPP_NEXT_BYTES: usize = OPPONENT_COUNT;
pub const DANGER_BYTES: usize = SPATIAL_TARGET_SIZE;
pub const DANGER_MASK_BYTES: usize = SPATIAL_TARGET_SIZE;
pub const OPTIONAL_ACTION_FLOAT32_BYTES: usize = HYDRA_ACTION_SPACE * 4;
pub const OPTIONAL_ACTION_MASK_BYTES: usize = HYDRA_ACTION_SPACE;

pub const BC_BASE_RECORD_SIZE: u32 = (OBS_F32_BYTES
    + 1
    + LEGAL_MASK_BYTES
    + 4
    + 1
    + ORACLE_FLOAT32_BYTES
    + ORACLE_MASK_BYTES
    + TENPAI_BYTES
    + OPP_NEXT_BYTES
    + DANGER_BYTES
    + DANGER_MASK_BYTES) as u32;

pub const BC_RECORD_SIZE_WITH_ALL_OPTIONALS: u32 = BC_BASE_RECORD_SIZE
    + (OPTIONAL_ACTION_FLOAT32_BYTES as u32 + OPTIONAL_ACTION_MASK_BYTES as u32) * 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BcShardSplit {
    Train,
    Validation,
}

impl BcShardSplit {
    pub const fn shard_prefix(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "val",
        }
    }

    pub const fn split_id(self) -> u32 {
        match self {
            Self::Train => 0,
            Self::Validation => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BcShardSplitMode {
    Both,
    Train,
    Validation,
}

impl BcShardSplitMode {
    pub const fn includes(self, split: BcShardSplit) -> bool {
        match (self, split) {
            (Self::Both, _) => true,
            (Self::Train, BcShardSplit::Train) => true,
            (Self::Validation, BcShardSplit::Validation) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardSidecarManifest {
    pub path: String,
    pub source_net_hash: u64,
    pub source_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardDescriptor {
    pub split: BcShardSplit,
    pub shard_index: usize,
    pub file_name: String,
    pub sample_count: u64,
    pub first_sample_index: u64,
    pub byte_len: u64,
    pub feature_flags: u32,
    pub record_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardSplitManifest {
    pub split: BcShardSplit,
    pub shard_count: usize,
    pub sample_count: u64,
    pub feature_flags: u32,
    pub record_size: u32,
    pub shards: Vec<BcShardDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BcShardBuildTotals {
    pub sample_count: u64,
    pub skipped_games: u64,
    pub empty_games: u64,
    pub shard_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BcShardManifest {
    pub manifest_version: u32,
    pub shard_version: u32,
    pub shard_header_size: u32,
    pub base_record_size: u32,
    pub max_record_size: u32,
    pub obs_size: usize,
    pub num_channels: usize,
    pub action_space: usize,
    pub train_fraction: f32,
    pub shard_samples: usize,
    pub augment_runtime: bool,
    pub input: String,
    pub output_dir: String,
    pub created_at: String,
    pub source_count: usize,
    pub source_total_games_hint: usize,
    pub source_train_count_hint: usize,
    pub source_val_count_hint: usize,
    pub source_counts_exact: bool,
    pub exit_sidecar: Option<BcShardSidecarManifest>,
    pub delta_q_sidecar: Option<BcShardSidecarManifest>,
    pub totals: BcShardBuildTotals,
    pub splits: Vec<BcShardSplitManifest>,
}

#[derive(Debug, Clone)]
pub struct BuildBcShardsConfig {
    pub input: PathBuf,
    pub output_dir: PathBuf,
    pub manifest_name: String,
    pub train_fraction: f32,
    pub shard_samples: usize,
    pub split_mode: BcShardSplitMode,
    pub exit_sidecar: Option<Arc<ExitSidecarIndex>>,
    pub exit_sidecar_path: Option<PathBuf>,
    pub exit_provenance: SidecarProvenance,
    pub delta_q_sidecar: Option<Arc<DeltaQSidecarIndex>>,
    pub delta_q_sidecar_path: Option<PathBuf>,
    pub delta_q_provenance: SidecarProvenance,
}

impl Default for BuildBcShardsConfig {
    fn default() -> Self {
        Self {
            input: PathBuf::from("."),
            output_dir: PathBuf::from("bc_shards"),
            manifest_name: "bc_shards_manifest.json".to_string(),
            train_fraction: 0.9,
            shard_samples: 10_000,
            split_mode: BcShardSplitMode::Both,
            exit_sidecar: None,
            exit_sidecar_path: None,
            exit_provenance: SidecarProvenance::default(),
            delta_q_sidecar: None,
            delta_q_sidecar_path: None,
            delta_q_provenance: SidecarProvenance::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BcShardBuildOutput {
    pub manifest_path: PathBuf,
    pub manifest: BcShardManifest,
}

pub struct BcShardBatch<B: Backend> {
    pub obs: Tensor<B, 3>,
    pub batch: MjaiBcBatch<B>,
    pub targets: HydraTargets<B>,
}

/// CPU-side host batch ready to cross a thread boundary.
///
/// All expensive shard I/O, parsing, and augmentation is already done.
/// Call [`materialize`](BcShardHostBatch::materialize) to create device
/// tensors from these flat buffers.
pub struct BcShardHostBatch {
    pub batch_size: usize,
    pub obs_flat: Vec<f32>,
    pub actions: Vec<i64>,
    pub legal_mask_flat: Vec<f32>,
    pub value_target: Vec<f32>,
    pub grp_target_flat: Vec<f32>,
    pub oracle_target_flat: Vec<f32>,
    pub oracle_target_mask: Vec<f32>,
    pub tenpai_flat: Vec<f32>,
    pub danger_flat: Vec<f32>,
    pub danger_mask_flat: Vec<f32>,
    pub opp_next_flat: Vec<f32>,
    pub score_pdf_flat: Vec<f32>,
    pub score_cdf_flat: Vec<f32>,
    pub safety_target_flat: Option<Vec<f32>>,
    pub safety_mask_flat: Option<Vec<f32>>,
    pub exit_target_flat: Option<Vec<f32>>,
    pub exit_mask_flat: Option<Vec<f32>>,
    pub delta_q_target_flat: Option<Vec<f32>>,
    pub delta_q_mask_flat: Option<Vec<f32>>,
    pub target_presence: TargetPresence,
}

// SAFETY: all fields are plain vecs of Copy types -- trivially Send + Sync.
unsafe impl Send for BcShardHostBatch {}
unsafe impl Sync for BcShardHostBatch {}

impl BcShardHostBatch {
    fn empty() -> Self {
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
            target_presence: TargetPresence::default(),
        }
    }
}

/// Reusable scratch buffers for the BC shard producer path.
///
/// Mirrors every field in [`BcShardHostBatch`] but is designed to be
/// reset and refilled across batches without reallocating.  The producer
/// thread creates one of these, calls
/// [`BcShardReader::collate_host_batch_into`] each iteration, then
/// [`take_batch`](BcShardHostScratch::take_batch) to extract an owned
/// `BcShardHostBatch` with zero-cost `mem::take` swaps (the scratch
/// retains heap capacity for the next iteration).
pub struct BcShardHostScratch {
    pub batch_size: usize,
    pub obs_flat: Vec<f32>,
    pub actions: Vec<i64>,
    pub legal_mask_flat: Vec<f32>,
    pub value_target: Vec<f32>,
    pub grp_target_flat: Vec<f32>,
    pub oracle_target_flat: Vec<f32>,
    pub oracle_target_mask: Vec<f32>,
    pub tenpai_flat: Vec<f32>,
    pub danger_flat: Vec<f32>,
    pub danger_mask_flat: Vec<f32>,
    pub opp_next_flat: Vec<f32>,
    pub score_pdf_flat: Vec<f32>,
    pub score_cdf_flat: Vec<f32>,
    pub safety_target_flat: Option<Vec<f32>>,
    pub safety_mask_flat: Option<Vec<f32>>,
    pub exit_target_flat: Option<Vec<f32>>,
    pub exit_mask_flat: Option<Vec<f32>>,
    pub delta_q_target_flat: Option<Vec<f32>>,
    pub delta_q_mask_flat: Option<Vec<f32>>,
    pub target_presence: TargetPresence,
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
            target_presence: TargetPresence::with_batch_size(batch_size),
        }
    }

    /// Zero-fill all buffers and resize to `batch_size` without
    /// deallocating when existing capacity is sufficient.
    pub fn reset(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
        // obs_flat: every element is overwritten by write_*_row_into_scratch,
        // so skip zeroing the largest buffer (~1.6MB at batch=64).
        resize_uninit_f32(&mut self.obs_flat, batch_size * OBS_SIZE);
        resize_zeroed_i64(&mut self.actions, batch_size);
        resize_uninit_f32(&mut self.legal_mask_flat, batch_size * HYDRA_ACTION_SPACE);
        resize_zeroed(&mut self.value_target, batch_size);
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
        self.target_presence = TargetPresence::with_batch_size(batch_size);
    }

    /// Swap the filled buffers out of this scratch into an owned
    /// [`BcShardHostBatch`].  The scratch retains the heap allocations
    /// (now length-zero) so the next [`reset`](Self::reset) call
    /// can refill them without allocating.
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
            target_presence: std::mem::replace(
                &mut self.target_presence,
                TargetPresence::default(),
            ),
        }
    }

    /// Extract a batch while recycling a previously-consumed batch's
    /// heap allocations back into the scratch.  This preserves Vec
    /// capacity across iterations, eliminating 18+ heap allocations
    /// per batch (including the ~1.6MB obs_flat buffer).
    ///
    /// `recycled` should be a batch whose data has been consumed (e.g.
    /// by [`BcShardHostBatch::materialize`]).  Its Vec shells (with
    /// their allocated-but-logically-empty backing memory) are swapped
    /// into the scratch so the next [`reset`] reuses that capacity.
    pub fn swap_batch(&mut self, recycled: &mut BcShardHostBatch) -> BcShardHostBatch {
        std::mem::swap(&mut self.obs_flat, &mut recycled.obs_flat);
        std::mem::swap(&mut self.actions, &mut recycled.actions);
        std::mem::swap(&mut self.legal_mask_flat, &mut recycled.legal_mask_flat);
        std::mem::swap(&mut self.value_target, &mut recycled.value_target);
        std::mem::swap(&mut self.grp_target_flat, &mut recycled.grp_target_flat);
        std::mem::swap(&mut self.oracle_target_flat, &mut recycled.oracle_target_flat);
        std::mem::swap(&mut self.oracle_target_mask, &mut recycled.oracle_target_mask);
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
        recycled.target_presence = std::mem::replace(
            &mut self.target_presence,
            TargetPresence::default(),
        );
        recycled.batch_size = self.batch_size;

        // `recycled` now holds the freshly-collated data; the scratch
        // holds the recycled (capacity-preserving) empty vecs.
        let mut out = BcShardHostBatch::empty();
        std::mem::swap(&mut out, recycled);
        out
    }
}

/// Resize a vec to `len` and zero-fill.  Reuses existing heap when
/// capacity is sufficient -- the entire point of the scratch pattern.
#[inline]
fn resize_zeroed(buf: &mut Vec<f32>, len: usize) {
    buf.clear();
    buf.resize(len, 0.0);
}

/// # Safety
/// Caller guarantees every element in `0..len` will be written before read.
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
fn resize_zeroed_i64(buf: &mut Vec<i64>, len: usize) {
    buf.clear();
    buf.resize(len, 0);
}

impl BcShardHostBatch {
    /// Materialize device tensors from CPU-side flat buffers.
    ///
    /// This is the only step that touches the `Backend` / device.
    pub fn materialize<B: Backend>(&self, device: &B::Device) -> BcShardBatch<B> {
        let batch = self.batch_size;

        let obs = Tensor::<B, 1>::from_floats(self.obs_flat.as_slice(), device)
            .reshape([batch, NUM_CHANNELS, TILE_COUNT]);
        let actions_tensor = Tensor::<B, 1, Int>::from_ints(self.actions.as_slice(), device);
        let legal_mask = Tensor::<B, 1>::from_floats(self.legal_mask_flat.as_slice(), device)
            .reshape([batch, HYDRA_ACTION_SPACE]);
        let value_target = Tensor::<B, 1>::from_floats(self.value_target.as_slice(), device);
        let grp_target = Tensor::<B, 1>::from_floats(self.grp_target_flat.as_slice(), device)
            .reshape([batch, GRP_CLASS_COUNT]);
        let oracle_target = Tensor::<B, 1>::from_floats(self.oracle_target_flat.as_slice(), device)
            .reshape([batch, PLAYER_COUNT]);
        let oracle_target_mask = Tensor::<B, 1>::from_floats(self.oracle_target_mask.as_slice(), device);
        let tenpai_target = Tensor::<B, 1>::from_floats(self.tenpai_flat.as_slice(), device)
            .reshape([batch, OPPONENT_COUNT]);
        let danger_target = Tensor::<B, 1>::from_floats(self.danger_flat.as_slice(), device)
            .reshape([batch, OPPONENT_COUNT, TILE_COUNT]);
        let danger_mask = Tensor::<B, 1>::from_floats(self.danger_mask_flat.as_slice(), device)
            .reshape([batch, OPPONENT_COUNT, TILE_COUNT]);
        let opp_next_target = Tensor::<B, 1>::from_floats(self.opp_next_flat.as_slice(), device)
            .reshape([batch, OPPONENT_COUNT, TILE_COUNT]);
        let score_pdf_target = Tensor::<B, 1>::from_floats(self.score_pdf_flat.as_slice(), device)
            .reshape([batch, SCORE_BINS]);
        let score_cdf_target = Tensor::<B, 1>::from_floats(self.score_cdf_flat.as_slice(), device)
            .reshape([batch, SCORE_BINS]);

        let exit_target_tensor = self.exit_target_flat.as_ref().map(|buf| {
            Tensor::<B, 1>::from_floats(buf.as_slice(), device).reshape([batch, HYDRA_ACTION_SPACE])
        });
        let exit_mask_tensor = self.exit_mask_flat.as_ref().map(|buf| {
            Tensor::<B, 1>::from_floats(buf.as_slice(), device).reshape([batch, HYDRA_ACTION_SPACE])
        });

        let policy_target = policy_target_from_actions::<B>(actions_tensor.clone(), batch);

        let batch_struct = MjaiBcBatch {
            actions: actions_tensor,
            exit_target: exit_target_tensor,
            exit_mask: exit_mask_tensor,
        };

        let targets = HydraTargets {
            policy_target,
            legal_mask,
            value_target,
            grp_target,
            tenpai_target,
            danger_target,
            danger_mask,
            opp_next_target,
            score_pdf_target,
            score_cdf_target,
            oracle_target: Some(oracle_target),
            belief_fields_target: None,
            belief_fields_mask: None,
            mixture_weight_target: None,
            mixture_weight_mask: None,
            opponent_hand_type_target: None,
            delta_q_target: self.delta_q_target_flat.as_ref().map(|buf| {
                Tensor::<B, 1>::from_floats(buf.as_slice(), device)
                    .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            delta_q_mask: self.delta_q_mask_flat.as_ref().map(|buf| {
                Tensor::<B, 1>::from_floats(buf.as_slice(), device).reshape([batch, HYDRA_ACTION_SPACE])
            }),
            safety_residual_target: self.safety_target_flat.as_ref().map(|buf| {
                Tensor::<B, 1>::from_floats(buf.as_slice(), device)
                    .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            safety_residual_mask: self.safety_mask_flat.as_ref().map(|buf| {
                Tensor::<B, 1>::from_floats(buf.as_slice(), device).reshape([batch, HYDRA_ACTION_SPACE])
            }),
            oracle_guidance_mask: Some(oracle_target_mask),
            target_presence: Some(self.target_presence),
        };

        BcShardBatch {
            obs,
            batch: batch_struct,
            targets,
        }
    }
}

pub struct BcShardReader {
    shards: Vec<ShardMap>,
}

struct ShardMap {
    start_sample: u64,
    sample_count: u64,
    feature_flags: u32,
    record_size: usize,
    mmap: Mmap,
}

struct ActiveShardWriter {
    split: BcShardSplit,
    shard_index: usize,
    file_name: String,
    first_sample_index: u64,
    sample_count: u64,
    feature_flags: u32,
    record_size: u32,
    writer: BufWriter<fs::File>,
}

struct SplitBuildState {
    split: BcShardSplit,
    next_shard_index: usize,
    total_samples: u64,
    feature_flags: u32,
    record_size: u32,
    shards: Vec<BcShardDescriptor>,
    active: Option<ActiveShardWriter>,
}

impl SplitBuildState {
    fn new(split: BcShardSplit, feature_flags: u32) -> Self {
        Self {
            split,
            next_shard_index: 0,
            total_samples: 0,
            feature_flags,
            record_size: record_size_for_flags(feature_flags),
            shards: Vec::new(),
            active: None,
        }
    }

    fn push_game(
        &mut self,
        output_dir: &Path,
        shard_samples: usize,
        game: &MjaiGame,
    ) -> io::Result<()> {
        if game.samples.is_empty() {
            return Ok(());
        }
        let game_samples = game.samples.len() as u64;
        if let Some(active) = self.active.as_ref()
            && active.sample_count > 0
            && active.sample_count + game_samples > shard_samples.max(1) as u64
        {
            self.finish_active()?;
        }
        if self.active.is_none() {
            let shard = ActiveShardWriter::new(
                output_dir,
                self.split,
                self.next_shard_index,
                self.total_samples,
                self.feature_flags,
            )?;
            self.next_shard_index += 1;
            self.active = Some(shard);
        }
        let active = self.active.as_mut().expect("active shard should exist");
        active.write_game(game)?;
        self.total_samples += game_samples;
        Ok(())
    }

    fn finish_active(&mut self) -> io::Result<()> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        let descriptor = active.finish()?;
        self.shards.push(descriptor);
        Ok(())
    }

    fn finalize(mut self) -> io::Result<BcShardSplitManifest> {
        self.finish_active()?;
        Ok(BcShardSplitManifest {
            split: self.split,
            shard_count: self.shards.len(),
            sample_count: self.total_samples,
            feature_flags: self.feature_flags,
            record_size: self.record_size,
            shards: self.shards,
        })
    }
}

impl ActiveShardWriter {
    fn new(
        output_dir: &Path,
        split: BcShardSplit,
        shard_index: usize,
        first_sample_index: u64,
        feature_flags: u32,
    ) -> io::Result<Self> {
        let file_name = format!("{}-{shard_index:05}.hydra-bc", split.shard_prefix());
        let path = output_dir.join(&file_name);
        let file = fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);
        let record_size = record_size_for_flags(feature_flags);
        write_shard_header(
            &mut writer,
            split,
            shard_index as u32,
            0,
            first_sample_index,
            feature_flags,
            record_size,
        )?;
        Ok(Self {
            split,
            shard_index,
            file_name,
            first_sample_index,
            sample_count: 0,
            feature_flags,
            record_size,
            writer,
        })
    }

    fn write_game(&mut self, game: &MjaiGame) -> io::Result<()> {
        for sample in &game.samples {
            write_sample_record(&mut self.writer, sample, self.feature_flags)?;
            self.sample_count += 1;
        }
        Ok(())
    }

    fn finish(mut self) -> io::Result<BcShardDescriptor> {
        self.writer.flush()?;
        let file = self.writer.get_mut();
        file.seek(SeekFrom::Start(0))?;
        write_shard_header(
            file,
            self.split,
            self.shard_index as u32,
            self.sample_count,
            self.first_sample_index,
            self.feature_flags,
            self.record_size,
        )?;
        file.flush()?;
        let byte_len = file.metadata()?.len();
        Ok(BcShardDescriptor {
            split: self.split,
            shard_index: self.shard_index,
            file_name: self.file_name,
            sample_count: self.sample_count,
            first_sample_index: self.first_sample_index,
            byte_len,
            feature_flags: self.feature_flags,
            record_size: self.record_size,
        })
    }
}

pub fn build_bc_shards(config: &BuildBcShardsConfig) -> io::Result<BcShardBuildOutput> {
    if config.shard_samples == 0 {
        return Err(invalid_data("shard_samples must be > 0"));
    }
    fs::create_dir_all(&config.output_dir)?;
    let source_manifest = scan_data_sources(&config.input)?;
    let feature_flags = feature_flags_from_config(config);

    let mut train_state = config
        .split_mode
        .includes(BcShardSplit::Train)
        .then(|| SplitBuildState::new(BcShardSplit::Train, feature_flags));
    let mut val_state = config
        .split_mode
        .includes(BcShardSplit::Validation)
        .then(|| SplitBuildState::new(BcShardSplit::Validation, feature_flags));
    let mut skipped_games = 0u64;
    let mut empty_games = 0u64;

    for source in &source_manifest.sources {
        match source {
            DataSource::LooseFile(path) => process_loose_file(
                path,
                config,
                &mut train_state,
                &mut val_state,
                &mut skipped_games,
                &mut empty_games,
            )?,
            DataSource::Archive(path) => process_archive(
                path,
                config,
                &mut train_state,
                &mut val_state,
                &mut skipped_games,
                &mut empty_games,
            )?,
        }
    }

    let mut split_manifests = Vec::new();
    if let Some(train_state) = train_state {
        split_manifests.push(train_state.finalize()?);
    }
    if let Some(val_state) = val_state {
        split_manifests.push(val_state.finalize()?);
    }

    let mut totals = BcShardBuildTotals {
        skipped_games,
        empty_games,
        ..BcShardBuildTotals::default()
    };
    for split in &split_manifests {
        totals.sample_count += split.sample_count;
        totals.shard_count += split.shard_count;
    }

    let manifest = BcShardManifest {
        manifest_version: BC_SHARD_MANIFEST_VERSION,
        shard_version: BC_SHARD_VERSION,
        shard_header_size: BC_SHARD_HEADER_SIZE,
        base_record_size: BC_BASE_RECORD_SIZE,
        max_record_size: BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
        obs_size: OBS_SIZE,
        num_channels: NUM_CHANNELS,
        action_space: HYDRA_ACTION_SPACE,
        train_fraction: config.train_fraction,
        shard_samples: config.shard_samples,
        augment_runtime: true,
        input: config.input.display().to_string(),
        output_dir: config.output_dir.display().to_string(),
        created_at: OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string()),
        source_count: source_manifest.sources.len(),
        source_total_games_hint: source_manifest.total_games,
        source_train_count_hint: source_manifest.train_count,
        source_val_count_hint: source_manifest.val_count,
        source_counts_exact: source_manifest.counts_exact,
        exit_sidecar: sidecar_manifest(config.exit_sidecar_path.as_deref(), config.exit_provenance),
        delta_q_sidecar: sidecar_manifest(
            config.delta_q_sidecar_path.as_deref(),
            config.delta_q_provenance,
        ),
        totals,
        splits: split_manifests,
    };
    let manifest_path = config.output_dir.join(&config.manifest_name);
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest)
            .map_err(|err| invalid_data(format!("failed to serialize BC shard manifest: {err}")))?,
    )?;
    Ok(BcShardBuildOutput {
        manifest_path,
        manifest,
    })
}

pub fn load_bc_shard_reader(
    manifest_path: &Path,
    split: BcShardSplit,
) -> Result<BcShardReader, String> {
    let raw = fs::read_to_string(manifest_path)
        .map_err(|err| format!("failed to read BC shard manifest {}: {err}", manifest_path.display()))?;
    let manifest: BcShardManifest = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse BC shard manifest {}: {err}", manifest_path.display()))?;
    // Reject shards built with a different encoder geometry. Without this
    // check a stale manifest silently passes header verification (which only
    // compares header-vs-manifest) and the reader panics deep inside
    // parse_compact_sample when OBS_F32_BYTES exceeds the on-disk record.
    if manifest.obs_size != OBS_SIZE {
        return Err(format!(
            "BC shard manifest obs_size {} does not match current OBS_SIZE {} \
             (num_channels: manifest={}, binary={}). \
             Shards must be rebuilt with the current encoder.",
            manifest.obs_size, OBS_SIZE,
            manifest.num_channels, NUM_CHANNELS,
        ));
    }
    if manifest.base_record_size != BC_BASE_RECORD_SIZE {
        return Err(format!(
            "BC shard manifest base_record_size {} does not match current \
             BC_BASE_RECORD_SIZE {}. Shards must be rebuilt with the current encoder.",
            manifest.base_record_size, BC_BASE_RECORD_SIZE,
        ));
    }

    let base_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let split_manifest = manifest
        .splits
        .iter()
        .find(|entry| entry.split == split)
        .ok_or_else(|| format!("BC shard manifest missing {:?} split", split))?;
    let mut shards = Vec::with_capacity(split_manifest.shards.len());
    for shard in &split_manifest.shards {
        let path = base_dir.join(&shard.file_name);
        let file = fs::File::open(&path)
            .map_err(|err| format!("failed to open BC shard {}: {err}", path.display()))?;
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|err| format!("failed to mmap BC shard {}: {err}", path.display()))?
        };
        // Shard access is a strict forward-sequential scan.  MADV_SEQUENTIAL
        // doubles the kernel read-ahead window and frees pages behind the
        // cursor, reducing page faults and RSS for 251MB+ shard files.
        let _ = mmap.advise(Advice::Sequential);
        verify_shard_header(&mmap, split, shard.feature_flags, shard.record_size)?;
        shards.push(ShardMap {
            start_sample: shard.first_sample_index,
            sample_count: shard.sample_count,
            feature_flags: shard.feature_flags,
            record_size: shard.record_size as usize,
            mmap,
        });
    }
    Ok(BcShardReader { shards })
}

impl BcShardReader {
    pub fn sample_count(&self) -> usize {
        self.shards.iter().map(|shard| shard.sample_count as usize).sum()
    }

    pub fn feature_flags(&self) -> u32 {
        self.shards.first().map_or(0, |s| s.feature_flags)
    }

    pub fn new_scratch(&self, batch_size: usize) -> BcShardHostScratch {
        let flags = self.feature_flags();
        BcShardHostScratch::new(
            batch_size,
            flags & FLAG_SAFETY_RESIDUAL != 0,
            flags & FLAG_EXIT != 0,
            flags & FLAG_DELTA_Q != 0,
        )
    }

    /// Full collation: parse shards, augment, then materialize on device.
    ///
    /// Equivalent to `collate_host_batch` followed by `BcShardHostBatch::materialize`.
    pub fn collate_batch<B: Backend>(
        &self,
        indices: &[usize],
        augment: bool,
        device: &B::Device,
    ) -> Result<BcShardBatch<B>, String> {
        self.collate_host_batch(indices, augment)
            .map(|host| host.materialize(device))
    }

    pub fn collate_batch_range<B: Backend>(
        &self,
        start: usize,
        len: usize,
        augment: bool,
        device: &B::Device,
    ) -> Result<BcShardBatch<B>, String> {
        self.collate_host_batch_range(start, len, augment)
            .map(|host| host.materialize(device))
    }

    /// CPU-only batch collation: shard I/O, parsing, and augmentation.
    ///
    /// Returns a backend-agnostic host batch suitable for crossing a thread
    /// boundary before device materialization.
    pub fn collate_host_batch(
        &self,
        indices: &[usize],
        augment: bool,
    ) -> Result<BcShardHostBatch, String> {
        if indices.is_empty() {
            return Err("bc shard batch indices must be non-empty".to_string());
        }
        let mut scratch = self.new_scratch(indices.len());
        self.collate_host_batch_into(indices, augment, &mut scratch)?;
        Ok(scratch.take_batch())
    }

    pub fn collate_host_batch_range(
        &self,
        start: usize,
        len: usize,
        augment: bool,
    ) -> Result<BcShardHostBatch, String> {
        if len == 0 {
            return Err("bc shard batch indices must be non-empty".to_string());
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| "bc shard batch range overflow".to_string())?;
        if end > self.sample_count() {
            return Err(format!(
                "bc shard batch range {start}..{end} exceeds sample count {}",
                self.sample_count()
            ));
        }
        let mut scratch = self.new_scratch(len);
        self.collate_host_batch_range_into(start, len, augment, &mut scratch)?;
        Ok(scratch.take_batch())
    }

    /// Collate into a pre-allocated scratch buffer, avoiding per-batch
    /// heap allocations when the scratch already has sufficient capacity.
    ///
    /// Caller must call [`BcShardHostScratch::reset`] before this if
    /// the scratch was previously used.
    pub fn collate_host_batch_into(
        &self,
        indices: &[usize],
        augment: bool,
        scratch: &mut BcShardHostScratch,
    ) -> Result<(), String> {
        if indices.is_empty() {
            return Err("bc shard batch indices must be non-empty".to_string());
        }

        let batch = indices.len();
        scratch.reset(batch);

        for (row, &sample_index) in indices.iter().enumerate() {
            let (shard, offset) = self.locate(sample_index)?;
            if augment {
                write_augmented_row_into_scratch(shard, offset, row, sample_index, scratch)?;
            } else {
                write_unaugmented_row_into_scratch(shard, offset, row, scratch)?;
            }
        }

        Ok(())
    }

    pub fn collate_host_batch_range_into(
        &self,
        start: usize,
        len: usize,
        augment: bool,
        scratch: &mut BcShardHostScratch,
    ) -> Result<(), String> {
        if len == 0 {
            return Err("bc shard batch indices must be non-empty".to_string());
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| "bc shard batch range overflow".to_string())?;
        if end > self.sample_count() {
            return Err(format!(
                "bc shard batch range {start}..{end} exceeds sample count {}",
                self.sample_count()
            ));
        }

        scratch.reset(len);
        let (mut shard_index, mut offset) = self.locate_index(start)?;
        for row in 0..len {
            let sample_index = start + row;
            let shard = &self.shards[shard_index];
            if augment {
                write_augmented_row_into_scratch(shard, offset, row, sample_index, scratch)?;
            } else {
                write_unaugmented_row_into_scratch(shard, offset, row, scratch)?;
            }

            offset += 1;
            if offset == shard.sample_count as usize && row + 1 < len {
                shard_index += 1;
                if shard_index >= self.shards.len() {
                    return Err("BC shard range collation ran past shard list".to_string());
                }
                offset = 0;
            }
        }

        Ok(())
    }

    fn locate(&self, sample_index: usize) -> Result<(&ShardMap, usize), String> {
        let (shard_index, offset) = self.locate_index(sample_index)?;
        Ok((&self.shards[shard_index], offset))
    }

    fn locate_index(&self, sample_index: usize) -> Result<(usize, usize), String> {
        let sample_index = sample_index as u64;
        let shard_index = self
            .shards
            .partition_point(|shard| shard.start_sample <= sample_index)
            .checked_sub(1)
            .ok_or_else(|| format!("BC shard sample index {sample_index} out of bounds"))?;
        let shard = &self.shards[shard_index];
        let shard_end = shard.start_sample + shard.sample_count;
        if sample_index >= shard_end {
            return Err(format!("BC shard sample index {sample_index} out of bounds"));
        }
        Ok((shard_index, (sample_index - shard.start_sample) as usize))
    }
}

fn write_unaugmented_row_into_scratch(
    shard: &ShardMap,
    row_index: usize,
    row: usize,
    scratch: &mut BcShardHostScratch,
) -> Result<(), String> {
    let start = BC_SHARD_HEADER_SIZE as usize + row_index * shard.record_size;
    let end = start + shard.record_size;
    if end > shard.mmap.len() {
        return Err("BC shard row extends past file end".to_string());
    }
    let bytes = &shard.mmap[start..end];
    let mut cursor = 0usize;

    let obs_dst = &mut scratch.obs_flat[row * OBS_SIZE..(row + 1) * OBS_SIZE];
    #[cfg(target_endian = "little")]
    unsafe {
        ptr::copy_nonoverlapping(
            bytes[cursor..cursor + OBS_F32_BYTES].as_ptr(),
            obs_dst.as_mut_ptr().cast::<u8>(),
            OBS_F32_BYTES,
        );
    }
    #[cfg(not(target_endian = "little"))]
    for (value, chunk) in obs_dst
        .iter_mut()
        .zip(bytes[cursor..cursor + OBS_F32_BYTES].chunks_exact(4))
    {
        *value = f32::from_le_bytes(chunk.try_into().expect("f32 chunk"));
    }
    cursor += OBS_F32_BYTES;

    scratch.actions[row] = bytes[cursor] as i64;
    cursor += 1;

    // Branchless u8 -> f32 boolean: min(1) clamps any nonzero to 1,
    // then integer-to-float produces exactly 0.0 or 1.0 without a branch.
    for (dst, &src) in scratch.legal_mask_flat
        [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE]
        .iter_mut()
        .zip(bytes[cursor..cursor + HYDRA_ACTION_SPACE].iter())
    {
        *dst = src.min(1) as f32;
    }
    cursor += HYDRA_ACTION_SPACE;

    let score_delta = read_i32_le(&bytes[cursor..cursor + 4]);
    cursor += 4;
    scratch.value_target[row] = score_delta_to_value(score_delta);
    let bin = score_delta_to_bin(score_delta);
    scratch.score_pdf_flat[row * SCORE_BINS + bin] = 1.0;
    scratch.score_cdf_flat[row * SCORE_BINS + bin..(row + 1) * SCORE_BINS].fill(1.0);

    let grp = (bytes[cursor] as usize).min(GRP_CLASS_COUNT - 1);
    scratch.grp_target_flat[row * GRP_CLASS_COUNT + grp] = 1.0;
    cursor += 1;

    // Direct mmap-to-scratch copy on little-endian; avoids stack intermediate.
    let oracle_dst = &mut scratch.oracle_target_flat[row * PLAYER_COUNT..(row + 1) * PLAYER_COUNT];
    #[cfg(target_endian = "little")]
    {
        let src_bytes = &bytes[cursor..cursor + ORACLE_FLOAT32_BYTES];
        // SAFETY: PLAYER_COUNT * 4 == ORACLE_FLOAT32_BYTES, and f32 has no
        // invalid bit patterns.  Alignment is guaranteed by slice layout.
        unsafe {
            ptr::copy_nonoverlapping(
                src_bytes.as_ptr(),
                oracle_dst.as_mut_ptr().cast::<u8>(),
                ORACLE_FLOAT32_BYTES,
            );
        }
    }
    #[cfg(not(target_endian = "little"))]
    oracle_dst.copy_from_slice(&read_oracle_f32(&bytes[cursor..cursor + ORACLE_FLOAT32_BYTES]));
    cursor += ORACLE_FLOAT32_BYTES;
    let oracle_present = bytes[cursor] > 0;
    cursor += ORACLE_MASK_BYTES;
    if oracle_present {
        scratch.oracle_target_mask[row] = 1.0;
        scratch.target_presence.counts[AdvancedHead::OracleCritic.index()] += 1;
    } else {
        oracle_dst.fill(0.0);
    }

    for (dst, &src) in scratch.tenpai_flat[row * OPPONENT_COUNT..(row + 1) * OPPONENT_COUNT]
        .iter_mut()
        .zip(bytes[cursor..cursor + OPPONENT_COUNT].iter())
    {
        *dst = src.min(1) as f32;
    }
    cursor += OPPONENT_COUNT;

    for (opp, &tile) in bytes[cursor..cursor + OPPONENT_COUNT].iter().enumerate() {
        if (tile as usize) < TILE_COUNT {
            scratch.opp_next_flat[row * SPATIAL_TARGET_SIZE + opp * TILE_COUNT + tile as usize] =
                1.0;
        }
    }
    cursor += OPPONENT_COUNT;

    for (dst, &src) in scratch.danger_flat
        [row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE]
        .iter_mut()
        .zip(bytes[cursor..cursor + SPATIAL_TARGET_SIZE].iter())
    {
        *dst = src.min(1) as f32;
    }
    cursor += SPATIAL_TARGET_SIZE;

    for (dst, &src) in scratch.danger_mask_flat
        [row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE]
        .iter_mut()
        .zip(bytes[cursor..cursor + SPATIAL_TARGET_SIZE].iter())
    {
        *dst = src.min(1) as f32;
    }
    cursor += SPATIAL_TARGET_SIZE;

    if shard.feature_flags & FLAG_SAFETY_RESIDUAL != 0 {
        let dst = &mut scratch.safety_target_flat.as_mut().expect("safety enabled")
            [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
        read_optional_action_f32_into(&bytes[cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES], dst);
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;

        let dst = &mut scratch.safety_mask_flat.as_mut().expect("safety enabled")
            [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
        if read_optional_action_mask_f32_into(&bytes[cursor..cursor + OPTIONAL_ACTION_MASK_BYTES], dst) {
            scratch.target_presence.counts[AdvancedHead::SafetyResidual.index()] += 1;
        }
        cursor += OPTIONAL_ACTION_MASK_BYTES;
    }

    if shard.feature_flags & FLAG_EXIT != 0 {
        let dst = &mut scratch.exit_target_flat.as_mut().expect("exit enabled")
            [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
        read_optional_action_f32_into(&bytes[cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES], dst);
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;

        let dst = &mut scratch.exit_mask_flat.as_mut().expect("exit enabled")
            [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
        read_optional_action_mask_f32_into(&bytes[cursor..cursor + OPTIONAL_ACTION_MASK_BYTES], dst);
        cursor += OPTIONAL_ACTION_MASK_BYTES;
    }

    if shard.feature_flags & FLAG_DELTA_Q != 0 {
        let dst = &mut scratch.delta_q_target_flat.as_mut().expect("delta_q enabled")
            [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
        read_optional_action_f32_into(&bytes[cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES], dst);
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;

        let dst = &mut scratch.delta_q_mask_flat.as_mut().expect("delta_q enabled")
            [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
        if read_optional_action_mask_f32_into(&bytes[cursor..cursor + OPTIONAL_ACTION_MASK_BYTES], dst) {
            let action_count = dst.iter().filter(|&&value| value > 0.0).count();
            if action_count > 0 {
                scratch.target_presence.counts[AdvancedHead::DeltaQ.index()] += 1;
                scratch.target_presence.delta_q_actions_present += action_count;
            }
        }
    }

    Ok(())
}

fn write_augmented_row_into_scratch(
    shard: &ShardMap,
    row_index: usize,
    row: usize,
    sample_index: usize,
    scratch: &mut BcShardHostScratch,
) -> Result<(), String> {
    let start = BC_SHARD_HEADER_SIZE as usize + row_index * shard.record_size;
    let end = start + shard.record_size;
    if end > shard.mmap.len() {
        return Err("BC shard row extends past file end".to_string());
    }
    let bytes = &shard.mmap[start..end];
    let mut cursor = 0usize;

    let perm_idx = (sample_index + row) % hydra_core::tile::ALL_PERMUTATIONS.len();
    let perm = &hydra_core::tile::ALL_PERMUTATIONS[perm_idx];
    let tables = permutation_tables();
    let tile_perm = &tables.tile_34[perm_idx];
    let action_perm = &tables.action_37[perm_idx];

    let obs_dst = &mut scratch.obs_flat[row * OBS_SIZE..(row + 1) * OBS_SIZE];
    augment_obs_suit_from_le_bytes(&bytes[cursor..cursor + OBS_F32_BYTES], perm, obs_dst);
    cursor += OBS_F32_BYTES;

    let action = bytes[cursor];
    scratch.actions[row] = if action <= 36 {
        action_perm[action as usize] as i64
    } else {
        action as i64
    };
    cursor += 1;

    let mask_src = &bytes[cursor..cursor + HYDRA_ACTION_SPACE];
    let mask_dst =
        &mut scratch.legal_mask_flat[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
    mask_dst.fill(0.0);
    for i in 0..37usize {
        mask_dst[action_perm[i]] = mask_src[i].min(1) as f32;
    }
    for (dst, &src) in mask_dst[37..HYDRA_ACTION_SPACE]
        .iter_mut()
        .zip(mask_src[37..HYDRA_ACTION_SPACE].iter())
    {
        *dst = src.min(1) as f32;
    }
    cursor += HYDRA_ACTION_SPACE;

    let score_delta = read_i32_le(&bytes[cursor..cursor + 4]);
    cursor += 4;
    scratch.value_target[row] = score_delta_to_value(score_delta);
    let bin = score_delta_to_bin(score_delta);
    scratch.score_pdf_flat[row * SCORE_BINS + bin] = 1.0;
    scratch.score_cdf_flat[row * SCORE_BINS + bin..(row + 1) * SCORE_BINS].fill(1.0);

    let grp = (bytes[cursor] as usize).min(GRP_CLASS_COUNT - 1);
    scratch.grp_target_flat[row * GRP_CLASS_COUNT + grp] = 1.0;
    cursor += 1;

    let oracle_dst = &mut scratch.oracle_target_flat[row * PLAYER_COUNT..(row + 1) * PLAYER_COUNT];
    #[cfg(target_endian = "little")]
    {
        let src_bytes = &bytes[cursor..cursor + ORACLE_FLOAT32_BYTES];
        unsafe {
            ptr::copy_nonoverlapping(
                src_bytes.as_ptr(),
                oracle_dst.as_mut_ptr().cast::<u8>(),
                ORACLE_FLOAT32_BYTES,
            );
        }
    }
    #[cfg(not(target_endian = "little"))]
    oracle_dst.copy_from_slice(&read_oracle_f32(&bytes[cursor..cursor + ORACLE_FLOAT32_BYTES]));
    cursor += ORACLE_FLOAT32_BYTES;
    let oracle_present = bytes[cursor] > 0;
    cursor += ORACLE_MASK_BYTES;
    if oracle_present {
        scratch.oracle_target_mask[row] = 1.0;
        scratch.target_presence.counts[AdvancedHead::OracleCritic.index()] += 1;
    } else {
        oracle_dst.fill(0.0);
    }

    for (dst, &src) in scratch.tenpai_flat[row * OPPONENT_COUNT..(row + 1) * OPPONENT_COUNT]
        .iter_mut()
        .zip(bytes[cursor..cursor + OPPONENT_COUNT].iter())
    {
        *dst = src.min(1) as f32;
    }
    cursor += OPPONENT_COUNT;

    for (opp, &tile) in bytes[cursor..cursor + OPPONENT_COUNT].iter().enumerate() {
        let permuted = if tile < 34 {
            tile_perm[tile as usize]
        } else {
            tile as usize
        };
        if permuted < TILE_COUNT {
            scratch.opp_next_flat[row * SPATIAL_TARGET_SIZE + opp * TILE_COUNT + permuted] = 1.0;
        }
    }
    cursor += OPPONENT_COUNT;

    let danger_dst =
        &mut scratch.danger_flat[row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE];
    const SUIT_TILES: usize = 9;
    const HONOR_START: usize = 27;
    const HONOR_COUNT: usize = TILE_COUNT - HONOR_START;
    for opp in 0..OPPONENT_COUNT {
        let src_start = cursor + opp * TILE_COUNT;
        let dst_start = opp * TILE_COUNT;
        for src_suit in 0..3usize {
            let dst_suit = perm[src_suit] as usize;
            for t in 0..SUIT_TILES {
                danger_dst[dst_start + dst_suit * SUIT_TILES + t] =
                    bytes[src_start + src_suit * SUIT_TILES + t].min(1) as f32;
            }
        }
        for t in 0..HONOR_COUNT {
            danger_dst[dst_start + HONOR_START + t] =
                bytes[src_start + HONOR_START + t].min(1) as f32;
        }
    }
    cursor += SPATIAL_TARGET_SIZE;

    let dmask_dst = &mut scratch.danger_mask_flat
        [row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE];
    for opp in 0..OPPONENT_COUNT {
        let src_start = cursor + opp * TILE_COUNT;
        let dst_start = opp * TILE_COUNT;
        for src_suit in 0..3usize {
            let dst_suit = perm[src_suit] as usize;
            for t in 0..SUIT_TILES {
                dmask_dst[dst_start + dst_suit * SUIT_TILES + t] =
                    bytes[src_start + src_suit * SUIT_TILES + t].min(1) as f32;
            }
        }
        for t in 0..HONOR_COUNT {
            dmask_dst[dst_start + HONOR_START + t] =
                bytes[src_start + HONOR_START + t].min(1) as f32;
        }
    }
    cursor += SPATIAL_TARGET_SIZE;

    if shard.feature_flags & FLAG_SAFETY_RESIDUAL != 0 {
        let values =
            read_optional_action_f32(&bytes[cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES]);
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;
        if let Some(values) = values {
            let dst = &mut scratch
                .safety_target_flat
                .as_mut()
                .expect("safety enabled")
                [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
            augment_action_vector_suit_into(&values, action_perm, dst);
        }
        let dst = &mut scratch
            .safety_mask_flat
            .as_mut()
            .expect("safety enabled")
            [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
        if expand_and_augment_mask_into(&bytes[cursor..cursor + OPTIONAL_ACTION_MASK_BYTES], action_perm, dst) {
            scratch.target_presence.counts[AdvancedHead::SafetyResidual.index()] += 1;
        }
        cursor += OPTIONAL_ACTION_MASK_BYTES;
    }

    if shard.feature_flags & FLAG_EXIT != 0 {
        let values =
            read_optional_action_f32(&bytes[cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES]);
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;
        if let Some(values) = values {
            let dst = &mut scratch
                .exit_target_flat
                .as_mut()
                .expect("exit enabled")
                [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
            augment_action_vector_suit_into(&values, action_perm, dst);
        }
        let dst = &mut scratch
            .exit_mask_flat
            .as_mut()
            .expect("exit enabled")
            [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
        expand_and_augment_mask_into(&bytes[cursor..cursor + OPTIONAL_ACTION_MASK_BYTES], action_perm, dst);
        cursor += OPTIONAL_ACTION_MASK_BYTES;
    }

    if shard.feature_flags & FLAG_DELTA_Q != 0 {
        let values =
            read_optional_action_f32(&bytes[cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES]);
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;
        if let Some(values) = values {
            let dst = &mut scratch
                .delta_q_target_flat
                .as_mut()
                .expect("delta_q enabled")
                [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
            augment_action_vector_suit_into(&values, action_perm, dst);
        }
        let dst = &mut scratch
            .delta_q_mask_flat
            .as_mut()
            .expect("delta_q enabled")
            [row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
        if expand_and_augment_mask_into(&bytes[cursor..cursor + OPTIONAL_ACTION_MASK_BYTES], action_perm, dst) {
            let action_count = dst.iter().filter(|&&value| value > 0.0).count();
            if action_count > 0 {
                scratch.target_presence.counts[AdvancedHead::DeltaQ.index()] += 1;
                scratch.target_presence.delta_q_actions_present += action_count;
            }
        }
    }

    Ok(())
}

fn feature_flags_from_config(config: &BuildBcShardsConfig) -> u32 {
    let mut flags = 0u32;
    if config.exit_sidecar.is_some() {
        flags |= FLAG_EXIT;
    }
    if config.delta_q_sidecar.is_some() {
        flags |= FLAG_DELTA_Q;
    }
    flags |= FLAG_SAFETY_RESIDUAL;
    flags
}

fn record_size_for_flags(flags: u32) -> u32 {
    let mut size = BC_BASE_RECORD_SIZE;
    if flags & FLAG_SAFETY_RESIDUAL != 0 {
        size += (OPTIONAL_ACTION_FLOAT32_BYTES + OPTIONAL_ACTION_MASK_BYTES) as u32;
    }
    if flags & FLAG_EXIT != 0 {
        size += (OPTIONAL_ACTION_FLOAT32_BYTES + OPTIONAL_ACTION_MASK_BYTES) as u32;
    }
    if flags & FLAG_DELTA_Q != 0 {
        size += (OPTIONAL_ACTION_FLOAT32_BYTES + OPTIONAL_ACTION_MASK_BYTES) as u32;
    }
    size
}

fn sidecar_manifest(path: Option<&Path>, provenance: SidecarProvenance) -> Option<BcShardSidecarManifest> {
    let (source_net_hash, source_version) = provenance.source_net_hash.zip(provenance.source_version)?;
    Some(BcShardSidecarManifest {
        path: path?.display().to_string(),
        source_net_hash,
        source_version,
    })
}

fn process_loose_file(
    path: &Path,
    config: &BuildBcShardsConfig,
    train_state: &mut Option<SplitBuildState>,
    val_state: &mut Option<SplitBuildState>,
    skipped_games: &mut u64,
    empty_games: &mut u64,
) -> io::Result<()> {
    let identity = identity_for_loose_file(path)?;
    let Some(split) = split_for_identity(&identity, config) else {
        return Ok(());
    };
    let result = if config.exit_sidecar.is_some() || config.delta_q_sidecar.is_some() {
        load_game_from_path_with_sidecar(
            path,
            config.exit_provenance,
            config.delta_q_provenance,
            config.exit_sidecar.as_deref(),
            config.delta_q_sidecar.as_deref(),
        )
    } else {
        load_game_from_path(path)
    };
    handle_loaded_game(
        &identity,
        split,
        result,
        config,
        train_state,
        val_state,
        skipped_games,
        empty_games,
    )
}

fn process_archive(
    path: &Path,
    config: &BuildBcShardsConfig,
    train_state: &mut Option<SplitBuildState>,
    val_state: &mut Option<SplitBuildState>,
    skipped_games: &mut u64,
    empty_games: &mut u64,
) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let reader: Box<dyn Read> = if is_tar_zst_file(path) {
        let zstd = zstd::Decoder::new(file).map_err(|err| {
            io::Error::other(format!("failed to open zstd archive {}: {err}", path.display()))
        })?;
        Box::new(zstd)
    } else {
        Box::new(file)
    };
    let mut archive = tar::Archive::new(reader);

    for entry_result in archive.entries()? {
        let entry = entry_result?;
        let entry_path = entry.path()?.into_owned();
        if !is_mjai_archive_entry(&entry_path) {
            continue;
        }
        let identity = identity_for_archive_entry(path, &entry_path)?;
        let Some(split) = split_for_identity(&identity, config) else {
            continue;
        };
        let result = if config.exit_sidecar.is_some() || config.delta_q_sidecar.is_some() {
            load_game_from_stream_with_sidecar(
                &identity,
                config.exit_provenance,
                config.delta_q_provenance,
                entry,
                config.exit_sidecar.as_deref(),
                config.delta_q_sidecar.as_deref(),
            )
        } else {
            load_game_from_stream(entry)
        };
        handle_loaded_game(
            &identity,
            split,
            result,
            config,
            train_state,
            val_state,
            skipped_games,
            empty_games,
        )?;
    }
    Ok(())
}

fn handle_loaded_game(
    identity: &str,
    split: BcShardSplit,
    result: io::Result<MjaiGame>,
    config: &BuildBcShardsConfig,
    train_state: &mut Option<SplitBuildState>,
    val_state: &mut Option<SplitBuildState>,
    skipped_games: &mut u64,
    empty_games: &mut u64,
) -> io::Result<()> {
    match result {
        Ok(game) => {
            if game.samples.is_empty() {
                *empty_games += 1;
                return Ok(());
            }
            match split {
                BcShardSplit::Train => {
                    if let Some(state) = train_state.as_mut() {
                        state.push_game(&config.output_dir, config.shard_samples, &game)?;
                    }
                }
                BcShardSplit::Validation => {
                    if let Some(state) = val_state.as_mut() {
                        state.push_game(&config.output_dir, config.shard_samples, &game)?;
                    }
                }
            }
        }
        Err(err) => {
            *skipped_games += 1;
            eprintln!(
                "Skipping {}: {}",
                compact_identity(identity),
                compact_error_message(&err)
            );
        }
    }
    Ok(())
}

fn split_for_identity(identity: &str, config: &BuildBcShardsConfig) -> Option<BcShardSplit> {
    let split = if is_train_game(identity, config.train_fraction) {
        BcShardSplit::Train
    } else {
        BcShardSplit::Validation
    };
    config.split_mode.includes(split).then_some(split)
}

fn identity_for_loose_file(path: &Path) -> io::Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| invalid_data(format!("invalid filename {}", path.display())))
}

fn identity_for_archive_entry(archive_path: &Path, entry_path: &Path) -> io::Result<String> {
    let archive_name = archive_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid_data(format!("invalid archive name {}", archive_path.display())))?;
    Ok(format!("{archive_name}/{}", entry_path.display()))
}

fn is_tar_zst_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".tar.zst") || name.contains(".tar-") && name.ends_with(".zst")
    )
}

fn is_mjai_archive_entry(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name)
            if name.ends_with(".json")
                || name.ends_with(".json.gz")
                || name.ends_with(".mjai.json")
                || name.ends_with(".mjai.json.gz")
    )
}

fn compact_identity(identity: &str) -> &str {
    identity.rsplit('/').next().unwrap_or(identity)
}

fn compact_error_message(err: &dyn std::fmt::Display) -> &'static str {
    let raw = err.to_string();
    if raw.contains("Replay desync") {
        "replay desync"
    } else if raw.contains("replay observation failed") {
        "replay observation failed"
    } else if raw.contains("replay action conversion failed") {
        "replay action conversion failed"
    } else if raw.contains("hydra action mapping failed") {
        "hydra action mapping failed"
    } else if raw.contains("failed to parse MJAI events") {
        "invalid mjai events"
    } else if raw.contains("failed to load MJAI events") {
        "failed to load mjai events"
    } else if raw.contains("failed to inspect MJAI stream") {
        "failed to inspect mjai stream"
    } else {
        "load error"
    }
}

fn write_shard_header<W: Write>(
    writer: &mut W,
    split: BcShardSplit,
    shard_index: u32,
    sample_count: u64,
    first_sample_index: u64,
    feature_flags: u32,
    record_size: u32,
) -> io::Result<()> {
    writer.write_all(&BC_SHARD_MAGIC)?;
    write_u32_le(writer, BC_SHARD_VERSION)?;
    write_u32_le(writer, BC_SHARD_HEADER_SIZE)?;
    write_u32_le(writer, record_size)?;
    write_u32_le(writer, split.split_id())?;
    write_u32_le(writer, shard_index)?;
    write_u64_le(writer, sample_count)?;
    write_u32_le(writer, NUM_CHANNELS as u32)?;
    write_u32_le(writer, TILE_COUNT as u32)?;
    write_u32_le(writer, HYDRA_ACTION_SPACE as u32)?;
    write_u64_le(writer, first_sample_index)?;
    write_u32_le(writer, feature_flags)?;
    write_u32_le(writer, 0)?;
    write_u64_le(writer, 0)?;
    write_u64_le(writer, 0)?;
    Ok(())
}

fn write_sample_record<W: Write>(writer: &mut W, sample: &crate::data::sample::MjaiSample, flags: u32) -> io::Result<()> {
    write_obs_f32(writer, &sample.obs)?;
    write_u8(writer, sample.action)?;
    write_mask_u8(writer, &sample.legal_mask)?;
    write_i32_le(writer, sample.score_delta)?;
    write_u8(writer, sample.grp_label)?;
    write_optional_oracle_f32(writer, sample.oracle_target.as_ref())?;
    write_u8(writer, u8::from(sample.oracle_target.is_some()))?;
    write_binary_triplet(writer, &sample.tenpai)?;
    writer.write_all(&sample.opp_next)?;
    write_binary_mask_u8(writer, &sample.danger)?;
    write_binary_mask_u8(writer, &sample.danger_mask)?;

    if flags & FLAG_SAFETY_RESIDUAL != 0 {
        write_optional_action_f32(writer, sample.safety_residual.as_ref())?;
        write_optional_action_mask_u8(writer, sample.safety_residual_mask.as_ref())?;
    }
    if flags & FLAG_EXIT != 0 {
        write_optional_action_f32(writer, sample.exit_target.as_ref())?;
        write_optional_action_mask_u8(writer, sample.exit_mask.as_ref())?;
    }
    if flags & FLAG_DELTA_Q != 0 {
        write_optional_action_f32(writer, sample.delta_q_target.as_ref())?;
        write_optional_action_mask_u8(writer, sample.delta_q_mask.as_ref())?;
    }
    Ok(())
}

fn verify_shard_header(mmap: &Mmap, split: BcShardSplit, feature_flags: u32, record_size: u32) -> Result<(), String> {
    if mmap.len() < BC_SHARD_HEADER_SIZE as usize {
        return Err("BC shard file too small for header".to_string());
    }
    if mmap[..8] != BC_SHARD_MAGIC {
        return Err("invalid BC shard magic".to_string());
    }
    let version = read_u32_le(&mmap[8..12]);
    if version != BC_SHARD_VERSION {
        return Err(format!("unsupported BC shard version {version}"));
    }
    let split_id = read_u32_le(&mmap[20..24]);
    if split_id != split.split_id() {
        return Err("BC shard split mismatch".to_string());
    }
    let header_record_size = read_u32_le(&mmap[16..20]);
    if header_record_size != record_size {
        return Err("BC shard record size mismatch".to_string());
    }
    let header_flags = read_u32_le(&mmap[56..60]);
    if header_flags != feature_flags {
        return Err("BC shard feature flags mismatch".to_string());
    }
    let expected_record_size = record_size_for_flags(feature_flags);
    if record_size != expected_record_size {
        return Err(format!(
            "BC shard record_size {record_size} incompatible with current binary \
             (expected {expected_record_size} for flags {feature_flags:#x}). \
             Rebuild shards with the current encoder.",
        ));
    }
    Ok(())
}

fn write_obs_f32<W: Write>(writer: &mut W, values: &[f32; OBS_SIZE]) -> io::Result<()> {
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_mask_u8<W: Write>(writer: &mut W, values: &[f32; HYDRA_ACTION_SPACE]) -> io::Result<()> {
    for &value in values {
        writer.write_all(&[u8::from(value > 0.0)])?;
    }
    Ok(())
}

fn write_binary_triplet<W: Write>(writer: &mut W, values: &[f32; OPPONENT_COUNT]) -> io::Result<()> {
    for &value in values {
        writer.write_all(&[u8::from(value > 0.0)])?;
    }
    Ok(())
}

fn write_optional_oracle_f32<W: Write>(writer: &mut W, values: Option<&[f32; PLAYER_COUNT]>) -> io::Result<()> {
    if let Some(values) = values {
        for &value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
    } else {
        write_zero_bytes(writer, ORACLE_FLOAT32_BYTES)?;
    }
    Ok(())
}

fn write_binary_mask_u8<W: Write>(writer: &mut W, values: &[f32; SPATIAL_TARGET_SIZE]) -> io::Result<()> {
    for &value in values {
        writer.write_all(&[u8::from(value > 0.0)])?;
    }
    Ok(())
}

fn write_optional_action_f32<W: Write>(writer: &mut W, values: Option<&[f32; HYDRA_ACTION_SPACE]>) -> io::Result<()> {
    if let Some(values) = values {
        for &value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
    } else {
        write_zero_bytes(writer, OPTIONAL_ACTION_FLOAT32_BYTES)?;
    }
    Ok(())
}

fn write_optional_action_mask_u8<W: Write>(writer: &mut W, values: Option<&[f32; HYDRA_ACTION_SPACE]>) -> io::Result<()> {
    if let Some(values) = values {
        for &value in values {
            writer.write_all(&[u8::from(value > 0.0)])?;
        }
    } else {
        write_zero_bytes(writer, OPTIONAL_ACTION_MASK_BYTES)?;
    }
    Ok(())
}

fn write_zero_bytes<W: Write>(writer: &mut W, total: usize) -> io::Result<()> {
    const ZERO_CHUNK: [u8; 4096] = [0u8; 4096];
    let mut remaining = total;
    while remaining > 0 {
        let chunk = remaining.min(ZERO_CHUNK.len());
        writer.write_all(&ZERO_CHUNK[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

fn write_u8<W: Write>(writer: &mut W, value: u8) -> io::Result<()> {
    writer.write_all(&[value])
}

fn write_u32_le<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64_le<W: Write>(writer: &mut W, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_i32_le<W: Write>(writer: &mut W, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes[0..4].try_into().expect("u32 slice"))
}

fn read_i32_le(bytes: &[u8]) -> i32 {
    i32::from_le_bytes(bytes[0..4].try_into().expect("i32 slice"))
}

#[cfg(not(target_endian = "little"))]
fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes[0..4].try_into().expect("f32 slice"))
}

fn read_f32_array<const N: usize>(bytes: &[u8]) -> [f32; N] {
    debug_assert_eq!(bytes.len(), N * std::mem::size_of::<f32>());
    #[cfg(target_endian = "little")]
    {
        let mut out = MaybeUninit::<[f32; N]>::uninit();
        unsafe {
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                out.as_mut_ptr().cast::<u8>(),
                N * std::mem::size_of::<f32>(),
            );
            out.assume_init()
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut out = [0.0f32; N];
        for (value, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
            *value = f32::from_le_bytes(chunk.try_into().expect("f32 chunk"));
        }
        out
    }
}

fn read_optional_action_f32(bytes: &[u8]) -> Option<[f32; HYDRA_ACTION_SPACE]> {
    debug_assert!(bytes.len() >= HYDRA_ACTION_SPACE * 4);
    if !any_nonzero_u8(&bytes[..HYDRA_ACTION_SPACE * 4]) {
        return None;
    }
    Some(read_f32_array::<HYDRA_ACTION_SPACE>(bytes))
}

#[inline]
fn any_nonzero_u8(bytes: &[u8]) -> bool {
    let (chunks, tail) = bytes.as_chunks::<8>();
    let mut acc = 0u64;
    for chunk in chunks {
        acc |= u64::from_ne_bytes(*chunk);
    }
    for &b in tail {
        acc |= b as u64;
    }
    acc != 0
}

/// Copy f32 action values directly from mmap bytes into `dst`, skipping
/// the stack intermediate.  Returns `true` if any values were nonzero.
#[inline]
fn read_optional_action_f32_into(bytes: &[u8], dst: &mut [f32]) -> bool {
    debug_assert!(bytes.len() >= HYDRA_ACTION_SPACE * 4);
    debug_assert_eq!(dst.len(), HYDRA_ACTION_SPACE);
    let region = &bytes[..HYDRA_ACTION_SPACE * 4];
    if !any_nonzero_u8(region) {
        return false;
    }
    #[cfg(target_endian = "little")]
    unsafe {
        ptr::copy_nonoverlapping(
            region.as_ptr(),
            dst.as_mut_ptr().cast::<u8>(),
            HYDRA_ACTION_SPACE * 4,
        );
    }
    #[cfg(not(target_endian = "little"))]
    for (d, chunk) in dst.iter_mut().zip(region.chunks_exact(4)) {
        *d = f32::from_le_bytes(chunk.try_into().expect("f32 chunk"));
    }
    true
}

/// Expand u8 mask bytes to f32 directly into `dst`.  Returns `true` if
/// any source byte was nonzero.
#[inline]
fn read_optional_action_mask_f32_into(bytes: &[u8], dst: &mut [f32]) -> bool {
    debug_assert!(bytes.len() >= HYDRA_ACTION_SPACE);
    debug_assert_eq!(dst.len(), HYDRA_ACTION_SPACE);
    if !any_nonzero_u8(&bytes[..HYDRA_ACTION_SPACE]) {
        return false;
    }
    for (d, &src) in dst.iter_mut().zip(bytes[..HYDRA_ACTION_SPACE].iter()) {
        *d = src.min(1) as f32;
    }
    true
}

/// Expand u8 mask bytes to f32 with suit permutation, writing directly
/// into `dst`.  Fuses the decode and scatter passes.
#[inline]
fn expand_and_augment_mask_into(
    bytes: &[u8],
    action_perm: &[usize; 37],
    dst: &mut [f32],
) -> bool {
    debug_assert!(bytes.len() >= HYDRA_ACTION_SPACE);
    debug_assert_eq!(dst.len(), HYDRA_ACTION_SPACE);
    if !any_nonzero_u8(&bytes[..HYDRA_ACTION_SPACE]) {
        return false;
    }
    for i in 0..37usize {
        dst[action_perm[i]] = bytes[i].min(1) as f32;
    }
    dst[37..HYDRA_ACTION_SPACE].iter_mut()
        .zip(bytes[37..HYDRA_ACTION_SPACE].iter())
        .for_each(|(d, &src)| *d = src.min(1) as f32);
    true
}

#[cfg(not(target_endian = "little"))]
fn read_oracle_f32(bytes: &[u8]) -> [f32; PLAYER_COUNT] {
    read_f32_array::<PLAYER_COUNT>(bytes)
}

fn policy_target_from_actions<B: Backend>(
    actions: Tensor<B, 1, Int>,
    batch_size: usize,
) -> Tensor<B, 2> {
    actions
        .one_hot::<2>(HYDRA_ACTION_SPACE)
        .reshape([batch_size, HYDRA_ACTION_SPACE])
        .float()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_core::action::HYDRA_ACTION_SPACE;
    use std::fs;

    use burn::backend::NdArray;

    use crate::data::sample::collate_samples_bc_owned;

    fn dummy_sample() -> crate::data::sample::MjaiSample {
        let mut legal_mask = [0.0; HYDRA_ACTION_SPACE];
        legal_mask[3] = 1.0;
        let mut safety = [0.0; HYDRA_ACTION_SPACE];
        safety[1] = 0.5;
        let mut safety_mask = [0.0; HYDRA_ACTION_SPACE];
        safety_mask[1] = 1.0;
        let mut exit_target = [0.0; HYDRA_ACTION_SPACE];
        exit_target[3] = 0.75;
        let mut exit_mask = [0.0; HYDRA_ACTION_SPACE];
        exit_mask[3] = 1.0;
        crate::data::sample::MjaiSample {
            obs: [0.25; OBS_SIZE],
            action: 3,
            legal_mask,
            placement: 1,
            score_delta: 1200,
            grp_label: 7,
            oracle_target: Some([0.1, 0.2, 0.3, 0.4]),
            tenpai: [1.0, 0.0, 1.0],
            opp_next: [3, 8, 255],
            danger: [0.0; SPATIAL_TARGET_SIZE],
            danger_mask: [1.0; SPATIAL_TARGET_SIZE],
            safety_residual: Some(safety),
            safety_residual_mask: Some(safety_mask),
            exit_target: Some(exit_target),
            exit_mask: Some(exit_mask),
            delta_q_target: None,
            delta_q_mask: None,
            belief_fields: Some([0.0; 16 * 34]),
            mixture_weights: Some([0.25; 4]),
            belief_fields_present: true,
            mixture_weights_present: true,
        }
    }

    fn tiny_real_mjai_replay() -> String {
        [
            r#"{"type":"start_game","names":["a","b","c","d"],"id":"game-1"}"#,
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
            r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
            r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
            r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
            r#"{"type":"ryukyoku"}"#,
            r#"{"type":"end_kyoku"}"#,
        ]
        .join("\n")
    }

    #[test]
    fn compact_header_size_constant_matches_written_bytes() {
        let mut bytes = Vec::new();
        write_shard_header(&mut bytes, BcShardSplit::Train, 2, 10, 100, FLAG_SAFETY_RESIDUAL, record_size_for_flags(FLAG_SAFETY_RESIDUAL))
            .expect("header write should succeed");
        assert_eq!(bytes.len(), BC_SHARD_HEADER_SIZE as usize);
    }

    #[test]
    fn compact_record_size_constant_matches_written_bytes() {
        let sample = dummy_sample();
        let flags = FLAG_SAFETY_RESIDUAL | FLAG_EXIT;
        let mut bytes = Vec::new();
        write_sample_record(&mut bytes, &sample, flags).expect("sample write should succeed");
        assert_eq!(bytes.len(), record_size_for_flags(flags) as usize);
    }

    #[test]
    fn shard_collation_matches_raw_collation_on_real_replay_fixture() {
        type B = NdArray<f32>;

        let root = std::env::temp_dir().join(format!(
            "bc-shard-compare-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time after epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("temp dir should be creatable");
        let replay_path = root.join("game.mjai.json");
        fs::write(&replay_path, tiny_real_mjai_replay()).expect("fixture should write");

        let raw_game = load_game_from_path(&replay_path).expect("raw game should load");
        let raw_samples = raw_game.samples;
        assert!(!raw_samples.is_empty(), "fixture should produce samples");

        let shard_dir = root.join("shards");
        let build = build_bc_shards(&BuildBcShardsConfig {
            input: replay_path.clone(),
            output_dir: shard_dir.clone(),
            manifest_name: "manifest.json".into(),
            train_fraction: 1.0,
            shard_samples: 10_000,
            split_mode: BcShardSplitMode::Train,
            exit_sidecar: None,
            exit_sidecar_path: None,
            exit_provenance: SidecarProvenance::default(),
            delta_q_sidecar: None,
            delta_q_sidecar_path: None,
            delta_q_provenance: SidecarProvenance::default(),
        })
        .expect("shards should build");
        let reader = load_bc_shard_reader(&build.manifest_path, BcShardSplit::Train)
            .expect("reader should load");
        let device = Default::default();
        let raw = collate_samples_bc_owned::<B>(&raw_samples, false, &device)
            .expect("raw collate should succeed")
            .expect("raw batch should exist");
        let indices: Vec<usize> = (0..raw_samples.len()).collect();
        let shard = reader
            .collate_batch::<B>(&indices, false, &device)
            .expect("shard collate should succeed");

        let (raw_obs, raw_batch, raw_targets) = raw;

        fn assert_tensor_close<const D: usize>(
            lhs: Tensor<NdArray<f32>, D>,
            rhs: Tensor<NdArray<f32>, D>,
            name: &str,
        ) {
            let lhs_data = lhs.into_data();
            let rhs_data = rhs.into_data();
            let lhs_slice = lhs_data.as_slice::<f32>().expect("lhs f32");
            let rhs_slice = rhs_data.as_slice::<f32>().expect("rhs f32");
            assert_eq!(lhs_slice.len(), rhs_slice.len(), "{name} len");
            for (idx, (a, b)) in lhs_slice.iter().zip(rhs_slice.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "{name}[{idx}] mismatch: {a} vs {b}"
                );
            }
        }

        assert_tensor_close(raw_obs, shard.obs, "obs");
        assert_tensor_close(raw_batch.actions.clone().float(), shard.batch.actions.clone().float(), "actions");
        assert_tensor_close(raw_targets.legal_mask.clone(), shard.targets.legal_mask.clone(), "legal_mask");
        assert_tensor_close(raw_targets.value_target.clone(), shard.targets.value_target.clone(), "value_target");
        assert_tensor_close(raw_targets.grp_target.clone(), shard.targets.grp_target.clone(), "grp_target");
        assert_tensor_close(raw_targets.tenpai_target.clone(), shard.targets.tenpai_target.clone(), "tenpai");
        assert_tensor_close(raw_targets.danger_target.clone(), shard.targets.danger_target.clone(), "danger");
        assert_tensor_close(raw_targets.danger_mask.clone(), shard.targets.danger_mask.clone(), "danger_mask");
        assert_tensor_close(raw_targets.opp_next_target.clone(), shard.targets.opp_next_target.clone(), "opp_next");
        assert_tensor_close(raw_targets.score_pdf_target.clone(), shard.targets.score_pdf_target.clone(), "score_pdf");
        assert_tensor_close(raw_targets.score_cdf_target.clone(), shard.targets.score_cdf_target.clone(), "score_cdf");

        let raw_presence = raw_targets.target_presence.expect("raw target presence");
        let shard_presence = shard.targets.target_presence.expect("shard target presence");
        assert_eq!(raw_presence.batch_size, shard_presence.batch_size);
        assert_eq!(raw_presence.delta_q_actions_present, shard_presence.delta_q_actions_present);
        assert_eq!(raw_presence.counts, shard_presence.counts);

        let _ = fs::remove_dir_all(root);
    }
}
