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

use crate::data::archive_helpers::{
    compact_error_message, compact_identity, identity_for_archive_entry, is_mjai_archive_entry,
    is_tar_zst_file,
};
#[cfg(not(target_endian = "little"))]
use crate::data::augment::augment_action_vector_suit_into;
use crate::data::augment::{augment_obs_suit_from_le_bytes, permutation_tables};
use crate::data::mjai_loader::{
    MjaiGame, ReplayLoadPolicy, SidecarProvenance, invalid_data, load_game_from_path_with_policy,
    load_game_from_stream_with_policy,
};
use crate::data::pipeline::{
    DataManifest, DataSource, identity_for_loose_file, is_train_game, scan_data_sources,
};
use crate::data::sample::{MjaiBcBatch, score_delta_to_bin, score_delta_to_value};
use crate::training::head_gates::{AdvancedHead, TargetPresence};
use crate::training::losses::HydraTargets;
use crate::training::replay_delta_q::DeltaQSidecarIndex;
use crate::training::replay_exit::ExitSidecarIndex;

pub use hydra_bc_shards::{
    BcShardManifest as ExtractedBcShardManifest, BcShardSplit as ExtractedBcShardSplit,
    validate_bc_shard_manifest_contract as validate_extracted_bc_shard_manifest_contract,
    validate_bc_shard_split_manifest_contract as validate_extracted_bc_shard_split_manifest_contract,
};

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

/// IEEE 754 bit pattern for `1.0f32`. Used with `f32::from_bits(mask * F32_ONE_BITS)`
/// to convert u8 nonzero checks to 0.0/1.0 without the `vcvtdq2ps` float conversion
/// that `byte.min(1) as f32` requires. LLVM lowers this to `vpmovzxbd` + `vpcmpgtd`
/// + `vpand` (3 instructions per 8 elements, vs 5 with the int-to-float path).
const F32_ONE_BITS: u32 = 0x3F80_0000;

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
        matches!(
            (self, split),
            (Self::Both, _)
                | (Self::Train, BcShardSplit::Train)
                | (Self::Validation, BcShardSplit::Validation)
        )
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
    pub source_manifest: Option<DataManifest>,
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
            source_manifest: None,
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
            target_presence: std::mem::take(&mut self.target_presence),
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
        recycled.target_presence = std::mem::take(&mut self.target_presence);
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

/// # Safety
/// Caller guarantees every element in `0..len` will be written before read.
#[inline]
fn resize_uninit_i64(buf: &mut Vec<i64>, len: usize) {
    buf.clear();
    if buf.capacity() < len {
        buf.reserve(len);
    }
    unsafe { buf.set_len(len) };
}

impl BcShardHostBatch {
    /// Materialize device tensors from CPU-side flat buffers.
    ///
    /// This is the only step that touches the `Backend` / device.
    pub fn materialize<B: Backend>(&self, device: &B::Device) -> BcShardBatch<B> {
        let batch = self.batch_size;

        let obs = Tensor::<B, 1>::from_floats(self.obs_flat.as_slice(), device).reshape([
            batch,
            NUM_CHANNELS,
            TILE_COUNT,
        ]);
        let actions_tensor = Tensor::<B, 1, Int>::from_ints(self.actions.as_slice(), device);
        let legal_mask = Tensor::<B, 1>::from_floats(self.legal_mask_flat.as_slice(), device)
            .reshape([batch, HYDRA_ACTION_SPACE]);
        let value_target = Tensor::<B, 1>::from_floats(self.value_target.as_slice(), device);
        let grp_target = Tensor::<B, 1>::from_floats(self.grp_target_flat.as_slice(), device)
            .reshape([batch, GRP_CLASS_COUNT]);
        let oracle_target = Tensor::<B, 1>::from_floats(self.oracle_target_flat.as_slice(), device)
            .reshape([batch, PLAYER_COUNT]);
        let oracle_target_mask =
            Tensor::<B, 1>::from_floats(self.oracle_target_mask.as_slice(), device);
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

        let policy_target =
            policy_target_from_action_slice::<B>(self.actions.as_slice(), batch, device);

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
                Tensor::<B, 1>::from_floats(buf.as_slice(), device)
                    .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            safety_residual_target: self.safety_target_flat.as_ref().map(|buf| {
                Tensor::<B, 1>::from_floats(buf.as_slice(), device)
                    .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            safety_residual_mask: self.safety_mask_flat.as_ref().map(|buf| {
                Tensor::<B, 1>::from_floats(buf.as_slice(), device)
                    .reshape([batch, HYDRA_ACTION_SPACE])
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

    /// Zero-copy variant of [`materialize`](Self::materialize) that
    /// consumes the batch, transferring Vec heap allocations directly
    /// into Burn's `TensorData` via pointer hand-off (no memcpy).
    ///
    /// Eliminates ~2-3 MB of per-batch allocation+copy on the
    /// non-pinned materialization path.
    pub fn materialize_owned<B: Backend>(self, device: &B::Device) -> BcShardBatch<B> {
        let b = self.batch_size;

        let obs = Tensor::<B, 3>::from_data(
            TensorData::new(self.obs_flat, [b, NUM_CHANNELS, TILE_COUNT]),
            device,
        );
        let actions = self.actions;
        let policy_target = policy_target_from_action_slice::<B>(actions.as_slice(), b, device);
        let actions_tensor = Tensor::<B, 1, Int>::from_data(TensorData::new(actions, [b]), device);
        let legal_mask = Tensor::<B, 2>::from_data(
            TensorData::new(self.legal_mask_flat, [b, HYDRA_ACTION_SPACE]),
            device,
        );
        let value_target =
            Tensor::<B, 1>::from_data(TensorData::new(self.value_target, [b]), device);
        let grp_target = Tensor::<B, 2>::from_data(
            TensorData::new(self.grp_target_flat, [b, GRP_CLASS_COUNT]),
            device,
        );
        let oracle_target = Tensor::<B, 2>::from_data(
            TensorData::new(self.oracle_target_flat, [b, PLAYER_COUNT]),
            device,
        );
        let oracle_target_mask =
            Tensor::<B, 1>::from_data(TensorData::new(self.oracle_target_mask, [b]), device);
        let tenpai_target = Tensor::<B, 2>::from_data(
            TensorData::new(self.tenpai_flat, [b, OPPONENT_COUNT]),
            device,
        );
        let danger_target = Tensor::<B, 3>::from_data(
            TensorData::new(self.danger_flat, [b, OPPONENT_COUNT, TILE_COUNT]),
            device,
        );
        let danger_mask = Tensor::<B, 3>::from_data(
            TensorData::new(self.danger_mask_flat, [b, OPPONENT_COUNT, TILE_COUNT]),
            device,
        );
        let opp_next_target = Tensor::<B, 3>::from_data(
            TensorData::new(self.opp_next_flat, [b, OPPONENT_COUNT, TILE_COUNT]),
            device,
        );
        let score_pdf_target = Tensor::<B, 2>::from_data(
            TensorData::new(self.score_pdf_flat, [b, SCORE_BINS]),
            device,
        );
        let score_cdf_target = Tensor::<B, 2>::from_data(
            TensorData::new(self.score_cdf_flat, [b, SCORE_BINS]),
            device,
        );

        let exit_target_tensor = self.exit_target_flat.map(|buf| {
            Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
        });
        let exit_mask_tensor = self.exit_mask_flat.map(|buf| {
            Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
        });

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
            delta_q_target: self.delta_q_target_flat.map(|buf| {
                Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
            }),
            delta_q_mask: self.delta_q_mask_flat.map(|buf| {
                Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
            }),
            safety_residual_target: self.safety_target_flat.map(|buf| {
                Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
            }),
            safety_residual_mask: self.safety_mask_flat.map(|buf| {
                Tensor::<B, 2>::from_data(TensorData::new(buf, [b, HYDRA_ACTION_SPACE]), device)
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
    let source_manifest = match &config.source_manifest {
        Some(manifest) => manifest.clone(),
        None => scan_data_sources(&config.input)?,
    };
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
            DataSource::ParsedSampleCache { path, .. } => {
                return Err(invalid_data(format!(
                    "parsed-sample cache input is not supported by build_bc_shards yet: {}",
                    path.display()
                )));
            }
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

pub fn read_bc_shard_manifest(manifest_path: &Path) -> Result<BcShardManifest, String> {
    let raw = fs::read_to_string(manifest_path).map_err(|err| {
        format!(
            "failed to read BC shard manifest {}: {err}",
            manifest_path.display()
        )
    })?;
    let manifest: BcShardManifest = serde_json::from_str(&raw).map_err(|err| {
        format!(
            "failed to parse BC shard manifest {}: {err}",
            manifest_path.display()
        )
    })?;
    validate_bc_shard_manifest_contract(&manifest)?;
    Ok(manifest)
}

fn validate_bc_shard_manifest_contract(manifest: &BcShardManifest) -> Result<(), String> {
    if manifest.obs_size != OBS_SIZE {
        return Err(format!(
            "BC shard manifest obs_size {} does not match current OBS_SIZE {} \
             (num_channels: manifest={}, binary={}). \
             Shards must be rebuilt with the current encoder.",
            manifest.obs_size, OBS_SIZE, manifest.num_channels, NUM_CHANNELS,
        ));
    }
    if manifest.base_record_size != BC_BASE_RECORD_SIZE {
        return Err(format!(
            "BC shard manifest base_record_size {} does not match current \
             BC_BASE_RECORD_SIZE {}. Shards must be rebuilt with the current encoder.",
            manifest.base_record_size, BC_BASE_RECORD_SIZE,
        ));
    }
    if manifest.action_space != HYDRA_ACTION_SPACE {
        return Err(format!(
            "BC shard manifest action_space {} does not match current HYDRA_ACTION_SPACE {}. \
             Shards must be rebuilt with the current action contract.",
            manifest.action_space, HYDRA_ACTION_SPACE,
        ));
    }
    let mut total_samples = 0u64;
    let mut total_shards = 0usize;
    for split in &manifest.splits {
        validate_bc_shard_split_manifest_contract(split)?;
        total_samples += split.sample_count;
        total_shards += split.shard_count;
    }
    if manifest.totals.sample_count != total_samples {
        return Err(format!(
            "BC shard manifest totals.sample_count {} does not match split total {}",
            manifest.totals.sample_count, total_samples
        ));
    }
    if manifest.totals.shard_count != total_shards {
        return Err(format!(
            "BC shard manifest totals.shard_count {} does not match split shard total {}",
            manifest.totals.shard_count, total_shards
        ));
    }
    Ok(())
}

fn validate_bc_shard_split_manifest_contract(split: &BcShardSplitManifest) -> Result<(), String> {
    if split.shard_count != split.shards.len() {
        return Err(format!(
            "BC shard manifest {:?} shard_count {} does not match descriptor count {}",
            split.split,
            split.shard_count,
            split.shards.len()
        ));
    }
    let mut expected_start = 0u64;
    for (idx, shard) in split.shards.iter().enumerate() {
        if shard.split != split.split {
            return Err(format!(
                "BC shard descriptor {} has split {:?}, expected {:?}",
                idx, shard.split, split.split
            ));
        }
        if shard.shard_index != idx {
            return Err(format!(
                "BC shard descriptor for {:?} has shard_index {}, expected {}",
                split.split, shard.shard_index, idx
            ));
        }
        if shard.first_sample_index != expected_start {
            return Err(format!(
                "BC shard descriptor {} for {:?} starts at {}, expected contiguous start {}",
                idx, split.split, shard.first_sample_index, expected_start
            ));
        }
        if shard.feature_flags != split.feature_flags {
            return Err(format!(
                "BC shard descriptor {} for {:?} feature_flags {} does not match split feature_flags {}",
                idx, split.split, shard.feature_flags, split.feature_flags
            ));
        }
        if shard.record_size != split.record_size {
            return Err(format!(
                "BC shard descriptor {} for {:?} record_size {} does not match split record_size {}",
                idx, split.split, shard.record_size, split.record_size
            ));
        }
        expected_start = expected_start
            .checked_add(shard.sample_count)
            .ok_or_else(|| "BC shard split sample_count overflow".to_string())?;
    }
    if split.sample_count != expected_start {
        return Err(format!(
            "BC shard split {:?} sample_count {} does not match descriptor total {}",
            split.split, split.sample_count, expected_start
        ));
    }
    Ok(())
}

pub fn load_bc_shard_reader(
    manifest_path: &Path,
    split: BcShardSplit,
) -> Result<BcShardReader, String> {
    let manifest = read_bc_shard_manifest(manifest_path)?;

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
        self.shards
            .iter()
            .map(|shard| shard.sample_count as usize)
            .sum()
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
            .map(|host| host.materialize_owned(device))
    }

    pub fn collate_batch_range<B: Backend>(
        &self,
        start: usize,
        len: usize,
        augment: bool,
        device: &B::Device,
    ) -> Result<BcShardBatch<B>, String> {
        self.collate_host_batch_range(start, len, augment)
            .map(|host| host.materialize_owned(device))
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

        if augment {
            for (row, &sample_index) in indices.iter().enumerate() {
                let (shard, offset) = self.locate(sample_index)?;
                write_augmented_row_into_scratch(shard, offset, row, sample_index, scratch)?;
            }
        } else {
            for (row, &sample_index) in indices.iter().enumerate() {
                let (shard, offset) = self.locate(sample_index)?;
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

        // Hoist the augment branch outside the loop so the branch
        // predictor never sees the per-row condition, and LLVM can
        // specialize each loop body independently (different inlining
        // thresholds, different register pressure).
        macro_rules! collate_range_loop {
            ($write_row:expr) => {
                for row in 0..len {
                    let sample_index = start + row;
                    let shard = &self.shards[shard_index];

                    // Prefetch next record into L2 while processing current.
                    #[cfg(target_arch = "x86_64")]
                    {
                        let next_offset = if offset + 1 < shard.sample_count as usize {
                            offset + 1
                        } else {
                            0
                        };
                        let next_shard = if next_offset == 0 && shard_index + 1 < self.shards.len()
                        {
                            &self.shards[shard_index + 1]
                        } else {
                            shard
                        };
                        if row + 1 < len {
                            let next_start = BC_SHARD_HEADER_SIZE as usize
                                + next_offset * next_shard.record_size;
                            let ptr = next_shard.mmap.as_ptr().wrapping_add(next_start);
                            unsafe {
                                use std::arch::x86_64::{_MM_HINT_T1, _mm_prefetch};
                                _mm_prefetch::<{ _MM_HINT_T1 }>(ptr.cast());
                                _mm_prefetch::<{ _MM_HINT_T1 }>(ptr.add(64).cast());
                                _mm_prefetch::<{ _MM_HINT_T1 }>(ptr.add(128).cast());
                                _mm_prefetch::<{ _MM_HINT_T1 }>(ptr.add(192).cast());
                            }
                        }
                    }

                    #[allow(
                        clippy::redundant_closure_call,
                        reason = "macro keeps the row writer generic"
                    )]
                    ($write_row)(shard, offset, row, sample_index, scratch)?;

                    offset += 1;
                    if offset == shard.sample_count as usize && row + 1 < len {
                        shard_index += 1;
                        if shard_index >= self.shards.len() {
                            return Err("BC shard range collation ran past shard list".to_string());
                        }
                        offset = 0;
                    }
                }
            };
        }

        if augment {
            collate_range_loop!(
                |shard: &ShardMap,
                 offset: usize,
                 row: usize,
                 sample_index: usize,
                 scratch: &mut BcShardHostScratch| {
                    write_augmented_row_into_scratch(shard, offset, row, sample_index, scratch)
                }
            );
        } else {
            collate_range_loop!(
                |shard: &ShardMap,
                 offset: usize,
                 row: usize,
                 _sample_index: usize,
                 scratch: &mut BcShardHostScratch| {
                    write_unaugmented_row_into_scratch(shard, offset, row, scratch)
                }
            );
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
            return Err(format!(
                "BC shard sample index {sample_index} out of bounds"
            ));
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

    // Prefetch the auxiliary fields that follow the 26KB obs blob. They sit
    // on a different page and would otherwise incur a TLB miss when reached.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let aux_ptr = bytes.as_ptr().add(OBS_F32_BYTES);
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(aux_ptr.cast());
    }

    // SAFETY: `row` is bounded by the caller's enumeration over `indices[0..batch_size]`.
    // The debug_assert below verifies this invariant in debug builds. All scratch buffers
    // are allocated with capacity `batch_size * FIELD_SIZE` in BcShardHostScratch::new/reset.
    // The `bytes` slice length equals `shard.record_size`, validated by the mmap bounds
    // check above and the shard header verification at load time. Cursor advances through
    // a fixed sequence of fields whose sizes sum exactly to `record_size`.
    debug_assert_eq!(bytes.len(), shard.record_size, "record size mismatch");
    debug_assert!(
        row < scratch.batch_size,
        "row {row} >= batch_size {}",
        scratch.batch_size
    );

    let obs_row = unsafe {
        scratch
            .obs_flat
            .get_unchecked_mut(row * OBS_SIZE..(row + 1) * OBS_SIZE)
    };
    let mask_row = unsafe {
        scratch
            .legal_mask_flat
            .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
    };
    let pdf_row = unsafe {
        scratch
            .score_pdf_flat
            .get_unchecked_mut(row * SCORE_BINS..(row + 1) * SCORE_BINS)
    };
    let cdf_row = unsafe {
        scratch
            .score_cdf_flat
            .get_unchecked_mut(row * SCORE_BINS..(row + 1) * SCORE_BINS)
    };
    let grp_row = unsafe {
        scratch
            .grp_target_flat
            .get_unchecked_mut(row * GRP_CLASS_COUNT..(row + 1) * GRP_CLASS_COUNT)
    };
    let oracle_row = unsafe {
        scratch
            .oracle_target_flat
            .get_unchecked_mut(row * PLAYER_COUNT..(row + 1) * PLAYER_COUNT)
    };
    let tenpai_row = unsafe {
        scratch
            .tenpai_flat
            .get_unchecked_mut(row * OPPONENT_COUNT..(row + 1) * OPPONENT_COUNT)
    };
    let opp_next_row = unsafe {
        scratch
            .opp_next_flat
            .get_unchecked_mut(row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE)
    };
    let danger_row = unsafe {
        scratch
            .danger_flat
            .get_unchecked_mut(row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE)
    };
    let dmask_row = unsafe {
        scratch
            .danger_mask_flat
            .get_unchecked_mut(row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE)
    };

    #[cfg(target_endian = "little")]
    unsafe {
        ptr::copy_nonoverlapping(
            bytes.get_unchecked(cursor..cursor + OBS_F32_BYTES).as_ptr(),
            obs_row.as_mut_ptr().cast::<u8>(),
            OBS_F32_BYTES,
        );
    }
    #[cfg(not(target_endian = "little"))]
    for (value, chunk) in obs_row
        .iter_mut()
        .zip(unsafe { bytes.get_unchecked(cursor..cursor + OBS_F32_BYTES) }.chunks_exact(4))
    {
        *value = f32::from_le_bytes(chunk.try_into().expect("f32 chunk"));
    }
    cursor += OBS_F32_BYTES;

    unsafe { *scratch.actions.get_unchecked_mut(row) = *bytes.get_unchecked(cursor) as i64 };
    cursor += 1;

    // IEEE 754 bitmask trick: nonzero u8 -> 1.0f32, zero -> 0.0f32.
    // Avoids vcvtdq2ps; LLVM lowers to vpmovzxbd + vpcmpgtd + vpand.
    for (dst, &src) in mask_row
        .iter_mut()
        .zip(unsafe { bytes.get_unchecked(cursor..cursor + HYDRA_ACTION_SPACE) }.iter())
    {
        *dst = f32::from_bits((src != 0) as u32 * F32_ONE_BITS);
    }
    cursor += HYDRA_ACTION_SPACE;

    let score_delta = read_i32_le(unsafe {
        &*(bytes.get_unchecked(cursor..cursor + 4).as_ptr() as *const [u8; 4])
    });
    cursor += 4;
    unsafe { *scratch.value_target.get_unchecked_mut(row) = score_delta_to_value(score_delta) };
    let bin = score_delta_to_bin(score_delta);
    unsafe { *pdf_row.get_unchecked_mut(bin) = 1.0 };
    unsafe { cdf_row.get_unchecked_mut(bin..) }.fill(1.0);

    let grp = (unsafe { *bytes.get_unchecked(cursor) } as usize).min(GRP_CLASS_COUNT - 1);
    unsafe { *grp_row.get_unchecked_mut(grp) = 1.0 };
    cursor += 1;

    // Direct mmap-to-scratch copy on little-endian; avoids stack intermediate.
    #[cfg(target_endian = "little")]
    {
        let src_bytes = unsafe { bytes.get_unchecked(cursor..cursor + ORACLE_FLOAT32_BYTES) };
        // SAFETY: PLAYER_COUNT * 4 == ORACLE_FLOAT32_BYTES, and f32 has no
        // invalid bit patterns.  Alignment is guaranteed by slice layout.
        unsafe {
            ptr::copy_nonoverlapping(
                src_bytes.as_ptr(),
                oracle_row.as_mut_ptr().cast::<u8>(),
                ORACLE_FLOAT32_BYTES,
            );
        }
    }
    #[cfg(not(target_endian = "little"))]
    oracle_row.copy_from_slice(&read_oracle_f32(unsafe {
        bytes.get_unchecked(cursor..cursor + ORACLE_FLOAT32_BYTES)
    }));
    cursor += ORACLE_FLOAT32_BYTES;
    let oracle_present = unsafe { *bytes.get_unchecked(cursor) > 0 };
    cursor += ORACLE_MASK_BYTES;
    if oracle_present {
        unsafe { *scratch.oracle_target_mask.get_unchecked_mut(row) = 1.0 };
        scratch.target_presence.counts[AdvancedHead::OracleCritic.index()] += 1;
    } else {
        oracle_row.fill(0.0);
    }

    for (dst, &src) in tenpai_row
        .iter_mut()
        .zip(unsafe { bytes.get_unchecked(cursor..cursor + OPPONENT_COUNT) }.iter())
    {
        *dst = f32::from_bits((src != 0) as u32 * F32_ONE_BITS);
    }
    cursor += OPPONENT_COUNT;

    for (opp, &tile) in unsafe { bytes.get_unchecked(cursor..cursor + OPPONENT_COUNT) }
        .iter()
        .enumerate()
    {
        if (tile as usize) < TILE_COUNT {
            unsafe { *opp_next_row.get_unchecked_mut(opp * TILE_COUNT + tile as usize) = 1.0 };
        }
    }
    cursor += OPPONENT_COUNT;

    for (dst, &src) in danger_row
        .iter_mut()
        .zip(unsafe { bytes.get_unchecked(cursor..cursor + SPATIAL_TARGET_SIZE) }.iter())
    {
        *dst = f32::from_bits((src != 0) as u32 * F32_ONE_BITS);
    }
    cursor += SPATIAL_TARGET_SIZE;

    for (dst, &src) in dmask_row
        .iter_mut()
        .zip(unsafe { bytes.get_unchecked(cursor..cursor + SPATIAL_TARGET_SIZE) }.iter())
    {
        *dst = f32::from_bits((src != 0) as u32 * F32_ONE_BITS);
    }
    cursor += SPATIAL_TARGET_SIZE;

    if shard.feature_flags & FLAG_SAFETY_RESIDUAL != 0 {
        let dst = unsafe {
            scratch
                .safety_target_flat
                .as_mut()
                .expect("safety enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        read_optional_action_f32_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES) },
            dst,
        );
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;

        let dst = unsafe {
            scratch
                .safety_mask_flat
                .as_mut()
                .expect("safety enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        if read_optional_action_mask_f32_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_MASK_BYTES) },
            dst,
        ) {
            scratch.target_presence.counts[AdvancedHead::SafetyResidual.index()] += 1;
        }
        cursor += OPTIONAL_ACTION_MASK_BYTES;
    }

    if shard.feature_flags & FLAG_EXIT != 0 {
        let dst = unsafe {
            scratch
                .exit_target_flat
                .as_mut()
                .expect("exit enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        read_optional_action_f32_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES) },
            dst,
        );
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;

        let dst = unsafe {
            scratch
                .exit_mask_flat
                .as_mut()
                .expect("exit enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        read_optional_action_mask_f32_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_MASK_BYTES) },
            dst,
        );
        cursor += OPTIONAL_ACTION_MASK_BYTES;
    }

    if shard.feature_flags & FLAG_DELTA_Q != 0 {
        let dst = unsafe {
            scratch
                .delta_q_target_flat
                .as_mut()
                .expect("delta_q enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        read_optional_action_f32_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES) },
            dst,
        );
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;

        let dst = unsafe {
            scratch
                .delta_q_mask_flat
                .as_mut()
                .expect("delta_q enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        if read_optional_action_mask_f32_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_MASK_BYTES) },
            dst,
        ) {
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

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let aux_ptr = bytes.as_ptr().add(OBS_F32_BYTES);
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(aux_ptr.cast());
    }

    // SAFETY: `row` is bounded by the caller's enumeration over `indices[0..batch_size]`.
    // The debug_assert below verifies this invariant in debug builds. All scratch buffers
    // are allocated with capacity `batch_size * FIELD_SIZE` in BcShardHostScratch::new/reset.
    // The `bytes` slice length equals `shard.record_size`, validated by the mmap bounds
    // check above and the shard header verification at load time. Cursor advances through
    // a fixed sequence of fields whose sizes sum exactly to `record_size`.
    debug_assert_eq!(bytes.len(), shard.record_size, "record size mismatch");
    debug_assert!(
        row < scratch.batch_size,
        "row {row} >= batch_size {}",
        scratch.batch_size
    );

    let obs_row = unsafe {
        scratch
            .obs_flat
            .get_unchecked_mut(row * OBS_SIZE..(row + 1) * OBS_SIZE)
    };
    let mask_row = unsafe {
        scratch
            .legal_mask_flat
            .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
    };
    let pdf_row = unsafe {
        scratch
            .score_pdf_flat
            .get_unchecked_mut(row * SCORE_BINS..(row + 1) * SCORE_BINS)
    };
    let cdf_row = unsafe {
        scratch
            .score_cdf_flat
            .get_unchecked_mut(row * SCORE_BINS..(row + 1) * SCORE_BINS)
    };
    let grp_row = unsafe {
        scratch
            .grp_target_flat
            .get_unchecked_mut(row * GRP_CLASS_COUNT..(row + 1) * GRP_CLASS_COUNT)
    };
    let oracle_row = unsafe {
        scratch
            .oracle_target_flat
            .get_unchecked_mut(row * PLAYER_COUNT..(row + 1) * PLAYER_COUNT)
    };
    let tenpai_row = unsafe {
        scratch
            .tenpai_flat
            .get_unchecked_mut(row * OPPONENT_COUNT..(row + 1) * OPPONENT_COUNT)
    };
    let opp_next_row = unsafe {
        scratch
            .opp_next_flat
            .get_unchecked_mut(row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE)
    };
    let danger_row = unsafe {
        scratch
            .danger_flat
            .get_unchecked_mut(row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE)
    };
    let dmask_row = unsafe {
        scratch
            .danger_mask_flat
            .get_unchecked_mut(row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE)
    };

    let perm_idx = (sample_index + row) % hydra_core::tile::ALL_PERMUTATIONS.len();
    let perm = &hydra_core::tile::ALL_PERMUTATIONS[perm_idx];
    let tables = permutation_tables();
    let tile_perm = &tables.tile_34[perm_idx];
    let action_perm = &tables.action_37[perm_idx];

    augment_obs_suit_from_le_bytes(
        unsafe { bytes.get_unchecked(cursor..cursor + OBS_F32_BYTES) },
        perm,
        obs_row,
    );
    cursor += OBS_F32_BYTES;

    let action = unsafe { *bytes.get_unchecked(cursor) };
    unsafe {
        *scratch.actions.get_unchecked_mut(row) = if action <= 36 {
            *action_perm.get_unchecked(action as usize) as i64
        } else {
            action as i64
        }
    };
    cursor += 1;

    let mask_src = unsafe { bytes.get_unchecked(cursor..cursor + HYDRA_ACTION_SPACE) };
    let mask_dst = mask_row;
    for i in 0..37usize {
        let permuted = unsafe { *action_perm.get_unchecked(i) };
        unsafe {
            *mask_dst.get_unchecked_mut(permuted) =
                f32::from_bits((*mask_src.get_unchecked(i) != 0) as u32 * F32_ONE_BITS);
        }
    }
    for (dst, &src) in unsafe { mask_dst.get_unchecked_mut(37..HYDRA_ACTION_SPACE) }
        .iter_mut()
        .zip(unsafe { mask_src.get_unchecked(37..HYDRA_ACTION_SPACE) }.iter())
    {
        *dst = f32::from_bits((src != 0) as u32 * F32_ONE_BITS);
    }
    cursor += HYDRA_ACTION_SPACE;

    let score_delta = read_i32_le(unsafe {
        &*(bytes.get_unchecked(cursor..cursor + 4).as_ptr() as *const [u8; 4])
    });
    cursor += 4;
    unsafe { *scratch.value_target.get_unchecked_mut(row) = score_delta_to_value(score_delta) };
    let bin = score_delta_to_bin(score_delta);
    unsafe { *pdf_row.get_unchecked_mut(bin) = 1.0 };
    unsafe { cdf_row.get_unchecked_mut(bin..) }.fill(1.0);

    let grp = (unsafe { *bytes.get_unchecked(cursor) } as usize).min(GRP_CLASS_COUNT - 1);
    unsafe { *grp_row.get_unchecked_mut(grp) = 1.0 };
    cursor += 1;

    #[cfg(target_endian = "little")]
    {
        let src_bytes = unsafe { bytes.get_unchecked(cursor..cursor + ORACLE_FLOAT32_BYTES) };
        unsafe {
            ptr::copy_nonoverlapping(
                src_bytes.as_ptr(),
                oracle_row.as_mut_ptr().cast::<u8>(),
                ORACLE_FLOAT32_BYTES,
            );
        }
    }
    #[cfg(not(target_endian = "little"))]
    oracle_row.copy_from_slice(&read_oracle_f32(unsafe {
        bytes.get_unchecked(cursor..cursor + ORACLE_FLOAT32_BYTES)
    }));
    cursor += ORACLE_FLOAT32_BYTES;
    let oracle_present = unsafe { *bytes.get_unchecked(cursor) > 0 };
    cursor += ORACLE_MASK_BYTES;
    if oracle_present {
        unsafe { *scratch.oracle_target_mask.get_unchecked_mut(row) = 1.0 };
        scratch.target_presence.counts[AdvancedHead::OracleCritic.index()] += 1;
    } else {
        oracle_row.fill(0.0);
    }

    for (dst, &src) in tenpai_row
        .iter_mut()
        .zip(unsafe { bytes.get_unchecked(cursor..cursor + OPPONENT_COUNT) }.iter())
    {
        *dst = f32::from_bits((src != 0) as u32 * F32_ONE_BITS);
    }
    cursor += OPPONENT_COUNT;

    for (opp, &tile) in unsafe { bytes.get_unchecked(cursor..cursor + OPPONENT_COUNT) }
        .iter()
        .enumerate()
    {
        let permuted = if tile < 34 {
            unsafe { *tile_perm.get_unchecked(tile as usize) }
        } else {
            tile as usize
        };
        if permuted < TILE_COUNT {
            unsafe { *opp_next_row.get_unchecked_mut(opp * TILE_COUNT + permuted) = 1.0 };
        }
    }
    cursor += OPPONENT_COUNT;

    let danger_dst = danger_row;
    const SUIT_TILES: usize = 9;
    const HONOR_START: usize = 27;
    const HONOR_COUNT: usize = TILE_COUNT - HONOR_START;
    for opp in 0..OPPONENT_COUNT {
        let src_start = cursor + opp * TILE_COUNT;
        let dst_start = opp * TILE_COUNT;
        for (src_suit, dst_suit) in perm.iter().copied().enumerate().take(3usize) {
            let dst_suit = dst_suit as usize;
            for t in 0..SUIT_TILES {
                unsafe {
                    *danger_dst.get_unchecked_mut(dst_start + dst_suit * SUIT_TILES + t) =
                        f32::from_bits(
                            (*bytes.get_unchecked(src_start + src_suit * SUIT_TILES + t) != 0)
                                as u32
                                * F32_ONE_BITS,
                        );
                }
            }
        }
        for t in 0..HONOR_COUNT {
            unsafe {
                *danger_dst.get_unchecked_mut(dst_start + HONOR_START + t) = f32::from_bits(
                    (*bytes.get_unchecked(src_start + HONOR_START + t) != 0) as u32 * F32_ONE_BITS,
                );
            }
        }
    }
    cursor += SPATIAL_TARGET_SIZE;

    let dmask_dst = dmask_row;
    for opp in 0..OPPONENT_COUNT {
        let src_start = cursor + opp * TILE_COUNT;
        let dst_start = opp * TILE_COUNT;
        for (src_suit, dst_suit) in perm.iter().copied().enumerate().take(3usize) {
            let dst_suit = dst_suit as usize;
            for t in 0..SUIT_TILES {
                unsafe {
                    *dmask_dst.get_unchecked_mut(dst_start + dst_suit * SUIT_TILES + t) =
                        f32::from_bits(
                            (*bytes.get_unchecked(src_start + src_suit * SUIT_TILES + t) != 0)
                                as u32
                                * F32_ONE_BITS,
                        );
                }
            }
        }
        for t in 0..HONOR_COUNT {
            unsafe {
                *dmask_dst.get_unchecked_mut(dst_start + HONOR_START + t) = f32::from_bits(
                    (*bytes.get_unchecked(src_start + HONOR_START + t) != 0) as u32 * F32_ONE_BITS,
                );
            }
        }
    }
    cursor += SPATIAL_TARGET_SIZE;

    if shard.feature_flags & FLAG_SAFETY_RESIDUAL != 0 {
        let dst = unsafe {
            scratch
                .safety_target_flat
                .as_mut()
                .expect("safety enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        augment_action_f32_from_bytes_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES) },
            action_perm,
            dst,
        );
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;
        let dst = unsafe {
            scratch
                .safety_mask_flat
                .as_mut()
                .expect("safety enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        if expand_and_augment_mask_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_MASK_BYTES) },
            action_perm,
            dst,
        ) {
            scratch.target_presence.counts[AdvancedHead::SafetyResidual.index()] += 1;
        }
        cursor += OPTIONAL_ACTION_MASK_BYTES;
    }

    if shard.feature_flags & FLAG_EXIT != 0 {
        let dst = unsafe {
            scratch
                .exit_target_flat
                .as_mut()
                .expect("exit enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        augment_action_f32_from_bytes_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES) },
            action_perm,
            dst,
        );
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;
        let dst = unsafe {
            scratch
                .exit_mask_flat
                .as_mut()
                .expect("exit enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        expand_and_augment_mask_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_MASK_BYTES) },
            action_perm,
            dst,
        );
        cursor += OPTIONAL_ACTION_MASK_BYTES;
    }

    if shard.feature_flags & FLAG_DELTA_Q != 0 {
        let dst = unsafe {
            scratch
                .delta_q_target_flat
                .as_mut()
                .expect("delta_q enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        augment_action_f32_from_bytes_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_FLOAT32_BYTES) },
            action_perm,
            dst,
        );
        cursor += OPTIONAL_ACTION_FLOAT32_BYTES;
        let dst = unsafe {
            scratch
                .delta_q_mask_flat
                .as_mut()
                .expect("delta_q enabled")
                .get_unchecked_mut(row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE)
        };
        if expand_and_augment_mask_into(
            unsafe { bytes.get_unchecked(cursor..cursor + OPTIONAL_ACTION_MASK_BYTES) },
            action_perm,
            dst,
        ) {
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

fn sidecar_manifest(
    path: Option<&Path>,
    provenance: SidecarProvenance,
) -> Option<BcShardSidecarManifest> {
    let (source_net_hash, source_version) =
        provenance.source_net_hash.zip(provenance.source_version)?;
    Some(BcShardSidecarManifest {
        path: path?.display().to_string(),
        source_net_hash,
        source_version,
    })
}

fn replay_target_profile_for_bc_shards(
    config: &BuildBcShardsConfig,
) -> crate::data::mjai_loader::ReplayTargetProfile {
    crate::data::mjai_loader::ReplayTargetProfile::with_optional_heads(
        false,
        false,
        false,
        false,
        config.exit_sidecar.is_some(),
        config.delta_q_sidecar.is_some(),
    )
}

fn replay_load_policy_for_bc_shards(config: &BuildBcShardsConfig) -> ReplayLoadPolicy<'_> {
    ReplayLoadPolicy::new(
        replay_target_profile_for_bc_shards(config),
        config.exit_provenance,
        config.delta_q_provenance,
        config.exit_sidecar.as_deref(),
        config.delta_q_sidecar.as_deref(),
    )
}

fn load_bc_shard_game_from_path(path: &Path, config: &BuildBcShardsConfig) -> io::Result<MjaiGame> {
    let policy = replay_load_policy_for_bc_shards(config);
    load_game_from_path_with_policy(path, Some(&policy))
}

fn load_bc_shard_game_from_stream<R: Read>(
    identity: &str,
    stream: R,
    config: &BuildBcShardsConfig,
) -> io::Result<MjaiGame> {
    let policy = replay_load_policy_for_bc_shards(config);
    load_game_from_stream_with_policy(identity, stream, Some(&policy))
}

struct LoadedGameContext<'a> {
    config: &'a BuildBcShardsConfig,
    train_state: &'a mut Option<SplitBuildState>,
    val_state: &'a mut Option<SplitBuildState>,
    skipped_games: &'a mut u64,
    empty_games: &'a mut u64,
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
    let result = load_bc_shard_game_from_path(path, config);
    let mut ctx = LoadedGameContext {
        config,
        train_state,
        val_state,
        skipped_games,
        empty_games,
    };
    handle_loaded_game(&identity, split, result, &mut ctx)
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
            io::Error::other(format!(
                "failed to open zstd archive {}: {err}",
                path.display()
            ))
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
        let result = load_bc_shard_game_from_stream(&identity, entry, config);
        let mut ctx = LoadedGameContext {
            config,
            train_state,
            val_state,
            skipped_games,
            empty_games,
        };
        handle_loaded_game(&identity, split, result, &mut ctx)?;
    }
    Ok(())
}

fn handle_loaded_game(
    identity: &str,
    split: BcShardSplit,
    result: io::Result<MjaiGame>,
    ctx: &mut LoadedGameContext<'_>,
) -> io::Result<()> {
    match result {
        Ok(game) => {
            if game.samples.is_empty() {
                *ctx.empty_games += 1;
                return Ok(());
            }
            match split {
                BcShardSplit::Train => {
                    if let Some(state) = ctx.train_state.as_mut() {
                        state.push_game(&ctx.config.output_dir, ctx.config.shard_samples, &game)?;
                    }
                }
                BcShardSplit::Validation => {
                    if let Some(state) = ctx.val_state.as_mut() {
                        state.push_game(&ctx.config.output_dir, ctx.config.shard_samples, &game)?;
                    }
                }
            }
        }
        Err(err) => {
            *ctx.skipped_games += 1;
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

fn write_sample_record<W: Write>(
    writer: &mut W,
    sample: &crate::data::sample::MjaiSample,
    flags: u32,
) -> io::Result<()> {
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

fn verify_shard_header(
    mmap: &Mmap,
    split: BcShardSplit,
    feature_flags: u32,
    record_size: u32,
) -> Result<(), String> {
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

fn write_binary_triplet<W: Write>(
    writer: &mut W,
    values: &[f32; OPPONENT_COUNT],
) -> io::Result<()> {
    for &value in values {
        writer.write_all(&[u8::from(value > 0.0)])?;
    }
    Ok(())
}

fn write_optional_oracle_f32<W: Write>(
    writer: &mut W,
    values: Option<&[f32; PLAYER_COUNT]>,
) -> io::Result<()> {
    if let Some(values) = values {
        for &value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
    } else {
        write_zero_bytes(writer, ORACLE_FLOAT32_BYTES)?;
    }
    Ok(())
}

fn write_binary_mask_u8<W: Write>(
    writer: &mut W,
    values: &[f32; SPATIAL_TARGET_SIZE],
) -> io::Result<()> {
    for &value in values {
        writer.write_all(&[u8::from(value > 0.0)])?;
    }
    Ok(())
}

fn write_optional_action_f32<W: Write>(
    writer: &mut W,
    values: Option<&[f32; HYDRA_ACTION_SPACE]>,
) -> io::Result<()> {
    if let Some(values) = values {
        for &value in values {
            writer.write_all(&value.to_le_bytes())?;
        }
    } else {
        write_zero_bytes(writer, OPTIONAL_ACTION_FLOAT32_BYTES)?;
    }
    Ok(())
}

fn write_optional_action_mask_u8<W: Write>(
    writer: &mut W,
    values: Option<&[f32; HYDRA_ACTION_SPACE]>,
) -> io::Result<()> {
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

fn read_i32_le(bytes: &[u8; 4]) -> i32 {
    i32::from_le_bytes(*bytes)
}

#[cfg(not(target_endian = "little"))]
fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes[0..4].try_into().expect("f32 slice"))
}

#[cfg_attr(
    target_endian = "little",
    allow(
        dead_code,
        reason = "big-endian fallback helper is unused on little-endian targets"
    )
)]
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

#[inline]
fn augment_action_f32_from_bytes_into(
    bytes: &[u8],
    action_perm: &[usize; 37],
    dst: &mut [f32],
) -> bool {
    debug_assert!(bytes.len() >= OPTIONAL_ACTION_FLOAT32_BYTES);
    debug_assert_eq!(dst.len(), HYDRA_ACTION_SPACE);
    let src = unsafe { bytes.get_unchecked(..OPTIONAL_ACTION_FLOAT32_BYTES) };
    if !any_nonzero_u8(src) {
        return false;
    }
    #[cfg(target_endian = "little")]
    {
        let src_f32 = src.as_ptr() as *const f32;
        for i in 0..37usize {
            unsafe {
                let perm_idx = *action_perm.get_unchecked(i);
                *dst.get_unchecked_mut(perm_idx) = *src_f32.add(i);
            }
        }
        unsafe {
            ptr::copy_nonoverlapping(
                src_f32.add(37),
                dst.get_unchecked_mut(37..HYDRA_ACTION_SPACE).as_mut_ptr(),
                HYDRA_ACTION_SPACE - 37,
            );
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        let values = read_f32_array::<HYDRA_ACTION_SPACE>(src);
        augment_action_vector_suit_into(&values, action_perm, dst);
    }
    true
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
    let region = unsafe { bytes.get_unchecked(..HYDRA_ACTION_SPACE * 4) };
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
    let src = unsafe { bytes.get_unchecked(..HYDRA_ACTION_SPACE) };
    if !any_nonzero_u8(src) {
        return false;
    }
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = f32::from_bits((s != 0) as u32 * F32_ONE_BITS);
    }
    true
}

/// Expand u8 mask bytes to f32 with suit permutation, writing directly
/// into `dst`.  Fuses the decode and scatter passes.
#[inline]
fn expand_and_augment_mask_into(bytes: &[u8], action_perm: &[usize; 37], dst: &mut [f32]) -> bool {
    debug_assert!(bytes.len() >= HYDRA_ACTION_SPACE);
    debug_assert_eq!(dst.len(), HYDRA_ACTION_SPACE);
    if !any_nonzero_u8(unsafe { bytes.get_unchecked(..HYDRA_ACTION_SPACE) }) {
        return false;
    }
    for i in 0..37usize {
        unsafe {
            let perm_idx = *action_perm.get_unchecked(i);
            *dst.get_unchecked_mut(perm_idx) =
                f32::from_bits((*bytes.get_unchecked(i) != 0) as u32 * F32_ONE_BITS);
        }
    }
    unsafe { dst.get_unchecked_mut(37..HYDRA_ACTION_SPACE) }
        .iter_mut()
        .zip(unsafe { bytes.get_unchecked(37..HYDRA_ACTION_SPACE) }.iter())
        .for_each(|(d, &src)| *d = f32::from_bits((src != 0) as u32 * F32_ONE_BITS));
    true
}

#[cfg(not(target_endian = "little"))]
fn read_oracle_f32(bytes: &[u8]) -> [f32; PLAYER_COUNT] {
    read_f32_array::<PLAYER_COUNT>(bytes)
}

pub fn policy_target_vec_from_actions(actions: &[i64], batch_size: usize) -> Vec<f32> {
    let mut policy_target = vec![0.0f32; batch_size * HYDRA_ACTION_SPACE];
    for (row, &action) in actions.iter().take(batch_size).enumerate() {
        if let Ok(action) = usize::try_from(action)
            && action < HYDRA_ACTION_SPACE
        {
            policy_target[row * HYDRA_ACTION_SPACE + action] = 1.0;
        }
    }
    policy_target
}

fn policy_target_from_action_slice<B: Backend>(
    actions: &[i64],
    batch_size: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(
        TensorData::new(
            policy_target_vec_from_actions(actions, batch_size),
            [batch_size, HYDRA_ACTION_SPACE],
        ),
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::mjai_loader::{load_game_from_path, prepare_replay_decision, update_safety};
    use crate::training::replay_delta_q::{DeltaQSidecarIndex, ReplayDeltaQRecordV1};
    use crate::training::replay_exit::{
        ExitSidecarIndex, ReplayDecisionKey, ReplayExitRecordV1, legal_mask_digest_from_f32,
        source_hash_from_identity,
    };
    use crate::training::{live_exit, replay_delta_q, replay_exit};
    use hydra_core::action::HYDRA_ACTION_SPACE;
    use hydra_core::encoder::ObservationEncoder;
    use hydra_core::safety::SafetyInfo;
    use riichienv_core::replay::read_mjai_events;
    use riichienv_core::rule::GameRule;
    use riichienv_core::state::GameState;
    use std::fs;
    use std::io::Cursor;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn::backend::NdArray;

    use crate::data::sample::collate_samples_bc_owned;

    #[test]
    fn bc_shard_manifest_geometry_uses_frozen_runtime_abi() {
        let manifest = BcShardManifest {
            manifest_version: BC_SHARD_MANIFEST_VERSION,
            shard_version: BC_SHARD_VERSION,
            shard_header_size: BC_SHARD_HEADER_SIZE,
            base_record_size: BC_BASE_RECORD_SIZE,
            max_record_size: BC_RECORD_SIZE_WITH_ALL_OPTIONALS,
            obs_size: OBS_SIZE,
            num_channels: NUM_CHANNELS,
            action_space: HYDRA_ACTION_SPACE,
            train_fraction: 1.0,
            shard_samples: 1,
            augment_runtime: true,
            input: String::new(),
            output_dir: String::new(),
            created_at: String::new(),
            source_count: 0,
            source_total_games_hint: 0,
            source_train_count_hint: 0,
            source_val_count_hint: 0,
            source_counts_exact: true,
            exit_sidecar: None,
            delta_q_sidecar: None,
            totals: BcShardBuildTotals::default(),
            splits: Vec::new(),
        };

        assert_eq!(OBS_SIZE, 6528);
        assert_eq!(NUM_CHANNELS, 192);
        assert_eq!(HYDRA_ACTION_SPACE, 46);
        assert_eq!(LEGAL_MASK_BYTES, 46);
        assert_eq!(OPTIONAL_ACTION_FLOAT32_BYTES, 184);
        assert_eq!(OPTIONAL_ACTION_MASK_BYTES, 46);
        assert_eq!(manifest.obs_size, OBS_SIZE);
        assert_eq!(manifest.num_channels, NUM_CHANNELS);
        assert_eq!(manifest.action_space, HYDRA_ACTION_SPACE);
        validate_bc_shard_manifest_contract(&manifest).expect("manifest geometry should be valid");
    }

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

    fn replay_sidecar_guardrail_log() -> String {
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

    fn replay_guardrail_decisions_for_identity(
        identity: &str,
    ) -> Vec<(ReplayDecisionKey, u8, [f32; HYDRA_ACTION_SPACE])> {
        let events = read_mjai_events(Cursor::new(replay_sidecar_guardrail_log()))
            .expect("parse sidecar replay");
        let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
        let mut safety = std::array::from_fn(|_| SafetyInfo::default());
        let mut encoder = ObservationEncoder::new();
        let mut decisions = Vec::new();

        for (idx, event) in events.iter().enumerate() {
            if let Some(decision) =
                prepare_replay_decision(event, &mut state, &safety, &mut encoder)
                    .expect("prepare replay decision")
            {
                decisions.push((
                    ReplayDecisionKey {
                        source_hash: source_hash_from_identity(identity),
                        event_index: idx as u32,
                        actor: decision.actor as u8,
                        obs_hash: live_exit::obs_hash(&decision.obs_encoded),
                    },
                    decision.action_id,
                    decision.legal_mask_f32,
                ));
            }
            update_safety(&mut safety, event).expect("update safety");
            state.apply_mjai_event(event.clone());
        }

        decisions
    }

    fn synthetic_exit_records(
        identity: &str,
        source_net_hash: u64,
        source_version: u32,
    ) -> Vec<ReplayExitRecordV1> {
        replay_guardrail_decisions_for_identity(identity)
            .into_iter()
            .take(2)
            .map(|(key, action, legal_mask)| {
                let mut target = [0.0f32; HYDRA_ACTION_SPACE];
                let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
                target[action as usize] = 1.0;
                mask[action as usize] = 1.0;
                ReplayExitRecordV1 {
                    version: 1,
                    semantics: replay_exit::REPLAY_EXIT_SEMANTICS_V1.to_string(),
                    provenance: replay_exit::REPLAY_EXIT_PROVENANCE.to_string(),
                    key,
                    action,
                    legal_mask_digest: legal_mask_digest_from_f32(&legal_mask),
                    source_net_hash,
                    source_version,
                    root_visit_count: 64,
                    legal_discard_count: legal_mask[..=36]
                        .iter()
                        .filter(|&&value| value > 0.0)
                        .count() as u8,
                    supported_actions: 1,
                    coverage: 1.0,
                    kl_to_base: 0.0,
                    target: target.to_vec(),
                    mask: mask.to_vec(),
                }
            })
            .collect()
    }

    fn synthetic_delta_q_records(
        identity: &str,
        source_net_hash: u64,
        source_version: u32,
    ) -> Vec<ReplayDeltaQRecordV1> {
        replay_guardrail_decisions_for_identity(identity)
            .into_iter()
            .take(2)
            .map(|(key, action, legal_mask)| {
                let mut target = [0.0f32; HYDRA_ACTION_SPACE];
                let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
                target[action as usize] = 0.25;
                mask[action as usize] = 1.0;
                ReplayDeltaQRecordV1 {
                    version: 1,
                    semantics: replay_delta_q::REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
                    provenance: replay_delta_q::REPLAY_DELTA_Q_PROVENANCE.to_string(),
                    key,
                    action,
                    legal_mask_digest: legal_mask_digest_from_f32(&legal_mask),
                    source_net_hash,
                    source_version,
                    target: target.to_vec(),
                    mask: mask.to_vec(),
                }
            })
            .collect()
    }

    fn unique_bc_shard_temp_dir(label: &str) -> PathBuf {
        let root = std::env::temp_dir();
        fs::create_dir_all(&root).expect("create bc_shard temp root");
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time after epoch")
            .as_nanos();
        let dir = root.join(format!(
            "hydra_bc_shards_{label}_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&dir).expect("create bc_shard temp dir");
        dir
    }

    #[test]
    fn compact_header_size_constant_matches_written_bytes() {
        let mut bytes = Vec::new();
        write_shard_header(
            &mut bytes,
            BcShardSplit::Train,
            2,
            10,
            100,
            FLAG_SAFETY_RESIDUAL,
            record_size_for_flags(FLAG_SAFETY_RESIDUAL),
        )
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
            source_manifest: None,
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
                assert!((a - b).abs() < 1e-6, "{name}[{idx}] mismatch: {a} vs {b}");
            }
        }

        assert_tensor_close(raw_obs, shard.obs, "obs");
        assert_tensor_close(
            raw_batch.actions.clone().float(),
            shard.batch.actions.clone().float(),
            "actions",
        );
        assert_tensor_close(
            raw_targets.legal_mask.clone(),
            shard.targets.legal_mask.clone(),
            "legal_mask",
        );
        assert_tensor_close(
            raw_targets.value_target.clone(),
            shard.targets.value_target.clone(),
            "value_target",
        );
        assert_tensor_close(
            raw_targets.grp_target.clone(),
            shard.targets.grp_target.clone(),
            "grp_target",
        );
        assert_tensor_close(
            raw_targets.tenpai_target.clone(),
            shard.targets.tenpai_target.clone(),
            "tenpai",
        );
        assert_tensor_close(
            raw_targets.danger_target.clone(),
            shard.targets.danger_target.clone(),
            "danger",
        );
        assert_tensor_close(
            raw_targets.danger_mask.clone(),
            shard.targets.danger_mask.clone(),
            "danger_mask",
        );
        assert_tensor_close(
            raw_targets.opp_next_target.clone(),
            shard.targets.opp_next_target.clone(),
            "opp_next",
        );
        assert_tensor_close(
            raw_targets.score_pdf_target.clone(),
            shard.targets.score_pdf_target.clone(),
            "score_pdf",
        );
        assert_tensor_close(
            raw_targets.score_cdf_target.clone(),
            shard.targets.score_cdf_target.clone(),
            "score_cdf",
        );

        let raw_presence = raw_targets.target_presence.expect("raw target presence");
        let shard_presence = shard
            .targets
            .target_presence
            .expect("shard target presence");
        assert_eq!(raw_presence.batch_size, shard_presence.batch_size);
        assert_eq!(
            raw_presence.delta_q_actions_present,
            shard_presence.delta_q_actions_present
        );
        assert_eq!(raw_presence.counts, shard_presence.counts);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn policy_target_vec_bounds_invalid_actions() {
        let policy_target =
            policy_target_vec_from_actions(&[0, -1, HYDRA_ACTION_SPACE as i64, 45], 4);
        assert_eq!(policy_target.len(), 4 * HYDRA_ACTION_SPACE);
        assert_eq!(policy_target[0], 1.0);
        assert!(
            policy_target[HYDRA_ACTION_SPACE..2 * HYDRA_ACTION_SPACE]
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            policy_target[2 * HYDRA_ACTION_SPACE..3 * HYDRA_ACTION_SPACE]
                .iter()
                .all(|&value| value == 0.0)
        );
        assert_eq!(policy_target[3 * HYDRA_ACTION_SPACE + 45], 1.0);
    }

    #[test]
    fn host_batch_materialize_bounds_invalid_policy_actions() {
        type B = NdArray<f32>;

        fn make_host(actions: Vec<i64>) -> BcShardHostBatch {
            let batch_size = actions.len();
            BcShardHostBatch {
                batch_size,
                obs_flat: vec![0.0; batch_size * OBS_SIZE],
                actions,
                legal_mask_flat: vec![1.0; batch_size * HYDRA_ACTION_SPACE],
                value_target: vec![0.0; batch_size],
                grp_target_flat: vec![0.0; batch_size * GRP_CLASS_COUNT],
                oracle_target_flat: vec![0.0; batch_size * PLAYER_COUNT],
                oracle_target_mask: vec![0.0; batch_size],
                tenpai_flat: vec![0.0; batch_size * OPPONENT_COUNT],
                danger_flat: vec![0.0; batch_size * SPATIAL_TARGET_SIZE],
                danger_mask_flat: vec![0.0; batch_size * SPATIAL_TARGET_SIZE],
                opp_next_flat: vec![0.0; batch_size * SPATIAL_TARGET_SIZE],
                score_pdf_flat: vec![0.0; batch_size * SCORE_BINS],
                score_cdf_flat: vec![0.0; batch_size * SCORE_BINS],
                safety_target_flat: None,
                safety_mask_flat: None,
                exit_target_flat: None,
                exit_mask_flat: None,
                delta_q_target_flat: None,
                delta_q_mask_flat: None,
                target_presence: TargetPresence::with_batch_size(batch_size),
            }
        }

        fn assert_policy_target(tensor: Tensor<B, 2>) {
            let data = tensor.into_data();
            let values = data.as_slice::<f32>().expect("policy_target f32");
            assert_eq!(values[3], 1.0);
            assert!(values[0..3].iter().all(|&value| value == 0.0));
            assert!(
                values[4..HYDRA_ACTION_SPACE]
                    .iter()
                    .all(|&value| value == 0.0)
            );
            assert!(
                values[HYDRA_ACTION_SPACE..2 * HYDRA_ACTION_SPACE]
                    .iter()
                    .all(|&value| value == 0.0)
            );
            assert!(
                values[2 * HYDRA_ACTION_SPACE..3 * HYDRA_ACTION_SPACE]
                    .iter()
                    .all(|&value| value == 0.0)
            );
        }

        let device = Default::default();
        let actions = vec![3, -1, HYDRA_ACTION_SPACE as i64];
        assert_policy_target(
            make_host(actions.clone())
                .materialize::<B>(&device)
                .targets
                .policy_target,
        );
        assert_policy_target(
            make_host(actions)
                .materialize_owned::<B>(&device)
                .targets
                .policy_target,
        );
    }

    #[test]
    fn shard_materialize_owned_matches_borrowed_materialize() {
        type B = NdArray<f32>;

        let root = std::env::temp_dir().join(format!(
            "bc-shard-owned-materialize-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time after epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("temp dir should be creatable");
        let replay_path = root.join("game.mjai.json");
        fs::write(&replay_path, tiny_real_mjai_replay()).expect("fixture should write");

        let shard_dir = root.join("shards");
        let build = build_bc_shards(&BuildBcShardsConfig {
            input: replay_path,
            output_dir: shard_dir,
            manifest_name: "manifest.json".into(),
            train_fraction: 1.0,
            shard_samples: 10_000,
            split_mode: BcShardSplitMode::Train,
            source_manifest: None,
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

        let sample_count = reader.sample_count();

        let mut scratch_borrowed = reader.new_scratch(sample_count);
        reader
            .collate_host_batch_range_into(0, sample_count, false, &mut scratch_borrowed)
            .expect("host collation should succeed");
        let borrowed_host = scratch_borrowed.take_batch();

        let mut scratch_owned = reader.new_scratch(sample_count);
        reader
            .collate_host_batch_range_into(0, sample_count, false, &mut scratch_owned)
            .expect("host collation should succeed");
        let owned_host = scratch_owned.take_batch();

        let borrowed = borrowed_host.materialize::<B>(&device);
        let owned = owned_host.materialize_owned::<B>(&device);

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
                assert!((a - b).abs() < 1e-6, "{name}[{idx}] mismatch: {a} vs {b}");
            }
        }

        assert_tensor_close(borrowed.obs, owned.obs, "obs");
        assert_tensor_close(
            borrowed.batch.actions.clone().float(),
            owned.batch.actions.clone().float(),
            "actions",
        );
        assert_tensor_close(
            borrowed.targets.legal_mask.clone(),
            owned.targets.legal_mask.clone(),
            "legal_mask",
        );
        assert_tensor_close(
            borrowed.targets.policy_target.clone(),
            owned.targets.policy_target.clone(),
            "policy_target",
        );
        assert_tensor_close(
            borrowed.targets.value_target.clone(),
            owned.targets.value_target.clone(),
            "value_target",
        );
        assert_tensor_close(
            borrowed.targets.grp_target.clone(),
            owned.targets.grp_target.clone(),
            "grp_target",
        );
        assert_tensor_close(
            borrowed.targets.tenpai_target.clone(),
            owned.targets.tenpai_target.clone(),
            "tenpai",
        );
        assert_tensor_close(
            borrowed.targets.danger_target.clone(),
            owned.targets.danger_target.clone(),
            "danger",
        );
        assert_tensor_close(
            borrowed.targets.danger_mask.clone(),
            owned.targets.danger_mask.clone(),
            "danger_mask",
        );
        assert_tensor_close(
            borrowed.targets.opp_next_target.clone(),
            owned.targets.opp_next_target.clone(),
            "opp_next",
        );
        assert_tensor_close(
            borrowed.targets.score_pdf_target.clone(),
            owned.targets.score_pdf_target.clone(),
            "score_pdf",
        );
        assert_tensor_close(
            borrowed.targets.score_cdf_target.clone(),
            owned.targets.score_cdf_target.clone(),
            "score_cdf",
        );
        match (
            borrowed.batch.exit_target.clone(),
            owned.batch.exit_target.clone(),
        ) {
            (Some(lhs), Some(rhs)) => assert_tensor_close(lhs, rhs, "exit_target"),
            (None, None) => {}
            (lhs, rhs) => panic!("exit_target presence mismatch: {lhs:?} vs {rhs:?}"),
        }
        match (
            borrowed.batch.exit_mask.clone(),
            owned.batch.exit_mask.clone(),
        ) {
            (Some(lhs), Some(rhs)) => assert_tensor_close(lhs, rhs, "exit_mask"),
            (None, None) => {}
            (lhs, rhs) => panic!("exit_mask presence mismatch: {lhs:?} vs {rhs:?}"),
        }
        match (
            borrowed.targets.safety_residual_target.clone(),
            owned.targets.safety_residual_target.clone(),
        ) {
            (Some(lhs), Some(rhs)) => assert_tensor_close(lhs, rhs, "safety_residual_target"),
            (None, None) => {}
            (lhs, rhs) => panic!("safety_residual_target presence mismatch: {lhs:?} vs {rhs:?}"),
        }
        match (
            borrowed.targets.safety_residual_mask.clone(),
            owned.targets.safety_residual_mask.clone(),
        ) {
            (Some(lhs), Some(rhs)) => assert_tensor_close(lhs, rhs, "safety_residual_mask"),
            (None, None) => {}
            (lhs, rhs) => panic!("safety_residual_mask presence mismatch: {lhs:?} vs {rhs:?}"),
        }
        match (
            borrowed.targets.delta_q_target.clone(),
            owned.targets.delta_q_target.clone(),
        ) {
            (Some(lhs), Some(rhs)) => assert_tensor_close(lhs, rhs, "delta_q_target"),
            (None, None) => {}
            (lhs, rhs) => panic!("delta_q_target presence mismatch: {lhs:?} vs {rhs:?}"),
        }
        match (
            borrowed.targets.delta_q_mask.clone(),
            owned.targets.delta_q_mask.clone(),
        ) {
            (Some(lhs), Some(rhs)) => assert_tensor_close(lhs, rhs, "delta_q_mask"),
            (None, None) => {}
            (lhs, rhs) => panic!("delta_q_mask presence mismatch: {lhs:?} vs {rhs:?}"),
        }

        let borrowed_presence = borrowed
            .targets
            .target_presence
            .expect("borrowed target presence");
        let owned_presence = owned
            .targets
            .target_presence
            .expect("owned target presence");
        assert_eq!(borrowed_presence.batch_size, owned_presence.batch_size);
        assert_eq!(borrowed_presence.counts, owned_presence.counts);
        assert_eq!(
            borrowed_presence.delta_q_actions_present,
            owned_presence.delta_q_actions_present
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn supplied_source_manifest_is_written_into_output_manifest() {
        let root = std::env::temp_dir().join(format!(
            "bc-shard-source-manifest-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time after epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("temp dir should be creatable");
        let replay_path = root.join("game.mjai.json");
        fs::write(&replay_path, tiny_real_mjai_replay()).expect("fixture should write");
        let shard_dir = root.join("shards");

        let supplied_manifest = DataManifest {
            sources: vec![DataSource::LooseFile(replay_path.clone())],
            total_games: 17,
            train_count: 11,
            val_count: 6,
            counts_exact: false,
        };

        let build = build_bc_shards(&BuildBcShardsConfig {
            input: replay_path,
            output_dir: shard_dir,
            manifest_name: "manifest.json".into(),
            train_fraction: 1.0,
            shard_samples: 10_000,
            split_mode: BcShardSplitMode::Train,
            source_manifest: Some(supplied_manifest.clone()),
            exit_sidecar: None,
            exit_sidecar_path: None,
            exit_provenance: SidecarProvenance::default(),
            delta_q_sidecar: None,
            delta_q_sidecar_path: None,
            delta_q_provenance: SidecarProvenance::default(),
        })
        .expect("shards should build");

        assert_eq!(build.manifest.source_count, supplied_manifest.sources.len());
        assert_eq!(
            build.manifest.source_total_games_hint,
            supplied_manifest.total_games
        );
        assert_eq!(
            build.manifest.source_train_count_hint,
            supplied_manifest.train_count
        );
        assert_eq!(
            build.manifest.source_val_count_hint,
            supplied_manifest.val_count
        );
        assert_eq!(
            build.manifest.source_counts_exact,
            supplied_manifest.counts_exact
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_bc_shard_game_from_path_uses_file_name_identity_for_sidecars() {
        let root = unique_bc_shard_temp_dir("sidecar_path");
        let replay_path = root.join("game-1.mjai.json");
        fs::write(&replay_path, replay_sidecar_guardrail_log()).expect("fixture should write");

        let exit_records = synthetic_exit_records("game-1.mjai.json", 123, 1);
        let delta_q_records = synthetic_delta_q_records("game-1.mjai.json", 456, 2);
        let config = BuildBcShardsConfig {
            input: replay_path.clone(),
            output_dir: root.join("shards"),
            exit_sidecar: Some(Arc::new(ExitSidecarIndex::from_records(exit_records))),
            exit_provenance: SidecarProvenance::new(Some(123), Some(1)),
            delta_q_sidecar: Some(Arc::new(DeltaQSidecarIndex::from_records(delta_q_records))),
            delta_q_provenance: SidecarProvenance::new(Some(456), Some(2)),
            ..BuildBcShardsConfig::default()
        };

        let game =
            load_bc_shard_game_from_path(&replay_path, &config).expect("path replay should load");

        assert!(
            game.samples
                .iter()
                .any(|sample| sample.exit_target.is_some())
        );
        assert!(game.samples.iter().any(|sample| sample.exit_mask.is_some()));
        assert!(
            game.samples
                .iter()
                .any(|sample| sample.delta_q_target.is_some())
        );
        assert!(
            game.samples
                .iter()
                .any(|sample| sample.delta_q_mask.is_some())
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_bc_shard_game_from_stream_uses_explicit_source_identity() {
        let identity = "archive.tar.zst/game-1.mjai.json";
        let exit_records = synthetic_exit_records(identity, 123, 1);
        let delta_q_records = synthetic_delta_q_records(identity, 456, 2);
        let config = BuildBcShardsConfig {
            input: PathBuf::from("archive.tar.zst"),
            output_dir: PathBuf::from("/home/nikketryhard/tmp/bc_shard_unused"),
            exit_sidecar: Some(Arc::new(ExitSidecarIndex::from_records(exit_records))),
            exit_provenance: SidecarProvenance::new(Some(123), Some(1)),
            delta_q_sidecar: Some(Arc::new(DeltaQSidecarIndex::from_records(delta_q_records))),
            delta_q_provenance: SidecarProvenance::new(Some(456), Some(2)),
            ..BuildBcShardsConfig::default()
        };

        let game = load_bc_shard_game_from_stream(
            identity,
            Cursor::new(replay_sidecar_guardrail_log()),
            &config,
        )
        .expect("archive replay should load");

        assert!(
            game.samples
                .iter()
                .any(|sample| sample.exit_target.is_some())
        );
        assert!(game.samples.iter().any(|sample| sample.exit_mask.is_some()));
        assert!(
            game.samples
                .iter()
                .any(|sample| sample.delta_q_target.is_some())
        );
        assert!(
            game.samples
                .iter()
                .any(|sample| sample.delta_q_mask.is_some())
        );
    }

    #[test]
    fn load_bc_shard_reader_rejects_manifest_missing_requested_split() {
        let root = std::env::temp_dir().join(format!(
            "bc-shard-missing-split-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time after epoch")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("temp dir should be creatable");
        let replay_path = root.join("game.mjai.json");
        fs::write(&replay_path, tiny_real_mjai_replay()).expect("fixture should write");
        let shard_dir = root.join("shards");

        let build = build_bc_shards(&BuildBcShardsConfig {
            input: replay_path,
            output_dir: shard_dir.clone(),
            manifest_name: "manifest.json".into(),
            train_fraction: 1.0,
            shard_samples: 10_000,
            split_mode: BcShardSplitMode::Train,
            source_manifest: None,
            exit_sidecar: None,
            exit_sidecar_path: None,
            exit_provenance: SidecarProvenance::default(),
            delta_q_sidecar: None,
            delta_q_sidecar_path: None,
            delta_q_provenance: SidecarProvenance::default(),
        })
        .expect("shards should build");

        let err = match load_bc_shard_reader(&build.manifest_path, BcShardSplit::Validation) {
            Ok(_) => panic!("missing split should be rejected"),
            Err(err) => err,
        };
        assert!(err.contains("missing Validation split"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_bc_shard_reader_rejects_non_contiguous_shard_descriptors() {
        let root = unique_bc_shard_temp_dir("non_contiguous_manifest");
        let replay_path = root.join("game.mjai.json");
        fs::write(&replay_path, tiny_real_mjai_replay()).expect("fixture should write");
        let shard_dir = root.join("shards");

        let build = build_bc_shards(&BuildBcShardsConfig {
            input: replay_path,
            output_dir: shard_dir.clone(),
            manifest_name: "manifest.json".into(),
            train_fraction: 1.0,
            shard_samples: 1,
            split_mode: BcShardSplitMode::Train,
            source_manifest: None,
            exit_sidecar: None,
            exit_sidecar_path: None,
            exit_provenance: SidecarProvenance::default(),
            delta_q_sidecar: None,
            delta_q_sidecar_path: None,
            delta_q_provenance: SidecarProvenance::default(),
        })
        .expect("shards should build");

        let mut manifest: BcShardManifest = serde_json::from_str(
            &fs::read_to_string(&build.manifest_path).expect("manifest should exist"),
        )
        .expect("manifest should deserialize");
        let split = manifest
            .splits
            .iter_mut()
            .find(|split| split.split == BcShardSplit::Train)
            .expect("train split exists");
        let mut extra = split.shards[0].clone();
        extra.shard_index = 1;
        extra.first_sample_index = split.shards[0].sample_count + 1;
        split.shards.push(extra);
        split.shard_count = split.shards.len();
        split.sample_count += split.shards[1].sample_count;
        manifest.totals.shard_count = split.shard_count;
        manifest.totals.sample_count = split.sample_count;
        fs::write(
            &build.manifest_path,
            serde_json::to_string_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("mutated manifest should write");

        let err = match load_bc_shard_reader(&build.manifest_path, BcShardSplit::Train) {
            Ok(_) => panic!("non-contiguous split should be rejected"),
            Err(err) => err,
        };
        assert!(err.contains("expected contiguous start") || err.contains("shard_index"));

        let _ = fs::remove_dir_all(root);
    }
}
