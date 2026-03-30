use std::fs;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use burn::prelude::*;
use burn::tensor::backend::Backend;
use half::f16;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::data::augment::{
    augment_action_suit, augment_action_vector_suit, augment_mask_suit, augment_obs_suit,
};
use crate::data::mjai_loader::{
    MjaiGame, SidecarProvenance, invalid_data, load_game_from_path, load_game_from_path_with_sidecar,
    load_game_from_stream, load_game_from_stream_with_sidecar,
};
use crate::data::pipeline::{DataSource, is_train_game, scan_data_sources};
use crate::data::sample::{MjaiBcBatch, score_delta_to_bin, score_delta_to_value};
use crate::training::head_gates::TargetPresence;
use crate::training::losses::HydraTargets;
use crate::training::replay_delta_q::DeltaQSidecarIndex;
use crate::training::replay_exit::ExitSidecarIndex;

const OPPONENT_COUNT: usize = 3;
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

pub const OBS_F16_BYTES: usize = OBS_SIZE * 2;
pub const LEGAL_MASK_BYTES: usize = HYDRA_ACTION_SPACE;
pub const TENPAI_BYTES: usize = OPPONENT_COUNT;
pub const OPP_NEXT_BYTES: usize = OPPONENT_COUNT;
pub const DANGER_BYTES: usize = SPATIAL_TARGET_SIZE;
pub const DANGER_MASK_BYTES: usize = SPATIAL_TARGET_SIZE;
pub const OPTIONAL_ACTION_FLOAT16_BYTES: usize = HYDRA_ACTION_SPACE * 2;
pub const OPTIONAL_ACTION_MASK_BYTES: usize = HYDRA_ACTION_SPACE;

pub const BC_BASE_RECORD_SIZE: u32 = (OBS_F16_BYTES
    + 1
    + LEGAL_MASK_BYTES
    + 4
    + 1
    + TENPAI_BYTES
    + OPP_NEXT_BYTES
    + DANGER_BYTES
    + DANGER_MASK_BYTES) as u32;

pub const BC_RECORD_SIZE_WITH_ALL_OPTIONALS: u32 = BC_BASE_RECORD_SIZE
    + (OPTIONAL_ACTION_FLOAT16_BYTES as u32 + OPTIONAL_ACTION_MASK_BYTES as u32) * 3;

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
    pub grp_target: Vec<f32>,
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
}

// SAFETY: all fields are plain vecs of Copy types -- trivially Send + Sync.
unsafe impl Send for BcShardHostBatch {}
unsafe impl Sync for BcShardHostBatch {}

impl BcShardHostBatch {
    /// Materialize device tensors from CPU-side flat buffers.
    ///
    /// This is the only step that touches the `Backend` / device.
    pub fn materialize<B: Backend>(self, device: &B::Device) -> BcShardBatch<B> {
        let batch = self.batch_size;

        let obs = Tensor::<B, 1>::from_floats(self.obs_flat.as_slice(), device)
            .reshape([batch, NUM_CHANNELS, TILE_COUNT]);
        let actions_tensor = Tensor::<B, 1, Int>::from_ints(self.actions.as_slice(), device);
        let legal_mask = Tensor::<B, 1>::from_floats(self.legal_mask_flat.as_slice(), device)
            .reshape([batch, HYDRA_ACTION_SPACE]);
        let value_target = Tensor::<B, 1>::from_floats(self.value_target.as_slice(), device);
        let grp_target = Tensor::<B, 1>::from_floats(self.grp_target.as_slice(), device)
            .reshape([batch, GRP_CLASS_COUNT]);
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

        let batch_struct = MjaiBcBatch {
            actions: actions_tensor.clone(),
            exit_target: exit_target_tensor.clone(),
            exit_mask: exit_mask_tensor.clone(),
        };

        let targets = HydraTargets {
            policy_target: policy_target_from_actions::<B>(actions_tensor.clone(), batch),
            legal_mask,
            value_target,
            grp_target,
            tenpai_target,
            danger_target,
            danger_mask,
            opp_next_target,
            score_pdf_target,
            score_cdf_target,
            oracle_target: None,
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
            oracle_guidance_mask: None,
            target_presence: Some(TargetPresence::default()),
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

        let batch = indices.len();
        let mut obs_flat = vec![0.0f32; batch * OBS_SIZE];
        let mut actions = vec![0i64; batch];
        let mut legal_mask_flat = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
        let mut value_target = vec![0.0f32; batch];
        let mut grp_target = vec![0.0f32; batch * GRP_CLASS_COUNT];
        let mut tenpai_flat = vec![0.0f32; batch * OPPONENT_COUNT];
        let mut danger_flat = vec![0.0f32; batch * SPATIAL_TARGET_SIZE];
        let mut danger_mask_flat = vec![0.0f32; batch * SPATIAL_TARGET_SIZE];
        let mut opp_next_flat = vec![0.0f32; batch * SPATIAL_TARGET_SIZE];
        let mut score_pdf_flat = vec![0.0f32; batch * SCORE_BINS];
        let mut score_cdf_flat = vec![0.0f32; batch * SCORE_BINS];

        let need_safety = self.shards.first().is_some_and(|s| s.feature_flags & FLAG_SAFETY_RESIDUAL != 0);
        let need_exit = self.shards.first().is_some_and(|s| s.feature_flags & FLAG_EXIT != 0);
        let need_delta_q = self.shards.first().is_some_and(|s| s.feature_flags & FLAG_DELTA_Q != 0);

        let mut safety_target_flat = need_safety.then(|| vec![0.0f32; batch * HYDRA_ACTION_SPACE]);
        let mut safety_mask_flat = need_safety.then(|| vec![0.0f32; batch * HYDRA_ACTION_SPACE]);
        let mut exit_target_flat = need_exit.then(|| vec![0.0f32; batch * HYDRA_ACTION_SPACE]);
        let mut exit_mask_flat = need_exit.then(|| vec![0.0f32; batch * HYDRA_ACTION_SPACE]);
        let mut delta_q_target_flat = need_delta_q.then(|| vec![0.0f32; batch * HYDRA_ACTION_SPACE]);
        let mut delta_q_mask_flat = need_delta_q.then(|| vec![0.0f32; batch * HYDRA_ACTION_SPACE]);

        for (row, &sample_index) in indices.iter().enumerate() {
            let (shard, offset) = self.locate(sample_index)?;
            let sample = parse_compact_sample(shard, offset)?;
            let perm = if augment {
                &hydra_core::tile::ALL_PERMUTATIONS[(sample_index + row) % hydra_core::tile::ALL_PERMUTATIONS.len()]
            } else {
                &hydra_core::tile::ALL_PERMUTATIONS[0]
            };

            let obs = if augment {
                augment_obs_suit(&sample.obs, perm)
            } else {
                sample.obs
            };
            obs_flat[row * OBS_SIZE..(row + 1) * OBS_SIZE].copy_from_slice(&obs);

            let action = if augment {
                augment_action_suit(sample.action, perm)
            } else {
                sample.action
            };
            actions[row] = action as i64;

            let legal_mask = if augment {
                augment_mask_suit(&sample.legal_mask, perm)
            } else {
                sample.legal_mask
            };
            legal_mask_flat[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE]
                .copy_from_slice(&legal_mask);

            value_target[row] = score_delta_to_value(sample.score_delta);
            if (sample.grp_label as usize) < GRP_CLASS_COUNT {
                grp_target[row * GRP_CLASS_COUNT + sample.grp_label as usize] = 1.0;
            }
            tenpai_flat[row * OPPONENT_COUNT..(row + 1) * OPPONENT_COUNT]
                .copy_from_slice(&sample.tenpai);

            let opp_next = if augment {
                permute_opp_next_targets(sample.opp_next, perm)
            } else {
                sample.opp_next
            };
            for (opp, tile) in opp_next.iter().copied().enumerate() {
                if tile < TILE_COUNT as u8 {
                    let idx = row * SPATIAL_TARGET_SIZE + opp * TILE_COUNT + tile as usize;
                    opp_next_flat[idx] = 1.0;
                }
            }

            let danger = if augment {
                permute_spatial_targets_3x34(sample.danger, perm)
            } else {
                sample.danger
            };
            danger_flat[row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE]
                .copy_from_slice(&danger);

            let danger_mask = if augment {
                permute_spatial_targets_3x34(sample.danger_mask, perm)
            } else {
                sample.danger_mask
            };
            danger_mask_flat[row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE]
                .copy_from_slice(&danger_mask);

            let score_bin = score_delta_to_bin(sample.score_delta);
            score_pdf_flat[row * SCORE_BINS + score_bin] = 1.0;
            score_cdf_flat[row * SCORE_BINS + score_bin..(row + 1) * SCORE_BINS].fill(1.0);

            if let Some(values) = sample.safety_residual {
                let values = if augment {
                    augment_action_vector_suit(&values, perm)
                } else {
                    values
                };
                safety_target_flat.as_mut().expect("safety enabled")[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE]
                    .copy_from_slice(&values);
            }
            if let Some(mask) = sample.safety_residual_mask {
                let mask = if augment {
                    augment_action_vector_suit(&mask, perm)
                } else {
                    mask
                };
                safety_mask_flat.as_mut().expect("safety enabled")[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE]
                    .copy_from_slice(&mask);
            }
            if let Some(values) = sample.exit_target {
                let values = if augment {
                    augment_action_vector_suit(&values, perm)
                } else {
                    values
                };
                exit_target_flat.as_mut().expect("exit enabled")[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE]
                    .copy_from_slice(&values);
            }
            if let Some(mask) = sample.exit_mask {
                let mask = if augment {
                    augment_action_vector_suit(&mask, perm)
                } else {
                    mask
                };
                exit_mask_flat.as_mut().expect("exit enabled")[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE]
                    .copy_from_slice(&mask);
            }
            if let Some(values) = sample.delta_q_target {
                let values = if augment {
                    augment_action_vector_suit(&values, perm)
                } else {
                    values
                };
                delta_q_target_flat.as_mut().expect("delta_q enabled")[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE]
                    .copy_from_slice(&values);
            }
            if let Some(mask) = sample.delta_q_mask {
                let mask = if augment {
                    augment_action_vector_suit(&mask, perm)
                } else {
                    mask
                };
                delta_q_mask_flat.as_mut().expect("delta_q enabled")[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE]
                    .copy_from_slice(&mask);
            }
        }

        Ok(BcShardHostBatch {
            batch_size: batch,
            obs_flat,
            actions,
            legal_mask_flat,
            value_target,
            grp_target,
            tenpai_flat,
            danger_flat,
            danger_mask_flat,
            opp_next_flat,
            score_pdf_flat,
            score_cdf_flat,
            safety_target_flat,
            safety_mask_flat,
            exit_target_flat,
            exit_mask_flat,
            delta_q_target_flat,
            delta_q_mask_flat,
        })
    }

    fn locate(&self, sample_index: usize) -> Result<(&ShardMap, usize), String> {
        let sample_index = sample_index as u64;
        self.shards
            .iter()
            .find_map(|shard| {
                let end = shard.start_sample + shard.sample_count;
                (sample_index >= shard.start_sample && sample_index < end)
                    .then_some((shard, (sample_index - shard.start_sample) as usize))
            })
            .ok_or_else(|| format!("BC shard sample index {sample_index} out of bounds"))
    }
}

struct CompactSample {
    obs: [f32; OBS_SIZE],
    action: u8,
    legal_mask: [f32; HYDRA_ACTION_SPACE],
    score_delta: i32,
    grp_label: u8,
    tenpai: [f32; OPPONENT_COUNT],
    opp_next: [u8; OPPONENT_COUNT],
    danger: [f32; SPATIAL_TARGET_SIZE],
    danger_mask: [f32; SPATIAL_TARGET_SIZE],
    safety_residual: Option<[f32; HYDRA_ACTION_SPACE]>,
    safety_residual_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
    exit_target: Option<[f32; HYDRA_ACTION_SPACE]>,
    exit_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
    delta_q_target: Option<[f32; HYDRA_ACTION_SPACE]>,
    delta_q_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
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
        size += (OPTIONAL_ACTION_FLOAT16_BYTES + OPTIONAL_ACTION_MASK_BYTES) as u32;
    }
    if flags & FLAG_EXIT != 0 {
        size += (OPTIONAL_ACTION_FLOAT16_BYTES + OPTIONAL_ACTION_MASK_BYTES) as u32;
    }
    if flags & FLAG_DELTA_Q != 0 {
        size += (OPTIONAL_ACTION_FLOAT16_BYTES + OPTIONAL_ACTION_MASK_BYTES) as u32;
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
    write_obs_f16(writer, &sample.obs)?;
    write_u8(writer, sample.action)?;
    write_mask_u8(writer, &sample.legal_mask)?;
    write_i32_le(writer, sample.score_delta)?;
    write_u8(writer, sample.grp_label)?;
    write_binary_triplet(writer, &sample.tenpai)?;
    writer.write_all(&sample.opp_next)?;
    write_binary_mask_u8(writer, &sample.danger)?;
    write_binary_mask_u8(writer, &sample.danger_mask)?;

    if flags & FLAG_SAFETY_RESIDUAL != 0 {
        write_optional_action_f16(writer, sample.safety_residual.as_ref())?;
        write_optional_action_mask_u8(writer, sample.safety_residual_mask.as_ref())?;
    }
    if flags & FLAG_EXIT != 0 {
        write_optional_action_f16(writer, sample.exit_target.as_ref())?;
        write_optional_action_mask_u8(writer, sample.exit_mask.as_ref())?;
    }
    if flags & FLAG_DELTA_Q != 0 {
        write_optional_action_f16(writer, sample.delta_q_target.as_ref())?;
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
    Ok(())
}

fn parse_compact_sample(shard: &ShardMap, row_index: usize) -> Result<CompactSample, String> {
    let start = BC_SHARD_HEADER_SIZE as usize + row_index * shard.record_size;
    let end = start + shard.record_size;
    if end > shard.mmap.len() {
        return Err("BC shard row extends past file end".to_string());
    }
    let bytes = &shard.mmap[start..end];
    let mut cursor = 0usize;

    let mut obs = [0.0f32; OBS_SIZE];
    for value in &mut obs {
        *value = read_f16_as_f32(&bytes[cursor..cursor + 2]);
        cursor += 2;
    }

    let action = bytes[cursor];
    cursor += 1;

    let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
    for value in &mut legal_mask {
        *value = if bytes[cursor] > 0 { 1.0 } else { 0.0 };
        cursor += 1;
    }

    let score_delta = read_i32_le(&bytes[cursor..cursor + 4]);
    cursor += 4;
    let grp_label = bytes[cursor];
    cursor += 1;

    let mut tenpai = [0.0f32; OPPONENT_COUNT];
    for value in &mut tenpai {
        *value = if bytes[cursor] > 0 { 1.0 } else { 0.0 };
        cursor += 1;
    }

    let mut opp_next = [255u8; OPPONENT_COUNT];
    opp_next.copy_from_slice(&bytes[cursor..cursor + OPPONENT_COUNT]);
    cursor += OPPONENT_COUNT;

    let mut danger = [0.0f32; SPATIAL_TARGET_SIZE];
    for value in &mut danger {
        *value = if bytes[cursor] > 0 { 1.0 } else { 0.0 };
        cursor += 1;
    }

    let mut danger_mask = [0.0f32; SPATIAL_TARGET_SIZE];
    for value in &mut danger_mask {
        *value = if bytes[cursor] > 0 { 1.0 } else { 0.0 };
        cursor += 1;
    }

    let safety_residual = if shard.feature_flags & FLAG_SAFETY_RESIDUAL != 0 {
        let value = read_optional_action_f16(&bytes[cursor..cursor + OPTIONAL_ACTION_FLOAT16_BYTES]);
        cursor += OPTIONAL_ACTION_FLOAT16_BYTES;
        value
    } else {
        None
    };
    let safety_residual_mask = if shard.feature_flags & FLAG_SAFETY_RESIDUAL != 0 {
        let value = read_optional_action_mask_u8(&bytes[cursor..cursor + OPTIONAL_ACTION_MASK_BYTES]);
        cursor += OPTIONAL_ACTION_MASK_BYTES;
        value
    } else {
        None
    };

    let exit_target = if shard.feature_flags & FLAG_EXIT != 0 {
        let value = read_optional_action_f16(&bytes[cursor..cursor + OPTIONAL_ACTION_FLOAT16_BYTES]);
        cursor += OPTIONAL_ACTION_FLOAT16_BYTES;
        value
    } else {
        None
    };
    let exit_mask = if shard.feature_flags & FLAG_EXIT != 0 {
        let value = read_optional_action_mask_u8(&bytes[cursor..cursor + OPTIONAL_ACTION_MASK_BYTES]);
        cursor += OPTIONAL_ACTION_MASK_BYTES;
        value
    } else {
        None
    };

    let delta_q_target = if shard.feature_flags & FLAG_DELTA_Q != 0 {
        let value = read_optional_action_f16(&bytes[cursor..cursor + OPTIONAL_ACTION_FLOAT16_BYTES]);
        cursor += OPTIONAL_ACTION_FLOAT16_BYTES;
        value
    } else {
        None
    };
    let delta_q_mask = if shard.feature_flags & FLAG_DELTA_Q != 0 {
        let value = read_optional_action_mask_u8(&bytes[cursor..cursor + OPTIONAL_ACTION_MASK_BYTES]);
        let _ = value.as_ref();
        value
    } else {
        None
    };

    Ok(CompactSample {
        obs,
        action,
        legal_mask,
        score_delta,
        grp_label,
        tenpai,
        opp_next,
        danger,
        danger_mask,
        safety_residual,
        safety_residual_mask,
        exit_target,
        exit_mask,
        delta_q_target,
        delta_q_mask,
    })
}

fn write_obs_f16<W: Write>(writer: &mut W, values: &[f32; OBS_SIZE]) -> io::Result<()> {
    for &value in values {
        writer.write_all(&f16::from_f32(value).to_le_bytes())?;
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

fn write_binary_mask_u8<W: Write>(writer: &mut W, values: &[f32; SPATIAL_TARGET_SIZE]) -> io::Result<()> {
    for &value in values {
        writer.write_all(&[u8::from(value > 0.0)])?;
    }
    Ok(())
}

fn write_optional_action_f16<W: Write>(writer: &mut W, values: Option<&[f32; HYDRA_ACTION_SPACE]>) -> io::Result<()> {
    if let Some(values) = values {
        for &value in values {
            writer.write_all(&f16::from_f32(value).to_le_bytes())?;
        }
    } else {
        write_zero_bytes(writer, OPTIONAL_ACTION_FLOAT16_BYTES)?;
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

fn read_f16_as_f32(bytes: &[u8]) -> f32 {
    f16::from_le_bytes(bytes[0..2].try_into().expect("f16 slice")).to_f32()
}

fn read_optional_action_f16(bytes: &[u8]) -> Option<[f32; HYDRA_ACTION_SPACE]> {
    let mut out = [0.0f32; HYDRA_ACTION_SPACE];
    let mut any = false;
    for (index, value) in out.iter_mut().enumerate() {
        let start = index * 2;
        let decoded = read_f16_as_f32(&bytes[start..start + 2]);
        if decoded != 0.0 {
            any = true;
        }
        *value = decoded;
    }
    any.then_some(out)
}

fn read_optional_action_mask_u8(bytes: &[u8]) -> Option<[f32; HYDRA_ACTION_SPACE]> {
    let mut out = [0.0f32; HYDRA_ACTION_SPACE];
    let mut any = false;
    for (index, value) in out.iter_mut().enumerate() {
        if bytes[index] > 0 {
            *value = 1.0;
            any = true;
        }
    }
    any.then_some(out)
}

fn policy_target_from_actions<B: Backend>(
    actions: Tensor<B, 1, Int>,
    batch_size: usize,
) -> Tensor<B, 2> {
    let mut one_hot = vec![0.0f32; batch_size * HYDRA_ACTION_SPACE];
    let action_data = actions.clone().into_data().convert::<i64>();
    let action_values = action_data
        .as_slice::<i64>()
        .expect("action tensor should be readable as i64");
    for (row, &action) in action_values.iter().enumerate() {
        let action = action as usize;
        if action < HYDRA_ACTION_SPACE {
            one_hot[row * HYDRA_ACTION_SPACE + action] = 1.0;
        }
    }
    Tensor::<B, 1>::from_floats(one_hot.as_slice(), &actions.device())
        .reshape([batch_size, HYDRA_ACTION_SPACE])
}

fn permute_opp_next_targets(opp_next: [u8; 3], perm: &[u8; 3]) -> [u8; 3] {
    let mut out = opp_next;
    for tile in &mut out {
        if *tile < 34 {
            *tile = hydra_core::tile::permute_tile_type(*tile, perm);
        }
    }
    out
}

fn permute_spatial_targets_3x34(values: [f32; 102], perm: &[u8; 3]) -> [f32; 102] {
    let mut out = [0.0f32; 102];
    for opp in 0..3usize {
        let start = opp * 34;
        for tile in 0..34usize {
            let new_tile = hydra_core::tile::permute_tile_type(tile as u8, perm) as usize;
            out[start + new_tile] = values[start + tile];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_core::action::HYDRA_ACTION_SPACE;

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
}
