//! Mmap BC shard reader and backend-agnostic host-batch collation.

use std::fs;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use hydra_data_core::sample::{score_delta_to_bin, score_delta_to_value};
use memmap2::{Advice, Mmap};

use crate::compact::{
    decode_counts_threshold_planes, unpack_action_mask_into, unpack_binary_mask_into,
    unpack_spatial_mask_into, unpack_tile_counts,
};
use crate::host::{BcShardHostBatch, BcShardHostScratch, GRP_CLASS_COUNT};
use crate::manifest::{
    BC_DENSE_SHARD_MAGIC, BC_SHARD_HEADER_SIZE, BC_SHARD_LAYOUT_VERSION, BC_SHARD_MAGIC,
    BC_SHARD_VERSION, BcShardDescriptor, BcShardManifest, BcShardSplit,
    COMPACT_OBS_BASELINE_FACT_BYTES, COMPACT_OBS_DENSE_BYTES, COMPACT_OBS_SCALAR_REPEATED_BYTES,
    DENSE_REBUILD_MESSAGE, FLAG_BELIEF_FIELDS, FLAG_DELTA_Q, FLAG_EXIT, FLAG_MIXTURE_WEIGHTS,
    FLAG_SAFETY_RESIDUAL, OPPONENT_COUNT, PACKED_ACTION_MASK_BYTES, PACKED_SPATIAL_MASK_BYTES,
    PLAYER_COUNT, SPATIAL_TARGET_SIZE, TILE_COUNT, TILE34_BITSET_BYTES, TILE34_COUNT_BYTES,
    checked_compact_record_size, validate_bc_shard_manifest_contract,
};

/// IEEE 754 bit pattern for `1.0f32`.
const F32_ONE_BITS: u32 = 0x3F80_0000;
/// Number of score bins materialized by the host reader.
const SCORE_BINS: usize = hydra_data_core::sample::SCORE_BINS;

/// Mmap-backed BC shard reader.
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

/// Reads and validates a BC shard manifest from disk.
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

/// Loads an mmap-backed BC shard reader for `split`.
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
        // SAFETY: file stays mapped by `Mmap`; access is read-only and bounded by header/record checks.
        let mmap = unsafe {
            Mmap::map(&file)
                .map_err(|err| format!("failed to mmap BC shard {}: {err}", path.display()))?
        };
        let _ = mmap.advise(Advice::Sequential);
        verify_shard_header(&mmap, split, shard)?;
        let expected_len = (BC_SHARD_HEADER_SIZE as u64)
            .checked_add(
                shard
                    .sample_count
                    .checked_mul(u64::from(shard.record_size))
                    .ok_or_else(|| "BC shard byte length overflow".to_string())?,
            )
            .ok_or_else(|| "BC shard byte length overflow".to_string())?;
        if mmap.len() as u64 != expected_len {
            return Err(format!(
                "BC shard {} length {} does not match header + records {}",
                path.display(),
                mmap.len(),
                expected_len
            ));
        }
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
    /// Returns total samples available in this reader.
    pub fn sample_count(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.sample_count as usize)
            .sum()
    }

    /// Returns shard feature flags, or zero for an empty reader.
    pub fn feature_flags(&self) -> u32 {
        self.shards.first().map_or(0, |s| s.feature_flags)
    }

    /// Creates reusable host scratch sized for this reader's features.
    pub fn new_scratch(&self, batch_size: usize) -> BcShardHostScratch {
        let flags = self.feature_flags();
        BcShardHostScratch::new(
            batch_size,
            flags & FLAG_SAFETY_RESIDUAL != 0,
            flags & FLAG_EXIT != 0,
            flags & FLAG_DELTA_Q != 0,
        )
    }

    /// CPU-only batch collation from arbitrary sample indices.
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

    /// CPU-only batch collation from a contiguous sample range.
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

    /// Collates arbitrary sample indices into pre-allocated host scratch.
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

    /// Collates a contiguous sample range into pre-allocated host scratch.
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

        macro_rules! collate_range_loop {
            ($write_row:expr) => {
                for row in 0..len {
                    let sample_index = start + row;
                    let shard = &self.shards[shard_index];

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
                            // SAFETY: prefetch is a non-dereferencing CPU hint; pointer comes from mmap bounds.
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
    let bytes = shard
        .mmap
        .get(start..end)
        .ok_or_else(|| "BC shard record outside mmap bounds".to_string())?;
    decode_row_bytes(bytes, shard.feature_flags, row, scratch, None)
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
    let bytes = shard
        .mmap
        .get(start..end)
        .ok_or_else(|| "BC shard record outside mmap bounds".to_string())?;
    let perm = suit_permutation(sample_index);
    decode_row_bytes(bytes, shard.feature_flags, row, scratch, Some(perm))
}

fn decode_row_bytes(
    bytes: &[u8],
    feature_flags: u32,
    row: usize,
    scratch: &mut BcShardHostScratch,
    suit_perm: Option<[usize; 3]>,
) -> Result<(), String> {
    let mut cursor = 0usize;
    let obs_facts = take(bytes, &mut cursor, COMPACT_OBS_BASELINE_FACT_BYTES)?;
    let obs_scalars = take(bytes, &mut cursor, COMPACT_OBS_SCALAR_REPEATED_BYTES)?;
    let obs_dense = take(bytes, &mut cursor, COMPACT_OBS_DENSE_BYTES)?;
    let obs_dst = &mut scratch.obs_flat[row * OBS_SIZE..(row + 1) * OBS_SIZE];
    decode_compact_obs(obs_facts, obs_scalars, obs_dense, obs_dst)?;
    if let Some(perm) = suit_perm {
        let mut unpermuted = [0.0f32; OBS_SIZE];
        unpermuted.copy_from_slice(obs_dst);
        augment_obs_suit(&unpermuted, perm, obs_dst);
    }

    scratch.actions[row] = take(bytes, &mut cursor, 1)?[0] as i64;

    let legal = take_array::<PACKED_ACTION_MASK_BYTES>(bytes, &mut cursor)?;
    let legal_dst =
        &mut scratch.legal_mask_flat[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
    if let Some(perm) = suit_perm {
        let mut unpermuted = [0.0f32; HYDRA_ACTION_SPACE];
        unpack_action_mask_into(legal, &mut unpermuted).map_err(|err| err.to_string())?;
        let mut packed_unpermuted = [0u8; HYDRA_ACTION_SPACE];
        for (dst, &src) in packed_unpermuted.iter_mut().zip(&unpermuted) {
            *dst = u8::from(src != 0.0);
        }
        expand_and_augment_mask_into(&packed_unpermuted, &action_permutation(perm), legal_dst);
    } else {
        unpack_action_mask_into(legal, legal_dst).map_err(|err| err.to_string())?;
    }

    let score_delta = read_i32_le(take_array::<4>(bytes, &mut cursor)?);
    scratch.value_target[row] = score_delta_to_value(score_delta);
    let score_bin = score_delta_to_bin(score_delta);
    scratch.score_pdf_flat[row * SCORE_BINS + score_bin] = 1.0;
    let cdf_start = row * SCORE_BINS;
    for idx in score_bin..SCORE_BINS {
        scratch.score_cdf_flat[cdf_start + idx] = 1.0;
    }

    let grp_label = take(bytes, &mut cursor, 1)?[0] as usize;
    if grp_label < GRP_CLASS_COUNT {
        scratch.grp_target_flat[row * GRP_CLASS_COUNT + grp_label] = 1.0;
    }

    let oracle = read_f32_array::<PLAYER_COUNT>(take(bytes, &mut cursor, PLAYER_COUNT * 4)?);
    let oracle_dst = &mut scratch.oracle_target_flat[row * PLAYER_COUNT..(row + 1) * PLAYER_COUNT];
    oracle_dst.copy_from_slice(&oracle);
    scratch.oracle_target_mask[row] = f32::from(take(bytes, &mut cursor, 1)?[0] != 0);

    let tenpai = take(bytes, &mut cursor, 1)?;
    unpack_binary_mask_into(
        tenpai,
        OPPONENT_COUNT,
        &mut scratch.tenpai_flat[row * OPPONENT_COUNT..(row + 1) * OPPONENT_COUNT],
    )
    .map_err(|err| err.to_string())?;

    let opp_next = take(bytes, &mut cursor, OPPONENT_COUNT)?;
    let opp_base = row * SPATIAL_TARGET_SIZE;
    for (opponent, &tile) in opp_next.iter().enumerate() {
        if (tile as usize) < TILE_COUNT {
            scratch.opp_next_flat[opp_base + opponent * TILE_COUNT + tile as usize] = 1.0;
        }
    }

    let danger = take_array::<PACKED_SPATIAL_MASK_BYTES>(bytes, &mut cursor)?;
    let danger_dst =
        &mut scratch.danger_flat[row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE];
    if suit_perm.is_some() {
        let mut unpermuted = [0.0f32; SPATIAL_TARGET_SIZE];
        unpack_spatial_mask_into(danger, &mut unpermuted).map_err(|err| err.to_string())?;
        expand_spatial_mask_f32(&unpermuted, danger_dst, suit_perm);
    } else {
        unpack_spatial_mask_into(danger, danger_dst).map_err(|err| err.to_string())?;
    }

    let danger_mask = take_array::<PACKED_SPATIAL_MASK_BYTES>(bytes, &mut cursor)?;
    let danger_mask_dst =
        &mut scratch.danger_mask_flat[row * SPATIAL_TARGET_SIZE..(row + 1) * SPATIAL_TARGET_SIZE];
    if suit_perm.is_some() {
        let mut unpermuted = [0.0f32; SPATIAL_TARGET_SIZE];
        unpack_spatial_mask_into(danger_mask, &mut unpermuted).map_err(|err| err.to_string())?;
        expand_spatial_mask_f32(&unpermuted, danger_mask_dst, suit_perm);
    } else {
        unpack_spatial_mask_into(danger_mask, danger_mask_dst).map_err(|err| err.to_string())?;
    }

    if feature_flags & FLAG_SAFETY_RESIDUAL != 0 {
        decode_optional_action_pair(
            bytes,
            &mut cursor,
            row,
            scratch,
            suit_perm,
            OptionalKind::Safety,
        )?;
    }
    if feature_flags & FLAG_EXIT != 0 {
        decode_optional_action_pair(
            bytes,
            &mut cursor,
            row,
            scratch,
            suit_perm,
            OptionalKind::Exit,
        )?;
    }
    if feature_flags & FLAG_DELTA_Q != 0 {
        decode_optional_action_pair(
            bytes,
            &mut cursor,
            row,
            scratch,
            suit_perm,
            OptionalKind::DeltaQ,
        )?;
    }
    if feature_flags & FLAG_BELIEF_FIELDS != 0 {
        let _ = take(bytes, &mut cursor, 16 * TILE_COUNT * 4)?;
    }
    if feature_flags & FLAG_MIXTURE_WEIGHTS != 0 {
        let _ = take(bytes, &mut cursor, PLAYER_COUNT * 4)?;
    }
    if cursor != bytes.len() {
        return Err(format!(
            "BC shard compact record has {} trailing byte(s)",
            bytes.len() - cursor
        ));
    }

    Ok(())
}

enum OptionalKind {
    Safety,
    Exit,
    DeltaQ,
}

fn decode_optional_action_pair(
    bytes: &[u8],
    cursor: &mut usize,
    row: usize,
    scratch: &mut BcShardHostScratch,
    suit_perm: Option<[usize; 3]>,
    kind: OptionalKind,
) -> Result<(), String> {
    let values = take(bytes, cursor, HYDRA_ACTION_SPACE * 4)?;
    let mask = take_array::<PACKED_ACTION_MASK_BYTES>(bytes, cursor)?;
    let action_perm = suit_perm.map(action_permutation);
    match kind {
        OptionalKind::Safety => {
            if let Some(buf) = scratch.safety_target_flat.as_mut() {
                let dst = &mut buf[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
                if let Some(perm) = action_perm.as_ref() {
                    augment_action_f32_from_bytes_into(values, perm, dst);
                } else {
                    read_optional_action_f32_into(values, dst);
                }
            }
            if let Some(buf) = scratch.safety_mask_flat.as_mut() {
                let dst = &mut buf[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
                if let Some(perm) = action_perm.as_ref() {
                    let mut unpermuted = [0.0f32; HYDRA_ACTION_SPACE];
                    unpack_action_mask_into(mask, &mut unpermuted)
                        .map_err(|err| err.to_string())?;
                    let mut packed_unpermuted = [0u8; HYDRA_ACTION_SPACE];
                    for (dst, &src) in packed_unpermuted.iter_mut().zip(&unpermuted) {
                        *dst = u8::from(src != 0.0);
                    }
                    expand_and_augment_mask_into(&packed_unpermuted, perm, dst);
                } else {
                    unpack_action_mask_into(mask, dst).map_err(|err| err.to_string())?;
                }
            }
        }
        OptionalKind::Exit => {
            if let Some(buf) = scratch.exit_target_flat.as_mut() {
                let dst = &mut buf[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
                if let Some(perm) = action_perm.as_ref() {
                    augment_action_f32_from_bytes_into(values, perm, dst);
                } else {
                    read_optional_action_f32_into(values, dst);
                }
            }
            if let Some(buf) = scratch.exit_mask_flat.as_mut() {
                let dst = &mut buf[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
                if let Some(perm) = action_perm.as_ref() {
                    let mut unpermuted = [0.0f32; HYDRA_ACTION_SPACE];
                    unpack_action_mask_into(mask, &mut unpermuted)
                        .map_err(|err| err.to_string())?;
                    let mut packed_unpermuted = [0u8; HYDRA_ACTION_SPACE];
                    for (dst, &src) in packed_unpermuted.iter_mut().zip(&unpermuted) {
                        *dst = u8::from(src != 0.0);
                    }
                    expand_and_augment_mask_into(&packed_unpermuted, perm, dst);
                } else {
                    unpack_action_mask_into(mask, dst).map_err(|err| err.to_string())?;
                }
            }
        }
        OptionalKind::DeltaQ => {
            if let Some(buf) = scratch.delta_q_target_flat.as_mut() {
                let dst = &mut buf[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
                if let Some(perm) = action_perm.as_ref() {
                    augment_action_f32_from_bytes_into(values, perm, dst);
                } else {
                    read_optional_action_f32_into(values, dst);
                }
            }
            if let Some(buf) = scratch.delta_q_mask_flat.as_mut() {
                let dst = &mut buf[row * HYDRA_ACTION_SPACE..(row + 1) * HYDRA_ACTION_SPACE];
                if let Some(perm) = action_perm.as_ref() {
                    let mut unpermuted = [0.0f32; HYDRA_ACTION_SPACE];
                    unpack_action_mask_into(mask, &mut unpermuted)
                        .map_err(|err| err.to_string())?;
                    let mut packed_unpermuted = [0u8; HYDRA_ACTION_SPACE];
                    for (dst, &src) in packed_unpermuted.iter_mut().zip(&unpermuted) {
                        *dst = u8::from(src != 0.0);
                    }
                    expand_and_augment_mask_into(&packed_unpermuted, perm, dst);
                } else {
                    unpack_action_mask_into(mask, dst).map_err(|err| err.to_string())?;
                }
            }
        }
    }
    Ok(())
}

fn verify_shard_header(
    mmap: &Mmap,
    split: BcShardSplit,
    descriptor: &BcShardDescriptor,
) -> Result<(), String> {
    if mmap.len() < BC_SHARD_HEADER_SIZE as usize {
        return Err("BC shard file too small for header".to_string());
    }
    if mmap[..8] == BC_DENSE_SHARD_MAGIC {
        return Err(DENSE_REBUILD_MESSAGE.to_string());
    }
    if mmap[..8] != BC_SHARD_MAGIC {
        return Err("invalid compact BC shard magic".to_string());
    }
    let version = read_u32_le(&mmap[8..12]);
    if version != BC_SHARD_VERSION {
        return Err(format!("unsupported compact BC shard version {version}"));
    }
    let header_size = read_u32_le(&mmap[12..16]);
    if header_size != BC_SHARD_HEADER_SIZE {
        return Err("BC shard header size mismatch".to_string());
    }
    let split_id = read_u32_le(&mmap[20..24]);
    if split_id != split.split_id() {
        return Err("BC shard split mismatch".to_string());
    }
    let header_record_size = read_u32_le(&mmap[16..20]);
    if header_record_size != descriptor.record_size {
        return Err("BC shard record size mismatch".to_string());
    }
    let header_sample_count = u64::from_le_bytes(mmap[28..36].try_into().expect("u64 slice"));
    if header_sample_count != descriptor.sample_count {
        return Err("BC shard sample count mismatch".to_string());
    }
    if read_u32_le(&mmap[36..40]) != NUM_CHANNELS as u32
        || read_u32_le(&mmap[40..44]) != TILE_COUNT as u32
    {
        return Err("BC shard encoder contract mismatch".to_string());
    }
    if read_u32_le(&mmap[44..48]) != HYDRA_ACTION_SPACE as u32 {
        return Err("BC shard action contract mismatch".to_string());
    }
    let header_first_sample_index = u64::from_le_bytes(mmap[48..56].try_into().expect("u64 slice"));
    if header_first_sample_index != descriptor.first_sample_index {
        return Err("BC shard first sample index mismatch".to_string());
    }
    let header_flags = read_u32_le(&mmap[56..60]);
    if header_flags != descriptor.feature_flags {
        return Err("BC shard feature flags mismatch".to_string());
    }
    let layout_version = read_u32_le(&mmap[60..64]);
    if layout_version != BC_SHARD_LAYOUT_VERSION {
        return Err(format!(
            "unsupported BC shard layout version {layout_version}"
        ));
    }
    let expected_record_size = checked_compact_record_size(descriptor.feature_flags)?;
    if descriptor.record_size != expected_record_size {
        return Err(format!(
            "BC shard record_size {} incompatible with current compact binary (expected {expected_record_size} for flags {:#x}). Rebuild shards from replay.",
            descriptor.record_size, descriptor.feature_flags,
        ));
    }
    Ok(())
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], String> {
    let start = *cursor;
    let end = start
        .checked_add(len)
        .ok_or_else(|| "BC shard decode cursor overflow".to_string())?;
    let slice = bytes
        .get(start..end)
        .ok_or_else(|| "BC shard record truncated".to_string())?;
    *cursor = end;
    Ok(slice)
}

fn take_array<'a, const N: usize>(
    bytes: &'a [u8],
    cursor: &mut usize,
) -> Result<&'a [u8; N], String> {
    take(bytes, cursor, N).map(|slice| slice.try_into().expect("fixed array length"))
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
        // SAFETY: `out` has exactly N f32 worth of initialized bytes copied from `bytes`.
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
    debug_assert_eq!(dst.len(), HYDRA_ACTION_SPACE);
    let mut any = false;
    for src in 0..37 {
        let off = src * 4;
        #[cfg(target_endian = "little")]
        let value = {
            let mut bits = [0u8; 4];
            bits.copy_from_slice(&bytes[off..off + 4]);
            f32::from_ne_bytes(bits)
        };
        #[cfg(not(target_endian = "little"))]
        let value = read_f32_le(&bytes[off..off + 4]);
        any |= value != 0.0;
        dst[action_perm[src]] = value;
    }
    for (action, out) in dst.iter_mut().enumerate().take(HYDRA_ACTION_SPACE).skip(37) {
        let off = action * 4;
        #[cfg(target_endian = "little")]
        let value = {
            let mut bits = [0u8; 4];
            bits.copy_from_slice(&bytes[off..off + 4]);
            f32::from_ne_bytes(bits)
        };
        #[cfg(not(target_endian = "little"))]
        let value = read_f32_le(&bytes[off..off + 4]);
        any |= value != 0.0;
        *out = value;
    }
    any
}

#[inline]
fn read_optional_action_f32_into(bytes: &[u8], dst: &mut [f32]) -> bool {
    let mut any = false;
    let values = read_f32_array::<HYDRA_ACTION_SPACE>(bytes);
    for (out, value) in dst.iter_mut().zip(values) {
        any |= value != 0.0;
        *out = value;
    }
    any
}

#[inline]
fn expand_and_augment_mask_into(bytes: &[u8], action_perm: &[usize; 37], dst: &mut [f32]) -> bool {
    let mut any = false;
    for src in 0..37 {
        let nonzero = bytes[src] != 0;
        any |= nonzero;
        dst[action_perm[src]] = f32::from_bits(u32::from(nonzero) * F32_ONE_BITS);
    }
    for action in 37..HYDRA_ACTION_SPACE {
        let nonzero = bytes[action] != 0;
        any |= nonzero;
        dst[action] = f32::from_bits(u32::from(nonzero) * F32_ONE_BITS);
    }
    any
}

fn expand_spatial_mask_f32(values: &[f32], dst: &mut [f32], suit_perm: Option<[usize; 3]>) {
    if let Some(perm) = suit_perm {
        for opponent in 0..OPPONENT_COUNT {
            for tile in 0..TILE_COUNT {
                let dst_tile = permute_tile(tile, perm);
                let src_idx = opponent * TILE_COUNT + tile;
                let dst_idx = opponent * TILE_COUNT + dst_tile;
                dst[dst_idx] = values[src_idx];
            }
        }
    } else {
        dst.copy_from_slice(values);
    }
}

fn decode_compact_obs(
    facts: &[u8],
    scalars: &[u8],
    dense: &[u8],
    dst: &mut [f32],
) -> Result<(), String> {
    if !scalars.is_empty() || !dense.is_empty() {
        return Err("BC shard compact observation advanced sections must be empty".to_string());
    }
    dst.fill(0.0);
    decode_baseline_obs_facts(facts, dst)
}
fn decode_baseline_obs_facts(bytes: &[u8], dst: &mut [f32]) -> Result<(), String> {
    if bytes.len() != COMPACT_OBS_BASELINE_FACT_BYTES {
        return Err("BC shard compact observation fact section has invalid length".to_string());
    }
    let mut cursor = 0usize;
    decode_tile_counts(
        take_array::<TILE34_COUNT_BYTES>(bytes, &mut cursor)?,
        dst,
        0,
    )?;
    decode_tile_counts(
        take_array::<TILE34_COUNT_BYTES>(bytes, &mut cursor)?,
        dst,
        4,
    )?;
    decode_tile_bitset(
        take_array::<TILE34_BITSET_BYTES>(bytes, &mut cursor)?,
        dst,
        8,
    )?;
    decode_tile_bitset(
        take_array::<TILE34_BITSET_BYTES>(bytes, &mut cursor)?,
        dst,
        9,
    )?;
    decode_tile_bitset(
        take_array::<TILE34_BITSET_BYTES>(bytes, &mut cursor)?,
        dst,
        10,
    )?;
    decode_discard_facts(bytes, &mut cursor, dst)?;
    decode_meld_facts(bytes, &mut cursor, dst)?;
    decode_dora_facts(take(bytes, &mut cursor, TILE_COUNT)?, dst)?;
    decode_aka_facts(take(bytes, &mut cursor, 1)?[0], dst);
    decode_metadata_facts(bytes, &mut cursor, dst)?;
    decode_safety_facts(bytes, &mut cursor, dst)?;
    debug_assert_eq!(cursor, COMPACT_OBS_BASELINE_FACT_BYTES);
    Ok(())
}

fn decode_tile_counts(
    bytes: &[u8; TILE34_COUNT_BYTES],
    dst: &mut [f32],
    channel_start: usize,
) -> Result<(), String> {
    let mut counts = [0u8; TILE_COUNT];
    unpack_tile_counts(bytes, &mut counts).map_err(|err| err.to_string())?;
    decode_counts_threshold_planes(&counts, dst, channel_start);
    Ok(())
}

fn decode_tile_bitset(
    bytes: &[u8; TILE34_BITSET_BYTES],
    dst: &mut [f32],
    channel: usize,
) -> Result<(), String> {
    let start = channel * TILE_COUNT;
    unpack_binary_mask_into(bytes, TILE_COUNT, &mut dst[start..start + TILE_COUNT])
        .map_err(|err| err.to_string())
}

fn decode_channel_bitsets(
    bytes: &[u8],
    cursor: &mut usize,
    dst: &mut [f32],
    channel_start: usize,
    channel_count: usize,
) -> Result<(), String> {
    for channel in channel_start..channel_start + channel_count {
        decode_tile_bitset(
            take_array::<TILE34_BITSET_BYTES>(bytes, cursor)?,
            dst,
            channel,
        )?;
    }
    Ok(())
}

fn decode_discard_facts(bytes: &[u8], cursor: &mut usize, dst: &mut [f32]) -> Result<(), String> {
    for player in 0..4usize {
        let base = 11 + player * 3;
        decode_tile_bitset(take_array::<TILE34_BITSET_BYTES>(bytes, cursor)?, dst, base)?;
        decode_tile_bitset(
            take_array::<TILE34_BITSET_BYTES>(bytes, cursor)?,
            dst,
            base + 1,
        )?;
        let row = (base + 2) * TILE_COUNT;
        for tile in 0..TILE_COUNT {
            let idx = read_u32_le(take(bytes, cursor, 4)?);
            dst[row + tile] = temporal_value(idx)?;
        }
    }
    Ok(())
}

fn temporal_value(index: u32) -> Result<f32, String> {
    #[allow(
        clippy::excessive_precision,
        reason = "table entries preserve encoder discard decay exactly"
    )]
    const DISCARD_EXP_TABLE: [f32; 31] = [
        1.0,
        0.818_730_8,
        0.670_320_0,
        0.548_811_6,
        0.449_329_0,
        0.367_879_5,
        0.301_194_2,
        0.246_597_0,
        0.201_896_5,
        0.165_298_9,
        0.135_335_3,
        0.110_803_2,
        0.090_717_96,
        0.074_273_58,
        0.060_810_06,
        0.049_787_07,
        0.040_762_20,
        0.033_373_27,
        0.027_323_72,
        0.022_370_77,
        0.018_315_64,
        0.014_995_58,
        0.012_277_34,
        0.010_051_84,
        0.008_229_747,
        0.006_737_947,
        0.005_516_564,
        0.004_516_581,
        0.003_697_864,
        0.003_027_555,
        0.002_478_752,
    ];
    if index == u32::MAX {
        return Ok(0.0);
    }
    DISCARD_EXP_TABLE
        .get(index as usize)
        .copied()
        .ok_or_else(|| format!("BC shard discard temporal index {index} out of range"))
}

fn decode_meld_facts(bytes: &[u8], cursor: &mut usize, dst: &mut [f32]) -> Result<(), String> {
    decode_channel_bitsets(bytes, cursor, dst, 23, 12)
}

fn decode_dora_facts(bytes: &[u8], dst: &mut [f32]) -> Result<(), String> {
    for (tile, &count) in bytes.iter().enumerate() {
        if count > 5 {
            return Err(format!("BC shard dora count {count} out of range"));
        }
        for threshold in 0..5usize {
            if count as usize > threshold {
                dst[(35 + threshold) * TILE_COUNT + tile] = 1.0;
            }
        }
    }
    Ok(())
}

fn decode_aka_facts(flags: u8, dst: &mut [f32]) {
    for suit in 0..3usize {
        if (flags & (1u8 << suit)) != 0 {
            dst[(40 + suit) * TILE_COUNT..(41 + suit) * TILE_COUNT].fill(1.0);
        }
    }
}

fn decode_metadata_facts(bytes: &[u8], cursor: &mut usize, dst: &mut [f32]) -> Result<(), String> {
    decode_repeated_bool_channels(
        take_array::<TILE34_BITSET_BYTES>(bytes, cursor)?,
        dst,
        43,
        4,
    )?;
    for channel in 47..55usize {
        let value = read_f32_array::<1>(take(bytes, cursor, 4)?)[0];
        dst[channel * TILE_COUNT..(channel + 1) * TILE_COUNT].fill(value);
    }
    decode_repeated_bool_channels(
        take_array::<TILE34_BITSET_BYTES>(bytes, cursor)?,
        dst,
        55,
        4,
    )?;
    for channel in 59..62usize {
        let value = read_f32_array::<1>(take(bytes, cursor, 4)?)[0];
        dst[channel * TILE_COUNT..(channel + 1) * TILE_COUNT].fill(value);
    }
    Ok(())
}

fn decode_repeated_bool_channels(
    bytes: &[u8; TILE34_BITSET_BYTES],
    dst: &mut [f32],
    channel_start: usize,
    channel_count: usize,
) -> Result<(), String> {
    let mut values = [0.0f32; TILE_COUNT];
    unpack_binary_mask_into(bytes, TILE_COUNT, &mut values).map_err(|err| err.to_string())?;
    for (channel_offset, &value) in values.iter().enumerate().take(channel_count) {
        if value != 0.0 {
            let channel = channel_start + channel_offset;
            dst[channel * TILE_COUNT..(channel + 1) * TILE_COUNT].fill(1.0);
        }
    }
    Ok(())
}

fn decode_safety_facts(bytes: &[u8], cursor: &mut usize, dst: &mut [f32]) -> Result<(), String> {
    decode_channel_bitsets(bytes, cursor, dst, 62, 9)?;
    for channel in 71..74usize {
        decode_dense_channel(take(bytes, cursor, TILE_COUNT * 4)?, dst, channel)?;
    }
    decode_channel_bitsets(bytes, cursor, dst, 74, 3)?;
    for channel in 77..80usize {
        decode_dense_channel(take(bytes, cursor, TILE_COUNT * 4)?, dst, channel)?;
    }
    decode_channel_bitsets(bytes, cursor, dst, 80, 5)
}

fn decode_dense_channel(bytes: &[u8], dst: &mut [f32], channel: usize) -> Result<(), String> {
    let values = read_f32_array::<TILE_COUNT>(bytes);
    dst[channel * TILE_COUNT..(channel + 1) * TILE_COUNT].copy_from_slice(&values);
    Ok(())
}

fn augment_obs_suit(values: &[f32; OBS_SIZE], suit_perm: [usize; 3], dst: &mut [f32]) {
    dst.fill(0.0);
    for channel in 0..192 {
        let src_base = channel * TILE_COUNT;
        let dst_base = src_base;
        for tile in 0..TILE_COUNT {
            let dst_tile = permute_tile(tile, suit_perm);
            dst[dst_base + dst_tile] = values[src_base + tile];
        }
    }
}

fn suit_permutation(sample_index: usize) -> [usize; 3] {
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    PERMS[sample_index % PERMS.len()]
}

fn permute_tile(tile: usize, suit_perm: [usize; 3]) -> usize {
    if tile < 27 {
        let suit = tile / 9;
        let rank = tile % 9;
        suit_perm[suit] * 9 + rank
    } else {
        tile
    }
}

fn action_permutation(suit_perm: [usize; 3]) -> [usize; 37] {
    let mut perm = [0usize; 37];
    let mut action = 0usize;
    while action < 37 {
        perm[action] = permute_tile(action, suit_perm);
        action += 1;
    }
    perm
}
