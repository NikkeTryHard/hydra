//! BC shard writer helpers for the frozen host binary format.

use std::fs;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use hydra_data_core::sample::MjaiSample;

use crate::host::record_size_for_flags;
use crate::manifest::{
    BC_SHARD_HEADER_SIZE, BC_SHARD_MAGIC, BC_SHARD_VERSION, BcShardDescriptor, BcShardSplit,
    BcShardSplitManifest, FLAG_DELTA_Q, FLAG_EXIT, FLAG_SAFETY_RESIDUAL, OPPONENT_COUNT,
    OPTIONAL_ACTION_FLOAT32_BYTES, OPTIONAL_ACTION_MASK_BYTES, ORACLE_FLOAT32_BYTES, PLAYER_COUNT,
    SPATIAL_TARGET_SIZE, TILE_COUNT,
};

/// Active writer for one BC shard file.
pub struct ActiveShardWriter {
    split: BcShardSplit,
    shard_index: usize,
    file_name: String,
    first_sample_index: u64,
    sample_count: u64,
    feature_flags: u32,
    record_size: u32,
    writer: BufWriter<fs::File>,
}

/// Incremental state for building one split's BC shard files.
pub struct SplitBuildState {
    split: BcShardSplit,
    next_shard_index: usize,
    total_samples: u64,
    feature_flags: u32,
    record_size: u32,
    shards: Vec<BcShardDescriptor>,
    active: Option<ActiveShardWriter>,
}

impl SplitBuildState {
    /// Creates split build state for `split` and `feature_flags`.
    pub fn new(split: BcShardSplit, feature_flags: u32) -> Self {
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

    /// Pushes all samples from one game into this split, opening or rotating shards as needed.
    pub fn push_samples(
        &mut self,
        output_dir: &Path,
        shard_samples: usize,
        samples: &[MjaiSample],
    ) -> io::Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        let game_samples = samples.len() as u64;
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
        active.write_samples(samples)?;
        self.total_samples += game_samples;
        Ok(())
    }

    /// Finishes any active shard and records its descriptor.
    pub fn finish_active(&mut self) -> io::Result<()> {
        let Some(active) = self.active.take() else {
            return Ok(());
        };
        let descriptor = active.finish()?;
        self.shards.push(descriptor);
        Ok(())
    }

    /// Finalizes this split and returns its manifest.
    pub fn finalize(mut self) -> io::Result<BcShardSplitManifest> {
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
    /// Opens a new active shard writer in `output_dir`.
    pub fn new(
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

    /// Writes samples into the active shard.
    pub fn write_samples(&mut self, samples: &[MjaiSample]) -> io::Result<()> {
        for sample in samples {
            write_sample_record(&mut self.writer, sample, self.feature_flags)?;
            self.sample_count += 1;
        }
        Ok(())
    }

    /// Finishes the shard and returns its descriptor.
    pub fn finish(mut self) -> io::Result<BcShardDescriptor> {
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

/// Writes one BC shard header.
pub fn write_shard_header<W: Write>(
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

/// Writes one BC shard sample record.
pub fn write_sample_record<W: Write>(
    writer: &mut W,
    sample: &MjaiSample,
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

#[cfg(test)]
mod tests;
