//! BC shard writer helpers for the frozen host binary format.

use std::fs;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use hydra_data_core::sample::MjaiSample;

use crate::host::record_size_for_flags;
use crate::manifest::{
    BcShardDescriptor, BcShardSplit, BcShardSplitManifest, checked_compact_record_size,
};

mod header;
mod masks;
mod obs;
mod primitives;
mod record;

pub use header::{rewrite_shard_header_for_descriptor, write_shard_header};
pub use record::{encode_sample_records, write_sample_record};

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
    /// Opens a new active shard writer in `output_dir` using the default shard file name.
    pub fn new(
        output_dir: &Path,
        split: BcShardSplit,
        shard_index: usize,
        first_sample_index: u64,
        feature_flags: u32,
    ) -> io::Result<Self> {
        let file_name = format!("{}-{shard_index:05}.hydra-bc", split.shard_prefix());
        Self::new_named(
            output_dir,
            split,
            shard_index,
            first_sample_index,
            feature_flags,
            file_name,
        )
    }

    /// Opens a new active shard writer in `output_dir` using `file_name`.
    pub fn new_named(
        output_dir: &Path,
        split: BcShardSplit,
        shard_index: usize,
        first_sample_index: u64,
        feature_flags: u32,
        file_name: String,
    ) -> io::Result<Self> {
        let path = output_dir.join(&file_name);
        let file = fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);
        let record_size = checked_compact_record_size(feature_flags).map_err(invalid_data)?;
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

    /// Writes already-validated compact records into the active shard.
    pub fn write_encoded_records(&mut self, records: &[u8], sample_count: usize) -> io::Result<()> {
        let expected_len = sample_count
            .checked_mul(self.record_size as usize)
            .ok_or_else(|| invalid_data("encoded BC shard record byte count overflow"))?;
        if records.len() != expected_len {
            return Err(invalid_data(
                "encoded BC shard records have invalid byte length",
            ));
        }
        self.writer.write_all(records)?;
        self.sample_count += sample_count as u64;
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

fn invalid_data(message: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

#[cfg(test)]
mod tests;
