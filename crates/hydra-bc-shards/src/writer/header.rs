//! BC shard header write helpers.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::NUM_CHANNELS;

use crate::manifest::{
    BC_SHARD_HEADER_SIZE, BC_SHARD_LAYOUT_VERSION, BC_SHARD_MAGIC, BC_SHARD_VERSION,
    BcShardDescriptor, BcShardSplit, TILE_COUNT,
};

use super::primitives::{write_u32_le, write_u64_le};

/// Rewrites a shard file header from a finalized descriptor.
pub fn rewrite_shard_header_for_descriptor(
    path: &Path,
    descriptor: &BcShardDescriptor,
) -> io::Result<()> {
    let mut file = fs::OpenOptions::new().write(true).open(path)?;
    write_shard_header(
        &mut file,
        descriptor.split,
        descriptor.shard_index as u32,
        descriptor.sample_count,
        descriptor.first_sample_index,
        descriptor.feature_flags,
        descriptor.record_size,
    )?;
    file.flush()?;
    file.sync_all()
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
    write_u32_le(writer, BC_SHARD_LAYOUT_VERSION)?;
    write_u64_le(writer, 0)?;
    write_u64_le(writer, 0)?;
    Ok(())
}
