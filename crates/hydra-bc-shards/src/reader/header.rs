//! Shard header verification and primitive byte reads.

use std::mem::MaybeUninit;
use std::ptr;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::NUM_CHANNELS;
use memmap2::Mmap;

use crate::manifest::{
    BC_DENSE_SHARD_MAGIC, BC_SHARD_HEADER_SIZE, BC_SHARD_LAYOUT_VERSION, BC_SHARD_MAGIC,
    BC_SHARD_VERSION, BcShardDescriptor, BcShardSplit, DENSE_REBUILD_MESSAGE, TILE_COUNT,
    checked_compact_record_size,
};

pub(super) fn verify_shard_header(
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

pub(super) fn take<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    len: usize,
) -> Result<&'a [u8], String> {
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

pub(super) fn take_array<'a, const N: usize>(
    bytes: &'a [u8],
    cursor: &mut usize,
) -> Result<&'a [u8; N], String> {
    take(bytes, cursor, N).map(|slice| slice.try_into().expect("fixed array length"))
}

pub(super) fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes[0..4].try_into().expect("u32 slice"))
}

pub(super) fn read_i32_le(bytes: &[u8; 4]) -> i32 {
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
pub(super) fn read_f32_array<const N: usize>(bytes: &[u8]) -> [f32; N] {
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
