//! Mmap BC shard reader and backend-agnostic host-batch collation.

use std::fs;
use std::path::Path;

use memmap2::{Advice, Mmap};

use crate::host::{BcShardHostBatch, BcShardHostScratch};
use crate::manifest::{
    BC_SHARD_HEADER_SIZE, BcShardManifest, BcShardSplit, FLAG_DELTA_Q, FLAG_EXIT,
    FLAG_SAFETY_RESIDUAL, validate_bc_shard_manifest_contract,
};

mod augment;
mod decode;
mod header;
mod obs;

use augment::suit_permutation;
use decode::decode_row_bytes;
use header::verify_shard_header;

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
