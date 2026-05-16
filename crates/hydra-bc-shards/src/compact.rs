//! Compact BC shard bit/int packing helpers.

use hydra_core::action::HYDRA_ACTION_SPACE;

use crate::manifest::{PACKED_ACTION_MASK_BYTES, PACKED_SPATIAL_MASK_BYTES, TILE34_COUNT_BYTES};

const F32_ONE_BITS: u32 = 0x3F80_0000;

#[allow(
    dead_code,
    reason = "compact observation fact packing lands after initial label/mask cutover"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CompactEncodeError {
    BufferTooSmall,
    NonBinaryMask {
        index: usize,
    },
    CountOutOfRange {
        index: usize,
        value: u8,
    },
    ValueOutOfRange {
        name: &'static str,
        value: u8,
        max_exclusive: u8,
    },
}

impl std::fmt::Display for CompactEncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferTooSmall => write!(f, "compact encode buffer too small"),
            Self::NonBinaryMask { index } => {
                write!(f, "compact mask value at index {index} is not binary")
            }
            Self::CountOutOfRange { index, value } => {
                write!(
                    f,
                    "compact tile count at index {index} is {value}, expected 0..=4"
                )
            }
            Self::ValueOutOfRange {
                name,
                value,
                max_exclusive,
            } => {
                write!(
                    f,
                    "compact {name} value {value} out of range 0..{}",
                    max_exclusive - 1
                )
            }
        }
    }
}

impl std::error::Error for CompactEncodeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CompactDecodeError {
    BufferTooSmall,
    CountOutOfRange { index: usize, value: u8 },
}

impl std::fmt::Display for CompactDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferTooSmall => write!(f, "compact decode buffer too small"),
            Self::CountOutOfRange { index, value } => {
                write!(
                    f,
                    "compact tile count at index {index} is {value}, expected 0..=4"
                )
            }
        }
    }
}

impl std::error::Error for CompactDecodeError {}

pub(crate) struct BitWriter<'a> {
    bytes: &'a mut [u8],
    bit: usize,
}

impl<'a> BitWriter<'a> {
    pub(crate) fn new(bytes: &'a mut [u8]) -> Self {
        bytes.fill(0);
        Self { bytes, bit: 0 }
    }

    pub(crate) fn write_bit(&mut self, value: bool) -> Result<(), CompactEncodeError> {
        if self.bit / 8 >= self.bytes.len() {
            return Err(CompactEncodeError::BufferTooSmall);
        }
        if value {
            self.bytes[self.bit / 8] |= 1u8 << (self.bit % 8);
        }
        self.bit += 1;
        Ok(())
    }

    #[allow(
        dead_code,
        reason = "tile-count packing uses this in compact observation facts"
    )]
    pub(crate) fn write_bits(&mut self, value: u32, width: u8) -> Result<(), CompactEncodeError> {
        for offset in 0..width {
            self.write_bit(((value >> offset) & 1) != 0)?;
        }
        Ok(())
    }
}

pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit: 0 }
    }

    pub(crate) fn read_bit(&mut self) -> Result<bool, CompactDecodeError> {
        if self.bit / 8 >= self.bytes.len() {
            return Err(CompactDecodeError::BufferTooSmall);
        }
        let value = ((self.bytes[self.bit / 8] >> (self.bit % 8)) & 1) != 0;
        self.bit += 1;
        Ok(value)
    }

    #[allow(
        dead_code,
        reason = "tile-count unpacking uses this in compact observation facts"
    )]
    pub(crate) fn read_bits(&mut self, width: u8) -> Result<u32, CompactDecodeError> {
        let mut value = 0u32;
        for offset in 0..width {
            if self.read_bit()? {
                value |= 1u32 << offset;
            }
        }
        Ok(value)
    }
}

#[inline]
pub(crate) fn validate_u8_range(
    name: &'static str,
    value: u8,
    max_exclusive: u8,
) -> Result<(), CompactEncodeError> {
    if value < max_exclusive {
        Ok(())
    } else {
        Err(CompactEncodeError::ValueOutOfRange {
            name,
            value,
            max_exclusive,
        })
    }
}

pub(crate) fn pack_binary_f32_mask(
    src: &[f32],
    bits: usize,
    dst: &mut [u8],
) -> Result<(), CompactEncodeError> {
    if src.len() < bits || dst.len() * 8 < bits {
        return Err(CompactEncodeError::BufferTooSmall);
    }
    let mut writer = BitWriter::new(dst);
    for (idx, &value) in src.iter().take(bits).enumerate() {
        if value == 0.0 {
            writer.write_bit(false)?;
        } else if value == 1.0 {
            writer.write_bit(true)?;
        } else {
            return Err(CompactEncodeError::NonBinaryMask { index: idx });
        }
    }
    Ok(())
}

pub(crate) fn unpack_binary_mask_into(
    src: &[u8],
    bits: usize,
    dst: &mut [f32],
) -> Result<(), CompactDecodeError> {
    if src.len() * 8 < bits || dst.len() < bits {
        return Err(CompactDecodeError::BufferTooSmall);
    }
    let mut reader = BitReader::new(src);
    for slot in dst.iter_mut().take(bits) {
        *slot = f32::from_bits(u32::from(reader.read_bit()?) * F32_ONE_BITS);
    }
    Ok(())
}

pub(crate) fn pack_action_mask(
    src: &[f32; HYDRA_ACTION_SPACE],
    dst: &mut [u8; PACKED_ACTION_MASK_BYTES],
) -> Result<(), CompactEncodeError> {
    pack_binary_f32_mask(src, HYDRA_ACTION_SPACE, dst)
}

pub(crate) fn unpack_action_mask_into(
    src: &[u8; PACKED_ACTION_MASK_BYTES],
    dst: &mut [f32],
) -> Result<(), CompactDecodeError> {
    unpack_binary_mask_into(src, HYDRA_ACTION_SPACE, dst)
}

pub(crate) fn pack_spatial_mask(
    src: &[f32],
    dst: &mut [u8; PACKED_SPATIAL_MASK_BYTES],
) -> Result<(), CompactEncodeError> {
    pack_binary_f32_mask(src, 102, dst)
}

pub(crate) fn unpack_spatial_mask_into(
    src: &[u8; PACKED_SPATIAL_MASK_BYTES],
    dst: &mut [f32],
) -> Result<(), CompactDecodeError> {
    unpack_binary_mask_into(src, 102, dst)
}

#[allow(
    dead_code,
    reason = "compact observation fact packing lands after initial label/mask cutover"
)]
pub(crate) fn pack_tile_counts(
    src: &[u8; 34],
    dst: &mut [u8; TILE34_COUNT_BYTES],
) -> Result<(), CompactEncodeError> {
    let mut writer = BitWriter::new(dst);
    for (idx, &count) in src.iter().enumerate() {
        if count > 4 {
            return Err(CompactEncodeError::CountOutOfRange {
                index: idx,
                value: count,
            });
        }
        writer.write_bits(u32::from(count), 3)?;
    }
    Ok(())
}

#[allow(
    dead_code,
    reason = "compact observation fact unpacking lands after initial label/mask cutover"
)]
pub(crate) fn unpack_tile_counts(
    src: &[u8; TILE34_COUNT_BYTES],
    dst: &mut [u8; 34],
) -> Result<(), CompactDecodeError> {
    let mut reader = BitReader::new(src);
    for (idx, slot) in dst.iter_mut().enumerate() {
        let count = reader.read_bits(3)? as u8;
        if count > 4 {
            return Err(CompactDecodeError::CountOutOfRange {
                index: idx,
                value: count,
            });
        }
        *slot = count;
    }
    Ok(())
}

#[allow(
    dead_code,
    reason = "compact observation fact decode will use this after dense placeholder removal"
)]
pub(crate) fn decode_counts_threshold_planes(
    counts: &[u8; 34],
    dst: &mut [f32],
    channel_start: usize,
) {
    for threshold in 0..4usize {
        let row = (channel_start + threshold) * 34;
        for (tile, &count) in counts.iter().enumerate() {
            dst[row + tile] = if count as usize > threshold { 1.0 } else { 0.0 };
        }
    }
}

#[cfg(test)]
mod tests;
