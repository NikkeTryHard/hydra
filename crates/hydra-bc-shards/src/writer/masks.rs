//! Packed mask write helpers.

use std::io::{self, Write};

use hydra_core::action::HYDRA_ACTION_SPACE;

use crate::compact::{
    pack_action_mask, pack_binary_f32_mask, pack_spatial_mask, validate_u8_range,
};
use crate::manifest::{
    OPPONENT_COUNT, PACKED_ACTION_MASK_BYTES, PACKED_SPATIAL_MASK_BYTES, SPATIAL_TARGET_SIZE,
    TILE_COUNT,
};

use super::invalid_data;
use super::primitives::write_zero_bytes;

pub(super) fn write_packed_action_mask<W: Write>(
    writer: &mut W,
    values: &[f32; HYDRA_ACTION_SPACE],
) -> io::Result<()> {
    let mut packed = [0u8; PACKED_ACTION_MASK_BYTES];
    pack_action_mask(values, &mut packed).map_err(invalid_data)?;
    writer.write_all(&packed)
}

pub(super) fn write_packed_triplet<W: Write>(
    writer: &mut W,
    values: &[f32; OPPONENT_COUNT],
) -> io::Result<()> {
    let mut packed = [0u8; 1];
    pack_binary_f32_mask(values, OPPONENT_COUNT, &mut packed).map_err(invalid_data)?;
    writer.write_all(&packed)
}

pub(super) fn write_opp_next<W: Write>(
    writer: &mut W,
    values: &[u8; OPPONENT_COUNT],
) -> io::Result<()> {
    for &value in values {
        if value != 255 {
            validate_u8_range("opp_next", value, TILE_COUNT as u8).map_err(invalid_data)?;
        }
    }
    writer.write_all(values)
}

pub(super) fn write_packed_spatial_mask<W: Write>(
    writer: &mut W,
    values: &[f32; SPATIAL_TARGET_SIZE],
) -> io::Result<()> {
    let mut packed = [0u8; PACKED_SPATIAL_MASK_BYTES];
    pack_spatial_mask(values, &mut packed).map_err(invalid_data)?;
    writer.write_all(&packed)
}

pub(super) fn write_optional_action_mask_packed<W: Write>(
    writer: &mut W,
    values: Option<&[f32; HYDRA_ACTION_SPACE]>,
) -> io::Result<()> {
    if let Some(values) = values {
        write_packed_action_mask(writer, values)
    } else {
        write_zero_bytes(writer, PACKED_ACTION_MASK_BYTES)
    }
}
