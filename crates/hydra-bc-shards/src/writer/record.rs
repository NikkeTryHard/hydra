//! Compact sample record encoding.

use std::io::{self, Write};

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_data_core::sample::MjaiSample;

use crate::compact::validate_u8_range;
use crate::host::record_size_for_flags;
use crate::manifest::{
    FLAG_BELIEF_FIELDS, FLAG_DELTA_Q, FLAG_EXIT, FLAG_MIXTURE_WEIGHTS, FLAG_SAFETY_RESIDUAL,
    OPTIONAL_ACTION_FLOAT32_BYTES, ORACLE_FLOAT32_BYTES, PLAYER_COUNT, checked_compact_record_size,
    validate_feature_flags,
};

use super::invalid_data;
use super::masks::{
    write_opp_next, write_optional_action_mask_packed, write_packed_action_mask,
    write_packed_spatial_mask, write_packed_triplet,
};
use super::obs::write_compact_obs;
use super::primitives::{write_i32_le, write_required_f32_slice, write_u8, write_zero_bytes};

/// Writes one compact BC shard sample record.
pub fn write_sample_record<W: Write>(
    writer: &mut W,
    sample: &MjaiSample,
    flags: u32,
) -> io::Result<()> {
    validate_feature_flags(flags).map_err(invalid_data)?;
    if record_size_for_flags(flags) != checked_compact_record_size(flags).map_err(invalid_data)? {
        return Err(invalid_data("BC shard compact record-size helper mismatch"));
    }
    write_compact_obs(writer, sample)?;
    validate_u8_range("action", sample.action, HYDRA_ACTION_SPACE as u8).map_err(invalid_data)?;
    write_u8(writer, sample.action)?;
    write_packed_action_mask(writer, &sample.legal_mask)?;
    write_i32_le(writer, sample.score_delta)?;
    validate_u8_range(
        "grp_label",
        sample.grp_label,
        crate::host::GRP_CLASS_COUNT as u8,
    )
    .map_err(invalid_data)?;
    write_u8(writer, sample.grp_label)?;
    write_optional_oracle_f32(writer, sample.oracle_target.as_ref())?;
    write_u8(writer, u8::from(sample.oracle_target.is_some()))?;
    write_packed_triplet(writer, &sample.tenpai)?;
    write_opp_next(writer, &sample.opp_next)?;
    write_packed_spatial_mask(writer, &sample.danger)?;
    write_packed_spatial_mask(writer, &sample.danger_mask)?;

    if flags & FLAG_SAFETY_RESIDUAL != 0 {
        write_optional_action_f32(writer, sample.safety_residual.as_ref())?;
        write_optional_action_mask_packed(writer, sample.safety_residual_mask.as_ref())?;
    }
    if flags & FLAG_EXIT != 0 {
        write_optional_action_f32(writer, sample.exit_target.as_ref())?;
        write_optional_action_mask_packed(writer, sample.exit_mask.as_ref())?;
    }
    if flags & FLAG_DELTA_Q != 0 {
        write_optional_action_f32(writer, sample.delta_q_target.as_ref())?;
        write_optional_action_mask_packed(writer, sample.delta_q_mask.as_ref())?;
    }
    if flags & FLAG_BELIEF_FIELDS != 0 {
        write_required_f32_slice(
            writer,
            sample.belief_fields.as_ref().ok_or_else(|| {
                invalid_data("belief fields flag set but sample has no belief fields")
            })?,
        )?;
    }
    if flags & FLAG_MIXTURE_WEIGHTS != 0 {
        write_required_f32_slice(
            writer,
            sample.mixture_weights.as_ref().ok_or_else(|| {
                invalid_data("mixture weights flag set but sample has no mixture weights")
            })?,
        )?;
    }
    Ok(())
}

/// Encodes compact BC shard sample records into caller-owned bytes.
pub fn encode_sample_records(
    samples: &[MjaiSample],
    flags: u32,
    record_size: u32,
) -> io::Result<Vec<u8>> {
    validate_feature_flags(flags).map_err(invalid_data)?;
    let checked = checked_compact_record_size(flags).map_err(invalid_data)?;
    if record_size_for_flags(flags) != checked || record_size != checked {
        return Err(invalid_data("BC shard compact record-size helper mismatch"));
    }
    let record_size = record_size as usize;
    let record_len = crate::manifest::checked_encoded_record_len(samples.len(), checked)
        .map_err(invalid_data)?;
    let mut records = vec![0u8; record_len];
    for (sample, dst) in samples.iter().zip(records.chunks_exact_mut(record_size)) {
        write_sample_record(&mut &mut *dst, sample, flags)?;
    }
    Ok(records)
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
