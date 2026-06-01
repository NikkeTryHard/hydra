use std::io::{self, Read, Write};

use hydra_data_core::sample::MjaiSample;

use crate::limits::OPPONENT_COUNT;
use crate::primitives::{
    read_bool, read_f32_array, read_i32, read_optional_f32_array, read_u8, read_u16, write_bool,
    write_f32_array, write_i32, write_optional_f32_array, write_u8, write_u16,
};
use crate::{invalid_data, invalid_input};

const FLAG_ORACLE_TARGET: u16 = 1 << 0;
const FLAG_SAFETY_RESIDUAL: u16 = 1 << 1;
const FLAG_SAFETY_RESIDUAL_MASK: u16 = 1 << 2;
const FLAG_EXIT_TARGET: u16 = 1 << 3;
const FLAG_EXIT_MASK: u16 = 1 << 4;
const FLAG_DELTA_Q_TARGET: u16 = 1 << 5;
const FLAG_DELTA_Q_MASK: u16 = 1 << 6;
const FLAG_BELIEF_FIELDS: u16 = 1 << 7;
const FLAG_MIXTURE_WEIGHTS: u16 = 1 << 8;
const KNOWN_OPTIONAL_FLAGS: u16 = FLAG_ORACLE_TARGET
    | FLAG_SAFETY_RESIDUAL
    | FLAG_SAFETY_RESIDUAL_MASK
    | FLAG_EXIT_TARGET
    | FLAG_EXIT_MASK
    | FLAG_DELTA_Q_TARGET
    | FLAG_DELTA_Q_MASK
    | FLAG_BELIEF_FIELDS
    | FLAG_MIXTURE_WEIGHTS;

pub(crate) fn write_sample(writer: &mut impl Write, sample: &MjaiSample) -> io::Result<()> {
    let flags = sample_optional_flags(sample);
    validate_presence_invariants(
        sample.belief_fields_present,
        flags & FLAG_BELIEF_FIELDS != 0,
        "belief_fields",
    )
    .map_err(|err| invalid_input(err.to_string()))?;
    validate_presence_invariants(
        sample.mixture_weights_present,
        flags & FLAG_MIXTURE_WEIGHTS != 0,
        "mixture_weights",
    )
    .map_err(|err| invalid_input(err.to_string()))?;

    write_f32_array(writer, &sample.obs)?;
    write_u8(writer, sample.action)?;
    write_f32_array(writer, &sample.legal_mask)?;
    write_u8(writer, sample.placement)?;
    write_i32(writer, sample.score_delta)?;
    write_u8(writer, sample.grp_label)?;
    write_f32_array(writer, &sample.tenpai)?;
    writer.write_all(&sample.opp_next)?;
    write_f32_array(writer, &sample.danger)?;
    write_f32_array(writer, &sample.danger_mask)?;
    write_bool(writer, sample.belief_fields_present)?;
    write_bool(writer, sample.mixture_weights_present)?;
    write_u16(writer, flags)?;

    write_optional_f32_array(writer, sample.oracle_target.as_ref())?;
    write_optional_f32_array(writer, sample.safety_residual.as_ref())?;
    write_optional_f32_array(writer, sample.safety_residual_mask.as_ref())?;
    write_optional_f32_array(writer, sample.exit_target.as_ref())?;
    write_optional_f32_array(writer, sample.exit_mask.as_ref())?;
    write_optional_f32_array(writer, sample.delta_q_target.as_ref())?;
    write_optional_f32_array(writer, sample.delta_q_mask.as_ref())?;
    write_optional_f32_array(writer, sample.belief_fields.as_ref())?;
    write_optional_f32_array(writer, sample.mixture_weights.as_ref())?;
    Ok(())
}

fn sample_optional_flags(sample: &MjaiSample) -> u16 {
    let mut flags = 0u16;
    if sample.oracle_target.is_some() {
        flags |= FLAG_ORACLE_TARGET;
    }
    if sample.safety_residual.is_some() {
        flags |= FLAG_SAFETY_RESIDUAL;
    }
    if sample.safety_residual_mask.is_some() {
        flags |= FLAG_SAFETY_RESIDUAL_MASK;
    }
    if sample.exit_target.is_some() {
        flags |= FLAG_EXIT_TARGET;
    }
    if sample.exit_mask.is_some() {
        flags |= FLAG_EXIT_MASK;
    }
    if sample.delta_q_target.is_some() {
        flags |= FLAG_DELTA_Q_TARGET;
    }
    if sample.delta_q_mask.is_some() {
        flags |= FLAG_DELTA_Q_MASK;
    }
    if sample.belief_fields.is_some() {
        flags |= FLAG_BELIEF_FIELDS;
    }
    if sample.mixture_weights.is_some() {
        flags |= FLAG_MIXTURE_WEIGHTS;
    }
    flags
}

fn validate_presence_invariants(
    bool_present: bool,
    flag_present: bool,
    field_name: &str,
) -> io::Result<()> {
    if bool_present != flag_present {
        return Err(invalid_data(format!(
            "parsed-sample cache {field_name} presence flag mismatch"
        )));
    }
    Ok(())
}

pub(crate) fn read_sample(reader: &mut impl Read) -> io::Result<MjaiSample> {
    let obs = read_f32_array(reader)?;
    let action = read_u8(reader)?;
    let legal_mask = read_f32_array(reader)?;
    let placement = read_u8(reader)?;
    let score_delta = read_i32(reader)?;
    let grp_label = read_u8(reader)?;
    let tenpai = read_f32_array(reader)?;
    let mut opp_next = [0u8; OPPONENT_COUNT];
    reader.read_exact(&mut opp_next)?;
    let danger = read_f32_array(reader)?;
    let danger_mask = read_f32_array(reader)?;
    let belief_fields_present = read_bool(reader)?;
    let mixture_weights_present = read_bool(reader)?;
    let flags = read_u16(reader)?;
    if flags & !KNOWN_OPTIONAL_FLAGS != 0 {
        return Err(invalid_data(format!(
            "parsed-sample cache sample has unknown optional flag bits: 0x{:04x}",
            flags & !KNOWN_OPTIONAL_FLAGS
        )));
    }
    validate_presence_invariants(
        belief_fields_present,
        flags & FLAG_BELIEF_FIELDS != 0,
        "belief_fields",
    )?;
    validate_presence_invariants(
        mixture_weights_present,
        flags & FLAG_MIXTURE_WEIGHTS != 0,
        "mixture_weights",
    )?;

    Ok(MjaiSample {
        obs,
        compact_facts: None,
        action,
        legal_mask,
        placement,
        score_delta,
        grp_label,
        oracle_target: read_optional_f32_array(reader, flags & FLAG_ORACLE_TARGET != 0)?,
        tenpai,
        opp_next,
        danger,
        danger_mask,
        safety_residual: read_optional_f32_array(reader, flags & FLAG_SAFETY_RESIDUAL != 0)?,
        safety_residual_mask: read_optional_f32_array(
            reader,
            flags & FLAG_SAFETY_RESIDUAL_MASK != 0,
        )?,
        exit_target: read_optional_f32_array(reader, flags & FLAG_EXIT_TARGET != 0)?,
        exit_mask: read_optional_f32_array(reader, flags & FLAG_EXIT_MASK != 0)?,
        delta_q_target: read_optional_f32_array(reader, flags & FLAG_DELTA_Q_TARGET != 0)?,
        delta_q_mask: read_optional_f32_array(reader, flags & FLAG_DELTA_Q_MASK != 0)?,
        belief_fields: read_optional_f32_array(reader, flags & FLAG_BELIEF_FIELDS != 0)?,
        mixture_weights: read_optional_f32_array(reader, flags & FLAG_MIXTURE_WEIGHTS != 0)?,
        belief_fields_present,
        mixture_weights_present,
    })
}

#[cfg(test)]
pub(crate) mod test_support {
    pub(crate) const FLAG_BELIEF_FIELDS: u16 = super::FLAG_BELIEF_FIELDS;
    pub(crate) const FLAG_MIXTURE_WEIGHTS: u16 = super::FLAG_MIXTURE_WEIGHTS;
}
