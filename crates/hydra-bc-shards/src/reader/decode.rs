//! BC shard row and optional-target decoding.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;
use hydra_data_core::sample::{score_delta_to_bin, score_delta_to_value};

use crate::compact::{unpack_action_mask_into, unpack_binary_mask_into, unpack_spatial_mask_into};
use crate::host::{BcShardHostScratch, GRP_CLASS_COUNT};
use crate::manifest::{
    COMPACT_OBS_BASELINE_FACT_BYTES, COMPACT_OBS_DENSE_BYTES, COMPACT_OBS_SCALAR_REPEATED_BYTES,
    FLAG_BELIEF_FIELDS, FLAG_DELTA_Q, FLAG_EXIT, FLAG_MIXTURE_WEIGHTS, FLAG_SAFETY_RESIDUAL,
    OPPONENT_COUNT, PACKED_ACTION_MASK_BYTES, PACKED_SPATIAL_MASK_BYTES, PLAYER_COUNT,
    SPATIAL_TARGET_SIZE, TILE_COUNT,
};

use super::augment::{
    action_permutation, augment_action_f32_from_bytes_into, augment_obs_suit,
    expand_and_augment_mask_into, expand_spatial_mask_f32, read_optional_action_f32_into,
};
use super::header::{read_f32_array, read_i32_le, take, take_array};
use super::obs::decode_compact_obs;

/// Number of score bins materialized by the host reader.
const SCORE_BINS: usize = hydra_data_core::sample::SCORE_BINS;

pub(super) fn decode_row_bytes(
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
