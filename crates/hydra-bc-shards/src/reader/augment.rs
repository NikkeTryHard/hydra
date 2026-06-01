//! Suit/action augmentation helpers.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;

use crate::manifest::{OPPONENT_COUNT, TILE_COUNT};

use super::header::read_f32_array;

/// IEEE 754 bit pattern for `1.0f32`.
const F32_ONE_BITS: u32 = 0x3F80_0000;

#[cfg(not(target_endian = "little"))]
fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes[0..4].try_into().expect("f32 slice"))
}

pub(super) fn augment_action_f32_from_bytes_into(
    bytes: &[u8],
    action_perm: &[usize; 37],
    dst: &mut [f32],
) -> bool {
    debug_assert_eq!(dst.len(), HYDRA_ACTION_SPACE);
    let mut any = false;
    for src in 0..37 {
        let off = src * 4;
        #[cfg(target_endian = "little")]
        let value = {
            let mut bits = [0u8; 4];
            bits.copy_from_slice(&bytes[off..off + 4]);
            f32::from_ne_bytes(bits)
        };
        #[cfg(not(target_endian = "little"))]
        let value = read_f32_le(&bytes[off..off + 4]);
        any |= value != 0.0;
        dst[action_perm[src]] = value;
    }
    for (action, out) in dst.iter_mut().enumerate().take(HYDRA_ACTION_SPACE).skip(37) {
        let off = action * 4;
        #[cfg(target_endian = "little")]
        let value = {
            let mut bits = [0u8; 4];
            bits.copy_from_slice(&bytes[off..off + 4]);
            f32::from_ne_bytes(bits)
        };
        #[cfg(not(target_endian = "little"))]
        let value = read_f32_le(&bytes[off..off + 4]);
        any |= value != 0.0;
        *out = value;
    }
    any
}

#[inline]
pub(super) fn read_optional_action_f32_into(bytes: &[u8], dst: &mut [f32]) -> bool {
    let mut any = false;
    let values = read_f32_array::<{ HYDRA_ACTION_SPACE }>(bytes);
    for (out, value) in dst.iter_mut().zip(values) {
        any |= value != 0.0;
        *out = value;
    }
    any
}

#[inline]
pub(super) fn expand_and_augment_mask_into(
    bytes: &[u8],
    action_perm: &[usize; 37],
    dst: &mut [f32],
) -> bool {
    let mut any = false;
    for src in 0..37 {
        let nonzero = bytes[src] != 0;
        any |= nonzero;
        dst[action_perm[src]] = f32::from_bits(u32::from(nonzero) * F32_ONE_BITS);
    }
    for action in 37..HYDRA_ACTION_SPACE {
        let nonzero = bytes[action] != 0;
        any |= nonzero;
        dst[action] = f32::from_bits(u32::from(nonzero) * F32_ONE_BITS);
    }
    any
}

pub(super) fn expand_spatial_mask_f32(
    values: &[f32],
    dst: &mut [f32],
    suit_perm: Option<[usize; 3]>,
) {
    if let Some(perm) = suit_perm {
        for opponent in 0..OPPONENT_COUNT {
            for tile in 0..TILE_COUNT {
                let dst_tile = permute_tile(tile, perm);
                let src_idx = opponent * TILE_COUNT + tile;
                let dst_idx = opponent * TILE_COUNT + dst_tile;
                dst[dst_idx] = values[src_idx];
            }
        }
    } else {
        dst.copy_from_slice(values);
    }
}

pub(super) fn augment_obs_suit(values: &[f32; OBS_SIZE], suit_perm: [usize; 3], dst: &mut [f32]) {
    dst.fill(0.0);
    for channel in 0..192 {
        let src_base = channel * TILE_COUNT;
        let dst_base = src_base;
        for tile in 0..TILE_COUNT {
            let dst_tile = permute_tile(tile, suit_perm);
            dst[dst_base + dst_tile] = values[src_base + tile];
        }
    }
}

pub(super) fn suit_permutation(sample_index: usize) -> [usize; 3] {
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    PERMS[sample_index % PERMS.len()]
}

pub(super) fn permute_tile(tile: usize, suit_perm: [usize; 3]) -> usize {
    if tile < 27 {
        let suit = tile / 9;
        let rank = tile % 9;
        suit_perm[suit] * 9 + rank
    } else {
        tile
    }
}

pub(super) fn action_permutation(suit_perm: [usize; 3]) -> [usize; 37] {
    let mut perm = [0usize; 37];
    let mut action = 0usize;
    while action < 37 {
        perm[action] = permute_tile(action, suit_perm);
        action += 1;
    }
    perm
}
