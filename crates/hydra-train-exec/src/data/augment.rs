//! Suit permutation augmentation (6x) for observation tensors and actions.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES, OBS_SIZE};
use hydra_core::tile::{ALL_PERMUTATIONS, permute_tile_extended, permute_tile_type};
use std::sync::OnceLock;

const AKA_CHANNEL_START: usize = 40;
const AKA_CHANNELS: usize = 3;

#[derive(Clone, Copy)]
pub struct SuitPermutationTables {
    pub tile_34: [[usize; NUM_TILES]; 6],
    pub action_37: [[usize; 37]; 6],
}

pub fn permutation_tables() -> &'static SuitPermutationTables {
    static TABLES: OnceLock<SuitPermutationTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut tile_34 = [[0usize; NUM_TILES]; 6];
        let mut action_37 = [[0usize; 37]; 6];
        for (perm_index, perm) in ALL_PERMUTATIONS.iter().enumerate() {
            for (tile, dst) in tile_34[perm_index].iter_mut().enumerate().take(NUM_TILES) {
                *dst = permute_tile_type(tile as u8, perm) as usize;
            }
            for action in 0..37u8 {
                action_37[perm_index][action as usize] =
                    permute_tile_extended(action, perm) as usize;
            }
        }
        SuitPermutationTables { tile_34, action_37 }
    })
}

pub fn permutation_index(perm: &[u8; 3]) -> usize {
    ALL_PERMUTATIONS
        .iter()
        .position(|candidate| candidate == perm)
        .expect("perm must be one of the six suit permutations")
}

pub fn augment_obs_suit(obs: &[f32; OBS_SIZE], perm: &[u8; 3]) -> [f32; OBS_SIZE] {
    let mut out = [0.0f32; OBS_SIZE];
    augment_obs_suit_into(obs, perm, &mut out);
    out
}

/// Permute obs in-place into `dst`, avoiding a 26KB return-value copy.
pub fn augment_obs_suit_into(obs: &[f32; OBS_SIZE], perm: &[u8; 3], dst: &mut [f32]) {
    debug_assert_eq!(dst.len(), OBS_SIZE);

    const SUIT_TILES: usize = 9;
    const HONOR_START: usize = 27;
    const HONOR_COUNT: usize = NUM_TILES - HONOR_START;

    for ch in 0..NUM_CHANNELS {
        let src_ch = ch * NUM_TILES;
        let dst_ch = ch * NUM_TILES;

        if (AKA_CHANNEL_START..AKA_CHANNEL_START + AKA_CHANNELS).contains(&ch) {
            let suit = ch - AKA_CHANNEL_START;
            let new_ch = AKA_CHANNEL_START + perm[suit] as usize;
            let d = &mut dst[new_ch * NUM_TILES..(new_ch + 1) * NUM_TILES];
            d.copy_from_slice(&obs[src_ch..src_ch + NUM_TILES]);
            continue;
        }

        // Block-copy each suit (9 tiles) to its permuted destination,
        // then copy honors (7 tiles) in place.  4 memcpy ops vs 34 scatter writes.
        for (src_suit, dst_suit) in perm.iter().copied().enumerate().take(3usize) {
            let dst_suit = dst_suit as usize;
            dst[dst_ch + dst_suit * SUIT_TILES..dst_ch + (dst_suit + 1) * SUIT_TILES]
                .copy_from_slice(
                    &obs[src_ch + src_suit * SUIT_TILES..src_ch + (src_suit + 1) * SUIT_TILES],
                );
        }
        dst[dst_ch + HONOR_START..dst_ch + HONOR_START + HONOR_COUNT]
            .copy_from_slice(&obs[src_ch + HONOR_START..src_ch + HONOR_START + HONOR_COUNT]);
    }
}

/// Read little-endian f32 obs from raw `&[u8]` mmap bytes, apply suit
/// permutation, and write directly into `dst`.  Eliminates both the 26KB
/// `[f32; OBS_SIZE]` parse intermediate AND the 26KB augment return value.
#[cfg(target_endian = "little")]
pub fn augment_obs_suit_from_le_bytes(src_bytes: &[u8], perm: &[u8; 3], dst: &mut [f32]) {
    debug_assert_eq!(src_bytes.len(), OBS_SIZE * 4);
    debug_assert_eq!(dst.len(), OBS_SIZE);

    const SUIT_TILES: usize = 9;
    const HONOR_START: usize = 27;
    const HONOR_COUNT: usize = NUM_TILES - HONOR_START;

    // SAFETY: All channel/suit indices are bounded by compile-time constants.
    // ch < NUM_CHANNELS=192, src_suit < 3, dst_suit = perm[src_suit] < 3.
    // Max src byte offset: (191 * 34 + 33) * 4 = 26108 < 26112 = OBS_SIZE * 4.
    // Max dst index: 191 * 34 + 33 = 6527 < 6528 = OBS_SIZE.
    let permute_normal_channel = |ch: usize, dst: &mut [f32], src_bytes: &[u8]| {
        let src_off = ch * NUM_TILES * 4;
        let dst_ch = ch * NUM_TILES;
        for (src_suit, dst_suit) in perm.iter().copied().enumerate().take(3usize) {
            let dst_suit = dst_suit as usize;
            let s_byte = src_off + src_suit * SUIT_TILES * 4;
            let d_idx = dst_ch + dst_suit * SUIT_TILES;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_bytes
                        .get_unchecked(s_byte..s_byte + SUIT_TILES * 4)
                        .as_ptr(),
                    dst.get_unchecked_mut(d_idx..d_idx + SUIT_TILES)
                        .as_mut_ptr()
                        .cast::<u8>(),
                    SUIT_TILES * 4,
                );
            }
        }
        let s_byte = src_off + HONOR_START * 4;
        let d_idx = dst_ch + HONOR_START;
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_bytes
                    .get_unchecked(s_byte..s_byte + HONOR_COUNT * 4)
                    .as_ptr(),
                dst.get_unchecked_mut(d_idx..d_idx + HONOR_COUNT)
                    .as_mut_ptr()
                    .cast::<u8>(),
                HONOR_COUNT * 4,
            );
        }
    };

    for ch in 0..AKA_CHANNEL_START {
        permute_normal_channel(ch, dst, src_bytes);
    }

    for ch in AKA_CHANNEL_START..AKA_CHANNEL_START + AKA_CHANNELS {
        let src_off = ch * NUM_TILES * 4;
        let suit = ch - AKA_CHANNEL_START;
        let new_ch = AKA_CHANNEL_START + perm[suit] as usize;
        let d_start = new_ch * NUM_TILES;
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_bytes
                    .get_unchecked(src_off..src_off + NUM_TILES * 4)
                    .as_ptr(),
                dst.get_unchecked_mut(d_start..d_start + NUM_TILES)
                    .as_mut_ptr()
                    .cast::<u8>(),
                NUM_TILES * 4,
            );
        }
    }

    for ch in AKA_CHANNEL_START + AKA_CHANNELS..NUM_CHANNELS {
        permute_normal_channel(ch, dst, src_bytes);
    }
}

#[cfg(not(target_endian = "little"))]
pub fn augment_obs_suit_from_le_bytes(src_bytes: &[u8], perm: &[u8; 3], dst: &mut [f32]) {
    debug_assert_eq!(src_bytes.len(), OBS_SIZE * 4);
    debug_assert_eq!(dst.len(), OBS_SIZE);
    let mut obs = [0.0f32; OBS_SIZE];
    for (dst, bytes) in obs.iter_mut().zip(src_bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes(bytes.try_into().expect("f32 bytes chunk has length 4"));
    }
    augment_obs_suit_into(&obs, perm, dst);
}

pub fn augment_action_suit(action: u8, perm: &[u8; 3]) -> u8 {
    if action <= 36 {
        permute_tile_extended(action, perm)
    } else {
        action
    }
}

pub fn augment_mask_suit(
    mask: &[f32; HYDRA_ACTION_SPACE],
    perm: &[u8; 3],
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut out = [0.0f32; HYDRA_ACTION_SPACE];
    let action_perm = &permutation_tables().action_37[permutation_index(perm)];
    for i in 0..37u8 {
        let new_i = action_perm[i as usize];
        out[new_i] = mask[i as usize];
    }
    out[37..HYDRA_ACTION_SPACE].copy_from_slice(&mask[37..HYDRA_ACTION_SPACE]);
    out
}

pub fn augment_action_vector_suit(
    values: &[f32; HYDRA_ACTION_SPACE],
    perm: &[u8; 3],
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut out = [0.0f32; HYDRA_ACTION_SPACE];
    let action_perm = &permutation_tables().action_37[permutation_index(perm)];
    for i in 0..37u8 {
        let new_i = action_perm[i as usize];
        out[new_i] = values[i as usize];
    }
    out[37..HYDRA_ACTION_SPACE].copy_from_slice(&values[37..HYDRA_ACTION_SPACE]);
    out
}

/// Permute f32 action values directly from `src` into `dst`, skipping
/// the stack intermediate + copy_from_slice pattern.
#[inline]
pub fn augment_action_vector_suit_into(
    values: &[f32; HYDRA_ACTION_SPACE],
    action_perm: &[usize; 37],
    dst: &mut [f32],
) {
    debug_assert_eq!(dst.len(), HYDRA_ACTION_SPACE);
    for i in 0..37usize {
        unsafe {
            let perm_idx = *action_perm.get_unchecked(i);
            *dst.get_unchecked_mut(perm_idx) = *values.get_unchecked(i);
        }
    }
    dst[37..HYDRA_ACTION_SPACE].copy_from_slice(&values[37..HYDRA_ACTION_SPACE]);
}

pub fn augment_action_vector_f32_mask_suit(
    values: &[f32; HYDRA_ACTION_SPACE],
    perm: &[u8; 3],
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut out = [0.0f32; HYDRA_ACTION_SPACE];
    let action_perm = &permutation_tables().action_37[permutation_index(perm)];
    for i in 0..37u8 {
        let new_i = action_perm[i as usize];
        out[new_i] = values[i as usize];
    }
    out[37..HYDRA_ACTION_SPACE].copy_from_slice(&values[37..HYDRA_ACTION_SPACE]);
    out
}

/// Permute f32 mask values directly from `src` into `dst`.
#[inline]
pub fn augment_action_vector_f32_mask_suit_into(
    values: &[f32; HYDRA_ACTION_SPACE],
    action_perm: &[usize; 37],
    dst: &mut [f32],
) {
    debug_assert_eq!(dst.len(), HYDRA_ACTION_SPACE);
    for i in 0..37usize {
        unsafe {
            let perm_idx = *action_perm.get_unchecked(i);
            *dst.get_unchecked_mut(perm_idx) = *values.get_unchecked(i);
        }
    }
    dst[37..HYDRA_ACTION_SPACE].copy_from_slice(&values[37..HYDRA_ACTION_SPACE]);
}

pub fn augment_mask_u8_suit(
    mask: &[u8; HYDRA_ACTION_SPACE],
    perm: &[u8; 3],
) -> [u8; HYDRA_ACTION_SPACE] {
    let mut out = [0u8; HYDRA_ACTION_SPACE];
    let action_perm = &permutation_tables().action_37[permutation_index(perm)];
    for i in 0..37u8 {
        let new_i = action_perm[i as usize];
        out[new_i] = mask[i as usize];
    }
    out[37..HYDRA_ACTION_SPACE].copy_from_slice(&mask[37..HYDRA_ACTION_SPACE]);
    out
}

pub fn augment_belief_fields_suit(values: &[f32; 16 * 34], perm: &[u8; 3]) -> [f32; 16 * 34] {
    let mut out = [0.0f32; 16 * 34];
    let tile_perm = &permutation_tables().tile_34[permutation_index(perm)];
    for channel in 0..16usize {
        for tile in 0..34usize {
            let new_tile = tile_perm[tile];
            out[channel * 34 + new_tile] = values[channel * 34 + tile];
        }
    }
    out
}

#[cfg(test)]
mod tests;
