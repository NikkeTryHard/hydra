//! Suit permutation augmentation (6x) for observation tensors and actions.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES, OBS_SIZE};
use hydra_core::tile::{permute_tile_extended, permute_tile_type, ALL_PERMUTATIONS};
use std::sync::OnceLock;

const AKA_CHANNEL_START: usize = 40;
const AKA_CHANNELS: usize = 3;

#[derive(Clone, Copy)]
pub(crate) struct SuitPermutationTables {
    pub(crate) tile_34: [[usize; NUM_TILES]; 6],
    pub(crate) action_37: [[usize; 37]; 6],
}

pub(crate) fn permutation_tables() -> &'static SuitPermutationTables {
    static TABLES: OnceLock<SuitPermutationTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut tile_34 = [[0usize; NUM_TILES]; 6];
        let mut action_37 = [[0usize; 37]; 6];
        for (perm_index, perm) in ALL_PERMUTATIONS.iter().enumerate() {
            for tile in 0..NUM_TILES {
                tile_34[perm_index][tile] = permute_tile_type(tile as u8, perm) as usize;
            }
            for action in 0..37u8 {
                action_37[perm_index][action as usize] =
                    permute_tile_extended(action, perm) as usize;
            }
        }
        SuitPermutationTables { tile_34, action_37 }
    })
}

pub(crate) fn permutation_index(perm: &[u8; 3]) -> usize {
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
        for src_suit in 0..3usize {
            let dst_suit = perm[src_suit] as usize;
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

    for ch in 0..NUM_CHANNELS {
        let src_off = ch * NUM_TILES * 4;
        let dst_ch = ch * NUM_TILES;

        if (AKA_CHANNEL_START..AKA_CHANNEL_START + AKA_CHANNELS).contains(&ch) {
            let suit = ch - AKA_CHANNEL_START;
            let new_ch = AKA_CHANNEL_START + perm[suit] as usize;
            let d = &mut dst[new_ch * NUM_TILES..(new_ch + 1) * NUM_TILES];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_bytes[src_off..src_off + NUM_TILES * 4].as_ptr(),
                    d.as_mut_ptr().cast::<u8>(),
                    NUM_TILES * 4,
                );
            }
            continue;
        }

        // Block-copy each suit (9 tiles) to its permuted destination,
        // then copy honors (7 tiles) in place.  4 memcpy ops vs 34 scatter writes.
        for src_suit in 0..3usize {
            let dst_suit = perm[src_suit] as usize;
            let s_byte = src_off + src_suit * SUIT_TILES * 4;
            let d_idx = dst_ch + dst_suit * SUIT_TILES;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_bytes[s_byte..s_byte + SUIT_TILES * 4].as_ptr(),
                    dst[d_idx..d_idx + SUIT_TILES].as_mut_ptr().cast::<u8>(),
                    SUIT_TILES * 4,
                );
            }
        }
        let s_byte = src_off + HONOR_START * 4;
        let d_idx = dst_ch + HONOR_START;
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_bytes[s_byte..s_byte + HONOR_COUNT * 4].as_ptr(),
                dst[d_idx..d_idx + HONOR_COUNT].as_mut_ptr().cast::<u8>(),
                HONOR_COUNT * 4,
            );
        }
    }
}

#[cfg(not(target_endian = "little"))]
pub fn augment_obs_suit_from_le_bytes(src_bytes: &[u8], perm: &[u8; 3], dst: &mut [f32]) {
    debug_assert_eq!(src_bytes.len(), OBS_SIZE * 4);
    debug_assert_eq!(dst.len(), OBS_SIZE);
    let obs = crate::data::bc_shards::read_f32_array::<OBS_SIZE>(src_bytes);
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

pub fn augment_action_vector_u8_suit(
    values: &[u8; HYDRA_ACTION_SPACE],
    perm: &[u8; 3],
) -> [u8; HYDRA_ACTION_SPACE] {
    let mut out = [0u8; HYDRA_ACTION_SPACE];
    let action_perm = &permutation_tables().action_37[permutation_index(perm)];
    for i in 0..37u8 {
        let new_i = action_perm[i as usize];
        out[new_i] = values[i as usize];
    }
    out[37..HYDRA_ACTION_SPACE].copy_from_slice(&values[37..HYDRA_ACTION_SPACE]);
    out
}

pub fn augment_mask_u8_suit(
    mask: &[u8; HYDRA_ACTION_SPACE],
    perm: &[u8; 3],
) -> [u8; HYDRA_ACTION_SPACE] {
    augment_action_vector_u8_suit(mask, perm)
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
mod tests {
    use super::*;
    #[test]
    fn augment_6x_distinct() {
        let mut obs = [0.0f32; OBS_SIZE];
        for (i, v) in obs.iter_mut().enumerate() {
            *v = (i % 256) as f32 / 255.0;
        }
        let results: Vec<_> = ALL_PERMUTATIONS
            .iter()
            .map(|p| augment_obs_suit(&obs, p))
            .collect();
        for i in 0..6 {
            for j in (i + 1)..6 {
                assert_ne!(results[i], results[j], "perms {i} and {j} identical");
            }
        }
    }

    #[test]
    fn augment_preserves_honors() {
        let mut obs = [0.0f32; OBS_SIZE];
        for ch in 0..NUM_CHANNELS {
            for tile in 27..NUM_TILES {
                obs[ch * NUM_TILES + tile] = 1.0;
            }
        }
        for perm in &ALL_PERMUTATIONS {
            let out = augment_obs_suit(&obs, perm);
            for ch in 0..NUM_CHANNELS {
                for tile in 27..NUM_TILES {
                    assert_eq!(
                        out[ch * NUM_TILES + tile],
                        1.0,
                        "honor tile {tile} ch {ch} changed by perm {perm:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn augment_action_preserves_non_discard() {
        for perm in &ALL_PERMUTATIONS {
            for a in 37..46u8 {
                assert_eq!(augment_action_suit(a, perm), a);
            }
        }
    }

    #[test]
    fn augment_identity_is_noop() {
        let identity = &ALL_PERMUTATIONS[0];
        let mut obs = [0.0f32; OBS_SIZE];
        for (i, v) in obs.iter_mut().enumerate() {
            *v = i as f32;
        }
        let out = augment_obs_suit(&obs, identity);
        assert_eq!(obs, out);
    }

    #[test]
    fn augment_obs_moves_aka_planes_between_suits() {
        let swap_mp = &ALL_PERMUTATIONS[2];
        let mut obs = [0.0f32; OBS_SIZE];
        obs[AKA_CHANNEL_START * NUM_TILES] = 1.0;
        obs[(AKA_CHANNEL_START + 2) * NUM_TILES + 33] = 1.0;

        let out = augment_obs_suit(&obs, swap_mp);

        assert_eq!(out[AKA_CHANNEL_START * NUM_TILES], 0.0);
        assert_eq!(out[(AKA_CHANNEL_START + 1) * NUM_TILES], 1.0);
        assert_eq!(out[(AKA_CHANNEL_START + 2) * NUM_TILES + 33], 1.0);
    }

    #[test]
    fn augment_mask_preserves_non_discard_entries() {
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        mask[37] = 1.0;
        mask[43] = 1.0;
        mask[45] = 1.0;
        for perm in &ALL_PERMUTATIONS {
            let out = augment_mask_suit(&mask, perm);
            assert_eq!(out[37], 1.0, "riichi unchanged");
            assert_eq!(out[43], 1.0, "agari unchanged");
            assert_eq!(out[45], 1.0, "pass unchanged");
        }
    }

    #[test]
    fn augment_action_roundtrip_for_swaps() {
        let swap_mp = &ALL_PERMUTATIONS[2];
        for a in 0..37u8 {
            let permuted = augment_action_suit(a, swap_mp);
            let back = augment_action_suit(permuted, swap_mp);
            assert_eq!(a, back, "double-swap should be identity for action {a}");
        }
    }

    #[test]
    fn augment_action_vector_preserves_non_discard_entries() {
        let mut values = [0.0f32; HYDRA_ACTION_SPACE];
        values[37] = 0.25;
        values[43] = 0.5;
        values[45] = 0.75;
        for perm in &ALL_PERMUTATIONS {
            let out = augment_action_vector_suit(&values, perm);
            assert_eq!(out[37], 0.25);
            assert_eq!(out[43], 0.5);
            assert_eq!(out[45], 0.75);
        }
    }

    #[test]
    fn augment_belief_fields_permutes_tile_axis_only() {
        let swap_mp = &ALL_PERMUTATIONS[2];
        let mut values = [0.0f32; 16 * 34];
        values[0] = 1.0;
        values[34 + 9] = 2.0;
        let out = augment_belief_fields_suit(&values, swap_mp);
        assert_eq!(out[9], 1.0);
        assert_eq!(out[34], 2.0);
    }
}
