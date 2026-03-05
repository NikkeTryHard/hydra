//! Suit permutation augmentation (6x) for observation tensors and actions.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES, OBS_SIZE};
use hydra_core::tile::{permute_tile_extended, permute_tile_type};

pub fn augment_obs_suit(obs: &[f32; OBS_SIZE], perm: &[u8; 3]) -> [f32; OBS_SIZE] {
    let mut out = [0.0f32; OBS_SIZE];
    for ch in 0..NUM_CHANNELS {
        for tile in 0..NUM_TILES {
            let new_tile = permute_tile_type(tile as u8, perm) as usize;
            out[ch * NUM_TILES + new_tile] = obs[ch * NUM_TILES + tile];
        }
    }
    out
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
    for i in 0..37u8 {
        let new_i = permute_tile_extended(i, perm) as usize;
        out[new_i] = mask[i as usize];
    }
    out[37..HYDRA_ACTION_SPACE].copy_from_slice(&mask[37..HYDRA_ACTION_SPACE]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_core::tile::ALL_PERMUTATIONS;

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
}
