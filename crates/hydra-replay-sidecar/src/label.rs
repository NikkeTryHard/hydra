//! Label array and legal-mask digest helpers for sidecar indexes.

use hydra_core::action::HYDRA_ACTION_SPACE;

/// Returns a stable digest over action legality/support bits.
pub fn legal_mask_digest_from_f32(mask: &[f32; HYDRA_ACTION_SPACE]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &value in mask {
        hash ^= u64::from(value > 0.0);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Returns a stable digest over action legality/support bits.
pub fn legal_mask_digest_from_bool(mask: &[bool; HYDRA_ACTION_SPACE]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &value in mask {
        hash ^= u64::from(value);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Copies vector-backed sidecar labels into fixed action-space arrays.
pub fn copy_label_arrays(
    target: &[f32],
    mask: &[f32],
) -> Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])> {
    if target.len() != HYDRA_ACTION_SPACE || mask.len() != HYDRA_ACTION_SPACE {
        return None;
    }
    let mut target_arr = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask_arr = [0.0f32; HYDRA_ACTION_SPACE];
    target_arr.copy_from_slice(target);
    mask_arr.copy_from_slice(mask);
    Some((target_arr, mask_arr))
}
