use hydra_core::encoder::OBS_SIZE;

/// FNV-1a hash on a downsampled subset of observation values.
///
/// Samples every 8th float for speed while maintaining enough
/// entropy for distinct observations at self-play scale.
pub fn obs_hash(obs: &[f32; OBS_SIZE]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for chunk in obs.chunks(8) {
        let bits = chunk[0].to_bits() as u64;
        hash ^= bits;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
