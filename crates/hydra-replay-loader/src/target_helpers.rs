//! Pure replay target helper functions.

use hydra_core::encoder::OBS_SIZE;

pub fn oracle_target_from_scores(final_scores: [i32; 4]) -> [f32; 4] {
    let mean = final_scores.iter().sum::<i32>() as f32 / 4.0;
    let mut target = [0.0f32; 4];
    for (i, &s) in final_scores.iter().enumerate() {
        target[i] = (s as f32 - mean) / 100_000.0;
    }
    target
}

pub fn obs_hash(obs: &[f32; OBS_SIZE]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for chunk in obs.chunks(8) {
        let bits = chunk[0].to_bits() as u64;
        hash ^= bits;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
