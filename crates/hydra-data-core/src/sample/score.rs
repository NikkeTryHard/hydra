/// Minimum score delta represented by binned score targets.
const SCORE_BIN_MIN: f32 = -50000.0;
/// Maximum score delta represented by binned score targets.
const SCORE_BIN_MAX: f32 = 60000.0;
/// Number of score-delta bins.
pub const SCORE_BINS: usize = 64;

/// Converts a score delta into a clamped score-bin index.
#[inline]
pub fn score_delta_to_bin(score_delta: i32) -> usize {
    const RANGE_INV: f32 = 1.0 / (SCORE_BIN_MAX - SCORE_BIN_MIN);
    let normalized = (score_delta as f32 - SCORE_BIN_MIN) * RANGE_INV;
    let bin = (normalized * SCORE_BINS as f32) as usize;
    bin.min(SCORE_BINS - 1)
}

/// Converts a score delta into the normalized scalar value target.
#[inline]
pub fn score_delta_to_value(score_delta: i32) -> f32 {
    const INV_100K: f32 = 1.0 / 100_000.0;
    (score_delta as f32 * INV_100K).clamp(-1.0, 1.0)
}

/// Converts a score delta into a one-hot score-bin PDF target.
pub fn score_delta_to_pdf(score_delta: i32) -> [f32; SCORE_BINS] {
    let mut pdf = [0.0f32; SCORE_BINS];
    pdf[score_delta_to_bin(score_delta)] = 1.0;
    pdf
}

/// Converts a score delta into a cumulative score-bin CDF target.
pub fn score_delta_to_cdf(score_delta: i32) -> [f32; SCORE_BINS] {
    let bin = score_delta_to_bin(score_delta);
    let mut cdf = [0.0f32; SCORE_BINS];
    for v in &mut cdf[bin..] {
        *v = 1.0;
    }
    cdf
}
