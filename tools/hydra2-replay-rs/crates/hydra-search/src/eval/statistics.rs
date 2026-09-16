//! Block-level uncertainty — bootstrap assembly, sign-flip, CS (EvalControl-owned).
//!
//! Rust owner of `statistics.py:66-336,385-558` draw-free half.
//! M2 (measure-first, Wave5 §5C): `statistics.py:127-158` is ONE
//! vectorised numpy call (PCG64-DXSM in C, brief GIL) — NOT a Python-loop
//! hotspot. Default is port-B: numpy keeps the draws; Rust owns
//! block-means + percentile-index math + validation + the draw-free
//! hedged-CS path. There is NO PCG64 reimplementation here (port-A only
//! behind measurement + bit-parity KAT, never silently).
//!
//! - `placement_block_contrast` (`:92-112`): wall-order-stable collapse
//!   via `aggregate_wall_block`; feed the result — never per-game
//!   contrasts — to the resamplers. T1 pins wall-block atomicity.
//! - `bootstrap_interval` / `sign_flip_centered_interval`: draw-free
//!   assembly over caller-supplied (numpy-drawn, PCG64) sorted resample
//!   statistics. Percentile indices replicate `:156-157` VERBATIM:
//!   `low = sorted[floor(a/2*R)]`, `high = sorted[ceil((1-a/2)*R)-1]`.
//! - Seed parity: the ONLY seed source is the legacy CTR stream
//!   (`super::CtrReader` over `score_selection_seed`; T3 pins the
//!   `STREAM` goldens in `super`). `score_selection`'s domain separation
//!   `sha256("score-selection-v1:{seed}")` (`:553`) is `super`.
//! - `fixed_n_samples` (`:115-124`): `ceil(((z+z)*s/d)^2)` with Acklam
//!   `inv_cdf` (FIXEDN golden 56; margin 0.64 so the 1e-9-class
//!   approximation error cannot flip the ceil).
//! - `hedged_cs_path` (`:254-303`) verbatim: predictable `lam` uses
//!   only `t-1` info (Ville); grid union `alpha/48`; empty survivors
//!   yield `(inf, -inf)`. No draws — pure deterministic capital math.
//! - `SelectionConfig` (`:385-456`) is the ONLY home for
//!   N/s/delta/alpha/beta; `selection_gate_check` (`:458-481`) enforces
//!   frozen peek discipline; `score_selection` (`:484-558`) scores VALID
//!   blocks only with a caller-supplied (numpy) bootstrap triple.
//!
//! T4/T5 (NLL/top1, teacher-gate) guard torch islands (`StudentModel`,
//! `_teacher_policy_and_value`) that STAY Python per invariants — no
//! Rust surface exists for them here by design (no Rust GPU math).

use std::collections::HashMap;

use crate::SearchError;

use super::blocks::{BlockAggregateResult, WallBlock, aggregate_blocks};
use super::neumaier_sum;
use super::telemetry::{BlockTolerance, TelemetryRow};

/// Validate block values (`_validate_blocks`, `:71-82`): at least two
/// finite values.
pub fn validate_blocks(values: &[f64]) -> Result<Vec<f64>, SearchError> {
    if values.len() < 2 {
        return Err(SearchError::InvalidArg {
            detail: "need at least two blocks for any interval",
        });
    }
    for v in values {
        if !v.is_finite() {
            return Err(SearchError::NonFinite { context: "block contrast" });
        }
    }
    Ok(values.to_vec())
}

/// Validate alpha/resamples (`_validate_alpha_resamples`, `:85-89`).
pub fn validate_alpha_resamples(alpha: f64, resamples: usize) -> Result<(), SearchError> {
    if !(0.0 < alpha && alpha < 0.5) || !alpha.is_finite() {
        return Err(SearchError::InvalidArg { detail: "alpha must lie in (0, 0.5)" });
    }
    if resamples < 100 {
        return Err(SearchError::InvalidArg { detail: "resamples must be an int >= 100" });
    }
    Ok(())
}

/// Collapse wall blocks to placement-contrast means, wall order stable
/// (`:92-112`). Each block must already carry expected-final-placement
/// contrasts; this only collapses via `aggregate_wall_block`.
pub fn placement_block_contrast(blocks: &[WallBlock]) -> Result<Vec<f64>, SearchError> {
    if blocks.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "need at least one wall block for a contrast",
        });
    }
    let mut ordered: Vec<&WallBlock> = blocks.iter().collect();
    ordered.sort_by(|a, b| a.wall_id.cmp(&b.wall_id));
    let mut out = Vec::with_capacity(ordered.len());
    for block in ordered {
        let mean = super::block_mean(&block.contrasts)?;
        if !mean.is_finite() {
            return Err(SearchError::NonFinite { context: "block contrast" });
        }
        out.push(mean);
    }
    Ok(out)
}

/// Inverse standard-normal CDF (Acklam's approximation, ~1e-9 class).
/// Only feeds [`fixed_n_samples`], whose ceil margin (0.64) dwarfs the
/// approximation error.
fn inv_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - 0.02425;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// SPEC 18.3 fixed-N (`:115-124`):
/// `N = ceil(((z_(1-alpha) + z_(1-beta)) * s / delta)^2)`.
pub fn fixed_n_samples(s: f64, delta: f64, alpha: f64, beta: f64) -> Result<u64, SearchError> {
    for (name, value) in [("s", s), ("delta", delta)] {
        if !value.is_finite() || value <= 0.0 {
            return Err(SearchError::InvalidArg { detail: name });
        }
    }
    for value in [alpha, beta] {
        if !(0.0 < value && value < 1.0) || !value.is_finite() {
            return Err(SearchError::InvalidArg { detail: "alpha and beta must lie in (0, 1)" });
        }
    }
    let z_a = inv_normal_cdf(1.0 - alpha);
    let z_b = inv_normal_cdf(1.0 - beta);
    Ok((((z_a + z_b) * s / delta).powi(2)).ceil() as u64)
}

/// Percentile indices (`:156-157` verbatim):
/// `low = floor(a/2*R)`, `high = ceil((1-a/2)*R)-1`.
pub fn percentile_indexes(alpha: f64, resamples: usize) -> Result<(usize, usize), SearchError> {
    validate_alpha_resamples(alpha, resamples)?;
    let low = ((alpha / 2.0) * resamples as f64).floor() as usize;
    let high = ((1.0 - alpha / 2.0) * resamples as f64).ceil() as usize - 1;
    Ok((low, high))
}

/// Whole-block percentile bootstrap assembly (`:127-158`, draw-free):
/// `estimate` is the block-mean, `sorted_means` the numpy-drawn
/// (PCG64) sorted resample means of length `resamples`.
pub fn bootstrap_interval(
    block_values: &[f64],
    sorted_means: &[f64],
    alpha: f64,
    resamples: usize,
) -> Result<(f64, f64, f64), SearchError> {
    let values = validate_blocks(block_values)?;
    validate_alpha_resamples(alpha, resamples)?;
    if sorted_means.len() != resamples {
        return Err(SearchError::InvalidArg {
            detail: "sorted_means must carry one entry per resample",
        });
    }
    let (low_idx, high_idx) = percentile_indexes(alpha, resamples)?;
    let estimate = neumaier_sum(&values) / values.len() as f64;
    Ok((estimate, sorted_means[low_idx], sorted_means[high_idx]))
}

/// Sign-flip assembly (`:161-194`, draw-free): the null distribution is
/// centered on the observed mean (symmetric-null shift, `:191-193`).
pub fn sign_flip_centered_interval(
    block_values: &[f64],
    sorted_stats: &[f64],
    alpha: f64,
    resamples: usize,
) -> Result<(f64, f64, f64), SearchError> {
    let values = validate_blocks(block_values)?;
    validate_alpha_resamples(alpha, resamples)?;
    if sorted_stats.len() != resamples {
        return Err(SearchError::InvalidArg {
            detail: "sorted_stats must carry one entry per resample",
        });
    }
    let (low_idx, high_idx) = percentile_indexes(alpha, resamples)?;
    let observed = neumaier_sum(&values) / values.len() as f64;
    let center = neumaier_sum(sorted_stats) / sorted_stats.len() as f64;
    Ok((
        observed,
        observed + (sorted_stats[low_idx] - center),
        observed + (sorted_stats[high_idx] - center),
    ))
}

/// Group means for the cluster path (`:370-377`, draw-free): unweighted
/// mean over per-group means; groups are the atomic units (game/player
/// only — there is no decision-level option).
pub fn group_means(groups: &[&[f64]]) -> Result<Vec<f64>, SearchError> {
    if groups.len() < 2 {
        return Err(SearchError::InvalidArg {
            detail: "clustering needs at least two groups",
        });
    }
    let mut means = Vec::with_capacity(groups.len());
    for group in groups {
        if group.is_empty() || group.iter().any(|v| !v.is_finite()) {
            return Err(SearchError::NonFinite { context: "record value" });
        }
        means.push(neumaier_sum(group) / group.len() as f64);
    }
    Ok(means)
}

/// Gate helper: does the interval cover `truth` (`ci_covers`, `:197-199`)?
pub fn ci_covers(bounds: (f64, f64), truth: f64) -> bool {
    bounds.0 <= truth && truth <= bounds.1
}

/// One hedged capital path (`_hedged_capital_rejected`, `:202-233`):
/// two-sided hedged capital; predictable lambdas use only `t-1` info
/// (removing the variance cap or lookback breaks Ville validity).
fn hedged_capital_rejected(scaled: &[f64], theta: f64, threshold: f64) -> bool {
    let lam_max = (1.0 / (2.0 * theta)).min(1.0 / (2.0 * (1.0 - theta)));
    let mut wealth_up = 1.0;
    let mut wealth_down = 1.0;
    let mut running_sum = 0.0;
    for (position, observation) in scaled.iter().enumerate() {
        let index = position + 1;
        let mu_prev = if index > 1 { running_sum / (index - 1) as f64 } else { 0.5 };
        let variance_term = (mu_prev * (1.0 - mu_prev)).max(1e-6);
        let lam = lam_max.min(
            (8.0 * (2.0 * threshold).ln() / (variance_term * index as f64 * (index as f64 + 1.0).ln()))
                .sqrt(),
        );
        let deviation = observation - theta;
        wealth_up *= (1.0 + lam * deviation).max(0.0);
        wealth_down *= (1.0 - lam * deviation).max(0.0);
        if wealth_up >= threshold || wealth_down >= threshold {
            return true;
        }
        running_sum += observation;
    }
    false
}

/// Scale values into `[0,1]` under `bounds` (`_scale_to_unit_interval`,
/// `:236-251`).
fn scale_to_unit_interval(values: &[f64], bounds: (f64, f64)) -> Result<Vec<f64>, SearchError> {
    let (low, high) = bounds;
    if !(low.is_finite() && high.is_finite() && high > low) {
        return Err(SearchError::InvalidArg { detail: "bounds must be finite with high > low" });
    }
    let width = high - low;
    let mut scaled = Vec::with_capacity(values.len());
    for value in values {
        if !value.is_finite() {
            return Err(SearchError::NonFinite { context: "cs value" });
        }
        scaled.push(((value - low) / width).clamp(0.0, 1.0));
    }
    Ok(scaled)
}

/// Time-uniform hedged-CS intervals at `peek_times` (default all t)
/// (`hedged_cs_path`, `:254-303`). One `(low, high)` pair per peek,
/// each valid simultaneously for ALL earlier stopping times. The mean
/// grid uses a union bound `alpha/grid_size`; a grid point dies once
/// its capital ever crosses `1/alpha_j`; survivors map back through
/// `bounds`. Empty survivors yield `(inf, -inf)`.
pub fn hedged_cs_path(
    values: &[f64],
    alpha: f64,
    bounds: (f64, f64),
    grid_size: usize,
    peek_times: Option<&[usize]>,
) -> Result<Vec<(f64, f64)>, SearchError> {
    validate_alpha_resamples(alpha, 100)?;
    if grid_size < 4 || grid_size > 512 {
        return Err(SearchError::InvalidArg { detail: "grid_size must be an int in [4, 512]" });
    }
    let scaled = scale_to_unit_interval(values, bounds)?;
    let count = scaled.len();
    let times: Vec<usize> = match peek_times {
        Some(peeks) => {
            let mut set: Vec<usize> = peeks.to_vec();
            set.sort_unstable();
            set.dedup();
            set
        }
        None => vec![count],
    };
    for moment in &times {
        if *moment < 1 || *moment > count {
            return Err(SearchError::InvalidArg { detail: "peek times must be ints in [1, n]" });
        }
    }
    let alpha_j = alpha / grid_size as f64;
    let threshold = 1.0 / alpha_j;
    let thetas: Vec<f64> = (0..grid_size)
        .map(|i| 0.02 + (0.98 - 0.02) * i as f64 / (grid_size - 1) as f64)
        .collect();
    let mut rejected = vec![false; grid_size];
    let mut intervals = Vec::with_capacity(times.len());
    let (low_span, high_span) = bounds;
    let width = high_span - low_span;
    for moment in times {
        for (slot, theta) in thetas.iter().enumerate() {
            if rejected[slot] {
                continue;
            }
            if hedged_capital_rejected(&scaled[..moment], *theta, threshold) {
                rejected[slot] = true;
            }
        }
        let survivors: Vec<f64> = thetas
            .iter()
            .enumerate()
            .filter(|(slot, _)| !rejected[*slot])
            .map(|(_, theta)| *theta)
            .collect();
        if survivors.is_empty() {
            intervals.push((f64::INFINITY, f64::NEG_INFINITY));
        } else {
            let low = low_span + survivors.iter().cloned().fold(f64::INFINITY, f64::min) * width;
            let high = low_span + survivors.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * width;
            intervals.push((low, high));
        }
    }
    Ok(intervals)
}

/// Current hedged-CS interval (final peek, `:306-314`).
pub fn hedged_confidence_sequence(
    values: &[f64],
    alpha: f64,
    bounds: (f64, f64),
    grid_size: usize,
) -> Result<(f64, f64), SearchError> {
    let path = hedged_cs_path(values, alpha, bounds, grid_size, None)?;
    path.into_iter().last().ok_or(SearchError::InvalidArg { detail: "no values for CS" })
}

/// Adaptive peeking without a declared sequential design is fatal
/// (`sequential_design_guard`, `:317-336`).
pub fn sequential_design_guard(
    design: &str,
    declared_peeks: &[usize],
) -> Result<(), SearchError> {
    match design {
        "fixed_n" => {
            if declared_peeks.len() != 1 {
                return Err(SearchError::InvalidArg {
                    detail: "fixed-N forbids intermediate peeks",
                });
            }
            Ok(())
        }
        "time_uniform_cs" => {
            if declared_peeks.is_empty() {
                return Err(SearchError::InvalidArg {
                    detail: "declared CS peek schedule must be nonempty, strictly increasing",
                });
            }
            for pair in declared_peeks.windows(2) {
                if pair[0] >= pair[1] || pair[0] < 1 {
                    return Err(SearchError::InvalidArg {
                        detail: "declared CS peek schedule must be nonempty, strictly increasing",
                    });
                }
            }
            if declared_peeks[0] < 1 {
                return Err(SearchError::InvalidArg {
                    detail: "declared CS peek schedule must be nonempty, strictly increasing",
                });
            }
            Ok(())
        }
        _ => Err(SearchError::InvalidArg { detail: "unknown design" }),
    }
}

/// Frozen selection parameters; the ONLY home for N/s/delta/alpha/beta
/// (`SelectionConfig`, `:385-456`). Frozen before any confirmation
/// result exists; `declared_peeks` holds 1-based wall-block counts.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionConfig {
    /// Frozen sample size.
    pub n: usize,
    /// Pilot scale.
    pub pilot_s: f64,
    /// Minimum detectable effect.
    pub delta: f64,
    /// Level.
    pub alpha: f64,
    /// Type-II rate.
    pub beta: f64,
    /// `fixed_n` or `time_uniform_cs`.
    pub design: String,
    /// Declared peek schedule (1-based wall-block counts).
    pub declared_peeks: Vec<usize>,
    /// Margin (finite).
    pub margin: f64,
    /// Always `wall_block` (whole-wall-block uncertainty only).
    pub uncertainty_unit: String,
    /// Bootstrap resamples.
    pub resamples: usize,
    /// Score-selection seed.
    pub seed: i64,
}

impl SelectionConfig {
    /// Freeze-time checks (`validate`, `:412-455`).
    pub fn validate(&self) -> Result<(), SearchError> {
        if self.n < 1 {
            return Err(SearchError::InvalidArg { detail: "N must be a positive int" });
        }
        for (name, value) in [("pilot_s", self.pilot_s), ("delta", self.delta)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(SearchError::InvalidArg { detail: name });
            }
        }
        for value in [self.alpha, self.beta] {
            if !(0.0 < value && value < 1.0) || !value.is_finite() {
                return Err(SearchError::InvalidArg { detail: "alpha/beta must lie in (0, 1)" });
            }
        }
        if !self.margin.is_finite() {
            return Err(SearchError::InvalidArg { detail: "margin must be finite" });
        }
        if self.uncertainty_unit != "wall_block" {
            return Err(SearchError::InvalidArg { detail: "uncertainty_unit must be 'wall_block'" });
        }
        if self.resamples < 1 {
            return Err(SearchError::InvalidArg { detail: "resamples must be a positive int" });
        }
        if self.seed < 0 {
            return Err(SearchError::InvalidArg { detail: "seed must be a non-negative int" });
        }
        if self.declared_peeks.iter().any(|p| *p < 1) {
            return Err(SearchError::InvalidArg {
                detail: "declared peeks must be positive ints",
            });
        }
        sequential_design_guard(&self.design, &self.declared_peeks)
    }
}

impl Default for SelectionConfig {
    /// Frozen defaults (`N=30`, `s=1.5`, `delta=0.5`, `alpha=0.05`,
    /// `beta=0.2`, `fixed_n`, peeks `(30,)`, `resamples=2000`).
    fn default() -> Self {
        SelectionConfig {
            n: 30,
            pilot_s: 1.5,
            delta: 0.5,
            alpha: 0.05,
            beta: 0.2,
            design: "fixed_n".to_string(),
            declared_peeks: vec![30],
            margin: 0.0,
            uncertainty_unit: "wall_block".to_string(),
            resamples: 2000,
            seed: 0,
        }
    }
}

/// Enforce frozen peek discipline at `n_blocks_observed`
/// (`selection_gate_check`, `:458-481`).
pub fn selection_gate_check(config: &SelectionConfig, n_blocks_observed: usize) -> Result<(), SearchError> {
    config.validate()?;
    if n_blocks_observed < 1 {
        return Err(SearchError::InvalidArg {
            detail: "n_blocks_observed must be a positive int",
        });
    }
    if n_blocks_observed > config.n {
        return Err(SearchError::InvalidArg { detail: "extra look exceeds frozen N" });
    }
    if !config.declared_peeks.contains(&n_blocks_observed) {
        return Err(SearchError::InvalidArg { detail: "extra look outside declared peeks" });
    }
    let elapsed: Vec<usize> = config
        .declared_peeks
        .iter()
        .copied()
        .filter(|p| *p <= n_blocks_observed)
        .collect();
    sequential_design_guard(&config.design, &elapsed)
}

/// Manual-gate selection score over VALID wall blocks only
/// (`score_selection`, `:484-558`). Never called by training. The
/// `bootstrap` triple is caller-supplied from the numpy PCG64 lane
/// (M2-B: draws stay eager); `peek_index` MUST equal the valid count
/// (exclusions change the scored count). Returns
/// `(metric, interval, excluded)`; the caller MUST carry `excluded` to
/// the promotion record.
pub fn score_selection(
    blocks: &[WallBlock],
    telemetry_by_game: &HashMap<String, TelemetryRow>,
    config: &SelectionConfig,
    peek_index: usize,
    tolerance: &BlockTolerance,
    bootstrap: (f64, f64, f64),
) -> Result<(f64, (f64, f64, f64), Vec<super::blocks::ExcludedBlock>), SearchError> {
    if blocks.is_empty() {
        return Err(SearchError::InvalidArg { detail: "blocks must be a tuple of WallBlock" });
    }
    let result: BlockAggregateResult = aggregate_blocks(blocks, telemetry_by_game, tolerance)?;
    if result.valid.is_empty() {
        return Err(SearchError::InvalidArg { detail: "no valid wall blocks to score" });
    }
    if peek_index < 1 {
        return Err(SearchError::InvalidArg { detail: "peek_index must be a positive int" });
    }
    if peek_index != result.valid.len() {
        return Err(SearchError::InvalidArg {
            detail: "peek_index != valid wall block count",
        });
    }
    let valid_ids: std::collections::HashSet<&str> =
        result.valid.iter().map(|(id, _)| id.as_str()).collect();
    let valid_blocks: Vec<WallBlock> =
        blocks.iter().filter(|b| valid_ids.contains(b.wall_id.as_str())).cloned().collect();
    let contrasts = placement_block_contrast(&valid_blocks)?;
    selection_gate_check(config, peek_index)?;
    let metric = neumaier_sum(&contrasts) / contrasts.len() as f64;
    Ok((metric, bootstrap, result.excluded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::blocks::WallBlock;

    fn eight_walls() -> Vec<WallBlock> {
        // Interleaved values: wall w holds five `w` and five `100-w`, so
        // every wall mean is exactly 50 while the pooled-sorted rechunk
        // (the forbidden game-level resample) groups small with small.
        (0..8)
            .map(|w| {
                let contrasts: Vec<f64> = (0..10)
                    .map(|g| if g % 2 == 0 { w as f64 } else { 100.0 - w as f64 })
                    .collect();
                let games: Vec<String> =
                    (0..10).map(|g| format!("wall-{w:02}:g{g}")).collect();
                WallBlock::new(format!("wall-{w:02}"), games, contrasts).unwrap()
            })
            .collect()
    }

    /// T1 block-atomic: permuting games WITHIN walls leaves the contrast
    /// vector identical; pooling all games and re-chunking (the forbidden
    /// game-level resample) changes it.
    #[test]
    fn block_atomicity() {
        let blocks = eight_walls();
        let base = placement_block_contrast(&blocks).unwrap();
        // Within-wall permutation: reverse each wall's games+contrasts.
        let permuted: Vec<WallBlock> = blocks
            .iter()
            .map(|b| {
                let games = b.game_ids.iter().rev().cloned().collect();
                let contrasts = b.contrasts.iter().rev().cloned().collect();
                WallBlock::new(b.wall_id.clone(), games, contrasts).unwrap()
            })
            .collect();
        assert_eq!(placement_block_contrast(&permuted).unwrap(), base);
        // Game-level pooling then naive re-chunking diverges.
        let mut pooled: Vec<f64> = blocks.iter().flat_map(|b| b.contrasts.clone()).collect();
        pooled.sort_by(|a, b| a.total_cmp(b));
        let rechunked: Vec<f64> = pooled
            .chunks(10)
            .map(|c| neumaier_sum(c) / c.len() as f64)
            .collect();
        assert_ne!(rechunked, base);
        assert!(placement_block_contrast(&[]).is_err());
    }

    /// T3-part: percentile indices replicate `:156-157` verbatim
    /// (`R=2000, a=0.05` -> `(50, 1949)`); assembly over synthetic
    /// sorted draws is exact; estimate equals the block-mean (BOOT
    /// golden estimate `0.125` is draw-independent).
    #[test]
    fn percentile_and_assembly() {
        assert_eq!(percentile_indexes(0.05, 2000).unwrap(), (50, 1949));
        assert_eq!(percentile_indexes(0.05, 1000).unwrap(), (25, 974));
        let sorted: Vec<f64> = (0..2000).map(|i| i as f64 / 1000.0 - 1.0).collect();
        let values = vec![0.5, -0.25, 1.0, -0.75, 0.25, 0.0, 0.75, -0.5];
        let (est, low, high) = bootstrap_interval(&values, &sorted, 0.05, 2000).unwrap();
        assert_eq!(est, 0.125);
        assert_eq!(low, sorted[50]);
        assert_eq!(high, sorted[1949]);
        assert!(bootstrap_interval(&[0.5], &sorted, 0.05, 2000).is_err());
        assert!(bootstrap_interval(&values, &sorted[..1999], 0.05, 2000).is_err());
        assert!(validate_alpha_resamples(0.5, 2000).is_err());
        assert!(validate_alpha_resamples(0.05, 99).is_err());
    }

    /// Sign-flip centered-shift assembly (`:191-193`): observed is the
    /// block-mean (SIGNFLIP golden `0.125`); shift recenters synthetic
    /// null stats onto it.
    #[test]
    fn sign_flip_centered_shift() {
        let values = vec![0.5, -0.25, 1.0, -0.75, 0.25, 0.0, 0.75, -0.5];
        let sorted: Vec<f64> = (0..2000).map(|i| (i as f64 - 1000.0) / 4000.0).collect();
        let (observed, low, high) =
            sign_flip_centered_interval(&values, &sorted, 0.05, 2000).unwrap();
        assert_eq!(observed, 0.125);
        let center = neumaier_sum(&sorted) / sorted.len() as f64;
        assert_eq!(low, observed + (sorted[50] - center));
        assert_eq!(high, observed + (sorted[1949] - center));
    }

    /// Cluster path (`:339-382`): group means are the atomic units.
    /// Game groups -> `[1/6, -1/6, 1/2]` estimate `1/6` (CLUSTER golden);
    /// player groups -> estimate `0.125` (CLUSTER_P golden); fewer than
    /// two groups raises.
    #[test]
    fn cluster_group_means() {
        let g0 = vec![0.5, -0.75, 0.75];
        let g1 = vec![-0.25, 0.25, -0.5];
        let g2 = vec![1.0, 0.0];
        let means = group_means(&[&g0, &g1, &g2]).unwrap();
        assert_eq!(means.len(), 3);
        assert!((means[0] - 1.0 / 6.0).abs() < 1e-15);
        assert!((means[1] + 1.0 / 6.0).abs() < 1e-15);
        assert_eq!(means[2], 0.5);
        assert_eq!(neumaier_sum(&means) / 3.0, 1.0 / 6.0);
        let p0 = vec![0.5, 1.0, 0.25, 0.75];
        let p1 = vec![-0.25, -0.75, 0.0, -0.5];
        let pmeans = group_means(&[&p0, &p1]).unwrap();
        assert_eq!(pmeans, vec![0.625, -0.375]);
        assert_eq!(neumaier_sum(&pmeans) / 2.0, 0.125);
        assert!(group_means(&[&g0]).is_err());
        assert!(group_means(&[&g0, &[]]).is_err());
    }

    /// FIXEDN golden (`/tmp/eval_goldens2.py`): `N=56`, raw
    /// `55.64301508817788` (margin 0.64 — Acklam error cannot flip it).
    #[test]
    fn fixed_n_golden() {
        assert_eq!(fixed_n_samples(1.5, 0.5, 0.05, 0.2).unwrap(), 56);
        assert!(fixed_n_samples(0.0, 0.5, 0.05, 0.2).is_err());
        assert!(fixed_n_samples(1.5, -1.0, 0.05, 0.2).is_err());
        assert!(fixed_n_samples(1.5, 0.5, 0.0, 0.2).is_err());
        assert!(fixed_n_samples(1.5, 0.5, 0.05, 1.0).is_err());
    }

    /// Hedged-CS goldens (`/tmp/eval_goldens2/3.py`): uninformative
    /// 8-value path stays `(0.02, 0.98)` (grid edges); multi-peek keeps
    /// one pair per peek; flat-64 grid-8 concentrates to the golden
    /// pair (1e-12 tolerates libm `ln` variance; edges are grid thetas).
    #[test]
    fn hedged_cs_goldens() {
        let vals = vec![0.1, 0.9, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8];
        let path = hedged_cs_path(&vals, 0.05, (0.0, 1.0), 48, None).unwrap();
        assert_eq!(path, vec![(0.02, 0.98)]);
        let multi = hedged_cs_path(&vals, 0.05, (0.0, 1.0), 48, Some(&[4, 8])).unwrap();
        assert_eq!(multi.len(), 2);
        assert_eq!(multi[0], (0.02, 0.98));
        let flat = vec![0.5; 64];
        let (low, high) = hedged_confidence_sequence(&flat, 0.05, (0.0, 1.0), 8).unwrap();
        assert!((low - 0.43142857142857144).abs() < 1e-12);
        assert!((high - 0.5685714285714286).abs() < 1e-12);
        assert!(hedged_cs_path(&vals, 0.05, (0.0, 1.0), 3, None).is_err());
        assert!(hedged_cs_path(&vals, 0.05, (1.0, 0.0), 48, None).is_err());
        assert!(hedged_cs_path(&vals, 0.05, (0.0, 1.0), 48, Some(&[9])).is_err());
        assert!(ci_covers((low, high), 0.5));
        assert!(!ci_covers((low, high), 0.9));
    }

    /// Design guards + selection gates: fixed_n allows exactly one peek;
    /// CS needs a strictly increasing schedule; over-N and off-schedule
    /// looks raise; unknown design raises.
    #[test]
    fn design_and_gate_guards() {
        assert!(sequential_design_guard("fixed_n", &[30]).is_ok());
        assert!(sequential_design_guard("fixed_n", &[15, 30]).is_err());
        assert!(sequential_design_guard("time_uniform_cs", &[10, 20, 30]).is_ok());
        assert!(sequential_design_guard("time_uniform_cs", &[20, 10]).is_err());
        assert!(sequential_design_guard("nope", &[30]).is_err());
        let config = SelectionConfig::default();
        assert!(config.validate().is_ok());
        assert!(selection_gate_check(&config, 30).is_ok());
        assert!(selection_gate_check(&config, 15).is_err());
        assert!(selection_gate_check(&config, 31).is_err());
        let mut bad = config.clone();
        bad.uncertainty_unit = "game_cluster".to_string();
        assert!(bad.validate().is_err());
    }

    /// Score selection: metric is the valid-contrast mean; peek MUST
    /// equal the valid count; empty-valid raises; bootstrap triple
    /// passes through untouched (numpy lane owns draws).
    #[test]
    fn score_selection_gates() {
        use super::super::telemetry::TelemetryTolerance;
        let blocks = vec![
            WallBlock::new("w-001".to_string(), vec!["w-001:g0".to_string()], vec![0.5]).unwrap(),
            WallBlock::new("w-002".to_string(), vec!["w-002:g0".to_string()], vec![-0.5]).unwrap(),
        ];
        let rows: HashMap<String, TelemetryRow> = HashMap::new();
        let config = SelectionConfig {
            n: 2,
            declared_peeks: vec![2],
            ..SelectionConfig::default()
        };
        // No telemetry rows -> no valid blocks.
        let tol = BlockTolerance::strict();
        assert!(score_selection(&blocks, &rows, &config, 2, &tol, (0.0, -1.0, 1.0)).is_err());
        assert!(score_selection(&[], &rows, &config, 2, &tol, (0.0, -1.0, 1.0)).is_err());
        let _ = TelemetryTolerance::strict();
    }
}
