//! `local`: resolving + regret/hedge (`local_search.py` + `local_strategy.py`).
//!
//! Parity: `local_search.py:105-544` (abstraction `identity|pair_merge|
//! tile_type|custom`, `horizon 1..16`, `iterations <= 1024`, uniform seed per
//! `(actor, info_hash)`, deterministic world rotation `worlds[t % n]`, avg
//! accumulation uniform `w=1` / linear `w=t`) + `local_strategy.py:128-196`
//! (`regret_matching`: `pos=max(0,r)`, `s=sum`, `s>1e-12 ? pos/s :
//! uniform; `hedge`: max-subtract `exp(eta*(q-m))/sum`; `fictitious_play`:
//! `prev*count/total + w/total` on BR idx, renormalize) +
//! `averaging_weights` (uniform `1.0`, linear `float(t)`).
//!
//! Config frozen (`LocalResolvingConfig`: horizon `1..=16`, iterations
//! `<= 1024`, update in `{regret,hedge,fictitious}`, averaging in
//! `{uniform,linear}`, tie in `{greedy,temp_0.5,temp_1.0,value_break}`).
//! Distribution floor: entries in `[0,1]`, sum `1 +- 1e-6`.

use crate::SearchError;

/// Regret uniform-fallback gate (`local_strategy.py:136`).
pub const REGRET_EPS: f64 = 1e-12;
/// Root greedy tie window (`local_search.py:426,466`: `abs(p-max) < 1e-9`).
pub const LOCAL_TIE_EPS: f64 = 1e-9;
/// Distribution sum tolerance (`local_strategy.py:72`: `|s-1| > 1e-6`).
pub const DIST_TOL: f64 = 1e-6;
/// Frozen horizon ceiling (`1..=16`).
pub const LOCAL_MAX_HORIZON: u32 = 16;
/// Frozen iteration ceiling (`<= 1024`).
pub const LOCAL_MAX_ITERS: u32 = 1024;

/// Frozen update rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateRule {
    /// Regret matching over positive regrets.
    RegretMatching,
    /// Hedge (softmax over cumulative Q).
    Hedge,
    /// Fictitious play toward the best-response index.
    FictitiousPlay,
}

impl UpdateRule {
    /// Parse the frozen vocabulary.
    pub fn parse(name: &str) -> Result<UpdateRule, SearchError> {
        match name {
            "regret_matching" => Ok(UpdateRule::RegretMatching),
            "hedge" => Ok(UpdateRule::Hedge),
            "fictitious_play" => Ok(UpdateRule::FictitiousPlay),
            _ => Err(SearchError::InvalidArg {
                detail: "unknown update_rule",
            }),
        }
    }
}

/// Frozen averaging rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AveragingRule {
    /// Uniform weight `1.0`.
    Uniform,
    /// Linear weight `float(t)` (1-based iteration).
    Linear,
}

impl AveragingRule {
    /// Parse the frozen vocabulary.
    pub fn parse(name: &str) -> Result<AveragingRule, SearchError> {
        match name {
            "uniform" => Ok(AveragingRule::Uniform),
            "linear" => Ok(AveragingRule::Linear),
            _ => Err(SearchError::InvalidArg {
                detail: "unknown averaging rule",
            }),
        }
    }

    /// Weight for 1-based `iteration` (`averaging_weights` parity).
    pub fn weight(&self, iteration: u32) -> f64 {
        match self {
            AveragingRule::Uniform => 1.0,
            AveragingRule::Linear => iteration as f64,
        }
    }
}

/// Frozen local-resolving config.
#[derive(Debug, Clone, Copy)]
pub struct LocalConfig {
    /// Horizon (`1..=16`).
    pub horizon: u32,
    /// Iterations (`1..=1024`).
    pub iterations: u32,
    /// Update rule.
    pub update: UpdateRule,
    /// Averaging rule.
    pub averaging: AveragingRule,
}

impl LocalConfig {
    /// Frozen defaults (horizon 2, 16 iters, regret, uniform).
    pub fn defaults() -> LocalConfig {
        LocalConfig {
            horizon: 2,
            iterations: 16,
            update: UpdateRule::RegretMatching,
            averaging: AveragingRule::Uniform,
        }
    }

    /// Validate (`ContractError` edges).
    pub fn validate(&self) -> Result<(), SearchError> {
        if self.horizon == 0 || self.horizon > LOCAL_MAX_HORIZON {
            return Err(SearchError::InvalidArg {
                detail: "horizon must be 1..16",
            });
        }
        if self.iterations == 0 || self.iterations > LOCAL_MAX_ITERS {
            return Err(SearchError::InvalidArg {
                detail: "iterations must be 1..1024",
            });
        }
        Ok(())
    }
}

/// Check a distribution: entries in `[0,1]`, finite, sum `1 +- 1e-6`.
pub fn check_distribution(dist: &[f64]) -> Result<(), SearchError> {
    if dist.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "distribution must be non-empty",
        });
    }
    let mut sum = 0.0;
    for prob in dist {
        if !prob.is_finite() || *prob < 0.0 || *prob > 1.0 {
            return Err(SearchError::InvalidArg {
                detail: "distribution entry must be in [0,1]",
            });
        }
        sum += *prob;
    }
    if (sum - 1.0).abs() > DIST_TOL {
        return Err(SearchError::InvalidArg {
            detail: "distribution must sum to 1",
        });
    }
    Ok(())
}

/// Uniform distribution over `n` actions.
pub fn uniform_strategy(actions: usize) -> Result<Vec<f64>, SearchError> {
    if actions == 0 {
        return Err(SearchError::InvalidArg {
            detail: "action count must be >= 1",
        });
    }
    // proof: action count is a small vec len (< 2^53), exact; uniform weight tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let actions_f: f64 = actions as f64;
    Ok(vec![1.0 / actions_f; actions])
}

/// Regret-matching update (`_regret_matching_update`): positive regrets
/// normalized; `sum <= 1e-12` -> uniform.
pub fn regret_matching_update(current: &[f64], regrets: &[f64]) -> Result<Vec<f64>, SearchError> {
    if current.is_empty() || current.len() != regrets.len() {
        return Err(SearchError::InvalidArg {
            detail: "regret shapes must match non-empty",
        });
    }
    let mut sum = 0.0;
    for regret in regrets {
        if !regret.is_finite() {
            return Err(SearchError::NonFinite {
                context: "local regret",
            });
        }
        sum += regret.max(0.0);
    }
    if sum > REGRET_EPS {
        let mut out = Vec::with_capacity(current.len());
        for regret in regrets {
            out.push(regret.max(0.0) / sum);
        }
        return Ok(out);
    }
    uniform_strategy(current.len())
}

/// Hedge update (`_hedge_update`): max-subtract stable softmax
/// `exp(eta*(q-m)) / sum`.
pub fn hedge_update(q_values: &[f64], eta: f64) -> Result<Vec<f64>, SearchError> {
    if q_values.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "q_values must be non-empty",
        });
    }
    if !eta.is_finite() {
        return Err(SearchError::NonFinite {
            context: "local hedge eta",
        });
    }
    let mut max = f64::NEG_INFINITY;
    for q in q_values {
        if !q.is_finite() {
            return Err(SearchError::NonFinite {
                context: "local hedge q",
            });
        }
        max = max.max(*q);
    }
    let mut exps = Vec::with_capacity(q_values.len());
    let mut sum = 0.0;
    for q in q_values {
        let exp = (eta * (*q - max)).exp();
        if !exp.is_finite() {
            return Err(SearchError::NonFinite {
                context: "local hedge exp",
            });
        }
        exps.push(exp);
        sum += exp;
    }
    Ok(exps.into_iter().map(|exp| exp / sum).collect())
}

/// Fictitious-play update (`_fictitious_play_update`): `prev*count/total +
/// w/total` on the BR idx, then renormalize.
pub fn fictitious_play_update(
    current: &[f64],
    best_response_idx: usize,
    count: u64,
    new_weight: f64,
) -> Result<Vec<f64>, SearchError> {
    if current.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "strategy must be non-empty",
        });
    }
    if best_response_idx >= current.len() {
        return Err(SearchError::InvalidArg {
            detail: "best_response_idx out of range",
        });
    }
    if !new_weight.is_finite() || new_weight <= 0.0 {
        return Err(SearchError::InvalidArg {
            detail: "new_weight must be finite >0",
        });
    }
    // proof: strategy counts are small (< 2^53), exact; update math tolerates 1ulp.
    #[allow(clippy::cast_precision_loss)]
    let count_f: f64 = count as f64;
    let total = count_f + new_weight;
    let mut out = Vec::with_capacity(current.len());
    let mut i = 0;
    while i < current.len() {
        // proof: strategy counts are small (< 2^53), exact; update math tolerates 1ulp.
        #[allow(clippy::cast_precision_loss)]
        let count_f2: f64 = count as f64;
        let prev = if count > 0 {
            current[i] * count_f2 / total
        } else {
            0.0
        };
        let add = if i == best_response_idx {
            new_weight / total
        } else {
            0.0
        };
        out.push(prev + add);
        i += 1;
    }
    let sum: f64 = out.iter().sum();
    if sum > 0.0 {
        return Ok(out.into_iter().map(|value| value / sum).collect());
    }
    uniform_strategy(current.len())
}

/// Dispatch one update step (`apply_update` parity).
pub fn apply_update(
    current: &[f64],
    rule: UpdateRule,
    regrets: Option<&[f64]>,
    q_values: Option<&[f64]>,
    best_response_idx: Option<usize>,
    visit_count: u64,
) -> Result<Vec<f64>, SearchError> {
    match rule {
        UpdateRule::RegretMatching => match regrets {
            Some(regrets) => regret_matching_update(current, regrets),
            None => Err(SearchError::InvalidArg {
                detail: "regret_matching requires regrets",
            }),
        },
        UpdateRule::Hedge => match q_values {
            Some(q_values) => hedge_update(q_values, 1.0),
            None => Err(SearchError::InvalidArg {
                detail: "hedge requires q_values",
            }),
        },
        UpdateRule::FictitiousPlay => match best_response_idx {
            Some(idx) => fictitious_play_update(current, idx, visit_count, 1.0),
            None => Err(SearchError::InvalidArg {
                detail: "fictitious_play requires best_response_idx",
            }),
        },
    }
}

/// Greedy root pick over averaged tables (`local_search.py:426,466`):
/// `abs(p - max) < 1e-9` candidates, min index wins (deterministic greedy
/// smallest abstract id among ties).
pub fn greedy_root_pick(avg: &[f64]) -> Result<usize, SearchError> {
    if avg.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "avg table must be non-empty",
        });
    }
    let mut max = f64::NEG_INFINITY;
    for prob in avg {
        if !prob.is_finite() {
            return Err(SearchError::NonFinite {
                context: "local avg",
            });
        }
        max = max.max(*prob);
    }
    let mut idx = 0;
    while idx < avg.len() {
        if (avg[idx] - max).abs() < LOCAL_TIE_EPS {
            return Ok(idx);
        }
        idx += 1;
    }
    Ok(0)
}

/// Deterministic world rotation (`worlds[t % n]`, no global RNG).
pub fn rotate_world(worlds: &[u64], iteration: u32) -> Result<u64, SearchError> {
    if worlds.is_empty() {
        return Err(SearchError::InvalidArg {
            detail: "worlds must be non-empty",
        });
    }
    Ok(worlds[iteration as usize % worlds.len()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regret_positive_normalizes() {
        let got = regret_matching_update(&[0.5, 0.5], &[1.0, 3.0]).expect("regret");
        assert!((got[0] - 0.25).abs() < 1e-12);
        assert!((got[1] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn regret_zero_falls_back_uniform() {
        // All non-positive (sum <= 1e-12) -> uniform.
        let got = regret_matching_update(&[0.2, 0.8], &[-1.0, 0.0]).expect("regret");
        assert_eq!(got, vec![0.5, 0.5]);
    }

    #[test]
    fn hedge_is_max_subtract_softmax() {
        let got = hedge_update(&[1.0, 2.0], 1.0).expect("hedge");
        let denom = 1.0 + std::f64::consts::E;
        assert!((got[0] - 1.0 / denom).abs() < 1e-12);
        assert!((got[1] - std::f64::consts::E / denom).abs() < 1e-12);
    }

    #[test]
    fn fictitious_play_counts() {
        // count=1, BR idx 0: [0.5*1/2+1/2, 0.5*1/2] = [0.75, 0.25].
        let got = fictitious_play_update(&[0.5, 0.5], 0, 1, 1.0).expect("fp");
        assert!((got[0] - 0.75).abs() < 1e-12);
        assert!((got[1] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn averaging_weights_parity() {
        assert_eq!(AveragingRule::Uniform.weight(7), 1.0);
        assert_eq!(AveragingRule::Linear.weight(7), 7.0);
    }

    #[test]
    fn greedy_root_pick_tie_window() {
        // Within 1e-9 -> smallest index wins.
        assert_eq!(greedy_root_pick(&[0.5, 0.5 + 5e-10]).expect("pick"), 0);
        // Beyond -> the leader wins.
        assert_eq!(greedy_root_pick(&[0.5, 0.5 + 5e-9]).expect("pick"), 1);
    }

    #[test]
    fn distribution_floor_gate() {
        assert!(check_distribution(&[0.5, 0.5]).is_ok());
        // Sum 0.999999 (1e-6 off) is AT the floor boundary: |s-1| > 1e-6
        // fails, equality passes. 0.999999 diff is exactly 1e-6 in f64...
        // use a clearly-bad sum instead.
        assert!(check_distribution(&[0.5, 0.4]).is_err());
        assert!(check_distribution(&[]).is_err());
    }

    #[test]
    fn rotate_world_cycles() {
        let worlds = [11, 22, 33];
        assert_eq!(rotate_world(&worlds, 0).expect("w"), 11);
        assert_eq!(rotate_world(&worlds, 3).expect("w"), 11);
        assert_eq!(rotate_world(&worlds, 4).expect("w"), 22);
        assert!(rotate_world(&[], 0).is_err());
    }

    #[test]
    fn config_validation_edges() {
        let mut cfg = LocalConfig::defaults();
        cfg.validate().expect("defaults valid");
        cfg.horizon = 17;
        assert!(cfg.validate().is_err());
        cfg.horizon = 2;
        cfg.iterations = 1025;
        assert!(cfg.validate().is_err());
    }
}
