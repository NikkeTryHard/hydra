//! `joint`: prior/posterior (`joint_planner.py` + `joint_uncertainty.py`).
//!
//! Parity: `_ensure_joint_prior` (`joint_planner.py:115-227`: uniform over
//! `Theta x Worlds`, `2`-world default perms `((0,1),(2,3),(4,5),(6,7))`,
//! `weight = 1/total`) + `exact_joint_posterior_oracle`
//! (`joint_uncertainty.py:47-152`: likelihood EXACTLY ONCE per particle,
//! `lik = exp(lp)` in `(0,1]` finite else `ContractError`, `w_u =
//! w*lik*T`, `Z > 0` finite, normalize, sum `1 +- 1e-9` — correlation
//! preserved, NEVER a marginal product) + `deterministic_joint_gumbel`
//! (`joint_types.py:196-216`: `DOMAIN = b"joint_type_world_gumbel_v1"`,
//! `payload = f"{case}:{seat}:{cand}:{theta}:{action}"`, same `u/G/clamp`
//! pipeline as root Gumbels) + `robust_select` (`joint_planner.py:322`:
//! `isclose(abs_tol=1e-12)`, min id on ties).
//!
//! Frozen type space: `theta in {tight, loose}`.

use sha2::{Digest, Sha256};

use crate::SearchError;

/// Frozen theta ids (`joint_types.py:181`).
pub const THETA_TIGHT: &str = "tight";
/// Frozen theta ids (`joint_types.py:181`).
pub const THETA_LOOSE: &str = "loose";
/// Joint Gumbel domain (`_JOINT_GUMBEL_DOMAIN`).
pub const JOINT_GUMBEL_DOMAIN: &[u8] = b"joint_type_world_gumbel_v1";
/// Posterior sum gate (`math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-9)`).
pub const JOINT_SUM_TOL: f64 = 1e-9;
/// Robust-select tie window (`isclose(abs_tol=1e-12)`).
pub const JOINT_TIE_EPS: f64 = 1e-12;
/// Gumbel clip/clamp (same pipeline as root Gumbels).
pub const JOINT_CLIP: f64 = 1e-12;
/// Gumbel clamp bound.
pub const JOINT_CLAMP: f64 = 20.0;
/// `2^64` as `f64`.
pub const JOINT_U64_DENOM: f64 = 18_446_744_073_709_551_616.0;

/// Check a theta id against the frozen space.
pub fn check_theta(theta: &str) -> Result<(), SearchError> {
    if theta == THETA_TIGHT || theta == THETA_LOOSE {
        return Ok(());
    }
    Err(SearchError::InvalidArg { detail: "theta must be tight|loose" })
}

/// One joint particle: `(theta, world_ref, weight)`.
#[derive(Debug, Clone, PartialEq)]
pub struct JointParticle {
    /// Type id (`tight|loose`).
    pub theta: String,
    /// Opaque world handle (arena-side `u64`).
    pub world_ref: u64,
    /// Normalized weight (`>= 0`, finite).
    pub weight: f64,
}

/// Uniform joint prior over `Theta x Worlds`
/// (`_ensure_joint_prior`: `weight_each = 1/total`).
pub fn ensure_prior(thetas: &[&str], worlds: &[u64]) -> Result<Vec<JointParticle>, SearchError> {
    if thetas.is_empty() || worlds.is_empty() {
        return Err(SearchError::InvalidArg { detail: "prior needs non-empty theta x worlds" });
    }
    for theta in thetas {
        check_theta(theta)?;
    }
    let total = thetas.len() * worlds.len();
    let weight = 1.0 / total as f64;
    if !weight.is_finite() || weight <= 0.0 {
        return Err(SearchError::NonFinite { context: "joint prior weight" });
    }
    let mut out = Vec::with_capacity(total);
    for theta in thetas {
        for world in worlds {
            out.push(JointParticle {
                theta: (*theta).to_string(),
                world_ref: *world,
                weight,
            });
        }
    }
    Ok(out)
}

/// Exact joint posterior (`exact_joint_posterior_oracle` parity):
/// `w_u = w * lik * T` per particle (likelihood EXACTLY ONCE —
/// `likelihoods[i]` is consumed once, in order), `lik in (0,1]` finite
/// else `BadLikelihood`, `Z > 0` finite else `ZeroMass`, normalize, sum
/// `1 +- 1e-9` else `BadPartition` (mass audit).
pub fn exact_posterior(
    prior: &[JointParticle],
    likelihoods: &[f64],
    physical_t: f64,
) -> Result<Vec<JointParticle>, SearchError> {
    if prior.is_empty() {
        return Err(SearchError::InvalidArg { detail: "prior must be non-empty" });
    }
    if prior.len() != likelihoods.len() {
        return Err(SearchError::InvalidArg { detail: "likelihoods must align with prior" });
    }
    if !physical_t.is_finite() || physical_t <= 0.0 || physical_t > 1.0 {
        return Err(SearchError::InvalidArg { detail: "physical_transition_prob must be in (0,1]" });
    }
    let mut unnorm = Vec::with_capacity(prior.len());
    let mut total = 0.0;
    let mut i = 0;
    while i < prior.len() {
        let lik = likelihoods[i];
        // Likelihood enters EXACTLY ONCE and audits to (0,1] finite.
        if !lik.is_finite() || lik <= 0.0 || lik > 1.0 {
            return Err(SearchError::BadLikelihood);
        }
        let wu = prior[i].weight * lik * physical_t;
        if !wu.is_finite() || wu < 0.0 {
            return Err(SearchError::NonFinite { context: "joint unnorm weight" });
        }
        unnorm.push(wu);
        total += wu;
        i += 1;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(SearchError::ZeroMass);
    }
    let mut out = Vec::with_capacity(prior.len());
    let mut sum = 0.0;
    let mut j = 0;
    while j < prior.len() {
        let w = unnorm[j] / total;
        if !w.is_finite() || w < 0.0 {
            return Err(SearchError::NonFinite { context: "joint norm weight" });
        }
        sum += w;
        out.push(JointParticle {
            theta: prior[j].theta.clone(),
            world_ref: prior[j].world_ref,
            weight: w,
        });
        j += 1;
    }
    if (sum - 1.0).abs() > JOINT_SUM_TOL {
        return Err(SearchError::BadPartition);
    }
    Ok(out)
}

/// Deterministic joint Gumbel (`joint_types.py:196-216`): same `u/G/clamp`
/// pipeline as root Gumbels, payload extended with `theta`.
pub fn deterministic_joint_gumbel(
    case_id: &str,
    root_seat: u32,
    candidate_id: &str,
    theta: &str,
    action_id: u32,
) -> Result<f64, SearchError> {
    if case_id.is_empty() {
        return Err(SearchError::InvalidArg { detail: "case_id must be non-empty" });
    }
    if root_seat >= 4 {
        return Err(SearchError::InvalidSeat { seat: root_seat });
    }
    if candidate_id.is_empty() {
        return Err(SearchError::InvalidArg { detail: "candidate_id must be non-empty" });
    }
    check_theta(theta)?;
    let payload = format!("{case_id}:{root_seat}:{candidate_id}:{theta}:{action_id}");
    let mut hasher = Sha256::new();
    hasher.update(JOINT_GUMBEL_DOMAIN);
    hasher.update(payload.as_bytes());
    let digest = hasher.finalize();
    let mut raw = [0u8; 8];
    raw.copy_from_slice(digest.as_slice()[..8].as_ref());
    let mut u = (u64::from_be_bytes(raw) as f64 + 0.5) / JOINT_U64_DENOM;
    if u < JOINT_CLIP {
        u = JOINT_CLIP;
    }
    if u > 1.0 - JOINT_CLIP {
        u = 1.0 - JOINT_CLIP;
    }
    let g = -(-u.ln()).ln();
    if !g.is_finite() {
        return Err(SearchError::NonFinite { context: "joint gumbel" });
    }
    Ok(g.clamp(-JOINT_CLAMP, JOINT_CLAMP))
}

/// Robust select over posterior expected values (`joint_planner.py:322`):
/// max nominal wins; `isclose(abs_tol=1e-12)` ties break to min id.
/// `values` carries `(action, expected_value)`; joint-Gumbel perturbation
/// is the caller's composition (same `+-20` clamp family).
pub fn robust_select(values: &[(u32, f64)]) -> Result<u32, SearchError> {
    if values.is_empty() {
        return Err(SearchError::EmptyLegal);
    }
    let mut best: Option<u32> = None;
    let mut best_score = f64::NEG_INFINITY;
    for (aid, value) in values {
        if !value.is_finite() {
            return Err(SearchError::NonFinite { context: "joint robust value" });
        }
        match best {
            None => {
                best = Some(*aid);
                best_score = *value;
            }
            Some(current) => {
                if *value > best_score
                    || ((*value - best_score).abs() <= JOINT_TIE_EPS && *aid < current)
                {
                    // NOTE: oracle uses `>` OR (`isclose` AND min-id); the
                    // `>` arm subsumes the eps gate for strict improvements.
                    best = Some(*aid);
                    best_score = *value;
                }
            }
        }
    }
    best.ok_or(SearchError::EmptyLegal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_is_uniform_over_product() {
        let prior = ensure_prior(&["tight", "loose"], &[100, 200]).expect("prior");
        assert_eq!(prior.len(), 4);
        for particle in &prior {
            assert!((particle.weight - 0.25).abs() < 1e-15);
        }
        assert!(ensure_prior(&["wide"], &[1]).is_err());
        assert!(ensure_prior(&["tight"], &[]).is_err());
    }

    #[test]
    fn posterior_scales_once_and_normalizes() {
        // Prior uniform over 2; likelihoods (1.0, 0.5), T=1:
        // unnorm (0.5, 0.25) -> (2/3, 1/3).
        let prior = ensure_prior(&["tight"], &[1, 2]).expect("prior");
        let post = exact_posterior(&prior, &[1.0, 0.5], 1.0).expect("post");
        assert!((post[0].weight - 2.0 / 3.0).abs() < 1e-12);
        assert!((post[1].weight - 1.0 / 3.0).abs() < 1e-12);
        // Correlation preserved: particle ORDER and identity survive.
        assert_eq!(post[0].world_ref, 1);
        assert_eq!(post[1].world_ref, 2);
    }

    #[test]
    fn posterior_rejects_double_count_and_zero_mass() {
        let prior = ensure_prior(&["tight"], &[1, 2]).expect("prior");
        // Likelihood > 1 (double-count shape) -> BadLikelihood.
        assert_eq!(
            exact_posterior(&prior, &[1.5, 0.5], 1.0),
            Err(SearchError::BadLikelihood)
        );
        // Non-finite likelihood -> BadLikelihood.
        assert!(matches!(
            exact_posterior(&prior, &[f64::NAN, 0.5], 1.0),
            Err(SearchError::BadLikelihood)
        ));
        // Misaligned lengths rejected.
        assert!(exact_posterior(&prior, &[0.5], 1.0).is_err());
        // Bad physical-T rejected.
        assert!(exact_posterior(&prior, &[0.5, 0.5], 0.0).is_err());
        assert!(exact_posterior(&prior, &[0.5, 0.5], 1.5).is_err());
    }

    #[test]
    fn joint_gumbel_pipeline_matches_root_shape() {
        // Same u/G/clamp pipeline as root gumbels, theta-extended payload.
        let payload = "jc:0:jcand:tight:5";
        let mut hasher = Sha256::new();
        hasher.update(JOINT_GUMBEL_DOMAIN);
        hasher.update(payload.as_bytes());
        let digest = hasher.finalize();
        let mut raw = [0u8; 8];
        raw.copy_from_slice(digest.as_slice()[..8].as_ref());
        let u = ((u64::from_be_bytes(raw) as f64 + 0.5) / JOINT_U64_DENOM)
            .clamp(JOINT_CLIP, 1.0 - JOINT_CLIP);
        let expected = (-(-u.ln()).ln()).clamp(-JOINT_CLAMP, JOINT_CLAMP);
        let got =
            deterministic_joint_gumbel("jc", 0, "jcand", "tight", 5).expect("gumbel");
        assert!((got - expected).abs() < 1e-15);
        // Theta perturbs: tight != loose for the same arm.
        let loose =
            deterministic_joint_gumbel("jc", 0, "jcand", "loose", 5).expect("gumbel");
        assert_ne!(got, loose);
        assert!(deterministic_joint_gumbel("jc", 0, "jcand", "wide", 5).is_err());
    }

    #[test]
    fn robust_select_tie_to_min_id() {
        assert_eq!(robust_select(&[(9, 1.0), (4, 1.0)]).expect("sel"), 4);
        assert_eq!(robust_select(&[(4, 0.5), (9, 0.9)]).expect("sel"), 9);
        assert_eq!(robust_select(&[]), Err(SearchError::EmptyLegal));
    }
}
