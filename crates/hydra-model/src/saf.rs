//! Search-as-Feature (SaF) MLP adaptor.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{
    NUM_CHANNELS, NUM_TILES, SEARCH_DELTA_Q_CHANNEL, SEARCH_MASK_CHANNEL_START,
    SEARCH_MIXTURE_ENTROPY_CHANNEL, SEARCH_MIXTURE_ESS_CHANNEL, SEARCH_RISK_CHANNEL_START,
    SEARCH_STRESS_CHANNEL_START,
};

/// Number of scalar features consumed by the SaF MLP per action.
pub const SAF_INPUT_DIM: usize = 8;
const MAX_MIXTURE_ENTROPY: f32 = 1.386_294_4;

/// Per-action search-derived features decoded from observation planes.
#[derive(Debug, Clone, Copy, Default)]
pub struct SafFeatures {
    /// Search delta-Q value for this action.
    pub delta_q: f32,
    /// Maximum robust risk across opponents.
    pub boole_risk: f32,
    /// Mean robust risk across opponents.
    pub hunter_risk: f32,
    /// Stress-weighted robust risk.
    pub robust_risk: f32,
    /// Normalized entropy reduction for belief mixture.
    pub entropy_drop: f32,
    /// Maximum robust stress value across opponents.
    pub tau_robust: f32,
    /// Variance of per-opponent risk values.
    pub variance: f32,
    /// Normalized effective sample size.
    pub ess: f32,
}

impl SafFeatures {
    /// Return features in stable SaF MLP input order.
    pub fn to_array(&self) -> [f32; SAF_INPUT_DIM] {
        [
            self.delta_q,
            self.boole_risk,
            self.hunter_risk,
            self.robust_risk,
            self.entropy_drop,
            self.tau_robust,
            self.variance,
            self.ess,
        ]
    }
}

#[inline]
fn obs_value(obs: &[f32], channel: usize, tile: usize) -> f32 {
    obs[channel * NUM_TILES + tile]
}

/// Decode per-action SaF features from the fixed-superset observation tensor.
///
/// This lets the fast inference path consume real Group C context instead of
/// falling back to all-zero SaF features.
pub fn saf_features_from_observation(obs: &[f32]) -> [SafFeatures; HYDRA_ACTION_SPACE] {
    assert!(
        obs.len() >= NUM_CHANNELS * NUM_TILES,
        "observation length {} shorter than expected {}",
        obs.len(),
        NUM_CHANNELS * NUM_TILES
    );

    let belief_present = obs_value(obs, SEARCH_MASK_CHANNEL_START, 0) > 0.5;
    let search_present = obs_value(obs, SEARCH_MASK_CHANNEL_START + 1, 0) > 0.5;
    let robust_present = obs_value(obs, SEARCH_MASK_CHANNEL_START + 2, 0) > 0.5;

    let entropy = if belief_present {
        obs_value(obs, SEARCH_MIXTURE_ENTROPY_CHANNEL, 0)
    } else {
        0.0
    };
    let entropy_drop = if belief_present {
        (1.0 - entropy / MAX_MIXTURE_ENTROPY).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let ess = if belief_present {
        (obs_value(obs, SEARCH_MIXTURE_ESS_CHANNEL, 0) / 4.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut features = [SafFeatures::default(); HYDRA_ACTION_SPACE];
    for (action, feature) in features.iter_mut().enumerate().take(NUM_TILES) {
        let delta_q = if search_present {
            obs_value(obs, SEARCH_DELTA_Q_CHANNEL, action)
        } else {
            0.0
        };

        let mut risk_values = [0.0f32; 3];
        let mut stress_values = [0.0f32; 3];
        if robust_present {
            for opp in 0..3 {
                risk_values[opp] = obs_value(obs, SEARCH_RISK_CHANNEL_START + opp, action);
                stress_values[opp] = obs_value(obs, SEARCH_STRESS_CHANNEL_START + opp, 0);
            }
        }

        let boole_risk = risk_values.iter().copied().fold(0.0f32, f32::max);
        let hunter_risk = (risk_values[0] + risk_values[1] + risk_values[2]) / 3.0;
        let robust_risk = risk_values
            .iter()
            .copied()
            .zip(stress_values.iter().copied())
            .map(|(risk, stress)| risk * stress)
            .fold(0.0f32, f32::max);
        let tau_robust = stress_values.iter().copied().fold(0.0f32, f32::max);
        let risk_mean = hunter_risk;
        let variance = risk_values
            .iter()
            .copied()
            .map(|risk| {
                let centered = risk - risk_mean;
                centered * centered
            })
            .sum::<f32>()
            / 3.0;

        *feature = SafFeatures {
            delta_q,
            boole_risk,
            hunter_risk,
            robust_risk,
            entropy_drop,
            tau_robust,
            variance,
            ess,
        };
    }

    features
}

/// Convert observation-derived SaF features into a `[46, 8]` tensor.
pub fn saf_tensor_from_observation<B: Backend>(
    obs: &[f32],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Tensor<B, 2> {
    let features = saf_features_from_observation(obs);
    let mut flat = [0.0f32; HYDRA_ACTION_SPACE * SAF_INPUT_DIM];
    for (action, feature) in features.iter().enumerate() {
        let start = action * SAF_INPUT_DIM;
        flat[start..start + SAF_INPUT_DIM].copy_from_slice(&feature.to_array());
    }
    Tensor::<B, 1>::from_floats(flat.as_slice(), device)
        .reshape([HYDRA_ACTION_SPACE, SAF_INPUT_DIM])
}

/// SaF training lane selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafTrainingMode {
    /// Train SaF by supervised regression to search outputs.
    SupervisedRegression,
    /// Train SaF jointly with policy learning.
    JointEndToEnd,
}

/// Build a binary SaF dropout mask from caller-provided random values.
pub fn saf_dropout_mask(batch_size: usize, drop_prob: f32, rng_vals: &[f32]) -> Vec<f32> {
    (0..batch_size)
        .map(|i| {
            if rng_vals.get(i).copied().unwrap_or(1.0) < drop_prob {
                0.0
            } else {
                1.0
            }
        })
        .collect()
}

impl SafConfig {
    /// Return a compact human-readable configuration summary.
    pub fn summary(&self) -> String {
        format!(
            "saf(alpha={:.1}, drop={:.1}, dim={})",
            self.alpha, self.dropout, self.hidden_dim
        )
    }

    /// Validate SaF dimensions and dropout range.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.hidden_dim == 0 {
            return Err("hidden_dim must be > 0");
        }
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err("dropout must be in [0,1)");
        }
        Ok(())
    }
}

/// Add SaF output into base logits under `mask` and `alpha`.
pub fn apply_saf_logit<B: Backend>(
    base_logits: Tensor<B, 2>,
    saf_output: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    alpha: f32,
) -> Tensor<B, 2> {
    base_logits + saf_output * mask * alpha
}

/// Configuration for the SaF MLP adaptor.
#[derive(Config, Debug)]
pub struct SafConfig {
    #[config(default = "1.0")]
    /// Scale applied to SaF logit residuals.
    pub alpha: f32,
    #[config(default = "0.3")]
    /// Probability of dropping SaF features during training.
    pub dropout: f32,
    #[config(default = "32")]
    /// Hidden layer width for the SaF MLP.
    pub hidden_dim: usize,
}

/// Two-layer MLP mapping SaF feature vectors to logit residuals.
#[derive(Module, Debug)]
pub struct SafMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl SafConfig {
    /// Initialize the SaF MLP on `device`.
    pub fn init<B: Backend>(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> SafMlp<B> {
        SafMlp {
            fc1: LinearConfig::new(SAF_INPUT_DIM, self.hidden_dim).init(device),
            fc2: LinearConfig::new(self.hidden_dim, 1).init(device),
        }
    }
}

impl<B: Backend> SafMlp<B> {
    /// Run SaF features through the MLP.
    pub fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::mish(self.fc1.forward(features));
        self.fc2.forward(h)
    }
}

#[cfg(test)]
mod tests;
