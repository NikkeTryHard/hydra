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

pub const SAF_INPUT_DIM: usize = 8;
const MAX_MIXTURE_ENTROPY: f32 = 1.386_294_4;

#[derive(Debug, Clone, Copy, Default)]
pub struct SafFeatures {
    pub delta_q: f32,
    pub boole_risk: f32,
    pub hunter_risk: f32,
    pub robust_risk: f32,
    pub entropy_drop: f32,
    pub tau_robust: f32,
    pub variance: f32,
    pub ess: f32,
}

impl SafFeatures {
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
pub fn saf_tensor_from_observation<B: Backend>(obs: &[f32], device: &B::Device) -> Tensor<B, 2> {
    let features = saf_features_from_observation(obs);
    let flat: Vec<f32> = features.iter().flat_map(|f| f.to_array()).collect();
    Tensor::<B, 1>::from_floats(flat.as_slice(), device)
        .reshape([HYDRA_ACTION_SPACE, SAF_INPUT_DIM])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafTrainingMode {
    SupervisedRegression,
    JointEndToEnd,
}

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
    pub fn summary(&self) -> String {
        format!(
            "saf(alpha={:.1}, drop={:.1}, dim={})",
            self.alpha, self.dropout, self.hidden_dim
        )
    }

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

pub fn apply_saf_logit<B: Backend>(
    base_logits: Tensor<B, 2>,
    saf_output: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    alpha: f32,
) -> Tensor<B, 2> {
    base_logits + saf_output * mask * alpha
}

#[derive(Config, Debug)]
pub struct SafConfig {
    #[config(default = "1.0")]
    pub alpha: f32,
    #[config(default = "0.3")]
    pub dropout: f32,
    #[config(default = "32")]
    pub hidden_dim: usize,
}

#[derive(Module, Debug)]
pub struct SafMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl SafConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SafMlp<B> {
        SafMlp {
            fc1: LinearConfig::new(SAF_INPUT_DIM, self.hidden_dim).init(device),
            fc2: LinearConfig::new(self.hidden_dim, 1).init(device),
        }
    }
}

impl<B: Backend> SafMlp<B> {
    pub fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = activation::mish(self.fc1.forward(features));
        self.fc2.forward(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use hydra_core::encoder::{
        NUM_CHANNELS, NUM_TILES, SEARCH_DELTA_Q_CHANNEL, SEARCH_MASK_CHANNEL_START,
        SEARCH_MIXTURE_ENTROPY_CHANNEL, SEARCH_MIXTURE_ESS_CHANNEL, SEARCH_RISK_CHANNEL_START,
        SEARCH_STRESS_CHANNEL_START,
    };

    type B = NdArray<f32>;

    fn set_obs(obs: &mut [f32], channel: usize, tile: usize, value: f32) {
        obs[channel * NUM_TILES + tile] = value;
    }

    #[test]
    fn saf_mlp_shape() {
        let device = Default::default();
        let mlp = SafConfig::new().init::<B>(&device);
        let x = Tensor::<B, 2>::zeros([4, SAF_INPUT_DIM], &device);
        let out = mlp.forward(x);
        assert_eq!(out.dims(), [4, 1]);
    }

    #[test]
    fn saf_logit_addition_correct() {
        let device = Default::default();
        let base = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
        let saf_out = Tensor::<B, 2>::from_floats([[0.5, 0.5, 0.5]], &device);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0, 0.0]], &device);
        let result = apply_saf_logit(base, saf_out, mask, 2.0);
        let data = result.to_data();
        let vals = data.as_slice::<f32>().expect("f32");
        assert!((vals[0] - 2.0).abs() < 1e-5);
        assert!((vals[1] - 3.0).abs() < 1e-5);
        assert!((vals[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn saf_zero_mask_is_noop() {
        let device = Default::default();
        let base = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &device);
        let saf_out = Tensor::<B, 2>::from_floats([[10.0, 10.0]], &device);
        let mask = Tensor::<B, 2>::zeros([1, 2], &device);
        let result = apply_saf_logit(base.clone(), saf_out, mask, 1.0);
        let b = base.to_data().as_slice::<f32>().expect("f32").to_vec();
        let r = result.to_data().as_slice::<f32>().expect("f32").to_vec();
        assert_eq!(b, r, "zero mask should produce identical logits");
    }

    #[test]
    fn saf_features_to_array_roundtrip() {
        let f = SafFeatures {
            delta_q: 0.1,
            boole_risk: 0.2,
            hunter_risk: 0.3,
            robust_risk: 0.4,
            entropy_drop: 0.5,
            tau_robust: 0.6,
            variance: 0.7,
            ess: 0.8,
        };
        let arr = f.to_array();
        assert_eq!(arr.len(), SAF_INPUT_DIM);
        assert!((arr[0] - 0.1).abs() < 1e-6);
        assert!((arr[7] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn saf_features_decode_from_observation_planes() {
        let mut obs = vec![0.0f32; NUM_CHANNELS * NUM_TILES];
        set_obs(&mut obs, SEARCH_MASK_CHANNEL_START, 0, 1.0);
        set_obs(&mut obs, SEARCH_MASK_CHANNEL_START + 1, 0, 1.0);
        set_obs(&mut obs, SEARCH_MASK_CHANNEL_START + 2, 0, 1.0);
        set_obs(&mut obs, SEARCH_MIXTURE_ENTROPY_CHANNEL, 0, 0.5);
        set_obs(&mut obs, SEARCH_MIXTURE_ESS_CHANNEL, 0, 2.0);
        set_obs(&mut obs, SEARCH_DELTA_Q_CHANNEL, 5, 0.25);
        set_obs(&mut obs, SEARCH_RISK_CHANNEL_START, 5, 0.1);
        set_obs(&mut obs, SEARCH_RISK_CHANNEL_START + 1, 5, 0.4);
        set_obs(&mut obs, SEARCH_RISK_CHANNEL_START + 2, 5, 0.2);
        set_obs(&mut obs, SEARCH_STRESS_CHANNEL_START, 0, 0.3);
        set_obs(&mut obs, SEARCH_STRESS_CHANNEL_START + 1, 0, 0.8);
        set_obs(&mut obs, SEARCH_STRESS_CHANNEL_START + 2, 0, 0.1);

        let features = saf_features_from_observation(&obs);
        let f = features[5];
        assert!((f.delta_q - 0.25).abs() < 1e-6);
        assert!((f.boole_risk - 0.4).abs() < 1e-6);
        assert!((f.hunter_risk - (0.1 + 0.4 + 0.2) / 3.0).abs() < 1e-6);
        assert!((f.robust_risk - 0.32).abs() < 1e-6);
        assert!((f.tau_robust - 0.8).abs() < 1e-6);
        assert!(f.entropy_drop > 0.0 && f.entropy_drop < 1.0);
        assert!((f.ess - 0.5).abs() < 1e-6);
    }

    #[test]
    fn saf_features_stay_zero_without_presence_masks() {
        let obs = vec![0.0f32; NUM_CHANNELS * NUM_TILES];
        let features = saf_features_from_observation(&obs);
        assert!(
            features
                .iter()
                .all(|f| f.to_array().iter().all(|&v| v == 0.0))
        );
    }

    #[test]
    fn saf_tensor_from_observation_has_expected_shape() {
        let device = Default::default();
        let mut obs = vec![0.0f32; NUM_CHANNELS * NUM_TILES];
        set_obs(&mut obs, SEARCH_MASK_CHANNEL_START + 1, 0, 1.0);
        set_obs(&mut obs, SEARCH_DELTA_Q_CHANNEL, 2, 0.5);
        let tensor = saf_tensor_from_observation::<B>(&obs, &device);
        assert_eq!(tensor.dims(), [HYDRA_ACTION_SPACE, SAF_INPUT_DIM]);
        let data = tensor.to_data();
        let vals = data.as_slice::<f32>().expect("f32");
        assert!((vals[2 * SAF_INPUT_DIM] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn saf_dropout_zeros_features_at_rate() {
        let rng_vals: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let mask = saf_dropout_mask(100, 0.3, &rng_vals);
        let zeros: usize = mask.iter().filter(|&&v| v == 0.0).count();
        let ones: usize = mask.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(zeros + ones, 100, "mask should be binary");
        assert_eq!(zeros, 30, "30% should be dropped with uniform rng_vals");
    }

    #[test]
    fn saf_mlp_output_finite() {
        let device = Default::default();
        let mlp = SafConfig::new().init::<B>(&device);
        let x = Tensor::<B, 2>::random(
            [8, SAF_INPUT_DIM],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = mlp.forward(x);
        let data = out.to_data();
        for &v in data.as_slice::<f32>().expect("f32") {
            assert!(v.is_finite(), "SaF output should be finite: {v}");
        }
    }
}
