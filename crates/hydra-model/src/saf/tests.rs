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
