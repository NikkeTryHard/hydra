use super::*;
use burn::backend::Autodiff;
use burn::backend::LibTorch;
use burn::backend::NdArray;
use burn::tensor::bf16;

type B = NdArray<f32>;
type AB = Autodiff<NdArray<f32>>;

fn assert_output_shapes(out: &HydraOutput<B>, batch: usize) {
    assert_eq!(out.policy_logits.dims(), [batch, 46]);
    assert_eq!(out.value.dims(), [batch, 1]);
    assert_eq!(out.score_pdf.dims(), [batch, 64]);
    assert_eq!(out.score_cdf.dims(), [batch, 64]);
    assert_eq!(out.opp_tenpai.dims(), [batch, 3]);
    assert_eq!(out.grp.dims(), [batch, 24]);
    assert_eq!(out.opp_next_discard.dims(), [batch, 3, 34]);
    assert_eq!(out.danger.dims(), [batch, 3, 34]);
    assert_eq!(out.oracle_critic.dims(), [batch, 4]);
    assert_eq!(out.belief_fields.dims(), [batch, 16, 34]);
    assert_eq!(out.mixture_weight_logits.dims(), [batch, 4]);
    assert_eq!(out.opponent_hand_type.dims(), [batch, 24]);
    assert_eq!(out.delta_q.dims(), [batch, 46]);
    assert_eq!(out.safety_residual.dims(), [batch, 46]);
}

#[test]
fn actor_net_all_output_shapes() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<B>(&device);
    let x = Tensor::<B, 3>::zeros([4, NUM_CHANNELS, 34], &device);
    let out = model.forward(x);
    assert_output_shapes(&out, 4);
}

#[test]
fn learner_net_all_output_shapes() {
    let device = Default::default();
    let model = HydraModelConfig::learner().init::<B>(&device);
    let x = Tensor::<B, 3>::zeros([2, NUM_CHANNELS, 34], &device);
    let out = model.forward(x);
    assert_output_shapes(&out, 2);
}

#[test]
fn value_head_bounded() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<B>(&device);
    let x = Tensor::<B, 3>::random(
        [4, NUM_CHANNELS, 34],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let out = model.forward(x);
    let data = out.value.to_data();
    for &v in data.as_slice::<f32>().expect("f32") {
        assert!((-1.0..=1.0).contains(&v), "value {v} out of [-1,1]");
    }
}

#[test]
fn policy_value_cpu_returns_correct_shapes() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<B>(&device);
    let obs = [0.0f32; OBS_SIZE];
    let (logits, value) = model.policy_value_cpu(&obs, &device);
    assert_eq!(logits.len(), HYDRA_ACTION_SPACE);
    assert!(value.is_finite());
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn policy_and_value_cpu_matches_policy_value_cpu() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<B>(&device);
    let obs = [0.125f32; OBS_SIZE];

    let direct = model.policy_value_cpu(&obs, &device);
    let via_helper = model.policy_and_value_cpu(&obs, &device);

    assert_eq!(direct.0, via_helper.0);
    assert!((direct.1 - via_helper.1).abs() < 1e-6);
}

#[test]
fn forward_policy_matches_forward_policy_value_logits() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<B>(&device);
    let x = Tensor::<B, 3>::zeros([2, NUM_CHANNELS, 34], &device);

    let policy_only = model.forward_policy(x.clone());
    let (policy_logits, _) = model.forward_policy_value(x);

    let policy_only_data = policy_only
        .to_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("policy-only logits should be readable")
        .to_vec();
    let policy_value_data = policy_logits
        .to_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("policy-value logits should be readable")
        .to_vec();

    assert_eq!(policy_only_data, policy_value_data);
}

#[test]
fn batch_policy_value_cpu_matches_single_sample_path() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<B>(&device);
    let obs_a = [0.0f32; OBS_SIZE];
    let obs_b = [0.25f32; OBS_SIZE];
    let observations = [obs_a, obs_b];

    let single_outputs: Vec<_> = observations
        .iter()
        .map(|obs| model.policy_value_cpu(obs, &device))
        .collect();
    let batch_outputs = model.batch_policy_value_cpu(&observations, &device);

    assert_eq!(batch_outputs.len(), single_outputs.len());
    for ((batch_logits, batch_value), (single_logits, single_value)) in
        batch_outputs.iter().zip(single_outputs.iter())
    {
        for (batch, single) in batch_logits.iter().zip(single_logits.iter()) {
            assert!((batch - single).abs() < 1e-6);
        }
        assert!((batch_value - single_value).abs() < 1e-6);
    }
}

#[test]
fn batch_policy_value_cpu_reuse_matches_non_reuse_path() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<B>(&device);
    let obs_a = [0.1f32; OBS_SIZE];
    let obs_b = [0.2f32; OBS_SIZE];
    let obs_c = [0.3f32; OBS_SIZE];
    let observations = [obs_a, obs_b, obs_c];

    let expected = model.batch_policy_value_cpu(&observations, &device);
    let mut flat_buf = vec![42.0f32; 17];
    let mut outputs_buf = Vec::new();
    let reused =
        model.batch_policy_value_cpu_reuse(&observations, &device, &mut flat_buf, &mut outputs_buf);

    assert_eq!(reused.len(), expected.len());
    for ((reuse_logits, reuse_value), (expected_logits, expected_value)) in
        reused.iter().zip(expected.iter())
    {
        for (reuse, expected) in reuse_logits.iter().zip(expected_logits.iter()) {
            assert!((reuse - expected).abs() < 1e-6);
        }
        assert!((reuse_value - expected_value).abs() < 1e-6);
    }
}

#[test]
fn batch_value_cpu_reuse_matches_policy_value_values_on_dirty_buffer() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<B>(&device);
    let observations = [
        [0.05f32; OBS_SIZE],
        [0.15f32; OBS_SIZE],
        [0.25f32; OBS_SIZE],
    ];

    let expected = model.batch_policy_value_cpu(&observations, &device);
    let mut flat_buf = vec![13.0f32; 29];
    let mut values_buf = Vec::new();
    let values =
        model.batch_value_cpu_reuse(&observations, &device, &mut flat_buf, &mut values_buf);

    assert_eq!(values.len(), expected.len());
    for (value, (_, expected_value)) in values.iter().zip(expected.iter()) {
        assert!((value - expected_value).abs() < 1e-6);
    }
}

#[test]
fn batch_value_cpu_reuse_supports_libtorch_bf16_backend() {
    type Bf16Backend = LibTorch<bf16>;

    let tiny_model_config = HydraModelConfig::new(1)
        .with_input_channels(NUM_CHANNELS)
        .with_hidden_channels(4)
        .with_num_groups(4)
        .with_se_bottleneck(1);

    let device = burn::backend::libtorch::LibTorchDevice::Cpu;
    let model = tiny_model_config.init::<Bf16Backend>(&device);
    let observations = [[0.05f32; OBS_SIZE]];
    let mut flat_buf = vec![7.0f32; 11];
    let mut values_buf = Vec::new();

    let values =
        model.batch_value_cpu_reuse(&observations, &device, &mut flat_buf, &mut values_buf);
    assert_eq!(values.len(), observations.len());
    assert!(values.iter().all(|value| value.is_finite()));

    let outputs = model.batch_policy_value_cpu(&observations, &device);
    for (value, (_, expected_value)) in values.iter().zip(outputs.iter()) {
        assert!((value - expected_value).abs() < 1e-4);
    }
}

#[test]
fn actor_and_learner_param_counts_differ() {
    let device = Default::default();
    let actor = HydraModelConfig::actor().init::<B>(&device);
    let learner = HydraModelConfig::learner().init::<B>(&device);
    let a_params = actor.num_params();
    let l_params = learner.num_params();
    assert!(
        l_params > a_params,
        "learner ({l_params}) should have more params than actor ({a_params})"
    );
    assert!(
        a_params > 1_000_000,
        "actor should have >1M params, got {a_params}"
    );
    assert!(
        l_params > 5_000_000,
        "learner should have >5M params, got {l_params}"
    );
}

#[test]
fn all_outputs_finite_for_random_input() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<B>(&device);
    let x = Tensor::<B, 3>::random(
        [8, NUM_CHANNELS, 34],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let out = model.forward(x);
    let check = |t: &Tensor<B, 2>, name: &str| {
        let d = t.to_data();
        for &v in d.as_slice::<f32>().expect("f32") {
            assert!(v.is_finite(), "{name} has non-finite: {v}");
        }
    };
    let check_spatial = |t: &Tensor<B, 3>, name: &str| {
        let d = t.to_data();
        for &v in d.as_slice::<f32>().expect("f32") {
            assert!(v.is_finite(), "{name} has non-finite: {v}");
        }
    };
    check(&out.policy_logits, "policy");
    check(&out.value, "value");
    check(&out.score_pdf, "score_pdf");
    check(&out.score_cdf, "score_cdf");
    check(&out.opp_tenpai, "opp_tenpai");
    check(&out.grp, "grp");
    check(&out.oracle_critic, "oracle_critic");
    check_spatial(&out.opp_next_discard, "opp_next_discard");
    check_spatial(&out.danger, "danger");
    check_spatial(&out.belief_fields, "belief_fields");
    check(&out.mixture_weight_logits, "mixture_weight_logits");
    check(&out.opponent_hand_type, "opponent_hand_type");
    check(&out.delta_q, "delta_q");
    check(&out.safety_residual, "safety_residual");
}

#[test]
fn oracle_head_does_not_backprop_to_backbone_input() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<AB>(&device);
    let x = Tensor::<AB, 3>::zeros([2, NUM_CHANNELS, 34], &device).require_grad();
    let out = model.forward(x.clone());
    let target = Tensor::<AB, 2>::ones([2, 4], &device);
    let diff = out.oracle_critic - target;
    let loss = (diff.clone() * diff).mean();
    let grads = loss.backward();

    assert!(
        x.grad(&grads).is_none(),
        "oracle-only loss must not backpropagate through the shared backbone"
    );
}

#[test]
fn delta_q_warmup_detaches_backbone_input() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<AB>(&device);
    let x = Tensor::<AB, 3>::zeros([2, NUM_CHANNELS, 34], &device).require_grad();
    let policy = HydraForwardPolicy {
        w_delta_q: 1.0,
        ..Default::default()
    };
    let out = model.forward_with_warmup(x.clone(), &policy, &[ModelAdvancedHead::DeltaQ]);
    let target = Tensor::<AB, 2>::ones([2, HYDRA_ACTION_SPACE], &device);
    let diff = out.delta_q - target;
    let loss = (diff.clone() * diff).mean();
    let grads = loss.backward();

    assert!(
        x.grad(&grads).is_none(),
        "delta_q warmup loss must not backpropagate through the shared backbone"
    );
}

#[test]
fn active_delta_q_backprops_to_backbone_input() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<AB>(&device);
    let x = Tensor::<AB, 3>::zeros([2, NUM_CHANNELS, 34], &device).require_grad();
    let policy = HydraForwardPolicy {
        w_delta_q: 1.0,
        ..Default::default()
    };
    let out = model.forward_active(x.clone(), &policy);
    let target = Tensor::<AB, 2>::ones([2, HYDRA_ACTION_SPACE], &device);
    let diff = out.delta_q - target;
    let loss = (diff.clone() * diff).mean();
    let grads = loss.backward();

    assert!(
        x.grad(&grads).is_some(),
        "active delta_q loss should backpropagate through the shared backbone"
    );
}

#[test]
fn inactive_advanced_heads_return_zero_tensors() {
    let device = Default::default();
    let model = HydraModelConfig::actor().init::<B>(&device);
    let x = Tensor::<B, 3>::zeros([2, NUM_CHANNELS, 34], &device);
    let policy = HydraForwardPolicy::default();
    let out = model.forward_active(x, &policy);

    for &value in out.oracle_critic.to_data().as_slice::<f32>().expect("f32") {
        assert_eq!(value, 0.0);
    }
    for &value in out.belief_fields.to_data().as_slice::<f32>().expect("f32") {
        assert_eq!(value, 0.0);
    }
    for &value in out
        .mixture_weight_logits
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
    {
        assert_eq!(value, 0.0);
    }
    for &value in out
        .opponent_hand_type
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
    {
        assert_eq!(value, 0.0);
    }
    for &value in out.delta_q.to_data().as_slice::<f32>().expect("f32") {
        assert_eq!(value, 0.0);
    }
    for &value in out
        .safety_residual
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
    {
        assert_eq!(value, 0.0);
    }
}

#[test]
fn model_config_actor_learner_defaults() {
    let actor = HydraModelConfig::actor();
    assert_eq!(actor.num_blocks, 12);
    assert_eq!(actor.hidden_channels, 256);
    assert_eq!(actor.num_groups, 32);
    let learner = HydraModelConfig::learner();
    assert_eq!(learner.num_blocks, 24);
    assert_eq!(learner.hidden_channels, 256);
}

#[test]
fn validate_passes_for_standard_configs() {
    assert!(HydraModelConfig::actor().validate().is_ok());
    assert!(HydraModelConfig::learner().validate().is_ok());
}
