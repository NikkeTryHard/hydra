use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::ct_smc::{CtSmc, CtSmcConfig};
use hydra_train::inference;
use hydra_train::model::HydraModelConfig;
use hydra_train::training::drda;
use hydra_train::training::exit;
use hydra_train::training::gae;
use hydra_train::training::losses::*;

type TestBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

fn no_nan_2d<B: Backend>(t: &Tensor<B, 2>, name: &str) {
    let data = t.to_data();
    let slice = data.as_slice::<f32>().expect("f32");
    for (i, &v) in slice.iter().enumerate() {
        assert!(v.is_finite(), "{name}[{i}] = {v} (not finite)");
    }
}

#[test]
fn full_pipeline_integration() {
    let device = Default::default();

    let actor_model = HydraModelConfig::actor().init::<InferBackend>(&device);
    let x = Tensor::<InferBackend, 3>::random(
        [4, 85, 34],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );
    let out = actor_model.forward(x);
    no_nan_2d(&out.policy_logits, "actor_policy");
    no_nan_2d(&out.value, "actor_value");

    let learner_model = HydraModelConfig::learner().init::<TestBackend>(&device);
    let targets = make_test_targets(&device, 4);
    let loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
    let breakdown = loss_fn.total_loss(
        &learner_model.forward(Tensor::zeros([4, 85, 34], &device)),
        &targets,
    );
    let loss_val: f64 = breakdown.total.clone().into_scalar().elem();
    assert!(loss_val.is_finite(), "total loss not finite: {loss_val}");
    assert!(loss_val > 0.0, "total loss not positive: {loss_val}");
    let grads = breakdown.total.backward();
    let grads = GradientsParams::from_grads(grads, &learner_model);
    let mut optim = AdamConfig::new().init();
    let _learner_model = optim.step(1e-4, learner_model, grads);

    let rewards = vec![0.5, -0.2, 1.0, 0.0, -0.5];
    let values = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.0];
    let dones = vec![false, false, false, false, true];
    let (mut adv, _ret) = gae::compute_gae(&rewards, &values, &dones, 0.995, 0.95);
    gae::normalize_advantages(&mut adv);
    let mean: f32 = adv.iter().sum::<f32>() / adv.len() as f32;
    assert!(mean.abs() < 1e-5, "GAE mean not ~0: {mean}");

    let base = Tensor::<InferBackend, 2>::from_floats([[1.0, 2.0, 3.0]], &device);
    let residual = Tensor::<InferBackend, 2>::from_floats([[0.5, 1.0, 1.5]], &device);
    let combined = drda::combined_logits(base, residual, 4.0);
    let data = combined.to_data();
    let vals = data.as_slice::<f32>().expect("f32");
    assert!((vals[0] - 1.125).abs() < 1e-4);

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut row_sums = [0u8; 34];
    row_sums[0] = 2;
    row_sums[1] = 1;
    row_sums[2] = 1;
    let col_sums = [1, 1, 1, 1];
    let log_omega = [[0.0f64; 4]; 34];
    let cfg = CtSmcConfig {
        num_particles: 32,
        ess_threshold: 0.4,
    };
    let mut smc = CtSmc::new(cfg);
    smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
    for p in &smc.particles {
        for (k, &expected) in row_sums.iter().enumerate() {
            let rs: u8 = p.allocation[k].iter().sum();
            assert_eq!(rs, expected, "particle row {k}");
        }
    }

    let exit_q = vec![1.0, 3.0, 2.0, 0.5];
    let exit_pi = exit::exit_policy_from_q(&exit_q, 1.0);
    let sum: f32 = exit_pi.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "exit policy sum: {sum}");

    let actor2 = HydraModelConfig::actor().init::<InferBackend>(&device);
    let x2 = Tensor::<InferBackend, 3>::random(
        [1, 85, 34],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );
    let out2 = actor2.forward(x2);
    let mut legal = [true; HYDRA_ACTION_SPACE];
    legal[0] = false;
    let (action, policy) = inference::infer_action(out2.policy_logits, &legal);
    assert!(legal[action as usize], "inference picked illegal {action}");
    let psum: f32 = policy.iter().sum();
    assert!((psum - 1.0).abs() < 0.01, "inference policy sum: {psum}");
}

fn make_test_targets(
    device: &<TestBackend as Backend>::Device,
    batch: usize,
) -> HydraTargets<TestBackend> {
    let mut pd = vec![0.0f32; batch * 46];
    for i in 0..batch {
        pd[i * 46 + 1] = 1.0;
    }
    let mut gd = vec![0.0f32; batch * 24];
    for i in 0..batch {
        gd[i * 24] = 1.0;
    }
    let mut od = vec![0.0f32; batch * 3 * 34];
    for i in 0..(batch * 3) {
        od[i * 34] = 1.0;
    }
    let mut sd = vec![0.0f32; batch * 64];
    for i in 0..batch {
        sd[i * 64 + 32] = 1.0;
    }
    HydraTargets {
        policy_target: Tensor::<TestBackend, 1>::from_floats(pd.as_slice(), device)
            .reshape([batch, 46]),
        legal_mask: Tensor::ones([batch, 46], device),
        value_target: Tensor::zeros([batch], device),
        grp_target: Tensor::<TestBackend, 1>::from_floats(gd.as_slice(), device)
            .reshape([batch, 24]),
        tenpai_target: Tensor::zeros([batch, 3], device),
        danger_target: Tensor::zeros([batch, 3, 34], device),
        danger_mask: Tensor::ones([batch, 3, 34], device),
        opp_next_target: Tensor::<TestBackend, 1>::from_floats(od.as_slice(), device)
            .reshape([batch, 3, 34]),
        score_pdf_target: Tensor::<TestBackend, 1>::from_floats(sd.as_slice(), device)
            .reshape([batch, 64]),
        score_cdf_target: Tensor::zeros([batch, 64], device),
        oracle_target: None,
    }
}
