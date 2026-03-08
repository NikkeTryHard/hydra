use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::afbs::AfbsTree;
use hydra_core::ct_smc::{CtSmc, CtSmcConfig};
use hydra_core::encoder::NUM_CHANNELS;
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
        [4, NUM_CHANNELS, 34],
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
        &learner_model.forward(Tensor::zeros([4, NUM_CHANNELS, 34], &device)),
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
        rng_seed: 42,
    };
    let mut smc = CtSmc::new(cfg);
    smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
    for p in &smc.particles {
        for (k, &expected) in row_sums.iter().enumerate() {
            let rs: u8 = p.allocation[k].iter().sum();
            assert_eq!(rs, expected, "particle row {k}");
        }
    }

    use hydra_train::training::distill;
    let learner = HydraModelConfig::learner().init::<InferBackend>(&device);
    let actor_for_distill = HydraModelConfig::actor().init::<InferBackend>(&device);
    let x_distill = Tensor::<InferBackend, 3>::zeros([2, NUM_CHANNELS, 34], &device);
    let l_out = learner.forward(x_distill.clone());
    let a_out = actor_for_distill.forward(x_distill);
    let mask = Tensor::<InferBackend, 2>::ones([2, 46], &device);
    let d_loss = distill::distill_loss(
        l_out.policy_logits,
        a_out.policy_logits,
        l_out.value,
        a_out.value,
        mask,
        1.0,
        0.5,
    );
    let d_val: f32 = d_loss.into_scalar().elem();
    assert!(d_val.is_finite(), "distill loss not finite: {d_val}");

    use hydra_core::arena::{Arena, ArenaConfig, Trajectory, TrajectoryStep};
    use hydra_core::encoder::OBS_SIZE;
    let mut arena = Arena::new(ArenaConfig::default());
    for g in 0..10u32 {
        let mut traj = Trajectory::new(g, g as u64 * 42);
        for turn in 0..5u16 {
            traj.steps.push(TrajectoryStep {
                obs: [0.1; OBS_SIZE],
                action: (turn % 34) as u8,
                pi_old: {
                    let mut p = [0.0; HYDRA_ACTION_SPACE];
                    p[0] = 1.0;
                    p
                },
                reward: 0.0,
                done: turn == 4,
                player_id: (turn % 4) as u8,
                game_id: g,
                turn,
                temperature: 1.0,
            });
        }
        traj.final_scores = [25000; 4];
        arena.add_trajectory(traj);
    }
    assert_eq!(arena.games_completed, 10);
    assert!(arena.total_steps() >= 50, "should have 50+ steps");

    use hydra_core::afbs::{AfbsTree, TOP_K};
    let mut tree = AfbsTree::new();
    let root = tree.add_node(0, 1.0, false);
    let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
    for (i, v) in logits[..10].iter_mut().enumerate() {
        *v = (10 - i) as f32;
    }
    let mut mask = [false; HYDRA_ACTION_SPACE];
    for v in &mut mask[..10] {
        *v = true;
    }
    tree.expand_node(root, &logits, &mask, false);
    assert_eq!(tree.nodes[root as usize].children.len(), TOP_K);
    for _ in 0..8 {
        if let Some((_, child)) = tree.puct_select(root) {
            tree.backpropagate(&[root, child], 0.5);
        }
    }
    assert!(
        tree.nodes[root as usize].visit_count >= 8,
        "AFBS visit_count < 8"
    );
    let search_exit = tree.root_exit_policy(root, 1.0);
    let search_sum: f32 = search_exit.iter().sum();
    assert!(
        (search_sum - 1.0).abs() < 0.01,
        "search exit policy sum: {search_sum}"
    );

    let exit_q = vec![1.0, 3.0, 2.0, 0.5];
    let exit_pi = exit::exit_policy_from_q(&exit_q, 1.0, None);
    let sum: f32 = exit_pi.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "exit policy sum: {sum}");

    let actor2 = HydraModelConfig::actor().init::<InferBackend>(&device);
    let x2 = Tensor::<InferBackend, 3>::random(
        [1, NUM_CHANNELS, 34],
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

#[test]
fn edge_case_smoke_test() {
    use hydra_core::ct_smc::{CtSmc, CtSmcConfig, Particle};
    use hydra_core::endgame;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut row_sums = [0u8; 34];
    row_sums[0] = 2;
    row_sums[1] = 2;
    row_sums[2] = 1;
    let col_sums = [2, 2, 1, 0];
    let log_omega = [[0.0f64; 4]; 34];
    let cfg = CtSmcConfig {
        num_particles: 32,
        ess_threshold: 0.4,
        rng_seed: 42,
    };
    let mut smc = CtSmc::new(cfg);
    smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);

    let mut mask = [false; HYDRA_ACTION_SPACE];
    mask[0] = true;
    mask[1] = true;
    mask[2] = true;
    let eval_fn = |_: &Particle, a: u8| (a as f32 + 1.0) * 0.1;
    let q = endgame::pimc_endgame_q(&smc.particles, &mask, &eval_fn);
    for (i, &v) in q.iter().enumerate() {
        if mask[i] {
            assert!(v.is_finite(), "endgame q[{i}] = {v} not finite");
        }
    }
}

#[test]
fn determinism_same_seed_same_particles() {
    use hydra_core::ct_smc::{CtSmc, CtSmcConfig};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut row_sums = [0u8; 34];
    row_sums[0] = 2;
    row_sums[1] = 1;
    row_sums[2] = 1;
    let col_sums = [1, 1, 1, 1];
    let log_omega = [[0.0f64; 4]; 34];
    let cfg = CtSmcConfig {
        num_particles: 16,
        ess_threshold: 0.4,
        rng_seed: 42,
    };

    let mut rng1 = ChaCha8Rng::seed_from_u64(42);
    let mut smc1 = CtSmc::new(cfg);
    smc1.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng1);

    let cfg2 = CtSmcConfig {
        num_particles: 16,
        ess_threshold: 0.4,
        rng_seed: 42,
    };
    let mut rng2 = ChaCha8Rng::seed_from_u64(42);
    let mut smc2 = CtSmc::new(cfg2);
    smc2.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng2);

    for i in 0..16 {
        assert_eq!(
            smc1.particles[i].allocation, smc2.particles[i].allocation,
            "particle {i} differs between runs with same seed"
        );
    }
}

#[test]
fn ach_rl_step_integration() {
    use hydra_train::training::rl::{RlBatch, RlConfig};

    let device = Default::default();
    let batch = 2;

    let model = HydraModelConfig::actor().init::<TestBackend>(&device);
    let targets = make_test_targets_infer(&device, batch);
    let obs = Tensor::<TestBackend, 3>::zeros([batch, NUM_CHANNELS, 34], &device);
    let base_logits = Tensor::<TestBackend, 2>::zeros([batch, 46], &device);
    let actions = Tensor::<TestBackend, 1, Int>::from_ints(&[1i32, 2][..], &device);
    let pi_old = Tensor::<TestBackend, 1>::from_floats([0.5, 0.3], &device);
    let advantages = Tensor::<TestBackend, 1>::from_floats([1.0, -0.5], &device);

    let rl_batch = RlBatch {
        obs,
        targets,
        base_logits,
        actions,
        pi_old,
        advantages,
        exit_target: None,
        exit_mask: None,
    };
    let loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
    let cfg = RlConfig::default_phase2();
    let mut optim = AdamConfig::new().init();
    let (_, loss) =
        hydra_train::training::rl::rl_step(model, &rl_batch, &cfg, &loss_fn, &mut optim);
    assert!(loss.is_finite(), "rl_step loss not finite: {loss}");
}

#[test]
fn exit_rl_step_with_target() {
    use hydra_train::training::exit::{build_exit_from_afbs_tree, collate_exit_targets};
    use hydra_train::training::rl::{RlBatch, RlConfig};

    let device = Default::default();
    let batch = 2;

    let model = HydraModelConfig::actor().init::<TestBackend>(&device);
    let targets = make_test_targets_infer(&device, batch);
    let obs = Tensor::<TestBackend, 3>::zeros([batch, NUM_CHANNELS, 34], &device);
    let base_logits = Tensor::<TestBackend, 2>::zeros([batch, 46], &device);
    let actions = Tensor::<TestBackend, 1, Int>::from_ints(&[1i32, 2][..], &device);
    let pi_old = Tensor::<TestBackend, 1>::from_floats([0.5, 0.3], &device);
    let advantages = Tensor::<TestBackend, 1>::from_floats([1.0, -0.5], &device);

    let mut tree = AfbsTree::new();
    let root = tree.add_node(7, 1.0, false);
    let mut legal_mask = [false; HYDRA_ACTION_SPACE];
    legal_mask[1] = true;
    legal_mask[2] = true;
    legal_mask[5] = true;
    let mut policy_logits = [0.0f32; HYDRA_ACTION_SPACE];
    policy_logits[1] = 3.0;
    policy_logits[2] = 2.0;
    policy_logits[5] = 1.0;
    tree.expand_node(root, &policy_logits, &legal_mask, false);

    let children = tree.nodes[root as usize].children.clone();
    for &(action, child) in &children {
        let node = &mut tree.nodes[child as usize];
        match action {
            1 => {
                node.visit_count = 10;
                node.total_value = 9.0;
            }
            2 => {
                node.visit_count = 8;
                node.total_value = 4.0;
            }
            5 => {
                node.visit_count = 6;
                node.total_value = 0.6;
            }
            _ => unreachable!("unexpected action in expanded root"),
        }
    }
    tree.nodes[root as usize].visit_count = 24;

    let mut base_pi = vec![1e-6f32; 46];
    base_pi[1] = 0.45;
    base_pi[2] = 0.35;
    base_pi[5] = 0.20;
    let legal_f32 = legal_mask.map(|x| if x { 1.0f32 } else { 0.0 });

    let accepted = build_exit_from_afbs_tree(&tree, root, &base_pi, &legal_f32, 8, 5.0)
        .expect("accepted exit target from child visits");

    let samples = vec![Some(accepted.clone()), Some(accepted)];
    let (exit_target, exit_mask) = collate_exit_targets::<TestBackend>(&samples, &device);

    let rl_batch = RlBatch {
        obs,
        targets,
        base_logits,
        actions,
        pi_old,
        advantages,
        exit_target,
        exit_mask,
    };
    let loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
    let cfg = RlConfig::default_phase3();
    let mut optim = AdamConfig::new().init();
    let (_, loss) =
        hydra_train::training::rl::rl_step(model, &rl_batch, &cfg, &loss_fn, &mut optim);
    assert!(loss.is_finite(), "exit rl_step loss not finite: {loss}");
}

#[test]
fn ctsmc_to_endgame_to_exit_pipeline() {
    use hydra_core::ct_smc::{CtSmc, CtSmcConfig, Particle};
    use hydra_core::endgame;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let mut row_sums = [0u8; 34];
    row_sums[0] = 3;
    row_sums[1] = 2;
    let col_sums = [2, 1, 1, 1];
    let log_omega = [[0.0f64; 4]; 34];
    let cfg = CtSmcConfig {
        num_particles: 64,
        ess_threshold: 0.4,
        rng_seed: 99,
    };
    let mut smc = CtSmc::new(cfg);
    smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
    assert!(!smc.particles.is_empty(), "should have particles");

    let mut mask = [false; HYDRA_ACTION_SPACE];
    mask[0] = true;
    mask[1] = true;
    let eval_fn = |p: &Particle, a: u8| p.allocation[a as usize][0] as f32;
    let q = endgame::pimc_endgame_q(&smc.particles, &mask, &eval_fn);
    let legal = [true, true, false, false];
    let exit_pi = exit::exit_policy_from_q(&q[..4], 1.0, Some(&legal));
    let sum: f32 = exit_pi.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "exit policy sum: {sum}");
    assert!(exit_pi[2] == 0.0, "illegal action should have 0 prob");
    assert!(exit_pi[3] == 0.0, "illegal action should have 0 prob");
}

fn make_test_targets_infer(
    device: &<TestBackend as Backend>::Device,
    batch: usize,
) -> HydraTargets<TestBackend> {
    make_test_targets(device, batch)
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
        belief_fields_target: None,
        belief_fields_mask: None,
        mixture_weight_target: None,
        mixture_weight_mask: None,
        opponent_hand_type_target: None,
        delta_q_target: None,
        safety_residual_target: None,
        safety_residual_mask: None,
        oracle_guidance_mask: None,
    }
}
