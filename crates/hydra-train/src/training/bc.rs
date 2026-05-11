//! Compatibility re-exports for behavioral cloning training execution.

pub use hydra_train_exec::bc_runtime::{
    BcExitConfig, BcTrainBatchInput, BcTrainStepContext, EpochStats, OracleGuidingBatchInput,
    OracleGuidingStepSchedule, OracleGuidingStepStats, bc_total_with_exit,
    bc_total_with_exit_from_breakdown, bc_total_with_optional_exit_from_breakdown, bc_train_step,
    cosine_annealing_lr, gated_bc_context, maybe_add_exit_loss, oracle_guidance_mask_tensor,
    oracle_guidance_mask_values, oracle_guiding_train_step, phase_learning_rate, policy_agreement,
    policy_agreement_counts, target_actions_from_policy_target, train_epoch, warmup_then_cosine_lr,
};
pub use hydra_train_types::checkpoint::CheckpointMeta;
pub use hydra_train_types::config::BCTrainerConfig;

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_train_exec::data::sample::MjaiBcBatch;
    use hydra_train_exec::losses::{HydraLoss, tests::make_dummy_targets};
    use hydra_train_exec::model::{HydraModel, HydraModelConfig, HydraModelInit};
    use hydra_train_types::losses::HydraLossConfig;
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::grad_clipping::GradientClippingConfig;
    use burn::optim::AdamConfig;
    use burn::prelude::*;

    use std::time::{SystemTime, UNIX_EPOCH};
    type TestBackend = Autodiff<NdArray<f32>>;

    fn bc_optimizer() -> impl burn::optim::Optimizer<HydraModel<TestBackend>, TestBackend> {
        AdamConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .init()
    }

    fn unique_checkpoint_base(label: &str) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir()
            .join(format!("hydra-{label}-{}-{nanos}", std::process::id()))
            .to_string_lossy()
            .into_owned()
    }

    fn empty_batch(
        device: &<TestBackend as Backend>::Device,
        batch: usize,
    ) -> MjaiBcBatch<TestBackend> {
        MjaiBcBatch {
            actions: Tensor::zeros([batch], device),
            exit_target: None,
            exit_mask: None,
        }
    }

    #[test]
    fn test_bc_config_defaults() {
        let cfg = BCTrainerConfig::new(HydraModelConfig::actor());
        assert!((cfg.lr - 2.5e-4).abs() < 1e-10);
        assert!((cfg.min_learning_rate - 1e-6).abs() < 1e-12);
        assert_eq!(cfg.batch_size, 2048);
        assert!((cfg.grad_clip_norm - 1.0).abs() < 1e-6);
        assert!((cfg.weight_decay - 1e-5).abs() < 1e-8);
        assert_eq!(cfg.warmup_steps, 1000);
    }

    #[test]
    fn effective_lr_respects_configured_min_learning_rate() {
        let cfg = BCTrainerConfig::new(HydraModelConfig::actor())
            .with_lr(1e-3)
            .with_min_learning_rate(1e-4)
            .with_warmup_steps(10);
        let lr = cfg.effective_lr(1_000, 1_000);
        assert!(
            lr >= 1e-4 - 1e-10,
            "lr floor should respect configured min: {lr}"
        );
    }

    #[test]
    fn test_bc_one_step() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::zeros([4, crate::config::INPUT_CHANNELS, 34], &device);
        let targets = make_dummy_targets::<TestBackend>(&device, 4);
        let loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
        let mut optimizer = bc_optimizer();
        let batch = empty_batch(&device, 4);
        let (_, loss1) = bc_train_step(
            model,
            BcTrainBatchInput {
                obs,
                batch: &batch,
                targets: &targets,
            },
            BcTrainStepContext {
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
                use_amp: false,
                lr: 1e-3,
            },
            &mut optimizer,
        );
        assert!(loss1.is_finite(), "loss should be finite: {loss1}");
        assert!(loss1 > 0.0, "loss should be positive: {loss1}");
    }

    #[test]
    fn test_bc_overfit_10_samples() {
        let device = Default::default();
        let mut model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::random(
            [10, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let targets = make_dummy_targets::<TestBackend>(&device, 10);
        let loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
        let mut optimizer = bc_optimizer();
        let batch = empty_batch(&device, 10);
        let mut last_loss = f64::MAX;
        for _ in 0..25 {
            let (m, loss) = bc_train_step(
                model,
                BcTrainBatchInput {
                    obs: obs.clone(),
                    batch: &batch,
                    targets: &targets,
                },
                BcTrainStepContext {
                    loss_fn: &loss_fn,
                    exit_cfg: &BcExitConfig::default(),
                    use_amp: false,
                    lr: 1e-3,
                },
                &mut optimizer,
            );
            model = m;
            last_loss = loss;
        }
        assert!(last_loss < 10.0, "should overfit: loss={last_loss}");
    }

    #[test]
    fn test_oracle_guidance_mask_values_follow_keep_probability() {
        let mask = oracle_guidance_mask_values(4, 0.5, &[0.1, 0.7, 0.49, 0.9]);
        assert_eq!(mask, vec![1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_oracle_guiding_train_step_skips_large_importance_post_dropout() {
        let device = Default::default();
        let model = HydraModelConfig::actor().init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::zeros([2, crate::config::INPUT_CHANNELS, 34], &device);
        let mut targets = make_dummy_targets::<TestBackend>(&device, 2);
        targets.oracle_target = Some(Tensor::<TestBackend, 2>::ones([2, 4], &device));
        let loss_fn =
            HydraLoss::<TestBackend>::new(HydraLossConfig::new().with_w_oracle_critic(1.0));
        let mut optimizer = bc_optimizer();
        let oracle_cfg = crate::config::OracleGuidingConfig::default();

        let (_, stats) = oracle_guiding_train_step(
            model,
            OracleGuidingBatchInput {
                obs,
                targets: &targets,
                loss_fn: &loss_fn,
                importance_weight: 3.0,
                max_importance_weight: 2.0,
                rng_values: &[0.1, 0.9],
            },
            OracleGuidingStepSchedule {
                base_lr: 1e-4,
                oracle_cfg: &oracle_cfg,
                step: 100,
                total_steps: 100,
            },
            &mut optimizer,
        );

        assert!(stats.skipped);
        assert!(stats.loss.is_none());
        assert!((stats.effective_lr - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn test_oracle_guiding_train_step_applies_dropout_mask_and_trains() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::random(
            [2, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let mut targets = make_dummy_targets::<TestBackend>(&device, 2);
        targets.oracle_target = Some(Tensor::<TestBackend, 2>::ones([2, 4], &device));
        targets.belief_fields_target = Some(Tensor::<TestBackend, 3>::ones([2, 16, 34], &device));
        let loss_fn = HydraLoss::<TestBackend>::new(
            HydraLossConfig::new()
                .with_w_oracle_critic(1.0)
                .with_w_belief_fields(1.0),
        );
        let mut optimizer = bc_optimizer();
        let oracle_cfg = crate::config::OracleGuidingConfig::default();

        let (_, stats) = oracle_guiding_train_step(
            model,
            OracleGuidingBatchInput {
                obs,
                targets: &targets,
                loss_fn: &loss_fn,
                importance_weight: 1.0,
                max_importance_weight: 2.0,
                rng_values: &[0.0, 0.9],
            },
            OracleGuidingStepSchedule {
                base_lr: 1e-4,
                oracle_cfg: &oracle_cfg,
                step: 50,
                total_steps: 100,
            },
            &mut optimizer,
        );

        assert!(!stats.skipped);
        assert!(stats.loss.expect("loss") > 0.0);
        assert!((stats.oracle_keep_prob - 0.5).abs() < 1e-6);
        assert!((stats.kept_oracle_fraction - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_policy_agreement_range() {
        let device: <NdArray<f32> as Backend>::Device = Default::default();
        let model = HydraModelConfig::actor().init::<NdArray<f32>>(&device);
        let x = Tensor::<NdArray<f32>, 3>::random(
            [4, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let output = model.forward(x);
        let mask = Tensor::<NdArray<f32>, 2>::ones([4, 46], &device);
        let targets = Tensor::<NdArray<f32>, 1, Int>::from_ints(&[0i32; 4][..], &device);
        let acc = policy_agreement(output.policy_logits, mask, targets);
        assert!((0.0..=1.0).contains(&acc), "agreement {acc} out of [0,1]");
    }

    #[test]
    fn test_policy_agreement_counts_match_fraction() {
        let device: <NdArray<f32> as Backend>::Device = Default::default();
        let logits = Tensor::<NdArray<f32>, 2>::from_floats(
            [
                [10.0, 1.0, 0.0],
                [0.0, 9.0, 1.0],
                [0.0, 1.0, 8.0],
                [3.0, 2.0, 1.0],
            ],
            &device,
        );
        let mask = Tensor::<NdArray<f32>, 2>::ones([4, 3], &device);
        let targets = Tensor::<NdArray<f32>, 1, Int>::from_ints([0, 1, 1, 2], &device);

        let acc = policy_agreement(logits.clone(), mask.clone(), targets.clone());
        let (correct, total) = policy_agreement_counts(logits, mask, targets);

        assert_eq!((correct, total), (2, 4));
        assert!((acc - correct as f64 / total as f64).abs() < 1e-12);
    }

    #[test]
    fn policy_target_argmax_matches_batch_actions() {
        let device = Default::default();
        let actions = Tensor::<TestBackend, 1, Int>::from_ints(&[0i32, 7, 45][..], &device);
        let mut policy_target = vec![0.0f32; 3 * 46];
        policy_target[0] = 1.0;
        policy_target[46 + 7] = 1.0;
        policy_target[2 * 46 + 45] = 1.0;
        let recovered = target_actions_from_policy_target(
            Tensor::<TestBackend, 1>::from_floats(policy_target.as_slice(), &device)
                .reshape([3, 46]),
        );
        let same = recovered.equal(actions).into_data().convert::<i64>();
        assert_eq!(
            same.as_slice::<i64>().expect("policy action parity"),
            &[1, 1, 1]
        );
    }
    #[test]
    fn test_train_epoch_reports_policy_agreement() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::random(
            [4, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let output = model.forward(obs.clone());
        let predicted = output
            .policy_logits
            .clone()
            .argmax(1)
            .squeeze_dim::<1>(1)
            .to_data();
        let predicted = predicted.as_slice::<i64>().expect("i64");
        let mut policy_target = vec![0.0f32; predicted.len() * 46];
        for (row, &action) in predicted.iter().enumerate() {
            policy_target[row * 46 + action as usize] = 1.0;
        }

        let mut targets = make_dummy_targets::<TestBackend>(&device, predicted.len());
        targets.policy_target =
            Tensor::<TestBackend, 1>::from_floats(policy_target.as_slice(), &device)
                .reshape([predicted.len(), 46]);

        let acc = policy_agreement(
            output.policy_logits,
            targets.legal_mask,
            target_actions_from_policy_target(targets.policy_target),
        );
        assert!(
            (acc - 1.0).abs() < 1e-6,
            "agreement should reflect matching targets, got {acc}"
        );
    }

    #[test]
    fn test_bc_step_advanced_aux_targets_change_loss() {
        let device = Default::default();
        let obs = Tensor::<TestBackend, 3>::random(
            [4, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );

        let model1 = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let baseline_targets = make_dummy_targets::<TestBackend>(&device, 4);
        let baseline_loss_fn = HydraLoss::<TestBackend>::new(HydraLossConfig::new());
        let mut opt1 = bc_optimizer();
        let baseline_batch = empty_batch(&device, 4);
        let (_, loss_baseline) = bc_train_step(
            model1,
            BcTrainBatchInput {
                obs: obs.clone(),
                batch: &baseline_batch,
                targets: &baseline_targets,
            },
            BcTrainStepContext {
                loss_fn: &baseline_loss_fn,
                exit_cfg: &BcExitConfig::default(),
                use_amp: false,
                lr: 1e-3,
            },
            &mut opt1,
        );

        let model2 = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let mut advanced_targets = make_dummy_targets::<TestBackend>(&device, 4);
        advanced_targets.belief_fields_target =
            Some(Tensor::<TestBackend, 3>::ones([4, 16, 34], &device));
        advanced_targets.belief_fields_mask = Some(Tensor::<TestBackend, 1>::ones([4], &device));
        advanced_targets.safety_residual_target = Some(Tensor::<TestBackend, 2>::from_floats(
            [[0.5f32; 46], [-0.5f32; 46], [0.3f32; 46], [-0.3f32; 46]],
            &device,
        ));
        advanced_targets.safety_residual_mask =
            Some(Tensor::<TestBackend, 2>::ones([4, 46], &device));
        advanced_targets.delta_q_target = Some(Tensor::<TestBackend, 2>::ones([4, 46], &device));
        advanced_targets.oracle_target = Some(Tensor::<TestBackend, 2>::from_floats(
            [
                [0.1, -0.1, 0.05, -0.05],
                [0.2, -0.2, 0.1, -0.1],
                [0.05, -0.05, 0.0, 0.0],
                [-0.1, 0.1, -0.05, 0.05],
            ],
            &device,
        ));
        advanced_targets.oracle_guidance_mask = Some(Tensor::<TestBackend, 1>::ones([4], &device));
        let advanced_loss_fn = HydraLoss::<TestBackend>::new(
            HydraLossConfig::new()
                .with_w_oracle_critic(0.1)
                .with_w_belief_fields(0.1)
                .with_w_safety_residual(0.1)
                .with_w_delta_q(0.05),
        );
        let mut opt2 = bc_optimizer();
        let advanced_batch = empty_batch(&device, 4);
        let (_, loss_advanced) = bc_train_step(
            model2,
            BcTrainBatchInput {
                obs,
                batch: &advanced_batch,
                targets: &advanced_targets,
            },
            BcTrainStepContext {
                loss_fn: &advanced_loss_fn,
                exit_cfg: &BcExitConfig::default(),
                use_amp: false,
                lr: 1e-3,
            },
            &mut opt2,
        );

        assert!(
            loss_baseline.is_finite(),
            "baseline BC loss should be finite"
        );
        assert!(
            loss_advanced.is_finite(),
            "advanced BC loss should be finite"
        );
        assert!(
            (loss_baseline - loss_advanced).abs() > 1e-6,
            "advanced aux targets should change BC loss: baseline={loss_baseline}, advanced={loss_advanced}"
        );
    }

    #[test]
    fn test_bc_train_epoch_with_advanced_targets() {
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<TestBackend>(&device);
        let obs = Tensor::<TestBackend, 3>::random(
            [4, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let mut targets = make_dummy_targets::<TestBackend>(&device, 4);
        targets.belief_fields_target = Some(Tensor::<TestBackend, 3>::ones([4, 16, 34], &device));
        targets.belief_fields_mask = Some(Tensor::<TestBackend, 1>::ones([4], &device));
        targets.safety_residual_target =
            Some(Tensor::<TestBackend, 2>::ones([4, 46], &device) * 0.25);
        targets.safety_residual_mask = Some(Tensor::<TestBackend, 2>::ones([4, 46], &device));
        targets.delta_q_target = Some(Tensor::<TestBackend, 2>::zeros([4, 46], &device));

        let loss_fn = HydraLoss::<TestBackend>::new(
            HydraLossConfig::new()
                .with_w_belief_fields(0.1)
                .with_w_safety_residual(0.1)
                .with_w_delta_q(0.05),
        );
        let mut optimizer = bc_optimizer();
        let batch = empty_batch(&device, 4);
        let (_, stats) = bc_train_step(
            model,
            BcTrainBatchInput {
                obs,
                batch: &batch,
                targets: &targets,
            },
            BcTrainStepContext {
                loss_fn: &loss_fn,
                exit_cfg: &BcExitConfig::default(),
                use_amp: false,
                lr: 1e-3,
            },
            &mut optimizer,
        );

        assert!(stats.is_finite(), "loss should be finite");
        assert!(stats > 0.0, "loss should be positive with advanced targets");
    }

    #[test]
    fn test_checkpoint_save_load() {
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        let device = Default::default();
        let model = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<NdArray<f32>>(&device);
        let x = Tensor::<NdArray<f32>, 3>::random(
            [2, crate::config::INPUT_CHANNELS, 34],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &device,
        );
        let out1 = model.forward(x.clone());
        let path = unique_checkpoint_base("test-ckpt");
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model.save_file(&path, &recorder).expect("save failed");
        let loaded = HydraModelConfig::new(2)
            .with_hidden_channels(32)
            .with_se_bottleneck(8)
            .with_num_groups(4)
            .init::<NdArray<f32>>(&device)
            .load_file(&path, &recorder, &device)
            .expect("load failed");
        let out2 = loaded.forward(x);
        let d1 = out1.policy_logits.to_data();
        let d2 = out2.policy_logits.to_data();
        let s1 = d1.as_slice::<f32>().expect("f32");
        let s2 = d2.as_slice::<f32>().expect("f32");
        for (i, (&a, &b)) in s1.iter().zip(s2.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "mismatch at {i}: {a} vs {b}");
        }
        std::fs::remove_file(format!("{path}.mpk")).ok();
    }

    #[test]
    fn checkpoint_meta_summary_handles_missing_eval_metrics() {
        let meta = CheckpointMeta::new(10, 2.5, None, None, None);
        assert_eq!(meta.summary(), "epoch=10 loss=2.5000 eval=n/a");
    }

    #[test]
    fn checkpoint_meta_summary_reports_eval_metrics() {
        let meta = CheckpointMeta::new(10, 2.5, Some(0.375), Some(1.75), Some(2.25));
        assert_eq!(
            meta.summary(),
            "epoch=10 loss=2.5000 policy_ce=1.7500 total=2.2500 agree=37.50%"
        );
    }

    #[test]
    fn checkpoint_meta_old_path_remains_available() {
        let meta: CheckpointMeta = hydra_train_types::checkpoint::CheckpointMeta::new(
            1,
            0.5,
            Some(0.25),
            Some(0.75),
            Some(1.25),
        );
        assert_eq!(
            meta.summary(),
            "epoch=1 loss=0.5000 policy_ce=0.7500 total=1.2500 agree=25.00%"
        );
    }

    #[test]
    fn learning_rate_helpers_cover_zero_total_and_post_warmup_edges() {
        assert!((cosine_annealing_lr(3, 0, 1e-3, 1e-5) - 1e-3).abs() < 1e-12);

        let warmup_lr = warmup_then_cosine_lr(1, 4, 10, 1e-3, 1e-5);
        assert!((warmup_lr - 2.5e-4).abs() < 1e-12);

        let post_warmup_lr = warmup_then_cosine_lr(7, 4, 10, 1e-3, 1e-5);
        let expected = cosine_annealing_lr(3, 6, 1e-3, 1e-5);
        assert!((post_warmup_lr - expected).abs() < 1e-12);
    }

    #[test]
    fn oracle_guidance_mask_tensor_uses_rng_fallback_for_missing_samples() {
        let device = Default::default();
        let mask = oracle_guidance_mask_tensor::<TestBackend>(3, 0.5, &[0.25], &device)
            .to_data()
            .as_slice::<f32>()
            .expect("f32")
            .to_vec();
        assert_eq!(mask, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn epoch_stats_summary_and_improving_compare_loss_only() {
        let previous = EpochStats {
            avg_loss: 2.0,
            policy_agreement: 0.1,
            num_batches: 4,
        };
        let current = EpochStats {
            avg_loss: 1.5,
            policy_agreement: 0.05,
            num_batches: 3,
        };
        assert_eq!(current.summary(), "loss=1.5000 agree=5.00% batches=3");
        assert!(current.is_improving(&previous));
    }
}
