use super::*;

#[test]
fn bc_config_defaults_match_legacy_contract() {
    let cfg = BCTrainerConfig::new(ModelShapeConfig::actor());
    assert!((cfg.lr - 2.5e-4).abs() < 1e-10);
    assert!((cfg.min_learning_rate - 1e-6).abs() < 1e-12);
    assert_eq!(cfg.batch_size, 2048);
    assert_eq!(cfg.warmup_steps, 1000);
    assert!(cfg.validate().is_ok());
}

#[test]
fn model_shape_defaults_match_legacy_model_config() {
    let actor = ModelShapeConfig::actor();
    assert_eq!(actor.num_blocks, 12);
    assert_eq!(actor.input_channels, 192);
    assert_eq!(actor.hidden_channels, 256);
    assert_eq!(actor.num_groups, 32);
    assert_eq!(actor.se_bottleneck, 64);
    assert_eq!(actor.backbone_activation, BackboneActivationConfig::Mish);
    assert_eq!(actor.backbone_se_every_n, 1);
    assert_eq!(actor.backbone_norm, BackboneNormConfig::Both);
    assert_eq!(actor.action_space, 46);
    assert_eq!(actor.score_bins, 64);
    assert_eq!(actor.num_opponents, 3);
    assert_eq!(actor.grp_classes, 24);
    assert_eq!(actor.num_belief_components, 4);
    assert_eq!(actor.opponent_hand_type_classes, 8);
    assert!(actor.is_actor());
    assert!(actor.validate().is_ok());

    let learner = ModelShapeConfig::learner();
    assert_eq!(learner.num_blocks, 24);
    assert!(learner.is_learner());
    assert!(learner.validate().is_ok());
}

#[test]
fn model_shape_summary_reports_exact_shape_kind() {
    assert_eq!(
        ModelShapeConfig::actor().summary(),
        "actor(blocks=12, input=192, hidden=256, groups=32, se=64, activation=Mish, se_every_n=1, norm=Both, actions=46, score_bins=64, opponents=3, grp=24, belief_components=4, hand_type_classes=8)"
    );
    assert_eq!(
        ModelShapeConfig::learner().summary(),
        "learner(blocks=24, input=192, hidden=256, groups=32, se=64, activation=Mish, se_every_n=1, norm=Both, actions=46, score_bins=64, opponents=3, grp=24, belief_components=4, hand_type_classes=8)"
    );
    assert!(
        ModelShapeConfig::new(16)
            .summary()
            .starts_with("custom(blocks=16,")
    );
}

#[test]
fn experimental_backbone_profile_changes_param_estimate() {
    let baseline = ModelShapeConfig::learner();
    let mut ablated = ModelShapeConfig::learner()
        .with_backbone_activation(BackboneActivationConfig::Relu)
        .with_backbone_se_every_n(4)
        .with_backbone_norm(BackboneNormConfig::FirstOnly)
        .with_hidden_channels(128);
    ablated.num_blocks = 12;

    assert!(ablated.validate().is_ok());
    assert!(ablated.estimated_params() < baseline.estimated_params());
    assert_eq!(ablated.backbone_activation, BackboneActivationConfig::Relu);
    assert_eq!(ablated.backbone_se_every_n, 4);
    assert_eq!(ablated.backbone_norm, BackboneNormConfig::FirstOnly);
}

#[test]
fn experimental_backbone_profile_rejects_zero_se_stride() {
    assert_eq!(
        ModelShapeConfig::learner()
            .with_backbone_se_every_n(0)
            .validate(),
        Err("backbone_se_every_n must be > 0")
    );
}

#[test]
fn rl_config_defaults_match_legacy_contract() {
    let phase2 = RlConfig::default_phase2();
    assert_eq!(phase2.tau_drda, 4.0);
    assert!((phase2.lr - 2.5e-4).abs() < f64::EPSILON);
    assert!((phase2.exit_weight - DEFAULT_EXIT_WEIGHT).abs() < f32::EPSILON);
    assert!((phase2.aux_weight - DEFAULT_AUX_WEIGHT).abs() < f32::EPSILON);
    assert!(phase2.validate().is_ok());

    let phase3 = RlConfig::default_phase3();
    assert!((phase3.lr - 1e-4).abs() < f64::EPSILON);
    assert!((phase3.exit_weight - DEFAULT_EXIT_WEIGHT).abs() < f32::EPSILON);
    assert!((phase3.aux_weight - DEFAULT_AUX_WEIGHT).abs() < f32::EPSILON);
    assert!(phase3.validate().is_ok());
}

#[test]
fn ach_config_defaults_and_validation_match_algo_contract() {
    let cfg = AchConfig::new();
    assert!((cfg.eta - 1.0).abs() < 1e-6);
    assert!((cfg.eps - 0.5).abs() < 1e-6);
    assert!((cfg.l_th - 8.0).abs() < 1e-6);
    assert!((cfg.beta_ent - 5e-4).abs() < 1e-8);
    assert!(cfg.validate().is_ok());

    assert_eq!(
        AchConfig::new().with_eta(0.0).validate(),
        Err("eta must be positive")
    );
    assert_eq!(
        AchConfig::new().with_eps(1.0).validate(),
        Err("eps must be in (0,1)")
    );
    assert_eq!(
        AchConfig::new().with_l_th(0.0).validate(),
        Err("l_th must be positive")
    );
}

#[test]
fn config_validation_rejects_non_finite_scalars() {
    assert_eq!(
        AchConfig::new().with_beta_ent(f32::NAN).validate(),
        Err("ach config values must be finite")
    );
    let oracle = OracleGuidingConfig {
        lr_decay_factor: f32::INFINITY,
        ..OracleGuidingConfig::default()
    };
    assert_eq!(
        oracle.validate(),
        Err("oracle config values must be finite")
    );
    assert_eq!(
        BCTrainerConfig::new(ModelShapeConfig::actor())
            .with_lr(f64::NAN)
            .validate(),
        Err("learning rates must be finite")
    );
    assert_eq!(
        RlConfig::default_phase2()
            .with_aux_weight(f32::INFINITY)
            .validate(),
        Err("rl config values must be finite")
    );
}

#[test]
fn learning_rate_helpers_cover_zero_total_and_post_warmup_edges() {
    assert!((cosine_annealing_lr(3, 0, 1e-3, 1e-5) - 1e-3).abs() < 1e-12);

    let warmup_lr = warmup_then_cosine_lr(1, 4, 10, 1e-3, 1e-5);
    assert!((warmup_lr - 5.0e-4).abs() < 1e-12);

    let post_warmup_lr = warmup_then_cosine_lr(7, 4, 10, 1e-3, 1e-5);
    let expected = cosine_annealing_lr(3, 6, 1e-3, 1e-5);
    assert!((post_warmup_lr - expected).abs() < 1e-12);

    let long_warmup_lr = warmup_then_cosine_lr(0, 10, 4, 1e-8, 1e-6);
    assert!((long_warmup_lr - 1e-6).abs() < 1e-12);
    let end_warmup_lr = warmup_then_cosine_lr(9, 10, 4, 1e-3, 1e-6);
    assert!((end_warmup_lr - 1e-3).abs() < 1e-12);
}
