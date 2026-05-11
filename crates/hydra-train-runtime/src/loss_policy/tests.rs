use super::*;

#[test]
fn build_loss_config_defaults_to_zero_optional_weights() {
    let cfg = build_loss_config(None).expect("default loss config should be valid");
    assert_eq!(cfg.w_safety_residual, 0.0);
    assert_eq!(cfg.w_delta_q, 0.0);

    let exit_cfg = build_bc_exit_config(None);
    assert_eq!(exit_cfg.exit_weight, 0.0);
}

#[test]
fn build_loss_config_rejects_blocked_advanced_loss_fields() {
    let advanced = AdvancedLossConfig {
        belief_fields: Some(0.1),
        ..AdvancedLossConfig::default()
    };
    let err = build_loss_config(Some(&advanced)).expect_err("belief fields should be blocked");
    assert!(err.contains("advanced_loss.belief_fields is not supported"));
}

#[test]
fn build_configs_propagate_supported_weights() {
    let advanced = AdvancedLossConfig {
        exit: Some(0.4),
        safety_residual: Some(0.2),
        delta_q: Some(0.3),
        ..AdvancedLossConfig::default()
    };

    let bc_loss = build_loss_config(Some(&advanced)).expect("bc loss config should build");
    assert_eq!(bc_loss.w_safety_residual, 0.2);
    assert_eq!(bc_loss.w_delta_q, 0.3);

    let bc_exit = build_bc_exit_config(Some(&advanced));
    assert_eq!(bc_exit.exit_weight, 0.4);

    let rl_loss = build_rl_loss_config(Some(&advanced)).expect("rl loss config should build");
    assert_eq!(rl_loss.w_safety_residual, 0.2);
    assert_eq!(rl_loss.w_delta_q, 0.3);
}
