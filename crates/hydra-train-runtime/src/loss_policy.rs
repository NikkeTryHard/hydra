//! Runtime loss configuration policy.

use hydra_train_types::config::BcExitConfig;
use hydra_train_types::losses::HydraLossConfig;

use crate::config::AdvancedLossConfig;

fn reject_blocked_advanced_loss_presence(field: &str, weight: Option<f32>) -> Result<(), String> {
    match weight {
        Some(_) => Err(format!(
            "advanced_loss.{field} is not supported in train.rs because this BC data path does not safely support it yet"
        )),
        None => Ok(()),
    }
}

/// Builds the BC loss config from runtime advanced-loss settings.
pub fn build_loss_config(
    advanced_loss: Option<&AdvancedLossConfig>,
) -> Result<HydraLossConfig, String> {
    if let Some(cfg) = advanced_loss {
        reject_blocked_advanced_loss_presence("belief_fields", cfg.belief_fields)?;
        reject_blocked_advanced_loss_presence("mixture_weight", cfg.mixture_weight)?;
        reject_blocked_advanced_loss_presence("opponent_hand_type", cfg.opponent_hand_type)?;
    }

    let safety_residual = advanced_loss
        .and_then(|cfg| cfg.safety_residual)
        .unwrap_or(0.0);
    let delta_q = advanced_loss.and_then(|cfg| cfg.delta_q).unwrap_or(0.0);

    let loss_config = HydraLossConfig::new()
        .with_w_safety_residual(safety_residual)
        .with_w_delta_q(delta_q);
    loss_config
        .validate()
        .map_err(|err| format!("invalid loss config: {err}"))?;
    Ok(loss_config)
}

/// Builds the BC ExIt sidecar loss config from runtime advanced-loss settings.
pub fn build_bc_exit_config(advanced_loss: Option<&AdvancedLossConfig>) -> BcExitConfig {
    let exit_weight = advanced_loss.and_then(|cfg| cfg.exit).unwrap_or(0.0);
    BcExitConfig { exit_weight }
}

/// Builds the RL loss config from runtime advanced-loss settings.
pub fn build_rl_loss_config(
    advanced_loss: Option<&AdvancedLossConfig>,
) -> Result<HydraLossConfig, String> {
    if let Some(cfg) = advanced_loss {
        reject_blocked_advanced_loss_presence("belief_fields", cfg.belief_fields)?;
        reject_blocked_advanced_loss_presence("mixture_weight", cfg.mixture_weight)?;
        reject_blocked_advanced_loss_presence("opponent_hand_type", cfg.opponent_hand_type)?;
    }

    let mut loss = HydraLossConfig::new();
    if let Some(cfg) = advanced_loss {
        loss = loss
            .with_w_safety_residual(cfg.safety_residual.unwrap_or(0.0))
            .with_w_delta_q(cfg.delta_q.unwrap_or(0.0));
    }
    loss.validate()
        .map_err(|err| format!("invalid RL loss config: {err}"))?;
    Ok(loss)
}

#[cfg(test)]
mod tests {
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
}
