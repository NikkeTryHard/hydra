use super::schema::{RlPhaseConfig, TrainConfig, default_rl_phase};

pub const DEFAULT_BC_STAGE: &str = "bc_baseline";
pub const T0_BC_STAGE: &str = DEFAULT_BC_STAGE;
pub const T1_PPO_CONTROL_STAGE: &str = "T1_ppo_control";
pub const T2_DIRECT_SAMPLED_ACH_STAGE: &str = "T2_direct_sampled_ach";
pub const T3_DRDA_RESIDUAL_ACH_STAGE: &str = "T3_drda_residual_ach";
pub const T4_PBRS_BETA_SWEEP_STAGE: &str = "T4_pbrs_beta_sweep";
pub const T5_EXIT_AUXILIARY_STAGE: &str = "T5_exit_auxiliary";
pub const T6_DELTAQ_EXPERIMENT_STAGE: &str = "T6_deltaq_experiment";
pub const T7_POPULATION_WINDOW_STAGE: &str = "T7_population_window";
pub const DEFAULT_PPO_STAGE: &str = T1_PPO_CONTROL_STAGE;

pub const fn rl_stage_for_phase(phase: RlPhaseConfig) -> &'static str {
    match phase {
        RlPhaseConfig::PpoControl => T1_PPO_CONTROL_STAGE,
        RlPhaseConfig::DrdaAchSelfPlay => T3_DRDA_RESIDUAL_ACH_STAGE,
        RlPhaseConfig::ExitPondering => T5_EXIT_AUXILIARY_STAGE,
    }
}

pub fn rl_stage_for_config(config: &TrainConfig) -> &str {
    if let Some(stage) = config.stage.as_deref() {
        return stage;
    }
    let phase = config.rl.as_ref().map_or(default_rl_phase(), |rl| rl.phase);
    rl_stage_for_phase(phase)
}
