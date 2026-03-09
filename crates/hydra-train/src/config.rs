//! Model configuration and constants for the Hydra neural architecture.

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES};

pub const INPUT_CHANNELS: usize = NUM_CHANNELS;
pub const TILE_DIM: usize = NUM_TILES;
pub const HIDDEN_CHANNELS: usize = 256;
pub const SE_REDUCTION: usize = 4;
pub const SE_BOTTLENECK: usize = HIDDEN_CHANNELS / SE_REDUCTION;
pub const NUM_GROUPS: usize = 32;
pub const ACTION_SPACE: usize = HYDRA_ACTION_SPACE;
pub const SCORE_BINS: usize = 64;
pub const NUM_OPPONENTS: usize = 3;
pub const GRP_CLASSES: usize = 24;

pub const RIICHI_ACTION: u8 = 37;

pub fn is_discard_action(action: u8) -> bool {
    action <= 36
}

pub const PASS_ACTION: u8 = 45;

pub const AGARI_ACTION: u8 = 43;
pub const RYUUKYOKU_ACTION: u8 = 44;
pub const KAN_ACTION: u8 = 42;
pub const FIRST_CALL_ACTION: u8 = 38;
pub const LAST_CALL_ACTION: u8 = 42;

pub fn is_agari_action(action: u8) -> bool {
    action == 43
}

pub fn action_type_name(action: u8) -> &'static str {
    match action {
        0..=36 => "discard",
        37 => "riichi",
        38..=42 => "call",
        43 => "agari",
        44 => "ryuukyoku",
        45 => "pass",
        _ => "unknown",
    }
}

pub fn is_pass_action(action: u8) -> bool {
    action == PASS_ACTION
}

pub fn is_riichi_action(action: u8) -> bool {
    action == RIICHI_ACTION
}

pub fn is_call_action(action: u8) -> bool {
    (FIRST_CALL_ACTION..=LAST_CALL_ACTION).contains(&action)
}

pub const GAE_GAMMA: f32 = 0.995;
pub const GAE_LAMBDA: f32 = 0.95;
pub const BC_LR: f64 = 2.5e-4;
pub const BC_LR_MIN: f64 = 1e-6;
pub const ACH_LR: f64 = 2.5e-4;
pub const GRAD_CLIP_NORM: f32 = 1.0;
pub const TAU_DRDA: f32 = 4.0;
pub const TAU_EXIT: f32 = 1.0;
pub const C_PUCT: f32 = 2.5;
pub const AFBS_TOP_K: usize = 5;
pub const CT_SMC_PARTICLES: usize = 128;

pub const ON_TURN_BEAM_W: usize = 64;
pub const ON_TURN_DEPTH: u8 = 4;
pub const ON_TURN_PARTICLES: usize = 128;
pub const PONDER_BEAM_W: usize = 256;
pub const PONDER_DEPTH: u8 = 10;
pub const PONDER_PARTICLES: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPhase {
    BenchmarkGates,
    BcWarmStart,
    OracleGuiding,
    DrdaAchSelfPlay,
    ExitPondering,
}

impl TrainingPhase {
    pub fn gpu_hours_budget(self) -> u32 {
        match self {
            Self::BenchmarkGates => 150,
            Self::BcWarmStart => 50,
            Self::OracleGuiding => 200,
            Self::DrdaAchSelfPlay => 800,
            Self::ExitPondering => 800,
        }
    }

    pub fn cumulative_budget_before(self) -> u32 {
        match self {
            Self::BenchmarkGates => 0,
            Self::BcWarmStart => Self::BenchmarkGates.gpu_hours_budget(),
            Self::OracleGuiding => {
                Self::BenchmarkGates.gpu_hours_budget() + Self::BcWarmStart.gpu_hours_budget()
            }
            Self::DrdaAchSelfPlay => {
                Self::BenchmarkGates.gpu_hours_budget()
                    + Self::BcWarmStart.gpu_hours_budget()
                    + Self::OracleGuiding.gpu_hours_budget()
            }
            Self::ExitPondering => {
                Self::BenchmarkGates.gpu_hours_budget()
                    + Self::BcWarmStart.gpu_hours_budget()
                    + Self::OracleGuiding.gpu_hours_budget()
                    + Self::DrdaAchSelfPlay.gpu_hours_budget()
            }
        }
    }

    pub fn cumulative_budget_through(self) -> u32 {
        self.cumulative_budget_before() + self.gpu_hours_budget()
    }

    pub fn exit_schedule_phase(self) -> u8 {
        match self {
            Self::BenchmarkGates | Self::BcWarmStart | Self::OracleGuiding => 1,
            Self::DrdaAchSelfPlay => 2,
            Self::ExitPondering => 3,
        }
    }

    pub fn next(self) -> Option<Self> {
        match self {
            Self::BenchmarkGates => Some(Self::BcWarmStart),
            Self::BcWarmStart => Some(Self::OracleGuiding),
            Self::OracleGuiding => Some(Self::DrdaAchSelfPlay),
            Self::DrdaAchSelfPlay => Some(Self::ExitPondering),
            Self::ExitPondering => None,
        }
    }

    pub fn uses_exit(self) -> bool {
        matches!(self, Self::DrdaAchSelfPlay | Self::ExitPondering)
    }

    pub fn uses_oracle(self) -> bool {
        matches!(
            self,
            Self::OracleGuiding | Self::DrdaAchSelfPlay | Self::ExitPondering
        )
    }

    pub fn phase_index(self) -> u8 {
        match self {
            Self::BenchmarkGates => 0,
            Self::BcWarmStart => 1,
            Self::OracleGuiding => 2,
            Self::DrdaAchSelfPlay => 3,
            Self::ExitPondering => 4,
        }
    }
}

pub struct OracleGuidingConfig {
    pub dropout_start: f32,
    pub dropout_end: f32,
    pub lr_decay_factor: f32,
}

impl OracleGuidingConfig {
    pub fn summary(&self) -> String {
        format!(
            "oracle(drop={:.1}->{:.1}, decay={:.2})",
            self.dropout_start, self.dropout_end, self.lr_decay_factor
        )
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.dropout_start < 0.0 || self.dropout_start > 1.0 {
            return Err("dropout_start in [0,1]");
        }
        if self.dropout_end < 0.0 || self.dropout_end > 1.0 {
            return Err("dropout_end in [0,1]");
        }
        if self.lr_decay_factor <= 0.0 || self.lr_decay_factor > 1.0 {
            return Err("lr_decay_factor in (0,1]");
        }
        Ok(())
    }

    pub fn dropout_at_step(&self, step: usize, total_steps: usize) -> f32 {
        if total_steps == 0 {
            return self.dropout_start;
        }
        let t = (step as f32 / total_steps as f32).min(1.0);
        self.dropout_start + (self.dropout_end - self.dropout_start) * t
    }

    pub fn effective_learning_rate(&self, base_lr: f64, step: usize, total_steps: usize) -> f64 {
        if self.dropout_at_step(step, total_steps) <= self.dropout_end + 1e-6 {
            base_lr * self.lr_decay_factor as f64
        } else {
            base_lr
        }
    }

    pub fn should_reject_importance_weight(
        &self,
        importance_weight: f32,
        max_importance_weight: f32,
        step: usize,
        total_steps: usize,
    ) -> bool {
        self.dropout_at_step(step, total_steps) <= self.dropout_end + 1e-6
            && importance_weight > max_importance_weight
    }
}

impl Default for OracleGuidingConfig {
    fn default() -> Self {
        Self {
            dropout_start: 1.0,
            dropout_end: 0.0,
            lr_decay_factor: 0.1,
        }
    }
}

pub fn validate_training_config(
    model_cfg: &crate::model::HydraModelConfig,
    ach_cfg: &crate::training::ach::AchConfig,
    exit_cfg: &crate::training::exit::ExitConfig,
) -> Result<(), &'static str> {
    model_cfg.validate()?;
    ach_cfg.validate()?;
    exit_cfg.validate()?;
    Ok(())
}

/// Validate all training configs at pipeline startup.
pub fn validate_all_configs(
    model_cfg: &crate::model::HydraModelConfig,
    loss_cfg: &crate::training::losses::HydraLossConfig,
    ach_cfg: &crate::training::ach::AchConfig,
    exit_cfg: &crate::training::exit::ExitConfig,
    drda_cfg: &crate::training::drda::DrdaConfig,
    gae_cfg: &crate::training::gae::GaeConfig,
    distill_cfg: &crate::training::distill::DistillConfig,
) -> Result<(), String> {
    model_cfg.validate().map_err(|e| format!("model: {e}"))?;
    loss_cfg.validate().map_err(|e| format!("loss: {e}"))?;
    ach_cfg.validate().map_err(|e| format!("ach: {e}"))?;
    exit_cfg.validate().map_err(|e| format!("exit: {e}"))?;
    drda_cfg.validate().map_err(|e| format!("drda: {e}"))?;
    gae_cfg.validate().map_err(|e| format!("gae: {e}"))?;
    distill_cfg
        .validate()
        .map_err(|e| format!("distill: {e}"))?;
    Ok(())
}

pub struct PipelineState {
    pub phase: TrainingPhase,
    pub gpu_hours_used: f32,
    pub total_games: u64,
    pub total_samples: u64,
    pub learner_version: u32,
    pub actor_version: u32,
}

impl Default for PipelineState {
    fn default() -> Self {
        Self {
            phase: TrainingPhase::BenchmarkGates,
            gpu_hours_used: 0.0,
            total_games: 0,
            total_samples: 0,
            learner_version: 0,
            actor_version: 0,
        }
    }
}

impl PipelineState {
    pub fn advance_phase(&mut self) {
        self.phase = match self.phase {
            TrainingPhase::BenchmarkGates => TrainingPhase::BcWarmStart,
            TrainingPhase::BcWarmStart => TrainingPhase::OracleGuiding,
            TrainingPhase::OracleGuiding => TrainingPhase::DrdaAchSelfPlay,
            TrainingPhase::DrdaAchSelfPlay => TrainingPhase::ExitPondering,
            TrainingPhase::ExitPondering => TrainingPhase::ExitPondering,
        };
    }

    pub fn remaining_budget(&self) -> f32 {
        2000.0 - self.gpu_hours_used
    }

    pub fn total_budget() -> f32 {
        2000.0
    }

    pub fn overall_progress(&self) -> f32 {
        (self.gpu_hours_used / Self::total_budget()).min(1.0)
    }

    pub fn phase_progress(&self) -> f32 {
        let budget = self.phase.gpu_hours_budget() as f32;
        if budget == 0.0 {
            return 0.0;
        }
        self.phase_hours_used() / budget
    }

    pub fn phase_hours_used(&self) -> f32 {
        let phase_start = self.phase.cumulative_budget_before() as f32;
        let phase_budget = self.phase.gpu_hours_budget() as f32;
        (self.gpu_hours_used - phase_start).clamp(0.0, phase_budget)
    }

    pub fn increment_learner_version(&mut self) {
        self.learner_version += 1;
    }
    pub fn increment_actor_version(&mut self) {
        self.actor_version += 1;
    }

    pub fn record_game(&mut self, num_samples: usize) {
        self.total_games += 1;
        self.total_samples += num_samples as u64;
    }

    pub fn tick_gpu_hours(&mut self, hours: f32) {
        self.gpu_hours_used += hours;
    }

    pub fn should_advance_phase(&self) -> bool {
        self.gpu_hours_used >= self.phase.cumulative_budget_through() as f32
    }

    pub fn progress_summary(&self) -> String {
        format!(
            "phase={:?} phase_hours={:.1}/{} total_hours={:.1} games={} v{}->v{}",
            self.phase,
            self.phase_hours_used(),
            self.phase.gpu_hours_budget(),
            self.gpu_hours_used,
            self.total_games,
            self.learner_version,
            self.actor_version
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_hydra_core() {
        assert_eq!(INPUT_CHANNELS, NUM_CHANNELS);
        assert_eq!(TILE_DIM, 34);
        assert_eq!(ACTION_SPACE, 46);
    }

    #[test]
    fn derived_constants_correct() {
        assert_eq!(SE_BOTTLENECK, 64);
        assert_eq!(HIDDEN_CHANNELS / NUM_GROUPS, 8);
        assert_eq!(INPUT_CHANNELS * TILE_DIM, NUM_CHANNELS * NUM_TILES);
    }

    #[test]
    fn hyperparameters_match_spec() {
        assert!((GAE_GAMMA - 0.995).abs() < 1e-6);
        assert!((GAE_LAMBDA - 0.95).abs() < 1e-6);
        assert!((BC_LR - 2.5e-4).abs() < 1e-10);
        assert!((TAU_DRDA - 4.0).abs() < 1e-6);
        assert!((C_PUCT - 2.5).abs() < 1e-6);
        assert_eq!(AFBS_TOP_K, 5);
        assert_eq!(CT_SMC_PARTICLES, 128);
    }

    #[test]
    fn pipeline_state_defaults() {
        let state = PipelineState::default();
        assert_eq!(state.phase, TrainingPhase::BenchmarkGates);
        assert!((state.remaining_budget() - 2000.0).abs() < 0.01);
    }

    #[test]
    fn phase_advancement() {
        let mut state = PipelineState::default();
        state.advance_phase();
        assert_eq!(state.phase, TrainingPhase::BcWarmStart);
        state.advance_phase();
        assert_eq!(state.phase, TrainingPhase::OracleGuiding);
    }

    #[test]
    fn phase_budgets_are_cumulative() {
        assert_eq!(TrainingPhase::BenchmarkGates.cumulative_budget_before(), 0);
        assert_eq!(TrainingPhase::BcWarmStart.cumulative_budget_before(), 150);
        assert_eq!(TrainingPhase::OracleGuiding.cumulative_budget_before(), 200);
        assert_eq!(
            TrainingPhase::DrdaAchSelfPlay.cumulative_budget_before(),
            400
        );
        assert_eq!(
            TrainingPhase::ExitPondering.cumulative_budget_before(),
            1200
        );
        assert_eq!(
            TrainingPhase::ExitPondering.cumulative_budget_through(),
            2000
        );
    }

    #[test]
    fn phase_progress_uses_phase_local_hours() {
        let state = PipelineState {
            phase: TrainingPhase::DrdaAchSelfPlay,
            gpu_hours_used: 600.0,
            ..PipelineState::default()
        };
        assert!((state.phase_hours_used() - 200.0).abs() < 1e-6);
        assert!((state.phase_progress() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn phase_advance_requires_cumulative_budget() {
        let state = PipelineState {
            phase: TrainingPhase::OracleGuiding,
            gpu_hours_used: 250.0,
            ..PipelineState::default()
        };
        assert!(!state.should_advance_phase());

        let ready = PipelineState {
            phase: TrainingPhase::OracleGuiding,
            gpu_hours_used: 400.0,
            ..PipelineState::default()
        };
        assert!(ready.should_advance_phase());
    }

    #[test]
    fn exit_schedule_phase_matches_hydra_final_rollout_plan() {
        assert_eq!(TrainingPhase::BcWarmStart.exit_schedule_phase(), 1);
        assert_eq!(TrainingPhase::OracleGuiding.exit_schedule_phase(), 1);
        assert_eq!(TrainingPhase::DrdaAchSelfPlay.exit_schedule_phase(), 2);
        assert_eq!(TrainingPhase::ExitPondering.exit_schedule_phase(), 3);
    }

    #[test]
    fn action_type_classification() {
        assert_eq!(action_type_name(0), "discard");
        assert_eq!(action_type_name(36), "discard");
        assert_eq!(action_type_name(RIICHI_ACTION), "riichi");
        assert_eq!(action_type_name(KAN_ACTION), "call");
        assert_eq!(action_type_name(AGARI_ACTION), "agari");
        assert_eq!(action_type_name(RYUUKYOKU_ACTION), "ryuukyoku");
        assert_eq!(action_type_name(PASS_ACTION), "pass");
        assert!(is_discard_action(5));
        assert!(!is_discard_action(RIICHI_ACTION));
        assert!(is_call_action(40));
        assert!(is_call_action(KAN_ACTION));
        assert!(!is_call_action(AGARI_ACTION));
        assert!(!is_call_action(RYUUKYOKU_ACTION));
        assert!(!is_call_action(PASS_ACTION));
        assert!(!is_call_action(0));
    }

    #[test]
    fn phase_uses_exit_oracle() {
        assert!(!TrainingPhase::BcWarmStart.uses_exit());
        assert!(TrainingPhase::ExitPondering.uses_exit());
        assert!(!TrainingPhase::BcWarmStart.uses_oracle());
        assert!(TrainingPhase::OracleGuiding.uses_oracle());
    }

    #[test]
    fn oracle_guiding_dropout_schedule_and_lr_decay() {
        let cfg = OracleGuidingConfig::default();
        assert!((cfg.dropout_at_step(0, 100) - 1.0).abs() < 1e-6);
        assert!((cfg.dropout_at_step(100, 100) - 0.0).abs() < 1e-6);
        assert!((cfg.effective_learning_rate(1e-4, 0, 100) - 1e-4).abs() < 1e-12);
        assert!((cfg.effective_learning_rate(1e-4, 100, 100) - 1e-5).abs() < 1e-12);
    }

    #[test]
    fn oracle_guiding_rejects_large_importance_weights_post_dropout() {
        let cfg = OracleGuidingConfig::default();
        assert!(!cfg.should_reject_importance_weight(3.0, 2.0, 50, 100));
        assert!(cfg.should_reject_importance_weight(3.0, 2.0, 100, 100));
        assert!(!cfg.should_reject_importance_weight(1.5, 2.0, 100, 100));
    }

    #[test]
    fn test_all_defaults_match_roadmap() {
        use crate::model::HydraModelConfig;
        use crate::training::{ach, bc, distill, drda, exit, gae, losses};

        // -- Model configs --
        let learner = HydraModelConfig::learner();
        assert_eq!(learner.num_blocks, 24, "learner num_blocks should be 24");
        let actor = HydraModelConfig::actor();
        assert_eq!(actor.num_blocks, 12, "actor num_blocks should be 12");
        // Shared model defaults (check on learner, same for actor)
        assert_eq!(
            learner.hidden_channels, 256,
            "hidden_channels should be 256"
        );
        assert_eq!(learner.num_groups, 32, "num_groups should be 32");
        assert_eq!(learner.se_bottleneck, 64, "se_bottleneck should be 64");

        // -- Loss weights --
        let loss = losses::HydraLossConfig::new();
        assert!(
            (loss.w_pi - 1.0).abs() < 1e-6,
            "w_pi should be 1.0, got {}",
            loss.w_pi
        );
        assert!(
            (loss.w_v - 0.5).abs() < 1e-6,
            "w_v should be 0.5, got {}",
            loss.w_v
        );
        assert!(
            (loss.w_grp - 0.2).abs() < 1e-6,
            "w_grp should be 0.2, got {}",
            loss.w_grp
        );
        assert!(
            (loss.w_tenpai - 0.1).abs() < 1e-6,
            "w_tenpai should be 0.1, got {}",
            loss.w_tenpai
        );
        assert!(
            (loss.w_danger - 0.1).abs() < 1e-6,
            "w_danger should be 0.1, got {}",
            loss.w_danger
        );
        assert!(
            (loss.w_opp - 0.1).abs() < 1e-6,
            "w_opp should be 0.1, got {}",
            loss.w_opp
        );
        assert!(
            (loss.w_score - 0.025).abs() < 1e-6,
            "w_score should be 0.025, got {}",
            loss.w_score
        );
        assert!(
            (loss.w_oracle_critic - 0.0).abs() < 1e-6,
            "w_oracle_critic should be 0.0, got {}",
            loss.w_oracle_critic
        );

        // -- ACH config --
        let ach = ach::AchConfig::new();
        assert!(
            (ach.eta - 1.0).abs() < 1e-6,
            "ACH eta should be 1.0, got {}",
            ach.eta
        );
        assert!(
            (ach.eps - 0.5).abs() < 1e-6,
            "ACH eps should be 0.5, got {}",
            ach.eps
        );
        assert!(
            (ach.l_th - 8.0).abs() < 1e-6,
            "ACH l_th should be 8.0, got {}",
            ach.l_th
        );
        assert!(
            (ach.beta_ent - 5e-4).abs() < 1e-8,
            "ACH beta_ent should be 5e-4, got {}",
            ach.beta_ent
        );

        // -- ExIt config --
        let ex = exit::ExitConfig::new();
        assert!(
            (ex.tau_exit - 1.0).abs() < 1e-6,
            "tau_exit should be 1.0, got {}",
            ex.tau_exit
        );
        assert!(
            (ex.exit_weight - 0.5).abs() < 1e-6,
            "exit_weight should be 0.5, got {}",
            ex.exit_weight
        );
        assert_eq!(
            ex.min_visits, 64,
            "min_visits should be 64, got {}",
            ex.min_visits
        );
        assert!(
            (ex.safety_valve_max_kl - 2.0).abs() < 1e-6,
            "safety_valve_max_kl should be 2.0, got {}",
            ex.safety_valve_max_kl
        );

        // -- DRDA config --
        let drda = drda::DrdaConfig::new();
        assert!(
            (drda.tau_drda - 4.0).abs() < 1e-6,
            "tau_drda should be 4.0, got {}",
            drda.tau_drda
        );

        // -- GAE config --
        let gae = gae::GaeConfig::default();
        assert!(
            (gae.gamma - 0.995).abs() < 1e-6,
            "GAE gamma should be 0.995, got {}",
            gae.gamma
        );
        assert!(
            (gae.lambda - 0.95).abs() < 1e-6,
            "GAE lambda should be 0.95, got {}",
            gae.lambda
        );

        // -- BC trainer config --
        let bc = bc::BCTrainerConfig::new(HydraModelConfig::learner());
        assert!(
            (bc.lr - 2.5e-4).abs() < 1e-10,
            "BC lr should be 2.5e-4, got {}",
            bc.lr
        );
        assert_eq!(
            bc.batch_size, 2048,
            "BC batch_size should be 2048, got {}",
            bc.batch_size
        );
        assert!(
            (bc.grad_clip_norm - 1.0).abs() < 1e-6,
            "BC grad_clip_norm should be 1.0, got {}",
            bc.grad_clip_norm
        );
        assert!(
            (bc.weight_decay - 1e-5).abs() < 1e-10,
            "BC weight_decay should be 1e-5, got {}",
            bc.weight_decay
        );

        // -- Distill config --
        let dist = distill::DistillConfig::new();
        assert!(
            (dist.kd_kl_weight - 1.0).abs() < 1e-6,
            "kd_kl_weight should be 1.0, got {}",
            dist.kd_kl_weight
        );
        assert!(
            (dist.kd_mse_weight - 0.5).abs() < 1e-6,
            "kd_mse_weight should be 0.5, got {}",
            dist.kd_mse_weight
        );
        assert!(
            (dist.ema_decay - 0.999).abs() < 1e-6,
            "ema_decay should be 0.999, got {}",
            dist.ema_decay
        );

        // -- Constants --
        assert_eq!(ACTION_SPACE, 46, "ACTION_SPACE should be 46");
        assert_eq!(HIDDEN_CHANNELS, 256, "HIDDEN_CHANNELS should be 256");
        assert_eq!(NUM_GROUPS, 32, "NUM_GROUPS should be 32");
        assert_eq!(SE_BOTTLENECK, 64, "SE_BOTTLENECK should be 64");
        assert_eq!(SCORE_BINS, 64, "SCORE_BINS should be 64");
        assert_eq!(GRP_CLASSES, 24, "GRP_CLASSES should be 24");
        assert!((C_PUCT - 2.5).abs() < 1e-6, "C_PUCT should be 2.5");
        assert_eq!(AFBS_TOP_K, 5, "AFBS_TOP_K should be 5");
        assert_eq!(CT_SMC_PARTICLES, 128, "CT_SMC_PARTICLES should be 128");
    }

    #[test]
    fn validate_all_configs_defaults_pass() {
        use crate::model::HydraModelConfig;
        use crate::training::{ach, distill, drda, exit, gae, losses};
        let result = validate_all_configs(
            &HydraModelConfig::learner(),
            &losses::HydraLossConfig::new(),
            &ach::AchConfig::new(),
            &exit::ExitConfig::new(),
            &drda::DrdaConfig::new(),
            &gae::GaeConfig::default(),
            &distill::DistillConfig::new(),
        );
        assert!(result.is_ok(), "default configs should pass: {result:?}");
    }

    #[test]
    fn validate_all_configs_catches_bad_ach() {
        use crate::model::HydraModelConfig;
        use crate::training::{ach, distill, drda, exit, gae, losses};
        let bad_ach = ach::AchConfig::new().with_eta(0.0);
        let result = validate_all_configs(
            &HydraModelConfig::learner(),
            &losses::HydraLossConfig::new(),
            &bad_ach,
            &exit::ExitConfig::new(),
            &drda::DrdaConfig::new(),
            &gae::GaeConfig::default(),
            &distill::DistillConfig::new(),
        );
        assert!(result.is_err(), "zero eta should fail validation");
        assert!(result.unwrap_err().contains("ach"));
    }
}
