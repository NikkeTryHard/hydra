//! Backend-independent training configuration contracts.
//!
//! These DTOs live below the train facade so bin/runtime code can share epoch/RL
//! configuration without depending on `hydra-train`.

use burn::config::Config;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;

/// Default RL physical microbatch size used when the train YAML omits one.
pub const DEFAULT_RL_MICROBATCH_SIZE: usize = 128;
/// Default ExIt loss weight for RL phases.
pub const DEFAULT_EXIT_WEIGHT: f32 = 0.5;
/// Default auxiliary-loss weight for RL phases.
pub const DEFAULT_AUX_WEIGHT: f32 = 0.1;
/// Generalized Advantage Estimation gamma used by RL self-play collation.
pub const GAE_GAMMA: f32 = 0.995;
/// Generalized Advantage Estimation lambda used by RL self-play collation.
pub const GAE_LAMBDA: f32 = 0.95;

#[derive(Config, Debug)]
pub struct AchConfig {
    #[config(default = "1.0")]
    pub eta: f32,
    #[config(default = "0.5")]
    pub eps: f32,
    #[config(default = "8.0")]
    pub l_th: f32,
    #[config(default = "5e-4")]
    pub beta_ent: f32,
}

impl AchConfig {
    pub fn summary(&self) -> String {
        format!(
            "ach(eta={:.1}, eps={:.1}, l_th={:.0}, ent={:.1e})",
            self.eta, self.eps, self.l_th, self.beta_ent
        )
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.eta.is_finite()
            || !self.eps.is_finite()
            || !self.l_th.is_finite()
            || !self.beta_ent.is_finite()
        {
            return Err("ach config values must be finite");
        }
        if self.eta <= 0.0 {
            return Err("eta must be positive");
        }
        if self.eps <= 0.0 || self.eps >= 1.0 {
            return Err("eps must be in (0,1)");
        }
        if self.l_th <= 0.0 {
            return Err("l_th must be positive");
        }
        if self.beta_ent < 0.0 {
            return Err("beta_ent must be non-negative");
        }
        Ok(())
    }
}

pub const MIN_TAU_DRDA: f32 = 2.0;

/// Computes a cosine-annealed learning rate clamped to the configured floor.
pub fn cosine_annealing_lr(step: usize, total_steps: usize, lr_max: f64, lr_min: f64) -> f64 {
    if total_steps == 0 {
        return lr_max;
    }
    let t = (step as f64 / total_steps as f64).min(1.0);
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * t).cos())
}

/// Computes a linear warmup followed by cosine annealing.
pub fn warmup_then_cosine_lr(
    step: usize,
    warmup_steps: usize,
    total_steps: usize,
    lr_max: f64,
    lr_min: f64,
) -> f64 {
    if warmup_steps > 0 && step < warmup_steps {
        let warmup_fraction = (step.saturating_add(1) as f64 / warmup_steps as f64).min(1.0);
        return (lr_max * warmup_fraction).max(lr_min);
    }

    let anneal_steps = total_steps.saturating_sub(warmup_steps);
    cosine_annealing_lr(
        step.saturating_sub(warmup_steps),
        anneal_steps,
        lr_max,
        lr_min,
    )
}

/// Behavioral-cloning ExIt loss weighting.
#[derive(Debug, Clone, Copy)]
pub struct BcExitConfig {
    pub exit_weight: f32,
}

impl Default for BcExitConfig {
    fn default() -> Self {
        Self { exit_weight: 0.0 }
    }
}

/// Oracle-guiding dropout and learning-rate schedule configuration.
pub struct OracleGuidingConfig {
    pub dropout_start: f32,
    pub dropout_end: f32,
    pub lr_decay_factor: f32,
}

impl OracleGuidingConfig {
    /// Human-readable compact schedule summary.
    pub fn summary(&self) -> String {
        format!(
            "oracle(drop={:.1}->{:.1}, decay={:.2})",
            self.dropout_start, self.dropout_end, self.lr_decay_factor
        )
    }

    /// Validates scalar schedule ranges.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.dropout_start.is_finite()
            || !self.dropout_end.is_finite()
            || !self.lr_decay_factor.is_finite()
        {
            return Err("oracle config values must be finite");
        }
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

    /// Linearly interpolates the oracle-target keep probability at `step`.
    pub fn dropout_at_step(&self, step: usize, total_steps: usize) -> f32 {
        if total_steps == 0 {
            return self.dropout_start;
        }
        let t = (step as f32 / total_steps as f32).min(1.0);
        self.dropout_start + (self.dropout_end - self.dropout_start) * t
    }

    /// Applies oracle-guiding LR decay after the dropout schedule reaches its floor.
    pub fn effective_learning_rate(&self, base_lr: f64, step: usize, total_steps: usize) -> f64 {
        if self.dropout_at_step(step, total_steps) <= self.dropout_end + 1e-6 {
            base_lr * self.lr_decay_factor as f64
        } else {
            base_lr
        }
    }

    /// Returns true when a post-dropout importance weight exceeds the configured cap.
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

/// Backbone activation used by experimental model profiles.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackboneActivationConfig {
    #[default]
    Mish,
    Silu,
    Relu,
}

/// Per-block normalization layout used by experimental model profiles.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackboneNormConfig {
    #[default]
    Both,
    FirstOnly,
}

/// Backend-independent Hydra model shape/configuration used by trainer contracts.
#[derive(Config, Debug)]
pub struct ModelShapeConfig {
    pub num_blocks: usize,
    #[config(default = "192")]
    pub input_channels: usize,
    #[config(default = "256")]
    pub hidden_channels: usize,
    #[config(default = "32")]
    pub num_groups: usize,
    #[config(default = "64")]
    pub se_bottleneck: usize,
    #[config(default = "BackboneActivationConfig::default()")]
    pub backbone_activation: BackboneActivationConfig,
    #[config(default = "1")]
    pub backbone_se_every_n: usize,
    #[config(default = "BackboneNormConfig::default()")]
    pub backbone_norm: BackboneNormConfig,
    #[config(default = "46")]
    pub action_space: usize,
    #[config(default = "64")]
    pub score_bins: usize,
    #[config(default = "3")]
    pub num_opponents: usize,
    #[config(default = "24")]
    pub grp_classes: usize,
    #[config(default = "4")]
    pub num_belief_components: usize,
    #[config(default = "8")]
    pub opponent_hand_type_classes: usize,
}

impl ModelShapeConfig {
    pub fn summary(&self) -> String {
        let kind = if self.is_actor() {
            "actor"
        } else if self.is_learner() {
            "learner"
        } else {
            "custom"
        };
        format!(
            "{}(blocks={}, input={}, hidden={}, groups={}, se={}, activation={:?}, se_every_n={}, norm={:?}, actions={}, score_bins={}, opponents={}, grp={}, belief_components={}, hand_type_classes={})",
            kind,
            self.num_blocks,
            self.input_channels,
            self.hidden_channels,
            self.num_groups,
            self.se_bottleneck,
            self.backbone_activation,
            self.backbone_se_every_n,
            self.backbone_norm,
            self.action_space,
            self.score_bins,
            self.num_opponents,
            self.grp_classes,
            self.num_belief_components,
            self.opponent_hand_type_classes,
        )
    }

    pub fn is_actor(&self) -> bool {
        self.num_blocks == 12
    }

    pub fn is_learner(&self) -> bool {
        self.num_blocks == 24
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_blocks == 0 {
            return Err("num_blocks must be > 0");
        }
        if self.input_channels == 0 {
            return Err("input_channels must be > 0");
        }
        if self.hidden_channels == 0 {
            return Err("hidden_channels must be > 0");
        }
        if self.num_groups == 0 || !self.hidden_channels.is_multiple_of(self.num_groups) {
            return Err("hidden_channels must be divisible by num_groups");
        }
        if self.se_bottleneck == 0 {
            return Err("se_bottleneck must be > 0");
        }
        if self.backbone_se_every_n == 0 {
            return Err("backbone_se_every_n must be > 0");
        }
        if self.action_space == 0 {
            return Err("action_space must be > 0");
        }
        if self.score_bins == 0 {
            return Err("score_bins must be > 0");
        }
        if self.num_opponents == 0 {
            return Err("num_opponents must be > 0");
        }
        if self.grp_classes == 0 {
            return Err("grp_classes must be > 0");
        }
        if self.num_belief_components == 0 {
            return Err("num_belief_components must be > 0");
        }
        if self.opponent_hand_type_classes == 0 {
            return Err("opponent_hand_type_classes must be > 0");
        }
        Ok(())
    }

    pub fn actor() -> Self {
        Self::new(12)
    }

    pub fn learner() -> Self {
        Self::new(24)
    }

    pub fn estimated_params(&self) -> usize {
        let h = self.hidden_channels;
        let se_b = self.se_bottleneck;
        let input_conv = self.input_channels * h * 3 + h;
        let gn = h * 2;
        let se_blocks = self.num_blocks / self.backbone_se_every_n;
        let norm_sites_per_block = match self.backbone_norm {
            BackboneNormConfig::Both => 2,
            BackboneNormConfig::FirstOnly => 1,
        };
        let block = (h * h * 3 + h) * 2 + gn * norm_sites_per_block;
        let se = (h * se_b + se_b) + (se_b * h + h);
        let backbone = input_conv + gn + block * self.num_blocks + se * se_blocks + gn;
        let policy = h * self.action_space + self.action_space;
        let value = h + 1;
        let score = (h * self.score_bins + self.score_bins) * 2;
        let tenpai = h * self.num_opponents + self.num_opponents;
        let grp = h * self.grp_classes + self.grp_classes;
        let opp_next = h * self.num_opponents + self.num_opponents;
        let danger = h * self.num_opponents + self.num_opponents;
        let oracle = h * 4 + 4;
        let belief_field = h * (self.num_belief_components * 4) + (self.num_belief_components * 4);
        let mixture_weight = h * self.num_belief_components + self.num_belief_components;
        let opponent_hand_type = h * (self.num_opponents * self.opponent_hand_type_classes)
            + (self.num_opponents * self.opponent_hand_type_classes);
        let delta_q = h * self.action_space + self.action_space;
        let safety_residual = h * self.action_space + self.action_space;
        backbone
            + policy
            + value
            + score
            + tenpai
            + grp
            + opp_next
            + danger
            + oracle
            + belief_field
            + mixture_weight
            + opponent_hand_type
            + delta_q
            + safety_residual
    }
}

/// Behavioral-cloning trainer hyperparameters and model shape.
#[derive(Config, Debug)]
pub struct BCTrainerConfig {
    pub model_config: ModelShapeConfig,
    #[config(default = "2.5e-4")]
    pub lr: f64,
    #[config(default = "1e-6")]
    pub min_learning_rate: f64,
    #[config(default = "2048")]
    pub batch_size: usize,
    #[config(default = "1.0")]
    pub grad_clip_norm: f32,
    #[config(default = "1e-5")]
    pub weight_decay: f32,
    #[config(default = "1000")]
    pub warmup_steps: usize,
}

impl BCTrainerConfig {
    pub fn summary(&self) -> String {
        format!(
            "lr={:.1e} min_lr={:.1e} batch={} clip={:.1} wd={:.1e}",
            self.lr,
            self.min_learning_rate,
            self.batch_size,
            self.grad_clip_norm,
            self.weight_decay
        )
    }

    pub fn default_actor() -> Self {
        Self::new(ModelShapeConfig::actor())
    }

    pub fn default_learner() -> Self {
        Self::new(ModelShapeConfig::learner())
    }

    pub fn estimated_training_time_hours(&self, num_samples: usize, samples_per_sec: f32) -> f32 {
        if samples_per_sec <= 0.0 {
            return 0.0;
        }
        num_samples as f32 / samples_per_sec / 3600.0
    }

    pub fn num_epochs_for(&self, num_samples: usize) -> usize {
        let batches = self.total_batches(num_samples);
        if batches == 0 {
            return 0;
        }
        batches
    }

    pub fn is_warmup(&self, step: usize) -> bool {
        step < self.warmup_steps
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.lr.is_finite() || !self.min_learning_rate.is_finite() {
            return Err("learning rates must be finite");
        }
        if !self.grad_clip_norm.is_finite() || !self.weight_decay.is_finite() {
            return Err("optimizer scalar values must be finite");
        }
        if self.lr <= 0.0 {
            return Err("lr must be positive");
        }
        if self.min_learning_rate <= 0.0 {
            return Err("min_learning_rate must be positive");
        }
        if self.min_learning_rate > self.lr {
            return Err("min_learning_rate must be <= lr");
        }
        if self.batch_size == 0 {
            return Err("batch_size must be > 0");
        }
        if self.grad_clip_norm <= 0.0 {
            return Err("grad_clip_norm must be positive");
        }
        if self.weight_decay < 0.0 {
            return Err("weight_decay must be non-negative");
        }
        if self.warmup_steps == 0 {
            return Err("warmup_steps must be > 0");
        }
        self.model_config.validate()
    }

    pub fn total_batches(&self, num_samples: usize) -> usize {
        if self.batch_size == 0 {
            return 0;
        }
        num_samples / self.batch_size
    }

    pub fn effective_lr(&self, step: usize, total_steps: usize) -> f64 {
        warmup_then_cosine_lr(
            step,
            self.warmup_steps,
            total_steps,
            self.lr,
            self.min_learning_rate,
        )
    }

    pub fn optimizer_config(&self) -> AdamConfig {
        AdamConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
                self.weight_decay,
            )))
            .with_grad_clipping(Some(GradientClippingConfig::Norm(self.grad_clip_norm)))
    }
}

/// RL step hyperparameters shared by train orchestration and runtime seams.
pub struct RlConfig {
    pub tau_drda: f32,
    pub ach_cfg: AchConfig,
    pub lr: f64,
    pub exit_weight: f32,
    pub aux_weight: f32,
    pub microbatch_size: Option<usize>,
}

impl RlConfig {
    pub fn default_phase2() -> Self {
        Self {
            tau_drda: 4.0,
            ach_cfg: AchConfig::new(),
            lr: 2.5e-4,
            exit_weight: DEFAULT_EXIT_WEIGHT,
            aux_weight: DEFAULT_AUX_WEIGHT,
            microbatch_size: None,
        }
    }

    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    pub fn with_exit_weight(mut self, w: f32) -> Self {
        self.exit_weight = w;
        self
    }

    pub fn with_aux_weight(mut self, w: f32) -> Self {
        self.aux_weight = w;
        self
    }

    pub fn default_phase3() -> Self {
        Self {
            tau_drda: 4.0,
            ach_cfg: AchConfig::new(),
            lr: 1e-4,
            exit_weight: DEFAULT_EXIT_WEIGHT,
            aux_weight: DEFAULT_AUX_WEIGHT,
            microbatch_size: None,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "rl(tau={:.1}, lr={:.1e}, exit_w={:.2}, aux_w={:.2})",
            self.tau_drda, self.lr, self.exit_weight, self.aux_weight
        )
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.tau_drda.is_finite()
            || !self.lr.is_finite()
            || !self.exit_weight.is_finite()
            || !self.aux_weight.is_finite()
        {
            return Err("rl config values must be finite");
        }
        if self.tau_drda < MIN_TAU_DRDA {
            return Err("tau_drda below minimum");
        }
        self.ach_cfg.validate()?;
        if self.lr <= 0.0 {
            return Err("lr must be positive");
        }
        if self.exit_weight < 0.0 || self.aux_weight < 0.0 {
            return Err("rl loss weights must be non-negative");
        }
        if self.microbatch_size == Some(0) {
            return Err("microbatch_size must be > 0");
        }
        Ok(())
    }

    pub fn effective_exit_weight(&self, phase: u8, progress: f32) -> f32 {
        match phase {
            0 | 1 => 0.0,
            2 => {
                let progress = progress.clamp(0.0, 1.0);
                if progress <= 0.5 {
                    0.0
                } else {
                    self.exit_weight * ((progress - 0.5) / 0.5)
                }
            }
            _ => self.exit_weight,
        }
    }
}

#[cfg(test)]
mod tests;
