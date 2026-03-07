<combined_run_record run_id="answer_9" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 9 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_9_VALIDATE_CONSERVATIVE_EXIT_AND_DELTA_Q.md">
  <![CDATA[# Prompt 9 — conservative exit and delta-q blueprint

<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest exact blueprint for conservative exit-policy supervision and delta-q supervision.

We want a detailed answer that makes clear:
- what the current codebase surfaces already support
- what is still missing or semantically unclosed
- whether conservative ExIt and delta-q are truly strong next moves or just neat-looking ideas
- what exact target objects, masks, teacher rules, and trust gates would make them credible
- what should stay narrow, what should stay deferred, and what should be rejected
- how to implement the surviving path with minimal guesswork

Use the artifacts below to derive your conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- include reasoning
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate it ourselves (pdfs, sources, links, similar projects)
- distinguish direct artifact support from your own inference
- use search/browse to find the original paper, then inspect the full PDF with skill; use abstracts or summaries only for discovery, not as the final evidence base
- use the bash tool to run Python for calculations, math checks, and validation when rigor matters
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs and prior handoffs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>

Artifact A — model output surface and parameter budget context:

```rust
//! Full HydraModel combining backbone and all output heads.

use burn::prelude::*;

use crate::backbone::{SEResNet, SEResNetConfig};
use crate::config::INPUT_CHANNELS;
use crate::heads::*;

pub struct HydraOutput<B: Backend> {
    pub policy_logits: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
    pub score_pdf: Tensor<B, 2>,
    pub score_cdf: Tensor<B, 2>,
    pub opp_tenpai: Tensor<B, 2>,
    pub grp: Tensor<B, 2>,
    pub opp_next_discard: Tensor<B, 3>,
    pub danger: Tensor<B, 3>,
    pub oracle_critic: Tensor<B, 2>,
    pub belief_fields: Tensor<B, 3>,
    pub mixture_weight_logits: Tensor<B, 2>,
    pub opponent_hand_type: Tensor<B, 2>,
    pub delta_q: Tensor<B, 2>,
    pub safety_residual: Tensor<B, 2>,
}

pub type ActorNet<B> = HydraModel<B>;
pub type LearnerNet<B> = HydraModel<B>;

impl<B: Backend> HydraOutput<B> {
    pub fn masked_policy(&self, legal_mask: Tensor<B, 2>) -> Tensor<B, 2> {
        let neg_inf = (legal_mask.ones_like() - legal_mask) * (-1e9f32);
        self.policy_logits.clone() + neg_inf
    }

    pub fn policy_logits_cpu(&self) -> Option<Vec<f32>> {
        self.policy_logits
            .to_data()
            .as_slice::<f32>()
            .ok()
            .map(|s| s.to_vec())
    }

    pub fn value_scalar(&self) -> Option<f32> {
        self.value
            .to_data()
            .as_slice::<f32>()
            .ok()
            .and_then(|s| s.first().copied())
    }

    pub fn is_finite(&self) -> bool {
        let check2 = |t: &Tensor<B, 2>| -> bool {
            if let Ok(s) = t.to_data().as_slice::<f32>() {
                s.iter().all(|v| v.is_finite())
            } else {
                false
            }
        };
        let check3 = |t: &Tensor<B, 3>| -> bool {
            if let Ok(s) = t.to_data().as_slice::<f32>() {
                s.iter().all(|v| v.is_finite())
            } else {
                false
            }
        };
        check2(&self.policy_logits)
            && check2(&self.value)
            && check2(&self.score_pdf)
            && check2(&self.score_cdf)
            && check2(&self.opp_tenpai)
            && check2(&self.grp)
            && check2(&self.oracle_critic)
            && check3(&self.opp_next_discard)
            && check3(&self.danger)
            && check3(&self.belief_fields)
            && check2(&self.mixture_weight_logits)
            && check2(&self.opponent_hand_type)
            && check2(&self.delta_q)
            && check2(&self.safety_residual)
    }
}

#[derive(Module, Debug)]
pub struct HydraModel<B: Backend> {
    backbone: SEResNet<B>,
    policy: PolicyHead<B>,
    value: ValueHead<B>,
    score_pdf: ScorePdfHead<B>,
    score_cdf: ScoreCdfHead<B>,
    opp_tenpai: OppTenpaiHead<B>,
    grp: GrpHead<B>,
    opp_next_discard: OppNextDiscardHead<B>,
    danger: DangerHead<B>,
    oracle_critic: OracleCriticHead<B>,
    belief_field: BeliefFieldHead<B>,
    mixture_weight: MixtureWeightHead<B>,
    opponent_hand_type: OpponentHandTypeHead<B>,
    delta_q: DeltaQHead<B>,
    safety_residual: SafetyResidualHead<B>,
}

#[derive(Config, Debug)]
pub struct HydraModelConfig {
    pub num_blocks: usize,
    #[config(default = "192")]
    pub input_channels: usize,
    #[config(default = "256")]
    pub hidden_channels: usize,
    #[config(default = "32")]
    pub num_groups: usize,
    #[config(default = "64")]
    pub se_bottleneck: usize,
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

impl HydraModelConfig {
    pub fn summary(&self) -> String {
        let kind = if self.num_blocks <= 12 {
            "actor"
        } else {
            "learner"
        };
        format!(
            "{}(blocks={}, ch={})",
            kind, self.num_blocks, self.hidden_channels
        )
    }

    pub fn is_actor(&self) -> bool {
        self.num_blocks == 12
    }
    pub fn is_learner(&self) -> bool {
        self.num_blocks == 24
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.num_groups == 0 || !self.hidden_channels.is_multiple_of(self.num_groups) {
            return Err("hidden_channels must be divisible by num_groups");
        }
        if self.num_blocks == 0 {
            return Err("num_blocks must be > 0");
        }
        if self.se_bottleneck == 0 {
            return Err("se_bottleneck must be > 0");
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
        Self::new(12).with_input_channels(INPUT_CHANNELS)
    }

    pub fn estimated_params(&self) -> usize {
        let h = self.hidden_channels;
        let se_b = self.se_bottleneck;
        let input_conv = self.input_channels * h * 3 + h;
        let gn = h * 2;
        let block = (h * h * 3 + h) * 2 + gn * 2 + (h * se_b + se_b) + (se_b * h + h);
        let backbone = input_conv + gn + block * self.num_blocks + gn;
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

    pub fn learner() -> Self {
        Self::new(24).with_input_channels(INPUT_CHANNELS)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> HydraModel<B> {
        let backbone_cfg = SEResNetConfig::new(
            self.num_blocks,
            self.input_channels,
            self.hidden_channels,
            self.num_groups,
            self.se_bottleneck,
        );
        let heads_cfg = HeadsConfig::new()
            .with_hidden_channels(self.hidden_channels)
            .with_action_space(self.action_space)
            .with_score_bins(self.score_bins)
            .with_num_opponents(self.num_opponents)
            .with_grp_classes(self.grp_classes)
            .with_num_belief_components(self.num_belief_components)
            .with_opponent_hand_type_classes(self.opponent_hand_type_classes);
        HydraModel {
            backbone: backbone_cfg.init(device),
            policy: heads_cfg.init_policy(device),
            value: heads_cfg.init_value(device),
            score_pdf: heads_cfg.init_score_pdf(device),
            score_cdf: heads_cfg.init_score_cdf(device),
            opp_tenpai: heads_cfg.init_opp_tenpai(device),
            grp: heads_cfg.init_grp(device),
            opp_next_discard: heads_cfg.init_opp_next_discard(device),
            danger: heads_cfg.init_danger(device),
            oracle_critic: heads_cfg.init_oracle_critic(device),
            belief_field: heads_cfg.init_belief_field(device),
            mixture_weight: heads_cfg.init_mixture_weight(device),
            opponent_hand_type: heads_cfg.init_opponent_hand_type(device),
            delta_q: heads_cfg.init_delta_q(device),
            safety_residual: heads_cfg.init_safety_residual(device),
        }
    }
}

impl<B: Backend> HydraModel<B> {
    pub fn policy_logits_for(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let (_, pooled) = self.backbone.forward(x);
        self.policy.forward(pooled)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> HydraOutput<B> {
        let (spatial, pooled) = self.backbone.forward(x);
        let oracle_input = pooled.clone().detach();
        HydraOutput {
            policy_logits: self.policy.forward(pooled.clone()),
            value: self.value.forward(pooled.clone()),
            score_pdf: self.score_pdf.forward(pooled.clone()),
            score_cdf: self.score_cdf.forward(pooled.clone()),
            opp_tenpai: self.opp_tenpai.forward(pooled.clone()),
            grp: self.grp.forward(pooled.clone()),
            opp_next_discard: self.opp_next_discard.forward(spatial.clone()),
            danger: self.danger.forward(spatial.clone()),
            oracle_critic: self.oracle_critic.forward(oracle_input),
            belief_fields: self.belief_field.forward(spatial),
            mixture_weight_logits: self.mixture_weight.forward(pooled.clone()),
            opponent_hand_type: self.opponent_hand_type.forward(pooled.clone()),
            delta_q: self.delta_q.forward(pooled.clone()),
            safety_residual: self.safety_residual.forward(pooled),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::backend::NdArray;

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
        let x = Tensor::<B, 3>::zeros([4, INPUT_CHANNELS, 34], &device);
        let out = model.forward(x);
        assert_output_shapes(&out, 4);
    }

    #[test]
    fn learner_net_all_output_shapes() {
        let device = Default::default();
        let model = HydraModelConfig::learner().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([2, INPUT_CHANNELS, 34], &device);
        let out = model.forward(x);
        assert_output_shapes(&out, 2);
    }
}
```

Artifact B — loss surface, optional targets, and target-dependent activation behavior:

```rust
//! Loss functions for all 9 heads + total weighted loss.

use burn::prelude::*;
use burn::tensor::activation;
use std::marker::PhantomData;

use crate::model::HydraOutput;

#[derive(Clone)]
pub struct HydraTargets<B: Backend> {
    pub policy_target: Tensor<B, 2>,
    pub legal_mask: Tensor<B, 2>,
    pub value_target: Tensor<B, 1>,
    pub grp_target: Tensor<B, 2>,
    pub tenpai_target: Tensor<B, 2>,
    pub danger_target: Tensor<B, 3>,
    pub danger_mask: Tensor<B, 3>,
    pub opp_next_target: Tensor<B, 3>,
    pub score_pdf_target: Tensor<B, 2>,
    pub score_cdf_target: Tensor<B, 2>,
    pub oracle_target: Option<Tensor<B, 2>>,
    pub belief_fields_target: Option<Tensor<B, 3>>,
    pub belief_fields_mask: Option<Tensor<B, 1>>,
    pub mixture_weight_target: Option<Tensor<B, 2>>,
    pub mixture_weight_mask: Option<Tensor<B, 1>>,
    pub opponent_hand_type_target: Option<Tensor<B, 2>>,
    pub delta_q_target: Option<Tensor<B, 2>>,
    pub safety_residual_target: Option<Tensor<B, 2>>,
    pub safety_residual_mask: Option<Tensor<B, 2>>,
    pub oracle_guidance_mask: Option<Tensor<B, 1>>,
}

#[derive(Config, Debug)]
pub struct HydraLossConfig {
    #[config(default = "1.0")]
    pub w_pi: f32,
    #[config(default = "0.5")]
    pub w_v: f32,
    #[config(default = "0.2")]
    pub w_grp: f32,
    #[config(default = "0.1")]
    pub w_tenpai: f32,
    #[config(default = "0.1")]
    pub w_danger: f32,
    #[config(default = "0.1")]
    pub w_opp: f32,
    #[config(default = "0.025")]
    pub w_score: f32,
    #[config(default = "0.0")]
    pub w_oracle_critic: f32,
    #[config(default = "0.0")]
    pub w_belief_fields: f32,
    #[config(default = "0.0")]
    pub w_mixture_weight: f32,
    #[config(default = "0.0")]
    pub w_opponent_hand_type: f32,
    #[config(default = "0.0")]
    pub w_delta_q: f32,
    #[config(default = "0.0")]
    pub w_safety_residual: f32,
}

impl HydraLossConfig {
    pub fn total_weight(&self) -> f32 {
        self.w_pi
            + self.w_v
            + self.w_grp
            + self.w_tenpai
            + self.w_danger
            + self.w_opp
            + self.w_score * 2.0
            + self.w_oracle_critic
            + self.w_belief_fields
            + self.w_mixture_weight
            + self.w_opponent_hand_type
            + self.w_delta_q
            + self.w_safety_residual
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.w_pi < 0.0
            || self.w_v < 0.0
            || self.w_grp < 0.0
            || self.w_tenpai < 0.0
            || self.w_danger < 0.0
            || self.w_opp < 0.0
            || self.w_score < 0.0
            || self.w_oracle_critic < 0.0
            || self.w_belief_fields < 0.0
            || self.w_mixture_weight < 0.0
            || self.w_opponent_hand_type < 0.0
            || self.w_delta_q < 0.0
            || self.w_safety_residual < 0.0
        {
            return Err("loss weights must be non-negative");
        }
        Ok(())
    }
}

pub fn dense_regression_mse<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1> {
    let diff = pred - target;
    (diff.clone() * diff).mean() * 0.5
}

pub fn masked_action_mse<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let diff = pred - target;
    let sq = diff.clone() * diff * 0.5;
    let masked = sq * mask.clone();
    let denom = mask.sum().into_scalar().elem::<f32>().max(1.0);
    masked.sum() / denom
}

pub fn soft_target_from_exit<B: Backend>(
    model_logits: Tensor<B, 2>,
    exit_target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    mix: f32,
) -> Tensor<B, 2> {
    let model_probs = burn::tensor::activation::softmax(
        model_logits + (mask.ones_like() - mask.clone()) * (-1e9f32),
        1,
    );
    model_probs * (1.0 - mix) + exit_target * mix
}

fn masked_mean<B: Backend>(per_sample: Tensor<B, 1>, mask: Option<Tensor<B, 1>>) -> Tensor<B, 1> {
    match mask {
        Some(mask) => {
            let denom = mask.clone().sum().clamp_min(1.0);
            (per_sample * mask).sum() / denom
        }
        None => per_sample.mean(),
    }
}

pub struct LossBreakdown<B: Backend> {
    pub policy: Tensor<B, 1>,
    pub value: Tensor<B, 1>,
    pub grp: Tensor<B, 1>,
    pub tenpai: Tensor<B, 1>,
    pub danger: Tensor<B, 1>,
    pub opp_next: Tensor<B, 1>,
    pub score_pdf: Tensor<B, 1>,
    pub score_cdf: Tensor<B, 1>,
    pub oracle_critic: Tensor<B, 1>,
    pub belief_fields: Tensor<B, 1>,
    pub mixture_weight: Tensor<B, 1>,
    pub opponent_hand_type: Tensor<B, 1>,
    pub delta_q: Tensor<B, 1>,
    pub safety_residual: Tensor<B, 1>,
    pub total: Tensor<B, 1>,
}

impl<B: Backend> HydraLoss<B> {
    pub fn total_loss(
        &self,
        outputs: &HydraOutput<B>,
        targets: &HydraTargets<B>,
    ) -> LossBreakdown<B> {
        let oracle_mask = targets.oracle_guidance_mask.clone();
        let zero = outputs.value.clone().sum() * 0.0;
        let l_hand_type = match &targets.opponent_hand_type_target {
            Some(target) => masked_mean(
                opponent_hand_type_ce_per_sample(
                    outputs.opponent_hand_type.clone(),
                    target.clone(),
                ),
                oracle_mask.clone(),
            ),
            None => zero.clone(),
        };
        let l_delta_q = match &targets.delta_q_target {
            Some(target) => dense_regression_mse(outputs.delta_q.clone(), target.clone()),
            None => zero.clone(),
        };
        let l_safety_residual = match (
            &targets.safety_residual_target,
            &targets.safety_residual_mask,
        ) {
            (Some(target), Some(mask)) => masked_action_mse(
                outputs.safety_residual.clone(),
                target.clone(),
                mask.clone(),
            ),
            _ => zero.clone(),
        };
        let c = &self.config;
        let total = l_delta_q.clone() * c.w_delta_q
            + l_safety_residual.clone() * c.w_safety_residual
            + l_hand_type.clone() * c.w_opponent_hand_type;
        LossBreakdown {
            policy: zero.clone(),
            value: zero.clone(),
            grp: zero.clone(),
            tenpai: zero.clone(),
            danger: zero.clone(),
            opp_next: zero.clone(),
            score_pdf: zero.clone(),
            score_cdf: zero.clone(),
            oracle_critic: zero.clone(),
            belief_fields: zero.clone(),
            mixture_weight: zero.clone(),
            opponent_hand_type: l_hand_type,
            delta_q: l_delta_q,
            safety_residual: l_safety_residual,
            total,
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::model::HydraModelConfig;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_optional_belief_losses_default_to_zero() {
        ...
        assert!(hand_type.abs() < 1e-8);
        assert!(delta_q.abs() < 1e-8);
        assert!(safety_residual.abs() < 1e-8);
    }

    #[test]
    fn test_optional_belief_losses_activate_when_targets_present() {
        ...
        targets.delta_q_target = Some(Tensor::<B, 2>::zeros([2, 46], &device));
        targets.safety_residual_target = Some(Tensor::<B, 2>::zeros([2, 46], &device));
        targets.safety_residual_mask = Some(Tensor::<B, 2>::ones([2, 46], &device));
        ...
        assert!(delta_q.is_finite() && delta_q >= 0.0);
        assert!(safety_residual.is_finite() && safety_residual >= 0.0);
    }

    #[test]
    fn test_safety_residual_requires_mask() {
        ...
        targets.safety_residual_target = Some(Tensor::<B, 2>::ones([2, 46], &device));
        let breakdown = loss_fn.total_loss(&outputs, &targets);
        let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
        assert!(safety_residual.abs() < 1e-8);
    }
}
```

Artifact C — sample/batch surfaces and current advanced-target presence behavior:

```rust
//! MjaiSample struct, GRP label construction, and batch collation.

pub struct MjaiSample {
    pub obs: [f32; OBS_SIZE],
    pub action: u8,
    pub legal_mask: [f32; HYDRA_ACTION_SPACE],
    pub placement: u8,
    pub score_delta: i32,
    pub grp_label: u8,
    pub oracle_target: Option<[f32; 4]>,
    pub tenpai: [f32; 3],
    pub opp_next: [u8; 3],
    pub danger: [f32; 102],
    pub danger_mask: [f32; 102],
    pub safety_residual: Option<[f32; HYDRA_ACTION_SPACE]>,
    pub safety_residual_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
    pub belief_fields: Option<[f32; 16 * 34]>,
    pub mixture_weights: Option<[f32; 4]>,
    pub belief_fields_present: bool,
    pub mixture_weights_present: bool,
}

pub struct MjaiBatch<B: Backend> {
    pub obs: Tensor<B, 3>,
    pub actions: Tensor<B, 1, Int>,
    pub legal_mask: Tensor<B, 2>,
    pub value_target: Tensor<B, 1>,
    pub grp_target: Tensor<B, 2>,
    pub oracle_target: Option<Tensor<B, 2>>,
    pub oracle_target_mask: Tensor<B, 1>,
    pub tenpai_target: Tensor<B, 2>,
    pub danger_target: Tensor<B, 3>,
    pub danger_mask: Tensor<B, 3>,
    pub safety_residual_target: Option<Tensor<B, 2>>,
    pub safety_residual_mask: Option<Tensor<B, 2>>,
    pub belief_fields_target: Option<Tensor<B, 3>>,
    pub mixture_weight_target: Option<Tensor<B, 2>>,
    pub belief_fields_mask: Option<Tensor<B, 1>>,
    pub mixture_weight_mask: Option<Tensor<B, 1>>,
    pub opp_next_target: Tensor<B, 3>,
    pub score_pdf_target: Tensor<B, 2>,
    pub score_cdf_target: Tensor<B, 2>,
}

impl<B: Backend> MjaiBatch<B> {
    pub fn into_hydra_targets(self) -> HydraTargets<B> {
        let batch = self.actions.dims()[0];
        let policy_target = self
            .actions
            .clone()
            .one_hot::<2>(46)
            .reshape([batch, 46])
            .float();
        HydraTargets {
            policy_target,
            legal_mask: self.legal_mask,
            value_target: self.value_target,
            grp_target: self.grp_target,
            tenpai_target: self.tenpai_target,
            danger_target: self.danger_target,
            danger_mask: self.danger_mask,
            safety_residual_target: self.safety_residual_target,
            opp_next_target: self.opp_next_target,
            score_pdf_target: self.score_pdf_target,
            score_cdf_target: self.score_cdf_target,
            oracle_target: self.oracle_target,
            belief_fields_target: self.belief_fields_target,
            mixture_weight_target: self.mixture_weight_target,
            opponent_hand_type_target: None,
            delta_q_target: None,
            safety_residual_mask: self.safety_residual_mask,
            belief_fields_mask: self.belief_fields_mask,
            mixture_weight_mask: self.mixture_weight_mask,
            oracle_guidance_mask: Some(self.oracle_target_mask),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn batch_to_hydra_targets_keeps_optional_advanced_targets_narrow() {
        ...
        assert!(targets.oracle_target.is_some());
        assert!(targets.belief_fields_target.is_none());
        assert!(targets.mixture_weight_target.is_none());
        assert!(targets.opponent_hand_type_target.is_none());
        assert!(targets.delta_q_target.is_none());
        assert!(targets.safety_residual_target.is_none());
    }

    #[test]
    fn batch_to_hydra_targets_carries_safety_residual() {
        ...
        assert_eq!(sr.dims(), [1, HYDRA_ACTION_SPACE]);
        assert_eq!(srm.dims(), [1, HYDRA_ACTION_SPACE]);
    }

    #[test]
    fn batch_to_hydra_targets_carries_projected_belief_targets() {
        ...
        assert_eq!(belief_target.dims(), [1, 16, 34]);
        assert_eq!(mix_target.dims(), [1, 4]);
    }
}
```

Artifact D — loader-built targets and provenance-producing paths:

```rust
fn build_safety_residual_targets(
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
    safety: &SafetyInfo,
    wait_sets: &[[f32; 34]; 3],
) -> ([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE]) {
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    for action in 0..=DISCARD_END {
        let action_idx = action as usize;
        if legal_mask[action_idx] <= 0.0 {
            continue;
        }
        let tile = match action {
            AKA_5M => 4,
            AKA_5P => 13,
            AKA_5S => 22,
            _ => action,
        };
        let public_score = public_safety_score(safety, tile);
        let exact_risk = exact_dealin_risk_from_waits(wait_sets, tile);
        target[action_idx] = (public_score - exact_risk).clamp(0.0, 1.0);
        mask[action_idx] = 1.0;
    }
    (target, mask)
}

fn build_stage_a_belief_targets(
    state: &GameState,
    actor: usize,
    obs: &riichienv_core::observation::Observation,
) -> (Option<[f32; 16 * 34]>, Option<[f32; 4]>, bool, bool) {
    let hand = hydra_core::bridge::extract_hand(obs);
    let discards = hydra_core::bridge::extract_discards(obs);
    let melds = hydra_core::bridge::extract_melds(obs);
    let dora = hydra_core::bridge::extract_dora(obs);
    let remaining = extract_public_remaining_counts(&hand, &discards, &melds, &dora);
    let hidden_tiles = state
        .players
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != actor)
        .map(|(_, p)| p.hand_len as usize)
        .sum::<usize>()
        + state.wall.remaining();
    let target = build_stage_a_teacher(&remaining, hidden_tiles, StageABeliefConfig::default());
    match target {
        Some(target) => (
            Some(target.belief_fields),
            target.mixture_weights,
            true,
            target.mixture_weights.is_some(),
        ),
        None => (None, None, false, false),
    }
}

samples.push(MjaiSample {
    ...
    safety_residual: Some(safety_residual),
    safety_residual_mask: Some(safety_residual_mask),
    belief_fields,
    mixture_weights,
    belief_fields_present,
    mixture_weights_present,
});
```

```rust
#[test]
fn load_game_from_reader_populates_safety_residual_for_discards_only() {
    let sample = game.samples.iter().find(|s| s.action <= DISCARD_END).expect("discard sample");
    let target = sample.safety_residual.expect("safety residual target");
    let mask = sample.safety_residual_mask.expect("safety residual mask");
    let masked_discards: f32 = mask[..=DISCARD_END as usize].iter().sum();
    assert!(masked_discards > 0.0);
    let masked_non_discards: f32 = mask[(DISCARD_END as usize + 1)..].iter().sum();
    assert!(masked_non_discards.abs() < 1e-6);
}

#[test]
fn load_game_from_reader_can_emit_stage_a_belief_targets() {
    let sample = game
        .samples
        .iter()
        .find(|s| s.belief_fields.is_some())
        .expect("expected at least one belief-target sample");
    let belief = sample.belief_fields.expect("belief fields");
    assert_eq!(belief.len(), 16 * 34);
    assert!(sample.belief_fields_present);
}
```

Artifact E — AFBS root outputs, pondering seam, and bridge-side search features:

```rust
pub struct AfbsNode {
    pub info_state_hash: u64,
    pub visit_count: u32,
    pub total_value: f64,
    pub prior: f32,
    pub children: ChildList,
    pub is_opponent: bool,
    pub particle_handle: Option<u32>,
}

impl AfbsNode {
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            return 0.0;
        }
        (self.total_value / self.visit_count as f64) as f32
    }
}

impl AfbsTree {
    pub fn root_exit_policy(&self, root_idx: NodeIdx, tau: f32) -> [f32; HYDRA_ACTION_SPACE] {
        ...
    }

    pub fn root_visit_count(&self, root: NodeIdx) -> u32 {
        self.nodes.get(root as usize).map_or(0, |n| n.visit_count)
    }

    pub fn node_q_value(&self, node: NodeIdx) -> f32 {
        self.nodes.get(node as usize).map_or(0.0, |n| n.q_value())
    }

    pub fn find_child_by_action(&self, parent_idx: NodeIdx, action: u8) -> Option<NodeIdx> {
        ...
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PonderResult {
    pub exit_policy: [f32; HYDRA_ACTION_SPACE],
    pub value: f32,
    pub search_depth: u8,
    pub visit_count: u32,
    pub timestamp: Instant,
}
```

```rust
/// Current sources:
/// - Mixture-SIB -> belief fields, weights, entropy, ESS
/// - AFBS root -> discard-level delta-Q summary for expanded discard actions
/// - safety/opponent model cache -> per-opponent stress and matagi danger fallback
/// - explicit robust risk/stress overrides when provided
pub fn build_search_features(
    safety: &SafetyInfo,
    context: &SearchContext<'_>,
) -> SearchFeaturePlanes {
    let mut features = SearchFeaturePlanes::default();

    if let (Some(tree), Some(root)) = (context.afbs_tree, context.afbs_root) {
        let root_q = tree.node_q_value(root);
        let mut any_delta_q = false;
        for action in 0..NUM_TILE_TYPES as u8 {
            if let Some(child) = tree.find_child_by_action(root, action) {
                features.delta_q[action as usize] = tree.node_q_value(child) - root_q;
                any_delta_q = true;
            }
        }
        if any_delta_q {
            features.search_features_present = true;
            features.context_features_present = true;
        }
    }
    features
}
```

Artifact F — combined-handoff and doctrine excerpts directly relevant to conservative ExIt and delta-q:

```text
The best immediate tranche is not a broader AFBS push, not Hand-EV realism, and not a belief-teacher program. It is a narrow advanced-target generation + activation tranche centered on sample.rs, mjai_loader.rs, losses.rs, bc.rs, and rl.rs, with only minimal support from existing AFBS/bridge outputs.
```

```text
The three active targets in this tranche:
* exit_target
* delta_q_target
* safety_residual_target
Everything else stays structurally present in the codebase but inactive unless a later tranche adds credible labels.
```

```text
Produced:
* exit_target
* delta_q_target
* safety_residual_target

Structurally present but deliberately not produced:
* oracle_v
* belief_fields_target
* mixture_weight_target
* opponent_hand_type_target
```

```text
The first coding tranche is about target-generation / supervision closure, which is more robust to optimizer-level uncertainty.
```

```text
Stronger target generation is a better immediate lever than a giant search rewrite.
AFBS should be selective and specialist, not the default path everywhere.
```

```text
Preferred order:
- first: safety_residual_target, delta_q_target, and any replay-credible target that can be computed without new search infra
- later in same tranche if credible: belief_fields_target, mixture_weight_target, opponent_hand_type_target
```

Artifact G — more AFBS tests and PonderResult artifacts:

```rust
#[test]
fn ponder_result_from_tree_reflects_root_stats() {
    let mut tree = AfbsTree::new();
    let root = tree.add_node(10, 1.0, false);
    let c0 = tree.add_node(11, 0.7, false);
    let c1 = tree.add_node(12, 0.3, false);
    tree.nodes[root as usize].children = smallvec::smallvec![(2, c0), (4, c1)];
    tree.nodes[root as usize].visit_count = 9;
    tree.nodes[c0 as usize].visit_count = 6;
    tree.nodes[c0 as usize].total_value = 3.0;
    tree.nodes[c1 as usize].visit_count = 3;
    tree.nodes[c1 as usize].total_value = 2.4;

    let result = PonderResult::from_tree(&tree, root, 0.42, 1.0);
    assert_eq!(result.visit_count, 9);
    assert_eq!(result.search_depth, 1);
    assert!((result.value - 0.42).abs() < 1e-6);
    let sum: f32 = result.exit_policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn exit_policy_sums_to_one() {
    let mut tree = AfbsTree::new();
    let root = tree.add_node(0, 1.0, false);
    let c0 = tree.add_node(1, 0.5, false);
    let c1 = tree.add_node(2, 0.3, false);
    let c2 = tree.add_node(3, 0.2, false);
    tree.nodes[root as usize].children = smallvec::smallvec![(0, c0), (5, c1), (10, c2)];
    tree.nodes[c0 as usize].visit_count = 10;
    tree.nodes[c0 as usize].total_value = 5.0;
    tree.nodes[c1 as usize].visit_count = 5;
    tree.nodes[c1 as usize].total_value = 3.0;
    tree.nodes[c2 as usize].visit_count = 3;
    tree.nodes[c2 as usize].total_value = 0.9;
    let policy = tree.root_exit_policy(root, 1.0);
    let sum: f32 = policy.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "exit policy sum: {sum}");
}

#[test]
fn exit_policy_with_zero_tau_becomes_argmax() {
    let mut tree = AfbsTree::new();
    let root = tree.add_node(0, 1.0, false);
    let c0 = tree.add_node(1, 0.5, false);
    let c1 = tree.add_node(2, 0.5, false);
    tree.nodes[root as usize].children = smallvec::smallvec![(3, c0), (7, c1)];
    tree.nodes[c0 as usize].visit_count = 2;
    tree.nodes[c0 as usize].total_value = 1.0;
    tree.nodes[c1 as usize].visit_count = 2;
    tree.nodes[c1 as usize].total_value = 3.0;
    let policy = tree.root_exit_policy(root, 0.0);
    assert_eq!(policy[7], 1.0);
}
```

Artifact H — more sample/target and loader artifacts:

```rust
#[test]
fn batch_to_hydra_targets_policy_matches_actions() {
    let samples = vec![dummy_sample(2, 0), dummy_sample(7, 0)];
    let batch = collate_batch::<B>(&samples, &device);
    let targets = batch.into_hydra_targets();
    assert_eq!(targets.policy_target.dims(), [2, 46]);
}

#[test]
fn batch_to_hydra_targets_keeps_oracle_absent_when_missing() {
    let batch = collate_batch::<B>(&[dummy_sample(3, 0)], &device);
    assert!(batch.oracle_target.is_none());
    let targets = batch.into_hydra_targets();
    assert!(targets.oracle_target.is_none());
}

#[test]
fn batch_to_hydra_targets_carries_safety_residual() {
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    target[0] = 0.4;
    target[34] = 0.7;
    mask[0] = 1.0;
    mask[34] = 1.0;
    sample.safety_residual = Some(target);
    sample.safety_residual_mask = Some(mask);
    let batch = collate_batch::<B>(&[sample], &device);
    let targets = batch.into_hydra_targets();
    let sr = targets.safety_residual_target.expect("safety residual target");
    let srm = targets.safety_residual_mask.expect("safety residual mask");
    assert_eq!(sr.dims(), [1, HYDRA_ACTION_SPACE]);
    assert_eq!(srm.dims(), [1, HYDRA_ACTION_SPACE]);
}
```

```rust
#[test]
fn load_game_from_reader_populates_safety_residual_for_discards_only() {
    let sample = game.samples.iter().find(|s| s.action <= DISCARD_END).expect("discard sample");
    let mask = sample.safety_residual_mask.expect("safety residual mask");
    let masked_discards: f32 = mask[..=DISCARD_END as usize].iter().sum();
    assert!(masked_discards > 0.0);
    let masked_non_discards: f32 = mask[(DISCARD_END as usize + 1)..].iter().sum();
    assert!(masked_non_discards.abs() < 1e-6);
}
```

Artifact I — more doctrine excerpts tied to this lane:

```text
Keep the rollout narrow.
Prefer ExIt target + delta-Q + safety-residual activation first.
Bring belief-field / mixture / hand-type targets online only where labels are credible.
```

```text
No broad AFBS rewrite.
No new heads.
No broad public-belief search.
```

```text
The smallest credible tranche is one optional advanced-target carrier, one activation policy, and exactly three live targets: ExIt, delta-Q, safety residual.
```

</artifacts>
]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_9.md">
  <![CDATA[# Conservative ExIt + delta-q supervision blueprint

## 0. Commit this lane, and keep it narrow

Ship exactly three live advanced targets:

* `exit_target` as **conservative policy distillation into the existing policy head**
* `delta_q_target` as **masked auxiliary regression into the existing `delta_q` head**
* `safety_residual_target` as **the already-produced masked auxiliary regression into the existing `safety_residual` head**

Do **not** add heads. Do **not** broaden AFBS. Do **not** turn on belief fields, mixture weights, opponent hand type, or oracle critic in this tranche.

Why this survives review:

ExIt’s original paper supports soft tree-policy imitation and explicitly argues that softer, cost-sensitive targets are better than chosen-action one-hots; in their Hex setup, tree-policy targets beat chosen-action targets by about 50 Elo even with similar move-prediction error. AlphaZero likewise trains the policy to match search probabilities rather than hard best-action labels. But later work on Gumbel policy improvement shows that plain visit-count policy updates can fail to improve the policy when search covers only a small subset of actions or when the simulation budget is small, which is exactly why the ExIt path here must be conservative and masked rather than global. AWR, AWAC, and CPI-style results point the same way: advantage-like signals are useful, but only when updates are constrained and semantically well-defined. ([NeurIPS Papers][1])

## 1. What the current surfaces already support

### Direct artifact support

From the artifacts alone:

* The model already has `policy_logits`, `delta_q`, and `safety_residual` outputs in the same backbone pass.
* No new parameterized object is needed for ExIt: it reuses `policy_logits`.
* `delta_q` and `safety_residual` heads already exist in both actor and learner configs.
* `soft_target_from_exit(...)` already exists in `losses.rs`.
* `safety_residual_target` is already produced by the loader, batched, carried into `HydraTargets`, masked to discard actions, and has tests.
* AFBS already surfaces enough raw ingredients to build search-derived targets narrowly:

  * root visit count
  * node q values
  * child lookup by action
  * `PonderResult.exit_policy`
* Bridge-side search features already derive a discard-level `delta_q` summary from AFBS root children.

### Parameter budget reality

Using Artifact A’s default sizes (`hidden_channels=256`, `action_space=46`):

* policy head params: `256*46 + 46 = 11,822`
* delta_q head params: `11,822`
* safety_residual head params: `11,822`

So the two already-present auxiliary heads together are only about `23,644` params:

* about `0.44%` of the default actor
* about `0.23%` of the default learner

ExIt adds **zero** params because it uses the existing policy head.

## 2. What is missing or semantically unclosed

These are the blockers that must be closed before the new supervision is credible.

### 2.1 `exit_target` is not actually carried anywhere

Artifacts B/C/D show:

* no `exit_target` in `MjaiSample`
* no `exit_target` in `MjaiBatch`
* no `exit_target` in `HydraTargets`
* no policy loss path that consumes an exit target

So ExIt is only structurally hinted at by `soft_target_from_exit`, not closed end-to-end.

### 2.2 `delta_q_target` exists as a head/loss placeholder but not as a real masked target

Artifacts B/C show:

* `HydraTargets.delta_q_target: Option<Tensor<B,2>>` exists
* but `MjaiBatch::into_hydra_targets()` sets it to `None`
* there is **no `delta_q_mask`**
* the current loss uses dense MSE, which would be wrong for sparse AFBS-derived support

That means the present `delta_q` path is not credible yet.

### 2.3 AFBS action semantics are narrower than Hydra action semantics

`build_search_features()` only iterates `0..NUM_TILE_TYPES` and builds discard-level delta-q for expanded discard actions.

That implies, at minimum:

* no full 46-action closure
* no proof that riichi/kan/tsumo/call actions are covered
* likely no closure for red-five discard identity either

Therefore both ExIt and delta-q must stay **discard-only** in this tranche.

### 2.4 Q-value perspective is not semantically closed

Artifact E exposes:

* `AfbsNode { is_opponent, total_value, visit_count }`
* `q_value()` = `total_value / visit_count`
* `build_search_features()` uses `child_q - root_q` directly

But the artifacts do **not** prove whether `q_value()` is already in root-player perspective or node-to-move perspective. Because `is_opponent` exists and `q_value()` ignores it, this semantic contract is unclosed.

Do not train `delta_q` until AFBS exposes an explicit “q from root-player perspective” API or an explicit test proves the current contract.

### 2.5 `root_exit_policy()` is not proven to be the exact teacher you want

The tests prove:

* it sums to 1
* `tau=0` becomes argmax on the best child

They do **not** prove whether `tau=1` equals normalized visit counts, q-softmax, or something mixed.

So do not blindly inherit `root_exit_policy()` as the ExIt teacher. Build the teacher explicitly from child visit counts unless and until a test closes that contract.

### 2.6 The loss artifact is internally inconsistent

Artifact B advertises a full Hydra loss surface, but the shown `total_loss()` only accumulates optional advanced losses and returns zero for the base policy/value/etc. That excerpt may be partial, stale, or wrong.

Before enabling any advanced supervision, add a hard test proving the active branch still produces non-zero base policy loss.

### 2.7 Possible target leakage audit is required

Artifact E shows bridge-side search features already include discard-level `delta_q`. If those same planes are present in learner observations for the same examples, then supervising `delta_q` would be partially teaching the head to copy its input.

Audit `OBS_SIZE` / channel packing before activation:

* if `delta_q` planes are already in the learner input for those samples, either zero those planes when `delta_q_target` is present, or do not train the `delta_q` head on those samples.

## 3. Final decision on the three ideas

### `safety_residual`: **strong next move now**

Keep it. It is already replay-credible, already masked, already produced, and already tested.

### `delta_q`: **strong next move only as a masked auxiliary head**

Keep it, but only after:

* root-perspective q semantics are closed
* sparse mask support is added
* action-family scope is narrowed to compatible discard states

Do **not** use it yet as a policy-improvement weight, action reranker, or decision override.

### conservative ExIt: **strong next move only as masked, partial-policy distillation**

Keep it, but only on states where the search teacher is compatible with the action family and has enough coverage.

Do **not** replace the base BC policy loss with ExIt.
Do **not** apply ExIt on all legal actions.
Do **not** use dense search zeros to suppress unsearched legal actions.

## 4. Exact target objects

## 4.1 Search-compatible state definition

Use search-derived `exit_target` and `delta_q_target` **only** on states satisfying:

```text
compatible_discard_state(s) :=
    all legal actions are discard-family actions
    AND no aka-discard ambiguity is present
```

Concretely:

```rust
fn compatible_discard_state(legal_mask: &[f32; HYDRA_ACTION_SPACE]) -> bool {
    let non_discard_legal =
        legal_mask[(DISCARD_END as usize + 1)..].iter().any(|&x| x > 0.0);
    if non_discard_legal {
        return false;
    }

    // Conservative rule for this tranche:
    // do not use search-derived ExIt/delta-q when red-five discard identity matters.
    let aka_legal =
        legal_mask[AKA_5M as usize] > 0.0 ||
        legal_mask[AKA_5P as usize] > 0.0 ||
        legal_mask[AKA_5S as usize] > 0.0;
    if aka_legal {
        return false;
    }

    true
}
```

Reason for the aka skip: safety residual can map aka to base tile risk because post-discard danger depends on the tile class. ExIt and delta-q cannot safely do that, because discarding red vs non-red five can change future value.

This is intentionally narrow. Relax it only after AFBS can surface full 46-action child stats.

## 4.2 `exit_target`

Teacher object:

[
t_{\text{exit}}(a \mid s) =
\begin{cases}
\frac{N(s,a)}{\sum_{b \in C(s)} N(s,b)} & a \in C(s) \
0 & \text{otherwise}
\end{cases}
]

Where:

* (N(s,a)) is AFBS root child visit count
* (C(s)) is the set of compatible discard actions with a root child and `child_visits >= 2`

Gate:

```text
legal_discard_count >= 2
avg_root_visits_per_legal_discard >= 8.0
exit_coverage >= 0.60
```

Definitions:

```text
avg_root_visits_per_legal_discard = root_visit_count / legal_discard_count
exit_coverage = |C(s)| / legal_discard_count
```

Mask:

[
m_{\text{exit}}(a \mid s) = \mathbf{1}[a \in C(s)]
]

Important: `exit_target` is normalized **only over the masked subset**, not over all legal actions.

Why this exact target:

* ExIt’s tree-policy targets are normalized root visit distributions and were empirically stronger than hard chosen-action targets.
* AlphaZero also distills search probabilities rather than hard actions.
* But Gumbel planning shows that ungated visit-count imitation can fail when many actions are unvisited or simulation counts are small, so this target must stay subset-masked and coverage-gated here. ([NeurIPS Papers][1])

## 4.3 `delta_q_target`

Teacher object:

[
\Delta Q(a \mid s) = \mathrm{clip}\big(Q_{\text{root-persp}}(s,a) - Q_{\text{root-persp}}(s), -2.0, 2.0\big)
]

Where:

* (Q_{\text{root-persp}}(s)) is root q in **root-player perspective**
* (Q_{\text{root-persp}}(s,a)) is child q in the **same perspective**

Support set:

* compatible discard actions with a root child and `child_visits >= 4`

Mask / weight:

[
m_{\Delta Q}(a \mid s) =
\begin{cases}
\min(1, \frac{N(s,a)}{16}) & \text{if child exists and } N(s,a)\ge 4 \
0 & \text{otherwise}
\end{cases}
]

Gate:

```text
legal_discard_count >= 1
avg_root_visits_per_legal_discard >= 4.0
delta_support_count >= 1
```

Why this exact object:

* ExIt explicitly identifies action-value targets as cost-sensitive.
* AWR/AWAC show advantage-like supervision can be powerful, but only in constrained forms; they do not justify an unconstrained direct actor rewrite from sparse, search-budget-dependent q-deltas.
* Therefore delta-q belongs here as a **masked auxiliary prediction task**, not as a policy weight yet. ([NeurIPS Papers][1])

## 4.4 `safety_residual_target`

Keep the existing loader rule exactly as-is:

[
t_{\text{safety}}(a) = \mathrm{clip}\big(\text{public_safety}(a) - \text{exact_dealin_risk}(a), 0, 1\big)
]

Mask only legal discard actions.

No broadening needed in this tranche.

## 5. Data model changes

Use a single optional search-derived carrier for the new narrow search supervision.

```rust
pub struct SearchSupervision {
    pub exit_target: [f32; HYDRA_ACTION_SPACE],
    pub exit_mask: [f32; HYDRA_ACTION_SPACE],
    pub delta_q_target: [f32; HYDRA_ACTION_SPACE],
    pub delta_q_mask: [f32; HYDRA_ACTION_SPACE],
    pub root_visit_count: u32,
    pub legal_discard_count: u8,
    pub exit_support_count: u8,
    pub delta_support_count: u8,
    pub exit_coverage: f32,
}
```

Add to `MjaiSample`:

```rust
pub struct MjaiSample {
    ...
    pub search_supervision: Option<SearchSupervision>,
    pub safety_residual: Option<[f32; HYDRA_ACTION_SPACE]>,
    pub safety_residual_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
    ...
}
```

Add to `MjaiBatch`:

```rust
pub struct MjaiBatch<B: Backend> {
    ...
    pub exit_target: Option<Tensor<B, 2>>,
    pub exit_mask: Option<Tensor<B, 2>>,
    pub delta_q_target: Option<Tensor<B, 2>>,
    pub delta_q_mask: Option<Tensor<B, 2>>,
    pub safety_residual_target: Option<Tensor<B, 2>>,
    pub safety_residual_mask: Option<Tensor<B, 2>>,
    ...
}
```

Add to `HydraTargets`:

```rust
pub struct HydraTargets<B: Backend> {
    ...
    pub exit_target: Option<Tensor<B, 2>>,
    pub exit_mask: Option<Tensor<B, 2>>,
    pub delta_q_target: Option<Tensor<B, 2>>,
    pub delta_q_mask: Option<Tensor<B, 2>>,
    pub safety_residual_target: Option<Tensor<B, 2>>,
    pub safety_residual_mask: Option<Tensor<B, 2>>,
    ...
}
```

Do not add gates as separate tensors unless later logging proves you need them. Hard gates plus masks are enough for the first credible merge.

## 6. AFBS API changes required before target generation

Add an explicit child-stats API that closes both visit-count access and q perspective.

```rust
pub struct ChildStats {
    pub action: u8,
    pub visit_count: u32,
    pub q_root_perspective: f32,
}

impl AfbsTree {
    pub fn root_child_stats(&self, root: NodeIdx) -> Vec<ChildStats> {
        ...
    }

    pub fn root_q_root_perspective(&self, root: NodeIdx) -> f32 {
        ...
    }
}
```

Rules:

* `q_root_perspective` must be documented and tested.
* If the current AFBS backup already stores root-oriented values, return them unchanged.
* If values are node-to-move oriented, apply the sign conversion here.
* Do not guess in the loader.

Also add one semantic test:

```rust
#[test]
fn child_stats_q_values_are_in_root_player_perspective() { ... }
```

And one contract-closing test if you want to keep `root_exit_policy` around:

```rust
#[test]
fn root_exit_policy_tau_one_matches_normalized_child_visits() { ... }
```

If that test fails, do not use `root_exit_policy()` for training.

## 7. Loader implementation

Add one builder in `mjai_loader.rs`:

```rust
fn build_search_supervision(
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
    tree: &AfbsTree,
    root: NodeIdx,
) -> Option<SearchSupervision> {
    if !compatible_discard_state(legal_mask) {
        return None;
    }

    let legal_discards: Vec<usize> = (0..=DISCARD_END as usize)
        .filter(|&a| legal_mask[a] > 0.0)
        .collect();

    if legal_discards.is_empty() {
        return None;
    }

    let legal_discard_count = legal_discards.len() as u8;
    let avg_visits =
        tree.root_visit_count(root) as f32 / legal_discards.len() as f32;

    let root_q = tree.root_q_root_perspective(root);
    let child_stats = tree.root_child_stats(root);

    let mut by_action = std::collections::HashMap::<u8, ChildStats>::new();
    for st in child_stats {
        by_action.insert(st.action, st);
    }

    let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut exit_mask   = [0.0f32; HYDRA_ACTION_SPACE];
    let mut delta_q_target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut delta_q_mask   = [0.0f32; HYDRA_ACTION_SPACE];

    let mut exit_support = 0usize;
    let mut delta_support = 0usize;
    let mut exit_mass = 0.0f32;

    for &a in &legal_discards {
        let search_action = a as u8; // discard-only, no aka mapping in this tranche
        if let Some(st) = by_action.get(&search_action) {
            if st.visit_count >= 2 {
                exit_mask[a] = 1.0;
                exit_target[a] = st.visit_count as f32;
                exit_support += 1;
                exit_mass += st.visit_count as f32;
            }
            if st.visit_count >= 4 {
                delta_q_target[a] = (st.q_root_perspective - root_q).clamp(-2.0, 2.0);
                delta_q_mask[a] = (st.visit_count as f32 / 16.0).min(1.0);
                delta_support += 1;
            }
        }
    }

    let exit_coverage = exit_support as f32 / legal_discards.len() as f32;

    let gate_exit =
        legal_discards.len() >= 2 &&
        avg_visits >= 8.0 &&
        exit_coverage >= 0.60 &&
        exit_support >= 2;

    if gate_exit {
        for a in 0..HYDRA_ACTION_SPACE {
            exit_target[a] /= exit_mass.max(1.0);
        }
    } else {
        exit_target = [0.0; HYDRA_ACTION_SPACE];
        exit_mask = [0.0; HYDRA_ACTION_SPACE];
        exit_support = 0;
    }

    let gate_delta =
        avg_visits >= 4.0 &&
        delta_support >= 1;

    if !gate_delta {
        delta_q_target = [0.0; HYDRA_ACTION_SPACE];
        delta_q_mask = [0.0; HYDRA_ACTION_SPACE];
        delta_support = 0;
    }

    if exit_support == 0 && delta_support == 0 {
        return None;
    }

    Some(SearchSupervision {
        exit_target,
        exit_mask,
        delta_q_target,
        delta_q_mask,
        root_visit_count: tree.root_visit_count(root),
        legal_discard_count,
        exit_support_count: exit_support as u8,
        delta_support_count: delta_support as u8,
        exit_coverage,
    })
}
```

Then in sample creation:

```rust
let search_supervision = match (context.afbs_tree, context.afbs_root) {
    (Some(tree), Some(root)) => build_search_supervision(&legal_mask, tree, root),
    _ => None,
};

samples.push(MjaiSample {
    ...
    search_supervision,
    safety_residual: Some(safety_residual),
    safety_residual_mask: Some(safety_residual_mask),
    ...
});
```

## 8. Batch collation

In `collate_batch`:

* if no sample has `search_supervision`, keep all four search tensors `None`
* otherwise allocate dense zero tensors `[batch, 46]` for `exit_target`, `exit_mask`, `delta_q_target`, `delta_q_mask`
* fill present rows
* absent rows remain all-zero

Then pass through `into_hydra_targets()` unchanged.

## 9. Loss implementation

### 9.1 Add config fields

```rust
#[derive(Config, Debug)]
pub struct HydraLossConfig {
    #[config(default = "1.0")]
    pub w_pi: f32,
    #[config(default = "0.0")]
    pub w_exit: f32,
    #[config(default = "0.25")]
    pub exit_mix: f32,
    #[config(default = "0.1")]
    pub w_delta_q: f32,
    #[config(default = "0.25")]
    pub delta_q_huber_beta: f32,
    #[config(default = "0.1")]
    pub w_safety_residual: f32,
    ...
}
```

Update `total_weight()` and `validate()`.

### 9.2 Conservative ExIt loss

Use the existing helper, but **detach the model logits before mixing**.

```rust
pub fn exit_policy_ce_per_sample<B: Backend>(
    logits: Tensor<B, 2>,
    exit_target: Tensor<B, 2>,
    exit_mask: Tensor<B, 2>,
    mix: f32,
) -> Tensor<B, 1> {
    let mixed_target = soft_target_from_exit(
        logits.clone().detach(),
        exit_target,
        exit_mask.clone(),
        mix,
    );
    let masked_logits = logits + (exit_mask.ones_like() - exit_mask.clone()) * (-1e9f32);
    let log_probs = burn::tensor::activation::log_softmax(masked_logits, 1);
    -(mixed_target * log_probs).sum_dim(1)
}
```

Why the detach matters:

If (q=\text{softmax}(z)), (t) is the exit teacher, and (\tilde t=(1-\alpha),\text{stopgrad}(q)+\alpha t), then

[
\nabla_z \mathrm{CE}(\tilde t, q) = q - \tilde t = \alpha (q - t)
]

on the masked subset. That is exactly the conservative behavior wanted: same direction as full search imitation, smaller step.

Numerical toy check confirms it:

```text
q            = [0.2807, 0.1540, 0.5653]
t_exit       = [0.2000, 0.3000, 0.5000]
alpha        = 0.35
gradient     = q - ((1-alpha) q + alpha t_exit)
             = 0.35 * (q - t_exit)
```

### 9.3 Masked Huber for delta-q

Replace dense MSE with masked Huber.

```rust
pub fn masked_action_huber<B: Backend>(
    pred: Tensor<B, 2>,
    target: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    beta: f32,
) -> Tensor<B, 1> {
    let diff = pred - target;
    let abs = diff.clone().abs();
    let quad = abs.clone().clamp_max(beta);
    let lin = abs - quad.clone();
    let per_elem = (quad.clone() * quad) * (0.5 / beta) + lin;
    let masked = per_elem * mask.clone();
    let denom = mask.sum().clamp_min(1.0);
    masked.sum() / denom
}
```

### 9.4 Total loss shape

Keep the base policy/value/etc path intact. Add the new terms, do not replace base BC.

```rust
let l_exit = match (&targets.exit_target, &targets.exit_mask) {
    (Some(target), Some(mask)) => {
        let valid = mask.clone().sum_dim(1).greater_elem(1.5).float();
        masked_mean(
            exit_policy_ce_per_sample(
                outputs.policy_logits.clone(),
                target.clone(),
                mask.clone(),
                self.config.exit_mix,
            ),
            Some(valid),
        )
    }
    _ => zero.clone(),
};

let l_delta_q = match (&targets.delta_q_target, &targets.delta_q_mask) {
    (Some(target), Some(mask)) => masked_action_huber(
        outputs.delta_q.clone(),
        target.clone(),
        mask.clone(),
        self.config.delta_q_huber_beta,
    ),
    _ => zero.clone(),
};

let l_safety_residual = ...
```

Then:

```rust
total =
    l_policy_bc * c.w_pi +
    l_exit * c.w_exit +
    l_delta_q * c.w_delta_q +
    l_safety_residual * c.w_safety_residual +
    ...
```

## 10. Trainer activation policy

### 10.1 Activation order

Implement all plumbing in one pass, but activate in this order:

1. `safety_residual`
2. `delta_q`
3. `exit_target`

Reason:

* safety residual is already semantically closed
* delta-q needs only AFBS q-perspective + mask closure
* exit needs the most trust-gating and the most careful loss wiring

### 10.2 Initial weights

Use these first:

```text
w_pi = 1.0
w_exit = 1.0
exit_mix = 0.25
w_delta_q = 0.1
delta_q_huber_beta = 0.25
w_safety_residual = 0.1
```

Because `exit_mix=0.25` already scales the ExIt gradient down by 4x, `w_exit=1.0` is conservative enough.

### 10.3 Refuse activation if these audits fail

Keep `w_exit = 0` if either of these is false:

* `root_exit_policy` semantics are not closed by test, and the explicit child-visit builder was not implemented
* median `exit_coverage` over compatible states is below `0.60`

Keep `w_delta_q = 0` if either of these is false:

* q perspective is not closed by test
* delta-q target is present in the learner input planes for those same samples and has not been zeroed out

## 11. Worked example

State:

* compatible discard-only state
* 10 legal discard actions
* root visit count = 96
* child visits on searched actions:

```text
a=1  -> 32
a=3  -> 24
a=5  -> 16
a=7  ->  8
a=9  ->  4
a=11 ->  4
others -> no child
```

Then:

```text
avg_visits = 96 / 10 = 9.6
exit_support = 6
exit_coverage = 6 / 10 = 0.60
```

So `exit_target` is active.

Normalized `exit_target` on the masked subset:

```text
sum = 32 + 24 + 16 + 8 + 4 + 4 = 88

t_exit[1]  = 32 / 88 = 0.3636
t_exit[3]  = 24 / 88 = 0.2727
t_exit[5]  = 16 / 88 = 0.1818
t_exit[7]  =  8 / 88 = 0.0909
t_exit[9]  =  4 / 88 = 0.0455
t_exit[11] =  4 / 88 = 0.0455
mask = 1 on {1,3,5,7,9,11}, else 0
```

Suppose root q = `0.18` and child q values in root perspective are:

```text
q(1)=0.31
q(3)=0.28
q(5)=0.21
q(7)=0.17
q(9)=0.14
q(11)=0.12
```

Then:

```text
delta_q_target[1]  = +0.13
delta_q_target[3]  = +0.10
delta_q_target[5]  = +0.03
delta_q_target[7]  = -0.01
delta_q_target[9]  = -0.04
delta_q_target[11] = -0.06
```

And with `delta_q_mask = min(1, visits/16)`:

```text
mask[1]  = 1.00
mask[3]  = 1.00
mask[5]  = 1.00
mask[7]  = 0.50
mask[9]  = 0.25
mask[11] = 0.25
```

This is the exact kind of state you want:

* ExIt gets a soft, cost-sensitive policy teacher over searched discard actions.
* Delta-q learns action ordering and margins without pretending all 46 actions were searched.
* Non-discard actions remain untouched.

## 12. Tests to add before merge

Add all of these.

### AFBS semantics

```rust
#[test] fn child_stats_q_values_are_in_root_player_perspective()
#[test] fn root_exit_policy_tau_one_matches_normalized_child_visits()
```

### Loader / target generation

```rust
#[test] fn build_search_supervision_skips_non_discard_states()
#[test] fn build_search_supervision_skips_aka_ambiguity()
#[test] fn build_search_supervision_emits_exit_only_when_coverage_gate_passes()
#[test] fn build_search_supervision_emits_delta_q_with_visit_weighted_mask()
#[test] fn build_search_supervision_normalizes_exit_target_over_mask_only()
```

### Batch plumbing

```rust
#[test] fn batch_to_hydra_targets_carries_search_supervision()
#[test] fn batch_to_hydra_targets_zero_fills_missing_search_supervision_rows()
```

### Losses

```rust
#[test] fn exit_loss_is_zero_when_exit_mask_absent()
#[test] fn exit_loss_ignores_unsearched_legal_actions()
#[test] fn delta_q_loss_requires_delta_q_mask()
#[test] fn safety_residual_still_requires_mask()
#[test] fn base_policy_loss_nonzero_when_behavior_target_present()
```

That last test is mandatory because Artifact B’s shown `total_loss()` is not trustworthy enough.

## 13. Keep narrow, defer, reject

### Keep narrow

* discard-only states
* no aka ambiguity
* auxiliary losses only
* base BC policy loss remains the anchor
* ExIt masked to searched subset only
* delta-q masked to visited subset only

### Defer

* ExIt on riichi/kan/tsumo/call states
* full 46-action AFBS supervision
* red-five-aware search targets
* delta-q-driven policy weighting
* completed-Q / Gumbel-style policy improvement
* belief fields / mixture weights / opponent hand type / oracle critic

Why defer the Gumbel-style path:

It is real, not decorative: the Gumbel paper explicitly shows that plain visit-count policy updates can fail to improve the policy when not all actions are visited, and its fix uses completed q-values and a more substantial planning change. That is a legitimate future lane, especially if AFBS keeps operating under very low simulation budgets, but it is not the smallest credible tranche here. ([OpenReview][2])

### Reject in this tranche

* dense unmasked `delta_q` regression on zero-filled vectors
* dense ExIt CE over full legal mask when search covered only a subset
* using `node_q_value()` without a perspective contract
* using `root_exit_policy()` as a teacher without a semantics test
* turning on belief targets because the loader can already emit them

The belief/mixture lane is structurally present, but the current teacher is stage-A public-remaining-count logic, not a closed oracle or search-backed posterior. Leave it off.

## 14. The exact merge sequence

1. Add `SearchSupervision` carrier to `MjaiSample`.
2. Add AFBS `ChildStats` API with explicit root-perspective q semantics.
3. Implement `build_search_supervision()` in loader with the discard-only + no-aka gate.
4. Plumb `exit_target`, `exit_mask`, `delta_q_target`, `delta_q_mask` through batch and `HydraTargets`.
5. Add `w_exit`, `exit_mix`, `delta_q_huber_beta` to loss config.
6. Implement `exit_policy_ce_per_sample()` using `soft_target_from_exit(outputs.policy_logits.detach(), ..., exit_mask, exit_mix)`.
7. Replace dense `delta_q` MSE with masked Huber.
8. Add the tests above.
9. Turn on `w_safety_residual=0.1`.
10. Turn on `w_delta_q=0.1` only after q-perspective + leakage audit passes.
11. Turn on `w_exit=1.0, exit_mix=0.25` only after coverage stats show the gate is actually producing a useful supervised subset.

That is the surviving path.

[1]: https://papers.neurips.cc/paper/7120-thinking-fast-and-slow-with-deep-learning-and-tree-search.pdf "https://papers.neurips.cc/paper/7120-thinking-fast-and-slow-with-deep-learning-and-tree-search.pdf"
[2]: https://openreview.net/pdf?id=bERaNdoegnO "https://openreview.net/pdf?id=bERaNdoegnO"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
