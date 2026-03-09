<combined_run_record run_id="answer_18" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 18 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_18_IMPLEMENT_TARGET_PROVENANCE_AND_SAFETY_RESIDUAL.md">
  <![CDATA[# Prompt 18 — target provenance and safety-residual blueprint

<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest exact blueprint for advanced target provenance in this training stack.

The answer should help us decide what is truly ready now, what is semantically wrong, what is only a typed hole, what should remain privileged-only, and what should stay off until the target object and provenance are defensible.

We want an answer that is implementation-usable, detailed, and easy to audit.
Do not be vague.
Do not hide behind broad taxonomies.
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
The artifacts below reflect what the current codebase/docs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>
Embedded artifact/context dump intentionally omitted during normalization. Use the paired answer body plus the repository source files for audit and follow-up.
</artifacts>

]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_18.md">
  <![CDATA[# Blueprint: advanced target provenance closure for the current tranche

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

pub fn belief_fields_bce_per_sample<B: Backend>(
    logits: Tensor<B, 3>,
    target: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let [batch, channels, tiles] = logits.dims();
    bce_with_logits_3d(logits, target)
        .reshape([batch, channels * tiles])
        .mean_dim(1)
        .squeeze_dim::<1>(1)
}

pub fn mixture_weight_ce_per_sample<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    cross_entropy_soft(activation::log_softmax(logits, 1), target)
}

pub fn opponent_hand_type_ce_per_sample<B: Backend>(
    logits: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    cross_entropy_soft(activation::log_softmax(logits, 1), target)
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
        let l_oracle = match &targets.oracle_target {
            Some(target) => masked_mean(
                oracle_critic_loss_per_sample(outputs.oracle_critic.clone(), target.clone()),
                oracle_mask.clone(),
            ),
            None => zero.clone(),
        };
        let l_belief = match (&targets.belief_fields_target, &targets.belief_fields_mask) {
            (Some(target), Some(mask)) => masked_mean(
                belief_fields_bce_per_sample(outputs.belief_fields.clone(), target.clone()),
                Some(mask.clone()),
            ),
            _ => zero.clone(),
        };
        let l_mix = match (&targets.mixture_weight_target, &targets.mixture_weight_mask) {
            (Some(target), Some(mask)) => masked_mean(
                mixture_weight_ce_per_sample(outputs.mixture_weight_logits.clone(), target.clone()),
                Some(mask.clone()),
            ),
            _ => zero.clone(),
        };
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
        let total = l_oracle.clone() * c.w_oracle_critic
            + l_belief.clone() * c.w_belief_fields
            + l_mix.clone() * c.w_mixture_weight
            + l_hand_type.clone() * c.w_opponent_hand_type
            + l_delta_q.clone() * c.w_delta_q
            + l_safety_residual.clone() * c.w_safety_residual;
        LossBreakdown {
            oracle_critic: l_oracle,
            belief_fields: l_belief,
            mixture_weight: l_mix,
            opponent_hand_type: l_hand_type,
            delta_q: l_delta_q,
            safety_residual: l_safety_residual,
            total,
            policy: zero.clone(),
            value: zero.clone(),
            grp: zero.clone(),
            tenpai: zero.clone(),
            danger: zero.clone(),
            opp_next: zero.clone(),
            score_pdf: zero.clone(),
            score_cdf: zero,
        }
    }
}
```

Artifact B — batch/sample and collation surface:

```rust
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

pub fn collate_batch<B: Backend>(samples: &[MjaiSample], device: &B::Device) -> MjaiBatch<B> {
    let batch = samples.len();
    let mut safety_residual_flat = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    let mut safety_residual_mask_flat = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    let mut any_safety_residual = false;
    let mut belief_fields_flat = vec![0.0f32; batch * 16 * 34];
    let mut mixture_weights_flat = vec![0.0f32; batch * 4];
    let mut any_belief_fields = false;
    let mut any_mixture_weights = false;
    let mut belief_fields_mask = vec![0.0f32; batch];
    let mut mixture_weight_mask = vec![0.0f32; batch];

    for (i, s) in samples.iter().enumerate() {
        if let Some(values) = s.safety_residual {
            safety_residual_flat[i * HYDRA_ACTION_SPACE..(i + 1) * HYDRA_ACTION_SPACE]
                .copy_from_slice(&values);
            any_safety_residual = true;
        }
        if let Some(values) = s.safety_residual_mask {
            safety_residual_mask_flat[i * HYDRA_ACTION_SPACE..(i + 1) * HYDRA_ACTION_SPACE]
                .copy_from_slice(&values);
            any_safety_residual = true;
        }
        if let Some(values) = s.belief_fields {
            belief_fields_flat[i * 16 * 34..(i + 1) * 16 * 34].copy_from_slice(&values);
            any_belief_fields = true;
        }
        if s.belief_fields_present {
            belief_fields_mask[i] = 1.0;
            any_belief_fields = true;
        }
        if let Some(values) = s.mixture_weights {
            mixture_weights_flat[i * 4..(i + 1) * 4].copy_from_slice(&values);
            any_mixture_weights = true;
        }
        if s.mixture_weights_present {
            mixture_weight_mask[i] = 1.0;
            any_mixture_weights = true;
        }
    }

    MjaiBatch {
        safety_residual_target: if any_safety_residual {
            Some(
                Tensor::<B, 1>::from_floats(safety_residual_flat.as_slice(), device)
                    .reshape([batch, HYDRA_ACTION_SPACE]),
            )
        } else {
            None
        },
        safety_residual_mask: if any_safety_residual {
            Some(
                Tensor::<B, 1>::from_floats(safety_residual_mask_flat.as_slice(), device)
                    .reshape([batch, HYDRA_ACTION_SPACE]),
            )
        } else {
            None
        },
        belief_fields_target: if any_belief_fields {
            Some(
                Tensor::<B, 1>::from_floats(belief_fields_flat.as_slice(), device)
                    .reshape([batch, 16, 34]),
            )
        } else {
            None
        },
        mixture_weight_target: if any_mixture_weights {
            Some(
                Tensor::<B, 1>::from_floats(mixture_weights_flat.as_slice(), device)
                    .reshape([batch, 4]),
            )
        } else {
            None
        },
        belief_fields_mask: if any_belief_fields {
            Some(Tensor::<B, 1>::from_floats(
                belief_fields_mask.as_slice(),
                device,
            ))
        } else {
            None
        },
        mixture_weight_mask: if any_mixture_weights {
            Some(Tensor::<B, 1>::from_floats(
                mixture_weight_mask.as_slice(),
                device,
            ))
        } else {
            None
        },
        obs: Tensor::<B, 3>::zeros([batch, 192, 34], device),
        actions: Tensor::<B, 1, Int>::zeros([batch], device),
        legal_mask: Tensor::<B, 2>::zeros([batch, HYDRA_ACTION_SPACE], device),
        value_target: Tensor::<B, 1>::zeros([batch], device),
        grp_target: Tensor::<B, 2>::zeros([batch, 24], device),
        oracle_target: None,
        oracle_target_mask: Tensor::<B, 1>::zeros([batch], device),
        tenpai_target: Tensor::<B, 2>::zeros([batch, 3], device),
        danger_target: Tensor::<B, 3>::zeros([batch, 3, 34], device),
        danger_mask: Tensor::<B, 3>::zeros([batch, 3, 34], device),
        opp_next_target: Tensor::<B, 3>::zeros([batch, 3, 34], device),
        score_pdf_target: Tensor::<B, 2>::zeros([batch, 64], device),
        score_cdf_target: Tensor::<B, 2>::zeros([batch, 64], device),
    }
}
```

Artifact C — real target builders and provenance-producing logic:

```rust
fn public_safety_score(safety: &SafetyInfo, tile: u8) -> f32 {
    let t = tile as usize;
    let mut score = 0.0f32;
    for opp in 0..3usize {
        if hydra_core::safety::bit_test(safety.genbutsu_all[opp], t) {
            score += 1.0;
        }
        score += 0.35 * safety.suji[opp][t];
        if hydra_core::safety::bit_test(safety.half_suji[opp], t) {
            score += 0.1;
        }
        score -= 0.25 * safety.matagi[opp][t];
        if safety.opponent_riichi[opp] || safety.cached_tenpai_prob[opp] > 0.5 {
            score -= 0.1;
        }
    }
    if hydra_core::safety::bit_test(safety.kabe, t) {
        score += 0.4;
    }
    if hydra_core::safety::bit_test(safety.one_chance, t) {
        score += 0.2;
    }
    score.clamp(0.0, 1.0)
}

fn exact_dealin_risk_from_waits(wait_sets: &[[f32; 34]; 3], tile: u8) -> f32 {
    let t = tile as usize;
    if wait_sets.iter().any(|waits| waits[t] > 0.0) {
        1.0
    } else {
        0.0
    }
}

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

let (safety_residual, safety_residual_mask) = build_safety_residual_targets(
    &legal_mask,
    &safety[actor],
    &wait_sets,
);
let (belief_fields, mixture_weights, belief_fields_present, mixture_weights_present) =
    build_stage_a_belief_targets(&state, actor, &obs);

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

Artifact D — belief teacher object and tests:

```rust
//! Generates Stage A projected belief teacher targets.

pub const BELIEF_COMPONENTS: usize = 4;
pub const BELIEF_ZONES: usize = 4;
pub const BELIEF_TILES: usize = 34;
pub const BELIEF_FIELDS_SIZE: usize = BELIEF_COMPONENTS * BELIEF_ZONES * BELIEF_TILES;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageABeliefTarget {
    pub belief_fields: [f32; BELIEF_FIELDS_SIZE],
    pub mixture_weights: Option<[f32; BELIEF_COMPONENTS]>,
    pub trust: f32,
    pub ess: f32,
    pub entropy: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageABeliefConfig {
    pub num_components: u8,
    pub trust_threshold: f32,
    pub mixture_entropy_threshold: f32,
}

impl Default for StageABeliefConfig {
    fn default() -> Self {
        Self {
            num_components: BELIEF_COMPONENTS as u8,
            trust_threshold: 0.55,
            mixture_entropy_threshold: 1.15,
        }
    }
}

pub fn build_uniform_kernel() -> [f64; BELIEF_TILES * BELIEF_ZONES] {
    [1.0; BELIEF_TILES * BELIEF_ZONES]
}

pub fn project_public_remaining_to_row_sums(
    remaining: &[f32; BELIEF_TILES],
) -> [f64; BELIEF_TILES] {
    let mut row_sums = [0.0f64; BELIEF_TILES];
    for (dst, &value) in row_sums.iter_mut().zip(remaining.iter()) {
        *dst = value.max(0.0) as f64;
    }
    row_sums
}

pub fn project_hidden_count_to_col_sums(hidden_tiles: usize) -> [f64; BELIEF_ZONES] {
    let base = hidden_tiles as f64 / BELIEF_ZONES as f64;
    [base, base, base, base]
}

pub fn build_stage_a_teacher(
    remaining: &[f32; BELIEF_TILES],
    hidden_tiles: usize,
    config: StageABeliefConfig,
) -> Option<StageABeliefTarget> {
    if hidden_tiles == 0 {
        return None;
    }

    let row_sums = project_public_remaining_to_row_sums(remaining);
    let total_row: f64 = row_sums.iter().sum();
    if total_row <= 0.0 {
        return None;
    }

    let col_sums = project_hidden_count_to_col_sums(hidden_tiles);
    let kernel = build_uniform_kernel();
    let mixture = MixtureSib::new(config.num_components, &kernel, &row_sums, &col_sums);
    let weights = mixture.weights();
    let entropy = mixture.weight_entropy() as f32;
    let ess = mixture.ess() as f32;
    let trust = ((ess / config.num_components as f32).clamp(0.0, 1.0) * 0.7
        + (1.0 - (entropy / 1.3863).clamp(0.0, 1.0)) * 0.3)
        .clamp(0.0, 1.0);

    if trust < config.trust_threshold {
        return None;
    }

    let mut belief_fields = [0.0f32; BELIEF_FIELDS_SIZE];
    for component in 0..BELIEF_COMPONENTS {
        for zone in 0..BELIEF_ZONES {
            let channel = component * BELIEF_ZONES + zone;
            for tile in 0..BELIEF_TILES {
                belief_fields[channel * BELIEF_TILES + tile] =
                    mixture.components[component].belief[tile * BELIEF_ZONES + zone] as f32;
            }
        }
    }

    let mixture_weights = if entropy <= config.mixture_entropy_threshold {
        let mut out = [0.0f32; BELIEF_COMPONENTS];
        for (dst, src) in out.iter_mut().zip(weights.iter().copied()) {
            *dst = src as f32;
        }
        Some(out)
    } else {
        None
    };

    Some(StageABeliefTarget {
        belief_fields,
        mixture_weights,
        trust,
        ess,
        entropy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_a_teacher_returns_none_without_hidden_tiles() {
        let remaining = [1.0f32; BELIEF_TILES];
        assert!(build_stage_a_teacher(&remaining, 0, StageABeliefConfig::default()).is_none());
    }

    #[test]
    fn stage_a_teacher_produces_projected_belief_fields() {
        let remaining = [1.0f32; BELIEF_TILES];
        let target = build_stage_a_teacher(&remaining, 40, StageABeliefConfig::default())
            .expect("teacher target");
        assert!(target.trust >= StageABeliefConfig::default().trust_threshold);
        assert!(target
            .belief_fields
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn stage_a_teacher_can_emit_mixture_weights() {
        let remaining = [1.0f32; BELIEF_TILES];
        let cfg = StageABeliefConfig {
            mixture_entropy_threshold: 10.0,
            ..StageABeliefConfig::default()
        };
        let target = build_stage_a_teacher(&remaining, 40, cfg).expect("teacher target");
        assert!(target.mixture_weights.is_some());
    }
}
```

Artifact E — model output surface and selected tests:

```rust
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

impl HydraModelConfig {
    pub fn actor() -> Self {
        Self::new(12).with_input_channels(INPUT_CHANNELS)
    }

    pub fn learner() -> Self {
        Self::new(24).with_input_channels(INPUT_CHANNELS)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn actor_net_all_output_shapes() {
        ...
        assert_eq!(out.belief_fields.dims(), [batch, 16, 34]);
        assert_eq!(out.mixture_weight_logits.dims(), [batch, 4]);
        assert_eq!(out.opponent_hand_type.dims(), [batch, 24]);
        assert_eq!(out.delta_q.dims(), [batch, 46]);
        assert_eq!(out.safety_residual.dims(), [batch, 46]);
    }

    #[test]
    fn oracle_head_does_not_backprop_to_backbone_input() {
        ...
        assert!(x.grad(&grads).is_none());
    }
}
```

Artifact F — selected tests around optional target behavior:

```rust
#[test]
fn batch_to_hydra_targets_keeps_optional_advanced_targets_narrow() {
    let targets = batch.into_hydra_targets();
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

#[test]
fn test_optional_belief_losses_default_to_zero() {
    ...
    assert!(belief.abs() < 1e-8);
    assert!(mixture.abs() < 1e-8);
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
```

Artifact G — doctrine excerpts:

```text
Stronger target generation is a better immediate lever than a giant search rewrite.
```

```text
Advanced heads exist, but default advanced loss weights are zero.
Advanced supervision hooks exist, but the normal batch collation path still mostly emits baseline targets.
```

```text
The first coding tranche is about target-generation/supervision closure.
Do not expand model surface in this tranche.
No new heads.
```

```text
Preferred order:
- first: safety_residual_target, delta_q_target, and any replay-credible target that can be computed without new search infra
- later in same tranche if credible: belief_fields_target, mixture_weight_target, opponent_hand_type_target
```

```text
Key rule:
target presence should control whether an advanced loss exists at all; weight alone should not hide broken plumbing.
```

Artifact H — additional target tests and edge behavior:

```rust
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
    let sr = targets
        .safety_residual_target
        .expect("safety residual target");
    let srm = targets.safety_residual_mask.expect("safety residual mask");
    assert_eq!(sr.dims(), [1, HYDRA_ACTION_SPACE]);
    assert_eq!(srm.dims(), [1, HYDRA_ACTION_SPACE]);
}

#[test]
fn batch_to_hydra_targets_carries_projected_belief_targets() {
    let mut belief = [0.0f32; 16 * 34];
    let mut mix = [0.0f32; 4];
    belief[0] = 0.2;
    belief[33] = 0.8;
    mix[0] = 0.7;
    mix[1] = 0.3;
    sample.belief_fields = Some(belief);
    sample.mixture_weights = Some(mix);
    sample.belief_fields_present = true;
    sample.mixture_weights_present = true;
    let batch = collate_batch::<B>(&[sample], &device);
    let targets = batch.into_hydra_targets();
    let belief_target = targets.belief_fields_target.expect("belief field target");
    let mix_target = targets.mixture_weight_target.expect("mixture weights");
    assert_eq!(belief_target.dims(), [1, 16, 34]);
    assert_eq!(mix_target.dims(), [1, 4]);
}

#[test]
fn batch_to_hydra_targets_keeps_belief_targets_absent_when_missing() {
    let targets = batch.into_hydra_targets();
    assert!(targets.belief_fields_target.is_none());
    assert!(targets.mixture_weight_target.is_none());
    assert!(targets.belief_fields_mask.is_none());
    assert!(targets.mixture_weight_mask.is_none());
}
```

```rust
#[test]
fn test_default_weights_match_roadmap() {
    let cfg = HydraLossConfig::new();
    assert!((cfg.w_oracle_critic - 0.0).abs() < 1e-6);
    assert!((cfg.w_belief_fields - 0.0).abs() < 1e-6);
    assert!((cfg.w_mixture_weight - 0.0).abs() < 1e-6);
    assert!((cfg.w_opponent_hand_type - 0.0).abs() < 1e-6);
    assert!((cfg.w_delta_q - 0.0).abs() < 1e-6);
    assert!((cfg.w_safety_residual - 0.0).abs() < 1e-6);
}

#[test]
fn test_safety_residual_all_zero_mask_zeroes_loss() {
    targets.safety_residual_target = Some(Tensor::<B, 2>::ones([2, 46], &device));
    targets.safety_residual_mask = Some(Tensor::<B, 2>::zeros([2, 46], &device));
    let breakdown = loss_fn.total_loss(&outputs, &targets);
    let safety_residual: f32 = breakdown.safety_residual.into_scalar().elem();
    assert!(safety_residual.abs() < 1e-8);
}
```

Artifact I — model/test artifacts showing typed surfaces exist:

```rust
#[test]
fn actor_net_all_output_shapes() {
    assert_eq!(out.belief_fields.dims(), [batch, 16, 34]);
    assert_eq!(out.mixture_weight_logits.dims(), [batch, 4]);
    assert_eq!(out.opponent_hand_type.dims(), [batch, 24]);
    assert_eq!(out.delta_q.dims(), [batch, 46]);
    assert_eq!(out.safety_residual.dims(), [batch, 46]);
}

#[test]
fn learner_net_all_output_shapes() {
    assert_eq!(out.belief_fields.dims(), [batch, 16, 34]);
    assert_eq!(out.mixture_weight_logits.dims(), [batch, 4]);
    assert_eq!(out.opponent_hand_type.dims(), [batch, 24]);
    assert_eq!(out.delta_q.dims(), [batch, 46]);
    assert_eq!(out.safety_residual.dims(), [batch, 46]);
}
```

Artifact J — additional tranche guidance excerpts:

```text
Concrete coding objectives:
1. Audit and populate advanced targets in sample construction.
2. Stage loss activation in one place.
3. Keep the rollout narrow.
4. Do not expand model surface in this tranche.
```

```text
Concrete tranche intent:
- populate advanced targets where feasible from existing replay/context machinery
- turn on nonzero advanced loss weights in a controlled staged way
- keep AFBS deeper integration for the following tranche, not this one
```

```text
Minimal tranche acceptance checklist:
- selected advanced targets are populated by real code paths, not always-None
- at least one train path produces nonzero advanced auxiliary loss contributions in tests
- no new heads
- no broad AFBS rewrite
- no duplicated belief stack
```

Artifact K — more optional-target and mask behavior tests:

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
fn augment_samples_6x_permutes_aux_tile_targets() {
    sample.safety_residual = Some(safety_residual);
    sample.safety_residual_mask = Some(safety_residual_mask);
    let augmented = augment_samples_6x(&[sample]);
    let swapped = augmented.iter().find(|s| s.action == 9).expect("perm");
    let sr = swapped.safety_residual.expect("safety residual target");
    let srm = swapped.safety_residual_mask.expect("safety residual mask");
    assert!((sr[9] - 0.75).abs() < 1e-6);
    assert!((srm[9] - 1.0).abs() < 1e-6);
}

#[test]
fn augment_samples_6x_permutes_belief_fields_and_preserves_mixture_weights() {
    sample.belief_fields = Some(belief);
    sample.belief_fields_present = true;
    sample.mixture_weights = Some(mix);
    sample.mixture_weights_present = true;
    let augmented = augment_samples_6x(&[sample]);
    let swapped = augmented.iter().find(|s| s.action == 9).expect("perm");
    let swapped_belief = swapped.belief_fields.expect("belief fields");
    let swapped_mix = swapped.mixture_weights.expect("mixture weights");
    assert!((swapped_belief[9] - 1.0).abs() < 1e-6);
    assert!((swapped_mix[0] - 0.8).abs() < 1e-6);
}
```

Artifact L — more loss activation tests:

```rust
#[test]
fn test_oracle_absent_with_mask_keeps_oracle_loss_zero() {
    let mut targets = make_dummy_targets::<B>(&device, 2);
    targets.oracle_target = None;
    targets.oracle_guidance_mask = Some(Tensor::<B, 1>::zeros([2], &device));
    let breakdown = loss_fn.total_loss(&outputs, &targets);
    let oracle_loss: f32 = breakdown.oracle_critic.into_scalar().elem();
    assert!(oracle_loss.abs() < 1e-8);
}

#[test]
fn test_oracle_target_contributes_to_total_when_weight_enabled() {
    let mut targets = make_dummy_targets::<B>(&device, 2);
    targets.oracle_target = Some(Tensor::<B, 2>::ones([2, 4], &device));
    let with_oracle = HydraLoss::<B>::new(HydraLossConfig::new().with_w_oracle_critic(1.0))
        .total_loss(&outputs, &targets);
    let oracle_loss: f32 = with_oracle.oracle_critic.into_scalar().elem();
    assert!(oracle_loss > 0.0);
}

#[test]
fn test_optional_belief_losses_activate_when_targets_present() {
    targets.belief_fields_target = Some(Tensor::<B, 3>::zeros([2, 16, 34], &device));
    targets.mixture_weight_target = Some(Tensor::<B, 2>::from_floats(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        &device,
    ));
    let breakdown = loss_fn.total_loss(&outputs, &targets);
    let belief: f32 = breakdown.belief_fields.into_scalar().elem();
    let mixture: f32 = breakdown.mixture_weight.into_scalar().elem();
    assert!(belief.is_finite() && belief >= 0.0);
    assert!(mixture.is_finite() && mixture >= 0.0);
}
```


Implement this blueprint exactly for the existing model surface. Do not add heads. Do not widen search. Do not turn on any advanced loss until the head is classified as `Ready` by the registry below.

Train-time privileged supervision is allowed only when it is explicitly marked train-only and the target genuinely contains information unavailable to the actor at inference. That is the LUPI/distillation regime: privileged information is available during training but not at test time, and teacher targets are useful when they add information the student does not already have; dense soft targets can accelerate learning, especially in RL-like low-data regimes. ([JMLR][1])

---

## 0. Hard gate before any advanced rollout

Treat these as stop-ship checks.

### 0.1 Verify `HydraLoss::total_loss` is not the literal code in Artifact A

**Artifact support.** The shown `total_loss` computes only advanced heads, sets all baseline heads to zero in the breakdown, and forms `total` from only advanced terms. Default advanced weights are all zero in `HydraLossConfig`.

**Inference.** If Artifact A is literal current code, default training loss is identically zero.

**Required action.**
Run a repo check immediately. If confirmed, patch before any provenance work:

```rust
let total =
    l_policy.clone() * c.w_pi +
    l_value.clone() * c.w_v +
    l_grp.clone() * c.w_grp +
    l_tenpai.clone() * c.w_tenpai +
    l_danger.clone() * c.w_danger +
    l_opp.clone() * c.w_opp +
    l_score_pdf.clone() * c.w_score +
    l_score_cdf.clone() * c.w_score +
    l_oracle.clone() * c.w_oracle_critic +
    l_belief.clone() * c.w_belief_fields +
    l_mix.clone() * c.w_mixture_weight +
    l_hand_type.clone() * c.w_opponent_hand_type +
    l_delta_q.clone() * c.w_delta_q +
    l_safety_residual.clone() * c.w_safety_residual;
```

Add this integration test:

```rust
#[test]
fn real_train_path_default_total_loss_is_nonzero() {
    let (model, batch, loss_fn) = make_real_train_fixture();
    let out = model.forward(batch.obs.clone());
    let targets = batch.into_hydra_targets();
    let total: f32 = loss_fn.total_loss(&out, &targets).total.into_scalar().elem();
    assert!(total.is_finite());
    assert!(total > 0.0);
}
```

### 0.2 Verify `collate_batch` is not the literal zero-fill baseline path in Artifact B

**Artifact support.** The shown `collate_batch` writes zeros for `obs`, `actions`, `legal_mask`, and all baseline targets.

**Inference.** If literal, the batch path is not just incomplete; it destroys baseline supervision and policy labels.

**Required action.**
Add this integration test against a real sample source:

```rust
#[test]
fn real_collate_batch_preserves_obs_actions_and_legal_mask() {
    let device = Default::default();
    let samples = make_real_nontrivial_samples(4);
    let batch = collate_batch::<B>(&samples, &device);

    let action_sum: i64 = batch.actions.clone().int().sum().into_scalar().elem();
    let legal_sum: f32 = batch.legal_mask.clone().sum().into_scalar().elem();
    let obs_sum: f32 = batch.obs.clone().sum().into_scalar().elem();

    assert!(action_sum != 0 || samples.iter().any(|s| s.action != 0));
    assert!(legal_sum > 0.0);
    assert!(obs_sum != 0.0);
}
```

If both gates pass, continue.

---

## 1. Install a target provenance registry now

The current code has typed surfaces but no auditable statement of what each target *is*. Add one registry and make all enablement flow through it.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticState {
    Ready,
    SemanticallyWrong,
    TypedHole,
    DiagnosticOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceKind {
    ReplayObserved,
    ReplayDerivedPublic,
    ReplayDerivedPrivileged,
    HeuristicProjection,
    SearchCounterfactual,
    Placeholder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    PublicAtInference,
    TrainOnlyPrivileged,
}

#[derive(Debug, Clone, Copy)]
pub struct HeadSpec {
    pub name: &'static str,
    pub version: u16,
    pub semantic: SemanticState,
    pub provenance: ProvenanceKind,
    pub visibility: Visibility,
}

pub const ADVANCED_HEAD_SPECS: &[HeadSpec] = &[
    HeadSpec {
        name: "oracle_critic",
        version: 1,
        semantic: SemanticState::DiagnosticOnly,
        provenance: ProvenanceKind::ReplayDerivedPrivileged,
        visibility: Visibility::TrainOnlyPrivileged,
    },
    HeadSpec {
        name: "belief_fields",
        version: 0,
        semantic: SemanticState::SemanticallyWrong,
        provenance: ProvenanceKind::HeuristicProjection,
        visibility: Visibility::PublicAtInference,
    },
    HeadSpec {
        name: "mixture_weight",
        version: 0,
        semantic: SemanticState::SemanticallyWrong,
        provenance: ProvenanceKind::HeuristicProjection,
        visibility: Visibility::PublicAtInference,
    },
    HeadSpec {
        name: "opponent_hand_type",
        version: 0,
        semantic: SemanticState::TypedHole,
        provenance: ProvenanceKind::Placeholder,
        visibility: Visibility::TrainOnlyPrivileged,
    },
    HeadSpec {
        name: "delta_q",
        version: 0,
        semantic: SemanticState::TypedHole,
        provenance: ProvenanceKind::Placeholder,
        visibility: Visibility::TrainOnlyPrivileged,
    },
    HeadSpec {
        name: "safety_residual",
        version: 1, // bump to 2 after patch in Section 3
        semantic: SemanticState::SemanticallyWrong,
        provenance: ProvenanceKind::ReplayDerivedPrivileged,
        visibility: Visibility::TrainOnlyPrivileged,
    },
];
```

Add one runtime validator:

```rust
pub fn validate_enabled_head(
    spec: HeadSpec,
    weight: f32,
    dataset_present_count: usize,
) -> anyhow::Result<()> {
    if weight <= 0.0 {
        return Ok(());
    }
    match spec.semantic {
        SemanticState::Ready => {}
        other => anyhow::bail!("head {} enabled with semantic state {:?}", spec.name, other),
    }
    if dataset_present_count == 0 {
        anyhow::bail!("head {} enabled but no targets appear in audit window", spec.name);
    }
    Ok(())
}
```

This enforces the doctrine rule: **target presence controls whether a loss can exist; weight does not hide broken plumbing.**

---

## 2. Head ledger: what is ready, what is wrong, what is a hole

| head                 | provenance now             | semantic status now | visibility            | train decision      |
| -------------------- | -------------------------- | ------------------: | --------------------- | ------------------- |
| `safety_residual`    | replay + hidden wait sets  |        wrong object | train-only privileged | patch, then enable  |
| `delta_q`            | none                       |          typed hole | undecidable           | keep off            |
| `belief_fields`      | public projection          |  semantically wrong | public-at-inference   | keep off            |
| `mixture_weight`     | public projection          |  semantically wrong | public-at-inference   | keep off            |
| `opponent_hand_type` | none                       |          typed hole | train-only privileged | keep off            |
| `oracle_critic`      | optional privileged branch |     diagnostic-only | train-only privileged | keep off by default |

**Ready now:** no advanced head is train-ready exactly as shown.

**Ready after one bounded patch:** only `safety_residual`.

---

## 3. Patch `safety_residual` and make it the only advanced head enabled in this tranche

### 3.1 Why the current object is semantically wrong

**Artifact support.**
Current code defines:

* `public_score = public_safety_score(...) ∈ [0,1]`
* `exact_risk = exact_dealin_risk_from_waits(...) ∈ {0,1}`
* `target = clamp(public_score - exact_risk, 0, 1)`

Since `exact_risk` is binary, the target collapses to:

[
t(a)=
\begin{cases}
s_{\text{pub}}(a) & \text{if no deal-in}\
0 & \text{if deal-in}
\end{cases}
]

That is not a residual. It is either the unchanged public score or zero.

Worked examples from the current algebra:

* `public=0.9`, actual deal-in = 1  → current target `0.0`
* `public=0.2`, actual safe = 1      → current target `0.2`

The head never learns the magnitude of a negative correction and never teaches the model that a weak public score can still be exactly safe.

Residual learning is additive: a zero residual should leave the base unchanged, and the learned term should represent a correction to that base. That is the right semantic template here. ([ar5iv][2])

### 3.2 Replace the target object with a signed correction

Define:

* (a): action index over discard actions only
* (s_{\text{pub}}(a)\in[0,1]): deterministic public heuristic score
* (y_{\text{deal}}(a)\in{0,1}): exact replay-time deal-in event from hidden waits
* (s_{\text{exact}}(a)=1-y_{\text{deal}}(a)\in{0,1}): exact safety label
* (\Delta_s(a)=s_{\text{exact}}(a)-s_{\text{pub}}(a)\in[-1,1])

Use (\Delta_s(a)) as the supervised target.

This is the only advanced head in the supplied artifacts that already uses genuinely train-only information (`wait_sets`) and therefore actually fits a privileged-information/distillation story. The target is also dense, which is desirable because soft teacher signals carry more information than hard labels, but only if the teacher adds information the student does not already have. Here it does; the current `belief` teacher does not. ([JMLR][1])

### 3.3 Patch the builder exactly

Rename the helper because it returns an event, not a probability.

```rust
fn exact_dealin_event_from_waits(wait_sets: &[[f32; 34]; 3], tile: u8) -> f32 {
    let t = tile as usize;
    if wait_sets.iter().any(|waits| waits[t] > 0.0) {
        1.0
    } else {
        0.0
    }
}

fn build_safety_residual_targets(
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
    safety: &SafetyInfo,
    wait_sets: &[[f32; 34]; 3],
) -> ([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE]) {
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];

    for action in 0..=DISCARD_END {
        let idx = action as usize;
        if legal_mask[idx] <= 0.0 {
            continue;
        }

        let tile = match action {
            AKA_5M => 4,
            AKA_5P => 13,
            AKA_5S => 22,
            _ => action,
        };

        let s_pub = public_safety_score(safety, tile);
        let y_dealin = exact_dealin_event_from_waits(wait_sets, tile);
        let s_exact = 1.0 - y_dealin;

        target[idx] = s_exact - s_pub; // signed residual in [-1, 1]
        mask[idx] = 1.0;
    }

    (target, mask)
}
```

Keep the existing masked action MSE for now:

[
L_{\text{sr}}
=============

\frac{1}{\max(1,\sum m_a)}
\sum_a m_a \cdot \frac{1}{2}\left(\hat{\Delta}_s(a)-\Delta_s(a)\right)^2
]

No new head is needed. Only the target object changes.

### 3.4 Bump the provenance entry after the patch

After the above lands, change the registry entry to:

```rust
HeadSpec {
    name: "safety_residual",
    version: 2,
    semantic: SemanticState::Ready,
    provenance: ProvenanceKind::ReplayDerivedPrivileged,
    visibility: Visibility::TrainOnlyPrivileged,
}
```

### 3.5 Required tests for `safety_residual`

```rust
#[test]
fn safety_residual_v2_roundtrips_to_exact_safety() {
    let (public_score, exact_dealin) = (0.2f32, 0.0f32);
    let residual = (1.0 - exact_dealin) - public_score;
    let reconstructed = (public_score + residual).clamp(0.0, 1.0);
    assert!((reconstructed - 1.0).abs() < 1e-6);
}

#[test]
fn safety_residual_v2_can_be_negative() {
    let public_score = 0.9f32;
    let exact_dealin = 1.0f32;
    let residual = (1.0 - exact_dealin) - public_score;
    assert!((residual + 0.9).abs() < 1e-6);
}

#[test]
fn safety_residual_targets_stay_in_closed_interval() {
    for pub_score in [0.0, 0.2, 0.5, 0.9, 1.0] {
        for exact_dealin in [0.0, 1.0] {
            let residual = (1.0 - exact_dealin) - pub_score;
            assert!(residual >= -1.0 && residual <= 1.0);
        }
    }
}

#[test]
fn safety_residual_aux_loss_is_nonzero_when_enabled_and_present() {
    let loss_fn = HydraLoss::<B>::new(HydraLossConfig::new().with_w_safety_residual(0.05));
    let (outputs, mut targets) = make_dummy_outputs_and_targets::<B>();
    targets.safety_residual_target = Some(Tensor::<B, 2>::from_floats([[0.5; 46]; 2], &device));
    targets.safety_residual_mask = Some(Tensor::<B, 2>::ones([2, 46], &device));
    let loss: f32 = loss_fn.total_loss(&outputs, &targets).safety_residual.into_scalar().elem();
    assert!(loss > 0.0);
}
```

### 3.6 Rollout for this tranche

Keep base defaults unchanged. Add one explicit staged config:

```rust
pub fn tranche_stage1() -> HydraLossConfig {
    HydraLossConfig::new()
        .with_w_safety_residual(0.05)
        .with_w_oracle_critic(0.0)
        .with_w_belief_fields(0.0)
        .with_w_mixture_weight(0.0)
        .with_w_opponent_hand_type(0.0)
        .with_w_delta_q(0.0)
}
```

Enable only this head after a dataset audit confirms nonzero coverage.

---

## 4. Keep `belief_fields` off; reclassify it as a heuristic projection, not a belief posterior

### 4.1 Why it is semantically wrong

**Artifact support.**

`build_stage_a_belief_targets` uses:

* actor-visible/publicly reconstructible features: hand, discards, melds, dora, hand lengths, wall remaining
* `remaining = extract_public_remaining_counts(...)`
* `hidden_tiles = sum(other hand_len) + wall.remaining()`

Then `build_stage_a_teacher` does all of the following:

* `row_sums = remaining`
* `col_sums = [hidden_tiles / 4; 4]`
* `kernel = [1.0; 34 * 4]`
* runs `MixtureSib`
* emits `belief_fields`
* emits `mixture_weights` only if entropy is low enough

**Inference.**
This target is not a posterior over hidden state. It is a projection from public marginals using a uniform kernel and equalized zone totals.

That violates the basic standard used in strong imperfect-information systems. ReBeL defines a public belief state as a belief distribution determined by public observations and the policies of all agents. DeepStack similarly conditions ranges and counterfactual values on the public state and strategies. The current teacher does neither; it ignores policy conditioning and overwrites actual zone totals with equal columns. ([NeurIPS Papers][3])

### 4.2 Concrete proof of semantic failure from the supplied code

Let the four zones be the only plausible ones implied by the code path: opponent 1 hand, opponent 2 hand, opponent 3 hand, wall.

If actual hidden counts are `[13, 13, 7, 40]`, the current code projects:

[
[13,13,7,40] \rightarrow [18.25,18.25,18.25,18.25]
]

That is not a “soft uncertainty” version of the truth. It destroys known structural information.

### 4.3 The trust gate is internally inconsistent

Current trust is:

[
\text{trust}
============

\text{clamp}\left(
0.7 \cdot \frac{\text{ESS}}{K}

* 0.3 \cdot \left(1-\frac{H}{1.3863}\right),
  0,1\right)
  ]

For (K=4):

* uniform weights `[0.25,0.25,0.25,0.25]` give `ESS=4`, `H≈ln 4`, `trust≈0.70`
* one-hot weights `[1,0,0,0]` give `ESS=1`, `H=0`, `trust=0.475`

So the current gate trusts an uninformative uniform mixture more than a concentrated one. That is enough by itself to keep the target off.

### 4.4 BCE is not justified by the current object

`belief_fields_bce_per_sample` assumes Bernoulli-style probabilities/logits. The shown teacher tests only enforce “finite and nonnegative,” not “bounded in [0,1]” and not any explicit probability semantics. If the tensor means expected counts or projected masses, BCE is the wrong loss.

### 4.5 Required decision

Do not train this head in this tranche.

Keep the output surface if you want, but relabel the target internally and in audits:

* audit name: `stage_a_public_projection_fields`
* semantic state: `SemanticallyWrong`
* provenance: `HeuristicProjection`
* weight: forced `0.0`

### 4.6 Future bar for revival

Do not revisit until the target object is explicitly one of the following:

1. **Posterior occupancy**
   [
   c_{z,t} = \mathbb{E}[\text{count of tile } t \text{ in zone } z \mid h_{\text{public}}, \pi]
   ]
   with exact zone-count constraints
   [
   \sum_t c_{z,t} = N_z,\quad
   \sum_z c_{z,t} = \text{remaining}_t
   ]

2. **Zone-normalized probabilities**
   [
   p_{z,t} = \frac{c_{z,t}}{N_z}
   ]
   if and only if the loss is changed to match that semantics.

And if mixture components remain, component order must be canonicalized. Otherwise label switching makes component-specific summaries unstable or meaningless. ([Duke University Statistical Science][4])

---

## 5. Keep `mixture_weight` off; it inherits the belief failure and adds label-switching risk

### 5.1 Why it is semantically wrong

**Artifact support.** `mixture_weight_target` comes from the same Stage A teacher as `belief_fields`, under a separate entropy gate.

**Inference.**

* Because the underlying teacher is a public projection, the weights are not privileged teacher labels.
* Because the components are mixture components with no shown canonical ordering, they are vulnerable to label switching.
* Because `belief_fields` can be present when `mixture_weights` are absent, the code can train component-indexed fields while withholding the only component-level signal that might partially stabilize identity.

Mixture models with symmetric likelihoods are exactly where label switching makes component-specific summaries nonsensical unless relabelled. ([Duke University Statistical Science][4])

### 5.2 Required decision

Keep `w_mixture_weight = 0.0`. Mark the head:

```rust
HeadSpec {
    name: "mixture_weight",
    version: 0,
    semantic: SemanticState::SemanticallyWrong,
    provenance: ProvenanceKind::HeuristicProjection,
    visibility: Visibility::PublicAtInference,
}
```

Do not count this as a privileged head. It is not.

---

## 6. Keep `delta_q` off; it is a typed hole in the current dense form

### 6.1 Why it is a typed hole

**Artifact support.**

* output surface exists: `[batch, 46]`
* tests can manually inject a dummy target
* real batch path sets `delta_q_target: None`
* there is no target builder and no mask

**Inference.**
A dense 46-action `delta_q` target is not replay-credible from the supplied machinery. In imperfect-information settings, the value of an action depends on policy and belief state; strong systems use public-belief states, ranges, and counterfactual values, not replay-only dense all-action labels. Without search or counterfactual reconstruction, dense `delta_q[a]` is not defensible. ([NeurIPS Papers][3])

### 6.2 Additional implementation problem in the current loss helper

`dense_regression_mse` averages over the whole tensor. It is not a per-sample or masked action loss. Even if a target builder existed, the current helper is the wrong shape for sparse availability.

### 6.3 Required decision

Keep `w_delta_q = 0.0`. Mark the head:

```rust
HeadSpec {
    name: "delta_q",
    version: 0,
    semantic: SemanticState::TypedHole,
    provenance: ProvenanceKind::Placeholder,
    visibility: Visibility::TrainOnlyPrivileged,
}
```

### 6.4 One acceptable future narrow variant

If a replay-credible target is needed before search integration, the only acceptable interim variant is **sparse chosen-action-only** with an explicit mask and an explicit semantic rename, for example:

* `delta_q_v1_taken_only`
* mask only the executed action
* provenance: `ReplayObserved` or `ReplayDerivedPublic`, depending on builder
* do **not** pretend it is dense counterfactual `ΔQ(s,a)` for all actions

That still does **not** make the current dense head ready. It simply defines a future honest variant.

---

## 7. Keep `opponent_hand_type` off; it is a typed hole and would be train-only privileged when it exists

### 7.1 Why it is a typed hole

**Artifact support.**

* output surface exists: `[batch, 24]`
* no batch builder populates it
* `into_hydra_targets` hardcodes `None`

There is no taxonomy, no mapping from hidden hands to the 24 classes, and no provenance definition.

### 7.2 Required decision

Keep `w_opponent_hand_type = 0.0`. Mark it:

```rust
HeadSpec {
    name: "opponent_hand_type",
    version: 0,
    semantic: SemanticState::TypedHole,
    provenance: ProvenanceKind::Placeholder,
    visibility: Visibility::TrainOnlyPrivileged,
}
```

Do not populate it until a class ontology and mapping spec exist.

---

## 8. Keep `oracle_critic` privileged-only and diagnostic

### 8.1 What is supported

**Artifact support.**

* target surface exists
* optional target and mask plumbing exist
* default weight is zero
* selected test says `oracle_head_does_not_backprop_to_backbone_input`

### 8.2 Inference and decision

Treat this as a privileged-only diagnostic head, not as mainline supervision closure for this tranche. If the no-backprop test reflects actual detach from the shared representation, then training this head does not provide the student-style transfer that distillation is normally meant to provide. Keep it off by default and do not count it as tranche acceptance.

Registry entry stays:

```rust
HeadSpec {
    name: "oracle_critic",
    version: 1,
    semantic: SemanticState::DiagnosticOnly,
    provenance: ProvenanceKind::ReplayDerivedPrivileged,
    visibility: Visibility::TrainOnlyPrivileged,
}
```

---

## 9. Replace loose optional fields with structured target slots

The current `Option<T> + present bool + mask Option<T>` surface can silently fabricate false zero targets.

**Artifact support.**

* `belief_fields_present` can be true independently of `belief_fields: Option<_>`
* `mixture_weights_present` can be true independently of `mixture_weights: Option<_>`
* `safety_residual` and `safety_residual_mask` are separate `Option`s

**Inference.**
If any sample sets a present bit without values, collation creates zero-filled targets with positive sample masks.

### 9.1 Minimum patch

Add hard assertions in the sample builder and collation:

```rust
debug_assert_eq!(s.belief_fields_present, s.belief_fields.is_some());
debug_assert_eq!(s.mixture_weights_present, s.mixture_weights.is_some());
debug_assert_eq!(s.safety_residual.is_some(), s.safety_residual_mask.is_some());
```

Turn them into `Result`-returning validation in non-test builds.

### 9.2 Preferred patch

Replace loose fields with structured target types:

```rust
#[derive(Clone)]
pub struct MaskedActionArray<const A: usize> {
    pub values: [f32; A],
    pub mask: [f32; A],
}

#[derive(Clone)]
pub struct DenseArray1<const N: usize> {
    pub values: [f32; N],
}

#[derive(Clone)]
pub struct DenseArray2<const C: usize, const T: usize> {
    pub values: [f32; C * T],
}

#[derive(Clone)]
pub struct TargetMeta {
    pub head: &'static str,
    pub version: u16,
    pub semantic: SemanticState,
    pub provenance: ProvenanceKind,
    pub visibility: Visibility,
    pub trust: f32,
}

#[derive(Clone)]
pub struct SampleAdvancedTargets {
    pub safety_residual: Option<(MaskedActionArray<HYDRA_ACTION_SPACE>, TargetMeta)>,
    pub belief_fields: Option<(DenseArray2<16, 34>, TargetMeta)>,
    pub mixture_weight: Option<(DenseArray1<4>, TargetMeta)>,
    pub delta_q: Option<(MaskedActionArray<HYDRA_ACTION_SPACE>, TargetMeta)>,
    pub opponent_hand_type: Option<(DenseArray1<24>, TargetMeta)>,
}
```

This makes it impossible to have “present” without values.

---

## 10. Add a dataset audit pass and make it part of trainer startup

Before training, scan a bounded audit window, for example the first 50k samples or 512 batches.

Emit this per head:

```rust
#[derive(Debug, Clone)]
pub struct TargetAuditStats {
    pub examples_present: usize,
    pub actions_present: usize,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub frac_negative: f32,
    pub frac_zero: f32,
    pub frac_positive: f32,
}
```

For `safety_residual`, require:

* `examples_present > 0`
* `actions_present > 0`
* `min >= -1.0 && max <= 1.0`
* `frac_negative > 0.0`
* `frac_positive > 0.0`

The sign check is critical. It proves the target is no longer collapsing to `{public_score, 0}`.

Startup validation:

```rust
pub fn validate_advanced_head_enablement(
    cfg: &HydraLossConfig,
    audit: &HashMap<&'static str, TargetAuditStats>,
) -> anyhow::Result<()> {
    for spec in ADVANCED_HEAD_SPECS {
        let weight = match spec.name {
            "oracle_critic" => cfg.w_oracle_critic,
            "belief_fields" => cfg.w_belief_fields,
            "mixture_weight" => cfg.w_mixture_weight,
            "opponent_hand_type" => cfg.w_opponent_hand_type,
            "delta_q" => cfg.w_delta_q,
            "safety_residual" => cfg.w_safety_residual,
            _ => 0.0,
        };
        let present = audit.get(spec.name).map(|s| s.examples_present).unwrap_or(0);
        validate_enabled_head(*spec, weight, present)?;
    }
    Ok(())
}
```

---

## 11. Keep belief and mixture outputs honest in names and logs

Do not let names launder semantics.

In audit logs and dashboards, rename:

* `belief_fields` → `stage_a_public_projection_fields`
* `mixture_weight` → `stage_a_public_projection_weights`

Keep the tensor fields if changing struct names is too invasive, but do **not** report them as “belief” in training reports until the target object is real.

---

## 12. Acceptance checklist for this tranche

The tranche is accepted only if all of the following are true:

1. Baseline total loss is nonzero on a real train batch.
2. No advanced head with `semantic != Ready` can be enabled by config.
3. `safety_residual` is patched to version 2 signed residual semantics.
4. `safety_residual` audit shows both positive and negative targets.
5. At least one train path with `w_safety_residual > 0` produces a nonzero auxiliary contribution in tests.
6. `belief_fields`, `mixture_weight`, `delta_q`, and `opponent_hand_type` remain weight-zero and fail validation if manually enabled.
7. `oracle_critic` remains explicitly train-only privileged and is not counted as mainline supervision closure.
8. No new heads are added.
9. No AFBS/search rewrite is added.

---

## 13. Exact rollout order

### PR 1 — hard gates and registry

* verify/fix `total_loss`
* verify/fix real `collate_batch`
* add `HeadSpec`, `SemanticState`, `ProvenanceKind`, `Visibility`
* add startup dataset audit
* add config validation against the registry

### PR 2 — safety residual closure

* patch `build_safety_residual_targets`
* rename `exact_dealin_risk_from_waits` to `exact_dealin_event_from_waits`
* bump `safety_residual` registry version to `2`
* add the round-trip/sign tests
* add one staged config with `w_safety_residual = 0.05`

### PR 3 — quarantine semantically wrong and hole heads

* mark `belief_fields` and `mixture_weight` as `SemanticallyWrong`
* mark `delta_q` and `opponent_hand_type` as `TypedHole`
* keep their weights at `0.0`
* rename belief/mixture audit labels to projection names
* add validation tests that enabling any of them fails

---

## 14. Final classification

Use this exact classification going forward.

* `safety_residual`: **semantically wrong now**, **ready after one target-object patch**, **train-only privileged**, **enable in this tranche only after patch**
* `belief_fields`: **semantically wrong**, **not privileged**, **off**
* `mixture_weight`: **semantically wrong**, **not privileged**, **off**
* `delta_q`: **typed hole**, **off**
* `opponent_hand_type`: **typed hole**, **train-only privileged when it eventually exists**, **off**
* `oracle_critic`: **privileged-only diagnostic**, **off by default**

That is the strongest defensible rollout supported by the supplied artifacts and by the standard teacher/student and imperfect-information precedents. ReBeL and DeepStack show what a real hidden-state/value target looks like in imperfect-information systems: policy-conditioned public beliefs, ranges, and counterfactual values. Vapnik/Hinton/Lopez-Paz show when privileged or distilled targets are justified: when they are train-only if necessary and actually add information the student lacks. The current Stage A belief path does not meet that bar; the patched safety residual does. ([JMLR][1])

[1]: https://jmlr.csail.mit.edu/papers/volume16/vapnik15b/vapnik15b.pdf "https://jmlr.csail.mit.edu/papers/volume16/vapnik15b/vapnik15b.pdf"
[2]: https://ar5iv.org/pdf/1812.06298 "https://ar5iv.org/pdf/1812.06298"
[3]: https://papers.nips.cc/paper_files/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf "https://papers.nips.cc/paper_files/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf"
[4]: https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Mixtures/LabelSwitchingStephensJRSSB.pdf "https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/Mixtures/LabelSwitchingStephensJRSSB.pdf"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
