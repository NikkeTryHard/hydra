<combined_run_record run_id="answer_13" variant_id="prompt_and_agent_pair" schema_version="1">
  <metadata>
    <notes>Combined record for Prompt 13 and its returned agent answer.</notes>
    <layout>single_markdown_file_prompt_then_answer</layout>
  </metadata>

  <prompt_section>
  <prompt_text status="preserved" source_path="PROMPT_13_AUDIT_HYDRA_LONG_RUN_TRAINING_RISKS.md">
  <![CDATA[# Prompt 13 — long-run training risk audit blueprint

<role>
Produce an implementation-ready blueprint.
Do not give a memo.
Your answer itself must be the blueprint.
</role>

<direction>
Work toward the strongest exact audit of long-run training risks in Hydra.

We want a detailed answer that makes clear:
- what the current active training posture really is
- which fears are real versus overstated
- where rollout distillation could reduce search quality
- where the learner/actor split could lose too much signal
- where the parameter budget and target surface may be overloaded
- where sequencing protects Hydra versus where it only hides risk
- what the narrowest safe long-run path really is if the goal remains beating LuckyJ
- whether an adjacent-field or cross-field formulation solves the same problem better without breaking the artifact constraints

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
- after grounding in the artifacts, actively search adjacent fields for stronger alternative formulations of the same problem; keep them only if they survive validation against the artifacts
- use the bash tool to run Python for calculations, math checks, and validation when rigor matters
- do not finish prematurely; keep looping through discovery, thinking, testing, and validation until the information is saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
</style>

<artifact_note>
The artifacts below reflect what the current codebase/docs and prior handoffs appear to say right now. They are not guaranteed to be fully correct. Treat them as evidence to inspect and critique, not truth to inherit. High chance some of them are incomplete, misleading, stale, or semantically wrong, so validate everything.
</artifact_note>

<artifacts>
Embedded artifact/context dump intentionally omitted during normalization. Use the paired answer body plus the repository source files for audit and follow-up.
</artifacts>

]]>
  </prompt_text>
  </prompt_section>

  <answer_section>
  <answer_text status="preserved" source_path="agent_13.md">
  <![CDATA[# Hydra long-run training risk audit blueprint

```text
- rollout-net distillation reducing search quality
- learner/actor split being too lossy
- too many advanced heads / targets for the available parameter budget
- too much hidden-state/search information baked into one shared representation
- 2000 GPU-hours being enough only for the conservative Hydra, not the maximal Hydra
```

Artifact B — current task boundaries:

```text
- this is a fear-audit prompt, not a defense prompt
- broad exploration has already been done elsewhere
- do not restart broad Hydra future-planning
- use new retrieval only to validate, falsify, or sharpen the risk audit
- this is a residual risk audit after reconciliation, not a fresh architecture-prioritization exercise
```

Artifact C — current calculation and validation expectations:

```text
- use Python in bash for sample-per-parameter arithmetic, storage/throughput sanity checks, target-density calculations, and parameter-budget comparisons that matter
- do not leave numerical feasibility claims uncomputed when they can be checked
- for every major fear, define the smallest benchmark or gate that would validate or kill it
- prefer narrow measurable gates over broad architecture debates
```

Artifact D — current critique targets:

```text
Ask explicitly:
- Which parts of Hydra are robust even if rollout distillation underperforms?
- Which losses/heads are likely to learn early versus just absorb noise?
- What is most likely to be undertrained or teacher-limited under the current budget?
- Does the two-tier network actually reduce risk, or just hide it?
- If one component had to be deferred, which one should it be and why?
```

Artifact E — current plan/doctrine excerpts:

```text
The active path is the conservative staged Hydra, not the maximal fantasy Hydra.
Sequencing authority matters.
Dormant-head activation strategy should not be reopened broadly except where it creates a concrete risk under the current staged plan.
```

```text
Mainline versus reserve shelf matters:
- active path = what should be optimized for now
- reserve shelf = ideas worth keeping if the active path underdelivers
- drop shelf = ideas that should stop consuming current attention
```

Artifact F — current deliverable targets:

```text
The answer should explicitly resolve:
1. Hydra posture reconstruction for training/inference/search
2. Learner vs actor vs rollout roles
3. Parameter/capacity risk analysis
4. Compute-budget risk analysis
5. Which heads/targets should learn first vs later
6. Which fears are real vs overstated
7. Concrete gates and benchmarks to de-risk the plan
8. Final recommendation: what to narrow, what to keep, what to defer
```

Artifact G — current hard constraints:

```text
- no generic “just scale compute” answers
- no broad architecture resets
- no pretending every advanced target should be trained equally early
- no ignoring reconciliation’s sequencing authority
```

Artifact H — current primary-source bundle to inspect and critique:

```text
Include the current long-run design docs, reconciliation notes, testing notes, seeding notes, model/output/loss surfaces, eval gates, and any prior combined handoff excerpts here before sending this prompt.
Do not summarize them. Paste the relevant raw excerpts directly.
```

Artifact I — model surfaces and param-budget artifacts:

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

    pub fn estimated_params(&self) -> usize {
        ...
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

#[test]
fn actor_and_learner_param_counts_differ() {
    let a_params = actor.num_params();
    let l_params = learner.num_params();
    assert!(l_params > a_params);
    assert!(a_params > 1_000_000);
    assert!(l_params > 5_000_000);
}
```

Artifact J — loss and target-pressure artifacts:

```rust
pub struct HydraTargets<B: Backend> {
    ...
    pub belief_fields_target: Option<Tensor<B, 3>>,
    pub mixture_weight_target: Option<Tensor<B, 2>>,
    pub opponent_hand_type_target: Option<Tensor<B, 2>>,
    pub delta_q_target: Option<Tensor<B, 2>>,
    pub safety_residual_target: Option<Tensor<B, 2>>,
    ...
}

#[derive(Config, Debug)]
pub struct HydraLossConfig {
    ...
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
```

```rust
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
```

Artifact K — sample/batch/loader artifacts showing current target presence and absence:

```rust
pub struct MjaiSample {
    ...
    pub safety_residual: Option<[f32; HYDRA_ACTION_SPACE]>,
    pub safety_residual_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
    pub belief_fields: Option<[f32; 16 * 34]>,
    pub mixture_weights: Option<[f32; 4]>,
    pub belief_fields_present: bool,
    pub mixture_weights_present: bool,
}
```

```rust
HydraTargets {
    ...
    belief_fields_target: self.belief_fields_target,
    mixture_weight_target: self.mixture_weight_target,
    opponent_hand_type_target: None,
    delta_q_target: None,
    safety_residual_mask: self.safety_residual_mask,
    ...
}
```

```rust
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

Artifact L — eval and benchmark-gate artifacts:

```rust
pub struct TrainingMetrics {
    pub epoch: u32,
    pub total_loss: f64,
    pub policy_agreement: f64,
    pub value_mse: f64,
    pub games_completed: u64,
    pub arena_mean_score: f32,
    pub distill_kl: f32,
    pub elo: f32,
}

pub struct BenchmarkGates {
    pub afbs_on_turn_ms: f32,
    pub ct_smc_dp_ms: f32,
    pub endgame_ms: f32,
    pub self_play_games_per_sec: f32,
    pub distill_kl_drift: f32,
}

impl BenchmarkGates {
    pub fn passes(&self) -> bool {
        self.afbs_on_turn_ms < 150.0
            && self.ct_smc_dp_ms < 1.0
            && self.endgame_ms < 100.0
            && self.self_play_games_per_sec > 20.0
            && self.distill_kl_drift < 0.1
    }
}
```

Artifact M — reconciliation and design excerpts directly relevant to long-run risk:

```text
Main consensus:
- The biggest blocker is not missing files; it is partially closed loops plus doc drift.
- Stronger target generation is a better immediate lever than a giant search rewrite.
- AFBS should be selective and specialist, not the default path everywhere.
- Hand-EV is worth moving earlier than deeper AFBS expansion.
```

```text
What is only partially true:
- Advanced losses exist, but default advanced loss weights are zero.
- Advanced supervision hooks exist, but the normal batch collation path still mostly emits baseline targets.
- AFBS exists as a search shell, but not as a fully integrated public-belief search runtime.
- Hand-EV exists, but is still heuristic rather than a full offensive oracle.
```

```text
Two-tier network:
- LearnerNet 24-block ~10M params
- ActorNet 12-block ~5M params
Continuous distillation Learner -> Actor.
```

```text
Compute budget (2000 GPU hours on 4x RTX 5000 Ada):
- Phase -1 benchmarks 150
- Phase 0 BC 50
- Phase 1 Oracle guiding 200
- Phase 2 DRDA-wrapped ACH 800
- Phase 3 ExIt + Pondering 800
```

```text
The active path is the conservative staged Hydra, not the maximal fantasy Hydra.
```

Artifact N — more model, sample, and evaluation artifacts:

```rust
impl HydraModelConfig {
    pub fn summary(&self) -> String {
        let kind = if self.num_blocks <= 12 { "actor" } else { "learner" };
        format!("{}(blocks={}, ch={})", kind, self.num_blocks, self.hidden_channels)
    }

    pub fn is_actor(&self) -> bool {
        self.num_blocks == 12
    }

    pub fn is_learner(&self) -> bool {
        self.num_blocks == 24
    }
}

#[test]
fn actor_net_all_output_shapes() {
    assert_eq!(out.policy_logits.dims(), [batch, 46]);
    assert_eq!(out.delta_q.dims(), [batch, 46]);
    assert_eq!(out.safety_residual.dims(), [batch, 46]);
}

#[test]
fn learner_net_all_output_shapes() {
    assert_eq!(out.policy_logits.dims(), [batch, 46]);
    assert_eq!(out.delta_q.dims(), [batch, 46]);
    assert_eq!(out.safety_residual.dims(), [batch, 46]);
}
```

```rust
pub struct MjaiBatch<B: Backend> {
    pub safety_residual_target: Option<Tensor<B, 2>>,
    pub safety_residual_mask: Option<Tensor<B, 2>>,
    pub belief_fields_target: Option<Tensor<B, 3>>,
    pub mixture_weight_target: Option<Tensor<B, 2>>,
    pub belief_fields_mask: Option<Tensor<B, 1>>,
    pub mixture_weight_mask: Option<Tensor<B, 1>>,
}
```

```rust
#[test]
fn batch_to_hydra_targets_keeps_optional_advanced_targets_narrow() {
    assert!(targets.belief_fields_target.is_none());
    assert!(targets.mixture_weight_target.is_none());
    assert!(targets.opponent_hand_type_target.is_none());
    assert!(targets.delta_q_target.is_none());
    assert!(targets.safety_residual_target.is_none());
}
```

```rust
impl BenchmarkGates {
    pub fn summary(&self) -> String {
        format!(
            "afbs={:.0}ms smc={:.2}ms endgame={:.0}ms play={:.0}g/s kl={:.3}",
            self.afbs_on_turn_ms,
            self.ct_smc_dp_ms,
            self.endgame_ms,
            self.self_play_games_per_sec,
            self.distill_kl_drift
        )
    }
}
```

Artifact O — more HYDRA_FINAL compute/risk excerpts:

```text
Two-tier architecture avoids the 40-block teacher data-starvation paradox.
24-block learner handles training and deep AFBS on hard positions.
12-block actor handles self-play data generation and shallow search-as-feature inference.
```

```text
Continuous distillation learner -> actor is part of the intended system identity, not an optional garnish.
```

```text
Hard positions only:
- top-2 policy gap < 10%
- high-risk defense
- low particle ESS
```

```text
If one mechanism raises ceiling but is too slow at inference, it belongs in pondering, deep search, offline solvers, or distillation targets — not the critical inference loop.
```

Artifact P — more reconciliation and tranche-order excerpts:

```text
Recommendation 1: close advanced target generation and supervision loops.
Recommendation 2: rework Hand-EV realism before deeper AFBS expansion.
Recommendation 3: keep AFBS specialist and hard-state gated.
```

```text
The first coding tranche should populate advanced targets where feasible from existing replay/context machinery and turn on nonzero advanced loss weights in a controlled staged way.
```

```text
No new heads, no broad AFBS rewrite, no duplicated belief stack.
```

Artifact Q — more HYDRA_RECONCILIATION excerpts on active versus reserve:

```text
Working principle:
- active path = what the team should optimize for now
- reserve shelf = good ideas kept for later if the active path underdelivers
- drop shelf = ideas that should stop consuming mainline attention for now
```

```text
Do not make full public-belief search the immediate mainline.
Do not make broad “search everywhere” AFBS rollout the current identity.
Do not add more output heads before existing advanced heads are properly trained.
```

Artifact R — testing and evaluation pressure excerpts:

```text
Training pipelines happily train on wrong data and produce models that play confidently wrong.
Every component touching training data must be verified against independent ground truth.
```

```text
The live encoder/model contract is 192x34.
Old 85x34 views remain useful only as the baseline prefix and should be tested as such.
```

```text
Golden regression tests exist specifically to catch silent drift in encoded feature logic.
```

Artifact S — more loss, target, and activation artifacts:

```rust
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

pub fn batch_kl_from_target<B: Backend>(
    logits: Tensor<B, 2>,
    mask: Tensor<B, 2>,
    target: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let log_probs = masked_log_softmax(logits, mask);
    let probs = log_probs.clone().exp();
    kl_divergence(probs, target)
}

pub fn grad_norm_approx<B: Backend>(loss: Tensor<B, 1>) -> f32 {
    loss.abs().into_scalar().elem::<f32>()
}

pub fn batch_value_variance<B: Backend>(values: Tensor<B, 2>) -> Tensor<B, 1> {
    let mean = values.clone().mean_dim(0);
    let diff = values - mean;
    (diff.clone() * diff).mean_dim(0).squeeze_dim::<1>(0)
}

pub fn batch_policy_entropy<B: Backend>(logits: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 1> {
    let log_probs = masked_log_softmax(logits, mask.clone());
    let probs = log_probs.clone().exp();
    (probs * log_probs * mask).sum_dim(1).neg().mean()
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
fn test_optional_belief_losses_default_to_zero() {
    assert!(belief.abs() < 1e-8);
    assert!(mixture.abs() < 1e-8);
    assert!(hand_type.abs() < 1e-8);
    assert!(delta_q.abs() < 1e-8);
    assert!(safety_residual.abs() < 1e-8);
}

#[test]
fn test_optional_belief_losses_activate_when_targets_present() {
    targets.delta_q_target = Some(Tensor::<B, 2>::zeros([2, 46], &device));
    targets.safety_residual_target = Some(Tensor::<B, 2>::zeros([2, 46], &device));
    targets.safety_residual_mask = Some(Tensor::<B, 2>::ones([2, 46], &device));
    assert!(delta_q.is_finite() && delta_q >= 0.0);
    assert!(safety_residual.is_finite() && safety_residual >= 0.0);
}
```

Artifact T — more AFBS and pondering runtime artifacts:

```rust
pub const C_PUCT: f32 = 2.5;
pub const TOP_K: usize = 5;
pub const MIN_BATCH: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct PonderResult {
    pub exit_policy: [f32; HYDRA_ACTION_SPACE],
    pub value: f32,
    pub search_depth: u8,
    pub visit_count: u32,
    pub timestamp: Instant,
}

pub struct GameStateSnapshot {
    pub info_state_hash: u64,
    pub top2_policy_gap: f32,
    pub risk_score: f32,
    pub particle_ess: f32,
}

pub fn compute_ponder_priority(top2_gap: f32, risk_score: f32, particle_ess: f32) -> f32 {
    let gap_term = (0.1 - top2_gap).max(0.0) * 10.0;
    let risk_term = risk_score.max(0.0);
    let ess_term = (1.0 - particle_ess).max(0.0);
    gap_term + risk_term + ess_term
}
```

```rust
#[test]
fn ponder_cache_hit_reuses_search() {
    let cache = PonderCache::new();
    let result = PonderResult {
        exit_policy: [0.0; HYDRA_ACTION_SPACE],
        value: 0.5,
        search_depth: 4,
        visit_count: 100,
        timestamp: Instant::now(),
    };
    cache.insert(42, result);
    let hit = cache.get(42).expect("should find cached result");
    assert_eq!(hit.visit_count, 100);
}

#[test]
fn ponder_manager_prioritizes_higher_score() {
    manager.enqueue_snapshot(GameStateSnapshot {
        info_state_hash: 1,
        top2_policy_gap: 0.2,
        risk_score: 0.1,
        particle_ess: 0.9,
    });
    manager.enqueue_snapshot(GameStateSnapshot {
        info_state_hash: 2,
        top2_policy_gap: 0.01,
        risk_score: 0.9,
        particle_ess: 0.2,
    });
    let next = manager.pop_task().expect("queued task");
    assert_eq!(next.info_state_hash, 2);
}
```

Artifact U — more bridge and encoder artifacts:

```rust
pub fn build_search_features(
    safety: &SafetyInfo,
    context: &SearchContext<'_>,
) -> SearchFeaturePlanes {
    let mut features = SearchFeaturePlanes::default();

    if let Some(mixture) = context.mixture {
        ...
        features.mixture_entropy = mixture.weight_entropy() as f32;
        features.mixture_ess = mixture.ess() as f32;
        features.belief_features_present = true;
        features.context_features_present = true;
    }

    if let (Some(tree), Some(root)) = (context.afbs_tree, context.afbs_root) {
        let root_q = tree.node_q_value(root);
        for action in 0..NUM_TILE_TYPES as u8 {
            if let Some(child) = tree.find_child_by_action(root, action) {
                features.delta_q[action as usize] = tree.node_q_value(child) - root_q;
            }
        }
        features.search_features_present = true;
        features.context_features_present = true;
    }
    features
}
```

```text
Current sources:
- Mixture-SIB -> belief fields, weights, entropy, ESS
- AFBS root -> discard-level delta-Q summary for expanded discard actions
- safety/opponent model cache -> per-opponent stress and matagi danger fallback
```

Artifact V — more HYDRA_FINAL training and risk excerpts:

```text
40-block teacher trained only on hard states gets just ~7 samples per parameter — catastrophic data starvation.
Two-tier architecture avoids this paradox.
```

```text
Phase -1 hard reality benchmarks must pass before committing the full budget:
- latency gate
- throughput gate
- distillation gate
- hyperparameter sweep
If gates fail, shrink AFBS/teacher usage and reallocate to more self-play.
```

```text
RolloutNet is ActorNet-sized and continuously distilled from LearnerNet.
```

```text
Hard positions only:
top-2 policy gap < 10%, high-risk defense, low particle ESS.
```

Artifact W — more HYDRA_RECONCILIATION excerpts on long-run risk:

```text
What is only partially true:
- advanced losses exist, but default advanced loss weights are zero
- advanced supervision hooks exist, but the normal batch collation path still mostly emits baseline targets
- AFBS exists as a search shell, but not as a fully integrated public-belief search runtime
- Hand-EV exists, but is still heuristic rather than a full offensive oracle
- Endgame exists, but as weighted particle/PIMC evaluation rather than true exactification
```

```text
Do not make full public-belief search the immediate project identity.
Do not center the roadmap on broad expensive search.
Reduce architectural confusion before scaling implementation.
```

```text
Best immediate next move:
close advanced target generation and supervision loops.
```

Artifact X — more testing pressure excerpts:

```text
Testing is critical for a mahjong AI because engine bugs silently corrupt training data.
A single incorrect legal action mask, a mis-scored hand, or a wrong tile encoding feeds the neural network garbage labels for hundreds of thousands of training steps before anyone notices.
```

```text
Unlike a web app where users report bugs, a training pipeline happily trains on wrong data and produces a model that plays confidently wrong.
```

Artifact Y — more sample, batch, and loader artifacts:

```rust
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
```

```rust
pub fn score_delta_to_value(score_delta: i32) -> f32 {
    (score_delta as f32 / 100_000.0).clamp(-1.0, 1.0)
}

pub fn score_delta_to_pdf(score_delta: i32) -> [f32; SCORE_BINS] {
    let mut pdf = [0.0f32; SCORE_BINS];
    pdf[score_delta_to_bin(score_delta)] = 1.0;
    pdf
}

pub fn score_delta_to_cdf(score_delta: i32) -> [f32; SCORE_BINS] {
    let bin = score_delta_to_bin(score_delta);
    let mut cdf = [0.0f32; SCORE_BINS];
    for v in &mut cdf[bin..] {
        *v = 1.0;
    }
    cdf
}
```

```rust
#[test]
fn test_batch_shapes() {
    let batch = collate_batch::<B>(&samples, &device);
    assert_eq!(batch.obs.dims(), [32, NUM_CHANNELS, 34]);
    assert_eq!(batch.actions.dims(), [32]);
    assert_eq!(batch.legal_mask.dims(), [32, 46]);
    assert_eq!(batch.value_target.dims(), [32]);
    assert_eq!(batch.grp_target.dims(), [32, 24]);
    assert!(batch.oracle_target.is_none());
    assert_eq!(batch.tenpai_target.dims(), [32, 3]);
    assert_eq!(batch.danger_target.dims(), [32, 3, 34]);
    assert!(batch.safety_residual_target.is_none());
    assert_eq!(batch.opp_next_target.dims(), [32, 3, 34]);
    assert_eq!(batch.score_pdf_target.dims(), [32, 64]);
    assert_eq!(batch.score_cdf_target.dims(), [32, 64]);
}

#[test]
fn test_score_pdf_is_one_hot() {
    let pdf = score_delta_to_pdf(5000);
    let sum: f32 = pdf.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_score_cdf_monotonic() {
    let cdf = score_delta_to_cdf(5000);
    for i in 1..64 {
        assert!(cdf[i] >= cdf[i - 1]);
    }
}
```

```rust
#[test]
fn load_game_from_reader_extracts_samples() {
    let game = load_game_from_reader(Cursor::new(log.join("\n"))).expect("load game");
    assert!(game.samples.len() > 50);
    assert!(
        game.samples
            .iter()
            .all(|sample| sample.legal_mask[sample.action as usize] > 0.0)
    );
}

#[test]
fn load_game_from_reader_populates_oracle_targets_from_final_scores() {
    let expected = oracle_target_from_scores(final_scores);
    for sample in game.samples.iter().take(8) {
        let got_target = sample.oracle_target.expect("oracle target should be present");
        for (got, want) in got_target.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6);
        }
    }
}
```

Artifact Z — more inference and deployment artifacts:

```rust
pub struct InferenceConfig {
    pub on_turn_budget_ms: u64,
    pub call_reaction_budget_ms: u64,
    pub agari_guard: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            on_turn_budget_ms: 150,
            call_reaction_budget_ms: 50,
            agari_guard: true,
        }
    }
}
```

```rust
pub fn infer_with_budget(
    &self,
    obs: &[f32; OBS_FLAT_SIZE],
    legal: &[bool; HYDRA_ACTION_SPACE],
    budget_ms: u64,
) -> (u8, [f32; HYDRA_ACTION_SPACE], bool) {
    let info_state_hash = Self::info_state_hash(obs);

    if let Some(pondered) = self.lookup_ponder(info_state_hash) {
        let policy = mask_policy_cpu(&pondered.exit_policy, legal);
        let action = self.guard_action(argmax_legal(&policy, legal), legal);
        let within = start.elapsed().as_millis() as u64 <= budget_ms;
        return (action, policy, within);
    }

    let base_logits = self.actor.policy_logits_for(input);
    let logits = self.apply_saf_fast_path(base_logits, obs, legal);
    let (action, policy, within) = infer_action_timed(logits, legal, budget_ms);
    (self.guard_action(action, legal), policy, within)
}
```

```rust
#[test]
fn inference_server_respects_time_budget() {
    server.config.on_turn_budget_ms = 5_000;
    let (action, policy, within) = server.infer_timed(&obs, &legal);
    assert!(legal[action as usize]);
    assert!(within);
    assert!((policy.iter().sum::<f32>() - 1.0).abs() < 0.01);
}
```

Artifact AA — more doctrine and benchmark excerpts:

```text
Pondering = label amplification. 75% idle time used for deepening current root search and precomputing searches for predicted near-future states.
```

```text
Search-as-Feature returns delta-Q(a), risk estimates, epistemic terms, robust stress indicators, and uncertainty terms.
```

```text
SaF-dropout is used during training to prevent over-reliance on search-derived features when present.
```

```text
Validation gates:
G0 positive decision improvement.
G1 robustness calibration.
G2 safety bound usefulness.
G3 SaF amortization.
```

Artifact AB — more reconciliation and reserve-shelf excerpts:

```text
Keep DRDA/ACH as the intended target architecture direction from HYDRA_FINAL, but do not make immediate implementation decisions depend on resolving every optimizer-level debate first.
```

```text
Keep robust-opponent search backups and selective exactification on the reserve shelf until supervision and feature realism improve.
```

```text
Do not expand head count further; first feed the existing advanced heads with better targets.
```

```text
These ideas are reserve, not active, because they are not obviously wrong but add enough complexity that they should not steer the next coding tranche.
```

Artifact AC — more testing and benchmarking excerpts:

```text
Model smoke tests:
- forward pass with random input [1, 192, 34]
- legal action masking
- inference output within tolerance
```

```text
Loss function tests:
- policy CE with known logits/labels
- GRP 24-way CE
- focal BCE
- composite weighted sum
```

```text
Data pipeline tests:
- batches of correct shape [2048, 192, 34]
- suit permutation diversity
- filtering and manifests
```

Artifact AD — final additional long-run risk artifacts:

```text
The repo is closer to “advanced baseline with partially inactive loops” than to “missing everything.”
```

```text
The best immediate next move is reconciliation plus a supervision-first first coding tranche.
```

```text
The first coding tranche should not widen into privileged-path / alignment work, a teacher-posterior tranche, Hand-EV realism, or a large runtime/search tranche all at once.
```

```text
The biggest risk if scope expands is turning a narrow tranche into a hidden second tranche.
```

```rust
#[test]
fn stable_dan_formula() {
    let dan_perfect = compute_stable_dan(1.0);
    assert!((dan_perfect - 10.0).abs() < 0.01);
    let dan_avg = compute_stable_dan(2.5);
    assert!(dan_avg > 0.0 && dan_avg < 10.0);
}

#[test]
fn eval_result_defaults() {
    let result = EvalResult::default();
    assert!((result.mean_placement - 2.5).abs() < 0.01);
}

#[test]
fn eval_reports_all_metrics() {
    let placements = vec![0, 0, 1, 2, 3, 1];
    let result = evaluate_from_placements(&placements);
    assert!(result.mean_placement > 1.0 && result.mean_placement < 4.0);
    assert!(result.stable_dan >= 0.0);
    assert!(result.win_rate > 0.0);
}
```

```rust
#[test]
fn validate_passes_for_standard_configs() {
    assert!(HydraModelConfig::actor().validate().is_ok());
    assert!(HydraModelConfig::learner().validate().is_ok());
}

#[test]
fn all_outputs_finite_for_random_input() {
    assert!(out.is_finite());
}
```


**Evidence labels**

* **Artifact** = directly supported by the supplied Hydra artifacts.
* **Paper** = supported by external primary sources.
* **Inference** = conclusion from Artifact + Paper.

## 1. Operational truth: what Hydra is right now

**Artifact.** Hydra today is a **two-tier single-family model**, not a family of separately scoped networks. `ActorNet` and `LearnerNet` are both `HydraModel`; actor = 12 blocks, learner = 24 blocks; both emit the full output surface, including policy, value, score PDF/CDF, tenpai, GRP, opp-next-discard, danger, oracle, belief, mixture, delta-Q, and safety residual. The intended identity is continuous `Learner -> Actor` distillation, with the actor also serving as rollout-sized inference.

**Artifact.** The *active training posture* is much narrower than the forward signature suggests. Advanced heads exist, but the normal batch path mostly does **not** feed them, and the advanced loss weights default to `0.0`. The repo is therefore closer to an **advanced baseline with partially inactive loops** than to a fully active maximal Hydra.

| Area              | Live posture now                                                             | Audit consequence                                                |
| ----------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Core model        | Full multi-head forward exists for both actor and learner                    | Capacity split exists, but target surface is shared              |
| Mainline training | Baseline dense targets dominate; advanced weights default to zero            | Hydra is not currently training “maximal Hydra”                  |
| Search runtime    | AFBS/search exists as a shell; specialist and hard-state gated               | Search is a support mechanism, not the mainline identity         |
| Inference         | 12-block actor under 150 ms on-turn / 50 ms reaction budget with agari guard | Any real gain must transfer to actor or stay explicitly off-loop |
| Distillation      | Continuous learner-to-actor distillation is mandatory, not optional          | Distillation quality is a first-class risk                       |

**Artifact.** The current hard-state/search gating is already explicit: deep search / pondering is for states with low top-2 policy gap, high defensive risk, or low particle ESS. Search-as-feature is present, and SaF-dropout already exists to reduce dependence on those derived features.

**Blueprint rule.** Treat Hydra as:

[
\text{Hydra today} = \text{dense baseline trunk} + \text{specialist hard-state search} + \text{mostly dormant advanced supervision}.
]

Do **not** plan or evaluate it as if all heads are already learning.

---

## 2. Phase map under the existing 2000 GPU-hour budget

**Artifact.** Use the current budget split as the control surface, not a new roadmap:

* **Phase -1 (150 GPUh):** instrumentation, correctness, latency, throughput, distillation sanity
* **Phase 0 (50 GPUh):** baseline BC
* **Phase 1 (200 GPUh):** oracle guiding / value realism tranche
* **Phase 2 (800 GPUh):** DRDA-wrapped ACH learner training
* **Phase 3 (800 GPUh):** ExIt + pondering only if prior gates pass

**Blueprint rule.** Every new head or search feature must earn entry through these phases. No hidden second tranche.

---

## 3. Fear-by-fear audit

### 3.1 Rollout-net distillation reducing search quality

**Verdict:** **real**

**Artifact.** Hydra makes distillation structural: rollout/actor is learner-sized? No; rollout is actor-sized, continuously distilled from the learner. The code already has `soft_target_from_exit(...)` and a KL drift benchmark, but that only checks distributional closeness, not whether distillation harms the search loop that later consumes the actor prior.

**Paper.** ExIt works because search is the local expert and the network is the global generalizer; but ExIt also says search-generated datasets dominate runtime, and online aggregation of all datasets is the practical improvement over restarting from scratch. GKD/DAgger-style results make the core distillation failure mode explicit: fixed expert-labeled data creates train-inference mismatch, while relabeling **student-generated** states reduces that mismatch. In imperfect-information games, ReBeL and DeepStack both show that policy/value quality depends on belief/public-state context; compressing search-conditioned decisions into a smaller student without respecting that context is exactly where fidelity is lost. ([NeurIPS Papers][1])

**Inference.** Hydra’s failure channels are:

1. **Hard-state dilution.** Search is only invoked on rare hard states, but plain KL is dominated by easy states.
2. **Unavailable-feature leakage.** Search-derived features may be present during training and absent during actor-only inference.
3. **Top-k collapse.** Small KL can still hide critical top-1 / top-2 swaps on dangerous actions.
4. **Search-prior regression.** A distilled actor can weaken future search if that actor is later used as the prior/value source inside the search loop.

**Gate that validates or kills the fear**

Create a fixed hard-state corpus (H) from **actor-generated** states, not learner-generated states.

[
H={s:\ \text{gap}(s)<0.10 \ \lor\ \text{risk}(s)>r_0 \ \lor\ \text{ESS}(s)<e_0 \ \lor\ a_A(s)\neq a_T(s)}
]

Then require:

* hard-set masked KL (\le 0.10)
* hard-set top-2 recall (\ge 0.90)
* hard-set teacher-regret reduced by at least 25% versus the undistilled actor
* **search-prior regression** non-negative at fixed node/time budget

Define teacher-regret on searched states as:

[
R_{\text{distill}}=\mathbb{E}_{s\in H}\left[Q_T(s,a_T)-Q_T(s,a_A)\right].
]

**Implementation**

Use the existing hard-state score, then make the distillation mix state-dependent instead of global:

```rust
fn hard_score(gap: f32, risk: f32, ess: f32) -> f32 {
    let gap_term = (0.1 - gap).max(0.0) * 10.0;
    let risk_term = risk.max(0.0);
    let ess_term = (1.0 - ess).max(0.0);
    gap_term + risk_term + ess_term
}

fn distill_mix(gap: f32, risk: f32, ess: f32) -> f32 {
    (0.15 + 0.15 * hard_score(gap, risk, ess)).clamp(0.15, 0.85)
}
```

And replace “KL everywhere” with “teacher-heavy only where search matters”:

```rust
if is_hard_state(snapshot) || actor_action != teacher_action {
    loss += jsd_topk(actor_logits, teacher_policy, legal_mask, 8);
    loss += rank_margin_loss(actor_logits, teacher_top1, teacher_top2, 0.2);
    loss += teacher_regret_loss(actor_logits, teacher_delta_q, legal_mask);
}
```

**Search-prior regression test**

Freeze a 10k-state hard corpus (P). Run search with old actor prior and new actor prior at the same budget.

[
\Delta_{\text{prior}} = EV_{\text{search}}(\pi_{A,new}) - EV_{\text{search}}(\pi_{A,old})
]

Continue only if (\Delta_{\text{prior}} \ge 0) within confidence intervals.

**Conclusion.** Distillation is not the problem. **Broad fixed-distribution distillation** is the problem. The narrow fix is **actor-induced hard-state relabeling with aggregated buffers**.

---

### 3.2 Learner/actor split being too lossy

**Verdict:** **real, but overstated as an architecture objection**

**Artifact.** The split is already a deliberate defense against the “40-block hard-state teacher gets ~7 samples/param” starvation problem. The actor exists to satisfy latency and self-play throughput; the learner exists to absorb slower supervision and deeper search.

**Paper.** IMPALA makes the general point cleanly: decoupled actor/learner systems are powerful, but policy lag is real and needs explicit correction. Suphx also decoupled inference/self-play from parameter updates and repeatedly refreshed policies to keep acting close to the latest learner. So the split itself is not suspect; **unmeasured lag and transfer loss** are suspect. ([Proceedings of Machine Learning Research][2])

**Inference.** Signal loss will concentrate in exactly the places where Hydra cannot afford it:

* low-gap decisions
* defensive reactions under short budgets
* sparse search-derived corrections
* any head that depends on search/belief context rather than raw observation

Average policy agreement is not enough.

**Gate that validates or kills the fear**

Do not credit teacher-only gains. Define:

[
\Delta_{L} = EV(\text{learner loop}) - EV(\text{baseline actor loop})
]
[
\Delta_{A} = EV(\text{actor loop after distill}) - EV(\text{baseline actor loop})
]
[
\text{creditable gain}=\min(\Delta_L,\Delta_A)
]

If (\Delta_L>0) but (\Delta_A\le 0), the learner/actor split is **hiding risk**, not reducing it.

Require the actor to preserve at least 70% of learner-side lift on the hard-state corpus before expanding the learner-only target surface.

**Implementation**

Add versioned distill samples:

```rust
pub struct DistillSample {
    pub obs: [f32; OBS_SIZE],
    pub legal_mask: [f32; HYDRA_ACTION_SPACE],
    pub actor_version: u64,
    pub hard_reason_bits: u8,
    pub teacher_policy: [f32; HYDRA_ACTION_SPACE],
    pub teacher_top1: u8,
    pub teacher_top2: u8,
    pub teacher_value: f32,
    pub teacher_delta_q: Option<[f32; HYDRA_ACTION_SPACE]>,
}
```

Add transfer metrics:

```rust
pub struct TransferMetrics {
    pub hard_top1_agreement: f32,
    pub hard_top2_recall: f32,
    pub hard_teacher_regret: f32,
    pub actor_retained_gain: f32, // Delta_A / Delta_L
}
```

**Conclusion.** Keep the split. But from now on, **teacher-only improvement does not count**.

---

### 3.3 Too many advanced heads / targets for the available parameter budget

**Verdict:** **overstated as a raw-parameter fear; real as a supervision-density and interference fear**

**Artifact.** Known output dimensionality already exceeds **1050 outputs excluding `opponent_hand_type`**. The dormant optional surfaces alone contribute **640 dims** before counting `opponent_hand_type` and before treating oracle separately. But the head parameter formula is linear in hidden width:

[
P_{\text{heads}} \approx h \cdot (1050 + k_{\text{handtype}}) + O(1).
]

That means the output heads are unlikely to be the dominant parameter consumer in a 5M/10M model. The dominant problem is not “too many linear heads”; it is **too many weakly supervised surfaces trying to shape one trunk**.

**Artifact.** The strongest evidence is current target density, not head count: the main batch path still emits mostly baseline targets; `delta_q_target` and `opponent_hand_type_target` are absent; belief/mixture/safety have hooks but are mostly not live end-to-end.

**Paper.** Deep-AMTFL’s point is exactly Hydra’s risk: reliable/easy tasks should shape shared features more than inaccurate/hard tasks, because noisy hard tasks can drag the shared representation backward through negative transfer. ([Proceedings of Machine Learning Research][3])

**Implementation rule**

For each head (h), log:

[
\rho_h = \frac{#\text{samples with target }h}{N}
]

and, for sparse search-derived heads,

[
\text{spp}*h = \frac{N_h}{P*{\text{learner}}}
]

where (N_h) is the number of labeled states for that head and (P_{\text{learner}}) is learner params.

Activate a head only if:

* label audit passes
* dense head: (\rho_h \ge 0.8)
* sparse search head: (\text{spp}_h \ge 5)
* negative gradient cosine against policy/value is not persistent

**Important artifact correction**

Do **not** use `grad_norm_approx(...)` for this gate. In the artifact it is a loss-magnitude proxy, not a true parameter-gradient norm.

Use real shared-trunk gradient cosine instead:

```rust
fn grad_cosine(loss_a: Tensor<B, 1>, loss_b: Tensor<B, 1>, shared: &[Param]) -> f32 {
    zero_grads(shared);
    loss_a.backward();
    let ga = flatten_grads(shared);

    zero_grads(shared);
    loss_b.backward();
    let gb = flatten_grads(shared);

    cosine(&ga, &gb)
}
```

Keep a head off if shared-trunk cosine with `policy+value` is negative on more than 30% of batches after warmup.

**Worked sample-per-parameter check**

Take the current gate literally:

* self-play floor: **20 games/s**
* GPUs: **4**
* phases with real search/distill pressure: **1600 GPUh** (Phase 2 + 3)
* worked example: **60 decisions/game**
* learner size: **10M params**

Then:

[
N_{\text{positions}} = 20 \cdot 3600 \cdot (1600/4) \cdot 60 = 1.728\times 10^9
]

So:

[
\text{spp}*{\text{hard}} = 172.8 \cdot f*{\text{hard}}
]

where (f_{\text{hard}}) is the fraction of positions that actually get search labels.

That gives:

* (f_{\text{hard}}=1%\Rightarrow 1.73) samples/param
* (f_{\text{hard}}=2%\Rightarrow 3.46)
* (f_{\text{hard}}=5%\Rightarrow 8.64)

**Inference.** Any search-derived head trained on only **1–2%** of positions is probably undertrained on the current budget. `delta_q` is the clearest casualty.

**Conclusion.** The current risk is not “too many params for too many heads.” The real risk is **lighting up sparse noisy heads before density and transfer exist**.

---

### 3.4 Too much hidden-state/search information baked into one shared representation

**Verdict:** **real, but currently latent**

**Artifact.** Search features already include mixture entropy/ESS, AFBS-root delta-Q summaries, and safety/opponent-cache signals. SaF-dropout exists, which is the correct direction.

**Paper.** Deep-AMTFL supports asymmetric transfer from reliable tasks into shared features, not the reverse. DeepStack/ReBeL support the deeper warning: in imperfect-information games, the correct local action depends on belief/public-state quantities that are not just “the visible board encoded harder.” ([Proceedings of Machine Learning Research][3])

**Inference.** The representation risk is dormant now because those heads are mostly off. It becomes real when belief, safety, and search heads all backprop into the same trunk together.

**Gate that validates or kills the fear**

For any model trained with search-derived features or belief targets, run a **feature-ablation evaluation**:

[
\Delta_{\text{SaF-off}} = EV(\text{features on}) - EV(\text{features zeroed})
]

Continue only if the actor remains usable with features zeroed:

* stable-dan drop < 0.1
* hard-state teacher-regret increase < 10%

**Implementation**

When turning on any sparse head, do it in two steps:

1. **head-only warmup** with trunk frozen for 10k-20k updates
2. unfreeze trunk only if feature-ablation and gradient-conflict gates pass

That is the narrowest safe anti-entanglement move that does **not** require a new architecture.

---

### 3.5 2000 GPU-hours being enough only for conservative Hydra, not maximal Hydra

**Verdict:** **very real**

**Artifact + arithmetic.** With the existing gate of **20 self-play games/s** and 4 GPUs:

[
\text{wall hours} = 2000/4 = 500
]
[
N_{\text{games}} = 20 \cdot 3600 \cdot 500 = 36{,}000{,}000
]

That is an upper bound, not a promise. If you use a worked 60 decisions/game, that is about **2.16B state-action positions** across the full budget. Phase 3 alone is **14.4M games** and about **864M positions** at that same worked density.

That sounds large until you remember two things:

1. search-labeled states are only a fraction of those positions
2. the budget must cover BC, oracle/value work, ACH, ExIt, pondering, and distillation—not one narrow training objective

**Paper.** The scale comparisons all point the same way. Suphx reports one RL agent run at **1.5M games**, costing **44 GPUs for two days** (~2112 GPU-hours). ACH’s published 1v1 Mahjong platform used **800 CPUs, 3200 GB memory, and 8 M40 GPUs**. ReBeL explicitly says data generation is the bottleneck and used up to **128 machines with 8 GPUs each** for generation. Tencent’s OLSS paper is even more telling for Mahjong-adjacent search: the two-player Mahjong setup used **8 V100s + 1200 CPUs** for blueprint training, **8 V100s + 2400 CPUs** for the environment model, then simplified online search to **discard-only**, **8-step** pUCT rollouts because richer search was too expensive. Hardware differs, so these are directional comparisons, not apples-to-apples throughput conversions—but directionally they all reject “maximal Hydra under 2000 GPUh.” ([arXiv][4])

**Conclusion.** The current budget fits **conservative staged Hydra**. It does **not** fit “turn on every advanced head, broaden AFBS, and search-distill everything.”

---

## 4. Where sequencing protects Hydra, and where it only hides risk

### Sequencing protects Hydra here

* **Zero default weights** keep bad sparse heads from poisoning the trunk too early.
* **Selective hard-state search** matches ExIt’s economics: search is expensive; use it where it adds local information, not everywhere. ([NeurIPS Papers][1])
* **Two-tier learner/actor** avoids the teacher-data-starvation paradox while preserving inference latency.
* **Hand-EV/value realism before AFBS breadth** matches the external pattern from Suphx: global reward prediction and oracle guiding added gains before runtime adaptation became the story. ([arXiv][4])

### Sequencing hides risk here

* A dormant head can look “implemented” while still being completely untrained.
* A learner-side gain can look “real” while never transferring to the actor.
* A global KL metric can look “stable” while hard-state regret stays bad.
* SaF-dropout can look “present” while the actor still quietly depends on search context.

**Blueprint rule:** a gain is real only if it survives the actor loop or remains explicitly inside the deployed search loop within budget.

---

## 5. Which heads should learn first, later, or not now

### Learn first

1. **policy_logits**
2. **value**
3. **score_pdf / score_cdf**
4. **grp**
5. **opp_tenpai**
6. **opp_next_discard**
7. **danger**

These are the dense, immediate, low-ambiguity surfaces.

### First advanced candidate

**`oracle_critic`**, but only after a density audit. The artifacts are inconsistent: oracle targets can be populated from final scores, but one batch test shows them absent in a standard fixture. So the correct move is **measure `ρ_oracle` first**, then activate at low weight if dense enough.

### Learn next, one at a time

1. **`safety_residual`**
   Reason: action-local, masked, directly relevant to defense, and the builder exists.

2. **`belief_fields` + `mixture_weight_logits`**
   Reason: stage-A targets and masks already exist in the data structures, so the loop can be closed without new heads. But they need calibration, not faith.

### Defer

**`delta_q`**
This is the one active-looking component that should be deferred. It is sparse, search-derived, teacher-limited, currently absent in the mainline target path, and it is exactly the head most likely to be data-starved under the current budget.

### Drop shelf for now

**`opponent_hand_type`**
No current mainline target path. Stop spending attention on it until labels exist.

### Not a head, but move earlier

**Hand-EV realism**
Move this earlier than deeper AFBS expansion. Suphx’s strongest jumps came from better value/reward shaping and oracle guidance; its Tenhou edge also showed up as strong defense and low deal-in / low 4th-place rates, which is much closer to Hydra’s value/safety stack than to a broad search rewrite. ([arXiv][4])

---

## 6. Minimal gate pack

### Gate 1 — target correctness

For every advanced target generator, verify against independent ground truth before any nonzero weight.

* legal masks: exact
* score / rank transforms: exact
* safety residual: exact or reference-equivalent on audited cases
* belief/mixture: reveal-state validation
* oracle target: deterministic from final scores

No training credit if this gate fails.

### Gate 2 — density

Log per-head density:

```rust
pub struct HeadCoverage {
    pub rho_oracle: f32,
    pub rho_belief: f32,
    pub rho_mixture: f32,
    pub rho_safety: f32,
    pub rho_delta_q: f32,
    pub rho_hand_type: f32,
}
```

Activation rule:

* dense direct head: `rho >= 0.8`
* sparse search head: `samples_per_param >= 5`

### Gate 3 — hard-state distillation

Use a 50k-state actor-induced hard corpus.

Pass only if:

* hard KL <= 0.10
* top-2 recall >= 0.90
* teacher-regret down >= 25% vs baseline actor

### Gate 4 — search-prior regression

Use a fixed 10k-state hard corpus.

Pass only if a newly distilled actor does **not** reduce search EV at a fixed node/time budget.

### Gate 5 — no-SaF dependence

Train with SaF if needed; evaluate with SaF zeroed.

Pass only if:

* stable-dan drop < 0.1
* hard-state regret increase < 10%

### Gate 6 — belief calibration

Use a reveal-state corpus of at least 100k states.

Pass only if belief/mixture beats a marginal-frequency baseline on calibration metrics (Brier/ECE) by at least 10%.

### Gate 7 — safety usefulness

Use a high-risk defensive arena slice.

Pass only if adding `safety_residual` reduces deal-in rate without reducing stable-dan.

### Gate 8 — compute

Keep the current artifact gates, and add “throughput regression after activation”:

* `afbs_on_turn_ms < 150`
* `ct_smc_dp_ms < 1`
* `endgame_ms < 100`
* `self_play_games_per_sec > 20`
* `distill_kl_drift < 0.1`
* post-activation throughput drop < 10%

If Gate 8 fails, follow the artifact doctrine exactly: **shrink AFBS/teacher usage and reallocate to self-play**.

---

## 7. Adjacent formulations: what survives validation

### Keep: hard-state on-policy distillation

This is the strongest adjacent formulation that survives the artifact constraints.

It is a hybrid of:

* ExIt’s expert/apprentice split
* DAgger/GKD’s actor-induced relabeling
* Hydra’s existing hard-state gating

It solves the right problem: **distribution-mismatched distillation on the rare states that matter most**. ([NeurIPS Papers][1])

### Keep: asymmetric multi-head activation

This is the strongest cross-field answer to “too many heads on one trunk” without changing the artifact constraints.

Implement it as:

* reliable dense heads first
* sparse/noisy heads one at a time
* head-only warmup
* gradient-conflict monitoring
* density and calibration gates

That is the Deep-AMTFL lesson translated into Hydra’s existing architecture, not a new architecture. ([Proceedings of Machine Learning Research][3])

### Reserve shelf: OLSS-style opponent-limited search

Tencent’s OLSS result is the most relevant search-adjacent alternative to Hydra’s selective AFBS doctrine. It shows that limiting opponent strategy space can be **orders of magnitude faster** than larger common-knowledge subgame-solving methods, and that in 2-player Mahjong one or two opponent strategies can already improve online performance. But the same paper also states that OLSS-II becomes **unsafe when the number of opponent strategies is limited**, and its Mahjong experiments had to simplify search heavily to discard-only, eight-step pUCT rollouts. That makes OLSS a **reserve-shelf search restriction**, not a mainline safety-certified Hydra identity. ([Proceedings of Machine Learning Research][5])

### Reserve shelf only: ReBeL / full public-belief RL+Search

The theory is cleaner than Hydra’s current partial search shell: ReBeL is explicit that AlphaZero-style RL+Search is not sound in imperfect-information games without public-belief state, and it fixes the state definition. But it is also a bigger architecture reset, with heavy data-generation cost. Under current artifact constraints and current budget, this is a reserve-shelf north star, not the next coding tranche. ([NeurIPS Proceedings][6])

### Reject now

* full public-belief search mainline
* broad “search everywhere” AFBS
* new output heads
* duplicate belief stack
* any plan that counts learner-only lift as mainline progress

---

## 8. Evaluation protocol

Use three fixed corpora plus paired-seed arena matches.

### Corpus H — hard-state distill corpus

50k actor-induced hard states with teacher search labels.

### Corpus R — reveal-state calibration corpus

100k states from completed games with hidden information revealed.

### Corpus P — search-prior regression corpus

10k hard states, frozen across distillation rounds.

### Arena protocol

Use paired-seed / duplicate-style evaluation wherever possible so variance does not swamp small improvements. OLSS’s Mahjong evaluation explicitly used many games and paired deck/seat swaps to reduce variance; Hydra should adopt the same spirit for go/no-go decisions. ([Proceedings of Machine Learning Research][5])

---

## 9. Final shelf decision

### Mainline now

* keep the **conservative staged Hydra**
* keep the **two-tier learner/actor split**
* close **advanced target generation + collation loops**
* activate only:

  * baseline dense heads
  * `oracle_critic` if dense enough
  * `safety_residual` next
  * `belief_fields` / `mixture_weight` only after calibration
* move **Hand-EV realism** before deeper AFBS expansion
* do **hard-state on-policy distillation** only

### Reserve shelf

* OLSS-style opponent-limited search
* robust-opponent search backups
* selective exactification
* side adapters for sparse heads only if gradient conflict forces it
* ReBeL/public-belief formulations as long-run theory, not near-term implementation

### Defer

* **`delta_q` activation**
* broader ExIt/ponder expansion until transfer gates pass

### Drop shelf

* **`opponent_hand_type`**
* new heads
* search-everywhere AFBS
* duplicated belief stack
* any teacher-only win claim

---

## 10. Narrowest safe long-run path to beating LuckyJ

**Target context.** Tencent’s published claim is that LuckyJ reached Tenhou 10 dan with a stable rank of **10.68** and did so from scratch in under 1500 matches. That is the bar Hydra is implicitly chasing. ([Haobo Fu @ Tencent][7])

**The narrowest safe path is not maximal Hydra.** It is:

1. **Dense baseline first**
   policy/value/score/defense heads reliable under 192x34 and current latency limits

2. **Value realism next**
   oracle/Hand-EV before broader search

3. **Hard-state search only**
   keep AFBS specialist and gated

4. **Actor-induced distillation only**
   distill on the states the actor actually creates

5. **Sparse heads only after density + transfer**
   safety, then belief/mixture, then maybe delta-Q

That path is the highest-probability way to close on a LuckyJ-class target without breaking the current artifact constraints, because it pushes exactly where prior successful Mahjong systems and imperfect-information search systems got leverage: **value realism, defensive reliability, selective online search, and careful compression of search into fast policies**—not by training every auxiliary target equally early. ([arXiv][4])

**Single most important deferral:** `delta_q` activation.
**Single most important immediate action:** close the existing supervision loops and instrument density/transfer so dormant complexity stops masquerading as trained capability.

[1]: https://papers.nips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf "https://papers.nips.cc/paper_files/paper/2017/file/d8e1344e27a5b08cdfd5d027d9b8d6de-Paper.pdf"
[2]: https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf "https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf"
[3]: https://proceedings.mlr.press/v80/lee18d/lee18d.pdf "https://proceedings.mlr.press/v80/lee18d/lee18d.pdf"
[4]: https://arxiv.org/pdf/2003.13590.pdf "https://arxiv.org/pdf/2003.13590.pdf"
[5]: https://proceedings.mlr.press/v202/liu23k/liu23k.pdf "Opponent-Limited Online Search for Imperfect Information Games"
[6]: https://proceedings.neurips.cc/paper_files/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf "https://proceedings.neurips.cc/paper_files/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf"
[7]: https://haobofu.github.io/ "Haobo Fu @ Tencent"
]]>
  </answer_text>
  </answer_section>
</combined_run_record>
