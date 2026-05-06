Read rest. Then patch exact code blocks only.Need exact diffs in code blocks. Loading compress skill, then compare blocks.Restoring exact fenced blocks from original.Verify blocks exact.<combined_run_record run_id="answer_20" variant_id="prompt_and_agent_pair" schema_version="1">
<metadata>
<notes>Combined record for Prompt 20 + returned agent answer.</notes>
<layout>single_markdown_file_prompt_then_answer</layout>
</metadata>

<prompt_section>
<prompt_text status="preserved" source_path="PROMPT_20_IMPLEMENT_ROLLOUT_PROVENANCE_AND_DISABLE_POLICY.md">
<![CDATA[# Prompt 20 — rollout provenance and disable-policy blueprint

<role>
Produce impl-ready blueprint.
Do not give memo.
Do not tell us write separate spec.
Answer itself must be blueprint.
</role>

<direction>
Work toward strongest blueprint for rollout provenance, cache provenance, trust boundaries, current safe rollout policy.

Want answer tell us:
- what current trust problem is
- what provenance fields missing
- what cache rules should be
- what should remain learner-only
- what safe current policy makes sense
- what future evidence needed before narrow rollout re-entry

Use artifacts below to derive conclusions.
</direction>

<style>
- no high-level survey
- no vague answer
- include reasoning
- include formulas when needed
- include code-like detail when helpful (python or rust)
- include worked examples when helpful
- include enough detail that we can validate ourselves (pdfs, sources, links, similar projects)
- distinguish direct artifact support from own inference
- use search/browse to find original paper, then inspect full PDF with skill; use abstracts or summaries only for discovery, not final evidence base
- use bash tool to run Python for calculations, math checks, validation when rigor matters
- do not finish early; keep looping through discovery, thinking, testing, validation until info saturated or blocked, and do not stop before at least 20+ such loops (as much loops as possible tho)
</style>

<artifact_note>
Artifacts below reflect what current codebase/docs seem say now. Not guaranteed fully correct. Treat as evidence to inspect + critique, not truth to inherit. High chance some incomplete, misleading, stale, semantically wrong, so validate all.
</artifact_note>

<artifacts>
Embedded artifact/context dump intentionally omitted during normalization. Use paired answer body + repository source files for audit and follow-up.
</artifacts>

]]>
</prompt_text>
</prompt_section>

<answer_section>
<answer_text status="preserved" source_path="agent_20.md">
<![CDATA[## 0. Set these defaults now

```rust
//! Anytime Factored-Belief Search (AFBS) with PUCT selection.

use crate::action::HYDRA_ACTION_SPACE;
use dashmap::DashMap;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::time::Instant;

pub const C_PUCT: f32 = 2.5;
pub const TOP_K: usize = 5;

fn masked_action_priors(
    policy_logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> Vec<(u8, f32)> {
    let legal_actions: Vec<(u8, f32)> = (0..HYDRA_ACTION_SPACE as u8)
        .filter(|&a| legal_mask[a as usize])
        .map(|a| (a, policy_logits[a as usize]))
        .collect();
    if legal_actions.is_empty() {
        return Vec::new();
    }
    let max_logit = legal_actions
        .iter()
        .map(|(_, logit)| *logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = legal_actions
        .iter()
        .map(|(_, logit)| (*logit - max_logit).exp())
        .sum();
    if exp_sum <= 0.0 || !exp_sum.is_finite() {
        let uniform = 1.0 / legal_actions.len() as f32;
        return legal_actions
            .into_iter()
            .map(|(action, _)| (action, uniform))
            .collect();
    }
    legal_actions
        .into_iter()
        .map(|(action, logit)| (action, (logit - max_logit).exp() / exp_sum))
        .collect()
}

pub type NodeIdx = u32;
type ChildList = SmallVec<[(u8, NodeIdx); TOP_K]>;

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

pub struct AfbsTree {
    pub nodes: Vec<AfbsNode>,
}

pub fn predicted_child_hash(parent_hash: u64, action: u8) -> u64 {
    parent_hash ^ (action as u64).wrapping_mul(0x9e3779b97f4a7c15)
}

impl AfbsTree {
    pub fn root_exit_policy(&self, root_idx: NodeIdx, tau: f32) -> [f32; HYDRA_ACTION_SPACE] {
        let mut policy = [0.0f32; HYDRA_ACTION_SPACE];
        let Some(root) = self.nodes.get(root_idx as usize) else {
            return policy;
        };
        if root.children.is_empty() {
            return policy;
        }

        if !tau.is_finite() || tau <= 0.0 {
            if let Some((action, _)) = root.children.iter().max_by(|(_, lhs), (_, rhs)| {
                self.nodes[*lhs as usize]
                    .q_value()
                    .partial_cmp(&self.nodes[*rhs as usize].q_value())
                    .unwrap_or(Ordering::Equal)
            }) {
                policy[*action as usize] = 1.0;
            }
            return policy;
        }

        let mut max_q = f32::NEG_INFINITY;
        for &(_, child_idx) in &root.children {
            let q = self.nodes[child_idx as usize].q_value();
            if q > max_q {
                max_q = q;
            }
        }
        let mut total = 0.0f32;
        for &(action, child_idx) in &root.children {
            let q = self.nodes[child_idx as usize].q_value();
            let exp_q = ((q - max_q) / tau).exp();
            policy[action as usize] = exp_q;
            total += exp_q;
        }
        if total > 0.0 {
            for p in &mut policy {
                *p /= total;
            }
        }
        policy
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

impl PonderResult {
    pub fn from_tree(tree: &AfbsTree, root_idx: NodeIdx, value: f32, tau: f32) -> Self {
        Self {
            exit_policy: tree.root_exit_policy(root_idx, tau),
            value,
            search_depth: tree.max_depth(root_idx),
            visit_count: tree.root_visit_count(root_idx),
            timestamp: Instant::now(),
        }
    }
}

pub struct PonderTask {
    pub info_state_hash: u64,
    pub priority_score: f32,
    pub game_state_snapshot: GameStateSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq)]
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

pub struct PonderCache {
    entries: DashMap<u64, PonderResult>,
}

impl PonderCache {
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
        }
    }

    pub fn get(&self, hash: u64) -> Option<PonderResult> {
        self.entries.get(&hash).map(|entry| *entry.value())
    }

    pub fn insert(&self, hash: u64, result: PonderResult) {
        self.entries.insert(hash, result);
    }

    pub fn predicted_child_key(parent_hash: u64, action: u8) -> u64 {
        predicted_child_hash(parent_hash, action)
    }

    pub fn get_predicted_child(&self, parent_hash: u64, action: u8) -> Option<PonderResult> {
        self.get(Self::predicted_child_key(parent_hash, action))
    }

    pub fn insert_predicted_child(&self, parent_hash: u64, action: u8, result: PonderResult) {
        self.insert(Self::predicted_child_key(parent_hash, action), result);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn remove(&self, hash: u64) -> Option<PonderResult> {
        self.entries.remove(&hash).map(|(_, value)| value)
    }

    pub fn contains(&self, hash: u64) -> bool {
        self.entries.contains_key(&hash)
    }

    pub fn clear(&self) {
        self.entries.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

pub struct PonderManager {
    pub cache: DashMap<u64, PonderResult>,
    pub priority_queue: std::collections::BinaryHeap<PonderTask>,
    pub worker_handle: Option<std::thread::JoinHandle<()>>,
}

impl PonderManager {
    pub fn new() -> Self {
        Self {
            cache: DashMap::new(),
            priority_queue: std::collections::BinaryHeap::new(),
            worker_handle: None,
        }
    }

    pub fn enqueue_snapshot(&mut self, snapshot: GameStateSnapshot) {
        let priority_score = compute_ponder_priority(
            snapshot.top2_policy_gap,
            snapshot.risk_score,
            snapshot.particle_ess,
        );
        self.priority_queue.push(PonderTask {
            info_state_hash: snapshot.info_state_hash,
            priority_score,
            game_state_snapshot: snapshot,
        });
    }

    pub fn cache_result(&self, hash: u64, result: PonderResult) {
        self.cache.insert(hash, result);
    }

    pub fn lookup(&self, hash: u64) -> Option<PonderResult> {
        self.cache.get(&hash).map(|entry| *entry.value())
    }
}
```

Artifact B — live inference authority path + tests:

```rust
//! Inference server: fast path (network + SaF) and slow path (pondered AFBS).

use burn::prelude::*;
use burn::tensor::activation;
use dashmap::DashMap;
use hydra_core::action::{AGARI, HYDRA_ACTION_SPACE};
use hydra_core::afbs::PonderResult;
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES};
use std::sync::Arc;

use crate::model::ActorNet;
use crate::saf::{SafConfig, SafMlp, apply_saf_logit, saf_tensor_from_observation};

pub const OBS_FLAT_SIZE: usize = NUM_CHANNELS * NUM_TILES;

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

pub struct InferenceServer<B: Backend> {
    pub actor: ActorNet<B>,
    pub ponder_cache: Arc<DashMap<u64, PonderResult>>,
    pub saf_mlp: SafMlp<B>,
    pub config: InferenceConfig,
    saf_alpha: f32,
    device: B::Device,
}

impl<B: Backend> InferenceServer<B> {
    pub fn info_state_hash(obs: &[f32; OBS_FLAT_SIZE]) -> u64 {
        obs.iter().fold(0xcbf29ce484222325, |hash, value| {
            hash.wrapping_mul(0x100000001b3) ^ value.to_bits() as u64
        })
    }

    pub fn cache_ponder_result(&self, info_state_hash: u64, result: PonderResult) {
        self.ponder_cache.insert(info_state_hash, result);
    }

    pub fn lookup_ponder(&self, info_state_hash: u64) -> Option<PonderResult> {
        self.ponder_cache
            .get(&info_state_hash)
            .map(|entry| *entry.value())
    }

    fn infer_with_budget(
        &self,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
        budget_ms: u64,
    ) -> (u8, [f32; HYDRA_ACTION_SPACE], bool) {
        let start = std::time::Instant::now();
        let info_state_hash = Self::info_state_hash(obs);

        if let Some(pondered) = self.lookup_ponder(info_state_hash) {
            let policy = mask_policy_cpu(&pondered.exit_policy, legal);
            let action = self.guard_action(argmax_legal(&policy, legal), legal);
            let within = start.elapsed().as_millis() as u64 <= budget_ms;
            return (action, policy, within);
        }

        let input = Tensor::<B, 1>::from_floats(obs.as_slice(), &self.device).reshape([
            1,
            NUM_CHANNELS,
            NUM_TILES,
        ]);
        let base_logits = self.actor.policy_logits_for(input);
        let logits = self.apply_saf_fast_path(base_logits, obs, legal);
        let (action, policy, within) = infer_action_timed(logits, legal, budget_ms);
        (self.guard_action(action, legal), policy, within)
    }

    fn apply_saf_fast_path(
        &self,
        base_logits: Tensor<B, 2>,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
    ) -> Tensor<B, 2> {
        let saf_features = saf_tensor_from_observation::<B>(obs.as_slice(), &self.device);
        let saf_delta = self
            .saf_mlp
            .forward(saf_features)
            .reshape([1, HYDRA_ACTION_SPACE]);
        let mask_tensor = legal_mask_to_tensor(legal, &self.device);
        apply_saf_logit(base_logits, saf_delta, mask_tensor, self.saf_alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn make_server(device: &<B as Backend>::Device) -> InferenceServer<B> {
        let actor = crate::model::HydraModelConfig::actor().init::<B>(device);
        InferenceServer::from_configs(
            actor,
            &SafConfig::new(),
            InferenceConfig::default(),
            *device,
        )
    }

    #[test]
    fn inference_server_respects_time_budget() {
        let device = Default::default();
        let mut server = make_server(&device);
        server.config.on_turn_budget_ms = 5_000;
        let obs = [0.0f32; OBS_FLAT_SIZE];
        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[0] = true;
        legal[1] = true;
        let (action, policy, within) = server.infer_timed(&obs, &legal);
        assert!(legal[action as usize]);
        assert!(within);
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn inference_server_reuses_cached_ponder_policy() {
        let device = Default::default();
        let server = make_server(&device);
        let obs = [0.0f32; OBS_FLAT_SIZE];
        let hash = InferenceServer::<B>::info_state_hash(&obs);
        let mut exit_policy = [0.0f32; HYDRA_ACTION_SPACE];
        exit_policy[5] = 0.9;
        exit_policy[6] = 0.1;
        server.cache_ponder_result(
            hash,
            PonderResult {
                exit_policy,
                value: 0.3,
                search_depth: 5,
                visit_count: 64,
                timestamp: std::time::Instant::now(),
            },
        );
        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[5] = true;
        legal[6] = true;
        let (action, policy) = server.infer(&obs, &legal);
        assert_eq!(action, 5);
        assert!(policy[5] > policy[6]);
    }

    #[test]
    fn inference_server_uses_call_reaction_budget() {
        let device = Default::default();
        let mut server = make_server(&device);
        server.config.call_reaction_budget_ms = 5_000;
        let obs = [0.0f32; OBS_FLAT_SIZE];
        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[3] = true;
        legal[4] = true;
        let (action, policy, within) = server.infer_call_reaction_timed(&obs, &legal);
        assert!(legal[action as usize]);
        assert!(within);
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }
}
```

Artifact C — model roles + benchmark gates:

```rust
pub type ActorNet<B> = HydraModel<B>;
pub type LearnerNet<B> = HydraModel<B>;

impl HydraModelConfig {
    pub fn is_actor(&self) -> bool {
        self.num_blocks == 12
    }
    pub fn is_learner(&self) -> bool {
        self.num_blocks == 24
    }

    pub fn actor() -> Self {
        Self::new(12).with_input_channels(INPUT_CHANNELS)
    }

    pub fn learner() -> Self {
        Self::new(24).with_input_channels(INPUT_CHANNELS)
    }
}
```

```rust
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

Artifact D — doctrine excerpts:

```text
If a mechanism raises ceiling but is too slow at inference, it belongs in pondering, deep search, offline solvers, or distillation targets -- not in the critical inference loop.
```

```text
Every “guarantee-like” claim must be either a theorem, a bound with explicit constants, or an empirical gate with a measurable pass/fail threshold.
```

```text
Fast path: Network forward + SaF adaptor. Slow path: Reuse pondered AFBS subtree. On-turn: 80-150ms. Call reactions: 20-50ms. Pondering: use all idle time.
```

```text
RolloutNet (ActorNet-sized, 12 blocks): policy + value for fast AFBS rollouts. Distilled from LearnerNet continuously.
```

```text
Hard positions only: top-2 policy gap < 10%, high-risk defense, low particle ESS.
```

Artifact E — extra AFBS cache/search tests:

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
fn predictive_child_key_matches_tree_expansion_hash() {
    let parent_hash = 12345;
    let action = 7;
    assert_eq!(
        PonderCache::predicted_child_key(parent_hash, action),
        predicted_child_hash(parent_hash, action)
    );
}

#[test]
fn predictive_ponder_cache_roundtrip() {
    let cache = PonderCache::new();
    let parent_hash = 777;
    let action = 11;
    let result = PonderResult {
        exit_policy: [0.0; HYDRA_ACTION_SPACE],
        value: 0.25,
        search_depth: 6,
        visit_count: 48,
        timestamp: Instant::now(),
    };
    cache.insert_predicted_child(parent_hash, action, result);
    let hit = cache
        .get_predicted_child(parent_hash, action)
        .expect("predicted child cache hit");
    assert_eq!(hit.visit_count, 48);
}

#[test]
fn ponder_result_from_tree_reflects_root_stats() {
    let result = PonderResult::from_tree(&tree, root, 0.42, 1.0);
    assert_eq!(result.visit_count, 9);
    assert_eq!(result.search_depth, 1);
    let sum: f32 = result.exit_policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}
```

Artifact F — benchmark/eval + model role context:

```rust
pub struct EvalResult {
    pub mean_placement: f32,
    pub stable_dan: f32,
    pub win_rate: f32,
    pub deal_in_rate: f32,
    pub tsumo_rate: f32,
}

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
```

```rust
pub type ActorNet<B> = HydraModel<B>;
pub type LearnerNet<B> = HydraModel<B>;

impl HydraModelConfig {
    pub fn summary(&self) -> String {
        let kind = if self.num_blocks <= 12 { "actor" } else { "learner" };
        format!("{}(blocks={}, ch={})", kind, self.num_blocks, self.hidden_channels)
    }
}
```

Artifact G — extra doctrine + limits:

```text
Main consensus:
- Do not let novelty outrank strength-per-effort.
- AFBS should be selective and specialist, not the default path everywhere.
```

```text
Current repo reality:
- advanced losses exist, but default advanced loss weights are zero
- AFBS exists as a search shell, but not as a fully integrated public-belief search runtime
```

```text
If gates fail, shrink AFBS/teacher usage and reallocate to more self-play.
```

```text
Limitations:
1. 4-player general-sum has no clean exploitability target.
2. Belief model misspecification remains the core risk.
3. Deep AFBS is expensive and depends on caching, pondering hit rate, and distillation efficiency.
4. Strategy fusion / determinization pitfalls are mitigated but not eliminated.
```

Artifact H — benchmark hooks:

```rust
use criterion::{Criterion, criterion_group, criterion_main};
use hydra_core::ct_smc::{CtSmc, CtSmcConfig};

fn bench_ct_smc_full_pipeline(c: &mut Criterion) {
    c.bench_function("ct_smc_dp_128_samples", |b| {
        b.iter(|| {
            let cfg = CtSmcConfig {
                num_particles: 128,
                ess_threshold: 0.4,
                rng_seed: 42,
            };
            let mut smc = CtSmc::new(cfg);
            smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
        });
    });
}
```

```rust
fn bench_encoder(c: &mut Criterion) {
    c.bench_function("encode_observation", |b| {
        b.iter(|| {
            bridge::encode_observation(&mut encoder, &obs, &safety, None);
            black_box(encoder.as_slice());
        });
    });
}
```

Artifact I — more inference/runtime helper surface:

```rust
pub fn legal_mask_to_tensor<B: Backend>(
    mask: &[bool; HYDRA_ACTION_SPACE],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut f32_mask = [0.0f32; HYDRA_ACTION_SPACE];
    for (i, &m) in mask.iter().enumerate() {
        f32_mask[i] = if m { 1.0 } else { 0.0 };
    }
    Tensor::<B, 1>::from_floats(&f32_mask[..], device).unsqueeze_dim::<2>(0)
}

pub fn normalize_policy_cpu(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut adjusted = [f32::NEG_INFINITY; HYDRA_ACTION_SPACE];
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            adjusted[i] = logits[i];
            if logits[i] > max_val {
                max_val = logits[i];
            }
        }
    }
    let mut probs = [0.0f32; HYDRA_ACTION_SPACE];
    let mut total = 0.0f32;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            probs[i] = (adjusted[i] - max_val).exp();
            total += probs[i];
        }
    }
    if total > 0.0 {
        for p in &mut probs {
            *p /= total;
        }
    }
    probs
}

pub fn mask_policy_cpu(
    policy: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> [f32; HYDRA_ACTION_SPACE] {
    let mut masked = [0.0f32; HYDRA_ACTION_SPACE];
    let mut total = 0.0f32;
    for i in 0..HYDRA_ACTION_SPACE {
        if legal_mask[i] {
            masked[i] = policy[i].max(0.0);
            total += masked[i];
        }
    }
    if total > 0.0 {
        for value in &mut masked {
            *value /= total;
        }
        return masked;
    }
    masked
}

pub fn policy_top2_gap(probs: &[f32; HYDRA_ACTION_SPACE]) -> f32 {
    let mut first = 0.0f32;
    let mut second = 0.0f32;
    for &p in probs {
        if p > first {
            second = first;
            first = p;
        } else if p > second {
            second = p;
        }
    }
    first - second
}
```

Artifact J — additional AFBS + cache tests:

```rust
#[test]
fn puct_balances_exploration_exploitation() {
    let mut tree = AfbsTree::new();
    let root = tree.add_node(0, 1.0, false);
    let c0 = tree.add_node(1, 0.5, false);
    let c1 = tree.add_node(2, 0.5, false);
    tree.nodes[root as usize].children = smallvec::smallvec![(0, c0), (1, c1)];
    tree.nodes[root as usize].visit_count = 10;
    tree.nodes[c0 as usize].visit_count = 8;
    tree.nodes[c0 as usize].total_value = 4.0;
    tree.nodes[c1 as usize].visit_count = 2;
    tree.nodes[c1 as usize].total_value = 1.5;
    let (action, _) = tree.puct_select(root).expect("select");
    assert_eq!(action, 1);
}

#[test]
fn exit_policy_sums_to_one() {
    let policy = tree.root_exit_policy(root, 1.0);
    let sum: f32 = policy.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[test]
fn exit_policy_with_zero_tau_becomes_argmax() {
    let policy = tree.root_exit_policy(root, 0.0);
    assert_eq!(policy[7], 1.0);
}
```

Artifact K — more doctrine + planning excerpts:

```text
Working principle:
- active path = what the team should optimize for now
- reserve shelf = good ideas kept for later if the active path underdelivers
- drop shelf = ideas that should stop consuming mainline attention for now
```

```text
What is only partially true:
- AFBS exists as a shell, but not as a fully integrated public-belief search runtime
- advanced supervision hooks exist, but the normal batch collation path still mostly emits baseline targets
```

```text
Decision:
- AFBS should be specialist / hard-state gated, not broad default runtime
```

```text
Validation gates:
G0 positive decision improvement, G1 robustness calibration, G2 safety bound usefulness, G3 SaF amortization.
```

Artifact L — more benchmark/eval snippets:

```rust
impl EvalResult {
    pub fn summary(&self) -> String {
        format!(
            "placement={:.2} dan={:.1} win={:.1}% deal_in={:.1}%",
            self.mean_placement,
            self.stable_dan,
            self.win_rate * 100.0,
            self.deal_in_rate * 100.0
        )
    }
}

impl TrainingMetrics {
    pub fn summary(&self) -> String {
        format!(
            "epoch={} loss={:.4} agree={:.2}% games={} elo={:.0}",
            self.epoch,
            self.total_loss,
            self.policy_agreement * 100.0,
            self.games_completed,
            self.elo
        )
    }
}
```

Artifact M — more runtime helper + test excerpts:

```rust
pub fn legal_mask_to_tensor<B: Backend>(
    mask: &[bool; HYDRA_ACTION_SPACE],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut f32_mask = [0.0f32; HYDRA_ACTION_SPACE];
    for (i, &m) in mask.iter().enumerate() {
        f32_mask[i] = if m { 1.0 } else { 0.0 };
    }
    Tensor::<B, 1>::from_floats(&f32_mask[..], device).unsqueeze_dim::<2>(0)
}

pub fn policy_top2_gap(probs: &[f32; HYDRA_ACTION_SPACE]) -> f32 {
    let mut first = 0.0f32;
    let mut second = 0.0f32;
    for &p in probs {
        if p > first {
            second = first;
            first = p;
        } else if p > second {
            second = p;
        }
    }
    first - second
}

#[test]
fn inference_server_reuses_cached_ponder_policy() {
    let hash = InferenceServer::<B>::info_state_hash(&obs);
    let mut exit_policy = [0.0f32; HYDRA_ACTION_SPACE];
    exit_policy[5] = 0.9;
    exit_policy[6] = 0.1;
    server.cache_ponder_result(
        hash,
        PonderResult {
            exit_policy,
            value: 0.3,
            search_depth: 5,
            visit_count: 64,
            timestamp: std::time::Instant::now(),
        },
    );
    let (action, policy) = server.infer(&obs, &legal);
    assert_eq!(action, 5);
    assert!(policy[5] > policy[6]);
}
```

Artifact N — more doctrine + deployment excerpts:

```text
Fast path: network forward + SaF adaptor. Slow path: reuse pondered AFBS subtree. On-turn: 80-150ms. Call reactions: 20-50ms. Pondering: use all idle time.
```

```text
If a mechanism raises ceiling but is too slow at inference, it belongs in pondering, deep search, offline solvers, or distillation targets -- not in the critical inference loop.
```

```text
Any guarantee-like rollout admission must be a measurable empirical gate, not a vibe.
```

```text
What is only partially true:
- AFBS exists as a shell, but not as a fully integrated public-belief search runtime
- advanced supervision hooks exist, but the normal batch collation path still mostly emits baseline targets
```

Artifact O — more benchmark + limitation excerpts:

```text
Hard reality benchmarks unlock before committing the full budget:
- latency gate
- throughput gate
- distillation gate
- hyperparameter sweep
If gates fail, shrink AFBS/teacher usage and reallocate to more self-play.
```

```text
Limitations:
1. 4-player general-sum has no clean exploitability target.
2. Belief model misspecification remains the core risk.
3. Deep AFBS is expensive.
4. Strategy fusion / determinization pitfalls are mitigated but not eliminated.
```

Artifact P — additional runtime, evaluation, + planning excerpts:

```text
Fast path vs ponder hooks already exist; broad default expensive search is not the intended current runtime identity.
```

```text
Current repo reality:
- actor and learner roles are explicit
- cached ponder results are consumed directly on hit
- benchmark gates are coarse and global rather than provenance-aware
```

```text
Keep AFBS specialist and hard-state gated.
```

```text
Do not let novelty outrank strength-per-effort.
```

```rust
pub fn compute_stable_dan(mean_placement: f32) -> f32 {
    (10.0 - (mean_placement - 1.0) * 4.0).clamp(0.0, 12.0)
}

pub fn compute_mean_placement(placements: &[u8]) -> f32 {
    if placements.is_empty() {
        return 2.5;
    }
    placements.iter().map(|&p| p as f32 + 1.0).sum::<f32>() / placements.len() as f32
}

pub fn compute_win_rate(placements: &[u8]) -> f32 {
    if placements.is_empty() {
        return 0.0;
    }
    placements.iter().filter(|&&p| p == 0).count() as f32 / placements.len() as f32
}
```


```rust
// production defaults
const ENABLE_RUNTIME_PONDER_AUTHORITY: bool = false;
const ENABLE_RUNTIME_PONDER_ADVISORY: bool = false;
const ENABLE_SPECULATIVE_CHILD_RUNTIME: bool = false;
const ENABLE_SAME_EPISODE_WARMSTART: bool = false; // until subtree snapshots exist
const ENABLE_PONDER_FOR_LEARNER_TARGETS: bool = true;
const ENABLE_HARD_STATE_IDLE_SEARCH: bool = true;
```

That right current policy.

Reason simple: current impl does **not** safely reuse pondered AFBS subtree. It replays summarized root policy/value from cache entry keyed by bare `u64` observation hash, or worse, by speculative `predicted_child_hash(parent, action)`. In imperfect-information search, that below trust threshold for live action authority. ReBeL, Student of Games, POMCP anchor search/value reuse on richer semantics—public-belief states or history-belief nodes—not flat observation hash divorced from policy/belief provenance. Safe subgame solving + depth-limited solving papers make same point from another angle: values in imperfect-information games not well-defined from local state alone, and subgames cannot be trusted in isolation without right surrounding context. ([NeurIPS Proceedings][1])

---

## 1. The current trust problem is **authority without semantic identity**

### Direct artifact support

From Artifacts A/B/E/M/P:

* `InferenceServer::infer_with_budget` does early return on any `lookup_ponder(info_state_hash)` hit.
* `PonderResult` stores only:

  * `exit_policy`
  * `value`
  * `search_depth`
  * `visit_count`
  * `timestamp`
* `PonderCache` + `PonderManager` use `DashMap<u64, PonderResult>`.
* `predicted_child_hash(parent_hash, action)` used as another cache key in same key space.
* `GameStateSnapshot` carries only `{info_state_hash, top2_policy_gap, risk_score, particle_ess}`.
* current benchmark gates coarse/global, not provenance-aware.
* doctrine says “slow path: reuse pondered AFBS subtree,” but stored object only root summary, not subtree.

Another artifact-level issue: `ChildList` capped at `TOP_K = 5`, and `root_exit_policy()` only assigns mass over `root.children`. Unless legal branching `<= 5` at runtime, cached exit policy is **partial policy summary**, not complete root policy.

### Inference

Main bug **not** “64-bit hashes might collide.”

Main bug = system treats:

```text
same observation hash
```

as if it implied:

```text
same semantic root for imperfect-information search
same public/belief context
same assumed policy bundle
same legal action set
same solver/value semantics
same freshness/generation
same admissible trust class
```

It does not.

In imperfect-information games, decision point value depends on belief context + continuation policies, not only local observation encoding. ReBeL explicit that naive RL+search not theoretically sound in imperfect-information games, and introduces public belief state (PBS) precisely because “state” must include common-knowledge belief distribution induced by public observations **and policies of all agents**. Student of Games makes same move with `β = (s_pub, r)` and requires search consistency along trajectory for sound self-play. POMCP likewise uses history nodes + belief updates, not flat state hashes. Safe/Nested Subgame Solving and OOS show why: non-locality + strategy fusion break assumptions that make local subgame reuse safe in perfect-information settings. ([NeurIPS Proceedings][1])

Quick birthday-bound check confirms raw 64-bit hash collision secondary. Using

[
p_{\text{coll}} \approx 1 - \exp\left(-\frac{n(n-1)}{2\cdot 2^{64}}\right),
]

collision probability about (2.7\times 10^{-8}) at (n=10^6) entries and (2.7\times 10^{-4}) at (n=10^8). Move to 128/256-bit digests anyway, but today’s problem semantic aliasing + speculative reuse, not birthday collisions.

Also “too weak / too brittle” problem. Current `info_state_hash(obs)` too weak because omits model/solver/lineage provenance, and too brittle because hashes raw float bits, so numerically equivalent states can miss due to tiny representation drift.

---

## 2. The trust boundaries that must exist

### Boundary A — observation proxy vs. search root semantics

Current code has `obs -> u64` identity. That hint, not trusted root identity.

Trusted root identity in this domain must be one of:

* `HistoryBelief`: actual observed public history + current belief/range digest
* `PublicBelief`: explicit PBS-style root
* never `ObservationProxy` for authority

ReBeL + SoG both make public-belief context relevant “state.” POMCP uses history nodes for same reason in POMDP planning. ([NeurIPS Proceedings][1])

### Boundary B — belief state vs. policy assumptions

PBS determined by public observations **and policies of all agents**. That means cache hit not trustworthy unless policy bundle used to build/update beliefs matches. If actor, rollout net, SaF adaptor, or opponent-policy assumptions changed, belief context changed. ([NeurIPS Proceedings][1])

### Boundary C — producer vs. consumer

Background ponder worker different trust domain from live action selection. Crossing boundary requires explicit provenance + admission logic. Raw `DashMap<u64, PonderResult>` access inside `InferenceServer` wrong abstraction because makes bypasses trivial.

### Boundary D — observed successor vs. speculative successor

`predicted_child_hash(parent, action)` crosses even harder boundary: it uses **hypothesized** successor identity before environment confirms actual successor observation/belief update. In stochastic, multi-agent, imperfect-information game, that only hint. Must never be authoritative. OOS strategy-fusion/non-locality warnings exact about this mistake family: cannot pretend hidden distinctions + off-path effects do not matter. ([IFAAMAS][2])

### Boundary E — learner/teacher artifacts vs. actor authority

Artifacts C/D/F/P make actor + learner roles explicit, and doctrine says rollout/teacher mechanisms belong in pondering, deep search, offline solvers, or distillation if too slow for critical path. So teacher outputs + rollout summaries should default learner-only unless separate online admission gate says otherwise. That also how SoG structured: search outputs become query-buffer/replay targets and are solved/trained asynchronously. ([PMC][3])

### Boundary F — global benchmarks vs. local admission

Current `BenchmarkGates` necessary capacity gates. Not local trust proofs. Global pass on latency, throughput, KL drift does not imply particular cache hit semantically valid.

---

## 3. The provenance fields that are missing

Split into **key fields** + **metadata fields**.

### 3.1 Required key fields

```rust
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceKey {
    pub schema_version: u16,

    // namespace prevents observed-root / speculative-child / learner-target mixing
    pub namespace: CacheNamespace, // ObservedRoot | SpeculativeChildHint | LearnerTarget

    // lifecycle / scope
    pub episode_id: u64,   // current safe policy: same-episode only for any future runtime reuse
    pub ply: u16,
    pub actor_seat: u8,

    // game/rules identity
    pub ruleset_digest: [u8; 16],
    pub action_space_version: u16,

    // root semantics
    pub root_kind: RootKind, // ObservationProxy | HistoryBelief | PublicBelief
    pub public_state_digest: [u8; 32],
    pub belief_digest: [u8; 32],
    pub legal_digest: [u8; 16],

    // policy / model assumptions
    pub policy_assumption_digest: [u8; 32], // policies used in belief update / common-knowledge model
    pub model_bundle_digest: [u8; 32],      // actor + saf + rollout + encoder + belief-updater bundle
    pub solver_bundle_digest: [u8; 32],     // AFBS impl + constants + rollout semantics + backup semantics

    // semantics
    pub value_semantics: ValueSemantics,
    pub utility_spec: UtilitySpec,

    // lineage
    pub lineage: Lineage, // Observed | PredictedChild
}
```

### 3.2 Required metadata fields

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PonderMeta {
    // debugging / hinting only
    pub obs_digest: [u8; 32],
    pub obs_encoder_version: u32,
    pub belief_builder_version: u32,

    // search-quality summary
    pub visit_count: u32,
    pub search_depth: u16,
    pub node_count: u32,
    pub leaf_eval_count: u32,
    pub search_budget_ms: u32,

    // root coverage: mandatory because TOP_K truncates children
    pub legal_action_count: u16,
    pub expanded_legal_count: u16,
    pub expanded_legal_fraction: f32,
    pub pruned_prior_mass: f32,

    // belief/risk diagnostics
    pub particle_count: u16,
    pub particle_ess: f32,
    pub belief_entropy: f32,
    pub top2_policy_gap: f32,
    pub risk_score: f32,

    // rollout / leaf semantics
    pub leaf_eval_kind: LeafEvalKind,     // Terminal | RolloutNet | ValueNet | CounterfactualSolver
    pub leaf_model_digest: [u8; 32],
    pub rollout_horizon: u16,
    pub backup_kind: BackupKind,          // mean return | counterfactual vector | risk-adjusted EV

    // freshness / invalidation
    pub created_generation: u64,
    pub created_wall_unix_ms: u64,
    pub created_mono_ns: u64,

    // admission result
    pub trust_level: TrustLevel,          // LearnerOnly | Advisory | WarmStart | Authoritative
    pub trust_reason: TrustReason,
}
```

### 3.3 Required payload changes

Current `PonderResult` insufficient for safe subtree reuse.

```rust
pub struct PonderEntry {
    pub key: ProvenanceKey,
    pub meta: PonderMeta,

    // payload
    pub exit_policy: [f32; HYDRA_ACTION_SPACE],
    pub value: f32, // learner/diagnostic only unless ValueSemantics makes it meaningful
    pub root_stats: SmallVec<[RootActionStat; 16]>,
    pub subtree: Option<CompressedAfbsTree>, // required for any future warm-start trust
}
```

### 3.4 Canonical belief digest

For particle-based beliefs, make digest order-invariant + numerically stable.

```rust
fn q16(p: f32) -> u16 {
    (p.clamp(0.0, 1.0) * 65535.0).round() as u16
}

fn belief_digest(particles: &[Particle], schema_version: u32) -> [u8; 32] {
    let mut rows: Vec<([u8; 32], u16)> = particles
        .iter()
        .map(|p| (latent_state_digest(&p.latent_state), q16(p.weight)))
        .collect();

    rows.sort_unstable();

    let mut h = blake3::Hasher::new();
    h.update(&schema_version.to_le_bytes());
    for (state_d, w_q16) in rows {
        h.update(&state_d);
        h.update(&w_q16.to_le_bytes());
    }
    *h.finalize().as_bytes()
}
```

`info_state_hash(obs)` can remain, but only as `obs_hint_digest`, never authority key.

---

## 4. Cache rules

### 4.1 Separate stores

Do not keep one raw map.

```rust
pub struct RuntimePonderCache {
    observed_roots: DashMap<ProvenanceKey, Arc<PonderEntry>>,
    speculative_hints: DashMap<ProvenanceKey, Arc<PonderEntry>>,
    learner_targets: DashMap<ProvenanceKey, Arc<PonderEntry>>,
    current_generation: AtomicU64,
}
```

Namespace separation matters. Observed root + predicted child must not share key space.

### 4.2 Insert rules

1. **ObservedRoot runtime store**

   * insert only after root observed
   * require full key + meta
   * require `root_kind in {HistoryBelief, PublicBelief}`
   * require `lineage == Observed`

2. **SpeculativeChildHint store**

   * may insert from `(parent, action)` prediction
   * must use `namespace = SpeculativeChildHint`
   * must expire on first mismatch or next environment transition
   * never used for final action selection

3. **LearnerTarget store**

   * may accept observed roots, speculative children, recursive queries, off-main-line solver outputs
   * must carry source tags + full provenance
   * never read by actor inference

### 4.3 Lookup / admission rules

Use hard gates, not fuzzy scalar trust score.

```rust
fn classify_runtime_use(e: &PonderEntry, q: &ProvenanceKey) -> TrustLevel {
    // current production rule
    if !ENABLE_RUNTIME_PONDER_AUTHORITY && !ENABLE_RUNTIME_PONDER_ADVISORY {
        return TrustLevel::LearnerOnly;
    }

    // absolute rejections
    if e.key.namespace != CacheNamespace::ObservedRoot { return TrustLevel::LearnerOnly; }
    if e.key.lineage != Lineage::Observed { return TrustLevel::LearnerOnly; }
    if matches!(e.key.root_kind, RootKind::ObservationProxy) { return TrustLevel::LearnerOnly; }

    if e.key.ruleset_digest != q.ruleset_digest { return TrustLevel::LearnerOnly; }
    if e.key.action_space_version != q.action_space_version { return TrustLevel::LearnerOnly; }
    if e.key.public_state_digest != q.public_state_digest { return TrustLevel::LearnerOnly; }
    if e.key.belief_digest != q.belief_digest { return TrustLevel::LearnerOnly; }
    if e.key.legal_digest != q.legal_digest { return TrustLevel::LearnerOnly; }

    if e.key.policy_assumption_digest != q.policy_assumption_digest { return TrustLevel::LearnerOnly; }
    if e.key.model_bundle_digest != q.model_bundle_digest { return TrustLevel::LearnerOnly; }
    if e.key.solver_bundle_digest != q.solver_bundle_digest { return TrustLevel::LearnerOnly; }
    if e.key.value_semantics != q.value_semantics { return TrustLevel::LearnerOnly; }
    if e.key.utility_spec != q.utility_spec { return TrustLevel::LearnerOnly; }

    // current safety scope: same episode only
    if e.key.episode_id != q.episode_id { return TrustLevel::LearnerOnly; }

    // low-ESS is a search-priority signal, not an authority signal
    if e.meta.particle_ess < 0.70 { return TrustLevel::LearnerOnly; }

    // root summary is incomplete if action coverage is incomplete
    if e.meta.expanded_legal_fraction < 1.0 { return TrustLevel::LearnerOnly; }
    if e.meta.pruned_prior_mass > 0.01 { return TrustLevel::LearnerOnly; }

    // no subtree, no warm-start
    if e.subtree.is_none() { return TrustLevel::Advisory; }

    // future only
    if e.meta.visit_count >= 128 && e.meta.search_depth >= 2 {
        return TrustLevel::WarmStart;
    }

    TrustLevel::Advisory
}
```

### 4.4 Runtime use rules

Current production:

* `LearnerOnly`: ignore in inference
* `Advisory`: disabled in production; optional research-only logit prior blending
* `WarmStart`: disabled until subtree snapshots exist
* `Authoritative`: disabled

Research-only advisory blend:

[
\ell'_a = \ell^{\text{live}}_a + \lambda \log(\pi^{\text{cache}}_a + \varepsilon),
\qquad 0 \le \lambda \le 0.15
]

Use only when provenance matches exactly except trust class, and log every decision.

### 4.5 Invalidation rules

Invalidate runtime stores on any of:

* actor checkpoint change
* SaF checkpoint change
* rollout net / teacher change
* encoder schema change
* belief-builder change
* AFBS config change (`C_PUCT`, `TOP_K`, temperature, backup kind, rollout semantics)
* ruleset/action-space change
* episode end

Global generation invalidation cheap:

```rust
fn rotate_generation(&self) {
    self.current_generation.fetch_add(1, Ordering::SeqCst);
}
```

### 4.6 Audit rules

Every lookup attempt must emit:

```rust
pub struct CacheDecisionAudit {
    pub query_key: ProvenanceKey,
    pub candidate_found: bool,
    pub admitted_trust: TrustLevel,
    pub reject_reason: Option<TrustReason>,
}
```

No silent fallback.

---

## 5. What should remain learner-only

Right now, **all current ponder/rollout outputs should remain learner-only for production runtime**.

That includes:

1. every current `PonderResult` in Artifacts A/B/E/M, because current entries are `ObservationProxy` roots with missing model/solver/legal/belief provenance and no subtree payload
2. every `predicted_child_hash` entry
3. every entry with `particle_ess < 0.70`
4. every cross-generation entry
5. every entry whose root policy truncated by `TOP_K` coverage
6. every scalar `value` from depth-limited or rollout-style search unless semantics explicitly versioned + empirically validated
7. every recursive/off-main-line search output not tied to validated runtime cohort

This not wasted work. Exactly how systems like Student of Games get value from expensive search: searches produce queries/targets keyed by semantically meaningful state `β`, those are solved more deeply, and results go to replay/training buffers. That right home for current AFBS shell + rollout outputs. ([PMC][3])

---

## 6. The safe current rollout policy

### Production runtime identity

* **On-turn**: `ActorNet + SaF` only
* **Call reactions**: `ActorNet + SaF` only
* **Pondering**: AFBS allowed on idle time, hard-state gated
* **Rollout nets / AFBS values**: search-internal or learner-target only
* **Cached ponder hits**: not on live authority path
* **Predicted child cache**: not on live authority path

This matches doctrine better than current code. Own text says:

* fast path is network + SaF
* expensive mechanisms belong in pondering/deep search/distillation when too slow
* AFBS should be specialist / hard-state gated, not default runtime
* current repo not fully integrated public-belief search runtime

Literature points same way. Strongest soundness results here tied to explicit public-belief / history-based search semantics, mostly in two-player zero-sum settings. ReBeL says that directly; its theoretical results in 2p0s, and it assumes common-knowledge policies in formal setup. Repo also explicitly admits “4-player general-sum has no clean exploitability target.” That means safe policy today must be empirical + conservative. ([NeurIPS Proceedings][1])

### Keep hard-state scheduler, but decouple from trust

Keep existing ponder-priority formula:

[
\text{priority} =
10\max(0, 0.1 - \text{top2_gap})

* \max(0,\text{risk_score})
* \max(0,1 - \text{particle_ess})
]

That scheduling rule.

It **not** admission rule.

Low ESS means “spend more idle compute here,” not “trust cached search here more.” In fact, by imperfect-information literature, low ESS reason to distrust replay authority, because optimal policy more sensitive to belief distribution. ([NeurIPS Papers][4])

---

## 7. Future evidence required before narrow rollout re-entry

Use artifact names `G0..G3`, but make local + measurable.

### G0 — positive decision improvement

Target cohort: hard states only.

Require paired comparison against `ActorNet + SaF` baseline on frozen evaluation suite + arena:

* `Δ mean_placement < 0` with 95% CI excluding 0
* `Δ deal_in_rate <= 0`
* no regression on stable-dan / win-rate summary metrics

Measure on exact cohort where runtime reuse would be allowed, not globally.

### G1 — robustness calibration

For **exact-match observed-root cohort**:

* top-1 agreement with fresh recomputed search: `>= 0.98`
* policy divergence to fresh recompute:
[
JS(\pi_{\text{cache}}, \pi_{\text{fresh}})_{p95} \le 0.05
]
* no admission when `ESS < 0.70`

For **belief perturbation test**:

* perturb belief within same quantization bucket
* verify low action-flip rate + bounded policy divergence

If policy highly unstable under tiny belief perturbations, cache class stays learner-only.

### G2 — safety-bound usefulness

Provenance gate must separate good hits from bad hits.

Require:

* `predicted_child_authority_hits == 0`
* `cross_generation_authority_hits == 0`
* `missing_provenance_authority_hits == 0`
* `low_ess_authority_hits == 0`

Also require “authoritative-eligible” cohort has materially lower disagreement/error than rejected cohort. If gate does not stratify risk, not useful.

### G3 — SaF amortization / strength-per-effort

Show expensive AFBS/rollout supervision buys enough signal to justify runtime complexity:

* current global gates still pass:

  * `afbs_on_turn_ms < 150`
  * `ct_smc_dp_ms < 1`
  * `endgame_ms < 100`
  * `self_play_games_per_sec > 20`
  * `distill_kl_drift < 0.1`
* plus provenance-aware gate:

  * on exact-match hard-state cohort, distilled actor behavior stays close enough that runtime cache use adds net value rather than papering over student/teacher mismatch

### Re-entry stages

**Stage 1 — allowed future re-entry**

* same-episode only
* `ObservedRoot` only
* `HistoryBelief` or `PublicBelief` roots only
* exact provenance match
* subtree warm-start only
* at least some fresh search work before action selection
* no direct replay of stale root summary

**Stage 2 — later**

* cross-episode reuse only after explicit PBS runtime exists and same gates pass on cross-episode exact-PBS cohorts

**Stage 3 — not planned goal**

* direct authoritative replay of cached `exit_policy` without continuing search

Would not make Stage 3 roadmap goal. Safer end state = exact-root warm-start + distillation, not “cache hit = answer.”

---

## 8. Immediate code and test changes

### 8.1 Remove unsafe live-authority path

Replace this pattern in `InferenceServer::infer_with_budget`:

```rust
if let Some(pondered) = self.lookup_ponder(info_state_hash) {
    ...
    return (action, policy, within);
}
```

with:

```rust
let query = ProvenanceKey::from_runtime_state(...);

// production: no runtime ponder authority
debug_assert!(!ENABLE_RUNTIME_PONDER_AUTHORITY);

let live = self.run_actor_saf_fast_path(obs, legal, budget_ms);

// optional research-only advisory path goes here, behind a flag
live
```

### 8.2 Rename `info_state_hash`

```rust
pub fn obs_hint_hash(obs: &[f32; OBS_FLAT_SIZE]) -> u64
```

Keep only for candidate retrieval/debugging.

### 8.3 Fix producer side

`PonderTask` must hold reconstructable root, not summary scalars.

```rust
pub struct PonderRootSnapshot {
    pub episode_id: u64,
    pub ply: u16,
    pub actor_seat: u8,
    pub obs: [f32; OBS_FLAT_SIZE],
    pub legal: [bool; HYDRA_ACTION_SPACE],
    pub public_state_bytes: Arc<[u8]>,
    pub belief_particles: Arc<[CanonicalParticle]>,
    pub policy_gap: f32,
    pub risk_score: f32,
    pub particle_ess: f32,
    pub model_bundle_digest: [u8; 32],
    pub solver_bundle_digest: [u8; 32],
}
```

Current `GameStateSnapshot` not enough.

### 8.4 Replace tests that encode wrong policy

Delete or rewrite:

* `inference_server_reuses_cached_ponder_policy`
* any test whose success condition “cache hit directly chooses action”

Add:

* `predicted_child_is_never_runtime_authoritative`
* `cross_generation_entry_is_rejected`
* `legal_digest_mismatch_is_rejected`
* `observation_proxy_root_is_learner_only`
* `low_ess_entry_is_learner_only`
* `topk_truncated_root_cannot_be_authoritative`
* `same_episode_exact_history_belief_with_subtree_can_warmstart` (future)
* `runtime_cache_rotates_on_model_bundle_change`

---

## 9. Worked examples

### Example A — hard state, not authority-worthy

Given:

* `top2_gap = 0.03`
* `risk_score = 0.80`
* `particle_ess = 0.22`

Current priority formula gives:

[
10(0.1 - 0.03) + 0.80 + (1 - 0.22)
= 0.70 + 0.80 + 0.78
= 2.28
]

That great candidate for idle-time pondering.

Bad candidate for runtime authority because `ESS = 0.22` means decision highly belief-sensitive.

### Example B — `TOP_K` truncation blocks authority

Suppose root has 9 legal actions but `ChildList` stores only 5 expanded children.

Then:

[
\text{expanded_legal_fraction} = 5/9 \approx 0.56
]

Even if cached policy sums to 1 over those 5 children, still only partial root summary. Authority forbidden. At most, learner target or advisory prior in research mode.

### Example C — predicted child is speculation, not identity

Current code can store:

```rust
predicted_child_key = parent_hash ^ f(action)
```

before observing successor.

That key contains no actual successor observation, no belief update, no legal mask, no model bundle, no solver bundle. It must live in `SpeculativeChildHint`, never in runtime authority path.

---

This blueprint:

* disable current runtime rollout/ponder authority
* keep AFBS/rollouts active for learner targets + diagnostics
* add explicit provenance keys + trust levels
* split observed runtime reuse from speculative hints + learner buffers
* require exact semantic identity before any future warm-start
* do not plan re-enter broad direct replay of cached policies

Until those gates pass, rollout output **learner signal**, not **live authority**.

[1]: https://proceedings.neurips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf "https://proceedings.neurips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf"
[2]: https://aamas.csc.liv.ac.uk/Proceedings/aamas2015/aamas/p27.pdf "https://aamas.csc.liv.ac.uk/Proceedings/aamas2015/aamas/p27.pdf"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10651118/ "https://pmc.ncbi.nlm.nih.gov/articles/PMC10651118/"
[4]: https://papers.neurips.cc/paper/7993-depth-limited-solving-for-imperfect-information-games.pdf "https://papers.neurips.cc/paper/7993-depth-limited-solving-for-imperfect-information-games.pdf"
]]>
</answer_text>
</answer_section>
</combined_run_record>