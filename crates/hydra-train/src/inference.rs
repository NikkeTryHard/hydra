//! Inference server: fast path (network + SaF) and slow path (pondered AFBS).

use burn::prelude::*;
use burn::tensor::activation;
use hydra_core::action::{AGARI, HYDRA_ACTION_SPACE};
use hydra_core::afbs::{PonderCache, PonderResult, TrustLevel};
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

impl InferenceConfig {
    pub fn summary(&self) -> String {
        format!(
            "infer(turn={}ms, call={}ms, guard={})",
            self.on_turn_budget_ms, self.call_reaction_budget_ms, self.agari_guard
        )
    }
}

pub struct InferenceServer<B: Backend> {
    pub actor: ActorNet<B>,
    pub ponder_cache: Arc<PonderCache>,
    pub saf_mlp: SafMlp<B>,
    pub config: InferenceConfig,
    saf_alpha: f32,
    device: B::Device,
}

impl<B: Backend> InferenceServer<B> {
    pub fn new(
        actor: ActorNet<B>,
        ponder_cache: Arc<PonderCache>,
        saf_mlp: SafMlp<B>,
        saf_alpha: f32,
        config: InferenceConfig,
        device: B::Device,
    ) -> Self {
        Self {
            actor,
            ponder_cache,
            saf_mlp,
            config,
            saf_alpha,
            device,
        }
    }

    pub fn from_configs(
        actor: ActorNet<B>,
        saf_config: &SafConfig,
        config: InferenceConfig,
        device: B::Device,
    ) -> Self {
        let saf_alpha = saf_config.alpha;
        let saf_mlp = saf_config.init(&device);
        Self::new(
            actor,
            Arc::new(PonderCache::new()),
            saf_mlp,
            saf_alpha,
            config,
            device,
        )
    }

    pub fn info_state_hash(obs: &[f32; OBS_FLAT_SIZE]) -> u64 {
        obs.iter().fold(0xcbf29ce484222325, |hash, value| {
            hash.wrapping_mul(0x100000001b3) ^ value.to_bits() as u64
        })
    }

    pub fn cache_ponder_result(&self, info_state_hash: u64, result: PonderResult) {
        self.ponder_cache.insert(info_state_hash, result);
    }

    /// Looks up a cached ponder result without trust-level filtering.
    ///
    /// Use `lookup_ponder_trusted` for runtime action selection.
    pub fn lookup_ponder(&self, info_state_hash: u64) -> Option<PonderResult> {
        self.ponder_cache.get(info_state_hash)
    }

    /// Looks up a cached result that meets the given minimum trust level.
    pub fn lookup_ponder_trusted(
        &self,
        info_state_hash: u64,
        min_trust: TrustLevel,
    ) -> Option<PonderResult> {
        self.ponder_cache.get_trusted(info_state_hash, min_trust)
    }

    /// Invalidates all cached entries (e.g. after a checkpoint change).
    pub fn invalidate_cache(&self) -> u64 {
        self.ponder_cache.invalidate()
    }

    pub fn infer(
        &self,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
    ) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
        let (action, policy, _) = self.infer_timed(obs, legal);
        (action, policy)
    }

    pub fn infer_call_reaction(
        &self,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
    ) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
        let (action, policy, _) = self.infer_call_reaction_timed(obs, legal);
        (action, policy)
    }

    pub fn infer_timed(
        &self,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
    ) -> (u8, [f32; HYDRA_ACTION_SPACE], bool) {
        self.infer_with_budget(obs, legal, self.config.on_turn_budget_ms)
    }

    pub fn infer_call_reaction_timed(
        &self,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
    ) -> (u8, [f32; HYDRA_ACTION_SPACE], bool) {
        self.infer_with_budget(obs, legal, self.config.call_reaction_budget_ms)
    }

    fn infer_with_budget(
        &self,
        obs: &[f32; OBS_FLAT_SIZE],
        legal: &[bool; HYDRA_ACTION_SPACE],
        budget_ms: u64,
    ) -> (u8, [f32; HYDRA_ACTION_SPACE], bool) {
        let start = std::time::Instant::now();
        let info_state_hash = Self::info_state_hash(obs);

        // Only use cache hits for runtime action selection when the result
        // has Authoritative trust.  Currently nothing qualifies, keeping
        // all ponder outputs learner-only per archive doctrine.
        if let Some(pondered) =
            self.lookup_ponder_trusted(info_state_hash, TrustLevel::Authoritative)
        {
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

    fn guard_action(&self, action: u8, legal: &[bool; HYDRA_ACTION_SPACE]) -> u8 {
        if self.config.agari_guard && action == AGARI && !legal[action as usize] {
            return argmax_legal(&mask_policy_cpu(&[0.0; HYDRA_ACTION_SPACE], legal), legal);
        }
        if legal[action as usize] {
            action
        } else {
            argmax_legal(&mask_policy_cpu(&[0.0; HYDRA_ACTION_SPACE], legal), legal)
        }
    }
}

/// Converts a boolean legal mask to a [1, 46] float tensor.
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

/// Computes softmax policy on CPU with legal masking and max subtraction.
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

    let legal_count = legal_mask.iter().filter(|&&m| m).count();
    if legal_count > 0 {
        let uniform = 1.0 / legal_count as f32;
        for (i, value) in masked.iter_mut().enumerate() {
            if legal_mask[i] {
                *value = uniform;
            }
        }
    }
    masked
}

pub fn validate_legal_mask(mask: &[bool; HYDRA_ACTION_SPACE]) -> bool {
    mask.iter().any(|&m| m)
}

pub fn policy_entropy(probs: &[f32; HYDRA_ACTION_SPACE]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 1e-8 {
            h -= p * p.ln();
        }
    }
    h
}

pub fn action_rank(probs: &[f32; HYDRA_ACTION_SPACE], action: u8) -> usize {
    let p = probs[action as usize];
    probs.iter().filter(|&&q| q > p).count()
}

pub fn needs_search(probs: &[f32; HYDRA_ACTION_SPACE], gap_threshold: f32) -> bool {
    policy_top2_gap(probs) < gap_threshold
}

pub fn is_confident(probs: &[f32; HYDRA_ACTION_SPACE], threshold: f32) -> bool {
    policy_top1_confidence(probs) >= threshold
}

pub fn sample_from_policy(probs: &[f32; HYDRA_ACTION_SPACE], rng_val: f32) -> u8 {
    let mut cumsum = 0.0f32;
    let mut last_positive = 0u8;
    for (i, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            last_positive = i as u8;
        }
        cumsum += p;
        if rng_val <= cumsum {
            return i as u8;
        }
    }
    last_positive
}

pub fn num_legal_actions(mask: &[bool; HYDRA_ACTION_SPACE]) -> usize {
    mask.iter().filter(|&&m| m).count()
}

pub fn argmax_legal(probs: &[f32; HYDRA_ACTION_SPACE], mask: &[bool; HYDRA_ACTION_SPACE]) -> u8 {
    let mut best = 0u8;
    let mut best_p = f32::NEG_INFINITY;
    for (i, (&p, &m)) in probs.iter().zip(mask.iter()).enumerate() {
        if m && p > best_p {
            best_p = p;
            best = i as u8;
        }
    }
    best
}

pub fn compute_entropy_from_logits(
    logits: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> f32 {
    let probs = normalize_policy_cpu(logits, legal_mask);
    policy_entropy(&probs)
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

pub fn policy_top1_confidence(probs: &[f32; HYDRA_ACTION_SPACE]) -> f32 {
    probs.iter().cloned().fold(0.0f32, f32::max)
}

pub fn batch_legal_masks_to_tensor<B: Backend>(
    masks: &[[bool; HYDRA_ACTION_SPACE]],
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch = masks.len();
    let mut flat = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    for (i, mask) in masks.iter().enumerate() {
        for (j, &m) in mask.iter().enumerate() {
            if m {
                flat[i * HYDRA_ACTION_SPACE + j] = 1.0;
            }
        }
    }
    Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([batch, HYDRA_ACTION_SPACE])
}

/// Runs inference with wall-clock time measurement against a budget.
pub fn infer_action_timed<B: Backend>(
    policy_logits: Tensor<B, 2>,
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
    budget_ms: u64,
) -> (u8, [f32; HYDRA_ACTION_SPACE], bool) {
    let start = std::time::Instant::now();
    let (action, policy) = infer_action(policy_logits, legal_mask);
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let within_budget = elapsed_ms <= budget_ms;
    (action, policy, within_budget)
}

/// Returns the fraction of batch elements where argmax picks an illegal action.
pub fn illegal_action_rate<B: Backend>(logits: Tensor<B, 2>, legal_mask: Tensor<B, 2>) -> f32 {
    let neg_inf = (legal_mask.clone().ones_like() - legal_mask.clone()) * (-1e9f32);
    let raw_predicted = logits.clone().argmax(1);
    let masked = logits + neg_inf;
    let predicted = masked.argmax(1);
    let same = predicted.equal(raw_predicted).int().sum();
    let batch = legal_mask.dims()[0] as f32;
    1.0 - same.into_scalar().elem::<f32>() / batch
}

/// Runs masked softmax inference, returns (best_action, policy_probs).
pub fn infer_action<B: Backend>(
    policy_logits: Tensor<B, 2>,
    legal_mask: &[bool; HYDRA_ACTION_SPACE],
) -> (u8, [f32; HYDRA_ACTION_SPACE]) {
    let device = policy_logits.device();
    let mask_tensor = legal_mask_to_tensor(legal_mask, &device);
    let neg_inf = (mask_tensor.ones_like() - mask_tensor) * (-1e9f32);
    let masked = policy_logits + neg_inf;
    let probs = activation::softmax(masked, 1);
    let probs_data = probs.to_data();
    let mut policy = [0.0f32; HYDRA_ACTION_SPACE];
    if let Ok(probs_slice) = probs_data.as_slice::<f32>() {
        policy.copy_from_slice(&probs_slice[..HYDRA_ACTION_SPACE]);
    }

    let mut best_action = 0u8;
    let mut best_prob = f32::NEG_INFINITY;
    for (i, &p) in policy.iter().enumerate() {
        if legal_mask[i] && p > best_prob {
            best_prob = p;
            best_action = i as u8;
        }
    }
    (best_action, policy)
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
    fn inference_picks_legal_action() {
        let device = Default::default();
        let logits = Tensor::<B, 2>::from_floats(
            [[
                10.0, -10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]],
            &device,
        );
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[1] = true;
        mask[2] = true;
        let (action, policy) = infer_action(logits, &mask);
        assert!(mask[action as usize], "picked illegal action {action}");
        let sum: f32 = policy.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "policy should sum to 1, got {sum}"
        );
    }

    #[test]
    fn agari_guard_prevents_illegal() {
        let device = Default::default();
        let mut logits_data = [0.0f32; HYDRA_ACTION_SPACE];
        logits_data[43] = 100.0;
        let logits = Tensor::<B, 1>::from_floats(&logits_data[..], &device).unsqueeze_dim::<2>(0);
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[45] = true;
        let (action, _) = infer_action(logits, &mask);
        assert_ne!(action, 43, "agari (43) is illegal but has highest logit");
        assert!(mask[action as usize], "must pick legal: got {action}");
    }

    #[test]
    fn inference_config_defaults() {
        let cfg = InferenceConfig::default();
        assert_eq!(cfg.on_turn_budget_ms, 150);
        assert_eq!(cfg.call_reaction_budget_ms, 50);
        assert!(cfg.agari_guard);
    }

    #[test]
    fn illegal_actions_get_zero_probability() {
        let device = Default::default();
        let mut logits_data = [0.0f32; HYDRA_ACTION_SPACE];
        logits_data[0] = 5.0;
        logits_data[1] = 3.0;
        logits_data[2] = 1.0;
        let logits = Tensor::<B, 1>::from_floats(&logits_data[..], &device).unsqueeze_dim::<2>(0);
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[2] = true;
        let (_, policy) = infer_action(logits, &mask);
        assert!(
            policy[1] < 1e-6,
            "illegal action 1 should have ~0 prob: {}",
            policy[1]
        );
        assert!(
            policy[0] > 0.1,
            "legal action 0 should have significant prob"
        );
        assert!(policy[2] > 0.01, "legal action 2 should have some prob");
    }

    #[test]
    fn normalize_policy_cpu_sums_to_one() {
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits[0] = 5.0;
        logits[5] = 3.0;
        logits[10] = 1.0;
        let mut mask = [false; HYDRA_ACTION_SPACE];
        mask[0] = true;
        mask[5] = true;
        mask[10] = true;
        let probs = normalize_policy_cpu(&logits, &mask);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum: {sum}");
        assert!(probs[0] > probs[5]);
        assert!(probs[5] > probs[10]);
    }

    #[test]
    fn mask_policy_cpu_renormalizes_legal_mass() {
        let mut policy = [0.0f32; HYDRA_ACTION_SPACE];
        policy[1] = 0.8;
        policy[2] = 0.2;
        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[2] = true;
        legal[3] = true;
        let masked = mask_policy_cpu(&policy, &legal);
        assert_eq!(masked[1], 0.0);
        assert!((masked[2] - 1.0).abs() < 1e-6);
        assert_eq!(masked[3], 0.0);
    }

    #[test]
    fn sample_from_policy_respects_distribution() {
        let mut probs = [0.0f32; HYDRA_ACTION_SPACE];
        probs[0] = 0.7;
        probs[1] = 0.3;
        let a0 = sample_from_policy(&probs, 0.0);
        assert_eq!(a0, 0);
        let a1 = sample_from_policy(&probs, 0.8);
        assert_eq!(a1, 1);
    }

    #[test]
    fn inference_respects_time_budget() {
        let device = Default::default();
        let model = crate::model::HydraModelConfig::actor().init::<B>(&device);
        let x = Tensor::<B, 3>::zeros([1, NUM_CHANNELS, 34], &device);
        let out = model.forward(x);
        let mut mask = [true; HYDRA_ACTION_SPACE];
        mask[45] = false;
        let (action, policy, within) = infer_action_timed(out.policy_logits, &mask, u64::MAX);
        assert!(mask[action as usize], "must pick legal action");
        let sum: f32 = policy.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "policy sum: {sum}");
        assert!(within, "unbounded budget should always report within=true");
    }

    #[test]
    fn inference_server_respects_time_budget() {
        let device = Default::default();
        let mut server = make_server(&device);
        server.config.on_turn_budget_ms = u64::MAX;
        let obs = [0.0f32; OBS_FLAT_SIZE];
        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[0] = true;
        legal[1] = true;
        let (action, policy, within) = server.infer_timed(&obs, &legal);
        assert!(legal[action as usize]);
        assert!(within, "unbounded budget should always report within=true");
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
        // Insert with Authoritative trust so the runtime path picks it up.
        let mut result = PonderResult::learner_only_stub(exit_policy, 0.3, 5, 64);
        result.trust_level = TrustLevel::Authoritative;
        server.cache_ponder_result(hash, result);
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
        server.config.call_reaction_budget_ms = u64::MAX;
        let obs = [0.0f32; OBS_FLAT_SIZE];
        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[3] = true;
        legal[4] = true;
        let (action, policy, within) = server.infer_call_reaction_timed(&obs, &legal);
        assert!(legal[action as usize]);
        assert!(
            within,
            "unbounded call budget should always report within=true"
        );
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_needs_search_close_gap() {
        let mut probs = [0.0f32; HYDRA_ACTION_SPACE];
        probs[0] = 0.35;
        probs[1] = 0.34;
        probs[2] = 0.31;
        assert!(
            needs_search(&probs, 0.05),
            "top-2 gap of 0.01 < threshold 0.05 should trigger search"
        );
    }

    #[test]
    fn learner_only_cache_does_not_influence_runtime() {
        let device = Default::default();
        let server = make_server(&device);
        let obs = [0.0f32; OBS_FLAT_SIZE];
        let hash = InferenceServer::<B>::info_state_hash(&obs);
        let mut exit_policy = [0.0f32; HYDRA_ACTION_SPACE];
        exit_policy[5] = 1.0;
        let result = PonderResult::learner_only_stub(exit_policy, 0.9, 8, 500);
        server.cache_ponder_result(hash, result);

        // learner-only lookup should find it
        assert!(server.lookup_ponder(hash).is_some());
        // trusted lookup for Authoritative should not
        assert!(
            server
                .lookup_ponder_trusted(hash, TrustLevel::Authoritative)
                .is_none()
        );

        let mut legal = [false; HYDRA_ACTION_SPACE];
        legal[0] = true;
        legal[5] = true;
        let (action, _, _) = server.infer_timed(&obs, &legal);
        // Runtime should NOT use the cached policy since it's LearnerOnly.
        // Without cache, the network decides (not necessarily action 5).
        assert!(
            legal[action as usize],
            "action must be legal regardless of trust path"
        );
    }

    #[test]
    fn cache_invalidation_prevents_reuse() {
        let device = Default::default();
        let server = make_server(&device);
        let obs = [0.0f32; OBS_FLAT_SIZE];
        let hash = InferenceServer::<B>::info_state_hash(&obs);
        let mut result = PonderResult::learner_only_stub([0.0f32; HYDRA_ACTION_SPACE], 0.5, 4, 100);
        result.trust_level = TrustLevel::Authoritative;
        server.cache_ponder_result(hash, result);
        assert!(
            server
                .lookup_ponder_trusted(hash, TrustLevel::Authoritative)
                .is_some()
        );

        server.invalidate_cache();
        assert!(
            server
                .lookup_ponder_trusted(hash, TrustLevel::Authoritative)
                .is_none(),
            "invalidated cache should reject stale entries"
        );
    }
}
