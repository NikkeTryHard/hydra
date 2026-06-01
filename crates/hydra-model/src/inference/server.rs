use burn::prelude::*;
use hydra_core::action::{AGARI, HYDRA_ACTION_SPACE};
use hydra_core::afbs::{PonderCache, PonderResult, TrustLevel};
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES};
use std::sync::Arc;

use crate::model::HydraModel;
use crate::saf::{SafConfig, SafMlp, apply_saf_logit, saf_tensor_from_observation};

use super::{
    InferenceConfig, OBS_FLAT_SIZE, argmax_legal, infer_action_timed, legal_mask_to_tensor,
    mask_policy_cpu,
};

pub struct InferenceServer<B: Backend> {
    pub actor: HydraModel<B>,
    pub ponder_cache: Arc<PonderCache>,
    pub saf_mlp: SafMlp<B>,
    pub config: InferenceConfig,
    saf_alpha: f32,
    device: <B as burn::tensor::backend::BackendTypes>::Device,
}

impl<B: Backend> InferenceServer<B> {
    pub fn new(
        actor: HydraModel<B>,
        ponder_cache: Arc<PonderCache>,
        saf_mlp: SafMlp<B>,
        saf_alpha: f32,
        config: InferenceConfig,
        device: <B as burn::tensor::backend::BackendTypes>::Device,
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
        actor: HydraModel<B>,
        saf_config: &SafConfig,
        config: InferenceConfig,
        device: <B as burn::tensor::backend::BackendTypes>::Device,
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
