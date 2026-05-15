use super::*;
use crate::model::HydraModelInit;
use burn::backend::NdArray;

type B = NdArray<f32>;

fn make_server(device: &<B as burn::tensor::backend::BackendTypes>::Device) -> InferenceServer<B> {
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
            10.0, -10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
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
