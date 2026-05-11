use super::*;

#[test]
fn log_sum_exp_and_config_helpers_cover_basic_edges() {
    assert_eq!(log_sum_exp_f32(&[]), f32::NEG_INFINITY);
    assert_eq!(
        log_sum_exp_f32(&[f32::NEG_INFINITY, f32::NEG_INFINITY]),
        f32::NEG_INFINITY
    );
    assert!((log_sum_exp_f32(&[0.0, 0.0]) - (2.0f32).ln()).abs() < 1e-6);

    let cfg = RobustOpponentConfig::new(0.3, 6)
        .with_tau_iters(12)
        .with_tau_arch(0.7)
        .with_archetypes(5)
        .with_epsilon(0.4);
    assert_eq!(cfg.epsilon, 0.4);
    assert_eq!(cfg.tau_search_iters, 12);
    assert_eq!(cfg.num_archetypes, 5);
    assert_eq!(cfg.tau_arch, 0.7);
    assert!(cfg.summary().contains("robust(eps=0.40, arch=5, tau=0.7)"));
    assert!(cfg.validate().is_ok());

    let bad_eps = RobustOpponentConfig::default().with_epsilon(0.0);
    assert_eq!(bad_eps.validate(), Err("epsilon must be positive"));
    let bad_arch = RobustOpponentConfig::default().with_archetypes(0);
    assert_eq!(bad_arch.validate(), Err("need at least 1 archetype"));
}

#[test]
fn archetype_weights_cover_reset_confidence_entropy_and_posterior_updates() {
    let mut weights = ArchetypeWeights::uniform(4);
    assert_eq!(weights.num_archetypes(), 4);
    assert!(!weights.is_confident(0.3));
    assert_eq!(
        weights.most_likely(),
        3,
        "ties break toward the last max in iterator order"
    );
    assert!((weights.entropy() - (4.0f32).ln()).abs() < 1e-5);

    weights.update_posterior(&[0.0, 0.0, 2.0, 1.0]);
    let sum: f32 = weights.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert_eq!(weights.most_likely(), 2);
    assert!(weights.is_confident(0.4));

    weights.reset();
    assert!(weights.weights.iter().all(|&w| (w - 0.25).abs() < 1e-6));
}

#[test]
fn epsilon_kl_and_expected_value_helpers_match_simple_cases() {
    assert_eq!(calibrate_epsilon(&[], 0.95), 0.2);
    assert_eq!(calibrate_epsilon(&[0.4, 0.1, 0.3, 0.2], 0.5), 0.3);
    assert_eq!(calibrate_epsilon(&[0.4, 0.1, 0.3, 0.2], 10.0), 0.4);

    let q = [0.2, 0.3, 0.5];
    let p = [0.1, 0.4, 0.5];
    assert!(kl_divergence(&q, &p) > 0.0);
    assert!((expected_q_value(&[0.25, 0.75], &[2.0, 4.0]) - 3.5).abs() < 1e-6);
}

#[test]
fn robust_tau_converges() {
    let p = vec![0.3, 0.5, 0.2];
    let q = vec![1.0, 0.5, 2.0];
    let eps = 0.1;
    let (tau, q_tau) = find_robust_tau(&p, &q, eps, 20);
    assert!(tau > 0.0);
    let sum: f32 = q_tau.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "q_tau should sum to 1");
    let mut kl = 0.0f32;
    for i in 0..3 {
        if q_tau[i] > 1e-10 && p[i] > 1e-10 {
            kl += q_tau[i] * (q_tau[i] / p[i]).ln();
        }
    }
    let kl_error = (kl - eps).abs() / eps;
    assert!(kl_error < 0.05, "KL={kl} should be within 5% of eps={eps}");
}

#[test]
fn archetype_softmin_equal_q() {
    let qs = vec![vec![1.0, 2.0, 3.0]; 4];
    let w = vec![0.25; 4];
    let result = archetype_softmin(&qs, &w, 1.0);
    assert_eq!(result.len(), 3);
    for (i, &val) in result.iter().enumerate().take(3) {
        let expected = i as f32 + 1.0;
        assert!(
            (val - expected).abs() < 0.1,
            "expected ~{expected}, got {val}",
        );
    }
}

#[test]
fn archetype_softmin_different_q_shifts() {
    let qs = vec![
        vec![1.0, 5.0, 3.0],
        vec![5.0, 1.0, 3.0],
        vec![3.0, 3.0, 1.0],
        vec![3.0, 3.0, 5.0],
    ];
    let w = vec![0.25; 4];
    let result = archetype_softmin(&qs, &w, 1.0);
    assert_eq!(result.len(), 3);
    for v in &result {
        assert!(v.is_finite());
    }
}

#[test]
fn robust_tau_uniform_policy() {
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let q = vec![1.0, 2.0, 3.0, 4.0];
    let (tau, q_tau) = find_robust_tau(&p, &q, 0.05, 20);
    assert!(tau > 0.0);
    let sum: f32 = q_tau.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[test]
fn robust_tau_zero_prior_no_nan() {
    let p = vec![0.0, 0.5, 0.0, 0.5];
    let q = vec![1.0, 2.0, 3.0, 4.0];
    let (tau, q_tau) = find_robust_tau(&p, &q, 0.1, 20);
    assert!(
        tau.is_finite(),
        "tau should be finite with zero priors: {tau}"
    );
    for (i, &v) in q_tau.iter().enumerate() {
        assert!(v.is_finite(), "q_tau[{i}] should be finite, got {v}");
    }
    let sum: f32 = q_tau.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "q_tau should sum to 1, got {sum}");
}

#[test]
fn robust_tau_identical_q_stays_close_to_prior() {
    let p = vec![0.3, 0.5, 0.2];
    let q = vec![1.0, 1.0, 1.0];
    let (_, q_tau) = find_robust_tau(&p, &q, 0.1, 20);
    let sum: f32 = q_tau.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
    for i in 0..3 {
        assert!(
            (q_tau[i] - p[i]).abs() < 0.2,
            "identical Q -> q_tau should be near prior: {} vs {}",
            q_tau[i],
            p[i]
        );
    }
}

#[test]
fn kl_divergence_zero_for_equal_distributions() {
    let p = [0.2, 0.3, 0.5];
    assert!(kl_divergence(&p, &p) < 1e-8);
}

#[test]
fn robust_backup_is_not_above_prior_expectation() {
    let p = vec![0.2, 0.5, 0.3];
    let q = vec![3.0, 1.0, 2.0];
    let prior_value = expected_q_value(&p, &q);
    let (_, robust_value, q_tau) = robust_backup(&p, &q, 0.1, 24);
    assert!(robust_value <= prior_value + 1e-4);
    assert!((q_tau.iter().sum::<f32>() - 1.0).abs() < 0.01);
    assert!(kl_divergence(&q_tau, &p) <= 0.11);
}

#[test]
fn robust_policy_matches_backup_policy() {
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let q = vec![4.0, 1.0, 3.0, 2.0];
    let policy = robust_policy(&p, &q, 0.05, 20);
    let (_, _, backup_policy) = robust_backup(&p, &q, 0.05, 20);
    assert_eq!(policy.len(), backup_policy.len());
    for i in 0..policy.len() {
        assert!((policy[i] - backup_policy[i]).abs() < 1e-6);
    }
}
