use super::*;

#[test]
fn sinkhorn_converges_to_margins() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let b = sinkhorn_project(&kernel, &row_sums, &col_sums, 100, 1e-6);
    for i in 0..34 {
        let s: f64 = (0..4).map(|j| b[i * 4 + j]).sum();
        assert!((s - 4.0).abs() < 0.01, "row {i} sum = {s}");
    }
    for j in 0..4 {
        let s: f64 = (0..34).map(|i| b[i * 4 + j]).sum();
        assert!((s - 34.0).abs() < 0.01, "col {j} sum = {s}");
    }
}

#[test]
fn mixture_weight_update_is_bayesian() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let mut mix = MixtureSib::new(3, &kernel, &row_sums, &col_sums);
    let w_before = mix.weights();
    let sum_before: f64 = w_before.iter().sum();
    assert!((sum_before - 1.0).abs() < 0.01);
    mix.bayesian_update(&[0.0, -1.0, -2.0]);
    let w_after = mix.weights();
    let sum_after: f64 = w_after.iter().sum();
    assert!(
        (sum_after - 1.0).abs() < 0.01,
        "weights should sum to 1 after update"
    );
    assert!(
        w_after[0] > w_after[1],
        "component 0 should have higher weight"
    );
    assert!(
        w_after[1] > w_after[2],
        "component 1 should have higher weight than 2"
    );
}

#[test]
fn sinkhorn_nonuniform_kernel() {
    let mut kernel = [1.0f64; 136];
    for i in 0..10 {
        kernel[i * 4] = 2.0;
    }
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let b = sinkhorn_project(&kernel, &row_sums, &col_sums, 500, 1e-6);
    for i in 0..34 {
        let s: f64 = (0..4).map(|j| b[i * 4 + j]).sum();
        assert!((s - 4.0).abs() < 0.5, "nonuniform row {i} sum = {s}");
    }
}

#[test]
fn mixture_marginal_sums_to_total() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
    let marginal = mix.marginal_belief();
    let total: f64 = marginal.iter().sum();
    assert!((total - 136.0).abs() < 1.0, "marginal sum = {total}");
}

#[test]
fn mixture_ess_equals_num_components_uniform() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
    let ess = mix.ess();
    assert!(
        (ess - 4.0).abs() < 0.01,
        "uniform weights -> ESS=N, got {ess}"
    );
}

#[test]
fn bayesian_update_collapsed_no_nan() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let mut mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
    mix.bayesian_update(&[-1000.0, -1000.0, -1000.0, 0.0]);
    let w = mix.weights();
    for (i, &wi) in w.iter().enumerate() {
        assert!(
            wi.is_finite(),
            "weight[{i}] should be finite after collapsed update, got {wi}"
        );
    }
    let sum: f64 = w.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "weights should sum to 1, got {sum}"
    );
}

#[test]
fn bayesian_update_all_degenerate_resets_uniform() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let mut mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
    mix.bayesian_update(&[
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::NEG_INFINITY,
    ]);
    let weights = mix.weights();
    assert_eq!(weights.len(), 4);
    for (idx, weight) in weights.iter().enumerate() {
        assert!(weight.is_finite(), "weight[{idx}] should be finite");
        assert!(
            (*weight - 0.25).abs() < 1e-12,
            "weight[{idx}] should reset uniform"
        );
    }
}

#[test]
fn mixture_ess_decreases_after_biased_update() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let mut mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
    let ess_before = mix.ess();
    mix.bayesian_update(&[0.0, -5.0, -10.0, -15.0]);
    let ess_after = mix.ess();
    assert!(ess_after < ess_before, "biased update should reduce ESS");
}

#[test]
fn entropy_regularizer_increases_weight_entropy() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let mut mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
    mix.bayesian_update(&[0.0, -10.0, -10.0, -10.0]);
    let before = mix.weight_entropy();
    mix.apply_entropy_regularizer(0.2);
    let after = mix.weight_entropy();
    assert!(
        after > before,
        "entropy regularizer should increase entropy"
    );
}

#[test]
fn split_low_ess_adds_component() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let mut mix = MixtureSib::new(3, &kernel, &row_sums, &col_sums);
    mix.bayesian_update(&[0.0, -20.0, -20.0]);
    let before = mix.num_components();
    assert!(mix.split_dominant_component_if_low_ess(0.9, 0.1));
    assert_eq!(mix.num_components(), before + 1);
}

#[test]
fn merge_identical_components_reduces_component_count() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let mut mix = MixtureSib::new(3, &kernel, &row_sums, &col_sums);
    let before = mix.num_components();
    assert!(mix.merge_closest_components(0.0));
    assert_eq!(mix.num_components(), before - 1);
}

#[test]
fn diversity_penalty_keeps_weights_normalized() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let mut mix = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
    mix.apply_diversity_penalty(0.5);
    let sum: f64 = mix.weights().iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "weights should remain normalized");
}

#[test]
fn sinkhorn_log_domain_handles_extreme_kernel_scales() {
    let mut kernel = [1.0f64; 136];
    kernel[0] = 1e-200;
    kernel[1] = 1e200;
    kernel[2] = 1e-120;
    kernel[3] = 1e120;
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0; 4];
    let belief = sinkhorn_project(&kernel, &row_sums, &col_sums, 200, 1e-6);

    for (idx, value) in belief.iter().enumerate() {
        assert!(
            value.is_finite(),
            "belief[{idx}] should be finite, got {value}"
        );
        assert!(
            *value >= 0.0,
            "belief[{idx}] should be non-negative, got {value}"
        );
    }

    for i in 0..34 {
        let row_sum: f64 = (0..4).map(|j| belief[i * 4 + j]).sum();
        assert!((row_sum - 4.0).abs() < 0.05, "row {i} sum = {row_sum}");
    }
}
