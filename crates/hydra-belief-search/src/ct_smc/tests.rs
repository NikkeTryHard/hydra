use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn compute_ess_from_log_weights_uniform() {
    let w = vec![0.0; 100];
    let ess = compute_ess_from_log_weights(&w);
    assert!((ess - 100.0).abs() < 0.01, "uniform -> ESS=N: {ess}");
}

#[test]
fn uniform_omega_matches_hypergeometric() {
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let mut row_sums = [0u8; 34];
    row_sums[0] = 1;
    row_sums[1] = 1;
    let col_sums = [1, 1, 0, 0];
    let log_omega = [[0.0f64; 4]; 34];
    let cfg = CtSmcConfig {
        rng_seed: 123,
        num_particles: 1000,
        ess_threshold: 0.4,
    };
    let mut smc = CtSmc::new(cfg);
    smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
    let mut count_00 = 0u32;
    for p in &smc.particles {
        if p.allocation[0][0] == 1 {
            count_00 += 1;
        }
    }
    let frac = count_00 as f64 / 1000.0;
    assert!(
        (frac - 0.5).abs() < 0.1,
        "uniform omega with r=[1,1] c=[1,1] should give ~50/50: {frac}"
    );
}

#[test]
fn particles_satisfy_constraints() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut row_sums = [0u8; 34];
    row_sums[0] = 2;
    row_sums[1] = 1;
    row_sums[2] = 1;
    let col_sums = [1, 1, 1, 1];
    let log_omega = [[0.0f64; 4]; 34];
    let cfg = CtSmcConfig {
        rng_seed: 42,
        num_particles: 32,
        ess_threshold: 0.4,
    };
    let mut smc = CtSmc::new(cfg);
    smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
    assert_eq!(smc.particles.len(), 32);
    for p in &smc.particles {
        for (k, &expected) in row_sums.iter().enumerate() {
            let rs: u8 = p.allocation[k].iter().sum();
            assert_eq!(rs, expected, "row {k}");
        }
        for (z, &expected) in col_sums.iter().enumerate() {
            let cs: usize = (0..34).map(|k| p.allocation[k][z] as usize).sum();
            assert_eq!(cs, expected, "col {z}");
        }
    }
}

#[test]
fn uniform_likelihood_high_ess() {
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let mut row_sums = [0u8; 34];
    row_sums[0] = 1;
    row_sums[1] = 1;
    let col_sums = [1, 1, 0, 0];
    let log_omega = [[0.0f64; 4]; 34];
    let cfg = CtSmcConfig {
        rng_seed: 42,
        num_particles: 64,
        ess_threshold: 0.4,
    };
    let mut smc = CtSmc::new(cfg);
    smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
    let ess = smc.ess();
    assert!(ess > 60.0, "ESS near P for uniform, got {ess}");
}

#[test]
fn compositions_counts() {
    assert_eq!(compositions(0).len(), 1);
    assert_eq!(compositions(1).len(), 4);
    assert_eq!(compositions(2).len(), 10);
    assert_eq!(compositions(3).len(), 20);
    assert_eq!(compositions(4).len(), 35);
}

#[test]
fn uniform_omega_marginals() {
    let mut rng = ChaCha8Rng::seed_from_u64(999);
    let mut row_sums = [0u8; 34];
    row_sums[0] = 1;
    row_sums[1] = 1;
    let col_sums = [1, 1, 0, 0];
    let log_omega = [[0.0f64; 4]; 34];
    let cfg = CtSmcConfig {
        rng_seed: 42,
        num_particles: 1000,
        ess_threshold: 0.4,
    };
    let mut smc = CtSmc::new(cfg);
    smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
    let mut count_tile0_col0 = 0usize;
    for p in &smc.particles {
        if p.allocation[0][0] == 1 {
            count_tile0_col0 += 1;
        }
    }
    let freq = count_tile0_col0 as f64 / 1000.0;
    assert!(
        (freq - 0.5).abs() < 0.1,
        "uniform marginal should be ~0.5, got {freq}"
    );
}

#[test]
fn update_with_likelihood_reweights() {
    let mut rng = ChaCha8Rng::seed_from_u64(77);
    let mut row_sums = [0u8; 34];
    row_sums[0] = 1;
    row_sums[1] = 1;
    let col_sums = [1, 1, 0, 0];
    let log_omega = [[0.0f64; 4]; 34];
    let cfg = CtSmcConfig {
        rng_seed: 77,
        num_particles: 64,
        ess_threshold: 0.4,
    };
    let mut smc = CtSmc::new(cfg);
    let likelihood = |p: &Particle| -> f64 { if p.allocation[0][0] == 1 { 0.0 } else { -10.0 } };
    smc.update(&row_sums, &col_sums, &log_omega, &likelihood, &mut rng);
    assert!(!smc.particles.is_empty());
}

#[test]
fn logsumexp_handles_extreme_values() {
    assert!((logsumexp(0.0, 0.0) - (2.0f64).ln()).abs() < 1e-10);
    assert!((logsumexp(f64::NEG_INFINITY, 0.0) - 0.0).abs() < 1e-10);
    assert!((logsumexp(0.0, f64::NEG_INFINITY) - 0.0).abs() < 1e-10);
    let big = logsumexp(1000.0, 1000.0);
    assert!((big - (1000.0 + (2.0f64).ln())).abs() < 1e-10);
    let small = logsumexp(-1000.0, -1000.0);
    assert!((small - (-1000.0 + (2.0f64).ln())).abs() < 1e-10);
}

#[test]
fn systematic_resample_preserves_count() {
    let mut rng = ChaCha8Rng::seed_from_u64(55);
    let mut row_sums = [0u8; 34];
    row_sums[0] = 1;
    let col_sums = [1, 0, 0, 0];
    let log_omega = [[0.0f64; 4]; 34];
    let cfg = CtSmcConfig {
        rng_seed: 55,
        num_particles: 32,
        ess_threshold: 0.4,
    };
    let mut smc = CtSmc::new(cfg);
    smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
    for (i, p) in smc.particles.iter_mut().enumerate() {
        p.log_weight = if i == 0 { 0.0 } else { -100.0 };
    }
    smc.systematic_resample(&mut rng);
    assert_eq!(smc.particles.len(), 32);
}

#[test]
fn extreme_omega_no_nan_inf() {
    let mut rng = ChaCha8Rng::seed_from_u64(88);
    let mut row_sums = [0u8; 34];
    row_sums[0] = 1;
    let col_sums = [1, 0, 0, 0];
    let mut log_omega = [[0.0f64; 4]; 34];
    log_omega[0] = [100.0, -100.0, -100.0, -100.0];
    let cfg = CtSmcConfig {
        rng_seed: 88,
        num_particles: 16,
        ess_threshold: 0.4,
    };
    let mut smc = CtSmc::new(cfg);
    smc.sample_particles(&row_sums, &col_sums, &log_omega, &mut rng);
    for p in &smc.particles {
        for k in 0..34 {
            for j in 0..4 {
                assert!(
                    (p.allocation[k][j] as f32).is_finite(),
                    "allocation should be finite"
                );
            }
        }
    }
}

#[test]
fn ess_degenerate_log_weights_use_uniform_mass() {
    let weights = [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NAN];
    let ess = compute_ess_from_log_weights(&weights);
    assert!(
        (ess - 3.0).abs() < 1e-6,
        "degenerate weights -> uniform ESS"
    );
}

#[test]
fn smc_degenerate_log_weights_use_uniform_mass() {
    let mut rng = ChaCha8Rng::seed_from_u64(91);
    let mut smc = CtSmc::new(CtSmcConfig {
        rng_seed: 91,
        num_particles: 2,
        ess_threshold: 0.4,
    });
    smc.particles = vec![
        Particle {
            allocation: [[1; 4]; 34],
            log_weight: f64::NEG_INFINITY,
        },
        Particle {
            allocation: [[3; 4]; 34],
            log_weight: f64::NAN,
        },
    ];

    assert!((smc.ess() - 2.0).abs() < 1e-6);
    assert!((smc.weighted_mean_tile_count(0, 0) - 2.0).abs() < 1e-6);
    smc.systematic_resample(&mut rng);
    assert_eq!(smc.particles.len(), 2);
    assert!(smc.particles.iter().all(|p| p.log_weight == 0.0));
}
