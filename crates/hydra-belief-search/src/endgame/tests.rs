use super::*;

#[test]
fn endgame_returns_finite_q() {
    let particles = vec![
        Particle {
            allocation: [[0; 4]; 34],
            log_weight: 0.0,
        },
        Particle {
            allocation: [[0; 4]; 34],
            log_weight: 0.0,
        },
    ];
    let mut mask = [false; HYDRA_ACTION_SPACE];
    mask[0] = true;
    mask[1] = true;
    let eval = |_: &Particle, a: u8| a as f32 * 0.1;
    let q = pimc_endgame_q(&particles, &mask, &eval);
    assert!(q[0].is_finite());
    assert!(q[1].is_finite());
    assert!((q[0] - 0.0).abs() < 1e-5);
    assert!((q[1] - 0.1).abs() < 1e-5);
}

#[test]
fn top_mass_selects_heavy_particles() {
    let particles = vec![
        Particle {
            allocation: [[0; 4]; 34],
            log_weight: 0.0,
        },
        Particle {
            allocation: [[0; 4]; 34],
            log_weight: -10.0,
        },
        Particle {
            allocation: [[0; 4]; 34],
            log_weight: -0.1,
        },
    ];
    let selected = top_mass_particles(&particles, 0.95);
    assert!(selected.len() <= 3);
    assert!(
        selected.contains(&0),
        "highest weight particle should be selected"
    );
}

#[test]
fn endgame_empty_particles() {
    let mask = [true; HYDRA_ACTION_SPACE];
    let eval = |_: &Particle, _: u8| 1.0f32;
    let q = pimc_endgame_q(&[], &mask, &eval);
    assert!(q.iter().all(|&v| v == 0.0), "empty particles -> zero Q");
}

#[test]
fn top_mass_empty_returns_empty() {
    assert!(top_mass_particles(&[], 0.95).is_empty());
}

#[test]
fn endgame_respects_legal_mask() {
    let particles = vec![Particle {
        allocation: [[0; 4]; 34],
        log_weight: 0.0,
    }];
    let mut mask = [false; HYDRA_ACTION_SPACE];
    mask[5] = true;
    let eval = |_: &Particle, a: u8| a as f32;
    let q = pimc_endgame_q(&particles, &mask, &eval);
    assert!((q[5] - 5.0).abs() < 1e-5);
    assert!(q[0] == 0.0, "illegal action should have Q=0");
    assert!(q[4] == 0.0, "illegal action should have Q=0");
}

#[test]
fn endgame_with_weighted_particles() {
    let particles = vec![
        Particle {
            allocation: [[1; 4]; 34],
            log_weight: 0.0,
        },
        Particle {
            allocation: [[100; 4]; 34],
            log_weight: -5.0,
        },
    ];
    let selected = top_mass_particles(&particles, 0.95);
    assert!(selected.contains(&0));
    let mut mask = [false; HYDRA_ACTION_SPACE];
    mask[0] = true;
    mask[1] = true;
    mask[2] = true;
    let eval = |p: &Particle, a: u8| p.allocation[a as usize][0] as f32;
    let q = pimc_endgame_q(&particles, &mask, &eval);
    let expected = ((1.0f32 * 1.0) + 100.0 * (-5.0f32).exp()) / (1.0 + (-5.0f32).exp());
    for (i, &qi) in q.iter().enumerate().take(3) {
        assert!(qi.is_finite(), "q[{i}] should be finite");
        assert!(
            (qi - expected).abs() < 1e-4,
            "q[{i}]={qi} expected {expected}"
        );
    }
}

#[test]
fn topk_endgame_preserves_selected_particle_weights() {
    let particles = vec![
        Particle {
            allocation: [[1; 4]; 34],
            log_weight: 0.0,
        },
        Particle {
            allocation: [[9; 4]; 34],
            log_weight: -2.0,
        },
    ];
    let mut mask = [false; HYDRA_ACTION_SPACE];
    mask[0] = true;
    let eval = |p: &Particle, _: u8| p.allocation[0][0] as f32;
    let q = pimc_endgame_q_topk(&particles, &mask, &eval, 1.0);
    let expected = ((1.0f32 * 1.0) + 9.0 * (-2.0f32).exp()) / (1.0 + (-2.0f32).exp());
    assert!(
        (q[0] - expected).abs() < 1e-4,
        "weighted top-k Q should match posterior mass"
    );
}

#[test]
fn endgame_degenerate_log_weights_use_uniform_mass() {
    let particles = vec![
        Particle {
            allocation: [[1; 4]; 34],
            log_weight: f64::NEG_INFINITY,
        },
        Particle {
            allocation: [[3; 4]; 34],
            log_weight: f64::NAN,
        },
    ];
    let mut mask = [false; HYDRA_ACTION_SPACE];
    mask[0] = true;
    let eval = |p: &Particle, _: u8| p.allocation[0][0] as f32;
    let q = pimc_endgame_q(&particles, &mask, &eval);
    assert!(q[0].is_finite());
    assert!((q[0] - 2.0).abs() < 1e-6, "uniform fallback expected");
}
