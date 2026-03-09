use criterion::{Criterion, criterion_group, criterion_main};
use hydra_core::ct_smc::{CtSmc, CtSmcConfig};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn bench_ct_smc_full_pipeline(c: &mut Criterion) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut row_sums = [0u8; 34];
    for (i, val) in row_sums.iter_mut().enumerate().take(13) {
        *val = if i < 4 { 2 } else { 1 };
    }
    let col_sums = [13, 13, 13, 70];
    let log_omega = [[0.0f64; 4]; 34];

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

criterion_group!(benches, bench_ct_smc_full_pipeline);
criterion_main!(benches);
