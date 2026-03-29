use std::fs;
use std::path::PathBuf;

use super::*;
use crate::preflight_runtime::classify_probe_detail;
use crate::test_loose_replay_fixtures::single_loose_train_manifest;
use hydra_train::preflight::{PreflightConfig, ProbeStatus};

fn dummy_config() -> TrainConfig {
    TrainConfig {
        data_dir: PathBuf::from("/tmp/data"),
        output_dir: PathBuf::from("/tmp/out"),
        num_epochs: 1,
        batch_size: 256,
        microbatch_size: Some(64),
        validation_microbatch_size: Some(32),
        exit_sidecar_path: Some(PathBuf::from("/tmp/exit.sidecar")),
        delta_q_sidecar_path: Some(PathBuf::from("/tmp/delta-q.sidecar")),
        train_fraction: 0.9,
        augment: true,
        resume_checkpoint: None,
        seed: 7,
        advanced_loss: None,
        rl: None,
        bc: Default::default(),
        device: "cpu".to_string(),
        buffer_games: 16,
        buffer_samples: 128,
        num_threads: Some(6),
        tensorboard: false,
        archive_queue_bound: 8,
        validation_every_n_epochs: 1,
        max_skip_logs_per_source: 4,
        log_every_n_steps: 10,
        validate_every_n_steps: 10,
        checkpoint_every_n_steps: 10,
        max_train_steps: None,
        max_validation_batches: None,
        max_validation_samples: None,
        preflight: PreflightConfig::default(),
        precision_mode: crate::config::PrecisionMode::Fp32,
    }
}

fn runtime_tuple_score_cache(
    entries: &[(RuntimeTuple, RuntimeTupleStats)],
) -> BTreeMap<RuntimeTuple, RuntimeTupleStats> {
    entries.iter().copied().collect()
}

fn runtime_refine_cache_entries(
    entries: &[(RuntimeTuple, RuntimeTupleStats)],
) -> BTreeMap<(usize, usize, usize), RuntimeTupleStats> {
    entries.iter().copied().collect()
}

fn ranked_runtime_tuples(ranked: &[RankedLoaderRuntime]) -> Vec<RuntimeTuple> {
    ranked.iter().map(|entry| entry.tuple).collect()
}

fn runtime_seed(
    train_microbatch_size: usize,
    tuple: RuntimeTuple,
    warmup_steps: usize,
    measure_steps: usize,
    stats: RuntimeTupleStats,
) -> LoaderRuntimeScoreSeed {
    LoaderRuntimeScoreSeed {
        train_microbatch_size,
        tuple,
        warmup_steps,
        measure_steps,
        stats,
    }
}

#[test]
fn runtime_tuple_helpers_cover_defaults_round_trip_mean_budget_and_formatting() {
    let mut config = dummy_config();

    assert_eq!(runtime_tuple_key(&config), (8, 128, 16));
    assert_eq!(current_runtime_tuple(&config), (8, 128, 16));

    let tuple = (17, 2048, 33);
    apply_runtime_tuple(&mut config, tuple);
    assert_eq!(current_runtime_tuple(&config), tuple);

    assert_eq!(RuntimeTupleStats::default().mean(), 0.0);
    assert!(
        (RuntimeTupleStats {
            count: 3,
            sum: 18.0
        }
        .mean()
            - 6.0)
            .abs()
            < 1e-12
    );
    assert_eq!(
        format_runtime_knob_candidate_summary("256", 123.456, "128", 120.1),
        "candidate=256 throughput=123.46 samples/s best=128 (120.10 samples/s)"
    );
    assert_eq!(
        format_runtime_knob_candidate_summary("64", -1.234, "32", -2.0),
        "candidate=64 throughput=-1.23 samples/s best=32 (-2.00 samples/s)"
    );
}

#[test]
fn runtime_tuple_mutation_and_refine_gate_helpers_cover_pure_decisions() {
    let mut config = dummy_config();
    apply_runtime_tuple(&mut config, (32, 512, 64));

    assert_eq!(current_runtime_tuple(&config), (32, 512, 64));
    assert!(should_refine_close_tuples(&[(8, 128, 16), (16, 256, 32)]));
    assert!(!should_refine_close_tuples(&[(8, 128, 16)]));
}

#[test]
fn rank_and_select_close_runtime_tuples_use_score_order_threshold_and_limit() {
    let mut scores = vec![
        ((32, 512, 64), 90.0),
        ((16, 256, 32), 100.0),
        ((64, 1024, 128), 89.9),
        ((8, 128, 16), 95.0),
    ];

    rank_runtime_tuple_scores(&mut scores);

    assert_eq!(scores[0].0, (16, 256, 32));
    assert_eq!(scores[1].0, (8, 128, 16));
    assert_eq!(scores[2].0, (32, 512, 64));
    assert_eq!(scores[3].0, (64, 1024, 128));

    assert_eq!(
        close_runtime_tuples(&scores, 100.0, 0.10, 2),
        vec![(16, 256, 32), (8, 128, 16)]
    );
    assert_eq!(
        close_runtime_tuples(&scores, 100.0, 0.10, 3),
        vec![(16, 256, 32), (8, 128, 16), (32, 512, 64)]
    );
}

#[test]
fn rank_runtime_tuple_scores_preserves_nan_position_when_comparison_is_equal() {
    let mut scores = vec![
        ((32, 512, 64), f64::NAN),
        ((16, 256, 32), 100.0),
        ((8, 128, 16), 95.0),
    ];

    rank_runtime_tuple_scores(&mut scores);

    assert!(scores[0].1.is_nan());
    assert_eq!(scores[1].0, (16, 256, 32));
    assert_eq!(scores[2].0, (8, 128, 16));
}

#[test]
fn rank_runtime_tuple_scores_handles_multiple_nans_without_reordering_finite_tail() {
    let mut scores = vec![
        ((32, 512, 64), f64::NAN),
        ((16, 256, 32), 100.0),
        ((8, 128, 16), f64::NAN),
        ((4, 64, 8), 95.0),
    ];

    rank_runtime_tuple_scores(&mut scores);

    assert!(scores[0].1.is_nan());
    assert_eq!(scores[1].0, (16, 256, 32));
    assert!(scores[2].1.is_nan());
    assert_eq!(scores[3].0, (4, 64, 8));
}

#[test]
fn close_runtime_tuples_handles_zero_limit_and_negative_best_score() {
    let scores = vec![((8, 128, 16), -9.0), ((16, 256, 32), -10.0)];

    assert!(close_runtime_tuples(&scores, -9.0, 0.10, 0).is_empty());
    assert_eq!(
        close_runtime_tuples(&scores, -9.0, 0.0, 2),
        vec![(8, 128, 16)]
    );
}

#[test]
fn close_runtime_tuples_negative_margin_ratio_can_exclude_every_candidate() {
    let scores = vec![((8, 128, 16), 100.0), ((16, 256, 32), 99.0)];

    assert!(close_runtime_tuples(&scores, 100.0, -0.10, 5).is_empty());
}

#[test]
fn close_runtime_tuples_returns_empty_when_best_score_is_nan() {
    let scores = vec![((8, 128, 16), 100.0), ((16, 256, 32), 99.0)];

    assert!(close_runtime_tuples(&scores, f64::NAN, 0.10, 5).is_empty());
}

#[test]
fn autotune_loader_runtime_rejects_zero_threads_before_measurement() {
    let mut config = dummy_config();
    config.num_threads = Some(0);
    let manifest = DataManifest {
        sources: vec![],
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: true,
    };

    let result = autotune_loader_runtime(&config, &manifest, &LibTorchDevice::Cpu);

    assert_eq!(
        result,
        Err("runtime autotune produced invalid num_threads=0".to_string())
    );
}

#[test]
fn format_runtime_refine_summary_handles_empty_tuple_list() {
    let summary = format_runtime_refine_summary(&[], 2);

    assert!(summary.contains("Runtime refine:"));
    assert!(summary.contains("close_tuples=[]"));
    assert!(summary.contains("extra_samples=2"));
}

#[test]
fn tune_runtime_knob_selects_highest_score_and_keeps_first_tie() {
    let base = dummy_config();
    let candidates = [8usize, 16, 32];
    let mut seen = Vec::new();
    let mut score = |candidate: &TrainConfig| {
        seen.push(candidate.archive_queue_bound);
        Ok(match candidate.archive_queue_bound {
            8 => 10.0,
            16 => 20.0,
            32 => 20.0,
            other => panic!("unexpected candidate {other}"),
        })
    };

    let best = tune_runtime_knob(
        &base,
        "archive_queue_bound",
        &candidates,
        |value| value.to_string(),
        |cfg, value| cfg.archive_queue_bound = value,
        &mut score,
    )
    .expect("knob tuning should succeed");

    assert_eq!(best, 16);
    assert_eq!(seen, vec![8, 16, 32]);
}

#[test]
fn tune_runtime_knob_single_candidate_returns_without_replacement_logic() {
    let base = dummy_config();
    let candidates = [16usize];
    let mut seen = Vec::new();
    let mut score = |candidate: &TrainConfig| {
        seen.push(candidate.archive_queue_bound);
        Ok(42.0)
    };

    let best = tune_runtime_knob(
        &base,
        "archive_queue_bound",
        &candidates,
        |value| value.to_string(),
        |cfg, value| cfg.archive_queue_bound = value,
        &mut score,
    )
    .expect("single-candidate tuning should succeed");

    assert_eq!(best, 16);
    assert_eq!(seen, vec![16]);
}

#[test]
fn tune_runtime_knob_can_apply_and_select_non_queue_fields() {
    let base = dummy_config();
    let candidates = [64usize, 128, 256];
    let mut seen = Vec::new();
    let mut score = |candidate: &TrainConfig| {
        seen.push(candidate.buffer_samples);
        Ok(match candidate.buffer_samples {
            64 => 1.0,
            128 => 3.0,
            256 => 2.0,
            other => panic!("unexpected candidate {other}"),
        })
    };

    let best = tune_runtime_knob(
        &base,
        "buffer_samples",
        &candidates,
        |value| value.to_string(),
        |cfg, value| cfg.buffer_samples = value,
        &mut score,
    )
    .expect("knob tuning should succeed for buffer_samples");

    assert_eq!(best, 128);
    assert_eq!(seen, vec![64, 128, 256]);
}

#[test]
fn tune_runtime_knob_rejects_empty_candidate_lists() {
    let base = dummy_config();
    let candidates: [usize; 0] = [];
    let mut score = |_candidate: &TrainConfig| Ok(0.0);

    assert_eq!(
        tune_runtime_knob(
            &base,
            "buffer_samples",
            &candidates,
            |value| value.to_string(),
            |cfg, value| cfg.buffer_samples = value,
            &mut score,
        ),
        Err("no candidates available for buffer_samples".to_string())
    );
}

#[test]
fn rl_runtime_autotune_uses_probe_oom_classification() {
    assert_eq!(
        classify_probe_detail("CUDA out of memory"),
        ProbeStatus::Oom
    );
    assert_eq!(
        classify_probe_detail("oom while probing rl batch"),
        ProbeStatus::Oom
    );
}

#[test]
fn runtime_candidate_generators_cover_zero_small_and_saturated_inputs() {
    let mut config = dummy_config();

    config.buffer_samples = 0;
    assert_eq!(autotune_buffer_samples_candidates(&config), vec![1, 2, 4]);
    config.buffer_samples = 1;
    assert_eq!(autotune_buffer_samples_candidates(&config), vec![1, 2, 4]);
    config.buffer_samples = usize::MAX;
    assert_eq!(
        autotune_buffer_samples_candidates(&config),
        vec![usize::MAX]
    );

    config.buffer_games = 0;
    assert_eq!(autotune_buffer_games_candidates(&config), vec![1, 2]);
    config.buffer_games = 1;
    assert_eq!(autotune_buffer_games_candidates(&config), vec![1, 2]);
    config.buffer_games = usize::MAX;
    assert_eq!(autotune_buffer_games_candidates(&config), vec![usize::MAX]);

    config.archive_queue_bound = 0;
    assert_eq!(autotune_archive_queue_candidates(&config), vec![1, 2]);
    config.archive_queue_bound = 1;
    assert_eq!(autotune_archive_queue_candidates(&config), vec![1, 2]);
    config.archive_queue_bound = 2;
    assert_eq!(autotune_archive_queue_candidates(&config), vec![1, 2, 4]);
    config.archive_queue_bound = 9;
    assert_eq!(autotune_archive_queue_candidates(&config), vec![4, 9, 18]);

    config.archive_queue_bound = usize::MAX;
    assert_eq!(
        autotune_archive_queue_candidates(&config),
        vec![usize::MAX / 2, usize::MAX]
    );
}

#[test]
fn validate_runtime_threads_rejects_zero_and_accepts_none_or_positive() {
    let mut config = dummy_config();
    config.num_threads = Some(0);
    assert_eq!(
        validate_runtime_threads(&config).expect_err("zero threads should be invalid"),
        "runtime autotune produced invalid num_threads=0"
    );

    config.num_threads = None;
    validate_runtime_threads(&config).expect("none threads should be accepted");

    config.num_threads = Some(2);
    validate_runtime_threads(&config).expect("positive threads should be accepted");
}

#[test]
fn runtime_probe_loader_config_projects_runtime_fields_and_clears_sidecars() {
    let mut config = dummy_config();
    config.buffer_games = 23;
    config.buffer_samples = 2048;
    config.train_fraction = 0.75;
    config.seed = 99;
    config.archive_queue_bound = 12;
    config.max_skip_logs_per_source = 5;
    config.exit_sidecar_path = Some(std::path::PathBuf::from("/tmp/exit.jsonl"));
    config.delta_q_sidecar_path = Some(std::path::PathBuf::from("/tmp/delta.jsonl"));

    let loader = runtime_probe_loader_config(&config);

    assert_eq!(loader.buffer_games, 23);
    assert_eq!(loader.buffer_samples, 2048);
    assert_eq!(loader.train_fraction, 0.75);
    assert_eq!(loader.seed, 99);
    assert_eq!(loader.archive_queue_bound, 12);
    assert_eq!(loader.max_skip_logs_per_source, 5);
    assert!(loader.aggregate_skip_logs);
    assert!(loader.exit_sidecar.is_none());
    assert!(loader.delta_q_sidecar.is_none());
    assert!(loader.exit_sidecar_source_net_hash.is_none());
    assert!(loader.exit_sidecar_source_version.is_none());
    assert!(loader.delta_q_sidecar_source_net_hash.is_none());
    assert!(loader.delta_q_sidecar_source_version.is_none());
}

#[test]
fn runtime_measurement_helpers_cover_warmup_boundaries_zero_inputs_and_default_elapsed() {
    assert!(should_start_measurement(3, 3));
    assert!(!should_start_measurement(2, 3));
    assert!(should_start_measurement(0, 0));
    assert!(should_count_measured_samples(4, 3));
    assert!(!should_count_measured_samples(3, 3));
    assert!(!should_count_measured_samples(0, 0));
    assert!(should_count_measured_samples(1, 0));
    assert_eq!(measured_train_samples(0, 64), 0);
    assert_eq!(measured_train_samples(3, 0), 0);
    assert_eq!(measured_train_samples(5, 64), 320);
    assert_eq!(measured_train_samples(7, 9), 63);
    assert_eq!(finalize_runtime_probe_throughput(None, 320), 0.0);
}

#[test]
fn should_refine_close_tuples_requires_at_least_two_candidates() {
    assert!(!should_refine_close_tuples(&[]));
    assert!(!should_refine_close_tuples(&[(8, 128, 16)]));
    assert!(should_refine_close_tuples(&[(8, 128, 16), (16, 256, 32)]));
}

#[test]
fn coarse_search_candidate_count_multiplies_candidate_dimensions() {
    assert_eq!(coarse_search_candidate_count(&[1, 2], &[3, 4, 5], &[6]), 6);
    assert_eq!(coarse_search_candidate_count(&[], &[3, 4], &[5]), 0);
    assert_eq!(coarse_search_candidate_count(&[1], &[], &[3]), 0);
    assert_eq!(coarse_search_candidate_count(&[1], &[2], &[]), 0);
}

#[test]
fn close_runtime_tuples_with_zero_best_score_keeps_non_negative_scores_only() {
    let scores = vec![
        ((8, 128, 16), 0.0),
        ((16, 256, 32), -0.01),
        ((32, 512, 64), 0.5),
    ];

    assert_eq!(
        close_runtime_tuples(&scores, 0.0, 0.25, 5),
        vec![(8, 128, 16), (32, 512, 64)]
    );
}

#[test]
fn runtime_close_tuple_helpers_cover_high_margin_and_empty_cases() {
    let empty: Vec<(RuntimeTuple, f64)> = Vec::new();
    assert!(close_runtime_tuples(&empty, 10.0, 0.25, 3).is_empty());

    let scores = vec![((8, 128, 16), 100.0), ((16, 256, 32), 70.0)];
    assert_eq!(
        close_runtime_tuples(&scores, 100.0, 1.0, 5),
        vec![(8, 128, 16), (16, 256, 32)]
    );
}

#[test]
fn close_runtime_tuples_respects_max_candidate_limit_even_when_all_scores_pass() {
    let scores = vec![
        ((8, 128, 16), 100.0),
        ((16, 256, 32), 99.5),
        ((32, 512, 64), 99.0),
    ];

    assert_eq!(
        close_runtime_tuples(&scores, 100.0, 0.05, 2),
        vec![(8, 128, 16), (16, 256, 32)]
    );
}

#[test]
fn tune_runtime_knob_propagates_scoring_errors() {
    let base = dummy_config();
    let candidates = [8usize, 16];
    let mut score = |_candidate: &TrainConfig| Err("probe failed".to_string());

    let err = tune_runtime_knob(
        &base,
        "archive_queue_bound",
        &candidates,
        |value| value.to_string(),
        |cfg, value| cfg.archive_queue_bound = value,
        &mut score,
    )
    .expect_err("score failures should bubble out of knob tuning");

    assert_eq!(err, "probe failed");
}

#[test]
fn runtime_tuple_scoring_cache_reuses_existing_samples() {
    let config = dummy_config();
    let manifest = DataManifest {
        sources: vec![],
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: true,
    };
    let key = runtime_tuple_key(&config);
    let mut cache = BTreeMap::new();
    cache.insert(
        key,
        RuntimeTupleStats {
            count: 2,
            sum: 24.0,
        },
    );

    let score = score_runtime_tuple(&config, &manifest, &LibTorchDevice::Cpu, &mut cache)
        .expect("cached scores should be returned without measurement");

    assert!((score - 12.0).abs() < 1e-12);
    assert_eq!(
        cache.get(&key),
        Some(&RuntimeTupleStats {
            count: 2,
            sum: 24.0
        })
    );
}

#[test]
fn seed_runtime_score_cache_initializes_current_tuple_stats_for_exact_seed() {
    let config = dummy_config();
    let key = runtime_tuple_key(&config);
    let seed = runtime_seed(
        64,
        key,
        config.preflight.warmup_steps,
        config.preflight.measure_steps,
        RuntimeTupleStats {
            count: 2,
            sum: 24.0,
        },
    );
    let mut cache = BTreeMap::new();

    seed_runtime_score_cache(&config, &mut cache, Some(seed));

    assert_eq!(
        cache.get(&key),
        Some(&RuntimeTupleStats {
            count: 2,
            sum: 24.0
        })
    );
}

#[test]
fn seed_runtime_score_cache_ignores_mismatched_seed_cases() {
    let config = dummy_config();
    let key = runtime_tuple_key(&config);
    let mismatch_cases = [
        runtime_seed(
            32,
            key,
            config.preflight.warmup_steps,
            config.preflight.measure_steps,
            RuntimeTupleStats { count: 1, sum: 5.0 },
        ),
        runtime_seed(
            64,
            (16, 256, 32),
            config.preflight.warmup_steps,
            config.preflight.measure_steps,
            RuntimeTupleStats { count: 1, sum: 6.0 },
        ),
        runtime_seed(
            64,
            key,
            config.preflight.warmup_steps + 1,
            config.preflight.measure_steps,
            RuntimeTupleStats { count: 1, sum: 7.0 },
        ),
        runtime_seed(
            64,
            key,
            config.preflight.warmup_steps,
            config.preflight.measure_steps + 1,
            RuntimeTupleStats { count: 1, sum: 8.0 },
        ),
        runtime_seed(
            64,
            key,
            config.preflight.warmup_steps,
            config.preflight.measure_steps,
            RuntimeTupleStats::default(),
        ),
    ];

    for seed in mismatch_cases {
        let mut cache = BTreeMap::new();
        seed_runtime_score_cache(&config, &mut cache, Some(seed));
        assert!(cache.is_empty());
    }
}

#[test]
fn seed_runtime_score_cache_keeps_existing_entry_for_current_tuple() {
    let config = dummy_config();
    let key = runtime_tuple_key(&config);
    let mut cache = BTreeMap::from([(
        key,
        RuntimeTupleStats {
            count: 3,
            sum: 36.0,
        },
    )]);

    seed_runtime_score_cache(
        &config,
        &mut cache,
        Some(runtime_seed(
            64,
            key,
            config.preflight.warmup_steps,
            config.preflight.measure_steps,
            RuntimeTupleStats { count: 1, sum: 9.0 },
        )),
    );

    assert_eq!(
        cache.get(&key),
        Some(&RuntimeTupleStats {
            count: 3,
            sum: 36.0
        })
    );
}

#[test]
fn ranked_loader_runtime_from_score_cache_uses_all_measured_entries_not_stale_coarse_shortlist() {
    let base = dummy_config();
    let tuned = base.clone();
    let current_tuple = current_runtime_tuple(&tuned);
    let stronger_measured_tuple = (32, 512, 64);
    let score_cache = runtime_tuple_score_cache(&[
        (
            current_tuple,
            RuntimeTupleStats {
                count: 1,
                sum: 100.0,
            },
        ),
        (
            (16, 256, 32),
            RuntimeTupleStats {
                count: 1,
                sum: 99.0,
            },
        ),
        (
            stronger_measured_tuple,
            RuntimeTupleStats {
                count: 2,
                sum: 205.0,
            },
        ),
    ]);

    let ranked = ranked_loader_runtime_from_score_cache(&base, &tuned, &score_cache, 3);

    assert_eq!(
        ranked_runtime_tuples(&ranked),
        vec![stronger_measured_tuple, current_tuple, (16, 256, 32)]
    );
    assert!((ranked[0].train_samples_per_second - 102.5).abs() < 1e-12);
}

#[test]
fn ranked_loader_runtime_from_score_cache_prefers_stronger_final_mean_after_refine_samples() {
    let base = dummy_config();
    let tuned = base.clone();
    let coarse_leader = current_runtime_tuple(&tuned);
    let refined_winner = (16, 256, 32);
    let score_cache = runtime_tuple_score_cache(&[
        (
            coarse_leader,
            RuntimeTupleStats {
                count: 1,
                sum: 110.0,
            },
        ),
        (
            refined_winner,
            RuntimeTupleStats {
                count: 3,
                sum: 333.0,
            },
        ),
    ]);

    let ranked = ranked_loader_runtime_from_score_cache(&base, &tuned, &score_cache, 2);

    assert_eq!(
        ranked_runtime_tuples(&ranked),
        vec![refined_winner, coarse_leader]
    );
    assert!((ranked[0].train_samples_per_second - 111.0).abs() < 1e-12);
}

#[test]
fn seeded_current_tuple_preserves_ranked_shortlist_semantics() {
    let base = dummy_config();
    let tuned = base.clone();
    let current_tuple = current_runtime_tuple(&tuned);
    let stronger_tuple = (16, 256, 32);
    let third_tuple = (32, 512, 64);
    let mut seeded_cache = runtime_tuple_score_cache(&[
        (
            stronger_tuple,
            RuntimeTupleStats {
                count: 1,
                sum: 110.0,
            },
        ),
        (
            third_tuple,
            RuntimeTupleStats {
                count: 1,
                sum: 90.0,
            },
        ),
    ]);
    seed_runtime_score_cache(
        &tuned,
        &mut seeded_cache,
        Some(runtime_seed(
            64,
            current_tuple,
            tuned.preflight.warmup_steps,
            tuned.preflight.measure_steps,
            RuntimeTupleStats {
                count: 2,
                sum: 200.0,
            },
        )),
    );
    let seeded_ranked = ranked_loader_runtime_from_score_cache(&base, &tuned, &seeded_cache, 3);
    let explicit_ranked = ranked_loader_runtime_from_score_cache(
        &base,
        &tuned,
        &runtime_tuple_score_cache(&[
            (
                current_tuple,
                RuntimeTupleStats {
                    count: 2,
                    sum: 200.0,
                },
            ),
            (
                stronger_tuple,
                RuntimeTupleStats {
                    count: 1,
                    sum: 110.0,
                },
            ),
            (
                third_tuple,
                RuntimeTupleStats {
                    count: 1,
                    sum: 90.0,
                },
            ),
        ]),
        3,
    );

    assert_eq!(seeded_ranked, explicit_ranked);
    assert_eq!(
        ranked_runtime_tuples(&seeded_ranked),
        vec![stronger_tuple, current_tuple, third_tuple]
    );
}

#[test]
fn ranked_loader_runtime_from_score_cache_dedups_current_tuple_and_respects_limit() {
    let base = dummy_config();
    let mut tuned = base.clone();
    apply_runtime_tuple(&mut tuned, (16, 256, 32));
    let current_tuple = current_runtime_tuple(&tuned);
    let score_cache = runtime_tuple_score_cache(&[
        (
            current_tuple,
            RuntimeTupleStats {
                count: 2,
                sum: 250.0,
            },
        ),
        (
            (8, 128, 16),
            RuntimeTupleStats {
                count: 1,
                sum: 120.0,
            },
        ),
        (
            (32, 512, 64),
            RuntimeTupleStats {
                count: 1,
                sum: 119.0,
            },
        ),
    ]);

    let ranked = ranked_loader_runtime_from_score_cache(&base, &tuned, &score_cache, 2);

    assert_eq!(ranked.len(), 2);
    assert_eq!(
        ranked_runtime_tuples(&ranked),
        vec![current_tuple, (8, 128, 16)]
    );
    assert_eq!(
        ranked
            .iter()
            .filter(|entry| entry.tuple == current_tuple)
            .count(),
        1
    );
    assert_eq!(ranked[0].loader, loader_runtime_config(&tuned));
}

#[test]
fn ranked_loader_runtime_from_score_cache_uses_deterministic_tuple_order_for_equal_scores() {
    let base = dummy_config();
    let tuned = base.clone();
    let score_cache = runtime_tuple_score_cache(&[
        (
            (32, 512, 64),
            RuntimeTupleStats {
                count: 2,
                sum: 200.0,
            },
        ),
        (
            (8, 128, 16),
            RuntimeTupleStats {
                count: 1,
                sum: 100.0,
            },
        ),
        (
            (16, 256, 32),
            RuntimeTupleStats {
                count: 3,
                sum: 300.0,
            },
        ),
    ]);

    let ranked = ranked_loader_runtime_from_score_cache(&base, &tuned, &score_cache, 3);

    assert_eq!(
        ranked_runtime_tuples(&ranked),
        vec![(8, 128, 16), (16, 256, 32), (32, 512, 64)]
    );
}

#[test]
fn push_runtime_tuple_sample_preserves_existing_cache_when_measurement_fails() {
    let config = dummy_config();
    let manifest = DataManifest {
        sources: vec![],
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: true,
    };
    let key = runtime_tuple_key(&config);
    let mut cache = BTreeMap::new();
    cache.insert(key, RuntimeTupleStats { count: 1, sum: 6.0 });

    let averaged = push_runtime_tuple_sample(&config, &manifest, &LibTorchDevice::Cpu, &mut cache)
        .expect_err("empty manifests still fail before adding a new sample");

    assert_eq!(averaged, "not enough train data to finish runtime probe");
    assert_eq!(
        cache.get(&key),
        Some(&RuntimeTupleStats { count: 1, sum: 6.0 })
    );
}

#[test]
fn finalize_runtime_probe_throughput_handles_present_measure_start_with_zero_samples() {
    let throughput = finalize_runtime_probe_throughput(Some(Instant::now()), 0);

    assert_eq!(throughput, 0.0);
}

#[test]
fn score_runtime_tuple_treats_empty_cached_samples_as_cache_miss() {
    let config = dummy_config();
    let manifest = DataManifest {
        sources: vec![],
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: true,
    };
    let key = runtime_tuple_key(&config);
    let mut cache = BTreeMap::new();
    cache.insert(key, RuntimeTupleStats::default());

    let err = score_runtime_tuple(&config, &manifest, &LibTorchDevice::Cpu, &mut cache)
        .expect_err("empty cached samples should still trigger measurement");

    assert_eq!(err, "not enough train data to finish runtime probe");
    assert_eq!(cache.get(&key), Some(&RuntimeTupleStats::default()));
}

#[test]
fn runtime_refine_helpers_apply_minimum_budget_and_format_summary() {
    assert_eq!(runtime_refine_sample_budget(0), 1);
    assert_eq!(runtime_refine_sample_budget(3), 3);

    let summary = format_runtime_refine_summary(&[(8, 128, 16), (16, 256, 32)], 0);

    assert!(summary.contains("Runtime refine:"));
    assert!(summary.contains("close_tuples=[(8, 128, 16), (16, 256, 32)]"));
    assert!(summary.contains("extra_samples=1"));
}

#[test]
fn runtime_refine_top_up_plan_computes_missing_sample_math_for_zero_partial_and_saturated_counts() {
    assert_eq!(
        runtime_refine_top_up_plan(RuntimeTupleStats::default(), 2),
        RuntimeRefineTopUpPlan {
            target_total_count: 3,
            missing_samples: 3,
        }
    );
    assert_eq!(
        runtime_refine_top_up_plan(
            RuntimeTupleStats {
                count: 2,
                sum: 40.0,
            },
            2,
        ),
        RuntimeRefineTopUpPlan {
            target_total_count: 3,
            missing_samples: 1,
        }
    );
    assert_eq!(
        runtime_refine_top_up_plan(
            RuntimeTupleStats {
                count: 3,
                sum: 60.0,
            },
            2,
        ),
        RuntimeRefineTopUpPlan {
            target_total_count: 3,
            missing_samples: 0,
        }
    );
    assert_eq!(
        runtime_refine_top_up_plan(
            RuntimeTupleStats {
                count: 5,
                sum: 100.0,
            },
            2,
        ),
        RuntimeRefineTopUpPlan {
            target_total_count: 3,
            missing_samples: 0,
        }
    );
}

#[test]
fn seeded_current_tuple_with_count_already_meeting_target_skips_refine_top_up() {
    let tuned = dummy_config();
    let current_tuple = current_runtime_tuple(&tuned);
    let other_tuple = (16, 256, 32);
    let close_tuples = vec![current_tuple, other_tuple];
    let mut score_cache = runtime_refine_cache_entries(&[
        (
            current_tuple,
            RuntimeTupleStats {
                count: 3,
                sum: 300.0,
            },
        ),
        (
            other_tuple,
            RuntimeTupleStats {
                count: 1,
                sum: 95.0,
            },
        ),
    ]);
    let mut seen = Vec::new();
    let mut best_score = 100.0;
    let mut best_tuple = current_tuple;

    refine_close_runtime_tuples(
        &tuned,
        &close_tuples,
        2,
        &mut score_cache,
        &mut best_score,
        &mut best_tuple,
        |candidate, cache| {
            let tuple = runtime_tuple_key(candidate);
            seen.push(tuple);
            let stats = cache.entry(tuple).or_default();
            *stats = stats.push(120.0 + seen.len() as f64);
            Ok(stats.mean())
        },
    )
    .expect("refine top-up should succeed");

    assert_eq!(seen, vec![other_tuple, other_tuple]);
    assert_eq!(
        score_cache.get(&current_tuple).map(|stats| stats.count),
        Some(3)
    );
    assert_eq!(
        score_cache.get(&other_tuple).map(|stats| stats.count),
        Some(3)
    );
}

#[test]
fn seeded_current_tuple_with_partial_count_only_tops_up_missing_amount() {
    let tuned = dummy_config();
    let current_tuple = current_runtime_tuple(&tuned);
    let other_tuple = (16, 256, 32);
    let close_tuples = vec![current_tuple, other_tuple];
    let mut score_cache = runtime_refine_cache_entries(&[
        (
            current_tuple,
            RuntimeTupleStats {
                count: 2,
                sum: 200.0,
            },
        ),
        (
            other_tuple,
            RuntimeTupleStats {
                count: 1,
                sum: 95.0,
            },
        ),
    ]);
    let mut seen = Vec::new();
    let mut best_score = 100.0;
    let mut best_tuple = current_tuple;

    refine_close_runtime_tuples(
        &tuned,
        &close_tuples,
        2,
        &mut score_cache,
        &mut best_score,
        &mut best_tuple,
        |candidate, cache| {
            let tuple = runtime_tuple_key(candidate);
            seen.push(tuple);
            let stats = cache.entry(tuple).or_default();
            *stats = stats.push(120.0 + seen.len() as f64);
            Ok(stats.mean())
        },
    )
    .expect("refine top-up should succeed");

    assert_eq!(seen, vec![current_tuple, other_tuple, other_tuple]);
    assert_eq!(
        score_cache.get(&current_tuple).map(|stats| stats.count),
        Some(3)
    );
    assert_eq!(
        score_cache.get(&other_tuple).map(|stats| stats.count),
        Some(3)
    );
}

#[test]
fn ranked_shortlist_semantics_match_equivalent_explicit_cache_contents() {
    let base = dummy_config();
    let tuned = base.clone();
    let current_tuple = current_runtime_tuple(&tuned);
    let stronger_tuple = (16, 256, 32);
    let third_tuple = (32, 512, 64);
    let seeded_cache = runtime_tuple_score_cache(&[
        (
            current_tuple,
            RuntimeTupleStats {
                count: 3,
                sum: 300.0,
            },
        ),
        (
            stronger_tuple,
            RuntimeTupleStats {
                count: 3,
                sum: 333.0,
            },
        ),
        (
            third_tuple,
            RuntimeTupleStats {
                count: 1,
                sum: 90.0,
            },
        ),
    ]);
    let explicit_cache = runtime_tuple_score_cache(&[
        (
            current_tuple,
            RuntimeTupleStats {
                count: 3,
                sum: 300.0,
            },
        ),
        (
            stronger_tuple,
            RuntimeTupleStats {
                count: 3,
                sum: 333.0,
            },
        ),
        (
            third_tuple,
            RuntimeTupleStats {
                count: 1,
                sum: 90.0,
            },
        ),
    ]);

    let seeded_ranked = ranked_loader_runtime_from_score_cache(&base, &tuned, &seeded_cache, 3);
    let explicit_ranked = ranked_loader_runtime_from_score_cache(&base, &tuned, &explicit_cache, 3);

    assert_eq!(seeded_ranked, explicit_ranked);
}

#[test]
fn format_runtime_refine_summary_preserves_nonzero_budget_without_clamping() {
    let summary = format_runtime_refine_summary(&[(32, 512, 64)], 3);

    assert!(summary.contains("Runtime refine:"));
    assert!(summary.contains("close_tuples=[(32, 512, 64)]"));
    assert!(summary.contains("extra_samples=3"));
}

#[test]
fn measure_train_runtime_throughput_fails_fast_without_train_data() {
    let config = dummy_config();
    let manifest = DataManifest {
        sources: vec![],
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: true,
    };
    let loader = runtime_probe_loader_config(&config);

    let err = measure_train_runtime_throughput(&config, &loader, &manifest, &LibTorchDevice::Cpu)
        .expect_err("empty manifests should fail before measuring throughput");

    assert_eq!(err, "not enough train data to finish runtime probe");
}

#[test]
fn runtime_probe_success_paths_measure_and_cache_real_train_data_variants() {
    let (manifest, path) = single_loose_train_manifest("success-paths");

    assert_runtime_probe_success_paths_measure_and_cache_real_train_data_case(
        &manifest,
        crate::config::PrecisionMode::Fp32,
    );
    assert_runtime_probe_success_paths_measure_and_cache_real_train_data_case(
        &manifest,
        crate::config::PrecisionMode::Bf16Autocast,
    );

    fs::remove_file(path).ok();
}

fn assert_runtime_probe_success_paths_measure_and_cache_real_train_data_case(
    manifest: &DataManifest,
    precision_mode: crate::config::PrecisionMode,
) {
    let mut config = dummy_config();
    config.batch_size = 1;
    config.microbatch_size = Some(1);
    config.augment = false;
    config.train_fraction = 1.0;
    config.buffer_games = 1;
    config.buffer_samples = 1;
    config.archive_queue_bound = 1;
    config.preflight.warmup_steps = 1;
    config.preflight.measure_steps = 1;
    config.precision_mode = precision_mode;

    let key = runtime_tuple_key(&config);
    let mut cache = BTreeMap::new();
    let measured_score = score_runtime_tuple(&config, manifest, &LibTorchDevice::Cpu, &mut cache)
        .expect("uncached runtime tuple should run the real training probe and insert a score");
    assert!(measured_score.is_finite());
    assert!(measured_score >= 0.0);
    assert_eq!(cache.get(&key).map(|stats| stats.count), Some(1));

    let cached_score = score_runtime_tuple(&config, manifest, &LibTorchDevice::Cpu, &mut cache)
        .expect("cached runtime tuple should reuse the inserted sample");
    assert_eq!(cached_score, measured_score);
    assert_eq!(cache.get(&key).map(|stats| stats.count), Some(1));
}

#[test]
fn push_runtime_tuple_sample_bubbles_measurement_errors() {
    let config = dummy_config();
    let manifest = DataManifest {
        sources: vec![],
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: true,
    };
    let mut cache = BTreeMap::new();

    let err = push_runtime_tuple_sample(&config, &manifest, &LibTorchDevice::Cpu, &mut cache)
        .expect_err("empty manifests should fail before storing tuple samples");

    assert_eq!(err, "not enough train data to finish runtime probe");
    assert!(!cache.contains_key(&runtime_tuple_key(&config)));
}

#[test]
fn score_runtime_tuple_bubbles_measurement_errors_when_cache_is_empty() {
    let config = dummy_config();
    let manifest = DataManifest {
        sources: vec![],
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: true,
    };
    let mut cache = BTreeMap::new();

    let err = score_runtime_tuple(&config, &manifest, &LibTorchDevice::Cpu, &mut cache)
        .expect_err("empty manifests should fail when tuple scores are uncached");

    assert_eq!(err, "not enough train data to finish runtime probe");
    assert!(cache.is_empty());
}

#[test]
fn autotune_loader_runtime_bubbles_empty_manifest_probe_failures() {
    let config = dummy_config();
    let manifest = DataManifest {
        sources: vec![],
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: true,
    };

    let err = autotune_loader_runtime(&config, &manifest, &LibTorchDevice::Cpu)
        .expect_err("empty manifests should fail during runtime autotune scoring");

    assert_eq!(err, "not enough train data to finish runtime probe");
}

#[test]
fn autotune_loader_runtime_returns_current_loader_when_rounds_and_extra_samples_are_disabled() {
    let mut config = dummy_config();
    config.preflight.loader_runtime_rounds = 0;
    config.preflight.loader_tuple_extra_samples = 0;
    config.num_threads = Some(1);
    let manifest = DataManifest {
        sources: vec![],
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: true,
    };

    let loader = autotune_loader_runtime(&config, &manifest, &LibTorchDevice::Cpu)
        .expect("disabled loader autotune should return the current loader without probing");

    assert_eq!(loader, loader_runtime_config(&config));
}

#[test]
fn score_runtime_tuple_returns_cached_single_sample_without_touching_measurement() {
    let config = dummy_config();
    let manifest = DataManifest {
        sources: vec![],
        total_games: 0,
        train_count: 0,
        val_count: 0,
        counts_exact: true,
    };
    let key = runtime_tuple_key(&config);
    let mut cache = BTreeMap::new();
    cache.insert(key, RuntimeTupleStats { count: 1, sum: 7.5 });

    let score = score_runtime_tuple(&config, &manifest, &LibTorchDevice::Cpu, &mut cache)
        .expect("single cached score should bypass runtime measurement");

    assert!((score - 7.5).abs() < 1e-12);
    assert_eq!(
        cache.get(&key),
        Some(&RuntimeTupleStats { count: 1, sum: 7.5 })
    );
}

#[test]
fn close_runtime_tuples_keeps_scores_exactly_on_threshold_boundary() {
    let scores = vec![
        ((8, 128, 16), 100.0),
        ((16, 256, 32), 90.0),
        ((32, 512, 64), 89.99),
    ];

    assert_eq!(
        close_runtime_tuples(&scores, 100.0, 0.10, 5),
        vec![(8, 128, 16), (16, 256, 32)]
    );
}

#[test]
fn close_runtime_tuples_with_negative_best_score_and_positive_margin_can_drop_everything() {
    let scores = vec![((8, 128, 16), -9.0), ((16, 256, 32), -10.0)];

    assert!(close_runtime_tuples(&scores, -9.0, 0.10, 5).is_empty());
}

#[test]
fn close_runtime_tuples_uses_inclusive_threshold_boundary() {
    let scores = vec![
        ((8, 128, 16), 100.0),
        ((16, 256, 32), 75.0),
        ((32, 512, 64), 74.99),
    ];

    assert_eq!(
        close_runtime_tuples(&scores, 100.0, 0.25, 5),
        vec![(8, 128, 16), (16, 256, 32)]
    );
}

#[test]
fn should_count_measured_samples_is_false_on_exact_warmup_boundary() {
    assert!(!should_count_measured_samples(5, 5));
    assert!(should_count_measured_samples(6, 5));
}
