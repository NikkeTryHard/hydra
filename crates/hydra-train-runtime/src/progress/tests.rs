use serde_json::Value;

use super::{
    BannerStats, BatchStats, EpochLogEntry, RareActionMetrics, RlStepLogEntry, ScalarAverages,
    StepLogEntry,
};
use crate::preflight::ProfilingEnvelope;

fn batch(sample_count: usize, total_loss: f64, agreement: f64) -> BatchStats {
    BatchStats {
        sample_count,
        batch_count: 1,
        total_loss,
        policy_agreement: agreement,
        ..Default::default()
    }
}

#[test]
fn scalar_averages_are_sample_weighted() {
    let mut stats = ScalarAverages::default();
    stats.record_batch(batch(4, 1.0, 0.25));
    stats.record_batch(batch(1, 4.0, 1.0));

    let stats = stats.finalize();
    assert_eq!(stats.num_batches, 2);
    assert_eq!(stats.num_samples, 5);
    assert!((stats.total_loss - 1.6).abs() < 1e-12);
    assert!((stats.policy_agreement - 0.4).abs() < 1e-12);
}

#[test]
fn zero_weight_batches_do_not_change_averages() {
    let mut stats = ScalarAverages::default();
    stats.record_batch(batch(0, 7.0, 0.9));
    stats.record_batch(batch(2, 3.0, 0.5));

    let stats = stats.finalize();
    assert_eq!(stats.num_batches, 1);
    assert_eq!(stats.num_samples, 2);
    assert!((stats.total_loss - 3.0).abs() < 1e-12);
    assert!((stats.policy_agreement - 0.5).abs() < 1e-12);
}

#[test]
fn finalize_leaves_empty_accumulator_unchanged() {
    let stats = ScalarAverages::default().finalize();
    assert_eq!(stats.num_batches, 0);
    assert_eq!(stats.num_samples, 0);
    assert_eq!(stats.total_loss, 0.0);
    assert_eq!(stats.loss_score_cdf, 0.0);
}

#[test]
fn record_batch_weights_all_loss_fields() {
    let mut stats = ScalarAverages::default();
    stats.record_batch(BatchStats {
        sample_count: 2,
        batch_count: 1,
        total_loss: 5.0,
        policy_agreement: 0.25,
        loss_policy: 1.0,
        loss_value: 2.0,
        loss_grp: 3.0,
        loss_tenpai: 4.0,
        loss_danger: 5.0,
        loss_opp_next: 6.0,
        loss_score_pdf: 7.0,
        loss_score_cdf: 8.0,
        rare_actions: RareActionMetrics::default(),
    });

    let stats = stats.finalize();
    assert_eq!(stats.num_batches, 1);
    assert_eq!(stats.num_samples, 2);
    assert_eq!(stats.loss_policy, 1.0);
    assert_eq!(stats.loss_value, 2.0);
    assert_eq!(stats.loss_grp, 3.0);
    assert_eq!(stats.loss_tenpai, 4.0);
    assert_eq!(stats.loss_danger, 5.0);
    assert_eq!(stats.loss_opp_next, 6.0);
    assert_eq!(stats.loss_score_pdf, 7.0);
    assert_eq!(stats.loss_score_cdf, 8.0);
}

#[test]
fn log_entries_and_banner_stats_cover_data_fields() {
    let epoch = EpochLogEntry::<(), String> {
        epoch: 2,
        global_step: 42,
        lr: 1e-3,
        train_total_loss: 1.0,
        train_policy_agreement: 0.5,
        train_loss_policy: 0.1,
        train_loss_value: 0.2,
        train_loss_grp: 0.3,
        train_loss_tenpai: 0.4,
        train_loss_danger: 0.5,
        train_loss_opp_next: 0.6,
        train_loss_score_pdf: 0.7,
        train_loss_score_cdf: 0.8,
        train_rare_actions: RareActionMetrics::default(),
        val_total_loss: Some(1.5),
        val_policy_loss: Some(1.25),
        val_policy_agreement: Some(0.75),
        val_delta_q_promotion: None,
        val_rare_actions: None,
        profiling: Some(ProfilingEnvelope::leaf("bc_epoch", 1.25)),
        advisories: Vec::new(),
        best_val_policy_loss: Some(1.0),
        best_val_agreement: Some(0.8),
        num_batches: 3,
    };
    let step = StepLogEntry::<(), String> {
        global_step: 42,
        epoch: 2,
        lr: 1e-3,
        train_total_loss: 1.0,
        train_policy_agreement: 0.5,
        train_loss_policy: 0.1,
        train_loss_value: 0.2,
        train_loss_grp: 0.3,
        train_loss_tenpai: 0.4,
        train_loss_danger: 0.5,
        train_loss_opp_next: 0.6,
        train_loss_score_pdf: 0.7,
        train_loss_score_cdf: 0.8,
        train_rare_actions: RareActionMetrics::default(),
        val_total_loss: Some(1.5),
        val_policy_loss: Some(1.25),
        val_policy_agreement: Some(0.75),
        val_delta_q_promotion: None,
        window_steps: 2,
        window_samples: 128,
        steps_per_second: 4.0,
        samples_per_second: 256.0,
        val_rare_actions: None,
        profiling: Some(ProfilingEnvelope::leaf("bc_interval", 0.5)),
        advisories: Vec::new(),
        best_val_policy_loss: Some(1.0),
        best_val_agreement: Some(0.8),
    };
    let rl = RlStepLogEntry::<String> {
        global_step: 5,
        phase: "ExitPondering".to_string(),
        loss: 0.25,
        effective_lr: 5e-4,
        exit_weight: 0.1,
        games_per_batch: 8,
        samples_in_batch: 64,
        total_games: 128,
        total_samples: 1024,
        delta_q_state: "Warmup".to_string(),
        profiling: Some(ProfilingEnvelope::leaf("rl_step", 0.75)),
        advisories: Vec::new(),
    };
    let banner = BannerStats {
        total_sources: 2,
        total_games: 30,
        train_count: 24,
        val_count: 6,
        accum_steps: 4,
        counts_exact: true,
    };

    let epoch_json = serde_json::to_value(epoch).expect("epoch log should serialize");
    let step_json = serde_json::to_value(step).expect("step log should serialize");
    let rl_json = serde_json::to_value(rl).expect("rl log should serialize");

    assert_eq!(epoch_json["global_step"], Value::from(42));
    assert_eq!(epoch_json["num_batches"], Value::from(3));
    assert_eq!(epoch_json["profiling"]["stage"], Value::from("bc_epoch"));
    assert_eq!(step_json["epoch"], Value::from(2));
    assert_eq!(step_json["profiling"]["stage"], Value::from("bc_interval"));
    assert_eq!(rl_json["phase"], Value::from("ExitPondering"));
    assert_eq!(rl_json["total_samples"], Value::from(1024));
    assert_eq!(rl_json["profiling"]["stage"], Value::from("rl_step"));

    assert_eq!(banner.total_sources, 2);
    assert_eq!(banner.total_games, 30);
    assert_eq!(banner.train_count, 24);
    assert_eq!(banner.val_count, 6);
    assert_eq!(banner.accum_steps, 4);
    assert!(banner.counts_exact);
}
