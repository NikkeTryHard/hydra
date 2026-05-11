use super::*;
use burn::backend::NdArray;
use hydra_model::model::{HydraModelConfig, HydraModelInit};

type B = NdArray<f32>;

fn small_test_model_config() -> HydraModelConfig {
    HydraModelConfig::new(1)
        .with_hidden_channels(2)
        .with_se_bottleneck(1)
        .with_num_groups(1)
}

#[test]
fn selfplay_validation_entry_points_run() {
    let device = Default::default();
    let model = small_test_model_config().init::<B>(&device);
    let seeds = [42u64];
    let mut exit_config = ExitConfig::new();
    exit_config.min_visits = 1;

    let exit_report = run_exit_validation(&seeds, 1.0, 123, &model, &device, exit_config.clone());
    let delta_q_report = run_delta_q_validation(&seeds, 1.0, 123, &model, &device, exit_config);

    assert!(exit_report.total_states > 0);
    assert_eq!(
        exit_report.total_states,
        exit_report.labels_emitted + exit_report.labels_rejected
    );
    assert!(delta_q_report.total_states > 0);
    assert_eq!(
        delta_q_report.total_states,
        delta_q_report.labels_emitted + delta_q_report.labels_rejected
    );
}
