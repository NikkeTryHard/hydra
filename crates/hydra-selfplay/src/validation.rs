//! Self-play-backed validation entry points for search-label reports.
//!
//! Search-label evaluators stay in `hydra_search_labels`; this module owns the
//! runtime bridge that produces live-labeled self-play trajectories.

use burn::prelude::Backend;
use hydra_model::model::HydraModel;
use hydra_search_labels::delta_q_validation::{
    DeltaQValidationReport, run_delta_q_validation_over_trajectories,
};
use hydra_search_labels::exit::ExitConfig;
use hydra_search_labels::exit_validation::{
    ExitValidationReport, run_exit_validation_over_trajectories,
};
use hydra_search_labels::live_exit::LiveExitConfig;

use crate::generate_self_play_batch_source_cooperative;

/// Runs shadow ExIt validation over self-play games.
///
/// The live producer is force-enabled for data collection, but the produced
/// labels are only inspected on the returned trajectories and are not used for
/// training. This keeps the harness fully observational while reusing the
/// existing self-play infrastructure.
pub fn run_exit_validation<B: Backend>(
    game_seeds: &[u64],
    temperature: f32,
    rng_seed: u64,
    model: &HydraModel<B>,
    device: &B::Device,
    exit_config: ExitConfig,
) -> ExitValidationReport {
    let source = generate_self_play_batch_source_cooperative(
        game_seeds,
        temperature,
        rng_seed,
        model,
        device,
        LiveExitConfig {
            enabled: true,
            exit_config: exit_config.clone(),
        },
    );

    run_exit_validation_over_trajectories(&source.trajectories, model, device, &exit_config)
}

/// Runs an observational delta-q validation pass over self-play trajectories.
pub fn run_delta_q_validation<B: Backend>(
    game_seeds: &[u64],
    temperature: f32,
    rng_seed: u64,
    model: &HydraModel<B>,
    device: &B::Device,
    exit_config: ExitConfig,
) -> DeltaQValidationReport {
    let source = generate_self_play_batch_source_cooperative(
        game_seeds,
        temperature,
        rng_seed,
        model,
        device,
        LiveExitConfig {
            enabled: true,
            exit_config: exit_config.clone(),
        },
    );

    run_delta_q_validation_over_trajectories(&source.trajectories, model, device, &exit_config)
}

#[cfg(test)]
mod tests {
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

        let exit_report =
            run_exit_validation(&seeds, 1.0, 123, &model, &device, exit_config.clone());
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
}
