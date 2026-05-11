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
mod tests;
