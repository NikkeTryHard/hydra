//! Backward-compatible ExIt validation exports.
//!
//! The canonical report/evaluator implementation lives in
//! `hydra_search_labels::exit_validation`. Runtime self-play entry points stay
//! in this module because they depend on `hydra-train` self-play orchestration.

use burn::prelude::Backend;
use hydra_search_labels::exit::ExitConfig;
use hydra_search_labels::live_exit::LiveExitConfig;

use hydra_model::model::HydraModel;
use hydra_selfplay::generate_self_play_batch_source_cooperative;

pub use hydra_search_labels::exit_validation::*;

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
