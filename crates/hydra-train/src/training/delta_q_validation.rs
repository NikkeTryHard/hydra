//! Backward-compatible delta-q validation exports.
//!
//! The canonical report/evaluator implementation lives in
//! `hydra_search_labels::delta_q_validation`. Runtime self-play entry points
//! stay in this module because they depend on `hydra-train` self-play orchestration.

use burn::prelude::Backend;
use hydra_search_labels::exit::ExitConfig;
use hydra_search_labels::live_exit::LiveExitConfig;

use crate::model::HydraModel;
use crate::selfplay::generate_self_play_batch_source_cooperative;

pub use hydra_search_labels::delta_q_validation::*;

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
