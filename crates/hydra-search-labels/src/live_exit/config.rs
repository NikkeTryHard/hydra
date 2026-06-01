use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;
use hydra_core::safety::SafetyInfo;
use riichienv_core::observation::Observation;
use riichienv_core::state::GameState;

use crate::exit::ExitConfig;

use super::StepRecord;
use super::TrajectorySearchLabels;
use super::adapter::SelfPlayExitAdapter;
use super::producer::try_live_search_labels;

/// Configuration for the live ExIt producer.
///
/// Wraps the standard [`ExitConfig`] with a feature gate.  The producer
/// is default-on after the infrastructure validation matrix cleared it.
/// Set `enabled = false` explicitly to disable label generation.
#[derive(Debug, Clone)]
pub struct LiveExitConfig {
    /// Whether the live producer is enabled.  Default: `true`.
    pub enabled: bool,
    /// The underlying ExIt gate configuration.
    pub exit_config: ExitConfig,
}

impl Default for LiveExitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exit_config: ExitConfig::default_live_exit(),
        }
    }
}

/// Creates an exit label closure wired with a [`SelfPlayExitAdapter`] for
/// use with [`run_self_play_game_with_exit_labels`].
///
/// When `cfg.enabled` is false, the returned closure always emits `None`.
pub fn make_live_exit_fn<M>(
    cfg: LiveExitConfig,
    mut model_pv: M,
) -> impl FnMut(&GameState, &Observation, &StepRecord, &SafetyInfo, u32) -> Option<TrajectorySearchLabels>
where
    M: FnMut(&[f32; OBS_SIZE]) -> ([f32; HYDRA_ACTION_SPACE], f32),
{
    let mut adapter = SelfPlayExitAdapter::new();
    let exit_config = cfg.exit_config;
    let enabled = cfg.enabled;

    move |state, obs, step, safety, _turn| {
        if !enabled {
            return None;
        }
        try_live_search_labels(
            state,
            obs,
            step,
            safety,
            &exit_config,
            &mut model_pv,
            &mut adapter,
        )
    }
}
