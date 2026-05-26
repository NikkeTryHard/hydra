//! Campaign/run artifact layout for Python learner launches.

use hydra_train_runtime::config::{
    DEFAULT_BC_STAGE, DEFAULT_PPO_STAGE, PythonLearnerCliOptions, PythonPpoControlCliOptions,
};

pub(crate) use crate::campaign_layout::CampaignRunLayout;

pub(crate) fn for_bc(options: &PythonLearnerCliOptions) -> CampaignRunLayout {
    CampaignRunLayout::new(
        &options.output_dir,
        options.stage.as_deref(),
        options.run_name.as_deref(),
        DEFAULT_BC_STAGE,
    )
}

pub(crate) fn for_ppo(options: &PythonPpoControlCliOptions) -> CampaignRunLayout {
    CampaignRunLayout::new(
        &options.output_dir,
        options.stage.as_deref(),
        options.run_name.as_deref(),
        DEFAULT_PPO_STAGE,
    )
}
