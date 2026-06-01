use burn::prelude::*;
use hydra_train_types::config::ModelShapeConfig;

use crate::backbone::SEResNetConfig;
use crate::heads::*;

use super::{HydraModel, HydraModelInit};

impl HydraModelInit for ModelShapeConfig {
    fn init<B: Backend>(
        &self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> HydraModel<B> {
        self.validate().expect("invalid Hydra model shape config");
        let backbone_cfg = SEResNetConfig::new(
            self.num_blocks,
            self.input_channels,
            self.hidden_channels,
            self.num_groups,
            self.se_bottleneck,
        )
        .with_activation(self.backbone_activation)
        .with_se_every_n(self.backbone_se_every_n)
        .with_norm(self.backbone_norm);
        let heads_cfg = HeadsConfig::new()
            .with_hidden_channels(self.hidden_channels)
            .with_action_space(self.action_space)
            .with_score_bins(self.score_bins)
            .with_num_opponents(self.num_opponents)
            .with_grp_classes(self.grp_classes)
            .with_num_belief_components(self.num_belief_components)
            .with_opponent_hand_type_classes(self.opponent_hand_type_classes);
        HydraModel {
            backbone: backbone_cfg.init(device),
            policy: heads_cfg.init_policy(device),
            value: heads_cfg.init_value(device),
            score_pdf: heads_cfg.init_score_pdf(device),
            score_cdf: heads_cfg.init_score_cdf(device),
            opp_tenpai: heads_cfg.init_opp_tenpai(device),
            grp: heads_cfg.init_grp(device),
            opp_next_discard: heads_cfg.init_opp_next_discard(device),
            danger: heads_cfg.init_danger(device),
            oracle_critic: heads_cfg.init_oracle_critic(device),
            belief_field: heads_cfg.init_belief_field(device),
            mixture_weight: heads_cfg.init_mixture_weight(device),
            opponent_hand_type: heads_cfg.init_opponent_hand_type(device),
            delta_q: heads_cfg.init_delta_q(device),
            safety_residual: heads_cfg.init_safety_residual(device),
        }
    }
}
