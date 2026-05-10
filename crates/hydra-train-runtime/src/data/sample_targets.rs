use burn::prelude::*;
use hydra_core::action::HYDRA_ACTION_SPACE;

use crate::data::sample::{MjaiBatch, MjaiBcBatch, one_hot_action};
use hydra_train_types::losses::HydraTargets;

fn policy_target_from_actions<B: Backend>(
    actions: Tensor<B, 1, Int>,
    batch_size: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let actions = actions.into_data().convert::<i64>();
    let actions = actions
        .as_slice::<i64>()
        .expect("policy actions should be readable as i64");
    let flat: Vec<f32> = actions
        .iter()
        .flat_map(|&action| one_hot_action(action as u8, HYDRA_ACTION_SPACE))
        .collect();
    Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([batch_size, HYDRA_ACTION_SPACE])
}

pub(crate) fn into_bc_batch_and_hydra_targets_inner<B: Backend>(
    batch: MjaiBatch<B>,
) -> (Tensor<B, 3>, MjaiBcBatch<B>, HydraTargets<B>) {
    let MjaiBatch {
        obs,
        actions,
        legal_mask,
        value_target,
        grp_target,
        oracle_target,
        oracle_target_mask,
        tenpai_target,
        danger_target,
        danger_mask,
        safety_residual_target,
        safety_residual_mask,
        exit_target,
        exit_mask,
        delta_q_target,
        delta_q_mask,
        belief_fields_target,
        mixture_weight_target,
        belief_fields_mask,
        mixture_weight_mask,
        opp_next_target,
        score_pdf_target,
        score_cdf_target,
        target_presence,
    } = batch;
    let batch_size = actions.dims()[0];
    let policy_target = policy_target_from_actions(actions.clone(), batch_size, &obs.device());

    (
        obs,
        MjaiBcBatch {
            actions,
            exit_target,
            exit_mask,
        },
        HydraTargets {
            policy_target,
            legal_mask,
            value_target,
            grp_target,
            tenpai_target,
            danger_target,
            danger_mask,
            safety_residual_target,
            opp_next_target,
            score_pdf_target,
            score_cdf_target,
            oracle_target,
            belief_fields_target,
            mixture_weight_target,
            opponent_hand_type_target: None,
            delta_q_target,
            delta_q_mask,
            safety_residual_mask,
            belief_fields_mask,
            mixture_weight_mask,
            oracle_guidance_mask: Some(oracle_target_mask),
            target_presence,
        },
    )
}

pub(crate) fn into_hydra_targets_inner<B: Backend>(batch: MjaiBatch<B>) -> HydraTargets<B> {
    let (_, _, targets) = into_bc_batch_and_hydra_targets_inner(batch);
    targets
}

pub(crate) fn cloned_hydra_targets<B: Backend>(batch: &MjaiBatch<B>) -> HydraTargets<B> {
    let batch_size = batch.actions.dims()[0];
    let policy_target =
        policy_target_from_actions(batch.actions.clone(), batch_size, &batch.obs.device());

    HydraTargets {
        policy_target,
        legal_mask: batch.legal_mask.clone(),
        value_target: batch.value_target.clone(),
        grp_target: batch.grp_target.clone(),
        tenpai_target: batch.tenpai_target.clone(),
        danger_target: batch.danger_target.clone(),
        danger_mask: batch.danger_mask.clone(),
        safety_residual_target: batch.safety_residual_target.clone(),
        opp_next_target: batch.opp_next_target.clone(),
        score_pdf_target: batch.score_pdf_target.clone(),
        score_cdf_target: batch.score_cdf_target.clone(),
        oracle_target: batch.oracle_target.clone(),
        belief_fields_target: batch.belief_fields_target.clone(),
        mixture_weight_target: batch.mixture_weight_target.clone(),
        opponent_hand_type_target: None,
        delta_q_target: batch.delta_q_target.clone(),
        delta_q_mask: batch.delta_q_mask.clone(),
        safety_residual_mask: batch.safety_residual_mask.clone(),
        belief_fields_mask: batch.belief_fields_mask.clone(),
        mixture_weight_mask: batch.mixture_weight_mask.clone(),
        oracle_guidance_mask: Some(batch.oracle_target_mask.clone()),
        target_presence: batch.target_presence,
    }
}
