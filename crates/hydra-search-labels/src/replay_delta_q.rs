//! Replay-indexed offline delta-q producer and sidecar join helpers.

use std::io;

use burn::prelude::Backend;
use hydra_core::action::{DISCARD_END, HYDRA_ACTION_SPACE};
use hydra_core::safety::SafetyInfo;
use riichienv_core::replay::MjaiEvent;
use riichienv_core::state::GameState;

pub use hydra_replay_sidecar::{
    DeltaQSidecarIndex, REPLAY_DELTA_Q_PROVENANCE, REPLAY_DELTA_Q_SEMANTICS_V1,
    ReplayDeltaQLookupKey, ReplayDeltaQRecordV1, validate_delta_q_contract,
};

use crate::delta_q_validation::DeltaQValidationReport;
use crate::exit::ExitConfig;
use crate::live_exit::{
    RootDecisionContext, SelfPlayExitAdapter, budget_from_legal_count, obs_hash,
    try_search_labels_from_context_with_batched_child_values,
};
use crate::replay_exit::{
    ReplayDecisionKey, legal_mask_digest_from_f32, source_hash_from_identity,
};
use hydra_model::model::HydraModel;
use hydra_replay_loader::mjai_loader::{prepare_replay_decision, update_safety};
use hydra_replay_loader::replay_targets::bool_mask_to_f32;

pub fn generate_replay_delta_q_records<B: Backend>(
    source_hash: u64,
    events: &[MjaiEvent],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,
    source_net_hash: u64,
    source_version: u32,
) -> io::Result<(Vec<ReplayDeltaQRecordV1>, DeltaQValidationReport)> {
    let mut state = GameState::new(
        0,
        true,
        Some(0),
        0,
        riichienv_core::rule::GameRule::default_tenhou(),
    );
    let mut safety = std::array::from_fn(|_| SafetyInfo::default());
    let mut encoder = hydra_core::encoder::ObservationEncoder::new();
    let mut adapter = SelfPlayExitAdapter::new();
    let mut flat_buf = Vec::new();
    let mut values_buf = Vec::new();
    let mut records = Vec::new();
    let mut report = DeltaQValidationReport::new();

    for (idx, event) in events.iter().enumerate() {
        if let Some(decision) = prepare_replay_decision(event, &mut state, &safety, &mut encoder)? {
            let actor = decision.actor;
            let ctx = RootDecisionContext {
                obs_encoded: decision.obs_encoded,
                legal_mask: decision.legal_mask,
                policy_logits: model.policy_cpu(&decision.obs_encoded, device),
                player_id: actor as u8,
            };
            let key = ReplayDecisionKey {
                source_hash,
                event_index: idx as u32,
                actor: actor as u8,
                obs_hash: obs_hash(&ctx.obs_encoded),
            };

            report.total_states += 1;
            let labels = try_search_labels_from_context_with_batched_child_values(
                &state,
                &decision.obs,
                &ctx,
                &safety[actor],
                exit_cfg,
                &mut |child_obs| {
                    model.fill_batch_value_cpu(child_obs, device, &mut flat_buf, &mut values_buf);
                    values_buf.clone()
                },
                &mut adapter,
            );

            if let Some(delta_q) = labels.and_then(|labels| labels.delta_q) {
                let legal_discard_count = ctx.legal_mask[..=DISCARD_END as usize]
                    .iter()
                    .filter(|&&is_legal| is_legal)
                    .count();
                let supported_actions = delta_q.mask.iter().filter(|&&m| m > 0.0).count();
                let coverage = if legal_discard_count == 0 {
                    0.0
                } else {
                    supported_actions as f64 / legal_discard_count as f64
                };
                report.labels_emitted += 1;
                report.coverage_sum += coverage;
                report.supported_actions_sum += supported_actions as u64;
                report.root_visits_sum +=
                    u64::from(budget_from_legal_count(exit_cfg, legal_discard_count));
                for action_idx in 0..HYDRA_ACTION_SPACE {
                    if delta_q.mask[action_idx] <= 0.0 {
                        continue;
                    }
                    let value = delta_q.target[action_idx] as f64;
                    report.masked_abs_sum += value.abs();
                    report.masked_entry_count += 1;
                    if value > 0.0 {
                        report.masked_positive_count += 1;
                    } else if value < 0.0 {
                        report.masked_negative_count += 1;
                    } else {
                        report.masked_zero_count += 1;
                    }
                }
                records.push(ReplayDeltaQRecordV1 {
                    version: 1,
                    semantics: REPLAY_DELTA_Q_SEMANTICS_V1.to_string(),
                    provenance: REPLAY_DELTA_Q_PROVENANCE.to_string(),
                    key,
                    action: decision.action_id,
                    legal_mask_digest: legal_mask_digest_from_f32(&bool_mask_to_f32(
                        ctx.legal_mask,
                    )),
                    source_net_hash,
                    source_version,
                    target: delta_q.target.to_vec(),
                    mask: delta_q.mask.to_vec(),
                });
            } else {
                report.labels_rejected += 1;
                report.rejected_other += 1;
            }
        }

        update_safety(&mut safety, event)?;
        state.apply_mjai_event(event.clone());
    }

    Ok((records, report))
}

pub fn replay_delta_q_records_for_identity<B: Backend>(
    source_identity: &str,
    events: &[MjaiEvent],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,
    source_net_hash: u64,
    source_version: u32,
) -> io::Result<(Vec<ReplayDeltaQRecordV1>, DeltaQValidationReport)> {
    generate_replay_delta_q_records(
        source_hash_from_identity(source_identity),
        events,
        model,
        device,
        exit_cfg,
        source_net_hash,
        source_version,
    )
}

#[cfg(test)]
mod tests;
