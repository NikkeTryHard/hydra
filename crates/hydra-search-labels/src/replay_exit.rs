//! Replay-indexed offline ExIt producer and sidecar join helpers.

use std::io;

use burn::prelude::Backend;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::safety::SafetyInfo;
use hydra_replay_sidecar::legal_mask_digest_from_bool;
use riichienv_core::replay::MjaiEvent;
use riichienv_core::state::GameState;

pub use hydra_replay_sidecar::{
    ExitSidecarIndex, REPLAY_EXIT_PROVENANCE, REPLAY_EXIT_SEMANTICS_V1, ReplayDecisionKey,
    ReplayExitLookupKey, ReplayExitRecordV1, copy_label_arrays, legal_mask_digest_from_f32,
    read_jsonl_records, source_hash_from_identity, source_net_hash_from_checkpoint_identity,
};

use crate::exit::ExitConfig;
use crate::exit_validation::ExitValidationReport;
use crate::live_exit::{
    RootDecisionContext, SelfPlayExitAdapter, budget_from_legal_count, obs_hash,
    try_exit_label_from_context_with_batched_child_values,
};
use hydra_model::model::HydraModel;
use hydra_replay_loader::mjai_loader::{prepare_replay_decision, update_safety};

pub type ReplayExitAdapter = SelfPlayExitAdapter;

pub fn generate_replay_exit_records<B: Backend>(
    source_hash: u64,
    events: &[MjaiEvent],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,
    source_net_hash: u64,
    source_version: u32,
) -> io::Result<(Vec<ReplayExitRecordV1>, ExitValidationReport)> {
    let mut state = GameState::new(
        0,
        true,
        Some(0),
        0,
        riichienv_core::rule::GameRule::default_tenhou(),
    );
    let mut safety = std::array::from_fn(|_| SafetyInfo::default());
    let mut encoder = hydra_core::encoder::ObservationEncoder::new();
    let mut adapter = ReplayExitAdapter::new();
    let mut flat_buf = Vec::new();
    let mut values_buf = Vec::new();
    let mut records = Vec::new();
    let mut report = ExitValidationReport::new();

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
            let label = try_exit_label_from_context_with_batched_child_values(
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

            if let Some(label) = label {
                let supported_actions = label.mask.iter().filter(|&&m| m > 0.0).count() as u8;
                let legal_discard_count = ctx.legal_mask[..=36]
                    .iter()
                    .filter(|&&is_legal| is_legal)
                    .count() as u8;
                let coverage = if legal_discard_count == 0 {
                    0.0
                } else {
                    supported_actions as f32 / legal_discard_count as f32
                };
                let base_pi = hydra_core::arena::softmax_temperature(
                    &ctx.policy_logits,
                    &ctx.legal_mask,
                    1.0,
                );
                let mut kl_to_base = 0.0f32;
                for (action, &q) in base_pi.iter().enumerate().take(HYDRA_ACTION_SPACE) {
                    let p = label.target[action];
                    if label.mask[action] > 0.0 && p > 1e-8 && q > 1e-8 {
                        kl_to_base += p * (p / q).ln();
                    }
                }

                report.labels_emitted += 1;
                report.coverage_sum += coverage as f64;
                report.supported_actions_sum += u64::from(supported_actions);
                report.root_visits_sum += u64::from(budget_from_legal_count(
                    exit_cfg,
                    legal_discard_count as usize,
                ));
                report.kl_sum += kl_to_base as f64;

                records.push(ReplayExitRecordV1 {
                    version: 1,
                    semantics: REPLAY_EXIT_SEMANTICS_V1.to_string(),
                    provenance: REPLAY_EXIT_PROVENANCE.to_string(),
                    key,
                    action: decision.action_id,
                    legal_mask_digest: legal_mask_digest_from_bool(&ctx.legal_mask),
                    source_net_hash,
                    source_version,
                    root_visit_count: budget_from_legal_count(
                        exit_cfg,
                        legal_discard_count as usize,
                    ),
                    legal_discard_count,
                    supported_actions,
                    coverage,
                    kl_to_base,
                    target: label.target.to_vec(),
                    mask: label.mask.to_vec(),
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

pub fn replay_exit_records_for_identity<B: Backend>(
    source_identity: &str,
    events: &[MjaiEvent],
    model: &HydraModel<B>,
    device: &B::Device,
    exit_cfg: &ExitConfig,
    source_net_hash: u64,
    source_version: u32,
) -> io::Result<(Vec<ReplayExitRecordV1>, ExitValidationReport)> {
    generate_replay_exit_records(
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
