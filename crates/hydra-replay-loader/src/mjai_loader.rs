//! MJAI `.json` / `.json.gz` / `.json.zst` loader for behavioral cloning data.

use crate::replay_targets::{
    build_safety_residual_targets, build_stage_a_belief_targets, build_stage_a_belief_targets_ref,
    exact_waits,
};
use crate::target_helpers::{obs_hash, oracle_target_from_scores};
use flate2::read::GzDecoder;
#[cfg(test)]
use hydra_core::action::{AKA_5M, DISCARD_END};
use hydra_core::action::{ActionPhase, HYDRA_ACTION_SPACE, riichienv_to_hydra};
use hydra_core::bridge::{
    BridgeEncodeProfile, encode_extracted_observation_facts_with_profile,
    extract_observation_facts, extract_observation_facts_ref,
};
use hydra_core::encoder::{OBS_SIZE, ObservationEncoder};
use hydra_core::safety::SafetyInfo;
use hydra_data_core::{
    CompactObservationFacts, MjaiSample, score_to_placements, scores_to_grp_index,
};
use hydra_replay_sidecar::{
    ActionLabelPair, DeltaQSidecarIndex, ExitSidecarIndex, ReplayDecisionKey, SidecarContractError,
    SidecarKind, source_hash_from_identity,
};
use riichienv_core::action::{Action as EngineAction, ActionType, Phase};
use riichienv_core::observation::Observation;
use riichienv_core::parser::mjai_to_tid;
use riichienv_core::replay::{MjaiEvent, mjai_event_actor, read_mjai_events};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;
use std::array;
use std::borrow::Cow;
use std::cell::Cell;
use std::fs;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};
use zstd::stream::read::Decoder as ZstdDecoder;

const MISSING_TILE_TARGET: u8 = 255;

mod dataset;
mod decisions;
mod implicit_pass;
mod scores;
mod sidecar;
mod sink;
mod stats;
mod stream;
mod tile;
mod validation;

pub use dataset::*;
#[cfg(test)]
pub use decisions::prepare_replay_decisions;
pub use decisions::{PreparedReplayDecision, prepare_replay_decision, should_sample_replay_event};
pub use scores::final_scores;
pub use sidecar::{ReplayLoadPolicy, ReplayTargetProfile, SidecarProvenance};
pub use sink::{ReplaySampleRecord, ReplaySampleSink};
pub use stats::{
    ReplayMaterializationStats, drain_replay_materialization_stats,
    peek_replay_materialization_stats,
};
pub use stream::*;
pub use tile::{next_discards_after, tile136_to_type, update_safety};

use decisions::*;
use implicit_pass::*;
use scores::*;
use sidecar::*;
use sink::*;
use stats::*;
use tile::*;
use validation::*;
#[derive(Clone, Copy)]
struct ReplayDecisionOptions {
    observation_profile: ReplayObservationProfile,
}

impl Default for ReplayDecisionOptions {
    fn default() -> Self {
        Self {
            observation_profile: ReplayObservationProfile::BcMinimal,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayObservationProfile {
    Full,
    BcMinimal,
}

impl ReplayObservationProfile {
    const fn bridge_profile(self) -> BridgeEncodeProfile {
        match self {
            Self::Full => BridgeEncodeProfile::full(),
            Self::BcMinimal => BridgeEncodeProfile::bc_minimal(),
        }
    }

    const fn uses_ref_observation(self) -> bool {
        matches!(self, Self::BcMinimal)
    }
}

#[inline]
pub fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[allow(
    clippy::too_many_arguments,
    reason = "loader seam carries target and sidecar policy"
)]
fn load_game_from_events_into_sink<S: ReplaySampleSink>(
    source_hash: Option<u64>,
    exit_provenance: SidecarProvenance,
    delta_q_provenance: SidecarProvenance,
    profile: ReplayTargetProfile,
    observation_profile: ReplayObservationProfile,
    events: Vec<MjaiEvent>,
    exit_sidecar: Option<&ExitSidecarIndex>,
    delta_q_sidecar: Option<&DeltaQSidecarIndex>,
    sink: &mut S,
) -> io::Result<[i32; 4]> {
    let mut stats = ReplayProfileStats::default();
    let t_precompute = Instant::now();
    let final_scores = final_scores(&events)?;
    let placements = score_to_placements(final_scores);
    let oracle_target = profile
        .oracle
        .then(|| oracle_target_from_scores(final_scores));
    let needs_opponent_targets = profile != ReplayTargetProfile::minimal_bc()
        && (profile.safety_residual || observation_profile != ReplayObservationProfile::BcMinimal);
    let next_discards = needs_opponent_targets
        .then(|| next_discards_after(&events))
        .transpose()?;
    let grp_label = scores_to_grp_index(final_scores).map_err(invalid_data)?;
    stats.precompute_ns += t_precompute.elapsed().as_nanos();
    let mut state = GameState::new(0, true, Some(0), 0, GameRule::default_tenhou());
    let mut safety = array::from_fn(|_| SafetyInfo::default());
    let mut encoder = ObservationEncoder::new();
    let needs_exit_lookup = profile.exit && exit_sidecar.is_some();
    let needs_delta_q_lookup = profile.delta_q && delta_q_sidecar.is_some();
    let needs_replay_key = source_hash.is_some() && (needs_exit_lookup || needs_delta_q_lookup);
    let decision_options = ReplayDecisionOptions {
        observation_profile,
    };

    if events.iter().any(|event| matches!(event, MjaiEvent::Other)) {
        return Err(invalid_data("unsupported MJAI event type"));
    }

    for (idx, event) in events.iter().enumerate() {
        stats.event_count += 1;
        let t_prepare = Instant::now();
        let decisions = prepare_replay_decisions_with_options(
            event,
            &mut state,
            &safety,
            &mut encoder,
            decision_options,
        )?;
        stats.prepare_decisions_ns += t_prepare.elapsed().as_nanos();
        let mut event_targets = next_discards
            .as_deref()
            .map(|next_discards| EventOpponentTargetCache::new(next_discards, idx));
        for decision in decisions {
            stats.decision_count += 1;
            let actor = decision.actor;
            let legal_mask = decision.legal_mask_f32;
            let actor_targets = if let Some(event_targets) = event_targets.as_mut() {
                let t_opp = Instant::now();
                let actor_targets = actor_relative_opponent_targets(actor, event_targets, &state);
                stats.opponent_targets_ns += t_opp.elapsed().as_nanos();
                stats.exact_waits_ns += event_targets.exact_waits_ns;
                event_targets.exact_waits_ns = 0;
                actor_targets
            } else {
                ActorRelativeOpponentTargets::default()
            };
            let (safety_residual, safety_residual_mask) = if profile.safety_residual {
                let t_safety = Instant::now();
                let (values, mask) = build_safety_residual_targets(
                    &legal_mask,
                    &safety[actor],
                    &actor_targets.wait_sets,
                );
                stats.safety_residual_ns += t_safety.elapsed().as_nanos();
                (Some(values), Some(mask))
            } else {
                (None, None)
            };
            let (belief_fields, mixture_weights, belief_fields_present, mixture_weights_present) =
                if profile.belief || profile.mixture {
                    let t_belief = Instant::now();
                    let (belief_fields, mixture_weights, belief_present, mixture_present) =
                        if decision.use_ref_targets {
                            let obs_ref = state.observe(actor as u8);
                            build_stage_a_belief_targets_ref(&state, actor, &obs_ref)
                        } else {
                            build_stage_a_belief_targets(&state, actor, &decision.obs)
                        };
                    stats.belief_targets_ns += t_belief.elapsed().as_nanos();
                    (
                        if profile.belief { belief_fields } else { None },
                        if profile.mixture {
                            mixture_weights
                        } else {
                            None
                        },
                        profile.belief && belief_present,
                        profile.mixture && mixture_present,
                    )
                } else {
                    (None, None, false, false)
                };
            let t_sidecar = Instant::now();
            let replay_key = needs_replay_key.then(|| ReplayDecisionKey {
                source_hash: source_hash.expect("needs_replay_key implies source hash"),
                event_index: idx as u32,
                actor: actor as u8,
                obs_hash: obs_hash(&decision.obs_encoded),
            });
            let joined_exit = lookup_joined_label(
                if needs_exit_lookup {
                    exit_sidecar
                } else {
                    None
                },
                replay_key,
                decision.action_id,
                &legal_mask,
                exit_provenance,
                SidecarKind::Exit,
                |sidecar, key, action, legal_mask, source_net_hash, source_version| {
                    sidecar.lookup_label(key, action, legal_mask, source_net_hash, source_version)
                },
            )
            .map_err(|err| invalid_data(err.to_string()))?;
            let joined_delta_q = lookup_joined_label(
                if needs_delta_q_lookup {
                    delta_q_sidecar
                } else {
                    None
                },
                replay_key,
                decision.action_id,
                &legal_mask,
                delta_q_provenance,
                SidecarKind::DeltaQ,
                |sidecar, key, action, legal_mask, source_net_hash, source_version| {
                    sidecar.lookup_label(key, action, legal_mask, source_net_hash, source_version)
                },
            )
            .map_err(|err| invalid_data(err.to_string()))?;
            stats.sidecar_lookup_ns += t_sidecar.elapsed().as_nanos();
            let t_push = Instant::now();
            sink.push_sample(ReplaySampleRecord {
                obs: decision.obs_encoded,
                compact_facts: decision.compact_facts,
                action: decision.action_id,
                legal_mask,
                placement: placements[actor],
                score_delta: final_scores[actor] - state.players[actor].score,
                grp_label,
                oracle_target,
                tenpai: actor_targets.tenpai,
                opp_next: actor_targets.opp_next,
                danger: actor_targets.danger,
                danger_mask: actor_targets.danger_mask,
                safety_residual,
                safety_residual_mask,
                exit_target: joined_exit.map(|(target, _)| target),
                exit_mask: joined_exit.map(|(_, mask)| mask),
                delta_q_target: joined_delta_q.map(|(target, _)| target),
                delta_q_mask: joined_delta_q.map(|(_, mask)| mask),
                belief_fields,
                mixture_weights,
                belief_fields_present,
                mixture_weights_present,
            })?;
            stats.sample_push_ns += t_push.elapsed().as_nanos();
        }

        let t_update = Instant::now();
        update_safety(&mut safety, event)?;
        stats.update_safety_ns += t_update.elapsed().as_nanos();
        validate_terminal_event(event, &state)?;
        let t_apply = Instant::now();
        state
            .try_apply_mjai_event(event.clone())
            .map_err(|err| invalid_data(format!("replay state update failed: {err}")))?;
        stats.apply_event_ns += t_apply.elapsed().as_nanos();
    }

    stats.implicit_pass_ns = REPLAY_IMPLICIT_PASS_NS.swap(0, Ordering::Relaxed) as u128;
    stats.replay_observation_ns = REPLAY_OBSERVATION_NS.swap(0, Ordering::Relaxed) as u128;
    stats.legal_mask_build_ns = REPLAY_LEGAL_MASK_BUILD_NS.swap(0, Ordering::Relaxed) as u128;
    stats.encode_observation_ns = REPLAY_ENCODE_OBS_NS.swap(0, Ordering::Relaxed) as u128;
    record_replay_materialization_stats(ReplayMaterializationStats {
        decompress_ns: 0,
        json_parse_ns: 0,
        replay_update_ns: stats.update_safety_ns.saturating_add(stats.apply_event_ns),
        observation_encode_ns: stats
            .replay_observation_ns
            .saturating_add(stats.encode_observation_ns),
        mask_build_ns: stats
            .legal_mask_build_ns
            .saturating_add(stats.legal_mask_convert_ns),
        target_synthesis_ns: stats
            .opponent_targets_ns
            .saturating_add(stats.exact_waits_ns)
            .saturating_add(stats.safety_residual_ns)
            .saturating_add(stats.belief_targets_ns)
            .saturating_add(stats.sidecar_lookup_ns)
            .saturating_add(stats.sample_push_ns)
            .saturating_add(stats.precompute_ns)
            .saturating_add(stats.prepare_decisions_ns),
        event_count: stats.event_count,
        decision_count: stats.decision_count,
    });

    maybe_print_replay_profile(&stats);

    Ok(final_scores)
}

#[cfg(test)]
mod tests;
