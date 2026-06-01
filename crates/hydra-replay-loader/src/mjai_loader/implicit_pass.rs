use super::*;

fn observation_for_implicit_pass(state: &mut GameState, actor: u8) -> io::Result<Observation> {
    let t_obs = Instant::now();
    let obs = state.get_observation(actor);
    REPLAY_OBSERVATION_NS.fetch_add(t_obs.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Ok(obs)
}

pub(super) fn mark_missed_agari_for_implicit_pass(state: &mut GameState, actor: u8, had_ron: bool) {
    if !had_ron {
        return;
    }

    let player = &mut state.players[actor as usize];
    player.missed_agari_doujun = true;
    if player.riichi_declared {
        player.missed_agari_riichi = true;
    }
}

pub(super) fn prepare_implicit_pass_decisions(
    next_event: &MjaiEvent,
    state: &mut GameState,
    safety: &[SafetyInfo; 4],
    encoder: &mut ObservationEncoder,
    options: ReplayDecisionOptions,
) -> io::Result<Vec<PreparedReplayDecision>> {
    let t_pass = Instant::now();
    let mut decisions = Vec::new();
    if state.phase != Phase::WaitResponse {
        REPLAY_IMPLICIT_PASS_NS.fetch_add(t_pass.elapsed().as_nanos() as u64, Ordering::Relaxed);
        return Ok(decisions);
    }

    match next_event {
        MjaiEvent::Other => {
            return Err(invalid_data(
                "unsupported MJAI event type during response window",
            ));
        }
        MjaiEvent::Dahai { .. }
        | MjaiEvent::Pon { .. }
        | MjaiEvent::Chi { .. }
        | MjaiEvent::Kan { .. }
        | MjaiEvent::Ankan { .. }
        | MjaiEvent::Kakan { .. }
        | MjaiEvent::Reach { .. }
        | MjaiEvent::Hora { .. } => {}
        _ => {
            state.resolve_replay_all_passes();
            REPLAY_IMPLICIT_PASS_NS
                .fetch_add(t_pass.elapsed().as_nanos() as u64, Ordering::Relaxed);
            return Ok(decisions);
        }
    }

    let responding_actor = mjai_event_actor(next_event)
        .filter(|actor| state.active_player_slice().contains(&(*actor as u8)));
    let resolve_all_passes = responding_actor.is_none();

    let active_players = state.active_player_slice().to_vec();
    let mut legal = Vec::new();
    for pid in active_players {
        if Some(pid as usize) == responding_actor {
            continue;
        }

        let pass_action = EngineAction::new(ActionType::Pass, None, &[], Some(pid));
        match options.observation_profile {
            ReplayObservationProfile::BcMinimal => {
                legal.clear();
                let t_obs = Instant::now();
                state.get_legal_actions_into(pid, &mut legal);
                REPLAY_OBSERVATION_NS
                    .fetch_add(t_obs.elapsed().as_nanos() as u64, Ordering::Relaxed);
                let (_, _, _, had_ron) = analyze_replay_legal_actions(
                    &legal,
                    ActionPhase::Normal,
                    hydra_core::action::PASS,
                );
                if let Some(decision) = finalize_prepared_replay_decision_ref(
                    pid as usize,
                    pass_action,
                    ActionPhase::Normal,
                    state,
                    safety,
                    encoder,
                    &legal,
                )? {
                    decisions.push(decision);
                }

                mark_missed_agari_for_implicit_pass(state, pid, had_ron);
            }
            ReplayObservationProfile::Full => {
                let obs = observation_for_implicit_pass(state, pid)?;
                let had_ron = {
                    let legal = obs.legal_actions_ref();
                    let (_, _, _, had_ron) = analyze_replay_legal_actions(
                        legal,
                        ActionPhase::Normal,
                        hydra_core::action::PASS,
                    );
                    had_ron
                };
                if let Some(decision) = finalize_prepared_replay_decision(
                    pid as usize,
                    pass_action,
                    obs,
                    ActionPhase::Normal,
                    state,
                    safety,
                    encoder,
                    options,
                )? {
                    decisions.push(decision);
                }

                mark_missed_agari_for_implicit_pass(state, pid, had_ron);
            }
        }
    }

    if resolve_all_passes {
        state.resolve_replay_all_passes();
    }

    REPLAY_IMPLICIT_PASS_NS.fetch_add(t_pass.elapsed().as_nanos() as u64, Ordering::Relaxed);

    Ok(decisions)
}
