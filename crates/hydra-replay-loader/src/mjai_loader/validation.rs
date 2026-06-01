use super::*;

struct HoraTerminalFields<'a> {
    actor: usize,
    target: usize,
    pai: Option<&'a str>,
    fu: Option<u32>,
    han: Option<u32>,
    scores: Option<&'a [i32]>,
    delta: Option<&'a [i32]>,
}

pub(super) fn validate_terminal_event(event: &MjaiEvent, state: &GameState) -> io::Result<()> {
    match event {
        MjaiEvent::Hora {
            actor,
            target,
            pai,
            fu,
            han,
            scores,
            delta,
            ..
        } => validate_hora_terminal(
            HoraTerminalFields {
                actor: *actor,
                target: *target,
                pai: pai.as_deref(),
                fu: *fu,
                han: *han,
                scores: scores.as_deref(),
                delta: delta.as_deref(),
            },
            state,
        ),
        MjaiEvent::Ryukyoku { scores, delta, .. } => {
            validate_terminal_score_fields(scores.as_deref(), delta.as_deref(), state, "ryukyoku")
        }
        _ => Ok(()),
    }
}

fn validate_hora_terminal(fields: HoraTerminalFields<'_>, state: &GameState) -> io::Result<()> {
    if fields.actor >= 4 || fields.target >= 4 {
        return Err(invalid_data("hora actor/target out of range"));
    }

    validate_terminal_score_fields(fields.scores, fields.delta, state, "hora")?;

    if let Some(pai) = fields.pai {
        mjai_tile(pai)?;
    }

    if let (Some(fu), Some(han), Some(delta)) = (fields.fu, fields.han, fields.delta) {
        validate_hora_point_delta(fields.actor, fields.target, fu, han, delta, state)?;
    }

    Ok(())
}

fn validate_terminal_score_fields(
    scores: Option<&[i32]>,
    delta: Option<&[i32]>,
    state: &GameState,
    context: &str,
) -> io::Result<()> {
    let before = current_scores(state);
    match (scores, delta) {
        (Some(scores), Some(delta)) => {
            let scores = score_vec4(scores, &format!("{context} scores"))?;
            let delta = score_vec4(delta, &format!("{context} delta"))?;
            for idx in 0..4 {
                if before[idx] + delta[idx] != scores[idx] {
                    return Err(invalid_data(format!("{context} scores do not match delta")));
                }
            }
        }
        (Some(scores), None) => {
            score_vec4(scores, &format!("{context} scores"))?;
        }
        (None, Some(delta)) => {
            score_vec4(delta, &format!("{context} delta"))?;
        }
        (None, None) => {}
    }
    Ok(())
}

fn current_scores(state: &GameState) -> [i32; 4] {
    [
        state.players[0].score,
        state.players[1].score,
        state.players[2].score,
        state.players[3].score,
    ]
}

fn validate_hora_point_delta(
    actor: usize,
    target: usize,
    fu: u32,
    han: u32,
    delta: &[i32],
    state: &GameState,
) -> io::Result<()> {
    if han == 0 || han > u8::MAX as u32 || fu > u8::MAX as u32 {
        return Ok(());
    }
    if state.honba != 0 || state.riichi_sticks != 0 || state.riichi_pending_acceptance.is_some() {
        return Ok(());
    }
    let delta = score_vec4(delta, "hora delta")?;
    let is_tsumo = actor == target;
    let score = riichienv_core::score::calculate_score(
        han as u8,
        fu as u8,
        actor as u8 == state.oya,
        is_tsumo,
        0,
        4,
    );
    if is_tsumo {
        for (seat, &seat_delta) in delta.iter().enumerate() {
            if seat == actor {
                continue;
            }
            let expected = if actor as u8 == state.oya || seat as u8 != state.oya {
                score.pay_tsumo_ko
            } else {
                score.pay_tsumo_oya
            } as i32;
            if seat_delta != -expected {
                return Err(invalid_data("hora tsumo points do not match han/fu"));
            }
        }
    } else if delta[target] != -(score.pay_ron as i32) {
        return Err(invalid_data("hora ron points do not match han/fu"));
    }
    Ok(())
}
