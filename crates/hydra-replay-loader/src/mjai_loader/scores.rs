use super::*;

pub fn final_scores(events: &[MjaiEvent]) -> io::Result<[i32; 4]> {
    let mut scores = [25_000; 4];
    for event in events {
        match event {
            MjaiEvent::StartKyoku { scores: round, .. } => {
                copy_score_vec(round, "start_kyoku scores", &mut scores)?;
            }
            MjaiEvent::ReachAccepted { actor } => {
                scores[*actor] -= 1_000;
            }
            MjaiEvent::Hora {
                scores: Some(after),
                ..
            }
            | MjaiEvent::Ryukyoku {
                scores: Some(after),
                ..
            } => {
                copy_score_vec(after, "terminal scores", &mut scores)?;
            }
            MjaiEvent::Hora {
                delta: Some(delta), ..
            }
            | MjaiEvent::Ryukyoku {
                delta: Some(delta), ..
            } => {
                apply_score_delta(delta, "terminal delta", &mut scores)?;
            }
            _ => {}
        }
    }
    Ok(scores)
}

pub(super) fn copy_score_vec(src: &[i32], context: &str, dst: &mut [i32; 4]) -> io::Result<()> {
    let values = score_vec4(src, context)?;
    *dst = values;
    Ok(())
}

pub(super) fn apply_score_delta(
    delta: &[i32],
    context: &str,
    scores: &mut [i32; 4],
) -> io::Result<()> {
    let values = score_vec4(delta, context)?;
    for (score, delta) in scores.iter_mut().zip(values) {
        *score += delta;
    }
    Ok(())
}

pub(super) fn score_vec4(values: &[i32], context: &str) -> io::Result<[i32; 4]> {
    let [a, b, c, d] = values else {
        return Err(invalid_data(format!("{context} must contain four scores")));
    };
    Ok([*a, *b, *c, *d])
}
