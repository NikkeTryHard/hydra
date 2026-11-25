use hydra_rules::census::ActionKind;

use super::super::records::tile;
use super::support::{
    Log, action, call_window, discard, dora, draw, encode_pair, hidden_call, legal, observation,
    record, tile as tile_value,
};

fn canonical(action: hydra_rules::rl::actions::CanonicalAction) -> u32 {
    hydra_rules::rl::actions::canonical_id(&action)
        .unwrap_or_else(|error| panic!("test action absent from census: {error}"))
}

#[test]
fn five_dora_slots_are_one_hot_and_counted_once() {
    let mut log = Log::new();
    let first = log.push(dora(0));
    let second = log.push(dora(4));
    let third = log.push(dora(8));
    let mut obs = observation(0, "draw_decision");
    obs.dora_indicators = [0, 4, 8, -1, -1];
    obs.history_indices = vec![0, first, second, third];
    let legal = legal(Vec::new());
    let (tokens, _) = encode_pair(&obs, log.arena(), &legal);

    assert_eq!(tile_value(&tokens, 0, tile::DORA0), 1.0);
    assert_eq!(tile_value(&tokens, 0, tile::DORA0 + 1), 0.0);
    assert_eq!(tile_value(&tokens, 1, tile::DORA0), 0.0);
    assert_eq!(tile_value(&tokens, 1, tile::DORA0 + 1), 1.0);
    assert_eq!(tile_value(&tokens, 2, tile::DORA0 + 2), 1.0);
    assert_eq!(tile_value(&tokens, 0, tile::PUBLIC_COUNT), 1.0);
    assert_eq!(tile_value(&tokens, 1, tile::PUBLIC_COUNT), 1.0);
    assert_eq!(tile_value(&tokens, 2, tile::PUBLIC_COUNT), 1.0);
    assert_eq!(record(&tokens, 0, 0, 1), 5.0);
    assert_eq!(record(&tokens, 0, 0, 8), 13.0);
}

#[test]
fn hidden_and_call_window_markers_do_not_shift_positions() {
    let mut plain = Log::new();
    plain.push(draw(0, 52));
    let plain_index = plain.push(discard(
        0,
        53,
        canonical(action(ActionKind::Discard, Some(53), None, &[], None)),
    ));

    let mut hidden = Log::new();
    hidden.push(draw(0, 52));
    hidden.push(hidden_call());
    let hidden_index = hidden.push(discard(
        0,
        53,
        canonical(action(ActionKind::Discard, Some(53), None, &[], None)),
    ));

    let mut windowed = Log::new();
    windowed.push(draw(0, 52));
    let window_index = windowed.push(call_window());
    let windowed_index = windowed.push(discard(
        0,
        53,
        canonical(action(ActionKind::Discard, Some(53), None, &[], None)),
    ));
    let pass = legal(vec![action(ActionKind::Pass, None, None, &[], None)]);

    let mut obs = observation(0, "discard_response");
    obs.history_indices = vec![0, 1, plain_index];
    let (plain_tokens, plain_candidates) = encode_pair(&obs, plain.arena(), &pass);

    obs.history_indices = vec![0, 1, hidden_index];
    let (hidden_tokens, hidden_candidates) = encode_pair(&obs, hidden.arena(), &pass);
    assert_eq!(plain_tokens, hidden_tokens);
    assert_eq!(plain_candidates, hidden_candidates);

    obs.history_indices = vec![0, 1, window_index, windowed_index];
    let (windowed_tokens, windowed_candidates) = encode_pair(&obs, windowed.arena(), &pass);
    assert_eq!(plain_tokens, windowed_tokens);
    assert_eq!(plain_candidates, windowed_candidates);
}
