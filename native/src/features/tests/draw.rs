use hydra_rules::census::ActionKind;

use super::super::WIDTH;
use super::super::layout::{desc, seat, tile};
use super::support::{
    Log, action, descriptor, discard, draw, encode_pair, legal, meld, observation, record,
    riichi_declared, tile as tile_value,
};

fn canonical(action: hydra_rules::rl::actions::CanonicalAction) -> u32 {
    hydra_rules::rl::actions::canonical_id(&action)
        .unwrap_or_else(|error| panic!("test action absent from census: {error}"))
}

#[test]
fn red_five_drawn_and_held_are_distinct() {
    let mut log = Log::new();
    log.push(draw(0, 52));
    let mut obs = observation(0, "draw_decision");
    obs.concealed_hand = vec![53];
    obs.own_drawn_tile = Some(52);
    obs.history_indices = vec![0, 1];
    let legal = legal(vec![
        action(ActionKind::Discard, Some(53), None, &[], None),
        action(ActionKind::Tsumogiri, Some(52), None, &[], None),
        action(ActionKind::RiichiDiscard, Some(53), None, &[], None),
        action(ActionKind::RiichiDiscard, Some(52), None, &[], None),
    ]);
    let (tokens, candidates) = encode_pair(&obs, log.arena(), &legal);

    assert_eq!(tile_value(&tokens, 13, tile::OWN_COUNT), 2.0);
    assert_eq!(tile_value(&tokens, 13, tile::OWN_RED), 1.0);
    assert_eq!(tile_value(&tokens, 13, tile::DRAWN_FACE), 1.0);
    assert_eq!(tile_value(&tokens, 13, tile::DRAWN_RED), 1.0);
    assert_eq!(tile_value(&tokens, 13, tile::PUBLIC_COUNT), 0.0);
    assert_eq!(record(&tokens, 13, 0, 1), 1.0);
    assert_eq!(record(&tokens, 13, 0, 6), 0.0);
    assert_eq!(record(&tokens, 13, 1, 1), 2.0);
    assert_eq!(record(&tokens, 13, 1, 6), 1.0);
    assert_eq!(record(&tokens, 13, 1, 7), 1.0);

    assert_eq!(descriptor(&candidates, 0, desc::KIND), 1);
    assert_eq!(descriptor(&candidates, 0, desc::FACE), 13);
    assert_eq!(descriptor(&candidates, 0, desc::RED), 0);
    assert_eq!(descriptor(&candidates, 0, desc::FROM_DRAW), 0);
    assert_eq!(descriptor(&candidates, 0, desc::REMOVALS), 13);
    assert_eq!(descriptor(&candidates, 0, desc::REMOVAL_RED), 0);

    assert_eq!(descriptor(&candidates, 1, desc::KIND), 2);
    assert_eq!(descriptor(&candidates, 1, desc::RED), 1);
    assert_eq!(descriptor(&candidates, 1, desc::FROM_DRAW), 1);
    assert_eq!(descriptor(&candidates, 1, desc::REMOVAL_RED), 1);

    assert_eq!(descriptor(&candidates, 2, desc::RIICHI), 1);
    assert_eq!(descriptor(&candidates, 2, desc::FROM_DRAW), 0);
    assert_eq!(descriptor(&candidates, 3, desc::RIICHI), 1);
    assert_eq!(descriptor(&candidates, 3, desc::FROM_DRAW), 1);
}

#[test]
fn stale_cached_draw_is_not_reintroduced_in_response() {
    let mut log = Log::new();
    log.push(draw(0, 52));
    let discard_id = canonical(action(ActionKind::Discard, Some(53), None, &[], None));
    log.push(discard(0, 53, discard_id));
    let other_id = canonical(action(ActionKind::Discard, Some(70), None, &[], None));
    log.push(discard(1, 70, other_id));

    let mut obs = observation(0, "discard_response");
    obs.concealed_hand = vec![52];
    obs.own_drawn_tile = Some(52);
    obs.history_indices = vec![0, 1, 2, 3];
    let legal = legal(Vec::new());
    let (tokens, _) = encode_pair(&obs, log.arena(), &legal);

    assert_eq!(tile_value(&tokens, 13, tile::OWN_COUNT), 1.0);
    assert_eq!(tile_value(&tokens, 13, tile::OWN_RED), 1.0);
    assert_eq!(tile_value(&tokens, 13, tile::DRAWN_FACE), 0.0);
    assert_eq!(tile_value(&tokens, 13, tile::DRAWN_RED), 0.0);
    assert_eq!(record(&tokens, 13, 0, 1), 1.0);
    assert_eq!(record(&tokens, 13, 0, 7), 0.0);
}

#[test]
fn kept_post_pon_draw_is_concealed_but_consumed_draw_is_not() {
    let mut kept_log = Log::new();
    let draw_index = kept_log.push(draw(0, 52));
    let discard_held = canonical(action(ActionKind::Discard, Some(53), None, &[], None));
    let discard_held_index = kept_log.push(discard(0, 53, discard_held));
    let discard_other = canonical(action(ActionKind::Discard, Some(70), None, &[], None));
    let discard_other_index = kept_log.push(discard(1, 70, discard_other));
    let pon_id = canonical(action(ActionKind::Pon, None, Some(70), &[68, 69], Some(1)));
    let pon_index = kept_log.push(meld("pon", 0, 70, 1, &[68, 69], pon_id));
    let mut kept = observation(0, "draw_decision");
    kept.own_drawn_tile = Some(52);
    kept.history_indices = vec![
        0,
        draw_index,
        discard_held_index,
        discard_other_index,
        pon_index,
    ];
    let (kept_tokens, _) = encode_pair(&kept, kept_log.arena(), &legal(Vec::new()));
    assert_eq!(tile_value(&kept_tokens, 13, tile::OWN_COUNT), 1.0);
    assert_eq!(tile_value(&kept_tokens, 13, tile::OWN_RED), 1.0);
    assert_eq!(tile_value(&kept_tokens, 13, tile::DRAWN_FACE), 0.0);
    assert_eq!(tile_value(&kept_tokens, 13, tile::DRAWN_RED), 0.0);
    assert_eq!(tile_value(&kept_tokens, 13, tile::PUBLIC_COUNT), 1.0);
    assert_eq!(record(&kept_tokens, 13, 0, 1), 1.0);
    assert_eq!(record(&kept_tokens, 13, 0, 7), 0.0);
    assert_eq!(tile_value(&kept_tokens, 17, tile::PUBLIC_COUNT), 3.0);

    let mut consumed_log = Log::new();
    let draw_index = consumed_log.push(draw(0, 52));
    let discard_held = canonical(action(ActionKind::Discard, Some(53), None, &[], None));
    let discard_held_index = consumed_log.push(discard(0, 53, discard_held));
    let discard_other = canonical(action(ActionKind::Discard, Some(54), None, &[], None));
    let discard_other_index = consumed_log.push(discard(1, 54, discard_other));
    let pon_id = canonical(action(ActionKind::Pon, None, Some(54), &[52, 55], Some(1)));
    let pon_index = consumed_log.push(meld("pon", 0, 54, 1, &[52, 55], pon_id));
    let mut consumed = observation(0, "draw_decision");
    consumed.own_drawn_tile = Some(52);
    consumed.history_indices = vec![
        0,
        draw_index,
        discard_held_index,
        discard_other_index,
        pon_index,
    ];
    let (consumed_tokens, _) = encode_pair(&consumed, consumed_log.arena(), &legal(Vec::new()));
    assert_eq!(tile_value(&consumed_tokens, 13, tile::OWN_COUNT), 0.0);
    assert_eq!(tile_value(&consumed_tokens, 13, tile::DRAWN_FACE), 0.0);
    assert_eq!(tile_value(&consumed_tokens, 13, tile::PUBLIC_COUNT), 4.0);
}

#[test]
fn riichi_discard_pair_merges_to_one_copy() {
    let mut log = Log::new();
    log.push(draw(0, 52));
    let riichi_id = canonical(action(ActionKind::RiichiDiscard, Some(52), None, &[], None));
    let discard_index = log.push(discard(0, 52, riichi_id));
    let riichi_index = log.push(riichi_declared(0, 52, riichi_id));

    let mut obs = observation(0, "discard_response");
    obs.own_drawn_tile = Some(52);
    obs.riichi_states[0] = "declared";
    obs.history_indices = vec![0, 1, discard_index, riichi_index];
    let legal = legal(Vec::new());
    let (tokens, _) = encode_pair(&obs, log.arena(), &legal);

    assert_eq!(tile_value(&tokens, 13, tile::PUBLIC_COUNT), 1.0);
    assert_eq!(record(&tokens, 13, 0, 1), 3.0);
    assert_eq!(record(&tokens, 13, 0, 7), 1.0);
    assert_eq!(record(&tokens, 13, 0, 8), 3.0);
    assert_eq!(
        tokens[34 * WIDTH + seat::RIICHI_DECLARED],
        super::super::layout::norm_position(Some(3))
    );
}
