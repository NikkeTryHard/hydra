use hydra_rules::census::ActionKind;

use super::super::candidates::desc;
use super::super::records::tile;
use super::support::{
    Log, action, ankan, descriptor, discard, draw, encode_pair, kakan, legal, meld, observation,
    record, tile as tile_value,
};

fn canonical(action: hydra_rules::rl::actions::CanonicalAction) -> u32 {
    hydra_rules::rl::actions::canonical_id(&action)
        .unwrap_or_else(|error| panic!("test action absent from census: {error}"))
}

#[test]
fn claimed_discard_is_counted_once_and_keeps_public_flags() {
    let mut log = Log::new();
    log.push(draw(1, 70));
    let discard_id = canonical(action(ActionKind::Tsumogiri, Some(70), None, &[], None));
    let discard_index = log.push(discard(1, 70, discard_id));
    let pon_id = canonical(action(ActionKind::Pon, None, Some(70), &[68, 69], Some(1)));
    let pon_index = log.push(meld("pon", 0, 70, 1, &[68, 69], pon_id));

    let mut obs = observation(0, "discard_response");
    obs.concealed_hand = vec![0, 4, 8];
    obs.meld_counts[0] = 1;
    obs.history_indices = vec![0, discard_index, pon_index];
    let legal = legal(vec![action(
        ActionKind::Pon,
        None,
        Some(70),
        &[68, 69],
        Some(1),
    )]);
    let (tokens, candidates) = encode_pair(&obs, log.arena(), &legal);

    assert_eq!(tile_value(&tokens, 17, tile::OWN_COUNT), 0.0);
    assert_eq!(tile_value(&tokens, 17, tile::PUBLIC_COUNT), 3.0);
    assert_eq!(record(&tokens, 17, 0, 1), 4.0);
    assert_eq!(record(&tokens, 17, 0, 2), 0.0);
    assert_eq!(record(&tokens, 17, 0, 3), 1.0);
    assert_eq!(record(&tokens, 17, 0, 7), 1.0);
    assert_eq!(record(&tokens, 17, 0, 8), 5.0);
    assert_eq!(record(&tokens, 17, 1, 1), 4.0);
    assert_eq!(record(&tokens, 17, 2, 1), 4.0);

    assert_eq!(descriptor(&candidates, 0, desc::KIND), 5);
    assert_eq!(descriptor(&candidates, 0, desc::SOURCE), 1);
    assert_eq!(descriptor(&candidates, 0, desc::FACE), 17);
    assert_eq!(descriptor(&candidates, 0, desc::FROM_DRAW), 1);
    assert_eq!(descriptor(&candidates, 0, desc::CONTESTED), 1);
    assert_eq!(descriptor(&candidates, 0, desc::REMOVALS), 17);
    assert_eq!(descriptor(&candidates, 0, desc::REMOVALS + 1), 17);
    assert_eq!(descriptor(&candidates, 0, desc::REMOVALS + 2), -1);
}

#[test]
fn ankan_and_kakan_keep_four_copy_provenance() {
    let mut ankan_log = Log::new();
    let ankan_id = canonical(action(ActionKind::Ankan, None, None, &[4, 5, 6, 7], None));
    let ankan_index = ankan_log.push(ankan(1, &[4, 5, 6, 7], ankan_id));
    let mut ankan_obs = observation(0, "discard_response");
    ankan_obs.history_indices = vec![0, ankan_index];
    let (ankan_tokens, _) = encode_pair(&ankan_obs, ankan_log.arena(), &legal(Vec::new()));
    assert_eq!(tile_value(&ankan_tokens, 1, tile::PUBLIC_COUNT), 4.0);
    assert_eq!(record(&ankan_tokens, 1, 0, 1), 4.0);
    assert_eq!(record(&ankan_tokens, 1, 0, 8), 7.0);

    let mut kakan_log = Log::new();
    let discard_id = canonical(action(ActionKind::Discard, Some(70), None, &[], None));
    let discard_index = kakan_log.push(discard(2, 70, discard_id));
    let pon_id = canonical(action(ActionKind::Pon, None, Some(70), &[68, 69], Some(2)));
    let pon_index = kakan_log.push(meld("pon", 1, 70, 2, &[68, 69], pon_id));
    let kakan_id = canonical(action(ActionKind::Kakan, Some(71), None, &[], None));
    let kakan_index = kakan_log.push(kakan(1, 71, kakan_id));
    let mut kakan_obs = observation(0, "discard_response");
    kakan_obs.history_indices = vec![0, discard_index, pon_index, kakan_index];
    let (kakan_tokens, _) = encode_pair(&kakan_obs, kakan_log.arena(), &legal(Vec::new()));
    assert_eq!(tile_value(&kakan_tokens, 17, tile::PUBLIC_COUNT), 4.0);
    assert_eq!(record(&kakan_tokens, 17, 0, 1), 4.0);
    assert_eq!(record(&kakan_tokens, 17, 0, 8), 8.0);
    assert_eq!(record(&kakan_tokens, 17, 3, 8), 8.0);
}
