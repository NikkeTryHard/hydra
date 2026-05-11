use super::*;
use crate::action::{Action, ActionType};

fn sample_observation() -> Observation {
    Observation::new(
        1,
        [vec![0, 1], vec![36], vec![], vec![]],
        Default::default(),
        [vec![4], vec![], vec![], vec![]],
        vec![12],
        [25_000, 24_000, 26_000, 25_000],
        [false, true, false, false],
        vec![
            Action::new(ActionType::Discard, Some(4), &[], Some(1)),
            Action::new(ActionType::Riichi, None, &[], Some(1)),
        ],
        vec!["event-a".to_string(), "event-b".to_string()],
        2,
        1,
        27,
        0,
        3,
        vec![7, 8],
        true,
        [None, Some(4), None, None],
        [Some(0), None, None, None],
        Some(4),
    )
}

#[test]
fn new_converts_input_collections_and_preserves_metadata() {
    let obs = sample_observation();
    assert_eq!(obs.player_id, 1);
    assert_eq!(obs.hands[0], vec![0, 1]);
    assert_eq!(obs.hands[1], vec![36]);
    assert_eq!(obs.discards[0], vec![4]);
    assert_eq!(obs.dora_indicators, vec![12]);
    assert_eq!(obs.honba, 2);
    assert_eq!(obs.riichi_sticks, 1);
    assert!(obs.is_tenpai);
}

#[test]
fn legal_action_helpers_clone_find_and_roundtrip_events() {
    let obs = sample_observation();
    let legal = obs.legal_actions_method();
    assert_eq!(legal.len(), 2);
    assert_eq!(obs.legal_actions_ref().len(), 2);

    let discard = obs.find_action(1).expect("discard action should be found");
    assert_eq!(discard.action_type, ActionType::Discard);
    assert_eq!(discard.tile, Some(4));
    assert!(obs.find_action(99).is_none());

    assert_eq!(
        obs.new_events(),
        vec!["event-a".to_string(), "event-b".to_string()]
    );
}

#[test]
fn base64_serialization_roundtrips_and_invalid_input_fails() {
    let obs = sample_observation();
    let encoded = obs.serialize_to_base64().expect("serialize observation");
    let decoded = Observation::deserialize_from_base64(&encoded).expect("deserialize observation");
    assert_eq!(decoded.player_id, obs.player_id);
    assert_eq!(decoded.scores, obs.scores);
    assert_eq!(decoded.waits, obs.waits);
    assert_eq!(decoded.last_discard, obs.last_discard);

    let err =
        Observation::deserialize_from_base64("not-base64").expect_err("invalid base64 should fail");
    assert!(matches!(err, RiichiError::Serialization { .. }));
}
