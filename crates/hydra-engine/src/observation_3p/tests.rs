use super::*;
use crate::action::{Action, ActionType};

fn sample_observation() -> Observation3P {
    Observation3P::new(
        2,
        [vec![0, 1], vec![36], vec![72]],
        Default::default(),
        [vec![4], vec![], vec![]],
        vec![12],
        [35_000, 28_000, 37_000],
        [false, true, false],
        vec![
            Action::new(ActionType::Discard, Some(0), &[], Some(2)),
            Action::new(ActionType::Kita, None, &[], Some(2)),
        ],
        vec!["event-a".to_string(), "event-b".to_string()],
        1,
        2,
        27,
        1,
        4,
        vec![3, 6],
        true,
        [None, Some(4), None],
        [Some(0), None, None],
        Some(4),
    )
}

#[test]
fn new_converts_input_collections_and_preserves_metadata() {
    let obs = sample_observation();
    assert_eq!(obs.player_id, 2);
    assert_eq!(obs.hands[0], vec![0, 1]);
    assert_eq!(obs.hands[2], vec![72]);
    assert_eq!(obs.discards[0], vec![4]);
    assert_eq!(obs.dora_indicators, vec![12]);
    assert_eq!(obs.riichi_sticks, 2);
    assert!(obs.is_tenpai);
}

#[test]
fn legal_action_helpers_and_base64_roundtrip_work_in_three_player_mode() {
    let obs = sample_observation();
    let legal = obs.legal_actions_method();
    assert_eq!(legal.len(), 2);

    let discard = obs.find_action(0).expect("discard action should be found");
    assert_eq!(discard.action_type, ActionType::Discard);
    assert!(obs.find_action(999).is_none());
    assert_eq!(obs.new_events().len(), 2);

    let encoded = obs.serialize_to_base64().expect("serialize observation3p");
    let decoded =
        Observation3P::deserialize_from_base64(&encoded).expect("deserialize observation3p");
    assert_eq!(decoded.player_id, obs.player_id);
    assert_eq!(decoded.scores, obs.scores);
    assert_eq!(decoded.last_discard, obs.last_discard);
}

#[test]
fn invalid_base64_returns_serialization_error() {
    let err = Observation3P::deserialize_from_base64("not-base64").expect_err("invalid base64");
    assert!(matches!(err, RiichiError::Serialization { .. }));
}
