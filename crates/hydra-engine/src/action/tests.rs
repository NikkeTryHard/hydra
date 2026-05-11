use super::*;

#[test]
fn action_new_sorts_and_truncates_consumed_tiles() {
    let action = Action::new(ActionType::Chi, Some(8), &[12, 4, 8, 0, 16], Some(2));
    assert_eq!(action.consume_count, 4);
    assert_eq!(action.consume_slice(), &[0, 4, 8, 12]);
}

#[test]
fn to_mjai_and_repr_include_expected_fields() {
    let action = Action::new(ActionType::Pon, Some(52), &[53, 54], Some(1));
    let mjai = action.to_mjai();
    assert!(mjai.contains("\"type\":\"pon\""));
    assert!(mjai.contains("\"actor\":1"));
    assert!(mjai.contains("5pr"));

    let repr = action.repr();
    assert!(repr.contains("Action(action_type=Pon"));
    assert!(repr.contains("actor=Some(1)"));
}

#[test]
fn four_player_encode_covers_required_action_variants() {
    assert_eq!(
        Action::new(ActionType::Discard, Some(8), &[], None)
            .encode()
            .unwrap(),
        2
    );
    assert_eq!(
        Action::new(ActionType::Riichi, None, &[], None)
            .encode()
            .unwrap(),
        37
    );
    assert_eq!(
        Action::new(ActionType::Chi, Some(8), &[0, 4], None)
            .encode()
            .unwrap(),
        40
    );
    assert_eq!(
        Action::new(ActionType::Pon, Some(52), &[53, 54], None)
            .encode()
            .unwrap(),
        41
    );
    assert_eq!(
        Action::new(ActionType::Pass, None, &[], None)
            .encode()
            .unwrap(),
        81
    );
    assert!(Action::new(ActionType::Discard, None, &[], None)
        .encode()
        .is_err());
}

#[test]
fn three_player_encoder_rejects_chi_and_middle_manzu_discards() {
    let encoder = ActionEncoder::ThreePlayer;
    let chi = Action::new(ActionType::Chi, Some(8), &[0, 4], None);
    assert!(encoder.encode(&chi).is_err());

    let invalid_discard = Action::new(ActionType::Discard, Some(4), &[], None);
    let err = encoder
        .encode(&invalid_discard)
        .expect_err("2m should be invalid in sanma");
    assert!(matches!(err, RiichiError::InvalidAction { .. }));
}

#[test]
fn three_player_encoder_maps_valid_actions_and_helpers() {
    let encoder = ActionEncoder::from_num_players(3);
    assert_eq!(encoder.action_space_size(), ACTION_SPACE_3P);

    let discard = Action::new(ActionType::Discard, Some(0), &[], None);
    assert_eq!(encoder.encode(&discard).unwrap(), 0);

    let kita = Action::new(ActionType::Kita, None, &[], None);
    assert_eq!(encoder.encode(&kita).unwrap(), 59);

    let daiminkan = Action::new(ActionType::Daiminkan, Some(108), &[], None);
    assert_eq!(encoder.encode(&daiminkan).unwrap(), 49);

    let kakan = Action::new(ActionType::Kakan, None, &[108, 109, 110, 111], None);
    assert_eq!(encoder.encode(&kakan).unwrap(), 49);
}
