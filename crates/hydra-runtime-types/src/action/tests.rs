use super::*;

#[test]
fn action_space_and_key_indices_are_frozen() {
    assert_eq!(HYDRA_ACTION_SPACE, 46);
    assert_eq!(DISCARD_START, 0);
    assert_eq!(DISCARD_END, 36);
    assert_eq!(AKA_5M, 34);
    assert_eq!(AKA_5P, 35);
    assert_eq!(AKA_5S, 36);
    assert_eq!(RIICHI, 37);
    assert_eq!(CHI_LEFT, 38);
    assert_eq!(CHI_MID, 39);
    assert_eq!(CHI_RIGHT, 40);
    assert_eq!(PON, 41);
    assert_eq!(KAN, 42);
    assert_eq!(AGARI, 43);
    assert_eq!(RYUUKYOKU, 44);
    assert_eq!(PASS, 45);
}

#[test]
fn action_validation_and_discard_mapping_match_abi() {
    for id in 0..HYDRA_ACTION_SPACE as u8 {
        let action = HydraAction::new(id).expect("action id should be valid");
        assert_eq!(action.id(), id);
        assert_eq!(action.is_discard(), id <= DISCARD_END);
        assert_eq!(
            action.is_aka_discard(),
            matches!(id, AKA_5M | AKA_5P | AKA_5S)
        );
    }

    assert!(HydraAction::new(HYDRA_ACTION_SPACE as u8).is_none());
    assert_eq!(HydraAction::new(0).unwrap().discard_tile_type(), Some(0));
    assert_eq!(HydraAction::new(33).unwrap().discard_tile_type(), Some(33));
    assert_eq!(
        HydraAction::new(AKA_5M).unwrap().discard_tile_type(),
        Some(4)
    );
    assert_eq!(
        HydraAction::new(AKA_5P).unwrap().discard_tile_type(),
        Some(13)
    );
    assert_eq!(
        HydraAction::new(AKA_5S).unwrap().discard_tile_type(),
        Some(22)
    );
    assert_eq!(HydraAction::new(RIICHI).unwrap().discard_tile_type(), None);
    assert_eq!(HydraAction::new(PASS).unwrap().discard_tile_type(), None);
}
