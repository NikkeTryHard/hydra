use super::*;

fn dummy_ctx() -> GameContext {
    GameContext {
        last_discard: Some(0),
        phase: ActionPhase::Normal,
        hand: [0u8; 14],
        hand_len: 0,
    }
}

#[test]
fn action_space_abi_constants_are_frozen() {
    assert_eq!(HYDRA_ACTION_SPACE, 46);
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
fn hydra_action_valid_range() {
    for i in 0..46u8 {
        assert!(HydraAction::new(i).is_some(), "id {i} should be valid");
    }
    assert!(HydraAction::new(46).is_none());
    assert!(HydraAction::new(255).is_none());
}

#[test]
fn discard_tile_type_normal() {
    for i in 0..34u8 {
        let a = HydraAction::new(i).unwrap();
        assert!(a.is_discard());
        assert!(!a.is_aka_discard());
        assert_eq!(a.discard_tile_type(), Some(i));
    }
}

#[test]
fn discard_tile_type_aka() {
    let a34 = HydraAction::new(34).unwrap();
    assert!(a34.is_discard());
    assert!(a34.is_aka_discard());
    assert_eq!(a34.discard_tile_type(), Some(4)); // 5m

    let a35 = HydraAction::new(35).unwrap();
    assert_eq!(a35.discard_tile_type(), Some(13)); // 5p

    let a36 = HydraAction::new(36).unwrap();
    assert_eq!(a36.discard_tile_type(), Some(22)); // 5s
}

#[test]
fn non_discard_has_no_tile_type() {
    for i in 37..46u8 {
        let a = HydraAction::new(i).unwrap();
        assert!(!a.is_discard());
        assert_eq!(a.discard_tile_type(), None);
    }
}

#[test]
fn roundtrip_pass() {
    let pass = Action::new(ActionType::Pass, None, &[], None);
    let hydra = riichienv_to_hydra(&pass).unwrap();
    assert_eq!(hydra.id(), PASS);
    let back = hydra_to_riichienv(hydra, &dummy_ctx()).unwrap();
    assert_eq!(back.action_type, ActionType::Pass);
}

#[test]
fn roundtrip_agari() {
    // Tsumo -> AGARI -> Tsumo (default)
    let tsumo = Action::new(ActionType::Tsumo, None, &[], None);
    let hydra = riichienv_to_hydra(&tsumo).unwrap();
    assert_eq!(hydra.id(), AGARI);

    // Ron also maps to AGARI
    let ron = Action::new(ActionType::Ron, None, &[], None);
    let hydra_ron = riichienv_to_hydra(&ron).unwrap();
    assert_eq!(hydra_ron.id(), AGARI);
}

#[test]
fn discard_normal_roundtrip() {
    // Discard 1m (type 0, 136-format = 0)
    let discard = Action::new(ActionType::Discard, Some(0), &[], None);
    let hydra = riichienv_to_hydra(&discard).unwrap();
    assert_eq!(hydra.id(), 0);
}

#[test]
fn discard_aka_roundtrip() {
    // Aka 5m: 136-index 16 -> Hydra 34
    let discard = Action::new(ActionType::Discard, Some(16), &[], None);
    let hydra = riichienv_to_hydra(&discard).unwrap();
    assert_eq!(hydra.id(), AKA_5M);
    assert!(hydra.is_aka_discard());

    // Roundtrip back: Hydra 34 -> 136-index 16
    let back = hydra_to_riichienv(hydra, &dummy_ctx()).unwrap();
    assert_eq!(back.tile, Some(AKA_MANZU_136));
}

#[test]
fn chi_variant_encoding() {
    // Called tile is lowest (left chi): e.g. call 3m, consume 4m+5m
    // 3m=type2, 4m=type3, 5m=type4 -> sorted [2,3,4], target=2 -> CHI_LEFT
    let chi = Action::new(
        ActionType::Chi,
        Some(2 * 4), // 3m in 136-format
        &[3 * 4, 4 * 4],
        None,
    );
    let hydra = riichienv_to_hydra(&chi).unwrap();
    assert_eq!(hydra.id(), CHI_LEFT);

    // Called tile is middle
    let chi_mid = Action::new(
        ActionType::Chi,
        Some(3 * 4), // 4m
        &[2 * 4, 4 * 4],
        None,
    );
    assert_eq!(riichienv_to_hydra(&chi_mid).unwrap().id(), CHI_MID);

    // Called tile is highest
    let chi_right = Action::new(
        ActionType::Chi,
        Some(4 * 4), // 5m
        &[2 * 4, 3 * 4],
        None,
    );
    assert_eq!(riichienv_to_hydra(&chi_right).unwrap().id(), CHI_RIGHT);
}

#[test]
fn kan_variants_all_map_to_kan() {
    let daiminkan = Action::new(ActionType::Daiminkan, Some(0), &[], None);
    assert_eq!(riichienv_to_hydra(&daiminkan).unwrap().id(), KAN);

    let ankan = Action::new(ActionType::Ankan, None, &[0, 1, 2, 3], None);
    assert_eq!(riichienv_to_hydra(&ankan).unwrap().id(), KAN);

    let kakan = Action::new(ActionType::Kakan, None, &[0], None);
    assert_eq!(riichienv_to_hydra(&kakan).unwrap().id(), KAN);
}

#[test]
fn legal_mask_basic() {
    let actions = vec![
        Action::new(ActionType::Discard, Some(0), &[], None), // 1m -> idx 0
        Action::new(ActionType::Discard, Some(16), &[], None), // aka 5m -> idx 34
        Action::new(ActionType::Pass, None, &[], None),       // -> idx 45
    ];
    let mask = build_legal_mask(&actions, ActionPhase::Normal);
    assert!(mask[0]);
    assert!(mask[34]);
    assert!(mask[45]);
    // Everything else should be false
    assert!(!mask[1]);
    assert!(!mask[37]);
}

#[test]
fn discard_5m_is_not_aka() {
    // Normal 5m (Hydra id=4) must NOT map to the aka 136-index (16)
    let hydra = HydraAction::new(4).unwrap();
    let action = hydra_to_riichienv(hydra, &dummy_ctx()).unwrap();
    let tile136 = action.tile.unwrap();
    assert_ne!(tile136, 16, "normal 5m must not use aka 136-index");
    assert_eq!(tile136 / 4, 4, "must still be tile type 5m");
}

#[test]
fn discard_aka_5m_is_aka() {
    // Aka 5m (Hydra id=34) MUST map to aka 136-index (16)
    let hydra = HydraAction::new(34).unwrap();
    let action = hydra_to_riichienv(hydra, &dummy_ctx()).unwrap();
    assert_eq!(action.tile.unwrap(), 16);
}

#[test]
fn all_five_tiles_avoid_aka_collision() {
    // Check 5m, 5p, 5s normal discards
    for (hydra_id, aka_136) in [(4u8, 16u8), (13, 52), (22, 88)] {
        let hydra = HydraAction::new(hydra_id).unwrap();
        let action = hydra_to_riichienv(hydra, &dummy_ctx()).unwrap();
        let tile136 = action.tile.unwrap();
        assert_ne!(
            tile136, aka_136,
            "normal discard of type {} must not use aka 136-index {}",
            hydra_id, aka_136,
        );
        assert_eq!(
            tile136 / 4,
            hydra_id,
            "tile type must still be {}",
            hydra_id,
        );
    }
}

#[test]
fn legal_mask_riichi_select_filters_non_discards() {
    let actions = vec![
        Action::new(ActionType::Discard, Some(0), &[], None),
        Action::new(ActionType::Discard, Some(16), &[], None),
        Action::new(ActionType::Pass, None, &[], None),
        Action::new(ActionType::Tsumo, None, &[], None),
    ];
    let mask = build_legal_mask(&actions, ActionPhase::RiichiSelect);
    assert!(mask[0]); // discard 1m allowed
    assert!(mask[34]); // discard aka 5m allowed
    assert!(!mask[45]); // pass NOT allowed in riichi select
    assert!(!mask[43]); // agari NOT allowed in riichi select
}

#[test]
fn chi_left_resolves_consume_tiles() {
    // Chi left (38): called tile is lowest (1m=type0), need type1 + type2 from hand
    // Discard: tile 0 (type 0, 1m). Hand has tiles 4 (type 1, 2m) and 8 (type 2, 3m)
    let mut hand = [0u8; 14];
    hand[..4].copy_from_slice(&[4, 8, 20, 24]);
    let ctx = GameContext {
        last_discard: Some(0),
        phase: ActionPhase::Normal,
        hand,
        hand_len: 4,
    };
    let hydra = HydraAction::new(CHI_LEFT).unwrap();
    let action = hydra_to_riichienv(hydra, &ctx).unwrap();
    assert_eq!(action.action_type, ActionType::Chi);
    assert_eq!(action.tile, Some(0));
    assert_eq!(action.consume_slice(), &[4, 8]);
}

#[test]
fn chi_left_prefers_first_matching_tiles_when_duplicates_exist() {
    let mut hand = [0u8; 14];
    hand[..6].copy_from_slice(&[5, 4, 9, 8, 20, 24]);
    let ctx = GameContext {
        last_discard: Some(0),
        phase: ActionPhase::Normal,
        hand,
        hand_len: 6,
    };
    let hydra = HydraAction::new(CHI_LEFT).unwrap();
    let action = hydra_to_riichienv(hydra, &ctx).unwrap();
    assert_eq!(action.action_type, ActionType::Chi);
    assert_eq!(action.consume_slice(), &[5, 9]);
}

#[test]
fn chi_edges_reject_out_of_suit_without_wrapping() {
    let mut hand = [0u8; 14];
    hand[..4].copy_from_slice(&[0, 4, 8, 12]);

    for (called_type, action_id) in [
        (0u8, CHI_MID),
        (1, CHI_RIGHT),
        (8, CHI_LEFT),
        (27, CHI_LEFT),
    ] {
        let ctx = GameContext {
            last_discard: Some(called_type * 4),
            phase: ActionPhase::Normal,
            hand,
            hand_len: 4,
        };
        let err = hydra_to_riichienv(HydraAction::new(action_id).unwrap(), &ctx).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("chi")
                && (msg.contains("out of suit range") || msg.contains("suited tile")),
            "unexpected error for called_type={called_type}, action_id={action_id}: {msg}"
        );
    }
}

#[test]
fn chi_mid_at_suit_boundary_is_rejected() {
    let mut hand = [0u8; 14];
    hand[..4].copy_from_slice(&[7 * 4, 9 * 4, 10 * 4, 11 * 4]);
    let ctx = GameContext {
        last_discard: Some(8 * 4),
        phase: ActionPhase::Normal,
        hand,
        hand_len: 4,
    };

    let err = hydra_to_riichienv(HydraAction::new(CHI_MID).unwrap(), &ctx).unwrap_err();
    assert!(err.to_string().contains("out of suit range"));
}

#[test]
fn agari_resolves_to_tsumo_in_normal_phase() {
    let ctx = GameContext {
        last_discard: None,
        phase: ActionPhase::Normal,
        hand: [0u8; 14],
        hand_len: 0,
    };
    let hydra = HydraAction::new(AGARI).unwrap();
    let action = hydra_to_riichienv(hydra, &ctx).unwrap();
    assert_eq!(action.action_type, ActionType::Tsumo);
}

#[test]
fn agari_resolves_to_ron_in_response_phase() {
    let ctx = GameContext {
        last_discard: Some(0),
        phase: ActionPhase::RiichiSelect, // non-Normal = response
        hand: [0u8; 14],
        hand_len: 0,
    };
    let hydra = HydraAction::new(AGARI).unwrap();
    let action = hydra_to_riichienv(hydra, &ctx).unwrap();
    assert_eq!(action.action_type, ActionType::Ron);
}
