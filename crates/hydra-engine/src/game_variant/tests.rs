use super::*;

#[test]
fn new_uses_four_player_state_for_standard_modes() {
    let variant = GameStateVariant::new(0, true, Some(7), 0, GameRule::default_tenhou());
    assert!(matches!(variant, GameStateVariant::FourPlayer(_)));
    assert_eq!(variant.num_players(), 4);
    assert!(!variant.is_three_player());
}

#[test]
fn new_uses_three_player_state_for_sanma_modes() {
    let variant = GameStateVariant::new(3, true, Some(11), 0, GameRule::default_tenhou_sanma());
    assert!(matches!(variant, GameStateVariant::ThreePlayer(_)));
    assert_eq!(variant.num_players(), 3);
    assert!(variant.is_three_player());
}
