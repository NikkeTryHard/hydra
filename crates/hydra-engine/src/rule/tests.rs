use super::*;

#[test]
fn default_rules_match_expected_open_kan_and_draw_policies() {
    let tenhou = GameRule::default_tenhou();
    let mortal = GameRule::default_mortal();
    let mjsoul = GameRule::default_mjsoul();

    assert!(tenhou.open_kan_dora_after_discard);
    assert!(tenhou.sanchaho_is_draw);
    assert!(!mortal.open_kan_dora_after_discard);
    assert!(mortal.sanchaho_is_draw);
    assert!(mjsoul.open_kan_dora_after_discard);
    assert!(!mjsoul.sanchaho_is_draw);
    assert!(mjsoul.is_daisuushii_double);
    assert!(mjsoul.yakuman_pao_is_liability_only);
}

#[test]
fn sanma_defaults_keep_three_player_specific_policy_split() {
    let tenhou = GameRule::default_tenhou_sanma();
    let mortal = GameRule::default_mortal_sanma();
    let mjsoul = GameRule::default_mjsoul_sanma();

    assert!(tenhou.open_kan_dora_after_discard);
    assert!(!mortal.open_kan_dora_after_discard);
    assert!(mjsoul.open_kan_dora_after_discard);

    assert!(!tenhou.sanchaho_is_draw);
    assert!(!mortal.sanchaho_is_draw);
    assert!(!mjsoul.sanchaho_is_draw);
    assert!(mjsoul.allows_ron_on_ankan_for_kokushi_musou);
    assert!(!mortal.allows_ron_on_ankan_for_kokushi_musou);
}

#[test]
fn default_impl_matches_mortal_rule_set() {
    assert_eq!(GameRule::default(), GameRule::default_mortal());
}
