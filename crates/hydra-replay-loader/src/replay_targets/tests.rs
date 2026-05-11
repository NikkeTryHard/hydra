use super::*;
use hydra_core::action::{AKA_5M, AKA_5P, AKA_5S};

#[test]
fn bool_mask_to_f32_preserves_legality_positions() {
    let mut mask = [false; HYDRA_ACTION_SPACE];
    mask[0] = true;
    mask[AKA_5M as usize] = true;
    mask[45] = true;

    let converted = bool_mask_to_f32(mask);
    assert_eq!(converted[0], 1.0);
    assert_eq!(converted[AKA_5M as usize], 1.0);
    assert_eq!(converted[45], 1.0);
    assert_eq!(converted[1], 0.0);
}

#[test]
fn build_safety_residual_targets_only_fills_legal_discards_and_remaps_aka_tiles() {
    let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
    legal_mask[1] = 1.0;
    legal_mask[AKA_5M as usize] = 1.0;
    legal_mask[AKA_5P as usize] = 1.0;
    legal_mask[AKA_5S as usize] = 1.0;
    legal_mask[(DISCARD_END as usize) + 1] = 1.0;

    let mut safety = SafetyInfo::new();
    hydra_core::safety::bit_set(&mut safety.genbutsu_all[0], 1);
    hydra_core::safety::bit_set(&mut safety.genbutsu_all[1], 4);
    hydra_core::safety::bit_set(&mut safety.kabe, 13);
    hydra_core::safety::bit_set(&mut safety.one_chance, 22);

    let mut wait_sets = [[0.0f32; 34]; 3];
    wait_sets[0][4] = 1.0;

    let (target, mask) = build_safety_residual_targets(&legal_mask, &safety, &wait_sets);

    assert_eq!(mask[1], 1.0);
    assert_eq!(mask[AKA_5M as usize], 1.0);
    assert_eq!(mask[AKA_5P as usize], 1.0);
    assert_eq!(mask[AKA_5S as usize], 1.0);
    assert_eq!(mask[(DISCARD_END as usize) + 1], 0.0);

    assert!(
        target[AKA_5M as usize] < target[1],
        "aka 5m should be penalized relative to a neutral legal discard when exact waits hit"
    );
    assert!(
        target[AKA_5M as usize] < 0.0,
        "aka 5m should reuse tile 4 and become dangerous when exact waits hit"
    );
    assert!(
        target[AKA_5P as usize] > 0.0,
        "aka 5p should reuse tile 13 and receive kabe safety signal"
    );
    assert!(
        target[AKA_5S as usize] > 0.0,
        "aka 5s should reuse tile 22 and receive one-chance safety signal"
    );
}

#[test]
fn exact_dealin_event_reports_if_any_wait_set_contains_tile() {
    let mut wait_sets = [[0.0f32; 34]; 3];
    wait_sets[1][7] = 1.0;
    assert_eq!(exact_dealin_event_from_waits(&wait_sets, 7), 1.0);
    assert_eq!(exact_dealin_event_from_waits(&wait_sets, 8), 0.0);
}
