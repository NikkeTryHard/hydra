use crate::teacher::belief::{StageABeliefConfig, build_stage_a_teacher};
use hydra_core::action::{AKA_5M, AKA_5P, AKA_5S, DISCARD_END, HYDRA_ACTION_SPACE};
use hydra_core::bridge::{
    extract_discards, extract_dora, extract_hand, extract_melds, extract_public_remaining_counts,
};
use hydra_core::safety::SafetyInfo;
use riichienv_core::observation::Observation;
use riichienv_core::shanten::calc_shanten_from_counts;
use riichienv_core::state::GameState;

#[inline]
fn tile136_to_type(tile136: u8) -> u8 {
    tile136 / 4
}

pub(crate) fn exact_waits(state: &GameState, player: usize) -> ([f32; 34], bool) {
    let mut counts = [0u8; 34];
    for &tile in state.players[player].hand_slice() {
        counts[tile136_to_type(tile) as usize] += 1;
    }
    let hand_total: u8 = counts.iter().sum();
    let tenpai = calc_shanten_from_counts(&counts, hand_total / 3) == 0;
    if !tenpai {
        return ([0.0; 34], false);
    }

    let mut waits = [0.0; 34];
    for tile in 0..34usize {
        if counts[tile] >= 4 {
            continue;
        }
        counts[tile] += 1;
        let complete = calc_shanten_from_counts(&counts, (hand_total + 1) / 3) == -1;
        counts[tile] -= 1;
        if complete {
            waits[tile] = 1.0;
        }
    }

    let furiten = state.players[player]
        .discards_slice()
        .iter()
        .map(|&discard| tile136_to_type(discard) as usize)
        .any(|tile| waits[tile] > 0.0);
    if furiten {
        waits.fill(0.0);
    }
    (waits, true)
}

pub(crate) fn bool_mask_to_f32(mask: [bool; HYDRA_ACTION_SPACE]) -> [f32; HYDRA_ACTION_SPACE] {
    mask.map(|is_legal| if is_legal { 1.0 } else { 0.0 })
}

fn public_safety_score(safety: &SafetyInfo, tile: u8) -> f32 {
    let t = tile as usize;
    let mut score = 0.0f32;
    for opp in 0..3usize {
        if hydra_core::safety::bit_test(safety.genbutsu_all[opp], t) {
            score += 1.0;
        }
        score += 0.35 * safety.suji[opp][t];
        if hydra_core::safety::bit_test(safety.half_suji[opp], t) {
            score += 0.1;
        }
        score -= 0.25 * safety.matagi[opp][t];
        if safety.opponent_riichi[opp] || safety.cached_tenpai_prob[opp] > 0.5 {
            score -= 0.1;
        }
    }
    if hydra_core::safety::bit_test(safety.kabe, t) {
        score += 0.4;
    }
    if hydra_core::safety::bit_test(safety.one_chance, t) {
        score += 0.2;
    }
    score.clamp(0.0, 1.0)
}

fn exact_dealin_event_from_waits(wait_sets: &[[f32; 34]; 3], tile: u8) -> f32 {
    let t = tile as usize;
    if wait_sets.iter().any(|waits| waits[t] > 0.0) {
        1.0
    } else {
        0.0
    }
}

pub(crate) fn build_safety_residual_targets(
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
    safety: &SafetyInfo,
    wait_sets: &[[f32; 34]; 3],
) -> ([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE]) {
    let mut target = [0.0f32; HYDRA_ACTION_SPACE];
    let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
    for action in 0..=DISCARD_END {
        let action_idx = action as usize;
        if legal_mask[action_idx] <= 0.0 {
            continue;
        }
        let tile = match action {
            AKA_5M => 4,
            AKA_5P => 13,
            AKA_5S => 22,
            _ => action,
        };
        let public_score = public_safety_score(safety, tile);
        let exact_dealin = exact_dealin_event_from_waits(wait_sets, tile);
        let exact_safety = 1.0 - exact_dealin;
        target[action_idx] = exact_safety - public_score;
        mask[action_idx] = 1.0;
    }
    (target, mask)
}

pub(crate) fn build_stage_a_belief_targets(
    state: &GameState,
    actor: usize,
    obs: &Observation,
) -> (Option<[f32; 16 * 34]>, Option<[f32; 4]>, bool, bool) {
    let hand = extract_hand(obs);
    let discards = extract_discards(obs);
    let melds = extract_melds(obs);
    let dora = extract_dora(obs);
    let remaining = extract_public_remaining_counts(&hand, &discards, &melds, &dora);
    let hidden_counts = [
        state.players[(actor + 1) % 4].hand_len as usize,
        state.players[(actor + 2) % 4].hand_len as usize,
        state.players[(actor + 3) % 4].hand_len as usize,
        state.wall.remaining(),
    ];
    let target = build_stage_a_teacher(&remaining, &hidden_counts, StageABeliefConfig::default());
    match target {
        Some(target) => (
            Some(target.belief_fields),
            target.mixture_weights,
            true,
            target.mixture_weights.is_some(),
        ),
        None => (None, None, false, false),
    }
}

#[cfg(test)]
mod tests {
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
}
