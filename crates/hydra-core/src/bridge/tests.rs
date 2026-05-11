use super::*;
use riichienv_core::action::{Action, ActionType};
use riichienv_core::rule::GameRule;
use riichienv_core::state::GameState;

/// Create a fresh observation from a newly dealt game.
fn fresh_obs() -> Observation {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(42), 0, rule);
    state.get_observation(0)
}

#[test]
fn extract_hand_has_13_or_14_tiles() {
    let obs = fresh_obs();
    let hand = extract_hand(&obs);
    let total: u8 = hand.iter().sum();
    assert!(
        (13..=14).contains(&total),
        "hand has {total} tiles, expected 13 or 14",
    );
}

#[test]
fn extract_discards_initially_empty() {
    let obs = fresh_obs();
    let discards = extract_discards(&obs);
    for pd in &discards {
        assert_eq!(pd.len, 0);
    }
}

#[test]
fn extract_discards_ref_matches_owned_observation_tedashi_flags() {
    let rule = GameRule::default_tenhou();
    let mut state = GameState::new(0, true, Some(42), 0, rule);
    let pid = state.current_player;
    if let Some(tile136) = state.players[pid as usize].hand_slice().first().copied() {
        let mut actions = [None; 4];
        actions[pid as usize] = Some(Action::new(ActionType::Discard, Some(tile136), &[], None));
        state.step_unchecked(&actions);
    }

    let owned = extract_discards(&state.get_observation(state.current_player));
    let observed = state.observe(state.current_player);
    let borrowed = extract_discards_ref(&observed);

    for rel in 0..4 {
        assert_eq!(owned[rel].len, borrowed[rel].len);
        for idx in 0..owned[rel].len as usize {
            assert_eq!(
                owned[rel].as_slice()[idx].tile,
                borrowed[rel].as_slice()[idx].tile
            );
            assert_eq!(
                owned[rel].as_slice()[idx].is_tedashi,
                borrowed[rel].as_slice()[idx].is_tedashi
            );
        }
    }
}

#[test]
fn extract_melds_initially_empty() {
    let obs = fresh_obs();
    let melds = extract_melds(&obs);
    for player_melds in &melds {
        assert_eq!(player_melds.len, 0);
    }
}

#[test]
fn extract_dora_has_one_indicator() {
    let obs = fresh_obs();
    let dora = extract_dora(&obs);
    assert_eq!(dora.indicator_count, 1, "initial game has 1 dora indicator");
    assert!(dora.indicators[0] < 34, "tile type must be 0-33");
}

#[test]
fn extract_metadata_sane_values() {
    let obs = fresh_obs();
    let hand = extract_hand(&obs);
    let meta = extract_metadata(&obs, &hand);
    assert_eq!(meta.kyoku_index, obs.kyoku_index);
    assert_eq!(meta.honba, 0);
    assert_eq!(meta.kyotaku, 0);
    // Shanten for a dealt hand should be reasonable (-1 to 8)
    assert!(
        (-1..=8).contains(&meta.shanten),
        "shanten {} out of range",
        meta.shanten,
    );
}

#[test]
fn extract_observer_meld_counts_initially_zero() {
    let obs = fresh_obs();
    let counts = extract_observer_meld_counts(&obs);
    assert_eq!(counts.iter().sum::<u8>(), 0, "no melds at game start");
}

#[test]
fn encode_observation_produces_nonzero() {
    let obs = fresh_obs();
    let safety = SafetyInfo::new();
    let mut encoder = ObservationEncoder::new();
    let result = encode_observation(&mut encoder, &obs, &safety, None);
    let nonzero = result.iter().filter(|&&v| v != 0.0).count();
    assert!(
        nonzero > 0,
        "encoded observation should have nonzero values"
    );
}

#[test]
fn public_remaining_counts_subtract_visible_tiles() {
    let mut hand = [0u8; NUM_TILE_TYPES];
    hand[0] = 2;
    hand[1] = 1;

    let mut discards = std::array::from_fn(|_| PlayerDiscards::new());
    discards[0].push(DiscardEntry {
        tile: 0,
        is_tedashi: true,
        turn: 0,
    });

    let mut melds = std::array::from_fn(|_| PlayerMelds::new());
    melds[1].push(MeldInfo {
        tiles: [1, 1, 1, 0],
        tile_count: 3,
        meld_type: MeldType::Pon,
    });

    let dora = DoraInfo {
        indicators: [0, 0, 0, 0, 0],
        indicator_count: 1,
        aka_flags: [false; 3],
    };

    let remaining = extract_public_remaining_counts(&hand, &discards, &melds, &dora);
    assert_eq!(
        remaining[0], 0.0,
        "2 in hand + 1 discard + 1 dora indicator exhaust tile 0"
    );
    assert_eq!(remaining[1], 0.0, "1 in hand + pon exhaust tile 1");
    assert_eq!(
        remaining[2], 4.0,
        "unseen tile should keep full remaining count"
    );
}

#[test]
fn compute_public_hand_ev_on_real_observation_has_signal() {
    let obs = fresh_obs();
    let hand = extract_hand(&obs);
    let discards = extract_discards(&obs);
    let melds = extract_melds(&obs);
    let dora = extract_dora(&obs);
    let hand_ev = compute_public_hand_ev(&hand, &discards, &melds, &dora);

    let any_tenpai = hand_ev
        .tenpai_prob
        .iter()
        .flat_map(|p| p.iter())
        .any(|&v| v > 0.0);
    let any_ukeire = hand_ev
        .ukeire
        .iter()
        .flat_map(|u| u.iter())
        .any(|&v| v > 0.0);

    assert!(
        any_tenpai || any_ukeire,
        "public Hand-EV should expose some nonzero signal"
    );
}

#[test]
fn extract_ct_smc_remaining_counts_uses_wall_column_only() {
    let mut smc = CtSmc::new(crate::ct_smc::CtSmcConfig::default().with_particles(2));
    smc.particles = vec![
        crate::ct_smc::Particle {
            allocation: {
                let mut allocation = [[0u8; 4]; 34];
                allocation[3] = [1, 1, 0, 0];
                allocation[7] = [0, 0, 1, 0];
                allocation
            },
            log_weight: 0.0,
        },
        crate::ct_smc::Particle {
            allocation: {
                let mut allocation = [[0u8; 4]; 34];
                allocation[3] = [0, 1, 1, 1];
                allocation
            },
            log_weight: 0.0,
        },
    ];

    let remaining = extract_ct_smc_remaining_counts(&smc);
    assert!((remaining[3] - 0.5).abs() < 1e-6);
    assert_eq!(remaining[7], 0.0);
    assert_eq!(remaining[2], 0.0);
}

#[test]
fn compute_ct_smc_hand_ev_uses_weighted_remaining_counts() {
    let mut hand = [0u8; NUM_TILE_TYPES];
    hand[0] = 1;
    hand[1] = 1;

    let mut smc = CtSmc::new(crate::ct_smc::CtSmcConfig::default().with_particles(2));
    smc.particles = vec![
        crate::ct_smc::Particle {
            allocation: {
                let mut allocation = [[0u8; 4]; 34];
                allocation[0] = [1, 0, 0, 0];
                allocation
            },
            log_weight: 0.0,
        },
        crate::ct_smc::Particle {
            allocation: {
                let mut allocation = [[0u8; 4]; 34];
                allocation[0] = [0, 0, 0, 1];
                allocation
            },
            log_weight: 0.0,
        },
    ];

    let features = compute_ct_smc_hand_ev(&hand, &smc);
    assert!(features.ukeire[1][0] > 0.0);
    assert!(features.expected_score[1] > 0.0);
}

#[test]
fn build_search_features_from_mixture_populates_belief_and_weights() {
    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0f64; 4];
    let mut mixture = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
    mixture.bayesian_update(&[1.5, 0.5, -0.5, -1.5]);

    let mut safety = SafetyInfo::new();
    safety.set_tenpai_prediction(0, 0.6);
    safety.on_discard(5, 1, true);

    let context = SearchContext {
        mixture: Some(&mixture),
        ..SearchContext::default()
    };
    let features = build_search_features(&safety, &context);

    assert!(features.belief_features_present);
    assert!(features.context_features_present);
    assert!(features.mixture_weights.iter().any(|&v| v > 0.0));
    assert!(features.mixture_entropy > 0.0);
    assert!(features.mixture_ess > 0.0);
    assert!(features.belief_fields.iter().flatten().any(|&v| v > 0.0));
    assert!(features.opponent_risk[1][4] > 0.0 || features.opponent_risk[1][6] > 0.0);
    assert!((features.opponent_stress[0] - 0.6).abs() < 1e-6);
}

#[test]
fn build_search_features_from_afbs_populates_delta_q() {
    let mut tree = AfbsTree::new();
    let root = tree.add_node(100, 1.0, false);
    tree.nodes[root as usize].visit_count = 10;
    tree.nodes[root as usize].total_value = 4.0; // q = 0.4

    let child_a = tree.add_node(101, 0.6, false);
    tree.nodes[child_a as usize].visit_count = 4;
    tree.nodes[child_a as usize].total_value = 3.2; // q = 0.8

    let child_b = tree.add_node(105, 0.4, false);
    tree.nodes[child_b as usize].visit_count = 4;
    tree.nodes[child_b as usize].total_value = 0.4; // q = 0.1

    tree.nodes[root as usize].children = vec![(0, child_a), (5, child_b)].into();

    let context = SearchContext {
        afbs_tree: Some(&tree),
        afbs_root: Some(root),
        ..SearchContext::default()
    };
    let features = build_search_features(&SafetyInfo::new(), &context);

    assert!(features.search_features_present);
    assert!(features.context_features_present);
    assert!((features.delta_q[0] - 0.4).abs() < 1e-6);
    assert!((features.delta_q[5] + 0.3).abs() < 1e-6);
}

#[test]
fn encode_observation_populates_hand_ev_planes() {
    let obs = fresh_obs();
    let safety = SafetyInfo::new();
    let mut encoder = ObservationEncoder::new();
    let result = encode_observation(&mut encoder, &obs, &safety, None);

    let mask_offset = crate::encoder::HAND_EV_MASK_CHANNEL * NUM_TILE_TYPES;
    assert_eq!(
        result[mask_offset], 1.0,
        "Hand-EV presence mask should be enabled"
    );

    let hand_ev_payload =
        &result[crate::encoder::HAND_EV_CHANNEL_START * NUM_TILE_TYPES..mask_offset];
    let nonzero = hand_ev_payload.iter().filter(|&&v| v != 0.0).count();
    assert!(
        nonzero > 0,
        "encoded observation should contain nonzero Hand-EV payload"
    );
}

#[test]
fn encode_observation_bc_minimal_skips_hand_ev_planes() {
    let obs = fresh_obs();
    let safety = SafetyInfo::new();
    let mut encoder = ObservationEncoder::new();
    let result = encode_observation_with_profile(
        &mut encoder,
        &obs,
        &safety,
        None,
        BridgeEncodeProfile::bc_minimal(),
    );

    let mask_offset = crate::encoder::HAND_EV_MASK_CHANNEL * NUM_TILE_TYPES;
    assert_eq!(
        result[mask_offset], 0.0,
        "BC-minimal encode should leave Hand-EV mask disabled"
    );

    let hand_ev_payload =
        &result[crate::encoder::HAND_EV_CHANNEL_START * NUM_TILE_TYPES..mask_offset];
    assert!(
        hand_ev_payload.iter().all(|&v| v == 0.0),
        "BC-minimal encode should zero Hand-EV payload"
    );
}

#[test]
fn encode_observation_ref_bc_minimal_skips_hand_ev_planes() {
    let rule = GameRule::default_tenhou();
    let state = GameState::new(0, true, Some(42), 0, rule);
    let obs = state.observe(0);
    let safety = SafetyInfo::new();
    let mut encoder = ObservationEncoder::new();
    let result = encode_observation_ref_with_profile(
        &mut encoder,
        &obs,
        &safety,
        BridgeEncodeProfile::bc_minimal(),
    );

    let mask_offset = crate::encoder::HAND_EV_MASK_CHANNEL * NUM_TILE_TYPES;
    assert_eq!(
        result[mask_offset], 0.0,
        "BC-minimal ref encode should leave Hand-EV mask disabled"
    );

    let hand_ev_payload =
        &result[crate::encoder::HAND_EV_CHANNEL_START * NUM_TILE_TYPES..mask_offset];
    assert!(
        hand_ev_payload.iter().all(|&v| v == 0.0),
        "BC-minimal ref encode should zero Hand-EV payload"
    );
}

#[test]
fn reused_encoder_full_then_bc_minimal_clears_hand_ev_planes() {
    let obs = fresh_obs();
    let safety = SafetyInfo::new();
    let mut encoder = ObservationEncoder::new();

    let full = encode_observation_with_profile(
        &mut encoder,
        &obs,
        &safety,
        None,
        BridgeEncodeProfile::full(),
    );
    let full_mask_offset = crate::encoder::HAND_EV_MASK_CHANNEL * NUM_TILE_TYPES;
    assert_eq!(
        full[full_mask_offset], 1.0,
        "full encode should populate Hand-EV mask"
    );

    let minimal = encode_observation_with_profile(
        &mut encoder,
        &obs,
        &safety,
        None,
        BridgeEncodeProfile::bc_minimal(),
    );
    assert_eq!(
        minimal[full_mask_offset], 0.0,
        "reused encoder should clear Hand-EV mask on BC-minimal encode"
    );
    let hand_ev_payload =
        &minimal[crate::encoder::HAND_EV_CHANNEL_START * NUM_TILE_TYPES..full_mask_offset];
    assert!(
        hand_ev_payload.iter().all(|&v| v == 0.0),
        "reused encoder should clear stale Hand-EV payload on BC-minimal encode"
    );
}

#[test]
fn encode_observation_with_search_context_populates_group_c_planes() {
    let obs = fresh_obs();
    let mut safety = SafetyInfo::new();
    safety.set_tenpai_prediction(0, 0.7);
    safety.on_discard(5, 1, true);

    let kernel = [1.0f64; 136];
    let row_sums = [4.0f64; 34];
    let col_sums = [34.0f64; 4];
    let mut mixture = MixtureSib::new(4, &kernel, &row_sums, &col_sums);
    mixture.bayesian_update(&[1.0, 0.0, -0.5, -1.0]);

    let mut tree = AfbsTree::new();
    let root = tree.add_node(7, 1.0, false);
    tree.nodes[root as usize].visit_count = 10;
    tree.nodes[root as usize].total_value = 2.0;
    let child = tree.add_node(11, 1.0, false);
    tree.nodes[child as usize].visit_count = 5;
    tree.nodes[child as usize].total_value = 3.0;
    tree.nodes[root as usize].children = vec![(0, child)].into();

    let context = SearchContext {
        mixture: Some(&mixture),
        afbs_tree: Some(&tree),
        afbs_root: Some(root),
        ..SearchContext::default()
    };

    let mut encoder = ObservationEncoder::new();
    let result =
        encode_observation_with_search_context(&mut encoder, &obs, &safety, None, &context);

    let belief_mask = crate::encoder::SEARCH_MASK_CHANNEL_START * NUM_TILE_TYPES;
    let search_mask = (crate::encoder::SEARCH_MASK_CHANNEL_START + 1) * NUM_TILE_TYPES;
    let robust_mask = (crate::encoder::SEARCH_MASK_CHANNEL_START + 2) * NUM_TILE_TYPES;
    assert_eq!(result[belief_mask], 1.0);
    assert_eq!(result[search_mask], 1.0);
    assert_eq!(result[robust_mask], 1.0);

    let belief_payload = result[crate::encoder::SEARCH_BELIEF_CHANNEL_START * NUM_TILE_TYPES
        ..crate::encoder::SEARCH_DELTA_Q_CHANNEL * NUM_TILE_TYPES]
        .iter()
        .filter(|&&v| v != 0.0)
        .count();
    let delta_q_payload = result[crate::encoder::SEARCH_DELTA_Q_CHANNEL * NUM_TILE_TYPES];
    assert!(
        belief_payload > 0,
        "belief/search payload should be nonzero"
    );
    assert!(
        delta_q_payload > 0.0,
        "delta-q channel should reflect AFBS context"
    );
}

#[test]
fn encode_observation_with_ct_smc_context_uses_belief_weighted_hand_ev() {
    let obs = fresh_obs();
    let safety = SafetyInfo::new();
    let hand = extract_hand(&obs);

    let mut smc = CtSmc::new(crate::ct_smc::CtSmcConfig::default().with_particles(2));
    smc.particles = vec![
        crate::ct_smc::Particle {
            allocation: {
                let mut allocation = [[0u8; 4]; 34];
                for tile in 0..NUM_TILE_TYPES {
                    if hand[tile] == 0 {
                        allocation[tile][3] = 1;
                    }
                }
                allocation
            },
            log_weight: 0.0,
        },
        crate::ct_smc::Particle {
            allocation: {
                let mut allocation = [[0u8; 4]; 34];
                for tile in 0..NUM_TILE_TYPES {
                    if hand[tile] == 0 && tile % 2 == 0 {
                        allocation[tile][2] = 1;
                    }
                }
                allocation
            },
            log_weight: 0.0,
        },
    ];

    let context = SearchContext {
        ct_smc: Some(&smc),
        ..SearchContext::default()
    };

    let mut encoder = ObservationEncoder::new();
    let result =
        encode_observation_with_search_context(&mut encoder, &obs, &safety, None, &context);

    let hand_ev_payload = &result[crate::encoder::HAND_EV_CHANNEL_START * NUM_TILE_TYPES
        ..crate::encoder::HAND_EV_MASK_CHANNEL * NUM_TILE_TYPES];
    let nonzero = hand_ev_payload.iter().filter(|&&v| v != 0.0).count();
    assert!(
        nonzero > 0,
        "CT-SMC context should produce nonzero Hand-EV payload"
    );
}

#[test]
fn encode_observation_default_context_leaves_group_c_zero() {
    let obs = fresh_obs();
    let safety = SafetyInfo::new();
    let mut encoder = ObservationEncoder::new();
    let result = encode_observation_with_profile(
        &mut encoder,
        &obs,
        &safety,
        None,
        BridgeEncodeProfile::bc_minimal(),
    );

    let search_start = crate::encoder::SEARCH_CHANNEL_START * NUM_TILE_TYPES;
    let search_end = crate::encoder::HAND_EV_CHANNEL_START * NUM_TILE_TYPES;
    assert!(result[search_start..search_end].iter().all(|&v| v == 0.0));
}

#[test]
fn tile136_to_type_basics() {
    assert_eq!(tile136_to_type(0), 0); // 1m copy 0
    assert_eq!(tile136_to_type(3), 0); // 1m copy 3
    assert_eq!(tile136_to_type(4), 1); // 2m copy 0
    assert_eq!(tile136_to_type(135), 33); // chun copy 3
}

#[test]
fn aka_flags_and_dora_parts_detect_red_fives_and_cap_indicators() {
    assert_eq!(aka_flags_from_tiles([16, 52, 88]), [true, true, true]);
    assert_eq!(aka_flags_from_tiles([0, 4, 8]), [false, false, false]);

    let dora = dora_info_from_parts([1, 2, 3, 4, 5, 6], [16, 0, 52]);
    assert_eq!(dora.indicators, [1, 2, 3, 4, 5]);
    assert_eq!(dora.indicator_count, 5);
    assert_eq!(dora.aka_flags, [true, true, false]);
}

#[test]
fn metadata_parts_rotate_relative_state_and_compute_shanten() {
    let mut hand_counts = [0u8; NUM_TILE_TYPES];
    hand_counts[0] = 3;
    hand_counts[1] = 3;
    hand_counts[2] = 3;
    hand_counts[27] = 2;
    hand_counts[28] = 2;

    let meta = metadata_from_parts(
        2,
        &[true, false, true, false],
        &[25000, 26000, 27000, 28000],
        3,
        1,
        2,
        &hand_counts,
    );

    assert_eq!(meta.riichi, [true, false, true, false]);
    assert_eq!(meta.scores, [27000, 28000, 25000, 26000]);
    assert_eq!(meta.kyoku_index, 3);
    assert_eq!(meta.honba, 1);
    assert_eq!(meta.kyotaku, 2);
    assert!((-1..=8).contains(&meta.shanten));
}

#[test]
fn ct_smc_empty_and_context_fallbacks_use_safe_defaults() {
    let hand = [0u8; NUM_TILE_TYPES];
    let discards = std::array::from_fn(|_| PlayerDiscards::new());
    let melds = std::array::from_fn(|_| PlayerMelds::new());
    let dora = DoraInfo {
        indicators: [0; 5],
        indicator_count: 0,
        aka_flags: [false; 3],
    };
    let empty_smc = CtSmc::new(crate::ct_smc::CtSmcConfig::default().with_particles(1));

    assert_eq!(
        extract_ct_smc_remaining_counts(&empty_smc),
        [0.0; NUM_TILE_TYPES]
    );

    let from_empty_context =
        compute_hand_ev_from_context(&hand, &discards, &melds, &dora, &SearchContext::default());
    let from_empty_smc = compute_hand_ev_from_context(
        &hand,
        &discards,
        &melds,
        &dora,
        &SearchContext {
            ct_smc: Some(&empty_smc),
            ..SearchContext::default()
        },
    );

    assert_eq!(from_empty_context.tenpai_prob, from_empty_smc.tenpai_prob);
    assert_eq!(from_empty_context.ukeire, from_empty_smc.ukeire);
    assert_eq!(
        from_empty_context.expected_score,
        from_empty_smc.expected_score
    );
}
