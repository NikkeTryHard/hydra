use super::*;
use hydra_safety::bit_set;

#[test]
fn encoder_geometry_abi_constants_are_frozen() {
    assert_eq!(BASELINE_CHANNELS, 85);
    assert_eq!(NUM_CHANNELS, 192);
    assert_eq!(NUM_TILES, 34);
    assert_eq!(OBS_SIZE, 6528);
    assert_eq!(OBS_SIZE, NUM_CHANNELS * NUM_TILES);
}

/// Helper: read a single cell from the obs buffer.
fn get(enc: &ObservationEncoder, ch: usize, tile: usize) -> f32 {
    enc.as_slice()[ch * NUM_TILES + tile]
}

#[test]
fn new_encoder_is_zeroed() {
    let enc = ObservationEncoder::new();
    assert!(enc.as_slice().iter().all(|&v| v == 0.0));
}

#[test]
fn clear_resets_buffer() {
    let mut enc = ObservationEncoder::new();
    enc.buffer[0] = 42.0;
    enc.buffer[OBS_SIZE - 1] = 99.0;
    enc.clear();
    assert!(enc.as_slice().iter().all(|&v| v == 0.0));
}

// -- Hand tests (ch 0-3) --

#[test]
fn hand_single_tile() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; NUM_TILES];
    hand[0] = 1;
    enc.encode_hand(&hand);
    assert_eq!(get(&enc, 0, 0), 1.0);
    assert_eq!(get(&enc, 1, 0), 0.0);
}

#[test]
fn hand_four_copies() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; NUM_TILES];
    hand[5] = 4;
    enc.encode_hand(&hand);
    assert_eq!(get(&enc, 0, 5), 1.0);
    assert_eq!(get(&enc, 1, 5), 1.0);
    assert_eq!(get(&enc, 2, 5), 1.0);
    assert_eq!(get(&enc, 3, 5), 1.0);
}

#[test]
fn hand_three_copies_not_four() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; NUM_TILES];
    hand[10] = 3;
    enc.encode_hand(&hand);
    assert_eq!(get(&enc, 2, 10), 1.0);
    assert_eq!(get(&enc, 3, 10), 0.0);
}

// -- Open meld hand tests (ch 4-7) --

#[test]
fn open_meld_hand_thermometer() {
    let mut enc = ObservationEncoder::new();
    let mut counts = [0u8; NUM_TILES];
    counts[0] = 3; // 3 tiles of type 0 from melds
    counts[9] = 1; // 1 tile of type 9
    enc.encode_open_meld_hand(&counts);
    // tile 0: ch4=1, ch5=1, ch6=1, ch7=0
    assert_eq!(get(&enc, 4, 0), 1.0);
    assert_eq!(get(&enc, 5, 0), 1.0);
    assert_eq!(get(&enc, 6, 0), 1.0);
    assert_eq!(get(&enc, 7, 0), 0.0);
    // tile 9: ch4=1, ch5=0
    assert_eq!(get(&enc, 4, 9), 1.0);
    assert_eq!(get(&enc, 5, 9), 0.0);
}

#[test]
fn open_meld_hand_four() {
    let mut enc = ObservationEncoder::new();
    let mut counts = [0u8; NUM_TILES];
    counts[27] = 4; // kan of East
    enc.encode_open_meld_hand(&counts);
    assert_eq!(get(&enc, 7, 27), 1.0);
}

// -- Drawn tile tests (ch 8) --

#[test]
fn drawn_tile_encoded() {
    let mut enc = ObservationEncoder::new();
    enc.encode_drawn_tile(Some(5));
    assert_eq!(get(&enc, 8, 5), 1.0);
    assert_eq!(get(&enc, 8, 0), 0.0);
}

#[test]
fn drawn_tile_none_leaves_channel_zero() {
    let mut enc = ObservationEncoder::new();
    enc.encode_drawn_tile(None);
    for t in 0..NUM_TILES {
        assert_eq!(get(&enc, 8, t), 0.0);
    }
}

// -- Shanten mask tests (ch 9-10) --

#[test]
fn shanten_masks_complete_hand() {
    // Complete 14-tile hand: 123m 456m 789m 123p 11s
    // 4 sequences + 1 pair = agari (shanten = -1)
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; NUM_TILES];
    hand[..9].fill(1); // 1-9m
    hand[9] = 1;
    hand[10] = 1;
    hand[11] = 1; // 1-3p
    hand[18] = 2; // 1s pair
    // 14 tiles, len_div3=4, shanten=-1
    enc.encode_shanten_masks(&hand);
    // After discarding any tile, shanten goes from -1 to 0 (worsens).
    // So next-shanten (ch10) should have NO tiles set.
    for t in 0..NUM_TILES {
        assert_eq!(get(&enc, 10, t), 0.0);
    }
}

#[test]
fn shanten_masks_one_away() {
    // Simple iishanten hand: 1m,2m,3m, 4m,5m,6m, 7m,8m,9m, 1p,1p,1p, 2p, drawn 5s
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; NUM_TILES];
    hand[0] = 1;
    hand[1] = 1;
    hand[2] = 1; // 123m
    hand[3] = 1;
    hand[4] = 1;
    hand[5] = 1; // 456m
    hand[6] = 1;
    hand[7] = 1;
    hand[8] = 1; // 789m
    hand[9] = 3; // 1p x3
    hand[10] = 1; // 2p
    hand[22] = 1; // 5s (drawn tile)
    // 14 tiles. This is tenpai (waiting on 2p or 5s-related).
    // Actually 123m 456m 789m 111p + 2p 5s = tenpai waiting on 3p
    // shanten = 0 (tenpai)
    enc.encode_shanten_masks(&hand);
    // Discarding 2p or 5s keeps tenpai (shanten stays 0), so ch9 should be set
    // The exact tiles depend on shanten calc, but at minimum some tiles on ch9
    let ch9_sum: f32 = (0..NUM_TILES).map(|t| get(&enc, 9, t)).sum();
    assert!(
        ch9_sum > 0.0,
        "keep-shanten mask should have some tiles set"
    );
}

// -- Discard tests (ch 11-22) --

fn empty_discards() -> [PlayerDiscards; NUM_PLAYERS] {
    [
        PlayerDiscards::new(),
        PlayerDiscards::new(),
        PlayerDiscards::new(),
        PlayerDiscards::new(),
    ]
}

#[test]
fn discard_presence_and_tedashi() {
    let mut enc = ObservationEncoder::new();
    let mut discards = empty_discards();
    discards[0].push(DiscardEntry {
        tile: 5,
        is_tedashi: true,
        turn: 0,
    });
    discards[1].push(DiscardEntry {
        tile: 10,
        is_tedashi: false,
        turn: 0,
    });
    enc.encode_discards(&discards);
    // Player 0: ch_base=11, presence=ch11, tedashi=ch12
    assert_eq!(get(&enc, 11, 5), 1.0);
    assert_eq!(get(&enc, 12, 5), 1.0);
    // Player 1: ch_base=14, presence=ch14, tedashi=ch15
    assert_eq!(get(&enc, 14, 10), 1.0);
    assert_eq!(get(&enc, 15, 10), 0.0); // tsumogiri
}

#[test]
fn discard_temporal_decay() {
    let mut enc = ObservationEncoder::new();
    let mut discards = empty_discards();
    discards[0].push(DiscardEntry {
        tile: 0,
        is_tedashi: false,
        turn: 0,
    });
    discards[0].push(DiscardEntry {
        tile: 1,
        is_tedashi: false,
        turn: 5,
    });
    enc.encode_discards(&discards);
    // temporal ch = 11 + 2 = 13
    assert!((get(&enc, 13, 1) - 1.0).abs() < 1e-6);
    let expected = (-1.0f32).exp();
    assert!((get(&enc, 13, 0) - expected).abs() < 1e-6);
}

// -- Meld tests (ch 23-34) --

fn empty_melds() -> [PlayerMelds; NUM_PLAYERS] {
    [
        PlayerMelds::new(),
        PlayerMelds::new(),
        PlayerMelds::new(),
        PlayerMelds::new(),
    ]
}

#[test]
fn meld_chi() {
    let mut enc = ObservationEncoder::new();
    let mut melds = empty_melds();
    melds[0].push(MeldInfo {
        tiles: [0, 1, 2, 0],
        tile_count: 3,
        meld_type: MeldType::Chi,
    });
    enc.encode_melds(&melds);
    // Player 0 chi = ch 23
    assert_eq!(get(&enc, 23, 0), 1.0);
    assert_eq!(get(&enc, 23, 1), 1.0);
    assert_eq!(get(&enc, 23, 2), 1.0);
    assert_eq!(get(&enc, 24, 0), 0.0); // pon channel empty
}

#[test]
fn meld_pon() {
    let mut enc = ObservationEncoder::new();
    let mut melds = empty_melds();
    melds[2].push(MeldInfo {
        tiles: [27, 27, 27, 0],
        tile_count: 3,
        meld_type: MeldType::Pon,
    });
    enc.encode_melds(&melds);
    // Player 2: ch_base = 23 + 3*2 = 29, pon = ch 30
    assert_eq!(get(&enc, 30, 27), 1.0);
}

#[test]
fn meld_kan() {
    let mut enc = ObservationEncoder::new();
    let mut melds = empty_melds();
    melds[1].push(MeldInfo {
        tiles: [31, 31, 31, 31],
        tile_count: 4,
        meld_type: MeldType::Kan,
    });
    enc.encode_melds(&melds);
    // Player 1: ch_base = 23 + 3*1 = 26, kan = ch 28
    assert_eq!(get(&enc, 28, 31), 1.0);
}

// -- Dora tests (ch 35-39) --

#[test]
fn dora_indicator_thermometer_single() {
    let mut enc = ObservationEncoder::new();
    let dora = DoraInfo {
        indicators: [0, 0, 0, 0, 0],
        indicator_count: 1,
        aka_flags: [false; 3],
    };
    enc.encode_dora(&dora);
    assert_eq!(get(&enc, 35, 0), 1.0); // >= 1
    assert_eq!(get(&enc, 36, 0), 0.0); // >= 2 not set
}

#[test]
fn dora_indicator_thermometer_multiple_same() {
    let mut enc = ObservationEncoder::new();
    let dora = DoraInfo {
        indicators: [5, 5, 5, 0, 0],
        indicator_count: 3,
        aka_flags: [false; 3],
    };
    enc.encode_dora(&dora);
    assert_eq!(get(&enc, 35, 5), 1.0); // >= 1
    assert_eq!(get(&enc, 36, 5), 1.0); // >= 2
    assert_eq!(get(&enc, 37, 5), 1.0); // >= 3
    assert_eq!(get(&enc, 38, 5), 0.0); // >= 4 not set
}

#[test]
fn dora_indicator_thermometer_different() {
    let mut enc = ObservationEncoder::new();
    let dora = DoraInfo {
        indicators: [0, 10, 0, 0, 0],
        indicator_count: 2,
        aka_flags: [false; 3],
    };
    enc.encode_dora(&dora);
    assert_eq!(get(&enc, 35, 0), 1.0);
    assert_eq!(get(&enc, 35, 10), 1.0);
    assert_eq!(get(&enc, 36, 0), 0.0);
    assert_eq!(get(&enc, 36, 10), 0.0);
}

// -- Aka tests (ch 40-42) --

#[test]
fn aka_dora_plane_fill() {
    let mut enc = ObservationEncoder::new();
    let dora = DoraInfo {
        indicators: [0, 0, 0, 0, 0],
        indicator_count: 0,
        aka_flags: [true, false, true],
    };
    enc.encode_aka(&dora);
    // Ch 40 (5m): entire channel filled
    assert_eq!(get(&enc, 40, 0), 1.0);
    assert_eq!(get(&enc, 40, 33), 1.0);
    // Ch 41 (5p): not set
    assert_eq!(get(&enc, 41, 0), 0.0);
    // Ch 42 (5s): entire channel filled
    assert_eq!(get(&enc, 42, 0), 1.0);
}

// -- Metadata tests (ch 43-61) --

fn test_metadata() -> GameMetadata {
    GameMetadata {
        riichi: [true, false, false, false],
        scores: [25000, 25000, 25000, 25000],
        shanten: 1,
        kyoku_index: 0,
        honba: 2,
        kyotaku: 1,
    }
}

#[test]
fn metadata_riichi_and_scores() {
    let mut enc = ObservationEncoder::new();
    enc.encode_metadata(&test_metadata());
    // Self riichi = ch 43 filled
    assert_eq!(get(&enc, 43, 0), 1.0);
    // Opponent 1 riichi = ch 44 -- NOT set
    assert_eq!(get(&enc, 44, 0), 0.0);
    // Score ch 47 = 25000/100000 = 0.25
    assert!((get(&enc, 47, 0) - 0.25).abs() < 1e-6);
}

#[test]
fn metadata_score_gaps() {
    let mut enc = ObservationEncoder::new();
    let mut meta = test_metadata();
    meta.scores = [30000, 25000, 20000, 25000];
    enc.encode_metadata(&meta);
    // gap[0] = (30000-30000)/30000 = 0.0
    assert!((get(&enc, 51, 0) - 0.0).abs() < 1e-6);
    // gap[1] = (30000-25000)/30000 = 0.1667
    assert!((get(&enc, 52, 0) - 5000.0 / 30000.0).abs() < 1e-4);
    // gap[2] = (30000-20000)/30000 = 0.3333
    assert!((get(&enc, 53, 0) - 10000.0 / 30000.0).abs() < 1e-4);
}

#[test]
fn metadata_shanten_one_hot() {
    // shanten = 0 (tenpai) -> ch 55
    let mut enc = ObservationEncoder::new();
    let mut meta = test_metadata();
    meta.shanten = 0;
    enc.encode_metadata(&meta);
    assert_eq!(get(&enc, 55, 0), 1.0);
    assert_eq!(get(&enc, 56, 0), 0.0);

    // shanten = 2 -> ch 57
    enc.clear();
    meta.shanten = 2;
    enc.encode_metadata(&meta);
    assert_eq!(get(&enc, 57, 0), 1.0);
    assert_eq!(get(&enc, 55, 0), 0.0);

    // shanten = 5 (clamped to 3+) -> ch 58
    enc.clear();
    meta.shanten = 5;
    enc.encode_metadata(&meta);
    assert_eq!(get(&enc, 58, 0), 1.0);
}

#[test]
fn metadata_round_honba_kyotaku() {
    let mut enc = ObservationEncoder::new();
    let mut meta = test_metadata();
    meta.kyoku_index = 4; // South 1
    meta.honba = 3;
    meta.kyotaku = 2;
    enc.encode_metadata(&meta);
    // Ch 59: 4/8 = 0.5
    assert!((get(&enc, 59, 0) - 0.5).abs() < 1e-6);
    // Ch 60: 3/10 = 0.3
    assert!((get(&enc, 60, 0) - 0.3).abs() < 1e-6);
    // Ch 61: 2/10 = 0.2
    assert!((get(&enc, 61, 0) - 0.2).abs() < 1e-6);
}

// -- Safety tests (ch 62-84, unchanged) --

#[test]
fn safety_genbutsu_channels() {
    let mut enc = ObservationEncoder::new();
    let mut si = SafetyInfo::new();
    bit_set(&mut si.genbutsu_all[0], 5);
    bit_set(&mut si.genbutsu_tedashi[1], 10);
    bit_set(&mut si.genbutsu_riichi_era[2], 20);
    enc.encode_safety(&si);
    assert_eq!(get(&enc, 62, 5), 1.0);
    assert_eq!(get(&enc, 66, 10), 1.0);
    assert_eq!(get(&enc, 70, 20), 1.0);
}

#[test]
fn safety_suji_channel() {
    let mut enc = ObservationEncoder::new();
    let mut si = SafetyInfo::new();
    si.suji[0][0] = 1.0;
    enc.encode_safety(&si);
    assert_eq!(get(&enc, 71, 0), 1.0);
}

#[test]
fn safety_kabe_and_one_chance() {
    let mut enc = ObservationEncoder::new();
    let mut si = SafetyInfo::new();
    bit_set(&mut si.kabe, 15);
    bit_set(&mut si.one_chance, 20);
    enc.encode_safety(&si);
    assert_eq!(get(&enc, 80, 15), 1.0);
    assert_eq!(get(&enc, 81, 20), 1.0);
}

#[test]
fn safety_tenpai_hint_uses_cached_prediction_threshold() {
    let mut enc = ObservationEncoder::new();
    let mut si = SafetyInfo::new();
    si.set_tenpai_prediction(1, 0.75);
    si.set_tenpai_prediction(2, 0.5);
    enc.encode_safety(&si);
    assert_eq!(get(&enc, 83, 0), 1.0);
    assert_eq!(get(&enc, 84, 0), 0.0);
}

#[test]
fn safety_tenpai_hint_always_respects_riichi_status() {
    let mut enc = ObservationEncoder::new();
    let mut si = SafetyInfo::new();
    si.on_riichi(0);
    si.set_tenpai_prediction(0, 0.1);
    enc.encode_safety(&si);
    assert_eq!(get(&enc, 82, 0), 1.0);
}

#[test]
fn baseline_encode_zero_fills_dynamic_groups() {
    let mut enc = ObservationEncoder::new();
    let hand = [0u8; NUM_TILES];
    let open_meld = [0u8; NUM_TILES];
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo {
        indicators: [0, 0, 0, 0, 0],
        indicator_count: 0,
        aka_flags: [false; 3],
    };
    let meta = test_metadata();
    let safety = SafetyInfo::new();
    enc.encode(
        &hand, None, &open_meld, &discards, &melds, &dora, &meta, &safety,
    );
    assert!(
        enc.as_slice()[CH_SEARCH * NUM_TILES..]
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[test]
fn search_features_fill_presence_masks_and_planes() {
    let mut enc = ObservationEncoder::new();
    let mut search = SearchFeaturePlanes {
        belief_features_present: true,
        search_features_present: true,
        robust_features_present: true,
        context_features_present: true,
        ..SearchFeaturePlanes::default()
    };
    search.belief_fields[0][5] = 0.75;
    search.mixture_weights[1] = 0.4;
    search.mixture_entropy = 0.8;
    search.mixture_ess = 2.5;
    search.delta_q[7] = -0.2;
    search.opponent_risk[2][9] = 0.6;
    search.opponent_stress[1] = 0.3;
    enc.encode_search_features(&search);
    assert!((get(&enc, CH_SEARCH_BELIEF, 5) - 0.75).abs() < 1e-6);
    assert!((get(&enc, CH_SEARCH_MIXTURE_WEIGHT + 1, 0) - 0.4).abs() < 1e-6);
    assert!((get(&enc, CH_SEARCH_MIXTURE_ENTROPY, 0) - 0.8).abs() < 1e-6);
    assert!((get(&enc, CH_SEARCH_MIXTURE_ESS, 0) - 2.5).abs() < 1e-6);
    assert!((get(&enc, CH_SEARCH_DELTA_Q, 7) + 0.2).abs() < 1e-6);
    assert!((get(&enc, CH_SEARCH_RISK + 2, 9) - 0.6).abs() < 1e-6);
    assert!((get(&enc, CH_SEARCH_STRESS + 1, 0) - 0.3).abs() < 1e-6);
    assert_eq!(get(&enc, CH_SEARCH_MASKS, 0), 1.0);
    assert_eq!(get(&enc, CH_SEARCH_MASKS + 1, 0), 1.0);
    assert_eq!(get(&enc, CH_SEARCH_MASKS + 2, 0), 1.0);
    assert_eq!(get(&enc, CH_SEARCH_MASKS + 3, 0), 1.0);
}

#[test]
fn hand_ev_features_fill_expected_planes() {
    let mut enc = ObservationEncoder::new();
    let mut hand_ev = HandEvFeatures::default();
    hand_ev.tenpai_prob[3][0] = 0.2;
    hand_ev.win_prob[3][2] = 0.5;
    hand_ev.expected_score[3] = 6400.0;
    hand_ev.ukeire[3][6] = 2.0;
    enc.encode_hand_ev_features(&hand_ev);
    assert!((get(&enc, CH_HAND_EV_TENPAI, 3) - 0.2).abs() < 1e-6);
    assert!((get(&enc, CH_HAND_EV_WIN + 2, 3) - 0.5).abs() < 1e-6);
    assert!((get(&enc, CH_HAND_EV_SCORE, 3) - 6400.0).abs() < 1e-6);
    assert!((get(&enc, CH_HAND_EV_UKEIRE + 6, 3) - 2.0).abs() < 1e-6);
    assert_eq!(get(&enc, CH_HAND_EV_MASK, 0), 1.0);
}

// -- Full encode test --

#[test]
fn full_encode_returns_correct_size() {
    let mut enc = ObservationEncoder::new();
    let hand = [0u8; NUM_TILES];
    let open_meld = [0u8; NUM_TILES];
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo {
        indicators: [0, 0, 0, 0, 0],
        indicator_count: 0,
        aka_flags: [false; 3],
    };
    let meta = test_metadata();
    let safety = SafetyInfo::new();
    let obs = enc.encode(
        &hand, None, &open_meld, &discards, &melds, &dora, &meta, &safety,
    );
    assert_eq!(obs.len(), OBS_SIZE);
}

#[test]
fn full_encode_clears_between_calls() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; NUM_TILES];
    hand[0] = 3;
    let open_meld = [0u8; NUM_TILES];
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo {
        indicators: [0, 0, 0, 0, 0],
        indicator_count: 0,
        aka_flags: [false; 3],
    };
    let meta = test_metadata();
    let safety = SafetyInfo::new();
    enc.encode(
        &hand, None, &open_meld, &discards, &melds, &dora, &meta, &safety,
    );
    assert_eq!(get(&enc, 2, 0), 1.0);

    let empty_hand = [0u8; NUM_TILES];
    enc.encode(
        &empty_hand,
        None,
        &open_meld,
        &discards,
        &melds,
        &dora,
        &meta,
        &safety,
    );
    assert_eq!(get(&enc, 2, 0), 0.0);
}

#[test]
fn incremental_matches_full_encode() {
    let mut hand = [0u8; NUM_TILES];
    hand[0] = 3;
    hand[9] = 2;
    hand[18] = 1;
    let open_meld = [0u8; NUM_TILES];
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo {
        indicators: [4, 0, 0, 0, 0],
        indicator_count: 1,
        aka_flags: [true, false, false],
    };
    let meta = test_metadata();
    let safety = SafetyInfo::new();

    // Full encode
    let mut full = ObservationEncoder::new();
    full.encode(
        &hand,
        Some(18),
        &open_meld,
        &discards,
        &melds,
        &dora,
        &meta,
        &safety,
    );

    // Incremental with ALL flags = same as full encode
    let mut inc = ObservationEncoder::new();
    inc.encode_incremental(
        DirtyFlags::ALL,
        &hand,
        Some(18),
        &open_meld,
        &discards,
        &melds,
        &dora,
        &meta,
        &safety,
    );

    for i in 0..OBS_SIZE {
        assert!(
            (full.as_slice()[i] - inc.as_slice()[i]).abs() < 1e-6,
            "mismatch at index {}: full={}, inc={}",
            i,
            full.as_slice()[i],
            inc.as_slice()[i],
        );
    }
}

#[test]
fn incremental_partial_updates_only_dirty() {
    let hand = [0u8; NUM_TILES];
    let open_meld = [0u8; NUM_TILES];
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo {
        indicators: [0, 0, 0, 0, 0],
        indicator_count: 0,
        aka_flags: [false; 3],
    };
    let meta = test_metadata();
    let safety = SafetyInfo::new();

    let mut enc = ObservationEncoder::new();
    // Full encode first to set baseline
    enc.encode(
        &hand, None, &open_meld, &discards, &melds, &dora, &meta, &safety,
    );
    let baseline = *enc.as_slice();

    // Incremental with only META dirty -- only ch 43-61 should change
    let mut meta2 = test_metadata();
    meta2.scores = [50000, 10000, 10000, 30000];
    enc.encode_incremental(
        DirtyFlags::META,
        &hand,
        None,
        &open_meld,
        &discards,
        &melds,
        &dora,
        &meta2,
        &safety,
    );

    // Ch 0-42 should be unchanged
    for (i, &base_val) in baseline[..(CH_META * NUM_TILES)].iter().enumerate() {
        assert_eq!(enc.as_slice()[i], base_val, "non-meta ch changed at {i}");
    }
    // Ch 43-61 should differ (scores changed)
    let meta_start = CH_META * NUM_TILES;
    let scores_start = meta_start + 4 * NUM_TILES; // ch 47
    assert!((enc.as_slice()[scores_start] - 0.5).abs() < 1e-6); // 50000/100000
}

#[test]
fn profile_encode_clears_stale_hand_ev_when_reused_without_hand_ev() {
    let mut enc = ObservationEncoder::new();
    let hand = [0u8; NUM_TILES];
    let open_meld = [0u8; NUM_TILES];
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo {
        indicators: [0, 0, 0, 0, 0],
        indicator_count: 0,
        aka_flags: [false; 3],
    };
    let meta = test_metadata();
    let safety = SafetyInfo::new();
    let shanten_batch = shanten_batch::batch_discard_shanten(&hand, 0);

    let mut hand_ev = HandEvFeatures::default();
    hand_ev.tenpai_prob[3][0] = 0.2;
    hand_ev.win_prob[3][2] = 0.5;
    hand_ev.expected_score[3] = 6400.0;
    hand_ev.ukeire[3][6] = 2.0;

    enc.encode_with_context_and_shanten_batch(
        &hand,
        None,
        &open_meld,
        &discards,
        &melds,
        &dora,
        &meta,
        &safety,
        &shanten_batch,
        None,
        Some(&hand_ev),
    );
    assert_eq!(get(&enc, CH_HAND_EV_MASK, 0), 1.0);

    enc.encode_with_context_and_shanten_batch(
        &hand,
        None,
        &open_meld,
        &discards,
        &melds,
        &dora,
        &meta,
        &safety,
        &shanten_batch,
        None,
        None,
    );

    let mask_offset = CH_HAND_EV_MASK * NUM_TILES;
    assert_eq!(enc.as_slice()[mask_offset], 0.0);
    assert!(
        enc.as_slice()[CH_HAND_EV * NUM_TILES..mask_offset]
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[test]
fn profile_encode_clears_stale_search_when_reused_without_search_features() {
    let mut enc = ObservationEncoder::new();
    let hand = [0u8; NUM_TILES];
    let open_meld = [0u8; NUM_TILES];
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo {
        indicators: [0, 0, 0, 0, 0],
        indicator_count: 0,
        aka_flags: [false; 3],
    };
    let meta = test_metadata();
    let safety = SafetyInfo::new();
    let shanten_batch = shanten_batch::batch_discard_shanten(&hand, 0);

    let mut search = SearchFeaturePlanes {
        belief_features_present: true,
        search_features_present: true,
        robust_features_present: true,
        context_features_present: true,
        ..SearchFeaturePlanes::default()
    };
    search.belief_fields[0][5] = 0.75;
    search.mixture_weights[1] = 0.4;
    search.mixture_entropy = 0.8;
    search.mixture_ess = 2.5;
    search.delta_q[7] = -0.2;
    search.opponent_risk[2][9] = 0.6;
    search.opponent_stress[1] = 0.3;

    enc.encode_with_context_and_shanten_batch(
        &hand,
        None,
        &open_meld,
        &discards,
        &melds,
        &dora,
        &meta,
        &safety,
        &shanten_batch,
        Some(&search),
        None,
    );
    assert_eq!(get(&enc, CH_SEARCH_MASKS, 0), 1.0);

    enc.encode_with_context_and_shanten_batch(
        &hand,
        None,
        &open_meld,
        &discards,
        &melds,
        &dora,
        &meta,
        &safety,
        &shanten_batch,
        None,
        None,
    );

    let mask_offset = CH_HAND_EV * NUM_TILES;
    assert!(
        enc.as_slice()[CH_SEARCH * NUM_TILES..mask_offset]
            .iter()
            .all(|&v| v == 0.0)
    );
}
