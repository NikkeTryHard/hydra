//! Golden encoder regression tests.
//!
//! Each test constructs a specific game state, runs the encoder, and
//! verifies specific channel/tile values against hardcoded expectations.

use hydra_core::encoder::*;
use hydra_core::safety::SafetyInfo;

/// Read a single cell: buffer[ch * 34 + tile].
fn get(enc: &ObservationEncoder, ch: usize, tile: usize) -> f32 {
    enc.as_slice()[ch * NUM_TILES + tile]
}

/// Assert float equality with epsilon.
fn assert_close(actual: f32, expected: f32, msg: &str) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "{msg}: expected {expected}, got {actual}"
    );
}

fn empty_discards() -> [PlayerDiscards; 4] {
    [
        PlayerDiscards { discards: vec![] },
        PlayerDiscards { discards: vec![] },
        PlayerDiscards { discards: vec![] },
        PlayerDiscards { discards: vec![] },
    ]
}

fn empty_melds() -> [Vec<MeldInfo>; 4] {
    [vec![], vec![], vec![], vec![]]
}

fn default_meta() -> GameMetadata {
    GameMetadata {
        round_wind: 0,
        seat_wind: 0,
        is_dealer: true,
        riichi: [false; 4],
        honba: 0,
        riichi_sticks: 0,
        tiles_remaining: 70,
        scores: [25000; 4],
    }
}

// =========================================================================
// States 1-4: Hand encoding (channels 0-3)
// =========================================================================

#[test]
fn golden_hand_single_tile() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; 34];
    hand[0] = 1; // one 1m
    enc.encode_hand(&hand, None);
    assert_eq!(get(&enc, 0, 0), 1.0, "ch0 tile0: count>=1");
    assert_eq!(get(&enc, 1, 0), 0.0, "ch1 tile0: count>=2 off");
    assert_eq!(get(&enc, 2, 0), 0.0, "ch2 tile0: count>=3 off");
    assert_eq!(get(&enc, 3, 0), 0.0, "ch3 tile0: count==4 off");
    // Other tiles untouched
    assert_eq!(get(&enc, 0, 1), 0.0, "ch0 tile1: not in hand");
}

#[test]
fn golden_hand_two_copies() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; 34];
    hand[9] = 2; // two 1p
    enc.encode_hand(&hand, None);
    assert_eq!(get(&enc, 0, 9), 1.0, "ch0: count>=1");
    assert_eq!(get(&enc, 1, 9), 1.0, "ch1: count>=2");
    assert_eq!(get(&enc, 2, 9), 0.0, "ch2: count>=3 off");
    assert_eq!(get(&enc, 3, 9), 0.0, "ch3: count==4 off");
}

#[test]
fn golden_hand_three_copies() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; 34];
    hand[18] = 3; // three 1s
    enc.encode_hand(&hand, None);
    assert_eq!(get(&enc, 0, 18), 1.0, "ch0: count>=1");
    assert_eq!(get(&enc, 1, 18), 1.0, "ch1: count>=2");
    assert_eq!(get(&enc, 2, 18), 1.0, "ch2: count>=3");
    assert_eq!(get(&enc, 3, 18), 0.0, "ch3: count==4 off");
}

#[test]
fn golden_hand_four_copies() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; 34];
    hand[27] = 4; // four East
    enc.encode_hand(&hand, None);
    assert_eq!(get(&enc, 0, 27), 1.0, "ch0: count>=1");
    assert_eq!(get(&enc, 1, 27), 1.0, "ch1: count>=2");
    assert_eq!(get(&enc, 2, 27), 1.0, "ch2: count>=3");
    assert_eq!(get(&enc, 3, 27), 1.0, "ch3: count==4");
}

// =========================================================================
// State 5: Drawn tile (channel 56)
// =========================================================================

#[test]
fn golden_drawn_tile_7p() {
    let mut enc = ObservationEncoder::new();
    enc.encode_hand(&[0u8; 34], Some(15)); // drew 7p (idx 15)
    assert_eq!(get(&enc, 56, 15), 1.0, "ch56 tile15: drawn tile set");
    // All other tiles on ch56 are 0
    for t in 0..34 {
        if t != 15 {
            assert_eq!(get(&enc, 56, t), 0.0, "ch56 tile{t}: not drawn");
        }
    }
}

#[test]
fn golden_drawn_tile_none() {
    let mut enc = ObservationEncoder::new();
    enc.encode_hand(&[0u8; 34], None);
    for t in 0..34 {
        assert_eq!(get(&enc, 56, t), 0.0, "ch56 tile{t}: no draw");
    }
}

// =========================================================================
// States 6-8: Discards (channels 4-15)
// =========================================================================

#[test]
fn golden_discard_presence_player0() {
    let mut enc = ObservationEncoder::new();
    let mut discards = empty_discards();
    discards[0].discards.push(DiscardEntry {
        tile: 5, is_tedashi: true, turn: 0,
    });
    enc.encode_discards(&discards);
    // Player 0: ch_base=4, presence=ch4, tedashi=ch5, temporal=ch6
    assert_eq!(get(&enc, 4, 5), 1.0, "p0 presence tile5");
    assert_eq!(get(&enc, 5, 5), 1.0, "p0 tedashi tile5");
    assert_close(get(&enc, 6, 5), 1.0, "p0 temporal tile5 (only discard)");
    // Tile not discarded stays zero
    assert_eq!(get(&enc, 4, 0), 0.0, "p0 presence tile0 not discarded");
}

#[test]
fn golden_discard_tsumogiri_no_tedashi() {
    let mut enc = ObservationEncoder::new();
    let mut discards = empty_discards();
    discards[1].discards.push(DiscardEntry {
        tile: 10, is_tedashi: false, turn: 0,
    });
    enc.encode_discards(&discards);
    // Player 1: ch_base=7
    assert_eq!(get(&enc, 7, 10), 1.0, "p1 presence tile10");
    assert_eq!(get(&enc, 8, 10), 0.0, "p1 tedashi off (tsumogiri)");
}

#[test]
fn golden_discard_temporal_decay() {
    let mut enc = ObservationEncoder::new();
    let mut discards = empty_discards();
    // Player 2 discards at turn 0 and turn 5
    discards[2].discards.push(DiscardEntry {
        tile: 0, is_tedashi: false, turn: 0,
    });
    discards[2].discards.push(DiscardEntry {
        tile: 1, is_tedashi: false, turn: 5,
    });
    enc.encode_discards(&discards);
    // Player 2: ch_base=10, temporal=ch12
    // t_max=5, tile1 at turn5: exp(-0.2*(5-5))=1.0
    assert_close(get(&enc, 12, 1), 1.0, "p2 temporal tile1 most recent");
    // tile0 at turn0: exp(-0.2*(5-0))=exp(-1.0)
    let expected = (-1.0f32).exp();
    assert_close(get(&enc, 12, 0), expected, "p2 temporal tile0 decayed");
}

// =========================================================================
// States 9-10: Melds (channels 16-27)
// =========================================================================

#[test]
fn golden_meld_open_chi() {
    let mut enc = ObservationEncoder::new();
    let mut melds = empty_melds();
    melds[0].push(MeldInfo {
        tiles: vec![0, 1, 2], // 1m-2m-3m chi
        meld_type: MeldType::Open,
    });
    enc.encode_melds(&melds);
    // Player 0 open = ch16
    assert_eq!(get(&enc, 16, 0), 1.0, "p0 open 1m");
    assert_eq!(get(&enc, 16, 1), 1.0, "p0 open 2m");
    assert_eq!(get(&enc, 16, 2), 1.0, "p0 open 3m");
    assert_eq!(get(&enc, 17, 0), 0.0, "p0 closed kan empty");
    assert_eq!(get(&enc, 18, 0), 0.0, "p0 kakan empty");
}

#[test]
fn golden_meld_closed_kan_player2() {
    let mut enc = ObservationEncoder::new();
    let mut melds = empty_melds();
    melds[2].push(MeldInfo {
        tiles: vec![27, 27, 27, 27], // East ankan
        meld_type: MeldType::ClosedKan,
    });
    enc.encode_melds(&melds);
    // Player 2: ch_base = 16 + 3*2 = 22, closed_kan = ch23
    assert_eq!(get(&enc, 23, 27), 1.0, "p2 closed kan East");
    assert_eq!(get(&enc, 22, 27), 0.0, "p2 open empty");
    assert_eq!(get(&enc, 24, 27), 0.0, "p2 kakan empty");
}

// =========================================================================
// States 11-13: Dora (channels 28-31)
// =========================================================================

#[test]
fn golden_dora_standard_mapping() {
    let mut enc = ObservationEncoder::new();
    let dora = DoraInfo {
        indicators: vec![0], // 1m indicator -> 2m is dora
        aka_flags: [false; 3],
    };
    enc.encode_dora(&dora);
    assert_eq!(get(&enc, 28, 0), 1.0, "ch28: indicator 1m");
    assert_eq!(get(&enc, 29, 1), 1.0, "ch29: actual dora 2m");
    assert_eq!(get(&enc, 29, 0), 0.0, "ch29: 1m is not dora");
}

#[test]
fn golden_dora_wind_wrap_n_to_e() {
    // North indicator (30) -> East dora (27)
    let mut enc = ObservationEncoder::new();
    let dora = DoraInfo {
        indicators: vec![30],
        aka_flags: [false; 3],
    };
    enc.encode_dora(&dora);
    assert_eq!(get(&enc, 28, 30), 1.0, "ch28: indicator North");
    assert_eq!(get(&enc, 29, 27), 1.0, "ch29: dora East (wrap)");
}

#[test]
fn golden_dora_dragon_wrap_chun_to_haku() {
    // Chun indicator (33) -> Haku dora (31)
    let mut enc = ObservationEncoder::new();
    let dora = DoraInfo {
        indicators: vec![33],
        aka_flags: [false; 3],
    };
    enc.encode_dora(&dora);
    assert_eq!(get(&enc, 28, 33), 1.0, "ch28: indicator Chun");
    assert_eq!(get(&enc, 29, 31), 1.0, "ch29: dora Haku (wrap)");
}

#[test]
fn golden_dora_aka_flags() {
    let mut enc = ObservationEncoder::new();
    let dora = DoraInfo {
        indicators: vec![],
        aka_flags: [true, false, true], // red 5m and red 5s
    };
    enc.encode_dora(&dora);
    assert_eq!(get(&enc, 30, 4), 1.0, "ch30: aka 5m");
    assert_eq!(get(&enc, 30, 13), 0.0, "ch30: no aka 5p");
    assert_eq!(get(&enc, 30, 22), 1.0, "ch30: aka 5s");
    // Ch31 (ura) stays zero during play
    for t in 0..34 {
        assert_eq!(get(&enc, 31, t), 0.0, "ch31 ura zero");
    }
}

// =========================================================================
// States 14-17: Metadata (channels 32-61)
// =========================================================================

#[test]
fn golden_meta_round_wind_south() {
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.round_wind = 1; // South
    enc.encode_metadata(&meta);
    // South -> ch33 filled, ch32 empty
    assert_eq!(get(&enc, 32, 0), 0.0, "ch32 East off");
    assert_eq!(get(&enc, 33, 0), 1.0, "ch33 South on");
    assert_eq!(get(&enc, 33, 17), 1.0, "ch33 filled uniformly");
    assert_eq!(get(&enc, 34, 0), 0.0, "ch34 West off");
}

#[test]
fn golden_meta_seat_wind_west() {
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.seat_wind = 2; // West
    enc.encode_metadata(&meta);
    // West -> ch38 (36+2)
    assert_eq!(get(&enc, 36, 0), 0.0, "ch36 East seat off");
    assert_eq!(get(&enc, 38, 0), 1.0, "ch38 West seat on");
    assert_eq!(get(&enc, 38, 33), 1.0, "ch38 filled uniformly");
}

#[test]
fn golden_meta_dealer_flag() {
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.is_dealer = true;
    enc.encode_metadata(&meta);
    // Dealer flag = ch40 (32+8)
    assert_eq!(get(&enc, 40, 0), 1.0, "ch40 dealer on");
    assert_eq!(get(&enc, 40, 33), 1.0, "ch40 filled uniformly");
}

#[test]
fn golden_meta_riichi_flags() {
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.riichi = [true, false, true, false]; // self + opp2 riichi
    enc.encode_metadata(&meta);
    // Self riichi = ch41 (32+9)
    assert_eq!(get(&enc, 41, 0), 1.0, "ch41 self riichi on");
    // Opp1 riichi = ch42 -- off
    assert_eq!(get(&enc, 42, 0), 0.0, "ch42 opp1 riichi off");
    // Opp2 riichi = ch43
    assert_eq!(get(&enc, 43, 0), 1.0, "ch43 opp2 riichi on");
    // Opp3 riichi = ch44 -- off
    assert_eq!(get(&enc, 44, 0), 0.0, "ch44 opp3 riichi off");
}

#[test]
fn golden_meta_honba_and_sticks() {
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.honba = 4;
    meta.riichi_sticks = 2;
    enc.encode_metadata(&meta);
    // Honba ch45 (32+13): 4/8 = 0.5
    assert_close(get(&enc, 45, 0), 0.5, "ch45 honba 4/8");
    assert_close(get(&enc, 45, 33), 0.5, "ch45 filled uniformly");
    // Riichi sticks ch46 (32+14): 2/4 = 0.5
    assert_close(get(&enc, 46, 0), 0.5, "ch46 riichi sticks 2/4");
}

#[test]
fn golden_meta_scores_normalized() {
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.scores = [50000, -10000, 30000, 30000];
    enc.encode_metadata(&meta);
    // Score ch48-51 (32+16..32+19): score/100000
    assert_close(get(&enc, 48, 0), 0.5, "ch48 p0 50k");
    assert_close(get(&enc, 49, 0), -0.1, "ch49 p1 -10k");
    assert_close(get(&enc, 50, 0), 0.3, "ch50 p2 30k");
}

// =========================================================================
// States 18-20: Safety (channels 62-84)
// =========================================================================

#[test]
fn golden_safety_genbutsu_all() {
    let mut enc = ObservationEncoder::new();
    let mut si = SafetyInfo::new();
    si.genbutsu_all[0][5] = true;  // 6m safe vs opp0
    si.genbutsu_all[2][33] = true; // chun safe vs opp2
    enc.encode_safety(&si);
    assert_eq!(get(&enc, 62, 5), 1.0, "ch62 opp0 genbutsu 6m");
    assert_eq!(get(&enc, 62, 0), 0.0, "ch62 opp0 1m not safe");
    assert_eq!(get(&enc, 64, 33), 1.0, "ch64 opp2 genbutsu chun");
    assert_eq!(get(&enc, 63, 33), 0.0, "ch63 opp1 chun not safe");
}

#[test]
fn golden_safety_tedashi_and_riichi_era() {
    let mut enc = ObservationEncoder::new();
    let mut si = SafetyInfo::new();
    si.genbutsu_tedashi[1][10] = true;     // opp1 tedashi 2p
    si.genbutsu_riichi_era[0][20] = true;  // opp0 riichi-era 3s
    enc.encode_safety(&si);
    // Tedashi: ch65 + opp -> ch66
    assert_eq!(get(&enc, 66, 10), 1.0, "ch66 opp1 tedashi 2p");
    assert_eq!(get(&enc, 65, 10), 0.0, "ch65 opp0 tedashi 2p off");
    // Riichi-era: ch68 + opp -> ch68
    assert_eq!(get(&enc, 68, 20), 1.0, "ch68 opp0 riichi-era 3s");
    assert_eq!(get(&enc, 69, 20), 0.0, "ch69 opp1 riichi-era 3s off");
}

#[test]
fn golden_safety_suji() {
    let mut enc = ObservationEncoder::new();
    let mut si = SafetyInfo::new();
    si.suji[0][0] = 1.0;   // 1m suji vs opp0
    si.suji[2][8] = 0.75;  // 9m partial suji vs opp2
    enc.encode_safety(&si);
    // Suji: ch71 + opp
    assert_close(get(&enc, 71, 0), 1.0, "ch71 opp0 suji 1m");
    assert_close(get(&enc, 73, 8), 0.75, "ch73 opp2 suji 9m");
    assert_eq!(get(&enc, 72, 0), 0.0, "ch72 opp1 suji 1m off");
}

#[test]
fn golden_safety_kabe_one_chance() {
    let mut enc = ObservationEncoder::new();
    let mut si = SafetyInfo::new();
    si.kabe[15] = true;       // 7p kabe
    si.one_chance[20] = true; // 3s one-chance
    enc.encode_safety(&si);
    // Kabe = ch80 (62+18), one-chance = ch81 (62+19)
    assert_eq!(get(&enc, 80, 15), 1.0, "ch80 kabe 7p");
    assert_eq!(get(&enc, 80, 0), 0.0, "ch80 kabe 1m off");
    assert_eq!(get(&enc, 81, 20), 1.0, "ch81 one-chance 3s");
    assert_eq!(get(&enc, 81, 0), 0.0, "ch81 one-chance 1m off");
    // Reserved channels 82-84 stay zero
    for ch in 82..=84 {
        for t in 0..34 {
            assert_eq!(get(&enc, ch, t), 0.0, "reserved ch{ch} zero");
        }
    }
}

// =========================================================================
// States 21-22: Full encode roundtrip
// =========================================================================

#[test]
fn golden_full_encode_roundtrip() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; 34];
    hand[0] = 2;
    hand[8] = 1;
    let mut discards = empty_discards();
    discards[0].discards.push(DiscardEntry {
        tile: 3, is_tedashi: true, turn: 0,
    });
    let mut melds = empty_melds();
    melds[1].push(MeldInfo {
        tiles: vec![9, 10, 11],
        meld_type: MeldType::Open,
    });
    let dora = DoraInfo {
        indicators: vec![4],
        aka_flags: [true, false, false],
    };
    let meta = default_meta();
    let si = SafetyInfo::new();
    enc.encode(&hand, Some(8), &discards, &melds, &dora, &meta, &si);
    assert_eq!(enc.as_slice().len(), OBS_SIZE, "buffer size");

    // Spot-check: hand ch0 tile0 (count>=1 for 2x 1m)
    assert_eq!(get(&enc, 0, 0), 1.0, "hand 1m present");
    assert_eq!(get(&enc, 1, 0), 1.0, "hand 1m count>=2");
    assert_eq!(get(&enc, 0, 8), 1.0, "hand 9m present");
    // Drawn tile
    assert_eq!(get(&enc, 56, 8), 1.0, "drawn 9m");
    // Discard presence
    assert_eq!(get(&enc, 4, 3), 1.0, "p0 discarded 4m");
    // Meld open p1
    assert_eq!(get(&enc, 19, 9), 1.0, "p1 open meld 1p");
    // Dora indicator
    assert_eq!(get(&enc, 28, 4), 1.0, "dora indicator 5m");
    assert_eq!(get(&enc, 29, 5), 1.0, "dora actual 6m");
    // Aka dora
    assert_eq!(get(&enc, 30, 4), 1.0, "aka 5m");
    // Nonzero count sanity
    let nz = enc.as_slice().iter().filter(|&&v| v != 0.0).count();
    assert!(nz > 50, "expected >50 nonzero cells, got {nz}");
}

#[test]
fn golden_full_encode_with_safety() {
    let mut enc = ObservationEncoder::new();
    let hand = [0u8; 34];
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo { indicators: vec![], aka_flags: [false; 3] };
    let mut meta = default_meta();
    meta.round_wind = 1; // South
    meta.seat_wind = 3; // North
    meta.is_dealer = false;
    let mut si = SafetyInfo::new();
    si.genbutsu_all[0][0] = true;
    si.kabe[27] = true;
    enc.encode(&hand, None, &discards, &melds, &dora, &meta, &si);
    // Verify across sections
    assert_eq!(get(&enc, 33, 0), 1.0, "round wind South");
    assert_eq!(get(&enc, 39, 0), 1.0, "seat wind North");
    assert_eq!(get(&enc, 40, 0), 0.0, "not dealer");
    assert_eq!(get(&enc, 62, 0), 1.0, "genbutsu opp0 1m");
    assert_eq!(get(&enc, 80, 27), 1.0, "kabe East");
}

// =========================================================================
// State 23: Encode clears between calls
// =========================================================================

#[test]
fn golden_encode_clears_stale_data() {
    let mut enc = ObservationEncoder::new();
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo { indicators: vec![], aka_flags: [false; 3] };
    let meta = default_meta();
    let si = SafetyInfo::new();

    // First encode: hand with 3x 1m
    let mut hand1 = [0u8; 34];
    hand1[0] = 3;
    enc.encode(&hand1, Some(5), &discards, &melds, &dora, &meta, &si);
    assert_eq!(get(&enc, 2, 0), 1.0, "first: ch2 tile0 set");
    assert_eq!(get(&enc, 56, 5), 1.0, "first: drawn 6m");

    // Second encode: empty hand, no draw
    let hand2 = [0u8; 34];
    enc.encode(&hand2, None, &discards, &melds, &dora, &meta, &si);
    assert_eq!(get(&enc, 2, 0), 0.0, "second: ch2 tile0 cleared");
    assert_eq!(get(&enc, 56, 5), 0.0, "second: drawn 6m cleared");
    assert_eq!(get(&enc, 0, 0), 0.0, "second: ch0 tile0 cleared");
}
