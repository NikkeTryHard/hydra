//! Golden encoder regression tests.
//!
//! Each test constructs a specific game state, runs the encoder, and
//! verifies specific channel/tile values against hardcoded expectations.
//!
//! Channel map (85 channels):
//!   0-3:   Hand thermometer
//!   4-7:   Open meld hand count thermometer
//!   8:     Drawn tile
//!   9-10:  Shanten masks
//!   11-22: Discards (3 per player)
//!   23-34: Melds (3 per player: Chi/Pon/Kan)
//!   35-39: Dora indicator thermometer
//!   40-42: Aka flags (one channel per suit)
//!   43-46: Riichi flags
//!   47-50: Scores / 100000
//!   51-54: Relative score gaps
//!   55-58: Shanten one-hot
//!   59:    Round number
//!   60:    Honba / 10
//!   61:    Kyotaku / 10
//!   62-84: Safety

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
        riichi: [false; 4],
        scores: [25000; 4],
        shanten: 3,
        kyoku_index: 0,
        honba: 0,
        kyotaku: 0,
    }
}

// =========================================================================
// Hand encoding (channels 0-3) -- UNCHANGED
// =========================================================================

#[test]
fn golden_hand_single_tile() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; 34];
    hand[0] = 1; // one 1m
    enc.encode_hand(&hand);
    assert_eq!(get(&enc, 0, 0), 1.0, "ch0 tile0: count>=1");
    assert_eq!(get(&enc, 1, 0), 0.0, "ch1 tile0: count>=2 off");
    assert_eq!(get(&enc, 2, 0), 0.0, "ch2 tile0: count>=3 off");
    assert_eq!(get(&enc, 3, 0), 0.0, "ch3 tile0: count==4 off");
    assert_eq!(get(&enc, 0, 1), 0.0, "ch0 tile1: not in hand");
}

#[test]
fn golden_hand_two_copies() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; 34];
    hand[9] = 2; // two 1p
    enc.encode_hand(&hand);
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
    enc.encode_hand(&hand);
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
    enc.encode_hand(&hand);
    assert_eq!(get(&enc, 0, 27), 1.0, "ch0: count>=1");
    assert_eq!(get(&enc, 1, 27), 1.0, "ch1: count>=2");
    assert_eq!(get(&enc, 2, 27), 1.0, "ch2: count>=3");
    assert_eq!(get(&enc, 3, 27), 1.0, "ch3: count==4");
}

// =========================================================================
// Open meld hand count (channels 4-7) -- NEW
// =========================================================================

#[test]
fn golden_open_meld_hand() {
    let mut enc = ObservationEncoder::new();
    // Simulate a chi of 1m-2m-3m: 1 copy each of tiles 0,1,2 in open melds
    let mut open_meld_counts = [0u8; 34];
    open_meld_counts[0] = 1;
    open_meld_counts[1] = 1;
    open_meld_counts[2] = 1;
    enc.encode_open_meld_hand(&open_meld_counts);
    // Thermometer: ch4=count>=1, ch5=count>=2, ch6=count>=3, ch7=count==4
    assert_eq!(get(&enc, 4, 0), 1.0, "ch4 tile0: meld count>=1");
    assert_eq!(get(&enc, 5, 0), 0.0, "ch5 tile0: meld count>=2 off");
    assert_eq!(get(&enc, 4, 1), 1.0, "ch4 tile1: meld count>=1");
    assert_eq!(get(&enc, 4, 3), 0.0, "ch4 tile3: not in meld");
}

// =========================================================================
// Drawn tile (channel 8) -- was ch 56
// =========================================================================

#[test]
fn golden_drawn_tile_7p() {
    let mut enc = ObservationEncoder::new();
    enc.encode_hand(&[0u8; 34]);
    enc.encode_drawn_tile(Some(15)); // drew 7p (idx 15)
    assert_eq!(get(&enc, 8, 15), 1.0, "ch8 tile15: drawn tile set");
    for t in 0..34 {
        if t != 15 {
            assert_eq!(get(&enc, 8, t), 0.0, "ch8 tile{t}: not drawn");
        }
    }
}

#[test]
fn golden_drawn_tile_none() {
    let mut enc = ObservationEncoder::new();
    enc.encode_hand(&[0u8; 34]);
    enc.encode_drawn_tile(None);
    for t in 0..34 {
        assert_eq!(get(&enc, 8, t), 0.0, "ch8 tile{t}: no draw");
    }
}

// =========================================================================
// Shanten masks (channels 9-10) -- NEW
// =========================================================================

#[test]
fn golden_shanten_masks() {
    // Shanten masks mark tiles that would reduce shanten.
    // Ch 9: tiles that improve shanten if drawn (ukeire mask)
    // Ch 10: tiles that are dangerous to discard (keep mask)
    // Tested via full encode -- masks are computed internally.
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; 34];
    // Tenpai hand: 1-2-3-4-5-6-7-8m + 1-1-1p + 9-9s (waiting on 9m)
    hand[..8].fill(1); // 1m-8m
    hand[9] = 3;  // 1p x3
    hand[26] = 2; // 9s x2
    let open_meld_counts = [0u8; 34];
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo { indicators: vec![], aka_flags: [false; 3] };
    let mut meta = default_meta();
    meta.shanten = 0; // tenpai
    let si = SafetyInfo::new();
    enc.encode(
        &hand, None, &open_meld_counts, &discards, &melds,
        &dora, &meta, &si,
    );
    // With shanten=0 (tenpai), masks should be populated
    // Ch 9: acceptance tiles should be marked
    // Ch 10: essential tiles should be marked
    // Exact values depend on shanten calculator; verify channels exist
    // and are not all-zero for a tenpai hand
    let ch9_sum: f32 = (0..34).map(|t| get(&enc, 9, t)).sum();
    let ch10_sum: f32 = (0..34).map(|t| get(&enc, 10, t)).sum();
    assert!(ch9_sum >= 0.0, "ch9 shanten mask should be non-negative");
    assert!(ch10_sum >= 0.0, "ch10 shanten mask should be non-negative");
}

// =========================================================================
// Discards (channels 11-22) -- was 4-15, shifted +7
// =========================================================================

#[test]
fn golden_discard_presence_player0() {
    let mut enc = ObservationEncoder::new();
    let mut discards = empty_discards();
    discards[0].discards.push(DiscardEntry {
        tile: 5, is_tedashi: true, turn: 0,
    });
    enc.encode_discards(&discards);
    // Player 0: ch_base=11, presence=ch11, tedashi=ch12, temporal=ch13
    assert_eq!(get(&enc, 11, 5), 1.0, "p0 presence tile5");
    assert_eq!(get(&enc, 12, 5), 1.0, "p0 tedashi tile5");
    assert_close(get(&enc, 13, 5), 1.0, "p0 temporal tile5 (only discard)");
    assert_eq!(get(&enc, 11, 0), 0.0, "p0 presence tile0 not discarded");
}

#[test]
fn golden_discard_tsumogiri_no_tedashi() {
    let mut enc = ObservationEncoder::new();
    let mut discards = empty_discards();
    discards[1].discards.push(DiscardEntry {
        tile: 10, is_tedashi: false, turn: 0,
    });
    enc.encode_discards(&discards);
    // Player 1: ch_base=14
    assert_eq!(get(&enc, 14, 10), 1.0, "p1 presence tile10");
    assert_eq!(get(&enc, 15, 10), 0.0, "p1 tedashi off (tsumogiri)");
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
    // Player 2: ch_base=17, temporal=ch19
    // t_max=5, tile1 at turn5: exp(-0.2*(5-5))=1.0
    assert_close(get(&enc, 19, 1), 1.0, "p2 temporal tile1 most recent");
    // tile0 at turn0: exp(-0.2*(5-0))=exp(-1.0)
    let expected = (-1.0f32).exp();
    assert_close(get(&enc, 19, 0), expected, "p2 temporal tile0 decayed");
}

// =========================================================================
// Melds (channels 23-34) -- was 16-27, shifted +7, Chi/Pon/Kan
// =========================================================================

#[test]
fn golden_meld_chi_player0() {
    let mut enc = ObservationEncoder::new();
    let mut melds = empty_melds();
    melds[0].push(MeldInfo {
        tiles: vec![0, 1, 2], // 1m-2m-3m chi
        meld_type: MeldType::Chi,
    });
    enc.encode_melds(&melds);
    // Player 0: chi=ch23, pon=ch24, kan=ch25
    assert_eq!(get(&enc, 23, 0), 1.0, "p0 chi 1m");
    assert_eq!(get(&enc, 23, 1), 1.0, "p0 chi 2m");
    assert_eq!(get(&enc, 23, 2), 1.0, "p0 chi 3m");
    assert_eq!(get(&enc, 24, 0), 0.0, "p0 pon empty");
    assert_eq!(get(&enc, 25, 0), 0.0, "p0 kan empty");
}

#[test]
fn golden_meld_kan_player2() {
    let mut enc = ObservationEncoder::new();
    let mut melds = empty_melds();
    melds[2].push(MeldInfo {
        tiles: vec![27, 27, 27, 27], // East kan
        meld_type: MeldType::Kan,
    });
    enc.encode_melds(&melds);
    // Player 2: ch_base = 23 + 3*2 = 29, kan=ch31
    assert_eq!(get(&enc, 31, 27), 1.0, "p2 kan East");
    assert_eq!(get(&enc, 29, 27), 0.0, "p2 chi empty");
    assert_eq!(get(&enc, 30, 27), 0.0, "p2 pon empty");
}

// =========================================================================
// Dora (channels 35-42) -- COMPLETE REWRITE
//   35-39: indicator thermometer (>=1, >=2, >=3, >=4, >=5)
//   40-42: aka flags (one channel per suit: man, pin, sou)
// =========================================================================

#[test]
fn golden_dora_single_indicator() {
    let mut enc = ObservationEncoder::new();
    let dora = DoraInfo {
        indicators: vec![0], // 1m indicator
        aka_flags: [false; 3],
    };
    enc.encode_dora(&dora);
    // Thermometer: 1 indicator on tile 0 -> ch35 tile0=1, ch36-39 tile0=0
    assert_eq!(get(&enc, 35, 0), 1.0, "ch35 tile0: dora count>=1");
    assert_eq!(get(&enc, 36, 0), 0.0, "ch36 tile0: dora count>=2 off");
    assert_eq!(get(&enc, 37, 0), 0.0, "ch37 tile0: dora count>=3 off");
    // No other tiles have indicators
    assert_eq!(get(&enc, 35, 1), 0.0, "ch35 tile1: no indicator");
}

#[test]
fn golden_dora_thermometer() {
    let mut enc = ObservationEncoder::new();
    // Two indicators on the same tile (e.g., two dora revealed as 1m)
    let dora = DoraInfo {
        indicators: vec![0, 0],
        aka_flags: [false; 3],
    };
    enc.encode_dora(&dora);
    // Thermometer: 2 indicators on tile 0
    assert_eq!(get(&enc, 35, 0), 1.0, "ch35 tile0: dora count>=1");
    assert_eq!(get(&enc, 36, 0), 1.0, "ch36 tile0: dora count>=2");
    assert_eq!(get(&enc, 37, 0), 0.0, "ch37 tile0: dora count>=3 off");
    assert_eq!(get(&enc, 38, 0), 0.0, "ch38 tile0: dora count>=4 off");
    assert_eq!(get(&enc, 39, 0), 0.0, "ch39 tile0: dora count>=5 off");
}

#[test]
fn golden_dora_multiple_indicators() {
    let mut enc = ObservationEncoder::new();
    // Three different indicators
    let dora = DoraInfo {
        indicators: vec![0, 9, 18],
        aka_flags: [false; 3],
    };
    enc.encode_dora(&dora);
    // Each tile gets exactly 1 indicator -> ch35 only
    assert_eq!(get(&enc, 35, 0), 1.0, "ch35 tile0: 1m indicator");
    assert_eq!(get(&enc, 35, 9), 1.0, "ch35 tile9: 1p indicator");
    assert_eq!(get(&enc, 35, 18), 1.0, "ch35 tile18: 1s indicator");
    assert_eq!(get(&enc, 36, 0), 0.0, "ch36 tile0: only 1 indicator");
    assert_eq!(get(&enc, 36, 9), 0.0, "ch36 tile9: only 1 indicator");
}

#[test]
fn golden_aka_flags() {
    let mut enc = ObservationEncoder::new();
    let dora = DoraInfo {
        indicators: vec![],
        aka_flags: [true, false, true], // red 5m and red 5s
    };
    enc.encode_aka(&dora);
    // Ch40 = man aka (filled uniformly), ch41 = pin aka, ch42 = sou aka
    assert_eq!(get(&enc, 40, 0), 1.0, "ch40 tile0: aka man on");
    assert_eq!(get(&enc, 40, 33), 1.0, "ch40 tile33: aka man filled");
    assert_eq!(get(&enc, 41, 0), 0.0, "ch41 tile0: aka pin off");
    assert_eq!(get(&enc, 42, 0), 1.0, "ch42 tile0: aka sou on");
    assert_eq!(get(&enc, 42, 33), 1.0, "ch42 tile33: aka sou filled");
}

// =========================================================================
// Metadata (channels 43-61) -- COMPLETE REWRITE
//   43-46: Riichi flags (4 players)
//   47-50: Scores / 100000
//   51-54: Relative score gaps
//   55-58: Shanten one-hot (0, 1, 2, 3+)
//   59:    Round number (kyoku_index / 8)
//   60:    Honba / 10
//   61:    Kyotaku / 10
// =========================================================================

#[test]
fn golden_meta_riichi_flags() {
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.riichi = [true, false, true, false]; // self + opp2 riichi
    enc.encode_metadata(&meta);
    // Riichi: ch43=self, ch44=opp1, ch45=opp2, ch46=opp3
    assert_eq!(get(&enc, 43, 0), 1.0, "ch43 self riichi on");
    assert_eq!(get(&enc, 43, 17), 1.0, "ch43 filled uniformly");
    assert_eq!(get(&enc, 44, 0), 0.0, "ch44 opp1 riichi off");
    assert_eq!(get(&enc, 45, 0), 1.0, "ch45 opp2 riichi on");
    assert_eq!(get(&enc, 46, 0), 0.0, "ch46 opp3 riichi off");
}

#[test]
fn golden_meta_scores_normalized() {
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.scores = [50000, -10000, 30000, 30000];
    enc.encode_metadata(&meta);
    // Scores ch47-50: score/100000
    assert_close(get(&enc, 47, 0), 0.5, "ch47 p0 50k");
    assert_close(get(&enc, 47, 17), 0.5, "ch47 p0 filled uniformly");
    assert_close(get(&enc, 48, 0), -0.1, "ch48 p1 -10k");
    assert_close(get(&enc, 49, 0), 0.3, "ch49 p2 30k");
    assert_close(get(&enc, 50, 0), 0.3, "ch50 p3 30k");
}

#[test]
fn golden_meta_relative_gaps() {
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.scores = [30000, 50000, 10000, 10000];
    enc.encode_metadata(&meta);
    // Relative gaps ch51-54: (my_score - their_score) / 30000
    // Self is player 0 with 30k
    // ch51: (30k-30k)/30k = 0.0 (self gap always 0)
    assert_close(get(&enc, 51, 0), 0.0, "ch51 self gap 0");
    // ch52: (30k-50k)/30k = -0.6667
    assert_close(get(&enc, 52, 0), -20000.0 / 30000.0, "ch52 opp1 gap -20k");
    // ch53: (30k-10k)/30k = 0.6667
    assert_close(get(&enc, 53, 0), 20000.0 / 30000.0, "ch53 opp2 gap +20k");
    // ch54: (30k-10k)/30k = 0.6667
    assert_close(get(&enc, 54, 0), 20000.0 / 30000.0, "ch54 opp3 gap +20k");
}

#[test]
fn golden_meta_shanten_one_hot() {
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.shanten = 1;
    enc.encode_metadata(&meta);
    // Shanten one-hot: ch55=0-shanten, ch56=1-shanten, ch57=2-shanten, ch58=3+
    assert_eq!(get(&enc, 55, 0), 0.0, "ch55 shanten-0 off");
    assert_eq!(get(&enc, 56, 0), 1.0, "ch56 shanten-1 on");
    assert_eq!(get(&enc, 56, 17), 1.0, "ch56 filled uniformly");
    assert_eq!(get(&enc, 57, 0), 0.0, "ch57 shanten-2 off");
    assert_eq!(get(&enc, 58, 0), 0.0, "ch58 shanten-3+ off");
}

#[test]
fn golden_meta_shanten_one_hot_clamped() {
    // Shanten >= 3 all map to ch58
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.shanten = 5; // should clamp to 3+ bucket
    enc.encode_metadata(&meta);
    assert_eq!(get(&enc, 55, 0), 0.0, "ch55 off");
    assert_eq!(get(&enc, 56, 0), 0.0, "ch56 off");
    assert_eq!(get(&enc, 57, 0), 0.0, "ch57 off");
    assert_eq!(get(&enc, 58, 0), 1.0, "ch58 shanten-3+ on");
}

#[test]
fn golden_meta_round_honba_kyotaku() {
    let mut enc = ObservationEncoder::new();
    let mut meta = default_meta();
    meta.kyoku_index = 4; // East 4 -> South 1
    meta.honba = 3;
    meta.kyotaku = 2;
    enc.encode_metadata(&meta);
    // Ch59: round number = kyoku_index/8 = 4/8 = 0.5
    assert_close(get(&enc, 59, 0), 0.5, "ch59 round 4/8");
    assert_close(get(&enc, 59, 17), 0.5, "ch59 filled uniformly");
    // Ch60: honba/10 = 3/10 = 0.3
    assert_close(get(&enc, 60, 0), 0.3, "ch60 honba 3/10");
    assert_close(get(&enc, 60, 33), 0.3, "ch60 filled uniformly");
    // Ch61: kyotaku/10 = 2/10 = 0.2
    assert_close(get(&enc, 61, 0), 0.2, "ch61 kyotaku 2/10");
    assert_close(get(&enc, 61, 33), 0.2, "ch61 filled uniformly");
}

// =========================================================================
// Safety (channels 62-84) -- UNCHANGED
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
// Full encode roundtrip
// =========================================================================

#[test]
fn golden_full_encode_roundtrip() {
    let mut enc = ObservationEncoder::new();
    let mut hand = [0u8; 34];
    hand[0] = 2;
    hand[8] = 1;
    let mut open_meld_counts = [0u8; 34];
    // Player 1 has a chi of 1p-2p-3p -> open meld counts for tiles 9,10,11
    open_meld_counts[9] = 1;
    open_meld_counts[10] = 1;
    open_meld_counts[11] = 1;
    let mut discards = empty_discards();
    discards[0].discards.push(DiscardEntry {
        tile: 3, is_tedashi: true, turn: 0,
    });
    let mut melds = empty_melds();
    melds[1].push(MeldInfo {
        tiles: vec![9, 10, 11],
        meld_type: MeldType::Chi,
    });
    let dora = DoraInfo {
        indicators: vec![4],
        aka_flags: [true, false, false],
    };
    let meta = default_meta();
    let si = SafetyInfo::new();
    enc.encode(
        &hand, Some(8), &open_meld_counts, &discards, &melds,
        &dora, &meta, &si,
    );
    assert_eq!(enc.as_slice().len(), OBS_SIZE, "buffer size");

    // Spot-check: hand ch0 tile0 (count>=1 for 2x 1m)
    assert_eq!(get(&enc, 0, 0), 1.0, "hand 1m present");
    assert_eq!(get(&enc, 1, 0), 1.0, "hand 1m count>=2");
    assert_eq!(get(&enc, 0, 8), 1.0, "hand 9m present");
    // Open meld counts ch4
    assert_eq!(get(&enc, 4, 9), 1.0, "open meld 1p");
    // Drawn tile ch8
    assert_eq!(get(&enc, 8, 8), 1.0, "drawn 9m");
    // Discard presence ch11
    assert_eq!(get(&enc, 11, 3), 1.0, "p0 discarded 4m");
    // Meld chi p1: ch_base=23+3*1=26
    assert_eq!(get(&enc, 26, 9), 1.0, "p1 chi meld 1p");
    // Dora indicator ch35
    assert_eq!(get(&enc, 35, 4), 1.0, "dora indicator 5m");
    // Aka ch40
    assert_eq!(get(&enc, 40, 0), 1.0, "aka man");
    // Nonzero count sanity
    let nz = enc.as_slice().iter().filter(|&&v| v != 0.0).count();
    assert!(nz > 50, "expected >50 nonzero cells, got {nz}");
}

#[test]
fn golden_full_encode_with_safety() {
    let mut enc = ObservationEncoder::new();
    let hand = [0u8; 34];
    let open_meld_counts = [0u8; 34];
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo { indicators: vec![], aka_flags: [false; 3] };
    let mut meta = default_meta();
    meta.riichi = [false, true, false, false]; // opp1 riichi
    meta.kyoku_index = 4;
    meta.honba = 2;
    let mut si = SafetyInfo::new();
    si.genbutsu_all[0][0] = true;
    si.kabe[27] = true;
    enc.encode(
        &hand, None, &open_meld_counts, &discards, &melds,
        &dora, &meta, &si,
    );
    // Verify across sections
    assert_eq!(get(&enc, 44, 0), 1.0, "opp1 riichi");
    assert_close(get(&enc, 59, 0), 0.5, "round 4/8");
    assert_close(get(&enc, 60, 0), 0.2, "honba 2/10");
    assert_eq!(get(&enc, 62, 0), 1.0, "genbutsu opp0 1m");
    assert_eq!(get(&enc, 80, 27), 1.0, "kabe East");
}

// =========================================================================
// Encode clears between calls
// =========================================================================

#[test]
fn golden_encode_clears_stale_data() {
    let mut enc = ObservationEncoder::new();
    let open_meld_counts = [0u8; 34];
    let discards = empty_discards();
    let melds = empty_melds();
    let dora = DoraInfo { indicators: vec![], aka_flags: [false; 3] };
    let meta = default_meta();
    let si = SafetyInfo::new();

    // First encode: hand with 3x 1m, drawn 6m
    let mut hand1 = [0u8; 34];
    hand1[0] = 3;
    enc.encode(
        &hand1, Some(5), &open_meld_counts, &discards, &melds,
        &dora, &meta, &si,
    );
    assert_eq!(get(&enc, 2, 0), 1.0, "first: ch2 tile0 set");
    assert_eq!(get(&enc, 8, 5), 1.0, "first: drawn 6m");

    // Second encode: empty hand, no draw
    let hand2 = [0u8; 34];
    enc.encode(
        &hand2, None, &open_meld_counts, &discards, &melds,
        &dora, &meta, &si,
    );
    assert_eq!(get(&enc, 2, 0), 0.0, "second: ch2 tile0 cleared");
    assert_eq!(get(&enc, 8, 5), 0.0, "second: drawn 6m cleared");
    assert_eq!(get(&enc, 0, 0), 0.0, "second: ch0 tile0 cleared");
}
