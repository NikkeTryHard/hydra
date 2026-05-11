use super::*;

#[test]
fn tile_and_aka_abi_constants_are_frozen() {
    assert_eq!(NUM_TILE_TYPES, 34);
    assert_eq!(NUM_TILES_136, 136);
    assert_eq!(AKA_MANZU_136, 16);
    assert_eq!(AKA_PINZU_136, 52);
    assert_eq!(AKA_SOUZU_136, 88);
    assert_eq!(AKA_MANZU_TYPE, 34);
    assert_eq!(AKA_PINZU_TYPE, 35);
    assert_eq!(AKA_SOUZU_TYPE, 36);
}

#[test]
fn tile_type_new_valid() {
    for i in 0..34u8 {
        assert!(
            TileType::new(i).is_some(),
            "TileType::new({i}) should be Some"
        );
    }
    assert!(TileType::new(34).is_none());
    assert!(TileType::new(255).is_none());
}

#[test]
fn suit_classification() {
    // Manzu 0-8
    for i in 0..9u8 {
        let t = TileType::new(i).unwrap();
        assert_eq!(t.suit(), Suit::Manzu, "tile {i} should be Manzu");
        assert!(t.is_suited());
        assert!(!t.is_honor());
    }
    // Pinzu 9-17
    for i in 9..18u8 {
        let t = TileType::new(i).unwrap();
        assert_eq!(t.suit(), Suit::Pinzu, "tile {i} should be Pinzu");
    }
    // Souzu 18-26
    for i in 18..27u8 {
        let t = TileType::new(i).unwrap();
        assert_eq!(t.suit(), Suit::Souzu, "tile {i} should be Souzu");
    }
    // Jihai 27-33
    for i in 27..34u8 {
        let t = TileType::new(i).unwrap();
        assert_eq!(t.suit(), Suit::Jihai, "tile {i} should be Jihai");
        assert!(t.is_honor());
        assert!(!t.is_suited());
    }
}

#[test]
fn tile_number() {
    // Suited tiles have 1-based numbers
    assert_eq!(TileType::new(0).unwrap().number(), Some(1)); // 1m
    assert_eq!(TileType::new(8).unwrap().number(), Some(9)); // 9m
    assert_eq!(TileType::new(9).unwrap().number(), Some(1)); // 1p
    assert_eq!(TileType::new(22).unwrap().number(), Some(5)); // 5s
    // Honors have no number
    assert_eq!(TileType::new(27).unwrap().number(), None);
    assert_eq!(TileType::new(33).unwrap().number(), None);
}

#[test]
fn terminal_detection() {
    let terminals = [0, 8, 9, 17, 18, 26]; // 1m,9m,1p,9p,1s,9s
    for &i in &terminals {
        let t = TileType::new(i).unwrap();
        assert!(t.is_terminal(), "tile {i} should be terminal");
        assert!(t.is_terminal_or_honor());
    }
    // Middle tiles are not terminal
    let middles = [1, 4, 10, 14, 19, 23];
    for &i in &middles {
        let t = TileType::new(i).unwrap();
        assert!(!t.is_terminal(), "tile {i} should NOT be terminal");
    }
    // Honors are not terminal but are terminal_or_honor
    for i in 27..34u8 {
        let t = TileType::new(i).unwrap();
        assert!(!t.is_terminal());
        assert!(t.is_terminal_or_honor());
    }
}

#[test]
fn tile136_to_type_correct() {
    // Each group of 4 consecutive 136-tiles maps to one type
    for t in 0..34u8 {
        for copy in 0..4u8 {
            let t136 = t * 4 + copy;
            assert_eq!(tile136_to_type(t136).id(), t);
        }
    }
}

#[test]
fn aka_detection_136() {
    assert!(tile136_is_aka(16)); // red 5m
    assert!(tile136_is_aka(52)); // red 5p
    assert!(tile136_is_aka(88)); // red 5s
    // Non-aka copies of the same tile types
    assert!(!tile136_is_aka(17)); // normal 5m
    assert!(!tile136_is_aka(18)); // normal 5m
    assert!(!tile136_is_aka(53)); // normal 5p
    assert!(!tile136_is_aka(0)); // 1m
}

#[test]
fn deaka_strips_aka() {
    assert_eq!(deaka(34), 4); // aka 5m -> 5m
    assert_eq!(deaka(35), 13); // aka 5p -> 5p
    assert_eq!(deaka(36), 22); // aka 5s -> 5s
    // Non-aka pass through
    assert_eq!(deaka(0), 0);
    assert_eq!(deaka(4), 4);
    assert_eq!(deaka(33), 33);
}

#[test]
fn re_akaize_roundtrip() {
    // Aka types roundtrip through deaka -> re_akaize
    for aka in [AKA_MANZU_TYPE, AKA_PINZU_TYPE, AKA_SOUZU_TYPE] {
        let base = deaka(aka);
        assert_eq!(re_akaize(base, true), aka);
    }
    // Non-aka tiles are unaffected
    assert_eq!(re_akaize(4, false), 4);
    assert_eq!(re_akaize(0, false), 0);
}

#[test]
fn permutation_identity() {
    let identity = &ALL_PERMUTATIONS[0];
    for i in 0..34u8 {
        assert_eq!(permute_tile_type(i, identity), i);
    }
}

#[test]
fn all_permutations_produce_valid_types() {
    for perm in &ALL_PERMUTATIONS {
        for i in 0..34u8 {
            let result = permute_tile_type(i, perm);
            assert!(
                result < NUM_TILE_TYPES as u8,
                "permute_tile_type({i}, {perm:?}) = {result} out of range"
            );
        }
    }
}

#[test]
fn permutation_honors_unchanged() {
    for perm in &ALL_PERMUTATIONS {
        for i in 27..34u8 {
            assert_eq!(
                permute_tile_type(i, perm),
                i,
                "honor tile {i} should not be affected by permutation {perm:?}"
            );
        }
    }
}

#[test]
fn permutation_swap_man_pin() {
    let perm = &ALL_PERMUTATIONS[2]; // [1, 0, 2] = swap man-pin
    assert_eq!(permute_tile_type(0, perm), 9); // 1m -> 1p
    assert_eq!(permute_tile_type(9, perm), 0); // 1p -> 1m
    assert_eq!(permute_tile_type(18, perm), 18); // 1s -> 1s
    assert_eq!(permute_tile_type(27, perm), 27); // E -> E
}

#[test]
fn permute_tile136_preserves_aka() {
    let perm = &ALL_PERMUTATIONS[2]; // swap man-pin
    // Red 5m (136-idx 16) -> should become red 5p (136-idx 52)
    let result = permute_tile136(AKA_MANZU_136, perm);
    assert_eq!(result, AKA_PINZU_136);
    assert!(tile136_is_aka(result), "aka status should be preserved");

    // Red 5p -> red 5m
    let result2 = permute_tile136(AKA_PINZU_136, perm);
    assert_eq!(result2, AKA_MANZU_136);
    assert!(tile136_is_aka(result2));
}

#[test]
fn permute_extended_aka() {
    let perm = &ALL_PERMUTATIONS[2]; // swap man-pin
    assert_eq!(permute_tile_extended(AKA_MANZU_TYPE, perm), AKA_PINZU_TYPE);
    assert_eq!(permute_tile_extended(AKA_PINZU_TYPE, perm), AKA_MANZU_TYPE);
    assert_eq!(permute_tile_extended(AKA_SOUZU_TYPE, perm), AKA_SOUZU_TYPE);
    // Non-aka pass through
    assert_eq!(permute_tile_extended(0, perm), 9);
}

#[test]
fn all_permutations_are_bijections() {
    // Each permutation should be a bijection on the 34 tile types
    for perm in &ALL_PERMUTATIONS {
        let mut seen = [false; NUM_TILE_TYPES];
        for i in 0..34u8 {
            let out = permute_tile_type(i, perm) as usize;
            assert!(!seen[out], "duplicate output {out} for perm {perm:?}");
            seen[out] = true;
        }
        assert!(seen.iter().all(|&s| s), "perm {perm:?} is not surjective");
    }
}

#[test]
fn tile_type_display() {
    assert_eq!(format!("{}", TileType::new(0).unwrap()), "1m");
    assert_eq!(format!("{}", TileType::new(8).unwrap()), "9m");
    assert_eq!(format!("{}", TileType::new(27).unwrap()), "E");
    assert_eq!(format!("{}", TileType::new(33).unwrap()), "C");
}

#[test]
fn mjai_names() {
    assert_eq!(tile_type_to_mjai(0), "1m");
    assert_eq!(tile_type_to_mjai(9), "1p");
    assert_eq!(tile_type_to_mjai(18), "1s");
    assert_eq!(tile_type_to_mjai(27), "E");
    assert_eq!(tile_type_to_mjai(31), "P");
    assert_eq!(tile_type_to_mjai(32), "F");
    assert_eq!(tile_type_to_mjai(33), "C");
    assert_eq!(tile_type_to_mjai(99), "??");
}
