/// Converts final scores into the global rank permutation class.
pub fn scores_to_grp_index(scores: [i32; 4]) -> Result<u8, &'static str> {
    let mut indexed: [(i32, u8); 4] = [
        (scores[0], 0),
        (scores[1], 1),
        (scores[2], 2),
        (scores[3], 3),
    ];
    indexed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let ranking = [indexed[0].1, indexed[1].1, indexed[2].1, indexed[3].1];
    GRP_PERM_TABLE
        .iter()
        .position(|p| *p == ranking)
        .map(|i| i as u8)
        .ok_or("invalid ranking permutation")
}

/// Table of all four-player rank permutations.
pub const GRP_PERM_TABLE: [[u8; 4]; 24] = generate_perm_table();

const fn generate_perm_table() -> [[u8; 4]; 24] {
    let mut table = [[0u8; 4]; 24];
    let mut idx = 0;
    let mut a = 0u8;
    while a < 4 {
        let mut b = 0u8;
        while b < 4 {
            if b != a {
                let mut c = 0u8;
                while c < 4 {
                    if c != a && c != b {
                        let d = 6 - a - b - c;
                        table[idx] = [a, b, c, d];
                        idx += 1;
                    }
                    c += 1;
                }
            }
            b += 1;
        }
        a += 1;
    }
    table
}

/// Returns one player's final placement, where `0` is first place.
///
/// Returns `None` when `player` is outside the four-player score array.
pub fn score_to_placement(scores: [i32; 4], player: u8) -> Option<u8> {
    score_to_placements(scores).get(player as usize).copied()
}

/// Returns final placements for all players, where `0` is first place.
pub fn score_to_placements(scores: [i32; 4]) -> [u8; 4] {
    let mut indexed: [(i32, u8); 4] = [
        (scores[0], 0),
        (scores[1], 1),
        (scores[2], 2),
        (scores[3], 3),
    ];
    indexed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut placements = [3u8; 4];
    for (placement, (_, player)) in indexed.into_iter().enumerate() {
        placements[player as usize] = placement as u8;
    }
    placements
}
